"""Dispatcher agent: routes action items into SaaS systems.

The node is deliberately decoupled from concrete connectors via the
:class:`DispatchPort` protocol. Integration modules (Jira, Notion, Zoho,
Freshdesk) register themselves in a process-wide registry; the node looks up
the target system and dispatches with per-item failure isolation.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

from app.agents.state import MeetingState

logger = logging.getLogger(__name__)


class DispatchError(Exception):
    """Raised by connectors when a target-system write fails."""


@runtime_checkable
class DispatchPort(Protocol):
    """Anything that can persist one action item into a target system."""

    async def dispatch_action_item(
        self,
        item: dict,
        *,
        tenant_id: str,
        meeting_id: str,
    ) -> str:
        """Create the item remotely; return its external id/key."""
        ...


class DispatcherRegistry:
    """Process-wide map of ``TargetSystem.value → DispatchPort``."""

    def __init__(self) -> None:
        self._ports: dict[str, DispatchPort] = {}

    def register(self, system: str, port: DispatchPort) -> None:
        self._ports[system] = port

    def get(self, system: str) -> DispatchPort | None:
        return self._ports.get(system)

    def systems(self) -> list[str]:
        return sorted(self._ports)

    def clear(self) -> None:
        self._ports.clear()


# Shared registry (integrations register at import/startup time).
registry = DispatcherRegistry()


def make_dispatcher_node(reg: DispatcherRegistry | None = None) -> Any:
    """Build the dispatcher node over a registry (defaults to global)."""

    active = reg if reg is not None else registry

    async def dispatcher_node(state: MeetingState) -> dict:
        items = state.get("action_items", []) or []
        tenant_id = state.get("tenant_id", "")
        meeting_id = state.get("meeting_id", "")

        updated_items: list[dict] = []
        status: dict[str, str] = {}
        errors: list[str] = []

        for item in items:
            target = item.get("target_system", "none")
            if target == "none":
                item = {**item, "status": "open", "external_id": None}
                updated_items.append(item)
                continue

            port = active.get(target)
            if port is None:
                # Not configured for this tenant → keep as open, don't fail.
                item = {**item, "status": "skipped", "external_id": None}
                updated_items.append(item)
                status[target] = "skipped (not connected)"
                continue

            try:
                external_id = await port.dispatch_action_item(
                    item, tenant_id=tenant_id, meeting_id=meeting_id
                )
            except Exception as exc:
                logger.exception("Dispatch to %s failed for meeting %s", target, meeting_id)
                item = {**item, "status": "failed"}
                errors.append(f"dispatch[{target}]: {exc}")
                status[target] = f"failed: {exc}"
            else:
                item = {**item, "status": "dispatched", "external_id": external_id}
                status[target] = f"created: {external_id}"
            updated_items.append(item)

        # Preserve non-action items untouched + replace the list wholesale.
        return {
            "action_items": updated_items,
            "integration_status": status,
            "errors": errors,
        }

    return dispatcher_node
