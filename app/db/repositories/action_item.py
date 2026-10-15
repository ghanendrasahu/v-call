"""Action item repository — idempotent extraction results + dispatch tracking."""

from __future__ import annotations

import uuid
from datetime import date

from sqlalchemy import delete, select

from app.db.enums import (
    ActionItemPriority,
    ActionItemStatus,
    TargetSystem,
)
from app.db.models import ActionItem
from app.db.repositories.base import BaseRepository


class ActionItemRepository(BaseRepository[ActionItem]):
    """Async data access for extracted action items."""

    model = ActionItem

    async def replace_for_meeting(
        self,
        *,
        meeting_id: uuid.UUID,
        tenant_id: uuid.UUID,
        items: list[dict],
    ) -> list[ActionItem]:
        """Idempotently replace all items for a meeting (agent re-runs safe)."""
        await self.session.execute(
            delete(ActionItem).where(
                ActionItem.meeting_id == meeting_id,
                ActionItem.tenant_id == tenant_id,
            )
        )
        rows = [
            ActionItem(
                meeting_id=meeting_id,
                tenant_id=tenant_id,
                assignee=item.get("assignee"),
                task=item["task"],
                priority=ActionItemPriority(item.get("priority", ActionItemPriority.MEDIUM.value)),
                due_date=_parse_due_date(item.get("due_date")),
                target_system=TargetSystem(item.get("target_system", TargetSystem.NONE.value)),
            )
            for item in items
        ]
        self.session.add_all(rows)
        await self.session.flush()
        return rows

    async def mark_dispatched(
        self,
        item_id: uuid.UUID,
        *,
        tenant_id: uuid.UUID,
        external_id: str,
        system: TargetSystem | None = None,
    ) -> ActionItem | None:
        """Record the created issue/page id in the target system."""
        item = await self.get(item_id, tenant_id=tenant_id)
        if item is None:
            return None
        item.status = ActionItemStatus.DISPATCHED
        item.external_id = external_id
        if system is not None:
            item.target_system = system
        await self.session.flush()
        return item

    async def mark_status(
        self,
        item_id: uuid.UUID,
        *,
        tenant_id: uuid.UUID,
        status: ActionItemStatus,
    ) -> ActionItem | None:
        """Generic status transition (failed dispatch, completion, skip)."""
        item = await self.get(item_id, tenant_id=tenant_id)
        if item is None:
            return None
        item.status = status
        await self.session.flush()
        return item

    async def list_open(
        self,
        *,
        tenant_id: uuid.UUID,
        limit: int = 100,
    ) -> list[ActionItem]:
        """Open items for a tenant, highest priority first."""
        stmt = (
            select(ActionItem)
            .where(
                ActionItem.tenant_id == tenant_id,
                ActionItem.status == ActionItemStatus.OPEN,
            )
            .order_by(ActionItem.priority.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def list_for_meeting(
        self,
        meeting_id: uuid.UUID,
        *,
        tenant_id: uuid.UUID,
    ) -> list[ActionItem]:
        """All action items belonging to one meeting."""
        stmt = (
            select(ActionItem)
            .where(
                ActionItem.meeting_id == meeting_id,
                ActionItem.tenant_id == tenant_id,
            )
            .order_by(ActionItem.created_at)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())


def _parse_due_date(value: object) -> date | None:
    """Accept ``date``, ISO string, or ``None`` from agent JSON output."""
    if value is None or value == "":
        return None
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))
