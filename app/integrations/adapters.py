"""Adapters: connect REST connectors to the agent dispatcher.

Each adapter implements the ``DispatchPort`` protocol expected by the
dispatcher node (``dispatch_action_item(item, tenant_id, meeting_id) -> id``)
and translates a generic action-item dict into provider-specific calls.

Credential resolution is injected via :class:`CredentialProvider` so this
module stays decoupled from the database layer (workers wire the real
DB-backed provider at startup).
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


class DispatchError(Exception):
    """Raised when an adapter cannot dispatch an action item."""


@dataclass(frozen=True, slots=True)
class ResolvedCredentials:
    """Decrypted, provider-ready credentials for one tenant."""

    values: dict[str, str]


#: ``(tenant_id, provider) -> credentials`` (async).
CredentialProvider = Callable[[str, str], Awaitable[ResolvedCredentials | None]]


class BaseDispatchAdapter:
    """Common scaffolding: resolve credentials, normalise failures."""

    provider: str = "base"

    def __init__(self, credential_provider: CredentialProvider) -> None:
        self._credentials = credential_provider

    async def _resolve(self, tenant_id: str) -> ResolvedCredentials:
        creds = await self._credentials(tenant_id, self.provider)
        if creds is None:
            raise DispatchError(f"No {self.provider} credentials configured for tenant {tenant_id}")
        return creds

    async def dispatch_action_item(
        self,
        item: dict[str, Any],
        *,
        tenant_id: str,
        meeting_id: str,
    ) -> str:
        """Dispatch one item; always returns an external id string."""
        raise NotImplementedError


class JiraDispatchAdapter(BaseDispatchAdapter):
    """File action items as Jira issues."""

    provider = "jira"

    def __init__(
        self,
        credential_provider: CredentialProvider,
        *,
        project_key: str,
    ) -> None:
        super().__init__(credential_provider)
        self._project_key = project_key

    async def dispatch_action_item(
        self, item: dict[str, Any], *, tenant_id: str, meeting_id: str
    ) -> str:
        from app.integrations.jira import JiraConnector

        creds = await self._resolve(tenant_id)
        async with JiraConnector(
            site=creds.values["site"],
            email=creds.values["email"],
            api_token=creds.values["api_token"],
        ) as jira:
            created = await jira.create_issue(
                project_key=self._project_key,
                summary=item["task"],
                description=f"From meeting {meeting_id}",
                labels=["vcall", f"meeting-{meeting_id}"],
                due_date=item.get("due_date"),
            )
        key = created.get("key")
        if not key:
            raise DispatchError(f"Jira response missing issue key: {created}")
        return str(key)


class NotionDispatchAdapter(BaseDispatchAdapter):
    """Create task pages in a Notion database."""

    provider = "notion"

    def __init__(
        self,
        credential_provider: CredentialProvider,
        *,
        database_id: str,
    ) -> None:
        super().__init__(credential_provider)
        self._database_id = database_id

    async def dispatch_action_item(
        self, item: dict[str, Any], *, tenant_id: str, meeting_id: str
    ) -> str:
        from app.integrations.notion import NotionConnector

        creds = await self._resolve(tenant_id)
        async with NotionConnector(integration_token=creds.values["token"]) as notion:
            created = await notion.create_page(
                parent_database_id=self._database_id,
                properties=NotionConnector.task_page_properties(
                    title=item["task"],
                    assignee=item.get("assignee"),
                    due_date=item.get("due_date"),
                ),
            )
        page_id = created.get("id")
        if not page_id:
            raise DispatchError(f"Notion response missing page id: {created}")
        return str(page_id)


class ZohoDeskDispatchAdapter(BaseDispatchAdapter):
    """Raise Zoho Desk tickets for action items."""

    provider = "zoho_desk"

    def __init__(
        self,
        credential_provider: CredentialProvider,
        *,
        contact_email: str,
    ) -> None:
        super().__init__(credential_provider)
        self._contact_email = contact_email

    async def dispatch_action_item(
        self, item: dict[str, Any], *, tenant_id: str, meeting_id: str
    ) -> str:
        from app.integrations.zoho_desk import ZohoDeskConnector

        creds = await self._resolve(tenant_id)
        async with ZohoDeskConnector(
            access_token=creds.values["access_token"],
            org_id=creds.values["org_id"],
        ) as desk:
            created = await desk.create_ticket(
                subject=item["task"],
                description=f"Action item from meeting {meeting_id}",
                contact_email=self._contact_email,
                priority=item.get("priority", "medium"),
            )
        ticket_id = created.get("id") or (created.get("data") or {}).get("id")
        if not ticket_id:
            raise DispatchError(f"Zoho Desk response missing ticket id: {created}")
        return str(ticket_id)


class FreshdeskDispatchAdapter(BaseDispatchAdapter):
    """Raise Freshdesk tickets for action items."""

    provider = "freshdesk"

    def __init__(
        self,
        credential_provider: CredentialProvider,
        *,
        requester_email: str,
    ) -> None:
        super().__init__(credential_provider)
        self._requester_email = requester_email

    async def dispatch_action_item(
        self, item: dict[str, Any], *, tenant_id: str, meeting_id: str
    ) -> str:
        from app.integrations.freshdesk import FreshdeskConnector

        creds = await self._resolve(tenant_id)
        async with FreshdeskConnector(
            domain=creds.values["domain"], api_key=creds.values["api_key"]
        ) as freshdesk:
            created = await freshdesk.create_ticket(
                subject=item["task"],
                description=f"Action item from meeting {meeting_id}",
                requester_email=self._requester_email,
                priority=item.get("priority", "medium"),
                tags=["vcall", f"meeting-{meeting_id}"],
            )
        ticket_id = created.get("id")
        if not ticket_id:
            raise DispatchError(f"Freshdesk response missing ticket id: {created}")
        return str(ticket_id)


def register_adapters(
    registry: Any,
    credential_provider: CredentialProvider,
    *,
    jira_project_key: str | None = None,
    notion_database_id: str | None = None,
    zoho_desk_contact_email: str | None = None,
    freshdesk_requester_email: str | None = None,
) -> list[str]:
    """Register whatever adapters are configured; return registered systems."""
    registered: list[str] = []
    if jira_project_key:
        registry.register(
            "jira", JiraDispatchAdapter(credential_provider, project_key=jira_project_key)
        )
        registered.append("jira")
    if notion_database_id:
        registry.register(
            "notion", NotionDispatchAdapter(credential_provider, database_id=notion_database_id)
        )
        registered.append("notion")
    if zoho_desk_contact_email:
        registry.register(
            "zoho_desk",
            ZohoDeskDispatchAdapter(credential_provider, contact_email=zoho_desk_contact_email),
        )
        registered.append("zoho_desk")
    if freshdesk_requester_email:
        registry.register(
            "freshdesk",
            FreshdeskDispatchAdapter(
                credential_provider, requester_email=freshdesk_requester_email
            ),
        )
        registered.append("freshdesk")
    logger.info("Registered dispatch adapters: %s", registered)
    return registered


__all__ = [
    "BaseDispatchAdapter",
    "CredentialProvider",
    "DispatchError",
    "FreshdeskDispatchAdapter",
    "JiraDispatchAdapter",
    "NotionDispatchAdapter",
    "ResolvedCredentials",
    "ZohoDeskDispatchAdapter",
    "register_adapters",
]
