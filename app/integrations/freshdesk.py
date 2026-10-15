"""Freshdesk connector (API v2, Basic auth with API key as username)."""

from __future__ import annotations

from typing import Any

from app.integrations.base import BaseConnector

BASE_URL_TEMPLATE = "https://{domain}.freshdesk.com/api/v2"

# Freshdesk priority: 1=Low, 2=Medium, 3=High, 4=Urgent.
PRIORITY_MAP = {"low": 1, "medium": 2, "high": 3, "urgent": 4}
# Freshdesk status: 2=Open, 3=Pending, 4=Resolved, 5=Closed.
STATUS_MAP = {"open": 2, "pending": 3, "resolved": 4, "closed": 5}


class FreshdeskConnector(BaseConnector):
    """Create and query support tickets in Freshdesk."""

    name = "freshdesk"

    def __init__(
        self,
        *,
        domain: str,
        api_key: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._base_url = BASE_URL_TEMPLATE.format(domain=domain)
        # Freshdesk Basic auth: API key as username, password literal "X".
        self._auth = (api_key, "X")

    async def create_ticket(
        self,
        *,
        subject: str,
        description: str,
        requester_email: str,
        priority: str = "medium",
        status: str = "open",
        tags: list[str] | None = None,
    ) -> dict:
        """Create one ticket; returns payload containing ``id``."""
        ticket: dict[str, Any] = {
            "subject": subject,
            "description": description,
            "email": requester_email,
            "priority": PRIORITY_MAP.get(priority.lower(), 2),
            "status": STATUS_MAP.get(status.lower(), 2),
        }
        if tags:
            ticket["tags"] = tags
        return await self._request(
            "POST",
            f"{self._base_url}/tickets",
            json=ticket,
            auth=self._auth,
        )

    async def get_ticket(self, ticket_id: int | str) -> dict:
        """Fetch one ticket by id."""
        return await self._request("GET", f"{self._base_url}/tickets/{ticket_id}", auth=self._auth)

    async def list_tickets(
        self, *, page: int = 1, per_page: int = 30, include_description: bool = False
    ) -> dict:
        """List tickets with cursor pagination."""
        params: dict[str, Any] = {"page": page, "per_page": per_page}
        if include_description:
            params["include"] = "description"
        return await self._request(
            "GET", f"{self._base_url}/tickets", params=params, auth=self._auth
        )
