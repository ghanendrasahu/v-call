"""Zoho Desk connector (API v1, ``Zoho-oauthtoken`` + ``orgId`` headers)."""

from __future__ import annotations

from typing import Any

from app.integrations.base import BaseConnector

BASE_URL = "https://desk.zoho.com/api/v1"

# Zoho Desk numeric priorities (API contract).
PRIORITY_MAP = {"low": 1, "medium": 2, "high": 3, "urgent": 4}


class ZohoDeskConnector(BaseConnector):
    """Create and query support tickets in Zoho Desk."""

    name = "zoho_desk"

    def __init__(
        self,
        *,
        access_token: str,
        org_id: str,
        api_domain: str = "https://desk.zoho.com",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._base_url = f"{api_domain.rstrip('/')}/api/v1/{org_id}"
        self._headers = {
            "Authorization": f"Zoho-oauthtoken {access_token}",
            "orgId": org_id,
        }

    async def create_ticket(
        self,
        *,
        subject: str,
        description: str,
        contact_email: str,
        priority: str = "medium",
        status: str = "Open",
        department_id: str | None = None,
    ) -> dict:
        """Create one ticket; returns payload containing ``id``."""
        ticket: dict[str, Any] = {
            "subject": subject,
            "description": description,
            "contactEmail": contact_email,
            "priority": PRIORITY_MAP.get(priority.lower(), 2),
            "status": status,
        }
        if department_id:
            ticket["departmentId"] = department_id
        return await self._request(
            "POST",
            f"{self._base_url}/tickets",
            json=ticket,
            headers=self._headers,
        )

    async def get_ticket(self, ticket_id: str) -> dict:
        """Fetch one ticket by id."""
        return await self._request(
            "GET",
            f"{self._base_url}/tickets/{ticket_id}",
            headers=self._headers,
        )

    async def list_tickets(self, *, status: str | None = None, limit: int = 50) -> dict:
        """List tickets, optionally filtered by status."""
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        return await self._request(
            "GET",
            f"{self._base_url}/tickets",
            params=params,
            headers=self._headers,
        )
