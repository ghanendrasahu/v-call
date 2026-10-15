"""Zoho CRM connector (Data API v3, ``Zoho-oauthtoken`` auth)."""

from __future__ import annotations

from typing import Any

from app.integrations.base import BaseConnector

BASE_URL = "https://www.zohoapis.com/crm/v3"


class ZohoCRMConnector(BaseConnector):
    """Create leads/contacts and run searches in Zoho CRM."""

    name = "zoho_crm"

    def __init__(
        self,
        *,
        access_token: str,
        api_domain: str = "https://www.zohoapis.com",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._base_url = f"{api_domain.rstrip('/')}/crm/v3"
        self._headers = {"Authorization": f"Zoho-oauthtoken {access_token}"}

    async def create_records(self, module: str, records: list[dict[str, Any]]) -> dict:
        """Bulk-create records in a module (Leads, Contacts, Deals, …)."""
        return await self._request(
            "POST",
            f"{self._base_url}/{module.lower()}",
            json={"data": records},
            headers=self._headers,
        )

    async def create_lead(
        self,
        *,
        first_name: str,
        last_name: str,
        email: str,
        company: str | None = None,
        phone: str | None = None,
        description: str | None = None,
    ) -> dict:
        """Create a single lead (mirrors CRM Lead layout)."""
        lead: dict[str, Any] = {
            "First_Name": first_name,
            "Last_Name": last_name,
            "Email": email,
        }
        if company:
            lead["Company"] = company
        if phone:
            lead["Phone"] = phone
        if description:
            lead["Description"] = description
        return await self.create_records("Leads", [lead])

    async def create_contact(
        self,
        *,
        first_name: str,
        last_name: str,
        email: str,
        phone: str | None = None,
    ) -> dict:
        """Create a single contact."""
        contact: dict[str, Any] = {
            "First_Name": first_name,
            "Last_Name": last_name,
            "Email": email,
        }
        if phone:
            contact["Phone"] = phone
        return await self.create_records("Contacts", [contact])

    async def search_records(
        self,
        module: str,
        *,
        email: str | None = None,
        criteria: str | None = None,
    ) -> dict:
        """Search a module by email or raw criteria expression."""
        params: dict[str, Any] = {}
        if email:
            params["email"] = email
        if criteria:
            params["criteria"] = criteria
        return await self._request(
            "GET",
            f"{self._base_url}/{module.lower()}/search",
            params=params,
            headers=self._headers,
        )
