"""Notion connector (REST API 2022-06-28, Bearer token auth)."""

from __future__ import annotations

from typing import Any

from app.integrations.base import BaseConnector

NOTION_VERSION = "2022-06-28"
BASE_URL = "https://api.notion.com/v1"


class NotionConnector(BaseConnector):
    """Create pages (action-item trackers) and query databases in Notion."""

    name = "notion"

    def __init__(self, *, integration_token: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._headers = {
            "Authorization": f"Bearer {integration_token}",
            "Notion-Version": NOTION_VERSION,
            "Content-Type": "application/json",
        }

    async def create_page(
        self,
        *,
        parent_database_id: str,
        properties: dict[str, Any],
        children: list[dict] | None = None,
    ) -> dict:
        """Create a page in a database with the given properties."""
        payload: dict[str, Any] = {
            "parent": {"database_id": parent_database_id},
            "properties": properties,
        }
        if children:
            payload["children"] = children
        return await self._request("POST", f"{BASE_URL}/pages", json=payload, headers=self._headers)

    async def append_blocks(self, *, page_id: str, blocks: list[dict]) -> dict:
        """Append block children to an existing page."""
        return await self._request(
            "PATCH",
            f"{BASE_URL}/blocks/{page_id}/children",
            json={"children": blocks},
            headers=self._headers,
        )

    async def query_database(
        self,
        *,
        database_id: str,
        filter: dict[str, Any] | None = None,
        page_size: int = 100,
    ) -> dict:
        """Query a database with an optional filter."""
        payload: dict[str, Any] = {"page_size": page_size}
        if filter:
            payload["filter"] = filter
        return await self._request(
            "POST",
            f"{BASE_URL}/databases/{database_id}/query",
            json=payload,
            headers=self._headers,
        )

    async def search(self, *, query: str, filter_type: str = "page") -> dict:
        """Search by title across the integration's shared space."""
        return await self._request(
            "POST",
            f"{BASE_URL}/search",
            json={"query": query, "filter": {"value": filter_type, "property": "object"}},
            headers=self._headers,
        )

    @staticmethod
    def task_page_properties(
        *,
        title: str,
        assignee: str | None = None,
        status: str = "Not started",
        due_date: str | None = None,
    ) -> dict[str, Any]:
        """Build a standard property map for a task-tracking database."""
        properties: dict[str, Any] = {
            "Name": {"title": [{"text": {"content": title}}]},
            "Status": {"status": {"name": status}},
        }
        if assignee:
            properties["Assignee"] = {"rich_text": [{"text": {"content": assignee}}]}
        if due_date:
            properties["Due"] = {"date": {"start": due_date}}
        return properties
