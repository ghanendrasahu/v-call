"""Jira Cloud connector (REST API v3, Basic auth with API token)."""

from __future__ import annotations

from typing import Any

from app.integrations.base import BaseConnector

API_VERSION = "3"


class JiraConnector(BaseConnector):
    """Create and manage Jira issues for a tenant's Atlassian site."""

    name = "jira"

    def __init__(
        self,
        *,
        site: str,
        email: str,
        api_token: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._base_url = f"https://{site}.atlassian.net/rest/api/{API_VERSION}"
        # Jira Cloud Basic auth: account email + API token.
        self._auth = (email, api_token)

    # ── Issues ─────────────────────────────────────────────────────────
    async def create_issue(
        self,
        *,
        project_key: str,
        summary: str,
        description: str | None = None,
        issue_type: str = "Task",
        assignee_account_id: str | None = None,
        labels: list[str] | None = None,
        due_date: str | None = None,
    ) -> dict:
        """Create one issue; returns the created issue payload (has ``key``)."""
        fields: dict[str, Any] = {
            "project": {"key": project_key},
            "summary": summary,
            "issuetype": {"name": issue_type},
        }
        if description:
            # Jira v3 expects Atlassian Document Format for description.
            fields["description"] = {
                "type": "doc",
                "version": 1,
                "content": [
                    {
                        "type": "paragraph",
                        "content": [{"type": "text", "text": description}],
                    }
                ],
            }
        if assignee_account_id:
            fields["assignee"] = {"id": assignee_account_id}
        if labels:
            fields["labels"] = labels
        if due_date:
            fields["duedate"] = due_date

        return await self._request(
            "POST",
            f"{self._base_url}/issue",
            json={"fields": fields},
            auth=self._auth,
        )

    async def get_issue(self, key: str) -> dict:
        """Fetch a single issue by key (e.g. ``PROJ-123``)."""
        return await self._request("GET", f"{self._base_url}/issue/{key}", auth=self._auth)

    async def add_comment(self, key: str, body: str) -> dict:
        """Add a comment to an issue."""
        return await self._request(
            "POST",
            f"{self._base_url}/issue/{key}/comment",
            json={"body": body},
            auth=self._auth,
        )

    async def search_issues(self, jql: str, *, max_results: int = 50) -> dict:
        """Run a JQL search."""
        return await self._request(
            "GET",
            f"{self._base_url}/search",
            params={"jql": jql, "maxResults": max_results},
            auth=self._auth,
        )
