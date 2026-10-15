"""Integration tests for REST connectors, dispatch adapters, and MCP clients.

HTTP behaviour is exercised through ``httpx.MockTransport`` — no real
network, deterministic responses, full request assertions.
"""

from __future__ import annotations

import json

import httpx
import pytest

from app.agents.nodes.dispatcher import DispatcherRegistry
from app.integrations.adapters import (
    DispatchError,
    JiraDispatchAdapter,
    ResolvedCredentials,
    register_adapters,
)
from app.integrations.base import (
    AuthError,
    BaseConnector,
    IntegrationError,
    NotFoundError,
    RateLimitedError,
    ServerError,
)
from app.integrations.jira import JiraConnector
from app.integrations.mcp_client import (
    GmailMCPClient,
    MCPClient,
    MCPServerConfig,
    MCPToolError,
)


class _StubConnector(BaseConnector):
    """Minimal concrete connector for exercising the request pipeline."""

    name = "stub"


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def _status_connector(status: int, *, body: bytes = b"{}", headers=None):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, content=body, headers=headers)

    connector = _StubConnector(client=_client(handler), max_attempts=1)
    await connector.open()
    return connector


# ── Error taxonomy (status → typed exception) ────────────────────────────
@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, AuthError),
        (403, AuthError),
        (404, NotFoundError),
        (400, IntegrationError),
        (422, IntegrationError),
    ],
)
async def test_request_maps_client_errors(status: int, expected: type[IntegrationError]):
    connector = await _status_connector(status, body=b'{"detail": "nope"}')
    with pytest.raises(expected) as exc:
        await connector._request("GET", "https://example.test/x")
    assert exc.value.status_code == status


async def test_rate_limit_and_server_errors_are_typed():
    for status, expected in [(429, RateLimitedError), (503, ServerError)]:
        connector = await _status_connector(status, body=b"slow down")
        with pytest.raises(expected) as exc:
            await connector._request("GET", "https://example.test/x")
        assert exc.value.status_code == status


async def test_server_errors_are_retried_then_raised():
    """5xx retried with backoff up to max_attempts, then reraised."""
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        return httpx.Response(500, content=b"boom")

    connector = _StubConnector(client=_client(handler), max_attempts=3)
    await connector.open()
    with pytest.raises(ServerError):
        await connector._request("GET", "https://example.test/x")

    assert attempts["count"] == 3


async def test_success_json_and_empty_bodies():
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        if request.url.path == "/empty":
            return httpx.Response(204)
        return httpx.Response(200, json={"ok": True})

    connector = _StubConnector(client=_client(handler))
    await connector.open()

    assert await connector._request("GET", "https://example.test/data") == {"ok": True}
    assert await connector._request("DELETE", "https://example.test/empty") == {}
    assert seen["path"] == "/empty"


async def test_transport_errors_propagate_after_retries():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    connector = _StubConnector(client=_client(handler), max_attempts=2)
    await connector.open()
    with pytest.raises(httpx.ConnectError):
        await connector._request("GET", "https://example.test/x")


def test_client_property_before_open_raises():
    connector = _StubConnector()
    with pytest.raises(RuntimeError, match="before open"):
        _ = connector.client


async def test_owned_client_closed_on_exit():
    async with _StubConnector() as connector:
        assert connector.client is not None
    assert connector._client is None


# ── Jira connector ───────────────────────────────────────────────────────
async def test_jira_create_issue_payload():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization", "")
        captured["json"] = json.loads(request.content)
        return httpx.Response(201, json={"id": "10001", "key": "PROJ-1"})

    connector = JiraConnector(
        site="acme", email="bot@acme.in", api_token="tok", client=_client(handler)
    )
    await connector.open()
    created = await connector.create_issue(
        project_key="PROJ",
        summary="Ship RBAC",
        description="From meeting 42",
        labels=["vcall"],
        due_date="2026-10-20",
    )
    await connector.close()

    assert created["key"] == "PROJ-1"
    assert captured["method"] == "POST"
    assert captured["url"] == "https://acme.atlassian.net/rest/api/3/issue"
    # Jira Cloud Basic auth = email:token base64.
    assert captured["auth"].startswith("Basic ")
    fields = captured["json"]["fields"]
    assert fields["project"]["key"] == "PROJ"
    assert fields["summary"] == "Ship RBAC"
    assert fields["issuetype"]["name"] == "Task"
    assert fields["labels"] == ["vcall"]
    assert fields["duedate"] == "2026-10-20"
    # v3 expects Atlassian Document Format for the description.
    assert fields["description"]["content"][0]["content"][0]["text"] == "From meeting 42"


# ── Dispatch adapters ────────────────────────────────────────────────────
async def _no_credentials(tenant_id: str, provider: str):
    return None


async def _some_credentials(tenant_id: str, provider: str):
    return ResolvedCredentials(values={"site": "acme", "email": "b@acme.in", "api_token": "t"})


async def test_adapter_raises_without_credentials():
    adapter = JiraDispatchAdapter(_no_credentials, project_key="PROJ")
    with pytest.raises(DispatchError, match="No jira credentials"):
        await adapter.dispatch_action_item({"task": "x"}, tenant_id="t-1", meeting_id="m-1")


async def test_jira_adapter_dispatches_and_returns_key(monkeypatch):
    captured: dict = {}

    class FakeJiraConnector:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

        async def create_issue(self, **kwargs):
            captured["issue"] = kwargs
            return {"key": "OPS-9"}

    monkeypatch.setattr("app.integrations.jira.JiraConnector", FakeJiraConnector)

    adapter = JiraDispatchAdapter(_some_credentials, project_key="OPS")
    external_id = await adapter.dispatch_action_item(
        {"task": "Rotate the keys", "due_date": "2026-11-01"},
        tenant_id="t-1",
        meeting_id="m-1",
    )

    assert external_id == "OPS-9"
    assert captured["init"]["api_token"] == "t"
    assert captured["issue"]["project_key"] == "OPS"
    assert captured["issue"]["summary"] == "Rotate the keys"
    assert captured["issue"]["due_date"] == "2026-11-01"
    assert "meeting-m-1" in captured["issue"]["labels"]


async def test_jira_adapter_rejects_response_without_key(monkeypatch):
    class FakeJiraConnector:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

        async def create_issue(self, **kwargs):
            return {"unexpected": "shape"}

    monkeypatch.setattr("app.integrations.jira.JiraConnector", FakeJiraConnector)

    adapter = JiraDispatchAdapter(_some_credentials, project_key="OPS")
    with pytest.raises(DispatchError, match="missing issue key"):
        await adapter.dispatch_action_item({"task": "x"}, tenant_id="t", meeting_id="m")


def test_register_adapters_only_configured_ones():
    reg = DispatcherRegistry()
    registered = register_adapters(
        reg,
        _some_credentials,
        jira_project_key="PROJ",
        notion_database_id="db-123",
        # zoho/freshdesk intentionally not configured
    )

    assert registered == ["jira", "notion"]
    assert reg.systems() == ["jira", "notion"]

    reg.clear()
    assert register_adapters(reg, _some_credentials) == []


# ── MCP clients ──────────────────────────────────────────────────────────
def test_mcp_client_requires_connection():
    client = MCPClient(MCPServerConfig(command="npx", args=("-y", "server")))

    with pytest.raises(MCPToolError, match="not connected"):
        _ = client._require_session()


class _FakeContent:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class _FakeToolResult:
    def __init__(self, texts: list[str], is_error: bool = False):
        self.content = [_FakeContent(t) for t in texts]
        self.isError = is_error


class _FakeSession:
    def __init__(self, result: _FakeToolResult):
        self._result = result
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name: str, arguments: dict | None = None):
        self.calls.append((name, arguments or {}))
        return self._result


def _connected_client(result: _FakeToolResult) -> MCPClient:
    client = MCPClient(MCPServerConfig(command="noop"))
    client._session = _FakeSession(result)
    return client


async def test_mcp_call_tool_unwraps_single_text():
    client = _connected_client(_FakeToolResult(["message-id-1"]))
    out = await client.call_tool("send_email", {"to": "a@b.in"})

    assert out == "message-id-1"
    assert client._session.calls == [("send_email", {"to": "a@b.in"})]


async def test_mcp_call_tool_raises_on_error_payload():
    client = _connected_client(_FakeToolResult(["denied"], is_error=True))
    with pytest.raises(MCPToolError, match="returned an error"):
        await client.call_tool("send_email")


async def test_mcp_call_tool_wraps_transport_failures():
    class ExplodingSession:
        async def call_tool(self, name, arguments=None):
            raise ConnectionResetError("pipe closed")

    client = MCPClient(MCPServerConfig(command="noop"))
    client._session = ExplodingSession()
    with pytest.raises(MCPToolError, match="failed"):
        await client.call_tool("send_email")


async def test_gmail_wrapper_builds_expected_arguments():
    client = GmailMCPClient(MCPServerConfig(command="noop"))
    client._session = _FakeSession(_FakeToolResult(["ok"]))

    await client.send_email(to="priya@acme.in", subject="Hi", body="Notes", cc="team@acme.in")

    name, args = client._session.calls[0]
    assert name == "send_email"
    assert args == {
        "to": "priya@acme.in",
        "subject": "Hi",
        "body": "Notes",
        "cc": "team@acme.in",
    }


# Imported at the end to keep the linter happy about registry usage above.
