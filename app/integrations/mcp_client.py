"""Async Model Context Protocol (MCP) client wrappers.

Wraps the official ``mcp`` SDK around a reusable session so Google Workspace
tool-servers (Gmail, Calendar) can be driven with plain async calls.

Lifecycle: MCP transports are stateful context managers, so :class:`MCPClient`
opens the stdio transport + session through an :class:`~contextlib.AsyncExitStack`.
Callers use ``async with client.connected(): ...`` (or the ``connected()``
helper on the typed wrappers) — sessions never leak.

Tool names follow the upstream Google MCP server conventions
(``send_email``, ``create_event``); pass explicit names to override.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


class MCPToolError(Exception):
    """Raised when an MCP tool call fails or returns an error payload."""


@dataclass(frozen=True, slots=True)
class MCPServerConfig:
    """Launch configuration for a stdio MCP server process."""

    command: str
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)


class MCPClient:
    """Minimal async MCP client over a stdio server."""

    def __init__(self, config: MCPServerConfig) -> None:
        self._config = config
        self._session: Any = None

    @asynccontextmanager
    async def connected(self) -> AsyncIterator[MCPClient]:
        """Open transport + session; guaranteed cleanup on exit."""
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=self._config.command,
            args=list(self._config.args),
            env=dict(self._config.env) or None,
        )
        async with (
            stdio_client(params) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            self._session = session
            try:
                yield self
            finally:
                self._session = None

    def _require_session(self) -> Any:
        if self._session is None:
            raise MCPToolError("MCP client is not connected — use 'async with client.connected()'")
        return self._session

    async def list_tools(self) -> list[str]:
        """Names of tools exposed by the connected server."""
        session = self._require_session()
        result = await session.list_tools()
        return [tool.name for tool in result.tools]

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """Invoke a tool and unwrap its text payload."""
        session = self._require_session()
        try:
            result = await session.call_tool(name, arguments=arguments or {})
        except Exception as exc:  # SDK raises on transport/protocol errors
            raise MCPToolError(f"MCP tool '{name}' failed: {exc}") from exc

        if getattr(result, "isError", False):
            raise MCPToolError(f"MCP tool '{name}' returned an error: {result.content}")

        # Unwrap common single-text-content payloads into plain strings/dicts.
        content = getattr(result, "content", []) or []
        texts = [c.text for c in content if getattr(c, "type", None) == "text"]
        if len(texts) == 1:
            return texts[0]
        return texts or result.model_dump(mode="json")


# ── Typed Google Workspace wrappers ─────────────────────────────────────
class GmailMCPClient(MCPClient):
    """Gmail tool-server convenience wrapper."""

    async def send_email(
        self,
        *,
        to: str,
        subject: str,
        body: str,
        cc: str | None = None,
    ) -> Any:
        """Send an email via the Gmail MCP server's ``send_email`` tool."""
        arguments: dict[str, Any] = {"to": to, "subject": subject, "body": body}
        if cc:
            arguments["cc"] = cc
        return await self.call_tool("send_email", arguments)


class CalendarMCPClient(MCPClient):
    """Google Calendar tool-server convenience wrapper."""

    async def create_event(
        self,
        *,
        summary: str,
        start_iso: str,
        end_iso: str,
        attendees: list[str] | None = None,
        description: str | None = None,
        timezone: str = "Asia/Kolkata",
    ) -> Any:
        """Create a calendar event via the ``create_event`` tool."""
        arguments: dict[str, Any] = {
            "summary": summary,
            "start": start_iso,
            "end": end_iso,
            "timezone": timezone,
        }
        if attendees:
            arguments["attendees"] = attendees
        if description:
            arguments["description"] = description
        return await self.call_tool("create_event", arguments)
