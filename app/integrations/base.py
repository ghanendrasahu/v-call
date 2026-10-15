"""Shared HTTP foundation for all REST connectors.

Responsibilities every connector inherits:
- Async ``httpx`` client with sane timeouts and connection pooling.
- ``tenacity`` exponential backoff on transient failures (network errors,
  429s, 5xx) so flaky SaaS APIs never wedge a pipeline run.
- Uniform error taxonomy mapping HTTP status codes to typed exceptions.
"""

from __future__ import annotations

import logging
from typing import Any, Self

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_ATTEMPTS = 4


class IntegrationError(Exception):
    """Base class for connector failures."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class AuthError(IntegrationError):
    """401/403 — token expired, revoked, or lacks scope."""


class NotFoundError(IntegrationError):
    """404 — referenced resource does not exist."""


class RateLimitedError(IntegrationError):
    """429 — caller exceeded the provider's rate limit."""


class ServerError(IntegrationError):
    """5xx — provider-side outage (retried, then raised)."""


class BaseConnector:
    """Reusable async HTTP base with retry + error mapping."""

    #: Provider identifier (e.g. ``"jira"``).
    name: str = "base"

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ) -> None:
        # An injected client is owned by the caller; otherwise we own ours.
        self._client = client
        self._owns_client = client is None
        self._timeout = timeout
        self._max_attempts = max_attempts

    # ── Lifecycle ──────────────────────────────────────────────────────
    async def __aenter__(self) -> Self:
        await self.open()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def open(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout),
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
                follow_redirects=True,
            )

    async def close(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError(
                f"{self.name} connector used before open(); use 'async with' or call open()"
            )
        return self._client

    # ── Request pipeline ───────────────────────────────────────────────
    async def _request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        json: Any | None = None,
        headers: dict[str, str] | None = None,
        auth: httpx.Auth | tuple[str, str] | None = None,
    ) -> Any:
        """HTTP call with exponential-backoff retry and typed errors."""
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_exponential(multiplier=1, min=1, max=30),
            retry=retry_if_exception_type((httpx.TransportError, RateLimitedError, ServerError)),
            reraise=True,
        ):
            with attempt:
                response = await self.client.request(
                    method, url, params=params, json=json, headers=headers, auth=auth
                )
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    logger.warning("%s rate-limited (retry-after=%s)", self.name, retry_after)
                    raise RateLimitedError(f"{self.name} rate limited", status_code=429)
                if response.status_code in (401, 403):
                    raise AuthError(
                        f"{self.name} auth failed ({response.status_code}): {response.text[:200]}",
                        status_code=response.status_code,
                    )
                if response.status_code == 404:
                    raise NotFoundError(f"{self.name} resource not found: {url}", status_code=404)
                if response.status_code >= 500:
                    raise ServerError(
                        f"{self.name} server error {response.status_code}",
                        status_code=response.status_code,
                    )
                if response.status_code >= 400:
                    raise IntegrationError(
                        f"{self.name} request failed ({response.status_code}): "
                        f"{response.text[:300]}",
                        status_code=response.status_code,
                    )
                if response.status_code == 204 or not response.content:
                    return {}
                return response.json()
