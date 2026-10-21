"""Cross-cutting HTTP middleware.

Pure-ASGI (no ``BaseHTTPMiddleware``) so streaming/WebSocket responses are
untouched and there's no per-request task overhead.

Adds:
- ``X-Request-ID`` propagation (incoming header honored, else generated).
- Security response headers (HSTS, nosniff, frame denial, referrer policy).
- Access logging with latency + status.
"""

from __future__ import annotations

import logging
import time
import uuid

import structlog
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger("vcall.access")

# Paths that never need security headers (health probes).
_EXEMPT_PATHS = frozenset({"/api/v1/health", "/api/v1/health/ready"})


class RequestContextMiddleware:
    """ASGI middleware: request IDs, security headers, access logs."""

    def __init__(self, app: ASGIApp, *, enable_hsts: bool = False) -> None:
        self.app = app
        self._enable_hsts = enable_hsts

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # ── Request ID ────────────────────────────────────────────────
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
        request_id = headers.get("x-request-id") or uuid.uuid4().hex
        scope.setdefault("state", {})
        scope["state"]["request_id"] = request_id
        # Every log line emitted while handling this request carries the id.
        structlog.contextvars.bind_contextvars(request_id=request_id)

        path = scope.get("path", "")
        start = time.perf_counter()
        status_code = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                raw = list(message.get("headers", []))
                raw.extend(
                    [
                        (b"x-request-id", request_id.encode()),
                        (b"x-content-type-options", b"nosniff"),
                        (b"x-frame-options", b"DENY"),
                        (b"referrer-policy", b"strict-origin-when-cross-origin"),
                        (b"permissions-policy", b"camera=(), microphone=(self), geolocation=()"),
                    ]
                )
                if self._enable_hsts:
                    raw.append(
                        (b"strict-transport-security", b"max-age=63072000; includeSubDomains")
                    )
                message = {**message, "headers": raw}
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            structlog.contextvars.clear_contextvars()
            if path not in _EXEMPT_PATHS:
                elapsed_ms = (time.perf_counter() - start) * 1000
                logger.info(
                    "%s %s %s %.1fms rid=%s",
                    scope.get("method", "?"),
                    path,
                    status_code,
                    elapsed_ms,
                    request_id,
                )
