"""Global rate limiting (slowapi).

Keyed per-IP by default; auth'd endpoints re-key on tenant via
``key_func`` reading the resolved context (falls back to IP).
"""

from __future__ import annotations

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import get_settings

settings = get_settings()


def _key_func(request: Request) -> str:
    """Prefer the authenticated tenant; fall back to client IP."""
    context = getattr(request.state, "tenant_context", None)
    if context is not None and getattr(context, "tenant_id", None):
        return f"tenant:{context.tenant_id}"
    return get_remote_address(request)


limiter = Limiter(key_func=_key_func, default_limits=[settings.rate_limit_default])


def rate_limit_heavy() -> str:
    """Limit string for expensive endpoints (uploads, reprocessing)."""
    return settings.rate_limit_heavy
