"""FastAPI application factory.

Wires CORS, rate limiting, routers, WebSocket streaming, and (optionally)
OpenTelemetry instrumentation. Heavy integrations are imported lazily so the
web image doesn't need torch.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app import __version__
from app.api import websocket
from app.api.routes import auth, health, integrations, meetings
from app.core.config import get_settings
from app.core.middleware import RequestContextMiddleware
from app.core.rate_limit import limiter
from app.db.session import dispose_engine

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup/shutdown hooks."""
    settings = get_settings()
    logger.info(
        "Starting %s v%s (%s)",
        settings.service_name,
        __version__,
        settings.environment,
    )
    yield
    # Gracefully close the shared DB engine.
    await dispose_engine()
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    """Build the ASGI application."""
    settings = get_settings()

    app = FastAPI(
        title="AI Meeting Intelligence Platform",
        description=(
            "Diarized transcripts, action items, sentiment, and follow-up "
            "emails — pushed into Jira, Notion, Zoho, and Freshdesk."
        ),
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs" if not settings.is_production else None,
        redoc_url=None,
    )

    # ── Rate limiting ──────────────────────────────────────────────────
    app.state.limiter = limiter
    # slowapi's handler narrows `exc` beyond FastAPI's `ExceptionHandler`
    # signature (reportArgumentType false positive — runtime is correct).
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore

    # ── Middleware (request IDs + security headers) ────────────────────
    app.add_middleware(RequestContextMiddleware, enable_hsts=settings.is_production)

    # ── CORS ───────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ────────────────────────────────────────────────────────
    api = settings.api_prefix
    app.include_router(health.router, prefix=api)
    app.include_router(auth.router, prefix=api)
    app.include_router(meetings.router, prefix=api)
    app.include_router(integrations.router, prefix=api)
    app.include_router(websocket.router)

    # ── Observability (optional) ───────────────────────────────────────
    _try_instrument(app)

    @app.get("/", include_in_schema=False)
    async def root() -> JSONResponse:
        return JSONResponse(
            {"service": settings.service_name, "version": __version__, "docs": "/docs"}
        )

    return app


def _try_instrument(app: FastAPI) -> None:
    """Attach OTel instrumentation when the observability extra is present."""
    try:
        from opentelemetry.instrumentation.fastapi import (  # type: ignore
            FastAPIInstrumentor,
        )

        FastAPIInstrumentor.instrument_app(app)
        logger.info("OpenTelemetry FastAPI instrumentation enabled")
    except ImportError:
        logger.debug("OpenTelemetry not installed; skipping instrumentation")


# Module-level ASGI entrypoint (uvicorn `app.main:app`).
app = create_app()


def run() -> None:
    """Console entrypoint: ``vcall-api``."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=not settings.is_production,
    )


if __name__ == "__main__":
    run()
