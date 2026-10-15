"""Health & readiness endpoints."""

from __future__ import annotations

import logging

import redis.asyncio as aioredis
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app import __version__
from app.core.config import get_settings
from app.db.session import get_engine
from app.schemas.api import HealthOut

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthOut)
async def liveness() -> HealthOut:
    """Cheap liveness probe (no dependencies)."""
    return HealthOut(status="ok", service=get_settings().service_name, version=__version__)


@router.get("/health/ready")
async def readiness() -> JSONResponse:
    """Readiness probe: verifies DB + Redis connectivity."""
    checks: dict[str, str] = {}

    # Database
    try:
        async with get_engine().connect() as conn:
            await conn.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as exc:
        logger.warning("Readiness: DB check failed: %s", exc)
        checks["database"] = f"error: {exc}"

    # Redis
    try:
        client = aioredis.from_url(get_settings().redis_url)
        try:
            await client.ping()
            checks["redis"] = "ok"
        finally:
            await client.aclose()
    except Exception as exc:
        logger.warning("Readiness: Redis check failed: %s", exc)
        checks["redis"] = f"error: {exc}"

    healthy = all(v == "ok" for v in checks.values())
    payload = HealthOut(
        status="ok" if healthy else "degraded",
        service=get_settings().service_name,
        version=__version__,
        checks=checks,
    )
    return JSONResponse(
        status_code=status.HTTP_200_OK if healthy else status.HTTP_503_SERVICE_UNAVAILABLE,
        content=payload.model_dump(),
    )
