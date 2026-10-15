"""Shared pytest fixtures.

Environment is configured *before* any app import so settings/engine pick up
the test database. Every test gets a fresh schema.

``DATABASE_URL`` can be overridden by the environment (CI points it at a real
PostgreSQL service); everything else is pinned for deterministic offline runs.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

# ── Configure env before importing app modules ──────────────────────────
_TMP_DIR = tempfile.mkdtemp(prefix="vcall-tests-")
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("DATABASE_URL", f"sqlite+aiosqlite:///{_TMP_DIR}/test.db")
os.environ["REDIS_URL"] = "redis://localhost:6399/0"  # unreachable → no side effects
os.environ["JWT_SECRET"] = "test-jwt-secret-0123456789abcdefghij"
os.environ["FERNET_KEY"] = "X_ePszmQLHCyPP_6t_hJv4WraB4-Ym40uJiLxxGWSh8="
os.environ["SCRUB_TRANSCRIPT_PII"] = "true"
os.environ["AUDIO_ARTIFACTS_DIR"] = f"{_TMP_DIR}/artifacts"

from app.core.config import clear_settings_cache, get_settings  # noqa: E402

clear_settings_cache()

import httpx  # noqa: E402
from httpx import ASGITransport  # noqa: E402

from app.core.rate_limit import limiter  # noqa: E402
from app.core.security import create_access_token, hash_password  # noqa: E402
from app.db.base import Base  # noqa: E402
from app.db.enums import UserRole  # noqa: E402
from app.db.models import Tenant, User  # noqa: E402
from app.db.session import dispose_engine, get_engine, get_session_factory  # noqa: E402

# Slowapi's in-memory counters persist across tests; disable for the suite.
limiter.enabled = False


@pytest.fixture(scope="session")
def settings():
    return get_settings()


@pytest.fixture(autouse=True)
async def fresh_db() -> AsyncIterator[None]:
    """Recreate the schema before every test; dispose engines after."""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield
    await dispose_engine()


@pytest.fixture
async def seed_tenant_user() -> tuple[Tenant, User, str]:
    """Seed one tenant + admin user; return (tenant, user, jwt)."""
    async with get_session_factory()() as session:
        tenant = Tenant(name="Acme India", slug=f"acme-{uuid.uuid4().hex[:6]}")
        session.add(tenant)
        await session.flush()
        user = User(
            tenant_id=tenant.id,
            email="priya@acme.in",
            hashed_password=hash_password("secret123"),
            full_name="Priya",
            role=UserRole.ADMIN,
        )
        session.add(user)
        await session.flush()
        await session.commit()
        token = create_access_token(subject=str(user.id), tenant_id=str(tenant.id), role="admin")
        return tenant, user, token


@pytest.fixture
async def client(seed_tenant_user) -> AsyncIterator[httpx.AsyncClient]:
    """Authenticated API client bound to the seeded tenant."""
    from app.main import create_app

    _, _, token = seed_tenant_user
    app = create_app()
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"Authorization": f"Bearer {token}"},
    ) as ac:
        yield ac


@pytest.fixture
async def anon_client() -> AsyncIterator[httpx.AsyncClient]:
    """Unauthenticated API client."""
    from app.main import create_app

    app = create_app()
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ── Audio helpers ───────────────────────────────────────────────────────
@pytest.fixture
def make_wav(tmp_path: Path):
    """Factory: write a small sine-wave WAV and return its path."""

    def _make(name: str = "sample.wav", seconds: float = 0.5, freq: int = 440) -> Path:
        import math
        import struct
        import wave

        path = tmp_path / name
        sample_rate = 16_000
        frames = int(sample_rate * seconds)
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(
                b"".join(
                    struct.pack("<h", int(12000 * math.sin(2 * math.pi * freq * i / sample_rate)))
                    for i in range(frames)
                )
            )
        return path

    return _make


# ── Agent fakes ─────────────────────────────────────────────────────────
class FakeStructured:
    """Mimics a ``with_structured_output`` model."""

    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    async def ainvoke(self, prompt: str):
        self.calls += 1
        return self.payload


class FailsOnce(FakeStructured):
    """Fails on the first call, then succeeds."""

    async def ainvoke(self, prompt: str):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("transient LLM failure")
        return self.payload


@pytest.fixture
def fake_structured_cls():
    return FakeStructured
