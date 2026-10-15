"""Unit/integration tests: password hashing, JWTs, and auth dependencies."""

from __future__ import annotations

import uuid

import pytest

from app.core.security import (
    AuthTokenError,
    create_access_token,
    decode_token,
    hash_password,
    verify_password,
)


# ── Password hashing (stdlib scrypt) ─────────────────────────────────────
def test_hash_and_verify_roundtrip():
    hashed = hash_password("correct horse battery")
    assert hashed.startswith("scrypt$ln=14,r=8,p=1$")
    assert verify_password("correct horse battery", hashed)


def test_wrong_password_rejected():
    hashed = hash_password("secret12345")
    assert not verify_password("secret1234", hashed)


def test_hash_is_salted():
    """Same password → different hashes (unique salt per hash)."""
    assert hash_password("secret12345") != hash_password("secret12345")


def test_short_password_rejected():
    with pytest.raises(ValueError, match="at least 8"):
        hash_password("short1")


def test_verify_malformed_and_missing_hashes():
    assert not verify_password("anything", None)
    assert not verify_password("anything", "garbage-without-fields")
    assert not verify_password("anything", "bcrypt$ln=14$aa$bb")


# ── JWT tokens ───────────────────────────────────────────────────────────
def test_access_token_roundtrip():
    tenant = str(uuid.uuid4())
    token = create_access_token(subject="user-1", tenant_id=tenant, role="admin")

    payload = decode_token(token)
    assert payload.sub == "user-1"
    assert payload.tenant_id == tenant
    assert payload.role == "admin"
    assert payload.exp is not None


def test_expired_token_rejected():
    token = create_access_token(subject="user-1", tenant_id=str(uuid.uuid4()), expires_minutes=-1)
    with pytest.raises(AuthTokenError, match="Invalid token"):
        decode_token(token)


def test_tampered_token_rejected():
    token = create_access_token(subject="user-1", tenant_id=str(uuid.uuid4()))
    with pytest.raises(AuthTokenError):
        decode_token(token[:-2] + "xy")


def test_garbage_token_rejected():
    with pytest.raises(AuthTokenError):
        decode_token("not.a.jwt")


def test_extra_claims_ignored_by_payload_model():
    """Unknown claims must not break validation (model ignores extras)."""
    token = create_access_token(
        subject="user-1",
        tenant_id=str(uuid.uuid4()),
        extra_claims={"unrelated": 42},
    )
    payload = decode_token(token)
    assert payload.model_dump(exclude={"exp", "iat"}) == {
        "sub": "user-1",
        "tenant_id": payload.tenant_id,
        "role": "member",
    }


# ── Auth dependency (HTTP level) ─────────────────────────────────────────
async def test_me_requires_token(anon_client):
    response = await anon_client.get("/api/v1/auth/me")
    assert response.status_code == 401
    assert response.headers["www-authenticate"].lower() == "bearer"


async def test_me_rejects_invalid_token(anon_client):
    response = await anon_client.get(
        "/api/v1/auth/me", headers={"Authorization": "Bearer not-a-token"}
    )
    assert response.status_code == 401


async def test_me_returns_identity_for_valid_token(anon_client, seed_tenant_user):
    tenant, user, token = seed_tenant_user
    response = await anon_client.get(
        "/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == str(user.id)
    assert body["tenant_id"] == str(tenant.id)
    assert body["role"] == "admin"


async def test_inactive_user_rejected(anon_client, seed_tenant_user):
    from app.db.session import get_session_factory

    _, user, token = seed_tenant_user
    async with get_session_factory()() as session:
        row = await session.get(type(user), user.id)
        assert row is not None
        row.is_active = False
        await session.commit()

    response = await anon_client.get(
        "/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 401
