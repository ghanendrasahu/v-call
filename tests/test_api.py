"""API tests: health, auth, meetings, integrations, WebSocket auth, headers.

Celery enqueueing is neutered per test (no live broker in CI); everything
else runs against the real ASGI app + fresh SQLite schema.
"""

from __future__ import annotations

import uuid

import pytest
from starlette.websockets import WebSocketDisconnect

WAV_BYTES = b"RIFF\x00\x00\x00\x00WAVEfmt " + b"\x00" * 64


def _neuter_queue(monkeypatch) -> list[tuple[str, str]]:
    """Replace Celery enqueue with a recorder (returns the recorder list)."""
    from app.workers import tasks

    queued: list[tuple[str, str]] = []
    monkeypatch.setattr(tasks.process_meeting, "delay", lambda *args: queued.append(args))
    return queued


# ── Health & root ────────────────────────────────────────────────────────
async def test_root_reports_service(anon_client):
    response = await anon_client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "v-call"


async def test_liveness(anon_client):
    response = await anon_client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "v-call"
    assert body["version"]


async def test_readiness_reports_degraded_without_redis(anon_client):
    """Redis is unreachable in tests → 503 with per-dependency detail."""
    response = await anon_client.get("/api/v1/health/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["checks"]["database"] == "ok"
    assert body["checks"]["redis"].startswith("error")


async def test_security_headers_and_request_id(anon_client):
    response = await anon_client.get("/", headers={"X-Request-ID": "req-123"})

    assert response.headers["x-request-id"] == "req-123"  # propagated
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "strict-origin-when-cross-origin"


# ── Auth ─────────────────────────────────────────────────────────────────
async def test_register_login_and_me(anon_client):
    payload = {
        "tenant_name": "Zeta Corp",
        "email": "admin@zeta.in",
        "password": "supersecret1",
        "full_name": "Zeta Admin",
    }
    created = await anon_client.post("/api/v1/auth/register", json=payload)
    assert created.status_code == 201
    body = created.json()
    assert body["email"] == "admin@zeta.in"
    assert body["role"] == "admin"

    # Duplicate bootstrap is rejected (slug + email conflicts).
    dup = await anon_client.post("/api/v1/auth/register", json=payload)
    assert dup.status_code == 409

    token_resp = await anon_client.post(
        "/api/v1/auth/token",
        data={"username": payload["email"], "password": payload["password"]},
    )
    assert token_resp.status_code == 200
    token = token_resp.json()["access_token"]
    assert token_resp.json()["token_type"] == "bearer"

    me = await anon_client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["tenant_id"] == body["tenant_id"]


async def test_login_rejects_bad_credentials(anon_client):
    bad = await anon_client.post(
        "/api/v1/auth/token", data={"username": "nobody@x.in", "password": "wrong-pass"}
    )
    assert bad.status_code == 401


async def test_register_rejects_invalid_payload(anon_client):
    resp = await anon_client.post(
        "/api/v1/auth/register",
        json={"tenant_name": "X", "email": "not-an-email", "password": "short"},
    )
    assert resp.status_code == 422


# ── Meetings ─────────────────────────────────────────────────────────────
async def test_meetings_require_auth(anon_client):
    assert (await anon_client.get("/api/v1/meetings")).status_code == 401


async def test_upload_list_detail_and_delete(client, monkeypatch, tmp_path):
    queued = _neuter_queue(monkeypatch)

    uploaded = await client.post(
        "/api/v1/meetings",
        files={"file": ("weekly-sync.wav", WAV_BYTES, "audio/wav")},
        data={"title": "Weekly sync", "source": "upload"},
    )
    assert uploaded.status_code == 202
    accepted = uploaded.json()
    meeting_id = accepted["meeting_id"]
    assert accepted["status"] == "pending"
    # Enqueued with string ids (Celery JSON serialisation).
    assert queued and queued[0][0] == meeting_id
    assert isinstance(queued[0][1], str)

    # Uploaded bytes were streamed to the tenant's artifacts directory.
    listed = await client.get("/api/v1/meetings")
    assert listed.status_code == 200
    assert [m["id"] for m in listed.json()] == [meeting_id]
    assert listed.json()[0]["title"] == "Weekly sync"

    detail = await client.get(f"/api/v1/meetings/{meeting_id}")
    assert detail.status_code == 200
    detail_body = detail.json()
    assert detail_body["segments"] == []
    assert detail_body["action_items"] == []
    assert detail_body["follow_up_emails"] == []
    assert detail_body["sentiment"] is None

    assert (await client.get(f"/api/v1/meetings/{meeting_id}/transcript")).json() == []
    assert (await client.get(f"/api/v1/meetings/{meeting_id}/action-items")).json() == []
    assert (await client.get(f"/api/v1/meetings/{meeting_id}/follow-up-emails")).json() == []

    # Right to erasure: delete removes the row (and any raw audio file).
    deleted = await client.delete(f"/api/v1/meetings/{meeting_id}")
    assert deleted.status_code == 204
    assert (await client.get(f"/api/v1/meetings/{meeting_id}")).status_code == 404


async def test_upload_validation_errors(client, monkeypatch):
    _neuter_queue(monkeypatch)

    unsupported = await client.post(
        "/api/v1/meetings",
        files={"file": ("notes.exe", b"MZ", "application/octet-stream")},
        data={"title": "Bad file"},
    )
    assert unsupported.status_code == 415

    empty = await client.post(
        "/api/v1/meetings",
        files={"file": ("empty.wav", b"", "audio/wav")},
        data={"title": "Empty"},
    )
    assert empty.status_code == 400

    missing_title = await client.post(
        "/api/v1/meetings",
        files={"file": ("a.wav", WAV_BYTES, "audio/wav")},
    )
    assert missing_title.status_code == 422


async def test_meeting_not_found_is_404(client):
    ghost = uuid.uuid4()
    assert (await client.get(f"/api/v1/meetings/{ghost}")).status_code == 404
    assert (await client.get(f"/api/v1/meetings/{ghost}/transcript")).status_code == 404
    assert (await client.delete(f"/api/v1/meetings/{ghost}")).status_code == 404
    assert (await client.post(f"/api/v1/meetings/{ghost}/reprocess")).status_code == 404


async def test_meeting_detail_includes_transcript_and_sentiment(client):
    from app.db.enums import AgentRunStatus, MeetingStatus
    from app.db.models import AgentRun
    from app.db.repositories import MeetingRepository
    from app.db.session import get_session_factory

    # Seed a completed meeting with transcript + agent run directly.
    async with get_session_factory()() as session:
        from app.core.security import decode_token

        token = client.headers["Authorization"].split()[1]
        tenant_id = uuid.UUID(decode_token(token).tenant_id)

        repo = MeetingRepository(session)
        meeting = await repo.create(
            tenant_id=tenant_id, title="Completed", storage_path="/tmp/gone.wav"
        )
        await repo.update_status(meeting, MeetingStatus.COMPLETED)
        await repo.attach_segments(
            meeting.id,
            tenant_id,
            [
                {"speaker_label": "Speaker 00", "start_ms": 0, "end_ms": 500, "text": "hi"},
                {"speaker_label": "Speaker 01", "start_ms": 500, "end_ms": 900, "text": "yo"},
            ],
        )
        session.add(
            AgentRun(
                meeting_id=meeting.id,
                tenant_id=tenant_id,
                status=AgentRunStatus.SUCCEEDED,
                graph_state={
                    "sentiment_analysis": {
                        "overall": "mixed",
                        "engagement_score": 0.55,
                        "speaker_dynamics": {},
                        "summary": "Heated but productive.",
                    }
                },
                errors=[],
                completed_at=None,
            )
        )
        await session.commit()
        meeting_id = str(meeting.id)

    detail = (await client.get(f"/api/v1/meetings/{meeting_id}")).json()
    assert [s["text"] for s in detail["segments"]] == ["hi", "yo"]
    assert detail["sentiment"]["overall"] == "mixed"
    assert detail["status"] == "completed"

    transcript = (await client.get(f"/api/v1/meetings/{meeting_id}/transcript")).json()
    assert [s["speaker_label"] for s in transcript] == ["Speaker 00", "Speaker 01"]


async def test_reprocess_requires_raw_audio(client, monkeypatch, seed_tenant_user):
    queued = _neuter_queue(monkeypatch)
    tenant, _, _ = seed_tenant_user

    from app.db.repositories import MeetingRepository
    from app.db.session import get_session_factory

    async with get_session_factory()() as session:
        repo = MeetingRepository(session)
        scrubbed = await repo.create(tenant_id=tenant.id, title="Scrubbed")
        stored = await repo.create(tenant_id=tenant.id, title="Stored", storage_path="/tmp/x.wav")
        await session.commit()
        scrubbed_id, stored_id = str(scrubbed.id), str(stored.id)

    conflict = await client.post(f"/api/v1/meetings/{scrubbed_id}/reprocess")
    assert conflict.status_code == 409
    assert "scrubbed" in conflict.json()["detail"]

    accepted = await client.post(f"/api/v1/meetings/{stored_id}/reprocess")
    assert accepted.status_code == 202
    assert queued[-1][0] == stored_id


async def test_upload_isolation_between_tenants(client, anon_client, monkeypatch, tmp_path):
    """A second tenant (registered through the API) cannot see our meetings."""
    _neuter_queue(monkeypatch)

    uploaded = await client.post(
        "/api/v1/meetings",
        files={"file": ("private.wav", WAV_BYTES, "audio/wav")},
        data={"title": "Private plans"},
    )
    meeting_id = uploaded.json()["meeting_id"]

    # Bootstrap another tenant through the API and authenticate as it.
    reg = await anon_client.post(
        "/api/v1/auth/register",
        json={
            "tenant_name": "Rival Corp",
            "email": "rival@corp.in",
            "password": "rival-pass1",
        },
    )
    assert reg.status_code == 201
    token = (
        await anon_client.post(
            "/api/v1/auth/token",
            data={"username": "rival@corp.in", "password": "rival-pass1"},
        )
    ).json()["access_token"]
    rival = anon_client
    rival.headers["Authorization"] = f"Bearer {token}"

    assert (await rival.get("/api/v1/meetings")).json() == []
    assert (await rival.get(f"/api/v1/meetings/{meeting_id}")).status_code == 404
    assert (await rival.delete(f"/api/v1/meetings/{meeting_id}")).status_code == 404


# ── Integrations ─────────────────────────────────────────────────────────
async def test_credential_lifecycle_is_masked(client):
    put = await client.put(
        "/api/v1/integrations/jira",
        json={"values": {"token": "jira-super-secret-token"}, "scopes": ["read"]},
    )
    assert put.status_code == 200
    body = put.json()
    assert body["provider"] == "jira"
    assert "jira-super-secret-token" not in str(body)
    assert body["access_token_masked"]  # masked, never blank

    listed = (await client.get("/api/v1/integrations")).json()
    assert [c["provider"] for c in listed] == ["jira"]
    assert "jira-super-secret-token" not in str(listed)

    assert (await client.delete("/api/v1/integrations/jira")).status_code == 204
    assert (await client.get("/api/v1/integrations")).json() == []
    assert (await client.delete("/api/v1/integrations/jira")).status_code == 404


async def test_unknown_provider_rejected(client):
    resp = await client.put("/api/v1/integrations/slack", json={"values": {"token": "x"}})
    assert resp.status_code == 422


async def test_integrations_require_auth(anon_client):
    assert (await anon_client.get("/api/v1/integrations")).status_code == 401


# ── WebSocket auth ───────────────────────────────────────────────────────
def _ws_client():
    """Sync ASGI test client (starlette); httpx has no WebSocket support."""
    from fastapi.testclient import TestClient

    from app.main import create_app

    return TestClient(create_app())


def test_websocket_rejects_invalid_token():
    test_client = _ws_client()
    with (
        pytest.raises(WebSocketDisconnect) as exc,
        test_client.websocket_connect(f"/ws/meetings/{uuid.uuid4()}/progress?token=nope"),
    ):
        pass

    assert exc.value.code == 4401


def test_websocket_rejects_missing_token():
    test_client = _ws_client()
    with (
        pytest.raises(WebSocketDisconnect) as exc,
        test_client.websocket_connect(f"/ws/meetings/{uuid.uuid4()}/progress"),
    ):
        pass

    assert exc.value.code == 4401
