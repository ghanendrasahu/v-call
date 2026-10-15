"""Worker tests: Celery configuration, task boundaries, compliance sweep.

No live Redis broker is required: tasks are exercised through their ``run``
entrypoints with Redis-touching collaborators monkeypatched.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from typing import ClassVar

import pytest

from app.workers import tasks
from app.workers.celery_app import celery_app


# ── Celery configuration ─────────────────────────────────────────────────
def test_celery_queue_configuration():
    conf = celery_app.conf
    queue_names = {queue.name for queue in conf.task_queues}

    assert {"default", "audio", "agents", "dlq"} <= queue_names
    assert conf.task_default_queue == "default"
    assert conf.task_routes["app.workers.tasks.process_meeting"] == {"queue": "audio"}
    assert conf.task_routes["app.workers.tasks.scrub_expired_audio"] == {"queue": "default"}


def test_celery_reliability_settings():
    conf = celery_app.conf

    # Late acks + reject-on-lost guarantee at-least-once for long audio jobs.
    assert conf.task_acks_late is True
    assert conf.task_reject_on_worker_lost is True
    assert conf.worker_prefetch_multiplier == 1  # fair scheduling across tenants
    assert conf.task_time_limit == 60 * 60
    assert conf.task_soft_time_limit == 55 * 60
    assert conf.broker_transport_options["visibility_timeout"] == 60 * 60 * 3
    # JSON-only wire format.
    assert "json" in conf.accept_content
    assert conf.task_serializer == "json"
    assert conf.enable_utc is True
    assert str(conf.timezone) in ("UTC", "Etc/UTC")


def test_beat_schedule_covers_compliance_and_dlq():
    beat = celery_app.conf.beat_schedule

    scrub = beat["scrub-expired-raw-audio"]
    assert scrub["task"] == "app.workers.tasks.scrub_expired_audio"
    assert scrub["schedule"] == 60 * 60.0  # hourly retention sweep

    replay = beat["replay-dead-letter"]
    assert replay["task"] == "app.workers.tasks.replay_dead_letter"
    assert replay["schedule"] == 60 * 30.0


def test_expected_tasks_are_registered():
    registered = set(celery_app.tasks)
    assert {
        "app.workers.tasks.process_meeting",
        "app.workers.tasks.scrub_expired_audio",
        "app.workers.tasks.replay_dead_letter",
        "app.workers.tasks.send_follow_up_emails",
    } <= registered


# ── process_meeting boundary (string IDs from Celery JSON) ───────────────
async def test_process_meeting_unknown_id_raises_value_error():
    """String ids must be coerced to UUIDs before hitting GUID columns."""
    with pytest.raises(ValueError, match="not found"):
        await tasks._process_meeting_async(str(uuid.uuid4()), str(uuid.uuid4()))


async def test_process_meeting_malformed_id_raises_value_error():
    with pytest.raises(ValueError):
        await tasks._process_meeting_async("not-a-uuid", str(uuid.uuid4()))


async def test_process_meeting_without_audio_is_rejected(seed_tenant_user):
    from app.db.repositories import MeetingRepository
    from app.db.session import get_session_factory

    tenant, _, _ = seed_tenant_user
    async with get_session_factory()() as session:
        repo = MeetingRepository(session)
        meeting = await repo.create(tenant_id=tenant.id, title="No audio stored")
        await session.commit()
        meeting_id, tenant_id = str(meeting.id), str(tenant.id)

    # Exercises the full string-id → UUID path up to the storage check.
    with pytest.raises(ValueError, match="no stored audio"):
        await tasks._process_meeting_async(meeting_id, tenant_id)


# ── Progress publishing (Redis unreachable in tests) ─────────────────────
async def test_publish_progress_swallows_broker_errors():
    # Broker cannot reach redis://localhost:6399 — must not raise.
    await tasks._publish_progress(str(uuid.uuid4()), "transcribing", 0.4, "working")


# ── Compliance sweep task ────────────────────────────────────────────────
async def test_scrub_expired_audio_sweeps_old_recordings(tmp_path, seed_tenant_user):
    from app.db.repositories import MeetingRepository
    from app.db.session import get_session_factory, session_scope

    tenant, _, _ = seed_tenant_user
    old_audio = tmp_path / "old.wav"
    old_audio.write_bytes(b"old-bytes")
    fresh_audio = tmp_path / "fresh.wav"
    fresh_audio.write_bytes(b"fresh-bytes")

    async with get_session_factory()() as session:
        repo = MeetingRepository(session)
        old = await repo.create(
            tenant_id=tenant.id, title="Past retention", storage_path=str(old_audio)
        )
        old.created_at = datetime.now(UTC) - timedelta(hours=5)
        fresh = await repo.create(
            tenant_id=tenant.id, title="Within retention", storage_path=str(fresh_audio)
        )
        await session.commit()
        old_id, fresh_id = old.id, fresh.id

    result = await asyncio.to_thread(tasks.scrub_expired_audio.run)

    assert result == {"scrubbed": 1}
    assert not old_audio.exists()  # securely removed from disk
    assert fresh_audio.exists()  # still inside the retention window

    async with session_scope() as session:
        repo = MeetingRepository(session)
        old_row = await repo.get(old_id, tenant_id=tenant.id)
        fresh_row = await repo.get(fresh_id, tenant_id=tenant.id)
        assert old_row is not None and old_row.storage_path is None  # cleared
        assert fresh_row is not None
        assert fresh_row.storage_path == str(fresh_audio)


# ── Dead-letter replay ───────────────────────────────────────────────────
async def test_replay_dead_letter_requeues_only_matching_entries(monkeypatch):
    async def fake_pop(limit=10):
        return [
            {
                "task": "app.workers.tasks.process_meeting",
                "meeting_id": "m-1",
                "tenant_id": "t-1",
            },
            {
                "task": "app.workers.tasks.send_follow_up_emails",
                "meeting_id": "m-2",
                "tenant_id": "t-1",
            },
        ]

    monkeypatch.setattr("app.workers.dead_letter.pop_dead_letter_batch", fake_pop)
    queued: list[tuple[str, str]] = []
    monkeypatch.setattr(tasks.process_meeting, "delay", lambda *args: queued.append(args))

    result = await asyncio.to_thread(tasks.replay_dead_letter.run)

    assert result == {"replayed": 1, "skipped": 1}
    assert queued == [("m-1", "t-1")]


# ── Follow-up email sender (no-op path) ──────────────────────────────────
async def test_send_follow_up_emails_noop_without_pending():
    # Unknown meeting → nothing pending → MCP server is never spawned.
    result = await asyncio.to_thread(
        tasks.send_follow_up_emails.run, str(uuid.uuid4()), str(uuid.uuid4())
    )
    assert result == {"sent": 0}


async def test_send_follow_up_emails_delivers_pending_drafts(seed_tenant_user, monkeypatch):
    from app.db.repositories import FollowUpEmailRepository, MeetingRepository
    from app.db.session import get_session_factory, session_scope

    tenant, _, _ = seed_tenant_user
    async with get_session_factory()() as session:
        meeting = await MeetingRepository(session).create(tenant_id=tenant.id, title="Drafts")
        await FollowUpEmailRepository(session).replace_for_meeting(
            meeting_id=meeting.id,
            tenant_id=tenant.id,
            emails=[{"recipient_email": "a@acme.in", "email_body": "hello there"}],
        )
        await session.commit()
        meeting_id, tenant_id = str(meeting.id), str(tenant.id)

    # Substitute the Gmail MCP tool-server with an in-memory double.
    class FakeGmailClient:
        instances: ClassVar[list] = []

        def __init__(self, config):
            self.config = config
            self.sent: list[str] = []
            FakeGmailClient.instances.append(self)

        def connected(self):
            return self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

        async def send_email(self, *, to, subject, body, cc=None):
            self.sent.append(to)

    monkeypatch.setattr("app.integrations.mcp_client.GmailMCPClient", FakeGmailClient)

    result = await asyncio.to_thread(tasks.send_follow_up_emails.run, meeting_id, tenant_id)

    assert result == {"sent": 1, "total": 1}
    assert FakeGmailClient.instances[0].sent == ["a@acme.in"]

    async with session_scope() as session:
        rows = await FollowUpEmailRepository(session).list_unsent(
            uuid.UUID(meeting_id), tenant_id=uuid.UUID(tenant_id)
        )
        assert rows == []  # marked sent → never re-sent on the next run


# ── Import-time sanity ───────────────────────────────────────────────────
def test_dead_letter_helpers_expose_expected_api():
    from app.workers import dead_letter

    assert dead_letter.DLQ_KEY == "vcall:dlq"
    assert dead_letter.REPLAY_BATCH == 10
    assert dead_letter.MAX_DLQ_SIZE == 10_000


def test_progress_channel_naming():
    from app.workers.progress import channel_for, snapshot_key

    assert channel_for("abc") == "vcall:progress:abc"
    assert snapshot_key("abc") == "vcall:progress:abc:snapshot"
