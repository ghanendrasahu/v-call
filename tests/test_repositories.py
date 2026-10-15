"""Repository tests: tenant scoping, lifecycle stamps, encrypt-on-write."""

from __future__ import annotations

import uuid

from app.core.security import hash_password
from app.db.enums import (
    ActionItemPriority,
    ActionItemStatus,
    IntegrationProvider,
    MeetingSource,
    MeetingStatus,
    TargetSystem,
    UserRole,
)
from app.db.models import Tenant
from app.db.repositories import (
    ActionItemRepository,
    FollowUpEmailRepository,
    IntegrationCredentialRepository,
    MeetingRepository,
    TenantRepository,
    UserRepository,
)
from app.db.session import get_session_factory


# ── Helpers ──────────────────────────────────────────────────────────────
async def _make_tenant(slug: str) -> Tenant:
    tenant = Tenant(name=slug.title(), slug=slug)
    async with get_session_factory()() as session:
        session.add(tenant)
        await session.flush()
        await session.commit()
    return tenant


async def _two_tenants() -> tuple[Tenant, Tenant]:
    return await _make_tenant(f"acme-{uuid.uuid4().hex[:6]}"), await _make_tenant(
        f"globex-{uuid.uuid4().hex[:6]}"
    )


# ── Meeting repository ───────────────────────────────────────────────────
async def test_meeting_create_and_get_is_tenant_scoped():
    tenant_a, tenant_b = await _two_tenants()
    async with get_session_factory()() as session:
        repo = MeetingRepository(session)
        meeting = await repo.create(
            tenant_id=tenant_a.id,
            title="Sprint review",
            source=MeetingSource.GOOGLE_MEET,
            provider_meeting_id="meet-abc",
        )

        found = await repo.get_with_details(meeting.id, tenant_id=tenant_a.id)
        assert found is not None
        assert found.title == "Sprint review"
        assert found.status == MeetingStatus.PENDING

        # Another tenant must not see it (row-level isolation).
        assert await repo.get_with_details(meeting.id, tenant_id=tenant_b.id) is None
        assert await repo.get(meeting.id, tenant_id=tenant_b.id) is None
        assert [m.id for m in await repo.list(tenant_id=tenant_b.id)] == []
        assert [m.id for m in await repo.list(tenant_id=tenant_a.id)] == [meeting.id]


async def test_meeting_status_lifecycle_stamps():
    tenant, _ = await _two_tenants()
    async with get_session_factory()() as session:
        repo = MeetingRepository(session)
        meeting = await repo.create(tenant_id=tenant.id, title="Lifecycle")

        await repo.update_status(meeting, MeetingStatus.TRANSCRIBING)
        assert meeting.started_at is not None
        assert meeting.completed_at is None

        await repo.update_status(
            meeting,
            MeetingStatus.ANALYZING,
            language="en",
            duration_seconds=123.5,
            audio_metadata={"format": "wav"},
        )
        assert meeting.language == "en"
        assert meeting.duration_seconds == 123.5
        assert meeting.audio_metadata == {"format": "wav"}
        assert meeting.completed_at is None  # only terminal states stamp it

        await repo.update_status(meeting, MeetingStatus.COMPLETED)
        assert meeting.completed_at is not None

        # Re-entering TRANSCRIBING (reprocess) keeps the first started_at.
        await repo.update_status(meeting, MeetingStatus.TRANSCRIBING)
        assert meeting.started_at is not None


async def test_segments_attach_list_and_cascade_delete():
    tenant, _ = await _two_tenants()
    async with get_session_factory()() as session:
        repo = MeetingRepository(session)
        meeting = await repo.create(tenant_id=tenant.id, title="Transcribed")

        await repo.attach_segments(
            meeting.id,
            tenant.id,
            [
                {"speaker_label": "Speaker 01", "start_ms": 1000, "end_ms": 2000, "text": "b"},
                {"speaker_label": "Speaker 00", "start_ms": 0, "end_ms": 900, "text": "a"},
            ],
        )
        segments = await repo.list_segments(meeting.id, tenant_id=tenant.id)
        assert [s.text for s in segments] == ["a", "b"]  # chronological order

        # Deleting the meeting purges transcript rows (no orphans).
        assert await repo.delete(meeting.id, tenant_id=tenant.id) is True
        assert await repo.list_segments(meeting.id, tenant_id=tenant.id) == []


async def test_clear_storage_path():
    tenant, _ = await _two_tenants()
    async with get_session_factory()() as session:
        repo = MeetingRepository(session)
        meeting = await repo.create(
            tenant_id=tenant.id, title="Has audio", storage_path="/tmp/x.wav"
        )

        assert await repo.clear_storage_path(meeting.id, tenant_id=tenant.id) is True
        assert meeting.storage_path is None
        assert await repo.clear_storage_path(uuid.uuid4(), tenant_id=tenant.id) is False


# ── Tenant + user repositories ───────────────────────────────────────────
async def test_tenant_repository_get_by_slug():
    slug = f"zenith-{uuid.uuid4().hex[:6]}"
    async with get_session_factory()() as session:
        tenants = TenantRepository(session)
        assert await tenants.get_by_slug(slug) is None

        created = await tenants.create(name="Zenith Labs", slug=slug)
        again = await tenants.get_by_slug(slug)
        assert again is not None and again.id == created.id


async def test_user_lookup_and_authentication():
    tenant, _ = await _two_tenants()
    async with get_session_factory()() as session:
        users = UserRepository(session)
        user = await users.create(
            tenant_id=tenant.id,
            email="  Priya@Acme.in ",  # normalised on write
            password_hash=hash_password("secret123"),
            role=UserRole.ADMIN,
        )
        assert user.email == "priya@acme.in"

        # Case-insensitive lookup.
        found = await users.get_by_email("PRIYA@ACME.IN")
        assert found is not None and found.id == user.id

        assert await users.authenticate(email="priya@acme.in", password="secret123") is user
        assert await users.authenticate(email="priya@acme.in", password="wrong-pass") is None
        assert await users.authenticate(email="ghost@acme.in", password="secret123") is None

        user.is_active = False
        await session.flush()
        assert await users.authenticate(email="priya@acme.in", password="secret123") is None


async def test_base_repository_count_is_tenant_scoped():
    tenant_a, tenant_b = await _two_tenants()
    async with get_session_factory()() as session:
        repo = MeetingRepository(session)
        await repo.create(tenant_id=tenant_a.id, title="one")
        await repo.create(tenant_id=tenant_a.id, title="two")
        await repo.create(tenant_id=tenant_b.id, title="other")

        assert await repo.count(tenant_id=tenant_a.id) == 2
        assert await repo.count(tenant_id=tenant_b.id) == 1


# ── Credential repository (encrypt-on-write) ─────────────────────────────
async def test_credentials_are_encrypted_at_rest():
    tenant, _ = await _two_tenants()
    secret = "zoho-oauth-refresh-token-XYZ"
    async with get_session_factory()() as session:
        repo = IntegrationCredentialRepository(session)
        credential = await repo.upsert(
            tenant_id=tenant.id,
            provider=IntegrationProvider.ZOHO_CRM,
            access_token=secret,
            refresh_token="refresh-123",
            scopes=["crm.read", "crm.write"],
        )

        # At rest: ciphertext, never plaintext.
        assert secret not in credential.access_token
        assert "refresh-123" not in (credential.refresh_token or "")

        decrypted = await repo.get_decrypted(tenant.id, IntegrationProvider.ZOHO_CRM)
        assert decrypted is not None
        assert decrypted.access_token == secret
        assert decrypted.refresh_token == "refresh-123"
        assert decrypted.provider == "zoho_crm"
        assert credential.last_used_at is not None  # bumped on read

        # Rotating updates the same row instead of inserting a second one.
        rotated = await repo.upsert(
            tenant_id=tenant.id,
            provider=IntegrationProvider.ZOHO_CRM,
            access_token="rotated-token",
        )
        assert rotated.id == credential.id
        rotated_row = await repo.get_decrypted(tenant.id, IntegrationProvider.ZOHO_CRM)
        assert rotated_row is not None
        assert rotated_row.access_token == "rotated-token"

        # The masked view never leaks plaintext.
        masked = repo.masked_view(credential)
        assert "rotated-token" not in str(masked)
        assert masked["provider"] == "zoho_crm"
        assert masked["has_refresh_token"] is False  # rotation dropped it

        assert await repo.revoke(tenant.id, IntegrationProvider.ZOHO_CRM) is True
        assert await repo.revoke(tenant.id, IntegrationProvider.ZOHO_CRM) is False
        assert await repo.get_for_provider(tenant.id, IntegrationProvider.ZOHO_CRM) is None


async def test_credentials_are_tenant_isolated():
    tenant_a, tenant_b = await _two_tenants()
    async with get_session_factory()() as session:
        repo = IntegrationCredentialRepository(session)
        await repo.upsert(
            tenant_id=tenant_a.id, provider=IntegrationProvider.JIRA, access_token="a-token"
        )

        assert await repo.get_for_provider(tenant_a.id, IntegrationProvider.JIRA) is not None
        assert await repo.get_for_provider(tenant_b.id, IntegrationProvider.JIRA) is None


# ── Action items ─────────────────────────────────────────────────────────
async def test_replace_for_meeting_is_idempotent():
    tenant, _ = await _two_tenants()
    async with get_session_factory()() as session:
        meetings = MeetingRepository(session)
        meeting = await meetings.create(tenant_id=tenant.id, title="With tasks")
        items = ActionItemRepository(session)
        payload = [
            {"task": "Do the thing", "priority": "high", "due_date": "2026-10-30"},
            {"task": "Do the other thing", "target_system": "jira"},
        ]

        await items.replace_for_meeting(meeting_id=meeting.id, tenant_id=tenant.id, items=payload)
        second = await items.replace_for_meeting(
            meeting_id=meeting.id, tenant_id=tenant.id, items=payload
        )

        stored = await items.list_for_meeting(meeting.id, tenant_id=tenant.id)
        assert len(stored) == len(second) == 2  # replaced, not duplicated
        assert {i.id for i in stored} == {i.id for i in second}
        high = next(i for i in stored if i.task == "Do the thing")
        assert high.due_date is not None
        assert high.due_date.isoformat() == "2026-10-30"
        assert high.priority == ActionItemPriority.HIGH


async def test_dispatch_status_transitions():
    tenant, _ = await _two_tenants()
    async with get_session_factory()() as session:
        meetings = MeetingRepository(session)
        meeting = await meetings.create(tenant_id=tenant.id, title="Dispatching")
        items = ActionItemRepository(session)
        [row] = await items.replace_for_meeting(
            meeting_id=meeting.id, tenant_id=tenant.id, items=[{"task": "file a ticket"}]
        )
        assert row.status == ActionItemStatus.OPEN

        await items.mark_dispatched(row.id, tenant_id=tenant.id, external_id="JIRA-5")
        assert row.status == ActionItemStatus.DISPATCHED
        assert row.external_id == "JIRA-5"
        assert row.target_system == TargetSystem.NONE

        await items.mark_status(row.id, tenant_id=tenant.id, status=ActionItemStatus.DONE)
        assert row.status == ActionItemStatus.DONE

        # Unknown ids are ignored rather than raising.
        missing = uuid.uuid4()
        assert await items.mark_dispatched(missing, tenant_id=tenant.id, external_id="x") is None


# ── Follow-up emails ─────────────────────────────────────────────────────
async def test_replace_preserves_sent_emails():
    tenant, _ = await _two_tenants()
    async with get_session_factory()() as session:
        meetings = MeetingRepository(session)
        meeting = await meetings.create(tenant_id=tenant.id, title="Emails")
        emails = FollowUpEmailRepository(session)

        [sent_row, draft_row] = await emails.replace_for_meeting(
            meeting_id=meeting.id,
            tenant_id=tenant.id,
            emails=[
                {"recipient_email": "a@acme.in", "email_body": "already sent"},
                {"recipient_email": "b@acme.in", "email_body": "still drafting"},
            ],
        )
        await emails.mark_sent(sent_row.id, tenant_id=tenant.id, mcp_message_id="msg-1")
        assert sent_row.sent is True and sent_row.mcp_message_id == "msg-1"

        # Re-analysis replaces only unsent drafts; sent mail is immutable.
        await emails.replace_for_meeting(
            meeting_id=meeting.id,
            tenant_id=tenant.id,
            emails=[{"recipient_email": "c@acme.in", "email_body": "new draft"}],
        )

        remaining = sorted(e.recipient_email for e in await emails.list(tenant_id=tenant.id))
        assert remaining == ["a@acme.in", "c@acme.in"]

        unsent = await emails.list_unsent(meeting.id, tenant_id=tenant.id)
        assert [e.recipient_email for e in unsent] == ["c@acme.in"]
        assert draft_row.id != sent_row.id  # sanity: distinct rows were created
