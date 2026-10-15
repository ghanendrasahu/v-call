"""DPDP Act / GDPR tests: PII redaction, secure deletion, retention sweep."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from app.core.compliance import (
    redact_pii,
    redact_segments,
    scrub_meeting_audio,
    secure_delete_file,
    should_scrub,
)


# ── PII redaction ────────────────────────────────────────────────────────
def test_redacts_email_addresses():
    out = redact_pii("ping priya.kumar@acme.in about the release")
    assert "priya.kumar@acme.in" not in out
    assert "[EMAIL]" in out


def test_redacts_indian_phone_numbers():
    for phone in ("+91 98765 43210", "9876543210", "0 98765-43210"):
        out = redact_pii(f"call me on {phone} tomorrow")
        assert "[PHONE]" in out, phone
        assert phone not in out


def test_redacts_aadhaar_but_not_longer_digit_runs():
    out = redact_pii("aadhaar 2345 5678 9012 on file")
    assert "[AADHAAR]" in out
    assert "2345" not in out

    # A 16-digit run must NOT be misread as a 12-digit Aadhaar prefix.
    digits = "9999999999999999"  # fails Luhn → not a card either
    assert redact_pii(f"ref {digits}") == f"ref {digits}"


def test_redacts_valid_card_numbers_only():
    valid = "4111 1111 1111 1111"  # Luhn-valid test PAN
    assert redact_pii(f"card {valid}") == "card [CARD]"

    invalid = "1234 5678 9012 3457"  # not Luhn-valid
    assert "[CARD]" not in redact_pii(f"card {invalid}")


def test_card_wins_over_aadhaar_for_grouped_16_digit_pan():
    """16-digit card in 4-4-4-4 groups must not be clipped into a fake Aadhaar."""
    out = redact_pii("paid via 4111 1111 1111 1111 today")
    assert out == "paid via [CARD] today"


def test_redacts_pan_and_ip():
    out = redact_pii("PAN ABCDE1234F served from 192.168.10.44")
    assert "[PAN]" in out and "ABCDE1234F" not in out
    assert "[IP]" in out and "192.168.10.44" not in out


def test_redact_kinds_filter():
    text = "mail a@b.co from 10.0.0.1"
    only_email = redact_pii(text, kinds={"EMAIL"})

    assert only_email == "mail [EMAIL] from 10.0.0.1"


def test_redact_empty_text_passthrough():
    assert redact_pii("") == ""


def test_redact_segments_preserves_structure():
    segments = [
        {"speaker_label": "Speaker 00", "start_ms": 0, "end_ms": 10, "text": "mail a@b.co"},
        {
            "speaker_label": "Speaker 01",
            "start_ms": 10,
            "end_ms": 20,
            "text": "ok",
            "confidence": 0.9,
        },
    ]
    [first, second] = redact_segments(segments)

    assert first["text"] == "mail [EMAIL]"
    assert first["speaker_label"] == "Speaker 00"  # untouched fields survive
    assert second == segments[1]
    # Originals are not mutated.
    assert segments[0]["text"] == "mail a@b.co"


# ── Secure deletion ──────────────────────────────────────────────────────
def test_secure_delete_overwrites_and_unlinks(tmp_path: Path):
    target = tmp_path / "raw.wav"
    target.write_bytes(b"\xaa" * 4096)

    result = secure_delete_file(target)

    assert result.removed is True
    assert result.bytes_overwritten == 4096
    assert not target.exists()


def test_secure_delete_without_overwrite(tmp_path: Path):
    target = tmp_path / "raw.wav"
    target.write_bytes(b"data")

    result = secure_delete_file(target, overwrite=False)

    assert result.removed is True
    assert result.bytes_overwritten == 0
    assert not target.exists()


def test_secure_delete_missing_file_is_noop(tmp_path: Path):
    result = secure_delete_file(tmp_path / "ghost.wav")
    assert result.removed is False
    assert result.bytes_overwritten == 0


# ── Retention window ─────────────────────────────────────────────────────
def test_should_scrub_respects_retention_window(settings):
    now = datetime.now(UTC).timestamp()
    retention = settings.raw_audio_retention_hours * 3600

    assert should_scrub(now, now_epoch=now) is False
    assert should_scrub(now - retention + 60, now_epoch=now) is False
    assert should_scrub(now - retention - 60, now_epoch=now) is True


# ── End-to-end scrub job ─────────────────────────────────────────────────
async def test_scrub_meeting_audio_erases_file_and_pointer(tmp_path, seed_tenant_user):
    from app.db.repositories import MeetingRepository
    from app.db.session import get_session_factory, session_scope

    tenant, _, _ = seed_tenant_user
    audio = tmp_path / "rec.wav"
    audio.write_bytes(b"RIFF....WAVEaudio")

    async with get_session_factory()() as session:
        repo = MeetingRepository(session)
        meeting = await repo.create(
            tenant_id=tenant.id, title="To be erased", storage_path=str(audio)
        )
        await session.commit()
        meeting_id, tenant_id = meeting.id, tenant.id

    result = await scrub_meeting_audio(str(meeting_id), str(tenant_id), storage_path=str(audio))

    assert result is not None and result.removed is True
    assert not audio.exists()

    async with session_scope() as session:
        repo = MeetingRepository(session)
        row = await repo.get(meeting_id, tenant_id=tenant_id)
        assert row is not None
        assert row.storage_path is None  # no code path can resurrect the file


async def test_scrub_meeting_audio_noop_without_storage(seed_tenant_user):
    tenant, _, _ = seed_tenant_user
    assert await scrub_meeting_audio(str(tenant.id), str(tenant.id), storage_path=None) is None


# The async delete helper is exercised indirectly by scrub_meeting_audio;
# assert the offloading wrapper itself works for a plain file.
async def test_async_secure_delete_offloads(tmp_path: Path):
    from app.core.compliance import async_secure_delete

    target = tmp_path / "off.wav"
    target.write_bytes(b"z" * 10)

    result = await async_secure_delete(target)
    assert result.removed is True
    assert not target.exists()
