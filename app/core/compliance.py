"""DPDP Act 2023 / GDPR compliance: PII redaction + raw-audio scrubbing.

Two guarantees the platform makes to every tenant:

1. **Data minimisation in transcripts** — when enabled, direct identifiers
   (emails, phone numbers, Aadhaar, PAN, card numbers, IPs) are redacted
   from transcript text *before* it is fed to LLMs or stored for analytics.
2. **Right to erasure / retention limits** — raw audio is securely deleted
   (overwritten, then unlinked) once it exceeds the configured retention
   window, or on demand per meeting. The DB pointer is cleared so no code
   path can resurrect it.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import secrets
from dataclasses import dataclass
from pathlib import Path

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# ── PII patterns ────────────────────────────────────────────────────────
# Order matters: more specific patterns first (Aadhaar before generic digit
# runs, PAN before word-like tokens).
PII_PATTERNS: dict[str, re.Pattern[str]] = {
    # Payment card numbers (13-19 digits, spaces/dashes allowed) + Luhn check.
    # The final digit is matched separately so a trailing separator isn't
    # consumed. Runs BEFORE Aadhaar so a 16-digit card in 4-4-4-4 groups
    # isn't clipped into a fake 12-digit Aadhaar match.
    "CARD": re.compile(r"\b(?:\d[ -]?){12,18}\d\b"),
    # Indian Aadhaar: 12 digits grouped 4-4-4; the lookahead refuses to match
    # a 12-digit prefix of a longer digit run (e.g. cards).
    "AADHAAR": re.compile(r"\b\d{4}(?:[\s-]?\d{4}){2}(?![\s-]?\d)\b"),
    # Indian PAN: 5 letters + 4 digits + letter.
    "PAN": re.compile(r"\b[A-Za-z]{5}[0-9]{4}[A-Za-z]\b"),
    # Email addresses.
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    # IPv4 addresses.
    "IP": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    # Phone numbers: +91/0 national prefixes, 10 digits starting 6-9 with
    # optional single-space/dash grouping anywhere.
    "PHONE": re.compile(r"\b(?:\+?(?:91|0)[\s-]?)?[6-9]\d(?:[\s-]?\d){8}\b"),
}

_REDACTION_TEMPLATE = "[{kind}]"


def _luhn_ok(candidate: str) -> bool:
    """Cheap Luhn check to avoid redacting random digit runs as card numbers."""
    digits = [int(c) for c in candidate if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    checksum = 0
    parity = len(digits) % 2
    for i, digit in enumerate(digits):
        if i % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def redact_pii(text: str, *, kinds: set[str] | None = None) -> str:
    """Replace direct identifiers with ``[KIND]`` placeholders.

    ``kinds`` limits redaction to specific categories (default: all).
    """
    if not text:
        return text

    def _card_sub(match: re.Match[str]) -> str:
        candidate = match.group(0)
        # Only redact when the Luhn checksum passes (fewer false positives).
        return _REDACTION_TEMPLATE.format(kind="CARD") if _luhn_ok(candidate) else candidate

    result = text
    for kind, pattern in PII_PATTERNS.items():
        if kinds is not None and kind not in kinds:
            continue
        if kind == "CARD":
            result = pattern.sub(_card_sub, result)
        else:
            result = pattern.sub(_REDACTION_TEMPLATE.format(kind=kind), result)
    return result


def redact_segments(segments: list[dict], *, kinds: set[str] | None = None) -> list[dict]:
    """Redact ``text`` in every transcript segment dict."""
    return [
        {**segment, "text": redact_pii(str(segment.get("text", "")), kinds=kinds)}
        for segment in segments
    ]


# ── Raw audio scrubbing ─────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class ScrubResult:
    """Outcome of one scrub operation."""

    path: str
    removed: bool
    bytes_overwritten: int


def secure_delete_file(path: str | Path, *, overwrite: bool = True) -> ScrubResult:
    """Best-effort secure delete: overwrite with random bytes, then unlink.

    Note: on CoW/SSD/journaled filesystems physical erasure isn't guaranteed
    by overwrite alone — pair with at-rest encryption (Fernet here) so the
    crypto-shredding of the key is the authoritative erasure mechanism.
    """
    target = Path(path)
    if not target.exists():
        return ScrubResult(path=str(target), removed=False, bytes_overwritten=0)

    overwritten = 0
    if overwrite and target.is_file():
        size = target.stat().st_size
        try:
            with target.open("r+b") as handle:
                # Single pass of cryptographically-random noise.
                chunk = secrets.token_bytes(min(size, 1024 * 1024))
                while overwritten < size:
                    n = min(len(chunk), size - overwritten)
                    handle.write(chunk[:n])
                    overwritten += n
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:  # best-effort; still unlink below
            logger.warning("Overwrite failed for %s: %s", target, exc)

    target.unlink(missing_ok=True)
    return ScrubResult(path=str(target), removed=True, bytes_overwritten=overwritten)


async def async_secure_delete(path: str | Path, *, overwrite: bool = True) -> ScrubResult:
    """Thread-offloaded secure delete (never block the event loop)."""
    return await asyncio.to_thread(secure_delete_file, path, overwrite=overwrite)


async def scrub_meeting_audio(
    meeting_id: str,
    tenant_id: str,
    *,
    storage_path: str | None,
) -> ScrubResult | None:
    """Compliance job: erase a meeting's raw audio + clear the DB pointer."""
    if not storage_path:
        return None

    result = await async_secure_delete(storage_path)

    import uuid as _uuid

    from app.db.repositories import MeetingRepository
    from app.db.session import session_scope

    async with session_scope() as session:
        repo = MeetingRepository(session)
        await repo.clear_storage_path(_uuid.UUID(meeting_id), tenant_id=_uuid.UUID(tenant_id))

    logger.info(
        "Scrubbed audio for meeting %s (removed=%s, bytes=%s)",
        meeting_id,
        result.removed,
        result.bytes_overwritten,
    )
    return result


def should_scrub(created_at_epoch: float, *, now_epoch: float | None = None) -> bool:
    """True when a recording has passed the retention window."""
    import time

    now = now_epoch if now_epoch is not None else time.time()
    retention_seconds = get_settings().raw_audio_retention_hours * 3600
    return (now - created_at_epoch) >= retention_seconds
