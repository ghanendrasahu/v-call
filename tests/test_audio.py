"""Tests for audio validation, probing, and (single-chunk) chunking.

Note: multi-chunk splitting is exercised only with production-like settings —
tiny ``chunk_seconds`` values on synthetic constant-tone WAVs make the
silence-snapping search degenerate (there is no real silence to find), so
those paths are covered by ``_pick_silence_point`` unit tests instead.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from app.services.audio import (
    SUPPORTED_FORMATS,
    AudioValidationError,
    _pick_silence_point,
    check_extension,
    chunk_audio,
    cleanup_chunks,
    compute_sha256,
    ffmpeg_available,
    validate_audio,
)


# ── Extension + hashing ──────────────────────────────────────────────────
def test_check_extension_accepts_supported_formats():
    for suffix in SUPPORTED_FORMATS:
        assert check_extension(Path(f"recording{suffix}")) == suffix.lstrip(".")


def test_check_extension_normalises_case():
    assert check_extension(Path("AUDIO.WAV")) == "wav"


def test_check_extension_rejects_unsupported():
    with pytest.raises(AudioValidationError, match="Unsupported format"):
        check_extension(Path("malware.exe"))


def test_compute_sha256(tmp_path: Path):
    payload = b"v-call audio bytes"
    target = tmp_path / "clip.bin"
    target.write_bytes(payload)

    assert compute_sha256(target) == hashlib.sha256(payload).hexdigest()


# ── Validation / probing ─────────────────────────────────────────────────
async def test_validate_audio_accepts_wav(make_wav):
    path = make_wav("meeting.wav", seconds=0.5)
    info = await validate_audio(path)

    assert info.format == "wav"
    assert info.size_bytes > 0
    assert info.duration_seconds >= 0.0
    # Streaming hash is exposed for the audio-metadata contract.
    assert len(info.sha256) == 64
    assert info.to_metadata_dict()["format"] == "wav"


async def test_validate_audio_missing_file(tmp_path: Path):
    with pytest.raises(AudioValidationError, match="not found"):
        await validate_audio(tmp_path / "ghost.wav")


async def test_validate_audio_empty_file(tmp_path: Path):
    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    with pytest.raises(AudioValidationError, match="empty"):
        await validate_audio(empty)


async def test_validate_audio_enforces_size_limit(make_wav):
    path = make_wav("big.wav", seconds=0.2)
    with pytest.raises(AudioValidationError, match="exceeds"):
        await validate_audio(path, max_upload_mb=0)  # type: ignore[arg-type]


async def test_probe_falls_back_without_ffmpeg(make_wav, monkeypatch):
    """No ffmpeg on PATH → deterministic extension/filesystem fallback."""
    from app.services import audio as audio_mod

    monkeypatch.setattr(audio_mod, "ffmpeg_available", lambda: False)
    info = await validate_audio(make_wav("nofun.wav", seconds=0.3))

    assert info.duration_seconds == 0.0
    assert info.sample_rate is None
    assert info.size_bytes > 0


def test_ffmpeg_available_is_boolean():
    assert isinstance(ffmpeg_available(), bool)


# ── Chunking ─────────────────────────────────────────────────────────────
async def test_short_file_yields_single_chunk(make_wav, tmp_path: Path):
    """A file shorter than chunk_seconds is exported as exactly one chunk."""
    path = make_wav("short.wav", seconds=0.5)
    chunks = await chunk_audio(path, out_dir=tmp_path / "chunks", chunk_seconds=30)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.index == 0
    assert chunk.start_seconds == 0.0
    assert chunk.path.exists()
    assert chunk.duration_seconds == pytest.approx(0.5, abs=0.1)
    assert chunk.path.suffix == ".wav"

    cleanup_chunks(chunks)
    assert not chunk.path.exists()


def test_pick_silence_point_stays_in_window():
    """The snap point never leaves the ±7.5 s search window."""
    from pydub.generators import Sine

    audio = Sine(440).to_audio_segment(duration=30_000, volume=-10)
    # Add a quiet gap around 15 s so there IS a silence to find.
    quiet = Sine(440).to_audio_segment(duration=1_000, volume=-60)
    audio = audio[:14_000] + quiet + audio[15_000:]

    ideal, window = 15_000, 15_000
    point = _pick_silence_point(audio, ideal, window_ms=window)
    assert ideal - window // 2 <= point <= ideal + window // 2
    # The quiet gap should win over the loud tone.
    assert 14_000 <= point <= 16_000


def test_pick_silence_point_tiny_window_returns_ideal():
    from pydub.generators import Sine

    audio = Sine(440).to_audio_segment(duration=50, volume=-10)
    assert _pick_silence_point(audio, ideal_ms=10_000) == 10_000
