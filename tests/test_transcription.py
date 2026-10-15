"""Tests for the transcription pipeline (fake ASR/diarization backends).

Heavy ML backends are injected behind the ``ASRBackend``/``Diarizer``
protocols so the orchestration (chunking, merging, PII redaction, cleanup,
progress emission) is verified offline.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.schemas.transcript import Utterance
from app.services.transcription import (
    ASROutput,
    DiarizationError,
    RawSegment,
    SpeakerTurn,
    TranscriptionError,
    TranscriptionService,
    apply_offset,
    build_raw_transcript,
    merge_speakers,
    normalize_speaker_label,
)


# ── Fake backends ────────────────────────────────────────────────────────
class FakeASR:
    name = "fake-asr/mini"

    def __init__(self, output: ASROutput | Exception) -> None:
        self._output = output
        self.calls: list[Path] = []

    def transcribe(self, path: Path, *, language: str | None = None) -> ASROutput:
        self.calls.append(path)
        if isinstance(self._output, Exception):
            raise self._output
        return self._output


class FakeDiarizer:
    name = "fake-diara/1"

    def __init__(self, turns: list[SpeakerTurn] | Exception) -> None:
        self._turns = turns

    def diarize(self, path: Path) -> list[SpeakerTurn]:
        if isinstance(self._turns, Exception):
            raise self._turns
        return self._turns


def _service(asr: FakeASR, diarizer: FakeDiarizer | None = None) -> TranscriptionService:
    return TranscriptionService(asr=asr, diarizer=diarizer or FakeDiarizer([]))


def _leftovers(settings, pattern: str) -> list[Path]:
    """Sync artifact probe (blocking Path I/O is illegal inside async tests)."""
    return list(Path(settings.audio_artifacts_dir).glob(pattern))


# ── Pure helpers ─────────────────────────────────────────────────────────
def test_normalize_speaker_label():
    assert normalize_speaker_label("SPEAKER_00") == "Speaker 00"
    assert normalize_speaker_label("speaker_3") == "Speaker 03"
    assert normalize_speaker_label("Speaker 12") == "Speaker 12"
    assert normalize_speaker_label("Host") == "Host"


def test_apply_offset_rebases_timestamps():
    segments = [RawSegment(start=1.0, end=2.0, text="hi", confidence=0.9)]
    [shifted] = apply_offset(segments, offset_seconds=60.0)

    assert shifted.start == 61.0
    assert shifted.end == 62.0
    assert shifted.text == "hi"
    assert shifted.confidence == 0.9


def test_merge_speakers_picks_dominant_overlap():
    segments = [
        RawSegment(start=0.0, end=2.0, text="hello", confidence=0.8),
        RawSegment(start=3.0, end=5.0, text="hi there", confidence=None),
        RawSegment(start=100.0, end=101.0, text="orphan", confidence=2.5),
    ]
    turns = [
        SpeakerTurn(start=0.0, end=2.0, label="SPEAKER_00"),
        SpeakerTurn(start=2.5, end=6.0, label="SPEAKER_01"),
    ]

    utterances = merge_speakers(segments, turns)

    assert [u.speaker_label for u in utterances] == ["Speaker 00", "Speaker 01", "Speaker 00"]
    # No overlap → best-effort Speaker 00; confidence clamped to [0, 1].
    assert utterances[2].confidence == 1.0
    assert utterances[1].confidence is None
    assert utterances[0].start_ms == 0 and utterances[0].end_ms == 2000


def test_merge_speakers_sorts_chronologically():
    segments = [
        RawSegment(start=5.0, end=6.0, text="second"),
        RawSegment(start=1.0, end=2.0, text="first"),
    ]
    utterances = merge_speakers(segments, [])
    assert [u.text for u in utterances] == ["first", "second"]


def test_build_raw_transcript():
    utterances = [
        Utterance(speaker_label="Speaker 00", start_ms=0, end_ms=100, text="one"),
        Utterance(speaker_label="Speaker 01", start_ms=100, end_ms=200, text="two"),
    ]
    assert build_raw_transcript(utterances) == "Speaker 00: one\nSpeaker 01: two"


# ── Pipeline ─────────────────────────────────────────────────────────────
async def test_pipeline_full_run_with_fake_backends(make_wav, settings):
    asr = FakeASR(
        ASROutput(
            segments=[
                RawSegment(start=0.0, end=0.4, text="ping me at priya@acme.in", confidence=0.95)
            ],
            language="en",
        )
    )
    diarizer = FakeDiarizer([SpeakerTurn(start=0.0, end=0.5, label="SPEAKER_00")])
    stages: list[tuple[str, float]] = []

    async def on_progress(event):
        stages.append((event.stage, event.progress))

    path = make_wav("call.wav", seconds=0.5)
    result = await _service(asr, diarizer).transcribe_file(path, on_progress=on_progress)

    # Progress is monotone and covers the full pipeline.
    assert stages[0][0] == "validating"
    assert {s for s, _ in stages} == {
        "validating",
        "chunking",
        "transcribing",
        "diarizing",
        "merging",
    }
    assert stages[-1] == ("merging", 1.0)
    progresses = [p for _, p in stages]
    assert progresses == sorted(progresses)

    # Contract fields.
    assert result.language == "en"
    assert result.speaker_count == 1
    assert result.models["asr"] == "fake-asr/mini"
    assert result.models["diarization"] == "fake-diara/1"
    assert result.raw_transcript.startswith("Speaker 00:")
    assert len(result.utterances) == 1

    # DPDP: direct identifiers are redacted before storage/LLM use.
    assert "priya@acme.in" not in result.utterances[0].text
    assert "[EMAIL]" in result.utterances[0].text

    # Intermediate chunks are always cleaned up.
    assert _leftovers(settings, "call_*") == []


async def test_pipeline_degrades_without_diarization(make_wav):
    asr = FakeASR(
        ASROutput(segments=[RawSegment(start=0.0, end=0.4, text="solo speaker")], language="en")
    )
    diarizer = FakeDiarizer(DiarizationError("pyannote unavailable"))

    result = await _service(asr, diarizer).transcribe_file(make_wav("degraded.wav", seconds=0.4))

    assert result.models["diarization"] == "unavailable"
    assert result.speaker_count == 1
    assert result.utterances[0].speaker_label == "Speaker 00"
    assert "solo speaker" in result.raw_transcript


async def test_pipeline_raises_when_asr_produces_nothing(make_wav, settings):
    asr = FakeASR(ASROutput(segments=[], language=None))
    path = make_wav("silent.wav", seconds=0.3)

    with pytest.raises(TranscriptionError, match="no text"):
        await _service(asr).transcribe_file(path)

    # Chunks are cleaned even on failure (compliance: no stray audio).
    assert _leftovers(settings, "silent_*") == []


async def test_pipeline_propagates_asr_failure(make_wav):
    asr = FakeASR(RuntimeError("model exploded"))

    with pytest.raises(RuntimeError, match="model exploded"):
        await _service(asr).transcribe_file(make_wav("boom.wav", seconds=0.3))


async def test_pipeline_accepts_explicit_language(make_wav):
    asr = FakeASR(
        ASROutput(segments=[RawSegment(start=0.0, end=0.3, text="नमस्ते दोस्तों")], language=None)
    )
    result = await _service(asr).transcribe_file(make_wav("hindi.wav", seconds=0.3), language="hi")
    # Caller-specified language wins over ASR detection (which is None here).
    assert result.language == "hi"
    assert "नमस्ते" in result.raw_transcript
