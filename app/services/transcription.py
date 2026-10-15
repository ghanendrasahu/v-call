"""High-throughput transcription pipeline.

Faster-Whisper (ASR) + pyannote.audio (diarization) wrapped behind small
protocol interfaces so the orchestration logic is unit-testable without the
heavy torch stack, and so backends can be swapped (cloud ASR) later.

Design notes
------------
- Heavy ML imports happen **lazily** inside the backend classes: importing
  this module never requires torch to be installed.
- All CPU-bound work runs in worker threads (``asyncio.to_thread``) so the
  event loop stays free for API/WebSocket traffic.
- Files above 100 MB are silence-aware chunked (with overlap) before ASR;
  chunk timestamps are re-based onto the original timeline.
- Diarization is best-effort: if it fails, the pipeline degrades to a single
  ``Speaker 00`` instead of failing the whole meeting.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from app.core.config import Settings, get_settings
from app.schemas.transcript import AudioMetadata, TranscriptionProgress, TranscriptResult, Utterance
from app.services.audio import (
    CHUNK_THRESHOLD_BYTES,
    AudioChunk,
    AudioInfo,
    chunk_audio,
    cleanup_chunks,
    validate_audio,
)

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[TranscriptionProgress], Awaitable[None]]


class TranscriptionError(Exception):
    """Fatal ASR failure."""


class DiarizationError(Exception):
    """Non-fatal diarization failure (pipeline falls back to one speaker)."""


# ── Backend contracts ───────────────────────────────────────────────────
@dataclass(slots=True)
class RawSegment:
    """ASR output before diarization (seconds)."""

    start: float
    end: float
    text: str
    confidence: float | None = None


@dataclass(slots=True)
class ASROutput:
    """Complete ASR result for one file/chunk."""

    segments: list[RawSegment]
    language: str | None = None
    duration_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class SpeakerTurn:
    """One diarized speaker window (seconds)."""

    start: float
    end: float
    label: str


class ASRBackend(Protocol):
    """Anything that can transcribe an audio file."""

    name: str

    def transcribe(self, path: Path, *, language: str | None = None) -> ASROutput: ...


class Diarizer(Protocol):
    """Anything that can produce speaker turns for an audio file."""

    name: str

    def diarize(self, path: Path) -> Sequence[SpeakerTurn]: ...


# ── Concrete backends (lazy heavy imports) ──────────────────────────────
class FasterWhisperBackend:
    """Local ASR via ``faster-whisper`` (CTranslate2)."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._model = None
        self.name = f"faster-whisper/{self._settings.whisper_model}"

    def _load(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:  # pragma: no cover - exercised in prod images
                raise TranscriptionError(
                    "faster-whisper is not installed. "
                    "Install with: pip install 'v-call[transcription]'"
                ) from exc
            self._model = WhisperModel(
                self._settings.whisper_model,
                device=self._settings.whisper_device,
                compute_type=self._settings.whisper_compute_type,
            )
        return self._model

    def transcribe(self, path: Path, *, language: str | None = None) -> ASROutput:
        model = self._load()
        segments_iter, info = model.transcribe(
            str(path),
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        segments = [
            RawSegment(
                start=float(seg.start),
                end=float(seg.end),
                text=seg.text.strip(),
                confidence=float(seg.avg_logprob) if getattr(seg, "avg_logprob", None) else None,
            )
            for seg in segments_iter
            if seg.text.strip()
        ]
        return ASROutput(
            segments=segments,
            language=getattr(info, "language", None),
            duration_seconds=float(info.duration) if getattr(info, "duration", None) else None,
        )


class PyannoteDiarizer:
    """Speaker diarization via ``pyannote.audio``."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._pipeline = None
        self.name = f"pyannote/{self._settings.diarization_model.rsplit('/', 1)[-1]}"

    def _load(self):
        if self._pipeline is None:
            try:
                from pyannote.audio import Pipeline
            except ImportError as exc:  # pragma: no cover - exercised in prod images
                raise DiarizationError(
                    "pyannote.audio is not installed. "
                    "Install with: pip install 'v-call[transcription]'"
                ) from exc
            token = (
                self._settings.pyannote_token.get_secret_value()
                if self._settings.pyannote_token
                else None
            )
            self._pipeline = Pipeline.from_pretrained(
                self._settings.diarization_model, use_auth_token=token
            )
        return self._pipeline

    def diarize(self, path: Path) -> Sequence[SpeakerTurn]:
        pipeline = self._load()
        annotation = pipeline(str(path))
        turns: list[SpeakerTurn] = []
        for turn, _, label in annotation.itertracks(yield_label=True):
            turns.append(
                SpeakerTurn(start=float(turn.start), end=float(turn.end), label=str(label))
            )
        return turns


# ── Pure helpers (unit-tested without models) ───────────────────────────
def normalize_speaker_label(label: str) -> str:
    """``SPEAKER_00`` / ``speaker_0`` → ``Speaker 00``."""
    stripped = label.strip()
    suffix = stripped.rsplit("_", 1)[-1] if "_" in stripped else stripped
    if suffix.isdigit():
        return f"Speaker {int(suffix):02d}"
    return stripped.replace("_", " ").title()


def apply_offset(segments: Sequence[RawSegment], offset_seconds: float) -> list[RawSegment]:
    """Re-base chunk-local timestamps onto the full-recording timeline."""
    return [
        RawSegment(
            start=seg.start + offset_seconds,
            end=seg.end + offset_seconds,
            text=seg.text,
            confidence=seg.confidence,
        )
        for seg in segments
    ]


def merge_speakers(
    segments: Sequence[RawSegment],
    turns: Sequence[SpeakerTurn],
) -> list[Utterance]:
    """Attach the dominant diarized speaker to each ASR segment.

    For every segment we pick the speaker turn with the largest temporal
    overlap. Segments with no overlap keep ``Speaker 00`` (best-effort).
    """
    utterances: list[Utterance] = []
    for seg in segments:
        best_label, best_overlap = "Speaker 00", 0.0
        for turn in turns:
            overlap = min(seg.end, turn.end) - max(seg.start, turn.start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = normalize_speaker_label(turn.label)
        utterances.append(
            Utterance(
                speaker_label=best_label,
                start_ms=max(0, round(seg.start * 1000)),
                end_ms=max(0, round(seg.end * 1000)),
                text=seg.text,
                confidence=max(0.0, min(1.0, seg.confidence))
                if seg.confidence is not None
                else None,
            )
        )
    utterances.sort(key=lambda u: (u.start_ms, u.end_ms))
    return utterances


def build_raw_transcript(utterances: Sequence[Utterance]) -> str:
    """Human-readable transcript: ``Speaker 00: hello`` per line."""
    return "\n".join(f"{u.speaker_label}: {u.text}" for u in utterances)


# ── Orchestration ───────────────────────────────────────────────────────
@dataclass(slots=True)
class TranscriptionService:
    """Validate → (chunk) → transcribe → diarize → merge → TranscriptResult."""

    settings: Settings = field(default_factory=get_settings)
    asr: ASRBackend | None = None
    diarizer: Diarizer | None = None
    _asr: ASRBackend | None = field(init=False, default=None)
    _diarizer: Diarizer | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        # Backends are created lazily so importing/constructing the service
        # never forces a model download.
        if self.asr is not None:
            self._asr = self.asr
        if self.diarizer is not None:
            self._diarizer = self.diarizer

    @property
    def asr_backend(self) -> ASRBackend:
        if self._asr is None:
            self._asr = FasterWhisperBackend(self.settings)
        return self._asr

    @property
    def diarizer_backend(self) -> Diarizer:
        if self._diarizer is None:
            self._diarizer = PyannoteDiarizer(self.settings)
        return self._diarizer

    async def transcribe_file(
        self,
        path: str | Path,
        *,
        language: str | None = None,
        on_progress: ProgressCallback | None = None,
        force_chunk: bool = False,
    ) -> TranscriptResult:
        """Run the full pipeline over one audio file."""
        path = Path(path)
        chunks: list[AudioChunk] = []

        async def emit(stage: str, progress: float, message: str | None = None) -> None:
            if on_progress is not None:
                await on_progress(
                    TranscriptionProgress(stage=stage, progress=progress, message=message)
                )

        try:
            # 1) Validate with ffprobe.
            await emit("validating", 0.0, "Probing audio stream")
            info: AudioInfo = await validate_audio(path)

            # 2) Chunk if the file is large (or explicitly forced).
            await emit("chunking", 0.05, "Preparing audio chunks")
            force = force_chunk or info.size_bytes > CHUNK_THRESHOLD_BYTES
            chunks = (
                await chunk_audio(path)
                if force
                else await chunk_audio(path, chunk_seconds=max(int(info.duration_seconds) + 1, 30))
            )

            # 3) Transcribe each chunk sequentially (GPU/CPU friendly) and
            #    re-base timestamps onto the original timeline.
            raw_segments: list[RawSegment] = []
            detected_language = language
            for i, chunk in enumerate(chunks):
                await emit(
                    "transcribing",
                    0.1 + 0.6 * (i / max(len(chunks), 1)),
                    f"Transcribing chunk {i + 1}/{len(chunks)}",
                )
                output = await asyncio.to_thread(
                    self.asr_backend.transcribe, chunk.path, language=language
                )
                raw_segments.extend(apply_offset(output.segments, chunk.start_seconds))
                if detected_language is None:
                    detected_language = output.language

            if not raw_segments:
                raise TranscriptionError(
                    "ASR produced no text — the recording may be silent or corrupt"
                )

            # 4) Diarize the original file once (best-effort).
            turns: Sequence[SpeakerTurn] = ()
            await emit("diarizing", 0.75, "Identifying speakers")
            try:
                turns = await asyncio.to_thread(self.diarizer_backend.diarize, path)
            except Exception as exc:
                logger.warning("Diarization failed for %s: %s", path.name, exc)
                await emit("diarizing", 0.85, f"Diarization unavailable ({exc})")

            # 5) Merge into the stable transcript contract.
            await emit("merging", 0.95, "Building transcript")
            utterances = merge_speakers(raw_segments, turns)
            await emit("merging", 1.0, "Done")

            return TranscriptResult(
                language=detected_language,
                duration_seconds=info.duration_seconds,
                raw_transcript=build_raw_transcript(utterances),
                utterances=utterances,
                audio_metadata=AudioMetadata(**info.to_metadata_dict()),
                speaker_count=len({u.speaker_label for u in utterances}),
                models={
                    "asr": self.asr_backend.name,
                    "diarization": (self.diarizer_backend.name if turns else "unavailable"),
                },
            )
        finally:
            # 6) Always clean up intermediate chunks (compliance: no stray audio).
            if chunks:
                await asyncio.to_thread(cleanup_chunks, chunks)
