"""Standardized transcript schemas.

The transcription engine always emits ``TranscriptResult`` — a strict,
version-stable JSON contract consumed by the LangGraph agents and persisted to
the database. Downstream code never touches Whisper/pyannote internals.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class AudioMetadata(BaseModel):
    """Validated audio facts (probe output + integrity hash)."""

    model_config = ConfigDict(extra="forbid")

    format: str = Field(description="Container/codec family, e.g. 'mp4', 'wav'")
    duration_seconds: float = Field(ge=0)
    size_bytes: int = Field(ge=0)
    sample_rate: int | None = Field(default=None, ge=1)
    channels: int | None = Field(default=None, ge=1)
    codec: str | None = None
    sha256: str | None = Field(default=None, description="Integrity hash of the source file")


class Utterance(BaseModel):
    """One diarized utterance window."""

    model_config = ConfigDict(extra="forbid", strict=True)

    speaker_label: str = Field(description="Stable label, e.g. 'Speaker 00'")
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    text: str = Field(min_length=1)
    confidence: float | None = Field(default=None, ge=0, le=1)


class TranscriptResult(BaseModel):
    """Full pipeline output — the single contract handed to the agents."""

    model_config = ConfigDict(extra="forbid")

    language: str | None = None
    duration_seconds: float = Field(ge=0)
    raw_transcript: str = Field(description="Plain-text transcript with speaker prefixes")
    utterances: list[Utterance]
    audio_metadata: AudioMetadata
    speaker_count: int = Field(ge=0)
    models: dict[str, str] = Field(
        default_factory=dict,
        description="Model identifiers used, e.g. {'asr': 'small', 'diarization': 'pyannote-3.1'}",
    )

    @property
    def speaker_labels(self) -> list[str]:
        """Distinct speakers in first-appearance order."""
        seen: dict[str, None] = {}
        for utterance in self.utterances:
            seen.setdefault(utterance.speaker_label, None)
        return list(seen)


class TranscriptionProgress(BaseModel):
    """Progress event streamed to workers/WebSocket clients."""

    model_config = ConfigDict(extra="forbid")

    stage: str = Field(description="validating | chunking | transcribing | diarizing | merging")
    progress: float = Field(ge=0, le=1, description="0.0 → 1.0")
    message: str | None = None
