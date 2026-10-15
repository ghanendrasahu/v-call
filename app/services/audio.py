"""Audio validation, probing, and chunking.

All heavy work (ffprobe, pydub decode/encode) is synchronous and executed via
:mod:`asyncio` thread offloading so the event loop stays responsive.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

from pydub import AudioSegment
from pydub.effects import normalize

from app.core.config import get_settings

# Containers we accept from Meet/Zoom exports and direct uploads.
SUPPORTED_FORMATS: frozenset[str] = frozenset(
    {".mp4", ".m4a", ".mp3", ".wav", ".flac", ".ogg", ".webm"}
)

# Files above this size are chunked before ASR to bound memory.
CHUNK_THRESHOLD_BYTES = 100 * 1024 * 1024  # 100 MB

# Overlap between consecutive chunks so words straddling a boundary survive.
CHUNK_OVERLAP_SECONDS = 5.0


class AudioValidationError(Exception):
    """Raised when an audio file is unsupported, corrupt, or too large."""


@dataclass(frozen=True, slots=True)
class AudioInfo:
    """Result of a successful ffprobe validation."""

    path: Path
    format: str
    duration_seconds: float
    size_bytes: int
    sample_rate: int | None
    channels: int | None
    codec: str | None

    @property
    def sha256(self) -> str:
        return compute_sha256(self.path)

    def to_metadata_dict(self) -> dict:
        return {
            "format": self.format,
            "duration_seconds": self.duration_seconds,
            "size_bytes": self.size_bytes,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "codec": self.codec,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class AudioChunk:
    """A sliced piece of a large recording."""

    path: Path
    index: int
    start_seconds: float
    duration_seconds: float


def ffmpeg_available() -> bool:
    """True when ffprobe/ffmpeg are on PATH."""
    return shutil.which("ffprobe") is not None and shutil.which("ffmpeg") is not None


def compute_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    """Streaming SHA-256 so huge files never load into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def check_extension(path: Path) -> str:
    """Validate the extension against the allowlist; return it lowercased."""
    suffix = Path(path).suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise AudioValidationError(
            f"Unsupported format '{suffix}'. Supported: {sorted(SUPPORTED_FORMATS)}"
        )
    return suffix.lstrip(".")


def _probe_sync(path: Path) -> AudioInfo:
    """Blocking ffprobe call returning validated metadata."""
    import subprocess

    path = Path(path)
    if not path.exists():
        raise AudioValidationError(f"Audio file not found: {path}")
    if path.stat().st_size == 0:
        raise AudioValidationError("Audio file is empty")

    fmt = check_extension(path)

    if not ffmpeg_available():
        # Fall back to extension + filesystem facts when ffmpeg is absent
        # (dev machines without ffmpeg still get deterministic behaviour).
        size = path.stat().st_size
        return AudioInfo(
            path=path,
            format=fmt,
            duration_seconds=0.0,
            size_bytes=size,
            sample_rate=None,
            channels=None,
            codec=None,
        )

    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(path),
            ],
            capture_output=True,
            check=True,
            timeout=60,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise AudioValidationError(f"ffprobe failed to read '{path.name}': {exc}") from exc

    payload = json.loads(completed.stdout.decode())
    streams = payload.get("streams", [])
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if audio_stream is None:
        raise AudioValidationError(f"No audio stream found in '{path.name}'")

    fmt_info = payload.get("format", {})
    duration = float(fmt_info.get("duration") or audio_stream.get("duration") or 0.0)
    if duration <= 0:
        raise AudioValidationError(f"Could not determine duration for '{path.name}'")

    return AudioInfo(
        path=path,
        format=fmt,
        duration_seconds=duration,
        size_bytes=path.stat().st_size,
        sample_rate=int(audio_stream["sample_rate"]) if audio_stream.get("sample_rate") else None,
        channels=int(audio_stream["channels"]) if audio_stream.get("channels") else None,
        codec=audio_stream.get("codec_name"),
    )


async def validate_audio(path: str | Path, *, max_upload_mb: int | None = None) -> AudioInfo:
    """Async wrapper around ffprobe validation with size-limit enforcement."""
    settings = get_settings()
    limit_mb = max_upload_mb if max_upload_mb is not None else settings.audio_max_upload_mb

    path = Path(path)
    if not path.exists():
        raise AudioValidationError(f"Audio file not found: {path}")

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > limit_mb:
        raise AudioValidationError(
            f"File is {size_mb:.1f} MB which exceeds the {limit_mb} MB limit"
        )

    return await asyncio.to_thread(_probe_sync, path)


def _pick_silence_point(
    audio: AudioSegment,
    ideal_ms: int,
    window_ms: int = 15_000,
) -> int:
    """Find the quietest 100 ms window within ±window of ``ideal_ms``."""
    half = window_ms // 2
    start = max(0, ideal_ms - half)
    end = min(len(audio), ideal_ms + half)
    if end - start < 100:
        return ideal_ms

    best_ms, best_db = ideal_ms, float("inf")
    step = 100
    for position in range(start, end - step, step):
        window = audio[position : position + step]
        dbfs = window.dBFS
        # pydub reports -inf for pure silence; treat as quietest possible.
        rms = dbfs if dbfs != float("-inf") else -120.0
        if rms < best_db:
            best_db, best_ms = rms, position + (step // 2)
    return best_ms


def _chunk_sync(
    path: Path,
    out_dir: Path,
    chunk_seconds: int,
    overlap_seconds: float,
    trim_silence: bool,
) -> list[AudioChunk]:
    """Blocking chunking: silence-aware splits with overlap.

    Also applies a light noise gate (silence trim) + normalization pass when
    ``trim_silence`` is set, which measurably improves ASR on noisy calls.
    """
    audio = AudioSegment.from_file(str(path))

    if trim_silence:
        # Gate room tone: drop leading/trailing silence longer than ~1s.
        audio = normalize(audio)
        audio = audio.strip_silence(silence_len=1000)

    chunk_ms = chunk_seconds * 1000
    overlap_ms = int(overlap_seconds * 1000)

    if len(audio) <= chunk_ms:
        single = out_dir / f"{path.stem}_000.wav"
        audio.export(single, format="wav")
        return [
            AudioChunk(
                path=single,
                index=0,
                start_seconds=0.0,
                duration_seconds=len(audio) / 1000.0,
            )
        ]

    chunks: list[AudioChunk] = []
    position, index = 0, 0
    while position < len(audio):
        ideal_end = position + chunk_ms
        # Snap the cut to the quietest point nearby to avoid splitting words.
        cut_at = _pick_silence_point(audio, ideal_end) if index > 0 else ideal_end
        cut_at = min(cut_at, len(audio))

        piece = audio[position:cut_at]
        chunk_path = out_dir / f"{path.stem}_{index:03d}.wav"
        # 16 kHz mono is the sweet spot for Whisper-family models.
        piece.set_frame_rate(16_000).set_channels(1).export(chunk_path, format="wav")

        chunks.append(
            AudioChunk(
                path=chunk_path,
                index=index,
                start_seconds=position / 1000.0,
                duration_seconds=len(piece) / 1000.0,
            )
        )

        index += 1
        if cut_at >= len(audio):
            break
        position = max(cut_at - overlap_ms, position + 1)

    return chunks


async def chunk_audio(
    path: str | Path,
    *,
    out_dir: str | Path | None = None,
    chunk_seconds: int | None = None,
    overlap_seconds: float = CHUNK_OVERLAP_SECONDS,
    trim_silence: bool = True,
) -> list[AudioChunk]:
    """Split a large recording into ASR-friendly WAV chunks.

    Chunks are written to a per-file directory (``<stem>_<uuid8>/``) that the
    caller must clean up. Small files produce a single full-length chunk.
    """
    settings = get_settings()
    seconds = chunk_seconds or settings.audio_chunk_seconds

    path = Path(path)
    base = Path(out_dir) if out_dir else Path(settings.audio_artifacts_dir)
    target_dir = base / f"{path.stem}_{uuid.uuid4().hex[:8]}"
    target_dir.mkdir(parents=True, exist_ok=True)

    return await asyncio.to_thread(
        _chunk_sync, path, target_dir, seconds, overlap_seconds, trim_silence
    )


def cleanup_chunks(chunks: list[AudioChunk]) -> None:
    """Delete chunk files and their parent directory best-effort."""
    import shutil as _shutil

    parents: set[Path] = set()
    for chunk in chunks:
        parents.add(chunk.path.parent)
        chunk.path.unlink(missing_ok=True)
    for parent in parents:
        # Only remove now-empty chunk dirs we created.
        try:
            if parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
        except OSError:
            pass
    del _shutil
