"""Meeting endpoints: upload, lifecycle, transcript, and results."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import (
    APIRouter,
    Depends,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import TenantContext, get_current_context, get_db
from app.core.config import get_settings
from app.core.rate_limit import limiter, rate_limit_heavy
from app.db.enums import MeetingSource
from app.db.models import AgentRun
from app.db.repositories import MeetingRepository
from app.schemas.api import (
    ActionItemOut,
    FollowUpEmailOut,
    MeetingAccepted,
    MeetingDetail,
    MeetingSummary,
    SentimentOut,
    TranscriptSegmentOut,
)
from app.services.audio import AudioValidationError, check_extension

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/meetings", tags=["meetings"])

ALLOWED_UPLOADS = {"audio/mpeg", "audio/mp4", "audio/wav", "audio/x-m4a", "audio/flac"}


def _uploads_dir(tenant_id: uuid.UUID) -> Path:
    base = Path(get_settings().audio_artifacts_dir) / "uploads" / str(tenant_id)
    base.mkdir(parents=True, exist_ok=True)
    return base


@router.post(
    "",
    response_model=MeetingAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
@limiter.limit(rate_limit_heavy)
async def upload_meeting(
    request: Request,
    file: UploadFile,
    title: str = Form(..., min_length=1, max_length=500),
    source: MeetingSource = Form(default=MeetingSource.UPLOAD),
    context: TenantContext = Depends(get_current_context),
    session: AsyncSession = Depends(get_db),
) -> MeetingAccepted:
    """Upload an audio recording and enqueue the intelligence pipeline."""
    settings = get_settings()

    # 1) Validate container type.
    try:
        fmt = check_extension(Path(file.filename or "audio.wav"))
    except AudioValidationError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc

    meeting_id = uuid.uuid4()
    destination = _uploads_dir(context.tenant_id) / f"{meeting_id}.{fmt}"

    # 2) Stream to disk in bounded chunks (never buffer whole file in RAM).
    size_bytes = 0
    max_bytes = settings.audio_max_upload_mb * 1024 * 1024
    with destination.open("wb") as out:
        while chunk := await file.read(1024 * 1024):
            size_bytes += len(chunk)
            if size_bytes > max_bytes:
                out.close()
                destination.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds {settings.audio_max_upload_mb} MB limit",
                )
            out.write(chunk)

    if size_bytes == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    # 3) Persist + enqueue.
    repo = MeetingRepository(session)
    meeting = await repo.create(
        tenant_id=context.tenant_id,
        title=title,
        source=source,
        created_by_id=context.user_id,
        storage_path=str(destination),
    )

    from app.workers.tasks import process_meeting

    process_meeting.delay(str(meeting.id), str(context.tenant_id))
    logger.info("Queued meeting %s for tenant %s", meeting.id, context.tenant_id)

    return MeetingAccepted(
        meeting_id=meeting.id,
        status=meeting.status,
        detail="Pipeline queued; subscribe to the progress WebSocket for updates",
    )


@router.get("", response_model=list[MeetingSummary])
async def list_meetings(
    limit: int = 20,
    offset: int = 0,
    context: TenantContext = Depends(get_current_context),
    session: AsyncSession = Depends(get_db),
) -> list[MeetingSummary]:
    """Paginated meeting list for the current tenant."""
    repo = MeetingRepository(session)
    meetings = await repo.list(tenant_id=context.tenant_id, limit=limit, offset=offset)
    return [MeetingSummary.model_validate(m) for m in meetings]


def _sentiment_from_runs(runs: list[AgentRun]) -> SentimentOut | None:
    for run in sorted(runs, key=lambda r: r.created_at, reverse=True):
        state = run.graph_state or {}
        sentiment = state.get("sentiment_analysis")
        if sentiment:
            return SentimentOut(**sentiment)
    return None


@router.get("/{meeting_id}", response_model=MeetingDetail)
async def get_meeting(
    meeting_id: uuid.UUID,
    context: TenantContext = Depends(get_current_context),
    session: AsyncSession = Depends(get_db),
) -> MeetingDetail:
    """Full meeting detail: transcript, action items, emails, sentiment."""
    repo = MeetingRepository(session)
    meeting = await repo.get_with_details(meeting_id, tenant_id=context.tenant_id)
    if meeting is None:
        raise HTTPException(status_code=404, detail="Meeting not found")

    detail = MeetingDetail.model_validate(meeting)
    detail.sentiment = _sentiment_from_runs(list(meeting.agent_runs))
    return detail


@router.get("/{meeting_id}/transcript", response_model=list[TranscriptSegmentOut])
async def get_transcript(
    meeting_id: uuid.UUID,
    context: TenantContext = Depends(get_current_context),
    session: AsyncSession = Depends(get_db),
) -> list[TranscriptSegmentOut]:
    """Diarized transcript segments in chronological order."""
    repo = MeetingRepository(session)
    meeting = await repo.get(meeting_id, tenant_id=context.tenant_id)
    if meeting is None:
        raise HTTPException(status_code=404, detail="Meeting not found")
    segments = await repo.list_segments(meeting_id, tenant_id=context.tenant_id)
    return [TranscriptSegmentOut.model_validate(s) for s in segments]


@router.get("/{meeting_id}/action-items", response_model=list[ActionItemOut])
async def get_action_items(
    meeting_id: uuid.UUID,
    context: TenantContext = Depends(get_current_context),
    session: AsyncSession = Depends(get_db),
) -> list[ActionItemOut]:
    """Extracted action items for one meeting."""
    from app.db.repositories import ActionItemRepository

    repo = MeetingRepository(session)
    meeting = await repo.get(meeting_id, tenant_id=context.tenant_id)
    if meeting is None:
        raise HTTPException(status_code=404, detail="Meeting not found")

    items = await ActionItemRepository(session).list_for_meeting(
        meeting_id, tenant_id=context.tenant_id
    )
    return [ActionItemOut.model_validate(i) for i in items]


@router.get("/{meeting_id}/follow-up-emails", response_model=list[FollowUpEmailOut])
async def get_follow_up_emails(
    meeting_id: uuid.UUID,
    context: TenantContext = Depends(get_current_context),
    session: AsyncSession = Depends(get_db),
) -> list[FollowUpEmailOut]:
    """Drafted follow-up emails for one meeting."""
    repo = MeetingRepository(session)
    meeting = await repo.get_with_details(meeting_id, tenant_id=context.tenant_id)
    if meeting is None:
        raise HTTPException(status_code=404, detail="Meeting not found")
    return [FollowUpEmailOut.model_validate(e) for e in meeting.follow_up_emails]


@router.post(
    "/{meeting_id}/reprocess",
    response_model=MeetingAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
@limiter.limit(rate_limit_heavy)
async def reprocess_meeting(
    request: Request,
    meeting_id: uuid.UUID,
    context: TenantContext = Depends(get_current_context),
    session: AsyncSession = Depends(get_db),
) -> MeetingAccepted:
    """Re-run the pipeline for a completed/failed meeting."""
    repo = MeetingRepository(session)
    meeting = await repo.get(meeting_id, tenant_id=context.tenant_id)
    if meeting is None:
        raise HTTPException(status_code=404, detail="Meeting not found")
    if not meeting.storage_path:
        raise HTTPException(
            status_code=409,
            detail="Raw audio has been scrubbed; re-upload is required",
        )

    from app.db.enums import MeetingStatus
    from app.workers.tasks import process_meeting

    await repo.update_status(meeting, MeetingStatus.PENDING, error=None)
    process_meeting.delay(str(meeting.id), str(context.tenant_id))
    return MeetingAccepted(
        meeting_id=meeting.id,
        status=meeting.status,
        detail="Reprocessing queued",
    )


@router.delete("/{meeting_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_meeting(
    meeting_id: uuid.UUID,
    context: TenantContext = Depends(get_current_context),
    session: AsyncSession = Depends(get_db),
) -> None:
    """Delete a meeting, its transcript, and its raw audio (right to erasure)."""
    repo = MeetingRepository(session)
    meeting = await repo.get(meeting_id, tenant_id=context.tenant_id)
    if meeting is None:
        raise HTTPException(status_code=404, detail="Meeting not found")

    if meeting.storage_path:
        import asyncio

        storage = meeting.storage_path
        await asyncio.to_thread(lambda: Path(storage).unlink(missing_ok=True))

    await repo.delete(meeting_id, tenant_id=context.tenant_id)
