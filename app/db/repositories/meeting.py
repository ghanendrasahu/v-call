"""Meeting repository — lifecycle + transcript persistence."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.db.enums import MeetingSource, MeetingStatus
from app.db.models import Meeting, TranscriptSegment
from app.db.repositories.base import BaseRepository


class MeetingRepository(BaseRepository[Meeting]):
    """Async data access for meetings and their transcript segments."""

    model = Meeting

    async def create(
        self,
        *,
        tenant_id: uuid.UUID,
        title: str,
        source: MeetingSource = MeetingSource.UPLOAD,
        created_by_id: uuid.UUID | None = None,
        provider_meeting_id: str | None = None,
        external_url: str | None = None,
        storage_path: str | None = None,
        occurred_at: datetime | None = None,
    ) -> Meeting:
        meeting = Meeting(
            tenant_id=tenant_id,
            created_by_id=created_by_id,
            title=title,
            source=source,
            provider_meeting_id=provider_meeting_id,
            external_url=external_url,
            storage_path=storage_path,
            occurred_at=occurred_at,
        )
        return await self.add(meeting)

    async def get_with_details(
        self,
        meeting_id: uuid.UUID,
        *,
        tenant_id: uuid.UUID,
    ) -> Meeting | None:
        """Eager-load segments, action items, emails, and agent runs."""
        stmt = (
            select(Meeting)
            .where(Meeting.id == meeting_id, Meeting.tenant_id == tenant_id)
            .options(
                selectinload(Meeting.segments),
                selectinload(Meeting.action_items),
                selectinload(Meeting.follow_up_emails),
                selectinload(Meeting.agent_runs),
            )
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def update_status(
        self,
        meeting: Meeting,
        status: MeetingStatus,
        *,
        error: str | None = None,
        language: str | None = None,
        duration_seconds: float | None = None,
        audio_metadata: dict | None = None,
    ) -> Meeting:
        """Advance the processing lifecycle, stamping timestamps on transitions."""
        meeting.status = status
        meeting.error = error
        if language is not None:
            meeting.language = language
        if duration_seconds is not None:
            meeting.duration_seconds = duration_seconds
        if audio_metadata is not None:
            meeting.audio_metadata = audio_metadata

        now = datetime.now(UTC)
        if status is MeetingStatus.TRANSCRIBING and meeting.started_at is None:
            meeting.started_at = now
        if status in (MeetingStatus.COMPLETED, MeetingStatus.FAILED):
            meeting.completed_at = now

        await self.session.flush()
        return meeting

    async def attach_segments(
        self,
        meeting_id: uuid.UUID,
        tenant_id: uuid.UUID,
        segments: list[dict],
    ) -> list[TranscriptSegment]:
        """Bulk-insert diarized segments (``speaker_label/start_ms/end_ms/text``)."""
        rows = [
            TranscriptSegment(
                meeting_id=meeting_id,
                tenant_id=tenant_id,
                speaker_label=segment["speaker_label"],
                start_ms=int(segment["start_ms"]),
                end_ms=int(segment["end_ms"]),
                text=segment["text"],
                confidence=segment.get("confidence"),
            )
            for segment in segments
        ]
        self.session.add_all(rows)
        await self.session.flush()
        return rows

    async def list_segments(
        self,
        meeting_id: uuid.UUID,
        *,
        tenant_id: uuid.UUID,
    ) -> list[TranscriptSegment]:
        """Chronological transcript for a meeting."""
        stmt = (
            select(TranscriptSegment)
            .where(
                TranscriptSegment.meeting_id == meeting_id,
                TranscriptSegment.tenant_id == tenant_id,
            )
            .order_by(TranscriptSegment.start_ms)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def clear_storage_path(
        self,
        meeting_id: uuid.UUID,
        *,
        tenant_id: uuid.UUID,
    ) -> bool:
        """Drop the raw-audio pointer after compliance scrubbing."""
        meeting = await self.get(meeting_id, tenant_id=tenant_id)
        if meeting is None:
            return False
        meeting.storage_path = None
        await self.session.flush()
        return True

    async def delete(
        self,
        entity_id: uuid.UUID,
        *,
        tenant_id: uuid.UUID,
    ) -> bool:
        """Override to purge segments first (no orphan transcript rows)."""
        from sqlalchemy import delete as sa_delete

        await self.session.execute(
            sa_delete(TranscriptSegment).where(
                TranscriptSegment.meeting_id == entity_id,
                TranscriptSegment.tenant_id == tenant_id,
            )
        )
        return await super().delete(entity_id, tenant_id=tenant_id)
