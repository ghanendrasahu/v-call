"""Celery task definitions: the full meeting-intelligence pipeline.

``process_meeting`` is the flagship task: audio in → diarized transcript +
agent intelligence persisted. Long-running CPU work (ASR) happens in worker
threads off the event loop; progress is streamed over Redis; failures park in
the DLQ after bounded retries.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.agents.graph import MeetingAnalysisGraph
from app.agents.nodes.dispatcher import registry as dispatcher_registry
from app.db.enums import ActionItemStatus, MeetingStatus
from app.db.repositories import (
    ActionItemRepository,
    FollowUpEmailRepository,
    MeetingRepository,
)
from app.db.session import session_scope
from app.schemas.transcript import TranscriptionProgress
from app.services.transcription import TranscriptionService
from app.workers.celery_app import celery_app
from app.workers.dead_letter import push_dead_letter
from app.workers.progress import get_progress_broker

logger = logging.getLogger(__name__)


def _run_async(coro: Any) -> Any:
    """Run a coroutine from sync Celery code."""
    return asyncio.run(coro)


async def _publish_progress(
    meeting_id: str, stage: str, progress: float, message: str | None
) -> None:
    broker = get_progress_broker()
    event = TranscriptionProgress(stage=stage, progress=progress, message=message)
    try:
        await broker.publish(meeting_id, event.model_dump())
    except Exception:
        logger.warning("Failed to publish progress for %s", meeting_id, exc_info=True)


async def _process_meeting_async(meeting_id: str, tenant_id: str) -> dict[str, Any]:
    """Async core of the pipeline (runs inside the Celery event loop)."""
    # Celery JSON-serialises task arguments to strings, but UUID-column
    # filters/bind params require uuid.UUID — coerce once at the boundary.
    meeting_uuid = uuid.UUID(str(meeting_id))
    tenant_uuid = uuid.UUID(str(tenant_id))

    async def on_progress(event: TranscriptionProgress) -> None:
        await _publish_progress(meeting_id, event.stage, event.progress, event.message)

    async with session_scope() as session:
        meetings = MeetingRepository(session)
        meeting = await meetings.get_with_details(meeting_uuid, tenant_id=tenant_uuid)
        if meeting is None:
            raise ValueError(f"Meeting {meeting_id} not found for tenant {tenant_id}")

        await meetings.update_status(meeting, MeetingStatus.TRANSCRIBING)

        if not meeting.storage_path:
            raise ValueError(f"Meeting {meeting_id} has no stored audio")

        # ── 1) Transcription + diarization ─────────────────────────
        service = TranscriptionService()
        result = await service.transcribe_file(meeting.storage_path, on_progress=on_progress)
        await meetings.update_status(
            meeting,
            MeetingStatus.ANALYZING,
            language=result.language,
            duration_seconds=result.duration_seconds,
            audio_metadata=result.audio_metadata.model_dump(),
        )
        await meetings.attach_segments(
            meeting_uuid,
            tenant_uuid,
            [
                {
                    "speaker_label": u.speaker_label,
                    "start_ms": u.start_ms,
                    "end_ms": u.end_ms,
                    "text": u.text,
                    "confidence": u.confidence,
                }
                for u in result.utterances
            ],
        )
        await _publish_progress(meeting_id, "agents", 0.0, "Running analysis agents")

        # ── 2) LangGraph multi-agent analysis ──────────────────────
        segments = [
            {
                "speaker_label": u.speaker_label,
                "start_ms": u.start_ms,
                "end_ms": u.end_ms,
                "text": u.text,
            }
            for u in result.utterances
        ]
        from app.db.enums import AgentRunStatus
        from app.db.models import AgentRun

        run = AgentRun(
            meeting_id=meeting_uuid,
            tenant_id=tenant_uuid,
            status=AgentRunStatus.RUNNING,
        )
        session.add(run)
        await session.flush()
        started = datetime.now(UTC)

        graph = MeetingAnalysisGraph(registry=dispatcher_registry)
        final_state = await graph.run(
            meeting_id=meeting_id,
            tenant_id=tenant_id,
            transcript_segments=segments,
        )

        run.graph_state = {
            "decisions_made": final_state.get("decisions_made", []),
            "unresolved_debates": final_state.get("unresolved_debates", []),
            "sentiment_analysis": final_state.get("sentiment_analysis", {}),
            "integration_status": final_state.get("integration_status", {}),
        }
        run.errors = final_state.get("errors", [])
        run.latency_ms = int((datetime.now(UTC) - started).total_seconds() * 1000)
        run.status = AgentRunStatus.SUCCEEDED if not run.errors else AgentRunStatus.PARTIAL
        run.completed_at = datetime.now(UTC)
        await session.flush()
        await _publish_progress(meeting_id, "agents", 1.0, "Analysis complete")

        # ── 3) Persist agent outputs ───────────────────────────────
        action_items = ActionItemRepository(session)
        items = await action_items.replace_for_meeting(
            meeting_id=meeting_uuid,
            tenant_id=tenant_uuid,
            items=final_state.get("action_items", []),
        )
        # The graph already dispatched items (external_id present) — mark them.
        for item, state in zip(items, final_state.get("action_items", []), strict=False):
            external_id = state.get("external_id")
            if external_id:
                await action_items.mark_dispatched(
                    item.id,
                    tenant_id=tenant_uuid,
                    external_id=external_id,
                )
            elif state.get("status") == "failed":
                await action_items.mark_status(
                    item.id, tenant_id=tenant_uuid, status=ActionItemStatus.FAILED
                )

        emails = FollowUpEmailRepository(session)
        await emails.replace_for_meeting(
            meeting_id=meeting_uuid,
            tenant_id=tenant_uuid,
            emails=final_state.get("follow_up_emails", []),
        )

        await meetings.update_status(meeting, MeetingStatus.COMPLETED)
        return {
            "meeting_id": meeting_id,
            "utterances": len(result.utterances),
            "speakers": result.speaker_count,
            "action_items": len(items),
            "decisions": len(final_state.get("decisions_made", [])),
            "errors": final_state.get("errors", []),
        }


@celery_app.task(
    bind=True,
    name="app.workers.tasks.process_meeting",
    max_retries=3,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=600,
    retry_jitter=True,
)
def process_meeting(self, meeting_id: str, tenant_id: str) -> dict[str, Any]:
    """Full pipeline: transcribe → analyze → persist."""
    try:
        return _run_async(_process_meeting_async(meeting_id, tenant_id))
    except Exception as exc:
        logger.exception("process_meeting failed for %s", meeting_id)
        remaining = (self.request.retries or 0) >= self.max_retries
        if remaining:
            _run_async(
                push_dead_letter(
                    {
                        "task": self.name,
                        "meeting_id": meeting_id,
                        "tenant_id": tenant_id,
                        "error": str(exc),
                    }
                )
            )
        raise


@celery_app.task(
    bind=True,
    name="app.workers.tasks.scrub_expired_audio",
    max_retries=2,
    autoretry_for=(Exception,),
)
def scrub_expired_audio(self) -> dict[str, int]:
    """Hourly compliance sweep: delete raw audio past the retention window."""
    from sqlalchemy import select

    from app.core.config import get_settings
    from app.db.models import Meeting

    async def _scrub() -> dict[str, int]:
        cutoff = datetime.now(UTC).timestamp() - (get_settings().raw_audio_retention_hours * 3600)
        removed = 0
        async with session_scope() as session:
            result = await session.execute(
                select(Meeting).where(
                    Meeting.storage_path.is_not(None),
                    Meeting.created_at <= datetime.fromtimestamp(cutoff, tz=UTC),
                )
            )
            for row in result.scalars():
                storage = row.storage_path or ""

                def _remove(p: str = storage) -> None:
                    target = Path(p)
                    if target.exists():
                        target.unlink(missing_ok=True)

                await asyncio.to_thread(_remove)
                row.storage_path = None
                removed += 1
        return {"scrubbed": removed}

    return _run_async(_scrub())


@celery_app.task(
    bind=True,
    name="app.workers.tasks.replay_dead_letter",
    max_retries=1,
)
def replay_dead_letter(self) -> dict[str, int]:
    """Periodically retry dead-lettered ``process_meeting`` jobs."""

    async def _replay() -> dict[str, int]:
        from app.workers.dead_letter import pop_dead_letter_batch

        entries = await pop_dead_letter_batch()
        replayed = 0
        for entry in entries:
            if entry.get("task") != "app.workers.tasks.process_meeting":
                continue
            process_meeting.delay(entry["meeting_id"], entry["tenant_id"])
            replayed += 1
        return {"replayed": replayed, "skipped": len(entries) - replayed}

    return _run_async(_replay())


@celery_app.task(
    bind=True,
    name="app.workers.tasks.send_follow_up_emails",
    max_retries=3,
    autoretry_for=(Exception,),
)
def send_follow_up_emails(self, meeting_id: str, tenant_id: str) -> dict[str, Any]:
    """Send drafted follow-up emails via the Gmail MCP server."""

    async def _send() -> dict[str, Any]:
        from app.integrations.mcp_client import GmailMCPClient, MCPServerConfig

        # Task args arrive as JSON strings; UUID columns need uuid.UUID.
        meeting_uuid = uuid.UUID(str(meeting_id))
        tenant_uuid = uuid.UUID(str(tenant_id))

        async with session_scope() as session:
            emails_repo = FollowUpEmailRepository(session)
            pending = await emails_repo.list_unsent(meeting_uuid, tenant_id=tenant_uuid)
            if not pending:
                return {"sent": 0}

            config = MCPServerConfig(
                command="npx",
                args=("-y", "@google/gmail-mcp-server"),
            )
            sent = 0
            gmail = GmailMCPClient(config)
            async with gmail.connected():
                for email in pending:
                    await gmail.send_email(
                        to=email.recipient_email,
                        subject=email.subject or "Meeting follow-up",
                        body=email.email_body,
                    )
                    email.sent = True
                    email.sent_at = datetime.now(UTC)
                    sent += 1
            return {"sent": sent, "total": len(pending)}

    return _run_async(_send())
