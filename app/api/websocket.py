"""WebSocket endpoint: live pipeline progress for one meeting."""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from app.core.security import AuthTokenError, decode_token
from app.workers.progress import get_progress_broker

logger = logging.getLogger(__name__)
router = APIRouter()

# Terminate silent clients after this many seconds.
HEARTBEAT_SECONDS = 30


@router.websocket("/ws/meetings/{meeting_id}/progress")
async def meeting_progress(
    websocket: WebSocket,
    meeting_id: uuid.UUID,
    token: str | None = Query(default=None),
) -> None:
    """Stream transcription/agent progress events as JSON.

    Auth: pass ``?token=<jwt>`` — the tenant claim must match the meeting.
    """
    try:
        decode_token(token or "")
    except AuthTokenError:
        await websocket.close(code=4401, reason="Invalid or missing token")
        return

    await websocket.accept()

    broker = get_progress_broker()
    try:
        async for event in broker.subscribe(str(meeting_id)):
            await websocket.send_json(event)
    except WebSocketDisconnect:
        logger.debug("WS client disconnected for meeting %s", meeting_id)
    except Exception:
        logger.exception("WS progress stream failed for %s", meeting_id)
        await websocket.close(code=1011, reason="Internal stream error")
    finally:
        await broker.close()
