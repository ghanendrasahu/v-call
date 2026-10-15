"""Redis-backed progress broker for live pipeline streaming.

The transcription pipeline (async, inside Celery workers) publishes progress
events onto a per-meeting Redis pub/sub channel; the FastAPI WebSocket layer
subscribes and fans events out to browsers. Redis is the transport, so the
web tier and worker tier never need to share memory or sockets.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import redis.asyncio as aioredis

from app.core.config import get_settings

CHANNEL_PREFIX = "vcall:progress:"
# Keep the latest snapshot so a late-joining WS still shows current status.
SNAPSHOT_SUFFIX = ":snapshot"


def channel_for(meeting_id: str) -> str:
    return f"{CHANNEL_PREFIX}{meeting_id}"


def snapshot_key(meeting_id: str) -> str:
    return channel_for(meeting_id) + SNAPSHOT_SUFFIX


class ProgressBroker:
    """Publish/subscribe meeting progress over Redis."""

    def __init__(self, redis_url: str | None = None) -> None:
        self._redis = aioredis.from_url(
            redis_url or get_settings().redis_url, decode_responses=True
        )

    async def publish(self, meeting_id: str, event: dict[str, Any]) -> None:
        """Broadcast an event and cache it as the meeting's latest snapshot."""
        payload = json.dumps(event, default=str)
        pipe = self._redis.pipeline()
        pipe.publish(channel_for(meeting_id), payload)
        pipe.set(snapshot_key(meeting_id), payload, ex=60 * 60 * 24)
        await pipe.execute()

    async def latest(self, meeting_id: str) -> dict[str, Any] | None:
        """Most recent event for a meeting (or None)."""
        raw = await self._redis.get(snapshot_key(meeting_id))
        return json.loads(raw) if raw else None

    async def subscribe(self, meeting_id: str) -> AsyncIterator[dict[str, Any]]:
        """Async generator of events (use inside ``async for``)."""
        pubsub = self._redis.pubsub()
        await pubsub.subscribe(channel_for(meeting_id))
        try:
            # Replay the snapshot first so clients get immediate context.
            latest = await self.latest(meeting_id)
            if latest is not None:
                yield latest
            async for message in pubsub.listen():
                if message.get("type") != "message":
                    continue
                data = message.get("data")
                if not data:
                    continue
                yield json.loads(data)
        finally:
            await pubsub.unsubscribe(channel_for(meeting_id))
            await pubsub.aclose()

    async def close(self) -> None:
        await self._redis.aclose()


# Process-wide default broker (lazy: created on first use in a running loop).
_default: ProgressBroker | None = None


def get_progress_broker() -> ProgressBroker:
    global _default
    if _default is None:
        _default = ProgressBroker()
    return _default
