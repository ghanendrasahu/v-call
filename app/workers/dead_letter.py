"""Dead-letter queue handling.

When a task exhausts its retries it is parked in a Redis list
(``vcall:dlq``) with full context. A periodic beat task replays entries
(bounded, FIFO) once the underlying issue looks transient; anything still
failing stays parked for operator review (e.g. via Flower).
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

import redis.asyncio as aioredis

from app.core.config import get_settings

logger = logging.getLogger(__name__)

DLQ_KEY = "vcall:dlq"
MAX_DLQ_SIZE = 10_000
REPLAY_BATCH = 10


def _redis() -> aioredis.Redis:
    return aioredis.from_url(get_settings().redis_url, decode_responses=True)


async def push_dead_letter(entry: dict[str, Any]) -> None:
    """Park a failed task; trim to a bounded size."""
    record = {
        **entry,
        "failed_at": datetime.now(UTC).isoformat(),
    }
    client = _redis()
    try:
        pipe = client.pipeline()
        pipe.lpush(DLQ_KEY, json.dumps(record, default=str))
        pipe.ltrim(DLQ_KEY, 0, MAX_DLQ_SIZE - 1)
        await pipe.execute()
    finally:
        await client.aclose()
    logger.error("Dead-lettered task: %s", record.get("task"))


async def pop_dead_letter_batch(limit: int = REPLAY_BATCH) -> list[dict[str, Any]]:
    """Pop up to ``limit`` entries (FIFO) for replay."""
    client = _redis()
    try:
        entries: list[dict[str, Any]] = []
        for _ in range(limit):
            raw = await client.rpop(DLQ_KEY)
            if raw is None:
                break
            entries.append(json.loads(raw))
        return entries
    finally:
        await client.aclose()


async def dlq_depth() -> int:
    """Number of parked tasks."""
    client = _redis()
    try:
        return int(await client.llen(DLQ_KEY))
    finally:
        await client.aclose()
