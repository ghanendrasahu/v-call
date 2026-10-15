"""Celery application: queues, retry policy, DLQ, and beat schedule.

Queues
------
- ``default``: lightweight housekeeping.
- ``audio``: long-running transcription jobs (isolated so heavy CPU work
  never starves the others).
- ``agents``: LangGraph analysis runs.
- ``dlq``: dead-letter queue — tasks that exhaust retries land here for
  manual inspection/replay (see :mod:`app.workers.dead_letter`).
"""

from __future__ import annotations

from celery import Celery
from kombu import Queue

from app.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "vcall",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.workers.tasks"],
)

celery_app.conf.update(
    # ── Serialization ──────────────────────────────────────────────
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    # ── Reliability ────────────────────────────────────────────────
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,  # fair scheduling for long audio jobs
    task_time_limit=60 * 60,  # hard kill after 1h
    task_soft_time_limit=55 * 60,
    # ── Retry / DLQ ────────────────────────────────────────────────
    task_default_retry_delay=30,
    broker_transport_options={
        "visibility_timeout": 60 * 60 * 3,  # 3h > time_limit
        "retry_on_timeout": True,
    },
    result_expires=60 * 60 * 24,
    # ── Queues ─────────────────────────────────────────────────────
    task_queues=(
        Queue("default", routing_key="default"),
        Queue("audio", routing_key="audio"),
        Queue("agents", routing_key="agents"),
        Queue("dlq", routing_key="dlq"),
    ),
    task_default_queue="default",
    task_routes={
        "app.workers.tasks.process_meeting": {"queue": "audio"},
        "app.workers.tasks.analyze_meeting": {"queue": "agents"},
        "app.workers.tasks.send_follow_up_emails": {"queue": "default"},
        "app.workers.tasks.scrub_expired_audio": {"queue": "default"},
    },
    # ── Beat schedule ──────────────────────────────────────────────
    beat_schedule={
        "scrub-expired-raw-audio": {
            "task": "app.workers.tasks.scrub_expired_audio",
            "schedule": 60 * 60.0,  # hourly
        },
        "replay-dead-letter": {
            "task": "app.workers.tasks.replay_dead_letter",
            "schedule": 60 * 30.0,  # every 30 min
        },
    },
    # ── Observability ──────────────────────────────────────────────
    worker_send_task_events=True,
    task_send_sent_event=True,
)


def run_worker() -> None:
    """Console entrypoint: ``vcall-worker``."""
    celery_app.worker_main(argv=["worker", "--loglevel=info"])


def run_beat() -> None:
    """Console entrypoint: ``vcall-beat``."""
    celery_app.start(argv=["beat", "--loglevel=info"])
