"""Structured logging configuration (structlog over stdlib logging).

One entry point: :func:`configure_logging`.

- ``json_logs=True``  → one JSON line per record (``ts``, ``level``, ``logger``,
  ``event``, plus contextvars-bound fields such as ``request_id``).
- default             → plain console rendering for local development.

Every existing ``logging.getLogger(__name__)`` call site keeps working: both
structlog and stdlib records flow through a single compliant
``ProcessorFormatter``, so ``vcall.access`` and all module loggers emit the
same shape without code changes.
"""

from __future__ import annotations

import logging

import structlog
from structlog.typing import EventDict

DEFAULT_LOG_LEVEL = "INFO"


def add_logger_name(logger: object, _method_name: str, event_dict: EventDict) -> EventDict:
    """Attach the originating logger name (stdlib records and structlog alike)."""
    record: logging.LogRecord | None = event_dict.get("_record")
    event_dict.setdefault(
        "logger",
        record.name if record is not None else getattr(logger, "name", "?"),
    )
    return event_dict


def configure_logging(*, json_logs: bool = False, level: str = DEFAULT_LOG_LEVEL) -> None:
    """Configure the root logger and structlog.

    Idempotent: repeated calls replace the previous handler instead of
    duplicating output.
    """
    timestamp = structlog.processors.TimeStamper(fmt="%Y-%m-%dT%H:%M:%S%z", key="ts")
    pre_chain: list[structlog.typing.Processor] = [
        add_logger_name,
        # Pulls fields bound via ``structlog.contextvars`` (e.g. request_id set
        # by RequestContextMiddleware) into every record.
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        timestamp,
        structlog.processors.StackInfoRenderer(),
    ]
    renderer: structlog.typing.Processor
    tail: list[structlog.typing.Processor] = [
        structlog.stdlib.ProcessorFormatter.remove_processors_meta
    ]
    if json_logs:
        renderer = structlog.processors.JSONRenderer()
        tail.append(structlog.processors.format_exc_info)
        tail.append(renderer)
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=False)
        tail.append(renderer)

    structlog.configure(
        processors=[*pre_chain, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        # Tests toggle json_logs repeatedly and expect the change to stick;
        # caching across the process would hide it.
        cache_logger_on_first_use=False,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        # Records emitted by plain ``logging`` flow through the same chain.
        foreign_pre_chain=pre_chain,
        processors=tail,
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level.upper())
