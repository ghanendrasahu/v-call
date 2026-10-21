"""Tests for structured logging configuration (structlog over stdlib)."""

import json
import logging

import pytest
import structlog

from app.core.logging import configure_logging


def _root() -> logging.Logger:
    return logging.getLogger()


@pytest.fixture(autouse=True)
def _restore_logging():
    """Keep each test's logging configuration from leaking into the suite."""
    yield
    configure_logging(json_logs=False)


def _stderr_json_line(capsys, needle: str) -> dict:
    err = capsys.readouterr().err
    lines = [line for line in err.splitlines() if needle in line]
    assert lines, f"no log line containing {needle!r} in stderr"
    return json.loads(lines[-1])


def test_json_mode_emits_valid_json(capsys):
    configure_logging(json_logs=True)
    logging.getLogger("vcall.test").info("hello %s", "world")

    data = _stderr_json_line(capsys, "hello world")
    assert data["event"] == "hello world"
    assert data["level"] == "info"
    assert data["logger"] == "vcall.test"
    assert "ts" in data


def test_request_id_is_bound_from_contextvars(capsys):
    configure_logging(json_logs=True)
    structlog.contextvars.bind_contextvars(request_id="rid-123")
    try:
        logging.getLogger("vcall.test").info("correlated record")
    finally:
        structlog.contextvars.clear_contextvars()

    data = _stderr_json_line(capsys, "correlated record")
    assert data["request_id"] == "rid-123"


def test_console_mode_stays_human_readable(capsys):
    configure_logging(json_logs=False)
    logging.getLogger("vcall.test").warning("console record")

    err = capsys.readouterr().err
    lines = [line for line in err.splitlines() if "console record" in line]
    assert lines
    with pytest.raises(json.JSONDecodeError):
        json.loads(lines[-1])


def test_importance_of_idempotency_no_duplicate_handlers():
    configure_logging(json_logs=True)
    handlers_after_first = len(_root().handlers)
    configure_logging(json_logs=True)  # second call must replace, not stack
    assert len(_root().handlers) == handlers_after_first


def test_log_level_is_applied(capsys):
    configure_logging(json_logs=True, level="WARNING")
    logging.getLogger("vcall.test").info("below threshold")
    logging.getLogger("vcall.test").warning("above threshold")

    err = capsys.readouterr().err
    assert "below threshold" not in err
    assert "above threshold" in err
