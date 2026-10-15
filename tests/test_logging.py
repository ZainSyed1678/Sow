"""Tests for structured logging."""

import json
import logging

from ai_ml_notion_researcher.logging_utils import JSONFormatter, get_logger, setup_logging


def test_json_formatter():
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="Test message with %s",
        args=("parameter",),
        exc_info=None,
    )
    record.custom_field = "custom_value"  # type: ignore

    formatted = formatter.format(record)
    parsed = json.loads(formatted)

    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "test_logger"
    assert parsed["message"] == "Test message with parameter"
    assert parsed["custom_field"] == "custom_value"
    assert "timestamp" in parsed


def test_setup_logging_stream():
    logger = setup_logging(log_level="DEBUG", json_format=False)
    assert logger.level == logging.DEBUG
    assert len(logger.handlers) == 1

    child = get_logger("collector")
    assert child.name == "ai_ml_notion_researcher.collector"
