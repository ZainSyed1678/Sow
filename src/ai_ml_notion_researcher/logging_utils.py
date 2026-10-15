"""Structured logging utility using Python standard library logging."""

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include standard exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Include custom extra fields if attached to record
        for key, val in record.__dict__.items():
            if key not in {
                "args",
                "asctime",
                "created",
                "exc_info",
                "exc_text",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "message",
                "msg",
                "name",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "stack_info",
                "thread",
                "threadName",
            }:
                log_entry[key] = val

        return json.dumps(log_entry, default=str)


def setup_logging(log_level: str = "INFO", json_format: bool = False) -> logging.Logger:
    """Configures root logging with either human-readable or structured JSON output.

    # ponytail: stdlib logging used directly, no external logging framework needed.
    """
    root_logger = logging.getLogger("ai_ml_notion_researcher")
    level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    if json_format:
        handler.setFormatter(JSONFormatter())
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)

    root_logger.addHandler(handler)
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Returns a child logger under the ai_ml_notion_researcher namespace."""
    return logging.getLogger(f"ai_ml_notion_researcher.{name}")
