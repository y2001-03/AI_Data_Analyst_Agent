"""
Structured logging configuration.

Supports two output modes selected via ``APP_ENV``:

- **development** — human-readable colourised text, ideal for local dev.
- **production** (or any other value) — JSON lines, machine-parseable
  by log aggregators (ELK, Datadog, CloudWatch, etc.).

Every log record automatically carries ``request_id`` and ``trace_id``
when they are set in the request-local ``contextvars`` (see
``app.core.context``).  No manual threading of identifiers is needed.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from app.core.context import request_id_var, trace_id_var

# ---------------------------------------------------------------------------
# Log level mapping
# ---------------------------------------------------------------------------

_LOG_LEVELS: dict[str, int] = {
    "development": logging.DEBUG,
    "testing": logging.INFO,
    "production": logging.INFO,
}


# ---------------------------------------------------------------------------
# Filter — injects request_id / trace_id into every LogRecord
# ---------------------------------------------------------------------------


class _RequestContextFilter(logging.Filter):
    """Attach ``request_id`` and ``trace_id`` from ``contextvars``.

    Because ``contextvars`` are natively supported by the stdlib
    ``logging`` module (since Python 3.8), the injected attributes are
    available in format strings via ``%(request_id)s`` / ``%(trace_id)s``
    without any extra plumbing.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        record.trace_id = trace_id_var.get()
        return True


# ---------------------------------------------------------------------------
# Development formatter — colourised terminal output
# ---------------------------------------------------------------------------

_COLOUR_MAP: dict[str, str] = {
    "DEBUG": "\033[36m",     # cyan
    "INFO": "\033[32m",      # green
    "WARNING": "\033[33m",   # yellow
    "ERROR": "\033[31m",     # red
    "CRITICAL": "\033[1;31m",  # bold red
}
_RESET = "\033[0m"


class _ColourFormatter(logging.Formatter):
    """Human-readable coloured log lines for local development."""

    def format(self, record: logging.LogRecord) -> str:
        colour = _COLOUR_MAP.get(record.levelname, "")
        level_coloured = f"{colour}{record.levelname:<8}{_RESET}"
        rid = getattr(record, "request_id", "")[:8] or "-"  # truncated for readability
        tid = getattr(record, "trace_id", "")[:8] or "-"
        return (
            f"{self.formatTime(record)} | {level_coloured} | "
            f"{record.name:<36} | "
            f"req={rid} trace={tid} | "
            f"{record.getMessage()}"
        )

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
        return dt.strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
# Production formatter — JSON lines
# ---------------------------------------------------------------------------


class _JSONFormatter(logging.Formatter):
    """One JSON object per line, suitable for structured-log pipelines."""

    def format(self, record: logging.LogRecord) -> str:
        obj: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "request_id": getattr(record, "request_id", ""),
            "trace_id": getattr(record, "trace_id", ""),
        }
        if record.exc_info and record.exc_info[1] is not None:
            obj["exception"] = {
                "type": type(record.exc_info[1]).__name__,
                "message": str(record.exc_info[1]),
            }
        return json.dumps(obj, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def configure_logging(environment: str) -> None:
    """Set up the root logger for the given environment.

    Called once from ``main.py`` at application startup.

    Args:
        environment: One of ``"development"``, ``"testing"``, ``"production"``.
    """
    level = _LOG_LEVELS.get(environment, logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)

    # Remove any handlers attached by uvicorn or third-party packages
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    if environment == "development":
        handler.setFormatter(_ColourFormatter())
    else:
        handler.setFormatter(_JSONFormatter())

    handler.addFilter(_RequestContextFilter())
    root.addHandler(handler)

    # Silence noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # Ensure our own loggers propagate to the root
    for name in ("app", "langgraph", "uvicorn"):
        logging.getLogger(name).setLevel(level)

    root.info("Logging configured | env=%s level=%s", environment, logging.getLevelName(level))


def get_logger(name: str) -> logging.Logger:
    """Convenience: return a logger with the given dotted name.

    Usage in any module::

        from app.core.logging import get_logger

        logger = get_logger(__name__)
        logger.info("Processing dataset %s", dataset_id)
    """
    return logging.getLogger(name)
