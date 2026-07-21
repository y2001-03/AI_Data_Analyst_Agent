"""
Enterprise exception hierarchy.

Every application-level error inherits from ``AppException`` so the
global FastAPI handler can produce a uniform JSON envelope.

Subclassing guide::

    AppException               ← base (error_code + message + status_code + details)
    ├── LLMException           ← LLM provider / model failures
    ├── DatasetException       ← upload, parsing, storage errors
    ├── ToolExecutionException ← pandas / SQL / viz tool failures
    ├── ValidationException    ← invalid input (schema, params)
    └── ConfigurationException ← missing / invalid settings

Service code should **never** raise bare ``ValueError`` or
``RuntimeError`` — always use one of the typed exceptions above
(or subclass further when a new domain emerges).
"""

from __future__ import annotations

import logging
import traceback
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.config import get_settings

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Base exception
# ═══════════════════════════════════════════════════════════════════


class AppException(Exception):
    """Root of the application exception hierarchy.

    Args:
        message: Human-readable description (returned to the client).
        error_code: Machine-readable slug, e.g. ``"LLM_TIMEOUT"``.
        status_code: HTTP status code (default 400).
        details: Optional dict with structured context (never includes
            secrets, stack traces, or internal paths in production).
    """

    def __init__(
        self,
        message: str,
        *,
        error_code: str = "APP_ERROR",
        status_code: int = 400,
        details: dict[str, object] | None = None,
    ) -> None:
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)


# ═══════════════════════════════════════════════════════════════════
# Domain exceptions
# ═══════════════════════════════════════════════════════════════════


class LLMException(AppException):
    """LLM provider or model invocation failure.

    Raised when the provider returns an error, times out, or produces
    invalid structured output.
    """

    def __init__(
        self,
        message: str,
        *,
        error_code: str = "LLM_PROVIDER_ERROR",
        status_code: int = 502,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(
            message,
            error_code=error_code,
            status_code=status_code,
            details=details,
        )


class DatasetException(AppException):
    """Dataset upload, parsing, or retrieval failure."""

    def __init__(
        self,
        message: str,
        *,
        error_code: str = "DATASET_ERROR",
        status_code: int = 400,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(
            message,
            error_code=error_code,
            status_code=status_code,
            details=details,
        )


class ToolExecutionException(AppException):
    """Pandas / SQL / visualization tool execution failure."""

    def __init__(
        self,
        message: str,
        *,
        error_code: str = "TOOL_EXECUTION_ERROR",
        status_code: int = 500,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(
            message,
            error_code=error_code,
            status_code=status_code,
            details=details,
        )


class ValidationException(AppException):
    """Request validation failure (missing fields, bad types, etc.)."""

    def __init__(
        self,
        message: str,
        *,
        error_code: str = "VALIDATION_ERROR",
        status_code: int = 422,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(
            message,
            error_code=error_code,
            status_code=status_code,
            details=details,
        )


class ConfigurationException(AppException):
    """Missing or invalid application configuration."""

    def __init__(
        self,
        message: str,
        *,
        error_code: str = "CONFIGURATION_ERROR",
        status_code: int = 500,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(
            message,
            error_code=error_code,
            status_code=status_code,
            details=details,
        )


# ═══════════════════════════════════════════════════════════════════
# FastAPI global exception handlers
# ═══════════════════════════════════════════════════════════════════


def _build_error_body(
    exc: AppException,
    *,
    debug: bool = False,
) -> dict[str, object]:
    """Build the standard JSON error envelope.

    Args:
        exc: The application exception.
        debug: When ``True`` (dev mode), include a sanitised stack
            trace.  Never enabled in production.

    Returns:
        A dict safe to serialise into ``JSONResponse`` content.
    """
    body: dict[str, object] = {
        "error_code": exc.error_code,
        "message": exc.message,
        # backward-compatible key for clients that read ``detail``
        "detail": exc.message,
        "details": exc.details,
    }
    if debug:
        body["debug"] = {
            "exception_type": type(exc).__name__,
            "traceback": _sanitised_traceback(exc),
        }
    return body


def _sanitised_traceback(exc: BaseException) -> list[str]:
    """Return traceback lines with the project root scrubbed."""
    tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
    return [
        line.rstrip("\n").replace("\\", "/")
        for line in tb
    ]


def register_exception_handlers(app: FastAPI) -> None:
    """Register FastAPI exception handlers.

    Must be called once during application startup (``main.py``).
    """

    settings = get_settings()
    is_debug = settings.app_debug

    @app.exception_handler(AppException)
    async def handle_app_exception(
        _request: Request,
        exc: AppException,
    ) -> JSONResponse:
        logger.warning(
            "Application exception | code=%s status=%d message=%s",
            exc.error_code,
            exc.status_code,
            exc.message,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=_build_error_body(exc, debug=is_debug),
        )

    @app.exception_handler(Exception)
    async def handle_unhandled_exception(
        _request: Request,
        exc: Exception,
    ) -> JSONResponse:
        """Catch-all for exceptions that are *not* ``AppException`` subclasses.

        Returns a generic 500 error.  The actual traceback is logged
        server-side but never exposed to the client.
        """
        logger.exception("Unhandled exception: %s", exc)
        body: dict[str, object] = {
            "error_code": "INTERNAL_ERROR",
            "message": "An unexpected error occurred. Please try again later.",
            "detail": "An unexpected error occurred. Please try again later.",
            "details": {},
        }
        if is_debug:
            body["debug"] = {
                "exception_type": type(exc).__name__,
                "traceback": _sanitised_traceback(exc),
            }
        return JSONResponse(status_code=500, content=body)

    # Also handle FastAPI's built-in validation errors so they use the
    # same envelope shape.
    from fastapi.exceptions import RequestValidationError

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        _request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        logger.warning("Request validation error: %s", exc.errors())
        return JSONResponse(
            status_code=422,
            content={
                "error_code": "VALIDATION_ERROR",
                "message": "Request validation failed.",
                "detail": "Request validation failed.",
                "details": {"errors": exc.errors()},
            },
        )
