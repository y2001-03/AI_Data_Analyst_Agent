"""
FastAPI middleware — request identification and observability.

RequestIDMiddleware
    Generates a unique ``X-Request-ID`` for every incoming HTTP request,
    sets it in the ``request_id_var`` context variable (so all logs
    emitted during the request carry it), and echoes it back in the
    response header.
"""

from __future__ import annotations

import logging
import time
from uuid import uuid4

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.context import request_id_var

logger = logging.getLogger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a unique ``X-Request-ID`` to every request-response cycle.

    Behaviour:
    1. Reads ``X-Request-ID`` from the incoming request (if the client
       sends one, it is reused for end-to-end tracing).
    2. Otherwise generates a new UUID4.
    3. Sets ``request_id_var`` → all logs in this request carry the id.
    4. Appends the header to the response.
    5. Logs the request method, path, status, and latency.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        # 1. Resolve request id
        request_id = request.headers.get("X-Request-ID", str(uuid4()))
        request_id_var.set(request_id)

        # 2. Process the request
        start = time.monotonic()
        response = await call_next(request)
        elapsed_ms = (time.monotonic() - start) * 1000

        # 3. Echo back to client
        response.headers["X-Request-ID"] = request_id

        # 4. Access log (one line per request)
        logger.info(
            "%s %s → %s | %.1fms",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response
