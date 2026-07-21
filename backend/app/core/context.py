"""
Request-local context variables.

Uses ``contextvars`` to propagate request-scoped identifiers
(request_id, trace_id) across async boundaries without threading
them through every function signature.

Usage::

    from app.core.context import request_id_var, trace_id_var

    rid = request_id_var.get()   # "" when outside an HTTP request
    trace_id_var.set("uuid-...") # set at the start of a graph run
"""

from __future__ import annotations

from contextvars import ContextVar

#: Populated by ``RequestIDMiddleware`` for every HTTP request.
#: Defaults to ``""`` so log formatters never crash on a missing value.
request_id_var: ContextVar[str] = ContextVar("request_id", default="")

#: Populated at the start of each LangGraph / dataset analysis run.
#: Allows correlating all nodes within one logical trace.
trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")
