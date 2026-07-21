"""FastAPI application entrypoint."""

from fastapi import FastAPI

from app.api.router import api_router
from app.core.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging, get_logger
from app.core.middleware import RequestIDMiddleware

settings = get_settings()
configure_logging(settings.app_env)

logger = get_logger(__name__)

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    debug=settings.app_debug,
)

# ── Observability ──────────────────────────────────────────────────
app.add_middleware(RequestIDMiddleware)

# ── Error handling ─────────────────────────────────────────────────
register_exception_handlers(app)

# ── Routes ─────────────────────────────────────────────────────────
app.include_router(api_router, prefix="/api")

logger.info(
    "Application starting | name=%s env=%s debug=%s",
    settings.app_name,
    settings.app_env,
    settings.app_debug,
)


@app.get("/", tags=["system"])
def root() -> dict[str, str]:
    """Return a basic service welcome payload."""
    return {"message": f"{settings.app_name} is running."}
