"""
Database session management — async-first, with sync fallback.

Provides:
- ``async_engine`` — SQLAlchemy async engine (psycopg_async driver).
- ``async_session_factory`` — ``async_sessionmaker`` for creating sessions.
- ``get_async_db()`` — FastAPI dependency yielding an ``AsyncSession``.

Usage (preferred)::

    from app.core.database import get_async_db

    @router.get("/items")
    async def list_items(db: AsyncSession = Depends(get_async_db)):
        ...

.. note::

    Engine creation is **lazy** — ``create_async_engine`` is called on
    first use, so the module is importable even without a database
    driver installed (useful for CI / testing / code inspection).
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# ── Lazy engine + session factory ─────────────────────────────────
_async_engine = None
_async_session_factory = None


def _init_async_engine():
    """Create the async engine and session factory (idempotent)."""
    global _async_engine, _async_session_factory
    if _async_engine is not None:
        return
    settings = get_settings()
    async_url = settings.database_url.replace(
        "postgresql+psycopg://", "postgresql+psycopg_async://", 1
    )
    logger.info("Creating async engine | url=%s", async_url)
    _async_engine = create_async_engine(
        async_url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        echo=False,
    )
    _async_session_factory = async_sessionmaker(
        bind=_async_engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )


def __getattr__(name: str):
    """Lazy module-level attributes."""
    if name == "async_engine":
        _init_async_engine()
        return _async_engine
    if name == "async_session_factory":
        _init_async_engine()
        return _async_session_factory
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def create_async_session() -> AsyncSession:
    """Create and return a new ``AsyncSession``.

    The caller is responsible for closing the session::

        session = create_async_session()
        try:
            ...
            await session.commit()
        finally:
            await session.close()

    For FastAPI routes, prefer ``get_async_db`` (dependency injection).
    """
    _init_async_engine()
    return _async_session_factory()  # type: ignore[misc]


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency — yield an ``AsyncSession``.

    The session is automatically closed when the request finishes,
    even if an exception is raised.
    """
    _init_async_engine()
    async with _async_session_factory() as session:  # type: ignore[misc]
        try:
            yield session
        finally:
            await session.close()
