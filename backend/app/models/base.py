"""SQLAlchemy base definitions.

.. note::

    The sync ``engine`` and ``SessionLocal`` below are **legacy**
    kept for backward compatibility.  New code should use the async
    engine and session factory from ``app.core.database`` together
    with the async repository layer.

    Engine creation is **lazy** — ``create_engine`` is only called
    when ``engine`` or ``SessionLocal`` is first accessed, so the
    module remains importable even without the database driver.
"""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.core.config import get_settings


class Base(DeclarativeBase):
    """Base declarative class shared by all ORM models."""


# ── Legacy sync engine (lazy, backward compatible) ────────────────
_engine = None
_SessionLocal = None


def _init_engine():
    """Lazily create the sync engine and session factory."""
    global _engine, _SessionLocal
    if _engine is None:
        settings = get_settings()
        _engine = create_engine(settings.database_url, pool_pre_ping=True)
        _SessionLocal = sessionmaker(
            bind=_engine, autoflush=False, autocommit=False
        )


def __getattr__(name: str):
    """Lazy module-level attributes to avoid a DB driver import at import time."""
    if name == "engine":
        _init_engine()
        return _engine
    if name == "SessionLocal":
        _init_engine()
        return _SessionLocal
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

