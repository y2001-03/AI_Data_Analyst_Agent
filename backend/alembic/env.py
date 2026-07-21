"""
Alembic async migration environment.

Reads the database URL from the application settings (via
``app.core.config``) and runs migrations against an async engine.

Usage::

    cd backend
    alembic upgrade head          # apply all pending migrations
    alembic revision --autogenerate -m "add new table"
"""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import create_async_engine

# ── Alembic Config object ──────────────────────────────────────────
config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ── Model metadata (for autogenerate) ─────────────────────────────
# Import *every* model so Base.metadata is fully populated.
from app.models.base import Base  # noqa: E402
from app.models.dataset import Dataset  # noqa: E402, F401
from app.models.analysis import AnalysisSession, AnalysisResult  # noqa: E402, F401

target_metadata = Base.metadata

# ── Database URL (derived from app settings) ──────────────────────
from app.core.config import get_settings  # noqa: E402

settings = get_settings()
SYNC_URL = settings.database_url
ASYNC_URL = SYNC_URL.replace(
    "postgresql+psycopg://", "postgresql+psycopg_async://", 1
)


# ═══════════════════════════════════════════════════════════════════
# Runners
# ═══════════════════════════════════════════════════════════════════


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emit SQL to stdout)."""
    context.configure(
        url=SYNC_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):
    """Execute migrations with the given connection."""
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Create an async engine and run migrations."""
    connectable = create_async_engine(ASYNC_URL, poolclass=pool.NullPool)
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode (against a real database)."""
    asyncio.run(run_async_migrations())


# ── Dispatch ──────────────────────────────────────────────────────
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
