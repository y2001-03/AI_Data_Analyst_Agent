"""
Database configuration.

Currently only PostgreSQL via SQLAlchemy / psycopg is supported.
Add connection-pool, replica, or read/write-split fields here when needed.

This is a **mixin** — see ``app.core.config.app`` for the pattern
rationale.
"""

from pydantic import Field


class DatabaseConfigMixin:
    """Database-connection fields merged flat into ``AppSettings``."""

    database_url: str = Field(
        default="postgresql+psycopg://postgres:postgres@localhost:5432/ai_data_analyst",
        alias="DATABASE_URL",
        description="SQLAlchemy-compatible database connection string.",
    )
