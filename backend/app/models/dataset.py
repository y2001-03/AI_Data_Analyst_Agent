"""Dataset persistence model."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import BigInteger, DateTime, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


def _utcnow() -> datetime:
    """Return the current UTC datetime (timezone-aware)."""
    return datetime.now(timezone.utc)


class Dataset(Base):
    """Uploaded dataset metadata stored in PostgreSQL."""

    __tablename__ = "datasets"

    # ── Identity ──────────────────────────────────────────────────
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    """UUID v4 primary key."""

    # ── File metadata ─────────────────────────────────────────────
    filename: Mapped[str] = mapped_column(
        String(512), nullable=False, comment="Original uploaded filename"
    )
    file_path: Mapped[str | None] = mapped_column(
        String(1024), nullable=True, comment="Server-side storage path (future use)"
    )
    file_size: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0, comment="File size in bytes"
    )
    file_type: Mapped[str] = mapped_column(
        String(16), nullable=False, default="csv", comment="File extension: csv | xlsx"
    )

    # ── Schema snapshot ───────────────────────────────────────────
    row_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="Number of data rows"
    )
    column_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="Number of columns"
    )
    schema_info: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True, comment="Column profile: [{name, data_type, …}]"
    )

    # ── Timestamps ────────────────────────────────────────────────
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, comment="Record creation time (UTC)"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        onupdate=_utcnow,
        comment="Last update time (UTC)",
    )

    def __repr__(self) -> str:
        return (
            f"<Dataset id={self.id!r} filename={self.filename!r} "
            f"rows={self.row_count} cols={self.column_count}>"
        )
