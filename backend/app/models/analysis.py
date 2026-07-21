"""Analysis session and result persistence models."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import DateTime, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AnalysisSession(Base):
    """One question-driven analysis run against a dataset.

    Relationship:  Dataset (1) ──< (N) AnalysisSession
    """

    __tablename__ = "analysis_sessions"

    # ── Identity ──────────────────────────────────────────────────
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )

    # ── Foreign key ───────────────────────────────────────────────
    dataset_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("datasets.id", ondelete="SET NULL"),
        nullable=True,
        comment="FK → datasets.id; NULL if dataset was deleted",
    )

    # ── Analysis context ──────────────────────────────────────────
    question: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="User's natural-language question"
    )
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="pending",
        comment="pending | running | completed | failed",
    )
    workflow_trace_id: Mapped[str | None] = mapped_column(
        String(36),
        nullable=True,
        comment="Correlates with the LangGraph trace_id in logs",
    )

    # ── Timestamps ────────────────────────────────────────────────
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Set when status transitions to 'completed' or 'failed'",
    )

    # ── Relationships ─────────────────────────────────────────────
    result = relationship(
        "AnalysisResult",
        back_populates="session",
        uselist=False,
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return (
            f"<AnalysisSession id={self.id!r} status={self.status!r} "
            f"dataset_id={self.dataset_id!r}>"
        )


class AnalysisResult(Base):
    """The structured output of one analysis session.

    Relationship:  AnalysisSession (1) ── (1) AnalysisResult
    """

    __tablename__ = "analysis_results"

    # ── Identity ──────────────────────────────────────────────────
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )

    # ── Foreign key ───────────────────────────────────────────────
    session_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("analysis_sessions.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        comment="FK → analysis_sessions.id; one-to-one",
    )

    # ── Payload ───────────────────────────────────────────────────
    result: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, comment="Full analysis output as JSON"
    )

    # ── Timestamps ────────────────────────────────────────────────
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )

    # ── Relationships ─────────────────────────────────────────────
    session = relationship("AnalysisSession", back_populates="result")

    def __repr__(self) -> str:
        return f"<AnalysisResult id={self.id!r} session_id={self.session_id!r}>"
