"""Analysis repository — async persistence for sessions and results."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.analysis import AnalysisResult, AnalysisSession
from app.repositories.base import BaseRepository


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AnalysisSessionRepository(BaseRepository[AnalysisSession]):
    """Async CRUD + domain queries for ``AnalysisSession``."""

    model = AnalysisSession

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    async def list_by_dataset(self, dataset_id: str, limit: int = 50) -> list[AnalysisSession]:
        """Return sessions for a given dataset, newest first."""
        return await self.list(
            AnalysisSession.dataset_id == dataset_id,
            order_by=AnalysisSession.created_at.desc(),
            limit=limit,
        )

    async def mark_completed(self, session_id: str) -> AnalysisSession | None:
        """Transition a session status to 'completed'."""
        session = await self.get(session_id)
        if session is None:
            return None
        session.status = "completed"
        session.completed_at = _utcnow()
        await self.session.flush()
        return session

    async def mark_failed(self, session_id: str) -> AnalysisSession | None:
        """Transition a session status to 'failed'."""
        session = await self.get(session_id)
        if session is None:
            return None
        session.status = "failed"
        session.completed_at = _utcnow()
        await self.session.flush()
        return session


class AnalysisResultRepository(BaseRepository[AnalysisResult]):
    """Async CRUD for ``AnalysisResult``."""

    model = AnalysisResult

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    async def get_by_session(self, session_id: str) -> AnalysisResult | None:
        """Return the result for a given session (one-to-one)."""
        stmt = (
            select(AnalysisResult)
            .where(AnalysisResult.session_id == session_id)
            .limit(1)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
