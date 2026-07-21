"""Dataset repository — async persistence for uploaded dataset metadata."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.dataset import Dataset
from app.repositories.base import BaseRepository


class DatasetRepository(BaseRepository[Dataset]):
    """Async CRUD + domain queries for ``Dataset``."""

    model = Dataset

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    async def get_by_filename(self, filename: str) -> Dataset | None:
        """Find a dataset by its original filename."""
        stmt = select(Dataset).where(Dataset.filename == filename).limit(1)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_recent(self, limit: int = 20) -> list[Dataset]:
        """Return the most recently uploaded datasets."""
        return await self.list(
            order_by=Dataset.created_at.desc(),
            limit=limit,
        )
