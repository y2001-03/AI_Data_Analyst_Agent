"""
Async repository base class.

Provides standard CRUD operations that every concrete repository
inherits.  All methods are ``async def`` and accept an ``AsyncSession``.

Generic parameter ``T`` is the SQLAlchemy ORM model type (must be a
subclass of ``app.models.base.Base``).
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import Select

T = TypeVar("T")


class BaseRepository(Generic[T]):
    """Async CRUD base for SQLAlchemy models.

    Usage::

        class DatasetRepository(BaseRepository[Dataset]):
            model = Dataset

            # Add domain-specific queries here.
    """

    # ── Subclass must override ────────────────────────────────────
    model: type[T]

    # ── Constructor ───────────────────────────────────────────────

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    # ── Create ────────────────────────────────────────────────────

    async def create(self, instance: T) -> T:
        """Persist a new instance and return it (with generated fields)."""
        self.session.add(instance)
        await self.session.flush()
        return instance

    # ── Read ──────────────────────────────────────────────────────

    async def get(self, id_value: Any) -> T | None:
        """Retrieve a single entity by primary key."""
        return await self.session.get(self.model, id_value)

    async def get_one(self, *criterion) -> T | None:
        """Retrieve the first entity matching ``criterion``."""
        stmt: Select = select(self.model).where(*criterion).limit(1)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def list(
        self,
        *criterion,
        offset: int = 0,
        limit: int = 50,
        order_by: Any = None,
    ) -> list[T]:
        """Return a page of entities matching ``criterion``."""
        stmt: Select = select(self.model).where(*criterion)
        if order_by is not None:
            stmt = stmt.order_by(order_by)
        stmt = stmt.offset(offset).limit(limit)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def count(self, *criterion) -> int:
        """Return the count of entities matching ``criterion``."""
        stmt = select(func.count()).select_from(self.model).where(*criterion)
        result = await self.session.execute(stmt)
        return result.scalar_one()

    # ── Update ────────────────────────────────────────────────────

    async def update(self, instance: T, **values: Any) -> T:
        """Update scalar fields on ``instance`` and flush."""
        for key, value in values.items():
            if hasattr(instance, key):
                setattr(instance, key, value)
        await self.session.flush()
        return instance

    # ── Delete ────────────────────────────────────────────────────

    async def delete(self, instance: T) -> None:
        """Remove ``instance`` from the database."""
        await self.session.delete(instance)
        await self.session.flush()

    async def delete_by_id(self, id_value: Any) -> bool:
        """Delete by primary key.  Returns ``True`` if a row was deleted."""
        instance = await self.get(id_value)
        if instance is None:
            return False
        await self.delete(instance)
        return True
