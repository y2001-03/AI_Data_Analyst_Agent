"""Repository package — async data-access layer.

All repositories follow the pattern:

    repo = SomeRepository(async_session)
    entity = await repo.get(id)

For new code prefer the async repositories; the legacy sync
``BaseRepository`` in ``base.py`` is kept for backward compatibility.
"""

from app.repositories.analysis_repository import (
    AnalysisResultRepository,
    AnalysisSessionRepository,
)
from app.repositories.base import BaseRepository
from app.repositories.dataset_repository import DatasetRepository

__all__ = [
    "AnalysisResultRepository",
    "AnalysisSessionRepository",
    "BaseRepository",
    "DatasetRepository",
]
