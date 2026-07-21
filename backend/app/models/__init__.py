"""Database models package.

- ``base.py`` — DeclarativeBase + legacy sync engine (backward compat).
- ``dataset.py`` — ``Dataset`` model (uploaded file metadata).
- ``analysis.py`` — ``AnalysisSession`` + ``AnalysisResult`` models.

For new code, prefer ``from app.core.database import get_async_db``
and the async repository layer.
"""

from app.models.analysis import AnalysisResult, AnalysisSession
from app.models.base import Base
from app.models.dataset import Dataset

__all__ = [
    "AnalysisResult",
    "AnalysisSession",
    "Base",
    "Dataset",
]
