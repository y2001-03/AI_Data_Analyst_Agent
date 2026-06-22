"""In-memory dataset context storage for chat-driven analysis."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from app.core.exceptions import AppException


@dataclass(frozen=True)
class StoredDataset:
    """Serializable dataset reference kept for later chat analysis."""

    dataset_id: str
    file_name: str
    content: bytes
    dataframe: pd.DataFrame
    schema: list[dict[str, object]]
    preview: list[dict[str, object]]
    created_at: str


class DatasetContextService:
    """Store uploaded datasets for subsequent question-driven analysis."""

    _datasets: dict[str, StoredDataset] = {}
    _latest_by_session: dict[str, str] = {}

    def save_dataset(
        self,
        dataset_id: str,
        file_name: str,
        content: bytes,
        dataframe: pd.DataFrame,
        schema: list[dict[str, object]],
        preview: list[dict[str, object]],
        session_id: str | None = None,
    ) -> str:
        """Persist one uploaded dataset under a stable reference."""
        self._datasets[dataset_id] = StoredDataset(
            dataset_id=dataset_id,
            file_name=file_name,
            content=content,
            dataframe=dataframe.copy(),
            schema=schema,
            preview=preview,
            created_at=datetime.utcnow().isoformat(),
        )
        if session_id:
            self._latest_by_session[session_id] = dataset_id
        return dataset_id

    def get_dataset(self, dataset_id: str) -> StoredDataset:
        """Resolve a previously uploaded dataset reference."""
        dataset = self._datasets.get(dataset_id)
        if dataset is None:
            raise AppException(
                "Dataset context was not found. Please upload the file again before asking a question.",
                status_code=404,
            )
        return dataset

    def get_latest_dataset_id(self, session_id: str) -> str | None:
        """Return the most recently uploaded dataset id for a session."""
        return self._latest_by_session.get(session_id)
