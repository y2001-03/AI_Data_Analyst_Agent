"""Dataset understanding helper — delegates LLM calls to the configured provider.

.. note::

    This module **no longer contains any DashScope SDK or HTTP code**.
    All LLM transport is handled by the ``LLMProvider`` abstraction in
    ``app.core.llm``, satisfying the Dependency Inversion Principle.
"""

from __future__ import annotations

import json

from app.core.exceptions import AppException
from app.core.llm import ChatMessage, get_llm_provider
from app.schemas.file import DatasetUploadResponse


class DashScopeService:
    """Thin wrapper around the LLM provider for dataset understanding.

    Retained for backward compatibility with ``DataUnderstandingService``.
    All HTTP / auth / retry logic lives in the provider — this class only
    owns the prompt template and the mock-data fallback (business logic).
    """

    def __init__(self) -> None:
        self._provider = get_llm_provider()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_dataset(self, file_info: DatasetUploadResponse) -> dict[str, object]:
        """Analyze dataset metadata via the LLM provider, or return mock data.

        Args:
            file_info: Parsed dataset metadata from ``FileService``.

        Returns:
            A dict with keys ``summary`` (str) and ``suggestions`` (list[str]).
        """
        messages = [
            ChatMessage(role="system", content=self._system_prompt()),
            ChatMessage(role="user", content=self._build_prompt(file_info)),
        ]
        try:
            return self._provider.structured_output(messages)
        except AppException:
            raise
        except Exception as exc:
            raise AppException(str(exc), 502) from exc

    def _mock_response(self, file_info: DatasetUploadResponse) -> dict[str, object]:
        """Return deterministic mock output when no API key is configured."""
        column_names = [column.name for column in file_info.columns[:3]]
        joined_names = ", ".join(column_names) if column_names else "unknown columns"
        summary = (
            f"This {file_info.file_type.upper()} dataset contains "
            f"{file_info.row_count} rows and {file_info.column_count} columns. "
            f"Key fields include {joined_names}."
        )
        suggestions = [
            "Check missing values and outliers in important columns.",
            "Analyze metric trends by time or category dimensions.",
            "Compare aggregated statistics across key business fields.",
        ]
        return {
            "summary": summary,
            "suggestions": suggestions,
        }

    # ------------------------------------------------------------------
    # Prompt templates (business logic — does not belong in the provider)
    # ------------------------------------------------------------------

    def _build_prompt(self, file_info: DatasetUploadResponse) -> str:
        """Serialize dataset metadata into the model prompt."""
        payload = {
            "file_name": file_info.file_name,
            "file_type": file_info.file_type,
            "row_count": file_info.row_count,
            "column_count": file_info.column_count,
            "columns": [column.model_dump() for column in file_info.columns],
            "preview": file_info.preview,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _system_prompt(self) -> str:
        """Return the system prompt for dataset understanding."""
        return (
            "You are a data understanding assistant. "
            "Given dataset schema and preview rows, respond with JSON only. "
            "Use keys: summary and suggestions. "
            "summary must be a concise dataset description in Chinese. "
            "suggestions must be an array of 3 to 5 Chinese analysis suggestions."
        )
