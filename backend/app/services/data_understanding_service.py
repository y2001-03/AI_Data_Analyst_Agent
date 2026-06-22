"""Dataset understanding service."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from fastapi import UploadFile

from app.core.exceptions import AppException
from app.schemas.file import (
    AIAnalysisResult,
    DatasetColumnProfile,
    DatasetUnderstandingResponse,
    DatasetUploadResponse,
)
from app.services.analysis_planner_service import AnalysisPlannerService
from app.services.dashscope_service import DashScopeService
from app.services.file_service import FileService


class DataUnderstandingService:
    """Compose file parsing with AI dataset understanding."""

    request_timeout = 12
    max_retries = 2

    def __init__(self) -> None:
        self.file_service = FileService()
        self.dashscope_service = DashScopeService()
        self.analysis_planner_service = AnalysisPlannerService()

    async def analyze_upload(self, file: UploadFile) -> DatasetUnderstandingResponse:
        """Parse the file and generate AI dataset understanding."""
        file_info = await self.file_service.parse_upload(file)
        result = self.dashscope_service.analyze_dataset(file_info)
        ai_analysis = self._build_ai_analysis(result)
        ai_analysis.tasks = self.analysis_planner_service.plan_tasks(
            file_info,
            ai_analysis,
        )
        return DatasetUnderstandingResponse(
            file_info=file_info,
            ai_analysis=ai_analysis,
        )

    def understand_file_info(
        self,
        file_info: DatasetUploadResponse,
        question: str | None = None,
        memory_context: dict[str, object] | None = None,
    ) -> tuple[AIAnalysisResult, bool, str | None]:
        """Generate dataset understanding with retry, timeout, and fallback."""
        effective_question = question or self._memory_question(memory_context)
        last_reason: str | None = None
        for _ in range(self.max_retries + 1):
            try:
                result = self._call_dashscope_with_timeout(file_info)
                ai_analysis = self._build_ai_analysis(result)
                return self._apply_memory_focus(
                    self._apply_question_focus(ai_analysis, effective_question),
                    memory_context,
                ), False, None
            except FutureTimeoutError:
                last_reason = "timeout"
            except AppException as exc:
                last_reason = exc.message
            except Exception as exc:
                last_reason = str(exc)
        fallback = self._build_rule_based_analysis(file_info)
        fallback = self._apply_question_focus(fallback, effective_question)
        fallback = self._apply_memory_focus(fallback, memory_context)
        return fallback, True, last_reason or "timeout"

    def _apply_question_focus(
        self,
        ai_analysis: AIAnalysisResult,
        question: str | None,
    ) -> AIAnalysisResult:
        """Inject the user's question into dataset understanding when available."""
        if not question:
            return ai_analysis
        focused_suggestions = [
            f"Answer the user question directly: {question}",
            *ai_analysis.suggestions,
        ]
        return AIAnalysisResult(
            summary=f"{ai_analysis.summary} The current analysis question is: {question}",
            suggestions=focused_suggestions[:3],
            tasks=ai_analysis.tasks,
        )

    def _apply_memory_focus(
        self,
        ai_analysis: AIAnalysisResult,
        memory_context: dict[str, object] | None,
    ) -> AIAnalysisResult:
        """Inject lightweight previous context into the analysis summary."""
        if not memory_context:
            return ai_analysis
        memory_parts: list[str] = []
        previous_question = memory_context.get("last_question")
        dataset_info = memory_context.get("dataset_info")
        if isinstance(previous_question, str) and previous_question:
            memory_parts.append(f"Previous question: {previous_question}.")
        if isinstance(dataset_info, dict):
            file_name = dataset_info.get("file_name")
            row_count = dataset_info.get("row_count")
            if file_name:
                memory_parts.append(f"Recent dataset: {file_name}.")
            if row_count is not None:
                memory_parts.append(f"Recent dataset rows: {row_count}.")
        if not memory_parts:
            return ai_analysis
        return AIAnalysisResult(
            summary=f"{ai_analysis.summary} {' '.join(memory_parts)}",
            suggestions=ai_analysis.suggestions,
            tasks=ai_analysis.tasks,
        )

    def _memory_question(self, memory_context: dict[str, object] | None) -> str | None:
        """Resolve the latest question from memory context."""
        if not memory_context:
            return None
        last_question = memory_context.get("last_question")
        return last_question if isinstance(last_question, str) and last_question else None

    def _call_dashscope_with_timeout(
        self,
        file_info: DatasetUploadResponse,
    ) -> dict[str, object]:
        """Call DashScope with a bounded timeout."""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.dashscope_service.analyze_dataset, file_info)
            return future.result(timeout=self.request_timeout)

    def _build_ai_analysis(self, result: dict[str, object]) -> AIAnalysisResult:
        """Validate the AI output shape."""
        summary = result.get("summary")
        suggestions = result.get("suggestions")
        if not isinstance(summary, str):
            raise AppException("DashScope response is missing summary.", 502)
        if not isinstance(suggestions, list):
            raise AppException("DashScope response is missing suggestions.", 502)
        normalized = [item for item in suggestions if isinstance(item, str)]
        if not normalized:
            raise AppException("DashScope suggestions are invalid.", 502)
        return AIAnalysisResult(summary=summary, suggestions=normalized, tasks=[])

    def _build_rule_based_analysis(
        self,
        file_info: DatasetUploadResponse,
    ) -> AIAnalysisResult:
        """Generate fallback understanding from schema and preview."""
        column_names = [column.name for column in file_info.columns]
        numeric_columns = self._pick_columns(file_info.columns, "number")
        datetime_columns = self._pick_columns(file_info.columns, "date")
        preview_hint = self._build_preview_hint(file_info.preview)
        summary_parts = [
            f"This dataset contains {file_info.row_count} rows and {file_info.column_count} columns.",
            f"Columns: {', '.join(column_names)}." if column_names else "Columns are unavailable.",
        ]
        if numeric_columns:
            summary_parts.append(f"Numeric fields include {', '.join(numeric_columns)}.")
        if datetime_columns:
            summary_parts.append(f"Datetime-like fields include {', '.join(datetime_columns)}.")
        if preview_hint:
            summary_parts.append(f"Preview insight: {preview_hint}.")
        suggestions = [
            "Review missing values and uniqueness across important fields.",
            "Compute descriptive statistics for numeric columns and compare key segments.",
            "Analyze time-based trends or grouped summaries where applicable.",
        ]
        return AIAnalysisResult(
            summary=" ".join(summary_parts),
            suggestions=suggestions,
            tasks=[],
        )

    def _pick_columns(
        self,
        columns: list[DatasetColumnProfile],
        column_family: str,
    ) -> list[str]:
        """Pick columns that match a simple semantic family."""
        picked: list[str] = []
        for column in columns:
            data_type = column.data_type.lower()
            name = column.name.lower()
            if column_family == "number" and any(
                token in data_type for token in ("int", "float", "double", "number")
            ):
                picked.append(column.name)
            if column_family == "date" and (
                "date" in name or "time" in name or "datetime" in data_type
            ):
                picked.append(column.name)
        return picked

    def _build_preview_hint(
        self,
        preview: list[dict[str, str | int | float | bool | None]],
    ) -> str:
        """Create a short preview-based hint for fallback summaries."""
        if not preview:
            return ""
        first_row = preview[0]
        keys = list(first_row.keys())[:3]
        if not keys:
            return ""
        values = [f"{key}={first_row[key]}" for key in keys]
        return "first row shows " + ", ".join(values)
