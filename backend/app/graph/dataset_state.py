"""LangGraph state for dataset upload orchestration."""

from typing import TypedDict
import pandas as pd

from app.schemas.file import AIAnalysisResult, AnalysisTask, DatasetUploadResponse, ExecutionResult


class DatasetGraphState(TypedDict):
    """Mutable state for the dataset analysis graph."""

    file_name: str
    content: bytes
    question: str | None
    memory_context: dict[str, object] | None
    dataframe: pd.DataFrame | None
    file_info: DatasetUploadResponse | None
    schema: list[dict[str, str | int | bool]]
    ai_analysis: AIAnalysisResult | None
    tasks: list[AnalysisTask]
    planner_plan: dict[str, object] | None
    planner_status: str | None
    planner_issues: list[dict[str, object]]
    normalized_intent: str | None
    execution_stages: list[dict[str, object]]
    execution_order: list[str]
    dependency_count: int
    execution_supported: bool
    task_execution_records: list[dict[str, object]]
    stage_execution_records: list[dict[str, object]]
    execution_summary: dict[str, object] | None
    failed_task_ids: list[str]
    blocked_task_ids: list[str]
    confirmation_required_task_ids: list[str]
    execution_results: list[ExecutionResult]
    planner_failed: bool
    execution_failed: bool
    error_stage: str | None
    error_reason: str | None
    node_status: dict[str, str]
    trace_log: list[dict[str, object]]
    execution_path: list[str]
    fallback_used: bool
    fallback_reason: str | None
