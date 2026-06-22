"""File schemas."""

from pydantic import BaseModel


class DatasetColumnProfile(BaseModel):
    """Detected profile information for a single dataset column."""

    name: str
    data_type: str
    missing_count: int
    unique_values: int


class DatasetUploadResponse(BaseModel):
    """Structured dataset understanding response."""

    file_name: str
    file_type: str
    row_count: int
    column_count: int
    preview: list[dict[str, str | int | float | bool | None]]
    columns: list[DatasetColumnProfile]


class AnalysisTask(BaseModel):
    """Planned analysis task."""

    task_name: str
    reasoning: str
    expected_output: str
    type: str | None = None
    params: dict[str, object] | None = None


class AIAnalysisResult(BaseModel):
    """AI-generated dataset understanding output."""

    summary: str
    suggestions: list[str]
    tasks: list[AnalysisTask]


class DatasetUnderstandingResponse(BaseModel):
    """Dataset upload response with AI analysis."""

    file_info: DatasetUploadResponse
    ai_analysis: AIAnalysisResult


class ExecutionChart(BaseModel):
    """Chart-friendly payload for frontend rendering."""

    chart_type: str
    x: list[str]
    y: list[float | int]


class ExecutionResult(BaseModel):
    """Structured execution output for a single task."""

    task_name: str
    type: str
    data: dict[str, object]
    chart: ExecutionChart | None = None


class ExecutionResponse(BaseModel):
    """Structured execution layer response."""

    execution_results: list[ExecutionResult]


class TraceEntry(BaseModel):
    """Single trace log entry for graph execution."""

    node: str
    status: str
    input_summary: dict[str, object]
    output_summary: dict[str, object]


class TraceResponse(BaseModel):
    """Observability payload for graph execution."""

    execution_path: list[str]
    node_status: dict[str, str]
    trace_log: list[TraceEntry]
    fallback_used: bool
    fallback_reason: str | None


class DebugFallbackSummary(BaseModel):
    """Fallback summary flags for debug mode."""

    understand: bool
    plan: bool


class DebugResponse(BaseModel):
    """Debug payload for interview demo mode."""

    execution_path: list[str]
    node_status: dict[str, str]
    trace_log: list[TraceEntry]
    fallback_summary: DebugFallbackSummary


class DatasetExecutionResponse(BaseModel):
    """Full dataset agent flow response."""

    dataset_id: str | None = None
    file_info: DatasetUploadResponse
    ai_analysis: AIAnalysisResult
    execution_results: list[ExecutionResult]
    trace: TraceResponse
    debug: DebugResponse
