"""Strongly typed Planner Contract models."""

from __future__ import annotations

import json
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.tool_capability import RiskLevel, _ensure_json_value

PlannerTaskSource = Literal["llm", "fallback", "repaired", "system"]
PlannerSource = Literal["llm", "fallback", "repaired"]
PlannerValidationStatus = Literal["pending", "valid", "invalid"]
PlannerPlanStatus = Literal["valid", "invalid", "clarification_required", "unsupported"]
PlannerIssueSeverity = Literal["error", "warning", "info"]
PlannerInputSourceType = Literal["original_dataset", "task_output", "planner_context", "literal"]

PlannerIssueCode = Literal[
    "UNKNOWN_TASK_TYPE",
    "UNKNOWN_TOOL",
    "TASK_TOOL_MISMATCH",
    "DUPLICATE_TASK_ID",
    "MISSING_REQUIRED_PARAMETER",
    "UNKNOWN_PARAMETER",
    "INVALID_PARAMETER_TYPE",
    "INVALID_PARAMETER_VALUE",
    "UNKNOWN_FIELD",
    "INVALID_FIELD_TYPE",
    "EMPTY_TASK_LIST",
    "INVALID_CONFIDENCE",
    "RISK_METADATA_MISMATCH",
    "NON_JSON_SAFE_PARAMS",
    "UNSUPPORTED_REQUEST",
    "CLARIFICATION_REQUIRED",
    "UNKNOWN_DEPENDENCY_TASK",
    "SELF_TASK_DEPENDENCY",
    "DUPLICATE_TASK_DEPENDENCY",
    "CYCLIC_TASK_DEPENDENCY",
    "INVALID_INPUT_BINDING",
    "UNKNOWN_INPUT_SOURCE_TASK",
    "INPUT_SOURCE_NOT_DEPENDENCY",
    "DUPLICATE_INPUT_BINDING",
    "UNSUPPORTED_TASK_OUTPUT_BINDING",
    "CONDITIONAL_EXECUTION_UNSUPPORTED",
    "EXECUTION_PLAN_INCOMPLETE",
]


class PlannerInputBinding(BaseModel):
    """Declarative task input source binding; it is not executed in this step."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_name: str = Field(min_length=1)
    source_type: PlannerInputSourceType = "original_dataset"
    source_task_id: str | None = None
    source_path: str | None = None
    literal_value: Any = None
    required: bool = True
    description: str = ""

    @model_validator(mode="after")
    def validate_binding_shape(self) -> Self:
        if self.source_type == "original_dataset":
            if self.source_task_id is not None:
                raise ValueError("original_dataset binding must not define source_task_id")
            if self.source_path not in {None, "$", "dataset"}:
                raise ValueError("original_dataset binding source_path must be empty or a stable dataset alias")
        elif self.source_type == "task_output":
            if not self.source_task_id:
                raise ValueError("task_output binding requires source_task_id")
            if not self.source_path:
                raise ValueError("task_output binding requires source_path")
        elif self.source_type == "planner_context":
            if not self.source_path:
                raise ValueError("planner_context binding requires source_path")
            if self.source_task_id is not None:
                raise ValueError("planner_context binding must not define source_task_id")
        elif self.source_type == "literal":
            if self.source_task_id is not None:
                raise ValueError("literal binding must not define source_task_id")
            _ensure_json_value(self.literal_value, f"input binding '{self.input_name}' literal_value")
        json.dumps(self.model_dump(), allow_nan=False)
        return self


class PlannerExecutionStage(BaseModel):
    """Stable execution stage derived from the task dependency DAG."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage_id: str = Field(min_length=1)
    order: int = Field(ge=1)
    task_ids: tuple[str, ...]
    parallelizable: bool
    risk_level: RiskLevel
    requires_confirmation: bool = False
    description: str = ""

    @model_validator(mode="after")
    def validate_stage(self) -> Self:
        if not self.task_ids:
            raise ValueError("execution stage must contain at least one task")
        if len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("execution stage task_ids must be unique")
        json.dumps(self.model_dump(), allow_nan=False)
        return self


class PlannerValidationIssue(BaseModel):
    """Structured non-sensitive validation issue for one plan or task."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: PlannerIssueCode
    message: str = Field(min_length=1)
    severity: PlannerIssueSeverity
    task_id: str | None = None
    field: str | None = None
    parameter_name: str | None = None
    expected: Any = None
    actual: Any = None
    repairable: bool = False

    @model_validator(mode="after")
    def validate_json_payload(self) -> Self:
        _ensure_json_value(self.expected, "issue.expected")
        _ensure_json_value(self.actual, "issue.actual")
        return self


class PlannerDiagnostics(BaseModel):
    """Planner behavior summary safe for trace/debug output."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: PlannerSource
    raw_task_count: int = 0
    normalized_task_count: int = 0
    validated_task_count: int = 0
    invalid_task_count: int = 0
    capability_catalog_version: str = "1.0"
    fallback_used: bool = False
    normalization_warnings: list[str] = Field(default_factory=list)
    parse_failure: str | None = None
    tool_selection_rule: str | None = None
    missing_information: list[str] = Field(default_factory=list)
    dependency_count: int = 0
    stage_count: int = 0
    maximum_parallel_width: int = 0
    topological_sort_applied: bool = False
    execution_supported: bool = True
    unsupported_binding_count: int = 0


class PlannerRequestColumn(BaseModel):
    """Dataset column projection available to the planner validator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    type: str = Field(min_length=1)


class PlannerRequestContext(BaseModel):
    """Non-sensitive planner request context."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    question: str | None = None
    dataset_id: str | None = None
    file_name: str | None = None
    columns: tuple[PlannerRequestColumn, ...] = ()
    row_count: int | None = None
    capability_context: str = ""
    request_id: str | None = None
    trace_id: str | None = None


class PlannerTask(BaseModel):
    """One normalized planner task before execution adaptation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(min_length=1)
    task_type: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""
    confidence: float = Field(ge=0.0, le=1.0)
    source: PlannerTaskSource
    risk_level: RiskLevel
    requires_confirmation: bool
    validation_status: PlannerValidationStatus = "pending"
    validation_issues: tuple[PlannerValidationIssue, ...] = ()
    depends_on: tuple[str, ...] = ()
    input_bindings: tuple[PlannerInputBinding, ...] = ()
    execution_group: int | None = None
    execution_order: int | None = None

    @model_validator(mode="after")
    def validate_json_payload(self) -> Self:
        _ensure_json_value(self.params, f"task '{self.task_id}' params")
        json.dumps(self.model_dump(), allow_nan=False)
        return self


class PlannerPlan(BaseModel):
    """Validated planner output contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    plan_id: str = Field(min_length=1)
    user_intent: str = ""
    normalized_intent: str = "unknown"
    tasks: tuple[PlannerTask, ...] = ()
    status: PlannerPlanStatus
    planner_source: PlannerSource
    capability_catalog_version: str = "1.0"
    warnings: list[str] = Field(default_factory=list)
    validation_issues: tuple[PlannerValidationIssue, ...] = ()
    diagnostics: PlannerDiagnostics
    metadata: dict[str, Any] = Field(default_factory=dict)
    execution_stages: tuple[PlannerExecutionStage, ...] = ()
    execution_order: tuple[str, ...] = ()
    has_dependencies: bool = False
    parallelizable: bool = False
    execution_supported: bool = True

    @model_validator(mode="after")
    def validate_plan(self) -> Self:
        ids = [task.task_id for task in self.tasks]
        if len(set(ids)) != len(ids):
            raise ValueError("task_id must be unique within a plan")
        if self.status == "valid" and not self.tasks:
            raise ValueError("valid plan requires at least one task")
        if self.status == "valid":
            for task in self.tasks:
                if task.validation_status != "valid":
                    raise ValueError("valid plan cannot contain non-valid tasks")
                if any(issue.severity == "error" for issue in task.validation_issues):
                    raise ValueError("valid plan cannot contain error issues")
        if not self.tasks and self.status not in {"unsupported", "clarification_required", "invalid"}:
            raise ValueError("empty task list requires unsupported, clarification_required, or invalid status")
        if len(set(self.execution_order)) != len(self.execution_order):
            raise ValueError("execution_order task ids must be unique")
        if self.status == "valid" and self.execution_supported and (self.execution_order or self.execution_stages):
            if set(self.execution_order) != set(ids):
                raise ValueError("valid executable plan requires execution_order to include every task exactly once")
            staged_ids = [task_id for stage in self.execution_stages for task_id in stage.task_ids]
            if set(staged_ids) != set(ids) or len(staged_ids) != len(ids):
                raise ValueError("valid executable plan requires every task in exactly one execution stage")
        _ensure_json_value(self.metadata, "plan.metadata")
        json.dumps(self.model_dump(), allow_nan=False)
        return self
