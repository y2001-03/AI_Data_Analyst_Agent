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
]


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
    normalization_warnings: tuple[str, ...] = ()
    parse_failure: str | None = None
    tool_selection_rule: str | None = None
    missing_information: tuple[str, ...] = ()


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
    warnings: tuple[str, ...] = ()
    validation_issues: tuple[PlannerValidationIssue, ...] = ()
    diagnostics: PlannerDiagnostics
    metadata: dict[str, Any] = Field(default_factory=dict)

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
        _ensure_json_value(self.metadata, "plan.metadata")
        json.dumps(self.model_dump(), allow_nan=False)
        return self
