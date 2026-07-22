"""Dependency-aware execution records for planner-driven workflows."""

from __future__ import annotations

import json
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.tool_capability import _ensure_json_value

TaskExecutionStatus = Literal[
    "pending",
    "running",
    "succeeded",
    "failed",
    "blocked",
    "skipped",
    "confirmation_required",
]

StageExecutionStatus = Literal[
    "pending",
    "running",
    "succeeded",
    "partially_failed",
    "failed",
    "blocked",
    "confirmation_required",
]


class TaskExecutionRecord(BaseModel):
    """JSON-safe status record for one planned task execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(min_length=1)
    task_type: str = Field(min_length=1)
    stage_id: str | None = None
    execution_order: int | None = None
    status: TaskExecutionStatus
    started_at: str | None = None
    finished_at: str | None = None
    elapsed_ms: int | None = None
    dependency_task_ids: list[str] = Field(default_factory=list)
    blocked_by: list[str] = Field(default_factory=list)
    result_index: int | None = None
    error_code: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_json_safe(self) -> Self:
        _ensure_json_value(self.metadata, "task execution metadata")
        json.dumps(self.model_dump(), allow_nan=False)
        return self


class StageExecutionRecord(BaseModel):
    """JSON-safe status record for one execution stage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage_id: str = Field(min_length=1)
    order: int = Field(ge=1)
    task_ids: list[str] = Field(default_factory=list)
    status: StageExecutionStatus
    started_at: str | None = None
    finished_at: str | None = None
    elapsed_ms: int | None = None
    succeeded_count: int = 0
    failed_count: int = 0
    blocked_count: int = 0
    confirmation_required_count: int = 0

    @model_validator(mode="after")
    def validate_json_safe(self) -> Self:
        json.dumps(self.model_dump(), allow_nan=False)
        return self


class DependencyExecutionSummary(BaseModel):
    """Stable summary for dependency-aware execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    total_tasks: int = 0
    succeeded_tasks: int = 0
    failed_tasks: int = 0
    blocked_tasks: int = 0
    confirmation_required_tasks: int = 0
    total_stages: int = 0
    succeeded_stages: int = 0
    failed_stages: int = 0
    elapsed_ms: int = 0
    completed: bool = True
    partial_success: bool = False

    @model_validator(mode="after")
    def validate_json_safe(self) -> Self:
        json.dumps(self.model_dump(), allow_nan=False)
        return self
