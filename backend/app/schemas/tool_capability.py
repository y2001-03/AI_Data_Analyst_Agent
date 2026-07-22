"""Strongly typed capability metadata for dataframe-backed tools."""

from __future__ import annotations

import json
import math
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

ParameterType = Literal["string", "integer", "number", "boolean", "array", "object"]
FieldRole = Literal["time", "target", "value", "group", "columns", "identifier", "query"]
FieldType = Literal["numeric", "categorical", "text", "boolean", "datetime", "timedelta", "any"]
RiskLevel = Literal["read_only", "preview_only", "mutating_copy", "external_side_effect"]


def _ensure_json_value(value: Any, path: str) -> None:
    """Reject values that cannot be represented by strict JSON."""
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain NaN or Infinity")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _ensure_json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} object keys must be strings")
            _ensure_json_value(item, f"{path}.{key}")
        return
    raise ValueError(f"{path} contains a non-JSON value: {type(value).__name__}")


class ToolParameterSpec(BaseModel):
    """Describe one supported tool parameter without replacing runtime validation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    type: ParameterType
    required: bool = False
    default: Any = None
    description: str = Field(min_length=1)
    allowed_values: tuple[Any, ...] | None = None
    minimum: float | None = None
    maximum: float | None = None
    aliases: tuple[str, ...] = ()
    sensitive: bool = False

    @model_validator(mode="after")
    def validate_constraints(self) -> Self:
        if self.minimum is not None and not math.isfinite(self.minimum):
            raise ValueError("minimum must be finite")
        if self.maximum is not None and not math.isfinite(self.maximum):
            raise ValueError("maximum must be finite")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("minimum must be less than or equal to maximum")
        _ensure_json_value(self.default, f"parameter '{self.name}' default")
        if self.allowed_values is not None:
            if not self.allowed_values:
                raise ValueError("allowed_values must not be empty")
            for index, value in enumerate(self.allowed_values):
                _ensure_json_value(value, f"parameter '{self.name}' allowed_values[{index}]")
            if self.default is not None and self.default not in self.allowed_values:
                raise ValueError("default must be included in allowed_values")
        if len(set(self.aliases)) != len(self.aliases):
            raise ValueError("parameter aliases must be unique")
        return self

class ToolFieldRequirement(BaseModel):
    """Describe a dataframe field role using planner-facing logical types."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    parameter_name: str | None = None
    role: FieldRole
    accepted_types: tuple[FieldType, ...]
    required: bool
    multiple: bool = False
    exclude_types: tuple[FieldType, ...] = ()
    description: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_types(self) -> Self:
        if not self.accepted_types:
            raise ValueError("accepted_types must not be empty")
        if len(set(self.accepted_types)) != len(self.accepted_types):
            raise ValueError("accepted_types must be unique")
        if len(set(self.exclude_types)) != len(self.exclude_types):
            raise ValueError("exclude_types must be unique")
        if set(self.accepted_types) & set(self.exclude_types):
            raise ValueError("accepted_types and exclude_types must not overlap")
        return self


class ToolCapability(BaseModel):
    """Stable, JSON-safe capability contract for one registered tool."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_name: str = Field(min_length=1)
    task_types: tuple[str, ...]
    display_name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    category: str = Field(min_length=1)
    version: str = Field(default="1.0", min_length=1)
    parameters: tuple[ToolParameterSpec, ...] = ()
    required_parameters: tuple[str, ...] = ()
    optional_parameters: tuple[str, ...] = ()
    field_requirements: tuple[ToolFieldRequirement, ...] = ()
    supports_multiple_columns: bool
    supports_grouping: bool
    supports_streaming: bool = False
    mutates_dataframe: bool
    returns_modified_dataframe: bool
    provides_chart: bool = Field(
        description=(
            "True only when the tool result itself returns a directly usable chart payload "
            "and workflow visualization should not append an extra visualization task. "
            "This does not mean every visualizable task can generate a chart later."
        )
    )
    chart_types: tuple[str, ...] = ()
    risk_level: RiskLevel
    deterministic: bool
    limitations: tuple[str, ...]
    examples: tuple[dict[str, Any], ...] = ()

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        if not self.task_types or any(not task_type for task_type in self.task_types):
            raise ValueError("task_types must contain at least one non-empty task type")
        if len(set(self.task_types)) != len(self.task_types):
            raise ValueError("task_types must be unique")
        parameter_names = [parameter.name for parameter in self.parameters]
        if len(set(parameter_names)) != len(parameter_names):
            raise ValueError("parameter names must be unique")
        required = set(self.required_parameters)
        optional = set(self.optional_parameters)
        known = set(parameter_names)
        if required & optional:
            raise ValueError("required_parameters and optional_parameters must not overlap")
        if required | optional != known:
            raise ValueError("required_parameters and optional_parameters must classify every parameter")
        declared_required = {parameter.name for parameter in self.parameters if parameter.required}
        if required != declared_required:
            raise ValueError("required_parameters must match parameter required flags")
        for requirement in self.field_requirements:
            if requirement.parameter_name is not None and requirement.parameter_name not in known:
                raise ValueError(
                    f"field requirement references unknown parameter '{requirement.parameter_name}'"
                )
        if not self.provides_chart and self.chart_types:
            raise ValueError("chart_types must be empty when provides_chart is false")
        if len(set(self.chart_types)) != len(self.chart_types):
            raise ValueError("chart_types must be unique")
        if self.risk_level in {"read_only", "preview_only"} and self.mutates_dataframe:
            raise ValueError(f"risk_level '{self.risk_level}' conflicts with mutates_dataframe=true")
        if self.risk_level == "read_only" and self.returns_modified_dataframe:
            raise ValueError("risk_level 'read_only' conflicts with returns_modified_dataframe=true")
        if not self.limitations or any(not limitation for limitation in self.limitations):
            raise ValueError("limitations must contain at least one non-empty statement")
        for index, example in enumerate(self.examples):
            _ensure_json_value(example, f"examples[{index}]")
        json.dumps(self.model_dump(), allow_nan=False)
        return self
