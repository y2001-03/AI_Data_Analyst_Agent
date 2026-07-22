"""Tests for enterprise tool capability metadata and read-only discovery."""

from __future__ import annotations

import inspect
import json

import pandas as pd
import pytest
from pydantic import ValidationError

from app.graph.dataset_workflow import DatasetGraphNodes
from app.schemas.file import AnalysisTask, DatasetColumnProfile, DatasetUploadResponse
from app.schemas.tool_capability import ToolCapability, ToolFieldRequirement, ToolParameterSpec
from app.services.planner_capability_context import build_planner_capability_context
from app.tools.capabilities import validate_capability
from app.tools.dataframe_tools import BaseDataframeTool, DatasetContext
from app.tools.registry import ToolRegistry


def _valid_parameter(**updates: object) -> ToolParameterSpec:
    values: dict[str, object] = {
        "name": "columns",
        "type": "array",
        "required": False,
        "description": "Columns to analyze.",
    }
    values.update(updates)
    return ToolParameterSpec(**values)


def _valid_capability(**updates: object) -> ToolCapability:
    parameter = _valid_parameter()
    values: dict[str, object] = {
        "tool_name": "example_tool",
        "task_types": ("example",),
        "display_name": "Example",
        "description": "Example read-only capability.",
        "category": "testing",
        "parameters": (parameter,),
        "required_parameters": (),
        "optional_parameters": ("columns",),
        "field_requirements": (
            ToolFieldRequirement(
                parameter_name="columns",
                role="columns",
                accepted_types=("numeric",),
                required=False,
                multiple=True,
                description="Optional numeric columns.",
            ),
        ),
        "supports_multiple_columns": True,
        "supports_grouping": False,
        "mutates_dataframe": False,
        "returns_modified_dataframe": False,
        "provides_chart": False,
        "risk_level": "read_only",
        "deterministic": True,
        "limitations": ("Example only.",),
    }
    values.update(updates)
    return ToolCapability(**values)


def _workflow_file_info() -> DatasetUploadResponse:
    return DatasetUploadResponse(
        file_name="sample.csv",
        file_type="csv",
        row_count=20,
        column_count=3,
        preview=[],
        columns=[
            DatasetColumnProfile(name="date", data_type="datetime", missing_count=0, unique_values=20),
            DatasetColumnProfile(name="region", data_type="object", missing_count=0, unique_values=2),
            DatasetColumnProfile(name="sales", data_type="float", missing_count=0, unique_values=20),
        ],
    )


def test_tool_capability_can_be_created_and_strictly_serialized() -> None:
    capability = _valid_capability()
    payload = capability.model_dump()
    assert payload["tool_name"] == "example_tool"
    json.dumps(payload, allow_nan=False)


def test_parameter_spec_supports_required_default_and_allowed_values() -> None:
    parameter = _valid_parameter(
        name="method",
        type="string",
        required=True,
        default="pearson",
        allowed_values=("pearson", "spearman"),
    )
    assert parameter.required is True
    assert parameter.default == "pearson"
    assert parameter.allowed_values == ("pearson", "spearman")


def test_field_requirement_uses_logical_roles_and_types() -> None:
    requirement = ToolFieldRequirement(
        parameter_name="target_column",
        role="target",
        accepted_types=("numeric",),
        required=True,
        exclude_types=("boolean", "datetime", "timedelta"),
        description="Numeric target.",
    )
    assert requirement.role == "target"
    assert requirement.accepted_types == ("numeric",)


@pytest.mark.parametrize(
    "field,value",
    [("risk_level", "dangerous"), ("type", "callable")],
)
def test_invalid_risk_level_and_parameter_type_are_rejected(field: str, value: str) -> None:
    with pytest.raises(ValidationError):
        if field == "risk_level":
            _valid_capability(risk_level=value)
        else:
            _valid_parameter(type=value)


def test_parameter_minimum_must_not_exceed_maximum() -> None:
    with pytest.raises(ValidationError, match="minimum"):
        _valid_parameter(type="integer", minimum=10, maximum=1)


def test_parameter_default_must_be_allowed() -> None:
    with pytest.raises(ValidationError, match="default"):
        _valid_parameter(type="string", default="kendall", allowed_values=("pearson", "spearman"))


def test_required_and_optional_parameters_must_not_overlap() -> None:
    parameter = _valid_parameter(required=True)
    with pytest.raises(ValidationError, match="overlap"):
        _valid_capability(
            parameters=(parameter,),
            required_parameters=("columns",),
            optional_parameters=("columns",),
        )


def test_duplicate_parameter_names_are_rejected() -> None:
    with pytest.raises(ValidationError, match="unique"):
        _valid_capability(
            parameters=(_valid_parameter(), _valid_parameter()),
            optional_parameters=("columns",),
        )


def test_non_finite_or_non_json_defaults_are_rejected() -> None:
    with pytest.raises(ValidationError, match="NaN"):
        _valid_parameter(default=float("nan"))
    with pytest.raises(ValidationError, match="non-JSON"):
        _valid_parameter(default=object())


def test_base_tool_exposes_stable_side_effect_free_capability() -> None:
    tool = BaseDataframeTool()
    before = dict(tool.__dict__)
    first = tool.capability
    second = tool.get_capability()
    assert first == second
    assert first is not second
    assert tool.__dict__ == before


def test_registered_tool_run_protocol_is_unchanged() -> None:
    for tool in ToolRegistry.with_default_tools()._tools.values():
        assert list(inspect.signature(tool.run).parameters) == ["dataframe", "task", "context"]


def test_all_registered_tools_have_valid_explicit_capabilities() -> None:
    registry = ToolRegistry.with_default_tools()
    capabilities = registry.list_capabilities()
    assert len(capabilities) == 10
    assert {capability.tool_name for capability in capabilities} == set(registry._tools)
    assert all(capability.task_types for capability in capabilities)
    assert all(isinstance(capability.mutates_dataframe, bool) for capability in capabilities)
    assert all(isinstance(capability.provides_chart, bool) for capability in capabilities)
    assert all(validate_capability(tool).tool_name == name for name, tool in registry._tools.items())


def test_provides_chart_true_tools_return_embedded_chart_payloads() -> None:
    registry = ToolRegistry.with_default_tools()
    dataframe = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=20, freq="D"),
            "sales": [float(value) for value in range(1, 21)],
            "profit": [float(value * 2) for value in range(1, 21)],
            "region": ["East", "West"] * 10,
        }
    )
    context = DatasetContext(
        numeric_columns=["sales", "profit"],
        categorical_columns=["region"],
        datetime_columns=["date"],
    )
    cases = {
        "correlation_analysis_tool": AnalysisTask(
            task_name="correlation",
            reasoning="",
            expected_output="",
            type="correlation_analysis",
            params={"columns": ["sales", "profit"]},
        ),
        "anomaly_detection_tool": AnalysisTask(
            task_name="anomaly",
            reasoning="",
            expected_output="",
            type="anomaly_detection",
            params={"mode": "global", "method": "iqr", "columns": ["sales"]},
        ),
        "distribution_analysis_tool": AnalysisTask(
            task_name="distribution",
            reasoning="",
            expected_output="",
            type="distribution_analysis",
            params={"mode": "numeric", "columns": ["sales"]},
        ),
        "forecast_analysis_tool": AnalysisTask(
            task_name="forecast",
            reasoning="",
            expected_output="",
            type="forecast_analysis",
            params={
                "time_column": "date",
                "target_column": "sales",
                "horizon": 3,
                "method": "naive",
            },
        ),
    }
    for tool_name, task in cases.items():
        capability = registry.get_capability(tool_name)
        assert capability is not None and capability.provides_chart is True
        result = registry.get(tool_name).run(dataframe, task, context)
        chart = result.data.get("chart")
        assert result.chart is None
        assert isinstance(chart, dict)
        assert chart.get("type") in capability.chart_types


def test_legacy_visualization_task_types_do_not_claim_embedded_chart_capability() -> None:
    registry = ToolRegistry.with_default_tools()
    for tool_name in ("stats_tool", "groupby_tool", "trend_tool"):
        capability = registry.get_capability(tool_name)
        assert capability is not None
        assert capability.provides_chart is False
        assert capability.chart_types == ()


def test_provides_chart_matches_workflow_no_extra_visualization_semantics() -> None:
    registry = ToolRegistry.with_default_tools()
    nodes = DatasetGraphNodes()
    file_info = _workflow_file_info()
    for capability in registry.list_capabilities():
        if not capability.provides_chart:
            continue
        task = AnalysisTask(
            task_name=f"{capability.tool_name} task",
            reasoning="",
            expected_output="",
            type=capability.task_types[0],
        )
        assert nodes._ensure_visualization_task(file_info, [task]) == [task]

    stats_task = AnalysisTask(
        task_name="Basic summary",
        reasoning="Describe numeric values.",
        expected_output="Summary table.",
        type="stats",
    )
    stats_tasks = nodes._ensure_visualization_task(file_info, [stats_task])
    assert len(stats_tasks) == 2
    assert registry.get_capability("stats_tool").provides_chart is False


def test_forecast_capability_matches_implemented_core_parameters() -> None:
    capability = ToolRegistry.with_default_tools().get_capability("forecast_analysis_tool")
    assert capability is not None
    parameters = {item.name: item for item in capability.parameters}
    assert {"time_column", "target_column", "horizon"} <= parameters.keys()
    assert parameters["horizon"].default == 7
    assert parameters["horizon"].minimum == 1
    assert parameters["horizon"].maximum == 365
    assert capability.chart_types == ("forecast_line",)


def test_anomaly_distribution_and_correlation_capabilities_match_modes() -> None:
    registry = ToolRegistry.with_default_tools()
    anomaly = registry.get_capability("anomaly_detection_tool")
    distribution = registry.get_capability("distribution_analysis_tool")
    correlation = registry.get_capability("correlation_analysis_tool")
    assert anomaly is not None and {"mode", "method"} <= {item.name for item in anomaly.parameters}
    distribution_names = {item.name for item in distribution.parameters} if distribution else set()
    assert {"mode", "columns", "quantiles", "bins", "top_k", "group_by"} <= distribution_names
    assert distribution is not None and distribution.supports_grouping is True
    assert correlation is not None and {"method", "columns"} <= {item.name for item in correlation.parameters}


def test_cleaning_and_sql_risk_boundaries_match_real_behavior() -> None:
    registry = ToolRegistry.with_default_tools()
    cleaning = registry.get_capability("data_cleaning_tool")
    sql = registry.get_capability("sql_tool")
    assert cleaning is not None
    assert cleaning.risk_level == "mutating_copy"
    assert cleaning.mutates_dataframe is False
    assert cleaning.returns_modified_dataframe is True
    assert any("Plan mode is preview-only" in limitation for limitation in cleaning.limitations)
    assert any("Execute mode produces changes only on an isolated dataframe copy" in limitation for limitation in cleaning.limitations)
    assert any("never overwritten" in limitation for limitation in cleaning.limitations)
    assert sql is not None and sql.risk_level == "read_only"
    assert any("Only SELECT" in limitation for limitation in sql.limitations)


def test_registry_capability_queries_are_stable_and_exact() -> None:
    registry = ToolRegistry.with_default_tools()
    first = [item.tool_name for item in registry.list_capabilities()]
    second = [item.tool_name for item in registry.list_capabilities()]
    assert first == second == sorted(first)
    assert registry.get_capability("missing") is None
    assert [item.tool_name for item in registry.find_by_task_type("forecast_analysis")] == ["forecast_analysis_tool"]
    assert [item.tool_name for item in registry.find_capabilities_by_category("forecasting")] == ["forecast_analysis_tool"]
    assert registry.find_by_task_type("forecast") == []


def test_capability_catalog_is_stable_indexed_and_json_safe() -> None:
    registry = ToolRegistry.with_default_tools()
    first = registry.get_capability_catalog()
    second = registry.get_capability_catalog()
    assert first == second
    assert first["metadata"]["tool_count"] == 10
    assert first["task_type_index"]["forecast_analysis"] == ["forecast_analysis_tool"]
    assert first["category_index"]["forecasting"] == ["forecast_analysis_tool"]
    assert [item["tool_name"] for item in first["tools"]] == sorted(item["tool_name"] for item in first["tools"])
    json.dumps(first, allow_nan=False)


def test_catalog_generation_does_not_execute_tools_and_preserves_get() -> None:
    capability = _valid_capability(tool_name="sentinel_tool", task_types=("sentinel",))

    class SentinelTool:
        name = "sentinel_tool"

        def get_capability(self) -> ToolCapability:
            return capability

        def run(self, dataframe, task, context):
            raise AssertionError("capability discovery must not execute tools")

    registry = ToolRegistry()
    tool = SentinelTool()
    registry.register(tool)
    assert registry.get("sentinel_tool") is tool
    assert registry.get_capability_catalog()["metadata"]["tool_count"] == 1


def test_planner_capability_context_contains_contract_fields_in_stable_order() -> None:
    registry = ToolRegistry.with_default_tools()
    first = build_planner_capability_context(registry)
    second = build_planner_capability_context(registry)
    assert first == second
    assert "task_types=forecast_analysis" in first
    assert "required=query" in first
    assert "risk=mutating_copy" in first
    assert "limitations=" in first
    assert first.index("anomaly_detection_tool") < first.index("forecast_analysis_tool")
    assert "<function" not in first and "ToolCapability(" not in first


def test_planner_capability_context_is_bounded_and_does_not_execute_tools() -> None:
    registry = ToolRegistry.with_default_tools()
    context = build_planner_capability_context(registry, max_length=500)
    assert len(context) <= 500
    with pytest.raises(ValueError):
        build_planner_capability_context(registry, max_length=255)


def test_validate_capability_detects_tool_name_mismatch() -> None:
    class WrongNameTool:
        name = "actual_tool"

        def get_capability(self) -> ToolCapability:
            return _valid_capability(tool_name="different_tool")

    with pytest.raises(ValueError, match="does not match"):
        validate_capability(WrongNameTool())


@pytest.mark.parametrize(
    "updates,match",
    [
        ({"required_parameters": ("missing",), "optional_parameters": ("columns",)}, "classify"),
        ({"provides_chart": False, "chart_types": ("bar",)}, "chart_types"),
        ({"risk_level": "read_only", "mutates_dataframe": True}, "risk_level"),
        ({
            "field_requirements": (
                ToolFieldRequirement(
                    parameter_name="missing",
                    role="columns",
                    accepted_types=("numeric",),
                    required=False,
                    description="Unknown parameter reference.",
                ),
            ),
        }, "unknown parameter"),
    ],
)
def test_consistency_validation_detects_constructed_contract_conflicts(
    updates: dict[str, object],
    match: str,
) -> None:
    malformed = _valid_capability().model_copy(update=updates)

    class MalformedTool:
        name = "example_tool"

        def get_capability(self) -> ToolCapability:
            return malformed

    with pytest.raises(ValidationError, match=match):
        validate_capability(MalformedTool())


def test_consistency_validation_detects_invalid_constructed_default() -> None:
    invalid_parameter = _valid_parameter().model_copy(
        update={"type": "string", "default": "invalid", "allowed_values": ("valid",)}
    )
    malformed = _valid_capability().model_copy(update={"parameters": (invalid_parameter,)})

    class MalformedTool:
        name = "example_tool"

        def get_capability(self) -> ToolCapability:
            return malformed

    with pytest.raises(ValidationError, match="default"):
        validate_capability(MalformedTool())
