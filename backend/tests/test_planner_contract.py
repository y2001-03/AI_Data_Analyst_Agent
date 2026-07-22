"""Tests for Enterprise Planner Contract normalization and validation."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from pydantic import ValidationError

from app.graph.dataset_workflow import DatasetGraphNodes
from app.schemas.file import (
    AIAnalysisResult,
    AnalysisTask,
    DatasetColumnProfile,
    DatasetUploadResponse,
    ExecutionResponse,
    ExecutionResult,
)
from app.schemas.planner_contract import (
    PlannerDiagnostics,
    PlannerPlan,
    PlannerTask,
    PlannerValidationIssue,
)
from app.schemas.tool_capability import ToolCapability, ToolFieldRequirement, ToolParameterSpec
from app.services.analysis_planner_service import AnalysisPlannerService
from app.services.dataset_flow_service import DatasetFlowService
from app.services.planner_contract_builder import PlannerContractBuilder
from app.services.pandas_execution_service import PandasExecutionService
from app.tools.registry import ToolRegistry


def _file_info() -> DatasetUploadResponse:
    return DatasetUploadResponse(
        file_name="sales.csv",
        file_type="csv",
        row_count=100,
        column_count=5,
        preview=[],
        columns=[
            DatasetColumnProfile(name="date", data_type="datetime64[ns]", missing_count=0, unique_values=100),
            DatasetColumnProfile(name="sales", data_type="float64", missing_count=0, unique_values=90),
            DatasetColumnProfile(name="price", data_type="float64", missing_count=0, unique_values=80),
            DatasetColumnProfile(name="is_active", data_type="bool", missing_count=0, unique_values=2),
            DatasetColumnProfile(name="region", data_type="object", missing_count=0, unique_values=4),
        ],
    )


def _builder(registry: ToolRegistry | None = None) -> PlannerContractBuilder:
    return PlannerContractBuilder(registry or ToolRegistry.with_default_tools())


def _ambiguous_forecast_file_info() -> DatasetUploadResponse:
    return DatasetUploadResponse(
        file_name="sales.csv",
        file_type="csv",
        row_count=100,
        column_count=4,
        preview=[],
        columns=[
            DatasetColumnProfile(name="order_date", data_type="datetime64[ns]", missing_count=0, unique_values=100),
            DatasetColumnProfile(name="ship_date", data_type="datetime64[ns]", missing_count=0, unique_values=100),
            DatasetColumnProfile(name="sales", data_type="float64", missing_count=0, unique_values=90),
            DatasetColumnProfile(name="profit", data_type="float64", missing_count=0, unique_values=80),
        ],
    )


def _valid_task(**updates: object) -> PlannerTask:
    values: dict[str, object] = {
        "task_id": "task_1",
        "task_type": "forecast_analysis",
        "tool_name": "forecast_analysis_tool",
        "name": "Forecast sales",
        "description": "Forecast future sales.",
        "params": {"time_column": "date", "target_column": "sales", "horizon": 7},
        "reason": "User requested future forecast.",
        "confidence": 0.9,
        "source": "llm",
        "risk_level": "read_only",
        "requires_confirmation": False,
        "validation_status": "valid",
        "validation_issues": (),
    }
    values.update(updates)
    return PlannerTask(**values)


def _valid_plan(**updates: object) -> PlannerPlan:
    task = _valid_task()
    values: dict[str, object] = {
        "plan_id": "plan_1",
        "user_intent": "forecast future sales",
        "normalized_intent": "forecast_analysis",
        "tasks": (task,),
        "status": "valid",
        "planner_source": "llm",
        "capability_catalog_version": "1.0",
        "warnings": (),
        "validation_issues": (),
        "diagnostics": PlannerDiagnostics(source="llm", raw_task_count=1, normalized_task_count=1, validated_task_count=1),
        "metadata": {"task_count": 1},
    }
    values.update(updates)
    return PlannerPlan(**values)


def test_planner_task_and_plan_are_json_safe() -> None:
    plan = _valid_plan()
    payload = plan.model_dump()
    assert payload["tasks"][0]["task_type"] == "forecast_analysis"
    json.dumps(payload, allow_nan=False)


@pytest.mark.parametrize(
    "updates",
    [
        {"confidence": -0.1},
        {"confidence": 1.1},
        {"source": "unknown"},
        {"validation_status": "done"},
        {"params": {"bad": object()}},
    ],
)
def test_planner_task_rejects_invalid_schema_values(updates: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        _valid_task(**updates)


def test_planner_plan_rejects_duplicate_task_ids_and_empty_valid_plan() -> None:
    duplicate = (_valid_task(), _valid_task(name="Second task"))
    with pytest.raises(ValidationError, match="task_id"):
        _valid_plan(tasks=duplicate)
    with pytest.raises(ValidationError, match="valid plan"):
        _valid_plan(tasks=())


def test_validation_issue_is_structured_and_json_safe() -> None:
    issue = PlannerValidationIssue(
        code="MISSING_REQUIRED_PARAMETER",
        message="forecast_analysis requires target_column.",
        severity="error",
        task_id="task_1",
        field="params.target_column",
        parameter_name="target_column",
        expected="numeric field",
        actual=None,
        repairable=True,
    )
    assert issue.code == "MISSING_REQUIRED_PARAMETER"
    json.dumps(issue.model_dump(), allow_nan=False)


def test_task_type_maps_to_tool_name_and_risk_from_capability() -> None:
    plan = _builder().build_plan(
        {
            "user_intent": "forecast",
            "tasks": [
                {
                    "task_type": "forecast_analysis",
                    "name": "Forecast",
                    "params": {"time_column": "date", "target_column": "sales", "horizon": 7},
                    "confidence": 0.8,
                }
            ],
        },
        _file_info(),
        question="forecast sales",
        source="llm",
    )
    task = plan.tasks[0]
    assert plan.status == "valid"
    assert task.tool_name == "forecast_analysis_tool"
    assert task.risk_level == "read_only"
    assert task.requires_confirmation is False


def test_unknown_task_type_and_tool_mismatch_are_validation_issues() -> None:
    unknown = _builder().build_plan({"tasks": [{"task_type": "made_up", "name": "Bad"}]}, _file_info())
    assert unknown.status == "unsupported"
    assert unknown.validation_issues[0].code == "UNKNOWN_TASK_TYPE"

    mismatch = _builder().build_plan(
        {"tasks": [{"task_type": "forecast_analysis", "tool_name": "stats_tool", "name": "Bad"}]},
        _file_info(),
    )
    assert mismatch.status == "invalid"
    assert any(issue.code == "TASK_TOOL_MISMATCH" for issue in mismatch.validation_issues)


def test_multiple_tools_for_same_task_type_are_selected_stably() -> None:
    parameter = ToolParameterSpec(name="query", type="string", required=True, description="Query.")
    capability_a = ToolCapability(
        tool_name="a_tool",
        task_types=("shared_task",),
        display_name="A",
        description="A test tool.",
        category="test",
        parameters=(parameter,),
        required_parameters=("query",),
        optional_parameters=(),
        field_requirements=(),
        supports_multiple_columns=False,
        supports_grouping=False,
        mutates_dataframe=False,
        returns_modified_dataframe=False,
        provides_chart=False,
        risk_level="read_only",
        deterministic=True,
        limitations=("Test only.",),
    )
    capability_b = capability_a.model_copy(update={"tool_name": "b_tool"})

    class Tool:
        def __init__(self, capability: ToolCapability) -> None:
            self.name = capability.tool_name
            self._capability = capability

        def get_capability(self) -> ToolCapability:
            return self._capability

        def run(self, dataframe, task, context):
            raise AssertionError("builder must not execute tools")

    registry = ToolRegistry()
    registry.register(Tool(capability_b))
    registry.register(Tool(capability_a))
    plan = _builder(registry).build_plan(
        {"tasks": [{"task_type": "shared_task", "name": "Shared", "params": {"query": "x"}}]},
        _file_info(),
    )
    assert plan.tasks[0].tool_name == "a_tool"
    assert "multiple_tools_first_sorted" in plan.diagnostics.normalization_warnings[0]


def test_data_cleaning_risk_overrides_plan_and_execute() -> None:
    plan = _builder().build_plan(
        {"tasks": [{"task_type": "data_cleaning_plan", "name": "Plan cleaning"}]},
        _file_info(),
        source="fallback",
    )
    execute = _builder().build_plan(
        {
            "tasks": [
                {
                    "task_type": "data_cleaning_execute",
                    "name": "Execute cleaning",
                    "params": {"mode": "execute", "actions": [{"action_type": "trim_strings"}]},
                }
            ]
        },
        _file_info(),
        source="fallback",
    )
    assert plan.tasks[0].risk_level == "preview_only"
    assert plan.tasks[0].requires_confirmation is False
    assert execute.tasks[0].risk_level == "mutating_copy"
    assert execute.tasks[0].requires_confirmation is True


def test_system_infers_clarification_required_for_ambiguous_forecast_fields() -> None:
    plan = _builder().build_plan(
        {"tasks": [{"task_type": "forecast_analysis", "name": "Forecast", "params": {}}]},
        _ambiguous_forecast_file_info(),
    )
    assert plan.status == "clarification_required"
    assert {issue.parameter_name for issue in plan.validation_issues} == {"time_column", "target_column"}
    assert all(issue.code == "CLARIFICATION_REQUIRED" for issue in plan.validation_issues)


def test_system_infers_clarification_required_for_group_comparison_without_group_by() -> None:
    plan = _builder().build_plan(
        {"tasks": [{"task_type": "distribution_analysis", "name": "Compare", "params": {"mode": "group_comparison", "columns": ["sales"]}}]},
        _file_info(),
    )
    assert plan.status == "clarification_required"
    assert any(issue.code == "CLARIFICATION_REQUIRED" and issue.parameter_name == "group_by" for issue in plan.validation_issues)


def test_system_infers_clarification_required_for_cleaning_execute_without_actions() -> None:
    missing = _builder().build_plan(
        {"tasks": [{"task_type": "data_cleaning_execute", "name": "Clean", "params": {"mode": "execute"}}]},
        _file_info(),
    )
    empty = _builder().build_plan(
        {"tasks": [{"task_type": "data_cleaning_execute", "name": "Clean", "params": {"mode": "execute", "actions": []}}]},
        _file_info(),
    )
    assert missing.status == "clarification_required"
    assert empty.status == "clarification_required"
    assert all(any(issue.parameter_name == "actions" for issue in plan.validation_issues) for plan in (missing, empty))


def test_invalid_field_and_unknown_task_do_not_become_clarification_required() -> None:
    invalid = _builder().build_plan(
        {"tasks": [{"task_type": "forecast_analysis", "name": "Forecast", "params": {"time_column": "missing"}}]},
        _ambiguous_forecast_file_info(),
    )
    unsupported = _builder().build_plan(
        {"tasks": [{"task_type": "not_a_capability", "name": "Unknown"}]},
        _file_info(),
    )
    assert invalid.status == "invalid"
    assert any(issue.code == "UNKNOWN_FIELD" for issue in invalid.validation_issues)
    assert unsupported.status == "unsupported"
    assert any(issue.code == "UNKNOWN_TASK_TYPE" for issue in unsupported.validation_issues)


def test_llm_risk_metadata_cannot_override_capability() -> None:
    plan = _builder().build_plan(
        {
            "tasks": [
                {
                    "task_type": "forecast_analysis",
                    "risk_level": "external_side_effect",
                    "name": "Forecast",
                    "params": {"time_column": "date", "target_column": "sales"},
                }
            ]
        },
        _file_info(),
    )
    assert plan.tasks[0].risk_level == "read_only"
    assert any(issue.code == "RISK_METADATA_MISMATCH" for issue in plan.validation_issues)


def test_parameter_validation_required_unknown_types_values_and_ranges() -> None:
    missing = _builder().build_plan({"tasks": [{"task_type": "sql", "name": "SQL"}]}, _file_info())
    assert any(issue.code == "MISSING_REQUIRED_PARAMETER" for issue in missing.validation_issues)

    unknown = _builder().build_plan(
        {"tasks": [{"task_type": "forecast_analysis", "name": "Forecast", "params": {"unexpected": 1}}]},
        _file_info(),
    )
    assert any(issue.code == "UNKNOWN_PARAMETER" for issue in unknown.validation_issues)

    typed = _builder().build_plan(
        {"tasks": [{"task_type": "forecast_analysis", "name": "Forecast", "params": {"horizon": True}}]},
        _file_info(),
    )
    assert any(issue.code == "INVALID_PARAMETER_TYPE" and issue.parameter_name == "horizon" for issue in typed.validation_issues)

    ranged = _builder().build_plan(
        {"tasks": [{"task_type": "forecast_analysis", "name": "Forecast", "params": {"horizon": 999}}]},
        _file_info(),
    )
    assert any(issue.code == "INVALID_PARAMETER_VALUE" and issue.parameter_name == "horizon" for issue in ranged.validation_issues)

    allowed = _builder().build_plan(
        {"tasks": [{"task_type": "forecast_analysis", "name": "Forecast", "params": {"method": "arima"}}]},
        _file_info(),
    )
    assert any(issue.code == "INVALID_PARAMETER_VALUE" and issue.parameter_name == "method" for issue in allowed.validation_issues)


def test_parameter_types_accept_valid_string_number_array_object_and_null_optional() -> None:
    plan = _builder().build_plan(
        {
            "tasks": [
                {
                    "task_type": "forecast_analysis",
                    "name": "Forecast",
                    "params": {
                        "time_column": "date",
                        "target_column": "sales",
                        "horizon": 7,
                        "clip_lower": 0.0,
                        "method": "auto",
                        "chart_group": None,
                    },
                },
                {
                    "task_type": "distribution_analysis",
                    "name": "Distribution",
                    "params": {"mode": "numeric", "columns": ["sales"], "quantiles": [0.25, 0.5]},
                },
                {
                    "task_type": "data_cleaning_execute",
                    "name": "Clean",
                    "params": {"mode": "execute", "actions": [{"action_type": "trim_strings"}]},
                },
            ]
        },
        _file_info(),
        source="fallback",
    )
    assert [task.validation_status for task in plan.tasks] == ["valid", "valid", "valid"]


def test_non_json_safe_params_are_rejected() -> None:
    plan = _builder().build_plan(
        {"tasks": [{"task_type": "forecast_analysis", "name": "Bad", "params": {"horizon": float("nan")}}]},
        _file_info(),
    )
    assert plan.status == "invalid"
    assert any(issue.code == "NON_JSON_SAFE_PARAMS" for issue in plan.validation_issues)


def test_field_validation_known_unknown_type_and_multiplicity() -> None:
    valid = _builder().build_plan(
        {"tasks": [{"task_type": "correlation_analysis", "name": "Correlation", "params": {"columns": ["sales", "price"]}}]},
        _file_info(),
    )
    assert valid.status == "valid"

    unknown = _builder().build_plan(
        {"tasks": [{"task_type": "correlation_analysis", "name": "Correlation", "params": {"columns": ["missing"]}}]},
        _file_info(),
    )
    assert any(issue.code == "UNKNOWN_FIELD" for issue in unknown.validation_issues)

    invalid_type = _builder().build_plan(
        {"tasks": [{"task_type": "correlation_analysis", "name": "Correlation", "params": {"columns": ["is_active"]}}]},
        _file_info(),
    )
    assert any(issue.code == "INVALID_FIELD_TYPE" for issue in invalid_type.validation_issues)

    list_for_single = _builder().build_plan(
        {"tasks": [{"task_type": "forecast_analysis", "name": "Forecast", "params": {"time_column": ["date"]}}]},
        _file_info(),
    )
    assert any(issue.code == "INVALID_PARAMETER_TYPE" and issue.parameter_name == "time_column" for issue in list_for_single.validation_issues)


def test_field_validation_datetime_categorical_group_and_text_time_boundaries() -> None:
    forecast = _builder().build_plan(
        {"tasks": [{"task_type": "forecast_analysis", "name": "Forecast", "params": {"time_column": "date", "target_column": "sales"}}]},
        _file_info(),
    )
    grouped = _builder().build_plan(
        {"tasks": [{"task_type": "distribution_analysis", "name": "Groups", "params": {"mode": "group_comparison", "group_by": "region", "columns": ["sales"]}}]},
        _file_info(),
    )
    bad_target = _builder().build_plan(
        {"tasks": [{"task_type": "forecast_analysis", "name": "Forecast", "params": {"time_column": "date", "target_column": "date"}}]},
        _file_info(),
    )
    text_time = _builder().build_plan(
        {"tasks": [{"task_type": "forecast_analysis", "name": "Forecast", "params": {"time_column": "region", "target_column": "sales"}}]},
        _file_info(),
    )
    assert forecast.status == "valid"
    assert grouped.status == "valid"
    assert any(issue.code == "INVALID_FIELD_TYPE" and issue.parameter_name == "target_column" for issue in bad_target.validation_issues)
    assert text_time.status == "valid"


def test_plan_status_rules_valid_invalid_clarification_unsupported_and_warning() -> None:
    valid = _builder().build_plan(
        {"tasks": [{"task_type": "forecast_analysis", "name": "Forecast", "params": {"time_column": "date", "target_column": "sales"}}]},
        _file_info(),
    )
    invalid = _builder().build_plan(
        {"tasks": [{"task_type": "forecast_analysis", "name": "Forecast", "params": {"time_column": "missing"}}]},
        _file_info(),
    )
    clarification = _builder().build_plan({"status": "clarification_required"}, _file_info())
    unsupported = _builder().build_plan({"status": "unsupported"}, _file_info())
    warning = _builder().build_plan(
        {
            "tasks": [
                {
                    "task_type": "forecast_analysis",
                    "name": "Forecast",
                    "risk_level": "mutating_copy",
                    "params": {"time_column": "date", "target_column": "sales"},
                }
            ]
        },
        _file_info(),
    )
    assert valid.status == "valid"
    assert invalid.status == "invalid"
    assert clarification.status == "clarification_required"
    assert unsupported.status == "unsupported"
    assert warning.status == "valid"
    assert warning.warnings


def test_llm_output_normalization_generates_task_id_tool_and_ignores_system_fields() -> None:
    plan = _builder().build_plan(
        {
            "user_intent": "forecast",
            "tasks": [
                {
                    "task_type": "forecast_analysis",
                    "name": "Forecast",
                    "validation_status": "invalid",
                    "risk_level": "external_side_effect",
                    "params": {"time_column": "date", "target_column": "sales"},
                }
            ],
        },
        _file_info(),
    )
    assert plan.tasks[0].task_id == "task_1"
    assert plan.tasks[0].tool_name == "forecast_analysis_tool"
    assert plan.tasks[0].validation_status == "valid"
    assert plan.tasks[0].risk_level == "read_only"


def test_detection_only_is_planner_internal_and_not_tool_capability_parameter() -> None:
    capability = ToolRegistry.with_default_tools().get_capability("anomaly_detection_tool")
    assert capability is not None
    assert "detection_only" not in {parameter.name for parameter in capability.parameters}

    plan = _builder().build_plan(
        {
            "tasks": [
                {
                    "task_type": "anomaly_detection",
                    "name": "Detect",
                    "params": {"mode": "global", "columns": ["sales"], "detection_only": True},
                }
            ]
        },
        _file_info(),
    )
    adapted = _builder().planner_task_to_analysis_task(plan.tasks[0])
    assert plan.status == "valid"
    assert "detection_only" not in plan.tasks[0].params
    assert "detection_only" not in adapted.params
    assert "detection_only_internal_flag_removed" in plan.diagnostics.normalization_warnings[0]


def test_missing_tasks_records_parse_failure_and_fallback_source_is_stable() -> None:
    failed = _builder().build_plan({}, _file_info(), parse_failure="JSONDecodeError")
    assert failed.status == "unsupported"
    assert failed.diagnostics.parse_failure == "JSONDecodeError"

    fallback = _builder().build_plan(
        [AnalysisTask(task_name="Summary", reasoning="Fallback", expected_output="Stats", type="stats", params={})],
        _file_info(),
        source="fallback",
        fallback_used=True,
    )
    assert fallback.status == "valid"
    assert fallback.tasks[0].source == "fallback"
    assert fallback.tasks[0].confidence == 0.85


def test_adapter_preserves_params_type_and_does_not_mutate_task() -> None:
    plan = _builder().build_plan(
        {"tasks": [{"task_type": "forecast_analysis", "name": "Forecast", "params": {"time_column": "date", "target_column": "sales", "horizon": 7}}]},
        _file_info(),
    )
    task = plan.tasks[0]
    before = task.model_dump()
    adapted = _builder().planner_task_to_analysis_task(task)
    assert adapted.type == task.task_type
    assert adapted.params == task.params
    assert task.model_dump() == before

    invalid = task.model_copy(update={"validation_status": "invalid"})
    with pytest.raises(ValueError):
        _builder().planner_task_to_analysis_task(invalid)


def test_workflow_stores_invalid_plan_and_does_not_enter_execute() -> None:
    invalid_plan = _builder().build_plan(
        {"tasks": [{"task_type": "forecast_analysis", "name": "Forecast", "params": {"time_column": "missing"}}]},
        _file_info(),
    )

    class Planner:
        contract_builder = _builder()

        def plan(self, file_info, ai_analysis, question=None, memory_context=None):
            return invalid_plan

    nodes = object.__new__(DatasetGraphNodes)
    nodes.analysis_planner_service = Planner()
    state = {
        "file_info": _file_info(),
        "ai_analysis": AIAnalysisResult(summary="s", suggestions=["x"], tasks=[]),
        "question": "forecast sales",
        "memory_context": None,
        "error_stage": None,
        "tasks": [],
        "execution_results": [],
        "planner_failed": False,
        "fallback_used": False,
        "fallback_reason": None,
        "trace_log": [],
        "node_status": {},
        "execution_path": [],
    }
    result = nodes.plan(state)
    assert result["tasks"] == []
    assert result["planner_status"] == "invalid"
    assert result["execution_results"][0].type == "planning_failure"
    assert nodes.route_after_plan(result) == "end"


@pytest.mark.parametrize(
    "plan",
    [
        _builder().build_plan(
            {"tasks": [{"task_type": "forecast_analysis", "name": "Forecast", "params": {"time_column": "missing"}}]},
            _file_info(),
        ),
        _builder().build_plan(
            {"tasks": [{"task_type": "distribution_analysis", "name": "Compare", "params": {"mode": "group_comparison", "columns": ["sales"]}}]},
            _file_info(),
        ),
        _builder().build_plan(
            {"tasks": [{"task_type": "made_up", "name": "Unsupported"}]},
            _file_info(),
        ),
    ],
)
def test_invalid_clarification_and_unsupported_plans_do_not_enter_execute(plan: PlannerPlan) -> None:
    assert plan.status in {"invalid", "clarification_required", "unsupported"}

    class Planner:
        contract_builder = _builder()

        def plan(self, file_info, ai_analysis, question=None, memory_context=None):
            return plan

    nodes = object.__new__(DatasetGraphNodes)
    nodes.analysis_planner_service = Planner()
    state = {
        "file_info": _file_info(),
        "ai_analysis": AIAnalysisResult(summary="s", suggestions=["x"], tasks=[]),
        "question": "q",
        "memory_context": None,
        "error_stage": None,
        "tasks": [],
        "execution_results": [],
        "planner_failed": False,
        "fallback_used": False,
        "fallback_reason": None,
        "trace_log": [],
        "node_status": {},
        "execution_path": [],
    }
    result = nodes.plan(state)
    assert result["tasks"] == []
    assert result["planner_status"] == plan.status
    assert result["execution_results"][0].type == "planning_failure"
    assert nodes.route_after_plan(result) == "end"


def test_valid_plan_still_enters_execute_with_existing_analysis_task_protocol() -> None:
    plan = _builder().build_plan(
        {"tasks": [{"task_type": "stats", "name": "Stats", "params": {}}]},
        _file_info(),
        source="fallback",
    )

    class Planner:
        contract_builder = _builder()

        def plan(self, file_info, ai_analysis, question=None, memory_context=None):
            return plan

    class ExecutionService:
        called = False

        def execute_tasks(self, dataframe, tasks):
            self.called = True
            assert tasks[0].type == "stats"
            return ExecutionResponse(
                execution_results=[
                    ExecutionResult(task_name="Stats", type="stats", data={"rows": []}, chart=None)
                ]
            )

    nodes = object.__new__(DatasetGraphNodes)
    nodes.analysis_planner_service = Planner()
    nodes.execution_service = ExecutionService()
    state = {
        "file_name": "sales.csv",
        "content": b"",
        "dataframe": pd.DataFrame({"sales": [1, 2, 3]}),
        "file_info": _file_info(),
        "ai_analysis": AIAnalysisResult(summary="s", suggestions=["x"], tasks=[]),
        "question": "summary",
        "memory_context": None,
        "error_stage": None,
        "tasks": [],
        "execution_results": [],
        "planner_failed": False,
        "execution_failed": False,
        "fallback_used": False,
        "fallback_reason": None,
        "trace_log": [],
        "node_status": {},
        "execution_path": [],
    }
    planned = nodes.plan(state)
    assert nodes.route_after_plan(planned) == "execute"
    executed = nodes.execute(planned)
    assert nodes.execution_service.called is True
    assert executed["execution_results"][0].type == "stats"


def test_fallback_stats_groupby_and_trend_visualization_behavior_is_preserved() -> None:
    planner = object.__new__(AnalysisPlannerService)
    nodes = object.__new__(DatasetGraphNodes)
    dataframe = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=6, freq="D"),
            "sales": [10, 12, 13, 15, 17, 20],
            "region": ["East", "West", "East", "West", "East", "West"],
        }
    )
    service = PandasExecutionService()

    stats_tasks = planner._mock_tasks(_file_info(), "summary sales")
    stats_with_visual = nodes._ensure_visualization_task(_file_info(), stats_tasks)
    assert stats_tasks[0].type == "stats"
    assert len(stats_with_visual) == 2

    trend_tasks = planner._mock_tasks(_file_info(), "daily trend for sales by date")
    trend_with_visual = nodes._ensure_visualization_task(_file_info(), trend_tasks)
    trend_result = service.execute_tasks(dataframe, trend_with_visual).execution_results[0]
    assert trend_with_visual[0].type == "trend"
    assert trend_result.chart is not None

    group_tasks = planner._mock_tasks(_file_info(), "group sales by region")
    group_with_visual = nodes._ensure_visualization_task(_file_info(), group_tasks)
    group_result = service.execute_tasks(dataframe, group_with_visual).execution_results[1]
    assert any(task.type == "groupby" for task in group_with_visual)
    assert group_result.type == "groupby"
    assert group_result.chart is not None


def test_dataset_flow_state_and_progress_payload_do_not_expose_prompt_or_capability_catalog() -> None:
    service = object.__new__(DatasetFlowService)
    state = service._build_initial_state(
        "sales.csv",
        b"a,b\n1,2\n",
        "summary",
        None,
        dataframe=None,
    )
    assert state["planner_plan"] is None
    assert state["planner_status"] is None
    assert state["planner_issues"] == []
    assert state["normalized_intent"] is None

    payload = service._build_progress_payload(
        "plan",
        {
            "trace_log": [
                {
                    "node": "plan",
                    "status": "success",
                    "input_summary": {},
                    "output_summary": {"task_count": 1, "issue_codes": []},
                }
            ],
            "node_status": {"plan": "success"},
        },
    )
    serialized = json.dumps(payload, allow_nan=False)
    assert payload["node"] == "planning"
    assert "capability catalog" not in serialized.lower()
    assert "capability_context" not in serialized
    assert "prompt" not in serialized.lower()
