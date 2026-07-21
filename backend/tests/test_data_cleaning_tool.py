"""Tests for safe data cleaning planning and execution."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pandas.testing as pdt

from app.schemas.file import AnalysisTask, ExecutionResult
from app.services.pandas_execution_service import PandasExecutionService
from app.tools import DataCleaningTool, DatasetContext, ToolRegistry


def _task(task_type: str, actions: list[dict[str, object]] | None = None) -> AnalysisTask:
    params: dict[str, object] = {"mode": "plan" if task_type.endswith("plan") else "execute"}
    if actions is not None:
        params["actions"] = actions
    return AnalysisTask(
        task_name="Data Cleaning",
        reasoning="Safely clean the dataset.",
        expected_output="Structured cleaning result.",
        type=task_type,
        params=params,
    )


def _run_plan(dataframe: pd.DataFrame):
    return DataCleaningTool().run(dataframe, _task("data_cleaning_plan"), _context())


def _run_execute(dataframe: pd.DataFrame, actions: list[dict[str, object]] | None = None):
    return DataCleaningTool().run(
        dataframe,
        _task("data_cleaning_execute", actions),
        _context(),
    )


def _context() -> DatasetContext:
    return DatasetContext(numeric_columns=[], categorical_columns=[], datetime_columns=[])


def _action(result, action_type: str) -> dict[str, object]:
    return next(item for item in result.data["actions"] if item["action_type"] == action_type)


def test_cleaning_plan_recommends_drop_duplicates() -> None:
    dataframe = pd.DataFrame({"id": [1, 1, 2], "sales": [10, 10, 20]})

    result = _run_plan(dataframe)
    action = _action(result, "drop_duplicates")

    assert result.type == "data_cleaning_plan"
    assert action["affected_count"] == 1
    assert action["parameters"] == {"keep": "first"}


def test_cleaning_plan_recommends_median_for_numeric_missing() -> None:
    dataframe = pd.DataFrame({"sales": [10.0, np.nan, 30.0]})

    action = _action(_run_plan(dataframe), "fill_missing")

    assert action["column"] == "sales"
    assert action["parameters"]["strategy"] == "median"
    assert action["affected_count"] == 1


def test_cleaning_plan_recommends_mode_for_text_missing() -> None:
    dataframe = pd.DataFrame({"region": ["East", None, "West"]})

    action = _action(_run_plan(dataframe), "fill_missing")

    assert action["column"] == "region"
    assert action["parameters"]["strategy"] == "mode"


def test_cleaning_plan_disables_high_risk_duplicate_deletion() -> None:
    dataframe = pd.DataFrame({"id": [1, 1, 1, 2]})

    result = _run_plan(dataframe)
    action = _action(result, "drop_duplicates")

    assert action["affected_ratio"] == 0.5
    assert action["risk_level"] == "high"
    assert action["default_enabled"] is False
    assert result.data["summary"]["high_risk_action_count"] == 1


def test_cleaning_plan_does_not_modify_original_dataframe() -> None:
    dataframe = pd.DataFrame({"name": [" Alice ", None], "sales": [1.0, np.nan]})
    original = dataframe.copy(deep=True)

    _run_plan(dataframe)

    pdt.assert_frame_equal(dataframe, original)


def test_cleaning_plan_is_strictly_json_serializable() -> None:
    dataframe = pd.DataFrame({"value": np.array([1.0, np.nan], dtype=np.float64)})

    serialized = json.dumps(_run_plan(dataframe).model_dump(), allow_nan=False)

    assert "NaN" not in serialized


def test_cleaning_plan_suggests_only_clear_dtype_conversion_and_keeps_it_disabled() -> None:
    dataframe = pd.DataFrame({"amount": ["10", "20.5", None]})

    action = _action(_run_plan(dataframe), "convert_dtype")

    assert action["column"] == "amount"
    assert action["parameters"] == {"target_type": "numeric"}
    assert action["default_enabled"] is False


def test_cleaning_plan_estimates_drop_missing_rows_and_disables_high_risk_action() -> None:
    dataframe = pd.DataFrame({"sales": [1.0, np.nan, np.nan, 4.0]})

    action = _action(_run_plan(dataframe), "drop_missing_rows")

    assert action["affected_count"] == 2
    assert action["affected_ratio"] == 0.5
    assert action["risk_level"] == "high"
    assert action["default_enabled"] is False


def test_execute_drop_duplicates_succeeds_and_audits() -> None:
    dataframe = pd.DataFrame({"id": [1, 1, 2], "sales": [10, 10, 20]})
    actions = [{"action_id": "dedupe", "action_type": "drop_duplicates", "parameters": {}}]

    result = _run_execute(dataframe, actions)

    assert result.data["summary"]["before"]["row_count"] == 3
    assert result.data["summary"]["after"]["row_count"] == 2
    assert result.data["action_results"][0]["affected_count"] == 1
    assert result.data["audit_log"][0] == {
        "sequence": 1,
        "action_id": "dedupe",
        "action_type": "drop_duplicates",
        "column": None,
        "before_value": 3,
        "after_value": 2,
        "affected_count": 1,
    }


def test_execute_numeric_median_fill_succeeds() -> None:
    dataframe = pd.DataFrame({"sales": [10.0, np.nan, 30.0]})
    actions = [
        {
            "action_type": "fill_missing",
            "column": "sales",
            "parameters": {"strategy": "median"},
        }
    ]

    result = _run_execute(dataframe, actions)

    assert result.data["preview"][1]["sales"] == 20.0
    assert result.data["summary"]["after"]["missing_count"] == 0


def test_execute_text_mode_fill_succeeds() -> None:
    dataframe = pd.DataFrame({"region": ["East", None, "East", "West"]})
    actions = [
        {
            "action_type": "fill_missing",
            "column": "region",
            "parameters": {"strategy": "mode"},
        }
    ]

    result = _run_execute(dataframe, actions)

    assert result.data["preview"][1]["region"] == "East"


def test_execute_constant_fill_requires_explicit_value() -> None:
    dataframe = pd.DataFrame({"sales": [1.0, np.nan]})
    actions = [
        {
            "action_type": "fill_missing",
            "column": "sales",
            "parameters": {"strategy": "constant"},
        }
    ]

    result = _run_execute(dataframe, actions)

    assert result.data["action_results"][0]["status"] == "failed"
    assert "requires an explicit value" in result.data["action_results"][0]["message"]
    assert result.data["summary"]["after"]["missing_count"] == 1


def test_execute_trim_strings_succeeds() -> None:
    dataframe = pd.DataFrame({"name": [" Alice ", "Bob  "], "sales": [1, 2]})
    actions = [{"action_type": "trim_strings", "parameters": {}}]

    result = _run_execute(dataframe, actions)

    assert [row["name"] for row in result.data["preview"]] == ["Alice", "Bob"]
    assert result.data["action_results"][0]["affected_count"] == 2


def test_execute_trim_strings_can_convert_blank_to_null() -> None:
    dataframe = pd.DataFrame({"name": ["  ", " Alice "]})
    actions = [
        {
            "action_type": "trim_strings",
            "parameters": {"convert_blank_to_null": True},
        }
    ]

    result = _run_execute(dataframe, actions)

    assert result.data["preview"][0]["name"] is None
    assert result.data["summary"]["after"]["missing_count"] == 1


def test_execute_string_to_numeric_succeeds() -> None:
    dataframe = pd.DataFrame({"amount": ["10", "20.5", None]})
    actions = [
        {
            "action_type": "convert_dtype",
            "column": "amount",
            "parameters": {"target_type": "numeric"},
        }
    ]

    result = _run_execute(dataframe, actions)
    action_result = result.data["action_results"][0]

    assert result.data["preview"][1]["amount"] == 20.5
    assert action_result["conversion_failure_count"] == 0


def test_execute_string_to_datetime_succeeds_and_serializes_timestamp() -> None:
    dataframe = pd.DataFrame({"ordered_at": ["2026-01-01", "2026-01-02"]})
    actions = [
        {
            "action_type": "convert_dtype",
            "column": "ordered_at",
            "parameters": {"target_type": "datetime"},
        }
    ]

    result = _run_execute(dataframe, actions)
    serialized = json.dumps(result.model_dump(), allow_nan=False)

    assert result.data["preview"][0]["ordered_at"].startswith("2026-01-01")
    assert "NaN" not in serialized


def test_execute_conversion_reports_failure_ratio_and_new_missing() -> None:
    dataframe = pd.DataFrame({"amount": ["10", "bad", "30"]})
    actions = [
        {
            "action_type": "convert_dtype",
            "column": "amount",
            "parameters": {"target_type": "numeric"},
        }
    ]

    result = _run_execute(dataframe, actions)
    action_result = result.data["action_results"][0]

    assert action_result["conversion_failure_count"] == 1
    assert action_result["conversion_failure_ratio"] == 0.333333
    assert action_result["new_missing_count"] == 1
    assert action_result["risk_level"] == "high"


def test_execute_does_not_modify_original_dataframe() -> None:
    dataframe = pd.DataFrame({"name": [" Alice ", "Bob"], "sales": [1.0, np.nan]})
    original = dataframe.copy(deep=True)
    actions = [
        {"action_type": "trim_strings", "parameters": {}},
        {
            "action_type": "fill_missing",
            "column": "sales",
            "parameters": {"strategy": "median"},
        },
    ]

    _run_execute(dataframe, actions)

    pdt.assert_frame_equal(dataframe, original)


def test_execute_continues_after_independent_action_failure() -> None:
    dataframe = pd.DataFrame({"name": [" Alice ", "Bob"]})
    actions = [
        {
            "action_id": "bad_fill",
            "action_type": "fill_missing",
            "column": "missing_column",
            "parameters": {"strategy": "mode"},
        },
        {"action_id": "trim", "action_type": "trim_strings", "parameters": {}},
    ]

    result = _run_execute(dataframe, actions)

    assert [item["status"] for item in result.data["action_results"]] == ["failed", "success"]
    assert result.data["preview"][0]["name"] == "Alice"


def test_execute_unknown_column_returns_clear_failure() -> None:
    dataframe = pd.DataFrame({"sales": [1.0, np.nan]})
    actions = [
        {
            "action_type": "fill_missing",
            "column": "revenue",
            "parameters": {"strategy": "median"},
        }
    ]

    result = _run_execute(dataframe, actions)

    assert result.data["action_results"][0]["status"] == "failed"
    assert "Unknown column(s): revenue" in result.data["action_results"][0]["message"]


def test_execute_unknown_action_type_returns_clear_failure() -> None:
    dataframe = pd.DataFrame({"sales": [1, 2]})

    result = _run_execute(dataframe, [{"action_type": "normalize"}])

    assert result.data["action_results"][0]["status"] == "failed"
    assert "Unsupported cleaning action_type 'normalize'" in result.data["action_results"][0]["message"]


def test_execute_preview_is_limited_and_strictly_json_serializable() -> None:
    dataframe = pd.DataFrame(
        {
            "value": np.arange(25, dtype=np.int64),
            "ratio": [np.inf, *[float(value) for value in range(24)]],
        }
    )

    result = _run_execute(dataframe, [])
    serialized = json.dumps(result.model_dump(), allow_nan=False)

    assert len(result.data["preview"]) == 20
    assert result.data["preview"][0]["ratio"] is None
    assert "Infinity" not in serialized


def test_execute_bool_column_is_not_treated_as_numeric() -> None:
    dataframe = pd.DataFrame({"active": pd.Series([True, None, False], dtype="boolean")})
    actions = [
        {
            "action_type": "fill_missing",
            "column": "active",
            "parameters": {"strategy": "mean"},
        },
        {
            "action_type": "fill_missing",
            "column": "active",
            "parameters": {"strategy": "mode"},
        },
    ]

    result = _run_execute(dataframe, actions)

    assert result.data["action_results"][0]["status"] == "failed"
    assert result.data["action_results"][1]["status"] == "success"
    assert result.data["summary"]["after"]["missing_count"] == 0


def test_execute_datetime_column_rejects_mean_or_median_fill() -> None:
    dataframe = pd.DataFrame({"ordered_at": pd.to_datetime(["2026-01-01", None])})
    actions = [
        {
            "action_type": "fill_missing",
            "column": "ordered_at",
            "parameters": {"strategy": "median"},
        }
    ]

    result = _run_execute(dataframe, actions)

    assert result.data["action_results"][0]["status"] == "failed"
    assert "only supported for numeric columns" in result.data["action_results"][0]["message"]


def test_execute_drop_missing_rows_reports_high_risk() -> None:
    dataframe = pd.DataFrame({"sales": [1.0, np.nan, np.nan, 4.0]})
    actions = [
        {
            "action_type": "drop_missing_rows",
            "parameters": {"columns": ["sales"], "how": "any"},
        }
    ]

    result = _run_execute(dataframe, actions)

    assert result.data["action_results"][0]["affected_ratio"] == 0.5
    assert result.data["action_results"][0]["risk_level"] == "high"


def test_execute_without_actions_performs_no_implicit_cleaning() -> None:
    dataframe = pd.DataFrame({"name": [" Alice ", " Alice "]})

    result = _run_execute(dataframe)

    assert result.data["summary"]["successful_action_count"] == 0
    assert result.data["summary"]["before"] == result.data["summary"]["after"]
    assert result.data["preview"][0]["name"] == " Alice "
    assert result.data["warnings"]


def test_tool_registry_finds_and_invokes_data_cleaning_tool() -> None:
    registry = ToolRegistry.with_default_tools()
    tool = registry.get("data_cleaning_tool")

    assert tool is not None
    result = tool.run(pd.DataFrame({"id": [1, 1]}), _task("data_cleaning_plan"), _context())
    assert result.type == "data_cleaning_plan"


def test_execution_service_dispatches_plan_and_execute_without_mutating_input() -> None:
    dataframe = pd.DataFrame({"ordered_at": ["2026-01-01", "2026-01-02"], "id": [1, 1]})
    original = dataframe.copy(deep=True)
    tasks = [
        _task("data_cleaning_plan"),
        _task(
            "data_cleaning_execute",
            [{"action_type": "drop_duplicates", "parameters": {"keep": "first"}}],
        ),
    ]

    response = PandasExecutionService().execute_tasks(dataframe, tasks)

    assert [result.type for result in response.execution_results] == [
        "data_cleaning_plan",
        "data_cleaning_execute",
    ]
    pdt.assert_frame_equal(dataframe, original)


def test_execution_service_build_context_does_not_modify_original_dataframe() -> None:
    dataframe = pd.DataFrame(
        {
            "ordered_at": ["2026-01-01", "invalid"],
            "sales": [10, 20],
        }
    )
    original = dataframe.copy(deep=True)

    context = PandasExecutionService()._build_context(dataframe)

    assert context.datetime_columns == ["ordered_at"]
    pdt.assert_frame_equal(dataframe, original)


def test_execution_service_gives_each_task_an_isolated_dataframe_copy() -> None:
    class MutatingTool:
        name = "mutating_test_tool"

        def run(self, dataframe, task, context):
            del context
            dataframe.loc[0, "value"] = 999
            return ExecutionResult(
                task_name=task.task_name,
                type="mutating_test",
                data={"value": int(dataframe.loc[0, "value"])},
            )

    class ObservingTool:
        name = "observing_test_tool"

        def run(self, dataframe, task, context):
            del context
            return ExecutionResult(
                task_name=task.task_name,
                type="observing_test",
                data={"value": int(dataframe.loc[0, "value"])},
            )

    registry = ToolRegistry()
    registry.register(MutatingTool())
    registry.register(ObservingTool())
    service = PandasExecutionService(registry)
    service.task_type_tools.update(
        {
            "mutating_test": "mutating_test_tool",
            "observing_test": "observing_test_tool",
        }
    )
    dataframe = pd.DataFrame({"value": [1, 2]})
    original = dataframe.copy(deep=True)
    tasks = [_task("mutating_test"), _task("observing_test")]

    response = service.execute_tasks(dataframe, tasks)

    assert response.execution_results[0].data["value"] == 999
    assert response.execution_results[1].data["value"] == 1
    pdt.assert_frame_equal(dataframe, original)


def test_existing_tools_still_resolve_after_cleaning_registration() -> None:
    service = PandasExecutionService()

    assert service._resolve_tool_name(_task("data_cleaning_plan")) == "data_cleaning_tool"
    assert service._resolve_tool_name(_task("data_cleaning_execute")) == "data_cleaning_tool"
    for task_type, tool_name in (
        ("data_quality_analysis", "data_quality_tool"),
        ("stats", "stats_tool"),
        ("groupby", "groupby_tool"),
        ("trend", "trend_tool"),
    ):
        assert service._resolve_tool_name(_task(task_type)) == tool_name
