"""Tests for enterprise anomaly detection modes and safeguards."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from app.core.exceptions import ToolExecutionException
from app.schemas.file import AnalysisTask, ExecutionResult
from app.services.pandas_execution_service import PandasExecutionService
from app.tools import AnomalyDetectionTool, DatasetContext, ToolRegistry
from app.tools.anomaly_detection_tool import (
    MAX_ANALYSIS_COLUMNS,
    MAX_ANOMALIES_PER_COLUMN,
    MAX_CHART_POINTS,
    MAX_GROUP_COUNT,
    MAX_TOTAL_ANOMALIES,
)


def _task(**params: object) -> AnalysisTask:
    return AnalysisTask(
        task_name="Anomaly Detection",
        reasoning="Identify candidate statistical anomalies without modifying data.",
        expected_output="Explainable anomaly records and bounded chart data.",
        type="anomaly_detection",
        params=params,
    )


def _run(dataframe: pd.DataFrame, **params: object) -> ExecutionResult:
    return AnomalyDetectionTool().run(
        dataframe,
        _task(**params),
        DatasetContext(numeric_columns=[], categorical_columns=[], datetime_columns=[]),
    )


def _anomalies(result: ExecutionResult, column: str = "value") -> list[dict[str, Any]]:
    return [item for item in result.data["anomalies"] if item["column"] == column]


def _contains_unsupported_object(value: object) -> bool:
    if isinstance(value, (pd.DataFrame, pd.Series, pd.Index, np.ndarray, np.generic)):
        return True
    if isinstance(value, dict):
        return any(_contains_unsupported_object(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_unsupported_object(item) for item in value)
    return False


def test_global_iqr_detects_obvious_high_anomaly() -> None:
    dataframe = pd.DataFrame({"value": [*range(10), 100.0]})

    result = _run(dataframe)
    anomaly = _anomalies(result)[0]

    assert result.data["summary"]["mode"] == "global"
    assert result.data["summary"]["method"] == "iqr"
    assert anomaly["value"] == 100.0
    assert anomaly["direction"] == "high"
    assert anomaly["reliable"] is True


def test_global_iqr_detects_obvious_low_anomaly_and_keeps_normal_values() -> None:
    dataframe = pd.DataFrame({"value": [-100.0, *range(1, 11)]})

    result = _run(dataframe)
    anomalies = _anomalies(result)

    assert [item["value"] for item in anomalies] == [-100.0]
    assert anomalies[0]["direction"] == "low"
    assert 5.0 not in {item["value"] for item in anomalies}


def test_iqr_statistics_and_bounds_are_reported() -> None:
    dataframe = pd.DataFrame({"value": [*range(10), 100.0]})

    statistics = _run(dataframe).data["columns"][0]["statistics"]

    assert statistics["q1"] == pytest.approx(2.5)
    assert statistics["q3"] == pytest.approx(7.5)
    assert statistics["iqr"] == pytest.approx(5.0)
    assert statistics["lower_bound"] == pytest.approx(-5.0)
    assert statistics["upper_bound"] == pytest.approx(15.0)


def test_iqr_zero_is_skipped_without_false_anomalies() -> None:
    dataframe = pd.DataFrame({"value": [1.0] * 9 + [100.0]})

    result = _run(dataframe)

    assert result.data["anomalies"] == []
    assert result.data["columns"][0]["status"] == "skipped"
    assert any("IQR is zero" in warning for warning in result.data["warnings"])


def test_custom_iqr_multiplier_changes_detection() -> None:
    dataframe = pd.DataFrame({"value": [*range(10), 15.0]})

    default_result = _run(dataframe)
    strict_result = _run(dataframe, iqr_multiplier=1.0)

    assert default_result.data["anomalies"] == []
    assert _anomalies(strict_result)[0]["value"] == 15.0


@pytest.mark.parametrize("multiplier", [0, -1, float("inf"), "1.5", True])
def test_invalid_iqr_multiplier_returns_clear_error(multiplier: object) -> None:
    with pytest.raises(ToolExecutionException) as exc_info:
        _run(pd.DataFrame({"value": range(10)}), iqr_multiplier=multiplier)

    assert exc_info.value.error_code == "INVALID_ANOMALY_THRESHOLD"


def test_zscore_detects_high_and_low_anomalies() -> None:
    high = _run(pd.DataFrame({"value": [0.0] * 30 + [100.0]}), method="zscore")
    low = _run(pd.DataFrame({"value": [0.0] * 30 + [-100.0]}), method="zscore")

    assert _anomalies(high)[0]["direction"] == "high"
    assert _anomalies(low)[0]["direction"] == "low"
    assert high.data["columns"][0]["statistics"]["ddof"] == 1


def test_zscore_zero_std_group_is_skipped_safely() -> None:
    dataframe = pd.DataFrame(
        {
            "group": ["constant"] * 5 + ["varying"] * 5,
            "value": [1.0] * 5 + [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    result = _run(
        dataframe,
        mode="groupwise",
        method="zscore",
        group_by="group",
        columns=["value"],
    )
    constant_group = result.data["columns"][0]["group_statistics"][0]

    assert constant_group["status"] == "skipped"
    assert constant_group["anomaly_count"] == 0
    assert constant_group["statistics"]["std"] == 0.0
    assert any("Standard deviation is zero" in warning for warning in result.data["warnings"])


def test_custom_zscore_threshold_changes_detection() -> None:
    dataframe = pd.DataFrame({"value": [0.0] * 10 + [10.0]})

    lenient = _run(dataframe, method="zscore", threshold=2.0)
    strict = _run(dataframe, method="zscore", threshold=4.0)

    assert len(_anomalies(lenient)) == 1
    assert _anomalies(strict) == []


@pytest.mark.parametrize("threshold", [0, -0.1, float("nan"), "3", False])
def test_invalid_zscore_threshold_returns_clear_error(threshold: object) -> None:
    with pytest.raises(ToolExecutionException):
        _run(pd.DataFrame({"value": range(10)}), method="zscore", threshold=threshold)


def test_robust_zscore_detects_extreme_and_reports_median_mad() -> None:
    dataframe = pd.DataFrame({"value": [*range(20), 100.0]})

    result = _run(dataframe, method="robust_zscore")
    statistics = result.data["columns"][0]["statistics"]

    assert _anomalies(result)[0]["value"] == 100.0
    assert statistics["median"] == 10.0
    assert statistics["mad"] == 5.0


def test_robust_zscore_mad_zero_is_skipped_safely() -> None:
    dataframe = pd.DataFrame({"value": [1.0] * 9 + [100.0]})

    result = _run(dataframe, method="robust_zscore")

    assert result.data["anomalies"] == []
    assert result.data["columns"][0]["status"] == "skipped"
    assert any("MAD is zero" in warning for warning in result.data["warnings"])


def test_small_sample_candidate_is_not_reliable_or_worded_as_certain() -> None:
    dataframe = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 100.0]})

    result = _run(dataframe, method="robust_zscore", threshold=3.5)
    anomaly = _anomalies(result)[0]

    assert anomaly["reliable"] is False
    assert "low-confidence candidate" in anomaly["message"]
    assert result.data["summary"]["reliable_anomaly_count"] == 0


def test_unsupported_types_are_excluded_before_numeric_detection() -> None:
    dataframe = pd.DataFrame(
        {
            "value": range(10),
            "active": [True, False] * 5,
            "ordered_at": pd.date_range("2026-01-01", periods=10),
            "elapsed": pd.to_timedelta(range(10), unit="D"),
            "text": ["a", "b"] * 5,
            "category": pd.Series(["x", "y"] * 5, dtype="category"),
        }
    )

    result = _run(dataframe)
    excluded = {item["column"]: item["reason"] for item in result.data["excluded_columns"]}

    assert [item["column"] for item in result.data["columns"]] == ["value"]
    assert excluded["active"] == "boolean"
    assert excluded["ordered_at"] == "datetime"
    assert excluded["elapsed"] == "timedelta"
    assert excluded["text"] in {"string", "object"}
    assert excluded["category"] == "category"


def test_constant_and_all_null_columns_are_excluded() -> None:
    dataframe = pd.DataFrame(
        {"value": range(10), "constant": [1.0] * 10, "empty": [np.nan] * 10}
    )

    result = _run(dataframe)
    excluded = {item["column"]: item["reason"] for item in result.data["excluded_columns"]}

    assert excluded["constant"] == "constant"
    assert excluded["empty"] == "all_null_or_non_finite"


def test_explicit_numeric_columns_work_and_invalid_fields_fail() -> None:
    dataframe = pd.DataFrame(
        {"sales": range(10), "price": np.arange(10) * 2, "region": ["x"] * 10}
    )

    result = _run(dataframe, columns=["price"])
    assert [item["column"] for item in result.data["columns"]] == ["price"]

    with pytest.raises(ToolExecutionException) as unknown:
        _run(dataframe, columns=["missing"])
    with pytest.raises(ToolExecutionException) as non_numeric:
        _run(dataframe, columns=["region"])
    assert unknown.value.error_code == "UNKNOWN_ANOMALY_COLUMNS"
    assert non_numeric.value.error_code == "NON_NUMERIC_ANOMALY_COLUMNS"


def test_original_dataframe_is_never_modified() -> None:
    dataframe = pd.DataFrame(
        {"value": [1.0, np.nan, np.inf, 4.0, 100.0], "region": ["a"] * 5}
    )
    original = dataframe.copy(deep=True)

    _run(dataframe)

    pdt.assert_frame_equal(dataframe, original)


def _groupwise_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "region": ["East"] * 10 + ["West"] * 10,
            "sales": [10, 11, 9, 10, 12, 8, 11, 10, 9, 100]
            + [100, 101, 99, 100, 102, 98, 101, 100, 99, 100],
        }
    )


def test_groupwise_uses_independent_thresholds_and_finds_local_anomaly() -> None:
    result = _run(
        _groupwise_frame(),
        mode="groupwise",
        group_by="region",
        columns=["sales"],
    )
    anomalies = _anomalies(result, "sales")
    groups = result.data["columns"][0]["group_statistics"]

    assert len(groups) == 2
    assert groups[0]["statistics"]["upper_bound"] != groups[1]["statistics"]["upper_bound"]
    assert len(anomalies) == 1
    assert anomalies[0]["group"] == "East"
    assert anomalies[0]["value"] == 100.0


def test_groupwise_small_group_is_skipped_with_warning() -> None:
    dataframe = pd.DataFrame(
        {
            "region": ["Small"] * 4 + ["Large"] * 10,
            "sales": [1, 2, 3, 100] + [*range(10)],
        }
    )

    result = _run(dataframe, mode="groupwise", group_by="region", columns=["sales"])
    small = result.data["columns"][0]["group_statistics"][0]

    assert small["status"] == "skipped"
    assert small["anomaly_count"] == 0
    assert any("at least 5" in warning for warning in result.data["warnings"])


def test_groupwise_missing_group_uses_special_label() -> None:
    dataframe = pd.DataFrame(
        {
            "region": [None] * 5 + ["West"] * 5,
            "sales": [1, 2, 3, 4, 100, 10, 11, 12, 13, 14],
        }
    )

    result = _run(dataframe, mode="groupwise", group_by="region", columns=["sales"])

    assert "__MISSING__" in result.data["chart"]["groups"]
    assert result.data["metadata"]["missing_group_label"] == "__MISSING__"


def test_groupwise_requires_existing_group_by() -> None:
    dataframe = pd.DataFrame({"sales": range(10)})

    with pytest.raises(ToolExecutionException) as missing:
        _run(dataframe, mode="groupwise")
    with pytest.raises(ToolExecutionException) as unknown:
        _run(dataframe, mode="groupwise", group_by="region")
    assert missing.value.error_code == "MISSING_GROUP_BY"
    assert unknown.value.error_code == "UNKNOWN_GROUP_BY"


def test_group_count_limit_is_stable_and_reported() -> None:
    dataframe = pd.DataFrame(
        {
            "group": [f"g{group}" for group in range(MAX_GROUP_COUNT + 1) for _ in range(5)],
            "value": [value for _ in range(MAX_GROUP_COUNT + 1) for value in range(5)],
        }
    )

    first = _run(dataframe, mode="groupwise", group_by="group", columns=["value"])
    second = _run(dataframe, mode="groupwise", group_by="group", columns=["value"])

    assert first.data["metadata"]["original_group_count"] == MAX_GROUP_COUNT + 1
    assert first.data["metadata"]["actual_group_count"] == MAX_GROUP_COUNT
    assert first.data["chart"]["groups"] == second.data["chart"]["groups"]
    assert first.data["warnings"]


def _time_frame(values: list[float], dates: list[object] | None = None) -> pd.DataFrame:
    if dates is None:
        dates = list(pd.date_range("2026-01-01", periods=len(values)))
    return pd.DataFrame({"date": dates, "sales": values})


def test_rolling_zscore_uses_only_shifted_history_and_detects_spike() -> None:
    values = [10, 11, 9, 10, 11, 9, 10, 100, 10, 11, 9, 10]
    result = _run(
        _time_frame(values),
        mode="time_series",
        time_column="date",
        value_columns=["sales"],
    )
    anomalies = _anomalies(result, "sales")

    assert anomalies[0]["value"] == 100.0
    assert anomalies[0]["threshold_info"]["rolling_mean"] == pytest.approx(10.0)
    assert result.data["metadata"]["future_data_leakage_prevented"] is True
    assert result.data["columns"][0]["statistics"]["shift"] == 1
    assert result.data["chart"]["anomaly_scores"][:7] == [None] * 7


def test_rolling_zscore_detects_sudden_drop() -> None:
    values = [100, 101, 99, 100, 101, 99, 100, 0, 100, 101]
    result = _run(
        _time_frame(values),
        mode="time_series",
        time_column="date",
        value_columns=["sales"],
    )

    assert _anomalies(result, "sales")[0]["direction"] == "low"


def test_rolling_window_warmup_and_zero_std_do_not_flag_anomalies() -> None:
    result = _run(
        _time_frame([10.0] * 7 + [100.0]),
        mode="time_series",
        time_column="date",
        value_columns=["sales"],
    )

    assert result.data["anomalies"] == []
    assert result.data["chart"]["anomaly_flags"] == [False] * 8
    assert result.data["warnings"]


def test_time_series_is_stably_sorted_and_duplicate_times_do_not_crash() -> None:
    dates = ["2026-01-03", "2026-01-01", "2026-01-02", "2026-01-02", "2026-01-04"]
    dataframe = _time_frame([3, 1, 2, 2.5, 4], dates)

    result = _run(
        dataframe,
        mode="time_series",
        method="iqr",
        time_column="date",
        value_columns=["sales"],
    )

    assert result.data["chart"]["x"] == sorted(result.data["chart"]["x"])
    assert len(result.data["chart"]["x"]) == 5


def test_time_parse_failure_ratio_is_reported_or_rejected() -> None:
    allowed_dates = [f"2026-01-{day:02d}" for day in range(1, 20)] + ["bad"]
    rejected_dates = [f"2026-01-{day:02d}" for day in range(1, 19)] + ["bad", "worse"]

    allowed = _run(
        _time_frame(list(range(20)), allowed_dates),
        mode="time_series",
        method="iqr",
        time_column="date",
        value_columns=["sales"],
    )
    assert allowed.data["metadata"]["time_parse_failure_ratio"] == 0.05
    assert allowed.data["warnings"]

    with pytest.raises(ToolExecutionException) as exc_info:
        _run(
            _time_frame(list(range(20)), rejected_dates),
            mode="time_series",
            method="iqr",
            time_column="date",
            value_columns=["sales"],
        )
    assert exc_info.value.error_code == "TIME_PARSE_FAILURE_RATIO_EXCEEDED"


@pytest.mark.parametrize(
    ("params", "error_code"),
    [
        ({"mode": "time_series", "value_columns": ["sales"]}, "MISSING_TIME_COLUMN"),
        ({"mode": "time_series", "time_column": "date"}, "MISSING_VALUE_COLUMNS"),
        (
            {
                "mode": "time_series",
                "time_column": "date",
                "value_columns": ["sales"],
                "window": 2,
            },
            "INVALID_ROLLING_WINDOW",
        ),
    ],
)
def test_time_series_required_parameters_are_validated(
    params: dict[str, object],
    error_code: str,
) -> None:
    dataframe = _time_frame(list(range(10)))

    with pytest.raises(ToolExecutionException) as exc_info:
        _run(dataframe, **params)
    assert exc_info.value.error_code == error_code


def test_numeric_time_column_is_not_silently_interpreted_as_datetime() -> None:
    dataframe = pd.DataFrame({"time": range(10), "sales": range(10)})

    with pytest.raises(ToolExecutionException) as exc_info:
        _run(
            dataframe,
            mode="time_series",
            time_column="time",
            value_columns=["sales"],
        )
    assert exc_info.value.error_code == "INVALID_TIME_COLUMN_TYPE"


def test_sample_size_below_two_and_below_minimum_are_skipped() -> None:
    one_sample = _run(pd.DataFrame({"value": [1.0, np.nan]}))
    four_samples = _run(pd.DataFrame({"value": [1.0, 2.0, 3.0, 100.0]}))

    assert one_sample.data["columns"][0]["status"] == "skipped"
    assert four_samples.data["columns"][0]["status"] == "skipped"
    assert one_sample.data["anomalies"] == four_samples.data["anomalies"] == []
    assert one_sample.data["warnings"] and four_samples.data["warnings"]


def test_empty_or_non_numeric_dataframe_returns_clear_error() -> None:
    with pytest.raises(ToolExecutionException, match="non-empty dataframe"):
        _run(pd.DataFrame({"value": []}))
    with pytest.raises(ToolExecutionException, match="no numeric value columns"):
        _run(pd.DataFrame({"region": ["East", "West"]}))


def test_anomaly_record_is_complete_and_string_index_is_json_safe() -> None:
    dataframe = pd.DataFrame(
        {"value": [*range(10), 100.0]},
        index=[f"row-{index}" for index in range(11)],
    )

    anomaly = _anomalies(_run(dataframe))[0]

    assert anomaly["row_reference"] == {"index": "row-10", "position": 10}
    assert {
        "column", "value", "mode", "method", "anomaly_score", "direction", "severity",
        "reliable", "group", "time", "threshold_info", "message",
    }.issubset(anomaly)


def test_anomalies_are_sorted_by_absolute_score_stably() -> None:
    dataframe = pd.DataFrame({"value": [*range(20), 100.0, 200.0]})

    anomalies = _anomalies(_run(dataframe))
    scores = [abs(item["anomaly_score"]) for item in anomalies]

    assert scores == sorted(scores, reverse=True)
    assert anomalies[0]["value"] == 200.0


def test_per_column_and_total_anomaly_limits_are_enforced() -> None:
    baseline = [float(value % 10) for value in range(1000)]
    one_column = pd.DataFrame({"value": baseline + [1000.0] * 150})
    per_column = _run(one_column)

    assert per_column.data["summary"]["detected_anomaly_count"] == 150
    assert per_column.data["summary"]["returned_anomaly_count"] == MAX_ANOMALIES_PER_COLUMN
    assert per_column.data["metadata"]["truncated"] is True

    many_columns = pd.DataFrame(
        {
            f"value_{column}": baseline + [1000.0 + column] * 120
            for column in range(6)
        }
    )
    total = _run(many_columns)
    assert total.data["summary"]["detected_anomaly_count"] == 720
    assert total.data["summary"]["returned_anomaly_count"] == MAX_TOTAL_ANOMALIES
    assert total.data["summary"]["truncated"] is True


def test_chart_and_column_limits_are_bounded_and_stable() -> None:
    dataframe = pd.DataFrame(
        {
            f"value_{column}": np.arange(MAX_CHART_POINTS + 100) * (column + 1)
            for column in range(MAX_ANALYSIS_COLUMNS + 2)
        }
    )

    first = _run(dataframe)
    second = _run(dataframe)

    assert len(first.data["columns"]) == MAX_ANALYSIS_COLUMNS
    assert len(first.data["chart"]["x"]) <= MAX_CHART_POINTS
    assert first.data["metadata"]["original_eligible_column_count"] == MAX_ANALYSIS_COLUMNS + 2
    assert [item["column"] for item in first.data["columns"]] == [
        item["column"] for item in second.data["columns"]
    ]
    assert first.data["warnings"]


def test_explicit_columns_over_limit_raise_clear_error() -> None:
    columns = [f"value_{index}" for index in range(MAX_ANALYSIS_COLUMNS + 1)]
    dataframe = pd.DataFrame({column: range(10) for column in columns})

    with pytest.raises(ToolExecutionException) as exc_info:
        _run(dataframe, columns=columns)
    assert exc_info.value.error_code == "ANOMALY_COLUMN_LIMIT_EXCEEDED"


def test_output_is_strictly_json_safe_with_non_finite_and_timestamp_values() -> None:
    dataframe = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=12),
            "sales": [1.0, 2.0, np.nan, np.inf, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 100.0],
        }
    )

    result = _run(
        dataframe,
        mode="time_series",
        method="iqr",
        time_column="date",
        value_columns=["sales"],
    )
    payload = result.model_dump()
    serialized = json.dumps(payload, allow_nan=False)

    assert "NaN" not in serialized
    assert "Infinity" not in serialized
    assert not _contains_unsupported_object(payload)


def test_deletion_request_remains_detection_only() -> None:
    result = _run(
        pd.DataFrame({"value": [*range(10), 100.0]}),
        requested_deletion=True,
    )

    assert result.data["metadata"]["detection_only"] is True
    assert result.data["metadata"]["original_dataframe_modified"] is False
    assert any("does not modify or remove" in warning for warning in result.data["warnings"])


@pytest.mark.parametrize(
    "params",
    [
        {"mode": "unknown"},
        {"mode": "global", "method": "rolling_zscore"},
        {"mode": "time_series", "method": "zscore"},
        {"mode": "global", "method": "isolation_forest"},
    ],
)
def test_unknown_or_incompatible_mode_method_returns_clear_error(
    params: dict[str, object],
) -> None:
    dataframe = _time_frame(list(range(10)))

    with pytest.raises(ToolExecutionException):
        _run(dataframe, **params)


def test_registry_execution_keyword_and_existing_mappings_are_preserved() -> None:
    registry = ToolRegistry.with_default_tools()
    tool = registry.get("anomaly_detection_tool")
    assert isinstance(tool, AnomalyDetectionTool)

    service = PandasExecutionService(registry)
    dataframe = pd.DataFrame({"value": [*range(10), 100.0]})
    response = service.execute_tasks(dataframe, [_task()])
    assert response.execution_results[0].type == "anomaly_detection"

    keyword_task = AnalysisTask(
        task_name="Find outliers",
        reasoning="Run anomaly detection.",
        expected_output="Candidate anomaly records.",
        params={},
    )
    assert service._resolve_tool_name(keyword_task) == "anomaly_detection_tool"
    for task_type, tool_name in (
        ("correlation_analysis", "correlation_analysis_tool"),
        ("data_quality_analysis", "data_quality_tool"),
        ("data_cleaning_plan", "data_cleaning_tool"),
        ("data_cleaning_execute", "data_cleaning_tool"),
        ("stats", "stats_tool"),
        ("groupby", "groupby_tool"),
        ("trend", "trend_tool"),
        ("sql", "sql_tool"),
    ):
        task = AnalysisTask(
            task_name="Existing Task",
            reasoning="Preserve explicit routing.",
            expected_output="Existing output.",
            type=task_type,
            params={},
        )
        assert service._resolve_tool_name(task) == tool_name


def test_multiple_execution_tasks_remain_dataframe_isolated() -> None:
    class MutatingTool:
        name = "mutating_anomaly_test_tool"

        def run(self, dataframe, task, context):
            del context
            dataframe.loc[0, "value"] = 999
            return ExecutionResult(task_name=task.task_name, type="mutating", data={})

    registry = ToolRegistry.with_default_tools()
    registry.register(MutatingTool())
    service = PandasExecutionService(registry)
    service.task_type_tools["mutating"] = "mutating_anomaly_test_tool"
    dataframe = pd.DataFrame({"value": [*range(10), 100.0]})
    original = dataframe.copy(deep=True)
    mutating_task = AnalysisTask(
        task_name="Mutate isolated copy",
        reasoning="Test isolation.",
        expected_output="No output.",
        type="mutating",
        params={},
    )

    response = service.execute_tasks(dataframe, [mutating_task, _task()])

    assert response.execution_results[1].data["anomalies"][0]["value"] == 100.0
    pdt.assert_frame_equal(dataframe, original)
