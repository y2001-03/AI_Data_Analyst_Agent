"""Tests for ordered baseline forecast analysis."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from app.core.exceptions import ToolExecutionException
from app.schemas.file import AnalysisTask
from app.services.pandas_execution_service import PandasExecutionService
from app.tools import DatasetContext, ForecastAnalysisTool, ToolRegistry
from app.tools.forecast_analysis_tool import EVALUATION_PROTOCOL, MAX_FORECAST_GROUPS


def _task(**params: object) -> AnalysisTask:
    return AnalysisTask(task_name="Forecast", reasoning="Predict future values with backtesting.",
                        expected_output="Forecast and metrics.", type="forecast_analysis", params=params)


def _context() -> DatasetContext:
    return DatasetContext(numeric_columns=["sales"], categorical_columns=["region"], datetime_columns=["date"])


def _frame(n: int = 40) -> pd.DataFrame:
    return pd.DataFrame({"date": pd.date_range("2026-01-01", periods=n, freq="D"),
                         "sales": [float(i) for i in range(n)],
                         "region": (["a", "b"] * ((n + 1) // 2))[:n]})


def _run(df: pd.DataFrame, **params: object):
    defaults = {"time_column": "date", "target_column": "sales", "horizon": 7}
    defaults.update(params)
    return ForecastAnalysisTool().run(df, _task(**defaults), _context())


def test_datetime_and_string_time_are_sorted_and_profiled() -> None:
    df = _frame().sort_values("date", ascending=False).reset_index(drop=True)
    df["date"] = df["date"].astype(str)
    result = _run(df, method="naive")
    profile = result.data["series_profile"]
    assert profile["sorted_by_time"] is True
    assert profile["invalid_time_count"] == 0
    assert result.data["backtest"]["timestamps"] == sorted(result.data["backtest"]["timestamps"])


def test_time_parse_failures_are_counted_or_rejected() -> None:
    safe = _frame(40); safe["date"] = safe["date"].astype(object); safe.loc[0, "date"] = "bad"
    result = _run(safe, method="naive")
    assert result.data["series_profile"]["invalid_time_count"] == 1
    bad = _frame(40); bad["date"] = bad["date"].astype(object); bad.loc[:2, "date"] = "bad"
    with pytest.raises(ToolExecutionException) as exc:
        _run(bad)
    assert exc.value.error_code == "FORECAST_TIME_PARSE_FAILURE"


@pytest.mark.parametrize("strategy,expected", [("aggregate", 3.0), ("first", 1.0), ("last", 2.0)])
def test_duplicate_time_strategies(strategy: str, expected: float) -> None:
    df = _frame(40)
    extra = df.iloc[[0]].copy(); extra["sales"] = 2.0
    df.loc[0, "sales"] = 1.0
    result = _run(pd.concat([df, extra], ignore_index=True), method="naive",
                  duplicate_time_strategy=strategy, aggregation="sum")
    assert result.data["series_profile"]["duplicate_timestamp_count"] == 2
    if strategy == "aggregate":
        assert result.data["chart"]["historical"]["y"][0] == expected


def test_duplicate_aggregate_mean_and_error() -> None:
    df = _frame(40); extra = df.iloc[[0]].copy(); extra["sales"] = 3.0; df.loc[0, "sales"] = 1.0
    mean = _run(pd.concat([df, extra]), method="naive", aggregation="mean")
    assert mean.data["chart"]["historical"]["y"][0] == 2.0
    with pytest.raises(ToolExecutionException) as exc:
        _run(pd.concat([df, extra]), duplicate_time_strategy="error")
    assert exc.value.error_code == "DUPLICATE_FORECAST_TIMESTAMPS"


def test_target_validation_infinity_and_original_dataframe() -> None:
    df = _frame(); original = df.copy(deep=True); df.loc[0, "sales"] = np.inf
    result = _run(df, method="naive")
    assert any("infinite target" in warning for warning in result.data["warnings"])
    pdt.assert_frame_equal(df, df.copy(deep=True))
    bad = df.assign(text="x")
    with pytest.raises(ToolExecutionException):
        ForecastAnalysisTool().run(bad, _task(time_column="date", target_column="text"), _context())
    bool_df = df.assign(flag=True)
    with pytest.raises(ToolExecutionException):
        ForecastAnalysisTool().run(bool_df, _task(time_column="date", target_column="flag"), _context())
    pdt.assert_frame_equal(original.drop(index=[]), original)


@pytest.mark.parametrize("freq,label", [("D", "daily"), ("W", "weekly"), ("MS", "monthly")])
def test_frequency_recognition_and_future_timestamps(freq: str, label: str) -> None:
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=40, freq=freq), "sales": range(40)})
    result = _run(df, method="naive", horizon=3)
    assert result.data["summary"]["inferred_frequency"] == label
    future = result.data["forecast"]
    assert len(future) == 3 and future[0]["timestamp"] < future[-1]["timestamp"]


def test_irregular_frequency_warns_and_single_timestamp_fails() -> None:
    df = _frame(); df.loc[20:, "date"] += pd.Timedelta(days=3)
    result = _run(df, method="naive")
    assert any("irregular" in warning for warning in result.data["warnings"])
    one_time = _frame(); one_time["date"] = pd.Timestamp("2026-01-01")
    with pytest.raises(ToolExecutionException) as exc:
        _run(one_time)
    assert exc.value.error_code == "UNKNOWN_FORECAST_INTERVAL"


@pytest.mark.parametrize("strategy", ["drop", "forward_fill", "interpolate"])
def test_missing_target_strategies_are_safe(strategy: str) -> None:
    df = _frame(); df.loc[[3, 10, 20], "sales"] = np.nan
    result = _run(df, method="naive", missing_target_strategy=strategy)
    assert result.data["summary"]["train_size"] >= 10
    assert all(value is not None for value in result.data["backtest"]["actual"])


def test_missing_error_high_ratio_and_all_empty() -> None:
    df = _frame(); df.loc[0, "sales"] = np.nan
    with pytest.raises(ToolExecutionException) as exc:
        _run(df, missing_target_strategy="error")
    assert exc.value.error_code == "MISSING_FORECAST_TARGET"
    high = _frame(); high.loc[:9, "sales"] = np.nan
    result = _run(high, method="naive", missing_target_strategy="forward_fill")
    assert any("exceeds 20%" in warning for warning in result.data["warnings"])
    empty = _frame(); empty["sales"] = np.nan
    with pytest.raises(ToolExecutionException): _run(empty)


def test_ordered_split_validation_steps_and_ratio() -> None:
    steps = _run(_frame(), method="naive", validation_steps=5)
    ratio = _run(_frame(), method="naive", validation_size=0.25)
    assert steps.data["backtest"]["validation_size"] == 5
    assert ratio.data["backtest"]["validation_size"] == 10
    assert max(steps.data["chart"]["historical"]["x"][:-5]) < steps.data["backtest"]["timestamps"][0]


def test_insufficient_train_and_validation_errors() -> None:
    with pytest.raises(ToolExecutionException): _run(_frame(15), validation_steps=6)
    with pytest.raises(ToolExecutionException): _run(_frame(), validation_steps=2)


def test_naive_backtest_future_and_constant_series() -> None:
    result = _run(_frame(), method="naive", validation_steps=5, horizon=3)
    assert result.data["backtest"]["predicted"] == [34.0, 35.0, 36.0, 37.0, 38.0]
    assert [x["prediction"] for x in result.data["forecast"]] == [39.0] * 3
    constant = _frame(); constant["sales"] = 5.0
    assert all(x["prediction"] == 5.0 for x in _run(constant, method="naive").data["forecast"])


def test_seasonal_naive_period_mapping_and_validation() -> None:
    df = _frame(); df["sales"] = [float(i % 7) for i in range(40)]
    result = _run(df, method="seasonal_naive", season_length=7, validation_steps=7, horizon=8)
    assert result.data["backtest"]["actual"] == result.data["backtest"]["predicted"]
    assert result.data["forecast"][0]["prediction"] == result.data["forecast"][7]["prediction"]
    with pytest.raises(ToolExecutionException): _run(df, method="seasonal_naive", season_length=1)
    with pytest.raises(ToolExecutionException): _run(_frame(15), method="seasonal_naive", season_length=7)


def test_moving_average_walk_forward_and_recursive_forecast() -> None:
    result = _run(_frame(), method="moving_average", window=3, validation_steps=3, horizon=2)
    assert result.data["backtest"]["predicted"][0] == pytest.approx(35.0)
    assert result.data["backtest"]["predicted"][1] == pytest.approx(36.0)
    assert result.data["forecast"][0]["prediction"] == pytest.approx(38.0)
    assert result.data["forecast"][1]["prediction"] == pytest.approx((38 + 39 + 38) / 3)
    with pytest.raises(ToolExecutionException): _run(_frame(15), method="moving_average", window=12)


@pytest.mark.parametrize("slope,direction", [(2.0, "increasing"), (-2.0, "decreasing"), (0.0, "flat")])
def test_linear_trend_parameters_direction_and_no_validation_fit(slope: float, direction: str) -> None:
    df = _frame(); df["sales"] = [10 + slope * i for i in range(40)]
    result = _run(df, method="linear_trend", validation_steps=5, horizon=2)
    assert result.data["summary"]["trend_direction"] == direction
    assert result.data["backtest"]["predicted"][0] == pytest.approx(10 + slope * 35)
    if slope != 0: assert result.data["model_parameters"]["slope"] == pytest.approx(slope)


def test_linear_negative_predictions_clip_only_when_requested() -> None:
    df = _frame(); df["sales"] = [40 - 2 * i for i in range(40)]
    raw = _run(df, method="linear_trend", horizon=3)
    clipped = _run(df, method="linear_trend", horizon=3, clip_lower=0)
    assert any(x["prediction"] < 0 for x in raw.data["forecast"])
    assert all(x["prediction"] >= 0 and (x["lower_bound"] is None or x["lower_bound"] >= 0)
               for x in clipped.data["forecast"])


def test_auto_outputs_candidates_and_selects_by_primary_metric() -> None:
    result = _run(_frame(), method="auto", primary_metric="mae", validation_steps=5)
    assert {x["model"] for x in result.data["model_comparison"]} == {"naive", "moving_average", "linear_trend"}
    assert result.data["summary"]["selected_model"] == "linear_trend"
    rmse = _run(_frame(), method="auto", primary_metric="rmse", validation_steps=5)
    assert rmse.data["summary"]["selected_model"] == "linear_trend"
    with pytest.raises(ToolExecutionException): _run(_frame(), primary_metric="unknown")


def test_metrics_are_correct_and_zero_safe() -> None:
    metrics = ForecastAnalysisTool()._metrics([0.0, 2.0], [0.0, 1.0])
    assert metrics["mae"] == pytest.approx(0.5)
    assert metrics["rmse"] == pytest.approx(np.sqrt(0.5))
    assert metrics["mape"] == pytest.approx(50.0) and metrics["mape_sample_count"] == 1
    assert metrics["smape"] == pytest.approx(200 / 3)
    assert ForecastAnalysisTool()._metrics([0.0], [1.0])["mape"] is None
    json.dumps(metrics, allow_nan=False)


def test_backtest_forecast_interval_chart_and_json_contract() -> None:
    result = _run(_frame(), method="naive", validation_steps=6, horizon=4)
    backtest = result.data["backtest"]
    assert len(backtest["actual"]) == len(backtest["predicted"]) == len(backtest["timestamps"])
    assert [x["step"] for x in result.data["forecast"]] == [1, 2, 3, 4]
    assert all(x["lower_bound"] is not None for x in result.data["forecast"])
    assert result.data["metadata"]["not_a_formal_confidence_interval"] is True
    assert result.data["chart"]["type"] == "forecast_line"
    assert set(result.data["chart"]) == {"type", "historical", "validation", "future"}
    assert result.chart is None
    json.dumps(result.model_dump(), allow_nan=False)


def test_interval_insufficient_residuals_and_backtest_display_truncation() -> None:
    short = _run(_frame(), method="naive", validation_steps=4)
    assert all(x["lower_bound"] is None and x["upper_bound"] is None for x in short.data["forecast"])
    long_df = _frame(320)
    long = _run(long_df, method="naive", validation_steps=250)
    assert len(long.data["backtest"]["actual"]) == 200
    assert long.data["backtest"]["display_truncated"] is True
    assert long.data["backtest"]["metrics"]["mae"] == pytest.approx(1.0)


def test_naive_walk_forward_uses_only_previously_observed_validation_values() -> None:
    tool = ForecastAnalysisTool()
    train = [1.0, 2.0, 3.0]
    baseline, _ = tool._backtest_naive(train, [4.0, 5.0, 6.0], {})
    changed, _ = tool._backtest_naive(train, [40.0, 5.0, 6.0], {})
    assert baseline == [3.0, 4.0, 5.0]
    assert changed[0] == baseline[0]
    assert changed[1] == 40.0


def test_seasonal_naive_walk_forward_does_not_use_current_or_future_values() -> None:
    tool = ForecastAnalysisTool()
    train = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
    baseline, _ = tool._backtest_seasonal_naive(train, [4.0, 5.0, 6.0, 7.0], {"season_length": 3})
    changed, _ = tool._backtest_seasonal_naive(train, [40.0, 50.0, 60.0, 7.0], {"season_length": 3})
    assert changed[:3] == baseline[:3]
    assert changed[3] == 40.0


def test_moving_average_walk_forward_does_not_use_current_or_future_values() -> None:
    tool = ForecastAnalysisTool()
    train = [1.0, 2.0, 3.0, 4.0]
    baseline, _ = tool._backtest_moving_average(train, [5.0, 6.0, 7.0], {"window": 2})
    changed, _ = tool._backtest_moving_average(train, [50.0, 60.0, 7.0], {"window": 2})
    assert changed[0] == baseline[0] == pytest.approx(3.5)
    assert changed[1] == pytest.approx(27.0)


def test_linear_trend_walk_forward_refits_using_only_available_history() -> None:
    tool = ForecastAnalysisTool()
    train = [1.0, 2.0, 3.0, 4.0]
    baseline, params = tool._backtest_linear_trend(train, [5.0, 6.0, 7.0], {})
    changed, _ = tool._backtest_linear_trend(train, [50.0, 6.0, 7.0], {})
    assert baseline == pytest.approx([5.0, 6.0, 7.0])
    assert changed[0] == pytest.approx(baseline[0])
    assert changed[1] != pytest.approx(baseline[1])
    assert params == {"refit_each_step": True, "refit_count": 3}


@pytest.mark.parametrize(
    "method,cfg",
    [
        ("_backtest_naive", {}),
        ("_backtest_seasonal_naive", {"season_length": 3}),
        ("_backtest_moving_average", {"window": 2}),
        ("_backtest_linear_trend", {}),
    ],
)
def test_changing_last_validation_value_never_changes_earlier_predictions(method: str, cfg: dict[str, object]) -> None:
    tool = ForecastAnalysisTool()
    train = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    baseline, _ = getattr(tool, method)(train, [7.0, 8.0, 9.0, 10.0], cfg)
    changed, _ = getattr(tool, method)(train, [7.0, 8.0, 9.0, 1000.0], cfg)
    assert changed == pytest.approx(baseline)


def test_auto_candidates_share_walk_forward_protocol_and_validation_points() -> None:
    tool = ForecastAnalysisTool()
    valid = [21.0, 22.0, 23.0, 24.0, 25.0]
    comparison = tool._compare_models(
        [float(value) for value in range(1, 21)],
        valid,
        {"window": 3, "season_length": 4},
    )
    successful = [candidate for candidate in comparison if candidate["status"] == "success"]
    assert successful
    assert all(candidate["evaluation_protocol"] == EVALUATION_PROTOCOL for candidate in successful)
    assert all(candidate["validation_point_count"] == len(valid) for candidate in successful)
    assert all(len(candidate["predicted"]) == len(valid) for candidate in successful)
    result = _run(_frame(), method="auto", validation_steps=5)
    assert result.data["backtest"]["evaluation_protocol"] == EVALUATION_PROTOCOL
    assert result.data["metadata"]["evaluation_protocol"] == EVALUATION_PROTOCOL


def test_interpolate_is_causal_linear_extrapolation_without_future_values() -> None:
    tool = ForecastAnalysisTool()
    times = list(pd.date_range("2026-01-01", periods=4, freq="D"))
    baseline_times, baseline = tool._handle_segment(times, [1.0, np.nan, 4.0, 5.0], "interpolate", 0.0)
    changed_times, changed = tool._handle_segment(times, [1.0, np.nan, 400.0, 500.0], "interpolate", 0.0)
    assert baseline_times == changed_times
    assert baseline[1] == changed[1] == pytest.approx(2.0)
    result = _run(_frame(), method="naive", missing_target_strategy="interpolate")
    assert result.data["metadata"]["interpolate_strategy_definition"] == (
        "interpolate strategy uses causal linear extrapolation from past observations and does not use future values"
    )


@pytest.mark.parametrize("horizon", [0, 366, 1.5, True])
def test_invalid_horizon_errors(horizon: object) -> None:
    with pytest.raises(ToolExecutionException): _run(_frame(), horizon=horizon)


def test_grouped_forecast_independence_missing_and_small_groups() -> None:
    a = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=30), "sales": range(30), "region": "a"})
    b = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=30), "sales": range(100, 130), "region": None})
    small = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=10), "sales": range(10), "region": "small"})
    result = _run(pd.concat([a, b, small]), mode="grouped", group_by="region", method="naive", horizon=2)
    assert result.data["summary"]["successful_group_count"] == 2
    assert result.data["summary"]["skipped_group_count"] == 1
    assert {x["group"] for x in result.data["forecast"]} == {"a", "__MISSING__"}
    assert len(result.data["forecast"]) == 4


def test_group_limit_and_main_chart_selection_are_stable() -> None:
    rows = []
    for group in range(MAX_FORECAST_GROUPS + 1):
        for i in range(15): rows.append({"date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i), "sales": i, "region": f"g{group}"})
    result = _run(pd.DataFrame(rows), mode="grouped", group_by="region", method="naive", horizon=1)
    assert result.data["summary"]["successful_group_count"] == MAX_FORECAST_GROUPS
    assert result.data["metadata"]["selected_chart_group"] == "g0"
    assert any("Group count exceeded" in warning for warning in result.data["warnings"])


def test_group_failure_isolated_and_explicit_chart_group() -> None:
    good = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=30), "sales": range(30), "region": "good"})
    chosen = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=30), "sales": range(100, 130), "region": "chosen"})
    bad = pd.DataFrame({"date": ["bad"] * 30, "sales": range(30), "region": "bad"})
    result = _run(pd.concat([good, chosen, bad]), mode="grouped", group_by="region", method="naive",
                  horizon=2, chart_group="chosen")
    assert result.data["summary"]["successful_group_count"] == 2
    assert result.data["summary"]["failed_group_count"] == 1
    assert result.data["metadata"]["selected_chart_group"] == "chosen"


def test_reliability_notice_registry_execution_and_task_isolation() -> None:
    small = _run(_frame(20), method="naive")
    large = _run(_frame(40), method="naive")
    assert small.data["summary"]["reliable"] is False
    assert large.data["summary"]["reliable"] is True
    assert "do not guarantee" in large.data["metadata"]["notice"]
    registry = ToolRegistry.with_default_tools(); service = PandasExecutionService(registry)
    assert isinstance(registry.get("forecast_analysis_tool"), ForecastAnalysisTool)
    assert service._resolve_tool_name(_task()) == "forecast_analysis_tool"
    keyword = AnalysisTask(task_name="Future forecast", reasoning="Predict future sales", expected_output="Backtest", params={})
    assert service._resolve_tool_name(keyword) == "forecast_analysis_tool"
    df = _frame(); original = df.copy(deep=True)
    response = service.execute_tasks(df, [_task(time_column="date", target_column="sales", method="naive")])
    assert response.execution_results[0].type == "forecast_analysis"
    pdt.assert_frame_equal(df, original)
