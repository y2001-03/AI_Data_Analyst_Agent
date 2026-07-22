"""Tests for enterprise descriptive distribution analysis."""

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
from app.tools import DatasetContext, DistributionAnalysisTool, ToolRegistry
from app.tools.distribution_analysis_tool import (
    MAX_CATEGORICAL_COLUMNS,
    MAX_CHART_POINTS,
    MAX_GROUP_COUNT,
    MAX_NUMERIC_COLUMNS,
)


def _task(**params: object) -> AnalysisTask:
    return AnalysisTask(
        task_name="Distribution Analysis",
        reasoning="Describe distributions without modifying source data.",
        expected_output="JSON-safe descriptive statistics and chart data.",
        type="distribution_analysis",
        params=params,
    )


def _run(dataframe: pd.DataFrame, **params: object) -> ExecutionResult:
    return DistributionAnalysisTool().run(
        dataframe,
        _task(**params),
        DatasetContext(numeric_columns=[], categorical_columns=[], datetime_columns=[]),
    )


def _contains_unsupported_object(value: object) -> bool:
    if isinstance(value, (pd.DataFrame, pd.Series, pd.Index, np.ndarray, np.generic)):
        return True
    if isinstance(value, dict):
        return any(_contains_unsupported_object(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_unsupported_object(item) for item in value)
    return False


def test_numeric_basic_statistics_missing_and_counts_are_correct() -> None:
    result = _run(pd.DataFrame({"value": [1.0, 2.0, 3.0, None]}), mode="numeric")
    column = result.data["columns"][0]
    statistics = column["statistics"]

    assert column["count"] == 3
    assert column["missing_count"] == 1
    assert column["missing_ratio"] == pytest.approx(0.25)
    assert statistics["mean"] == pytest.approx(2.0)
    assert statistics["median"] == pytest.approx(2.0)
    assert statistics["std"] == pytest.approx(1.0)
    assert statistics["variance"] == pytest.approx(1.0)
    assert statistics["range"] == pytest.approx(2.0)


def test_default_and_custom_quantiles_are_correct_and_stably_normalized() -> None:
    dataframe = pd.DataFrame({"value": list(range(1, 11))})
    default = _run(dataframe, mode="numeric").data["columns"][0]["quantiles"]
    custom = _run(
        dataframe,
        mode="numeric",
        quantiles=[0.9, 0.5, 0.1, 0.5],
    ).data["columns"][0]["quantiles"]

    assert list(default) == ["0.01", "0.05", "0.25", "0.5", "0.75", "0.95", "0.99"]
    assert default["0.5"] == pytest.approx(5.5)
    assert list(custom) == ["0.1", "0.5", "0.9"]
    assert custom["0.1"] == pytest.approx(1.9)


@pytest.mark.parametrize("quantiles", [[], [-0.1], [1.1], [float("nan")], [True], ["0.5"]])
def test_invalid_quantiles_return_clear_error(quantiles: object) -> None:
    with pytest.raises(ToolExecutionException) as exc_info:
        _run(pd.DataFrame({"value": range(10)}), mode="numeric", quantiles=quantiles)

    assert exc_info.value.error_code == "INVALID_DISTRIBUTION_QUANTILES"


def test_quantile_count_limit_returns_clear_error() -> None:
    with pytest.raises(ToolExecutionException) as exc_info:
        _run(
            pd.DataFrame({"value": range(30)}),
            mode="numeric",
            quantiles=[index / 20 for index in range(21)],
        )

    assert exc_info.value.error_code == "DISTRIBUTION_QUANTILE_LIMIT_EXCEEDED"


def test_coefficient_of_variation_and_zero_mean_guard_are_correct() -> None:
    positive = _run(pd.DataFrame({"value": [1.0, 2.0, 3.0] * 4}), mode="numeric")
    centered = _run(pd.DataFrame({"value": [-1.0, 1.0] * 5}), mode="numeric")

    assert positive.data["columns"][0]["statistics"]["coefficient_of_variation"] == pytest.approx(
        pd.Series([1.0, 2.0, 3.0] * 4).std(ddof=1) / 2.0
    )
    assert centered.data["columns"][0]["statistics"]["coefficient_of_variation"] is None
    assert any("mean is zero" in item for item in centered.data["columns"][0]["warnings"])


def test_zero_negative_and_unique_statistics_are_correct() -> None:
    statistics = _run(
        pd.DataFrame({"value": [-2, -1, 0, 0, 1, 2, 3, 4, 5, 6]}),
        mode="numeric",
    ).data["columns"][0]["statistics"]

    assert statistics["zero_count"] == 2
    assert statistics["zero_ratio"] == pytest.approx(0.2)
    assert statistics["negative_count"] == 2
    assert statistics["negative_ratio"] == pytest.approx(0.2)
    assert statistics["unique_count"] == 9


def test_symmetric_positive_and_negative_skew_labels() -> None:
    symmetric = _run(pd.DataFrame({"value": list(range(-5, 6))}), mode="numeric")
    positive = _run(pd.DataFrame({"value": [1.0] * 10 + [100.0]}), mode="numeric")
    negative = _run(pd.DataFrame({"value": [-100.0] + [-1.0] * 10}), mode="numeric")

    assert symmetric.data["columns"][0]["shape"]["skew_level"] == "approximately_symmetric"
    assert symmetric.data["columns"][0]["shape"]["skew_direction"] == "none"
    assert positive.data["columns"][0]["shape"]["skew_direction"] == "positive"
    assert positive.data["columns"][0]["shape"]["skew_level"] == "highly_skewed"
    assert negative.data["columns"][0]["shape"]["skew_direction"] == "negative"


def test_small_sample_and_constant_suppress_misleading_shape_conclusions() -> None:
    small = _run(pd.DataFrame({"value": [1.0, 2.0, 50.0]}), mode="numeric")
    constant = _run(pd.DataFrame({"value": [5.0] * 10}), mode="numeric")

    assert small.data["columns"][0]["reliable"] is False
    assert small.data["columns"][0]["shape"]["skew_level"] == "unavailable"
    assert small.data["columns"][0]["shape"]["long_tail_candidate"] is False
    assert constant.data["columns"][0]["constant_column"] is True
    assert constant.data["columns"][0]["statistics"]["std"] == 0.0
    assert constant.data["columns"][0]["statistics"]["iqr"] == 0.0
    assert constant.data["columns"][0]["statistics"]["skewness"] is None
    assert constant.data["columns"][0]["statistics"]["kurtosis"] is None


def test_metadata_documents_descriptive_non_normality_boundary() -> None:
    metadata = _run(pd.DataFrame({"value": range(10)}), mode="numeric").data["metadata"]

    assert metadata["descriptive_analysis_only"] is True
    assert metadata["not_a_formal_normality_test"] is True
    assert metadata["no_causal_conclusions"] is True
    assert "Fisher excess" in metadata["kurtosis_definition"]


def test_histogram_counts_custom_bins_and_constant_column_are_safe() -> None:
    dataframe = pd.DataFrame({"value": list(range(10))})
    histogram = _run(dataframe, mode="numeric", bins=5).data["columns"][0]["histogram"]
    constant = _run(pd.DataFrame({"value": [2.0] * 10}), mode="numeric", bins=7)
    constant_histogram = constant.data["columns"][0]["histogram"]

    assert histogram["bin_count"] == 5
    assert sum(histogram["counts"]) == 10
    assert len(histogram["bin_edges"]) == 6
    assert sum(constant_histogram["counts"]) == 10
    assert len(constant_histogram["bin_edges"]) == 8


@pytest.mark.parametrize("bins", [4, 101, 10.0, True, "20"])
def test_invalid_histogram_bins_return_clear_error(bins: object) -> None:
    with pytest.raises(ToolExecutionException) as exc_info:
        _run(pd.DataFrame({"value": range(10)}), mode="numeric", bins=bins)

    assert exc_info.value.error_code == "INVALID_DISTRIBUTION_BINS"


def test_boxplot_quartiles_whiskers_and_outlier_summary_are_correct() -> None:
    boxplot = _run(
        pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 100.0]}),
        mode="numeric",
    ).data["columns"][0]["boxplot"]

    assert boxplot["q1"] == pytest.approx(2.0)
    assert boxplot["median"] == pytest.approx(3.0)
    assert boxplot["q3"] == pytest.approx(4.0)
    assert boxplot["iqr"] == pytest.approx(2.0)
    assert boxplot["lower_whisker"] == pytest.approx(1.0)
    assert boxplot["upper_whisker"] == pytest.approx(4.0)
    assert boxplot["outlier_count"] == 1
    assert "outliers" not in boxplot


def test_numeric_type_exclusions_and_infinity_handling_are_explicit() -> None:
    dataframe = pd.DataFrame(
        {
            "value": [1.0, 2.0, np.inf, -np.inf, 5.0],
            "flag": [True, False, True, False, True],
            "date": pd.date_range("2026-01-01", periods=5),
            "delta": pd.to_timedelta(range(5), unit="D"),
            "text": ["1", "2", "3", "4", "5"],
            "category": pd.Series(["a", "b", "a", "b", "a"], dtype="category"),
            "complex": np.array([1 + 1j] * 5),
        }
    )
    result = _run(dataframe, mode="numeric")
    excluded = {item["column"]: item["reason"] for item in result.data["excluded_columns"]}

    assert [item["column"] for item in result.data["columns"]] == ["value"]
    assert result.data["columns"][0]["count"] == 3
    assert excluded == {
        "flag": "boolean",
        "date": "datetime",
        "delta": "timedelta",
        "text": "text",
        "category": "category",
        "complex": "complex",
    }
    assert any("infinite values" in warning for warning in result.data["warnings"])


def test_all_null_is_excluded_and_explicit_invalid_columns_raise() -> None:
    dataframe = pd.DataFrame({"value": range(10), "empty": [None] * 10, "text": ["a"] * 10})
    result = _run(dataframe, mode="numeric")
    excluded = {item["column"]: item["reason"] for item in result.data["excluded_columns"]}
    assert excluded["empty"] == "all_null_or_non_finite"

    with pytest.raises(ToolExecutionException) as unknown:
        _run(dataframe, mode="numeric", columns=["missing"])
    with pytest.raises(ToolExecutionException) as non_numeric:
        _run(dataframe, mode="numeric", columns=["text"])
    assert unknown.value.error_code == "UNKNOWN_DISTRIBUTION_COLUMNS"
    assert non_numeric.value.error_code == "NON_NUMERIC_DISTRIBUTION_COLUMNS"


def test_original_dataframe_is_not_modified() -> None:
    dataframe = pd.DataFrame({"value": [1.0, np.inf, None, 4.0], "region": ["a", "a", "b", "b"]})
    original = dataframe.copy(deep=True)

    _run(dataframe, mode="numeric", columns=["value"])
    _run(dataframe, mode="categorical", columns=["region"])
    _run(dataframe, mode="group_comparison", group_by="region", columns=["value"])

    pdt.assert_frame_equal(dataframe, original)


def test_singleton_pair_and_reliable_sample_rules() -> None:
    singleton = _run(pd.DataFrame({"value": [5.0]}), mode="numeric").data["columns"][0]
    pair = _run(pd.DataFrame({"value": [1.0, 3.0]}), mode="numeric").data["columns"][0]
    small = _run(pd.DataFrame({"value": range(5)}), mode="numeric").data["columns"][0]
    reliable = _run(pd.DataFrame({"value": range(10)}), mode="numeric").data["columns"][0]

    assert singleton["statistics"]["median"] == 5.0
    assert singleton["statistics"]["std"] is None
    assert pair["statistics"]["std"] == pytest.approx(np.sqrt(2.0))
    assert pair["statistics"]["skewness"] is None
    assert small["reliable"] is False
    assert reliable["reliable"] is True


def test_numeric_column_limit_is_stable_and_explicit_overflow_errors() -> None:
    dataframe = pd.DataFrame({f"c{index}": range(12) for index in range(MAX_NUMERIC_COLUMNS + 1)})
    result = _run(dataframe, mode="numeric")

    assert len(result.data["columns"]) == MAX_NUMERIC_COLUMNS
    assert [item["column"] for item in result.data["columns"]] == [
        f"c{index}" for index in range(MAX_NUMERIC_COLUMNS)
    ]
    assert result.data["metadata"]["original_candidate_column_count"] == MAX_NUMERIC_COLUMNS + 1
    with pytest.raises(ToolExecutionException) as exc_info:
        _run(dataframe, mode="numeric", columns=list(dataframe.columns))
    assert exc_info.value.error_code == "DISTRIBUTION_COLUMN_LIMIT_EXCEEDED"


def test_numeric_shape_heuristics_are_cautious_and_named() -> None:
    values = [0.0] * 10 + [1.0] * 9 + [100.0]
    shape = _run(pd.DataFrame({"value": values}), mode="numeric").data["columns"][0]["shape"]

    assert shape["zero_inflated_candidate"] is True
    assert shape["long_tail_candidate"] is True
    assert isinstance(shape["wide_dispersion"], bool)


def test_categorical_frequency_mode_top_k_and_other_are_correct() -> None:
    result = _run(
        pd.DataFrame({"region": ["East", "West", "East", "North", "East", None]}),
        mode="categorical",
        top_k=2,
    )
    column = result.data["columns"][0]

    assert column["count"] == 5
    assert column["missing_count"] == 1
    assert column["mode"] == "East"
    assert column["mode_count"] == 3
    assert column["mode_ratio"] == pytest.approx(0.6)
    assert [(item["value"], item["count"]) for item in column["top_categories"]] == [
        ("East", 3), ("West", 1)
    ]
    assert column["other_count"] == 1
    assert column["other_ratio"] == pytest.approx(0.2)


def test_categorical_tie_order_is_first_appearance_stable() -> None:
    column = _run(
        pd.DataFrame({"category": ["b", "a", "c", "a", "b", "c"]}),
        mode="categorical",
    ).data["columns"][0]

    assert column["mode"] == "b"
    assert [item["value"] for item in column["top_categories"]] == ["b", "a", "c"]


@pytest.mark.parametrize("top_k", [0, 51, 3.0, True, "10"])
def test_invalid_top_k_returns_clear_error(top_k: object) -> None:
    with pytest.raises(ToolExecutionException) as exc_info:
        _run(pd.DataFrame({"category": ["a", "b"]}), mode="categorical", top_k=top_k)

    assert exc_info.value.error_code == "INVALID_DISTRIBUTION_TOP_K"


def test_rare_categories_and_high_cardinality_are_descriptive() -> None:
    rare = _run(
        pd.DataFrame({"category": ["common"] * 199 + ["rare"]}),
        mode="categorical",
    ).data["columns"][0]
    high = _run(
        pd.DataFrame({"category": [f"id-{index}" for index in range(120)]}),
        mode="categorical",
    ).data["columns"][0]

    assert rare["rare_category_count"] == 1
    assert rare["rare_value_count"] == 1
    assert rare["rare_value_ratio"] == pytest.approx(0.005)
    assert high["high_cardinality"] is True
    assert any("not an error" in warning for warning in high["warnings"])


def test_categorical_constant_boolean_entropy_and_concentration() -> None:
    equal = _run(
        pd.DataFrame({"flag": [True, False] * 5}),
        mode="categorical",
    ).data["columns"][0]
    constant = _run(
        pd.DataFrame({"category": ["only"] * 10}),
        mode="categorical",
    ).data["columns"][0]

    assert equal["inferred_type"] == "boolean"
    assert equal["entropy"] == pytest.approx(1.0)
    assert 0 <= equal["normalized_entropy"] <= 1
    assert equal["concentration_ratio_top1"] == pytest.approx(0.5)
    assert equal["concentration_ratio_top3"] == pytest.approx(1.0)
    assert constant["constant_column"] is True
    assert constant["normalized_entropy"] == 0.0
    assert constant["highly_concentrated"] is True


def test_long_and_special_category_values_are_json_safe() -> None:
    long_value = "x" * 500
    result = _run(
        pd.DataFrame({"category": [long_value, pd.Timestamp("2026-01-01"), long_value] + ["a"] * 7}),
        mode="categorical",
    )
    top = result.data["columns"][0]["top_categories"]

    long_item = next(item for item in top if isinstance(item["value"], str) and item["value"].startswith("x"))
    assert len(long_item["value"]) == 200
    assert long_item["value_truncated"] is True
    json.dumps(result.model_dump(), allow_nan=False)


def test_continuous_numeric_is_not_categorical_by_default_but_low_cardinality_opt_in_works() -> None:
    dataframe = pd.DataFrame({"status_code": [1, 2, 1, 2, 1, 2], "label": ["a"] * 6})
    default = _run(dataframe, mode="categorical")
    opted_in = _run(
        dataframe,
        mode="categorical",
        columns=["status_code"],
        allow_low_cardinality_numeric=True,
    )

    assert [item["column"] for item in default.data["columns"]] == ["label"]
    assert opted_in.data["columns"][0]["column"] == "status_code"
    with pytest.raises(ToolExecutionException):
        _run(dataframe, mode="categorical", columns=["status_code"])


def test_categorical_column_limit_is_stable() -> None:
    dataframe = pd.DataFrame(
        {f"c{index}": [f"v{index}", "shared"] * 5 for index in range(MAX_CATEGORICAL_COLUMNS + 1)}
    )
    result = _run(dataframe, mode="categorical")

    assert len(result.data["columns"]) == MAX_CATEGORICAL_COLUMNS
    assert [item["column"] for item in result.data["columns"]] == [
        f"c{index}" for index in range(MAX_CATEGORICAL_COLUMNS)
    ]
    assert any("Categorical column count exceeded" in warning for warning in result.data["warnings"])
    with pytest.raises(ToolExecutionException) as exc_info:
        _run(dataframe, mode="categorical", columns=list(dataframe.columns))
    assert exc_info.value.error_code == "DISTRIBUTION_COLUMN_LIMIT_EXCEEDED"


def test_zero_sample_inputs_fail_with_clear_typed_errors() -> None:
    with pytest.raises(ToolExecutionException) as empty:
        _run(pd.DataFrame({"value": pd.Series(dtype="float64")}), mode="numeric")
    with pytest.raises(ToolExecutionException) as all_missing:
        _run(pd.DataFrame({"value": [np.nan, np.nan]}), mode="numeric")

    assert empty.value.error_code == "EMPTY_DISTRIBUTION_DATASET"
    assert all_missing.value.error_code == "NO_NUMERIC_DISTRIBUTION_COLUMNS"


def test_default_mode_prefers_numeric_then_categorical() -> None:
    numeric = _run(pd.DataFrame({"value": range(10), "region": ["a"] * 10}))
    categorical = _run(pd.DataFrame({"region": ["a", "b"] * 5}))
    non_finite_numeric = _run(
        pd.DataFrame({"value": [np.inf, -np.inf], "region": ["a", "b"]})
    )

    assert numeric.data["mode"] == "numeric"
    assert categorical.data["mode"] == "categorical"
    assert non_finite_numeric.data["mode"] == "categorical"


@pytest.mark.parametrize("mode", ["unknown", "", 1, True])
def test_unknown_modes_return_clear_error(mode: object) -> None:
    with pytest.raises(ToolExecutionException) as exc_info:
        _run(pd.DataFrame({"value": range(10)}), mode=mode)

    assert exc_info.value.error_code == "INVALID_DISTRIBUTION_MODE"


def test_group_comparison_statistics_and_extreme_groups_are_correct() -> None:
    dataframe = pd.DataFrame(
        {
            "region": ["East"] * 10 + ["West"] * 10,
            "sales": list(range(1, 11)) + list(range(101, 111)),
        }
    )
    result = _run(dataframe, mode="group_comparison", group_by="region", columns=["sales"])
    column = result.data["columns"][0]
    east, west = column["group_summary"]

    assert east["group"] == "East" and east["sample_size"] == 10
    assert east["mean"] == pytest.approx(5.5)
    assert east["median"] == pytest.approx(5.5)
    assert east["iqr"] == pytest.approx(4.5)
    assert west["mean"] == pytest.approx(105.5)
    assert column["highest_mean_group"] == "West"
    assert column["lowest_mean_group"] == "East"
    assert column["highest_median_group"] == "West"
    assert column["widest_iqr_group"] == "East"


def test_group_comparison_small_and_missing_groups_are_safe() -> None:
    dataframe = pd.DataFrame(
        {
            "region": [None] * 3 + ["West"] * 10,
            "sales": [1, 2, 100] + list(range(10)),
        }
    )
    result = _run(dataframe, mode="group_comparison", group_by="region", columns=["sales"])
    groups = result.data["columns"][0]["group_summary"]
    missing = next(group for group in groups if group["group"] == "__MISSING__")

    assert missing["reliable"] is False
    assert missing["skewness"] is None
    assert any("complex statistics are unreliable" in warning for warning in result.data["warnings"])
    assert result.data["metadata"]["missing_group_label"] == "__MISSING__"


def test_group_comparison_validates_group_and_numeric_columns() -> None:
    dataframe = pd.DataFrame({"region": ["a"] * 10, "sales": range(10)})
    with pytest.raises(ToolExecutionException) as missing_group:
        _run(dataframe, mode="group_comparison", columns=["sales"])
    with pytest.raises(ToolExecutionException) as unknown_group:
        _run(dataframe, mode="group_comparison", group_by="missing", columns=["sales"])
    with pytest.raises(ToolExecutionException) as unknown_value:
        _run(dataframe, mode="group_comparison", group_by="region", columns=["missing"])

    assert missing_group.value.error_code == "MISSING_DISTRIBUTION_GROUP_BY"
    assert unknown_group.value.error_code == "UNKNOWN_DISTRIBUTION_GROUP_BY"
    assert unknown_value.value.error_code == "UNKNOWN_DISTRIBUTION_COLUMNS"


def test_group_count_limit_is_stable_and_metadata_is_complete() -> None:
    dataframe = pd.DataFrame(
        {
            "group": [f"g{index}" for index in range(MAX_GROUP_COUNT + 1) for _ in range(5)],
            "value": [float(index) for index in range(MAX_GROUP_COUNT + 1) for _ in range(5)],
        }
    )
    result = _run(dataframe, mode="group_comparison", group_by="group", columns=["value"])

    groups = result.data["columns"][0]["group_summary"]
    assert len(groups) == MAX_GROUP_COUNT
    assert [group["group"] for group in groups[:3]] == ["g0", "g1", "g2"]
    assert result.data["metadata"]["original_group_count"] == MAX_GROUP_COUNT + 1
    assert result.data["metadata"]["actual_group_count"] == MAX_GROUP_COUNT
    assert any("Group count exceeded" in warning for warning in result.data["warnings"])


def test_group_comparison_is_descriptive_without_significance_claims() -> None:
    result = _run(
        pd.DataFrame({"group": ["a"] * 10 + ["b"] * 10, "value": range(20)}),
        mode="group_comparison",
        group_by="group",
        columns=["value"],
    )
    serialized = json.dumps(result.data).lower()

    assert result.data["metadata"]["descriptive_analysis_only"] is True
    assert "statistically significant" not in serialized
    assert "p-value" not in serialized


def test_all_three_chart_payloads_and_selection_rules_are_bounded() -> None:
    numeric = _run(
        pd.DataFrame({"symmetric": range(20), "skewed": [1] * 19 + [100]}),
        mode="numeric",
    )
    categorical = _run(
        pd.DataFrame({"id": [f"id-{index}" for index in range(120)], "region": ["a", "b"] * 60}),
        mode="categorical",
    )
    grouped = _run(
        pd.DataFrame({"group": ["a"] * 10 + ["b"] * 10, "x": range(20), "y": range(20, 40)}),
        mode="group_comparison",
        group_by="group",
        columns=["y", "x"],
    )

    assert numeric.data["chart"]["type"] == "histogram"
    assert numeric.data["chart"]["column"] == "skewed"
    assert categorical.data["chart"]["type"] == "category_bar"
    assert categorical.data["chart"]["column"] == "region"
    assert grouped.data["chart"]["type"] == "group_boxplot_summary"
    assert grouped.data["chart"]["column"] == "y"
    assert grouped.data["metadata"]["chart_selection_rule"] == "first_explicit_column"
    assert len(grouped.data["chart"]["groups"]) <= MAX_CHART_POINTS
    assert numeric.chart is categorical.chart is grouped.chart is None


def test_strict_json_serialization_and_no_pandas_numpy_objects() -> None:
    dataframe = pd.DataFrame(
        {
            "value": [1.0, np.nan, np.inf, -np.inf, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "category": [pd.NA, "a", "b", "a", "b", "a", "b", "a", "b", "a"],
        }
    )
    numeric = _run(dataframe, mode="numeric", columns=["value"])
    categorical = _run(dataframe, mode="categorical", columns=["category"])

    json.dumps(numeric.model_dump(), allow_nan=False)
    json.dumps(categorical.model_dump(), allow_nan=False)
    assert not _contains_unsupported_object(numeric.model_dump())
    assert not _contains_unsupported_object(categorical.model_dump())


def test_registry_execution_keyword_and_existing_mappings_are_preserved() -> None:
    registry = ToolRegistry.with_default_tools()
    service = PandasExecutionService(registry)

    assert isinstance(registry.get("distribution_analysis_tool"), DistributionAnalysisTool)
    assert service._resolve_tool_name(_task(mode="numeric")) == "distribution_analysis_tool"
    keyword_task = AnalysisTask(
        task_name="Histogram distribution",
        reasoning="Analyze skewness and quantiles.",
        expected_output="Histogram data.",
        params={},
    )
    assert service._resolve_tool_name(keyword_task) == "distribution_analysis_tool"
    plain_keyword_task = AnalysisTask(
        task_name="Sales distribution",
        reasoning="Describe sales values.",
        expected_output="Distribution summary.",
        params={},
    )
    assert service._resolve_tool_name(plain_keyword_task) == "distribution_analysis_tool"
    assert service.task_type_tools["anomaly_detection"] == "anomaly_detection_tool"
    assert service.task_type_tools["correlation_analysis"] == "correlation_analysis_tool"
    assert service.task_type_tools["data_quality_analysis"] == "data_quality_tool"


def test_multiple_execution_tasks_remain_dataframe_isolated() -> None:
    dataframe = pd.DataFrame({"value": range(10), "region": ["a", "b"] * 5})
    original = dataframe.copy(deep=True)
    service = PandasExecutionService()

    response = service.execute_tasks(
        dataframe,
        [
            _task(mode="numeric", columns=["value"]),
            _task(mode="categorical", columns=["region"]),
        ],
    )

    assert [result.type for result in response.execution_results] == [
        "distribution_analysis", "distribution_analysis"
    ]
    pdt.assert_frame_equal(dataframe, original)
