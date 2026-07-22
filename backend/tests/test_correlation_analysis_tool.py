"""Tests for enterprise numeric correlation analysis."""

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
from app.tools import CorrelationAnalysisTool, DatasetContext, ToolRegistry
from app.tools.correlation_analysis_tool import MAX_CORRELATION_COLUMNS


def _task(**params: object) -> AnalysisTask:
    return AnalysisTask(
        task_name="Correlation Analysis",
        reasoning="Analyze statistical associations without causal claims.",
        expected_output="Correlation matrix and pair details.",
        type="correlation_analysis",
        params=params,
    )


def _run(dataframe: pd.DataFrame, **params: object) -> ExecutionResult:
    return CorrelationAnalysisTool().run(
        dataframe,
        _task(**params),
        DatasetContext(numeric_columns=[], categorical_columns=[], datetime_columns=[]),
    )


def _pair(result: ExecutionResult, column_x: str, column_y: str) -> dict[str, Any]:
    return next(
        pair
        for pair in result.data["pairs"]
        if pair["column_x"] == column_x and pair["column_y"] == column_y
    )


def _contains_pandas_object(value: object) -> bool:
    if isinstance(value, (pd.DataFrame, pd.Series, np.generic)):
        return True
    if isinstance(value, dict):
        return any(_contains_pandas_object(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_pandas_object(item) for item in value)
    return False


def test_pearson_detects_reliable_positive_and_negative_relationships() -> None:
    dataframe = pd.DataFrame(
        {
            "sales": np.arange(1, 11, dtype=float),
            "price": np.arange(2, 22, 2, dtype=float),
            "discount": np.arange(10, 0, -1, dtype=float),
        }
    )

    result = _run(dataframe)
    positive = _pair(result, "sales", "price")
    negative = _pair(result, "sales", "discount")

    assert result.type == "correlation_analysis"
    assert positive["coefficient"] == 1.0
    assert positive["direction"] == "positive"
    assert positive["strength"] == "strong"
    assert positive["sufficient_sample"] is True
    assert negative["coefficient"] == -1.0
    assert negative["direction"] == "negative"
    assert result.data["summary"]["strong_positive_count"] >= 1
    assert result.data["summary"]["strong_negative_count"] >= 1


def test_spearman_detects_monotonic_non_linear_relationship() -> None:
    x = np.arange(1, 11, dtype=float)
    result = _run(pd.DataFrame({"x": x, "squared": x**2}), method="spearman")

    assert result.data["summary"]["method"] == "spearman"
    assert _pair(result, "x", "squared")["coefficient"] == 1.0


def test_matrix_is_symmetric_with_unit_diagonal_and_unique_pairs() -> None:
    dataframe = pd.DataFrame(
        {
            "a": np.arange(10),
            "b": np.arange(10) * 2,
            "c": [3, 1, 4, 2, 7, 5, 9, 6, 10, 8],
        }
    )

    data = _run(dataframe).data

    assert len(data["pairs"]) == 3
    assert data["matrix"]["a"]["b"] == data["matrix"]["b"]["a"]
    assert all(data["matrix"][column][column] == 1.0 for column in data["columns"])
    unordered_pairs = {
        frozenset((pair["column_x"], pair["column_y"])) for pair in data["pairs"]
    }
    assert len(unordered_pairs) == len(data["pairs"])


@pytest.mark.parametrize(
    ("coefficient", "direction", "strength"),
    [
        (0.7, "positive", "strong"),
        (-0.7, "negative", "strong"),
        (0.4, "positive", "moderate"),
        (0.2, "positive", "weak"),
        (0.0, "none", "negligible"),
        (None, "none", "unavailable"),
    ],
)
def test_strength_and_direction_boundaries(
    coefficient: float | None,
    direction: str,
    strength: str,
) -> None:
    tool = CorrelationAnalysisTool()

    assert tool._direction(coefficient) == direction
    assert tool._strength(coefficient) == strength


def test_default_field_selection_excludes_unsupported_types() -> None:
    dataframe = pd.DataFrame(
        {
            "sales": np.arange(10, dtype=float),
            "price": np.arange(10, dtype=float) * 2,
            "active": [True, False] * 5,
            "ordered_at": pd.date_range("2026-01-01", periods=10),
            "elapsed": pd.to_timedelta(np.arange(10), unit="D"),
            "region": ["East", "West"] * 5,
            "category": pd.Series(["a", "b"] * 5, dtype="category"),
        }
    )

    data = _run(dataframe).data
    excluded = {item["column"]: item["reason"] for item in data["excluded_columns"]}

    assert data["columns"] == ["sales", "price"]
    assert excluded["active"] == "boolean"
    assert excluded["ordered_at"] == "datetime"
    assert excluded["elapsed"] == "timedelta"
    assert excluded["region"] in {"object", "string"}
    assert excluded["category"] == "category"


def test_constant_and_all_null_columns_are_excluded_without_nan_leakage() -> None:
    dataframe = pd.DataFrame(
        {
            "a": np.arange(10, dtype=float),
            "b": np.arange(10, dtype=float) * 3,
            "constant": np.ones(10),
            "empty": np.full(10, np.nan),
        }
    )

    result = _run(dataframe)

    assert result.data["constant_columns"] == [
        {"column": "constant", "reason": "constant"}
    ]
    assert {item["column"] for item in result.data["excluded_columns"]} == {"empty"}
    assert "NaN" not in json.dumps(result.model_dump(), allow_nan=False)


def test_explicit_numeric_columns_preserve_requested_order() -> None:
    dataframe = pd.DataFrame(
        {
            "sales": np.arange(10),
            "price": np.arange(10) * 2,
            "quantity": np.arange(10) * 3,
        }
    )

    result = _run(dataframe, columns=["quantity", "sales"])

    assert result.data["columns"] == ["quantity", "sales"]
    assert result.data["chart"]["x"] == ["quantity", "sales"]


def test_explicit_unknown_or_non_numeric_columns_raise_clear_errors() -> None:
    dataframe = pd.DataFrame(
        {"sales": np.arange(10), "price": np.arange(10) * 2, "region": ["x"] * 10}
    )

    with pytest.raises(ToolExecutionException, match="Unknown correlation column") as unknown:
        _run(dataframe, columns=["sales", "revenue"])
    with pytest.raises(ToolExecutionException, match="must be numeric") as non_numeric:
        _run(dataframe, columns=["sales", "region"])

    assert unknown.value.error_code == "UNKNOWN_CORRELATION_COLUMNS"
    assert non_numeric.value.error_code == "NON_NUMERIC_CORRELATION_COLUMNS"


def test_pairwise_sample_sizes_are_calculated_per_pair() -> None:
    dataframe = pd.DataFrame(
        {
            "a": [1.0, 2.0, np.nan, 4.0],
            "b": [1.0, np.nan, 3.0, 4.0],
            "c": [1.0, 2.0, 3.0, 4.0],
        }
    )

    data = _run(dataframe, missing_strategy="pairwise").data

    assert data["sample_size_matrix"]["a"]["b"] == 2
    assert data["sample_size_matrix"]["a"]["c"] == 3
    assert data["sample_size_matrix"]["b"]["c"] == 3
    assert data["summary"]["missing_strategy"] == "pairwise"


def test_listwise_uses_one_complete_case_sample_for_every_pair() -> None:
    dataframe = pd.DataFrame(
        {
            "a": [1.0, 2.0, np.nan, 4.0],
            "b": [1.0, np.nan, 3.0, 4.0],
            "c": [1.0, 2.0, 3.0, 4.0],
        }
    )

    data = _run(dataframe, missing_strategy="listwise").data

    assert data["summary"]["missing_strategy"] == "listwise"
    assert all(
        data["sample_size_matrix"][left][right] == 2
        for left in data["columns"]
        for right in data["columns"]
    )


def test_small_sample_correlation_is_returned_but_not_promoted_as_reliable() -> None:
    dataframe = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.0]})

    result = _run(dataframe)
    pair = _pair(result, "a", "b")

    assert pair["coefficient"] == 1.0
    assert pair["sufficient_sample"] is False
    assert result.data["strong_relationships"] == []
    assert result.data["multicollinearity_risks"] == []
    assert result.data["warnings"]


def test_reliable_high_correlation_creates_heuristic_multicollinearity_risk() -> None:
    dataframe = pd.DataFrame(
        {"price": np.arange(10, dtype=float), "unit_price": np.arange(10, dtype=float) * 1.01}
    )

    result = _run(dataframe)
    risk = result.data["multicollinearity_risks"][0]

    assert risk["risk_level"] == "high"
    assert result.data["metadata"]["multicollinearity_is_heuristic"] is True
    assert result.data["metadata"]["vif_calculated"] is False


def test_heatmap_and_entire_result_are_strictly_json_serializable() -> None:
    dataframe = pd.DataFrame(
        {
            "a": [1.0, 2.0, np.nan, 4.0],
            "b": [1.0, 2.0, 3.0, 4.0],
            "c": [np.inf, 2.0, 4.0, 8.0],
        }
    )

    result = _run(dataframe)
    chart = result.data["chart"]
    serialized = json.dumps(result.model_dump(), allow_nan=False)

    assert chart["type"] == "heatmap"
    assert chart["x"] == chart["y"] == result.data["columns"]
    assert len(chart["values"]) == len(chart["x"])
    assert all(len(row) == len(chart["x"]) for row in chart["values"])
    assert "NaN" not in serialized
    assert "Infinity" not in serialized
    assert not _contains_pandas_object(result.model_dump())


def test_pair_with_fewer_than_two_samples_uses_null_and_unavailable_labels() -> None:
    dataframe = pd.DataFrame(
        {
            "a": [1.0, np.nan, 3.0],
            "b": [np.nan, 2.0, 3.0],
        }
    )

    result = _run(dataframe)
    pair = _pair(result, "a", "b")

    assert pair["sample_size"] == 1
    assert pair["coefficient"] is None
    assert pair["absolute_coefficient"] is None
    assert pair["direction"] == "none"
    assert pair["strength"] == "unavailable"
    assert result.data["matrix"]["a"]["b"] is None
    assert result.data["chart"]["values"][0][1] is None


@pytest.mark.parametrize(
    ("dataframe", "params", "message"),
    [
        (pd.DataFrame({"a": []}), {}, "non-empty dataframe"),
        (pd.DataFrame({"region": ["a", "b"]}), {}, "no numeric columns"),
        (pd.DataFrame({"a": [1, 2], "region": ["x", "y"]}), {}, "at least two"),
        (pd.DataFrame({"a": [1, 1], "b": [2, 2]}), {}, "at least two"),
        (pd.DataFrame({"a": [1, 2], "b": [2, 4]}), {"method": "kendall"}, "pearson"),
        (
            pd.DataFrame({"a": [1, 2], "b": [2, 4]}),
            {"missing_strategy": "fill"},
            "pairwise",
        ),
    ],
)
def test_invalid_overall_inputs_raise_tool_execution_exception(
    dataframe: pd.DataFrame,
    params: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ToolExecutionException, match=message):
        _run(dataframe, **params)


def test_listwise_deletion_that_removes_every_row_raises_clear_error() -> None:
    dataframe = pd.DataFrame(
        {"a": [1.0, np.nan, 3.0, np.nan], "b": [np.nan, 2.0, np.nan, 4.0]}
    )

    with pytest.raises(ToolExecutionException, match="removed all rows") as exc_info:
        _run(dataframe, missing_strategy="listwise")

    assert exc_info.value.error_code == "NO_LISTWISE_ROWS"


def test_column_limit_selection_is_stable_and_reported() -> None:
    dataframe = pd.DataFrame(
        {
            f"column_{index}": np.arange(10, dtype=float) * (index + 1)
            for index in range(MAX_CORRELATION_COLUMNS + 2)
        }
    )

    first = _run(dataframe).data
    second = _run(dataframe).data

    assert len(first["columns"]) == MAX_CORRELATION_COLUMNS
    assert first["columns"] == second["columns"]
    assert first["metadata"]["original_eligible_column_count"] == MAX_CORRELATION_COLUMNS + 2
    assert first["metadata"]["actual_analyzed_column_count"] == MAX_CORRELATION_COLUMNS
    assert sum(item["reason"] == "column_limit" for item in first["excluded_columns"]) == 2
    assert first["warnings"]


def test_explicit_columns_over_limit_raise_instead_of_silently_ignoring_fields() -> None:
    columns = [f"column_{index}" for index in range(MAX_CORRELATION_COLUMNS + 1)]
    dataframe = pd.DataFrame({column: np.arange(10) for column in columns})

    with pytest.raises(ToolExecutionException) as exc_info:
        _run(dataframe, columns=columns)

    assert exc_info.value.error_code == "CORRELATION_COLUMN_LIMIT_EXCEEDED"


def test_analysis_never_modifies_original_dataframe() -> None:
    dataframe = pd.DataFrame(
        {"a": [1.0, np.nan, 3.0], "b": [np.inf, 2.0, 3.0], "constant": [1, 1, 1]}
    )
    original = dataframe.copy(deep=True)

    _run(dataframe)

    pdt.assert_frame_equal(dataframe, original)


def test_registry_and_execution_service_dispatch_correlation_analysis() -> None:
    registry = ToolRegistry.with_default_tools()
    dataframe = pd.DataFrame({"a": np.arange(10), "b": np.arange(10) * 2})

    assert isinstance(registry.get("correlation_analysis_tool"), CorrelationAnalysisTool)
    response = PandasExecutionService(registry).execute_tasks(dataframe, [_task()])
    assert response.execution_results[0].type == "correlation_analysis"
    assert response.execution_results[0].data["metadata"]["correlation_is_not_causation"] is True


def test_execution_keyword_fallback_and_existing_type_mappings_are_preserved() -> None:
    service = PandasExecutionService()
    keyword_task = AnalysisTask(
        task_name="Pearson correlation",
        reasoning="Check the correlation coefficient.",
        expected_output="Correlation matrix.",
        params={},
    )

    assert service._resolve_tool_name(keyword_task) == "correlation_analysis_tool"
    for task_type, tool_name in (
        ("data_quality_analysis", "data_quality_tool"),
        ("data_cleaning_plan", "data_cleaning_tool"),
        ("data_cleaning_execute", "data_cleaning_tool"),
        ("stats", "stats_tool"),
        ("groupby", "groupby_tool"),
        ("trend", "trend_tool"),
    ):
        task = AnalysisTask(
            task_name="Existing task",
            reasoning="Preserve existing routing.",
            expected_output="Existing output.",
            type=task_type,
            params={},
        )
        assert service._resolve_tool_name(task) == tool_name
