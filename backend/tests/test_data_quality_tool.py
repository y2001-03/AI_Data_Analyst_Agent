"""Tests for the enterprise data quality analysis tool."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from app.schemas.file import AnalysisTask
from app.services.pandas_execution_service import PandasExecutionService
from app.tools import DatasetContext, ToolRegistry
from app.tools.data_quality_tool import DataQualityTool


def _task(task_type: str = "data_quality_analysis") -> AnalysisTask:
    return AnalysisTask(
        task_name="Data Quality Check",
        reasoning="Check whether the dataset is reliable before analysis.",
        expected_output="Structured data quality report.",
        type=task_type,
        params={},
    )


def _run_tool(dataframe: pd.DataFrame):
    tool = DataQualityTool()
    return tool.run(
        dataframe,
        _task(),
        DatasetContext(numeric_columns=[], categorical_columns=[], datetime_columns=[]),
    )


def _column(report: dict[str, object], name: str) -> dict[str, object]:
    return next(column for column in report["columns"] if column["name"] == name)


def test_data_quality_tool_profiles_normal_mixed_types() -> None:
    dataframe = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4],
            "sales": [10.0, 20.0, 30.0, 40.0],
            "region": ["East", "West", "East", "North"],
            "active": [True, False, True, True],
            "ordered_at": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]),
        }
    )

    result = _run_tool(dataframe)
    report = result.data

    assert result.type == "data_quality_analysis"
    assert report["tool_name"] == "data_quality_analysis"
    assert report["summary"]["row_count"] == 4
    assert report["summary"]["column_count"] == 5
    assert _column(report, "sales")["inferred_type"] == "numeric"
    assert _column(report, "region")["inferred_type"] == "text"
    assert _column(report, "active")["inferred_type"] == "boolean"
    assert _column(report, "ordered_at")["inferred_type"] == "datetime"


def test_data_quality_tool_reports_missing_values() -> None:
    dataframe = pd.DataFrame({"customer_id": [1, 2, 3], "sales": [10.0, np.nan, np.nan]})

    report = _run_tool(dataframe).data
    sales = _column(report, "sales")

    assert sales["missing_count"] == 2
    assert sales["missing_ratio"] == 0.666667
    assert any(issue["issue_type"] == "missing_values" and issue["column"] == "sales" for issue in report["issues"])


def test_data_quality_tool_reports_duplicate_rows() -> None:
    dataframe = pd.DataFrame({"customer_id": [1, 1, 2], "sales": [10, 10, 20]})

    report = _run_tool(dataframe).data

    assert report["summary"]["duplicate_row_count"] == 1
    assert report["summary"]["duplicate_row_ratio"] == 0.333333
    assert any(issue["issue_type"] == "duplicate_rows" for issue in report["issues"])


def test_data_quality_tool_reports_iqr_outliers() -> None:
    dataframe = pd.DataFrame({"sales": [10, 11, 12, 13, 1000]})

    report = _run_tool(dataframe).data
    sales = _column(report, "sales")

    assert sales["outlier_count"] == 1
    assert sales["outlier_ratio"] == 0.2
    assert any(issue["issue_type"] == "numeric_outliers" and issue["column"] == "sales" for issue in report["issues"])


def test_data_quality_tool_reports_constant_columns() -> None:
    dataframe = pd.DataFrame({"status": ["paid", "paid", "paid"], "sales": [1, 2, 3]})

    report = _run_tool(dataframe).data
    status = _column(report, "status")

    assert status["constant_column"] is True
    assert report["summary"]["constant_column_count"] == 1
    assert any(issue["issue_type"] == "constant_column" and issue["column"] == "status" for issue in report["issues"])


def test_data_quality_tool_reports_possible_id_columns() -> None:
    dataframe = pd.DataFrame({"customer_id": [101, 102, 103, 104], "sales": [1, 2, 3, 4]})

    report = _run_tool(dataframe).data
    customer_id = _column(report, "customer_id")

    assert customer_id["possible_id_column"] is True
    assert any(
        issue["issue_type"] == "possible_id_column" and issue["column"] == "customer_id"
        for issue in report["issues"]
    )


def test_data_quality_tool_handles_empty_dataframe() -> None:
    dataframe = pd.DataFrame(columns=["customer_id", "sales"])

    report = _run_tool(dataframe).data

    assert report["summary"]["row_count"] == 0
    assert report["summary"]["column_count"] == 2
    assert report["summary"]["quality_score"] == 0.0
    assert any(issue["issue_type"] == "empty_dataset" for issue in report["issues"])


def test_data_quality_tool_output_is_json_serializable_without_nan() -> None:
    dataframe = pd.DataFrame(
        {
            "np_int": np.array([1, 2, 3], dtype=np.int64),
            "np_float": np.array([1.5, np.nan, 3.5], dtype=np.float64),
        }
    )

    result = _run_tool(dataframe)
    payload = result.model_dump()
    serialized = json.dumps(payload, allow_nan=False)

    assert "NaN" not in serialized


def test_data_quality_tool_handles_bool_column_without_iqr_or_id_detection() -> None:
    dataframe = pd.DataFrame({"active": [True, False, True, False]})

    result = _run_tool(dataframe)
    active = _column(result.data, "active")
    serialized = json.dumps(result.model_dump(), allow_nan=False)

    assert active["dtype"] == "bool"
    assert active["inferred_type"] == "boolean"
    assert active["missing_count"] == 0
    assert active["missing_ratio"] == 0.0
    assert active["unique_count"] == 2
    assert active["unique_ratio"] == 0.5
    assert active["constant_column"] is False
    assert active["possible_id_column"] is False
    assert active["outlier_count"] == 0
    assert active["outlier_ratio"] == 0.0
    assert "NaN" not in serialized


def test_tool_registry_finds_and_invokes_data_quality_tool() -> None:
    registry = ToolRegistry.with_default_tools()
    tool = registry.get("data_quality_tool")

    assert tool is not None
    result = tool.run(
        pd.DataFrame({"sales": [1, 2, 3]}),
        _task(),
        DatasetContext(numeric_columns=["sales"], categorical_columns=[], datetime_columns=[]),
    )

    assert result.type == "data_quality_analysis"
    assert result.data["summary"]["row_count"] == 3


def test_execution_service_dispatches_data_quality_type_and_preserves_existing_tools() -> None:
    dataframe = pd.DataFrame(
        {
            "region": ["East", "West", "East"],
            "sales": [10, 20, 30],
            "ordered_at": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
        }
    )
    tasks = [
        _task("data_quality_analysis"),
        AnalysisTask(
            task_name="Descriptive Stats",
            reasoning="Calculate summary statistics.",
            expected_output="Summary statistics using sales.",
            type="stats",
            params={},
        ),
        AnalysisTask(
            task_name="Grouped Sales",
            reasoning="Compare sales by region.",
            expected_output="Grouped comparison using region and sales.",
            type="groupby",
            params={},
        ),
        AnalysisTask(
            task_name="Sales Trend",
            reasoning="Analyze sales trend over ordered_at.",
            expected_output="Trend chart using ordered_at and sales.",
            type="trend",
            params={},
        ),
    ]

    response = PandasExecutionService().execute_tasks(dataframe, tasks)

    assert [result.type for result in response.execution_results] == [
        "data_quality_analysis",
        "stats",
        "groupby",
        "trend",
    ]


def test_execution_service_dispatches_quality_keywords() -> None:
    dataframe = pd.DataFrame({"sales": [1, 2, np.nan]})
    task = AnalysisTask(
        task_name="Check data quality",
        reasoning="Inspect missing values and duplicate rows before analysis.",
        expected_output="Report missing values and duplicate rows.",
        type=None,
        params={},
    )

    response = PandasExecutionService().execute_tasks(dataframe, [task])

    assert response.execution_results[0].type == "data_quality_analysis"
