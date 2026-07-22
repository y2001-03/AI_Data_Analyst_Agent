"""Planner and workflow tests for anomaly detection intent."""

from __future__ import annotations

from app.graph.dataset_workflow import DatasetGraphNodes
from app.schemas.file import AnalysisTask, DatasetColumnProfile, DatasetUploadResponse
from app.services.analysis_planner_service import AnalysisPlannerService


def _file_info() -> DatasetUploadResponse:
    return DatasetUploadResponse(
        file_name="sales.csv",
        file_type="csv",
        row_count=100,
        column_count=5,
        preview=[],
        columns=[
            DatasetColumnProfile(name="sales", data_type="float64", missing_count=0, unique_values=90),
            DatasetColumnProfile(name="price", data_type="float64", missing_count=1, unique_values=80),
            DatasetColumnProfile(name="region", data_type="object", missing_count=0, unique_values=4),
            DatasetColumnProfile(name="order_date", data_type="datetime64[ns]", missing_count=0, unique_values=100),
            DatasetColumnProfile(name="active", data_type="bool", missing_count=0, unique_values=2),
        ],
    )


def test_mock_planner_recognizes_iqr_global_anomaly_detection() -> None:
    task = AnalysisPlannerService()._mock_tasks(
        _file_info(),
        "使用 IQR 检查 sales 异常值",
    )[0]

    assert task.type == "anomaly_detection"
    assert task.params["mode"] == "global"
    assert task.params["method"] == "iqr"
    assert task.params["columns"] == ["sales"]


def test_mock_planner_recognizes_zscore_and_robust_zscore() -> None:
    planner = AnalysisPlannerService()

    zscore = planner._mock_tasks(_file_info(), "用 Z-score 检测 price 异常点")[0]
    robust = planner._mock_tasks(_file_info(), "使用 robust z-score 查找 sales outlier")[0]

    assert zscore.params["method"] == "zscore"
    assert zscore.params["columns"] == ["price"]
    assert robust.params["method"] == "robust_zscore"
    assert robust.params["columns"] == ["sales"]


def test_mock_planner_parses_groupwise_mode_group_and_value_column() -> None:
    task = AnalysisPlannerService()._mock_tasks(
        _file_info(),
        "按地区检查 sales 异常",
    )[0]

    assert task.type == "anomaly_detection"
    assert task.params["mode"] == "groupwise"
    assert task.params["group_by"] == "region"
    assert task.params["columns"] == ["sales"]


def test_mock_planner_parses_time_series_columns_window_and_threshold() -> None:
    task = AnalysisPlannerService()._mock_tasks(
        _file_info(),
        "使用滚动窗口 window 14 阈值 4 检测 order_date 上 sales 的突增突降",
    )[0]

    assert task.type == "anomaly_detection"
    assert task.params["mode"] == "time_series"
    assert task.params["method"] == "rolling_zscore"
    assert task.params["time_column"] == "order_date"
    assert task.params["value_columns"] == ["sales"]
    assert task.params["window"] == 14
    assert task.params["threshold"] == 4.0


def test_mock_planner_keeps_overall_quality_request_as_data_quality() -> None:
    task = AnalysisPlannerService()._mock_tasks(
        _file_info(),
        "检查数据质量中的异常值概览",
    )[0]

    assert task.type == "data_quality_analysis"


def test_plain_error_description_is_not_misrouted_to_anomaly_detection() -> None:
    task = AnalysisPlannerService()._mock_tasks(
        _file_info(),
        "系统发生异常错误怎么办？",
    )[0]

    assert task.type != "anomaly_detection"


def test_delete_anomaly_request_routes_to_detection_only() -> None:
    task = AnalysisPlannerService()._mock_tasks(
        _file_info(),
        "删除 sales 异常值",
    )[0]

    assert task.type == "anomaly_detection"
    assert task.params["detection_only"] is True
    assert task.params["requested_deletion"] is True
    assert "no automatic deletion" in task.expected_output


def test_normalized_llm_anomaly_task_gets_mode_specific_defaults() -> None:
    planner = AnalysisPlannerService()
    global_task = planner._normalize_task(
        {
            "task_name": "Global anomalies",
            "type": "anomaly_detection",
            "reason": "Find candidate anomaly points.",
            "params": {},
        },
        _file_info(),
    )
    time_task = planner._normalize_task(
        {
            "task_name": "Time anomalies",
            "type": "anomaly_detection",
            "reason": "Find time-series spikes.",
            "params": {"mode": "time_series"},
        },
        _file_info(),
    )

    assert global_task is not None and global_task.params["method"] == "iqr"
    assert time_task is not None and time_task.params["method"] == "rolling_zscore"
    assert global_task.params["detection_only"] is True


def test_planner_prompt_documents_anomaly_safety_and_supported_modes() -> None:
    prompt = AnalysisPlannerService()._system_prompt()

    assert "anomaly_detection" in prompt
    assert "global, groupwise, and time_series" in prompt
    assert "rolling_zscore" in prompt
    assert "without modifying or deleting data" in prompt
    assert "do not add a separate chart task" in prompt


def test_existing_correlation_trend_and_cleaning_routes_are_unchanged() -> None:
    planner = AnalysisPlannerService()

    correlation = planner._mock_tasks(_file_info(), "分析 sales 和 price 的相关性")[0]
    trend = planner._mock_tasks(_file_info(), "查看 sales 按日期的趋势")[0]
    cleaning = planner._mock_tasks(_file_info(), "用中位数填充 sales 缺失值")[0]

    assert correlation.type == "correlation_analysis"
    assert trend.type == "trend"
    assert cleaning.type == "data_cleaning_execute"


def test_workflow_does_not_append_visualization_to_anomaly_task() -> None:
    nodes = object.__new__(DatasetGraphNodes)
    task = AnalysisTask(
        task_name="Anomaly Detection",
        reasoning="Find candidate anomalies.",
        expected_output="Anomaly records and chart data.",
        type="anomaly_detection",
        params={"mode": "global", "method": "iqr"},
    )

    tasks = nodes._ensure_visualization_task(_file_info(), [task])

    assert tasks == [task]


def test_workflow_existing_stats_visualization_behavior_is_unchanged() -> None:
    nodes = object.__new__(DatasetGraphNodes)
    task = AnalysisTask(
        task_name="Statistics",
        reasoning="Summarize metrics.",
        expected_output="Summary statistics.",
        type="stats",
        params={},
    )

    tasks = nodes._ensure_visualization_task(_file_info(), [task])

    assert len(tasks) == 2
    assert tasks[0] == task
