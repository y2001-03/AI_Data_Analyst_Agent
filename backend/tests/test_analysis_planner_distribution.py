"""Planner and workflow tests for distribution analysis intent."""

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


def test_mock_planner_recognizes_numeric_distribution_and_parameters() -> None:
    task = AnalysisPlannerService()._mock_tasks(
        _file_info(),
        "分析 sales 分布，直方图分箱 12，分位数 0.1, 0.5, 0.9",
    )[0]

    assert task.type == "distribution_analysis"
    assert task.params["mode"] == "numeric"
    assert task.params["columns"] == ["sales"]
    assert task.params["bins"] == 12
    assert task.params["quantiles"] == [0.1, 0.5, 0.9]


def test_mock_planner_recognizes_categorical_distribution_and_top_k() -> None:
    task = AnalysisPlannerService()._mock_tasks(
        _file_info(),
        "分析 region 类别频数 top 5",
    )[0]

    assert task.type == "distribution_analysis"
    assert task.params["mode"] == "categorical"
    assert task.params["columns"] == ["region"]
    assert task.params["top_k"] == 5


def test_mock_planner_recognizes_group_comparison_and_group_alias() -> None:
    task = AnalysisPlannerService()._mock_tasks(
        _file_info(),
        "比较不同地区的 sales 分布差异",
    )[0]

    assert task.type == "distribution_analysis"
    assert task.params["mode"] == "group_comparison"
    assert task.params["group_by"] == "region"
    assert task.params["columns"] == ["sales"]


def test_region_count_distribution_is_categorical_not_group_comparison() -> None:
    task = AnalysisPlannerService()._mock_tasks(_file_info(), "各地区数量分布")[0]

    assert task.type == "distribution_analysis"
    assert task.params["mode"] == "categorical"
    assert task.params["columns"] == ["region"]


def test_normalized_distribution_task_gets_schema_aware_default_mode() -> None:
    planner = AnalysisPlannerService()
    numeric_task = planner._normalize_task(
        {
            "task_name": "Describe distribution",
            "type": "distribution_analysis",
            "reason": "Describe available fields.",
            "params": {},
        },
        _file_info(),
    )
    categorical_info = DatasetUploadResponse(
        file_name="categories.csv",
        file_type="csv",
        row_count=10,
        column_count=1,
        preview=[],
        columns=[DatasetColumnProfile(name="region", data_type="object", missing_count=0, unique_values=2)],
    )
    categorical_task = planner._normalize_task(
        {
            "task_name": "Describe categories",
            "type": "distribution_analysis",
            "reason": "Describe category frequencies.",
            "params": {},
        },
        categorical_info,
    )

    assert numeric_task is not None and numeric_task.params["mode"] == "numeric"
    assert categorical_task is not None and categorical_task.params["mode"] == "categorical"
    assert "no formal normality test" in numeric_task.expected_output


def test_planner_prompt_documents_distribution_boundaries_and_modes() -> None:
    prompt = AnalysisPlannerService()._system_prompt()

    assert "distribution_analysis" in prompt
    assert "numeric, categorical, and group_comparison" in prompt
    assert "formal normality tests" in prompt
    assert "do not add a separate chart task" in prompt


def test_anomaly_quality_correlation_and_trend_routes_remain_unchanged() -> None:
    planner = AnalysisPlannerService()

    anomaly = planner._mock_tasks(_file_info(), "查找 sales 异常值")[0]
    quality = planner._mock_tasks(_file_info(), "检查整体数据质量")[0]
    correlation = planner._mock_tasks(_file_info(), "分析 sales 和 price 的相关性")[0]
    trend = planner._mock_tasks(_file_info(), "查看 sales 按日期变化趋势")[0]

    assert anomaly.type == "anomaly_detection"
    assert quality.type == "data_quality_analysis"
    assert correlation.type == "correlation_analysis"
    assert trend.type == "trend"


def test_groupby_aggregation_is_not_misrouted_to_distribution() -> None:
    tasks = AnalysisPlannerService()._mock_tasks(_file_info(), "按地区求 sales 总销售额")

    assert any(task.type == "groupby" for task in tasks)
    assert all(task.type != "distribution_analysis" for task in tasks)


def test_clean_distribution_request_does_not_execute_cleaning() -> None:
    task = AnalysisPlannerService()._mock_tasks(_file_info(), "清洗 sales 分布异常，先看直方图")[0]

    assert task.type == "distribution_analysis"


def test_workflow_does_not_append_visualization_to_distribution_task() -> None:
    nodes = DatasetGraphNodes()
    task = AnalysisTask(
        task_name="Distribution",
        reasoning="Analyze sales distribution.",
        expected_output="Histogram and descriptive statistics.",
        type="distribution_analysis",
        params={"mode": "numeric", "columns": ["sales"]},
    )

    tasks = nodes._ensure_visualization_task(_file_info(), [task])

    assert tasks == [task]


def test_workflow_existing_stats_visualization_behavior_is_unchanged() -> None:
    nodes = DatasetGraphNodes()
    task = AnalysisTask(
        task_name="Statistics",
        reasoning="Summarize sales.",
        expected_output="Summary statistics.",
        type="stats",
        params={},
    )

    tasks = nodes._ensure_visualization_task(_file_info(), [task])

    assert len(tasks) == 2
    assert tasks[0] == task
