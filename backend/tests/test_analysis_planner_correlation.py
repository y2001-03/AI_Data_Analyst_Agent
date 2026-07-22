"""Planner and workflow tests for correlation analysis intent."""

from __future__ import annotations

from app.graph.dataset_workflow import DatasetGraphNodes
from app.schemas.file import AnalysisTask, DatasetColumnProfile, DatasetUploadResponse
from app.services.analysis_planner_service import AnalysisPlannerService


def _file_info() -> DatasetUploadResponse:
    return DatasetUploadResponse(
        file_name="sales.csv",
        file_type="csv",
        row_count=100,
        column_count=4,
        preview=[],
        columns=[
            DatasetColumnProfile(name="sales", data_type="float64", missing_count=0, unique_values=90),
            DatasetColumnProfile(name="price", data_type="float64", missing_count=1, unique_values=80),
            DatasetColumnProfile(name="quantity", data_type="int64", missing_count=0, unique_values=20),
            DatasetColumnProfile(name="region", data_type="object", missing_count=0, unique_values=4),
        ],
    )


def test_mock_planner_routes_general_correlation_intent_with_defaults() -> None:
    task = AnalysisPlannerService()._mock_tasks(_file_info(), "哪些字段高度相关？")[0]

    assert task.type == "correlation_analysis"
    assert task.params == {"method": "pearson", "missing_strategy": "pairwise"}
    assert "does not imply causation" in task.expected_output


def test_mock_planner_routes_explicit_field_relationship_intent() -> None:
    task = AnalysisPlannerService()._mock_tasks(_file_info(), "price 和 sales 的关系")[0]

    assert task.type == "correlation_analysis"
    assert task.params["columns"] == ["sales", "price"]


def test_mock_planner_extracts_spearman_and_explicit_columns() -> None:
    task = AnalysisPlannerService()._mock_tasks(
        _file_info(),
        "请分析 sales 与 price 的 Spearman 单调关系",
    )[0]

    assert task.type == "correlation_analysis"
    assert task.params == {
        "method": "spearman",
        "missing_strategy": "pairwise",
        "columns": ["sales", "price"],
    }


def test_mock_planner_extracts_explicit_pearson_and_listwise() -> None:
    task = AnalysisPlannerService()._mock_tasks(
        _file_info(),
        "Use listwise Pearson correlation for price and quantity.",
    )[0]

    assert task.params == {
        "method": "pearson",
        "missing_strategy": "listwise",
        "columns": ["price", "quantity"],
    }


def test_mock_planner_routes_ambiguous_impact_request_without_causal_claim() -> None:
    task = AnalysisPlannerService()._mock_tasks(_file_info(), "销售额受什么影响？")[0]

    assert task.type == "correlation_analysis"
    assert "without making causal claims" in task.reasoning
    assert "does not imply causation" in task.expected_output


def test_normalized_llm_task_gets_safe_correlation_defaults() -> None:
    task = AnalysisPlannerService()._normalize_task(
        {
            "task_name": "Correlation",
            "type": "correlation_analysis",
            "reason": "Analyze association.",
            "params": {"columns": ["sales", "price"]},
        },
        _file_info(),
    )

    assert task is not None
    assert task.params == {
        "columns": ["sales", "price"],
        "method": "pearson",
        "missing_strategy": "pairwise",
    }


def test_planner_prompt_documents_methods_params_and_causation_boundary() -> None:
    prompt = AnalysisPlannerService()._system_prompt()

    assert "correlation_analysis" in prompt
    assert "pearson or spearman" in prompt
    assert "pairwise or listwise" in prompt
    assert "correlation does not imply causation" in prompt


def test_workflow_does_not_append_visualization_to_correlation_task() -> None:
    nodes = object.__new__(DatasetGraphNodes)
    task = AnalysisTask(
        task_name="Correlation Analysis",
        reasoning="Analyze correlations.",
        expected_output="Correlation matrix and heatmap data.",
        type="correlation_analysis",
        params={"method": "pearson"},
    )

    tasks = nodes._ensure_visualization_task(_file_info(), [task])

    assert tasks == [task]


def test_workflow_visualization_behavior_for_existing_stats_task_is_unchanged() -> None:
    nodes = object.__new__(DatasetGraphNodes)
    task = AnalysisTask(
        task_name="Statistics",
        reasoning="Summarize numeric fields.",
        expected_output="Summary statistics.",
        type="stats",
        params={},
    )

    tasks = nodes._ensure_visualization_task(_file_info(), [task])

    assert len(tasks) == 2
    assert tasks[0] == task
