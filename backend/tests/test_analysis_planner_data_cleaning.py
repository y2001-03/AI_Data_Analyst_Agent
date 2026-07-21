"""Planner intent tests for data cleaning plan and execution."""

from __future__ import annotations

from app.schemas.file import DatasetColumnProfile, DatasetUploadResponse
from app.services.analysis_planner_service import AnalysisPlannerService


def _file_info() -> DatasetUploadResponse:
    return DatasetUploadResponse(
        file_name="sales.csv",
        file_type="csv",
        row_count=3,
        column_count=3,
        preview=[],
        columns=[
            DatasetColumnProfile(name="customer_id", data_type="int64", missing_count=0, unique_values=3),
            DatasetColumnProfile(name="sales", data_type="float64", missing_count=1, unique_values=2),
            DatasetColumnProfile(name="order_date", data_type="object", missing_count=0, unique_values=3),
        ],
    )


def test_mock_planner_keeps_quality_question_as_quality_analysis() -> None:
    task = AnalysisPlannerService()._mock_tasks(_file_info(), "缺失值多不多？")[0]

    assert task.type == "data_quality_analysis"


def test_mock_planner_selects_cleaning_plan_for_advice() -> None:
    task = AnalysisPlannerService()._mock_tasks(_file_info(), "这份数据应该怎么清洗？")[0]

    assert task.type == "data_cleaning_plan"
    assert task.params == {"mode": "plan"}


def test_mock_planner_selects_execute_only_for_explicit_action() -> None:
    task = AnalysisPlannerService()._mock_tasks(
        _file_info(),
        "用中位数填充 sales 缺失值",
    )[0]

    assert task.type == "data_cleaning_execute"
    assert task.params["actions"] == [
        {
            "action_type": "fill_missing",
            "column": "sales",
            "parameters": {"strategy": "median"},
        }
    ]


def test_mock_planner_does_not_execute_when_action_is_only_a_question() -> None:
    task = AnalysisPlannerService()._mock_tasks(_file_info(), "重复行要不要删除？")[0]

    assert task.type == "data_cleaning_plan"


def test_mock_planner_extracts_duplicate_and_dtype_actions() -> None:
    planner = AnalysisPlannerService()

    duplicate_task = planner._mock_tasks(_file_info(), "删除重复行")[0]
    convert_task = planner._mock_tasks(_file_info(), "把 order_date 转换成日期")[0]

    assert duplicate_task.params["actions"][0]["action_type"] == "drop_duplicates"
    assert convert_task.params["actions"][0] == {
        "action_type": "convert_dtype",
        "column": "order_date",
        "parameters": {"target_type": "datetime"},
    }


def test_planner_prompt_documents_cleaning_safety_boundary() -> None:
    prompt = AnalysisPlannerService()._system_prompt()

    assert "data_cleaning_plan" in prompt
    assert "data_cleaning_execute" in prompt
    assert "only explicitly requested actions" in prompt
