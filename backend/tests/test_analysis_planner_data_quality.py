"""Planner tests for data quality task selection."""

from __future__ import annotations

from app.schemas.file import DatasetColumnProfile, DatasetUploadResponse
from app.services.analysis_planner_service import AnalysisPlannerService


def test_mock_planner_selects_data_quality_task_for_quality_intent() -> None:
    file_info = DatasetUploadResponse(
        file_name="sales.csv",
        file_type="csv",
        row_count=3,
        column_count=2,
        preview=[],
        columns=[
            DatasetColumnProfile(name="customer_id", data_type="int64", missing_count=0, unique_values=3),
            DatasetColumnProfile(name="sales", data_type="float64", missing_count=1, unique_values=2),
        ],
    )

    tasks = AnalysisPlannerService()._mock_tasks(file_info, "Check data quality before analysis.")

    assert len(tasks) == 1
    assert tasks[0].type == "data_quality_analysis"
