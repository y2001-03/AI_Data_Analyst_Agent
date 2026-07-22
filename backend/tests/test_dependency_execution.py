"""Tests for dependency-aware workflow execution."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from app.graph.dataset_workflow import DatasetGraphNodes
from app.schemas.file import AIAnalysisResult, AnalysisTask, DatasetColumnProfile, DatasetUploadResponse, ExecutionResponse, ExecutionResult
from app.services.dependency_execution_service import DependencyAwareExecutionService
from app.services.pandas_execution_service import PandasExecutionService
from app.services.planner_contract_builder import PlannerContractBuilder


def _file_info() -> DatasetUploadResponse:
    return DatasetUploadResponse(
        file_name="sales.csv",
        file_type="csv",
        row_count=10,
        column_count=2,
        preview=[],
        columns=[
            DatasetColumnProfile(name="sales", data_type="float64", missing_count=0, unique_values=10),
            DatasetColumnProfile(name="region", data_type="object", missing_count=0, unique_values=2),
        ],
    )


def _plan(raw_tasks: list[dict[str, object]]):
    return PlannerContractBuilder().build_plan({"tasks": raw_tasks}, _file_info(), source="fallback")


def _tasks(plan):
    return PlannerContractBuilder().to_analysis_tasks(plan)


class FakeExecutionService:
    def __init__(self, fail_task_names: set[str] | None = None) -> None:
        self.fail_task_names = fail_task_names or set()
        self.calls: list[str] = []

    def execute_tasks(self, dataframe, tasks):
        task = tasks[0]
        self.calls.append(task.task_name)
        if task.task_name in self.fail_task_names:
            raise RuntimeError(f"boom {task.task_name}")
        return ExecutionResponse(
            execution_results=[
                ExecutionResult(
                    task_name=task.task_name,
                    type=task.type or "stats",
                    data={"status": "ok", "task": task.task_name},
                    chart=None,
                )
            ]
        )


def test_single_task_success_records_summary_and_result_index() -> None:
    plan = _plan([{"task_id": "task_1", "task_type": "stats", "name": "Stats"}])
    service = FakeExecutionService()
    result = DependencyAwareExecutionService(service).execute_plan(pd.DataFrame({"sales": [1]}), plan, _tasks(plan))

    assert service.calls == ["Stats"]
    assert result.task_records[0].status == "succeeded"
    assert result.task_records[0].result_index == 0
    assert result.stage_records[0].status == "succeeded"
    assert result.summary.succeeded_tasks == 1
    json.dumps(result.summary.model_dump(), allow_nan=False)


def test_independent_tasks_execute_serially_in_execution_order() -> None:
    plan = _plan(
        [
            {"task_id": "task_1", "task_type": "stats", "name": "Stats A"},
            {"task_id": "task_2", "task_type": "data_quality_analysis", "name": "Quality"},
        ]
    )
    service = FakeExecutionService()
    result = DependencyAwareExecutionService(service).execute_plan(pd.DataFrame({"sales": [1]}), plan, _tasks(plan))

    assert service.calls == ["Stats A", "Quality"]
    assert [record.task_id for record in result.task_records] == ["task_1", "task_2"]
    assert [record.result_index for record in result.task_records] == [0, 1]
    assert result.stage_records[0].task_ids == ["task_1", "task_2"]


def test_dependency_success_allows_downstream_execution() -> None:
    plan = _plan(
        [
            {"task_id": "task_1", "task_type": "data_quality_analysis", "name": "Quality"},
            {"task_id": "task_2", "task_type": "stats", "name": "Stats", "depends_on": ["task_1"]},
        ]
    )
    service = FakeExecutionService()
    result = DependencyAwareExecutionService(service).execute_plan(pd.DataFrame({"sales": [1]}), plan, _tasks(plan))

    assert service.calls == ["Quality", "Stats"]
    assert [record.status for record in result.task_records] == ["succeeded", "succeeded"]
    assert [stage.status for stage in result.stage_records] == ["succeeded", "succeeded"]


def test_failed_dependency_blocks_downstream_but_not_independent_task() -> None:
    plan = _plan(
        [
            {"task_id": "task_1", "task_type": "data_quality_analysis", "name": "Quality"},
            {"task_id": "task_2", "task_type": "stats", "name": "Dependent Stats", "depends_on": ["task_1"]},
            {"task_id": "task_3", "task_type": "stats", "name": "Independent Stats"},
        ]
    )
    service = FakeExecutionService(fail_task_names={"Quality"})
    result = DependencyAwareExecutionService(service).execute_plan(pd.DataFrame({"sales": [1]}), plan, _tasks(plan))

    assert service.calls == ["Quality", "Independent Stats"]
    records = {record.task_id: record for record in result.task_records}
    assert records["task_1"].status == "failed"
    assert records["task_2"].status == "blocked"
    assert records["task_2"].blocked_by == ["task_1"]
    assert records["task_3"].status == "succeeded"
    assert result.stage_records[0].status == "partially_failed"
    assert result.stage_records[1].status == "blocked"
    assert result.summary.partial_success is True


def test_confirmation_required_task_is_not_executed_and_blocks_downstream() -> None:
    plan = _plan(
        [
            {
                "task_id": "task_1",
                "task_type": "data_cleaning_execute",
                "name": "Clean",
                "params": {"mode": "execute", "actions": [{"action_type": "trim_strings"}]},
            },
            {"task_id": "task_2", "task_type": "stats", "name": "Stats", "depends_on": ["task_1"]},
        ]
    )
    service = FakeExecutionService()
    result = DependencyAwareExecutionService(service).execute_plan(pd.DataFrame({"sales": [1]}), plan, _tasks(plan))

    assert service.calls == []
    assert [record.status for record in result.task_records] == ["confirmation_required", "blocked"]
    assert result.summary.confirmation_required_tasks == 1
    assert result.summary.blocked_tasks == 1


def test_task_is_executed_once_and_results_are_stable() -> None:
    plan = _plan(
        [
            {"task_id": "task_1", "task_type": "stats", "name": "Stats A"},
            {"task_id": "task_2", "task_type": "stats", "name": "Stats B"},
        ]
    )
    service = FakeExecutionService()
    result = DependencyAwareExecutionService(service).execute_plan(pd.DataFrame({"sales": [1]}), plan, _tasks(plan))

    assert service.calls == ["Stats A", "Stats B"]
    assert [execution.task_name for execution in result.execution_results] == ["Stats A", "Stats B"]
    assert [record.result_index for record in result.task_records] == [0, 1]


def test_execution_supported_false_plan_is_rejected_by_coordinator() -> None:
    plan = _plan(
        [
            {
                "task_id": "task_1",
                "task_type": "data_cleaning_execute",
                "name": "Clean",
                "params": {"mode": "execute", "actions": [{"action_type": "trim_strings"}]},
            },
            {
                "task_id": "task_2",
                "task_type": "stats",
                "name": "Stats",
                "depends_on": ["task_1"],
                "input_bindings": [
                    {
                        "input_name": "dataset",
                        "source_type": "task_output",
                        "source_task_id": "task_1",
                        "source_path": "data.cleaned_preview",
                    }
                ],
            },
        ]
    )
    assert plan.execution_supported is False
    with pytest.raises(ValueError, match="valid executable"):
        DependencyAwareExecutionService(FakeExecutionService()).execute_plan(pd.DataFrame({"sales": [1]}), plan, [])


def test_invalid_plan_shape_fails_before_partial_execution() -> None:
    plan = _plan([{"task_id": "task_1", "task_type": "stats", "name": "Stats"}])
    invalid_plan = plan.model_copy(update={"execution_order": ("missing",)})
    service = FakeExecutionService()
    with pytest.raises(ValueError):
        DependencyAwareExecutionService(service).execute_plan(pd.DataFrame({"sales": [1]}), invalid_plan, _tasks(plan))
    assert service.calls == []


def test_dataframe_remains_isolated_through_pandas_execution_service() -> None:
    class MutatingTool:
        name = "stats_tool"

        def __init__(self) -> None:
            self.calls = 0

        def run(self, dataframe, task, context):
            self.calls += 1
            assert "mutated" not in dataframe.columns
            dataframe["mutated"] = self.calls
            return ExecutionResult(task_name=task.task_name, type="stats", data={"call": self.calls}, chart=None)

    class Registry:
        def __init__(self, tool) -> None:
            self.tool = tool

        def get(self, name):
            return self.tool if name == "stats_tool" else None

    tool = MutatingTool()
    dataframe = pd.DataFrame({"sales": [1, 2]})
    plan = _plan(
        [
            {"task_id": "task_1", "task_type": "stats", "name": "Stats A"},
            {"task_id": "task_2", "task_type": "stats", "name": "Stats B"},
        ]
    )
    execution = PandasExecutionService(Registry(tool))
    result = DependencyAwareExecutionService(execution).execute_plan(dataframe, plan, _tasks(plan))

    assert tool.calls == 2
    assert "mutated" not in dataframe.columns
    assert [record.status for record in result.task_records] == ["succeeded", "succeeded"]


def test_workflow_execute_node_stores_dependency_records() -> None:
    plan = _plan([{"task_id": "task_1", "task_type": "stats", "name": "Stats"}])
    nodes = object.__new__(DatasetGraphNodes)
    nodes.execution_service = FakeExecutionService()
    nodes.dependency_execution_service = DependencyAwareExecutionService(nodes.execution_service)
    state = {
        "file_name": "sales.csv",
        "content": b"",
        "dataframe": pd.DataFrame({"sales": [1]}),
        "tasks": _tasks(plan),
        "planner_plan": plan.model_dump(mode="json"),
        "error_stage": None,
        "execution_results": [],
        "execution_failed": False,
        "trace_log": [],
        "node_status": {},
        "execution_path": [],
        "task_execution_records": [],
        "stage_execution_records": [],
        "execution_summary": None,
        "failed_task_ids": [],
        "blocked_task_ids": [],
        "confirmation_required_task_ids": [],
    }
    result = nodes.execute(state)

    assert result["execution_results"][0].task_name == "Stats"
    assert result["task_execution_records"][0]["status"] == "succeeded"
    assert result["stage_execution_records"][0]["status"] == "succeeded"
    assert result["execution_summary"]["succeeded_tasks"] == 1
    assert result["trace_log"][-1]["node"] == "execute"
