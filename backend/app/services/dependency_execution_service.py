"""Dependency-aware serial execution coordinator for PlannerPlan tasks."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd

from app.schemas.dependency_execution import (
    DependencyExecutionSummary,
    StageExecutionRecord,
    TaskExecutionRecord,
)
from app.schemas.file import AnalysisTask, ExecutionResult
from app.schemas.planner_contract import PlannerExecutionStage, PlannerPlan, PlannerTask
from app.services.pandas_execution_service import PandasExecutionService


@dataclass(frozen=True)
class DependencyExecutionResponse:
    """Coordinator response used by the workflow execute node."""

    execution_results: list[ExecutionResult]
    task_records: list[TaskExecutionRecord]
    stage_records: list[StageExecutionRecord]
    summary: DependencyExecutionSummary


class DependencyAwareExecutionService:
    """Execute validated planner tasks by dependency stage without parallelism."""

    def __init__(self, execution_service: PandasExecutionService | None = None) -> None:
        self.execution_service = execution_service or PandasExecutionService()

    def execute_plan(
        self,
        dataframe: pd.DataFrame,
        plan: PlannerPlan,
        tasks: list[AnalysisTask],
    ) -> DependencyExecutionResponse:
        """Execute a valid executable plan with dependency-aware fail-soft semantics."""
        if plan.status != "valid" or not plan.execution_supported:
            raise ValueError("Dependency-aware execution requires a valid executable PlannerPlan.")
        self._validate_plan_shape(plan, tasks)
        started = time.monotonic()
        task_by_id = {task.task_id: task for task in plan.tasks}
        stage_by_task_id = self._stage_by_task_id(plan.execution_stages)
        analysis_task_by_id = {
            task_id: tasks[index]
            for index, task_id in enumerate(plan.execution_order)
        }
        execution_results: list[ExecutionResult] = []
        records: dict[str, TaskExecutionRecord] = {}
        stage_records: list[StageExecutionRecord] = []

        for stage in sorted(plan.execution_stages, key=lambda item: item.order):
            stage_start = time.monotonic()
            stage_started_at = self._now()
            for task_id in stage.task_ids:
                planner_task = task_by_id[task_id]
                blocked_by = self._blocked_dependencies(planner_task, records)
                if blocked_by:
                    records[task_id] = self._task_record(
                        planner_task,
                        stage_by_task_id[task_id],
                        status="blocked",
                        blocked_by=blocked_by,
                        error_code="DEPENDENCY_NOT_SUCCEEDED",
                        error_message="One or more dependency tasks did not succeed.",
                    )
                    continue
                if planner_task.requires_confirmation:
                    records[task_id] = self._task_record(
                        planner_task,
                        stage_by_task_id[task_id],
                        status="confirmation_required",
                        error_code="CONFIRMATION_REQUIRED",
                        error_message="Task requires confirmation, but no confirmation mechanism is available in this workflow.",
                    )
                    continue
                record, result = self._execute_one(
                    dataframe,
                    planner_task,
                    analysis_task_by_id[task_id],
                    stage_by_task_id[task_id],
                    result_index=len(execution_results),
                )
                records[task_id] = record
                if result is not None:
                    execution_results.append(result)
            stage_records.append(
                self._stage_record(
                    stage,
                    [records[task_id] for task_id in stage.task_ids],
                    started_at=stage_started_at,
                    elapsed_ms=self._elapsed_ms(stage_start),
                )
            )

        ordered_records = [records[task_id] for task_id in plan.execution_order]
        extra_tasks = tasks[len(plan.execution_order):]
        if extra_tasks:
            extra_records, extra_stage, extra_results = self._execute_extra_workflow_tasks(
                dataframe,
                extra_tasks,
                result_start_index=len(execution_results),
                execution_order_start=len(ordered_records) + 1,
            )
            execution_results.extend(extra_results)
            ordered_records.extend(extra_records)
            stage_records.append(extra_stage)
        return DependencyExecutionResponse(
            execution_results=execution_results,
            task_records=ordered_records,
            stage_records=stage_records,
            summary=self._summary(ordered_records, stage_records, self._elapsed_ms(started)),
        )

    def _execute_extra_workflow_tasks(
        self,
        dataframe: pd.DataFrame,
        tasks: list[AnalysisTask],
        *,
        result_start_index: int,
        execution_order_start: int,
    ) -> tuple[list[TaskExecutionRecord], StageExecutionRecord, list[ExecutionResult]]:
        started = time.monotonic()
        started_at = self._now()
        records: list[TaskExecutionRecord] = []
        results: list[ExecutionResult] = []
        for index, task in enumerate(tasks):
            task_id = f"workflow_extra_{index + 1}"
            task_started = time.monotonic()
            task_started_at = self._now()
            try:
                response = self.execution_service.execute_tasks(dataframe, [task])
                results.extend(response.execution_results)
                result_index = result_start_index + len(results) - 1
                records.append(
                    TaskExecutionRecord(
                        task_id=task_id,
                        task_type=task.type or "stats",
                        stage_id="workflow_extra",
                        execution_order=execution_order_start + index,
                        status="succeeded",
                        started_at=task_started_at,
                        finished_at=self._now(),
                        elapsed_ms=self._elapsed_ms(task_started),
                        result_index=result_index,
                        metadata={"workflow_added": True},
                    )
                )
            except Exception as exc:
                records.append(
                    TaskExecutionRecord(
                        task_id=task_id,
                        task_type=task.type or "stats",
                        stage_id="workflow_extra",
                        execution_order=execution_order_start + index,
                        status="failed",
                        started_at=task_started_at,
                        finished_at=self._now(),
                        elapsed_ms=self._elapsed_ms(task_started),
                        error_code=type(exc).__name__,
                        error_message=self._safe_error_message(exc),
                        metadata={"workflow_added": True},
                    )
                )
        return records, self._stage_record(
            PlannerExecutionStage(
                stage_id="workflow_extra",
                order=max((record.execution_order or 0 for record in records), default=execution_order_start),
                task_ids=tuple(record.task_id for record in records),
                parallelizable=False,
                risk_level="read_only",
                requires_confirmation=False,
                description="Workflow-added compatibility tasks executed after planner tasks.",
            ),
            records,
            started_at=started_at,
            elapsed_ms=self._elapsed_ms(started),
        ), results

    def _execute_one(
        self,
        dataframe: pd.DataFrame,
        planner_task: PlannerTask,
        analysis_task: AnalysisTask,
        stage_id: str,
        *,
        result_index: int,
    ) -> tuple[TaskExecutionRecord, ExecutionResult | None]:
        started = time.monotonic()
        started_at = self._now()
        try:
            response = self.execution_service.execute_tasks(dataframe, [analysis_task])
            result = response.execution_results[0]
            return (
                self._task_record(
                    planner_task,
                    stage_id,
                    status="succeeded",
                    started_at=started_at,
                    elapsed_ms=self._elapsed_ms(started),
                    result_index=result_index,
                ),
                result,
            )
        except Exception as exc:
            return (
                self._task_record(
                    planner_task,
                    stage_id,
                    status="failed",
                    started_at=started_at,
                    elapsed_ms=self._elapsed_ms(started),
                    error_code=type(exc).__name__,
                    error_message=self._safe_error_message(exc),
                ),
                None,
            )

    def _validate_plan_shape(self, plan: PlannerPlan, tasks: list[AnalysisTask]) -> None:
        task_ids = [task.task_id for task in plan.tasks]
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("PlannerPlan contains duplicate task ids.")
        if set(plan.execution_order) != set(task_ids) or len(plan.execution_order) != len(task_ids):
            raise ValueError("PlannerPlan execution_order must include every task exactly once.")
        if len(tasks) < len(plan.execution_order):
            raise ValueError("Adapted AnalysisTask list is missing planner tasks.")
        staged_ids = [task_id for stage in plan.execution_stages for task_id in stage.task_ids]
        if set(staged_ids) != set(task_ids) or len(staged_ids) != len(task_ids):
            raise ValueError("PlannerPlan execution_stages must include every task exactly once.")

    def _blocked_dependencies(
        self,
        task: PlannerTask,
        records: dict[str, TaskExecutionRecord],
    ) -> list[str]:
        blocked: list[str] = []
        for dependency_id in task.depends_on:
            dependency = records.get(dependency_id)
            if dependency is None or dependency.status != "succeeded":
                blocked.append(dependency_id)
        return blocked

    def _task_record(
        self,
        task: PlannerTask,
        stage_id: str,
        *,
        status: str,
        started_at: str | None = None,
        elapsed_ms: int | None = None,
        blocked_by: list[str] | None = None,
        result_index: int | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> TaskExecutionRecord:
        finished_at = self._now() if status in {"succeeded", "failed", "blocked", "confirmation_required"} else None
        return TaskExecutionRecord(
            task_id=task.task_id,
            task_type=task.task_type,
            stage_id=stage_id,
            execution_order=task.execution_order,
            status=status,
            started_at=started_at,
            finished_at=finished_at,
            elapsed_ms=elapsed_ms,
            dependency_task_ids=list(task.depends_on),
            blocked_by=blocked_by or [],
            result_index=result_index,
            error_code=error_code,
            error_message=error_message,
            metadata={"requires_confirmation": task.requires_confirmation},
        )

    def _stage_record(
        self,
        stage: PlannerExecutionStage,
        task_records: list[TaskExecutionRecord],
        *,
        started_at: str,
        elapsed_ms: int,
    ) -> StageExecutionRecord:
        succeeded_count = sum(record.status == "succeeded" for record in task_records)
        failed_count = sum(record.status == "failed" for record in task_records)
        blocked_count = sum(record.status == "blocked" for record in task_records)
        confirmation_required_count = sum(record.status == "confirmation_required" for record in task_records)
        return StageExecutionRecord(
            stage_id=stage.stage_id,
            order=stage.order,
            task_ids=list(stage.task_ids),
            status=self._stage_status(
                total=len(task_records),
                succeeded=succeeded_count,
                failed=failed_count,
                blocked=blocked_count,
                confirmation_required=confirmation_required_count,
            ),
            started_at=started_at,
            finished_at=self._now(),
            elapsed_ms=elapsed_ms,
            succeeded_count=succeeded_count,
            failed_count=failed_count,
            blocked_count=blocked_count,
            confirmation_required_count=confirmation_required_count,
        )

    def _stage_status(
        self,
        *,
        total: int,
        succeeded: int,
        failed: int,
        blocked: int,
        confirmation_required: int,
    ) -> str:
        if succeeded == total:
            return "succeeded"
        if blocked == total:
            return "blocked"
        if confirmation_required == total:
            return "confirmation_required"
        if failed == total:
            return "failed"
        if confirmation_required > 0:
            return "confirmation_required" if succeeded == 0 and failed == 0 else "partially_failed"
        if failed > 0 and succeeded == 0:
            return "failed"
        if failed > 0 or blocked > 0:
            return "partially_failed"
        return "blocked"

    def _summary(
        self,
        task_records: list[TaskExecutionRecord],
        stage_records: list[StageExecutionRecord],
        elapsed_ms: int,
    ) -> DependencyExecutionSummary:
        succeeded = sum(record.status == "succeeded" for record in task_records)
        failed = sum(record.status == "failed" for record in task_records)
        blocked = sum(record.status == "blocked" for record in task_records)
        confirmation_required = sum(record.status == "confirmation_required" for record in task_records)
        return DependencyExecutionSummary(
            total_tasks=len(task_records),
            succeeded_tasks=succeeded,
            failed_tasks=failed,
            blocked_tasks=blocked,
            confirmation_required_tasks=confirmation_required,
            total_stages=len(stage_records),
            succeeded_stages=sum(record.status == "succeeded" for record in stage_records),
            failed_stages=sum(record.status in {"failed", "partially_failed"} for record in stage_records),
            elapsed_ms=elapsed_ms,
            completed=True,
            partial_success=succeeded > 0 and succeeded < len(task_records),
        )

    def _stage_by_task_id(self, stages: tuple[PlannerExecutionStage, ...]) -> dict[str, str]:
        return {
            task_id: stage.stage_id
            for stage in stages
            for task_id in stage.task_ids
        }

    def _now(self) -> str:
        return datetime.now(UTC).isoformat()

    def _elapsed_ms(self, started: float) -> int:
        return int((time.monotonic() - started) * 1000)

    def _safe_error_message(self, exc: Exception) -> str:
        message = str(exc) or type(exc).__name__
        return message[:300]
