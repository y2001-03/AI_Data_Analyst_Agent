"""LangGraph workflow for dataset upload orchestration."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
from starlette.datastructures import UploadFile

from langgraph.graph import END, START, StateGraph

from app.core.exceptions import AppException
from app.graph.dataset_state import DatasetGraphState
from app.schemas.file import AIAnalysisResult, AnalysisTask, ExecutionResult
from app.schemas.file import ExecutionChart
from app.services.analysis_planner_service import AnalysisPlannerService
from app.services.data_understanding_service import DataUnderstandingService
from app.services.file_service import FileService
from app.services.pandas_execution_service import PandasExecutionService


class DatasetGraphNodes:
    """Node implementations for the dataset upload graph."""

    def __init__(self) -> None:
        self.file_service = FileService()
        self.data_understanding_service = DataUnderstandingService()
        self.analysis_planner_service = AnalysisPlannerService()
        self.execution_service = PandasExecutionService()

    async def understand(self, state: DatasetGraphState) -> DatasetGraphState:
        """Parse the upload and generate AI understanding."""
        memory_context = state.get("memory_context")
        effective_question = state.get("question")
        if not effective_question and isinstance(memory_context, dict):
            remembered_question = memory_context.get("last_question")
            if isinstance(remembered_question, str) and remembered_question:
                state["question"] = remembered_question
                effective_question = remembered_question
        input_summary = {
            "file_name": state["file_name"],
            "question": effective_question,
            "memory_context": bool(memory_context),
        }
        try:
            upload = UploadFile(
                file=BytesIO(state["content"]),
                filename=state["file_name"],
            )
            file_info = await self.file_service.parse_upload(upload)
            ai_analysis, fallback_used, fallback_reason = (
                self.data_understanding_service.understand_file_info(
                    file_info,
                    effective_question,
                    memory_context,
                )
            )
        except AppException as exc:
            state["error_stage"] = "understand"
            state["error_reason"] = exc.message
            self._record_trace(
                state,
                node="understand",
                status="failed",
                input_summary=input_summary,
                output_summary={"reason": exc.message},
            )
            return state
        except Exception as exc:
            state["error_stage"] = "understand"
            state["error_reason"] = str(exc)
            self._record_trace(
                state,
                node="understand",
                status="failed",
                input_summary=input_summary,
                output_summary={"reason": str(exc)},
            )
            return state
        state["file_info"] = file_info
        state["schema"] = [column.model_dump() for column in file_info.columns]
        state["ai_analysis"] = ai_analysis
        if fallback_used:
            state["fallback_used"] = True
            if not state["fallback_reason"]:
                state["fallback_reason"] = fallback_reason or "timeout"
        self._record_trace(
            state,
            node="understand",
            status="fallback" if fallback_used else "success",
            input_summary=input_summary,
            output_summary={
                "row_count": file_info.row_count,
                "column_count": file_info.column_count,
                "suggestion_count": len(ai_analysis.suggestions),
                "fallback_reason": fallback_reason,
            },
        )
        return state

    def plan(self, state: DatasetGraphState) -> DatasetGraphState:
        """Generate planned analysis tasks."""
        if state["error_stage"] or not state["file_info"] or not state["ai_analysis"]:
            self._record_trace(
                state,
                node="plan",
                status="skipped",
                input_summary={"reason": "understand_failed_or_missing_inputs"},
                output_summary={},
            )
            return state
        input_summary = {
            "question": state.get("question"),
            "memory_context": bool(state.get("memory_context")),
            "summary": state["ai_analysis"].summary,
            "suggestion_count": len(state["ai_analysis"].suggestions),
        }
        try:
            mock_tasks = self.analysis_planner_service._mock_tasks(
                state["file_info"],
                state.get("question"),
            )
            tasks = self.analysis_planner_service.plan_tasks(
                state["file_info"],
                state["ai_analysis"],
                state.get("question"),
                state.get("memory_context"),
            )
            state["planner_failed"] = self._same_tasks(tasks, mock_tasks)
        except AppException as exc:
            state["planner_failed"] = True
            state["fallback_used"] = True
            if not state["fallback_reason"]:
                state["fallback_reason"] = exc.message
            state["tasks"] = self.analysis_planner_service._mock_tasks(
                state["file_info"],
                state.get("question"),
            )
            state["tasks"] = self._ensure_visualization_task(state["file_info"], state["tasks"])
            if state["ai_analysis"] is not None:
                state["ai_analysis"].tasks = state["tasks"]
            self._record_trace(
                state,
                node="plan",
                status="fallback",
                input_summary=input_summary,
                output_summary={"reason": exc.message, "task_count": len(state["tasks"])},
            )
            return state
        except Exception as exc:
            state["planner_failed"] = True
            state["fallback_used"] = True
            if not state["fallback_reason"]:
                state["fallback_reason"] = str(exc)
            state["tasks"] = self.analysis_planner_service._mock_tasks(
                state["file_info"],
                state.get("question"),
            )
            state["tasks"] = self._ensure_visualization_task(state["file_info"], state["tasks"])
            if state["ai_analysis"] is not None:
                state["ai_analysis"].tasks = state["tasks"]
            self._record_trace(
                state,
                node="plan",
                status="fallback",
                input_summary=input_summary,
                output_summary={"reason": str(exc), "task_count": len(state["tasks"])},
            )
            return state
        state["tasks"] = tasks
        state["tasks"] = self._ensure_visualization_task(state["file_info"], state["tasks"])
        if state["ai_analysis"] is not None:
            state["ai_analysis"].tasks = state["tasks"]
        status = "fallback" if state["planner_failed"] else "success"
        if state["planner_failed"]:
            state["fallback_used"] = True
            if not state["fallback_reason"]:
                state["fallback_reason"] = "Planner returned fallback tasks."
        self._record_trace(
            state,
            node="plan",
            status=status,
            input_summary=input_summary,
            output_summary={"task_count": len(state["tasks"])},
        )
        return state

    def execute(self, state: DatasetGraphState) -> DatasetGraphState:
        """Execute planned tasks with pandas."""
        if state["error_stage"] == "understand" or not state["tasks"]:
            self._record_trace(
                state,
                node="execute",
                status="skipped",
                input_summary={"reason": "no_tasks_or_understand_failed"},
                output_summary={},
            )
            return state
        input_summary = {"task_count": len(state["tasks"])}
        try:
            dataframe = state.get("dataframe")
            if dataframe is None:
                dataframe = self._load_dataframe(state["file_name"], state["content"])
            result = self.execution_service.execute_tasks(dataframe, state["tasks"])
            state["execution_results"] = result.execution_results
            self._record_trace(
                state,
                node="execute",
                status="success",
                input_summary=input_summary,
                output_summary={"result_count": len(result.execution_results)},
            )
        except AppException as exc:
            state["execution_failed"] = True
            state["execution_results"] = [self._skipped_execution_result(exc.message)]
            self._record_trace(
                state,
                node="execute",
                status="skipped",
                input_summary=input_summary,
                output_summary={"reason": exc.message},
            )
        except Exception as exc:
            state["execution_failed"] = True
            state["execution_results"] = [self._skipped_execution_result(str(exc))]
            self._record_trace(
                state,
                node="execute",
                status="skipped",
                input_summary=input_summary,
                output_summary={"reason": str(exc)},
            )
        return state

    def route_after_understand(self, state: DatasetGraphState) -> str:
        """Route to end when understanding fails."""
        return "plan"

    def route_after_plan(self, state: DatasetGraphState) -> str:
        """Route from planning into execution or fallback execution."""
        if state["tasks"]:
            return "execute"
        return "end"

    def route_after_execute(self, _state: DatasetGraphState) -> str:
        """Always finish after execution."""
        return "end"

    def _load_dataframe(self, file_name: str, content: bytes) -> pd.DataFrame:
        """Load bytes into a dataframe for execution."""
        extension = Path(file_name).suffix.lower()
        buffer = BytesIO(content)
        if extension == ".csv":
            return pd.read_csv(buffer)
        return pd.read_excel(buffer)

    def _same_tasks(
        self,
        first: list[AnalysisTask],
        second: list[AnalysisTask],
    ) -> bool:
        """Compare task lists by their serialized values."""
        left = [task.model_dump() for task in first]
        right = [task.model_dump() for task in second]
        return left == right

    def _skipped_execution_result(self, reason: str) -> ExecutionResult:
        """Return a graph-level skipped execution result."""
        return ExecutionResult(
            task_name="execution",
            type="chart",
            data={
                "status": "skipped",
                "reason": reason,
                "rows": [{"status": "skipped", "value": 0}],
                "chart_type": "bar",
                "labels": ["skipped"],
                "values": [0],
            },
            chart=ExecutionChart(chart_type="bar", x=["skipped"], y=[0]),
        )

    def _ensure_visualization_task(
        self,
        file_info,
        tasks: list[AnalysisTask],
    ) -> list[AnalysisTask]:
        """Append one visualization-oriented task when the plan lacks one."""
        task_texts = [
            " ".join(
                [
                    task.task_name,
                    task.reasoning,
                    task.expected_output,
                    task.type or "",
                ]
            ).lower()
            for task in tasks
        ]
        visualization_keywords = ("chart", "visual", "trend", "group", "segment", "plot")
        if any(any(keyword in text for keyword in visualization_keywords) for text in task_texts):
            return tasks
        has_datetime = any(
            "date" in column.data_type.lower() or "time" in column.name.lower()
            for column in file_info.columns
        )
        if has_datetime:
            visualization_task = AnalysisTask(
                task_name="Visualization Trend Task",
                reasoning="Ensure the analysis includes a chart-friendly trend output.",
                expected_output="A trend chart for the most relevant metric over time.",
            )
        else:
            visualization_task = AnalysisTask(
                task_name="Visualization GroupBy Task",
                reasoning="Ensure the analysis includes a chart-friendly grouped output.",
                expected_output="A grouped bar chart for the most relevant category and metric.",
            )
        return [*tasks, visualization_task]

    def _record_trace(
        self,
        state: DatasetGraphState,
        node: str,
        status: str,
        input_summary: dict[str, object],
        output_summary: dict[str, object],
    ) -> None:
        """Record node observability data into graph state."""
        state["execution_path"].append(node)
        state["node_status"][node] = status
        state["trace_log"].append(
            {
                "node": node,
                "status": status,
                "input_summary": input_summary,
                "output_summary": output_summary,
            }
        )


def build_dataset_upload_graph():
    """Build the LangGraph workflow for the dataset upload pipeline."""
    nodes = DatasetGraphNodes()
    graph = StateGraph(DatasetGraphState)
    graph.add_node("understand", nodes.understand)
    graph.add_node("plan", nodes.plan)
    graph.add_node("execute", nodes.execute)
    graph.add_edge(START, "understand")
    graph.add_conditional_edges(
        "understand",
        nodes.route_after_understand,
        {
            "plan": "plan",
            "end": END,
        },
    )
    graph.add_conditional_edges(
        "plan",
        nodes.route_after_plan,
        {
            "execute": "execute",
            "end": END,
        },
    )
    graph.add_conditional_edges(
        "execute",
        nodes.route_after_execute,
        {
            "end": END,
        },
    )
    return graph.compile()
