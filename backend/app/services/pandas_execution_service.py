"""Pandas execution layer for planned analysis tasks."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.schemas.file import AnalysisTask, ExecutionResponse, ExecutionResult
from app.schemas.file import ExecutionChart
from app.tools import DatasetContext, ToolRegistry


@dataclass(frozen=True)
class TaskToolMapping:
    """Explicit mapping rule from a planned task to a tool name."""

    tool_name: str
    keywords: tuple[str, ...]

    def matches(self, task_text: str) -> bool:
        """Return whether the rule applies to the task text."""
        return any(keyword in task_text for keyword in self.keywords)


class PandasExecutionService:
    """Execute planned tasks by dispatching them to registered tools."""

    def __init__(self, registry: ToolRegistry | None = None) -> None:
        self.registry = registry or ToolRegistry.with_default_tools()
        self.task_type_tools = {
            "correlation_analysis": "correlation_analysis_tool",
            "data_cleaning_plan": "data_cleaning_tool",
            "data_cleaning_execute": "data_cleaning_tool",
            "data_quality_analysis": "data_quality_tool",
            "sql": "sql_tool",
            "trend": "trend_tool",
            "groupby": "groupby_tool",
            "stats": "stats_tool",
        }
        self.task_tool_mappings = (
            TaskToolMapping(
                tool_name="correlation_analysis_tool",
                keywords=(
                    "correlation",
                    "correlation coefficient",
                    "pearson",
                    "spearman",
                    "multicollinearity",
                    "相关",
                    "关联",
                    "相关系数",
                    "共线性",
                ),
            ),
            TaskToolMapping(
                tool_name="data_cleaning_tool",
                keywords=(
                    "data cleaning",
                    "cleaning plan",
                    "clean data",
                    "fill missing",
                    "drop duplicate",
                    "trim string",
                    "convert dtype",
                    "数据清洗",
                    "清洗方案",
                    "填充缺失",
                    "删除重复",
                    "清理空格",
                    "类型转换",
                ),
            ),
            TaskToolMapping(
                tool_name="data_quality_tool",
                keywords=(
                    "data quality",
                    "quality",
                    "missing",
                    "duplicate",
                    "reliable",
                    "reliability",
                    "outlier",
                    "constant column",
                    "id column",
                    "数据质量",
                    "缺失",
                    "重复",
                    "异常值",
                    "字段质量",
                    "有没有问题",
                    "是否可靠",
                    "分析前",
                ),
            ),
            TaskToolMapping(
                tool_name="sql_tool",
                keywords=("select ", "group by", "order by", "sql"),
            ),
            TaskToolMapping(
                tool_name="trend_tool",
                keywords=("trend", "time", "date", "daily", "weekly", "monthly"),
            ),
            TaskToolMapping(
                tool_name="groupby_tool",
                keywords=("group", "segment", "category", "product", "by "),
            ),
            TaskToolMapping(
                tool_name="stats_tool",
                keywords=(
                    "summary",
                    "statistics",
                    "distribution",
                    "top",
                    "highest",
                    "lowest",
                    "sort",
                    "rank",
                ),
            ),
        )

    def execute_tasks(
        self,
        dataframe: pd.DataFrame,
        tasks: list[AnalysisTask],
    ) -> ExecutionResponse:
        """Execute a list of tasks against a dataframe."""
        context = self._build_context(dataframe)
        results = [
            self._execute_task(dataframe.copy(deep=True), task, context)
            for task in tasks
        ]
        return ExecutionResponse(execution_results=results)

    def _execute_task(
        self,
        dataframe: pd.DataFrame,
        task: AnalysisTask,
        context: DatasetContext,
    ) -> ExecutionResult:
        """Resolve a task to a tool and execute it."""
        tool_name = self._resolve_tool_name(task)
        tool = self.registry.get(tool_name)
        if tool is None:
            return self._skipped_result(task, f"Tool '{tool_name}' is not registered.")
        return tool.run(dataframe, task, context)

    def _resolve_tool_name(self, task: AnalysisTask) -> str:
        """Explicitly map a task to one registered tool."""
        if task.type in self.task_type_tools:
            return self.task_type_tools[task.type]
        task_text = self._task_text(task)
        for mapping in self.task_tool_mappings:
            if mapping.matches(task_text):
                return mapping.tool_name
        return "stats_tool"

    def _build_context(self, dataframe: pd.DataFrame) -> DatasetContext:
        """Infer reusable dataframe metadata."""
        datetime_columns: list[str] = []
        for column in dataframe.columns:
            series = dataframe[column]
            if pd.api.types.is_datetime64_any_dtype(series):
                datetime_columns.append(column)
                continue
            if self._looks_like_datetime_column(series):
                inferred_datetime = pd.to_datetime(series, errors="coerce")
                if inferred_datetime.notna().any():
                    datetime_columns.append(column)
        numeric_columns = dataframe.select_dtypes(include=["number"]).columns.tolist()
        categorical_columns = [
            column
            for column in dataframe.columns
            if column not in numeric_columns and column not in datetime_columns
        ]
        return DatasetContext(
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            datetime_columns=datetime_columns,
        )

    def _looks_like_datetime_column(self, series: pd.Series) -> bool:
        """Heuristically identify datetime-like object columns."""
        if not pd.api.types.is_object_dtype(series):
            return False
        sample = series.dropna().astype(str).head(5)
        if sample.empty:
            return False
        keywords = ("date", "time", "day", "month", "year")
        return sample.str.contains(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}").any() or any(
            keyword in str(series.name).lower() for keyword in keywords
        )

    def _task_text(self, task: AnalysisTask) -> str:
        """Merge task fields into a normalized text blob."""
        content = " ".join([task.task_name, task.reasoning, task.expected_output])
        return content.lower()

    def _skipped_result(self, task: AnalysisTask, reason: str) -> ExecutionResult:
        """Return a skipped execution result."""
        return ExecutionResult(
            task_name=task.task_name,
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
