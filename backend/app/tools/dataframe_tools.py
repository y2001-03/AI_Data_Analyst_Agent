"""Deterministic dataframe tools used by the execution layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from app.schemas.file import AnalysisTask, ExecutionChart, ExecutionResult


@dataclass
class DatasetContext:
    """Convenience container for dataframe metadata."""

    numeric_columns: list[str]
    categorical_columns: list[str]
    datetime_columns: list[str]


class DataframeTool(Protocol):
    """Protocol for dataframe-backed execution tools."""

    name: str

    def run(
        self,
        dataframe: pd.DataFrame,
        task: AnalysisTask,
        context: DatasetContext,
    ) -> ExecutionResult:
        """Execute a task against the provided dataframe."""


class BaseDataframeTool:
    """Shared utilities for dataframe execution tools."""

    name = "base_tool"

    def _task_text(self, task: AnalysisTask) -> str:
        content = " ".join([task.task_name, task.reasoning, task.expected_output])
        return content.lower()

    def _match_column(self, description: str, columns: list[str]) -> str | None:
        lowered = description.lower()
        for column in columns:
            if column.lower() in lowered:
                return column
        return columns[0] if columns else None

    def _pick_metric_column(self, task: AnalysisTask, context: DatasetContext) -> str | None:
        return self._match_column(self._task_text(task), context.numeric_columns)

    def _skipped_result(self, task: AnalysisTask, reason: str) -> ExecutionResult:
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

    def _build_chart(
        self,
        chart_type: str,
        labels: list[object],
        values: list[object],
    ) -> ExecutionChart | None:
        if not labels or not values:
            return None
        normalized_values: list[float | int] = []
        normalized_labels: list[str] = []
        for label, value in zip(labels, values, strict=False):
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            normalized_labels.append(str(label))
            normalized_values.append(int(numeric_value) if numeric_value.is_integer() else numeric_value)
        if not normalized_labels or not normalized_values:
            return None
        return ExecutionChart(
            chart_type=chart_type,
            x=normalized_labels,
            y=normalized_values,
        )


class GroupByTool(BaseDataframeTool):
    """Execute categorical segmented aggregations."""

    name = "groupby_tool"

    def run(
        self,
        dataframe: pd.DataFrame,
        task: AnalysisTask,
        context: DatasetContext,
    ) -> ExecutionResult:
        group_column = self._match_column(self._task_text(task), context.categorical_columns)
        metric_column = self._pick_metric_column(task, context)
        aggregation = self._pick_aggregation(task)
        if not group_column:
            return self._skipped_result(task, "No categorical column is available for groupby.")
        if aggregation != "count" and not metric_column:
            return self._skipped_result(task, "No numeric column is available for aggregation.")
        grouped = dataframe.groupby(group_column, dropna=False)
        if aggregation == "count":
            result = grouped.size().reset_index(name="count")
        else:
            result = grouped[metric_column].agg(aggregation).reset_index()
            result = result.rename(columns={metric_column: f"{metric_column}_{aggregation}"})
        result = result.sort_values(by=result.columns[-1], ascending=False)
        rows = result.head(20).to_dict(orient="records")
        return ExecutionResult(
            task_name=task.task_name,
            type="groupby",
            data={
                "rows": rows,
                "chart_type": "bar",
                "labels": [row[result.columns[0]] for row in rows],
                "values": [row[result.columns[-1]] for row in rows],
            },
            chart=self._build_chart(
                "bar",
                [row[result.columns[0]] for row in rows],
                [row[result.columns[-1]] for row in rows],
            ),
        )

    def _pick_aggregation(self, task: AnalysisTask) -> str:
        description = self._task_text(task)
        if "mean" in description or "average" in description:
            return "mean"
        if "count" in description or "number" in description:
            return "count"
        return "sum"


class StatsTool(BaseDataframeTool):
    """Execute descriptive statistics and ranked record views."""

    name = "stats_tool"

    def run(
        self,
        dataframe: pd.DataFrame,
        task: AnalysisTask,
        context: DatasetContext,
    ) -> ExecutionResult:
        if self._is_sort_task(task) and context.numeric_columns:
            return self._run_sort(dataframe, task, context)
        if not context.numeric_columns:
            return self._skipped_result(task, "No numeric columns are available for descriptive statistics.")
        stats = dataframe[context.numeric_columns].describe().reset_index()
        stats = stats.rename(columns={"index": "stat"})
        rows = stats.to_dict(orient="records")
        mean_row = next((row for row in rows if row.get("stat") == "mean"), rows[0] if rows else None)
        labels = [column for column in context.numeric_columns if mean_row and column in mean_row]
        values = [mean_row[column] for column in labels] if mean_row else []
        return ExecutionResult(
            task_name=task.task_name,
            type="stats",
            data={
                "rows": rows,
                "chart_type": "bar",
                "labels": labels,
                "values": values,
            },
            chart=self._build_chart("bar", labels, values),
        )

    def _run_sort(
        self,
        dataframe: pd.DataFrame,
        task: AnalysisTask,
        context: DatasetContext,
    ) -> ExecutionResult:
        metric_column = self._pick_metric_column(task, context)
        if not metric_column:
            return self._skipped_result(task, "No numeric column is available for sorting.")
        task_text = self._task_text(task)
        ascending = "asc" in task_text and "desc" not in task_text
        sorted_frame = dataframe.sort_values(by=metric_column, ascending=ascending)
        rows = sorted_frame.head(20).to_dict(orient="records")
        label_column = next((column for column in dataframe.columns if column != metric_column), dataframe.columns[0])
        return ExecutionResult(
            task_name=task.task_name,
            type="stats",
            data={
                "rows": rows,
                "chart_type": "bar",
                "labels": [row.get(label_column, index + 1) for index, row in enumerate(rows)],
                "values": [row.get(metric_column, 0) for row in rows],
            },
            chart=self._build_chart(
                "bar",
                [row.get(label_column, index + 1) for index, row in enumerate(rows)],
                [row.get(metric_column, 0) for row in rows],
            ),
        )

    def _is_sort_task(self, task: AnalysisTask) -> bool:
        keywords = ("top", "highest", "lowest", "sort", "rank")
        return any(keyword in self._task_text(task) for keyword in keywords)


class TrendTool(BaseDataframeTool):
    """Execute time-series style aggregations."""

    name = "trend_tool"

    def run(
        self,
        dataframe: pd.DataFrame,
        task: AnalysisTask,
        context: DatasetContext,
    ) -> ExecutionResult:
        time_column = self._match_column(self._task_text(task), context.datetime_columns)
        metric_column = self._pick_metric_column(task, context)
        aggregation = self._pick_aggregation(task)
        frequency = self._pick_frequency(task)
        if not time_column:
            return self._skipped_result(task, "No datetime column is available for time series analysis.")
        if aggregation != "count" and not metric_column:
            return self._skipped_result(task, "No numeric column is available for time series aggregation.")
        working = dataframe.dropna(subset=[time_column]).copy()
        working = working.set_index(time_column).sort_index()
        if aggregation == "count":
            result = working.resample(frequency).size().reset_index(name="count")
        else:
            result = working[[metric_column]].resample(frequency).agg(aggregation).reset_index()
            result = result.rename(columns={metric_column: f"{metric_column}_{aggregation}"})
        result[time_column] = result[time_column].astype(str)
        rows = result.to_dict(orient="records")
        return ExecutionResult(
            task_name=task.task_name,
            type="trend",
            data={
                "rows": rows,
                "chart_type": "line",
                "labels": [row[time_column] for row in rows],
                "values": [row[result.columns[-1]] for row in rows],
            },
            chart=self._build_chart(
                "line",
                [row[time_column] for row in rows],
                [row[result.columns[-1]] for row in rows],
            ),
        )

    def _pick_aggregation(self, task: AnalysisTask) -> str:
        description = self._task_text(task)
        if "mean" in description or "average" in description:
            return "mean"
        if "count" in description or "number" in description:
            return "count"
        return "sum"

    def _pick_frequency(self, task: AnalysisTask) -> str:
        description = self._task_text(task)
        if "month" in description or "monthly" in description:
            return "ME"
        if "week" in description or "weekly" in description:
            return "W"
        return "D"
