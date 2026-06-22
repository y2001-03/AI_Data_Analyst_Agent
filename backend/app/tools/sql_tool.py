"""DuckDB-backed SQL analysis tool."""

from __future__ import annotations

import re

import duckdb
import pandas as pd

from app.schemas.file import AnalysisTask, ExecutionResult
from app.tools.dataframe_tools import BaseDataframeTool, DatasetContext


class SQLTool(BaseDataframeTool):
    """Execute safe read-only SQL against a dataframe."""

    name = "sql_tool"
    _forbidden_sql = re.compile(r"\b(insert|update|delete|drop|alter|create|truncate|attach|copy)\b", re.I)

    def run(
        self,
        dataframe: pd.DataFrame,
        task: AnalysisTask,
        context: DatasetContext,
    ) -> ExecutionResult:
        del context
        query = self._extract_query(task)
        if not query:
            return self._skipped_result(task, "No SQL query was provided for this task.")
        if not self._is_safe_select(query):
            return self._skipped_result(task, "Only safe SELECT SQL queries are allowed.")
        connection = duckdb.connect(database=":memory:")
        try:
            connection.register("dataset", dataframe)
            limited_query = self._with_limit(query, 100)
            result = connection.execute(limited_query).fetchdf()
        finally:
            connection.close()
        rows = result.head(100).to_dict(orient="records")
        return ExecutionResult(
            task_name=task.task_name,
            type="sql",
            data={"rows": rows},
            chart=None,
        )

    def _extract_query(self, task: AnalysisTask) -> str | None:
        params = task.params or {}
        query = params.get("query")
        return query.strip() if isinstance(query, str) else None

    def _is_safe_select(self, query: str) -> bool:
        normalized = query.strip().rstrip(";")
        if not normalized.lower().startswith("select"):
            return False
        return self._forbidden_sql.search(normalized) is None

    def _with_limit(self, query: str, limit: int) -> str:
        normalized = query.strip().rstrip(";")
        if re.search(r"\blimit\b", normalized, re.I):
            return normalized
        return f"{normalized} LIMIT {limit}"
