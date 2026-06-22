"""Analysis planning service."""

from __future__ import annotations

import json
from socket import timeout as SocketTimeout
from urllib import error, request

from app.core.config import get_settings
from app.core.exceptions import AppException
from app.schemas.file import AIAnalysisResult, AnalysisTask, DatasetUploadResponse


class AnalysisPlannerService:
    """Generate structured analysis tasks from dataset understanding."""

    endpoint = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    request_timeout = 12
    max_retries = 2

    def __init__(self) -> None:
        self.settings = get_settings()

    def plan_tasks(
        self,
        file_info: DatasetUploadResponse,
        ai_analysis: AIAnalysisResult,
        question: str | None = None,
        memory_context: dict[str, object] | None = None,
    ) -> list[AnalysisTask]:
        """Generate analysis tasks from summary and suggestions."""
        if not self.settings.dashscope_api_key:
            return self._mock_tasks(file_info, question)
        payload = self._build_payload(file_info, ai_analysis, question, memory_context)
        try:
            body = self._send_with_retry(payload)
            return self._parse_tasks(body, file_info)
        except Exception:
            return self._mock_tasks(file_info, question)

    def _build_payload(
        self,
        file_info: DatasetUploadResponse,
        ai_analysis: AIAnalysisResult,
        question: str | None,
        memory_context: dict[str, object] | None,
    ) -> dict[str, object]:
        """Build the planner request payload."""
        prompt = self._build_prompt(file_info, ai_analysis, question, memory_context)
        return {
            "model": self.settings.llm_model,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": prompt},
            ],
        }

    def _build_prompt(
        self,
        file_info: DatasetUploadResponse,
        ai_analysis: AIAnalysisResult,
        question: str | None,
        memory_context: dict[str, object] | None,
    ) -> str:
        """Serialize planning input for the model."""
        payload = {
            "user_question": question,
            "memory_context": memory_context,
            "dataset_schema": [column.model_dump() for column in file_info.columns],
            "dataset_preview": file_info.preview,
            "dataset_summary": ai_analysis.summary,
            "suggested_analysis": ai_analysis.suggestions,
            "dataset_info": {
                "file_name": file_info.file_name,
                "file_type": file_info.file_type,
                "row_count": file_info.row_count,
                "column_count": file_info.column_count,
            },
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _send_with_retry(self, payload: dict[str, object]) -> str:
        """Send the planning request with retry support."""
        last_error: Exception | None = None
        for _ in range(self.max_retries + 1):
            try:
                return self._send_request(payload)
            except (error.HTTPError, error.URLError, SocketTimeout, TimeoutError) as exc:
                last_error = exc
            except Exception as exc:
                last_error = exc
        if last_error is None:
            raise RuntimeError("Analysis planner failed without an error.")
        raise last_error

    def _send_request(self, payload: dict[str, object]) -> str:
        """Send the planning request to DashScope."""
        req = request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=self._build_headers(),
            method="POST",
        )
        with request.urlopen(req, timeout=self.request_timeout) as response:
            return response.read().decode("utf-8")

    def _build_headers(self) -> dict[str, str]:
        """Build planner request headers."""
        return {
            "Authorization": f"Bearer {self.settings.dashscope_api_key}",
            "Content-Type": "application/json",
        }

    def _parse_tasks(
        self,
        body: str,
        file_info: DatasetUploadResponse,
    ) -> list[AnalysisTask]:
        """Parse tasks from the model response."""
        try:
            payload = json.loads(body)
            content = payload["choices"][0]["message"]["content"]
            result = json.loads(content)
            tasks = result["tasks"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise AppException("Analysis planner returned invalid tasks.", 502) from exc
        parsed = [
            self._normalize_task(task, file_info)
            for task in tasks
            if isinstance(task, dict)
        ]
        parsed = [task for task in parsed if task is not None]
        if not parsed:
            raise AppException("Analysis planner returned no usable tasks.", 502)
        return parsed

    def _normalize_task(
        self,
        task: dict[str, object],
        file_info: DatasetUploadResponse,
    ) -> AnalysisTask | None:
        """Normalize LLM planner output into the backward-compatible task schema."""
        task_name = task.get("task_name")
        task_type = task.get("type")
        reason = task.get("reason") or task.get("reasoning")
        params = task.get("params") if isinstance(task.get("params"), dict) else {}
        if not isinstance(task_name, str) or not isinstance(task_type, str) or not isinstance(reason, str):
            return None
        expected_output = self._build_expected_output(task_type, params, file_info)
        return AnalysisTask(
            task_name=task_name,
            reasoning=reason,
            expected_output=expected_output,
            type=task_type,
            params=params,
        )

    def _build_expected_output(
        self,
        task_type: str,
        params: dict[str, object],
        file_info: DatasetUploadResponse,
    ) -> str:
        """Synthesize an execution-friendly expected output string."""
        column_names = {column.name.lower(): column.name for column in file_info.columns}
        serialized_params = json.dumps(params, ensure_ascii=False).lower()
        mentioned_columns = [
            original for lowered, original in column_names.items() if lowered in serialized_params
        ]
        columns_text = ", ".join(mentioned_columns) if mentioned_columns else "relevant dataset columns"
        if task_type == "trend":
            return f"Trend chart over time using {columns_text}."
        if task_type == "groupby":
            return f"Grouped comparison using {columns_text}."
        if task_type == "sql":
            return "SQL query result against table dataset."
        if task_type == "chart":
            return f"Visualization output using {columns_text}."
        return f"Summary statistics using {columns_text}."

    def _system_prompt(self) -> str:
        """Return the system prompt for task planning."""
        return (
            "You are a data analysis planner. "
            "Given dataset schema + user question: "
            "Generate a minimal but complete set of analytical tasks. "
            "Rules: "
            "Prefer 1~3 tasks only. "
            "Must include chart task if trend or comparison exists. "
            "Must use dataset columns only. "
            "When SQL is the most natural expression, you may output a task with type sql. "
            "For SQL tasks, use table name dataset and only generate safe SELECT queries. "
            "Output JSON only. "
            "Return {\"tasks\": [...]} where each task includes: "
            "task_name, type, reason, params."
        )

    def _mock_tasks(
        self,
        file_info: DatasetUploadResponse,
        question: str | None = None,
    ) -> list[AnalysisTask]:
        """Return deterministic mock tasks when no API key is configured."""
        primary_column = file_info.columns[0].name if file_info.columns else "primary field"
        lowered_question = (question or "").lower()
        if question:
            trend_tokens = (
                "trend",
                "monthly",
                "weekly",
                "daily",
                "date",
                "time",
                "趋势",
                "趋势图",
                "时间",
                "日期",
                "按天",
                "按周",
                "按月",
                "变化",
            )
            group_tokens = (
                "by",
                "segment",
                "category",
                "product",
                "region",
                "group",
                "分组",
                "按",
                "产品",
                "类别",
                "地区",
                "对比",
                "统计",
            )
            if any(token in lowered_question for token in trend_tokens):
                return [
                    AnalysisTask(
                        task_name="Question-Focused Trend Analysis",
                        reasoning=f"Answer the user's question: {question}",
                        expected_output="A time-based trend for the most relevant metric.",
                        type="trend",
                        params={},
                    ),
                    AnalysisTask(
                        task_name="Question-Focused Chart Task",
                        reasoning="Provide a chart for the detected trend request.",
                        expected_output="A chart of the relevant trend output.",
                        type="chart",
                        params={},
                    ),
                ]
            if any(token in lowered_question for token in group_tokens):
                return [
                    AnalysisTask(
                        task_name="Question-Focused Statistics",
                        reasoning=f"Answer the user's question: {question}",
                        expected_output="Summary statistics for the relevant metric.",
                        type="stats",
                        params={},
                    ),
                    AnalysisTask(
                        task_name="Question-Focused Group Comparison",
                        reasoning=f"Answer the user's question: {question}",
                        expected_output="A grouped comparison for the most relevant category and metric.",
                        type="groupby",
                        params={},
                    ),
                    AnalysisTask(
                        task_name="Question-Focused Chart Task",
                        reasoning="Provide a chart for the detected comparison request.",
                        expected_output="A chart of the grouped comparison output.",
                        type="chart",
                        params={},
                    ),
                ]
            return [
                AnalysisTask(
                    task_name="Question-Focused Statistics",
                    reasoning=f"Answer the user's question: {question}",
                    expected_output="Summary statistics or ranked values for the most relevant metric.",
                    type="stats",
                    params={},
                ),
            ]
        return [
            AnalysisTask(
                task_name="Data Quality Review",
                reasoning="Confirm the dataset is complete and reliable before deeper analysis.",
                expected_output=f"Missing value and uniqueness summary for {primary_column}.",
                type="stats",
                params={"focus_column": primary_column},
            ),
            AnalysisTask(
                task_name="Descriptive Statistics",
                reasoning="Establish a baseline understanding of key fields and distributions.",
                expected_output="Summary statistics and distribution highlights for major columns.",
                type="stats",
                params={},
            ),
            AnalysisTask(
                task_name="Trend or Segment Analysis",
                reasoning="Explore whether values vary over time or across categories.",
                expected_output="A grouped comparison or trend view for the most relevant dimensions.",
                type="chart",
                params={},
            ),
        ]
