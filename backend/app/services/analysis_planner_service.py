"""Analysis planning service — delegates LLM calls to the configured provider.

.. note::

    This module **no longer contains any DashScope SDK, HTTP, or
    ``urllib`` code**.  All transport is handled by the ``LLMProvider``
    abstraction in ``app.core.llm``.
"""

from __future__ import annotations

import json
import re

from app.core.config import get_settings
from app.core.exceptions import AppException
from app.core.llm import ChatMessage, get_llm_provider
from app.schemas.file import AIAnalysisResult, AnalysisTask, DatasetUploadResponse


class AnalysisPlannerService:
    """Generate structured analysis tasks from dataset understanding.

    Owns the task-planning prompt template and task normalisation logic
    (business rules).  The actual LLM call is delegated to the configured
    ``LLMProvider`` so this class is completely vendor-agnostic.
    """

    request_timeout = 12
    max_retries = 2

    def __init__(self) -> None:
        self.settings = get_settings()
        self._provider = get_llm_provider()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan_tasks(
        self,
        file_info: DatasetUploadResponse,
        ai_analysis: AIAnalysisResult,
        question: str | None = None,
        memory_context: dict[str, object] | None = None,
    ) -> list[AnalysisTask]:
        """Generate analysis tasks from summary and suggestions.

        Falls back to ``_mock_tasks`` when no API key is configured or
        when the LLM call fails.
        """
        if not self.settings.dashscope_api_key:
            return self._mock_tasks(file_info, question)

        messages = [
            ChatMessage(role="system", content=self._system_prompt()),
            ChatMessage(role="user", content=self._build_prompt(
                file_info, ai_analysis, question, memory_context,
            )),
        ]
        try:
            result = self._provider.structured_output(messages)
            return self._parse_tasks_from_result(result, file_info)
        except Exception:
            return self._mock_tasks(file_info, question)

    # ------------------------------------------------------------------
    # Prompt templates (business logic)
    # ------------------------------------------------------------------

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
            "Available deterministic tools: "
            "anomaly_detection identifies candidate statistical anomalies without modifying or deleting data. "
            "Choose it only for explicit anomaly records, outliers, group-local anomalies, or time-series spikes "
            "and drops. Supported modes are global, groupwise, and time_series. Supported methods are iqr, zscore, "
            "robust_zscore, and rolling_zscore, subject to mode compatibility. Params may include columns, group_by, "
            "time_column, value_columns, threshold, iqr_multiplier, and window. It already returns bounded chart "
            "data, so do not add a separate chart task. Requests to clean or delete anomalies must still produce "
            "detection only; anomalies are not automatically data errors. "
            "distribution_analysis describes numeric, categorical, or group-comparison distributions without "
            "changing data or making causal claims. Choose it for distributions, quantiles, histograms, boxplot "
            "summaries, skewness, kurtosis, long-tail descriptions, category frequencies, entropy, rare categories, "
            "high cardinality, or descriptive distribution comparisons across groups. Supported modes are numeric, "
            "categorical, and group_comparison. Params may include columns, group_by, bins, top_k, quantiles, and "
            "allow_low_cardinality_numeric. It already returns bounded chart data, so do not add a separate chart "
            "task. Distribution labels are descriptive heuristics, not formal normality tests, anomaly conclusions, "
            "data-quality errors, or evidence of causation. "
            "correlation_analysis calculates a Pearson or Spearman correlation matrix for numeric fields, "
            "pair sample sizes, reliable strong relationships, and heuristic multicollinearity risks. Choose it "
            "for correlation, association, Pearson, Spearman, or multicollinearity requests. Its params may contain "
            "columns, method (pearson or spearman), and missing_strategy (pairwise or listwise). Default to Pearson "
            "and pairwise when unspecified. The task already returns heatmap data, so do not add a separate chart "
            "task. Do not use causal wording: correlation does not imply causation. "
            "data_quality_analysis checks dataset reliability before formal analysis, including row/column counts, "
            "missing values, duplicate rows, uniqueness, constant columns, possible ID columns, IQR numeric outlier "
            "counts, quality issues, and a generic heuristic quality score; choose it when users ask to check data "
            "quality, whether data has problems, missing values, duplicate data, poor-quality fields, or to inspect "
            "the data before analysis. "
            "data_cleaning_plan proposes explainable actions without changing data; choose it when users ask how "
            "the data should be cleaned or explicitly request a cleaning plan. "
            "data_cleaning_execute applies only explicitly requested actions to a dataframe copy; choose it only "
            "when the user uses a clear action verb such as remove, fill, trim, convert, apply, or execute. "
            "For data_cleaning_execute, params.actions must list explicit actions using action_type, optional column, "
            "and parameters. Supported action_type values are drop_duplicates, fill_missing, drop_missing_rows, "
            "trim_strings, and convert_dtype. Never turn a question about missing or duplicate data into execution. "
            "stats summarizes numeric distributions or ranks records. "
            "groupby compares metrics across categorical segments. "
            "trend analyzes time/date based changes. "
            "When SQL is the most natural expression, you may output a task with type sql. "
            "For SQL tasks, use table name dataset and only generate safe SELECT queries. "
            "Output JSON only. "
            "Return {\"tasks\": [...]} where each task includes: "
            "task_name, type, reason, params."
        )

    # ------------------------------------------------------------------
    # Response parsing (business logic)
    # ------------------------------------------------------------------

    def _parse_tasks_from_result(
        self,
        result: dict[str, object],
        file_info: DatasetUploadResponse,
    ) -> list[AnalysisTask]:
        """Extract and normalise analysis tasks from the structured LLM output.

        The provider has already parsed the JSON; we validate the expected
        shape and convert each entry to a domain model.
        """
        try:
            tasks = result["tasks"]
        except KeyError as exc:
            raise AppException(
                "Analysis planner returned a response without a 'tasks' key.", 502
            ) from exc
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
        if task_type == "anomaly_detection":
            params = dict(params)
            mode = str(params.get("mode", "global")).lower()
            params["mode"] = mode
            params.setdefault(
                "method",
                "rolling_zscore" if mode == "time_series" else "iqr",
            )
            params["detection_only"] = True
        if task_type == "distribution_analysis":
            params = dict(params)
            if "mode" not in params:
                params["mode"] = (
                    "numeric"
                    if any(self._profile_is_numeric(profile) for profile in file_info.columns)
                    else "categorical"
                )
        if task_type == "correlation_analysis":
            params = dict(params)
            params.setdefault("method", "pearson")
            params.setdefault("missing_strategy", "pairwise")
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
        if task_type == "anomaly_detection":
            return (
                f"Bounded, explainable candidate anomaly records and chart data using {columns_text}; "
                "detection only, with no automatic deletion or replacement."
            )
        if task_type == "distribution_analysis":
            return (
                f"Descriptive distribution statistics and bounded chart data using {columns_text}; "
                "heuristic labels only, with no formal normality test, data modification, or causal claims."
            )
        if task_type == "correlation_analysis":
            return (
                f"Correlation matrix, pair sample sizes, relationship strengths, heuristic "
                f"multicollinearity risks, and heatmap data using {columns_text}; correlation does not imply causation."
            )
        if task_type == "data_quality_analysis":
            return (
                "Structured data quality report with missing values, duplicate rows, uniqueness, "
                "constant columns, possible ID columns, IQR outlier counts, issues, and quality score."
            )
        if task_type == "data_cleaning_plan":
            return "Explainable cleaning plan with affected-row estimates, risk levels, and no data changes."
        if task_type == "data_cleaning_execute":
            return (
                "Cleaning execution summary with per-action status, audit log, and JSON-safe preview; "
                "the original dataset remains unchanged."
            )
        if task_type == "chart":
            return f"Visualization output using {columns_text}."
        return f"Summary statistics using {columns_text}."

    # ------------------------------------------------------------------
    # Fallback / mock (business logic)
    # ------------------------------------------------------------------

    def _mock_tasks(
        self,
        file_info: DatasetUploadResponse,
        question: str | None = None,
    ) -> list[AnalysisTask]:
        """Return deterministic mock tasks when no API key is configured."""
        primary_column = file_info.columns[0].name if file_info.columns else "primary field"
        lowered_question = (question or "").lower()
        if question:
            quality_priority_tokens = (
                "data quality", "quality overview", "数据质量", "整体质量", "质量概览",
            )
            anomaly_tokens = (
                "anomaly detection", "detect anomaly", "find anomaly", "find outlier", "outlier detection",
                "outlier", "abnormal record", "unusual record", "spike", "sudden drop",
                "异常检测", "异常值", "异常点", "异常数据", "异常记录", "不正常记录", "哪些记录不正常",
                "数据异常", "robust z-score", "robust_zscore", "rolling z-score", "rolling_zscore",
                "突增", "突降", "突然变化",
            )
            correlation_tokens = (
                "correlation", "correlate", "association", "relationship", "pearson", "spearman",
                "multicollinearity", "相关", "关联", "相关系数", "共线性", "的关系", "受什么影响",
                "影响因素",
            )
            distribution_tokens = (
                "distribution", "frequency", "histogram", "boxplot", "box plot", "quantile",
                "percentile", "skewness", "kurtosis", "long tail", "entropy", "cardinality",
                "分布", "频数", "频率", "分位数", "百分位", "偏度", "峰度", "直方图",
                "箱线图", "长尾", "零值很多", "类别占比", "哪个渠道最多", "高基数",
                "稀有类别", "很少见", "信息熵",
            )
            cleaning_plan_tokens = (
                "cleaning plan", "cleaning advice", "how to clean", "should clean", "should i",
                "what should", "can i",
                "清洗方案", "清洗建议", "怎么清洗", "如何清洗", "应该怎么清洗", "先给我方案",
                "如何处理", "怎么处理", "应该如何处理", "要不要", "是否应该", "该不该", "建议",
            )
            cleaning_execute_tokens = (
                "remove ", "delete ", "drop ", "fill ", "trim ", "convert ", "execute ", "apply ",
                "clean the data", "clean this data",
                "删除", "填充", "清理", "转换", "执行", "应用", "去重",
            )
            quality_tokens = (
                "data quality", "quality", "missing", "duplicate", "duplicates",
                "reliable", "reliability", "outlier", "constant column", "id column",
                "数据质量", "缺失", "重复", "异常值", "字段质量", "有没有问题", "是否可靠", "分析前",
            )
            trend_tokens = (
                "trend", "monthly", "weekly", "daily", "date", "time",
                "趋势", "趋势图", "时间", "日期", "按天", "按周", "按月", "变化",
            )
            group_tokens = (
                "by", "segment", "category", "product", "region", "group",
                "分组", "按", "产品", "类别", "地区", "对比", "统计",
            )
            if any(token in lowered_question for token in quality_priority_tokens):
                return [
                    AnalysisTask(
                        task_name="Question-Focused Data Quality Analysis",
                        reasoning="Check overall dataset reliability without locating individual anomaly records.",
                        expected_output=(
                            "Structured data quality report covering missing values, duplicates, "
                            "uniqueness, constant columns, possible ID columns, outliers, issues, and score."
                        ),
                        type="data_quality_analysis",
                        params={},
                    ),
                ]
            if self._has_anomaly_intent(lowered_question, anomaly_tokens):
                params = self._fallback_anomaly_params(file_info, lowered_question)
                return [
                    AnalysisTask(
                        task_name="Question-Focused Anomaly Detection",
                        reasoning=(
                            "Identify explainable candidate statistical anomalies without changing or "
                            "deleting source data."
                        ),
                        expected_output=(
                            "Bounded anomaly records, thresholds, reliability flags, warnings, and chart data; "
                            "detection only, with no automatic deletion or replacement."
                        ),
                        type="anomaly_detection",
                        params=params,
                    ),
                ]
            if any(token in lowered_question for token in correlation_tokens):
                params = self._fallback_correlation_params(file_info, lowered_question)
                return [
                    AnalysisTask(
                        task_name="Question-Focused Correlation Analysis",
                        reasoning=(
                            "Analyze statistical associations among the requested numeric fields without "
                            "making causal claims."
                        ),
                        expected_output=(
                            "Correlation matrix, pair sample sizes, relationship strengths, heuristic "
                            "multicollinearity risks, and heatmap data; correlation does not imply causation."
                        ),
                        type="correlation_analysis",
                        params=params,
                    ),
                ]
            if any(token in lowered_question for token in distribution_tokens):
                params = self._fallback_distribution_params(file_info, lowered_question)
                return [
                    AnalysisTask(
                        task_name="Question-Focused Distribution Analysis",
                        reasoning=(
                            "Describe the requested distribution with bounded statistics and heuristic labels "
                            "without changing source data or making causal claims."
                        ),
                        expected_output=(
                            "Descriptive numeric, categorical, or grouped distribution statistics and chart data; "
                            "no formal normality test, automatic cleaning, or causal conclusion."
                        ),
                        type="distribution_analysis",
                        params=params,
                    ),
                ]
            if any(token in lowered_question for token in cleaning_plan_tokens):
                return [
                    AnalysisTask(
                        task_name="Question-Focused Data Cleaning Plan",
                        reasoning=f"Propose safe cleaning actions for the user's request: {question}",
                        expected_output=(
                            "Explainable cleaning plan with affected-row estimates and risk levels; "
                            "no changes to the original dataset."
                        ),
                        type="data_cleaning_plan",
                        params={"mode": "plan"},
                    ),
                ]
            if any(token in lowered_question for token in cleaning_execute_tokens):
                return [
                    AnalysisTask(
                        task_name="Question-Focused Data Cleaning Execution",
                        reasoning=f"Execute only the cleaning actions explicitly requested: {question}",
                        expected_output=(
                            "Cleaning execution summary with action status, audit log, and preview."
                        ),
                        type="data_cleaning_execute",
                        params={
                            "mode": "execute",
                            "actions": self._fallback_cleaning_actions(file_info, lowered_question),
                        },
                    ),
                ]
            if any(token in lowered_question for token in quality_tokens):
                return [
                    AnalysisTask(
                        task_name="Question-Focused Data Quality Analysis",
                        reasoning=f"Check dataset reliability for the user's question: {question}",
                        expected_output=(
                            "Structured data quality report covering missing values, duplicates, "
                            "uniqueness, constant columns, possible ID columns, outliers, issues, and score."
                        ),
                        type="data_quality_analysis",
                        params={},
                    ),
                ]
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
                expected_output=f"Data quality report covering missing values and uniqueness for {primary_column}.",
                type="data_quality_analysis",
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

    def _fallback_anomaly_params(
        self,
        file_info: DatasetUploadResponse,
        question: str,
    ) -> dict[str, object]:
        """Extract deterministic anomaly detection options from an explicit request."""
        time_tokens = (
            "time series", "rolling", "spike", "sudden drop", "时间序列", "滚动", "突增", "突降",
            "突然变化", "哪些日期",
        )
        group_tokens = ("groupwise", " by ", "按", "分组", "哪些地区")
        if any(token in question for token in time_tokens):
            mode = "time_series"
        elif any(token in question for token in group_tokens):
            mode = "groupwise"
        else:
            mode = "global"

        if any(token in question for token in ("rolling_zscore", "rolling z-score", "滚动窗口", "滚动 z")):
            method = "rolling_zscore"
        elif any(token in question for token in ("robust_zscore", "robust z-score", "robust zscore", "稳健 z")):
            method = "robust_zscore"
        elif any(token in question for token in ("zscore", "z-score", "z score", "标准分")):
            method = "zscore"
        elif "iqr" in question or "四分位" in question:
            method = "iqr"
        else:
            method = "rolling_zscore" if mode == "time_series" else "iqr"

        mentioned = [
            profile.name
            for profile in file_info.columns
            if self._question_mentions_column(question, profile.name)
        ]
        numeric = [
            profile.name for profile in file_info.columns if self._profile_is_numeric(profile)
        ]
        params: dict[str, object] = {
            "mode": mode,
            "method": method,
            "detection_only": True,
            "requested_deletion": any(
                token in question for token in ("delete", "remove", "drop", "clean", "删除", "清理", "清洗")
            ),
        }

        if mode == "groupwise":
            group_by = self._fallback_group_by(file_info, question, mentioned)
            if group_by:
                params["group_by"] = group_by
            value_columns = [column for column in mentioned if column in numeric and column != group_by]
            if value_columns:
                params["columns"] = value_columns
        elif mode == "time_series":
            time_column = self._fallback_time_column(file_info, mentioned)
            if time_column:
                params["time_column"] = time_column
            value_columns = [column for column in mentioned if column in numeric and column != time_column]
            if not value_columns:
                value_columns = [column for column in numeric if column != time_column]
            if value_columns:
                params["value_columns"] = value_columns
        else:
            value_columns = [column for column in mentioned if column in numeric]
            if value_columns:
                params["columns"] = value_columns

        threshold = self._number_after_keywords(question, ("threshold", "阈值"))
        if threshold is not None and method != "iqr":
            params["threshold"] = threshold
        multiplier = self._number_after_keywords(question, ("multiplier", "倍数"))
        if multiplier is not None and method == "iqr":
            params["iqr_multiplier"] = multiplier
        window = self._window_from_question(question)
        if window is not None and method == "rolling_zscore":
            params["window"] = window
        return params

    def _has_anomaly_intent(
        self,
        question: str,
        tokens: tuple[str, ...],
    ) -> bool:
        if any(token in question for token in tokens):
            return True
        return re.search(r"(?:检查|找出|检测|识别).{0,24}异常", question) is not None

    def _fallback_group_by(
        self,
        file_info: DatasetUploadResponse,
        question: str,
        mentioned: list[str],
    ) -> str | None:
        for profile in file_info.columns:
            name = profile.name.lower()
            if re.search(rf"(?:\bby\b|按)\s*{re.escape(name)}", question):
                return profile.name
        aliases = {
            "地区": ("region", "area", "location"),
            "区域": ("region", "area", "location"),
            "类别": ("category", "type"),
            "产品": ("product", "product_name"),
            "渠道": ("channel",),
        }
        for token, names in aliases.items():
            if token in question:
                match = next(
                    (profile.name for profile in file_info.columns if profile.name.lower() in names),
                    None,
                )
                if match:
                    return match
        return next(
            (
                column
                for column in mentioned
                if not self._profile_is_numeric(
                    next(profile for profile in file_info.columns if profile.name == column)
                )
            ),
            None,
        )

    def _fallback_time_column(
        self,
        file_info: DatasetUploadResponse,
        mentioned: list[str],
    ) -> str | None:
        mentioned_time = next(
            (
                profile.name
                for profile in file_info.columns
                if profile.name in mentioned and self._profile_is_datetime(profile)
            ),
            None,
        )
        if mentioned_time:
            return mentioned_time
        return next(
            (profile.name for profile in file_info.columns if self._profile_is_datetime(profile)),
            None,
        )

    def _profile_is_numeric(self, profile) -> bool:
        data_type = profile.data_type.lower()
        excluded = ("bool", "date", "time", "object", "string", "category")
        if any(token in data_type for token in excluded):
            return False
        return any(token in data_type for token in ("int", "float", "double", "decimal", "number"))

    def _profile_is_datetime(self, profile) -> bool:
        data_type = profile.data_type.lower()
        name = profile.name.lower()
        return any(token in data_type for token in ("date", "time", "timestamp")) or any(
            token in name for token in ("date", "time", "日期", "时间")
        )

    def _number_after_keywords(
        self,
        question: str,
        keywords: tuple[str, ...],
    ) -> float | None:
        keyword_pattern = "|".join(re.escape(keyword) for keyword in keywords)
        match = re.search(
            rf"(?:{keyword_pattern})\s*(?:=|:|为)?\s*(\d+(?:\.\d+)?)",
            question,
        )
        return float(match.group(1)) if match else None

    def _window_from_question(self, question: str) -> int | None:
        match = re.search(r"(?:window|窗口)\s*(?:=|:|为)?\s*(\d+)", question)
        if not match:
            match = re.search(r"(\d+)\s*(?:day|days|天)?\s*滚动", question)
        return int(match.group(1)) if match else None

    def _fallback_distribution_params(
        self,
        file_info: DatasetUploadResponse,
        question: str,
    ) -> dict[str, object]:
        """Extract deterministic descriptive distribution options from a request."""
        mentioned = [
            profile.name
            for profile in file_info.columns
            if self._question_mentions_column(question, profile.name)
        ]
        numeric = [
            profile.name for profile in file_info.columns if self._profile_is_numeric(profile)
        ]
        group_by = self._fallback_group_by(file_info, question, mentioned)
        comparison_tokens = (
            "compare", "comparison", "difference", "different groups", "across groups",
            "比较", "对比", "差异", "不同组", "各组", "不同地区", "不同渠道",
        )
        categorical_tokens = (
            "frequency", "category", "categories", "most common", "entropy", "cardinality",
            "频数", "频率", "类别", "占比", "哪个渠道最多", "高基数", "稀有类别",
            "很少见", "信息熵", "各地区数量",
        )
        mentioned_numeric = [column for column in mentioned if column in numeric]
        if (
            group_by
            and mentioned_numeric
            and any(token in question for token in comparison_tokens)
        ):
            mode = "group_comparison"
        elif any(token in question for token in categorical_tokens) or (
            group_by and not mentioned_numeric and "分布" in question
        ):
            mode = "categorical"
        else:
            mode = "numeric" if numeric else "categorical"

        params: dict[str, object] = {"mode": mode}
        if mode == "group_comparison":
            params["group_by"] = group_by
            columns = [column for column in mentioned_numeric if column != group_by]
            if columns:
                params["columns"] = columns
        elif mode == "categorical":
            columns = [column for column in mentioned if column not in numeric]
            if not columns and group_by and group_by not in numeric:
                columns = [group_by]
            if columns:
                params["columns"] = columns
            params["allow_low_cardinality_numeric"] = any(
                token in question
                for token in ("low cardinality numeric", "低基数数值", "整数类别")
            )
        elif mentioned_numeric:
            params["columns"] = mentioned_numeric

        bins = self._integer_after_keywords(question, ("bins", "bin", "直方图分箱", "分箱"))
        if bins is not None and mode == "numeric":
            params["bins"] = bins
        top_k = self._integer_after_keywords(question, ("top_k", "top k", "top", "前"))
        if top_k is not None and mode == "categorical":
            params["top_k"] = top_k
        quantiles = self._quantiles_from_question(question)
        if quantiles and mode == "numeric":
            params["quantiles"] = quantiles
        return params

    def _integer_after_keywords(
        self,
        question: str,
        keywords: tuple[str, ...],
    ) -> int | None:
        keyword_pattern = "|".join(re.escape(keyword) for keyword in keywords)
        match = re.search(
            rf"(?:{keyword_pattern})\s*(?:=|:|为)?\s*(\d+)",
            question,
        )
        return int(match.group(1)) if match else None

    def _quantiles_from_question(self, question: str) -> list[float] | None:
        match = re.search(
            r"(?:quantiles?|percentiles?|分位数|百分位)\s*(?:=|:|为)?\s*([\[\]0-9.,%\s]+)",
            question,
        )
        if not match:
            return None
        values: list[float] = []
        for number, percent in re.findall(r"(\d+(?:\.\d+)?)\s*(%)?", match.group(1)):
            value = float(number)
            if percent:
                value /= 100
            values.append(value)
        return values or None

    def _fallback_correlation_params(
        self,
        file_info: DatasetUploadResponse,
        question: str,
    ) -> dict[str, object]:
        """Extract deterministic correlation options from an explicit user question."""
        method = "spearman" if any(
            token in question for token in ("spearman", "排名相关", "秩相关", "单调关系")
        ) else "pearson"
        missing_strategy = "listwise" if any(
            token in question for token in ("listwise", "完整案例", "整行删除")
        ) else "pairwise"
        columns = [
            profile.name
            for profile in file_info.columns
            if self._question_mentions_column(question, profile.name)
        ]
        params: dict[str, object] = {
            "method": method,
            "missing_strategy": missing_strategy,
        }
        if columns:
            params["columns"] = columns
        return params

    def _question_mentions_column(self, question: str, column: str) -> bool:
        """Match ASCII column names on identifier boundaries and other names literally."""
        normalized = column.lower()
        if re.fullmatch(r"[a-z0-9_]+", normalized):
            pattern = rf"(?<![a-z0-9_]){re.escape(normalized)}(?![a-z0-9_])"
            return re.search(pattern, question) is not None
        return normalized in question

    def _fallback_cleaning_actions(
        self,
        file_info: DatasetUploadResponse,
        question: str,
    ) -> list[dict[str, object]]:
        """Extract only unambiguous cleaning actions for deterministic fallback."""
        actions: list[dict[str, object]] = []
        column = next(
            (profile.name for profile in file_info.columns if profile.name.lower() in question),
            None,
        )
        if any(token in question for token in ("duplicate", "duplicates", "重复", "去重")) and any(
            token in question for token in ("remove", "delete", "drop", "删除", "去重")
        ):
            actions.append(
                {
                    "action_type": "drop_duplicates",
                    "parameters": {"keep": "first"},
                }
            )

        fill_strategy = None
        if any(token in question for token in ("median", "中位数")):
            fill_strategy = "median"
        elif any(token in question for token in ("mean", "average", "均值", "平均值")):
            fill_strategy = "mean"
        elif any(token in question for token in ("mode", "众数")):
            fill_strategy = "mode"
        if column and fill_strategy and any(token in question for token in ("fill", "填充")):
            actions.append(
                {
                    "action_type": "fill_missing",
                    "column": column,
                    "parameters": {"strategy": fill_strategy},
                }
            )

        if any(token in question for token in ("trim", "whitespace", "空格")) and any(
            token in question for token in ("trim", "remove", "clean", "清理", "删除")
        ):
            action: dict[str, object] = {
                "action_type": "trim_strings",
                "parameters": {"convert_blank_to_null": True},
            }
            if column:
                action["column"] = column
            actions.append(action)

        target_type = None
        if any(token in question for token in ("datetime", "date", "日期", "时间")):
            target_type = "datetime"
        elif any(token in question for token in ("numeric", "number", "数值", "数字")):
            target_type = "numeric"
        elif any(token in question for token in ("boolean", "bool", "布尔")):
            target_type = "boolean"
        if column and target_type and any(token in question for token in ("convert", "转换")):
            actions.append(
                {
                    "action_type": "convert_dtype",
                    "column": column,
                    "parameters": {"target_type": target_type},
                }
            )
        return actions
