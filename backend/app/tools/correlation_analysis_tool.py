"""Enterprise numeric correlation analysis tool."""

from __future__ import annotations

import math
import time
from datetime import date, datetime, timezone
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from app.core.exceptions import ToolExecutionException
from app.core.logging import get_logger
from app.schemas.file import AnalysisTask, ExecutionResult
from app.tools.dataframe_tools import BaseDataframeTool, DatasetContext

logger = get_logger(__name__)

STRONG_CORRELATION_THRESHOLD = 0.7
MODERATE_CORRELATION_THRESHOLD = 0.4
WEAK_CORRELATION_THRESHOLD = 0.2
MULTICOLLINEARITY_RISK_THRESHOLD = 0.85
MIN_RELIABLE_SAMPLE_SIZE = 10
MAX_CORRELATION_COLUMNS = 50
SUPPORTED_METHODS = {"pearson", "spearman"}
SUPPORTED_MISSING_STRATEGIES = {"pairwise", "listwise"}


class CorrelationAnalysisTool(BaseDataframeTool):
    """Analyze statistical associations between numeric dataframe columns."""

    name = "correlation_analysis_tool"
    tool_name = "correlation_analysis"
    result_type = "correlation_analysis"

    def run(
        self,
        dataframe: pd.DataFrame,
        task: AnalysisTask,
        context: DatasetContext,
    ) -> ExecutionResult:
        """Return a JSON-safe correlation report without mutating the dataframe."""
        del context
        started_at = time.monotonic()
        params = task.params or {}
        method = str(params.get("method", "pearson")).lower()
        missing_strategy = str(params.get("missing_strategy", "pairwise")).lower()
        analyzed_column_count = 0
        analyzed_pair_count = 0
        excluded_column_count = 0
        try:
            self._validate_options(method, missing_strategy)
            report = self._build_report(dataframe, params, method, missing_strategy)
            summary = report["summary"]
            analyzed_column_count = int(summary["analyzed_column_count"])
            analyzed_pair_count = int(summary["analyzed_pair_count"])
            excluded_column_count = int(summary["excluded_column_count"])
        except ToolExecutionException:
            self._log_execution(
                method,
                missing_strategy,
                analyzed_column_count,
                analyzed_pair_count,
                excluded_column_count,
                "failed",
                started_at,
            )
            raise
        except Exception as exc:
            self._log_execution(
                method,
                missing_strategy,
                analyzed_column_count,
                analyzed_pair_count,
                excluded_column_count,
                "failed",
                started_at,
            )
            raise ToolExecutionException(
                "Correlation analysis failed.",
                details={"tool_name": self.name},
            ) from exc

        self._log_execution(
            method,
            missing_strategy,
            analyzed_column_count,
            analyzed_pair_count,
            excluded_column_count,
            "success",
            started_at,
        )
        return ExecutionResult(
            task_name=task.task_name,
            type=self.result_type,
            data=self._json_safe(report),
            chart=None,
        )

    def _build_report(
        self,
        dataframe: pd.DataFrame,
        params: dict[str, object],
        method: str,
        missing_strategy: str,
    ) -> dict[str, object]:
        if dataframe.empty:
            raise self._input_error("Correlation analysis requires a non-empty dataframe.")

        requested_columns = self._requested_columns(params)
        candidates, excluded_columns = self._select_numeric_columns(
            dataframe,
            requested_columns,
        )
        original_numeric_column_count = len(candidates)
        constant_columns, all_null_columns, eligible_columns = self._partition_columns(
            dataframe,
            candidates,
        )
        excluded_columns.extend(all_null_columns)
        original_eligible_column_count = len(eligible_columns)
        warnings: list[str] = []

        if requested_columns is not None and len(eligible_columns) > MAX_CORRELATION_COLUMNS:
            raise self._input_error(
                f"Explicit correlation analysis supports at most {MAX_CORRELATION_COLUMNS} columns.",
                error_code="CORRELATION_COLUMN_LIMIT_EXCEEDED",
            )
        if requested_columns is None and len(eligible_columns) > MAX_CORRELATION_COLUMNS:
            selected, limited = self._limit_columns(dataframe, eligible_columns)
            eligible_columns = selected
            excluded_columns.extend(
                {"column": column, "reason": "column_limit"} for column in limited
            )
            warnings.append(
                f"{original_eligible_column_count} eligible numeric columns exceeded the limit of "
                f"{MAX_CORRELATION_COLUMNS}; columns were selected deterministically by non-null count, "
                "variance, and original order."
            )

        if len(eligible_columns) < 2:
            raise self._input_error(
                "Correlation analysis requires at least two non-constant numeric columns.",
                details={
                    "numeric_column_count": original_numeric_column_count,
                    "eligible_column_count": len(eligible_columns),
                    "constant_column_count": len(constant_columns),
                },
            )

        working = self._numeric_frame(dataframe, eligible_columns)
        if missing_strategy == "listwise":
            working = working.dropna(axis=0, how="any")
            if working.empty:
                raise self._input_error(
                    "Listwise deletion removed all rows available for correlation analysis.",
                    error_code="NO_LISTWISE_ROWS",
                )

        matrix, sample_size_matrix, pairs = self._calculate_pairs(
            working,
            eligible_columns,
            method,
            missing_strategy,
            warnings,
        )
        reliable_pairs = [
            pair
            for pair in pairs
            if pair["sufficient_sample"] and pair["coefficient"] is not None
        ]
        strong_relationships = [
            pair.copy() for pair in reliable_pairs if pair["strength"] == "strong"
        ]
        strong_positive = [
            pair.copy() for pair in strong_relationships if pair["direction"] == "positive"
        ]
        strong_negative = [
            pair.copy() for pair in strong_relationships if pair["direction"] == "negative"
        ]
        moderate_relationships = [
            pair.copy() for pair in reliable_pairs if pair["strength"] == "moderate"
        ]
        negligible_relationships = [
            pair.copy() for pair in reliable_pairs if pair["strength"] == "negligible"
        ]
        multicollinearity_risks = self._multicollinearity_risks(reliable_pairs)

        return {
            "tool_name": self.tool_name,
            "summary": {
                "method": method,
                "missing_strategy": missing_strategy,
                "analyzed_column_count": len(eligible_columns),
                "analyzed_pair_count": len(pairs),
                "strong_positive_count": len(strong_positive),
                "strong_negative_count": len(strong_negative),
                "moderate_count": len(moderate_relationships),
                "constant_column_count": len(constant_columns),
                "excluded_column_count": len(excluded_columns),
            },
            "columns": eligible_columns,
            "matrix": matrix,
            "sample_size_matrix": sample_size_matrix,
            "pairs": pairs,
            "strong_relationships": strong_relationships,
            "strong_positive_relationships": strong_positive,
            "strong_negative_relationships": strong_negative,
            "moderate_relationships": moderate_relationships,
            "negligible_relationships": negligible_relationships,
            "multicollinearity_risks": multicollinearity_risks,
            "constant_columns": constant_columns,
            "excluded_columns": excluded_columns,
            "warnings": warnings,
            "metadata": {
                "correlation_is_not_causation": True,
                "correlation_notice": "Correlation does not imply causation.",
                "correlation_notice_zh": (
                    "相关性只能表示变量之间的统计关联，不能直接证明一个变量导致另一个变量变化。"
                ),
                "thresholds": {
                    "strong": STRONG_CORRELATION_THRESHOLD,
                    "moderate": MODERATE_CORRELATION_THRESHOLD,
                    "weak": WEAK_CORRELATION_THRESHOLD,
                },
                "threshold_note": (
                    "These thresholds are generic interpretive guidance, not absolute standards for every "
                    "business context."
                ),
                "minimum_reliable_sample_size": MIN_RELIABLE_SAMPLE_SIZE,
                "multicollinearity_risk_threshold": MULTICOLLINEARITY_RISK_THRESHOLD,
                "multicollinearity_is_heuristic": True,
                "vif_calculated": False,
                "max_correlation_columns": MAX_CORRELATION_COLUMNS,
                "original_column_count": int(len(dataframe.columns)),
                "original_numeric_column_count": original_numeric_column_count,
                "original_eligible_column_count": original_eligible_column_count,
                "actual_analyzed_column_count": len(eligible_columns),
                "missing_strategy": missing_strategy,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "modifies_original": False,
            },
            "chart": {
                "type": "heatmap",
                "x": eligible_columns,
                "y": eligible_columns,
                "values": [
                    [matrix[row][column] for column in eligible_columns]
                    for row in eligible_columns
                ],
            },
        }

    def _validate_options(self, method: str, missing_strategy: str) -> None:
        if method not in SUPPORTED_METHODS:
            raise self._input_error(
                "Correlation method must be 'pearson' or 'spearman'.",
                error_code="INVALID_CORRELATION_METHOD",
            )
        if missing_strategy not in SUPPORTED_MISSING_STRATEGIES:
            raise self._input_error(
                "Missing strategy must be 'pairwise' or 'listwise'.",
                error_code="INVALID_MISSING_STRATEGY",
            )

    def _requested_columns(self, params: dict[str, object]) -> list[str] | None:
        raw_columns = params.get("columns")
        if raw_columns is None:
            return None
        if (
            not isinstance(raw_columns, list)
            or not raw_columns
            or not all(isinstance(column, str) and column for column in raw_columns)
        ):
            raise self._input_error(
                "Correlation columns must be a non-empty list of column names.",
                error_code="INVALID_CORRELATION_COLUMNS",
            )
        if len(set(raw_columns)) != len(raw_columns):
            raise self._input_error(
                "Correlation columns must not contain duplicates.",
                error_code="INVALID_CORRELATION_COLUMNS",
            )
        if len(raw_columns) > MAX_CORRELATION_COLUMNS:
            raise self._input_error(
                f"Explicit correlation analysis supports at most {MAX_CORRELATION_COLUMNS} columns.",
                error_code="CORRELATION_COLUMN_LIMIT_EXCEEDED",
            )
        return raw_columns

    def _select_numeric_columns(
        self,
        dataframe: pd.DataFrame,
        requested_columns: list[str] | None,
    ) -> tuple[list[str], list[dict[str, str]]]:
        if requested_columns is not None:
            unknown = [column for column in requested_columns if column not in dataframe.columns]
            if unknown:
                raise self._input_error(
                    f"Unknown correlation column(s): {', '.join(unknown)}.",
                    error_code="UNKNOWN_CORRELATION_COLUMNS",
                    details={"unknown_columns": unknown},
                )
            invalid = [
                {"column": column, "reason": self._exclusion_reason(dataframe[column])}
                for column in requested_columns
                if self._exclusion_reason(dataframe[column]) is not None
            ]
            if invalid:
                description = ", ".join(
                    f"{item['column']} ({item['reason']})" for item in invalid
                )
                raise self._input_error(
                    f"Explicit correlation columns must be numeric: {description}.",
                    error_code="NON_NUMERIC_CORRELATION_COLUMNS",
                    details={"invalid_columns": invalid},
                )
            return list(requested_columns), []

        numeric: list[str] = []
        excluded: list[dict[str, str]] = []
        for column in dataframe.columns:
            reason = self._exclusion_reason(dataframe[column])
            if reason is None:
                numeric.append(str(column))
            else:
                excluded.append({"column": str(column), "reason": reason})
        if not numeric:
            raise self._input_error("Correlation analysis found no numeric columns.")
        return numeric, excluded

    def _exclusion_reason(self, series: pd.Series) -> str | None:
        if pd.api.types.is_bool_dtype(series):
            return "boolean"
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        if pd.api.types.is_timedelta64_dtype(series):
            return "timedelta"
        if isinstance(series.dtype, pd.CategoricalDtype):
            return "category"
        if pd.api.types.is_string_dtype(series.dtype):
            return "string"
        if pd.api.types.is_object_dtype(series):
            return "object"
        if pd.api.types.is_complex_dtype(series):
            return "unsupported"
        if not pd.api.types.is_numeric_dtype(series):
            return "unsupported"
        return None

    def _partition_columns(
        self,
        dataframe: pd.DataFrame,
        columns: list[str],
    ) -> tuple[list[dict[str, str]], list[dict[str, str]], list[str]]:
        constant: list[dict[str, str]] = []
        all_null: list[dict[str, str]] = []
        eligible: list[str] = []
        for column in columns:
            finite = self._numeric_series(dataframe[column]).dropna()
            if finite.empty:
                all_null.append({"column": column, "reason": "all_null_or_non_finite"})
            elif int(finite.nunique(dropna=True)) <= 1:
                constant.append({"column": column, "reason": "constant"})
            else:
                eligible.append(column)
        return constant, all_null, eligible

    def _limit_columns(
        self,
        dataframe: pd.DataFrame,
        columns: list[str],
    ) -> tuple[list[str], list[str]]:
        original_position = {column: index for index, column in enumerate(columns)}

        def sort_key(column: str) -> tuple[float, float, int]:
            series = self._numeric_series(dataframe[column])
            non_null_count = int(series.notna().sum())
            variance = float(series.var()) if non_null_count >= 2 else float("-inf")
            if not math.isfinite(variance):
                variance = float("-inf")
            return (-non_null_count, -variance, original_position[column])

        ordered = sorted(columns, key=sort_key)
        selected_set = set(ordered[:MAX_CORRELATION_COLUMNS])
        selected = [column for column in columns if column in selected_set]
        limited = [column for column in columns if column not in selected_set]
        return selected, limited

    def _numeric_frame(self, dataframe: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            {column: self._numeric_series(dataframe[column]) for column in columns},
            index=dataframe.index,
        )

    def _numeric_series(self, series: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce").astype("float64")
        return numeric.replace([np.inf, -np.inf], np.nan)

    def _calculate_pairs(
        self,
        dataframe: pd.DataFrame,
        columns: list[str],
        method: str,
        missing_strategy: str,
        warnings: list[str],
    ) -> tuple[
        dict[str, dict[str, float | None]],
        dict[str, dict[str, int]],
        list[dict[str, object]],
    ]:
        matrix = {column: {other: None for other in columns} for column in columns}
        sample_sizes = {column: {other: 0 for other in columns} for column in columns}
        for column in columns:
            values = dataframe[column].dropna()
            sample_size = int(len(values))
            sample_sizes[column][column] = sample_size
            if sample_size >= 2 and int(values.nunique()) > 1:
                matrix[column][column] = 1.0

        pairs: list[dict[str, object]] = []
        for column_x, column_y in combinations(columns, 2):
            pair_frame = dataframe[[column_x, column_y]]
            if missing_strategy == "pairwise":
                pair_frame = pair_frame.dropna(axis=0, how="any")
            sample_size = int(len(pair_frame))
            sample_sizes[column_x][column_y] = sample_size
            sample_sizes[column_y][column_x] = sample_size
            coefficient = self._coefficient(
                pair_frame,
                column_x,
                column_y,
                method,
                missing_strategy,
                sample_size,
                warnings,
            )
            matrix[column_x][column_y] = coefficient
            matrix[column_y][column_x] = coefficient
            pair = self._pair_result(column_x, column_y, coefficient, sample_size)
            pairs.append(pair)
            if sample_size < MIN_RELIABLE_SAMPLE_SIZE:
                warnings.append(
                    f"Pair '{column_x}' and '{column_y}' has only {sample_size} valid samples; "
                    "its coefficient must not be treated as a reliable strong relationship."
                )
        return matrix, sample_sizes, pairs

    def _coefficient(
        self,
        pair_frame: pd.DataFrame,
        column_x: str,
        column_y: str,
        method: str,
        missing_strategy: str,
        sample_size: int,
        warnings: list[str],
    ) -> float | None:
        if sample_size < 2:
            return None
        try:
            left = pair_frame[column_x]
            right = pair_frame[column_y]
            if method == "spearman":
                left = left.rank(method="average")
                right = right.rank(method="average")
            raw = left.corr(right, method="pearson")
        except (ArithmeticError, TypeError, ValueError):
            logger.warning(
                "Correlation pair failed | method=%s missing_strategy=%s status=pair_failed",
                method,
                missing_strategy,
            )
            warnings.append(
                f"Correlation could not be calculated for '{column_x}' and '{column_y}'."
            )
            return None
        coefficient = float(raw)
        if not math.isfinite(coefficient):
            warnings.append(
                f"Correlation is unavailable for '{column_x}' and '{column_y}', usually because "
                "one field is constant within the valid pairwise sample."
            )
            return None
        return round(max(-1.0, min(1.0, coefficient)), 6)

    def _pair_result(
        self,
        column_x: str,
        column_y: str,
        coefficient: float | None,
        sample_size: int,
    ) -> dict[str, object]:
        sufficient_sample = sample_size >= MIN_RELIABLE_SAMPLE_SIZE
        direction = self._direction(coefficient)
        strength = self._strength(coefficient)
        if coefficient is None:
            message = (
                f"{column_x} and {column_y} do not have enough paired samples or usable variation "
                "for correlation."
            )
        elif not sufficient_sample:
            message = (
                f"The coefficient for {column_x} and {column_y} is {coefficient:.6g} and falls in the "
                f"generic {strength} {direction} range, but the sample is too small for a reliable conclusion."
            )
        else:
            message = (
                f"{column_x} and {column_y} have a {strength} {direction} correlation "
                f"({coefficient:.6g})."
            )
        return {
            "column_x": column_x,
            "column_y": column_y,
            "coefficient": coefficient,
            "absolute_coefficient": abs(coefficient) if coefficient is not None else None,
            "direction": direction,
            "strength": strength,
            "sample_size": sample_size,
            "sufficient_sample": sufficient_sample,
            "message": message,
        }

    def _direction(self, coefficient: float | None) -> str:
        if coefficient is None or coefficient == 0:
            return "none"
        return "positive" if coefficient > 0 else "negative"

    def _strength(self, coefficient: float | None) -> str:
        if coefficient is None:
            return "unavailable"
        absolute = abs(coefficient)
        if absolute >= STRONG_CORRELATION_THRESHOLD:
            return "strong"
        if absolute >= MODERATE_CORRELATION_THRESHOLD:
            return "moderate"
        if absolute >= WEAK_CORRELATION_THRESHOLD:
            return "weak"
        return "negligible"

    def _multicollinearity_risks(
        self,
        reliable_pairs: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        risks: list[dict[str, object]] = []
        for pair in reliable_pairs:
            absolute = pair["absolute_coefficient"]
            if not isinstance(absolute, (int, float)) or absolute < MULTICOLLINEARITY_RISK_THRESHOLD:
                continue
            risks.append(
                {
                    "column_x": pair["column_x"],
                    "column_y": pair["column_y"],
                    "coefficient": pair["coefficient"],
                    "risk_level": "high",
                    "message": (
                        "The two input fields are highly correlated and may create redundancy or "
                        "multicollinearity risk in later regression or machine-learning work."
                    ),
                    "suggestion": (
                        "Review business meaning before keeping or removing either field; calculate VIF "
                        "separately during a future modeling stage if needed."
                    ),
                }
            )
        return risks

    def _input_error(
        self,
        message: str,
        *,
        error_code: str = "INVALID_CORRELATION_INPUT",
        details: dict[str, object] | None = None,
    ) -> ToolExecutionException:
        payload = {"tool_name": self.name}
        if details:
            payload.update(details)
        return ToolExecutionException(
            message,
            error_code=error_code,
            status_code=422,
            details=payload,
        )

    def _json_safe(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): self._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(item) for item in value]
        if isinstance(value, (pd.Timestamp, datetime, date)):
            return value.isoformat()
        if isinstance(value, pd.Timedelta):
            return str(value)
        if value is None:
            return None
        if hasattr(value, "item"):
            return self._json_safe(value.item())
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        if isinstance(value, (str, int, bool)):
            return value
        return str(value)

    def _log_execution(
        self,
        method: str,
        missing_strategy: str,
        analyzed_column_count: int,
        analyzed_pair_count: int,
        excluded_column_count: int,
        status: str,
        started_at: float,
    ) -> None:
        log_method = logger.info if status == "success" else logger.warning
        log_method(
            "Correlation analysis | method=%s missing_strategy=%s analyzed_column_count=%d "
            "analyzed_pair_count=%d excluded_column_count=%d status=%s elapsed_ms=%.0f",
            method,
            missing_strategy,
            analyzed_column_count,
            analyzed_pair_count,
            excluded_column_count,
            status,
            (time.monotonic() - started_at) * 1000,
        )
