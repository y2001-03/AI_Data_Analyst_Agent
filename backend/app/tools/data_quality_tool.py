"""Enterprise data quality analysis tool."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from app.core.exceptions import ToolExecutionException
from app.core.logging import get_logger
from app.schemas.file import AnalysisTask, ExecutionResult
from app.tools.dataframe_tools import BaseDataframeTool, DatasetContext

logger = get_logger(__name__)


class DataQualityTool(BaseDataframeTool):
    """Profile dataset reliability before deeper analysis."""

    name = "data_quality_tool"
    result_type = "data_quality_analysis"
    high_missing_threshold = 0.3
    high_outlier_threshold = 0.1
    id_unique_ratio_threshold = 0.95

    def run(
        self,
        dataframe: pd.DataFrame,
        task: AnalysisTask,
        context: DatasetContext,
    ) -> ExecutionResult:
        """Return a serializable data quality report for the dataframe."""
        del context
        try:
            report = self._build_report(dataframe, task)
        except ToolExecutionException:
            raise
        except Exception as exc:
            raise ToolExecutionException(
                "Data quality analysis failed.",
                details={"tool_name": self.name},
            ) from exc
        return ExecutionResult(
            task_name=task.task_name,
            type=self.result_type,
            data=self._json_safe(report),
            chart=None,
        )

    def _build_report(self, dataframe: pd.DataFrame, task: AnalysisTask) -> dict[str, object]:
        row_count = int(len(dataframe))
        column_count = int(len(dataframe.columns))
        issues: list[dict[str, object]] = []
        columns: list[dict[str, object]] = []

        duplicate_row_count = int(dataframe.duplicated().sum()) if row_count else 0
        duplicate_row_ratio = self._ratio(duplicate_row_count, row_count)
        total_cells = row_count * column_count
        overall_missing_count = int(dataframe.isna().sum().sum()) if total_cells else 0
        overall_missing_ratio = self._ratio(overall_missing_count, total_cells)

        if row_count == 0:
            issues.append(
                {
                    "severity": "high",
                    "issue_type": "empty_dataset",
                    "column": None,
                    "message": "Dataset has no rows, so data quality checks are limited.",
                    "suggestion": "Upload or select a dataset with rows before running business analysis.",
                }
            )

        if duplicate_row_count > 0:
            issues.append(
                {
                    "severity": "high" if duplicate_row_ratio >= 0.1 else "medium",
                    "issue_type": "duplicate_rows",
                    "column": None,
                    "message": f"Dataset contains {duplicate_row_count} duplicate rows.",
                    "suggestion": "Review whether duplicated records are expected before aggregating metrics.",
                }
            )

        constant_column_count = 0
        total_outlier_count = 0
        for column_index, column_name in enumerate(dataframe.columns):
            series = dataframe.iloc[:, column_index]
            series.name = column_name
            try:
                column_report = self._profile_column(series, row_count)
            except Exception as exc:
                logger.warning(
                    "Column quality profiling failed | tool=%s column=%s error=%s",
                    self.name,
                    column_name,
                    str(exc),
                )
                column_report = self._failed_column_report(series, row_count)
            if column_report["constant_column"]:
                constant_column_count += 1
            total_outlier_count += int(column_report["outlier_count"])
            columns.append(column_report)
            issues.extend(self._column_issues(column_report))

        constant_column_ratio = self._ratio(constant_column_count, column_count)
        numeric_value_count = int(
            sum(
                series.count()
                for _, series in dataframe.items()
                if self._infer_type(series) == "numeric"
            )
        )
        high_outlier_ratio = self._ratio(total_outlier_count, numeric_value_count)
        quality_score = (
            0.0
            if row_count == 0 or column_count == 0
            else self._quality_score(
                missing_ratio=overall_missing_ratio,
                duplicate_ratio=duplicate_row_ratio,
                constant_column_ratio=constant_column_ratio,
                high_outlier_ratio=high_outlier_ratio,
            )
        )

        return {
            "tool_name": self.result_type,
            "summary": {
                "row_count": row_count,
                "column_count": column_count,
                "duplicate_row_count": duplicate_row_count,
                "duplicate_row_ratio": duplicate_row_ratio,
                "overall_missing_count": overall_missing_count,
                "overall_missing_ratio": overall_missing_ratio,
                "constant_column_count": constant_column_count,
                "constant_column_ratio": constant_column_ratio,
                "total_outlier_count": total_outlier_count,
                "high_outlier_ratio": high_outlier_ratio,
                "quality_score": quality_score,
                "score_explanation": (
                    "Generic heuristic score only: 100 minus weighted penalties for missing values, "
                    "duplicate rows, constant columns, and numeric outliers. Empty datasets are scored as 0."
                ),
            },
            "columns": columns,
            "issues": issues,
            "metadata": {
                "outlier_method": "iqr",
                "outlier_rule": "values below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR",
                "high_missing_threshold": self.high_missing_threshold,
                "high_outlier_threshold": self.high_outlier_threshold,
                "score_weights": {
                    "missing_ratio": 40,
                    "duplicate_row_ratio": 25,
                    "constant_column_ratio": 20,
                    "high_outlier_ratio": 15,
                },
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "task_params": task.params or {},
            },
        }

    def _profile_column(self, series: pd.Series, row_count: int) -> dict[str, object]:
        missing_count = int(series.isna().sum())
        missing_ratio = self._ratio(missing_count, row_count)
        unique_count = int(series.nunique(dropna=True))
        unique_ratio = self._ratio(unique_count, row_count)
        non_missing_count = int(series.notna().sum())
        constant_column = bool(non_missing_count > 0 and unique_count <= 1)
        inferred_type = self._infer_type(series)
        possible_id_column = bool(
            row_count > 0
            and unique_ratio >= self.id_unique_ratio_threshold
            and missing_count == 0
            and not constant_column
            and inferred_type in {"numeric", "text"}
            and (self._name_looks_like_id(series.name) or inferred_type in {"numeric", "text"})
        )
        outlier_count, outlier_ratio = self._outlier_summary(series, row_count)
        return {
            "name": str(series.name),
            "dtype": str(series.dtype),
            "inferred_type": inferred_type,
            "missing_count": missing_count,
            "missing_ratio": missing_ratio,
            "unique_count": unique_count,
            "unique_ratio": unique_ratio,
            "constant_column": constant_column,
            "possible_id_column": possible_id_column,
            "outlier_count": outlier_count,
            "outlier_ratio": outlier_ratio,
            "issues": [],
        }

    def _failed_column_report(self, series: pd.Series, row_count: int) -> dict[str, object]:
        missing_count = int(series.isna().sum()) if row_count else 0
        return {
            "name": str(series.name),
            "dtype": str(series.dtype),
            "inferred_type": "unsupported",
            "missing_count": missing_count,
            "missing_ratio": self._ratio(missing_count, row_count),
            "unique_count": 0,
            "unique_ratio": 0.0,
            "constant_column": False,
            "possible_id_column": False,
            "outlier_count": 0,
            "outlier_ratio": 0.0,
            "issues": [
                {
                    "severity": "low",
                    "issue_type": "column_profile_failed",
                    "message": "Column could not be fully profiled.",
                }
            ],
        }

    def _column_issues(self, column_report: dict[str, object]) -> list[dict[str, object]]:
        column_name = str(column_report["name"])
        issues: list[dict[str, object]] = []
        if float(column_report["missing_ratio"]) >= self.high_missing_threshold:
            issues.append(
                {
                    "severity": "high",
                    "issue_type": "missing_values",
                    "column": column_name,
                    "message": f"Column '{column_name}' has a high missing value ratio.",
                    "suggestion": "Validate whether this field is required for the planned analysis.",
                }
            )
        elif int(column_report["missing_count"]) > 0:
            issues.append(
                {
                    "severity": "medium",
                    "issue_type": "missing_values",
                    "column": column_name,
                    "message": f"Column '{column_name}' contains missing values.",
                    "suggestion": "Account for missing values before interpreting this field.",
                }
            )
        if bool(column_report["constant_column"]):
            issues.append(
                {
                    "severity": "medium",
                    "issue_type": "constant_column",
                    "column": column_name,
                    "message": f"Column '{column_name}' has only one non-missing value.",
                    "suggestion": "Exclude this field from segmentation or modeling unless it has business meaning.",
                }
            )
        if bool(column_report["possible_id_column"]):
            issues.append(
                {
                    "severity": "low",
                    "issue_type": "possible_id_column",
                    "column": column_name,
                    "message": f"Column '{column_name}' looks like an identifier.",
                    "suggestion": "Avoid treating identifier-like fields as continuous metrics.",
                }
            )
        if float(column_report["outlier_ratio"]) >= self.high_outlier_threshold:
            issues.append(
                {
                    "severity": "medium",
                    "issue_type": "numeric_outliers",
                    "column": column_name,
                    "message": f"Column '{column_name}' has an elevated IQR outlier ratio.",
                    "suggestion": "Inspect extreme values before relying on averages or totals.",
                }
            )
        column_report["issues"] = [
            {
                "severity": issue["severity"],
                "issue_type": issue["issue_type"],
                "message": issue["message"],
            }
            for issue in issues
        ]
        return issues

    def _infer_type(self, series: pd.Series) -> str:
        if pd.api.types.is_bool_dtype(series):
            return "boolean"
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"
        if isinstance(series.dtype, pd.CategoricalDtype):
            return "categorical"
        return "text"

    def _outlier_summary(self, series: pd.Series, row_count: int) -> tuple[int, float]:
        if row_count == 0 or self._infer_type(series) != "numeric":
            return 0, 0.0
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if numeric.empty:
            return 0, 0.0
        q1 = float(numeric.quantile(0.25))
        q3 = float(numeric.quantile(0.75))
        iqr = q3 - q1
        if iqr == 0:
            return 0, 0.0
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_count = int(((numeric < lower_bound) | (numeric > upper_bound)).sum())
        return outlier_count, self._ratio(outlier_count, row_count)

    def _quality_score(
        self,
        *,
        missing_ratio: float,
        duplicate_ratio: float,
        constant_column_ratio: float,
        high_outlier_ratio: float,
    ) -> float:
        # Transparent generic heuristic: 100 minus weighted quality penalties.
        penalty = (
            missing_ratio * 40
            + duplicate_ratio * 25
            + constant_column_ratio * 20
            + high_outlier_ratio * 15
        )
        return round(max(0.0, min(100.0, 100.0 - penalty)), 2)

    def _ratio(self, numerator: int | float, denominator: int | float) -> float:
        if denominator <= 0:
            return 0.0
        return round(float(numerator) / float(denominator), 6)

    def _name_looks_like_id(self, name: object) -> bool:
        normalized = str(name).lower()
        return normalized == "id" or normalized.endswith("_id") or normalized.endswith("id")

    def _json_safe(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): self._json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._json_safe(item) for item in value]
        if isinstance(value, tuple):
            return [self._json_safe(item) for item in value]
        if pd.isna(value) and not isinstance(value, (list, tuple, dict)):
            return None
        if hasattr(value, "item"):
            return self._json_safe(value.item())
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
