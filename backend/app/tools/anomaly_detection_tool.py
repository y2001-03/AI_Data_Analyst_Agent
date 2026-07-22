"""Enterprise, explainable anomaly detection for dataframe data."""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Callable

import numpy as np
import pandas as pd

from app.core.exceptions import ToolExecutionException
from app.core.logging import get_logger
from app.schemas.file import AnalysisTask, ExecutionResult
from app.tools.dataframe_tools import BaseDataframeTool, DatasetContext

logger = get_logger(__name__)

DEFAULT_MODE = "global"
DEFAULT_IQR_MULTIPLIER = 1.5
DEFAULT_ZSCORE_THRESHOLD = 3.0
DEFAULT_ROBUST_ZSCORE_THRESHOLD = 3.5
DEFAULT_ROLLING_ZSCORE_THRESHOLD = 3.0
DEFAULT_ROLLING_WINDOW = 7
MIN_ROLLING_WINDOW = 3
ZSCORE_DDOF = 1
TIME_PARSE_FAILURE_THRESHOLD = 0.05

MIN_GLOBAL_SAMPLE_SIZE = 5
MIN_GROUP_SAMPLE_SIZE = 5
MIN_RELIABLE_SAMPLE_SIZE = 10
MAX_ANALYSIS_COLUMNS = 30
MAX_GROUP_COUNT = 100
MAX_ANOMALIES_PER_COLUMN = 100
MAX_TOTAL_ANOMALIES = 500
MAX_CHART_POINTS = 500
MAX_WARNINGS = 200

IQR_HIGH_SEVERITY_SCORE = 3.0
IQR_CRITICAL_SEVERITY_SCORE = 5.0
ZSCORE_HIGH_SEVERITY_SCORE = 4.0
ZSCORE_CRITICAL_SEVERITY_SCORE = 5.0
ROBUST_HIGH_SEVERITY_SCORE = 5.0
ROBUST_CRITICAL_SEVERITY_SCORE = 7.0
ROLLING_HIGH_SEVERITY_OFFSET = 1.0
ROLLING_CRITICAL_SEVERITY_OFFSET = 2.0

SUPPORTED_MODES = {"global", "groupwise", "time_series"}
MODE_METHODS = {
    "global": {"iqr", "zscore", "robust_zscore"},
    "groupwise": {"iqr", "zscore", "robust_zscore"},
    "time_series": {"rolling_zscore", "iqr", "robust_zscore"},
}
DEFAULT_METHODS = {
    "global": "iqr",
    "groupwise": "iqr",
    "time_series": "rolling_zscore",
}


@dataclass
class ColumnSelection:
    """Validated numeric columns and their safe numeric values."""

    columns: list[str]
    frame: pd.DataFrame
    excluded_columns: list[dict[str, str]]
    original_numeric_column_count: int
    original_eligible_column_count: int


@dataclass
class DetectionOutcome:
    """Internal result for one method applied to one numeric series."""

    sample_size: int
    scores: pd.Series
    flags: pd.Series
    statistics: dict[str, object]
    status: str = "analyzed"
    warning: str | None = None
    details: dict[str, pd.Series] = field(default_factory=dict)


@dataclass
class AnomalyCandidate:
    """Internal sortable anomaly record."""

    column: str
    position: int
    absolute_score: float
    detection_order: int
    record: dict[str, object]


class AnomalyDetectionTool(BaseDataframeTool):
    """Detect candidate statistical anomalies without changing source data."""

    name = "anomaly_detection_tool"
    tool_name = "anomaly_detection"
    result_type = "anomaly_detection"

    def __init__(self) -> None:
        self._mode_dispatchers: dict[
            str,
            Callable[[pd.DataFrame, dict[str, object], list[str]], dict[str, object]],
        ] = {
            "global": self._run_global,
            "groupwise": self._run_groupwise,
            "time_series": self._run_time_series,
        }
        self._method_dispatchers: dict[
            str,
            Callable[[pd.Series, dict[str, object]], DetectionOutcome],
        ] = {
            "iqr": self._detect_iqr,
            "zscore": self._detect_zscore,
            "robust_zscore": self._detect_robust_zscore,
            "rolling_zscore": self._detect_rolling_zscore,
        }

    def run(
        self,
        dataframe: pd.DataFrame,
        task: AnalysisTask,
        context: DatasetContext,
    ) -> ExecutionResult:
        """Run a validated detection mode and return a JSON-safe report."""
        del context
        started_at = time.monotonic()
        raw_params = task.params or {}
        mode = str(raw_params.get("mode", DEFAULT_MODE)).lower()
        method = str(raw_params.get("method", DEFAULT_METHODS.get(mode, "iqr"))).lower()
        log_summary = {
            "analyzed_column_count": 0,
            "analyzed_sample_count": 0,
            "detected_anomaly_count": 0,
            "returned_anomaly_count": 0,
            "truncated": False,
        }
        try:
            if dataframe.empty:
                raise self._input_error("Anomaly detection requires a non-empty dataframe.")
            config = self._validated_config(raw_params, mode, method)
            warnings: list[str] = []
            report = self._mode_dispatchers[mode](dataframe, config, warnings)
            summary = report["summary"]
            for key in log_summary:
                log_summary[key] = summary.get(key, log_summary[key])
        except ToolExecutionException:
            self._log_execution(mode, method, log_summary, "failed", started_at)
            raise
        except Exception as exc:
            self._log_execution(mode, method, log_summary, "failed", started_at)
            raise ToolExecutionException(
                "Anomaly detection failed.",
                details={"tool_name": self.name},
            ) from exc

        self._log_execution(mode, method, log_summary, "success", started_at)
        return ExecutionResult(
            task_name=task.task_name,
            type=self.result_type,
            data=self._json_safe(report),
            chart=None,
        )

    def _validated_config(
        self,
        params: dict[str, object],
        mode: str,
        method: str,
    ) -> dict[str, object]:
        if mode not in SUPPORTED_MODES:
            raise self._input_error(
                "Anomaly detection mode must be 'global', 'groupwise', or 'time_series'.",
                error_code="INVALID_ANOMALY_MODE",
            )
        if method not in MODE_METHODS[mode]:
            allowed = ", ".join(sorted(MODE_METHODS[mode]))
            raise self._input_error(
                f"Method '{method}' is not compatible with mode '{mode}'. Allowed methods: {allowed}.",
                error_code="INVALID_ANOMALY_METHOD",
            )

        config = dict(params)
        config["mode"] = mode
        config["method"] = method
        if method == "iqr":
            config["iqr_multiplier"] = self._positive_number(
                params.get("iqr_multiplier", DEFAULT_IQR_MULTIPLIER),
                "iqr_multiplier",
            )
        elif method == "robust_zscore":
            config["threshold"] = self._positive_number(
                params.get("threshold", DEFAULT_ROBUST_ZSCORE_THRESHOLD),
                "threshold",
            )
        elif method == "zscore":
            config["threshold"] = self._positive_number(
                params.get("threshold", DEFAULT_ZSCORE_THRESHOLD),
                "threshold",
            )
        else:
            config["threshold"] = self._positive_number(
                params.get("threshold", DEFAULT_ROLLING_ZSCORE_THRESHOLD),
                "threshold",
            )
            window = params.get("window", DEFAULT_ROLLING_WINDOW)
            if isinstance(window, bool) or not isinstance(window, int) or window < MIN_ROLLING_WINDOW:
                raise self._input_error(
                    f"rolling_zscore window must be an integer greater than or equal to {MIN_ROLLING_WINDOW}.",
                    error_code="INVALID_ROLLING_WINDOW",
                )
            config["window"] = window
        return config

    def _run_global(
        self,
        dataframe: pd.DataFrame,
        config: dict[str, object],
        warnings: list[str],
    ) -> dict[str, object]:
        method = str(config["method"])
        selection = self._select_numeric_columns(
            dataframe,
            config.get("columns"),
            "columns",
            set(),
            warnings,
        )
        original_indices = list(dataframe.index)
        candidates: list[AnomalyCandidate] = []
        column_results: list[dict[str, object]] = []
        detection_order = 0

        for column_order, column in enumerate(selection.columns):
            try:
                outcome = self._detect_with_sample_guard(
                    selection.frame[column],
                    method,
                    config,
                    MIN_GLOBAL_SAMPLE_SIZE,
                )
            except (ArithmeticError, TypeError, ValueError):
                self._log_unit_failure("global", method)
                self._add_warning(warnings, f"Column '{column}' could not be analyzed and was skipped.")
                continue
            self._record_outcome_warning(warnings, column, outcome)
            reliable = (
                outcome.status == "analyzed"
                and outcome.sample_size >= MIN_RELIABLE_SAMPLE_SIZE
            )
            column_candidates, detection_order = self._outcome_candidates(
                outcome,
                column,
                "global",
                method,
                reliable,
                original_indices,
                selection.frame[column],
                detection_order,
                config,
            )
            candidates.extend(column_candidates)
            column_results.append(
                self._column_result(column, outcome, len(column_candidates), reliable)
            )

        chart, chart_metadata = self._global_chart(
            selection,
            column_results,
            candidates,
            original_indices,
        )
        return self._assemble_report(
            dataframe,
            config,
            selection,
            column_results,
            candidates,
            warnings,
            chart,
            chart_metadata,
            {},
        )

    def _run_groupwise(
        self,
        dataframe: pd.DataFrame,
        config: dict[str, object],
        warnings: list[str],
    ) -> dict[str, object]:
        group_by = config.get("group_by")
        if not isinstance(group_by, str) or not group_by:
            raise self._input_error(
                "groupwise mode requires a non-empty group_by column.",
                error_code="MISSING_GROUP_BY",
            )
        if group_by not in dataframe.columns:
            raise self._input_error(
                f"Unknown group_by column: {group_by}.",
                error_code="UNKNOWN_GROUP_BY",
            )
        group_series = dataframe[group_by]
        if (
            pd.api.types.is_datetime64_any_dtype(group_series)
            or pd.api.types.is_timedelta64_dtype(group_series)
        ):
            raise self._input_error(
                "group_by supports text, category, boolean, or discrete numeric fields, not datetime/timedelta.",
                error_code="INVALID_GROUP_BY_TYPE",
            )

        method = str(config["method"])
        selection = self._select_numeric_columns(
            dataframe,
            config.get("columns"),
            "columns",
            {group_by},
            warnings,
        )
        groups = self._ordered_groups(dataframe[group_by])
        original_group_count = len(groups)
        selected_groups = self._limit_groups(groups, warnings)
        original_indices = list(dataframe.index)
        candidates: list[AnomalyCandidate] = []
        column_results: list[dict[str, object]] = []
        group_anomaly_counts = {label: 0 for label, _ in selected_groups}
        detection_order = 0

        for column in selection.columns:
            group_statistics: list[dict[str, object]] = []
            column_candidate_count = 0
            total_sample_size = 0
            for group_label, positions in selected_groups:
                group_series = selection.frame.loc[positions, column]
                try:
                    outcome = self._detect_with_sample_guard(
                        group_series,
                        method,
                        config,
                        MIN_GROUP_SAMPLE_SIZE,
                    )
                except (ArithmeticError, TypeError, ValueError):
                    self._log_unit_failure("groupwise", method)
                    self._add_warning(
                        warnings,
                        f"One group for column '{column}' could not be analyzed and was skipped.",
                    )
                    continue
                total_sample_size += outcome.sample_size
                self._record_outcome_warning(
                    warnings,
                    f"Column '{column}' in group '{self._safe_group_label(group_label)}'",
                    outcome,
                )
                reliable = (
                    outcome.status == "analyzed"
                    and outcome.sample_size >= MIN_RELIABLE_SAMPLE_SIZE
                )
                group_candidates, detection_order = self._outcome_candidates(
                    outcome,
                    column,
                    "groupwise",
                    method,
                    reliable,
                    original_indices,
                    selection.frame[column],
                    detection_order,
                    config,
                    group=group_label,
                )
                candidates.extend(group_candidates)
                anomaly_count = len(group_candidates)
                column_candidate_count += anomaly_count
                group_anomaly_counts[group_label] += anomaly_count
                group_statistics.append(
                    {
                        "group": group_label,
                        "sample_size": outcome.sample_size,
                        "status": outcome.status,
                        "reliable": reliable,
                        "anomaly_count": anomaly_count,
                        "statistics": outcome.statistics,
                    }
                )
            column_results.append(
                {
                    "column": column,
                    "sample_size": total_sample_size,
                    "anomaly_count": column_candidate_count,
                    "anomaly_ratio": self._ratio(column_candidate_count, total_sample_size),
                    "reliable": any(
                        bool(item["reliable"]) and item["status"] == "analyzed"
                        for item in group_statistics
                    ),
                    "status": "analyzed",
                    "group_statistics": group_statistics,
                }
            )

        chart = {
            "type": "group_anomaly_summary",
            "groups": [self._safe_group_label(label) for label, _ in selected_groups],
            "anomaly_counts": [group_anomaly_counts[label] for label, _ in selected_groups],
        }
        mode_metadata = {
            "group_by": group_by,
            "missing_group_label": "__MISSING__",
            "original_group_count": original_group_count,
            "actual_group_count": len(selected_groups),
            "max_group_count": MAX_GROUP_COUNT,
            "group_selection_rule": "sample_count_desc_then_first_appearance",
        }
        return self._assemble_report(
            dataframe,
            config,
            selection,
            column_results,
            candidates,
            warnings,
            chart,
            {},
            mode_metadata,
        )

    def _run_time_series(
        self,
        dataframe: pd.DataFrame,
        config: dict[str, object],
        warnings: list[str],
    ) -> dict[str, object]:
        time_column = config.get("time_column")
        if not isinstance(time_column, str) or not time_column:
            raise self._input_error(
                "time_series mode requires a non-empty time_column.",
                error_code="MISSING_TIME_COLUMN",
            )
        if time_column not in dataframe.columns:
            raise self._input_error(
                f"Unknown time_column: {time_column}.",
                error_code="UNKNOWN_TIME_COLUMN",
            )
        if config.get("value_columns") is None:
            raise self._input_error(
                "time_series mode requires at least one value_columns entry.",
                error_code="MISSING_VALUE_COLUMNS",
            )

        parsed_time, parse_metadata = self._parse_time_column(dataframe[time_column], warnings)
        selection = self._select_numeric_columns(
            dataframe,
            config.get("value_columns"),
            "value_columns",
            {time_column},
            warnings,
        )
        order_frame = pd.DataFrame(
            {
                "position": np.arange(len(dataframe), dtype=np.int64),
                "time": parsed_time.reset_index(drop=True),
            }
        ).dropna(subset=["time"])
        order_frame = order_frame.sort_values("time", kind="mergesort")
        sorted_positions = [int(position) for position in order_frame["position"].tolist()]
        if not sorted_positions:
            raise self._input_error(
                "No valid timestamp rows are available for time-series anomaly detection.",
                error_code="NO_VALID_TIME_ROWS",
            )

        method = str(config["method"])
        original_indices = list(dataframe.index)
        candidates: list[AnomalyCandidate] = []
        column_results: list[dict[str, object]] = []
        outcomes: dict[str, DetectionOutcome] = {}
        detection_order = 0
        for column in selection.columns:
            ordered_series = selection.frame.loc[sorted_positions, column]
            try:
                outcome = self._detect_with_sample_guard(
                    ordered_series,
                    method,
                    config,
                    MIN_GLOBAL_SAMPLE_SIZE,
                )
            except (ArithmeticError, TypeError, ValueError):
                self._log_unit_failure("time_series", method)
                self._add_warning(warnings, f"Time-series column '{column}' could not be analyzed.")
                continue
            outcomes[column] = outcome
            self._record_outcome_warning(warnings, column, outcome)
            reliable = (
                outcome.status == "analyzed"
                and outcome.sample_size >= MIN_RELIABLE_SAMPLE_SIZE
            )
            time_values = {
                position: parsed_time.iloc[position] for position in sorted_positions
            }
            column_candidates, detection_order = self._outcome_candidates(
                outcome,
                column,
                "time_series",
                method,
                reliable,
                original_indices,
                selection.frame[column],
                detection_order,
                config,
                time_values=time_values,
            )
            candidates.extend(column_candidates)
            column_results.append(
                self._column_result(column, outcome, len(column_candidates), reliable)
            )

        chart, chart_metadata = self._time_series_chart(
            selection,
            column_results,
            candidates,
            outcomes,
            parsed_time,
            sorted_positions,
        )
        mode_metadata = {
            "time_column": time_column,
            "time_parse_failure_count": parse_metadata["failure_count"],
            "time_parse_failure_ratio": parse_metadata["failure_ratio"],
            "time_rows_after_parsing": len(sorted_positions),
            "stable_time_sort": True,
            "future_data_leakage_prevented": method == "rolling_zscore",
        }
        return self._assemble_report(
            dataframe,
            config,
            selection,
            column_results,
            candidates,
            warnings,
            chart,
            chart_metadata,
            mode_metadata,
        )

    def _select_numeric_columns(
        self,
        dataframe: pd.DataFrame,
        raw_columns: object,
        parameter_name: str,
        excluded_names: set[str],
        warnings: list[str],
    ) -> ColumnSelection:
        requested = self._requested_columns(raw_columns, parameter_name)
        if requested is not None and len(requested) > MAX_ANALYSIS_COLUMNS:
            raise self._input_error(
                f"Explicit anomaly detection supports at most {MAX_ANALYSIS_COLUMNS} columns.",
                error_code="ANOMALY_COLUMN_LIMIT_EXCEEDED",
            )

        excluded_columns: list[dict[str, str]] = []
        if requested is not None:
            unknown = [column for column in requested if column not in dataframe.columns]
            if unknown:
                raise self._input_error(
                    f"Unknown anomaly detection column(s): {', '.join(unknown)}.",
                    error_code="UNKNOWN_ANOMALY_COLUMNS",
                    details={"unknown_columns": unknown},
                )
            if any(column in excluded_names for column in requested):
                blocked = [column for column in requested if column in excluded_names]
                raise self._input_error(
                    f"Grouping or time columns cannot also be anomaly value columns: {', '.join(blocked)}.",
                    error_code="INVALID_ANOMALY_COLUMNS",
                )
            invalid = [
                {"column": column, "reason": self._exclusion_reason(dataframe[column])}
                for column in requested
                if self._exclusion_reason(dataframe[column]) is not None
            ]
            if invalid:
                description = ", ".join(
                    f"{item['column']} ({item['reason']})" for item in invalid
                )
                raise self._input_error(
                    f"Explicit anomaly detection columns must be numeric: {description}.",
                    error_code="NON_NUMERIC_ANOMALY_COLUMNS",
                    details={"invalid_columns": invalid},
                )
            candidates = list(requested)
        else:
            candidates = []
            for column in dataframe.columns:
                column_name = str(column)
                if column_name in excluded_names:
                    excluded_columns.append({"column": column_name, "reason": "mode_key_column"})
                    continue
                reason = self._exclusion_reason(dataframe[column])
                if reason is None:
                    candidates.append(column_name)
                else:
                    excluded_columns.append({"column": column_name, "reason": reason})

        if not candidates:
            raise self._input_error("Anomaly detection found no numeric value columns.")

        numeric_values: dict[str, pd.Series] = {}
        eligible: list[str] = []
        for column in candidates:
            source = dataframe[column]
            converted = pd.to_numeric(source, errors="coerce").astype("float64")
            infinity_count = int(np.isinf(converted).sum())
            numeric = self._numeric_series(converted)
            numeric_values[column] = numeric
            finite = numeric.dropna()
            if infinity_count:
                self._add_warning(
                    warnings,
                    f"Column '{column}' contains {infinity_count} infinite values; they were treated as missing.",
                )
            if finite.empty:
                excluded_columns.append({"column": column, "reason": "all_null_or_non_finite"})
            elif len(finite) >= 2 and int(finite.nunique(dropna=True)) <= 1:
                excluded_columns.append({"column": column, "reason": "constant"})
            else:
                eligible.append(column)

        original_numeric_count = len(candidates)
        original_eligible_count = len(eligible)
        if requested is None and len(eligible) > MAX_ANALYSIS_COLUMNS:
            selected, limited = self._limit_columns(numeric_values, eligible)
            eligible = selected
            excluded_columns.extend(
                {"column": column, "reason": "column_limit"} for column in limited
            )
            self._add_warning(
                warnings,
                f"{original_eligible_count} eligible numeric columns exceeded the limit of "
                f"{MAX_ANALYSIS_COLUMNS}; columns were selected deterministically.",
            )
        if not eligible:
            raise self._input_error(
                "Anomaly detection requires at least one non-constant numeric column.",
                details={"numeric_column_count": original_numeric_count},
            )

        frame = pd.DataFrame(
            {column: numeric_values[column].reset_index(drop=True) for column in eligible}
        )
        return ColumnSelection(
            columns=eligible,
            frame=frame,
            excluded_columns=excluded_columns,
            original_numeric_column_count=original_numeric_count,
            original_eligible_column_count=original_eligible_count,
        )

    def _requested_columns(self, raw_columns: object, parameter_name: str) -> list[str] | None:
        if raw_columns is None:
            return None
        if (
            not isinstance(raw_columns, list)
            or not raw_columns
            or not all(isinstance(column, str) and column for column in raw_columns)
        ):
            raise self._input_error(
                f"{parameter_name} must be a non-empty list of column names.",
                error_code="INVALID_ANOMALY_COLUMNS",
            )
        if len(set(raw_columns)) != len(raw_columns):
            raise self._input_error(
                f"{parameter_name} must not contain duplicate columns.",
                error_code="INVALID_ANOMALY_COLUMNS",
            )
        return raw_columns

    def _exclusion_reason(self, series: pd.Series) -> str | None:
        if pd.api.types.is_bool_dtype(series):
            return "boolean"
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        if pd.api.types.is_timedelta64_dtype(series):
            return "timedelta"
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_complex_dtype(series):
            return None
        if isinstance(series.dtype, pd.CategoricalDtype):
            return "category"
        if pd.api.types.is_string_dtype(series.dtype):
            return "string"
        if pd.api.types.is_object_dtype(series):
            return "object"
        return "unsupported"

    def _numeric_series(self, series: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce").astype("float64")
        return numeric.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)

    def _limit_columns(
        self,
        values: dict[str, pd.Series],
        columns: list[str],
    ) -> tuple[list[str], list[str]]:
        original_position = {column: index for index, column in enumerate(columns)}

        def sort_key(column: str) -> tuple[float, float, int]:
            series = values[column]
            non_null_count = int(series.notna().sum())
            variance = float(series.var(ddof=ZSCORE_DDOF)) if non_null_count >= 2 else float("-inf")
            if not math.isfinite(variance):
                variance = float("-inf")
            return (-non_null_count, -variance, original_position[column])

        ordered = sorted(columns, key=sort_key)
        selected_set = set(ordered[:MAX_ANALYSIS_COLUMNS])
        return (
            [column for column in columns if column in selected_set],
            [column for column in columns if column not in selected_set],
        )

    def _detect_with_sample_guard(
        self,
        series: pd.Series,
        method: str,
        config: dict[str, object],
        minimum_sample_size: int,
    ) -> DetectionOutcome:
        sample_size = int(series.notna().sum())
        if sample_size < 2:
            return self._skipped_outcome(
                series,
                sample_size,
                "Fewer than two valid samples are available; detection was skipped.",
            )
        if sample_size < minimum_sample_size:
            return self._skipped_outcome(
                series,
                sample_size,
                f"Only {sample_size} valid samples are available; at least {minimum_sample_size} are required.",
            )
        outcome = self._method_dispatchers[method](series, config)
        if sample_size < MIN_RELIABLE_SAMPLE_SIZE:
            reliability_warning = (
                f"Only {sample_size} valid samples are available; candidates are marked unreliable "
                f"until at least {MIN_RELIABLE_SAMPLE_SIZE} samples are available."
            )
            outcome.warning = (
                f"{outcome.warning} {reliability_warning}"
                if outcome.warning
                else reliability_warning
            )
        return outcome

    def _detect_iqr(
        self,
        series: pd.Series,
        config: dict[str, object],
    ) -> DetectionOutcome:
        valid = series.dropna()
        q1 = float(valid.quantile(0.25))
        q3 = float(valid.quantile(0.75))
        iqr = q3 - q1
        multiplier = float(config["iqr_multiplier"])
        if not math.isfinite(iqr) or iqr <= 0:
            return self._skipped_outcome(
                series,
                int(len(valid)),
                "IQR is zero or unavailable; the column is constant or near-constant for this sample.",
                {"q1": q1, "q3": q3, "iqr": self._finite_or_none(iqr)},
            )
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        scores = pd.Series(0.0, index=series.index, dtype="float64")
        scores[series.isna()] = np.nan
        high = series > q3
        low = series < q1
        scores.loc[high] = (series.loc[high] - q3) / iqr
        scores.loc[low] = (series.loc[low] - q1) / iqr
        flags = ((series < lower_bound) | (series > upper_bound)).fillna(False)
        return DetectionOutcome(
            sample_size=int(len(valid)),
            scores=scores,
            flags=flags.astype(bool),
            statistics={
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "iqr_multiplier": multiplier,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
            },
        )

    def _detect_zscore(
        self,
        series: pd.Series,
        config: dict[str, object],
    ) -> DetectionOutcome:
        valid = series.dropna()
        mean = float(valid.mean())
        std = float(valid.std(ddof=ZSCORE_DDOF))
        threshold = float(config["threshold"])
        if not math.isfinite(std) or std <= 0:
            return self._skipped_outcome(
                series,
                int(len(valid)),
                "Standard deviation is zero or unavailable; z-score detection was skipped.",
                {"mean": mean, "std": self._finite_or_none(std), "threshold": threshold},
            )
        scores = (series - mean) / std
        flags = (scores.abs() >= threshold).fillna(False)
        return DetectionOutcome(
            sample_size=int(len(valid)),
            scores=scores,
            flags=flags.astype(bool),
            statistics={
                "mean": mean,
                "std": std,
                "threshold": threshold,
                "ddof": ZSCORE_DDOF,
            },
        )

    def _detect_robust_zscore(
        self,
        series: pd.Series,
        config: dict[str, object],
    ) -> DetectionOutcome:
        valid = series.dropna()
        median = float(valid.median())
        mad = float((valid - median).abs().median())
        threshold = float(config["threshold"])
        if not math.isfinite(mad) or mad <= 0:
            return self._skipped_outcome(
                series,
                int(len(valid)),
                "MAD is zero or unavailable; the column is constant or near-constant for robust z-score detection.",
                {"median": median, "mad": self._finite_or_none(mad), "threshold": threshold},
            )
        scores = 0.6745 * (series - median) / mad
        flags = (scores.abs() >= threshold).fillna(False)
        return DetectionOutcome(
            sample_size=int(len(valid)),
            scores=scores,
            flags=flags.astype(bool),
            statistics={"median": median, "mad": mad, "threshold": threshold},
        )

    def _detect_rolling_zscore(
        self,
        series: pd.Series,
        config: dict[str, object],
    ) -> DetectionOutcome:
        window = int(config["window"])
        threshold = float(config["threshold"])
        historical = series.shift(1)
        rolling_mean = historical.rolling(window=window, min_periods=window).mean()
        rolling_std = historical.rolling(window=window, min_periods=window).std(ddof=ZSCORE_DDOF)
        valid_std = rolling_std.where(rolling_std > 0)
        scores = (series - rolling_mean) / valid_std
        scores = scores.replace([np.inf, -np.inf], np.nan)
        flags = (scores.abs() >= threshold).fillna(False)
        warning = None
        if int(scores.notna().sum()) == 0:
            warning = (
                "No rolling score could be calculated because the historical window was incomplete "
                "or its standard deviation was zero."
            )
        return DetectionOutcome(
            sample_size=int(series.notna().sum()),
            scores=scores,
            flags=flags.astype(bool),
            statistics={
                "window": window,
                "threshold": threshold,
                "ddof": ZSCORE_DDOF,
                "uses_historical_values_only": True,
                "shift": 1,
            },
            warning=warning,
            details={"rolling_mean": rolling_mean, "rolling_std": rolling_std},
        )

    def _skipped_outcome(
        self,
        series: pd.Series,
        sample_size: int,
        warning: str,
        statistics: dict[str, object] | None = None,
    ) -> DetectionOutcome:
        return DetectionOutcome(
            sample_size=sample_size,
            scores=pd.Series(np.nan, index=series.index, dtype="float64"),
            flags=pd.Series(False, index=series.index, dtype=bool),
            statistics=statistics or {},
            status="skipped",
            warning=warning,
        )

    def _outcome_candidates(
        self,
        outcome: DetectionOutcome,
        column: str,
        mode: str,
        method: str,
        reliable: bool,
        original_indices: list[object],
        values: pd.Series,
        detection_order: int,
        config: dict[str, object],
        *,
        group: object = None,
        time_values: dict[int, object] | None = None,
    ) -> tuple[list[AnomalyCandidate], int]:
        candidates: list[AnomalyCandidate] = []
        for raw_position in outcome.flags[outcome.flags].index.tolist():
            position = int(raw_position)
            score = float(outcome.scores.loc[raw_position])
            value = float(values.loc[position])
            direction = "high" if score > 0 else "low" if score < 0 else "none"
            severity = self._severity(method, abs(score), config)
            threshold_info = self._threshold_info(outcome, method, raw_position)
            record = {
                "row_reference": self._row_reference(original_indices[position], position),
                "column": column,
                "value": value,
                "mode": mode,
                "method": method,
                "anomaly_score": score,
                "direction": direction,
                "severity": severity,
                "reliable": reliable,
                "group": group if mode == "groupwise" else None,
                "time": (
                    time_values.get(position) if mode == "time_series" and time_values else None
                ),
                "threshold_info": threshold_info,
                "message": self._anomaly_message(column, method, direction, reliable),
            }
            candidates.append(
                AnomalyCandidate(
                    column=column,
                    position=position,
                    absolute_score=abs(score),
                    detection_order=detection_order,
                    record=record,
                )
            )
            detection_order += 1
        return candidates, detection_order

    def _threshold_info(
        self,
        outcome: DetectionOutcome,
        method: str,
        position: object,
    ) -> dict[str, object]:
        if method == "iqr":
            return {
                key: outcome.statistics.get(key)
                for key in ("lower_bound", "upper_bound", "iqr_multiplier")
            }
        if method == "zscore":
            return {
                key: outcome.statistics.get(key)
                for key in ("mean", "std", "threshold", "ddof")
            }
        if method == "robust_zscore":
            return {
                key: outcome.statistics.get(key)
                for key in ("median", "mad", "threshold")
            }
        return {
            "rolling_mean": self._series_value(outcome.details.get("rolling_mean"), position),
            "rolling_std": self._series_value(outcome.details.get("rolling_std"), position),
            "threshold": outcome.statistics.get("threshold"),
            "window": outcome.statistics.get("window"),
        }

    def _assemble_report(
        self,
        dataframe: pd.DataFrame,
        config: dict[str, object],
        selection: ColumnSelection,
        column_results: list[dict[str, object]],
        candidates: list[AnomalyCandidate],
        warnings: list[str],
        chart: dict[str, object],
        chart_metadata: dict[str, object],
        mode_metadata: dict[str, object],
    ) -> dict[str, object]:
        returned, truncated = self._truncate_candidates(candidates, selection.columns)
        if bool(config.get("requested_deletion")):
            self._add_warning(
                warnings,
                "The request mentioned cleaning or deletion, but this tool only detects candidates and "
                "does not modify or remove any values.",
            )
        if truncated:
            self._add_warning(
                warnings,
                "Anomaly results were truncated deterministically by absolute anomaly score.",
            )
        detected_count = len(candidates)
        returned_count = len(returned)
        analyzed_samples = max(
            (int(result.get("sample_size", 0)) for result in column_results),
            default=0,
        )
        records = [candidate.record for candidate in returned]
        summary = {
            "mode": config["mode"],
            "method": config["method"],
            "analyzed_column_count": len(column_results),
            "analyzed_sample_count": analyzed_samples,
            "detected_anomaly_count": detected_count,
            "returned_anomaly_count": returned_count,
            "high_count": sum(record["direction"] == "high" for record in records),
            "low_count": sum(record["direction"] == "low" for record in records),
            "reliable_anomaly_count": sum(bool(record["reliable"]) for record in records),
            "truncated": truncated,
        }
        metadata = {
            "method": config["method"],
            "mode": config["mode"],
            "heuristic_detection": True,
            "anomalies_are_not_automatically_errors": True,
            "detection_only": True,
            "original_dataframe_modified": False,
            "causal_conclusions": False,
            "severity_is_generic_heuristic": True,
            "critical_does_not_imply_business_incident": True,
            "severity_rules": self._severity_rules(config),
            "zscore_standard_deviation_ddof": ZSCORE_DDOF,
            "robust_zscore_note": (
                "Robust z-score is usually less sensitive to extreme values than ordinary z-score, "
                "but it is not universally suitable for every business context."
            ),
            "minimum_global_sample_size": MIN_GLOBAL_SAMPLE_SIZE,
            "minimum_group_sample_size": MIN_GROUP_SAMPLE_SIZE,
            "minimum_reliable_sample_size": MIN_RELIABLE_SAMPLE_SIZE,
            "max_analysis_columns": MAX_ANALYSIS_COLUMNS,
            "max_anomalies_per_column": MAX_ANOMALIES_PER_COLUMN,
            "max_total_anomalies": MAX_TOTAL_ANOMALIES,
            "max_chart_points": MAX_CHART_POINTS,
            "original_column_count": int(len(dataframe.columns)),
            "original_numeric_column_count": selection.original_numeric_column_count,
            "original_eligible_column_count": selection.original_eligible_column_count,
            "actual_analyzed_column_count": len(column_results),
            "detected_anomaly_count": detected_count,
            "returned_anomaly_count": returned_count,
            "truncated": truncated,
            "column_selection_rule": "non_null_count_desc_then_variance_desc_then_original_order",
            "chart_selection_rule": "highest_anomaly_count_then_column_order",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            **self._config_metadata(config),
            **chart_metadata,
            **mode_metadata,
        }
        return {
            "tool_name": self.tool_name,
            "summary": summary,
            "columns": column_results,
            "anomalies": records,
            "excluded_columns": selection.excluded_columns,
            "warnings": warnings,
            "metadata": metadata,
            "chart": chart,
        }

    def _truncate_candidates(
        self,
        candidates: list[AnomalyCandidate],
        columns: list[str],
    ) -> tuple[list[AnomalyCandidate], bool]:
        kept: list[AnomalyCandidate] = []
        for column in columns:
            column_candidates = [item for item in candidates if item.column == column]
            column_candidates.sort(key=lambda item: (-item.absolute_score, item.detection_order))
            kept.extend(column_candidates[:MAX_ANOMALIES_PER_COLUMN])
        kept.sort(key=lambda item: (-item.absolute_score, item.detection_order))
        returned = kept[:MAX_TOTAL_ANOMALIES]
        return returned, len(returned) < len(candidates)

    def _global_chart(
        self,
        selection: ColumnSelection,
        column_results: list[dict[str, object]],
        candidates: list[AnomalyCandidate],
        original_indices: list[object],
    ) -> tuple[dict[str, object], dict[str, object]]:
        main_column = self._main_chart_column(selection.columns, column_results)
        if main_column is None:
            return {"type": "anomaly_scatter", "x": [], "y": [], "anomaly_flags": [], "column": None}, {}
        series = selection.frame[main_column]
        valid_positions = [int(position) for position in series[series.notna()].index.tolist()]
        flagged_scores = {
            item.position: item.absolute_score for item in candidates if item.column == main_column
        }
        positions = self._limited_chart_positions(valid_positions, flagged_scores)
        return (
            {
                "type": "anomaly_scatter",
                "x": [self._chart_x(original_indices[position]) for position in positions],
                "y": [float(series.loc[position]) for position in positions],
                "anomaly_flags": [position in flagged_scores for position in positions],
                "column": main_column,
            },
            {
                "chart_column": main_column,
                "chart_original_point_count": len(valid_positions),
                "chart_returned_point_count": len(positions),
                "chart_truncated": len(positions) < len(valid_positions),
            },
        )

    def _time_series_chart(
        self,
        selection: ColumnSelection,
        column_results: list[dict[str, object]],
        candidates: list[AnomalyCandidate],
        outcomes: dict[str, DetectionOutcome],
        parsed_time: pd.Series,
        sorted_positions: list[int],
    ) -> tuple[dict[str, object], dict[str, object]]:
        main_column = self._main_chart_column(selection.columns, column_results)
        if main_column is None or main_column not in outcomes:
            return {"type": "time_series_anomaly", "x": [], "y": [], "anomaly_flags": [], "column": None}, {}
        series = selection.frame[main_column]
        valid_positions = [position for position in sorted_positions if pd.notna(series.loc[position])]
        flagged_scores = {
            item.position: item.absolute_score for item in candidates if item.column == main_column
        }
        positions = self._limited_chart_positions(valid_positions, flagged_scores)
        return (
            {
                "type": "time_series_anomaly",
                "x": [parsed_time.iloc[position] for position in positions],
                "y": [float(series.loc[position]) for position in positions],
                "anomaly_scores": [
                    self._finite_or_none(outcomes[main_column].scores.loc[position])
                    for position in positions
                ],
                "anomaly_flags": [position in flagged_scores for position in positions],
                "column": main_column,
            },
            {
                "chart_column": main_column,
                "chart_original_point_count": len(valid_positions),
                "chart_returned_point_count": len(positions),
                "chart_truncated": len(positions) < len(valid_positions),
            },
        )

    def _main_chart_column(
        self,
        columns: list[str],
        column_results: list[dict[str, object]],
    ) -> str | None:
        counts = {str(result["column"]): int(result.get("anomaly_count", 0)) for result in column_results}
        if not counts:
            return None
        return max(columns, key=lambda column: (counts.get(column, -1), -columns.index(column)))

    def _limited_chart_positions(
        self,
        positions: list[int],
        flagged_scores: dict[int, float],
    ) -> list[int]:
        if len(positions) <= MAX_CHART_POINTS:
            return positions
        position_order = {position: order for order, position in enumerate(positions)}
        anomaly_positions = sorted(
            (position for position in positions if position in flagged_scores),
            key=lambda position: (-flagged_scores[position], position_order[position]),
        )[:MAX_CHART_POINTS]
        selected = set(anomaly_positions)
        slots = MAX_CHART_POINTS - len(selected)
        if slots > 0:
            remaining = [position for position in positions if position not in selected]
            if slots >= len(remaining):
                selected.update(remaining)
            elif slots == 1:
                selected.add(remaining[len(remaining) // 2])
            else:
                for index in range(slots):
                    offset = round(index * (len(remaining) - 1) / (slots - 1))
                    selected.add(remaining[offset])
        return [position for position in positions if position in selected][:MAX_CHART_POINTS]

    def _ordered_groups(self, series: pd.Series) -> list[tuple[object, list[int]]]:
        groups: OrderedDict[object, list[int]] = OrderedDict()
        normalized = series.astype("object").reset_index(drop=True)
        for position, value in enumerate(normalized.tolist()):
            group = "__MISSING__" if self._is_missing(value) else value
            groups.setdefault(group, []).append(position)
        return list(groups.items())

    def _limit_groups(
        self,
        groups: list[tuple[object, list[int]]],
        warnings: list[str],
    ) -> list[tuple[object, list[int]]]:
        if len(groups) <= MAX_GROUP_COUNT:
            return groups
        ranked = sorted(
            enumerate(groups),
            key=lambda item: (-len(item[1][1]), item[0]),
        )
        selected_indices = {index for index, _ in ranked[:MAX_GROUP_COUNT]}
        self._add_warning(
            warnings,
            f"Group count exceeded {MAX_GROUP_COUNT}; only the largest groups were analyzed deterministically.",
        )
        return [group for index, group in enumerate(groups) if index in selected_indices]

    def _parse_time_column(
        self,
        series: pd.Series,
        warnings: list[str],
    ) -> tuple[pd.Series, dict[str, object]]:
        source = series.reset_index(drop=True)
        if (
            pd.api.types.is_bool_dtype(source)
            or pd.api.types.is_timedelta64_dtype(source)
            or (
                pd.api.types.is_numeric_dtype(source)
                and not pd.api.types.is_datetime64_any_dtype(source)
            )
        ):
            raise self._input_error(
                "time_column must be datetime or an explicitly parseable text/category field.",
                error_code="INVALID_TIME_COLUMN_TYPE",
            )
        if pd.api.types.is_datetime64_any_dtype(source):
            parsed = pd.to_datetime(source, errors="coerce")
        else:
            parsed = pd.to_datetime(source, errors="coerce", format="mixed")
        non_missing_count = int(source.notna().sum())
        failure_count = int((source.notna() & parsed.isna()).sum())
        failure_ratio = self._ratio(failure_count, non_missing_count)
        if failure_ratio > TIME_PARSE_FAILURE_THRESHOLD:
            raise self._input_error(
                "Time column parsing failure ratio exceeded 5%; time-series detection was not run.",
                error_code="TIME_PARSE_FAILURE_RATIO_EXCEEDED",
                details={
                    "failure_count": failure_count,
                    "failure_ratio": failure_ratio,
                },
            )
        if failure_count:
            self._add_warning(
                warnings,
                f"{failure_count} time values could not be parsed and were excluded from time-series detection.",
            )
        return parsed, {"failure_count": failure_count, "failure_ratio": failure_ratio}

    def _column_result(
        self,
        column: str,
        outcome: DetectionOutcome,
        anomaly_count: int,
        reliable: bool,
    ) -> dict[str, object]:
        return {
            "column": column,
            "sample_size": outcome.sample_size,
            "anomaly_count": anomaly_count,
            "anomaly_ratio": self._ratio(anomaly_count, outcome.sample_size),
            "reliable": reliable,
            "status": outcome.status,
            "statistics": outcome.statistics,
        }

    def _severity(
        self,
        method: str,
        absolute_score: float,
        config: dict[str, object],
    ) -> str:
        if method == "iqr":
            if absolute_score >= IQR_CRITICAL_SEVERITY_SCORE:
                return "critical"
            if absolute_score >= IQR_HIGH_SEVERITY_SCORE:
                return "high"
            return "medium"
        if method == "zscore":
            if absolute_score >= ZSCORE_CRITICAL_SEVERITY_SCORE:
                return "critical"
            if absolute_score >= ZSCORE_HIGH_SEVERITY_SCORE:
                return "high"
            return "medium"
        if method == "robust_zscore":
            if absolute_score >= ROBUST_CRITICAL_SEVERITY_SCORE:
                return "critical"
            if absolute_score >= ROBUST_HIGH_SEVERITY_SCORE:
                return "high"
            return "medium"
        threshold = float(config["threshold"])
        if absolute_score >= threshold + ROLLING_CRITICAL_SEVERITY_OFFSET:
            return "critical"
        if absolute_score >= threshold + ROLLING_HIGH_SEVERITY_OFFSET:
            return "high"
        return "medium"

    def _anomaly_message(
        self,
        column: str,
        method: str,
        direction: str,
        reliable: bool,
    ) -> str:
        reliability = (
            "The sample size meets the generic reliability threshold."
            if reliable
            else "The sample is small, so this is only a low-confidence candidate."
        )
        return (
            f"{column} is a candidate {direction} statistical anomaly under {method}. "
            f"{reliability} Business validation is required; detection does not imply an error or cause."
        )

    def _row_reference(self, index: object, position: int) -> dict[str, object]:
        if isinstance(index, (int, np.integer)) and not isinstance(index, (bool, np.bool_)):
            safe_index: object = int(index)
        else:
            converted = self._json_safe(index)
            safe_index = str(converted)
        return {"index": safe_index, "position": int(position)}

    def _positive_number(self, value: object, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise self._input_error(
                f"{name} must be a finite number greater than zero.",
                error_code="INVALID_ANOMALY_THRESHOLD",
            )
        numeric = float(value)
        if not math.isfinite(numeric) or numeric <= 0:
            raise self._input_error(
                f"{name} must be a finite number greater than zero.",
                error_code="INVALID_ANOMALY_THRESHOLD",
            )
        return numeric

    def _record_outcome_warning(
        self,
        warnings: list[str],
        label: str,
        outcome: DetectionOutcome,
    ) -> None:
        if outcome.warning:
            self._add_warning(warnings, f"{label}: {outcome.warning}")

    def _add_warning(self, warnings: list[str], message: str) -> None:
        if message in warnings:
            return
        if len(warnings) < MAX_WARNINGS:
            warnings.append(message)
        elif len(warnings) == MAX_WARNINGS:
            warnings.append("Additional warnings were omitted to keep the response bounded.")

    def _log_unit_failure(self, mode: str, method: str) -> None:
        logger.warning(
            "Anomaly detection unit failed | mode=%s method=%s status=unit_failed",
            mode,
            method,
        )

    def _log_execution(
        self,
        mode: str,
        method: str,
        summary: dict[str, object],
        status: str,
        started_at: float,
    ) -> None:
        log_method = logger.info if status == "success" else logger.warning
        log_method(
            "Anomaly detection | mode=%s method=%s analyzed_column_count=%d analyzed_sample_count=%d "
            "detected_anomaly_count=%d returned_anomaly_count=%d truncated=%s status=%s elapsed_ms=%.0f",
            mode,
            method,
            int(summary["analyzed_column_count"]),
            int(summary["analyzed_sample_count"]),
            int(summary["detected_anomaly_count"]),
            int(summary["returned_anomaly_count"]),
            bool(summary["truncated"]),
            status,
            (time.monotonic() - started_at) * 1000,
        )

    def _config_metadata(self, config: dict[str, object]) -> dict[str, object]:
        keys = ("iqr_multiplier", "threshold", "window")
        return {key: config[key] for key in keys if key in config}

    def _severity_rules(self, config: dict[str, object]) -> dict[str, object]:
        method = str(config["method"])
        if method == "iqr":
            return {
                "medium": f"score >= configured IQR multiplier ({config['iqr_multiplier']})",
                "high": IQR_HIGH_SEVERITY_SCORE,
                "critical": IQR_CRITICAL_SEVERITY_SCORE,
            }
        if method == "zscore":
            return {
                "medium": f"absolute score >= configured threshold ({config['threshold']})",
                "high": ZSCORE_HIGH_SEVERITY_SCORE,
                "critical": ZSCORE_CRITICAL_SEVERITY_SCORE,
            }
        if method == "robust_zscore":
            return {
                "medium": f"absolute score >= configured threshold ({config['threshold']})",
                "high": ROBUST_HIGH_SEVERITY_SCORE,
                "critical": ROBUST_CRITICAL_SEVERITY_SCORE,
            }
        threshold = float(config["threshold"])
        return {
            "medium": threshold,
            "high": threshold + ROLLING_HIGH_SEVERITY_OFFSET,
            "critical": threshold + ROLLING_CRITICAL_SEVERITY_OFFSET,
        }

    def _series_value(self, series: pd.Series | None, position: object) -> object:
        if series is None or position not in series.index:
            return None
        return self._finite_or_none(series.loc[position])

    def _finite_or_none(self, value: object) -> float | None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return numeric if math.isfinite(numeric) else None

    def _ratio(self, numerator: int | float, denominator: int | float) -> float:
        if denominator <= 0:
            return 0.0
        return round(float(numerator) / float(denominator), 6)

    def _safe_group_label(self, value: object) -> object:
        converted = self._json_safe(value)
        if isinstance(converted, (str, int, float, bool)) or converted is None:
            return converted
        return str(converted)

    def _chart_x(self, value: object) -> str | int:
        if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
            return int(value)
        return str(self._json_safe(value))

    def _is_missing(self, value: object) -> bool:
        try:
            return bool(pd.isna(value))
        except (TypeError, ValueError):
            return False

    def _input_error(
        self,
        message: str,
        *,
        error_code: str = "INVALID_ANOMALY_INPUT",
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
        if isinstance(value, (list, tuple, set)):
            return [self._json_safe(item) for item in value]
        if isinstance(value, np.ndarray):
            return [self._json_safe(item) for item in value.tolist()]
        if isinstance(value, pd.Index):
            return [self._json_safe(item) for item in value.tolist()]
        if isinstance(value, (pd.Timestamp, datetime, date)):
            return value.isoformat()
        if isinstance(value, (pd.Timedelta, np.timedelta64)):
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
