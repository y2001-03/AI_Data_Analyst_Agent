"""Enterprise descriptive distribution analysis for dataframe data."""

from __future__ import annotations

import json
import math
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Callable

import numpy as np
import pandas as pd

from app.core.exceptions import ToolExecutionException
from app.core.logging import get_logger
from app.schemas.file import AnalysisTask, ExecutionResult
from app.tools.dataframe_tools import BaseDataframeTool, DatasetContext

logger = get_logger(__name__)

DEFAULT_QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
MAX_QUANTILES = 20
DEFAULT_HISTOGRAM_BINS = 20
MIN_HISTOGRAM_BINS = 5
MAX_HISTOGRAM_BINS = 100
DEFAULT_TOP_K = 10
MIN_TOP_K = 1
MAX_TOP_K = 50

MIN_NUMERIC_SAMPLE_SIZE = 3
MIN_RELIABLE_SAMPLE_SIZE = 10
MIN_GROUP_SAMPLE_SIZE = 5
MAX_NUMERIC_COLUMNS = 30
MAX_CATEGORICAL_COLUMNS = 30
MAX_GROUP_COUNT = 100
MAX_TOP_CATEGORIES = 50
MAX_CHART_POINTS = 500
MAX_WARNINGS = 200
MAX_CATEGORY_DISPLAY_LENGTH = 200

SKEW_MODERATE_THRESHOLD = 0.5
SKEW_HIGH_THRESHOLD = 1.0
ZERO_INFLATED_RATIO_THRESHOLD = 0.5
HIGH_CONCENTRATION_IQR_RANGE_RATIO = 0.1
WIDE_DISPERSION_CV_THRESHOLD = 1.0
SPARSE_NUMERIC_MISSING_RATIO = 0.5
LONG_TAIL_DISTANCE_MULTIPLIER = 3.0
CV_NEAR_ZERO_EPSILON = 1e-12
BOXPLOT_IQR_MULTIPLIER = 1.5

RARE_CATEGORY_RATIO_THRESHOLD = 0.01
HIGH_CARDINALITY_UNIQUE_COUNT = 100
HIGH_CARDINALITY_RATIO_THRESHOLD = 0.5
HIGH_CARDINALITY_MIN_UNIQUE_COUNT = 20
HIGH_CATEGORY_CONCENTRATION_THRESHOLD = 0.8
LOW_CARDINALITY_UNIQUE_COUNT = 20
LOW_CARDINALITY_UNIQUE_RATIO = 0.05
LOW_CARDINALITY_SMALL_SAMPLE_SIZE = 100

SUPPORTED_MODES = {"numeric", "categorical", "group_comparison"}


@dataclass
class ColumnSelection:
    """Validated columns and bounded-selection metadata."""

    columns: list[str]
    excluded_columns: list[dict[str, str]]
    original_candidate_count: int
    requested: bool


class DistributionAnalysisTool(BaseDataframeTool):
    """Describe numeric, categorical, and grouped distributions safely."""

    name = "distribution_analysis_tool"
    tool_name = "distribution_analysis"
    result_type = "distribution_analysis"

    def __init__(self) -> None:
        self._mode_dispatchers: dict[
            str,
            Callable[[pd.DataFrame, dict[str, object], list[str]], dict[str, object]],
        ] = {
            "numeric": self._run_numeric,
            "categorical": self._run_categorical,
            "group_comparison": self._run_group_comparison,
        }

    def run(
        self,
        dataframe: pd.DataFrame,
        task: AnalysisTask,
        context: DatasetContext,
    ) -> ExecutionResult:
        """Return a bounded JSON-safe descriptive distribution report."""
        del context
        started_at = time.monotonic()
        summary = {
            "analyzed_column_count": 0,
            "excluded_column_count": 0,
            "analyzed_group_count": 0,
            "sample_count": 0,
        }
        mode = "unknown"
        try:
            if dataframe.empty:
                raise self._input_error(
                    "Distribution analysis requires a non-empty dataframe.",
                    "EMPTY_DISTRIBUTION_DATASET",
                )
            config = self._validated_config(dataframe, task.params or {})
            mode = str(config["mode"])
            warnings: list[str] = []
            report = self._mode_dispatchers[mode](dataframe, config, warnings)
            report_summary = report["summary"]
            for key in summary:
                summary[key] = int(report_summary.get(key, summary[key]))
        except ToolExecutionException:
            self._log_execution(mode, summary, "failed", started_at)
            raise
        except Exception as exc:
            self._log_execution(mode, summary, "failed", started_at)
            raise ToolExecutionException(
                "Distribution analysis failed.",
                details={"tool_name": self.name},
            ) from exc

        self._log_execution(mode, summary, "success", started_at)
        return ExecutionResult(
            task_name=task.task_name,
            type=self.result_type,
            data=self._json_safe(report),
            chart=None,
        )

    def _validated_config(
        self,
        dataframe: pd.DataFrame,
        params: dict[str, object],
    ) -> dict[str, object]:
        config = dict(params)
        raw_mode = config.get("mode")
        if raw_mode is None:
            mode = self._default_mode(dataframe)
        elif not isinstance(raw_mode, str) or raw_mode.lower() not in SUPPORTED_MODES:
            raise self._input_error(
                "Distribution analysis mode must be 'numeric', 'categorical', or 'group_comparison'.",
                "INVALID_DISTRIBUTION_MODE",
            )
        else:
            mode = raw_mode.lower()
        config["mode"] = mode

        if mode == "numeric":
            config["quantiles"] = self._validated_quantiles(config.get("quantiles"))
            config["bins"] = self._bounded_integer(
                config.get("bins", DEFAULT_HISTOGRAM_BINS),
                "bins",
                MIN_HISTOGRAM_BINS,
                MAX_HISTOGRAM_BINS,
            )
        elif mode == "categorical":
            config["top_k"] = self._bounded_integer(
                config.get("top_k", DEFAULT_TOP_K),
                "top_k",
                MIN_TOP_K,
                MAX_TOP_K,
            )
            allow_numeric = config.get("allow_low_cardinality_numeric", False)
            if not isinstance(allow_numeric, bool):
                raise self._input_error(
                    "allow_low_cardinality_numeric must be boolean.",
                    "INVALID_LOW_CARDINALITY_OPTION",
                )
            config["allow_low_cardinality_numeric"] = allow_numeric
        else:
            group_by = config.get("group_by")
            if not isinstance(group_by, str) or not group_by:
                raise self._input_error(
                    "Group comparison requires a non-empty group_by column.",
                    "MISSING_DISTRIBUTION_GROUP_BY",
                )
            if group_by not in dataframe.columns:
                raise self._input_error(
                    f"Unknown group_by column '{group_by}'.",
                    "UNKNOWN_DISTRIBUTION_GROUP_BY",
                )
            if self._type_name(dataframe[group_by]) in {"datetime", "timedelta", "unsupported"}:
                raise self._input_error(
                    "group_by must be a categorical, boolean, text, or discrete numeric column.",
                    "INVALID_DISTRIBUTION_GROUP_BY",
                )
        return config

    def _default_mode(self, dataframe: pd.DataFrame) -> str:
        if any(
            self._numeric_exclusion_reason(dataframe[column]) is None
            and self._finite_numeric(dataframe[column])[0].notna().any()
            for column in dataframe.columns
        ):
            return "numeric"
        if any(
            dataframe[column].notna().any()
            and self._categorical_exclusion_reason(dataframe[column], False) is None
            for column in dataframe.columns
        ):
            return "categorical"
        raise self._input_error(
            "Distribution analysis found no supported numeric or categorical columns.",
            "NO_DISTRIBUTION_COLUMNS",
        )

    def _run_numeric(
        self,
        dataframe: pd.DataFrame,
        config: dict[str, object],
        warnings: list[str],
    ) -> dict[str, object]:
        selection = self._select_numeric_columns(
            dataframe,
            config.get("columns"),
            excluded_names=set(),
            warnings=warnings,
            limit=MAX_NUMERIC_COLUMNS,
        )
        results: list[dict[str, object]] = []
        for column in selection.columns:
            try:
                results.append(
                    self._numeric_column_report(
                        dataframe[column],
                        column,
                        list(config["quantiles"]),
                        int(config["bins"]),
                        warnings,
                    )
                )
            except (ArithmeticError, TypeError, ValueError) as exc:
                self._add_warning(warnings, f"Column '{column}' could not be analyzed: {exc}")
                self._log_unit_failure("numeric")
        if not results:
            raise self._input_error(
                "Distribution analysis found no numeric columns with finite values.",
                "NO_NUMERIC_DISTRIBUTION_COLUMNS",
            )

        chart_column = self._select_numeric_chart_column(results, selection.columns)
        selected = next(result for result in results if result["column"] == chart_column)
        chart = {
            "type": "histogram",
            "column": chart_column,
            "bin_edges": selected["histogram"]["bin_edges"],
            "counts": selected["histogram"]["counts"],
        }
        summary = {
            "analyzed_column_count": len(results),
            "excluded_column_count": len(selection.excluded_columns),
            "reliable_column_count": sum(bool(result["reliable"]) for result in results),
            "highly_skewed_column_count": sum(
                result["shape"]["skew_level"] == "highly_skewed" for result in results
            ),
            "long_tail_candidate_count": sum(
                bool(result["shape"]["long_tail_candidate"]) for result in results
            ),
            "sample_count": sum(int(result["sample_size"]) for result in results),
            "analyzed_group_count": 0,
        }
        return self._base_report(
            "numeric",
            summary,
            results,
            selection,
            warnings,
            chart,
            {
                "selected_chart_column": chart_column,
                "chart_selection_rule": "highest_absolute_skewness_then_sample_size_then_column_order",
                "quantiles": config["quantiles"],
                "histogram_bins": config["bins"],
            },
        )

    def _numeric_column_report(
        self,
        source: pd.Series,
        column: str,
        quantiles: list[float],
        bins: int,
        report_warnings: list[str],
    ) -> dict[str, object]:
        numeric, infinity_count = self._finite_numeric(source)
        valid = numeric.dropna()
        sample_size = int(len(valid))
        row_count = int(len(source))
        missing_count = row_count - sample_size
        unique_count = int(valid.nunique(dropna=True))
        constant = bool(sample_size > 0 and unique_count <= 1)
        reliable = sample_size >= MIN_RELIABLE_SAMPLE_SIZE
        warnings: list[str] = []
        if infinity_count:
            message = (
                f"Column '{column}' contains {infinity_count} infinite values; "
                "they were treated as missing."
            )
            warnings.append(message)
            self._add_warning(report_warnings, message)
        if sample_size < MIN_NUMERIC_SAMPLE_SIZE:
            warnings.append(
                f"Only {sample_size} finite samples are available; only limited statistics are reliable."
            )
        elif not reliable:
            warnings.append(
                f"Only {sample_size} finite samples are available; higher-order shape labels are suppressed."
            )

        mean = self._finite_or_none(valid.mean()) if sample_size else None
        median = self._finite_or_none(valid.median()) if sample_size else None
        minimum = self._finite_or_none(valid.min()) if sample_size else None
        maximum = self._finite_or_none(valid.max()) if sample_size else None
        std = self._finite_or_none(valid.std(ddof=1)) if sample_size >= 2 else None
        variance = self._finite_or_none(valid.var(ddof=1)) if sample_size >= 2 else None
        value_range = maximum - minimum if minimum is not None and maximum is not None else None
        computed_quantiles = {
            self._quantile_key(quantile): self._finite_or_none(valid.quantile(quantile))
            for quantile in quantiles
        }
        q1 = self._finite_or_none(valid.quantile(0.25)) if sample_size else None
        q3 = self._finite_or_none(valid.quantile(0.75)) if sample_size else None
        iqr = q3 - q1 if q1 is not None and q3 is not None else None

        if constant:
            skewness = None
            kurtosis = None
            warnings.append("The column is constant; skewness and kurtosis are not meaningful.")
        else:
            skewness = self._finite_or_none(valid.skew()) if sample_size >= 3 else None
            kurtosis = self._finite_or_none(valid.kurt()) if sample_size >= 4 else None

        coefficient_of_variation = None
        if std is not None and mean is not None:
            if abs(mean) <= CV_NEAR_ZERO_EPSILON:
                warnings.append(
                    "Coefficient of variation is unavailable because the mean is zero or near zero."
                )
            else:
                coefficient_of_variation = self._finite_or_none(std / abs(mean))

        zero_count = int((valid == 0).sum())
        negative_count = int((valid < 0).sum())
        histogram = self._histogram(valid, bins)
        boxplot = self._boxplot(valid)
        shape = self._numeric_shape(
            valid,
            reliable,
            constant,
            skewness,
            coefficient_of_variation,
            iqr,
            value_range,
            zero_count,
            missing_count,
            row_count,
        )
        return {
            "column": column,
            "inferred_type": "numeric",
            "count": sample_size,
            "sample_size": sample_size,
            "missing_count": missing_count,
            "missing_ratio": self._ratio(missing_count, row_count),
            "reliable": reliable,
            "constant_column": constant,
            "statistics": {
                "mean": mean,
                "median": median,
                "min": minimum,
                "max": maximum,
                "std": std,
                "variance": variance,
                "range": value_range,
                "coefficient_of_variation": coefficient_of_variation,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "skewness": skewness,
                "kurtosis": kurtosis,
                "zero_count": zero_count,
                "zero_ratio": self._ratio(zero_count, sample_size),
                "negative_count": negative_count,
                "negative_ratio": self._ratio(negative_count, sample_size),
                "unique_count": unique_count,
                "unique_ratio": self._ratio(unique_count, sample_size),
            },
            "quantiles": computed_quantiles,
            "shape": shape,
            "histogram": histogram,
            "boxplot": boxplot,
            "warnings": warnings,
        }

    def _numeric_shape(
        self,
        valid: pd.Series,
        reliable: bool,
        constant: bool,
        skewness: float | None,
        coefficient_of_variation: float | None,
        iqr: float | None,
        value_range: float | None,
        zero_count: int,
        missing_count: int,
        row_count: int,
    ) -> dict[str, object]:
        if not reliable or skewness is None:
            direction = "unavailable"
            level = "unavailable"
        elif abs(skewness) < SKEW_MODERATE_THRESHOLD:
            direction = "none"
            level = "approximately_symmetric"
        else:
            direction = "positive" if skewness > 0 else "negative"
            level = (
                "highly_skewed"
                if abs(skewness) >= SKEW_HIGH_THRESHOLD
                else "moderately_skewed"
            )

        long_tail = False
        if reliable and skewness is not None and abs(skewness) >= SKEW_HIGH_THRESHOLD and len(valid):
            median = float(valid.median())
            p95 = float(valid.quantile(0.95))
            p99 = float(valid.quantile(0.99))
            scale = max(abs(float(iqr or 0.0)), abs(median) * 0.1, CV_NEAR_ZERO_EPSILON)
            long_tail = max(abs(p95 - median), abs(p99 - median)) >= (
                LONG_TAIL_DISTANCE_MULTIPLIER * scale
            )

        concentrated = bool(
            reliable
            and value_range is not None
            and value_range > 0
            and iqr is not None
            and (iqr / value_range) <= HIGH_CONCENTRATION_IQR_RANGE_RATIO
        )
        return {
            "skew_direction": direction,
            "skew_level": level,
            "shape_reliable": reliable,
            "long_tail_candidate": bool(long_tail),
            "zero_inflated_candidate": bool(
                reliable and self._ratio(zero_count, len(valid)) >= ZERO_INFLATED_RATIO_THRESHOLD
            ),
            "highly_concentrated": concentrated,
            "wide_dispersion": bool(
                reliable
                and coefficient_of_variation is not None
                and coefficient_of_variation >= WIDE_DISPERSION_CV_THRESHOLD
            ),
            "constant": constant,
            "sparse_numeric": bool(
                self._ratio(missing_count, row_count) >= SPARSE_NUMERIC_MISSING_RATIO
                or len(valid) < MIN_RELIABLE_SAMPLE_SIZE
            ),
        }

    def _histogram(self, valid: pd.Series, bins: int) -> dict[str, object]:
        counts, edges = np.histogram(valid.to_numpy(dtype=float), bins=bins)
        return {
            "bin_edges": [self._finite_or_none(value) for value in edges.tolist()],
            "counts": [int(value) for value in counts.tolist()],
            "bin_count": bins,
        }

    def _boxplot(self, valid: pd.Series) -> dict[str, object]:
        q1 = float(valid.quantile(0.25))
        median = float(valid.quantile(0.5))
        q3 = float(valid.quantile(0.75))
        iqr = q3 - q1
        lower_fence = q1 - BOXPLOT_IQR_MULTIPLIER * iqr
        upper_fence = q3 + BOXPLOT_IQR_MULTIPLIER * iqr
        within = valid[(valid >= lower_fence) & (valid <= upper_fence)]
        outlier_count = int(((valid < lower_fence) | (valid > upper_fence)).sum())
        return {
            "q1": q1,
            "median": median,
            "q3": q3,
            "iqr": iqr,
            "lower_whisker": self._finite_or_none(within.min()) if not within.empty else None,
            "upper_whisker": self._finite_or_none(within.max()) if not within.empty else None,
            "lower_fence": lower_fence,
            "upper_fence": upper_fence,
            "outlier_count": outlier_count,
            "outlier_ratio": self._ratio(outlier_count, len(valid)),
            "candidate_extreme_values_only": True,
        }

    def _run_categorical(
        self,
        dataframe: pd.DataFrame,
        config: dict[str, object],
        warnings: list[str],
    ) -> dict[str, object]:
        selection = self._select_categorical_columns(
            dataframe,
            config.get("columns"),
            bool(config["allow_low_cardinality_numeric"]),
            warnings,
        )
        results: list[dict[str, object]] = []
        for column in selection.columns:
            try:
                results.append(
                    self._categorical_column_report(
                        dataframe[column], column, int(config["top_k"])
                    )
                )
            except (ArithmeticError, TypeError, ValueError) as exc:
                self._add_warning(warnings, f"Column '{column}' could not be analyzed: {exc}")
                self._log_unit_failure("categorical")
        if not results:
            raise self._input_error(
                "Distribution analysis found no categorical columns with non-missing values.",
                "NO_CATEGORICAL_DISTRIBUTION_COLUMNS",
            )

        chart_column = self._select_categorical_chart_column(results, selection.columns)
        selected = next(result for result in results if result["column"] == chart_column)
        chart = {
            "type": "category_bar",
            "column": chart_column,
            "categories": [item["value"] for item in selected["top_categories"]],
            "counts": [item["count"] for item in selected["top_categories"]],
        }
        summary = {
            "analyzed_column_count": len(results),
            "excluded_column_count": len(selection.excluded_columns),
            "reliable_column_count": sum(bool(result["reliable"]) for result in results),
            "high_cardinality_column_count": sum(
                bool(result["high_cardinality"]) for result in results
            ),
            "constant_column_count": sum(bool(result["constant_column"]) for result in results),
            "sample_count": sum(int(result["sample_size"]) for result in results),
            "analyzed_group_count": 0,
        }
        return self._base_report(
            "categorical",
            summary,
            results,
            selection,
            warnings,
            chart,
            {
                "selected_chart_column": chart_column,
                "chart_selection_rule": "non_high_cardinality_then_sample_size_then_column_order",
                "top_k": config["top_k"],
                "allow_low_cardinality_numeric": config["allow_low_cardinality_numeric"],
            },
        )

    def _categorical_column_report(
        self,
        source: pd.Series,
        column: str,
        top_k: int,
    ) -> dict[str, object]:
        row_count = int(len(source))
        values = [value for value in source.tolist() if not self._is_missing(value)]
        sample_size = len(values)
        missing_count = row_count - sample_size
        categories = self._stable_categories(values)
        ordered = sorted(categories.values(), key=lambda item: (-item["count"], item["first"]))
        unique_count = len(ordered)
        unique_ratio = self._ratio(unique_count, sample_size)
        mode_item = ordered[0]
        top_items = ordered[:top_k]
        other_count = sum(int(item["count"]) for item in ordered[top_k:])
        probabilities = [float(item["count"]) / sample_size for item in ordered]
        entropy = -sum(probability * math.log2(probability) for probability in probabilities)
        normalized_entropy = entropy / math.log2(unique_count) if unique_count > 1 else 0.0
        normalized_entropy = max(0.0, min(1.0, normalized_entropy))
        rare = [item for item in ordered if (int(item["count"]) / sample_size) < RARE_CATEGORY_RATIO_THRESHOLD]
        rare_value_count = sum(int(item["count"]) for item in rare)
        high_cardinality = bool(
            unique_count >= HIGH_CARDINALITY_UNIQUE_COUNT
            or (
                unique_count >= HIGH_CARDINALITY_MIN_UNIQUE_COUNT
                and unique_ratio >= HIGH_CARDINALITY_RATIO_THRESHOLD
            )
        )
        top1 = probabilities[0]
        top3 = sum(probabilities[:3])
        mode_value, mode_value_truncated = self._display_category(mode_item["value"])
        warnings: list[str] = []
        if high_cardinality:
            warnings.append(
                "High cardinality is descriptive, not an error; check whether this field is an ID, serial number, or free text."
            )
        if sample_size < MIN_RELIABLE_SAMPLE_SIZE:
            warnings.append(
                f"Only {sample_size} non-missing samples are available; concentration labels are low confidence."
            )
        return {
            "column": column,
            "inferred_type": self._type_name(source),
            "count": sample_size,
            "sample_size": sample_size,
            "missing_count": missing_count,
            "missing_ratio": self._ratio(missing_count, row_count),
            "unique_count": unique_count,
            "unique_ratio": unique_ratio,
            "mode": mode_value,
            "mode_value_truncated": mode_value_truncated,
            "mode_count": int(mode_item["count"]),
            "mode_ratio": self._ratio(int(mode_item["count"]), sample_size),
            "top_categories": [self._category_output(item, sample_size) for item in top_items],
            "other_count": other_count,
            "other_ratio": self._ratio(other_count, sample_size),
            "rare_category_count": len(rare),
            "rare_category_ratio": self._ratio(len(rare), unique_count),
            "rare_value_count": rare_value_count,
            "rare_value_ratio": self._ratio(rare_value_count, sample_size),
            "entropy": entropy,
            "normalized_entropy": normalized_entropy,
            "concentration_ratio_top1": top1,
            "concentration_ratio_top3": top3,
            "high_cardinality": high_cardinality,
            "highly_concentrated": bool(
                sample_size >= MIN_RELIABLE_SAMPLE_SIZE
                and top1 >= HIGH_CATEGORY_CONCENTRATION_THRESHOLD
            ),
            "constant_column": unique_count == 1,
            "reliable": sample_size >= MIN_RELIABLE_SAMPLE_SIZE,
            "warnings": warnings,
        }

    def _stable_categories(self, values: list[object]) -> OrderedDict[str, dict[str, object]]:
        categories: OrderedDict[str, dict[str, object]] = OrderedDict()
        for position, value in enumerate(values):
            safe_value = self._json_safe(value)
            token = f"{type(value).__name__}:{json.dumps(safe_value, ensure_ascii=False, sort_keys=True)}"
            if token not in categories:
                categories[token] = {"value": safe_value, "count": 0, "first": position}
            categories[token]["count"] = int(categories[token]["count"]) + 1
        return categories

    def _category_output(
        self,
        item: dict[str, object],
        sample_size: int,
    ) -> dict[str, object]:
        value, truncated = self._display_category(item["value"])
        return {
            "value": value,
            "value_truncated": truncated,
            "count": int(item["count"]),
            "ratio": self._ratio(int(item["count"]), sample_size),
        }

    def _display_category(self, value: object) -> tuple[object, bool]:
        if isinstance(value, str) and len(value) > MAX_CATEGORY_DISPLAY_LENGTH:
            return value[:MAX_CATEGORY_DISPLAY_LENGTH], True
        return value, False

    def _run_group_comparison(
        self,
        dataframe: pd.DataFrame,
        config: dict[str, object],
        warnings: list[str],
    ) -> dict[str, object]:
        group_by = str(config["group_by"])
        selection = self._select_numeric_columns(
            dataframe,
            config.get("columns"),
            excluded_names={group_by},
            warnings=warnings,
            limit=MAX_NUMERIC_COLUMNS,
        )
        groups = self._select_groups(dataframe[group_by], warnings)
        column_results: list[dict[str, object]] = []
        for column in selection.columns:
            group_summaries: list[dict[str, object]] = []
            for group in groups["selected"]:
                positions = group["positions"]
                source = dataframe[column].iloc[positions]
                numeric, infinity_count = self._finite_numeric(source)
                valid = numeric.dropna()
                sample_size = int(len(valid))
                group_size = len(positions)
                reliable = sample_size >= MIN_RELIABLE_SAMPLE_SIZE
                if sample_size < MIN_GROUP_SAMPLE_SIZE:
                    self._add_warning(
                        warnings,
                        f"A group in '{group_by}' has only {sample_size} valid samples for '{column}'; complex statistics are unreliable.",
                    )
                if infinity_count:
                    self._add_warning(
                        warnings,
                        f"A group in '{group_by}' contains infinite '{column}' values; they were treated as missing.",
                    )
                group_summaries.append(
                    self._group_numeric_summary(
                        group["label"], valid, group_size, reliable
                    )
                )
            column_results.append(
                self._group_column_report(column, group_summaries)
            )

        chart_column = self._select_group_chart_column(
            column_results, selection.columns, selection.requested
        )
        chart_result = next(
            result for result in column_results if result["column"] == chart_column
        )
        chart_groups = chart_result["group_summary"][:MAX_CHART_POINTS]
        chart = {
            "type": "group_boxplot_summary",
            "column": chart_column,
            "groups": [item["group"] for item in chart_groups],
            "q1": [item["q1"] for item in chart_groups],
            "median": [item["median"] for item in chart_groups],
            "q3": [item["q3"] for item in chart_groups],
            "lower_whisker": [item["lower_whisker"] for item in chart_groups],
            "upper_whisker": [item["upper_whisker"] for item in chart_groups],
        }
        summary = {
            "analyzed_column_count": len(column_results),
            "excluded_column_count": len(selection.excluded_columns),
            "group_count": groups["original_count"],
            "analyzed_group_count": len(groups["selected"]),
            "skipped_groups": groups["original_count"] - len(groups["selected"]),
            "sample_count": sum(
                int(item["sample_size"])
                for result in column_results
                for item in result["group_summary"]
            ),
        }
        return self._base_report(
            "group_comparison",
            summary,
            column_results,
            selection,
            warnings,
            chart,
            {
                "group_by": group_by,
                "missing_group_label": "__MISSING__",
                "original_group_count": groups["original_count"],
                "actual_group_count": len(groups["selected"]),
                "group_selection_rule": "sample_size_descending_then_first_appearance",
                "selected_chart_column": chart_column,
                "chart_selection_rule": (
                    "first_explicit_column"
                    if selection.requested
                    else "largest_group_median_range_then_column_order"
                ),
            },
        )

    def _select_groups(
        self,
        series: pd.Series,
        warnings: list[str],
    ) -> dict[str, object]:
        groups: OrderedDict[str, dict[str, object]] = OrderedDict()
        for position, value in enumerate(series.tolist()):
            label: object = "__MISSING__" if self._is_missing(value) else self._json_safe(value)
            token = json.dumps(label, ensure_ascii=False, sort_keys=True)
            if token not in groups:
                groups[token] = {
                    "label": label,
                    "positions": [],
                    "first": position,
                }
            groups[token]["positions"].append(position)
        ordered = sorted(
            groups.values(),
            key=lambda group: (-len(group["positions"]), int(group["first"])),
        )
        if len(ordered) > MAX_GROUP_COUNT:
            self._add_warning(
                warnings,
                f"Group count exceeded {MAX_GROUP_COUNT}; the largest groups were selected deterministically.",
            )
        return {
            "original_count": len(ordered),
            "selected": ordered[:MAX_GROUP_COUNT],
        }

    def _group_numeric_summary(
        self,
        label: object,
        valid: pd.Series,
        group_size: int,
        reliable: bool,
    ) -> dict[str, object]:
        sample_size = int(len(valid))
        if not sample_size:
            return {
                "group": label,
                "sample_size": 0,
                "missing_count": group_size,
                "reliable": False,
                "mean": None,
                "median": None,
                "std": None,
                "q1": None,
                "q3": None,
                "iqr": None,
                "min": None,
                "max": None,
                "skewness": None,
                "outlier_count": 0,
                "lower_whisker": None,
                "upper_whisker": None,
            }
        boxplot = self._boxplot(valid)
        q1 = float(valid.quantile(0.25))
        q3 = float(valid.quantile(0.75))
        constant = int(valid.nunique(dropna=True)) <= 1
        return {
            "group": label,
            "sample_size": sample_size,
            "missing_count": group_size - sample_size,
            "reliable": reliable,
            "mean": self._finite_or_none(valid.mean()),
            "median": self._finite_or_none(valid.median()),
            "std": self._finite_or_none(valid.std(ddof=1)) if sample_size >= 2 else None,
            "q1": q1,
            "q3": q3,
            "iqr": q3 - q1,
            "min": self._finite_or_none(valid.min()),
            "max": self._finite_or_none(valid.max()),
            "skewness": (
                self._finite_or_none(valid.skew())
                if reliable and sample_size >= 3 and not constant
                else None
            ),
            "outlier_count": boxplot["outlier_count"],
            "lower_whisker": boxplot["lower_whisker"],
            "upper_whisker": boxplot["upper_whisker"],
        }

    def _group_column_report(
        self,
        column: str,
        groups: list[dict[str, object]],
    ) -> dict[str, object]:
        usable = [group for group in groups if group["sample_size"]]
        return {
            "column": column,
            "inferred_type": "numeric",
            "group_summary": groups,
            "highest_mean_group": self._extreme_group(usable, "mean", True),
            "lowest_mean_group": self._extreme_group(usable, "mean", False),
            "highest_median_group": self._extreme_group(usable, "median", True),
            "widest_iqr_group": self._extreme_group(usable, "iqr", True),
            "group_median_range": self._value_range(usable, "median"),
        }

    def _extreme_group(
        self,
        groups: list[dict[str, object]],
        field: str,
        highest: bool,
    ) -> object:
        candidates = [group for group in groups if group[field] is not None]
        if not candidates:
            return None
        selected = max(candidates, key=lambda group: float(group[field])) if highest else min(
            candidates, key=lambda group: float(group[field])
        )
        return selected["group"]

    def _value_range(self, groups: list[dict[str, object]], field: str) -> float | None:
        values = [float(group[field]) for group in groups if group[field] is not None]
        return max(values) - min(values) if values else None

    def _select_numeric_columns(
        self,
        dataframe: pd.DataFrame,
        raw_columns: object,
        excluded_names: set[str],
        warnings: list[str],
        limit: int,
    ) -> ColumnSelection:
        requested = self._requested_columns(raw_columns)
        if requested is not None and len(requested) > limit:
            raise self._input_error(
                f"Explicit distribution analysis supports at most {limit} numeric columns.",
                "DISTRIBUTION_COLUMN_LIMIT_EXCEEDED",
            )
        excluded: list[dict[str, str]] = []
        if requested is not None:
            self._validate_known_columns(dataframe, requested)
            blocked = [column for column in requested if column in excluded_names]
            if blocked:
                raise self._input_error(
                    f"Grouping columns cannot also be distribution value columns: {', '.join(blocked)}.",
                    "INVALID_DISTRIBUTION_COLUMNS",
                )
            invalid = [
                {"column": column, "reason": self._numeric_exclusion_reason(dataframe[column])}
                for column in requested
                if self._numeric_exclusion_reason(dataframe[column]) is not None
            ]
            if invalid:
                raise self._input_error(
                    "Explicit numeric distribution columns must be numeric: "
                    + ", ".join(f"{item['column']} ({item['reason']})" for item in invalid)
                    + ".",
                    "NON_NUMERIC_DISTRIBUTION_COLUMNS",
                )
            candidates = list(requested)
        else:
            candidates = []
            for column in dataframe.columns:
                name = str(column)
                if name in excluded_names:
                    excluded.append({"column": name, "reason": "group_by"})
                    continue
                if dataframe[column].isna().all():
                    excluded.append({"column": name, "reason": "all_null_or_non_finite"})
                    continue
                reason = self._numeric_exclusion_reason(dataframe[column])
                if reason is None:
                    candidates.append(name)
                else:
                    excluded.append({"column": name, "reason": reason})

        eligible: list[str] = []
        values: dict[str, pd.Series] = {}
        for column in candidates:
            numeric, infinity_count = self._finite_numeric(dataframe[column])
            values[column] = numeric
            if infinity_count:
                self._add_warning(
                    warnings,
                    f"Column '{column}' contains {infinity_count} infinite values; they were treated as missing.",
                )
            if numeric.notna().sum() == 0:
                excluded.append({"column": column, "reason": "all_null_or_non_finite"})
            else:
                eligible.append(column)
        original_count = len(eligible)
        if requested is None and len(eligible) > limit:
            eligible, omitted = self._limit_numeric_columns(values, eligible, limit)
            excluded.extend({"column": column, "reason": "column_limit"} for column in omitted)
            self._add_warning(
                warnings,
                f"Numeric column count exceeded {limit}; columns were selected by finite count, variance, and original order.",
            )
        if not eligible:
            raise self._input_error(
                "Distribution analysis found no numeric columns with finite values.",
                "NO_NUMERIC_DISTRIBUTION_COLUMNS",
            )
        return ColumnSelection(eligible, excluded, original_count, requested is not None)

    def _select_categorical_columns(
        self,
        dataframe: pd.DataFrame,
        raw_columns: object,
        allow_numeric: bool,
        warnings: list[str],
    ) -> ColumnSelection:
        requested = self._requested_columns(raw_columns)
        if requested is not None and len(requested) > MAX_CATEGORICAL_COLUMNS:
            raise self._input_error(
                f"Explicit distribution analysis supports at most {MAX_CATEGORICAL_COLUMNS} categorical columns.",
                "DISTRIBUTION_COLUMN_LIMIT_EXCEEDED",
            )
        excluded: list[dict[str, str]] = []
        if requested is not None:
            self._validate_known_columns(dataframe, requested)
            invalid = []
            for column in requested:
                reason = self._categorical_exclusion_reason(dataframe[column], allow_numeric)
                if reason is not None:
                    invalid.append({"column": column, "reason": reason})
            if invalid:
                raise self._input_error(
                    "Explicit categorical distribution columns are unsupported: "
                    + ", ".join(f"{item['column']} ({item['reason']})" for item in invalid)
                    + ".",
                    "NON_CATEGORICAL_DISTRIBUTION_COLUMNS",
                )
            candidates = list(requested)
        else:
            candidates = []
            for column in dataframe.columns:
                name = str(column)
                if dataframe[column].isna().all():
                    excluded.append({"column": name, "reason": "all_null"})
                    continue
                reason = self._categorical_exclusion_reason(dataframe[column], False)
                if reason is None:
                    candidates.append(name)
                else:
                    excluded.append({"column": name, "reason": reason})

        eligible: list[str] = []
        for column in candidates:
            if dataframe[column].notna().sum() == 0:
                excluded.append({"column": column, "reason": "all_null"})
            else:
                eligible.append(column)
        original_count = len(eligible)
        if requested is None and len(eligible) > MAX_CATEGORICAL_COLUMNS:
            eligible, omitted = self._limit_categorical_columns(
                dataframe, eligible, MAX_CATEGORICAL_COLUMNS
            )
            excluded.extend({"column": column, "reason": "column_limit"} for column in omitted)
            self._add_warning(
                warnings,
                f"Categorical column count exceeded {MAX_CATEGORICAL_COLUMNS}; columns were selected by non-missing count, cardinality, and original order.",
            )
        if not eligible:
            raise self._input_error(
                "Distribution analysis found no categorical columns with non-missing values.",
                "NO_CATEGORICAL_DISTRIBUTION_COLUMNS",
            )
        return ColumnSelection(eligible, excluded, original_count, requested is not None)

    def _requested_columns(self, raw_columns: object) -> list[str] | None:
        if raw_columns is None:
            return None
        if (
            not isinstance(raw_columns, list)
            or not raw_columns
            or not all(isinstance(column, str) and column for column in raw_columns)
        ):
            raise self._input_error(
                "columns must be a non-empty list of column names.",
                "INVALID_DISTRIBUTION_COLUMNS",
            )
        if len(set(raw_columns)) != len(raw_columns):
            raise self._input_error(
                "columns must not contain duplicates.",
                "INVALID_DISTRIBUTION_COLUMNS",
            )
        return list(raw_columns)

    def _validate_known_columns(self, dataframe: pd.DataFrame, columns: list[str]) -> None:
        unknown = [column for column in columns if column not in dataframe.columns]
        if unknown:
            raise self._input_error(
                f"Unknown distribution column(s): {', '.join(unknown)}.",
                "UNKNOWN_DISTRIBUTION_COLUMNS",
            )

    def _numeric_exclusion_reason(self, series: pd.Series) -> str | None:
        type_name = self._type_name(series)
        return None if type_name == "numeric" else type_name

    def _categorical_exclusion_reason(
        self,
        series: pd.Series,
        allow_numeric: bool,
    ) -> str | None:
        type_name = self._type_name(series)
        if type_name in {"boolean", "text", "category"}:
            return None
        if type_name == "numeric" and allow_numeric and self._is_low_cardinality_numeric(series):
            return None
        if type_name == "numeric" and allow_numeric:
            return "numeric_not_low_cardinality"
        return type_name

    def _type_name(self, series: pd.Series) -> str:
        if pd.api.types.is_bool_dtype(series):
            return "boolean"
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        if pd.api.types.is_timedelta64_dtype(series):
            return "timedelta"
        if pd.api.types.is_numeric_dtype(series):
            return "complex" if pd.api.types.is_complex_dtype(series) else "numeric"
        if isinstance(series.dtype, pd.CategoricalDtype):
            return "category"
        if pd.api.types.is_string_dtype(series.dtype):
            return "text"
        if pd.api.types.is_object_dtype(series):
            return "text"
        return "unsupported"

    def _is_low_cardinality_numeric(self, series: pd.Series) -> bool:
        numeric, _ = self._finite_numeric(series)
        sample_size = int(numeric.notna().sum())
        unique_count = int(numeric.nunique(dropna=True))
        return bool(
            sample_size > 0
            and unique_count <= LOW_CARDINALITY_UNIQUE_COUNT
            and (
                self._ratio(unique_count, sample_size) <= LOW_CARDINALITY_UNIQUE_RATIO
                or sample_size < LOW_CARDINALITY_SMALL_SAMPLE_SIZE
            )
        )

    def _limit_numeric_columns(
        self,
        values: dict[str, pd.Series],
        columns: list[str],
        limit: int,
    ) -> tuple[list[str], list[str]]:
        position = {column: index for index, column in enumerate(columns)}

        def key(column: str) -> tuple[float, float, int]:
            series = values[column]
            count = int(series.notna().sum())
            variance = float(series.var(ddof=1)) if count >= 2 else float("-inf")
            if not math.isfinite(variance):
                variance = float("-inf")
            return (-count, -variance, position[column])

        ordered = sorted(columns, key=key)
        selected = set(ordered[:limit])
        return (
            [column for column in columns if column in selected],
            [column for column in columns if column not in selected],
        )

    def _limit_categorical_columns(
        self,
        dataframe: pd.DataFrame,
        columns: list[str],
        limit: int,
    ) -> tuple[list[str], list[str]]:
        position = {column: index for index, column in enumerate(columns)}
        ordered = sorted(
            columns,
            key=lambda column: (
                -int(dataframe[column].notna().sum()),
                int(dataframe[column].nunique(dropna=True)),
                position[column],
            ),
        )
        selected = set(ordered[:limit])
        return (
            [column for column in columns if column in selected],
            [column for column in columns if column not in selected],
        )

    def _select_numeric_chart_column(
        self,
        results: list[dict[str, object]],
        columns: list[str],
    ) -> str:
        position = {column: index for index, column in enumerate(columns)}

        def key(result: dict[str, object]) -> tuple[float, int, int]:
            skewness = result["statistics"]["skewness"]
            score = abs(float(skewness)) if skewness is not None else -1.0
            return (-score, -int(result["sample_size"]), position[str(result["column"])])

        return str(min(results, key=key)["column"])

    def _select_categorical_chart_column(
        self,
        results: list[dict[str, object]],
        columns: list[str],
    ) -> str:
        position = {column: index for index, column in enumerate(columns)}
        selected = min(
            results,
            key=lambda result: (
                bool(result["high_cardinality"]),
                -int(result["sample_size"]),
                position[str(result["column"])],
            ),
        )
        return str(selected["column"])

    def _select_group_chart_column(
        self,
        results: list[dict[str, object]],
        columns: list[str],
        requested: bool,
    ) -> str:
        if requested:
            return columns[0]
        position = {column: index for index, column in enumerate(columns)}
        selected = min(
            results,
            key=lambda result: (
                -float(result["group_median_range"] or 0.0),
                position[str(result["column"])],
            ),
        )
        return str(selected["column"])

    def _base_report(
        self,
        mode: str,
        summary: dict[str, object],
        columns: list[dict[str, object]],
        selection: ColumnSelection,
        warnings: list[str],
        chart: dict[str, object],
        mode_metadata: dict[str, object],
    ) -> dict[str, object]:
        return {
            "tool_name": self.tool_name,
            "mode": mode,
            "summary": summary,
            "columns": columns,
            "excluded_columns": selection.excluded_columns,
            "warnings": warnings,
            "metadata": {
                "descriptive_analysis_only": True,
                "not_a_formal_normality_test": True,
                "heuristic_thresholds": True,
                "no_causal_conclusions": True,
                "distribution_labels_are_not_data_quality_errors": True,
                "original_dataframe_modified": False,
                "skewness_definition": "pandas unbiased sample skewness",
                "kurtosis_definition": "Fisher excess kurtosis; a normal distribution is approximately 0",
                "coefficient_of_variation_definition": "sample_std / abs(mean); unavailable near zero mean",
                "boxplot_rule": "1.5 * IQR candidate extreme-value summary only",
                "sample_size_thresholds": {
                    "minimum_numeric": MIN_NUMERIC_SAMPLE_SIZE,
                    "minimum_reliable": MIN_RELIABLE_SAMPLE_SIZE,
                    "minimum_group": MIN_GROUP_SAMPLE_SIZE,
                },
                "limits": {
                    "max_numeric_columns": MAX_NUMERIC_COLUMNS,
                    "max_categorical_columns": MAX_CATEGORICAL_COLUMNS,
                    "max_group_count": MAX_GROUP_COUNT,
                    "max_top_categories": MAX_TOP_CATEGORIES,
                    "max_chart_points": MAX_CHART_POINTS,
                },
                "original_candidate_column_count": selection.original_candidate_count,
                "actual_column_count": len(columns),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                **mode_metadata,
            },
            "chart": chart,
        }

    def _validated_quantiles(self, raw: object) -> list[float]:
        if raw is None:
            return list(DEFAULT_QUANTILES)
        if not isinstance(raw, list) or not raw:
            raise self._input_error(
                "quantiles must be a non-empty list of numbers between 0 and 1.",
                "INVALID_DISTRIBUTION_QUANTILES",
            )
        values: list[float] = []
        for value in raw:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise self._input_error(
                    "quantiles must contain only finite numbers between 0 and 1.",
                    "INVALID_DISTRIBUTION_QUANTILES",
                )
            numeric = float(value)
            if not math.isfinite(numeric) or not 0 <= numeric <= 1:
                raise self._input_error(
                    "quantiles must contain only finite numbers between 0 and 1.",
                    "INVALID_DISTRIBUTION_QUANTILES",
                )
            values.append(numeric)
        unique = sorted(set(values))
        if len(unique) > MAX_QUANTILES:
            raise self._input_error(
                f"At most {MAX_QUANTILES} unique quantiles are supported.",
                "DISTRIBUTION_QUANTILE_LIMIT_EXCEEDED",
            )
        return unique

    def _bounded_integer(self, value: object, name: str, minimum: int, maximum: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
            raise self._input_error(
                f"{name} must be an integer between {minimum} and {maximum}.",
                f"INVALID_DISTRIBUTION_{name.upper()}",
            )
        return value

    def _finite_numeric(self, series: pd.Series) -> tuple[pd.Series, int]:
        numeric = pd.to_numeric(series, errors="coerce").astype("float64")
        infinity_count = int(np.isinf(numeric).sum())
        return numeric.replace([np.inf, -np.inf], np.nan), infinity_count

    def _quantile_key(self, quantile: float) -> str:
        return format(quantile, ".12g")

    def _ratio(self, numerator: int | float, denominator: int | float) -> float:
        if denominator <= 0:
            return 0.0
        return round(float(numerator) / float(denominator), 6)

    def _finite_or_none(self, value: object) -> float | None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return numeric if math.isfinite(numeric) else None

    def _is_missing(self, value: object) -> bool:
        if value is None:
            return True
        try:
            missing = pd.isna(value)
        except (TypeError, ValueError):
            return False
        return bool(missing) if isinstance(missing, (bool, np.bool_)) else False

    def _add_warning(self, warnings: list[str], message: str) -> None:
        if message in warnings:
            return
        if len(warnings) < MAX_WARNINGS:
            warnings.append(message)
        elif len(warnings) == MAX_WARNINGS:
            warnings.append("Additional warnings were omitted to keep the response bounded.")

    def _input_error(
        self,
        message: str,
        error_code: str = "INVALID_DISTRIBUTION_INPUT",
    ) -> ToolExecutionException:
        return ToolExecutionException(
            message,
            error_code=error_code,
            status_code=422,
            details={"tool_name": self.name},
        )

    def _log_unit_failure(self, mode: str) -> None:
        logger.warning("Distribution analysis unit failed | mode=%s status=unit_failed", mode)

    def _log_execution(
        self,
        mode: str,
        summary: dict[str, int],
        status: str,
        started_at: float,
    ) -> None:
        log_method = logger.info if status == "success" else logger.warning
        log_method(
            "Distribution analysis | mode=%s analyzed_column_count=%d excluded_column_count=%d "
            "analyzed_group_count=%d sample_count=%d status=%s elapsed_ms=%.0f",
            mode,
            summary["analyzed_column_count"],
            summary["excluded_column_count"],
            summary["analyzed_group_count"],
            summary["sample_count"],
            status,
            (time.monotonic() - started_at) * 1000,
        )

    def _json_safe(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): self._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set, np.ndarray, pd.Index)):
            return [self._json_safe(item) for item in list(value)]
        if isinstance(value, (pd.Timestamp, datetime, date)):
            return value.isoformat()
        if isinstance(value, (pd.Timedelta, np.timedelta64)):
            return str(value)
        if value is None:
            return None
        if isinstance(value, np.generic):
            return self._json_safe(value.item())
        if isinstance(value, float) and not math.isfinite(value):
            return None
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        if isinstance(value, (str, int, float, bool)):
            return value
        return str(value)
