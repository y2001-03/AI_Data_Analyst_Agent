"""Explainable baseline time-series forecasting with ordered backtesting."""

from __future__ import annotations

import math
import time
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

DEFAULT_HORIZON = 7
MIN_HORIZON = 1
MAX_HORIZON = 365
DEFAULT_VALIDATION_SIZE = 0.2
MIN_VALIDATION_RATIO = 0.1
MAX_VALIDATION_RATIO = 0.4
MIN_SERIES_POINTS = 15
MIN_TRAIN_POINTS = 10
MIN_VALIDATION_POINTS = 3
MIN_GROUP_SERIES_POINTS = 15
MIN_INTERVAL_RESIDUALS = 5
MAX_FORECAST_GROUPS = 50
MAX_HISTORY_CHART_POINTS = 300
MAX_BACKTEST_POINTS = 200
MAX_TOTAL_FORECAST_RECORDS = 5000
MAX_MODEL_CANDIDATES = 4
MIN_WINDOW = 2
MAX_WINDOW = 90
DEFAULT_WINDOW = 7
MIN_SEASON_LENGTH = 2
INTERVAL_Z = 1.96
TIME_PARSE_FAILURE_THRESHOLD = 0.05
HIGH_MISSING_TARGET_RATIO = 0.2
REGULARITY_THRESHOLD = 0.8
FLAT_SLOPE_RELATIVE_THRESHOLD = 1e-6
MODEL_TOLERANCE = 1e-9
EVALUATION_PROTOCOL = "one_step_ahead_walk_forward"

SUPPORTED_MODES = {"univariate", "grouped"}
SUPPORTED_METHODS = {"naive", "seasonal_naive", "moving_average", "linear_trend", "auto"}
SUPPORTED_METRICS = {"mae", "rmse", "mape", "smape"}
DUPLICATE_STRATEGIES = {"aggregate", "first", "last", "error"}
AGGREGATIONS = {"sum", "mean", "median"}
MISSING_STRATEGIES = {"drop", "forward_fill", "interpolate", "error"}
MODEL_COMPLEXITY = {"naive": 0, "moving_average": 1, "seasonal_naive": 2, "linear_trend": 3}


@dataclass
class SeriesData:
    times: list[pd.Timestamp]
    values: list[float]
    profile: dict[str, object]


class ForecastAnalysisTool(BaseDataframeTool):
    """Forecast one numeric series or independent grouped series."""

    name = "forecast_analysis_tool"
    tool_name = "forecast_analysis"
    result_type = "forecast_analysis"

    def __init__(self) -> None:
        self._mode_dispatchers: dict[str, Callable[..., dict[str, object]]] = {
            "univariate": self._run_univariate,
            "grouped": self._run_grouped,
        }
        self._backtest_dispatchers: dict[str, Callable[..., tuple[list[float], dict[str, object]]]] = {
            "naive": self._backtest_naive,
            "seasonal_naive": self._backtest_seasonal_naive,
            "moving_average": self._backtest_moving_average,
            "linear_trend": self._backtest_linear_trend,
        }
        self._forecast_dispatchers: dict[str, Callable[..., tuple[list[float], dict[str, object]]]] = {
            "naive": self._forecast_naive,
            "seasonal_naive": self._forecast_seasonal_naive,
            "moving_average": self._forecast_moving_average,
            "linear_trend": self._forecast_linear_trend,
        }

    def run(self, dataframe: pd.DataFrame, task: AnalysisTask, context: DatasetContext) -> ExecutionResult:
        started = time.monotonic()
        summary = {"selected_model": "", "historical_point_count": 0, "train_size": 0,
                   "validation_size": 0, "group_count": 0, "successful_group_count": 0,
                   "failed_group_count": 0}
        mode = "unknown"
        try:
            if dataframe.empty:
                raise self._error("Forecast analysis requires a non-empty dataframe.", "EMPTY_FORECAST_DATASET")
            config = self._validate_config(dataframe, task.params or {}, context)
            mode = str(config["mode"])
            report = self._mode_dispatchers[mode](dataframe, config)
            for key in summary:
                summary[key] = report["summary"].get(key, summary[key])
        except ToolExecutionException:
            self._log(mode, config.get("target_column", "") if "config" in locals() else "", 0, summary, "failed", started)
            raise
        except Exception as exc:
            self._log(mode, "", 0, summary, "failed", started)
            raise ToolExecutionException("Forecast analysis failed.", details={"tool_name": self.name}) from exc
        self._log(mode, str(config["target_column"]), int(config["horizon"]), summary, "success", started)
        return ExecutionResult(task_name=task.task_name, type=self.result_type,
                               data=self._json_safe(report), chart=None)

    def _validate_config(self, df: pd.DataFrame, params: dict[str, object], context: DatasetContext) -> dict[str, object]:
        cfg = dict(params)
        mode = str(cfg.get("mode", "univariate")).lower()
        method = str(cfg.get("method", "auto")).lower()
        if mode not in SUPPORTED_MODES:
            raise self._error("Forecast mode must be 'univariate' or 'grouped'.", "INVALID_FORECAST_MODE")
        if method not in SUPPORTED_METHODS:
            raise self._error("Unsupported forecast method.", "INVALID_FORECAST_METHOD")
        cfg["mode"], cfg["method"] = mode, method
        cfg["horizon"] = self._integer(cfg.get("horizon", DEFAULT_HORIZON), "horizon", MIN_HORIZON, MAX_HORIZON)
        cfg["window"] = self._integer(cfg.get("window", DEFAULT_WINDOW), "window", MIN_WINDOW, MAX_WINDOW)
        if cfg.get("season_length") is not None:
            cfg["season_length"] = self._integer(cfg["season_length"], "season_length", MIN_SEASON_LENGTH, MAX_HORIZON)
        if method == "seasonal_naive" and cfg.get("season_length") is None:
            raise self._error("seasonal_naive requires season_length.", "MISSING_SEASON_LENGTH")
        metric = str(cfg.get("primary_metric", "mae")).lower()
        if metric not in SUPPORTED_METRICS:
            raise self._error("primary_metric must be mae, rmse, mape, or smape.", "INVALID_FORECAST_METRIC")
        cfg["primary_metric"] = metric
        duplicate = str(cfg.get("duplicate_time_strategy", "aggregate")).lower()
        aggregation = str(cfg.get("aggregation", "sum")).lower()
        missing = str(cfg.get("missing_target_strategy", "drop")).lower()
        if duplicate not in DUPLICATE_STRATEGIES:
            raise self._error("Unknown duplicate_time_strategy.", "INVALID_DUPLICATE_TIME_STRATEGY")
        if aggregation not in AGGREGATIONS:
            raise self._error("aggregation must be sum, mean, or median.", "INVALID_FORECAST_AGGREGATION")
        if missing not in MISSING_STRATEGIES:
            raise self._error("Unknown missing_target_strategy.", "INVALID_MISSING_TARGET_STRATEGY")
        cfg.update(duplicate_time_strategy=duplicate, aggregation=aggregation, missing_target_strategy=missing)
        if "validation_steps" in cfg:
            cfg["validation_steps"] = self._integer(cfg["validation_steps"], "validation_steps", MIN_VALIDATION_POINTS, MAX_TOTAL_FORECAST_RECORDS)
        else:
            ratio = cfg.get("validation_size", DEFAULT_VALIDATION_SIZE)
            if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not MIN_VALIDATION_RATIO <= float(ratio) <= MAX_VALIDATION_RATIO:
                raise self._error("validation_size must be between 0.1 and 0.4.", "INVALID_VALIDATION_SIZE")
            cfg["validation_size"] = float(ratio)
        clip = cfg.get("clip_lower")
        if clip is not None and (isinstance(clip, bool) or not isinstance(clip, (int, float)) or not math.isfinite(float(clip))):
            raise self._error("clip_lower must be a finite number.", "INVALID_CLIP_LOWER")
        cfg["clip_lower"] = float(clip) if clip is not None else None
        cfg["time_column"] = self._resolve_time_column(df, cfg.get("time_column"), context)
        cfg["target_column"] = self._resolve_target_column(df, cfg.get("target_column"), cfg["time_column"], context)
        if mode == "grouped":
            group = cfg.get("group_by")
            if not isinstance(group, str) or not group:
                raise self._error("Grouped forecast requires group_by.", "MISSING_FORECAST_GROUP_BY")
            if group not in df.columns:
                raise self._error(f"Unknown group_by column '{group}'.", "UNKNOWN_FORECAST_GROUP_BY")
            if group in {cfg["time_column"], cfg["target_column"]}:
                raise self._error("group_by must differ from time and target columns.", "INVALID_FORECAST_GROUP_BY")
        return cfg

    def _resolve_time_column(self, df: pd.DataFrame, raw: object, context: DatasetContext) -> str:
        if raw is not None:
            if not isinstance(raw, str) or raw not in df.columns:
                raise self._error("Unknown time_column.", "UNKNOWN_FORECAST_TIME_COLUMN")
            return raw
        candidates = [c for c in context.datetime_columns if c in df.columns]
        if len(candidates) == 1:
            return candidates[0]
        raise self._error("Forecast requires an explicit time_column.", "MISSING_FORECAST_TIME_COLUMN")

    def _resolve_target_column(self, df: pd.DataFrame, raw: object, time_column: str, context: DatasetContext) -> str:
        if raw is not None:
            if not isinstance(raw, str) or raw not in df.columns:
                raise self._error("Unknown target_column.", "UNKNOWN_FORECAST_TARGET_COLUMN")
            candidate = raw
        else:
            candidates = [c for c in context.numeric_columns if c != time_column and self._numeric(df[c])]
            if len(candidates) != 1:
                raise self._error("Forecast requires an explicit target_column.", "MISSING_FORECAST_TARGET_COLUMN")
            candidate = candidates[0]
        if candidate == time_column or not self._numeric(df[candidate]):
            raise self._error("target_column must be a non-boolean real numeric column.", "INVALID_FORECAST_TARGET_COLUMN")
        return candidate

    def _numeric(self, s: pd.Series) -> bool:
        return bool(not pd.api.types.is_bool_dtype(s) and not pd.api.types.is_datetime64_any_dtype(s)
                    and not pd.api.types.is_timedelta64_dtype(s) and pd.api.types.is_numeric_dtype(s)
                    and not pd.api.types.is_complex_dtype(s))

    def _run_univariate(self, df: pd.DataFrame, cfg: dict[str, object]) -> dict[str, object]:
        warnings: list[str] = []
        series = self._prepare_series(df, cfg, warnings)
        result = self._analyze_series(series, cfg, warnings, None)
        return self._assemble("univariate", cfg, result, warnings, [result], None)

    def _run_grouped(self, df: pd.DataFrame, cfg: dict[str, object]) -> dict[str, object]:
        warnings: list[str] = []
        group_by = str(cfg["group_by"])
        groups: dict[str, dict[str, object]] = {}
        for pos, value in enumerate(df[group_by].tolist()):
            label = "__MISSING__" if self._missing(value) else self._json_safe(value)
            token = repr(label)
            groups.setdefault(token, {"label": label, "positions": [], "first": pos})["positions"].append(pos)
        for group in groups.values():
            raw = df[str(cfg["time_column"])].iloc[group["positions"]]
            group["valid_time_count"] = int(pd.to_datetime(raw, errors="coerce", format="mixed").notna().sum())
        ordered = sorted(groups.values(), key=lambda x: (-int(x["valid_time_count"]), int(x["first"])))
        original_count = len(ordered)
        max_groups_by_records = MAX_TOTAL_FORECAST_RECORDS // int(cfg["horizon"])
        limit = min(MAX_FORECAST_GROUPS, max_groups_by_records)
        selected = ordered[:limit]
        if original_count > limit:
            warnings.append(f"Group count exceeded {limit}; largest groups were selected deterministically.")
        results, skipped, failed = [], [], []
        for group in selected:
            label, positions = group["label"], group["positions"]
            subset = df.iloc[positions]
            try:
                series = self._prepare_series(subset, cfg, warnings)
                if len(series.values) < MIN_GROUP_SERIES_POINTS:
                    skipped.append({"group": label, "status": "skipped", "reason": "insufficient_series_points"})
                    continue
                results.append(self._analyze_series(series, cfg, warnings, label))
            except ToolExecutionException as exc:
                failed.append({"group": label, "status": "failed", "reason": exc.message})
                logger.warning("Forecast group failed | mode=grouped status=failed")
        if not results:
            raise self._error("No grouped series had enough valid data for forecasting.", "NO_FORECAST_GROUPS")
        requested_chart_group = cfg.get("chart_group")
        chart_result = next((x for x in results if requested_chart_group is not None and x["group"] == requested_chart_group), None)
        if chart_result is None:
            chart_result = max(results, key=lambda x: int(x["summary"]["historical_point_count"]))
        return self._assemble("grouped", cfg, chart_result, warnings, results, {
            "original_group_count": original_count, "actual_group_count": len(selected),
            "skipped": skipped, "failed": failed,
        })

    def _prepare_series(self, df: pd.DataFrame, cfg: dict[str, object], warnings: list[str]) -> SeriesData:
        time_col, target = str(cfg["time_column"]), str(cfg["target_column"])
        raw_time = df[time_col]
        if pd.api.types.is_numeric_dtype(raw_time) or pd.api.types.is_bool_dtype(raw_time) or pd.api.types.is_timedelta64_dtype(raw_time):
            raise self._error("time_column must be datetime or safely parseable text.", "INVALID_FORECAST_TIME_COLUMN")
        parsed = pd.to_datetime(raw_time, errors="coerce", format="mixed")
        invalid = int(parsed.isna().sum())
        ratio = invalid / len(df) if len(df) else 0.0
        if ratio > TIME_PARSE_FAILURE_THRESHOLD:
            raise self._error("Time parsing failure ratio exceeds 5%.", "FORECAST_TIME_PARSE_FAILURE")
        work = pd.DataFrame({"time": parsed, "target": pd.to_numeric(df[target], errors="coerce")})
        inf = int(np.isinf(work["target"].astype(float)).sum())
        work["target"] = work["target"].replace([np.inf, -np.inf], np.nan)
        if invalid:
            warnings.append(f"{invalid} rows with invalid time values were excluded.")
        if inf:
            warnings.append(f"{inf} infinite target values were treated as missing.")
        work = work.dropna(subset=["time"]).sort_values("time", kind="stable").reset_index(drop=True)
        duplicates = int(work.duplicated("time", keep=False).sum())
        before = len(work)
        strategy = str(cfg["duplicate_time_strategy"])
        if duplicates and strategy == "error":
            raise self._error("Duplicate timestamps are present.", "DUPLICATE_FORECAST_TIMESTAMPS")
        if duplicates:
            warnings.append(f"{duplicates} rows participated in duplicate timestamps; strategy={strategy}.")
            if strategy == "aggregate":
                if cfg["aggregation"] == "sum":
                    work = work.groupby("time", as_index=False, sort=False)["target"].sum(min_count=1)
                else:
                    work = work.groupby("time", as_index=False, sort=False)["target"].agg(str(cfg["aggregation"]))
            else:
                work = work.drop_duplicates("time", keep=strategy).reset_index(drop=True)
        if len(work) < 2:
            raise self._error("Cannot infer a future time interval.", "UNKNOWN_FORECAST_INTERVAL")
        missing_target = int(work["target"].isna().sum())
        missing_ratio = missing_target / len(work) if len(work) else 0.0
        if missing_ratio > HIGH_MISSING_TARGET_RATIO:
            warnings.append("Target missing ratio exceeds 20%; forecast reliability is high risk.")
        profile = {"original_row_count": len(df), "valid_time_row_count": before,
                   "invalid_time_count": invalid, "invalid_time_ratio": ratio,
                   "duplicate_timestamp_count": duplicates, "rows_after_duplicate_handling": len(work),
                   "missing_target_count": missing_target, "sorted_by_time": True}
        return SeriesData(work["time"].tolist(), [float(v) if pd.notna(v) else math.nan for v in work["target"]], profile)

    def _analyze_series(self, series: SeriesData, cfg: dict[str, object], warnings: list[str], group: object) -> dict[str, object]:
        raw_n = len(series.values)
        steps = int(cfg.get("validation_steps") or max(MIN_VALIDATION_POINTS, round(raw_n * float(cfg.get("validation_size", DEFAULT_VALIDATION_SIZE)))))
        if raw_n - steps < MIN_TRAIN_POINTS:
            raise self._error("Insufficient training points after ordered split.", "INSUFFICIENT_FORECAST_TRAINING")
        if steps < MIN_VALIDATION_POINTS or steps >= raw_n:
            raise self._error("Insufficient validation points.", "INSUFFICIENT_FORECAST_VALIDATION")
        train_t, train = self._handle_segment(series.times[:-steps], series.values[:-steps],
                                              str(cfg["missing_target_strategy"]), None)
        valid_t, valid = self._handle_segment(series.times[-steps:], series.values[-steps:],
                                              str(cfg["missing_target_strategy"]), train[-1] if train else None)
        if len(train) < MIN_TRAIN_POINTS or len(valid) < MIN_VALIDATION_POINTS:
            raise self._error("Missing-value handling left insufficient train or validation points.", "INSUFFICIENT_FORECAST_POINTS")
        all_values = train + valid
        all_times = train_t + valid_t
        if len(all_values) < MIN_SERIES_POINTS:
            raise self._error("At least 15 valid series points are required.", "INSUFFICIENT_FORECAST_POINTS")
        freq = self._frequency(series.times)
        if not freq["regular_series"] or int(freq["missing_time_point_count"]) > 0:
            warnings.append("Time intervals are irregular; future steps use the median interval heuristic.")
        comparison = self._compare_models(train, valid, cfg)
        successes = [x for x in comparison if x["status"] == "success"]
        method = str(cfg["method"])
        if method == "auto":
            metric_name = str(cfg["primary_metric"])
            usable = [x for x in successes if x["metrics"][metric_name] is not None]
            if not usable:
                raise self._error("The primary metric is unavailable for every candidate model.", "FORECAST_METRIC_UNAVAILABLE")
            best_score = min(float(x["metrics"][metric_name]) for x in usable)
            near_best = [x for x in usable if float(x["metrics"][metric_name]) <= best_score + MODEL_TOLERANCE]
            chosen = min(near_best, key=lambda x: MODEL_COMPLEXITY[str(x["model"])])
        else:
            chosen = next((x for x in successes if x["model"] == method), None)
            if chosen is None:
                reason = next((x["reason"] for x in comparison if x["model"] == method), "unavailable")
                raise self._error(f"Requested model is unavailable: {reason}.", "FORECAST_MODEL_UNAVAILABLE")
        model = str(chosen["model"])
        predicted = list(chosen["predicted"])
        future, model_params = self._forecast_dispatchers[model](all_values, int(cfg["horizon"]), cfg)
        future = self._clip(future, cfg.get("clip_lower"))
        residuals = [a - p for a, p in zip(valid, predicted, strict=True)]
        residual_std = float(np.std(residuals, ddof=1)) if len(residuals) >= MIN_INTERVAL_RESIDUALS else None
        future_times = self._future_times(all_times[-1], int(cfg["horizon"]), freq)
        forecast = []
        for i, (ts, value) in enumerate(zip(future_times, future, strict=True), 1):
            lower = value - INTERVAL_Z * residual_std if residual_std is not None else None
            upper = value + INTERVAL_Z * residual_std if residual_std is not None else None
            if cfg.get("clip_lower") is not None and lower is not None:
                lower = max(float(cfg["clip_lower"]), lower); upper = max(float(cfg["clip_lower"]), upper)
            record = {"timestamp": ts, "step": i, "prediction": value,
                      "lower_bound": lower, "upper_bound": upper}
            if group is not None: record["group"] = group
            forecast.append(record)
        metrics = chosen["metrics"]
        backtest = {"train_size": len(train), "validation_size": len(valid),
                    "evaluation_protocol": EVALUATION_PROTOCOL,
                    "train_start": train_t[0], "train_end": train_t[-1],
                    "validation_start": valid_t[0], "validation_end": valid_t[-1],
                    "actual": valid[-MAX_BACKTEST_POINTS:], "predicted": predicted[-MAX_BACKTEST_POINTS:],
                    "timestamps": valid_t[-MAX_BACKTEST_POINTS:], "metrics": metrics,
                    "display_truncated": len(valid) > MAX_BACKTEST_POINTS}
        descriptive_slope = float(np.polyfit(np.arange(len(all_values)), np.asarray(all_values), 1)[0])
        direction = self._trend_direction(descriptive_slope, all_values)
        summary = {"time_column": cfg["time_column"], "target_column": cfg["target_column"],
                   "selected_model": model, "primary_metric": cfg["primary_metric"],
                   "validation_metric_value": metrics[str(cfg["primary_metric"])], "horizon": cfg["horizon"],
                   "inferred_frequency": freq["inferred_frequency"], "historical_point_count": len(all_values),
                   "train_size": len(train), "validation_size": len(valid), "reliable": len(all_values) >= 30,
                   "trend_direction": direction, "forecast_start": future_times[0], "forecast_end": future_times[-1]}
        chart = {"type": "forecast_line", "historical": {"x": all_times[-MAX_HISTORY_CHART_POINTS:], "y": all_values[-MAX_HISTORY_CHART_POINTS:]},
                 "validation": {"x": valid_t[-MAX_BACKTEST_POINTS:], "actual": valid[-MAX_BACKTEST_POINTS:], "predicted": predicted[-MAX_BACKTEST_POINTS:]},
                 "future": {"x": future_times, "predicted": future,
                            "lower_bound": [x["lower_bound"] for x in forecast], "upper_bound": [x["upper_bound"] for x in forecast]}}
        return {"group": group, "summary": summary, "series_profile": {**series.profile, **freq,
                "start_time": all_times[0], "end_time": all_times[-1]}, "model_comparison": self._without_predictions(comparison),
                "backtest": backtest, "forecast": forecast, "chart": chart,
                "model_parameters": model_params}

    def _handle_segment(self, times: list[pd.Timestamp], values: list[float], strategy: str,
                        initial: float | None) -> tuple[list[pd.Timestamp], list[float]]:
        missing = sum(not math.isfinite(x) for x in values)
        if missing and strategy == "error": raise self._error("Target contains missing values.", "MISSING_FORECAST_TARGET")
        if strategy == "drop":
            pairs = [(t, x) for t, x in zip(times, values, strict=True) if math.isfinite(x)]
            return [x[0] for x in pairs], [x[1] for x in pairs]
        result_t, result, history = [], [], ([] if initial is None else [initial])
        for timestamp, value in zip(times, values, strict=True):
            if math.isfinite(value): result_t.append(timestamp); result.append(value); history.append(value); continue
            if strategy == "forward_fill" and history: filled = history[-1]
            elif strategy == "interpolate" and len(history) >= 2:
                # This is causal linear extrapolation from past observations, not bidirectional interpolation.
                filled = history[-1] + (history[-1] - history[-2])
            elif history: filled = history[-1]
            else: continue
            result_t.append(timestamp); result.append(filled); history.append(filled)
        return result_t, result

    def _compare_models(self, train: list[float], valid: list[float], cfg: dict[str, object]) -> list[dict[str, object]]:
        candidates = ["naive", "moving_average", "linear_trend"]
        if cfg.get("season_length") is not None: candidates.insert(2, "seasonal_naive")
        output = []
        for model in candidates[:MAX_MODEL_CANDIDATES]:
            try:
                predicted, params = self._backtest_dispatchers[model](train, valid, cfg)
                if len(predicted) != len(valid):
                    raise ValueError("walk-forward predictions must match the validation length")
                output.append({"model": model, "status": "success", "metrics": self._metrics(valid, predicted),
                               "parameters": params, "predicted": predicted,
                               "evaluation_protocol": EVALUATION_PROTOCOL,
                               "validation_point_count": len(valid)})
            except ValueError as exc:
                output.append({"model": model, "status": "unavailable", "reason": str(exc), "metrics": None})
        return output

    def _backtest_naive(self, train, valid, cfg):
        history, predicted = list(train), []
        for actual in valid:
            predicted.append(history[-1])
            history.append(actual)
        return predicted, {}
    def _forecast_naive(self, values, horizon, cfg): return [values[-1]] * horizon, {}
    def _backtest_seasonal_naive(self, train, valid, cfg):
        season = int(cfg.get("season_length") or 0)
        if season < 2 or len(train) < 2 * season: raise ValueError("seasonal history is insufficient")
        history, out = list(train), []
        for actual in valid:
            out.append(history[-season])
            history.append(actual)
        return out, {"season_length": season}
    def _forecast_seasonal_naive(self, values, horizon, cfg):
        season = int(cfg.get("season_length") or 0)
        if season < 2 or len(values) < season: raise ValueError("seasonal history is insufficient")
        history, out = list(values), []
        for _ in range(horizon): out.append(history[-season]); history.append(out[-1])
        return out, {"season_length": season}
    def _backtest_moving_average(self, train, valid, cfg):
        window = int(cfg["window"])
        if window >= len(train): raise ValueError("moving-average window must be smaller than training history")
        history, out = list(train), []
        for actual in valid:
            out.append(float(np.mean(history[-window:])))
            history.append(actual)
        return out, {"window": window, "evaluation": "one_step_ahead_walk_forward"}
    def _forecast_moving_average(self, values, horizon, cfg):
        window = int(cfg["window"])
        if window >= len(values): raise ValueError("moving-average window must be smaller than history")
        history, out = list(values), []
        for _ in range(horizon): out.append(float(np.mean(history[-window:]))); history.append(out[-1])
        return out, {"window": window, "forecast_strategy": "recursive"}
    def _backtest_linear_trend(self, train, valid, cfg):
        if len(train) < 3: raise ValueError("linear trend needs at least three training points")
        history, predicted = list(train), []
        for actual in valid:
            slope, intercept = np.polyfit(np.arange(len(history)), np.asarray(history), 1)
            predicted.append(float(slope * len(history) + intercept))
            history.append(actual)
        return predicted, {"refit_each_step": True, "refit_count": len(valid)}
    def _forecast_linear_trend(self, values, horizon, cfg):
        slope, intercept = np.polyfit(np.arange(len(values)), np.asarray(values), 1)
        pred = slope * np.arange(len(values), len(values) + horizon) + intercept
        return pred.astype(float).tolist(), {"slope": float(slope), "intercept": float(intercept),
                "clip_lower": cfg.get("clip_lower")}

    def _metrics(self, actual: list[float], predicted: list[float]) -> dict[str, object]:
        a, p = np.asarray(actual, float), np.asarray(predicted, float); diff = a - p
        nz = a != 0; sd = (np.abs(a) + np.abs(p)) != 0
        return {"mae": float(np.mean(np.abs(diff))), "rmse": float(np.sqrt(np.mean(diff ** 2))),
                "mape": float(np.mean(np.abs(diff[nz] / a[nz])) * 100) if nz.any() else None,
                "mape_sample_count": int(nz.sum()),
                "smape": float(np.mean(2 * np.abs(diff[sd]) / (np.abs(a[sd]) + np.abs(p[sd]))) * 100) if sd.any() else None,
                "smape_sample_count": int(sd.sum())}

    def _frequency(self, times: list[pd.Timestamp]) -> dict[str, object]:
        unique = sorted(set(times))
        if len(unique) < 2: raise self._error("Cannot infer a future time interval.", "UNKNOWN_FORECAST_INTERVAL")
        seconds = np.asarray([(unique[i] - unique[i-1]).total_seconds() for i in range(1, len(unique))])
        median = float(np.median(seconds)); mode = float(pd.Series(seconds).mode().iloc[0])
        regularity = float(np.mean(np.isclose(seconds, mode, rtol=0.05, atol=1)))
        days = median / 86400
        if 0.9 <= days <= 1.1: label = "daily"
        elif 6 <= days <= 8: label = "weekly"
        elif 27 <= days <= 32: label = "monthly"
        elif 80 <= days <= 100: label = "quarterly"
        elif 350 <= days <= 380: label = "yearly"
        else: label = "irregular"
        return {"inferred_frequency": label, "median_interval_seconds": median,
                "regularity_ratio": regularity, "regular_series": regularity >= REGULARITY_THRESHOLD,
                "missing_time_point_count": int(sum(max(0, round(x / median) - 1) for x in seconds)) if median > 0 else 0}

    def _future_times(self, last: pd.Timestamp, horizon: int, freq: dict[str, object]) -> list[pd.Timestamp]:
        label = freq["inferred_frequency"]
        if label == "monthly": return [last + pd.DateOffset(months=i) for i in range(1, horizon + 1)]
        if label == "quarterly": return [last + pd.DateOffset(months=3*i) for i in range(1, horizon + 1)]
        if label == "yearly": return [last + pd.DateOffset(years=i) for i in range(1, horizon + 1)]
        seconds = float(freq["median_interval_seconds"])
        if not math.isfinite(seconds) or seconds <= 0: raise self._error("Cannot infer a future time interval.", "UNKNOWN_FORECAST_INTERVAL")
        return [last + pd.to_timedelta(seconds * i, unit="s") for i in range(1, horizon + 1)]

    def _assemble(self, mode, cfg, chart_result, warnings, results, grouped):
        all_forecast = [record for result in results for record in result["forecast"]]
        summary = dict(chart_result["summary"])
        if grouped:
            summary.update(group_count=grouped["original_group_count"], successful_group_count=len(results),
                           skipped_group_count=len(grouped["skipped"]), failed_group_count=len(grouped["failed"]))
        else: summary.update(group_count=0, successful_group_count=0, failed_group_count=0)
        return {"tool_name": self.tool_name, "mode": mode, "summary": summary,
                "series_profile": chart_result["series_profile"], "model_comparison": chart_result["model_comparison"],
                "backtest": chart_result["backtest"], "forecast": all_forecast,
                "model_parameters": chart_result["model_parameters"],
                "groups": results if mode == "grouped" else [], "skipped_groups": grouped["skipped"] if grouped else [],
                "failed_groups": grouped["failed"] if grouped else [], "warnings": warnings,
                "metadata": {"time_ordered_split": True, "future_data_leakage_prevented": True,
                    "heuristic_prediction": True, "historical_patterns_may_not_continue": True,
                    "notice": "Historical patterns do not guarantee future outcomes; validate against business changes and external factors.",
                    "heuristic_interval": True, "not_a_formal_confidence_interval": True,
                    "interval_method": "prediction +/- 1.96 * backtest residual sample standard deviation",
                    "metric_definitions": {"mae": "mean absolute error", "rmse": "root mean squared error",
                        "mape": "mean absolute percentage error excluding zero actuals",
                        "smape": "symmetric mean absolute percentage error excluding zero denominators"},
                    "evaluation_protocol": EVALUATION_PROTOCOL,
                    "no_full_resampling_or_interpolation": True, "original_dataframe_modified": False,
                    "missing_target_strategy": cfg["missing_target_strategy"],
                    "interpolation_is_causal": cfg["missing_target_strategy"] == "interpolate",
                    "interpolate_strategy_definition": (
                        "interpolate strategy uses causal linear extrapolation from past observations "
                        "and does not use future values"
                    ),
                    "sample_thresholds": {"min_series": MIN_SERIES_POINTS, "min_train": MIN_TRAIN_POINTS,
                        "min_validation": MIN_VALIDATION_POINTS, "basic_reliability": 30},
                    "selected_chart_group": chart_result.get("group"),
                    "chart_selection_rule": "explicit_chart_group_else_largest_history_then_first_appearance",
                    "generated_at": datetime.now(timezone.utc).isoformat()}, "chart": chart_result["chart"]}

    def _without_predictions(self, comparison):
        return [{k: v for k, v in item.items() if k != "predicted"} for item in comparison]
    def _clip(self, values, lower): return [max(float(lower), x) for x in values] if lower is not None else values
    def _trend_direction(self, slope, values):
        scale = max(max(values) - min(values), abs(float(np.mean(values))), 1.0)
        if abs(slope) <= FLAT_SLOPE_RELATIVE_THRESHOLD * scale: return "flat"
        return "increasing" if slope > 0 else "decreasing"
    def _integer(self, value, name, low, high):
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise self._error(f"{name} must be an integer between {low} and {high}.", f"INVALID_{name.upper()}")
        return value
    def _missing(self, value):
        try: result = pd.isna(value)
        except (TypeError, ValueError): return False
        return bool(result) if isinstance(result, (bool, np.bool_)) else False
    def _error(self, message, code):
        return ToolExecutionException(message, error_code=code, status_code=422, details={"tool_name": self.name})
    def _log(self, mode, target, horizon, summary, status, started):
        (logger.info if status == "success" else logger.warning)(
            "Forecast analysis | mode=%s target_column=%s selected_model=%s horizon=%d historical_point_count=%d train_size=%d validation_size=%d group_count=%d successful_group_count=%d failed_group_count=%d status=%s elapsed_ms=%.0f",
            mode, target, summary["selected_model"], horizon, summary["historical_point_count"], summary["train_size"],
            summary["validation_size"], summary["group_count"], summary["successful_group_count"], summary["failed_group_count"],
            status, (time.monotonic() - started) * 1000)
    def _json_safe(self, value: Any) -> Any:
        if isinstance(value, dict): return {str(k): self._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set, np.ndarray, pd.Index)): return [self._json_safe(v) for v in list(value)]
        if isinstance(value, (pd.Timestamp, datetime, date)): return value.isoformat()
        if isinstance(value, (pd.Timedelta, np.timedelta64)): return str(value)
        if value is None: return None
        if isinstance(value, np.generic): return self._json_safe(value.item())
        if isinstance(value, float) and not math.isfinite(value): return None
        try:
            if pd.isna(value): return None
        except (TypeError, ValueError): pass
        return value if isinstance(value, (str, int, float, bool)) else str(value)
