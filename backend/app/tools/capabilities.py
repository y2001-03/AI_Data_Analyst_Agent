"""Built-in tool capability definitions and explicit consistency validation."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from app.schemas.tool_capability import ToolCapability, ToolFieldRequirement, ToolParameterSpec
from app.tools.anomaly_detection_tool import (
    DEFAULT_IQR_MULTIPLIER,
    DEFAULT_MODE as DEFAULT_ANOMALY_MODE,
    DEFAULT_ROLLING_WINDOW,
    MAX_ANALYSIS_COLUMNS,
    MIN_ROLLING_WINDOW,
)
from app.tools.correlation_analysis_tool import MAX_CORRELATION_COLUMNS
from app.tools.distribution_analysis_tool import (
    DEFAULT_HISTOGRAM_BINS,
    DEFAULT_QUANTILES,
    DEFAULT_TOP_K,
    MAX_HISTOGRAM_BINS,
    MAX_NUMERIC_COLUMNS,
    MAX_TOP_K,
    MIN_HISTOGRAM_BINS,
    MIN_TOP_K,
)
from app.tools.forecast_analysis_tool import (
    DEFAULT_HORIZON,
    DEFAULT_VALIDATION_SIZE,
    DEFAULT_WINDOW,
    MAX_HORIZON,
    MAX_TOTAL_FORECAST_RECORDS,
    MAX_WINDOW,
    MIN_HORIZON,
    MIN_SEASON_LENGTH,
    MIN_VALIDATION_POINTS,
    MIN_VALIDATION_RATIO,
    MIN_WINDOW,
)


def _parameter(
    name: str,
    type_name: str,
    description: str,
    *,
    required: bool = False,
    default: Any = None,
    allowed_values: tuple[Any, ...] | None = None,
    minimum: float | None = None,
    maximum: float | None = None,
    aliases: tuple[str, ...] = (),
    sensitive: bool = False,
) -> ToolParameterSpec:
    return ToolParameterSpec(
        name=name,
        type=type_name,
        required=required,
        default=default,
        description=description,
        allowed_values=allowed_values,
        minimum=minimum,
        maximum=maximum,
        aliases=aliases,
        sensitive=sensitive,
    )


def _field(
    parameter_name: str | None,
    role: str,
    accepted_types: tuple[str, ...],
    description: str,
    *,
    required: bool,
    multiple: bool = False,
    exclude_types: tuple[str, ...] = (),
) -> ToolFieldRequirement:
    return ToolFieldRequirement(
        parameter_name=parameter_name,
        role=role,
        accepted_types=accepted_types,
        required=required,
        multiple=multiple,
        exclude_types=exclude_types,
        description=description,
    )


def _capability(
    *,
    tool_name: str,
    task_types: tuple[str, ...],
    display_name: str,
    description: str,
    category: str,
    parameters: tuple[ToolParameterSpec, ...] = (),
    field_requirements: tuple[ToolFieldRequirement, ...] = (),
    supports_multiple_columns: bool,
    supports_grouping: bool,
    mutates_dataframe: bool = False,
    returns_modified_dataframe: bool = False,
    provides_chart: bool = False,
    chart_types: tuple[str, ...] = (),
    risk_level: str = "read_only",
    deterministic: bool = True,
    limitations: tuple[str, ...],
    examples: tuple[dict[str, Any], ...] = (),
) -> ToolCapability:
    return ToolCapability(
        tool_name=tool_name,
        task_types=task_types,
        display_name=display_name,
        description=description,
        category=category,
        parameters=parameters,
        required_parameters=tuple(item.name for item in parameters if item.required),
        optional_parameters=tuple(item.name for item in parameters if not item.required),
        field_requirements=field_requirements,
        supports_multiple_columns=supports_multiple_columns,
        supports_grouping=supports_grouping,
        mutates_dataframe=mutates_dataframe,
        returns_modified_dataframe=returns_modified_dataframe,
        provides_chart=provides_chart,
        chart_types=chart_types,
        risk_level=risk_level,
        deterministic=deterministic,
        limitations=limitations,
        examples=examples,
    )


def _base_capability() -> ToolCapability:
    return _capability(
        tool_name="base_tool",
        task_types=("unconfigured",),
        display_name="Base Dataframe Tool",
        description="Base contract for deterministic dataframe tools.",
        category="internal",
        supports_multiple_columns=False,
        supports_grouping=False,
        limitations=("Concrete registered tools must provide a built-in capability definition.",),
    )


def _stats_capability() -> ToolCapability:
    return _capability(
        tool_name="stats_tool",
        task_types=("stats",),
        display_name="Descriptive Statistics",
        description="Computes descriptive statistics or a bounded ranked record view.",
        category="descriptive",
        field_requirements=(
            _field(None, "columns", ("numeric",), "Numeric fields are inferred from dataset context.", required=False, multiple=True),
        ),
        supports_multiple_columns=True,
        supports_grouping=False,
        provides_chart=False,
        limitations=(
            "Field and sort direction selection are inferred from task text rather than structured params.",
            "Ranked record output is limited to 20 rows.",
            "Workflow may add a separate visualization task for ordinary stats plans.",
        ),
    )


def _groupby_capability() -> ToolCapability:
    return _capability(
        tool_name="groupby_tool",
        task_types=("groupby",),
        display_name="Grouped Aggregation",
        description="Aggregates one numeric metric or row count by one categorical field.",
        category="aggregation",
        field_requirements=(
            _field(None, "group", ("categorical", "text", "boolean"), "Grouping field is inferred from task text.", required=True),
            _field(None, "value", ("numeric",), "Metric field is inferred unless count aggregation is requested.", required=False),
        ),
        supports_multiple_columns=False,
        supports_grouping=True,
        provides_chart=False,
        limitations=(
            "Supports one grouping field and one metric field.",
            "Aggregation is limited to sum, mean, or count and output is limited to 20 groups.",
            "Workflow visualization handling remains separate from this legacy task type.",
        ),
    )


def _trend_capability() -> ToolCapability:
    return _capability(
        tool_name="trend_tool",
        task_types=("trend",),
        display_name="Historical Trend",
        description="Resamples historical observations into daily, weekly, or monthly aggregates.",
        category="time_series",
        field_requirements=(
            _field(None, "time", ("datetime",), "Time field is inferred from dataset context.", required=True),
            _field(None, "value", ("numeric",), "Metric field is optional for count aggregation.", required=False),
        ),
        supports_multiple_columns=False,
        supports_grouping=False,
        provides_chart=False,
        limitations=(
            "Frequency and aggregation are inferred from task text.",
            "This tool describes historical trends and does not forecast future values.",
            "Workflow visualization handling remains separate from this legacy task type.",
        ),
    )


def _sql_capability() -> ToolCapability:
    query = _parameter("query", "string", "Read-only SELECT query executed against the dataset table.", required=True)
    return _capability(
        tool_name="sql_tool",
        task_types=("sql",),
        display_name="Read-only SQL",
        description="Executes a bounded safe SELECT query against the in-memory dataset.",
        category="query",
        parameters=(query,),
        field_requirements=(
            _field("query", "query", ("any",), "SQL may reference fields exposed by the dataset table.", required=True, multiple=True),
        ),
        supports_multiple_columns=True,
        supports_grouping=True,
        provides_chart=False,
        risk_level="read_only",
        limitations=(
            "Only SELECT statements are accepted; write and DDL keywords are rejected.",
            "The tool queries an in-memory table named dataset and limits results to 100 rows by default.",
        ),
        examples=({"query": "SELECT region, SUM(sales) FROM dataset GROUP BY region"},),
    )


def _quality_capability() -> ToolCapability:
    return _capability(
        tool_name="data_quality_tool",
        task_types=("data_quality_analysis",),
        display_name="Data Quality Analysis",
        description="Profiles missing values, duplicates, constant fields, identifier candidates, and IQR outliers.",
        category="data_quality",
        field_requirements=(
            _field(None, "columns", ("any",), "All dataframe fields are profiled independently.", required=False, multiple=True),
        ),
        supports_multiple_columns=True,
        supports_grouping=False,
        provides_chart=False,
        limitations=(
            "Quality score and issue thresholds are generic heuristics.",
            "IQR outliers are descriptive candidates and are not removed or modified.",
        ),
    )


def _cleaning_capability() -> ToolCapability:
    parameters = (
        _parameter("mode", "string", "Plan recommendations or execute explicit actions on a deep copy.", allowed_values=("plan", "execute")),
        _parameter("actions", "array", "Explicit cleaning action objects for execute mode.", default=[]),
    )
    return _capability(
        tool_name="data_cleaning_tool",
        task_types=("data_cleaning_plan", "data_cleaning_execute"),
        display_name="Data Cleaning Workflow",
        description="Builds a cleaning plan or applies explicit supported actions to a dataframe copy.",
        category="data_cleaning",
        parameters=parameters,
        field_requirements=(
            _field("actions", "columns", ("any",), "Action objects may target one or more existing fields.", required=False, multiple=True),
        ),
        supports_multiple_columns=True,
        supports_grouping=False,
        mutates_dataframe=False,
        returns_modified_dataframe=True,
        provides_chart=False,
        risk_level="mutating_copy",
        limitations=(
            "Plan mode is preview-only and does not apply cleaning actions.",
            "Execute mode produces changes only on an isolated dataframe copy.",
            "The original dataframe is never overwritten.",
            "Results expose an audit log and bounded preview, not a persisted dataset replacement.",
            "Supported actions are drop_duplicates, fill_missing, drop_missing_rows, trim_strings, and convert_dtype.",
        ),
        examples=({"mode": "plan"}, {"mode": "execute", "actions": []}),
    )


def _correlation_capability() -> ToolCapability:
    parameters = (
        _parameter("method", "string", "Correlation coefficient method.", default="pearson", allowed_values=("pearson", "spearman")),
        _parameter("missing_strategy", "string", "Missing-value handling across pairs.", default="pairwise", allowed_values=("pairwise", "listwise")),
        _parameter("columns", "array", "Optional explicit numeric columns to analyze."),
    )
    return _capability(
        tool_name="correlation_analysis_tool",
        task_types=("correlation_analysis",),
        display_name="Correlation Analysis",
        description="Computes Pearson or Spearman association matrices for numeric fields.",
        category="correlation",
        parameters=parameters,
        field_requirements=(
            _field("columns", "columns", ("numeric",), "At least two eligible numeric fields are required.", required=False, multiple=True,
                   exclude_types=("boolean", "datetime", "timedelta")),
        ),
        supports_multiple_columns=True,
        supports_grouping=False,
        provides_chart=True,
        chart_types=("heatmap",),
        limitations=(
            "Correlation describes statistical association and does not establish causation.",
            f"At most {MAX_CORRELATION_COLUMNS} explicit numeric columns are analyzed.",
            "No significance test or regression model is performed.",
        ),
    )


def _anomaly_capability() -> ToolCapability:
    parameters = (
        _parameter("mode", "string", "Detection scope.", default=DEFAULT_ANOMALY_MODE,
                   allowed_values=("global", "groupwise", "time_series")),
        _parameter("method", "string", "Detection method; compatibility depends on mode.", default="iqr",
                   allowed_values=("iqr", "zscore", "robust_zscore", "rolling_zscore")),
        _parameter("columns", "array", "Numeric columns for global or groupwise detection."),
        _parameter("group_by", "string", "Grouping field required in groupwise mode."),
        _parameter("time_column", "string", "Time field required in time_series mode."),
        _parameter("value_columns", "array", "Numeric value fields required in time_series mode."),
        _parameter("iqr_multiplier", "number", "Positive IQR fence multiplier.", default=DEFAULT_IQR_MULTIPLIER),
        _parameter("threshold", "number", "Positive score threshold for z-score based methods."),
        _parameter("window", "integer", "Trailing window for rolling_zscore.", default=DEFAULT_ROLLING_WINDOW,
                   minimum=MIN_ROLLING_WINDOW),
        _parameter("requested_deletion", "boolean", "Records whether deletion was requested; detection remains read-only.", default=False),
    )
    return _capability(
        tool_name="anomaly_detection_tool",
        task_types=("anomaly_detection",),
        display_name="Anomaly Detection",
        description="Finds bounded candidate statistical anomalies without modifying records.",
        category="anomaly_detection",
        parameters=parameters,
        field_requirements=(
            _field("columns", "columns", ("numeric",), "Numeric fields for global or grouped detection.", required=False, multiple=True,
                   exclude_types=("boolean", "datetime", "timedelta")),
            _field("group_by", "group", ("categorical", "text", "boolean", "numeric"), "Grouping field for groupwise mode.", required=False),
            _field("time_column", "time", ("datetime", "text"), "Ordered time field for time-series mode.", required=False),
            _field("value_columns", "value", ("numeric",), "Numeric time-series value fields.", required=False, multiple=True,
                   exclude_types=("boolean", "datetime", "timedelta")),
        ),
        supports_multiple_columns=True,
        supports_grouping=True,
        provides_chart=True,
        chart_types=("anomaly_scatter", "group_anomaly_summary", "time_series_anomaly"),
        limitations=(
            "Detected records are statistical candidates, not confirmed business anomalies.",
            f"Analysis is bounded to {MAX_ANALYSIS_COLUMNS} numeric columns.",
            "The tool never deletes, replaces, or repairs source records.",
        ),
    )


def _distribution_capability() -> ToolCapability:
    parameters = (
        _parameter("mode", "string", "Distribution mode; inferred when omitted.",
                   allowed_values=("numeric", "categorical", "group_comparison")),
        _parameter("columns", "array", "Optional fields to analyze."),
        _parameter("quantiles", "array", "Numeric quantiles between zero and one.", default=list(DEFAULT_QUANTILES)),
        _parameter("bins", "integer", "Histogram bin count.", default=DEFAULT_HISTOGRAM_BINS,
                   minimum=MIN_HISTOGRAM_BINS, maximum=MAX_HISTOGRAM_BINS),
        _parameter("top_k", "integer", "Maximum displayed categories.", default=DEFAULT_TOP_K,
                   minimum=MIN_TOP_K, maximum=MAX_TOP_K),
        _parameter("allow_low_cardinality_numeric", "boolean", "Allow eligible low-cardinality numeric fields as categories.", default=False),
        _parameter("group_by", "string", "Grouping field required for group_comparison mode."),
    )
    return _capability(
        tool_name="distribution_analysis_tool",
        task_types=("distribution_analysis",),
        display_name="Distribution Analysis",
        description="Describes numeric, categorical, or grouped distributions with bounded chart payloads.",
        category="distribution",
        parameters=parameters,
        field_requirements=(
            _field("columns", "columns", ("numeric", "categorical", "text", "boolean"), "Accepted field types depend on mode.", required=False, multiple=True,
                   exclude_types=("datetime", "timedelta")),
            _field("group_by", "group", ("categorical", "text", "boolean", "numeric"), "Grouping field for descriptive group comparison.", required=False),
        ),
        supports_multiple_columns=True,
        supports_grouping=True,
        provides_chart=True,
        chart_types=("histogram", "category_bar", "group_boxplot_summary"),
        limitations=(
            "Distribution labels are descriptive heuristics and are not formal statistical tests.",
            f"Automatic numeric selection is bounded to {MAX_NUMERIC_COLUMNS} columns.",
            "Outlier summaries do not replace anomaly detection and do not modify data.",
        ),
    )


def _forecast_capability() -> ToolCapability:
    parameters = (
        _parameter("mode", "string", "Single-series or independent grouped forecast mode.", default="univariate",
                   allowed_values=("univariate", "grouped")),
        _parameter("time_column", "string", "Datetime or safely parseable text time field."),
        _parameter("target_column", "string", "Real numeric field to forecast."),
        _parameter("group_by", "string", "Grouping field required for grouped mode."),
        _parameter("horizon", "integer", "Number of future periods.", default=DEFAULT_HORIZON,
                   minimum=MIN_HORIZON, maximum=MAX_HORIZON, aliases=("steps", "periods")),
        _parameter("method", "string", "Explainable baseline forecast method.", default="auto",
                   allowed_values=("auto", "naive", "seasonal_naive", "moving_average", "linear_trend")),
        _parameter("window", "integer", "Moving-average window.", default=DEFAULT_WINDOW,
                   minimum=MIN_WINDOW, maximum=MAX_WINDOW),
        _parameter("season_length", "integer", "Season length required by seasonal_naive.", minimum=MIN_SEASON_LENGTH,
                   maximum=MAX_HORIZON),
        _parameter("validation_size", "number", "Ordered validation ratio.", default=DEFAULT_VALIDATION_SIZE,
                   minimum=MIN_VALIDATION_RATIO, maximum=0.4),
        _parameter("validation_steps", "integer", "Explicit ordered validation point count.",
                   minimum=MIN_VALIDATION_POINTS, maximum=MAX_TOTAL_FORECAST_RECORDS),
        _parameter("primary_metric", "string", "Metric used by auto model selection.", default="mae",
                   allowed_values=("mae", "rmse", "mape", "smape")),
        _parameter("duplicate_time_strategy", "string", "Duplicate timestamp handling.", default="aggregate",
                   allowed_values=("aggregate", "first", "last", "error")),
        _parameter("aggregation", "string", "Aggregation used for duplicate timestamps.", default="sum",
                   allowed_values=("sum", "mean", "median")),
        _parameter("missing_target_strategy", "string", "Causal missing-target handling.", default="drop",
                   allowed_values=("drop", "forward_fill", "interpolate", "error")),
        _parameter("clip_lower", "number", "Optional explicit lower prediction bound."),
        _parameter("chart_group", "string", "Optional grouped result selected for the main chart."),
    )
    return _capability(
        tool_name="forecast_analysis_tool",
        task_types=("forecast_analysis",),
        display_name="Forecast Analysis",
        description="Performs ordered walk-forward backtesting and explainable baseline forecasting.",
        category="forecasting",
        parameters=parameters,
        field_requirements=(
            _field("time_column", "time", ("datetime", "text"), "One ordered time field; it may be inferred only when unambiguous.", required=False),
            _field("target_column", "target", ("numeric",), "One real numeric target; it may be inferred only when unambiguous.", required=False,
                   exclude_types=("boolean", "datetime", "timedelta")),
            _field("group_by", "group", ("categorical", "text", "boolean", "numeric"), "One group field in grouped mode.", required=False),
        ),
        supports_multiple_columns=False,
        supports_grouping=True,
        provides_chart=True,
        chart_types=("forecast_line",),
        limitations=(
            "Forecasts are heuristic baselines and historical patterns may not continue.",
            "Only one target field and one time field are supported per task.",
            "No exogenous variables, formal confidence intervals, or advanced forecasting models are provided.",
        ),
        examples=({"time_column": "date", "target_column": "sales", "horizon": 7, "method": "auto"},),
    )


_CAPABILITY_BUILDERS: dict[str, Callable[[], ToolCapability]] = {
    "base_tool": _base_capability,
    "stats_tool": _stats_capability,
    "groupby_tool": _groupby_capability,
    "trend_tool": _trend_capability,
    "sql_tool": _sql_capability,
    "data_quality_tool": _quality_capability,
    "data_cleaning_tool": _cleaning_capability,
    "correlation_analysis_tool": _correlation_capability,
    "anomaly_detection_tool": _anomaly_capability,
    "distribution_analysis_tool": _distribution_capability,
    "forecast_analysis_tool": _forecast_capability,
}


def get_builtin_tool_capability(tool_name: str) -> ToolCapability:
    """Build a fresh immutable capability object for a known built-in tool."""
    builder = _CAPABILITY_BUILDERS.get(tool_name)
    if builder is None:
        raise ValueError(f"No capability metadata is registered for tool '{tool_name}'.")
    return builder()


def validate_capability(tool: object) -> ToolCapability:
    """Validate a tool/capability pair without executing the tool."""
    tool_name = getattr(tool, "name", None)
    if not isinstance(tool_name, str) or not tool_name:
        raise ValueError("Tool must expose a non-empty name.")
    getter = getattr(tool, "get_capability", None)
    capability = getter() if callable(getter) else getattr(tool, "capability", None)
    if not isinstance(capability, ToolCapability):
        raise ValueError(f"Tool '{tool_name}' must expose a ToolCapability.")
    validated = ToolCapability.model_validate(capability.model_dump())
    if validated.tool_name != tool_name:
        raise ValueError(
            f"Capability tool_name '{validated.tool_name}' does not match registered name '{tool_name}'."
        )
    json.dumps(validated.model_dump(), allow_nan=False)
    return validated
