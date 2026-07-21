"""Safe, auditable dataframe cleaning plan and execution tool."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Callable

import pandas as pd

from app.core.exceptions import ToolExecutionException
from app.core.logging import get_logger
from app.schemas.file import AnalysisTask, ExecutionResult
from app.tools.dataframe_tools import BaseDataframeTool, DatasetContext

logger = get_logger(__name__)


@dataclass
class ActionOutcome:
    """Internal result of one atomic cleaning action."""

    dataframe: pd.DataFrame
    affected_count: int
    before_value: object
    after_value: object
    details: dict[str, object] = field(default_factory=dict)


class DataCleaningTool(BaseDataframeTool):
    """Create cleaning plans or execute explicit actions on a deep copy."""

    name = "data_cleaning_tool"
    tool_name = "data_cleaning"
    plan_type = "data_cleaning_plan"
    execute_type = "data_cleaning_execute"
    preview_limit = 20

    # Deletion risk thresholds are ratios of the original row count.
    medium_drop_missing_ratio = 0.10
    high_drop_missing_ratio = 0.30
    high_duplicate_drop_ratio = 0.20
    conversion_risk_ratio = 0.05

    def __init__(self) -> None:
        self._dispatchers: dict[
            str,
            Callable[[pd.DataFrame, dict[str, object]], ActionOutcome],
        ] = {
            "drop_duplicates": self._execute_drop_duplicates,
            "fill_missing": self._execute_fill_missing,
            "drop_missing_rows": self._execute_drop_missing_rows,
            "trim_strings": self._execute_trim_strings,
            "convert_dtype": self._execute_convert_dtype,
        }

    def run(
        self,
        dataframe: pd.DataFrame,
        task: AnalysisTask,
        context: DatasetContext,
    ) -> ExecutionResult:
        """Plan or execute cleaning without mutating the caller's dataframe."""
        del context
        try:
            mode = self._resolve_mode(task)
            if mode == "plan":
                data = self._build_plan(dataframe)
                result_type = self.plan_type
            else:
                data = self._execute_actions(dataframe, task.params or {})
                result_type = self.execute_type
        except ToolExecutionException:
            raise
        except Exception as exc:
            raise ToolExecutionException(
                "Data cleaning failed.",
                details={"tool_name": self.name},
            ) from exc

        return ExecutionResult(
            task_name=task.task_name,
            type=result_type,
            data=self._json_safe(data),
            chart=None,
        )

    def _resolve_mode(self, task: AnalysisTask) -> str:
        if task.type == self.plan_type:
            return "plan"
        if task.type == self.execute_type:
            return "execute"
        mode = (task.params or {}).get("mode")
        if mode in {"plan", "execute"}:
            return str(mode)
        raise ToolExecutionException(
            "Data cleaning mode must be 'plan' or 'execute'.",
            error_code="INVALID_CLEANING_MODE",
            status_code=422,
            details={"tool_name": self.name},
        )

    def _build_plan(self, dataframe: pd.DataFrame) -> dict[str, object]:
        row_count = int(len(dataframe))
        actions: list[dict[str, object]] = []
        warnings: list[str] = []

        duplicate_count = int(dataframe.duplicated(keep="first").sum()) if row_count else 0
        if duplicate_count:
            duplicate_ratio = self._ratio(duplicate_count, row_count)
            risk_level = "high" if duplicate_ratio > self.high_duplicate_drop_ratio else "low"
            actions.append(
                self._plan_action(
                    action_id="remove_duplicate_rows",
                    action_type="drop_duplicates",
                    column=None,
                    reason=f"Found {duplicate_count} fully duplicate rows.",
                    affected_count=duplicate_count,
                    affected_ratio=duplicate_ratio,
                    risk_level=risk_level,
                    default_enabled=risk_level != "high",
                    parameters={"keep": "first"},
                )
            )

        for column_index, column_name in enumerate(dataframe.columns):
            series = dataframe.iloc[:, column_index]
            series.name = column_name
            missing_count = int(series.isna().sum())
            if missing_count:
                inferred_type = self._infer_type(series)
                strategy = self._recommended_fill_strategy(series, inferred_type)
                if strategy is None:
                    warnings.append(
                        f"Column '{column_name}' has missing values but no safe automatic fill strategy was proposed."
                    )
                else:
                    missing_ratio = self._ratio(missing_count, row_count)
                    risk_level = self._missing_fill_risk(missing_ratio)
                    actions.append(
                        self._plan_action(
                            action_id=self._action_id("fill_missing", column_name),
                            action_type="fill_missing",
                            column=str(column_name),
                            reason=(
                                f"Column '{column_name}' has {missing_count} missing values "
                                f"({missing_ratio:.2%}); {strategy} is a transparent baseline strategy."
                            ),
                            affected_count=missing_count,
                            affected_ratio=missing_ratio,
                            risk_level=risk_level,
                            default_enabled=False,
                            parameters={"strategy": strategy},
                        )
                    )

            trim_count = self._trim_affected_rows(series)
            if trim_count:
                trim_ratio = self._ratio(trim_count, row_count)
                actions.append(
                    self._plan_action(
                        action_id=self._action_id("trim_strings", column_name),
                        action_type="trim_strings",
                        column=str(column_name),
                        reason=f"Column '{column_name}' contains leading, trailing, or blank-only whitespace.",
                        affected_count=trim_count,
                        affected_ratio=trim_ratio,
                        risk_level="low",
                        default_enabled=True,
                        parameters={"convert_blank_to_null": True},
                    )
                )

            target_type = self._recommended_conversion(series)
            if target_type:
                convertible_count = int(series.notna().sum())
                actions.append(
                    self._plan_action(
                        action_id=self._action_id("convert_dtype", column_name),
                        action_type="convert_dtype",
                        column=str(column_name),
                        reason=(
                            f"All non-missing string values in '{column_name}' can be converted "
                            f"to {target_type} without coercion failures."
                        ),
                        affected_count=convertible_count,
                        affected_ratio=self._ratio(convertible_count, row_count),
                        risk_level="low",
                        default_enabled=False,
                        parameters={"target_type": target_type},
                    )
                )

        missing_columns = [
            str(dataframe.columns[index])
            for index in range(len(dataframe.columns))
            if dataframe.iloc[:, index].isna().any()
        ]
        if missing_columns:
            affected_count = int(dataframe[missing_columns].isna().any(axis=1).sum())
            affected_ratio = self._ratio(affected_count, row_count)
            risk_level = self._drop_missing_risk(affected_ratio)
            actions.append(
                self._plan_action(
                    action_id="drop_rows_with_missing_values",
                    action_type="drop_missing_rows",
                    column=None,
                    reason=(
                        f"Dropping rows with any missing value in the selected columns would remove "
                        f"{affected_count} rows ({affected_ratio:.2%})."
                    ),
                    affected_count=affected_count,
                    affected_ratio=affected_ratio,
                    risk_level=risk_level,
                    default_enabled=False,
                    parameters={"columns": missing_columns, "how": "any"},
                )
            )

        high_risk_count = sum(action["risk_level"] == "high" for action in actions)
        if high_risk_count:
            warnings.append("High-risk actions are disabled by default and require explicit execution.")
        estimated_affected_rows = min(
            row_count,
            sum(int(action["affected_count"]) for action in actions),
        )
        return {
            "tool_name": self.tool_name,
            "mode": "plan",
            "summary": {
                "row_count": row_count,
                "column_count": int(len(dataframe.columns)),
                "recommended_action_count": len(actions),
                "high_risk_action_count": int(high_risk_count),
                "estimated_affected_rows": estimated_affected_rows,
            },
            "actions": actions,
            "warnings": warnings,
            "metadata": {
                "modifies_original": False,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "heuristic_plan": True,
                "note": (
                    "Recommendations are generic, explainable heuristics and require business review "
                    "before execution."
                ),
            },
        }

    def _execute_actions(
        self,
        dataframe: pd.DataFrame,
        params: dict[str, object],
    ) -> dict[str, object]:
        raw_actions = params.get("actions")
        if raw_actions is None:
            raw_actions = []
        if not isinstance(raw_actions, list):
            raise ToolExecutionException(
                "Data cleaning execute actions must be a list.",
                error_code="INVALID_CLEANING_ACTIONS",
                status_code=422,
                details={"tool_name": self.name},
            )

        cleaned_dataframe = dataframe.copy(deep=True)
        before = self._dataset_metrics(cleaned_dataframe)
        action_results: list[dict[str, object]] = []
        audit_log: list[dict[str, object]] = []
        warnings: list[str] = []

        if not raw_actions:
            warnings.append("No actions were provided; no cleaning was performed.")

        for sequence, raw_action in enumerate(raw_actions, start=1):
            if not isinstance(raw_action, dict):
                action = {"action_type": "invalid_action"}
                action_id = f"invalid_action_{sequence}"
                error = ValueError("Each cleaning action must be an object.")
                action_results.append(self._failed_action_result(action_id, error))
                self._log_action("invalid_action", None, 0, "failed")
                continue

            action = raw_action
            action_type = str(action.get("action_type") or "")
            action_id = str(action.get("action_id") or f"{action_type or 'action'}_{sequence}")
            column = action.get("column")
            try:
                dispatcher = self._dispatchers.get(action_type)
                if dispatcher is None:
                    raise ValueError(f"Unsupported cleaning action_type '{action_type}'.")
                candidate = cleaned_dataframe.copy(deep=True)
                outcome = dispatcher(candidate, action)
                cleaned_dataframe = outcome.dataframe
                result = {
                    "action_id": action_id,
                    "action_type": action_type,
                    "column": str(column) if column is not None else None,
                    "status": "success",
                    "affected_count": int(outcome.affected_count),
                    "message": f"Action '{action_type}' completed successfully.",
                    **outcome.details,
                }
                action_results.append(result)
                audit_log.append(
                    {
                        "sequence": sequence,
                        "action_id": action_id,
                        "action_type": action_type,
                        "column": str(column) if column is not None else None,
                        "before_value": outcome.before_value,
                        "after_value": outcome.after_value,
                        "affected_count": int(outcome.affected_count),
                    }
                )
                self._log_action(action_type, column, outcome.affected_count, "success")
            except (KeyError, TypeError, ValueError) as exc:
                action_results.append(self._failed_action_result(action_id, exc, action_type, column))
                self._log_action(action_type or "unknown", column, 0, "failed")

        after = self._dataset_metrics(cleaned_dataframe)
        successful_count = sum(result["status"] == "success" for result in action_results)
        failed_count = len(action_results) - successful_count
        return {
            "tool_name": self.tool_name,
            "mode": "execute",
            "summary": {
                "before": before,
                "after": after,
                "successful_action_count": int(successful_count),
                "failed_action_count": int(failed_count),
            },
            "action_results": action_results,
            "audit_log": audit_log,
            "preview": cleaned_dataframe.head(self.preview_limit).to_dict(orient="records"),
            "warnings": warnings,
            "metadata": {
                "original_dataframe_modified": False,
                "dataset_context_updated": False,
                "preview_row_limit": self.preview_limit,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
        }

    def _execute_drop_duplicates(
        self,
        dataframe: pd.DataFrame,
        action: dict[str, object],
    ) -> ActionOutcome:
        parameters = self._parameters(action)
        keep_value = parameters.get("keep", "first")
        if keep_value == "false":
            keep_value = False
        if keep_value not in {"first", "last", False}:
            raise ValueError("drop_duplicates keep must be 'first', 'last', or false.")
        before_count = int(len(dataframe))
        cleaned = dataframe.drop_duplicates(keep=keep_value).copy()
        cleaned = cleaned.reset_index(drop=True)
        affected_count = before_count - int(len(cleaned))
        return ActionOutcome(cleaned, affected_count, before_count, int(len(cleaned)))

    def _execute_fill_missing(
        self,
        dataframe: pd.DataFrame,
        action: dict[str, object],
    ) -> ActionOutcome:
        column = self._required_column(dataframe, action)
        parameters = self._parameters(action)
        strategy = parameters.get("strategy")
        if strategy not in {"mean", "median", "mode", "constant"}:
            raise ValueError("fill_missing strategy must be mean, median, mode, or constant.")

        series = dataframe[column]
        inferred_type = self._infer_type(series)
        missing_count = int(series.isna().sum())
        if strategy in {"mean", "median"}:
            if inferred_type != "numeric":
                raise ValueError(f"Strategy '{strategy}' is only supported for numeric columns.")
            non_missing = series.dropna()
            if non_missing.empty:
                raise ValueError(f"Column '{column}' is fully missing; {strategy} cannot be calculated.")
            fill_value = non_missing.mean() if strategy == "mean" else non_missing.median()
        elif strategy == "mode":
            if inferred_type not in {"boolean", "text", "categorical"}:
                raise ValueError("Strategy 'mode' is only supported for boolean, text, or category columns.")
            modes = series.mode(dropna=True)
            if modes.empty:
                raise ValueError(f"Column '{column}' is fully missing; mode cannot be calculated.")
            fill_value = modes.iloc[0]
        else:
            if "value" not in parameters:
                raise ValueError("Strategy 'constant' requires an explicit value parameter.")
            fill_value = parameters["value"]
            self._validate_constant(inferred_type, fill_value)

        if inferred_type == "categorical" and fill_value not in series.cat.categories:
            series = series.cat.add_categories([fill_value])
        dataframe[column] = series.fillna(fill_value)
        after_missing = int(dataframe[column].isna().sum())
        affected_count = missing_count - after_missing
        return ActionOutcome(
            dataframe,
            affected_count,
            missing_count,
            after_missing,
            {"strategy": str(strategy)},
        )

    def _execute_drop_missing_rows(
        self,
        dataframe: pd.DataFrame,
        action: dict[str, object],
    ) -> ActionOutcome:
        parameters = self._parameters(action)
        columns = parameters.get("columns")
        if columns is None and action.get("column") is not None:
            columns = [action["column"]]
        if columns is not None:
            if not isinstance(columns, list) or not all(isinstance(column, str) for column in columns):
                raise ValueError("drop_missing_rows columns must be a list of column names.")
            self._validate_columns(dataframe, columns)
        how = parameters.get("how", "any")
        if how not in {"any", "all"}:
            raise ValueError("drop_missing_rows how must be 'any' or 'all'.")

        before_count = int(len(dataframe))
        cleaned = dataframe.dropna(subset=columns, how=str(how)).copy().reset_index(drop=True)
        affected_count = before_count - int(len(cleaned))
        affected_ratio = self._ratio(affected_count, before_count)
        return ActionOutcome(
            cleaned,
            affected_count,
            before_count,
            int(len(cleaned)),
            {
                "affected_ratio": affected_ratio,
                "risk_level": self._drop_missing_risk(affected_ratio),
            },
        )

    def _execute_trim_strings(
        self,
        dataframe: pd.DataFrame,
        action: dict[str, object],
    ) -> ActionOutcome:
        parameters = self._parameters(action)
        convert_blank_to_null = parameters.get("convert_blank_to_null", False)
        if not isinstance(convert_blank_to_null, bool):
            raise ValueError("trim_strings convert_blank_to_null must be boolean.")
        columns = self._string_target_columns(dataframe, action, parameters)
        changed_rows = pd.Series(False, index=dataframe.index, dtype=bool)

        for column in columns:
            before_series = dataframe[column]
            after_series = before_series.astype(object).map(
                lambda value: self._trim_value(value, convert_blank_to_null)
            )
            equal = before_series.eq(after_series) | (before_series.isna() & after_series.isna())
            changed_rows = changed_rows | ~equal.fillna(False)
            dataframe[column] = after_series

        affected_count = int(changed_rows.sum())
        return ActionOutcome(
            dataframe,
            affected_count,
            affected_count,
            0,
            {"processed_columns": columns},
        )

    def _execute_convert_dtype(
        self,
        dataframe: pd.DataFrame,
        action: dict[str, object],
    ) -> ActionOutcome:
        column = self._required_column(dataframe, action)
        parameters = self._parameters(action)
        target_type = parameters.get("target_type")
        if target_type not in {"numeric", "datetime", "boolean"}:
            raise ValueError("convert_dtype target_type must be numeric, datetime, or boolean.")

        source = dataframe[column]
        source_type = self._infer_type(source)
        before_dtype = str(source.dtype)
        non_missing_count = int(source.notna().sum())
        if target_type == "numeric":
            if source_type not in {"text", "categorical"}:
                raise ValueError("Conversion to numeric is only supported from string/category columns.")
            converted = pd.to_numeric(source, errors="coerce")
        elif target_type == "datetime":
            if source_type not in {"text", "categorical"}:
                raise ValueError("Conversion to datetime is only supported from string/category columns.")
            converted = pd.to_datetime(source, errors="coerce")
        else:
            converted = self._convert_boolean(source, source_type)

        failure_count = int((source.notna() & converted.isna()).sum())
        failure_ratio = self._ratio(failure_count, non_missing_count)
        dataframe[column] = converted
        return ActionOutcome(
            dataframe,
            non_missing_count - failure_count,
            before_dtype,
            str(dataframe[column].dtype),
            {
                "target_type": str(target_type),
                "conversion_failure_count": failure_count,
                "conversion_failure_ratio": failure_ratio,
                "risk_level": self._conversion_risk(failure_ratio),
                "new_missing_count": failure_count,
            },
        )

    def _convert_boolean(self, series: pd.Series, source_type: str) -> pd.Series:
        if source_type == "numeric":
            non_missing_values = set(series.dropna().tolist())
            if not non_missing_values.issubset({0, 1, 0.0, 1.0}):
                raise ValueError("Numeric to boolean conversion requires values limited to 0 and 1.")
            return series.map({0: False, 1: True, 0.0: False, 1.0: True}).astype("boolean")
        if source_type not in {"text", "categorical"}:
            raise ValueError("Conversion to boolean is only supported from numeric or string/category columns.")
        mapping = {
            "true": True,
            "false": False,
            "yes": True,
            "no": False,
            "1": True,
            "0": False,
        }
        normalized = series.map(lambda value: value.strip().lower() if isinstance(value, str) else value)
        non_missing_values = set(normalized.dropna().tolist())
        if not non_missing_values.issubset(mapping):
            raise ValueError(
                "String to boolean conversion requires values limited to true/false, yes/no, or 1/0."
            )
        return normalized.map(mapping).astype("boolean")

    def _parameters(self, action: dict[str, object]) -> dict[str, object]:
        parameters = action.get("parameters", {})
        if not isinstance(parameters, dict):
            raise ValueError("Cleaning action parameters must be an object.")
        return parameters

    def _required_column(self, dataframe: pd.DataFrame, action: dict[str, object]) -> str:
        column = action.get("column")
        if not isinstance(column, str) or not column:
            raise ValueError("Cleaning action requires a column name.")
        self._validate_columns(dataframe, [column])
        return column

    def _validate_columns(self, dataframe: pd.DataFrame, columns: list[str]) -> None:
        unknown = [column for column in columns if column not in dataframe.columns]
        if unknown:
            raise ValueError(f"Unknown column(s): {', '.join(unknown)}.")

    def _string_target_columns(
        self,
        dataframe: pd.DataFrame,
        action: dict[str, object],
        parameters: dict[str, object],
    ) -> list[str]:
        raw_columns = parameters.get("columns")
        if raw_columns is None and action.get("column") is not None:
            raw_columns = [action["column"]]
        if raw_columns is None:
            columns = [
                str(column)
                for column in dataframe.columns
                if self._infer_type(dataframe[column]) in {"text", "categorical"}
            ]
        else:
            if not isinstance(raw_columns, list) or not all(
                isinstance(column, str) for column in raw_columns
            ):
                raise ValueError("trim_strings columns must be a list of column names.")
            columns = raw_columns
            self._validate_columns(dataframe, columns)
            unsupported = [
                column
                for column in columns
                if self._infer_type(dataframe[column]) not in {"text", "categorical"}
            ]
            if unsupported:
                raise ValueError(
                    f"trim_strings only supports text/category columns: {', '.join(unsupported)}."
                )
        return columns

    def _recommended_fill_strategy(self, series: pd.Series, inferred_type: str) -> str | None:
        if series.dropna().empty:
            return None
        if inferred_type == "numeric":
            return "median"
        if inferred_type in {"boolean", "text", "categorical"}:
            return "mode"
        return None

    def _recommended_conversion(self, series: pd.Series) -> str | None:
        if self._infer_type(series) not in {"text", "categorical"}:
            return None
        values = series.dropna()
        if values.empty or not values.map(lambda value: isinstance(value, str)).all():
            return None
        normalized = values.astype(str).str.strip()
        boolean_values = {"true", "false", "yes", "no", "1", "0"}
        if set(normalized.str.lower()).issubset(boolean_values):
            return "boolean"

        column_name = str(series.name).lower()
        if any(token in column_name for token in ("date", "time", "日期", "时间")):
            converted_datetime = pd.to_datetime(normalized, errors="coerce")
            if converted_datetime.notna().all():
                return "datetime"
        converted_numeric = pd.to_numeric(normalized, errors="coerce")
        if converted_numeric.notna().all():
            return "numeric"
        return None

    def _infer_type(self, series: pd.Series) -> str:
        if pd.api.types.is_bool_dtype(series):
            return "boolean"
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"
        if isinstance(series.dtype, pd.CategoricalDtype):
            return "categorical"
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            return "text"
        return "unsupported"

    def _validate_constant(self, inferred_type: str, value: object) -> None:
        if value is None:
            raise ValueError("Constant fill value cannot be null.")
        if inferred_type == "numeric" and (not isinstance(value, (int, float)) or isinstance(value, bool)):
            raise ValueError("Numeric columns require a numeric constant value.")
        if inferred_type == "boolean" and not isinstance(value, bool):
            raise ValueError("Boolean columns require a boolean constant value.")
        if inferred_type in {"text", "categorical"} and not isinstance(value, str):
            raise ValueError("Text/category columns require a string constant value.")
        if inferred_type in {"datetime", "unsupported"}:
            raise ValueError(f"Constant fill is not supported for {inferred_type} columns.")

    def _trim_affected_rows(self, series: pd.Series) -> int:
        if self._infer_type(series) not in {"text", "categorical"}:
            return 0
        return int(
            series.map(
                lambda value: isinstance(value, str) and (value != value.strip() or not value.strip())
            ).sum()
        )

    def _trim_value(self, value: object, convert_blank_to_null: bool) -> object:
        if not isinstance(value, str):
            return value
        trimmed = value.strip()
        if convert_blank_to_null and not trimmed:
            return None
        return trimmed

    def _dataset_metrics(self, dataframe: pd.DataFrame) -> dict[str, int]:
        return {
            "row_count": int(len(dataframe)),
            "column_count": int(len(dataframe.columns)),
            "missing_count": int(dataframe.isna().sum().sum()),
            "duplicate_count": int(dataframe.duplicated(keep="first").sum()),
        }

    def _plan_action(
        self,
        *,
        action_id: str,
        action_type: str,
        column: str | None,
        reason: str,
        affected_count: int,
        affected_ratio: float,
        risk_level: str,
        default_enabled: bool,
        parameters: dict[str, object],
    ) -> dict[str, object]:
        return {
            "action_id": action_id,
            "action_type": action_type,
            "column": column,
            "reason": reason,
            "affected_count": int(affected_count),
            "affected_ratio": affected_ratio,
            "risk_level": risk_level,
            "default_enabled": default_enabled,
            "parameters": parameters,
        }

    def _failed_action_result(
        self,
        action_id: str,
        error: Exception,
        action_type: str = "invalid_action",
        column: object = None,
    ) -> dict[str, object]:
        return {
            "action_id": action_id,
            "action_type": action_type,
            "column": str(column) if column is not None else None,
            "status": "failed",
            "affected_count": 0,
            "message": str(error),
            "error_type": type(error).__name__,
        }

    def _log_action(
        self,
        action_type: str,
        column: object,
        affected_count: int,
        status: str,
    ) -> None:
        log_method = logger.info if status == "success" else logger.warning
        log_method(
            "Data cleaning action | action_type=%s column=%s affected_count=%d status=%s",
            action_type,
            str(column) if column is not None else "",
            int(affected_count),
            status,
        )

    def _missing_fill_risk(self, ratio: float) -> str:
        if ratio > self.high_drop_missing_ratio:
            return "high"
        if ratio > self.medium_drop_missing_ratio:
            return "medium"
        return "low"

    def _drop_missing_risk(self, ratio: float) -> str:
        if ratio > self.high_drop_missing_ratio:
            return "high"
        if ratio > self.medium_drop_missing_ratio:
            return "medium"
        return "low"

    def _conversion_risk(self, ratio: float) -> str:
        if ratio > self.high_drop_missing_ratio:
            return "high"
        if ratio > self.conversion_risk_ratio:
            return "medium"
        return "low"

    def _action_id(self, action_type: str, column: object) -> str:
        normalized = re.sub(r"[^a-zA-Z0-9_]+", "_", str(column)).strip("_").lower()
        return f"{action_type}_{normalized or 'column'}"

    def _ratio(self, numerator: int | float, denominator: int | float) -> float:
        if denominator <= 0:
            return 0.0
        return round(float(numerator) / float(denominator), 6)

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
