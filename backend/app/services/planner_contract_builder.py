"""Capability-driven Planner Contract normalization and validation."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from app.schemas.file import AnalysisTask, DatasetUploadResponse
from app.schemas.planner_contract import (
    PlannerDiagnostics,
    PlannerExecutionStage,
    PlannerInputBinding,
    PlannerIssueCode,
    PlannerPlan,
    PlannerRequestColumn,
    PlannerRequestContext,
    PlannerSource,
    PlannerTask,
    PlannerTaskSource,
    PlannerValidationIssue,
)
from app.schemas.tool_capability import FieldType, RiskLevel, ToolCapability, ToolFieldRequirement, ToolParameterSpec, _ensure_json_value
from app.services.planner_capability_context import build_planner_capability_context
from app.tools.registry import ToolRegistry

TASK_RISK_OVERRIDES: dict[str, RiskLevel] = {
    "data_cleaning_plan": "preview_only",
    "data_cleaning_execute": "mutating_copy",
}

RISK_LEVEL_PRIORITY: dict[RiskLevel, int] = {
    "read_only": 0,
    "preview_only": 1,
    "mutating_copy": 2,
    "external_side_effect": 3,
}

ALLOWED_PLANNER_CONTEXT_PATHS = {
    "question",
    "dataset_id",
    "file_name",
    "columns",
    "row_count",
    "request_id",
    "trace_id",
}


class PlannerContractBuilder:
    """Normalize raw planner outputs into a validated PlannerPlan."""

    def __init__(self, registry: ToolRegistry | None = None) -> None:
        self.registry = registry or ToolRegistry.with_default_tools()

    def build_request_context(
        self,
        file_info: DatasetUploadResponse,
        *,
        question: str | None = None,
        dataset_id: str | None = None,
        request_id: str | None = None,
        trace_id: str | None = None,
    ) -> PlannerRequestContext:
        """Build a non-sensitive request context for planning and validation."""
        return PlannerRequestContext(
            question=question,
            dataset_id=dataset_id,
            file_name=file_info.file_name,
            columns=tuple(
                PlannerRequestColumn(name=column.name, type=self._logical_column_type(column.data_type, column.name))
                for column in file_info.columns
            ),
            row_count=file_info.row_count,
            capability_context=build_planner_capability_context(self.registry),
            request_id=request_id,
            trace_id=trace_id,
        )

    def build_plan(
        self,
        raw_output: object,
        file_info: DatasetUploadResponse,
        *,
        question: str | None = None,
        source: PlannerSource = "llm",
        fallback_used: bool = False,
        parse_failure: str | None = None,
    ) -> PlannerPlan:
        """Normalize and validate a raw planner response without executing tools."""
        raw_tasks = self._extract_raw_tasks(raw_output)
        request_context = self.build_request_context(file_info, question=question)
        plan_issues: list[PlannerValidationIssue] = []
        diagnostics_notes: list[str] = []
        requested_status = raw_output.get("status") if isinstance(raw_output, dict) else None
        if requested_status == "clarification_required":
            plan_issues.append(
                self._issue(
                    "CLARIFICATION_REQUIRED",
                    "Planner indicated that additional user information is required.",
                    "error",
                    repairable=True,
                )
            )
        elif requested_status == "unsupported":
            plan_issues.append(
                self._issue(
                    "UNSUPPORTED_REQUEST",
                    "Planner indicated that the request is unsupported by current capabilities.",
                    "error",
                    repairable=False,
                )
            )
        if parse_failure:
            diagnostics_notes.append("llm_parse_failure")
        if self._has_unsupported_conditional_request(question):
            plan_issues.append(
                self._issue(
                    "CONDITIONAL_EXECUTION_UNSUPPORTED",
                    "Conditional execution is not supported in this planning step; the plan will not be executed automatically.",
                    "error",
                    repairable=True,
                )
            )
        if raw_tasks is None:
            plan_issues.append(
                self._issue(
                    "UNSUPPORTED_REQUEST",
                    "Planner output did not contain a usable tasks list.",
                    "error",
                    repairable=False,
                )
            )
            raw_tasks = []
        if not raw_tasks and not plan_issues:
            plan_issues.append(
                self._issue(
                    "EMPTY_TASK_LIST",
                    "Planner output contained an empty tasks list.",
                    "error",
                    repairable=True,
                )
            )

        normalized_tasks: list[PlannerTask] = []
        for index, raw_task in enumerate(raw_tasks):
            normalized_tasks.append(
                self._normalize_task(
                    raw_task,
                    index=index,
                    request_context=request_context,
                    source="fallback" if source == "fallback" else "llm",
                    diagnostics_notes=diagnostics_notes,
                )
            )

        seen_task_ids: set[str] = set()
        deduplicated_tasks: list[PlannerTask] = []
        for index, task in enumerate(normalized_tasks):
            if task.task_id not in seen_task_ids:
                seen_task_ids.add(task.task_id)
                deduplicated_tasks.append(task)
                continue
            original_task_id = task.task_id
            replacement_task_id = f"{original_task_id}_{index + 1}"
            issue = self._issue(
                "DUPLICATE_TASK_ID",
                f"Duplicate task_id '{original_task_id}' appears in the plan.",
                "error",
                task_id=replacement_task_id,
                field="task_id",
                actual=original_task_id,
                repairable=True,
            )
            seen_task_ids.add(replacement_task_id)
            deduplicated_tasks.append(
                task.model_copy(
                    update={
                        "task_id": replacement_task_id,
                        "validation_status": "invalid",
                        "validation_issues": (*task.validation_issues, issue),
                    }
                )
            )
        normalized_tasks = deduplicated_tasks

        dependency_issues = self._validate_dependencies(normalized_tasks)
        binding_issues = self._validate_input_bindings(normalized_tasks)
        normalized_tasks = self._attach_task_issues(
            normalized_tasks,
            [*dependency_issues, *binding_issues],
        )
        all_issues = [
            *plan_issues,
            *(issue for task in normalized_tasks for issue in task.validation_issues),
        ]
        status = self._plan_status(normalized_tasks, all_issues)
        execution_order: tuple[str, ...] = ()
        execution_stages: tuple[PlannerExecutionStage, ...] = ()
        if not self._has_error([*dependency_issues, *plan_issues]):
            execution_order = self._topological_sort(normalized_tasks)
            execution_stages = self._build_execution_stages(normalized_tasks)
            normalized_tasks = self._apply_execution_metadata(
                normalized_tasks,
                execution_order,
                execution_stages,
            )
            all_issues = [
                *plan_issues,
                *(issue for task in normalized_tasks for issue in task.validation_issues),
            ]
            status = self._plan_status(normalized_tasks, all_issues)
        unsupported_binding_count = sum(
            issue.code == "UNSUPPORTED_TASK_OUTPUT_BINDING"
            for issue in all_issues
        )
        execution_supported = (
            status == "valid"
            and unsupported_binding_count == 0
            and not any(issue.code == "CONDITIONAL_EXECUTION_UNSUPPORTED" for issue in all_issues)
        )
        dependency_count = sum(len(task.depends_on) for task in normalized_tasks)
        diagnostics = PlannerDiagnostics(
            source=source,
            raw_task_count=len(raw_tasks),
            normalized_task_count=len(normalized_tasks),
            validated_task_count=sum(task.validation_status == "valid" for task in normalized_tasks),
            invalid_task_count=sum(task.validation_status == "invalid" for task in normalized_tasks),
            capability_catalog_version=self._catalog_version(),
            fallback_used=fallback_used or source == "fallback",
            normalization_warnings=list(diagnostics_notes),
            parse_failure=parse_failure,
            tool_selection_rule="task_type_exact_capability_index_first_tool_name_ascending",
            missing_information=[
                issue.parameter_name or issue.field or issue.code
                for issue in all_issues
                if issue.code == "CLARIFICATION_REQUIRED"
            ],
            dependency_count=dependency_count,
            stage_count=len(execution_stages),
            maximum_parallel_width=max((len(stage.task_ids) for stage in execution_stages), default=0),
            topological_sort_applied=bool(execution_order),
            execution_supported=execution_supported,
            unsupported_binding_count=unsupported_binding_count,
        )
        intent = self._intent(raw_output, question)
        metadata = {
            "task_count": len(normalized_tasks),
            "dependency_count": dependency_count,
            "stage_count": len(execution_stages),
            "maximum_parallel_width": max((len(stage.task_ids) for stage in execution_stages), default=0),
            "issue_count": len(all_issues),
            "error_count": sum(issue.severity == "error" for issue in all_issues),
            "warning_count": sum(issue.severity == "warning" for issue in all_issues),
            "execution_supported": execution_supported,
            "task_output_binding_supported": False,
            "limitations": [
                "This step records task dependencies and stages but still adapts executable plans to a flat serial AnalysisTask list.",
                "Only original_dataset bindings are currently executable; task_output bindings are declarative and unsupported at runtime.",
            ],
        }
        return PlannerPlan(
            plan_id=self._stable_plan_id(intent, normalized_tasks, source),
            user_intent=intent,
            normalized_intent=self._normalized_intent(normalized_tasks, status),
            tasks=tuple(normalized_tasks),
            status=status,
            planner_source=source,
            capability_catalog_version=self._catalog_version(),
            warnings=[issue.message for issue in all_issues if issue.severity == "warning"],
            validation_issues=tuple(all_issues),
            diagnostics=diagnostics,
            metadata=metadata,
            execution_stages=execution_stages,
            execution_order=execution_order,
            has_dependencies=dependency_count > 0,
            parallelizable=len(execution_stages) == 1 and len(normalized_tasks) > 1,
            execution_supported=execution_supported,
        )

    def planner_task_to_analysis_task(self, task: PlannerTask) -> AnalysisTask:
        """Adapt one valid PlannerTask into the existing execution schema."""
        if task.validation_status != "valid":
            raise ValueError("Only valid PlannerTask objects can be adapted for execution.")
        return AnalysisTask(
            task_name=task.name,
            reasoning=task.reason,
            expected_output=task.description,
            type=task.task_type,
            params=dict(task.params),
        )

    def to_analysis_tasks(self, plan: PlannerPlan) -> list[AnalysisTask]:
        """Adapt a valid plan into existing AnalysisTask objects."""
        if plan.status != "valid":
            raise ValueError("Only valid PlannerPlan objects can be adapted for execution.")
        if not plan.execution_supported:
            raise ValueError("Only executable PlannerPlan objects can be adapted for execution.")
        task_by_id = {task.task_id: task for task in plan.tasks}
        ordered_task_ids = plan.execution_order or tuple(task.task_id for task in plan.tasks)
        return [self.planner_task_to_analysis_task(task_by_id[task_id]) for task_id in ordered_task_ids]

    def _normalize_task(
        self,
        raw_task: object,
        *,
        index: int,
        request_context: PlannerRequestContext,
        source: PlannerTaskSource,
        diagnostics_notes: list[str],
    ) -> PlannerTask:
        raw = self._task_to_dict(raw_task)
        task_id = self._string(raw.get("task_id")) or f"task_{index + 1}"
        task_type = self._string(raw.get("task_type")) or self._string(raw.get("type")) or ""
        requested_tool = self._string(raw.get("tool_name"))
        name = self._string(raw.get("name")) or self._string(raw.get("task_name")) or f"Task {index + 1}"
        description = self._string(raw.get("description")) or self._string(raw.get("expected_output")) or ""
        reason = self._string(raw.get("reason")) or self._string(raw.get("reasoning")) or ""
        confidence, confidence_issue = self._confidence(raw.get("confidence"), source)
        issues: list[PlannerValidationIssue] = []
        if confidence_issue is not None:
            issues.append(confidence_issue.model_copy(update={"task_id": task_id}))

        capabilities = self.registry.find_by_task_type(task_type) if task_type else []
        capability = self._resolve_capability(task_id, task_type, requested_tool, capabilities, issues)
        tool_name = capability.tool_name if capability is not None else (requested_tool or "unknown_tool")
        risk_level: RiskLevel = self._risk_for_task(task_type, capability)
        raw_risk = self._string(raw.get("risk_level"))
        if capability is not None and raw_risk and raw_risk != risk_level:
            issues.append(
                self._issue(
                    "RISK_METADATA_MISMATCH",
                    "Planner supplied risk metadata that differs from capability metadata; capability value was used.",
                    "warning",
                    task_id=task_id,
                    field="risk_level",
                    expected=risk_level,
                    actual=raw_risk,
                    repairable=True,
                )
            )
        raw_params = self._strip_internal_params(task_type, raw.get("params", {}), diagnostics_notes)
        params, param_issues = self._validate_params(task_id, raw_params, capability)
        issues.extend(param_issues)
        depends_on, dependency_shape_issues = self._normalize_depends_on(raw.get("depends_on"), task_id)
        issues.extend(dependency_shape_issues)
        input_bindings, input_binding_shape_issues = self._normalize_input_bindings(raw.get("input_bindings"), task_id)
        issues.extend(input_binding_shape_issues)
        if capability is not None:
            issues.extend(self._validate_fields(task_id, params, capability, request_context))
        if capability is not None and not self._has_error(issues):
            issues.extend(
                self._validate_clarification_requirements(
                    task_id,
                    task_type,
                    params,
                    request_context,
                )
            )
        validation_status = "invalid" if any(issue.severity == "error" for issue in issues) else "valid"
        if len(capabilities) > 1 and not requested_tool:
            diagnostics_notes.append(f"{task_id}:multiple_tools_first_sorted")
        return PlannerTask(
            task_id=task_id,
            task_type=task_type or "unknown",
            tool_name=tool_name,
            name=name,
            description=description,
            params=params,
            reason=reason,
            confidence=confidence,
            source=source,
            risk_level=risk_level,
            requires_confirmation=self._requires_confirmation(task_type, risk_level),
            validation_status=validation_status,
            validation_issues=tuple(issues),
            depends_on=depends_on,
            input_bindings=input_bindings,
        )

    def _resolve_capability(
        self,
        task_id: str,
        task_type: str,
        requested_tool: str | None,
        capabilities: list[ToolCapability],
        issues: list[PlannerValidationIssue],
    ) -> ToolCapability | None:
        if not task_type:
            issues.append(
                self._issue("UNKNOWN_TASK_TYPE", "Planner task is missing task_type.", "error", task_id=task_id, field="task_type", repairable=True)
            )
            return None
        if not capabilities:
            issues.append(
                self._issue(
                    "UNKNOWN_TASK_TYPE",
                    f"Task type '{task_type}' is not present in the capability catalog.",
                    "error",
                    task_id=task_id,
                    field="task_type",
                    actual=task_type,
                    repairable=True,
                )
            )
            return None
        if requested_tool:
            requested_capability = self.registry.get_capability(requested_tool)
            if requested_capability is None:
                issues.append(
                    self._issue(
                        "UNKNOWN_TOOL",
                        f"Tool '{requested_tool}' is not registered.",
                        "error",
                        task_id=task_id,
                        field="tool_name",
                        actual=requested_tool,
                        repairable=True,
                    )
                )
                return capabilities[0]
            if task_type not in requested_capability.task_types:
                issues.append(
                    self._issue(
                        "TASK_TOOL_MISMATCH",
                        f"Tool '{requested_tool}' does not support task_type '{task_type}'.",
                        "error",
                        task_id=task_id,
                        field="tool_name",
                        expected=list(requested_capability.task_types),
                        actual=task_type,
                        repairable=True,
                    )
                )
                return requested_capability
            return requested_capability
        return capabilities[0]

    def _validate_params(
        self,
        task_id: str,
        raw_params: object,
        capability: ToolCapability | None,
    ) -> tuple[dict[str, object], list[PlannerValidationIssue]]:
        issues: list[PlannerValidationIssue] = []
        if not isinstance(raw_params, dict):
            return {}, [
                self._issue(
                    "INVALID_PARAMETER_TYPE",
                    "params must be an object.",
                    "error",
                    task_id=task_id,
                    field="params",
                    expected="object",
                    actual=type(raw_params).__name__,
                    repairable=True,
                )
            ]
        try:
            _ensure_json_value(raw_params, "params")
        except ValueError as exc:
            return {}, [
                self._issue(
                    "NON_JSON_SAFE_PARAMS",
                    str(exc),
                    "error",
                    task_id=task_id,
                    field="params",
                    repairable=True,
                )
            ]
        if capability is None:
            return dict(raw_params), issues
        specs = {parameter.name: parameter for parameter in capability.parameters}
        normalized: dict[str, object] = {}
        for name, value in raw_params.items():
            if name not in specs:
                issues.append(
                    self._issue(
                        "UNKNOWN_PARAMETER",
                        f"Parameter '{name}' is not supported by tool '{capability.tool_name}'.",
                        "error",
                        task_id=task_id,
                        field=f"params.{name}",
                        parameter_name=name,
                        repairable=True,
                    )
                )
                continue
            spec = specs[name]
            issues.extend(self._validate_parameter_value(task_id, spec, value))
            normalized[name] = value
        for name in capability.required_parameters:
            if name not in normalized:
                issues.append(
                    self._issue(
                        "MISSING_REQUIRED_PARAMETER",
                        f"Required parameter '{name}' is missing for tool '{capability.tool_name}'.",
                        "error",
                        task_id=task_id,
                        field=f"params.{name}",
                        parameter_name=name,
                        repairable=True,
                    )
                )
        return normalized, issues

    def _validate_parameter_value(
        self,
        task_id: str,
        spec: ToolParameterSpec,
        value: object,
    ) -> list[PlannerValidationIssue]:
        issues: list[PlannerValidationIssue] = []
        if value is None:
            if spec.required:
                issues.append(
                    self._issue(
                        "MISSING_REQUIRED_PARAMETER",
                        f"Required parameter '{spec.name}' cannot be null.",
                        "error",
                        task_id=task_id,
                        field=f"params.{spec.name}",
                        parameter_name=spec.name,
                        actual=None,
                        repairable=True,
                    )
                )
            return issues
        if not self._matches_parameter_type(value, spec.type):
            issues.append(
                self._issue(
                    "INVALID_PARAMETER_TYPE",
                    f"Parameter '{spec.name}' must be {spec.type}.",
                    "error",
                    task_id=task_id,
                    field=f"params.{spec.name}",
                    parameter_name=spec.name,
                    expected=spec.type,
                    actual=type(value).__name__,
                    repairable=True,
                )
            )
            return issues
        if spec.allowed_values is not None and value not in spec.allowed_values:
            issues.append(
                self._issue(
                    "INVALID_PARAMETER_VALUE",
                    f"Parameter '{spec.name}' must be one of the allowed values.",
                    "error",
                    task_id=task_id,
                    field=f"params.{spec.name}",
                    parameter_name=spec.name,
                    expected=list(spec.allowed_values),
                    actual=value if isinstance(value, (str, int, float, bool)) else type(value).__name__,
                    repairable=True,
                )
            )
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric = float(value)
            if spec.minimum is not None and numeric < spec.minimum:
                issues.append(self._range_issue(task_id, spec, "minimum", spec.minimum, value))
            if spec.maximum is not None and numeric > spec.maximum:
                issues.append(self._range_issue(task_id, spec, "maximum", spec.maximum, value))
        return issues

    def _validate_fields(
        self,
        task_id: str,
        params: dict[str, object],
        capability: ToolCapability,
        request_context: PlannerRequestContext,
    ) -> list[PlannerValidationIssue]:
        issues: list[PlannerValidationIssue] = []
        field_types = {column.name: column.type for column in request_context.columns}
        for requirement in capability.field_requirements:
            if requirement.parameter_name is None or requirement.parameter_name not in params:
                continue
            if requirement.role == "query" or requirement.parameter_name == "actions":
                continue
            value = params[requirement.parameter_name]
            fields = self._field_values(value, requirement, task_id, issues)
            for field_name in fields:
                actual_type = field_types.get(field_name)
                if actual_type is None:
                    issues.append(
                        self._issue(
                            "UNKNOWN_FIELD",
                            f"Field '{field_name}' does not exist in the dataset schema.",
                            "error",
                            task_id=task_id,
                            field=f"params.{requirement.parameter_name}",
                            parameter_name=requirement.parameter_name,
                            actual=field_name,
                            repairable=True,
                        )
                    )
                    continue
                if not self._field_type_allowed(actual_type, requirement):
                    issues.append(
                        self._issue(
                            "INVALID_FIELD_TYPE",
                            f"Field '{field_name}' has type '{actual_type}', which is not accepted for parameter '{requirement.parameter_name}'.",
                            "error",
                            task_id=task_id,
                            field=f"params.{requirement.parameter_name}",
                            parameter_name=requirement.parameter_name,
                            expected=list(requirement.accepted_types),
                            actual=actual_type,
                            repairable=True,
                        )
                    )
        return issues

    def _validate_clarification_requirements(
        self,
        task_id: str,
        task_type: str,
        params: dict[str, object],
        request_context: PlannerRequestContext,
    ) -> list[PlannerValidationIssue]:
        """Detect clear intent with missing information that cannot be inferred safely."""
        if task_type == "forecast_analysis":
            return self._forecast_clarification_issues(task_id, params, request_context)
        if task_type == "distribution_analysis" and params.get("mode") == "group_comparison":
            if not params.get("group_by"):
                return [
                    self._clarification_issue(
                        task_id,
                        "group_by",
                        "Group comparison requires group_by because no grouping field can be safely inferred.",
                    )
                ]
        if task_type == "data_cleaning_execute":
            actions = params.get("actions")
            if not isinstance(actions, list) or len(actions) == 0:
                return [
                    self._clarification_issue(
                        task_id,
                        "actions",
                        "Cleaning execution requires explicit actions before it can be planned safely.",
                    )
                ]
        return []

    def _forecast_clarification_issues(
        self,
        task_id: str,
        params: dict[str, object],
        request_context: PlannerRequestContext,
    ) -> list[PlannerValidationIssue]:
        issues: list[PlannerValidationIssue] = []
        if "time_column" not in params or params.get("time_column") is None:
            candidates = self._candidate_columns(request_context, {"datetime"})
            if len(candidates) != 1:
                issues.append(
                    self._clarification_issue(
                        task_id,
                        "time_column",
                        "Forecast analysis requires an explicit time_column when it cannot be inferred unambiguously.",
                        expected="one datetime or parseable text field",
                        actual=candidates,
                    )
                )
        if "target_column" not in params or params.get("target_column") is None:
            candidates = self._candidate_columns(request_context, {"numeric"})
            if len(candidates) != 1:
                issues.append(
                    self._clarification_issue(
                        task_id,
                        "target_column",
                        "Forecast analysis requires an explicit target_column when multiple numeric fields are available.",
                        expected="one numeric target field",
                        actual=candidates,
                    )
                )
        return issues

    def _clarification_issue(
        self,
        task_id: str,
        parameter_name: str,
        message: str,
        *,
        expected: object = None,
        actual: object = None,
    ) -> PlannerValidationIssue:
        return self._issue(
            "CLARIFICATION_REQUIRED",
            message,
            "error",
            task_id=task_id,
            field=f"params.{parameter_name}",
            parameter_name=parameter_name,
            expected=expected,
            actual=actual,
            repairable=True,
        )

    def _field_values(
        self,
        value: object,
        requirement: ToolFieldRequirement,
        task_id: str,
        issues: list[PlannerValidationIssue],
    ) -> list[str]:
        parameter = requirement.parameter_name or "field"
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            if not requirement.multiple:
                issues.append(
                    self._issue(
                        "INVALID_PARAMETER_TYPE",
                        f"Parameter '{parameter}' accepts only one field.",
                        "error",
                        task_id=task_id,
                        field=f"params.{parameter}",
                        parameter_name=parameter,
                        expected="string",
                        actual="array",
                        repairable=True,
                    )
                )
                return []
            invalid = [item for item in value if not isinstance(item, str)]
            if invalid:
                issues.append(
                    self._issue(
                        "INVALID_PARAMETER_TYPE",
                        f"Parameter '{parameter}' must contain only field names.",
                        "error",
                        task_id=task_id,
                        field=f"params.{parameter}",
                        parameter_name=parameter,
                        expected="list[string]",
                        actual="array",
                        repairable=True,
                    )
                )
                return []
            return list(value)
        issues.append(
            self._issue(
                "INVALID_PARAMETER_TYPE",
                f"Parameter '{parameter}' must be a field name or list of field names.",
                "error",
                task_id=task_id,
                field=f"params.{parameter}",
                parameter_name=parameter,
                expected="string or list[string]",
                actual=type(value).__name__,
                repairable=True,
            )
        )
        return []

    def _normalize_depends_on(
        self,
        raw_depends_on: object,
        task_id: str,
    ) -> tuple[tuple[str, ...], list[PlannerValidationIssue]]:
        if raw_depends_on is None:
            return (), []
        if not isinstance(raw_depends_on, list):
            return (), [
                self._issue(
                    "INVALID_INPUT_BINDING",
                    "depends_on must be an array of task ids.",
                    "error",
                    task_id=task_id,
                    field="depends_on",
                    expected="array[string]",
                    actual=type(raw_depends_on).__name__,
                    repairable=True,
                )
            ]
        depends_on: list[str] = []
        issues: list[PlannerValidationIssue] = []
        for index, value in enumerate(raw_depends_on):
            if not isinstance(value, str) or not value:
                issues.append(
                    self._issue(
                        "INVALID_INPUT_BINDING",
                        "depends_on must contain only non-empty task ids.",
                        "error",
                        task_id=task_id,
                        field=f"depends_on[{index}]",
                        expected="string",
                        actual=type(value).__name__,
                        repairable=True,
                    )
                )
                continue
            depends_on.append(value)
        return tuple(depends_on), issues

    def _normalize_input_bindings(
        self,
        raw_input_bindings: object,
        task_id: str,
    ) -> tuple[tuple[PlannerInputBinding, ...], list[PlannerValidationIssue]]:
        if raw_input_bindings is None:
            return (), []
        if not isinstance(raw_input_bindings, list):
            return (), [
                self._issue(
                    "INVALID_INPUT_BINDING",
                    "input_bindings must be an array.",
                    "error",
                    task_id=task_id,
                    field="input_bindings",
                    expected="array[object]",
                    actual=type(raw_input_bindings).__name__,
                    repairable=True,
                )
            ]
        bindings: list[PlannerInputBinding] = []
        issues: list[PlannerValidationIssue] = []
        for index, raw_binding in enumerate(raw_input_bindings):
            if not isinstance(raw_binding, dict):
                issues.append(
                    self._issue(
                        "INVALID_INPUT_BINDING",
                        "Each input binding must be an object.",
                        "error",
                        task_id=task_id,
                        field=f"input_bindings[{index}]",
                        expected="object",
                        actual=type(raw_binding).__name__,
                        repairable=True,
                    )
                )
                continue
            try:
                bindings.append(PlannerInputBinding.model_validate(raw_binding))
            except ValueError as exc:
                issues.append(
                    self._issue(
                        "INVALID_INPUT_BINDING",
                        str(exc),
                        "error",
                        task_id=task_id,
                        field=f"input_bindings[{index}]",
                        repairable=True,
                    )
                )
        return tuple(bindings), issues

    def _validate_dependencies(self, tasks: list[PlannerTask]) -> list[PlannerValidationIssue]:
        task_ids = {task.task_id for task in tasks}
        issues: list[PlannerValidationIssue] = []
        for task in tasks:
            seen: set[str] = set()
            for dependency_id in task.depends_on:
                if dependency_id == task.task_id:
                    issues.append(
                        self._issue(
                            "SELF_TASK_DEPENDENCY",
                            f"Task '{task.task_id}' cannot depend on itself.",
                            "error",
                            task_id=task.task_id,
                            field="depends_on",
                            actual=dependency_id,
                            repairable=True,
                        )
                    )
                if dependency_id in seen:
                    issues.append(
                        self._issue(
                            "DUPLICATE_TASK_DEPENDENCY",
                            f"Task '{task.task_id}' contains duplicate dependency '{dependency_id}'.",
                            "error",
                            task_id=task.task_id,
                            field="depends_on",
                            actual=dependency_id,
                            repairable=True,
                        )
                    )
                seen.add(dependency_id)
                if dependency_id not in task_ids:
                    issues.append(
                        self._issue(
                            "UNKNOWN_DEPENDENCY_TASK",
                            f"Task '{task.task_id}' depends on unknown task '{dependency_id}'.",
                            "error",
                            task_id=task.task_id,
                            field="depends_on",
                            actual=dependency_id,
                            repairable=True,
                        )
                    )
        if not issues and self._detect_cycle(tasks):
            issues.append(
                self._issue(
                    "CYCLIC_TASK_DEPENDENCY",
                    "Task dependencies contain a cycle.",
                    "error",
                    task_id=tasks[0].task_id if tasks else None,
                    expected="acyclic dependency graph",
                    actual=[task.task_id for task in tasks],
                    repairable=True,
                )
            )
        return issues

    def _validate_input_bindings(self, tasks: list[PlannerTask]) -> list[PlannerValidationIssue]:
        task_ids = {task.task_id for task in tasks}
        issues: list[PlannerValidationIssue] = []
        for task in tasks:
            seen_input_names: set[str] = set()
            for binding in task.input_bindings:
                if binding.input_name in seen_input_names:
                    issues.append(
                        self._issue(
                            "DUPLICATE_INPUT_BINDING",
                            f"Task '{task.task_id}' has duplicate input binding '{binding.input_name}'.",
                            "error",
                            task_id=task.task_id,
                            field="input_bindings",
                            actual=binding.input_name,
                            repairable=True,
                        )
                    )
                seen_input_names.add(binding.input_name)
                if binding.source_type == "planner_context":
                    source_path = binding.source_path or ""
                    if source_path not in ALLOWED_PLANNER_CONTEXT_PATHS:
                        issues.append(
                            self._issue(
                                "INVALID_INPUT_BINDING",
                                f"Planner context path '{source_path}' is not allowed.",
                                "error",
                                task_id=task.task_id,
                                field="input_bindings.source_path",
                                expected=sorted(ALLOWED_PLANNER_CONTEXT_PATHS),
                                actual=source_path,
                                repairable=True,
                            )
                        )
                if binding.source_type != "task_output":
                    continue
                source_task_id = binding.source_task_id or ""
                if source_task_id not in task_ids:
                    issues.append(
                        self._issue(
                            "UNKNOWN_INPUT_SOURCE_TASK",
                            f"Input binding references unknown source task '{source_task_id}'.",
                            "error",
                            task_id=task.task_id,
                            field="input_bindings.source_task_id",
                            actual=source_task_id,
                            repairable=True,
                        )
                    )
                    continue
                if source_task_id not in task.depends_on:
                    issues.append(
                        self._issue(
                            "INPUT_SOURCE_NOT_DEPENDENCY",
                            "task_output binding source_task_id must also appear in depends_on.",
                            "error",
                            task_id=task.task_id,
                            field="input_bindings.source_task_id",
                            expected=list(task.depends_on),
                            actual=source_task_id,
                            repairable=True,
                        )
                    )
                issues.append(
                    self._issue(
                        "UNSUPPORTED_TASK_OUTPUT_BINDING",
                        "Task output bindings are declarative only in this step and cannot be executed as dataframe inputs.",
                        "warning",
                        task_id=task.task_id,
                        field="input_bindings",
                        actual={
                            "input_name": binding.input_name,
                            "source_task_id": source_task_id,
                            "source_path": binding.source_path,
                        },
                        repairable=False,
                    )
                )
        return issues

    def _attach_task_issues(
        self,
        tasks: list[PlannerTask],
        issues: list[PlannerValidationIssue],
    ) -> list[PlannerTask]:
        if not issues:
            return tasks
        issues_by_task: dict[str, list[PlannerValidationIssue]] = {}
        plan_level_issues: list[PlannerValidationIssue] = []
        for issue in issues:
            if issue.task_id:
                issues_by_task.setdefault(issue.task_id, []).append(issue)
            else:
                plan_level_issues.append(issue)
        updated: list[PlannerTask] = []
        for task in tasks:
            task_issues = [*task.validation_issues, *issues_by_task.get(task.task_id, ())]
            if any(issue.severity == "error" for issue in task_issues):
                validation_status = "invalid"
            else:
                validation_status = task.validation_status
            updated.append(
                task.model_copy(
                    update={
                        "validation_issues": tuple(task_issues),
                        "validation_status": validation_status,
                    }
                )
            )
        if plan_level_issues and updated:
            first = updated[0]
            task_issues = (*first.validation_issues, *plan_level_issues)
            updated[0] = first.model_copy(
                update={
                    "validation_issues": task_issues,
                    "validation_status": "invalid",
                }
            )
        return updated

    def _detect_cycle(self, tasks: list[PlannerTask]) -> bool:
        return len(self._topological_sort(tasks)) != len(tasks)

    def _topological_sort(self, tasks: list[PlannerTask]) -> tuple[str, ...]:
        """Return a stable Kahn topological order, or empty tuple for invalid graphs."""
        task_ids = [task.task_id for task in tasks]
        task_id_set = set(task_ids)
        dependencies: dict[str, list[str]] = {}
        children: dict[str, list[str]] = {task_id: [] for task_id in task_ids}
        for task in tasks:
            if task.task_id in task.depends_on:
                return ()
            if len(set(task.depends_on)) != len(task.depends_on):
                return ()
            if any(dependency_id not in task_id_set for dependency_id in task.depends_on):
                return ()
            dependencies[task.task_id] = list(task.depends_on)
            for dependency_id in task.depends_on:
                children[dependency_id].append(task.task_id)
        ready = [task_id for task_id in task_ids if not dependencies[task_id]]
        order: list[str] = []
        while ready:
            current = ready.pop(0)
            order.append(current)
            for child_id in children[current]:
                dependencies[child_id] = [item for item in dependencies[child_id] if item != current]
                if not dependencies[child_id] and child_id not in ready and child_id not in order:
                    ready.append(child_id)
            ready.sort(key=task_ids.index)
        return tuple(order) if len(order) == len(tasks) else ()

    def _build_execution_stages(self, tasks: list[PlannerTask]) -> tuple[PlannerExecutionStage, ...]:
        if not tasks:
            return ()
        task_by_id = {task.task_id: task for task in tasks}
        task_ids = [task.task_id for task in tasks]
        execution_order = self._topological_sort(tasks)
        if not execution_order:
            return ()
        stage_by_task_id: dict[str, int] = {}
        for task_id in execution_order:
            task = task_by_id[task_id]
            if task.depends_on:
                stage_by_task_id[task.task_id] = 1 + max(stage_by_task_id[dependency_id] for dependency_id in task.depends_on)
            else:
                stage_by_task_id[task.task_id] = 1
        stages: list[PlannerExecutionStage] = []
        for stage_order in sorted(set(stage_by_task_id.values())):
            stage_task_ids = tuple(task_id for task_id in task_ids if stage_by_task_id[task_id] == stage_order)
            stage_tasks = [task_by_id[task_id] for task_id in stage_task_ids]
            stages.append(
                PlannerExecutionStage(
                    stage_id=f"stage_{stage_order}",
                    order=stage_order,
                    task_ids=stage_task_ids,
                    parallelizable=len(stage_task_ids) > 1,
                    risk_level=self._highest_risk(task.risk_level for task in stage_tasks),
                    requires_confirmation=any(task.requires_confirmation for task in stage_tasks),
                    description=(
                        f"Stage {stage_order} contains {len(stage_task_ids)} task(s) derived from dependency order; "
                        "parallelizable describes dependency shape only, not actual concurrent execution."
                    ),
                )
            )
        return tuple(stages)

    def _apply_execution_metadata(
        self,
        tasks: list[PlannerTask],
        execution_order: tuple[str, ...],
        execution_stages: tuple[PlannerExecutionStage, ...],
    ) -> list[PlannerTask]:
        order_index = {task_id: index + 1 for index, task_id in enumerate(execution_order)}
        stage_index = {
            task_id: stage.order
            for stage in execution_stages
            for task_id in stage.task_ids
        }
        return [
            task.model_copy(
                update={
                    "execution_order": order_index.get(task.task_id),
                    "execution_group": stage_index.get(task.task_id),
                }
            )
            for task in tasks
        ]

    def _highest_risk(self, risk_levels) -> RiskLevel:
        ordered = sorted(risk_levels, key=lambda level: RISK_LEVEL_PRIORITY[level], reverse=True)
        return ordered[0] if ordered else "read_only"

    def _has_unsupported_conditional_request(self, question: str | None) -> bool:
        if not question:
            return False
        lowered = question.lower()
        conditional_tokens = (
            "if no problem",
            "if there is no problem",
            "if clean",
            "only if",
            "provided that",
            "如果",
            "假如",
            "若",
            "没有问题再",
            "通过后再",
            "否则",
        )
        return any(token in lowered for token in conditional_tokens)

    def _extract_raw_tasks(self, raw_output: object) -> list[object] | None:
        if isinstance(raw_output, list):
            return raw_output
        if isinstance(raw_output, tuple):
            return list(raw_output)
        if isinstance(raw_output, dict):
            tasks = raw_output.get("tasks")
            return tasks if isinstance(tasks, list) else None
        return None

    def _task_to_dict(self, raw_task: object) -> dict[str, object]:
        if isinstance(raw_task, AnalysisTask):
            return raw_task.model_dump()
        if isinstance(raw_task, dict):
            return dict(raw_task)
        return {}

    def _confidence(self, raw: object, source: PlannerTaskSource) -> tuple[float, PlannerValidationIssue | None]:
        if raw is None:
            return (0.85 if source == "fallback" else 0.75), None
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
            return 0.0, self._issue(
                "INVALID_CONFIDENCE",
                "confidence must be a finite number between 0 and 1.",
                "error",
                field="confidence",
                actual=type(raw).__name__,
                repairable=True,
            )
        value = float(raw)
        if value < 0 or value > 1:
            return max(0.0, min(1.0, value)), self._issue(
                "INVALID_CONFIDENCE",
                "confidence must be between 0 and 1.",
                "error",
                field="confidence",
                actual=value,
                repairable=True,
            )
        return value, None

    def _matches_parameter_type(self, value: object, type_name: str) -> bool:
        if type_name == "string":
            return isinstance(value, str)
        if type_name == "integer":
            return not isinstance(value, bool) and isinstance(value, int)
        if type_name == "number":
            return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))
        if type_name == "boolean":
            return isinstance(value, bool)
        if type_name == "array":
            return isinstance(value, list)
        if type_name == "object":
            return isinstance(value, dict)
        return False

    def _field_type_allowed(self, actual_type: str, requirement: ToolFieldRequirement) -> bool:
        accepted = set(requirement.accepted_types)
        excluded = set(requirement.exclude_types)
        if "any" in accepted:
            return actual_type not in excluded
        return actual_type in accepted and actual_type not in excluded

    def _strip_internal_params(
        self,
        task_type: str,
        raw_params: object,
        diagnostics_notes: list[str],
    ) -> object:
        """Remove planner-only safety flags before capability validation and execution adaptation."""
        if not isinstance(raw_params, dict):
            return raw_params
        params = dict(raw_params)
        if task_type == "anomaly_detection" and "detection_only" in params:
            params.pop("detection_only")
            diagnostics_notes.append("anomaly_detection:detection_only_internal_flag_removed")
        return params

    def _candidate_columns(
        self,
        request_context: PlannerRequestContext,
        accepted_types: set[str],
    ) -> list[str]:
        return [
            column.name
            for column in request_context.columns
            if column.type in accepted_types
        ]

    def _has_error(self, issues: list[PlannerValidationIssue]) -> bool:
        return any(issue.severity == "error" for issue in issues)

    def _logical_column_type(self, data_type: str, name: str) -> FieldType:
        lowered = data_type.lower()
        lowered_name = name.lower()
        if "bool" in lowered:
            return "boolean"
        if any(token in lowered for token in ("datetime", "timestamp", "date", "time")) or any(
            token in lowered_name for token in ("date", "time")
        ):
            return "datetime"
        if "timedelta" in lowered or "duration" in lowered:
            return "timedelta"
        if any(token in lowered for token in ("int", "float", "double", "decimal", "number")):
            return "numeric"
        if "category" in lowered or "categorical" in lowered:
            return "categorical"
        if any(token in lowered for token in ("object", "string", "str", "text")):
            return "text"
        return "any"

    def _risk_for_task(self, task_type: str, capability: ToolCapability | None) -> RiskLevel:
        if task_type in TASK_RISK_OVERRIDES:
            return TASK_RISK_OVERRIDES[task_type]
        return capability.risk_level if capability is not None else "read_only"

    def _requires_confirmation(self, task_type: str, risk_level: RiskLevel) -> bool:
        if risk_level == "external_side_effect":
            return True
        if risk_level == "mutating_copy":
            return task_type == "data_cleaning_execute"
        return False

    def _plan_status(
        self,
        tasks: list[PlannerTask],
        issues: list[PlannerValidationIssue],
    ) -> str:
        if any(issue.code in {"CLARIFICATION_REQUIRED", "CONDITIONAL_EXECUTION_UNSUPPORTED"} for issue in issues):
            return "clarification_required"
        if not tasks:
            return "unsupported"
        if any(issue.severity == "error" for issue in issues):
            if all(issue.code in {"UNKNOWN_TASK_TYPE", "UNSUPPORTED_REQUEST", "EMPTY_TASK_LIST"} for issue in issues):
                return "unsupported"
            return "invalid"
        return "valid"

    def _normalized_intent(self, tasks: list[PlannerTask], status: str) -> str:
        if tasks:
            return tasks[0].task_type
        return status

    def _intent(self, raw_output: object, question: str | None) -> str:
        if isinstance(raw_output, dict) and isinstance(raw_output.get("user_intent"), str):
            return str(raw_output["user_intent"])
        return question or ""

    def _stable_plan_id(self, intent: str, tasks: list[PlannerTask], source: PlannerSource) -> str:
        payload = {
            "intent": intent,
            "source": source,
            "tasks": [
                {
                    "task_id": task.task_id,
                    "task_type": task.task_type,
                    "tool_name": task.tool_name,
                    "params": task.params,
                    "depends_on": list(task.depends_on),
                    "input_bindings": [binding.model_dump(mode="json") for binding in task.input_bindings],
                }
                for task in tasks
            ],
        }
        digest = hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        return f"plan_{digest[:12]}"

    def _catalog_version(self) -> str:
        versions = {capability.version for capability in self.registry.list_capabilities()}
        return sorted(versions)[0] if versions else "1.0"

    def _range_issue(
        self,
        task_id: str,
        spec: ToolParameterSpec,
        bound: str,
        expected: float,
        actual: object,
    ) -> PlannerValidationIssue:
        return self._issue(
            "INVALID_PARAMETER_VALUE",
            f"Parameter '{spec.name}' is outside the configured {bound}.",
            "error",
            task_id=task_id,
            field=f"params.{spec.name}",
            parameter_name=spec.name,
            expected={bound: expected},
            actual=actual if isinstance(actual, (str, int, float, bool)) else type(actual).__name__,
            repairable=True,
        )

    def _issue(
        self,
        code: PlannerIssueCode,
        message: str,
        severity: str,
        *,
        task_id: str | None = None,
        field: str | None = None,
        parameter_name: str | None = None,
        expected: object = None,
        actual: object = None,
        repairable: bool = False,
    ) -> PlannerValidationIssue:
        return PlannerValidationIssue(
            code=code,
            message=message,
            severity=severity,
            task_id=task_id,
            field=field,
            parameter_name=parameter_name,
            expected=expected,
            actual=actual,
            repairable=repairable,
        )

    def _string(self, value: object) -> str | None:
        return value if isinstance(value, str) and value else None
