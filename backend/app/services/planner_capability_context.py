"""Bounded Planner-facing projection of the tool capability catalog."""

from __future__ import annotations

from app.tools.registry import ToolRegistry

DEFAULT_MAX_CAPABILITY_CONTEXT_LENGTH = 12_000
MIN_CAPABILITY_CONTEXT_LENGTH = 256


def build_planner_capability_context(
    registry: ToolRegistry,
    *,
    max_length: int = DEFAULT_MAX_CAPABILITY_CONTEXT_LENGTH,
) -> str:
    """Return stable compact text without invoking tools or an LLM."""
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < MIN_CAPABILITY_CONTEXT_LENGTH:
        raise ValueError(f"max_length must be an integer greater than or equal to {MIN_CAPABILITY_CONTEXT_LENGTH}.")

    header = "Available deterministic tool capabilities:"
    lines = [header]
    for capability in registry.list_capabilities():
        required = ",".join(capability.required_parameters) or "none"
        optional = ",".join(capability.optional_parameters) or "none"
        fields = ";".join(
            (
                f"{requirement.role}:"
                f"{','.join(requirement.accepted_types)}:"
                f"{'required' if requirement.required else 'conditional'}"
            )
            for requirement in capability.field_requirements
        ) or "none"
        limitations = " | ".join(capability.limitations[:2])
        line = (
            f"- {capability.tool_name}; task_types={','.join(capability.task_types)}; "
            f"description={capability.description[:180]}; required={required}; optional={optional}; "
            f"fields={fields}; risk={capability.risk_level}; limitations={limitations[:300]}"
        )
        candidate = "\n".join([*lines, line])
        if len(candidate) > max_length:
            suffix = "\n- capability catalog truncated by max_length"
            available = max_length - len("\n".join(lines))
            if available >= len(suffix):
                lines.append(suffix.lstrip("\n"))
            break
        lines.append(line)
    return "\n".join(lines)[:max_length]
