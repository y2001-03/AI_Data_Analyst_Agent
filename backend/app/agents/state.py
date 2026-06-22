"""LangGraph state definitions."""

from typing import Any, TypedDict


class AgentState(TypedDict):
    """Mutable workflow state."""

    session_id: str
    question: str
    file_id: str | None
    messages: list[str]
    steps: list[str]
    charts: list[dict[str, str]]
    report_markdown: str | None
    intent: str | None
    data_summary: dict[str, Any] | None
    plan: list[str] | None
    final_answer: str

