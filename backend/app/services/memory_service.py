"""Lightweight in-memory chat context storage."""

from __future__ import annotations


class MemoryService:
    """Store minimal per-session context for chat analysis."""

    _memory: dict[str, dict[str, object]] = {}

    def set(self, session_id: str, key: str, value: object) -> None:
        """Set one memory field for a session."""
        memory = self._memory.setdefault(session_id, {})
        memory[key] = value

    def get(self, session_id: str) -> dict[str, object]:
        """Return memory for a session."""
        return dict(self._memory.get(session_id, {}))

    def update_context(
        self,
        session_id: str,
        question: str | None,
        dataset_info: dict[str, object] | None,
        last_execution: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Merge the latest chat context into memory."""
        memory = self._memory.setdefault(session_id, {})
        if question:
            memory["last_question"] = question
        if dataset_info is not None:
            memory["dataset_info"] = dataset_info
        if last_execution is not None:
            memory["last_execution"] = last_execution
        return dict(memory)
