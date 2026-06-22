"""Memory service placeholder."""


class MemoryService:
    """Placeholder memory service."""

    def load_context(self, session_id: str) -> str:
        """Return placeholder context for a session."""
        return f"Memory context placeholder for session: {session_id}"

    def save_context(self, session_id: str, content: str) -> None:
        """Persist placeholder memory content."""
        _ = (session_id, content)

