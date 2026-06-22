"""RAG service placeholder."""


class RAGService:
    """Placeholder RAG service using fixed references."""

    def retrieve(self, query: str) -> list[str]:
        """Return placeholder knowledge references."""
        return [
            f"Best practice snippet for query: {query}",
            "Metric definition handbook placeholder",
        ]

