"""RAG schemas."""

from pydantic import BaseModel


class RetrievalRequest(BaseModel):
    """RAG retrieval request."""

    query: str
    top_k: int = 5

