"""Memory schemas."""

from pydantic import BaseModel


class MemoryRecord(BaseModel):
    """Memory entry model."""

    session_id: str
    content: str

