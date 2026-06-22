"""Analysis schemas."""

from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    """User analysis request."""

    session_id: str = Field(..., description="Conversation session identifier.")
    question: str = Field(..., description="Natural language analysis request.")
    file_id: str | None = Field(default=None, description="Uploaded file identifier.")


class ChatAnalysisRequest(BaseModel):
    """Question-driven dataset analysis request."""

    session_id: str | None = Field(default=None, description="Conversation session identifier.")
    question: str = Field(..., description="Natural language analysis request.")
    dataset_id: str | None = Field(default=None, description="Uploaded dataset identifier.")
    file_name: str | None = Field(default=None, description="Uploaded dataset file reference.")


class AnalysisResponse(BaseModel):
    """Agent response payload."""

    session_id: str
    answer: str
    steps: list[str]
    charts: list[dict[str, str]]
    report_markdown: str | None = None
