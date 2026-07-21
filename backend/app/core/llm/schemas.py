"""
LLM Provider domain schemas.

These are provider-agnostic data models shared across all LLM provider
implementations.  They decouple the business layer from any particular
vendor's wire format.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class ChatMessage(BaseModel):
    """A single message in a chat-completion conversation.

    Mirrors the de-facto industry format (OpenAI / DashScope / etc.) so
    every provider can map to its native representation without leaking
    vendor-specific details into the service layer.
    """

    role: Literal["system", "user", "assistant"] = "user"
    content: str


class LLMResponse(BaseModel):
    """Normalised response from any LLM provider.

    Providers are responsible for translating their native response into
    this canonical shape so callers never depend on a specific API
    structure.
    """

    content: str
    """The text content of the first (and usually only) choice."""

    model: str | None = None
    """The model identifier the provider actually served."""

    usage: dict[str, int] | None = None
    """Optional token-usage breakdown (prompt_tokens, completion_tokens, ...)."""


class StructuredOutputError(Exception):
    """Raised when a provider cannot return valid structured output."""

    def __init__(self, message: str, raw_content: str | None = None) -> None:
        super().__init__(message)
        self.raw_content = raw_content
