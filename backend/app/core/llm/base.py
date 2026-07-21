"""
Abstract LLM Provider interface.

Defines the contract every LLM provider must fulfil.  The business layer
imports *only* this abstraction — never a concrete provider class — so
the system obeys the Dependency Inversion Principle (D in SOLID):

    High-level modules (services) depend on the abstraction.
    Low-level modules (providers) also depend on the abstraction.

To add a new provider (e.g. OpenAI, Claude, Gemini):

1. Subclass ``LLMProvider``.
2. Implement the three abstract methods.
3. Register the provider in ``factory.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator

from app.core.llm.schemas import ChatMessage, LLMResponse


class LLMProvider(ABC):
    """Uniform interface for any LLM backend.

    Every method receives a list of ``ChatMessage`` objects and returns
    either a plain string, a parsed structured dict, or an async stream
    of chunks.  Provider-specific knobs (temperature, top_p, max_tokens,
    etc.) are passed as ``**kwargs`` and silently ignored when the
    backend does not support them — this keeps the calling code portable.

    Subclasses must be **stateless with respect to the conversation**:
    the full message history is passed in on every call.
    """

    # ------------------------------------------------------------------
    # Abstract methods — override in every subclass
    # ------------------------------------------------------------------

    @abstractmethod
    def chat(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        **kwargs: object,
    ) -> LLMResponse:
        """Send a chat-completion request and return the full response.

        Args:
            messages: Ordered conversation history (system → user → assistant → …).
            model: Optional model override; when ``None`` the provider uses
                its default (usually read from ``Settings.llm_model``).
            **kwargs: Provider-specific parameters (temperature, max_tokens,
                top_p, …).  Unknown keys are ignored silently.

        Returns:
            ``LLMResponse`` with the assistant content and optional metadata.

        Raises:
            ``AppException`` (or a subclass) on transport / auth / server errors.
        """
        ...

    @abstractmethod
    async def stream_chat(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        **kwargs: object,
    ) -> AsyncIterator[str]:
        """Stream a chat-completion response as an async iterable of tokens.

        The returned iterator yields **delta text** chunks (not SSE frames),
        so callers can consume them with ``async for chunk in …``.

        Args:
            messages: Same semantics as ``chat()``.
            model: Same semantics as ``chat()``.
            **kwargs: Same semantics as ``chat()``.

        Yields:
            One string per streaming chunk.  The concatenation of all
            chunks equals the full assistant message.
        """
        ...

    @abstractmethod
    def structured_output(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        **kwargs: object,
    ) -> dict[str, object]:
        """Send a chat-completion request that returns **parsed JSON**.

        This is the most common pattern in the current codebase: ask the
        LLM for a JSON response and receive a ``dict`` directly.

        Internally the provider should set ``response_format`` or the
        equivalent to enforce JSON output, then parse the returned text
        with ``json.loads``.

        Args:
            messages: Same semantics as ``chat()``.
            model: Same semantics as ``chat()``.
            **kwargs: Same semantics as ``chat()``.

        Returns:
            The parsed JSON object from the assistant content.

        Raises:
            ``StructuredOutputError`` when the response is not valid JSON.
        """
        ...
