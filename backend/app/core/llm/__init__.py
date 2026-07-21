"""
LLM Provider Adapter — enterprise abstraction over AI model backends.

Public API surface::

    from app.core.llm import (
        ChatMessage,           # schema
        LLMProvider,           # abstract base
        LLMResponse,           # schema
        StructuredOutputError, # schema
        get_llm_provider,      # factory function (preferred)
        list_providers,        # introspection
    )

Design principles:
- **SOLID**: services depend on ``LLMProvider``, not on DashScope / OpenAI.
- **DI**: the factory wires concrete providers; callers never use ``import``.
- **Open/Closed**: add a new provider by writing one file + registering it.
"""

from app.core.llm.base import LLMProvider
from app.core.llm.factory import get_llm_provider, list_providers, register_provider
from app.core.llm.schemas import ChatMessage, LLMResponse, StructuredOutputError

__all__ = [
    "ChatMessage",
    "LLMProvider",
    "LLMResponse",
    "StructuredOutputError",
    "get_llm_provider",
    "list_providers",
    "register_provider",
]
