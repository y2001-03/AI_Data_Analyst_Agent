"""
LLM Provider Factory.

Centralises provider instantiation so the rest of the application never
imports a concrete provider class.  To swap backends, change the
``LLM_PROVIDER`` environment variable (or ``Settings.llm_provider``).

Supported provider keys (extend by adding entries to ``_PROVIDER_REGISTRY``):

+-------------+---------------------------+-------------------------------+
| Key         | Class                     | Notes                         |
+=============+===========================+===============================+
| ``"qwen"``  | ``QwenProvider``          | DashScope compatible mode     |
+-------------+---------------------------+-------------------------------+

Adding a new provider (e.g. OpenAI):

1. Create ``openai_provider.py`` in this package.
2. Add the import inside ``_PROVIDER_REGISTRY`` below.
3. Done — no changes needed in the service layer.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol

from app.core.config import get_settings
from app.core.llm.base import LLMProvider


class _ProviderFactory(Protocol):
    """Callable that returns an ``LLMProvider`` with no arguments."""

    def __call__(self) -> LLMProvider: ...


def _create_qwen_provider() -> LLMProvider:
    """Lazily import and construct the Qwen (DashScope) provider."""
    from app.core.llm.qwen_provider import QwenProvider  # noqa: PLC0415

    return QwenProvider()


# ---------------------------------------------------------------------------
# Provider registry — extend this dict to add new backends
# ---------------------------------------------------------------------------

_PROVIDER_REGISTRY: dict[str, _ProviderFactory] = {
    "qwen": _create_qwen_provider,
    # Future providers (uncomment and implement):
    # "openai":   _create_openai_provider,
    # "deepseek": _create_deepseek_provider,
    # "claude":   _create_claude_provider,
    # "gemini":   _create_gemini_provider,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_llm_provider(provider_key: str | None = None) -> LLMProvider:
    """Return the configured LLM provider (singleton per process).

    Args:
        provider_key: Optional override.  When ``None`` reads
            ``Settings.llm_provider`` (env ``LLM_PROVIDER``).

    Returns:
        A cached ``LLMProvider`` instance.

    Raises:
        ValueError: When the configured provider key is not registered.
    """
    settings = get_settings()
    key = provider_key or settings.llm_provider
    factory = _PROVIDER_REGISTRY.get(key)
    if factory is None:
        registered = ", ".join(sorted(_PROVIDER_REGISTRY.keys()))
        raise ValueError(
            f"Unknown LLM provider '{key}'. "
            f"Registered providers: {registered}. "
            f"Set LLM_PROVIDER to one of them."
        )
    return factory()


def register_provider(key: str, factory: _ProviderFactory) -> None:
    """Register a custom provider at runtime (for tests / extensions).

    Args:
        key: Provider identifier (e.g. ``"my_mock"``).
        factory: A zero-argument callable that returns an ``LLMProvider``.
    """
    _PROVIDER_REGISTRY[key] = factory


def list_providers() -> list[str]:
    """Return the sorted list of registered provider keys."""
    return sorted(_PROVIDER_REGISTRY.keys())
