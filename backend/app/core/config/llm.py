"""
LLM / AI-provider configuration.

Groups every setting related to the language-model backend:
provider selection, API credentials, model identifiers, timeout,
and retry policy.

This is a **mixin** — see ``app.core.config.app`` for the pattern
rationale.
"""

from pydantic import Field


class LLMConfigMixin:
    """LLM provider fields merged flat into ``AppSettings``.

    .. note::

        The field ``dashscope_api_key`` is retained with its original
        name and ``DASHSCOPE_API_KEY`` alias for **full backward
        compatibility**.  When adding a new provider (e.g. OpenAI),
        add a corresponding ``openai_api_key`` (alias ``OPENAI_API_KEY``)
        to this mixin — the factory will route the correct key to each
        provider.
    """

    llm_provider: str = Field(
        default="qwen",
        alias="LLM_PROVIDER",
        description="Provider key registered in ``app.core.llm.factory`` (qwen | openai | …).",
    )

    dashscope_api_key: str = Field(
        default="",
        alias="DASHSCOPE_API_KEY",
        description="DashScope / Alibaba Cloud API key.  Used by ``QwenProvider``.",
    )

    llm_model: str = Field(
        default="deepseek-v4-flash",
        alias="LLM_MODEL",
        description="Default model identifier passed to the LLM provider on every call.",
    )

    embedding_model: str = Field(
        default="text-embedding-v3",
        alias="EMBEDDING_MODEL",
        description="Model identifier for text-embedding (RAG / vector search).",
    )

    llm_timeout: int = Field(
        default=30,
        alias="LLM_TIMEOUT",
        description="HTTP request timeout in seconds for LLM API calls.",
    )

    llm_max_retries: int = Field(
        default=2,
        alias="LLM_MAX_RETRIES",
        description="Maximum retry attempts on transient LLM API failures.",
    )
