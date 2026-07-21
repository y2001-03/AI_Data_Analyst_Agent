"""
Qwen LLM Provider (DashScope / Alibaba Cloud).

Implements the ``LLMProvider`` interface for Alibaba Cloud's DashScope
compatible-mode API, which speaks the OpenAI chat-completions protocol.

Usage::

    from app.core.llm.factory import get_llm_provider

    provider = get_llm_provider()
    response = provider.structured_output(messages, model="qwen-plus")
"""

from __future__ import annotations

import json
import time
from socket import timeout as SocketTimeout
from typing import AsyncIterator
from urllib import error, request

from app.core.config import get_settings
from app.core.exceptions import AppException
from app.core.llm.base import LLMProvider
from app.core.llm.schemas import ChatMessage, LLMResponse, StructuredOutputError
from app.core.logging import get_logger

logger = get_logger(__name__)


class QwenProvider(LLMProvider):
    """DashScope-compatible (OpenAI-protocol) LLM provider.

    Talks to ``https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions``
    using the standard Bearer-token auth scheme.

    .. note::

        This class is **not** imported directly by business services.
        Use ``get_llm_provider()`` from the factory module instead.
    """

    # ------------------------------------------------------------------
    # Provider identity
    # ------------------------------------------------------------------

    endpoint: str = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    """DashScope compatible-mode chat endpoint (OpenAI-protocol)."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str | None = None,
        timeout: int | None = None,
        max_retries: int | None = None,
    ) -> None:
        """Initialise the Qwen provider.

        Args:
            api_key: DashScope API key.  When ``None`` reads ``DASHSCOPE_API_KEY``
                from the application settings.
            default_model: Model id used when ``model`` is not passed to a call.
                Defaults to ``Settings.llm_model``.
            timeout: HTTP request timeout in seconds.
                Defaults to ``Settings.llm_timeout`` (currently 30).
            max_retries: Number of retries on transient failures.
                Defaults to ``Settings.llm_max_retries`` (currently 2).
        """
        settings = get_settings()
        self._api_key = api_key if api_key is not None else settings.dashscope_api_key
        self._default_model = default_model or settings.llm_model
        self._timeout = timeout if timeout is not None else settings.llm_timeout
        self._max_retries = max_retries if max_retries is not None else settings.llm_max_retries

    # ------------------------------------------------------------------
    # Public API — LLMProvider implementation
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        **kwargs: object,
    ) -> LLMResponse:
        """Send a chat-completion request (sync, with retry)."""
        resolved_model = model or self._default_model
        start_ts = time.monotonic()
        payload = self._build_payload(messages, resolved_model, **kwargs)
        try:
            body = self._send_with_retry(payload)
            response = self._parse_response(body, resolved_model)
        except Exception:
            elapsed_ms = (time.monotonic() - start_ts) * 1000
            self._log_llm_call(
                model=resolved_model,
                latency_ms=elapsed_ms,
                success=False,
                message_count=len(messages),
                error=str(
                    AppException("LLM call failed", error_code="LLM_PROVIDER_ERROR", status_code=502)
                ),
            )
            raise
        elapsed_ms = (time.monotonic() - start_ts) * 1000
        self._log_llm_call(
            model=resolved_model,
            latency_ms=elapsed_ms,
            success=True,
            message_count=len(messages),
            token_usage=response.usage,
        )
        return response

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        **kwargs: object,
    ) -> AsyncIterator[str]:
        """Stream a chat-completion response.

        .. note::

            This implementation uses a **synchronous** HTTP call under the hood
            (``urllib``) wrapped as an async generator.  For high-concurrency
            streaming prefer an ``httpx``-backed provider registered in the
            factory.
        """
        resolved_model = model or self._default_model
        streaming_kwargs = {**kwargs, "stream": True}
        payload = self._build_payload(messages, resolved_model, **streaming_kwargs)

        # urllib does not natively support async iteration, so we collect
        # the full SSE body and yield chunks from an in-memory buffer.
        body = self._send_with_retry(payload)
        for line in body.splitlines():
            line = line.strip()
            if not line or not line.startswith("data:"):
                continue
            data_str = line[len("data:"):].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    yield content
            except (json.JSONDecodeError, KeyError, IndexError):
                continue

    def structured_output(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        **kwargs: object,
    ) -> dict[str, object]:
        """Chat-completion with enforced JSON response → parsed ``dict``.

        Sets ``response_format={"type": "json_object"}`` (the OpenAI-compatible
        protocol) so the model is instructed to produce valid JSON.
        """
        json_kwargs = {**kwargs, "response_format": {"type": "json_object"}}
        response = self.chat(messages, model=model, **json_kwargs)
        try:
            result = json.loads(response.content)
        except json.JSONDecodeError as exc:
            raise StructuredOutputError(
                f"LLM returned invalid JSON: {exc}",
                raw_content=response.content,
            ) from exc
        if not isinstance(result, dict):
            raise StructuredOutputError(
                f"LLM returned {type(result).__name__} instead of a JSON object.",
                raw_content=response.content,
            )
        return result

    # ------------------------------------------------------------------
    # Internal helpers — payload / headers / parsing
    # ------------------------------------------------------------------

    def _build_payload(
        self,
        messages: list[ChatMessage],
        model: str,
        **kwargs: object,
    ) -> dict[str, object]:
        """Construct the OpenAI-protocol JSON body."""
        serialised_messages = [
            {"role": msg.role, "content": msg.content} for msg in messages
        ]
        payload: dict[str, object] = {
            "model": model,
            "messages": serialised_messages,
        }
        # Merge provider-agnostic kwargs (temperature, top_p, …) into the body.
        for key, value in kwargs.items():
            if key == "response_format":
                payload[key] = value
            elif key not in ("stream",):
                payload[key] = value
        # stream must appear *after* kwargs so the caller's explicit stream=True wins.
        if "stream" in kwargs:
            payload["stream"] = kwargs["stream"]
        return payload

    def _build_headers(self) -> dict[str, str]:
        """Return HTTP headers required by the DashScope compatible endpoint."""
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _parse_response(self, body: str, model: str) -> LLMResponse:
        """Extract content and usage from an OpenAI-protocol JSON body."""
        try:
            payload = json.loads(body)
            content = payload["choices"][0]["message"]["content"]
            usage = payload.get("usage")
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise AppException(
                f"LLM provider returned an invalid response: {exc}", 502
            ) from exc
        return LLMResponse(
            content=content,
            model=payload.get("model", model),
            usage=usage if isinstance(usage, dict) else None,
        )

    # ------------------------------------------------------------------
    # Transport — request / retry
    # ------------------------------------------------------------------

    def _send_with_retry(self, payload: dict[str, object]) -> str:
        """POST the payload to the endpoint with retry on transient errors."""
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return self._send_request(payload)
            except (error.HTTPError, error.URLError, SocketTimeout, TimeoutError) as exc:
                last_error = exc
            except Exception as exc:
                last_error = exc
        if last_error is None:
            raise AppException("LLM request failed without an error.", 502)
        if isinstance(last_error, error.HTTPError):
            detail = self._read_error_body(last_error)
            raise AppException(f"LLM request failed: {detail}", 502) from last_error
        if isinstance(last_error, error.URLError):
            raise AppException("LLM provider is unreachable.", 502) from last_error
        raise last_error

    def _send_request(self, payload: dict[str, object]) -> str:
        """Execute a single HTTP POST and return the raw response body."""
        req = request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=self._build_headers(),
            method="POST",
        )
        with request.urlopen(req, timeout=self._timeout) as response:
            return response.read().decode("utf-8")

    @staticmethod
    def _read_error_body(http_error: error.HTTPError) -> str:
        """Safely decode an HTTP error response body."""
        try:
            return http_error.read().decode("utf-8", errors="ignore")
        except Exception:
            return str(http_error)

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    @staticmethod
    def _log_llm_call(
        *,
        model: str,
        latency_ms: float,
        success: bool,
        message_count: int,
        token_usage: dict[str, int] | None = None,
        error: str | None = None,
    ) -> None:
        """Emit a structured log record for one LLM invocation.

        **Never** includes the API key, full message content, or raw
        response payload — only metadata useful for monitoring and
        debugging.
        """
        prompt_tokens = token_usage.get("prompt_tokens") if token_usage else None
        completion_tokens = token_usage.get("completion_tokens") if token_usage else None
        total_tokens = token_usage.get("total_tokens") if token_usage else None

        if success:
            logger.info(
                "LLM call succeeded | provider=qwen model=%s latency=%.0fms "
                "messages=%d prompt_tokens=%s completion_tokens=%s total_tokens=%s",
                model,
                latency_ms,
                message_count,
                prompt_tokens,
                completion_tokens,
                total_tokens,
            )
        else:
            logger.error(
                "LLM call failed | provider=qwen model=%s latency=%.0fms "
                "messages=%d error=%s",
                model,
                latency_ms,
                message_count,
                error,
            )
