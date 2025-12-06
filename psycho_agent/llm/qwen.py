"""OpenAI-compatible client wrapper for Qwen."""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

from openai import OpenAI

from ..config import settings

LOGGER = logging.getLogger(__name__)

DEFAULT_BASE_URL = settings.qwen.base_url
DEFAULT_API_KEY = settings.qwen.api_key
DEFAULT_MODEL = settings.qwen.model

_client_lock = threading.Lock()
_cached_client: OpenAI | None = None


def get_qwen_client() -> OpenAI:
    """Lazily initialise the OpenAI-compatible client for Qwen."""
    global _cached_client
    if _cached_client is not None:
        return _cached_client

    with _client_lock:
        if _cached_client is None:
            LOGGER.debug("Initialising Qwen OpenAI-compatible client at %s", DEFAULT_BASE_URL)
            _cached_client = OpenAI(base_url=DEFAULT_BASE_URL, api_key=DEFAULT_API_KEY)
    return _cached_client


class QwenLLM:
    """Thin convenience wrapper that normalises chat completions."""

    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        timeout_ms: int | None = None,
    ) -> None:
        self._client = get_qwen_client()
        self._model = model or settings.qwen.model
        self._temperature = temperature if temperature is not None else settings.qwen.temperature
        self._max_tokens = max_output_tokens or settings.qwen.max_output_tokens
        self._timeout_ms = timeout_ms or settings.qwen.timeout

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        request = {
            "model": self._model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self._temperature),
            "max_tokens": kwargs.get("max_tokens", self._max_tokens),
            "timeout": kwargs.get("timeout", self._timeout_ms / 1000),
        }
        LOGGER.debug("Calling Qwen chat with %d messages", len(messages))
        completion = self._client.chat.completions.create(**request)
        return completion.choices[0].message.content or ""

    def structured_chat(self, messages: List[Dict[str, str]], response_format: Optional[Dict[str, Any]] = None) -> str:
        request = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        if response_format:
            request["response_format"] = response_format
        completion = self._client.chat.completions.create(**request)
        return completion.choices[0].message.content or ""
