"""Anthropic Claude client for answer generation."""
from __future__ import annotations

from typing import Iterator

from anthropic import Anthropic, APIError, APIStatusError, RateLimitError


class GenerationError(Exception):
    """Raised when the LLM fails to produce an answer."""


class ClaudeClient:
    """Thin wrapper around Anthropic's messages API."""

    def __init__(self, api_key: str, model: str, max_tokens: int = 1024) -> None:
        if not api_key:
            raise GenerationError("Anthropic API key is empty.")
        self._client = Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
        except RateLimitError as exc:
            raise GenerationError(f"Anthropic rate limit reached: {exc}") from exc
        except APIStatusError as exc:
            raise GenerationError(f"Anthropic returned status {exc.status_code}: {exc.message}") from exc
        except APIError as exc:
            raise GenerationError(f"Anthropic API error: {exc}") from exc

        if not response.content:
            raise GenerationError("Anthropic returned an empty response.")

        text_blocks = [block.text for block in response.content if block.type == "text"]
        if not text_blocks:
            raise GenerationError("Anthropic response contained no text blocks.")

        return "\n".join(text_blocks).strip()

    def stream(self, system_prompt: str, user_prompt: str) -> Iterator[str]:
        """Yield response text tokens as they arrive — used by the streaming UI."""
        try:
            with self._client.messages.stream(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            ) as stream:
                yield from stream.text_stream
        except RateLimitError as exc:
            raise GenerationError(f"Anthropic rate limit reached: {exc}") from exc
        except APIStatusError as exc:
            raise GenerationError(f"Anthropic returned status {exc.status_code}: {exc.message}") from exc
        except APIError as exc:
            raise GenerationError(f"Anthropic API error: {exc}") from exc
