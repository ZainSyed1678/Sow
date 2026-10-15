"""Groq LLM provider client using OpenAI-compatible API and JSON mode."""

from __future__ import annotations

import json
import re
import time
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from ai_ml_notion_researcher.config import LLMConfig
from ai_ml_notion_researcher.llm.base import (
    BaseLLMClient,
    LLMError,
    LLMRateLimitError,
    LLMValidationError,
)
from ai_ml_notion_researcher.logging_utils import get_logger

logger = get_logger("llm.groq")
T = TypeVar("T", bound=BaseModel)


class GroqClient(BaseLLMClient):
    """Client for Groq's low-latency inference API."""

    def __init__(
        self,
        config: LLMConfig,
        client: httpx.Client | None = None,
    ):
        self.config = config
        self.api_key = config.api_key.get_secret_value() if config.api_key else ""
        self.base_url = config.base_url.rstrip("/")
        self.model = config.model
        self.max_retries = config.max_retries
        self.timeout_seconds = config.timeout_seconds

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        self._client = client or httpx.Client(
            headers=headers,
            timeout=self.timeout_seconds,
        )

    def __enter__(self) -> GroqClient:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def _execute_chat_completion(
        self,
        messages: list[dict[str, str]],
        json_mode: bool = True,
    ) -> str:
        """Executes chat completion with bounded retries on 429 rate limits or transient errors."""
        url = f"{self.base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        attempt = 0
        backoff = 2.0

        while attempt <= self.max_retries:
            try:
                resp = self._client.post(url, json=payload)

                if resp.status_code == 200:
                    data = resp.json()
                    choices = data.get("choices", [])
                    if not choices:
                        raise LLMError("Groq API returned response with no choices.")
                    return choices[0]["message"]["content"]

                if resp.status_code == 429:
                    attempt += 1
                    if attempt > self.max_retries:
                        raise LLMRateLimitError(
                            "Groq API rate limit exceeded after maximum retries.",
                        )
                    retry_after = resp.headers.get("Retry-After")
                    sleep_time = float(retry_after) if retry_after else 0.0

                    # Parse 'Please try again in X.Xs' from error message if available
                    if sleep_time <= 0:
                        match = re.search(r"try again in (\d+(?:\.\d+)?)s", resp.text)
                        if match:
                            sleep_time = float(match.group(1)) + 0.5

                    # Fallback to exponential backoff with a 5.0s minimum floor
                    if sleep_time <= 0:
                        sleep_time = max(backoff ** attempt, 5.0)

                    logger.warning(
                        f"Groq rate limit (429) hit. Backing off for {sleep_time:.2f}s...",
                        extra={"attempt": attempt, "sleep_time": sleep_time},
                    )
                    time.sleep(sleep_time)
                    continue

                if 500 <= resp.status_code < 600:
                    attempt += 1
                    if attempt > self.max_retries:
                        raise LLMError(f"Groq server error ({resp.status_code}): {resp.text}")
                    time.sleep(backoff ** attempt)
                    continue

                # Any other client error (e.g. 400, 401, 403)
                raise LLMError(f"Groq API error ({resp.status_code}): {resp.text}")

            except httpx.RequestError as exc:
                attempt += 1
                if attempt > self.max_retries:
                    raise LLMError(f"HTTP transport error communicating with Groq: {exc}") from exc
                time.sleep(backoff ** attempt)

        raise LLMError("Unexpected termination of retry loop in GroqClient.")

    def generate_structured(
        self,
        prompt: str,
        system_prompt: str,
        response_model: type[T],
    ) -> T:
        """Invokes Groq in JSON mode and validates the output strictly against response_model."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        raw_response = self._execute_chat_completion(messages, json_mode=True)

        try:
            # Parse into dict first to handle potential markdown wrapper ```json ... ```
            clean_json = raw_response.strip()
            if clean_json.startswith("```"):
                # Strip code fence
                lines = clean_json.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                clean_json = "\n".join(lines).strip()

            parsed_data = json.loads(clean_json)
            return response_model.model_validate(parsed_data)
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.error(f"Failed to parse or validate LLM response: {exc}\nRaw: {raw_response[:300]}")
            raise LLMValidationError(f"Groq response did not conform to schema: {exc}") from exc
