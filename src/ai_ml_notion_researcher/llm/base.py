"""Abstract base interface for modular LLM providers."""

from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """Base exception for LLM provider errors."""


class LLMRateLimitError(LLMError):
    """Raised when LLM API rate limit persists after bounded retries."""


class LLMValidationError(LLMError):
    """Raised when LLM output fails schema validation."""


class BaseLLMClient(ABC):
    """Abstract interface for LLM inference providers."""

    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        system_prompt: str,
        response_model: type[T],
    ) -> T:
        """Generates structured output strictly conforming to the given Pydantic model."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Closes underlying HTTP sessions."""
        raise NotImplementedError
