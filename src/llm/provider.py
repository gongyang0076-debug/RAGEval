"""Application-facing contract, independent of the OpenAI SDK."""

from abc import ABC, abstractmethod

from .models import LLMResponse, ProviderFailure


class ProviderError(RuntimeError):
    def __init__(self, details: ProviderFailure):
        self.details = details
        super().__init__(details.message)


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, messages: list[dict[str, str]], *, temperature: float | None = None,
                 response_format: dict | None = None, max_attempts: int | None = None) -> LLMResponse:
        """Return text with execution trace, or raise a classified ProviderError."""
