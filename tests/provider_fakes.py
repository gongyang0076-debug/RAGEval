"""Mock SDK boundary while exercising the actual Provider implementation."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from src.llm import LLMProvider
from src.llm.openai_compatible import OpenAICompatibleProvider


def completion(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def async_sdk():
    sdk = AsyncMock()
    sdk.__aenter__.return_value = sdk
    return sdk


class SDKFixtureProvider(LLMProvider):
    def __init__(self, settings, sdk):
        self.sdk = async_sdk()
        async def create(**kwargs):
            return sdk.chat.completions.create(**kwargs)
        self.sdk.chat.completions.create.side_effect = create
        self.provider = OpenAICompatibleProvider(
            api_key=settings.api_key, base_url=settings.base_url, model=settings.model,
            timeout_seconds=settings.timeout_seconds, max_attempts=settings.max_attempts,
            retry_delay_seconds=settings.retry_delay_seconds,
        )

    def generate(self, messages, **kwargs):
        with patch("src.llm.openai_compatible.AsyncOpenAI", return_value=self.sdk), patch("src.llm.openai_compatible.sleep", new_callable=AsyncMock):
            return self.provider.generate(messages, **kwargs)
