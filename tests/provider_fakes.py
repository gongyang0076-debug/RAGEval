# 文件作用：提供假的 SDK 响应和可注入测试的 Provider 适配器。
# 为什么有它：保留真实重试逻辑，同时阻止单元测试访问外部模型或真的等待退避。
"""Mock SDK boundary while exercising the actual Provider implementation."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from src.llm import LLMProvider
from src.llm.openai_compatible import OpenAICompatibleProvider


# 做什么：把文本包装成 SDK 风格的响应对象。
# 为什么需要：客户端测试可以像读取真实响应一样读取预设内容。
def completion(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


# 做什么：创建支持异步上下文管理的 SDK 替身。
# 为什么需要：适配 Provider 内部的异步调用方式。
def async_sdk():
    sdk = AsyncMock()
    sdk.__aenter__.return_value = sdk
    return sdk


# 这个类：在测试中用假的 SDK 执行真实 Provider 逻辑。
# 为什么需要：同时验证重试行为和避免外部调用。
class SDKFixtureProvider(LLMProvider):
    # 做什么：将同步测试响应接到异步替身，并构造实际 Provider。
    # 为什么需要：沿用真实重试代码而不连接外部服务。
    def __init__(self, settings, sdk):
        self.sdk = async_sdk()
        # 做什么：把异步请求转交给传入的同步 SDK 替身。
        # 为什么需要：复用现有测试设定的返回值与异常。
        async def create(**kwargs):
            return sdk.chat.completions.create(**kwargs)
        self.sdk.chat.completions.create.side_effect = create
        self.provider = OpenAICompatibleProvider(
            api_key=settings.api_key, base_url=settings.base_url, model=settings.model,
            timeout_seconds=settings.timeout_seconds, max_attempts=settings.max_attempts,
            retry_delay_seconds=settings.retry_delay_seconds,
        )

    # 做什么：替换真实 SDK 和等待函数后执行实际 Provider。
    # 为什么需要：测试接口策略同时避免联网和真实退避等待。
    def generate(self, messages, **kwargs):
        with patch("src.llm.openai_compatible.AsyncOpenAI", return_value=self.sdk), patch("src.llm.openai_compatible.sleep", new_callable=AsyncMock):
            return self.provider.generate(messages, **kwargs)
