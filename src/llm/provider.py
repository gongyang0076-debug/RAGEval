# 文件作用：定义模型服务调用的统一接口和通用错误对象。
# 为什么有它：让 RAG 和 Judge 依赖统一约定，便于替换供应商和注入测试替身。
"""Application-facing contract, independent of the OpenAI SDK."""

from abc import ABC, abstractmethod

from .models import LLMResponse, ProviderFailure


# 这个类：携带模型服务失败详情的通用异常。
# 为什么需要：连接 SDK 错误与项目统一错误记录。
class ProviderError(RuntimeError):
    # 做什么：将模型服务失败详情保存在统一异常中。
    # 为什么需要：RAG 和 Judge 可以用相同方式读取失败类别与轨迹。
    def __init__(self, details: ProviderFailure):
        self.details = details
        super().__init__(details.message)


# 这个类：约定所有模型提供方都要实现的生成接口。
# 为什么需要：真实服务与测试替身可以互换。
class LLMProvider(ABC):
    # 做什么：规定模型调用需要接收的消息和参数以及返回结果。
    # 为什么需要：不同供应商实现和测试替身能够被上层同样调用。
    @abstractmethod
    def generate(self, messages: list[dict[str, str]], *, temperature: float | None = None,
                 response_format: dict | None = None, max_attempts: int | None = None) -> LLMResponse:
        """Return text with execution trace, or raise a classified ProviderError."""
