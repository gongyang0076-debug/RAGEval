# 文件作用：标记“模型服务访问”目录为 Python 包，供其他模块导入。
# 为什么有它：让相关文件能够通过统一的包路径组织和引用。
"""SDK-independent language model contracts and provider implementations."""

from .models import LLMResponse, ProviderAttempt, ProviderFailure, ProviderTrace
from .provider import LLMProvider, ProviderError

__all__ = ["LLMProvider", "ProviderError", "LLMResponse", "ProviderAttempt", "ProviderFailure", "ProviderTrace"]
