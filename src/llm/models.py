# 文件作用：规定模型调用结果、每次尝试、总轨迹和失败信息的数据结构。
# 为什么有它：让失败原因、耗时和重试次数能够被上层统一保存。
"""Serializable request traces; never store credentials or provider error bodies."""

from pydantic import BaseModel, ConfigDict, Field


# 这个类：记录一次模型请求的耗时、错误和等待信息。
# 为什么需要：精确核对实际尝试过程。
class ProviderAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid")
    attempt: int = Field(ge=1)
    latency_ms: float = Field(ge=0, allow_inf_nan=False)
    error_type: str | None = None
    category: str | None = None
    status_code: int | None = None
    provider_code: str | None = None
    retryable: bool = False
    retry_delay_seconds: float = Field(default=0, ge=0, allow_inf_nan=False)


# 这个类：汇总一次逻辑调用的配置和全部请求记录。
# 为什么需要：让上层知道用了哪个服务以及重试了多少次。
class ProviderTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    base_url: str
    timeout_seconds: float
    max_attempts: int
    thinking: str | None = None
    retry_count: int = Field(ge=0)
    latency_ms: float = Field(ge=0, allow_inf_nan=False)
    attempts: list[ProviderAttempt]


# 这个类：保存模型返回文本与调用轨迹。
# 为什么需要：业务代码不必直接读取 SDK 私有结构。
class LLMResponse(BaseModel):
    text: str | None
    trace: ProviderTrace


# 这个类：保存模型服务终止失败的类别与轨迹。
# 为什么需要：不同调用方可以统一处理错误。
class ProviderFailure(BaseModel):
    category: str
    message: str
    trace: ProviderTrace
