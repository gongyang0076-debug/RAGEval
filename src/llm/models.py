"""Serializable request traces; never store credentials or provider error bodies."""

from pydantic import BaseModel, ConfigDict, Field


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


class LLMResponse(BaseModel):
    text: str | None
    trace: ProviderTrace


class ProviderFailure(BaseModel):
    category: str
    message: str
    trace: ProviderTrace
