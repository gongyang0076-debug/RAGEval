# 文件作用：通过兼容 SDK 调用模型，并统一处理超时、错误分类和有限重试。
# 为什么有它：把外部服务的不稳定性集中处理，避免每个业务模块重复实现。
"""Bounded transport retries and cancellable per-request deadlines."""

import asyncio
import logging
import math
import re
from asyncio import sleep
from email.utils import parsedate_to_datetime
from time import perf_counter, time
from urllib.parse import urlsplit

from openai import AsyncOpenAI, APIError, APIConnectionError, APITimeoutError, APIStatusError

from config.settings import ConfigurationError
from .models import LLMResponse, ProviderAttempt, ProviderFailure, ProviderTrace
from .provider import LLMProvider, ProviderError

logger = logging.getLogger(__name__)


# 做什么：解析 Retry-After 或采用有上限的退避等待。
# 为什么需要：尊重服务端等待要求，超出本轮等待预算时停止。
def retry_delay(header: str | None, fallback: float) -> float | None:
    """None means Retry-After exceeds the 30s wait budget; never retry early."""
    if header:
        try:
            delay = float(header)
        except ValueError:
            try:
                delay = parsedate_to_datetime(header).timestamp() - time()
            except (ValueError, TypeError, OverflowError):
                delay = fallback
        if math.isfinite(delay) and delay >= 0:
            return delay if delay <= 30 else None
    return min(fallback, 30.0)


# 做什么：根据异常、状态码和供应商错误码判断类别及是否可重试。
# 为什么需要：认证或额度问题不应被当成临时网络故障反复请求。
def classify(error: Exception, secret: str) -> tuple[str, int | None, str | None, bool]:
    status = getattr(error, "status_code", None)
    body = getattr(error, "body", None)
    body = body.get("error", body) if isinstance(body, dict) else {}
    code = str(body.get("code") or "") if isinstance(body, dict) else ""
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", code) or (secret and secret in code):
        code = None
    if isinstance(error, (TimeoutError, asyncio.TimeoutError, APITimeoutError)) or status in {408, 504}:
        return "timeout", status, code, True
    if isinstance(error, APIConnectionError):
        return "connection", status, code, True
    if status == 401:
        category = "authentication"
    elif status == 403:
        category = "permission"
    elif status == 404 or code in {"model_not_found", "1211"}:
        category = "model_not_found"
    elif code in {"insufficient_quota", "1113", "1308", "1309", "1310"}:
        category = "quota"
    elif status == 429:
        return "rate_limit", status, code, True
    elif status is not None and 500 <= status < 600:
        return "server", status, code, True
    elif status is not None and 400 <= status < 500:
        category = "invalid_request"
    else:
        category = "api_response"
    return category, status, code, False


# 这个类：使用兼容 SDK 的具体模型访问实现。
# 为什么需要：统一配置、超时和重试，隔离供应商调用细节。
class OpenAICompatibleProvider(LLMProvider):
    # 做什么：检查并保存服务地址、模型、超时、尝试预算和可选参数。
    # 为什么需要：阻止非法配置进入 SDK，也避免地址携带敏感凭据。
    def __init__(self, *, api_key: str, base_url: str, model: str, timeout_seconds: float = 60,
                 max_attempts: int = 3, retry_delay_seconds: float = 5, thinking: str | None = None):
        if not all(value.strip() for value in (api_key, base_url, model)):
            raise ConfigurationError("Provider requires api_key, base_url and model")
        endpoint = urlsplit(base_url)
        if endpoint.scheme not in {"http", "https"} or not endpoint.netloc or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
            raise ConfigurationError("Provider base_url must be HTTP(S) without embedded credentials, query or fragment")
        if isinstance(timeout_seconds, bool) or not 0 < timeout_seconds <= 120:
            raise ConfigurationError("Provider timeout_seconds must be in (0, 120]")
        if type(max_attempts) is not int or not 1 <= max_attempts <= 5:
            raise ConfigurationError("Provider max_attempts must be an integer in [1, 5]")
        if isinstance(retry_delay_seconds, bool) or not 0 < retry_delay_seconds <= 30:
            raise ConfigurationError("Provider retry_delay_seconds must be in (0, 30]")
        if thinking not in {None, "enabled", "disabled"}:
            raise ConfigurationError("Provider thinking must be enabled, disabled or omitted")
        self._api_key = api_key
        self.base_url, self.model = base_url, model
        self.timeout_seconds, self.max_attempts = timeout_seconds, max_attempts
        self.retry_delay_seconds, self.thinking = retry_delay_seconds, thinking

    # 做什么：用 RAG 配置创建统一模型 Provider。
    # 为什么需要：把生成服务配置转换集中在一处。
    @classmethod
    def from_rag_settings(cls, settings):
        settings.require_llm()
        return cls(api_key=settings.rag_api_key, base_url=settings.rag_base_url, model=settings.rag_model,
                   timeout_seconds=settings.generation_timeout, max_attempts=settings.rag_max_attempts,
                   retry_delay_seconds=settings.rag_retry_delay_seconds, thinking=settings.rag_thinking)

    # 做什么：以同步接口启动内部异步请求，并限制调用预算。
    # 为什么需要：让顺序 Runner 易于调用，同时不能扩大配置允许的尝试次数。
    def generate(self, messages, *, temperature=None, response_format=None, max_attempts=None):
        limit = self.max_attempts if max_attempts is None else max_attempts
        if type(limit) is not int or not 1 <= limit <= self.max_attempts:
            raise ValueError("max_attempts override must be positive and within the configured budget")
        # A fresh async client per logical call avoids sharing a client across closed loops.
        return asyncio.run(self._generate(messages, temperature, response_format, limit))

    # 做什么：执行 SDK 请求、单次超时取消、分类重试和退避。
    # 为什么需要：模型服务不稳定时仍能受控结束并留下每次尝试。
    async def _generate(self, messages, temperature, response_format, limit):
        start = perf_counter()
        history = []
        options = {}
        if temperature is not None:
            options["temperature"] = temperature
        if response_format is not None:
            options["response_format"] = response_format
        if self.thinking is not None:
            options["extra_body"] = {"thinking": {"type": self.thinking}}

        # 做什么：汇总模型参数、总耗时和全部请求记录。
        # 为什么需要：成功响应和失败异常都携带一致的追溯信息。
        def trace():
            return ProviderTrace(model=self.model, base_url=self.base_url, timeout_seconds=self.timeout_seconds,
                                 max_attempts=limit, thinking=self.thinking, retry_count=len(history) - 1,
                                 latency_ms=(perf_counter() - start) * 1000, attempts=history)

        async with AsyncOpenAI(api_key=self._api_key, base_url=self.base_url,
                               timeout=self.timeout_seconds, max_retries=0) as client:
            for attempt in range(1, limit + 1):
                request_start = perf_counter()
                try:
                    response = await asyncio.wait_for(
                        client.chat.completions.create(model=self.model, messages=messages, **options),
                        timeout=self.timeout_seconds,
                    )
                except (APIError, asyncio.TimeoutError) as exc:
                    category, status, code, retryable = classify(exc, self._api_key)
                    row = ProviderAttempt(attempt=attempt, latency_ms=(perf_counter() - request_start) * 1000,
                                          error_type=type(exc).__name__, category=category, status_code=status,
                                          provider_code=code, retryable=retryable)
                    history.append(row)
                    header = exc.response.headers.get("retry-after") if isinstance(exc, APIStatusError) else None
                    delay = retry_delay(header, min(self.retry_delay_seconds * 2 ** (attempt - 1), 30))
                    if not retryable or attempt == limit or delay is None:
                        message = f"{type(exc).__name__}: {category}" + (f" (HTTP {status})" if status else "")
                        raise ProviderError(ProviderFailure(category=category, message=message, trace=trace())) from exc
                    row.retry_delay_seconds = delay
                    logger.warning("LLM %s%s; retry %s/%s in %.1fs", category,
                                   f" code={code}" if code else "", attempt + 1, limit, delay)
                    await sleep(delay)
                else:
                    history.append(ProviderAttempt(attempt=attempt, latency_ms=(perf_counter() - request_start) * 1000))
                    return LLMResponse(text=response.choices[0].message.content if response.choices else None, trace=trace())
        raise AssertionError("Unreachable attempt limit")
