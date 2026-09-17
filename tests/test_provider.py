# 文件作用：测试模型接口参数、重试、Retry-After、超时取消和永久错误。
# 为什么有它：让网络故障能受控结束，并保留真实尝试次数。
import asyncio
from time import perf_counter
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from openai import APIConnectionError, APITimeoutError, APIStatusError, AuthenticationError, BadRequestError, RateLimitError

from config.settings import ConfigurationError, Settings
from pathlib import Path
from src.llm import LLMProvider, ProviderError
from src.llm.openai_compatible import OpenAICompatibleProvider
from tests.provider_fakes import async_sdk, completion


# 做什么：按状态码、供应商代码和 Retry-After 构造 SDK 异常。
# 为什么需要：不用制造真实故障也能测试各种服务响应。
def api_error(kind, status=None, code=None, retry_after=None):
    request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    if kind in {APIConnectionError, APITimeoutError}:
        return kind(request=request)
    return kind("secret-do-not-log", response=httpx.Response(status, request=request,
                headers={} if retry_after is None else {"Retry-After": retry_after}), body={"code": code})


# 做什么：同时阻止同步与异步 HTTP 发送。
# 为什么需要：Provider 测试必须完全不联网。
@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    # 做什么：发现实际网络请求就抛断言错误。
    # 为什么需要：保证异常模拟不会误触真实服务。
    def forbidden(*args, **kwargs):
        raise AssertionError("No real network in provider unit tests")
    monkeypatch.setattr(httpx.Client, "send", forbidden)
    monkeypatch.setattr(httpx.AsyncClient, "send", forbidden)


# 做什么：提供实际 Provider、假 SDK 和假的退避等待。
# 为什么需要：验证请求与重试逻辑而不产生网络或等待成本。
@pytest.fixture
def setup():
    sdk = async_sdk()
    sdk.chat.completions.create.return_value = completion("Answer")
    with patch("src.llm.openai_compatible.AsyncOpenAI", return_value=sdk) as factory, patch("src.llm.openai_compatible.sleep", new_callable=AsyncMock) as sleep:
        provider = OpenAICompatibleProvider(api_key="secret-do-not-log", base_url="https://example.invalid/v1", model="fake")
        yield provider, sdk, factory, sleep


# 做什么：验证统一 Provider 不能直接当具体实现实例化。
# 为什么需要：调用者必须使用真正实现了生成方法的适配器。
def test_provider_contract_is_abstract():
    with pytest.raises(TypeError):
        LLMProvider()


# 做什么：验证服务地址、模型、超时和可选参数正确传递并保留轨迹。
# 为什么需要：配置不能在接口边界丢失。
def test_endpoint_model_timeout_options_and_trace(setup):
    provider, sdk, factory, _ = setup
    provider.timeout_seconds = 0.5
    provider.thinking = "disabled"
    result = provider.generate([{"role": "user", "content": "Question"}], temperature=0)
    factory.assert_called_once_with(api_key="secret-do-not-log", base_url="https://example.invalid/v1", timeout=0.5, max_retries=0)
    kwargs = sdk.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "fake" and kwargs["temperature"] == 0
    assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
    assert result.text == "Answer" and result.trace.retry_count == 0
    assert result.trace.timeout_seconds == 0.5 and result.trace.latency_ms >= 0
    assert len(result.trace.attempts) == 1
    assert "secret-do-not-log" not in result.model_dump_json()


# 做什么：验证临时错误后可以恢复且重试计数正确。
# 为什么需要：有限重试需要既能恢复也能留痕。
@pytest.mark.parametrize("kind,status,code,category", [
    (RateLimitError, 429, "1305", "rate_limit"), (APITimeoutError, None, None, "timeout"),
    (APIConnectionError, None, None, "connection"), (APIStatusError, 500, None, "server"),
    (APIStatusError, 503, None, "server"),
    (APIStatusError, 408, None, "timeout"), (APIStatusError, 504, None, "timeout"),
])
def test_transient_errors_recover_with_retry_count(setup, kind, status, code, category):
    provider, sdk, _, sleep = setup
    sdk.chat.completions.create.side_effect = [api_error(kind, status, code), completion("Recovered")]
    result = provider.generate([])
    assert result.text == "Recovered" and result.trace.retry_count == 1
    assert result.trace.attempts[0].category == category
    assert result.trace.attempts[0].provider_code == code
    assert sdk.chat.completions.create.call_count == 2
    sleep.assert_awaited_once_with(5)


# 做什么：验证认证、参数或额度等永久错误不反复请求。
# 为什么需要：避免无效调用、等待和额外费用。
@pytest.mark.parametrize("kind,status,code,category", [
    (AuthenticationError, 401, None, "authentication"), (BadRequestError, 400, None, "invalid_request"),
    (APIStatusError, 404, None, "model_not_found"), (APIStatusError, 400, "1211", "model_not_found"),
    (APIStatusError, 403, None, "permission"), (RateLimitError, 429, "1113", "quota"),
])
def test_permanent_errors_are_not_retried(setup, kind, status, code, category):
    provider, sdk, _, sleep = setup
    sdk.chat.completions.create.side_effect = api_error(kind, status, code)
    with pytest.raises(ProviderError) as caught:
        provider.generate([])
    assert caught.value.details.category == category
    assert caught.value.details.trace.retry_count == 0
    assert "secret-do-not-log" not in caught.value.details.model_dump_json()
    sdk.chat.completions.create.assert_called_once()
    sleep.assert_not_awaited()


# 做什么：验证预算耗尽后保存每次失败及指数等待。
# 为什么需要：失败也要知道总共尝试了多少次。
def test_exhaustion_preserves_all_attempts_and_exponential_backoff(setup):
    provider, sdk, _, sleep = setup
    sdk.chat.completions.create.side_effect = api_error(RateLimitError, 429, "1305")
    with pytest.raises(ProviderError) as caught:
        provider.generate([])
    assert caught.value.details.trace.retry_count == 2
    assert len(caught.value.details.trace.attempts) == 3
    assert [call.args[0] for call in sleep.call_args_list] == [5, 10]


# 做什么：验证秒数形式的服务端等待要求与预算边界。
# 为什么需要：不能忽略服务端要求提前重试。
@pytest.mark.parametrize("header,delay", [("12", 12), ("garbage", 5), ("NaN", 5), ("0", 0), ("90", None)])
def test_retry_after(setup, header, delay):
    provider, sdk, _, sleep = setup
    sdk.chat.completions.create.side_effect = [api_error(RateLimitError, 429, retry_after=header), completion("OK")]
    if delay is None:
        with pytest.raises(ProviderError) as caught:
            provider.generate([])
        assert caught.value.details.trace.retry_count == 0
        sleep.assert_not_awaited()
    else:
        provider.generate([])
        sleep.assert_awaited_once_with(delay)


# 做什么：验证时间日期形式的 Retry-After。
# 为什么需要：兼容不同服务端响应格式。
def test_http_date_retry_after(setup):
    provider, sdk, _, sleep = setup
    sdk.chat.completions.create.side_effect = [api_error(RateLimitError, 429, retry_after="Thu, 01 Jan 1970 00:00:12 GMT"), completion("OK")]
    with patch("src.llm.openai_compatible.time", return_value=0):
        provider.generate([])
    sleep.assert_awaited_once_with(12)


# 做什么：验证慢协程被请求期限取消且总次数受控。
# 为什么需要：防止请求一直挂起或无限超时重试。
def test_wall_deadline_cancels_slow_request_and_is_bounded(setup):
    provider, sdk, _, _ = setup
    provider.timeout_seconds = 0.02
    provider.max_attempts = 2
    cancelled = []
    # 做什么：模拟一直等待的异步请求并记录取消。
    # 为什么需要：证明超时机制能中止慢操作。
    async def hanging(**kwargs):
        try:
            await asyncio.sleep(30)
        finally:
            cancelled.append(True)
    sdk.chat.completions.create.side_effect = hanging
    start = perf_counter()
    with pytest.raises(ProviderError) as caught:
        provider.generate([])
    assert perf_counter() - start < 1
    assert len(cancelled) == 2
    sdk.__aexit__.assert_awaited_once()
    assert caught.value.details.category == "timeout"
    assert caught.value.details.trace.retry_count == 1


# 做什么：验证传入剩余预算只能缩小而不能扩大尝试上限。
# 为什么需要：嵌套调用不能绕过总重试限制。
def test_remaining_budget_cannot_expand_attempts(setup):
    provider, sdk, _, _ = setup
    sdk.chat.completions.create.side_effect = api_error(RateLimitError, 429)
    with pytest.raises(ProviderError) as caught:
        provider.generate([], max_attempts=1)
    assert caught.value.details.trace.retry_count == 0
    with pytest.raises(ValueError, match="budget"):
        provider.generate([], max_attempts=4)


# 做什么：验证缺少生成服务必填配置时明确报错。
# 为什么需要：不应带着空密钥或空地址进入请求。
@pytest.mark.parametrize("name", ["rag_api_key", "rag_base_url", "rag_model"])
def test_missing_rag_settings_are_clear(name):
    values = dict(data_dir=Path("data"), rag_api_key="key", rag_base_url="https://example.invalid/v1", rag_model="fake")
    values[name] = ""
    with pytest.raises(ConfigurationError, match=name.upper()):
        OpenAICompatibleProvider.from_rag_settings(Settings(**values))
