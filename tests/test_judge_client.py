import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
import pytest
from openai import (
    APIConnectionError, APIResponseValidationError, APIStatusError, APITimeoutError,
    AuthenticationError, BadRequestError, PermissionDeniedError, RateLimitError,
)

from config.judge import JudgeSettings
from config.settings import ConfigurationError
from src.judge import JudgeAPIError, JudgeInput, JudgeOutputError, JudgeResult, LLMJudgeClient
from src.judge.prompts import PROMPT_VERSION
from tests.provider_fakes import SDKFixtureProvider


def make_judge(settings, sdk=None, **kwargs):
    provider = SDKFixtureProvider(settings, sdk) if sdk is not None else None
    return LLMJudgeClient(settings, provider, **kwargs)


@pytest.fixture(autouse=True)
def forbid_real_http(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Judge unit tests must not make real HTTP requests")
    monkeypatch.setattr(httpx.Client, "send", forbidden)


@pytest.fixture
def settings():
    return JudgeSettings(api_key="test-only-secret", base_url="https://example.invalid/v1", model="mock-judge")


@pytest.fixture
def judge_input():
    return JudgeInput(case_id="case-1", query="多久关闭？", expected_answer="30分钟。", retrieved_context="30分钟关闭。", rag_answer="30分钟。", answerable=True)


def output(**changes):
    return json.dumps({
        "answer_correctness": 5, "faithfulness": 5, "answer_relevance": 5,
        "completeness": 5, "hallucination": False, "unsupported_claims": [],
        "reason": "正确且有依据。", **changes,
    }, ensure_ascii=False)


def completion(raw):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=raw))])


def test_normal_judge_uses_sdk_settings_prompt_schema_and_preserves_metadata(settings, judge_input):
    raw = output()
    sdk = Mock()
    sdk.chat.completions.create.return_value = completion(raw)
    with patch("src.judge.client.perf_counter", side_effect=[10, 10.25]):
        result = make_judge(settings, sdk).judge(judge_input)
    kwargs = sdk.chat.completions.create.call_args.kwargs
    assert result.metadata.provider_traces[0].timeout_seconds == 60
    assert isinstance(result, JudgeResult)
    assert result.answer_correctness == 5
    assert kwargs["temperature"] == 0
    assert kwargs["model"] == "mock-judge"
    assert "response_format" not in kwargs  # Generic compatible mode relies on local validation.
    system, user = kwargs["messages"]
    assert system["role"] == "system"
    assert "不回答用户问题" in system["content"]
    assert "expected_answer 是正确性参照" in system["content"]
    assert "不得把 expected_answer" in system["content"]
    assert "JSON Schema" in system["content"]
    assert json.loads(user["content"]) == judge_input.model_dump()
    metadata = result.metadata
    assert metadata.case_id == "case-1"
    assert metadata.judge_model == "mock-judge"
    assert metadata.prompt_version == PROMPT_VERSION == "judge_v1"
    assert metadata.latency_ms == pytest.approx(250)
    assert metadata.raw_output == raw
    assert metadata.attempts == 1
    assert metadata.attempt_history[0].error_category is None
    assert JudgeResult.model_validate_json(result.model_dump_json()) == result


def test_structured_output_mode_uses_only_verdict_schema(settings, judge_input):
    sdk = Mock()
    sdk.chat.completions.create.return_value = completion(output())
    result = make_judge(replace(settings, response_format="json_schema"), sdk).judge(judge_input)
    response_format = sdk.chat.completions.create.call_args.kwargs["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    schema = response_format["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
    assert "metadata" not in schema["properties"]
    assert result.metadata.response_format == "json_schema"


def test_context_supported_but_incorrect_answer(settings, judge_input):
    sdk = Mock()
    sdk.chat.completions.create.return_value = completion(output(answer_correctness=0, faithfulness=5))
    judge_input.retrieved_context = "60分钟关闭。"
    judge_input.rag_answer = "60分钟关闭。"
    result = make_judge(settings, sdk).judge(judge_input)
    assert result.answer_correctness == 0
    assert result.faithfulness == 5
    assert result.hallucination is False


@pytest.mark.parametrize("fabricated", [False, True])
def test_unanswerable_still_gets_judged(settings, fabricated):
    sdk = Mock()
    score = 0 if fabricated else 5
    sdk.chat.completions.create.return_value = completion(output(
        answer_correctness=score, faithfulness=score, hallucination=fabricated,
        unsupported_claims=["会员售价9.9元"] if fabricated else [],
    ))
    value = JudgeInput(case_id="unknown", query="会员售价？", expected_answer="无法确定。", retrieved_context="未提供售价。",
                       rag_answer="会员售价9.9元。" if fabricated else "无法确定。", answerable=False)
    result = make_judge(settings, sdk).judge(value)
    sdk.chat.completions.create.assert_called_once()
    assert result.hallucination is fabricated
    assert result.answer_correctness == score
    assert result.unsupported_claims == (["会员售价9.9元"] if fabricated else [])


@pytest.mark.parametrize("raw,category", [
    ("not JSON", "invalid_json"), ("```json\n{}\n```", "invalid_json"),
    ('{"reason":"a","reason":"b"}', "invalid_json"), ('{"score":NaN}', "invalid_json"),
    (None, "invalid_json"), ("{}", "invalid_schema"), ("[]", "invalid_schema"),
    (output(answer_correctness=6), "invalid_schema"),
    (output(hallucination=True, unsupported_claims=[]), "invalid_schema"),
])
def test_invalid_outputs_exhaust_finite_attempts_with_trace(settings, judge_input, raw, category):
    sdk = Mock()
    sdk.chat.completions.create.return_value = completion(raw)
    with pytest.raises(JudgeOutputError) as error:
        make_judge(replace(settings, max_attempts=2), sdk).judge(judge_input)
    assert sdk.chat.completions.create.call_count == 2
    failure = error.value.details
    assert failure.category == category
    assert failure.metadata.attempts == 2
    assert failure.metadata.raw_output == raw
    assert failure.metadata.prompt_version == "judge_v1"
    assert failure.metadata.latency_ms >= 0
    assert [entry.error_category for entry in failure.metadata.attempt_history] == [category, category]


def test_parse_failure_then_success_preserves_both_attempts(settings, judge_input):
    sdk = Mock()
    valid = output()
    sdk.chat.completions.create.side_effect = [completion("broken"), completion(valid)]
    result = make_judge(settings, sdk).judge(judge_input)
    assert result.metadata.attempts == 2
    assert [entry.raw_output for entry in result.metadata.attempt_history] == ["broken", valid]
    calls = sdk.chat.completions.create.call_args_list
    assert len(calls[0].kwargs["messages"]) == 2
    assert len(calls[1].kwargs["messages"]) == 4
    assert calls[0].kwargs["messages"][:2] == calls[1].kwargs["messages"][:2]
    assert "invalid_json" in calls[1].kwargs["messages"][-1]["content"]


def test_missing_schema_field_then_retry_success(settings, judge_input):
    sdk = Mock()
    values = json.loads(output())
    del values["completeness"]
    sdk.chat.completions.create.side_effect = [completion(json.dumps(values)), completion(output())]
    result = make_judge(settings, sdk).judge(judge_input)
    assert result.metadata.attempt_history[0].error_category == "invalid_schema"
    assert "completeness" in result.metadata.attempt_history[0].error_message


@pytest.mark.parametrize("error_type,category,status", [
    (APITimeoutError, "timeout", None), (APIConnectionError, "connection", None),
    (AuthenticationError, "authentication", 401), (PermissionDeniedError, "permission", 403),
    (RateLimitError, "rate_limit", 429), (BadRequestError, "invalid_request", 400),
    (APIStatusError, "server", 503), (APIResponseValidationError, "api_response", 200),
])
def test_api_errors_are_classified_and_retries_are_bounded(settings, judge_input, error_type, category, status):
    request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    if error_type is APITimeoutError:
        api_error = error_type(request=request)
    elif error_type is APIConnectionError:
        api_error = error_type(request=request)
    elif error_type is APIResponseValidationError:
        api_error = error_type(response=httpx.Response(status, request=request), body={})
    else:
        api_error = error_type("test-only-secret must not be logged", response=httpx.Response(status, request=request), body={})
    sdk = Mock()
    sdk.chat.completions.create.side_effect = api_error
    with pytest.raises(JudgeAPIError) as error:
        make_judge(settings, sdk).judge(judge_input)
    expected_attempts = 3 if category in {"timeout", "connection", "rate_limit", "server"} else 1
    assert sdk.chat.completions.create.call_count == expected_attempts
    details = error.value.details
    assert details.category == category
    assert details.metadata.attempts == expected_attempts
    assert details.metadata.retry_count == expected_attempts - 1
    assert details.metadata.attempt_history[0].status_code == status
    assert "test-only-secret" not in details.model_dump_json()


def test_api_failure_after_parse_failure_keeps_prior_raw_output(settings, judge_input):
    sdk = Mock()
    request = httpx.Request("POST", "https://example.invalid")
    sdk.chat.completions.create.side_effect = [completion("broken"), APITimeoutError(request=request), APITimeoutError(request=request)]
    with pytest.raises(JudgeAPIError) as error:
        make_judge(settings, sdk).judge(judge_input)
    assert error.value.details.metadata.attempts == 3
    assert error.value.details.metadata.attempt_history[0].raw_output == "broken"
    assert error.value.details.category == "timeout"


@pytest.mark.parametrize("field,env_name", [("api_key", "JUDGE_API_KEY"), ("base_url", "JUDGE_BASE_URL"), ("model", "JUDGE_MODEL")])
def test_missing_configuration_is_clear(settings, field, env_name):
    with patch("src.judge.client.OpenAICompatibleProvider") as factory:
        with pytest.raises(ConfigurationError, match=env_name):
            make_judge(replace(settings, **{field: "  "}))
        factory.assert_not_called()


def test_api_retries_and_json_repairs_share_one_budget(settings, judge_input):
    sdk = Mock()
    response = httpx.Response(503, request=httpx.Request("POST", "https://example.invalid"))
    sdk.chat.completions.create.side_effect = [APIStatusError("busy", response=response, body={}), completion("broken"), completion(output())]
    result = make_judge(settings, sdk).judge(judge_input)
    assert result.metadata.attempts == 3 and result.metadata.retry_count == 2
    assert [a.error_category for a in result.metadata.attempt_history] == ["server", "invalid_json", None]
    assert [len(t.attempts) for t in result.metadata.provider_traces] == [2, 1]
    assert sdk.chat.completions.create.call_count == 3


def test_json_failure_then_api_failure_never_multiplies_budget(settings, judge_input):
    sdk = Mock()
    response = httpx.Response(503, request=httpx.Request("POST", "https://example.invalid"))
    error = APIStatusError("busy", response=response, body={})
    sdk.chat.completions.create.side_effect = [completion("broken"), error, error]
    with pytest.raises(JudgeAPIError) as caught:
        make_judge(settings, sdk).judge(judge_input)
    assert sdk.chat.completions.create.call_count == 3
    assert caught.value.details.metadata.retry_count == 2
    assert caught.value.details.metadata.attempt_history[0].raw_output == "broken"


def test_judge_request_deadline_cancels_and_preserves_timeout(settings, judge_input):
    import asyncio
    from time import perf_counter
    from src.llm.openai_compatible import OpenAICompatibleProvider
    from tests.provider_fakes import async_sdk
    sdk = async_sdk()
    cancelled = []
    async def hanging(**kwargs):
        try:
            await asyncio.sleep(30)
        finally:
            cancelled.append(True)
    sdk.chat.completions.create.side_effect = hanging
    settings = replace(settings, timeout_seconds=0.02, max_attempts=1)
    provider = OpenAICompatibleProvider(api_key=settings.api_key, base_url=settings.base_url, model=settings.model,
                                        timeout_seconds=settings.judge_timeout, max_attempts=1)
    start = perf_counter()
    with patch("src.llm.openai_compatible.AsyncOpenAI", return_value=sdk), pytest.raises(JudgeAPIError) as caught:
        LLMJudgeClient(settings, provider).judge(judge_input)
    assert perf_counter() - start < 1 and cancelled == [True]
    assert caught.value.details.category == "timeout"
    assert caught.value.details.metadata.provider_traces[0].timeout_seconds == 0.02
    assert caught.value.details.metadata.retry_count == 0


@pytest.mark.parametrize("prompt_version", ["judge_v1", "judge_v2"])
def test_json_object_mode_reaches_sdk_and_keeps_metadata(settings, judge_input, prompt_version):
    raw = output(**({"refusal_detected": False} if prompt_version == "judge_v2" else {}))
    sdk = Mock()
    sdk.chat.completions.create.return_value = completion(raw)
    result = make_judge(replace(settings, response_format="json_object"), sdk,
                        prompt_version=prompt_version).judge(judge_input)
    kwargs = sdk.chat.completions.create.call_args.kwargs
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["temperature"] == 0
    assert "JSON Schema" in kwargs["messages"][0]["content"]
    assert result.metadata.response_format == "json_object"
    assert result.metadata.prompt_version == prompt_version
    assert result.metadata.raw_output == raw


@pytest.mark.parametrize("raw,category", [
    ("\x60\x60\x60json\n{}\n\x60\x60\x60", "invalid_json"),
    ("{}", "invalid_schema"),
    (output(answer_correctness=6), "invalid_schema"),
])
def test_json_object_mode_still_validates_and_bounds_repair(settings, judge_input, raw, category):
    sdk = Mock()
    sdk.chat.completions.create.return_value = completion(raw)
    with pytest.raises(JudgeOutputError) as caught:
        make_judge(replace(settings, response_format="json_object", max_attempts=2), sdk).judge(judge_input)
    assert caught.value.details.category == category
    assert caught.value.details.metadata.response_format == "json_object"
    assert caught.value.details.metadata.raw_output == raw
    assert caught.value.details.metadata.retry_count == 1
    assert sdk.chat.completions.create.call_count == 2
    assert all(call.kwargs["response_format"] == {"type": "json_object"}
               for call in sdk.chat.completions.create.call_args_list)


def test_json_object_mode_repair_success_preserves_invalid_attempt(settings, judge_input):
    sdk = Mock()
    sdk.chat.completions.create.side_effect = [completion("{}"), completion(output())]
    result = make_judge(replace(settings, response_format="json_object"), sdk).judge(judge_input)
    assert result.metadata.attempts == 2
    assert result.metadata.attempt_history[0].error_category == "invalid_schema"
    assert result.metadata.attempt_history[0].raw_output == "{}"
    assert all(call.kwargs["response_format"] == {"type": "json_object"}
               for call in sdk.chat.completions.create.call_args_list)
