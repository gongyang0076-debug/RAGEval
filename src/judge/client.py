# 文件作用：调用模型判卷，严格校验 JSON 和评分结构，并在有限预算内修复输出。
# 为什么有它：把不合规输出和 API 故障明确记录，不能把坏数据当成有效评分。
"""Judge validation and transport repair share one total request budget."""

import json
from time import perf_counter
from pydantic import ValidationError

from config.judge import JudgeSettings, load_judge_settings
from src.llm import LLMProvider, ProviderError
from src.llm.openai_compatible import OpenAICompatibleProvider
from .models import JudgeAttempt, JudgeFailure, JudgeInput, JudgeMetadata, JudgeResult
from .prompts import PROMPT_VERSION, build_messages, verdict_model


# 这个类：裁判相关错误的共同类型并携带结构化详情。
# 为什么需要：Runner 可以统一捕获和记录。
class JudgeError(RuntimeError):
    # 做什么：把结构化裁判失败详情附加到异常上。
    # 为什么需要：上层既能捕获失败，也能保存完整诊断信息。
    def __init__(self, details: JudgeFailure):
        self.details = details
        super().__init__(f"Judge {details.category}: {details.message}")


# 这个类：表示裁判在尝试预算内仍未给出合规结果。
# 为什么需要：区分输出问题和外部接口故障。
class JudgeOutputError(JudgeError):
    """No valid verdict within the total request budget."""


# 这个类：表示模型服务调用终止失败。
# 为什么需要：保留认证、超时或服务端故障等类别。
class JudgeAPIError(JudgeError):
    """A terminal provider failure, including exhausted transport retries."""


# 做什么：在解析裁判 JSON 时拒绝同名字段。
# 为什么需要：避免同一分数出现两个值却被静默覆盖。
def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


# 做什么：拒绝 JSON 中的 NaN 或无穷等非标准常量。
# 为什么需要：防止非法数值绕过正常评分格式。
def _reject_constant(value):
    raise ValueError(f"Non-standard JSON constant: {value}")


# 这个类：按固定规则请求模型评分并校验结果。
# 为什么需要：把判卷请求、修复重试和证据保存集中管理。
class LLMJudgeClient:
    # 做什么：校验裁判配置并选择 Prompt、评分模型和 Provider。
    # 为什么需要：在请求前确定判卷规则，也方便测试注入替身。
    def __init__(self, settings: JudgeSettings | None = None, provider: LLMProvider | None = None,
                 prompt_version: str = PROMPT_VERSION):
        self.settings = settings if settings is not None else load_judge_settings()
        self.settings.validate()
        self.prompt_version = prompt_version
        self.verdict_model = verdict_model(prompt_version)
        self.provider = provider if provider is not None else OpenAICompatibleProvider(
            api_key=self.settings.api_key, base_url=self.settings.base_url, model=self.settings.model,
            timeout_seconds=self.settings.judge_timeout, max_attempts=self.settings.max_attempts,
            retry_delay_seconds=self.settings.retry_delay_seconds,
        )

    # 做什么：发送判卷请求，严格解析输出，并在共享预算内有限重试。
    # 为什么需要：得到可保存的合法评分或明确失败，而不是无限等待。
    def judge(self, judge_input: JudgeInput) -> JudgeResult:
        start = perf_counter()
        history: list[JudgeAttempt] = []
        traces = []
        messages = build_messages(judge_input, self.prompt_version)
        response_format = None
        if self.settings.response_format == "json_object":
            response_format = {"type": "json_object"}
        elif self.settings.response_format == "json_schema":
            response_format = {"type": "json_schema", "json_schema": {
                "name": self.prompt_version, "strict": True, "schema": self.verdict_model.model_json_schema(),
            }}

        # 做什么：整理当前耗时、模型、原始输出和每次尝试记录。
        # 为什么需要：成功或失败都能追溯这次判卷的实际过程。
        def metadata():
            return JudgeMetadata(case_id=judge_input.case_id, judge_model=self.settings.model,
                                 prompt_version=self.prompt_version, response_format=self.settings.response_format,
                                 latency_ms=(perf_counter() - start) * 1000, attempts=len(history),
                                 retry_count=len(history) - 1, raw_output=history[-1].raw_output,
                                 attempt_history=history, provider_traces=traces)

        # 做什么：把 Provider 的失败尝试追加到裁判历史。
        # 为什么需要：让网络重试与 JSON 修复使用同一总预算。
        def record_errors(attempts):
            for item in attempts:
                history.append(JudgeAttempt(attempt=len(history) + 1, raw_output=None,
                                            error_category=item.category, error_message=item.error_type,
                                            status_code=item.status_code))

        while len(history) < self.settings.max_attempts:
            try:
                response = self.provider.generate(list(messages), temperature=0, response_format=response_format,
                                                  max_attempts=self.settings.max_attempts - len(history))
            except ProviderError as exc:
                traces.append(exc.details.trace)
                record_errors(exc.details.trace.attempts)
                raise JudgeAPIError(JudgeFailure(category=exc.details.category, message=exc.details.message,
                                                 metadata=metadata())) from exc
            traces.append(response.trace)
            record_errors(response.trace.attempts[:-1])
            raw = response.text
            attempt = len(history) + 1
            try:
                data = json.loads(raw or "", object_pairs_hook=_reject_duplicate_keys, parse_constant=_reject_constant)
            except ValueError as exc:
                category, message = "invalid_json", str(exc)
            else:
                try:
                    verdict = self.verdict_model.model_validate(data)
                except ValidationError as exc:
                    category = "invalid_schema"
                    message = json.dumps(exc.errors(include_input=False, include_context=False, include_url=False), ensure_ascii=False)
                else:
                    history.append(JudgeAttempt(attempt=attempt, raw_output=raw))
                    return JudgeResult(**verdict.model_dump(), metadata=metadata())
            history.append(JudgeAttempt(attempt=attempt, raw_output=raw, error_category=category, error_message=message))
            if len(history) == self.settings.max_attempts:
                raise JudgeOutputError(JudgeFailure(category=category,
                    message=f"No valid Judge JSON after {attempt} attempt(s): {message}", metadata=metadata()))
            messages.extend([
                {"role": "assistant", "content": raw or ""},
                {"role": "user", "content": f"上次输出未通过校验（{category}）：{message}\n请重新输出完整且符合原 JSON Schema 的评测对象，不要解释。"},
            ])
        raise AssertionError("Unreachable attempt limit")
