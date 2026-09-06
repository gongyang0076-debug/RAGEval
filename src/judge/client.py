"""Judge validation and transport repair share one total request budget."""

import json
from time import perf_counter
from pydantic import ValidationError

from config.judge import JudgeSettings, load_judge_settings
from src.llm import LLMProvider, ProviderError
from src.llm.openai_compatible import OpenAICompatibleProvider
from .models import JudgeAttempt, JudgeFailure, JudgeInput, JudgeMetadata, JudgeResult
from .prompts import PROMPT_VERSION, build_messages, verdict_model


class JudgeError(RuntimeError):
    def __init__(self, details: JudgeFailure):
        self.details = details
        super().__init__(f"Judge {details.category}: {details.message}")


class JudgeOutputError(JudgeError):
    """No valid verdict within the total request budget."""


class JudgeAPIError(JudgeError):
    """A terminal provider failure, including exhausted transport retries."""


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"Non-standard JSON constant: {value}")


class LLMJudgeClient:
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

        def metadata():
            return JudgeMetadata(case_id=judge_input.case_id, judge_model=self.settings.model,
                                 prompt_version=self.prompt_version, response_format=self.settings.response_format,
                                 latency_ms=(perf_counter() - start) * 1000, attempts=len(history),
                                 retry_count=len(history) - 1, raw_output=history[-1].raw_output,
                                 attempt_history=history, provider_traces=traces)

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
