# 文件作用：测试 judge_v2 的显式拒答字段和 Prompt 版本限制。
# 为什么有它：让 Runner 使用的拒答统计有真实字段来源，不凭其他分数猜测。
from tests.provider_fakes import SDKFixtureProvider
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from config.judge import JudgeSettings
from src.judge import JudgeInput, LLMJudgeClient
from src.judge.models import JudgeVerdictV2
from src.judge.prompts import build_messages


# 做什么：验证 v2 必填拒答字段且客户端保留其值。
# 为什么需要：拒答统计需要明确观察而不是靠分数推断。
def test_v2_requires_explicit_refusal_observation_and_client_preserves_it():
    values = dict(answer_correctness=5, faithfulness=5, answer_relevance=5, completeness=5,
                  hallucination=False, unsupported_claims=[], reason="明确拒答")
    with pytest.raises(ValidationError, match="refusal_detected"):
        JudgeVerdictV2(**values)
    values["refusal_detected"] = True
    sdk = Mock()
    sdk.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(values)))])
    sample = JudgeInput(case_id="u", query="价格？", expected_answer="无法回答", retrieved_context="无价格信息", rag_answer="无法回答", answerable=False)
    settings = JudgeSettings(api_key="fake", base_url="https://example.invalid/v1", model="fake")
    client = LLMJudgeClient(settings, SDKFixtureProvider(settings, sdk), prompt_version="judge_v2")
    result = client.judge(sample)
    assert result.refusal_detected is True
    assert result.metadata.prompt_version == "judge_v2"
    assert "refusal_detected" in build_messages(sample, "judge_v2")[0]["content"]


# 做什么：验证不认识的 Prompt 版本立即报错。
# 为什么需要：避免按错误规则解释裁判输出。
def test_unknown_prompt_version_is_rejected():
    with pytest.raises(ValueError, match="Unknown Judge prompt"):
        LLMJudgeClient(JudgeSettings(api_key="fake", base_url="https://example.invalid/v1", model="fake"), Mock(), prompt_version="unknown")
