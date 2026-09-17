# 文件作用：提供明确标记为模拟的裁判，用占位分数和简单拒答探针验证流程。
# 为什么有它：没有真实 API 时仍可测试流水线，但这些结果不能冒充真实质量。
"""Explicit smoke-only test double; zero scores are placeholders, not judgments."""

import json
from time import perf_counter

from config.judge import JudgeSettings
from src.judge import JudgeInput, JudgeResult
from src.judge.models import JudgeAttempt, JudgeMetadata
from src.judge.prompts import RUNNER_PROMPT_VERSION


# 这个类：提供明确标记为模拟的裁判实现。
# 为什么需要：测试流水线时不必访问真实模型。
class MockJudge:
    prompt_version = RUNNER_PROMPT_VERSION

    # 做什么：设置模拟裁判的配置和服务身份。
    # 为什么需要：让流程测试的来源与真实模型调用明确区分。
    def __init__(self):
        self.settings = JudgeSettings(model="mock-judge-placeholder-v1", base_url="mock://judge", response_format="json")

    # 做什么：返回占位分数和简单拒答探针，并标明模拟来源。
    # 为什么需要：无真实模型时验证数据流，不能拿它证明答案质量。
    def judge(self, value: JudgeInput) -> JudgeResult:
        start = perf_counter()
        # This lexical observation only exercises the refusal plumbing; it is not a real Judge.
        refusal = any(marker in value.rag_answer for marker in ("无法确定", "无法回答", "信息不足", "未提供"))
        verdict = {
            "answer_correctness": 0, "faithfulness": 0, "answer_relevance": 0, "completeness": 0,
            "hallucination": False, "unsupported_claims": [], "refusal_detected": refusal,
            "reason": "JUDGE_MODE=MOCK：分数和幻觉标记仅为占位；拒答为关键词探针，不代表真实质量判定。",
        }
        raw = json.dumps(verdict, ensure_ascii=False)
        return JudgeResult(**verdict, metadata=JudgeMetadata(
            case_id=value.case_id, judge_model=self.settings.model, prompt_version=self.prompt_version,
            response_format="json", latency_ms=(perf_counter() - start) * 1000, attempts=1,
            raw_output=raw, attempt_history=[JudgeAttempt(attempt=1, raw_output=raw)],
        ))
