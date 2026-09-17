# 文件作用：规定裁判输入、评分字段、拒答信息及调用元数据。
# 为什么有它：先检查裁判输出是否结构合法，再允许进入报告和缓存。
"""Strict Judge verdicts and application-owned execution metadata."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.dataset.models import NonEmptyString
from src.llm.models import ProviderTrace


Score = Annotated[int, Field(ge=0, le=5, strict=True)]


# 这个类：保存裁判需要的问题、依据和待评答案。
# 为什么需要：让各评分维度使用明确的参照。
class JudgeInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    case_id: NonEmptyString
    query: NonEmptyString
    expected_answer: str
    retrieved_context: str
    rag_answer: str
    answerable: bool


# 这个类：规定裁判必须返回的四项分数、幻觉证据和理由。
# 为什么需要：模型自由文本不能直接作为合法评测记录。
class JudgeVerdict(BaseModel):
    """The seven fields the external model must return, and nothing else."""

    model_config = ConfigDict(extra="forbid", strict=True)

    answer_correctness: Score
    faithfulness: Score
    answer_relevance: Score
    completeness: Score
    hallucination: bool
    unsupported_claims: list[NonEmptyString]
    reason: NonEmptyString

    # 做什么：要求幻觉标记与无依据断言列表一致。
    # 为什么需要：防止只说有幻觉却没有证据，或列出编造事实却标为无幻觉。
    @model_validator(mode="after")
    def check_hallucination_evidence(self) -> "JudgeVerdict":
        if self.hallucination != bool(self.unsupported_claims):
            raise ValueError("hallucination must be true exactly when unsupported_claims is non-empty")
        return self


# 这个类：记录一次判卷尝试的原文和错误。
# 为什么需要：重试后仍能知道前面失败了什么。
class JudgeAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attempt: int = Field(ge=1)
    raw_output: str | None
    error_category: str | None = None
    error_message: str | None = None
    status_code: int | None = None


# 这个类：在基础评分上增加明确的拒答观察字段。
# 为什么需要：Runner 的拒答统计需要直接依据。
class JudgeVerdictV2(JudgeVerdict):
    """Explicit refusal observation; required by the evaluation runner."""

    refusal_detected: bool


# 这个类：保存由程序记录的模型、版本、耗时与请求轨迹。
# 为什么需要：这些运行事实不能由裁判自行编造。
class JudgeMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    judge_model: str
    prompt_version: str
    response_format: str
    temperature: int = 0
    latency_ms: float = Field(ge=0, allow_inf_nan=False)
    attempts: int = Field(ge=1)
    raw_output: str | None
    attempt_history: list[JudgeAttempt]
    retry_count: int = Field(default=0, ge=0)
    provider_traces: list[ProviderTrace] = Field(default_factory=list)


# 这个类：把合法评分与运行元数据合并。
# 为什么需要：报告和缓存能同时保存质量结果与来源。
class JudgeResult(JudgeVerdict):
    # None denotes a historical judge_v1 result, not a negative refusal observation.
    refusal_detected: bool | None = None
    metadata: JudgeMetadata


# 这个类：把裁判失败类别、说明与元数据合并。
# 为什么需要：失败路径也能完整序列化留存。
class JudgeFailure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str
    message: str
    metadata: JudgeMetadata
