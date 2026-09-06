"""Strict Judge verdicts and application-owned execution metadata."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.dataset.models import NonEmptyString
from src.llm.models import ProviderTrace


Score = Annotated[int, Field(ge=0, le=5, strict=True)]


class JudgeInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    case_id: NonEmptyString
    query: NonEmptyString
    expected_answer: str
    retrieved_context: str
    rag_answer: str
    answerable: bool


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

    @model_validator(mode="after")
    def check_hallucination_evidence(self) -> "JudgeVerdict":
        if self.hallucination != bool(self.unsupported_claims):
            raise ValueError("hallucination must be true exactly when unsupported_claims is non-empty")
        return self


class JudgeAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attempt: int = Field(ge=1)
    raw_output: str | None
    error_category: str | None = None
    error_message: str | None = None
    status_code: int | None = None


class JudgeVerdictV2(JudgeVerdict):
    """Explicit refusal observation; required by the evaluation runner."""

    refusal_detected: bool


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


class JudgeResult(JudgeVerdict):
    # None denotes a historical judge_v1 result, not a negative refusal observation.
    refusal_detected: bool | None = None
    metadata: JudgeMetadata


class JudgeFailure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str
    message: str
    metadata: JudgeMetadata
