"""Serializable case outcomes and dataset-level evaluation reports."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.dataset import RAGResult
from src.judge import JudgeResult
from src.judge.models import JudgeFailure
from src.metrics import RetrievalMetricResult
from src.llm.models import ProviderFailure


Status = Literal["SUCCESS", "RAG_ERROR", "RETRIEVAL_ERROR", "JUDGE_ERROR", "TIMEOUT"]


class CaseError(BaseModel):
    stage: Literal["rag", "retrieval", "judge"]
    status: Status
    error_type: str
    message: str
    judge_failure: JudgeFailure | None = None
    provider_failure: ProviderFailure | None = None


class CaseEvaluationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    query: str
    category: str
    answerable: bool
    rag_result: RAGResult | None = None
    retrieval_metrics: RetrievalMetricResult | None = None
    judge_result: JudgeResult | None = None
    refusal_correct: bool | None = None
    status: Status = "SUCCESS"
    error_type: str | None = None
    errors: list[CaseError] = Field(default_factory=list)
    latency_ms: float = Field(default=0, ge=0, allow_inf_nan=False)
    judge_cache_hit: bool = False
    cache_warning: str | None = None
    rag_retry_count: int = Field(default=0, ge=0)
    judge_retry_count: int = Field(default=0, ge=0)
    retry_count: int = Field(default=0, ge=0)


class RetrievalSummary(BaseModel):
    evaluated_retrieval_cases: int
    excluded_unanswerable_cases: int
    unavailable_retrieval_cases: int
    recall_at_k: float | None
    precision_at_k: float | None
    mrr: float | None


class GenerationSummary(BaseModel):
    evaluated_judge_cases: int
    avg_correctness: float | None
    avg_faithfulness: float | None
    avg_relevance: float | None
    avg_completeness: float | None


class SafetySummary(BaseModel):
    evaluated_judge_cases: int
    hallucination_cases: int
    hallucination_rate: float | None
    unanswerable_cases: int
    correct_refusal_cases: int
    unassessed_refusal_cases: int
    refusal_accuracy: float | None


class EngineeringSummary(BaseModel):
    total_cases: int
    success_cases: int
    failed_cases: int
    evaluation_success_rate: float | None
    average_latency: float | None
    judge_cache_hits: int
    retry_count: int = 0
    failure_breakdown: dict[str, int] = Field(default_factory=dict)


class EvaluationSummary(BaseModel):
    retrieval: RetrievalSummary
    generation: GenerationSummary
    safety: SafetySummary
    engineering: EngineeringSummary


class EvaluationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    dataset_path: str
    dataset_version: str
    dataset_sha256: str
    corpus_sha256: str
    top_k: int = Field(gt=0, strict=True)
    embedding_model: str
    rag_model: str
    judge_model: str
    judge_prompt_version: str
    rag_mode: Literal["LIVE", "MOCK"]
    judge_mode: Literal["LIVE", "MOCK"]
    quality_metrics_are_synthetic: bool
    run_latency_ms: float = Field(ge=0, allow_inf_nan=False)
    summary: EvaluationSummary
    case_results: list[CaseEvaluationResult]
    execution_config: dict = Field(default_factory=dict)
