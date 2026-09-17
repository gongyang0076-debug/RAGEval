# 文件作用：规定逐题结果、分项汇总和整份评测报告的数据结构。
# 为什么有它：让成功、失败、缺失分数和运行版本都有固定位置可查询。
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


# 这个类：保存某一阶段的错误与相关调用详情。
# 为什么需要：失败不能只留一个模糊状态。
class CaseError(BaseModel):
    stage: Literal["rag", "retrieval", "judge"]
    status: Status
    error_type: str
    message: str
    judge_failure: JudgeFailure | None = None
    provider_failure: ProviderFailure | None = None


# 这个类：保存一道题各阶段结果和执行状态。
# 为什么需要：成功与失败都可以逐条追溯。
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


# 这个类：保存检索均分及有效、排除、不可用题数。
# 为什么需要：解释检索分数的实际统计范围。
class RetrievalSummary(BaseModel):
    evaluated_retrieval_cases: int
    excluded_unanswerable_cases: int
    unavailable_retrieval_cases: int
    recall_at_k: float | None
    precision_at_k: float | None
    mrr: float | None


# 这个类：保存有效 Judge 数量和四项平均评分。
# 为什么需要：没有成功评分的题不应假装参与平均。
class GenerationSummary(BaseModel):
    evaluated_judge_cases: int
    avg_correctness: float | None
    avg_faithfulness: float | None
    avg_relevance: float | None
    avg_completeness: float | None


# 这个类：保存幻觉与拒答的数量和比例。
# 为什么需要：让安全指标的分子分母清楚可查。
class SafetySummary(BaseModel):
    evaluated_judge_cases: int
    hallucination_cases: int
    hallucination_rate: float | None
    unanswerable_cases: int
    correct_refusal_cases: int
    unassessed_refusal_cases: int
    refusal_accuracy: float | None


# 这个类：保存总题数、成功失败、耗时和重试等。
# 为什么需要：区分工程稳定性与答案质量。
class EngineeringSummary(BaseModel):
    total_cases: int
    success_cases: int
    failed_cases: int
    evaluation_success_rate: float | None
    average_latency: float | None
    judge_cache_hits: int
    retry_count: int = 0
    failure_breakdown: dict[str, int] = Field(default_factory=dict)


# 这个类：把四类汇总放在一起。
# 为什么需要：报告和页面可以按同一结构读取总览。
class EvaluationSummary(BaseModel):
    retrieval: RetrievalSummary
    generation: GenerationSummary
    safety: SafetySummary
    engineering: EngineeringSummary


# 这个类：保存整次评测的版本、配置、汇总及逐题证据。
# 为什么需要：生成可序列化、可比较的完整报告。
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
