"""Stage-specific denominators; failed and unassessed cases remain visible."""

from collections.abc import Sequence
from collections import Counter
from statistics import mean

from .models import (
    CaseEvaluationResult, EngineeringSummary, EvaluationSummary,
    GenerationSummary, RetrievalSummary, SafetySummary,
)


def aggregate_results(results: Sequence[CaseEvaluationResult]) -> EvaluationSummary:
    if len({row.case_id for row in results}) != len(results):
        raise ValueError("Cannot aggregate duplicate case_id values")
    retrieval = [row.retrieval_metrics for row in results if row.answerable and row.retrieval_metrics is not None]
    if len({row.k for row in retrieval}) > 1:
        raise ValueError("Cannot aggregate different retrieval k values")
    judged = [row.judge_result for row in results if row.judge_result is not None]
    unanswerable = [row for row in results if not row.answerable]
    correct_refusals = sum(row.refusal_correct is True for row in unanswerable)
    hallucinations = sum(row.hallucination for row in judged)
    success = sum(row.status == "SUCCESS" for row in results)
    return EvaluationSummary(
        retrieval=RetrievalSummary(
            evaluated_retrieval_cases=len(retrieval), excluded_unanswerable_cases=len(unanswerable),
            unavailable_retrieval_cases=sum(row.answerable for row in results) - len(retrieval),
            recall_at_k=mean(row.recall_at_k for row in retrieval) if retrieval else None,
            precision_at_k=mean(row.precision_at_k for row in retrieval) if retrieval else None,
            mrr=mean(row.rr for row in retrieval) if retrieval else None,
        ),
        generation=GenerationSummary(
            evaluated_judge_cases=len(judged),
            avg_correctness=mean(row.answer_correctness for row in judged) if judged else None,
            avg_faithfulness=mean(row.faithfulness for row in judged) if judged else None,
            avg_relevance=mean(row.answer_relevance for row in judged) if judged else None,
            avg_completeness=mean(row.completeness for row in judged) if judged else None,
        ),
        safety=SafetySummary(
            evaluated_judge_cases=len(judged), hallucination_cases=hallucinations,
            hallucination_rate=hallucinations / len(judged) if judged else None,
            unanswerable_cases=len(unanswerable), correct_refusal_cases=correct_refusals,
            unassessed_refusal_cases=sum(row.refusal_correct is None for row in unanswerable),
            refusal_accuracy=correct_refusals / len(unanswerable) if unanswerable else None,
        ),
        engineering=EngineeringSummary(
            total_cases=len(results), success_cases=success, failed_cases=len(results) - success,
            evaluation_success_rate=success / len(results) if results else None,
            average_latency=mean(row.latency_ms for row in results) if results else None,
            judge_cache_hits=sum(row.judge_cache_hit for row in results),
            retry_count=sum(row.retry_count for row in results),
            failure_breakdown=dict(Counter(row.error_type for row in results if row.status != "SUCCESS")),
        ),
    )
