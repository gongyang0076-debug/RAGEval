# 文件作用：按知识块编号计算 Recall、Precision、RR 和多题平均结果。
# 为什么有它：使用确定的公式衡量检索表现，不让模型来猜测是否命中。
"""Chunk-level, binary relevance metrics with explicit duplicate semantics."""

from collections.abc import Sequence
from statistics import mean

from src.dataset import EvalCase, RetrievedDocument

from .models import RetrievalMetricResult, RetrievalMetricSummary


# 做什么：检查 K 是正整数并拒绝布尔值。
# 为什么需要：防止无意义的截断参数进入计算。
def _validate_k(k: int) -> None:
    if type(k) is not int or k <= 0:
        raise ValueError("k must be a positive integer")


# 做什么：校验排名从 1 连续排列，然后取前 K 条。
# 为什么需要：RR 依赖真实位置，不能静默修复矛盾排名。
def _top_k(retrieved_docs: Sequence[RetrievedDocument], k: int) -> list[RetrievedDocument]:
    _validate_k(k)
    # Refuse contradictory rankings instead of silently changing the RR denominator.
    if any(doc.rank != position for position, doc in enumerate(retrieved_docs, start=1)):
        raise ValueError("retrieved_docs must be ordered with consecutive ranks starting at 1")
    return list(retrieved_docs[:k])


# 做什么：根据唯一相关块计算召回、精确率和第一命中的倒数排名。
# 为什么需要：把重复命中、空结果和分母规则集中到同一处。
def _scores(top_docs: Sequence[RetrievedDocument], relevant_chunk_ids: Sequence[str]) -> tuple[float, float, float]:
    relevant = set(relevant_chunk_ids)
    if not relevant:
        raise ValueError("relevant_chunk_ids must not be empty; exclude unanswerable cases instead")
    # Duplicate hits cannot increase the numerator, but still occupy returned slots.
    hits = len({doc.chunk_id for doc in top_docs} & relevant)
    recall = hits / len(relevant)
    precision = hits / len(top_docs) if top_docs else 0.0
    rr = next((1.0 / doc.rank for doc in top_docs if doc.chunk_id in relevant), 0.0)
    return recall, precision, rr


# 做什么：计算需要的相关块在前 K 条中找到了多少。
# 为什么需要：衡量检索是否漏掉必要资料。
def recall_at_k(retrieved_docs: Sequence[RetrievedDocument], relevant_chunk_ids: Sequence[str], k: int) -> float:
    """Unique relevant hits in Top-K divided by unique relevant labels."""
    return _scores(_top_k(retrieved_docs, k), relevant_chunk_ids)[0]


# 做什么：计算前 K 条实际返回结果中有多少唯一相关命中。
# 为什么需要：衡量返回内容的标注相关程度，重复结果仍占位置。
def precision_at_k(retrieved_docs: Sequence[RetrievedDocument], relevant_chunk_ids: Sequence[str], k: int) -> float:
    """Unique relevant hits divided by actual returned Top-K slots, including duplicates."""
    return _scores(_top_k(retrieved_docs, k), relevant_chunk_ids)[1]


# 做什么：计算给定排名中第一个相关块的位置倒数。
# 为什么需要：衡量正确资料是否排得靠前。
def reciprocal_rank(retrieved_docs: Sequence[RetrievedDocument], relevant_chunk_ids: Sequence[str]) -> float:
    """Reciprocal rank of the first relevant chunk in the supplied ranking, or zero."""
    return _scores(_top_k(retrieved_docs, max(1, len(retrieved_docs))), relevant_chunk_ids)[2]


# 做什么：为单道题生成完整检索指标，并对不可回答题保留空分数。
# 为什么需要：统一普通计分与明确排除的处理。
def evaluate_retrieval(case: EvalCase, retrieved_docs: Sequence[RetrievedDocument], k: int) -> RetrievalMetricResult:
    """Score one case at K; retain excluded cases with None metrics for counting."""
    top_docs = _top_k(retrieved_docs, k)
    relevant = list(dict.fromkeys(case.relevant_doc_ids))
    if not case.answerable:
        if relevant:
            raise ValueError(f"case {case.id!r}: answerable=false requires relevant_doc_ids=[]")
        recall = precision = rr = None
    else:
        if not relevant:
            raise ValueError(f"case {case.id!r}: answerable=true requires non-empty relevant_doc_ids")
        recall, precision, rr = _scores(top_docs, relevant)
    return RetrievalMetricResult(
        case_id=case.id, k=k, answerable=case.answerable,
        recall_at_k=recall, precision_at_k=precision, rr=rr,
        retrieved_chunk_ids=[doc.chunk_id for doc in top_docs], relevant_chunk_ids=relevant,
    )


# 做什么：检查 K 与题号后对有效题做宏平均。
# 为什么需要：防止重复题、不同 K 或不可回答题混入分母。
def aggregate_retrieval_metrics(results: Sequence[RetrievalMetricResult], k: int) -> RetrievalMetricSummary:
    """Macro-average valid cases at one K, never treating exclusions as zero."""
    _validate_k(k)
    if any(result.k != k for result in results):
        raise ValueError("Cannot aggregate metric results with different k values")
    if len({result.case_id for result in results}) != len(results):
        raise ValueError("Cannot aggregate duplicate case_id values")
    included = [result for result in results if result.answerable]
    return RetrievalMetricSummary(
        k=k, evaluated_cases=len(included),
        excluded_unanswerable_cases=len(results) - len(included),
        mean_recall_at_k=mean(result.recall_at_k for result in included) if included else None,
        mean_precision_at_k=mean(result.precision_at_k for result in included) if included else None,
        mrr=mean(result.rr for result in included) if included else None,
    )
