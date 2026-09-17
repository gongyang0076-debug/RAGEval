# 文件作用：标记“检索指标”目录为 Python 包，供其他模块导入。
# 为什么有它：让相关文件能够通过统一的包路径组织和引用。
"""Retrieval-only metrics; no generation or LLM judging."""

from .models import RetrievalMetricResult, RetrievalMetricSummary
from .retrieval import (
    aggregate_retrieval_metrics,
    evaluate_retrieval,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

__all__ = [
    "RetrievalMetricResult", "RetrievalMetricSummary", "recall_at_k", "precision_at_k",
    "reciprocal_rank", "evaluate_retrieval", "aggregate_retrieval_metrics",
]
