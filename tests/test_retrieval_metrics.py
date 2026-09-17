# 文件作用：用手算数据测试检索公式、边界值和宏平均。
# 为什么有它：确保分数来自正确的编号、排名和分母，而不是模型判断。
import pytest
from pydantic import ValidationError

from src.dataset import EvalCase, RetrievedDocument
from src.metrics import (
    RetrievalMetricResult,
    aggregate_retrieval_metrics,
    evaluate_retrieval,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


# 做什么：按传入编号顺序生成连续排名的检索结果。
# 为什么需要：让测试预期可以直接手算。
def documents(chunk_ids):
    return [
        RetrievedDocument(doc_id="parent", chunk_id=chunk_id, content="Fixture", score=0.5, rank=rank)
        for rank, chunk_id in enumerate(chunk_ids, 1)
    ]


# 做什么：生成指定相关块与可回答性的题目。
# 为什么需要：为不同公式边界提供可控标注。
def case(relevant, *, case_id="case-1", answerable=True):
    return EvalCase(
        id=case_id, query="Fixture query", expected_answer="Fixture answer",
        relevant_doc_ids=relevant, answerable=answerable,
        category="normal" if answerable else "unanswerable",
    )


# 做什么：用手工排名核对召回、精确率与倒数排名。
# 为什么需要：测试预期来自独立手算而非重复同一实现。
@pytest.mark.parametrize("retrieved,relevant,k,expected", [
    (["a"], ["a"], 1, (1, 1, 1)),
    (["x", "a"], ["a"], 2, (1, 0.5, 0.5)),
    (["x", "y", "a"], ["a"], 3, (1, 1 / 3, 1 / 3)),
    (["x", "y", "a"], ["a"], 2, (0, 0, 0)),
    (["x", "y"], ["a"], 3, (0, 0, 0)),
    (["x", "a", "b"], ["a", "b"], 3, (1, 2 / 3, 0.5)),
    (["a", "x"], ["a", "b"], 2, (0.5, 0.5, 1)),
    (["a", "b"], ["a", "b"], 1, (0.5, 1, 1)),
    (["a", "x"], ["a"], 5, (1, 0.5, 1)),
    (["a", "a", "b"], ["a", "b"], 3, (1, 2 / 3, 1)),
    (["a", "a", "b"], ["a", "b"], 2, (0.5, 0.5, 1)),
    (["x", "x", "a"], ["a"], 3, (1, 1 / 3, 1 / 3)),
    (["a"], ["a", "a", "b"], 1, (0.5, 1, 1)),
    ([], ["a"], 3, (0, 0, 0)),
    (["child"], ["parent"], 1, (0, 0, 0)),
])
def test_hand_calculated_metrics(retrieved, relevant, k, expected):
    docs = documents(retrieved)
    result = evaluate_retrieval(case(relevant), docs, k)
    assert (result.recall_at_k, result.precision_at_k, result.rr) == pytest.approx(expected)
    assert recall_at_k(docs, relevant, k) == pytest.approx(expected[0])
    assert precision_at_k(docs, relevant, k) == pytest.approx(expected[1])
    assert reciprocal_rank(docs[:k], relevant) == pytest.approx(expected[2])
    assert result.retrieved_chunk_ids == retrieved[:k]
    assert result.relevant_chunk_ids == list(dict.fromkeys(relevant))


# 做什么：验证 RR 只看给定范围内首次命中。
# 为什么需要：不能把后续命中累加或把 K 外结果算进来。
def test_rr_uses_first_hit_and_supplied_ranking_cutoff():
    docs = documents(["x", "a", "b"])
    assert reciprocal_rank(docs, ["a", "b"]) == 0.5
    assert evaluate_retrieval(case(["a", "b"]), docs, 1).rr == 0
    assert evaluate_retrieval(case(["a", "b"]), docs, 3).rr == 0.5


# 做什么：验证不可回答题也拒绝非法 K。
# 为什么需要：排除计分不代表可以绕过接口参数校验。
@pytest.mark.parametrize("k", [0, -1, -100, 1.5, True])
def test_invalid_k_is_clear_even_for_excluded_cases(k):
    for metric in (recall_at_k, precision_at_k):
        with pytest.raises(ValueError, match="k must be a positive integer"):
            metric([], ["a"], k)
    with pytest.raises(ValueError, match="k must be a positive integer"):
        evaluate_retrieval(case([], answerable=False), [], k)
    with pytest.raises(ValueError, match="k must be a positive integer"):
        aggregate_retrieval_metrics([], k)


# 做什么：验证非连续或矛盾排名明确报错。
# 为什么需要：不能静默改排名导致 RR 含义变化。
@pytest.mark.parametrize("ranks", [[2], [1, 3], [2, 1], [1, 1]])
def test_inconsistent_rankings_are_rejected(ranks):
    docs = documents([f"chunk-{index}" for index in range(len(ranks))])
    for doc, rank in zip(docs, ranks):
        doc.rank = rank
    with pytest.raises(ValueError, match="consecutive ranks starting at 1"):
        evaluate_retrieval(case(["a"]), docs, 3)


# 做什么：验证不可回答题保留空分数并标为排除。
# 为什么需要：不应人为用零分拉低普通检索成绩。
def test_unanswerable_is_excluded_with_null_scores():
    result = evaluate_retrieval(case([], answerable=False), documents(["a"]), 3)
    assert result.answerable is False
    assert (result.recall_at_k, result.precision_at_k, result.rr) == (None, None, None)
    assert result.retrieved_chunk_ids == ["a"]
    assert result.relevant_chunk_ids == []
    assert RetrievalMetricResult.model_validate_json(result.model_dump_json()) == result


# 做什么：验证空或矛盾相关标注明确失败。
# 为什么需要：错误数据不能被掩盖成没有命中。
def test_invalid_relevance_labels_do_not_silently_score_zero():
    with pytest.raises(ValueError, match="answerable=true requires non-empty relevant_doc_ids"):
        evaluate_retrieval(case([]), [], 3)
    with pytest.raises(ValueError, match="answerable=false requires relevant_doc_ids"):
        evaluate_retrieval(case(["a"], answerable=False), [], 3)
    with pytest.raises(ValueError, match="relevant_chunk_ids must not be empty"):
        recall_at_k([], [], 3)
    with pytest.raises(ValueError, match="relevant_chunk_ids must not be empty"):
        precision_at_k([], [], 3)
    with pytest.raises(ValueError, match="relevant_chunk_ids must not be empty"):
        reciprocal_rank([], [])


# 做什么：验证宏平均与 MRR 只统计有效可回答题。
# 为什么需要：各道题权重及排除规则必须正确。
def test_macro_aggregation_and_mrr_exclude_unanswerable():
    results = [
        evaluate_retrieval(case(["a", "b"], case_id="first"), documents(["a"]), 3),
        evaluate_retrieval(case(["c"], case_id="second"), documents(["x", "c"]), 3),
        evaluate_retrieval(case(["d"], case_id="third"), documents(["x", "y", "d"]), 3),
        evaluate_retrieval(case(["e"], case_id="miss"), [], 3),
        evaluate_retrieval(case([], case_id="skip", answerable=False), [], 3),
    ]
    summary = aggregate_retrieval_metrics(results, 3)
    assert summary.evaluated_cases == 4
    assert summary.excluded_unanswerable_cases == 1
    assert summary.mean_recall_at_k == pytest.approx((0.5 + 1 + 1 + 0) / 4)
    assert summary.mean_precision_at_k == pytest.approx((1 + 0.5 + 1 / 3 + 0) / 4)
    assert summary.mrr == pytest.approx((1 + 0.5 + 1 / 3 + 0) / 4)


# 做什么：验证没有有效题时平均值为空。
# 为什么需要：没有测量不能生成假零分或发生除零。
@pytest.mark.parametrize("excluded_count", [0, 2])
def test_no_evaluable_cases_produce_null_averages(excluded_count):
    results = [
        evaluate_retrieval(case([], case_id=f"skip-{index}", answerable=False), [], 3)
        for index in range(excluded_count)
    ]
    summary = aggregate_retrieval_metrics(results, 3)
    assert summary.evaluated_cases == 0
    assert summary.excluded_unanswerable_cases == excluded_count
    assert (summary.mean_recall_at_k, summary.mean_precision_at_k, summary.mrr) == (None, None, None)


# 做什么：验证不同 K 或重复题号不能混合聚合。
# 为什么需要：保证统计口径与样本权重一致。
def test_aggregation_rejects_mixed_k_and_duplicate_case_ids():
    result = evaluate_retrieval(case(["a"]), documents(["a"]), 1)
    with pytest.raises(ValueError, match="different k"):
        aggregate_retrieval_metrics([result], 3)
    with pytest.raises(ValueError, match="duplicate case_id"):
        aggregate_retrieval_metrics([result, result], 1)


# 做什么：验证结果模型拒绝非法分数和参评状态。
# 为什么需要：不能手工构造矛盾结果绕过前面的计算检查。
@pytest.mark.parametrize("update", [
    {"recall_at_k": 1.1}, {"precision_at_k": -0.1}, {"rr": float("nan")},
    {"rr": None}, {"answerable": False}, {"relevant_chunk_ids": []},
    {"retrieved_chunk_ids": ["a", "b"]},
])
def test_result_model_rejects_invalid_scores_or_participation(update):
    result = evaluate_retrieval(case(["a"]), documents(["a"]), 1).model_dump()
    with pytest.raises(ValidationError):
        RetrievalMetricResult(**(result | update))
