# 文件作用：测试检索基准实验的调用次数、K 截断和失败明细。
# 为什么有它：保证不同 K 使用同一次排名，错误标注在检索前被发现。
from unittest.mock import Mock

import pytest

from src.dataset import EvalCase, RetrievedDocument
from src.metrics.benchmark import benchmark_retrieval


# 做什么：按给定编号和相关块生成评测题。
# 为什么需要：便于安排部分命中、未命中与排除场景。
def make_case(case_id, refs, answerable=True):
    return EvalCase(
        id=case_id, query=f"Question {case_id}", expected_answer="Fixture",
        relevant_doc_ids=refs, answerable=answerable,
        category="normal" if answerable else "unanswerable",
    )


# 做什么：验证每道可回答题只检索一次并保存失败明细。
# 为什么需要：不同 K 必须比较同一排名前缀。
def test_benchmark_uses_one_ranking_per_answerable_case_and_records_failures():
    cases = [make_case("multi", ["a", "b"]), make_case("miss", ["z"]), make_case("skip", [], False)]
    retrieve = Mock(return_value=[
        RetrievedDocument(doc_id="parent", chunk_id=chunk, content="Fixture", score=0.5, rank=rank)
        for rank, chunk in enumerate(["x", "a", "b", "y", "w"], 1)
    ])
    report = benchmark_retrieval(cases, retrieve)
    assert retrieve.call_count == 2
    assert [call.args for call in retrieve.call_args_list] == [("Question multi", 5), ("Question miss", 5)]
    assert report["excluded_case_ids"] == ["skip"]
    assert report["by_k"]["1"]["summary"]["evaluated_cases"] == 2
    assert report["by_k"]["3"]["summary"]["excluded_unanswerable_cases"] == 1
    assert report["by_k"]["3"]["summary"]["mrr"] == 0.25
    assert report["by_k"]["1"]["failure_cases"] == 2
    assert report["by_k"]["3"]["failure_cases"] == 1
    failure = report["by_k"]["3"]["failures"][0]
    assert failure["case_id"] == "miss"
    assert failure["query"] == "Question miss"
    assert failure["relevant_chunk_ids"] == ["z"]
    assert failure["retrieved_chunk_ids"] == ["x", "a", "b"]
    assert failure["missing_relevant_chunk_ids"] == ["z"]
    assert all(result["rr"] is None for result in report["by_k"]["5"]["results"] if not result["answerable"])


# 做什么：验证部分漏召回与完全未命中分开统计。
# 为什么需要：两类检索问题不能混成同一个结论。
def test_partial_relevance_is_a_failure_but_not_a_no_hit():
    retrieve = Mock(return_value=[
        RetrievedDocument(doc_id="parent", chunk_id="a", content="Fixture", score=1, rank=1)
    ])
    result = benchmark_retrieval([make_case("partial", ["a", "b"])], retrieve)["by_k"]["1"]
    assert result["failure_cases"] == 1
    assert result["no_hit_cases"] == 0
    assert result["failures"][0]["recall_at_k"] == 0.5


# 做什么：验证错误标注在检索之前被拒绝。
# 为什么需要：避免用无效试卷浪费真实检索工作。
def test_bad_labels_fail_before_any_retrieval():
    retrieve = Mock()
    with pytest.raises(ValueError, match="non-empty relevant_doc_ids"):
        benchmark_retrieval([make_case("valid", ["a"]), make_case("invalid", [])], retrieve)
    retrieve.assert_not_called()


# 做什么：验证重复题号在基准实验开始前失败。
# 为什么需要：重复覆盖会破坏逐题结果与分母。
def test_duplicate_case_ids_fail_before_any_retrieval():
    retrieve = Mock()
    with pytest.raises(ValueError, match="duplicate case IDs"):
        benchmark_retrieval([make_case("same", ["a"]), make_case("same", ["b"])], retrieve)
    retrieve.assert_not_called()
