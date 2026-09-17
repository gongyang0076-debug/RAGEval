# 文件作用：测试知识块、评测题和 RAG 结果的基础字段约束及 JSON 往返。
# 为什么有它：保证模块间数据格式一致，空值与非法值有明确区别。
import pytest
from pydantic import ValidationError

from src.dataset import CorpusDocument, EvalCase, RAGResult, RetrievedDocument


# 做什么：创建允许覆盖字段的基础评测题。
# 为什么需要：减少重复准备并突出每个测试的边界条件。
def make_case(**overrides):
    values = dict(
        id="case-1", query="What is RAGEval?", expected_answer="An evaluation framework",
        relevant_doc_ids=["chunk-1"], answerable=True, category="definition",
    )
    return EvalCase(**(values | overrides))


# 做什么：验证两条知识块的默认元数据互不影响。
# 为什么需要：修改一条资料不能意外改变另一条。
def test_corpus_metadata_defaults_are_independent():
    first = CorpusDocument(doc_id="doc-1", chunk_id="chunk-1", content="First chunk")
    second = CorpusDocument(doc_id="doc-1", chunk_id="chunk-2", content="Second chunk")
    first.metadata["source"] = "manual"
    assert second.metadata == {}


# 做什么：验证可选同义组与不可回答样例能表达。
# 为什么需要：数据模型必须覆盖试卷里的不同题型。
def test_eval_case_optional_variant_and_unanswerable_case():
    assert make_case().variant_group is None
    case = make_case(answerable=False, expected_answer="", relevant_doc_ids=[], variant_group="g1")
    assert case.variant_group == "g1"
    assert case.answerable is False


# 做什么：验证嵌套 RAG 结果能保存成 JSON 再完整恢复。
# 为什么需要：报告交换不能丢失检索结果类型和内容。
def test_nested_result_json_round_trip():
    result = RAGResult(
        query="What is RAGEval?",
        retrieved_docs=[dict(doc_id="doc-1", chunk_id="chunk-1", content="Evaluation framework", score=-0.5, rank=1)],
        rag_answer="An evaluation framework", latency_ms=12.5,
    )
    assert isinstance(result.retrieved_docs[0], RetrievedDocument)
    assert RAGResult.model_validate_json(result.model_dump_json()) == result


# 做什么：验证基础模型允许空检索和空答案记录。
# 为什么需要：模型层需要表达实际空结果，生成是否成功由客户端再判断。
def test_empty_retrieval_and_answer_are_valid():
    result = RAGResult(query="Unknown?", retrieved_docs=[], rag_answer="", latency_ms=0)
    assert result.retrieved_docs == []


# 做什么：验证知识块必填文本不能只含空白。
# 为什么需要：空编号和空内容没有可靠业务含义。
@pytest.mark.parametrize("field", ["doc_id", "chunk_id", "content"])
def test_corpus_rejects_blank_required_text(field):
    values = dict(doc_id="doc-1", chunk_id="chunk-1", content="Text")
    values[field] = "   "
    with pytest.raises(ValidationError):
        CorpusDocument(**values)


# 做什么：验证排名必须是合法正整数。
# 为什么需要：RR 依赖明确的排名位置。
@pytest.mark.parametrize("rank", [0, -1, 1.5, True])
def test_retrieved_document_rejects_invalid_rank(rank):
    with pytest.raises(ValidationError):
        RetrievedDocument(doc_id="doc-1", chunk_id="chunk-1", content="Text", score=0.5, rank=rank)


# 做什么：验证检索分数不能是 NaN 或无穷。
# 为什么需要：非法浮点值不能进入排序和报告。
@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_retrieved_document_rejects_nonfinite_score(score):
    with pytest.raises(ValidationError):
        RetrievedDocument(doc_id="doc-1", chunk_id="chunk-1", content="Text", score=score, rank=1)


# 做什么：验证知识块编号不能为空或非法。
# 为什么需要：评测必须能对齐真实标注。
@pytest.mark.parametrize("chunk_id", [None, "", "   "])
def test_retrieved_document_rejects_invalid_chunk_id(chunk_id):
    with pytest.raises(ValidationError):
        RetrievedDocument(doc_id="doc-1", chunk_id=chunk_id, content="Text", score=1, rank=1)


# 做什么：验证检索结果缺 chunk_id 时失败。
# 为什么需要：防止旧契约遗漏评分需要的关键字段。
def test_retrieved_document_requires_chunk_id():
    with pytest.raises(ValidationError, match="chunk_id"):
        RetrievedDocument(doc_id="doc-1", content="Text", score=1, rank=1)


# 做什么：验证耗时不能为负数或非有限值。
# 为什么需要：工程统计需要可解释的实际数值。
@pytest.mark.parametrize("latency", [-1, float("nan"), float("inf")])
def test_result_rejects_invalid_latency(latency):
    with pytest.raises(ValidationError):
        RAGResult(query="Query", retrieved_docs=[], rag_answer="", latency_ms=latency)


# 做什么：验证题目必须带标注且不能出现拼错的额外字段。
# 为什么需要：尽早发现数据契约使用错误。
def test_eval_case_requires_labels_and_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        EvalCase(id="case-1", query="Query")
    with pytest.raises(ValidationError):
        make_case(unexpected="typo")
