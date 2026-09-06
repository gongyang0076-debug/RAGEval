import pytest
from pydantic import ValidationError

from src.dataset import CorpusDocument, EvalCase, RAGResult, RetrievedDocument


def make_case(**overrides):
    values = dict(
        id="case-1", query="What is RAGEval?", expected_answer="An evaluation framework",
        relevant_doc_ids=["chunk-1"], answerable=True, category="definition",
    )
    return EvalCase(**(values | overrides))


def test_corpus_metadata_defaults_are_independent():
    first = CorpusDocument(doc_id="doc-1", chunk_id="chunk-1", content="First chunk")
    second = CorpusDocument(doc_id="doc-1", chunk_id="chunk-2", content="Second chunk")
    first.metadata["source"] = "manual"
    assert second.metadata == {}


def test_eval_case_optional_variant_and_unanswerable_case():
    assert make_case().variant_group is None
    case = make_case(answerable=False, expected_answer="", relevant_doc_ids=[], variant_group="g1")
    assert case.variant_group == "g1"
    assert case.answerable is False


def test_nested_result_json_round_trip():
    result = RAGResult(
        query="What is RAGEval?",
        retrieved_docs=[dict(doc_id="doc-1", chunk_id="chunk-1", content="Evaluation framework", score=-0.5, rank=1)],
        rag_answer="An evaluation framework", latency_ms=12.5,
    )
    assert isinstance(result.retrieved_docs[0], RetrievedDocument)
    assert RAGResult.model_validate_json(result.model_dump_json()) == result


def test_empty_retrieval_and_answer_are_valid():
    result = RAGResult(query="Unknown?", retrieved_docs=[], rag_answer="", latency_ms=0)
    assert result.retrieved_docs == []


@pytest.mark.parametrize("field", ["doc_id", "chunk_id", "content"])
def test_corpus_rejects_blank_required_text(field):
    values = dict(doc_id="doc-1", chunk_id="chunk-1", content="Text")
    values[field] = "   "
    with pytest.raises(ValidationError):
        CorpusDocument(**values)


@pytest.mark.parametrize("rank", [0, -1, 1.5, True])
def test_retrieved_document_rejects_invalid_rank(rank):
    with pytest.raises(ValidationError):
        RetrievedDocument(doc_id="doc-1", chunk_id="chunk-1", content="Text", score=0.5, rank=rank)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_retrieved_document_rejects_nonfinite_score(score):
    with pytest.raises(ValidationError):
        RetrievedDocument(doc_id="doc-1", chunk_id="chunk-1", content="Text", score=score, rank=1)


@pytest.mark.parametrize("chunk_id", [None, "", "   "])
def test_retrieved_document_rejects_invalid_chunk_id(chunk_id):
    with pytest.raises(ValidationError):
        RetrievedDocument(doc_id="doc-1", chunk_id=chunk_id, content="Text", score=1, rank=1)


def test_retrieved_document_requires_chunk_id():
    with pytest.raises(ValidationError, match="chunk_id"):
        RetrievedDocument(doc_id="doc-1", content="Text", score=1, rank=1)


@pytest.mark.parametrize("latency", [-1, float("nan"), float("inf")])
def test_result_rejects_invalid_latency(latency):
    with pytest.raises(ValidationError):
        RAGResult(query="Query", retrieved_docs=[], rag_answer="", latency_ms=latency)


def test_eval_case_requires_labels_and_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        EvalCase(id="case-1", query="Query")
    with pytest.raises(ValidationError):
        make_case(unexpected="typo")
