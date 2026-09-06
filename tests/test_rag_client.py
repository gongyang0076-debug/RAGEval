from unittest.mock import Mock, patch

import pytest

from src.dataset import RAGResult, RetrievedDocument
from src.llm import LLMResponse, ProviderError, ProviderTrace, ProviderAttempt
from src.rag import DemoRAGClient, RAGClient
from src.rag.client import RAGRetrievalError


@pytest.fixture
def retriever():
    client = Mock()
    client.retrieve.return_value = [RetrievedDocument(doc_id="policy", chunk_id="policy_001", content="仅支持原路退款。", score=0.9, rank=1)]
    return client


@pytest.fixture
def provider():
    client = Mock()
    client.generate.return_value = LLMResponse(text="仅支持原路退款。", trace=ProviderTrace(
        model="fixture", base_url="mock://provider", timeout_seconds=1, max_attempts=3,
        retry_count=1, latency_ms=20, attempts=[ProviderAttempt(attempt=1, latency_ms=10, category="timeout"),
                                               ProviderAttempt(attempt=2, latency_ms=10)]))
    return client


def test_rag_client_is_abstract():
    with pytest.raises(TypeError):
        RAGClient()


def test_provider_independent_rag_result_context_and_retry_count(retriever, provider):
    with patch("src.rag.client.perf_counter", side_effect=[10, 10.125]):
        result = DemoRAGClient(retriever, provider).run("如何退款？", 2)
    assert isinstance(result, RAGResult)
    assert result.query == "如何退款？" and result.rag_answer == "仅支持原路退款。"
    assert result.latency_ms == pytest.approx(125)
    assert result.retry_count == 1
    assert result.generation_trace.model == "fixture"
    assert result.retrieved_docs == retriever.retrieve.return_value
    retriever.retrieve.assert_called_once_with("如何退款？", 2)
    system, user = provider.generate.call_args.args[0]
    assert "仅依据检索上下文" in system["content"] and "上下文不足" in system["content"]
    assert "不" in system["content"] and "编造" in system["content"]
    assert "policy_001" in user["content"] and "仅支持原路退款。" in user["content"] and "如何退款？" in user["content"]
    assert RAGResult.model_validate_json(result.model_dump_json()) == result


def test_retrieve_never_calls_provider(retriever, provider):
    assert DemoRAGClient(retriever, provider).retrieve("Question", 1) == retriever.retrieve.return_value
    provider.generate.assert_not_called()


@pytest.mark.parametrize("text", [None, "", "   "])
def test_empty_provider_response_retains_trace(retriever, provider, text):
    provider.generate.return_value.text = text
    with pytest.raises(ProviderError, match="no text answer") as caught:
        DemoRAGClient(retriever, provider).run("Question")
    assert caught.value.details.trace.retry_count == 1
    assert caught.value.details.category == "invalid_response"


def test_run_preserves_retrieval_failure_before_generation(retriever, provider):
    error = ValueError("empty index")
    retriever.retrieve.side_effect = error
    with pytest.raises(RAGRetrievalError) as caught:
        DemoRAGClient(retriever, provider).run("Question")
    assert caught.value.__cause__ is error
    provider.generate.assert_not_called()
