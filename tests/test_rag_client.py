# 文件作用：测试 RAG 接口、上下文组装、输出结果和检索异常。
# 为什么有它：确认被测系统正确连接检索器与 Provider，而不直接依赖真实服务。
from unittest.mock import Mock, patch

import pytest

from src.dataset import RAGResult, RetrievedDocument
from src.llm import LLMResponse, ProviderError, ProviderTrace, ProviderAttempt
from src.rag import DemoRAGClient, RAGClient
from src.rag.client import RAGRetrievalError


# 做什么：构造返回固定退款知识块的检索器替身。
# 为什么需要：只测试 RAG 组装逻辑而不依赖向量模型。
@pytest.fixture
def retriever():
    client = Mock()
    client.retrieve.return_value = [RetrievedDocument(doc_id="policy", chunk_id="policy_001", content="仅支持原路退款。", score=0.9, rank=1)]
    return client


# 做什么：构造带回答和一次历史重试轨迹的模型替身。
# 为什么需要：检查生成内容与轨迹是否正确进入 RAGResult。
@pytest.fixture
def provider():
    client = Mock()
    client.generate.return_value = LLMResponse(text="仅支持原路退款。", trace=ProviderTrace(
        model="fixture", base_url="mock://provider", timeout_seconds=1, max_attempts=3,
        retry_count=1, latency_ms=20, attempts=[ProviderAttempt(attempt=1, latency_ms=10, category="timeout"),
                                               ProviderAttempt(attempt=2, latency_ms=10)]))
    return client


# 做什么：验证 RAGClient 必须由具体实现提供方法。
# 为什么需要：保证 Runner 调用的对象具有完整约定。
def test_rag_client_is_abstract():
    with pytest.raises(TypeError):
        RAGClient()


# 做什么：验证 RAG 组装上下文、答案、耗时和重试信息。
# 为什么需要：检索结果与模型响应必须正确进入 RAGResult。
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


# 做什么：验证单独检索不会触发生成调用。
# 为什么需要：只测检索时不应消耗模型请求。
def test_retrieve_never_calls_provider(retriever, provider):
    assert DemoRAGClient(retriever, provider).retrieve("Question", 1) == retriever.retrieve.return_value
    provider.generate.assert_not_called()


# 做什么：验证模型返回空答案时报错并保留轨迹。
# 为什么需要：HTTP 成功不能掩盖没有业务答案。
@pytest.mark.parametrize("text", [None, "", "   "])
def test_empty_provider_response_retains_trace(retriever, provider, text):
    provider.generate.return_value.text = text
    with pytest.raises(ProviderError, match="no text answer") as caught:
        DemoRAGClient(retriever, provider).run("Question")
    assert caught.value.details.trace.retry_count == 1
    assert caught.value.details.category == "invalid_response"


# 做什么：验证检索失败保留原始原因且不继续生成。
# 为什么需要：没有成功资料结果时不能悄悄调用模型掩盖故障。
def test_run_preserves_retrieval_failure_before_generation(retriever, provider):
    error = ValueError("empty index")
    retriever.retrieve.side_effect = error
    with pytest.raises(RAGRetrievalError) as caught:
        DemoRAGClient(retriever, provider).run("Question")
    assert caught.value.__cause__ is error
    provider.generate.assert_not_called()
