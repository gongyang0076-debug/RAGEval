# 文件作用：用假模型检查重排序规则与公平对比，不访问网络。
# 为什么需要：排序变化必须可预测，测试不依赖大模型下载。
from unittest.mock import Mock
import pytest
from src.dataset import RetrievedDocument, EvalCase
from src.rag.reranker import CrossEncoderReranker, RerankingRetriever
from src.metrics.rerank_benchmark import compare_reranking


# 做什么：生成带身份信息的测试知识块。
# 为什么需要：检查重排序是否保留正文和编号。
def documents():
    return [RetrievedDocument(doc_id="parent", chunk_id=x, content=x, score=0.9, rank=i)
            for i, x in enumerate(["wrong", "right", "other"], 1)]


# 做什么：验证排序、分数和身份，以及不修改原始结果。
# 为什么需要：新排名必须能继续与 ground truth 对齐。
def test_order_identity_and_immutable_input():
    model = Mock(); model.predict.return_value = [-2, 5, 1]
    docs = documents()
    result = CrossEncoderReranker(model=model).rerank("question", docs, 2)
    assert [d.chunk_id for d in result] == ["right", "other"]
    assert [d.rank for d in result] == [1, 2]
    assert [d.score for d in result] == [5, 1]
    assert result[0].doc_id == "parent" and result[0].content == "right"
    assert docs[0].chunk_id == "wrong" and docs[1].rank == 2
    assert model.predict.call_args.args[0] == [("question", x.content) for x in docs]


# 做什么：验证空结果不加载模型，同分稳定，K 可大于返回量。
# 为什么需要：边界输入不应制造假结果或额外加载。
def test_empty_ties_and_short_results():
    model = Mock(); model.predict.return_value = [1, 1, 1]
    ranker = CrossEncoderReranker(model=model)
    assert ranker.rerank("q", [], 3) == []
    model.predict.assert_not_called()
    assert [d.chunk_id for d in ranker.rerank("q", documents(), 8)] == ["wrong", "right", "other"]


# 做什么：拒绝非法 K 和空问题。
# 为什么需要：错误参数必须在模型调用前报错。
@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_invalid_k(k):
    with pytest.raises(ValueError):
        CrossEncoderReranker(model=Mock()).rerank("q", documents(), k)


# 做什么：拒绝空问题和重复候选编号。
# 为什么需要：避免重复文档和无意义查询进入重排序。
def test_bad_query_and_duplicates():
    ranker = CrossEncoderReranker(model=Mock())
    with pytest.raises(ValueError, match="query"):
        ranker.rerank(" ", documents())
    with pytest.raises(ValueError, match="duplicate"):
        ranker.rerank("q", documents() + documents())


# 做什么：验证模型返回错误分数时明确失败。
# 为什么需要：避免截短结果或无效数值静默进入指标。
@pytest.mark.parametrize("scores", [[1], [1, float("nan"), 2], [1, float("inf"), 2]])
def test_bad_scores(scores):
    model = Mock(); model.predict.return_value = scores
    with pytest.raises(ValueError):
        CrossEncoderReranker(model=model).rerank("q", documents())


# 做什么：验证模型失败向上抛出，不能伪装为成功。
# 为什么需要：评测必须记录真实故障。
def test_provider_failure():
    model = Mock(); model.predict.side_effect = RuntimeError("model failed")
    with pytest.raises(RuntimeError, match="model failed"):
        CrossEncoderReranker(model=model).rerank("q", documents())


# 做什么：验证两阶段检索的候选数量与最终数量。
# 为什么需要：重排序必须先拿到更多候选，而非只改变返回条数。
def test_two_stage_retriever():
    base = Mock(); base.retrieve.return_value = documents()
    model = Mock(); model.predict.return_value = [0, 2, 1]
    retriever = RerankingRetriever(base, CrossEncoderReranker(model=model), candidate_k=10)
    assert len(retriever.retrieve("q", 2)) == 2
    base.retrieve.assert_called_once_with("q", 10)
    with pytest.raises(ValueError):
        retriever.retrieve("q", 11)
    with pytest.raises(ValueError):
        RerankingRetriever(base, Mock(), candidate_k=0)


# 做什么：检查配对实验指标和排除规则，并确保每题只初召回一次。
# 为什么需要：对比结果必须公平且不能混入不可回答题。
def test_paired_benchmark():
    cases = [EvalCase(id="a", query="q", expected_answer="right", relevant_doc_ids=["right"],
                      answerable=True, category="normal"),
             EvalCase(id="b", query="unknown", expected_answer="refuse", relevant_doc_ids=[],
                      answerable=False, category="unanswerable")]
    base = Mock(); base.retrieve.return_value = documents()
    model = Mock(); model.predict.return_value = [0, 2, 1]
    report = compare_reranking(cases, base, CrossEncoderReranker(model=model))
    base.retrieve.assert_called_once_with("q", 10)
    one = report["comparison"]["1"]["metrics"]
    assert one["mean_recall_at_k"] == {"baseline": 0, "candidate": 1, "delta": 1}
    assert report["comparison"]["3"]["metrics"]["mrr"]["delta"] == 0.5
    assert report["baseline"]["excluded_case_ids"] == ["b"]
    assert report["mean_candidate_recall"] == 1
    assert report["timing"]["mean_rerank_ms"] >= 0
    with pytest.raises(ValueError):
        compare_reranking(cases, base, Mock(), candidate_k=3)
    with pytest.raises(ValueError):
        compare_reranking([cases[0], cases[0]], base, Mock())


# 做什么：检查空数据不会除零或制造零分结论。
# 为什么需要：没有参与评测的题目时应保留空指标。
def test_empty_benchmark():
    result = compare_reranking([], Mock(), Mock())
    assert result["comparison"]["1"]["metrics"]["mrr"]["delta"] is None
    assert result["timing"]["mean_rerank_ms"] is None


# 做什么：验证命令行选择重排序或保留原始检索。
# 为什么需要：用户开关必须真正连接到检索器，不能影响默认流程。
@pytest.mark.parametrize("enabled", [False, True])
def test_cli_switch(monkeypatch, enabled, capsys):
    import src.rag.__main__ as cli
    base = Mock(); base.retrieve.return_value = documents()
    wrapped = Mock(); wrapped.retrieve.return_value = documents()[:1]
    factory = Mock(return_value=wrapped)
    monkeypatch.setattr(cli, "load_settings", Mock(return_value=Mock()))
    monkeypatch.setattr(cli, "ChromaRetriever", Mock(return_value=base))
    monkeypatch.setattr(cli, "CrossEncoderReranker", Mock())
    monkeypatch.setattr(cli, "RerankingRetriever", factory)
    monkeypatch.setattr("sys.argv", ["rag", "retrieve", "--query", "q", "--top-k", "1"] + (["--rerank"] if enabled else []))
    cli.main()
    if enabled:
        factory.assert_called_once()
        wrapped.retrieve.assert_called_once_with("q", 1)
        base.retrieve.assert_not_called()
    else:
        factory.assert_not_called()
        base.retrieve.assert_called_once_with("q", 1)
    assert "chunk_id" in capsys.readouterr().out
