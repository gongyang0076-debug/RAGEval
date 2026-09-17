# 文件作用：用固定向量和真实本地 Chroma 测试写入、排序、更新和参数校验。
# 为什么有它：在不下载 Embedding 模型的情况下验证向量库接入行为。
from dataclasses import replace
from pathlib import Path

import pytest

from config.settings import Settings
from src.dataset import CorpusDocument, DataValidationError, load_corpus
from src.rag import ChromaRetriever


# 这个类：用固定向量代替真实 Embedding 模型。
# 为什么需要：数据库测试能快速、稳定地得到可预测排名。
class FakeEmbedder:
    """Deterministic test vectors only; never used by the demo or smoke test."""

    model_name = "test-vectors"
    query_instruction = ""

    # 做什么：给资料依次分配几组固定单位向量。
    # 为什么需要：检索排序可预测，不需要下载真实模型。
    def embed_documents(self, texts):
        vectors = [[1.0, 0.0, 0.0], [0.6, 0.8, 0.0], [-1.0, 0.0, 0.0]]
        return [vectors[index % 3] for index, _ in enumerate(texts)]

    # 做什么：返回固定查询向量。
    # 为什么需要：保持每次测试搜索方向一致。
    def embed_query(self, query):
        return [1.0, 0.0, 0.0]


# 做什么：把向量库路径指向独立临时目录。
# 为什么需要：避免测试覆盖实际索引或互相污染。
@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path, chroma_dir=tmp_path / "chroma")


# 做什么：构造同文档下三个不同知识块。
# 为什么需要：检查块编号、排名与更新删除行为。
@pytest.fixture
def corpus():
    return [
        CorpusDocument(doc_id="policy", chunk_id=f"policy_{index}", content=f"Policy text {index}")
        for index in range(1, 4)
    ]


# 做什么：创建使用固定向量的 Chroma 检索器并预先写入资料。
# 为什么需要：测试真实数据库行为同时隔离外部模型。
@pytest.fixture
def retriever(settings, corpus):
    retriever = ChromaRetriever(settings, embedder=FakeEmbedder())
    retriever.index(corpus)
    return retriever


# 做什么：验证真实 Chroma 存入全部 Demo 知识块并保留内容与编号。
# 为什么需要：确认数据库接入没有丢失评测依据。
def test_full_demo_corpus_written_to_real_chroma(settings):
    path = Path(__file__).resolve().parents[1] / "data/corpus/ecommerce_v1.json"
    corpus = load_corpus(path)
    retriever = ChromaRetriever(settings, embedder=FakeEmbedder())
    assert retriever.index(corpus) == 36
    stored = retriever.collection.get(ids=[corpus[0].chunk_id], include=["documents", "metadatas"])
    assert stored["ids"] == [corpus[0].chunk_id]
    assert stored["documents"] == [corpus[0].content]
    assert stored["metadatas"][0] == {"doc_id": corpus[0].doc_id, "chunk_id": corpus[0].chunk_id}


# 做什么：验证不同 K 的数量、连续排名、编号和分数方向。
# 为什么需要：检索输出必须满足后续指标契约。
@pytest.mark.parametrize("top_k,expected_count", [(1, 1), (2, 2), (3, 3), (10, 3)])
def test_top_k_rank_ids_and_score(retriever, top_k, expected_count):
    hits = retriever.retrieve("Policy question", top_k)
    assert len(hits) == expected_count
    assert [hit.rank for hit in hits] == list(range(1, expected_count + 1))
    assert [hit.chunk_id for hit in hits] == [f"policy_{index}" for index in range(1, expected_count + 1)]
    assert all(hit.doc_id == "policy" for hit in hits)
    assert [hit.score for hit in hits] == pytest.approx([1.0, 0.6, -1.0][:expected_count], abs=1e-6)


# 做什么：验证重新打开集合仍能读到此前资料。
# 为什么需要：持久化索引不能只在当前对象中有效。
def test_persistent_index_can_be_reopened(settings, retriever):
    reopened = ChromaRetriever(settings, embedder=FakeEmbedder())
    assert reopened.collection.count() == 3
    assert reopened.retrieve("Policy question", 1)[0].chunk_id == "policy_1"


# 做什么：验证重建索引会更新内容并删除旧块。
# 为什么需要：向量库必须与当前知识库一致。
def test_reindex_updates_content_and_removes_obsolete_chunks(retriever, corpus):
    corpus[0].content = "Updated policy"
    assert retriever.index(corpus[:2]) == 2
    assert retriever.index(corpus[:2]) == 2
    stored = retriever.collection.get(ids=["policy_1"], include=["documents"])
    assert stored["documents"] == ["Updated policy"]
    assert retriever.collection.get(ids=["policy_3"])["ids"] == []


# 做什么：验证非法 K 被检索器拒绝。
# 为什么需要：不能向数据库发送不合理的结果数量。
@pytest.mark.parametrize("top_k", [0, -1, 1.5, True])
def test_invalid_top_k(retriever, top_k):
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        retriever.retrieve("Question", top_k)


# 做什么：验证空白或非文本问题被拒绝。
# 为什么需要：无效查询需要清楚报错。
@pytest.mark.parametrize("query", ["", "  ", None])
def test_invalid_query(retriever, query):
    with pytest.raises(ValueError, match="query must be a non-empty string"):
        retriever.retrieve(query)


# 做什么：验证未建索引时给出操作提示。
# 为什么需要：避免空库查询返回让人误解的结果。
def test_empty_collection_requires_index(settings):
    retriever = ChromaRetriever(settings, embedder=FakeEmbedder())
    with pytest.raises(ValueError, match="index the corpus"):
        retriever.retrieve("Question")


# 做什么：验证空语料和重复块在写入前被拒绝。
# 为什么需要：不能损坏或模糊已有索引内容。
def test_index_rejects_empty_or_duplicate_chunks(retriever, corpus):
    with pytest.raises(DataValidationError, match="empty corpus"):
        retriever.index([])
    with pytest.raises(DataValidationError, match="duplicate chunk_id"):
        retriever.index([corpus[0], corpus[0]])
    assert retriever.collection.count() == 3


# 做什么：验证不同 Embedding 配置不能复用同一集合。
# 为什么需要：不同编码空间的向量不能混用。
def test_incompatible_embedding_configuration_rejected(settings, retriever):
    different = FakeEmbedder()
    different.model_name = "different-model"
    with pytest.raises(ValueError, match="embedding settings differ"):
        ChromaRetriever(settings, embedder=different)


# 做什么：验证已有集合的距离规则必须符合项目约定。
# 为什么需要：score 的转换依赖余弦距离配置。
def test_noncosine_collection_rejected(settings, retriever):
    other = replace(settings, chroma_collection="l2_collection")
    retriever.client.create_collection(
        name=other.chroma_collection, embedding_function=None,
        configuration={"hnsw": {"space": "l2"}}, metadata=retriever.collection.metadata,
    )
    with pytest.raises(ValueError, match="embedding settings differ"):
        ChromaRetriever(other, embedder=FakeEmbedder())
