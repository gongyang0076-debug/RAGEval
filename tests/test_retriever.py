from dataclasses import replace
from pathlib import Path

import pytest

from config.settings import Settings
from src.dataset import CorpusDocument, DataValidationError, load_corpus
from src.rag import ChromaRetriever


class FakeEmbedder:
    """Deterministic test vectors only; never used by the demo or smoke test."""

    model_name = "test-vectors"
    query_instruction = ""

    def embed_documents(self, texts):
        vectors = [[1.0, 0.0, 0.0], [0.6, 0.8, 0.0], [-1.0, 0.0, 0.0]]
        return [vectors[index % 3] for index, _ in enumerate(texts)]

    def embed_query(self, query):
        return [1.0, 0.0, 0.0]


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path, chroma_dir=tmp_path / "chroma")


@pytest.fixture
def corpus():
    return [
        CorpusDocument(doc_id="policy", chunk_id=f"policy_{index}", content=f"Policy text {index}")
        for index in range(1, 4)
    ]


@pytest.fixture
def retriever(settings, corpus):
    retriever = ChromaRetriever(settings, embedder=FakeEmbedder())
    retriever.index(corpus)
    return retriever


def test_full_demo_corpus_written_to_real_chroma(settings):
    path = Path(__file__).resolve().parents[1] / "data/corpus/ecommerce_v1.json"
    corpus = load_corpus(path)
    retriever = ChromaRetriever(settings, embedder=FakeEmbedder())
    assert retriever.index(corpus) == 36
    stored = retriever.collection.get(ids=[corpus[0].chunk_id], include=["documents", "metadatas"])
    assert stored["ids"] == [corpus[0].chunk_id]
    assert stored["documents"] == [corpus[0].content]
    assert stored["metadatas"][0] == {"doc_id": corpus[0].doc_id, "chunk_id": corpus[0].chunk_id}


@pytest.mark.parametrize("top_k,expected_count", [(1, 1), (2, 2), (3, 3), (10, 3)])
def test_top_k_rank_ids_and_score(retriever, top_k, expected_count):
    hits = retriever.retrieve("Policy question", top_k)
    assert len(hits) == expected_count
    assert [hit.rank for hit in hits] == list(range(1, expected_count + 1))
    assert [hit.chunk_id for hit in hits] == [f"policy_{index}" for index in range(1, expected_count + 1)]
    assert all(hit.doc_id == "policy" for hit in hits)
    assert [hit.score for hit in hits] == pytest.approx([1.0, 0.6, -1.0][:expected_count], abs=1e-6)


def test_persistent_index_can_be_reopened(settings, retriever):
    reopened = ChromaRetriever(settings, embedder=FakeEmbedder())
    assert reopened.collection.count() == 3
    assert reopened.retrieve("Policy question", 1)[0].chunk_id == "policy_1"


def test_reindex_updates_content_and_removes_obsolete_chunks(retriever, corpus):
    corpus[0].content = "Updated policy"
    assert retriever.index(corpus[:2]) == 2
    assert retriever.index(corpus[:2]) == 2
    stored = retriever.collection.get(ids=["policy_1"], include=["documents"])
    assert stored["documents"] == ["Updated policy"]
    assert retriever.collection.get(ids=["policy_3"])["ids"] == []


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True])
def test_invalid_top_k(retriever, top_k):
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        retriever.retrieve("Question", top_k)


@pytest.mark.parametrize("query", ["", "  ", None])
def test_invalid_query(retriever, query):
    with pytest.raises(ValueError, match="query must be a non-empty string"):
        retriever.retrieve(query)


def test_empty_collection_requires_index(settings):
    retriever = ChromaRetriever(settings, embedder=FakeEmbedder())
    with pytest.raises(ValueError, match="index the corpus"):
        retriever.retrieve("Question")


def test_index_rejects_empty_or_duplicate_chunks(retriever, corpus):
    with pytest.raises(DataValidationError, match="empty corpus"):
        retriever.index([])
    with pytest.raises(DataValidationError, match="duplicate chunk_id"):
        retriever.index([corpus[0], corpus[0]])
    assert retriever.collection.count() == 3


def test_incompatible_embedding_configuration_rejected(settings, retriever):
    different = FakeEmbedder()
    different.model_name = "different-model"
    with pytest.raises(ValueError, match="embedding settings differ"):
        ChromaRetriever(settings, embedder=different)


def test_noncosine_collection_rejected(settings, retriever):
    other = replace(settings, chroma_collection="l2_collection")
    retriever.client.create_collection(
        name=other.chroma_collection, embedding_function=None,
        configuration={"hnsw": {"space": "l2"}}, metadata=retriever.collection.metadata,
    )
    with pytest.raises(ValueError, match="embedding settings differ"):
        ChromaRetriever(other, embedder=FakeEmbedder())
