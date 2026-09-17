# 文件作用：将知识块同步到 Chroma，并按问题返回带编号和排名的 Top-K 结果。
# 为什么有它：为 Demo RAG 提供可持久化、可核对标注的本地检索能力。
"""Persistent local Chroma retrieval using explicitly supplied embeddings."""

from collections.abc import Sequence

import chromadb
from chromadb.config import Settings as ChromaSettings

from config.settings import Settings
from src.dataset import CorpusDocument, DataValidationError, RetrievedDocument

from .embeddings import SentenceTransformerEmbedder


# 这个类：负责知识块索引与向量查询。
# 为什么需要：保存并找出可以对齐标注的本地资料。
class ChromaRetriever:
    # 做什么：创建或打开持久化集合，并核对向量模型和距离配置。
    # 为什么需要：避免把不同编码规则的向量混在同一个索引。
    def __init__(self, settings: Settings, embedder: SentenceTransformerEmbedder | None = None):
        self.embedder = embedder if embedder is not None else SentenceTransformerEmbedder(settings)
        self.client = chromadb.PersistentClient(
            path=str(settings.chroma_dir), settings=ChromaSettings(anonymized_telemetry=False)
        )
        metadata = {
            "embedding_model": self.embedder.model_name,
            "query_instruction": self.embedder.query_instruction,
            "normalized_embeddings": True,
        }
        self.collection = self.client.get_or_create_collection(
            name=settings.chroma_collection, embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}}, metadata=metadata,
        )
        if (
            self.collection.metadata != metadata
            or self.collection.configuration["hnsw"]["space"] != "cosine"
        ):
            raise ValueError(
                "Chroma collection embedding settings differ; use a new RAG_CHROMA_COLLECTION "
                "and index the corpus again."
            )

    # 做什么：编码并插入或更新知识块，同时删除已不在语料中的旧块。
    # 为什么需要：让向量库内容与当前 Corpus 保持一致。
    def index(self, corpus: Sequence[CorpusDocument]) -> int:
        """Sync this collection to the supplied corpus, removing obsolete chunks."""
        if not corpus:
            raise DataValidationError("Cannot index an empty corpus")
        ids = [document.chunk_id for document in corpus]
        if len(set(ids)) != len(ids):
            raise DataValidationError("Cannot index corpus with duplicate chunk_id")
        embeddings = self.embedder.embed_documents([document.content for document in corpus])
        self.collection.upsert(
            ids=ids, embeddings=embeddings,
            documents=[document.content for document in corpus],
            metadatas=[{"doc_id": document.doc_id, "chunk_id": document.chunk_id} for document in corpus],
        )
        obsolete = sorted(set(self.collection.get()["ids"]) - set(ids))
        if obsolete:
            self.collection.delete(ids=obsolete)
        return self.collection.count()

    # 做什么：校验问题和 K，搜索向量并返回保留编号的排名结果。
    # 为什么需要：向评测提供可与标注对齐且分数方向一致的检索输出。
    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedDocument]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        count = self.collection.count()
        if not count:
            raise ValueError("Chroma collection is empty; index the corpus before retrieval")
        result = self.collection.query(
            query_embeddings=[self.embedder.embed_query(query.strip())],
            n_results=min(top_k, count), include=["documents", "metadatas", "distances"],
        )
        hits = zip(result["metadatas"][0], result["documents"][0], result["distances"][0])
        return [
            RetrievedDocument(
                doc_id=metadata["doc_id"], chunk_id=metadata["chunk_id"], content=content,
                score=1.0 - float(distance), rank=rank,
            )
            for rank, (metadata, content, distance) in enumerate(hits, start=1)
        ]
