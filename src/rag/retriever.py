"""Persistent local Chroma retrieval using explicitly supplied embeddings."""

from collections.abc import Sequence

import chromadb
from chromadb.config import Settings as ChromaSettings

from config.settings import Settings
from src.dataset import CorpusDocument, DataValidationError, RetrievedDocument

from .embeddings import SentenceTransformerEmbedder


class ChromaRetriever:
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
