"""Shared data contracts; no retrieval, generation, or evaluation logic."""

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from src.llm.models import ProviderTrace


NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class CorpusDocument(BaseModel):
    """One corpus chunk; several chunks may share the same doc_id."""

    model_config = ConfigDict(extra="forbid")

    doc_id: NonEmptyString
    chunk_id: NonEmptyString
    content: NonEmptyString
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalCase(BaseModel):
    """One labeled query; relevant_doc_ids references corpus chunk_id values."""

    model_config = ConfigDict(extra="forbid")

    id: NonEmptyString
    query: NonEmptyString
    expected_answer: str
    relevant_doc_ids: list[NonEmptyString] = Field(
        description="Relevant corpus chunk_id values (legacy field name retained)."
    )
    answerable: bool
    category: NonEmptyString
    variant_group: NonEmptyString | None = None


class RetrievedDocument(BaseModel):
    """A chunk-level hit; higher scores indicate greater relevance."""

    model_config = ConfigDict(extra="forbid")

    doc_id: NonEmptyString
    chunk_id: NonEmptyString
    content: NonEmptyString
    score: float = Field(allow_inf_nan=False)
    rank: int = Field(ge=1, strict=True)


class RAGResult(BaseModel):
    """Output of a RAG run, including retrieval and generation latency."""

    model_config = ConfigDict(extra="forbid")

    query: NonEmptyString
    retrieved_docs: list[RetrievedDocument]
    rag_answer: str
    latency_ms: float = Field(ge=0, allow_inf_nan=False)
    retry_count: int = Field(default=0, ge=0)
    generation_trace: ProviderTrace | None = None
