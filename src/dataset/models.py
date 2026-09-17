# 文件作用：规定知识块、评测题、检索结果和 RAG 输出应该包含哪些字段。
# 为什么有它：让各模块交换相同结构的数据，尽早发现字段缺失和非法值。
"""Shared data contracts; no retrieval, generation, or evaluation logic."""

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from src.llm.models import ProviderTrace


NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


# 这个类：保存一块知识资料的编号、内容和元数据。
# 为什么需要：检索与标注都需要明确的资料单位。
class CorpusDocument(BaseModel):
    """One corpus chunk; several chunks may share the same doc_id."""

    model_config = ConfigDict(extra="forbid")

    doc_id: NonEmptyString
    chunk_id: NonEmptyString
    content: NonEmptyString
    metadata: dict[str, Any] = Field(default_factory=dict)


# 这个类：保存一道题、标准答案、相关块及可回答性。
# 为什么需要：评测系统需要知道输入和预期行为。
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


# 这个类：保存一条实际检索结果的编号、内容、分数和排名。
# 为什么需要：指标需要对齐 chunk_id 并使用排名。
class RetrievedDocument(BaseModel):
    """A chunk-level hit; higher scores indicate greater relevance."""

    model_config = ConfigDict(extra="forbid")

    doc_id: NonEmptyString
    chunk_id: NonEmptyString
    content: NonEmptyString
    score: float = Field(allow_inf_nan=False)
    rank: int = Field(ge=1, strict=True)


# 这个类：保存被测系统的检索结果、答案、耗时和调用轨迹。
# 为什么需要：评测层不必依赖 RAG 的内部实现。
class RAGResult(BaseModel):
    """Output of a RAG run, including retrieval and generation latency."""

    model_config = ConfigDict(extra="forbid")

    query: NonEmptyString
    retrieved_docs: list[RetrievedDocument]
    rag_answer: str
    latency_ms: float = Field(ge=0, allow_inf_nan=False)
    retry_count: int = Field(default=0, ge=0)
    generation_trace: ProviderTrace | None = None
