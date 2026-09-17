# 文件作用：从 JSON 文件读取知识库与试卷，并检查编号、结构和相关块引用。
# 为什么有它：避免错误标注或重复数据进入检索和评测流程。
"""Load JSON arrays and validate corpus-to-dataset references."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, TypeAdapter, ValidationError

from .models import CorpusDocument, EvalCase


# 这个类：表示语料或试卷文件不符合加载要求。
# 为什么需要：在评测前明确指出数据问题。
class DataValidationError(ValueError):
    """A data file could not be read or violates the dataset contract."""


ModelT = TypeVar("ModelT", bound=BaseModel)


# 做什么：读取 UTF-8 JSON 列表并转换成指定数据模型。
# 为什么需要：统一处理文件、JSON 和字段错误，报出可定位的信息。
def _load_records(path: str | Path, model: type[ModelT]) -> list[ModelT]:
    path = Path(path)
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise DataValidationError(f"{path}: cannot read UTF-8 data: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DataValidationError(
            f"{path}: invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    try:
        return TypeAdapter(list[model]).validate_python(data)
    except ValidationError as exc:
        raise DataValidationError(f"{path}: invalid {model.__name__} structure: {exc}") from exc


# 做什么：检查全局知识块编号并返回编号集合。
# 为什么需要：防止重复块或不确定引用影响检索标注。
def _unique_chunk_ids(corpus: Sequence[CorpusDocument], source: str) -> set[str]:
    chunk_ids: set[str] = set()
    for document in corpus:
        if document.chunk_id in chunk_ids:
            raise DataValidationError(f"{source}: duplicate chunk_id {document.chunk_id!r}")
        chunk_ids.add(document.chunk_id)
    return chunk_ids


# 做什么：加载知识块并检查 chunk_id 唯一性。
# 为什么需要：给检索系统提供合法且可区分的资料。
def load_corpus(path: str | Path) -> list[CorpusDocument]:
    """Load corpus chunks; doc_id may repeat, but chunk_id must be unique."""
    corpus = _load_records(path, CorpusDocument)
    _unique_chunk_ids(corpus, str(path))
    return corpus


# 做什么：加载试卷并检查题号、可回答性和相关块引用。
# 为什么需要：确保每道题的标注与实际知识库一致。
def load_dataset(path: str | Path, corpus: Sequence[CorpusDocument]) -> list[EvalCase]:
    """Load cases and resolve relevant_doc_ids against corpus chunk_id values."""
    cases = _load_records(path, EvalCase)
    chunk_ids = _unique_chunk_ids(corpus, f"{path}: supplied corpus")
    case_ids: set[str] = set()
    for case in cases:
        context = f"{path}: case {case.id!r}"
        if case.id in case_ids:
            raise DataValidationError(f"{context}: duplicate case id")
        case_ids.add(case.id)
        if case.category == "unanswerable" and case.answerable:
            raise DataValidationError(f"{context}: unanswerable category requires answerable=false")
        if not case.answerable and case.relevant_doc_ids:
            raise DataValidationError(f"{context}: answerable=false requires relevant_doc_ids=[]")
        if case.answerable:
            if not case.relevant_doc_ids:
                raise DataValidationError(f"{context}: answerable=true requires relevant_doc_ids")
            missing = sorted(set(case.relevant_doc_ids) - chunk_ids)
            if missing:
                raise DataValidationError(f"{context}: unknown relevant_doc_ids (chunk_id): {missing}")
    return cases
