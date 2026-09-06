"""Load JSON arrays and validate corpus-to-dataset references."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, TypeAdapter, ValidationError

from .models import CorpusDocument, EvalCase


class DataValidationError(ValueError):
    """A data file could not be read or violates the dataset contract."""


ModelT = TypeVar("ModelT", bound=BaseModel)


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


def _unique_chunk_ids(corpus: Sequence[CorpusDocument], source: str) -> set[str]:
    chunk_ids: set[str] = set()
    for document in corpus:
        if document.chunk_id in chunk_ids:
            raise DataValidationError(f"{source}: duplicate chunk_id {document.chunk_id!r}")
        chunk_ids.add(document.chunk_id)
    return chunk_ids


def load_corpus(path: str | Path) -> list[CorpusDocument]:
    """Load corpus chunks; doc_id may repeat, but chunk_id must be unique."""
    corpus = _load_records(path, CorpusDocument)
    _unique_chunk_ids(corpus, str(path))
    return corpus


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
