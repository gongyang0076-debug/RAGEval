"""Core dataset and evaluation data contracts."""

from .models import CorpusDocument, EvalCase, RAGResult, RetrievedDocument
from .loader import DataValidationError, load_corpus, load_dataset

__all__ = [
    "CorpusDocument", "EvalCase", "RetrievedDocument", "RAGResult",
    "DataValidationError", "load_corpus", "load_dataset",
]
