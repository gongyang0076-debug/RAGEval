# 文件作用：标记“数据模型与加载”目录为 Python 包，供其他模块导入。
# 为什么有它：让相关文件能够通过统一的包路径组织和引用。
"""Core dataset and evaluation data contracts."""

from .models import CorpusDocument, EvalCase, RAGResult, RetrievedDocument
from .loader import DataValidationError, load_corpus, load_dataset

__all__ = [
    "CorpusDocument", "EvalCase", "RetrievedDocument", "RAGResult",
    "DataValidationError", "load_corpus", "load_dataset",
]
