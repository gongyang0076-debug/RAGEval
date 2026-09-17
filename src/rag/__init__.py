# 文件作用：标记“被测 RAG 系统”目录为 Python 包，供其他模块导入。
# 为什么有它：让相关文件能够通过统一的包路径组织和引用。
"""Local demo RAG: the system under test, not the evaluation framework."""

from .client import DemoRAGClient, RAGClient
from .retriever import ChromaRetriever

__all__ = ["RAGClient", "DemoRAGClient", "ChromaRetriever"]
