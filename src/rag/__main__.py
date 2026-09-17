# 文件作用：提供从命令行运行一次 Demo RAG 的入口。
# 为什么有它：便于单独验证被测系统，而不必先跑完整评测。
"""Run with python -m src.rag {index,retrieve,run}."""

import argparse
import json

from config.settings import load_settings
from src.dataset import load_corpus
from src.llm.openai_compatible import OpenAICompatibleProvider

from .client import DemoRAGClient
from .retriever import ChromaRetriever
from .reranker import CrossEncoderReranker, RerankingRetriever


# 做什么：解析 index、retrieve、run 子命令，分别执行建索引、检索或生成。
# 为什么需要：方便单独验证被测系统，检索和生成前需已有索引。
def main() -> None:
    parser = argparse.ArgumentParser(description="Local Demo RAG (system under test)")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("index", help="Sync the ecommerce corpus into the configured collection")
    for name in ("retrieve", "run"):
        command = commands.add_parser(name)
        command.add_argument("--query", required=True)
        command.add_argument("--top-k", type=int, default=3)
        command.add_argument("--rerank", action="store_true", help="Rerank a larger candidate pool")
        command.add_argument("--candidate-k", type=int, default=10)
        command.add_argument("--reranker-model", default="BAAI/bge-reranker-base")
    args = parser.parse_args()
    if args.command != "index" and args.rerank and not 0 < args.top_k <= args.candidate_k:
        parser.error("--top-k must be positive and at most --candidate-k when reranking")
    settings = load_settings()
    if args.command == "run":
        settings.require_llm()
    retriever = ChromaRetriever(settings)
    if args.command != "index" and args.rerank:
        retriever = RerankingRetriever(
            retriever, CrossEncoderReranker(args.reranker_model, settings.embedding_device,
                                           settings.embedding_cache_dir), args.candidate_k,
        )
    if args.command == "index":
        corpus = load_corpus(settings.data_dir / "corpus" / "ecommerce_v1.json")
        print(f"Indexed {retriever.index(corpus)} chunks")
    elif args.command == "retrieve":
        documents = retriever.retrieve(args.query, args.top_k)
        print(json.dumps([doc.model_dump() for doc in documents], ensure_ascii=False, indent=2))
    else:
        result = DemoRAGClient(retriever, OpenAICompatibleProvider.from_rag_settings(settings)).run(args.query, args.top_k)
        print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
