"""Run with python -m src.rag {index,retrieve,run}."""

import argparse
import json

from config.settings import load_settings
from src.dataset import load_corpus
from src.llm.openai_compatible import OpenAICompatibleProvider

from .client import DemoRAGClient
from .retriever import ChromaRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Local Demo RAG (system under test)")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("index", help="Sync the ecommerce corpus into the configured collection")
    for name in ("retrieve", "run"):
        command = commands.add_parser(name)
        command.add_argument("--query", required=True)
        command.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    settings = load_settings()
    if args.command == "run":
        settings.require_llm()
    retriever = ChromaRetriever(settings)
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
