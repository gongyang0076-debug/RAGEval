"""One retrieval-only benchmark for the existing demo; no generation runner."""

import argparse
import hashlib
import json
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from config.settings import load_settings
from src.dataset import EvalCase, RetrievedDocument, load_corpus, load_dataset

from .retrieval import aggregate_retrieval_metrics, evaluate_retrieval


def benchmark_retrieval(
    cases: Sequence[EvalCase],
    retrieve: Callable[[str, int], list[RetrievedDocument]],
) -> dict:
    """Retrieve once at K=5 per answerable case; score fixed prefixes at 1/3/5."""
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("Cannot benchmark duplicate case IDs")
    # Validate labels before performing any retrieval.
    for case in cases:
        evaluate_retrieval(case, [], 1)
    retrieved = {
        case.id: retrieve(case.query, 5) if case.answerable else []
        for case in cases
    }
    by_k = {}
    for k in (1, 3, 5):
        results = [evaluate_retrieval(case, retrieved[case.id], k) for case in cases]
        failures = [
            {
                "case_id": case.id, "query": case.query, "category": case.category,
                "relevant_chunk_ids": result.relevant_chunk_ids,
                "retrieved_chunk_ids": result.retrieved_chunk_ids,
                "missing_relevant_chunk_ids": sorted(set(result.relevant_chunk_ids) - set(result.retrieved_chunk_ids)),
                "recall_at_k": result.recall_at_k, "rr": result.rr,
            }
            for case, result in zip(cases, results)
            if result.answerable and result.recall_at_k < 1.0
        ]
        by_k[str(k)] = {
            "summary": aggregate_retrieval_metrics(results, k).model_dump(),
            "failure_cases": len(failures),
            "no_hit_cases": sum(result.answerable and result.rr == 0 for result in results),
            "failures": failures,
            "results": [result.model_dump() for result in results],
        }
    return {
        "total_cases": len(cases),
        "excluded_case_ids": [case.id for case in cases if not case.answerable],
        "rr_scope": "First hit within the Top-K prefix; mrr is therefore truncated at K (MRR@K).",
        "failure_definition": "An answerable case with recall_at_k < 1, including partial relevance coverage.",
        "by_k": by_k,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval-only benchmark at K=1,3,5")
    parser.add_argument("--output", type=Path, default=Path("artifacts/sprint4/retrieval_benchmark.json"))
    args = parser.parse_args()
    settings = load_settings()
    corpus_path = settings.data_dir / "corpus" / "ecommerce_v1.json"
    dataset_path = settings.data_dir / "datasets" / "ecommerce_eval_v1.json"
    paths = (corpus_path, dataset_path)
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    corpus = load_corpus(corpus_path)
    cases = load_dataset(dataset_path, corpus)

    from src.rag import ChromaRetriever

    retriever = ChromaRetriever(settings)
    print(f"Indexed {retriever.index(corpus)} chunks", flush=True)
    report = benchmark_retrieval(cases, retriever.retrieve)
    after = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    if before != after:
        raise RuntimeError("Corpus or Dataset changed during the benchmark")
    report["metadata"] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus_chunks": len(corpus), "input_sha256": before,
        "embedding_model": settings.embedding_model,
        "embedding_device": settings.embedding_device,
        "query_instruction": settings.embedding_query_instruction,
        "normalized_embeddings": True, "chroma_collection": settings.chroma_collection,
        "distance": "cosine", "score": "1 - cosine distance",
        "versions": {name: version(name) for name in ("sentence-transformers", "transformers", "chromadb")},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for k, entry in report["by_k"].items():
        print(json.dumps({**entry["summary"], "failure_cases": entry["failure_cases"], "no_hit_cases": entry["no_hit_cases"]}, ensure_ascii=False))
        for failure in entry["failures"]:
            print(json.dumps({"k": int(k), **failure}, ensure_ascii=False))
    print(f"Saved complete results to {args.output}", flush=True)


if __name__ == "__main__":
    main()
