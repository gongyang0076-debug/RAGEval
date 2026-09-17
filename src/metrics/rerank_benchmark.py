# 文件作用：在完全相同的初召回候选上比较重排序前后检索质量。
# 为什么需要：保留真实逐题结果、改善和退步案例，避免只报告好看的均值。
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from importlib.metadata import version
from time import perf_counter

from config.settings import load_settings
from src.dataset import load_corpus, load_dataset
from src.metrics.benchmark import benchmark_retrieval
from src.metrics.retrieval import evaluate_retrieval
from src.rag.reranker import CrossEncoderReranker


# 做什么：共用一次初召回，分别评估原排名与重排后的排名。
# 为什么需要：固定知识库、题目和候选集，让差异来自重排序。
def compare_reranking(cases, retriever, reranker, candidate_k=10):
    if type(candidate_k) is not int or candidate_k < 5:
        raise ValueError("candidate_k must be an integer >= 5")
    if len({c.id for c in cases}) != len(cases):
        raise ValueError("duplicate case IDs")
    for case in cases:
        evaluate_retrieval(case, [], 1)
    originals, reranked, details = [], [], []
    for case in cases:
        if not case.answerable:
            continue
        start = perf_counter()
        candidates = retriever.retrieve(case.query, candidate_k)
        retrieval_ms = (perf_counter() - start) * 1000
        start = perf_counter()
        ranked = reranker.rerank(case.query, candidates, 5)
        rerank_ms = (perf_counter() - start) * 1000
        originals.append(candidates[:5])
        reranked.append(ranked)
        details.append({"case_id": case.id, "query": case.query,
                        "relevant_chunk_ids": case.relevant_doc_ids,
                        "candidates": [d.model_dump() for d in candidates],
                        "reranked": [d.model_dump() for d in ranked],
                        "retrieval_ms": retrieval_ms, "rerank_ms": rerank_ms,
                        "candidate_recall": evaluate_retrieval(case, candidates, candidate_k).recall_at_k})
    original_iter, reranked_iter = iter(originals), iter(reranked)
    baseline = benchmark_retrieval(cases, lambda q, k: next(original_iter))
    candidate = benchmark_retrieval(cases, lambda q, k: next(reranked_iter))
    comparison = {}
    for k in ("1", "3", "5"):
        left, right = baseline["by_k"][k], candidate["by_k"][k]
        changes = []
        for a, b in zip(left["results"], right["results"]):
            if a["answerable"] and (a["rr"] != b["rr"] or a["recall_at_k"] != b["recall_at_k"]):
                changes.append({"case_id": a["case_id"], "before": a, "after": b})
        comparison[k] = {"metrics": {
            name: {"baseline": left["summary"][name], "candidate": right["summary"][name],
                   "delta": (right["summary"][name] - left["summary"][name])
                   if left["summary"][name] is not None else None}
            for name in ("mean_recall_at_k", "mean_precision_at_k", "mrr")},
            "changed_cases": changes}
    n = len(details)
    return {"mode": "REAL_RETRIEVAL_NO_LLM", "candidate_k": candidate_k,
            "baseline": baseline, "candidate": candidate, "comparison": comparison,
            "timing": {"mean_retrieval_ms": sum(x["retrieval_ms"] for x in details) / n if n else None,
                       "mean_rerank_ms": sum(x["rerank_ms"] for x in details) / n if n else None},
            "mean_candidate_recall": sum(x["candidate_recall"] for x in details) / n if n else None,
            "cases": details}


# 做什么：运行真实模型对比并保存可复核 JSON。
# 为什么需要：不用配置生成或裁判 API，也能复现实验。
def main():
    parser = argparse.ArgumentParser(description="Real vector retrieval versus cross-encoder reranking")
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-base")
    parser.add_argument("--output", type=Path, default=Path("artifacts/reranker/comparison.json"))
    args = parser.parse_args()
    if args.candidate_k < 5:
        parser.error("--candidate-k must be >= 5")
    settings = load_settings()
    paths = [settings.data_dir / "corpus/ecommerce_v1.json", settings.data_dir / "datasets/ecommerce_eval_v1.json"]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    corpus = load_corpus(paths[0])
    cases = load_dataset(paths[1], corpus)
    from src.rag.retriever import ChromaRetriever
    retriever = ChromaRetriever(settings)
    retriever.index(corpus)
    reranker = CrossEncoderReranker(args.reranker_model, settings.embedding_device, settings.embedding_cache_dir)
    print("Loading reranker (first use downloads model)...", flush=True)
    reranker.load()
    # 预热一次，耗时统计不包含下载、加载和首次推理初始化。
    warm = next(case for case in cases if case.answerable)
    reranker.rerank(warm.query, retriever.retrieve(warm.query, args.candidate_k), 5)
    print("Running paired retrieval benchmark...", flush=True)
    report = compare_reranking(cases, retriever, reranker, args.candidate_k)
    if hashes != {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}:
        raise RuntimeError("Input data changed during benchmark")
    report["metadata"] = {"timestamp": datetime.now(timezone.utc).isoformat(), "input_sha256": hashes,
                          "embedding_model": settings.embedding_model, "reranker_model": args.reranker_model,
                          "device": settings.embedding_device, "query_instruction": settings.embedding_query_instruction,
                          "max_pair_tokens": 512, "batch_size": 8,
                          "versions": {name: version(name) for name in ("sentence-transformers", "transformers", "torch", "chromadb")},
                          "baseline_score": "1 - cosine distance", "candidate_score": "raw cross-encoder logit; higher is better",
                          "timing_scope": "Warm inference only; reranked total = retrieval + rerank. No generation or Judge."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v["metrics"] for k, v in report["comparison"].items()}, indent=2), flush=True)
    print(json.dumps(report["timing"]), flush=True)
    print(f"Saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
