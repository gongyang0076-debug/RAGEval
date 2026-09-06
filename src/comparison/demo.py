"""Offline simulation: controlled ranking fixtures, not a new RAG benchmark."""

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from src.dataset import RAGResult, RetrievedDocument, load_corpus, load_dataset
from src.evaluation.aggregation import aggregate_results
from src.evaluation.cache import atomic_write_json
from src.evaluation.mock_judge import MockJudge
from src.evaluation.models import CaseEvaluationResult, EvaluationReport
from src.judge import JudgeInput
from src.metrics import evaluate_retrieval

from .engine import compare_reports
from .gate import apply_gate, load_gate_config
from .snapshot import load_baseline, save_baseline


def simulated_report(corpus_path: Path, dataset_path: Path, top_k: int) -> EvaluationReport:
    corpus = load_corpus(corpus_path)
    cases = load_dataset(dataset_path, corpus)
    by_id = {doc.chunk_id: doc for doc in corpus}
    judge = MockJudge()
    rows = []
    answerable_index = 0
    for case in cases:
        relevant = [by_id[key] for key in dict.fromkeys(case.relevant_doc_ids)]
        others = [doc for doc in corpus if doc.chunk_id not in case.relevant_doc_ids]
        # Ground truth is deliberately used ONLY to construct controlled mock rankings.
        # Every fourth answerable case has a distractor at rank 1; this is not retrieval quality evidence.
        if case.answerable:
            ranking = [others[0], *relevant, *others[1:]] if answerable_index % 4 == 0 else [*relevant, *others]
            answerable_index += 1
        else:
            ranking = others
        retrieved = [RetrievedDocument(doc_id=doc.doc_id, chunk_id=doc.chunk_id, content=doc.content,
                                       score=1 / rank, rank=rank)
                     for rank, doc in enumerate(ranking[:top_k], 1)]
        rag = RAGResult(query=case.query, retrieved_docs=retrieved,
                        rag_answer="SIMULATION：知识库信息不足，无法确定。", latency_ms=8 + 4 * top_k)
        verdict = judge.judge(JudgeInput(case_id=case.id, query=case.query, expected_answer=case.expected_answer,
                                        retrieved_context="\n".join(doc.content for doc in retrieved),
                                        rag_answer=rag.rag_answer, answerable=case.answerable))
        verdict.metadata.latency_ms = 5  # Synthetic timing for reproducible gate examples.
        rows.append(CaseEvaluationResult(
            case_id=case.id, query=case.query, category=case.category, answerable=case.answerable,
            rag_result=rag, retrieval_metrics=evaluate_retrieval(case, retrieved, top_k) if case.answerable else None,
            judge_result=verdict, refusal_correct=True if not case.answerable else None,
            latency_ms=rag.latency_ms + verdict.metadata.latency_ms,
        ))
    dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    return EvaluationReport(
        timestamp=datetime.now(timezone.utc), dataset_path=str(dataset_path),
        dataset_version=f"{dataset_path.stem}@sha256:{dataset_hash}", dataset_sha256=dataset_hash,
        corpus_sha256=hashlib.sha256(corpus_path.read_bytes()).hexdigest(), top_k=top_k,
        embedding_model="mock-fixed-ranking", rag_model="mock-rag-sprint8",
        judge_model=judge.settings.model, judge_prompt_version=judge.prompt_version,
        rag_mode="MOCK", judge_mode="MOCK", quality_metrics_are_synthetic=True,
        run_latency_ms=sum(row.latency_ms for row in rows), summary=aggregate_results(rows), case_results=rows,
        execution_config={"simulation": True, "latency_is_synthetic": True,
                          "ranking_fixture": "Every fourth answerable case has its first relevant chunk at rank 2"},
    )


def run_demo(output: str | Path = "artifacts/sprint8"):
    output = Path(output)
    corpus_path, dataset_path = Path("data/corpus/ecommerce_v1.json"), Path("data/datasets/ecommerce_eval_v1.json")
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (corpus_path, dataset_path)}
    baseline = simulated_report(corpus_path, dataset_path, 3)
    candidate = simulated_report(corpus_path, dataset_path, 1)
    atomic_write_json(output / "baseline.json", baseline.model_dump(mode="json"))
    atomic_write_json(output / "candidate.json", candidate.model_dump(mode="json"))
    save_baseline(baseline, output / "baseline_snapshot.json")
    comparison = compare_reports(load_baseline(output / "baseline_snapshot.json"), candidate)
    config = load_gate_config("config/regression_gate_demo.yaml")
    passed = apply_gate(compare_reports(baseline, baseline), config)
    failed = apply_gate(comparison, config)
    atomic_write_json(output / "comparison.json", comparison.model_dump(mode="json"))
    atomic_write_json(output / "gate_pass.json", passed.model_dump(mode="json"))
    atomic_write_json(output / "gate_fail.json", failed.model_dump(mode="json"))
    if any(hashlib.sha256(p.read_bytes()).hexdigest() != digest for p, digest in hashes.items()):
        raise RuntimeError("Corpus or Dataset changed during simulation")
    return comparison, passed, failed


def main():
    parser = argparse.ArgumentParser(description="SIMULATION ONLY: compare controlled top_k=3 and top_k=1 reports")
    parser.add_argument("--output", type=Path, default=Path("artifacts/sprint8"))
    args = parser.parse_args()
    comparison, passed, failed = run_demo(args.output)
    print("SIMULATION ONLY: RAG_MODE=MOCK, JUDGE_MODE=MOCK; timings and judgments are synthetic")
    for metric in comparison.metrics.values():
        print(f"{metric.metric}: {metric.baseline_value} -> {metric.candidate_value}; delta={metric.delta}")
    print(f"Baseline self-check: {passed.status}\nCandidate gate: {failed.status}")
    for reason in failed.reasons:
        print(reason)
    print(f"Saved to {args.output}")
    if passed.status != "PASS" or failed.status != "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
