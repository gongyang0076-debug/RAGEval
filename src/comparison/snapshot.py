"""Small versioned snapshots, saved atomically independently of live evaluation."""

import json
from pathlib import Path

from pydantic import ValidationError

from src.evaluation.cache import atomic_write_json
from src.evaluation.models import EvaluationReport
from .models import BaselineSnapshot

DEFAULT_BASELINE = Path("artifacts/comparison/baseline.json")


class SnapshotError(ValueError):
    """A baseline file is unreadable or violates its contract."""


def to_snapshot(report: EvaluationReport | BaselineSnapshot) -> BaselineSnapshot:
    if isinstance(report, BaselineSnapshot):
        return BaselineSnapshot.model_validate(report)
    if not isinstance(report, EvaluationReport):
        raise TypeError("Expected EvaluationReport or BaselineSnapshot")
    r, g, s, e = report.summary.retrieval, report.summary.generation, report.summary.safety, report.summary.engineering
    return BaselineSnapshot(
        dataset_version=report.dataset_version, corpus_version=f"sha256:{report.corpus_sha256}",
        embedding_model=report.embedding_model, rag_model=report.rag_model, top_k=report.top_k,
        judge_model=report.judge_model, judge_prompt_version=report.judge_prompt_version,
        rag_mode=report.rag_mode, judge_mode=report.judge_mode,
        quality_metrics_are_synthetic=report.quality_metrics_are_synthetic, timestamp=report.timestamp,
        metrics={"recall_at_k": r.recall_at_k, "precision_at_k": r.precision_at_k, "mrr": r.mrr,
                 "correctness": g.avg_correctness, "faithfulness": g.avg_faithfulness,
                 "relevance": g.avg_relevance, "completeness": g.avg_completeness,
                 "hallucination_rate": s.hallucination_rate, "refusal_accuracy": s.refusal_accuracy,
                 "evaluation_success_rate": e.evaluation_success_rate, "avg_latency": e.average_latency},
        cohorts={"all": sorted(row.case_id for row in report.case_results),
                 "retrieval": sorted(row.case_id for row in report.case_results if row.answerable and row.retrieval_metrics is not None),
                 "judge": sorted(row.case_id for row in report.case_results if row.judge_result is not None)},
    )


def save_baseline(report: EvaluationReport | BaselineSnapshot, path: str | Path = DEFAULT_BASELINE) -> BaselineSnapshot:
    snapshot = to_snapshot(report)
    try:
        atomic_write_json(Path(path), snapshot.model_dump(mode="json"))
    except OSError as exc:
        raise SnapshotError(f"Cannot save baseline {path}: {type(exc).__name__}") from exc
    return snapshot


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate snapshot key: {key}")
        result[key] = value
    return result


def load_baseline(path: str | Path = DEFAULT_BASELINE) -> BaselineSnapshot:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_keys)
        return BaselineSnapshot.model_validate(data)
    except (OSError, UnicodeError, ValueError, ValidationError) as exc:
        raise SnapshotError(f"Cannot load baseline {path}: {exc}") from exc
