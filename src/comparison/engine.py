"""Direction-aware deltas with explicit provenance and cohort compatibility."""

from math import isclose

from .models import METRICS, MetricComparison, ReportComparison
from .snapshot import to_snapshot


def compare_reports(baseline, candidate) -> ReportComparison:
    baseline, candidate = to_snapshot(baseline), to_snapshot(candidate)
    fields = ("dataset_version", "corpus_version", "embedding_model", "rag_model", "top_k",
              "judge_model", "judge_prompt_version", "rag_mode", "judge_mode")
    changes = {name: [getattr(baseline, name), getattr(candidate, name)] for name in fields
               if getattr(baseline, name) != getattr(candidate, name)}
    warnings = [f"{name}: {values[0]} -> {values[1]}" for name, values in changes.items()]
    if baseline.top_k != candidate.top_k:
        warnings.append("Retrieval deltas compare different K settings; they are not fixed-K improvements.")
    results = {}
    for name, (direction, _, cohort) in METRICS.items():
        before, after = baseline.metrics.get(name), candidate.metrics.get(name)
        reasons = []
        if before is None or after is None:
            reasons.append("Metric missing or null; no zero imputation")
        if "dataset_version" in changes or "corpus_version" in changes:
            reasons.append("Dataset or corpus version differs")
        if ("all" not in baseline.cohorts or "all" not in candidate.cohorts
                or set(baseline.cohorts["all"]) != set(candidate.cohorts["all"])):
            reasons.append("Dataset case IDs differ or are unavailable")
        if (cohort not in baseline.cohorts or cohort not in candidate.cohorts
                or set(baseline.cohorts[cohort]) != set(candidate.cohorts[cohort])):
            reasons.append(f"Evaluated {cohort} case IDs differ or are unavailable")
        if not baseline.cohorts.get(cohort) or not candidate.cohorts.get(cohort):
            reasons.append(f"No evaluated {cohort} cases")
        if "rag_mode" in changes or (cohort != "retrieval" and "judge_mode" in changes):
            reasons.append("LIVE/MOCK measurement modes differ")
        if cohort == "judge" and ("judge_model" in changes or "judge_prompt_version" in changes):
            reasons.append("Judge model or rubric version differs")
        synthetic = any(
            s.rag_mode == "MOCK" or (cohort != "retrieval" and s.quality_metrics_are_synthetic)
            for s in (baseline, candidate)
        )
        delta = after - before if before is not None and after is not None else None
        signed = delta * (1 if direction == "higher" else -1) if delta is not None else None
        unchanged = signed is not None and isclose(signed, 0, abs_tol=1e-12)
        results[name] = MetricComparison(
            metric=name, direction=direction, baseline_value=before, candidate_value=after, delta=delta,
            improvement=None if reasons else signed > 0 and not unchanged,
            regression=None if reasons else signed < 0 and not unchanged,
            comparable=not reasons, synthetic=synthetic, reason="; ".join(reasons) if reasons else None,
        )
    return ReportComparison(baseline=baseline, candidate=candidate, metrics=results,
                            configuration_changes=changes, warnings=warnings)
