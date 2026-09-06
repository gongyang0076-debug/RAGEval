import json
from datetime import datetime, timezone

import httpx
import pytest
from pydantic import ValidationError

from src.comparison import (
    BaselineSnapshot, GateConfig, GateConfigurationError, SnapshotError, apply_gate,
    compare_reports, load_baseline, load_gate_config, save_baseline,
)
from src.comparison.demo import run_demo
from src.evaluation.models import EvaluationReport


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Comparison tests must not call APIs")
    monkeypatch.setattr(httpx.Client, "send", forbidden)
    monkeypatch.setattr(httpx.AsyncClient, "send", forbidden)


@pytest.fixture
def baseline():
    return BaselineSnapshot(
        dataset_version="dataset-v1", corpus_version="sha256:corpus-v1", embedding_model="fixture-embedding",
        rag_model="fixture-rag", top_k=3, judge_model="fixture-judge", judge_prompt_version="judge_v2",
        rag_mode="LIVE", judge_mode="LIVE", quality_metrics_are_synthetic=False,
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        cohorts={"all": ["a", "u"], "retrieval": ["a"], "judge": ["a", "u"]},
        metrics={"recall_at_k": 0.91, "precision_at_k": 0.6, "mrr": 0.9,
                 "correctness": 4.3, "faithfulness": 4.6, "relevance": 4.5, "completeness": 4.2,
                 "hallucination_rate": 0.02, "refusal_accuracy": 0.95,
                 "evaluation_success_rate": 0.99, "avg_latency": 100},
    )


def changed(baseline, **metrics):
    candidate = baseline.model_copy(deep=True)
    candidate.metrics.update(metrics)
    candidate.rag_model = "fixture-candidate"
    return candidate


def test_default_baseline_save_load_roundtrip(baseline, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    saved = save_baseline(baseline)
    assert load_baseline() == saved == baseline
    payload = json.loads((tmp_path / "artifacts/comparison/baseline.json").read_text(encoding="utf-8"))
    assert payload["corpus_version"] == "sha256:corpus-v1"
    assert payload["timestamp"].endswith("Z")
    assert payload["metrics"]["mrr"] == 0.9


@pytest.mark.parametrize("content", ["bad json", "{}", '{"schema_version":1,"schema_version":1}'])
def test_bad_snapshot_file_is_clear(tmp_path, content):
    path = tmp_path / "baseline.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(SnapshotError, match="Cannot load baseline"):
        load_baseline(path)


def test_nonexistent_snapshot_is_clear(tmp_path):
    with pytest.raises(SnapshotError):
        load_baseline(tmp_path / "missing.json")


def test_missing_metrics_object_is_a_contract_error(baseline, tmp_path):
    values = baseline.model_dump(mode="json")
    del values["metrics"]
    path = tmp_path / "missing-metrics.json"
    path.write_text(json.dumps(values), encoding="utf-8")
    with pytest.raises(SnapshotError, match="metrics"):
        load_baseline(path)


@pytest.mark.parametrize("name,after,delta,improvement,regression", [
    ("recall_at_k", 0.82, -0.09, False, True), ("mrr", 0.95, 0.05, True, False),
    ("correctness", 4.5, 0.2, True, False), ("faithfulness", 4.5, -0.1, False, True),
    ("relevance", 4.5, 0, False, False), ("completeness", 4.0, -0.2, False, True),
    ("hallucination_rate", 0.01, -0.01, True, False), ("refusal_accuracy", 0.9, -0.05, False, True),
    ("evaluation_success_rate", 1.0, 0.01, True, False), ("avg_latency", 125, 25, False, True),
    ("precision_at_k", 0.7, 0.1, True, False),
])
def test_metric_deltas_and_direction(baseline, name, after, delta, improvement, regression):
    comparison = compare_reports(baseline, changed(baseline, **{name: after}))
    metric = comparison.metrics[name]
    assert metric.delta == pytest.approx(delta)
    assert metric.improvement is improvement and metric.regression is regression
    assert metric.comparable
    assert comparison.configuration_changes["rag_model"] == ["fixture-rag", "fixture-candidate"]


def test_gate_pass_including_boundary(baseline):
    config = load_gate_config("config/regression_gate.yaml")
    result = apply_gate(compare_reports(baseline, changed(baseline, recall_at_k=0.85, avg_latency=120)), config)
    assert result.status == "PASS" and result.reasons == [] and not result.simulation


def test_gate_fail_lists_both_absolute_drop_and_relative_latency(baseline):
    comparison = compare_reports(baseline, changed(baseline, recall_at_k=0.82, avg_latency=125))
    result = apply_gate(comparison, load_gate_config("config/regression_gate.yaml"))
    assert result.status == "FAIL" and len(result.reasons) == 2
    assert any("0.91 -> 0.82" in reason for reason in result.reasons)
    assert any("25.00%" in reason and "20.00%" in reason for reason in result.reasons)


@pytest.mark.parametrize("missing", ["baseline", "candidate"])
def test_missing_metric_has_no_delta_and_gate_fails(baseline, missing):
    candidate = changed(baseline)
    del (baseline if missing == "baseline" else candidate).metrics["mrr"]
    comparison = compare_reports(baseline, candidate)
    assert comparison.metrics["mrr"].delta is None
    assert comparison.metrics["mrr"].regression is None
    gate = apply_gate(comparison, GateConfig(thresholds={"mrr": {"min": 0.8}}))
    assert gate.status == "FAIL" and "missing" in gate.reasons[0]


def test_null_metric_does_not_become_zero(baseline):
    comparison = compare_reports(baseline, changed(baseline, hallucination_rate=None))
    assert comparison.metrics["hallucination_rate"].candidate_value is None
    result = apply_gate(comparison, GateConfig(thresholds={"hallucination_rate": {"max": 0.1}}))
    assert result.status == "FAIL"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True, "0.9", 1.1])
def test_invalid_metric_values_are_rejected(baseline, value):
    with pytest.raises(ValidationError):
        compare_reports(baseline, changed(baseline, recall_at_k=value))


@pytest.mark.parametrize("yaml_text", [
    "thresholds: {}", "thresholds: [", "thresholds:\n  typo_metric:\n    min: 0.8",
    "thresholds:\n  mrr:\n    min: 0.9\n    max: 0.8",
    "thresholds:\n  mrr:\n    min: true", "thresholds:\n  mrr:\n    min: .nan",
    "thresholds:\n  mrr:\n    min: 1.1", "thresholds:\n  mrr: {}",
    "thresholds:\n  mrr:\n    max_increase: 0.2", "thresholds:\n  latency:\n    max_increase: -0.2",
    "thresholds:\n  mrr:\n    min: 0.8\n    min: 0.1",
    "thresholds:\n  recall_at_k:\n    min: 0.8\n  recall_at_3:\n    min: 0.7",
    "thresholds:\n  mrr:\n    maximum: 0.8", "thresholds:\n  mrr:\n    min: '0.8'",
    "allow_synthetic: 'false'\nthresholds:\n  mrr:\n    min: 0.8",
])
def test_invalid_threshold_configs_fail_closed(tmp_path, yaml_text):
    path = tmp_path / "gate.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    with pytest.raises(GateConfigurationError, match="Invalid gate configuration"):
        load_gate_config(path)


@pytest.mark.parametrize("after,expected", [(0, "PASS"), (1, "FAIL")])
def test_relative_latency_zero_baseline(baseline, after, expected):
    baseline.metrics["avg_latency"] = 0
    result = apply_gate(compare_reports(baseline, changed(baseline, avg_latency=after)),
                        GateConfig(thresholds={"latency": {"max_increase": 0.2}}))
    assert result.status == expected
    assert "Infinity" not in result.model_dump_json()


def test_top_k_change_is_visible_and_specific_k_rule_is_not_misapplied(baseline):
    candidate = changed(baseline)
    candidate.top_k = 1
    comparison = compare_reports(baseline, candidate)
    assert any("different K" in warning for warning in comparison.warnings)
    assert comparison.configuration_changes["top_k"] == [3, 1]
    assert apply_gate(comparison, GateConfig(thresholds={"recall_at_3": {"min": 0.85}})).status == "FAIL"
    assert apply_gate(comparison, GateConfig(thresholds={"recall_at_k": {"min": 0.85}})).status == "PASS"


@pytest.mark.parametrize("field,value", [("dataset_version", "different"), ("corpus_version", "different")])
def test_incompatible_data_versions_cannot_pass(baseline, field, value):
    candidate = changed(baseline)
    setattr(candidate, field, value)
    result = apply_gate(compare_reports(baseline, candidate), GateConfig(thresholds={"mrr": {"min": 0.8}}))
    assert result.status == "FAIL" and "version differs" in result.reasons[0]


def test_different_evaluated_cases_cannot_masquerade_as_improvement(baseline):
    candidate = changed(baseline, correctness=5)
    candidate.cohorts["judge"] = ["a"]
    comparison = compare_reports(baseline, candidate)
    assert comparison.metrics["correctness"].delta == pytest.approx(0.7)
    assert comparison.metrics["correctness"].improvement is None
    assert apply_gate(comparison, GateConfig(thresholds={"correctness": {"min": 4}})).status == "FAIL"


def test_changed_judge_protocol_only_invalidates_judged_metrics(baseline):
    candidate = changed(baseline)
    candidate.judge_prompt_version = "judge_v3"
    comparison = compare_reports(baseline, candidate)
    assert not comparison.metrics["faithfulness"].comparable
    assert comparison.metrics["mrr"].comparable


def test_empty_cohort_cannot_pass_with_a_supplied_score(baseline):
    baseline.cohorts["judge"] = []
    comparison = compare_reports(baseline, changed(baseline, correctness=5))
    result = apply_gate(comparison, GateConfig(thresholds={"correctness": {"min": 4}}))
    assert result.status == "FAIL" and "No evaluated judge cases" in result.reasons[0]


def test_mock_judge_zero_hallucination_cannot_pass_live_gate(baseline):
    baseline.judge_mode = "MOCK"
    baseline.quality_metrics_are_synthetic = True
    candidate = changed(baseline, hallucination_rate=0)
    comparison = compare_reports(baseline, candidate)
    config = GateConfig(thresholds={"hallucination_rate": {"max": 0.1}})
    assert apply_gate(comparison, config).status == "FAIL"
    config.allow_synthetic = True
    result = apply_gate(comparison, config)
    assert result.status == "PASS" and result.simulation
    # Real retrieval remains usable even when the Judge is a mock.
    assert not comparison.metrics["mrr"].synthetic


def test_demo_generates_two_reports_snapshot_comparison_and_gate_examples(tmp_path):
    comparison, passed, failed = run_demo(tmp_path)
    assert passed.status == "PASS" and failed.status == "FAIL"
    assert passed.simulation and failed.simulation
    assert comparison.baseline.top_k == 3 and comparison.candidate.top_k == 1
    assert comparison.metrics["recall_at_k"].regression
    assert comparison.metrics["precision_at_k"].improvement
    assert comparison.metrics["avg_latency"].improvement
    for name in ["baseline.json", "candidate.json"]:
        report = EvaluationReport.model_validate_json((tmp_path / name).read_text(encoding="utf-8"))
        assert len(report.case_results) == 60
        assert report.rag_mode == report.judge_mode == "MOCK"
    assert load_baseline(tmp_path / "baseline_snapshot.json") == comparison.baseline
    assert json.loads((tmp_path / "gate_fail.json").read_text(encoding="utf-8"))["status"] == "FAIL"
