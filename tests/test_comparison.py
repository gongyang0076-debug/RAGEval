# 文件作用：测试基线读写、指标可比性、差值方向和回归门禁。
# 为什么有它：避免错误样本、缺失指标或 MOCK 分数被判成真实改进。
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


# 做什么：拦截同步与异步 HTTP 发送。
# 为什么需要：版本比较测试不应该访问模型或网络。
@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    # 做什么：一旦发生真实网络发送就让测试失败。
    # 为什么需要：及时发现意外绕过替身的调用。
    def forbidden(*args, **kwargs):
        raise AssertionError("Comparison tests must not call APIs")
    monkeypatch.setattr(httpx.Client, "send", forbidden)
    monkeypatch.setattr(httpx.AsyncClient, "send", forbidden)


# 做什么：构造固定指标与样本集合的合法基线。
# 为什么需要：每个比较测试都有可控的历史参照。
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


# 做什么：复制基线并修改指定指标与候选模型名。
# 为什么需要：只改变当前测试关心的因素且不污染原基线。
def changed(baseline, **metrics):
    candidate = baseline.model_copy(deep=True)
    candidate.metrics.update(metrics)
    candidate.rag_model = "fixture-candidate"
    return candidate


# 做什么：验证基线按默认路径保存后能完整读回。
# 为什么需要：防止快照在持久化时丢失版本或指标。
def test_default_baseline_save_load_roundtrip(baseline, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    saved = save_baseline(baseline)
    assert load_baseline() == saved == baseline
    payload = json.loads((tmp_path / "artifacts/comparison/baseline.json").read_text(encoding="utf-8"))
    assert payload["corpus_version"] == "sha256:corpus-v1"
    assert payload["timestamp"].endswith("Z")
    assert payload["metrics"]["mrr"] == 0.9


# 做什么：验证损坏或不合规快照给出明确错误。
# 为什么需要：坏基线不能进入版本比较。
@pytest.mark.parametrize("content", ["bad json", "{}", '{"schema_version":1,"schema_version":1}'])
def test_bad_snapshot_file_is_clear(tmp_path, content):
    path = tmp_path / "baseline.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(SnapshotError, match="Cannot load baseline"):
        load_baseline(path)


# 做什么：验证基线文件不存在时明确报错。
# 为什么需要：避免缺少历史参照被当作空基线。
def test_nonexistent_snapshot_is_clear(tmp_path):
    with pytest.raises(SnapshotError):
        load_baseline(tmp_path / "missing.json")


# 做什么：验证快照必须包含指标对象。
# 为什么需要：没有成绩的快照不能伪装成合法比较输入。
def test_missing_metrics_object_is_a_contract_error(baseline, tmp_path):
    values = baseline.model_dump(mode="json")
    del values["metrics"]
    path = tmp_path / "missing-metrics.json"
    path.write_text(json.dumps(values), encoding="utf-8")
    with pytest.raises(SnapshotError, match="metrics"):
        load_baseline(path)


# 做什么：验证候选减基线的差值和指标改善方向。
# 为什么需要：延迟降低与分数降低的含义不同。
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


# 做什么：验证满足规则及恰好到达边界时通过。
# 为什么需要：防止合格版本因边界判断错误被拒绝。
def test_gate_pass_including_boundary(baseline):
    config = load_gate_config("config/regression_gate.yaml")
    result = apply_gate(compare_reports(baseline, changed(baseline, recall_at_k=0.85, avg_latency=120)), config)
    assert result.status == "PASS" and result.reasons == [] and not result.simulation


# 做什么：验证分数低于下限和延迟增长都被列为失败原因。
# 为什么需要：不能只显示第一个失败而隐藏其他退步。
def test_gate_fail_lists_both_absolute_drop_and_relative_latency(baseline):
    comparison = compare_reports(baseline, changed(baseline, recall_at_k=0.82, avg_latency=125))
    result = apply_gate(comparison, load_gate_config("config/regression_gate.yaml"))
    assert result.status == "FAIL" and len(result.reasons) == 2
    assert any("0.91 -> 0.82" in reason for reason in result.reasons)
    assert any("25.00%" in reason and "20.00%" in reason for reason in result.reasons)


# 做什么：验证缺指标时没有虚构差值且门禁失败。
# 为什么需要：缺失数据不能被解释成质量达标。
@pytest.mark.parametrize("missing", ["baseline", "candidate"])
def test_missing_metric_has_no_delta_and_gate_fails(baseline, missing):
    candidate = changed(baseline)
    del (baseline if missing == "baseline" else candidate).metrics["mrr"]
    comparison = compare_reports(baseline, candidate)
    assert comparison.metrics["mrr"].delta is None
    assert comparison.metrics["mrr"].regression is None
    gate = apply_gate(comparison, GateConfig(thresholds={"mrr": {"min": 0.8}}))
    assert gate.status == "FAIL" and "missing" in gate.reasons[0]


# 做什么：验证空指标不被自动转换为零。
# 为什么需要：区分没有测量与测量结果很差。
def test_null_metric_does_not_become_zero(baseline):
    comparison = compare_reports(baseline, changed(baseline, hallucination_rate=None))
    assert comparison.metrics["hallucination_rate"].candidate_value is None
    result = apply_gate(comparison, GateConfig(thresholds={"hallucination_rate": {"max": 0.1}}))
    assert result.status == "FAIL"


# 做什么：验证快照拒绝越界或非法指标值。
# 为什么需要：避免错误成绩参与比较。
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True, "0.9", 1.1])
def test_invalid_metric_values_are_rejected(baseline, value):
    with pytest.raises(ValidationError):
        compare_reports(baseline, changed(baseline, recall_at_k=value))


# 做什么：验证非法或重复门禁配置明确失败。
# 为什么需要：规则损坏不能导致默认放行。
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


# 做什么：验证零基线下的相对延迟规则。
# 为什么需要：避免除零或对没有定义的增幅给出假通过。
@pytest.mark.parametrize("after,expected", [(0, "PASS"), (1, "FAIL")])
def test_relative_latency_zero_baseline(baseline, after, expected):
    baseline.metrics["avg_latency"] = 0
    result = apply_gate(compare_reports(baseline, changed(baseline, avg_latency=after)),
                        GateConfig(thresholds={"latency": {"max_increase": 0.2}}))
    assert result.status == expected
    assert "Infinity" not in result.model_dump_json()


# 做什么：验证 K 变化被提示且特定 K 的规则不被误用。
# 为什么需要：Recall@1 不能冒充 Recall@3 达标。
def test_top_k_change_is_visible_and_specific_k_rule_is_not_misapplied(baseline):
    candidate = changed(baseline)
    candidate.top_k = 1
    comparison = compare_reports(baseline, candidate)
    assert any("different K" in warning for warning in comparison.warnings)
    assert comparison.configuration_changes["top_k"] == [3, 1]
    assert apply_gate(comparison, GateConfig(thresholds={"recall_at_3": {"min": 0.85}})).status == "FAIL"
    assert apply_gate(comparison, GateConfig(thresholds={"recall_at_k": {"min": 0.85}})).status == "PASS"


# 做什么：验证数据或语料版本不同会阻止直接通过。
# 为什么需要：不能把换试卷后的分数解释成系统进步。
@pytest.mark.parametrize("field,value", [("dataset_version", "different"), ("corpus_version", "different")])
def test_incompatible_data_versions_cannot_pass(baseline, field, value):
    candidate = changed(baseline)
    setattr(candidate, field, value)
    result = apply_gate(compare_reports(baseline, candidate), GateConfig(thresholds={"mrr": {"min": 0.8}}))
    assert result.status == "FAIL" and "version differs" in result.reasons[0]


# 做什么：验证有效题目集合变化时标记不可比。
# 为什么需要：只剩简单成功题不能造成虚假提升。
def test_different_evaluated_cases_cannot_masquerade_as_improvement(baseline):
    candidate = changed(baseline, correctness=5)
    candidate.cohorts["judge"] = ["a"]
    comparison = compare_reports(baseline, candidate)
    assert comparison.metrics["correctness"].delta == pytest.approx(0.7)
    assert comparison.metrics["correctness"].improvement is None
    assert apply_gate(comparison, GateConfig(thresholds={"correctness": {"min": 4}})).status == "FAIL"


# 做什么：验证裁判规则变化只影响依赖裁判的比较。
# 为什么需要：检索指标与模型评分使用不同的可比性条件。
def test_changed_judge_protocol_only_invalidates_judged_metrics(baseline):
    candidate = changed(baseline)
    candidate.judge_prompt_version = "judge_v3"
    comparison = compare_reports(baseline, candidate)
    assert not comparison.metrics["faithfulness"].comparable
    assert comparison.metrics["mrr"].comparable


# 做什么：验证没有有效题却提供分数的快照不能通过。
# 为什么需要：防止没有实验依据的数值被采信。
def test_empty_cohort_cannot_pass_with_a_supplied_score(baseline):
    baseline.cohorts["judge"] = []
    comparison = compare_reports(baseline, changed(baseline, correctness=5))
    result = apply_gate(comparison, GateConfig(thresholds={"correctness": {"min": 4}}))
    assert result.status == "FAIL" and "No evaluated judge cases" in result.reasons[0]


# 做什么：验证模拟零幻觉不能通过真实门禁。
# 为什么需要：不能把占位数据宣传成实际质量。
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


# 做什么：验证离线演示生成两份报告及完整门禁示例。
# 为什么需要：保证模拟版本比较入口可以独立运行。
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
