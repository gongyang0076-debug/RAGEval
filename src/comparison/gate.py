# 文件作用：读取阈值配置并逐项给出 PASS 或 FAIL 及原因。
# 为什么有它：让版本验收按明确规则执行，缺指标和错误配置不能默认为通过。
"""Strict YAML thresholds and a fail-closed, deterministic regression gate."""

from math import isclose
from pathlib import Path

import yaml
from pydantic import ValidationError

from .models import GateCheck, GateConfig, GateResult, ReportComparison, metric_key


# 这个类：表示门禁规则文件读取或校验失败。
# 为什么需要：规则坏了不能被默认为通过。
class GateConfigurationError(ValueError):
    """Invalid threshold file; must never be interpreted as an empty PASS gate."""


# 这个类：给安全 YAML 解析器增加重复键检查。
# 为什么需要：避免相同规则被静默覆盖。
class _UniqueLoader(yaml.SafeLoader):
    pass


# 做什么：解析 YAML 映射时拒绝重复或非字符串键。
# 为什么需要：避免阈值规则被后面的同名配置悄悄覆盖。
def _mapping(loader, node, deep=False):
    loader.flatten_mapping(node)
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str) or key in result:
            raise ValueError(f"Invalid or duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


# 做什么：读取 YAML 并校验门禁规则。
# 为什么需要：错误配置必须明确失败，不能退化成空规则通过。
def load_gate_config(path: str | Path) -> GateConfig:
    try:
        return GateConfig.model_validate(yaml.load(Path(path).read_text(encoding="utf-8"), Loader=_UniqueLoader))
    except (OSError, UnicodeError, ValueError, yaml.YAMLError, ValidationError) as exc:
        raise GateConfigurationError(f"Invalid gate configuration {path}: {exc}") from exc


# 做什么：在小数误差容忍范围之外判断是否超过界限。
# 为什么需要：避免浮点尾差让恰好达标的指标误失败。
def _exceeds(value, limit):
    return value > limit and not isclose(value, limit, rel_tol=0, abs_tol=1e-12)


# 做什么：检查可比性、真实或模拟来源、K 和每项阈值，生成门禁结论。
# 为什么需要：把验收要求转成可解释的 PASS 或 FAIL。
def apply_gate(comparison: ReportComparison, config: GateConfig) -> GateResult:
    config = GateConfig.model_validate(config)
    checks = []
    simulation = False
    for requested, rule in config.thresholds.items():
        key, required_k = metric_key(requested)
        metric = comparison.metrics.get(key)
        reasons = []
        if required_k is not None and comparison.candidate.top_k != required_k:
            reasons.append(f"{requested}: candidate top_k={comparison.candidate.top_k}, expected {required_k}")
        if metric is None or not metric.comparable:
            reasons.append(f"{requested}: {metric.reason if metric else 'Metric missing from comparison'}")
        if metric is not None:
            simulation |= metric.synthetic
            if metric.synthetic and not config.allow_synthetic:
                reasons.append(f"{requested}: synthetic/MOCK metric cannot pass a live gate")
        if not reasons:
            before, after = metric.baseline_value, metric.candidate_value
            if rule.min is not None and _exceeds(rule.min, after):
                reasons.append(f"{requested}: {before:g} -> {after:g}; below min {rule.min:g}")
            if rule.max is not None and _exceeds(after, rule.max):
                reasons.append(f"{requested}: {before:g} -> {after:g}; above max {rule.max:g}")
            if rule.max_increase is not None:
                if before == 0 and after > 0:
                    reasons.append(f"{requested}: 0 -> {after:g}; relative increase is undefined for zero baseline")
                else:
                    increase = (after - before) / before if before else 0
                    if _exceeds(increase, rule.max_increase):
                        reasons.append(f"{requested}: {before:g} -> {after:g}; increased {increase:.2%}, max {rule.max_increase:.2%}")
        checks.append(GateCheck(metric=requested, passed=not reasons, reasons=reasons))
    failures = [reason for check in checks for reason in check.reasons]
    return GateResult(status="FAIL" if failures else "PASS", simulation=simulation, checks=checks, reasons=failures)
