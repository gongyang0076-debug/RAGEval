# 文件作用：规定基线、指标差值、门禁规则和门禁结论的数据结构。
# 为什么有它：统一指标名称、方向和取值，避免错误配置产生虚假通过。
"""Snapshot, comparison and gate contracts; all metric names use one registry."""

import re
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# name -> (higher/lower is better, maximum valid value, evaluation cohort)
METRICS = {
    "recall_at_k": ("higher", 1, "retrieval"),
    "precision_at_k": ("higher", 1, "retrieval"),
    "mrr": ("higher", 1, "retrieval"),
    "correctness": ("higher", 5, "judge"),
    "faithfulness": ("higher", 5, "judge"),
    "relevance": ("higher", 5, "judge"),
    "completeness": ("higher", 5, "judge"),
    "hallucination_rate": ("lower", 1, "judge"),
    "refusal_accuracy": ("higher", 1, "judge"),
    "evaluation_success_rate": ("higher", 1, "all"),
    "avg_latency": ("lower", None, "all"),
}
Number = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
Text = Annotated[str, Field(min_length=1)]


# 做什么：把配置指标名转换成统一名称，并提取可选的 K。
# 为什么需要：让 recall_at_3 等规则与报告字段正确对应。
def metric_key(name: str) -> tuple[str, int | None]:
    if name == "latency":
        return "avg_latency", None
    if name in METRICS:
        return name, None
    match = re.fullmatch(r"(recall|precision)_at_([1-9][0-9]*)", name)
    if match:
        return f"{match[1]}_at_k", int(match[2])
    raise ValueError(f"Unknown metric: {name}")


# 这个类：为比较相关数据模型统一禁止额外字段并重新校验。
# 为什么需要：减少各模型遗漏基础约束的风险。
class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", revalidate_instances="always")


# 这个类：保存一份历史报告的版本、分数和有效题目集合。
# 为什么需要：以后可以在相同口径下比较候选版本。
class BaselineSnapshot(Contract):
    schema_version: Literal[1] = 1
    dataset_version: Text
    corpus_version: Text
    embedding_model: Text
    rag_model: Text
    top_k: int = Field(gt=0, strict=True)
    judge_model: Text
    judge_prompt_version: Text
    rag_mode: Literal["LIVE", "MOCK"]
    judge_mode: Literal["LIVE", "MOCK"]
    quality_metrics_are_synthetic: bool = Field(strict=True)
    metrics: dict[str, Number | None]
    timestamp: datetime
    # Preserve sample identities so changes in available cases cannot masquerade as improvements.
    cohorts: dict[Literal["all", "retrieval", "judge"], list[str]]

    # 做什么：要求快照时间包含时区。
    # 为什么需要：避免不同机器把同一时间解释成不同时刻。
    @field_validator("timestamp")
    @classmethod
    def aware_timestamp(cls, value):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Snapshot timestamp must include a timezone")
        return value

    # 做什么：校验指标名称和允许的最大值。
    # 为什么需要：防止拼错名称或越界分数进入版本比较。
    @field_validator("metrics")
    @classmethod
    def valid_metrics(cls, values):
        for name, value in values.items():
            if name not in METRICS:
                raise ValueError(f"Unknown snapshot metric: {name}")
            maximum = METRICS[name][1]
            if value is not None and maximum is not None and value > maximum:
                raise ValueError(f"{name} must be between 0 and {maximum}")
        return values

    # 做什么：检查 MOCK 标记和各组有效题目编号是否自洽。
    # 为什么需要：避免把模拟成绩当真实成绩，或把不属于数据集的题纳入统计。
    @model_validator(mode="after")
    def consistent_provenance(self):
        if (self.rag_mode == "MOCK" or self.judge_mode == "MOCK") and not self.quality_metrics_are_synthetic:
            raise ValueError("MOCK snapshots must mark quality_metrics_are_synthetic=true")
        for name, ids in self.cohorts.items():
            if len(ids) != len(set(ids)):
                raise ValueError(f"Duplicate case IDs in {name} cohort")
            if name != "all" and not set(ids) <= set(self.cohorts.get("all", [])):
                raise ValueError(f"{name} cohort must be a subset of all cases")
        return self


# 这个类：保存一项指标的前后值、差值、方向及可比性。
# 为什么需要：展示变化时说明是否真的能比较。
class MetricComparison(Contract):
    metric: str
    direction: Literal["higher", "lower"]
    baseline_value: float | None
    candidate_value: float | None
    delta: float | None
    improvement: bool | None
    regression: bool | None
    comparable: bool
    synthetic: bool
    reason: str | None = None


# 这个类：保存两份快照及所有指标比较结果。
# 为什么需要：为页面和门禁提供统一输入。
class ReportComparison(Contract):
    baseline: BaselineSnapshot
    candidate: BaselineSnapshot
    metrics: dict[str, MetricComparison]
    configuration_changes: dict[str, list[str | int]]
    warnings: list[str]


# 这个类：保存一项指标的上下限或相对增幅规则。
# 为什么需要：让质量要求可以用配置表达。
class Threshold(Contract):
    min: Number | None = None
    max: Number | None = None
    max_increase: Number | None = None

    # 做什么：检查阈值至少有一项限制且上下界不矛盾。
    # 为什么需要：防止没有作用或永远不可能满足的门禁规则。
    @model_validator(mode="after")
    def valid_bounds(self):
        if self.min is None and self.max is None and self.max_increase is None:
            raise ValueError("Threshold must define min, max or max_increase")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("Threshold min cannot exceed max")
        return self


# 这个类：保存完整门禁规则和是否允许模拟数据。
# 为什么需要：避免代码里硬编码不同场景的验收要求。
class GateConfig(Contract):
    thresholds: dict[str, Threshold] = Field(min_length=1)
    allow_synthetic: bool = Field(default=False, strict=True)

    # 做什么：检查重复规则、指标范围和相对增幅的使用方向。
    # 为什么需要：保证配置中的验收条件符合指标本身含义。
    @model_validator(mode="after")
    def validate_metrics(self):
        seen = set()
        for name, rule in self.thresholds.items():
            canonical, _ = metric_key(name)
            if canonical in seen:
                raise ValueError(f"Duplicate rules for metric: {canonical}")
            seen.add(canonical)
            direction, maximum, _ = METRICS[canonical]
            if rule.max_increase is not None and direction != "lower":
                raise ValueError(f"max_increase is only supported for lower-is-better metrics: {name}")
            if maximum is not None and any(v is not None and v > maximum for v in (rule.min, rule.max)):
                raise ValueError(f"{name} bounds must be between 0 and {maximum}")
        return self


# 这个类：保存一项门禁检查是否通过及原因。
# 为什么需要：用户能够定位哪条要求没有满足。
class GateCheck(Contract):
    metric: str
    passed: bool
    reasons: list[str]


# 这个类：保存整体 PASS 或 FAIL 及所有原因。
# 为什么需要：自动化和页面可以读取明确验收结论。
class GateResult(Contract):
    status: Literal["PASS", "FAIL"]
    simulation: bool
    checks: list[GateCheck]
    reasons: list[str]
