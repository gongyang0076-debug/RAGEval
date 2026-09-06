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


def metric_key(name: str) -> tuple[str, int | None]:
    if name == "latency":
        return "avg_latency", None
    if name in METRICS:
        return name, None
    match = re.fullmatch(r"(recall|precision)_at_([1-9][0-9]*)", name)
    if match:
        return f"{match[1]}_at_k", int(match[2])
    raise ValueError(f"Unknown metric: {name}")


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", revalidate_instances="always")


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

    @field_validator("timestamp")
    @classmethod
    def aware_timestamp(cls, value):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Snapshot timestamp must include a timezone")
        return value

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


class ReportComparison(Contract):
    baseline: BaselineSnapshot
    candidate: BaselineSnapshot
    metrics: dict[str, MetricComparison]
    configuration_changes: dict[str, list[str | int]]
    warnings: list[str]


class Threshold(Contract):
    min: Number | None = None
    max: Number | None = None
    max_increase: Number | None = None

    @model_validator(mode="after")
    def valid_bounds(self):
        if self.min is None and self.max is None and self.max_increase is None:
            raise ValueError("Threshold must define min, max or max_increase")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("Threshold min cannot exceed max")
        return self


class GateConfig(Contract):
    thresholds: dict[str, Threshold] = Field(min_length=1)
    allow_synthetic: bool = Field(default=False, strict=True)

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


class GateCheck(Contract):
    metric: str
    passed: bool
    reasons: list[str]


class GateResult(Contract):
    status: Literal["PASS", "FAIL"]
    simulation: bool
    checks: list[GateCheck]
    reasons: list[str]
