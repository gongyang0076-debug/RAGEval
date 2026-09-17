# 文件作用：规定单题检索指标和整组检索汇总的数据格式。
# 为什么有它：区分有分数的可回答题与应排除的不可回答题，避免错误统计。
"""Small, serializable retrieval metric records."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.dataset.models import NonEmptyString


Ratio = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


# 这个类：保存单题指定 K 下的检索分数与块编号。
# 为什么需要：分数必须能回查对应资料和参评状态。
class RetrievalMetricResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: NonEmptyString
    k: int = Field(gt=0, strict=True)
    answerable: bool
    recall_at_k: Ratio | None
    precision_at_k: Ratio | None
    rr: Ratio | None
    retrieved_chunk_ids: list[NonEmptyString]
    relevant_chunk_ids: list[NonEmptyString]

    # 做什么：检查题目是否应有分数，以及结果数量是否超过 K。
    # 为什么需要：不可回答题必须排除，缺标注或不完整分数不能参与普通聚合。
    @model_validator(mode="after")
    def check_participation(self) -> "RetrievalMetricResult":
        scores = (self.recall_at_k, self.precision_at_k, self.rr)
        if self.answerable:
            if not self.relevant_chunk_ids or any(score is None for score in scores):
                raise ValueError("answerable metric results require relevance labels and all three scores")
        elif self.relevant_chunk_ids or any(score is not None for score in scores):
            raise ValueError("unanswerable metric results require empty relevance labels and None scores")
        if len(self.retrieved_chunk_ids) > self.k:
            raise ValueError("retrieved_chunk_ids must contain at most k returned entries")
        return self


# 这个类：保存同一 K 下的检索宏平均与排除数量。
# 为什么需要：避免把不相同的评测范围混在一起。
class RetrievalMetricSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    k: int = Field(gt=0, strict=True)
    evaluated_cases: int = Field(ge=0)
    excluded_unanswerable_cases: int = Field(ge=0)
    mean_recall_at_k: Ratio | None
    mean_precision_at_k: Ratio | None
    mrr: Ratio | None
