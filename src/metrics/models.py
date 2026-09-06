"""Small, serializable retrieval metric records."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.dataset.models import NonEmptyString


Ratio = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


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


class RetrievalMetricSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    k: int = Field(gt=0, strict=True)
    evaluated_cases: int = Field(ge=0)
    excluded_unanswerable_cases: int = Field(ge=0)
    mean_recall_at_k: Ratio | None
    mean_precision_at_k: Ratio | None
    mrr: Ratio | None
