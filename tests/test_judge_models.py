import pytest
from pydantic import ValidationError

from src.judge import JudgeInput, JudgeVerdict


def verdict(**changes):
    return {
        "answer_correctness": 5, "faithfulness": 5, "answer_relevance": 5,
        "completeness": 5, "hallucination": False, "unsupported_claims": [],
        "reason": "答案正确且有上下文支持。", **changes,
    }


def test_correctness_and_faithfulness_are_independent():
    result = JudgeVerdict(**verdict(answer_correctness=0, faithfulness=5))
    assert result.answer_correctness == 0
    assert result.faithfulness == 5
    assert not result.hallucination


def test_hallucination_has_specific_unsupported_claims():
    result = JudgeVerdict(**verdict(hallucination=True, unsupported_claims=["商城赔付100元"], faithfulness=1))
    assert result.unsupported_claims == ["商城赔付100元"]


@pytest.mark.parametrize("hallucination,claims", [(True, []), (False, ["编造的价格"])])
def test_hallucination_flag_must_agree_with_claims(hallucination, claims):
    with pytest.raises(ValidationError, match="hallucination must be true"):
        JudgeVerdict(**verdict(hallucination=hallucination, unsupported_claims=claims))


@pytest.mark.parametrize("field", ["answer_correctness", "faithfulness", "answer_relevance", "completeness"])
@pytest.mark.parametrize("score", [-1, 6, 2.5, "5", True])
def test_scores_are_strict_integers_between_zero_and_five(field, score):
    with pytest.raises(ValidationError):
        JudgeVerdict(**verdict(**{field: score}))


def test_missing_field_and_unknown_field_are_rejected():
    values = verdict()
    values.pop("completeness")
    with pytest.raises(ValidationError, match="completeness"):
        JudgeVerdict(**values)
    with pytest.raises(ValidationError):
        JudgeVerdict(**verdict(metadata={"judge_model": "spoofed"}))


def test_empty_reason_and_empty_claim_are_rejected():
    with pytest.raises(ValidationError):
        JudgeVerdict(**verdict(reason="  "))
    with pytest.raises(ValidationError):
        JudgeVerdict(**verdict(hallucination=True, unsupported_claims=["  "]))


def test_input_supports_unanswerable_and_empty_context():
    value = JudgeInput(case_id="unknown", query="多少钱？", expected_answer="", retrieved_context="", rag_answer="无法确定。", answerable=False)
    assert value.answerable is False
    assert JudgeInput.model_validate_json(value.model_dump_json()) == value


def test_input_rejects_missing_fields_and_non_boolean_answerability():
    with pytest.raises(ValidationError):
        JudgeInput(case_id="case")
    with pytest.raises(ValidationError):
        JudgeInput(case_id="case", query="价格？", expected_answer="", retrieved_context="", rag_answer="", answerable="false")
