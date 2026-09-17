# 文件作用：测试裁判输入输出的类型、分数范围和幻觉证据约束。
# 为什么有它：防止看似是 JSON 但不符合评分契约的数据进入报告。
import pytest
from pydantic import ValidationError

from src.judge import JudgeInput, JudgeVerdict


# 做什么：生成可修改字段的合法评分字典。
# 为什么需要：每个模型约束测试只需改动待验证的字段。
def verdict(**changes):
    return {
        "answer_correctness": 5, "faithfulness": 5, "answer_relevance": 5,
        "completeness": 5, "hallucination": False, "unsupported_claims": [],
        "reason": "答案正确且有上下文支持。", **changes,
    }


# 做什么：验证正确性和忠实性字段互不强制相等。
# 为什么需要：数据契约要允许两个维度反映不同问题。
def test_correctness_and_faithfulness_are_independent():
    result = JudgeVerdict(**verdict(answer_correctness=0, faithfulness=5))
    assert result.answer_correctness == 0
    assert result.faithfulness == 5
    assert not result.hallucination


# 做什么：验证幻觉结果保存具体无依据断言。
# 为什么需要：让安全判断有可核对的内容。
def test_hallucination_has_specific_unsupported_claims():
    result = JudgeVerdict(**verdict(hallucination=True, unsupported_claims=["商城赔付100元"], faithfulness=1))
    assert result.unsupported_claims == ["商城赔付100元"]


# 做什么：验证幻觉布尔值与证据列表必须一致。
# 为什么需要：避免结构上自相矛盾的评分。
@pytest.mark.parametrize("hallucination,claims", [(True, []), (False, ["编造的价格"])])
def test_hallucination_flag_must_agree_with_claims(hallucination, claims):
    with pytest.raises(ValidationError, match="hallucination must be true"):
        JudgeVerdict(**verdict(hallucination=hallucination, unsupported_claims=claims))


# 做什么：验证分数必须是范围内整数。
# 为什么需要：字符串、布尔值和越界值不能被悄悄当成合法分数。
@pytest.mark.parametrize("field", ["answer_correctness", "faithfulness", "answer_relevance", "completeness"])
@pytest.mark.parametrize("score", [-1, 6, 2.5, "5", True])
def test_scores_are_strict_integers_between_zero_and_five(field, score):
    with pytest.raises(ValidationError):
        JudgeVerdict(**verdict(**{field: score}))


# 做什么：验证缺字段或额外字段被拒绝。
# 为什么需要：固定 Schema 才能让后续代码可靠读取评分。
def test_missing_field_and_unknown_field_are_rejected():
    values = verdict()
    values.pop("completeness")
    with pytest.raises(ValidationError, match="completeness"):
        JudgeVerdict(**values)
    with pytest.raises(ValidationError):
        JudgeVerdict(**verdict(metadata={"judge_model": "spoofed"}))


# 做什么：验证理由和断言文本不能为空。
# 为什么需要：防止只有字段名而没有可复核内容。
def test_empty_reason_and_empty_claim_are_rejected():
    with pytest.raises(ValidationError):
        JudgeVerdict(**verdict(reason="  "))
    with pytest.raises(ValidationError):
        JudgeVerdict(**verdict(hallucination=True, unsupported_claims=["  "]))


# 做什么：验证不可回答题及空上下文能够进入判卷。
# 为什么需要：这些是需要评测的场景而不是一律非法输入。
def test_input_supports_unanswerable_and_empty_context():
    value = JudgeInput(case_id="unknown", query="多少钱？", expected_answer="", retrieved_context="", rag_answer="无法确定。", answerable=False)
    assert value.answerable is False
    assert JudgeInput.model_validate_json(value.model_dump_json()) == value


# 做什么：验证输入必须完整且可回答性为真实布尔值。
# 为什么需要：避免字符串等模糊输入改变裁判语义。
def test_input_rejects_missing_fields_and_non_boolean_answerability():
    with pytest.raises(ValidationError):
        JudgeInput(case_id="case")
    with pytest.raises(ValidationError):
        JudgeInput(case_id="case", query="价格？", expected_answer="", retrieved_context="", rag_answer="", answerable="false")
