import pytest

from src.evaluation.cache import JudgeCache
from src.judge import JudgeInput


def test_cache_key_is_stable(tmp_path):
    cache = JudgeCache(tmp_path)
    value = JudgeInput(case_id="a", query="Question", expected_answer="Expected", retrieved_context="Context", rag_answer="Answer", answerable=True)
    identity = dict(judge_model="model", prompt_version="judge_v2", judge_mode="LIVE", base_url="https://example.invalid/v1", response_format="json")
    assert cache.key(value, **identity) == cache.key(value, **identity)
    assert len(cache.key(value, **identity)) == 64


@pytest.mark.parametrize("field,new_value", [
    ("query", "changed"), ("rag_answer", "changed"), ("retrieved_context", "changed"),
    ("expected_answer", "changed"), ("answerable", False), ("case_id", "different"),
    ("judge_model", "different"), ("prompt_version", "judge_v1"), ("judge_mode", "MOCK"),
    ("base_url", "https://other.invalid/v1"), ("response_format", "json_schema"),
])
def test_cache_invalidates_every_input_and_judge_identity(tmp_path, field, new_value):
    cache = JudgeCache(tmp_path)
    value = JudgeInput(case_id="a", query="Question", expected_answer="Expected", retrieved_context="Context", rag_answer="Answer", answerable=True)
    identity = dict(judge_model="model", prompt_version="judge_v2", judge_mode="LIVE", base_url="https://example.invalid/v1", response_format="json")
    before = cache.key(value, **identity)
    if field in type(value).model_fields:
        value = value.model_copy(update={field: new_value})
    else:
        identity[field] = new_value
    assert cache.key(value, **identity) != before


def test_prompt_text_change_invalidates_cache(tmp_path, monkeypatch):
    from src.judge import prompts
    cache = JudgeCache(tmp_path)
    value = JudgeInput(case_id="a", query="Q", expected_answer="E", retrieved_context="C", rag_answer="A", answerable=True)
    identity = dict(judge_model="model", prompt_version="judge_v2", judge_mode="LIVE", base_url="https://example.invalid", response_format="json")
    before = cache.key(value, **identity)
    monkeypatch.setattr(prompts, "JUDGE_V1", prompts.JUDGE_V1 + "changed rubric")
    assert cache.key(value, **identity) != before
