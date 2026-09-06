from unittest.mock import Mock, patch

from config.judge import JudgeSettings
from src.judge.smoke import run_live_smoke, smoke_samples


def test_smoke_samples_cover_four_required_scenarios():
    supported, conflict, hallucination, unknown = smoke_samples()
    assert supported.answerable
    assert "60" in conflict.retrieved_context and "60" in conflict.rag_answer
    assert "30" in conflict.expected_answer
    assert "100元" in hallucination.rag_answer and "100元" not in hallucination.retrieved_context
    assert not unknown.answerable
    assert "9.9" in unknown.rag_answer and "9.9" not in unknown.retrieved_context


def test_unconfigured_live_smoke_does_not_fabricate_outputs_or_call_sdk():
    with patch("src.judge.smoke.load_judge_settings", return_value=JudgeSettings()), patch("src.judge.client.OpenAICompatibleProvider") as sdk:
        report = run_live_smoke()
    sdk.assert_not_called()
    assert report["status"] == "NOT RUN"
    assert report["results"] == []
    assert "JUDGE_API_KEY" in report["reason"]


def test_configured_smoke_only_calls_four_judge_samples():
    judge = Mock()
    judge.judge.return_value.model_dump.return_value = {"mock_only": True}
    with patch("src.judge.smoke.load_judge_settings", return_value=JudgeSettings(model="mock-judge")), patch("src.judge.smoke.LLMJudgeClient", return_value=judge):
        report = run_live_smoke()
    assert judge.judge.call_count == 4
    assert report["status"] == "PASS"
    assert len(report["results"]) == 4
