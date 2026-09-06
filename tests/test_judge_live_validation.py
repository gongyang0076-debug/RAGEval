from unittest.mock import Mock, patch

from config.judge import JudgeSettings
from src.judge.live_validation import run_validation, validation_samples


def test_ten_samples_cover_required_categories():
    samples = validation_samples()
    assert len(samples) == len({s.case_id for s in samples}) == 10
    assert sum(not s.answerable for s in samples) == 4
    assert {"smoke_supported", "smoke_context_conflict", "smoke_hallucination", "smoke_unanswerable_fabrication"} <= {s.case_id for s in samples}


def test_missing_live_judge_does_not_create_provider_or_fabricate_results():
    with patch("src.judge.live_validation.load_judge_settings", return_value=JudgeSettings()), patch("src.judge.live_validation.LLMJudgeClient") as client:
        report = run_validation()
    assert report["status"] == "LIVE JUDGE NOT AVAILABLE"
    assert report["results"] == [] and len(report["samples"]) == 10
    client.assert_not_called()


def test_configured_validation_executes_all_ten():
    settings = JudgeSettings(api_key="fake", base_url="https://example.invalid", model="fixture")
    client = Mock()
    client.judge.return_value.model_dump.return_value = {"fixture": True}
    with patch("src.judge.live_validation.load_judge_settings", return_value=settings), patch("src.judge.live_validation.LLMJudgeClient", return_value=client):
        report = run_validation()
    assert client.judge.call_count == 10 and report["status"] == "PASS"
    assert report["judge_model"] == "fixture" and report["prompt_version"] == "judge_v2"
