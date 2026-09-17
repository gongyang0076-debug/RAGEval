# 文件作用：测试十条真实验证样例的范围、配置缺失和执行调度。
# 为什么有它：验证 live 检查脚本按约定运行，测试本身仍使用替身。
from unittest.mock import Mock, patch

from config.judge import JudgeSettings
from src.judge.live_validation import run_validation, validation_samples


# 做什么：验证十条样例包含要求的场景。
# 为什么需要：真实验证不能只测试容易通过的正确答案。
def test_ten_samples_cover_required_categories():
    samples = validation_samples()
    assert len(samples) == len({s.case_id for s in samples}) == 10
    assert sum(not s.answerable for s in samples) == 4
    assert {"smoke_supported", "smoke_context_conflict", "smoke_hallucination", "smoke_unanswerable_fabrication"} <= {s.case_id for s in samples}


# 做什么：验证缺配置时不创建服务也不生成假评分。
# 为什么需要：保持真实验证结论诚实。
def test_missing_live_judge_does_not_create_provider_or_fabricate_results():
    with patch("src.judge.live_validation.load_judge_settings", return_value=JudgeSettings()), patch("src.judge.live_validation.LLMJudgeClient") as client:
        report = run_validation()
    assert report["status"] == "LIVE JUDGE NOT AVAILABLE"
    assert report["results"] == [] and len(report["samples"]) == 10
    client.assert_not_called()


# 做什么：用替身验证配置齐全后执行全部十条样例。
# 为什么需要：检查脚本调度而不调用外部模型。
def test_configured_validation_executes_all_ten():
    settings = JudgeSettings(api_key="fake", base_url="https://example.invalid", model="fixture")
    client = Mock()
    client.judge.return_value.model_dump.return_value = {"fixture": True}
    with patch("src.judge.live_validation.load_judge_settings", return_value=settings), patch("src.judge.live_validation.LLMJudgeClient", return_value=client):
        report = run_validation()
    assert client.judge.call_count == 10 and report["status"] == "PASS"
    assert report["judge_model"] == "fixture" and report["prompt_version"] == "judge_v2"
