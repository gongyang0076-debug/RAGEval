# 文件作用：测试四条冒烟样例以及缺配置时的处理。
# 为什么有它：确保真实验证不会自动伪造结果，单元测试也不发外部请求。
from unittest.mock import Mock, patch

from config.judge import JudgeSettings
from src.judge.smoke import run_live_smoke, smoke_samples


# 做什么：验证四条冒烟样例覆盖指定问题类型。
# 为什么需要：最小验证也要包含错误与编造场景。
def test_smoke_samples_cover_four_required_scenarios():
    supported, conflict, hallucination, unknown = smoke_samples()
    assert supported.answerable
    assert "60" in conflict.retrieved_context and "60" in conflict.rag_answer
    assert "30" in conflict.expected_answer
    assert "100元" in hallucination.rag_answer and "100元" not in hallucination.retrieved_context
    assert not unknown.answerable
    assert "9.9" in unknown.rag_answer and "9.9" not in unknown.retrieved_context


# 做什么：验证缺 API 配置时既不调用 SDK 也不伪造结果。
# 为什么需要：区分未运行和真实通过。
def test_unconfigured_live_smoke_does_not_fabricate_outputs_or_call_sdk():
    with patch("src.judge.smoke.load_judge_settings", return_value=JudgeSettings()), patch("src.judge.client.OpenAICompatibleProvider") as sdk:
        report = run_live_smoke()
    sdk.assert_not_called()
    assert report["status"] == "NOT RUN"
    assert report["results"] == []
    assert "JUDGE_API_KEY" in report["reason"]


# 做什么：验证冒烟脚本只判定预设四条样例。
# 为什么需要：确保小规模接入检查不意外扩大为全量请求。
def test_configured_smoke_only_calls_four_judge_samples():
    judge = Mock()
    judge.judge.return_value.model_dump.return_value = {"mock_only": True}
    with patch("src.judge.smoke.load_judge_settings", return_value=JudgeSettings(model="mock-judge")), patch("src.judge.smoke.LLMJudgeClient", return_value=judge):
        report = run_live_smoke()
    assert judge.judge.call_count == 4
    assert report["status"] == "PASS"
    assert len(report["results"]) == 4
