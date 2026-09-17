# 文件作用：测试裁判环境配置读取、优先级和参数边界。
# 为什么有它：让缺失或非法配置在真正访问模型前清楚报错。
from dataclasses import replace

import pytest

from config.judge import JudgeSettings, load_judge_settings
from config.settings import ConfigurationError


# 做什么：验证 .env 与进程环境变量优先级且配置表示不暴露密钥。
# 为什么需要：保证环境切换正确并保护敏感值。
def test_environment_and_dotenv_precedence_without_exposing_key(tmp_path, monkeypatch):
    values = {
        "JUDGE_API_KEY": "test-only-key", "JUDGE_BASE_URL": "https://example.invalid/v1", "JUDGE_MODEL": "test-judge",
        "JUDGE_MAX_ATTEMPTS": "2", "JUDGE_TIMEOUT_SECONDS": "15", "JUDGE_RESPONSE_FORMAT": "json_schema",
    }
    for key in values:
        monkeypatch.delenv(key, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("\n".join(f"{key}={value}" for key, value in values.items()), encoding="utf-8")
    settings = load_judge_settings(env_file)
    settings.validate()
    assert settings.api_key == "test-only-key"
    assert settings.model == "test-judge"
    assert settings.base_url == "https://example.invalid/v1"
    assert settings.max_attempts == 2
    assert settings.timeout_seconds == 15
    assert settings.response_format == "json_schema"
    assert "test-only-key" not in repr(settings)
    monkeypatch.setenv("JUDGE_MODEL", "external-judge")
    assert load_judge_settings(env_file).model == "external-judge"


# 做什么：验证尝试次数、超时和输出模式边界。
# 为什么需要：非法配置要在请求前失败。
@pytest.mark.parametrize("changes,variable", [
    ({"max_attempts": 0}, "JUDGE_MAX_ATTEMPTS"), ({"max_attempts": 6}, "JUDGE_MAX_ATTEMPTS"),
    ({"max_attempts": True}, "JUDGE_MAX_ATTEMPTS"), ({"timeout_seconds": 0}, "JUDGE_TIMEOUT_SECONDS"),
    ({"timeout_seconds": float("nan")}, "JUDGE_TIMEOUT_SECONDS"), ({"timeout_seconds": 121}, "JUDGE_TIMEOUT_SECONDS"),
    ({"response_format": "invalid"}, "JUDGE_RESPONSE_FORMAT"),
])
def test_bounded_configuration(changes, variable):
    settings = JudgeSettings(api_key="fake", base_url="https://example.invalid", model="fake")
    with pytest.raises(ConfigurationError, match=variable):
        replace(settings, **changes).validate()


# 做什么：验证环境变量不是合法数字时错误明确。
# 为什么需要：帮助使用者定位配置项而不是面对底层转换异常。
@pytest.mark.parametrize("variable", ["JUDGE_MAX_ATTEMPTS", "JUDGE_TIMEOUT_SECONDS"])
def test_invalid_numeric_environment_has_clear_error(monkeypatch, variable):
    monkeypatch.setenv(variable, "not-a-number")
    with pytest.raises(ConfigurationError, match=variable):
        load_judge_settings(None)
