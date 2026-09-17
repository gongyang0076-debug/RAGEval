# 文件作用：读取并检查 Judge 使用的模型、地址、密钥、超时和输出格式。
# 为什么有它：裁判与生成模型需要独立配置，配置错误应在调用服务前发现。
"""Judge configuration is independent from the RAG generation service."""

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from .settings import ConfigurationError


# 这个类：保存裁判服务需要的配置。
# 为什么需要：允许 Judge 与生成模型分别配置。
@dataclass(frozen=True)
class JudgeSettings:
    api_key: str = field(default="", repr=False)
    base_url: str = ""
    model: str = ""
    max_attempts: int = 3
    timeout_seconds: float = 60.0
    response_format: str = "json"
    retry_delay_seconds: float = 5.0

    # 做什么：提供统一的裁判超时读取属性。
    # 为什么需要：让调用层不必关心配置内部字段名称。
    @property
    def judge_timeout(self) -> float:
        return self.timeout_seconds

    # 做什么：检查裁判必填配置、尝试次数、超时和输出模式是否合法。
    # 为什么需要：避免发起请求后才发现配置错误。
    def validate(self) -> None:
        missing = [name for name, value in (
            ("JUDGE_API_KEY", self.api_key), ("JUDGE_BASE_URL", self.base_url), ("JUDGE_MODEL", self.model),
        ) if not value.strip()]
        if missing:
            raise ConfigurationError("Missing Judge configuration: " + ", ".join(missing))
        if type(self.max_attempts) is not int or not 1 <= self.max_attempts <= 5:
            raise ConfigurationError("JUDGE_MAX_ATTEMPTS must be an integer between 1 and 5 (total attempts)")
        if not math.isfinite(self.timeout_seconds) or not 0 < self.timeout_seconds <= 120:
            raise ConfigurationError("JUDGE_TIMEOUT_SECONDS must be finite and in (0, 120]")
        if self.response_format not in {"json", "json_object", "json_schema"}:
            raise ConfigurationError("JUDGE_RESPONSE_FORMAT must be json, json_object or json_schema")
        if isinstance(self.retry_delay_seconds, bool) or not 0 < self.retry_delay_seconds <= 30:
            raise ConfigurationError("JUDGE_RETRY_DELAY_SECONDS must be in (0, 30]")


# 做什么：从环境变量和可选 .env 文件构造裁判配置。
# 为什么需要：允许更换服务而不修改代码，并保留进程环境变量的优先级。
def load_judge_settings(env_file: str | Path | None = ".env") -> JudgeSettings:
    if env_file is not None:
        load_dotenv(dotenv_path=env_file, override=False)
    try:
        max_attempts = int(os.environ.get("JUDGE_MAX_ATTEMPTS", "3"))
    except ValueError as exc:
        raise ConfigurationError("JUDGE_MAX_ATTEMPTS must be an integer between 1 and 5") from exc
    try:
        timeout = float(os.environ.get("JUDGE_TIMEOUT_SECONDS", "60"))
    except ValueError as exc:
        raise ConfigurationError("JUDGE_TIMEOUT_SECONDS must be a number in (0, 120]") from exc
    try:
        retry_delay = float(os.environ.get("JUDGE_RETRY_DELAY_SECONDS", "5"))
    except ValueError as exc:
        raise ConfigurationError("JUDGE_RETRY_DELAY_SECONDS must be a number in (0, 30]") from exc
    return JudgeSettings(
        api_key=os.environ.get("JUDGE_API_KEY", "").strip(),
        base_url=os.environ.get("JUDGE_BASE_URL", "").strip(),
        model=os.environ.get("JUDGE_MODEL", "").strip(),
        max_attempts=max_attempts, timeout_seconds=timeout,
        response_format=os.environ.get("JUDGE_RESPONSE_FORMAT", "json").strip(),
        retry_delay_seconds=retry_delay,
    )
