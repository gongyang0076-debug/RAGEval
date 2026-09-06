"""Judge configuration is independent from the RAG generation service."""

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from .settings import ConfigurationError


@dataclass(frozen=True)
class JudgeSettings:
    api_key: str = field(default="", repr=False)
    base_url: str = ""
    model: str = ""
    max_attempts: int = 3
    timeout_seconds: float = 60.0
    response_format: str = "json"
    retry_delay_seconds: float = 5.0

    @property
    def judge_timeout(self) -> float:
        return self.timeout_seconds

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
