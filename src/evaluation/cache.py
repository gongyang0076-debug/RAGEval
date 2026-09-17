# 文件作用：按评分输入与配置保存和读取成功 Judge 结果，并提供安全写 JSON 的方法。
# 为什么有它：减少重复模型调用，同时避免损坏缓存或半写入文件误导评测。
"""Content-addressed successful Judge results; no RAG generation caching."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

from pydantic import ValidationError

from src.judge import JudgeInput, JudgeResult
from src.judge.prompts import build_messages


# 做什么：先写临时文件，再替换目标 JSON 文件。
# 为什么需要：降低读到写了一半文件的风险。
def atomic_write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


# 这个类：按输入与配置存取成功裁判结果。
# 为什么需要：减少重复评分并隔离不同评分条件。
class JudgeCache:
    # 做什么：记住裁判缓存所在目录。
    # 为什么需要：让不同运行环境选择自己的缓存位置。
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    # 做什么：对完整评分输入、模型、Prompt 和模式等计算内容摘要。
    # 为什么需要：只有评分依据相同的请求才复用结果。
    def key(self, value: JudgeInput, *, judge_model: str, prompt_version: str,
            judge_mode: str, base_url: str, response_format: str) -> str:
        payload = {
            "cache_version": 1, "input": value.model_dump(), "judge_model": judge_model,
            "prompt_version": prompt_version, "judge_mode": judge_mode, "base_url": base_url,
            "response_format": response_format, "temperature": 0,
            "system_prompt": build_messages(value, prompt_version)[0]["content"],
        }
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    # 做什么：读取并检查对应缓存，损坏时返回未命中和警告。
    # 为什么需要：避免坏缓存阻止重新评测或产生假分数。
    def get(self, key: str) -> tuple[JudgeResult | None, str | None]:
        path = self.directory / f"{key}.json"
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(entry, dict) or entry.get("cache_key") != key:
                raise ValueError("Cache key mismatch")
            result = JudgeResult.model_validate(entry["result"])
            if result.refusal_detected is None:
                raise ValueError("Cache lacks a refusal observation")
            return result, None
        except FileNotFoundError:
            return None, None
        except (OSError, UnicodeError, ValueError, KeyError, ValidationError) as exc:
            return None, f"Ignored invalid/unreadable Judge cache: {type(exc).__name__}"

    # 做什么：保存一次成功的裁判结果，写入失败时返回警告。
    # 为什么需要：复用有效评分，同时不因缓存故障丢弃成功结果。
    def put(self, key: str, result: JudgeResult) -> str | None:
        try:
            atomic_write_json(self.directory / f"{key}.json", {"cache_key": key, "result": result.model_dump(mode="json")})
        except OSError as exc:
            return f"Judge succeeded but cache write failed: {type(exc).__name__}"
        return None
