"""Content-addressed successful Judge results; no RAG generation caching."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

from pydantic import ValidationError

from src.judge import JudgeInput, JudgeResult
from src.judge.prompts import build_messages


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


class JudgeCache:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

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

    def put(self, key: str, result: JudgeResult) -> str | None:
        try:
            atomic_write_json(self.directory / f"{key}.json", {"cache_key": key, "result": result.model_dump(mode="json")})
        except OSError as exc:
            return f"Judge succeeded but cache write failed: {type(exc).__name__}"
        return None
