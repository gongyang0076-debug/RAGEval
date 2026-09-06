"""Load configuration explicitly; importing this module has no side effects."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


class ConfigurationError(ValueError):
    """Required service configuration is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_device: str = "cpu"
    embedding_cache_dir: Path = Path(".cache/huggingface")
    embedding_query_instruction: str = "为这个句子生成表示以用于检索相关文章："
    chroma_dir: Path = Path(".cache/chroma")
    chroma_collection: str = "ecommerce_v1"
    rag_api_key: str = field(default="", repr=False)
    rag_base_url: str = ""
    rag_model: str = ""
    rag_max_attempts: int = 3
    rag_retry_delay_seconds: float = 5.0
    generation_timeout: float = 60.0
    rag_thinking: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.generation_timeout, bool) or not 0 < self.generation_timeout <= 120:
            raise ConfigurationError("RAG_GENERATION_TIMEOUT_SECONDS must be in (0, 120]")
        if self.rag_thinking not in {None, "enabled", "disabled"}:
            raise ConfigurationError("RAG_THINKING must be enabled, disabled or empty")
        if type(self.rag_max_attempts) is not int or not 1 <= self.rag_max_attempts <= 5:
            raise ConfigurationError("RAG_MAX_ATTEMPTS must be an integer between 1 and 5")
        if (isinstance(self.rag_retry_delay_seconds, bool)
                or not isinstance(self.rag_retry_delay_seconds, (int, float))
                or not 0 < self.rag_retry_delay_seconds <= 30):
            raise ConfigurationError("RAG_RETRY_DELAY_SECONDS must be a number greater than 0 and at most 30")

    def require_llm(self) -> None:
        missing = [
            name for name, value in (
                ("RAG_API_KEY", self.rag_api_key),
                ("RAG_BASE_URL", self.rag_base_url),
                ("RAG_MODEL", self.rag_model),
            ) if not value.strip()
        ]
        if missing:
            raise ConfigurationError(
                "Missing LLM configuration: " + ", ".join(missing)
                + ". Set these environment variables or .env; retrieval-only does not require them."
            )


def load_settings(env_file: str | Path | None = ".env") -> Settings:
    """Read optional dotenv values without overriding the process environment."""
    if env_file is not None:
        load_dotenv(dotenv_path=env_file, override=False)
    try:
        rag_max_attempts = int(os.environ.get("RAG_MAX_ATTEMPTS", "3"))
    except ValueError as exc:
        raise ConfigurationError("RAG_MAX_ATTEMPTS must be an integer between 1 and 5") from exc
    try:
        rag_retry_delay_seconds = float(os.environ.get("RAG_RETRY_DELAY_SECONDS", "5"))
    except ValueError as exc:
        raise ConfigurationError("RAG_RETRY_DELAY_SECONDS must be a number greater than 0 and at most 30") from exc
    try:
        generation_timeout = float(os.environ.get("RAG_GENERATION_TIMEOUT_SECONDS", "60"))
    except ValueError as exc:
        raise ConfigurationError("RAG_GENERATION_TIMEOUT_SECONDS must be a number in (0, 120]") from exc
    return Settings(
        data_dir=Path(os.environ.get("RAGEVAL_DATA_DIR", "data")),
        embedding_model=os.environ.get("RAG_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"),
        embedding_device=os.environ.get("RAG_EMBEDDING_DEVICE", "cpu"),
        embedding_cache_dir=Path(os.environ.get("RAG_EMBEDDING_CACHE_DIR", ".cache/huggingface")),
        embedding_query_instruction=os.environ.get(
            "RAG_EMBEDDING_QUERY_INSTRUCTION", "为这个句子生成表示以用于检索相关文章："
        ),
        chroma_dir=Path(os.environ.get("RAG_CHROMA_DIR", ".cache/chroma")),
        chroma_collection=os.environ.get("RAG_CHROMA_COLLECTION", "ecommerce_v1"),
        rag_api_key=os.environ.get("RAG_API_KEY", "").strip(),
        rag_base_url=os.environ.get("RAG_BASE_URL", "").strip(),
        rag_model=os.environ.get("RAG_MODEL", "").strip(),
        rag_max_attempts=rag_max_attempts,
        rag_retry_delay_seconds=rag_retry_delay_seconds,
        generation_timeout=generation_timeout,
        rag_thinking=os.environ.get("RAG_THINKING", "").strip() or None,
    )
