from pathlib import Path
import pytest

from config.settings import ConfigurationError, load_settings


def test_default_settings(monkeypatch):
    monkeypatch.delenv("RAGEVAL_DATA_DIR", raising=False)
    assert load_settings(env_file=None).data_dir == Path("data")


def test_dotenv_and_environment_precedence(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("RAGEVAL_DATA_DIR=fixture-data\n", encoding="utf-8")
    monkeypatch.delenv("RAGEVAL_DATA_DIR", raising=False)
    assert load_settings(env_file).data_dir == Path("fixture-data")
    monkeypatch.setenv("RAGEVAL_DATA_DIR", "external-data")
    assert load_settings(env_file).data_dir == Path("external-data")


def test_rag_environment_settings(monkeypatch):
    variables = {
        "RAG_EMBEDDING_MODEL": "local-model", "RAG_EMBEDDING_DEVICE": "cpu",
        "RAG_EMBEDDING_CACHE_DIR": "local-cache", "RAG_EMBEDDING_QUERY_INSTRUCTION": "query: ",
        "RAG_CHROMA_DIR": "local-chroma", "RAG_CHROMA_COLLECTION": "test_collection",
        "RAG_API_KEY": "test-only-secret", "RAG_BASE_URL": "https://example.invalid/v1",
        "RAG_MODEL": "test-model",
    }
    for name, value in variables.items():
        monkeypatch.setenv(name, value)
    settings = load_settings(env_file=None)
    settings.require_llm()
    assert settings.embedding_model == "local-model"
    assert settings.embedding_device == "cpu"
    assert settings.embedding_cache_dir == Path("local-cache")
    assert settings.embedding_query_instruction == "query: "
    assert settings.chroma_dir == Path("local-chroma")
    assert settings.chroma_collection == "test_collection"
    assert settings.rag_api_key == "test-only-secret"
    assert settings.rag_base_url == "https://example.invalid/v1"
    assert settings.rag_model == "test-model"
    assert "test-only-secret" not in repr(settings)


def test_rag_retry_settings(monkeypatch):
    monkeypatch.setenv("RAG_MAX_ATTEMPTS", "2")
    monkeypatch.setenv("RAG_RETRY_DELAY_SECONDS", "7.5")
    settings = load_settings(env_file=None)
    assert settings.rag_max_attempts == 2
    assert settings.rag_retry_delay_seconds == 7.5


@pytest.mark.parametrize("name,value", [
    ("RAG_MAX_ATTEMPTS", "0"), ("RAG_MAX_ATTEMPTS", "6"), ("RAG_MAX_ATTEMPTS", "abc"),
    ("RAG_RETRY_DELAY_SECONDS", "0"), ("RAG_RETRY_DELAY_SECONDS", "31"),
    ("RAG_RETRY_DELAY_SECONDS", "nan"), ("RAG_RETRY_DELAY_SECONDS", "inf"),
])
def test_invalid_rag_retry_settings(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ConfigurationError, match=name):
        load_settings(env_file=None)


def test_generation_timeout_and_optional_thinking(monkeypatch):
    monkeypatch.setenv("RAG_GENERATION_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("RAG_THINKING", "disabled")
    settings = load_settings(env_file=None)
    assert settings.generation_timeout == 45 and settings.rag_thinking == "disabled"


@pytest.mark.parametrize("value", ["0", "121", "nan", "inf", "invalid"])
def test_invalid_generation_timeout(monkeypatch, value):
    monkeypatch.setenv("RAG_GENERATION_TIMEOUT_SECONDS", value)
    with pytest.raises(ConfigurationError, match="RAG_GENERATION_TIMEOUT_SECONDS"):
        load_settings(env_file=None)
