from __future__ import annotations

import json

import pytest

from memry.config import Config

KEYS = [
    "MEMRY_CONFIG", "MEMRY_DB_PATH", "MEMRY_BACKEND", "MEMRY_DEFAULT_USER",
    "MEMRY_API_KEY", "MEMRY_LLM_PROVIDER", "MEMRY_LLM_MODEL", "MEMRY_LLM_API_KEY",
    "MEMRY_LLM_BASE_URL", "MEMRY_LLM_EFFORT", "MEMRY_EMBEDDING_PROVIDER",
    "MEMRY_EMBEDDING_MODEL", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "VOYAGE_API_KEY",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for key in KEYS:
        monkeypatch.delenv(key, raising=False)
    # point the default config file somewhere empty
    monkeypatch.setenv("MEMRY_CONFIG", str(tmp_path / "missing.json"))


def test_defaults_zero_key(tmp_path):
    cfg = Config.load()
    assert cfg.backend == "local"
    assert cfg.llm.provider == "none"
    assert cfg.embedding.provider == "hash"
    assert cfg.default_user_id == "default"


def test_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMRY_DB_PATH", str(tmp_path / "x.db"))
    monkeypatch.setenv("MEMRY_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("MEMRY_LLM_MODEL", "qwen3")
    monkeypatch.setenv("MEMRY_DEFAULT_USER", "cosmin")
    cfg = Config.load()
    assert cfg.db_path == str(tmp_path / "x.db")
    assert cfg.llm.provider == "ollama"
    assert cfg.llm.resolved_model() == "qwen3"
    assert cfg.default_user_id == "cosmin"


def test_file_config_env_wins(monkeypatch, tmp_path):
    file = tmp_path / "config.json"
    file.write_text(json.dumps({"default_user_id": "from-file", "backend": "local",
                                "llm": {"provider": "openai"}}), encoding="utf-8")
    monkeypatch.setenv("MEMRY_CONFIG", str(file))
    monkeypatch.setenv("MEMRY_DEFAULT_USER", "from-env")
    cfg = Config.load()
    assert cfg.default_user_id == "from-env"  # env beats file
    assert cfg.llm.provider == "openai"       # file beats defaults


def test_provider_autodetect_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    cfg = Config.load()
    assert cfg.llm.provider == "anthropic"
    assert cfg.llm.resolved_model() == "claude-opus-4-8"
    assert cfg.embedding.provider == "hash"  # anthropic has no embeddings API


def test_provider_autodetect_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg = Config.load()
    assert cfg.llm.provider == "openai"
    assert cfg.embedding.provider == "openai"


def test_explicit_provider_beats_autodetect(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("MEMRY_LLM_PROVIDER", "ollama")
    cfg = Config.load()
    assert cfg.llm.provider == "ollama"


def test_pinned_hash_embeddings_survive_openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("MEMRY_EMBEDDING_PROVIDER", "hash")
    cfg = Config.load()
    assert cfg.embedding.provider == "hash"
    assert cfg.llm.provider == "openai"  # LLM not pinned -> still autodetects


def test_pinned_none_llm_survives_api_keys(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("MEMRY_LLM_PROVIDER", "none")
    cfg = Config.load()
    assert cfg.llm.provider == "none"


def test_file_pinned_embedding_survives_openai_key(monkeypatch, tmp_path):
    file = tmp_path / "config.json"
    file.write_text(json.dumps({"embedding": {"provider": "hash"}}), encoding="utf-8")
    monkeypatch.setenv("MEMRY_CONFIG", str(file))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg = Config.load()
    assert cfg.embedding.provider == "hash"


def test_redacted_hides_secrets():
    cfg = Config(api_key="topsecret")
    cfg.llm.api_key = "sk-hidden"
    dump = cfg.redacted()
    assert dump["api_key"] == "***"
    assert dump["llm"]["api_key"] == "***"
    assert "topsecret" not in json.dumps(dump)
