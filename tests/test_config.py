from __future__ import annotations

import json

import pytest

from memry.config import Config

KEYS = [
    "MEMRY_CONFIG", "MEMRY_DB_PATH", "MEMRY_DEFAULT_USER",
    "MEMRY_API_KEY", "MEMRY_LLM_PROVIDER", "MEMRY_LLM_MODEL", "MEMRY_LLM_API_KEY",
    "MEMRY_LLM_BASE_URL", "MEMRY_LLM_EFFORT", "MEMRY_EMBEDDING_PROVIDER",
    "MEMRY_EMBEDDING_MODEL", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "VOYAGE_API_KEY",
    "MEMRY_DECISION_PROVIDER", "MEMRY_DECISION_API_KEY", "MEMRY_DECISION_MODEL",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for key in KEYS:
        monkeypatch.delenv(key, raising=False)
    # point the default config file somewhere empty
    monkeypatch.setenv("MEMRY_CONFIG", str(tmp_path / "missing.json"))


def test_defaults_zero_key(tmp_path):
    cfg = Config.load()
    assert cfg.llm.provider == "none"
    assert cfg.embedding.provider == "hash"
    assert cfg.default_user_id == "default"


def test_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMRY_DB_PATH", str(tmp_path / "x.db"))
    monkeypatch.setenv("MEMRY_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("MEMRY_LLM_MODEL", "qwen3")
    monkeypatch.setenv("MEMRY_DEFAULT_USER", "marcus")
    cfg = Config.load()
    assert cfg.db_path == str(tmp_path / "x.db")
    assert cfg.llm.provider == "ollama"
    assert cfg.llm.resolved_model() == "qwen3"
    assert cfg.default_user_id == "marcus"


def test_extraction_asks_the_model_the_comparison_kept():
    """Extraction sends each save to the OpenAI default, gpt-6-luna. Compared
    with gpt-5.6-luna on generated saves and owner texts it was as good on
    every measure, within run noise, at half the price, so the default
    stayed."""
    import httpx

    from memry.config import DEFAULT_LLM_MODELS, LLMConfig
    from memry.intelligence.extraction import extract_facts
    from memry.providers.llm import OpenAILLM

    assert DEFAULT_LLM_MODELS["openai"] == "gpt-6-luna"
    sent = []

    def reply(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [
            {"message": {"content": json.dumps({"facts": []})}}]})

    llm = OpenAILLM(LLMConfig(provider="openai", api_key="k"))
    llm._client = httpx.Client(transport=httpx.MockTransport(reply))
    assert extract_facts(llm, [{"role": "user", "content": "Ada moved to Leeds"}]) == []
    assert [body["model"] for body in sent] == ["gpt-6-luna"]
    llm.close()


def test_the_extraction_model_is_a_setting(monkeypatch):
    monkeypatch.setenv("MEMRY_LLM_PROVIDER", "openai")
    assert Config.load().llm.resolved_model() == "gpt-6-luna"
    monkeypatch.setenv("MEMRY_LLM_MODEL", "gpt-5.6-luna")
    assert Config.load().llm.resolved_model() == "gpt-5.6-luna"


def test_file_config_env_wins(monkeypatch, tmp_path):
    file = tmp_path / "config.json"
    file.write_text(json.dumps({"default_user_id": "from-file",
                                "llm": {"provider": "openai"}}), encoding="utf-8")
    monkeypatch.setenv("MEMRY_CONFIG", str(file))
    monkeypatch.setenv("MEMRY_DEFAULT_USER", "from-env")
    cfg = Config.load()
    assert cfg.default_user_id == "from-env"  # env beats file
    assert cfg.llm.provider == "openai"       # file beats defaults


def _pretend_anthropic_sdk_installed(monkeypatch):
    """Autodetect only picks Anthropic when the optional SDK is importable; CI
    installs the package without that extra, so make the check succeed."""
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **k: object() if name == "anthropic" else real(name, *a, **k),
    )


def test_provider_autodetect_anthropic(monkeypatch):
    _pretend_anthropic_sdk_installed(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    cfg = Config.load()
    assert cfg.llm.provider == "anthropic"
    assert cfg.llm.resolved_model() == "claude-haiku-4-5"
    assert cfg.embedding.provider == "hash"  # anthropic has no embeddings API


def test_provider_autodetect_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg = Config.load()
    assert cfg.llm.provider == "openai"
    assert cfg.embedding.provider == "openai"


def test_both_keys_pick_one_provider_for_everything(monkeypatch):
    """Two keys must not mean two vendors.

    Anthropic has no embeddings API, so preferring it for the LLM whenever its
    key exists would split the deployment: Anthropic for extraction, OpenAI for
    vectors. OpenAI serves both, so it wins when present.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg = Config.load()
    assert cfg.llm.provider == "openai"
    assert cfg.embedding.provider == "openai"
    assert cfg.llm.resolved_model() == "gpt-6-luna"


def test_anthropic_still_wins_when_it_is_the_only_key(monkeypatch):
    _pretend_anthropic_sdk_installed(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("VOYAGE_API_KEY", "pa-test")
    cfg = Config.load()
    assert cfg.llm.provider == "anthropic"
    assert cfg.embedding.provider == "voyage"  # the only embeddings option here


def test_anthropic_can_still_be_pinned_alongside_openai(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("MEMRY_LLM_PROVIDER", "anthropic")
    cfg = Config.load()
    assert cfg.llm.provider == "anthropic"
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


def test_autodetect_anthropic_without_sdk_stays_keyless(monkeypatch, caplog):
    """A bare `pip install memry` + ANTHROPIC_API_KEY must not crash at startup:
    autodetection only upgrades when the optional SDK is importable."""
    import importlib.util

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **k: None if name == "anthropic" else real_find_spec(name, *a, **k),
    )
    with caplog.at_level("WARNING", logger="memry"):
        cfg = Config.load()
    assert cfg.llm.provider == "none"
    assert "memry[anthropic]" in caplog.text
    # and the store still builds (this is the exact call that used to raise)
    from memry.store import MemoryStore

    cfg.db_path = ":memory:"
    assert MemoryStore(cfg).llm.available is False



# -- the two models a server needs ------------------------------------------
def test_a_server_needs_a_text_model_and_a_decision_model():
    from memry.config import model_requirements, require_models

    cfg = Config.load()
    assert len(model_requirements(cfg)) == 2
    with pytest.raises(SystemExit) as stopped:
        require_models(cfg)
    assert "OPENAI_API_KEY" in str(stopped.value)
    assert "MEMRY_DECISION_PROVIDER=jev" in str(stopped.value)


def test_a_text_model_alone_is_not_enough(monkeypatch):
    from memry.config import model_requirements

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    [missing] = model_requirements(Config.load())
    assert "decision model" in missing


def test_jev_needs_its_key(monkeypatch):
    from memry.config import model_requirements, require_models

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("MEMRY_DECISION_PROVIDER", "jev")
    [missing] = model_requirements(Config.load())
    assert "MEMRY_DECISION_API_KEY" in missing
    monkeypatch.setenv("MEMRY_DECISION_API_KEY", "ts-test")
    require_models(Config.load())  # nothing missing


@pytest.mark.parametrize("provider", ["llm", "none"])
def test_the_text_model_decides_only_when_chosen(monkeypatch, caplog, provider):
    from memry.config import require_models

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("MEMRY_DECISION_PROVIDER", provider)
    with caplog.at_level("WARNING", logger="memry"):
        require_models(Config.load())
    assert "merges entities only by fixed rules" in caplog.text


def test_the_decision_key_is_redacted(monkeypatch):
    monkeypatch.setenv("MEMRY_DECISION_PROVIDER", "jev")
    monkeypatch.setenv("MEMRY_DECISION_API_KEY", "ts-secret")
    assert "ts-secret" not in json.dumps(Config.load().redacted())
