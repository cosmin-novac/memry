"""Configuration for Memry.

Resolution order (later wins):
1. built-in defaults
2. config file (JSON) - ``MEMRY_CONFIG`` or ``~/.memry/config.json``
3. environment variables (``MEMRY_*``)
4. explicit kwargs / ``Config(...)`` construction

Zero-config behavior: with no keys configured, Memry runs fully local -
verbatim memory writes (no LLM extraction) + hybrid retrieval over FTS5 BM25
and deterministic hash embeddings. Setting ``ANTHROPIC_API_KEY`` or
``OPENAI_API_KEY`` upgrades extraction/reconciliation automatically.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

DEFAULT_DIR = Path.home() / ".memry"

LLMProvider = Literal["anthropic", "openai", "ollama", "none"]
EmbeddingProvider = Literal["openai", "ollama", "voyage", "hash", "none"]

DEFAULT_LLM_MODELS: dict[str, str] = {
    "anthropic": "claude-opus-4-8",
    "openai": "gpt-5-mini",
    "ollama": "llama3.1",
}

DEFAULT_EMBEDDING_MODELS: dict[str, str] = {
    "openai": "text-embedding-3-small",
    "ollama": "nomic-embed-text",
    "voyage": "voyage-3.5-lite",
    "hash": "hash-v1",
}


class LLMConfig(BaseModel):
    provider: LLMProvider = "none"
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    max_tokens: int = 2000
    effort: str = "low"  # Anthropic effort level for extraction calls
    timeout: float = 120.0

    def resolved_model(self) -> str:
        return self.model or DEFAULT_LLM_MODELS.get(self.provider, "")


class EmbeddingConfig(BaseModel):
    provider: EmbeddingProvider = "hash"
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    dimensions: int | None = None
    timeout: float = 60.0

    def resolved_model(self) -> str:
        return self.model or DEFAULT_EMBEDDING_MODELS.get(self.provider, "")


class RetrievalConfig(BaseModel):
    rrf_k: int = 60
    vector_weight: float = 1.0
    keyword_weight: float = 1.0
    fused_weight: float = 0.70
    recency_weight: float = 0.15
    importance_weight: float = 0.15
    recency_half_life_days: float = 30.0
    candidate_multiplier: int = 3
    reconcile_similarity_limit: int = 5


class DecayConfig(BaseModel):
    enabled: bool = True
    half_life_days: float = 90.0
    floor: float = 0.15  # decayed importance never drops below floor * importance


class AnnConfig(BaseModel):
    """Approximate-nearest-neighbor settings (usearch HNSW sidecar).

    Requires ``pip install memry[ann]``. Below ``min_rows`` exact brute-force
    search is used (it's faster there anyway); above it, the HNSW index
    over-fetches candidates that are then exact-rescored.
    """

    enabled: bool = True
    min_rows: int = 5000
    overfetch: int = 8


class TenantConfig(BaseModel):
    """One tenant of a multi-tenant server: its own API key, its own
    transparently-namespaced memory space."""

    name: str
    api_key: str


class Config(BaseModel):
    db_path: str = str(DEFAULT_DIR / "memry.db")
    backend: Literal["local", "mem0", "postgres"] = "local"
    postgres_dsn: str | None = None  # for backend="postgres"
    default_user_id: str = "default"
    api_key: str | None = None  # admin bearer token for the REST/MCP server
    tenants: list[TenantConfig] = Field(default_factory=list)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    decay: DecayConfig = Field(default_factory=DecayConfig)
    ann: AnnConfig = Field(default_factory=AnnConfig)

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | os.PathLike[str] | None = None, **overrides: Any) -> "Config":
        data: dict[str, Any] = {}

        file_path = path or os.environ.get("MEMRY_CONFIG") or (DEFAULT_DIR / "config.json")
        try:
            raw = Path(file_path).read_text(encoding="utf-8")
            data = _deep_merge(data, json.loads(raw))
        except (FileNotFoundError, OSError):
            pass

        data = _deep_merge(data, _from_env())
        data = _deep_merge(data, overrides)
        cfg = cls.model_validate(data)
        return _autodetect_providers(cfg)

    def redacted(self) -> dict[str, Any]:
        d = self.model_dump()
        for section in ("llm", "embedding"):
            if d[section].get("api_key"):
                d[section]["api_key"] = "***"
        if d.get("api_key"):
            d["api_key"] = "***"
        if d.get("postgres_dsn"):
            d["postgres_dsn"] = "***"
        for tenant in d.get("tenants", []):
            tenant["api_key"] = "***"
        return d


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _from_env() -> dict[str, Any]:
    e = os.environ.get
    data: dict[str, Any] = {}

    def put(section: str | None, key: str, value: Any) -> None:
        if value is None or value == "":
            return
        if section is None:
            data[key] = value
        else:
            data.setdefault(section, {})[key] = value

    put(None, "db_path", e("MEMRY_DB_PATH"))
    put(None, "backend", e("MEMRY_BACKEND"))
    put(None, "postgres_dsn", e("MEMRY_POSTGRES_DSN"))
    put(None, "default_user_id", e("MEMRY_DEFAULT_USER"))
    put(None, "api_key", e("MEMRY_API_KEY"))
    tenants_json = e("MEMRY_TENANTS")
    if tenants_json:
        try:
            data["tenants"] = json.loads(tenants_json)
        except json.JSONDecodeError:
            pass

    put("llm", "provider", e("MEMRY_LLM_PROVIDER"))
    put("llm", "model", e("MEMRY_LLM_MODEL"))
    put("llm", "api_key", e("MEMRY_LLM_API_KEY"))
    put("llm", "base_url", e("MEMRY_LLM_BASE_URL"))
    put("llm", "effort", e("MEMRY_LLM_EFFORT"))

    put("embedding", "provider", e("MEMRY_EMBEDDING_PROVIDER"))
    put("embedding", "model", e("MEMRY_EMBEDDING_MODEL"))
    put("embedding", "api_key", e("MEMRY_EMBEDDING_API_KEY"))
    put("embedding", "base_url", e("MEMRY_EMBEDDING_BASE_URL"))
    return data


def _autodetect_providers(cfg: Config) -> Config:
    """Upgrade the zero-config defaults when well-known API keys are present."""
    env = os.environ
    if cfg.llm.provider == "none":
        if env.get("ANTHROPIC_API_KEY"):
            cfg.llm.provider = "anthropic"
        elif env.get("OPENAI_API_KEY"):
            cfg.llm.provider = "openai"
    if cfg.embedding.provider == "hash":
        if env.get("OPENAI_API_KEY"):
            cfg.embedding.provider = "openai"
        elif env.get("VOYAGE_API_KEY"):
            cfg.embedding.provider = "voyage"
    return cfg
