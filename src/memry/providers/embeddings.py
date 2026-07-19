"""Embedding providers.

``model_id`` is stored on every memory row so mixed-provider databases stay
consistent: vector search only compares embeddings produced by the currently
configured model, and ``memry reindex`` re-embeds everything after a
provider switch.

The ``hash`` provider is a deterministic, dependency-free local embedder
(feature-hashed word/character n-grams, L2-normalized). It is not a semantic
model - it provides fuzzy lexical similarity so that reconciliation and vector
search work out of the box with zero API keys. FTS5 BM25 keyword search does
the heavy lifting in that mode.
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod

import httpx

from ..config import EmbeddingConfig


class Embedder(ABC):
    name: str = "embedder"
    dimensions: int | None = None

    @property
    def model_id(self) -> str:
        return f"{self.name}:{self._model}"

    _model: str = ""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns one vector per input."""


class NoneEmbedder(Embedder):
    """No embeddings: retrieval degrades to keyword (BM25) + recency + importance."""

    name = "none"
    dimensions = None
    _model = "none"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[] for _ in texts]


_TOKEN_RE = re.compile(r"[a-z0-9]+")


class HashEmbedder(Embedder):
    """Deterministic local embedding via feature hashing (no model download)."""

    name = "hash"

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions
        self._model = f"v1-{dimensions}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _features(self, text: str) -> list[str]:
        words = _TOKEN_RE.findall(text.lower())
        feats: list[str] = list(words)
        feats += [f"{a}_{b}" for a, b in zip(words, words[1:])]
        for w in words:
            padded = f"^{w}$"
            feats += [padded[i : i + 3] for i in range(len(padded) - 2)]
        return feats

    def _embed_one(self, text: str) -> list[float]:
        dims = self.dimensions or 256
        vec = [0.0] * dims
        for feat in self._features(text):
            h = int.from_bytes(
                hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest(), "little"
            )
            sign = 1.0 if (h >> 62) & 1 else -1.0
            vec[h % dims] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec


class OpenAIEmbedder(Embedder):
    name = "openai"

    def __init__(self, cfg: EmbeddingConfig) -> None:
        import os

        self._model = cfg.resolved_model()
        self.dimensions = cfg.dimensions or 1536
        self.base_url = (cfg.base_url or "https://api.openai.com").rstrip("/")
        self.api_key = cfg.api_key or os.environ.get("OPENAI_API_KEY", "")
        self.timeout = cfg.timeout

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = httpx.post(
            f"{self.base_url}/v1/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self._model, "input": texts},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = sorted(resp.json()["data"], key=lambda d: d["index"])
        return [d["embedding"] for d in data]


class OllamaEmbedder(Embedder):
    name = "ollama"

    def __init__(self, cfg: EmbeddingConfig) -> None:
        self._model = cfg.resolved_model()
        self.dimensions = cfg.dimensions or 768
        self.base_url = (cfg.base_url or "http://localhost:11434").rstrip("/")
        self.timeout = cfg.timeout

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = httpx.post(
            f"{self.base_url}/api/embed",
            json={"model": self._model, "input": texts},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]


class VoyageEmbedder(Embedder):
    """Voyage AI - Anthropic's recommended embeddings partner."""

    name = "voyage"

    def __init__(self, cfg: EmbeddingConfig) -> None:
        import os

        self._model = cfg.resolved_model()
        self.dimensions = cfg.dimensions or 1024
        self.base_url = (cfg.base_url or "https://api.voyageai.com").rstrip("/")
        self.api_key = cfg.api_key or os.environ.get("VOYAGE_API_KEY", "")
        self.timeout = cfg.timeout

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = httpx.post(
            f"{self.base_url}/v1/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self._model, "input": texts},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = sorted(resp.json()["data"], key=lambda d: d["index"])
        return [d["embedding"] for d in data]


def build_embedder(cfg: EmbeddingConfig) -> Embedder:
    if cfg.provider == "openai":
        return OpenAIEmbedder(cfg)
    if cfg.provider == "ollama":
        return OllamaEmbedder(cfg)
    if cfg.provider == "voyage":
        return VoyageEmbedder(cfg)
    if cfg.provider == "hash":
        return HashEmbedder(cfg.dimensions or 256)
    return NoneEmbedder()
