from __future__ import annotations

import math

from memry.config import EmbeddingConfig
from memry.providers.embeddings import HashEmbedder, NoneEmbedder, build_embedder


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))  # vectors are L2-normalized


def test_hash_embedder_deterministic():
    e1, e2 = HashEmbedder(128), HashEmbedder(128)
    assert e1.embed(["the quick brown fox"]) == e2.embed(["the quick brown fox"])
    assert e1.model_id == "hash:v1-128"


def test_hash_embedder_normalized():
    vec = HashEmbedder(256).embed(["some text to embed"])[0]
    assert math.isclose(math.sqrt(sum(v * v for v in vec)), 1.0, rel_tol=1e-6)


def test_hash_embedder_similarity_ordering():
    e = HashEmbedder(256)
    query, close, far = e.embed(
        [
            "user lives in berlin germany",
            "the user lives in berlin",
            "recipe for chocolate cake with strawberries",
        ]
    )
    assert cosine(query, close) > cosine(query, far)


def test_hash_embedder_empty_text():
    vec = HashEmbedder(64).embed([""])[0]
    assert len(vec) == 64
    assert all(v == 0.0 for v in vec)


def test_none_embedder():
    e = NoneEmbedder()
    assert e.dimensions is None
    assert e.embed(["a", "b"]) == [[], []]


def test_build_embedder_dispatch():
    assert isinstance(build_embedder(EmbeddingConfig(provider="hash")), HashEmbedder)
    assert isinstance(build_embedder(EmbeddingConfig(provider="none")), NoneEmbedder)
    openai = build_embedder(EmbeddingConfig(provider="openai", api_key="k"))
    assert openai.model_id == "openai:text-embedding-3-small"
    voyage = build_embedder(EmbeddingConfig(provider="voyage", api_key="k"))
    assert voyage.model_id == "voyage:voyage-3.5-lite"
