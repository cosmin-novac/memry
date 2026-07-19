from __future__ import annotations

from datetime import datetime, timedelta, timezone

from memry.backends.local import LocalBackend
from memry.config import RetrievalConfig
from memry.models import Memory, Scope
from memry.providers.embeddings import HashEmbedder, NoneEmbedder
from memry.retrieval import hybrid_search, recency_score


def seeded_backend(emb) -> LocalBackend:
    b = LocalBackend(":memory:")
    texts = [
        "the user lives in berlin",
        "the user works at northwind as a data engineer",
        "the user prefers typescript strict mode",
    ]
    for text in texts:
        vec = emb.embed([text])[0] if emb.dimensions else None
        b.insert_memory(
            Memory(content=text, user_id="ada", embedding_model=emb.model_id if vec else None),
            vec,
        )
    return b


def test_hybrid_combines_vector_and_keyword():
    emb = HashEmbedder(128)
    b = seeded_backend(emb)
    results = hybrid_search(
        backend=b, embedder=emb, query="which city does the user live in",
        scope=Scope(user_id="ada"), limit=3,
    )
    assert results[0].memory.content == "the user lives in berlin"
    assert "fused" in results[0].signals


def test_keyword_only_mode_with_none_embedder():
    emb = NoneEmbedder()
    b = seeded_backend(emb)
    results = hybrid_search(
        backend=b, embedder=emb, query="typescript",
        scope=Scope(user_id="ada"), limit=3,
    )
    assert results
    assert results[0].memory.content == "the user prefers typescript strict mode"
    assert "vector" not in results[0].signals


def test_recency_boost_breaks_ties():
    emb = NoneEmbedder()
    b = LocalBackend(":memory:")
    now = datetime.now(timezone.utc)
    old_ts = (now - timedelta(days=300)).isoformat(timespec="seconds")

    fresh = Memory(content="user enjoys hiking on weekends")
    stale = Memory(content="user enjoys hiking in mountains")
    stale.created_at = stale.updated_at = old_ts
    b.insert_memory(fresh)
    b.insert_memory(stale)

    results = hybrid_search(
        backend=b, embedder=emb, query="hiking enjoys user",
        scope=Scope(), limit=2, cfg=RetrievalConfig(recency_weight=0.5),
    )
    assert results[0].memory.id == fresh.id
    assert results[0].signals["recency"] > results[1].signals["recency"]


def test_recency_score_halves_at_half_life():
    now = datetime.now(timezone.utc)
    m = Memory(content="x")
    m.updated_at = (now - timedelta(days=30)).isoformat(timespec="seconds")
    score = recency_score(m, now, half_life_days=30)
    assert abs(score - 0.5) < 0.01


def test_empty_store_returns_nothing():
    emb = HashEmbedder(64)
    b = LocalBackend(":memory:")
    assert hybrid_search(backend=b, embedder=emb, query="anything", scope=Scope(), limit=5) == []
