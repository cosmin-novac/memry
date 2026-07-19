from __future__ import annotations

import pytest

from memry.backends.ann import HAS_USEARCH
from memry.backends.local import LocalBackend
from memry.config import AnnConfig
from memry.models import Memory, Scope
from memry.providers.embeddings import HashEmbedder


# ---------------------------------------------------------------- categories
def seeded(verbatim_store):
    verbatim_store.add("User is vegetarian", user_id="ada", infer=False,
                       categories=["diet", "health"])
    verbatim_store.add("User prefers TypeScript", user_id="ada", infer=False,
                       categories=["tooling"])
    verbatim_store.add("User runs marathons", user_id="ada", infer=False,
                       categories=["health"])
    return verbatim_store


def test_get_all_category_filter(verbatim_store):
    store = seeded(verbatim_store)
    health = store.get_all(user_id="ada", categories=["health"])
    assert {m.content for m in health} == {"User is vegetarian", "User runs marathons"}
    tooling = store.get_all(user_id="ada", categories=["TOOLING"])  # case-insensitive
    assert [m.content for m in tooling] == ["User prefers TypeScript"]
    assert store.get_all(user_id="ada", categories=["nope"]) == []


def test_search_category_filter(verbatim_store):
    store = seeded(verbatim_store)
    hits = store.search("user", user_id="ada", categories=["diet"])
    assert [h.memory.content for h in hits] == ["User is vegetarian"]
    # unfiltered search sees all three
    assert len(store.search("user", user_id="ada")) == 3


# ---------------------------------------------------------------- ANN sidecar
@pytest.mark.skipif(not HAS_USEARCH, reason="usearch not installed")
def test_ann_path_matches_bruteforce():
    emb = HashEmbedder(64)
    ann_backend = LocalBackend(":memory:", ann=AnnConfig(enabled=True, min_rows=1))
    exact_backend = LocalBackend(":memory:", ann=AnnConfig(enabled=False))

    texts = [f"the user visited city number {i} on holiday" for i in range(50)]
    texts += ["the user lives in berlin germany"]
    for backend in (ann_backend, exact_backend):
        for text in texts:
            vec = emb.embed([text])[0]
            backend.insert_memory(
                Memory(content=text, user_id="ada", embedding_model=emb.model_id), vec
            )

    query = emb.embed(["which city in germany does the user live in"])[0]
    ann_hits = ann_backend.vector_search(query, emb.model_id, Scope(user_id="ada"), limit=5)
    exact_hits = exact_backend.vector_search(query, emb.model_id, Scope(user_id="ada"), limit=5)

    assert ann_hits[0][0].content == exact_hits[0][0].content
    assert ann_hits[0][0].content == "the user lives in berlin germany"
    assert ann_backend.stats()["ann"]["active"] is True


@pytest.mark.skipif(not HAS_USEARCH, reason="usearch not installed")
def test_ann_removal_on_invalidate():
    emb = HashEmbedder(64)
    backend = LocalBackend(":memory:", ann=AnnConfig(enabled=True, min_rows=1))
    memories = []
    for i in range(10):
        text = f"fact number {i} about the user"
        vec = emb.embed([text])[0]
        memories.append(
            backend.insert_memory(
                Memory(content=text, user_id="ada", embedding_model=emb.model_id), vec
            )
        )
    before = backend.stats()["ann"]["indexed"]
    backend.invalidate_memory(memories[0].id)
    assert backend.stats()["ann"]["indexed"] == before - 1
    # invalidated memory never comes back from vector search
    query = emb.embed(["fact number 0 about the user"])[0]
    hits = backend.vector_search(query, emb.model_id, Scope(user_id="ada"), limit=10)
    assert all(h[0].id != memories[0].id for h in hits)


@pytest.mark.skipif(not HAS_USEARCH, reason="usearch not installed")
def test_ann_persistence_and_rebuild(tmp_path):
    emb = HashEmbedder(32)
    db = str(tmp_path / "ann.db")
    backend = LocalBackend(db, ann=AnnConfig(enabled=True, min_rows=1))
    for i in range(5):
        text = f"persisted fact {i}"
        backend.insert_memory(
            Memory(content=text, user_id="ada", embedding_model=emb.model_id),
            emb.embed([text])[0],
        )
    backend.close()  # saves the sidecar

    reopened = LocalBackend(db, ann=AnnConfig(enabled=True, min_rows=1))
    query = emb.embed(["persisted fact 3"])[0]
    hits = reopened.vector_search(query, emb.model_id, Scope(user_id="ada"), limit=3)
    assert hits and hits[0][0].content == "persisted fact 3"
    reopened.close()


def test_ann_disabled_still_correct():
    emb = HashEmbedder(32)
    backend = LocalBackend(":memory:", ann=AnnConfig(enabled=False))
    vec = emb.embed(["hello world"])[0]
    backend.insert_memory(
        Memory(content="hello world", user_id="ada", embedding_model=emb.model_id), vec
    )
    hits = backend.vector_search(vec, emb.model_id, Scope(user_id="ada"), limit=1)
    assert hits[0][0].content == "hello world"
