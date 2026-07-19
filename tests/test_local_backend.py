from __future__ import annotations

from memry.backends.local import LocalBackend
from memry.models import Episode, Memory, MemoryEvent, Scope
from memry.providers.embeddings import HashEmbedder


def make_backend() -> LocalBackend:
    return LocalBackend(":memory:")


def test_insert_get_list_scoping():
    b = make_backend()
    m1 = b.insert_memory(Memory(content="likes coffee", user_id="ada"))
    b.insert_memory(Memory(content="likes tea", user_id="bob"))

    assert b.get_memory(m1.id).content == "likes coffee"
    ada = b.list_memories(Scope(user_id="ada"))
    assert [m.content for m in ada] == ["likes coffee"]
    everyone = b.list_memories(Scope())
    assert len(everyone) == 2


def test_keyword_search_bm25():
    b = make_backend()
    b.insert_memory(Memory(content="Ada prefers TypeScript strict mode", user_id="ada"))
    b.insert_memory(Memory(content="Ada lives in Berlin", user_id="ada"))

    hits = b.keyword_search("typescript", Scope(user_id="ada"))
    assert len(hits) == 1
    assert "TypeScript" in hits[0][0].content


def test_vector_search_ranks_similar_first():
    b = make_backend()
    emb = HashEmbedder(128)
    texts = ["the user lives in berlin germany", "the user has a cat named miso"]
    for text in texts:
        vec = emb.embed([text])[0]
        b.insert_memory(
            Memory(content=text, user_id="ada", embedding_model=emb.model_id), vec
        )
    query = emb.embed(["which city in germany does the user live in"])[0]
    hits = b.vector_search(query, emb.model_id, Scope(user_id="ada"), limit=2)
    assert hits[0][0].content.startswith("the user lives in berlin")
    assert hits[0][1] > hits[1][1]


def test_vector_search_filters_by_embedding_model():
    b = make_backend()
    emb = HashEmbedder(128)
    vec = emb.embed(["hello world"])[0]
    b.insert_memory(Memory(content="hello world", embedding_model="other:model"), vec)
    assert b.vector_search(vec, emb.model_id, Scope()) == []


def test_invalidate_hides_from_default_views():
    b = make_backend()
    m = b.insert_memory(Memory(content="lives in munich", user_id="ada"))
    b.invalidate_memory(m.id, superseded_by="xyz")

    assert b.list_memories(Scope(user_id="ada")) == []
    assert b.keyword_search("munich", Scope(user_id="ada")) == []
    all_rows = b.list_memories(Scope(user_id="ada"), include_invalid=True)
    assert len(all_rows) == 1
    assert all_rows[0].invalid_at is not None
    assert all_rows[0].superseded_by == "xyz"


def test_update_content_keeps_fts_in_sync():
    b = make_backend()
    m = b.insert_memory(Memory(content="works at siemens", user_id="ada"))
    b.update_memory(m.id, content="works at asml")

    assert b.keyword_search("siemens", Scope(user_id="ada")) == []
    assert len(b.keyword_search("asml", Scope(user_id="ada"))) == 1


def test_episodes_and_events():
    b = make_backend()
    b.add_episodes([Episode(content="hello", user_id="ada")])
    assert b.list_episodes(Scope(user_id="ada"))[0].content == "hello"

    b.add_event(MemoryEvent(memory_id="m1", event="ADD", new_content="x"))
    b.add_event(MemoryEvent(memory_id="m1", event="UPDATE", old_content="x", new_content="y"))
    events = b.history("m1")
    assert [e.event for e in events] == ["ADD", "UPDATE"]


def test_stats_and_reset():
    b = make_backend()
    b.insert_memory(Memory(content="a", user_id="ada"))
    m = b.insert_memory(Memory(content="b", user_id="ada"))
    b.invalidate_memory(m.id)
    stats = b.stats()
    assert stats["active_memories"] == 1
    assert stats["invalidated_memories"] == 1
    b.reset()
    assert b.stats()["active_memories"] == 0
