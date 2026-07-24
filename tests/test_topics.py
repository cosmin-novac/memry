from __future__ import annotations

from memry.backends.local import LocalBackend
from memry.models import Memory, Scope, SyntheticTag, Topic, TopicRelation


def test_topics_dual_write_filter_and_count(verbatim_store):
    store = verbatim_store
    store.add("Ada runs", user_id="ada", infer=False, categories=["Health", "running"])
    store.add("Ada budgets", user_id="ada", infer=False, categories=["finance"])
    store.add("Bob runs", user_id="bob", infer=False, categories=["health"])

    assert [m.content for m in store.get_all(user_id="ada", categories=["HEALTH"])] == [
        "Ada runs"
    ]
    assert store.categories(user_id="ada") == [
        {"category": "finance", "count": 1},
        {"category": "health", "count": 1},
        {"category": "running", "count": 1},
    ]
    topics = store.backend.list_topics(Scope(user_id="ada"))
    assert [topic.normalized for topic in topics] == ["finance", "health", "running"]


def test_topic_links_follow_compatibility_updates(verbatim_store):
    store = verbatim_store
    memory = store.add(
        "Ada runs", user_id="ada", infer=False, categories=["running", "fitness"]
    )
    memory_id = memory.actions[0].memory_id

    updated = store.update(memory_id, categories=["health"])
    assert updated.categories == ["health"]
    assert store.get_all(user_id="ada", categories=["running"]) == []
    assert [m.id for m in store.get_all(user_id="ada", categories=["health"])] == [memory_id]


def test_existing_categories_backfill_once(tmp_path):
    path = tmp_path / "topics.db"
    backend = LocalBackend(str(path))
    memory = backend.insert_memory(Memory(content="legacy", user_id="ada", categories=["old"]))
    with backend._lock:
        backend._db.execute("DELETE FROM memory_topics")
        backend._db.execute("DELETE FROM topics")
        backend._db.execute("DELETE FROM meta WHERE key = 'schema:topics:v1'")
        backend._db.commit()
    backend.close()

    reopened = LocalBackend(str(path))
    try:
        assert [m.id for m in reopened.list_memories(
            Scope(user_id="ada"), categories=["OLD"]
        )] == [memory.id]
        assert reopened.topic_counts(Scope(user_id="ada")) == [
            {"category": "old", "count": 1}
        ]
    finally:
        reopened.close()


def test_hard_delete_removes_topic_links(verbatim_store):
    backend = verbatim_store.backend
    memory = backend.insert_memory(Memory(content="temporary", categories=["ephemeral"]))
    assert backend.delete_memory(memory.id)
    count = backend._db.execute(
        "SELECT COUNT(*) FROM memory_topics WHERE memory_id = ?", (memory.id,)
    ).fetchone()[0]
    assert count == 0

def test_parent_topic_expands_at_query_time_without_copying(verbatim_store):
    store = verbatim_store
    store.add("Ada runs", user_id="ada", infer=False, categories=["running"])
    store.add("Ada sleeps", user_id="ada", infer=False, categories=["sleep"])
    backend = store.backend
    topics = {topic.normalized: topic for topic in backend.list_topics(Scope(user_id="ada"))}
    parent = backend.upsert_topic(
        Topic(name="health", normalized="health", user_id="ada", provenance="synthetic")
    )
    for child in (topics["running"], topics["sleep"]):
        backend.add_topic_relation(
            TopicRelation(
                broader_topic_id=parent.id,
                narrower_topic_id=child.id,
                user_id="ada",
            )
        )

    matches = store.get_all(user_id="ada", categories=["health"], limit=20)
    assert {memory.content for memory in matches} == {"Ada runs", "Ada sleeps"}
    assert all("health" not in memory.categories for memory in matches)
    assert {row["category"]: row["count"] for row in store.categories(user_id="ada")}["health"] == 2


def test_legacy_copied_synthetic_tags_migrate_to_edges(tmp_path):
    path = tmp_path / "synthetic-topic-migration.db"
    backend = LocalBackend(str(path))
    memory = backend.insert_memory(
        Memory(content="Ada runs", user_id="ada", categories=["running", "health"])
    )
    backend.record_synthetic_tag(
        SyntheticTag(tag="health", source_tags=["running"], user_id="ada")
    )
    with backend._lock:
        backend._db.execute("DELETE FROM meta WHERE key = 'schema:topic-relations:v1'")
        backend._db.commit()
    backend.close()

    reopened = LocalBackend(str(path))
    try:
        stored = reopened.get_memory(memory.id)
        assert stored.categories == ["running"]
        assert [row.id for row in reopened.list_memories(
            Scope(user_id="ada"), categories=["health"]
        )] == [memory.id]
    finally:
        reopened.close()

def test_topic_edits_preserve_and_remove_hierarchy(verbatim_store):
    store = verbatim_store
    store.add("Ada runs", user_id="ada", infer=False, categories=["running"])
    backend = store.backend
    child = backend.list_topics(Scope(user_id="ada"))[0]
    parent = backend.upsert_topic(
        Topic(name="health", normalized="health", user_id="ada", provenance="synthetic")
    )
    backend.add_topic_relation(
        TopicRelation(
            broader_topic_id=parent.id,
            narrower_topic_id=child.id,
            user_id="ada",
        )
    )
    backend.record_synthetic_tag(
        SyntheticTag(tag="health", source_tags=["running"], user_id="ada")
    )

    assert store.rename_tag("running", "jogging", user_id="ada") == 1
    assert [memory.content for memory in store.get_all(
        user_id="ada", categories=["health"]
    )] == ["Ada runs"]

    assert store.rename_tag("health", "wellness", user_id="ada") == 0
    assert [memory.content for memory in store.get_all(
        user_id="ada", categories=["wellness"]
    )] == ["Ada runs"]
    assert store.synthetic_tags(user_id="ada") == []

    assert store.delete_tag("wellness", user_id="ada") == 0
    assert store.get_all(user_id="ada", categories=["wellness"]) == []
    assert [memory.content for memory in store.get_all(
        user_id="ada", categories=["jogging"]
    )] == ["Ada runs"]
