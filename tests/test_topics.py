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

def test_plural_topic_is_merged_automatically_on_write(verbatim_store):
    verbatim_store.add("Ada likes vegetables", user_id="ada", infer=False, categories=["foods"])
    verbatim_store.add("Ada plans meals", user_id="ada", infer=False, categories=["food"])

    assert verbatim_store.categories(user_id="ada") == [
        {"category": "food", "count": 2}
    ]
    assert {
        tuple(memory.categories)
        for memory in verbatim_store.get_all(user_id="ada", limit=10)
    } == {("food",)}


def test_existing_plural_topics_are_merged_by_maintenance(verbatim_store):
    backend = verbatim_store.backend
    backend.insert_memory(
        Memory(content="one", user_id="ada", categories=["project"])
    )
    backend.insert_memory(
        Memory(content="two", user_id="ada", categories=["projects"])
    )

    assert verbatim_store.merge_obvious_topics(user_id="ada") == {
        "groups_merged": 1,
        "memories_changed": 1,
    }
    assert verbatim_store.categories(user_id="ada") == [
        {"category": "project", "count": 2}
    ]


# ---------------------------------------- one subject written two ways
def test_a_company_tag_with_its_legal_form_joins_the_one_without(verbatim_store):
    verbatim_store.add("Invoices go to one inbox", user_id="ada", infer=False,
                       categories=["fundation"])
    verbatim_store.add("Liability insurance renewed", user_id="ada", infer=False,
                       categories=["fundation gmbh"])
    assert verbatim_store.categories(user_id="ada") == [
        {"category": "fundation", "count": 2}
    ]


def test_a_domain_tag_joins_its_name_only_when_the_store_knows_the_name(verbatim_store):
    from memry.models import Entity

    store = verbatim_store
    store.add("Bildy makes pictures", user_id="ada", infer=False, categories=["bildy"])
    store.add("Terms use German law", user_id="ada", infer=False, categories=["bildy.ai"])
    assert {c["category"] for c in store.categories(user_id="ada")} == {"bildy", "bildy.ai"}

    store.backend.insert_entity(Entity(name="Bildy", normalized="bildy",
                                       entity_type="product", user_id="ada"))
    store.merge_obvious_topics(user_id="ada")
    assert store.categories(user_id="ada") == [{"category": "bildy", "count": 2}]


def test_the_rules_leave_real_tags_that_are_one_letter_apart_alone():
    """From a real store of 417 tags: a rule on "one letter apart" would merge
    three pairs that are different subjects."""
    from memry.intelligence.clustering import obvious_canonical_merges, swapped_letter_typos

    tags = [{"category": name, "count": count} for name, count in (
        ("finance", 22), ("yfinance", 1), ("memory", 6), ("memry", 13),
        ("preference", 20), ("reference", 3), ("cologne", 11), ("colonge", 1),
        ("three.js", 15), ("character.ai", 2), ("character", 9),
    )]
    assert obvious_canonical_merges(tags) == []
    assert swapped_letter_typos(tags) == [("colonge", "cologne")]


def _tag_judge(same: bool):
    from memry.providers.decisions import Answer, Answers, NoneDecider

    class Judge(NoneDecider):
        name = "stub"
        available = True

        def decide(self, state, questions):
            return Answers({key: Answer(1.0 if same else 0.0, {}, 0.9, True)
                            for key in questions})

    return Judge()


def test_a_swapped_letter_typo_merges_when_the_judge_agrees(verbatim_store):
    store = verbatim_store
    store.decider = _tag_judge(same=True)
    for i in range(5):
        store.add(f"Office fact {i}", user_id="ada", infer=False, categories=["cologne"])
    store.add("Mother's flat", user_id="ada", infer=False, categories=["colonge"])
    store.merge_obvious_topics(user_id="ada")
    assert store.categories(user_id="ada") == [{"category": "cologne", "count": 6}]


def test_a_swapped_letter_pair_stays_when_the_judge_disagrees(verbatim_store):
    store = verbatim_store
    store.decider = _tag_judge(same=False)
    for i in range(5):
        store.add(f"Study {i}", user_id="ada", infer=False, categories=["causal"])
    store.add("Friday dress code", user_id="ada", infer=False, categories=["casual"])
    store.merge_obvious_topics(user_id="ada")
    assert {c["category"] for c in store.categories(user_id="ada")} == {"causal", "casual"}
