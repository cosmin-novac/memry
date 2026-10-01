from __future__ import annotations

from memry.backends.local import LocalBackend
from memry.models import Memory, Scope


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
        # the legacy index is rebuilt, and the live counter agrees with it
        assert [t.normalized for t in reopened.list_topics(Scope(user_id="ada"))] == ["old"]
        assert reopened.topic_mention_counts(Scope(user_id="ada")) == [
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


def test_an_old_stores_synthetic_parents_widen_no_filter_and_are_left_as_they_were(tmp_path):
    """Synthetic parent tags are gone: a store that recorded some (a parent
    topic, its hierarchy edges, its record) opens, a filter on the parent
    reaches nothing, a filter on a tag reaches the memories filed under it,
    and a tag edit works. The old rows are left where they are."""
    path = tmp_path / "synthetic-parents.db"
    backend = LocalBackend(str(path))
    backend.insert_memory(Memory(content="Ada runs", user_id="ada", categories=["running"]))
    backend.insert_memory(Memory(content="Ada sleeps", user_id="ada", categories=["sleep"]))
    with backend._lock:
        db = backend._db
        db.executescript("""
            CREATE TABLE IF NOT EXISTS topic_relations (
                id TEXT PRIMARY KEY, broader_topic_id TEXT NOT NULL,
                narrower_topic_id TEXT NOT NULL, user_id TEXT,
                provenance TEXT NOT NULL DEFAULT 'synthetic', created_at TEXT NOT NULL,
                UNIQUE (broader_topic_id, narrower_topic_id));
            CREATE TABLE IF NOT EXISTS synthetic_tags (
                id TEXT PRIMARY KEY, tag TEXT NOT NULL, user_id TEXT,
                source_tags TEXT NOT NULL, created_at TEXT NOT NULL);
        """)
        db.execute("INSERT INTO topics (id, name, normalized, user_id, provenance, created_at, "
                   "updated_at) VALUES ('parent', 'health', 'health', 'ada', 'synthetic', "
                   "'2026-01-01', '2026-01-01')")
        for child in db.execute("SELECT id FROM topics WHERE normalized IN ('running', 'sleep')"):
            db.execute("INSERT INTO topic_relations (id, broader_topic_id, narrower_topic_id, "
                       "user_id, created_at) VALUES (?, 'parent', ?, 'ada', '2026-01-01')",
                       (f"edge-{child['id']}", child["id"]))
        db.execute("INSERT INTO synthetic_tags (id, tag, user_id, source_tags, created_at) "
                   "VALUES ('s1', 'health', 'ada', '[\"running\", \"sleep\"]', '2026-01-01')")
        db.commit()
    backend.close()

    reopened = LocalBackend(str(path))
    try:
        ada = Scope(user_id="ada")
        assert reopened.list_memories(ada, categories=["health"]) == []
        assert [m.content for m in reopened.list_memories(ada, categories=["running"])] == [
            "Ada runs"]
        assert reopened.topic_mention_counts(ada) == [
            {"category": "running", "count": 1}, {"category": "sleep", "count": 1}]
        assert reopened.retag_topics(ada, {"running"}, "jogging") == 1
        assert [m.content for m in reopened.list_memories(ada, categories=["jogging"])] == [
            "Ada runs"]
        with reopened._lock:
            left = [reopened._db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    for table in ("topic_relations", "synthetic_tags")]
        assert left == [2, 1]
    finally:
        reopened.close()


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


# ---------------------- tags decided by a calibrated judge, as entity pairs
def _tag_judge(same: float):
    from memry.providers.decisions import Answer, Answers, NoneDecider

    class Judge(NoneDecider):
        name = "stub"
        available = True
        calibrated = True
        tag_merge_probability = 0.55

        def __init__(self):
            self.states = []

        def decide(self, state, questions):
            if "tag" not in questions:
                return Answers({})
            self.states.append(state)
            return Answers({"tag": Answer("same" if same >= 0.5 else "different",
                                          {"same": same, "different": 1 - same}, 0.9, True)})

    return Judge()


def _weekly(store):
    """The weekly pass's entity pairs: raised, then compared."""
    store.resolve_entities(user_id="ada")


def _tagged(store, tag, times):
    for i in range(times):
        store.add(f"{tag} fact {i}", user_id="ada", infer=False, categories=[tag])


def test_a_judged_tag_pair_merges_into_the_more_used_tag(verbatim_store):
    verbatim_store.decider = _tag_judge(0.9)
    _tagged(verbatim_store, "quality assurance", 5)
    _tagged(verbatim_store, "qa", 1)
    _weekly(verbatim_store)
    assert verbatim_store.categories(user_id="ada") == [
        {"category": "quality assurance", "count": 6}]


def test_a_tag_pair_under_the_threshold_stays(verbatim_store):
    verbatim_store.decider = _tag_judge(0.5)
    _tagged(verbatim_store, "quality assurance", 5)
    _tagged(verbatim_store, "qa", 1)
    _weekly(verbatim_store)
    assert len(verbatim_store.categories(user_id="ada")) == 2


def test_without_a_calibrated_judge_only_formatting_merges(verbatim_store):
    """Any other pair of tags waits for a person, as a merge proposal."""
    _tagged(verbatim_store, "quality assurance", 5)
    _tagged(verbatim_store, "qa", 1)
    _weekly(verbatim_store)
    assert len(verbatim_store.categories(user_id="ada")) == 2
    [pair] = verbatim_store.proposals_for_a_person("ada")
    assert {verbatim_store.backend.get_entity(e).name for e in (pair.entity_a, pair.entity_b)} \
        == {"qa", "quality assurance"}


def test_the_judge_is_told_which_tags_name_an_entity(verbatim_store):
    """Without it "memry" read as a typo of "memory"."""
    from memry.models import Entity

    judge = _tag_judge(0.1)
    verbatim_store.decider = judge
    verbatim_store.backend.insert_entity(Entity(name="Memry", normalized="memry",
                                                entity_type="product", user_id="ada"))
    _tagged(verbatim_store, "memory", 5)
    _tagged(verbatim_store, "memry", 5)
    _weekly(verbatim_store)
    assert judge.states and all('a product named "Memry"' in s for s in judge.states)
    assert all("memry fact 4" in s and "memory fact 4" in s for s in judge.states)
    assert len(verbatim_store.categories(user_id="ada")) == 2


def test_a_tag_pair_is_compared_when_found_and_once_more_at_10_memories(verbatim_store):
    judge = _tag_judge(0.3)
    verbatim_store.decider = judge
    added = {"quality assurance": 0, "qa": 0}

    def tag(name, times):
        for _ in range(times):
            verbatim_store.add(f"{name} fact {added[name]}", user_id="ada", infer=False,
                               categories=[name])
            added[name] += 1

    tag("quality assurance", 5)
    tag("qa", 1)
    _weekly(verbatim_store)
    _weekly(verbatim_store)
    assert len(judge.states) == 2  # both orders, once
    tag("quality assurance", 5)
    _weekly(verbatim_store)
    assert len(judge.states) == 2  # "qa" is still on one memory
    tag("qa", 9)
    _weekly(verbatim_store)
    _weekly(verbatim_store)
    assert len(judge.states) == 4
    tag("qa", 40)
    _weekly(verbatim_store)
    assert len(judge.states) == 4  # never after


def test_tags_that_only_share_a_word_are_not_compared(verbatim_store):
    """Tags are short phrases that share words across related subjects."""
    judge = _tag_judge(0.9)
    verbatim_store.decider = judge
    _tagged(verbatim_store, "art assets", 2)
    _tagged(verbatim_store, "art direction", 2)
    _weekly(verbatim_store)
    assert judge.states == []


def test_the_tag_questions_answers_from_before_are_carried_to_their_pairs(verbatim_store):
    """Before tags were entity pairs the tag question's funnel kept the step
    each pair of tags was compared at (upkeep "tag_pairs"), and the Upkeep
    list of tags that looked like one subject kept the pairs a person kept
    apart. Each becomes the pair's merge proposal, so the weekly pass asks
    none of them again: compared at its step, or kept apart. A pair whose tag
    is gone is dropped, and both lists go. A pair never compared is asked."""
    from memry.models import Scope

    judge = _tag_judge(0.3)
    verbatim_store.decider = judge
    for tag, times in (("quality assurance", 5), ("qa", 1), ("tech", 3), ("technical", 3),
                       ("cologne", 2), ("colonge", 1)):
        _tagged(verbatim_store, tag, times)
    verbatim_store._upkeep_set("tag_pairs", "ada", {"qa\nquality assurance": 1, "gone\nqa": 1})
    verbatim_store._upkeep_set("tag_split:ignored", "ada", [["tech", "technical"]])
    _weekly(verbatim_store)
    assert len(judge.states) == 2 and all('"colonge"' in state for state in judge.states)

    def named(proposal):
        return tuple(sorted(verbatim_store.backend.get_entity(e).name
                            for e in (proposal.entity_a, proposal.entity_b)))

    pairs = {named(p): (p.status, p.compared_step)
             for p in verbatim_store.backend.list_proposals(Scope(user_id="ada"), status=None)}
    assert pairs == {("qa", "quality assurance"): ("proposed", 1),
                     ("tech", "technical"): ("rejected", 0),
                     ("cologne", "colonge"): ("proposed", 1)}
    assert verbatim_store._upkeep_get("tag_pairs", "ada", None) is None
    assert verbatim_store._upkeep_get("tag_split:ignored", "ada", None) is None
    _weekly(verbatim_store)
    assert len(judge.states) == 2


def test_without_a_judge_only_tags_spelled_alike_are_raised(verbatim_store):
    """Tags close only in meaning ("travel" and "trips") are raised where a
    judge will answer them; without one Memry raises the obvious cases only,
    tags spelled alike ("hepatology" and "hepatolgy"), for a person."""
    from memry.intelligence.entities import propose_same_name_duplicates
    from memry.models import Scope

    for tag in ("travel", "trips", "hepatology", "hepatolgy"):
        _tagged(verbatim_store, tag, 1)

    def embed(names):  # "travel" and "trips" mean one thing, the rest nothing alike
        return [[1.0, 0.0, 0.0] if n in ("travel", "trips") else
                [0.0, 1.0, 0.0] if n == "hepatology" else [0.0, 0.0, 1.0] for n in names]

    def raised(decider):
        scope = Scope(user_id="ada")
        with verbatim_store.backend._lock:
            verbatim_store.backend._db.execute("DELETE FROM entity_proposals")
            verbatim_store.backend._db.commit()
        propose_same_name_duplicates(backend=verbatim_store.backend, scope=scope,
                                     decider=decider, embed=embed)
        return {tuple(sorted(verbatim_store.backend.get_entity(e).name
                             for e in (p.entity_a, p.entity_b)))
                for p in verbatim_store.backend.list_proposals(scope, status=None)}

    assert raised(None) == {("hepatolgy", "hepatology")}
    assert raised(_tag_judge(0.3)) == {("hepatolgy", "hepatology"), ("travel", "trips")}
