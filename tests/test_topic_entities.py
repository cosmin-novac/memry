"""Tags are entities of type "topic".

A memory's categories stay in its ``categories`` column, which every filter,
backup and export reads. Each category is also a mention of the topic entity of
that tag, so tags, people and products are one kind of thing with one merge
machinery and one set of links.
"""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest
from conftest import FakeLLM, fact, facts_response
from starlette.testclient import TestClient

from memry.config import Config
from memry.models import TOPIC_TYPE, Entity, EntityMention, Memory, Scope
from memry.providers.embeddings import HashEmbedder
from memry.providers.llm import NoneLLM
from memry.store import MemoryStore, _text_hash


@pytest.fixture
def tagged(verbatim_store):
    return verbatim_store


def _topics_of(store, memory_id):
    return sorted(e.name for e in store.backend.entities_of_memory(memory_id, kind="topic"))


def _agree(store, user_id):
    """Every active memory mentions exactly the topic entities its column
    names, each once, under the tag as written."""
    for memory in store.get_all(user_id=user_id, limit=10_000):
        column = sorted({str(c).strip().lower() for c in memory.categories})
        assert _topics_of(store, memory.id) == column, memory.content
        rows = store.backend._db.execute(
            "SELECT em.surface, e.normalized FROM entity_mentions em "
            "JOIN entities e ON e.id = em.entity_id "
            "WHERE em.memory_id = ? AND e.entity_type = 'topic'", (memory.id,)).fetchall()
        assert sorted(r["normalized"] for r in rows) == column, memory.content
        assert all(r["surface"].strip().lower() == r["normalized"] for r in rows)


# ------------------------------------------------------------------ on save
def test_each_saved_category_is_a_mention_of_its_topic_entity(tagged):
    first = tagged.add("Ada runs", user_id="ada", infer=False,
                       categories=["Health", "running"]).actions[0].memory_id
    second = tagged.add("Ada sleeps", user_id="ada", infer=False,
                        categories=["health"]).actions[0].memory_id
    tagged.add("Bob runs", user_id="bob", infer=False, categories=["health"])

    # the column is written exactly as before (a saved tag is lowercased)
    assert tagged.get(first).categories == ["health", "running"]
    topics = tagged.entities(user_id="ada", kind="topic")
    # one per user and normalized tag, named by the normalized tag
    assert sorted((e.name, e.normalized, e.entity_type) for e in topics) == [
        ("health", "health", TOPIC_TYPE), ("running", "running", TOPIC_TYPE)]
    assert all(e.user_id == "ada" for e in topics)
    assert len(tagged.entities(user_id="bob", kind="topic")) == 1
    health = next(e for e in topics if e.name == "health")
    mentions = tagged.backend.entity_mentions(health.id)
    assert sorted((m.memory_id, m.surface) for m in mentions) == sorted(
        [(first, "health"), (second, "health")])
    # a memory written straight to the backend keeps its casing, and so does
    # its mention
    third = tagged.backend.insert_memory(Memory(content="Ada swims", user_id="ada",
                                                categories=["Health"]))
    assert [(m.surface) for m in tagged.backend.entity_mentions(health.id)
            if m.memory_id == third.id] == ["Health"]
    # a tag is not a named thing: the named listings and lookups leave it out
    assert tagged.entities(user_id="ada") == []
    assert tagged.backend.entities_of_memory(first) == []
    assert tagged.backend.find_entity_candidates("health", Scope(user_id="ada")) == []
    _agree(tagged, "ada")


def test_extracted_categories_become_mentions_through_apply_candidates():
    llm = FakeLLM()
    config = Config(db_path=":memory:")
    store = MemoryStore(config, llm=llm, embedder=HashEmbedder(64))
    try:
        llm.queue(
            facts_response(fact("Ada buys oat milk at Lidl", categories=["groceries", "Foods"])),
            json.dumps({"missing": []}),
        )
        memory_id = store.add("I buy oat milk at Lidl", user_id="ada").actions[0].memory_id
        assert _topics_of(store, memory_id) == ["foods", "groceries"]
        # the obvious canonical form applies to topic entity names as to tags
        llm.queue(
            facts_response(fact("Ada cooks pasta on Sundays", categories=["food"])),
            json.dumps({"missing": []}),
        )
        store.add("I cook pasta on Sundays", user_id="ada")
        assert sorted(e.name for e in store.entities(user_id="ada", kind="topic")) == [
            "food", "groceries"]
        assert store.categories(user_id="ada") == [
            {"category": "food", "count": 2}, {"category": "groceries", "count": 1}]
        _agree(store, "ada")
    finally:
        store.close()


def test_the_manual_update_path_keeps_mentions_with_the_column(tagged):
    memory_id = tagged.add("Ada runs", user_id="ada", infer=False,
                           categories=["running"]).actions[0].memory_id
    tagged.update(memory_id, categories=["health", "sport"])
    assert _topics_of(tagged, memory_id) == ["health", "sport"]
    # a content edit replaces the named mentions and keeps the tags'
    tagged.update(memory_id, content="Ada runs every morning")
    assert _topics_of(tagged, memory_id) == ["health", "sport"]
    tagged.update(memory_id, categories=[])
    assert _topics_of(tagged, memory_id) == []
    assert tagged.categories(user_id="ada") == []
    _agree(tagged, "ada")


def test_the_store_and_backend_accept_the_topic_type_and_extraction_does_not_offer_it(tagged):
    from memry.intelligence import extraction
    from memry.models import ENTITY_TYPES

    assert ENTITY_TYPES == (*extraction.ENTITY_TYPES, TOPIC_TYPE)
    schema = extraction.EXTRACTION_SCHEMA["properties"]["facts"]["items"]
    enum = schema["properties"]["entities"]["items"]["properties"]["type"]["enum"]
    assert TOPIC_TYPE not in enum
    entity = tagged.backend.insert_entity(Entity(name="x", user_id="ada"))
    tagged.backend.set_entity_type(entity.id, TOPIC_TYPE)
    assert tagged.backend.get_entity(entity.id).entity_type == TOPIC_TYPE
    with pytest.raises(ValueError):
        tagged.backend.set_entity_type(entity.id, "planet")


def test_the_entity_types_are_defined_once():
    """Extraction's types are the models' named types, so a type added in one
    place is a type in the other, and ``set_entity_type`` accepts it."""
    from memry import models
    from memry.intelligence import extraction

    assert extraction.ENTITY_TYPES is models.NAMED_ENTITY_TYPES
    assert models.ENTITY_TYPES == (*models.NAMED_ENTITY_TYPES, TOPIC_TYPE)


# ----------------------------------------------------------------- counting
def test_categories_count_topic_entities_by_their_active_mentions(tagged):
    tagged.add("a", user_id="ada", infer=False, categories=["work", "diet"])
    gone = tagged.add("b", user_id="ada", infer=False, categories=["work"]).actions[0].memory_id
    tagged.add("c", user_id="ada", infer=False, categories=["Work", "travel"], run_id="r1")
    tagged.add("d", user_id="bob", infer=False, categories=["work"])
    tagged.delete(gone)
    # the legacy tag index is not what is counted
    with tagged.backend._lock:
        tagged.backend._db.execute("DELETE FROM memory_topics")
        tagged.backend._db.execute("DELETE FROM topics")
        tagged.backend._db.commit()
    assert tagged.categories(user_id="ada") == [
        {"category": "work", "count": 2},
        {"category": "diet", "count": 1},
        {"category": "travel", "count": 1},
    ]
    assert tagged.categories(user_id="ada", run_id="r1") == [
        {"category": "travel", "count": 1}, {"category": "work", "count": 1}]
    assert tagged.categories(user_id="bob") == [{"category": "work", "count": 1}]


def test_the_vocabulary_offered_to_extraction_is_the_same_as_before(tagged):
    """Read from topic entities, the vocabulary holds the tags the legacy index
    held, in the same order: most used first, then by name."""
    for i, tags in enumerate([["tax"], ["tax", "home"], ["Home"], ["garden"],
                              ["tax"], ["b-tag"], ["a tag"]]):
        tagged.add(f"note {i}", user_id="ada", infer=False, categories=tags)
    scope = Scope(user_id="ada")
    # the legacy index's direct counts, most used first, then by name
    legacy = [row["category"] for row in tagged.backend._db.execute(
        "SELECT t.normalized AS category, COUNT(DISTINCT mt.memory_id) AS count "
        "FROM topics t JOIN memory_topics mt ON mt.topic_id = t.id "
        "JOIN memories m ON m.id = mt.memory_id "
        "WHERE m.invalid_at IS NULL AND t.user_id = 'ada' AND m.user_id = 'ada' "
        "GROUP BY t.normalized ORDER BY count DESC, category")]
    assert tagged._tag_vocabulary(scope, text="anything") == legacy == [
        "tax", "home", "a tag", "b-tag", "garden"]


def test_list_categories_counts_through_mcp_and_rest():
    from test_servers import call_tool

    from memry.mcp_server import create_server
    from memry.rest import create_app

    store = MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(), embedder=HashEmbedder(64))
    try:
        for content, tags in [("a", ["work", "diet"]), ("b", ["work"]), ("c", ["Work"]),
                              ("d", ["diet"])]:
            store.add(content, user_id="u", infer=False, categories=tags)
        store.add("e", user_id="other", infer=False, categories=["work"])
        forgotten = store.add("f", user_id="u", infer=False, categories=["diet"])
        store.delete(forgotten.actions[0].memory_id)
        # counted from the topic entities, not the legacy tag index
        with store.backend._lock:
            store.backend._db.execute("DELETE FROM memory_topics")
            store.backend._db.commit()
        expected = [{"category": "work", "count": 3}, {"category": "diet", "count": 2}]
        assert call_tool(create_server(store), "list_categories", {"user_id": "u"}) == expected
        with TestClient(create_app(store)) as client:
            assert client.get("/api/v1/categories", params={"user_id": "u"}).json() == expected
    finally:
        store.close()


# ------------------------------------------------------------------ merging
def test_merging_tags_moves_mentions_and_the_column_agrees(tagged):
    a = tagged.add("a", user_id="ada", infer=False,
                   categories=["finance", "budget"]).actions[0].memory_id
    tagged.add("b", user_id="ada", infer=False, categories=["financial"])
    tagged.add("c", user_id="ada", infer=False, categories=["financial", "finance"])
    financial = tagged.backend.topic_entity("financial", Scope(user_id="ada"), create=False)
    finance = tagged.backend.topic_entity("finance", Scope(user_id="ada"), create=False)

    assert tagged.merge_tags(["financial"], "finance", user_id="ada") == 2
    # one merge machinery: the variant is folded into the canonical entity
    assert tagged.backend.get_entity(financial.id).merged_into == finance.id
    assert tagged.backend.resolve_entity_id(financial.id) == finance.id
    assert tagged.backend.count_entity_memories(finance.id) == 3
    assert tagged.categories(user_id="ada")[0] == {"category": "finance", "count": 3}
    assert tagged.get(a).categories == ["finance", "budget"]
    _agree(tagged, "ada")


def _active_topics(store, name):
    return [row["id"] for row in store.backend._db.execute(
        "SELECT id FROM entities WHERE entity_type = 'topic' AND normalized = ? "
        "AND merged_into IS NULL", (name,))]


def test_a_memory_restored_after_a_tag_merge_carries_the_merged_tag(tagged):
    """A merge rewrites the column of invalid memories too: a memory forgotten
    while "taxes" was merged into "tax" comes back filed under "tax", and no
    "taxes" topic comes back with it. (Saved through the backend: a save
    through the store would already file "taxes" under "tax".)"""
    def save(content, tag):
        return tagged.backend.insert_memory(
            Memory(content=content, user_id="ada", categories=[tag])).id

    gone = save("gone", "taxes")
    save("kept", "taxes")
    save("other", "tax")
    tagged.delete(gone)
    tax = tagged.backend.topic_entity("tax", Scope(user_id="ada"), create=False)
    # the count is of the memories in use
    assert tagged.merge_tags(["taxes"], "tax", user_id="ada") == 1
    assert tagged.backend.get_memory(gone).categories == ["tax"]
    assert tagged.unforget(gone)
    assert _active_topics(tagged, "taxes") == []
    assert [e.id for e in tagged.backend.entities_of_memory(gone, kind="topic")] == [tax.id]
    assert tagged.categories(user_id="ada") == [{"category": "tax", "count": 3}]
    _agree(tagged, "ada")


def test_a_column_still_naming_a_merged_tag_mentions_the_tag_it_went_into(tagged):
    """A column written with a tag merged away (a restored backup, an import,
    a memory written straight to the backend) is filed as every write files
    it: the column names the surviving tag, the mention goes to its topic
    instead of bringing the old one back, and so does the filter index, so
    the count and the filter agree. A filter on the name merged away finds
    nothing."""
    a = tagged.backend.insert_memory(Memory(content="a", user_id="ada", categories=["taxes"]))
    b = tagged.backend.insert_memory(Memory(content="b", user_id="ada", categories=["tax"]))
    tax = tagged.backend.topic_entity("tax", Scope(user_id="ada"), create=False)
    assert tagged.merge_tags(["taxes"], "tax", user_id="ada") == 1
    old = tagged.backend.insert_memory(Memory(content="c", user_id="ada", categories=["taxes"]))
    assert old.categories == ["tax"]  # the column names the survivor
    assert tagged.backend.get_memory(old.id).categories == ["tax"]
    _agree(tagged, "ada")
    assert _active_topics(tagged, "taxes") == []
    assert [e.id for e in tagged.backend.entities_of_memory(old.id, kind="topic")] == [tax.id]
    # a lookup reads the active topics; following the tombstone is asked for
    assert tagged.backend.topic_entity("taxes", Scope(user_id="ada"), create=False) is None
    assert tagged.backend.topic_entity("taxes", Scope(user_id="ada"), create=False,
                                       follow_merged=True).id == tax.id
    assert tagged.categories(user_id="ada") == [{"category": "tax", "count": 3}]
    assert {m.id for m in tagged.get_all(user_id="ada", categories=["tax"])} == {
        a.id, b.id, old.id}
    assert tagged.get_all(user_id="ada", categories=["taxes"]) == []
    # deleting the tag takes it off that column too: nothing counts or files under it
    assert tagged.delete_tag("tax", user_id="ada") == 3
    assert tagged.backend.get_memory(old.id).categories == []
    assert tagged.categories(user_id="ada") == []
    assert tagged.get_all(user_id="ada", categories=["tax"]) == []


def _tombstones(store, name):
    return [row["id"] for row in store.backend._db.execute(
        "SELECT id FROM entities WHERE entity_type = 'topic' AND normalized = ? "
        "AND merged_into IS NOT NULL", (name,))]


def test_merging_a_tag_back_into_a_name_merged_away_keeps_its_survivor(tagged):
    """After "tax" went into "taxes", the name "tax" means "taxes": merging
    "taxes" into "tax" merges it into its own survivor, which changes
    nothing, and never makes a fresh topic of the retired name. A column
    saying "tax" afterwards is written as "taxes"."""
    tagged.backend.insert_memory(Memory(content="a", user_id="ada", categories=["tax"]))
    tagged.backend.insert_memory(Memory(content="b", user_id="ada", categories=["taxes"]))
    tagged.merge_tags(["tax"], "taxes", user_id="ada")
    assert tagged.categories(user_id="ada") == [{"category": "taxes", "count": 2}]
    [taxes] = _active_topics(tagged, "taxes")
    assert tagged.merge_tags(["taxes"], "tax", user_id="ada") == 0
    assert tagged.categories(user_id="ada") == [{"category": "taxes", "count": 2}]
    assert _active_topics(tagged, "tax") == [] and _active_topics(tagged, "taxes") == [taxes]
    assert _tombstones(tagged, "tax") and not _tombstones(tagged, "taxes")
    _agree(tagged, "ada")
    later = tagged.backend.insert_memory(Memory(content="c", user_id="ada",
                                                categories=["tax"]))
    assert later.categories == ["taxes"]
    assert [e.id for e in tagged.backend.entities_of_memory(later.id, kind="topic")] == [taxes]
    assert _active_topics(tagged, "tax") == []
    assert tagged.categories(user_id="ada") == [{"category": "taxes", "count": 3}]
    assert len(tagged.get_all(user_id="ada", categories=["taxes"])) == 3
    assert tagged.get_all(user_id="ada", categories=["tax"]) == []
    _agree(tagged, "ada")


def test_a_name_merged_away_is_not_merged_again_through_its_tombstone(tagged):
    """After "taxes" went into "tax", merging "taxes" into "levies" finds no
    active topic named "taxes": nothing is merged, "tax" keeps its name and
    its memories, and no "levies" topic appears."""
    for content, tag in (("a", "taxes"), ("b", "tax")):
        tagged.backend.insert_memory(Memory(content=content, user_id="ada", categories=[tag]))
    assert tagged.merge_tags(["taxes"], "tax", user_id="ada") == 1
    [tax] = _active_topics(tagged, "tax")
    assert tagged.merge_tags(["taxes"], "levies", user_id="ada") == 0
    kept = tagged.backend.get_entity(tax)
    assert (kept.name, kept.normalized, kept.merged_into) == ("tax", "tax", None)
    assert tagged.categories(user_id="ada") == [{"category": "tax", "count": 2}]
    assert _active_topics(tagged, "levies") == [] and _active_topics(tagged, "tax") == [tax]
    _agree(tagged, "ada")


def test_a_tag_merged_into_the_name_of_a_named_thing_goes_into_that_thing(tagged):
    """ "bildy" the tag was found to be "Bildy" the product. Merging "bildy
    app" into "bildy" folds it into the product, as a memory filed under
    "bildy" mentions the product: no topic "bildy" is created."""
    product = tagged.backend.insert_entity(Entity(
        name="Bildy", normalized="bildy", entity_type="product", user_id="ada"))
    tagged.backend.insert_memory(Memory(content="x", user_id="ada", categories=["bildy"]))
    topic = tagged.backend.topic_entity("bildy", Scope(user_id="ada"), create=False)
    assert tagged.backend.merge_entities(product.id, topic.id)
    tagged.backend.insert_memory(Memory(content="y", user_id="ada", categories=["bildy app"]))
    variant = tagged.backend.topic_entity("bildy app", Scope(user_id="ada"), create=False)
    assert tagged.merge_tags(["bildy app"], "bildy", user_id="ada") == 1
    assert tagged.backend.get_entity(variant.id).merged_into == product.id
    assert _active_topics(tagged, "bildy") == []
    assert tagged.backend.count_entity_memories(product.id) == 2


def test_an_update_naming_a_merged_tag_files_it_under_the_topic_it_went_into(tagged):
    """After "taxes" went into "tax", an update (the PATCH route) setting a
    memory's tags to ["taxes"] writes them as a save does: the column says
    "tax", ``categories()`` counts the memory there and a filter on "tax"
    finds it. A filter on "taxes" finds nothing, as it counts nothing."""
    ids = [tagged.backend.insert_memory(Memory(content=content, user_id="ada",
                                               categories=[tag])).id
           for content, tag in (("paid the taxes", "taxes"), ("tax return filed", "tax"),
                                ("tax office letter arrived", "home"))]
    assert tagged.merge_tags(["taxes"], "tax", user_id="ada") == 1
    patched = tagged.update(ids[2], categories=["taxes"])
    assert patched.categories == ["tax"]
    assert tagged.categories(user_id="ada") == [{"category": "tax", "count": 3}]
    assert {m.id for m in tagged.get_all(user_id="ada", categories=["tax"])} == set(ids)
    found = tagged.search("tax office letter", user_id="ada", categories=["tax"], limit=5)
    assert ids[2] in {r.memory.id for r in found}
    assert tagged.search("tax office letter", user_id="ada", categories=["taxes"]) == []
    assert tagged.get_all(user_id="ada", categories=["taxes"]) == []
    _agree(tagged, "ada")


def test_a_merge_into_a_name_merged_away_goes_into_its_survivor(tagged):
    """After "tax" went into "taxes", merging "levies" into "tax" merges it
    into "taxes", the survivor of the name asked for: a retired name never
    becomes a topic again."""
    for content, tag in (("a", "tax"), ("b", "taxes"), ("c", "levies")):
        tagged.backend.insert_memory(Memory(content=content, user_id="ada", categories=[tag]))
    tagged.merge_tags(["tax"], "taxes", user_id="ada")
    [taxes] = _active_topics(tagged, "taxes")
    levies = tagged.backend.topic_entity("levies", Scope(user_id="ada"), create=False)
    assert tagged.merge_tags(["levies"], "tax", user_id="ada") == 1
    assert tagged.categories(user_id="ada") == [{"category": "taxes", "count": 3}]
    assert _active_topics(tagged, "tax") == [] and _active_topics(tagged, "levies") == []
    assert tagged.backend.get_entity(levies.id).merged_into == taxes
    _agree(tagged, "ada")


def test_renaming_a_tag_folds_it_into_a_topic_of_the_new_name(tagged):
    """A rename is a merge into a topic of the new name: the old name keeps a
    tombstone pointing there, so a column still naming it files under the new
    one."""
    tagged.add("a", user_id="ada", infer=False, categories=["budget"])
    budget = tagged.backend.topic_entity("budget", Scope(user_id="ada"), create=False)
    assert tagged.rename_tag("budget", "money", user_id="ada") == 1
    money = tagged.backend.topic_entity("money", Scope(user_id="ada"), create=False)
    assert money.id != budget.id
    assert (money.name, money.merged_into, money.metadata) == (
        "money", None, {"renamed_from": budget.id})
    assert tagged.backend.get_entity(budget.id).merged_into == money.id
    _agree(tagged, "ada")
    assert tagged.delete_tag("money", user_id="ada") == 1
    assert tagged.categories(user_id="ada") == []
    _agree(tagged, "ada")


def test_an_obvious_merge_in_upkeep_folds_the_entities(tagged):
    tagged.backend.insert_memory(Memory(content="one", user_id="ada", categories=["project"]))
    tagged.backend.insert_memory(Memory(content="two", user_id="ada", categories=["projects"]))
    plural = tagged.backend.topic_entity("projects", Scope(user_id="ada"), create=False)
    assert tagged.merge_obvious_topics(user_id="ada") == {
        "groups_merged": 1, "memories_changed": 1}
    single = tagged.backend.topic_entity("project", Scope(user_id="ada"), create=False)
    assert tagged.backend.get_entity(plural.id).merged_into == single.id
    _agree(tagged, "ada")


def _tag_judge(same: float):
    from memry.providers.decisions import Answer, Answers, NoneDecider

    class Judge(NoneDecider):
        name = "stub"
        available = True
        calibrated = True
        tag_merge_probability = 0.55  # JevDecider's measured bar

        def __init__(self):
            self.states = []

        def decide(self, state, questions):
            if "tag" not in questions:
                return Answers({})
            self.states.append(state)
            return Answers({"tag": Answer("same" if same >= 0.5 else "different",
                                          {"same": same, "different": 1 - same}, 0.9, True)})

    return Judge()


@pytest.mark.parametrize("same, merged", [(0.56, True), (0.54, False)])
def test_two_topics_are_judged_by_the_tag_question_at_its_bar(tagged, same, merged):
    judge = _tag_judge(same)
    tagged.decider = judge
    for i in range(5):
        tagged.add(f"quality assurance fact {i}", user_id="ada", infer=False,
                   categories=["quality assurance"])
    tagged.add("qa fact", user_id="ada", infer=False, categories=["qa"])
    qa = tagged.backend.topic_entity("qa", Scope(user_id="ada"), create=False)
    tagged.merge_obvious_topics(user_id="ada")
    assert judge.states and all("Two tags" in state for state in judge.states)
    folded = tagged.backend.get_entity(qa.id).merged_into is not None
    assert folded is merged
    assert len(tagged.categories(user_id="ada")) == (1 if merged else 2)
    # two tags never become a pair for the entity identity funnel
    assert tagged.backend.list_proposals(Scope(user_id="ada"), status=None) == []
    _agree(tagged, "ada")


def test_merging_two_topic_entities_merges_their_tags(tagged):
    tagged.add("a", user_id="ada", infer=False, categories=["tech"])
    tagged.add("b", user_id="ada", infer=False, categories=["technical", "tech"])
    tech = tagged.backend.topic_entity("tech", Scope(user_id="ada"), create=False)
    technical = tagged.backend.topic_entity("technical", Scope(user_id="ada"), create=False)
    assert tagged.merge_entities(tech.id, technical.id)
    assert tagged.categories(user_id="ada") == [{"category": "tech", "count": 2}]
    _agree(tagged, "ada")


def _pair_judge():
    """A calibrated judge that finds every pair one thing, and keeps the
    states it was shown."""
    from memry.providers.decisions import Answer, Answers, NoneDecider

    class Judge(NoneDecider):
        name = "stub"
        available = True
        calibrated = True
        pair_merge_probability = 0.95

        def __init__(self):
            self.states = []

        def decide(self, state, questions):
            if "pair" not in questions:
                return Answers({})
            self.states.append(state)
            probabilities = {"same": 0.99, "different": 0.0, "unsure": 0.01}
            return Answers({"pair": Answer("same", probabilities, 0.9, True)})

    return Judge()


@pytest.mark.parametrize("judged", [True, False], ids=["calibrated judge", "text model only"])
def test_a_tag_folds_into_the_named_thing_of_its_name(judged):
    """ "bildy" the tag and "Bildy" the product. A calibrated judge answers
    the ordinary pair question, with the tag's memories as its side. Without
    one the text model is not asked (its answer could fold nothing, and a
    "different" from it kept the two apart for good): one name is one thing,
    as at save, and the tag folds into the thing by rule."""
    llm = FakeLLM()
    judge = _pair_judge() if judged else None
    store = MemoryStore(Config(db_path=":memory:"), llm=llm, embedder=HashEmbedder(64),
                        decider=judge)
    backend = store.backend
    try:
        product = backend.insert_entity(Entity(
            name="Bildy", normalized="bildy", entity_type="product", user_id="ada"))
        for text in ("Bildy is a construction app", "Bildy runs on AWS"):
            memory = backend.insert_memory(Memory(content=text, user_id="ada"))
            backend.add_mention(EntityMention(
                entity_id=product.id, memory_id=memory.id, surface="Bildy"))
        for text in ("Shipped the invoice export", "Fixed the login bug"):
            backend.insert_memory(Memory(content=text, user_id="ada", categories=["bildy"]))
        topic = backend.topic_entity("bildy", Scope(user_id="ada"), create=False)

        outcome = store.resolve_entities(user_id="ada")
        assert (outcome["proposed"], outcome["confirmed"]) == (1, 1)
        assert llm.calls == []
        if judge is not None:
            assert judge.states and all(
                "Shipped the invoice export" in state and "Bildy runs on AWS" in state
                for state in judge.states)
        [pair] = store.merge_proposals(user_id="ada", status="confirmed")
        assert pair.reason == ("stub: same" if judged else "one name, joined by rule")
        # the named thing is kept, whichever side the pair listed first
        assert backend.get_entity(topic.id).merged_into == product.id
        assert backend.get_entity(product.id).entity_type == "product"
        assert backend.count_entity_memories(product.id) == 4
        # the column keeps the tag and its filter still works; a new memory
        # filed under it mentions the thing the tag was found to be
        assert {m.content for m in store.get_all(user_id="ada", categories=["bildy"])} == {
            "Fixed the login bug", "Shipped the invoice export"}
        later = backend.insert_memory(Memory(content="Added dark mode", user_id="ada",
                                             categories=["bildy"]))
        assert [e.id for e in backend.entities_of_memory(later.id)] == [product.id]
        assert store.entities(user_id="ada", kind="topic") == []
    finally:
        store.close()


def test_a_topic_folded_the_wrong_way_round_still_leaves_the_named_thing(tagged):
    product = tagged.backend.insert_entity(Entity(
        name="Bildy", normalized="bildy", entity_type="product", user_id="ada"))
    tagged.add("x", user_id="ada", infer=False, categories=["bildy"])
    topic = tagged.backend.topic_entity("bildy", Scope(user_id="ada"), create=False)
    assert tagged.backend.merge_entities(topic.id, product.id)
    assert tagged.backend.get_entity(product.id).merged_into is None
    assert tagged.backend.get_entity(topic.id).merged_into == product.id


# ------------------------------------------------------------------- search
def test_masking_reads_names_as_it_but_leaves_tags_as_written(tagged):
    memory_id = tagged.add("spent 34 euros on groceries at Lidl", user_id="ada",
                           infer=False, categories=["groceries"]).actions[0].memory_id
    lidl = tagged.backend.insert_entity(Entity(
        name="Lidl", normalized="lidl", entity_type="organization", user_id="ada"))
    tagged.backend.add_mention(EntityMention(entity_id=lidl.id, memory_id=memory_id,
                                             surface="Lidl"))
    everything = [e.id for e in tagged.backend.entities_of_memory(memory_id, kind="any")]
    assert len(everything) == 2  # the organization and the topic "groceries"
    # what masks reads its entities through lookups that leave tags out (in SQL)
    named = [e.id for e in tagged.backend.entities_of_memory(memory_id)]
    assert named == [lidl.id]
    assert tagged.backend.entity_memory_links(Scope(user_id="ada")) == [(lidl.id, memory_id)]
    read: list[str] = []
    get_entity = tagged.backend.get_entity

    def counted(entity_id):
        read.append(entity_id)
        return get_entity(entity_id)

    tagged.backend.get_entity = counted
    try:
        masked = tagged._masked_texts({memory_id: "spent 34 euros on groceries at Lidl"},
                                      {memory_id: named})
    finally:
        del tagged.backend.get_entity
    assert masked == {memory_id: "spent 34 euros on groceries at it"}
    assert read == [lidl.id]  # once, for its names: no second lookup per entity
    assert tagged.refresh_property_vectors(user_id="ada") == 1
    stored = tagged.backend.property_vector_hashes([memory_id])[memory_id]
    assert stored[0] == _text_hash("spent 34 euros on groceries at it")
    assert tagged.refresh_property_vectors(memory_ids=[memory_id]) == 0  # unchanged


def test_a_topic_never_seeds_the_linked_search(tagged, monkeypatch):
    from memry import store as store_module
    from memry.intelligence.structure import is_hub

    ada = tagged.backend.insert_entity(Entity(
        name="Ada", normalized="ada", entity_type="person", user_id="ada"))
    for text in ("Ada likes pasta", "Ada likes sushi", "Bob hates olives"):
        memory_id = tagged.add(text, user_id="ada", infer=False,
                               categories=["food"]).actions[0].memory_id
        if text.startswith("Ada"):
            tagged.backend.add_mention(EntityMention(entity_id=ada.id, memory_id=memory_id,
                                                     surface="Ada"))
    food = tagged.backend.topic_entity("food", Scope(user_id="ada"), create=False)
    # three memories would make any other name a hub; a tag never is one
    assert is_hub("concept", 3, 0) and not is_hub(TOPIC_TYPE, 3, 0)
    assert tagged._is_hub(ada.id) and not tagged._is_hub(food.id)

    seeded = []
    real = store_module.activation_paths

    def spy(backend, seeds, **kwargs):
        seeded.append(list(seeds))
        return real(backend, seeds, **kwargs)

    monkeypatch.setattr(store_module, "activation_paths", spy)
    tagged.search("What food does Ada like?", user_id="ada", limit=5)
    assert seeded == [[ada.id]]
    # even when the name lookup is made to return the tag, it is not a seed
    monkeypatch.setattr(store_module, "detect_query_entities",
                        lambda *args, **kwargs: [food.id, ada.id])
    tagged.search("What food does Ada like?", user_id="ada", limit=5)
    assert seeded[-1] == [ada.id]


# ---------------------------------------------------------------- migration
def _legacy(store):
    """The store as a database from before tags were entities: the same
    memories and legacy tag tables, no topic entities, no tag mentions."""
    with store.backend._lock:
        store.backend._db.execute(
            "DELETE FROM entity_mentions WHERE entity_id IN "
            "(SELECT id FROM entities WHERE entity_type = 'topic')")
        store.backend._db.execute("DELETE FROM entities WHERE entity_type = 'topic'")
        store.backend._db.commit()


def _tables(store):
    db = store.backend._db
    return {table: [tuple(row) for row in db.execute(f"SELECT * FROM {table} ORDER BY 1, 2")]
            for table in ("topics", "memory_topics", "topic_relations", "entities",
                          "entity_mentions")}


def test_tags_to_topics_migrates_counts_and_is_idempotent(tagged):
    from memry.models import Topic, TopicRelation

    tagged.add("Ada runs", user_id="ada", infer=False, categories=["Health", "running"])
    tagged.add("Ada sleeps", user_id="ada", infer=False, categories=["health"])
    tagged.add("Bob works", user_id="bob", infer=False, categories=["work"])
    parent = tagged.backend.upsert_topic(Topic(name="wellbeing", normalized="wellbeing",
                                               user_id="ada", provenance="synthetic"))
    running = next(t for t in tagged.backend.list_topics(Scope(user_id="ada"))
                   if t.normalized == "running")
    tagged.backend.add_topic_relation(TopicRelation(
        broader_topic_id=parent.id, narrower_topic_id=running.id, user_id="ada"))
    _legacy(tagged)
    assert tagged.categories(user_id="ada") == []
    before = _tables(tagged)

    dry = tagged.tags_to_topics(dry_run=True)
    assert _tables(tagged) == before  # a dry run changes nothing
    report = tagged.tags_to_topics()
    assert [{k: v for k, v in row.items() if k != "dry_run"} for row in dry] == [
        {k: v for k, v in row.items() if k != "dry_run"} for row in report]
    by_user = {row["user_id"]: row for row in report}
    assert by_user["ada"] == {
        "user_id": "ada", "topics": 3, "skipped_parents": 1, "entities_created": 2,
        "entities_existing": 0, "mentions_created": 3, "mentions_existing": 0,
        "dry_run": False}
    assert by_user["bob"]["entities_created"] == 1 and by_user["bob"]["mentions_created"] == 1

    assert tagged.categories(user_id="ada") == [
        {"category": "health", "count": 2}, {"category": "running", "count": 1}]
    _agree(tagged, "ada")
    _agree(tagged, "bob")
    after = _tables(tagged)
    # the legacy tables are the record: read, never written; relations stay put
    for table in ("topics", "memory_topics", "topic_relations"):
        assert after[table] == before[table]

    again = tagged.tags_to_topics()
    assert _tables(tagged) == after  # a second run changes nothing
    assert sum(row["entities_created"] + row["mentions_created"] for row in again) == 0
    assert {row["user_id"]: row["mentions_existing"] for row in again} == {"ada": 3, "bob": 1}


def test_tags_to_things_command(monkeypatch, tmp_path, capsys):
    from memry.backends.local import LocalBackend
    from memry.cli import main

    db = tmp_path / "legacy.db"
    monkeypatch.setenv("MEMRY_DB_PATH", str(db))
    monkeypatch.setenv("MEMRY_CONFIG", str(tmp_path / "missing.json"))
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "VOYAGE_API_KEY",
                "MEMRY_LLM_PROVIDER", "MEMRY_EMBEDDING_PROVIDER"):
        monkeypatch.delenv(key, raising=False)
    backend = LocalBackend(str(db))
    backend.insert_memory(Memory(content="one", user_id="ada", categories=["tax"]))
    backend.insert_memory(Memory(content="two", user_id="ada", categories=["tax", "home"]))
    backend.insert_memory(Memory(content="three", user_id="bob", categories=["tax"]))
    with backend._lock:
        backend._db.execute("DELETE FROM entity_mentions")
        backend._db.execute("DELETE FROM entities")
        backend._db.commit()
    backend.close()

    def run(*argv):
        assert main(["tags-to-things", *argv]) == 0
        return json.loads(capsys.readouterr().out)

    dry = run("--dry-run")
    assert dry["dry_run"] is True
    assert dry["total"]["entities_created"] == 3 and dry["total"]["mentions_created"] == 4
    assert run("--dry-run")["total"] == dry["total"]  # nothing was written
    only_ada = run("--user", "ada")
    assert [row["user_id"] for row in only_ada["scopes"]] == ["ada"]
    assert only_ada["total"]["mentions_created"] == 3
    rest = run()
    assert rest["total"]["entities_created"] == 1 and rest["total"]["mentions_created"] == 1
    assert rest["total"]["mentions_existing"] == 3
    assert run()["total"]["mentions_created"] == 0

    reopened = MemoryStore(Config(db_path=str(db)), llm=NoneLLM(), embedder=HashEmbedder(64))
    try:
        assert reopened.categories(user_id="ada") == [
            {"category": "tax", "count": 2}, {"category": "home", "count": 1}]
    finally:
        reopened.close()


def test_an_upgraded_database_gets_its_topic_entities_when_it_opens(tmp_path):
    """A database from before tags were entities (legacy tag tables, no topic
    entities) shows its tags on the first open, with no ``tags-to-things``
    run: opening migrates them. A second open does not run it again."""
    from memry.backends.local import LocalBackend

    db = tmp_path / "upgraded.db"
    backend = LocalBackend(str(db))
    backend.insert_memory(Memory(content="one", user_id="ada", categories=["tax"]))
    backend.insert_memory(Memory(content="two", user_id="ada", categories=["tax", "home"]))
    backend.insert_memory(Memory(content="three", user_id="bob", categories=["tax"]))
    with backend._lock:
        backend._db.execute("DELETE FROM entity_mentions")
        backend._db.execute("DELETE FROM entities")
        backend._db.execute("DELETE FROM meta WHERE key LIKE 'schema:tag-entities%'")
        backend._db.commit()
    backend.close()

    def opened():
        return MemoryStore(Config(db_path=str(db)), llm=NoneLLM(), embedder=HashEmbedder(64))

    store = opened()
    try:
        assert store.categories(user_id="ada") == [
            {"category": "tax", "count": 2}, {"category": "home", "count": 1}]
        assert store.categories(user_id="bob") == [{"category": "tax", "count": 1}]
        _agree(store, "ada")
        # the command still runs, and finds nothing left to do
        assert sum(row["mentions_created"] for row in store.tags_to_topics()) == 0
        with store.backend._lock:
            store.backend._db.execute("DELETE FROM entity_mentions")
            store.backend._db.execute("DELETE FROM entities")
            store.backend._db.commit()
    finally:
        store.close()
    again = opened()
    try:
        assert again.categories(user_id="ada") == []  # opened once, migrated once
    finally:
        again.close()


def test_a_migration_stopped_between_users_resumes_where_it_stopped(tmp_path, monkeypatch):
    """The migration at open commits user by user and sets its marker after
    the last one: stopped at "bob", the next open finds "ada" migrated and
    migrates "bob"."""
    import gc
    import sqlite3

    from memry.backends.local import LocalBackend

    db = tmp_path / "stopped.db"
    backend = LocalBackend(str(db))
    backend.insert_memory(Memory(content="one", user_id="ada", categories=["tax"]))
    backend.insert_memory(Memory(content="two", user_id="bob", categories=["home"]))
    with backend._lock:
        backend._db.execute("DELETE FROM entity_mentions")
        backend._db.execute("DELETE FROM entities")
        backend._db.execute("DELETE FROM meta WHERE key LIKE 'schema:tag-entities%'")
        backend._db.commit()
    backend.close()

    migrate = LocalBackend._tags_to_topics_locked

    def stopped_at_bob(self, user_id):
        if user_id == "bob":
            raise RuntimeError("the process was stopped")
        return migrate(self, user_id)

    monkeypatch.setattr(LocalBackend, "_tags_to_topics_locked", stopped_at_bob)
    with pytest.raises(RuntimeError):
        LocalBackend(str(db))
    monkeypatch.undo()
    gc.collect()

    def topics():
        with sqlite3.connect(db) as peek:
            return sorted(peek.execute(
                "SELECT user_id, normalized FROM entities WHERE entity_type = 'topic'"))

    def marker():
        with sqlite3.connect(db) as peek:
            return peek.execute(
                "SELECT 1 FROM meta WHERE key = 'schema:tag-entities:v1'").fetchone()

    assert topics() == [("ada", "tax")] and marker() is None
    reopened = LocalBackend(str(db))
    try:
        assert reopened.topic_mention_counts(Scope(user_id="bob")) == [
            {"category": "home", "count": 1}]
    finally:
        reopened.close()
    assert topics() == [("ada", "tax"), ("bob", "home")] and marker() is not None


def test_two_processes_creating_one_topic_leave_one_active_entity(tmp_path):
    """Two processes on one database both look for the topic "tax", find none
    and create it. The unique index on active topics lets the first insert
    through; the second inserts nothing and reads the first one's back."""
    from memry.backends.local import LocalBackend

    path = str(tmp_path / "race.db")
    first, second = LocalBackend(path), LocalBackend(path)
    scope = Scope(user_id="ada")

    class Interleaved:
        """The second process's connection: the first creates the topic just
        before the second's insert runs, after the second found none."""

        def __init__(self, db):
            self.db = db

        def execute(self, sql, *args):
            if sql.lstrip().upper().startswith("INSERT") and "INTO entities" in sql:
                first.topic_entity("tax", scope)
            return self.db.execute(sql, *args)

        def __getattr__(self, name):
            return getattr(self.db, name)

    connection = second._db
    second._db = Interleaved(connection)
    try:
        created = second.topic_entity("tax", scope)
    finally:
        second._db = connection
    try:
        winner = first.topic_entity("tax", scope, create=False)
        assert created.id == winner.id
        active = connection.execute(
            "SELECT id FROM entities WHERE entity_type = 'topic' AND normalized = 'tax' "
            "AND merged_into IS NULL").fetchall()
        assert [row["id"] for row in active] == [winner.id]
    finally:
        first.close()
        second.close()


def test_a_database_with_two_active_topics_of_one_name_opens_with_one(tmp_path):
    """A database written before the unique index, where two processes did
    create one topic twice, opens with the later folded into the earlier: one
    active topic, every memory mentioning it, the index in place."""
    import sqlite3

    from memry.backends.local import LocalBackend

    path = tmp_path / "doubled.db"
    backend = LocalBackend(str(path))
    first = backend.insert_memory(Memory(content="one", user_id="ada", categories=["tax"]))
    second = backend.insert_memory(Memory(content="two", user_id="ada", categories=["tax"]))
    [original] = [row["id"] for row in backend._db.execute(
        "SELECT id FROM entities WHERE entity_type = 'topic'")]
    backend.close()
    with sqlite3.connect(path) as db:
        db.execute("DROP INDEX IF EXISTS ux_entities_active_topic_ns")
        db.execute("INSERT INTO entities (id, name, normalized, entity_type, user_id, metadata, "
                   "created_at, updated_at) VALUES ('twin', 'tax', 'tax', 'topic', 'ada', '{}', "
                   "'2999-01-01T00:00:00+00:00', '2999-01-01T00:00:00+00:00')")
        db.execute("UPDATE entity_mentions SET entity_id = 'twin' WHERE memory_id = ?",
                   (second.id,))
    reopened = LocalBackend(str(path))
    try:
        active = [row["id"] for row in reopened._db.execute(
            "SELECT id FROM entities WHERE entity_type = 'topic' AND merged_into IS NULL")]
        assert active == [original]
        assert reopened.get_entity("twin").merged_into == original
        assert sorted(m.id for m in reopened.entity_memories(original, limit=10)) == sorted(
            [first.id, second.id])
        assert reopened.topic_mention_counts(Scope(user_id="ada")) == [
            {"category": "tax", "count": 2}]
        assert reopened._db.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'ux_entities_active_topic_ns'").fetchone()
    finally:
        reopened.close()


def test_an_existing_database_opens_with_the_additive_schema(tmp_path):
    """A database written before this change opens: the only schema change is
    an index created if missing."""
    import sqlite3

    from memry.backends.local import LocalBackend

    path = tmp_path / "old.db"
    backend = LocalBackend(str(path))
    backend.insert_memory(Memory(content="old", user_id="ada", categories=["old"]))
    backend.close()
    with sqlite3.connect(path) as db:
        db.execute("DROP INDEX idx_entities_type_user")
    reopened = LocalBackend(str(path))
    try:
        assert reopened.topic_mention_counts(Scope(user_id="ada")) == [
            {"category": "old", "count": 1}]
        assert reopened._db.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'idx_entities_type_user'").fetchone()
    finally:
        reopened.close()


# ---------------------------------------------------------------- dashboard
def test_the_dashboard_lists_tags_as_topic_entities():
    from memry.rest import create_app

    store = MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(), embedder=HashEmbedder(64))
    try:
        store.add("a", user_id="default", infer=False, categories=["work", "diet"])
        store.add("b", user_id="default", infer=False, categories=["work"])
        ada = store.backend.insert_entity(Entity(name="Ada", entity_type="person",
                                                 user_id="default"))
        memory = store.get_all(user_id="default", limit=1)[0]
        store.backend.add_mention(EntityMention(entity_id=ada.id, memory_id=memory.id,
                                                surface="Ada"))
        with TestClient(create_app(store)) as client:
            listed = {kind: client.get("/api/v1/entities",
                                       params={"limit": 100, "kind": kind}).json()
                      for kind in ("any", "topic", "named")}
            default = client.get("/api/v1/entities", params={"limit": 100}).json()
            html = client.get("/").text
        topics = sorted((row["name"], row["entity_type"], row["memories"], row["hub"])
                        for row in listed["topic"])
        assert topics == [("diet", "topic", 1, False), ("work", "topic", 2, False)]
        assert sorted(row["name"] for row in listed["any"]) == ["Ada", "diet", "work"]
        # named things alone unless asked, as before
        assert [row["name"] for row in listed["named"]] == ["Ada"] == [
            row["name"] for row in default]
        # the names page lists every type, "topic" among them
        assert "api('/api/v1/entities?limit=100000&include_merged=true&kind=any')" in html
    finally:
        store.close()


@pytest.mark.skipif(shutil.which("node") is None, reason="node renders the tag page")
def test_the_dashboard_tag_page_renders_from_topic_entities():
    """The tag page draws the counts of the topic entities: a tag with no
    legacy index row is listed, a synthetic parent adds no rolled-up row."""
    import re

    from memry.models import SyntheticTag
    from memry.rest import create_app

    store = MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(), embedder=HashEmbedder(64))
    try:
        store.add("a", user_id="default", infer=False, categories=["work", "diet"])
        store.add("b", user_id="default", infer=False, categories=["work"])
        store.backend.record_synthetic_tag(SyntheticTag(
            tag="life", source_tags=["work", "diet"], user_id="default"))
        with store.backend._lock:
            store.backend._db.execute("DELETE FROM memory_topics")
            store.backend._db.commit()
        with TestClient(create_app(store)) as client:
            html = client.get("/").text
            categories = client.get("/api/v1/categories").json()
    finally:
        store.close()
    source = "\n".join(re.findall(r"<script>(.*?)</script>", html, re.S))
    helpers = "\n".join(source[source.index(start):source.index("\n", source.index(start))]
                        for start in ("function esc(s)", "function jsArg(v)"))
    page = source[source.index("function tagSel()"):source.index("async function tagOp(")]
    contract = helpers + "\nlet allTags=[];\n" + r"""
const nodes={taglist:{innerHTML:''},tagsearch:{value:''},tagsel:{textContent:''}};
const document={getElementById:id=>nodes[id],querySelectorAll:()=>[]};
let asked=null;
const api=async path=>{asked=path;return JSON.parse(process.argv[2])};
""" + page + r"""
function check(condition,message){if(!condition)throw new Error(message)}
(async()=>{
  await loadTags();
  const html=nodes.taglist.innerHTML;
  check(asked==='/api/v1/categories','the page reads the categories endpoint');
  const rows=[...html.matchAll(/<b>([^<]+)<\/b> <span class="cnt">(\d+)<\/span>/g)]
    .map(m=>m[1]+'='+m[2]);
  check(rows.join()==='diet=1,work=2','rows: '+rows.join());
  check(!html.includes('synthetic parent'),'no parent row');
})().catch(e=>{console.error(e.message);process.exit(1)});
"""
    result = subprocess.run(["node", "-", json.dumps(categories)], input=contract,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


# ------------------------------------------------------------------- upkeep
def test_synthetic_parents_do_not_run_in_upkeep_unless_configured():
    llm = FakeLLM()
    config = Config(db_path=":memory:")
    store = MemoryStore(config, llm=llm, embedder=HashEmbedder(64))
    try:
        for key in ("dedup_entities", "durability", "consolidation", "structure"):
            store.set_maintenance_enabled(key, False)
        store.add("a", user_id="ada", infer=False, categories=["x"])
        # a stored switch alone does not bring the pass back
        store.set_maintenance_enabled("tag_abstraction", True)
        assert not store.maintenance_enabled("tag_abstraction")
        assert "tag_abstraction" not in store.run_upkeep_cycle(user_id="ada")
        config.tags.enabled = True
        ran = store.run_upkeep_cycle(user_id="ada")
        assert "tags" in ran["tag_abstraction"]["skipped"]  # reachable, too few tags
    finally:
        store.close()


def test_stats_count_topics_apart_from_named_entities(tagged):
    tagged.add("a", user_id="ada", infer=False, categories=["work", "diet"])
    tagged.backend.insert_entity(Entity(name="Ada", entity_type="person", user_id="ada"))
    stats = tagged.stats()
    assert (stats["entities"], stats["topics"]) == (1, 2)


# ---------------------------------------------- one invariant, one code path
def _index_of(store, memory_id):
    """The names the legacy filter index files a memory under."""
    return sorted(row["normalized"] for row in store.backend._db.execute(
        "SELECT t.normalized FROM memory_topics mt JOIN topics t ON t.id = mt.topic_id "
        "WHERE mt.memory_id = ?", (memory_id,)))


def test_a_column_left_naming_a_merged_tag_is_filed_again_at_open(tmp_path):
    """A database from before every write filed its tags through one function:
    a memory's column still names "taxes", merged into "tax", its legacy index
    files it under "taxes", and its mention followed the merge. The first
    open rewrites the column to the survivor, once (a marker), and its index
    and mention follow, so the column, the index, the mentions and
    ``categories()`` agree, and deleting "tax" takes the tag off it too.
    Another user's own "taxes" is theirs and stays."""
    import sqlite3

    db = tmp_path / "survivors.db"

    def opened():
        return MemoryStore(Config(db_path=str(db)), llm=NoneLLM(), embedder=HashEmbedder(64))

    store = opened()
    try:
        insert = store.backend.insert_memory
        a = insert(Memory(content="a", user_id="ada", categories=["taxes"])).id
        b = insert(Memory(content="b", user_id="ada", categories=["tax"])).id
        stale = insert(Memory(content="c", user_id="ada", categories=["home"])).id
        bob = insert(Memory(content="d", user_id="bob", categories=["taxes"])).id
        assert store.merge_tags(["taxes"], "tax", user_id="ada") == 1
        tax = store.backend.topic_entity("tax", Scope(user_id="ada"), create=False).id
    finally:
        store.close()
    stamp = "2024-01-01T00:00:00+00:00"
    with sqlite3.connect(db) as raw:  # as the writes before this change left it
        raw.execute("UPDATE memories SET categories = ? WHERE id = ?",
                    (json.dumps(["taxes", "home"]), stale))
        raw.execute("INSERT INTO topics (id, name, normalized, user_id, provenance, "
                    "created_at, updated_at) VALUES ('old', 'taxes', 'taxes', 'ada', "
                    "'memory', ?, ?)", (stamp, stamp))
        raw.execute("INSERT INTO memory_topics (memory_id, topic_id) VALUES (?, 'old')",
                    (stale,))
        raw.execute("INSERT INTO entity_mentions (id, entity_id, memory_id, surface, "
                    "created_at) VALUES ('m-old', ?, ?, 'taxes', ?)", (tax, stale, stamp))
        raw.execute("DELETE FROM meta WHERE key = 'schema:tag-survivors:v1'")
    raw.close()

    store = opened()
    try:
        assert store.get(stale).categories == ["tax", "home"]
        assert _index_of(store, stale) == ["home", "tax"]
        assert _topics_of(store, stale) == ["home", "tax"]
        _agree(store, "ada")
        assert store.categories(user_id="ada") == [
            {"category": "tax", "count": 3}, {"category": "home", "count": 1}]
        assert {m.id for m in store.get_all(user_id="ada", categories=["tax"])} == {
            a, b, stale}
        assert store.get_all(user_id="ada", categories=["taxes"]) == []
        assert store.get(bob).categories == ["taxes"]
        assert store.delete_tag("tax", user_id="ada") == 3
        assert store.get(stale).categories == ["home"]
        assert _index_of(store, stale) == ["home"]
        assert store.categories(user_id="ada") == [{"category": "home", "count": 1}]
        _agree(store, "ada")
    finally:
        store.close()
    with sqlite3.connect(db) as raw:  # the marker holds: a later open does not run it
        raw.execute("UPDATE memories SET categories = ? WHERE id = ?",
                    (json.dumps(["taxes"]), a))
    raw.close()
    store = opened()
    try:
        assert store.get(a).categories == ["taxes"]
    finally:
        store.close()


def test_saves_file_a_retired_name_under_its_survivor_and_group_active_names_only(tagged):
    """After "tax" went into "levies", a save tagged "Tax" is filed under
    "levies", through the name's own tombstone. Obvious variants are grouped
    among names still active only: "taxes", never a topic, is not sent to
    the survivor of the retired "tax" beside it but is a tag of its own, and
    a later save of "tax" leaves that topic alone. No "tax" topic is active
    at any point."""
    insert = tagged.backend.insert_memory
    insert(Memory(content="paid the tax", user_id="ada", categories=["tax"]))
    insert(Memory(content="levy notice", user_id="ada", categories=["levies"]))
    assert tagged.merge_tags(["tax"], "levies", user_id="ada") == 1
    [levies] = _active_topics(tagged, "levies")
    single = tagged.add("tax office letter", user_id="ada", infer=False,
                        categories=["Tax"]).actions[0].memory_id
    plural = tagged.add("paid the taxes", user_id="ada", infer=False,
                        categories=["taxes"]).actions[0].memory_id
    assert tagged.get(single).categories == ["levies"]
    assert tagged.get(plural).categories == ["taxes"]
    [taxes] = _active_topics(tagged, "taxes")
    assert _active_topics(tagged, "tax") == []
    assert tagged.categories(user_id="ada") == [
        {"category": "levies", "count": 3}, {"category": "taxes", "count": 1}]
    _agree(tagged, "ada")

    tagged.add("tax refund", user_id="ada", infer=False, categories=["tax"])
    assert _active_topics(tagged, "tax") == []
    assert _active_topics(tagged, "levies") == [levies]
    assert _active_topics(tagged, "taxes") == [taxes]
    assert tagged.get(plural).categories == ["taxes"]
    assert tagged.categories(user_id="ada") == [
        {"category": "levies", "count": 4}, {"category": "taxes", "count": 1}]
    _agree(tagged, "ada")


def test_a_renamed_tag_keeps_its_description_and_metadata(tagged):
    """A rename is a merge into a topic of the new name; that topic takes over
    the old one's description (with the time it was written) and metadata,
    and records the old id as ``renamed_from``, so a reference stored under
    the old id can be followed. A merge into a tag that exists is no rename:
    the kept topic keeps its own."""
    tagged.add("a", user_id="ada", infer=False, categories=["budget"])
    budget = tagged.backend.topic_entity("budget", Scope(user_id="ada"), create=False)
    written = "2026-01-02T03:04:05+00:00"
    tagged.backend.set_entity_description(budget.id, "The household budget.", written)
    tagged.backend.set_entity_metadata(budget.id, {"home": "finances", "note": "kept"})
    assert tagged.rename_tag("budget", "money", user_id="ada") == 1
    money = tagged.backend.topic_entity("money", Scope(user_id="ada"), create=False)
    assert (money.description, money.description_updated_at) == (
        "The household budget.", written)
    assert money.metadata == {"home": "finances", "note": "kept", "renamed_from": budget.id}
    assert tagged.backend.resolve_entity_id(money.metadata["renamed_from"]) == money.id
    # the entity route renames a tag the same way
    cash = tagged.rename_entity(money.id, "cash")
    assert (cash.name, cash.description, cash.metadata["renamed_from"]) == (
        "cash", "The household budget.", money.id)
    tagged.add("b", user_id="ada", infer=False, categories=["savings"])
    savings = tagged.backend.topic_entity("savings", Scope(user_id="ada"), create=False)
    assert tagged.merge_tags(["cash"], "savings", user_id="ada") == 1
    kept = tagged.backend.get_entity(savings.id)
    assert (kept.merged_into, kept.description, kept.metadata) == (None, None, {})
    _agree(tagged, "ada")


def test_the_empty_user_and_no_user_are_two_namespaces_for_tags(tagged):
    """ "" is a user of its own, not the memories without one (None), for
    tags as for every memory lookup: each namespace has its own "tax" topic,
    every memory mentions its own, and neither save fails or drops a
    mention. A merge in one leaves the other alone."""
    insert = tagged.backend.insert_memory
    nobody = insert(Memory(content="a", user_id=None, categories=["tax"])).id
    empty = insert(Memory(content="b", user_id="", categories=["tax"])).id
    also = insert(Memory(content="c", user_id="", categories=["tax"], run_id="r1")).id
    topics = {row["user_id"]: row["id"] for row in tagged.backend._db.execute(
        "SELECT id, user_id FROM entities WHERE entity_type = 'topic' "
        "AND normalized = 'tax' AND merged_into IS NULL")}
    assert set(topics) == {None, ""}
    mentioned = {memory_id: [e.id for e in tagged.backend.entities_of_memory(
        memory_id, kind="topic")] for memory_id in (nobody, empty, also)}
    assert mentioned == {nobody: [topics[None]], empty: [topics[""]], also: [topics[""]]}
    assert tagged.backend.topic_mention_counts(Scope(user_id=""), exact_user=True) == [
        {"category": "tax", "count": 2}]
    assert tagged.backend.topic_mention_counts(Scope(user_id=None), exact_user=True) == [
        {"category": "tax", "count": 1}]
    assert tagged.merge_tags(["tax"], "levies", user_id="") == 2
    assert tagged.get(nobody).categories == ["tax"]
    assert [tagged.get(m).categories for m in (empty, also)] == [["levies"], ["levies"]]
    assert tagged.backend.get_entity(topics[None]).merged_into is None


def test_an_index_that_folded_the_two_namespaces_is_replaced_at_open(tmp_path):
    """A database whose unique index keyed ``IFNULL(user_id, '')`` held one
    "tax" topic for None and "" together, so a memory of "" lost its mention
    to the other namespace's topic. The open replaces the index, files the
    memories of both namespaces again (the mention comes back, on a topic of
    its own namespace), and folds no topic of one into the other's."""
    import sqlite3

    from memry.backends.local import LocalBackend

    path = tmp_path / "namespaces.db"
    backend = LocalBackend(str(path))
    nobody = backend.insert_memory(Memory(content="a", user_id=None, categories=["tax"])).id
    empty = backend.insert_memory(Memory(content="b", user_id="", categories=["tax"])).id
    backend.close()
    with sqlite3.connect(path) as raw:  # the "" topic never made it under the old index
        raw.execute("DROP INDEX ux_entities_active_topic_ns")
        raw.execute("DELETE FROM entity_mentions WHERE memory_id = ?", (empty,))
        raw.execute("DELETE FROM entities WHERE user_id = ''")
        raw.execute("CREATE UNIQUE INDEX ux_entities_active_topic ON entities("
                    "IFNULL(user_id, ''), normalized) "
                    "WHERE entity_type = 'topic' AND merged_into IS NULL")
    raw.close()
    def owners(backend):
        return {memory_id: [(e.user_id, e.entity_type) for e in backend.entities_of_memory(
            memory_id, kind="topic")] for memory_id in (nobody, empty)}

    def active(backend):
        return sorted(repr(row["user_id"]) for row in backend._db.execute(
            "SELECT user_id FROM entities WHERE entity_type = 'topic' AND merged_into IS NULL"))

    reopened = LocalBackend(str(path))
    try:
        indexes = {row["name"] for row in reopened._db.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index'")}
        assert "ux_entities_active_topic_ns" in indexes
        assert "ux_entities_active_topic" not in indexes
        assert owners(reopened) == {nobody: [(None, TOPIC_TYPE)], empty: [("", TOPIC_TYPE)]}
        assert active(reopened) == ["''", "None"]
    finally:
        reopened.close()
    # a database from before any index, one "tax" in each namespace: no twins
    with sqlite3.connect(path) as raw:
        raw.execute("DROP INDEX ux_entities_active_topic_ns")
    raw.close()
    reopened = LocalBackend(str(path))
    try:
        assert active(reopened) == ["''", "None"]
        assert owners(reopened) == {nobody: [(None, TOPIC_TYPE)], empty: [("", TOPIC_TYPE)]}
    finally:
        reopened.close()


def test_an_open_stopped_between_the_index_and_the_refile_files_again_at_the_next(
        tmp_path, monkeypatch):
    """Replacing the index that folded "" and None dropped it in a commit of
    its own, and nothing recorded that the memories still had to be filed
    again: an open stopped after the drop left a database with neither index,
    and every later open skipped the refile. The refile is marked done once it
    ran, so a database whose old index is gone but whose marker is absent is
    filed again at the next open; and the drop, the fold of twins and the new
    index are one transaction, so a failure among them keeps the old index."""
    import sqlite3

    from memry.backends.local import LocalBackend

    path = tmp_path / "stopped.db"
    backend = LocalBackend(str(path))
    nobody = backend.insert_memory(Memory(content="a", user_id=None, categories=["tax"])).id
    empty = backend.insert_memory(Memory(content="b", user_id="", categories=["tax"])).id
    backend.close()

    def owners(backend):
        return {memory_id: [e.user_id for e in backend.entities_of_memory(
            memory_id, kind="topic")] for memory_id in (nobody, empty)}

    def stopped_after_the_drop():  # the mention the old index cost, no index, no marker
        with sqlite3.connect(path) as raw:
            raw.execute("DROP INDEX IF EXISTS ux_entities_active_topic_ns")
            raw.execute("DELETE FROM entity_mentions WHERE memory_id = ?", (empty,))
            raw.execute("DELETE FROM entities WHERE user_id = ''")
            raw.execute("DELETE FROM meta WHERE key = 'schema:active-topic-ns:v1'")
        raw.close()

    stopped_after_the_drop()
    reopened = LocalBackend(str(path))
    try:
        assert owners(reopened) == {nobody: [None], empty: [""]}
    finally:
        reopened.close()

    # stopped again, this time during the refile: the index is in, the marker
    # is not, and the next open files the memories again
    stopped_after_the_drop()

    def fail(*args, **kwargs):
        raise RuntimeError("stopped")

    with monkeypatch.context() as patch:
        patch.setattr(LocalBackend, "_refile_locked", fail)
        with pytest.raises(RuntimeError):
            LocalBackend(str(path))
    reopened = LocalBackend(str(path))
    try:
        assert owners(reopened) == {nobody: [None], empty: [""]}
    finally:
        reopened.close()

    # the old index (by its name; not unique here, so twins can stand beside
    # it) and twins: a failure while folding them leaves the old index
    with sqlite3.connect(path) as raw:
        raw.execute("DROP INDEX ux_entities_active_topic_ns")
        raw.execute("DELETE FROM entity_mentions WHERE memory_id = ?", (empty,))
        raw.execute("DELETE FROM entities WHERE user_id = ''")
        raw.execute("CREATE INDEX ux_entities_active_topic ON entities("
                    "IFNULL(user_id, ''), normalized) "
                    "WHERE entity_type = 'topic' AND merged_into IS NULL")
        for twin, created in (("twin", "2999-01-01"), ("first", "2000-01-01")):
            raw.execute("INSERT INTO entities (id, name, normalized, entity_type, user_id, "
                        "metadata, created_at, updated_at) VALUES (?, 'home', 'home', "
                        "'topic', 'ada', '{}', ?, ?)", (twin, created, created))
    raw.close()
    with monkeypatch.context() as patch:
        patch.setattr(LocalBackend, "_merge_entities_locked", fail)
        with pytest.raises(RuntimeError):
            LocalBackend(str(path))
    with sqlite3.connect(path) as raw:
        indexes = {row[0] for row in raw.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index'")}
    raw.close()
    assert "ux_entities_active_topic" in indexes
    assert "ux_entities_active_topic_ns" not in indexes
    reopened = LocalBackend(str(path))
    try:
        assert owners(reopened) == {nobody: [None], empty: [""]}
        assert reopened.get_entity("twin").merged_into == "first"
    finally:
        reopened.close()


def test_an_update_retags_its_own_memory_only(tagged, monkeypatch):
    """PATCHing one memory's tags writes them in their obvious canonical form
    and files a name merged away under its survivor, without the
    vocabulary-wide merge: the other memories tagged "taxes" keep their
    column and history, and only the incoming tags' obvious variants are
    read, not every topic of the user. The next save runs the merge."""
    from memry.backends.local import LocalBackend

    insert = tagged.backend.insert_memory
    a = insert(Memory(content="a", user_id="ada", categories=["taxes"])).id
    b = insert(Memory(content="b", user_id="ada", categories=["taxes"])).id
    c = insert(Memory(content="c", user_id="ada", categories=["home"])).id
    before = {m: (tagged.get(m).updated_at, tagged.history(m)) for m in (a, b)}
    read: list = []
    names = LocalBackend.topic_names
    listed = LocalBackend.list_entities
    monkeypatch.setattr(LocalBackend, "topic_names", lambda self, scope, **kw: (
        read.append(kw.get("prefixes")) or names(self, scope, **kw)))
    monkeypatch.setattr(LocalBackend, "list_entities", lambda self, *args, **kw: (
        read.append(("list_entities", kw.get("kind"))) or listed(self, *args, **kw)))
    assert tagged.update(c, categories=["Tax"]).categories == ["tax"]
    assert read == [{"tax"}]
    for memory_id in (a, b):
        memory = tagged.get(memory_id)
        assert memory.categories == ["taxes"]
        assert (memory.updated_at, tagged.history(memory_id)) == before[memory_id]
    assert _active_topics(tagged, "taxes") and _active_topics(tagged, "tax")
    _agree(tagged, "ada")
    monkeypatch.undo()
    tagged.add("tax refund", user_id="ada", infer=False, categories=["tax"])
    assert [tagged.get(m).categories for m in (a, b, c)] == [["tax"], ["tax"], ["tax"]]
    assert _active_topics(tagged, "taxes") == []
    _agree(tagged, "ada")


def test_a_restored_backup_naming_a_merged_tag_is_filed_under_its_survivor():
    """A backup taken before "taxes" went into "tax" restores into a store
    where it did: the restored column names "tax", and its index and mention
    follow it (the invariant holds for the import as for any write)."""
    source = MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(), embedder=HashEmbedder(64))
    target = MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(), embedder=HashEmbedder(64))
    try:
        memory_id = source.add("paid the taxes", user_id="ada", infer=False,
                               categories=["taxes"]).actions[0].memory_id
        backup = source.export_backup(user_id="ada")
        for table in ("topics", "memory_topics", "entities", "entity_mentions"):
            backup["tables"][table] = []  # the memory row alone, as an older export
        target.add("tax return", user_id="ada", infer=False, categories=["tax"])
        target.backend.insert_memory(Memory(content="x", user_id="ada", categories=["taxes"]))
        assert target.merge_tags(["taxes"], "tax", user_id="ada") == 1
        target.import_backup(backup, owner_prefix="ada")
        assert target.get(memory_id).categories == ["tax"]
        assert _index_of(target, memory_id) == ["tax"]
        assert _topics_of(target, memory_id) == ["tax"]
        assert target.categories(user_id="ada") == [{"category": "tax", "count": 3}]
        assert _active_topics(target, "taxes") == []
        _agree(target, "ada")
    finally:
        source.close()
        target.close()


def test_a_column_follows_tombstones_topic_to_topic_and_keeps_the_tag_of_a_thing(tagged):
    """"taxes" went into "tax", and "tax" was then found to be "Tax Office"
    the organization. A column naming "taxes" is written "tax", the last tag
    name of the chain, and its mention goes to the organization; the filter
    on "tax" finds it, and no topic of either name comes back."""
    insert = tagged.backend.insert_memory
    insert(Memory(content="a", user_id="ada", categories=["taxes"]))
    insert(Memory(content="b", user_id="ada", categories=["tax"]))
    assert tagged.merge_tags(["taxes"], "tax", user_id="ada") == 1
    office = tagged.backend.insert_entity(Entity(
        name="Tax Office", normalized="tax office", entity_type="organization",
        user_id="ada"))
    tax = tagged.backend.topic_entity("tax", Scope(user_id="ada"), create=False)
    assert tagged.backend.merge_entities(office.id, tax.id)
    later = insert(Memory(content="c", user_id="ada", categories=["taxes", "home"]))
    assert later.categories == ["tax", "home"]
    assert _index_of(tagged, later.id) == ["home", "tax"]
    assert [e.id for e in tagged.backend.entities_of_memory(later.id)] == [office.id]
    assert _topics_of(tagged, later.id) == ["home"]
    assert _active_topics(tagged, "tax") == [] and _active_topics(tagged, "taxes") == []
    assert later.id in {m.id for m in tagged.get_all(user_id="ada", categories=["tax"])}
    assert tagged.backend.tag_filing(["Taxes", "tax", "home", "new"], Scope(user_id="ada")) == {
        "taxes": "tax", "tax": "tax", "home": "home", "new": "new"}


@pytest.mark.parametrize("retire", ["not an entity", "name kept as a tag", "snapshot of before"])
def test_a_tag_folded_into_a_thing_is_a_tag_again_once_the_thing_is_retired(
        tagged, retire, monkeypatch):
    """"bildy" the tag went into "Bildy" the product, which was then retired
    (removed as not an entity, or removed keeping its name as a tag): the tag
    is a topic again and its memories are filed under it, a later save with
    the tag files there too, and restoring the product does not fold the tag
    back. Before, the tag's tombstone went with the product, the save made a
    fresh topic beside memories that mentioned nothing, and the restore
    brought the tombstone back: two entities claimed "bildy", and the count
    said 1 where the filter found 3. A product retired before this rule (its
    snapshot holds the tag's tombstone) is restored the same way: the
    tombstone points at the topic the save made, and the memories are filed
    under it."""
    from memry.backends.local import LocalBackend

    scope = Scope(user_id="ada")
    old = [tagged.add(f"note {i}", user_id="ada", infer=False,
                      categories=["bildy"]).actions[0].memory_id for i in range(2)]
    product = tagged.backend.insert_entity(Entity(
        name="Bildy", normalized="bildy", entity_type="product", user_id="ada"))
    tag = tagged.backend.topic_entity("bildy", scope, create=False)
    assert tagged.merge_entities(product.id, tag.id)
    assert tagged.backend.get_entity(tag.id).merged_into == product.id
    if retire == "name kept as a tag":
        assert tagged.remove_entity_preserving_tag(product.id)["removed"] == 1
    else:
        with monkeypatch.context() as patch:
            if retire == "snapshot of before":
                patch.setattr(LocalBackend, "_unfold_tags_locked", lambda self, entity_id: None)
            assert tagged.remove_entities([product.id], reason="not an entity") == 1
    new = tagged.add("note 2", user_id="ada", infer=False,
                     categories=["bildy"]).actions[0].memory_id
    assert tagged.restore_entities([product.id]) == 1

    claimants = {tagged.backend.resolve_entity_id(row["id"]) for row in tagged.backend._db.execute(
        "SELECT id FROM entities WHERE entity_type = 'topic' AND normalized = 'bildy' "
        "AND user_id = 'ada'")}
    [claimant] = claimants
    assert tagged.backend.get_entity(claimant).entity_type == TOPIC_TYPE
    if retire != "snapshot of before":
        assert claimant == tag.id  # the tag's own topic, active again
    filtered = {m.id for m in tagged.get_all(user_id="ada", categories=["bildy"])}
    assert filtered == {*old, new}
    assert tagged.categories(user_id="ada") == [{"category": "bildy", "count": 3}]
    assert {m.id for m in tagged.backend.entity_memories(claimant, limit=10)} == filtered
    assert tagged.backend.get_entity(product.id).merged_into is None  # restored, a thing
    _agree(tagged, "ada")


def test_an_edit_with_no_user_is_made_in_every_namespace_that_carries_the_tag(tagged):
    """An admin's rename (no user) renames "taxes" for each user that has it,
    each in full: the user's topic folds into a "levies" of that user, which
    takes over its description, the columns and mentions follow, and no
    "taxes" topic stays active anywhere. Before, the columns of every user
    were rewritten but only the topics without a user were merged. A merge
    and a delete with no user reach every namespace the same way."""
    from memry.models import utcnow

    memories = {}
    for user in ("ada", "bob"):
        memories[user] = tagged.add(f"{user} files taxes", user_id=user, infer=False,
                                    categories=["taxes"]).actions[0].memory_id
        topic = tagged.backend.topic_entity("taxes", Scope(user_id=user), create=False)
        tagged.backend.set_entity_description(topic.id, f"what {user} owes", utcnow())
    assert tagged.rename_tag("taxes", "levies") == 2
    assert _active_topics(tagged, "taxes") == []
    for user in ("ada", "bob"):
        scope = Scope(user_id=user)
        levies = tagged.backend.topic_entity("levies", scope, create=False)
        assert levies.user_id == user and levies.description == f"what {user} owes"
        assert tagged.backend.topic_entity(
            "taxes", scope, create=False, follow_merged=True).id == levies.id
        assert tagged.get(memories[user]).categories == ["levies"]
        assert tagged.categories(user_id=user) == [{"category": "levies", "count": 1}]
        _agree(tagged, user)

    tagged.add("ada pays duties", user_id="ada", infer=False, categories=["duties"])
    tagged.add("bob pays duties", user_id="bob", infer=False, categories=["duties"])
    assert tagged.merge_tags(["duties"], "levies") == 2
    assert _active_topics(tagged, "duties") == []
    assert [tagged.categories(user_id=user) for user in ("ada", "bob")] == [
        [{"category": "levies", "count": 2}]] * 2
    assert tagged.delete_tag("levies") == 4
    assert [tagged.categories(user_id=user) for user in ("ada", "bob")] == [[], []]
    for user in ("ada", "bob"):
        _agree(tagged, user)


# ------------------------------------------------------- fifth review round
@pytest.mark.parametrize("name", ["x" * 70, "steuernummer (tin, koeln vingst)"])
def test_a_tag_merges_into_the_chosen_topic_by_id_whatever_its_stored_name(tagged, name):
    """A stored tag name that ``clean_tags`` refuses (longer than a tag may be,
    or holding brackets and commas, as older imports and the migration left
    them) stays the topic's name, and merging another tag into that topic
    merges it into that topic by id. Before, the name was cleaned first: the
    long one cleaned to nothing and the merge deleted the other tag; the
    bracketed one cleaned to "steuernummer" and the other tag went into a
    fresh topic of that name; either way the merge then reported failure,
    after the data had changed."""
    insert = tagged.backend.insert_memory
    scope = Scope(user_id="ada")
    kept = insert(Memory(content="a", user_id="ada", categories=[name])).id
    moved = insert(Memory(content="b", user_id="ada", categories=["tax id"])).id
    keep = tagged.backend.topic_entity(name, scope, create=False)
    other = tagged.backend.topic_entity("tax id", scope, create=False)
    assert keep is not None and keep.name == name
    assert tagged.merge_entities(keep.id, other.id)
    assert tagged.backend.get_entity(other.id).merged_into == keep.id
    assert tagged.backend.get_entity(keep.id).name == name  # kept as it was
    assert [tagged.get(m).categories for m in (kept, moved)] == [[name], [name]]
    assert tagged.categories(user_id="ada") == [{"category": name, "count": 2}]
    assert _active_topics(tagged, "steuernummer") == []
    _agree(tagged, "ada")
    # merged by name, the tag that exists is taken as stored too
    third = insert(Memory(content="c", user_id="ada", categories=["tin"])).id
    assert tagged.merge_tags(["tin"], name, user_id="ada") == 1
    assert tagged.get(third).categories == [name]
    assert _active_topics(tagged, name) == [keep.id]
    _agree(tagged, "ada")


def test_a_refused_tag_merge_changes_nothing(tagged, monkeypatch):
    """The fold is validated and written before any column is: when the
    backend refuses it (the other side was folded meanwhile), no memory is
    retagged and the merge says it failed."""
    insert = tagged.backend.insert_memory
    scope = Scope(user_id="ada")
    ids = [insert(Memory(content=c, user_id="ada", categories=[t])).id
           for c, t in (("a", "tech"), ("b", "technical"))]
    tech = tagged.backend.topic_entity("tech", scope, create=False)
    technical = tagged.backend.topic_entity("technical", scope, create=False)
    monkeypatch.setattr(tagged.backend, "merge_entities", lambda keep, merge: False)
    assert not tagged.merge_entities(tech.id, technical.id)
    assert [tagged.get(m).categories for m in ids] == [["tech"], ["technical"]]
    assert tagged.backend.get_entity(technical.id).merged_into is None
    _agree(tagged, "ada")


def test_a_merge_that_finds_its_side_folded_meanwhile_leaves_no_transaction_open(
        tmp_path, monkeypatch):
    """The fold's UPDATE matched no row (another process folded the side
    after it was resolved): the merge fails and rolls back, so the database
    is not left locked for every other connection."""
    import sqlite3

    from memry.backends.local import LocalBackend

    path = tmp_path / "memry.db"
    backend = LocalBackend(str(path))
    try:
        a, b, c = (backend.insert_entity(Entity(name=n, normalized=n.lower(), user_id="ada"))
                   for n in ("Ada", "Ada L", "Ada Lovelace"))
        assert backend.merge_entities(a.id, b.id)
        # the resolution read before the other process's fold
        monkeypatch.setattr(backend, "resolve_entity_id", lambda entity_id: entity_id)
        assert backend.merge_entities(c.id, b.id) is False
        assert not backend._db.in_transaction
        other = sqlite3.connect(str(path), timeout=0)
        try:
            other.execute("INSERT INTO meta (key, value) VALUES ('probe', '1')")
            other.commit()
        finally:
            other.close()
    finally:
        backend.close()


def test_each_name_merged_away_is_filed_under_its_own_survivor(tagged):
    """ "tax" went into "levies" and "taxes" into "duties": a save tagged
    "taxes" is filed under "duties", one tagged "Tax" under "levies". Before,
    the two retired names were grouped as obvious variants and both went to
    the survivor of the first ("levies")."""
    insert = tagged.backend.insert_memory
    for content, tag in (("a", "tax"), ("b", "taxes"), ("c", "levies"), ("d", "duties")):
        insert(Memory(content=content, user_id="ada", categories=[tag]))
    assert tagged.merge_tags(["tax"], "levies", user_id="ada") == 1
    assert tagged.merge_tags(["taxes"], "duties", user_id="ada") == 1
    plural = tagged.add("paid the taxes", user_id="ada", infer=False,
                        categories=["taxes"]).actions[0].memory_id
    single = tagged.add("tax office letter", user_id="ada", infer=False,
                        categories=["Tax"]).actions[0].memory_id
    assert tagged.get(plural).categories == ["duties"]
    assert tagged.get(single).categories == ["levies"]
    assert tagged.categories(user_id="ada") == [
        {"category": "duties", "count": 3}, {"category": "levies", "count": 3}]
    assert _active_topics(tagged, "tax") == [] and _active_topics(tagged, "taxes") == []
    patched = tagged.update(single, categories=["taxes", "tax"])
    assert patched.categories == ["duties", "levies"]
    _agree(tagged, "ada")


def test_a_restored_thing_brings_back_its_named_mentions_only(tagged):
    """"bildy" the tag was found to be "Bildy" the product (a pair raised and
    confirmed), then the product was retired and restored. Retiring unfolds
    the tag and takes the product's mentions that came from the tag with it,
    so the restore brings back the named mention only: each memory mentions
    one of the two, the counts agree with the column and the filter, and the
    pair is not raised again. Before, the restore put the tag's mentions back
    on the product beside the topic's own (the product counted 3 memories
    where one names it) and the pair was raised again."""
    from memry.intelligence.entities import propose_same_name_duplicates

    scope = Scope(user_id="ada")
    product = tagged.backend.insert_entity(Entity(
        name="Bildy", normalized="bildy", entity_type="product", user_id="ada"))
    named = tagged.backend.insert_memory(Memory(content="Bildy runs on AWS", user_id="ada"))
    tagged.backend.add_mention(EntityMention(entity_id=product.id, memory_id=named.id,
                                             surface="Bildy"))
    notes = [tagged.add(f"note {i}", user_id="ada", infer=False,
                        categories=["bildy"]).actions[0].memory_id for i in range(2)]
    tag = tagged.backend.topic_entity("bildy", scope, create=False)
    assert propose_same_name_duplicates(backend=tagged.backend, scope=scope) == 1
    [pair] = tagged.merge_proposals(user_id="ada")
    assert {pair.entity_a, pair.entity_b} == {product.id, tag.id}
    assert tagged.confirm_merge(pair.id)
    assert tagged.backend.count_entity_memories(product.id) == 3

    assert tagged.remove_entities([product.id], reason="not an entity") == 1
    assert tagged.restore_entities([product.id]) == 1
    assert [e.id for e in tagged.backend.entities_of_memory(named.id, kind="any")] == [
        product.id]
    for note in notes:
        assert [e.id for e in tagged.backend.entities_of_memory(note, kind="any")] == [tag.id]
    assert tagged.backend.count_entity_memories(product.id) == 1
    assert tagged.backend.count_entity_memories(tag.id) == 2
    assert tagged.categories(user_id="ada") == [{"category": "bildy", "count": 2}]
    assert {m.id for m in tagged.get_all(user_id="ada", categories=["bildy"])} == set(notes)
    assert propose_same_name_duplicates(backend=tagged.backend, scope=scope) == 0
    assert tagged.merge_proposals(user_id="ada") == []
    _agree(tagged, "ada")


def test_a_deleted_tag_leaves_no_topic_behind(tagged):
    """delete_tag("groceries") takes the tag off every memory and retires its
    topic entity with its last mention, so no "groceries" topic stays active
    with nothing filed under it and no pair with "Groceries" the thing is
    raised for it. Before, the topic stayed forever: the orphan purge skipped
    tags. The purge now retires a topic nothing mentions and no tombstone
    points at; a later save of the tag makes a fresh topic, and the retired
    one is not restored over it."""
    from memry.intelligence.entities import propose_same_name_duplicates

    scope = Scope(user_id="ada")
    thing = tagged.backend.insert_entity(Entity(
        name="Groceries", normalized="groceries", entity_type="concept", user_id="ada"))
    named = tagged.add("Groceries are bought on Saturdays", user_id="ada",
                       infer=False).actions[0].memory_id
    tagged.backend.add_mention(EntityMention(entity_id=thing.id, memory_id=named,
                                             surface="Groceries"))
    for i in range(2):
        tagged.add(f"bought milk {i}", user_id="ada", infer=False, categories=["groceries"])
    old = tagged.backend.topic_entity("groceries", scope, create=False)
    assert tagged.delete_tag("groceries", user_id="ada") == 2
    assert _active_topics(tagged, "groceries") == []
    assert tagged.backend.get_entity(old.id) is None
    assert propose_same_name_duplicates(backend=tagged.backend, scope=scope) == 0
    assert tagged.backend.list_proposals(scope, status=None) == []
    assert tagged.backend.get_entity(thing.id).merged_into is None

    # the purge: a stray topic goes, one a tombstone points at stays
    stray = tagged.backend.topic_entity("stray", scope)
    tax = tagged.backend.topic_entity("tax", scope)
    taxes = tagged.backend.topic_entity("taxes", scope)
    assert tagged.backend.merge_entities(tax.id, taxes.id)
    assert tagged.backend.purge_orphan_entities(scope) == 1
    assert tagged.backend.get_entity(stray.id) is None
    assert tagged.backend.get_entity(tax.id).merged_into is None
    assert tagged.backend.resolve_entity_id(taxes.id) == tax.id

    later = tagged.add("bought bread", user_id="ada", infer=False,
                       categories=["groceries"]).actions[0].memory_id
    [fresh] = _active_topics(tagged, "groceries")
    assert fresh != old.id
    assert tagged.restore_entities([old.id]) == 0  # the name lives on in a fresh topic
    assert _active_topics(tagged, "groceries") == [fresh]
    assert _topics_of(tagged, later) == ["groceries"]
    _agree(tagged, "ada")


def test_an_open_pair_of_a_thing_and_its_tag_never_draws_the_tags_memories(
        tagged, monkeypatch):
    """ "Groceries" the thing and "groceries" the tag, raised as a pair and
    not decided yet: the pair is a "same" link at 0.5, which reached the tag,
    and every question naming the thing read up to FAMILY_SCAN of the tag's
    memories. A tag is never what the linked search is about, so no link
    reaches it, as none makes it a seed."""
    from memry.intelligence.graph_retrieval import activation_paths
    from memry.models import MergeProposal

    scope = Scope(user_id="ada")
    thing = tagged.backend.insert_entity(Entity(
        name="Groceries", normalized="groceries", entity_type="concept", user_id="ada"))
    for text in ("Groceries are bought on Saturdays", "Groceries come from Lidl",
                 "Groceries cost 80 euros a week"):
        memory_id = tagged.add(text, user_id="ada", infer=False).actions[0].memory_id
        tagged.backend.add_mention(EntityMention(entity_id=thing.id, memory_id=memory_id,
                                                 surface="Groceries"))
    for i in range(5):
        tagged.add(f"bought milk {i}", user_id="ada", infer=False, categories=["groceries"])
    tag = tagged.backend.topic_entity("groceries", scope, create=False)
    tagged.backend.add_proposal(MergeProposal(
        entity_a=thing.id, entity_b=tag.id, user_id="ada", confidence=0.5,
        reason="not yet compared"))
    assert tagged._is_hub(thing.id)
    act, _ = activation_paths(tagged.backend, [thing.id], depth=1)
    assert act == {thing.id: 1.0}

    read: list[str] = []
    entity_memories = tagged.backend.entity_memories

    def spy(entity_id, **kwargs):
        read.append(entity_id)
        return entity_memories(entity_id, **kwargs)

    monkeypatch.setattr(tagged.backend, "entity_memories", spy)
    tagged.search("When are Groceries bought?", user_id="ada", limit=10)
    assert thing.id in read and tag.id not in read
