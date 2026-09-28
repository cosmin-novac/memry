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
    assert tagged.direct_categories(user_id="bob") == [{"category": "work", "count": 1}]


def test_the_vocabulary_offered_to_extraction_is_the_same_as_before(tagged):
    """Read from topic entities, the vocabulary holds the tags the legacy index
    held, in the same order: most used first, then by name."""
    for i, tags in enumerate([["tax"], ["tax", "home"], ["Home"], ["garden"],
                              ["tax"], ["b-tag"], ["a tag"]]):
        tagged.add(f"note {i}", user_id="ada", infer=False, categories=tags)
    scope = Scope(user_id="ada")
    legacy = [row["category"] for row in tagged.backend.direct_topic_counts(scope)]
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
    """A column the merge did not rewrite (a memory invalidated before merges
    reached invalid memories, a restored backup, an import) names the tag
    merged away: its mention follows the merge to the surviving topic instead
    of bringing the old one back, and so does the filter index, so the count
    and the filter agree. A filter on the name merged away finds nothing."""
    a = tagged.backend.insert_memory(Memory(content="a", user_id="ada", categories=["taxes"]))
    b = tagged.backend.insert_memory(Memory(content="b", user_id="ada", categories=["tax"]))
    tax = tagged.backend.topic_entity("tax", Scope(user_id="ada"), create=False)
    assert tagged.merge_tags(["taxes"], "tax", user_id="ada") == 1
    old = tagged.backend.insert_memory(Memory(content="c", user_id="ada", categories=["taxes"]))
    assert tagged.backend.get_memory(old.id).categories == ["taxes"]  # kept as written
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


def test_a_tag_merged_back_the_other_way_keeps_the_name_asked_for(tagged):
    """Merging "tax" into "taxes" and then "taxes" into "tax" ends with one
    active topic named "tax". No survivor is renamed: "tax" gets a topic of
    its own and "taxes" folds into it, so both names resolve to that topic
    and "taxes" keeps its tombstone. A column still saying "taxes" afterwards
    mentions it and creates no topic."""
    tagged.backend.insert_memory(Memory(content="a", user_id="ada", categories=["tax"]))
    tagged.backend.insert_memory(Memory(content="b", user_id="ada", categories=["taxes"]))
    tagged.merge_tags(["tax"], "taxes", user_id="ada")
    assert tagged.categories(user_id="ada") == [{"category": "taxes", "count": 2}]
    tagged.merge_tags(["taxes"], "tax", user_id="ada")
    assert tagged.categories(user_id="ada") == [{"category": "tax", "count": 2}]
    assert _active_topics(tagged, "taxes") == []
    [tax] = _active_topics(tagged, "tax")
    assert _tombstones(tagged, "taxes") and _tombstones(tagged, "tax")
    for tombstone in _tombstones(tagged, "taxes") + _tombstones(tagged, "tax"):
        assert tagged.backend.resolve_entity_id(tombstone) == tax
    _agree(tagged, "ada")
    later = tagged.backend.insert_memory(Memory(content="c", user_id="ada",
                                                categories=["taxes"]))
    assert [e.id for e in tagged.backend.entities_of_memory(later.id, kind="topic")] == [tax]
    assert _active_topics(tagged, "taxes") == [] and _active_topics(tagged, "tax") == [tax]
    assert tagged.categories(user_id="ada") == [{"category": "tax", "count": 3}]
    assert len(tagged.get_all(user_id="ada", categories=["tax"])) == 3


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


def test_a_tag_merged_away_can_be_asked_for_again(tagged):
    """After "tax" went into "taxes", merging "levies" into "tax" makes "tax"
    a topic again, of the levies alone; "taxes" keeps its memories."""
    for content, tag in (("a", "tax"), ("b", "taxes"), ("c", "levies")):
        tagged.backend.insert_memory(Memory(content=content, user_id="ada", categories=[tag]))
    tagged.merge_tags(["tax"], "taxes", user_id="ada")
    assert tagged.merge_tags(["levies"], "tax", user_id="ada") == 1
    assert tagged.categories(user_id="ada") == [
        {"category": "taxes", "count": 2}, {"category": "tax", "count": 1}]
    assert len(_active_topics(tagged, "tax")) == 1 and _active_topics(tagged, "levies") == []
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
    assert (money.name, money.merged_into, money.metadata) == ("money", None, {})
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


def test_a_tag_and_a_named_thing_of_its_name_go_through_the_identity_funnel():
    """ "bildy" the tag against "Bildy" the product: the ordinary entity pair
    question decides, with the tag's memories as its side."""
    llm = FakeLLM()
    config = Config(db_path=":memory:")
    config.decision.auto_confirm_confidence = 0.95
    store = MemoryStore(config, llm=llm, embedder=HashEmbedder(64))
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

        llm.queue(json.dumps({"verdict": "same", "confidence": 0.99, "reason": "one app"}))
        outcome = store.resolve_entities(user_id="ada")
        assert (outcome["proposed"], outcome["confirmed"]) == (1, 1)
        asked = llm.calls[-1][1]
        assert "Shipped the invoice export" in asked and "Bildy runs on AWS" in asked
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
    tagged.config.retrieval.relational_fusion = "linked"
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

    tagged.config.retrieval.relational_fusion = "linked"
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
        db.execute("DROP INDEX IF EXISTS ux_entities_active_topic")
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
            "SELECT 1 FROM sqlite_master WHERE name = 'ux_entities_active_topic'").fetchone()
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
