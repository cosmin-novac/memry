"""Splitting a memory that holds several facts (``memry split-memories``).

Merges before the rule that a merge keeps one fact folded each new claim
about one subject into the memory before it. The repair asks the text model
to split such a memory into single facts and replaces it with them: each
fact keeps the old memory's dates, sources, tags, run and agent, the entity
links and relations go to the facts that name them, and the old memory is
listed under Archive, where its undo brings it back and forgets the facts.
A stub text model answers here.
"""

from __future__ import annotations

import json

from conftest import FakeLLM
from starlette.testclient import TestClient

from memry.cli import format_split_report
from memry.config import Config
from memry.intelligence.split import (
    SPLIT_SYSTEM, is_candidate, proper_names, sentences, without_subject)
from memry.models import Entity, EntityMention, Memory, MemoryEvent, Relation, Scope
from memry.providers.embeddings import HashEmbedder
from memry.rest import create_app
from memry.store import MemoryStore

SAID = "2026-03-02T09:00:00+00:00"
LAST = "2026-05-11T09:00:00+00:00"
ESSAY = ("The central claim of Ana Reyes's thesis is that small models can match large ones on "
         "narrow tasks. The thesis further argues that benchmark contamination explains most "
         "reported gains. Ana Reyes explicitly accepts that large models write better prose, "
         "as her advisor Tom Hale pointed out.")
FACTS = [
    "The central claim of Ana Reyes's thesis is that small models can match large ones on "
    "narrow tasks.",
    "Ana Reyes's thesis argues that benchmark contamination explains most reported gains.",
    "Ana Reyes explicitly accepts that large models write better prose, as her advisor Tom Hale "
    "pointed out.",
]


def split_answer(*facts: str) -> str:
    return json.dumps({"facts": list(facts)})


def audit(*missing: str) -> str:
    return json.dumps({"missing": list(missing)})


def _store(llm: FakeLLM) -> MemoryStore:
    return MemoryStore(Config(db_path=":memory:"), llm=llm, embedder=HashEmbedder(64))


def _world(store: MemoryStore) -> dict:
    """The essay as merges left it: saved in three saves of run r2 by agent
    a1, tagged, dated, linked to Ana Reyes, the thesis (named by no single
    fact as a whole) and Tom Hale, with a relation between Ana and Tom."""
    episodes = store.add("Ana's thesis says small models can match large ones.", user_id="ada",
                         agent_id="a1", run_id="r2", infer=False, created_at=SAID).episode_ids
    essay = store.backend.insert_memory(Memory(
        content=ESSAY, user_id="ada", agent_id="a1", run_id="r2", importance=0.8,
        memory_type="semantic", categories=["phd thesis", "ai research"],
        entities=["Ana Reyes", "Tom Hale"],
        metadata={"context": "thesis notes", "when": {"start": "2026-03-01"}},
        created_at=SAID, updated_at=LAST, valid_from=SAID,
        source_episode_ids=[*episodes, "e-2", "e-3"], embedding_model=store.embedder.model_id),
        embedding=store.embedder.embed([ESSAY])[0])
    ana = store.backend.insert_entity(Entity(name="Ana Reyes", normalized="ana reyes",
                                             entity_type="person", user_id="ada"))
    tom = store.backend.insert_entity(Entity(name="Tom Hale", normalized="tom hale",
                                             entity_type="person", user_id="ada"))
    lab = store.backend.insert_entity(Entity(name="Reyes Lab", normalized="reyes lab",
                                             entity_type="organization", user_id="ada"))
    for entity in (ana, tom, lab):
        store.backend.add_mention(EntityMention(entity_id=entity.id, memory_id=essay.id,
                                                surface=entity.name, entity_type="person"))
    store.backend.add_relation(Relation(subject=tom.id, predicate="advises", object=ana.id,
                                        user_id="ada", memory_id=essay.id))
    # made on the first save, merged into until the last
    store.backend.add_event(MemoryEvent(memory_id=essay.id, event="ADD", created_at=SAID,
                                        new_content=FACTS[0]))
    store.backend.add_event(MemoryEvent(memory_id=essay.id, event="UPDATE", created_at=LAST,
                                        new_content=ESSAY))
    # the first save's own memory is gone, as the merges left it
    for memory in store.get_all(user_id="ada"):
        if memory.id != essay.id:
            store.backend.invalidate_memory(memory.id)
    store.repair_updated_at(user_id="ada")
    return {"essay": essay, "ana": ana, "tom": tom, "lab": lab}


def _linked(store: MemoryStore, memory_id: str) -> set[str]:
    return {e.name for e in store.backend.entities_of_memory(memory_id)}


def _relations(store: MemoryStore) -> list[tuple[str, str | None]]:
    return [(r.predicate, r.memory_id) for r in store.backend.list_relations(Scope(user_id="ada"))
            if r.invalid_at is None]


def test_a_memory_of_several_facts_is_split_keeping_what_it_rests_on():
    llm = FakeLLM([split_answer(*FACTS), audit()])
    store = _store(llm)
    world = _world(store)
    essay = world["essay"]

    summary = store.split_memories(user_id="ada")

    assert {k: summary[k] for k in ("in_use", "candidates", "one_fact", "split", "facts",
                                    "lossy", "failed")} == {
        "in_use": 1, "candidates": 1, "one_fact": 0, "split": 1, "facts": 3, "lossy": 0,
        "failed": 0}
    system, prompt = llm.calls[0]
    assert system == SPLIT_SYSTEM and prompt == f"Memory:\n{ESSAY}\n\nSplit it as JSON."
    [entry] = summary["splits"]
    ids = entry["memory_ids"]
    parts = [store.get(i) for i in ids]
    assert [p.content for p in parts] == FACTS
    for part in parts:
        assert (part.created_at, part.updated_at, part.valid_from) == (SAID, LAST, SAID)
        assert part.source_episode_ids == essay.source_episode_ids
        assert (part.user_id, part.agent_id, part.run_id) == ("ada", "a1", "r2")
        assert part.categories == ["phd thesis", "ai research"]
        assert (part.importance, part.memory_type) == (0.8, "semantic")
        assert part.metadata == {"context": "thesis notes", "when": {"start": "2026-03-01"},
                                 "split_from": essay.id}
        assert part.invalid_at is None and part.embedding_model == store.embedder.model_id
    # each named thing on the facts that name it; one no fact names on all of them
    assert [_linked(store, i) for i in ids] == [
        {"Ana Reyes", "Reyes Lab"}, {"Ana Reyes", "Reyes Lab"},
        {"Ana Reyes", "Tom Hale", "Reyes Lab"}]
    assert [p.entities for p in parts] == [["Ana Reyes"], ["Ana Reyes"], ["Ana Reyes", "Tom Hale"]]
    # the relation rests on the fact naming both ends
    assert _relations(store) == [("advises", ids[2])]
    # the old memory is out of use and out of search, kept with the event
    old = store.get(essay.id)
    assert old.invalid_at is not None and old.superseded_by == ids[0]
    assert old.metadata["split_into"] == ids
    [event] = [e for e in store.history(essay.id) if e.event == "SUPERSEDE"]
    assert event.kind == "split" and event.old_content == ESSAY
    assert event.new_content == "\n".join(FACTS)
    found = [r.memory.id for r in store.search("Ana Reyes thesis contamination", user_id="ada")]
    assert essay.id not in found and ids[1] in found
    for i in ids:  # dated at the memory's time, so repair-dates keeps it
        [added] = store.history(i)
        assert added.event == "ADD" and added.created_at == LAST
        assert f"from memory {essay.id}, one of its 3 facts" in added.reason
    assert store.repair_updated_at(user_id="ada") == {"fixed": 0}
    assert store.get_all(user_id="ada", run_id="r2", agent_id="a1", limit=10)
    store.close()


def test_a_split_is_listed_under_archive_and_its_undo_brings_the_memory_back():
    llm = FakeLLM([split_answer(*FACTS), audit()])
    store = _store(llm)
    world = _world(store)
    essay = world["essay"]
    ids = store.split_memories(user_id="ada")["splits"][0]["memory_ids"]

    [row] = store.replaced(user_id="ada")
    assert row["memory"].id == essay.id and row["split"] is True
    assert row["contradiction"] is False
    assert [p.id for p in row["parts"]] == ids

    assert store.undo_replacement(essay.id)
    back = store.get(essay.id)
    assert back.invalid_at is None and back.superseded_by is None
    assert "split_into" not in back.metadata
    assert [m.id for m in store.get_all(user_id="ada")] == [essay.id]
    for i in ids:
        gone = store.get(i)
        assert gone.invalid_at is not None
        assert store.history(i)[-1].event == "DELETE"
    assert _relations(store) == [("advises", essay.id)]
    assert _linked(store, essay.id) == {"Ana Reyes", "Tom Hale", "Reyes Lab"}
    assert store.replaced(user_id="ada") == []
    assert store.history(essay.id)[-1].reason == "you undid its split into 3 facts"
    found = [r.memory.id for r in store.search("Ana Reyes thesis contamination", user_id="ada")]
    assert found[0] == essay.id and not set(ids) & set(found)
    store.close()


def test_the_undo_leaves_a_fact_that_changed_since():
    """A fact merged with a later detail, or deleted, since the split is left
    as it is; the others are forgotten."""
    llm = FakeLLM([split_answer(*FACTS), audit()])
    store = _store(llm)
    essay = _world(store)["essay"]
    ids = store.split_memories(user_id="ada")["splits"][0]["memory_ids"]
    store.delete(ids[1])
    assert store.undo_replacement(essay.id)
    assert [e.event for e in store.history(ids[1])] == ["ADD", "DELETE"]
    assert store.history(ids[1])[-1].reason == "manual delete (invalidated)"
    store.close()


def test_a_dry_run_asks_and_writes_nothing():
    llm = FakeLLM([split_answer(*FACTS), audit()])
    store = _store(llm)
    _world(store)
    before = store.export_backup(user_id="ada")

    summary = store.split_memories(user_id="ada", dry_run=True)

    assert (summary["dry_run"], summary["split"], summary["facts"]) == (True, 1, 3)
    [entry] = summary["splits"]
    assert entry["facts"] == FACTS and "memory_ids" not in entry
    after = store.export_backup(user_id="ada")
    assert after["tables"] == before["tables"]
    report = format_split_report([summary])
    assert "1 would be split into 3 facts" in report and "dry run: nothing written" in report
    assert f"  3. {FACTS[2]}" in report
    store.close()


def test_a_memory_of_one_fact_is_left_alone():
    """One sentence is one statement, and is not asked about. Two sentences
    that the model reads as one fact (a fact and its detail) are left alone
    too."""
    one = "Tom has a dog named Rex, a three-year-old German shepherd."
    two = "Tom has a dog named Rex. Rex is a three-year-old German shepherd."
    llm = FakeLLM([split_answer(two)])
    store = _store(llm)
    for text in (one, two):
        store.backend.insert_memory(Memory(content=text, user_id="ada", created_at=SAID,
                                           updated_at=SAID))
    before = store.export_backup(user_id="ada")

    summary = store.split_memories(user_id="ada")

    assert (summary["candidates"], summary["one_fact"], summary["split"]) == (1, 1, 0)
    assert len(llm.calls) == 1 and two in llm.calls[0][1]
    assert store.export_backup(user_id="ada")["tables"] == before["tables"]
    store.close()


def test_a_split_that_would_lose_a_detail_is_not_made():
    llm = FakeLLM([split_answer(*FACTS[:2]), audit("Tom Hale pointed it out")])
    store = _store(llm)
    essay = _world(store)["essay"]
    summary = store.split_memories(user_id="ada")
    assert (summary["split"], summary["lossy"]) == (0, 1)
    assert "Tom Hale pointed it out" in summary["splits"][0]["not_split"]
    assert store.get(essay.id).invalid_at is None
    store.close()


class SentenceSplitter(FakeLLM):
    """Answers by the prompt, whatever order the calls come in: a memory is
    split at its sentences, and the audit misses nothing."""

    def complete(self, system, user, *, json_schema=None):
        self.calls.append((system, user))
        if system == SPLIT_SYSTEM:
            text = user.split("Memory:\n", 1)[1].split("\n\nSplit it", 1)[0]
            return split_answer(*sentences(text))
        return audit()


def test_many_memories_are_asked_at_once_and_written_in_order():
    llm = SentenceSplitter()
    store = _store(llm)
    texts = [f"Tom's project {n} uses Go. Tom's project {n} runs on one VPS." for n in range(9)]
    ids = [store.backend.insert_memory(Memory(content=t, user_id="ada", created_at=SAID,
                                              updated_at=SAID)).id for t in texts]
    summary = store.split_memories(user_id="ada")
    assert (summary["candidates"], summary["split"], summary["facts"]) == (9, 9, 18)
    assert sorted(e["memory_id"] for e in summary["splits"]) == sorted(ids)
    assert len(llm.calls) == 18
    live = sorted(m.content for m in store.get_all(user_id="ada", limit=100))
    assert live == sorted(s for t in texts for s in sentences(t))
    store.close()


def test_the_split_prompt_asks_every_fact_to_name_its_subject():
    """Read alone, months later, a fact must say what it is about: 2 of 12
    two-fact splits of a real store lost it ("Merge the first change first"
    with no project), and such a fact is worse than the memory it came
    from."""
    rule = " ".join(SPLIT_SYSTEM.split())
    assert ("Every fact must stand alone, read on its own months later with nothing around "
            "it. It names its subject by the name the memory uses for it") in rule
    assert ('Never write "he", "she", "it", "they", "this", "the thesis", "the project" or '
            '"the repository" without the name') in rule
    assert "never leave an instruction or a decision without the thing it is about" in rule


def test_the_split_prompt_keeps_a_list_as_one_fact():
    """Without the rule, the model split one memory's list of skills into 23
    facts that each named one skill, and a review's seven unmeasured cells
    into seven facts: one vector each for what is one fact."""
    rule = " ".join(SPLIT_SYSTEM.split())
    assert ('A list is one fact too: "Ada\'s skills include Python, SQL and Go." stays '
            "whole") in rule
    assert ("An item becomes a fact of its own only when it carries details of its own "
            "(its own price, date or reason).") in rule


def test_a_fact_that_does_not_state_its_subject_is_found():
    memory = Memory(content="PR 12 in the Tern repository builds on PR 11. Merge PR 11 first.",
                    categories=["tern releases"])
    linked = ["Tern repository", "Tern"]
    assert without_subject(["PR 12 in the Tern repository builds on PR 11.",
                            "Merge PR 11 first."], memory, linked) == ["Merge PR 11 first."]
    assert without_subject(["PR 12 in the Tern repository builds on PR 11.",
                            "In Tern, merge PR 11 before PR 12."], memory, linked) == []
    # a pointing word fails whatever the fact names later
    assert without_subject(["It builds on PR 11 in Tern."], memory, linked) == [
        "It builds on PR 11 in Tern."]
    # a tag the memory's text names is a subject too
    tagged = Memory(content="The kitchen renovation budget is $18,000; oak cabinets chosen.",
                    categories=["kitchen renovation"])
    assert without_subject(["The kitchen renovation budget is $18,000.",
                            "Oak cabinets were chosen."], tagged, []) == [
        "Oak cabinets were chosen."]
    # linked to nothing it names: the names its text states
    loose = Memory(content="Maria chose oak cabinets from Holt Joinery; the budget is $18,000.")
    assert proper_names(loose.content) == ["Holt Joinery"]
    assert without_subject(["Maria chose oak cabinets from Holt Joinery.",
                            "The budget is $18,000."], loose, []) == ["The budget is $18,000."]
    # a text that names nothing is checked for pointing words only
    plain = Memory(content="the user likes tea; the user dislikes coffee.")
    assert without_subject(["the user likes tea.", "They dislike coffee."], plain, []) == [
        "They dislike coffee."]


def test_a_fact_may_name_the_owner_a_linked_thing_or_a_name_the_memory_states():
    """Of 51 long memories of a real store, the first check kept 18 whole,
    several for facts that did say what they were about: "The user prefers
    ...", a linked thing the memory's text spells another way, a model name
    the text states. A one-word abbreviation names no subject."""
    about_owner = Memory(content="Prefers green tea. Writes Python at Corlan Labs.")
    linked = ["Corlan Labs"]
    assert without_subject(["The user prefers green tea.",
                            "The user writes Python at Corlan Labs."],
                           about_owner, linked, ["Ada Lind"]) == []
    assert without_subject(["Ada Lind prefers green tea."], about_owner, linked,
                           ["Ada Lind"]) == []
    assert without_subject(["Prefers green tea."], about_owner, linked, ["Ada Lind"]) == [
        "Prefers green tea."]
    # a German genitive names the owner too
    assert without_subject(["Adas Umsatz 2025 betrug 7.300 €."],
                           Memory(content="Umsatz 2025: 7.300 €. Kunde: Corlan Labs."),
                           linked, ["Ada"]) == []
    # a linked thing the text spells another way, and a name the text states
    prices = Memory(content="The M4 Max costs €2,829 refurbished. The M3 Ultra starts at €4,295.")
    assert proper_names(prices.content) == ["M4 Max", "M3 Ultra"]
    assert without_subject(["The Mac Studio M4 Max costs €2,829 refurbished.",
                            "The M3 Ultra starts at €4,295."], prices, ["Mac Studio"]) == []
    # "PR" and "API" say what kind of thing, not which one
    review = Memory(content="Tern's PR 12 changes the API. Merge PR 11 first.")
    assert without_subject(["Tern's PR 12 changes the API.", "Merge PR 11 first."],
                           review, ["Tern"]) == ["Merge PR 11 first."]


def test_a_split_with_a_fact_that_lost_its_subject_is_not_made():
    """The check runs before anything is written, with no call to the model
    beyond the split: a fact that names none of the things the memory is
    linked to or tagged with keeps the whole memory, and the dry run says
    which fact."""
    orphan = "The thesis further argues that benchmark contamination explains most gains."
    llm = FakeLLM([split_answer(FACTS[0], orphan, FACTS[2])])
    store = _store(llm)
    essay = _world(store)["essay"]
    before = store.export_backup(user_id="ada")

    summary = store.split_memories(user_id="ada", dry_run=True)

    assert (summary["split"], summary["no_subject"], summary["lossy"]) == (0, 1, 0)
    assert len(llm.calls) == 1  # no audit for a split that is not made
    [entry] = summary["splits"]
    assert entry["not_split"] == f"a fact would not state its subject: {orphan}"
    assert "1 left because a fact would not state its subject" in format_split_report([summary])
    assert store.export_backup(user_id="ada")["tables"] == before["tables"]
    llm.queue(split_answer(FACTS[0], orphan, FACTS[2]))
    assert store.split_memories(user_id="ada")["no_subject"] == 1
    assert store.get(essay.id).invalid_at is None
    store.close()


def test_which_memories_are_asked():
    """More than one sentence, not waiting for extraction or for a person;
    ``min_words`` narrows the run. Titles do not end a sentence."""
    assert sentences("Dr. Okafor holds that sleep debt cannot be repaid. She cites a study.") == [
        "Dr. Okafor holds that sleep debt cannot be repaid.", "She cites a study."]
    assert sentences("Maria chose oak; the budget is $18,000.") == [
        "Maria chose oak;", "the budget is $18,000."]
    assert sentences("Tom works at Corlan Inc. in Leeds.") == [
        "Tom works at Corlan Inc. in Leeds."]
    two = Memory(content="Tom likes tea. He drinks green tea daily.")
    assert is_candidate(two) and not is_candidate(two, min_words=20)
    assert not is_candidate(Memory(content="Tom likes green tea, daily."))
    assert not is_candidate(two.model_copy(update={"metadata": {"pending_distillation": True}}))
    assert not is_candidate(two.model_copy(update={"metadata": {"conflict": {"with": "x"}}}))
    assert not is_candidate(two.model_copy(update={"invalid_at": SAID}))


def test_the_rest_endpoint_splits_and_the_archive_shows_the_facts():
    llm = FakeLLM([split_answer(*FACTS), audit(), split_answer(*FACTS), audit()])
    store = _store(llm)
    essay = _world(store)["essay"]
    with TestClient(create_app(store)) as client:
        dry = client.post("/api/v1/memories/split", json={"user_id": "ada", "dry_run": True})
        assert dry.status_code == 200 and dry.json()["split"] == 1
        assert store.get(essay.id).invalid_at is None
        done = client.post("/api/v1/memories/split", json={"user_id": "ada"}).json()
        ids = done["splits"][0]["memory_ids"]
        [row] = client.get("/api/v1/memories/replaced", params={"user_id": "ada"}).json()
        assert row["split"] is True and [p["id"] for p in row["parts"]] == ids
        undone = client.post(f"/api/v1/memories/{essay.id}/undo-replacement", json={})
        assert undone.json() == {"restored": True}
    assert store.get(essay.id).invalid_at is None
    store.close()
