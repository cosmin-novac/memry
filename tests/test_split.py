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

import pytest
from conftest import FakeLLM
from starlette.testclient import TestClient

from memry.backends.local import LocalBackend
from memry.cli import format_split_report
from memry.config import Config
from memry.intelligence.split import (
    PLAN_FORMAT, SPLIT_SYSTEM, is_candidate, make_plan, plan_entries, proper_names,
    sentences, without_subject)
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


#: What each fact of the essay is about: the thesis's lab and both tags on
#: every fact, though none names them, and Tom Hale on the one naming him.
ABOUT = [["Ana Reyes", "Reyes Lab", "phd thesis", "ai research"]] * 2 + [
    ["Ana Reyes", "Tom Hale", "Reyes Lab", "phd thesis", "ai research"]]


def split_answer(*facts: str, about: list[list] | None = None) -> str:
    """The model's split: each fact with what it is about, by the names of
    the prompt's "About" list (``Splitter`` numbers them) or by number."""
    about = about or [[] for _ in facts]
    return json.dumps({"facts": [{"text": fact, "about": list(things)}
                                 for fact, things in zip(facts, about)]})


class Splitter(FakeLLM):
    """FakeLLM whose split answers name the things a fact is about; each name
    becomes its number in the prompt's "About" list, whose order follows
    the mentions. A number stays as it is."""

    def complete(self, system, user, *, json_schema=None):
        answer = super().complete(system, user, json_schema=json_schema)
        if system != SPLIT_SYSTEM:
            return answer
        listed = user.split("About:\n", 1)[1].split("\n\n", 1)[0] if "About:\n" in user else ""
        numbers = {line.split(". ", 1)[1].rsplit(" (", 1)[0]: int(line.split(".", 1)[0])
                   for line in listed.splitlines()}
        parsed = json.loads(answer)
        for fact in parsed["facts"]:
            fact["about"] = [numbers[n] if isinstance(n, str) else n for n in fact["about"]]
        return json.dumps(parsed)


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
    llm = Splitter([split_answer(*FACTS, about=ABOUT), audit()])
    store = _store(llm)
    world = _world(store)
    essay = world["essay"]

    summary = store.split_memories(user_id="ada")

    assert {k: summary[k] for k in ("in_use", "candidates", "one_fact", "split", "facts",
                                    "lossy", "failed")} == {
        "in_use": 1, "candidates": 1, "one_fact": 0, "split": 1, "facts": 3, "lossy": 0,
        "failed": 0}
    system, prompt = llm.calls[0]
    assert system == SPLIT_SYSTEM and prompt.startswith(f"Memory:\n{ESSAY}\n\nAbout:\n1. ")
    assert prompt.endswith("\n\nSplit it as JSON.")
    # the named things first, then the tags, each with its type
    listed = prompt.split("About:\n")[1].split("\n\n")[0].splitlines()
    assert sorted(line.split(". ", 1)[1] for line in listed[:3]) == [
        "Ana Reyes (person)", "Reyes Lab (person)", "Tom Hale (person)"]
    assert sorted(line.split(". ", 1)[1] for line in listed[3:]) == [
        "ai research (tag)", "phd thesis (tag)"]
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
    # each named thing on the facts the model said are about it, named or not
    assert [_linked(store, i) for i in ids] == [
        {"Ana Reyes", "Reyes Lab"}, {"Ana Reyes", "Reyes Lab"},
        {"Ana Reyes", "Tom Hale", "Reyes Lab"}]
    assert [sorted(p.entities) for p in parts] == [
        ["Ana Reyes", "Reyes Lab"], ["Ana Reyes", "Reyes Lab"],
        ["Ana Reyes", "Reyes Lab", "Tom Hale"]]
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
    llm = Splitter([split_answer(*FACTS, about=ABOUT), audit()])
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
    llm = Splitter([split_answer(*FACTS, about=ABOUT), audit()])
    store = _store(llm)
    essay = _world(store)["essay"]
    ids = store.split_memories(user_id="ada")["splits"][0]["memory_ids"]
    store.delete(ids[1])
    assert store.undo_replacement(essay.id)
    assert [e.event for e in store.history(ids[1])] == ["ADD", "DELETE"]
    assert store.history(ids[1])[-1].reason == "manual delete (invalidated)"
    store.close()


def test_a_dry_run_asks_and_writes_nothing():
    llm = Splitter([split_answer(*FACTS, about=ABOUT), audit()])
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
    # under each fact what it keeps, for the owner to read before it is made
    about = report.split(f"  3. {FACTS[2]}\n")[1].splitlines()[0]
    assert about.startswith("     about: ")
    assert sorted(about[len("     about: "):].split(", ")) == [
        "Ana Reyes (person)", "Reyes Lab (person)", "Tom Hale (person)",
        "ai research (tag)", "phd thesis (tag)"]
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
    # every entity kept (Tom Hale on a fact that drops what he said), a detail lost
    llm = Splitter([split_answer(*FACTS[:2], about=[ABOUT[0], ABOUT[2]]),
                    audit("Tom Hale pointed it out")])
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
    llm = Splitter([split_answer(FACTS[0], orphan, FACTS[2], about=ABOUT)])
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
    llm.queue(split_answer(FACTS[0], orphan, FACTS[2], about=ABOUT))
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
    llm = Splitter([split_answer(*FACTS, about=ABOUT), audit()] * 2)
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


def _fail_on(store: MemoryStore, method: str, call: int, *, kind: str | None = None) -> None:
    """Make the backend's ``method`` raise on its ``call``-th call (of events
    of ``kind`` only, for ``add_event``), as a crash midway through a split
    would."""
    original = getattr(store.backend, method)
    seen = {"calls": 0}

    def failing(*args, **kwargs):
        if kind is None or getattr(args[0], "kind", None) == kind:
            seen["calls"] += 1
            if seen["calls"] == call:
                raise RuntimeError(f"{method} failed")
        return original(*args, **kwargs)

    setattr(store.backend, method, failing)


@pytest.mark.parametrize("method, call, kind", [
    ("insert_memory", 2, None),  # the second fact
    ("add_event", 1, "split"),   # the very last write: the old memory's SUPERSEDE
])
def test_a_split_that_fails_midway_leaves_the_memory_as_it_was(method, call, kind):
    """Each fact was once embedded and saved in turn, and the old memory left
    use only after the last: a failure midway left the facts saved so far in
    use beside it, twins its undo could not find. The facts are embedded in
    one call now, and every write of one memory's split is one transaction."""
    llm = Splitter([split_answer(*FACTS, about=ABOUT), audit()])
    store = _store(llm)
    essay = _world(store)["essay"]
    before = store.export_backup(user_id="ada")
    _fail_on(store, method, call, kind=kind)

    summary = store.split_memories(user_id="ada")

    assert (summary["split"], summary["facts"], summary["failed"]) == (0, 0, 1)
    [entry] = summary["splits"]
    assert entry["not_split"] == f"the split failed, the memory is as it was: {method} failed"
    assert "memory_ids" not in entry
    assert store.export_backup(user_id="ada")["tables"] == before["tables"]
    old = store.get(essay.id)
    assert old.invalid_at is None and "split_into" not in old.metadata
    assert [m.id for m in store.get_all(user_id="ada", include_invalid=True, limit=100)
            if (m.metadata or {}).get("split_from")] == []
    assert store.undo_replacement(essay.id) is False  # nothing to undo
    found = [r.memory.id for r in store.search("Ana Reyes thesis contamination", user_id="ada")]
    assert found[0] == essay.id  # back in the vector index too
    store.close()


def test_a_split_embeds_its_facts_in_one_call():
    llm = Splitter([split_answer(*FACTS, about=ABOUT), audit()])
    store = _store(llm)
    _world(store)
    calls: list[list[str]] = []
    embed = store.embedder.embed
    store.embedder.embed = lambda texts: calls.append(list(texts)) or embed(texts)

    ids = store.split_memories(user_id="ada")["splits"][0]["memory_ids"]

    assert calls[0] == FACTS
    assert all(store.get(i).embedding_model == store.embedder.model_id for i in ids)
    store.close()


def test_the_split_prompt_keeps_tense_and_a_heading_date_once():
    """A memory headed "Decision (2026-09-12): ..." had "On 2026-09-12," put
    in front of each of its facts, where it read as the date of each, and a
    present state ("Reporting ... are included in every tier") came back in
    the past. Each fact keeps the memory's dates anyway."""
    rule = " ".join(SPLIT_SYSTEM.split())
    assert ("Keep each time and tense as the memory writes it" in rule
            and "what the memory states in the present stays in the present" in rule)
    assert ('A date that heads the whole memory, as in "Decision (2026-09-12): ...", dates '
            "the memory, not each fact: keep it once, in the first fact, and do not put it "
            "in front of the others.") in rule
    assert "A fact about an event with a date of its own keeps that date." in rule


def _plan_of(summary: dict) -> list[dict]:
    """A dry run's plan as a file would carry it: written, read and checked."""
    return plan_entries(json.loads(json.dumps(make_plan([summary]))))


def test_a_plan_makes_exactly_the_splits_reviewed_without_the_model():
    """The model answers a little differently each time, so the dry run is a
    preview only; a plan makes exactly what was read, asking nothing."""
    llm = Splitter([split_answer(*FACTS, about=ABOUT), audit()])
    store = _store(llm)
    essay = _world(store)["essay"]
    plan = _plan_of(store.split_memories(user_id="ada", dry_run=True))
    [planned] = plan
    assert {k: planned[k] for k in ("memory_id", "user", "content", "facts")} == {
        "memory_id": essay.id, "user": "ada", "content": ESSAY, "facts": FACTS}
    # each fact with the ids of the entities it keeps (version 2)
    names = {e.id: e.name for e in store.backend.entities_of_memory(essay.id, kind="any")}
    assert [sorted(names[i] for i in ids) for ids in planned["about"]] == [
        sorted(things) for things in ABOUT]
    asked = len(llm.calls)

    summary = store.split_memories(user_id="ada", plan=plan)

    assert len(llm.calls) == asked  # not one call to the model
    assert (summary["planned"], summary["split"], summary["facts"], summary["stale"],
            summary["failed"]) == (1, 1, 3, 0, 0)
    ids = summary["splits"][0]["memory_ids"]
    assert [store.get(i).content for i in ids] == FACTS
    assert [_linked(store, i) for i in ids] == [
        {"Ana Reyes", "Reyes Lab"}, {"Ana Reyes", "Reyes Lab"},
        {"Ana Reyes", "Tom Hale", "Reyes Lab"}]
    assert sorted(m.content for m in store.get_all(user_id="ada")) == sorted(FACTS)
    assert "1 split into 3 facts as planned" in format_split_report([summary])
    # a split made from a plan is undone like any other
    assert store.undo_replacement(essay.id)
    assert [m.id for m in store.get_all(user_id="ada")] == [essay.id]
    store.close()


def test_a_plan_skips_a_memory_that_changed_or_left_use_since():
    llm = Splitter([split_answer(*FACTS, about=ABOUT), audit()])
    store = _store(llm)
    essay = _world(store)["essay"]
    plan = _plan_of(store.split_memories(user_id="ada", dry_run=True))
    store.backend.update_memory(essay.id, content=ESSAY + " Tom Hale agrees.")
    before = store.export_backup(user_id="ada")

    summary = store.split_memories(user_id="ada", plan=plan)

    assert (summary["split"], summary["stale"]) == (0, 1)
    assert summary["splits"][0]["not_split"].startswith("its text changed since the plan")
    assert store.export_backup(user_id="ada")["tables"] == before["tables"]
    report = format_split_report([summary])
    assert "1 skipped because the memory left use or changed since the plan" in report

    store.delete(essay.id)
    summary = store.split_memories(user_id="ada", plan=plan)
    assert summary["splits"][0]["not_split"] == "not in use in this namespace"
    store.close()


def test_a_plan_reaches_no_memory_of_another_namespace():
    """A planned memory is looked up among the namespace's memories in use:
    a plan put to another namespace splits nothing there."""
    llm = Splitter([split_answer(*FACTS, about=ABOUT), audit()])
    store = _store(llm)
    essay = _world(store)["essay"]
    plan = _plan_of(store.split_memories(user_id="ada", dry_run=True))
    store.backend.insert_memory(Memory(content="Bo likes tea. Bo dislikes coffee.",
                                       user_id="bo"))

    for user in ("bo", None):
        summary = store.split_memories(user_id=user, exact_user=True, plan=plan)
        assert (summary["split"], summary["stale"]) == (0, 1)
    assert store.get(essay.id).invalid_at is None
    store.close()


def test_a_plan_follows_a_merge_and_skips_an_entity_that_is_gone():
    """A plan names entities by id: one merged since is its survivor, which
    the memory's mention went to; one deleted since makes the plan's split
    stale."""
    llm = Splitter([split_answer(*FACTS, about=ABOUT), audit()])
    store = _store(llm)
    world = _world(store)
    plan = _plan_of(store.split_memories(user_id="ada", dry_run=True))
    labs = store.backend.insert_entity(Entity(name="Reyes Laboratory",
                                              normalized="reyes laboratory",
                                              entity_type="organization", user_id="ada"))
    store.backend.merge_entities(labs.id, world["lab"].id)

    summary = store.split_memories(user_id="ada", plan=plan, dry_run=True)

    assert (summary["split"], summary["stale"]) == (1, 0)
    assert all(labs.id in ids and world["lab"].id not in ids
               for ids in summary["splits"][0]["about"])

    store.backend.delete_entity(world["tom"].id)
    summary = store.split_memories(user_id="ada", plan=plan)
    assert (summary["split"], summary["stale"]) == (0, 1)
    assert summary["splits"][0]["not_split"] == "its entities changed since the plan"
    assert store.get(world["essay"].id).invalid_at is None
    store.close()


def test_a_plan_is_checked_whole_before_any_split():
    good = {"format": PLAN_FORMAT, "version": 2, "splits": [
        {"memory_id": "m1", "user": None, "content": "A. B.",
         "facts": [{"text": "A.", "about": ["e1"]}, {"text": "B.", "about": []}]}]}
    assert plan_entries(good)[0]["facts"] == ["A.", "B."]
    assert plan_entries(good)[0]["about"] == [["e1"], []]
    version_1 = {"format": PLAN_FORMAT, "version": 1, "splits": [
        {"memory_id": "m1", "user": None, "content": "A. B.", "facts": ["A.", "B."]}]}
    with pytest.raises(ValueError, match="version 1; this Memry reads version 2"):
        plan_entries(version_1)  # its facts keep no entities: made again, not guessed
    for bad in ({"format": "memry-backup", "version": 2, "splits": []},
                {**good, "splits": [{**good["splits"][0], "facts": ["A.", "B."]}]},
                {**good, "splits": [{**good["splits"][0],
                                     "facts": [{"text": "A. B.", "about": []}]}]},
                {**good, "splits": [{**good["splits"][0],
                                     "facts": [{"text": "A."}, {"text": "B.", "about": []}]}]},
                {**good, "splits": [*good["splits"], {"memory_id": 7, "facts": []}]}):
        with pytest.raises(ValueError):
            plan_entries(bad)


def test_a_walk_over_namespaces_asks_about_each_memory_once():
    """No user means every user's memories: the pass for the memories
    without a user once took in every namespace, and each named one was
    asked about again (1446 memories, then the 1440 of "default")."""
    llm = SentenceSplitter()
    store = _store(llm)
    for user in (None, "a", "b"):
        store.backend.insert_memory(Memory(
            content=f"Tom's project {user} uses Go. Tom's project {user} runs on one VPS.",
            user_id=user, created_at=SAID, updated_at=SAID))

    summaries = [store.split_memories(user_id=user, dry_run=True, exact_user=True)
                 for user in store.backend.distinct_user_ids()]

    assert sorted((s["user"] or "", s["in_use"], s["split"]) for s in summaries) == [
        ("", 1, 1), ("a", 1, 1), ("b", 1, 1)]
    assert len(llm.calls) == 6  # a split and an audit for each memory, once
    store.close()


HERDS = "Elephants live in herds led by the oldest female."
ETOSHA = "Etosha National Park holds about 2,500 elephants."
TALLEST = "The tallest bulls in Etosha National Park stand 4 m at the shoulder."


def _elephants(store: MemoryStore) -> dict:
    """A memory about elephants and the park they live in, linked to both."""
    elephant = store.backend.insert_entity(Entity(
        name="Elephant", normalized="elephant", entity_type="animal", user_id="ada"))
    park = store.backend.insert_entity(Entity(
        name="Etosha National Park", normalized="etosha national park",
        entity_type="place", user_id="ada"))
    memory = store.backend.insert_memory(Memory(
        content=f"{HERDS} {ETOSHA} {TALLEST}", user_id="ada", categories=["wildlife"],
        created_at=SAID, updated_at=SAID))
    for entity in (elephant, park):
        store.backend.add_mention(EntityMention(entity_id=entity.id, memory_id=memory.id,
                                                surface=entity.name))
    return {"memory": memory, "elephant": elephant, "park": park}


def test_a_fact_keeps_an_entity_it_does_not_name():
    """"The tallest bulls ... stand 4 m" names the park, not Elephant: given
    the entities the facts that name them, it lost Elephant, and the linked
    search found it for no question about elephants. The model says what
    each fact is about, in the split's one call."""
    llm = Splitter([split_answer(
        HERDS, ETOSHA, TALLEST,
        about=[["Elephant", "wildlife"], ["Elephant", "Etosha National Park", "wildlife"],
               ["Elephant", "Etosha National Park", "wildlife"]]), audit()])
    store = _store(llm)
    world = _elephants(store)

    summary = store.split_memories(user_id="ada")

    assert len(llm.calls) == 2  # the split and the audit, nothing more
    ids = summary["splits"][0]["memory_ids"]
    assert [_linked(store, i) for i in ids] == [
        {"Elephant"}, {"Elephant", "Etosha National Park"},
        {"Elephant", "Etosha National Park"}]
    tallest = store.get(ids[2])
    assert tallest.content == TALLEST and "Elephant" in tallest.entities
    assert ids[2] in {m.id for m in store.get_all(user_id="ada",
                                                  entity_id=world["elephant"].id)}
    found = {r.memory.id: r for r in store.search("How tall is an Elephant?", user_id="ada")}
    assert ids[2] in found and found[ids[2]].signals.get("about") == 1.0
    store.close()


SUMMARY_FACTS = {
    "travel": "Ada booked flights to Lisbon for May.",
    "home": "The renovation of the Elm Street kitchen rose to 18,000 euros.",
    "learning": "The Python course with Coursera starts on Monday.",
    "work": "The Tern release is due on Friday.",
    "health": "The dentist appointment with Dr. Okafor moved to June 3.",
}


def test_each_fact_of_a_summary_keeps_only_its_own_tags():
    """A conversation summary of five topics was split into five facts that
    each took all five tags, and each ranked for all five. A fact takes the
    tags among the entities it keeps; the undo gives the memory all five
    back."""
    tags = list(SUMMARY_FACTS)
    llm = Splitter([split_answer(*SUMMARY_FACTS.values(), about=[[t] for t in tags]),
                    audit()])
    store = _store(llm)
    memory = store.backend.insert_memory(Memory(
        content="Weekly call: " + " ".join(SUMMARY_FACTS.values()), user_id="ada",
        categories=tags, created_at=SAID, updated_at=SAID))
    before = {e.name for e in store.backend.entities_of_memory(memory.id, kind="topic")}
    assert before == set(tags)

    summary = store.split_memories(user_id="ada")

    assert summary["split"] == 1
    ids = summary["splits"][0]["memory_ids"]
    for tag, part_id in zip(tags, ids):
        part = store.get(part_id)
        assert part.categories == [tag]
        assert [e.name for e in store.backend.entities_of_memory(part_id, kind="topic")] == [tag]
    assert store.undo_replacement(memory.id)
    back = store.get(memory.id)
    assert back.categories == tags
    assert {e.name for e in store.backend.entities_of_memory(memory.id, kind="topic")} == before
    store.close()


def test_a_split_with_a_fact_that_keeps_no_entity_is_not_made():
    """A fact linked to nothing is found by no linked search: the split waits
    for a person, and the dry run says which fact."""
    bare = "The tallest bulls stand 4 m at the shoulder."
    llm = Splitter([split_answer(HERDS, ETOSHA, bare,
                                 about=[["Elephant", "wildlife"],
                                        ["Etosha National Park"], []])])
    store = _store(llm)
    world = _elephants(store)

    summary = store.split_memories(user_id="ada")

    assert (summary["split"], summary["no_entity"], summary["lost_entity"]) == (0, 1, 0)
    assert len(llm.calls) == 1  # no audit for a split that is not made
    assert summary["splits"][0]["not_split"] == (
        f"a fact would keep none of the memory's entities: {bare}")
    report = format_split_report([summary])
    assert "1 left because a fact would keep none of the memory's entities" in report
    assert "     about: nothing" in report
    assert store.get(world["memory"].id).invalid_at is None
    store.close()


def test_a_split_that_would_lose_an_entity_is_not_made():
    """Reyes Lab, which no fact names, given to no fact: the memory's link
    to it would be lost."""
    without_lab = [[name for name in things if name != "Reyes Lab"] for things in ABOUT]
    llm = Splitter([split_answer(*FACTS, about=without_lab)])
    store = _store(llm)
    essay = _world(store)["essay"]

    summary = store.split_memories(user_id="ada")

    assert (summary["split"], summary["no_entity"], summary["lost_entity"]) == (0, 0, 1)
    assert len(llm.calls) == 1
    assert summary["splits"][0]["not_split"] == "an entity would be lost: Reyes Lab (person)"
    assert "1 left because an entity would be lost" in format_split_report([summary])
    assert store.get(essay.id).invalid_at is None
    store.close()


def test_what_a_fact_is_about_is_read_by_the_numbers_listed():
    """A number off the list, one given twice, or one that is no number is
    left out; a name the fact states is kept should the model miss it, and
    an old answer of bare strings is facts about nothing."""
    from memry.intelligence.split import fact_homes, split_facts

    elephant = Entity(name="Elephant", normalized="elephant", entity_type="animal")
    park = Entity(name="Etosha", normalized="etosha", entity_type="place")
    memory = Memory(content=f"{HERDS} {TALLEST}")
    answer = json.dumps({"facts": [{"text": HERDS, "about": [1, 1, 3, 0, -1, "2", True]},
                                   {"text": TALLEST, "about": [1]}]})
    facts = split_facts(FakeLLM([answer]), memory, [elephant, park])
    assert facts == [{"text": HERDS, "about": [elephant.id]},
                     {"text": TALLEST, "about": [elephant.id]}]
    homes = fact_homes([f["text"] for f in facts], [f["about"] for f in facts],
                       [(elephant.id, ["Elephant"]), (park.id, ["Etosha"])])
    assert homes == [[elephant.id], [elephant.id, park.id]]  # the park by its name
    old = split_facts(FakeLLM([json.dumps({"facts": [HERDS, TALLEST]})]), memory, [])
    assert old == [{"text": HERDS, "about": []}, {"text": TALLEST, "about": []}]


def test_the_split_prompt_asks_what_each_fact_is_about():
    rule = " ".join(SPLIT_SYSTEM.split())
    assert ('"About" under the memory numbers the things it is linked to. Give each fact '
            "the numbers of those it is about, as many as apply, also one it does not "
            "spell out") in rule
    assert ("Every fact is about one of them at least, and each of them stays with one "
            "fact at least.") in rule


class _NoTransactions(LocalBackend):
    """A backend that cannot keep one memory's writes together."""

    supports_transactions = False


def test_a_real_split_refuses_a_backend_without_transactions():
    """Without a transaction, a failure midway leaves facts in use beside the
    memory they came from; the dry run still runs, and its plan does not."""
    llm = Splitter([split_answer(*FACTS, about=ABOUT), audit()])
    store = MemoryStore(Config(db_path=":memory:"), backend=_NoTransactions(":memory:"),
                        llm=llm, embedder=HashEmbedder(64))
    essay = _world(store)["essay"]

    summary = store.split_memories(user_id="ada", dry_run=True)
    assert summary["split"] == 1
    asked = len(llm.calls)
    for plan in (None, _plan_of(summary)):
        with pytest.raises(ValueError, match="cannot keep one memory's writes together"):
            store.split_memories(user_id="ada", plan=plan)
    assert len(llm.calls) == asked  # refused before the model is asked
    assert store.get(essay.id).invalid_at is None
    with TestClient(create_app(store)) as client:
        refused = client.post("/api/v1/memories/split", json={"user_id": "ada"})
        assert refused.status_code == 409 and "transactions" in refused.json()["error"]
    store.close()


def test_the_mem0_adapter_refuses_the_memories_without_a_user():
    """Mem0 filters by a user it is given; asked for the memories without
    one, it would read everyone's, so the adapter refuses."""
    from memry.backends.mem0_adapter import Mem0ComparisonAdapter, _scope_kwargs

    assert Mem0ComparisonAdapter.supports_transactions is False
    assert _scope_kwargs(Scope(user_id="ada", exact_user=True)) == {"user_id": "ada"}
    assert _scope_kwargs(Scope()) == {}
    with pytest.raises(ValueError, match="cannot list the memories without a user"):
        _scope_kwargs(Scope(exact_user=True))


DECISION = ("Decision (2026-09-12): Tern ships monthly releases. Reporting and exports are "
            "included in every Tern tier.")
DECIDED = ["On 2026-09-12 it was decided that Tern ships monthly releases.",
           "Reporting and exports are included in every Tern tier."]


def _decision(store: MemoryStore, content: str = DECISION, **metadata) -> Memory:
    return store.backend.insert_memory(Memory(
        content=content, user_id="ada", metadata=metadata, created_at=LAST,
        updated_at=LAST, valid_from=LAST))


def test_a_date_heading_the_memory_dates_every_fact():
    """A decision saved days after it was made held from the day it was
    saved in every fact; the heading's date is when it holds from. The
    undo brings the memory back with its own dates."""
    llm = Splitter([split_answer(*DECIDED), audit()])
    store = _store(llm)
    memory = _decision(store)

    ids = store.split_memories(user_id="ada")["splits"][0]["memory_ids"]

    for i in ids:
        part = store.get(i)
        assert part.valid_from == "2026-09-12T00:00:00+00:00"
        assert (part.created_at, part.updated_at) == (LAST, LAST)
    assert store.undo_replacement(memory.id)
    assert store.get(memory.id).valid_from == LAST
    store.close()


def test_a_plan_dates_its_facts_by_the_heading_too():
    llm = Splitter([split_answer(*DECIDED), audit()])
    store = _store(llm)
    _decision(store)
    plan = _plan_of(store.split_memories(user_id="ada", dry_run=True))
    ids = store.split_memories(user_id="ada", plan=plan)["splits"][0]["memory_ids"]
    assert {store.get(i).valid_from for i in ids} == {"2026-09-12T00:00:00+00:00"}
    store.close()


@pytest.mark.parametrize("content, when", [
    ("Tern ships monthly releases. Reporting is included in every Tern tier.", None),
    ("Note (draft): Tern ships monthly releases. Reporting is included in every Tern tier.",
     None),
    (DECISION, {"start": "2026-09-14"}),  # an occurrence time is not overridden
])
def test_without_a_dated_heading_the_facts_keep_the_memorys_valid_from(content, when):
    from memry.intelligence.split import heading_date

    llm = Splitter([split_answer(*DECIDED), audit()])
    store = _store(llm)
    memory = _decision(store, content, **({"when": when} if when else {}))

    ids = store.split_memories(user_id="ada")["splits"][0]["memory_ids"]

    assert {store.get(i).valid_from for i in ids} == {LAST}
    assert heading_date("2026-09-12: Tern ships monthly.") == "2026-09-12T00:00:00+00:00"
    assert heading_date("Decision (2026-13-40): Tern ships monthly.") is None
    assert heading_date("Plan for 2026-09-12: Tern ships monthly.") is None
    assert store.get(memory.id).invalid_at is not None
    store.close()
