"""Facts with their evidence: a search returns, with the memories found, the
source turns they rest on that best match the query, each once, only from the
scope searched, within a token budget and in the order they were said. A
turn of a memory that was forgotten or deleted is never shown, nor one whose
every memory is out of use."""

from __future__ import annotations

import json
import sqlite3
from copy import deepcopy

import pytest

from memry.backends.local import LocalBackend
from memry.config import Config
from memry.intelligence.context import estimate_tokens, turn_line
from memry.providers.embeddings import HashEmbedder
from memry.providers.llm import LLM
from memry.store import MemoryStore


class ScriptedLLM(LLM):
    """Extraction answers from a queue of fact lists; a reconcile decision is
    always ADD and the coverage audit finds nothing missing."""

    name = "scripted"
    available = True

    def __init__(self) -> None:
        self.facts: list[list[dict]] = []

    def complete(self, system: str, user: str, *, json_schema=None) -> str:
        if system.startswith("You are the long-term memory extraction system"):
            return json.dumps({"facts": self.facts.pop(0)})
        if "decide one action" in system:
            return json.dumps({"action": "ADD", "target": None, "content": None,
                               "reason": "new"})
        if system.startswith("You audit"):
            return json.dumps({"missing": []})
        return "{}"


def _fact(content: str, *sources: int) -> dict:
    return {"content": content, "type": "episodic", "importance": 0.6, "categories": [],
            "entities": [], "relations": [], "when": None, "sources": list(sources)}


@pytest.fixture
def store():
    llm = ScriptedLLM()
    s = MemoryStore(Config(db_path=":memory:"), llm=llm, embedder=HashEmbedder(128))
    s.llm_script = llm
    yield s
    s.close()


_TRIP = [
    {"role": "Ada", "content": "We spent Saturday at the Lisbon aquarium with the kids."},
    {"role": "Bea", "content": "Did the kids like the sea otters at the aquarium?"},
    {"role": "Ada", "content": "They loved the sea otters, two of them were holding hands."},
    {"role": "Bea", "content": "I am repainting my kitchen a pale green this week."},
    {"role": "Ada", "content": "Pale green sounds calm; send me a photo of the kitchen."},
]


def _save_trip(store, *, user_id="ada", run_id=None, created_at="2026-05-02T10:00:00+00:00"):
    store.llm_script.facts.append([
        _fact("Ada visited the Lisbon aquarium with her kids on Saturday", 1, 2),
        _fact("Ada's kids loved the sea otters at the Lisbon aquarium", 2, 3),
        _fact("Bea is repainting her kitchen pale green", 4, 5),
    ])
    return store.add(_TRIP, user_id=user_id, run_id=run_id, created_at=created_at)


def _by_content(store, user_id="ada", run_id=None):
    return {m.content: m for m in store.get_all(user_id=user_id, run_id=run_id)}


def test_a_search_returns_the_best_matching_source_turns_once_each_in_the_order_said(store):
    saved = _save_trip(store)
    results = store.search("What did the kids think of the sea otters at the aquarium?",
                           user_id="ada", limit=5)
    turns = [t for r in results for t in r.evidence]
    ids = [t.episode_id for t in turns]
    assert len(ids) == len(set(ids))  # the turn two facts rest on is shown once
    # every turn is a source of the result it is attached to, which is the
    # best ranked result resting on it
    for rank, result in enumerate(results):
        for turn in result.evidence:
            assert turn.episode_id in result.memory.source_episode_ids
            assert not any(turn.episode_id in r.memory.source_episode_ids
                           for r in results[:rank])
    # the turns about the otters are shown; together, in the order they were said
    chosen = store.evidence("What did the kids think of the sea otters at the aquarium?",
                            results, user_id="ada")
    assert [t.episode_id for t in chosen] == [e for e in saved.episode_ids
                                             if e in {t.episode_id for t in chosen}]
    assert saved.episode_ids[2] in [t.episode_id for t in chosen]
    third = next(t for t in chosen if t.episode_id == saved.episode_ids[2])
    assert (third.speaker, third.said_at, third.content) == (
        "Ada", "2026-05-02T10:00:00+00:00", _TRIP[2]["content"])


def test_the_turns_stay_within_their_token_budget_best_match_first(store):
    saved = _save_trip(store)
    query = "Were two of them holding hands?"
    results = store.search(query, user_id="ada", limit=5, evidence=False)
    one = estimate_tokens(turn_line(store.evidence(query, results, user_id="ada")[0])) + 1
    # a budget for one turn: the turn that best matches the question
    [best] = store.evidence(query, results, user_id="ada", token_budget=one)
    assert best.episode_id == saved.episode_ids[2]
    everything = store.evidence(query, results, user_id="ada", token_budget=10_000)
    assert len(everything) == 5
    for budget in (0, one, 2 * one, 40):
        chosen = store.evidence(query, results, user_id="ada", token_budget=budget)
        assert sum(estimate_tokens(turn_line(t)) + 1 for t in chosen) <= budget
    store.config.retrieval.evidence_tokens = 0
    assert not any(r.evidence for r in store.search(query, user_id="ada", limit=5))


def test_turns_come_only_from_the_scope_searched(store):
    _save_trip(store, user_id="ada", run_id="r1")
    other = _save_trip(store, user_id="kai", run_id="r1")
    later = _save_trip(store, user_id="ada", run_id="r2",
                       created_at="2026-06-01T10:00:00+00:00")
    query = "sea otters at the aquarium"
    for result in store.search(query, user_id="ada", limit=10):
        assert not set(other.episode_ids) & {t.episode_id for t in result.evidence}
    # a memory of run r1 that also rests on a turn of run r2 (a save in r2
    # updated it) shows only r1's turns in a search of r1
    memory = _by_content(store, run_id="r1")[
        "Ada's kids loved the sea otters at the Lisbon aquarium"]
    store.backend.update_memory(memory.id, source_episode_ids=[
        *memory.source_episode_ids, *later.episode_ids], touch=False)
    results = store.search(query, user_id="ada", run_id="r1", limit=10)
    shown = {t.episode_id for r in results for t in r.evidence}
    assert shown and not shown & set(later.episode_ids)
    # searched in r2, the same memory shows r2's turns only
    in_r2 = store.evidence(query, results, user_id="ada", run_id="r2", token_budget=10_000)
    assert in_r2 and {t.episode_id for t in in_r2} <= set(later.episode_ids)


def test_a_forgotten_or_deleted_memory_never_shows_its_turns(store):
    saved = _save_trip(store)
    memories = _by_content(store)
    query = "the kids and the sea otters at the aquarium; the pale green kitchen"
    everything = 10_000

    def shown():
        results = store.search(query, user_id="ada", limit=10, evidence=False)
        return {t.episode_id for t in store.evidence(query, results, user_id="ada",
                                                    token_budget=everything)}

    assert shown() == set(saved.episode_ids)
    # forgotten: its turns go, also the one another fact rests on too (line 2)
    otters = memories["Ada's kids loved the sea otters at the Lisbon aquarium"]
    store.delete(otters.id)
    assert shown() == {saved.episode_ids[0], saved.episode_ids[3], saved.episode_ids[4]}
    # brought back, they are shown again
    store.unforget(otters.id)
    assert shown() == set(saved.episode_ids)
    # deleted for good: its turns are never shown again, whatever else rests on them
    kitchen = memories["Bea is repainting her kitchen pale green"]
    aquarium = memories["Ada visited the Lisbon aquarium with her kids on Saturday"]
    store.backend.update_memory(aquarium.id, source_episode_ids=[
        *aquarium.source_episode_ids, saved.episode_ids[4]], touch=False)
    store.delete(kitchen.id, hard=True)
    assert shown() == set(saved.episode_ids[:3])
    withheld = store.backend.episodes_by_id(saved.episode_ids)
    assert [bool(withheld[e].withheld_at) for e in saved.episode_ids] == [
        False, False, False, True, True]


def test_a_turn_whose_every_memory_is_out_of_use_is_not_shown(store):
    saved = _save_trip(store)
    kitchen = _by_content(store)["Bea is repainting her kitchen pale green"]
    replacement = store.add("Bea painted her kitchen blue in the end.", user_id="ada",
                            infer=False).actions[0].memory_id
    store.backend.invalidate_memory(kitchen.id, superseded_by=replacement)
    results = store.search("Bea kitchen pale green", user_id="ada", limit=10,
                           include_invalid=True, evidence=False)
    assert kitchen.id in {r.memory.id for r in results}
    shown = {t.episode_id for t in store.evidence("Bea kitchen pale green", results,
                                                  user_id="ada", token_budget=10_000)}
    assert not shown & set(saved.episode_ids[3:])


def test_a_turn_that_says_no_more_than_its_memory_is_not_repeated(store):
    store.add("Ada's locker code hint is the year she moved.", user_id="ada", infer=False)
    [result] = store.search("locker code hint", user_id="ada", limit=5)
    assert result.evidence == []


def test_the_context_shows_the_facts_then_their_turns_within_its_budget(store):
    _save_trip(store)
    context = store.reconstruct_context("What did the kids think of the sea otters?",
                                        user_id="ada", token_budget=400)
    facts, turns = context.text.split("\n\nWhat was said, in the order it was said:\n")
    assert "- Ada's kids loved the sea otters at the Lisbon aquarium (said 2 May 2026)" in facts
    assert "- 2 May 2026: Ada: They loved the sea otters, two of them were holding hands." in (
        turns)
    assert context.episode_ids and context.token_estimate <= 400
    # every turn shown rests under a fact shown
    shown = [store.get(m) for m in context.memory_ids]
    assert set(context.episode_ids) <= {e for m in shown for e in m.source_episode_ids}


def test_episodes_are_embedded_at_save_and_a_deferred_save_at_its_distillation(store):
    saved = _save_trip(store)
    model = store.embedder.model_id
    assert set(store.backend.episode_vectors_of(saved.episode_ids, model)) == set(
        saved.episode_ids)
    deferred = store.add_deferred("We saw the otters again on Sunday.", user_id="ada")
    assert store.backend.episode_vectors_of(deferred.episode_ids, model) == {}
    store.llm_script.facts.append([_fact("Ada saw the otters again on Sunday", 1)])
    store._distill_pending_group([deferred.actions[0].memory_id])
    assert set(store.backend.episode_vectors_of(deferred.episode_ids, model)) == set(
        deferred.episode_ids)
    # the full-text index has every episode
    assert set(store.backend.episode_keyword_scores("otters", [
        *saved.episode_ids, *deferred.episode_ids])) == {
        saved.episode_ids[1], saved.episode_ids[2], *deferred.episode_ids}


def test_reindex_embeds_the_episodes_an_older_store_has_no_vectors_for(store):
    saved = _save_trip(store)
    store.backend._db.execute("UPDATE episodes SET embedding = NULL, embedding_model = NULL")
    store.reindex()
    assert set(store.backend.episode_vectors_of(saved.episode_ids,
                                                store.embedder.model_id)) == set(
        saved.episode_ids)


def test_an_episode_table_from_before_evidence_gains_its_columns_and_index(tmp_path):
    path = tmp_path / "old.db"
    LocalBackend(str(path)).close()
    db = sqlite3.connect(path)
    for statement in ("DROP TRIGGER episodes_ai", "DROP TRIGGER episodes_ad",
                      "DROP TABLE episodes_fts", "ALTER TABLE episodes DROP COLUMN withheld_at",
                      "ALTER TABLE episodes DROP COLUMN embedding",
                      "ALTER TABLE episodes DROP COLUMN embedding_model"):
        db.execute(statement)
    db.execute("INSERT INTO episodes (id, content, role, metadata, created_at) "
               "VALUES ('e1', 'The otters held hands.', 'Ada', '{}', '2026-05-02')")
    db.commit()
    db.close()
    backend = LocalBackend(str(path))
    try:
        assert backend.episodes_by_id(["e1"])["e1"].withheld_at is None
        assert set(backend.episode_keyword_scores("otters", ["e1"])) == {"e1"}
    finally:
        backend.close()


def test_the_mcp_search_rows_carry_both_dates_and_the_evidence(store):
    from test_servers import call_tool

    from memry.mcp_server import create_server

    _save_trip(store)
    kitchen = _by_content(store)["Bea is repainting her kitchen pale green"]
    store.backend.update_memory(kitchen.id, metadata={
        **kitchen.metadata, "when": {"start": "2026-05-04"}}, touch=False)
    rows = call_tool(create_server(store, manage_enrichment_worker=False), "search_memories",
                     {"query": "sea otters holding hands; the kitchen green", "user_id": "ada"})
    assert {row["said"] for row in rows} == {"2026-05-02"}
    assert {row.get("happened") for row in rows} == {None, "happened 2026-05-04"}
    turns = [turn for row in rows for turn in row.get("evidence", [])]
    assert {"said": "2026-05-02", "speaker": "Ada", "text": _TRIP[2]["content"]} in turns


def test_a_backup_keeps_the_episodes_and_one_from_before_their_new_columns_restores(store):
    _save_trip(store)
    backup = store.export_backup(user_id="ada")
    older = deepcopy(backup)
    for row in older["tables"]["episodes"]:
        for column in ("withheld_at", "embedding", "embedding_model"):
            del row[column]
    target = MemoryStore(Config(db_path=":memory:"), llm=ScriptedLLM(),
                         embedder=HashEmbedder(128))
    try:
        assert target.import_backup(older, owner_prefix="ada")["inserted"] > 0
        results = target.search("sea otters", user_id="ada", limit=5, evidence=False)
        # restored without vectors, the turns are still chosen, by their words
        assert target.evidence("sea otters", results, user_id="ada")
    finally:
        target.close()
