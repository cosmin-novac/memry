from __future__ import annotations

import json

import pytest
from conftest import decision, fact, facts_response

from memry.models import Scope


def coverage(*missing: str) -> str:
    return json.dumps({"missing": list(missing)})


def test_add_infer_false_stores_verbatim(verbatim_store):
    result = verbatim_store.add(
        "Ada prefers dark mode", user_id="ada", infer=False, importance=0.9
    )
    assert result.summary() == {"ADD": 1}
    memories = verbatim_store.get_all(user_id="ada")
    assert memories[0].content == "Ada prefers dark mode"
    assert memories[0].importance == 0.9
    # provenance: episode recorded and linked
    assert memories[0].source_episode_ids == result.episode_ids
    episodes = verbatim_store.episodes(user_id="ada")
    assert episodes[0].content == "Ada prefers dark mode"


def test_add_without_llm_falls_back_to_verbatim(verbatim_store):
    result = verbatim_store.add(
        [
            {"role": "user", "content": "I live in Berlin"},
            {"role": "user", "content": "I like espresso"},
        ],
        user_id="ada",
    )
    assert result.summary()["ADD"] == 2
    # verbatim-because-no-LLM memories are flagged for later distillation
    assert all(
        m.metadata.get("pending_distillation") for m in verbatim_store.get_all(user_id="ada")
    )


def test_llm_failure_degrades_to_verbatim_with_warning(store, fake_llm):
    # FakeLLM with an empty queue raises on complete(), simulating a provider
    # outage (e.g. exhausted API credits) mid-save.
    result = store.add("I live in Berlin", user_id="ada")
    assert result.summary() == {"ADD": 1}
    assert result.warnings and "stored verbatim" in result.warnings[0]
    memory = store.get_all(user_id="ada")[0]
    assert memory.content == "I live in Berlin"
    assert memory.metadata.get("pending_distillation") is True


def test_distill_replaces_verbatim_memory(store, fake_llm):
    result = store.add("I live in Berlin and use uv", user_id="ada")  # LLM fails
    assert result.warnings
    original = store.get_all(user_id="ada")[0]

    fake_llm.queue(
        facts_response(
            fact("User lives in Berlin", categories=["location"]),
            fact("User prefers uv over pip", type="procedural"),
        ),
        # 2nd fact sees the 1st as similar -> reconcile call (original excluded)
        decision("ADD", reason="unrelated preference"),
    )
    distilled = store.distill(original.id)
    assert distilled.summary() == {"ADD": 2}

    active = store.get_all(user_id="ada")
    assert {m.content for m in active} == {
        "User lives in Berlin",
        "User prefers uv over pip",
    }
    assert not any(m.metadata.get("pending_distillation") for m in active)
    # original invalidated with audit trail, superseded by a distilled fact
    gone = store.get(original.id)
    assert gone.invalid_at is not None
    assert gone.superseded_by in {m.id for m in active}
    assert any(
        e.event == "SUPERSEDE" and "distilled" in (e.reason or "")
        for e in store.history(original.id)
    )
    # provenance carried over
    assert all(m.source_episode_ids == original.source_episode_ids for m in active)


def test_distill_nothing_extracted_keeps_memory(store, fake_llm):
    store.add("hmm ok", user_id="ada")  # LLM fails -> pending verbatim
    original = store.get_all(user_id="ada")[0]
    fake_llm.queue(facts_response())  # extraction finds nothing
    result = store.distill(original.id)
    assert result.warnings and "kept verbatim" in result.warnings[0]
    kept = store.get(original.id)
    assert kept.invalid_at is None
    assert "pending_distillation" not in kept.metadata


def test_batch_facts_stay_discrete(store, fake_llm):
    """Facts extracted from ONE payload must not chain-merge into a single
    memory (the ADD followed by N UPDATEs failure): same-call memories are
    excluded from each other's similarity sets, so no reconcile LLM call."""
    fake_llm.queue(
        facts_response(
            fact("The zeitnachweis skill fills timesheets"),
            fact("The zeitnachweis skill must edit XML directly rather than openpyxl"),
            fact("openpyxl corrupts the timesheet template, hence direct XML"),
        ),
        coverage(),  # audit pass: nothing missing
    )
    result = store.add(
        "zeitnachweis fills timesheets; must edit XML directly, not openpyxl, "
        "because openpyxl corrupts the template",
        user_id="u",
    )
    assert result.summary() == {"ADD": 3}
    assert not result.warnings
    assert len(store.get_all(user_id="u")) == 3
    # extraction + coverage only; no reconcile calls between batch siblings
    assert len(fake_llm.calls) == 2


def test_coverage_audit_reports_dropped_details(store, fake_llm):
    fake_llm.queue(
        facts_response(fact("The skill fills timesheets")),
        coverage("must edit XML directly rather than openpyxl"),
    )
    result = store.add(
        "The skill fills timesheets and must edit XML directly rather than "
        "openpyxl because openpyxl corrupts the template.",
        user_id="u",
    )
    assert result.summary() == {"ADD": 1}
    assert result.warnings and "XML" in result.warnings[0]


def test_import_verbatim_bulk(verbatim_store):
    result = verbatim_store.import_verbatim(
        [
            {"content": "Ada lives in Berlin", "categories": ["location"], "importance": 0.9},
            {"content": "Ada prefers espresso", "user_id": "ada", "categories": "diet, coffee"},
            {"content": "   "},  # empty -> skipped
            {"content": "typed", "memory_type": "bogus-type"},  # falls back to semantic
        ],
        user_id="fallback",
    )
    assert result["imported"] == 3
    assert result["skipped"] == 1
    assert len(result["memory_ids"]) == 3

    everyone = verbatim_store.get_all()
    by_content = {m.content: m for m in everyone}
    assert by_content["Ada lives in Berlin"].user_id == "fallback"
    assert by_content["Ada lives in Berlin"].importance == 0.9
    assert by_content["Ada prefers espresso"].user_id == "ada"
    assert by_content["Ada prefers espresso"].categories == ["diet", "coffee"]
    assert by_content["typed"].memory_type == "semantic"
    # no LLM, no reconciliation, but full provenance + audit trail
    for m in everyone:
        assert m.source_episode_ids
        assert [e.event for e in verbatim_store.history(m.id)] == ["ADD"]
    # embeddings arrive in one batch (hash embedder: all rows embedded)
    assert all(m.embedding_model for m in everyone)


def test_import_verbatim_never_calls_llm(store, fake_llm):
    # FakeLLM raises on any call; a verbatim import must not touch it.
    result = store.import_verbatim([{"content": "a"}, {"content": "b"}])
    assert result["imported"] == 2
    assert fake_llm.calls == []


def test_distill_requires_llm_and_valid_target(verbatim_store, store, fake_llm):
    import pytest

    verbatim_store.add("note", user_id="ada", infer=False)
    memory = verbatim_store.get_all(user_id="ada")[0]
    with pytest.raises(ValueError):
        verbatim_store.distill(memory.id)
    assert store.distill("no-such-id") is None


def test_add_with_extraction_and_reconcile_add(store, fake_llm):
    fake_llm.queue(
        facts_response(
            fact("User lives in Berlin", categories=["location"]),
            fact("User prefers uv over pip", type="procedural"),
        ),
        # the 2nd fact sees the 1st as a (weakly) similar memory -> reconcile call
        decision("ADD", reason="unrelated preference"),
    )
    result = store.add("I live in Berlin and use uv", user_id="ada")
    assert result.summary() == {"ADD": 2}
    contents = {m.content for m in store.get_all(user_id="ada")}
    assert "User lives in Berlin" in contents


def test_exact_duplicate_skipped_without_llm_call(store, fake_llm):
    fake_llm.queue(facts_response(fact("User lives in Berlin")), coverage())
    store.add("I live in Berlin", user_id="ada")

    fake_llm.queue(facts_response(fact("User lives in Berlin")), coverage())
    result = store.add("I live in Berlin", user_id="ada")
    assert result.summary() == {"NONE": 1}
    assert len(store.get_all(user_id="ada")) == 1
    # extraction + coverage audit only; the exact-duplicate fast path never
    # needs a reconcile decision
    assert not any("decide one action" in system for system, _ in fake_llm.calls)


def test_contradiction_supersedes_old_memory(store, fake_llm):
    fake_llm.queue(facts_response(fact("User lives in Munich")))
    store.add("I live in Munich", user_id="ada")
    old = store.get_all(user_id="ada")[0]

    fake_llm.queue(
        facts_response(fact("User lives in Amsterdam")),
        decision("WRONG", target=0, reason="never lived in Munich"),
    )
    result = store.add("Correction: I live in Amsterdam, never did in Munich", user_id="ada")
    assert result.actions[0].event == "DELETE"

    active = store.get_all(user_id="ada")
    assert [m.content for m in active] == ["User lives in Amsterdam"]

    archived = store.get(old.id)
    assert archived.invalid_at is not None
    assert archived.superseded_by == active[0].id
    assert [e.kind for e in store.history(old.id) if e.event == "SUPERSEDE"] == ["contradiction"]
    # never true: out of search
    found = store.search("Where does the user live?", user_id="ada", limit=5)
    assert [r.memory.id for r in found] == [active[0].id]


def test_more_merges_into_a_new_memory_that_supersedes_the_old_one(store, fake_llm):
    fake_llm.queue(facts_response(fact("User works at Northwind")))
    store.add("I work at Northwind", user_id="ada")
    target = store.get_all(user_id="ada")[0]

    fake_llm.queue(
        facts_response(fact("User works at Northwind as a data engineer")),
        decision(
            "MORE", target=0, content="User works at Northwind as a data engineer"
        ),
        facts_response(fact("User works at Northwind as a data engineer")),
    )
    result = store.add("I'm a data engineer there", user_id="ada")
    action = result.actions[0]
    assert action.event == "UPDATE" and action.memory_id != target.id
    assert [m.content for m in store.get_all(user_id="ada")] == [
        "User works at Northwind as a data engineer"]
    old = store.get(target.id)
    assert (old.content, old.superseded_by) == ("User works at Northwind", action.memory_id)
    assert [e.event for e in store.history(target.id)] == ["ADD", "SUPERSEDE"]
    assert [e.kind for e in store.history(target.id) if e.event == "SUPERSEDE"] == ["update"]


def test_search_scoping_isolated(verbatim_store):
    verbatim_store.add("likes coffee", user_id="ada", infer=False)
    verbatim_store.add("likes matcha", user_id="bob", infer=False)
    hits = verbatim_store.search("likes", user_id="ada", limit=10)
    assert [h.memory.content for h in hits] == ["likes coffee"]


def test_search_signals_present(verbatim_store):
    verbatim_store.add("Ada prefers TypeScript", user_id="ada", infer=False)
    hits = verbatim_store.search("typescript", user_id="ada")
    assert hits
    signals = hits[0].signals
    assert "fused" in signals and "recency" in signals and "importance" in signals


def test_manual_update_delete_history(verbatim_store):
    verbatim_store.add("temp fact", user_id="ada", infer=False)
    memory = verbatim_store.get_all(user_id="ada")[0]

    verbatim_store.update(memory.id, content="edited fact")
    assert verbatim_store.get(memory.id).content == "edited fact"

    assert verbatim_store.delete(memory.id)  # soft
    assert verbatim_store.get_all(user_id="ada") == []
    assert verbatim_store.get(memory.id) is not None  # still in DB
    events = [e.event for e in verbatim_store.history(memory.id)]
    assert events == ["ADD", "UPDATE", "DELETE"]

    assert verbatim_store.delete(memory.id, hard=True)
    assert verbatim_store.get(memory.id) is None


def test_delete_all_and_reset(verbatim_store):
    verbatim_store.add("a", user_id="ada", infer=False)
    verbatim_store.add("b", user_id="ada", infer=False)
    assert verbatim_store.delete_all(user_id="ada") == 2
    assert verbatim_store.get_all(user_id="ada") == []
    verbatim_store.reset()
    assert verbatim_store.stats()["episodes"] == 0


def test_a_reset_forgets_each_namespaces_upkeep_state_and_keeps_the_stores_settings(
        verbatim_store):
    """Queues, the owner's name and entity, and when each pass last ran belong
    to the memories a reset deletes; the pause switch, a pass turned off and
    the schema's migration markers belong to the store."""
    store = verbatim_store
    store.add("Ada lives in Leeds", user_id="ada", infer=False)
    store.set_owner_name("ada", "Ada Quint")
    store._upkeep_set("consolidation:pending", "ada", [{"ids": ["m1", "m2"]}])
    assert store.run_upkeep_cycle(user_id="ada")  # stamps when the passes ran
    assert store.last_pass_run("dedup_entities", "ada") is not None
    store.set_maintenance_enabled("consolidation", False)
    store.set_upkeep_paused(True)
    markers = {key: store.backend.get_meta(key) for key in (
        "schema:topics:v1", "schema:tag-entities:v1", "schema:tag-survivors:v1")}
    store.reset()
    assert store.owner_name("ada") == "the user"
    assert store._upkeep_get("consolidation:pending", "ada", None) is None
    assert store.last_pass_run("dedup_entities", "ada") is None
    assert store.backend.get_meta("entity_dedup:v2:last_run:ada") is None
    assert store.upkeep_paused() and not store.maintenance_enabled("consolidation")
    assert all(markers.values())
    assert {key: store.backend.get_meta(key) for key in markers} == markers


def test_reconstruct_context(verbatim_store):
    verbatim_store.add("Ada lives in Berlin", user_id="ada", infer=False)
    verbatim_store.add("Ada prefers dark mode", user_id="ada", infer=False)
    ctx = verbatim_store.reconstruct_context("where does ada live", user_id="ada")
    assert "Berlin" in ctx.text
    assert ctx.memory_ids


def test_decay_sweep_forgets_stale(verbatim_store):
    from datetime import datetime, timedelta, timezone

    verbatim_store.add("old trivial detail", user_id="ada", infer=False, importance=0.2)
    memory = verbatim_store.get_all(user_id="ada")[0]
    old_ts = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat(timespec="seconds")
    verbatim_store.backend._db.execute(
        "UPDATE memories SET updated_at = ? WHERE id = ?", (old_ts, memory.id)
    )
    forgotten = verbatim_store.decay_sweep(threshold=0.1)
    assert memory.id in forgotten
    assert verbatim_store.get_all(user_id="ada") == []


def test_reindex(verbatim_store):
    verbatim_store.add("some fact", user_id="ada", infer=False)
    count = verbatim_store.reindex()
    assert count == 1


def test_malformed_reconcile_decision_falls_back_to_add(store, fake_llm):
    fake_llm.queue(facts_response(fact("User has a dog")))
    store.add("I have a dog", user_id="ada")

    fake_llm.queue(
        facts_response(fact("User has a dog named Rex")),
        "this is not json at all",
    )
    result = store.add("The dog is called Rex", user_id="ada")
    assert result.actions[0].event == "ADD"
    assert len(store.get_all(user_id="ada")) == 2


def test_stats_counts_forgotten_memories_without_listing_them(verbatim_store):
    """The Forgotten tab count must match the Forgotten tab: deleted memories
    count, replaced ones are history. It is counted, not built as a list."""
    from memry.models import Memory

    backend = verbatim_store.backend
    kept = backend.insert_memory(Memory(content="Ada lives in Amsterdam", user_id="ada"))
    gone = backend.insert_memory(Memory(content="Ada lives in Utrecht", user_id="ada"))
    verbatim_store.delete(gone.id)
    old = backend.insert_memory(Memory(content="Ada works at Northwind", user_id="ada"))
    backend.invalidate_memory(old.id, superseded_by=kept.id)

    stats = verbatim_store.stats()
    assert stats["forgotten_memories"] == len(verbatim_store.forgotten(user_id="ada")) == 1
    assert stats["invalidated_memories"] == 2


@pytest.mark.parametrize("how", ["hard delete", "purge"])
def test_what_a_memory_replaced_is_forgotten_once_it_is_deleted_for_good(verbatim_store, how):
    """Two copies of a fact were consolidated into a third memory, which is
    then deleted for good. Before, the copies kept pointing at a memory that
    no longer existed and showed in no list: Forgotten skipped them as
    replaced, Replaced lists contradictions and updates only. Now nothing
    points at it, and they are under Forgotten with why, to be brought back."""
    from memry.models import Memory

    store, backend = verbatim_store, verbatim_store.backend
    embedder = store.embedder
    copies = [backend.insert_memory(
        Memory(content="Kestrel eats salmon kibble", user_id="ada",
               embedding_model=embedder.model_id),
        embedder.embed(["Kestrel eats salmon kibble"])[0]) for _ in range(2)]
    [group] = store.consolidate_memories(user_id="ada")["groups"]
    survivor = group["survivor"]
    assert {backend.get_memory(c.id).superseded_by for c in copies} == {survivor}
    if how == "purge":
        assert store.delete(survivor)
        assert store.purge(survivor)
    else:
        assert store.delete(survivor, hard=True)

    assert all(backend.get_memory(c.id).superseded_by is None for c in copies)
    rows = store.forgotten(user_id="ada")
    assert sorted(row["memory"].id for row in rows) == sorted(c.id for c in copies)
    assert {(row["actor"], row["trigger"]) for row in rows} == {(
        "system", f"The memory that had replaced it ({survivor}) was deleted for good.")}
    assert store.stats()["forgotten_memories"] == 2
    assert store.unforget(copies[0].id)
    assert [m.id for m in store.get_all(user_id="ada")] == [copies[0].id]


# ------------------------------------------------ one person's saves, any run
def test_a_duplicate_saved_in_another_run_is_recorded_on_the_memory(verbatim_store):
    """An exact duplicate of another run's memory is the fact said again: no
    second copy. The save is recorded on the memory as evidence (its episode
    joins the sources, a NONE event says when), and that episode is of the
    save's run, so a search of that run finds the memory."""
    first = verbatim_store.add("Ada likes green tea", user_id="ada", run_id="r1", infer=False)
    again = verbatim_store.add("Ada likes green tea", user_id="ada", run_id="r2", infer=False)
    memory_id = first.actions[0].memory_id
    assert (again.actions[0].event, again.actions[0].memory_id) == ("NONE", memory_id)
    memory = verbatim_store.get(memory_id)
    assert memory.run_id == "r1"
    assert memory.source_episode_ids == first.episode_ids + again.episode_ids
    assert [e.event for e in verbatim_store.history(memory_id)] == ["ADD", "NONE"]
    for run in ("r1", "r2", None):
        found = verbatim_store.search("green tea", user_id="ada", run_id=run, limit=5)
        assert [r.memory.id for r in found] == [memory_id]
    assert verbatim_store.search("green tea", user_id="ada", run_id="r3", limit=5) == []
    # listing a run lists the memories it holds as its own
    assert verbatim_store.get_all(user_id="ada", run_id="r2") == []
    assert len(verbatim_store.get_all(user_id="ada")) == 1


def test_a_change_from_another_run_supersedes_and_lands_in_the_saves_run(store, fake_llm):
    """Reconcile reads the user's memories across runs and acts on them alike:
    "I moved to Amsterdam" saved under s2 changes "I live in Munich" of s1,
    which ends at the save and stays as history, after the current value."""
    fake_llm.queue(facts_response(fact("User lives in Munich")), coverage())
    munich = store.add("I live in Munich", user_id="ada", run_id="s1").actions[0]
    fake_llm.queue(facts_response(fact("User lives in Amsterdam")),
                   decision("CHANGED", target=0, reason="moved cities"), coverage())
    moved = store.add("I moved to Amsterdam", user_id="ada", run_id="s2").actions[0]
    assert moved.event == "SUPERSEDE"
    retired = store.get(munich.memory_id)
    assert retired.invalid_at is not None and retired.superseded_by == moved.memory_id
    assert store.get(moved.memory_id).run_id == "s2"
    found = store.search("Where does the user live?", user_id="ada", limit=5)
    assert [r.memory.content for r in found] == ["User lives in Amsterdam", "User lives in Munich"]
    in_run = store.search("Where does the user live?", user_id="ada", run_id="s2", limit=5)
    assert [r.memory.id for r in in_run] == [moved.memory_id]
    assert fake_llm.responses == []


@pytest.mark.parametrize("run", ["s1", "s2"])
def test_another_runs_memory_saying_it_already_is_recorded_on_it(store, fake_llm, run):
    """A SAME of a memory of any run records the save on that memory and adds
    nothing; a search of the save's run finds the memory."""
    fake_llm.queue(facts_response(fact("User likes green tea")), coverage())
    first = store.add("I like green tea", user_id="ada", run_id="s1").actions[0]
    fake_llm.queue(facts_response(fact("User enjoys green tea")),
                   decision("SAME", target=0), coverage())
    again = store.add("I enjoy green tea", user_id="ada", run_id=run).actions[0]
    assert (again.event, again.memory_id) == ("NONE", first.memory_id)
    assert [m.id for m in store.get_all(user_id="ada")] == [first.memory_id]
    assert [e.event for e in store.history(first.memory_id)] == ["ADD", "NONE"]
    found = store.search("green tea", user_id="ada", run_id=run, limit=5)
    assert [r.memory.id for r in found] == [first.memory_id]
    assert fake_llm.responses == []


@pytest.mark.parametrize("run", ["s1", "s2"])
def test_another_runs_memory_a_detail_adds_to_is_merged(store, fake_llm, run):
    """A MORE of a memory of any run writes the merged text as a new memory of
    the save's run, superseding the old one; its sources keep it in the old
    run's search too."""
    fake_llm.queue(facts_response(fact("User likes green tea")), coverage())
    first = store.add("I like green tea", user_id="ada", run_id="s1").actions[0]
    merged = "User likes green tea, brewed at 80 degrees"
    fake_llm.queue(facts_response(fact("User brews green tea at 80 degrees")),
                   decision("MORE", target=0, content=merged), facts_response(), coverage())
    result = store.add("I brew it at 80 degrees", user_id="ada", run_id=run).actions[0]
    assert result.event == "UPDATE" and result.memory_id != first.memory_id
    new = store.get(result.memory_id)
    assert (new.content, new.run_id) == (merged, run)
    assert store.get(first.memory_id).superseded_by == result.memory_id
    assert [m.id for m in store.get_all(user_id="ada")] == [result.memory_id]
    for searched in ("s1", run):
        found = store.search("green tea", user_id="ada", run_id=searched, limit=5)
        assert found[0].memory.id == result.memory_id
    assert fake_llm.responses == []


def test_an_exact_duplicate_of_another_runs_memory_is_recorded_without_asking(store, fake_llm):
    """An exact duplicate (normalized text) of another run's memory is SAME by
    rule: no model is asked (nothing is queued for a reconcile answer), and
    the save is recorded on that memory."""
    fake_llm.queue(facts_response(fact("User likes green tea")), coverage())
    first = store.add("I like green tea", user_id="ada", run_id="s1").actions[0]
    fake_llm.queue(facts_response(fact("user likes  green tea.")), coverage())
    again = store.add("I like green tea", user_id="ada", run_id="s2").actions[0]
    assert (again.event, again.memory_id) == ("NONE", first.memory_id)
    assert again.reason == "exact duplicate"
    assert store.get(first.memory_id).invalid_at is None
    assert [e.event for e in store.history(first.memory_id)] == ["ADD", "NONE"]
    assert fake_llm.responses == []


def test_reconcile_never_asks_the_decider_about_another_runs_exact_duplicate():
    from conftest import FakeLLM

    from memry.backends.local import LocalBackend
    from memry.intelligence.reconcile import reconcile_candidate
    from memry.models import CandidateFact, Memory, SearchResult
    from memry.providers.decisions import NoneDecider
    from memry.providers.embeddings import HashEmbedder

    class Recording(NoneDecider):
        name = "stub"
        available = True
        asked: list = []

        def decide(self, state, questions):
            self.asked.append(questions)
            raise AssertionError("the decider was asked")

    backend = LocalBackend(":memory:")
    try:
        other = backend.insert_memory(Memory(content="User likes green tea", user_id="ada",
                                             run_id="s1"))
        decider = Recording()
        action = reconcile_candidate(
            candidate=CandidateFact(content="User likes green tea"),
            scope=Scope(user_id="ada", run_id="s2"),
            similar=[SearchResult(memory=other, score=1.0)],
            backend=backend, embedder=HashEmbedder(64), llm=FakeLLM(),
            episode_ids=["e2"], decider=decider,
        )
        assert decider.asked == []
        assert (action.event, action.memory_id) == ("NONE", other.id)
        assert backend.get_memory(other.id).source_episode_ids == ["e2"]
        assert backend.get_memory(other.id).invalid_at is None
    finally:
        backend.close()


def test_within_one_run_same_and_more_act_on_the_runs_memory(store, fake_llm):
    fake_llm.queue(facts_response(fact("User likes green tea")), coverage())
    first = store.add("I like green tea", user_id="ada", run_id="s1").actions[0]
    fake_llm.queue(facts_response(fact("User enjoys green tea")),
                   decision("SAME", target=0), coverage())
    same = store.add("I enjoy green tea", user_id="ada", run_id="s1").actions[0]
    assert (same.event, same.memory_id) == ("NONE", first.memory_id)
    merged = "User likes green tea, brewed at 80 degrees"
    fake_llm.queue(facts_response(fact("User brews green tea at 80 degrees")),
                   decision("MORE", target=0, content=merged), facts_response(), coverage())
    more = store.add("I brew it at 80 degrees", user_id="ada", run_id="s1").actions[0]
    assert more.event == "UPDATE" and more.memory_id != first.memory_id
    assert [m.content for m in store.get_all(user_id="ada")] == [merged]
    assert fake_llm.responses == []


def test_the_tag_vocabulary_offered_includes_topics_from_other_runs(store, fake_llm):
    store.add("Kitchen sockets are ordered", user_id="ada", run_id="r1", infer=False,
              categories=["kitchen renovation"])
    store.add("Ada likes green tea", user_id="bob", run_id="r2", infer=False,
              categories=["tea"])  # another person's topic is never offered
    assert store._tag_vocabulary(Scope(user_id="ada", run_id="r2")) == ["kitchen renovation"]

    fake_llm.queue(facts_response(fact("The tiles arrive on Friday")), coverage())
    store.add("Tiles arrive on Friday", user_id="ada", run_id="r2")
    prompt = fake_llm.calls[0][1]
    assert "kitchen renovation" in prompt and '"tea"' not in prompt


def test_a_tag_is_canonicalized_against_topics_from_other_runs(verbatim_store):
    verbatim_store.add("Kitchen sockets are ordered", user_id="ada", run_id="r1",
                       infer=False, categories=["kitchen renovation"])
    verbatim_store.add("Tiles arrive on Friday", user_id="ada", run_id="r2",
                       infer=False, categories=["kitchen-renovation"])
    tags = {m.content: m.categories for m in verbatim_store.get_all(user_id="ada")}
    assert tags["Tiles arrive on Friday"] == ["kitchen renovation"]


# ------------------------------------------- replaying a dated conversation
STAMP = "2023-05-08T13:56:02+00:00"


def _dated(content, when=None):
    return {**fact(content), "when": when}


def test_add_takes_the_time_of_the_save_and_metadata_for_its_memories(verbatim_store):
    result = verbatim_store.add("Maya adopted a cat", user_id="ada", infer=False,
                                metadata={"context": "chat"}, created_at=STAMP,
                                memory_metadata={"bench": {"turns": ["D1:3"]}})
    memory = verbatim_store.get(result.actions[0].memory_id)
    assert memory.created_at == memory.updated_at == memory.valid_from == STAMP
    assert memory.metadata["bench"] == {"turns": ["D1:3"]}
    assert memory.metadata["context"] == "chat"
    [episode] = verbatim_store.episodes(user_id="ada")
    assert episode.created_at == STAMP and episode.metadata == {"context": "chat"}


def test_extraction_reads_now_and_every_extracted_memory_gets_the_save_time(store, fake_llm):
    from datetime import datetime, timezone

    fake_llm.queue(
        facts_response(
            _dated("Maya adopted a cat on 2023-05-07",
                   {"start": "2023-05-07", "end": None, "recurrence": None}),
            # the write date read back from a text that names none: dropped,
            # which only happens when the day of writing is ``now``
            _dated("Maya is happy", {"start": "2023-05-08", "end": None, "recurrence": None}),
        ),
        coverage(),
    )
    result = store.add("Yesterday I adopted a cat, I am so happy", user_id="ada",
                       created_at=STAMP, now=datetime(2023, 5, 8, 13, 56, tzinfo=timezone.utc),
                       memory_metadata={"bench": {"session": "s1"},
                                        "when": {"start": "2023-05-08", "by": "session"}})

    assert "Today's date is 2023-05-08" in fake_llm.calls[0][0]
    memories = {m.content: m for m in (store.get(a.memory_id) for a in result.actions)}
    assert all(m.created_at == m.updated_at == m.valid_from == STAMP for m in memories.values())
    assert all(m.metadata["bench"] == {"session": "s1"} for m in memories.values())
    # the memory's own "when" is kept; the caller's fills in where it had none
    assert memories["Maya adopted a cat on 2023-05-07"].metadata["when"]["start"] == "2023-05-07"
    assert memories["Maya is happy"].metadata["when"] == {"start": "2023-05-08", "by": "session"}


def test_a_merged_text_is_dated_at_the_save_and_the_old_one_ends_there(store, fake_llm):
    """A MORE no longer rewrites the old memory in place under its old date:
    the merged text is a new memory of the save's time, and the old one goes
    out of use at that time, kept as history."""
    fake_llm.queue(facts_response(fact("User works at Northwind")), coverage())
    target = store.add("I work at Northwind", user_id="ada", created_at=STAMP).actions[0]
    fake_llm.queue(
        facts_response(fact("User works at Northwind as a data engineer")),
        decision("MORE", target=0, content="User works at Northwind as a data engineer"),
        facts_response(),
        coverage(),
    )
    later = "2023-05-25T19:30:00+00:00"
    result = store.add("I'm a data engineer there", user_id="ada", created_at=later)
    assert result.actions[0].event == "UPDATE"
    merged = store.get(result.actions[0].memory_id)
    assert (merged.created_at, merged.updated_at, merged.valid_from) == (later, later, later)
    old = store.get(target.memory_id)
    assert (old.content, old.created_at, old.invalid_at) == (
        "User works at Northwind", STAMP, later)


def test_a_memory_a_dated_save_supersedes_goes_out_of_use_at_the_save_time(store, fake_llm):
    """A replay's contradiction retires the old memory when the replayed save
    happened, not when the replay ran: its ``invalid_at`` and ``updated_at``
    are the save's ``created_at``."""
    fake_llm.queue(facts_response(fact("User lives in Munich")), coverage())
    old = store.add("I live in Munich", user_id="ada", created_at=STAMP).actions[0]
    later = "2023-05-25T19:30:00+00:00"
    fake_llm.queue(facts_response(fact("User lives in Amsterdam")),
                   decision("WRONG", target=0, reason="it was never Munich"), coverage())
    result = store.add("I moved to Amsterdam", user_id="ada", created_at=later)
    assert result.actions[0].event == "DELETE"
    retired = store.get(old.memory_id)
    assert retired.superseded_by == result.actions[0].memory_id
    assert (retired.created_at, retired.invalid_at, retired.updated_at) == (STAMP, later, later)


def test_a_dated_supersede_never_moves_updated_at_back_and_repair_agrees(store, fake_llm):
    """A memory changed live at T2 and then superseded by a replayed save dated
    T1 < T2 goes out of use at T1 and keeps updated_at T2. Every event the
    replay records carries T1, so ``repair_updated_at`` reads the same times
    and changes nothing."""
    from memry.models import MemoryEvent

    fake_llm.queue(facts_response(fact("User lives in Munich")), coverage())
    munich = store.add("I live in Munich", user_id="ada", created_at=STAMP).actions[0].memory_id
    live = "2023-06-01T09:00:00+00:00"  # T2: a change made live, after the replayed save
    store.backend.update_memory(munich, content="User lives in Munich, Germany", touch=False)
    store.backend.set_memory_timestamp(munich, live)
    store.backend.add_event(MemoryEvent(
        memory_id=munich, event="UPDATE", old_content="User lives in Munich",
        new_content="User lives in Munich, Germany", actor="user", created_at=live))
    replayed = "2023-05-25T19:30:00+00:00"  # T1
    fake_llm.queue(facts_response(fact("User lives in Amsterdam")),
                   decision("WRONG", target=0, reason="it was never Munich"), coverage())
    moved = store.add("I moved to Amsterdam", user_id="ada", created_at=replayed).actions[0]
    assert moved.event == "DELETE"
    retired = store.get(munich)
    assert (retired.invalid_at, retired.updated_at) == (replayed, live)
    [supersede] = [e for e in store.history(munich) if e.event == "SUPERSEDE"]
    assert supersede.created_at == replayed
    assert store.repair_updated_at(user_id="ada") == {"fixed": 0}
    assert store.get(munich).updated_at == live
    assert store.get(moved.memory_id).updated_at == replayed


def test_repair_compares_times_as_times_not_as_text(store):
    """A live "...T10:00:00.500000+00:00" is later than a replayed
    "...T10:00:00Z", though the replayed one sorts later as text: repair keeps
    the live updated_at, as the write path does (``later_ts``), and does not
    rewrite one that names the same instant in another form. A housekeeping
    bump past the last content event is still repaired."""
    from memry.models import Memory, MemoryEvent

    live, replayed = "2024-01-01T10:00:00.500000+00:00", "2024-01-01T10:00:00Z"
    memory = store.backend.insert_memory(Memory(
        content="User lives in Munich", user_id="ada", created_at=live, updated_at=live))
    store.backend.add_event(MemoryEvent(memory_id=memory.id, event="ADD",
                                        new_content=memory.content, created_at=live))
    store.backend.add_event(MemoryEvent(memory_id=memory.id, event="SUPERSEDE",
                                        old_content=memory.content, created_at=replayed))
    same = store.backend.insert_memory(Memory(
        content="User likes tea", user_id="ada", created_at="2024-01-01T09:00:00+00:00",
        updated_at="2024-01-01T09:00:00Z"))
    assert store.repair_updated_at(user_id="ada") == {"fixed": 0}
    assert store.get(memory.id).updated_at == live
    assert store.get(same.id).updated_at == "2024-01-01T09:00:00Z"
    store.backend.set_memory_timestamp(memory.id, "2024-02-01T00:00:00+00:00")
    assert store.repair_updated_at(user_id="ada") == {"fixed": 1}
    assert store.get(memory.id).updated_at == live


def test_a_deferred_save_keeps_its_time_metadata_and_date_for_distillation(store, fake_llm):
    from datetime import datetime, timezone

    raw = store.add_deferred("Maya adopted a cat yesterday", user_id="ada", created_at=STAMP,
                             memory_metadata={"bench": {"session": "s1"}},
                             now=datetime(2023, 5, 8, tzinfo=timezone.utc))
    pending = store.get(raw.actions[0].memory_id)
    assert pending.created_at == pending.valid_from == STAMP
    assert pending.metadata["bench"] == {"session": "s1"}
    [episode] = store.episodes(user_id="ada")
    assert episode.created_at == STAMP
    # the quiet period counts from when it was queued, not from created_at
    assert store.process_pending_enrichments(quiet_seconds=120)["claimed"] == 0

    fake_llm.queue(facts_response(fact("Maya adopted a cat on 2023-05-07")), coverage())
    assert store.process_pending_enrichments(quiet_seconds=0)["succeeded"] == 1
    assert "Today's date is 2023-05-08" in fake_llm.calls[0][0]
    [distilled] = store.get_all(user_id="ada")
    assert distilled.content == "Maya adopted a cat on 2023-05-07"
    assert distilled.created_at == distilled.updated_at == distilled.valid_from == STAMP
    assert distilled.metadata["bench"] == {"session": "s1"}
    assert "_enrichment" not in store.get(pending.id).metadata
    # the raw memory went out of use at the save's time too
    raw_after = store.get(pending.id)
    assert raw_after.invalid_at == raw_after.updated_at == STAMP
