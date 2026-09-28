from __future__ import annotations

import json

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
        decision("DELETE", target=0, reason="moved cities"),
    )
    result = store.add("I moved to Amsterdam", user_id="ada")
    assert result.actions[0].event == "DELETE"

    active = store.get_all(user_id="ada")
    assert [m.content for m in active] == ["User lives in Amsterdam"]

    archived = store.get(old.id)
    assert archived.invalid_at is not None
    assert archived.superseded_by == active[0].id
    events = [e.event for e in store.history(old.id)]
    assert "SUPERSEDE" in events


def test_update_merges_existing_memory(store, fake_llm):
    fake_llm.queue(facts_response(fact("User works at Northwind")))
    store.add("I work at Northwind", user_id="ada")
    target = store.get_all(user_id="ada")[0]

    fake_llm.queue(
        facts_response(fact("User works at Northwind as a data engineer")),
        decision(
            "UPDATE", target=0, content="User works at Northwind as a data engineer"
        ),
        facts_response(fact("User works at Northwind as a data engineer")),
    )
    result = store.add("I'm a data engineer there", user_id="ada")
    assert result.actions[0].event == "UPDATE"
    updated = store.get(target.id)
    assert updated.content == "User works at Northwind as a data engineer"
    assert [e.event for e in store.history(target.id)] == ["ADD", "UPDATE"]


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


# ------------------------------------------------ one person's saves, any run
def test_a_duplicate_saved_in_another_run_is_added_to_that_run(verbatim_store):
    """Reconcile stays within the save's run: the same fact saved in a second
    session is a memory of that session, so a search of the session finds it
    (the consolidation pass merges duplicates across runs). Within one run it
    is a duplicate."""
    first = verbatim_store.add("Ada likes green tea", user_id="ada", run_id="r1", infer=False)
    again = verbatim_store.add("Ada likes green tea", user_id="ada", run_id="r2", infer=False)
    assert again.actions[0].event == "ADD"
    assert again.actions[0].memory_id != first.actions[0].memory_id
    found = verbatim_store.search("green tea", user_id="ada", run_id="r2", limit=5)
    assert [r.memory.id for r in found] == [again.actions[0].memory_id]
    same_run = verbatim_store.add("Ada likes green tea", user_id="ada", run_id="r2",
                                  infer=False)
    assert same_run.actions[0].event == "NONE"
    assert same_run.actions[0].memory_id == again.actions[0].memory_id
    assert len(verbatim_store.get_all(user_id="ada")) == 2


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


def test_a_rewrite_is_stamped_with_the_time_of_the_save(store, fake_llm):
    fake_llm.queue(facts_response(fact("User works at Northwind")), coverage())
    target = store.add("I work at Northwind", user_id="ada", created_at=STAMP).actions[0]
    fake_llm.queue(
        facts_response(fact("User works at Northwind as a data engineer")),
        decision("UPDATE", target=0, content="User works at Northwind as a data engineer"),
        facts_response(),
        coverage(),
    )
    later = "2023-05-25T19:30:00+00:00"
    result = store.add("I'm a data engineer there", user_id="ada", created_at=later)
    assert result.actions[0].event == "UPDATE"
    updated = store.get(target.memory_id)
    assert (updated.created_at, updated.updated_at) == (STAMP, later)


def test_a_memory_a_dated_save_supersedes_goes_out_of_use_at_the_save_time(store, fake_llm):
    """A replay's contradiction retires the old memory when the replayed save
    happened, not when the replay ran: its ``invalid_at`` and ``updated_at``
    are the save's ``created_at``."""
    fake_llm.queue(facts_response(fact("User lives in Munich")), coverage())
    old = store.add("I live in Munich", user_id="ada", created_at=STAMP).actions[0]
    later = "2023-05-25T19:30:00+00:00"
    fake_llm.queue(facts_response(fact("User lives in Amsterdam")),
                   decision("DELETE", target=0, reason="moved cities"), coverage())
    result = store.add("I moved to Amsterdam", user_id="ada", created_at=later)
    assert result.actions[0].event == "DELETE"
    retired = store.get(old.memory_id)
    assert retired.superseded_by == result.actions[0].memory_id
    assert (retired.created_at, retired.invalid_at, retired.updated_at) == (STAMP, later, later)


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
