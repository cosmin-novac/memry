from __future__ import annotations

from conftest import decision, fact, facts_response


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
    fake_llm.queue(facts_response(fact("User lives in Berlin")))
    store.add("I live in Berlin", user_id="ada")

    fake_llm.queue(facts_response(fact("User lives in Berlin")))
    result = store.add("I live in Berlin", user_id="ada")
    assert result.summary() == {"NONE": 1}
    assert len(store.get_all(user_id="ada")) == 1
    # extraction called twice, reconcile decision never needed
    assert len(fake_llm.calls) == 2


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
