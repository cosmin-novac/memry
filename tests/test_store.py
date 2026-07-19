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
