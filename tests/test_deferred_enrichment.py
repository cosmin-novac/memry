from __future__ import annotations

import asyncio
from datetime import timedelta
import json
import time

from mcp.types import CallToolResult
from starlette.testclient import TestClient

from conftest import FakeLLM, fact, facts_response, mcp_call

from memry.config import Config
from memry.enrichment import EnrichmentWorker
from memry.intelligence.extraction import COVERAGE_SYSTEM
from memry.mcp_server import INSTRUCTIONS, create_server
from memry.models import parse_ts
from memry.providers.embeddings import HashEmbedder
from memry.rest import create_app
from memry.store import MemoryStore


def _audit(*missing: str) -> str:
    """The coverage audit's answer, which runs after every distillation."""
    return json.dumps({"missing": list(missing)})


def _store(db_path: str, llm: FakeLLM) -> MemoryStore:
    return MemoryStore(
        Config(db_path=db_path),
        llm=llm,
        embedder=HashEmbedder(64),
    )


def _call_tool(server, name: str, arguments: dict) -> dict:
    result = asyncio.run(server.call_tool(name, arguments))
    if isinstance(result, CallToolResult):
        blocks = result.content
    else:
        blocks = result[0] if isinstance(result, tuple) else result
    return json.loads(blocks[0].text)


def test_deferred_save_is_durable_without_calling_provider(tmp_path):
    llm = FakeLLM()
    store = _store(str(tmp_path / "memry.db"), llm)
    result = store.add_deferred("Marcus prefers concise answers", user_id="marcus")

    assert result.summary() == {"ADD": 1}
    assert llm.calls == []
    memory = store.get(result.actions[0].memory_id)
    assert memory.content == "Marcus prefers concise answers"
    assert memory.invalid_at is None
    assert memory.metadata["pending_distillation"] is True
    assert memory.metadata["_enrichment"]["status"] == "pending"
    assert store.episodes(user_id="marcus")[0].content == memory.content
    store.close()


def test_pending_save_is_recovered_after_restart(tmp_path):
    path = str(tmp_path / "memry.db")
    first = _store(path, FakeLLM())
    original_id = first.add_deferred(
        "Marcus prefers concise answers", user_id="marcus"
    ).actions[0].memory_id
    first.close()

    llm = FakeLLM([
        facts_response(fact("Marcus prefers concise answers")), _audit(),
    ])
    second = _store(path, llm)
    outcome = second.process_pending_enrichments()

    assert outcome == {"claimed": 1, "succeeded": 1, "failed": 0, "errors": []}
    assert second.get(original_id).invalid_at is not None
    active = second.get_all(user_id="marcus")
    assert [memory.content for memory in active] == ["Marcus prefers concise answers"]
    assert not active[0].metadata.get("pending_distillation")
    second.close()


def test_failed_enrichment_keeps_active_raw_memory_for_retry(tmp_path):
    store = _store(str(tmp_path / "memry.db"), FakeLLM())
    memory_id = store.add_deferred("Never lose this fact", user_id="marcus").actions[0].memory_id

    outcome = store.process_pending_enrichments()

    assert outcome["claimed"] == 1
    assert outcome["failed"] == 1
    memory = store.get(memory_id)
    assert memory.invalid_at is None
    assert memory.content == "Never lose this fact"
    assert memory.metadata["pending_distillation"] is True
    assert memory.metadata["_enrichment"]["status"] == "retry"
    assert store.stats()["retrying_enrichments"] == 1
    store.close()


def test_worker_batch_limit_preserves_independent_pending_records(tmp_path):
    llm = FakeLLM([
        facts_response(fact("Fact one")), _audit(),
        facts_response(fact("Fact two")), _audit(),
    ])
    store = _store(str(tmp_path / "memry.db"), llm)
    for number in ("one", "two", "three"):
        store.add_deferred(f"Fact {number}", user_id=number)

    outcome = store.process_pending_enrichments(limit=2)

    assert outcome["claimed"] == 2
    assert outcome["succeeded"] == 2
    assert store.stats()["pending_enrichments"] == 1
    pending = store.backend.list_pending_memories()
    assert len(pending) == 1
    assert pending[0].user_id == "three"
    store.close()


def test_two_minute_quiet_period_delays_enrichment(tmp_path):
    llm = FakeLLM([
        facts_response(fact("Marcus prefers concise answers")), _audit(),
    ])
    store = _store(str(tmp_path / "memry.db"), llm)
    saved = store.add_deferred(
        "Marcus prefers concise answers", user_id="marcus"
    )
    queued = store.get(saved.actions[0].memory_id)

    early = store.process_pending_enrichments(
        quiet_seconds=120,
        now=parse_ts(queued.created_at) + timedelta(seconds=119),
    )
    ready = store.process_pending_enrichments(
        quiet_seconds=120,
        now=parse_ts(queued.created_at) + timedelta(seconds=121),
    )

    assert early == {"claimed": 0, "succeeded": 0, "failed": 0, "errors": []}
    assert ready == {"claimed": 1, "succeeded": 1, "failed": 0, "errors": []}
    assert len(llm.calls) == 2  # extraction and the coverage audit
    store.close()


def test_related_pending_saves_are_extracted_as_one_context(tmp_path):
    llm = FakeLLM([
        facts_response(
            fact(
                "The evaluation service uses local tests for deterministic validation.",
                categories=["ai evaluation"],
            ),
            fact(
                "The evaluation service uses E2E tests for final quality validation.",
                categories=["regression testing"],
            ),
        ),
        _audit(),
    ])
    store = _store(str(tmp_path / "memry.db"), llm)
    metadata = {
        "context": "AI-agent evaluation strategy",
        "tag_hints": ["AI Evaluation", "Regression Testing"],
    }
    first = store.add_deferred(
        "Local tool tests cover deterministic behavior.",
        user_id="marcus",
        run_id="run-1",
        metadata=metadata,
        categories=["ai evaluation"],
    )
    second = store.add_deferred(
        "E2E tests cover final agent quality.",
        user_id="marcus",
        run_id="run-1",
        metadata=metadata,
        categories=["regression testing"],
    )
    latest = store.get(second.actions[0].memory_id)

    outcome = store.process_pending_enrichments(
        quiet_seconds=120,
        now=parse_ts(latest.created_at) + timedelta(seconds=121),
    )

    assert outcome == {"claimed": 2, "succeeded": 2, "failed": 0, "errors": []}
    assert len(llm.calls) == 2  # one extraction for both, and the coverage audit
    prompt = llm.calls[0][1]
    assert "Local tool tests cover deterministic behavior." in prompt
    assert "E2E tests cover final agent quality." in prompt
    assert "Shared context for these related inputs:\nAI-agent evaluation strategy" in prompt
    assert "Client-suggested tags" in prompt
    assert '["ai evaluation", "regression testing"]' in prompt
    episode_ids = set(first.episode_ids + second.episode_ids)
    active = store.get_all(user_id="marcus", run_id="run-1")
    assert len(active) == 2
    assert all(set(memory.source_episode_ids) == episode_ids for memory in active)
    # the facts keep the label: the identity judge is shown it, and it groups a
    # conversation's memories when the client sent no session id
    assert {m.metadata.get("context") for m in active} == {"AI-agent evaluation strategy"}
    store.close()


def test_a_direct_save_keeps_its_context_label(tmp_path):
    llm = FakeLLM([facts_response(fact("The kitchen needs new sockets."))])
    store = _store(str(tmp_path / "memry.db"), llm)
    store.add("The kitchen needs new sockets.", user_id="ada",
              metadata={"context": "kitchen renovation"})
    [memory] = store.get_all(user_id="ada")
    assert memory.metadata["context"] == "kitchen renovation"
    store.close()

def test_lost_context_labels_are_restored_from_the_saves(tmp_path):
    """Memories distilled before facts kept the save's label get it back from
    the save's episode; a save without a label leaves nothing to restore."""
    llm = FakeLLM([
        facts_response(fact("The kitchen needs new sockets."), fact("Tiles arrive on Friday.")),
        _audit(),
        facts_response(fact("The user likes green tea.")),
        _audit(),
    ])
    store = _store(str(tmp_path / "memry.db"), llm)
    store.add_deferred("Kitchen: new sockets, tiles on Friday.", user_id="ada", run_id="r1",
                       metadata={"context": "kitchen renovation"})
    store.add_deferred("I like green tea.", user_id="ada", run_id="r2")
    for pending in store.backend.list_pending_memories(limit=10):
        store.distill(pending.id)
    for memory in store.get_all(user_id="ada"):  # as distilled before the fix
        store.backend.update_memory(
            memory.id, metadata={k: v for k, v in memory.metadata.items() if k != "context"},
            touch=False)

    assert store.restore_context_labels(user_id="ada", dry_run=True) == {
        "without_label": 3, "restorable": 2, "restored": 0, "save_had_no_label": 1}
    assert not any(m.metadata.get("context") for m in store.get_all(user_id="ada"))
    assert store.restore_context_labels(user_id="ada")["restored"] == 2
    labels = {m.content: m.metadata.get("context") for m in store.get_all(user_id="ada")}
    assert labels == {"The kitchen needs new sockets.": "kitchen renovation",
                      "Tiles arrive on Friday.": "kitchen renovation",
                      "The user likes green tea.": None}
    assert store.restore_context_labels(user_id="ada")["restored"] == 0
    store.close()


def test_same_scope_burst_coalesces_without_explicit_context(tmp_path):
    llm = FakeLLM([
        facts_response(
            fact("Fact one in the shared client burst."),
            fact("Fact two in the shared client burst."),
        ),
        _audit(),
    ])
    store = _store(str(tmp_path / "memry.db"), llm)
    first = store.add_deferred("Fact one", user_id="marcus", run_id="run-1")
    second = store.add_deferred("Fact two", user_id="marcus", run_id="run-1")
    latest = store.get(second.actions[0].memory_id)

    outcome = store.process_pending_enrichments(
        quiet_seconds=120,
        now=parse_ts(latest.created_at) + timedelta(seconds=121),
    )

    assert outcome["claimed"] == 2
    assert outcome["succeeded"] == 2
    assert len(llm.calls) == 2  # one extraction for both, and the coverage audit
    assert set(first.episode_ids + second.episode_ids) == {
        episode_id
        for memory in store.get_all(user_id="marcus", run_id="run-1")
        for episode_id in memory.source_episode_ids
    }
    store.close()


def test_a_distilled_group_takes_its_latest_save_time_compared_as_times(tmp_path):
    """Two saves of one group given their times in two ISO forms: the facts
    take the later instant. Compared as text, "...00Z" sorts after
    "...00.500000+00:00" and the earlier time won."""
    from datetime import datetime, timezone

    llm = FakeLLM([facts_response(fact("Ada moved to Berlin in spring.")), _audit()])
    store = _store(str(tmp_path / "memry.db"), llm)
    store.add_deferred("Ada moved", user_id="ada", created_at="2026-01-01T10:00:00Z")
    store.add_deferred("to Berlin in spring", user_id="ada",
                       created_at="2026-01-01T10:00:00.500000+00:00")

    outcome = store.process_pending_enrichments(
        quiet_seconds=120, now=datetime.now(timezone.utc) + timedelta(seconds=300))

    assert outcome["claimed"] == 2 and outcome["succeeded"] == 2
    [distilled] = store.get_all(user_id="ada")
    assert distilled.content == "Ada moved to Berlin in spring."
    assert parse_ts(distilled.created_at) == parse_ts("2026-01-01T10:00:00.500000+00:00")
    store.close()


def test_mcp_acknowledges_pending_save_before_enrichment(tmp_path):
    llm = FakeLLM()
    store = _store(str(tmp_path / "memry.db"), llm)
    server = create_server(store)

    saved = _call_tool(server, "save_memories", {
        "content": "Ada lives in Berlin",
        "context": "Ada profile",
        "tags": ["People", "Location", "Travel", "ignored fourth tag"],
    })

    assert saved["saved"] == {"ADD": 1}
    assert saved["enrichment"]["status"] == "pending"
    assert saved["enrichment"]["quiet_period_seconds"] == 120
    assert llm.calls == []
    hits = _call_tool(server, "search_memories", {"query": "Berlin"})
    assert hits[0]["content"] == "Ada lives in Berlin"
    assert hits[0]["enrichment"]["status"] == "pending"
    pending = store.get(saved["actions"][0]["memory_id"])
    assert pending.metadata["context"] == "Ada profile"
    assert pending.metadata["tag_hints"] == ["people", "location", "travel"]
    assert pending.categories == ["people", "location", "travel"]
    assert "batch related facts into ONE call" in INSTRUCTIONS
    assert "two minutes of quiet" in INSTRUCTIONS
    store.close()

def test_hosted_mcp_worker_enriches_after_ack(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "memry.rest.EnrichmentWorker",
        lambda store: EnrichmentWorker(store, quiet_seconds=0),
    )
    llm = FakeLLM([
        facts_response(fact("Ada lives in Berlin")), _audit(),
    ])
    store = _store(str(tmp_path / "memry.db"), llm)
    with TestClient(create_app(store), base_url="http://127.0.0.1:8787") as client:
        saved = json.loads(
            mcp_call(client, "unused", "save_memories", {"content": "Ada lives in Berlin"})
        )
        assert saved["enrichment"]["status"] == "pending"

        deadline = time.monotonic() + 2
        while store.stats()["pending_enrichments"] and time.monotonic() < deadline:
            time.sleep(0.01)

        assert store.stats()["pending_enrichments"] == 0
        assert [m.content for m in store.get_all()] == ["Ada lives in Berlin"]
    store.close()


def test_the_coverage_audit_runs_after_distillation(tmp_path):
    """The deferred save (the MCP default) is audited as a direct save is:
    after distillation, one call compares the saved text with the facts, and
    a gap is reported and noted where the raw text went."""
    llm = FakeLLM([
        facts_response(fact("The skill fills timesheets")),
        _audit("must edit XML directly rather than openpyxl"),
    ])
    store = _store(str(tmp_path / "memry.db"), llm)
    raw = store.add_deferred(
        "The skill fills timesheets and must edit XML directly rather than openpyxl.",
        user_id="u").actions[0].memory_id

    result = store.distill(raw)

    assert llm.calls[1][0] == COVERAGE_SYSTEM
    assert "STORED FACTS:\n- The skill fills timesheets" in llm.calls[1][1]
    assert result.warnings and "XML directly" in result.warnings[0]
    distilled = next(e for e in store.history(raw) if e.event == "SUPERSEDE")
    assert distilled.reason == ("distilled with its context into 1 fact(s); not captured "
                                "as facts: must edit XML directly rather than openpyxl")
    assert distilled.kind == "distillation"
    store.close()


def test_the_worker_audits_what_it_distilled(tmp_path):
    llm = FakeLLM([facts_response(fact("Ada lives in Berlin")), _audit()])
    store = _store(str(tmp_path / "memry.db"), llm)
    store.add_deferred("Ada lives in Berlin", user_id="ada")

    assert store.process_pending_enrichments()["succeeded"] == 1
    assert llm.calls[1][0] == COVERAGE_SYSTEM
    assert llm.responses == []
    store.close()
