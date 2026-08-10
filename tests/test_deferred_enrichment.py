from __future__ import annotations

import asyncio
from datetime import timedelta
import json
import time

from starlette.testclient import TestClient

from conftest import FakeLLM, fact, facts_response, mcp_call

from memry.config import Config
from memry.enrichment import EnrichmentWorker
from memry.mcp_server import INSTRUCTIONS, create_server
from memry.models import parse_ts
from memry.providers.embeddings import HashEmbedder
from memry.rest import create_app
from memry.store import MemoryStore


def _store(db_path: str, llm: FakeLLM) -> MemoryStore:
    return MemoryStore(
        Config(db_path=db_path),
        llm=llm,
        embedder=HashEmbedder(64),
    )


def _call_tool(server, name: str, arguments: dict) -> dict:
    result = asyncio.run(server.call_tool(name, arguments))
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
        facts_response(fact("Marcus prefers concise answers")),
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
        facts_response(fact("Fact one")),
        facts_response(fact("Fact two")),
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
        facts_response(fact("Marcus prefers concise answers")),
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
    assert len(llm.calls) == 1
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
    assert len(llm.calls) == 1
    prompt = llm.calls[0][1]
    assert "Local tool tests cover deterministic behavior." in prompt
    assert "E2E tests cover final agent quality." in prompt
    assert "Shared context for these related inputs:\nAI-agent evaluation strategy" in prompt
    assert "Client-suggested tags" in prompt
    assert "ai evaluation, regression testing" in prompt
    episode_ids = set(first.episode_ids + second.episode_ids)
    active = store.get_all(user_id="marcus", run_id="run-1")
    assert len(active) == 2
    assert all(set(memory.source_episode_ids) == episode_ids for memory in active)
    store.close()

def test_same_scope_burst_coalesces_without_explicit_context(tmp_path):
    llm = FakeLLM([
        facts_response(
            fact("Fact one in the shared client burst."),
            fact("Fact two in the shared client burst."),
        ),
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
    assert len(llm.calls) == 1
    assert set(first.episode_ids + second.episode_ids) == {
        episode_id
        for memory in store.get_all(user_id="marcus", run_id="run-1")
        for episode_id in memory.source_episode_ids
    }
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
        facts_response(fact("Ada lives in Berlin")),
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
