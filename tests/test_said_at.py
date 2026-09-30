"""A caller can say on which day what it saves was said (``said_at``).

Content said on another day, such as an import or an earlier conversation, is
saved over MCP (``save_memories``) or REST (``POST /api/v1/memories``) with the
day it was said. Memry stores it as the save's time (``created_at``) and reads
relative times against it: the extraction prompt's "today" and the day the
when-check takes as the day of writing (``_confirm_candidate_whens``).
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest
from mcp.server.fastmcp.exceptions import ToolError
from starlette.testclient import TestClient

from conftest import FakeLLM, decision, facts_response

from memry.config import Config
from memry.mcp_server import INSTRUCTIONS, create_server
from memry import models
from memry.models import parse_ts
from memry.providers.embeddings import HashEmbedder
from memry.rest import create_app
from memry.store import MemoryStore

SAID = "2023-05-08"  # a Monday: "last Friday" is 2023-05-05


def _store(llm: FakeLLM) -> MemoryStore:
    return MemoryStore(Config(db_path=":memory:"), llm=llm, embedder=HashEmbedder(64))


def _fact(content: str, when: dict | None) -> dict:
    return {"content": content, "type": "episodic", "importance": 0.6, "categories": [],
            "entities": [], "relations": [], "when": when, "sources": [1]}


def _extraction(*facts: dict) -> list[str]:
    """What the stub text model answers: the facts, then the coverage audit."""
    return [facts_response(*facts), json.dumps({"missing": []})]


def _call(server, name: str, arguments: dict):
    result = asyncio.run(server.call_tool(name, arguments))
    return json.loads(result.content[0].text)


def _relative_time_facts() -> list[str]:
    # The stub resolves "last Friday" as a model does against the prompt's
    # today, and dates "I signed the lease" on the day of writing, which the
    # when-check drops as the write date read back when that day is the said
    # day and the text names no date.
    return _extraction(
        _fact("Ada adopted a puppy on Friday 2023-05-05", {"start": "2023-05-05"}),
        _fact("Ada signed the lease for her flat", {"start": SAID}),
    )


def _assert_read_against_the_said_day(store: MemoryStore, llm: FakeLLM) -> None:
    extraction_prompt = llm.calls[0][0]
    assert f"Today's date is {SAID}." in extraction_prompt
    facts = {m.content: m for m in store.get_all(user_id="ada")}
    puppy = facts["Ada adopted a puppy on Friday 2023-05-05"]
    lease = facts["Ada signed the lease for her flat"]
    assert puppy.metadata["when"] == {"start": "2023-05-05"}
    assert "when" not in lease.metadata  # the said day read back as the "when"
    for memory in (puppy, lease):
        assert memory.created_at == f"{SAID}T00:00:00+00:00"
        assert memory.updated_at == memory.valid_from == memory.created_at
    [episode] = store.episodes(user_id="ada")
    assert episode.created_at == f"{SAID}T00:00:00+00:00"


# ---------------------------------------------------------------- MCP


def test_mcp_said_at_dates_a_deferred_save_and_extraction_reads_against_it():
    llm = FakeLLM(_relative_time_facts())
    store = _store(llm)
    server = create_server(store)

    saved = _call(server, "save_memories", {
        "content": "Ada: I adopted a puppy last Friday.\nAda: I signed the lease for my flat.",
        "user_id": "ada", "said_at": SAID,
    })
    pending = store.get(saved["actions"][0]["memory_id"])
    assert pending.created_at == f"{SAID}T00:00:00+00:00"
    assert llm.calls == []  # the save itself asks no model

    outcome = store.process_pending_enrichments(
        quiet_seconds=120, now=datetime.now(timezone.utc) + timedelta(seconds=300))

    assert outcome["succeeded"] == 1
    _assert_read_against_the_said_day(store, llm)
    rows = _call(server, "search_memories", {"query": "Ada puppy", "user_id": "ada"})
    puppy = next(r for r in rows if "puppy" in r["content"])
    assert puppy["said"] == SAID
    assert puppy["happened"] == "happened 2023-05-05"
    assert [turn["said"] for turn in puppy["evidence"]] == [SAID]
    store.close()


def test_mcp_said_at_dates_a_verbatim_save():
    store = _store(FakeLLM())
    server = create_server(store)
    saved = _call(server, "save_memories", {
        "content": "Ada: my flat is in Porto.", "user_id": "ada", "infer": False,
        "said_at": "2023-05-08T14:30:00+02:00",
    })
    memory = store.get(saved["actions"][0]["memory_id"])
    assert memory.created_at == "2023-05-08T12:30:00+00:00"  # the same instant, in UTC
    store.close()


def test_mcp_without_said_at_saves_as_before():
    store = _store(FakeLLM())
    server = create_server(store)
    before = datetime.now(timezone.utc).replace(microsecond=0)
    saved = _call(server, "save_memories", {"content": "Ada lives in Porto", "infer": False})
    assert parse_ts(store.get(saved["actions"][0]["memory_id"]).created_at) >= before
    store.close()


@pytest.mark.parametrize("said_at,message", [
    ("8 May 2023", "is not an ISO 8601 date"),
    ("yesterday", "is not an ISO 8601 date"),
    ("2023-02-30", "is not an ISO 8601 date"),
    ((datetime.now(timezone.utc) + timedelta(days=2)).date().isoformat(), "is after today"),
])
def test_mcp_refuses_a_malformed_or_future_said_at(said_at, message):
    store = _store(FakeLLM())
    server = create_server(store)
    with pytest.raises(ToolError, match=message):
        asyncio.run(server.call_tool("save_memories", {
            "content": "Ada lives in Porto", "user_id": "ada", "said_at": said_at}))
    assert store.get_all(user_id="ada") == []
    assert store.episodes(user_id="ada") == []
    store.close()


def test_the_save_tool_describes_said_at():
    server = create_server(_store(FakeLLM()))
    tools = {tool.name: tool for tool in asyncio.run(server.list_tools())}
    save = tools["save_memories"]
    assert save.inputSchema["properties"]["said_at"]["default"] == ""
    assert "said_at" not in save.inputSchema.get("required", [])
    assert "import or an earlier conversation" in save.description
    assert "concise" not in save.description


# ---------------------------------------------------------------- REST


def test_rest_said_at_dates_a_save_and_extraction_reads_against_it():
    llm = FakeLLM(_relative_time_facts())
    store = _store(llm)
    with TestClient(create_app(store)) as client:
        response = client.post("/api/v1/memories", json={
            "messages": [{"role": "Ada", "content": "I adopted a puppy last Friday."},
                         {"role": "Ada", "content": "I signed the lease for my flat."}],
            "user_id": "ada", "said_at": SAID,
        })
    assert response.status_code == 201, response.text
    assert "This conversation names its speakers" in llm.calls[0][1]
    facts = {m.content: m for m in store.get_all(user_id="ada")}
    assert facts["Ada adopted a puppy on Friday 2023-05-05"].metadata["when"] == {
        "start": "2023-05-05"}
    assert f"Today's date is {SAID}." in llm.calls[0][0]
    assert "when" not in facts["Ada signed the lease for her flat"].metadata
    assert {e.created_at for e in store.episodes(user_id="ada")} == {f"{SAID}T00:00:00+00:00"}
    assert {m.created_at for m in facts.values()} == {f"{SAID}T00:00:00+00:00"}
    store.close()


def test_rest_said_at_dates_a_deferred_save():
    store = _store(FakeLLM())
    with TestClient(create_app(store)) as client:
        response = client.post("/api/v1/memories", json={
            "content": "Ada adopted a puppy last Friday.", "user_id": "ada",
            "defer": True, "said_at": SAID})
    assert response.status_code == 202, response.text
    pending = store.get(response.json()["actions"][0]["memory_id"])
    assert pending.created_at == f"{SAID}T00:00:00+00:00"
    job = pending.metadata["_enrichment"]
    assert job["created_at"] == f"{SAID}T00:00:00+00:00"
    assert job["now"].startswith(f"{SAID}T00:00:00")  # extraction's today when distilled
    store.close()


@pytest.mark.parametrize("said_at", ["8 May 2023", 20230508, "2999-01-01"])
def test_rest_refuses_a_malformed_or_future_said_at(said_at):
    store = _store(FakeLLM())
    with TestClient(create_app(store)) as client:
        response = client.post("/api/v1/memories", json={
            "content": "Ada lives in Porto", "user_id": "ada", "infer": False,
            "said_at": said_at})
    assert response.status_code == 400
    assert "said_at" in response.json()["error"]
    assert store.get_all(user_id="ada") == []
    store.close()


# ---------------------------------------------------------------- grouping


class _ByPrompt(FakeLLM):
    """Answers each kind of prompt on its own: the queued extractions in order,
    NEW to every reconcile question, and a clean coverage audit."""

    def complete(self, system: str, user: str, *, json_schema=None) -> str:
        self.calls.append((system, user))
        if system.startswith("You are the long-term memory extraction system"):
            return self.responses.pop(0)
        if system.startswith("You audit"):
            return json.dumps({"missing": []})
        return decision("NEW")


def test_saves_said_on_different_days_are_not_extracted_as_one_group():
    """One extraction has one "today". A save said on another day and a save
    said now, in one scope and context, are extracted apart, each read
    against its own day."""
    llm = _ByPrompt([
        facts_response(_fact("Ada adopted a puppy on Friday 2023-05-05", {"start": "2023-05-05"})),
        facts_response(_fact("Ada walks the puppy every morning", None)),
    ])
    store = _store(llm)
    store.add_deferred("Ada: I adopted a puppy last Friday.", user_id="ada",
                       metadata={"context": "Ada's puppy"},
                       created_at=f"{SAID}T00:00:00+00:00", now=parse_ts(SAID))
    store.add_deferred("Ada: I walk the puppy every morning.", user_id="ada",
                       metadata={"context": "Ada's puppy"})

    outcome = store.process_pending_enrichments(
        quiet_seconds=120, now=datetime.now(timezone.utc) + timedelta(seconds=300))

    assert outcome["succeeded"] == 2
    # pending saves are taken oldest first: the one said on SAID, then today's
    extractions = [(system, user) for system, user in llm.calls if "Today's date is" in system]
    today = datetime.now(timezone.utc).date().isoformat()
    assert len(extractions) == 2
    assert f"Today's date is {SAID}." in extractions[0][0]
    assert "last Friday" in extractions[0][1] and "every morning" not in extractions[0][1]
    assert f"Today's date is {today}." in extractions[1][0]
    assert "every morning" in extractions[1][1] and "last Friday" not in extractions[1][1]
    dated = {m.content: m.created_at[:10] for m in store.get_all(user_id="ada")}
    assert dated["Ada adopted a puppy on Friday 2023-05-05"] == SAID
    assert dated["Ada walks the puppy every morning"] == today
    store.close()


# ---------------------------------------------------------------- parsing


def test_parse_said_at_reads_a_date_or_a_time_in_utc():
    now = datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc)

    def said(value):
        return models.parse_said_at(value, now=now)

    assert said("") is None
    assert said(None) is None
    assert said("2023-05-08") == datetime(2023, 5, 8, tzinfo=timezone.utc)
    assert said("2023-05-08T14:30") == datetime(2023, 5, 8, 14, 30, tzinfo=timezone.utc)
    assert said("2023-05-08T23:30:00-05:00") == datetime(2023, 5, 9, 4, 30, tzinfo=timezone.utc)
    # a later time today is now (a client clock a little ahead); a later day is refused
    assert said("2026-09-30T18:00:00Z") == now
    with pytest.raises(ValueError, match="after today"):
        said("2026-10-01")
    for bad in ("8 May 2023", "2023-05", "2023-13-01", 20230508):
        with pytest.raises(ValueError, match="said_at"):
            said(bad)


# ---------------------------------------------------------------- the instructions


def test_the_instructions_tell_an_agent_to_send_what_was_said():
    assert "concise multiline" not in INSTRUCTIONS
    assert "send what was said, close to the words used, one statement per line" in (
        " ".join(INSTRUCTIONS.split()))
    assert "name any speaker who is not the user" in " ".join(INSTRUCTIONS.split())
    # the batching, grouping and tag rules stay
    assert "batch related facts into ONE call" in INSTRUCTIONS
    assert "two minutes of quiet" in INSTRUCTIONS
    assert "context label and run_id" in " ".join(INSTRUCTIONS.split())
    assert "up to three tags" in INSTRUCTIONS


def test_the_instructions_say_how_to_read_dates_and_when_to_pass_said_at():
    text = " ".join(INSTRUCTIONS.split())
    assert "(said 8 May 2023)" in text and "[until <date>]" in text
    assert '"What was said"' in text and '"evidence"' in text
    assert "pass said_at (YYYY-MM-DD) only for content said on another day" in text


def test_the_instructions_keep_old_values_to_memry():
    text = " ".join(INSTRUCTIONS.split())
    assert "save the new statement as said" in text
    assert "do not delete or rewrite it yourself" in text
    assert "Use update_memory only to fix a memory Memry wrote wrong" in text
    assert "delete_memory only when the user asks you to forget" in text
