from __future__ import annotations

import asyncio
import json

import pytest
from starlette.testclient import TestClient

from memry.config import Config
from memry.mcp_server import create_server
from memry.providers.embeddings import HashEmbedder
from memry.providers.llm import NoneLLM
from memry.rest import create_app
from memry.store import MemoryStore


def make_store(**config_kwargs) -> MemoryStore:
    cfg = Config(db_path=":memory:", **config_kwargs)
    return MemoryStore(cfg, llm=NoneLLM(), embedder=HashEmbedder(64))


# ---------------------------------------------------------------- MCP tools
def call_tool(server, name: str, args: dict):
    result = asyncio.run(server.call_tool(name, args))
    # FastMCP returns a list of content blocks (or a (blocks, meta) tuple)
    blocks = result[0] if isinstance(result, tuple) else result
    return json.loads(blocks[0].text) if blocks[0].text.startswith(("{", "[")) else blocks[0].text


def test_mcp_save_search_roundtrip():
    server = create_server(make_store())
    saved = call_tool(server, "save_memories", {"content": "Ada lives in Berlin", "infer": False})
    assert saved["saved"] == {"ADD": 1}

    hits = call_tool(server, "search_memories", {"query": "berlin"})
    assert hits and hits[0]["content"] == "Ada lives in Berlin"

    ctx = call_tool(server, "get_memory_context", {"query": "where does ada live"})
    assert "Berlin" in ctx

    listing = call_tool(server, "list_memories", {})
    assert len(listing) == 1

    memory_id = listing[0]["id"]
    updated = call_tool(server, "update_memory", {"memory_id": memory_id, "content": "Ada lives in Paris"})
    assert updated["content"] == "Ada lives in Paris"

    history = call_tool(server, "memory_history", {"memory_id": memory_id})
    assert [e["event"] for e in history] == ["ADD", "UPDATE"]

    deleted = call_tool(server, "delete_memory", {"memory_id": memory_id})
    assert deleted["deleted"] is True

    stats = call_tool(server, "memory_stats", {})
    assert stats["backend"] == "local"


def test_mcp_tools_registered():
    server = create_server(make_store())
    tools = asyncio.run(server.list_tools())
    names = {t.name for t in tools}
    assert {
        "save_memories", "search_memories", "get_memory_context", "list_memories",
        "list_categories", "update_memory", "delete_memory", "memory_history",
        "memory_stats",
    } <= names


def test_mcp_list_categories_sorted_desc():
    store = make_store()
    server = create_server(store)
    store.add("a", user_id="u", infer=False, categories=["work", "diet"])
    store.add("b", user_id="u", infer=False, categories=["work"])
    store.add("c", user_id="u", infer=False, categories=["Work", "travel"])
    cats = call_tool(server, "list_categories", {"user_id": "u"})
    assert cats[0] == {"category": "work", "count": 3}
    assert {c["category"] for c in cats} == {"work", "diet", "travel"}
    counts = [c["count"] for c in cats]
    assert counts == sorted(counts, reverse=True)


# ---------------------------------------------------------------- REST API
@pytest.fixture
def client():
    app = create_app(make_store())
    with TestClient(app) as c:
        yield c


def test_rest_health_and_dashboard(client):
    assert client.get("/health").json()["status"] == "ok"
    dashboard = client.get("/").text
    assert "memry" in dashboard
    assert "const planetOrder=hov?" in dashboard
    assert "ctx.strokeText(labelText" in dashboard
    assert "html.knowledge-open,body.knowledge-open{overflow:hidden}" in dashboard
    assert "function setKnowledgeOpen(open)" in dashboard
    assert 'id="user"' not in dashboard
    assert dashboard.index('id="entlist"') < dashboard.index('id="entitydetail"')
    assert ".tagrow .entity-type{flex:0 0 6.5rem" in dashboard


def test_rest_crud_and_search(client):
    created = client.post(
        "/api/v1/memories",
        json={"content": "Ada prefers dark mode", "user_id": "ada", "infer": False},
    )
    assert created.status_code == 201
    memory_id = created.json()["actions"][0]["memory_id"]

    listing = client.get("/api/v1/memories", params={"user_id": "ada"}).json()
    assert len(listing) == 1

    got = client.get(f"/api/v1/memories/{memory_id}").json()
    assert got["content"] == "Ada prefers dark mode"

    search = client.post(
        "/api/v1/search", json={"query": "dark mode", "user_id": "ada"}
    ).json()
    assert search[0]["memory"]["id"] == memory_id

    ctx = client.post("/api/v1/context", json={"query": "preferences", "user_id": "ada"}).json()
    assert "dark mode" in ctx["text"]

    patched = client.patch(f"/api/v1/memories/{memory_id}", json={"content": "light mode"})
    assert patched.json()["content"] == "light mode"

    history = client.get(f"/api/v1/memories/{memory_id}/history").json()
    assert [e["event"] for e in history] == ["ADD", "UPDATE"]

    deleted = client.delete(f"/api/v1/memories/{memory_id}")
    assert deleted.json()["deleted"] is True

    stats = client.get("/api/v1/stats").json()
    assert stats["backend"] == "local"


def test_rest_missing_content_400(client):
    assert client.post("/api/v1/memories", json={}).status_code == 400


def test_rest_categories(client):
    client.post("/api/v1/memories", json={"content": "a", "user_id": "u", "infer": False, "categories": ["work", "diet"]})
    client.post("/api/v1/memories", json={"content": "b", "user_id": "u", "infer": False, "categories": ["work"]})
    cats = client.get("/api/v1/categories", params={"user_id": "u"}).json()
    assert cats == [{"category": "work", "count": 2}, {"category": "diet", "count": 1}]


def test_rest_lossless_export_and_idempotent_restore(client):
    client.post("/api/v1/memories", json={
        "content": "backup this", "user_id": "backup-user", "infer": False,
        "categories": ["backup"],
    })
    response = client.get("/api/v1/export", params={"user_id": "backup-user"})
    assert response.status_code == 200
    backup = response.json()
    assert backup["format"] == "memry-backup"
    assert backup["tables"]["memories"][0]["content"] == "backup this"
    assert backup["tables"]["episodes"]
    assert backup["tables"]["memory_events"]

    restored = client.post("/api/v1/import", json=backup)
    assert restored.status_code == 200
    assert restored.json()["inserted"] == 0

    backup["tables"]["memories"][0]["content"] = "conflicting content"
    conflict = client.post("/api/v1/import", json=backup)
    assert conflict.status_code == 409

def test_rest_bulk_import(client):
    rows = [
        {"content": "row one", "categories": ["a"]},
        {"content": "row two", "user_id": "other"},
        {"content": ""},
    ]
    res = client.post("/api/v1/import", json={"memories": rows, "user_id": "bulk"})
    assert res.status_code == 201
    body = res.json()
    assert body["imported"] == 2 and body["skipped"] == 1

    listed = client.get("/api/v1/memories", params={"user_id": "bulk"}).json()
    assert [m["content"] for m in listed] == ["row one"]
    assert client.get("/api/v1/memories", params={"user_id": "other"}).json()[0]["content"] == "row two"

    # bare-array form and validation
    assert client.post("/api/v1/import", json=[{"content": "three"}]).status_code == 201
    assert client.post("/api/v1/import", json={}).status_code == 400
    assert client.post("/api/v1/import", json={"memories": []}).status_code == 400


def test_rest_auth_enforced():
    app = create_app(make_store(api_key="sekret"))
    with TestClient(app) as client:
        assert client.get("/api/v1/stats").status_code == 401
        ok = client.get("/api/v1/stats", headers={"Authorization": "Bearer sekret"})
        assert ok.status_code == 200
        # health and dashboard stay open
        assert client.get("/health").status_code == 200


def test_rest_distill_endpoint():
    from conftest import FakeLLM, fact, facts_response

    llm = FakeLLM()
    store = MemoryStore(Config(db_path=":memory:"), llm=llm, embedder=HashEmbedder(64))
    with TestClient(create_app(store)) as client:
        # empty FakeLLM queue raises on complete() -> save degrades to
        # verbatim with a warning and the pending_distillation flag
        created = client.post(
            "/api/v1/memories", json={"content": "Ada lives in Berlin", "user_id": "ada"}
        )
        body = created.json()
        assert body["warnings"] and "stored verbatim" in body["warnings"][0]
        memory_id = body["actions"][0]["memory_id"]
        listed = client.get("/api/v1/memories", params={"user_id": "ada"}).json()
        assert listed[0]["metadata"]["pending_distillation"] is True

        # LLM still failing -> distillation reports the provider error
        failed = client.post(f"/api/v1/memories/{memory_id}/distill")
        assert failed.status_code == 502
        assert "distillation failed" in failed.json()["error"]

        # LLM recovered -> distill replaces the verbatim memory
        llm.queue(facts_response(fact("Ada lives in Berlin", categories=["location"])))
        ok = client.post(f"/api/v1/memories/{memory_id}/distill")
        assert ok.status_code == 200
        assert ok.json()["actions"][0]["event"] == "ADD"
        listed = client.get("/api/v1/memories", params={"user_id": "ada"}).json()
        assert len(listed) == 1
        assert listed[0]["content"] == "Ada lives in Berlin"
        assert not listed[0]["metadata"].get("pending_distillation")

    # without any LLM configured the endpoint says so instead of 500ing
    with TestClient(create_app(make_store())) as client:
        created = client.post(
            "/api/v1/memories", json={"content": "note", "user_id": "u", "infer": False}
        )
        memory_id = created.json()["actions"][0]["memory_id"]
        resp = client.post(f"/api/v1/memories/{memory_id}/distill")
        assert resp.status_code == 400
        assert "no LLM" in resp.json()["error"]


def test_mcp_http_url_key_auth():
    """claude.ai custom connectors cannot send headers, so /mcp accepts the
    admin key embedded in the URL: /mcp/<key> or /mcp?key=<key>."""
    initialize = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "0"},
        },
    }
    headers = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
    }
    app = create_app(make_store(api_key="sekret"))
    with TestClient(app) as client:
        post = lambda path, **kw: client.post(path, json=initialize, headers={**headers, **kw.pop("extra", {})})
        assert post("/mcp").status_code == 401
        assert post("/mcp/wrong-key").status_code == 401
        assert post("/mcp?key=wrong-key").status_code == 401
        assert post("/mcp/sekret").status_code == 200
        assert post("/mcp?key=sekret").status_code == 200
        assert post("/mcp", extra={"Authorization": "Bearer sekret"}).status_code == 200
        # /mcp must answer directly, never 307 to /mcp/: clients drop the
        # Authorization header across a redirect and then see a 401.
        bare = client.post(
            "/mcp",
            json=initialize,
            headers={**headers, "Authorization": "Bearer sekret"},
            follow_redirects=False,
        )
        assert bare.status_code == 200


# ------------------------------------------------- upkeep transparency
def test_maintenance_status_lists_every_automatic_pass(client):
    """Background work that rewrites memories must be inspectable."""
    info = client.get("/api/v1/maintenance?user_id=u").json()
    passes = {p["key"]: p for p in info["passes"]}
    assert set(passes) == {"dedup_entities", "tag_abstraction", "consolidation"}
    # tag abstraction is off unless configured, and needs an LLM
    assert passes["tag_abstraction"]["automatic"] is False
    assert passes["tag_abstraction"]["needs_llm"] is True
    # consolidation never runs on its own
    assert passes["consolidation"]["automatic"] is False
    assert info["embedding_model"].startswith("hash:")


def test_consolidate_defaults_to_a_dry_run(client):
    client.post("/api/v1/memories",
                json={"content": "User is Marcus Vandenberg", "user_id": "u", "infer": False})
    client.post("/api/v1/memories",
                json={"content": "The user's name is Marc.", "user_id": "u", "infer": False})
    before = client.get("/api/v1/memories?user_id=u").json()

    preview = client.post("/api/v1/maintenance/consolidate",
                          json={"user_id": "u", "threshold": 0.2}).json()
    assert preview["merged"] == 0  # dry run by default
    after = client.get("/api/v1/memories?user_id=u").json()
    assert len(after) == len(before)


def test_maintenance_reports_tag_health(client):
    """Fragmentation must be visible without anyone going looking for it."""
    for text in ("liver enzyme panel high", "liver enzyme panel repeated",
                 "liver enzyme panel reviewed"):
        client.post("/api/v1/memories", json={
            "content": text, "user_id": "u", "infer": False,
            "categories": ["liver lab results"]})
    for text in ("liver enzyme panel high again", "liver enzyme panel once more"):
        client.post("/api/v1/memories", json={
            "content": text, "user_id": "u", "infer": False,
            "categories": ["liver bloods"]})

    health = client.get("/api/v1/maintenance?user_id=u").json()["tag_health"]
    assert health["memories"] == 5
    assert health["tags"] == 2
    assert health["untagged"] == 0
    assert {row["tag"] for row in health["largest_tags"]} == {
        "liver lab results", "liver bloods"}


# ------------------------------------------------- forgotten memories
def test_deleted_memories_land_in_forgotten_and_purge_needs_two_steps(client):
    """Delete is reversible-ish (the record survives); purge is not.

    Permanent deletion is therefore gated on the memory already being
    forgotten, so an irreversible delete takes two decisions rather than one.
    """
    keep = client.post("/api/v1/memories", json={
        "content": "keep me", "user_id": "u", "infer": False,
    }).json()["actions"][0]["memory_id"]
    gone = client.post("/api/v1/memories", json={
        "content": "delete me", "user_id": "u", "infer": False,
    }).json()["actions"][0]["memory_id"]

    # an active memory cannot be purged
    assert client.post(f"/api/v1/memories/{gone}/purge").status_code == 409

    client.delete(f"/api/v1/memories/{gone}")
    rows = client.get("/api/v1/memories/forgotten?user_id=u").json()
    assert [r["memory"]["content"] for r in rows] == ["delete me"]
    assert rows[0]["actor"] == "user"
    assert rows[0]["reason"]

    assert client.post(f"/api/v1/memories/{gone}/purge").json() == {"purged": True}
    assert client.get("/api/v1/memories/forgotten?user_id=u").json() == []
    assert [m["id"] for m in client.get("/api/v1/memories?user_id=u").json()] == [keep]


def test_superseded_memories_are_not_treated_as_forgotten():
    """A replaced memory belongs to its successor's history, not to deletions.

    Reconciliation, consolidation and distillation all invalidate the old row
    but set superseded_by. Only records with nothing standing in for them are
    forgotten.
    """
    store = make_store()
    with TestClient(create_app(store)) as client:
        old_id = client.post("/api/v1/memories", json={
            "content": "old version", "user_id": "u", "infer": False,
        }).json()["actions"][0]["memory_id"]
        new_id = client.post("/api/v1/memories", json={
            "content": "new version", "user_id": "u", "infer": False,
        }).json()["actions"][0]["memory_id"]
        # exactly what reconcile.py does when a fact contradicts an older one
        store.backend.invalidate_memory(old_id, superseded_by=new_id)

        assert client.get("/api/v1/memories/forgotten?user_id=u").json() == []
        # and a plain delete of the survivor does show up
        client.delete(f"/api/v1/memories/{new_id}")
        rows = client.get("/api/v1/memories/forgotten?user_id=u").json()
        assert [r["memory"]["id"] for r in rows] == [new_id]


def test_forgotten_is_namespaced(client):
    a = client.post("/api/v1/memories", json={
        "content": "ada secret", "user_id": "ada", "infer": False,
    }).json()["actions"][0]["memory_id"]
    client.delete(f"/api/v1/memories/{a}")
    assert client.get("/api/v1/memories/forgotten?user_id=bob").json() == []
    assert len(client.get("/api/v1/memories/forgotten?user_id=ada").json()) == 1
