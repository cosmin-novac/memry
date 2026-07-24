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
