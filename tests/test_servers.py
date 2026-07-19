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
        "update_memory", "delete_memory", "memory_history", "memory_stats",
    } <= names


# ---------------------------------------------------------------- REST API
@pytest.fixture
def client():
    app = create_app(make_store())
    with TestClient(app) as c:
        yield c


def test_rest_health_and_dashboard(client):
    assert client.get("/health").json()["status"] == "ok"
    assert "memry" in client.get("/").text


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


def test_rest_auth_enforced():
    app = create_app(make_store(api_key="sekret"))
    with TestClient(app) as client:
        assert client.get("/api/v1/stats").status_code == 401
        ok = client.get("/api/v1/stats", headers={"Authorization": "Bearer sekret"})
        assert ok.status_code == 200
        # health and dashboard stay open
        assert client.get("/health").status_code == 200
