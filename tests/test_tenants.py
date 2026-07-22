from __future__ import annotations

import json

import pytest
from starlette.testclient import TestClient

from memry.config import Config, TenantConfig
from memry.providers.embeddings import HashEmbedder
from memry.providers.llm import NoneLLM
from memry.rest import create_app
from memry.store import MemoryStore


def make_app(**config_kwargs):
    cfg = Config(db_path=":memory:", **config_kwargs)
    store = MemoryStore(cfg, llm=NoneLLM(), embedder=HashEmbedder(64))
    return create_app(store), store


ACME = {"Authorization": "Bearer acme-key"}
GLOBEX = {"Authorization": "Bearer globex-key"}
ADMIN = {"Authorization": "Bearer admin-key"}


@pytest.fixture
def multi():
    app, store = make_app(
        api_key="admin-key",
        tenants=[
            TenantConfig(name="acme", api_key="acme-key"),
            TenantConfig(name="globex", api_key="globex-key"),
        ],
    )
    # base_url must match the host FastMCP declares, or its DNS-rebinding
    # protection rejects /mcp requests with 421
    with TestClient(app, base_url="http://127.0.0.1:8787") as client:
        yield client, store


def test_unknown_key_rejected(multi):
    client, _ = multi
    assert client.get("/api/v1/stats").status_code == 401
    assert client.get(
        "/api/v1/stats", headers={"Authorization": "Bearer wrong"}
    ).status_code == 401


def test_tenants_are_namespaced_and_isolated(multi):
    client, store = multi
    client.post("/api/v1/memories", headers=ACME,
                json={"content": "acme secret roadmap", "user_id": "u1", "infer": False})
    client.post("/api/v1/memories", headers=GLOBEX,
                json={"content": "globex pricing table", "user_id": "u1", "infer": False})

    # same user_id, different tenants -> different namespaces
    acme_list = client.get("/api/v1/memories", headers=ACME, params={"user_id": "u1"}).json()
    globex_list = client.get("/api/v1/memories", headers=GLOBEX, params={"user_id": "u1"}).json()
    assert [m["content"] for m in acme_list] == ["acme secret roadmap"]
    assert [m["content"] for m in globex_list] == ["globex pricing table"]
    assert acme_list[0]["user_id"] == "acme::u1"

    # search is isolated too
    hits = client.post("/api/v1/search", headers=GLOBEX,
                       json={"query": "roadmap secret pricing", "user_id": "u1"}).json()
    assert [h["memory"]["content"] for h in hits] == ["globex pricing table"]


def test_cross_tenant_direct_id_access_denied(multi):
    client, _ = multi
    created = client.post("/api/v1/memories", headers=ACME,
                          json={"content": "acme only", "infer": False}).json()
    memory_id = created["actions"][0]["memory_id"]

    assert client.get(f"/api/v1/memories/{memory_id}", headers=ACME).status_code == 200
    # the other tenant sees 404, not 403: existence is not leaked
    assert client.get(f"/api/v1/memories/{memory_id}", headers=GLOBEX).status_code == 404
    assert client.delete(f"/api/v1/memories/{memory_id}", headers=GLOBEX).status_code == 404
    assert client.get(f"/api/v1/memories/{memory_id}/history", headers=GLOBEX).status_code == 404
    # admin sees everything
    assert client.get(f"/api/v1/memories/{memory_id}", headers=ADMIN).status_code == 200


def test_tenant_stats_scoped(multi):
    client, _ = multi
    client.post("/api/v1/memories", headers=ACME, json={"content": "a", "infer": False})
    client.post("/api/v1/memories", headers=ACME, json={"content": "b", "infer": False})
    client.post("/api/v1/memories", headers=GLOBEX, json={"content": "c", "infer": False})

    acme_stats = client.get("/api/v1/stats", headers=ACME).json()
    assert acme_stats["tenant"] == "acme"
    assert acme_stats["active_memories"] == 2
    admin_stats = client.get("/api/v1/stats", headers=ADMIN).json()
    assert admin_stats["active_memories"] == 3


def test_mcp_http_accepts_tenant_keys(multi):
    """Tenant keys authenticate on /mcp now that tool calls are scoped."""
    client, _ = multi
    body = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                       "clientInfo": {"name": "t", "version": "0"}}}
    headers = {"accept": "application/json, text/event-stream"}
    assert client.post("/mcp/", json=body, headers=headers).status_code == 401
    assert client.post(
        "/mcp/", json=body, headers={**headers, "Authorization": "Bearer wrong"}
    ).status_code == 401
    assert client.post("/mcp/", json=body, headers={**headers, **ACME}).status_code == 200
    assert client.post("/mcp/", json=body, headers={**headers, **ADMIN}).status_code == 200
    # the URL-key form works for tenants too
    assert client.post(
        "/mcp/acme-key", json=body, headers=headers
    ).status_code == 200


def test_open_mode_unchanged():
    app, _ = make_app()
    with TestClient(app) as client:
        assert client.get("/api/v1/stats").status_code == 200


def test_rest_entities_endpoints(multi):
    client, store = multi
    # seed an entity via the python API in the acme namespace
    store.add("Jonas fact", user_id="acme::u1", infer=False)
    memory = store.get_all(user_id="acme::u1")[0]
    from memry.intelligence.entities import resolve_mentions
    from memry.models import Scope

    resolve_mentions(backend=store.backend, llm=store.llm, scope=Scope(user_id="acme::u1"),
                     memory_id=memory.id, memory_content=memory.content, surfaces=["Jonas"])

    listed = client.get("/api/v1/entities", headers=ACME, params={"user_id": "u1"}).json()
    assert len(listed) == 1
    entity_id = listed[0]["id"]

    detail = client.get(f"/api/v1/entities/{entity_id}", headers=ACME).json()
    assert detail["entity"]["name"] == "Jonas"
    assert len(detail["memories"]) == 1

    # other tenant cannot see it
    assert client.get(f"/api/v1/entities/{entity_id}", headers=GLOBEX).status_code == 404


# ---------------------------------------------------------------- MCP scoping
def _sse_json(text: str) -> dict:
    for line in text.splitlines():
        if line.startswith("data: "):
            return json.loads(line[6:])
    raise AssertionError(f"no SSE data frame in {text!r}")


def mcp_call(client, key: str, tool: str, arguments: dict) -> str:
    """Drive a real MCP session over streamable HTTP and return the tool text.

    Goes through the whole transport (initialize, initialized, tools/call) on
    purpose: the identity these tests are about is attached to the HTTP request
    and has to survive the session task boundary to reach the tool body.
    """
    headers = {
        "accept": "application/json, text/event-stream",
        "content-type": "application/json",
        "Authorization": f"Bearer {key}",
    }
    init = client.post("/mcp/", headers=headers, json={
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                   "clientInfo": {"name": "test", "version": "0"}}})
    assert init.status_code == 200, init.text
    headers["mcp-session-id"] = init.headers["mcp-session-id"]
    client.post("/mcp/", headers=headers, json={
        "jsonrpc": "2.0", "method": "notifications/initialized"})
    called = client.post("/mcp/", headers=headers, json={
        "jsonrpc": "2.0", "id": 2, "method": "tools/call",
        "params": {"name": tool, "arguments": arguments}})
    assert called.status_code == 200, called.text
    return _sse_json(called.text)["result"]["content"][0]["text"]


def test_mcp_tool_writes_land_in_the_tenant_namespace(multi):
    client, store = multi
    mcp_call(client, "acme-key", "save_memories", {"content": "acme fact", "infer": False})
    mcp_call(client, "globex-key", "save_memories", {"content": "globex fact", "infer": False})

    owners = {m.content: m.user_id for m in store.get_all(limit=50)}
    assert owners == {"acme fact": "acme::default", "globex fact": "globex::default"}

    mine = json.loads(mcp_call(client, "acme-key", "list_memories", {}))
    assert [m["content"] for m in mine] == ["acme fact"]


def test_mcp_user_id_argument_cannot_escape_the_namespace(multi):
    """The tool argument selects a sub-namespace, never someone else's."""
    client, store = multi
    mcp_call(client, "globex-key", "save_memories", {"content": "globex fact", "infer": False})

    escaped = json.loads(mcp_call(
        client, "acme-key", "list_memories", {"user_id": "globex::default"}
    ))
    assert escaped == []

    mcp_call(client, "acme-key", "save_memories",
             {"content": "planted", "infer": False, "user_id": "globex::default"})
    planted = next(m for m in store.get_all(limit=50) if m.content == "planted")
    assert planted.user_id == "acme::globex::default"


def test_mcp_id_addressed_tools_are_confined(multi):
    """The isolation matrix: every by-id tool must refuse another tenant's id."""
    client, store = multi
    mcp_call(client, "acme-key", "save_memories", {"content": "acme secret", "infer": False})
    victim = next(m for m in store.get_all(limit=50) if m.content == "acme secret")

    updated = json.loads(mcp_call(
        client, "globex-key", "update_memory",
        {"memory_id": victim.id, "content": "hijacked"}))
    assert "error" in updated
    assert store.get(victim.id).content == "acme secret"

    deleted = json.loads(mcp_call(
        client, "globex-key", "delete_memory", {"memory_id": victim.id}))
    assert deleted["deleted"] is False
    assert store.get(victim.id).invalid_at is None

    assert json.loads(mcp_call(
        client, "globex-key", "memory_history", {"memory_id": victim.id})) == []

    # the owner can still do all three
    assert json.loads(mcp_call(
        client, "acme-key", "memory_history", {"memory_id": victim.id}))
    assert json.loads(mcp_call(
        client, "acme-key", "delete_memory", {"memory_id": victim.id}))["deleted"] is True


def test_mcp_stats_do_not_leak_other_namespaces(multi):
    client, store = multi
    store.add("a", user_id="acme::default", infer=False)
    store.add("b", user_id="globex::default", infer=False)
    store.add("c", user_id="globex::default", infer=False)

    stats = json.loads(mcp_call(client, "acme-key", "memory_stats", {}))
    assert stats["tenant"] == "acme"
    assert stats["active_memories"] == 1
