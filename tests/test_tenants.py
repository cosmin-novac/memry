from __future__ import annotations

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


def test_mcp_http_admin_only_when_auth_configured(multi):
    client, _ = multi
    body = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                       "clientInfo": {"name": "t", "version": "0"}}}
    headers = {"accept": "application/json, text/event-stream"}
    assert client.post("/mcp/", json=body, headers=headers).status_code == 401
    assert client.post("/mcp/", json=body, headers={**headers, **ACME}).status_code == 401
    assert client.post("/mcp/", json=body, headers={**headers, **ADMIN}).status_code == 200


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
