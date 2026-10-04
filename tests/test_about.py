"""GET /api/v1/about: backups, footprint, response times and host for the
administrator; the version and its own counts for anyone else."""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from memry.about import ResponseWindow, percentile
from memry.accounts import AccountStore
from memry.config import Config, SnapshotConfig, TenantConfig
from memry.providers.embeddings import HashEmbedder
from memry.providers.llm import NoneLLM
from memry.rest import create_app
from memry.snapshot import take_snapshot
from memry.store import MemoryStore

ADMIN = {"Authorization": "Bearer admin-key"}
ACME = {"Authorization": "Bearer acme-key"}


def test_the_window_keeps_the_last_requests_and_reads_median_and_p95():
    window = ResponseWindow(size=200)
    assert window.summary() == {"count": 0, "window": 200, "median_ms": None, "p95_ms": None}
    for ms in range(1, 101):
        window.record(float(ms))
    assert window.summary() == {"count": 100, "window": 200, "median_ms": 50.0, "p95_ms": 95.0}
    for _ in range(300):
        window.record(1000.0)
    summary = window.summary()
    assert summary["count"] == 200 and summary["median_ms"] == 1000.0
    assert percentile([5.0], 0.95) == 5.0


@pytest.fixture
def server(tmp_path):
    config = Config(
        db_path=str(tmp_path / "data" / "memry.db"),
        api_key="admin-key",
        tenants=[TenantConfig(name="acme", api_key="acme-key")],
        snapshot=SnapshotConfig(dir=str(tmp_path / "backups"), host_dir="/var/backups/memry"),
    )
    store = MemoryStore(config, llm=NoneLLM(), embedder=HashEmbedder(64))
    accounts = AccountStore(str(tmp_path / "data" / "auth.db"))
    with TestClient(create_app(store, accounts=accounts)) as client:
        yield client, store, accounts
    store.close()


def test_the_administrator_sees_backups_footprint_timing_and_host(server):
    client, store, _ = server
    client.post("/api/v1/memories", json={"content": "Ada likes tea.", "user_id": "ada",
                                          "infer": False}, headers=ADMIN)
    assert take_snapshot(store.config)["ok"]
    for _ in range(5):
        client.get("/health")
    client.get("/api/v1/stats", headers=ADMIN)

    response = client.get("/api/v1/about", headers=ADMIN)
    assert response.status_code == 200
    about = response.json()
    assert about["scope"] == "server" and about["version"]

    backups = about["backups"]
    assert backups["enabled"] and backups["at"] == "03:30" and backups["timezone"]
    assert backups["host_dir"] == "/var/backups/memry" and backups["offsite"] is False
    assert backups["last_success"]["verified"] and backups["last_success"]["size"] > 0
    assert backups["last_failure"] is None

    footprint = about["footprint"]
    assert footprint["db_bytes"] > 0 and footprint["memories_in_use"] == 1
    for key in ("wal_bytes", "auth_db_bytes", "memories_history", "memories_forgotten",
                "entities", "episodes", "disk_free_bytes"):
        assert key in footprint

    timing = about["response_times"]
    # /health is left out; the requests before this one are in
    assert timing["count"] >= 2 and timing["median_ms"] is not None and timing["p95_ms"] is not None

    host = about["host"]
    assert host["hostname"] and host["python"] and host["uptime_s"] >= 0
    assert host["llm"] and host["embedder"] and "decider" in host
    assert "admin-key" not in response.text and "acme-key" not in response.text


def test_a_tenant_gets_the_version_and_its_own_counts_only(server):
    client, _, _ = server
    client.post("/api/v1/memories", json={"content": "Acme ships widgets.", "infer": False},
                headers=ACME)
    client.post("/api/v1/memories", json={"content": "Not theirs.", "user_id": "ada",
                                          "infer": False}, headers=ADMIN)
    about = client.get("/api/v1/about", headers=ACME).json()
    assert set(about) == {"version", "scope", "counts"}
    assert about["scope"] == "account"
    assert about["counts"]["memories_in_use"] == 1


def test_an_account_that_is_not_an_administrator_gets_no_host_details(server):
    client, _, accounts = server
    accounts.create("owner", password="pw-123456")  # the first account administers
    accounts.create("guest", password="pw-123456")
    owner_key = accounts.issue_key("owner")
    guest_key = accounts.issue_key("guest")
    guest = client.get("/api/v1/about", headers={"Authorization": f"Bearer {guest_key}"}).json()
    assert set(guest) == {"version", "scope", "counts"}
    owner = client.get("/api/v1/about", headers={"Authorization": f"Bearer {owner_key}"}).json()
    assert owner["scope"] == "server" and "host" in owner


def test_about_needs_a_key(server):
    client, _, _ = server
    assert client.get("/api/v1/about").status_code == 401


def test_backups_read_off_when_no_directory_is_set():
    store = MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(), embedder=HashEmbedder(64))
    try:
        with TestClient(create_app(store)) as client:
            about = client.get("/api/v1/about").json()
    finally:
        store.close()
    assert about["backups"]["enabled"] is False and about["backups"]["last_success"] is None
    assert about["footprint"]["db_bytes"] is None
