"""Accounts: credential handling and end-to-end confinement.

The credential tests matter as much as the isolation ones: an account is only
as isolated as the key that proves who it is.
"""

from __future__ import annotations

import json

import pytest
from conftest import mcp_call
from starlette.testclient import TestClient

from memry.accounts import AccountStore, default_auth_db_path, hash_key
from memry.config import Config, TenantConfig
from memry.providers.embeddings import HashEmbedder
from memry.providers.llm import NoneLLM
from memry.rest import create_app
from memry.store import MemoryStore


@pytest.fixture
def accounts() -> AccountStore:
    store = AccountStore(":memory:")
    yield store
    store.close()


@pytest.fixture
def served(accounts):
    """A server whose only credentials are accounts (no admin key, no tenants)."""
    cfg = Config(db_path=":memory:", api_key="admin-key")
    store = MemoryStore(cfg, llm=NoneLLM(), embedder=HashEmbedder(64))
    alice_key = accounts.issue_key(accounts.create("alice").name)
    bob_key = accounts.issue_key(accounts.create("bob").name)
    app = create_app(store, accounts=accounts)
    with TestClient(app, base_url="http://127.0.0.1:8787") as client:
        yield client, store, alice_key, bob_key


# ---------------------------------------------------------------- credentials
def test_api_keys_are_stored_hashed_only(accounts):
    accounts.create("alice")
    key = accounts.issue_key("alice")

    rows = accounts._db.execute("SELECT key_hash FROM account_keys").fetchall()
    assert [r["key_hash"] for r in rows] == [hash_key(key)]
    # the key itself appears nowhere in the database
    dump = "\n".join(accounts._db.iterdump())
    assert key not in dump

    assert accounts.account_for_key(key).name == "alice"
    assert accounts.account_for_key(key + "x") is None
    assert accounts.account_for_key("") is None


def test_passwords_are_salted_and_verified(accounts):
    accounts.create("alice", password="correct horse")
    alice = accounts.get_by_name("alice")
    assert alice.check_password("correct horse")
    assert not alice.check_password("Correct Horse")
    assert not alice.check_password("")

    # same password, different account -> different stored hash (unique salt)
    accounts.create("bob", password="correct horse")
    stored = accounts._db.execute("SELECT password FROM accounts").fetchall()
    assert stored[0]["password"] != stored[1]["password"]


def test_account_without_password_cannot_be_logged_into(accounts):
    accounts.create("service")
    account = accounts.get_by_name("service")
    assert account.has_password is False
    assert account.check_password("") is False
    assert account.check_password("anything") is False


def test_duplicate_and_malformed_names_are_refused(accounts):
    accounts.create("alice")
    with pytest.raises(ValueError, match="already exists"):
        accounts.create("alice")
    with pytest.raises(ValueError, match="must not be empty"):
        accounts.create("  ")
    # "::" is the namespace separator: allowing it would let one account's
    # space nest inside another's
    with pytest.raises(ValueError, match="::"):
        accounts.create("alice::bob")


def test_disabling_an_account_kills_its_keys(accounts):
    accounts.create("alice")
    key = accounts.issue_key("alice")
    assert accounts.account_for_key(key) is not None

    accounts.set_disabled("alice", True)
    assert accounts.account_for_key(key) is None
    accounts.set_disabled("alice", False)
    assert accounts.account_for_key(key) is not None

    assert accounts.revoke_keys("alice") == 1
    assert accounts.account_for_key(key) is None


def test_auth_db_sits_next_to_the_memory_db(tmp_path):
    assert default_auth_db_path(str(tmp_path / "memry.db")) == str(tmp_path / "auth.db")
    assert default_auth_db_path(":memory:") == ":memory:"


# ---------------------------------------------------------------- confinement
def test_account_key_authenticates_rest_and_is_namespaced(served):
    client, store, alice_key, bob_key = served
    alice = {"Authorization": f"Bearer {alice_key}"}
    bob = {"Authorization": f"Bearer {bob_key}"}

    client.post("/api/v1/memories", headers=alice,
                json={"content": "alice note", "infer": False})
    client.post("/api/v1/memories", headers=bob,
                json={"content": "bob note", "infer": False})

    owners = {m.content: m.user_id for m in store.get_all(limit=50)}
    assert owners == {"alice note": "alice::default", "bob note": "bob::default"}

    listed = client.get("/api/v1/memories", headers=alice).json()
    assert [m["content"] for m in listed] == ["alice note"]

    assert client.get("/api/v1/memories", headers={
        "Authorization": "Bearer mk_not-a-real-key"}).status_code == 401


def test_account_key_works_over_mcp_and_is_confined(served):
    client, store, alice_key, bob_key = served
    mcp_call(client, alice_key, "save_memories",
             {"content": "alice secret", "infer": False})
    victim = next(m for m in store.get_all(limit=50) if m.content == "alice secret")
    assert victim.user_id == "alice::default"

    assert json.loads(mcp_call(client, bob_key, "list_memories", {})) == []
    hijack = json.loads(mcp_call(client, bob_key, "update_memory",
                                 {"memory_id": victim.id, "content": "hijacked"}))
    assert "error" in hijack
    assert store.get(victim.id).content == "alice secret"


def test_admin_key_still_sees_everything(served):
    client, store, alice_key, _ = served
    mcp_call(client, alice_key, "save_memories",
             {"content": "alice note", "infer": False})
    memory = store.get_all(limit=50)[0]
    admin = {"Authorization": "Bearer admin-key"}
    assert client.get(f"/api/v1/memories/{memory.id}", headers=admin).status_code == 200


def test_accounts_and_config_tenants_coexist():
    """A deployment can use both; each is confined to its own namespace."""
    accounts = AccountStore(":memory:")
    cfg = Config(
        db_path=":memory:",
        api_key="admin-key",
        tenants=[TenantConfig(name="acme", api_key="acme-key")],
    )
    store = MemoryStore(cfg, llm=NoneLLM(), embedder=HashEmbedder(64))
    accounts.create("alice")
    alice_key = accounts.issue_key("alice")
    app = create_app(store, accounts=accounts)
    try:
        with TestClient(app, base_url="http://127.0.0.1:8787") as client:
            client.post("/api/v1/memories",
                        headers={"Authorization": "Bearer acme-key"},
                        json={"content": "acme note", "infer": False})
            client.post("/api/v1/memories",
                        headers={"Authorization": f"Bearer {alice_key}"},
                        json={"content": "alice note", "infer": False})
            owners = {m.content: m.user_id for m in store.get_all(limit=50)}
            assert owners == {
                "acme note": "acme::default",
                "alice note": "alice::default",
            }
    finally:
        accounts.close()


def test_accounts_alone_close_an_otherwise_open_server():
    """Creating the first account must not leave the server open."""
    accounts = AccountStore(":memory:")
    cfg = Config(db_path=":memory:")  # no admin key, no tenants
    store = MemoryStore(cfg, llm=NoneLLM(), embedder=HashEmbedder(64))
    try:
        with TestClient(create_app(store, accounts=accounts)) as client:
            assert client.get("/api/v1/stats").status_code == 200  # open mode

        accounts.create("alice")
        with TestClient(create_app(store, accounts=accounts)) as client:
            assert client.get("/api/v1/stats").status_code == 401
            key = accounts.issue_key("alice")
            assert client.get(
                "/api/v1/stats", headers={"Authorization": f"Bearer {key}"}
            ).status_code == 200
    finally:
        accounts.close()
