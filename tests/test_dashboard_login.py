"""Dashboard session login: cookie auth for humans, bearer path untouched.

The dashboard used to prompt for a raw API key and stash it in localStorage.
Now a human logs in once at /login and rides a session cookie; these tests pin
that the cookie authenticates, is scoped to the account, and that programmatic
bearer access is unchanged.
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from memry.accounts import AccountStore
from memry.config import Config
from memry.providers.embeddings import HashEmbedder
from memry.providers.llm import NoneLLM
from memry.rest import SESSION_COOKIE, create_app
from memry.store import MemoryStore


@pytest.fixture
def app_ctx():
    accounts = AccountStore(":memory:")
    accounts.create("alice", password="hunter2")
    accounts.create("bob", password="bobpw")
    cfg = Config(db_path=":memory:", api_key="admin-key")
    store = MemoryStore(cfg, llm=NoneLLM(), embedder=HashEmbedder(64))
    app = create_app(store, accounts=accounts)
    # no auto-redirect following, so we can assert on 302s
    with TestClient(app, follow_redirects=False) as client:
        yield client, store, accounts


def login(client, **form):
    return client.post("/login", data=form)


# ---------------------------------------------------------------- page gating
def test_dashboard_redirects_to_login_when_unauthenticated(app_ctx):
    client, _, _ = app_ctx
    resp = client.get("/")
    assert resp.status_code == 302
    assert resp.headers["location"] == "/login"
    assert client.get("/login").status_code == 200


def test_open_mode_dashboard_needs_no_login():
    accounts = AccountStore(":memory:")
    cfg = Config(db_path=":memory:")  # no key, no tenants, no accounts
    store = MemoryStore(cfg, llm=NoneLLM(), embedder=HashEmbedder(64))
    with TestClient(create_app(store, accounts=accounts)) as client:
        assert client.get("/").status_code == 200
    accounts.close()


# ---------------------------------------------------------------- account login
def test_first_account_login_sets_cookie_and_uses_admin_view(app_ctx):
    client, store, _ = app_ctx
    store.add("owner note", user_id="default", infer=False)
    store.add("bob note", user_id="bob::default", infer=False)

    resp = login(client, account="alice", password="hunter2")
    assert resp.status_code == 302 and resp.headers["location"] == "/"
    assert SESSION_COOKIE in resp.cookies

    # The first account is the named administrator and sees the existing default
    # namespace as well as later members' namespaces.
    listed = client.get("/api/v1/memories").json()
    assert {memory["content"] for memory in listed} == {"owner note", "bob note"}
    page = client.get("/")
    assert page.status_code == 200 and "@alice" in page.text


def test_later_account_login_remains_scoped(app_ctx):
    client, store, _ = app_ctx
    store.add("owner note", user_id="default", infer=False)
    store.add("bob note", user_id="bob::default", infer=False)

    resp = login(client, account="bob", password="bobpw")
    assert resp.status_code == 302
    listed = client.get("/api/v1/memories").json()
    assert [memory["content"] for memory in listed] == ["bob note"]


def test_wrong_password_does_not_log_in(app_ctx):
    client, _, _ = app_ctx
    resp = login(client, account="alice", password="nope")
    assert resp.status_code == 401
    assert "Wrong account or password" in resp.text
    assert SESSION_COOKIE not in resp.cookies


def test_unknown_account_is_indistinguishable_from_wrong_password(app_ctx):
    client, _, _ = app_ctx
    resp = login(client, account="ghost", password="whatever")
    assert resp.status_code == 401
    assert "Wrong account or password" in resp.text


def test_disabled_account_cannot_log_in(app_ctx):
    client, _, accounts = app_ctx
    accounts.set_disabled("alice", True)
    resp = login(client, account="alice", password="hunter2")
    assert resp.status_code == 401


def test_disabling_account_kills_an_existing_session(app_ctx):
    client, _, accounts = app_ctx
    login(client, account="alice", password="hunter2")
    assert client.get("/api/v1/memories").status_code == 200
    accounts.set_disabled("alice", True)
    # the live cookie stops working immediately
    assert client.get("/api/v1/memories").status_code == 401


# ---------------------------------------------------------------- admin login
def test_admin_login_with_api_key(app_ctx):
    client, store, _ = app_ctx
    store.add("alice note", user_id="alice::default", infer=False)
    store.add("bob note", user_id="bob::default", infer=False)

    assert login(client, admin_key="wrong").status_code == 401
    resp = login(client, admin_key="admin-key")
    assert resp.status_code == 302
    # admin session sees every namespace
    listed = client.get("/api/v1/memories").json()
    assert {m["content"] for m in listed} == {"alice note", "bob note"}
    assert "@admin" in client.get("/").text


# ---------------------------------------------------------------- logout
def test_logout_clears_the_session(app_ctx):
    client, _, _ = app_ctx
    login(client, account="alice", password="hunter2")
    assert client.get("/api/v1/memories").status_code == 200
    out = client.get("/logout")
    assert out.status_code == 302 and out.headers["location"] == "/login"
    assert client.get("/api/v1/memories").status_code == 401


# ---------------------------------------------------------------- bearer intact
def test_bearer_tokens_still_work_without_a_session(app_ctx):
    """Programmatic clients never see the login page."""
    client, _, accounts = app_ctx
    alice_key = accounts.issue_key("alice")
    assert client.get(
        "/api/v1/stats", headers={"Authorization": f"Bearer {alice_key}"}
    ).status_code == 200
    assert client.get(
        "/api/v1/stats", headers={"Authorization": "Bearer admin-key"}
    ).status_code == 200
    assert client.get("/api/v1/stats").status_code == 401
