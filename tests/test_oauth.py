"""OAuth authorization server, driven end to end over its real HTTP endpoints.

Exercises the full dance an MCP client performs - discovery, dynamic client
registration, PKCE authorize, code exchange, refresh, revoke - because that is
exactly what failed in the field: VS Code could not discover the server at all.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
from urllib.parse import parse_qs, urlparse

import pytest
from starlette.testclient import TestClient

from conftest import mcp_call
from memry.accounts import AccountStore
from memry.config import Config
from memry.providers.embeddings import HashEmbedder
from memry.providers.llm import NoneLLM
from memry.rest import create_app
from memry.store import MemoryStore

PUBLIC_URL = "https://memory.example.test"


@pytest.fixture
def oauth_client():
    accounts = AccountStore(":memory:")
    accounts.create("alice", password="hunter2")
    cfg = Config(db_path=":memory:", public_url=PUBLIC_URL)
    store = MemoryStore(cfg, llm=NoneLLM(), embedder=HashEmbedder(64))
    app = create_app(store, accounts=accounts)
    # base_url matches PUBLIC_URL so redirect_uri and issuer line up
    with TestClient(app, base_url=PUBLIC_URL) as client:
        yield client, store, accounts
    accounts.close()


def _pkce() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()
    ).decode().rstrip("=")
    return verifier, challenge


def _register(client) -> dict:
    resp = client.post("/register", json={
        "redirect_uris": ["https://client.example/callback"],
        "client_name": "Test IDE",
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "scope": "memry",
        "token_endpoint_auth_method": "client_secret_post",
    })
    assert resp.status_code == 201, resp.text
    return resp.json()


def _authorize_and_login(client, reg, challenge, *, account="alice", password="hunter2"):
    """Run /authorize then the login form; return the callback query params."""
    auth = client.get("/authorize", params={
        "client_id": reg["client_id"],
        "redirect_uri": "https://client.example/callback",
        "response_type": "code",
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "scope": "memry",
        "state": "xyz",
    }, follow_redirects=False)
    assert auth.status_code in (302, 307), auth.text
    login_url = auth.headers["location"]
    request_id = parse_qs(urlparse(login_url).query)["request"][0]

    # the login page renders
    page = client.get("/oauth/login", params={"request": request_id})
    assert page.status_code == 200 and "Test IDE" in page.text

    submit = client.post("/oauth/login", data={
        "request": request_id,
        "account": account,
        "password": password,
        "decision": "approve",
    }, follow_redirects=False)
    assert submit.status_code == 302, submit.text
    return parse_qs(urlparse(submit.headers["location"]).query)


# ---------------------------------------------------------------- discovery
def test_metadata_documents_live_at_the_domain_root(oauth_client):
    client, _, _ = oauth_client
    # this is the document VS Code failed to find in the original bug
    prm = client.get("/.well-known/oauth-protected-resource/mcp")
    assert prm.status_code == 200
    assert prm.json()["authorization_servers"] == [PUBLIC_URL + "/"]

    asm = client.get("/.well-known/oauth-authorization-server")
    assert asm.status_code == 200
    meta = asm.json()
    assert meta["authorization_endpoint"] == f"{PUBLIC_URL}/authorize"
    assert meta["token_endpoint"] == f"{PUBLIC_URL}/token"
    assert meta["registration_endpoint"] == f"{PUBLIC_URL}/register"


def test_unauthorized_mcp_points_at_the_resource_metadata(oauth_client):
    client, _, _ = oauth_client
    body = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                       "clientInfo": {"name": "t", "version": "0"}}}
    resp = client.post("/mcp/", json=body, headers={
        "accept": "application/json, text/event-stream"})
    assert resp.status_code == 401
    # the WWW-Authenticate header is how the client discovers where to log in
    assert "oauth-protected-resource/mcp" in resp.headers.get("WWW-Authenticate", "")


# ---------------------------------------------------------------- full flow
def test_full_authorization_code_flow_yields_a_working_token(oauth_client):
    client, store, _ = oauth_client
    reg = _register(client)
    verifier, challenge = _pkce()
    callback = _authorize_and_login(client, reg, challenge)
    assert callback["state"] == ["xyz"]
    code = callback["code"][0]

    token = client.post("/token", data={
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": "https://client.example/callback",
        "client_id": reg["client_id"],
        "client_secret": reg["client_secret"],
        "code_verifier": verifier,
    })
    assert token.status_code == 200, token.text
    access = token.json()["access_token"]

    # Alice is the first account, so OAuth grants her only the existing default
    # memory space. Her administrator role does not expose other accounts.
    mcp_call(client, access, "save_memories", {"content": "from oauth", "infer": False})
    owners = {m.content: m.user_id for m in store.get_all(limit=50)}
    assert owners == {"from oauth": "default"}


def test_pkce_is_enforced(oauth_client):
    client, _, _ = oauth_client
    reg = _register(client)
    _, challenge = _pkce()
    code = _authorize_and_login(client, reg, challenge)["code"][0]

    bad = client.post("/token", data={
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": "https://client.example/callback",
        "client_id": reg["client_id"],
        "client_secret": reg["client_secret"],
        "code_verifier": "not-the-verifier",
    })
    assert bad.status_code == 400
    assert bad.json()["error"] == "invalid_grant"


def test_authorization_code_is_single_use(oauth_client):
    client, _, _ = oauth_client
    reg = _register(client)
    verifier, challenge = _pkce()
    code = _authorize_and_login(client, reg, challenge)["code"][0]
    form = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": "https://client.example/callback",
        "client_id": reg["client_id"],
        "client_secret": reg["client_secret"],
        "code_verifier": verifier,
    }
    assert client.post("/token", data=form).status_code == 200
    assert client.post("/token", data=form).status_code == 400


def test_wrong_password_does_not_issue_a_code(oauth_client):
    client, _, _ = oauth_client
    reg = _register(client)
    _, challenge = _pkce()
    auth = client.get("/authorize", params={
        "client_id": reg["client_id"],
        "redirect_uri": "https://client.example/callback",
        "response_type": "code",
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "scope": "memry",
        "state": "xyz",
    }, follow_redirects=False)
    request_id = parse_qs(urlparse(auth.headers["location"]).query)["request"][0]
    submit = client.post("/oauth/login", data={
        "request": request_id, "account": "alice", "password": "WRONG",
        "decision": "approve",
    }, follow_redirects=False)
    assert submit.status_code == 200  # back to the form, no redirect
    assert "Wrong account or password" in submit.text


def test_deny_redirects_with_access_denied(oauth_client):
    client, _, _ = oauth_client
    reg = _register(client)
    _, challenge = _pkce()
    auth = client.get("/authorize", params={
        "client_id": reg["client_id"],
        "redirect_uri": "https://client.example/callback",
        "response_type": "code",
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "scope": "memry", "state": "xyz",
    }, follow_redirects=False)
    request_id = parse_qs(urlparse(auth.headers["location"]).query)["request"][0]
    submit = client.post("/oauth/login", data={
        "request": request_id, "account": "alice", "password": "hunter2",
        "decision": "deny",
    }, follow_redirects=False)
    assert submit.status_code == 302
    params = parse_qs(urlparse(submit.headers["location"]).query)
    assert params["error"] == ["access_denied"]


def test_refresh_rotates_and_old_token_dies(oauth_client):
    client, _, _ = oauth_client
    reg = _register(client)
    verifier, challenge = _pkce()
    code = _authorize_and_login(client, reg, challenge)["code"][0]
    first = client.post("/token", data={
        "grant_type": "authorization_code", "code": code,
        "redirect_uri": "https://client.example/callback",
        "client_id": reg["client_id"], "client_secret": reg["client_secret"],
        "code_verifier": verifier,
    }).json()

    refreshed = client.post("/token", data={
        "grant_type": "refresh_token",
        "refresh_token": first["refresh_token"],
        "client_id": reg["client_id"], "client_secret": reg["client_secret"],
    })
    assert refreshed.status_code == 200, refreshed.text
    new = refreshed.json()
    assert new["access_token"] != first["access_token"]

    # the rotated-out refresh token no longer works
    again = client.post("/token", data={
        "grant_type": "refresh_token",
        "refresh_token": first["refresh_token"],
        "client_id": reg["client_id"], "client_secret": reg["client_secret"],
    })
    assert again.status_code == 400


def test_revocation_kills_the_token(oauth_client):
    client, store, _ = oauth_client
    reg = _register(client)
    verifier, challenge = _pkce()
    code = _authorize_and_login(client, reg, challenge)["code"][0]
    tok = client.post("/token", data={
        "grant_type": "authorization_code", "code": code,
        "redirect_uri": "https://client.example/callback",
        "client_id": reg["client_id"], "client_secret": reg["client_secret"],
        "code_verifier": verifier,
    }).json()
    access = tok["access_token"]

    # confirm it works, then revoke, then confirm it stops
    mcp_call(client, access, "list_memories", {})
    revoke = client.post("/revoke", data={
        "token": access,
        "client_id": reg["client_id"], "client_secret": reg["client_secret"],
    })
    assert revoke.status_code == 200

    body = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                       "clientInfo": {"name": "t", "version": "0"}}}
    resp = client.post("/mcp/", json=body, headers={
        "accept": "application/json, text/event-stream",
        "Authorization": f"Bearer {access}"})
    assert resp.status_code == 401


def test_disabled_account_token_stops_working(oauth_client):
    client, _, accounts = oauth_client
    reg = _register(client)
    verifier, challenge = _pkce()
    code = _authorize_and_login(client, reg, challenge)["code"][0]
    access = client.post("/token", data={
        "grant_type": "authorization_code", "code": code,
        "redirect_uri": "https://client.example/callback",
        "client_id": reg["client_id"], "client_secret": reg["client_secret"],
        "code_verifier": verifier,
    }).json()["access_token"]

    # disabling the account must take effect immediately, not at token expiry
    accounts.set_disabled("alice", True)
    body = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                       "clientInfo": {"name": "t", "version": "0"}}}
    resp = client.post("/mcp/", json=body, headers={
        "accept": "application/json, text/event-stream",
        "Authorization": f"Bearer {access}"})
    assert resp.status_code == 401
