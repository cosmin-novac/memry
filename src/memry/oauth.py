"""OAuth 2.1 authorization server for Memry.

Why Memry is its own authorization server rather than delegating to an IdP:
MCP clients register themselves dynamically (RFC 7591), and the providers most
self-hosters already have (GitHub, Google) do not support that. Delegating
would mean every self-hoster signs up for a commercial IdP before multiuser
works at all, which is the opposite of the point. So Memry issues its own
tokens against its own accounts (see memry.accounts).

The protocol machinery - PKCE verification, redirect_uri matching, code expiry,
client authentication, and both metadata documents - is provided by the MCP
SDK's handlers. What lives here is persistence plus the human step: a login and
consent page that turns "this client wants access" into an account-bound
authorization code.

Route placement matters. The SDK's FastMCP would mount these under the MCP app,
which Memry mounts at /mcp, putting the metadata at /mcp/.well-known/... where
no client looks. memry.rest therefore builds these routes at the domain root.
"""

from __future__ import annotations

import json
import secrets
import time
from typing import Any
from urllib.parse import urlencode

from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    OAuthAuthorizationServerProvider,
    RefreshToken,
    TokenError,
    construct_redirect_uri,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

from .accounts import AccountStore
from .models import new_id, utcnow

MEMRY_SCOPE = "memry"
AUTH_CODE_TTL = 300  # 5 minutes, per OAuth guidance for codes
ACCESS_TOKEN_TTL = 3600
REFRESH_TOKEN_TTL = 60 * 60 * 24 * 30

_SCHEMA = """
CREATE TABLE IF NOT EXISTS oauth_clients (
    client_id    TEXT PRIMARY KEY,
    info         TEXT NOT NULL,
    created_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS oauth_pending (
    id           TEXT PRIMARY KEY,
    client_id    TEXT NOT NULL,
    params       TEXT NOT NULL,
    created_at   REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS oauth_codes (
    code         TEXT PRIMARY KEY,
    payload      TEXT NOT NULL,
    expires_at   REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS oauth_tokens (
    token        TEXT PRIMARY KEY,
    kind         TEXT NOT NULL,          -- 'access' | 'refresh'
    client_id    TEXT NOT NULL,
    subject      TEXT NOT NULL,          -- account name; the namespace owner
    scopes       TEXT NOT NULL,
    resource     TEXT,
    grant_id     TEXT NOT NULL,          -- ties an access/refresh pair together
    expires_at   INTEGER
);
CREATE INDEX IF NOT EXISTS idx_oauth_tokens_grant ON oauth_tokens(grant_id);
"""


class MemryOAuthProvider(
    OAuthAuthorizationServerProvider[AuthorizationCode, RefreshToken, AccessToken]
):
    """Authorization server backed by the auth SQLite database.

    Shares the AccountStore's connection and lock: one auth database, one
    writer, no second file to back up or lose.
    """

    def __init__(self, accounts: AccountStore, *, public_url: str) -> None:
        self.accounts = accounts
        self.public_url = public_url.rstrip("/")
        with accounts.lock:
            accounts.db.executescript(_SCHEMA)
            accounts.db.commit()

    # -- small helpers --------------------------------------------------
    def _write(self, sql: str, params: tuple) -> None:
        with self.accounts.lock:
            self.accounts.db.execute(sql, params)
            self.accounts.db.commit()

    def _read(self, sql: str, params: tuple):
        with self.accounts.lock:
            return self.accounts.db.execute(sql, params).fetchone()

    # -- clients (dynamic registration) ---------------------------------
    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        row = self._read(
            "SELECT info FROM oauth_clients WHERE client_id = ?", (client_id,)
        )
        if row is None:
            return None
        return OAuthClientInformationFull.model_validate_json(row["info"])

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        self._write(
            "INSERT OR REPLACE INTO oauth_clients (client_id, info, created_at) "
            "VALUES (?, ?, ?)",
            (client_info.client_id, client_info.model_dump_json(), utcnow()),
        )

    # -- authorization --------------------------------------------------
    async def authorize(
        self, client: OAuthClientInformationFull, params: AuthorizationParams
    ) -> str:
        """Park the request and send the browser to Memry's login page.

        The client never sees this id; it is a server-side handle so the
        request's redirect_uri and PKCE challenge cannot be tampered with
        between the login form and the code we eventually issue.
        """
        request_id = new_id()
        self._write(
            "INSERT INTO oauth_pending (id, client_id, params, created_at) "
            "VALUES (?, ?, ?, ?)",
            (
                request_id,
                client.client_id,
                params.model_dump_json(),
                time.time(),
            ),
        )
        return f"{self.public_url}/oauth/login?{urlencode({'request': request_id})}"

    def load_pending(self, request_id: str) -> tuple[str, AuthorizationParams] | None:
        row = self._read(
            "SELECT client_id, params FROM oauth_pending WHERE id = ?", (request_id,)
        )
        if row is None:
            return None
        return row["client_id"], AuthorizationParams.model_validate_json(row["params"])

    def complete_authorization(self, request_id: str, account_name: str) -> str | None:
        """Turn an approved login into a code and return the client redirect.

        Called by the login handler once credentials check out. The pending
        request is consumed here so a replayed form post cannot mint a second
        code.
        """
        pending = self.load_pending(request_id)
        if pending is None:
            return None
        client_id, params = pending
        self._write("DELETE FROM oauth_pending WHERE id = ?", (request_id,))

        code = secrets.token_urlsafe(32)  # >160 bits, per the provider docs
        payload = AuthorizationCode(
            code=code,
            scopes=params.scopes or [MEMRY_SCOPE],
            expires_at=time.time() + AUTH_CODE_TTL,
            client_id=client_id,
            code_challenge=params.code_challenge,
            redirect_uri=params.redirect_uri,
            redirect_uri_provided_explicitly=params.redirect_uri_provided_explicitly,
            resource=params.resource,
            subject=account_name,
        )
        self._write(
            "INSERT INTO oauth_codes (code, payload, expires_at) VALUES (?, ?, ?)",
            (code, payload.model_dump_json(), payload.expires_at),
        )
        return construct_redirect_uri(
            str(params.redirect_uri), code=code, state=params.state
        )

    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> AuthorizationCode | None:
        row = self._read(
            "SELECT payload FROM oauth_codes WHERE code = ?", (authorization_code,)
        )
        if row is None:
            return None
        code = AuthorizationCode.model_validate_json(row["payload"])
        if code.client_id != client.client_id:
            return None
        return code

    async def exchange_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: AuthorizationCode
    ) -> OAuthToken:
        # single use: burn it before issuing anything
        self._write(
            "DELETE FROM oauth_codes WHERE code = ?", (authorization_code.code,)
        )
        if authorization_code.subject is None:  # pragma: no cover - defensive
            raise TokenError("invalid_grant", "authorization code has no subject")
        if self._account_unusable(authorization_code.subject):
            raise TokenError("invalid_grant", "account is disabled")
        return self._issue_grant(
            client_id=client.client_id,
            subject=authorization_code.subject,
            scopes=authorization_code.scopes,
            resource=authorization_code.resource,
        )

    # -- refresh ---------------------------------------------------------
    async def load_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: str
    ) -> RefreshToken | None:
        row = self._read(
            "SELECT * FROM oauth_tokens WHERE token = ? AND kind = 'refresh'",
            (refresh_token,),
        )
        if row is None or row["client_id"] != client.client_id:
            return None
        if row["expires_at"] and row["expires_at"] < time.time():
            return None
        return RefreshToken(
            token=row["token"],
            client_id=row["client_id"],
            scopes=json.loads(row["scopes"]),
            expires_at=row["expires_at"],
            subject=row["subject"],
        )

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: RefreshToken,
        scopes: list[str],
    ) -> OAuthToken:
        if self._account_unusable(refresh_token.subject):
            raise TokenError("invalid_grant", "account is disabled")
        # rotate: the whole grant goes, both halves replaced
        self._revoke_grant_of(refresh_token.token)
        return self._issue_grant(
            client_id=client.client_id,
            subject=refresh_token.subject or "",
            scopes=scopes or refresh_token.scopes,
            resource=None,
        )

    # -- access tokens ---------------------------------------------------
    async def load_access_token(self, token: str) -> AccessToken | None:
        return self.verify_access_token(token)

    def verify_access_token(self, token: str) -> AccessToken | None:
        """Synchronous twin of load_access_token, for the request guard."""
        row = self._read(
            "SELECT * FROM oauth_tokens WHERE token = ? AND kind = 'access'", (token,)
        )
        if row is None:
            return None
        if row["expires_at"] and row["expires_at"] < time.time():
            return None
        if self._account_unusable(row["subject"]):
            # disabling an account has to take effect immediately, not at the
            # next token expiry
            return None
        return AccessToken(
            token=row["token"],
            client_id=row["client_id"],
            scopes=json.loads(row["scopes"]),
            expires_at=row["expires_at"],
            resource=row["resource"],
            subject=row["subject"],
        )

    async def revoke_token(self, token: AccessToken | RefreshToken) -> None:
        self._revoke_grant_of(token.token)

    # -- internals -------------------------------------------------------
    def _account_unusable(self, subject: str | None) -> bool:
        if not subject:
            return True
        account = self.accounts.get_by_name(subject)
        return account is None or account.disabled

    def _issue_grant(
        self, *, client_id: str, subject: str, scopes: list[str], resource: str | None
    ) -> OAuthToken:
        grant_id = new_id()
        access = secrets.token_urlsafe(32)
        refresh = secrets.token_urlsafe(32)
        now = int(time.time())
        with self.accounts.lock:
            self.accounts.db.execute(
                "INSERT INTO oauth_tokens (token, kind, client_id, subject, scopes, "
                "resource, grant_id, expires_at) VALUES (?, 'access', ?, ?, ?, ?, ?, ?)",
                (access, client_id, subject, json.dumps(scopes), resource, grant_id,
                 now + ACCESS_TOKEN_TTL),
            )
            self.accounts.db.execute(
                "INSERT INTO oauth_tokens (token, kind, client_id, subject, scopes, "
                "resource, grant_id, expires_at) VALUES (?, 'refresh', ?, ?, ?, ?, ?, ?)",
                (refresh, client_id, subject, json.dumps(scopes), resource, grant_id,
                 now + REFRESH_TOKEN_TTL),
            )
            self.accounts.db.commit()
        return OAuthToken(
            access_token=access,
            token_type="Bearer",
            expires_in=ACCESS_TOKEN_TTL,
            scope=" ".join(scopes),
            refresh_token=refresh,
        )

    def _revoke_grant_of(self, token: str) -> None:
        """Revoke both halves of the grant, whichever half was handed to us
        (RFC 7009 says implementations SHOULD do this)."""
        with self.accounts.lock:
            row = self.accounts.db.execute(
                "SELECT grant_id FROM oauth_tokens WHERE token = ?", (token,)
            ).fetchone()
            if row is None:
                return
            self.accounts.db.execute(
                "DELETE FROM oauth_tokens WHERE grant_id = ?", (row["grant_id"],)
            )
            self.accounts.db.commit()

    def purge_expired(self) -> dict[str, int]:
        """Housekeeping for the three time-bounded tables."""
        now = time.time()
        with self.accounts.lock:
            codes = self.accounts.db.execute(
                "DELETE FROM oauth_codes WHERE expires_at < ?", (now,)
            ).rowcount
            tokens = self.accounts.db.execute(
                "DELETE FROM oauth_tokens WHERE expires_at IS NOT NULL "
                "AND expires_at < ?",
                (now,),
            ).rowcount
            pending = self.accounts.db.execute(
                "DELETE FROM oauth_pending WHERE created_at < ?", (now - 900,)
            ).rowcount
            self.accounts.db.commit()
        return {"codes": codes, "tokens": tokens, "pending": pending}

    def tokens_for(self, subject: str) -> list[dict[str, Any]]:
        with self.accounts.lock:
            rows = self.accounts.db.execute(
                "SELECT client_id, kind, expires_at FROM oauth_tokens "
                "WHERE subject = ? ORDER BY expires_at",
                (subject,),
            ).fetchall()
        return [dict(r) for r in rows]
