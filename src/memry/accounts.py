"""Accounts: the identities a multiuser Memry server can authenticate.

Accounts currently use a separate SQLite file from the memory store. That kept
identity independent of the selectable Mem0 comparison adapter, but it also
creates a second backup and migration lifecycle. SQLite is now the sole production
store, so consolidation is an explicit follow-up decision in the duplication audit.

Two credential kinds, for two different callers:

- **API keys** for programmatic clients (MCP over HTTP, REST). Only a hash is
  stored, so a leaked database does not hand over working keys. Keys are high
  entropy, so a plain SHA-256 is the right lookup primitive - a slow KDF here
  would buy nothing and cost a hash on every request.
- **Passwords** for humans, used by the OAuth login step and the dashboard.
  These are low entropy and guessable, so they get scrypt with a per-account
  salt.

An account's name is its namespace: account ``alice`` owns ``alice::*``, the
same shape config tenants already use, so the two are one mechanism to the rest
of the server.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from .models import new_id, utcnow

_SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL UNIQUE,
    password     TEXT,
    disabled     INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL,
    metadata     TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS account_keys (
    key_hash     TEXT PRIMARY KEY,
    account_id   TEXT NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    label        TEXT,
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_account_keys_account ON account_keys(account_id);
CREATE TABLE IF NOT EXISTS dashboard_sessions (
    token_hash   TEXT PRIMARY KEY,
    account      TEXT,                 -- NULL = admin session
    created_at   TEXT NOT NULL,
    expires_at   REAL NOT NULL
);
"""

SESSION_TTL = 60 * 60 * 24 * 14  # dashboard cookies last two weeks

KEY_PREFIX = "mk_"  # so a leaked string is recognisable as a Memry key
_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1


def default_auth_db_path(db_path: str) -> str:
    """Auth DB sits next to the memory DB, so a backup of one finds the other."""
    if db_path == ":memory:":
        return ":memory:"
    return str(Path(db_path).with_name("auth.db"))


def hash_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P,
        dklen=32,
    )
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str | None) -> bool:
    if not stored:
        return False
    try:
        scheme, n, r, p, salt_hex, digest_hex = stored.split("$")
        if scheme != "scrypt":
            return False
        digest = hashlib.scrypt(
            password.encode("utf-8"), salt=bytes.fromhex(salt_hex),
            n=int(n), r=int(r), p=int(p), dklen=len(bytes.fromhex(digest_hex)),
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(digest.hex(), digest_hex)


class Account:
    """A row of the accounts table."""

    __slots__ = ("id", "name", "disabled", "created_at", "metadata", "_password")

    def __init__(self, row: sqlite3.Row) -> None:
        self.id: str = row["id"]
        self.name: str = row["name"]
        self.disabled: bool = bool(row["disabled"])
        self.created_at: str = row["created_at"]
        self.metadata: str = row["metadata"]
        self._password: str | None = row["password"]

    def check_password(self, password: str) -> bool:
        return verify_password(password, self._password)

    @property
    def has_password(self) -> bool:
        return bool(self._password)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Account {self.name}{' disabled' if self.disabled else ''}>"


class AccountStore:
    """SQLite-backed account directory. Thread-safe like the local backend."""

    def __init__(self, db_path: str = ":memory:") -> None:
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        if db_path != ":memory:":
            self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(_SCHEMA)
        self._db.commit()
        self.db_path = db_path

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # The auth database is shared with the OAuth server (memry.oauth), which
    # adds its own tables. One connection and one lock, so there is a single
    # writer and a single file to back up.
    @property
    def db(self) -> sqlite3.Connection:
        return self._db

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    # -- accounts ------------------------------------------------------
    def create(self, name: str, *, password: str | None = None) -> Account:
        """Create an account. Raises ValueError if the name is taken."""
        name = name.strip()
        if not name:
            raise ValueError("account name must not be empty")
        if "::" in name:
            # names become namespace prefixes; "::" would let one account's
            # space nest inside another's
            raise ValueError("account name must not contain '::'")
        with self._lock:
            try:
                self._db.execute(
                    "INSERT INTO accounts (id, name, password, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        new_id(),
                        name,
                        hash_password(password) if password else None,
                        utcnow(),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"account {name!r} already exists") from exc
            self._db.commit()
        account = self.get_by_name(name)
        assert account is not None
        return account

    def get_by_name(self, name: str) -> Account | None:
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM accounts WHERE name = ?", (name,)
            ).fetchone()
        return Account(row) if row else None

    def list(self) -> list[Account]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM accounts ORDER BY name"
            ).fetchall()
        return [Account(r) for r in rows]

    def set_password(self, name: str, password: str) -> bool:
        with self._lock:
            cur = self._db.execute(
                "UPDATE accounts SET password = ? WHERE name = ?",
                (hash_password(password), name),
            )
            self._db.commit()
        return cur.rowcount > 0

    def set_disabled(self, name: str, disabled: bool) -> bool:
        with self._lock:
            cur = self._db.execute(
                "UPDATE accounts SET disabled = ? WHERE name = ?",
                (1 if disabled else 0, name),
            )
            self._db.commit()
        return cur.rowcount > 0

    def delete(self, name: str) -> bool:
        with self._lock:
            account = self.get_by_name(name)
            if account is None:
                return False
            self._db.execute(
                "DELETE FROM account_keys WHERE account_id = ?", (account.id,)
            )
            self._db.execute("DELETE FROM accounts WHERE id = ?", (account.id,))
            self._db.commit()
        return True

    # -- API keys ------------------------------------------------------
    def issue_key(self, name: str, *, label: str | None = None) -> str:
        """Mint an API key for an account and return it.

        This is the only moment the key exists in plaintext; only its hash is
        stored, so it cannot be shown again later.
        """
        account = self.get_by_name(name)
        if account is None:
            raise ValueError(f"no such account: {name!r}")
        api_key = KEY_PREFIX + secrets.token_urlsafe(32)
        with self._lock:
            self._db.execute(
                "INSERT INTO account_keys (key_hash, account_id, label, created_at) "
                "VALUES (?, ?, ?, ?)",
                (hash_key(api_key), account.id, label, utcnow()),
            )
            self._db.commit()
        return api_key

    def account_for_key(self, api_key: str) -> Account | None:
        """Resolve a presented API key, or None if unknown or disabled."""
        if not api_key:
            return None
        with self._lock:
            row = self._db.execute(
                "SELECT a.* FROM accounts a "
                "JOIN account_keys k ON k.account_id = a.id "
                "WHERE k.key_hash = ?",
                (hash_key(api_key),),
            ).fetchone()
        if row is None:
            return None
        account = Account(row)
        return None if account.disabled else account

    def keys_for(self, name: str) -> list[dict[str, Any]]:
        account = self.get_by_name(name)
        if account is None:
            return []
        with self._lock:
            rows = self._db.execute(
                "SELECT label, created_at FROM account_keys WHERE account_id = ? "
                "ORDER BY created_at",
                (account.id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def revoke_keys(self, name: str) -> int:
        account = self.get_by_name(name)
        if account is None:
            return 0
        with self._lock:
            cur = self._db.execute(
                "DELETE FROM account_keys WHERE account_id = ?", (account.id,)
            )
            self._db.commit()
        return cur.rowcount

    def is_empty(self) -> bool:
        """No accounts configured means the server stays in single-user mode."""
        with self._lock:
            row = self._db.execute("SELECT 1 FROM accounts LIMIT 1").fetchone()
        return row is None

    # -- dashboard sessions --------------------------------------------
    def create_session(self, account: str | None) -> str:
        """Open a dashboard session and return its cookie token.

        ``account`` None is an admin session. Like API keys, only a hash of the
        token is stored, so a leaked database does not hand over live sessions.
        """
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._db.execute(
                "INSERT INTO dashboard_sessions (token_hash, account, created_at, "
                "expires_at) VALUES (?, ?, ?, ?)",
                (hash_key(token), account, utcnow(), time.time() + SESSION_TTL),
            )
            self._db.commit()
        return token

    def resolve_session(self, token: str) -> tuple[str, str | None] | None:
        """Map a cookie token to ("admin", None) or ("account", name), or None.

        Expired sessions and sessions whose account was deleted or disabled
        return None, so revoking an account takes effect on the dashboard too.
        """
        if not token:
            return None
        with self._lock:
            row = self._db.execute(
                "SELECT account, expires_at FROM dashboard_sessions WHERE token_hash = ?",
                (hash_key(token),),
            ).fetchone()
        if row is None or row["expires_at"] < time.time():
            return None
        if row["account"] is None:
            return ("admin", None)
        account = self.get_by_name(row["account"])
        if account is None or account.disabled:
            return None
        return ("account", account.name)

    def delete_session(self, token: str) -> None:
        with self._lock:
            self._db.execute(
                "DELETE FROM dashboard_sessions WHERE token_hash = ?", (hash_key(token),)
            )
            self._db.commit()

    def purge_expired_sessions(self) -> int:
        with self._lock:
            cur = self._db.execute(
                "DELETE FROM dashboard_sessions WHERE expires_at < ?", (time.time(),)
            )
            self._db.commit()
        return cur.rowcount
