"""Count, time and cap the model API calls a benchmark run makes.

``UsageMeter.install()`` wraps ``httpx.Client.send`` in this process, so every
call made through httpx is seen once, whatever object made it: the store's
text model and embedder, a decision provider, the answering model, a judge.
Each call is one row of a SQLite ledger that the processes of a parallel run
share. A row holds when the call started, how long it took, the endpoint
group, the model, the stage the caller was in (``stage``), and the tokens the
response's ``usage`` reports. A response without ``usage`` leaves the token
columns empty, and the request and response sizes (``request_chars``,
``response_chars``, in bytes of JSON) are what there is. Neither headers nor
bodies are written, so no key and no content reaches the ledger.

A cap is a number of calls per group ("chat", "embeddings", "jev", "other";
``group_of``), counted over the whole ledger. A call that would pass its
group's cap is refused before it is sent: ``CapReached`` is raised. It is a
``BaseException``, so the ``except Exception`` a store uses to carry on after
a failed provider call (a failed extraction is stored verbatim) does not
swallow it, and the run stops where it is.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import sqlite3
import statistics
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

import httpx

#: The stage a call is counted under, set by the caller with ``stage``, and
#: the part of the run (a conversation), set with ``labelled``. A thread a
#: store starts for its own calls (a pool of judge calls) does not inherit
#: them, so the latest value set in the process stands in (``_LATEST``).
_STAGE: contextvars.ContextVar[str] = contextvars.ContextVar("api_usage_stage", default="")
_LABEL: contextvars.ContextVar[str] = contextvars.ContextVar("api_usage_label", default="")
_LATEST: dict[str, list[str]] = {"api_usage_stage": [], "api_usage_label": []}


@contextlib.contextmanager
def _setting(var: contextvars.ContextVar[str], value: str) -> Iterator[None]:
    token = var.set(value)
    latest = _LATEST[var.name]
    latest.append(value)
    try:
        yield
    finally:
        var.reset(token)
        if latest and latest[-1] == value:
            latest.pop()
        elif value in latest:
            latest.remove(value)


def _current(var: contextvars.ContextVar[str]) -> str:
    latest = _LATEST[var.name]
    return var.get() or (latest[-1] if latest else "")


def stage(name: str) -> contextlib.AbstractContextManager[None]:
    """Count the calls made inside the block under the stage ``name``."""
    return _setting(_STAGE, name)


def labelled(name: str) -> contextlib.AbstractContextManager[None]:
    """Count the calls made inside the block under the label ``name``."""
    return _setting(_LABEL, name)


def current_stage() -> str:
    return _current(_STAGE)


class CapReached(BaseException):
    """A call was refused: its group reached the cap."""


def group_of(url: httpx.URL) -> str:
    """The endpoint group a call is capped and reported under."""
    path = url.path.rstrip("/")
    if path.endswith("/chat/completions"):
        return "chat"
    if path.endswith("/embeddings"):
        return "embeddings"
    if path.endswith("/systemone"):
        return "jev"
    return "other"


def _json(data: bytes) -> Any:
    try:
        return json.loads(data) if data else None
    except (ValueError, UnicodeDecodeError):
        return None


def read_usage(payload: Any) -> dict[str, int | None]:
    """Input, output, cached and reasoning tokens from a response's ``usage``
    (OpenAI's names and the input/output names), None where it says nothing."""
    usage = payload.get("usage") if isinstance(payload, dict) else None
    if not isinstance(usage, dict):
        return {"input_tokens": None, "output_tokens": None, "cached_tokens": None,
                "reasoning_tokens": None}

    def first(*keys: str, source: dict[str, Any] = usage) -> int | None:
        for key in keys:
            value = source.get(key)
            if isinstance(value, (int, float)):
                return int(value)
        return None

    prompt_details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    output_details = (usage.get("completion_tokens_details")
                      or usage.get("output_tokens_details") or {})
    return {
        "input_tokens": first("prompt_tokens", "input_tokens"),
        "output_tokens": first("completion_tokens", "output_tokens"),
        "cached_tokens": first("cached_tokens", source=prompt_details)
        if isinstance(prompt_details, dict) else None,
        "reasoning_tokens": first("reasoning_tokens", source=output_details)
        if isinstance(output_details, dict) else None,
    }


_SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    id INTEGER PRIMARY KEY,
    label TEXT, grp TEXT, host TEXT, path TEXT, model TEXT, stage TEXT,
    started REAL, seconds REAL, status INTEGER, ok INTEGER,
    input_tokens INTEGER, output_tokens INTEGER, cached_tokens INTEGER,
    reasoning_tokens INTEGER, has_usage INTEGER, items INTEGER,
    request_chars INTEGER, response_chars INTEGER, error TEXT
)"""

#: The active meter of this process (one at a time).
_active: UsageMeter | None = None
_original_send = httpx.Client.send


#: How long a process waits for another one holding the ledger.
_BUSY_SECONDS = 120


def _write_ahead(db: sqlite3.Connection) -> None:
    """Put the ledger in write-ahead mode, once per file (the mode is kept in
    the file). Worker processes open one ledger at the same moment, and the
    switch takes the file exclusively without waiting on the busy timeout, so
    a worker that finds it taken tries again until ``_BUSY_SECONDS`` pass."""
    deadline = time.monotonic() + _BUSY_SECONDS
    while True:
        try:
            if db.execute("PRAGMA journal_mode").fetchone()[0].lower() != "wal":
                db.execute("PRAGMA journal_mode=WAL")
            return
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc) or time.monotonic() > deadline:
                raise
            time.sleep(0.05)


class UsageMeter:
    """One process's view of the shared ledger.

    ``label`` names this process's share of the run (a conversation);
    ``caps`` maps a group to its most calls over the whole ledger;
    ``refine(stage, group, body)`` may name a finer stage from the request
    body (which prompt a chat call carries), or return None to keep ``stage``.
    """

    def __init__(self, path: str, *, label: str = "", caps: dict[str, int] | None = None,
                 refine: Callable[[str, str, Any], str | None] | None = None) -> None:
        self.path = str(path)
        self.label = label
        self.caps = dict(caps or {})
        self.refine = refine
        self.capped: str | None = None
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, timeout=_BUSY_SECONDS, isolation_level=None,
                                   check_same_thread=False)
        _write_ahead(self._db)
        self._db.execute(_SCHEMA)

    # -- installing ---------------------------------------------------------
    def install(self) -> UsageMeter:
        global _active
        if _active is not None and _active is not self:
            raise RuntimeError("another UsageMeter is installed")
        _active = self
        httpx.Client.send = _metered_send  # type: ignore[method-assign]
        return self

    def uninstall(self) -> None:
        global _active
        if _active is self:
            httpx.Client.send = _original_send  # type: ignore[method-assign]
            _active = None

    def close(self) -> None:
        self.uninstall()
        self._db.close()

    def __enter__(self) -> UsageMeter:
        return self.install()

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- counting -----------------------------------------------------------
    def calls(self, group: str | None = None) -> int:
        with self._lock:
            if group is None:
                return self._db.execute("SELECT count(*) FROM calls").fetchone()[0]
            return self._db.execute("SELECT count(*) FROM calls WHERE grp = ?",
                                    (group,)).fetchone()[0]

    def _reserve(self, row: dict[str, Any]) -> int:
        """The call's row, written before it is sent, inside the cap."""
        cap = self.caps.get(row["grp"])
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                if cap is not None:
                    made = self._db.execute("SELECT count(*) FROM calls WHERE grp = ?",
                                            (row["grp"],)).fetchone()[0]
                    if made >= cap:
                        self._db.execute("ROLLBACK")
                        self.capped = row["grp"]
                        raise CapReached(f"{row['grp']}: {made} calls, cap {cap}")
                cur = self._db.execute(
                    "INSERT INTO calls (label, grp, host, path, model, stage, started, items, "
                    "request_chars) VALUES (:label, :grp, :host, :path, :model, :stage, "
                    ":started, :items, :request_chars)", row)
                self._db.execute("COMMIT")
            except sqlite3.Error:
                self._db.execute("ROLLBACK")
                raise
        return int(cur.lastrowid)

    def _finish(self, row_id: int, values: dict[str, Any]) -> None:
        sets = ", ".join(f"{key} = :{key}" for key in values)
        with self._lock:
            self._db.execute(f"UPDATE calls SET {sets} WHERE id = :id", {**values, "id": row_id})

    def send(self, client: httpx.Client, request: httpx.Request, *args: Any,
             **kwargs: Any) -> httpx.Response:
        body = _json(request.content)
        grp = group_of(request.url)
        name = current_stage()
        if self.refine is not None:
            name = self.refine(name, grp, body) or name
        items = body.get("input") if isinstance(body, dict) else None
        row_id = self._reserve({
            "label": _current(_LABEL) or self.label, "grp": grp, "host": request.url.host,
            "path": request.url.path, "stage": name,
            "model": body.get("model") if isinstance(body, dict) else None,
            "started": time.time(), "items": len(items) if isinstance(items, list) else None,
            "request_chars": len(request.content or b""),
        })
        started = time.perf_counter()
        try:
            response = _original_send(client, request, *args, **kwargs)
        except BaseException as exc:
            self._finish(row_id, {"seconds": time.perf_counter() - started, "ok": 0,
                                  "error": type(exc).__name__})
            raise
        seconds = time.perf_counter() - started
        content = response.content if not kwargs.get("stream") else b""
        payload = _json(content)
        usage = read_usage(payload)
        self._finish(row_id, {
            "seconds": seconds, "status": response.status_code,
            "ok": int(response.is_success), "response_chars": len(content),
            "has_usage": int(isinstance(payload, dict) and isinstance(payload.get("usage"), dict)),
            **usage,
            "error": None if response.is_success else f"HTTP {response.status_code}",
        })
        return response


def _metered_send(client: httpx.Client, request: httpx.Request, *args: Any,
                  **kwargs: Any) -> httpx.Response:
    meter = _active
    if meter is None:
        return _original_send(client, request, *args, **kwargs)
    return meter.send(client, request, *args, **kwargs)


# -- reading the ledger ---------------------------------------------------------
_COLUMNS = ("label", "grp", "model", "stage")


def summarize(path: str, by: tuple[str, ...] = ("grp", "model", "stage")) -> list[dict[str, Any]]:
    """Calls, tokens and seconds per group of ``by`` (any of label, grp,
    model, stage): the sums, the median seconds of a call, and how many calls
    reported no usage."""
    if not set(by) <= set(_COLUMNS):
        raise ValueError(f"by: some of {_COLUMNS}")
    db = sqlite3.connect(str(path), timeout=120)
    try:
        rows = db.execute(
            f"SELECT {', '.join(by) or '1'}, seconds, ok, input_tokens, output_tokens, "
            "cached_tokens, reasoning_tokens, has_usage, request_chars, response_chars, items "
            "FROM calls").fetchall()
    finally:
        db.close()
    groups: dict[tuple, list[tuple]] = {}
    for row in rows:
        groups.setdefault(tuple(row[:len(by)]), []).append(row[len(by):])
    out = []
    for key in sorted(groups, key=lambda k: tuple(str(v) for v in k)):
        calls = groups[key]
        seconds = [c[0] for c in calls if c[0] is not None]

        def total(i: int) -> int:
            return sum(c[i] or 0 for c in calls)

        out.append({
            **dict(zip(by, key)), "calls": len(calls),
            "failed": sum(1 for c in calls if c[1] == 0),
            "input_tokens": total(2), "output_tokens": total(3),
            "cached_tokens": total(4), "reasoning_tokens": total(5),
            "without_usage": sum(1 for c in calls if not c[6]),
            "request_chars": total(7), "response_chars": total(8), "items": total(9),
            "seconds": round(sum(seconds), 3),
            "median_seconds": round(statistics.median(seconds), 3) if seconds else None,
        })
    return out
