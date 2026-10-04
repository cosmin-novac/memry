"""What the dashboard's About shows about the server: backups, footprint,
response times and the host (``GET /api/v1/about``).

Response times are measured in this process by ``ResponseTimeMiddleware``: the
time to the first byte of each response, kept for the last few hundred
requests. /health is left out, since the container's health check would
otherwise make up most of the window.
"""

from __future__ import annotations

import math
import os
import platform
import shutil
import socket
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import snapshot

WINDOW = 200
SKIPPED_PATHS = ("/health",)


class ResponseWindow:
    """The last ``size`` response times, in milliseconds."""

    def __init__(self, size: int = WINDOW) -> None:
        self.size = size
        self._times: deque[float] = deque(maxlen=size)
        self._lock = threading.Lock()

    def record(self, ms: float) -> None:
        with self._lock:
            self._times.append(ms)

    def summary(self) -> dict[str, Any]:
        with self._lock:
            values = sorted(self._times)
        if not values:
            return {"count": 0, "window": self.size, "median_ms": None, "p95_ms": None}
        return {
            "count": len(values),
            "window": self.size,
            "median_ms": round(percentile(values, 0.5), 1),
            "p95_ms": round(percentile(values, 0.95), 1),
        }


def percentile(sorted_values: list[float], q: float) -> float:
    """Nearest-rank percentile of an already sorted, non-empty list."""
    rank = max(1, math.ceil(q * len(sorted_values)))
    return sorted_values[min(rank, len(sorted_values)) - 1]


class ResponseTimeMiddleware:
    """Pure ASGI: times each HTTP request to its response's first byte, so a
    streamed response (MCP over SSE) counts its wait, not how long it stays
    open."""

    def __init__(self, app: Any, window: ResponseWindow) -> None:
        self.app = app
        self.window = window

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or scope.get("path") in SKIPPED_PATHS:
            await self.app(scope, receive, send)
            return
        started = time.perf_counter()
        recorded = False

        async def timed_send(message: Any) -> None:
            nonlocal recorded
            if not recorded and message.get("type") == "http.response.start":
                recorded = True
                self.window.record((time.perf_counter() - started) * 1000)
            await send(message)

        await self.app(scope, receive, timed_send)


def _size(path: str | None) -> int | None:
    try:
        return os.path.getsize(path) if path else None
    except OSError:
        return None


def _host_uptime() -> float | None:
    try:
        return float(Path("/proc/uptime").read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def about_payload(
    store: Any,
    *,
    principal: Any,
    window: ResponseWindow,
    started_at: datetime,
    started_monotonic: float,
) -> dict[str, Any]:
    """The About payload. An account or tenant gets the version and its own
    memory counts only: nothing about the host or anyone else's data."""
    from . import __version__
    from .accounts import default_auth_db_path

    if not principal.is_admin:
        mine = store.count_memories(owner_prefix=principal.prefix)
        return {
            "version": __version__,
            "scope": "account",
            "counts": {
                "memories_in_use": mine["active"],
                "memories_history": mine["invalidated"] - mine["forgotten"],
                "memories_forgotten": mine["forgotten"],
            },
        }

    config = store.config
    stats = store.stats()
    counts = store.count_memories()
    db_path = config.db_path if config.db_path != ":memory:" else None
    auth_path = config.auth_db_path or (default_auth_db_path(db_path) if db_path else None)
    disk: dict[str, Any] = {"free": None, "total": None}
    if db_path:
        try:
            usage = shutil.disk_usage(Path(db_path).parent)
            disk = {"free": usage.free, "total": usage.total}
        except OSError:
            pass
    return {
        "version": __version__,
        "scope": "server",
        "backups": snapshot.status(config),
        "footprint": {
            "db_path": db_path,
            "db_bytes": _size(db_path),
            "wal_bytes": _size(db_path + "-wal") if db_path else None,
            "auth_db_bytes": _size(auth_path),
            "memories_in_use": counts["active"],
            "memories_history": counts["invalidated"] - counts["forgotten"],
            "memories_forgotten": counts["forgotten"],
            "entities": stats.get("entities"),
            "tags": stats.get("topics"),
            "episodes": stats.get("episodes"),
            "disk_free_bytes": disk["free"],
            "disk_total_bytes": disk["total"],
        },
        "response_times": window.summary(),
        "host": {
            "hostname": socket.gethostname(),
            "python": platform.python_version(),
            "started_at": started_at.isoformat(timespec="seconds"),
            "uptime_s": round(time.monotonic() - started_monotonic),
            "host_uptime_s": _host_uptime(),
            "llm": stats.get("llm"),
            "embedder": stats.get("embedder"),
            "decider": stats.get("decider"),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
