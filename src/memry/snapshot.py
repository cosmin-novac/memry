"""A nightly copy of the database that can be restored after losing the live one.

There is one copy, and a new one replaces it only once the new one is written
and checked, so a failed or corrupt run never costs the copy that was there:

1. each live file (``memry.db``, and ``auth.db`` when it exists) is copied with
   SQLite's online backup API from a connection of its own, a few pages at a
   time with a short pause between steps. The server keeps reading and writing
   during the copy, and Memry's own backend lock is never taken;
2. the copy goes to a temporary file in the snapshot directory, is flushed to
   disk, and is checked: ``PRAGMA integrity_check`` must answer "ok" and the
   counts (memories, entities, episodes) must read;
3. only then does it replace the previous copy (``os.replace``, atomic on one
   filesystem), and ``snapshot.json`` records when, which version, each file's
   size and sha256, the counts and how long it took.

A failure leaves the previous copy and ``snapshot.json`` as they were, removes
only the temporary file, and writes what went wrong to
``snapshot-failure.json`` beside them, which the dashboard's About shows.

The snapshot directory must be one Memry never writes to otherwise, so not the
data directory. On a server with one disk this protects against a damaged
database file, a bad migration or a mistaken delete, but not against losing the
disk: that needs the optional offsite copy (``memry.offsite``).

Restore: stop the server, copy ``memry.db`` (and ``auth.db``) from the snapshot
directory back over the files in the data directory, delete any
``memry.db-wal`` and ``memry.db-shm`` left beside them, and start the server.
docs/self-hosting.md has the exact commands.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

from .config import Config
from .models import TOPIC_TYPE

log = logging.getLogger("memry")

MANIFEST = "snapshot.json"
FAILURE = "snapshot-failure.json"
DEFAULT_AT = (3, 30)
#: The age of the last good copy past which the server makes one soon after it
#: starts, whatever the time of day.
CATCH_UP_AFTER = timedelta(hours=26)
#: After a failed run the scheduler waits this long before trying again.
RETRY_AFTER = timedelta(hours=1)
#: Pages copied per backup step, and the pause between steps that lets the
#: server's writes through.
PAGES_PER_STEP = 1024
STEP_PAUSE = 0.005
#: A copy still not done stepwise after this long (writes kept restarting it)
#: is finished in one step. In WAL mode one step only holds a read
#: transaction, which does not block the server's writes either.
STEPWISE_DEADLINE = 60.0
#: Free space kept beyond the new copy's size: on a one-disk server, filling
#: the disk would hurt the live database too.
SPACE_MARGIN = 64 * 1024 * 1024
STALE_TEMP_AGE = 24 * 3600

_run_lock = threading.Lock()


class SnapshotError(RuntimeError):
    """A snapshot step that failed; the previous copy is untouched."""


class _TooBusy(Exception):
    pass


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _version() -> str:
    from . import __version__

    return __version__


def source_files(config: Config) -> dict[str, Path]:
    """The live files a snapshot copies, by the name the copy gets."""
    from .accounts import default_auth_db_path

    if not config.db_path or config.db_path == ":memory:":
        raise SnapshotError("an in-memory store has no file to snapshot")
    files = {"memry.db": Path(config.db_path)}
    auth = config.auth_db_path or default_auth_db_path(config.db_path)
    if auth != ":memory:" and Path(auth).exists():
        files["auth.db"] = Path(auth)
    return files


def _ro_uri(path: Path) -> str:
    return "file:" + quote(path.resolve().as_posix(), safe="/:") + "?mode=ro"


def _connect_ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(_ro_uri(path), uri=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    with open(path, "rb+") as fh:
        os.fsync(fh.fileno())


def _fsync_dir(path: Path) -> None:
    if os.name == "nt":  # Windows has no directory handles to flush
        return
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _unlink_temp(path: Path) -> None:
    for extra in ("", "-journal", "-wal", "-shm"):
        try:
            Path(str(path) + extra).unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:  # pragma: no cover - a file still open elsewhere
            log.warning("snapshot: could not remove temporary file %s: %s", path, exc)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """Write ``data`` next to ``path`` and move it into place in one step."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            _unlink_temp(tmp)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def read_manifest(directory: str | os.PathLike[str]) -> dict[str, Any] | None:
    return _read_json(Path(directory) / MANIFEST)


def read_failure(directory: str | os.PathLike[str]) -> dict[str, Any] | None:
    return _read_json(Path(directory) / FAILURE)


def copy_database(source: Path, dest: Path) -> None:
    """Copy a live SQLite file with the online backup API.

    The copy reads from a connection of its own, so it never waits on
    Memry's backend lock, and steps a few pages at a time with a short pause,
    so the server's writes get through between steps."""
    if not source.exists():
        raise SnapshotError(f"{source} does not exist")
    src = sqlite3.connect(str(source), timeout=30)
    try:
        started = time.monotonic()

        def pause(status: int, remaining: int, total: int) -> None:
            if time.monotonic() - started > STEPWISE_DEADLINE:
                raise _TooBusy()
            time.sleep(STEP_PAUSE)

        dst = sqlite3.connect(str(dest))
        try:
            try:
                src.backup(dst, pages=PAGES_PER_STEP, progress=pause)
            except _TooBusy:
                log.info("snapshot: %s kept changing; finishing the copy in one step", source)
                src.backup(dst, pages=-1)
            # A rollback-journal file: the copy then is one file, with no
            # -wal or -shm beside it, and opens read-only without writing.
            dst.execute("PRAGMA journal_mode=DELETE")
            dst.commit()
        finally:
            dst.close()
    finally:
        src.close()


def integrity(path: Path) -> str:
    """``PRAGMA integrity_check`` of a file, read-only: "ok" when sound."""
    try:
        conn = _connect_ro(path)
    except sqlite3.Error as exc:
        return f"cannot open: {exc}"
    try:
        rows = conn.execute("PRAGMA integrity_check").fetchall()
    except sqlite3.DatabaseError as exc:  # not a database at all
        return str(exc)
    finally:
        conn.close()
    return "; ".join(str(row[0]) for row in rows) or "no answer"


def read_counts(path: Path) -> dict[str, int]:
    """What a copy of ``memry.db`` holds, read from the copy itself."""
    conn = _connect_ro(path)
    try:
        one = lambda sql: int(conn.execute(sql).fetchone()[0])  # noqa: E731
        return {
            "memories_in_use": one("SELECT COUNT(*) FROM memories WHERE invalid_at IS NULL"),
            "memories_history": one(
                "SELECT COUNT(*) FROM memories WHERE invalid_at IS NOT NULL "
                "AND superseded_by IS NOT NULL AND superseded_by != ''"),
            "memories_forgotten": one(
                "SELECT COUNT(*) FROM memories WHERE invalid_at IS NOT NULL "
                "AND (superseded_by IS NULL OR superseded_by = '')"),
            "entities": one(
                "SELECT COUNT(*) FROM entities WHERE merged_into IS NULL "
                f"AND IFNULL(entity_type, '') != '{TOPIC_TYPE}'"),
            "tags": one(
                "SELECT COUNT(*) FROM entities WHERE merged_into IS NULL "
                f"AND entity_type = '{TOPIC_TYPE}'"),
            "episodes": one("SELECT COUNT(*) FROM episodes"),
        }
    finally:
        conn.close()


def _refuse_data_dir(target: Path, sources: dict[str, Path]) -> None:
    resolved = target.resolve()
    for path in sources.values():
        if path.resolve().parent == resolved:
            raise SnapshotError(
                f"the snapshot directory {target} is the data directory; use a "
                "directory Memry does not write to otherwise"
            )


def _check_space(target: Path, sources: dict[str, Path]) -> None:
    needed = sum(p.stat().st_size for p in sources.values() if p.exists())
    for p in sources.values():
        wal = Path(str(p) + "-wal")
        if wal.exists():
            needed += wal.stat().st_size
    free = shutil.disk_usage(target).free
    if free < needed + SPACE_MARGIN:
        raise SnapshotError(
            f"not enough free space in {target}: {free} bytes free, "
            f"{needed + SPACE_MARGIN} needed"
        )


def _remove_stale_temps(target: Path) -> None:
    """Temporary files a killed run left behind (a run still going is younger)."""
    cutoff = time.time() - STALE_TEMP_AGE
    for path in target.glob(".*.tmp"):
        try:
            if path.stat().st_mtime < cutoff:
                _unlink_temp(path)
        except OSError:
            pass


def take_snapshot(
    config: Config,
    target: str | os.PathLike[str] | None = None,
    *,
    offsite: bool = True,
    offsite_client: Any = None,
) -> dict[str, Any]:
    """Make the snapshot. Never raises for a failed run: the answer says
    ``ok: False`` with the error, which is also in ``snapshot-failure.json``."""
    directory = target or config.snapshot.dir
    if not directory:
        return {"ok": False, "error": "no snapshot directory: set MEMRY_SNAPSHOT_DIR or pass --to"}
    target_dir = Path(directory)
    started_at = _utcnow()
    with _run_lock:
        started = time.monotonic()
        temps: list[tuple[str, Path]] = []
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            sources = source_files(config)
            _refuse_data_dir(target_dir, sources)
            _remove_stale_temps(target_dir)
            _check_space(target_dir, sources)
            files: dict[str, dict[str, Any]] = {}
            counts: dict[str, int] = {}
            for name, source in sources.items():
                tmp = target_dir / f".{name}.{os.getpid()}.tmp"
                temps.append((name, tmp))
                copy_database(source, tmp)
                _fsync_file(tmp)
                verdict = integrity(tmp)
                if verdict != "ok":
                    raise SnapshotError(f"the copy of {name} failed its integrity check: {verdict}")
                if name == "memry.db":
                    counts = read_counts(tmp)
                files[name] = {
                    "size": tmp.stat().st_size,
                    "sha256": sha256_file(tmp),
                    "source": str(source),
                }
            # Every copy is written and sound: only now replace the old ones.
            for name, tmp in temps:
                os.replace(tmp, target_dir / name)
            temps = []
            _fsync_dir(target_dir)
            manifest: dict[str, Any] = {
                "format": "memry-snapshot",
                "created_at": started_at,
                "finished_at": _utcnow(),
                "duration_s": round(time.monotonic() - started, 3),
                "memry_version": _version(),
                "integrity": "ok",
                "files": files,
                "counts": counts,
            }
            _write_json(target_dir / MANIFEST, manifest)
        except Exception as exc:
            for _, tmp in temps:
                _unlink_temp(tmp)
            error = f"{type(exc).__name__}: {exc}" if not isinstance(exc, SnapshotError) else str(exc)
            failure = {"at": started_at, "error": error, "memry_version": _version()}
            try:
                target_dir.mkdir(parents=True, exist_ok=True)
                _write_json(target_dir / FAILURE, failure)
            except Exception:  # pragma: no cover - the directory itself is gone
                pass
            log.error("snapshot to %s failed, the previous copy is kept: %s", target_dir, error)
            return {"ok": False, "dir": str(target_dir), **failure}
    log.info("snapshot written to %s in %.1fs", target_dir, manifest["duration_s"])
    from . import offsite as offsite_mod

    if offsite and offsite_mod.configured(config):
        manifest["offsite"] = offsite_mod.upload_snapshot(
            config, target_dir, manifest, client=offsite_client
        )
        try:
            _write_json(target_dir / MANIFEST, manifest)
        except Exception as exc:  # pragma: no cover - the copy itself stands
            log.error("snapshot: could not record the offsite result: %s", exc)
    return {"ok": True, "dir": str(target_dir), **manifest}


def check_snapshot(directory: str | os.PathLike[str]) -> dict[str, Any]:
    """Check the copy against its manifest: each file there, of the size and
    sha256 recorded, and sound by ``integrity_check``. Writes nothing."""
    target = Path(directory)
    manifest = read_manifest(target)
    if manifest is None:
        return {"ok": False, "dir": str(target), "problems": [f"no {MANIFEST} in {target}"]}
    problems: list[str] = []
    checked: dict[str, dict[str, Any]] = {}
    for name, info in (manifest.get("files") or {}).items():
        path = target / name
        if not path.exists():
            problems.append(f"{name} is missing")
            continue
        size = path.stat().st_size
        digest = sha256_file(path)
        verdict = integrity(path)
        checked[name] = {"size": size, "sha256": digest, "integrity": verdict}
        if size != info.get("size"):
            problems.append(f"{name} is {size} bytes, the manifest says {info.get('size')}")
        if digest != info.get("sha256"):
            problems.append(f"{name} does not match the sha256 in the manifest")
        if verdict != "ok":
            problems.append(f"{name} failed its integrity check: {verdict}")
    if not checked and not problems:
        problems.append("the manifest lists no files")
    return {
        "ok": not problems,
        "dir": str(target),
        "created_at": manifest.get("created_at"),
        "memry_version": manifest.get("memry_version"),
        "counts": manifest.get("counts"),
        "files": checked,
        "problems": problems,
    }


# -- schedule -------------------------------------------------------------


def parse_at(value: str | None) -> tuple[int, int]:
    """``"HH:MM"`` on the server's clock; anything else is the default."""
    try:
        hour, minute = (int(part) for part in str(value).strip().split(":"))
        if 0 <= hour < 24 and 0 <= minute < 60:
            return hour, minute
    except (ValueError, TypeError):
        pass
    if value:
        log.warning("MEMRY_SNAPSHOT_AT=%r is not HH:MM; using %02d:%02d", value, *DEFAULT_AT)
    return DEFAULT_AT


def _parse_time(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def snapshot_due(
    now: datetime,
    at: tuple[int, int],
    last_success: datetime | None,
    last_failure: datetime | None = None,
) -> bool:
    """Whether a snapshot should run at ``now`` (aware, the server's local time).

    Due once a day from ``at`` on, unless one already succeeded that day; and
    at any time when the last good copy is older than ``CATCH_UP_AFTER`` (or
    there is none). A failure waits ``RETRY_AFTER`` before the next try."""
    if (
        last_failure is not None
        and (last_success is None or last_failure > last_success)
        and now - last_failure < RETRY_AFTER
    ):
        return False
    if last_success is None or now - last_success >= CATCH_UP_AFTER:
        return True
    slot = now.replace(hour=at[0], minute=at[1], second=0, microsecond=0)
    return now >= slot and last_success.astimezone(now.tzinfo).date() < now.date()


class SnapshotScheduler:
    """Runs ``take_snapshot`` once a day, from the server's lifespan.

    The clock and the run are injectable so the timing is tested without
    waiting. Nothing a run does can stop the server: failures are logged and
    recorded, and the next try waits ``RETRY_AFTER``."""

    def __init__(
        self,
        config: Config,
        *,
        clock: Callable[[], datetime] | None = None,
        runner: Callable[[], dict[str, Any]] | None = None,
        startup_delay: float = 90.0,
        check_every: float = 60.0,
    ) -> None:
        self.config = config
        self.at = parse_at(config.snapshot.at)
        self.clock = clock or (lambda: datetime.now().astimezone())
        self.runner = runner or (lambda: take_snapshot(config))
        self.startup_delay = startup_delay
        self.check_every = check_every
        directory = config.snapshot.dir
        manifest = read_manifest(directory) if directory else None
        failure = read_failure(directory) if directory else None
        self.last_success = _parse_time((manifest or {}).get("created_at"))
        self.last_failure = _parse_time((failure or {}).get("at"))

    def due(self) -> bool:
        return snapshot_due(self.clock(), self.at, self.last_success, self.last_failure)

    def _record(self, result: dict[str, Any] | None, when: datetime) -> None:
        if result and result.get("ok"):
            self.last_success = when
        else:
            self.last_failure = when

    def tick(self) -> dict[str, Any] | None:
        """Run a snapshot if one is due, in this thread."""
        if not self.due():
            return None
        when = self.clock()
        result = None
        try:
            result = self.runner()
        finally:
            self._record(result, when)
        return result

    async def run_forever(self) -> None:
        await asyncio.sleep(self.startup_delay)
        while True:
            try:
                if self.due():
                    when = self.clock()
                    result = None
                    try:
                        result = await asyncio.to_thread(self.runner)
                    finally:
                        self._record(result, when)
            except asyncio.CancelledError:
                raise
            except Exception:  # a snapshot must never take the server down
                log.exception("snapshot scheduler: the run failed")
            await asyncio.sleep(self.check_every)


def status(config: Config) -> dict[str, Any]:
    """What the dashboard's About shows about backups."""
    from . import offsite as offsite_mod

    snap = config.snapshot
    hour, minute = parse_at(snap.at)
    out: dict[str, Any] = {
        "enabled": bool(snap.dir),
        "dir": snap.dir,
        "host_dir": snap.host_dir,
        "at": f"{hour:02d}:{minute:02d}",
        "timezone": time.strftime("%Z") or "local",
        "offsite": offsite_mod.configured(config),
        "last_success": None,
        "last_failure": None,
    }
    if not snap.dir:
        return out
    manifest = read_manifest(snap.dir)
    if manifest:
        files = manifest.get("files") or {}
        out["last_success"] = {
            "at": manifest.get("created_at"),
            "size": sum(int(f.get("size") or 0) for f in files.values()),
            "files": sorted(files),
            "verified": manifest.get("integrity") == "ok",
            "duration_s": manifest.get("duration_s"),
            "counts": manifest.get("counts"),
            "offsite": manifest.get("offsite"),
        }
    failure = read_failure(snap.dir)
    if failure:
        out["last_failure"] = {"at": failure.get("at"), "error": failure.get("error")}
    return out
