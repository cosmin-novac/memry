"""The nightly snapshot: one verified copy, replaced only by a newer verified copy."""

from __future__ import annotations

import asyncio
import json
import shutil
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from memry import snapshot
from memry.accounts import AccountStore
from memry.cli import main as cli_main
from memry.config import Config, SnapshotConfig
from memry.providers.embeddings import HashEmbedder
from memry.providers.llm import NoneLLM
from memry.snapshot import (
    FAILURE,
    MANIFEST,
    SnapshotScheduler,
    check_snapshot,
    snapshot_due,
    take_snapshot,
)
from memry.store import MemoryStore


def make_store(db_path: Path, **snap) -> MemoryStore:
    config = Config(db_path=str(db_path), snapshot=SnapshotConfig(**snap))
    return MemoryStore(config, llm=NoneLLM(), embedder=HashEmbedder(64))


@pytest.fixture
def live(tmp_path):
    """A file-backed store with three memories (one forgotten) and an account."""
    store = make_store(tmp_path / "data" / "memry.db")
    ids = [
        store.add(text, user_id="ada", infer=False).actions[0].memory_id
        for text in ("Ada likes tea.", "Ada works on Helios.", "Ada lives in Lyon.")
    ]
    store.delete(ids[2])
    accounts = AccountStore(str(tmp_path / "data" / "auth.db"))
    accounts.create("ada", password="pw-123456")
    accounts.close()
    yield store, tmp_path / "backups"
    store.close()


def _leftovers(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir()
                  if p.name.endswith(".tmp") or p.name.endswith(("-wal", "-shm", "-journal")))


def test_a_snapshot_writes_verified_copies_and_a_manifest(live):
    store, target = live
    result = take_snapshot(store.config, target)

    assert result["ok"], result
    assert sorted(result["files"]) == ["auth.db", "memry.db"]
    manifest = json.loads((target / MANIFEST).read_text(encoding="utf-8"))
    assert manifest["format"] == "memry-snapshot"
    assert manifest["integrity"] == "ok"
    assert manifest["counts"]["memories_in_use"] == 2
    assert manifest["counts"]["memories_forgotten"] == 1
    assert manifest["counts"]["episodes"] >= 0 and "entities" in manifest["counts"]
    for name, info in manifest["files"].items():
        assert (target / name).stat().st_size == info["size"]
        assert snapshot.sha256_file(target / name) == info["sha256"]
        assert snapshot.integrity(target / name) == "ok"
    assert manifest["duration_s"] >= 0 and manifest["memry_version"]
    # one file per database: no temporary, journal, WAL or shared-memory file
    assert _leftovers(target) == []
    assert not (target / FAILURE).exists()


def test_the_copy_never_takes_the_backends_lock(live):
    """The live server keeps working: the copy reads from its own connection,
    so a held backend lock does not stop it."""
    store, target = live
    held, release = threading.Event(), threading.Event()

    def hold():
        with store.backend._lock:
            held.set()
            release.wait(10)

    holder = threading.Thread(target=hold)
    holder.start()
    held.wait(5)
    try:
        result = take_snapshot(store.config, target)
    finally:
        release.set()
        holder.join()
    assert result["ok"], result


def test_the_server_writes_on_while_a_stepwise_copy_runs(live, monkeypatch):
    store, target = live
    monkeypatch.setattr(snapshot, "PAGES_PER_STEP", 1)
    stop = threading.Event()

    def write():
        i = 0
        while not stop.is_set() and i < 200:
            store.add(f"Note number {i}.", user_id="ada", infer=False)
            i += 1

    writer = threading.Thread(target=write)
    writer.start()
    try:
        result = take_snapshot(store.config, target)
    finally:
        stop.set()
        writer.join()
    assert result["ok"], result
    assert check_snapshot(target)["ok"]


def _state(target: Path) -> tuple[bytes, bytes]:
    return (target / "memry.db").read_bytes(), (target / MANIFEST).read_bytes()


def test_a_failed_integrity_check_keeps_the_previous_copy(live, monkeypatch):
    store, target = live
    assert take_snapshot(store.config, target)["ok"]
    before = _state(target)
    store.add("Ada learns Rust.", user_id="ada", infer=False)

    monkeypatch.setattr(snapshot, "integrity", lambda path: "*** page 7 is never used")
    result = take_snapshot(store.config, target)

    assert result["ok"] is False and "integrity check" in result["error"]
    assert _state(target) == before
    failure = json.loads((target / FAILURE).read_text(encoding="utf-8"))
    assert "page 7" in failure["error"] and failure["at"]
    assert _leftovers(target) == []
    monkeypatch.undo()
    assert check_snapshot(target)["ok"]


def test_an_exception_mid_way_keeps_the_previous_copy(live, monkeypatch):
    store, target = live
    assert take_snapshot(store.config, target)["ok"]
    before = _state(target)

    def boom(path):
        raise RuntimeError("disk went away")

    monkeypatch.setattr(snapshot, "read_counts", boom)
    result = take_snapshot(store.config, target)

    assert result["ok"] is False and "disk went away" in result["error"]
    assert _state(target) == before
    assert "disk went away" in snapshot.read_failure(target)["error"]
    assert _leftovers(target) == []


def test_the_old_copy_stays_valid_while_the_new_one_is_written(live, monkeypatch):
    store, target = live
    assert take_snapshot(store.config, target)["ok"]
    old = snapshot.read_manifest(target)
    store.add("Ada adopted a cat.", user_id="ada", infer=False)
    seen = []
    real_copy = snapshot.copy_database

    def copy_and_look(source, dest, **kw):
        real_copy(source, dest, **kw)
        # mid-run: the new copy exists beside the old, which still checks out
        seen.append(check_snapshot(target)["ok"])
        assert snapshot.sha256_file(target / "memry.db") == old["files"]["memry.db"]["sha256"]

    monkeypatch.setattr(snapshot, "copy_database", copy_and_look)
    result = take_snapshot(store.config, target)

    assert result["ok"] and seen and all(seen)
    assert result["counts"]["memories_in_use"] == old["counts"]["memories_in_use"] + 1
    assert check_snapshot(target)["ok"]


def test_check_verifies_the_copy_against_its_manifest_and_writes_nothing(live):
    store, target = live
    assert check_snapshot(target)["ok"] is False  # nothing there yet
    assert take_snapshot(store.config, target)["ok"]
    listing = {p.name: p.stat().st_mtime_ns for p in target.iterdir()}

    checked = check_snapshot(target)
    assert checked["ok"], checked
    assert set(checked["files"]) == {"memry.db", "auth.db"}
    assert {p.name: p.stat().st_mtime_ns for p in target.iterdir()} == listing

    with open(target / "memry.db", "ab") as fh:
        fh.write(b"\0")
    damaged = check_snapshot(target)
    assert damaged["ok"] is False
    assert any("sha256" in p for p in damaged["problems"])
    assert any("bytes" in p for p in damaged["problems"])


def test_a_restored_copy_opens_with_the_same_counts(live, tmp_path):
    store, target = live
    assert take_snapshot(store.config, target)["ok"]
    expected = store.count_memories()

    restored = tmp_path / "restored"
    restored.mkdir()
    for name in ("memry.db", "auth.db"):
        shutil.copy2(target / name, restored / name)
    again = make_store(restored / "memry.db")
    try:
        assert again.count_memories() == expected
        assert [m.content for m in again.get_all(user_id="ada")] == [
            m.content for m in store.get_all(user_id="ada")
        ]
    finally:
        again.close()
    accounts = AccountStore(str(restored / "auth.db"))
    try:
        assert accounts.get_by_name("ada").check_password("pw-123456")
    finally:
        accounts.close()


def test_the_data_directory_is_refused_as_a_snapshot_directory(live):
    store, _ = live
    data_dir = Path(store.config.db_path).parent
    result = take_snapshot(store.config, data_dir)
    assert result["ok"] is False and "data directory" in result["error"]


def test_no_directory_and_an_in_memory_store_are_refused(tmp_path):
    assert take_snapshot(Config(db_path=":memory:"))["ok"] is False
    assert take_snapshot(Config(db_path=":memory:"), tmp_path)["ok"] is False


def test_the_cli_writes_and_checks_a_snapshot(live, monkeypatch, capsys):
    store, target = live
    monkeypatch.setenv("MEMRY_CONFIG", str(target.parent / "none.json"))
    monkeypatch.setenv("MEMRY_DB_PATH", store.config.db_path)
    monkeypatch.delenv("MEMRY_SNAPSHOT_DIR", raising=False)
    assert cli_main(["snapshot"]) == 2
    assert cli_main(["snapshot", "--to", str(target), "--no-offsite"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"]
    monkeypatch.setenv("MEMRY_SNAPSHOT_DIR", str(target))
    assert cli_main(["snapshot", "--check"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"]
    (target / "memry.db").write_bytes(b"not a database")
    assert cli_main(["snapshot", "--check"]) == 1


# -- schedule ---------------------------------------------------------------

TZ = timezone(timedelta(hours=2))
AT = (3, 30)


def at(day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 10, day, hour, minute, tzinfo=TZ)


def test_due_once_a_day_from_the_configured_time():
    last = at(4, 3, 31)
    assert not snapshot_due(at(4, 12), AT, last)          # done today
    assert not snapshot_due(at(5, 3, 29), AT, last)       # not yet time
    assert snapshot_due(at(5, 3, 30), AT, last)           # time, not done today
    assert not snapshot_due(at(5, 3, 31), AT, at(5, 3, 30, ))


def test_catch_up_soon_after_start_when_the_copy_is_old_or_missing():
    assert snapshot_due(at(5, 1), AT, None)                 # never made
    assert snapshot_due(at(6, 1), AT, at(4, 22))            # 27 hours old
    assert not snapshot_due(at(5, 1), AT, at(4, 3, 31))     # 21.5 hours: wait for 03:30


def test_a_failure_waits_before_the_next_try():
    failed = at(5, 3, 30)
    assert not snapshot_due(at(5, 4), AT, at(4, 3, 30), failed)
    assert snapshot_due(at(5, 4, 31), AT, at(4, 3, 30), failed)
    # a failure older than the last success no longer holds anything back
    assert snapshot_due(at(6, 3, 30), AT, at(5, 3, 40), failed)


def test_the_scheduler_runs_when_due_and_remembers_the_result(tmp_path):
    now = [at(5, 1)]
    runs = []
    outcome = [{"ok": True}]

    def runner():
        runs.append(now[0])
        return outcome[0]

    config = Config(db_path=str(tmp_path / "memry.db"),
                    snapshot=SnapshotConfig(dir=str(tmp_path / "b"), at="03:30"))
    scheduler = SnapshotScheduler(config, clock=lambda: now[0], runner=runner)
    assert scheduler.tick() == {"ok": True}       # no copy yet: catch up at once
    assert scheduler.tick() is None               # done today
    now[0] = at(5, 4)
    assert scheduler.tick() is None               # done today, even past 03:30
    now[0] = at(6, 3, 30)
    outcome[0] = {"ok": False, "error": "x"}
    assert scheduler.tick()["ok"] is False
    now[0] = at(6, 4)
    assert scheduler.tick() is None               # failed: wait an hour
    now[0] = at(6, 4, 31)
    outcome[0] = {"ok": True}
    assert scheduler.tick() == {"ok": True}
    assert len(runs) == 3


def test_the_scheduler_reads_the_last_copy_from_its_manifest(live):
    store, target = live
    config = store.config.model_copy(update={"snapshot": SnapshotConfig(dir=str(target))})
    assert take_snapshot(config)["ok"]
    made = datetime.fromisoformat(snapshot.read_manifest(target)["created_at"]).astimezone()
    scheduler = SnapshotScheduler(config, clock=lambda: made + timedelta(minutes=5),
                                  runner=lambda: pytest.fail("ran again"))
    assert scheduler.tick() is None


def test_a_failing_run_never_stops_the_scheduler(tmp_path):
    calls = []

    def runner():
        calls.append(1)
        raise RuntimeError("boom")

    config = Config(db_path=str(tmp_path / "memry.db"),
                    snapshot=SnapshotConfig(dir=str(tmp_path / "b")))
    scheduler = SnapshotScheduler(config, runner=runner, startup_delay=0, check_every=0.01)

    async def go():
        task = asyncio.create_task(scheduler.run_forever())
        await asyncio.sleep(0.2)
        assert not task.done()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(go())
    assert calls == [1]  # one try, then the retry wait
    assert scheduler.last_failure is not None


def test_the_time_of_day_falls_back_to_the_default():
    assert snapshot.parse_at("04:05") == (4, 5)
    assert snapshot.parse_at("25:00") == (3, 30)
    assert snapshot.parse_at("soon") == (3, 30)
    assert snapshot.parse_at(None) == (3, 30)


def test_config_reads_the_snapshot_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMRY_CONFIG", str(tmp_path / "none.json"))
    monkeypatch.setenv("MEMRY_SNAPSHOT_DIR", "/backups")
    monkeypatch.setenv("MEMRY_SNAPSHOT_AT", "02:15")
    monkeypatch.setenv("MEMRY_SNAPSHOT_HOST_DIR", "/var/backups/memry")
    monkeypatch.setenv("MEMRY_SNAPSHOT_OFFSITE_SECRET", "s3cret")
    config = Config.load()
    assert config.snapshot.dir == "/backups" and config.snapshot.at == "02:15"
    assert config.snapshot.host_dir == "/var/backups/memry"
    assert config.redacted()["snapshot"]["offsite_secret"] == "***"
    assert Config().snapshot.dir is None  # off unless set
