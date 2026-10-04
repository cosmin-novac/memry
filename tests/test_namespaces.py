"""Every memory has a namespace.

A memory saved without one (``store.add`` with no user, a library call, an
old backup) was read with every namespace's, since no user means all users
in a read, and walked as a namespace of its own; "" was a namespace apart
from None that looked like none. Every write now goes to the default
namespace when it names none, and ``adopt_unscoped`` gives the rows an
older store holds without one the namespace they belong to.
"""

from __future__ import annotations

import json

import pytest

from memry.config import Config
from memry.models import Entity, EntityMention, Memory, Relation, Scope
from memry.providers.embeddings import HashEmbedder
from memry.providers.llm import NoneLLM
from memry.store import MemoryStore


@pytest.fixture
def store():
    s = MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(), embedder=HashEmbedder(64))
    yield s
    s.close()


def _unscoped(store: MemoryStore) -> dict[str, int]:
    db = store.backend._db
    return {table: db.execute(
        f"SELECT COUNT(*) FROM {table} WHERE user_id IS NULL OR user_id = ''").fetchone()[0]
        for table in ("episodes", "memories", "topics", "entities", "entity_proposals",
                      "relations", "retired_entities", "entity_merges")}


def test_a_write_without_a_user_goes_to_the_default_namespace(store):
    added = store.add("Ada prefers green tea", infer=False)
    blank = store.add("Ada dislikes coffee", user_id="", infer=False)
    deferred = store.add_deferred("Ada walks to work")
    imported = store.import_verbatim([{"content": "Ada lives in Leeds", "user_id": ""}])

    for result in (added, blank, deferred):
        memory = store.get(result.actions[0].memory_id)
        assert memory.user_id == "default"
        assert {e.user_id for e in store.backend.episodes_by_id(
            result.episode_ids).values()} == {"default"}
    assert imported["imported"] == 1
    assert {m.user_id for m in store.get_all(limit=100)} == {"default"}
    assert all(n == 0 for n in _unscoped(store).values())
    # a read with no user is still every namespace's
    store.add("Bo likes jazz", user_id="bo", infer=False)
    assert len(store.get_all(limit=100)) == 5


def test_a_backup_row_without_a_namespace_is_restored_into_the_default(store):
    other = MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(), embedder=HashEmbedder(64))
    other.backend.insert_memory(Memory(content="Ada prefers green tea", user_id=None))
    backup = other.export_backup()
    other.close()

    store.import_backup(backup)

    [memory] = store.get_all(limit=10)
    assert memory.user_id == "default"


def _old_store(store: MemoryStore) -> dict:
    """A store from before every write had a namespace: memories of None and
    of "", their tags (one, "travel", also a tag of the default namespace),
    a product both have under one name and type, a name both have as
    different types, a relation, and upkeep state of no namespace."""
    def memory(content, user_id, categories=()):
        return store.backend.insert_memory(
            Memory(content=content, user_id=user_id, categories=list(categories),
                   embedding_model=store.embedder.model_id),
            embedding=store.embedder.embed([content])[0])

    mine = memory("Harlow runs the Lisbon trip budget", "default", ["travel"])
    old = memory("Harlow books the Lisbon flights in May", None, ["Travel", "work"])
    blank = memory("Mira plans the Porto stop", "", ["travel"])

    def entity(name, kind, user_id):
        return store.backend.insert_entity(Entity(
            name=name, normalized=name.lower(), entity_type=kind, user_id=user_id))

    harlow, old_harlow = entity("Harlow", "product", "default"), entity("Harlow", "product", None)
    mira, old_mira = entity("Mira", "place", "default"), entity("Mira", "person", None)
    lisbon = entity("Lisbon", "place", None)
    for e, m in ((harlow, mine), (old_harlow, old), (old_mira, blank), (lisbon, old)):
        store.backend.add_mention(EntityMention(entity_id=e.id, memory_id=m.id, surface=e.name))
    store.backend.add_relation(Relation(subject=old_harlow.id, predicate="visits",
                                        object=lisbon.id, user_id=None, memory_id=old.id))
    store.backend.set_meta("upkeep:owner_name:", json.dumps("Ada Lind"))
    store.backend.set_meta("upkeep:last:structure:", json.dumps({"at": "2026-01-01"}))
    store.backend.set_meta("upkeep:last:structure:default", json.dumps({"at": "2026-09-01"}))
    return {"old": old, "blank": blank, "harlow": harlow, "old_harlow": old_harlow,
            "mira": mira, "old_mira": old_mira}


def test_a_dry_run_reports_what_would_move_and_writes_nothing(store):
    _old_store(store)
    before = store.export_backup()
    meta = store.backend.meta_items("upkeep:")

    report = store.adopt_unscoped(dry_run=True)

    assert report["into"] == "default" and report["dry_run"] is True
    assert report["tables"] == {**_unscoped(store), "topics": report["tables"]["topics"]}
    assert report["tables"]["memories"] == 2 and report["tables"]["relations"] == 1
    assert sorted(report["tags_folded"]) == ["travel", "travel"]  # None's and ""'s
    assert report["things_folded"] == ["Harlow (product)"]
    assert report["things_left_for_review"] == ["Mira"]
    assert report["state_carried"] == ["upkeep:owner_name:default"]
    assert report["state_kept"] == ["upkeep:last:structure:default"]
    assert store.export_backup()["tables"] == before["tables"]
    assert store.backend.meta_items("upkeep:") == meta


def test_adopting_moves_everything_once_and_folds_the_obvious_twins(store):
    world = _old_store(store)

    report = store.adopt_unscoped()

    assert all(n == 0 for n in _unscoped(store).values())
    assert report["things_folded"] == ["Harlow (product)"]
    # the memories are the default namespace's now, and found there
    found = {r.memory.id for r in store.search("Lisbon flights", user_id="default")}
    assert world["old"].id in found
    assert {m.user_id for m in store.get_all(limit=100)} == {"default"}
    # one active "travel" tag, which every memory tagged so mentions
    travel = [e for e in store.backend.list_entities(Scope(user_id="default"), limit=100,
                                                     kind="topic") if e.normalized == "travel"]
    assert len(travel) == 1
    for memory_id in (world["old"].id, world["blank"].id):
        assert travel[0].id in {e.id for e in store.backend.entities_of_memory(
            memory_id, kind="topic")}
    # the product folded into the default's, recorded and undoable; the two Miras kept
    assert store.backend.resolve_entity_id(world["old_harlow"].id) == world["harlow"].id
    [merge] = store.backend.list_merges(Scope(user_id="default"))
    assert merge["entity_id"] == world["old_harlow"].id
    assert store.backend.get_entity(world["old_mira"].id).merged_into is None
    assert [r.subject for r in store.relations(user_id="default")] == [world["harlow"].id]
    # state: the owner's name carried, the default's own last run kept
    assert store.owner_name("default") == "Ada Lind"
    assert json.loads(store.backend.get_meta("upkeep:last:structure:default")) == {
        "at": "2026-09-01"}
    assert store.backend.get_meta("upkeep:owner_name:") == ""

    before = store.export_backup()
    again = store.adopt_unscoped()
    assert all(n == 0 for n in again["tables"].values())
    assert again["tags_folded"] == again["things_folded"] == []
    assert again["state_carried"] == again["state_kept"] == []
    assert store.export_backup()["tables"] == before["tables"]
    assert store.undo_merge(world["old_harlow"].id)


def test_a_failure_midway_leaves_the_store_as_it_was(store):
    """One transaction: a fold that fails rolls back the tags folded, the
    rows moved and the state carried before it."""
    _old_store(store)
    before = store.export_backup()
    meta = store.backend.meta_items("upkeep:")
    folds = store.backend._merge_entities_locked
    calls = []

    def failing(keep, merge):
        calls.append(merge)
        if len(calls) == 3:  # after both tag folds and the move: the product
            raise RuntimeError("merge failed")
        return folds(keep, merge)

    store.backend._merge_entities_locked = failing
    with pytest.raises(RuntimeError, match="merge failed"):
        store.adopt_unscoped()

    assert store.export_backup()["tables"] == before["tables"]
    assert store.backend.meta_items("upkeep:") == meta
    assert _unscoped(store)["memories"] == 2


def test_adopt_unscoped_from_the_command_line(monkeypatch, tmp_path, capsys):
    from memry.cli import main

    monkeypatch.setenv("MEMRY_DB_PATH", str(tmp_path / "cli.db"))
    monkeypatch.setenv("MEMRY_CONFIG", str(tmp_path / "missing.json"))
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "VOYAGE_API_KEY",
                "MEMRY_LLM_PROVIDER", "MEMRY_EMBEDDING_PROVIDER"):
        monkeypatch.delenv(key, raising=False)
    seeded = MemoryStore(Config.load())
    seeded.backend.insert_memory(Memory(content="Ada prefers green tea", user_id=None))
    seeded.close()

    assert main(["adopt-unscoped", "--into", "ada", "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["tables"]["memories"] == 1
    assert main(["adopt-unscoped", "--into", "ada"]) == 0
    capsys.readouterr()
    assert main(["adopt-unscoped", "--into", "ada", "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["tables"]["memories"] == 0
    store = MemoryStore(Config.load())
    assert [m.user_id for m in store.get_all(limit=10)] == ["ada"]
    store.close()
