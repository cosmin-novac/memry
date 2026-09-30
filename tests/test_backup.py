"""Lossless, same-namespace backup and restore."""

from __future__ import annotations

from copy import deepcopy

import pytest

from memry.config import Config
from memry.models import (
    Entity,
    EntityMention,
    MergeProposal,
    Relation,
)
from memry.providers.embeddings import HashEmbedder
from memry.providers.llm import NoneLLM
from memry.store import MemoryStore


def make_store() -> MemoryStore:
    return MemoryStore(
        Config(db_path=":memory:"), llm=NoneLLM(), embedder=HashEmbedder(64)
    )


def populated_store() -> MemoryStore:
    store = make_store()
    first = store.add(
        "Marcus studies memory systems.", user_id="ada", infer=False,
        categories=["research"],
    ).actions[0].memory_id
    second = store.add(
        "Helios is a memory project.", user_id="ada", infer=False,
        categories=["research", "projects"],
    ).actions[0].memory_id
    assert first and second
    store.update(first, content="Marcus studies long-term memory systems.")
    store.delete(second)  # invalidation and its audit event must survive

    marcus = store.backend.insert_entity(Entity(
        name="Marcus Vandenberg", normalized="marcus vandenberg", entity_type="person",
        user_id="ada", description="Researches memory systems.",
        description_updated_at="2026-07-24T12:00:00+00:00",
        metadata={"aliases": ["Marcus"]},
    ))
    helios = store.backend.insert_entity(Entity(
        name="Helios", normalized="helios", entity_type="project", user_id="ada"
    ))
    store.backend.add_mention(EntityMention(
        entity_id=marcus.id, memory_id=first, surface="Marcus"
    ))
    store.backend.add_mention(EntityMention(
        entity_id=helios.id, memory_id=second, surface="Helios"
    ))
    store.backend.add_relation(Relation(
        subject=marcus.id, predicate="studies", object=helios.id,
        user_id="ada", memory_id=first,
    ))
    store.backend.add_proposal(MergeProposal(
        entity_a=marcus.id, entity_b=helios.id, user_id="ada",
        status="rejected", reason="different types",
    ))
    return store


def test_backup_restores_exact_knowledge_and_is_idempotent():
    source = populated_store()
    target = make_store()
    try:
        backup = source.export_backup(user_id="ada")
        assert backup["format"] == "memry-backup" and backup["version"] == 1
        assert len(backup["tables"]["episodes"]) == 2
        assert len(backup["tables"]["memories"]) == 2
        assert len(backup["tables"]["memory_events"]) >= 4
        # two names, and each tag a memory is filed under (research twice,
        # projects once) as a mention of its topic entity
        topics = {row["id"] for row in backup["tables"]["entities"]
                  if row["entity_type"] == "topic"}
        mentions = backup["tables"]["entity_mentions"]
        assert len([m for m in mentions if m["entity_id"] not in topics]) == 2
        assert sorted(m["surface"] for m in mentions if m["entity_id"] in topics) == [
            "projects", "research", "research"]
        assert len(backup["tables"]["relations"]) == 1

        result = target.import_backup(backup, owner_prefix="ada")
        assert result["inserted"] > 0 and result["unchanged"] == 0
        restored = target.export_backup(user_id="ada")
        assert restored["scope"] == backup["scope"]
        assert restored["tables"] == backup["tables"]

        again = target.import_backup(backup, owner_prefix="ada")
        assert again["inserted"] == 0
        assert again["unchanged"] == result["inserted"]
    finally:
        source.close()
        target.close()


def test_backup_rejects_other_namespace_and_conflicting_identity():
    source = populated_store()
    target = make_store()
    try:
        backup = source.export_backup(user_id="ada")
        with pytest.raises(ValueError, match="outside this account"):
            target.import_backup(backup, owner_prefix="bob")

        target.import_backup(backup, owner_prefix="ada")
        conflicting = deepcopy(backup)
        conflicting["tables"]["memories"][0]["content"] = "different content"
        with pytest.raises(ValueError, match="conflicts with existing memories"):
            target.import_backup(conflicting, owner_prefix="ada")
        assert target.export_backup(user_id="ada")["tables"] == backup["tables"]
    finally:
        source.close()
        target.close()

def test_a_backup_carries_no_tag_hierarchy_and_one_from_before_restores_without_it():
    """Synthetic parent tags are gone, and with them the tables that held
    them: a backup has no ``topic_relations`` or ``synthetic_tags``, and one
    made before, which has them, restores what it holds besides them."""
    source = populated_store()
    target = make_store()
    try:
        backup = source.export_backup(user_id="ada")
        assert not {"topic_relations", "synthetic_tags"} & set(backup["tables"])
        older = deepcopy(backup)
        older["tables"]["topic_relations"] = [{
            "id": "edge", "broader_topic_id": "parent", "narrower_topic_id": "child",
            "user_id": "ada", "provenance": "synthetic", "created_at": "2026-01-01"}]
        older["tables"]["synthetic_tags"] = [{
            "id": "s1", "tag": "knowledge", "user_id": "ada",
            "source_tags": '["research"]', "created_at": "2026-01-01"}]
        assert target.import_backup(older, owner_prefix="ada")["inserted"] > 0
        assert target.export_backup(user_id="ada")["tables"] == backup["tables"]
    finally:
        source.close()
        target.close()


def test_a_backup_from_before_a_column_was_added_still_restores():
    """Merge decisions gained ``compared_step``; older backups lack it. The
    column's default fills it in, and any other difference is still refused."""
    source = populated_store()
    target = make_store()
    try:
        backup = source.export_backup(user_id="ada")
        older = deepcopy(backup)
        for row in older["tables"]["entity_proposals"]:
            del row["compared_step"]
        assert target.import_backup(older, owner_prefix="ada")["inserted"] > 0
        assert target.import_backup(older, owner_prefix="ada")["inserted"] == 0
        assert target.export_backup(user_id="ada")["tables"] == backup["tables"]

        unknown = deepcopy(backup)
        unknown["tables"]["entity_proposals"][0]["surprise"] = 1
        with pytest.raises(ValueError, match="wrong columns"):
            make_store().import_backup(unknown, owner_prefix="ada")
        missing = deepcopy(backup)
        del missing["tables"]["entity_proposals"][0]["reason"]  # no default: refused
        with pytest.raises(ValueError, match="wrong columns"):
            make_store().import_backup(missing, owner_prefix="ada")
    finally:
        source.close()
        target.close()


def test_a_database_from_before_mentions_kept_what_joined_them_gains_the_column(tmp_path):
    import sqlite3

    from memry.backends.local import LocalBackend

    path = tmp_path / "old.db"
    LocalBackend(str(path)).close()
    db = sqlite3.connect(path)
    db.execute("ALTER TABLE entity_mentions DROP COLUMN decided")
    db.execute("INSERT INTO entity_mentions (id, entity_id, memory_id, surface, created_at) "
               "VALUES ('n1', 'e1', 'm1', 'Quillon', '2026-01-01')")
    db.commit()
    db.close()
    backend = LocalBackend(str(path))
    try:
        [mention] = backend.entity_mentions("e1")
        assert (mention.surface, mention.decided) == ("Quillon", None)
    finally:
        backend.close()


def test_a_database_from_before_the_funnel_gains_its_column(tmp_path):
    import sqlite3

    from memry.backends.local import LocalBackend

    path = tmp_path / "old.db"
    LocalBackend(str(path)).close()
    db = sqlite3.connect(path)
    db.execute("ALTER TABLE entity_proposals DROP COLUMN compared_step")
    db.execute("INSERT INTO entity_proposals (id, entity_a, entity_b, created_at) "
               "VALUES ('p1', 'a', 'b', '2026-01-01')")
    db.commit()
    db.close()
    backend = LocalBackend(str(path))
    try:
        assert backend.get_proposal("p1").compared_step == 0
    finally:
        backend.close()
