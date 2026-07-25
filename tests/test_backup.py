"""Lossless, same-namespace backup and restore."""

from __future__ import annotations

from copy import deepcopy

import pytest

from memry.config import Config
from memry.models import (
    Collection,
    Entity,
    EntityMention,
    MergeProposal,
    Relation,
    Scope,
    SyntheticTag,
    TopicRelation,
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
        "Cosmin studies memory systems.", user_id="ada", infer=False,
        categories=["research"],
    ).actions[0].memory_id
    second = store.add(
        "Helios is a memory project.", user_id="ada", infer=False,
        categories=["research", "projects"],
    ).actions[0].memory_id
    assert first and second
    store.update(first, content="Cosmin studies long-term memory systems.")
    store.delete(second)  # invalidation and its audit event must survive

    cosmin = store.backend.insert_entity(Entity(
        name="Cosmin Novac", normalized="cosmin novac", entity_type="person",
        user_id="ada", description="Researches memory systems.",
        description_updated_at="2026-07-24T12:00:00+00:00",
        metadata={"aliases": ["Cosmin"]},
    ))
    helios = store.backend.insert_entity(Entity(
        name="Helios", normalized="helios", entity_type="project", user_id="ada"
    ))
    store.backend.add_mention(EntityMention(
        entity_id=cosmin.id, memory_id=first, surface="Cosmin"
    ))
    store.backend.add_mention(EntityMention(
        entity_id=helios.id, memory_id=second, surface="Helios"
    ))
    store.backend.add_relation(Relation(
        subject=cosmin.id, predicate="studies", object=helios.id,
        user_id="ada", memory_id=first,
    ))
    store.backend.add_proposal(MergeProposal(
        entity_a=cosmin.id, entity_b=helios.id, user_id="ada",
        status="rejected", reason="different types",
    ))
    store.backend.record_synthetic_tag(SyntheticTag(
        tag="knowledge", source_tags=["research", "projects"], user_id="ada"
    ))
    topics = store.backend.list_topics(Scope(user_id="ada"))
    store.backend.add_topic_relation(TopicRelation(
        broader_topic_id=topics[0].id, narrower_topic_id=topics[1].id,
        user_id="ada", provenance="manual",
    ))
    store.backend.record_collection(Collection(
        title="Memory research", summary="Current research work.",
        memory_ids=[first, second], user_id="ada",
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
        assert len(backup["tables"]["entity_mentions"]) == 2
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