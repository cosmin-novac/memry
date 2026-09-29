"""Two builds of one store answer alike.

A bulk import, a benchmark build or a backup restore gives many memories one
timestamp, and a restore writes the rows in another order than they were
first saved. Every ranked read breaks a tie by id (``ORDER BY updated_at
DESC, id`` and the like), so what search returns depends on the store's
content and ids, not on the order its rows were written in."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

from memry.backends.local import LocalBackend
from memry.config import Config
from memry.models import Entity, EntityMention, Memory, MergeProposal, Relation, Scope
from memry.providers.embeddings import HashEmbedder
from memry.providers.llm import NoneLLM
from memry.store import MemoryStore

ROOT = Path(__file__).resolve().parent.parent
STAMP = "2026-06-01T09:00:00+00:00"


def _id(*parts: object) -> str:
    """The same id in every build, in no particular order."""
    return hashlib.sha1(" ".join(map(str, parts)).encode()).hexdigest()[:32]


def _build(world: dict, *, reverse: bool) -> MemoryStore:
    """The dense world as ``relative_retrieval_benchmark.build_store`` stores it
    with oracle links, every record at one time and under the same id in every
    build, written in the world's order or the reverse."""
    sys.path.insert(0, str(ROOT))
    from evals.relative_retrieval_benchmark import USER

    def ordered(items):
        items = list(items)
        return items[::-1] if reverse else items

    store = MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(), embedder=HashEmbedder(64))
    for name, entity_type in ordered(world["types"].items()):
        store.backend.insert_entity(Entity(
            id=_id("entity", name), name=name, normalized=name.lower(),
            entity_type=entity_type, user_id=USER, created_at=STAMP, updated_at=STAMP))
    texts = [m["text"] for m in world["memories"]]
    vectors = dict(zip(texts, store.embedder.embed(texts)))
    for index, m in ordered(enumerate(world["memories"])):
        store.backend.insert_memory(
            Memory(id=_id("memory", index), content=m["text"], user_id=USER,
                   created_at=STAMP, updated_at=STAMP,
                   embedding_model=store.embedder.model_id),
            embedding=vectors[m["text"]])
        for name in m["entities"]:
            store.backend.add_mention(EntityMention(
                id=_id("mention", index, name), entity_id=_id("entity", name),
                memory_id=_id("memory", index), surface=name, created_at=STAMP))
    for subject, predicate, obj in ordered(world["relations"]):
        store.backend.add_relation(Relation(
            id=_id("relation", subject, predicate, obj), subject=_id("entity", subject),
            predicate=predicate, object=_id("entity", obj), user_id=USER, created_at=STAMP))
    for child, parent, kind in ordered(world["pairs"]):
        related = kind in ("version", "occurrence", "component")
        belongs = {"a_kind_of_b": 0.0, "a_part_of_b": 0.0, "b_kind_of_a": 0.0,
                   "b_part_of_a": 0.0, "neither": 0.0 if related else 1.0}
        if related:
            belongs["a_part_of_b" if kind == "component" else "a_kind_of_b"] = 1.0
        store.backend.add_proposal(MergeProposal(
            id=_id("pair", child, parent), entity_a=_id("entity", child),
            entity_b=_id("entity", parent), user_id=USER, confidence=0.0, different=1.0,
            belongs=belongs, compared_step=1, created_at=STAMP))
    store.refresh_property_vectors(user_id=USER)
    return store


@pytest.fixture(scope="module")
def builds():
    sys.path.insert(0, str(ROOT))
    from evals.relative_retrieval_benchmark import USER, build_world_dense

    world = build_world_dense(250)
    first, second = _build(world, reverse=False), _build(world, reverse=True)
    questions = [q for items in world["queries"].values() for q, _, _ in items[:1]]
    yield first, second, questions, USER
    first.close()
    second.close()


@pytest.mark.parametrize("relational", [False, True], ids=["hybrid", "linked"])
def test_two_builds_of_one_world_search_alike(builds, relational):
    """Written in opposite orders, the two builds rank every question the
    same, memory for memory: through the text ranking alone and through the
    linked search, whose family scan reads an entity's newest memories, all
    of one time here."""
    first, second, questions, user = builds
    linked = 0
    for question in questions:
        a = first.search(question, user_id=user, limit=20, relational=relational)
        b = second.search(question, user_id=user, limit=20, relational=relational)
        assert [r.memory.id for r in a] == [r.memory.id for r in b], question
        linked += any("about" in r.signals for r in a)
    assert (linked > 0) == relational  # the linked route ran where it should


def test_entity_memories_of_one_time_read_alike_in_two_builds():
    """Memories of one time, written in opposite orders, come back from
    ``entity_memories`` in one order, by id, and a limit cuts the same ones."""
    def build(order):
        backend = LocalBackend(":memory:")
        backend.insert_entity(Entity(id="e1", name="Bildy", normalized="bildy", user_id="ada"))
        for n in order:
            backend.insert_memory(Memory(id=f"m{n}", content=f"Bildy fact {n}", user_id="ada",
                                         created_at=STAMP, updated_at=STAMP))
            backend.add_mention(EntityMention(entity_id="e1", memory_id=f"m{n}",
                                              surface="Bildy", created_at=STAMP))
        return backend

    ids = [3, 7, 1, 9, 5, 0, 8, 2, 6, 4]
    first, second = build(ids), build(ids[::-1])
    for limit in (10, 4):
        read = [m.id for m in first.entity_memories("e1", limit=limit, scope=Scope(user_id="ada"))]
        assert read == [f"m{n}" for n in range(limit)]
        assert [m.id for m in second.entity_memories("e1", limit=limit)] == read
    assert [m.id for m in first.list_memories(Scope(user_id="ada"), limit=3)] == ["m0", "m1", "m2"]
    assert [m.id for m in second.list_memories(Scope(user_id="ada"), limit=3)] == ["m0", "m1", "m2"]
