"""Entity questions (``retrieval.entity_questions``): the questions the owner
would ask about an entity related to them, by its role ("Where does my sister
work?"). With the flag on, a question by role is about the entity whose
questions alone contain its role word, at or above the bar, and is read with
"my sister" as "it". With it off, or for a question in the first person
about the owner, the search starts from the owner as before. Offline: hash
vectors and a scripted model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from memry.config import Config
from memry.intelligence.entity_questions import (
    distinct_holders,
    mask_role,
    role_words,
    write_entity_questions,
)
from memry.models import Entity, EntityMention, Memory, Relation, Scope
from memry.providers.embeddings import HashEmbedder
from memry.store import MemoryStore

from conftest import FakeLLM

ROOT = Path(__file__).resolve().parent.parent
USER = "ilva"
SCOPE = Scope(user_id=USER)


# ------------------------------------------------------------ the rule


def test_role_words_are_the_words_after_my():
    assert role_words("Where does my sister work?") == {"sister"}
    assert role_words("When is my sister's birthday?") == {"sister"}
    assert role_words("Where does the user's manager live?") == {"manager"}
    assert role_words("Where do I work?") == set()
    assert role_words("Who leads the Kaven planner team?") == set()


def test_mask_role_reads_the_role_as_it():
    assert mask_role("Where does my sister work?", "sister") == "Where does it work?"
    assert mask_role("When is my sister's birthday?", "sister") == "When is its birthday?"
    assert mask_role("Where does my brother work?", "sister") == "Where does my brother work?"


def test_a_role_word_two_entities_use_tells_neither_apart():
    rows = [("e1", "Who is my sister?"), ("e2", "Who is my colleague?"),
            ("e3", "Where does my colleague sit?")]
    assert distinct_holders(rows) == {"sister": "e1"}


# ------------------------------------------------------------ the search


def _store(flag: bool = True) -> tuple[MemoryStore, dict[str, str], dict[str, str]]:
    cfg = Config(db_path=":memory:")
    cfg.retrieval.entity_questions = flag
    cfg.retrieval.relational_relevance = "vector"
    store = MemoryStore(cfg, llm=FakeLLM(), embedder=HashEmbedder(64))
    people = {"owner": "Ilva Marsh", "sister": "Mira Lund", "brother": "Kai Berg"}
    ids = {}
    for role, name in people.items():
        ids[role] = store.backend.insert_entity(Entity(
            name=name, normalized=name.lower(), entity_type="person", user_id=USER)).id
    store._upkeep_set("owner_entity", USER, ids["owner"])
    texts = {
        "owner_work": ("Ilva Marsh works at Nordlicht GmbH.", ["owner"]),
        "owner_city": ("Ilva Marsh lives in Lisbon.", ["owner"]),
        "sister_is": ("Mira Lund is Ilva Marsh's sister.", ["owner", "sister"]),
        "sister_work": ("Mira Lund works at Kestrel Labs.", ["sister"]),
        "sister_dog": ("Mira Lund has a dog called Rufus.", ["sister"]),
        "brother_is": ("Kai Berg is Ilva Marsh's brother.", ["owner", "brother"]),
        "brother_work": ("Kai Berg works at Torvik Energy.", ["brother"]),
    }
    mids = {}
    for key, (text, about) in texts.items():
        memory = store.backend.insert_memory(
            Memory(content=text, user_id=USER, embedding_model=store.embedder.model_id),
            embedding=store.embedder.embed([text])[0])
        mids[key] = memory.id
        for n, role in enumerate(about):
            store.backend.add_mention(EntityMention(
                id=f"{memory.id}.{n}", entity_id=ids[role], memory_id=memory.id,
                surface=people[role]))
    for role in ("sister", "brother"):
        store.backend.add_relation(Relation(subject=ids["owner"], predicate=f"has_{role}",
                                            object=ids[role], user_id=USER))
    store.refresh_property_vectors(user_id=USER)
    store._write_entity_questions(ids["sister"], [
        "Who is my sister?", "Where does my sister work?"], "model")
    store._write_entity_questions(ids["brother"], [
        "Who is my brother?", "Where does my brother work?"], "model")
    return store, ids, mids


def test_a_question_by_role_starts_from_the_role_entity_and_its_answer_comes_first():
    store, ids, mids = _store()
    plan = store._plan("Where does my sister work?", _reads(store), True)
    assert plan.seeds == [ids["sister"]]
    assert plan.question == "Where does it work?"
    found = store.search("Where does my sister work?", user_id=USER, limit=3)
    assert found[0].memory.id == mids["sister_work"]
    store.close()


def test_off_the_question_by_role_starts_from_the_owner():
    store, ids, mids = _store(flag=False)
    plan = store._plan("Where does my sister work?", _reads(store), True)
    assert plan.seeds == [ids["owner"]] and plan.first_person
    store.close()


def test_the_owners_own_question_keeps_the_owner():
    store, ids, _ = _store()
    plan = store._plan("Where do I work?", _reads(store), True)
    assert plan.seeds == [ids["owner"]] and plan.first_person
    store.close()


def test_a_question_naming_a_hub_is_about_that_hub():
    store, ids, _ = _store()
    plan = store._plan("Where does Kai Berg work?", _reads(store), True)
    assert plan.seeds == [ids["brother"]]
    store.close()


def test_below_the_bar_the_question_is_about_the_owner():
    store, ids, _ = _store()
    store.config.retrieval.entity_question_bar = 1.01
    plan = store._plan("Where does my sister work?", _reads(store), True)
    assert plan.seeds == [ids["owner"]]
    store.close()


def test_a_role_word_shared_by_two_entities_seeds_neither():
    store, ids, _ = _store()
    store._write_entity_questions(ids["brother"], ["Who is my sister?"], "model")
    plan = store._plan("Where does my sister work?", _reads(store), True)
    assert plan.seeds == [ids["owner"]]
    store.close()


def test_a_merged_entitys_questions_are_not_read():
    store, ids, _ = _store()
    with store.backend._lock:
        store.backend._db.execute("UPDATE entities SET merged_into = ? WHERE id = ?",
                                  (ids["brother"], ids["sister"]))
    assert {row[0] for row in store.backend.entity_question_rows(SCOPE, "x")} == {ids["brother"]}
    store.close()


# ------------------------------------------- backups, removal and merges


def _texts(store, entity_id):
    return [q["text"] for q in store.backend.entity_questions_of([entity_id]).get(entity_id, [])]


def test_a_backup_carries_the_entity_question_texts_and_a_restore_embeds_them():
    store, ids, _ = _store()
    backup = store.export_backup(user_id=USER)
    rows = backup["tables"]["entity_questions"]
    assert {r["entity_id"] for r in rows} == {ids["sister"], ids["brother"]}
    assert all("embedding" not in r and "embedding_model" not in r for r in rows)
    json.dumps(backup)  # a backup is plain JSON
    store.close()

    fresh = MemoryStore(Config(db_path=":memory:"), llm=FakeLLM(), embedder=HashEmbedder(64))
    fresh.backend.import_backup(backup)
    assert _texts(fresh, ids["sister"]) == ["Who is my sister?", "Where does my sister work?"]
    assert len(fresh.backend.entity_questions_without_vectors(
        SCOPE, fresh.embedder.model_id)) == 4
    fresh.close()

    cfg = Config(db_path=":memory:")
    cfg.retrieval.entity_questions = True
    keyed = MemoryStore(cfg, llm=FakeLLM(), embedder=HashEmbedder(64))
    keyed.import_backup(backup)
    assert keyed.backend.entity_questions_without_vectors(SCOPE, keyed.embedder.model_id) == []
    keyed.close()


def test_an_older_backup_without_the_entity_questions_table_restores():
    store, ids, _ = _store()
    backup = store.export_backup(user_id=USER)
    store.close()
    older = {**backup, "tables": {k: v for k, v in backup["tables"].items()
                                  if k != "entity_questions"}}
    fresh = MemoryStore(Config(db_path=":memory:"), llm=FakeLLM(), embedder=HashEmbedder(64))
    fresh.import_backup(older)
    assert fresh.backend.get_entity(ids["sister"]) is not None
    assert fresh.backend.entity_questions_of([ids["sister"]]) == {}
    fresh.close()


def test_a_backup_entity_question_of_an_entity_outside_it_is_refused():
    store, _, _ = _store()
    backup = store.export_backup(user_id=USER)
    store.close()
    backup["tables"]["entity_questions"] = [
        {"entity_id": "nowhere", "n": 0, "text": "Who is my aunt?", "source": "model"}]
    fresh = MemoryStore(Config(db_path=":memory:"), llm=FakeLLM(), embedder=HashEmbedder(64))
    with pytest.raises(ValueError, match="entity question"):
        fresh.import_backup(backup)
    fresh.close()


def test_a_removed_entitys_questions_go_and_a_restore_brings_them_back():
    store, ids, _ = _store()
    assert store.backend.retire_entity(ids["sister"])
    assert store.backend.entity_questions_of([ids["sister"]]) == {}
    assert store.restore_entities([ids["sister"]]) == 1
    assert _texts(store, ids["sister"]) == ["Who is my sister?", "Where does my sister work?"]
    # the trash keeps the texts; the store embeds them again
    assert store.backend.entity_questions_without_vectors(SCOPE, store.embedder.model_id) == []
    plan = store._plan("Where does my sister work?", _reads(store), True)
    assert plan.seeds == [ids["sister"]]
    store.close()


def test_a_deleted_entitys_questions_go_with_it():
    store, ids, _ = _store()
    assert store.backend.delete_entity(ids["brother"])
    assert store.backend.entity_questions_of([ids["brother"]]) == {}
    store.close()


def test_a_merged_entitys_questions_stay_with_its_tombstone_until_the_merge_is_undone():
    store, ids, _ = _store()
    assert store.merge_entities(ids["brother"], ids["sister"])
    assert {row[0] for row in store.backend.entity_question_rows(SCOPE, "x")} == {ids["brother"]}
    assert len(_texts(store, ids["sister"])) == 2
    store.undo_merge(ids["sister"])
    assert {row[0] for row in store.backend.entity_question_rows(SCOPE, "x")} == {
        ids["sister"], ids["brother"]}
    # removing the kept entity takes the questions of what was merged into it
    assert store.merge_entities(ids["brother"], ids["sister"])
    assert store.backend.delete_entity(ids["brother"])
    assert store.backend.entity_questions_of([ids["sister"], ids["brother"]]) == {}
    store.close()


def _reads(store):
    from memry.store import _Reads

    return _Reads(SCOPE)


# ------------------------------------------------------------ the writer


def test_the_writer_reads_the_answer_and_drops_names_and_questions_without_a_role():
    llm = FakeLLM([json.dumps({"items": [
        {"n": 1, "questions": ["Who is my sister?", "Where does Mira Lund work?",
                               "What car is it?", "Where does my sister work?"]},
        {"n": 7, "questions": ["Who is my cousin?"]}]})])
    got = write_entity_questions(llm, "Ilva Marsh", [
        {"name": "Mira Lund", "description": "Ilva's sister, works at Kestrel Labs.",
         "relations": ["Ilva Marsh has_sister Mira Lund"]}])
    assert got == [["Who is my sister?", "Where does my sister work?"]]
    system, user = llm.calls[0]
    assert "Ilva Marsh" in system and "has_sister" in user and "Kestrel Labs" in user


def test_write_entity_questions_asks_about_described_relations_of_the_owner_once():
    store, ids, _ = _store()
    store.backend.set_entity_questions(ids["sister"], [])
    store.backend.set_entity_questions(ids["brother"], [])
    store.backend.set_entity_description(ids["sister"], "Ilva's sister.", "2026-10-09T00:00:00Z")
    store.llm = FakeLLM([json.dumps({"items": [
        {"n": 1, "questions": ["Who is my sister?", "Which company employs my sister?"]}]})])
    assert store.write_entity_questions(user_id=USER) == {"checked": 1, "written": 1}
    assert "Mira Lund" in store.llm.calls[0][1] and "Kai Berg" not in store.llm.calls[0][1]
    rows = store.backend.entity_questions_of([ids["sister"]])[ids["sister"]]
    assert [r["text"] for r in rows] == ["Who is my sister?", "Which company employs my sister?"]
    assert all(r["embedding_model"] == store.embedder.model_id for r in rows)
    assert store.write_entity_questions(user_id=USER) == {"checked": 0, "written": 0}
    store.close()


# ------------------------------------------------------------ the benchmark family


@pytest.fixture(scope="module")
def bench():
    sys.path.insert(0, str(ROOT))
    from evals import relative_retrieval_benchmark as module

    return module


def test_the_role_family_is_the_same_for_the_same_seed_and_leaves_the_world_as_it_was(bench):
    plain = bench.build_world_dense(60, owner=True)
    a = bench.build_world_dense(60, owner=True, roles=True)
    b = bench.build_world_dense(60, owner=True, roles=True)
    assert a["queries"]["role"] == b["queries"]["role"]
    assert a["entity_questions"] == b["entity_questions"]
    assert len(a["queries"]["role"]) == 24
    n = len(plain["memories"])
    assert a["memories"][:n] == plain["memories"]
    assert {f: q for f, q in a["queries"].items() if f != "role"} == plain["queries"]
    for question, gold, wrong in a["queries"]["role"]:
        (word,) = bench.role_words(question)
        about = a["memories"][gold[0]]["entities"]
        assert len(about) == 1 and (bench.OWNER, f"has_{word}", about[0]) in a["relations"]
        assert wrong and gold[0] not in wrong
        assert bench.OWNER not in question
    with pytest.raises(ValueError):
        bench.build_world_dense(60, roles=True)
