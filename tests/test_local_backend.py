from __future__ import annotations

import re
import sqlite3

from memry.backends.local import LocalBackend
from memry.models import Entity, EntityMention, Episode, Memory, MemoryEvent, Scope
from memry.providers.embeddings import HashEmbedder


def make_backend() -> LocalBackend:
    return LocalBackend(":memory:")


def test_insert_get_list_scoping():
    b = make_backend()
    m1 = b.insert_memory(Memory(content="likes coffee", user_id="ada"))
    b.insert_memory(Memory(content="likes tea", user_id="bob"))

    assert b.get_memory(m1.id).content == "likes coffee"
    ada = b.list_memories(Scope(user_id="ada"))
    assert [m.content for m in ada] == ["likes coffee"]
    everyone = b.list_memories(Scope())
    assert len(everyone) == 2


def test_knowledge_map_aggregates_all_memories_without_content():
    backend = make_backend()
    first = backend.insert_memory(
        Memory(
            content="sensitive first memory",
            categories=["Work", "AI"],
            memory_type="semantic",
            user_id="ada",
        )
    )
    second = backend.insert_memory(
        Memory(
            content="sensitive second memory",
            categories=["Work"],
            memory_type="procedural",
            user_id="ada",
        )
    )
    backend.insert_memory(
        Memory(content="other tenant secret", categories=["Private"], user_id="bob")
    )
    ada = backend.insert_entity(
        Entity(name="Ada", entity_type="person", user_id="ada")
    )
    rag = backend.insert_entity(
        Entity(name="RAG", entity_type="concept", user_id="ada")
    )
    for entity, memory in ((ada, first), (rag, first), (ada, second)):
        backend.add_mention(
            EntityMention(
                entity_id=entity.id, memory_id=memory.id, surface=entity.name
            )
        )

    data = backend.knowledge_map(Scope(user_id="ada"))

    assert data["memories"] == 2
    assert data["entity_memories"] == 2
    assert {node["label"]: node["count"] for node in data["tags"]} == {
        "ai": 1,
        "work": 2,
    }
    entities = {node["label"]: node for node in data["entities"]}
    assert entities["Ada"]["count"] == 2
    assert entities["Ada"]["type_counts"] == {"semantic": 1, "procedural": 1}
    assert entities["RAG"]["entity_type"] == "concept"
    assert len(data["tag_edges"]) == 1
    assert {data["tag_edges"][0]["a"], data["tag_edges"][0]["b"]} == {
        "tag:ai", "tag:work"
    }
    assert data["tag_edges"][0]["weight"] == 1
    assert data["entity_edges"][0]["weight"] == 1
    serialized = str(data)
    assert "sensitive" not in serialized
    assert "other tenant" not in serialized


def test_keyword_search_bm25():
    b = make_backend()
    b.insert_memory(Memory(content="Ada prefers TypeScript strict mode", user_id="ada"))
    b.insert_memory(Memory(content="Ada lives in Berlin", user_id="ada"))

    hits = b.keyword_search("typescript", Scope(user_id="ada"))
    assert len(hits) == 1
    assert "TypeScript" in hits[0][0].content


def test_vector_search_ranks_similar_first():
    b = make_backend()
    emb = HashEmbedder(128)
    texts = ["the user lives in berlin germany", "the user has a cat named miso"]
    for text in texts:
        vec = emb.embed([text])[0]
        b.insert_memory(
            Memory(content=text, user_id="ada", embedding_model=emb.model_id), vec
        )
    query = emb.embed(["which city in germany does the user live in"])[0]
    hits = b.vector_search(query, emb.model_id, Scope(user_id="ada"), limit=2)
    assert hits[0][0].content.startswith("the user lives in berlin")
    assert hits[0][1] > hits[1][1]


def test_vector_search_filters_by_embedding_model():
    b = make_backend()
    emb = HashEmbedder(128)
    vec = emb.embed(["hello world"])[0]
    b.insert_memory(Memory(content="hello world", embedding_model="other:model"), vec)
    assert b.vector_search(vec, emb.model_id, Scope()) == []


def test_invalidate_hides_from_default_views():
    b = make_backend()
    m = b.insert_memory(Memory(content="lives in munich", user_id="ada"))
    b.invalidate_memory(m.id, superseded_by="xyz")

    assert b.list_memories(Scope(user_id="ada")) == []
    assert b.keyword_search("munich", Scope(user_id="ada")) == []
    all_rows = b.list_memories(Scope(user_id="ada"), include_invalid=True)
    assert len(all_rows) == 1
    assert all_rows[0].invalid_at is not None
    assert all_rows[0].superseded_by == "xyz"


def test_update_content_keeps_fts_in_sync():
    b = make_backend()
    m = b.insert_memory(Memory(content="works at siemens", user_id="ada"))
    b.update_memory(m.id, content="works at asml")

    assert b.keyword_search("siemens", Scope(user_id="ada")) == []
    assert len(b.keyword_search("asml", Scope(user_id="ada"))) == 1


def test_episodes_and_events():
    b = make_backend()
    b.add_episodes([Episode(content="hello", user_id="ada")])
    assert b.list_episodes(Scope(user_id="ada"))[0].content == "hello"

    b.add_event(MemoryEvent(memory_id="m1", event="ADD", new_content="x"))
    b.add_event(MemoryEvent(memory_id="m1", event="UPDATE", old_content="x", new_content="y"))
    events = b.history("m1")
    assert [e.event for e in events] == ["ADD", "UPDATE"]


def test_stats_and_reset():
    b = make_backend()
    b.insert_memory(Memory(content="a", user_id="ada"))
    m = b.insert_memory(Memory(content="b", user_id="ada"))
    b.invalidate_memory(m.id)
    stats = b.stats()
    assert stats["active_memories"] == 1
    assert stats["invalidated_memories"] == 1
    b.reset()
    assert b.stats()["active_memories"] == 0


def test_a_reset_leaves_no_row_behind_but_the_stores_own_settings():
    """The reset listed the tables it empties by hand, and a table added later
    was left out: property vectors once, then relations, removed entities and
    synthetic tags. Of ``meta`` the migration markers stay, and the settings
    the caller keeps."""
    from memry.models import Relation, SyntheticTag

    b = make_backend()
    quillon, team, stray = (
        b.insert_entity(Entity(name=name, normalized=name.lower(), user_id="ada"))
        for name in ("Quillon", "Harrow team", "stray phrase"))
    memory = b.insert_memory(Memory(content="Quillon is built by the Harrow team", user_id="ada"))
    for entity in (quillon, team, stray):
        b.add_mention(EntityMention(entity_id=entity.id, memory_id=memory.id, surface=entity.name))
    b.add_relation(Relation(subject=quillon.id, predicate="built_by", object=team.id,
                            user_id="ada", memory_id=memory.id))
    b.set_property_vectors({memory.id: [1.0, 0.0]}, "hash-v1", {memory.id: "masked"})
    b.record_synthetic_tag(SyntheticTag(tag="engineering", source_tags=["build"], user_id="ada"))
    assert b.retire_entity(stray.id)
    b.set_meta("upkeep:owner_name:ada", '"Ada"')
    b.set_meta("maintenance:paused", "true")
    marker = b.get_meta("schema:tag-entities:v1")
    b.reset(keep_meta=("maintenance:",))
    assert b.list_relations(Scope(user_id="ada")) == []
    assert b.list_retired_entities(Scope(user_id="ada")) == []
    assert b.list_synthetic_tags(Scope(user_id="ada")) == []
    assert b.property_vector_hashes([memory.id]) == {}
    assert b.list_entities(Scope(user_id="ada"), include_merged=True, kind="any") == []
    assert b.get_meta("upkeep:owner_name:ada") is None
    assert b.get_meta("maintenance:paused") == "true"
    assert marker and b.get_meta("schema:tag-entities:v1") == marker


def test_count_memories_counts_what_a_walk_of_every_memory_would():
    """Stats used to load every memory, and in account mode every account's
    memories up to 100,000, just to count them. The SQL count must give the
    same answer as that walk for every kind of owner, including account names
    with the characters LIKE treats as wildcards."""
    from memry.backends.base import MemoryBackend

    backend = make_backend()
    owners = ["ada::default", "ada::work", "bob::default", "a_b::x", "axb::y",
              "100%::z", "1000::z", "exact-user", None]
    for i, owner in enumerate(owners):
        kept = backend.insert_memory(Memory(content=f"kept {i}", user_id=owner))
        deleted = backend.insert_memory(Memory(content=f"deleted {i}", user_id=owner))
        replaced = backend.insert_memory(Memory(content=f"replaced {i}", user_id=owner))
        backend.invalidate_memory(deleted.id)                          # forgotten
        backend.invalidate_memory(replaced.id, superseded_by=kept.id)  # history, not forgotten
        if i % 2:
            backend.insert_memory(Memory(content=f"extra {i}", user_id=owner))

    everything = backend.list_memories(Scope(), include_invalid=True, limit=10_000)

    def walk(prefix):
        def owned(user_id):
            if prefix is None:
                return True
            if not user_id:
                return False
            return user_id.startswith(prefix) if prefix.endswith("::") else user_id == prefix
        mine = [m for m in everything if owned(m.user_id)]
        return {"active": sum(m.invalid_at is None for m in mine),
                "invalidated": sum(m.invalid_at is not None for m in mine),
                "forgotten": sum(m.invalid_at is not None and not m.superseded_by for m in mine)}

    for prefix in (None, "ada::", "bob::", "a_b::", "100%::", "exact-user", "nobody::", "ada"):
        expected = walk(prefix)
        assert backend.count_memories(prefix) == expected, prefix
        assert MemoryBackend.count_memories(backend, prefix) == expected, prefix

    # its own kept and extra memory; LIKE 'a_b::%' would also count axb::y
    assert backend.count_memories("a_b::")["active"] == 2
    # likewise LIKE '100%::%' would also count 1000::z
    assert backend.count_memories("100%::")["active"] == 2


# ---------------------------------------------- the names, found by any part
def _names_read_one_by_one(backend, words, scope):
    """``entity_names_holding`` as it read the names before their index: every
    name of every entity, each wording of its mentions, the names merged into
    it and its aliases, each read for the words."""
    from memry.backends.local import _kind_clause, _scope_clause

    words = sorted({word.strip().lower() for word in words if word.strip()})
    clause, params = _scope_clause(scope, prefix="e.")
    live = f"e.merged_into IS NULL AND {_kind_clause('named', 'e.')} AND {clause}"

    def holds(value):
        return "(" + " OR ".join(f"instr(lower({value}), ?) > 0" for _ in words) + ")"

    branches = [("e.name", "entities e"),
                ("em.surface", "entity_mentions em JOIN entities e ON e.id = em.entity_id"),
                ("merged.name", "entities merged JOIN entities e ON e.id = merged.merged_into"),
                ("CAST(alias.value AS TEXT)",
                 "entities e, json_each(e.metadata, '$.aliases') alias")]
    sql = " UNION ".join(f"SELECT e.id AS id, {value} AS name FROM {source} "
                         f"WHERE {live} AND {holds(value)}" for value, source in branches)
    rows = backend._db.execute(sql, [v for _ in branches for v in (*params, *words)])
    return sorted((row["id"], row["name"]) for row in rows if row["name"])


NAME_WORDS = ["talkeetna", "keet", "a t", "ta", "x", "jolene", "lodge", "perseid",
              "denali", "münchen", "munchen", "mount talk", "yoga", "shower"]


def _found(backend, scope, words=NAME_WORDS):
    found = sorted(backend.entity_names_holding(words, scope))
    assert found == _names_read_one_by_one(backend, words, scope)
    return found


def test_the_name_index_follows_every_change_of_a_name():
    """A name is found by any part of it, three letters or more through the
    index, as reading every name found it: after it is added, renamed, given
    an alias, merged, and as a mention's wording comes and goes. The index
    holds each name of an entity once, however many mentions use it."""
    b = make_backend()
    ada, bob = Scope(user_id="ada"), Scope(user_id="bob")
    place = b.insert_entity(Entity(name="Mount Talkeetna", entity_type="place", user_id="ada"))
    jolene = b.insert_entity(Entity(name="Jolene", entity_type="person", user_id="ada"))
    shower = b.insert_entity(Entity(name="Perseid shower", entity_type="event", user_id="ada"))
    b.insert_entity(Entity(name="yoga", entity_type="topic", user_id="ada"))
    b.insert_entity(Entity(name="MÜNCHEN Hbf", entity_type="place", user_id="ada"))
    b.insert_entity(Entity(name="München", entity_type="place", user_id="ada"))
    b.insert_entity(Entity(name="Mount Talkeetna", entity_type="place", user_id="bob"))
    memory = b.insert_memory(Memory(content="Jolene did yoga on Mount Talkeetna", user_id="ada"))
    for _ in range(3):
        b.add_mention(EntityMention(entity_id=place.id, memory_id=memory.id,
                                    surface="Talkeetna"))
    found = _found(b, ada)
    assert (place.id, "Talkeetna") in found and (place.id, "Mount Talkeetna") in found
    assert not any(name == "yoga" for _, name in found)  # a tag is no named thing
    assert len(_found(b, bob)) == 1
    assert b._db.execute("SELECT count(*) FROM entity_names WHERE entity_id = ? AND "
                         "name = 'Talkeetna'", (place.id,)).fetchone()[0] == 1

    b.rename_entity(place.id, "Talkeetna Peak")  # the old name stays as an alias
    assert {name for entity_id, name in _found(b, ada) if entity_id == place.id} == {
        "Talkeetna Peak", "Mount Talkeetna", "Talkeetna"}
    b.add_entity_alias(shower.id, "Denali meteor watch")
    assert (shower.id, "Denali meteor watch") in _found(b, ada)

    mention = EntityMention(entity_id=jolene.id, memory_id=memory.id, surface="Jo of the lodge")
    b.add_mention(mention)
    assert (jolene.id, "Jo of the lodge") in _found(b, ada)
    b._db.execute("DELETE FROM entity_mentions WHERE id = ?", (mention.id,))
    assert (jolene.id, "Jo of the lodge") not in _found(b, ada)
    assert b._db.execute("SELECT count(*) FROM entity_names WHERE name = 'Jo of the lodge'"
                         ).fetchone()[0] == 0

    b.add_mention(EntityMention(entity_id=shower.id, memory_id=memory.id, surface="Perseids"))
    b.merge_entities(jolene.id, shower.id)
    found = _found(b, ada)
    assert {name for entity_id, name in found if entity_id == jolene.id} >= {
        "Perseid shower", "Perseids"}
    assert not any(entity_id == shower.id for entity_id, _ in found)
    b.delete_memory(memory.id)
    b.reset()
    assert b._db.execute("SELECT count(*) FROM entity_names").fetchone()[0] == 0
    assert b.entity_names_holding(NAME_WORDS, ada) == []


def test_the_name_index_is_filled_for_a_database_from_before_it(tmp_path):
    """A database from before the index gets it filled when opened, once,
    and finds what reading every name found."""
    from memry.backends.local import _ENTITY_NAMES_MARKER, _ENTITY_NAMES_TRIGGERS

    path = tmp_path / "old.db"
    b = LocalBackend(str(path))
    ada = Scope(user_id="ada")
    place = b.insert_entity(Entity(name="Mount Talkeetna", entity_type="place", user_id="ada",
                                   metadata={"aliases": ["Denali base", 3, None]}))
    lodge = b.insert_entity(Entity(name="Talkeetna Lodge", entity_type="place", user_id="ada"))
    memory = b.insert_memory(Memory(content="a hut below the summit", user_id="ada"))
    for surface in ("Talkeetna", "Talkeetna", "the mountain"):
        b.add_mention(EntityMention(entity_id=place.id, memory_id=memory.id, surface=surface))
    b.merge_entities(place.id, lodge.id)
    words = [*NAME_WORDS, "3"]
    before = _found(b, ada, words)
    b.close()
    db = sqlite3.connect(path)
    for trigger in _ENTITY_NAMES_TRIGGERS:
        db.execute(f"DROP TRIGGER {trigger}")
    db.execute("DROP TABLE entity_names_fts")
    db.execute("DROP TABLE entity_names")
    db.execute("DELETE FROM meta WHERE key = ?", (_ENTITY_NAMES_MARKER,))
    db.commit()
    db.close()
    reopened = LocalBackend(str(path))
    try:
        assert reopened.get_meta(_ENTITY_NAMES_MARKER)
        assert _found(reopened, ada, words) == before
        assert {name for entity_id, name in before if entity_id == place.id} == {
            "Mount Talkeetna", "Talkeetna", "the mountain", "Denali base", "3",
            "Talkeetna Lodge"}
        # each name once: the five of the place, the own name of the lodge merged into it
        assert reopened._db.execute("SELECT count(*) FROM entity_names").fetchone()[0] == 6
    finally:
        reopened.close()


def test_a_database_opened_without_trigrams_keeps_working_and_gets_its_index_back(
        tmp_path, monkeypatch):
    """An SQLite before 3.34 has no trigram tokenizer: opened there, a
    database drops the triggers that would write to the index, finds names
    by reading them, and is written to as before; opened again where the
    tokenizer is, its index is filled with what changed meanwhile."""
    from memry.backends import local

    path = tmp_path / "moved.db"
    ada = Scope(user_id="ada")
    b = LocalBackend(str(path))
    b.insert_entity(Entity(name="Mount Talkeetna", entity_type="place", user_id="ada"))
    b.close()
    monkeypatch.setattr(local.sqlite3, "sqlite_version_info", (3, 31, 1))
    old = LocalBackend(str(path))
    try:
        assert not old._name_index
        assert old._db.execute(
            "SELECT count(*) FROM sqlite_master WHERE type = 'trigger' AND name IN (%s)"
            % ",".join("?" * len(local._ENTITY_NAMES_TRIGGERS)), local._ENTITY_NAMES_TRIGGERS
        ).fetchone()[0] == 0
        assert old.get_meta(local._ENTITY_NAMES_MARKER) is None
        jolene = old.insert_entity(Entity(name="Jolene", entity_type="person", user_id="ada"))
        memory = old.insert_memory(Memory(content="Jolene hiked", user_id="ada"))
        old.add_mention(EntityMention(entity_id=jolene.id, memory_id=memory.id,
                                      surface="Jo of the lodge"))
        old.add_entity_alias(jolene.id, "Denali Jo")
        found = _found(old, ada)
        assert (jolene.id, "Jo of the lodge") in found
    finally:
        old.close()
    monkeypatch.undo()
    new = LocalBackend(str(path))
    try:
        assert new._name_index
        assert _found(new, ada) == found
    finally:
        new.close()


# ------------------------------------------------------- the keyword search
def test_the_keyword_search_reads_a_word_with_accents_whole():
    """"München" is one word, as the full-text index reads it, written with
    its accent on the letter or after it; read as ASCII it was "M" and
    "nchen", and matched nothing."""
    b = make_backend()
    munich = b.insert_memory(Memory(content="Ada moved to München in May", user_id="ada"))
    b.insert_memory(Memory(content="Ada moved to Berlin in June", user_id="ada"))
    for question in ("Where is München?", "Where is München?", "munchen"):
        hits = b.keyword_search(question, Scope(user_id="ada"))
        assert [m.id for m, _ in hits] == [munich.id], question


def _scored_everywhere(backend, query, scope, limit, *, include_invalid=False,
                       categories=None, entity_id=None, history=False):
    """The keyword search as it scored before it pruned: every word in every
    memory in scope that holds it."""
    from collections import Counter

    from memry.backends import local

    asked = Counter(w.lower() for w in local._WORD_RE.findall(query)[:32])
    clause, params = local._search_scope_clause(scope, "m")
    cat_clause, cat_params = local._category_clause(categories, "m.id")
    entity_clause, entity_params = local._entity_clause(entity_id, "m.id")
    if not include_invalid:
        clause += (f" AND (m.invalid_at IS NULL OR ({local._history_clause('m')}))"
                   if history else " AND m.invalid_at IS NULL")
    weights = backend._word_weights(list(asked))
    scores = {}
    for word, times in asked.items():
        for row in backend._db.execute(
                "SELECT m.id AS id, bm25(memories_fts) AS rank_score FROM memories_fts "
                "CROSS JOIN memories m ON m.rowid = memories_fts.rowid WHERE memories_fts "
                f"MATCH ? AND {clause} AND {cat_clause} AND {entity_clause}",
                (f'"{word}"', *params, *cat_params, *entity_params)):
            scores[row["id"]] = (scores.get(row["id"], 0.0)
                                 - times * weights[word] * float(row["rank_score"]))
    best = sorted(scores, key=lambda memory_id: (-scores[memory_id], memory_id))[:limit]
    return [(memory_id, scores[memory_id]) for memory_id in best]


def test_the_keyword_search_finds_the_memories_scoring_every_word_everywhere_finds(
        monkeypatch):
    """The first memories of the keyword search are those scoring every word
    of the question in every memory gives, with the same scores, whichever
    words are read as common and whatever the filters: 600 memories of two
    people and two runs, most with the turn they were said in, seven words in
    about a third of them each and sixty in fewer, memories said twice
    (ties), memories kept as history and out of use."""
    import random

    from memry.backends import local

    rng = random.Random(5)
    b = make_backend()
    common = ["the", "did", "we", "at", "to", "and", "was"]
    rare = [f"word{i}" for i in range(60)]
    person = b.insert_entity(Entity(name="Harlow", entity_type="person", user_id="ada"))
    texts, memories = [], []
    for i in range(600):
        if texts and rng.random() < 0.1:
            text = rng.choice(texts)  # said twice: a tie
        else:
            text = " ".join(rng.sample(common, rng.randint(0, 5))
                            + [rng.choice(rare[:rng.randint(1, 60)])
                               for _ in range(rng.randint(1, 4))])
        texts.append(text)
        memory = b.insert_memory(Memory(
            content=text, user_id=rng.choice(["ada", "ada", "bob"]),
            run_id=rng.choice(["r1", "r2"]),
            categories=[rng.choice(["travel", "work"])] if rng.random() < 0.3 else []))
        if rng.random() < 0.7:
            turn = Episode(content=" ".join(rng.sample(common, 3)) + " " + text,
                           user_id=memory.user_id, run_id=memory.run_id)
            b.add_episodes([turn])
            b.update_memory(memory.id, source_episode_ids=[turn.id], touch=False)
        if rng.random() < 0.2:
            b.add_mention(EntityMention(entity_id=person.id, memory_id=memory.id,
                                        surface="Harlow"))
        memories.append(memory)
    for old, new in zip(memories[:40], memories[40:80]):
        if old.user_id == new.user_id:
            b.add_event(MemoryEvent(memory_id=old.id, event="SUPERSEDE", kind="update"))
        b.invalidate_memory(old.id, superseded_by=new.id)
    questions = [" ".join(rng.sample(common, rng.randint(0, 6))
                          + rng.sample(rare, rng.randint(0, 3))) + "?" for _ in range(25)]
    questions += ["the the did we?", "word3 word3 the", "nothing here", "the"]
    filters = [{}, {"categories": ["travel"]}, {"entity_id": person.id}, {"history": True},
               {"include_invalid": True}]
    scopes = [Scope(user_id="ada"), Scope(user_id="ada", run_id="r2"), Scope()]
    # each scope with each filter in turn
    kinds = [(scope, kept) for scope in scopes for kept in filters]
    turn = 0
    for rows in (0, 30, 200):
        monkeypatch.setattr(local, "_WHOLE_WORD_ROWS", rows)
        for question in questions:
            for limit in (1, 3, 10, 40, 700):
                scope, kept = kinds[turn % len(kinds)]
                turn += 1
                got = [(m.id, s) for m, s in b.keyword_search(question, scope, limit, **kept)]
                assert got == _scored_everywhere(b, question, scope, limit, **kept), (
                    rows, question, limit, scope, kept)


# ------------------------------------------ the work grows with the question
GROWN_QUESTIONS = ["When did Harlow do yoga on Arvel with the kids?",
                   "What did the zebra say?"]


def _grown(extra: int, others: int = 0) -> LocalBackend:
    """Harlow, who did yoga on Mount Arvel with the kids, and ``extra`` of
    everything else that shares no word with the questions: entities with an
    alias each, a memory naming each, and the turn it was said in. With
    ``others``, as many of each of another account, bob's, half of them
    saying the questions' words: people named Harlow too, places whose name
    holds "Arvel", "Harlow did yoga with the kids on Arvel" said and saved."""
    b = make_backend()
    harlow = b.insert_entity(Entity(name="Harlow", entity_type="person", user_id="ada",
                                    metadata={"aliases": ["Harry"]}))
    arvel = b.insert_entity(Entity(name="Mount Arvel", entity_type="place", user_id="ada"))
    hut = b.insert_entity(Entity(name="Arvel hut", entity_type="place", user_id="ada"))
    b.merge_entities(arvel.id, hut.id)
    for text, named in [("Harlow did yoga on top of Mount Arvel with the kids", [harlow, arvel]),
                        ("Mount Arvel has a hut below the summit", [arvel]),
                        ("Harlow did yoga in the park", [harlow]),
                        ("the kids did their homework", [])]:
        turn = Episode(content=f"We said: {text}.", user_id="ada")
        b.add_episodes([turn])
        memory = b.insert_memory(Memory(content=text, user_id="ada",
                                        source_episode_ids=[turn.id]))
        for entity in named:
            b.add_mention(EntityMention(entity_id=entity.id, memory_id=memory.id,
                                        surface=entity.name))
    for i in range(extra):
        other = b.insert_entity(Entity(name=f"Qz {i:05d}", entity_type="thing", user_id="ada",
                                       metadata={"aliases": [f"Qy {i:05d}"]}))
        turn = Episode(content=f"qz{i} said zucchini {i}", user_id="ada")
        b.add_episodes([turn])
        memory = b.insert_memory(Memory(content=f"qz{i} bought {i} zucchini", user_id="ada",
                                        source_episode_ids=[turn.id]))
        b.add_mention(EntityMention(entity_id=other.id, memory_id=memory.id,
                                    surface=other.name))
    for i in range(others):
        name = ["Harlow", f"Arvel Ridge {i:05d}", f"Qz bob {i:05d}"][i % 3]
        entity = b.insert_entity(Entity(name=name, entity_type="place", user_id="bob"))
        text = (f"Harlow {i} did yoga with the kids on Arvel, the zebra said" if i % 2
                else f"bob{i} bought {i} zucchini")
        turn = Episode(content=f"We said: {text}.", user_id="bob")
        b.add_episodes([turn])
        memory = b.insert_memory(Memory(content=text, user_id="bob",
                                        source_episode_ids=[turn.id]))
        b.add_mention(EntityMention(entity_id=entity.id, memory_id=memory.id,
                                    surface=name))
    # each full-text index in one segment, as its merges leave it in time
    # (how many a build leaves depends on the order of its writes)
    for (index,) in b._db.execute(
            "SELECT name FROM sqlite_master WHERE sql LIKE 'CREATE VIRTUAL TABLE%fts5%'"
    ).fetchall():
        b._db.execute(f"INSERT INTO {index}({index}) VALUES ('optimize')")
    return b


def _steps(backend: LocalBackend, call) -> int:
    """The SQLite virtual machine instructions ``call`` runs: a count of work
    that no clock or machine changes."""
    steps = 0

    def count() -> int:
        nonlocal steps
        steps += 1
        return 0

    backend._db.set_progress_handler(count, 1)
    try:
        call()
    finally:
        backend._db.set_progress_handler(None, 1)
    return steps


def test_the_lookups_of_a_search_grow_with_the_question_not_with_the_store(monkeypatch):
    """Finding the entities a question names and its keyword matches reads
    the question's matches, not the store: with four times the entities,
    aliases, memories and turns, none of which the questions match, both do
    the same work, counted in SQLite instructions, the keyword search also
    with every word read as common. Reading every entity's names for a word
    of the question (``entity_names_holding`` before its index) did about
    four times as much."""
    from memry.backends import local
    from memry.intelligence.graph_retrieval import detect_query_entities

    scope = Scope(user_id="ada")
    small, large = _grown(200), _grown(800)
    harlow, arvel = (small.find_entities_by_aliases([name], scope)[0].name
                     for name in ("harlow", "mount arvel"))
    for question, named, matches in [(GROWN_QUESTIONS[0], [harlow, arvel], 4),
                                     (GROWN_QUESTIONS[1], [], 4)]:
        for backend in (small, large):
            assert [backend.get_entity(entity_id).name for entity_id in
                    detect_query_entities(backend, scope, question)] == named
            assert len(backend.keyword_search(question, scope, 40)) == matches
        for rows in (200, 0):
            monkeypatch.setattr(local, "_WHOLE_WORD_ROWS", rows, raising=False)
            for work in (lambda backend: detect_query_entities(backend, scope, question),
                         lambda backend: backend.keyword_search(question, scope, 40)):
                few = _steps(small, lambda: work(small))
                many = _steps(large, lambda: work(large))
                assert many <= few * 1.05, (question, rows, few, many)


def _scored(backend: LocalBackend, call) -> int:
    """How many memories ``bm25()`` scores in ``call``: each reads the length
    of the memory it scores, one read of the index's lengths a memory."""
    reads = 0

    def trace(sql: str) -> None:
        nonlocal reads
        reads += "SELECT sz FROM 'main'.'memories_fts_docsize'" in sql

    backend._db.set_trace_callback(trace)
    try:
        call()
    finally:
        backend._db.set_trace_callback(None)
    return reads


def test_a_search_scores_none_of_another_accounts_memories(monkeypatch):
    """The full-text indexes hold every account's memories and turns, and the
    index of names every account's names. With another account four times
    as large, half of what it says in the questions' words, the searched
    account's keyword search scores no more memories, every word read as
    rare or as common. With the common words read together it walks the
    other account's rows about as the one query of fe094fc did (2.5 SQLite
    instructions for each of its memories, turns and entities there), and
    the name lookup reads none of the other account's names."""
    from memry.backends import local

    scope = Scope(user_id="ada")
    small, large = _grown(20, 200), _grown(20, 800)
    added = 600
    for question in GROWN_QUESTIONS:
        forms = sorted(set(re.findall(r"[^\W_]{3,}", question.lower())))
        assert (_steps(small, lambda: small.entity_names_holding(forms, scope))
                == _steps(large, lambda: large.entity_names_holding(forms, scope)))
        for rows in (10**6, 0):
            monkeypatch.setattr(local, "_WHOLE_WORD_ROWS", rows)
            few = _scored(small, lambda: small.keyword_search(question, scope, 40))
            many = _scored(large, lambda: large.keyword_search(question, scope, 40))
            assert few == many, (question, rows, few, many)
            if rows == 0:
                few = _steps(small, lambda: small.keyword_search(question, scope, 40))
                many = _steps(large, lambda: large.keyword_search(question, scope, 40))
                assert many - few <= 3 * added, (question, few, many)
