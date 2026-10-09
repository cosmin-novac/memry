"""Local SQLite backend - the default, zero-service storage engine.

One file holds everything: raw episodes, derived memories, an FTS5 index
(BM25 keyword search), float32 embeddings (brute-force cosine via numpy -
fast enough into the hundreds of thousands of memories), and the full event
history. WAL mode + a process-wide lock make it safe for the MCP/REST servers.
"""

from __future__ import annotations

import base64
import contextlib
import heapq
import json
import math
import re
import sqlite3
import threading
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from ..config import AnnConfig
from ..models import (
    Entity,
    EntityMention,
    Episode,
    Memory,
    MemoryEvent,
    MergeProposal,
    Relation,
    ENTITY_TYPES,
    HISTORY_KINDS,
    TOPIC_TYPE,
    TYPE_SET_BY_OWNER,
    Scope,
    Topic,
    later_ts,
    new_id,
    utcnow,
)
from .ann import HAS_USEARCH, HnswSidecar
from .base import MemoryBackend

_SCHEMA = """
-- withheld_at: when a memory resting on the episode was deleted for good; from
-- then on the episode is never shown as evidence (``evidence_episodes``). The
-- embedding is what evidence is chosen by, stored as a memory's is. name: the
-- speaker's name a message gave besides its role (NULL: shown by its role). Each
-- column defaults to NULL, so a backup from before it restores.
CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'user',
    name TEXT DEFAULT NULL,
    user_id TEXT,
    agent_id TEXT,
    run_id TEXT,
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    withheld_at TEXT DEFAULT NULL,
    embedding BLOB DEFAULT NULL,
    embedding_model TEXT DEFAULT NULL
);
CREATE INDEX IF NOT EXISTS idx_episodes_scope ON episodes(user_id, agent_id, run_id);
CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
    content, content='episodes', content_rowid='rowid'
);
CREATE TRIGGER IF NOT EXISTS episodes_ai AFTER INSERT ON episodes BEGIN
    INSERT INTO episodes_fts(rowid, content) VALUES (new.rowid, new.content);
END;
CREATE TRIGGER IF NOT EXISTS episodes_ad AFTER DELETE ON episodes BEGIN
    INSERT INTO episodes_fts(episodes_fts, rowid, content)
    VALUES ('delete', old.rowid, old.content);
END;

CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    memory_type TEXT NOT NULL DEFAULT 'semantic',
    user_id TEXT,
    agent_id TEXT,
    run_id TEXT,
    importance REAL NOT NULL DEFAULT 0.5,
    categories TEXT NOT NULL DEFAULT '[]',
    entities TEXT NOT NULL DEFAULT '[]',
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    valid_from TEXT,
    invalid_at TEXT,
    superseded_by TEXT,
    source_episode_ids TEXT NOT NULL DEFAULT '[]',
    embedding BLOB,
    embedding_model TEXT
);
CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(user_id, agent_id, run_id);

-- A memory with its own entity names replaced by "it", embedded: what it says
-- about whatever it is about. Derived, like the ANN index, so not in backups.
-- masked_hash identifies the masked text, so a refresh re-embeds only what
-- changed.
CREATE TABLE IF NOT EXISTS memory_property_vectors (
    memory_id TEXT PRIMARY KEY,
    embedding BLOB NOT NULL,
    embedding_model TEXT,
    masked_hash TEXT
);
-- The questions a memory answers (``intelligence.questions``), kept as
-- search keys beside it: one row per question, with its vector in float16,
-- cut to ``retrieval.property_dimensions``. The texts are in a backup (they
-- cost model calls); the vectors are derived and filled in again after a
-- restore (``MemoryStore.refresh_question_vectors``). source: "save" (the
-- extractor), "backfill", "agent" (sent with the save) or "night".
CREATE TABLE IF NOT EXISTS memory_questions (
    memory_id TEXT NOT NULL,
    n INTEGER NOT NULL,
    text TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'save',
    embedding BLOB DEFAULT NULL,
    embedding_model TEXT DEFAULT NULL,
    PRIMARY KEY (memory_id, n)
);
CREATE VIRTUAL TABLE IF NOT EXISTS memory_questions_fts USING fts5(
    text, content='memory_questions', content_rowid='rowid'
);
CREATE TRIGGER IF NOT EXISTS memory_questions_ai AFTER INSERT ON memory_questions BEGIN
    INSERT INTO memory_questions_fts(rowid, text) VALUES (new.rowid, new.text);
END;
CREATE TRIGGER IF NOT EXISTS memory_questions_ad AFTER DELETE ON memory_questions BEGIN
    INSERT INTO memory_questions_fts(memory_questions_fts, rowid, text)
    VALUES ('delete', old.rowid, old.text);
END;
CREATE TRIGGER IF NOT EXISTS memory_questions_au AFTER UPDATE OF text ON memory_questions BEGIN
    INSERT INTO memory_questions_fts(memory_questions_fts, rowid, text)
    VALUES ('delete', old.rowid, old.text);
    INSERT INTO memory_questions_fts(rowid, text) VALUES (new.rowid, new.text);
END;
-- The search log (``retrieval.search_log``): one row per search, kept
-- ``SEARCH_LOG_DAYS`` days. mode: "linked" (the question was about a known
-- entity: seeds > 0), "text" (about none) or "browse" (no query text).
-- best_judged: the judge's highest relevance of the judged pool (NULL when
-- the search was not judged). keyed: how many memories took the query as a
-- question key (``MemoryStore._traffic_keys``). Never in a backup unless
-- asked: it contains what people asked.
CREATE TABLE IF NOT EXISTS search_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    at TEXT NOT NULL,
    user_id TEXT,
    agent_id TEXT,
    run_id TEXT,
    query TEXT NOT NULL,
    mode TEXT NOT NULL,
    seeds INTEGER NOT NULL DEFAULT 0,
    first_person INTEGER NOT NULL DEFAULT 0,
    filtered INTEGER NOT NULL DEFAULT 0,
    judged INTEGER NOT NULL DEFAULT 0,
    best_judged REAL DEFAULT NULL,
    results INTEGER NOT NULL DEFAULT 0,
    latency_ms REAL NOT NULL DEFAULT 0,
    keyed INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_search_log_ns ON search_log(user_id, at);
CREATE INDEX IF NOT EXISTS idx_search_log_at ON search_log(at);
CREATE INDEX IF NOT EXISTS idx_memories_invalid ON memories(invalid_at);
-- what a memory replaced, read when it is deleted for good (``replaced_by``)
CREATE INDEX IF NOT EXISTS idx_memories_superseded ON memories(superseded_by)
    WHERE superseded_by IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_memories_pending_enrichment
    ON memories(invalid_at, created_at)
    WHERE json_extract(metadata, '$.pending_distillation') = 1;

CREATE TABLE IF NOT EXISTS topics (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    normalized TEXT NOT NULL,
    user_id TEXT,
    agent_id TEXT,
    run_id TEXT,
    provenance TEXT NOT NULL DEFAULT 'memory',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_topics_scope_ns ON topics(
    user_id IS NULL, IFNULL(user_id, ''), agent_id IS NULL, IFNULL(agent_id, ''),
    run_id IS NULL, IFNULL(run_id, ''), normalized
);
CREATE TABLE IF NOT EXISTS memory_topics (
    memory_id TEXT NOT NULL,
    topic_id TEXT NOT NULL,
    PRIMARY KEY (memory_id, topic_id)
);
CREATE INDEX IF NOT EXISTS idx_memory_topics_topic ON memory_topics(topic_id, memory_id);
CREATE INDEX IF NOT EXISTS idx_memory_topics_memory ON memory_topics(memory_id, topic_id);
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content, content='memories', content_rowid='rowid'
);
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content)
    VALUES ('delete', old.rowid, old.content);
END;
CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE OF content ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content)
    VALUES ('delete', old.rowid, old.content);
    INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
END;

CREATE TABLE IF NOT EXISTS memory_events (
    id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL,
    event TEXT NOT NULL,
    old_content TEXT,
    new_content TEXT,
    reason TEXT,
    actor TEXT NOT NULL DEFAULT 'system',
    created_at TEXT NOT NULL,
    kind TEXT DEFAULT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_memory ON memory_events(memory_id);

CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    normalized TEXT NOT NULL,
    entity_type TEXT,
    user_id TEXT,
    agent_id TEXT,
    run_id TEXT,
    description TEXT,
    description_updated_at TEXT,
    metadata TEXT NOT NULL DEFAULT '{}',
    merged_into TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_entities_norm ON entities(
    normalized, user_id, agent_id, run_id
);
-- Tags are entities of type "topic" (models.TOPIC_TYPE): one per user and
-- normalized tag, found by type and user when a memory's categories are saved
-- and when they are counted.
CREATE INDEX IF NOT EXISTS idx_entities_type_user ON entities(
    entity_type, user_id, normalized
);
-- the names merged into an entity, read among its names (``entity_aliases``)
CREATE INDEX IF NOT EXISTS idx_entities_merged ON entities(merged_into)
    WHERE merged_into IS NOT NULL;

-- decided: what joined a name to an entity the store had (JSON, see
-- ``EntityMention.decided``); NULL for a mention that made its entity.
-- entity_type: the type extraction gave the name in this memory; NULL when
-- it gave none (a tag's mention, one saved before the column).
CREATE TABLE IF NOT EXISTS entity_mentions (
    id TEXT PRIMARY KEY,
    entity_id TEXT NOT NULL,
    memory_id TEXT NOT NULL,
    surface TEXT NOT NULL,
    created_at TEXT NOT NULL,
    decided TEXT,
    entity_type TEXT
);
CREATE INDEX IF NOT EXISTS idx_mentions_entity ON entity_mentions(entity_id);
CREATE INDEX IF NOT EXISTS idx_mentions_memory ON entity_mentions(memory_id);
CREATE INDEX IF NOT EXISTS idx_mentions_surface
    ON entity_mentions(lower(trim(surface)), entity_id);

CREATE TABLE IF NOT EXISTS entity_proposals (
    id TEXT PRIMARY KEY,
    entity_a TEXT NOT NULL,
    entity_b TEXT NOT NULL,
    user_id TEXT,
    status TEXT NOT NULL DEFAULT 'proposed',
    confidence REAL NOT NULL DEFAULT 0.5,
    reason TEXT,
    created_at TEXT NOT NULL,
    decided_at TEXT,
    compared_step INTEGER NOT NULL DEFAULT 0,
    different REAL,
    belongs TEXT
);
CREATE INDEX IF NOT EXISTS idx_proposals_status ON entity_proposals(status, user_id);

CREATE TABLE IF NOT EXISTS retired_entities (
    entity_id   TEXT PRIMARY KEY,
    user_id     TEXT,
    name        TEXT NOT NULL,
    entity_type TEXT,
    reason      TEXT,
    retired_at  TEXT NOT NULL,
    snapshot    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_retired_entities_user
    ON retired_entities(user_id, retired_at);

-- What each merge of two entities moved, so that a person can undo it
-- (``undo_merge``): both entity rows and names as they were, and the ids of
-- the mentions, relations and pairs the merge pointed at the kept one.
-- Recovery data, like retired_entities, so not in backups.
CREATE TABLE IF NOT EXISTS entity_merges (
    id        TEXT PRIMARY KEY,
    keep_id   TEXT NOT NULL,
    merge_id  TEXT NOT NULL,
    user_id   TEXT,
    merged_at TEXT NOT NULL,
    snapshot  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_entity_merges_merge ON entity_merges(merge_id);
CREATE INDEX IF NOT EXISTS idx_entity_merges_user ON entity_merges(user_id, merged_at);

CREATE TABLE IF NOT EXISTS ann_keys (
    key INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS relations (
    id         TEXT PRIMARY KEY,
    subject    TEXT NOT NULL,
    predicate  TEXT NOT NULL,
    object     TEXT NOT NULL,
    user_id    TEXT,
    memory_id  TEXT,
    created_at TEXT NOT NULL,
    valid_from TEXT NOT NULL,
    invalid_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_relations_subject ON relations(subject);
CREATE INDEX IF NOT EXISTS idx_relations_object ON relations(object);
CREATE INDEX IF NOT EXISTS idx_relations_user ON relations(user_id);
"""
#: The tables of the schema; the full-text index follows ``memories`` by its
#: triggers.
_TABLES = tuple(re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", _SCHEMA))

# The names an entity gives (its own and each alias of its metadata), and
# whether an ``entity_names`` row is still given by its entity or by one of
# its mentions, for the triggers below.
_GIVEN_BY = (
    "SELECT {e}.name AS value UNION SELECT CAST(alias.value AS TEXT) FROM json_each("
    "CASE WHEN json_valid({e}.metadata) THEN {e}.metadata END, '$.aliases') AS alias "
    "WHERE alias.value IS NOT NULL"
)
_STILL_GIVEN = (
    "EXISTS (SELECT 1 FROM entities e WHERE e.id = entity_names.entity_id "
    "AND entity_names.name IN (" + _GIVEN_BY.format(e="e") + ")) OR EXISTS ("
    "SELECT 1 FROM entity_mentions m WHERE lower(trim(m.surface)) = "
    "lower(trim(entity_names.name)) AND m.entity_id = entity_names.entity_id "
    "AND m.surface = entity_names.name)"
)


def _owner(entity: str) -> str:
    """The owner of an entity's names: the user of the entity they are read
    for (the one it was merged into, if it was), between unit separators so
    that the trigram index finds it whole, however short (``_owned``)."""
    return (f"(SELECT char(31, 31) || t.user_id || char(31) FROM entities x JOIN entities t "
            f"ON t.id = IFNULL(x.merged_into, x.id) WHERE x.id = {entity})")


def _owned(user_id: str) -> str:
    """How ``_owner`` writes a user in the index of names."""
    return "\x1f\x1f" + user_id + "\x1f"


def _add_names(entity: str, values: str) -> str:
    return (f"INSERT INTO entity_names (entity_id, name, owner) SELECT {entity}, value, "
            f"{_owner(entity)} FROM ({values}) WHERE value IS NOT NULL AND value != '' "
            f"AND NOT EXISTS (SELECT 1 FROM entity_names WHERE entity_id = {entity} "
            "AND name = value);")


def _drop_names(entity: str, values: str) -> str:
    return (f"DELETE FROM entity_names WHERE entity_id = {entity} AND name IN ({values}) "
            f"AND NOT ({_STILL_GIVEN});")


def _own_names(entity: str) -> str:
    """The names of ``entity`` and of the entities merged into it given
    their owner again."""
    return (f"UPDATE entity_names SET owner = {_owner('entity_names.entity_id')} "
            f"WHERE entity_id IN (SELECT {entity} UNION SELECT id FROM entities "
            f"WHERE merged_into = {entity});")


# Every name an entity answers to, once per entity and name: its own, each
# alias of its metadata and each wording its mentions use (an entity merged
# into another keeps its own name here, read for the one it was merged into),
# with the user it is read for. Derived, like memories_fts, and kept by the
# triggers: a name comes in with what gives it and goes when nothing gives it
# any more. entity_names_fts finds a name by any part of it three letters or
# longer, of one user's names alone where the search has a user
# (``LocalBackend.entity_names_holding``). Made where SQLite has the trigram
# tokenizer (3.34 and later).
_ENTITY_NAMES_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS entity_names (
    id INTEGER PRIMARY KEY,
    entity_id TEXT NOT NULL,
    name TEXT NOT NULL,
    owner TEXT,
    UNIQUE (entity_id, name)
);
CREATE VIRTUAL TABLE IF NOT EXISTS entity_names_fts USING fts5(
    name, owner, content='entity_names', content_rowid='id', tokenize='trigram'
);
CREATE INDEX IF NOT EXISTS idx_entity_names_folded
    ON entity_names(lower(trim(name)), owner);
CREATE TRIGGER IF NOT EXISTS entity_names_ai AFTER INSERT ON entity_names BEGIN
    INSERT INTO entity_names_fts(rowid, name, owner) VALUES (new.id, new.name, new.owner);
END;
CREATE TRIGGER IF NOT EXISTS entity_names_au AFTER UPDATE OF owner ON entity_names
WHEN old.owner IS NOT new.owner BEGIN
    INSERT INTO entity_names_fts(entity_names_fts, rowid, name, owner)
    VALUES ('delete', old.id, old.name, old.owner);
    INSERT INTO entity_names_fts(rowid, name, owner) VALUES (new.id, new.name, new.owner);
END;
CREATE TRIGGER IF NOT EXISTS entity_names_ad AFTER DELETE ON entity_names BEGIN
    INSERT INTO entity_names_fts(entity_names_fts, rowid, name, owner)
    VALUES ('delete', old.id, old.name, old.owner);
END;
CREATE TRIGGER IF NOT EXISTS entities_names_ai AFTER INSERT ON entities BEGIN
    {_own_names("new.id")}
    {_add_names("new.id", _GIVEN_BY.format(e="new"))}
END;
CREATE TRIGGER IF NOT EXISTS entities_names_au AFTER UPDATE OF name, metadata ON entities BEGIN
    {_drop_names("old.id", _GIVEN_BY.format(e="old"))}
    {_add_names("new.id", _GIVEN_BY.format(e="new"))}
END;
CREATE TRIGGER IF NOT EXISTS entities_owner_au AFTER UPDATE OF user_id, merged_into ON entities
BEGIN
    {_own_names("new.id")}
END;
CREATE TRIGGER IF NOT EXISTS entities_names_ad AFTER DELETE ON entities BEGIN
    {_drop_names("old.id", "SELECT name FROM entity_names WHERE entity_id = old.id")}
    {_own_names("old.id")}
END;
CREATE TRIGGER IF NOT EXISTS entity_mentions_names_ai AFTER INSERT ON entity_mentions BEGIN
    {_add_names("new.entity_id", "SELECT new.surface AS value")}
END;
CREATE TRIGGER IF NOT EXISTS entity_mentions_names_au
AFTER UPDATE OF entity_id, surface ON entity_mentions BEGIN
    {_drop_names("old.entity_id", "SELECT old.surface")}
    {_add_names("new.entity_id", "SELECT new.surface AS value")}
END;
CREATE TRIGGER IF NOT EXISTS entity_mentions_names_ad AFTER DELETE ON entity_mentions BEGIN
    {_drop_names("old.entity_id", "SELECT old.surface")}
END;
"""
_ENTITY_NAMES_TRIGGERS = tuple(
    re.findall(r"CREATE TRIGGER IF NOT EXISTS (\w+)", _ENTITY_NAMES_SCHEMA))

_MEMORY_COLS = (
    "id, content, memory_type, user_id, agent_id, run_id, importance, categories, "
    "entities, metadata, created_at, updated_at, valid_from, invalid_at, superseded_by, "
    "source_episode_ids, embedding_model"
)

_BACKUP_TABLE_KEYS: dict[str, tuple[str, ...]] = {
    "episodes": ("id",),
    "memories": ("id",),
    "memory_questions": ("memory_id", "n"),
    "memory_events": ("id",),
    "topics": ("id",),
    "memory_topics": ("memory_id", "topic_id"),
    "entities": ("id",),
    "entity_mentions": ("id",),
    "entity_proposals": ("id",),
    "relations": ("id",),
}
_BACKUP_ORDER = tuple(_BACKUP_TABLE_KEYS)
#: The columns of a kept search as written (``log_search``). The search log
#: is not in ``_BACKUP_TABLE_KEYS``: a backup carries it only when asked.
_SEARCH_LOG_COLS = (
    "at", "user_id", "agent_id", "run_id", "query", "mode", "seeds", "first_person",
    "filtered", "judged", "best_judged", "results", "latency_ms",
)
#: The tables whose rows carry a namespace, but for the legacy tag index
#: (``topics``), which ``adopt_unscoped`` moves apart.
_NAMESPACED_TABLES = (
    "episodes", "memories", "entities", "entity_proposals", "relations",
    "retired_entities", "entity_merges",
)
_BACKUP_USER_TABLES = {
    "episodes", "memories", "topics", "entities", "entity_proposals", "relations",
}
_BACKUP_BYTES = "__memry_base64__"

#: The partial unique index holding one active topic entity per namespace
#: (``user_id`` exactly, None apart from "") and tag, and the one it replaced,
#: which keyed ``IFNULL(user_id, '')`` and so folded the two namespaces
#: (``LocalBackend._ensure_one_active_topic_per_name``).
_ACTIVE_TOPIC_INDEX = "ux_entities_active_topic_ns"
_ACTIVE_TOPIC_INDEX_V1 = "ux_entities_active_topic"
#: Set once ``_ACTIVE_TOPIC_INDEX`` is in place and the memories of the two
#: namespaces the old index folded are filed again.
_ACTIVE_TOPIC_MARKER = "schema:active-topic-ns:v1"

#: Set once the columns still naming a tag merged away (written before a
#: memory's tags were filed through ``LocalBackend._file_tags_locked``) name
#: its survivor, and their index and mentions follow.
_TAG_SURVIVORS_MARKER = "schema:tag-survivors:v1"

#: Set once ``entity_names`` holds every name of the entities a database had
#: before it (``LocalBackend._ensure_entity_names``); its triggers keep it from
#: then on.
_ENTITY_NAMES_MARKER = "schema:entity-names:v1"

#: A word of a question as the full-text indexes read one (FTS5's unicode61):
#: letters and digits of any script, "München" whole, and the accents written
#: as marks after a letter kept with it (FTS5 drops them, so "Mu\u0308nchen"
#: is "munchen" there too).
_WORD_RE = re.compile(r"[^\W_](?:[^\W_]|[\u0300-\u036f])*")

#: What one word adds to a memory's BM25 at most, over its inverse document
#: frequency: FTS5's term part, f(k1 + 1) / (f + k1(1 - b + b·len/avglen)),
#: stays below k1 + 1, with k1 = 1.2 (``LocalBackend._keyword_scores``).
_BM25_TERM_BOUND = 2.2
#: A word held by at most this many memories is scored in each of them; a
#: commoner one only where it can still change the first ``limit``
#: (``LocalBackend._keyword_scores``).
_WHOLE_WORD_ROWS = 200
#: The share by which a bound on a score is widened against rounding.
_ROUNDING = 1e-9
#: Of the floor under the k-th best keyword score, the share the common words
#: set aside may add up to while others are read (``_keyword_scores``).
_ASIDE_SHARE = 0.25


def _bm25_idf(hits: int, rows: int) -> float:
    """The inverse document frequency ``bm25()`` gives a term found in
    ``hits`` of ``rows`` rows, as FTS5 computes it (a term in more than half
    of them weighs next to nothing)."""
    idf = math.log((rows - hits + 0.5) / (hits + 0.5))
    return idf if idf > 0 else 1e-6


class _WordCounts(NamedTuple):
    """How many memories and turns the full-text indexes hold, and for each
    word how many of each hold it."""
    memories: int
    turns: int
    hits: dict[str, tuple[int, int]]


def _summed(
    words: list[str], factor: dict[str, float], ranks: dict[str, dict[int, float]],
    ids: dict[int, str],
) -> dict[str, float]:
    """The keyword score of each memory of ``ids`` (rowid -> memory id): each
    word's ``bm25()`` alone (``ranks``, lower is better) times its ``factor``,
    summed in the question's order."""
    scores: dict[str, float] = {}
    for word in words:
        for rowid, rank in ranks[word].items():
            memory_id = ids.get(rowid)
            if memory_id is not None:
                scores[memory_id] = scores.get(memory_id, 0.0) - factor[word] * rank
    return scores


def _band_bounds(
    words: list[tuple[float, float]],
) -> tuple[Callable[[float], float], Callable[[float], float]]:
    """For words read together, each as (weight, idf), once for each time
    the question says it: the least and the most they give a memory whose
    ``bm25()`` alone for each, flipped, sum to ``u``. Each is below its idf
    times k1 + 1, so the least fills the words of the smallest weight first
    and the most those of the largest, both widened against rounding."""
    words = sorted((weight, idf * _BM25_TERM_BOUND) for weight, idf in words)

    def filled(u: float, order: Iterable[tuple[float, float]]) -> tuple[float, float]:
        total = 0.0
        for weight, cap in order:
            part = min(u, cap)
            total += weight * part
            u -= part
            if u <= 0:
                break
        return total, max(u, 0.0)

    def least(u: float) -> float:
        return filled(u * (1 - _ROUNDING), words)[0] * (1 - _ROUNDING)

    def most(u: float) -> float:
        if not words:
            return 0.0
        total, left = filled(u * (1 + _ROUNDING), reversed(words))
        return (total + left * words[-1][0]) * (1 + _ROUNDING)

    return least, most


def _owner_typed(row: sqlite3.Row) -> bool:
    """Whether an entity row's type is one the owner chose
    (``TYPE_SET_BY_OWNER``), which no recount of its mentions changes."""
    try:
        return bool(json.loads(row["metadata"] or "{}").get(TYPE_SET_BY_OWNER))
    except (TypeError, ValueError):
        return False


def _scope_clause(scope: Scope, prefix: str = "") -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if scope.exact_user and scope.user_id is None:
        clauses.append(f"{prefix}user_id IS NULL")
    for field in ("user_id", "agent_id", "run_id"):
        value = getattr(scope, field)
        if value is not None:
            clauses.append(f"{prefix}{field} = ?")
            params.append(value)
    return (" AND ".join(clauses) if clauses else "1=1"), params


def _user_scope(scope: Scope) -> Scope:
    """``scope`` narrowed to its user, the agent and run left out; a user of
    None stays the memories without one when the scope says so
    (``exact_user``)."""
    return Scope(user_id=scope.user_id, exact_user=scope.exact_user)


def _search_scope_clause(scope: Scope, memory: str) -> tuple[str, list[Any]]:
    """``_scope_clause`` for a search of memories (``memory`` is the table or
    its alias). Its run is where a memory was said, read from the memory's
    evidence: a memory is in a run when it is the run's own or when one of
    its source episodes is of the run (a save that restated a memory of
    another run, reconcile's SAME, or a merged text carrying the sources of
    the memory it replaced)."""
    clause, params = _scope_clause(
        Scope(user_id=scope.user_id, agent_id=scope.agent_id, exact_user=scope.exact_user),
        prefix=f"{memory}.")
    if scope.run_id is None:
        return clause, params
    return (
        f"{clause} AND ({memory}.run_id = ? OR EXISTS (SELECT 1 FROM "
        f"json_each({memory}.source_episode_ids) AS source JOIN episodes "
        "ON episodes.id = source.value WHERE episodes.run_id = ?))",
        [*params, scope.run_id, scope.run_id],
    )


def _history_clause(memory: str) -> str:
    """A memory out of use that stays retrievable as history: superseded, and
    its latest SUPERSEDE event of a kind in ``HISTORY_KINDS`` (an update). A
    memory brought back and superseded again is read by its latest event."""
    kinds = ", ".join(f"'{kind}'" for kind in HISTORY_KINDS)
    return (
        f"{memory}.invalid_at IS NOT NULL AND {memory}.superseded_by IS NOT NULL AND "
        "(SELECT event.kind FROM memory_events AS event WHERE event.memory_id = "
        f"{memory}.id AND event.event = 'SUPERSEDE' ORDER BY event.rowid DESC LIMIT 1) "
        f"IN ({kinds})"
    )


def _category_clause(categories: list[str] | None, memory_id: str) -> tuple[str, list[Any]]:
    if not categories:
        return "1=1", []
    normalized = [c.strip().lower() for c in categories if c.strip()]
    if not normalized:
        return "1=1", []
    placeholders = ",".join("?" * len(normalized))
    return (
        "EXISTS (SELECT 1 FROM memory_topics mt JOIN topics t ON t.id = mt.topic_id "
        f"WHERE mt.memory_id = {memory_id} AND t.normalized IN ({placeholders}))",
        normalized,
    )


def _entity_clause(
    entity_id: str | list[str] | None, memory_id: str
) -> tuple[str, list[Any]]:
    """Filter to memories mentioning the entity, or ANY of several.

    OR semantics, matching ``_category_clause``: picking two people means
    "either of them", which is what selecting two rows in a list implies.
    """
    ids = [entity_id] if isinstance(entity_id, str) else list(entity_id or [])
    ids = [e for e in ids if e]
    if not ids:
        return "1=1", []
    placeholders = ",".join("?" * len(ids))
    return (
        "EXISTS (SELECT 1 FROM entity_mentions em "
        f"WHERE em.memory_id = {memory_id} AND em.entity_id IN ({placeholders}))",
        ids,
    )


def _among_clause(among: Any, memory_id: str) -> tuple[str, list[Any]]:
    """Filter to a set of memory ids (``among``, None for no such filter):
    the memories a search's filters admit (``MemoryStore._admitted``), read
    in SQL with the scope, so no first N is taken from memories they drop.
    One JSON parameter, however many ids, so no variable limit is reached."""
    if among is None:
        return "1=1", []
    return f"{memory_id} IN (SELECT value FROM json_each(?))", [json.dumps(sorted(among))]


def _entity_reads_clause(
    *, include_invalid: bool, scope: Scope | None, history: bool,
    categories: list[str] | None, mentioning: str | list[str] | None,
    among: Any = None,
) -> tuple[str, list[Any]]:
    """Which of an entity's memories (``m``) a lookup reads, as a search reads
    its text ranking (``keyword_search``): those in use, with ``history`` also
    those kept as history, every one with ``include_invalid``; of ``scope``,
    a run's being those said in it (``_search_scope_clause``); filed under
    one of ``categories``; mentioning ``mentioning`` (or any of several)."""
    clause, params = _search_scope_clause(scope or Scope(), "m")
    if not include_invalid:
        clause += (f" AND (m.invalid_at IS NULL OR ({_history_clause('m')}))" if history
                   else " AND m.invalid_at IS NULL")
    cat_clause, cat_params = _category_clause(categories, "m.id")
    entity_clause, entity_params = _entity_clause(mentioning, "m.id")
    among_clause, among_params = _among_clause(among, "m.id")
    return (f"{clause} AND {cat_clause} AND {entity_clause} AND {among_clause}",
            [*params, *cat_params, *entity_params, *among_params])


def _kind_clause(kind: str, prefix: str = "") -> str:
    """Which entities a lookup reads: "named" (people, products, ... every
    type but ``TOPIC_TYPE``), "topic" (tags) or "any"."""
    if kind == "named":
        return f"IFNULL({prefix}entity_type, '') != '{TOPIC_TYPE}'"
    if kind == "topic":
        return f"{prefix}entity_type = '{TOPIC_TYPE}'"
    if kind == "any":
        return "1=1"
    raise ValueError(f"unknown entity kind: {kind!r}")


def _exact_scope_clause(scope: Scope, prefix: str = "") -> tuple[str, list[Any]]:
    """``_scope_clause``, except that the user is always matched: no user
    means the rows without one, not every user's. A topic entity belongs to
    one user (or to none), and merging it must not reach anyone else's."""
    clauses = [f"{prefix}user_id IS ?"]
    params: list[Any] = [scope.user_id]
    for field in ("agent_id", "run_id"):
        value = getattr(scope, field)
        if value is not None:
            clauses.append(f"{prefix}{field} = ?")
            params.append(value)
    return " AND ".join(clauses), params


def _row_scope(row: sqlite3.Row) -> Scope:
    """The scope of a row carrying ``user_id``, ``agent_id`` and ``run_id``."""
    return Scope(user_id=row["user_id"], agent_id=row["agent_id"], run_id=row["run_id"])


_EPISODE_COLS = (
    "id, content, role, name, user_id, agent_id, run_id, metadata, created_at, withheld_at"
)


def _row_to_episode(row: sqlite3.Row) -> Episode:
    return Episode(
        id=row["id"],
        content=row["content"],
        role=row["role"],
        name=row["name"],
        user_id=row["user_id"],
        agent_id=row["agent_id"],
        run_id=row["run_id"],
        metadata=json.loads(row["metadata"]),
        created_at=row["created_at"],
        withheld_at=row["withheld_at"],
    )


def _pack_half(vector: list[float]) -> bytes:
    """A vector as float16 bytes (a question key's vector: a few per memory)."""
    return np.asarray(vector, dtype=np.float16).tobytes()


def _unpack_half(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float16).astype(np.float32)


def _prefixed(columns: str, alias: str) -> str:
    """``_MEMORY_COLS`` with each column under ``alias``."""
    return ", ".join(f"{alias}.{column.strip()}" for column in columns.split(","))


def _row_to_memory(row: sqlite3.Row) -> Memory:
    return Memory(
        id=row["id"],
        content=row["content"],
        memory_type=row["memory_type"],
        user_id=row["user_id"],
        agent_id=row["agent_id"],
        run_id=row["run_id"],
        importance=row["importance"],
        categories=json.loads(row["categories"]),
        entities=json.loads(row["entities"]),
        metadata=json.loads(row["metadata"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        valid_from=row["valid_from"],
        invalid_at=row["invalid_at"],
        superseded_by=row["superseded_by"],
        source_episode_ids=json.loads(row["source_episode_ids"]),
        embedding_model=row["embedding_model"],
    )


class LocalBackend(MemoryBackend):
    supports_transactions = True

    def __init__(self, db_path: str = ":memory:", ann: AnnConfig | None = None) -> None:
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        # open ``transaction`` blocks of the thread holding the lock: while one
        # is open, ``_commit`` leaves the writes for the block to commit
        self._tx_depth = 0
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        # an episode table from before its columns and full-text index: the
        # columns are added first, and the index is filled once it exists
        episode_columns = {
            row["name"] for row in self._db.execute("PRAGMA table_info(episodes)").fetchall()
        }
        for column, kind in (("withheld_at", "TEXT"), ("embedding", "BLOB"),
                             ("embedding_model", "TEXT"), ("name", "TEXT")):
            if episode_columns and column not in episode_columns:
                self._db.execute(f"ALTER TABLE episodes ADD COLUMN {column} {kind} DEFAULT NULL")
        indexed = self._db.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'episodes_fts'").fetchone() is not None
        self._db.executescript(_SCHEMA)
        if not indexed:
            self._db.execute("INSERT INTO episodes_fts(episodes_fts) VALUES ('rebuild')")
        self._ensure_entity_description_columns()
        self._ensure_exact_topic_scopes()
        self._ensure_one_active_topic_per_name()
        self._backfill_topics()
        self._migrate_tags_to_topic_entities()
        self._one_row_per_pair_everywhere()
        self._refile_merged_tags()
        self._name_index = self._ensure_entity_names()
        self._commit()
        # each full-text index's terms, with how many rows hold each
        # (``_word_counts``); of this connection only
        for table in ("memories_fts", "episodes_fts"):
            self._db.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS temp.{table}_terms "
                             f"USING fts5vocab(main, {table}, row)")
        self._has_metadata_aliases = self._db.execute(
            "SELECT 1 FROM entities WHERE metadata LIKE '%\"aliases\"%' LIMIT 1"
        ).fetchone() is not None
        self.db_path = db_path
        self._ann_cfg = ann or AnnConfig()
        # One sidecar per embedding model: a multiuser server with per-account
        # BYO-key can run several models against this one DB, and a single slot
        # would rebuild the whole HNSW index every time the model alternated.
        self._anns: dict[tuple[str, int], HnswSidecar] = {}
        self._ann_pending_saves = 0

    # -- schema migration + normalized topics ----------------------------
    def _ensure_entity_description_columns(self) -> None:
        columns = {
            row["name"] for row in self._db.execute("PRAGMA table_info(entities)").fetchall()
        }
        if "description" not in columns:
            self._db.execute("ALTER TABLE entities ADD COLUMN description TEXT")
        if "description_updated_at" not in columns:
            self._db.execute(
                "ALTER TABLE entities ADD COLUMN description_updated_at TEXT"
            )
        proposal_columns = {
            row["name"]
            for row in self._db.execute("PRAGMA table_info(entity_proposals)").fetchall()
        }
        if "compared_step" not in proposal_columns:
            self._db.execute(
                "ALTER TABLE entity_proposals ADD COLUMN compared_step INTEGER NOT NULL DEFAULT 0"
            )
        if "different" not in proposal_columns:
            self._db.execute("ALTER TABLE entity_proposals ADD COLUMN different REAL")
        if "belongs" not in proposal_columns:
            self._db.execute("ALTER TABLE entity_proposals ADD COLUMN belongs TEXT")
        event_columns = {
            row["name"]
            for row in self._db.execute("PRAGMA table_info(memory_events)").fetchall()
        }
        mention_columns = {
            row["name"]
            for row in self._db.execute("PRAGMA table_info(entity_mentions)").fetchall()
        }
        if "decided" not in mention_columns:
            self._db.execute("ALTER TABLE entity_mentions ADD COLUMN decided TEXT")
        if "entity_type" not in mention_columns:
            self._db.execute("ALTER TABLE entity_mentions ADD COLUMN entity_type TEXT")
        if "kind" not in event_columns:
            # what took a memory out of use (``MemoryEvent.kind``); older rows
            # keep NULL and are read by their reason. The default lets a backup
            # from before the column restore.
            self._db.execute("ALTER TABLE memory_events ADD COLUMN kind TEXT DEFAULT NULL")

    def _topic_locked(self, name: str, scope: Scope, provenance: str = "memory") -> Topic:
        display = name.strip()
        normalized = display.lower()
        row = self._db.execute(
            "SELECT * FROM topics WHERE normalized = ? AND user_id IS ? "
            "AND agent_id IS ? AND run_id IS ?",
            (normalized, scope.user_id, scope.agent_id, scope.run_id),
        ).fetchone()
        if row:
            return self._row_to_topic(row)
        topic = Topic(
            name=display,
            normalized=normalized,
            user_id=scope.user_id,
            agent_id=scope.agent_id,
            run_id=scope.run_id,
            provenance=provenance,
        )
        self._db.execute(
            "INSERT INTO topics (id, name, normalized, user_id, agent_id, run_id, "
            "provenance, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                topic.id, topic.name, topic.normalized, topic.user_id, topic.agent_id,
                topic.run_id, topic.provenance, topic.created_at, topic.updated_at,
            ),
        )
        return topic

    @staticmethod
    def _row_to_topic(row: sqlite3.Row) -> Topic:
        return Topic(
            id=row["id"], name=row["name"], normalized=row["normalized"],
            user_id=row["user_id"], agent_id=row["agent_id"], run_id=row["run_id"],
            provenance=row["provenance"], created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _backfill_topics(self) -> None:
        marker = self._db.execute(
            "SELECT value FROM meta WHERE key = 'schema:topics:v1'"
        ).fetchone()
        if marker:
            return
        self._refile_locked("1=1", (), cache={})
        self._db.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema:topics:v1', ?)",
            (utcnow(),),
        )

    def _ensure_exact_topic_scopes(self) -> None:
        """The legacy ``topics`` rows are unique per scope and name with each
        scope field read exactly, "" apart from None, as ``_topic_locked``
        looks them up (``idx_topics_scope_ns``, in the schema). The index it
        replaces (``idx_topics_scope_norm``) keyed ``IFNULL(field, '')``, so a
        tag of the user "" beside the same tag without a user failed its
        insert and the save with it. That index is dropped; it held no pair
        the new one refuses."""
        self._db.execute("DROP INDEX IF EXISTS idx_topics_scope_norm")

    def _ensure_one_active_topic_per_name(self) -> None:
        """One active topic entity per namespace and tag, held by a partial
        unique index, so two processes creating the same topic at once end
        with one (``_create_topic_locked`` inserts or ignores and reads back).

        The namespace is ``user_id`` exactly, as every lookup reads it
        (``user_id IS ?``, ``_scope_clause``, ``_exact_scope_clause``): ""
        is a user of its own, not the memories without one (None). The index
        keys ``(user_id IS NULL, IFNULL(user_id, ''))`` for that, since a
        NULL key never collides. An index from before
        (``ux_entities_active_topic``, keyed ``IFNULL(user_id, '')``) folded
        the two, so a topic of one blocked the other's insert and that
        memory's mention was dropped: it is replaced, and the memories of
        both namespaces are filed again (``_file_tags_locked``). A database
        from before any index may hold twins: each later one is folded into
        the earliest of its namespace (``merge_entities``), never across
        namespaces, then the index is made.

        The old index is dropped, the twins folded and the new index made in
        one transaction, and the refile runs after it and is marked done
        (``_ACTIVE_TOPIC_MARKER``) only once it ran: an open stopped anywhere
        in between leaves the old index, or the marker unset, and the next
        open does the rest."""
        indexes = {
            row["name"] for row in self._db.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index' AND name IN (?, ?)",
                (_ACTIVE_TOPIC_INDEX, _ACTIVE_TOPIC_INDEX_V1),
            ).fetchall()
        }
        refiled = self._db.execute(
            "SELECT 1 FROM meta WHERE key = ?", (_ACTIVE_TOPIC_MARKER,)
        ).fetchone() is not None
        if _ACTIVE_TOPIC_INDEX in indexes and _ACTIVE_TOPIC_INDEX_V1 not in indexes and refiled:
            return
        self._commit()  # the open's earlier steps, apart from this transaction
        # One transaction: DDL outside one would be committed on its own, and
        # a stop between the drop and the create lost the old index's trace.
        self._db.execute("BEGIN")
        try:
            self._replace_active_topic_index_locked(_ACTIVE_TOPIC_INDEX_V1 in indexes)
        except Exception:
            self._db.rollback()
            raise
        self._commit()
        if not refiled or _ACTIVE_TOPIC_INDEX_V1 in indexes:
            if self._db.execute(
                "SELECT 1 FROM memories WHERE user_id = '' LIMIT 1"
            ).fetchone():
                # a mention dropped while "" and None shared the index comes back
                self._refile_locked("user_id = '' OR user_id IS NULL", (), cache={})
            self._db.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                (_ACTIVE_TOPIC_MARKER, utcnow()),
            )
            self._commit()

    def _replace_active_topic_index_locked(self, old_index: bool) -> None:
        """Inside the open's transaction: drop the index keyed
        ``IFNULL(user_id, '')`` when ``old_index``, fold the active twins of
        each namespace, and make ``_ACTIVE_TOPIC_INDEX``."""
        if old_index:
            self._db.execute(f"DROP INDEX {_ACTIVE_TOPIC_INDEX_V1}")
        twins = self._db.execute(
            "SELECT e.id, e.user_id, e.normalized FROM entities e JOIN ("
            "SELECT user_id IS NULL AS nobody, IFNULL(user_id, '') AS owner, normalized "
            "FROM entities WHERE entity_type = ? AND merged_into IS NULL "
            "GROUP BY user_id IS NULL, IFNULL(user_id, ''), normalized HAVING COUNT(*) > 1) d "
            "ON (e.user_id IS NULL) = d.nobody AND IFNULL(e.user_id, '') = d.owner "
            "AND e.normalized = d.normalized "
            "WHERE e.entity_type = ? AND e.merged_into IS NULL "
            "ORDER BY d.nobody, d.owner, e.normalized, e.created_at, e.id",
            (TOPIC_TYPE, TOPIC_TYPE),
        ).fetchall()
        kept: dict[tuple[str | None, str], str] = {}
        for twin in twins:
            key = (twin["user_id"], twin["normalized"])  # None and "" apart
            if key not in kept:
                kept[key] = twin["id"]
                continue
            self._merge_entities_locked(kept[key], twin["id"])
        for keep_id in kept.values():
            # a memory that mentioned both twins mentions the one kept once
            self._db.execute(
                "DELETE FROM entity_mentions WHERE entity_id = ? AND id NOT IN ("
                "SELECT MIN(id) FROM entity_mentions WHERE entity_id = ? GROUP BY memory_id)",
                (keep_id, keep_id),
            )
        self._db.execute(
            f"CREATE UNIQUE INDEX IF NOT EXISTS {_ACTIVE_TOPIC_INDEX} "
            "ON entities(user_id IS NULL, IFNULL(user_id, ''), normalized) "
            f"WHERE entity_type = '{TOPIC_TYPE}' AND merged_into IS NULL"
        )

    def _migrate_tags_to_topic_entities(self) -> None:
        """Give the legacy tags their topic entities and mentions exactly once
        (``tags_to_topics``, which needs no model), so an upgraded database
        shows its tags without ``memry tags-to-things``. Each user's share is
        committed on its own and the marker is set after the last, so an open
        stopped midway resumes at the next open where it stopped (what exists
        is counted and left alone); a later open finds the marker and does
        nothing."""
        marker = self._db.execute(
            "SELECT value FROM meta WHERE key = 'schema:tag-entities:v1'"
        ).fetchone()
        if marker:
            return
        self._commit()  # the open's earlier steps, before the first user's commit
        self.tags_to_topics(all_users=True)
        self._db.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema:tag-entities:v1', ?)",
            (utcnow(),),
        )
        self._commit()

    # -- transactions ---------------------------------------------------
    def _commit(self) -> None:
        """Commit a method's writes, unless a ``transaction`` is open: then
        they are committed with the rest of it, at its end."""
        if not self._tx_depth:
            self._db.commit()

    @contextlib.contextmanager
    def transaction(self) -> Iterator[None]:
        """Every write made inside is committed at the end, or none is.

        The lock is held throughout, so another thread's write cannot land
        in the middle and be committed or rolled back with these; it is an
        RLock, so the write methods called inside take it again. They leave
        their commit to the block (``_commit``); a block inside another joins
        it. On an error the writes are rolled back and the ANN sidecars are
        rebuilt from SQLite at their next use, since they took the vectors
        added and dropped meanwhile and are not part of the rollback."""
        with self._lock:
            outer = self._tx_depth == 0
            if outer:
                if self._db.in_transaction:
                    self._db.commit()  # earlier writes, apart from this transaction
                self._db.execute("BEGIN")
            self._tx_depth += 1
            try:
                yield
            except BaseException:
                if outer:
                    self._tx_depth = 0
                    self._db.rollback()
                    for sidecar in self._anns.values():
                        sidecar.mark_stale()
                raise
            else:
                if outer:
                    self._tx_depth = 0
                    self._db.commit()
            finally:
                if not outer:
                    self._tx_depth -= 1

    # -- ANN sidecar ----------------------------------------------------
    def _ann_index(self, model_id: str, dimensions: int) -> HnswSidecar | None:
        """Lazily create/load the sidecar for a given model; rebuild if stale."""
        if not (self._ann_cfg.enabled and HAS_USEARCH) or dimensions <= 0:
            return None
        key = (model_id, dimensions)
        sidecar = self._anns.get(key)
        if sidecar is None:
            sidecar = HnswSidecar(self.db_path, dimensions, model_id)
            self._anns[key] = sidecar
        if sidecar.needs_rebuild:
            self.rebuild_ann(model_id, dimensions)
        return self._anns.get(key)

    def _ann_key(self, memory_id: str) -> int:
        self._db.execute(
            "INSERT OR IGNORE INTO ann_keys (memory_id) VALUES (?)", (memory_id,)
        )
        return self._db.execute(
            "SELECT key FROM ann_keys WHERE memory_id = ?", (memory_id,)
        ).fetchone()[0]

    def _ann_add(self, memory_id: str, embedding: list[float], model_id: str) -> None:
        index = self._ann_index(model_id, len(embedding))
        if index is None:
            return
        index.add(self._ann_key(memory_id), embedding)
        self._ann_pending_saves += 1
        if self._ann_pending_saves >= 64:
            index.save()
            self._ann_pending_saves = 0

    def _ann_remove(self, memory_id: str) -> None:
        if not self._anns:
            return
        row = self._db.execute(
            "SELECT key FROM ann_keys WHERE memory_id = ?", (memory_id,)
        ).fetchone()
        if not row:
            return
        # A memory lives in exactly one model's index, but which one is not
        # known here; usearch remove() is a no-op for absent keys, so clearing
        # it from every loaded index is correct and cheap.
        for sidecar in self._anns.values():
            sidecar.remove(row[0])

    def rebuild_ann(self, model_id: str, dimensions: int) -> int:
        """Rebuild the sidecar from SQLite (the source of truth)."""
        if not (self._ann_cfg.enabled and HAS_USEARCH) or dimensions <= 0:
            return 0
        with self._lock:
            self._db.execute(
                "INSERT OR IGNORE INTO ann_keys (memory_id) "
                "SELECT id FROM memories WHERE embedding IS NOT NULL"
            )
            rows = self._db.execute(
                "SELECT ak.key, m.embedding FROM memories m "
                "JOIN ann_keys ak ON ak.memory_id = m.id "
                "WHERE m.embedding IS NOT NULL AND m.embedding_model = ? "
                "AND m.invalid_at IS NULL",
                (model_id,),
            ).fetchall()
            self._commit()
        key = (model_id, dimensions)
        sidecar = self._anns.get(key)
        if sidecar is None:
            sidecar = HnswSidecar(self.db_path, dimensions, model_id)
            self._anns[key] = sidecar
        sidecar.rebuild([(r[0], r[1]) for r in rows])
        return len(rows)

    # -- episodes -------------------------------------------------------
    def add_episodes(self, episodes: list[Episode]) -> None:
        with self._lock:
            self._db.executemany(
                "INSERT INTO episodes (id, content, role, name, user_id, agent_id, run_id, "
                "metadata, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                [
                    (
                        e.id,
                        e.content,
                        e.role,
                        e.name,
                        e.user_id,
                        e.agent_id,
                        e.run_id,
                        json.dumps(e.metadata),
                        e.created_at,
                    )
                    for e in episodes
                ],
            )
            self._commit()

    def list_episodes(self, scope: Scope, limit: int = 100) -> list[Episode]:
        clause, params = _scope_clause(scope)
        with self._lock:
            rows = self._db.execute(
                f"SELECT {_EPISODE_COLS} FROM episodes WHERE {clause} "
                "ORDER BY created_at DESC, id LIMIT ?",
                (*params, limit),
            ).fetchall()
        return [_row_to_episode(r) for r in rows]

    def episodes_by_id(self, episode_ids: list[str]) -> dict[str, Episode]:
        out: dict[str, Episode] = {}
        ids = list(dict.fromkeys(episode_ids))
        for start in range(0, len(ids), 500):
            chunk = ids[start:start + 500]
            with self._lock:
                rows = self._db.execute(
                    f"SELECT {_EPISODE_COLS} FROM episodes "
                    f"WHERE id IN ({','.join('?' * len(chunk))})", chunk,
                ).fetchall()
            for r in rows:
                out[r["id"]] = _row_to_episode(r)
        return out

    def set_episode_vectors(self, vectors: dict[str, list[float]], embedding_model: str) -> None:
        rows = [(np.asarray(vector, dtype=np.float32).tobytes(), embedding_model, episode_id)
                for episode_id, vector in vectors.items() if vector]
        if not rows:
            return
        with self._lock:
            self._db.executemany(
                "UPDATE episodes SET embedding = ?, embedding_model = ? WHERE id = ?", rows)
            self._commit()

    def episode_vectors_of(
        self, episode_ids: list[str], embedding_model: str
    ) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        ids = list(dict.fromkeys(episode_ids))
        for start in range(0, len(ids), 500):
            chunk = ids[start:start + 500]
            with self._lock:
                rows = self._db.execute(
                    f"SELECT id, embedding FROM episodes WHERE id IN ({','.join('?' * len(chunk))}) "
                    "AND embedding IS NOT NULL AND embedding_model = ?",
                    (*chunk, embedding_model),
                ).fetchall()
            out.update((r["id"], np.frombuffer(r["embedding"], dtype=np.float32)) for r in rows)
        return out

    def episodes_to_embed(self, embedding_model: str, *, limit: int = 1000) -> list[Episode]:
        with self._lock:
            rows = self._db.execute(
                f"SELECT {_EPISODE_COLS} FROM episodes "
                "WHERE embedding_model IS NOT ? ORDER BY rowid LIMIT ?",
                (embedding_model, limit),
            ).fetchall()
        return [_row_to_episode(r) for r in rows]

    def episode_keyword_scores(self, query: str, episode_ids: list[str]) -> dict[str, float]:
        tokens = _WORD_RE.findall(query)
        ids = list(dict.fromkeys(episode_ids))
        if not tokens or not ids:
            return {}
        match = " OR ".join(f'"{t}"' for t in tokens[:32])
        out: dict[str, float] = {}
        for start in range(0, len(ids), 500):
            chunk = ids[start:start + 500]
            with self._lock:
                rows = self._db.execute(
                    "SELECT e.id, bm25(episodes_fts) AS rank_score FROM episodes_fts "
                    "JOIN episodes e ON e.rowid = episodes_fts.rowid "
                    f"WHERE episodes_fts MATCH ? AND e.id IN ({','.join('?' * len(chunk))})",
                    (match, *chunk),
                ).fetchall()
            # bm25() is lower-is-better (negative); flipped to higher-is-better
            out.update((r["id"], -float(r["rank_score"])) for r in rows)
        return out

    def evidence_episodes(self, episode_ids: list[str]) -> list[Episode]:
        ids = list(dict.fromkeys(episode_ids))
        if not ids:
            return []
        episodes: list[tuple[int, Episode]] = []
        for start in range(0, len(ids), 500):
            chunk = ids[start:start + 500]
            with self._lock:
                rows = self._db.execute(
                    f"SELECT rowid AS seq, {_EPISODE_COLS} FROM episodes "
                    f"WHERE id IN ({','.join('?' * len(chunk))}) AND withheld_at IS NULL",
                    chunk,
                ).fetchall()
            episodes.extend((r["seq"], _row_to_episode(r)) for r in rows)
        if not episodes:
            return []
        # the memories resting on an episode are of the episode's own user
        users = sorted({e.user_id for _, e in episodes if e.user_id is not None})
        owner = " OR ".join(
            ([f"m.user_id IN ({','.join('?' * len(users))})"] if users else [])
            + (["m.user_id IS NULL"] if any(e.user_id is None for _, e in episodes) else []))
        wanted = [e.id for _, e in episodes]
        in_use: set[str] = set()
        removed: set[str] = set()
        for start in range(0, len(wanted), 500):
            chunk = wanted[start:start + 500]
            with self._lock:
                rows = self._db.execute(
                    "SELECT j.value AS episode_id, m.invalid_at, m.superseded_by, "
                    f"({_history_clause('m')}) AS history "
                    "FROM memories m, json_each(m.source_episode_ids) j "
                    f"WHERE ({owner}) AND j.value IN ({','.join('?' * len(chunk))})",
                    (*users, *chunk),
                ).fetchall()
            for r in rows:
                # a memory kept as history rests on what was said while it
                # held: its turns are shown as a memory's in use are
                if r["invalid_at"] is None or r["history"]:
                    in_use.add(r["episode_id"])
                elif r["superseded_by"] is None:
                    removed.add(r["episode_id"])
        shown = [(e.created_at, seq, e) for seq, e in episodes
                 if e.id in in_use and e.id not in removed]
        return [e for *_, e in sorted(shown, key=lambda item: (item[0], item[1]))]

    def history_ids(self, memory_ids: list[str]) -> set[str]:
        ids = list(dict.fromkeys(memory_ids))
        out: set[str] = set()
        for start in range(0, len(ids), 500):
            chunk = ids[start:start + 500]
            with self._lock:
                rows = self._db.execute(
                    f"SELECT id FROM memories WHERE id IN ({','.join('?' * len(chunk))}) "
                    f"AND {_history_clause('memories')}",
                    chunk,
                ).fetchall()
            out.update(r["id"] for r in rows)
        return out

    # -- memories -------------------------------------------------------
    def insert_memory(self, memory: Memory, embedding: list[float] | None = None) -> Memory:
        blob = np.asarray(embedding, dtype=np.float32).tobytes() if embedding else None
        if memory.valid_from is None:
            memory.valid_from = memory.created_at
        with self._lock:
            self._db.execute(
                f"INSERT INTO memories ({_MEMORY_COLS}, embedding) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    memory.id,
                    memory.content,
                    memory.memory_type,
                    memory.user_id,
                    memory.agent_id,
                    memory.run_id,
                    memory.importance,
                    json.dumps(memory.categories),
                    json.dumps(memory.entities),
                    json.dumps(memory.metadata),
                    memory.created_at,
                    memory.updated_at,
                    memory.valid_from,
                    memory.invalid_at,
                    memory.superseded_by,
                    json.dumps(memory.source_episode_ids),
                    memory.embedding_model,
                    blob,
                ),
            )
            # the column as filed: a tag merged away names its survivor
            memory.categories = self._file_tags_locked(
                memory.id, memory.categories, memory.scope(),
                stored=json.loads(json.dumps(memory.categories)),
            )
            if embedding and memory.embedding_model:
                self._ann_add(memory.id, embedding, memory.embedding_model)
            self._commit()
        return memory

    def update_memory(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        embedding: list[float] | None = None,
        embedding_model: str | None = None,
        importance: float | None = None,
        memory_type: str | None = None,
        categories: list[str] | None = None,
        entities: list[str] | None = None,
        mentions: list[EntityMention] | None = None,
        metadata: dict[str, Any] | None = None,
        source_episode_ids: list[str] | None = None,
        touch: bool = True,
    ) -> Memory | None:
        from ..models import utcnow

        if mentions is not None and any(m.memory_id != memory_id for m in mentions):
            raise ValueError("replacement mention belongs to another memory")

        # touch=False is for housekeeping (tagging, backfill markers, re-embedding):
        # it changes stored fields without counting as a content edit, so the
        # memory's updated_at - which drives recency ranking and decay age - is
        # left alone. Only genuine content changes should move that clock.
        sets: list[str] = ["updated_at = ?"] if touch else []
        params: list[Any] = [utcnow()] if touch else []
        if content is not None:
            sets.append("content = ?")
            params.append(content)
        if embedding is not None:
            sets.append("embedding = ?")
            params.append(np.asarray(embedding, dtype=np.float32).tobytes())
        if embedding_model is not None:
            sets.append("embedding_model = ?")
            params.append(embedding_model)
        if importance is not None:
            sets.append("importance = ?")
            params.append(importance)
        if memory_type is not None:
            sets.append("memory_type = ?")
            params.append(memory_type)
        if categories is not None:
            sets.append("categories = ?")
            params.append(json.dumps(categories))
        if entities is not None:
            sets.append("entities = ?")
            params.append(json.dumps(entities))
        if metadata is not None:
            sets.append("metadata = ?")
            params.append(json.dumps(metadata))
        if source_episode_ids is not None:
            sets.append("source_episode_ids = ?")
            params.append(json.dumps(source_episode_ids))
        if not sets:  # nothing to change (touch=False with no fields)
            return self.get_memory(memory_id)
        with self._lock:
            cur = self._db.execute(
                f"UPDATE memories SET {', '.join(sets)} WHERE id = ?", (*params, memory_id)
            )
            if cur.rowcount and mentions is not None:
                old_entity_ids = {
                    row["entity_id"] for row in self._db.execute(
                        "SELECT DISTINCT entity_id FROM entity_mentions WHERE memory_id = ?",
                        (memory_id,),
                    ).fetchall()
                }
                # The named mentions are replaced; the tags' mentions follow the
                # categories column, and are filed with it below.
                self._db.execute(
                    "DELETE FROM entity_mentions WHERE memory_id = ? AND entity_id NOT IN "
                    "(SELECT id FROM entities WHERE entity_type = ?)",
                    (memory_id, TOPIC_TYPE),
                )
                # each with what decided it and the type extraction gave it
                self._insert_mentions_locked(mentions)
                affected = old_entity_ids | {m.entity_id for m in mentions}
                if affected:
                    placeholders = ",".join("?" * len(affected))
                    self._db.execute(
                        "UPDATE entities SET updated_at = ?, description_updated_at = NULL "
                        f"WHERE id IN ({placeholders})",
                        (utcnow(), *sorted(affected)),
                    )
                    self._settle_types_locked(affected)
            if cur.rowcount and (categories is not None or mentions is not None):
                row = self._db.execute(
                    "SELECT categories, user_id, agent_id, run_id FROM memories WHERE id = ?",
                    (memory_id,),
                ).fetchone()
                stored = json.loads(row["categories"])
                self._file_tags_locked(memory_id, stored, _row_scope(row), stored=stored)
            if cur.rowcount and embedding is not None and embedding_model is not None:
                self._ann_add(memory_id, embedding, embedding_model)
            if cur.rowcount and touch and mentions is None:
                self._db.execute(
                    "UPDATE entities SET updated_at = ?, description_updated_at = NULL "
                    "WHERE id IN (SELECT entity_id FROM entity_mentions WHERE memory_id = ?)",
                    (utcnow(), memory_id),
                )
            self._commit()
        if cur.rowcount == 0:
            return None
        return self.get_memory(memory_id)

    def list_pending_memories(
        self, limit: int = 100, *, due_before: str | None = None
    ) -> list[Memory]:
        due_clause = ""
        params: list[Any] = []
        if due_before is not None:
            due_clause = (
                "AND (json_extract(metadata, '$._enrichment.next_attempt_at') IS NULL "
                "OR json_extract(metadata, '$._enrichment.next_attempt_at') <= ?) "
            )
            params.append(due_before)
        with self._lock:
            rows = self._db.execute(
                f"SELECT {_MEMORY_COLS} FROM memories "
                "WHERE invalid_at IS NULL "
                "AND json_extract(metadata, '$.pending_distillation') = 1 "
                f"{due_clause}ORDER BY created_at, id LIMIT ?",
                (*params, limit),
            ).fetchall()
        return [_row_to_memory(row) for row in rows]

    def set_memory_timestamp(self, memory_id: str, updated_at: str) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE memories SET updated_at = ? WHERE id = ?", (updated_at, memory_id)
            )
            self._commit()

    def revalidate_memory(self, memory_id: str) -> Memory | None:
        """Undo an invalidation: the memory is believed true again.

        Mirrors ``invalidate_memory`` exactly - the vector rejoins the ANN
        index and relations whose evidence this memory is come back with it.
        """
        from ..models import utcnow

        with self._lock:
            row = self._db.execute(
                "SELECT embedding, embedding_model, categories, user_id, agent_id, run_id "
                "FROM memories WHERE id = ? AND invalid_at IS NOT NULL",
                (memory_id,),
            ).fetchone()
            if row is None:
                return None
            changed_at = utcnow()
            self._db.execute(
                "UPDATE memories SET invalid_at = NULL, superseded_by = NULL, "
                "updated_at = ? WHERE id = ?",
                (changed_at, memory_id),
            )
            # Its column is filed again: a tag merge while it was gone
            # rewrote it too (``retag_topics``), and one it did not reach
            # names the survivor now.
            stored = json.loads(row["categories"])
            self._file_tags_locked(memory_id, stored, _row_scope(row), stored=stored)
            if row["embedding"] is not None and row["embedding_model"]:
                embedding = np.frombuffer(row["embedding"], dtype=np.float32)
                self._ann_add(memory_id, embedding.tolist(), row["embedding_model"])
            self._db.execute(
                "UPDATE entities SET updated_at = ?, description_updated_at = NULL "
                "WHERE id IN (SELECT entity_id FROM entity_mentions WHERE memory_id = ?)",
                (changed_at, memory_id),
            )
            self._db.execute(
                "UPDATE relations SET invalid_at = NULL WHERE memory_id = ?",
                (memory_id,),
            )
            self._commit()
        return self.get_memory(memory_id)

    def invalidate_memory(
        self, memory_id: str, *, superseded_by: str | None = None, at: str | None = None
    ) -> Memory | None:
        from ..models import utcnow

        # ``at``: a replayed save's time, for the memory and its relations;
        # the entities' ``updated_at`` marks a change to refresh, so the clock
        stamp = at or utcnow()
        with self._lock:
            row = self._db.execute(
                "SELECT updated_at FROM memories WHERE id = ? AND invalid_at IS NULL",
                (memory_id,),
            ).fetchone()
            # a replayed save older than the memory's last change does not
            # move its updated_at back
            updated = later_ts(row["updated_at"], stamp) if row is not None else stamp
            cur = self._db.execute(
                "UPDATE memories SET invalid_at = ?, superseded_by = ?, updated_at = ? "
                "WHERE id = ? AND invalid_at IS NULL",
                (stamp, superseded_by, updated, memory_id),
            )
            if cur.rowcount:
                self._ann_remove(memory_id)
                changed_at = utcnow()
                self._db.execute(
                    "UPDATE entities SET updated_at = ?, description_updated_at = NULL "
                    "WHERE id IN (SELECT entity_id FROM entity_mentions WHERE memory_id = ?)",
                    (changed_at, memory_id),
                )
                self._db.execute(
                    "UPDATE relations SET invalid_at = ? "
                    "WHERE memory_id = ? AND invalid_at IS NULL",
                    (stamp, memory_id),
                )
            self._commit()
        if cur.rowcount == 0:
            return None
        return self.get_memory(memory_id)

    def forget_history(self, memory_id: str) -> Memory | None:
        from ..models import utcnow

        stamp = utcnow()
        with self._lock:
            cur = self._db.execute(
                "UPDATE memories SET superseded_by = NULL, invalid_at = ?, updated_at = ? "
                f"WHERE id = ? AND {_history_clause('memories')}",
                (stamp, stamp, memory_id),
            )
            if cur.rowcount:
                self._db.execute(
                    "UPDATE entities SET updated_at = ?, description_updated_at = NULL "
                    "WHERE id IN (SELECT entity_id FROM entity_mentions WHERE memory_id = ?)",
                    (stamp, memory_id),
                )
            self._commit()
        return self.get_memory(memory_id) if cur.rowcount else None

    def delete_memory(self, memory_id: str) -> bool:
        with self._lock:
            entity_ids = [
                row["entity_id"]
                for row in self._db.execute(
                    "SELECT DISTINCT entity_id FROM entity_mentions WHERE memory_id = ?",
                    (memory_id,),
                ).fetchall()
            ]
            sources = self._db.execute(
                "SELECT source_episode_ids FROM memories WHERE id = ?", (memory_id,)).fetchone()
            cur = self._db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            if cur.rowcount:
                # what it rested on is never shown as evidence again
                episode_ids = json.loads(sources["source_episode_ids"] or "[]")
                if episode_ids:
                    self._db.execute(
                        "UPDATE episodes SET withheld_at = ? WHERE withheld_at IS NULL "
                        f"AND id IN ({','.join('?' * len(episode_ids))})",
                        (utcnow(), *episode_ids))
                self._ann_remove(memory_id)
                self._db.execute("DELETE FROM ann_keys WHERE memory_id = ?", (memory_id,))
                self._db.execute("DELETE FROM entity_mentions WHERE memory_id = ?", (memory_id,))
                self._db.execute("DELETE FROM memory_topics WHERE memory_id = ?", (memory_id,))
                self._db.execute("DELETE FROM relations WHERE memory_id = ?", (memory_id,))
                self._db.execute(
                    "DELETE FROM memory_property_vectors WHERE memory_id = ?", (memory_id,))
                self._db.execute("DELETE FROM memory_questions WHERE memory_id = ?", (memory_id,))
                # what it replaced has nothing standing in for it any more
                self._db.execute(
                    "UPDATE memories SET superseded_by = NULL WHERE superseded_by = ?",
                    (memory_id,))
                if entity_ids:
                    placeholders = ",".join("?" * len(entity_ids))
                    self._db.execute(
                        f"UPDATE entities SET updated_at = ?, description_updated_at = NULL "
                        f"WHERE id IN ({placeholders})",
                        (utcnow(), *entity_ids),
                    )
            self._commit()
        return cur.rowcount > 0

    def replaced_by(self, memory_id: str) -> list[Memory]:
        with self._lock:
            rows = self._db.execute(
                f"SELECT {_MEMORY_COLS} FROM memories WHERE superseded_by = ? ORDER BY id",
                (memory_id,),
            ).fetchall()
        return [_row_to_memory(row) for row in rows]

    def get_memory(self, memory_id: str) -> Memory | None:
        with self._lock:
            row = self._db.execute(
                f"SELECT {_MEMORY_COLS} FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
        return _row_to_memory(row) if row else None

    def list_memories(
        self,
        scope: Scope,
        *,
        include_invalid: bool = False,
        limit: int = 100,
        offset: int = 0,
        categories: list[str] | None = None,
        entity_id: str | None = None,
    ) -> list[Memory]:
        clause, params = _scope_clause(scope)
        cat_clause, cat_params = _category_clause(categories, "memories.id")
        entity_clause, entity_params = _entity_clause(entity_id, "memories.id")
        if not include_invalid:
            clause += " AND invalid_at IS NULL"
        with self._lock:
            rows = self._db.execute(
                f"SELECT {_MEMORY_COLS} FROM memories WHERE {clause} AND {cat_clause} "
                f"AND {entity_clause} ORDER BY updated_at DESC, id LIMIT ? OFFSET ?",
                (*params, *cat_params, *entity_params, limit, offset),
            ).fetchall()
        return [_row_to_memory(r) for r in rows]

    def knowledge_map(self, scope: Scope, *, kind: str = "named") -> dict[str, Any]:
        """Return a content-free, all-memory graph for the dashboard map.

        This is intentionally independent of card pagination. SQL performs the
        aggregation so no memory text or per-memory entity payload is sent to
        the browser and large stores do not trigger an N+1 entity lookup.
        ``kind`` as for ``list_entities``: "named" draws people and things,
        "any" tags too, as the nodes of their topic entities, linked by the
        memories that mention both ends like any other node.
        """
        memory_clause, memory_params = _scope_clause(scope, prefix="m.")
        entity1_clause, entity1_params = _scope_clause(scope, prefix="e1.")
        entity2_clause, entity2_params = _scope_clause(scope, prefix="e2.")

        with self._lock:
            total = self._db.execute(
                f"SELECT COUNT(*) AS count FROM memories m "
                f"WHERE m.invalid_at IS NULL AND {memory_clause}",
                memory_params,
            ).fetchone()["count"]
            entity_memories = self._db.execute(
                "SELECT COUNT(DISTINCT m.id) AS count FROM memories m "
                "JOIN entity_mentions em ON em.memory_id = m.id "
                "JOIN entities e ON e.id = em.entity_id "
                f"WHERE m.invalid_at IS NULL AND e.merged_into IS NULL AND {memory_clause} "
                f"AND {_kind_clause(kind, 'e.')}",
                memory_params,
            ).fetchone()["count"]
            entity_rows = self._db.execute(
                "SELECT e1.id, e1.name, "
                "COALESCE(NULLIF(e1.entity_type, ''), 'untyped') AS entity_type, "
                "m.memory_type, COUNT(DISTINCT m.id) AS count, "
                "MAX(m.created_at) AS last_said "
                "FROM entity_mentions em "
                "JOIN entities e1 ON e1.id = em.entity_id "
                "JOIN memories m ON m.id = em.memory_id "
                "WHERE e1.merged_into IS NULL AND m.invalid_at IS NULL "
                f"AND {entity1_clause} AND {memory_clause} "
                f"AND {_kind_clause(kind, 'e1.')} "
                "GROUP BY e1.id, e1.name, e1.entity_type, m.memory_type "
                "ORDER BY e1.name, e1.id, m.memory_type",
                (*entity1_params, *memory_params),
            ).fetchall()
            entity_edge_rows = self._db.execute(
                "SELECT e1.id AS a, e2.id AS b, "
                "COUNT(DISTINCT m.id) AS weight "
                "FROM entity_mentions em1 "
                "JOIN entity_mentions em2 ON em2.memory_id = em1.memory_id "
                " AND em2.entity_id > em1.entity_id "
                "JOIN entities e1 ON e1.id = em1.entity_id "
                "JOIN entities e2 ON e2.id = em2.entity_id "
                "JOIN memories m ON m.id = em1.memory_id "
                "WHERE e1.merged_into IS NULL AND e2.merged_into IS NULL "
                "AND m.invalid_at IS NULL "
                f"AND {entity1_clause} AND {entity2_clause} AND {memory_clause} "
                f"AND {_kind_clause(kind, 'e1.')} AND {_kind_clause(kind, 'e2.')} "
                "GROUP BY e1.id, e2.id "
                "ORDER BY weight DESC, a, b LIMIT 50000",
                (*entity1_params, *entity2_params, *memory_params),
            ).fetchall()
            # one row per memory and entity it mentions, for a map that draws
            # every memory on its own: the day it was said and its type, no text
            point_rows = self._db.execute(
                "SELECT m.id, m.created_at, m.memory_type, em.entity_id "
                "FROM entity_mentions em "
                "JOIN entities e1 ON e1.id = em.entity_id "
                "JOIN memories m ON m.id = em.memory_id "
                "WHERE e1.merged_into IS NULL AND m.invalid_at IS NULL "
                f"AND {entity1_clause} AND {memory_clause} "
                f"AND {_kind_clause(kind, 'e1.')} "
                "ORDER BY m.created_at, m.id LIMIT 200000",
                (*entity1_params, *memory_params),
            ).fetchall()

        entities: dict[str, dict[str, Any]] = {}
        for row in entity_rows:
            key = f"entity:{row['id']}"
            node = entities.setdefault(
                key,
                {
                    "key": key,
                    "label": row["name"],
                    "kind": "entity",
                    "entity_id": row["id"],
                    "entity_type": row["entity_type"],
                    "count": 0,
                    "type_counts": {},
                    "last_said": "",
                },
            )
            node["count"] += row["count"]
            node["type_counts"][row["memory_type"]] = row["count"]
            # the day the newest memory about it was said: the map's time layouts
            node["last_said"] = max(node["last_said"], str(row["last_said"] or ""))

        points: dict[str, dict[str, Any]] = {}
        for row in point_rows:
            point = points.setdefault(row["id"], {
                "id": row["id"], "said": str(row["created_at"] or ""),
                "type": row["memory_type"], "entities": []})
            point["entities"].append(f"entity:{row['entity_id']}")
        return {
            "memories": total,
            "entity_memories": entity_memories,
            "memory_points": list(points.values()),
            "entities": list(entities.values()),
            "entity_edges": [
                {
                    "a": f"entity:{row['a']}",
                    "b": f"entity:{row['b']}",
                    "weight": row["weight"],
                }
                for row in entity_edge_rows
            ],
        }

    # -- search -----------------------------------------------------------
    def _score_rows(
        self, rows: list[sqlite3.Row], embedding: list[float], limit: int
    ) -> list[tuple[Memory, float]]:
        if not rows or limit <= 0:
            return []
        query = np.asarray(embedding, dtype=np.float32)
        qnorm = np.linalg.norm(query)
        if qnorm == 0:
            return []
        mats = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
        norms = np.linalg.norm(mats, axis=1)
        norms[norms == 0] = 1e-9
        sims = (mats @ query) / (norms * qnorm)
        # the best ``limit``, a tie broken by memory id: rows come in storage
        # order, which two builds of one store (a restore, a bulk import) need
        # not share
        best = np.arange(len(rows))
        if len(rows) > limit:
            cut = np.partition(-sims, limit - 1)[limit - 1]
            best = np.flatnonzero(-sims <= cut)
        order = sorted(best, key=lambda i: (-sims[i], rows[i]["id"]))[:limit]
        return [(_row_to_memory(rows[i]), float(sims[i])) for i in order]

    def vector_search(
        self,
        embedding: list[float],
        embedding_model: str,
        scope: Scope,
        limit: int = 20,
        include_invalid: bool = False,
        categories: list[str] | None = None,
        entity_id: str | None = None,
        history: bool = False,
        among: Any = None,
    ) -> list[tuple[Memory, float]]:
        clause, params = _search_scope_clause(scope, "memories")
        cat_clause, cat_params = _category_clause(categories, "memories.id")
        entity_clause, entity_params = _entity_clause(entity_id, "memories.id")
        among_clause, among_params = _among_clause(among, "memories.id")
        filters = f"{clause} AND {cat_clause} AND {entity_clause} AND {among_clause}"
        filter_params = (*params, *cat_params, *entity_params, *among_params)
        found = self._vector_rows(
            embedding, embedding_model, limit,
            filters if include_invalid else f"{filters} AND memories.invalid_at IS NULL",
            filter_params)
        if history and not include_invalid:
            # out of the ANN index with the rest of what is out of use, and
            # few: an exact scan of them alone
            with self._lock:
                rows = self._db.execute(
                    f"SELECT {_MEMORY_COLS}, embedding FROM memories "
                    f"WHERE {filters} AND {_history_clause('memories')} "
                    "AND embedding IS NOT NULL AND embedding_model = ?",
                    (*filter_params, embedding_model),
                ).fetchall()
            found = sorted(found + self._score_rows(rows, embedding, limit),
                           key=lambda pair: (-pair[1], pair[0].id))[:limit]
        return found

    def _vector_rows(
        self, embedding: list[float], embedding_model: str, limit: int, filters: str,
        params: tuple[Any, ...],
    ) -> list[tuple[Memory, float]]:
        # ANN fast path: over-fetch approximate neighbors, filter in SQL,
        # exact-rescore. Falls back to the full scan if it can't fill `limit`.
        index = self._ann_index(embedding_model, len(embedding))
        if index is not None and index.size >= self._ann_cfg.min_rows:
            k = max(limit * self._ann_cfg.overfetch, 200)
            keys = index.search(embedding, k)
            if keys:
                key_ph = ",".join("?" * len(keys))
                with self._lock:
                    rows = self._db.execute(
                        f"SELECT {_MEMORY_COLS}, embedding FROM memories "
                        f"WHERE id IN (SELECT memory_id FROM ann_keys WHERE key IN ({key_ph})) "
                        f"AND {filters} AND embedding IS NOT NULL AND embedding_model = ?",
                        (*keys, *params, embedding_model),
                    ).fetchall()
                if len(rows) >= limit:
                    return self._score_rows(rows, embedding, limit)
                # restrictive filters starved the ANN candidates -> exact scan

        with self._lock:
            rows = self._db.execute(
                f"SELECT {_MEMORY_COLS}, embedding FROM memories "
                f"WHERE {filters} AND embedding IS NOT NULL AND embedding_model = ?",
                (*params, embedding_model),
            ).fetchall()
        return self._score_rows(rows, embedding, limit)

    def keyword_search(
        self,
        query: str,
        scope: Scope,
        limit: int = 20,
        include_invalid: bool = False,
        categories: list[str] | None = None,
        entity_id: str | None = None,
        history: bool = False,
        among: Any = None,
    ) -> list[tuple[Memory, float]]:
        """BM25 over the memories, each word of the question weighed by how
        rare it is in everything the store holds: its memories and the turns
        they were said in (``_word_weights``). In a store of third-person
        facts "did" and "do" are rare among the memories and common in what
        was said, so they no longer outweigh the name a question asks about.
        Without turns a word weighs as ``bm25()`` weighs it.

        A memory's score is the sum, over the words it holds, of the word's
        ``bm25()`` alone times its weight and the times the question says it
        (``_keyword_scores``, which finds the first ``limit`` without scoring
        every memory a common word is in)."""
        tokens = _WORD_RE.findall(query)
        if not tokens:
            return []
        # one term whatever its case, counted as often as the question says it,
        # as an OR query of the words counts it
        asked = Counter(token.lower() for token in tokens[:32])
        clause, params = _search_scope_clause(scope, "m")
        cat_clause, cat_params = _category_clause(categories, "m.id")
        entity_clause, entity_params = _entity_clause(entity_id, "m.id")
        among_clause, among_params = _among_clause(among, "m.id")
        if not include_invalid:
            clause += (f" AND (m.invalid_at IS NULL OR ({_history_clause('m')}))" if history
                       else " AND m.invalid_at IS NULL")
        with self._lock:
            scores = self._keyword_scores(
                asked, limit,
                f"{clause} AND {cat_clause} AND {entity_clause} AND {among_clause}",
                [*params, *cat_params, *entity_params, *among_params])
            best = sorted(scores, key=lambda memory_id: (-scores[memory_id], memory_id))[:limit]
            rows = []
            for start in range(0, len(best), 500):
                chunk = best[start:start + 500]
                rows += self._db.execute(
                    f"SELECT {_MEMORY_COLS} FROM memories WHERE id IN "
                    f"({','.join('?' * len(chunk))})", chunk,
                ).fetchall()
        found = {row["id"]: _row_to_memory(row) for row in rows}
        return [(found[memory_id], scores[memory_id]) for memory_id in best]

    def _keyword_scores(
        self, asked: Counter[str], limit: int, kept: str, kept_params: list[Any]
    ) -> dict[str, float]:
        """The keyword score of each memory ``kept`` keeps (the search's scope
        and filters, over ``memories m``) that can be among the first
        ``limit``, as ``keyword_search`` scores it: so the first ``limit``
        themselves, ties and all. Every read keeps to them in SQL, so another
        account's memories are walked in the index and never scored; the
        counts behind the weights are the whole index's, as ``bm25()``'s
        are. Caller holds the lock.

        A word held by at most ``_WHOLE_WORD_ROWS`` memories is scored in
        each of them. A commoner one is not (the exact top-k of MaxScore):
        a word adds less than its bound to any memory, its weight times the
        times it is asked times its idf times k1 + 1 (``_BM25_TERM_BOUND``).
        The k-th best of what the rarer words give is a floor under the k-th
        best score. Common words whose bounds add up to less than it cannot
        lift a memory that holds none of the others to it: they are set
        aside, and scored only for the memories found otherwise. The other
        common words are read in one query of all of them, the memories
        ranked by the sum of their ``bm25()``, the first four times
        ``limit`` of them; the floor rises with what is read. When the most
        that sum can give a memory past those (``_band_bounds``) is below the
        floor, none can reach it; where it is not, every word is scored in
        every memory, as a question of common words alone needs. The words
        set aside count at their bounds for every memory read, so all are set
        aside where all can be, and otherwise only as many as add up to
        ``_ASIDE_SHARE`` of the floor. Each memory that can still reach the
        floor is then scored in every word it holds."""
        counts = self._word_counts(list(asked))
        weights = self._word_weights(list(asked), counts)
        # a word in no memory adds nothing
        words = [word for word in asked if counts.hits[word][0]]
        factor = {word: asked[word] * weights[word] for word in words}
        idf = {word: _bm25_idf(counts.hits[word][0], counts.memories) for word in words}
        bound = {word: factor[word] * idf[word] * _BM25_TERM_BOUND * (1 + _ROUNDING)
                 for word in words}
        # word -> memory rowid -> the word's bm25() alone in that memory, and
        # the id of each memory read
        ranks: dict[str, dict[int, float]] = {word: {} for word in words}
        ids: dict[int, str] = {}
        scored: set[str] = set()
        # Every read keeps to the search's scope and filters before bm25()
        # reads a memory: the full-text index holds every account's memories.
        # CROSS JOIN: the words' rows first, each then looked up by its rowid
        # (the planner would otherwise read the scope's index for every
        # memory in it).
        kept_rows = ("FROM memories_fts CROSS JOIN memories m ON m.rowid = memories_fts.rowid "
                     f"WHERE memories_fts MATCH ? AND {kept}")

        def rows(sql: str, args: list[Any]) -> sqlite3.Cursor:
            cursor = self._db.cursor()
            cursor.row_factory = None
            return cursor.execute(sql, args)

        def score(word: str) -> None:
            scored.add(word)
            for rowid, memory_id, rank in rows(
                    f"SELECT m.rowid, m.id, bm25(memories_fts) {kept_rows}",
                    [f'"{word}"', *kept_params]):
                ranks[word][rowid] = rank
                ids[rowid] = memory_id

        def score_among(word: str, among: list[int]) -> None:
            # memories kept already, read along the word's rows between the
            # first and the last of them (the + keeps SQLite from matching the
            # word again for each memory; bm25() reads the same either way,
            # its idf counted over the whole index)
            if among:
                ranks[word].update(rows(
                    "SELECT rowid, bm25(memories_fts) FROM memories_fts WHERE memories_fts "
                    "MATCH ? AND rowid BETWEEN ? AND ? "
                    "AND +rowid IN (SELECT value FROM json_each(?))",
                    [f'"{word}"', min(among), max(among), json.dumps(sorted(among))]))

        def kth_best(values: Iterable[float]) -> float:
            best = heapq.nlargest(limit, values)
            return best[-1] if len(best) == limit else -math.inf

        def everywhere() -> dict[str, float]:
            for word in words:
                if word not in scored:
                    score(word)
            return _summed(words, factor, ranks, ids)

        common = [word for word in words if counts.hits[word][0] > _WHOLE_WORD_ROWS]
        for word in words:
            if word not in common:
                score(word)
        if limit < 1 or not common:
            return everywhere()
        # what the rarer words give each memory kept, and a floor under its score
        given: dict[int, float] = {}
        for word in words:
            for rowid, rank in ranks[word].items():
                given[rowid] = given.get(rowid, 0.0) - factor[word] * rank
        floor = {rowid: part * (1 - _ROUNDING) for rowid, part in given.items()}
        needed = kth_best(floor.values())
        aside: list[str] = []
        rest = 0.0
        share = 1.0 if sum(bound[word] for word in common) < needed else _ASIDE_SHARE
        for word in sorted(common, key=bound.__getitem__):
            if rest + bound[word] >= needed * share:
                break
            aside.append(word)
            rest += bound[word]
        # each word as often as the question says it, so that its weight
        # alone tells the words apart
        read = [word for word in common if word not in aside for _ in range(asked[word])]
        least, most = _band_bounds([(weights[word], idf[word]) for word in read])
        summed: dict[int, float] = {}  # rowid -> the read words' bm25() summed, flipped
        last = 0.0
        if read:
            size = 4 * max(limit, 16)
            batch = list(rows(
                f"SELECT m.rowid, m.id, bm25(memories_fts) {kept_rows} ORDER BY 3 LIMIT ?",
                [" OR ".join(f'"{word}"' for word in read), *kept_params, size]))
            for rowid, memory_id, rank in batch:
                ids[rowid] = memory_id
                summed[rowid] = -rank
                floor[rowid] = given.get(rowid, 0.0) * (1 - _ROUNDING) + least(-rank)
            needed = kth_best(floor.values())
            # every memory kept and not read holds at most the last sum read,
            # or none of the words
            last = -batch[-1][2] if len(batch) == size else 0.0
            if last and most(last) + rest >= needed:
                # what the words can give does not tell the first apart from
                # the rest: every word is scored in every memory
                return everywhere()
        within = [rowid for rowid in ids if given.get(rowid, 0.0) * (1 + _ROUNDING)
                  + most(summed.get(rowid, last)) + rest >= needed]
        for word in common:
            score_among(word, within)
        return _summed(words, factor, ranks, {rowid: ids[rowid] for rowid in within})

    def _word_counts(self, words: list[str]) -> _WordCounts:
        """How many memories and turns the full-text indexes hold, and how many
        of each hold each word, of every account (as ``bm25()`` counts them).
        A word written in ASCII is one term of the index, its letters lowered,
        and is counted from the index's list of terms (``fts5vocab``) without
        reading its rows; another is counted by matching it. Caller holds the
        lock."""
        def rows(table: str) -> int:
            return int(self._db.execute(f"SELECT count(*) FROM {table}_docsize").fetchone()[0])

        def hits(table: str, word: str) -> int:
            if word.isascii():
                row = self._db.execute(
                    f"SELECT doc FROM temp.{table}_terms WHERE term = ?", (word.lower(),)
                ).fetchone()
                return int(row[0]) if row else 0
            return int(self._db.execute(
                f"SELECT count(*) FROM {table} WHERE {table} MATCH ?", (f'"{word}"',)
            ).fetchone()[0])

        memories, turns = rows("memories_fts"), rows("episodes_fts")
        counts: dict[str, tuple[int, int]] = {}
        for word in words:
            in_memories = hits("memories_fts", word)
            # a word in no memory weighs nothing that is read
            counts[word] = (in_memories,
                            hits("episodes_fts", word) if turns and in_memories else 0)
        return _WordCounts(memories, turns, counts)

    def _word_weights(
        self, words: list[str], counts: _WordCounts | None = None
    ) -> dict[str, float]:
        """For each word, its inverse document frequency over the memories and
        the turns together, as a factor on the one ``bm25()`` reads over the
        memories alone (both as FTS5 computes it). A turn is what someone
        said, in the words a question is asked in; a memory restates it as a
        fact, with fewer of the words that only build a sentence. Caller
        holds the lock."""
        counts = counts or self._word_counts(words)
        weights: dict[str, float] = {}
        for word in words:
            in_memories, in_turns = counts.hits[word]
            weights[word] = (_bm25_idf(in_memories + in_turns, counts.memories + counts.turns)
                             / _bm25_idf(in_memories, counts.memories))
        return weights

    # -- events -----------------------------------------------------------
    def add_event(self, event: MemoryEvent) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO memory_events (id, memory_id, event, old_content, new_content, "
                "reason, actor, created_at, kind) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    event.id,
                    event.memory_id,
                    event.event,
                    event.old_content,
                    event.new_content,
                    event.reason,
                    event.actor,
                    event.created_at,
                    event.kind,
                ),
            )
            self._commit()

    def history(self, memory_id: str) -> list[MemoryEvent]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM memory_events WHERE memory_id = ? ORDER BY created_at",
                (memory_id,),
            ).fetchall()
        return [
            MemoryEvent(
                id=r["id"],
                memory_id=r["memory_id"],
                event=r["event"],
                old_content=r["old_content"],
                new_content=r["new_content"],
                reason=r["reason"],
                actor=r["actor"],
                created_at=r["created_at"],
                kind=r["kind"],
            )
            for r in rows
        ]

    # -- normalized topics -------------------------------------------------
    def list_topics(self, scope: Scope, *, limit: int = 1000) -> list[Topic]:
        clause, params = _scope_clause(scope)
        with self._lock:
            rows = self._db.execute(
                f"SELECT * FROM topics WHERE {clause} ORDER BY normalized, id LIMIT ?",
                (*params, limit),
            ).fetchall()
        return [self._row_to_topic(row) for row in rows]

    def purge_orphan_entities(
        self, scope: Scope, *, reason: str = "nothing referenced it"
    ) -> int:
        clause, params = _scope_clause(scope, prefix="e.")
        with self._lock:
            rows = self._db.execute(
                "SELECT e.id FROM entities e "
                # a name or a tag (a topic entity): a tag no memory carries
                # any more files nothing, and a memory tagged so later makes
                # a fresh topic
                f"WHERE e.merged_into IS NULL AND {clause} "
                # nothing mentions it ...
                "AND NOT EXISTS (SELECT 1 FROM entity_mentions m WHERE m.entity_id = e.id) "
                # ... it is on no typed edge ...
                "AND NOT EXISTS (SELECT 1 FROM relations r "
                "                WHERE r.subject = e.id OR r.object = e.id) "
                # ... and nothing was ever merged into it, so no id redirects here
                "AND NOT EXISTS (SELECT 1 FROM entities o WHERE o.merged_into = e.id)",
                params,
            ).fetchall()
            ids = [row["id"] for row in rows]
            if not ids:
                return 0
            # Snapshotted like any other removal: an unreferenced entity is
            # still a name the user may recognise, so it lands in the trash
            # rather than going away for good.
            purged = sum(self._retire_locked(entity_id, reason) for entity_id in ids)
            self._commit()
        return purged

    def retag_topics(
        self, scope: Scope, remove: set[str], add: str | None, *, exact_user: bool = False
    ) -> int:
        """Rewrite the ``categories`` column of the memories carrying a tag in
        ``remove``: those tags go, ``add`` (if any) comes in. Each tag is read
        as filed (``_file_tags_locked``), so one naming a tag merged into one
        of them goes too, and the column is filed through the same function,
        which brings its index and mentions in line. Invalid memories are
        rewritten too, so one restored later does not bring back a tag merged
        or deleted meanwhile; the count returned is of the active ones.
        ``exact_user`` confines it to ``scope.user_id`` even when that is None
        (the memories without a user), as a topic entity is confined. With no
        ``add`` (``delete_tag``) the topic entity of each tag removed is
        retired once nothing mentions it, so no tag deleted stays active
        with nothing filed under it."""
        normalized = {item.strip().lower() for item in remove if item.strip()}
        if not normalized:
            return 0
        add = add.strip().lower() if add and add.strip() else None
        placeholders = ",".join("?" * len(normalized))
        clause = _exact_scope_clause if exact_user else _scope_clause
        topic_clause, topic_params = clause(scope, prefix="t.")
        memory_clause, memory_params = clause(scope, prefix="m.")
        with self._lock:
            topic_rows = self._db.execute(
                f"SELECT t.* FROM topics t WHERE {topic_clause} "
                f"AND t.normalized IN ({placeholders})",
                (*topic_params, *sorted(normalized)),
            ).fetchall()
            old_ids = {row["id"] for row in topic_rows}
            rows = self._db.execute(
                "SELECT DISTINCT m.id, m.categories, m.user_id, m.agent_id, m.run_id, "
                "m.invalid_at FROM memories m JOIN memory_topics mt ON mt.memory_id = m.id "
                "JOIN topics t ON t.id = mt.topic_id "
                f"WHERE {memory_clause} AND t.normalized IN ({placeholders})",
                (*memory_params, *sorted(normalized)),
            ).fetchall()
            changed = 0
            cache: dict[Any, Any] = {}  # one resolution per tag for the whole rewrite
            for row in rows:
                categories = json.loads(row["categories"])
                kept: list[str] = []
                for item in categories:
                    written = str(item).strip()
                    if not written or written.lower() in normalized:
                        continue
                    name, _ = self._filed_tag_locked(written, row["user_id"], cache)
                    if name.lower() not in normalized:
                        kept.append(name)
                if add and add not in {name.lower() for name in kept}:
                    kept.append(add)
                filed = self._file_tags_locked(
                    row["id"], kept, _row_scope(row), stored=categories,
                    provenance="user", cache=cache,
                )
                if filed != categories and row["invalid_at"] is None:
                    changed += 1

            for old_id in old_ids:
                self._db.execute(
                    "DELETE FROM topics WHERE id = ? "
                    "AND NOT EXISTS (SELECT 1 FROM memory_topics WHERE topic_id = ?)",
                    (old_id, old_id),
                )
            if not add:
                # a deleted tag's topic goes with its last mention, retired as
                # any entity is (snapshot kept), tags merged into it with it
                entity_clause, entity_params = clause(_user_scope(scope), prefix="e.")
                for row in self._db.execute(
                    "SELECT e.id FROM entities e WHERE e.entity_type = ? "
                    f"AND e.merged_into IS NULL AND {entity_clause} "
                    f"AND e.normalized IN ({placeholders}) "
                    "AND NOT EXISTS (SELECT 1 FROM entity_mentions m WHERE m.entity_id = e.id)",
                    (TOPIC_TYPE, *entity_params, *sorted(normalized)),
                ).fetchall():
                    self._retire_locked(row["id"], "tag deleted")
            self._commit()
        return changed

    def tag_namespaces(self, names: Iterable[str]) -> list[str | None]:
        wanted = sorted({str(name).strip().lower() for name in names if str(name).strip()})
        if not wanted:
            return []
        marks = ",".join("?" * len(wanted))
        with self._lock:
            rows = self._db.execute(
                "SELECT user_id FROM entities WHERE entity_type = ? AND merged_into IS NULL "
                f"AND normalized IN ({marks}) "
                "UNION SELECT m.user_id FROM memories m "
                "JOIN memory_topics mt ON mt.memory_id = m.id "
                f"JOIN topics t ON t.id = mt.topic_id WHERE t.normalized IN ({marks})",
                (TOPIC_TYPE, *wanted, *wanted),
            ).fetchall()
        # None first, then by name: a stable order for the edits made one by one
        return sorted((row["user_id"] for row in rows),
                      key=lambda user: (user is not None, user or ""))

    # -- tags as topic entities ----------------------------------------------
    # A tag is an entity of type ``TOPIC_TYPE``, one per namespace (``user_id``
    # exactly) and normalized tag, created the first time a memory carries it.
    #
    # One invariant ties a memory's ``categories`` column, the legacy index
    # the filters read (``memory_topics``, ``_category_clause``) and its tag
    # mentions, and one function holds it, ``_file_tags_locked``, which every
    # writer of a memory's tags goes through: insert, update, revalidate,
    # backup import, ``retag_topics`` and the migrations (at open and
    # ``tags_to_topics``).
    # - The column names surviving tags only. A tag merged into another topic
    #   is written as that topic's name, following tombstones topic to topic;
    #   one merged into a named thing keeps the last tag name of its chain and
    #   its mention goes to the thing. A name merged away never gets a fresh
    #   topic.
    # - The index holds the column's names, and the memory mentions the entity
    #   each resolves to, once, under the name as the column writes it.
    # Each tag is resolved once per write (``_filed_tag_locked``), and once per
    # call over a bulk write (its ``cache``).
    def _active_topic_locked(self, normalized: str, user_id: str | None) -> str | None:
        """The id of the active topic entity named ``normalized`` of ``user_id``
        (one at most: ``_ACTIVE_TOPIC_INDEX``)."""
        row = self._db.execute(
            "SELECT id FROM entities WHERE entity_type = ? AND user_id IS ? "
            "AND normalized = ? AND merged_into IS NULL ORDER BY created_at, id LIMIT 1",
            (TOPIC_TYPE, user_id, normalized),
        ).fetchone()
        return row["id"] if row is not None else None

    def _merged_topic_locked(
        self, normalized: str, user_id: str | None
    ) -> tuple[str, str] | None:
        """For a tag merged away (a topic entity of that name folded into
        another): (the tag name a column files it under, the active entity
        its mention goes to). Its tombstone is followed topic to topic
        ("taxes" into "tax" into "levies": "levies"); where the chain reaches
        a named thing ("bildy" the tag into "Bildy" the product) the name is
        the last tag's and the entity the thing. None when no topic of that
        name was merged."""
        for tombstone in self._db.execute(
            "SELECT id, merged_into FROM entities WHERE entity_type = ? AND user_id IS ? "
            "AND normalized = ? AND merged_into IS NOT NULL ORDER BY created_at, id",
            (TOPIC_TYPE, user_id, normalized),
        ).fetchall():
            name = normalized
            current: str | None = tombstone["merged_into"]
            seen = {tombstone["id"]}
            while current is not None and current not in seen:
                seen.add(current)
                row = self._db.execute(
                    "SELECT id, entity_type, normalized, merged_into FROM entities WHERE id = ?",
                    (current,),
                ).fetchone()
                if row is None:
                    break
                if row["entity_type"] == TOPIC_TYPE:
                    name = row["normalized"]
                if row["merged_into"] is None:
                    return name, row["id"]
                current = row["merged_into"]
        return None

    def _create_topic_locked(self, normalized: str, user_id: str | None) -> tuple[str, bool]:
        """(id, created) of a new active topic ``normalized`` of ``user_id``:
        inserted or ignored against ``_ACTIVE_TOPIC_INDEX`` and read back, so
        two processes creating one topic at once end with one. The index keys
        the namespace the lookup reads, so an ignored insert always finds the
        topic that won; it never yields nothing (a mention is never dropped)."""
        entity = Entity(
            name=normalized, normalized=normalized, entity_type=TOPIC_TYPE, user_id=user_id
        )
        cur = self._db.execute(
            "INSERT OR IGNORE INTO entities (id, name, normalized, entity_type, user_id, "
            "agent_id, run_id, description, description_updated_at, metadata, merged_into, "
            "created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                entity.id, entity.name, entity.normalized, entity.entity_type,
                entity.user_id, None, None, None, None, "{}", None,
                entity.created_at, entity.updated_at,
            ),
        )
        if cur.rowcount:
            return entity.id, True
        # another process created it between the lookup and the insert
        existing = self._active_topic_locked(normalized, user_id)
        if existing is None:
            raise sqlite3.IntegrityError(
                f"topic {normalized!r} of {user_id!r} was neither created nor found")
        return existing, False

    def _resolve_tag_locked(
        self, normalized: str, user_id: str | None, *, create: bool = True
    ) -> tuple[str | None, str | None, bool]:
        """How the tag ``normalized`` of ``user_id`` is filed: (the name a
        column writes instead of it, None when it is written as it is; the
        entity its mention goes to; whether that entity was created now).

        The active topic of that name; else, for a name merged away, its
        survivor (``_merged_topic_locked``): a merge rewrites every column it
        reaches, and one it did not (a backup restored, a column written
        straight to the backend) must not bring the merged tag back; else a
        new topic, when ``create``."""
        active = self._active_topic_locked(normalized, user_id)
        if active is not None:
            return None, active, False
        merged = self._merged_topic_locked(normalized, user_id)
        if merged is not None:
            name, entity_id = merged
            return (name if name != normalized else None), entity_id, False
        if not create:
            return None, None, False
        entity_id, created = self._create_topic_locked(normalized, user_id)
        return None, entity_id, created

    def _filed_tag_locked(
        self, written: str, user_id: str | None, cache: dict[Any, Any],
        counts: dict[str, Any] | None = None,
    ) -> tuple[str, str | None]:
        """(the name the column writes, the entity mentioned) for one tag as
        written (stripped), resolved once per ``cache``: a bulk write passes
        one cache for all its memories. ``counts`` tallies the entities
        created and found (``tags_to_topics``)."""
        normalized = written.lower()
        key = ("tag", user_id, normalized)
        if key not in cache:
            renamed, entity_id, created = self._resolve_tag_locked(normalized, user_id)
            cache[key] = (renamed, entity_id)
            if counts is not None and entity_id is not None:
                counts["entities_created" if created else "entities_existing"] += 1
        renamed, entity_id = cache[key]
        return renamed or written, entity_id

    def _file_tags_locked(
        self,
        memory_id: str,
        tags: list[Any] | None,
        scope: Scope,
        *,
        stored: list[Any] | None = None,
        provenance: str = "memory",
        cache: dict[Any, Any] | None = None,
        counts: dict[str, Any] | None = None,
    ) -> list[str]:
        """File a memory's tags: the one path by which its ``categories``
        column, its legacy index and its tag mentions are written, so the
        three agree (the invariant above). Returns the column as written.

        Each tag is resolved once (``_filed_tag_locked``); a tag merged away
        is written as its survivor, and a tag named twice once. The column is
        written when it differs from ``stored`` (read when None); the index
        and the mentions are brought in line with it, adding and removing
        what differs, so filing an unchanged memory writes nothing. Mentions
        of named things are never removed here. ``scope`` is the memory's."""
        cache = {} if cache is None else cache
        column: list[str] = []
        filed: dict[str, str] = {}  # entity id -> the name it is mentioned under
        seen: set[str] = set()
        for raw in tags or []:
            written = str(raw).strip()
            if not written:
                continue
            name, entity_id = self._filed_tag_locked(written, scope.user_id, cache, counts)
            if name.lower() in seen:
                continue
            seen.add(name.lower())
            column.append(name)
            if entity_id is not None:
                filed.setdefault(entity_id, name)
        if stored is None:
            row = self._db.execute(
                "SELECT categories FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
            stored = json.loads(row["categories"]) if row is not None else column
        if column != stored:
            self._db.execute(
                "UPDATE memories SET categories = ? WHERE id = ?",
                (json.dumps(column), memory_id),
            )
        self._index_tags_locked(memory_id, column, scope, provenance, cache)
        self._mention_tags_locked(memory_id, filed, counts)
        return column

    def _index_tags_locked(
        self, memory_id: str, column: list[str], scope: Scope, provenance: str,
        cache: dict[Any, Any],
    ) -> None:
        """The legacy index of a memory's column (``_file_tags_locked``): a
        link to the ``topics`` row of each name in the memory's scope."""
        wanted: set[str] = set()
        for name in column:
            key = ("topic", scope.user_id, scope.agent_id, scope.run_id, name.lower())
            if key not in cache:
                cache[key] = self._topic_locked(name, scope, provenance).id
            wanted.add(cache[key])
        present = {
            row["topic_id"] for row in self._db.execute(
                "SELECT topic_id FROM memory_topics WHERE memory_id = ?", (memory_id,)
            ).fetchall()
        }
        for topic_id in sorted(present - wanted):
            self._db.execute(
                "DELETE FROM memory_topics WHERE memory_id = ? AND topic_id = ?",
                (memory_id, topic_id),
            )
        for topic_id in sorted(wanted - present):
            self._db.execute(
                "INSERT OR IGNORE INTO memory_topics (memory_id, topic_id) VALUES (?,?)",
                (memory_id, topic_id),
            )

    def _mention_tags_locked(
        self, memory_id: str, filed: dict[str, str], counts: dict[str, Any] | None
    ) -> None:
        """A memory's tag mentions (``_file_tags_locked``): one mention of each
        entity in ``filed`` under its name; a mention of a topic entity it no
        longer names goes. Mentions of named things stay, and one of a thing a
        tag was merged into counts as that tag's."""
        rows = self._db.execute(
            "SELECT em.id, em.entity_id, em.surface, e.entity_type FROM entity_mentions em "
            "LEFT JOIN entities e ON e.id = em.entity_id "
            "WHERE em.memory_id = ? ORDER BY em.created_at, em.id",
            (memory_id,),
        ).fetchall()
        present: set[str] = set()
        stale: list[str] = []
        for row in rows:
            entity_id = row["entity_id"]
            if row["entity_type"] == TOPIC_TYPE:
                if entity_id not in filed or entity_id in present:
                    stale.append(row["id"])
                    continue
                surface = filed[entity_id]
                if row["surface"].strip().lower() != surface.lower():
                    # moved here by a merge: "taxes" on the entity of "tax"
                    self._db.execute(
                        "UPDATE entity_mentions SET surface = ? WHERE id = ?",
                        (surface, row["id"]),
                    )
            present.add(entity_id)
        if stale:
            self._db.execute(
                f"DELETE FROM entity_mentions WHERE id IN ({','.join('?' * len(stale))})",
                stale,
            )
        now = utcnow()
        for entity_id, surface in filed.items():
            if entity_id in present:
                if counts is not None:
                    counts["mentions_existing"] += 1
                continue
            self._db.execute(
                "INSERT INTO entity_mentions (id, entity_id, memory_id, surface, created_at) "
                "VALUES (?,?,?,?,?)",
                (new_id(), entity_id, memory_id, surface, now),
            )
            if counts is not None:
                counts["mentions_created"] += 1

    def _refile_locked(
        self, where: str, params: tuple[Any, ...], *, cache: dict[Any, Any],
        counts: dict[str, Any] | None = None,
    ) -> int:
        """File again the tags of every memory matching ``where`` (valid or
        not; ``_file_tags_locked``). Returns how many columns changed."""
        changed = 0
        for row in self._db.execute(
            "SELECT id, categories, user_id, agent_id, run_id FROM memories "
            f"WHERE {where} ORDER BY created_at, id",
            params,
        ).fetchall():
            stored = json.loads(row["categories"])
            filed = self._file_tags_locked(
                row["id"], stored, _row_scope(row), stored=stored, cache=cache, counts=counts
            )
            changed += filed != stored
        return changed

    def _refile_merged_tags(self) -> None:
        """File again, once, every column still naming a tag merged away.

        Before every writer filed its tags through ``_file_tags_locked``, such
        a column (restored from a backup, written straight to the backend, or
        missed by a merge) was indexed under the old name while its mention
        followed the merge, so ``categories()``, a filter on the survivor and
        ``delete_tag`` of it disagreed about the memory. Its column is
        rewritten to the survivor and its index and mentions follow. Each
        user's share is committed on its own and the marker set after the
        last, as the migration does (``_migrate_tags_to_topic_entities``)."""
        if self._db.execute(
            "SELECT 1 FROM meta WHERE key = ?", (_TAG_SURVIVORS_MARKER,)
        ).fetchone():
            return
        self._commit()  # the open's earlier steps, before the first user's commit
        users = [
            row["user_id"] for row in self._db.execute(
                "SELECT DISTINCT user_id FROM entities "
                "WHERE entity_type = ? AND merged_into IS NOT NULL",
                (TOPIC_TYPE,),
            ).fetchall()
        ]
        for user in users:
            with self._lock:
                try:
                    retired = {
                        row["normalized"] for row in self._db.execute(
                            "SELECT DISTINCT t.normalized FROM entities t "
                            "WHERE t.entity_type = ? AND t.user_id IS ? "
                            "AND t.merged_into IS NOT NULL AND NOT EXISTS ("
                            "SELECT 1 FROM entities a WHERE a.entity_type = t.entity_type "
                            "AND a.user_id IS t.user_id AND a.normalized = t.normalized "
                            "AND a.merged_into IS NULL)",
                            (TOPIC_TYPE, user),
                        ).fetchall()
                    }
                    ids = [
                        row["id"] for row in self._db.execute(
                            "SELECT id, categories FROM memories WHERE user_id IS ?", (user,)
                        ).fetchall()
                        if any(str(tag).strip().lower() in retired
                               for tag in json.loads(row["categories"]))
                    ]
                    cache: dict[Any, Any] = {}
                    for start in range(0, len(ids), 500):
                        chunk = ids[start:start + 500]
                        self._refile_locked(
                            f"id IN ({','.join('?' * len(chunk))})", tuple(chunk), cache=cache)
                except Exception:
                    self._db.rollback()
                    raise
                self._commit()
        self._db.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            (_TAG_SURVIVORS_MARKER, utcnow()),
        )
        self._commit()

    def _ensure_entity_names(self) -> bool:
        """Make the index of the entities' names (``_ENTITY_NAMES_SCHEMA``) and
        fill it once for a database from before it (``_ENTITY_NAMES_MARKER``).
        Whether there is one: an SQLite without the trigram tokenizer cannot
        write to it, so the triggers that would are dropped there, and the
        next open that can fills it again."""
        columns = {row["name"] for row in self._db.execute(
            "PRAGMA table_info(entity_names)").fetchall()}
        if sqlite3.sqlite_version_info < (3, 34, 0) or columns and "owner" not in columns:
            for trigger in _ENTITY_NAMES_TRIGGERS:
                self._db.execute(f"DROP TRIGGER IF EXISTS {trigger}")
            self._db.execute("DELETE FROM meta WHERE key = ?", (_ENTITY_NAMES_MARKER,))
            if sqlite3.sqlite_version_info < (3, 34, 0):
                return False
            # an index made before the names kept their owner is made again
            self._db.execute("DROP TABLE IF EXISTS entity_names_fts")
            self._db.execute("DROP TABLE entity_names")
        self._db.executescript(_ENTITY_NAMES_SCHEMA)
        if self._db.execute(
            "SELECT 1 FROM meta WHERE key = ?", (_ENTITY_NAMES_MARKER,)
        ).fetchone():
            return True
        # what an open without the tokenizer left behind goes first
        self._db.execute("DELETE FROM entity_names")
        self._db.execute("INSERT INTO entity_names_fts(entity_names_fts) VALUES ('delete-all')")
        self._db.execute(
            "INSERT INTO entity_names (entity_id, name, owner) SELECT entity_id, value, "
            f"{_owner('entity_id')} FROM ("
            "SELECT id AS entity_id, name AS value FROM entities UNION "
            "SELECT e.id, CAST(alias.value AS TEXT) FROM entities e, json_each("
            "CASE WHEN json_valid(e.metadata) THEN e.metadata END, '$.aliases') AS alias "
            "WHERE alias.value IS NOT NULL UNION "
            "SELECT entity_id, surface FROM entity_mentions"
            ") WHERE value != ''"
        )
        self._db.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            (_ENTITY_NAMES_MARKER, utcnow()),
        )
        return True

    def topic_entity(
        self, name: str, scope: Scope, *, create: bool = True, follow_merged: bool = False
    ) -> Entity | None:
        normalized = str(name).strip().lower()
        if not normalized:
            return None
        with self._lock:
            entity_id = self._active_topic_locked(normalized, scope.user_id)
            if entity_id is None and (follow_merged or create):
                merged = self._merged_topic_locked(normalized, scope.user_id)
                if merged is not None:
                    entity_id = merged[1]  # never a fresh topic of a name merged away
                elif create:
                    entity_id, _ = self._create_topic_locked(normalized, scope.user_id)
            if self._db.in_transaction:  # created, or an insert another process won
                self._commit()
        return self.get_entity(entity_id) if entity_id else None

    def tag_filing(self, names: Iterable[str], scope: Scope) -> dict[str, str]:
        with self._lock:
            filed: dict[str, str] = {}
            for raw in names:
                normalized = str(raw).strip().lower()
                if not normalized or normalized in filed:
                    continue
                renamed, _, _ = self._resolve_tag_locked(
                    normalized, scope.user_id, create=False)
                filed[normalized] = renamed or normalized
        return filed

    def topic_names(
        self, scope: Scope, *, prefixes: Iterable[str] | None = None
    ) -> dict[str, bool]:
        clause = "entity_type = ? AND user_id IS ?"
        params: list[Any] = [TOPIC_TYPE, scope.user_id]
        if prefixes is not None:
            wanted = sorted({str(prefix) for prefix in prefixes})
            if not wanted:
                return {}
            # a name's leading separators do not count (``obvious_topic_key``)
            clause += " AND (" + " OR ".join(
                ["ltrim(normalized, ' -_') LIKE ? ESCAPE '\\'"] * len(wanted)) + ")"
            params.extend(
                prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
                for prefix in wanted
            )
        with self._lock:
            rows = self._db.execute(
                "SELECT normalized, MIN(merged_into IS NOT NULL) AS merged FROM entities "
                f"WHERE {clause} GROUP BY normalized ORDER BY normalized",
                params,
            ).fetchall()
        return {row["normalized"]: not row["merged"] for row in rows}

    def topic_mention_counts(
        self, scope: Scope, *, exact_user: bool = False
    ) -> list[dict[str, Any]]:
        """Active memories per tag, from the topic entities' mentions: each
        tag's direct count (no rollup), largest first, then by name."""
        clause = _exact_scope_clause if exact_user else _scope_clause
        entity_clause, entity_params = clause(_user_scope(scope), prefix="e.")
        memory_clause, memory_params = clause(scope, prefix="m.")
        with self._lock:
            rows = self._db.execute(
                "SELECT e.normalized AS category, COUNT(DISTINCT m.id) AS count "
                "FROM entities e JOIN entity_mentions em ON em.entity_id = e.id "
                "JOIN memories m ON m.id = em.memory_id "
                "WHERE e.entity_type = ? AND e.merged_into IS NULL AND m.invalid_at IS NULL "
                f"AND {entity_clause} AND {memory_clause} "
                "GROUP BY e.normalized HAVING count > 0 ORDER BY count DESC, category",
                (TOPIC_TYPE, *entity_params, *memory_params),
            ).fetchall()
        return [{"category": row["category"], "count": row["count"]} for row in rows]

    def topic_mention_links(
        self, scope: Scope, *, exact_user: bool = False
    ) -> list[tuple[str, str]]:
        """``(tag, memory_id)`` for every active memory mentioning a tag."""
        clause = _exact_scope_clause if exact_user else _scope_clause
        entity_clause, entity_params = clause(_user_scope(scope), prefix="e.")
        memory_clause, memory_params = clause(scope, prefix="m.")
        with self._lock:
            rows = self._db.execute(
                "SELECT DISTINCT e.normalized AS category, m.id AS memory_id "
                "FROM entities e JOIN entity_mentions em ON em.entity_id = e.id "
                "JOIN memories m ON m.id = em.memory_id "
                "WHERE e.entity_type = ? AND e.merged_into IS NULL AND m.invalid_at IS NULL "
                f"AND {entity_clause} AND {memory_clause} "
                "ORDER BY e.normalized, m.id",
                (TOPIC_TYPE, *entity_params, *memory_params),
            ).fetchall()
        return [(row["category"], row["memory_id"]) for row in rows]

    def tags_to_topics(
        self, *, user_id: str | None = None, all_users: bool = True, dry_run: bool = False
    ) -> list[dict[str, Any]]:
        """Topic entities and mentions for the tags of the memories' columns,
        one user at a time, each memory filed as every writer files it
        (``_file_tags_locked``): an upgraded database's legacy tag index
        (``topics``, ``memory_topics``) is derived from the same columns, and
        is left as it is where it already agrees with them. Returns what was
        (or with ``dry_run``, would be) created per user, with the user's
        legacy ``topics`` rows counted. Idempotent: an entity or a mention
        that exists is counted as existing and left alone.
        ``all_users`` False migrates ``user_id`` alone (None: the memories
        without a user)."""
        with self._lock:
            if all_users:
                users = [
                    row["user_id"] for row in self._db.execute(
                        "SELECT user_id FROM (SELECT user_id FROM topics UNION "
                        "SELECT user_id FROM memories WHERE categories != '[]') "
                        "ORDER BY user_id IS NOT NULL, user_id"
                    ).fetchall()
                ]
            else:
                users = [user_id]
        report: list[dict[str, Any]] = []
        for user in users:  # one transaction per user: a rerun picks up after a failure
            with self._lock:
                try:
                    counts = self._tags_to_topics_locked(user)
                except Exception:
                    self._db.rollback()
                    raise
                if dry_run:
                    self._db.rollback()
                else:
                    self._commit()
            report.append({**counts, "dry_run": dry_run})
        return report

    def _tags_to_topics_locked(self, user_id: str | None) -> dict[str, Any]:
        counts = {
            "user_id": user_id, "topics": self._db.execute(
                "SELECT COUNT(*) FROM topics WHERE user_id IS ?", (user_id,)).fetchone()[0],
            "entities_created": 0, "entities_existing": 0,
            "mentions_created": 0, "mentions_existing": 0,
        }
        self._refile_locked(
            "user_id IS ? AND categories != '[]'", (user_id,), cache={}, counts=counts)
        return counts

    # -- meta ------------------------------------------------------------------
    def distinct_user_ids(self) -> list[str | None]:
        with self._lock:
            rows = self._db.execute(
                "SELECT DISTINCT user_id FROM memories"
            ).fetchall()
        return [r["user_id"] for r in rows]

    def adopt_unscoped(self, into: str, *, dry_run: bool = False) -> dict[str, Any]:
        """Move every row without a namespace (None, or "") into ``into``, in
        one transaction (``transaction``; a caller's joins it).

        A tag of the same name in ``into`` takes the unscoped tag's memories
        (folded as tags merge, ``_merge_entities_locked``): one active tag
        per namespace and name is an index, so it could not be moved beside
        it. Unscoped tags of one name, some None and some "", become one the
        same way. A named thing of ``into`` with the same name and type as
        an unscoped one, and the only one, takes it after the move, merged
        as a person's merge is (recorded, undone under Archive > Merged
        names); any other name both have is left to the identity passes. A
        row of the legacy tag index whose name ``into`` holds gives its links
        to that row and goes, as a tag merge does it (``retag_topics``).
        Every memory moved has its tags filed again. Nothing else is
        removed. ``dry_run`` counts and writes nothing; once nothing is
        unscoped, a run changes nothing."""
        if not into:
            raise ValueError("adopting the unscoped rows needs a namespace to adopt them into")
        unscoped = "(user_id IS NULL OR user_id = '')"
        with self._lock:
            tables = {table: self._db.execute(
                f"SELECT COUNT(*) FROM {table} WHERE {unscoped}").fetchone()[0]
                for table in (*_NAMESPACED_TABLES, "topics")}
            tags = {row["normalized"]: row["id"] for row in self._db.execute(
                "SELECT id, normalized FROM entities WHERE entity_type = ? "
                "AND merged_into IS NULL AND user_id = ?", (TOPIC_TYPE, into)).fetchall()}
            tag_folds: list[tuple[str, str, str]] = []
            for row in self._db.execute(
                    "SELECT id, name, normalized FROM entities WHERE entity_type = ? "
                    f"AND merged_into IS NULL AND {unscoped} ORDER BY created_at, id",
                    (TOPIC_TYPE,)).fetchall():
                if row["normalized"] in tags:
                    tag_folds.append((tags[row["normalized"]], row["id"], row["name"]))
                else:
                    tags[row["normalized"]] = row["id"]  # it moves in as the tag
            named: dict[tuple[str, str | None], list[str]] = {}
            for row in self._db.execute(
                    "SELECT id, normalized, entity_type FROM entities "
                    f"WHERE {_kind_clause('named')} AND merged_into IS NULL AND user_id = ?",
                    (into,)).fetchall():
                named.setdefault((row["normalized"], row["entity_type"]), []).append(row["id"])
            names_into = {normalized for normalized, _ in named}
            entity_folds: list[tuple[str, str, str, str]] = []
            left: list[str] = []
            for row in self._db.execute(
                    "SELECT id, name, normalized, entity_type FROM entities "
                    f"WHERE {_kind_clause('named')} AND merged_into IS NULL AND {unscoped} "
                    "ORDER BY created_at, id").fetchall():
                twins = named.get((row["normalized"], row["entity_type"]), [])
                if row["entity_type"] and len(twins) == 1:
                    entity_folds.append((twins[0], row["id"], row["name"], row["entity_type"]))
                elif row["normalized"] in names_into:
                    left.append(row["name"])
            legacy = self._db.execute(
                f"SELECT id, normalized, agent_id, run_id FROM topics WHERE {unscoped} "
                "ORDER BY created_at, id").fetchall()
            report: dict[str, Any] = {
                "into": into, "dry_run": dry_run, "tables": tables,
                "tags_folded": [name for _, _, name in tag_folds],
                "things_folded": [f"{name} ({kind})" for _, _, name, kind in entity_folds],
                "things_left_for_review": left,
                "kept_ids": sorted({keep for keep, _, _, _ in entity_folds}),
            }
            if dry_run:
                report["legacy_tags_folded"] = len(legacy) - len(
                    {(r["agent_id"], r["run_id"], r["normalized"]) for r in legacy} - {
                        (r["agent_id"], r["run_id"], r["normalized"]) for r in
                        self._db.execute("SELECT agent_id, run_id, normalized FROM topics "
                                         "WHERE user_id = ?", (into,)).fetchall()})
                return report
            with self.transaction():
                for keep, merge, _ in tag_folds:
                    self._merge_entities_locked(keep, merge)
                folded = 0
                for row in legacy:
                    twin = self._db.execute(
                        "SELECT id FROM topics WHERE user_id = ? AND agent_id IS ? "
                        "AND run_id IS ? AND normalized = ?",
                        (into, row["agent_id"], row["run_id"], row["normalized"])).fetchone()
                    if twin is None:
                        self._db.execute("UPDATE topics SET user_id = ? WHERE id = ?",
                                         (into, row["id"]))
                        continue
                    self._db.execute(
                        "INSERT OR IGNORE INTO memory_topics (memory_id, topic_id) "
                        "SELECT memory_id, ? FROM memory_topics WHERE topic_id = ?",
                        (twin["id"], row["id"]))
                    self._db.execute("DELETE FROM memory_topics WHERE topic_id = ?", (row["id"],))
                    self._db.execute("DELETE FROM topics WHERE id = ?", (row["id"],))
                    folded += 1
                report["legacy_tags_folded"] = folded
                moved = [row["id"] for row in self._db.execute(
                    f"SELECT id FROM memories WHERE {unscoped}").fetchall()]
                for table in _NAMESPACED_TABLES:
                    self._db.execute(f"UPDATE {table} SET user_id = ? WHERE {unscoped}", (into,))
                for keep, merge, _, _ in entity_folds:
                    self._merge_entities_locked(keep, merge)
                cache: dict[Any, Any] = {}
                for start in range(0, len(moved), 500):
                    chunk = moved[start:start + 500]
                    self._refile_locked(
                        f"id IN ({','.join('?' * len(chunk))})", tuple(chunk), cache=cache)
        return report

    def meta_items(self, prefix: str) -> dict[str, str]:
        """The meta keys starting with ``prefix``, with their values."""
        with self._lock:
            rows = self._db.execute(
                "SELECT key, value FROM meta WHERE substr(key, 1, ?) = ?",
                (len(prefix), prefix)).fetchall()
        return {row["key"]: row["value"] for row in rows}

    def get_meta(self, key: str) -> str | None:
        with self._lock:
            row = self._db.execute(
                "SELECT value FROM meta WHERE key = ?", (key,)
            ).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value)
            )
            self._commit()

    # -- typed relations ---------------------------------------------------
    @staticmethod
    def _row_to_relation(row: sqlite3.Row) -> Relation:
        return Relation(
            id=row["id"], subject=row["subject"], predicate=row["predicate"],
            object=row["object"], user_id=row["user_id"], memory_id=row["memory_id"],
            created_at=row["created_at"], valid_from=row["valid_from"],
            invalid_at=row["invalid_at"],
        )

    def add_relation(self, relation: Relation) -> Relation:
        with self._lock:
            # dedupe: one active edge per (subject, predicate, object, namespace)
            existing = self._db.execute(
                "SELECT id FROM relations WHERE subject=? AND predicate=? AND object=? "
                "AND IFNULL(user_id,'')=IFNULL(?,'') AND invalid_at IS NULL",
                (relation.subject, relation.predicate, relation.object, relation.user_id),
            ).fetchone()
            if existing:
                return relation
            self._db.execute(
                "INSERT INTO relations (id, subject, predicate, object, user_id, "
                "memory_id, created_at, valid_from, invalid_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (relation.id, relation.subject, relation.predicate, relation.object,
                 relation.user_id, relation.memory_id, relation.created_at,
                 relation.valid_from, relation.invalid_at),
            )
            self._commit()
        return relation

    def list_relations(self, scope: Scope, *, limit: int = 1000) -> list[Relation]:
        clause, params = _scope_clause(_user_scope(scope))
        with self._lock:
            rows = self._db.execute(
                f"SELECT * FROM relations WHERE {clause} AND invalid_at IS NULL "
                "ORDER BY created_at DESC, id LIMIT ?",
                (*params, limit),
            ).fetchall()
        return [self._row_to_relation(r) for r in rows]

    def relations_of(self, entity_ids: list[str]) -> list[Relation]:
        if not entity_ids:
            return []
        placeholders = ",".join("?" * len(entity_ids))
        with self._lock:
            rows = self._db.execute(
                f"SELECT * FROM relations WHERE invalid_at IS NULL AND "
                f"(subject IN ({placeholders}) OR object IN ({placeholders})) ORDER BY id",
                (*entity_ids, *entity_ids),
            ).fetchall()
        return [self._row_to_relation(r) for r in rows]

    def proposals_of(self, entity_ids: list[str]) -> list[MergeProposal]:
        if not entity_ids:
            return []
        placeholders = ",".join("?" * len(entity_ids))
        with self._lock:
            rows = self._db.execute(
                f"SELECT * FROM entity_proposals WHERE status != 'confirmed' AND "
                f"(entity_a IN ({placeholders}) OR entity_b IN ({placeholders})) ORDER BY id",
                (*entity_ids, *entity_ids),
            ).fetchall()
        return [self._row_to_proposal(r) for r in rows]

    # -- vectors -----------------------------------------------------------
    def memory_vectors(self, scope: Scope, *, limit: int = 5000):
        clause, params = _scope_clause(scope)
        with self._lock:
            rows = self._db.execute(
                f"SELECT id, embedding FROM memories WHERE {clause} "
                "AND embedding IS NOT NULL AND invalid_at IS NULL "
                "ORDER BY updated_at DESC, id LIMIT ?",
                (*params, limit),
            ).fetchall()
        return [
            (r["id"], np.frombuffer(r["embedding"], dtype=np.float32)) for r in rows
        ]

    def vectors_of(
        self, memory_ids: list[str], embedding_model: str | None = None
    ) -> dict[str, np.ndarray]:
        model_clause = " AND embedding_model = ?" if embedding_model else ""
        out: dict[str, np.ndarray] = {}
        for start in range(0, len(memory_ids), 500):
            chunk = memory_ids[start:start + 500]
            with self._lock:
                rows = self._db.execute(
                    f"SELECT id, embedding FROM memories WHERE id IN ({','.join('?' * len(chunk))}) "
                    f"AND embedding IS NOT NULL{model_clause}",
                    [*chunk, *([embedding_model] if embedding_model else [])],
                ).fetchall()
            out.update(
                (r["id"], np.frombuffer(r["embedding"], dtype=np.float32)) for r in rows
            )
        return out

    def unlabelled_vector_ids(self, scope: Scope) -> list[str]:
        clause, params = _scope_clause(scope)
        with self._lock:
            rows = self._db.execute(
                f"SELECT id FROM memories WHERE {clause} AND invalid_at IS NULL "
                "AND embedding IS NOT NULL AND embedding_model IS NULL",
                params,
            ).fetchall()
        return [row["id"] for row in rows]

    def consolidated_memories(self, scope: Scope) -> list[Memory]:
        clause, params = _scope_clause(scope)
        with self._lock:
            rows = self._db.execute(
                f"SELECT {_MEMORY_COLS} FROM memories WHERE {clause} AND invalid_at IS NULL "
                "AND metadata LIKE '%consolidated_from%'",
                params,
            ).fetchall()
        memories = [_row_to_memory(r) for r in rows]
        return [m for m in memories if (m.metadata or {}).get("consolidated_from")]

    def set_property_vectors(
        self, vectors: dict[str, list[float]], embedding_model: str,
        hashes: dict[str, str] | None = None,
    ) -> None:
        hashes = hashes or {}
        with self._lock:
            self._db.executemany(
                "INSERT OR REPLACE INTO memory_property_vectors "
                "(memory_id, embedding, embedding_model, masked_hash) VALUES (?,?,?,?)",
                [(mid, np.asarray(v, dtype=np.float32).tobytes(), embedding_model,
                  hashes.get(mid))
                 for mid, v in vectors.items()],
            )
            self._commit()

    def property_vectors_of(
        self, memory_ids: list[str], embedding_model: str | None = None
    ) -> dict[str, np.ndarray]:
        model_clause = " AND embedding_model = ?" if embedding_model else ""
        out: dict[str, np.ndarray] = {}
        for start in range(0, len(memory_ids), 500):
            chunk = memory_ids[start:start + 500]
            with self._lock:
                rows = self._db.execute(
                    "SELECT memory_id, embedding FROM memory_property_vectors "
                    f"WHERE memory_id IN ({','.join('?' * len(chunk))}){model_clause}",
                    [*chunk, *([embedding_model] if embedding_model else [])],
                ).fetchall()
            out.update(
                (r["memory_id"], np.frombuffer(r["embedding"], dtype=np.float32)) for r in rows
            )
        return out

    def property_vector_hashes(
        self, memory_ids: list[str]
    ) -> dict[str, tuple[str | None, str | None]]:
        out: dict[str, tuple[str | None, str | None]] = {}
        for start in range(0, len(memory_ids), 500):
            chunk = memory_ids[start:start + 500]
            with self._lock:
                rows = self._db.execute(
                    "SELECT memory_id, masked_hash, embedding_model FROM memory_property_vectors "
                    f"WHERE memory_id IN ({','.join('?' * len(chunk))})",
                    chunk,
                ).fetchall()
            out.update((r["memory_id"], (r["masked_hash"], r["embedding_model"])) for r in rows)
        return out

    def delete_property_vectors(self, memory_ids: list[str]) -> None:
        with self._lock:
            for start in range(0, len(memory_ids), 500):
                chunk = memory_ids[start:start + 500]
                self._db.execute(
                    "DELETE FROM memory_property_vectors "
                    f"WHERE memory_id IN ({','.join('?' * len(chunk))})",
                    chunk,
                )
            self._commit()

    # --- question keys (``intelligence.questions``) ---------------------------

    def set_questions(
        self, memory_id: str, questions: list[tuple[str, str]],
        vectors: list[list[float] | None] | None = None, embedding_model: str | None = None,
    ) -> None:
        """Replace a memory's question keys with ``questions`` ((text, source)
        each, in order) and their vectors where given (None for a question
        without one), stored in float16."""
        with self._lock:
            self._db.execute("DELETE FROM memory_questions WHERE memory_id = ?", (memory_id,))
            for n, (text, source) in enumerate(questions):
                vector = vectors[n] if vectors and n < len(vectors) else None
                self._db.execute(
                    "INSERT INTO memory_questions (memory_id, n, text, source, embedding, "
                    "embedding_model) VALUES (?, ?, ?, ?, ?, ?)",
                    (memory_id, n, text, source,
                     _pack_half(vector) if vector else None,
                     embedding_model if vector else None))
            self._commit()

    def add_question(
        self, memory_id: str, text: str, source: str, vector: list[float] | None = None,
        embedding_model: str | None = None, *, limit: int,
    ) -> bool:
        """Add one question key after a memory's others, unless the memory
        has it already (ignoring case) or has ``limit`` keys. Returns whether
        it was added."""
        with self._lock:
            rows = self._db.execute(
                "SELECT n, text FROM memory_questions WHERE memory_id = ?",
                (memory_id,)).fetchall()
            if len(rows) >= limit or any(
                    row["text"].casefold() == text.casefold() for row in rows):
                return False
            n = max((row["n"] for row in rows), default=-1) + 1
            self._db.execute(
                "INSERT INTO memory_questions (memory_id, n, text, source, embedding, "
                "embedding_model) VALUES (?, ?, ?, ?, ?, ?)",
                (memory_id, n, text, source, _pack_half(vector) if vector else None,
                 embedding_model if vector else None))
            self._commit()
        return True

    def set_question_vectors(
        self, vectors: dict[tuple[str, int], list[float]], embedding_model: str,
    ) -> None:
        """The vectors of these question rows ((memory_id, n) each)."""
        with self._lock:
            for (memory_id, n), vector in vectors.items():
                self._db.execute(
                    "UPDATE memory_questions SET embedding = ?, embedding_model = ? "
                    "WHERE memory_id = ? AND n = ?",
                    (_pack_half(vector), embedding_model, memory_id, n))
            self._commit()

    def questions_of(self, memory_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
        """Each memory's question keys, in order: ``n``, ``text``, ``source``
        and ``embedding_model`` (None for a question without a vector)."""
        out: dict[str, list[dict[str, Any]]] = {}
        with self._lock:
            for start in range(0, len(memory_ids), 500):
                chunk = memory_ids[start:start + 500]
                rows = self._db.execute(
                    "SELECT memory_id, n, text, source, embedding_model FROM memory_questions "
                    f"WHERE memory_id IN ({','.join('?' * len(chunk))}) ORDER BY memory_id, n",
                    chunk).fetchall()
                for row in rows:
                    out.setdefault(row["memory_id"], []).append(
                        {"n": row["n"], "text": row["text"], "source": row["source"],
                         "embedding_model": row["embedding_model"]})
        return out

    def questions_without_vectors(
        self, scope: Scope, embedding_model: str, limit: int = 100_000,
    ) -> list[tuple[str, int, str]]:
        """The question keys of valid memories in ``scope`` that have no
        vector from ``embedding_model`` ((memory_id, n, text) each)."""
        clause, params = _scope_clause(scope, prefix="m.")
        with self._lock:
            rows = self._db.execute(
                "SELECT q.memory_id, q.n, q.text FROM memory_questions q "
                f"JOIN memories m ON m.id = q.memory_id WHERE {clause} "
                "AND m.invalid_at IS NULL AND (q.embedding IS NULL OR q.embedding_model IS NOT ?) "
                "ORDER BY q.memory_id, q.n LIMIT ?",
                (*params, embedding_model, limit)).fetchall()
        return [(row["memory_id"], row["n"], row["text"]) for row in rows]

    def memories_without_questions(self, scope: Scope, limit: int = 100_000) -> list[Memory]:
        """Valid memories in ``scope`` with no question key at all and not
        marked as asked about (``questions_checked``), oldest first (a
        backfill's work)."""
        clause, params = _scope_clause(scope, prefix="m.")
        with self._lock:
            rows = self._db.execute(
                f"SELECT {_prefixed(_MEMORY_COLS, 'm')} FROM memories m WHERE {clause} "
                "AND m.invalid_at IS NULL AND NOT EXISTS (SELECT 1 FROM memory_questions q "
                "WHERE q.memory_id = m.id) "
                "AND json_extract(m.metadata, '$.questions_checked') IS NULL "
                "ORDER BY m.created_at, m.id LIMIT ?",
                (*params, limit)).fetchall()
        return [_row_to_memory(row) for row in rows]

    def question_keyword_search(
        self,
        query: str,
        scope: Scope,
        limit: int = 20,
        include_invalid: bool = False,
        categories: list[str] | None = None,
        entity_id: str | None = None,
        history: bool = False,
        among: Any = None,
    ) -> list[tuple[Memory, float]]:
        """BM25 over the question keys: a memory scores as its best question
        does. The same scope and filters as ``keyword_search``, in SQL before
        bm25() reads a row, so another account's questions are never scored."""
        tokens = _WORD_RE.findall(query)
        if not tokens:
            return []
        match = " OR ".join(f'"{t}"' for t in dict.fromkeys(t.lower() for t in tokens[:32]))
        clause, params = _search_scope_clause(scope, "m")
        cat_clause, cat_params = _category_clause(categories, "m.id")
        entity_clause, entity_params = _entity_clause(entity_id, "m.id")
        among_clause, among_params = _among_clause(among, "m.id")
        if not include_invalid:
            clause += (f" AND (m.invalid_at IS NULL OR ({_history_clause('m')}))" if history
                       else " AND m.invalid_at IS NULL")
        with self._lock:
            # bm25() is read per question row (SQLite cannot aggregate it);
            # a memory has at most QUESTIONS_LIMIT rows, so the best
            # ``limit`` memories are among the best ``limit * 9`` rows
            rows = self._db.execute(
                "SELECT q.memory_id AS id, bm25(memory_questions_fts) AS rank_score "
                "FROM memory_questions_fts CROSS JOIN memory_questions q "
                "ON q.rowid = memory_questions_fts.rowid CROSS JOIN memories m "
                f"ON m.id = q.memory_id WHERE memory_questions_fts MATCH ? AND {clause} "
                f"AND {cat_clause} AND {entity_clause} AND {among_clause} "
                "ORDER BY rank_score, q.memory_id LIMIT ?",
                (match, *params, *cat_params, *entity_params, *among_params, limit * 9),
            ).fetchall()
            best: dict[str, float] = {}
            for row in rows:
                score = -float(row["rank_score"])  # bm25() is lower-is-better
                if score > best.get(row["id"], -math.inf):
                    best[row["id"]] = score
            scores = dict(sorted(best.items(), key=lambda item: (-item[1], item[0]))[:limit])
            found = self._memories_by_id(list(scores))
        return [(found[mid], scores[mid]) for mid in scores if mid in found]

    def question_vector_search(
        self,
        embedding: list[float],
        embedding_model: str,
        scope: Scope,
        limit: int = 20,
        include_invalid: bool = False,
        categories: list[str] | None = None,
        entity_id: str | None = None,
        history: bool = False,
        among: Any = None,
    ) -> list[tuple[Memory, float]]:
        """Cosine over the question keys' vectors, read exactly (there are a
        few per memory and no index): a memory scores as its best question
        does. ``embedding`` is compared cut to the stored length. The same
        scope and filters as ``vector_search``."""
        clause, params = _search_scope_clause(scope, "m")
        cat_clause, cat_params = _category_clause(categories, "m.id")
        entity_clause, entity_params = _entity_clause(entity_id, "m.id")
        among_clause, among_params = _among_clause(among, "m.id")
        if not include_invalid:
            clause += (f" AND (m.invalid_at IS NULL OR ({_history_clause('m')}))" if history
                       else " AND m.invalid_at IS NULL")
        with self._lock:
            rows = self._db.execute(
                "SELECT q.memory_id, q.embedding FROM memory_questions q "
                f"JOIN memories m ON m.id = q.memory_id WHERE {clause} AND {cat_clause} "
                f"AND {entity_clause} AND {among_clause} AND q.embedding IS NOT NULL "
                "AND q.embedding_model = ?",
                (*params, *cat_params, *entity_params, *among_params, embedding_model),
            ).fetchall()
            if not rows or limit <= 0:
                return []
            mats = np.stack([_unpack_half(r["embedding"]) for r in rows])
            query = np.asarray(embedding, dtype=np.float32)[:mats.shape[1]]
            qnorm = float(np.linalg.norm(query))
            if qnorm == 0 or mats.shape[1] != query.shape[0]:
                return []
            norms = np.linalg.norm(mats, axis=1)
            norms[norms == 0] = 1e-9
            sims = (mats @ query) / (norms * qnorm)
            best: dict[str, float] = {}
            for row, sim in zip(rows, sims):
                mid = row["memory_id"]
                if sim > best.get(mid, -2.0):
                    best[mid] = float(sim)
            order = sorted(best, key=lambda mid: (-best[mid], mid))[:limit]
            found = self._memories_by_id(order)
        return [(found[mid], best[mid]) for mid in order if mid in found]

    # --- the search log (``retrieval.search_log``) -----------------------------

    def log_search(self, row: dict[str, Any]) -> int:
        """Keep one search (the columns of ``search_log`` but ``id`` and
        ``keyed``); returns its id."""
        cols = [c for c in _SEARCH_LOG_COLS if c in row]
        with self._lock:
            cur = self._db.execute(
                f"INSERT INTO search_log ({', '.join(cols)}) "
                f"VALUES ({', '.join('?' * len(cols))})",
                [row[c] for c in cols])
            self._commit()
        return int(cur.lastrowid)

    def search_log_rows(
        self, *, user_id: str | None = None, since: str | None = None,
        until: str | None = None, run_id: str | None = None,
        owner_prefix: str | None = None, newest_first: bool = False,
        limit: int = 1_000_000,
    ) -> list[dict[str, Any]]:
        """The kept searches, oldest first (``newest_first`` the other way):
        of one namespace (``user_id``), one run, an account's namespaces
        (``owner_prefix``, as ``_backup_owner_matches`` reads one), and
        ``since`` <= at <= ``until`` (ISO 8601 times)."""
        clauses, params = ["1=1"], []
        if user_id is not None:
            clauses.append("user_id = ?")
            params.append(user_id)
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        if since is not None:
            clauses.append("at >= ?")
            params.append(since)
        if until is not None:
            clauses.append("at <= ?")
            params.append(until)
        if owner_prefix is not None:
            if owner_prefix.endswith("::"):
                clauses.append("substr(user_id, 1, ?) = ?")
                params += [len(owner_prefix), owner_prefix]
            else:
                clauses.append("user_id = ?")
                params.append(owner_prefix)
        order = "DESC" if newest_first else "ASC"
        with self._lock:
            rows = self._db.execute(
                f"SELECT * FROM search_log WHERE {' AND '.join(clauses)} "
                f"ORDER BY at {order}, id {order} LIMIT ?", (*params, limit)).fetchall()
        return [dict(row) for row in rows]

    def mark_search_keyed(self, log_ids: dict[int, int]) -> None:
        """Count, per kept search, the memories that took its query as a key."""
        with self._lock:
            for log_id, count in log_ids.items():
                self._db.execute("UPDATE search_log SET keyed = keyed + ? WHERE id = ?",
                                 (count, log_id))
            self._commit()

    def prune_search_log(
        self, before: str, *, user_id: str | None = None, exact_user: bool = False,
    ) -> int:
        """Delete the kept searches older than ``before``: of one namespace
        (``user_id``; with ``exact_user`` None is the searches without one),
        or of every namespace. Returns how many."""
        clause, params = "at < ?", [before]
        if user_id is not None:
            clause += " AND user_id = ?"
            params.append(user_id)
        elif exact_user:
            clause += " AND user_id IS NULL"
        with self._lock:
            cur = self._db.execute(f"DELETE FROM search_log WHERE {clause}", params)
            self._commit()
        return cur.rowcount

    def delete_search_log(self, *, user_id: str | None) -> int:
        """Delete every kept search of a namespace (None: those without one)."""
        with self._lock:
            cur = self._db.execute(
                "DELETE FROM search_log WHERE user_id IS ?", (user_id,))
            self._commit()
        return cur.rowcount

    def restore_search_log(
        self, rows: list[dict[str, Any]], *, owner_prefix: str | None = None,
        check_only: bool = False,
    ) -> int:
        """Add kept searches from an export that asked for them
        (``export_backup(..., search_log=True)``), each under a new id; a row
        outside the account (``owner_prefix``) or with other columns is
        refused before anything is written. ``check_only``: refuse or pass,
        and write nothing."""
        allowed = set(_SEARCH_LOG_COLS) | {"id", "keyed"}
        if not isinstance(rows, list):
            raise ValueError("backup search_log must be a list")
        for row in rows:
            if not isinstance(row, dict) or not set(row) <= allowed \
                    or not {"at", "query", "mode"} <= set(row):
                raise ValueError("search log row has the wrong columns")
            if not self._backup_owner_matches(row.get("user_id"), owner_prefix):
                raise ValueError("backup contains search_log outside this account")
        if check_only:
            return 0
        added = 0
        with self._lock:
            for row in rows:
                # a search restored twice is kept once
                if self._db.execute(
                        "SELECT 1 FROM search_log WHERE at = ? AND user_id IS ? "
                        "AND run_id IS ? AND query = ?",
                        (row["at"], row.get("user_id"), row.get("run_id"),
                         row["query"])).fetchone():
                    continue
                cols = [c for c in row if c != "id"]
                self._db.execute(
                    f"INSERT INTO search_log ({', '.join(cols)}) "
                    f"VALUES ({', '.join('?' * len(cols))})", [row[c] for c in cols])
                added += 1
            self._commit()
        return added

    def _memories_by_id(self, memory_ids: list[str]) -> dict[str, Memory]:
        """The memories of these ids that exist. Caller holds the lock."""
        found: dict[str, Memory] = {}
        for start in range(0, len(memory_ids), 500):
            chunk = memory_ids[start:start + 500]
            if not chunk:
                continue
            rows = self._db.execute(
                f"SELECT {_MEMORY_COLS} FROM memories WHERE id IN "
                f"({','.join('?' * len(chunk))})", chunk).fetchall()
            found.update((row["id"], _row_to_memory(row)) for row in rows)
        return found

    def session_memories(
        self, memory: Memory, *, hours: float = 3.0, limit: int = 50
    ) -> list[Memory]:
        context = (memory.metadata or {}).get("context")
        if memory.run_id:
            same, params = "run_id = ?", [memory.run_id]
        elif memory.agent_id and context:
            same, params = "agent_id = ? AND json_extract(metadata, '$.context') = ?", [
                memory.agent_id, context]
        else:
            return []
        try:
            at = datetime.fromisoformat(memory.created_at)
        except (TypeError, ValueError):
            return []
        window = timedelta(hours=hours)
        owner = "user_id IS NULL" if memory.user_id is None else "user_id = ?"
        with self._lock:
            rows = self._db.execute(
                f"SELECT {_MEMORY_COLS} FROM memories WHERE {owner} AND id != ? "
                f"AND invalid_at IS NULL AND {same} AND created_at BETWEEN ? AND ? "
                "ORDER BY created_at DESC, id LIMIT ?",
                (*([] if memory.user_id is None else [memory.user_id]), memory.id, *params,
                 (at - window).isoformat(timespec="seconds"),
                 (at + window).isoformat(timespec="seconds"), limit),
            ).fetchall()
        return [_row_to_memory(r) for r in rows]

    @staticmethod
    def _row_to_entity(row: sqlite3.Row) -> Entity:
        return Entity(
            id=row["id"],
            name=row["name"],
            normalized=row["normalized"],
            entity_type=row["entity_type"],
            user_id=row["user_id"],
            agent_id=row["agent_id"],
            run_id=row["run_id"],
            description=row["description"],
            description_updated_at=row["description_updated_at"],
            metadata=json.loads(row["metadata"]),
            merged_into=row["merged_into"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _row_to_proposal(row: sqlite3.Row) -> MergeProposal:
        return MergeProposal(
            id=row["id"],
            entity_a=row["entity_a"],
            entity_b=row["entity_b"],
            user_id=row["user_id"],
            status=row["status"],
            confidence=row["confidence"],
            reason=row["reason"],
            created_at=row["created_at"],
            decided_at=row["decided_at"],
            compared_step=row["compared_step"],
            different=row["different"] if "different" in row.keys() else None,
            belongs=(json.loads(row["belongs"])
                     if "belongs" in row.keys() and row["belongs"] else None),
        )

    def insert_entity(self, entity: Entity) -> Entity:
        if not entity.normalized:
            entity.normalized = entity.name.strip().lower()
        with self._lock:
            self._db.execute(
                "INSERT INTO entities (id, name, normalized, entity_type, user_id, agent_id, "
                "run_id, description, description_updated_at, metadata, merged_into, "
                "created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    entity.id, entity.name, entity.normalized, entity.entity_type,
                    entity.user_id, entity.agent_id, entity.run_id,
                    entity.description, entity.description_updated_at,
                    json.dumps(entity.metadata), entity.merged_into,
                    entity.created_at, entity.updated_at,
                ),
            )
            aliases = entity.metadata.get("aliases", [])
            if aliases:
                self._has_metadata_aliases = True
            self._commit()
        return entity

    def get_entity(self, entity_id: str) -> Entity | None:
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM entities WHERE id = ?", (entity_id,)
            ).fetchone()
        return self._row_to_entity(row) if row else None

    def find_entities(self, normalized: str, scope: Scope) -> list[Entity]:
        clause, params = _scope_clause(scope)
        with self._lock:
            rows = self._db.execute(
                f"SELECT * FROM entities WHERE normalized = ? AND merged_into IS NULL "
                f"AND {_kind_clause('named')} AND {clause} ORDER BY updated_at DESC, id",
                (normalized.strip().lower(), *params),
            ).fetchall()
        return [self._row_to_entity(r) for r in rows]

    def _entity_candidates(
        self, normalized: list[str], scope: Scope, *, limit: int
    ) -> list[Entity]:
        names = sorted({value.strip().lower() for value in normalized if value.strip()})
        if not names:
            return []
        placeholders = ",".join("?" * len(names))
        scope_clause, scope_params = _scope_clause(scope, prefix="e.")
        indexed_sql = (
            "WITH matched(id) AS ("
            f"SELECT id FROM entities WHERE normalized IN ({placeholders}) "
            "UNION "
            "SELECT entity_id FROM entity_mentions "
            f"WHERE lower(trim(surface)) IN ({placeholders}) "
            "UNION "
            "SELECT merged_into FROM entities WHERE merged_into IS NOT NULL "
            f"AND normalized IN ({placeholders})"
            ") SELECT e.* FROM matched JOIN entities e ON e.id = matched.id "
            "WHERE e.merged_into IS NULL "
            # a name finds named things: a tag is never what a save or a
            # question names (``topic_entity`` finds tags)
            f"AND {_kind_clause('named', 'e.')} "
            f"AND {scope_clause} ORDER BY e.updated_at DESC, e.id LIMIT ?"
        )
        with self._lock:
            rows = self._db.execute(
                indexed_sql,
                (*names, *names, *names, *scope_params, limit),
            ).fetchall()
            # User aliases intentionally remain metadata until measurements earn
            # a separate table. Pay the JSON scan only when indexed identity
            # evidence found nothing and this database actually has such aliases:
            # of the entities that hold such a name, where the index of names
            # tells which (an alias is one of them), else of every entity.
            if not rows and self._has_metadata_aliases:
                owner = ([] if not self._name_index or scope.user_id is None
                         else [_owned(scope.user_id)])
                named = (f"e.id IN (SELECT entity_id FROM entity_names WHERE "
                         f"lower(trim(name)) IN ({placeholders})"
                         f"{' AND owner = ?' if owner else ''}) AND "
                         if self._name_index else "")
                rows = self._db.execute(
                    f"SELECT e.* FROM entities e WHERE {named}"
                    f"e.merged_into IS NULL AND {_kind_clause('named', 'e.')} "
                    f"AND {scope_clause} AND EXISTS ("
                    "SELECT 1 FROM json_each(e.metadata, '$.aliases') alias "
                    f"WHERE lower(trim(CAST(alias.value AS TEXT))) IN ({placeholders})"
                    ") ORDER BY e.updated_at DESC, e.id LIMIT ?",
                    (*(names if self._name_index else ()), *owner, *scope_params, *names,
                     limit),
                ).fetchall()
        return [self._row_to_entity(row) for row in rows]

    def find_entity_candidates(
        self, normalized: str, scope: Scope, *, limit: int = 20
    ) -> list[Entity]:
        return self._entity_candidates([normalized], scope, limit=limit)

    def find_entities_by_aliases(
        self, normalized: list[str], scope: Scope, *, limit: int = 50
    ) -> list[Entity]:
        return self._entity_candidates(normalized, scope, limit=limit)

    def entity_names_holding(self, words: list[str], scope: Scope) -> list[tuple[str, str]]:
        words = sorted({word.strip().lower() for word in words if word.strip()})
        if not words:
            return []
        scope_clause, scope_params = _scope_clause(scope, prefix="e.")
        live = f"e.merged_into IS NULL AND {_kind_clause('named', 'e.')} AND {scope_clause}"

        def holds(value: str) -> str:
            return "(" + " OR ".join(f"instr(lower({value}), ?) > 0" for _ in words) + ")"

        # The names that may hold a word, each with the entity that gives it:
        # found by the index of names (a word of three letters or more by its
        # letter trigrams, a shorter one by reading every name in it), of the
        # search's user alone where it has one, or without the index read
        # where they are kept. What follows keeps those a
        # live entity answers to, as ``entity_aliases`` gives them: its own,
        # the wordings its mentions use, its aliases, the names of entities
        # merged into it.
        found = [word for word in words if len(word) >= 3]
        short = [word for word in words if len(word) < 3]
        if self._name_index:
            # of the user's names alone, where the search has a user
            owner = None if scope.user_id is None else _owned(scope.user_id)
            match = "name : (" + " OR ".join(
                '"' + word.replace('"', '""') + '"' for word in found) + ")"
            if owner is not None:
                match += ' AND owner : "' + owner.replace('"', '""') + '"'
            parts = ([] if not found else [
                "SELECT n.entity_id AS entity_id, n.name AS name FROM entity_names_fts "
                "JOIN entity_names n ON n.id = entity_names_fts.rowid "
                "WHERE entity_names_fts MATCH ?"])
            parts += [] if not short else [
                "SELECT entity_id, name FROM entity_names WHERE "
                + ("owner = ? AND " if owner is not None else "") + "("
                + " OR ".join("instr(lower(name), ?) > 0" for _ in short) + ")"]
            params = (([match] if found else [])
                      + ([owner] if short and owner is not None else []) + short)
        else:
            parts = ["SELECT id AS entity_id, name FROM entities WHERE " + holds("name"),
                     "SELECT entity_id, surface FROM entity_mentions WHERE " + holds("surface")]
            params = [*words, *words]
            if self._has_metadata_aliases:
                parts.append("SELECT e.id, CAST(alias.value AS TEXT) FROM entities e, "
                             "json_each(e.metadata, '$.aliases') alias WHERE "
                             + holds("CAST(alias.value AS TEXT)"))
                params += words
        alias = (" OR held.name IN (SELECT CAST(alias.value AS TEXT) "
                 "FROM json_each(e.metadata, '$.aliases') alias)"
                 if self._has_metadata_aliases else "")
        # a name held by an entity merged away is read for the one it was
        # merged into (its own name only, as ``entity_aliases`` reads it)
        sql = (
            f"SELECT DISTINCT e.id AS id, held.name AS name FROM ({' UNION '.join(parts)}) "
            "AS held JOIN entities h ON h.id = held.entity_id "
            "JOIN entities e ON e.id = IFNULL(h.merged_into, h.id) "
            f"WHERE {live} AND {holds('held.name')} AND (h.merged_into IS NOT NULL "
            "AND held.name = h.name OR h.merged_into IS NULL AND (held.name = e.name "
            "OR EXISTS (SELECT 1 FROM entity_mentions em WHERE lower(trim(em.surface)) = "
            "lower(trim(held.name)) AND em.entity_id = e.id AND em.surface = held.name)"
            f"{alias})) ORDER BY 1, 2"
        )
        with self._lock:
            rows = self._db.execute(sql, [*params, *scope_params, *words]).fetchall()
        return [(row["id"], row["name"]) for row in rows if row["name"]]

    def word_use(self, word: str, entity_id: str, scope: Scope) -> tuple[int, int]:
        word = word.strip()
        if not word:
            return 0, 0
        person = _user_scope(scope)
        memory_clause, memory_params = _scope_clause(person, prefix="m.")
        turn_clause, turn_params = _scope_clause(person, prefix="e.")
        phrase = f'"{word}"'
        with self._lock:
            memories = self._db.execute(
                "SELECT count(*), IFNULL(sum(m.id IN (SELECT memory_id FROM entity_mentions "
                "WHERE entity_id = ?)), 0) FROM memories_fts CROSS JOIN memories m "
                "ON m.rowid = memories_fts.rowid WHERE memories_fts MATCH ? "
                f"AND m.invalid_at IS NULL AND {memory_clause}",
                (entity_id, phrase, *memory_params),
            ).fetchone()
            turns = self._db.execute(
                "SELECT count(*), IFNULL(sum(e.id IN (SELECT source.value FROM entity_mentions em "
                "JOIN memories m ON m.id = em.memory_id, json_each(m.source_episode_ids) AS source "
                "WHERE em.entity_id = ? AND m.invalid_at IS NULL)), 0) FROM episodes_fts "
                "CROSS JOIN episodes e ON e.rowid = episodes_fts.rowid WHERE episodes_fts MATCH ? "
                f"AND e.withheld_at IS NULL AND {turn_clause}",
                (entity_id, phrase, *turn_params),
            ).fetchone()
        return int(memories[0]) + int(turns[0]), int(memories[1]) + int(turns[1])

    def entity_aliases(self, entity_id: str) -> list[str]:
        entity = self.get_entity(entity_id)
        if entity is None:
            return []
        with self._lock:
            surfaces = self._db.execute(
                "SELECT DISTINCT surface FROM entity_mentions WHERE entity_id = ?",
                (entity_id,),
            ).fetchall()
            merged_names = self._db.execute(
                "SELECT name FROM entities WHERE merged_into = ?", (entity_id,)
            ).fetchall()
        values: list[str] = [entity.name]
        values.extend(row["surface"] for row in surfaces)
        values.extend(row["name"] for row in merged_names)
        raw_aliases = entity.metadata.get("aliases", [])
        if isinstance(raw_aliases, str):
            raw_aliases = [raw_aliases]
        if isinstance(raw_aliases, list):
            values.extend(str(alias) for alias in raw_aliases)
        aliases: list[str] = []
        seen: set[str] = set()
        for value in values:
            display = value.strip()
            key = display.lower()
            if display and key not in seen:
                seen.add(key)
                aliases.append(display)
        return aliases

    def add_entity_alias(self, entity_id: str, alias: str) -> Entity | None:
        display = alias.strip()
        if not display:
            return self.get_entity(entity_id)
        with self._lock:
            row = self._db.execute(
                "SELECT name, metadata FROM entities WHERE id = ?", (entity_id,)
            ).fetchone()
            if row is None:
                return None
            metadata = json.loads(row["metadata"])
            raw_aliases = metadata.get("aliases", [])
            if isinstance(raw_aliases, str):
                raw_aliases = [raw_aliases]
            aliases = [str(value).strip() for value in raw_aliases if str(value).strip()]
            known = {row["name"].strip().lower(), *(value.lower() for value in aliases)}
            added = display.lower() not in known
            if added:
                aliases.append(display)
                metadata["aliases"] = aliases
                self._db.execute(
                    "UPDATE entities SET metadata = ?, updated_at = ?, "
                    "description_updated_at = NULL WHERE id = ?",
                    (json.dumps(metadata), utcnow(), entity_id),
                )
                self._has_metadata_aliases = True
                self._commit()
        if added and self.names_changed is not None:
            self.names_changed([entity_id])
        return self.get_entity(entity_id)

    def rename_entity(self, entity_id: str, name: str) -> Entity | None:
        display = name.strip()
        if not display:
            return None
        with self._lock:
            row = self._db.execute(
                "SELECT name, metadata, merged_into FROM entities WHERE id = ?",
                (entity_id,),
            ).fetchone()
            if row is None or row["merged_into"] is not None:
                return None
            old_name = row["name"].strip()
            metadata = json.loads(row["metadata"])
            raw_aliases = metadata.get("aliases", [])
            if isinstance(raw_aliases, str):
                raw_aliases = [raw_aliases]
            aliases = [
                str(value).strip()
                for value in raw_aliases
                if str(value).strip()
                and str(value).strip().lower() != display.lower()
            ]
            known = {value.lower() for value in aliases}
            if old_name.lower() != display.lower() and old_name.lower() not in known:
                aliases.append(old_name)
            if aliases:
                metadata["aliases"] = aliases
                self._has_metadata_aliases = True
            else:
                metadata.pop("aliases", None)
            self._db.execute(
                "UPDATE entities SET name = ?, normalized = ?, metadata = ?, "
                "updated_at = ?, description_updated_at = NULL "
                "WHERE id = ? AND merged_into IS NULL",
                (
                    display,
                    display.lower(),
                    json.dumps(metadata),
                    utcnow(),
                    entity_id,
                ),
            )
            self._commit()
        if display != old_name and self.names_changed is not None:
            self.names_changed([entity_id])
        return self.get_entity(entity_id)

    def set_entity_description(
        self, entity_id: str, description: str, generated_at: str
    ) -> Entity | None:
        with self._lock:
            cur = self._db.execute(
                "UPDATE entities SET description = ?, description_updated_at = ? "
                "WHERE id = ? AND merged_into IS NULL",
                (description, generated_at, entity_id),
            )
            self._commit()
        return self.get_entity(entity_id) if cur.rowcount else None

    def entity_evidence_updated_at(self, entity_id: str) -> str | None:
        with self._lock:
            row = self._db.execute(
                "SELECT updated_at FROM entities WHERE id = ? AND merged_into IS NULL",
                (entity_id,),
            ).fetchone()
        return row["updated_at"] if row else None

    def list_entities(
        self, scope: Scope, *, include_merged: bool = False, limit: int = 100,
        kind: str = "named",
    ) -> list[Entity]:
        clause, params = _scope_clause(scope)
        clause += f" AND {_kind_clause(kind)}"
        if not include_merged:
            clause += " AND merged_into IS NULL"
        with self._lock:
            rows = self._db.execute(
                f"SELECT * FROM entities WHERE {clause} ORDER BY updated_at DESC, id LIMIT ?",
                (*params, limit),
            ).fetchall()
        return [self._row_to_entity(r) for r in rows]

    def add_mention(self, mention: EntityMention) -> None:
        """Attach a memory to an entity. A wording the entity was not called
        before is one of its names from now on: ``names_changed`` hears of it.
        A mention that gives the name a type counts towards the entity's
        (``_settle_types_locked``)."""
        known = {alias.lower() for alias in self.entity_aliases(mention.entity_id)}
        with self._lock:
            self._insert_mentions_locked([mention])
            self._db.execute(
                "UPDATE entities SET updated_at = ?, description_updated_at = NULL "
                "WHERE id = ?",
                (mention.created_at, mention.entity_id),
            )
            if mention.entity_type:
                row = self._db.execute(
                    "SELECT entity_type FROM entities WHERE id = ?", (mention.entity_id,)
                ).fetchone()
                if row is not None and row["entity_type"] != mention.entity_type:
                    self._settle_types_locked([mention.entity_id])
            self._commit()
        wording = mention.surface.strip().lower()
        if wording and wording not in known and self.names_changed is not None:
            self.names_changed([mention.entity_id])

    def _insert_mentions_locked(self, mentions: Iterable[EntityMention]) -> None:
        self._db.executemany(
            "INSERT INTO entity_mentions (id, entity_id, memory_id, surface, created_at, "
            "decided, entity_type) VALUES (?,?,?,?,?,?,?)",
            [(m.id, m.entity_id, m.memory_id, m.surface, m.created_at,
              json.dumps(m.decided) if m.decided is not None else None, m.entity_type)
             for m in mentions],
        )

    def _type_votes_locked(self, entity_id: str, own: str | None) -> Counter[str]:
        """How many of the entity's mentions give each type. A mention that
        gives none (a tag folded into the thing, one saved before mentions
        kept a type) counts for ``own``, the type the entity has."""
        votes: Counter[str] = Counter()
        for count in self._db.execute(
            "SELECT entity_type, COUNT(*) AS n FROM entity_mentions "
            "WHERE entity_id = ? GROUP BY entity_type",
            (entity_id,),
        ).fetchall():
            kind = count["entity_type"] or own
            if kind and kind != TOPIC_TYPE:
                votes[kind] += count["n"]
        return votes

    @staticmethod
    def _most_given(votes: Counter[str], ties: list[str | None]) -> str | None:
        """The type most ``votes`` give; on a tie, the first of ``ties``
        among the leaders, or ``ties[0]`` when none of them leads."""
        if not votes:
            return ties[0]
        most = max(votes.values())
        leaders = [kind for kind, n in votes.items() if n == most]
        if len(leaders) == 1:
            return leaders[0]
        return next((kind for kind in ties if kind in leaders), ties[0])

    def _settle_types_locked(self, entity_ids: Iterable[str]) -> None:
        """Give each of these named entities the type most of its mentions
        give (``_type_votes_locked``), keeping its own on a tie. The type
        extraction gives a name comes from one sentence ("the shop" is a
        project in most, a product in its listing's), so it is evidence of
        the thing's type, not of another thing. Nothing is committed here."""
        for entity_id in sorted(set(entity_ids)):
            row = self._db.execute(
                "SELECT entity_type, metadata FROM entities "
                "WHERE id = ? AND merged_into IS NULL",
                (entity_id,),
            ).fetchone()
            if row is None or row["entity_type"] == TOPIC_TYPE or _owner_typed(row):
                continue
            own = row["entity_type"]
            kind = self._most_given(self._type_votes_locked(entity_id, own), [own])
            if kind != own:
                self._db.execute(
                    "UPDATE entities SET entity_type = ? WHERE id = ?", (kind, entity_id))

    def entity_mentions(self, entity_id: str) -> list[EntityMention]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM entity_mentions WHERE entity_id = ? ORDER BY created_at",
                (entity_id,),
            ).fetchall()
        return [
            EntityMention(
                id=r["id"], entity_id=r["entity_id"], memory_id=r["memory_id"],
                surface=r["surface"], created_at=r["created_at"],
                decided=json.loads(r["decided"]) if r["decided"] else None,
                entity_type=r["entity_type"],
            )
            for r in rows
        ]

    def entity_memories(
        self, entity_id: str, limit: int = 10, *, include_invalid: bool = False,
        scope: Scope | None = None, history: bool = False,
        categories: list[str] | None = None, mentioning: str | list[str] | None = None,
        among: Any = None,
    ) -> list[Memory]:
        clause, params = _entity_reads_clause(
            include_invalid=include_invalid, scope=scope, history=history,
            categories=categories, mentioning=mentioning, among=among)
        with self._lock:
            rows = self._db.execute(
                f"SELECT DISTINCT {', '.join('m.' + c.strip() for c in _MEMORY_COLS.split(','))} "
                "FROM entity_mentions em JOIN memories m ON m.id = em.memory_id "
                f"WHERE em.entity_id = ? AND {clause} "
                "ORDER BY m.updated_at DESC, m.id LIMIT ?",
                (entity_id, *params, limit),
            ).fetchall()
        return [_row_to_memory(r) for r in rows]

    def count_entity_memories(self, entity_id: str) -> int:
        with self._lock:
            row = self._db.execute(
                "SELECT COUNT(DISTINCT m.id) FROM entity_mentions em "
                "JOIN memories m ON m.id = em.memory_id "
                "WHERE em.entity_id = ? AND m.invalid_at IS NULL",
                (entity_id,),
            ).fetchone()
        return int(row[0])

    def set_entity_type(self, entity_id: str, entity_type: str) -> None:
        if entity_type not in ENTITY_TYPES:
            raise ValueError(f"unknown entity type: {entity_type!r}")
        with self._lock:
            self._db.execute(
                "UPDATE entities SET entity_type = ?, updated_at = ?, "
                "description_updated_at = NULL WHERE id = ?",
                (entity_type, utcnow(), entity_id),
            )
            self._commit()

    def set_entity_metadata(self, entity_id: str, metadata: dict[str, Any]) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE entities SET metadata = ? WHERE id = ?",
                (json.dumps(metadata), entity_id),
            )
            self._commit()

    def entity_memory_links(
        self, scope: Scope, *, kind: str = "named"
    ) -> list[tuple[str, str]]:
        clause, params = _scope_clause(scope, prefix="e.")
        with self._lock:
            rows = self._db.execute(
                "SELECT DISTINCT em.entity_id, em.memory_id FROM entity_mentions em "
                "JOIN entities e ON e.id = em.entity_id "
                "JOIN memories m ON m.id = em.memory_id "
                f"WHERE e.merged_into IS NULL AND m.invalid_at IS NULL AND {clause} "
                f"AND {_kind_clause(kind, 'e.')}",
                params,
            ).fetchall()
        return [(row["entity_id"], row["memory_id"]) for row in rows]

    def entities_of_memory(self, memory_id: str, *, kind: str = "named") -> list[Entity]:
        with self._lock:
            rows = self._db.execute(
                "SELECT e.* FROM entity_mentions em JOIN entities e ON e.id = em.entity_id "
                f"WHERE em.memory_id = ? AND e.merged_into IS NULL AND {_kind_clause(kind, 'e.')} "
                "ORDER BY em.created_at, em.id",
                (memory_id,),
            ).fetchall()
        # distinct by id (a memory can mention an entity under several surfaces)
        seen, out = set(), []
        for r in rows:
            if r["id"] not in seen:
                seen.add(r["id"])
                out.append(self._row_to_entity(r))
        return out

    def entities_of_memories(
        self, memory_ids: list[str], *, kind: str = "named"
    ) -> dict[str, list[Entity]]:
        out: dict[str, list[Entity]] = {mid: [] for mid in memory_ids}
        ids = list(out)
        seen: set[tuple[str, str]] = set()
        for start in range(0, len(ids), 500):
            chunk = ids[start:start + 500]
            with self._lock:
                rows = self._db.execute(
                    "SELECT em.memory_id AS mentioned_in, e.* FROM entity_mentions em "
                    "JOIN entities e ON e.id = em.entity_id "
                    f"WHERE em.memory_id IN ({','.join('?' * len(chunk))}) "
                    f"AND e.merged_into IS NULL AND {_kind_clause(kind, 'e.')} "
                    "ORDER BY em.created_at, em.id",
                    chunk,
                ).fetchall()
            for r in rows:  # distinct by memory and entity, as entities_of_memory
                if (r["mentioned_in"], r["id"]) not in seen:
                    seen.add((r["mentioned_in"], r["id"]))
                    out[r["mentioned_in"]].append(self._row_to_entity(r))
        return out

    def entity_memory_counts(
        self, entity_ids: list[str], *, scope: Scope | None = None, history: bool = False,
        categories: list[str] | None = None, mentioning: str | list[str] | None = None,
        among: Any = None,
    ) -> dict[str, int]:
        out = dict.fromkeys(entity_ids, 0)
        ids = list(out)
        clause, params = _entity_reads_clause(
            include_invalid=False, scope=scope, history=history, categories=categories,
            mentioning=mentioning, among=among)
        for start in range(0, len(ids), 500):
            chunk = ids[start:start + 500]
            with self._lock:
                rows = self._db.execute(
                    "SELECT em.entity_id AS entity_id, COUNT(DISTINCT m.id) AS n "
                    "FROM entity_mentions em JOIN memories m ON m.id = em.memory_id "
                    f"WHERE em.entity_id IN ({','.join('?' * len(chunk))}) "
                    f"AND {clause} GROUP BY em.entity_id",
                    (*chunk, *params),
                ).fetchall()
            out.update((r["entity_id"], int(r["n"])) for r in rows)
        return out

    def topic_ids(self, entity_ids: Iterable[str]) -> set[str]:
        ids = sorted(set(entity_ids))
        out: set[str] = set()
        for start in range(0, len(ids), 500):
            chunk = ids[start:start + 500]
            with self._lock:
                rows = self._db.execute(
                    "SELECT id FROM entities WHERE entity_type = ? "
                    f"AND id IN ({','.join('?' * len(chunk))})",
                    (TOPIC_TYPE, *chunk),
                ).fetchall()
            out.update(row["id"] for row in rows)
        return out

    def delete_entity(self, entity_id: str) -> bool:
        """Remove an entity and everything that points at it, for good.

        Internal: the store retires entities instead, so that a removal can be
        taken back. This is the irreversible form, kept for callers inside the
        backend that have already preserved whatever needed preserving.
        """
        with self._lock:
            row = self._db.execute(
                "SELECT id FROM entities WHERE id = ?", (entity_id,)
            ).fetchone()
            if row is None:
                return False
            # tombstones redirecting here would dangle; they carry nothing
            self._remove_entity_rows_locked(entity_id)
            self._commit()
        return True

    # -- retirement: removal with a way back --------------------------------
    def _chain_locked(self, entity_id: str) -> list[str]:
        """The entity and every entity merged into it, directly or through
        another ("Tomi" into "T. Vell" into "Tomas Vell"), the entity first."""
        return [entity_id] + [row["id"] for row in self._db.execute(
            "WITH RECURSIVE chain(id) AS (SELECT ? UNION "
            "SELECT e.id FROM entities e JOIN chain ON e.merged_into = chain.id) "
            "SELECT id FROM chain WHERE id != ? ORDER BY id",
            (entity_id, entity_id),
        ).fetchall()]

    def _entity_snapshot_locked(self, entity_id: str) -> dict[str, Any] | None:
        """Everything that would be lost by removing this entity: its row
        and its tombstones (every entity merged into it, however far down
        the chain), and the mentions, relations, pairs and merge records of
        any of them.

        Kept as plain row dicts so a restore can put the rows back exactly as
        they were, without depending on the model classes of the day.
        """
        entity = self._db.execute(
            "SELECT * FROM entities WHERE id = ?", (entity_id,)
        ).fetchone()
        if entity is None:
            return None
        chain = self._chain_locked(entity_id)
        marks = ",".join("?" * len(chain))

        def rows(sql: str, times: int) -> list[dict[str, Any]]:
            return [dict(row) for row in self._db.execute(sql, tuple(chain) * times).fetchall()]

        return {
            "entity": dict(entity),
            "aliases": self.entity_aliases(entity_id),
            "tombstones": [
                row for row in rows(f"SELECT * FROM entities WHERE id IN ({marks})", 1)
                if row["id"] != entity_id
            ],
            "mentions": rows(
                f"SELECT * FROM entity_mentions WHERE entity_id IN ({marks})", 1),
            "relations": rows(
                f"SELECT * FROM relations WHERE subject IN ({marks}) OR object IN ({marks})", 2),
            "proposals": rows(
                "SELECT * FROM entity_proposals "
                f"WHERE entity_a IN ({marks}) OR entity_b IN ({marks})", 2),
            "merges": rows(
                "SELECT * FROM entity_merges "
                f"WHERE keep_id IN ({marks}) OR merge_id IN ({marks})", 2),
        }

    def _remove_entity_rows_locked(self, entity_id: str) -> None:
        """The deletions of ``delete_entity``, without the bookkeeping: the
        entity, every tombstone of its chain, and the rows naming any of
        them, so no tombstone, pair or merge record points at nothing."""
        chain = self._chain_locked(entity_id)
        marks = ",".join("?" * len(chain))
        for sql, times in (
            (f"DELETE FROM entity_mentions WHERE entity_id IN ({marks})", 1),
            (f"DELETE FROM relations WHERE subject IN ({marks}) OR object IN ({marks})", 2),
            ("DELETE FROM entity_proposals "
             f"WHERE entity_a IN ({marks}) OR entity_b IN ({marks})", 2),
            (f"DELETE FROM entity_merges WHERE keep_id IN ({marks}) OR merge_id IN ({marks})",
             2),
            (f"DELETE FROM entities WHERE id IN ({marks})", 1),
        ):
            self._db.execute(sql, tuple(chain) * times)

    def _retire_locked(self, entity_id: str, reason: str) -> bool:
        row = self._db.execute(
            "SELECT entity_type FROM entities WHERE id = ?", (entity_id,)
        ).fetchone()
        if row is None:
            return False
        if row["entity_type"] != TOPIC_TYPE:
            # A tag folded into it, or into anything merged into it, is a tag
            # again: its tombstone would otherwise point at nothing, filing
            # would make a fresh topic beside it, and a restore would bring
            # the fold back, two entities claiming one tag. A retired tag's
            # own tombstones (tags merged into it) go with it.
            self._unfold_tags_locked(entity_id)
        snapshot = self._entity_snapshot_locked(entity_id)
        if snapshot is None:
            return False
        entity = snapshot["entity"]
        self._db.execute(
            "INSERT OR REPLACE INTO retired_entities "
            "(entity_id, user_id, name, entity_type, reason, retired_at, snapshot) "
            "VALUES (?,?,?,?,?,?,?)",
            (
                entity_id, entity["user_id"], entity["name"], entity["entity_type"],
                reason, utcnow(), json.dumps(snapshot),
            ),
        )
        self._remove_entity_rows_locked(entity_id)
        return True

    def _unfold_tags_locked(self, entity_id: str) -> None:
        """Make each tag folded into ``entity_id`` (a topic entity whose
        tombstone points at it, or at a named thing merged into it: "bildy"
        the tag into "Bildy" the product) a topic again, before the entity
        is retired: its tombstone becomes the active topic of its name (or,
        where one is active already, points at that one), and the memories
        filed under the name are filed again, so their mentions go to the
        topic instead of the entity going away. The record of the fold goes:
        the fold is undone.

        The entity's mentions that came from the tag go first
        (``_unmention_tags_locked``), so its snapshot holds its named
        mentions only and a restore brings back no second mention of a
        memory beside the topic's. The pair of the entity and each unfolded
        topic is recorded as kept apart (``_record_unfolded_pair_locked``):
        a restore does not fold the tag back, and the pair is not raised
        again for the judge to fold it back either."""
        now = utcnow()
        names: dict[str | None, set[str]] = {}
        topics: dict[str, str | None] = {}
        chain = self._chain_locked(entity_id)
        marks = ",".join("?" * len(chain))
        for tombstone in self._db.execute(
            "SELECT t.id, t.user_id, t.normalized FROM entities t "
            "JOIN entities thing ON thing.id = t.merged_into "
            f"WHERE t.merged_into IN ({marks}) AND t.entity_type = ? "
            "AND IFNULL(thing.entity_type, '') != ? ORDER BY t.created_at, t.id",
            (*chain, TOPIC_TYPE, TOPIC_TYPE),
        ).fetchall():
            active = self._active_topic_locked(tombstone["normalized"], tombstone["user_id"])
            self._db.execute(
                "UPDATE entities SET merged_into = ?, updated_at = ?, "
                "description_updated_at = NULL WHERE id = ?",
                (active, now, tombstone["id"]),
            )
            self._db.execute("DELETE FROM entity_merges WHERE merge_id = ?", (tombstone["id"],))
            names.setdefault(tombstone["user_id"], set()).add(tombstone["normalized"])
            topics.setdefault(active or tombstone["id"], tombstone["user_id"])
        for user_id, unfolded in names.items():
            self._unmention_tags_locked(entity_id, user_id, unfolded)
            self._refile_named_locked(user_id, unfolded)
        for topic_id, user_id in topics.items():
            self._record_unfolded_pair_locked(entity_id, topic_id, user_id, now)

    def _unmention_tags_locked(
        self, entity_id: str, user_id: str | None, names: set[str]
    ) -> None:
        """Remove the mentions of ``entity_id`` that a tag in ``names`` folded
        into it made: on each memory of ``user_id`` whose column names such a
        tag, the mention whose surface is the tag as the column writes it
        (``_mention_tags_locked`` writes a tag's mention under that name, and
        a merge moved the topic's mentions, written so, onto the entity). A
        mention under another surface ("Bildy" in "Bildy runs on AWS", beside
        the tag "bildy") names the entity, and stays."""
        for row in self._memories_naming_locked(user_id, names):
            surfaces = sorted({
                str(tag).strip() for tag in json.loads(row["categories"])
                if str(tag).strip().lower() in names
            })
            if not surfaces:
                continue
            self._db.execute(
                "DELETE FROM entity_mentions WHERE entity_id = ? AND memory_id = ? "
                f"AND trim(surface) IN ({','.join('?' * len(surfaces))})",
                (entity_id, row["id"], *surfaces),
            )

    def _record_unfolded_pair_locked(
        self, entity_id: str, topic_id: str, user_id: str | None, now: str
    ) -> None:
        """Record the entity and a topic unfolded from it as a pair kept
        apart, unless the two are a pair already. It lands in the entity's
        snapshot with its other pairs and comes back on a restore, so the
        pair of a thing and the tag of its name
        (``entities.propose_same_name_duplicates``) is not raised again."""
        if self._db.execute(
            "SELECT 1 FROM entity_proposals WHERE (entity_a = ? AND entity_b = ?) "
            "OR (entity_a = ? AND entity_b = ?)",
            (entity_id, topic_id, topic_id, entity_id),
        ).fetchone() is not None:
            return
        self._db.execute(
            "INSERT INTO entity_proposals (id, entity_a, entity_b, user_id, status, "
            "confidence, reason, created_at, decided_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (new_id(), entity_id, topic_id, user_id, "rejected", 0.0,
             "kept apart: the tag was unfolded when this was removed", now, now),
        )

    def _memories_naming_locked(
        self, user_id: str | None, names: set[str]
    ) -> list[sqlite3.Row]:
        """(id, categories) of the memories of ``user_id`` (exactly) whose
        column names one of ``names``, read through the legacy index."""
        wanted = sorted(names)
        if not wanted:
            return []
        return self._db.execute(
            "SELECT DISTINCT m.id, m.categories FROM memories m "
            "JOIN memory_topics mt ON mt.memory_id = m.id "
            "JOIN topics t ON t.id = mt.topic_id "
            f"WHERE m.user_id IS ? AND t.normalized IN ({','.join('?' * len(wanted))})",
            (user_id, *wanted),
        ).fetchall()

    def _refile_named_locked(self, user_id: str | None, names: set[str]) -> None:
        """File again the memories of ``user_id`` (exactly) whose column names
        one of ``names`` (``_refile_locked``)."""
        ids = [row["id"] for row in self._memories_naming_locked(user_id, names)]
        cache: dict[Any, Any] = {}
        for start in range(0, len(ids), 500):
            chunk = ids[start:start + 500]
            self._refile_locked(
                f"id IN ({','.join('?' * len(chunk))})", tuple(chunk), cache=cache)

    def retire_entity(self, entity_id: str, reason: str = "removed") -> bool:
        """Remove an entity the recoverable way: snapshot first, then delete.

        Same end state as ``delete_entity`` - the entity, every tombstone of
        its merge chain, and their mentions, relations, pairs and merge
        records are gone and the memories are untouched - except that
        everything removed is kept in ``retired_entities`` so
        ``restore_entity`` can put it back. The memories that read its names
        as "it" are the caller's to refresh (``MemoryStore._retire``):
        once it is gone nothing links them to it.
        """
        with self._lock:
            retired = self._retire_locked(entity_id, reason)
            if retired:
                self._commit()
            return retired

    def _insert_row_locked(self, table: str, row: dict[str, Any]) -> None:
        columns = ", ".join(row)
        marks = ",".join("?" * len(row))
        self._db.execute(
            f"INSERT OR IGNORE INTO {table} ({columns}) VALUES ({marks})",
            tuple(row.values()),
        )

    def restore_entity(self, entity_id: str) -> bool:
        """Put a retired entity back, with whatever it pointed at still exists.

        Its tombstones come back pointing as they did, so a merge into it,
        however far down the chain, can still be undone. Mentions come back
        only for memories that are still there; edges, pairs and merge
        records only where the entities at both ends are there, and an edge
        in use only while its memory is: an edge whose memory went out of
        use meanwhile comes back out of use with it (as ``invalidate_memory``
        would have left it; ``revalidate_memory`` brings both back), and one
        whose memory is gone does not come back. A restore must not
        resurrect rows that dangle. A retired tag whose name a fresh topic
        has taken since (a memory tagged so after ``delete_tag``) is not
        restored: one active topic per name. ``names_changed`` hears of the
        entity back.
        """
        with self._lock:
            row = self._db.execute(
                "SELECT snapshot FROM retired_entities WHERE entity_id = ?",
                (entity_id,),
            ).fetchone()
            if row is None:
                return False
            if self._db.execute(
                "SELECT 1 FROM entities WHERE id = ?", (entity_id,)
            ).fetchone() is not None:
                return False  # something already lives under this id
            snapshot = json.loads(row["snapshot"])
            entity = snapshot["entity"]
            if entity.get("entity_type") == TOPIC_TYPE and self._active_topic_locked(
                    entity["normalized"], entity["user_id"]) is not None:
                return False  # the name lives on in a fresh topic
            self._insert_row_locked("entities", entity)
            refile: dict[str | None, set[str]] = {}
            # a retired tag's own tombstones (tags merged into it) come back
            # pointing at it; only a thing's are unfolded
            unfold = entity.get("entity_type") != TOPIC_TYPE
            for tombstone in snapshot.get("tombstones", []):
                if unfold and tombstone.get("entity_type") == TOPIC_TYPE:
                    # a tag folded into it before retiring unfolded tags (a
                    # snapshot of before) comes back a topic, not folded again:
                    # the active topic of its name, or pointing at that one
                    tombstone = {**tombstone, "merged_into": self._active_topic_locked(
                        tombstone["normalized"], tombstone["user_id"])}
                    refile.setdefault(tombstone["user_id"], set()).add(
                        tombstone["normalized"])
                self._insert_row_locked("entities", tombstone)
            for mention in snapshot.get("mentions", []):
                if self._db.execute(
                    "SELECT 1 FROM memories WHERE id = ?", (mention["memory_id"],)
                ).fetchone() is not None:
                    self._insert_row_locked("entity_mentions", mention)

            def present(*entity_ids: str) -> bool:
                return all(self._db.execute(
                    "SELECT 1 FROM entities WHERE id = ?", (other,)
                ).fetchone() is not None for other in entity_ids)

            for relation in snapshot.get("relations", []):
                if not present(relation["subject"], relation["object"]):
                    continue
                if relation.get("memory_id") is not None:
                    evidence = self._db.execute(
                        "SELECT invalid_at FROM memories WHERE id = ?",
                        (relation["memory_id"],),
                    ).fetchone()
                    if evidence is None:
                        continue
                    if relation.get("invalid_at") is None and evidence["invalid_at"]:
                        relation = {**relation, "invalid_at": evidence["invalid_at"]}
                self._insert_row_locked("relations", relation)
            for proposal in snapshot.get("proposals", []):
                if present(proposal["entity_a"], proposal["entity_b"]):
                    self._insert_row_locked("entity_proposals", proposal)
            for merge in snapshot.get("merges", []):
                if present(merge["keep_id"], merge["merge_id"]):
                    self._insert_row_locked("entity_merges", merge)
            for user_id, names in refile.items():
                # a snapshot of before holds the mentions the tags made too
                self._unmention_tags_locked(entity_id, user_id, names)
                self._refile_named_locked(user_id, names)
            for tombstone in snapshot.get("tombstones", []):
                if unfold and tombstone.get("entity_type") == TOPIC_TYPE:
                    self._record_unfolded_pair_locked(
                        entity_id, self.resolve_entity_id(tombstone["id"]) or tombstone["id"],
                        tombstone["user_id"], utcnow())
            self._restore_aliases_locked(entity_id, snapshot.get("aliases", []))
            self._db.execute(
                "DELETE FROM retired_entities WHERE entity_id = ?", (entity_id,)
            )
            self._commit()
        # its names read "it" in its memories again
        if self.names_changed is not None:
            self.names_changed([entity_id])
        return True

    def _restore_aliases_locked(self, entity_id: str, aliases: list[str]) -> None:
        """Keep every name the entity answered to, even without its evidence.

        Most aliases come back with the rows they were derived from. Any that
        do not - a surface whose memory is gone, a tombstone that could not be
        restored - are written into the entity's own metadata, because a name
        the user taught the system is not something a cleanup should cost them.
        """
        known = {value.strip().lower() for value in self.entity_aliases(entity_id)}
        missing = [
            value.strip()
            for value in aliases
            if value.strip() and value.strip().lower() not in known
        ]
        if not missing:
            return
        row = self._db.execute(
            "SELECT metadata FROM entities WHERE id = ?", (entity_id,)
        ).fetchone()
        if row is None:
            return
        metadata = json.loads(row["metadata"])
        raw = metadata.get("aliases", [])
        if isinstance(raw, str):
            raw = [raw]
        metadata["aliases"] = [
            str(value).strip() for value in raw if str(value).strip()
        ] + missing
        self._db.execute(
            "UPDATE entities SET metadata = ? WHERE id = ?",
            (json.dumps(metadata), entity_id),
        )
        self._has_metadata_aliases = True

    def list_retired_entities(
        self, scope: Scope, *, limit: int = 200
    ) -> list[dict[str, Any]]:
        """Retired entities, newest first. The snapshot itself stays behind."""
        # The trash row carries the namespace only; agent/run live in the
        # snapshot, and nothing lists retired names per agent or run.
        clause, params = _scope_clause(_user_scope(scope))
        with self._lock:
            rows = self._db.execute(
                "SELECT entity_id, user_id, name, entity_type, reason, retired_at "
                f"FROM retired_entities WHERE {clause} "
                "ORDER BY retired_at DESC, name, entity_id LIMIT ?",
                (*params, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def merge_entities(self, keep_id: str, merge_id: str) -> bool:
        """Idempotently fold both IDs' active roots into one entity.

        A tag folded together with a named thing ("bildy" the tag and "Bildy"
        the product) always goes into the thing, whichever side the caller
        kept: the thing keeps its type, relations and name, and the tag's
        memories become its mentions. A side found folded meanwhile (the
        fold's UPDATE matched no row) rolls back: the UPDATE opened the
        transaction, which would otherwise hold the write lock. A fold
        written is announced to ``names_changed`` with the entity kept."""
        with self._lock:
            folded = self._merge_entities_locked(keep_id, merge_id)
            if folded:
                self._commit()
            elif folded is None and self._db.in_transaction:
                self._db.rollback()
        if folded and self.names_changed is not None:
            self.names_changed([self.resolve_entity_id(keep_id) or keep_id])
        return folded is not None

    def _merge_entities_locked(self, keep_id: str, merge_id: str) -> bool | None:
        """The writes of ``merge_entities``, left uncommitted, so that a caller
        can make them part of its own transaction (the open's
        ``_ensure_one_active_topic_per_name``). True when a fold was written,
        False when the two were one already, None when either is missing or
        was folded meanwhile; nothing is written but for True."""
        keep_root = self.resolve_entity_id(keep_id)
        merge_root = self.resolve_entity_id(merge_id)
        if keep_root is None or merge_root is None:
            return None
        if keep_root == merge_root:
            return False
        types = {
            row["id"]: row["entity_type"] for row in self._db.execute(
                "SELECT id, entity_type FROM entities WHERE id IN (?, ?)",
                (keep_root, merge_root),
            ).fetchall()
        }
        if types.get(keep_root) == TOPIC_TYPE and types.get(merge_root) != TOPIC_TYPE:
            keep_root, merge_root = merge_root, keep_root
        # A tag kept means two tags merged, and that rewrites their memories'
        # tags (``retag_topics``), which is not undone here; every other merge
        # (a tag folded into a thing too) records what it moves.
        snapshot = (self._merge_snapshot_locked(keep_root, merge_root)
                    if types.get(keep_root) != TOPIC_TYPE else None)
        changed_at = utcnow()
        cur = self._db.execute(
            "UPDATE entities SET merged_into = ?, updated_at = ?, "
            "description_updated_at = NULL "
            "WHERE id = ? AND merged_into IS NULL",
            (keep_root, changed_at, merge_root),
        )
        if cur.rowcount == 0:
            return None
        kind = self._merged_type_locked(keep_root, merge_root)
        self._db.execute(
            "UPDATE entity_mentions SET entity_id = ? WHERE entity_id = ?",
            (keep_root, merge_root),
        )
        self._db.execute(
            "UPDATE entities SET entity_type = ? WHERE id = ? AND entity_type IS NOT ?",
            (kind, keep_root, kind),
        )
        self._db.execute(
            "UPDATE relations SET subject = ? WHERE subject = ?",
            (keep_root, merge_root),
        )
        self._db.execute(
            "UPDATE relations SET object = ? WHERE object = ?",
            (keep_root, merge_root),
        )
        self._db.execute(
            "UPDATE relations SET invalid_at = ? "
            "WHERE subject = object AND invalid_at IS NULL",
            (changed_at,),
        )
        # The pair of the two keeps its ends, which say which two entities
        # were found to be one, beside the answer or the rule that decided
        # it; so do the pairs earlier merges confirmed. The merged entity's
        # other pairs, open or kept apart, become the kept one's, one row a
        # pair (``_one_row_per_pair_locked``).
        self._db.execute(
            "UPDATE entity_proposals SET status = 'confirmed', decided_at = ? "
            "WHERE status = 'proposed' AND ((entity_a = ? AND entity_b = ?) "
            "OR (entity_a = ? AND entity_b = ?))",
            (changed_at, keep_root, merge_root, merge_root, keep_root),
        )
        dropped = self._one_row_per_pair_locked(keep_root, merge_root)
        if snapshot is not None:
            snapshot["dropped"] = dropped
        moved = [row["id"] for row in self._db.execute(
            "SELECT id FROM entity_proposals WHERE status = 'proposed' "
            "AND (entity_a = ? OR entity_b = ?) AND entity_a != ? AND entity_b != ?",
            (merge_root, merge_root, keep_root, keep_root),
        ).fetchall()]
        self._db.execute(
            "UPDATE entity_proposals SET entity_a = ? "
            "WHERE entity_a = ? AND entity_b != ? AND status != 'confirmed'",
            (keep_root, merge_root, keep_root),
        )
        self._db.execute(
            "UPDATE entity_proposals SET entity_b = ? "
            "WHERE entity_b = ? AND entity_a != ? AND status != 'confirmed'",
            (keep_root, merge_root, keep_root),
        )
        # Only the pairs the merged entity brought start the funnel again:
        # they were judged against its memories alone. The kept entity's own
        # pairs keep their step, since the funnel compares a pair again once
        # its smaller side reaches the next step anyway (``identity.rounds``).
        # Restarting them all re-judged about 1000 pairs after one owner merge
        # on a live store, 987 of them to the same "wait".
        self._db.executemany(
            "UPDATE entity_proposals SET compared_step = 0 WHERE id = ?",
            [(proposal_id,) for proposal_id in moved],
        )
        self._db.execute(
            "UPDATE entities SET updated_at = ?, description_updated_at = NULL "
            "WHERE id = ?",
            (changed_at, keep_root),
        )
        if snapshot is not None:
            self._db.execute(
                "INSERT INTO entity_merges (id, keep_id, merge_id, user_id, merged_at, snapshot) "
                "VALUES (?,?,?,?,?,?)",
                (new_id(), keep_root, merge_root, snapshot["merged"]["user_id"], changed_at,
                 json.dumps(snapshot)),
            )
        return True

    def _merged_type_locked(self, keep_id: str, merge_id: str) -> str | None:
        """The type of what a merge keeps: the one most of both entities'
        mentions give (``_type_votes_locked``), and on a tie the type of the
        one the store had first. A second entity of a known name, made
        because one sentence typed it otherwise, is the newer one, so a tie
        does not hand the thing that sentence's type. Two tags merge as a
        tag."""
        rows = {row["id"]: row for row in self._db.execute(
            "SELECT id, entity_type, created_at, metadata FROM entities WHERE id IN (?, ?)",
            (keep_id, merge_id),
        ).fetchall()}
        keep, merged = rows[keep_id], rows[merge_id]
        if keep["entity_type"] == TOPIC_TYPE:
            return keep["entity_type"]
        # a type the owner chose outlasts the count, the kept one's first
        for row in (keep, merged):
            if row["entity_type"] != TOPIC_TYPE and _owner_typed(row):
                return row["entity_type"]

        def own(row: sqlite3.Row) -> str | None:
            return None if row["entity_type"] == TOPIC_TYPE else row["entity_type"]

        votes = (self._type_votes_locked(keep_id, own(keep))
                 + self._type_votes_locked(merge_id, own(merged)))
        first = min((keep, merged), key=lambda row: (row["created_at"], row["id"]))
        ties = [kind for kind in (own(first), own(keep)) if kind] or [own(keep)]
        return self._most_given(votes, ties)

    #: How decided a pair's row is: of two rows for one pair, the more
    #: decided stays (``_one_row_per_pair_locked``).
    _DECIDED = {"proposed": 0, "rejected": 1, "confirmed": 2}

    def _one_row_per_pair_locked(self, keep_id: str, merge_id: str) -> list[dict[str, Any]]:
        """Before a merge points the merged entity's pairs at the kept one:
        where the kept one has a row for the same pair (both were compared
        with a third entity), one row stays. That is the more decided one (a
        decision over an open pair), of two alike the later answer, the kept
        entity's own on a tie; it carries its answer or decision, and the
        other row goes. A confirmed row keeps the ends it was decided on and
        is never dropped. Returns the rows dropped, as they were, for the
        merge record: an undo puts them back."""
        dropped: list[dict[str, Any]] = []
        moving = self._db.execute(
            "SELECT * FROM entity_proposals WHERE status != 'confirmed' "
            "AND ((entity_a = ? AND entity_b != ?) OR (entity_b = ? AND entity_a != ?)) "
            "ORDER BY created_at, id",
            (merge_id, keep_id, merge_id, keep_id),
        ).fetchall()
        for row in moving:
            other = row["entity_b"] if row["entity_a"] == merge_id else row["entity_a"]
            held = self._db.execute(
                "SELECT * FROM entity_proposals WHERE (entity_a = ? AND entity_b = ?) "
                "OR (entity_a = ? AND entity_b = ?) ORDER BY created_at, id",
                (keep_id, other, other, keep_id),
            ).fetchall()
            if not held:
                continue
            if any(r["status"] == "confirmed" for r in held):
                # the kept one and the third were found one: the pair is moot
                losers = [row]
            else:
                rows = [row, *held]
                best = max(rows, key=lambda r: (
                    self._DECIDED.get(r["status"], 0), r["decided_at"] or r["created_at"],
                    r["id"] != row["id"]))
                losers = [r for r in rows if r["id"] != best["id"]]
            for loser in losers:
                dropped.append(dict(loser))
                self._db.execute("DELETE FROM entity_proposals WHERE id = ?", (loser["id"],))
        return dropped

    def _one_row_per_pair_everywhere(self) -> None:
        """Once per database: each pair of entities keeps one row. A merge
        before 0.2.40 pointed the merged entity's pairs at the kept one without
        looking for a row the kept one had for the same third entity, so a
        pass that paired the store owner with "Cosmin" and "Cosmin Novac"
        left, once those two were merged, two rows for the owner and "Cosmin",
        both created in that pass. Of each pair's rows the one a merge keeps
        stays (``_one_row_per_pair_locked``): confirmed rows all stay as they
        are, and beside one the rest go (the two are one, the pair is moot);
        otherwise the more decided, of two alike the later answer. The rows
        that go are kept under the marker, so each can be put back."""
        if self._db.execute(
            "SELECT 1 FROM meta WHERE key = 'schema:one-row-per-pair:v1'"
        ).fetchone():
            return
        groups = self._db.execute(
            "SELECT MIN(entity_a, entity_b) AS lo, MAX(entity_a, entity_b) AS hi "
            "FROM entity_proposals GROUP BY lo, hi HAVING COUNT(*) > 1"
        ).fetchall()
        dropped: list[dict[str, Any]] = []
        for group in groups:
            rows = self._db.execute(
                "SELECT * FROM entity_proposals WHERE (entity_a = ? AND entity_b = ?) "
                "OR (entity_a = ? AND entity_b = ?) ORDER BY created_at, id",
                (group["lo"], group["hi"], group["hi"], group["lo"]),
            ).fetchall()
            if any(r["status"] == "confirmed" for r in rows):
                losers = [r for r in rows if r["status"] != "confirmed"]
            else:
                best = max(rows, key=lambda r: (
                    self._DECIDED.get(r["status"], 0), r["decided_at"] or r["created_at"],
                    r["id"]))
                losers = [r for r in rows if r["id"] != best["id"]]
            for loser in losers:
                dropped.append(dict(loser))
                self._db.execute("DELETE FROM entity_proposals WHERE id = ?", (loser["id"],))
        self._db.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema:one-row-per-pair:v1', ?)",
            (json.dumps({"at": utcnow(), "dropped": dropped}),),
        )

    def _merge_snapshot_locked(self, keep_id: str, merge_id: str) -> dict[str, Any]:
        """What a merge of ``merge_id`` into ``keep_id`` is about to move, read
        before it moves it: both entity rows and the names each answered to,
        the merged one's mentions, the relations and the pairs that will point
        at the kept one (with the ends that will change, and whether a
        relation between the two becomes a loop and is closed), and the funnel
        step of each open pair the merge starts again."""
        def row(entity_id: str) -> dict[str, Any]:
            return dict(self._db.execute(
                "SELECT * FROM entities WHERE id = ?", (entity_id,)).fetchone())

        both = {keep_id, merge_id}
        relations = [
            {"id": r["id"], "subject": r["subject"] == merge_id, "object": r["object"] == merge_id,
             "loop": r["invalid_at"] is None and {r["subject"], r["object"]} <= both}
            for r in self._db.execute(
                "SELECT id, subject, object, invalid_at FROM relations "
                "WHERE subject = ? OR object = ?", (merge_id, merge_id)).fetchall()
        ]
        proposals = [
            {"id": p["id"], "a": p["entity_a"] == merge_id, "b": p["entity_b"] == merge_id}
            for p in self._db.execute(
                "SELECT id, entity_a, entity_b FROM entity_proposals WHERE status != 'confirmed' "
                "AND ((entity_a = ? AND entity_b != ?) OR (entity_b = ? AND entity_a != ?))",
                (merge_id, keep_id, merge_id, keep_id)).fetchall()
        ]
        steps = {
            p["id"]: p["compared_step"] for p in self._db.execute(
                "SELECT id, compared_step FROM entity_proposals WHERE status = 'proposed' "
                "AND (entity_a IN (?, ?) OR entity_b IN (?, ?))",
                (keep_id, merge_id, keep_id, merge_id)).fetchall()
        }
        return {
            "keep": row(keep_id), "merged": row(merge_id),
            "keep_names": self.entity_aliases(keep_id),
            "merged_names": self.entity_aliases(merge_id),
            "mentions": [m["id"] for m in self._db.execute(
                "SELECT id FROM entity_mentions WHERE entity_id = ?", (merge_id,)).fetchall()],
            "relations": relations, "proposals": proposals, "steps": steps,
        }

    def list_merges(self, scope: Scope, *, limit: int = 200) -> list[dict[str, Any]]:
        """Merges that can be undone, newest first: the entity merged away and
        the one it went into, by id and by name as they are now, when, and
        what decided it (the reason on their pair)."""
        clause, params = _scope_clause(_user_scope(scope), prefix="g.")
        with self._lock:
            rows = self._db.execute(
                "SELECT g.merge_id AS entity_id, g.keep_id, g.user_id, g.merged_at, "
                "m.name AS name, m.entity_type AS entity_type, k.name AS keep_name, "
                "(SELECT p.reason FROM entity_proposals p WHERE p.status = 'confirmed' AND "
                " ((p.entity_a = g.keep_id AND p.entity_b = g.merge_id) OR "
                "  (p.entity_a = g.merge_id AND p.entity_b = g.keep_id)) "
                " ORDER BY p.decided_at DESC LIMIT 1) AS decided "
                "FROM entity_merges g LEFT JOIN entities m ON m.id = g.merge_id "
                "LEFT JOIN entities k ON k.id = g.keep_id "
                f"WHERE {clause} ORDER BY g.merged_at DESC, g.id DESC LIMIT ?",
                (*params, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def merge_record(self, entity_id: str) -> dict[str, Any] | None:
        """The last merge of ``entity_id`` into another on record: the entity
        it went into (``keep_id``), its namespace and when."""
        with self._lock:
            row = self._db.execute(
                "SELECT keep_id, merge_id, user_id, merged_at FROM entity_merges "
                "WHERE merge_id = ? ORDER BY merged_at DESC, id DESC LIMIT 1",
                (entity_id,),
            ).fetchone()
        return dict(row) if row else None

    def undo_merge(self, entity_id: str) -> bool:
        """Undo the last merge of ``entity_id`` into another entity, from what
        the merge recorded (``_merge_snapshot_locked``). The merged entity
        comes back as it was, with its names; its mentions, relations and
        pairs point at it again, and the pairs whose funnel the merge started
        again are back at their steps. A tag folded into a thing is a tag
        again, and the memories carrying it are filed under it.

        Memories saved since the merge stay with the kept entity, unless a
        mention of theirs calls it by a name only the merged one answered to
        before the merge: that mention goes back to the merged entity, and a
        relation its memory stated since goes with it when the memory no
        longer names the kept one. A mention another merge into the kept
        entity brought stays.

        The pair of the two is recorded as kept apart, "undone by you", so the
        funnel does not merge them again on the same evidence. False, writing
        nothing, when no merge is on record, or when the two are no longer as
        the merge left them: the kept entity was merged into another since
        (undo that one first) or removed, or the merged one is folded
        elsewhere. ``names_changed`` hears of both."""
        with self._lock:
            record = self._db.execute(
                "SELECT * FROM entity_merges WHERE merge_id = ? "
                "ORDER BY merged_at DESC, id DESC LIMIT 1", (entity_id,)).fetchone()
            if record is None:
                return False
            keep_id, merged_at = record["keep_id"], record["merged_at"]
            merged = self._db.execute(
                "SELECT merged_into FROM entities WHERE id = ?", (entity_id,)).fetchone()
            keep = self._db.execute(
                "SELECT merged_into, metadata FROM entities WHERE id = ?", (keep_id,)).fetchone()
            if (merged is None or merged["merged_into"] != keep_id
                    or keep is None or keep["merged_into"] is not None):
                return False
            snapshot = json.loads(record["snapshot"])
            now = utcnow()
            entity = dict(snapshot["merged"])
            tag = entity["entity_type"] == TOPIC_TYPE
            # a tag folded into a thing is the active topic of its name again
            entity["merged_into"] = (self._active_topic_locked(entity["normalized"],
                                                               entity["user_id"])
                                     if tag else None)
            columns = [column for column in entity if column != "id"]
            self._db.execute(
                f"UPDATE entities SET {', '.join(f'{c} = ?' for c in columns)} WHERE id = ?",
                (*(entity[c] for c in columns), entity_id),
            )
            moved: set[str] = set()
            if tag:
                names = {entity["normalized"]}
                self._unmention_tags_locked(keep_id, entity["user_id"], names)
                self._refile_named_locked(entity["user_id"], names)
            else:
                self._db.executemany(
                    "UPDATE entity_mentions SET entity_id = ? WHERE id = ? AND entity_id = ?",
                    [(entity_id, mention_id, keep_id) for mention_id in snapshot["mentions"]])
                brought = {
                    mention_id
                    for row in self._db.execute(
                        "SELECT snapshot FROM entity_merges WHERE keep_id = ? AND id != ?",
                        (keep_id, record["id"])).fetchall()
                    for mention_id in json.loads(row["snapshot"])["mentions"]
                }
                only_theirs = ({n.lower() for n in snapshot["merged_names"]}
                               - {n.lower() for n in snapshot["keep_names"]})
                for mention in self._db.execute(
                    "SELECT id, memory_id, surface FROM entity_mentions "
                    "WHERE entity_id = ? AND created_at >= ?", (keep_id, merged_at)).fetchall():
                    if (mention["id"] not in brought
                            and mention["surface"].strip().lower() in only_theirs):
                        self._db.execute("UPDATE entity_mentions SET entity_id = ? WHERE id = ?",
                                         (entity_id, mention["id"]))
                        moved.add(mention["memory_id"])
            for relation in snapshot["relations"]:
                current = self._db.execute(
                    "SELECT subject, object, invalid_at FROM relations WHERE id = ?",
                    (relation["id"],)).fetchone()
                if current is None:
                    continue
                subject = (entity_id if relation["subject"] and current["subject"] == keep_id
                           else current["subject"])
                obj = (entity_id if relation["object"] and current["object"] == keep_id
                       else current["object"])
                invalid_at = (None if relation["loop"] and current["invalid_at"] == merged_at
                              else current["invalid_at"])
                self._db.execute(
                    "UPDATE relations SET subject = ?, object = ?, invalid_at = ? WHERE id = ?",
                    (subject, obj, invalid_at, relation["id"]))
            naming_kept = {
                row["memory_id"] for row in self._db.execute(
                    "SELECT DISTINCT memory_id FROM entity_mentions WHERE entity_id = ?",
                    (keep_id,)).fetchall()
            }
            for memory_id in sorted(moved - naming_kept):
                for end in ("subject", "object"):
                    self._db.execute(
                        f"UPDATE relations SET {end} = ? WHERE {end} = ? AND memory_id = ? "
                        "AND created_at >= ?", (entity_id, keep_id, memory_id, merged_at))
            for proposal in snapshot["proposals"]:
                for end, flag in (("entity_a", "a"), ("entity_b", "b")):
                    if proposal[flag]:
                        self._db.execute(
                            f"UPDATE entity_proposals SET {end} = ? WHERE id = ? AND {end} = ?",
                            (entity_id, proposal["id"], keep_id))
            # a row the merge dropped for a pair both had is theirs again
            for row in snapshot.get("dropped", []):
                self._insert_row_locked("entity_proposals", row)
            self._db.executemany(
                "UPDATE entity_proposals SET compared_step = ? "
                "WHERE id = ? AND status = 'proposed' AND compared_step = 0",
                [(step, proposal_id) for proposal_id, step in snapshot["steps"].items()])
            pairs = self._db.execute(
                "UPDATE entity_proposals SET status = 'rejected', reason = ?, decided_at = ? "
                "WHERE (entity_a = ? AND entity_b = ?) OR (entity_a = ? AND entity_b = ?)",
                ("undone by you", now, keep_id, entity_id, entity_id, keep_id)).rowcount
            if not pairs:
                self._db.execute(
                    "INSERT INTO entity_proposals (id, entity_a, entity_b, user_id, status, "
                    "confidence, reason, created_at, decided_at) VALUES (?,?,?,?,?,?,?,?,?)",
                    (new_id(), keep_id, entity_id, entity["user_id"], "rejected", 0.0,
                     "undone by you", now, now))
            # the owner flag a merge moved onto the kept entity goes back
            metadata = json.loads(keep["metadata"])
            if (json.loads(entity["metadata"] or "{}").get("owner")
                    and not json.loads(snapshot["keep"]["metadata"] or "{}").get("owner")):
                metadata.pop("owner", None)
            # the kept one's type as it was, then as its mentions now give it
            self._db.execute(
                "UPDATE entities SET metadata = ?, entity_type = ?, updated_at = ?, "
                "description_updated_at = NULL WHERE id = ?",
                (json.dumps(metadata), snapshot["keep"]["entity_type"], now, keep_id))
            self._settle_types_locked([keep_id, entity_id])
            self._db.execute("DELETE FROM entity_merges WHERE id = ?", (record["id"],))
            self._commit()
        if self.names_changed is not None:
            self.names_changed([keep_id, entity_id])
        return True

    def add_proposal(self, proposal: MergeProposal) -> MergeProposal:
        """Record a pair, one row a pair: where the two already have a row,
        in either order, nothing is written and that row is returned. Every
        caller looks for the pair first (``find_proposal``), but a save and
        the weekly pass run in different threads, and two processes may share
        the file, so the look and the insert are one statement here: a pair
        found missing by both was written twice."""
        pair = (proposal.entity_a, proposal.entity_b, proposal.entity_b, proposal.entity_a)
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO entity_proposals (id, entity_a, entity_b, user_id, status, "
                "confidence, reason, created_at, decided_at, compared_step, different, "
                "belongs) SELECT ?,?,?,?,?,?,?,?,?,?,?,? WHERE NOT EXISTS ("
                "SELECT 1 FROM entity_proposals WHERE (entity_a = ? AND entity_b = ?) "
                "OR (entity_a = ? AND entity_b = ?))",
                (
                    proposal.id, proposal.entity_a, proposal.entity_b, proposal.user_id,
                    proposal.status, proposal.confidence, proposal.reason,
                    proposal.created_at, proposal.decided_at, proposal.compared_step,
                    proposal.different,
                    json.dumps(proposal.belongs) if proposal.belongs is not None else None,
                    *pair,
                ),
            )
            self._commit()
            if cur.rowcount:
                return proposal
            row = self._db.execute(
                "SELECT * FROM entity_proposals WHERE (entity_a = ? AND entity_b = ?) "
                "OR (entity_a = ? AND entity_b = ?) ORDER BY created_at, id", pair,
            ).fetchone()
        return self._row_to_proposal(row) if row else proposal

    def reopen_proposal(self, proposal_id: str, reason: str) -> MergeProposal | None:
        """Open a pair again as never compared: no answer, no decision, the
        funnel at its start, ``reason`` saying why."""
        with self._lock:
            cur = self._db.execute(
                "UPDATE entity_proposals SET status = 'proposed', confidence = 0.5, "
                "reason = ?, decided_at = NULL, compared_step = 0, different = NULL, "
                "belongs = NULL WHERE id = ? AND status != 'confirmed'",
                (reason, proposal_id),
            )
            self._commit()
        return self.get_proposal(proposal_id) if cur.rowcount else None

    def get_proposal(self, proposal_id: str) -> MergeProposal | None:
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM entity_proposals WHERE id = ?", (proposal_id,)
            ).fetchone()
        return self._row_to_proposal(row) if row else None

    def find_proposal(self, entity_a: str, entity_b: str) -> MergeProposal | None:
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM entity_proposals WHERE (entity_a = ? AND entity_b = ?) "
                "OR (entity_a = ? AND entity_b = ?)",
                (entity_a, entity_b, entity_b, entity_a),
            ).fetchone()
        return self._row_to_proposal(row) if row else None

    def list_proposals(
        self, scope: Scope, *, status: str | None = "proposed", limit: int = 100
    ) -> list[MergeProposal]:
        clauses, params = [], []
        if scope.user_id is not None:
            clauses.append("user_id = ?")
            params.append(scope.user_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        where = " AND ".join(clauses) if clauses else "1=1"
        with self._lock:
            rows = self._db.execute(
                f"SELECT * FROM entity_proposals WHERE {where} "
                "ORDER BY created_at DESC, id LIMIT ?",
                (*params, limit),
            ).fetchall()
        return [self._row_to_proposal(r) for r in rows]

    def update_proposal_judgement(
        self, proposal_id: str, *, confidence: float, reason: str | None,
        compared_step: int | None = None, different: float | None = None,
        belongs: dict[str, float] | None = None,
    ) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE entity_proposals SET confidence = ?, reason = ?, "
                "compared_step = COALESCE(?, compared_step), "
                "different = COALESCE(?, different), "
                "belongs = COALESCE(?, belongs) "
                "WHERE id = ? AND status = 'proposed'",
                (confidence, reason, compared_step, different,
                 json.dumps(belongs) if belongs is not None else None, proposal_id),
            )
            self._commit()

    def set_proposal_status(
        self, proposal_id: str, status: str, reason: str | None = None
    ) -> MergeProposal | None:
        with self._lock:
            cur = self._db.execute(
                "UPDATE entity_proposals SET status = ?, decided_at = ?, "
                "reason = COALESCE(?, reason) WHERE id = ?",
                (status, utcnow(), reason, proposal_id),
            )
            self._commit()
        return self.get_proposal(proposal_id) if cur.rowcount else None

    # -- lossless backup / restore ---------------------------------------
    @staticmethod
    def _backup_value(value: Any) -> Any:
        if isinstance(value, bytes):
            return {_BACKUP_BYTES: base64.b64encode(value).decode("ascii")}
        return value

    @staticmethod
    def _restore_value(value: Any) -> Any:
        if isinstance(value, dict) and set(value) == {_BACKUP_BYTES}:
            try:
                return base64.b64decode(value[_BACKUP_BYTES], validate=True)
            except (ValueError, TypeError) as exc:
                raise ValueError("backup contains invalid binary data") from exc
        return value

    def _select_backup_rows(
        self, table: str, where: str = "1=1", params: tuple[Any, ...] = ()
    ) -> list[dict[str, Any]]:
        keys = _BACKUP_TABLE_KEYS[table]
        rows = self._db.execute(
            f"SELECT * FROM {table} WHERE {where} ORDER BY {', '.join(keys)}", params
        ).fetchall()
        return [
            {key: self._backup_value(value) for key, value in dict(row).items()}
            for row in rows
        ]

    def _backup_rows_for_ids(
        self, table: str, column: str, values: set[str]
    ) -> list[dict[str, Any]]:
        if not values:
            return []
        placeholders = ",".join("?" * len(values))
        return self._select_backup_rows(
            table, f"{column} IN ({placeholders})", tuple(sorted(values))
        )

    def export_backup(self, scope: Scope, *, search_log: bool = False) -> dict[str, Any]:
        """Export exact source records; FTS and ANN remain derived indexes.
        The search log (what people asked) is left out unless ``search_log``:
        then the backup carries the scope's kept searches under
        "search_log", beside the tables."""
        with self._lock:
            clause, params = _scope_clause(scope)
            tables: dict[str, list[dict[str, Any]]] = {
                "episodes": self._select_backup_rows("episodes", clause, tuple(params)),
                "memories": self._select_backup_rows("memories", clause, tuple(params)),
                "topics": self._select_backup_rows("topics", clause, tuple(params)),
                "entities": self._select_backup_rows("entities", clause, tuple(params)),
            }
            memory_ids = {row["id"] for row in tables["memories"]}
            topic_ids = {row["id"] for row in tables["topics"]}
            entity_ids = {row["id"] for row in tables["entities"]}
            # the questions' texts; their vectors are derived, like the ANN
            # index, and computed again after a restore
            tables["memory_questions"] = [
                {key: value for key, value in row.items()
                 if key not in ("embedding", "embedding_model")}
                for row in (self._select_backup_rows("memory_questions") if scope.is_empty()
                            else self._backup_rows_for_ids(
                                "memory_questions", "memory_id", memory_ids))]

            if scope.is_empty():
                for table in (
                    "memory_events", "memory_topics", "entity_mentions",
                    "entity_proposals", "relations",
                ):
                    tables[table] = self._select_backup_rows(table)
            else:
                tables["memory_events"] = self._backup_rows_for_ids(
                    "memory_events", "memory_id", memory_ids
                )
                tables["memory_topics"] = [
                    row for row in self._backup_rows_for_ids(
                        "memory_topics", "memory_id", memory_ids
                    ) if row["topic_id"] in topic_ids
                ]
                tables["entity_mentions"] = [
                    row for row in self._backup_rows_for_ids(
                        "entity_mentions", "memory_id", memory_ids
                    ) if row["entity_id"] in entity_ids
                ]
                tables["entity_proposals"] = [
                    row for row in self._backup_rows_for_ids(
                        "entity_proposals", "entity_a", entity_ids
                    ) if row["entity_b"] in entity_ids
                ]
                tables["relations"] = [
                    row for row in self._backup_rows_for_ids(
                        "relations", "subject", entity_ids
                    ) if row["object"] in entity_ids
                    and (row["memory_id"] is None or row["memory_id"] in memory_ids)
                ]

            ordered = {table: tables.get(table, []) for table in _BACKUP_ORDER}
            searches = None
            if search_log:
                log_clause, log_params = _scope_clause(scope)
                searches = [dict(row) for row in self._db.execute(
                    f"SELECT * FROM search_log WHERE {log_clause} ORDER BY at, id",
                    log_params).fetchall()]
        backup = {
            "format": "memry-backup", "version": 1, "created_at": utcnow(),
            "scope": scope.model_dump(), "tables": ordered,
        }
        if searches is not None:
            backup["search_log"] = searches
        return backup

    @staticmethod
    def _backup_owner_matches(user_id: Any, owner_prefix: str | None) -> bool:
        if owner_prefix is None:
            return True
        if not isinstance(user_id, str):
            return False
        return user_id.startswith(owner_prefix) if owner_prefix.endswith("::") else user_id == owner_prefix

    def _validate_backup(
        self, backup: dict[str, Any], owner_prefix: str | None
    ) -> dict[str, list[dict[str, Any]]]:
        if backup.get("format") != "memry-backup" or backup.get("version") != 1:
            raise ValueError("unsupported Memry backup format or version")
        raw_tables = backup.get("tables")
        # Every table this schema needs must be present; anything extra is from
        # an older Memry and is ignored rather than refused.
        if isinstance(raw_tables, dict) and "memory_questions" not in raw_tables:
            raw_tables = {**raw_tables, "memory_questions": []}  # a backup from before
        if not isinstance(raw_tables, dict) or not set(_BACKUP_ORDER) <= set(raw_tables):
            raise ValueError("backup table set is incomplete or unknown")
        tables: dict[str, list[dict[str, Any]]] = {}
        for table in _BACKUP_ORDER:
            raw_rows = raw_tables[table]
            if not isinstance(raw_rows, list):
                raise ValueError(f"backup table {table} must be a list")
            info = self._db.execute(f"PRAGMA table_info({table})").fetchall()
            columns = {row["name"] for row in info}
            # A backup from before a column was added lacks it; the column's
            # default fills it in. Any other difference is refused.
            defaulted = {row["name"] for row in info if row["dflt_value"] is not None}
            rows: list[dict[str, Any]] = []
            for raw in raw_rows:
                if (not isinstance(raw, dict) or not set(raw) <= columns
                        or not columns - set(raw) <= defaulted):
                    raise ValueError(f"backup row for {table} has the wrong columns")
                row = {key: self._restore_value(value) for key, value in raw.items()}
                if table in _BACKUP_USER_TABLES and not self._backup_owner_matches(
                    row.get("user_id"), owner_prefix
                ):
                    raise ValueError(f"backup contains {table} outside this account")
                rows.append(row)
            tables[table] = rows

        memory_ids = {row["id"] for row in tables["memories"]}
        episode_ids = {row["id"] for row in tables["episodes"]}
        topic_ids = {row["id"] for row in tables["topics"]}
        entity_ids = {row["id"] for row in tables["entities"]}
        for row in tables["memories"]:
            sources = json.loads(row["source_episode_ids"])
            if not isinstance(sources, list) or not set(sources) <= episode_ids:
                raise ValueError("memory provenance references episodes outside the backup")
        for row in tables["memory_events"]:
            if row["memory_id"] not in memory_ids and owner_prefix is not None:
                raise ValueError("memory history references a memory outside the backup")
        for row in tables["memory_questions"]:
            if row["memory_id"] not in memory_ids:
                raise ValueError("question key references a memory outside the backup")
        for row in tables["memory_topics"]:
            if row["memory_id"] not in memory_ids or row["topic_id"] not in topic_ids:
                raise ValueError("topic assignment references data outside the backup")
        for row in tables["entity_mentions"]:
            if row["memory_id"] not in memory_ids or row["entity_id"] not in entity_ids:
                raise ValueError("entity link references data outside the backup")
        for row in tables["entity_proposals"]:
            if row["entity_a"] not in entity_ids or row["entity_b"] not in entity_ids:
                raise ValueError("entity merge decision references data outside the backup")
        for row in tables["relations"]:
            if row["subject"] not in entity_ids or row["object"] not in entity_ids:
                raise ValueError("entity relation references data outside the backup")
            if row["memory_id"] is not None and row["memory_id"] not in memory_ids:
                raise ValueError("entity relation evidence is outside the backup")
        return tables

    def import_backup(
        self, backup: dict[str, Any], *, owner_prefix: str | None = None
    ) -> dict[str, Any]:
        """Restore exact records transactionally; never rewrite an identity.

        Each restored memory's tags are then filed as every writer files them
        (``_file_tags_locked``): a column naming a tag merged away here is
        written as its survivor, and its index and mentions follow the column
        (a backup of one agent or run holds no topic entities, which carry
        no agent or run). A consistent backup restores unchanged."""
        with self._lock:
            tables = self._validate_backup(backup, owner_prefix)
            inserted = unchanged = 0
            by_table: dict[str, dict[str, int]] = {}
            try:
                self._db.execute("BEGIN IMMEDIATE")
                for table in _BACKUP_ORDER:
                    table_inserted = table_unchanged = 0
                    keys = _BACKUP_TABLE_KEYS[table]
                    for row in tables[table]:
                        where = " AND ".join(f"{key} = ?" for key in keys)
                        existing = self._db.execute(
                            f"SELECT * FROM {table} WHERE {where}",
                            tuple(row[key] for key in keys),
                        ).fetchone()
                        if existing is not None:
                            # Compared on the backup's columns: a backup from
                            # before a column was added does not carry it.
                            if {key: existing[key] for key in row} != row:
                                identity = ", ".join(f"{key}={row[key]!r}" for key in keys)
                                raise ValueError(f"backup conflicts with existing {table} row ({identity})")
                            table_unchanged += 1; unchanged += 1
                            continue
                        columns = tuple(row)
                        self._db.execute(
                            f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join('?' * len(columns))})",
                            tuple(row[column] for column in columns),
                        )
                        table_inserted += 1; inserted += 1
                    by_table[table] = {"inserted": table_inserted, "unchanged": table_unchanged}
                restored = [row["id"] for row in tables["memories"]]
                cache: dict[Any, Any] = {}
                for start in range(0, len(restored), 500):
                    chunk = restored[start:start + 500]
                    self._refile_locked(
                        f"id IN ({','.join('?' * len(chunk))})", tuple(chunk), cache=cache)
                self._db.execute(
                    "INSERT OR IGNORE INTO ann_keys (memory_id) SELECT id FROM memories WHERE embedding IS NOT NULL"
                )
                self._commit()
            except sqlite3.IntegrityError as exc:
                self._db.rollback()
                raise ValueError(f"backup conflicts with existing indexed data: {exc}") from exc
            except Exception:
                self._db.rollback(); raise
            self._has_metadata_aliases = self._db.execute(
                "SELECT 1 FROM entities WHERE metadata LIKE '%\"aliases\"%' LIMIT 1"
            ).fetchone() is not None
            models = self._db.execute(
                "SELECT embedding_model, MAX(length(embedding)) FROM memories "
                "WHERE embedding IS NOT NULL AND embedding_model IS NOT NULL GROUP BY embedding_model"
            ).fetchall()
        for model_id, byte_length in models:
            self.rebuild_ann(model_id, int(byte_length) // 4)
        return {
            "format": "memry-backup", "version": 1,
            "inserted": inserted, "unchanged": unchanged, "tables": by_table,
        }
    # -- maintenance --------------------------------------------------------
    def all_memories_iter(self, include_invalid: bool = True) -> list[Memory]:
        clause = "1=1" if include_invalid else "invalid_at IS NULL"
        with self._lock:
            rows = self._db.execute(
                f"SELECT {_MEMORY_COLS} FROM memories WHERE {clause}"
            ).fetchall()
        return [_row_to_memory(r) for r in rows]

    def count_memories(self, owner_prefix: str | None = None) -> dict[str, int]:
        """See ``MemoryBackend.count_memories``: the same answer, counted in SQL
        rather than by loading every memory on the server."""
        if owner_prefix is None:
            where, params = "1=1", ()
        elif owner_prefix.endswith("::"):
            # substr, not LIKE: an account name may contain % or _.
            where, params = "substr(user_id, 1, ?) = ?", (len(owner_prefix), owner_prefix)
        else:
            where, params = "user_id = ?", (owner_prefix,)
        with self._lock:
            active, invalidated, forgotten = self._db.execute(
                "SELECT "
                "COALESCE(SUM(invalid_at IS NULL), 0), "
                "COALESCE(SUM(invalid_at IS NOT NULL), 0), "
                "COALESCE(SUM(invalid_at IS NOT NULL "
                "AND (superseded_by IS NULL OR superseded_by = '')), 0) "
                f"FROM memories WHERE {where}",
                params,
            ).fetchone()
        return {"active": active, "invalidated": invalidated, "forgotten": forgotten}

    def stats(self) -> dict[str, Any]:
        with self._lock:
            active = self._db.execute(
                "SELECT COUNT(*) FROM memories WHERE invalid_at IS NULL"
            ).fetchone()[0]
            invalid = self._db.execute(
                "SELECT COUNT(*) FROM memories WHERE invalid_at IS NOT NULL"
            ).fetchone()[0]
            episodes = self._db.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
            events = self._db.execute("SELECT COUNT(*) FROM memory_events").fetchone()[0]
            pending_enrichments = self._db.execute(
                "SELECT COUNT(*) FROM memories WHERE invalid_at IS NULL "
                "AND json_extract(metadata, '$.pending_distillation') = 1"
            ).fetchone()[0]
            retrying_enrichments = self._db.execute(
                "SELECT COUNT(*) FROM memories WHERE invalid_at IS NULL "
                "AND json_extract(metadata, '$.pending_distillation') = 1 "
                "AND json_extract(metadata, '$._enrichment.status') = 'retry'"
            ).fetchone()[0]
            by_type = dict(
                self._db.execute(
                    "SELECT memory_type, COUNT(*) FROM memories "
                    "WHERE invalid_at IS NULL GROUP BY memory_type"
                ).fetchall()
            )
            users = [
                r[0]
                for r in self._db.execute(
                    "SELECT DISTINCT user_id FROM memories WHERE user_id IS NOT NULL"
                ).fetchall()
            ]
        with self._lock:
            entities = self._db.execute(
                "SELECT COUNT(*) FROM entities WHERE merged_into IS NULL "
                f"AND {_kind_clause('named')}"
            ).fetchone()[0]
            topics = self._db.execute(
                "SELECT COUNT(*) FROM entities WHERE merged_into IS NULL "
                f"AND {_kind_clause('topic')}"
            ).fetchone()[0]
            proposals = self._db.execute(
                "SELECT COUNT(*) FROM entity_proposals WHERE status = 'proposed'"
            ).fetchone()[0]
        return {
            "backend": "local",
            "db_path": self.db_path,
            "active_memories": active,
            "invalidated_memories": invalid,
            "episodes": episodes,
            "events": events,
            "pending_enrichments": pending_enrichments,
            "retrying_enrichments": retrying_enrichments,
            "memories_by_type": by_type,
            "users": users,
            "entities": entities,
            "topics": topics,
            "open_merge_proposals": proposals,
            "ann": {
                "available": HAS_USEARCH,
                "active": any(
                    s.size >= self._ann_cfg.min_rows for s in self._anns.values()
                ),
                "indexed": sum(s.size for s in self._anns.values()),
            },
        }

    def reset(self, *, keep_meta: Iterable[str] = ()) -> None:
        """Empty every table of the schema, read from the schema: a list kept
        by hand missed each table added after it. Of ``meta`` the migration
        markers stay (``schema:``) and the keys starting with one of
        ``keep_meta``; the rest is a namespace's state (queues, the owner,
        when each pass ran) and goes with its memories."""
        kept = ("schema:", *keep_meta)
        with self._lock:
            for table in _TABLES:
                if table != "meta":
                    self._db.execute(f"DELETE FROM {table}")
            for row in self._db.execute("SELECT key FROM meta").fetchall():
                if not row["key"].startswith(kept):
                    self._db.execute("DELETE FROM meta WHERE key = ?", (row["key"],))
            self._commit()
        for sidecar in self._anns.values():
            sidecar.rebuild([])

    def close(self) -> None:
        for sidecar in self._anns.values():
            sidecar.save()
        with self._lock:
            self._db.close()
