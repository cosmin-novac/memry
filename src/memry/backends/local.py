"""Local SQLite backend - the default, zero-service storage engine.

One file holds everything: raw episodes, derived memories, an FTS5 index
(BM25 keyword search), float32 embeddings (brute-force cosine via numpy -
fast enough into the hundreds of thousands of memories), and the full event
history. WAL mode + a process-wide lock make it safe for the MCP/REST servers.
"""

from __future__ import annotations

import base64
import json
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any

import numpy as np

from ..config import AnnConfig
from ..models import (
    Collection,
    Entity,
    EntityMention,
    Episode,
    Memory,
    MemoryEvent,
    MergeProposal,
    Relation,
    Scope,
    SyntheticTag,
    Topic,
    TopicRelation,
    utcnow,
)
from .ann import HAS_USEARCH, HnswSidecar
from .base import MemoryBackend

_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'user',
    user_id TEXT,
    agent_id TEXT,
    run_id TEXT,
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_episodes_scope ON episodes(user_id, agent_id, run_id);

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
CREATE INDEX IF NOT EXISTS idx_memories_invalid ON memories(invalid_at);
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
CREATE UNIQUE INDEX IF NOT EXISTS idx_topics_scope_norm ON topics(
    IFNULL(user_id, ''), IFNULL(agent_id, ''), IFNULL(run_id, ''), normalized
);
CREATE TABLE IF NOT EXISTS memory_topics (
    memory_id TEXT NOT NULL,
    topic_id TEXT NOT NULL,
    PRIMARY KEY (memory_id, topic_id)
);
CREATE INDEX IF NOT EXISTS idx_memory_topics_topic ON memory_topics(topic_id, memory_id);
CREATE INDEX IF NOT EXISTS idx_memory_topics_memory ON memory_topics(memory_id, topic_id);
CREATE TABLE IF NOT EXISTS topic_relations (
    id TEXT PRIMARY KEY,
    broader_topic_id TEXT NOT NULL,
    narrower_topic_id TEXT NOT NULL,
    user_id TEXT,
    provenance TEXT NOT NULL DEFAULT 'synthetic',
    created_at TEXT NOT NULL,
    UNIQUE (broader_topic_id, narrower_topic_id)
);
CREATE INDEX IF NOT EXISTS idx_topic_relations_broader
    ON topic_relations(broader_topic_id, narrower_topic_id);
CREATE INDEX IF NOT EXISTS idx_topic_relations_narrower
    ON topic_relations(narrower_topic_id, broader_topic_id);
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
    created_at TEXT NOT NULL
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

CREATE TABLE IF NOT EXISTS entity_mentions (
    id TEXT PRIMARY KEY,
    entity_id TEXT NOT NULL,
    memory_id TEXT NOT NULL,
    surface TEXT NOT NULL,
    created_at TEXT NOT NULL
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
    decided_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_proposals_status ON entity_proposals(status, user_id);

CREATE TABLE IF NOT EXISTS ann_keys (
    key INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS synthetic_tags (
    id          TEXT PRIMARY KEY,
    tag         TEXT NOT NULL,
    user_id     TEXT,
    source_tags TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_synthetic_tags_user ON synthetic_tags(user_id);
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
CREATE TABLE IF NOT EXISTS collections (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    summary     TEXT NOT NULL DEFAULT '',
    memory_ids  TEXT NOT NULL,
    user_id     TEXT,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_collections_user ON collections(user_id);
"""

_MEMORY_COLS = (
    "id, content, memory_type, user_id, agent_id, run_id, importance, categories, "
    "entities, metadata, created_at, updated_at, valid_from, invalid_at, superseded_by, "
    "source_episode_ids, embedding_model"
)

_BACKUP_TABLE_KEYS: dict[str, tuple[str, ...]] = {
    "episodes": ("id",),
    "memories": ("id",),
    "memory_events": ("id",),
    "topics": ("id",),
    "memory_topics": ("memory_id", "topic_id"),
    "topic_relations": ("id",),
    "entities": ("id",),
    "entity_mentions": ("id",),
    "entity_proposals": ("id",),
    "synthetic_tags": ("id",),
    "relations": ("id",),
    "collections": ("id",),
}
_BACKUP_ORDER = tuple(_BACKUP_TABLE_KEYS)
_BACKUP_USER_TABLES = {
    "episodes", "memories", "topics", "topic_relations", "entities",
    "entity_proposals", "synthetic_tags", "relations", "collections",
}
_BACKUP_BYTES = "__memry_base64__"

_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _scope_clause(scope: Scope, prefix: str = "") -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    for field in ("user_id", "agent_id", "run_id"):
        value = getattr(scope, field)
        if value is not None:
            clauses.append(f"{prefix}{field} = ?")
            params.append(value)
    return (" AND ".join(clauses) if clauses else "1=1"), params


def _category_clause(categories: list[str] | None, memory_id: str) -> tuple[str, list[Any]]:
    if not categories:
        return "1=1", []
    normalized = [c.strip().lower() for c in categories if c.strip()]
    if not normalized:
        return "1=1", []
    placeholders = ",".join("?" * len(normalized))
    return (
        "EXISTS (WITH RECURSIVE descendants(topic_id, depth) AS ("
        f"SELECT id, 0 FROM topics WHERE normalized IN ({placeholders}) "
        "UNION SELECT tr.narrower_topic_id, d.depth + 1 FROM topic_relations tr "
        "JOIN descendants d ON tr.broader_topic_id = d.topic_id "
        "WHERE d.depth < 8) "
        "SELECT 1 FROM memory_topics mt JOIN descendants d ON d.topic_id = mt.topic_id "
        f"WHERE mt.memory_id = {memory_id})",
        normalized,
    )


def _entity_clause(entity_id: str | None, memory_id: str) -> tuple[str, list[Any]]:
    if not entity_id:
        return "1=1", []
    return (
        "EXISTS (SELECT 1 FROM entity_mentions em "
        f"WHERE em.memory_id = {memory_id} AND em.entity_id = ?)",
        [entity_id],
    )


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
    def __init__(self, db_path: str = ":memory:", ann: AnnConfig | None = None) -> None:
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.executescript(_SCHEMA)
        self._ensure_entity_description_columns()
        self._backfill_topics()
        self._migrate_synthetic_topic_relations()
        self._db.commit()
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

    def _sync_memory_topics_locked(
        self,
        memory_id: str,
        categories: list[str],
        scope: Scope,
        provenance: str = "memory",
    ) -> None:
        self._db.execute("DELETE FROM memory_topics WHERE memory_id = ?", (memory_id,))
        seen: set[str] = set()
        for raw in categories:
            name = str(raw).strip()
            normalized = name.lower()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            topic = self._topic_locked(name, scope, provenance)
            self._db.execute(
                "INSERT OR IGNORE INTO memory_topics (memory_id, topic_id) VALUES (?,?)",
                (memory_id, topic.id),
            )

    def _backfill_topics(self) -> None:
        marker = self._db.execute(
            "SELECT value FROM meta WHERE key = 'schema:topics:v1'"
        ).fetchone()
        if marker:
            return
        rows = self._db.execute(
            "SELECT id, categories, user_id, agent_id, run_id FROM memories"
        ).fetchall()
        for row in rows:
            self._sync_memory_topics_locked(
                row["id"],
                json.loads(row["categories"]),
                Scope(
                    user_id=row["user_id"], agent_id=row["agent_id"], run_id=row["run_id"]
                ),
            )
        self._db.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema:topics:v1', ?)",
            (utcnow(),),
        )

    def _migrate_synthetic_topic_relations(self) -> None:
        """Turn legacy copied umbrella tags into hierarchy edges exactly once."""
        marker = self._db.execute(
            "SELECT value FROM meta WHERE key = 'schema:topic-relations:v1'"
        ).fetchone()
        if marker:
            return
        rows = self._db.execute(
            "SELECT tag, user_id, source_tags FROM synthetic_tags"
        ).fetchall()
        for row in rows:
            tag = str(row["tag"]).strip()
            sources = [
                str(value).strip() for value in json.loads(row["source_tags"])
                if str(value).strip()
            ]
            if not tag or not sources:
                continue
            scope = Scope(user_id=row["user_id"])
            parent = self._topic_locked(tag, scope, provenance="synthetic")
            for source in sources:
                matches = self._db.execute(
                    "SELECT * FROM topics WHERE normalized = ? AND user_id IS ?",
                    (source.lower(), row["user_id"]),
                ).fetchall()
                if not matches:
                    matches = [
                        self._topic_locked(source, scope, provenance="memory").model_dump()
                    ]
                for match in matches:
                    child_id = match["id"]
                    if child_id == parent.id:
                        continue
                    relation = TopicRelation(
                        broader_topic_id=parent.id,
                        narrower_topic_id=child_id,
                        user_id=row["user_id"],
                        provenance="synthetic-migration",
                    )
                    self._db.execute(
                        "INSERT OR IGNORE INTO topic_relations "
                        "(id, broader_topic_id, narrower_topic_id, user_id, provenance, created_at) "
                        "VALUES (?,?,?,?,?,?)",
                        (
                            relation.id, relation.broader_topic_id,
                            relation.narrower_topic_id, relation.user_id,
                            relation.provenance, relation.created_at,
                        ),
                    )
            source_set = {source.lower() for source in sources}
            memories = self._db.execute(
                "SELECT id, categories, user_id, agent_id, run_id FROM memories "
                "WHERE user_id IS ?",
                (row["user_id"],),
            ).fetchall()
            for memory in memories:
                categories = json.loads(memory["categories"])
                normalized = {str(value).strip().lower() for value in categories}
                if tag.lower() not in normalized or not normalized.intersection(source_set):
                    continue
                kept = [
                    value for value in categories
                    if str(value).strip().lower() != tag.lower()
                ]
                self._db.execute(
                    "UPDATE memories SET categories = ? WHERE id = ?",
                    (json.dumps(kept), memory["id"]),
                )
                self._sync_memory_topics_locked(
                    memory["id"], kept,
                    Scope(
                        user_id=memory["user_id"], agent_id=memory["agent_id"],
                        run_id=memory["run_id"],
                    ),
                )
        self._db.execute(
            "INSERT OR REPLACE INTO meta (key, value) "
            "VALUES ('schema:topic-relations:v1', ?)",
            (utcnow(),),
        )

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
            self._db.commit()
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
                "INSERT INTO episodes (id, content, role, user_id, agent_id, run_id, "
                "metadata, created_at) VALUES (?,?,?,?,?,?,?,?)",
                [
                    (
                        e.id,
                        e.content,
                        e.role,
                        e.user_id,
                        e.agent_id,
                        e.run_id,
                        json.dumps(e.metadata),
                        e.created_at,
                    )
                    for e in episodes
                ],
            )
            self._db.commit()

    def list_episodes(self, scope: Scope, limit: int = 100) -> list[Episode]:
        clause, params = _scope_clause(scope)
        with self._lock:
            rows = self._db.execute(
                f"SELECT * FROM episodes WHERE {clause} ORDER BY created_at DESC LIMIT ?",
                (*params, limit),
            ).fetchall()
        return [
            Episode(
                id=r["id"],
                content=r["content"],
                role=r["role"],
                user_id=r["user_id"],
                agent_id=r["agent_id"],
                run_id=r["run_id"],
                metadata=json.loads(r["metadata"]),
                created_at=r["created_at"],
            )
            for r in rows
        ]

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
            self._sync_memory_topics_locked(memory.id, memory.categories, memory.scope())
            if embedding and memory.embedding_model:
                self._ann_add(memory.id, embedding, memory.embedding_model)
            self._db.commit()
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
            if cur.rowcount and categories is not None:
                row = self._db.execute(
                    "SELECT user_id, agent_id, run_id FROM memories WHERE id = ?",
                    (memory_id,),
                ).fetchone()
                self._sync_memory_topics_locked(
                    memory_id,
                    categories,
                    Scope(
                        user_id=row["user_id"], agent_id=row["agent_id"], run_id=row["run_id"]
                    ),
                )
            if cur.rowcount and mentions is not None:
                old_entity_ids = {
                    row["entity_id"] for row in self._db.execute(
                        "SELECT DISTINCT entity_id FROM entity_mentions WHERE memory_id = ?",
                        (memory_id,),
                    ).fetchall()
                }
                self._db.execute(
                    "DELETE FROM entity_mentions WHERE memory_id = ?", (memory_id,)
                )
                self._db.executemany(
                    "INSERT INTO entity_mentions "
                    "(id, entity_id, memory_id, surface, created_at) VALUES (?,?,?,?,?)",
                    [
                        (m.id, m.entity_id, m.memory_id, m.surface, m.created_at)
                        for m in mentions
                    ],
                )
                affected = old_entity_ids | {m.entity_id for m in mentions}
                if affected:
                    placeholders = ",".join("?" * len(affected))
                    self._db.execute(
                        "UPDATE entities SET updated_at = ?, description_updated_at = NULL "
                        f"WHERE id IN ({placeholders})",
                        (utcnow(), *sorted(affected)),
                    )
            if cur.rowcount and embedding is not None and embedding_model is not None:
                self._ann_add(memory_id, embedding, embedding_model)
            if cur.rowcount and touch and mentions is None:
                self._db.execute(
                    "UPDATE entities SET updated_at = ?, description_updated_at = NULL "
                    "WHERE id IN (SELECT entity_id FROM entity_mentions WHERE memory_id = ?)",
                    (utcnow(), memory_id),
                )
            self._db.commit()
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
                f"{due_clause}ORDER BY created_at LIMIT ?",
                (*params, limit),
            ).fetchall()
        return [_row_to_memory(row) for row in rows]

    def set_memory_timestamp(self, memory_id: str, updated_at: str) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE memories SET updated_at = ? WHERE id = ?", (updated_at, memory_id)
            )
            self._db.commit()

    def invalidate_memory(
        self, memory_id: str, *, superseded_by: str | None = None
    ) -> Memory | None:
        from ..models import utcnow

        with self._lock:
            cur = self._db.execute(
                "UPDATE memories SET invalid_at = ?, superseded_by = ?, updated_at = ? "
                "WHERE id = ? AND invalid_at IS NULL",
                (utcnow(), superseded_by, utcnow(), memory_id),
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
                    (changed_at, memory_id),
                )
            self._db.commit()
        if cur.rowcount == 0:
            return None
        return self.get_memory(memory_id)

    def delete_memory(self, memory_id: str) -> bool:
        with self._lock:
            entity_ids = [
                row["entity_id"]
                for row in self._db.execute(
                    "SELECT DISTINCT entity_id FROM entity_mentions WHERE memory_id = ?",
                    (memory_id,),
                ).fetchall()
            ]
            cur = self._db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            if cur.rowcount:
                self._ann_remove(memory_id)
                self._db.execute("DELETE FROM ann_keys WHERE memory_id = ?", (memory_id,))
                self._db.execute("DELETE FROM entity_mentions WHERE memory_id = ?", (memory_id,))
                self._db.execute("DELETE FROM memory_topics WHERE memory_id = ?", (memory_id,))
                self._db.execute("DELETE FROM relations WHERE memory_id = ?", (memory_id,))
                if entity_ids:
                    placeholders = ",".join("?" * len(entity_ids))
                    self._db.execute(
                        f"UPDATE entities SET updated_at = ?, description_updated_at = NULL "
                        f"WHERE id IN ({placeholders})",
                        (utcnow(), *entity_ids),
                    )
            self._db.commit()
        return cur.rowcount > 0

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
                f"AND {entity_clause} ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                (*params, *cat_params, *entity_params, limit, offset),
            ).fetchall()
        return [_row_to_memory(r) for r in rows]

    # -- search -----------------------------------------------------------
    def _score_rows(
        self, rows: list[sqlite3.Row], embedding: list[float], limit: int
    ) -> list[tuple[Memory, float]]:
        if not rows:
            return []
        query = np.asarray(embedding, dtype=np.float32)
        qnorm = np.linalg.norm(query)
        if qnorm == 0:
            return []
        mats = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
        norms = np.linalg.norm(mats, axis=1)
        norms[norms == 0] = 1e-9
        sims = (mats @ query) / (norms * qnorm)
        order = np.argsort(-sims)[:limit]
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
    ) -> list[tuple[Memory, float]]:
        clause, params = _scope_clause(scope)
        cat_clause, cat_params = _category_clause(categories, "memories.id")
        entity_clause, entity_params = _entity_clause(entity_id, "memories.id")
        if not include_invalid:
            clause += " AND invalid_at IS NULL"

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
                        f"AND {clause} AND {cat_clause} AND {entity_clause} "
                        "AND embedding IS NOT NULL AND embedding_model = ?",
                        (*keys, *params, *cat_params, *entity_params, embedding_model),
                    ).fetchall()
                if len(rows) >= limit:
                    return self._score_rows(rows, embedding, limit)
                # restrictive filters starved the ANN candidates -> exact scan

        with self._lock:
            rows = self._db.execute(
                f"SELECT {_MEMORY_COLS}, embedding FROM memories "
                f"WHERE {clause} AND {cat_clause} AND {entity_clause} "
                "AND embedding IS NOT NULL AND embedding_model = ?",
                (*params, *cat_params, *entity_params, embedding_model),
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
    ) -> list[tuple[Memory, float]]:
        tokens = _WORD_RE.findall(query)
        if not tokens:
            return []
        match = " OR ".join(f'"{t}"' for t in tokens[:32])
        clause, params = _scope_clause(scope, prefix="m.")
        cat_clause, cat_params = _category_clause(categories, "m.id")
        entity_clause, entity_params = _entity_clause(entity_id, "m.id")
        if not include_invalid:
            clause += " AND m.invalid_at IS NULL"
        sql = (
            f"SELECT {', '.join('m.' + c.strip() for c in _MEMORY_COLS.split(','))}, "
            "bm25(memories_fts) AS rank_score "
            "FROM memories_fts JOIN memories m ON m.rowid = memories_fts.rowid "
            f"WHERE memories_fts MATCH ? AND {clause} AND {cat_clause} "
            f"AND {entity_clause} ORDER BY rank_score LIMIT ?"
        )
        with self._lock:
            rows = self._db.execute(
                sql, (match, *params, *cat_params, *entity_params, limit)
            ).fetchall()
        # bm25() returns lower-is-better (negative); flip to higher-is-better.
        return [(_row_to_memory(r), -float(r["rank_score"])) for r in rows]

    # -- events -----------------------------------------------------------
    def add_event(self, event: MemoryEvent) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO memory_events (id, memory_id, event, old_content, new_content, "
                "reason, actor, created_at) VALUES (?,?,?,?,?,?,?,?)",
                (
                    event.id,
                    event.memory_id,
                    event.event,
                    event.old_content,
                    event.new_content,
                    event.reason,
                    event.actor,
                    event.created_at,
                ),
            )
            self._db.commit()

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
            )
            for r in rows
        ]

    # -- normalized topics -------------------------------------------------
    def upsert_topic(self, topic: Topic) -> Topic:
        scope = Scope(user_id=topic.user_id, agent_id=topic.agent_id, run_id=topic.run_id)
        with self._lock:
            stored = self._topic_locked(topic.name, scope, topic.provenance)
            self._db.commit()
        return stored

    def list_topics(self, scope: Scope, *, limit: int = 1000) -> list[Topic]:
        clause, params = _scope_clause(scope)
        with self._lock:
            rows = self._db.execute(
                f"SELECT * FROM topics WHERE {clause} ORDER BY normalized LIMIT ?",
                (*params, limit),
            ).fetchall()
        return [self._row_to_topic(row) for row in rows]

    def topic_counts(self, scope: Scope) -> list[dict[str, Any]]:
        root_clause, root_params = _scope_clause(scope, prefix="root.")
        memory_clause, memory_params = _scope_clause(scope, prefix="m.")
        with self._lock:
            rows = self._db.execute(
                "WITH RECURSIVE descendants(root_id, topic_id, depth) AS ("
                "SELECT id, id, 0 FROM topics UNION "
                "SELECT d.root_id, tr.narrower_topic_id, d.depth + 1 "
                "FROM descendants d JOIN topic_relations tr "
                "ON tr.broader_topic_id = d.topic_id WHERE d.depth < 8) "
                "SELECT root.normalized AS category, "
                "COUNT(DISTINCT mt.memory_id) AS count FROM topics root "
                "JOIN descendants d ON d.root_id = root.id "
                "JOIN memory_topics mt ON mt.topic_id = d.topic_id "
                "JOIN memories m ON m.id = mt.memory_id "
                f"WHERE m.invalid_at IS NULL AND {root_clause} AND {memory_clause} "
                "GROUP BY root.normalized HAVING count > 0 "
                "ORDER BY count DESC, category",
                (*root_params, *memory_params),
            ).fetchall()
        return [{"category": row["category"], "count": row["count"]} for row in rows]

    def topic_memory_ids(self, scope: Scope) -> list[tuple[str, str]]:
        topic_clause, topic_params = _scope_clause(scope, prefix="t.")
        memory_clause, memory_params = _scope_clause(scope, prefix="m.")
        with self._lock:
            rows = self._db.execute(
                "SELECT t.normalized AS category, mt.memory_id AS memory_id "
                "FROM topics t JOIN memory_topics mt ON mt.topic_id = t.id "
                "JOIN memories m ON m.id = mt.memory_id "
                f"WHERE m.invalid_at IS NULL AND {topic_clause} AND {memory_clause}",
                (*topic_params, *memory_params),
            ).fetchall()
        return [(row["category"], row["memory_id"]) for row in rows]

    def direct_topic_counts(self, scope: Scope) -> list[dict[str, Any]]:
        """Counts for directly-attached topics only, without descendant rollup."""
        topic_clause, topic_params = _scope_clause(scope, prefix="t.")
        memory_clause, memory_params = _scope_clause(scope, prefix="m.")
        with self._lock:
            rows = self._db.execute(
                "SELECT t.normalized AS category, "
                "COUNT(DISTINCT mt.memory_id) AS count FROM topics t "
                "JOIN memory_topics mt ON mt.topic_id = t.id "
                "JOIN memories m ON m.id = mt.memory_id "
                f"WHERE m.invalid_at IS NULL AND {topic_clause} AND {memory_clause} "
                "GROUP BY t.normalized HAVING count > 0 "
                "ORDER BY count DESC, category",
                (*topic_params, *memory_params),
            ).fetchall()
        return [{"category": row["category"], "count": row["count"]} for row in rows]

    def retag_topics(self, scope: Scope, remove: set[str], add: str | None) -> int:
        normalized = {item.strip().lower() for item in remove if item.strip()}
        if not normalized:
            return 0
        add = add.strip().lower() if add and add.strip() else None
        placeholders = ",".join("?" * len(normalized))
        topic_clause, topic_params = _scope_clause(scope, prefix="t.")
        memory_clause, memory_params = _scope_clause(scope, prefix="m.")
        with self._lock:
            topic_rows = self._db.execute(
                f"SELECT t.* FROM topics t WHERE {topic_clause} "
                f"AND t.normalized IN ({placeholders})",
                (*topic_params, *sorted(normalized)),
            ).fetchall()
            old_ids = {row["id"] for row in topic_rows}
            target_ids: dict[str, str] = {}
            if add:
                for row in topic_rows:
                    target = self._topic_locked(
                        add,
                        Scope(
                            user_id=row["user_id"], agent_id=row["agent_id"],
                            run_id=row["run_id"],
                        ),
                        provenance="user",
                    )
                    target_ids[row["id"]] = target.id

            rows = self._db.execute(
                "SELECT DISTINCT m.id, m.categories, m.user_id, m.agent_id, m.run_id "
                "FROM memories m JOIN memory_topics mt ON mt.memory_id = m.id "
                "JOIN topics t ON t.id = mt.topic_id "
                f"WHERE m.invalid_at IS NULL AND {memory_clause} "
                f"AND t.normalized IN ({placeholders})",
                (*memory_params, *sorted(normalized)),
            ).fetchall()
            changed = 0
            for row in rows:
                categories = json.loads(row["categories"])
                kept = [
                    item for item in categories
                    if str(item).strip().lower() not in normalized
                ]
                if add and add not in {str(item).strip().lower() for item in kept}:
                    kept.append(add)
                if kept == categories:
                    continue
                self._db.execute(
                    "UPDATE memories SET categories = ? WHERE id = ?",
                    (json.dumps(kept), row["id"]),
                )
                self._sync_memory_topics_locked(
                    row["id"], kept,
                    Scope(
                        user_id=row["user_id"], agent_id=row["agent_id"],
                        run_id=row["run_id"],
                    ),
                    provenance="user",
                )
                changed += 1

            if old_ids:
                edge_placeholders = ",".join("?" * len(old_ids))
                edges = self._db.execute(
                    "SELECT * FROM topic_relations "
                    f"WHERE broader_topic_id IN ({edge_placeholders}) "
                    f"OR narrower_topic_id IN ({edge_placeholders})",
                    (*old_ids, *old_ids),
                ).fetchall()
                for edge in edges:
                    self._db.execute(
                        "DELETE FROM topic_relations WHERE id = ?", (edge["id"],)
                    )
                    if not add:
                        continue
                    broader = target_ids.get(edge["broader_topic_id"], edge["broader_topic_id"])
                    narrower = target_ids.get(edge["narrower_topic_id"], edge["narrower_topic_id"])
                    if broader == narrower:
                        continue
                    rewritten = TopicRelation(
                        broader_topic_id=broader,
                        narrower_topic_id=narrower,
                        user_id=edge["user_id"],
                        provenance=edge["provenance"],
                        created_at=edge["created_at"],
                    )
                    self._db.execute(
                        "INSERT OR IGNORE INTO topic_relations "
                        "(id, broader_topic_id, narrower_topic_id, user_id, provenance, created_at) "
                        "VALUES (?,?,?,?,?,?)",
                        (
                            rewritten.id, rewritten.broader_topic_id,
                            rewritten.narrower_topic_id, rewritten.user_id,
                            rewritten.provenance, rewritten.created_at,
                        ),
                    )

            synthetic_rows = self._db.execute(
                "SELECT * FROM synthetic_tags WHERE user_id IS ?", (scope.user_id,)
            ).fetchall()
            for synthetic in synthetic_rows:
                tag = str(synthetic["tag"]).strip().lower()
                sources = [
                    str(value).strip().lower()
                    for value in json.loads(synthetic["source_tags"])
                    if str(value).strip()
                ]
                if tag in normalized:
                    self._db.execute(
                        "DELETE FROM synthetic_tags WHERE id = ?", (synthetic["id"],)
                    )
                    continue
                rewritten_sources = [
                    add if source in normalized and add else source
                    for source in sources
                    if source not in normalized or add
                ]
                rewritten_sources = list(dict.fromkeys(rewritten_sources))
                self._db.execute(
                    "UPDATE synthetic_tags SET tag = ?, source_tags = ? WHERE id = ?",
                    (tag, json.dumps(rewritten_sources), synthetic["id"]),
                )

            for old_id in old_ids:
                if target_ids.get(old_id) == old_id:
                    continue
                self._db.execute(
                    "DELETE FROM topics WHERE id = ? "
                    "AND NOT EXISTS (SELECT 1 FROM memory_topics WHERE topic_id = ?) "
                    "AND NOT EXISTS (SELECT 1 FROM topic_relations "
                    "WHERE broader_topic_id = ? OR narrower_topic_id = ?)",
                    (old_id, old_id, old_id, old_id),
                )
            self._db.commit()
        return changed

    def add_topic_relation(self, relation: TopicRelation) -> TopicRelation:
        with self._lock:
            self._db.execute(
                "INSERT OR IGNORE INTO topic_relations "
                "(id, broader_topic_id, narrower_topic_id, user_id, provenance, created_at) "
                "VALUES (?,?,?,?,?,?)",
                (
                    relation.id, relation.broader_topic_id, relation.narrower_topic_id,
                    relation.user_id, relation.provenance, relation.created_at,
                ),
            )
            self._db.commit()
        return relation

    def list_topic_relations(self, scope: Scope) -> list[TopicRelation]:
        clause, params = _scope_clause(Scope(user_id=scope.user_id))
        with self._lock:
            rows = self._db.execute(
                f"SELECT * FROM topic_relations WHERE {clause} ORDER BY created_at",
                params,
            ).fetchall()
        return [
            TopicRelation(
                id=row["id"], broader_topic_id=row["broader_topic_id"],
                narrower_topic_id=row["narrower_topic_id"], user_id=row["user_id"],
                provenance=row["provenance"], created_at=row["created_at"],
            )
            for row in rows
        ]

    # -- synthetic tags + meta ---------------------------------------------
    def record_synthetic_tag(self, tag: SyntheticTag) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO synthetic_tags "
                "(id, tag, user_id, source_tags, created_at) VALUES (?,?,?,?,?)",
                (tag.id, tag.tag, tag.user_id, json.dumps(tag.source_tags),
                 tag.created_at),
            )
            self._db.commit()

    def list_synthetic_tags(self, scope: Scope) -> list[SyntheticTag]:
        clause, params = _scope_clause(Scope(user_id=scope.user_id))
        with self._lock:
            rows = self._db.execute(
                f"SELECT * FROM synthetic_tags WHERE {clause} ORDER BY created_at DESC",
                params,
            ).fetchall()
        return [
            SyntheticTag(
                id=r["id"], tag=r["tag"], user_id=r["user_id"],
                source_tags=json.loads(r["source_tags"]), created_at=r["created_at"],
            )
            for r in rows
        ]

    def delete_synthetic_tag(self, scope: Scope, tag: str) -> None:
        clause, params = _scope_clause(Scope(user_id=scope.user_id))
        with self._lock:
            self._db.execute(
                f"DELETE FROM synthetic_tags WHERE {clause} AND lower(tag) = ?",
                (*params, tag.strip().lower()),
            )
            self._db.commit()

    def distinct_user_ids(self) -> list[str | None]:
        with self._lock:
            rows = self._db.execute(
                "SELECT DISTINCT user_id FROM memories"
            ).fetchall()
        return [r["user_id"] for r in rows]

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
            self._db.commit()

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
            self._db.commit()
        return relation

    def list_relations(self, scope: Scope, *, limit: int = 1000) -> list[Relation]:
        clause, params = _scope_clause(Scope(user_id=scope.user_id))
        with self._lock:
            rows = self._db.execute(
                f"SELECT * FROM relations WHERE {clause} AND invalid_at IS NULL "
                "ORDER BY created_at DESC LIMIT ?",
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
                f"(subject IN ({placeholders}) OR object IN ({placeholders}))",
                (*entity_ids, *entity_ids),
            ).fetchall()
        return [self._row_to_relation(r) for r in rows]

    # -- collections + vectors ---------------------------------------------
    def memory_vectors(self, scope: Scope, *, limit: int = 5000):
        clause, params = _scope_clause(scope)
        with self._lock:
            rows = self._db.execute(
                f"SELECT id, embedding FROM memories WHERE {clause} "
                "AND embedding IS NOT NULL AND invalid_at IS NULL "
                "ORDER BY updated_at DESC LIMIT ?",
                (*params, limit),
            ).fetchall()
        return [
            (r["id"], np.frombuffer(r["embedding"], dtype=np.float32)) for r in rows
        ]

    def record_collection(self, collection: Collection) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO collections (id, title, summary, memory_ids, "
                "user_id, created_at) VALUES (?,?,?,?,?,?)",
                (collection.id, collection.title, collection.summary,
                 json.dumps(collection.memory_ids), collection.user_id,
                 collection.created_at),
            )
            self._db.commit()

    def list_collections(self, scope: Scope) -> list[Collection]:
        clause, params = _scope_clause(Scope(user_id=scope.user_id))
        with self._lock:
            rows = self._db.execute(
                f"SELECT * FROM collections WHERE {clause} ORDER BY created_at DESC",
                params,
            ).fetchall()
        return [
            Collection(id=r["id"], title=r["title"], summary=r["summary"],
                       memory_ids=json.loads(r["memory_ids"]), user_id=r["user_id"],
                       created_at=r["created_at"])
            for r in rows
        ]

    def clear_collections(self, scope: Scope) -> int:
        clause, params = _scope_clause(Scope(user_id=scope.user_id))
        with self._lock:
            cur = self._db.execute(f"DELETE FROM collections WHERE {clause}", params)
            self._db.commit()
        return cur.rowcount

    # -- entities -----------------------------------------------------------
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
            self._db.commit()
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
                f"AND {clause} ORDER BY updated_at DESC",
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
            f"AND {scope_clause} ORDER BY e.updated_at DESC LIMIT ?"
        )
        with self._lock:
            rows = self._db.execute(
                indexed_sql,
                (*names, *names, *names, *scope_params, limit),
            ).fetchall()
            # User aliases intentionally remain metadata until measurements earn
            # a separate table. Pay the JSON scan only when indexed identity
            # evidence found nothing and this database actually has such aliases.
            if not rows and self._has_metadata_aliases:
                rows = self._db.execute(
                    "SELECT e.* FROM entities e "
                    "WHERE e.merged_into IS NULL "
                    f"AND {scope_clause} AND EXISTS ("
                    "SELECT 1 FROM json_each(e.metadata, '$.aliases') alias "
                    f"WHERE lower(trim(CAST(alias.value AS TEXT))) IN ({placeholders})"
                    ") ORDER BY e.updated_at DESC LIMIT ?",
                    (*scope_params, *names, limit),
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
            if display.lower() not in known:
                aliases.append(display)
                metadata["aliases"] = aliases
                self._db.execute(
                    "UPDATE entities SET metadata = ?, updated_at = ?, "
                    "description_updated_at = NULL WHERE id = ?",
                    (json.dumps(metadata), utcnow(), entity_id),
                )
                self._has_metadata_aliases = True
                self._db.commit()
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
            self._db.commit()
        return self.get_entity(entity_id) if cur.rowcount else None

    def entity_evidence_updated_at(self, entity_id: str) -> str | None:
        with self._lock:
            row = self._db.execute(
                "SELECT updated_at FROM entities WHERE id = ? AND merged_into IS NULL",
                (entity_id,),
            ).fetchone()
        return row["updated_at"] if row else None

    def list_entities(
        self, scope: Scope, *, include_merged: bool = False, limit: int = 100
    ) -> list[Entity]:
        clause, params = _scope_clause(scope)
        if not include_merged:
            clause += " AND merged_into IS NULL"
        with self._lock:
            rows = self._db.execute(
                f"SELECT * FROM entities WHERE {clause} ORDER BY updated_at DESC LIMIT ?",
                (*params, limit),
            ).fetchall()
        return [self._row_to_entity(r) for r in rows]

    def add_mention(self, mention: EntityMention) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO entity_mentions (id, entity_id, memory_id, surface, created_at) "
                "VALUES (?,?,?,?,?)",
                (mention.id, mention.entity_id, mention.memory_id, mention.surface,
                 mention.created_at),
            )
            self._db.execute(
                "UPDATE entities SET updated_at = ?, description_updated_at = NULL "
                "WHERE id = ?",
                (mention.created_at, mention.entity_id),
            )
            self._db.commit()

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
            )
            for r in rows
        ]

    def entity_memories(
        self, entity_id: str, limit: int = 10, *, include_invalid: bool = False
    ) -> list[Memory]:
        active_clause = "" if include_invalid else " AND m.invalid_at IS NULL"
        with self._lock:
            rows = self._db.execute(
                f"SELECT DISTINCT {', '.join('m.' + c.strip() for c in _MEMORY_COLS.split(','))} "
                "FROM entity_mentions em JOIN memories m ON m.id = em.memory_id "
                f"WHERE em.entity_id = ?{active_clause} "
                "ORDER BY m.updated_at DESC LIMIT ?",
                (entity_id, limit),
            ).fetchall()
        return [_row_to_memory(r) for r in rows]

    def set_entity_type(self, entity_id: str, entity_type: str) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE entities SET entity_type = ?, updated_at = ?, "
                "description_updated_at = NULL WHERE id = ?",
                (entity_type, utcnow(), entity_id),
            )
            self._db.commit()

    def entities_of_memory(self, memory_id: str) -> list[Entity]:
        with self._lock:
            rows = self._db.execute(
                "SELECT e.* FROM entity_mentions em JOIN entities e ON e.id = em.entity_id "
                "WHERE em.memory_id = ? AND e.merged_into IS NULL",
                (memory_id,),
            ).fetchall()
        # distinct by id (a memory can mention an entity under several surfaces)
        seen, out = set(), []
        for r in rows:
            if r["id"] not in seen:
                seen.add(r["id"])
                out.append(self._row_to_entity(r))
        return out

    def touch_entity(self, entity_id: str) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE entities SET updated_at = ?, description_updated_at = NULL "
                "WHERE id = ?",
                (utcnow(), entity_id),
            )
            self._db.commit()

    def merge_entities(self, keep_id: str, merge_id: str) -> bool:
        """Idempotently fold both IDs' active roots into one entity."""
        with self._lock:
            keep_root = self.resolve_entity_id(keep_id)
            merge_root = self.resolve_entity_id(merge_id)
            if keep_root is None or merge_root is None:
                return False
            if keep_root == merge_root:
                return True
            changed_at = utcnow()
            cur = self._db.execute(
                "UPDATE entities SET merged_into = ?, updated_at = ?, "
                "description_updated_at = NULL "
                "WHERE id = ? AND merged_into IS NULL",
                (keep_root, changed_at, merge_root),
            )
            if cur.rowcount == 0:
                self._db.rollback()
                return False
            self._db.execute(
                "UPDATE entity_mentions SET entity_id = ? WHERE entity_id = ?",
                (keep_root, merge_root),
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
            self._db.execute(
                "UPDATE entity_proposals SET entity_a = ? WHERE entity_a = ?",
                (keep_root, merge_root),
            )
            self._db.execute(
                "UPDATE entity_proposals SET entity_b = ? WHERE entity_b = ?",
                (keep_root, merge_root),
            )
            self._db.execute(
                "UPDATE entity_proposals SET status = 'confirmed', decided_at = ? "
                "WHERE entity_a = entity_b AND status = 'proposed'",
                (changed_at,),
            )
            self._db.execute(
                "UPDATE entities SET updated_at = ?, description_updated_at = NULL "
                "WHERE id = ?",
                (changed_at, keep_root),
            )
            self._db.commit()
        return True

    def add_proposal(self, proposal: MergeProposal) -> MergeProposal:
        with self._lock:
            self._db.execute(
                "INSERT INTO entity_proposals (id, entity_a, entity_b, user_id, status, "
                "confidence, reason, created_at, decided_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    proposal.id, proposal.entity_a, proposal.entity_b, proposal.user_id,
                    proposal.status, proposal.confidence, proposal.reason,
                    proposal.created_at, proposal.decided_at,
                ),
            )
            self._db.commit()
        return proposal

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
                "ORDER BY created_at DESC LIMIT ?",
                (*params, limit),
            ).fetchall()
        return [self._row_to_proposal(r) for r in rows]

    def set_proposal_status(self, proposal_id: str, status: str) -> MergeProposal | None:
        with self._lock:
            cur = self._db.execute(
                "UPDATE entity_proposals SET status = ?, decided_at = ? WHERE id = ?",
                (status, utcnow(), proposal_id),
            )
            self._db.commit()
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

    def _backup_rows_for_users(
        self, table: str, user_ids: set[str | None]
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[str] = []
        named = sorted(value for value in user_ids if value is not None)
        if named:
            clauses.append(f"user_id IN ({','.join('?' * len(named))})")
            params.extend(named)
        if None in user_ids:
            clauses.append("user_id IS NULL")
        if not clauses:
            return []
        return self._select_backup_rows(table, " OR ".join(clauses), tuple(params))

    def export_backup(self, scope: Scope) -> dict[str, Any]:
        """Export exact source records; FTS and ANN remain derived indexes."""
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
            user_ids = {
                row["user_id"]
                for table in ("episodes", "memories", "topics", "entities")
                for row in tables[table]
            }
            if scope.user_id is not None:
                user_ids.add(scope.user_id)

            if scope.is_empty():
                for table in (
                    "memory_events", "memory_topics", "topic_relations",
                    "entity_mentions", "entity_proposals", "synthetic_tags",
                    "relations", "collections",
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
                tables["topic_relations"] = [
                    row for row in self._backup_rows_for_ids(
                        "topic_relations", "broader_topic_id", topic_ids
                    ) if row["narrower_topic_id"] in topic_ids
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
                tables["synthetic_tags"] = self._backup_rows_for_users(
                    "synthetic_tags", user_ids
                )
                collections = self._backup_rows_for_users("collections", user_ids)
                if scope.agent_id is not None or scope.run_id is not None:
                    collections = [
                        row for row in collections
                        if set(json.loads(row["memory_ids"])) <= memory_ids
                    ]
                tables["collections"] = collections

            ordered = {table: tables.get(table, []) for table in _BACKUP_ORDER}
        return {
            "format": "memry-backup", "version": 1, "created_at": utcnow(),
            "scope": scope.model_dump(), "tables": ordered,
        }

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
        if not isinstance(raw_tables, dict) or set(raw_tables) != set(_BACKUP_ORDER):
            raise ValueError("backup table set is incomplete or unknown")
        tables: dict[str, list[dict[str, Any]]] = {}
        for table in _BACKUP_ORDER:
            raw_rows = raw_tables[table]
            if not isinstance(raw_rows, list):
                raise ValueError(f"backup table {table} must be a list")
            columns = {row["name"] for row in self._db.execute(f"PRAGMA table_info({table})")}
            rows: list[dict[str, Any]] = []
            for raw in raw_rows:
                if not isinstance(raw, dict) or set(raw) != columns:
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
        for row in tables["memory_topics"]:
            if row["memory_id"] not in memory_ids or row["topic_id"] not in topic_ids:
                raise ValueError("topic assignment references data outside the backup")
        for row in tables["topic_relations"]:
            if row["broader_topic_id"] not in topic_ids or row["narrower_topic_id"] not in topic_ids:
                raise ValueError("topic hierarchy references data outside the backup")
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
        for row in tables["collections"]:
            members = json.loads(row["memory_ids"])
            if not isinstance(members, list) or not set(members) <= memory_ids:
                raise ValueError("collection references memories outside the backup")
        return tables

    def import_backup(
        self, backup: dict[str, Any], *, owner_prefix: str | None = None
    ) -> dict[str, Any]:
        """Restore exact records transactionally; never rewrite an identity."""
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
                            if dict(existing) != row:
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
                self._db.execute(
                    "INSERT OR IGNORE INTO ann_keys (memory_id) SELECT id FROM memories WHERE embedding IS NOT NULL"
                )
                self._db.commit()
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
                "SELECT COUNT(*) FROM entities WHERE merged_into IS NULL"
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
            "open_merge_proposals": proposals,
            "ann": {
                "available": HAS_USEARCH,
                "active": any(
                    s.size >= self._ann_cfg.min_rows for s in self._anns.values()
                ),
                "indexed": sum(s.size for s in self._anns.values()),
            },
        }

    def reset(self) -> None:
        with self._lock:
            for table in (
                "memory_topics", "topic_relations", "topics", "memories", "episodes",
                "memory_events", "entities", "entity_mentions", "entity_proposals", "ann_keys",
            ):
                self._db.execute(f"DELETE FROM {table}")
            self._db.commit()
        for sidecar in self._anns.values():
            sidecar.rebuild([])

    def close(self) -> None:
        for sidecar in self._anns.values():
            sidecar.save()
        with self._lock:
            self._db.close()
