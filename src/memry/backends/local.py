"""Local SQLite backend - the default, zero-service storage engine.

One file holds everything: raw episodes, derived memories, an FTS5 index
(BM25 keyword search), float32 embeddings (brute-force cosine via numpy -
fast enough into the hundreds of thousands of memories), and the full event
history. WAL mode + a process-wide lock make it safe for the MCP/REST servers.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any

import numpy as np

from ..config import AnnConfig
from ..models import (
    Entity,
    EntityMention,
    Episode,
    Memory,
    MemoryEvent,
    MergeProposal,
    Scope,
    SyntheticTag,
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
    metadata TEXT NOT NULL DEFAULT '{}',
    merged_into TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_entities_norm ON entities(normalized, user_id);

CREATE TABLE IF NOT EXISTS entity_mentions (
    id TEXT PRIMARY KEY,
    entity_id TEXT NOT NULL,
    memory_id TEXT NOT NULL,
    surface TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mentions_entity ON entity_mentions(entity_id);
CREATE INDEX IF NOT EXISTS idx_mentions_memory ON entity_mentions(memory_id);

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
"""

_MEMORY_COLS = (
    "id, content, memory_type, user_id, agent_id, run_id, importance, categories, "
    "entities, metadata, created_at, updated_at, valid_from, invalid_at, superseded_by, "
    "source_episode_ids, embedding_model"
)

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


def _category_clause(categories: list[str] | None, column: str) -> tuple[str, list[Any]]:
    if not categories:
        return "1=1", []
    placeholders = ",".join("?" * len(categories))
    return (
        f"EXISTS (SELECT 1 FROM json_each({column}) "
        f"WHERE lower(json_each.value) IN ({placeholders}))",
        [c.lower() for c in categories],
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
        self._db.commit()
        self.db_path = db_path
        self._ann_cfg = ann or AnnConfig()
        # One sidecar per embedding model: a multiuser server with per-account
        # BYO-key can run several models against this one DB, and a single slot
        # would rebuild the whole HNSW index every time the model alternated.
        self._anns: dict[tuple[str, int], HnswSidecar] = {}
        self._ann_pending_saves = 0

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
        metadata: dict[str, Any] | None = None,
        source_episode_ids: list[str] | None = None,
    ) -> Memory | None:
        from ..models import utcnow

        sets: list[str] = ["updated_at = ?"]
        params: list[Any] = [utcnow()]
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
        with self._lock:
            cur = self._db.execute(
                f"UPDATE memories SET {', '.join(sets)} WHERE id = ?", (*params, memory_id)
            )
            if cur.rowcount and embedding is not None and embedding_model is not None:
                self._ann_add(memory_id, embedding, embedding_model)
            self._db.commit()
        if cur.rowcount == 0:
            return None
        return self.get_memory(memory_id)

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
            self._db.commit()
        if cur.rowcount == 0:
            return None
        return self.get_memory(memory_id)

    def delete_memory(self, memory_id: str) -> bool:
        with self._lock:
            cur = self._db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            if cur.rowcount:
                self._ann_remove(memory_id)
                self._db.execute("DELETE FROM ann_keys WHERE memory_id = ?", (memory_id,))
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
    ) -> list[Memory]:
        clause, params = _scope_clause(scope)
        cat_clause, cat_params = _category_clause(categories, "categories")
        if not include_invalid:
            clause += " AND invalid_at IS NULL"
        with self._lock:
            rows = self._db.execute(
                f"SELECT {_MEMORY_COLS} FROM memories WHERE {clause} AND {cat_clause} "
                "ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                (*params, *cat_params, limit, offset),
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
    ) -> list[tuple[Memory, float]]:
        clause, params = _scope_clause(scope)
        cat_clause, cat_params = _category_clause(categories, "categories")
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
                        f"AND {clause} AND {cat_clause} "
                        "AND embedding IS NOT NULL AND embedding_model = ?",
                        (*keys, *params, *cat_params, embedding_model),
                    ).fetchall()
                if len(rows) >= limit:
                    return self._score_rows(rows, embedding, limit)
                # restrictive filters starved the ANN candidates -> exact scan

        with self._lock:
            rows = self._db.execute(
                f"SELECT {_MEMORY_COLS}, embedding FROM memories "
                f"WHERE {clause} AND {cat_clause} "
                "AND embedding IS NOT NULL AND embedding_model = ?",
                (*params, *cat_params, embedding_model),
            ).fetchall()
        return self._score_rows(rows, embedding, limit)

    def keyword_search(
        self,
        query: str,
        scope: Scope,
        limit: int = 20,
        include_invalid: bool = False,
        categories: list[str] | None = None,
    ) -> list[tuple[Memory, float]]:
        tokens = _WORD_RE.findall(query)
        if not tokens:
            return []
        match = " OR ".join(f'"{t}"' for t in tokens[:32])
        clause, params = _scope_clause(scope, prefix="m.")
        cat_clause, cat_params = _category_clause(categories, "m.categories")
        if not include_invalid:
            clause += " AND m.invalid_at IS NULL"
        sql = (
            f"SELECT {', '.join('m.' + c.strip() for c in _MEMORY_COLS.split(','))}, "
            "bm25(memories_fts) AS rank_score "
            "FROM memories_fts JOIN memories m ON m.rowid = memories_fts.rowid "
            f"WHERE memories_fts MATCH ? AND {clause} AND {cat_clause} "
            "ORDER BY rank_score LIMIT ?"
        )
        with self._lock:
            rows = self._db.execute(sql, (match, *params, *cat_params, limit)).fetchall()
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
                "run_id, metadata, merged_into, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    entity.id, entity.name, entity.normalized, entity.entity_type,
                    entity.user_id, entity.agent_id, entity.run_id,
                    json.dumps(entity.metadata), entity.merged_into,
                    entity.created_at, entity.updated_at,
                ),
            )
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

    def entity_memories(self, entity_id: str, limit: int = 10) -> list[Memory]:
        with self._lock:
            rows = self._db.execute(
                f"SELECT DISTINCT {', '.join('m.' + c.strip() for c in _MEMORY_COLS.split(','))} "
                "FROM entity_mentions em JOIN memories m ON m.id = em.memory_id "
                "WHERE em.entity_id = ? ORDER BY m.updated_at DESC LIMIT ?",
                (entity_id, limit),
            ).fetchall()
        return [_row_to_memory(r) for r in rows]

    def merge_entities(self, keep_id: str, merge_id: str) -> bool:
        if keep_id == merge_id:
            return False
        with self._lock:
            keep = self._db.execute("SELECT id FROM entities WHERE id = ?", (keep_id,)).fetchone()
            cur = self._db.execute(
                "UPDATE entities SET merged_into = ?, updated_at = ? "
                "WHERE id = ? AND merged_into IS NULL",
                (keep_id, utcnow(), merge_id),
            )
            if not keep or cur.rowcount == 0:
                self._db.rollback()
                return False
            self._db.execute(
                "UPDATE entity_mentions SET entity_id = ? WHERE entity_id = ?",
                (keep_id, merge_id),
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
            for table in ("memories", "episodes", "memory_events", "entities",
                          "entity_mentions", "entity_proposals", "ann_keys"):
                self._db.execute(f"DELETE FROM {table}")
            self._db.commit()
        for sidecar in self._anns.values():
            sidecar.rebuild([])

    def close(self) -> None:
        for sidecar in self._anns.values():
            sidecar.save()
        with self._lock:
            self._db.close()
