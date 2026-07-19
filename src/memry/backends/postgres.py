"""PostgreSQL backend - the multi-writer, fleet-scale storage engine.

Use when several server processes (or machines) must write one memory store:
Postgres provides real concurrent writes, replication, and operational HA that
single-file SQLite cannot. Requires ``pip install memry[postgres]`` and the
`pgvector <https://github.com/pgvector/pgvector>`_ extension (preinstalled on
every major managed Postgres and in the ``pgvector/pgvector`` Docker image).

    MEMRY_BACKEND=postgres
    MEMRY_POSTGRES_DSN=postgresql://user:pass@host:5432/memry

Parity: implements the full MemoryBackend contract - episodes, bi-temporal
memories, events, entities/proposals, categories filters. Keyword search uses
``tsvector``/``ts_rank``; vector search uses pgvector cosine distance.
Timestamps stay ISO-8601 TEXT for cross-backend consistency.
"""

from __future__ import annotations

import json
import threading
from typing import Any

from ..models import (
    Entity,
    EntityMention,
    Episode,
    Memory,
    MemoryEvent,
    MergeProposal,
    Scope,
    utcnow,
)
from .base import MemoryBackend

_TOKEN_SAFE = __import__("re").compile(r"[A-Za-z0-9]+")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memry_episodes (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'user',
    user_id TEXT, agent_id TEXT, run_id TEXT,
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memry_episodes_scope ON memry_episodes(user_id, agent_id, run_id);

CREATE TABLE IF NOT EXISTS memry_memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    memory_type TEXT NOT NULL DEFAULT 'semantic',
    user_id TEXT, agent_id TEXT, run_id TEXT,
    importance DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    categories JSONB NOT NULL DEFAULT '[]',
    entities JSONB NOT NULL DEFAULT '[]',
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    valid_from TEXT, invalid_at TEXT, superseded_by TEXT,
    source_episode_ids JSONB NOT NULL DEFAULT '[]',
    embedding vector,
    embedding_model TEXT,
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED
);
CREATE INDEX IF NOT EXISTS idx_memry_memories_scope ON memry_memories(user_id, agent_id, run_id);
CREATE INDEX IF NOT EXISTS idx_memry_memories_invalid ON memry_memories(invalid_at);
CREATE INDEX IF NOT EXISTS idx_memry_memories_tsv ON memry_memories USING GIN (tsv);

CREATE TABLE IF NOT EXISTS memry_events (
    id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL,
    event TEXT NOT NULL,
    old_content TEXT, new_content TEXT, reason TEXT,
    actor TEXT NOT NULL DEFAULT 'system',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memry_events_memory ON memry_events(memory_id);

CREATE TABLE IF NOT EXISTS memry_entities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    normalized TEXT NOT NULL,
    entity_type TEXT,
    user_id TEXT, agent_id TEXT, run_id TEXT,
    metadata JSONB NOT NULL DEFAULT '{}',
    merged_into TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memry_entities_norm ON memry_entities(normalized, user_id);

CREATE TABLE IF NOT EXISTS memry_entity_mentions (
    id TEXT PRIMARY KEY,
    entity_id TEXT NOT NULL,
    memory_id TEXT NOT NULL,
    surface TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memry_mentions_entity ON memry_entity_mentions(entity_id);

CREATE TABLE IF NOT EXISTS memry_entity_proposals (
    id TEXT PRIMARY KEY,
    entity_a TEXT NOT NULL,
    entity_b TEXT NOT NULL,
    user_id TEXT,
    status TEXT NOT NULL DEFAULT 'proposed',
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    reason TEXT,
    created_at TEXT NOT NULL,
    decided_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_memry_proposals_status ON memry_entity_proposals(status, user_id);
"""

_MEMORY_COLS = (
    "id, content, memory_type, user_id, agent_id, run_id, importance, categories, "
    "entities, metadata, created_at, updated_at, valid_from, invalid_at, superseded_by, "
    "source_episode_ids, embedding_model"
)


def _scope_clause(scope: Scope, prefix: str = "") -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    for field in ("user_id", "agent_id", "run_id"):
        value = getattr(scope, field)
        if value is not None:
            clauses.append(f"{prefix}{field} = %s")
            params.append(value)
    return (" AND ".join(clauses) if clauses else "TRUE"), params


def _category_clause(categories: list[str] | None, column: str) -> tuple[str, list[Any]]:
    if not categories:
        return "TRUE", []
    placeholders = ",".join(["%s"] * len(categories))
    return (
        f"EXISTS (SELECT 1 FROM jsonb_array_elements_text({column}) c "
        f"WHERE lower(c) IN ({placeholders}))",
        [c.lower() for c in categories],
    )


def _vec_literal(embedding: list[float]) -> str:
    return "[" + ",".join(f"{v:.8g}" for v in embedding) + "]"


def _jsonb(value: Any) -> str:
    return json.dumps(value)


def _row_to_memory(row: dict[str, Any]) -> Memory:
    return Memory(
        id=row["id"], content=row["content"], memory_type=row["memory_type"],
        user_id=row["user_id"], agent_id=row["agent_id"], run_id=row["run_id"],
        importance=row["importance"], categories=row["categories"],
        entities=row["entities"], metadata=row["metadata"],
        created_at=row["created_at"], updated_at=row["updated_at"],
        valid_from=row["valid_from"], invalid_at=row["invalid_at"],
        superseded_by=row["superseded_by"], source_episode_ids=row["source_episode_ids"],
        embedding_model=row["embedding_model"],
    )


class PostgresBackend(MemoryBackend):
    def __init__(self, config: Any) -> None:
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "The postgres backend requires psycopg: pip install 'memry[postgres]'"
            ) from exc
        dsn = getattr(config, "postgres_dsn", None)
        if not dsn:
            raise RuntimeError("backend=postgres requires MEMRY_POSTGRES_DSN")
        self._lock = threading.RLock()
        self._db = psycopg.connect(dsn, autocommit=True, row_factory=dict_row)
        with self._lock, self._db.cursor() as cur:
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            except Exception as exc:
                raise RuntimeError(
                    "memry's postgres backend needs the pgvector extension "
                    "(available on all major managed Postgres offerings and in "
                    "the pgvector/pgvector Docker image)."
                ) from exc
            cur.execute(_SCHEMA)

    def _exec(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        with self._lock, self._db.cursor() as cur:
            cur.execute(sql, params)
            if cur.description is None:
                return []
            return cur.fetchall()

    def _exec_rowcount(self, sql: str, params: tuple = ()) -> int:
        with self._lock, self._db.cursor() as cur:
            cur.execute(sql, params)
            return cur.rowcount

    # -- episodes -------------------------------------------------------
    def add_episodes(self, episodes: list[Episode]) -> None:
        with self._lock, self._db.cursor() as cur:
            cur.executemany(
                "INSERT INTO memry_episodes (id, content, role, user_id, agent_id, run_id, "
                "metadata, created_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                [
                    (e.id, e.content, e.role, e.user_id, e.agent_id, e.run_id,
                     _jsonb(e.metadata), e.created_at)
                    for e in episodes
                ],
            )

    def list_episodes(self, scope: Scope, limit: int = 100) -> list[Episode]:
        clause, params = _scope_clause(scope)
        rows = self._exec(
            f"SELECT * FROM memry_episodes WHERE {clause} ORDER BY created_at DESC LIMIT %s",
            (*params, limit),
        )
        return [
            Episode(
                id=r["id"], content=r["content"], role=r["role"], user_id=r["user_id"],
                agent_id=r["agent_id"], run_id=r["run_id"], metadata=r["metadata"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    # -- memories -------------------------------------------------------
    def insert_memory(self, memory: Memory, embedding: list[float] | None = None) -> Memory:
        if memory.valid_from is None:
            memory.valid_from = memory.created_at
        self._exec(
            "INSERT INTO memry_memories (id, content, memory_type, user_id, agent_id, run_id, "
            "importance, categories, entities, metadata, created_at, updated_at, valid_from, "
            "invalid_at, superseded_by, source_episode_ids, embedding, embedding_model) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::vector,%s)",
            (
                memory.id, memory.content, memory.memory_type, memory.user_id,
                memory.agent_id, memory.run_id, memory.importance,
                _jsonb(memory.categories), _jsonb(memory.entities), _jsonb(memory.metadata),
                memory.created_at, memory.updated_at, memory.valid_from, memory.invalid_at,
                memory.superseded_by, _jsonb(memory.source_episode_ids),
                _vec_literal(embedding) if embedding else None, memory.embedding_model,
            ),
        )
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
        sets, params = ["updated_at = %s"], [utcnow()]
        for column, value, transform in (
            ("content", content, None),
            ("embedding_model", embedding_model, None),
            ("importance", importance, None),
            ("memory_type", memory_type, None),
            ("categories", categories, _jsonb),
            ("entities", entities, _jsonb),
            ("metadata", metadata, _jsonb),
            ("source_episode_ids", source_episode_ids, _jsonb),
        ):
            if value is not None:
                sets.append(f"{column} = %s")
                params.append(transform(value) if transform else value)
        if embedding is not None:
            sets.append("embedding = %s::vector")
            params.append(_vec_literal(embedding))
        count = self._exec_rowcount(
            f"UPDATE memry_memories SET {', '.join(sets)} WHERE id = %s",
            (*params, memory_id),
        )
        return self.get_memory(memory_id) if count else None

    def invalidate_memory(
        self, memory_id: str, *, superseded_by: str | None = None
    ) -> Memory | None:
        count = self._exec_rowcount(
            "UPDATE memry_memories SET invalid_at = %s, superseded_by = %s, updated_at = %s "
            "WHERE id = %s AND invalid_at IS NULL",
            (utcnow(), superseded_by, utcnow(), memory_id),
        )
        return self.get_memory(memory_id) if count else None

    def delete_memory(self, memory_id: str) -> bool:
        return self._exec_rowcount(
            "DELETE FROM memry_memories WHERE id = %s", (memory_id,)
        ) > 0

    def get_memory(self, memory_id: str) -> Memory | None:
        rows = self._exec(
            f"SELECT {_MEMORY_COLS} FROM memry_memories WHERE id = %s", (memory_id,)
        )
        return _row_to_memory(rows[0]) if rows else None

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
        rows = self._exec(
            f"SELECT {_MEMORY_COLS} FROM memry_memories WHERE {clause} AND {cat_clause} "
            "ORDER BY updated_at DESC LIMIT %s OFFSET %s",
            (*params, *cat_params, limit, offset),
        )
        return [_row_to_memory(r) for r in rows]

    # -- search -----------------------------------------------------------
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
        vec = _vec_literal(embedding)
        rows = self._exec(
            f"SELECT {_MEMORY_COLS}, 1 - (embedding <=> %s::vector) AS sim "
            f"FROM memry_memories WHERE {clause} AND {cat_clause} "
            "AND embedding IS NOT NULL AND embedding_model = %s "
            "ORDER BY embedding <=> %s::vector LIMIT %s",
            (vec, *params, *cat_params, embedding_model, vec, limit),
        )
        return [(_row_to_memory(r), float(r["sim"])) for r in rows]

    def keyword_search(
        self,
        query: str,
        scope: Scope,
        limit: int = 20,
        include_invalid: bool = False,
        categories: list[str] | None = None,
    ) -> list[tuple[Memory, float]]:
        tokens = _TOKEN_SAFE.findall(query)
        if not tokens:
            return []
        tsquery = " | ".join(tokens[:32])
        clause, params = _scope_clause(scope)
        cat_clause, cat_params = _category_clause(categories, "categories")
        if not include_invalid:
            clause += " AND invalid_at IS NULL"
        rows = self._exec(
            f"SELECT {_MEMORY_COLS}, ts_rank(tsv, to_tsquery('simple', %s)) AS score "
            f"FROM memry_memories WHERE tsv @@ to_tsquery('simple', %s) "
            f"AND {clause} AND {cat_clause} ORDER BY score DESC LIMIT %s",
            (tsquery, tsquery, *params, *cat_params, limit),
        )
        return [(_row_to_memory(r), float(r["score"])) for r in rows]

    # -- events -----------------------------------------------------------
    def add_event(self, event: MemoryEvent) -> None:
        self._exec(
            "INSERT INTO memry_events (id, memory_id, event, old_content, new_content, "
            "reason, actor, created_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
            (event.id, event.memory_id, event.event, event.old_content, event.new_content,
             event.reason, event.actor, event.created_at),
        )

    def history(self, memory_id: str) -> list[MemoryEvent]:
        rows = self._exec(
            "SELECT * FROM memry_events WHERE memory_id = %s ORDER BY created_at",
            (memory_id,),
        )
        return [
            MemoryEvent(
                id=r["id"], memory_id=r["memory_id"], event=r["event"],
                old_content=r["old_content"], new_content=r["new_content"],
                reason=r["reason"], actor=r["actor"], created_at=r["created_at"],
            )
            for r in rows
        ]

    # -- entities -----------------------------------------------------------
    @staticmethod
    def _row_to_entity(r: dict[str, Any]) -> Entity:
        return Entity(
            id=r["id"], name=r["name"], normalized=r["normalized"],
            entity_type=r["entity_type"], user_id=r["user_id"], agent_id=r["agent_id"],
            run_id=r["run_id"], metadata=r["metadata"], merged_into=r["merged_into"],
            created_at=r["created_at"], updated_at=r["updated_at"],
        )

    @staticmethod
    def _row_to_proposal(r: dict[str, Any]) -> MergeProposal:
        return MergeProposal(
            id=r["id"], entity_a=r["entity_a"], entity_b=r["entity_b"], user_id=r["user_id"],
            status=r["status"], confidence=r["confidence"], reason=r["reason"],
            created_at=r["created_at"], decided_at=r["decided_at"],
        )

    def insert_entity(self, entity: Entity) -> Entity:
        if not entity.normalized:
            entity.normalized = entity.name.strip().lower()
        self._exec(
            "INSERT INTO memry_entities (id, name, normalized, entity_type, user_id, agent_id, "
            "run_id, metadata, merged_into, created_at, updated_at) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (entity.id, entity.name, entity.normalized, entity.entity_type, entity.user_id,
             entity.agent_id, entity.run_id, _jsonb(entity.metadata), entity.merged_into,
             entity.created_at, entity.updated_at),
        )
        return entity

    def get_entity(self, entity_id: str) -> Entity | None:
        rows = self._exec("SELECT * FROM memry_entities WHERE id = %s", (entity_id,))
        return self._row_to_entity(rows[0]) if rows else None

    def find_entities(self, normalized: str, scope: Scope) -> list[Entity]:
        clause, params = _scope_clause(scope)
        rows = self._exec(
            f"SELECT * FROM memry_entities WHERE normalized = %s AND merged_into IS NULL "
            f"AND {clause} ORDER BY updated_at DESC",
            (normalized.strip().lower(), *params),
        )
        return [self._row_to_entity(r) for r in rows]

    def list_entities(
        self, scope: Scope, *, include_merged: bool = False, limit: int = 100
    ) -> list[Entity]:
        clause, params = _scope_clause(scope)
        if not include_merged:
            clause += " AND merged_into IS NULL"
        rows = self._exec(
            f"SELECT * FROM memry_entities WHERE {clause} ORDER BY updated_at DESC LIMIT %s",
            (*params, limit),
        )
        return [self._row_to_entity(r) for r in rows]

    def add_mention(self, mention: EntityMention) -> None:
        self._exec(
            "INSERT INTO memry_entity_mentions (id, entity_id, memory_id, surface, created_at) "
            "VALUES (%s,%s,%s,%s,%s)",
            (mention.id, mention.entity_id, mention.memory_id, mention.surface,
             mention.created_at),
        )

    def entity_mentions(self, entity_id: str) -> list[EntityMention]:
        rows = self._exec(
            "SELECT * FROM memry_entity_mentions WHERE entity_id = %s ORDER BY created_at",
            (entity_id,),
        )
        return [
            EntityMention(id=r["id"], entity_id=r["entity_id"], memory_id=r["memory_id"],
                          surface=r["surface"], created_at=r["created_at"])
            for r in rows
        ]

    def entity_memories(self, entity_id: str, limit: int = 10) -> list[Memory]:
        rows = self._exec(
            f"SELECT DISTINCT {', '.join('m.' + c.strip() for c in _MEMORY_COLS.split(','))} "
            "FROM memry_entity_mentions em JOIN memry_memories m ON m.id = em.memory_id "
            "WHERE em.entity_id = %s ORDER BY m.updated_at DESC LIMIT %s",
            (entity_id, limit),
        )
        return [_row_to_memory(r) for r in rows]

    def merge_entities(self, keep_id: str, merge_id: str) -> bool:
        if keep_id == merge_id or self.get_entity(keep_id) is None:
            return False
        count = self._exec_rowcount(
            "UPDATE memry_entities SET merged_into = %s, updated_at = %s "
            "WHERE id = %s AND merged_into IS NULL",
            (keep_id, utcnow(), merge_id),
        )
        if not count:
            return False
        self._exec(
            "UPDATE memry_entity_mentions SET entity_id = %s WHERE entity_id = %s",
            (keep_id, merge_id),
        )
        return True

    def add_proposal(self, proposal: MergeProposal) -> MergeProposal:
        self._exec(
            "INSERT INTO memry_entity_proposals (id, entity_a, entity_b, user_id, status, "
            "confidence, reason, created_at, decided_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (proposal.id, proposal.entity_a, proposal.entity_b, proposal.user_id,
             proposal.status, proposal.confidence, proposal.reason, proposal.created_at,
             proposal.decided_at),
        )
        return proposal

    def get_proposal(self, proposal_id: str) -> MergeProposal | None:
        rows = self._exec(
            "SELECT * FROM memry_entity_proposals WHERE id = %s", (proposal_id,)
        )
        return self._row_to_proposal(rows[0]) if rows else None

    def find_proposal(self, entity_a: str, entity_b: str) -> MergeProposal | None:
        rows = self._exec(
            "SELECT * FROM memry_entity_proposals WHERE (entity_a = %s AND entity_b = %s) "
            "OR (entity_a = %s AND entity_b = %s)",
            (entity_a, entity_b, entity_b, entity_a),
        )
        return self._row_to_proposal(rows[0]) if rows else None

    def list_proposals(
        self, scope: Scope, *, status: str | None = "proposed", limit: int = 100
    ) -> list[MergeProposal]:
        clauses, params = [], []
        if scope.user_id is not None:
            clauses.append("user_id = %s")
            params.append(scope.user_id)
        if status is not None:
            clauses.append("status = %s")
            params.append(status)
        where = " AND ".join(clauses) if clauses else "TRUE"
        rows = self._exec(
            f"SELECT * FROM memry_entity_proposals WHERE {where} "
            "ORDER BY created_at DESC LIMIT %s",
            (*params, limit),
        )
        return [self._row_to_proposal(r) for r in rows]

    def set_proposal_status(self, proposal_id: str, status: str) -> MergeProposal | None:
        count = self._exec_rowcount(
            "UPDATE memry_entity_proposals SET status = %s, decided_at = %s WHERE id = %s",
            (status, utcnow(), proposal_id),
        )
        return self.get_proposal(proposal_id) if count else None

    # -- maintenance --------------------------------------------------------
    def all_memories_iter(self, include_invalid: bool = True) -> list[Memory]:
        clause = "TRUE" if include_invalid else "invalid_at IS NULL"
        rows = self._exec(f"SELECT {_MEMORY_COLS} FROM memry_memories WHERE {clause}")
        return [_row_to_memory(r) for r in rows]

    def stats(self) -> dict[str, Any]:
        def one(sql: str) -> Any:
            return self._exec(sql)[0]["count"]

        return {
            "backend": "postgres",
            "active_memories": one(
                "SELECT COUNT(*) AS count FROM memry_memories WHERE invalid_at IS NULL"
            ),
            "invalidated_memories": one(
                "SELECT COUNT(*) AS count FROM memry_memories WHERE invalid_at IS NOT NULL"
            ),
            "episodes": one("SELECT COUNT(*) AS count FROM memry_episodes"),
            "events": one("SELECT COUNT(*) AS count FROM memry_events"),
            "entities": one(
                "SELECT COUNT(*) AS count FROM memry_entities WHERE merged_into IS NULL"
            ),
            "open_merge_proposals": one(
                "SELECT COUNT(*) AS count FROM memry_entity_proposals WHERE status = 'proposed'"
            ),
        }

    def reset(self) -> None:
        for table in ("memry_memories", "memry_episodes", "memry_events", "memry_entities",
                      "memry_entity_mentions", "memry_entity_proposals"):
            self._exec(f"DELETE FROM {table}")

    def close(self) -> None:
        with self._lock:
            self._db.close()
