"""Mem0 comparison/import adapter (optional: pip install memry[mem0]).

This module is never selected by Config, the CLI, REST, MCP, or either server.
Comparison and import code may instantiate it directly when it needs to read
or exercise Mem0 through Memry's storage interface.

It is deliberately incomplete: Mem0 has no Memry raw-episode store or temporal
invalidation, and event history is limited to what mem0.history() returns.
It must not be used as persistence for a running Memry product.
"""

from __future__ import annotations

from typing import Any

from ..models import Episode, Memory, MemoryEvent, Scope, utcnow
from .base import MemoryBackend


def _scope_kwargs(scope: Scope) -> dict[str, str]:
    kwargs: dict[str, str] = {}
    if scope.user_id:
        kwargs["user_id"] = scope.user_id
    if scope.agent_id:
        kwargs["agent_id"] = scope.agent_id
    if scope.run_id:
        kwargs["run_id"] = scope.run_id
    return kwargs


class Mem0ComparisonAdapter(MemoryBackend):
    def __init__(self) -> None:
        try:
            from mem0 import Memory as Mem0Memory
        except ImportError as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "The Mem0 comparison adapter requires mem0ai: pip install 'memry[mem0]'"
            ) from exc
        self._m = Mem0Memory()

    # -- episodes: mem0 has no raw-event store --------------------------
    def add_episodes(self, episodes: list[Episode]) -> None:
        return None

    def list_episodes(self, scope: Scope, limit: int = 100) -> list[Episode]:
        return []

    # -- memories --------------------------------------------------------
    def _record_to_memory(self, rec: dict[str, Any]) -> Memory:
        meta = rec.get("metadata") or {}
        return Memory(
            id=str(rec.get("id")),
            content=rec.get("memory") or rec.get("text") or "",
            memory_type=meta.get("memory_type", "semantic"),
            user_id=rec.get("user_id"),
            agent_id=rec.get("agent_id"),
            run_id=rec.get("run_id"),
            importance=float(meta.get("importance", 0.5)),
            categories=meta.get("categories", []) or [],
            entities=meta.get("entities", []) or [],
            metadata={k: v for k, v in meta.items() if k not in ("memory_type", "importance", "categories", "entities")},
            created_at=rec.get("created_at") or utcnow(),
            updated_at=rec.get("updated_at") or rec.get("created_at") or utcnow(),
        )

    def insert_memory(self, memory: Memory, embedding: list[float] | None = None) -> Memory:
        metadata = {
            **memory.metadata,
            "memory_type": memory.memory_type,
            "importance": memory.importance,
            "categories": memory.categories,
            "entities": memory.entities,
        }
        result = self._m.add(
            memory.content,
            infer=False,
            metadata=metadata,
            **_scope_kwargs(memory.scope()),
        )
        records = result.get("results", []) if isinstance(result, dict) else []
        if records:
            memory.id = str(records[0].get("id", memory.id))
        return memory

    def update_memory(self, memory_id: str, *, content: str | None = None, **_: Any) -> Memory | None:
        if content is not None:
            self._m.update(memory_id, content)
        return self.get_memory(memory_id)

    def invalidate_memory(self, memory_id: str, *, superseded_by: str | None = None) -> Memory | None:
        # mem0 has no temporal invalidation; fall back to hard delete.
        memory = self.get_memory(memory_id)
        self._m.delete(memory_id)
        if memory:
            memory.invalid_at = utcnow()
            memory.superseded_by = superseded_by
        return memory

    def delete_memory(self, memory_id: str) -> bool:
        self._m.delete(memory_id)
        return True

    def get_memory(self, memory_id: str) -> Memory | None:
        rec = self._m.get(memory_id)
        return self._record_to_memory(rec) if rec else None

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
        if entity_id:
            return []  # the reduced Mem0 adapter has no stable entity IDs
        result = self._m.get_all(**_scope_kwargs(scope), limit=limit + offset)
        records = result.get("results", []) if isinstance(result, dict) else (result or [])
        memories = [self._record_to_memory(r) for r in records]
        if categories:
            wanted = {value.strip().casefold() for value in categories if value.strip()}
            memories = [
                memory for memory in memories
                if wanted & {value.casefold() for value in memory.categories}
            ]
        return memories[offset : offset + limit]

    # -- search ------------------------------------------------------------
    def native_search(self, query: str, scope: Scope, limit: int = 20):
        result = self._m.search(query, limit=limit, **_scope_kwargs(scope))
        records = result.get("results", []) if isinstance(result, dict) else (result or [])
        return [
            (self._record_to_memory(r), float(r.get("score", 0.0) or 0.0)) for r in records
        ]

    def vector_search(
        self, embedding, embedding_model, scope, limit=20, include_invalid=False,
        categories=None, entity_id=None,
    ):
        return []  # native_search covers retrieval for this backend

    def keyword_search(
        self, query, scope, limit=20, include_invalid=False,
        categories=None, entity_id=None,
    ):
        return []  # native_search covers retrieval for this backend

    # -- events --------------------------------------------------------------
    def add_event(self, event: MemoryEvent) -> None:
        return None  # mem0 records its own history internally

    def history(self, memory_id: str) -> list[MemoryEvent]:
        entries = self._m.history(memory_id) or []
        events: list[MemoryEvent] = []
        for entry in entries:
            events.append(
                MemoryEvent(
                    memory_id=memory_id,
                    event=str(entry.get("event", "UPDATE")).upper(),  # type: ignore[arg-type]
                    old_content=entry.get("old_memory"),
                    new_content=entry.get("new_memory"),
                    created_at=entry.get("created_at") or utcnow(),
                )
            )
        return events

    # -- maintenance ----------------------------------------------------------
    def all_memories_iter(self, include_invalid: bool = True) -> list[Memory]:
        return self.list_memories(Scope(), limit=10_000)

    def stats(self) -> dict[str, Any]:
        return {"backend": "mem0", "note": "stats limited on the mem0 adapter"}

    def reset(self) -> None:
        self._m.reset()
