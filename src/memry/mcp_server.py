"""Memry MCP server.

Exposes the memory layer to any MCP client (Claude Code, Claude Desktop,
Cursor, Windsurf, Codex, ...) over stdio or streamable HTTP.

    memry mcp                      # stdio (for local agent config)
    memry mcp --transport http     # streamable HTTP on :8787/mcp
"""

from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from .config import Config
from .store import MemoryStore

INSTRUCTIONS = """Memry gives you persistent long-term memory across sessions.

- At the START of a task, call get_memory_context (or search_memories) with a
  short description of the task to recall relevant knowledge about the user.
- WHENEVER the user shares stable facts, preferences, decisions, corrections,
  or plans, call save_memories with that information so it persists.
- Never store secrets (passwords, API keys, tokens).
"""


def _memory_row(m: Any, score: float | None = None) -> dict[str, Any]:
    row = {
        "id": m.id,
        "content": m.content,
        "type": m.memory_type,
        "importance": m.importance,
        "categories": m.categories,
        "created_at": m.created_at,
        "updated_at": m.updated_at,
    }
    if m.invalid_at:
        row["invalid_at"] = m.invalid_at
    if score is not None:
        row["score"] = round(score, 4)
    return row


def create_server(
    store: MemoryStore | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = 8787,
) -> FastMCP:
    mcp = FastMCP("memry", instructions=INSTRUCTIONS, host=host, port=port)
    store = store or MemoryStore()
    default_user = store.config.default_user_id

    def _uid(user_id: str) -> str | None:
        return user_id or default_user

    @mcp.tool()
    def save_memories(
        content: str,
        user_id: str = "",
        agent_id: str = "",
        run_id: str = "",
        infer: bool = True,
    ) -> str:
        """Store information in long-term memory. Call this whenever the user
        shares stable facts, preferences, decisions, corrections, or plans.
        With infer=true (default) the text is distilled into discrete facts and
        reconciled against existing memories (duplicates skipped, contradictions
        superseded). With infer=false the text is saved verbatim as one memory.
        """
        result = store.add(
            content,
            user_id=_uid(user_id),
            agent_id=agent_id or None,
            run_id=run_id or None,
            infer=infer,
        )
        return json.dumps(
            {
                "saved": result.summary(),
                "actions": [
                    {"event": a.event, "memory_id": a.memory_id, "content": a.content}
                    for a in result.actions
                ],
            },
            ensure_ascii=False,
        )

    @mcp.tool()
    def search_memories(
        query: str,
        user_id: str = "",
        agent_id: str = "",
        run_id: str = "",
        limit: int = 8,
        categories: str = "",
    ) -> str:
        """Search long-term memory. Call this at the start of a task (and any
        time you need background about the user, their preferences, or past
        decisions). Returns the most relevant memories, best first.
        Optionally restrict to categories (comma-separated, e.g. "diet,health").
        """
        category_list = [c.strip() for c in categories.split(",") if c.strip()] or None
        results = store.search(
            query,
            user_id=_uid(user_id),
            agent_id=agent_id or None,
            run_id=run_id or None,
            limit=limit,
            categories=category_list,
        )
        return json.dumps(
            [_memory_row(r.memory, r.score) for r in results], ensure_ascii=False
        )

    @mcp.tool()
    def get_memory_context(
        query: str,
        user_id: str = "",
        token_budget: int = 1200,
    ) -> str:
        """Get a ready-to-use context block of the most relevant memories for a
        task, packed to fit the given token budget. Prefer this over
        search_memories when you just want background context injected."""
        ctx = store.reconstruct_context(
            query, user_id=_uid(user_id), token_budget=token_budget
        )
        return ctx.text or "(no relevant memories yet)"

    @mcp.tool()
    def list_memories(user_id: str = "", limit: int = 50) -> str:
        """List the most recently updated memories for a user."""
        memories = store.get_all(user_id=_uid(user_id), limit=limit)
        return json.dumps([_memory_row(m) for m in memories], ensure_ascii=False)

    @mcp.tool()
    def update_memory(memory_id: str, content: str) -> str:
        """Rewrite the content of an existing memory (e.g. after the user
        corrects a stored fact)."""
        memory = store.update(memory_id, content=content)
        if memory is None:
            return json.dumps({"error": f"memory {memory_id} not found"})
        return json.dumps(_memory_row(memory), ensure_ascii=False)

    @mcp.tool()
    def delete_memory(memory_id: str) -> str:
        """Forget a memory (soft delete: it is invalidated and kept in the
        audit history, not destroyed). Use when the user asks you to forget
        something."""
        ok = store.delete(memory_id)
        return json.dumps({"deleted": ok, "memory_id": memory_id})

    @mcp.tool()
    def memory_history(memory_id: str) -> str:
        """Show the full audit trail of a memory (ADD/UPDATE/SUPERSEDE/DELETE
        events with old and new content)."""
        events = store.history(memory_id)
        return json.dumps(
            [
                {
                    "event": e.event,
                    "old": e.old_content,
                    "new": e.new_content,
                    "reason": e.reason,
                    "at": e.created_at,
                }
                for e in events
            ],
            ensure_ascii=False,
        )

    @mcp.tool()
    def memory_stats() -> str:
        """Show memory store statistics (counts, backend, models in use)."""
        return json.dumps(store.stats(), ensure_ascii=False, default=str)

    return mcp


def main(
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 8787,
    config: Config | None = None,
) -> None:
    store = MemoryStore(config)
    server = create_server(store, host=host, port=port)
    if transport == "http":
        server.run(transport="streamable-http")
    else:
        server.run()


if __name__ == "__main__":
    main()
