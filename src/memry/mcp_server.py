"""Memry MCP server.

Exposes the memory layer to any MCP client (Claude Code, Claude Desktop,
Cursor, Windsurf, Codex, ...) over stdio or streamable HTTP.

    memry mcp                      # stdio (for local agent config)
    memry mcp --transport http     # streamable HTTP on :8787/mcp
"""

from __future__ import annotations

import json
from functools import partial
from typing import Any

import anyio.to_thread
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from .config import Config
from .principal import ADMIN, Principal
from .store import MemoryStore

# ASGI scope key the HTTP server uses to hand the authenticated identity to
# the tools below. See memry.rest.create_app.
PRINCIPAL_SCOPE_KEY = "memry.principal"

INSTRUCTIONS = """Memry is your long-term memory across sessions. Treat it as a
working habit, not a filing cabinet you visit at the end: recall before you
reason, and save as you learn. Following this loop is what makes you feel like
you remember the user instead of meeting them fresh each time.

RECALL - call get_memory_context (or search_memories) with a short description
of the subject:
- at the START of a session, before your first substantive answer;
- WHENEVER the conversation turns to a new topic, project, person, tool, or
  decision. The moment a new subject comes up, check what you already know
  about it before responding. This is the most important habit: a new topic is
  the trigger to recall.
- when the user implies prior context ("as I mentioned", "my usual setup",
  "the project"). Recall is cheap and stops you contradicting or re-asking what
  you were already told.

SAVE - call save_memories:
- whenever the user states a stable fact, preference, decision, correction, or
  plan - capture it in their own words;
- as a running checkpoint, not just at the end: once you have learned a handful
  of new facts, or when the topic is about to change, save what is worth keeping
  before moving on. A good rhythm is "recall on a new topic, save when leaving
  one."
- Prefer several focused calls (one topic each) over one giant multi-topic dump,
  and ALWAYS check the response's "warnings" field: it lists input details that
  did not survive distillation, so you can save them explicitly (or retry with
  infer=false for must-keep-verbatim content).

Never store secrets (passwords, API keys, tokens). When unsure whether a durable
fact is worth keeping, saving it is better than losing it.
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
    # The SDK's DNS-rebinding protection only accepts localhost-style Host
    # headers, which 421s every request arriving through a reverse proxy on a
    # public domain. That protection exists for unauthenticated localhost
    # servers; Memry's HTTP transport carries its own bearer auth.
    mcp = FastMCP(
        "memry",
        instructions=INSTRUCTIONS,
        host=host,
        port=port,
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=False
        ),
    )
    store = store or MemoryStore()
    default_user = store.config.default_user_id

    def _principal() -> Principal:
        """Who the call in flight acts as.

        The HTTP transport attaches the Starlette request to every JSON-RPC
        message (ServerMessageMetadata.request_context), so this reaches the
        tool body across the session task boundary that a plain contextvar
        would not survive. stdio has no request and no auth: that is the local
        single-user case, which stays admin.
        """
        try:
            request = mcp.get_context().request_context.request
        except (ValueError, AttributeError, LookupError):
            return ADMIN
        scope = getattr(request, "scope", None) or {}
        return scope.get(PRINCIPAL_SCOPE_KEY) or ADMIN

    def _uid(user_id: str) -> str | None:
        """Namespace for a tool argument.

        The argument is never trusted as an identity: for a confined principal
        it can only ever select a sub-namespace of that principal's own space,
        so passing someone else's user_id lands in your own namespace rather
        than reaching theirs.
        """
        return _principal().namespace(user_id or default_user)

    # Store calls are synchronous (SQLite + provider HTTP). FastMCP runs sync
    # tools directly on the event loop, so every tool below is async and hops
    # to a worker thread - one slow LLM call must not stall the whole server.
    async def _threaded(fn, /, **kwargs):
        return await anyio.to_thread.run_sync(partial(fn, **kwargs))

    @mcp.tool()
    async def save_memories(
        content: str,
        user_id: str = "",
        agent_id: str = "",
        run_id: str = "",
        infer: bool = True,
    ) -> str:
        """Store information in long-term memory. Call this whenever the user
        shares a stable fact, preference, decision, correction, or plan - and
        also as a periodic checkpoint, once you have learned several new facts
        or the topic is changing, rather than only at the end of the chat. With
        infer=true (default) the text is distilled into discrete facts and
        reconciled against existing memories (duplicates skipped, contradictions
        superseded). With infer=false the text is saved verbatim as one memory.
        Prefer focused, one-topic calls over a single multi-topic dump.
        """
        result = await _threaded(
            store.add,
            content=content,
            user_id=_uid(user_id),
            agent_id=agent_id or None,
            run_id=run_id or None,
            infer=infer,
        )
        payload: dict[str, Any] = {
            "saved": result.summary(),
            "actions": [
                {"event": a.event, "memory_id": a.memory_id, "content": a.content}
                for a in result.actions
            ],
        }
        if result.warnings:
            payload["warnings"] = result.warnings
        return json.dumps(payload, ensure_ascii=False)

    @mcp.tool()
    async def search_memories(
        query: str,
        user_id: str = "",
        agent_id: str = "",
        run_id: str = "",
        limit: int = 8,
        categories: str = "",
    ) -> str:
        """Search long-term memory for what you already know. Call this at the
        start of a session AND whenever the conversation turns to a new topic,
        project, person, or decision - recall before you answer, so you don't
        contradict or re-ask what the user already told you. Returns the most
        relevant memories, best first. Optionally restrict to categories
        (comma-separated, e.g. "diet,health").
        """
        category_list = [c.strip() for c in categories.split(",") if c.strip()] or None
        results = await _threaded(
            store.search,
            query=query,
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
    async def get_memory_context(
        query: str,
        user_id: str = "",
        token_budget: int = 1200,
    ) -> str:
        """Get a ready-to-use context block of the most relevant memories for
        the current subject, packed to fit the given token budget. Prefer this
        over search_memories when you just want background injected before you
        answer. Worth calling at the start of a session and whenever a new topic
        comes up."""
        ctx = await _threaded(
            store.reconstruct_context,
            query=query, user_id=_uid(user_id), token_budget=token_budget,
        )
        return ctx.text or "(no relevant memories yet)"

    @mcp.tool()
    async def list_memories(user_id: str = "", limit: int = 50) -> str:
        """List the most recently updated memories for a user."""
        memories = await _threaded(store.get_all, user_id=_uid(user_id), limit=limit)
        return json.dumps([_memory_row(m) for m in memories], ensure_ascii=False)

    @mcp.tool()
    async def list_categories(user_id: str = "") -> str:
        """List all memory categories (tags) with their memory counts, sorted
        by count descending. Use this to see how knowledge is organized before
        drilling into a category with search_memories."""
        cats = await _threaded(store.categories, user_id=_uid(user_id))
        return json.dumps(cats, ensure_ascii=False)

    @mcp.tool()
    async def update_memory(memory_id: str, content: str) -> str:
        """Rewrite the content of an existing memory (e.g. after the user
        corrects a stored fact)."""
        memory = await _threaded(
            store.update,
            memory_id=memory_id,
            content=content,
            owner_prefix=_principal().prefix,
        )
        if memory is None:
            return json.dumps({"error": f"memory {memory_id} not found"})
        return json.dumps(_memory_row(memory), ensure_ascii=False)

    @mcp.tool()
    async def delete_memory(memory_id: str) -> str:
        """Forget a memory (soft delete: it is invalidated and kept in the
        audit history, not destroyed). Use when the user asks you to forget
        something."""
        ok = await _threaded(
            store.delete, memory_id=memory_id, owner_prefix=_principal().prefix
        )
        return json.dumps({"deleted": ok, "memory_id": memory_id})

    @mcp.tool()
    async def memory_history(memory_id: str) -> str:
        """Show the full audit trail of a memory (ADD/UPDATE/SUPERSEDE/DELETE
        events with old and new content)."""
        events = await _threaded(
            store.history, memory_id=memory_id, owner_prefix=_principal().prefix
        )
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
    async def memory_stats() -> str:
        """Show memory store statistics (counts, backend, models in use)."""
        principal = _principal()
        if principal.prefix is not None:
            # Global counts would leak the size of other namespaces, so a
            # confined principal gets counts over its own space only.
            mine = await _threaded(
                store.get_all,
                user_id=None,
                include_invalid=True,
                limit=100_000,
            )
            mine = [m for m in mine if (m.user_id or "").startswith(principal.prefix)]
            return json.dumps(
                {
                    "tenant": principal.name,
                    "active_memories": sum(1 for m in mine if m.invalid_at is None),
                    "invalidated_memories": sum(
                        1 for m in mine if m.invalid_at is not None
                    ),
                },
                ensure_ascii=False,
            )
        stats = await _threaded(store.stats)
        return json.dumps(stats, ensure_ascii=False, default=str)

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
