"""Memry MCP server.

Exposes the memory layer to any MCP client (Claude Code, Claude Desktop,
Cursor, Windsurf, Codex, ...). ``memry mcp`` runs locally over stdio;
``memry serve`` mounts the same tools remotely at ``/mcp``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from functools import partial
from typing import Any

import anyio.to_thread
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from .config import Config
from .enrichment import EnrichmentWorker
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
- as a running checkpoint when the topic is about to change, rather than only
  at the end of the chat;
- batch related facts into ONE call as concise multiline content. Do not call
  once per sentence: Memry extracts the atomic facts itself while seeing their
  shared context. Keep unrelated topics in separate calls;
- if related facts must arrive in separate calls, reuse the same short semantic
  context label and run_id. Memry waits for two minutes of quiet, then extracts
  that group together;
- add up to three tags only when they are useful recurring retrieval subjects.
  Tags are hints; context is the temporary ingestion grouping.

With infer=true, a successful response means the exact text is durable and
searchable while enrichment is pending. Use infer=false only for content that
must always remain as one verbatim memory.

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
    if m.metadata.get("pending_distillation"):
        job = m.metadata.get("_enrichment") or {"status": "pending"}
        row["enrichment"] = {
            key: job[key]
            for key in ("status", "attempts", "next_attempt_at", "last_error")
            if key in job
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
    enrichment_worker: EnrichmentWorker | None = None,
    manage_enrichment_worker: bool = True,
) -> FastMCP:
    store = store or MemoryStore()
    enrichment_worker = enrichment_worker or EnrichmentWorker(store)

    @contextlib.asynccontextmanager
    async def _lifespan(_: FastMCP):
        task: asyncio.Task | None = None
        if manage_enrichment_worker:
            task = asyncio.create_task(enrichment_worker.run())
        try:
            yield {}
        finally:
            if task is not None:
                task.cancel()
                with contextlib.suppress(BaseException):
                    await task

    # The SDK's DNS-rebinding protection only accepts localhost-style Host
    # headers, which 421s every request arriving through a reverse proxy on a
    # public domain. That protection exists for unauthenticated localhost
    # servers; the mounted `memry serve` path applies Memry's bearer auth.
    mcp = FastMCP(
        "memry",
        instructions=INSTRUCTIONS,
        host=host,
        port=port,
        lifespan=_lifespan,
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=False
        ),
    )
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
        context: str = "",
        tags: list[str] | None = None,
        infer: bool = True,
    ) -> str:
        """Store information in long-term memory. Batch related facts into one
        concise multiline content value so Memry can extract atomic facts with
        their shared context; do not call once per sentence. If related facts
        must be sent across several calls, reuse the same semantic context label
        and run_id. Enrichment starts after that group has been quiet for two
        minutes. Optional tags are up to three suggested recurring retrieval
        subjects, not grouping identifiers.

        infer=true commits the exact text immediately and distills it in the
        managed background worker. A provider failure leaves the raw memory
        active for retry. infer=false keeps the content as one verbatim memory.

        Args:
            content: One durable fact or a concise multiline bundle of related facts.
            context: Short shared subject reused across related calls.
            tags: Up to three suggested recurring retrieval subjects.
            run_id: Stable client run identifier; reuse it for related calls.
        """
        add = store.add_deferred if infer else store.add
        shared_context = " ".join(context.split())[:200]
        tag_hints: list[str] = []
        for raw_tag in tags or []:
            tag = " ".join(str(raw_tag).strip().lower().split())[:80]
            if tag and tag not in tag_hints:
                tag_hints.append(tag)
            if len(tag_hints) == 3:
                break
        metadata: dict[str, Any] = {}
        if shared_context:
            metadata["context"] = shared_context
        if tag_hints:
            metadata["tag_hints"] = tag_hints
        kwargs: dict[str, Any] = {
            "content": content,
            "user_id": _uid(user_id),
            "agent_id": agent_id or None,
            "run_id": run_id or None,
            "metadata": metadata or None,
            "categories": tag_hints or None,
        }
        if not infer:
            kwargs["infer"] = False
        result = await _threaded(add, **kwargs)
        if infer and result.actions:
            enrichment_worker.notify()
        payload: dict[str, Any] = {
            "saved": result.summary(),
            "actions": [
                {"event": a.event, "memory_id": a.memory_id, "content": a.content}
                for a in result.actions
            ],
        }
        if infer and result.actions:
            payload["enrichment"] = {
                "status": "pending",
                "quiet_period_seconds": int(enrichment_worker.quiet_seconds),
                "memory_ids": [a.memory_id for a in result.actions if a.memory_id],
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
        entity_id: str = "",
        since: str = "",
        until: str = "",
    ) -> str:
        """Search long-term memory for what you already know. Call this at the
        start of a session AND whenever the conversation turns to a new topic,
        project, person, or decision - recall before you answer, so you don't
        contradict or re-ask what the user already told you. Returns the most
        relevant memories, best first. You can also filter by topic, entity, and date:
        restrict to categories (comma-separated), an exact entity ID, and/or a
        date window with since/until (YYYY-MM-DD, e.g. since="2026-01-01"). Pass
        an empty query with just categories or a date to browse rather than rank.

        PASS categories WHENEVER YOU KNOW THE SUBJECT. You are holding the
        conversation, so you know what it is about even when the user's words do
        not say so. Scoping to the right topic measurably beats an unfiltered
        search, and it helps most exactly where the query is vaguest ("what's
        left to do?", "where did I land on this?") - those carry no topic as
        text, so an unfiltered search has nothing to work with, while you do.
        Use the specific topic ("liver health"), not a broad area ("health").
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
            entity_id=entity_id or None,
            since=since or None,
            until=until or None,
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
    async def list_memories(
        user_id: str = "",
        limit: int = 50,
        categories: str = "",
        entity_id: str = "",
        since: str = "",
        until: str = "",
    ) -> str:
        """List memories, most recently updated first. Optionally filter by tag
        (categories, comma-separated), exact entity ID, and/or a date window
        (since/until as YYYY-MM-DD) to browse what was recorded about a topic or in a period."""
        category_list = [c.strip() for c in categories.split(",") if c.strip()] or None
        memories = await _threaded(
            store.get_all, user_id=_uid(user_id), limit=limit,
            categories=category_list, entity_id=entity_id or None,
            since=since or None, until=until or None,
        )
        return json.dumps([_memory_row(m) for m in memories], ensure_ascii=False)

    @mcp.tool()
    async def list_categories(user_id: str = "") -> str:
        """List all memory categories (tags) with their memory counts, sorted
        by count descending. Use this to see how knowledge is organized before
        drilling into a category with search_memories. Some tags are synthetic:
        higher-level themes Memry adds to cluster related tags."""
        cats = await _threaded(store.categories, user_id=_uid(user_id))
        synthetic = {
            t.tag for t in await _threaded(store.synthetic_tags, user_id=_uid(user_id))
        }
        for c in cats:
            if c["category"] in synthetic:
                c["synthetic"] = True
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
            mine = [m for m in mine if principal.owns(m.user_id)]
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


def main(config: Config | None = None) -> None:
    """Run the local stdio MCP server.

    Remote MCP is served only by ``memry serve``, which applies the configured
    network authentication and mounts these same tools at ``/mcp``.
    """
    store = MemoryStore(config)
    try:
        create_server(store).run()
    finally:
        store.close()


if __name__ == "__main__":
    main()
