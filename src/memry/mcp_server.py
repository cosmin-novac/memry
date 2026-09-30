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
from typing import Annotated, Any

import anyio.to_thread
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import CallToolResult, TextContent, ToolAnnotations
from pydantic import BaseModel, ConfigDict

from .config import Config, require_models
from .enrichment import EnrichmentWorker
from .intelligence.context import said_at
from .intelligence.when import describe_when
from .models import EventType, MemoryType, parse_said_at
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

In what comes back:
- "[happened 2023-05-07] <text> (said 8 May 2023)" (search rows: "happened",
  "said") gives when the thing happens, where known, and the day it was said.
  Do not take one for the other;
- "[until <date>]" (search rows: "invalid_at") marks a value that stopped
  holding that day; the current value is in another memory;
- "What was said" (search rows: "evidence") lists the saved turns the memories
  rest on, with date and speaker. They keep details a memory leaves out.

SAVE - call save_memories:
- whenever the user states a fact, preference, decision, correction, or plan,
  or tells you about something that happened;
- send what was said, close to the words used, one statement per line, and
  name any speaker who is not the user ("Ada: I got the job"). Keep feelings,
  advice and event details, and say what a shared photo shows. Memry extracts
  the facts itself and later shows your text as what was said, so a summary
  loses whatever it leaves out;
- as a running checkpoint when the topic is about to change, rather than only
  at the end of the chat;
- batch related facts into ONE call. Do not call once per sentence: Memry
  extracts the atomic facts itself while seeing their shared context. Keep
  unrelated topics in separate calls;
- if related facts must arrive in separate calls, reuse the same short semantic
  context label and run_id. Memry waits for two minutes of quiet, then extracts
  that group together. The label is also shown when Memry judges whether two
  names are one person or thing;
- pass said_at (YYYY-MM-DD) only for content said on another day, such as an
  import or an earlier conversation. Memry dates it that day and reads
  "yesterday" or "next month" against it;
- when something changed or the user corrects a fact, save the new statement
  as said. Memry keeps the old value as dated history, or retires it when it
  was wrong; do not delete or rewrite it yourself. Use update_memory only to
  fix a memory Memry wrote wrong, and delete_memory only when the user asks
  you to forget something;
- add up to three tags only when they are useful recurring retrieval subjects.
  Each tag becomes a topic in the user's tag list. Tags are hints; context is
  the temporary ingestion grouping.

With infer=true, a successful response means the exact text is durable and
searchable while enrichment is pending. Use infer=false only for content that
must always remain as one verbatim memory.

Never store secrets (passwords, API keys, tokens). When unsure whether a durable
fact is worth keeping, saving it is better than losing it.
"""


class MemoryEnrichmentOutput(BaseModel):
    status: str
    attempts: int | None = None
    next_attempt_at: str | None = None
    last_error: str | None = None


class EvidenceTurnOutput(BaseModel):
    said: str
    speaker: str
    text: str


class MemoryRowOutput(BaseModel):
    id: str
    content: str
    type: MemoryType
    importance: float
    categories: list[str]
    created_at: str
    updated_at: str
    said: str | None = None
    happened: str | None = None
    enrichment: MemoryEnrichmentOutput | None = None
    invalid_at: str | None = None
    score: float | None = None
    evidence: list[EvidenceTurnOutput] | None = None


class SaveActionOutput(BaseModel):
    event: EventType
    memory_id: str | None = None
    content: str | None = None


class PendingEnrichmentOutput(BaseModel):
    status: str
    quiet_period_seconds: int
    memory_ids: list[str]


class SaveMemoriesOutput(BaseModel):
    saved: dict[str, int]
    actions: list[SaveActionOutput]
    enrichment: PendingEnrichmentOutput | None = None
    warnings: list[str] | None = None


class SearchMemoriesOutput(BaseModel):
    memories: list[MemoryRowOutput]


class MemoryContextOutput(BaseModel):
    context: str
    memory_ids: list[str]
    token_estimate: int


class ListMemoriesOutput(BaseModel):
    memories: list[MemoryRowOutput]


class CategoryOutput(BaseModel):
    category: str
    count: int
    synthetic: bool | None = None


class ListCategoriesOutput(BaseModel):
    categories: list[CategoryOutput]


class UpdateMemoryOutput(BaseModel):
    memory: MemoryRowOutput | None = None
    error: str | None = None


class DeleteMemoryOutput(BaseModel):
    deleted: bool
    memory_id: str


class MemoryHistoryEventOutput(BaseModel):
    event: EventType
    old: str | None = None
    new: str | None = None
    reason: str | None = None
    at: str


class MemoryHistoryOutput(BaseModel):
    events: list[MemoryHistoryEventOutput]


class MemoryStatsData(BaseModel):
    backend: str | None = None
    note: str | None = None
    tenant: str | None = None
    db_path: str | None = None
    active_memories: int | None = None
    invalidated_memories: int | None = None
    episodes: int | None = None
    events: int | None = None
    pending_enrichments: int | None = None
    retrying_enrichments: int | None = None
    memories_by_type: dict[str, int] | None = None
    users: list[str] | None = None
    entities: int | None = None
    topics: int | None = None
    open_merge_proposals: int | None = None
    ann: dict[str, Any] | None = None
    llm: str | None = None
    embedder: str | None = None
    forgotten_memories: int | None = None
    generated_at: str | None = None

    model_config = ConfigDict(extra="allow")


class MemoryStatsOutput(BaseModel):
    stats: MemoryStatsData


def _tool_result(
    text_payload: Any,
    structured_payload: BaseModel,
    *,
    plain_text: bool = False,
) -> CallToolResult:
    """Return structured data without changing the legacy text response."""
    text = (
        str(text_payload)
        if plain_text
        else json.dumps(text_payload, ensure_ascii=False, default=str)
    )
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        structuredContent=structured_payload.model_dump(mode="json", exclude_none=True),
    )


def _memory_row(m: Any, score: float | None = None, evidence: Any = ()) -> dict[str, Any]:
    """A memory as the tools return it. "said" is the day it was recorded
    (its last change; for one out of use, such as an update's old value kept
    as history, the day it began to hold, and ``invalid_at`` the day it held
    until) and "happened" when the thing it tells happens, where known: the
    dates the context builder labels (``context.memory_line``). "evidence"
    are the source turns a search chose for it (``MemoryStore.evidence``)."""
    row = {
        "id": m.id,
        "content": m.content,
        "type": m.memory_type,
        "importance": m.importance,
        "categories": m.categories,
        "created_at": m.created_at,
        "updated_at": m.updated_at,
        "said": (said_at(m) or "")[:10],
    }
    happened = describe_when((m.metadata or {}).get("when"))
    if happened:
        row["happened"] = happened
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
    if evidence:
        row["evidence"] = [{"said": t.said_at[:10], "speaker": t.speaker, "text": t.content}
                           for t in evidence]
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

    @mcp.tool(
        annotations=ToolAnnotations(
            readOnlyHint=False,
            openWorldHint=False,
            destructiveHint=True,
        )
    )
    async def save_memories(
        content: str,
        user_id: str = "",
        agent_id: str = "",
        run_id: str = "",
        context: str = "",
        tags: list[str] | None = None,
        infer: bool = True,
        said_at: str = "",
    ) -> Annotated[CallToolResult, SaveMemoriesOutput]:
        """Store what was said in long-term memory. Send it close to the words
        used, one statement per line, and name any speaker who is not the user
        ("Ada: I got the job"). Memry extracts the facts itself and keeps this
        text as the turn it shows with them later ("evidence"), so a summary
        loses what it leaves out. Batch related statements into one call; do
        not call once per sentence. If related statements must be sent across
        several calls, reuse the same semantic context label and run_id.
        Enrichment starts after that group has been quiet for two minutes.
        Optional tags are up to three suggested recurring retrieval subjects,
        not grouping identifiers.

        A change or a correction is saved as said, like anything else: Memry
        keeps the old value as dated history, or retires it when it was wrong.

        infer=true commits the exact text immediately and distills it in the
        managed background worker. A provider failure leaves the raw memory
        active for retry. infer=false keeps the content as one verbatim memory.

        Args:
            content: What was said, one statement per line, naming any
                speaker who is not the user.
            context: Short shared subject reused across related calls.
            tags: Up to three suggested recurring retrieval subjects.
            run_id: Stable client run identifier; reuse it for related calls.
            said_at: The day the content was said (YYYY-MM-DD, or an ISO date
                and time, read in UTC), only for content said on another day,
                such as an import or an earlier conversation. The memories are
                dated that day and "yesterday" or "next month" is read against
                it. Leave it empty for what is said now; a day after today is
                an error.
        """
        said = parse_said_at(said_at)  # a malformed or future value is a tool error
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
        if said is not None:
            # the save's time, and the day extraction reads "yesterday" against
            kwargs["created_at"] = said.isoformat(timespec="seconds")
            kwargs["now"] = said
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
        return _tool_result(payload, SaveMemoriesOutput.model_validate(payload))

    @mcp.tool(
        annotations=ToolAnnotations(
            readOnlyHint=True,
            openWorldHint=False,
            destructiveHint=False,
        )
    )
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
        when_since: str = "",
        when_until: str = "",
    ) -> Annotated[CallToolResult, SearchMemoriesOutput]:
        """Search long-term memory for what you already know. Call this at the
        start of a session AND whenever the conversation turns to a new topic,
        project, person, or decision - recall before you answer, so you don't
        contradict or re-ask what the user already told you. Returns the most
        relevant memories, best first. You can also filter by topic, entity, and date:
        restrict to categories (comma-separated), an exact entity ID, and/or a
        date window with since/until (YYYY-MM-DD, e.g. since="2026-01-01"). Pass
        an empty query with just categories or a date to browse rather than rank.

        when_since/when_until (YYYY-MM-DD) filter on when the thing itself
        happens rather than when it was saved, which is what answers "what is on
        this weekend"; only memories that carry an occurrence time match.

        Each row carries "said" (the day it was said), "happened" (when the
        thing happens, where known; do not read "said" as that day),
        "invalid_at" on a value kept as history (it held until then; the
        current value is another memory), and "evidence": the saved turns it
        rests on (said, speaker, text), which keep details the memory leaves
        out.

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
            when_since=when_since or None,
            when_until=when_until or None,
        )
        memory_rows = [_memory_row(r.memory, r.score, r.evidence) for r in results]
        return _tool_result(
            memory_rows,
            SearchMemoriesOutput(memories=memory_rows),
        )

    @mcp.tool(
        annotations=ToolAnnotations(
            readOnlyHint=False,
            openWorldHint=False,
            destructiveHint=False,
        )
    )
    async def get_memory_context(
        query: str,
        user_id: str = "",
        token_budget: int = 1200,
    ) -> Annotated[CallToolResult, MemoryContextOutput]:
        """Get a ready-to-use context block of the most relevant memories for
        the current subject, packed to fit the given token budget. Prefer this
        over search_memories when you just want background injected before you
        answer. Worth calling at the start of a session and whenever a new topic
        comes up. This can refresh and persist a derived entity summary when the
        stored summary is stale; it never changes the underlying memories.

        A memory reads "[happened <date>] <text> (said <date>)": when the
        thing happens, where known, and the day it was said. "[until <date>]"
        marks a value that stopped holding that day. Under "What was said" are
        the saved turns the memories rest on, with date and speaker."""
        ctx = await _threaded(
            store.reconstruct_context,
            query=query, user_id=_uid(user_id), token_budget=token_budget,
        )
        text = ctx.text or "(no relevant memories yet)"
        return _tool_result(
            text,
            MemoryContextOutput(
                context=text,
                memory_ids=ctx.memory_ids,
                token_estimate=ctx.token_estimate,
            ),
            plain_text=True,
        )

    @mcp.tool(
        annotations=ToolAnnotations(
            readOnlyHint=True,
            openWorldHint=False,
            destructiveHint=False,
        )
    )
    async def list_memories(
        user_id: str = "",
        limit: int = 50,
        categories: str = "",
        entity_id: str = "",
        since: str = "",
        until: str = "",
        when_since: str = "",
        when_until: str = "",
    ) -> Annotated[CallToolResult, ListMemoriesOutput]:
        """List memories, most recently updated first. Optionally filter by tag
        (categories, comma-separated), exact entity ID, and/or a date window
        (since/until as YYYY-MM-DD) to browse what was recorded about a topic or in a period.
        when_since/when_until (YYYY-MM-DD) filter instead on when the thing
        itself happens, and only reach memories that carry an occurrence time."""
        category_list = [c.strip() for c in categories.split(",") if c.strip()] or None
        memories = await _threaded(
            store.get_all, user_id=_uid(user_id), limit=limit,
            categories=category_list, entity_id=entity_id or None,
            since=since or None, until=until or None,
            when_since=when_since or None, when_until=when_until or None,
        )
        memory_rows = [_memory_row(m) for m in memories]
        return _tool_result(
            memory_rows,
            ListMemoriesOutput(memories=memory_rows),
        )

    @mcp.tool(
        annotations=ToolAnnotations(
            readOnlyHint=True,
            openWorldHint=False,
            destructiveHint=False,
        )
    )
    async def list_categories(
        user_id: str = "",
    ) -> Annotated[CallToolResult, ListCategoriesOutput]:
        """List all memory categories (tags) with their memory counts, sorted
        by count descending. Use this to see how knowledge is organized before
        drilling into a category with search_memories. Each count is the
        memories filed directly under that tag."""
        cats = await _threaded(store.categories, user_id=_uid(user_id))
        synthetic = {
            t.tag for t in await _threaded(store.synthetic_tags, user_id=_uid(user_id))
        }
        for c in cats:
            if c["category"] in synthetic:
                c["synthetic"] = True
        return _tool_result(cats, ListCategoriesOutput(categories=cats))

    @mcp.tool(
        annotations=ToolAnnotations(
            readOnlyHint=False,
            openWorldHint=False,
            destructiveHint=True,
        )
    )
    async def update_memory(
        memory_id: str,
        content: str,
    ) -> Annotated[CallToolResult, UpdateMemoryOutput]:
        """Rewrite the text of a memory Memry wrote wrong, such as a misread
        name or number. The new text replaces the old one and is dated today;
        the old text stays only in memory_history. When something changed, or
        the user corrects what they said, call save_memories with the new
        statement instead: Memry then keeps the old value as dated history or
        retires it as wrong."""
        memory = await _threaded(
            store.update,
            memory_id=memory_id,
            content=content,
            owner_prefix=_principal().prefix,
        )
        if memory is None:
            error = {"error": f"memory {memory_id} not found"}
            return _tool_result(error, UpdateMemoryOutput(**error))
        memory_row = _memory_row(memory)
        return _tool_result(memory_row, UpdateMemoryOutput(memory=memory_row))

    @mcp.tool(
        annotations=ToolAnnotations(
            readOnlyHint=False,
            openWorldHint=False,
            destructiveHint=True,
        )
    )
    async def delete_memory(
        memory_id: str,
    ) -> Annotated[CallToolResult, DeleteMemoryOutput]:
        """Forget a memory when the user asks you to forget something (soft
        delete: it leaves search and context, its saved turns are no longer
        shown, and it is kept in the audit history and can be brought back).
        Do not delete a memory because it changed or was wrong: save the new
        statement with save_memories, and Memry keeps or retires the old one."""
        ok = await _threaded(
            store.delete, memory_id=memory_id, owner_prefix=_principal().prefix
        )
        payload = {"deleted": ok, "memory_id": memory_id}
        return _tool_result(payload, DeleteMemoryOutput(**payload))

    @mcp.tool(
        annotations=ToolAnnotations(
            readOnlyHint=True,
            openWorldHint=False,
            destructiveHint=False,
        )
    )
    async def memory_history(
        memory_id: str,
    ) -> Annotated[CallToolResult, MemoryHistoryOutput]:
        """Show the full audit trail of a memory (ADD/UPDATE/SUPERSEDE/DELETE
        events with old and new content)."""
        events = await _threaded(
            store.history, memory_id=memory_id, owner_prefix=_principal().prefix
        )
        event_rows = [
            {
                "event": e.event,
                "old": e.old_content,
                "new": e.new_content,
                "reason": e.reason,
                "at": e.created_at,
            }
            for e in events
        ]
        return _tool_result(
            event_rows,
            MemoryHistoryOutput(events=event_rows),
        )

    @mcp.tool(
        annotations=ToolAnnotations(
            readOnlyHint=True,
            openWorldHint=False,
            destructiveHint=False,
        )
    )
    async def memory_stats() -> Annotated[CallToolResult, MemoryStatsOutput]:
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
            stats = {
                "tenant": principal.name,
                "active_memories": sum(1 for m in mine if m.invalid_at is None),
                "invalidated_memories": sum(
                    1 for m in mine if m.invalid_at is not None
                ),
            }
        else:
            stats = await _threaded(store.stats)
        return _tool_result(stats, MemoryStatsOutput(stats=stats))

    return mcp


def main(config: Config | None = None) -> None:
    """Run the local stdio MCP server.

    Remote MCP is served only by ``memry serve``, which applies the configured
    network authentication and mounts these same tools at ``/mcp``.
    """
    config = config or Config.load()
    require_models(config)
    store = MemoryStore(config)
    try:
        create_server(store).run()
    finally:
        store.close()


if __name__ == "__main__":
    main()
