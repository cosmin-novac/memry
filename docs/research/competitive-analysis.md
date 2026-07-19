# Competitive analysis - AI agent memory systems

*Research date: 2026-07-17. Sources: arXiv 2504.19413 (Mem0 paper), vendor sites/docs, GitHub repos.*

This document surveys the memory-layer landscape that Memry competes in, catalogs the
features each system offers, and derives the feature set we consider most valuable - the
basis for Memry's v0.1 scope and its research roadmap.

---

## 1. System-by-system survey

### Mem0 (mem0ai/mem0) - the reference point

- **Paper (arXiv 2504.19413):** two-phase pipeline - *extraction* (LLM distills salient facts
  from conversation) and *update* (LLM reconciles each fact against similar existing memories
  with ADD / UPDATE / DELETE / NOOP operations). A graph variant (Mem0-g) stores
  entity-relation triples. Results vs. baselines on LOCOMO: +26% LLM-as-a-Judge over OpenAI
  memory, 91% lower p95 latency and >90% token savings vs. full-context.
- **Product (2026):** the algorithm has since evolved - single-pass **ADD-only extraction**
  (UPDATE/DELETE removed from the hot path) with **multi-signal retrieval**: semantic
  vectors + BM25 keyword + entity linking + temporal reasoning at query time
  (current-state vs. historical-event queries). Reported: LoCoMo 92.5, LongMemEval 94.4.
- **API:** `add / search / get_all / update / delete / history / reset`; scoping by
  `user_id / agent_id / run_id`; `infer=False` stores verbatim (bypass extraction).
- **Stack:** Python + TypeScript SDKs, pluggable LLMs/embedders/vector stores (Qdrant
  default), OpenMemory MCP server, managed platform + OSS (Apache-2.0, ~61k stars).
- **OSS gaps (per their own platform-vs-OSS docs):** temporal reasoning and memory decay are
  platform-only; analytics, export, webhooks limited in OSS; published benchmark numbers
  include proprietary platform optimizations.

### Zep - temporal knowledge graphs

- Core: **Graphiti** temporal knowledge graph. Entities + facts as edges with
  **valid_at / invalid_at** windows; new contradicting facts *invalidate* (not delete) old
  ones, preserving an audit trail.
- **Provenance:** every fact traces to its source episode (conversation turn or ingested
  record).
- "Observations": pattern mining over the graph (recurring behavior, co-occurrence).
- Sub-200ms retrieval claim at 10K-100M entities; LoCoMo 94.7% at ~5.7K context tokens.
- Enterprise: ABAC policies, retention schedules, legal holds, SOC2/HIPAA; Cloud/BYOK/BYOC.
- Takeaway: **temporal validity + provenance is the defensible idea** - memory as a
  bi-temporal record, not a mutable KV store.

### Letta (MemGPT) - the agent-OS view

- Memory as **virtual context management**: self-editing *memory blocks* in-context
  (persona/human/custom), archival memory (vector store), recall memory (conversation
  search). The agent edits its own memory via tools.
- Newer directions: **MemFS** (git-tracked memory filesystem), *sleep-time compute*
  (consolidation during idle), continual learning, shared memory blocks across agents.
- Takeaway: memory as **agent-editable state** and **offline consolidation** are powerful
  ideas; heavyweight because it owns the whole agent runtime, not just memory.

### Supermemory - memory-as-context-engineering platform

- Custom vector-graph engine with "ontology-aware edges"; hybrid vector+keyword retrieval,
  sub-300ms.
- **Connectors** (Notion, Google Drive, Gmail, S3) + multimodal extractors (PDF, images,
  audio) - memory populated from a whole digital footprint, not just chat.
- **User-profile synthesis:** builds an evolving profile (intent, preferences) from
  accumulated memories.
- TS/Python SDKs, REST, MCP server card. Closed-source SaaS.

### claude-mem - session-lifecycle capture for coding agents

- Hooks into Claude Code (and Codex/Gemini/Copilot/OpenCode…) session lifecycle: captures
  everything the agent did, **compresses transcripts with an LLM**, stores in local SQLite +
  Chroma, and **injects relevant context into future sessions** automatically.
- 4 MCP tools with a token-efficient layered search workflow; multi-machine sync over SSH;
  "Endless Mode" compresses tool outputs into ~500-token observations mid-session.
- ~46k stars. Takeaway: **zero-effort automatic capture + automatic injection** is the UX
  bar for developer agents; local-first SQLite is a proven, popular choice.

### EverMind (EverOS / EverMemOS) - memory operating system

- Open-source "memory OS": **episodic / semantic / procedural** memory taxonomy;
  *Skill Memory* - successful agent trajectories distilled into reusable procedures
  (self-evolving skills); offline consolidation; multi-level scoping (personal / group /
  agent); markdown-native, local-first, user-owned.
- mRAG hybrid retrieval, claims 93%+ accuracy, <500ms p95, ~10× lower cost. Cloud API:
  memories/groups/senders/tasks/storage endpoints; multimodal ingestion.
- Takeaway: the **memory-type taxonomy** and **procedural/skill memory** are the interesting
  research directions.

### Recall (Recall MCP) - local-first shared memory over MCP

- Fully local persistent memory over MCP: SQLite storage, **100% offline embeddings**
  (MiniLM-L6-v2, no API keys), hybrid search, Bearer-auth SSE server so **multiple agents
  (Claude/Gemini/Codex) share one memory DB** simultaneously.
- **Memory decay / GC** with configurable STRONG → MEDIUM → WEAK rules; automatic clustering
  and merging of similar memories (originals kept as history); named workflow threads
  spanning sessions; survives context compaction.
- Takeaway: **works-offline, multi-agent-shared, decays gracefully** - the strongest
  self-hosting story in the field.

### MemSync (OpenGradient) - cross-app universal memory

- "Plaid for memory": one memory layer across ChatGPT, Claude, Perplexity, etc. (browser
  extension + API). Extracts facts with verifiable LLM inference; classifies **semantic vs.
  episodic**; auto-generates user profiles and insights; sync path with last-write-wins +
  vector-clock conflict resolution; invalidation path marks stale memories; E2E encryption
  claims. a16z crypto-backed.
- Takeaway: **cross-platform continuity** is the consumer wedge; the MCP-server approach
  gets the same effect for agent tooling without a browser extension.

### Contextberg - passive local capture

- Windows-local app that **watches screens, browser, and agent transcripts** in the
  background and serves it all to coding agents via MCP. Three memory tiers: activity
  (what you did), daily (grouped by date), long-term. Fully offline with a local LLM.
- Takeaway: **passive capture** (vs. explicit save) is a distinct paradigm; privacy-first
  local processing resonates.

---

## 2. Feature matrix

| Feature | Mem0 | Zep | Letta | Supermemory | claude-mem | EverMind | Recall | MemSync | Contextberg |
|---|---|---|---|---|---|---|---|---|---|
| LLM fact extraction | ✅ | ✅ | ✅ (self-edit) | ✅ | ✅ (compression) | ✅ | ➖ | ✅ | ✅ (local LLM) |
| Reconciliation (update/contradiction) | ✅ (v3: retrieval-side) | ✅ invalidation | ✅ self-edit | ➖ | ➖ | ✅ | ✅ merge/cluster | ✅ invalidate | ➖ |
| Temporal validity (valid/invalid_at) | platform only | ✅ core idea | ➖ | ➖ | ➖ | partial | ➖ | partial | ➖ |
| Provenance (fact → source episode) | partial | ✅ | ➖ | ➖ | ✅ | ✅ | ➖ | ➖ | ✅ |
| Hybrid retrieval (vector+BM25+…) | ✅ | ✅ graph | vector | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Recency/importance ranking | ✅ temporal | ✅ | ➖ | ✅ | ✅ | ✅ | ✅ decay | ✅ | ✅ |
| Scoping (user/agent/session) | ✅ | ✅ | ✅ | ✅ | project | ✅ | ✅ threads | ✅ | project |
| Memory types (semantic/episodic/procedural) | ➖ | ➖ | ✅ blocks | ➖ | ➖ | ✅ | ➖ | ✅ | ✅ tiers |
| Graph/relations | ✅ optional | ✅ core | ➖ | ✅ edges | ➖ | ✅ | ➖ | ➖ | ➖ |
| MCP server | ✅ OpenMemory | ➖ | ➖ | ✅ | ✅ | ✅ | ✅ core | ➖ | ✅ core |
| Works with zero API keys (offline) | ➖ | ➖ | ➖ | ➖ | partial | ✅ | ✅ | ➖ | ✅ |
| Self-hostable OSS | ✅ Apache-2.0 | BYOC only | ✅ | ➖ | ✅ | ✅ | ✅ | ➖ | local app |
| Decay / forgetting | platform only | retention | ➖ | ➖ | ➖ | ✅ | ✅ | ✅ | ➖ |
| Multi-agent shared memory | ✅ | ✅ | ✅ blocks | ✅ | ✅ | ✅ groups | ✅ | ✅ | ✅ |
| Audit history | ✅ history() | ✅ | git (MemFS) | ➖ | ✅ | ✅ | ✅ | ➖ | ✅ |
| Profile synthesis | ➖ | observations | persona block | ✅ | ➖ | ➖ | ➖ | ✅ | ➖ |
| Connectors / multimodal ingestion | ✅ | ✅ business data | ➖ | ✅ strong | ➖ | ✅ | ➖ | ➖ | screen capture |
| Sleep-time / offline consolidation | ➖ | ➖ | ✅ | ➖ | ✅ | ✅ | ✅ GC | ➖ | ✅ |
| Eval framework published | paper only | paper only | ✅ research | ➖ | ➖ | ✅ | ➖ | claims only | ➖ |

---

## 3. Most valuable features (ranked)

1. **LLM extraction + reconciliation** (Mem0's core loop) - turning raw dialogue into
   deduplicated, contradiction-aware facts is *the* product. Everything else is plumbing.
2. **Hybrid retrieval** (vector + BM25 + recency + importance) - every serious system
   converged on this; pure vector search is not competitive.
3. **Temporal validity + provenance** (Zep) - invalidate, don't delete; every fact traces
   to its source episode. Mem0-OSS lacks this → differentiation opportunity.
4. **MCP-native** - the distribution channel. One server, every agent (Claude Code, Cursor,
   Windsurf, Codex…). Recall/claude-mem/Contextberg prove the demand.
5. **Local-first, zero-key operation** (Recall/EverMind/Contextberg) - self-hosting that
   *actually works out of the box*: SQLite, keyword search without embeddings, local
   embedding fallback. Mem0 requires an LLM + vector DB to do anything.
6. **Scoping** (user/agent/run) + multi-agent shared memory.
7. **Decay/forgetting** - importance × recency decay with soft-invalidation (Recall). Also
   platform-only in Mem0 → opportunity.
8. **Audit history** (`history()`, events) - table stakes for trust and debugging.
9. **Evaluation framework** - nobody ships a usable one in OSS. For a research-grade
   codebase this is the moat: LongMemEval/LoCoMo/RULER harness + own workloads.
10. **Memory-type taxonomy** (semantic/episodic/procedural/working) - cheap to model now,
    enables routing research later.
11. Profile synthesis, graph memory, connectors, passive capture, sleep-time consolidation -
    valuable but v2+; graph memory bought Mem0 only ~2% on benchmarks at high complexity cost.

## 4. Implications for Memry

**Own the API; make backends replaceable.** Applications call `MemoryStore`
(add/search/get_all/update/delete/history/reconstruct_context). The default backend is our
local SQLite engine ("storage you control": raw episodes + derived memories + events + FTS5 +
vectors in one file, no services). A **Mem0 adapter** (optional extra, `infer=False` mode so
our intelligence layer stays ours) provides interop, a migration path, and - importantly for
the research agenda - a baseline to benchmark against inside the same eval harness.

**v0.1 scope** = items 1-10 above: extraction → reconciliation (ADD/UPDATE/DELETE-as-
supersede/NONE) with pluggable LLMs, hybrid retrieval with RRF + recency + importance,
temporal invalidation + provenance + full event history, memory types, scoping, decay sweep,
MCP server (stdio + streamable HTTP), REST + dashboard, zero-key mode (FTS5 + hash
embeddings), eval harness with a synthetic dataset and loaders for public benchmarks.

