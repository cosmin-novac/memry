# Self-hosting Memry

Memry is one Python process with SQLite and no external database, queue, or vector service.
Knowledge lives in `memry.db`; runtime accounts and OAuth live in the adjacent `auth.db`.
Keeping them separate prevents knowledge restore/reset operations from changing login data.
A complete server backup must include both files.

## Option 1 - bare (recommended for personal use)

```bash
pip install memry
memry serve --host 0.0.0.0 --port 8787
```

- Dashboard: `http://<host>:8787/`
- REST API: `http://<host>:8787/api/v1/...`
- MCP (streamable HTTP): `http://<host>:8787/mcp`
- Knowledge data: `~/.memry/memry.db` (override with `MEMRY_DB_PATH`)
- Login data: `~/.memry/auth.db` when accounts/OAuth are used (override with `MEMRY_AUTH_DB_PATH`)

## Option 2 - Docker

```bash
docker compose up -d --build
```

The compose file mounts a named volume at `/data` and reads the same `MEMRY_*`
environment variables. See [`docker-compose.yml`](../docker-compose.yml).

Docker automatically reuses the package layer for source-only updates. A first build or
a change to `requirements-docker.txt` installs all dependencies; normal code and dashboard
updates install only Memry itself. To deliberately refresh every package:

```bash
docker compose build --no-cache memry
docker compose up -d memry
```

## Option 3 - VPS, one command (Docker + automatic HTTPS)

On a fresh Ubuntu/Debian server at any provider (Contabo, Hetzner,
DigitalOcean, ...):

```bash
curl -fsSL https://raw.githubusercontent.com/cosmin-novac/memry/main/deploy/install.sh \
  | MEMRY_DOMAIN=memory.example.com bash
```

Installs Docker, builds Memry, puts Caddy in front for automatic HTTPS, and
generates a `MEMRY_API_KEY`. Re-run the same command to update. Full
walkthrough (cloud-init, DNS, backups, uninstall): [deploy-vps.md](deploy-vps.md).

## Securing the server

1. **Set an API key** - `MEMRY_API_KEY=<random>` requires
   `Authorization: Bearer <key>` on `/api/*`. Create the first runtime account for the human
   administrator. Every dashboard user signs in at `/login` with an account name and password;
   the HttpOnly session cookie remains confined to that account's memories. Programmatic and
   recovery clients keep using the operator bearer key.
2. **Bind privately** - without a key, tenants or accounts every request is treated as admin, so
   `memry serve` refuses to bind anything but loopback (`127.0.0.1`, `localhost`, `::1`) and
   tells you why. If a private network or a reverse proxy that does its own auth
   (Caddy/Traefik/nginx) really protects the port, set `MEMRY_ALLOW_OPEN=1` to override.
3. **Backups** - a complete server backup must capture `memry.db` and `auth.db`
   together, including any live SQLite `-wal`/`-shm` files. A directory/volume snapshot
   does that. `memry export` is a lossless knowledge backup, but it does not include
   accounts, sessions, OAuth clients, or tokens from `auth.db`.

## Multi-tenant mode

Serve several teams or customers from one Memry server, each with their own API key and an
isolated memory space:

```bash
export MEMRY_API_KEY="admin-key-with-global-access"     # optional but recommended
export MEMRY_TENANTS='[{"name":"acme","api_key":"acme-secret"},
                       {"name":"globex","api_key":"globex-secret"}]'
memry serve --host 0.0.0.0
```

(or the same under `"tenants": [...]` in `~/.memry/config.json`.)

How isolation works:

- A tenant's requests are transparently namespaced: user `u1` under key `acme-secret`
  reads and writes `acme::u1`. Tenants never see, guess, or address each other's data;
  cross-tenant access by memory/entity id returns 404.
- `GET /api/v1/stats` returns per-tenant counts for tenant keys, global stats for the
  admin key.
- MCP over HTTP (`/mcp`) accepts tenant keys too, with the same confinement: a tool's
  `user_id` argument selects a namespace *under* the calling tenant, so asking for
  another tenant's namespace lands in your own rather than reaching theirs. `/mcp/<key>`
  works for tenant keys as well as the admin key. Local stdio MCP (`memry mcp`) has no
  auth and stays single-user.

Keys live in config on your infrastructure; treat the config file like a secret.

Tenants are fixed in config. For a server where people sign themselves up and connect from
any MCP client, use **accounts** instead.

## Accounts and OAuth

Accounts are runtime-managed identities (not config). The first account is the bootstrap
administrator and keeps only the server's existing `default` memory space. Every later
account gets one private `<account>::default` space. The administrator role does not grant
access to other accounts' memories. Accounts are reachable either with an API key or through
a real OAuth login from clients like Claude Code, Cursor, or VS Code. `MEMRY_API_KEY` remains
the separate operator credential with global access and should be protected accordingly.

Create accounts with the CLI:

```bash
memry account add alice --password s3cret   # prints an API key (shown once)
memry account list
memry account issue-key alice --label laptop
memry account disable alice                 # keys and tokens stop working immediately
```

Accounts live in `auth.db` next to your memory database (override with `MEMRY_AUTH_DB_PATH`).
Back it up together with `memry.db`; a knowledge export alone cannot restore accounts or
OAuth state. An account's API key works on both `/api` and `/mcp`, including the
`/mcp/<key>` URL form.

**OAuth.** Set a public URL and Memry becomes an OAuth 2.1 authorization server for its own
accounts:

```bash
export MEMRY_PUBLIC_URL="https://memory.example.com"
memry serve --host 0.0.0.0
```

That turns on, at the domain root:

- `/.well-known/oauth-authorization-server` and
  `/.well-known/oauth-protected-resource/mcp` - discovery documents clients read first.
- `/register` (Dynamic Client Registration), `/authorize`, `/token`, `/revoke` - the flow,
  with PKCE required and refresh-token rotation.
- `/oauth/login` - Memry's own sign-in and consent page, where the account name and password
  are entered.

A client pointed at `https://memory.example.com/mcp` with no key now discovers the
authorization server (via the `WWW-Authenticate` header on the 401), registers itself, sends
the user through login, and receives a token scoped to that account. No key to copy by hand.
Memry verifies the human against its own accounts, so no third-party IdP is required.

The bare origin (`https://memory.example.com`, no `/mcp`) answers the MCP handshake too, and
carries its own `/.well-known/oauth-protected-resource` document. Connector UIs ask for a
server URL and people paste the site they have open; without this the OAuth dance completed
and the handshake after it hit the dashboard, which answers POST with 405 - reported by the
client as nothing more specific than "there was a problem connecting". Browsers still get the
dashboard at the root: only MCP-shaped requests (POST, DELETE, or an event-stream GET) are
rerouted.

Connecting ChatGPT this way: [connect-chatgpt.md](connect-chatgpt.md).

## Typed decisions (optional)

Parts of the pipeline do not need a text model. Deciding whether two people called
Jonas are the same person is a choice between `same`, `different` and `unsure`, and
Memry already gates automatic merges on the confidence attached to it. Today that
confidence is a number the text model was asked to report about itself, which nothing
calibrates.

`MEMRY_DECISION_PROVIDER` selects who answers those questions. It is **off by default**
and an existing deployment behaves exactly as before:

| Value | Behaviour |
|---|---|
| `none` (default) | No decision provider. Identity judgement uses the prompt path Memry has always used. |
| `llm` | The same questions, typed, answered by the configured text model. An answer outside the declared options is rejected rather than accepted. |
| `jev` | [TypeSafe Jev](https://typesafe.ai), a System One model that answers typed questions directly and returns a probability per option. |

```bash
export MEMRY_DECISION_PROVIDER=jev
export MEMRY_DECISION_API_KEY=...        # TypeSafe API key
export MEMRY_DECISION_MODEL=jev-latest   # optional
export MEMRY_DECISION_BASE_URL=...       # optional, for a proxy
```

Jev is a hosted API, so turning it on means identity questions leave the machine, the
same trade as configuring an LLM provider. It is a second provider to weigh, not a
replacement for the first: extraction still needs a text model.

The provider can never fail a write. A transport error, a rate limit, a malformed reply
or an answer outside the declared options all read as "no answer", and the caller falls
back to the conservative path rather than treating silence as a verdict.

One caveat worth keeping in mind: "cannot hallucinate" means the reply always matches
the schema, not that it is right. A confidently wrong `same` still merges two people, so
the merge-proposal review under **Knowledge > Upkeep** matters as much as it did before.

### Measured against jev-1.13.0

Numbers from this repo, not from TypeSafe's marketing. `jev-latest` resolved to
`jev-1.13.0`; every reply names the version that answered it.

| | |
|---|---|
| One identity check | ~250 ms, ~450 input tokens |
| 128 questions in one call | 330 ms (one question alone: ~690 ms) |
| Largest state accepted | 32 KB fine, 128 KB rejected with `max_tokens_exceeded` |
| Choice options | 200 answered without complaint |

Batching is close to free, which is the whole reason this is worth doing: an episode's
identity checks are several sequential calls today and can become one.

On twelve identity cases with a known answer - two different people called Jonas, a
nickname for someone already in the store, a person and a project sharing a name, two
cities, a bare first name with no evidence - Jev agreed with the expected verdict 12 out
of 12, and never proposed an automatic merge that should not have happened.

Its confidence tracked the difficulty: 0.93-0.95 on the clear-cut cases, 0.31 on a bare
first name with nothing to go on, 0.35 on a nickname that needs a leap. Those low scores
are the useful part. They are the cases a person should look at.

### Jev against the text model, on 56 labelled cases

Same cases, same prompt shape, both providers. The labels say what the store should end
up doing: one entity, two entities, or a decision a person should make.

| | Jev 1.13.0 | gpt-5-mini (the path without Jev) |
|---|---|---|
| Verdicts that would not corrupt the store | 53/56 | 49/56 |
| Genuinely the same person, spotted | 22/22 | 22/22 |
| Genuinely different, kept apart | 22/22 | 21/22 |
| Genuinely undecidable, left for a person | 9/12 | 6/12 |
| Median latency | 211 ms | 2,535 ms |

The verdicts are close. The confidence is not, and that is what decides how much the
store can look after itself:

| | Jev | gpt-5-mini |
|---|---|---|
| Highest confidence on a merge that would have been **wrong** | 0.50 | 0.90 |
| Confidence range on merges that were **right** | 0.40-0.95, median 0.89 | 0.85-0.90, median 0.90 |
| Lowest gate that lets nothing wrong through | **0.70** | 0.95 |
| Correct merges made automatically at that gate | **20 of 22** | 4 of 22 |

The text model's wrong answers score as high as its right ones, so the two distributions
sit on top of each other and no threshold separates them. Its numbers are also suspiciously
round - 0.70, 0.80, 0.85, 0.90 - which is what self-reporting looks like.

**The current 0.9 gate is not safe on the path without Jev.** On this set it merges two
entities that should have stayed apart. One of them is the case Memry's entity handling
exists for: a partner called Jonas and a Snowflake architect called Jonas, which
gpt-5-mini calls the same person with 0.85 confidence. Jev answers `unsure` at 0.39.

So the gate belongs to the provider, and each one carries its own
(`Decider.auto_confirm_confidence`): 0.9 without Jev, 0.7 with it. 0.7 leaves 0.20 of
headroom above the worst mistake Jev made. Override with
`MEMRY_DECISION_MERGE_CONFIDENCE` once you have measured your own data; 56 cases pin a
threshold roughly, not precisely.

### The other stages

Three more places where the answers are known before the call is made. Each was probed
against the live model, and one of the three did not survive it.

| Stage | Result | Shipped |
|---|---|---|
| Entity typing | 14/16, and all sixteen in **one 608 ms call** | Yes. It was already a batch call, so this is a straight swap. Both misses were fair: a German company-register number typed as code at 0.28 confidence, which is the model flagging its own doubt, and Hetzner typed as an organization, which it is. |
| Reconcile (ADD / UPDATE / DELETE / NONE) | 9/10, ~344 ms | Yes, for the decision only. Every high-confidence answer was right, the one miss scored 0.73 and the genuinely hard retraction scored 0.48. Writing the merged sentence for an UPDATE is a writing task and stays with the text model, so this is a cheap call in front of a rarer expensive one. |
| Re-ranking search results | recall@3 0.933 -> 0.967, MRR 0.767 -> 0.928 | Yes, off by default. See below. |
| Durability, for decay | 11/12 in one 713 ms call | Not wired yet. Ready. |
| Consolidation (do two memories say the same thing) | 11/12 in one 230 ms call | Not wired yet. Ready. |
| Tag drift (are two tags the same idea) | 8/10 in one 205 ms call | Not wired yet. Weakest of the three. |

### Re-ranking search results

On by default once `MEMRY_DECISION_PROVIDER=jev` is set. `MEMRY_DECISION_RERANK=0` turns
it off, and it stays off for every other provider for the reason in the second table below.

The first version of this replaced the hybrid ranking with the relevance judgement and
measured worse than doing nothing. That result was about the implementation, not the idea:
the hybrid rank carries recency, decayed importance, entity anchors and the typed-relation
hops that make multi-hop questions work, and ordering purely by "does this text answer the
question" throws all of it away. Blending keeps it and adds what wording alone cannot see.

Over a 228-memory store with 90 questions:

| | recall@3 | MRR |
|---|---|---|
| Hybrid only | 0.933 | 0.767 |
| Replaced by relevance | 0.889 | 0.776 |
| Blended, 20% relevance | 0.933 | 0.906 |
| **Blended, 35% relevance, non-answers demoted** | **0.967** | **0.928** |
| Blended, 50% relevance | 0.933 | 0.906 |

35% beat both 20% and 50%. Two things do the work: the blend moves the right answer up the
list, and a floor at 0.15 pushes an obvious non-answer to the back however well it matched
on wording.

### Why a text model does not get to do this

The same re-ranking, same prompt, same dataset, through the shipped code path:

| Re-ranker | recall@3 | MRR | p50 | p95 |
|---|---|---|---|---|
| None (hybrid only) | 0.933 | 0.767 | **3 ms** | 4 ms |
| **Jev 1.13.0** | **0.967** | **0.917** | 190 ms | 282 ms |
| gpt-5-mini, same prompt | 0.922 | 0.839 | 10,659 ms | 13,943 ms |

The text model is the only one of the three that makes recall *worse* than not re-ranking,
and it takes ten and a half seconds a query to do it. At a p95 of fourteen seconds it is
not a feature anyone would ship, which is why `reranks_by_default` is a property of the
provider rather than a global switch.

On latency, be careful which comparison you make. Hybrid search alone is 3 ms because it is
a local index lookup with no network in it, so adding any hosted model puts a round trip in
front of that. Against *no re-ranking*, Jev costs 187 ms and buys 0.034 recall and 0.150
MRR. Against a text model doing the same job, Jev is 56 times faster and more accurate.

### Upkeep stages

All three are wired, and all three report what they did.

**How long facts stay relevant** (11/12 on a labelled set, one 713 ms call). Forgetting
decays importance on a half-life per memory type, which treats "the train was delayed this
morning" and "allergic to penicillin" identically because both are semantic. A pass now
records a per-fact estimate in the memory's metadata and decay prefers it, falling back to
the type rate for anything unscored. "Allergic to penicillin" scored top of the scale at
0.98 confidence and the train delay scored bottom at 0.81. Run it from **Knowledge >
Upkeep** or `POST /api/v1/maintenance/durability`.

**Consolidation** (11/12, one 230 ms call). The typed question now runs first and the text
model is only asked when the answer is yes, because writing the merged sentence is the only
part that needs prose. Most candidate groups are not the same fact, so most of those calls
stop happening. Genuine restatements scored 0.62-0.94 and everything else 0.02-0.08,
including pairs built to look alike: "Ada lives in Amsterdam" against "Ada works in
Amsterdam" came back 0.06.

**Tag drift** (8/10, one 205 ms call). Added as a fourth suggestion pass behind the three
that already exist, and deliberately never as automation. It missed "food" beside "diet"
and "project-phoenix" beside "phoenix", both of which a person would merge, but it never
proposed an unrelated pair. Missing a suggestion costs nothing; inventing one costs trust.

Every pass writes a line to the server log saying what it changed, and the Upkeep panel
shows the same thing next to the button that ran it.

### A trap worth remembering

An early entity-typing probe scored 2 out of 16 because the names only ever appeared in
the shared state, never in the questions, and every answer came back "person" at around
0.75. Confidence describes the answer to the question that was asked. A question carrying
no information still gets a confident-looking reply.

## Scaling up

| Situation | Setting |
|---|---|
| Faster vector search past ~5k memories | `pip install "memry[ann]"` - a usearch HNSW sidecar supplies candidates above the configured threshold; `memry reindex` rebuilds it |
| Many agents or devices sharing memory | Point every client at the same `memry serve` URL. They share one server process and one SQLite store. |
| Several server replicas or machines writing one store | Unsupported. Do not point multiple Memry processes at the same database file. This would require a separately reviewed storage architecture. |

## Connecting agents to a shared server

Multiple agents on multiple machines can share one memory server over MCP HTTP:

```jsonc
{
  "mcpServers": {
    "memry": {
      "type": "http",
      "url": "https://memory.example.com/mcp",
      "headers": { "Authorization": "Bearer <MEMRY_API_KEY>" }
    }
  }
}
```

Clients that cannot send headers (claude.ai custom connectors) may embed the
admin key in the URL instead: `https://memory.example.com/mcp/<MEMRY_API_KEY>`
(or `/mcp?key=...`). See [connect-claude-ai.md](connect-claude-ai.md).

For local single-machine use, prefer stdio (`memry mcp`) - no port, no auth surface.

## Provider configuration

| Goal | Setting |
|---|---|
| Default Anthropic extraction | `ANTHROPIC_API_KEY` + `pip install "memry[anthropic]"` (defaults to the fast, lower-cost `claude-haiku-4-5`) |
| Larger Anthropic model | `MEMRY_LLM_MODEL=claude-opus-4-8` (explicitly trades more latency and cost for extraction quality) |
| OpenAI end-to-end | `OPENAI_API_KEY` (LLM `gpt-5-mini`, embeddings `text-embedding-3-small`) |
| Fully offline | `MEMRY_LLM_PROVIDER=ollama` + `MEMRY_EMBEDDING_PROVIDER=ollama` (e.g. `llama3.1`, `nomic-embed-text`) |
| Zero keys, zero model downloads | nothing - verbatim writes + BM25/hash retrieval |

After switching embedding providers, run `memry reindex` once to re-embed the store.

MCP saves with the default `infer=true` commit the exact text before replying. The
server then enriches pending memories in its managed worker. If the provider is down,
the raw memory stays searchable and is retried; restarting the server resumes pending rows.
No external queue service is required.

## Maintenance

```bash
memry sweep --threshold 0.1   # soft-forget stale, low-importance memories
memry stats                   # counts, providers, db path
memry export > backup.json    # knowledge only: IDs, provenance, entities, relations, history
memry abstract-tags           # LLM clusters tags into higher-level ones now
```

When accounts or OAuth are enabled, also back up `auth.db` with `memry.db`. The JSON export
does not contain login data.

A weekly `sweep` in cron/Task Scheduler keeps long-running stores lean; forgotten memories
are invalidated (auditable, recoverable), never destroyed.

## Managing topics and entities

The dashboard's **Knowledge** area contains Tags, People and things, Forgotten, and Upkeep.
Tags show memory counts, can be filtered by name, and can be renamed, combined, or deleted
under the current user filter. The same topic operations remain available at
`POST /api/v1/tags/edit` for API compatibility.

An optional, off-by-default LLM pass proposes higher-level parents for browsing, such as
`health` over `liver health` and `weekly gym`. Leave it off unless you want that navigation
view: retrieval measures best when a filter names the specific level, and a broad parent
adds candidates without adding coverage (`MEMRY_TAG_ABSTRACTION=on`,
`MEMRY_TAG_ABSTRACTION_INTERVAL_DAYS=7`, or `memry abstract-tags`). Memry stores hierarchy
edges and expands a parent filter at query time; it does not copy the parent label onto each
memory. Synthetic parents remain visible through `/api/v1/categories` and
`GET /api/v1/tags/synthetic`.

People and things open as entity hubs with aliases, a bounded description, and active
supporting memories. Relations are listed under the entity they describe and can open their
members.

## Searching by tag and date

Beyond relevance search, both `search_memories`/`POST /api/v1/search` and
`list_memories`/`GET /api/v1/memories` accept a `categories` (tag) filter and a `since`/
`until` date window (`YYYY-MM-DD`, the `until` day inclusive). Pass an empty query with just
a tag or date to browse rather than rank, e.g. "everything tagged `travel` since 2026-01-01".
