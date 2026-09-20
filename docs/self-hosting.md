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

## Typed decisions (experimental, off by default)

**This is experimental and off unless you turn it on.** It sends identity and
housekeeping questions to a third-party API, it changes how much of the upkeep happens
without you, and the thresholds behind it were chosen from a small sample. Leave it off
unless you want to try it.

Parts of the pipeline do not need a text model. Deciding whether two people called Jonas
are the same person is a choice between `same`, `different` and `unsure`, and Memry
already gates automatic merges on the confidence attached to it. Without a decision
provider that confidence is a number the text model was asked to report about itself,
which nothing calibrates.

`MEMRY_DECISION_PROVIDER` selects who answers those questions:

| Value | Behaviour |
|---|---|
| `none` (default) | No decision provider. Everything works exactly as it did before. |
| `llm` | The same questions, typed, answered by the configured text model. An answer outside the declared options is rejected rather than accepted. |
| `jev` | [TypeSafe Jev](https://typesafe.ai), a System One model that answers typed questions directly and returns a probability per option. |

```bash
export MEMRY_DECISION_PROVIDER=jev
export MEMRY_DECISION_API_KEY=...        # TypeSafe API key
export MEMRY_DECISION_MODEL=jev-latest   # optional
export MEMRY_DECISION_BASE_URL=...       # optional, for a proxy
```

Jev is a hosted API, so turning it on means these questions leave the machine, the same
trade as configuring an LLM provider. It does not replace one: extraction still needs a
text model.

The provider can never fail a write. A transport error, a rate limit, a malformed reply
or an answer outside the declared options all read as "no answer", and the caller falls
back to the path it would have taken anyway.

One caveat worth keeping in mind: "cannot hallucinate" means the reply always matches the
schema, not that it is right. A confidently wrong `same` still merges two people, so the
merge-proposal review under **Upkeep** matters as much as it did before.

### What it is wired to

| Stage | What changes |
|---|---|
| Entity identity | The verdict and the confidence the automatic-merge gate reads. |
| Entity typing | One question per name in a single call, instead of one call per batch through the text model. |
| Reconcile | The action and its target. Writing the merged sentence for an UPDATE still needs the text model. |
| How long facts stay relevant | A per-fact estimate, which forgetting prefers over one decay rate per memory type. |
| Consolidation | A cheap check first, so the text model is only asked to write a merge when there is one. Word-for-word duplicates merge on their own; a merge the model proposed waits under Upkeep, because that judgement has not been measured. |
| Tag drift | Suggestions only, for review under Upkeep. Never applied automatically. |
| Search re-ranking | On with Jev, off otherwise. `MEMRY_DECISION_RERANK=0` turns it off; `=1` turns it on for a text model measured to help (gpt-5.6-luna), and is refused for one that was not. |

### The settings, and where they came from

Two numbers are not obvious, so both were measured rather than guessed. The datasets and
harnesses are in `evals/` if you want to re-run them against your own data, which is the
only way to know whether these hold for your store.

**The automatic-merge gate** (`Decider.auto_confirm_confidence`) is 0.70 with Jev and
0.95 with gpt-5-mini as the text model. It is a property of the model because the number
only means something relative to how that model's confidence is spread: a model
reporting a number about itself scores its wrong answers about as high as its right ones,
so the gate has to sit high and little gets automated. Override with
`MEMRY_DECISION_MERGE_CONFIDENCE`.

**A text model nobody has measured never merges on its own.** On the same 56 cases,
gpt-5.6-luna got 52 verdicts safe, better than gpt-5-mini's 49, and put its worst wrong
"same" at 0.98, above any threshold. There is no number that is safe for a model that
has not been run against the labelled set, so for any text model other than gpt-5-mini
every proposed merge waits for you under **Upkeep**. To measure your own
model, run `evals/identity_benchmark.py llm --model <name>` and set the gate it reports
with `MEMRY_DECISION_MERGE_CONFIDENCE`. A confident "different" still blocks an
obvious-looking merge at 0.95 whatever the gate, so raising the gate never makes merging
easier.

Raising the gpt-5-mini gate from 0.9 to 0.95 was a change to existing behaviour, and a
fix: on the labelled set, 0.9 merged two entities that should have stayed apart.

**Re-ranking** blends the relevance judgement with the hybrid rank at 0.35 rather than
replacing it, and pushes anything under 0.15 to the back. Replacing the hybrid rank
outright measured worse than not re-ranking at all, because that rank already carries
recency, decayed importance, entity anchors and the typed-relation hops multi-hop
questions depend on.

It is on by default with Jev. With a text model it depends on which one, measured over
the same 228 memories and 90 questions: gpt-5.6-luna lifted recall@3 from 0.933 to 0.956
and MRR from 0.828 to 0.933 at 1.7 seconds a search, so `MEMRY_DECISION_RERANK=1` turns
it on; gpt-5-mini scored below not re-ranking at all at nearly ten seconds a search, so
for it, and for any model not measured, the setting is refused.

### A trap worth remembering

An early probe scored 2 out of 16 because the names only ever appeared in the shared
state, never in the questions, so sixteen identical questions got sixteen identical
answers at around 0.75 confidence. Confidence describes the answer to the question that
was asked. A question carrying no information still gets a confident-looking reply.

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

The dashboard's **Upkeep** button opens four tabs: Upkeep (what needs you), Entities, Tags, and
Archive (what was removed). A badge on the button counts what is waiting.
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

Entities open as hubs with aliases, a bounded description, and active
supporting memories. Relations are listed under the entity they describe and can open their
members.

Upkeep runs on its own and asks only for what it will not decide: it lists the entity
merges below the gate, the memory merges a model proposed, the tag pairs that look like one
subject split in two, and the names the model judged not to be entities, each with a yes
and a no. Everything else (entity self-healing, word-for-word duplicate consolidation, and
durability scoring when a decision provider is configured) runs on its interval, records what it changed, and can be paused with one switch.
`POST /api/v1/maintenance/run/<pass>` runs any pass now.

### Hubs, homes and shared names

An extractor turns far more phrases into entities than a store has things. On the store
these rules were built on, 3,314 entities came out of 985 memories, and 2,073 of them
appeared in exactly one memory. Memry keeps every one of them and shows you the ones that
earned it.

- **A hub** is a name the map and the Entities list show. With a decision
  provider, a name is a hub when the provider called it a named thing, or when it is a
  person, organization, project, product or place that the provider did not call a value or
  a role. Without a provider, it is one of those five types, or a name two memories mention.
  Hub status is computed each time, so a phrase you mention again next month is a hub then.
  The map asks for a little more: a planet is a hub that came up in at least two memories,
  and a person is one from the first mention. On the store above the hubs alone were 1,424
  planets, 610 of them things seen exactly once.
- **A home** is the project or product a part belongs to, shown as
  `AI-Flow / privacy policy`. A stated `part_of` relation sets it. Otherwise one project or
  product has to appear in at least 70% of the part's memories, and a part seen once needs
  that project to be the only project, product or organization in its memory. An
  organization becomes a home only through a stated relation.
- **A shared name** is read through home. Two entities with the same name under different
  homes are never proposed for merging. Two with the same name and nothing setting them
  apart are merged. Two people are never merged on a name alone.

New names are screened before they become entities. Measurements and counts ("250 ms",
"22 tests") are dropped by rule. With a decision provider, each new name gets one typed
question in the memory it came from, and a name judged a value or a role with at least 0.80
probability is not made an entity. The phrase stays on the memory. Names already in the
store get the same question during upkeep, and the ones judged a value or a role wait under
**Upkeep** for a yes or a no.

An entity is never deleted. When you or a rule removes a name, Memry retires it, and
**Upkeep > Archive > Removed names** lists it with the reason and a restore button.
`POST /api/v1/maintenance/run/structure` with `{"dry_run": true}` returns every home and
every merge the pass would make, and changes nothing.

| Rule | Measured on | Result |
|---|---|---|
| Measurement and count rules | 360 labelled names | matched 12, none of them a real thing; 96 of 3,314 names store-wide |
| Name screen at the 0.80 gate | 360 labelled names | screened out 31, none of them a real thing; clean from 0.70 up |
| Hub rule using the provider's verdict | 360 labelled names | 72% of hubs are real things, and 98% of real things are hubs |
| Hub rule, first draft: type, or two memories, or a relation | the same names | 52% and 88% |
| Home, as shipped | 141 labelled homes | 86% correct |
| Home from co-mention alone | the same homes | 68% correct, and 47% when the home is an organization |
| Same name, not a person, nothing setting them apart | 78 past merge decisions | all 78 had been confirmed |

An independent reader labelled the names and homes, all from one real store. Two first
drafts failed the labels and were changed: recurrence turned out to find topics like
"billing", and the first screening question listed "path" among the values, so the provider
screened out source files and street addresses. `evals/entity_structure_benchmark.py` runs
the same scoring on your own store.

## Searching by tag and date

Beyond relevance search, both `search_memories`/`POST /api/v1/search` and
`list_memories`/`GET /api/v1/memories` accept a `categories` (tag) filter and a `since`/
`until` date window (`YYYY-MM-DD`, the `until` day inclusive). Pass an empty query with just
a tag or date to browse rather than rank, e.g. "everything tagged `travel` since 2026-01-01".

## When a memory happens

`since`/`until` filter on the day a memory was recorded. A memory can also carry the time
the fact itself happens, in `metadata["when"]`:

```json
{"start": "2026-10-03", "end": "2026-10-07", "recurrence": "yearly"}
```

`start` is `YYYY-MM-DD`, `YYYY-MM-DDTHH:MM`, or `--MM-DD` for a yearly date whose year is
unknown, which is how a birthday is stored. `end` and `recurrence` (`yearly`, `monthly`,
`weekly`, `daily`) are optional.

Extraction sets a `when` only for something that happened or will happen at a particular
time: a meeting, a launch, a release, a purchase, a trip, an appointment, a deadline, a
move, or a decision made on a date. A price observed on a day, a test log and a
specification all carry dates without occurring, so a date in the text on its own does not
produce a `when`.

`search_memories`/`POST /api/v1/search` and `list_memories`/`GET /api/v1/memories` take
`when_since`/`when_until` (`YYYY-MM-DD`, both days inclusive) beside `since`/`until`. They
match on the occurrence time, and a memory without one never matches, which is what makes
"what is on this weekend" answerable. A recurring `when` matches when any of its
occurrences falls in the window. The REST memory payload carries `when` and the computed
`next_occurrence`, and the dashboard memory card shows the same in a small chip.

To read occurrence times out of memories saved before this existed:

```bash
curl -X POST http://localhost:8080/api/v1/maintenance/run/when \
  -H 'content-type: application/json' -d '{"dry_run": true, "limit": 20}'
```

A dry run writes nothing and returns what it would set. Without `dry_run` it stores each
`when` it finds and marks the rest as checked, so a second run over the same memories
spends nothing. The pass needs an LLM; without one it reports that and changes nothing.
