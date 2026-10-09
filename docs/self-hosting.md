# Self-hosting Memry

Memry is one Python process with SQLite and no external database, queue, or vector service.
Knowledge lives in `memry.db`; runtime accounts and OAuth live in the adjacent `auth.db`.
Keeping them separate prevents knowledge restore/reset operations from changing login data.
A complete server backup must include both files.

## The two models needed when setting up the server

Before you start a Memry server, set a text model and a decision model. Without either
one, `memry serve` and `memry mcp` don't start and print the setting that is still empty.

| | What Memry uses it for | Setting |
|---|---|---|
| Text model | Extracting facts and names from what you save, and writing entity descriptions | `OPENAI_API_KEY` (default model `gpt-6-luna`), or `ANTHROPIC_API_KEY`, or `MEMRY_LLM_PROVIDER=ollama` |
| Decision model | Decision questions, such as whether two names belong to one thing, what kind of thing a name is, whether an old memory is out of date once you save a new one and how relevant a search result is | `MEMRY_DECISION_PROVIDER=jev` and `MEMRY_DECISION_API_KEY` (a [TypeSafe](https://typesafe.ai) key) |

```bash
export OPENAI_API_KEY=sk-...
export MEMRY_DECISION_PROVIDER=jev
export MEMRY_DECISION_API_KEY=...
```

Use a System One model like Jev as the decision model. For every decision question Jev
returns a calibrated probability per answer, and Memry merges duplicates on its own at the
thresholds measured for Jev.

If you set `MEMRY_DECISION_PROVIDER=llm`, Memry sends these questions to the text model.
A text model's confidence scores have not been calibrated, so Memry then merges entities
only by fixed rules, for example two entities with the same name where one of them has no
memories yet. You confirm the other merges under Upkeep in the dashboard. The other upkeep
passes run as usual. The server logs a warning at start in that mode.

`memry config` prints the resolved configuration and lists any model setting that is
still empty.

## Option 1 - bare (recommended for personal use)

```bash
pip install memry
export OPENAI_API_KEY=sk-... MEMRY_DECISION_PROVIDER=jev MEMRY_DECISION_API_KEY=...
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
environment variables from `.env` (copy [`.env.example`](../.env.example) and set the API
key and both models). See [`docker-compose.yml`](../docker-compose.yml).

Docker automatically reuses the package layer for source-only updates. A first build or
a change to `requirements-docker.txt` installs all dependencies; normal code and dashboard
updates install only Memry itself. The image includes usearch (the `ann` extra), so it
builds the ANN index once a store passes the threshold. To deliberately refresh every package:

```bash
docker compose build --no-cache memry
docker compose up -d memry
```

## Option 3 - VPS, one command (Docker + automatic HTTPS)

On a fresh Ubuntu/Debian server at any provider (Contabo, Hetzner,
DigitalOcean, ...):

```bash
curl -fsSL https://raw.githubusercontent.com/cosmin-novac/memry/main/deploy/install.sh \
  | MEMRY_DOMAIN=memory.example.com OPENAI_API_KEY=sk-... \
    MEMRY_DECISION_PROVIDER=jev MEMRY_DECISION_API_KEY=... bash
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
   together, including any live SQLite `-wal`/`-shm` files. The
   [nightly snapshot](#nightly-snapshot) does that, and so does a directory or volume
   snapshot. `memry export` is a lossless knowledge backup, but it does not include
   accounts, sessions, OAuth clients, or tokens from `auth.db`.

## Nightly snapshot

Set `MEMRY_SNAPSHOT_DIR` and the server keeps one copy of `memry.db` (and `auth.db` when it
exists) in that directory, made once a day. The VPS compose file sets it for you, to
`/var/backups/memry` on the host.

How a run goes:

1. Memry copies each file with SQLite's online backup API, from a connection of its own,
   1,024 pages at a time with a 5 ms pause between steps. The server keeps reading and
   writing during the copy, and the copy never takes Memry's own backend lock.
2. The copy goes to a temporary file in the snapshot directory. Memry flushes it to disk
   and checks it: `PRAGMA integrity_check` has to answer `ok`, and the counts of memories,
   entities and episodes have to read from the copy.
3. Only then does Memry replace the previous copy (`os.replace`, atomic on one filesystem)
   and write `snapshot.json`: when, which Memry version, each file's size and sha256, the
   counts, and how long the run took.

When a run fails, Memry keeps the previous copy and `snapshot.json` exactly as they were,
removes only its temporary file, logs the error and writes it to `snapshot-failure.json`.
The dashboard shows the failure under About > This server, and the server tries again an
hour later.

| Setting | Default | What it does |
|---|---|---|
| `MEMRY_SNAPSHOT_DIR` | unset (off) | Directory for the copy. Pick one Memry never writes to otherwise. Memry refuses the data directory itself. |
| `MEMRY_SNAPSHOT_AT` | `03:30` | Time of day, `HH:MM`, on the server's clock. |
| `MEMRY_SNAPSHOT_HOST_DIR` | unset | Shown in the dashboard only: where `MEMRY_SNAPSHOT_DIR` is on the host when it is a bind mount. |

The server runs the snapshot once a day from `MEMRY_SNAPSHOT_AT` on, and skips the day when
a copy already succeeded that day. When the last good copy is more than 26 hours old, or
there is none, the server makes one about 90 seconds after it starts.

`MEMRY_SNAPSHOT_AT` is read on the server's local clock. In Docker that is the container's
`TZ`, and a container without `TZ` runs on UTC, so `03:30` means 03:30 UTC (05:30 in Berlin
in summer). The VPS compose file passes `MEMRY_TZ` from `.env` on as `TZ`. A POSIX rule such
as `MEMRY_TZ=CET-1CEST,M3.5.0,M10.5.0/3` always works; a zone name such as `Europe/Berlin`
works only if the image has zone data. The About panel shows the zone in force next to the
time.

Memry copies the two files one after the other, a moment apart. `auth.db` holds accounts,
keys and sessions, which change rarely, so the pair is a usable restore point.

### Checking a snapshot, and making one by hand

```bash
memry snapshot                 # make one now, into MEMRY_SNAPSHOT_DIR
memry snapshot --to /some/dir  # make one into another directory
memry snapshot --check         # verify the copy against snapshot.json; writes nothing
```

`--check` reads each file, compares its size and sha256 with `snapshot.json` and runs
`PRAGMA integrity_check`. It exits 0 when the copy is sound, and 1 with the problems listed
when it is not. In the VPS deployment, run it inside the container:

```bash
docker compose --env-file /opt/memry/.env -f /opt/memry/app/deploy/vps/docker-compose.yml \
  exec memry memry snapshot --check
```

### Restoring the database from the snapshot

1. Check the copy first: `memry snapshot --check` (inside the container, as above).
2. Stop the server. On the VPS:
   `docker compose --env-file /opt/memry/.env -f /opt/memry/app/deploy/vps/docker-compose.yml stop memry`
3. Copy the files back into the data directory and move the old WAL files beside them out
   of the way. On the VPS the data directory is the `memry_memry-data` volume:

   ```bash
   DATA="$(docker volume inspect -f '{{ .Mountpoint }}' memry_memry-data)"
   mkdir -p /root/memry-replaced && mv "$DATA"/*.db* /root/memry-replaced/
   cp /var/backups/memry/memry.db "$DATA/memry.db"
   cp /var/backups/memry/auth.db "$DATA/auth.db"     # when the snapshot has one
   ```

   The `-wal` and `-shm` files belong to the database you are replacing. Left in place,
   SQLite would try to apply them to the restored file. Moving them aside keeps them until
   you are sure the restore worked.
4. Start the server with the same command and `up -d memry`, then open the dashboard and
   look at your memories.

`tests/test_snapshot.py` restores a snapshot into a new directory this way, opens it and
compares the counts.

### Offsite copy in an S3-compatible bucket (optional, off by default)

A copy on the same disk as the database survives a damaged file, a bad migration or a
mistaken delete. It does not survive losing the disk. The memry.tech server has one virtual
disk and runs local-only for now; the offsite copy is there for when that changes.

When `MEMRY_SNAPSHOT_OFFSITE_URL` and `MEMRY_SNAPSHOT_OFFSITE_BUCKET` are set, Memry also
gzips each good local snapshot and uploads it to an S3-compatible bucket. Memry uploads each
file to a temporary key, reads its size and sha256 back, copies it to the final key on the
storage side and deletes the temporary key. If any step fails, the object already in the
bucket stays as it was, and the local snapshot stands either way. `snapshot.json` goes up
last. Memry records the result (ok, or the error) in the local `snapshot.json` and shows it
in About. Memry signs the requests itself (AWS Signature Version 4 over httpx), so you need
no extra package.

| Setting | Example | Notes |
|---|---|---|
| `MEMRY_SNAPSHOT_OFFSITE_URL` | `https://<account-id>.r2.cloudflarestorage.com` | The S3 endpoint, without the bucket. |
| `MEMRY_SNAPSHOT_OFFSITE_BUCKET` | `memry-backups` | The bucket name. |
| `MEMRY_SNAPSHOT_OFFSITE_KEY_ID` | | Access key ID. |
| `MEMRY_SNAPSHOT_OFFSITE_SECRET` | | Secret access key. `memry config` shows it as `***`. |
| `MEMRY_SNAPSHOT_OFFSITE_REGION` | `auto` | R2 takes `auto`; B2 and Contabo take their region, such as `eu-central-003` or `eu2`. |
| `MEMRY_SNAPSHOT_OFFSITE_PREFIX` | `memry/` | Key prefix. The objects are `memry/memry.db.gz`, `memry/auth.db.gz` and `memry/snapshot.json`. |

Cloudflare R2 is the suggested store: 10 GB of storage is free and downloads cost nothing.
To set it up:

1. In the Cloudflare dashboard, open R2 Object Storage and create a bucket, for example
   `memry-backups`. Note your account ID from the R2 overview page.
2. Under R2 > Manage API tokens, create an API token with "Object Read & Write"
   permission, limited to that one bucket. Copy the Access Key ID and the Secret Access
   Key; Cloudflare shows the secret once.
3. Add to `/opt/memry/.env`:

   ```bash
   MEMRY_SNAPSHOT_OFFSITE_URL=https://<account-id>.r2.cloudflarestorage.com
   MEMRY_SNAPSHOT_OFFSITE_BUCKET=memry-backups
   MEMRY_SNAPSHOT_OFFSITE_KEY_ID=<access key id>
   MEMRY_SNAPSHOT_OFFSITE_SECRET=<secret access key>
   ```

4. Recreate the container (`up -d memry`) and run `memry snapshot` inside it once. Its
   output ends with an `offsite` entry that says `"ok": true`, and the bucket then holds
   the objects.

Backblaze B2 (`https://s3.<region>.backblazeb2.com`) and Contabo Object Storage
(`https://<region>.contabostorage.com`) work the same way with their own keys. To restore
from the bucket, download `memry.db.gz` (and `auth.db.gz`), unzip them with `gunzip`, and
follow the restore steps above.

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

## Where Memry sends its decision questions

For some steps of the pipeline you don't need a text model. Deciding whether two people
called Jonas are the same person means picking `same`, `different` or `unsure`, and Memry
merges automatically only above a confidence threshold. When Memry sends these questions to
a text model, that confidence is a number the model writes about its own answer, and nobody
calibrates it.

You set where Memry sends those questions with `MEMRY_DECISION_PROVIDER`:

| Value | Behaviour |
|---|---|
| `jev` | [TypeSafe Jev](https://typesafe.ai), a System One model. For each decision question it returns a calibrated probability per answer, and Memry merges duplicates on its own at the measured thresholds. |
| `llm` | Memry sends the same decision questions to the configured text model and rejects any answer outside the declared options. Set it only on purpose: Memry then merges entities only by fixed rules, and you confirm the other merges yourself. |
| `none` | Memry sends no decision questions, uses the text model's older prompts and skips the upkeep passes built on a decision provider. Set it only on purpose: Memry then merges entities only by fixed rules, and you confirm the other merges yourself. |
| unset | The server doesn't start (see [the two models needed when setting up the server](#the-two-models-needed-when-setting-up-the-server)). |

```bash
export MEMRY_DECISION_PROVIDER=jev
export MEMRY_DECISION_API_KEY=...        # TypeSafe API key
export MEMRY_DECISION_MODEL=jev-latest   # optional
export MEMRY_DECISION_BASE_URL=...       # optional, for a proxy
```

Jev is a hosted API, so Memry sends these questions to TypeSafe's servers, the same way it
sends what you save to a hosted text model. You still need a text model for extraction. On
a fully offline server with Ollama, set `MEMRY_DECISION_PROVIDER=llm` and confirm merges
yourself.

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
| Reconcile | The action and its target. The text model writes the merged sentence for an UPDATE; without one, the old memory is kept and superseded by the new one, so no text is lost. A contradiction only replaces a memory on its own where little is at stake; see below. |
| How long facts stay relevant | A per-fact estimate (days, months or years), recorded on each memory and acted on by nothing yet: no memory is forgotten for its age, and search does not read it. It is kept for a planned experiment on relevance per entity. Off unless `MEMRY_DURABILITY=1` (`decay.durability`), for the scheduler and "run now" alike; the score does not move a memory's `updated_at`. |
| Consolidation | A cheap check first, so the text model is only asked to write a merge when there is one. Word-for-word duplicates merge on their own; a merge the model proposed waits under Upkeep, because that judgement has not been measured. |
| Tag drift | Two tags that may be one are a merge proposal like two names. They merge on their own when the provider, shown each with its 10 most recent memories and asked in both orders, puts P(same subject) at its tag bar or more (0.55 for Jev; a text model is not asked). Without a calibrated judge the pair waits under Upkeep with the other merge proposals. |
| Search re-ranking | On with Jev, off otherwise. `MEMRY_DECISION_RERANK=0` turns it off; `=1` turns it on for a text model measured to help (gpt-5.6-luna, gpt-5-mini), and is refused for one that was not. Where it is on, `retrieval.relational_relevance` "auto" (the default) has the provider judge the first 20 of every search, filtered or not, in the linked search's order when the question names a hub and in the text ranking's otherwise, and the results are ordered by that judgement. `"vector"` judges no search: a question naming a hub is ordered by the property vectors, and one naming none by the text ranking. |

### The settings, and where they came from

Two numbers are not obvious, so both were measured rather than guessed. The datasets and
harnesses are in `evals/` if you want to re-run them against your own data, which is the
only way to know whether these hold for your store.

**The automatic-merge gate** (`Decider.auto_confirm_confidence`) is 0.70 with Jev, and
would be 0.95 for gpt-5-mini as the text model. It is a property of the model because
the number only means something relative to how that model's confidence is spread: a model
reporting a number about itself scores its wrong answers about as high as its right ones,
so the gate has to sit high and little gets automated. Override with
`MEMRY_DECISION_MERGE_CONFIDENCE`.

**A text model does not decide identity.** On the same 56 cases, gpt-5.6-luna got 52
verdicts safe, better than gpt-5-mini's 49, and put its worst wrong "same" at 0.98, above
any threshold: a number a model reports about itself says too little to merge on, and a
confident "different" from it would keep two records of one person apart for good. So
without a calibrated decision provider (a text model only, or `MEMRY_DECISION_PROVIDER=llm`)
Memry asks no model whether two entities are one, at save or in upkeep, and writes no
model's confidence on a pair. A name the store has joins its entity by rule, two entities
of one name are joined by the same rule in the weekly pass, a tag folds into the thing of
its very name, and every other pair waits for you under **Upkeep**.
`evals/identity_benchmark.py llm --model <name>` measures a text model's gate all the same,
for comparison.

**When a contradiction may replace a memory.** Replacing is the one reconcile action that
takes a fact out of use, and it rests on one model reading one text. So it only happens
on its own when the memory it would replace is rated below 0.8 in importance, was stated
in a single save, and, with a typed decision provider, the judgement reaches its bar.
A memory rated important or stated in several saves is still replaced without asking when
the decision provider reads it, in the same call, as a changeable state that has since
moved on (a listing deleted, a task stopped, a document uploaded), and puts at least 0.8
on its no longer holding (changed and wrong together): importance says how much a fact matters, not how risky replacing it is, and the old
memory stays as history. Read as still true beside the new fact, both are kept and nobody
is asked. A lasting fact or a standing rule (health, identity, a relationship, "never do
X"), an unsure reading, and any answer of the text model alone still wait under **Upkeep
> Contradictions**, where you say which is right or that both are. A replacement that did
go ahead is listed under **Upkeep > Archive > Replaced by a newer memory** and can be
undone there. The thresholds are `MEMRY_SUPERSEDE_PROTECT_IMPORTANCE`,
`MEMRY_SUPERSEDE_PROTECT_SOURCES`, `MEMRY_SUPERSEDE_CONFIDENCE` and
`MEMRY_SUPERSEDE_STATE_CONFIDENCE` (0.8).

Questions already waiting can be asked again under the new rule:

```bash
memry reconcile-queue           # per question: the judge's answer, the old and the new decision
memry reconcile-queue --apply   # act on the new decisions; undo a replacement under Archive
```

It costs one decision-provider call per question and writes nothing without `--apply`.

**Re-ranking** has the decision provider judge the first 20 of a search in one call and
orders the results by that judgement. When it first asked whether a memory "helps answer
the question", the judgement alone measured worse than not re-ranking at all, and it was
blended with the hybrid rank instead. Every search now asks one question, whether someone
who reads only the memory can answer it, and measured again on the same 228 memories and
90 questions (`evals/datasets/distractors_v1.jsonl`, the `memry eval` protocol) the
judgement alone did as well as any blend, so the blend is gone. A question naming
something Memry knows is ordered by the linked search first (see `docs/architecture.md`,
read path), which follows the links from it directed and weighted, one link deep; that is
the only link mode (`retrieval.relational_mode` "directed", `relational_fusion` "linked"),
and a config naming a removed one ("typed", "undirected", "rescue", "weighted", "inherit",
"gated") is refused at startup. Where re-ranking is on, the provider judges its first 20
(`retrieval.relational_relevance` "auto", see the table).

It is on by default with Jev. With a text model it depends on which one, measured over
the same 228 memories and 90 questions in the wording every search now asks in: judging
every search, gpt-5.6-luna and gpt-5-mini both put the answer higher than no judging at
all, and both below Jev, gpt-5.6-luna at about 2.6 seconds a call and gpt-5-mini at about
9. So `MEMRY_DECISION_RERANK=1` turns either on, neither is on by default, and for any
model not measured the setting is refused. (In the wording re-ranking first asked in,
gpt-5-mini had scored below no re-ranking and was refused.)

### A trap worth remembering

An early probe scored 2 out of 16 because the names only ever appeared in the shared
state, never in the questions, so sixteen identical questions got sixteen identical
answers at around 0.75 confidence. Confidence describes the answer to the question that
was asked. A question carrying no information still gets a confident-looking reply.

## Scaling up

| Situation | Setting |
|---|---|
| Faster vector search past ~5k memories | `pip install "memry[ann]"` (the Docker image has it) - a usearch HNSW sidecar supplies candidates above the configured threshold; `memry reindex` rebuilds it |
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
| OpenAI end-to-end | `OPENAI_API_KEY` (LLM `gpt-6-luna`, embeddings `text-embedding-3-small`) |
| Previous OpenAI default | `MEMRY_LLM_MODEL=gpt-5.6-luna` (measured for re-ranking, with gpt-5-mini) |
| Fully offline | `MEMRY_LLM_PROVIDER=ollama` + `MEMRY_EMBEDDING_PROVIDER=ollama` (e.g. `llama3.1`, `nomic-embed-text`) + `MEMRY_DECISION_PROVIDER=llm` |

You also need a decision model for every server (see
[the two models needed when setting up the server](#the-two-models-needed-when-setting-up-the-server)).

After switching embedding providers, run `memry reindex` once to re-embed the store.

MCP saves with the default `infer=true` commit the exact text before replying. The
server then enriches pending memories in its managed worker. If the provider is down,
the raw memory stays searchable and is retried; restarting the server resumes pending rows.
No external queue service is required.

## Maintenance

```bash
memry stats                   # counts, providers, db path
memry export > backup.json    # knowledge only: IDs, provenance, entities, relations, history
memry snapshot --check        # verify the nightly snapshot against its manifest
memry tags-to-things --dry-run   # tags to topic entities (done at first open): count only
memry split-memories --dry-run   # memories that hold several facts: print each split, write nothing
memry learn-owner --dry-run      # who "the user" is, from what was said: print it, write nothing
```

A memory should hold one fact. Before merges were kept to one fact, a store could grow
memories one claim at a time: each save that said something new about one subject was
merged into the memory before it, until one memory read "The central claim of Ana's
thesis is ... The thesis further argues ... The thesis explicitly rejects ...". One
memory like that is found less well by a search for any one of its claims.
`memry split-memories [--user USER] [--min-words N] [--dry-run [--plan-out PATH]] [--plan-in PATH]` repairs them:

- It asks the text model about each memory in use whose text has more than one sentence.
  A memory of one sentence is left alone without asking. `--min-words N` asks only about
  memories of at least N words.
- The model splits the memory into single facts. A list stays one fact ("Ada's skills
  include Python, SQL and Go") unless its items carry details of their own. When it finds
  one fact (a fact and its details can take two sentences), the memory stays as it is.
- Each fact must say what it is about, by a name the memory uses: a fact such as "Merge
  the first change first", with no project named, would be worse than the memory. If any
  fact names none of the people or things the memory is linked to, nor a name the memory
  states, nor you, the memory stays whole, and the report names that fact.
- A split is made only when the same check a save gets finds that the facts keep every
  detail of the memory. Otherwise the memory stays, and the report says what would have
  been lost.
- The model also says what each fact is about among the people, things and tags the
  memory is linked to, including ones the fact does not spell out. Each fact keeps those
  and any its text names, and its tags are the tags among them. If a fact would keep none
  of them, or one of them would be on no fact, the memory stays whole.
- Each fact becomes a memory with the old one's dates, the turns it rests on, importance,
  run and agent, and the relations go to the fact that keeps both ends. The old memory
  leaves search and is listed under Archive in the dashboard.
- `--dry-run` asks the model and prints each memory with the facts it would become and
  what each is about, writing nothing. Run it first and read the splits. Asked again the
  model answers a little differently, so add `--plan-out plan.json` to keep what you read,
  then `memry split-memories --plan-in plan.json` makes exactly those splits without the
  model, skipping any memory that changed since.
- Each memory is split in one transaction: a failure or a stop leaves it as it was.

Undo a split under Archive (the memory comes back and its facts are forgotten), with
`memry split-memories --undo MEMORY_ID`, or with
`POST /api/v1/memories/{id}/undo-replacement`. Over REST the command is
`POST /api/v1/memories/split` with `{"user_id": ..., "dry_run": true, "min_words": ...}`.
It costs one text-model call for each memory asked, and one more for each memory split.

Every memory has a namespace. A write that names no user (the CLI's `memry add` without
`-u`, a library call, a backup row without one) goes to the default namespace
(`MEMRY_DEFAULT_USER`, `default`), as the REST and MCP servers always did. A read without a
user still means every namespace. A store from before this may hold memories without a
namespace; `memry adopt-unscoped [--into NAMESPACE] [--dry-run]` moves them, with their
turns, entities, relations and upkeep state, into one (the default namespace unless
`--into` names another), in one transaction:

```bash
memry adopt-unscoped --dry-run   # counts per table, the tags and names that would fold
memry adopt-unscoped             # the move; a second run finds nothing to do
```

- A tag the target already has takes the memories of the moved one.
- A person or thing the target has under the same name and type, and only one, takes the
  moved one as a merge you can undo under Archive > Merged names. Any other name both have
  is left for the usual identity passes, and the report lists it.
- Upkeep state (when each pass ran, the queues, the owner's name) goes with the memories
  where the target has none of its own; where it has, its own is kept. The report lists
  both.
- Nothing is deleted. A backup with rows without a namespace restores into the default
  namespace; into the store it came from, run `adopt-unscoped` first.

Each namespace has an owner entity, the person the memories belong to. It takes the
account's name where an account named it, and is otherwise called "the user" until a
conversation states who the user is: the user gives their name, signs, is called by it, or
a memory says it ("The user's name is Cos."). Extraction asks for the name only while the
owner is "the user", so a named owner's saves pay nothing for it. The owner is then folded into the person who
carries that name (the person keeps it and becomes the owner; undo under Archive > Merged
names), or renamed to it when nobody does. The identity judge is never asked about an
owner still called "the user": "the user" is a role, and on a real store the judge read the
owner and the person it was as two people. Once named, the owner is compared like anyone.
The first name stated holds; a later different one is recorded and changes nothing unless
it corrects the first ("Cosima, not Cosmin"). An account's name wins over a stated one.

For a store saved before extraction reported stated names, `memry learn-owner [--user USER]
[--dry-run]` looks for them in what the namespace holds, and the upkeep cycle runs it once
per namespace on its own while the owner is "the user":

```bash
memry learn-owner --dry-run   # the statements found, the person chosen, what would be folded
memry learn-owner             # fold or rename; a second run finds the owner named
```

- Patterns pick out the memories, in use and forgotten, and the saved turns that may say
  the user's name ("my name is", "the user's name is", "I'm", "call me", a turn in role
  user that carries a speaker's name). With none, nothing is asked and nothing changes.
- Otherwise one text-model call per namespace reads them against the namespace's people,
  with their aliases and memory counts, and says which person they say the owner is, or
  none. Without a text model, only statements that say the name outright count.
- The first open after upgrading also opens again the judge's earlier answers on pairs of
  an owner still called "the user" with a person, so they no longer keep the two apart, and
  keeps one row for each pair of entities. Pairs you decided stay as you decided them.

Tags are entities of type `topic`. A database or backup from before that change keeps its
tags in the `categories` column and the legacy `topics`/`memory_topics` tables, which every
filter still reads. The first open of such a database gives each tag its topic entity and
each tagged memory its mention, so the Entities list and the tag counts see them, committing
user by user and recording that it did so after the last (an open stopped midway picks up
where it stopped); `memry tags-to-things [--user USER]` runs the same migration by hand. It
only reads the legacy tables, and a second run changes nothing.

When accounts or OAuth are enabled, also back up `auth.db` with `memry.db`. The JSON export
does not contain login data.

Nothing forgets a memory for its age. `memry sweep`, which invalidated memories whose
importance had decayed below a threshold, was retired in 0.2.44: a fact must not leave
search because it is old. A memory leaves use when you delete it, when a later one
replaces it, or when consolidation merges it, and each of those is recorded and can be
undone. Memories an older version let fade out stay under Forgotten, where they can be
restored.

## Managing topics and entities

The dashboard's **Upkeep** button opens three tabs: Upkeep (what needs you), Entities, and
Archive (what was removed). A badge on the button counts what is waiting.
Entities lists people, things and tags together, with a filter per type (tags are the type
"tag") and by name. Each shows its memory count; a tag can be renamed, combined, or
deleted under the current user filter, and checked entries of any type can be combined.
The same topic operations remain available at `POST /api/v1/tags/edit` for API
compatibility.

A memory card shows what the memory is about in one row of chips: the people and things
it mentions, then its tags, each with its type as the Entities list names it. A click on a
chip filters the list by it through the About filter. The API lists the tags in a memory's
`entity_links` as entities of type `topic`, and keeps its `categories`.

There are no parent tags. The pass that proposed them, its settings
(`MEMRY_TAG_ABSTRACTION`, `MEMRY_TAG_ABSTRACTION_INTERVAL_DAYS`), the `memry abstract-tags`
command and `GET /api/v1/tags/synthetic` are gone, and a tag filter finds the memories filed
under that tag only. A database that recorded parents keeps those rows, and nothing reads
them.

Entities open as hubs with aliases, a bounded description, and active
supporting memories. An entity with one memory in use has no description: the memory says
it, and no text-model call is spent on repeating it. Relations are listed under the entity they describe and can open their
members.

Upkeep runs on its own and asks only for what it will not decide: it lists the entity
merges below the gate, the memory merges a model proposed, the tag pairs that look like one
subject split in two, and the names the model judged not to be entities, each with a yes
and a no. Everything else (entity self-healing, word-for-word duplicate consolidation, and
durability scoring when a decision provider is configured and `MEMRY_DURABILITY=1` is set) runs on its interval, records what it changed, and can be paused with one switch.
`POST /api/v1/maintenance/run/<pass>` runs any pass that is on now; a pass that is off
runs neither there nor on its interval, and no run of it is recorded.

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
- **A home** is the project, product or organization a part belongs to, shown as
  `AI-Flow / privacy policy`. A stated `part_of` relation sets it, or the comparison below.
  Appearing together in memories does not: homes from co-mention measured 68% right (86%
  restricted) and are no longer derived.
- **A version or a part** gets a home from the comparison itself. Whenever Memry compares
  two entities with Jev, it asks in the same call whether one is a version, a dated
  occurrence or a part of the other. At 0.80 or more, "bildy v4" gets "bildy" as its home
  and "Tovel Forum 2025" gets "Tovel Forum", whatever their type, and Memry doesn't merge
  the two. A home stated in a memory comes first. A person or a place never gets a home
  this way.
- **A shared name** is read through home. Two entities with the same name under different
  homes are never proposed for merging. Two with the same name and nothing setting them
  apart are merged. Two people are never merged on a name alone, and two that the judge or
  you kept apart (a rejected pair, or P(different) of 0.5 or more) are never merged on
  their name, nor joined through a third.

New names are screened before they become entities. Measurements and counts ("250 ms",
"22 tests") are dropped by rule. With a decision provider, each new name gets one typed
question in the memory it came from, and a name judged a value or a role with at least 0.80
probability is not made an entity. The phrase stays on the memory, and a name that does
become an entity keeps its verdict. Names in the store without a verdict get the same
question during upkeep, and the ones judged a value or a role wait under **Upkeep** for a
yes or a no.

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
| Home from co-mention, restricted (no longer derived) | 141 labelled homes | 86% correct |
| Home from co-mention alone (no longer derived) | the same homes | 68% correct, and 47% when the home is an organization |
| Same name, not a person, nothing setting them apart | 78 past merge decisions | all 78 had been confirmed |
| Version or part at 0.80 | 427 pairs: the identity benchmark and 271 generated | no true merge held back; no wrong home outside web domains and handles; none pointing the wrong way |

An independent reader labelled the names and homes, all from one real store. Two first
drafts failed the labels and were changed: recurrence turned out to find topics like
"billing", and the first screening question listed "path" among the values, so the provider
screened out source files and street addresses. `evals/entity_structure_benchmark.py` runs
the same scoring on your own store.

## The fields you can send to POST /api/v1/memories

| Field | Meaning |
|---|---|
| `content` or `messages` | The text to save, or a list of `{"role": ..., "content": ...}` messages, one turn each. A `role` other than `user`, `assistant`, `system`, `developer`, `tool` or `function` is the speaker's name (`{"role": "Ada", "content": "I got the job"}`), and so is a `name` field. Memry shows each turn with that name as its speaker. Anything other than a text or a list of objects gets a `400`. |
| `user_id`, `agent_id`, `run_id` | The namespace, agent and run the memories belong to. Without `user_id`, Memry uses `MEMRY_DEFAULT_USER` (`default`). |
| `infer` | With `true` (the default), Memry extracts facts and reconciles them with what the store has. With `false`, Memry keeps the text as one memory. |
| `defer` | With `infer` and `defer` both `true`, Memry stores what was said at once, replies `202` and extracts the facts in the background after two minutes of quiet. A `messages` list keeps one turn per message with its speaker, as without `defer`, and extraction reads it the same way. Until then it is searchable as one memory with a `Speaker: text` line per message. |
| `said_at` | The day the content was said, as `YYYY-MM-DD` or an ISO date and time (read in UTC). Leave it out for what is said now. |
| `metadata` | Memry keeps it with the saved turns. `metadata.context` is the context label: Memry extracts related saves with one label together. |
| `categories` | Memry passes up to three of these tags to extraction as hints. With `infer=false` they are the memory's tags. |
| `memory_type`, `importance` | With `infer=false` they are the memory's type (`semantic` by default) and importance (0.5 by default). |

Send what was said, close to the words used. Memry keeps the saved turns and
shows them with each memory in later searches (`evidence`), so if you send a
summary, those searches show the summary.

Use `said_at` for content said on another day, such as an import or an earlier
conversation. Memry dates the save and its memories that day, and "yesterday"
or "last Friday" in the text counts from it. A value that is not an ISO date gets a `400` with the reason, and so
does a day after today: a future date is most likely the day something will
happen, and that date belongs in the text. A time later today is taken as now.
MCP `save_memories` takes the same `said_at`.

When a fact changes or someone corrects it, save the new statement and leave the
old memory as it is. Memry keeps the old value as dated history, or retires it
when it was wrong.

## Question keys (on by default)

With each fact it extracts, Memry writes two or three questions the fact answers. Memry
keeps them beside the memory, and every search matches the question against them as well
as against the memory's text. They cost no extra call: the extraction call writes them,
with about 40 more output tokens a fact. Memry embeds them with the memory.
`MEMRY_QUESTION_KEYS=0` (or `retrieval.question_keys` set to false in
`~/.memry/config.json`) turns them off. For memories saved before, `memry
backfill-questions` writes them.

## Entity questions (on by default)

Someone may ask "Where does my sister work?" without the sister's name. With
`retrieval.entity_questions` on, Memry starts that
search from the sister. Each person related to the owner has two or three stored questions,
written by a text model from the person's description and relations. Memry picks the person
whose stored questions alone contain the word after "my". The best match must reach
`retrieval.entity_question_bar` (0.5). Otherwise Memry starts from the owner, as with the
flag off. `MEMRY_ENTITY_QUESTIONS=0` (or `retrieval.entity_questions` set to false in
`~/.memry/config.json`) turns the flag off.

With the flag on, Memry writes the questions during upkeep. In each upkeep tick, Memry asks
the text model about the described people and things related to the owner that have no
questions yet. Memry asks again about one whose description or relations changed since.
Memry makes at most three calls of ten entities each in one tick and leaves the rest for the
next tick. To write them for a whole store at once, run `memry write-entity-questions`. With
`--dry-run`, Memry asks the model nothing and prints the number of entities, calls and
estimated tokens. Backups and exports contain the question texts, and Memry embeds them
again after a restore.

## The search log (on by default)

With `retrieval.search_log` on, Memry keeps one row per search. `MEMRY_SEARCH_LOG=0`
turns the log off, and Memry then keeps no new search. A row
contains the time, the namespace, agent and run, the query, the seeds (the entities Memry
recognised in the question), the result count and the time taken. Memry deletes rows older
than 90 days during upkeep. A namespace's rows are deleted with its memories.

The log contains what people asked, so it is left out of copies by default. `memry export`
includes it only with `--with-search-log`. The nightly snapshot copy is emptied of it unless
`snapshot.include_search_log` is true.

With `memry search-stats --days 30`, an operator gets the counts per namespace: searches,
those with seeds, those without and their share, and the ten most common searches without
seeds.

With `MEMRY_TRAFFIC_KEYS=1` (`retrieval.traffic_keys`), Memry stores a search's query as a
question key of a memory saved after it. The save must come within the same run in the hour
after the search, or within ten minutes in the same namespace. Ordered again after the save,
the search must have that memory in its first 20. Traffic keys are off by default
and need the log and question keys. With question keys off, Memry logs one warning at
startup and leaves traffic keys off.

## Search filters

Memry does not guess dates or names from the words of a question. The caller states them,
and every filter is applied before anything is ranked: the text ranking, the linked pool,
the judged pool, the set call and the context all read only the memories the filters
admit, so a memory filtered out never comes back because it matches the words better.

The agent tools (`search_memories`, `get_memory_context`, `list_memories`) take two:

| Filter | What it matches |
|---|---|
| `when` | The time a question is about: a day `2025-04-01`, a month `2025-04`, a year `2025`, or a range `2025-04-01..2025-06-30` (ends may be months or years, `2025-04..2025-06`, or left open, `2025-04..`). A memory matches when its occurrence time (`metadata["when"]`, below) overlaps the period, or, for a memory without one, when the day it was saved lies inside it. |
| `about` | One or more names, comma-separated, of people, projects, things or tags. Each is resolved in the caller's namespace, case aside, through entity names, aliases and merges, and as a tag; a memory matches any of them. |

A phrase in double quotes inside the query (`the "Blue Fig" dinner`) must appear in the
memory exactly, case aside. So "Was habe ich am 01. April 2025 gemacht?" is
`when="2025-04-01"`, and "what did Bochra work on?" is `about="Bochra Saffar"`.

Overlap means a memory dated only to April 2025 matches a question about 1 April. Its row
then says so: `"happened": "happened 2025-04 (month)"` (and `[happened 2025-04 (month)]`
in a context), so an agent does not claim it happened that day. A name found nowhere is
never dropped silently: the result carries a `note` naming it, with the closest names the
namespace holds. When filters match nothing, a filtered result says which filters were
applied, and for a time asked about it lists up to three memories nearest that time under
`nearest`, each with how many days `before` or `after` it lies. A call using `when`,
`about` or a quoted phrase answers `{filters, memories, note?, nearest?}`; a call without
them answers the plain list as before. An empty query with filters browses what they
admit, newest first by the time asked about, else by the day said.

The REST endpoints take the same filters and finer ones, for scripts and the dashboard,
in the body of `POST /api/v1/search` and `POST /api/v1/context` and the query string of
`GET /api/v1/memories`:

| Filter | What it matches |
|---|---|
| `happened` | A period as for `when`, on the occurrence time alone; a memory without one never matches. |
| `said` | A period, on the day a memory was said, the day its row shows as `said`. |
| `entity` | Names of people, projects or things (not tags), resolved as for `about`. |
| `entity_type` | Memories linked to at least one entity of a type: `person`, `organization`, `project`, `product`, `place`, `event`, `document`, `code`, `concept`, `other`. Tags are filtered with `tag`. |
| `tag` | Tag names, comma-separated or a list. |
| `contains` | An exact phrase, case aside, matched as written: quotes, `%` and `_` are not syntax. |
| `memory_type` | `semantic`, `episodic`, `procedural` or `working`. |

A search naming any of these answers `{results, filters, note?, nearest?}`; one naming
none answers the plain list. A malformed value is a 400 that says what the filter accepts.
On `GET /api/v1/memories` a name found nowhere is a 404 with the close names, as an
unknown `entity_id` is.

The older parameters still work and are no longer advertised to agents: `categories` (tags),
`entity_id` (entity ids), `since`/`until` (`YYYY-MM-DD`, on the day a memory was first
saved) and `when_since`/`when_until` (below). They are pre-filters too.

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

The `when` and `happened` filters (above) read it, as do the older `when_since`/`when_until`
(`YYYY-MM-DD`, both days inclusive), which is what makes "what is on this weekend"
answerable. A recurring `when` matches when any of its occurrences falls in the period.
"In April 2025" is stored as the whole month, `2025-04-01` to `2025-04-30`, and a whole
year the same way. The REST memory payload carries `when` and the computed
`next_occurrence`, and the dashboard memory card shows the same in a small chip. The
dashboard timeline places a memory dated to a whole month or year at the start of that
period and labels it "April 2025" or "2025"; a memory with a day or a time shows that day
and time.

To read occurrence times out of memories saved before this existed:

```bash
curl -X POST http://localhost:8080/api/v1/maintenance/run/when \
  -H 'content-type: application/json' -d '{"dry_run": true, "limit": 20}'
```

A dry run writes nothing and returns what it would set. Without `dry_run` it stores each
`when` it finds and marks the rest as checked, so a second run over the same memories
spends nothing. The pass needs an LLM; without one it reports that and changes nothing.

A date in a memory does not make it an event, and how far a text model can be trusted to
tell the difference depends on the model. Measured on 160 labelled memories from one store,
44 of them events, with two checks Memry runs on every `when` a model proposes, on the
write path and in the backfill:

| | gpt-5.6-luna | gpt-5-mini |
|---|---|---|
| The text model alone | 85% precise, 75% of events found | 56%, 77% |
| Minus write dates read back (a `when` on the recording day, in a text naming no date) | 86%, 68% | 63%, 70% |
| And the decision provider vetoes what it calls a record with 0.80 probability or more | 97%, 66% | 86%, 68% |

gpt-5-mini dates work logs and price checks after being told not to, which is one reason
it is not the OpenAI default; gpt-5.6-luna also read the 160 memories four times faster.
gpt-6-luna, the default now, has not been measured on this set; Memry applies the decision
provider's veto to its dates the same way. A wrong
`when` is worse than none, so the veto applies whenever a decision provider is configured.
Requiring the provider to say "event" was tried first and lost a fifth of the real events
for no gain in precision.

### When upkeep runs

The server runs the upkeep cycle once a night at 02:05 UTC (`MEMRY_UPKEEP_AT`, HH:MM in UTC), before the 03:30 snapshot, and never at start. Each pass still keeps its own interval (the identity pass weekly), so the nightly cycle runs only what is due. A night the server is down is skipped. "Run now" under Upkeep runs a pass at once. A cycle right after a deploy once judged about 1000 entity pairs and held requests up for 25 minutes.
