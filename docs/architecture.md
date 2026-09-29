# Memry architecture

This document records the architecture that exists in the repository, the product reason
for each consequential choice, and the limits that follow from it. It is descriptive, not
a roadmap.

## 1. Product topology

A normal deployment is one Memry Python process backed by one SQLite memory database.
Many agents, browsers, and API clients can connect to that process over MCP or REST, but
they are not independent database writers: the server owns the database connection and
serializes writes inside the process.

```text
Agents and applications
  |-- MCP over stdio (local client)
  |-- MCP streamable HTTP at /mcp
  |-- REST/JSON at /api/v1
  |-- dashboard in a browser
  `-- Python API / CLI
              |
         MemoryStore
          |       |
          |   managed enrichment worker
          |   (pending rows, bounded batches)
          |
    intelligence and retrieval
              |
       LocalBackend (SQLite)
        |-- memry.db: knowledge data and search indexes
        `-- *.usearch: optional rebuildable ANN sidecars

    account and OAuth store
              |
        auth.db: accounts, sessions, clients, and tokens
```

The VPS bundle adds Caddy in front of the single Memry process for HTTPS and reverse
proxying. There is no supported horizontally scaled or active-active write topology.

### Why SQLite is the only production store

The actual product requirement is a self-hosted memory service that is cheap to install,
backup, and operate. SQLite satisfies that requirement without an external database
service and is already used by the shipped Docker and VPS deployments. Maintaining a
second SQL implementation
did not serve a demonstrated customer deployment and made every schema or knowledge
feature require two implementations, two migrations, and two test paths. PostgreSQL was
therefore removed.

Business benefit:

- installation needs no database service, credentials, migration operator, or managed
  database bill;
- a small customer can back up and restore the durable memory state as files;
- every product feature has one production persistence path, reducing delay and parity
  defects;
- the supported operating model is easy to explain: run one Memry server and let all
  clients connect to it.

The cost is explicit: Memry does not currently support several server replicas or several
machines writing the same store, database-native high availability, or zero-downtime
horizontal write scaling. If a real product requirement later needs those capabilities,
that is a new, reviewed architecture decision with a migration plan. It is not a hidden
alternative backend.

Decision provenance: the PostgreSQL path first entered the original Git history in commit
`162b4eee3652752f70b301e7a330aa18b701492a` on 2026-07-17. The commit was authored by
Cosmin Novac and carries `Co-Authored-By: Claude Fable 5`; its stated reason was the generic
phrase "multi-writer deployments." The repository contained no named customer, deployment,
load target, or product requirement that needed that topology. A later history rewrite
folded the change into root commit `6e74c9c902f1198ef135777b3f642d74d2519e68`. The
SQLite-only decision reverses that unsupported architecture choice explicitly.

### Why Mem0 is comparison/import-only

Mem0 runtime selection was removed on 2026-07-24. Its reduced adapter cannot preserve
Memry episodes, invalidation history, normalized topics, entity links, relations,
or descriptions. Offering it as a runtime setting therefore made the same
product silently behave differently and lose features depending on one environment
variable. No user or deployment depended on that mode.

The business decision is one complete runtime product on SQLite. The optional `mem0ai`
dependency and adapter remain available only to explicit comparison/import code, where
those limits are visible and intentional.

## 2. Main layers

| Layer | Responsibility | Important rule |
|---|---|---|
| Public surfaces | Python API, CLI, REST, dashboard, MCP | They call `MemoryStore`; they do not implement memory behavior independently. |
| `MemoryStore` | Ownership checks, write/read workflows, knowledge operations | It is the product facade and the main invariant boundary. |
| Intelligence | Extraction, reconciliation, entity resolution, relation extraction, topic abstraction, descriptions, decay, consolidation | Derived outputs remain rebuildable from evidence. |
| Retrieval | FTS5/BM25, vectors, reciprocal-rank fusion, recency/importance, the linked search over entity links | Invalidated evidence is excluded by default and work is bounded. |
| Providers | LLM and embedding integrations | External providers are optional; zero-key fallbacks remain functional. |
| Production persistence | `LocalBackend` | SQLite is the sole production source of truth. |

The `MemoryBackend` interface isolates persistence and permits explicit test fixtures.
`MemoryStore` always constructs `LocalBackend` in normal operation. The optional Mem0
adapter can only be instantiated directly by comparison/import code; there is no config,
environment variable, CLI flag, REST option, or server option that selects it.

## 3. Stored knowledge model

### Evidence records

| Record | Purpose | Lifecycle |
|---|---|---|
| `Episode` | Raw input captured before derived processing, one per message | Append-only source evidence. Not searched on its own: it is shown as evidence of the memories found that rest on it (section 5). `withheld_at` ends that for good. |
| `Memory` | One derived or verbatim claim | May be updated, invalidated, or superseded; hard deletion is explicit. |
| `MemoryEvent` | Audit event for add/update/supersede/delete decisions | Append-only audit trail. |
| Embedding and FTS row | Search representation of a memory, and of an episode for choosing it as evidence | Derived and rebuildable (`memry reindex` embeds episodes an older store has no vector for). |

A memory has content, one memory type, importance, public `categories`, compatibility
`entities`, metadata, scope (`user_id`, `agent_id`, `run_id`), timestamps, source episode
IDs, and validity fields (`valid_from`, `invalid_at`, `superseded_by`). The validity fields
preserve old claims instead of pretending that the latest claim erased history.

The source episode IDs are the lines a memory rests on. Extraction numbers the lines of the
transcript it reads ("[1] Ada: ..."), one per message and so one per episode, and each fact
it returns names the lines it rests on (`sources`). The memory is linked to the episodes of
those lines. A fact that names no line, or a line the transcript does not have, rests on
every episode of its save, as every fact did before facts named their lines. A verbatim
memory rests on its own message, and a distilled fact on the raw saves of its lines. An
update merges the sources of the memories it joins (a MORE's merged text carries those of
the memory it replaced), and a restatement (reconcile's SAME, section 4) adds the episodes
of the lines it rests on to the memory it restates.

An episode's validity as evidence is `withheld_at`, beside a memory's `invalid_at`. It is
set on a memory's own source episodes when that memory is deleted for good. From then on the
episode is never shown as evidence, even of another memory resting on it, because it still
says what was deleted. A forgotten memory (out of use with nothing in its place) withholds
its episodes while it stays forgotten. This is checked when evidence is chosen, so bringing
the memory back shows them again.

### Tags (backend names: categories and topics)

The product and dashboard call deterministic classification labels such as `liver health`
or `2026 taxes` **tags**. The existing Python/REST field remains `categories`, and the memory's
JSON `categories` list stays the record every filter, backup and export reads. Each tag is also
an entity of type `topic` (one active one per namespace and normalized tag, held by a
partial unique index, created on first use; the namespace is `user_id` exactly, so `""` is
a user of its own, apart from the memories without one) and each tagged memory mentions
it, so tags, people, products and projects are one kind of thing with one merge machinery:
a tag merge is an entity merge plus a rewrite of the `categories` column.

One invariant ties the column, the legacy filter index (`topics`/`memory_topics`) and the
tag mentions, and one backend function holds it (`_file_tags_locked`), which every writer
goes through: insert, update, restore, backup import, the tag rewrite and the migrations.
The column names surviving tags only: a tag merged into another topic is written as that
topic's name, following tombstones topic to topic; one merged into a named thing keeps the
tag name and its mention goes to the thing. The index and the mentions are derived from
the column, each tag resolved once per write (once per call in a bulk rewrite). A merge
resolves the names merged away among the active topics only, never through a merged one's
tombstone, and never renames a survivor; the name it merges into is read as a column files
it, so a name merged away means its survivor and never gets a fresh topic. A name asked for
that has no topic at all gets a new one and the others fold into it, so every name merged
away keeps its tombstone. A rename is such a merge, and the new topic takes over the old
one's description, its time and metadata, with `renamed_from` naming the old id. A merge
rewrites the column of invalid memories too, so restoring a memory never brings a merged
tag back and the counts and the filters agree; a filter on the name merged away finds
nothing. Two tags merged from the entity page merge by id into the topic kept, under its
name as stored, even one a new tag could not have (over 64 characters, brackets, commas).
A save writes each name merged away as its own survivor (after "tax" went into "levies"
and "taxes" into "duties", "taxes" is written "duties") and each other tag in the obvious
canonical form it shares with the user's active tags, grouping names still active only,
and folds stored variants into it; an update does the same for its own tags alone,
reading just their obvious variants and merging nothing, so no other memory is retagged.
A deleted tag's topic entity is retired with its last mention, and the orphan purge
retires a topic nothing mentions and no tombstone points at. A named thing retired after
a tag was folded into it gives the tag back as a topic, the mentions the tag made going
with it, and the two are recorded as kept apart, so a restore brings back the thing's
named mentions only and the pair is not raised again. Tag counts and the vocabulary offered to extraction are
read from the topic entities. The legacy index is the record the first open of an upgraded
database, or `memry tags-to-things`, migrates from, committing user by user and marking the
migration done after the last, so an open stopped midway resumes where it stopped; a later
one-off pass at open files again any column a merge left naming a tag merged away.
A topic entity is never a hub, is never masked in a property vector, is never found by
a name lookup, and no link of the linked search reaches it (an open pair of a thing and
the tag of its name included); a tag and a named thing of the same name are compared by
the entity identity funnel (without a calibrated judge the tag folds into the thing by
rule), two tags by the tag question.

Mechanical separator and singular/plural duplicates are merged deterministically once two
real stored labels map to the same form. Semantic synonym merges remain reviewable.
Synthetic umbrella topics are hierarchy edges, for example `health` broader than
`liver health`. They are a browsing aid, not a retrieval mechanism, and the pass never
consumes its own output: abstraction reads a direct-tag histogram, so a generated parent
can never become a member of a broader one and decay the useful level a run at a time.
The parent label is not copied onto every child memory. Filters expand the
hierarchy in SQLite at query time, so taxonomy changes do not rewrite the memory corpus.
Topic hierarchy edges are separate from real-world entity relations.

### Entities

An entity is a stable identity hub for a referent. `EntityMention` is the authoritative
link from a memory to an entity and keeps the observed surface text.

#### Entity types and what they actually do

The set is `person`, `organization`, `project`, `product`, `place`, `event`, `document`,
`code`, `concept`, `other`. `document` and `code` were added after auditing what a real
store had dumped into `other`: on one side contracts, invoices, certificates and
registration numbers (`HRB 110232`, `TÜV Kaufvertrag`), on the other files, symbols,
tables and config keys (`lib/sync.ts`, `canUserSync`, `BILDY_AWS_S3_BUCKET`). Both are
large, coherent groups in ordinary use.

**The type does not affect search ranking.** Nothing in retrieval reads it: hybrid scoring
uses vectors, BM25, recency and importance; the linked search follows *relations* and
compared pairs between entities, which are a different thing from the entity's own type.
An `entity_id` filter selects specific entities, never a type. Three things do use it:

1. **Disambiguation guardrail.** With a calibrated judge, a known type conflict blocks an
   automatic merge, so a `document` never silently absorbs a `person` that happens to
   share its name. Absent or equal types leave the decision to the evidence. Without one,
   the type extraction gives a name is not evidence of another thing: it comes from one
   sentence (a shop is a `project` in most, a `product` in its listing's), so a name the
   store has joins its entity whatever type the mention gives it.
2. **Browsing.** Upkeep > Entities groups by type, capped per group.
3. **Cleanup triage.** Only `concept`, `other` and `event` entities are offered to the
   non-referent review, because those are where extraction puts style instructions and
   task descriptions. A `person` is never proposed for removal.

Each mention keeps the type extraction gave the name in its memory, and an entity's type
is the one most of its mentions give (a mention that gives none, a tag's or an older one,
counts for the type the entity has); on a tie it keeps its own. A merge keeps the type
most of both entities' mentions give, and on a tie the type of the one the store had
first: a second entity of a known name, made because one sentence typed it otherwise, is
the newer one.

The set is kept deliberately small. Every additional type is another way for extraction to
mis-sort, and the benefit is confined to those three uses, none of which is retrieval
quality. A new type should be added only when a real store shows a large group of entities
that the existing types describe badly, which is the standard `document` and `code` met.

#### How Memry decides that two entities are one

Where it happens (`src/memry/intelligence/identity.py`, `entities.py`, `store.py`):

1. **Extraction names the entities.** The extractor is given the store's existing entity
   names that the text may mean (a rare shared name word, or initials), and the owner's
   name ("the user" until the account names it). It writes a stored name as stored when a
   fact names that entity, and lists the owner under the owner's name.
2. **Save time (`resolve_mentions`).** Each extracted name is looked up across the whole
   namespace (not only the save's session): entities with that name or alias, plus up to
   five from the name index (a rare shared word, similar spelling, a typo, initials, a
   name close in meaning with a semantic embedder). Initials may skip the words a name
   writes in lower case ("ICAM" and "Ilustre Colegio de la Abogacía de Madrid"). A
   one-word name is also paired with the names that carry that word when at most five
   do ("Sofia" with "Sofia Marin" and "Sofia Petrescu"): in a store of under 150 names
   three such names already make the word too common to count as rare. The owner's name attaches to the owner
   entity without a comparison, and so does a name the memory already names an entity by
   (a memory rewritten by an UPDATE or an edit that still names it keeps that entity;
   compared with it, the memory sat on both sides and was left out of both, so nothing
   was left to compare and a second entity of the name was made). Every other candidate
   is compared with the new memory.
   A name the store already has joins the entity of that name with the highest P(same),
   unless the judge says "different" at 0.5 or more; the merge bar does not apply. (Held
   to it, 88 of 431 mentions of a known name became one-memory entities in a replayed
   store, and those never reached step 3.) For a name written another way, the first
   "merge" wins. Otherwise a new entity is made and each waiting pair is recorded as a
   proposal (a kept-apart pair is recorded too). When the judge cannot answer (an
   outage), the pair waits at step 0 and the weekly pass compares it again.
3. **The comparison (`compare`).** The judge (Jev) gets both entries side by side, in
   both orders, and the answers are averaged. Each side shows its name, type,
   description (if one has been built), whether it is the store owner, and up to 10
   memories (50 at the last step): the most recent 30% and the rest closest in embedding
   to the other side's. A memory naming both entries is left out of both sides. Every
   memory carries when it was recorded, when it became true if known, the saved text and
   session it came from (numbered the same on both sides), the client and the context
   label. The judge answers same / different / unsure.
4. **The decision.** Merge when P(same) reaches the bar for the evidence on the smaller
   side (`Decider.pair_merge_by_step`; for Jev 0.97 / 0.96 / 0.85 / 0.80 at 1 / 3 / 10 /
   50 memories). Keep apart for good when P(different) reaches 0.5, but only from 10
   memories on. Anything else waits, and nobody is asked.
5. **The funnel.** A waiting pair is compared again only when its smaller side reaches
   3, 10 and 50 memories, and never after that. With Jev this check runs on every save
   that mentions either side; it costs nothing unless a step was reached. A merge
   restarts the funnel for the merged entity's open pairs. An answer speaks for the two
   entities it compared: a pass applies its merges likeliest first, and once one side of
   a pair has been merged into a third, that pair is compared again on its own instead
   of joining the other side to the third ("Johnny" found to be both Johnny the
   electrician and Johnny the plumber joins the likelier; the two are then compared).
   **Conversation step (step 2).** A pair still waiting after the first comparison,
   with a side of fewer than 3 memories, is compared once more with up to 5 other
   memories from the conversations that saved that side's memories (same session, or
   the same client and context label, within 3 hours), closest in meaning, leaving out
   any memory that names either side. It runs only once those memories are an hour old,
   so a conversation still adding memories is not judged on part of them, and it is
   skipped (counted as done) when there is nothing to add. It uses the first step's bar
   until it is measured on its own.
   **Choosing among namesakes.** No answer about a first name reaches the merge bar, but
   the rest of a conversation tells namesakes apart (on 40 generated cases the right
   person scored higher in 37 instead of 34). So in the weekly pass, a name with fewer
   than 3 memories and open pairs to two or more entities, all asked at step 2, joins
   the likeliest when it leads the next by 0.10 and has P(same) of at least 0.5
   (`choose_among_candidates`; 29 right and 1 wrong on those cases). Only candidates the
   judge has not ruled out (P(different) under 0.5, stored on each pair) count: with one
   candidate left it keeps waiting, since it may be a third person. It asks the judge
   nothing.
6. **The weekly pass (`resolve_entities`, upkeep key `dedup_entities`).** It raises new
   pairs from the name index over all entities (identical names included) and pairs the
   owner with the three people whose memories are closest to its own. Names that only
   share a word written in capitals ("PR #42" and "the Dutch address PR") are looked at
   by the judge on the two names alone first, up to 10 per name and 20 names per pass;
   a pair it rules out (P(different) of 0.9 or more, provisional) is recorded as
   rejected and not looked at again. Then it compares every open pair that reached a
   new step. When the owner merges with a person, the person
   keeps the name and becomes the owner. It also removes orphan entities. The structure
   pass (`same_name_plan`) merges entities of one name only where nothing sets them
   apart: a pair kept apart (a rejected proposal, or P(different) of 0.5 or more) is
   never merged there, nor joined through a third.
7. **Descriptions** are built from up to 50 memories when an entity is opened or recalled
   into context, not on the save path.

Merge proposals never reach the Upkeep queue when a calibrated judge decides pairs.
A merge keeps what decided it on its proposal: the two entities (the one merged away stays
as a tombstone pointing at the other), the judge's answer and the step it was given at,
and, where no single answer decided it, the rule (one name that the judge did not call
different, the clear favourite among namesakes, one side with no memories), "confirmed by
you" or, for a merge made on the entity page, "merged by you". A name a save joins to an
entity the store has keeps the rule or the answer that joined it on its mention
(`EntityMention.decided`), and so does a mention a rewritten memory's new text makes. A
merge leaves one row per pair: where both entities had a pair with a third, the more
decided row stays (a decision over an open pair), of two alike the later answer, and the
other goes into the merge record.

A merge can be undone (`undo_merge`; Upkeep > Archive > Merged names, `POST
/api/v1/entities/unmerge`, `memry entities unmerge`). Each merge records what it moved
(`entity_merges`): both entity rows and names, the merged one's mentions, the relations and
pairs it pointed at the kept one (and a row it dropped for a pair both had), and the
funnel steps it restarted. The undo puts them
back, files a tag folded into a thing under the tag again, and records the pair as kept
apart ("undone by you"), so no pass merges them again on the same evidence. A memory saved
since the merge stays with the kept entity unless its mention calls it by a name only the
merged one had; that mention goes back, with the relations its memory stated since. An
undo is refused while the kept entity is itself merged into another (undo that first), and
two tags merged into one are not undone (their memories' tags were rewritten).

Removing an entity (`remove_entities`) takes its whole merge chain to the Archive with it:
every entity merged into it, however far down ("Tomi" into "T. Vell" into "Tomas Vell"),
their mentions, relations, pairs and merge records, so nothing is left pointing at an
entity that is gone. A restore brings them back where both ends still exist, so a merge
into it can still be undone; a relation comes back in use only while its memory is (one
whose memory went out of use meanwhile comes back out of use with it). A name that came
back while the entity was gone (a save named it and made a new entity) meets it as at
save: with a calibrated judge the two are a pair the funnel compares, at once when the
judge is quick enough to ask inside a save; without one they are joined by rule. The
property vectors of the entity's memories follow a removal and a restore at once.

Without a calibrated judge (a text model only, or a decision provider whose answers carry
no computed probabilities), no model is asked an identity question, at save or in upkeep,
and no model's number decides or weighs anything: a text model's own confidence could
merge nothing, and a confident "different" from it kept a pair apart for good on a guess,
while each memory naming a known person made one more entity and one more open pair. A
name the store already has (name or alias) joins its entity by rule, whatever type the
mention gives it, and the rule is kept on the mention: the one entity of that name, or of
several, the one the memory's conversation already names, else the one with the most
memories. Any other name makes a new entity. The weekly pass pairs identical names only and
joins them by the same rule, the one with more memories kept (two kept apart by a person
are never joined, not even through a third); a tag and a thing of its very name are joined
so too, the tag folding into the thing. Any other open pair waits for a person with no
confidence written on it, and the linked search gives a pair no calibrated judge answered
no "same" link.

Tags follow the same pattern (`judged_tag_merges`): candidate pairs from the name index
(no shared-word signal for tags), judged in both orders with the 10 most recent memories
per tag, merged from P(same subject) 0.55, compared when found and once more when both
tags are on 10 memories. The dashboard's suggest button asks the same question through the
same function: its last pass, when at most 20 tags are left that nothing else flagged, hands
`judged_tag_merges` every pair of them and only suggests the groups that reach 0.55. It
neither reads nor writes the pass's record of compared pairs, so a click never keeps the
weekly pass from comparing a pair of its own. No tag question is asked on the names alone:
judged on their names, "memry" read as a typo of "memory" (0.98).

The measurements behind these numbers are in the PhD repository,
`papers/memry-field-studies/findings/identity-obvious-merges.md` and
`identity-threshold-by-evidence.md`.

Each entity has two derived profile fields only:

- `description`: a bounded synthesis of active linked memories;
- `description_updated_at`: the synthesis watermark.

The description is a cache, not a fact store. It is generated lazily when the entity is
opened or selected for context, costs nothing on the normal write path, and is rebuilt
when mentions, linked memory content, invalidation, deletion, type, alias, or merge state
changes. Active linked memories remain the evidence returned with the hub.

### Relations

Entity relations are typed subject-predicate-object edges with an optional evidence memory.
Invalidating or deleting that evidence also invalidates or removes the relation. The linked
search follows relations, with the version and part links of compared pairs, only when the
query names a known entity (one link deep by default).
They are shown under the entity they describe, which is also where they are used from: an
edge only means something next to the thing it connects.

## 4. Write path

### Durable MCP save and managed enrichment

The default `save_memories(infer=true)` path is intentionally split at the safe boundary:

1. Commit the exact input as both an immutable episode and an active, searchable memory.
2. Mark that memory `pending_distillation` in its existing SQLite metadata and return the
   MCP acknowledgement. No LLM or embedding request runs before this response.
3. Wake one in-process worker. It waits until a pending ingestion group has been quiet for
   two minutes. Saves with the same user/agent/run scope and optional semantic `context`
   label are then sent through one extraction pass, capped at eight raw records per pass.
   Optional client `tags` are prompt hints, not grouping identifiers.
4. The extractor sees the whole related input while still producing small atomic facts.
   The group's episodes are embedded first, since the save made no provider call. Every
   derived fact keeps the source episode IDs of the saves whose lines it names (all of the
   group's when it names none) and the save's context
   label (before 28066d1 the label was lost; `memry restore-context`, or
   `POST /api/v1/memories/restore-context`, puts it back from the episodes, with
   `--dry-run` / `{"dry_run": true}` to count first). On success, reconcile the
   facts, run the coverage audit a direct save gets (one text-model call naming input
   details no stored fact captured; a gap is noted on each raw memory's SUPERSEDE event)
   and supersede the raw pending memories. If extraction finds no facts, keep the raw
   memories and clear their pending markers. The quiet period counts from when a save was
   queued, whatever `created_at` it was given.
5. On provider or processing failure, keep every raw memory active, record the error on
   each record, and retry with exponential backoff capped at five minutes. After a process
   restart, the worker discovers the same pending rows, including interrupted work.

The active pending memory is both usable knowledge and the recovery marker. This avoids a
second queue database or broker and ensures acknowledgement never means "accepted only in
RAM." Status is visible on MCP memory rows and in aggregate statistics.

### Synchronous library and REST write

`store.add(...)` and the REST write route retain the synchronous workflow:

1. Store raw input as episodes before inference, one per message, each with its embedding
   and full-text entry (an embedding failure leaves the episode to its words).
2. With an LLM, extract small candidate memories, types, importance, topics, entities, and
   possible relations, offered the user's tags from every run. Without an LLM, store the
   input verbatim.
3. Retrieve the five most similar memories in use of the user across runs (with the
   agent) and reconcile each candidate (`intelligence/reconcile.py`). The judge (the
   decision provider, or the text model where it abstains) sees each memory with the date
   it was said and the new fact with the save's date, and gives one of five answers. The
   answer acts the same way whatever run the memory it names belongs to; the save's run
   decides only where a new memory is stored.
   - NEW: new information, including another occurrence of the same kind of event (two
     yoga classes stay two). The fact is added.
   - SAME: it says nothing the memory does not. No second copy is stored: the save is
     recorded on the memory as evidence. Its episodes (those of the lines the fact rests
     on) join the memory's `source_episode_ids`, so they are among the turns a search
     shows with it (section 5), and a NONE event at the save's time says when it was last
     said. The memory's `updated_at` does not move.
   - MORE: it adds detail to a memory that stays true. The text model writes one text of
     both, stored as a new memory dated at the save (in the save's run, with the old one's
     tags and sources), which supersedes the old one as an update. Its names are read as an
     edit of the old memory's. With no merged text written, the new fact itself supersedes
     the old one as an update.
   - CHANGED: the memory was true and is no longer. The new memory is added and the old
     one's validity ends at its date (`invalid_at`, `superseded_by`), superseded as an
     update.
   - WRONG: the memory was never true (a correction). It is superseded as a contradiction.
   SAME, MORE, CHANGED and WRONG act only at or above the decision provider's bar for that
   answer (`Decider.reconcile_bars`, measured for Jev with `evals/reconcile_benchmark.py`;
   `supersede.confidence` for a provider nobody measured; the text model's prompt answers
   carry no confidence). Below its bar a SAME or a MORE is stored as NEW, and a CHANGED or
   a WRONG keeps both memories in use, the new one marked as a conflict that waits under
   Upkeep. CHANGED, WRONG and a MORE with no merged text also wait there when the old
   memory is rated important or was stated in two or more separate saves: the saves behind
   its evidence (the episodes it was made from and those of each save that said it again),
   a save being its run and its time, which every message of one save shares. A fact from
   one save counts once however many of its messages it rests on. An exact duplicate (normalized text) is SAME
   with no model asked, unless it is an event (either memory episodic, or with an occurrence
   time) said on another day, which the judge decides. Each SUPERSEDE event records its `kind`
   (contradiction, update, consolidation or distillation), which the Archive and search
   read; an event from before the column is classified by its reason. The Archive lists a
   memory an update or a contradiction replaced; the undo of an update brings the old one
   back beside the newer one, the undo of a contradiction forgets the newer one unless the
   person keeps both.

   Runs are read from evidence. A search restricted to a run returns the memories said in
   it: the run's own, and those whose source episodes include an episode of the run, such
   as a memory of another run a save of this one restated (SAME) or a merged text that
   carries the sources of what it replaced. No table or key records it apart from the
   episodes, so backups, restores and deletes carry it with the memory. Listing a run
   (`get_all`, `delete_all`) still reads the memories it holds as its own. (Tags, topic
   canonicalization and entity lookup read the whole user too.)
4. Store or update the memory, its tags (the column, the filter index and the topic
   mentions, `_file_tags_locked`), embedding, and FTS row.
5. Resolve entity mentions conservatively. Alias matches only narrow the candidates. A new
   name's screen verdict is kept on the entity it creates, so the weekly screen skips it.
6. Store evidence-grounded relations whose endpoints resolved in that memory.
7. Append audit events.
8. With an LLM, the coverage audit names input details no stored fact captured, as a
   warning on the result.

`add` (and `add_deferred`) take `created_at` (the episodes' and new memories' time and
`valid_from`, a MORE's merged text included, the `invalid_at` of a memory the save
supersedes, a distilled raw memory included, and the time of the events the save records,
the NONE event of a restatement included; a superseded memory keeps a later `updated_at`
it has, so `repair_updated_at` reads the same times), `memory_metadata`
(merged into every memory the save produces; a key Memry sets, such as "when", is kept)
and `now` (the day extraction and the when-check read as today). They exist for replaying
dated conversations (`evals/external_benchmarks.py`).

When existing memory text is edited manually, Memry analyzes the final text before
committing the change and replaces that memory's entity-name snapshot and authoritative
mention links together. A MORE's merged text is analyzed the same way, as an edit of the
memory it replaces, before anything is written, and its links go to the new memory. A
failed LLM analysis leaves the old text and links unchanged. In zero-key mode, existing links are retained or removed by exact known-alias
matching; discovering a brand-new entity still requires an LLM.

Entity descriptions and synthetic topic hierarchy are not mandatory write-path work. This
keeps ingestion latency and provider cost bounded.

## 5. Read path

For a normal text query:

1. FTS5 produces BM25 keyword candidates.
2. The configured embedder produces vector candidates. Small stores use exact NumPy cosine
   scoring; the optional usearch HNSW sidecar supplies candidates above its threshold.
3. Reciprocal Rank Fusion combines the candidate lists.
4. Relevance is blended with recency and importance according to configuration.
5. If canonical or alias candidate lookup resolves a query entity that is a hub (or the
   question speaks in the first person and the owner is one), the linked search runs: it
   follows the links from that entity, directed and weighted by kind, direction and
   probability (`relational_depth`, 1 by default; an open pair is a "same" link only on a
   calibrated judge's answer), adds the best memories of each entity
   linked strongly enough (read within the user, agent and run searched before the newest
   500 are taken, as the set pool's topic scan is), and orders every candidate by how well
   it states the property asked (its property vector, entity names read as "it") times how
   strongly it is about the entity named. A question naming several hubs is compared as
   written with each memory's ordinary vector, names kept, as the judge reads it: masked,
   "Why do Ada and Kai find Mira inspiring?" reads "Why do it and it find it
   inspiring?", which cannot tell the memories about Mira from anything one of them finds
   inspiring. The keyword search's best match keeps a place
   among the first 20 (`decision.rerank_pool`) whatever its score: an identifier the
   question names ("invoice 2024-117") is seen by the words alone. This is the only link
   mode; the earlier "typed"
   and "undirected" walks and the "rescue", "weighted", "inherit" and "gated" fusions were
   removed, and a config naming one is refused.
6. With `relational_relevance = "jev"` (the default "auto" is "jev" where the decision
   provider re-ranks: Jev unless `decision.rerank` is off, or a text model measured to help
   with it on; "vector" elsewhere), the decision model judges the first 20 of that
   order in one call and says whether the question asks for one property and whether it
   needs several memories. A question with one answer, or about everything, is answered
   from that call. A question needing several (a list, a total, a comparison) gets one more
   call on up to `set_pool` (80) memories not judged yet: those filed under the topics
   (tags) the first 20 share, a small topic most of them carry counting most (its size
   counted in the scope searched; of a tie at the cut, the newest are scored, as many as
   places are left and 20 more), and, where those are fewer (an untagged store, or
   they share none), the memories nearest to the members the first call found, half by
   memory vector and half by property vector (with the names read "it", one car's price
   is nearest other prices, not that car's other facts). Only with no member to start
   from is it the order past the first 20. The set's
   members from both calls come first and are returned past the limit, up to 100. In
   this mode a question naming no hub is judged the same way, the text ranking's first
   20 in one call whose two meta questions decide the set path: a set question is
   answered as above, and a question with one answer or about everything is ordered by
   the 0.35 blend of the judged score with the text ranking's position (a judged score
   under 0.15 pushed back), not by the judged score alone, which measured worse there
   (R-117: recall@3 0.844 against 0.933 on distractors_v1). A question naming a hub is
   ordered by the judged score times aboutness; an answer from a thing the named entity
   belongs to counts as far as none of the entity's own memories answers, nor one of a
   thing between them (the version it builds on). The same blend, with a call of its own,
   re-ranks a search that was neither ordered by the linked search nor judged (a tag or
   entity filter, or `relational=False`).
7. The memories found are returned with their evidence (`MemoryStore.evidence`). This is
   provenance, not a second search. The candidates are the source episodes of the results
   in use or kept as history (below), each once, credited to the best ranked memory
   resting on it. A memory kept as history shows its turns under the same rules as any
   memory: they are what was said while it held, each with its date, which a model reads
   beside the memory's "[until <date>]". A memory out of use otherwise (a contradiction,
   asked for with `include_invalid`) shows none. Only episodes of the scope searched count:
   a search of a run finds a memory another run's save restated there (section 4) and
   shows that run's turns only. None is withheld, none rests under a forgotten memory, and
   none says no more than a memory resting on it (a verbatim save). They are taken by the
   similarity of their vector to the query, with their full-text match breaking a tie. Each
   is taken while it fits `retrieval.evidence_tokens` (600 by default; 0 shows none), and
   they are returned in the order they were said. A result carries the turns credited to
   it (`SearchResult.evidence`).
8. Context reconstruction may prepend a bounded, lazily refreshed entity description. It
   then packs exact memories into the remaining token budget, leaving a share for their
   evidence (`retrieval.evidence_tokens`, at most half of what is left). The evidence of
   the memories that fit fills that share. One function renders memories for a model
   (`intelligence.context.memory_lines`), used by `reconstruct_context` and the
   benchmark runner alike. A memory reads "[happened 2023-05-07] <text> (said 8 May
   2023)": when the thing it tells happens (`metadata["when"]`, where known), and the day
   it was recorded (its last change). Both are labelled so that a model does not take the
   day a fact was written down for the day it happened. A memory kept as history reads
   "<text> (said 8 May 2023) [until 15 July 2023]": said the day it began to hold
   (`valid_from`, since taking it out of use moved its `updated_at`), and held until the
   day the memory that replaced it was said, written as that memory's "said" date is. The
   memories are followed by their evidence turns in the order they were said, each "<said
   date>: <speaker>: <text>". The MCP `search_memories` rows carry the same as data:
   `said`, `happened`, `invalid_at` for a memory out of use, and `evidence` (said,
   speaker, text). The benchmark runner passes these lines as Mem0's memory list
   (`evals/mem0_judge.py` renders nothing of its own).

Every ranked read breaks a tie by memory id (`ORDER BY updated_at DESC, id` and the like),
so memories of one time (a bulk import, a restore) rank alike in every build of a store.

The ANN file is a cache. SQLite remains authoritative, ANN candidates are exact-rescored,
and the index can be rebuilt.

A search reads the memories in use and, as history, those superseded as an update
(`models.HISTORY_KINDS`: a changed value, or a text a detail was merged into): each held
until its `invalid_at`. They are out of the ANN index with the rest of what is out of use
and few, so the vector search scans them exactly beside it. When the memory in use that
replaced one (followed through a chain of updates) is among the results, it is moved up to
just before it, so for one question the current value comes first and a question about the
past keeps its answer where it ranked. It is rendered with the date it was said and
"[until <date>]" (step 8), its source turns are shown as its evidence under the same rules
as any memory's (step 7), and MCP rows carry its `invalid_at`. Hiding them lost LoCoMo
questions about the past. A memory superseded otherwise (a
contradiction, a consolidation, a distillation) or deleted is excluded unless a caller
explicitly requests every memory (`include_invalid`). Reconcile's candidates are memories
in use only.

## 6. Product surfaces and security

- The Python API and `memry` CLI expose the same store workflows.
- `memry mcp` runs only the local stdio transport.
- Remote streamable HTTP/HTTPS MCP is available at `/mcp` only through `memry serve`, which
  hosts REST, the dashboard, OAuth endpoints when enabled, and MCP in one Starlette/Uvicorn
  process. Caddy supplies HTTPS in the VPS bundle.
- The standalone `memry mcp --transport http` launcher was removed on 2026-07-24. Removing it
  does not affect local stdio clients or remote clients pointed at a `memry serve` URL. It
  removes an MCP-only network server that bypassed Memry's account, OAuth, and bearer-key
  middleware and otherwise acted as the global administrator.
- A single operator bearer key, configured tenants, or runtime accounts can authenticate
  network calls. The operator key is the explicit global credential. The oldest/first runtime
  account is persisted as bootstrap administrator but is memory-confined to the existing
  `default` space; every later account is confined to one `<name>::default` space. The
  dashboard exposes no storage namespace selector. Administrator role and memory ownership
  are independent, and knowledge merges never cross account boundaries.
- Runtime API keys are stored as SHA-256 hashes; human passwords use scrypt with a random
  salt; comparisons are constant-time.
- OAuth uses the MCP SDK authorization-server interfaces with dynamic client registration,
  PKCE, short-lived authorization codes, access/refresh tokens, refresh rotation, and
  revocation.
- The dashboard uses an HTTP-only session cookie. OAuth discovery is mounted at the domain
  root; MCP is mounted at `/mcp`.

### Why memory and login data use separate files

This separation was kept deliberately on 2026-07-24. `memry.db` contains knowledge and
search data. `auth.db` contains accounts, password hashes, sessions, OAuth clients, and
tokens. As a result, knowledge export, import, or reset cannot overwrite who can log in or
invalidate credentials accidentally. That is the product benefit.

The cost is operational: a complete server backup must include both `memry.db` and
`auth.db` from the same point in time. They live in the same directory by default and in
the same Docker data volume, so one coordinated directory or volume snapshot captures
both. `memry export` is a lossless knowledge backup only; it does not contain login data.

## 7. Technologies used

This inventory lists technologies actually imported, executed, or shipped by the
repository. "Optional" means the default installation or deployment can function without
it.

### Runtime and data

| Technology | Required? | Where and why it is used |
|---|---:|---|
| Python 3.11+ | Yes | Application language, CLI, servers, intelligence, providers, and evals. The Docker image currently uses Python 3.12 slim. |
| SQLite through Python `sqlite3` | Yes | Sole production persistence for memories; also the current runtime account/OAuth store. |
| SQLite WAL | Yes for file databases | Permits reads while the single server process serializes writes. |
| SQLite FTS5 | Yes | Content index and BM25 keyword retrieval. |
| SQLite JSON1 | Yes | Reads the public `categories` projection and metadata aliases; normalized topic links are the indexed path. |
| NumPy | Yes | Float32 embeddings, exact cosine scoring, clustering, and vector math. |
| Pydantic 2 | Yes | Configuration and typed domain/API models. |
| JSON and JSONL | Yes | Configuration values, REST payloads, exports/imports, provider structured output, and eval datasets. |
| Python standard library cryptography primitives | Yes for accounts | `hashlib`, scrypt, SHA-256, HMAC comparison, and `secrets` for credentials and tokens. |

### Servers, protocols, and UI

| Technology | Required? | Where and why it is used |
|---|---:|---|
| Model Context Protocol Python SDK / FastMCP | Yes | MCP tools, stdio transport, streamable HTTP, and OAuth server interfaces. |
| AnyIO | Yes | Moves synchronous store/provider work off MCP event-loop tasks. |
| Starlette / ASGI | Yes for `memry serve` | REST routes, dashboard, middleware, sessions, OAuth routes, and the mounted MCP app. |
| Uvicorn | Yes for `memry serve` | Runs the combined ASGI application. |
| HTTPX | Yes | Reusable OpenAI, Voyage, and Ollama HTTP clients keep connections warm across background enrichment calls; also used by tests. |
| python-multipart | Yes for account/OAuth forms | Parses dashboard and OAuth login form bodies through Starlette. |
| REST over HTTP with JSON | Yes for network API | Application integration and dashboard data access. |
| MCP stdio | Optional surface | Zero-port local agent connection. |
| MCP streamable HTTP | Optional surface | Remote agents and multiple client devices connecting to one Memry server. |
| OAuth 2.1-style flows, DCR, and PKCE | Optional | Account sign-in for OAuth-capable MCP clients when `MEMRY_PUBLIC_URL` is configured. |
| HTML5, CSS, vanilla JavaScript, Canvas 2D | Yes for dashboard | Server-embedded dashboard and the topic galaxy visualization; no frontend build tool or framework. |

### Provider integrations and optional accelerators

| Technology | Required? | Where and why it is used |
|---|---:|---|
| Deterministic hash embeddings | Built-in fallback | Offline, zero-key fuzzy lexical vectors. |
| Anthropic Python SDK | Optional extra `memry[anthropic]` | Anthropic LLM completion and structured output. |
| OpenAI Chat Completions HTTP API | Optional | LLM extraction, reconciliation, summaries, and descriptions. |
| OpenAI Embeddings HTTP API | Optional | Semantic memory embeddings. |
| Voyage embeddings HTTP API | Optional | Alternative semantic embeddings. |
| Ollama HTTP API | Optional | Local LLM and embedding provider. |
| usearch HNSW | Optional extra `memry[ann]` | Persistent approximate-nearest-neighbor candidate sidecars for larger stores. |
| Mem0 / `mem0ai` | Optional extra `memry[mem0]` | Used only when comparison/import code explicitly instantiates the reduced adapter; never selected by the running product. |

### Build, test, and deployment

| Technology | Required? | Where and why it is used |
|---|---:|---|
| Hatchling | Build-time | Builds the Python wheel and source distribution. |
| pytest | Development | Unit, integration, tenant, server, retrieval, and migration tests. |
| GitHub Actions | Release-time | Runs tests and publishes package artifacts. |
| Docker | Optional deployment | Builds the packaged single-process server image. |
| Docker Compose | Optional deployment | Runs the Memry container and, in the VPS bundle, Caddy. |
| Caddy 2 | Optional VPS deployment | TLS termination, gzip, and reverse proxying to the single Memry process. |
| Bash and curl | Optional VPS installer | Installs and operates the bundled Docker deployment on a Linux VPS. |

#### Release rule: shipped code never changes without a version bump

The version lives in exactly one place, `__version__` in `src/memry/__init__.py`
(`pyproject.toml` reads it through hatch's dynamic version, and `GET /health`
reports it). A push to `main` whose `__version__` has no tag yet IS the release:
CI tests, builds, publishes to PyPI and creates the GitHub release `v<version>`.

`deploy/release_check.py` enforces the rule in three places, so PyPI, GitHub
`main` and any deployment cannot drift apart silently again (as happened between
v0.2.25 on 2026-07-26 and 2026-08-13, when mcp 2.0 broke the published wheel
while `main` already carried the `mcp<2` fix without a bump):

- CI (`publish.yml`, every push and pull request): fails when `src/memry`,
  `pyproject.toml`, `Dockerfile` or `requirements-docker.txt` differ from the
  tag of the version they claim to be. Docs, tests, website and deploy scripts
  may change freely without a release.
- Local pre-push hook: `git config core.hooksPath deploy/git-hooks` once per
  clone runs the same check before a push leaves the machine.
- Deployment: the VPS update script deploys only a clean tree whose HEAD is the
  released tag that PyPI already carries, and confirms `/health` afterwards.

`pypi-canary.yml` additionally installs the published package from PyPI on a
clean runner every Monday, imports it, checks it matches the latest release and
runs that tag's tests, so an ecosystem break (a new major of a dependency) shows
up as a red run within a week instead of in a user's terminal.

## 8. Current limits and non-promises

- One Memry process owns a production database. Multiple write replicas are unsupported.
- Complete backups must capture `memry.db` and `auth.db` together; ANN sidecars may be discarded and
  rebuilt.
- The Mem0 adapter is comparison/import-only and is not a supported runtime persistence path.
- There is no external IdP/SSO integration, per-key rate limiter, external queue
  service, separate vector database, or distributed cache.
- Description faithfulness and end-to-end memory quality still require public evaluation;
  a clean schema and synthetic benchmarks do not establish "best in class" quality.
- Exact inline entity highlighting is deferred because mention surfaces do not provide
  unambiguous character spans. Reliable entity chips are the shipped navigation path.
- The keyword search matches every word of the question, function words included, so in a
  store of third-person facts a rare "did" or "do" can outweigh the name a question asks
  about, and the one keyword match the linked search keeps in its first 20 is then the wrong
  one.
- A question names an entity only by one of its names or aliases in full: "Arvel" does
  not find the place stored as "Mount Arvel", so a memory naming the place beside the
  person the question names is compared through that person's links, with the place's name
  read as "it".

## 9. Decision record

| Decision | Product reason | Implemented? |
|---|---|---:|
| SQLite is the only runtime database | One complete, cheap self-hosted product is more valuable than maintaining an unused second SQL implementation. | Yes |
| Mem0 is comparison/import-only | Its adapter cannot preserve the complete Memry knowledge model and no runtime user depended on it. | Yes |
| Knowledge and login data remain in `memry.db` and `auth.db` | Knowledge restore/reset cannot overwrite credentials; a complete server backup must capture both files together. | Yes |
| Local MCP uses `memry mcp`; remote MCP uses `/mcp` from `memry serve` | This preserves local zero-port use and one network server where configured authentication is applied. The separate unauthenticated HTTP launcher added risk without a used product case. | Yes |
| Edited memory text is re-analyzed for entity links | Entity chips and entity filters must describe the current text, not names left behind by an older version. | Yes |
| The UI says tags; the public backend field remains `categories`; a tag is a topic entity, and `topics`/`memory_topics` remain the filter index | Users get one familiar word without a breaking API/schema rename, and tags merge by the same machinery as names. | Yes |
| MCP saves persist raw text before acknowledgement and enrich it in one managed worker | Agent calls return after a cheap SQLite commit instead of waiting on several provider calls, while the active pending row prevents data loss and enables restart recovery without another queue system. | Yes |
| Background work uses bounded database batches but separate prompts per memory | Bounded draining improves throughput; separate prompts preserve each user scope, provenance, retry, and failure boundary. | Yes |
| Anthropic defaults to claude-haiku-4-5 | Memory extraction is frequent background work, so the lower-cost, lower-latency model is the useful default; operators can explicitly select a larger model when quality justifies the extra cost. | Yes |
| Provider HTTP clients are reused for the store lifetime | Reusing connections removes repeated connection setup from enrichment latency without adding a service or a second execution path. | Yes |
| Reconcile answers NEW, SAME, MORE, CHANGED or WRONG and acts alike in every run; a changed value stays searchable as history; a restatement is recorded as evidence on the memory it restates | Acting only on a memory of the save's own run left every changed value live and every restatement duplicated when each session was its own run. Keeping the older value dated answers questions about the past, and the save's episodes already say which run said it, so no new record is needed. | Yes |
| A memory is linked to the lines it rests on, and memories found are shown with those source turns as evidence | A memory is a summary, and the words it came from keep what the summary left out (a feeling, a name, what a photo showed). Episodes stay provenance: they are never searched on their own, only chosen among the sources of the memories found, within a token budget, and a deleted or forgotten memory never shows them; an update's old value kept as history shows its own, as any memory found does. | Yes |

Any future consequential architecture change must be added here with its product reason and
implementation status before it is treated as decided work.
