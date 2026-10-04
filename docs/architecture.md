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
| Intelligence | Extraction, reconciliation, entity resolution, relation extraction, topic abstraction, descriptions, consolidation | Derived outputs remain rebuildable from evidence. Nothing forgets a memory for its age. |
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
| `Episode` | Raw input captured before derived processing, one per message, with its role and the speaker's `name` when the message gives one | Append-only source evidence. Not searched on its own: it is shown as evidence of the memories found that rest on it (section 5). `withheld_at` ends that for good. |
| `Memory` | One derived or verbatim claim | May be updated, invalidated, or superseded; hard deletion is explicit. |
| `MemoryEvent` | Audit event for add/update/supersede/delete decisions | Append-only audit trail. |
| Embedding and FTS row | Search representation of a memory, and of an episode for choosing it as evidence | Derived and rebuildable (`memry reindex` embeds episodes an older store has no vector for). |
| Entity name row (`entity_names`) | Each name an entity answers to, once per entity and name: its own, each alias, each wording of its mentions, with the user it is read for; found by any three letters of it, among one user's names | Derived: kept by triggers on entities and mentions, filled once when an older database is opened. |

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
partial unique index, created on first use; the namespace is `user_id` exactly, so in an
older store `""` is a user of its own, apart from the memories without one; a write gives
both the default namespace now) and each tagged memory mentions
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
rule). Two tags are an entity pair too, raised, stored and merged as any pair; the funnel
asks them the tag question instead of the pair question (Entities, below).

The dashboard shows tags where it shows people and things. Upkeep > Entities lists every
live entity, tags included, with one chip per type and its count (a topic reads "tag" on
the page), a name filter and a checkbox per row. A tag is renamed there on every memory
filed under it (`PATCH /api/v1/entities/{id}`) and deleted as the tag endpoint deletes it
(`POST /api/v1/tags/edit`, op `delete`: the memories stay). "Combine selected..." keeps the
one picked and merges each other entry into it (`POST /api/v1/entities/merge`); with a
person or thing among them only a person or thing is offered to keep, since a tag combined
with one goes into it. Two tags merged so file every memory under the tag kept, as the tag
endpoint's merge does, except that the kept tag keeps its place in a memory's list. Two
tags that may be one are listed with the other merge proposals. The map is the entity map: tags are one
of its types, off at first (the choice kept per browser with the other types), drawn as
the nodes of their topic entities and linked by co-mentions like any node; the planet and
part rules stay for named things. `GET /api/v1/map?kind=any` reads them, and `kind=named`,
the default, reads none. On a sample store of 4,000 memories, 399 tags and 800 named things
the map took a median of 257 ms before this change (with the old tag graph, which is gone),
191 ms after without tags and 307 ms with them. The memory list has one About filter over
every entity with a memory, grouped by type: a tag picked goes to `categories` and anything
else to `entity_id`, as the two filters did before. Several picks of one kind match any of
them, and a tag with a person or thing matches the memories that have both. A memory card
shows what the memory is about in one row of chips: the people and things it mentions,
then its tags, each with its type as the Entities list names it ("person", "tag"). A
chip picks its entity in the About filter, and a second click takes it off. The API's
`entity_links` lists the tags too, as entities of type `topic`, and `categories` stays.

Mechanical separator and singular/plural duplicates are merged deterministically once two
real stored labels map to the same form. Any other pair of tags goes through the entity
pairs (Entities, below).
There are no parent tags: a tag filter finds the memories filed under that tag and no
other. A question whose answer is a set reads the topics its first answers share (section
5). A database from a version that recorded parent tags keeps its `topic_relations` and
`synthetic_tags` rows; nothing reads them, and a backup no longer carries them.

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
2. **Browsing.** Upkeep > Entities lists every live entity, tags included, filtered by
   type; with every type shown, each type's rows are capped until expanded. The map's type
   menu chooses the types it draws.
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

Where it happens (`src/memry/intelligence/identity.py`, `entities.py`, `owner.py`, `store.py`):

1. **Extraction names the entities.** The extractor is given the store's existing entity
   names that the text may mean (a rare shared name word, or initials), and the owner's
   name ("the user" until the account names it or a conversation states it). It writes a
   stored name as stored when a fact names that entity, and lists the owner under the
   owner's name. While the owner has no name it also reports the user's own name
   (`user_name`) when the conversation states it, and only then (see "Who the owner is"
   below).
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
   is compared with the new memory. The owner while it is still called "the user" is no
   candidate for any other name ("User Research" shares the word "user" with it).
   A name the store already has joins the entity of that name with the highest P(same),
   unless the judge says "different" at 0.5 or more; the merge bar does not apply. (Held
   to it, 88 of 431 mentions of a known name became one-memory entities in a replayed
   store, and those never reached step 3.) For a name written another way, the first
   "merge" wins. Otherwise a new entity is made and each waiting pair is recorded as a
   proposal (a kept-apart pair is recorded too). When the judge cannot answer (an
   outage), the pair waits at step 0 and the weekly pass compares it again.
3. **The comparison (`compare`).** The judge (Jev) gets both entries side by side, in
   both orders, and the answers are averaged: asked in one order, the merge bars moved
   between runs. The check of a name the store already has is the exception: it is asked
   once, with the entity first and the new memory second, because it reads no merge bar,
   only P(different) at 0.5, which candidate is likelier, and the belongs bar. That halves
   the calls of most saves, and in two runs it decided as well as both orders. Each side
   shows its name, type, description (if one has been built), whether it is the store owner, and up to 10
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
   pairs from the name index over all entities (identical names included) and pairs a
   named owner with the three people whose memories are closest to its own (an account's
   login shares no name with the person it is). The owner still called "the user" is paired
   with nobody, and an open pair it has is not compared. Names that only
   share a word written in capitals ("PR #42" and "the Dutch address PR") are looked at
   by the judge on the two names alone first, up to 10 per name and 20 names per pass;
   a pair it rules out (P(different) of 0.9 or more, provisional) is recorded as
   rejected and not looked at again. Then it compares every open pair that reached a
   new step. When the owner merges with a person, the person
   keeps the name and becomes the owner. It also removes orphan entities. The structure
   pass (`same_name_plan`) merges entities of one name only where nothing sets them
   apart: a pair kept apart (a rejected proposal, or P(different) of 0.5 or more) is
   never merged there, nor joined through a third. The owner still called "the user" is
   left out of it.
7. **Descriptions** are built from the entity's newest 40 memories in use
   (`entities.DESCRIPTION_FACTS`) when an entity is opened or recalled into context, not
   on the save path. The writer reads each memory's own text and writes the lasting
   picture of the entity: what it is, its roles, relationships, preferences and the facts
   that stay true. It leaves one-off events, past or planned, to the memories, and
   mentions one only as far as it tells what the entity is, without its date
   (`entities.DESCRIPTION_SYSTEM`). An entity with fewer than 2 memories in use
   (`entities.DESCRIPTION_MIN_MEMORIES`) gets none: no text-model call, nothing stored,
   and nothing shown in context or on the entity page, since the memory speaks for itself
   and a description would only repeat it. One stored when it had more memories stays
   stored but is not shown until it has 2 again.

Merge proposals never reach the Upkeep queue when a calibrated judge decides pairs.
A merge keeps what decided it on its proposal: the two entities (the one merged away stays
as a tombstone pointing at the other), the judge's answer and the step it was given at,
and, where no single answer decided it, the rule (one name that the judge did not call
different, the clear favourite among namesakes, one side with no memories), "confirmed by
you", for a merge made on the entity page "merged by you", or the statement that said who
the owner is. A pair kept apart from the Upkeep list says "kept apart by you". A name a
save joins to an
entity the store has keeps the rule or the answer that joined it on its mention
(`EntityMention.decided`), and so does a mention a rewritten memory's new text makes. A
merge leaves one row per pair: where both entities had a pair with a third, the more
decided row stays (a decision over an open pair), of two alike the later answer, and the
other goes into the merge record. A pair is recorded once: `add_proposal` looks for a row
of the two, in either order, and writes in one statement, so a save and the weekly pass
running at once cannot both add it.

#### Who the owner is

Each namespace has one owner entity, the person the memories belong to. Without an
account name it is called "the user", and "the user" is a role, not a name. Asked whether
that entity and a named person are one, the judge compares two sets of memories: on a real
store it read the owner (61 memories) and "Cosmin" (363), the person it was, as two people
at P(different) 0.94-0.95, and the pair was kept apart for good, so the owner stayed "the
user". Who the owner is is stated instead, and the judge is never asked about the owner
while it is called "the user" (`identity.unnamed_owner`): not at save, not in the weekly
pass, not in the choice among namesakes, the structure pass or the check of a restored
name. Once the owner has a name it is compared like any person.

- **Where a name is stated.** Extraction reports `user_name` when the user gives their
  name ("I'm Cos", "my name is", a signature), the assistant calls the user by it, or a
  line says it ("The user's name is Cos."); never as a guess. A save whose turns in role
  user carry one speaker's name (`name`) names the owner too. Neither costs an extra call.
  The question (a rule in the prompt and a field in the answer, about 80 prompt and 6
  output tokens) is asked only while the owner is called "the user" and no account names
  it. For a named owner the prompt and schema are byte for byte those without it, which
  keeps the provider's prompt cache; so a conversation giving a named owner another name
  goes unnoticed at save. Corrections are rare, and `learn_owner_name` still takes one
  from any caller.
- **What a stated name does** (`MemoryStore.learn_owner_name`, after the save's facts are
  written). While the owner is called "the user", it is folded into the person who carries
  the name, through the merge any pair goes through: the person keeps the name and becomes
  the owner, "the user" becomes one of their merged names, the pair is confirmed with the
  statement as its reason, and Archive > Merged names undoes it. A judge's "different" on
  that pair does not stand in the way; a person's ("kept apart by you", "undone by you")
  does, and the owner then takes the name itself. With no person of that name, the owner
  is renamed to it, "the user" kept as an alias. From then on the extractor lists the owner
  under the real name. The name and what stated it (the text, the memories and the turns)
  are kept in the namespace's upkeep state (`owner_stated`).
- **Which person a name is** (`owner.person_for`): the one person who carries it as name
  or alias, ignoring case and accents. Failing that, for one word: the one person whose
  name starts with that word ("Dan" is "Dan Popescu", also beside a "Dana"), else, from
  three letters on, the one person whose first name begins with it ("Cos" is "Cosmin" when
  no other person's name starts with "Cos"). Two people who fit are no answer, and the
  owner takes the stated name itself.
- **One name holds.** The first name stated is kept. A later different name is recorded
  as a conflict and changes nothing, unless it is a correction (it names the first name
  and says it was wrong: "my name is Cosima, not Cosmin") and the owner still carries the
  first name itself. After a fold into a person, a correction is left to you: undo the
  merge. An account's name wins over any stated one; with an account name, statements are
  only recorded. An owner entity made before its account named it takes the account's
  name, or the person who carries exactly that name.
- **Stores saved before** (`learn_owner`, `memry learn-owner`, and once per namespace by
  the upkeep cycle while the owner is "the user"). Patterns pick out the memories, in use
  and forgotten (a forgotten duplicate still states the fact), and the saved turns that may
  state the name: "my name is", "the user's name is", "I'm", "call me", "the user (Cos)", a
  greeting by name in an assistant turn, turns in role user that carry a name. With none,
  nothing is asked. Otherwise one text-model call reads them against the namespace's people
  (names, aliases and memory counts) and says which person the statements say the owner is,
  or none, and the name they give. Without a text model only the statements that say the
  name outright count, by the matching rule. `--dry-run` prints the evidence, the person
  chosen and what would be folded or renamed, writing nothing.
- **The upgrade** (once, at the first open). Pairs of an owner still called "the user"
  with a person that the judge decided, rejected or answered and left open (a reason that
  starts with the judge's name, or the names-alone screen's), are opened again as never
  compared, and wait until the owner has a name; a pair a person decided says "by you" and
  stays. A rejection made from the Upkeep list before rejections said so kept the judge's
  reason and is opened again too. Rows written twice for one pair are collapsed to the one
  a merge keeps, confirmed rows staying as they are: a merge before 0.2.40 moved the merged
  entity's pairs without looking for one the kept entity had, so the owner paired with
  "Cosmin" and "Cosmin Novac" in one pass had two rows once those two were merged. What the
  upgrade opened or dropped is kept under its marker (`schema:owner-pairs:v1`,
  `schema:one-row-per-pair:v1`).

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

Two tags go through the same path, with their own candidates and their own question. The
weekly pass pairs tags on the name index without the shared-word signal
(`identity.topic_pairs`: tags are short phrases that share words across related subjects),
and stores each pair as a merge proposal like any. The funnel asks two tags the tag question
(`identity.compare_topics`), not the pair question: each tag is shown with how many
memories it is on, the named thing of its name where the store has one, and its 10 most
recent memories, in both orders. The pair merges from the provider's tag bar (0.55 for Jev)
into the tag with more memories, whose name every memory filed under the other then
carries; below the bar it waits. It is asked when found and once more when both tags are on
10 memories, never after, and no bar keeps two tags apart for good. The pair question does
worse on tags: tags in one person's store mostly file memories about that person, so no
fact contradicts "one thing". Without a calibrated judge two tags wait for a person, listed
under Upkeep with the other pairs; a yes merges them as the judge's merge does. A database
from before this kept the step of each pair of tags it had compared, and the pairs a person
had kept apart; the weekly pass turns each into its pair's proposal, so none is asked
again.

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

Time stays with the memories. Wherever a description is shown, memories are shown with
it: in the context an agent gets and in the benchmark's memory list each memory carries
the day it was said and, where known, the day its event happened, rendered when it is
read; in the identity question each fact carries its date; on the entity page the
entity's memories follow it. A description is text written once and read later, so it
does not carry the dates of one-off events: placed first in a context, the date of one
event there was read as the date of another event the question asked about. A lasting
fact may still say since when it holds. The writer reads the memories' own texts: given
their rendered lines, it wrote more dates into the description.

### Relations

Entity relations are typed subject-predicate-object edges with an optional evidence memory.
Invalidating or deleting that evidence also invalidates or removes the relation. The linked
search follows relations, with the version and part links of compared pairs, only when the
query names a known entity (one link deep by default).
They are shown under the entity they describe, which is also where they are used from: an
edge only means something next to the thing it connects.

## 4. Write path

### Durable MCP save and managed enrichment

The default `save_memories(infer=true)` path, and REST `POST /api/v1/memories` with
`defer`, are intentionally split at the safe boundary (`store.add_deferred`):

1. Commit the exact input as immutable episodes and one active, searchable memory. A text
   is one episode, said by the user. A list of messages (REST only) is kept as a direct
   save keeps it: one episode per message that says something, with its speaker. Its
   memory reads one `Speaker: text` line per message, and the messages are kept with the
   pending marker for step 4.
2. Mark that memory `pending_distillation` in its existing SQLite metadata and return the
   acknowledgement. No LLM or embedding request runs before this response.
3. Wake one in-process worker. It waits until a pending ingestion group has been quiet for
   two minutes. Saves with the same user/agent/run scope, optional semantic `context`
   label and given day (`created_at`, a save's `said_at`; `store._said_day`) are then sent
   through one extraction pass, capped at eight raw records per pass: a pass reads
   relative times against one day.
   Optional client `tags` are prompt hints, not grouping identifiers.
4. The extractor sees the whole related input while still producing small atomic facts.
   Each text is one line said by the user, and each message of a saved list is its own
   line with its speaker, so a conversation reaches the extractor as a direct save's
   does: the same numbered lines, and the same instruction when the speakers are named.
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
   input verbatim. Extraction leaves out the saving agent's own notes on how it uses
   Memry: which context label, run or tag to use or reuse, that a conversation belongs to
   a context, or what to recall next time. They say nothing about the user or the world,
   and such a note, often a restatement of the shared context label the prompt offers,
   was stored as a fact beside the real ones.
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
     edit of the old memory's. Because the merged text is dated at the save, the writer
     reads both facts as Memry shows any memory to a model, with the date each was said and
     the date it happened where that is known, and is told to write every time as the date
     or period it names, reading "last year" or "yesterday" against the day its own text
     was said, and to keep each date with its own event. Without the dates, "a road trip
     last year", said in April, was merged beside a trip of the December before as "the
     previous year's road trip", and read as a year too early. The writer may also answer
     that the new fact is about another event or thing of the same kind (another trip,
     another game, another deal): then nothing is merged and the fact is added as NEW. The
     decision provider's MORE on such a pair is often as confident as on a real added
     detail, so no bar can tell them apart, while the writer, reading both texts with their
     dates, often can. With no merged text written and no such answer, the new fact itself
     supersedes the old one as an update.

     A merge keeps a memory to one fact. It adds a detail to the same claim, event or
     attribute (a reason, a condition, who, when, how sure); another claim about the same
     subject is a memory of its own. So the writer also answers NEW when the new fact
     states another claim than the memory does: a thesis's second argument, a position it
     rejects, another decision about one project. Without that, each save that said
     something new about one subject was merged into the memory before it, until one memory
     held every claim about a thesis, and its one vector matched none of them well. The
     writer writes every merged text: after the decision provider's MORE, and after the
     text model's own MORE where no decision provider answers (the text model's own merged
     text then stands only if the writer writes nothing). The rule is in the writer's
     request only. Put in the judge's instructions, it made the text model call a reworded
     fact or a changed value NEW; worded into the decision provider's question, it lowered
     MORE on real added details.
   - CHANGED: the memory was true and is no longer. The new memory is added and the old
     one's validity ends at its date (`invalid_at`, `superseded_by`), superseded as an
     update. The decision provider's question gives two examples of it, a plan that then
     happened and a status that moved on: without them Jev read "is planning to run the
     marathon", then "ran the marathon" as MORE under its bar, and the plan stayed in use.
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
   one save counts once however many of its messages it rests on. Importance says how much
   a fact matters, not how risky replacing it is, so where such a protected memory is among
   those compared, the decision provider is also asked, in the same call, what it is
   (`STANDING_QUESTION`, about 92 words, asked only then): still true beside the new fact,
   a changeable state or status as it stood when said (a listing active, a task running, a
   document present or missing, a plan in progress), or a lasting fact or standing rule
   (health, identity, origin, a relationship, "never do X"). Read as a state at 0.6 or more
   (`STANDING_BAR`), with P(it no longer holds) at 0.8 or more (`no_longer_holds`: CHANGED
   and WRONG of the action question together; `supersede.state_confidence`), the change
   replaces the memory without asking, as an update kept as history even where the judge
   said WRONG. The top answer alone was too low a bar: on a real queue Jev answered CHANGED
   at 0.35-0.65 to a listing deleted and a task stopped, with most of the rest on WRONG; read as still true, both stay in use and
   nobody is asked; a lasting fact, a standing rule, an unsure reading and the text model's
   answers (no probabilities) wait under Upkeep as before. A partial change ("the task
   stopped", of a memory that also listed its criteria) replaces the whole memory: what
   stays true is still read in its history, and a memory that holds one fact
   (`split_memories`) does not mix the two. `memry reconcile-queue [--user U] [--apply]`
   asks the decision provider again about each question waiting under Upkeep and prints
   what the old rule and this one decide, writing nothing without `--apply`; with it a
   replacement is made by Memry with its reason and undone under Archive. An exact duplicate (normalized text) is SAME
   with no model asked, unless it is an event (either memory episodic, or with an occurrence
   time) said on another day, which the judge decides. Each SUPERSEDE event records its `kind`
   (contradiction, update, consolidation, distillation or split), which the Archive and
   search read; an event from before the column is classified by its reason. The Archive
   lists a memory an update or a contradiction replaced, and one split into single facts
   (below); the undo of an update brings the old one back beside the newer one, the undo
   of a contradiction forgets the newer one unless the person keeps both, and the undo of
   a split brings the memory back and forgets the facts it was split into.

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
and `now` (the day extraction and the when-check read as today). MCP `save_memories` and
REST `POST /api/v1/memories` take `said_at`, the day content said on another day was said,
and pass it as both `created_at` and `now` (`models.parse_said_at`: an ISO date or date and
time, in UTC; a malformed value or a day after today is refused, a later time today is
now). The benchmark runner replays dated conversations with them
(`evals/external_benchmarks.py`).

When existing memory text is edited manually, Memry analyzes the final text before
committing the change and replaces that memory's entity-name snapshot and authoritative
mention links together. A MORE's merged text is analyzed the same way, as an edit of the
memory it replaces, before anything is written, and its links go to the new memory. A
failed LLM analysis leaves the old text and links unchanged. In zero-key mode, existing links are retained or removed by exact known-alias
matching; discovering a brand-new entity still requires an LLM.

Entity descriptions are not mandatory write-path work. This keeps ingestion latency and
provider cost bounded.

### Maintenance commands

The one-time and repair commands run from the CLI on one user or every user:
`repair-dates` (each memory's `updated_at` from its audit trail), `restore-context` (the
context label from the episodes), `backfill-relations`, `backfill-entity-types`,
`backfill-property-vectors`, `tags-to-things`, `split-memories` and `adopt-unscoped`.
All but `backfill-property-vectors`, `tags-to-things` and `adopt-unscoped` also have a REST
route under `/api/v1`. `restore-context`, `tags-to-things` and `split-memories` take `--dry-run`
(`{"dry_run": true}`) to see first what they would do.

`split-memories` (`POST /api/v1/memories/split`, `intelligence/split.py`,
`MemoryStore.split_memories`) repairs memories that hold several facts, as merges made
them before a merge was kept to one fact (section 4, MORE):

- Candidates are the memories in use whose text has more than one sentence (a ".", "!"
  or "?" before a capital, or any ";"), left alone while waiting for extraction or for a
  person under Upkeep. `--min-words N` narrows the run to longer texts. A text of one
  sentence is one statement and is not asked about.
- The text model splits each candidate into single facts, each naming its subject, each
  detail kept with the fact it belongs to. A list stays one fact unless its items carry
  details of their own (a price, a date, a reason). One fact back means the memory states
  one fact (a detail or a reason can take a sentence of its own), and it is left alone.
- Every fact must state its subject, since a fact read alone later has nothing around it:
  the prompt asks for the name the memory uses, never "he", "it" or "the project" alone.
  Before anything is written, each fact is checked: it must name one of the things the
  memory is linked to, a tag the memory's text names, a name the text states (a one-word
  abbreviation such as "PR" or "API" does not count) or the owner, and must not open on
  "he", "it", "this" or the like. A fact without its subject is worse than the memory it
  came from, so one such fact keeps the memory whole, and the report names the fact.
- The coverage audit a save gets reads the facts against the memory. A split it finds
  lossy is not made; the report says what would have been lost.
- The same call says what each fact is about: the memory's linked entities, named things
  and tags, are numbered under it, and each fact comes back with the numbers of those it
  is about, as many as apply, also one it does not spell out ("The tallest bulls in Etosha
  stand 4 m." in a memory about elephants keeps Elephant, so the linked search still finds
  it). A fact keeps those and any linked entity its text names. A split that would leave a
  fact with none of the memory's entities, or one of them on no fact, is not made.
- Each fact becomes a memory with the old one's `created_at`, `updated_at`, `valid_from`,
  sources, importance, type, metadata ("when" included), run and agent, and
  `split_from`. It is linked to the entities it keeps, its tags are the tags among them (a
  five-topic summary gives each fact its own topic, not all five), and each relation rests
  on the fact that keeps both ends.
  The ADD event of each fact is dated at the old memory's `updated_at`, so `repair-dates`
  reads the same times.
- The old memory leaves use with a SUPERSEDE of kind `split` and `split_into` on it. Like a
  consolidated or distilled memory it leaves search, since its facts live on in the new
  memories. The Archive lists it with its facts; the undo (`undo_replacement`, or
  `memry split-memories --undo ID`) brings it back and forgets the facts that are still as
  they were made.
- `--dry-run` asks the model and writes nothing; the CLI prints each memory with the facts
  it would become and what each is about, for a person to read before the real run. Asked
  again, the model answers a little differently, so `--dry-run --plan-out PATH` keeps the
  splits shown and `--plan-in PATH` makes exactly those, without the model. A planned
  memory no longer in use, with a changed text, or with changed entities is skipped.
- Each memory's split is written in one transaction (`MemoryBackend.transaction`), its facts
  embedded first in one call: a failure or a stop leaves that memory as it was. A backend
  without transactions (`supports_transactions`, false for the Mem0 adapter) refuses a real
  split and runs only the dry run.
- A memory that opens on a dated heading ("Decision (2026-09-12): ..." or "2026-09-12:
  ...") gives its facts that date as `valid_from`; the date is written once, in the first
  fact. A memory with an occurrence time ("when") keeps its dates. A fact's own date is not
  read, so every fact takes the heading's.
- Walking every namespace, the commands and the upkeep scheduler read the memories without
  a namespace as one of them (`Scope.exact_user`), not as all memories at once: those are
  never upkept, deduplicated or consolidated together with anyone else's. The Mem0 adapter
  cannot ask for "no user" and refuses such a scope.
- Every write has a namespace: `MemoryStore._namespace` gives a write without a user
  (None or `""`) `config.default_user_id`, in `add`, `add_deferred`, `import_verbatim`,
  `import_backup` and the distillation of a save queued before. Reads keep no user as every
  namespace. `adopt-unscoped` (`MemoryStore.adopt_unscoped`, `LocalBackend.adopt_unscoped`)
  moves an older store's rows without one into a namespace in one transaction: a tag of
  the same name there takes the moved tag (folded as tags merge, since one active tag per
  namespace and name is an index), the only same-named thing of the same type takes the
  moved one by a recorded merge after the move, a legacy `topics` row the target holds
  gives it its links, the moved memories' tags are filed again, and the upkeep state kept
  under the empty key goes to the target where it has none. `--dry-run` counts and writes
  nothing; a second run changes nothing. The upkeep keys of None and `""` stay one key, so
  an older store's state is not orphaned before the move.
- The dashboard shows no split run, only a split made, under Archive with its undo. The
  response of `POST /api/v1/memories/split` counts the memories held back by reason:
  `no_entity`, `lost_entity`, a fact without its subject, a lossy split, and, for a plan,
  `stale`.

## 5. Read path

Every search with a text query runs one pipeline (`MemoryStore.search`), its stages in a
fixed order. Each rule lives in one stage and applies to every search, whether it has a tag,
entity or date filter, whether its question names anything, and with `relational=False`
(which has no seeds, so no linked pool).

1. **Seeds** (`MemoryStore._seeds`). The query's phrases are resolved through canonical
   and alias candidates, and so is a word that names one entity on its own
   (`graph_retrieval.named_by_a_word`): a word of that entity's names alone, no other
   entity's name or alias carrying it (the rare shared word of the name index at its
   rarest), which the store uses for that entity more likely than not: of the person's
   memories in use and turns saved that hold it, more than half are the entity's memories
   or the turns they rest on. "Arvel" finds the place stored as "Mount Arvel"; "park", a
   word of twenty park names, finds none, nor does "city" in "New York City" where most of
   what says "city" is about other places. The names that hold a word are found by its
   letter trigrams in `entity_names` (FTS5), among the searched user's names alone, and a
   phrase that is only an alias through the same table, so neither lookup reads every name
   the store has, nor another account's. Only hubs are kept: a stray phrase stored as an
   entity does not decide what a search is about. The longest-name rule then runs among
   the hubs: "bildy v4" and not also "bildy", since a search from bildy reaches every
   version below it. Because the hubs are kept first, a stray entity whose name holds a
   hub's ("bildy sync", one memory, no type) does not hide the hub. A question naming no
   hub that speaks in the first person ("Where do I live?") is about the store's owner,
   when the owner is a hub.
2. **Candidates**, as deep for every search (eight per result asked for, at least 40 and
   at most 500). The text ranking: FTS5 BM25 keyword candidates, each word of the question
   weighed by how rare it is in everything the store holds, its memories and the turns
   they were said in (`LocalBackend.keyword_search`; the turns only weigh the words, they
   are not searched): in a store of third-person facts "did" and "do" are rare among the
   memories and common in what was said, and weighed by the memories alone they outweighed
   the name a question asks about. Without turns a word weighs as `bm25()` weighs it. The
   first candidates are found exactly, without scoring every memory a common word is in
   (MaxScore, `LocalBackend._keyword_scores`): a word adds at most its weight times its
   idf times 2.2 to a memory's score. The words held by few memories are scored in all of
   them and set a floor under the score of the last candidate kept. The common words that
   together cannot reach that floor are scored only for the memories found otherwise; the
   others are read in one query, best first, until what they can still add falls below the
   floor. Every read keeps to the search's scope and filters in SQL: a hosted memry keeps
   every account in one file, and another account's memories are walked in the index,
   never scored. Beside them, the configured embedder's vector candidates (exact NumPy
   cosine scoring in small stores, the optional usearch HNSW sidecar above its threshold);
   the two are combined with Reciprocal Rank Fusion and blended with recency and
   importance according to configuration. With seeds, the linked pool: the links from the
   seeds are followed, directed and weighted by kind, direction and probability
   (`relational_depth`, 1 by default; an open pair is a "same" link only on a calibrated
   judge's answer), and every entity they reach, however weakly, adds the 10 of its
   memories that best state the property asked, chosen among its newest 500. Every filter
   is applied here, to both, before anything is ordered or judged: the user and agent, the
   run (a run's memories are those said in it, section 4), history (the memories in use
   and those kept as history, below; every memory with `include_invalid`), the tags, the
   entity, and the date windows (`since`/`until` on when a memory was saved,
   `when_since`/`when_until` on when what it tells happens), and the filters a caller
   states (`memry.filters.Filters`: `when`, `about` and a quoted phrase for agents, and
   for REST also `happened`, `said`, `entity`, `entity_type`, `tag`, `contains`,
   `memory_type`). The tags, entity, run and history are kept to in SQL before any limit
   counts. The filters read from each memory (the periods, the phrases, the memory and
   entity types, `about`) are read once over the user's memories before anything is
   ranked (`MemoryStore._admitted`), and the set they admit goes into the SQL of every
   stage as one more condition (`_Reads.among`, `m.id IN json_each(?)`): no stage takes
   its first N from memories they drop, so a memory of April 2025 is found under 80 of
   April 2027 that match the words better. Names are resolved in the caller's namespace
   (`resolve_filters`); an unknown one matches nothing and says so, with the close names.
   A filtered context shows no entity description, which is written from all of an
   entity's memories, and an empty one lists the memories nearest the time asked about
   (`nearest_dated`).
3. **Order**. With seeds, the linked order: every candidate by how well it states the
   property asked (its property vector, the names of the entities the links reach read
   "it") times how strongly it is about the entity named (aboutness), a tie by memory id.
   A memory of an entity the links reach counts at least as much as one of an entity they
   do not reach or one naming none: a link, however weak, never ranks below no link.
   A question naming several hubs is compared as written with each memory's ordinary
   vector, names kept: masked, "Why do Ada and Kai find Mira inspiring?" reads "Why do it
   and it find it inspiring?", which cannot tell the memories about Mira from anything one
   of them finds inspiring. Without seeds, the text ranking's order.
4. **Judged pool**: the first 20 of the order (`decision.rerank_pool`), which the decision
   model reads where it judges. The keyword search's best match keeps a place among them
   whatever its score, on every search, judged or not: an identifier the question names
   ("invoice 2024-117") is seen by the words alone, while the vectors cannot tell it from
   another number and newer memories matching more of the question's other words fill the
   text ranking's first places.
5. **Judge**. The decision model judges the pool in one call, in one wording
   (`MemoryStore._judged_relevance`: whether someone who reads only the memory can answer
   the question), and says whether the question asks for one property and whether it
   needs several memories. With one seed, "it" stands for the seed in the question, and in
   each memory for the entity the links reach most strongly (a tie by entity id, not by
   which mention was written first) and the things that entity more likely than not
   belongs to; with several seeds or none, the question and the memories are read as
   written. A search is judged if and only if `relational_relevance` is "jev"; the
   default "auto" is "jev" where the decision provider re-ranks (Jev unless
   `decision.rerank` is off, or a text model measured to help with it on), and
   `decision.rerank` does nothing else. With "vector" no search is judged.
6. **Set call**. A question with one answer, or about everything, is answered from the
   first call. A question needing several (a list, a total, a comparison) gets one more
   call on up to `set_pool` (80) memories not judged yet, read as stage 2 reads (the same
   scope, history, tags, entity and date windows): those filed under the topics (tags) the
   first 20 share, a small topic most of them carry counting most (its size counted where
   the search looks; of a tie at the cut, the newest are scored, as many as places are
   left and 20 more), and, where those are fewer (an untagged store, or they share none),
   the memories nearest to the members the first call found, half by memory vector and
   half by property vector (with the names read "it", one car's price is nearest other
   prices, not that car's other facts). Only with no member to start from is it the order
   past the first 20. The set's members are found over both calls' scores.
7. **Final order, limit, evidence**. A judged search is ordered by the judged score, times
   aboutness with seeds, the set's members first; they are returned past the limit, up to
   100. With seeds, an answer from a thing the named entity belongs to counts as far as
   none of the entity's own memories answers, nor one of a thing between them (the
   version it builds on), and never less than an answer about something else judged the
   same, as for any weak link. The judged score counts to the power of P(the question
   asks for one property), so on "Show everything about it" aboutness alone orders the
   list. A tie keeps the order judged, whose ties go by memory id; the
   memories not judged follow in their order. A search not judged keeps its order. A
   question naming no hub was once ordered by a blend of the judged score and the text
   ranking's position, because the judgement alone measured worse in the wording the
   re-rank then asked in; measured again on `distractors_v1` in the one wording every
   search now asks in, the judgement alone put an answer first on every question, as well
   as any blend did, and the blend was removed (R-117). A memory kept as history then
   comes right after the memory in use that replaced it (below), and the results are cut
   to the limit.

   The memories found are returned with their evidence (`MemoryStore.evidence`). This is
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
Context reconstruction (`reconstruct_context`) runs a search and puts first the
descriptions of the entities the query names (`described_entities`: the first three it
names, each description refreshed where stale, as many as fit a quarter of the budget,
from 80 to 300 tokens). It then packs exact memories into the remaining token
budget, leaving a share for their evidence (`retrieval.evidence_tokens`, at most half of
what is left). The evidence of the memories that fit fills that share. One function
renders a context for a model (`intelligence.context.context_lines`), used by
`reconstruct_context` and the benchmark runner alike. An entity reads "Caroline
(person): <description>". A memory reads "[happened
2023-05-07] <text> (said 8 May 2023)": when the thing it tells happens (`metadata["when"]`,
where known), and the day it was recorded (its last change). Both are labelled so that a
model does not take the day a fact was written down for the day it happened. A memory kept
as history reads "<text> (said 8 May 2023) [until 15 July 2023]": said the day it began to
hold (`valid_from`, since taking it out of use moved its `updated_at`), and held until the
day the memory that replaced it was said, written as that memory's "said" date is. The
memories are followed by their evidence turns in the order they were said, each "<said
date>: <speaker>: <text>", the speaker being the message's `name` when it gave one, else
its role (an episode saved before names were kept shows its role). The role still decides
what a role decides, such as whether "the user" is the owner. The MCP `search_memories` rows carry the same as data: `said`,
`happened`, `invalid_at` for a memory out of use, and `evidence` (said, speaker, text). The
benchmark runner passes these lines, the descriptions first, as Mem0's memory list
(`evals/mem0_judge.py` renders nothing of its own); `--no-descriptions` leaves the
descriptions out, for an ablation.

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
"[until <date>]" (context reconstruction, above), its source turns are shown as its
evidence under the same rules as any memory's (stage 7), and MCP rows carry its
`invalid_at`. Hiding them lost LoCoMo questions about the past. Deleting one forgets it as
deleting a memory in use does: it no longer records what replaced it, search reads it no
more, and it is listed as forgotten, where it can be brought back or purged. A memory
superseded otherwise (a contradiction, a consolidation, a distillation) or deleted is
excluded unless a caller explicitly requests every memory (`include_invalid`). Reconcile's
candidates are memories in use only.

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
| SQLite FTS5 | Yes | Content index and BM25 keyword retrieval; with the trigram tokenizer (SQLite 3.34 and later) the index of entity names. Without it every name is read. |
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
| HTML5, CSS, vanilla JavaScript, Canvas 2D | Yes for dashboard | Server-embedded dashboard and the galaxy map of entities (tags among them); no frontend build tool or framework. |

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
- The keyword search weighs a word by the turns only where the store holds them: in
  memories imported without their turns a rare "did" or "do" can still outweigh the name a
  question asks about, and the one keyword match a search keeps in its judged pool (stage
  4) is then the wrong one.
- A word of a longer name seeds a search only when no other entity's names carry it:
  "Arvel" finds "Mount Arvel", but not while "Arvel Lodge" is also stored.
- Two events or things of one kind can still become one memory: the merge writer tells
  many such pairs apart, not all (two injuries, two paintings, two bowls told by the same
  person). The merged text then usually says each fact with its own date, but a writer
  that takes the two for one event can still give that event one date.
- The line between a detail of one claim and another claim about the same thing is the
  writer's reading. It keeps apart most other claims, and sometimes also a detail given
  in words of its own (which kind of dog, which book is the favorite); such a detail is
  then a memory beside the one it adds to, and nothing is lost.

## 9. Decision record

| Decision | Product reason | Implemented? |
|---|---|---:|
| SQLite is the only runtime database | One complete, cheap self-hosted product is more valuable than maintaining an unused second SQL implementation. | Yes |
| Mem0 is comparison/import-only | Its adapter cannot preserve the complete Memry knowledge model and no runtime user depended on it. | Yes |
| Knowledge and login data remain in `memry.db` and `auth.db` | Knowledge restore/reset cannot overwrite credentials; a complete server backup must capture both files together. | Yes |
| Local MCP uses `memry mcp`; remote MCP uses `/mcp` from `memry serve` | This preserves local zero-port use and one network server where configured authentication is applied. The separate unauthenticated HTTP launcher added risk without a used product case. | Yes |
| Edited memory text is re-analyzed for entity links | Entity chips and entity filters must describe the current text, not names left behind by an older version. | Yes |
| The UI says tags; the public backend field remains `categories`; a tag is a topic entity, listed, combined and drawn with people and things in one Entities view filtered by type, and picked in one About filter that sends it to `categories`; `topics`/`memory_topics` remain the filter index | Users get one familiar word without a breaking API/schema rename, and tags merge by the same machinery as names, so one view holds every entity that machinery merges. The tag filter still reaches the tags under a broader one, which the entity filter does not. | Yes |
| MCP saves persist raw text before acknowledgement and enrich it in one managed worker | Agent calls return after a cheap SQLite commit instead of waiting on several provider calls, while the active pending row prevents data loss and enables restart recovery without another queue system. | Yes |
| Background work uses bounded database batches but separate prompts per memory | Bounded draining improves throughput; separate prompts preserve each user scope, provenance, retry, and failure boundary. | Yes |
| Anthropic defaults to claude-haiku-4-5 | Memory extraction is frequent background work, so the lower-cost, lower-latency model is the useful default; operators can explicitly select a larger model when quality justifies the extra cost. | Yes |
| Provider HTTP clients are reused for the store lifetime | Reusing connections removes repeated connection setup from enrichment latency without adding a service or a second execution path. | Yes |
| Reconcile answers NEW, SAME, MORE, CHANGED or WRONG and acts alike in every run; a changed value stays searchable as history; a restatement is recorded as evidence on the memory it restates | Acting only on a memory of the save's own run left every changed value live and every restatement duplicated when each session was its own run. Keeping the older value dated answers questions about the past, and the save's episodes already say which run said it, so no new record is needed. | Yes |
| A memory holds one fact: a merge adds a detail to the same claim, event or attribute, another claim about the same subject is its own memory, and the merge writer writes every merged text | One vector per memory matches one fact well. Merges that folded each new claim about a subject into one memory left memories that no search for any one of their claims found well. | Yes |
| `split-memories` repairs a memory that holds several facts by replacing it with one memory per fact, undoable from the Archive | The repair must keep what the memory rested on (dates, sources, tags, links, run and agent) and be reversible, since the split is a model's reading; a lossy split is not made. | Yes |
| A memory is linked to the lines it rests on, and memories found are shown with those source turns as evidence | A memory is a summary, and the words it came from keep what the summary left out (a feeling, a name, what a photo showed). Episodes stay provenance: they are never searched on their own, only chosen among the sources of the memories found, within a token budget, and a deleted or forgotten memory never shows them; an update's old value kept as history shows its own, as any memory found does. | Yes |

Any future consequential architecture change must be added here with its product reason and
implementation status before it is treated as decided work.
