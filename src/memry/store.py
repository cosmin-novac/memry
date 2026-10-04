"""MemoryStore - the public API of Memry.

Applications (and the MCP/REST servers) call this facade only; storage
backends, LLMs, and embedders are all replaceable underneath it.

    from memry import MemoryStore

    store = MemoryStore()
    store.add("I'm Ada. I prefer TypeScript and live in Berlin.", user_id="ada")
    results = store.search("where does the user live?", user_id="ada")
    context = store.reconstruct_context("help me set up my editor", user_id="ada")
"""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import hashlib
import json
import logging
import re
import threading
import time
from typing import Any, Callable

import numpy as np

log = logging.getLogger("memry")

from .backends.base import MemoryBackend
from .backends.local import LocalBackend
from .config import Config
from .intelligence.clustering import (
    obvious_canonical_merges,
    obvious_variant_prefix,
)
from .intelligence.consolidate import judge_group, representative, similarity_groups
from .intelligence.context import (
    CONTEXT_TOKENS,
    build_context,
    description_budget,
    entities_fitting,
    entities_text,
    estimate_tokens,
    fitting,
    turn_line,
)
from .intelligence.decay import (
    DURABILITY_KEY,
    score_durability,
)
from .intelligence.entities import (
    _gate,
    classify_entity_types,
    DESCRIPTION_FACTS,
    DESCRIPTION_MIN_MEMORIES,
    judge_entity_referents,
    non_referent_reason,
    screen_names,
    SCREEN_GATE,
    SCREEN_SKIPS,
    propose_same_name_duplicates,
    resolve_mentions,
    resolve_open_proposals,
    synthesize_entity_description,
)
from .intelligence.graph_retrieval import (
    FAMILY_SCAN,
    FAMILY_TOP,
    LOW,
    SET_BAR,
    SET_NEAREST,
    SET_RESULT_CAP,
    SET_SCAN,
    SET_SHARED,
    SET_TIE_MARGIN,
    aboutness,
    activation_paths,
    detect_query_entities,
    homes_of,
    longest_names,
    mask_first_person,
    mask_names,
    parts_of,
    set_members,
    speaks_in_first_person,
)
from .intelligence.identity import (
    BELONGS_BAR,
    NameIndex,
    belonging,
    decided_by_a_person,
    fold_topic,
    is_topic,
    judges_pairs,
    merge_pair,
    unnamed_owner,
)
from .intelligence.owner import (
    STATEMENT_CHARS,
    Statement,
    choose_owner,
    fold as fold_name,
    is_correction,
    person_for,
    rule_choice,
    statements_in,
)
from .intelligence.extraction import (
    OWNER_PLACEHOLDER,
    VOCABULARY_LIMIT,
    clean_stated_name,
    extract_facts,
    extract_relations,
    speaker_name,
    speaks_with_the_user,
    verbatim_candidates,
    verify_coverage,
)
from .intelligence.reconcile import (
    CONFLICT_KEY,
    UPDATE_SUPERSEDE_REASON,
    _decide_action,
    bar_for,
    held_back,
    no_longer_holds,
    reconcile_candidate,
    reconcile_state,
    replacement_verdict,
    saves_of,
)
from .intelligence.structure import (
    ANCHOR_TYPES,
    Node,
    derive_homes,
    hub_reason,
    is_hub,
    same_name_plan,
)
from .intelligence.when import confirm_whens, extract_when, overlaps as when_overlaps
from .models import (
    MEMORY_TYPES,
    AddAction,
    AddResult,
    CandidateFact,
    ContextResult,
    Entity,
    EntityMention,
    Episode,
    EvidenceTurn,
    Memory,
    MemoryEvent,
    MemoryType,
    MergeProposal,
    Relation,
    Scope,
    SearchResult,
    TOPIC_TYPE,
    clean_tags,
    later_ts,
    parse_ts,
    same_ts,
    utcnow,
)
from .providers.embeddings import Embedder, build_embedder
from .providers.decisions import NEVER_AUTO_MERGE, Decider, Noul, build_decider
from .providers.llm import LLM, build_llm
from .retrieval import hybrid_search


_ENRICHMENT_KEY = "_enrichment"
_ENRICHMENT_BATCH_SIZE = 8
_ENRICHMENT_MAX_BACKOFF_SECONDS = 300
#: How many memories ``split_memories`` asks the text model about at once.
_SPLIT_WORKERS = 4


def _queued_at(memory: Memory) -> datetime:
    """When a pending save was queued: the quiet period counts from it. A save
    given an earlier ``created_at`` (``add_deferred``) still waits its turn."""
    job = (memory.metadata or {}).get(_ENRICHMENT_KEY) or {}
    return parse_ts(job.get("queued_at") or memory.created_at)


def _said_day(memory: Memory) -> str | None:
    """The day a pending save was given as its time (``add_deferred``'s
    ``created_at``, in UTC), or None for a save dated when it is distilled."""
    given = ((memory.metadata or {}).get(_ENRICHMENT_KEY) or {}).get("created_at")
    if not given:
        return None
    try:
        return parse_ts(str(given)).astimezone(timezone.utc).date().isoformat()
    except (ValueError, OverflowError):
        return str(given)[:10]


def _ingestion_context(metadata: dict[str, Any] | None) -> str:
    return " ".join(str((metadata or {}).get("context") or "").split())[:200]


def _as_messages(content: str | list[dict[str, str]]) -> list[dict[str, str]]:
    """A save's content as messages: a text is one message, said by the user."""
    return [{"role": "user", "content": content}] if isinstance(content, str) else content


def _says_something(message: dict[str, str]) -> bool:
    return bool((message.get("content") or "").strip())


def _said_episodes(
    messages: list[dict[str, str]],
    *,
    user_id: str | None,
    agent_id: str | None,
    run_id: str | None,
    metadata: dict[str, Any] | None,
    created_at: str,
) -> list[Episode]:
    """The turns a save keeps, a direct save's (``add``) and a deferred one's
    (``add_deferred``) alike: one episode per message that says something, in
    order, with its role and the speaker's name it gives (``name``), all at
    the save's time. They are the lines extraction numbers
    (``extraction._transcript``)."""
    return [
        Episode(
            content=m.get("content", ""),
            role=m.get("role", "user"),
            name=speaker_name(m) or None,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            metadata=metadata or {},
            created_at=created_at,
        )
        for m in messages
        if _says_something(m)
    ]


def _speaker_lines(messages: list[dict[str, str]]) -> str:
    """Messages as a person reads them, the text of a deferred save of
    messages while it waits: one "Speaker: text" line per message that says
    something, the speaker being its ``name``, else its role."""
    lines = []
    for m in messages:
        if _says_something(m):
            lines.append(f"{speaker_name(m) or m.get('role', 'user')}: "
                         f"{str(m['content']).strip()}")
    return "\n".join(lines)


def _pending_lines(memory: Memory) -> tuple[list[dict[str, str]], list[list[str]]]:
    """What a pending memory says, as the messages extraction reads, and the
    episodes of each line. A deferred save of messages keeps them with its
    work marker (``add_deferred``): one line each, resting on its own episode,
    which are the memory's first sources in order (a restatement adds its
    episodes after them). Any other pending memory, and one whose text was
    edited while it waited, is one line, its text said by the user, resting
    on all its episodes."""
    job = (memory.metadata or {}).get(_ENRICHMENT_KEY) or {}
    sources = list(memory.source_episode_ids or [])
    said = [dict(m) for m in job.get("messages") or [] if _says_something(m)]
    if said and memory.content == _speaker_lines(said):
        return said, [[episode_id] for episode_id in sources[:len(said)]]
    if not memory.content.strip():
        return [], []
    return [{"role": "user", "content": memory.content}], [sources]


def _keep_context(candidates: list[CandidateFact], context: str) -> None:
    """Facts extracted from a save keep the save's context label. The identity
    judge is shown it with each fact, and it tells which memories came from one
    conversation when the client sent no session id."""
    if context:
        for candidate in candidates:
            candidate.metadata.setdefault("context", context)


def _with_memory_metadata(
    candidates: list[CandidateFact], memory_metadata: dict[str, Any] | None
) -> None:
    """Merge a save's ``memory_metadata`` into every candidate's metadata. A
    key Memry set for the memory itself (its "when", its "context", a pending
    marker) is kept: the caller's value fills in, it does not overwrite."""
    if not memory_metadata:
        return
    for candidate in candidates:
        candidate.metadata = {**memory_metadata, **(candidate.metadata or {})}


def _client_tag_hints(
    metadata: dict[str, Any] | None,
    categories: list[str] | None = None,
) -> list[str]:
    raw: list[Any] = list(categories or [])
    supplied = (metadata or {}).get("tag_hints")
    if isinstance(supplied, str):
        raw.append(supplied)
    elif isinstance(supplied, list):
        raw.extend(supplied)
    hints: list[str] = []
    for value in raw:
        hint = " ".join(str(value).strip().lower().split())[:80]
        if hint and hint not in hints:
            hints.append(hint)
        if len(hints) == 3:
            break
    return hints


def _normalized_content(text: str) -> str:
    """Casefolded, punctuation-free text, for spotting identical restatements."""
    return " ".join(re.findall(r"[^\W_]+", (text or "").casefold()))


def _owned(record: Any, owner_prefix: str | None) -> bool:
    """Ownership gate for every id-addressed operation.

    ``owner_prefix`` None means operator access with no confinement. A selector
    ending in ``::`` is a tenant prefix; any other value is one exact account
    namespace.

    This lives in the store rather than at each call site on purpose. Ids are
    guessable-ish and callers are many (REST handlers, MCP tools, the CLI, the
    dashboard); one forgotten check is a cross-account read. Putting the gate
    behind the same door as the data means a new caller cannot skip it.
    """
    if record is None:
        return False
    if owner_prefix is None:
        return True
    user_id = getattr(record, "user_id", None)
    if not user_id:
        return False
    value = str(user_id)
    return (
        value.startswith(owner_prefix)
        if owner_prefix.endswith("::")
        else value == owner_prefix
    )


class _Owner:
    """A namespace with no record behind it, for ``_owned``.

    Retired entities are rows in the trash, not ``Entity`` objects, and the
    ownership gate reads ``user_id`` off a record; this carries one.
    """

    __slots__ = ("user_id",)

    def __init__(self, user_id: str | None) -> None:
        self.user_id = user_id


#: The tables of a backup whose rows carry a namespace (``import_backup``).
_NAMESPACED_BACKUP_TABLES = (
    "episodes", "memories", "topics", "entities", "entity_proposals", "relations")


# The keys of a namespace's upkeep state. None and "" share one key: a store
# from before every write had a namespace keeps its state there, which
# ``MemoryStore.adopt_unscoped`` carries over to the namespace its memories
# go to. No write makes a memory without a namespace now, so only such a
# store reads them; they stay as they are so that its state is not orphaned.
def _dedup_run_key(user_id: str | None) -> str:
    return f"entity_dedup:v2:last_run:{user_id or ''}"


def _consolidation_run_key(user_id: str | None) -> str:
    return f"consolidation:last_run:{user_id or ''}"


def _upkeep_key(name: str, user_id: str | None) -> str:
    return f"upkeep:{name}:{user_id or ''}"


#: The reason on a pair of the owner without a name and a person, opened
#: again because only the judge had decided it (``_settle_owner_pairs``).
OWNER_PAIR_REOPENED = ("opened again: the judge was asked about the owner before "
                       "the owner had a name")


#: Meta keys that are the store's own settings, not a namespace's state: the
#: pause switch and each pass turned on or off (``maintenance:``). A reset
#: keeps them; the queues, the owner and when each pass ran go.
_SETTINGS_KEYS = ("maintenance:",)


def _group_id(parts) -> str:
    """A stable, URL-safe id for a set of names or memory ids."""
    joined = "\x1f".join(sorted(str(part) for part in parts))
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]


def _due(last_run: str | None, interval_days: float, now: datetime) -> bool:
    """Has ``interval_days`` elapsed since ``last_run``? Never run, or an
    unreadable stamp, counts as due rather than wedging the scheduler."""
    if not last_run:
        return True
    try:
        last = datetime.fromisoformat(last_run)
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return True
    return (now - last) >= timedelta(days=max(interval_days, 0.0))


def _forgetting_trigger(event: Any) -> str:
    """What made a memory go, as a sentence someone can read."""
    if event is None:
        return "Nothing recorded what removed it. It predates the event log."
    reason = event.reason or ""
    if event.actor == "user":
        return "You deleted it."
    if event.actor == "decay":
        # the forgetting sweep of versions before 0.2.44; nothing forgets by age now
        score = re.search(r"importance ([0-9.]+) < ([0-9.]+)", reason)
        if score:
            return (f"The forgetting sweep, since retired, removed it: its importance had "
                    f"faded to {score.group(1)}, below the {score.group(2)} it needed to stay.")
        return "The forgetting sweep, since retired, removed it: its importance had faded too far."
    if event.event == "SUPERSEDE" and "into 0 fact" in reason:
        return ("It was a raw saved message, and distilling it produced nothing new: "
                "every fact in it was already stored, or there was nothing to keep.")
    return reason or f"Removed by {event.actor or 'the system'}, with no reason recorded."


def _is_update_supersede(event: MemoryEvent) -> bool:
    """A SUPERSEDE of an update (``reconcile.SUPERSEDE_KIND``): a newer memory
    said what changed, merged a detail in, or added to it with no merged text
    written. The old one was never contradicted: it held until then, and
    stays searchable as history (``models.HISTORY_KINDS``). Read from the
    event's ``kind``; an older row without one, from its reason."""
    if event.kind is not None:
        return event.kind == "update"
    return (event.reason or "").startswith(UPDATE_SUPERSEDE_REASON)


def _coverage_warning(missing: list[str]) -> str:
    """The warning a save returns when the coverage audit
    (``MemoryStore._coverage_gaps``) names details no fact captured; the
    same for a direct save and a distillation."""
    return ("some details were not captured as facts; consider saving "
            "them explicitly: " + "; ".join(missing))


def _conflict_mark(memory: Memory) -> dict[str, Any]:
    """The conflict marker of a memory kept beside the one it would have
    replaced (``reconcile.CONFLICT_KEY``), empty when it has none. Its
    ``kind`` is "update" when a change (or a MORE with no merged text) was
    held back, and absent for a contradiction."""
    mark = (memory.metadata or {}).get(CONFLICT_KEY)
    return mark if isinstance(mark, dict) else {}


def _is_contradiction(event: MemoryEvent) -> bool:
    """A SUPERSEDE that reconciliation made because the new memory contradicts
    the old one, as opposed to a merge of duplicates, the distilling of a raw
    message or an update kept and superseded. Read from the event's ``kind``
    (``models.SUPERSEDE_KINDS``); an older row without one is classified by
    the reason those others record."""
    if event.kind is not None:
        return event.kind == "contradiction"
    reason = event.reason or ""
    return not (
        reason.startswith("consolidated into")
        or reason.startswith("distilled with its context")
        or _is_update_supersede(event)
    )


def _within(created_at: str, since: str | None, until: str | None) -> bool:
    """Is an ISO ``created_at`` inside the [since, until] window?

    Bounds accept a plain date (YYYY-MM-DD) or a full ISO timestamp; a date-only
    ``until`` is inclusive of that whole day, which is what a human means by
    "up to the 22nd".
    """
    def _dt(value: str, *, end_of_day: bool) -> datetime | None:
        value = value.strip()
        if not value:
            return None
        try:
            if len(value) == 10:  # date only
                base = datetime.fromisoformat(value).replace(tzinfo=timezone.utc)
                return base + timedelta(days=1) if end_of_day else base
            dt = datetime.fromisoformat(value)
            return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
        except ValueError:
            return None

    try:
        created = datetime.fromisoformat(created_at)
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return True  # unparseable timestamps are never filtered out
    lo = _dt(since, end_of_day=False) if since else None
    hi = _dt(until, end_of_day=True) if until else None
    if lo is not None and created < lo:
        return False
    if hi is not None and created >= hi:
        return False
    return True


WHEN_KEY = "when"
WHEN_CHECKED_KEY = "when_checked"


def _when_within(
    memory: Memory, when_since: str | None, when_until: str | None
) -> bool:
    """Does this memory's occurrence time fall in the [when_since, when_until]
    window? A memory with no occurrence time never matches: "what is on this
    weekend" must not answer with everything that was merely saved then.
    """
    if not (when_since or when_until):
        return True
    return when_overlaps(
        (memory.metadata or {}).get(WHEN_KEY), when_since, when_until
    )


@dataclass(frozen=True)
class _Reads:
    """What a search reads (stage 2 of ``MemoryStore.search``), kept to by
    every candidate it gathers: the text ranking, the linked pool and the
    set call's. In SQL, as ``MemoryBackend.keyword_search`` reads: the scope
    searched (a run's memories are those said in it), the memories in use
    and those kept as history (every memory with ``include_invalid``), the
    tags and the entities asked for. Here, the date windows (``admits``)."""

    scope: Scope
    include_invalid: bool = False
    categories: list[str] | None = None
    entity_id: str | list[str] | None = None
    since: str | None = None
    until: str | None = None
    when_since: str | None = None
    when_until: str | None = None

    def admits(self, memory: Memory) -> bool:
        """Whether a memory was saved inside ``since``/``until`` and what it
        tells happens inside ``when_since``/``when_until``."""
        return ((not (self.since or self.until)
                 or _within(memory.created_at, self.since, self.until))
                and _when_within(memory, self.when_since, self.when_until))

    def entity_memories(
        self, backend: MemoryBackend, entity_id: str, limit: int
    ) -> list[Memory]:
        """The newest ``limit`` memories of an entity that the search reads,
        its filters applied in SQL before ``limit`` counts."""
        return [memory for memory in backend.entity_memories(
                    entity_id, limit=limit, include_invalid=self.include_invalid,
                    scope=self.scope, history=True, categories=self.categories,
                    mentioning=self.entity_id)
                if self.admits(memory)]


@dataclass
class _SearchPlan:
    """One search as it goes through the stages of ``MemoryStore.search``:
    what it reads and is about, and what each stage leaves the next."""

    reads: _Reads
    #: stage 1: the hubs the question is about, whether that is the owner of
    #: a question in the first person, and the question as the order and
    #: the judge read it ("it" for the one seed's names)
    seeds: list[str] = field(default_factory=list)
    first_person: bool = False
    question: str = ""
    #: whether the decision provider judges the search (stages 4 to 6)
    judges: bool = False
    #: stages 2 and 3 with seeds: how strongly the links reach each entity,
    #: those reached by a step up, each memory's entities as read, and the
    #: question's vector as the property comparison reads it
    act: dict[str, float] = field(default_factory=dict)
    above: set[str] = field(default_factory=set)
    entities: dict[str, list[Entity]] = field(default_factory=dict)
    asked: np.ndarray | None = None
    #: stages 5 and 6: each memory judged (member of the set, score), in the
    #: order judged
    judged: dict[str, tuple[bool, float]] = field(default_factory=dict)


def _cut(vector: list[float], keep: int | None) -> list[float]:
    """The first ``keep`` numbers of a vector, scaled back to length 1."""
    if not keep or keep >= len(vector):
        return vector
    short = np.asarray(vector[:keep], dtype=np.float32)
    return (short / (float(np.linalg.norm(short)) or 1.0)).tolist()


def _similarity(asked: np.ndarray, vector: np.ndarray | None) -> float:
    """Cosine of a memory's vector and the question's (already of length one),
    the vector cut to the question's length; 0 for a missing or shorter one
    and for an opposite one."""
    if vector is None or vector.shape[0] < asked.shape[0]:
        return 0.0
    vector = vector[: asked.shape[0]]  # an ordinary vector is cut like the question
    return max(float(vector @ asked) / (float(np.linalg.norm(vector)) or 1.0), 0.0)


def _centre(vectors: list[np.ndarray]) -> np.ndarray | None:
    """The mean direction of vectors of one length, of length one; None for
    none, or for vectors of different lengths."""
    if not vectors or len({v.shape for v in vectors}) != 1:
        return None
    centre = np.mean([v / (float(np.linalg.norm(v)) or 1.0) for v in vectors], axis=0)
    return centre / (float(np.linalg.norm(centre)) or 1.0)


def _across_runs(scope: Scope) -> Scope:
    """A save's scope as the lookups across one person's saves read it: the
    whole user (with the agent), not one run. Reconcile's candidates and the
    tag vocabulary offered to extraction use it; topic canonicalization and
    entity lookup (``entities.resolve_mentions``) read the whole user too.
    Reconcile acts on the memory it matched whatever its run
    (``reconcile.reconcile_candidate``); the save's run decides only where a
    new memory is stored, and a restatement of another run's memory is
    recorded on it as evidence of the save, whose episodes keep it findable
    by a search of the save's run."""
    if scope.user_id is None:
        return scope
    return Scope(user_id=scope.user_id, agent_id=scope.agent_id)


def _rests_on(
    candidate: CandidateFact, line_episodes: list[list[str]] | None, episode_ids: list[str]
) -> list[str]:
    """The episodes a candidate fact rests on: those of the transcript lines it
    names (``CandidateFact.sources``). A fact that names no line, or a line the
    transcript does not have, rests on every episode of the save, as a fact
    did before facts named their lines: a number out of range says the model
    lost count, so none of its numbers is trusted."""
    lines = candidate.sources
    if not lines or not line_episodes or any(not 1 <= n <= len(line_episodes) for n in lines):
        return episode_ids
    return list(dict.fromkeys(e for n in lines for e in line_episodes[n - 1])) or episode_ids


def _text_hash(text: str) -> str:
    """Identifies a property vector's masked text, to tell when it changed."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class MemoryStore:
    def __init__(
        self,
        config: Config | None = None,
        *,
        backend: MemoryBackend | None = None,
        llm: LLM | None = None,
        decider: Decider | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.config = config or Config.load()
        self.backend = backend or LocalBackend(self.config.db_path, ann=self.config.ann)
        self.llm = llm or build_llm(self.config.llm)
        # Typed judgements (entity identity). Defaults to the text model,
        # so a store that configures nothing behaves exactly as before.
        self.decider = decider or build_decider(self.config.decision, self.llm)
        self.embedder = embedder or build_embedder(self.config.embedding)
        # One lock per pass and namespace. The scheduler and "run now" can
        # start the same pass at once, and two runs each read the queue, add
        # to their own copy and write it back, so one run's additions were
        # lost and every group was sent to the model twice.
        self._pass_locks: dict[tuple[str, str | None], threading.RLock] = {}
        self._pass_locks_guard = threading.Lock()
        self.backend.names_changed = self._names_changed
        try:
            self._settle_owner_pairs()
        except Exception as exc:  # an upgrade step must never stop a store opening
            log.warning("pairs of the owner without a name were not settled: %s", exc)

    # ------------------------------------------------------------------
    # write path
    # ------------------------------------------------------------------
    def add(
        self,
        content: str | list[dict[str, str]],
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        infer: bool = True,
        memory_type: MemoryType = "semantic",
        importance: float = 0.5,
        categories: list[str] | None = None,
        created_at: str | None = None,
        memory_metadata: dict[str, Any] | None = None,
        now: datetime | None = None,
    ) -> AddResult:
        """Record raw content and derive memories from it.

        ``infer=True`` runs extraction + reconciliation (needs an LLM;
        degrades to verbatim mode without one). ``infer=False`` stores the
        content directly as a single memory - the "just save this fact" path.

        ``metadata`` goes to the episodes. ``memory_metadata`` is merged into
        every memory the save produces (a key Memry sets itself, such as
        "when", is kept). ``created_at`` (ISO 8601) is the time of the save:
        the episodes' and new memories' ``created_at``, ``updated_at`` and
        ``valid_from`` (a merged text of a MORE included), the ``invalid_at``
        of one it supersedes, and the time of the events it records (the
        NONE event of a memory it restates). A memory it supersedes keeps a
        later ``updated_at`` it has (``repair_updated_at`` reads the same).
        ``now`` is the reference date extraction resolves "yesterday"
        against, and the when-confirmation reads as the day of writing,
        instead of the clock. All three are for replaying dated
        conversations, as the benchmarks do.

        No ``user_id`` (None or "") saves to the default namespace
        (``_namespace``), as the servers do.
        """
        user_id = self._namespace(user_id)
        scope = Scope(user_id=user_id, agent_id=agent_id, run_id=run_id)
        messages = _as_messages(content)
        # One time for every message of the save: a save is its run and its
        # time, which is how the saves stating a memory are counted
        # (``reconcile.saves_of``).
        saved_at = created_at or utcnow()
        episodes = _said_episodes(messages, user_id=user_id, agent_id=agent_id,
                                  run_id=run_id, metadata=metadata, created_at=saved_at)
        if not episodes:
            return AddResult()
        if episodes:
            self.backend.add_episodes(episodes)
            self._embed_episodes(episodes)
        episode_ids = [e.id for e in episodes]

        candidates: list[CandidateFact]
        warnings: list[str] = []
        stated: list[str] = []
        if not infer:
            text = content if isinstance(content, str) else "\n".join(
                m.get("content", "") for m in messages
            )
            candidates = [
                CandidateFact(
                    content=text.strip(),
                    memory_type=memory_type,
                    importance=importance,
                    categories=clean_tags(categories),
                )
            ]
        elif self.llm.available:
            try:
                candidates = self._extract(
                    messages,
                    scope,
                    now=now,
                    context=_ingestion_context(metadata),
                    tag_hints=_client_tag_hints(metadata, categories),
                    stated=stated,
                )
            except Exception as exc:
                # Provider outage / exhausted credits must not lose the save:
                # degrade to verbatim, tell the caller, and flag the memories
                # so distillation can be re-run later (store.distill).
                candidates = self._pending_verbatim(messages)
                warnings.append(
                    f"extraction failed; stored verbatim instead (distill later): {exc}"
                )
        else:
            candidates = self._pending_verbatim(messages)

        _keep_context(candidates, _ingestion_context(metadata))
        _with_memory_metadata(candidates, memory_metadata)
        actions = self._apply_candidates(candidates, scope, episode_ids, created_at=created_at,
                                         messages=messages,
                                         line_episodes=[[e] for e in episode_ids])
        if infer:
            self._learn_from_save(scope, messages, stated, [[e] for e in episode_ids], actions)

        missing = self._coverage_gaps(messages, actions) if infer else []
        if missing:
            warnings.append(_coverage_warning(missing))
        return AddResult(episode_ids=episode_ids, actions=actions, warnings=warnings)

    def _coverage_gaps(
        self, messages: list[dict[str, str]], actions: list[AddAction]
    ) -> list[str]:
        """Post-write audit: extraction is lossy and non-deterministic, and a
        dropped constraint is invisible in a "success" response. One cheap LLM
        pass compares the input against what landed and names the gap. Run
        after a direct save and after distillation (the deferred save), with a
        text model and something written; best-effort, it never fails a save."""
        if not (self.llm.available and actions):
            return []
        stored = [a.content for a in actions if a.content]
        try:
            return verify_coverage(self.llm, messages, stored)
        except Exception:
            return []

    def _extract(
        self,
        messages: list[dict[str, str]],
        scope: Scope,
        *,
        now: datetime | None,
        context: str,
        tag_hints: list[str],
        stated: list[str] | None = None,
    ) -> list[CandidateFact]:
        """The facts extraction finds in a direct save's messages (``add``)
        or a pending group's (``_distill_pending_group``), asked the same way:
        offered the tags and entities their words may name and the owner as
        these messages speak of them (``owner_name``), each fact's time then
        checked against ``now``. A deferred save is extracted as it would have
        been saved directly. ``stated`` receives the user's name where the
        messages state it (``extraction.extract_facts``), asked only while the
        owner has no name (``_owner_unnamed``): for a named owner the prompt
        and schema are those without the question, so a conversation that
        gives a named owner another name goes unnoticed at save. Corrections
        are rare, and the question cost every call about 80 prompt tokens."""
        said = "\n".join(str(m.get("content") or "") for m in messages)
        candidates = extract_facts(
            self.llm,
            messages,
            now=now,
            vocabulary=self._tag_vocabulary(scope, text=said),
            context=context,
            tag_hints=tag_hints,
            owner=self.owner_name(scope.user_id, messages),
            entity_names=self._entity_vocabulary(scope, said),
            identity=stated if stated is not None and self._owner_unnamed(scope.user_id)
            else None,
        )
        self._confirm_candidate_whens(candidates, now=now)
        return candidates

    def add_deferred(
        self,
        content: str | list[dict[str, str]],
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        memory_type: MemoryType = "episodic",
        importance: float = 0.5,
        categories: list[str] | None = None,
        created_at: str | None = None,
        memory_metadata: dict[str, Any] | None = None,
        now: datetime | None = None,
    ) -> AddResult:
        """Durably save raw content for managed background enrichment.

        ``content`` is a text or a list of messages, as for ``add``, and its
        turns are kept as ``add`` keeps them: one episode per message that
        says something, with its role (a text is one, said by the user). One
        searchable pending memory holds the save: the text, or for messages
        one "Speaker: text" line each (``_speaker_lines``). Messages are also
        kept with the work marker, so the distillation reads them as a direct
        save's extraction does: the same numbered lines and speakers, each
        line resting on its own episode (``_pending_lines``).

        This path performs no provider calls. The episodes and searchable pending
        memory are committed before the caller receives the result; the pending
        metadata is the restart-safe work marker consumed by the server worker.
        ``created_at``, ``memory_metadata`` and ``now`` mean what they mean for
        ``add``: they apply to the pending memory and are kept with the work
        marker for the distillation that follows. The quiet period counts from
        when the save was queued, whatever ``created_at`` says. No
        ``user_id`` saves to the default namespace, as for ``add``.
        """
        user_id = self._namespace(user_id)
        if isinstance(content, str):
            content = content.strip()
        queued_at = utcnow()
        stamp = created_at or queued_at
        episodes = _said_episodes(_as_messages(content), user_id=user_id, agent_id=agent_id,
                                  run_id=run_id, metadata=metadata, created_at=stamp)
        if not episodes:
            return AddResult()
        text = content if isinstance(content, str) else _speaker_lines(content)
        pending_metadata = {**(memory_metadata or {}), **(metadata or {})}
        pending_metadata["pending_distillation"] = True
        job: dict[str, Any] = {"status": "pending", "attempts": 0, "queued_at": queued_at}
        if not isinstance(content, str):
            # what extraction reads of each message (extraction._transcript)
            job["messages"] = [
                {"role": m.get("role", "user"), "content": m["content"],
                 **({"name": m["name"]} if m.get("name") else {})}
                for m in content
                if _says_something(m)
            ]
        if created_at:
            job["created_at"] = created_at
        if memory_metadata:
            job["memory_metadata"] = dict(memory_metadata)
        if now is not None:
            job["now"] = now.isoformat()
        pending_metadata[_ENRICHMENT_KEY] = job
        memory = Memory(
            content=text,
            memory_type=memory_type,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            importance=importance,
            categories=clean_tags(categories),
            metadata=pending_metadata,
            source_episode_ids=[episode.id for episode in episodes],
            created_at=stamp,
            updated_at=stamp,
        )
        self.backend.add_episodes(episodes)
        self.backend.insert_memory(memory)
        self.backend.add_event(
            MemoryEvent(
                memory_id=memory.id,
                event="ADD",
                new_content=text,
                reason="durably queued for background enrichment",
            )
        )
        return AddResult(
            episode_ids=[episode.id for episode in episodes],
            actions=[
                AddAction(
                    event="ADD",
                    memory_id=memory.id,
                    content=memory.content,
                    reason="pending background enrichment",
                )
            ],
        )

    @staticmethod
    def _pending_verbatim(messages: list[dict[str, str]]) -> list[CandidateFact]:
        """Verbatim candidates flagged for later distillation."""
        candidates = verbatim_candidates(messages)
        for candidate in candidates:
            candidate.metadata = {"pending_distillation": True}
        return candidates

    def _canonicalize_obvious_topics(
        self, candidates: list[CandidateFact], scope: Scope
    ) -> None:
        """Write each candidate's tags as a save stores them (``_canonical_tags``)."""
        tags = self._canonical_tags([candidate.categories for candidate in candidates], scope)
        for candidate, canonical in zip(candidates, tags):
            candidate.categories = canonical

    def _canonical_tags(
        self, tag_lists: list[list[str]], scope: Scope, *, merge_stored: bool = True
    ) -> list[list[str]]:
        """Each list of tags as the store writes it, on a save and on an
        update alike: a name merged away written as its own survivor
        (``MemoryBackend.tag_filing``), and each tag in the obvious canonical
        form (singular, plural, spacing) it shares with the user's tags, so
        the column names the tag its memory is counted and filtered under.

        Each incoming name merged away is resolved through its own tombstone
        first ("tax" went into "levies", "taxes" into "duties": "taxes" is
        written "duties"), and obvious variants are grouped only among names
        still active (the user's active topics and the incoming names as
        resolved): a retired name is never grouped, so a group never sends a
        variant to another name's survivor or brings a retired name back. A
        save (``merge_stored``) runs the pass over the whole vocabulary and
        folds each stored variant into the form its group is written as
        (``_merge_topics``). An update rewrites its own tags only
        (``merge_stored`` False): it reads just the incoming tags' obvious
        variants and merges nothing, so no other memory is retagged; the
        vocabulary-wide pass is the next save's, or upkeep's."""
        incoming = {
            str(tag).strip().casefold()
            for tags in tag_lists
            for tag in tags
            if str(tag).strip()
        }
        if not incoming:
            return [list(tags) for tags in tag_lists]
        user = Scope(user_id=scope.user_id)
        prefixes = None if merge_stored else {obvious_variant_prefix(tag) for tag in incoming}
        vocabulary = self.backend.topic_names(user, prefixes=prefixes)
        # only names merged away need resolving here; every other tag is
        # resolved once, where the backend files the column
        survivor = self.backend.tag_filing(
            sorted(name for name in incoming if vocabulary.get(name) is False), user)
        resolved = {name: survivor.get(name, name) for name in incoming}
        if prefixes is not None:
            # a survivor's own obvious variants, read as the incoming ones' are
            more = {obvious_variant_prefix(name) for name in resolved.values()} - prefixes
            if more:
                vocabulary.update(self.backend.topic_names(user, prefixes=more))
        active = {name for name, alive in vocabulary.items() if alive}
        # a survivor that is a named thing's tag (merged away) stays as it is
        candidates = active | {name for name in resolved.values() if name not in vocabulary
                               or vocabulary[name]}
        wanted = set(resolved.values())
        groups = [
            group for group in obvious_canonical_merges(
                [{"category": name} for name in candidates])
            if merge_stored or wanted.intersection(group["variants"])
        ]
        replacements: dict[str, str] = {}
        for group in groups:
            target = group["canonical"]
            replacements.update({variant: target for variant in group["variants"]})
            stored = {name for name in group["variants"] if name in active and name != target}
            if merge_stored and stored:
                self._merge_topics(scope.user_id, stored, target, exact_user=True)
        rewritten_lists: list[list[str]] = []
        for tags in tag_lists:
            rewritten: list[str] = []
            seen: set[str] = set()
            for raw in tags:
                normalized = str(raw).strip().casefold()
                canonical = resolved.get(normalized, normalized)  # merged away: its survivor
                canonical = replacements.get(canonical, canonical)
                if canonical and canonical not in seen:
                    seen.add(canonical)
                    rewritten.append(canonical)
            rewritten_lists.append(rewritten)
        return rewritten_lists

    def _apply_candidates(
        self,
        candidates: list[CandidateFact],
        scope: Scope,
        episode_ids: list[str],
        *,
        exclude_ids: set[str] | None = None,
        created_at: str | None = None,
        messages: list[dict[str, str]] | None = None,
        line_episodes: list[list[str]] | None = None,
    ) -> list[AddAction]:
        """Reconcile candidates into the store (shared by add and distill).

        ``messages`` are what the candidates were extracted from: they say
        whether an owner without a name is "the user" (``owner_name``).

        ``line_episodes`` are the episodes of each transcript line, in order:
        a candidate rests on the episodes of the lines it names
        (``CandidateFact.sources``), and on all ``episode_ids`` when it names
        none or a line there is not (``_rests_on``).

        ``exclude_ids`` keeps memories out of the similarity set: distillation
        must not reconcile facts against the verbatim memory they came from,
        and candidates from ONE call must not reconcile against each other.
        The extractor already split the payload into discrete facts; without
        this, fact N finds facts 1..N-1 as "similar" and the reconciler chains
        them into a single memory via lossy UPDATE rewrites (the observed
        ADD followed by N UPDATEs on one id). Memories touched by this call
        are therefore accumulated into the exclusion set as we go.
        """
        self._canonicalize_obvious_topics(candidates, scope)
        excluded: set[str] = set(exclude_ids or ())
        actions: list[AddAction] = []
        for candidate in candidates:
            # the user's memories across runs, judged alike whatever their
            # run; the save's scope is where a new memory goes
            similar = hybrid_search(
                backend=self.backend,
                embedder=self.embedder,
                query=candidate.content,
                scope=_across_runs(scope),
                limit=self.config.retrieval.reconcile_similarity_limit,
                cfg=self.config.retrieval,
            )
            if excluded:
                similar = [r for r in similar if r.memory.id not in excluded]
            action = reconcile_candidate(
                candidate=candidate,
                scope=scope,
                similar=similar,
                backend=self.backend,
                embedder=self.embedder,
                llm=self.llm,
                episode_ids=_rests_on(candidate, line_episodes, episode_ids),
                decider=self.decider,
                retrieval_cfg=self.config.retrieval,
                supersede_cfg=self.config.supersede,
                prepare_update=lambda memory_id, final_content: (
                    self._reanalyze_edited_entities(memory_id, final_content, scope)
                ),
                created_at=created_at,
            )
            actions.append(action)
            if action.conflicts_with and action.memory_id:
                self._queue_conflict(scope.user_id, action)
            if action.event != "NONE" and action.memory_id:
                excluded.add(action.memory_id)
            # Entity mentions attach to the memory the action landed on
            # (conservative disambiguation; see intelligence/entities.py).
            if action.event not in ("NONE", "UPDATE") and action.memory_id and candidate.entities:
                # Pairs already waiting before this memory arrived; any the
                # memory mentions get compared again below, with its evidence.
                open_before = self._open_proposals_to_recheck(scope)
                resolved = resolve_mentions(
                    backend=self.backend,
                    llm=self.llm,
                    decider=self.decider,
                    scope=scope,
                    memory_id=action.memory_id,
                    memory_content=action.content or candidate.content,
                    surfaces=candidate.entities,
                    types=candidate.entity_types,
                    owner=self._owner_for(scope, candidate.entities, messages),
                )
                self._resolve_relations(
                    candidate.relations, resolved, scope, action.memory_id
                )
                self._recheck_proposals(
                    scope, open_before, {entity.id for entity in resolved.values()}
                )
        self._property_vectors_after_save(
            [a.memory_id for a in actions if a.event != "NONE" and a.memory_id])
        return actions

    def _property_vectors_after_save(self, memory_ids: list[str]) -> None:
        """Property vectors of memories just saved or edited, once their
        mentions are attached. A failure never fails the save: the weekly
        refresh computes what is missing."""
        if not memory_ids:
            return
        try:
            self.refresh_property_vectors(memory_ids=memory_ids)
        except Exception as exc:
            log.warning("property vectors not computed on save: %s", exc)

    def _names_changed(self, entity_ids: list[str]) -> None:
        """Property vectors after a merge, a rename, a new alias or a restore
        (``MemoryBackend.names_changed``): of the memories that read these
        entities' names as "it" (``_memories_reading``). Left to the weekly
        refresh, a merged name read as a name in them for up to a week. A
        failure never fails the change: the weekly refresh computes what is
        missing."""
        try:
            self.refresh_property_vectors(memory_ids=self._memories_reading(entity_ids))
        except Exception as exc:
            log.warning("property vectors not refreshed after a name changed: %s", exc)

    def _memories_reading(self, entity_ids: list[str]) -> list[str]:
        """The memories that read these entities' names as "it": their own,
        and their versions' and parts', which read the names of what they
        belong to as "it" too. A tag's memories mask no tag, so a tag's
        change concerns none."""
        named = {entity_id for entity_id in entity_ids
                 if (entity := self.backend.get_entity(entity_id)) is not None
                 and entity.entity_type != TOPIC_TYPE}
        named |= parts_of(self.backend, sorted(named))
        return sorted({memory.id for entity_id in sorted(named)
                       for memory in self.backend.entity_memories(entity_id, limit=1_000_000)})

    def _retire(self, entity_id: str, reason: str) -> bool:
        """Retire an entity (``MemoryBackend.retire_entity``) and refresh at
        once the property vectors of the memories that read its names as
        "it", which read them as names again. Once it is gone nothing links
        them to it, so they are read first."""
        try:
            reading = self._memories_reading([entity_id])
        except Exception as exc:
            log.warning("property vectors of a removed name not found: %s", exc)
            reading = []
        if not self.backend.retire_entity(entity_id, reason):
            return False
        try:
            self.refresh_property_vectors(memory_ids=reading)
        except Exception as exc:
            log.warning("property vectors not refreshed after a name was removed: %s", exc)
        return True

    def _open_proposals_to_recheck(self, scope: Scope) -> list[MergeProposal]:
        """Open proposals a save may compare again: none without a calibrated
        judge, whose answer is the only thing new evidence can change, or
        when the judge is too slow to ask inside a save."""
        if not (judges_pairs(self.decider) and self.decider.rejudges_on_new_evidence):
            return []
        return self.backend.list_proposals(scope, status="proposed", limit=1000)

    def _recheck_proposals(
        self, scope: Scope, open_before: list[MergeProposal], entity_ids: set[str]
    ) -> None:
        """Offer every pair a new memory just added evidence to for comparing.

        New evidence is the only thing that can make an unsure pair sure, so a
        pair is looked at when it gets some, not only on the weekly pass. A
        calibrated judge is asked only when the pair's smaller side has reached
        the next step of the funnel (``identity.PAIR_STEPS``); on other saves
        this costs two memory counts. A pair raised by this same save was
        judged moments ago and is left alone. A failure here must never fail
        the save.
        """
        touched = {
            proposal.id for proposal in open_before
            if proposal.entity_a in entity_ids or proposal.entity_b in entity_ids
        }
        if not touched:
            return
        try:
            outcome = resolve_open_proposals(
                backend=self.backend, decider=self.decider,
                scope=scope, proposal_ids=touched,
            )
        except Exception as exc:  # a provider hiccup must not fail a save
            log.warning("re-checking merge proposals after a save failed: %s", exc)
            return
        log.info(
            "new evidence on %d open merge proposal(s): merged %d, kept apart %d, "
            "still open %d", len(touched), outcome["confirmed"], outcome["rejected"],
            outcome["kept"],
        )

    def _resolve_relations(
        self,
        relations: list[dict[str, str]],
        resolved: dict[str, Any],
        scope: Scope,
        memory_id: str,
    ) -> None:
        """Turn (subject, predicate, object) surface triples into typed edges
        between the entities they linked to. Both endpoints must have resolved
        to real entities in this same memory, so an edge is always grounded."""
        for rel in relations:
            subj = resolved.get(str(rel.get("subject", "")).strip().lower())
            obj = resolved.get(str(rel.get("object", "")).strip().lower())
            predicate = str(rel.get("predicate", "")).strip().lower()
            if subj is None or obj is None or not predicate or subj.id == obj.id:
                continue
            self.backend.add_relation(
                Relation(
                    subject=subj.id,
                    predicate=predicate,
                    object=obj.id,
                    user_id=scope.user_id,
                    memory_id=memory_id,
                )
            )

    def _reanalyze_edited_entities(
        self, memory_id: str, content: str, scope: Scope
    ) -> dict[str, Any]:
        """Return the complete entity fields for edited memory text.

        With an LLM, extraction and identity resolution finish before the
        caller replaces the stored text and mentions: a name the memory
        already names an entity by keeps that entity, with nothing compared,
        and only a name new to the memory is resolved (``resolve_mentions``).
        Each mention is written with what decided it and the type extraction
        gave the name, as a save writes it. Without one, Memry can still
        retain or remove existing links by matching their known aliases;
        zero-key mode cannot discover a brand-new entity name.
        """
        if not self.llm.available:
            surfaces: list[str] = []
            mentions: list[EntityMention] = []
            for entity in self.backend.entities_of_memory(memory_id):
                aliases = sorted(
                    self.backend.entity_aliases(entity.id), key=len, reverse=True
                )
                surface = next(
                    (
                        alias for alias in aliases
                        if alias and re.search(
                            rf"(?<!\w){re.escape(alias)}(?!\w)",
                            content,
                            flags=re.IGNORECASE,
                        )
                    ),
                    None,
                )
                if surface:
                    surfaces.append(surface)
                    mentions.append(
                        EntityMention(
                            entity_id=entity.id,
                            memory_id=memory_id,
                            surface=surface,
                            decided={"reason": "the memory already names it"},
                        )
                    )
            return {"entities": surfaces, "mentions": mentions}
        try:
            candidates = extract_facts(
                self.llm, [{"role": "user", "content": content}],
                owner=self.owner_name(scope.user_id),
                entity_names=self._entity_vocabulary(scope, content),
            )
            surfaces = []
            types: dict[str, str] = {}
            seen: set[str] = set()
            for candidate in candidates:
                types.update(candidate.entity_types)
                for surface in candidate.entities:
                    normalized = surface.strip().lower()
                    if normalized and normalized not in seen:
                        seen.add(normalized)
                        surfaces.append(surface.strip())
            # written as resolved: with what decided each and its type
            mentions = []
            resolve_mentions(
                backend=self.backend,
                llm=self.llm,
                decider=self.decider,
                scope=scope,
                memory_id=memory_id,
                memory_content=content,
                surfaces=surfaces,
                types=types,
                attach=False,
                owner=self._owner_for(scope, surfaces),
                mentions=mentions,
            )
        except Exception as exc:
            raise ValueError(
                f"memory text was not changed because entity re-analysis failed: {exc}"
            ) from exc
        return {"entities": surfaces, "mentions": mentions}

    def _has_near_duplicate(
        self,
        embedding: list[float],
        scope: Scope,
        threshold: float,
    ) -> bool:
        matches = self.backend.vector_search(
            embedding,
            self.embedder.model_id,
            scope,
            limit=1,
        )
        return bool(matches and matches[0][1] >= threshold)

    def import_verbatim(
        self,
        rows: list[dict[str, Any]],
        *,
        user_id: str | None = None,
        dedup: bool = True,
        dedup_threshold: float = 0.97,
    ) -> dict[str, Any]:
        """Bulk verbatim import without extraction or reconciliation.

        Embeddings are fetched in batches. Duplicate rows in the same import and
        near-identical memories already in the target user scope are skipped by
        default without creating orphan episodes.
        """
        default_uid = self._namespace(user_id)
        prepared: list[dict[str, Any]] = []
        skipped = 0
        for row in rows:
            content = str(row.get("content") or "").strip()
            if not content:
                skipped += 1
                continue
            categories = clean_tags(row.get("categories"))
            memory_type = row.get("memory_type", "semantic")
            if memory_type not in MEMORY_TYPES:
                memory_type = "semantic"
            try:
                importance = float(row.get("importance", 0.5))
            except (TypeError, ValueError):
                importance = 0.5
            prepared.append(
                {
                    "content": content,
                    "user_id": str(row.get("user_id") or default_uid),
                    "agent_id": row.get("agent_id") or None,
                    "run_id": row.get("run_id") or None,
                    "categories": [str(c) for c in categories],
                    "memory_type": memory_type,
                    "importance": importance,
                }
            )
        if not prepared:
            return {
                "imported": 0,
                "skipped": skipped,
                "deduplicated": 0,
                "memory_ids": [],
            }

        vectors: list[list[float] | None] = [None] * len(prepared)
        if self.embedder.dimensions:
            chunk = 256  # stay well under provider batch limits
            for start in range(0, len(prepared), chunk):
                batch = prepared[start : start + chunk]
                try:
                    embedded = self.embedder.embed([p["content"] for p in batch])
                except Exception:
                    embedded = []  # import anyway; `memry reindex` can backfill
                for offset, vector in enumerate(embedded):
                    vectors[start + offset] = vector or None

        accepted: list[tuple[dict[str, Any], list[float] | None]] = []
        deduplicated = 0
        seen_content: set[tuple[str, str]] = set()
        for row, vector in zip(prepared, vectors):
            if dedup:
                key = (row["user_id"], row["content"].strip().lower())
                if key in seen_content:
                    deduplicated += 1
                    continue
                seen_content.add(key)
                if vector and self._has_near_duplicate(
                    vector, Scope(user_id=row["user_id"]), dedup_threshold
                ):
                    deduplicated += 1
                    continue
            accepted.append((row, vector))

        episodes = [
            Episode(
                content=row["content"],
                user_id=row["user_id"],
                agent_id=row["agent_id"],
                run_id=row["run_id"],
                metadata={"imported": True},
            )
            for row, _ in accepted
        ]
        if episodes:
            self.backend.add_episodes(episodes)
            # an imported row's episode says what its memory says: one vector
            self.backend.set_episode_vectors(
                {episode.id: vector for (_, vector), episode in zip(accepted, episodes) if vector},
                self.embedder.model_id)

        memory_ids: list[str] = []
        for (row, vector), episode in zip(accepted, episodes):
            memory = Memory(
                content=row["content"],
                memory_type=row["memory_type"],
                user_id=row["user_id"],
                agent_id=row["agent_id"],
                run_id=row["run_id"],
                importance=row["importance"],
                categories=row["categories"],
                source_episode_ids=[episode.id],
            )
            if vector:
                memory.embedding_model = self.embedder.model_id
            stored = self.backend.insert_memory(memory, vector)
            self.backend.add_event(
                MemoryEvent(
                    memory_id=stored.id,
                    event="ADD",
                    new_content=stored.content,
                    reason="imported",
                )
            )
            memory_ids.append(stored.id)
        return {
            "imported": len(memory_ids),
            "skipped": skipped,
            "deduplicated": deduplicated,
            "memory_ids": memory_ids,
        }
    def export_backup(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Export exact knowledge records for this scope as one versioned bundle."""
        return self.backend.export_backup(
            Scope(user_id=user_id, agent_id=agent_id, run_id=run_id)
        )

    def import_backup(
        self, backup: dict[str, Any], *, owner_prefix: str | None = None
    ) -> dict[str, Any]:
        """Restore a Memry backup exactly and transactionally, but for one
        thing: a row of it without a namespace (a backup of a store from
        before every write had one) is restored into the default namespace
        (``_namespace``). Restored into the store it came from before
        ``adopt_unscoped`` ran there, such a row conflicts with itself and
        the restore is refused: adopt first."""
        tables = backup.get("tables") if isinstance(backup, dict) else None
        if isinstance(tables, dict):
            tables = dict(tables)
            for table in _NAMESPACED_BACKUP_TABLES:
                rows = tables.get(table)
                if isinstance(rows, list):
                    tables[table] = [
                        {**row, "user_id": self._namespace(row.get("user_id"))}
                        if isinstance(row, dict) and "user_id" in row
                        and not row.get("user_id") else row
                        for row in rows]
            backup = {**backup, "tables": tables}
        return self.backend.import_backup(backup, owner_prefix=owner_prefix)

    def _namespace(self, user_id: str | None) -> str:
        """The namespace a write goes to: ``user_id``, else the default one
        (``config.default_user_id``). No memory lives without a namespace:
        a memory of none was read with every namespace's (no user means all
        users in a read) and walked as one of its own, and "" was a
        namespace apart from None that looked like none."""
        return user_id or self.config.default_user_id or "default"

    def adopt_unscoped(
        self, *, into: str | None = None, dry_run: bool = False,
    ) -> dict[str, Any]:
        """Give the rows a store holds without a namespace (from before every
        write had one) the namespace ``into`` (default: the default
        namespace), in one transaction: memories, their episodes, entities,
        relations, pairs, merges and retired names
        (``backend.adopt_unscoped``, which says how tags and same-named
        things of ``into`` take them). Their upkeep state (when each pass
        ran, the queues, who the owner is) goes with them where ``into`` has
        none of its own; where it has, its own is kept and theirs is
        dropped. ``dry_run`` reports what would move, fold and carry, and
        writes nothing. A second run finds nothing to do."""
        into = self._namespace(into)
        if not self.backend.supports_transactions:
            raise ValueError("this storage backend cannot keep the move together "
                             "(no transactions)")
        state = self._unscoped_state(into)
        if dry_run:
            report = self.backend.adopt_unscoped(into, dry_run=True)
        else:
            with self.backend.transaction():
                report = self.backend.adopt_unscoped(into)
                for source, target, value, kept in state:
                    if kept is None:
                        self.backend.set_meta(target, value)
                    self.backend.set_meta(source, "")  # carried, or dropped for theirs
        kept_ids = report.pop("kept_ids", [])
        report["state_carried"] = sorted(s[1] for s in state if s[3] is None)
        report["state_kept"] = sorted(s[1] for s in state if s[3] is not None)
        if kept_ids and not dry_run:
            self._names_changed(kept_ids)  # merged names: property vectors after
        return report

    def _unscoped_state(self, into: str) -> list[tuple[str, str, str, str | None]]:
        """The upkeep state kept for no namespace (None and "" share its
        keys) that has a value: (its key, the key of ``into``, its value,
        the value ``into`` has, None when it has none)."""
        keys = [key for key in self.backend.meta_items("upkeep:") if key.endswith(":")]
        keys += [_dedup_run_key(None), _consolidation_run_key(None)]
        out = []
        for key in keys:
            value = self.backend.get_meta(key)
            if not value:
                continue
            target = key + into
            out.append((key, target, value, self.backend.get_meta(target) or None))
        return out

    @staticmethod
    def _clear_enrichment_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value for key, value in metadata.items()
            if key not in ("pending_distillation", _ENRICHMENT_KEY)
        }

    def process_pending_enrichments(
        self,
        limit: int = _ENRICHMENT_BATCH_SIZE,
        *,
        quiet_seconds: float = 0.0,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Process one bounded batch of durable pending memories.

        Related saves in the same scope, with the same optional ``context``
        metadata and given the same day (``_said_day``), are distilled
        together after the group has been quiet. Raw
        records keep independent provenance and retry state, while extraction
        sees the complete thought instead of one client call at a time.
        """
        summary: dict[str, Any] = {
            "claimed": 0,
            "succeeded": 0,
            "failed": 0,
            "errors": [],
        }
        if not self.llm.available:
            summary["blocked"] = "no LLM configured"
            return summary
        now = now or datetime.now(timezone.utc)
        batch_limit = max(1, limit)
        # Look beyond the processing cap so a new item near the end of the same
        # context can reset that context's quiet period before older items run.
        pending = self.backend.list_pending_memories(
            max(batch_limit * 16, 128),
            due_before=now.isoformat(timespec="seconds"),
        )
        groups: dict[tuple[Any, ...], list[Memory]] = {}
        for memory in pending:
            # A group is extracted with one date, so saves said on different
            # days (a ``created_at`` given, or none) are never one group.
            key = (
                memory.user_id,
                memory.agent_id,
                memory.run_id,
                _ingestion_context(memory.metadata).casefold(),
                _said_day(memory),
            )
            groups.setdefault(key, []).append(memory)

        quiet = timedelta(seconds=max(0.0, quiet_seconds))
        cutoff = now - quiet
        bursts: list[list[Memory]] = []
        for related in groups.values():
            burst: list[Memory] = []
            for memory in sorted(related, key=_queued_at):
                if burst and _queued_at(memory) - _queued_at(burst[-1]) > quiet:
                    bursts.append(burst)
                    burst = []
                burst.append(memory)
            if burst:
                bursts.append(burst)

        for group in bursts:
            if summary["claimed"] >= batch_limit:
                break
            if max(_queued_at(memory) for memory in group) > cutoff:
                continue
            remaining = batch_limit - summary["claimed"]
            claimed: list[Memory] = []
            attempts_by_id: dict[str, int] = {}
            for memory in group[:remaining]:
                current = self.backend.get_memory(memory.id)
                if (
                    current is None
                    or current.invalid_at is not None
                    or not current.metadata.get("pending_distillation")
                ):
                    continue
                job = dict(current.metadata.get(_ENRICHMENT_KEY) or {})
                attempts = int(job.get("attempts") or 0) + 1
                job.update(
                    {
                        "status": "processing",
                        "attempts": attempts,
                        "last_started_at": utcnow(),
                    }
                )
                job.pop("next_attempt_at", None)
                processing_metadata = dict(current.metadata)
                processing_metadata[_ENRICHMENT_KEY] = job
                self.backend.update_memory(
                    current.id, metadata=processing_metadata, touch=False
                )
                current.metadata = processing_metadata
                claimed.append(current)
                attempts_by_id[current.id] = attempts

            if not claimed:
                continue
            summary["claimed"] += len(claimed)
            try:
                result = self._distill_pending_group(
                    [memory.id for memory in claimed]
                )
                if result is not None:
                    summary["succeeded"] += len(claimed)
            except Exception as exc:
                for current in claimed:
                    latest = self.backend.get_memory(current.id)
                    if (
                        latest is not None
                        and latest.invalid_at is None
                        and latest.metadata.get("pending_distillation")
                    ):
                        retry_job = dict(
                            latest.metadata.get(_ENRICHMENT_KEY) or {}
                        )
                        attempts = attempts_by_id[current.id]
                        delay = min(
                            2 ** min(attempts, 8),
                            _ENRICHMENT_MAX_BACKOFF_SECONDS,
                        )
                        retry_job.update(
                            {
                                "status": "retry",
                                "attempts": attempts,
                                "last_error": str(exc)[:500],
                                "next_attempt_at": (
                                    datetime.now(timezone.utc)
                                    + timedelta(seconds=delay)
                                ).isoformat(timespec="seconds"),
                            }
                        )
                        retry_metadata = dict(latest.metadata)
                        retry_metadata[_ENRICHMENT_KEY] = retry_job
                        self.backend.update_memory(
                            latest.id, metadata=retry_metadata, touch=False
                        )
                    summary["failed"] += 1
                    summary["errors"].append(
                        {"memory_id": current.id, "error": str(exc)[:500]}
                    )
        return summary

    def _distill_pending_group(
        self,
        memory_ids: list[str],
        *,
        owner_prefix: str | None = None,
    ) -> AddResult | None:
        memories = [self.backend.get_memory(memory_id) for memory_id in memory_ids]
        active = [
            memory
            for memory in memories
            if memory is not None
            and _owned(memory, owner_prefix)
            and memory.invalid_at is None
        ]
        if not active:
            return None
        first_scope = active[0].scope()
        if any(memory.scope() != first_scope for memory in active[1:]):
            raise ValueError("cannot distill memories from different scopes together")
        # a save queued before every write had a namespace: its facts get one
        first_scope = first_scope.model_copy(
            update={"user_id": self._namespace(first_scope.user_id)})
        if not self.llm.available:
            raise ValueError("no LLM configured; distillation needs one")

        # one transcript line per raw text, and per message of a deferred
        # save of messages, each with the episodes it rests on
        messages: list[dict[str, str]] = []
        line_episodes: list[list[str]] = []
        for memory in active:
            said, lines = _pending_lines(memory)
            messages.extend(said)
            line_episodes.extend(lines)
        contexts = list(
            dict.fromkeys(
                value
                for memory in active
                if (value := _ingestion_context(memory.metadata))
            )
        )
        context = " | ".join(contexts)[:200]
        tag_hints: list[str] = []
        for memory in active:
            for hint in _client_tag_hints(memory.metadata, memory.categories):
                if hint not in tag_hints:
                    tag_hints.append(hint)
                if len(tag_hints) == 3:
                    break
            if len(tag_hints) == 3:
                break
        episode_ids = list(
            dict.fromkeys(
                episode_id
                for memory in active
                for episode_id in memory.source_episode_ids
            )
        )
        # a deferred save made no provider call: its episodes are embedded now
        embedded = self.backend.episode_vectors_of(episode_ids, self.embedder.model_id)
        self._embed_episodes([episode for episode in self.backend.episodes_by_id(
            [e for e in episode_ids if e not in embedded]).values()])
        # What the saves asked of their memories (add_deferred): the latest
        # time and reference date given, and every memory_metadata merged.
        jobs = [memory.metadata.get(_ENRICHMENT_KEY) or {} for memory in active]
        created_at: str | None = None
        for job in jobs:
            if job.get("created_at"):  # compared as times: "...Z" is "+00:00"
                created_at = later_ts(created_at, job["created_at"])
        now = max((parse_ts(j["now"]) for j in jobs if j.get("now")), default=None)
        memory_metadata: dict[str, Any] = {}
        for job in jobs:
            memory_metadata.update(job.get("memory_metadata") or {})
        stated: list[str] = []
        candidates = self._extract(
            messages, first_scope, now=now, context=context, tag_hints=tag_hints,
            stated=stated)
        if not candidates:
            # a group extracted to nothing may still say who the user is
            self._learn_from_save(first_scope, messages, stated, line_episodes, [])
            for memory in active:
                metadata = self._clear_enrichment_metadata(memory.metadata)
                self.backend.update_memory(
                    memory.id, metadata=metadata, touch=False
                )
            return AddResult(
                episode_ids=episode_ids,
                warnings=["no facts extracted; memories kept verbatim"],
            )

        _keep_context(candidates, context)
        _with_memory_metadata(candidates, memory_metadata)
        actions = self._apply_candidates(
            candidates,
            first_scope,
            episode_ids,
            exclude_ids={memory.id for memory in active},
            created_at=created_at,
            messages=messages,
            line_episodes=line_episodes,
        )
        self._learn_from_save(first_scope, messages, stated, line_episodes, actions)
        landed = sum(1 for action in actions if action.event != "NONE")
        new_id = next(
            (
                action.memory_id
                for action in actions
                if action.event != "NONE" and action.memory_id
            ),
            None,
        )
        # The same audit as a direct save. Nobody waits on a deferred save, so
        # the gap is also noted where the raw text goes: its SUPERSEDE event.
        missing = self._coverage_gaps(messages, actions)
        warnings = []
        gap = ""
        if missing:
            warnings.append(_coverage_warning(missing))
            gap = "; not captured as facts: " + "; ".join(missing)
            log.warning("distillation of %s did not capture: %s",
                        ", ".join(m.id for m in active), "; ".join(missing))
        for memory in active:
            invalidated = self.backend.invalidate_memory(
                memory.id, superseded_by=new_id, at=created_at
            )
            if invalidated is not None:
                self.backend.update_memory(
                    memory.id,
                    metadata=self._clear_enrichment_metadata(
                        invalidated.metadata
                    ),
                    touch=False,
                )
            self.backend.add_event(
                MemoryEvent(
                    memory_id=memory.id,
                    event="SUPERSEDE",
                    old_content=memory.content,
                    reason=f"distilled with its context into {landed} fact(s){gap}",
                    kind="distillation",
                    **({"created_at": created_at} if created_at else {}),
                )
            )
        return AddResult(episode_ids=episode_ids, actions=actions, warnings=warnings)

    def distill(
        self, memory_id: str, *, owner_prefix: str | None = None
    ) -> AddResult | None:
        """Distill one active verbatim memory through the grouped write path."""
        return self._distill_pending_group([memory_id], owner_prefix=owner_prefix)

    # ------------------------------------------------------------------
    # read path
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        limit: int = 10,
        include_invalid: bool = False,
        categories: list[str] | None = None,
        entity_id: str | list[str] | None = None,
        since: str | None = None,
        until: str | None = None,
        when_since: str | None = None,
        when_until: str | None = None,
        relational: bool = True,
        evidence: bool = True,
    ) -> list[SearchResult]:
        """The memories that best answer ``query``, best first.

        Every search runs one pipeline, its stages in this order, each rule
        in one stage (docs/architecture.md, read path):

        1. the seeds (``_seeds``): the hubs the question names, the longest
           names among them, else the owner for a question in the first
           person; none with ``relational=False``;
        2. the candidates, as deep for every search: the text ranking
           (``_text_ranking``) and, with seeds, the linked pool
           (``_search_linked``), every filter (scope and run, history, tags,
           entity, date windows: ``_Reads``) applied as they are gathered,
           before anything is ordered or judged;
        3. the order: the linked order with seeds, the text ranking's
           without (``_search_linked``);
        4. the judged pool: the first ``decision.rerank_pool`` of the order,
           the keyword search's best match keeping a place in it on every
           search, judged or not (``_with_the_keyword_place``);
        5. the judge (``_judge_ranking``), in one wording
           (``_judged_relevance``), names read "it" only for a single seed,
           on every search where ``relevance_mode()`` is "jev" and none
           elsewhere;
        6. the set call and the set's members (``_set_pool``);
        7. the final order (``_final_order``: judged, the members first and
           then the judged score, a tie in the order judged, whose ties go by
           memory id), the limit (a set question returns every member found,
           up to ``SET_RESULT_CAP``), then the evidence: with ``evidence``
           each result carries the source turns it is the best ranked of the
           results to rest on, chosen within ``retrieval.evidence_tokens``
           (``evidence``)."""
        scope = Scope(user_id=user_id, agent_id=agent_id, run_id=run_id)
        if entity_id:
            entity_id = self._resolve_entity_filter(entity_id)
            if not entity_id:
                return []
        # No query text = browse by tag/date rather than rank by relevance.
        if not (query or "").strip():
            memories = self.get_all(
                user_id=user_id, agent_id=agent_id, run_id=run_id,
                include_invalid=include_invalid, limit=limit,
                categories=categories, entity_id=entity_id, since=since, until=until,
                when_since=when_since, when_until=when_until,
            )
            return [SearchResult(memory=m, score=0.0) for m in memories]
        reads = _Reads(scope, include_invalid, categories, entity_id, since, until,
                       when_since, when_until)
        # 1. the seeds, the question as it is read, and whether it is judged
        plan = self._plan(query, reads, relational)
        # 2. the candidates: the text ranking, as deep for every search
        query_vector = self._query_vector(query)  # the ranking and the evidence read it
        results = self._text_ranking(query, reads, limit, query_vector)
        # 2 to 4. the linked pool, the order and the judged pool
        ranked = self._search_linked(query, scope, results, include_invalid, plan=plan)
        # 5 and 6. the judge and the set call
        if plan.judges:
            ranked = self._judge_ranking(plan.question, ranked, scope, include_invalid, plan)
        # 7. the final order and the limit, then the evidence
        ranked = self._final_order(ranked, plan)
        members = sum(1 for r in ranked if r.signals.get("member"))
        found = ranked[:max(limit, min(members, SET_RESULT_CAP))]
        if evidence:
            by_id = {r.memory.id: r for r in found}
            for turn in self.evidence(query, found, user_id=user_id, agent_id=agent_id,
                                      run_id=run_id, query_vector=query_vector):
                by_id[turn.memory_ids[0]].evidence.append(turn)
        return found

    def _text_ranking(
        self, query: str, reads: _Reads, limit: int, query_vector: list[float] | None = None,
    ) -> list[SearchResult]:
        """Stage 2's text ranking: keyword and vector candidates fused
        (``retrieval.hybrid_search``), as deep for every search (eight per
        result asked for, at least 40 and at most 500), of what the search
        reads (``reads``). What was true until an update replaced it is read
        too, for questions about the past (``_final_order`` puts it after the
        current value). ``query_vector`` None embeds the query."""
        return [r for r in hybrid_search(
            backend=self.backend, embedder=self.embedder, query=query, scope=reads.scope,
            limit=min(max(limit * 8, 40), 500), cfg=self.config.retrieval,
            include_invalid=reads.include_invalid, categories=reads.categories,
            entity_id=reads.entity_id, history=True, query_vector=query_vector,
        ) if reads.admits(r.memory)]

    def _seeds(self, query: str, scope: Scope) -> tuple[list[str], bool]:
        """Stage 1 of a search: the entities its question is about, and
        whether that is the owner of a question in the first person.

        Only a hub counts (``_is_hub``): a stray phrase stored as an entity
        does not decide what a search is about. Of the hubs named, those
        whose name another's holds are dropped ("bildy v4", not also
        "bildy": a search from bildy reaches every version below it). The
        hubs are kept first, so a stray name holding a hub's ("bildy sync")
        does not hide it. A question naming no hub that speaks in the first
        person ("Where do I live?") is about the store's owner, when the
        owner is one."""
        hubs = [e for e in detect_query_entities(self.backend, scope, query) if self._is_hub(e)]
        seeds = longest_names(self.backend, hubs)
        if seeds or not speaks_in_first_person(query):
            return seeds, False
        owner = self.owner_entity(scope.user_id)
        if owner is not None and self._is_hub(owner.id):
            return [owner.id], True
        return [], False

    def _plan(self, query: str, reads: _Reads, relational: bool) -> _SearchPlan:
        """A search's seeds (stage 1, none with ``relational=False``), the
        question as the linked order and the judge read it, and whether the
        decision provider judges it.

        With one seed its names read "it" ("Where does it live?", and "it"
        for "I" when the seed is the owner); a question naming several hubs
        is read as written ("Did it like it?" says nothing), and so is one
        naming none.

        The decision provider judges a search if and only if
        ``relevance_mode()`` is "jev" ("auto" resolves from the provider and
        ``decision.rerank``); with "vector" no search is judged."""
        seeds, first_person = self._seeds(query, reads.scope) if relational else ([], False)
        question = query
        if len(seeds) == 1:
            question = mask_names(query, self.backend.entity_aliases(seeds[0]))
            if first_person:
                question = mask_first_person(question)
        judges = bool(self.decider.available) and self.relevance_mode() == "jev"
        return _SearchPlan(reads=reads, seeds=seeds, first_person=first_person,
                           question=question, judges=judges)

    def _current_first(self, results: list[SearchResult]) -> list[SearchResult]:
        """A memory kept as history (``models.HISTORY_KINDS``) comes right
        after the memory in use that replaced it, followed through a chain
        of updates, when that one is among the results: for one question the
        current value comes first. It is moved up, not the history down, so a
        question about the past keeps its answer as high as it ranked. The
        rest keep their order."""
        if all(r.memory.invalid_at is None for r in results):
            return results
        by_id = {r.memory.id: r for r in results}

        def current(memory: Memory) -> Memory:
            seen: set[str] = set()
            while memory.invalid_at is not None and memory.superseded_by and memory.id not in seen:
                seen.add(memory.id)
                found = by_id.get(memory.superseded_by)
                later = found.memory if found else self.backend.get_memory(memory.superseded_by)
                if later is None:
                    break
                memory = later
            return memory

        ordered: list[SearchResult] = []
        placed: set[str] = set()
        for result in results:
            if result.memory.id in placed:
                continue
            head = current(result.memory) if result.memory.invalid_at is not None else None
            if head is not None and head.id != result.memory.id and head.id in by_id \
                    and head.id not in placed:
                ordered.append(by_id[head.id])
                placed.add(head.id)
            ordered.append(result)
            placed.add(result.memory.id)
        return ordered

    def _query_vector(self, query: str) -> list[float]:
        """The query's vector, or [] with no embedder or while it is down (the
        ranking then reads the words alone)."""
        if not self.embedder.dimensions:
            return []
        try:
            return self.embedder.embed([query])[0] or []
        except Exception:
            return []

    def _embed_episodes(self, episodes: list[Episode]) -> None:
        """Store each episode's vector, as a memory's is stored. Best effort:
        without one an episode is still chosen as evidence by its words."""
        if not episodes or not self.embedder.dimensions:
            return
        try:
            vectors = self.embedder.embed([e.content for e in episodes])
        except Exception:
            return
        self.backend.set_episode_vectors(
            {e.id: v for e, v in zip(episodes, vectors) if v}, self.embedder.model_id)

    def evidence(
        self,
        query: str,
        results: list[SearchResult],
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        token_budget: int | None = None,
        query_vector: list[float] | None = None,
    ) -> list[EvidenceTurn]:
        """The source turns of the memories found that best match the query,
        in the order they were said: provenance, shown with the facts.

        Only the episodes the results in use or kept as history rest on are
        candidates (``Memory.source_episode_ids``): a memory an update
        replaced, which search returns after the one that replaced it, shows
        what was said while it held, as any memory does; one out of use
        otherwise (``include_invalid``) shows none. Each is taken once,
        credited to the best ranked memory resting on it; only those of the
        scope searched; none withheld or resting under a memory removed
        (``MemoryBackend.evidence_episodes``); none whose words a memory
        resting on it already says whole (a verbatim save). They are taken by
        their similarity to the query (the full-text match breaks a tie), each
        while it fits the budget (``token_budget``, default
        ``retrieval.evidence_tokens``; ``build_context`` counts a turn as it
        renders it)."""
        budget = self.config.retrieval.evidence_tokens if token_budget is None else token_budget
        if budget <= 0 or not (query or "").strip():
            return []
        history = self.backend.history_ids(
            [r.memory.id for r in results if r.memory.invalid_at is not None])
        resting: dict[str, list[str]] = {}
        texts: dict[str, str] = {}
        for result in results:
            memory = result.memory
            if memory.invalid_at is not None and memory.id not in history:
                continue
            texts[memory.id] = " ".join(memory.content.casefold().split())
            for episode_id in memory.source_episode_ids or []:
                resting.setdefault(episode_id, []).append(memory.id)
        if not resting:
            return []
        scope = Scope(user_id=user_id, agent_id=agent_id, run_id=run_id)
        episodes = [
            e for e in self.backend.evidence_episodes(list(resting))
            if all(getattr(scope, f) is None or getattr(scope, f) == getattr(e, f)
                   for f in ("user_id", "agent_id", "run_id"))
            and not any(" ".join(e.content.casefold().split()) in texts[mid]
                        for mid in resting[e.id])
        ]
        if not episodes:
            return []
        if query_vector is None:
            query_vector = self._query_vector(query)
        asked = np.asarray(query_vector, dtype=np.float32) if query_vector else None
        vectors = (self.backend.episode_vectors_of([e.id for e in episodes],
                                                   self.embedder.model_id)
                   if asked is not None else {})
        if asked is not None:
            asked /= float(np.linalg.norm(asked)) or 1.0
        words = self.backend.episode_keyword_scores(query, [e.id for e in episodes])
        turns = [
            EvidenceTurn(episode_id=e.id, content=e.content, speaker=e.speaker,
                         said_at=e.created_at, memory_ids=resting[e.id],
                         score=round(_similarity(asked, vectors.get(e.id))
                                     if asked is not None else 0.0, 6))
            for e in episodes
        ]
        order = {turn.episode_id: i for i, turn in enumerate(turns)}  # as said
        chosen: set[str] = set()
        used = 0
        for turn in sorted(turns, key=lambda t: (-t.score, -words.get(t.episode_id, 0.0),
                                                 order[t.episode_id])):
            cost = estimate_tokens(turn_line(turn)) + 1
            if used + cost > budget:
                continue
            chosen.add(turn.episode_id)
            used += cost
        return [turn for turn in turns if turn.episode_id in chosen]

    def _reranks(self) -> bool:
        """Whether the decision provider re-ranks. The setting decides where it
        is set; otherwise the provider's default stands. Either way a provider
        that was not measured to beat no re-ranking cannot be talked into it
        (``providers.decisions.MEASURED_RERANKERS``)."""
        cfg = self.config.decision
        wanted = cfg.rerank if cfg.rerank is not None else self.decider.reranks_by_default
        return bool(wanted and self.decider.may_rerank)

    def relevance_mode(self) -> str:
        """What judges whether a memory answers: ``retrieval.
        relational_relevance``, with "auto" read as "jev" where the decision
        provider re-ranks (``_reranks``: Jev by default, a text model measured
        to help when ``decision.rerank`` is on) and as "vector" elsewhere.
        "jev" has the provider judge every search; with "vector" none is
        judged (``_plan``)."""
        mode = self.config.retrieval.relational_relevance
        if mode != "auto":
            return mode
        return "jev" if self._reranks() else "vector"

    def _search_linked(
        self, query: str, scope: Scope, results: list[SearchResult], include_invalid: bool,
        *, plan: _SearchPlan,
    ) -> list[SearchResult]:
        """Stages 2 to 4 of a search, after its text ranking (``results``,
        what the search reads of it): the linked pool and the order, then the
        judged pool.

        With seeds (``plan.seeds``), the candidates are the text ranking's
        and, for every entity the links reach, however weakly, the
        ``FAMILY_TOP`` of its memories that the search reads (``_Reads``:
        its scope and run, history, tags, entity and date windows, applied
        before they are chosen) that best state the property asked. The
        links are followed directed and weighted, ``relational_depth`` deep
        (``graph_retrieval.activation_paths``). No link reaches a tag
        (``links_of``), so a tag's memories are never such candidates, as a
        tag is never a seed. Each candidate is ordered by how well it states
        the property asked (similarity of the question and the memory with
        the names of the entities the links reach replaced by "it"; a
        question naming several hubs is compared as written, with each
        memory's names kept) to the power ``relational_sharpness``, times how
        strongly it is about the entity the query names (``aboutness``), a
        tie by memory id. Without seeds the order is the text ranking's.

        The judged pool is the first ``decision.rerank_pool`` of the order,
        where the keyword search's best match keeps a place on every search,
        judged or not (``_with_the_keyword_place``)."""
        ranked = self._linked_order(results, plan) if plan.seeds else results
        return self._with_the_keyword_place(ranked, results)

    def _linked_order(self, results: list[SearchResult], plan: _SearchPlan) -> list[SearchResult]:
        """The linked pool and the linked order of a search with seeds
        (``_search_linked``); what the judge needs of them is left in
        ``plan``."""
        cfg = self.config.retrieval
        seeds = plan.seeds
        act, above = activation_paths(self.backend, seeds, depth=cfg.relational_depth)
        # With several hubs named, "it" could stand for any of them: masked,
        # "Why do Ada and Kai find Mira inspiring?" reads "Why do it and it
        # find it inspiring?", which says nothing about which of their
        # memories answers. So the names stay, as the judge reads them
        # (``_plan``), and every memory is compared by its ordinary vector,
        # names kept: those naming more of the things asked about come first.
        several = len(seeds) > 1
        asked = self._asked_vector(plan.question)
        pool: dict[str, SearchResult] = {r.memory.id: r for r in results}
        for entity_id in act:
            # what the search reads of an entity's memories is kept to in
            # SQL before the newest FAMILY_SCAN are taken
            members = plan.reads.entity_memories(self.backend, entity_id, FAMILY_SCAN)
            vectors = (self.backend.vectors_of([m.id for m in members], self.embedder.model_id)
                       if several else self._property_vectors([m.id for m in members]))
            # a tie keeps the order read: the newest first, then by memory id
            for memory in sorted(members,
                                 key=lambda m: -_similarity(asked, vectors.get(m.id)))[:FAMILY_TOP]:
                pool.setdefault(memory.id, SearchResult(memory=memory, score=0.0))
        scores = self._linked_scores(asked, list(pool), act, plan.entities, names_kept=several)
        scored = []
        for mid, result in pool.items():
            relevance, about = scores[mid]
            result.signals = {**result.signals, "property": round(relevance, 4),
                              "about": round(about, 3)}
            scored.append((relevance ** cfg.relational_sharpness * about, result.score, result))
        # a tie in both scores by memory id, not by the order the pool was filled in
        scored.sort(key=lambda item: (-item[0], -item[1], item[2].memory.id))
        plan.act, plan.above, plan.asked = act, above, asked
        return [result for _, _, result in scored]

    def _with_the_keyword_place(
        self, ranked: list[SearchResult], results: list[SearchResult]
    ) -> list[SearchResult]:
        """Stage 4, the judged pool, on every search: the keyword search's
        best match among ``results`` (the text ranking) keeps a place among
        the first ``decision.rerank_pool`` of ``ranked``, which the decision
        provider reads where it judges. Only the keyword search sees an
        identifier ("invoice
        2024-117") the question shares with a memory: the vectors, and so the
        property similarity, cannot tell 2024-117 from 2024-118, a memory
        linked to nothing counts as about something else (0.3), and in the
        text ranking newer memories matching more of the question's other
        words fill the first places. Judged, the answer ranks above the
        non-answers."""
        size = max(self.config.decision.rerank_pool, 2)
        worded = [r for r in results if "keyword" in r.signals]
        if not worded:
            return ranked
        best = max(worded, key=lambda r: r.signals["keyword"]).memory.id
        place = next(i for i, r in enumerate(ranked) if r.memory.id == best)
        if place < size:
            return ranked
        ranked = list(ranked)
        ranked.insert(size - 1, ranked.pop(place))
        return ranked

    def _asked_vector(self, question: str) -> np.ndarray:
        """The question's vector as the property comparison reads it: cut to
        ``retrieval.property_dimensions`` and of length one."""
        asked = np.asarray(self.embedder.embed([question])[0], dtype=np.float32)
        asked = asked[: self.config.retrieval.property_dimensions or len(asked)]
        asked /= float(np.linalg.norm(asked)) or 1.0
        return asked

    def _linked_scores(
        self, asked: np.ndarray, memory_ids: list[str], act: dict[str, float],
        entities: dict[str, list[Entity]], names_kept: bool = False,
    ) -> dict[str, tuple[float, float]]:
        """(property similarity to ``asked``, aboutness) of each memory, as the
        linked search scores it. Only the names the links account for are
        masked: a memory naming an entity they reach is compared by its
        property vector, any other by its ordinary one, names kept ("Lena Blum
        works on Project Ekmibo" would otherwise read "It works on it", as
        empty as "What do I know about it?"). With ``names_kept`` (a question
        naming several hubs, asked as written) every memory is compared by its
        ordinary vector. ``entities`` caches each memory's
        entities and is filled in, those not cached yet read at once."""
        missing = [mid for mid in dict.fromkeys(memory_ids) if mid not in entities]
        if missing:
            entities.update(self.backend.entities_of_memories(missing))
        reached = [] if names_kept else [
            mid for mid in memory_ids if any(e.id in act for e in entities[mid])]
        vectors = self._property_vectors(reached)
        vectors.update(self.backend.vectors_of([mid for mid in memory_ids if mid not in vectors],
                                               self.embedder.model_id))
        return {mid: (_similarity(asked, vectors.get(mid)),
                      aboutness([act.get(e.id) for e in entities[mid]]))
                for mid in memory_ids}

    def _judge_ranking(
        self, question: str, ranked: list[SearchResult], scope: Scope, include_invalid: bool,
        plan: _SearchPlan,
    ) -> list[SearchResult]:
        """Stages 5 and 6 of a search: the decision provider judges the
        judged pool (the first ``decision.rerank_pool`` of ``ranked``), and
        with them what kind of question it is, in one call:
        - about everything ("Show everything about X") or with one answer
          ("Where does Ada live?"): that is all;
        - several ("Which car is the cheapest?", "How much did I spend on
          groceries?"): one more call judges up to ``retrieval.set_pool``
          memories more that the search reads (``_set_pool``), and the
          members of the set are found over both calls' scores
          (``set_members``).
        The "calls" signal says how many calls were made (1 or 2), "pool" how
        many memories the second judged.

        With one seed, "it" stands for each memory's own entity (the one the
        links reach most strongly, a tie by entity id) in the question and
        the memories; with several or none, both are read as written. With
        seeds aboutness weighs each score, and a thing's answer yields to
        its version's own, never below an answer about something else
        judged the same (``aboutness``). Each memory judged, with whether it
        is a member and its score, is left in ``plan.judged`` for the final
        order (``_final_order``); ``ranked`` is returned with the memories
        the second call added after it."""
        size = max(self.config.decision.rerank_pool, 2)
        act, above, entities = plan.act, plan.above, plan.entities
        seeds = set(plan.seeds)
        homes: dict[str, set[str]] = {}
        aliases: dict[str, list[str]] = {}

        def ents(mid: str) -> list:
            if mid not in entities:
                entities[mid] = self.backend.entities_of_memory(mid)
            return entities[mid]

        def subject(mid: str) -> str | None:
            # the strongest reached, a tie by entity id: not by which of the
            # memory's mentions was written first
            linked = sorted(e.id for e in ents(mid) if e.id in act)
            return max(linked, key=act.get) if linked else None

        # With one entity named, "it" stands for it in the question and the
        # memories; with several ("Did Ilva like Olive Kitchen?") the names
        # stay, or the question would read "Did it like it?".
        masking = len(seeds) == 1

        def text_of(result: SearchResult) -> str:
            # "it" stands for the entity a memory's aboutness comes from and
            # the things that entity more likely than not belongs to ("The
            # first release of bildy" in a memory of bildy v1). Every other
            # name stays: it can be the answer ("uses Redis"), someone else
            # ("Kai Lund works on it", not "it works on it") or another
            # entity ("Bildy Bakery").
            who = subject(result.memory.id) if masking else None
            if who is None:
                return result.memory.content
            it = {who} | homes.get(who, set())
            for entity_id in it - aliases.keys():
                aliases[entity_id] = self.backend.entity_aliases(entity_id)
            return mask_names(
                result.memory.content, [n for entity_id in it for n in aliases[entity_id]],
                keep=[e.name for e in ents(result.memory.id) if e.id not in it])

        def judge(batch: list[SearchResult], meta: bool):
            if seeds:
                unread = [r.memory.id for r in batch if r.memory.id not in entities]
                if unread:  # read at once, not one memory at a time
                    entities.update(self.backend.entities_of_memories(unread))
                new = {subject(r.memory.id) for r in batch} - homes.keys() - {None}
                homes.update(homes_of(self.backend, sorted(new)))
            return self._judged_relevance(
                question, [(r.memory.id, text_of(r)) for r in batch], meta=meta)

        judged, specific, several = judge(ranked[:size], True)
        if not judged:
            return ranked
        found: dict[str, SearchResult] = {r.memory.id: r for r in ranked}
        extra: list[SearchResult] = []
        calls, pooled = 1, 0

        members: set[str] = set()
        if specific >= 0.5 and several >= SET_BAR:
            asked = plan.asked
            if asked is None:
                try:
                    asked = self._asked_vector(question)
                except Exception:  # embedding service down: the batch keeps its order
                    asked = None
            batch = self._set_pool(ranked, size, judged, scope, include_invalid,
                                   asked=asked, act=act, entities=entities,
                                   members=set_members(judged), names_kept=len(seeds) > 1,
                                   reads=plan.reads)
            if batch:
                for result in batch:
                    if result.memory.id not in found:
                        found[result.memory.id] = result
                        extra.append(result)
                got, _, _ = judge(batch, False)
                calls, pooled = 2, len(batch)
                judged.update(got)
            members = set_members(judged)
        # What is true of the thing a seed belongs to holds for the seed only
        # where nothing nearer says otherwise: an answer reached by a step up
        # counts as far as none of the seed's own memories answers, nor one of
        # a thing between them. bildy v4, a version of bildy v3 and of bildy,
        # takes v3's change over bildy's default, as v3 does. A thing is
        # between when the seed more likely than not belongs to it and it to
        # the thing the answer is about (``homes_of``); a sibling reached
        # through the thing is not. Both relevance and that override are per
        # property, so they count as far as the question asks for one ("Show
        # everything about it" does not).
        own = max((value for mid, value in judged.items()
                   if seeds and any(e.id in seeds for e in ents(mid))), default=0.0)
        best: dict[str | None, float] = defaultdict(float)  # the best answer about each entity
        if seeds:
            homes.update(homes_of(self.backend, sorted(seeds - homes.keys())))
            for mid, value in judged.items():
                best[subject(mid)] = max(best[subject(mid)], value)

        def overridden(thing: str) -> float:
            between = {x for seed in seeds for x in homes[seed]
                       if x != thing and thing in homes.get(x, ())}
            return max([own] + [best[x] for x in between])

        for mid, value in judged.items():
            result = found[mid]
            held = 1.0  # what the override leaves of the answer
            if seeds and subject(mid) in above:
                discount = overridden(subject(mid))
                held = (1.0 - discount) ** specific
                result.signals = {**result.signals, "overridden": round(discount, 4)}
            about = 1.0
            if seeds:
                about = result.signals.get("about") or aboutness([act.get(e.id) for e in ents(mid)])
                result.signals = {**result.signals, "about": round(about, 3)}
            result.signals = {**result.signals, "judged": round(value ** specific * held, 4),
                              "specific": round(specific, 4), "several": round(several, 4),
                              "calls": calls, "pool": pooled,
                              **({"member": True} if mid in members else {})}
            # The override comes after aboutness's floor, so what it leaves is
            # floored again: an answer from a thing the entity belongs to,
            # however weak its link and however much of it the entity's own
            # memories replace, counts at least as much as an answer about
            # something else judged the same (``aboutness``: a link never
            # ranks below no link).
            plan.judged[mid] = (mid in members, value ** specific * max(held * about, LOW))
        return ranked + extra

    def _final_order(self, ranked: list[SearchResult], plan: _SearchPlan) -> list[SearchResult]:
        """Stage 7 of a search, the final order, before the limit. A judged
        search puts the members of a set first, then every memory judged by
        its judged score (times aboutness with seeds), a tie in the order
        judged (the order's, whose ties went by memory id), and the rest
        after, as they were; a search not judged keeps its order. A memory
        kept as history then comes right after the memory in use that
        replaced it (``_current_first``)."""
        judged = plan.judged
        if judged:
            turn = {mid: i for i, mid in enumerate(judged)}
            first = sorted((r for r in ranked if r.memory.id in judged),
                           key=lambda r: (not judged[r.memory.id][0],
                                          -judged[r.memory.id][1], turn[r.memory.id]))
            ranked = first + [r for r in ranked if r.memory.id not in judged]
        return self._current_first(ranked)

    def _set_pool(
        self, ranked: list[SearchResult], size: int, judged: dict[str, float], scope: Scope,
        include_invalid: bool, *, asked: np.ndarray | None, act: dict[str, float],
        entities: dict[str, list[Entity]], members: set[str], names_kept: bool = False,
        reads: _Reads | None = None,
    ) -> list[SearchResult]:
        """What the second call of a question needing several memories judges:
        at most ``retrieval.set_pool`` memories not judged yet, of those the
        search reads (``reads``: its scope and run, history, tags, entity and
        date windows, as the first call's).

        The members of a set are the same kind of fact and filed under the same
        topics (tags). So the topics that at least ``SET_SHARED`` of the first
        ``size`` of the ranking carry are gathered, and every memory filed
        under one of them (its newest ``SET_SCAN``) scores, over those topics,
        how many of the first carry the topic over how many memories the topic
        has in the scope searched: a small topic most of them share counts
        most. The best are taken, a tie at the cut by the property ranking,
        over the tied candidates the newest first (then by memory id) as many
        as places are left and ``SET_TIE_MARGIN`` more (a tie of hundreds, one
        topic's share, is not scored whole). Measured on the dense world's set
        questions, those held 85 to 100% of each set within 100 candidates.

        Where the topics give fewer than the budget (the first share none: an
        untagged store, memories saved with ``infer=False`` or imported
        verbatim), the rest are the unjudged memories nearest the ``members``
        found in the first call (``_nearest_unjudged``, half by memory vector
        and half by property vector): members of a set are the same kind of
        fact, so their neighbours hold more of the rest of the set than the
        ranking past the first does.
        Only with no member to start from is the ranking past the first taken.
        Either way the batch is ordered as the linked search orders (the
        property similarity to ``asked``, to the power
        ``relational_sharpness``, times aboutness)."""
        budget = max(self.config.retrieval.set_pool, 0)
        if not budget:
            return []
        reads = reads or _Reads(scope, include_invalid)
        sharpness = self.config.retrieval.relational_sharpness

        def linked_order(memory_ids: list[str]) -> dict[str, float]:
            if asked is None or not memory_ids:
                return dict.fromkeys(memory_ids, 0.0)
            scores = self._linked_scores(asked, memory_ids, act, entities,
                                         names_kept=names_kept)
            return {mid: relevance ** sharpness * about
                    for mid, (relevance, about) in scores.items()}

        first = [r.memory.id for r in ranked[:size]]
        done = set(first) | set(judged)
        topics = self.backend.entities_of_memories(first, kind="topic")
        carried = Counter(topic.id for mid in first for topic in topics.get(mid, []))
        shared = [topic_id for topic_id, count in carried.items() if count >= SET_SHARED]
        # how many memories each has where the search looks, as ``filed`` is read
        sizes = self.backend.entity_memory_counts(
            shared, scope=reads.scope, history=True, categories=reads.categories,
            mentioning=reads.entity_id) if shared else {}
        walk: dict[str, float] = defaultdict(float)
        memories: dict[str, Memory] = {}
        for topic_id in shared:
            # what the search reads is kept to in SQL, before the newest SET_SCAN
            filed = reads.entity_memories(self.backend, topic_id, SET_SCAN)
            share = carried[topic_id] / max(sizes.get(topic_id, 0), len(filed), 1)
            for memory in filed:
                if memory.id in done:
                    continue
                memories[memory.id] = memory
                walk[memory.id] += share
        known = {r.memory.id: r for r in ranked}
        order: dict[str, float] = {}
        batch: list[SearchResult] = []
        if walk:
            # by the share, then the newest first, then by memory id: a key
            # that costs nothing (each sort keeps the order of the one before)
            best = sorted(walk)
            best.sort(key=lambda mid: memories[mid].updated_at or "", reverse=True)
            best.sort(key=lambda mid: -round(walk[mid], 9))
            edge = round(walk[best[min(budget, len(best)) - 1]], 9)
            # the candidates: all above the cut, and of those tied at it the
            # first by that key, as many as places are left and a margin
            # (SET_TIE_MARGIN), each scored once for the tie-break and the
            # batch's order alike; a tie of hundreds is not scored whole
            chosen = [mid for mid in best if round(walk[mid], 9) > edge]
            tied = [mid for mid in best if round(walk[mid], 9) == edge]
            tied = tied[: max(budget - len(chosen), 0) + SET_TIE_MARGIN]
            order = linked_order(chosen + tied)
            tied = sorted(tied, key=lambda mid: -order[mid])
            best = chosen + tied[: max(budget - len(chosen), 0)]
            batch = [known.get(mid) or SearchResult(memory=memories[mid], score=0.0)
                     for mid in best]
        if len(batch) < budget:
            taken = done | {r.memory.id for r in batch}
            rest = [known.get(m.id) or SearchResult(memory=m, score=0.0)
                    for m in self._nearest_unjudged(members, taken, scope, include_invalid,
                                                    budget - len(batch), reads=reads)]
            if not batch and not rest:  # no member to start from
                rest = [r for r in ranked[size:] if r.memory.id not in done][:budget]
            order.update(linked_order([r.memory.id for r in rest]))
            batch += rest
        return sorted(batch, key=lambda r: -order[r.memory.id])

    def _nearest_unjudged(
        self, members: set[str], taken: set[str], scope: Scope, include_invalid: bool,
        count: int, reads: _Reads | None = None,
    ) -> list[Memory]:
        """The ``count`` memories the search reads (``reads``, by default the
        scope searched) nearest the members of a set found so far, none of
        ``taken``: half by memory vector (the store's vector search from the
        centroid of the members' vectors), half by property vector (the rest
        of the ``SET_NEAREST`` nearest, ordered by their property similarity
        to the centroid of the members'). Members of
        a set are the same kind of fact ("It costs 21,000 euros"), so they sit
        closer to each other than to the question. A memory vector also
        follows the names in the text, and where the names weigh most the
        nearest to one car's price are that car's insurance and test drive;
        with the names read "it" the property vector keeps the kind of fact.
        Empty with no member, or none with a vector of the embedder in use."""
        if count <= 0 or not members:
            return []
        model = self.embedder.model_id
        centre = _centre(list(self.backend.vectors_of(sorted(members), model).values()))
        if centre is None:
            return []
        reads = reads or _Reads(scope, include_invalid)
        hits = self.backend.vector_search(centre.tolist(), model, reads.scope,
                                          limit=max(count, SET_NEAREST) + len(taken),
                                          include_invalid=reads.include_invalid,
                                          categories=reads.categories,
                                          entity_id=reads.entity_id, history=True)
        near = [memory for memory, _ in hits
                if memory.id not in taken and reads.admits(memory)]
        picked, rest = near[: count // 2], near[count // 2:]
        keep = self.config.retrieval.property_dimensions
        alike = _centre([v[: keep or len(v)]
                         for v in self._property_vectors(sorted(members)).values()])
        if alike is not None:
            vectors = self._property_vectors([m.id for m in rest])
            # a tie keeps the order of the memory vectors
            rest.sort(key=lambda m: -_similarity(alike, vectors.get(m.id)))
        return picked + rest[: count - len(picked)]

    def _judged_relevance(
        self, asked: str, memories: list[tuple[str, str]], meta: bool = True,
    ) -> tuple[dict[str, float], float, float]:
        """P(the memory answers the question) from the decision provider, for
        (memory id, text) pairs, in one call. With ``meta``, also what kind of
        question it is: P(it asks for one particular property rather than
        everything about its entity) and P(it needs several memories: a
        comparison, a list or a total). A memory the provider did not answer
        for is left out; an unanswered meta question counts as a property
        question with one answer."""
        if not self.decider.available or not memories:
            return {}, 1.0, 0.0
        questions: dict[str, Noul] = {
            f"m{i}": Noul(instructions="Someone who reads only this memory can answer the "
                                       f"question. Memory: {text}")
            for i, (_, text) in enumerate(memories)}
        if meta:
            questions["property"] = Noul(instructions="The question asks for one particular "
                                                      "property or fact of it, not for "
                                                      "everything about it.")
            questions["several"] = Noul(instructions="The question needs several memories to "
                                                     "be answered, such as a comparison, a "
                                                     "list or a total.")
        answers = self.decider.decide(f"QUESTION: {asked}", questions)
        judged = {mid: float(answers[f"m{i}"].value) for i, (mid, _) in enumerate(memories)
                  if answers[f"m{i}"].available}
        if not meta:
            return judged, 1.0, 0.0
        specific, several = answers["property"], answers["several"]
        return (judged, float(specific.value) if specific.available else 1.0,
                float(several.value) if several.available else 0.0)

    def _property_vectors(self, memory_ids: list[str]) -> dict[str, np.ndarray]:
        """Property vectors, and the ordinary vector for a memory saved before
        property vectors existed."""
        vectors = self.backend.property_vectors_of(memory_ids, self._property_label())
        missing = [mid for mid in memory_ids if mid not in vectors]
        if missing:
            vectors.update(self.backend.vectors_of(missing, self.embedder.model_id))
        return vectors

    def _property_label(self) -> str:
        """The embedding model and length a property vector was stored at: a
        vector of another model or length is not read, and is re-embedded."""
        return f"{self.embedder.model_id}#{self.config.retrieval.property_dimensions or 'all'}"

    def _masked_texts(
        self, contents: dict[str, str], entities: dict[str, list[str]]
    ) -> dict[str, str]:
        """Each memory's text with every alias of its entities, and of the
        things those belong to (``graph_retrieval.HOME_P``), read as "it".

        ``entities`` holds named things only: a tag is not masked, and the
        callers' lookups (``kind="named"``) leave tags out in SQL. "Spent 34
        euros on groceries at Lidl" filed under "groceries" says "Spent 34
        euros on groceries at it", since the tag is what the memory states,
        not what it is about."""
        homes = homes_of(self.backend, sorted({e for ids in entities.values() for e in ids}))
        aliases: dict[str, list[str]] = {}
        masked = {}
        for memory_id, content in contents.items():
            named = set(entities.get(memory_id, ()))
            for entity_id in list(named):
                named |= homes.get(entity_id, set())
            for entity_id in named - aliases.keys():
                aliases[entity_id] = self.backend.entity_aliases(entity_id)
            masked[memory_id] = mask_names(
                content, [n for entity_id in named for n in aliases[entity_id]])
        return masked

    def refresh_property_vectors(
        self, *, user_id: str | None = None, memory_ids: list[str] | None = None,
        exact_user: bool = False,
    ) -> int:
        """Embed the property vector of each valid memory whose masked text is
        new, changed (a merge, a rename, a new home) or was embedded by another
        model, in batches of 64. A memory whose masked text is its text gets no
        row: search reads its ordinary vector, which is the same. With
        ``memory_ids`` only those memories (a save, an edit, a merge or a
        rename); otherwise every memory of the namespace that names an entity
        or holds a row (the weekly upkeep, a backfill; ``exact_user`` as in
        ``get_all``). Returns how many it embedded."""
        if not self.embedder.dimensions:
            return 0
        entities: dict[str, list[str]] = defaultdict(list)
        if memory_ids is None:
            scope = Scope(user_id=user_id, exact_user=exact_user)
            for entity_id, memory_id in self.backend.entity_memory_links(scope, kind="named"):
                entities[memory_id].append(entity_id)
            listed = self.backend.list_memories(scope, limit=10_000_000)
            # A memory whose last entity was removed names nothing now, but
            # its row still reads the name as "it" until it is dropped here.
            held = self.backend.property_vector_hashes([m.id for m in listed])
            contents = {m.id: m.content for m in listed if m.id in entities or m.id in held}
        else:
            contents = {}
            for memory_id in memory_ids:
                memory = self.backend.get_memory(memory_id)
                if memory is not None and memory.invalid_at is None:
                    contents[memory_id] = memory.content
                    entities[memory_id] = [
                        e.id for e in self.backend.entities_of_memory(memory_id, kind="named")]
        masked = self._masked_texts(contents, entities)
        stored = self.backend.property_vector_hashes(list(masked))
        model = self._property_label()
        keep = self.config.retrieval.property_dimensions
        unmasked = [mid for mid, text in masked.items() if text == contents[mid] and mid in stored]
        if unmasked:
            self.backend.delete_property_vectors(unmasked)
        due = [(mid, text) for mid, text in masked.items()
               if text != contents[mid] and stored.get(mid) != (_text_hash(text), model)]
        embedded = 0
        for start in range(0, len(due), 64):
            batch = due[start:start + 64]
            vectors = self.embedder.embed([text for _, text in batch])
            rows = {mid: _cut(vector, keep) for (mid, _), vector in zip(batch, vectors) if vector}
            self.backend.set_property_vectors(
                rows, model, {mid: _text_hash(text) for mid, text in batch if mid in rows})
            embedded += len(rows)
        return embedded

    def _is_hub(self, entity_id: str) -> bool:
        """Whether an entity counts as one a query can name: a hub by the
        structure rules, so a stray phrase stored as an entity ("go",
        "upkeep") does not decide what a search is about."""
        entity = self.backend.get_entity(entity_id)
        if entity is None or entity.entity_type == TOPIC_TYPE:
            return False  # a tag's word in a question never makes it the subject
        return is_hub(entity.entity_type, self.backend.count_entity_memories(entity_id),
                      len(self.backend.relations_of([entity_id])),
                      (entity.metadata or {}).get("screen"))

    def reconstruct_context(
        self,
        query: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        token_budget: int = CONTEXT_TOKENS,
        limit: int = 20,
    ) -> ContextResult:
        """The memories found for ``query`` that fit ``token_budget``, rendered
        for a model (``intelligence.context``), after the descriptions of the
        entities the query names (``described_entities``, within
        ``context.description_budget``). The memories that fit take the budget
        but a share for their evidence (``retrieval.evidence_tokens``, at most
        half of what is left), and their source turns that best match the
        query fill that share (``evidence``)."""
        results = self.search(
            query, user_id=user_id, agent_id=agent_id, run_id=run_id, limit=limit,
            evidence=False,
        )
        entities = self.described_entities(
            query, user_id=user_id, agent_id=agent_id, run_id=run_id,
            token_budget=description_budget(token_budget),
        )
        entity_text = entities_text(entities)
        entity_memory_ids = [memory.id for entity in entities
                             for memory in self.backend.entity_memories(entity.id, limit=20)]
        remaining = max(0, token_budget - estimate_tokens(entity_text))
        share = min(max(self.config.retrieval.evidence_tokens, 0), remaining // 2)
        shown = fitting(results, remaining - share)
        turns = self.evidence(query, shown, user_id=user_id, agent_id=agent_id,
                              run_id=run_id, token_budget=share)
        memory_context = build_context(shown, token_budget=remaining, evidence=turns)
        parts = [part for part in (entity_text, memory_context.text) if part]
        combined = "\n\n".join(parts)
        memory_ids = list(dict.fromkeys([*entity_memory_ids, *memory_context.memory_ids]))
        return ContextResult(
            text=combined,
            memory_ids=memory_ids,
            token_estimate=estimate_tokens(combined) if combined else 0,
            episode_ids=memory_context.episode_ids,
        )

    def described_entities(
        self,
        query: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        token_budget: int = description_budget(CONTEXT_TOKENS),
    ) -> list[Entity]:
        """The entities ``query`` names (the first three
        ``detect_query_entities`` finds), each with its description, built
        or rebuilt where it is stale (``_refresh_entity_description``), as
        many as fit ``token_budget`` (``context.entities_fitting``): what
        ``reconstruct_context`` shows before the memories, and what the
        benchmark runner shows before its memory list. The default budget
        is ``reconstruct_context``'s at its default."""
        scope = Scope(user_id=user_id, agent_id=agent_id, run_id=run_id)
        described = [self._refresh_entity_description(entity_id)
                     for entity_id in detect_query_entities(self.backend, scope, query)[:3]]
        return entities_fitting([e for e in described if e is not None], token_budget)

    def _resolve_entity_filter(
        self, entity_id: str | list[str]
    ) -> str | list[str] | None:
        """Follow merge history for one entity filter, or several.

        A merged entity keeps its old id as a redirect, so a filter saved before
        a merge must still land on the surviving record.
        """
        if isinstance(entity_id, str):
            return self.backend.resolve_entity_id(entity_id)
        resolved = [self.backend.resolve_entity_id(e) for e in entity_id if e]
        return [e for e in resolved if e] or None

    def get(self, memory_id: str, *, owner_prefix: str | None = None) -> Memory | None:
        memory = self.backend.get_memory(memory_id)
        return memory if _owned(memory, owner_prefix) else None

    def knowledge_map(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        kind: str = "named",
    ) -> dict[str, Any]:
        """Content-free aggregate graph over every active memory in scope.

        A planet is a hub that came up at least twice; a person is one from the
        first mention. On a real store the hubs alone were still 1,424 planets,
        610 of them things typed as a product or project and seen exactly once,
        so the map asks for a second sighting and the list does not. A part
        that has a home is not a planet of its own: it rides along on its home
        as one of its ``parts``. Nothing is hidden for good, since all of this
        is recomputed from the memories each time.

        With ``kind`` "any" every tag with an active memory is a planet too
        (its topic entity): a tag is never a hub nor a home, so the rules
        above are for named things only.
        """
        data = self.backend.knowledge_map(
            Scope(user_id=user_id, agent_id=agent_id, run_id=run_id), kind=kind)
        structure = self.entity_structure(user_id=user_id)
        nodes = data.get("entities") or []
        by_id = {node.get("entity_id"): node for node in nodes}
        planets: list[dict[str, Any]] = []
        parts: dict[str, list[dict[str, Any]]] = {}
        def is_planet(entity_id: str, entity_type: str | None) -> bool:
            info = structure.get(entity_id)
            return bool(info and info["hub"] and (
                info["memories"] >= 2 or entity_type == "person"))

        for node in nodes:
            if node.get("entity_type") == TOPIC_TYPE:
                planets.append(node)
                continue
            info = structure.get(node.get("entity_id"))
            if not info or info.get("screened_out"):
                continue
            home = info["home"]
            if (
                home and home["id"] in by_id
                and node.get("entity_type") not in ANCHOR_TYPES
                and is_planet(home["id"], by_id[home["id"]].get("entity_type"))
            ):
                parts.setdefault(home["id"], []).append({
                    "entity_id": node["entity_id"], "label": node["label"],
                    "count": node["count"],
                })
                continue
            if is_planet(node["entity_id"], node.get("entity_type")):
                planets.append(node)
        for node in planets:
            mine = sorted(parts.get(node["entity_id"], []),
                          key=lambda part: (-part["count"], part["label"].lower()))
            node["parts"] = mine[:24]
            node["part_count"] = len(mine)
        shown = {node["key"] for node in planets}
        data["entity_names"] = sum(1 for node in nodes
                                   if node.get("entity_type") != TOPIC_TYPE)
        data["entities"] = planets
        data["entity_edges"] = [
            edge for edge in data.get("entity_edges") or []
            if edge["a"] in shown and edge["b"] in shown
        ]
        return data

    def categories(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Category histogram over active memories, largest count first.

        Each tag is a topic entity, counted by the active memories that
        mention it."""
        scope = Scope(user_id=user_id, agent_id=agent_id, run_id=run_id)
        indexed = self.backend.topic_mention_counts(scope)
        if indexed is not None:
            return indexed
        counter: dict[str, int] = {}
        for memory in self.get_all(
            user_id=user_id, agent_id=agent_id, run_id=run_id, limit=1_000_000
        ):
            for raw in memory.categories or []:
                category = str(raw).strip().lower()
                if category:
                    counter[category] = counter.get(category, 0) + 1
        return [
            {"category": c, "count": n}
            for c, n in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
        ]

    def _entity_vocabulary(self, scope: Scope, text: str) -> list[tuple[str, str | None]]:
        """Existing entities a text may be naming, offered to extraction as
        (name, type) so it writes their names as stored. On 11 texts that
        named a stored entity another way ("bildy.ai", "AWS", "Prof. Olsen"),
        extraction wrote the stored name 11 times with the offer and once
        without; on 7 texts about a new thing with a look-alike name ("Kestrel
        Capital", "Priya Sharma") it used a stored name 0 times either way.
        """
        if not text.strip():
            return []
        lookup = Scope(user_id=scope.user_id) if scope.user_id is not None else scope
        try:
            entities = self.backend.list_entities(lookup, limit=100_000)
        except Exception:
            return []
        named = NameIndex(entities).named_in(text)
        return [(e.name, e.entity_type) for e in named if not (e.metadata or {}).get("owner")]

    def _tag_vocabulary(
        self, scope: Scope, text: str = "", limit: int = VOCABULARY_LIMIT
    ) -> list[str]:
        """Direct tags offered back to extraction, so tagging stays convergent.

        Parents are deliberately excluded: offering ``health`` back would invite
        extraction to tag straight at the level that retrieval does worst with.

        Selection is by relevance first, then by frequency. Sending only the
        most-used tags works until a store passes the budget, at which point the
        long tail stops being offered - and an unoffered tag is precisely the one
        that gets a near-synonym coined for it next time its subject comes up.
        A conversation about liver results must see ``liver lab results`` even
        when it is the 300th most common tag.
        """
        try:
            counts = self.backend.topic_mention_counts(_across_runs(scope))
        except Exception:
            return []
        if not counts:
            return []
        names = [str(row["category"]) for row in counts if row.get("category")]
        if len(names) <= limit or not text.strip() or not self.embedder.dimensions:
            return names[:limit]

        # Half the budget goes to what this conversation is actually about, the
        # rest stays frequency-ranked so common tags are always on offer.
        relevant_budget = limit // 2
        try:
            vectors = self.embedder.embed([text[:4000], *names])
        except Exception:
            return names[:limit]
        query = np.asarray(vectors[0], dtype=float)
        matrix = np.asarray(vectors[1:], dtype=float)
        norms = np.linalg.norm(matrix, axis=1)
        query_norm = np.linalg.norm(query) or 1.0
        similarity = (matrix @ query) / (np.where(norms == 0, 1.0, norms) * query_norm)
        nearest = [names[i] for i in np.argsort(-similarity)[:relevant_budget]]
        chosen = list(dict.fromkeys(nearest))
        for name in names:  # top up with the most-used, skipping duplicates
            if len(chosen) >= limit:
                break
            if name not in chosen:
                chosen.append(name)
        return chosen

    def get_all(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        include_invalid: bool = False,
        limit: int = 100,
        offset: int = 0,
        categories: list[str] | None = None,
        entity_id: str | list[str] | None = None,
        since: str | None = None,
        until: str | None = None,
        when_since: str | None = None,
        when_until: str | None = None,
        exact_user: bool = False,
    ) -> list[Memory]:
        """The memories in scope, newest change first. ``exact_user``: no
        ``user_id`` means the memories without a user, not every user's
        (``Scope.exact_user``)."""
        scope = Scope(user_id=user_id, agent_id=agent_id, run_id=run_id,
                      exact_user=exact_user)
        if entity_id:
            entity_id = self._resolve_entity_filter(entity_id)
            if not entity_id:
                return []
        if not (since or until or when_since or when_until):
            return self.backend.list_memories(
                scope, include_invalid=include_invalid, limit=limit, offset=offset,
                categories=categories, entity_id=entity_id,
            )
        # Date-windowed browse: the filter is backend-agnostic (applied here), so
        # pull a broad page ordered by the backend, filter, then paginate.
        rows = self.backend.list_memories(
            scope, include_invalid=include_invalid, limit=1_000_000, offset=0,
            categories=categories, entity_id=entity_id,
        )
        if since or until:
            rows = [m for m in rows if _within(m.created_at, since, until)]
        if when_since or when_until:
            rows = [m for m in rows if _when_within(m, when_since, when_until)]
        return rows[offset : offset + limit]

    def history(
        self, memory_id: str, *, owner_prefix: str | None = None
    ) -> list[MemoryEvent]:
        # Admin path is untouched: events outlive their memory row (hard delete
        # keeps the audit trail), so only look the row up when confining.
        if owner_prefix is not None and not _owned(
            self.backend.get_memory(memory_id), owner_prefix
        ):
            return []
        return self.backend.history(memory_id)

    def episodes(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        limit: int = 100,
    ) -> list[Episode]:
        scope = Scope(user_id=user_id, agent_id=agent_id, run_id=run_id)
        return self.backend.list_episodes(scope, limit=limit)

    # ------------------------------------------------------------------
    # mutation
    # ------------------------------------------------------------------
    def update(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        importance: float | None = None,
        categories: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        owner_prefix: str | None = None,
    ) -> Memory | None:
        old = self.backend.get_memory(memory_id)
        if not _owned(old, owner_prefix):
            return None
        if categories is not None:
            # written as a save writes them (the column names the tags the
            # memory is counted and filtered under), without retagging any
            # other memory: the vocabulary-wide merge is the save's and upkeep's
            [categories] = self._canonical_tags(
                [clean_tags(categories)], old.scope(), merge_stored=False)
        entity_update: dict[str, Any] = {}
        if content is not None and content != old.content:
            entity_update = self._reanalyze_edited_entities(
                memory_id, content, old.scope()
            )
        embedding = None
        embedding_model = None
        if content is not None and self.embedder.dimensions:
            try:
                embedding = self.embedder.embed([content])[0]
                embedding_model = self.embedder.model_id
            except Exception:
                embedding = None
        updated = self.backend.update_memory(
            memory_id,
            content=content,
            embedding=embedding,
            embedding_model=embedding_model,
            importance=importance,
            categories=categories,
            metadata=metadata,
            **entity_update,
        )
        if updated and content is not None and content != old.content:
            self.backend.add_event(
                MemoryEvent(
                    memory_id=memory_id,
                    event="UPDATE",
                    old_content=old.content,
                    new_content=content,
                    reason="manual update",
                    actor="user",
                )
            )
            self._property_vectors_after_save([memory_id])
        return updated

    def delete(
        self, memory_id: str, *, hard: bool = False, owner_prefix: str | None = None
    ) -> bool:
        """Forget a memory (with ``hard``, delete it for good). A memory kept
        as history (``models.HISTORY_KINDS``), which search still reads, is
        forgotten as one in use is: search reads it no more, and it is listed
        as forgotten, where it can be brought back or purged."""
        memory = self.backend.get_memory(memory_id)
        if not _owned(memory, owner_prefix):
            return False
        if hard:
            ok = self._delete_for_good(memory_id)
        elif memory.invalid_at is None:
            ok = self.backend.invalidate_memory(memory_id) is not None
        else:
            ok = self.backend.forget_history(memory_id) is not None
        if ok:
            self.backend.add_event(
                MemoryEvent(
                    memory_id=memory_id,
                    event="DELETE",
                    old_content=memory.content,
                    reason="hard delete" if hard else "manual delete (invalidated)",
                    actor="user",
                )
            )
        return ok

    def _delete_for_good(self, memory_id: str) -> bool:
        """Delete a memory for good (``MemoryBackend.delete_memory``). The
        memories it had replaced (consolidated or distilled into it, or
        contradicted or updated by it) have nothing standing in for them any
        more: no pointer to it is left, and each is listed under Forgotten,
        where it can be brought back, with why it is there."""
        originals = self.backend.replaced_by(memory_id)
        if not self.backend.delete_memory(memory_id):
            return False
        for original in originals:
            self.backend.add_event(MemoryEvent(
                memory_id=original.id, event="DELETE", old_content=original.content,
                actor="system",
                reason=f"The memory that had replaced it ({memory_id}) was deleted for good.",
            ))
        return True

    def forgotten(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Memories that were removed, with why and by whom.

        Removed, not replaced: a memory that was superseded (reconciled away,
        consolidated, distilled) has ``superseded_by`` pointing at whatever took
        its place and is part of that memory's history, not something the user
        threw out. Only records with nothing standing in for them belong here,
        which includes those whose replacement was deleted for good
        (``_delete_for_good``).
        """
        scope = Scope(user_id=user_id, agent_id=agent_id, run_id=run_id)
        out: list[dict[str, Any]] = []
        for memory in self.backend.list_memories(
            scope, include_invalid=True, limit=1_000_000
        ):
            if memory.invalid_at is None or memory.superseded_by:
                continue
            # Whatever ended it: a delete (yours, or the retired forgetting sweep), or
            # a distillation that put nothing in its place. Looking for DELETE
            # alone is what left "forgotten by system" with no explanation.
            removal = next(
                (
                    event
                    for event in reversed(self.backend.history(memory.id))
                    if event.event in ("DELETE", "SUPERSEDE")
                ),
                None,
            )
            out.append({
                "memory": memory,
                "forgotten_at": memory.invalid_at,
                "actor": removal.actor if removal else "system",
                "reason": removal.reason if removal else None,
                "trigger": _forgetting_trigger(removal),
            })
            if len(out) >= limit:
                break
        out.sort(key=lambda row: row["forgotten_at"] or "", reverse=True)
        return out

    def unforget(self, memory_id: str, *, owner_prefix: str | None = None) -> bool:
        """Bring a forgotten memory back into active use.

        Only for memories in the forgotten list: restoring one that was
        superseded would resurrect a duplicate right next to its replacement,
        so those stay part of their successor's history.
        """
        memory = self.backend.get_memory(memory_id)
        if not _owned(memory, owner_prefix):
            return False
        if memory.invalid_at is None:
            return False  # nothing to undo
        if memory.superseded_by:
            raise ValueError(
                "this memory was replaced by another; if that was a mistake, "
                "undo the replacement under Archive"
            )
        restored = self.backend.revalidate_memory(memory_id)
        if restored is None:
            return False
        self.backend.add_event(
            MemoryEvent(
                memory_id=memory_id,
                event="ADD",
                new_content=memory.content,
                reason="unforgotten by the user",
                actor="user",
            )
        )
        return True

    # -- contradictions -----------------------------------------------------
    # A contradiction that was not allowed to replace anything waits under
    # Upkeep; one that was allowed to is listed under Archive, where it can be
    # undone. Between them no replacement is both silent and final.
    def _queue_conflict(self, user_id: str | None, action: AddAction) -> None:
        pending = self._upkeep_get("conflict:pending", user_id, [])
        if any(entry["id"] == action.memory_id for entry in pending):
            return
        pending.append({
            "id": action.memory_id,
            "with": action.conflicts_with,
            "reason": action.reason,
        })
        self._upkeep_set("conflict:pending", user_id, pending)

    def _open_conflicts(
        self, user_id: str | None
    ) -> list[tuple[dict[str, Any], Memory, Memory]]:
        """Queued contradictions whose two memories are both still in use."""
        pending = self._upkeep_get("conflict:pending", user_id, [])
        live: list[tuple[dict[str, Any], Memory, Memory]] = []
        for entry in pending:
            new = self.backend.get_memory(entry["id"])
            old = self.backend.get_memory(entry["with"])
            if new is None or old is None:
                continue
            if new.invalid_at is not None or old.invalid_at is not None:
                # settled some other way: one of them was deleted or replaced
                self._clear_conflict_mark(new)
                continue
            live.append((entry, new, old))
        if len(live) != len(pending):
            self._upkeep_set(
                "conflict:pending", user_id, [entry for entry, _, _ in live]
            )
        return live

    def _clear_conflict_mark(self, memory: Memory) -> None:
        if CONFLICT_KEY not in (memory.metadata or {}):
            return
        metadata = dict(memory.metadata)
        metadata.pop(CONFLICT_KEY, None)
        self.backend.update_memory(memory.id, metadata=metadata, touch=False)

    def _decide_conflict(
        self, item_id: str, decision: str, *,
        user_id: str | None, owner_prefix: str | None,
        actor: str = "user", why: str | None = None, update: bool | None = None,
    ) -> bool:
        """Settle one queued contradiction: "accept" (the new one replaces the
        old), "decline" (the new one is wrong) or "other" (both are true).
        ``actor``, ``why`` and ``update`` are for a decision Memry makes itself
        (``redecide_conflicts``): who decided, the reason recorded, and
        whether a replacement is an update (kept as history)."""
        found = next(
            (row for row in self._open_conflicts(user_id) if row[0]["id"] == item_id),
            None,
        )
        if found is None:
            return False
        _, new, old = found
        if not (_owned(new, owner_prefix) and _owned(old, owner_prefix)):
            return False
        # held back from an update (a change, or a MORE with no merged text):
        # a confirmed replacement is an update's, which keeps the old one as
        # history and which the Archive's undo reverses keeping both
        # (``undo_replacement``)
        if update is None:
            update = _conflict_mark(new).get("kind") == "update"
        if decision == "accept":  # the new one is right
            self.backend.invalidate_memory(old.id, superseded_by=new.id)
            self.backend.add_event(MemoryEvent(
                memory_id=old.id, event="SUPERSEDE", old_content=old.content,
                new_content=new.content, actor=actor,
                reason=why or (f"you confirmed that memory {new.id} updates it" if update
                               else f"you confirmed that memory {new.id} replaces it"),
                kind="update" if update else "contradiction",
            ))
        elif decision == "decline":  # the old one is right
            self.backend.invalidate_memory(new.id)
            self.backend.add_event(MemoryEvent(
                memory_id=new.id, event="DELETE", old_content=new.content,
                actor="user",
                reason=f"you judged it wrong: it {'updated' if update else 'contradicted'} "
                       f"memory {old.id}, which you kept",
            ))
        else:  # both are true
            self.backend.add_event(MemoryEvent(
                memory_id=new.id, event="NONE", new_content=new.content,
                actor=actor,
                reason=why or f"you kept it beside memory {old.id}: both are true",
            ))
        self._clear_conflict_mark(new)
        self._upkeep_set(
            "conflict:pending", user_id,
            [e for e in self._upkeep_get("conflict:pending", user_id, [])
             if e["id"] != item_id],
        )
        return True

    def redecide_conflicts(
        self, *, user_id: str | None = None, apply: bool = False
    ) -> list[dict[str, Any]]:
        """Ask the decision provider again about each queued contradiction
        and say, per item, what the rule it was held under decides
        (``reconcile.held_back``: importance holds it) and what the rule of
        now decides (``reconcile.replacement_verdict``: a protected state that
        moved on is replaced at a raised bar, a memory still true is kept
        beside, a lasting fact or rule still asks). One provider call per
        item; nothing is written unless ``apply``. With ``apply`` a "replace"
        or "update" supersedes the old memory as a person's yes would, by
        Memry and with the reason, listed under Archive and undone there; a
        "both" keeps both and clears the question; an "ask" stays."""
        cfg = self.config.supersede
        out: list[dict[str, Any]] = []
        for entry, new, old in self._open_conflicts(user_id):
            row: dict[str, Any] = {
                "id": new.id, "with": old.id, "new": new.content, "old": old.content,
                "importance": old.importance, "queued": entry.get("reason")}
            state = reconcile_state([old], new.content, new.created_at)
            try:
                judged = _decide_action(self.decider, state, 1, standing=True)
            except Exception as exc:  # one item's outage leaves the others
                judged = None
                row["error"] = str(exc)
            if judged is None:
                row.update(answer=None, before="ask", now="ask",
                           why="the decision provider gave no answer")
                out.append(row)
                continue
            action = judged["action"]
            saves = saves_of(self.backend, old)
            bar = bar_for(self.decider, action, cfg)
            gone = no_longer_holds(judged)
            row["answer"] = {"action": action, "confidence": judged.get("confidence"),
                             "no_longer_holds": round(gone, 3) if gone is not None else None,
                             "standing": judged.get("standing")}
            if action in ("CHANGED", "WRONG"):
                held = held_back(old, judged, cfg, bar=bar, saves=saves)
                row["before"] = "ask" if held else "replace"
                row["now"], row["why"] = replacement_verdict(
                    action, judged, old, cfg, bar=bar, saves=saves)
                if row["now"] == "replace" and action == "WRONG":
                    row["now"] = "contradiction"
            else:  # NEW, SAME or MORE: no conflict between the two
                row["before"] = row["now"] = "both"
                row["why"] = f"the judge answered {action}: no conflict"
            row["applied"] = False
            if apply and row["now"] != "ask":
                why = f"Memry decided again ({row['answer']['action']}): {row['why']}"
                row["applied"] = self._decide_conflict(
                    new.id, "other" if row["now"] == "both" else "accept",
                    user_id=user_id, owner_prefix=None, actor="system", why=why,
                    update=row["now"] in ("replace", "update"))
            out.append(row)
        return out

    def replaced(
        self, *, user_id: str | None = None, limit: int = 200
    ) -> list[dict[str, Any]]:
        """Memories a contradiction or an update took out of use, newest
        first; ``contradiction`` says which.

        A memory that was consolidated or distilled lives on inside what
        replaced it, so there is nothing to undo. One that was contradicted is
        the opposite case - the store stopped believing it on one model's
        say-so - and that is the judgement worth a second look. One an update
        superseded (a change, a merged detail, or an addition with no merged
        text written) held until then and stays searchable as history: if the
        update was a mistake, its undo brings it back into use beside the
        newer one. One split into single facts (``split_memories``) is listed
        too, with the facts (``parts``): its undo brings it back and forgets
        them.
        """
        scope = Scope(user_id=user_id)
        out: list[dict[str, Any]] = []
        for memory in self.backend.list_memories(
            scope, include_invalid=True, limit=1_000_000
        ):
            if memory.invalid_at is None or not memory.superseded_by:
                continue
            event = next(
                (
                    e for e in reversed(self.backend.history(memory.id))
                    if e.event == "SUPERSEDE"
                ),
                None,
            )
            if event is None:
                continue
            contradiction = _is_contradiction(event)
            split = event.kind == "split"
            if not (contradiction or split or _is_update_supersede(event)):
                continue
            out.append({
                "memory": memory,
                "replaced_at": memory.invalid_at,
                "replacement": self.backend.get_memory(memory.superseded_by),
                "reason": event.reason,
                "actor": event.actor,
                "contradiction": contradiction,
                "split": split,
                "parts": [part for part_id in
                          ((memory.metadata or {}).get("split_into") or [] if split else [])
                          if (part := self.backend.get_memory(part_id)) is not None],
            })
        out.sort(key=lambda row: row["replaced_at"] or "", reverse=True)
        return out[:limit]

    def undo_replacement(
        self, memory_id: str, *, keep_new: bool = False,
        owner_prefix: str | None = None,
    ) -> bool:
        """Bring back into use a memory that a contradiction or an update
        replaced, or that was split into single facts.

        ``keep_new`` leaves the replacement in use as well, for when both turn
        out to be true. Otherwise the replacement is forgotten - it goes to the
        Archive like any deleted memory, so this is itself undoable. The
        replacement of an update never contradicted the memory and is always
        kept. A split's facts are forgotten, unless ``keep_new``
        (``_undo_split``).
        """
        memory = self.backend.get_memory(memory_id)
        if not _owned(memory, owner_prefix):
            return False
        if memory.invalid_at is None or not memory.superseded_by:
            return False
        event = next(
            (e for e in reversed(self.backend.history(memory_id))
             if e.event == "SUPERSEDE"),
            None,
        )
        if event is not None and event.kind == "split":
            return self._undo_split(memory, keep_new=keep_new, owner_prefix=owner_prefix)
        if event is not None and _is_update_supersede(event):
            keep_new = True  # the newer memory adds to it; both stay
        elif event is None or not _is_contradiction(event):
            raise ValueError(
                "this memory was merged into its replacement, not contradicted "
                "by it; there is nothing to undo"
            )
        replacement = self.backend.get_memory(memory.superseded_by)
        if self.backend.revalidate_memory(memory_id) is None:
            return False
        self.backend.add_event(MemoryEvent(
            memory_id=memory_id, event="ADD", new_content=memory.content,
            actor="user",
            reason=f"you undid its replacement by memory {memory.superseded_by}",
        ))
        if (
            not keep_new and replacement is not None
            and replacement.invalid_at is None and _owned(replacement, owner_prefix)
        ):
            self.backend.invalidate_memory(replacement.id)
            self.backend.add_event(MemoryEvent(
                memory_id=replacement.id, event="DELETE",
                old_content=replacement.content, actor="user",
                reason=f"you judged it wrong: it had replaced memory {memory_id}",
            ))
        return True

    def purge(self, memory_id: str, *, owner_prefix: str | None = None) -> bool:
        """Delete a forgotten memory for good. Refuses anything still active.

        Permanent deletion is the one operation with no audit trail left to
        inspect afterwards, so it is deliberately a second step: a memory has to
        have been forgotten first. That makes an accidental irreversible delete
        take two decisions instead of one bad click.
        """
        memory = self.backend.get_memory(memory_id)
        if not _owned(memory, owner_prefix):
            return False
        if memory.invalid_at is None:
            raise ValueError("only a forgotten memory can be permanently deleted")
        return self._delete_for_good(memory_id)

    def delete_all(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        hard: bool = False,
    ) -> int:
        scope = Scope(user_id=user_id, agent_id=agent_id, run_id=run_id)
        memories = self.backend.list_memories(scope, limit=1_000_000)
        for memory in memories:
            self.delete(memory.id, hard=hard)
        return len(memories)

    # ------------------------------------------------------------------
    # entities
    # ------------------------------------------------------------------
    def entities(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        include_merged: bool = False,
        limit: int = 100,
        kind: str = "named",
        exact_user: bool = False,
    ) -> list[Entity]:
        """Entities in scope: named things by default, tags (topic entities)
        with ``kind="topic"``, both with ``kind="any"``. ``exact_user`` as in
        ``get_all``."""
        scope = Scope(user_id=user_id, agent_id=agent_id, run_id=run_id,
                      exact_user=exact_user)
        return self.backend.list_entities(
            scope, include_merged=include_merged, limit=limit, kind=kind)

    def relations(self, *, user_id: str | None = None, limit: int = 1000) -> list[Relation]:
        return self.backend.list_relations(Scope(user_id=user_id), limit=limit)

    def restore_context_labels(
        self, *, user_id: str | None = None, dry_run: bool = False, exact_user: bool = False,
    ) -> dict[str, Any]:
        """Give memories back the context label of the saves they came from.

        Facts extracted from a save did not keep the save's context label
        (fixed in the write path); the save's episode kept it, and every fact
        keeps its episode ids. A memory without a label takes the labels of its
        episodes, distinct ones joined as distillation joins them. Only
        memories without a label are looked at, so a second run changes
        nothing. Token-free. ``dry_run`` counts without writing; ``exact_user``
        as in ``get_all``."""
        missing = [
            m for m in self.get_all(user_id=user_id, exact_user=exact_user, limit=1_000_000)
            if not _ingestion_context(m.metadata) and m.source_episode_ids
        ]
        episodes = self.backend.episodes_by_id(
            [e for m in missing for e in m.source_episode_ids]
        )
        summary = {"without_label": len(missing), "restorable": 0, "restored": 0,
                   "save_had_no_label": 0}
        for memory in missing:
            labels = list(dict.fromkeys(
                label for episode_id in memory.source_episode_ids
                if (episode := episodes.get(episode_id))
                and (label := _ingestion_context(episode.metadata))
            ))
            if not labels:
                summary["save_had_no_label"] += 1
                continue
            summary["restorable"] += 1
            if not dry_run:
                self.backend.update_memory(
                    memory.id,
                    metadata={**memory.metadata, "context": " | ".join(labels)[:200]},
                    touch=False,
                )
                summary["restored"] += 1
        return summary

    def repair_updated_at(
        self, *, user_id: str | None = None, exact_user: bool = False,
    ) -> dict[str, Any]:
        """Reconstruct each memory's updated_at from its audit trail.

        Housekeeping (tagging, relation backfill, re-embedding) used to bump
        updated_at; this recomputes the true value as the time of the last
        content-changing event (ADD/UPDATE/SUPERSEDE), or created_at if there was
        none. Token-free; idempotent. Fixes recency and decay after such a run.

        Times are compared as times (``later_ts``, ``same_ts``), as the write
        path compares them: a replayed save's "...T10:00:00Z" is earlier than
        a live "...T10:00:00.500000+00:00", though it sorts later as text.
        ``exact_user`` as in ``get_all``.
        """
        fixed = 0
        for memory in self.get_all(user_id=user_id, include_invalid=True,
                                   exact_user=exact_user, limit=1_000_000):
            times = [
                e.created_at for e in self.backend.history(memory.id)
                if e.event in ("ADD", "UPDATE", "SUPERSEDE")
            ]
            true_ts = memory.created_at
            for stamp in times:
                true_ts = later_ts(true_ts, stamp)
            if not same_ts(true_ts, memory.updated_at):
                self.backend.set_memory_timestamp(memory.id, true_ts)
                fixed += 1
        return {"fixed": fixed}

    def split_memories(
        self, *, user_id: str | None = None, dry_run: bool = False, min_words: int = 0,
        exact_user: bool = False, plan: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Split each memory in use that holds several facts into one memory
        per fact (``intelligence/split.py``).

        A memory is asked about when its text has more than one sentence (and
        at least ``min_words`` words); the text model splits it, and one fact
        back leaves it alone. A split is made only when every fact states its
        subject and no detail is lost. A fact that names nothing the memory's
        text names, nor the owner (``split.without_subject``), is worse than
        the memory it came from, so its memory is kept whole, and
        the coverage audit a save gets reads the facts against the memory:
        a split it finds lossy is not made either. The same call gives each
        fact the entities of the memory it is about (``split.split_facts``):
        named things and tags, also one the fact does not spell out (the
        tallest bulls are elephants too). A fact keeps those and any its text
        names (``split.fact_homes``); a split with a fact that would keep
        none, or an entity no fact would keep, is not made either
        (``no_entity``, ``lost_entity``). Each fact becomes a memory with the
        old one's dates (``created_at``, ``updated_at``, ``valid_from``; a
        date heading the whole memory is the ``valid_from``,
        ``_split_valid_from``), sources, importance, type, metadata ("when"
        included), run and agent, linked to the entities it keeps, and tagged
        with the tags among them (``_split_tags``); each of the old memory's
        relations rests on the fact keeping both ends. The old memory
        leaves use as a SUPERSEDE of kind "split", listed under
        Archive, whose undo (``undo_replacement``) brings it back and forgets
        the facts. Each memory is split in one transaction (``_split_memory``):
        one whose writes fail is left as it was, counted as failed.

        ``dry_run`` asks the model and writes nothing. The summary counts the
        memories in use, the candidates, those one fact, those split (or that
        would be) and the facts they make, those kept for their entities
        (above), because a fact would not state its subject (``no_subject``)
        or a detail would be lost (``lossy``), and lists each split and each
        kept, with the entities each fact keeps (``about``, by id;
        ``labels``, each id as a person reads it).

        ``exact_user``: no ``user_id`` means the memories without a user, not
        every user's (``Scope.exact_user``), for a walk over the namespaces.

        ``plan``: the splits of a dry run a person read (``split.make_plan``,
        ``split.plan_entries``), made as they are, with no call to the text
        model. Asked again, the model answers a little differently, so a dry
        run alone is only a preview. A planned memory is split into the
        plan's facts only while it is in use in this namespace with the text
        the plan judged, and into the entities the plan gave each fact (an
        entity merged since is its survivor); one that left use, changed
        since, or whose entities did, is skipped and listed (``stale``).

        A real split (not ``dry_run``) refuses to run on a backend without
        transactions (``supports_transactions``): each memory's writes are
        made together or not at all."""
        from .intelligence.split import (
            entity_gaps, entity_label, fact_homes, is_candidate, split_facts,
            without_subject)

        if not dry_run and not self.backend.supports_transactions:
            # without one, a failure midway leaves facts beside the memory
            raise ValueError("this storage backend cannot keep one memory's writes "
                             "together (no transactions), so a split could be left half "
                             "made; only a dry run can run on it")
        if plan is not None:
            return self._apply_split_plan(plan, user_id=user_id, dry_run=dry_run,
                                          exact_user=exact_user)
        if not self.llm.available:
            raise ValueError("no LLM configured; splitting memories needs one")
        in_use = self.get_all(user_id=user_id, exact_user=exact_user, limit=1_000_000)
        candidates = [m for m in in_use if is_candidate(m, min_words=min_words)]
        summary: dict[str, Any] = {
            "user": user_id, "dry_run": dry_run, "in_use": len(in_use),
            "candidates": len(candidates), "one_fact": 0, "split": 0, "facts": 0,
            "no_entity": 0, "lost_entity": 0, "no_subject": 0, "lossy": 0, "failed": 0,
            "splits": [],
        }
        # what each memory is about: its entities, each with its names
        links = {m.id: self._split_links(m.id) for m in candidates}
        # the names of its named things, a fact's subject when the text names one
        linked = {mid: [name for entity, names in pairs if entity.entity_type != TOPIC_TYPE
                        for name in names] for mid, pairs in links.items()}
        # who "the user" is: a fact may name the owner by their name instead
        owner_entity = self.owner_entity(user_id)
        owner = [name for name in (
            self.owner_name(user_id),
            *((owner_entity.name, *self.backend.entity_aliases(owner_entity.id))
              if owner_entity is not None else ())) if name]

        def ask(memory: Memory) -> dict[str, Any] | Exception:
            try:
                pairs = links[memory.id]
                answer = split_facts(self.llm, memory, [entity for entity, _ in pairs])
                facts = [fact["text"] for fact in answer]
                if len(facts) < 2:
                    return {"facts": facts}
                about = fact_homes(facts, [fact["about"] for fact in answer],
                                   [(entity.id, names) for entity, names in pairs])
                homeless, lost = entity_gaps(about, [entity.id for entity, _ in pairs])
                if homeless or lost:  # no audit for a split that is not made anyway
                    return {"facts": facts, "about": about, "homeless": homeless,
                            "lost": lost}
                orphans = without_subject(facts, memory, linked[memory.id], owner)
                if orphans:
                    return {"facts": facts, "about": about, "orphans": orphans}
                return {"facts": facts, "about": about, "missing": verify_coverage(
                    self.llm, [{"role": "user", "content": memory.content}], facts)}
            except Exception as exc:  # one failed call leaves that memory as it is
                return exc

        # the model is asked about several memories at once; the writes that
        # follow go one memory at a time, in the order of the candidates
        with ThreadPoolExecutor(max_workers=_SPLIT_WORKERS) as pool:
            answers = list(pool.map(ask, candidates))
        for memory, answer in zip(candidates, answers):
            if isinstance(answer, Exception):
                summary["failed"] += 1
                log.warning("splitting memory %s failed: %s", memory.id, answer)
                continue
            facts = answer["facts"]
            if len(facts) < 2:
                summary["one_fact"] += 1
                continue
            labels = {entity.id: entity_label(entity) for entity, _ in links[memory.id]}
            entry: dict[str, Any] = {"memory_id": memory.id, "user": memory.user_id,
                                     "content": memory.content, "facts": facts,
                                     "about": answer["about"], "labels": labels}
            if answer.get("homeless"):
                summary["no_entity"] += 1
                entry["not_split"] = ("a fact would keep none of the memory's entities: "
                                      + "; ".join(facts[i] for i in answer["homeless"]))
                summary["splits"].append(entry)
                continue
            if answer.get("lost"):
                summary["lost_entity"] += 1
                entry["not_split"] = ("an entity would be lost: "
                                      + ", ".join(labels[i] for i in answer["lost"]))
                summary["splits"].append(entry)
                continue
            if answer.get("orphans"):
                summary["no_subject"] += 1
                entry["not_split"] = ("a fact would not state its subject: "
                                      + "; ".join(answer["orphans"]))
                summary["splits"].append(entry)
                continue
            if answer.get("missing"):
                summary["lossy"] += 1
                entry["not_split"] = "the facts would lose: " + "; ".join(answer["missing"])
                summary["splits"].append(entry)
                continue
            self._split_entry(memory, entry, summary, dry_run=dry_run)
        return summary

    def _apply_split_plan(
        self, plan: list[dict[str, Any]], *, user_id: str | None, dry_run: bool,
        exact_user: bool,
    ) -> dict[str, Any]:
        """``split_memories`` with a ``plan``: no model is asked. A planned
        memory is looked up among this namespace's memories in use, never by
        its id alone, so a plan cannot reach another namespace's memory. One
        not found there, whose text is not the text the plan judged, or that
        waits for its extraction or for a person now, is skipped. So is one
        whose entities changed: a planned entity that is gone, or that the
        memory is no longer linked to (merged ones are followed to their
        survivor), or an entity linked since that no fact would keep."""
        from .intelligence.split import entity_gaps, entity_label, fact_homes, is_candidate

        in_use = {m.id: m for m in self.get_all(
            user_id=user_id, exact_user=exact_user, limit=1_000_000)}
        summary: dict[str, Any] = {
            "user": user_id, "dry_run": dry_run, "in_use": len(in_use),
            "planned": len(plan), "split": 0, "facts": 0, "stale": 0, "failed": 0,
            "splits": [],
        }
        for planned in plan:
            memory = in_use.get(planned["memory_id"])
            entry: dict[str, Any] = {"memory_id": planned["memory_id"], "user": user_id,
                                     "content": planned["content"],
                                     "facts": list(planned["facts"]),
                                     "about": [list(ids) for ids in planned["about"]],
                                     "labels": {}}
            stale = None
            if memory is None:
                stale = "not in use in this namespace"
            elif memory.content != planned["content"]:
                stale = f"its text changed since the plan, to: {memory.content}"
            elif not is_candidate(memory):
                stale = "it waits for its extraction or for a person now"
            else:
                pairs = self._split_links(memory.id)
                entry["labels"] = {entity.id: entity_label(entity) for entity, _ in pairs}
                resolved = [[self.backend.resolve_entity_id(i) for i in ids]
                            for ids in planned["about"]]
                if any(i is None or i not in entry["labels"] for ids in resolved for i in ids):
                    stale = "its entities changed since the plan"
                else:
                    entry["about"] = fact_homes(
                        entry["facts"], resolved, [(e.id, names) for e, names in pairs])
                    homeless, lost = entity_gaps(entry["about"], list(entry["labels"]))
                    if homeless or lost:
                        stale = "its entities changed since the plan"
            if stale is None:
                self._split_entry(memory, entry, summary, dry_run=dry_run)
                continue
            summary["stale"] += 1
            entry["not_split"] = stale
            summary["splits"].append(entry)
        return summary

    def _split_links(self, memory_id: str) -> list[tuple[Entity, list[str]]]:
        """The entities a memory is linked to, as a split numbers them for
        the model: its named things, then its tags, each with its names and
        aliases."""
        return [(entity, [entity.name, *self.backend.entity_aliases(entity.id)])
                for kind in ("named", "topic")
                for entity in self.backend.entities_of_memory(memory_id, kind=kind)]

    def _split_tags(
        self, memory: Memory, facts: list[str], about: list[list[str]],
    ) -> list[list[str]]:
        """Each fact's tags: of the memory's, those whose entity the fact
        keeps, as the memory's column writes them. Every fact took them all,
        and a conversation summary of five topics made five facts that each
        ranked for all five. A tag with no entity the memory is linked to (a
        column out of step with its mentions) goes on the facts whose text
        names it, else on all of them; a tag entity the fact keeps that no
        tag of the column names is added by its name."""
        from .intelligence.split import names_in

        scope = Scope(user_id=memory.user_id)
        linked = {entity.id: entity
                  for entity in self.backend.entities_of_memory(memory.id, kind="any")}
        owner: dict[str, str | None] = {}
        for tag in memory.categories:
            entity = self.backend.topic_entity(tag, scope, create=False, follow_merged=True)
            owner[tag] = entity.id if entity is not None and entity.id in linked else None
        named = {tag: [i for i, fact in enumerate(facts) if names_in(fact, [tag])]
                 or list(range(len(facts))) for tag, entity_id in owner.items()
                 if entity_id is None}
        out: list[list[str]] = []
        for i, kept in enumerate(about):
            tags = [tag for tag, entity_id in owner.items()
                    if (entity_id in kept if entity_id else i in named[tag])]
            tags += [linked[entity_id].name for entity_id in kept
                     if entity_id in linked and linked[entity_id].entity_type == TOPIC_TYPE
                     and entity_id not in owner.values()]
            out.append(tags)
        return out

    def _split_entry(
        self, memory: Memory, entry: dict[str, Any], summary: dict[str, Any], *,
        dry_run: bool,
    ) -> None:
        """Make one split of ``split_memories`` (unless ``dry_run``) and count
        it in ``summary``. Writes that fail leave the memory as it was
        (``_split_memory``): counted as failed, and listed with the reason."""
        if not dry_run:
            try:
                entry["memory_ids"] = self._split_memory(memory, entry["facts"],
                                                         entry["about"])
            except Exception as exc:
                summary["failed"] += 1
                log.warning("splitting memory %s failed: %s", memory.id, exc)
                entry["not_split"] = f"the split failed, the memory is as it was: {exc}"
                summary["splits"].append(entry)
                return
        summary["split"] += 1
        summary["facts"] += len(entry["facts"])
        summary["splits"].append(entry)

    def _split_memory(
        self, memory: Memory, facts: list[str], about: list[list[str]],
    ) -> list[str]:
        """Replace ``memory`` with one memory per fact; return their ids.
        ``about``: the ids of the entities each fact keeps (``fact_homes``).
        Its named things are linked to it with the mention the memory had,
        and its tags come with its column (``_split_tags``), as every write
        files them.

        The facts are embedded first, in one call, and every write is then
        made in one transaction (``backend.transaction``). Each fact was once
        embedded and saved in turn, and the old memory left use only after
        the last: a failure or a stop midway left the facts saved so far in
        use beside it, twins its undo could not find. Now it leaves the
        memory as it was. A memory another writer took out of use meanwhile
        is not split (ValueError).

        The ADD event of each fact is dated at the old memory's
        ``updated_at``, which the fact keeps, so ``repair_updated_at`` reads
        the same time; the reason says when the split happened."""
        from .intelligence.split import names_in

        now = utcnow()
        valid_from = self._split_valid_from(memory)
        metadata = {key: value for key, value in (memory.metadata or {}).items()
                    if key not in (CONFLICT_KEY, "pending_distillation", _ENRICHMENT_KEY)}
        metadata["split_from"] = memory.id
        linked: list[tuple[Entity, EntityMention | None]] = []
        for entity in self.backend.entities_of_memory(memory.id):
            mention = next((m for m in self.backend.entity_mentions(entity.id)
                            if m.memory_id == memory.id), None)
            linked.append((entity, mention))
        homes = {entity.id: [i for i, kept in enumerate(about) if entity.id in kept]
                 for entity, _ in linked}
        tags = self._split_tags(memory, facts, about)
        parts: list[Memory] = []
        for i, fact in enumerate(facts):
            # the named things it keeps, and a name the memory listed that it states
            names = [entity.name for entity, _ in linked if i in homes[entity.id]]
            names += [name for name in memory.entities if names_in(fact, [name])
                      and name.casefold() not in {n.casefold() for n in names}]
            parts.append(Memory(
                content=fact, memory_type=memory.memory_type, user_id=memory.user_id,
                agent_id=memory.agent_id, run_id=memory.run_id, importance=memory.importance,
                categories=tags[i], entities=names,
                metadata=dict(metadata), created_at=memory.created_at,
                updated_at=memory.updated_at, valid_from=valid_from,
                source_episode_ids=list(memory.source_episode_ids),
            ))
        # the one call over the network, before the transaction holds the database
        vectors = (self.embedder.embed(list(facts)) if self.embedder.dimensions
                   else [None] * len(facts))
        ids = [part.id for part in parts]
        with self.backend.transaction():
            relations = [r for r in self.backend.list_relations(
                Scope(user_id=memory.user_id, exact_user=True), limit=1_000_000)
                if r.memory_id == memory.id and r.invalid_at is None]
            # out of use first: its relations end with it, and each comes back
            # on a fact (one live edge per triple)
            if self.backend.invalidate_memory(memory.id, superseded_by=ids[0], at=now) is None:
                raise ValueError(f"memory {memory.id} left use before it was split")
            for part, vector in zip(parts, vectors):
                if vector:
                    part.embedding_model = self.embedder.model_id
                self.backend.insert_memory(part, embedding=vector or None)
            for entity, mention in linked:
                for i in homes[entity.id]:
                    if entity.id in {e.id for e in self.backend.entities_of_memory(
                            ids[i], kind="any")}:
                        continue  # a tag of its column merged into this thing linked it
                    self.backend.add_mention(EntityMention(
                        entity_id=entity.id, memory_id=ids[i],
                        surface=mention.surface if mention else entity.name,
                        decided=mention.decided if mention else None,
                        entity_type=mention.entity_type if mention else None))
            retired = self.backend.get_memory(memory.id)
            self.backend.update_memory(
                memory.id, metadata={**(retired.metadata if retired else memory.metadata),
                                     "split_into": ids}, touch=False)
            for relation in relations:
                ends = [homes.get(relation.subject, []), homes.get(relation.object, [])]
                both = [i for i in ends[0] if i in ends[1]]
                home = (both or ends[0] or ends[1] or [0])[0]
                self.backend.add_relation(Relation(
                    subject=relation.subject, predicate=relation.predicate,
                    object=relation.object, user_id=relation.user_id, memory_id=ids[home],
                    created_at=now, valid_from=relation.valid_from))
            for part in parts:
                self.backend.add_event(MemoryEvent(
                    memory_id=part.id, event="ADD", new_content=part.content,
                    reason=f"split on {now[:10]} from memory {memory.id}, one of its "
                           f"{len(parts)} facts",
                    created_at=memory.updated_at))
            self.backend.add_event(MemoryEvent(
                memory_id=memory.id, event="SUPERSEDE", old_content=memory.content,
                new_content="\n".join(facts),
                reason=f"split into {len(parts)} facts, one memory each: {', '.join(ids)}",
                kind="split", created_at=now))
        self._property_vectors_after_save(ids)
        return ids

    @staticmethod
    def _split_valid_from(memory: Memory) -> str | None:
        """When the facts of a split hold from: the date heading the whole
        memory ("Decision (2026-09-12): ..."), which the prompt keeps in the
        first fact only (``split.heading_date``), else the memory's own
        ``valid_from``. A save sets ``valid_from`` to when it was said, so a
        decision written down three days after it was made held from three
        days late in every fact. A memory with an occurrence time ("when")
        keeps its ``valid_from``, and so does one already dated that day,
        whose time is the more exact. Every fact takes the heading's date,
        one about an event of its own date too: a fact's own date is not
        read."""
        from .intelligence.split import heading_date

        heading = heading_date(memory.content)
        if (heading is None or (memory.metadata or {}).get(WHEN_KEY)
                or (memory.valid_from or "")[:10] == heading[:10]):
            return memory.valid_from
        return heading

    def _undo_split(
        self, memory: Memory, *, keep_new: bool, owner_prefix: str | None,
    ) -> bool:
        """Bring back a memory that was split, and forget the facts it was
        split into that are still in use as they were made (a fact changed
        since, by a merge or a delete, is left as it is). ``keep_new`` keeps
        them in use as well."""
        parts = [self.backend.get_memory(i)
                 for i in (memory.metadata or {}).get("split_into") or []]
        if not keep_new:
            for part in parts:
                if (part is not None and part.invalid_at is None
                        and _owned(part, owner_prefix)):
                    self.backend.invalidate_memory(part.id)
                    self.backend.add_event(MemoryEvent(
                        memory_id=part.id, event="DELETE", old_content=part.content,
                        actor="user",
                        reason=f"you undid the split of memory {memory.id}, which is back"))
        if self.backend.revalidate_memory(memory.id) is None:
            return False
        restored = self.backend.get_memory(memory.id)
        metadata = dict(restored.metadata if restored else memory.metadata)
        metadata.pop("split_into", None)
        self.backend.update_memory(memory.id, metadata=metadata, touch=False)
        self.backend.add_event(MemoryEvent(
            memory_id=memory.id, event="ADD", new_content=memory.content, actor="user",
            reason=f"you undid its split into {len(parts)} facts"))
        return True

    def backfill_relations(
        self, *, user_id: str | None = None, limit: int = 100_000, exact_user: bool = False,
    ) -> dict[str, Any]:
        """One-time: extract typed relations from existing memories.

        Only memories with 2+ linked entities are considered (a relation needs
        two), each does one small focused LLM call, and each is marked done so a
        re-run spends no tokens. Cheap and resumable by design. ``exact_user``
        as in ``get_all``.
        """
        summary = {"scanned": 0, "processed": 0, "relations_added": 0, "skipped": 0}
        if not self.llm.available:
            summary["error"] = "no LLM configured"
            return summary
        for memory in self.get_all(user_id=user_id, exact_user=exact_user, limit=limit):
            summary["scanned"] += 1
            if memory.metadata.get("relations_backfilled"):
                continue
            entities = self.backend.entities_of_memory(memory.id)
            if len(entities) < 2:
                summary["skipped"] += 1
                self.backend.update_memory(
                    memory.id, metadata={**memory.metadata, "relations_backfilled": True},
                    touch=False,
                )
                continue
            by_norm = {e.normalized or e.name.lower(): e for e in entities}
            try:
                triples = extract_relations(
                    self.llm, memory.content, [e.name for e in entities]
                )
            except Exception:
                continue  # provider hiccup: leave unmarked, retry on next run
            for t in triples:
                subj = by_norm.get(t["subject"].strip().lower())
                obj = by_norm.get(t["object"].strip().lower())
                if subj is None or obj is None or subj.id == obj.id:
                    continue
                self.backend.add_relation(
                    Relation(subject=subj.id, predicate=t["predicate"], object=obj.id,
                             user_id=memory.user_id, memory_id=memory.id)
                )
                summary["relations_added"] += 1
            summary["processed"] += 1
            self.backend.update_memory(
                memory.id, metadata={**memory.metadata, "relations_backfilled": True},
                touch=False,
            )
        return summary

    def backfill_entity_types(
        self, *, user_id: str | None = None, batch: int = 40, exact_user: bool = False,
    ) -> dict[str, Any]:
        """Classify entities that were linked before typing existed. Batched:
        one LLM call per ``batch`` entities, so a whole namespace is a handful of
        calls. Only untyped entities are touched, so re-runs cost nothing.
        ``exact_user`` as in ``get_all``."""
        summary = {"typed": 0}
        if not self.llm.available:
            summary["skipped"] = "no LLM configured"
            return summary
        untyped = [
            e for e in self.backend.list_entities(
                Scope(user_id=user_id, exact_user=exact_user), limit=1_000_000)
            if not e.entity_type
        ]
        for i in range(0, len(untyped), batch):
            group = untyped[i : i + batch]
            try:
                types = classify_entity_types(
                    self.llm, [e.name for e in group], self.decider
                )
            except Exception:
                continue
            for e in group:
                etype = types.get(e.name.lower())
                if etype:
                    self.backend.set_entity_type(e.id, etype)
                    summary["typed"] += 1
        return summary

    def _confirm_candidate_whens(
        self, candidates: list[Any], *, now: datetime | None = None
    ) -> None:
        """Drop an extracted "when" the checks in intelligence/when.py do not
        believe. The text model proposes a date; it is not taken at its word.
        ``now`` is the day of writing (the clock unless a save gave one)."""
        dated = [c for c in candidates if (c.metadata or {}).get(WHEN_KEY)]
        if not dated:
            return
        today = now.isoformat(timespec="seconds") if now is not None else utcnow()
        kept = confirm_whens(
            self.decider,
            [{"content": c.content, "recorded_at": today} for c in dated],
            [c.metadata[WHEN_KEY] for c in dated],
        )
        for candidate, when in zip(dated, kept):
            if when is None:
                candidate.metadata.pop(WHEN_KEY, None)

    def backfill_when(
        self,
        *,
        user_id: str | None = None,
        batch: int = 20,
        limit: int | None = None,
        dry_run: bool = False,
        types: tuple[str, ...] = ("episodic", "semantic"),
    ) -> dict[str, Any]:
        """Read an occurrence time out of memories written before there was one.

        Only memories with neither a ``when`` nor a ``when_checked`` mark are
        looked at, and every memory that comes back without one is marked, so a
        second run over the same store spends nothing. ``dry_run`` writes
        nothing and hands back what it would have set, which is the way to see
        what a batch of proposals looks like before paying for the whole store.
        """
        if not self.llm.available:
            return {"skipped": "no LLM configured"}
        summary: dict[str, Any] = {"checked": 0, "found": 0}
        proposals: list[dict[str, Any]] = []
        pending = [
            m for m in self.get_all(user_id=user_id, limit=1_000_000)
            if m.memory_type in types
            and not (m.metadata or {}).get(WHEN_KEY)
            and not (m.metadata or {}).get(WHEN_CHECKED_KEY)
        ]
        if limit is not None:
            pending = pending[: max(int(limit), 0)]
        for index in range(0, len(pending), max(int(batch), 1)):
            group = pending[index : index + max(int(batch), 1)]
            found = extract_when(
                self.llm,
                [
                    {"content": m.content, "recorded_at": m.created_at}
                    for m in group
                ],
            )
            found = confirm_whens(
                self.decider,
                [{"content": m.content, "recorded_at": m.created_at} for m in group],
                found,
            )
            for memory, when in zip(group, found):
                summary["checked"] += 1
                if when:
                    summary["found"] += 1
                if dry_run:
                    if when:
                        proposals.append(
                            {"id": memory.id, "content": memory.content, "when": when}
                        )
                    continue
                metadata = dict(memory.metadata or {})
                if when:
                    metadata[WHEN_KEY] = when
                else:
                    # Marked, not retried: a memory the model already read and
                    # found no time in costs nothing on the next run.
                    metadata[WHEN_CHECKED_KEY] = True
                self.backend.update_memory(
                    memory.id, metadata=metadata, touch=False
                )
        if dry_run:
            summary["proposals"] = proposals
        return summary

    def _undescribed(self, entity: Entity) -> Entity:
        """``entity`` shown without a description while it has fewer than
        ``DESCRIPTION_MIN_MEMORIES`` memories in use: the memory speaks for
        itself. One stored from when it had more stays stored, unshown, for
        when it has them again."""
        if entity.description is None or (
                self.backend.count_entity_memories(entity.id) >= DESCRIPTION_MIN_MEMORIES):
            return entity
        return entity.model_copy(update={"description": None})

    def _refresh_entity_description(
        self, entity_id: str, *, force: bool = False
    ) -> Entity | None:
        """The entity with its description, built or rebuilt where it is stale.
        With fewer than ``DESCRIPTION_MIN_MEMORIES`` memories in use there is
        none: no model is asked, nothing is stored, and one stored before is
        not shown (``_undescribed``)."""
        entity = self.backend.get_entity(entity_id)
        if entity is None or not entity.is_active:
            return None
        if self.backend.count_entity_memories(entity_id) < DESCRIPTION_MIN_MEMORIES:
            return entity.model_copy(update={"description": None})
        evidence_updated_at = self.backend.entity_evidence_updated_at(entity_id)
        if (
            not force
            and entity.description is not None
            and entity.description_updated_at is not None
            and (
                evidence_updated_at is None
                or entity.description_updated_at >= evidence_updated_at
            )
        ):
            return entity
        memories = self.backend.entity_memories(entity_id, limit=DESCRIPTION_FACTS)
        description = synthesize_entity_description(
            self.llm,
            entity,
            [memory.content for memory in memories],
            self.backend.entity_aliases(entity_id),
        )
        generated_at = utcnow()
        stored = self.backend.set_entity_description(
            entity_id, description, generated_at
        )
        if stored is not None:
            return stored
        entity.description = description
        entity.description_updated_at = generated_at
        return entity

    def entity(
        self,
        entity_id: str,
        *,
        owner_prefix: str | None = None,
        refresh_description: bool = True,
    ) -> dict[str, Any] | None:
        """One entity hub with aliases and active supporting memories."""
        entity = self.backend.get_entity(entity_id)
        if not _owned(entity, owner_prefix):
            return None
        if refresh_description:
            entity = self._refresh_entity_description(entity_id)
            if entity is None:
                return None
        else:
            entity = self._undescribed(entity)
        # Relations belong to the entity being looked at, not to a list of every
        # edge in the store: an edge only means something next to the thing it
        # connects. These are also what relational retrieval traverses, so
        # seeing them here is seeing why a search reached what it reached.
        relations = self.backend.relations_of([entity_id])
        endpoints = {r.subject for r in relations} | {r.object for r in relations}
        names = {
            other.id: other.name
            for other in (self.backend.get_entity(e) for e in endpoints)
            if other is not None
        }
        return {
            "entity": entity,
            "aliases": self.backend.entity_aliases(entity_id),
            "mentions": self.backend.entity_mentions(entity_id),
            "memories": self.backend.entity_memories(entity_id, limit=20),
            "relations": relations,
            "relation_names": names,
        }

    def add_entity_alias(
        self, entity_id: str, alias: str, *, owner_prefix: str | None = None
    ) -> Entity | None:
        entity = self.backend.get_entity(entity_id)
        if not _owned(entity, owner_prefix):
            return None
        return self.backend.add_entity_alias(entity_id, alias)

    def rename_entity(
        self, entity_id: str, name: str, *, owner_prefix: str | None = None
    ) -> Entity | None:
        """Rename the canonical entity while retaining its old name as an alias.

        A tag (topic entity) is renamed as a tag: on every memory carrying it
        (``rename_tag``), so its memories' ``categories`` say the new name.
        Returned is the entity the tag went into: a new topic of that name,
        the topic already named so, or, for a name merged away, its survivor
        (``_merge_topics``)."""
        entity = self.backend.get_entity(entity_id)
        if not _owned(entity, owner_prefix) or not name.strip():
            return None
        if entity.entity_type == TOPIC_TYPE:
            tag = next(iter(clean_tags(name)), None)
            if entity.merged_into is not None or tag is None:
                return None
            self._retag(entity.user_id, {entity.normalized}, tag, exact_user=True)
            return self.backend.topic_entity(
                tag.lower(), Scope(user_id=entity.user_id), create=False, follow_merged=True)
        return self.backend.rename_entity(entity_id, name)

    def merge_proposals(
        self,
        *,
        user_id: str | None = None,
        status: str | None = "proposed",
        limit: int = 100,
    ) -> list[MergeProposal]:
        proposals = self.backend.list_proposals(
            Scope(user_id=user_id), status=status, limit=limit
        )
        if status != "proposed":
            return proposals
        active: list[MergeProposal] = []
        for proposal in proposals:
            entity_a = self.backend.resolve_entity_id(proposal.entity_a)
            entity_b = self.backend.resolve_entity_id(proposal.entity_b)
            if entity_a is None or entity_b is None:
                self.backend.set_proposal_status(proposal.id, "rejected")
                continue
            if entity_a == entity_b:
                self.backend.set_proposal_status(proposal.id, "confirmed")
                continue
            active.append(
                proposal.model_copy(update={"entity_a": entity_a, "entity_b": entity_b})
            )
        return active

    def confirm_merge(
        self, proposal_id: str, *, owner_prefix: str | None = None
    ) -> bool:
        """Confirm a proposal after resolving both endpoints through merge history."""
        proposal = self.backend.get_proposal(proposal_id)
        if not _owned(proposal, owner_prefix) or proposal.status != "proposed":
            return False
        entity_a = self.backend.resolve_entity_id(proposal.entity_a)
        entity_b = self.backend.resolve_entity_id(proposal.entity_b)
        if entity_a is None or entity_b is None:
            return False
        if owner_prefix is not None and not all(
            _owned(self.backend.get_entity(entity_id), owner_prefix)
            for entity_id in (entity_a, entity_b)
        ):
            return False
        a, b = self.backend.get_entity(entity_a), self.backend.get_entity(entity_b)
        if entity_a != entity_b and not (
            merge_pair(self.backend, a, b) if is_topic(a) and is_topic(b)
            else self.backend.merge_entities(entity_a, entity_b)
        ):
            return False
        self.backend.set_proposal_status(proposal_id, "confirmed", reason="confirmed by you")
        return True

    def reject_merge(
        self, proposal_id: str, *, owner_prefix: str | None = None
    ) -> bool:
        """User says: these are different entities. They stay separate for good,
        and the pair says a person decided it ("kept apart by you"), not the
        answer it held: a judge's answer on a pair with the owner while it had
        no name is opened again (``_settle_owner_pairs``), a person's is not."""
        proposal = self.backend.get_proposal(proposal_id)
        if not _owned(proposal, owner_prefix) or proposal.status != "proposed":
            return False
        self.backend.set_proposal_status(proposal_id, "rejected", reason="kept apart by you")
        return True

    def entity_junk(
        self, *, user_id: str | None = None, judge: bool = False, exact_user: bool = False,
    ) -> dict[str, Any]:
        """Entities that should never have been entities.

        Two tiers, matching how certain we can be:
        - ``mechanical``: bare dates, amounts, URLs, salutations, placeholders.
          Decidable without a model; the maintenance pass removes these itself.
        - ``judged`` (only when ``judge=True``, costs one LLM call per ~60
          names): style instructions, task descriptions and fragments that only
          a reader can tell apart from real niche terms like a tax rule or an
          index name. Never removed automatically - the user confirms.
        """
        entities = self.entities(user_id=user_id, limit=100_000, exact_user=exact_user)
        mechanical = []
        candidates = []
        for entity in entities:
            reason = non_referent_reason(entity.name)
            if reason:
                mechanical.append({"id": entity.id, "name": entity.name,
                                   "reason": reason})
            elif (entity.entity_type or "concept") in ("concept", "other", "event"):
                candidates.append(entity)
        judged: list[dict[str, Any]] = []
        if judge and self.llm.available and candidates:
            by_name = {e.name: e for e in candidates}
            names = list(by_name)
            for start in range(0, len(names), 60):
                for name in judge_entity_referents(self.llm, names[start:start + 60]):
                    judged.append({"id": by_name[name].id, "name": name,
                                   "reason": "not a referent (AI review)"})
        return {"mechanical": mechanical, "judged": judged,
                "reviewable": len(candidates)}

    def remove_entities(
        self,
        entity_ids: list[str],
        *,
        owner_prefix: str | None = None,
        reason: str = "removed by you",
    ) -> int:
        """Retire the listed entities. Their memories are untouched.

        Retired, not deleted: the name lands in Upkeep > Archive with its
        mentions, aliases and relations kept, so a removal made in error - by
        the user or by an automatic pass - can be taken back. Its memories'
        property vectors read its names as names again at once (``_retire``).
        """
        removed = 0
        for entity_id in entity_ids:
            entity = self.backend.get_entity(entity_id)
            # A tag is removed from its memories on the tag page (delete_tag);
            # retiring its entity alone would leave the memories filed under it.
            if entity is not None and entity.entity_type == TOPIC_TYPE:
                continue
            if _owned(entity, owner_prefix):
                removed += int(self._retire(entity_id, reason))
        return removed

    def retired_entities(
        self, *, user_id: str | None = None, limit: int = 200
    ) -> list[dict[str, Any]]:
        """Names that were removed and can still be brought back."""
        return self.backend.list_retired_entities(Scope(user_id=user_id), limit=limit)

    def restore_entities(
        self, entity_ids: list[str], *, owner_prefix: str | None = None
    ) -> int:
        """Bring retired entities back, with the evidence that still exists
        (``MemoryBackend.restore_entity``), each then met as a save meets a
        name the store has (``_meet_namesakes``)."""
        owners = {
            row["entity_id"]: row.get("user_id")
            for row in self.backend.list_retired_entities(Scope(), limit=1_000_000)
        }
        restored = 0
        for entity_id in entity_ids:
            if entity_id not in owners:
                continue
            if not _owned(_Owner(owners[entity_id]), owner_prefix):
                continue
            if self.backend.restore_entity(entity_id):
                restored += 1
                self._meet_namesakes(entity_id)
        return restored

    def _meet_namesakes(self, entity_id: str) -> None:
        """A restored entity and the entities that took its name while it was
        gone (a save named it, found no entity and made one), treated as at
        save: with a calibrated judge each is a pair the funnel compares, at
        once when the judge is quick enough to ask inside a save
        (``Decider.rejudges_on_new_evidence``) and otherwise in the weekly
        pass; without one, two of one name are joined by rule
        (``entities.resolve_open_proposals``). Left alone, the two stood
        side by side with no pair between them. A pair the snapshot brought
        back, kept apart by a person, stays apart. A failure never fails the
        restore: the weekly pass pairs them."""
        entity = self.backend.get_entity(entity_id)
        if (entity is None or entity.entity_type == TOPIC_TYPE or not entity.normalized
                or unnamed_owner(entity)):
            return
        scope = Scope(user_id=entity.user_id)
        try:
            pairs: set[str] = set()
            for other in self.backend.find_entity_candidates(entity.normalized, scope):
                if other.id == entity.id or unnamed_owner(other):
                    continue
                proposal = self.backend.find_proposal(entity.id, other.id)
                if proposal is None:
                    # the restored one first: a merge the judge decides keeps it
                    proposal = self.backend.add_proposal(MergeProposal(
                        entity_a=entity.id, entity_b=other.id, user_id=entity.user_id,
                        confidence=0.5, reason="not yet compared"))
                if proposal.status == "proposed":
                    pairs.add(proposal.id)
            # a judge too slow to ask inside a save compares them weekly
            slow = judges_pairs(self.decider) and not self.decider.rejudges_on_new_evidence
            if pairs and not slow:
                resolve_open_proposals(backend=self.backend, decider=self.decider,
                                       scope=scope, proposal_ids=pairs)
        except Exception as exc:  # a provider hiccup must not fail a restore
            log.warning("a restored name was not met with its namesakes: %s", exc)

    def merges(self, *, user_id: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        """Merges of two entities that can be undone, newest first, with what
        decided each."""
        return self.backend.list_merges(Scope(user_id=user_id), limit=limit)

    def undo_merge(
        self, entity_id: str, *, owner_prefix: str | None = None
    ) -> dict[str, Any]:
        """Undo the merge of ``entity_id`` into another entity: both are as
        they were before it, and their pair is kept apart ("undone by you"),
        so no pass merges them again on the same evidence
        (``MemoryBackend.undo_merge``, which also says where memories saved
        since go). Refused, with the reason, when no merge of it is on record
        or the kept entity was merged into another since: that merge is
        undone first."""
        record = self.backend.merge_record(entity_id)
        if record is None or not _owned(_Owner(record["user_id"]), owner_prefix):
            return {"undone": False, "reason": "no merge of this entity is on record"}
        keep = self.backend.get_entity(record["keep_id"])
        merged = self.backend.get_entity(entity_id)
        if keep is None:
            return {"undone": False,
                    "reason": "the entity it was merged into was removed: restore that first"}
        if keep.merged_into is not None:
            later = self.backend.get_entity(self.backend.resolve_entity_id(keep.id) or "")
            return {"undone": False, "reason": (
                f'"{keep.name}" was merged into "{later.name if later else keep.merged_into}" '
                "since: undo that merge first")}
        if merged is None or merged.merged_into != keep.id:
            return {"undone": False, "reason": "it is no longer merged into that entity"}
        if not self.backend.undo_merge(entity_id):
            return {"undone": False, "reason": "it changed while being undone: try again"}
        if (merged.metadata or {}).get("owner"):
            # the store owner was merged away: it is the owner again
            self._upkeep_set("owner_entity", merged.user_id, entity_id)
        return {"undone": True, "entity_id": entity_id, "keep_id": keep.id}

    def remove_entity_preserving_tag(
        self, entity_id: str, *, owner_prefix: str | None = None
    ) -> dict[str, Any]:
        """Retire one mistaken entity, retaining repeated evidence as a tag."""
        entity = self.backend.get_entity(entity_id)
        if not _owned(entity, owner_prefix):
            return {"removed": 0, "tagged": 0, "tag": None}
        if entity.entity_type == TOPIC_TYPE:
            # already a tag, and nothing but one: there is nothing to remove
            return {"removed": 0, "tagged": 0, "tag": entity.name}

        memories = [
            memory
            for memory in self.backend.entity_memories(entity_id, limit=1_000_000)
            if _owned(memory, owner_prefix)
        ]
        tag = entity.name.strip().lower()
        preserve = bool(tag) and len(memories) > 1
        tagged = 0
        if preserve:
            for memory in memories:
                categories = list(memory.categories or [])
                normalized = {str(value).strip().lower() for value in categories}
                if tag not in normalized:
                    categories.append(tag)
                    if self.backend.update_memory(
                        memory.id, categories=categories, touch=False
                    ) is None:
                        continue
                tagged += 1

        removed = int(self._retire(
            entity_id,
            "removed by you, name kept as a tag" if preserve else "removed by you",
        ))
        return {
            "removed": removed,
            "tagged": tagged if removed else 0,
            "tag": tag if removed and preserve else None,
        }

    def merge_entities(
        self, keep_id: str, merge_id: str, *, owner_prefix: str | None = None
    ) -> bool:
        """Idempotent direct merge outside of a proposal.

        Two tags merge as tags, so the memories of the one folded in are
        filed under the one kept (``identity.fold_topic``: by id, whatever the
        kept topic's stored name). A tag and a named thing fold into the thing
        (``MemoryBackend.merge_entities``)."""
        keep_root = self.backend.resolve_entity_id(keep_id)
        merge_root = self.backend.resolve_entity_id(merge_id)
        if keep_root is None or merge_root is None:
            return False
        if owner_prefix is not None and not all(
            _owned(self.backend.get_entity(entity_id), owner_prefix)
            for entity_id in (keep_root, merge_root)
        ):
            return False
        keep, other = self.backend.get_entity(keep_root), self.backend.get_entity(merge_root)
        if (
            keep is not None and other is not None and keep_root != merge_root
            and keep.entity_type == TOPIC_TYPE and other.entity_type == TOPIC_TYPE
        ):
            merged = fold_topic(self.backend, keep, other)
        else:
            merged = self.backend.merge_entities(keep_root, merge_root)
        if merged and keep_root != merge_root:
            self._record_merged_by_you(keep_root, merge_root)
        return merged

    def _record_merged_by_you(self, keep_id: str, merge_id: str) -> None:
        """Record on the pair of the two that a person merged them, where the
        merge history reads what decided a merge: on their pair, open or
        decided before, or on a new one."""
        proposal = self.backend.find_proposal(keep_id, merge_id)
        if proposal is not None:
            self.backend.set_proposal_status(proposal.id, "confirmed", reason="merged by you")
            return
        entity = self.backend.get_entity(keep_id)
        self.backend.add_proposal(MergeProposal(
            entity_a=keep_id, entity_b=merge_id, user_id=entity.user_id if entity else None,
            status="confirmed", confidence=1.0, reason="merged by you", decided_at=utcnow()))

    # -- the store owner ----------------------------------------------------
    #: Statements of the established name, and of other names, kept per
    #: namespace with the owner's upkeep state (``learn_owner_name``).
    OWNER_EVIDENCE_KEPT = 10
    OWNER_CONFLICTS_KEPT = 20

    def set_owner_name(self, user_id: str | None, name: str) -> None:
        """Record the name of the person a namespace belongs to, from their
        account. The owner entity starts with it, and an owner entity made
        before the account named it, still called "the user", takes it now
        (``_name_owner``: the person who carries exactly that name, else the
        name itself). An account's name wins over a stated one
        (``learn_owner_name``). The identity judge may later find a named
        owner to be a named person in the store, whose name it keeps.
        """
        name = " ".join(str(name or "").split())[:80]
        if not name:
            return
        if self._upkeep_get("owner_name", user_id, None) != name:
            self._upkeep_set("owner_name", user_id, name)
        owner = self.owner_entity(user_id)
        if unnamed_owner(owner) and clean_stated_name(name):
            outcome = self._name_owner(user_id, owner, name, f'the account is named "{name}"',
                                       short_names=False)
            self._upkeep_set("owner_stated", user_id, {
                **(self._upkeep_get("owner_stated", user_id, None) or {}), "outcome": outcome})

    def owner_entity(self, user_id: str | None) -> Entity | None:
        """The entity of the person this namespace belongs to, once a memory
        has mentioned them. Followed through merges: the entity the owner was
        merged into is the owner now."""
        pointer = self._upkeep_get("owner_entity", user_id, None)
        root = self.backend.resolve_entity_id(pointer) if pointer else None
        entity = self.backend.get_entity(root) if root else None
        if entity is None:
            return None
        if root != pointer:
            self._upkeep_set("owner_entity", user_id, root)
        if not (entity.metadata or {}).get("owner"):
            metadata = {**(entity.metadata or {}), "owner": True}
            self.backend.set_entity_metadata(entity.id, metadata)
            entity = entity.model_copy(update={"metadata": metadata})
        return entity

    def owner_name(
        self, user_id: str | None, messages: list[dict[str, str]] | None = None
    ) -> str | None:
        """The name the extractor lists the owner of ``messages`` under: the
        owner entity's, else the account's, else the one stated
        (``learn_owner_name``), else "the user". Without a real name, "the
        user" is the owner only of a conversation with the user
        (``speaks_with_the_user``; no ``messages`` asks about one): where the
        speakers are named it would be one of them, so there is none (None)."""
        entity = self.owner_entity(user_id)
        if entity is not None:
            name = entity.name
        else:
            name = (self._upkeep_get("owner_name", user_id, None)
                    or (self._upkeep_get("owner_stated", user_id, None) or {}).get("name"))
        if name and name.strip().casefold() != OWNER_PLACEHOLDER:
            return name
        if messages is not None and not speaks_with_the_user(messages):
            return None
        return name or OWNER_PLACEHOLDER

    def _owner_for(
        self, scope: Scope, surfaces: list[str],
        messages: list[dict[str, str]] | None = None,
    ) -> Entity | None:
        """The owner entity, created when these extracted names, from
        ``messages``, first include the owner's (``owner_name``). It belongs to
        the whole namespace, not to one run."""
        owner = self.owner_entity(scope.user_id)
        if owner is not None:
            return owner
        name = self.owner_name(scope.user_id, messages)
        if name is None or not any(
                str(s).strip().casefold() == name.casefold() for s in surfaces):
            return None
        return self._new_owner(scope.user_id, name)

    def _owner_unnamed(self, user_id: str | None) -> bool:
        """Whether the owner still has no name: called "the user", with no
        account name. Only then does extraction ask who the user is."""
        account = self._upkeep_get("owner_name", user_id, None)
        return (self.owner_name(user_id) == OWNER_PLACEHOLDER
                and not (account and clean_stated_name(account)))

    def _new_owner(self, user_id: str | None, name: str) -> Entity:
        owner = self.backend.insert_entity(Entity(
            name=name, normalized=name.lower(), entity_type="person",
            user_id=user_id, metadata={"owner": True},
        ))
        self._upkeep_set("owner_entity", user_id, owner.id)
        return owner

    def learn_owner_name(
        self, user_id: str | None, name: str, *,
        evidence: dict[str, Any] | None = None, person_id: str | None = None,
    ) -> dict[str, Any]:
        """Learn that the owner of a namespace was stated to be ``name``: the
        user gave it, signed with it, was called by it, or a memory says it.
        ``evidence`` is what stated it (its ``text``, ``memory_ids`` and
        ``episode_ids``), kept with the name in the owner's upkeep state
        (``owner_stated``). ``person_id`` is the person the statements were
        read to be (``learn_owner``), in place of the name's matching rule.

        Who the owner is, is stated, not judged (``identity.unnamed_owner``).
        While the owner is called "the user", or has no entity yet, it takes
        the stated identity (``_name_owner``): folded into the person who
        carries the name, through the merge any pair goes through, so the
        person keeps their name and becomes the owner, "the user" one of their
        merged names, recorded with the statement as its reason and undone
        under Archive > Merged names like any merge. A judge's "different" on
        that pair does not stand in the way: it answered a question it should
        not have been asked. A person's does ("kept apart by you", "undone by
        you"): the owner then takes the stated name itself. With no person of
        that name, the owner is renamed to it, "the user" kept as an alias.
        From then on the extractor lists the owner under its real name.

        One name holds, so statements cannot flip it back and forth. The
        first name stated is kept; a later different name is recorded as a
        conflict and changes nothing, unless it is a correction
        (``owner.is_correction``: it names the first name and says it was
        wrong) and the owner still carries the first name itself, not the
        name of a person it was folded into. An account's name wins
        (``set_owner_name``): with one, statements are only recorded.
        Returns what was done ("folded", "renamed", "recorded", "kept",
        "conflict" or "none") with why."""
        name = clean_stated_name(name)
        if not name or fold_name(name) == OWNER_PLACEHOLDER:
            return {"action": "none", "reason": "no name was stated"}
        state = dict(self._upkeep_get("owner_stated", user_id, None) or {})
        statement = {"name": name, "at": utcnow(), **(evidence or {})}
        statement["text"] = str(statement.get("text") or "")[:STATEMENT_CHARS]

        def keep(outcome: dict[str, Any], *, conflict: bool = False) -> dict[str, Any]:
            if conflict:
                state["conflicts"] = ([*state.get("conflicts", []),
                                       {**statement, "why": outcome["reason"]}]
                                      [-self.OWNER_CONFLICTS_KEPT:])
            else:
                known = {(e.get("text"), e.get("source")) for e in state.get("evidence", [])}
                if (statement["text"], statement.get("source")) not in known:
                    state["evidence"] = ([*state.get("evidence", []), statement]
                                         [-self.OWNER_EVIDENCE_KEPT:])
            if outcome["action"] in ("folded", "renamed"):
                state["outcome"] = outcome
            self._upkeep_set("owner_stated", user_id, state)
            return outcome

        account = self._upkeep_get("owner_name", user_id, None)
        if account and clean_stated_name(account) and fold_name(account) != OWNER_PLACEHOLDER:
            same = fold_name(account) == fold_name(name)
            return keep({"action": "recorded",
                         "reason": f'the account\'s name, "{account}", wins'}, conflict=not same)
        owner = self.owner_entity(user_id)
        established = state.get("name")
        corrected = False
        if established and fold_name(established) != fold_name(name):
            if not is_correction(statement["text"], established, name):
                return keep({"action": "conflict", "reason": (
                    f'"{established}" was stated first and stays')}, conflict=True)
            outcome = state.get("outcome") or {}
            folded = (owner is not None and outcome.get("action") == "folded"
                      and outcome.get("entity_id") == owner.id)
            if owner is not None and not unnamed_owner(owner) and (
                    folded or fold_name(owner.name) != fold_name(established)):
                return keep({"action": "conflict", "reason": (
                    f'a correction of "{established}", but the owner was folded into '
                    f'"{owner.name}": undo that merge under Archive > Merged names to change it')},
                    conflict=True)
            corrected = True
        state["name"] = name
        if owner is not None and not unnamed_owner(owner) and not corrected:
            return keep({"action": "kept", "reason": f'the owner is "{owner.name}"'})
        reason = (f'the owner\'s name was stated: "{statement["text"][:120]}"'
                  if statement["text"] else f'the owner\'s name was stated as "{name}"')
        return keep(self._name_owner(user_id, owner, name, reason, person_id=person_id))

    def _owner_people(
        self, user_id: str | None, owner: Entity | None
    ) -> list[dict[str, Any]]:
        """The namespace's people besides the owner, each with its names and
        aliases and how many memories it is on, most first."""
        people = [e for e in self.backend.list_entities(
            Scope(user_id=user_id, exact_user=True), limit=1_000_000)
            if e.entity_type == "person" and e.merged_into is None
            and not (e.metadata or {}).get("owner") and (owner is None or e.id != owner.id)]
        counts = self.backend.entity_memory_counts([e.id for e in people])
        rows = [{"id": e.id, "name": e.name, "aliases": self.backend.entity_aliases(e.id),
                 "memories": counts.get(e.id, 0)} for e in people]
        return sorted(rows, key=lambda row: (-row["memories"], fold_name(row["name"])))

    def _kept_apart_by_a_person(self, entity_a: str, entity_b: str) -> bool:
        proposal = self.backend.find_proposal(entity_a, entity_b)
        return (proposal is not None and proposal.status == "rejected"
                and decided_by_a_person(proposal.reason))

    def _owner_identity_plan(
        self, user_id: str | None, owner: Entity | None, name: str, *,
        person_id: str | None = None, short_names: bool = True,
    ) -> tuple[dict[str, Any] | None, str]:
        """The person the owner is to be folded into, or None, and why."""
        people = self._owner_people(user_id, owner)
        if person_id is None:
            person_id = person_for(name, [(p["id"], p["aliases"]) for p in people],
                                   short_names=short_names)
        person = next((p for p in people if p["id"] == person_id), None)
        if person is None:
            return None, f'no one person carries the name "{name}"'
        if owner is not None and self._kept_apart_by_a_person(owner.id, person["id"]):
            return None, f'"{person["name"]}" was kept apart from the owner by a person'
        return person, f'"{person["name"]}" carries the name "{name}"'

    def _name_owner(
        self, user_id: str | None, owner: Entity | None, name: str, reason: str, *,
        person_id: str | None = None, short_names: bool = True,
    ) -> dict[str, Any]:
        """Give the owner the identity ``name`` states (``learn_owner_name``):
        fold it into the person who carries the name, else rename it, else,
        with no owner entity yet, keep the name for when one is made."""
        person, why = self._owner_identity_plan(user_id, owner, name, person_id=person_id,
                                                short_names=short_names)
        if person is not None:
            was = owner.name if owner is not None else OWNER_PLACEHOLDER
            # no owner entity yet: one is made to be folded, so the merge is
            # recorded and can be undone like any other
            owner = owner or self._new_owner(user_id, OWNER_PLACEHOLDER)
            if self._fold_owner(owner, person["id"], reason):
                return {"action": "folded", "owner": was, "into": person["name"],
                        "entity_id": person["id"], "reason": reason}
            why = f'the merge into "{person["name"]}" was refused'
        if owner is None:
            return {"action": "recorded", "name": name, "reason": (
                "no owner entity yet: the extractor lists the owner under this name")}
        if fold_name(owner.name) != fold_name(name):
            renamed = self.backend.rename_entity(owner.id, name)
            if renamed is not None:
                return {"action": "renamed", "owner": owner.name, "to": renamed.name,
                        "entity_id": owner.id, "reason": f"{reason}; {why}"}
        return {"action": "kept", "reason": f'the owner is "{owner.name}"'}

    def _fold_owner(self, owner: Entity, person_id: str, reason: str) -> bool:
        """Fold the owner into the person (``identity.merge_pair``: the person
        keeps their name and becomes the owner) and record on their pair what
        decided it, where Archive > Merged names reads it."""
        person = self.backend.get_entity(person_id)
        if person is None or not merge_pair(self.backend, owner, person):
            return False
        proposal = self.backend.find_proposal(owner.id, person.id)
        if proposal is None:
            self.backend.add_proposal(MergeProposal(
                entity_a=person.id, entity_b=owner.id, user_id=owner.user_id,
                status="confirmed", confidence=1.0, reason=reason, decided_at=utcnow()))
        else:
            self.backend.set_proposal_status(proposal.id, "confirmed", reason=reason)
        return True

    def _learn_from_save(
        self, scope: Scope, messages: list[dict[str, str]], stated: list[str],
        line_episodes: list[list[str]], actions: list[AddAction],
    ) -> None:
        """What a save states about who the user is (``learn_owner_name``):
        the name the extractor reported (``extraction.stated_user_name``),
        for messages with a turn in role user, and the one speaker's name the
        turns in role user carry. ``line_episodes`` are the episodes of each
        message that says something, in order. Never fails the save."""
        try:
            said = [m for m in messages if _says_something(m)]
            lines = list(zip(said, line_episodes))

            def in_role_user(message: dict[str, str]) -> bool:
                return str(message.get("role", "user")).strip().casefold() == "user"

            found: list[tuple[str, dict[str, Any]]] = []
            signed = {speaker_name(m) for m in said if in_role_user(m) and speaker_name(m)}
            if len(signed) == 1:
                [signer] = signed
                found.append((signer, {
                    "text": f"turns in role user are signed {signer}", "source": "turn",
                    "episode_ids": [e for m, es in lines
                                    if in_role_user(m) and speaker_name(m) == signer
                                    for e in es][:5]}))
            if stated and any(in_role_user(m) for m in said):
                name = stated[0]

                def names_it(text: str) -> bool:
                    return fold_name(name) in fold_name(text)

                memories = [a for a in actions
                            if a.memory_id and a.content and names_it(a.content)]
                turns = [(m, es) for m, es in lines if names_it(str(m.get("content") or ""))]
                text = (memories[0].content if memories
                        else str(turns[0][0].get("content") or "") if turns else "")
                found.append((name, {
                    "text": text, "source": "save",
                    "memory_ids": list(dict.fromkeys(a.memory_id for a in memories))[:5],
                    "episode_ids": [e for _, es in turns for e in es][:5]}))
            for name, evidence in found:
                outcome = self.learn_owner_name(scope.user_id, name, evidence=evidence)
                if outcome["action"] in ("folded", "renamed", "conflict"):
                    log.info("the owner's name was stated as %r: %s", name, outcome)
        except Exception as exc:  # learning who the owner is must not fail a save
            log.warning("a stated name of the owner was not learned: %s", exc)

    def _owner_statements(self, user_id: str | None) -> list[Statement]:
        """What the namespace has stored and saved that may state who the
        user is (``owner.statements_in``): memories in use and forgotten
        (a forgotten duplicate still states the fact), and saved turns, a turn
        in role user that carries a speaker's name among them. Those that say
        the name outright first, then the most recent."""
        scope = Scope(user_id=user_id, exact_user=True)
        out: list[Statement] = []
        seen: set[str] = set()

        def add(statement: Statement) -> None:
            key = fold_name(statement.text)
            if key and key not in seen:
                seen.add(key)
                out.append(statement)

        for memory in self.backend.list_memories(scope, include_invalid=True, limit=10**9):
            for name, strong in statements_in(memory.content, role="memory"):
                add(Statement(text=memory.content[:STATEMENT_CHARS], name=name, strong=strong,
                              memory_id=memory.id, forgotten=memory.invalid_at is not None))
                break
        signed: Counter[str] = Counter()
        signed_at: dict[str, str] = {}
        for episode in self.backend.list_episodes(scope, limit=10**9):
            role = str(episode.role or "user").strip().casefold()
            if role == "user" and episode.name:
                signed[episode.name] += 1
                signed_at.setdefault(episode.name, episode.id)
            for name, strong in statements_in(episode.content, role=role):
                speaker = f"{episode.name} ({role})" if episode.name else role
                add(Statement(text=f"{speaker}: {episode.content}"[:STATEMENT_CHARS],
                              name=name, strong=strong, episode_id=episode.id))
                break
        for signer, count in signed.most_common():
            add(Statement(text=f"{count} saved turns in role user are signed {signer}",
                          name=signer, strong=len(signed) == 1,
                          episode_id=signed_at[signer]))
        return sorted(out, key=lambda s: not s.strong)

    def learn_owner(self, *, user_id: str | None = None, dry_run: bool = False) -> dict[str, Any]:
        """Learn who the owner of a namespace is from what it already holds,
        for stores saved before extraction reported a stated name: once a
        namespace, by the upkeep cycle while the owner is called "the user",
        and by ``memry learn-owner``.

        Cheap first: patterns pick out the memories (in use and forgotten)
        and saved turns that may state the user's name
        (``_owner_statements``); with none, nothing is asked. Otherwise one
        text-model call reads them against the namespace's people, with
        their aliases and memory counts, and says which person the
        statements say the owner is, or none, and the name they give
        (``owner.choose_owner``). Without a text model, or without an answer,
        only statements that say the name outright count, by the matching
        rule (``owner.rule_choice``). Then the owner takes that identity as
        any stated one (``learn_owner_name``). ``dry_run`` asks the same and
        writes nothing: the report shows the evidence, the person chosen and
        what would be folded or renamed."""
        owner = self.owner_entity(user_id)
        current = self.owner_name(user_id)
        report: dict[str, Any] = {"user": user_id, "owner": current, "dry_run": dry_run,
                                  "evidence": [], "decision": None, "action": None}

        def done(action: dict[str, Any]) -> dict[str, Any]:
            report["action"] = action
            if not dry_run:
                self._upkeep_set("owner_learned", user_id, {
                    "at": utcnow(), "action": action.get("action")})
            return report

        if current != OWNER_PLACEHOLDER:
            return done({"action": "none", "reason": f'the owner has a name: "{current}"'})
        statements = self._owner_statements(user_id)
        report["evidence"] = [s.as_dict() for s in statements]
        if not statements:
            return done({"action": "none", "reason": "nothing states who the user is"})
        people = self._owner_people(user_id, owner)
        if owner is not None:
            report["owner_memories"] = self.backend.count_entity_memories(owner.id)
        decision = None
        if self.llm.available and people:
            try:
                decision = choose_owner(self.llm, statements, people)
                if decision is not None:
                    decision["by"] = "the text model"
            except Exception as exc:  # an outage leaves the rule
                log.warning("the text model did not read who the owner is: %s", exc)
        if decision is None:
            decision = rule_choice(statements, people)
            if decision is not None:
                decision["by"] = "the matching rule"
        report["decision"] = decision
        name = (decision or {}).get("name") or (decision or {}).get("person")
        if not name:
            return done({"action": "none",
                         "reason": "the statements do not say who the user is"})
        chosen = decision.get("person_id")
        said = [fold_name(n) for n in (name, decision.get("person")) if n]
        backing = [s for s in statements if any(n in fold_name(s.text) for n in said)]
        backing = backing or statements
        evidence = {
            "text": backing[0].text, "source": "learn-owner",
            "memory_ids": [s.memory_id for s in backing if s.memory_id][:5],
            "episode_ids": [s.episode_id for s in backing if s.episode_id][:5],
        }
        if dry_run:
            person, why = self._owner_identity_plan(user_id, owner, name, person_id=chosen)
            if person is not None:
                report["action"] = {
                    "action": "would fold",
                    "owner": owner.name if owner is not None else OWNER_PLACEHOLDER,
                    "owner_memories": report.get("owner_memories", 0),
                    "into": person["name"], "into_memories": person["memories"],
                    "reason": why}
            elif owner is not None:
                report["action"] = {"action": "would rename", "owner": owner.name,
                                    "to": name, "reason": why}
            else:
                report["action"] = {"action": "would record", "name": name, "reason": why}
            return report
        return done(self.learn_owner_name(user_id, name, evidence=evidence, person_id=chosen))

    def _settle_owner_pairs(self) -> None:
        """Once per database (marker ``schema:owner-pairs:v1``): the judge's
        answers on pairs of an owner still called "the user" with a person no
        longer keep the two apart. The judge should never have been asked
        (``identity.unnamed_owner``): such a pair rejected, or left open with
        an answer, by the judge (its reason starts with the judge's name, as
        "jev: different") or by the names-alone screen, is open again as
        never compared, and waits until the owner has a name. A pair a person
        decided says so ("by you") and stays. A rejection a person made from
        the Upkeep list before it said so kept the judge's reason, and cannot
        be told apart: it is opened again too, which the owner of the store
        this was found on asked for. Each pair opened is listed under the
        marker as it was, so it can be put back."""
        marker = "schema:owner-pairs:v1"
        if self.backend.get_meta(marker):
            return
        reopened: list[dict[str, Any]] = []
        for key in self.backend.meta_items("upkeep:owner_entity:"):
            user_id = key[len("upkeep:owner_entity:"):] or None
            owner = self.owner_entity(user_id)
            if not unnamed_owner(owner):
                continue
            for proposal in self.backend.list_proposals(
                    Scope(user_id=user_id), status=None, limit=1_000_000):
                if proposal.status == "confirmed" or decided_by_a_person(proposal.reason):
                    continue
                if (proposal.status == "proposed" and proposal.different is None
                        and not proposal.compared_step):
                    continue  # never answered: nothing to void
                ends = {self.backend.resolve_entity_id(proposal.entity_a),
                        self.backend.resolve_entity_id(proposal.entity_b)}
                if owner.id not in ends or None in ends or len(ends) != 2:
                    continue
                [other_id] = ends - {owner.id}
                other = self.backend.get_entity(other_id)
                if other is None or other.entity_type != "person":
                    continue
                if self.backend.reopen_proposal(proposal.id, OWNER_PAIR_REOPENED):
                    reopened.append(proposal.model_dump(mode="json"))
        self.backend.set_meta(marker, json.dumps({"at": utcnow(), "reopened": reopened}))
        if reopened:
            log.info("reopened %d pair(s) the judge decided for an owner without a name",
                     len(reopened))

    def resolve_entities(
        self, *, user_id: str | None = None, exact_user: bool = False,
    ) -> dict[str, int]:
        """Re-judge open proposals with accumulated evidence; auto-confirm only
        clear, high-confidence matches. Everything ambiguous stays proposed.

        Then drop entities nothing references. Extraction inevitably produces
        some records that never attach to anything, and without this they
        accumulate forever: a real store reached 206 such rows out of 519.
        ``exact_user`` as in ``get_all``.
        """
        scope = Scope(user_id=user_id, exact_user=exact_user)
        self._carry_tag_decisions(user_id)
        # Surface duplicates first: proposals are otherwise only made at write
        # time, so anything already duplicated has nothing scheduled to look at
        # it again and would sit there for good. Names close in meaning only
        # count with a semantic embedder: hash vectors put "Köln" nowhere near
        # "Cologne".
        semantic = self.embedder.dimensions and self.embedder.name != "hash"
        proposed = propose_same_name_duplicates(
            backend=self.backend, scope=scope, decider=self.decider,
            embed=self.embedder.embed if semantic else None,
            limit=300 if self.decider.calibrated and self.decider.available else 50,
            owner=self.owner_entity(user_id),
        )
        outcome = resolve_open_proposals(
            backend=self.backend, decider=self.decider, scope=scope
        )
        outcome["proposed"] = proposed
        outcome["purged"] = self.backend.purge_orphan_entities(
            scope, reason="nothing referenced it"
        )
        # Mechanical non-referents ("2019", "$149", a URL) are removed without
        # review: no accumulation of evidence will ever make one a thing with an
        # identity. Judgement cases stay for the user under Upkeep.
        # Each carries the rule that caught it, so the Forgotten list can say
        # why a name went rather than only that it did.
        junk = self.entity_junk(user_id=user_id, exact_user=exact_user)["mechanical"]
        outcome["junk_removed"] = sum(
            self.remove_entities([item["id"]], reason=item["reason"])
            for item in junk
        )
        return outcome

    def _carry_tag_decisions(self, user_id: str | None) -> None:
        """Before tags were entity pairs, the tag question's funnel kept the
        step each pair of tags was compared at (upkeep ``tag_pairs``), and
        the Upkeep list of tags that looked like one subject kept the pairs
        a person kept apart (``tag_split:ignored``). Each becomes the pair's
        proposal, so no pair is asked again: compared at its step, or kept
        apart. A pair whose tag is gone is dropped, and so are both lists."""
        compared = self._upkeep_get("tag_pairs", user_id, None)
        ignored = self._upkeep_get("tag_split:ignored", user_id, None)
        if compared is None and ignored is None:
            return
        scope = Scope(user_id=user_id)
        pairs = [(key.split("\n"), int(step)) for key, step in (compared or {}).items()]
        pairs += [(list(pair), None) for pair in ignored or []]
        for names, step in pairs:
            topics = [self.backend.topic_entity(name, scope, create=False) for name in names]
            if len(topics) != 2 or None in topics or topics[0].id == topics[1].id:
                continue
            proposal = self.backend.find_proposal(topics[0].id, topics[1].id)
            if step is None:
                if proposal is None:
                    self.backend.add_proposal(MergeProposal(
                        entity_a=topics[0].id, entity_b=topics[1].id, user_id=user_id,
                        status="rejected", reason="kept apart by you", decided_at=utcnow()))
                elif proposal.status == "proposed":
                    self.backend.set_proposal_status(proposal.id, "rejected",
                                                     reason="kept apart by you")
            elif proposal is None:
                self.backend.add_proposal(MergeProposal(
                    entity_a=topics[0].id, entity_b=topics[1].id, user_id=user_id,
                    compared_step=step, reason="compared by the tag question"))
            elif proposal.status == "proposed" and proposal.compared_step < step:
                self.backend.update_proposal_judgement(
                    proposal.id, confidence=proposal.confidence, reason=proposal.reason,
                    compared_step=step)
        for name in ("tag_pairs", "tag_split:ignored", "tag_split:count"):
            self.backend.set_meta(_upkeep_key(name, user_id), "")

    # ------------------------------------------------------------------
    # tags
    # ------------------------------------------------------------------
    def tags_to_topics(
        self, *, user_id: str | None = None, all_users: bool = True, dry_run: bool = False
    ) -> list[dict[str, Any]]:
        """Give every tag of the legacy ``topics`` table its topic entity and
        every ``memory_topics`` link its mention, user by user; see
        ``LocalBackend.tags_to_topics``. Idempotent; ``dry_run`` only counts."""
        return self.backend.tags_to_topics(
            user_id=user_id, all_users=all_users, dry_run=dry_run)

    def merge_obvious_topics(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Collapse formatting and plural duplicates. Any other pair of tags
        is an entity pair: raised, compared and merged as one
        (``resolve_entities``), by the tag question where both are tags.

        Tags are topic entities of one user, so a merge is theirs whole:
        ``agent_id`` and ``run_id`` narrow which memories are counted, not
        which are rewritten. Each merge folds the variant's topic entity into
        the canonical one and rewrites the ``categories`` column
        (``_merge_topics``)."""
        scope = Scope(user_id=user_id, agent_id=agent_id, run_id=run_id)
        rows = self.backend.topic_mention_counts(scope, exact_user=True)
        groups = obvious_canonical_merges(rows if rows is not None else self.categories(
            user_id=user_id, agent_id=agent_id, run_id=run_id))
        changed = 0
        for group in groups:
            remove = set(group["variants"]) - {group["canonical"]}
            changed += self._merge_topics(
                user_id, remove, group["canonical"], exact_user=True) or 0
        return {"groups_merged": len(groups), "memories_changed": changed}

    def _merge_topics(
        self, user_id: str | None, remove: set[str], into: str | None, *,
        exact_user: bool = False,
    ) -> int:
        """Merge tags the way entities merge: each tag in ``remove`` that is an
        active topic entity of ``user_id`` is folded into the topic entity of
        ``into`` (``MemoryBackend.merge_entities``: its mentions move, its id
        redirects), then the ``categories`` column is rewritten
        (``retag_topics``), which files every rewritten memory's tags again.
        ``into`` None drops the tags. Returns how many memories changed, or
        None when the backend keeps no tag index.

        ``into`` is read as a column files it (``MemoryBackend.tag_filing``):
        a name merged away means its survivor, never a fresh topic of a
        retired name. After "tax" went into "levies", merging "taxes" into
        "tax" merges it into "levies"; after "bildy" the tag went into
        "Bildy" the product, merging "bildy app" into "bildy" folds it into
        the product. The names merged away resolve among the active topics
        only, never through a tombstone: a name merged away before names
        nothing to merge again ("taxes" into "tax", then "taxes" into
        "levies" leaves "tax" alone). When ``into`` names no topic at all it
        gets one of its own and the others fold into it, so every name merged
        away keeps its tombstone; one tag folded into a new name is a rename,
        and the new topic takes over the old one's description and metadata
        (``_carry_renamed_topic``)."""
        scope = Scope(user_id=user_id)
        wanted = {str(tag).strip().lower() for tag in remove if str(tag).strip()}
        target = str(into).strip().lower() if into and str(into).strip() else None
        if target is not None:
            target = self.backend.tag_filing([target], scope).get(target, target)
            variants = [
                other for other in (
                    self.backend.topic_entity(tag, scope, create=False)
                    for tag in sorted(wanted - {target})
                )
                if other is not None
            ]
            if variants:
                # the active topic of that name, or the named thing a tag of
                # that name went into
                keep = self.backend.topic_entity(target, scope, create=False,
                                                 follow_merged=True)
                fresh = keep is None
                if keep is None:
                    keep = self.backend.topic_entity(target, scope)
                for other in variants:
                    if keep is not None and other.id != keep.id:
                        self.backend.merge_entities(keep.id, other.id)
                if fresh and keep is not None and len(variants) == 1:
                    self._carry_renamed_topic(variants[0], keep)
        return self.backend.retag_topics(scope, wanted, target, exact_user=exact_user)

    def _carry_renamed_topic(self, old: Entity, new: Entity) -> None:
        """A renamed tag's topic entity (``new``, made for the new name) takes
        over what the old one (``old``, now a tombstone pointing to it) had:
        its description with the time it was written, since the tag files the
        same memories, and its metadata, with ``renamed_from`` naming the old
        id, so a reference stored under it can be followed."""
        self.backend.set_entity_metadata(new.id, {**old.metadata, "renamed_from": old.id})
        if old.description:
            self.backend.set_entity_description(
                new.id, old.description, old.description_updated_at)

    def consolidate_memories(
        self,
        *,
        user_id: str | None = None,
        threshold: float = 0.90,
        max_groups: int = 25,
        apply: bool = True,
        only: list[list[str]] | None = None,
        exclude: set[frozenset[str]] | None = None,
        exact_user: bool = False,
    ) -> dict[str, Any]:
        """Merge memories that record the same fact more than once.

        ``apply=False`` returns exactly what would happen without touching
        anything, which is what the dashboard shows before the user confirms.
        ``only`` restricts the run to the listed groups, so a user can accept
        some proposals from a preview and leave the rest alone. ``exclude``
        skips groups already judged, so the upkeep pass asks the model about
        each group once rather than every cycle.

        The merged text becomes a NEW memory carrying the union of the group's
        tags, entities and provenance, and every original is invalidated with
        ``superseded_by`` pointing at it. Rewriting one of the originals in
        place would have made an arbitrary member masquerade as the merge, with
        its own creation date and history; a distinct record says plainly that
        this text came from consolidating several. Nothing is destroyed, so the
        audit trail and time-travel still resolve. ``exact_user`` as in
        ``get_all``.
        """
        scope = Scope(user_id=user_id, exact_user=exact_user)
        wanted = {frozenset(group) for group in only} if only else None
        vectors = self.backend.memory_vectors(scope, limit=1_000_000)
        summary: dict[str, Any] = {
            "scanned": len(vectors), "groups": [], "merged": 0, "superseded": 0,
        }
        groups = similarity_groups(vectors, threshold=threshold)
        if not groups:
            return summary

        # densest first: the most obviously redundant families are worth the
        # LLM budget before a long tail of borderline pairs
        groups.sort(key=len, reverse=True)
        groups = [
            g for g in groups
            if (wanted is None or frozenset(g) in wanted)
            and not (exclude and frozenset(g) in exclude)
        ]
        for member_ids in groups[:max_groups]:
            memories = [m for m in (self.backend.get_memory(i) for i in member_ids)
                        if m is not None and m.invalid_at is None]
            if len(memories) < 2:
                continue
            normalized = {_normalized_content(m.content) for m in memories}
            if len(normalized) == 1:
                verdict = {"same_fact": True,
                           "content": representative(memories).content,
                           "reason": "identical text"}
            elif self.llm.available:
                verdict = judge_group(self.llm, memories, self.decider)
            else:
                continue  # never merge on similarity alone

            entry = {
                "memory_ids": [m.id for m in memories],
                "contents": [m.content for m in memories],
                "same_fact": bool(verdict["same_fact"]),
                "reason": verdict["reason"],
                "merged_content": verdict["content"],
            }
            summary["groups"].append(entry)
            if not verdict["same_fact"] or not apply:
                continue
            entry["survivor"] = self._merge_group(
                memories, verdict["content"], user_id=user_id
            )
            summary["merged"] += 1
            summary["superseded"] += len(memories)
        return summary

    def _merge_group(
        self, memories: list[Memory], content: str, *, user_id: str | None
    ) -> str:
        """Replace ``memories`` with one new memory saying ``content``.

        The originals are invalidated with ``superseded_by`` pointing at the
        survivor, never deleted. Returns the survivor's id.
        """
        oldest = min(memories, key=lambda m: m.created_at)
        merged = Memory(
            content=content,
            memory_type=oldest.memory_type,
            user_id=user_id,
            importance=max(m.importance for m in memories),
            categories=list(dict.fromkeys(
                [c for m in memories for c in (m.categories or [])])),
            entities=list(dict.fromkeys(
                [e for m in memories for e in (m.entities or [])])),
            # keep the earliest creation date: the fact is as old as the
            # first time it was recorded, not as old as the merge
            created_at=oldest.created_at,
            source_episode_ids=list(dict.fromkeys(
                [e for m in memories for e in (m.source_episode_ids or [])])),
            metadata={"consolidated_from": [m.id for m in memories]},
        )
        embedding = (
            self.embedder.embed([content])[0] if self.embedder.dimensions else None
        )
        if embedding:
            # vector search reads only vectors labelled with the current model
            merged.embedding_model = self.embedder.model_id
        stored = self.backend.insert_memory(merged, embedding=embedding)
        self._carry_mentions(stored.id, [m.id for m in memories])
        self.backend.add_event(MemoryEvent(
            memory_id=stored.id, event="ADD", new_content=content,
            reason=f"consolidated {len(memories)} duplicate memories",
        ))
        # every original is forgotten, including the one it reads most like
        for memory in memories:
            self.backend.invalidate_memory(memory.id, superseded_by=stored.id)
            self.backend.add_event(MemoryEvent(
                memory_id=memory.id, event="SUPERSEDE",
                old_content=memory.content, new_content=content,
                reason=f"consolidated into {stored.id}", kind="consolidation",
            ))
        return stored.id

    def _carry_mentions(self, memory_id: str, originals: list[str]) -> None:
        """The entities its originals mention, onto a consolidated memory, so
        it stays on those entities' pages and in their links."""
        seen: set[str] = set()
        for original in originals:
            for entity in self.backend.entities_of_memory(original):
                if entity.id not in seen:
                    seen.add(entity.id)
                    self.backend.add_mention(EntityMention(
                        entity_id=entity.id, memory_id=memory_id, surface=entity.name))

    def repair_consolidated(
        self, *, user_id: str | None = None, exact_user: bool = False,
    ) -> dict[str, int]:
        """Memories consolidated before the merge kept their vector's model and
        their originals' mentions: re-embed those stored without a model (the
        model that made them is unknown) and give back the mentions.
        ``exact_user`` as in ``get_all``."""
        scope = Scope(user_id=user_id, exact_user=exact_user)
        embedded = 0
        unlabelled = self.backend.unlabelled_vector_ids(scope)
        if unlabelled and self.embedder.dimensions:
            memories = [m for m in (self.backend.get_memory(i) for i in unlabelled) if m]
            for start in range(0, len(memories), 64):
                batch = memories[start:start + 64]
                for memory, vector in zip(batch, self.embedder.embed([m.content for m in batch])):
                    if vector:
                        self.backend.update_memory(
                            memory.id, embedding=vector,
                            embedding_model=self.embedder.model_id, touch=False)
                        embedded += 1
        mentioned = 0
        for memory in self.backend.consolidated_memories(scope):
            originals = (memory.metadata or {}).get("consolidated_from") or []
            if originals and not self.backend.entities_of_memory(memory.id):
                self._carry_mentions(memory.id, list(originals))
                mentioned += bool(self.backend.entities_of_memory(memory.id))
        return {"re_embedded": embedded, "mentions_restored": mentioned}

    # -- manual tag curation -----------------------------------------------
    def rename_tag(self, tag: str, to: str, *, user_id: str | None = None) -> int:
        """Rename one tag to another across every memory. Returns the count."""
        return self._retag(user_id, {tag.strip().lower()}, to.strip().lower())

    def merge_tags(self, tags: list[str], to: str, *, user_id: str | None = None) -> int:
        """Combine several tags into one across every memory."""
        remove = {t.strip().lower() for t in tags if t.strip()}
        return self._retag(user_id, remove, to.strip().lower())

    def delete_tag(self, tag: str, *, user_id: str | None = None) -> int:
        """Remove a tag from every memory (the memories stay). Its topic
        entity is retired with its last mention (``retag_topics``), so no
        active tag is left with nothing filed under it."""
        return self._retag(user_id, {tag.strip().lower()}, None)

    def _retag(
        self, user_id: str | None, remove: set[str], add: str | None, *,
        exact_user: bool = False,
    ) -> int:
        """Strip ``remove`` tags from matching memories and optionally add
        ``add``, preserving the other tags and their original casing. The
        tags' topic entities merge with it (``_merge_topics``).

        A topic entity belongs to one namespace, so the edit is made one
        namespace at a time: ``user_id``'s, or, with ``user_id`` None (an
        admin's edit, ``exact_user`` unset), every namespace that carries one
        of the tags (``MemoryBackend.tag_namespaces``), each in full: its
        entities, mentions, a renamed topic's description, its columns.
        """
        remove = {r for r in remove if r}
        if not remove:
            return 0
        if user_id is None and not exact_user:
            namespaces = self.backend.tag_namespaces(remove)
            if namespaces is not None:
                return sum(self._retag(namespace, remove, add, exact_user=True)
                           for namespace in namespaces)
        if add is not None and self.backend.topic_entity(
                add, Scope(user_id=user_id), create=False) is None:
            # A new name is a tag like any other, so it is held to the same
            # shape; one that cleans away to nothing is a plain removal. The
            # name of a tag that exists is taken as stored, even one a new
            # tag could not have (``identity.fold_topic``).
            add = next(iter(clean_tags(add)), None)
        indexed = self._merge_topics(user_id, remove, add, exact_user=exact_user)
        if indexed is not None:
            return indexed
        changed = 0
        for memory in self.get_all(
            user_id=user_id, categories=list(remove), limit=1_000_000
        ):
            cats = list(memory.categories or [])
            kept = [c for c in cats if str(c).strip().lower() not in remove]
            if add and add not in {str(c).strip().lower() for c in kept}:
                kept.append(add)
            if kept != cats:
                self.backend.update_memory(memory.id, categories=kept, touch=False)
                changed += 1
        return changed

    # -- maintenance switches ----------------------------------------------
    # The scheduler's passes were config-only, which meant turning one off was
    # an env-var edit and a restart. A runtime override lives in the meta table
    # so the dashboard toggle survives restarts; config stays the default when
    # no override was ever set.
    _MAINTENANCE_KEYS = (
        "dedup_entities", "durability", "consolidation", "structure",
    )
    #: The passes a config switch gates, with why one is off while its
    #: switch is: a stored toggle cannot turn such a pass on
    #: (``maintenance_enabled``), "run now" says why it did not run
    #: (``_pass_off_reason``) and the dashboard offers no toggle for it
    #: (``pass_allowed``).
    _CONFIG_GATES: dict[str, tuple[Callable[[Config], bool], str]] = {
        "durability": (lambda config: config.decay.durability,
                       "decay.durability is off (MEMRY_DURABILITY)"),
    }

    #: How many memories one durability pass scores. Jev answers 128 questions
    #: in a single call, so the batch is bounded by prudence, not by cost.
    DURABILITY_BATCH = 64

    def score_memory_durability(
        self, *, user_id: str | None = None, limit: int | None = None,
        exact_user: bool = False,
    ) -> dict[str, Any]:
        """Record how long each memory is worth keeping, for memories missing it.

        An estimate per fact (days, months or years) that nothing acts on yet:
        no memory is forgotten by age, and search does not read it. It is kept
        for a planned experiment on relevance per entity, where "the train was
        delayed this morning" and "allergic to penicillin" should not count
        alike.

        Off unless ``decay.durability`` is set, whoever asks (the scheduler,
        "run now", the REST route). The score is housekeeping: it is written
        without moving the memory's ``updated_at``, which drives recency.
        """
        outcome: dict[str, Any] = {"scored": 0, "skipped": 0, "provider": self.decider.name}
        if not self.pass_allowed("durability"):
            outcome["skipped"] = -1
            outcome["reason"] = self._pass_off_reason("durability")
            return outcome
        if not self.decider.available:
            outcome["skipped"] = -1
            log.info("durability: no decision provider configured, nothing scored")
            return outcome
        batch = limit or self.DURABILITY_BATCH
        pending = [
            m for m in self.get_all(user_id=user_id, limit=100_000, exact_user=exact_user)
            if DURABILITY_KEY not in (m.metadata or {})
        ][:batch]
        if not pending:
            return outcome
        started = time.time()
        scores = score_durability(self.decider, [m.content for m in pending])
        for index, memory in enumerate(pending):
            score = scores.get(index)
            if score is None:
                outcome["skipped"] += 1
                continue
            metadata = dict(memory.metadata or {})
            metadata[DURABILITY_KEY] = round(score, 3)
            self.backend.update_memory(memory.id, metadata=metadata, touch=False)
            outcome["scored"] += 1
        outcome["ms"] = round((time.time() - started) * 1000)
        log.info(
            "durability: scored %d of %d memories in %d ms via %s (%d unanswered)",
            outcome["scored"], len(pending), outcome["ms"], self.decider.name,
            outcome["skipped"],
        )
        return outcome

    def pass_allowed(self, key: str) -> bool:
        """Whether the config lets the pass run at all (``_CONFIG_GATES``): a
        pass no switch gates always may."""
        gate = self._CONFIG_GATES.get(key)
        return gate is None or bool(gate[0](self.config))

    def _pass_off_reason(self, key: str) -> str:
        """Why a pass that is off (``maintenance_enabled``) is off."""
        if not self.pass_allowed(key):
            return self._CONFIG_GATES[key][1]
        return "this pass is off; turn it on under Upkeep to run it"

    def maintenance_enabled(self, key: str) -> bool:
        if not self.pass_allowed(key):
            return False  # off by its config switch, whatever a toggle says
        override = self.backend.get_meta(f"maintenance:{key}:enabled")
        if override is not None:
            return override == "true"
        if key == "dedup_entities":
            return self.config.dedup_entities
        if key == "durability":
            return self.decider.available
        if key == "consolidation":
            return self.llm.available
        if key == "structure":
            return True
        return False

    def set_maintenance_enabled(self, key: str, enabled: bool) -> bool:
        if key not in self._MAINTENANCE_KEYS:
            return False
        self.backend.set_meta(
            f"maintenance:{key}:enabled", "true" if enabled else "false"
        )
        return True

    # ------------------------------------------------------------------
    # entity structure: hubs, homes and shared names (intelligence/structure.py)
    # ------------------------------------------------------------------
    #: Names screened per pass; one typed question each, asked in parallel.
    SCREEN_BATCH = 1000

    def _structure_inputs(self, user_id: str | None, exact_user: bool = False):
        scope = Scope(user_id=user_id, exact_user=exact_user)
        entities = self.backend.list_entities(scope, limit=1_000_000)
        links = self.backend.entity_memory_links(scope)
        relations = [
            r for r in self.backend.list_relations(scope, limit=1_000_000)
            if r.invalid_at is None
        ]
        memories = Counter(entity_id for entity_id, _ in links)
        involved: Counter[str] = Counter()
        for relation in relations:
            involved[relation.subject] += 1
            involved[relation.object] += 1
        nodes = [
            Node(
                id=e.id, name=e.name, normalized=e.normalized or e.name.strip().lower(),
                entity_type=e.entity_type, memories=memories[e.id],
                relations=involved[e.id], created_at=e.created_at,
            )
            for e in entities
        ]
        triples = [(r.subject, r.predicate, r.object) for r in relations]
        return entities, nodes, triples, self._judged_homes(scope)

    def _held_apart(self, scope: Scope) -> list[tuple[str, str]]:
        """Entity pairs the judge or a person held apart: a rejected proposal
        (the judge's "apart", or a person's "keep separate"), or one whose
        latest comparison gave P(different) at the judge's apart bar, which
        waits before ``identity.APART_STEP`` without being rejected. The rule
        ``entities.join_namesakes`` keeps, for the structure pass."""
        bar = self.decider.pair_apart_probability
        pairs: list[tuple[str, str]] = []
        for status in ("rejected", "proposed"):
            for proposal in self.backend.list_proposals(scope, status=status, limit=100_000):
                if status == "proposed" and (proposal.different is None
                                             or proposal.different < bar):
                    continue
                a = self.backend.resolve_entity_id(proposal.entity_a)
                b = self.backend.resolve_entity_id(proposal.entity_b)
                if a is not None and b is not None and a != b:
                    pairs.append((a, b))
        return pairs

    def _judged_homes(self, scope: Scope) -> list[tuple[str, str, float]]:
        """(child, parent, probability) for every compared pair the decision
        provider answered is a version or a part, at ``BELONGS_BAR`` or above.
        Kept on open and ruled-out pairs alike: a version is ruled out as the
        same thing and still belongs to its thing."""
        judged: list[tuple[str, str, float]] = []
        for status in ("proposed", "rejected"):
            for proposal in self.backend.list_proposals(scope, status=status, limit=100_000):
                side, probability = belonging(proposal.belongs)
                if side is None or probability < BELONGS_BAR:
                    continue
                a = self.backend.resolve_entity_id(proposal.entity_a)
                b = self.backend.resolve_entity_id(proposal.entity_b)
                if a is None or b is None or a == b:
                    continue
                judged.append((a, b, probability) if side == "a" else (b, a, probability))
        return judged

    def entity_structure(self, *, user_id: str | None = None) -> dict[str, dict[str, Any]]:
        """Hub status and home for every active entity.

        Computed on request and never stored, so a phrase seen a second time is
        a hub the next time anyone looks, and nothing has to be kept in step.
        """
        entities, nodes, triples, judged = self._structure_inputs(user_id)
        homes = derive_homes(nodes, triples, judged)
        names = {node.id: node.name for node in nodes}
        screens = {e.id: (e.metadata or {}).get("screen") for e in entities}
        out: dict[str, dict[str, Any]] = {}
        for node in nodes:
            home = homes.get(node.id)
            why = hub_reason(node.entity_type, node.memories, node.relations,
                             screens.get(node.id))
            screen = screens.get(node.id) or {}
            out[node.id] = {
                "hub": bool(why),
                "why": why,
                "screened_out": bool(
                    not screen.get("kept")
                    and screen.get("verdict") in SCREEN_SKIPS
                    and float(screen.get("probability") or 0.0) >= SCREEN_GATE
                ),
                "memories": node.memories,
                "relations": node.relations,
                "home": ({"id": home["id"], "name": names.get(home["id"], ""),
                          "share": home["share"], "source": home["source"]}
                         if home else None),
            }
        # Tags are listed with their memories, and are never a hub nor a home:
        # the structure rules above only ever see named things.
        tagged = Counter(
            entity_id for entity_id, _ in self.backend.entity_memory_links(
                Scope(user_id=user_id), kind="topic"))
        for entity_id, memories in tagged.items():
            out[entity_id] = {"hub": False, "why": "", "screened_out": False,
                              "memories": memories, "relations": 0, "home": None}
        return out

    def run_structure_pass(
        self, *, user_id: str | None = None, dry_run: bool = False, exact_user: bool = False,
    ) -> dict[str, Any]:
        """Record where parts belong, and settle names that are shared.

        Nothing is deleted. A home is a note in the entity's metadata,
        recomputed every pass. A merge sets ``merged_into`` on the entity with
        less evidence, exactly as a confirmed proposal does. ``dry_run=True``
        changes nothing and returns the full plan instead. ``exact_user`` as
        in ``get_all``.
        """
        entities, nodes, triples, judged = self._structure_inputs(user_id, exact_user)
        homes = derive_homes(nodes, triples, judged)
        names = {node.id: node.name for node in nodes}
        outcome: dict[str, Any] = {
            "homes": len(homes), "homes_changed": 0,
            "merged": 0, "asked": 0, "separate": 0, "dry_run": dry_run,
        }
        for entity in entities:
            wanted = homes.get(entity.id)
            stored = None
            if wanted:
                stored = {"id": wanted["id"], "name": names.get(wanted["id"], ""),
                          "share": wanted["share"], "source": wanted["source"]}
            metadata = dict(entity.metadata or {})
            if metadata.get("home") == stored:
                continue
            outcome["homes_changed"] += 1
            if dry_run:
                continue
            if stored:
                metadata["home"] = stored
            else:
                metadata.pop("home", None)
            self.backend.set_entity_metadata(entity.id, metadata)

        # the owner without a name shares "the user" with nobody it is
        # (``identity.unnamed_owner``): it is in no same-name step
        unnamed = {e.id for e in entities if unnamed_owner(e)}
        plan = same_name_plan([node for node in nodes if node.id not in unnamed], homes,
                              self._held_apart(Scope(user_id=user_id, exact_user=exact_user)))
        tally = {"merge": "merged", "ask": "asked", "separate": "separate"}
        for step in plan:
            outcome[tally[step["action"]]] += 1
            if step["action"] == "merge" and not dry_run:
                self.backend.merge_entities(step["keep"], step["other"])
        if dry_run:
            outcome["plan"] = [
                {"action": step["action"], "name": step["name"], "reason": step["reason"],
                 "keep_home": names.get((homes.get(step["keep"]) or {}).get("id", ""), ""),
                 "other_home": names.get((homes.get(step["other"]) or {}).get("id", ""), "")}
                for step in plan
            ]
            outcome["home_list"] = sorted(
                (names.get(entity_id, ""), names.get(home["id"], ""), home["share"], home["source"])
                for entity_id, home in homes.items()
            )
        return outcome

    def run_name_screen(
        self, *, user_id: str | None = None, limit: int | None = None,
        exact_user: bool = False,
    ) -> dict[str, Any]:
        """Ask the decision provider what each not-yet-screened name is.

        The verdict is a note on the entity and nothing more: a name judged a
        value or a role shows up under Upkeep for a yes or a no, and a name
        judged a named thing is a hub. A name already screened, here or when
        the save that created it screened it (``entities.resolve_mentions``),
        is not asked about again.
        """
        outcome: dict[str, Any] = {"screened": 0, "queued": 0, "skipped": 0}
        if not self.decider.available:
            outcome["skipped"] = -1
            return outcome
        scope = Scope(user_id=user_id, exact_user=exact_user)
        # The store owner is a person by construction, whatever its name
        # ("the user" until the judge finds who it is).
        pending = [
            e for e in self.backend.list_entities(scope, limit=1_000_000)
            if "screen" not in (e.metadata or {}) and not (e.metadata or {}).get("owner")
        ][: limit or self.SCREEN_BATCH]

        def ask(entity: Entity):
            memories = self.backend.entity_memories(entity.id, limit=1)
            if not memories:
                return entity, None
            verdict = screen_names(self.decider, memories[0].content, [entity.name])
            return entity, verdict.get(entity.name.strip().lower())

        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(ask, pending))
        for entity, verdict in results:
            if verdict is None:
                outcome["skipped"] += 1
                continue
            metadata = dict(entity.metadata or {})
            metadata["screen"] = {**verdict, "at": utcnow()}
            self.backend.set_entity_metadata(entity.id, metadata)
            outcome["screened"] += 1
            if verdict["verdict"] in SCREEN_SKIPS and verdict["probability"] >= SCREEN_GATE:
                outcome["queued"] += 1
        return outcome

    def _screen_rows(self, user_id: str | None) -> list[tuple[Entity, dict[str, Any]]]:
        """Entities whose screening verdict is waiting for a person."""
        rows = []
        for entity in self.backend.list_entities(Scope(user_id=user_id), limit=1_000_000):
            screen = (entity.metadata or {}).get("screen")
            if (
                isinstance(screen, dict)
                and not screen.get("kept")
                and screen.get("verdict") in SCREEN_SKIPS
                and float(screen.get("probability") or 0.0) >= SCREEN_GATE
            ):
                rows.append((entity, screen))
        return rows

    # ------------------------------------------------------------------
    # upkeep: what runs on its own, and the queue of what needs a person
    # ------------------------------------------------------------------
    #: Similarity at which the automatic consolidation pass looks for
    #: duplicates; the dashboard's old "close" setting.
    UPKEEP_CONSOLIDATION_THRESHOLD = 0.90
    #: Judged consolidation groups remembered per namespace, newest kept.
    UPKEEP_SEEN_LIMIT = 5000

    def upkeep_paused(self) -> bool:
        return self.backend.get_meta("maintenance:paused") == "true"

    def set_upkeep_paused(self, paused: bool) -> None:
        self.backend.set_meta("maintenance:paused", "true" if paused else "false")

    def _upkeep_get(self, name: str, user_id: str | None, default: Any) -> Any:
        raw = self.backend.get_meta(_upkeep_key(name, user_id))
        if not raw:
            return default
        try:
            return json.loads(raw)
        except ValueError:
            return default

    def _upkeep_set(self, name: str, user_id: str | None, value: Any) -> None:
        self.backend.set_meta(_upkeep_key(name, user_id), json.dumps(value))

    def last_pass_run(self, key: str, user_id: str | None) -> dict[str, Any] | None:
        """When a pass last ran for this namespace and what it reported."""
        return self._upkeep_get(f"last:{key}", user_id, None)

    def _pass_lock(self, key: str, user_id: str | None) -> threading.RLock:
        with self._pass_locks_guard:
            return self._pass_locks.setdefault((key, user_id), threading.RLock())

    def run_upkeep_pass(
        self, key: str, *, user_id: str | None = None, record: bool = True,
        at: datetime | None = None, exact_user: bool = False,
    ) -> dict[str, Any]:
        """Run one pass now, with the same code the scheduler uses.

        ``at`` is the tick the scheduler is working through, which is what the
        run is stamped with. Stamping the wall clock instead put the next run
        due at a different time than the tick that triggered it.

        A pass that is off (``maintenance_enabled``) runs nowhere: neither the
        scheduler nor "run now" starts it, and no run is recorded; the result
        says so (``ran`` False, with the reason).

        ``exact_user``: no ``user_id`` means the memories and entities without
        a user, not every user's (``Scope.exact_user``), as the scheduler's
        walk over the namespaces needs; the tags of a namespace are its own
        whatever it is (``merge_obvious_topics``).
        """
        if key not in self._MAINTENANCE_KEYS:
            raise ValueError(f"unknown pass: {key}")
        if not self.maintenance_enabled(key):
            return {"ran": False, "reason": self._pass_off_reason(key)}
        with self._pass_lock(key, user_id):
            stamp = at.isoformat(timespec="seconds") if at is not None else utcnow()
            if key == "dedup_entities":
                self.merge_obvious_topics(user_id=user_id)
                result = self.resolve_entities(user_id=user_id, exact_user=exact_user)
                # A calibrated provider judges names one by one in their memory; the
                # text model's batch review is the fallback when there is none.
                if self.decider.available:
                    result.update(self.run_name_screen(user_id=user_id, exact_user=exact_user))
                elif self.llm.available:
                    result.update(self.run_entity_review(user_id=user_id,
                                                         exact_user=exact_user))
                self.backend.set_meta(_dedup_run_key(user_id), stamp)
            elif key == "structure":
                result = self.run_structure_pass(user_id=user_id, exact_user=exact_user)
            elif key == "durability":
                result = self.score_memory_durability(user_id=user_id, exact_user=exact_user)
            elif key == "consolidation":
                result = self.run_consolidation_pass(user_id=user_id, exact_user=exact_user)
                self.backend.set_meta(_consolidation_run_key(user_id), stamp)
            else:
                raise ValueError(f"unknown pass: {key}")
            if record:
                self._upkeep_set(f"last:{key}", user_id, {"at": stamp, "result": result})
            return result

    def run_upkeep_cycle(
        self, *, user_id: str | None = None, now: datetime | None = None,
        exact_user: bool = False,
    ) -> dict[str, Any]:
        """One scheduler tick for one namespace: every pass that is on and due.

        Returns what ran, keyed by pass, so the scheduler can spread work over
        cycles on a many-account server. ``exact_user`` as in
        ``run_upkeep_pass``: the scheduler walks the namespaces with it, so
        its tick for None is the memories without a user, not everyone's.

        The first tick of a namespace whose owner is still called "the user"
        also looks for who the owner is in what it holds (``learn_owner``);
        it is marked done either way, so it costs nothing after.
        """
        now = now or datetime.now(timezone.utc)
        ran: dict[str, Any] = {}
        if self.upkeep_paused():
            return ran
        if (self._upkeep_get("owner_learned", user_id, None) is None
                and self.owner_name(user_id) == OWNER_PLACEHOLDER):
            # once a namespace: who the owner is, from what it already holds
            try:
                learned = self.learn_owner(user_id=user_id)
            except Exception as exc:
                log.warning("learning who the owner is failed: %s", exc)
                learned = None
            if learned and (learned["action"] or {}).get("action") != "none":
                ran["learn_owner"] = learned["action"]
        every = self.config.dedup_interval_days
        dedup_due = _due(self.backend.get_meta(_dedup_run_key(user_id)), every, now)
        if self.maintenance_enabled("structure") and dedup_due:
            ran["structure"] = self.run_upkeep_pass(
                "structure", user_id=user_id, at=now, exact_user=exact_user)
        if self.maintenance_enabled("dedup_entities") and dedup_due:
            ran["dedup_entities"] = self.run_upkeep_pass(
                "dedup_entities", user_id=user_id, at=now, exact_user=exact_user)
        if (
            self.maintenance_enabled("consolidation") and self.llm.available
            and _due(self.backend.get_meta(_consolidation_run_key(user_id)), every, now)
        ):
            ran["consolidation"] = self.run_upkeep_pass(
                "consolidation", user_id=user_id, at=now, exact_user=exact_user)
        if self.maintenance_enabled("durability") and self.decider.available:
            # Cheap when nothing is unscored, so it runs every tick; only a
            # tick that scored something is worth remembering as a run.
            result = self.run_upkeep_pass("durability", user_id=user_id, record=False,
                                          exact_user=exact_user)
            if result.get("scored"):
                self._upkeep_set("last:durability", user_id,
                                 {"at": now.isoformat(timespec="seconds"), "result": result})
                ran["durability"] = result
        if dedup_due:
            # After this week's merges and new homes: re-embed the memories
            # whose masked names changed. Nothing to embed is no run.
            try:
                embedded = self.refresh_property_vectors(user_id=user_id,
                                                         exact_user=exact_user)
            except Exception as exc:
                log.warning("property vector refresh failed: %s", exc)
                embedded = 0
            if embedded:
                ran["property_vectors"] = {"embedded": embedded}
        return ran

    def run_consolidation_pass(
        self, *, user_id: str | None = None, exact_user: bool = False,
    ) -> dict[str, Any]:
        """Merge exact duplicates on sight; queue what only a model vouched for.

        Entity identity was measured against a labelled set before it was
        allowed to merge on its own; a model's "same fact" has not been, so
        those merges wait for a person under Upkeep. Every judged group is
        remembered, so the model is asked about each one once.
        """
        with self._pass_lock("consolidation", user_id):
            repaired = self.repair_consolidated(user_id=user_id, exact_user=exact_user)
            seen_lists = self._upkeep_get("consolidation:seen", user_id, [])
            seen = {frozenset(ids) for ids in seen_lists}
            result = self.consolidate_memories(
                user_id=user_id, threshold=self.UPKEEP_CONSOLIDATION_THRESHOLD,
                apply=False, exclude=seen, exact_user=exact_user,
            )
            pending = self._upkeep_get("consolidation:pending", user_id, [])
            known = {frozenset(entry["memory_ids"]) for entry in pending}
            outcome = {"scanned": result["scanned"], "judged": 0, "merged": 0, "queued": 0,
                       **{key: n for key, n in repaired.items() if n}}
            for group in result["groups"]:
                ids = frozenset(group["memory_ids"])
                seen_lists.append(sorted(ids))
                outcome["judged"] += 1
                if not group["same_fact"]:
                    continue
                if group["reason"] == "identical text":
                    memories = [
                        m for m in (self.backend.get_memory(i) for i in ids)
                        if m is not None and m.invalid_at is None
                    ]
                    if len(memories) >= 2:
                        self._merge_group(memories, group["merged_content"], user_id=user_id)
                        outcome["merged"] += 1
                    continue
                if ids in known:
                    continue
                pending.append({
                    "id": _group_id(ids),
                    "memory_ids": sorted(ids),
                    "contents": group["contents"],
                    "merged_content": group["merged_content"],
                    "reason": group["reason"],
                    "found_at": utcnow(),
                })
                known.add(ids)
                outcome["queued"] += 1
            self._upkeep_set("consolidation:seen", user_id, seen_lists[-self.UPKEEP_SEEN_LIMIT:])
            self._upkeep_set("consolidation:pending", user_id, pending)
            return outcome

    def run_entity_review(
        self, *, user_id: str | None = None, exact_user: bool = False,
    ) -> dict[str, Any]:
        """Ask the model which concept names are not things; queue its verdicts.

        Nothing is removed here. A name the user chose to keep is never asked
        about again.
        """
        kept = set(self._upkeep_get("entity_review:kept", user_id, []))
        judged = self.entity_junk(user_id=user_id, judge=True,
                                  exact_user=exact_user)["judged"]
        pending = [
            {"id": j["id"], "name": j["name"]} for j in judged if j["id"] not in kept
        ]
        self._upkeep_set("entity_review:pending", user_id, pending)
        return {"reviewed": len(judged), "queued": len(pending)}

    def upkeep_queue(self, *, user_id: str | None = None) -> list[dict[str, Any]]:
        """Everything upkeep will not decide on its own, as rows a person can clear.

        Each row carries the labels for its two buttons so the dashboard never
        has to know what kind of thing it is showing.
        """
        items: list[dict[str, Any]] = []

        def entity_name(entity_id: str) -> str:
            entity = self.backend.get_entity(entity_id)
            return entity.name if entity is not None else entity_id

        for proposal in self.proposals_for_a_person(user_id):
            items.append({
                "kind": "proposal", "id": proposal.id,
                "title": f"{entity_name(proposal.entity_a)} and {entity_name(proposal.entity_b)}",
                "accept": "merge", "decline": "keep separate",
            })

        pending = self._upkeep_get("consolidation:pending", user_id, [])
        live: list[dict[str, Any]] = []
        for entry in pending:
            memories = [
                m for m in (self.backend.get_memory(i) for i in entry["memory_ids"])
                if m is not None and m.invalid_at is None
            ]
            if len(memories) < 2:
                continue
            live.append(entry)
            items.append({
                "kind": "consolidation", "id": entry["id"],
                "title": entry["merged_content"],
                "detail": entry["reason"],
                "replaces": [m.content for m in memories],
                "accept": "merge", "decline": "keep all",
            })
        if len(live) != len(pending):
            self._upkeep_set("consolidation:pending", user_id, live)

        for entry, new, old in self._open_conflicts(user_id):
            mark = _conflict_mark(new)
            held = mark.get("held")
            update = mark.get("kind") == "update"
            items.append({
                "kind": "conflict", "id": new.id,
                "title": new.content,
                "detail": ("This updates a memory you already have: the older one "
                           "would stay as history." if update
                           else "This contradicts a memory you already have.")
                          + (f" Memry kept both because {held}." if held else ""),
                "replaces": [old.content],
                "replaces_label": ("the memory it updates" if update
                                   else "the memory it contradicts"),
                "accept": "the new one is right",
                "decline": "the old one is right",
                "other": "both are true",
            })

        for entry in self._upkeep_get("entity_review:pending", user_id, []):
            if self.backend.get_entity(entry["id"]) is None:
                continue
            items.append({
                "kind": "entity_review", "id": entry["id"],
                "title": entry["name"],
                "detail": "Judged not to be a person, place or thing. Removing it "
                          "never touches the memories behind it.",
                "accept": "remove", "decline": "keep",
            })

        listed = {item["id"] for item in items if item["kind"] == "entity_review"}
        for entity, screen in self._screen_rows(user_id):
            if entity.id in listed:
                continue
            # A role ("landlord", "customers") is a word in a memory, not a thing
            # with a name. Removing the name records nothing in its place: the
            # memory keeps saying who is a landlord, and a stated link between
            # two things is already a relation. Guessing the holder from who
            # else the memory mentions was wrong about half the time on a real
            # store (a tax memory listing profile types is not a list of what
            # its owner is), so nothing is guessed.
            role = screen["verdict"] == "role"
            items.append({
                "kind": "role" if role else "entity_review", "id": entity.id,
                "title": entity.name, "detail": "",
                "accept": "remove", "decline": "keep",
            })
        return items

    def proposals_for_a_person(self, user_id: str | None) -> list[MergeProposal]:
        """Entity pairs Upkeep asks a person about. With a calibrated judge,
        none: a pair it could not settle waits for new evidence and is
        compared again, since a person would be guessing from the same facts."""
        if judges_pairs(self.decider):
            return []
        return self.merge_proposals(user_id=user_id, limit=1000)

    def upkeep_count(self, *, user_id: str | None = None) -> int:
        """How many rows wait under Upkeep, without asking a model anything:
        the dashboard shows this as a badge on every load."""
        waiting = {entity.id for entity, _ in self._screen_rows(user_id)}
        waiting |= {p["id"] for p in self._upkeep_get("entity_review:pending", user_id, [])}
        return (
            len(self.proposals_for_a_person(user_id))
            + len(self._upkeep_get("conflict:pending", user_id, []))
            + len(self._upkeep_get("consolidation:pending", user_id, []))
            + len(waiting)
        )

    def decide_upkeep(
        self, kind: str, item_id: str, decision: str, *,
        user_id: str | None = None, owner_prefix: str | None = None,
    ) -> bool:
        """Clear one queue row. ``decision`` is "accept" or "decline"; a
        contradiction also takes "other", for when both memories are true."""
        accept = decision == "accept"
        if kind == "conflict":
            return self._decide_conflict(
                item_id, decision, user_id=user_id, owner_prefix=owner_prefix
            )
        if decision == "other":
            return False
        if kind == "proposal":
            if accept:
                return self.confirm_merge(item_id, owner_prefix=owner_prefix)
            return self.reject_merge(item_id, owner_prefix=owner_prefix)
        if kind == "consolidation":
            pending = self._upkeep_get("consolidation:pending", user_id, [])
            entry = next((p for p in pending if p["id"] == item_id), None)
            if entry is None:
                return False
            # declined groups stay in the judged set, so they are not proposed again
            self._upkeep_set(
                "consolidation:pending", user_id, [p for p in pending if p["id"] != item_id]
            )
            if not accept:
                return True
            memories = [
                m for m in (self.backend.get_memory(i) for i in entry["memory_ids"])
                if m is not None and m.invalid_at is None and _owned(m, owner_prefix)
            ]
            if len(memories) < 2:
                return False
            self._merge_group(memories, entry["merged_content"], user_id=user_id)
            return True
        if kind in ("entity_review", "role"):
            entity = self.backend.get_entity(item_id)
            screen = (entity.metadata or {}).get("screen") if entity else None
            if isinstance(screen, dict) and _owned(entity, owner_prefix):
                # the text-model review may have listed the same name earlier;
                # one decision settles both
                stale = self._upkeep_get("entity_review:pending", user_id, [])
                if any(p["id"] == item_id for p in stale):
                    self._upkeep_set(
                        "entity_review:pending", user_id,
                        [p for p in stale if p["id"] != item_id],
                    )
                if not accept:
                    metadata = dict(entity.metadata or {})
                    metadata["screen"] = {**screen, "kept": True}
                    self.backend.set_entity_metadata(entity.id, metadata)
                    return True
                return self.remove_entities([item_id], owner_prefix=owner_prefix) > 0
        if kind == "entity_review":
            pending = self._upkeep_get("entity_review:pending", user_id, [])
            if not any(p["id"] == item_id for p in pending):
                return False
            self._upkeep_set(
                "entity_review:pending", user_id, [p for p in pending if p["id"] != item_id]
            )
            if accept:
                return self.remove_entities([item_id], owner_prefix=owner_prefix) > 0
            kept = self._upkeep_get("entity_review:kept", user_id, [])
            if item_id not in kept:
                kept.append(item_id)
            self._upkeep_set("entity_review:kept", user_id, kept)
            return True
        return False

    # ------------------------------------------------------------------
    # maintenance
    # ------------------------------------------------------------------
    def reindex(self) -> int:
        """Re-embed every memory with the currently configured embedder, then
        rebuild the ANN sidecar (when available)."""
        if not self.embedder.dimensions:
            return 0
        memories = self.backend.all_memories_iter(include_invalid=True)
        count = 0
        batch_size = 64
        for i in range(0, len(memories), batch_size):
            batch = memories[i : i + batch_size]
            vectors = self.embedder.embed([m.content for m in batch])
            for memory, vector in zip(batch, vectors):
                if vector:
                    self.backend.update_memory(
                        memory.id, embedding=vector, embedding_model=self.embedder.model_id,
                        touch=False
                    )
                    count += 1
        rebuild = getattr(self.backend, "rebuild_ann", None)
        if rebuild is not None:
            rebuild(self.embedder.model_id, self.embedder.dimensions)
        # one namespace at a time, None being the memories without a user
        for user_id in self.backend.distinct_user_ids() or [None]:
            self.refresh_property_vectors(user_id=user_id, exact_user=True)
        # the episodes' vectors, which evidence is chosen by (older stores had none)
        while episodes := self.backend.episodes_to_embed(self.embedder.model_id,
                                                         limit=batch_size):
            vectors = self.embedder.embed([e.content for e in episodes])
            self.backend.set_episode_vectors(
                {e.id: v for e, v in zip(episodes, vectors)}, self.embedder.model_id)
            if not all(vectors):
                break  # an episode the embedder gives no vector would come back forever
        return count

    def stats(self) -> dict[str, Any]:
        data = self.backend.stats()
        data.update(
            {
                "llm": f"{self.llm.name}"
                + (f":{getattr(self.llm, 'model', '')}" if getattr(self.llm, "model", "") else ""),
                "embedder": self.embedder.model_id,
                # Which provider answers typed judgements. "none" is the
                # default and means the built-in prompt path.
                "decider": self.decider.name
                + (f":{getattr(self.decider, 'model', '')}"
                   if getattr(self.decider, "model", "") else ""),
                # The merge gate in force right now, so the About panel can say
                # whether merges happen on their own and above what. Above 1.0
                # means never: the model answering has not been measured.
                "merge_gate": self.merge_gate(),
                # "invalidated" lumps together deleted memories and old versions
                # of updated ones. Only the first kind is recoverable, and only
                # that kind is what the Forgotten tab lists, so report it apart.
                # Counted, not listed: building the Forgotten list walked every
                # memory and looked up each removed one's history, on every call.
                "forgotten_memories": self.backend.count_memories()["forgotten"],
                "generated_at": utcnow(),
            }
        )
        return data

    def merge_gate(self) -> float:
        """The automatic-merge gate in force: a calibrated judge's own while it
        can answer. Without one no model's confidence merges anything
        (``entities.resolve_open_proposals``): above 1.0, merges never happen
        on a model's answer."""
        if not judges_pairs(self.decider):
            return NEVER_AUTO_MERGE
        return _gate(self.decider, self.llm)

    def count_memories(self, *, owner_prefix: str | None = None) -> dict[str, int]:
        """Active, invalidated and forgotten counts; see ``MemoryBackend.count_memories``."""
        return self.backend.count_memories(owner_prefix)

    def reset(self) -> None:
        self.backend.reset(keep_meta=_SETTINGS_KEYS)

    def close(self) -> None:
        try:
            self.decider.close()
        finally:
            try:
                self.llm.close()
            finally:
                try:
                    self.embedder.close()
                finally:
                    self.backend.close()
