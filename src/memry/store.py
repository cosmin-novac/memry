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
from datetime import datetime, timedelta, timezone
import hashlib
import json
import logging
import re
import threading
import time
from typing import Any

import numpy as np

log = logging.getLogger("memry")

from .backends.base import MemoryBackend
from .backends.local import LocalBackend
from .config import Config
from .intelligence.clustering import (
    judge_tag_pairs,
    obvious_canonical_merges,
    obvious_variant_prefix,
    propose_synthetic_tags,
    semantic_duplicate_tags,
    suggest_canonical_merges,
)
from .intelligence.consolidate import judge_group, representative, similarity_groups
from .intelligence.context import build_context, estimate_tokens
from .intelligence.decay import (
    DURABILITY_KEY,
    decay_sweep,
    score_durability,
)
from .intelligence.entities import (
    _gate,
    classify_entity_types,
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
    FAMILY_MIN,
    FAMILY_SCAN,
    FAMILY_TOP,
    LINKED_RELATION,
    SET_BAR,
    SET_RESULT_CAP,
    SET_SCAN,
    SET_SHARED,
    aboutness,
    activation_paths,
    detect_query_entities,
    homes_of,
    mask_first_person,
    mask_names,
    set_members,
    speaks_in_first_person,
)
from .intelligence.identity import (
    BELONGS_BAR,
    TAG_EXAMPLES,
    NameIndex,
    belonging,
    judged_tag_merges,
    judges_pairs,
    name_vectors,
)
from .intelligence.extraction import (
    VOCABULARY_LIMIT,
    extract_facts,
    extract_relations,
    verbatim_candidates,
    verify_coverage,
)
from .intelligence.reconcile import (
    CONFLICT_KEY,
    UPDATE_SUPERSEDE_REASON,
    reconcile_candidate,
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
    Memory,
    MemoryEvent,
    MemoryType,
    MergeProposal,
    Relation,
    Scope,
    SearchResult,
    SyntheticTag,
    TOPIC_TYPE,
    clean_tags,
    Topic,
    TopicRelation,
    later_ts,
    parse_ts,
    same_ts,
    utcnow,
)
from .providers.embeddings import Embedder, build_embedder
from .providers.decisions import Decider, Noul, build_decider
from .providers.llm import LLM, build_llm
from .retrieval import hybrid_search


_ENRICHMENT_KEY = "_enrichment"
_ENRICHMENT_BATCH_SIZE = 8
_ENRICHMENT_MAX_BACKOFF_SECONDS = 300


def _queued_at(memory: Memory) -> datetime:
    """When a pending save was queued: the quiet period counts from it. A save
    given an earlier ``created_at`` (``add_deferred``) still waits its turn."""
    job = (memory.metadata or {}).get(_ENRICHMENT_KEY) or {}
    return parse_ts(job.get("queued_at") or memory.created_at)


def _ingestion_context(metadata: dict[str, Any] | None) -> str:
    return " ".join(str((metadata or {}).get("context") or "").split())[:200]


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


def _dedup_run_key(user_id: str | None) -> str:
    return f"entity_dedup:v2:last_run:{user_id or ''}"


def _consolidation_run_key(user_id: str | None) -> str:
    return f"consolidation:last_run:{user_id or ''}"


def _upkeep_key(name: str, user_id: str | None) -> str:
    return f"upkeep:{name}:{user_id or ''}"


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
        score = re.search(r"importance ([0-9.]+) < ([0-9.]+)", reason)
        if score:
            return (f"The forgetting sweep removed it: its importance had faded to "
                    f"{score.group(1)}, below the {score.group(2)} it needs to stay.")
        return "The forgetting sweep removed it: its importance had faded too far."
    if event.event == "SUPERSEDE" and "into 0 fact" in reason:
        return ("It was a raw saved message, and distilling it produced nothing new: "
                "every fact in it was already stored, or there was nothing to keep.")
    return reason or f"Removed by {event.actor or 'the system'}, with no reason recorded."


def _is_update_supersede(event: MemoryEvent) -> bool:
    """A SUPERSEDE of an UPDATE nobody could write a merged text for: the newer
    memory adds to the old one, which is kept and was never contradicted. Read
    from the event's ``kind``; an older row without one, from its reason."""
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
    ``kind`` is "update" when an UPDATE nobody wrote the merged text for was
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


def _tag_run_key(user_id: str | None) -> str:
    """Meta key under which the last tag-abstraction run time is stamped."""
    return f"tag_abstraction:last_run:{user_id or ''}"


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


def _across_runs(scope: Scope) -> Scope:
    """A save's scope as the lookups across one person's saves read it: the
    whole user (with the agent), not one run. Reconcile's candidates and the
    tag vocabulary offered to extraction use it; topic canonicalization and
    entity lookup (``entities.resolve_mentions``) read the whole user too.
    Reconcile then acts by where the memory it matched lives
    (``reconcile.reconcile_candidate``): a contradiction supersedes a memory
    of any run, but a duplicate or an update of another run's memory adds the
    fact to the save's run, so a search of the run finds it (the
    consolidation pass merges duplicates across runs)."""
    if scope.user_id is None:
        return scope
    return Scope(user_id=scope.user_id, agent_id=scope.agent_id)


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
        ``valid_from``, the ``updated_at`` of a memory it rewrites, the
        ``invalid_at`` of one it supersedes, and the time of the events it
        records. A memory it rewrites or supersedes keeps a later
        ``updated_at`` it has (``repair_updated_at`` reads the same).
        ``now`` is the reference date extraction resolves "yesterday"
        against, and the when-confirmation reads as the day of writing,
        instead of the clock. All three are for replaying dated
        conversations, as the benchmarks do.
        """
        scope = Scope(user_id=user_id, agent_id=agent_id, run_id=run_id)
        messages = (
            [{"role": "user", "content": content}] if isinstance(content, str) else content
        )
        episodes = [
            Episode(
                content=m.get("content", ""),
                role=m.get("role", "user"),
                user_id=user_id,
                agent_id=agent_id,
                run_id=run_id,
                metadata=metadata or {},
                **({"created_at": created_at} if created_at else {}),
            )
            for m in messages
            if (m.get("content") or "").strip()
        ]
        if not episodes:
            return AddResult()
        if episodes:
            self.backend.add_episodes(episodes)
        episode_ids = [e.id for e in episodes]

        candidates: list[CandidateFact]
        warnings: list[str] = []
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
                candidates = extract_facts(
                    self.llm,
                    messages,
                    now=now,
                    vocabulary=self._tag_vocabulary(
                        scope,
                        text="\n".join(
                            str(m.get("content") or "") for m in messages
                        ),
                    ),
                    context=_ingestion_context(metadata),
                    tag_hints=_client_tag_hints(metadata, categories),
                    owner=self.owner_name(scope.user_id),
                    entity_names=self._entity_vocabulary(
                        scope,
                        "\n".join(str(m.get("content") or "") for m in messages),
                    ),
                )
                self._confirm_candidate_whens(candidates, now=now)
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
        actions = self._apply_candidates(candidates, scope, episode_ids, created_at=created_at)

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

    def add_deferred(
        self,
        content: str,
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
        """Durably save raw text for managed background enrichment.

        This path performs no provider calls. The episode and searchable pending
        memory are committed before the caller receives the result; the pending
        metadata is the restart-safe work marker consumed by the server worker.
        ``created_at``, ``memory_metadata`` and ``now`` mean what they mean for
        ``add``: they apply to the pending memory and are kept with the work
        marker for the distillation that follows. The quiet period counts from
        when the save was queued, whatever ``created_at`` says.
        """
        text = content.strip()
        if not text:
            return AddResult()
        queued_at = utcnow()
        stamp = created_at or queued_at
        episode = Episode(
            content=text,
            role="user",
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            metadata=metadata or {},
            created_at=stamp,
        )
        pending_metadata = {**(memory_metadata or {}), **(metadata or {})}
        pending_metadata["pending_distillation"] = True
        job: dict[str, Any] = {"status": "pending", "attempts": 0, "queued_at": queued_at}
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
            source_episode_ids=[episode.id],
            created_at=stamp,
            updated_at=stamp,
        )
        self.backend.add_episodes([episode])
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
            episode_ids=[episode.id],
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
        update alike: each tag in the obvious canonical form (singular,
        plural, spacing) it shares with the user's tags, and a name merged
        away written as its survivor (``MemoryBackend.tag_filing``), so the
        column names the tag its memory is counted and filtered under.

        Names merged away are resolved before obvious variants are grouped: a
        group holding one ("tax", merged into "levies", beside "taxes") is
        written as that name's survivor ("levies"), so a group never brings a
        retired name back. A save (``merge_stored``) runs the pass over the
        whole vocabulary and folds each stored variant into the form its
        group is written as (``_merge_topics``). An update rewrites its own
        tags only (``merge_stored`` False): it reads just the incoming tags'
        obvious variants and merges nothing, so no other memory is retagged;
        the vocabulary-wide pass is the next save's, or upkeep's."""
        incoming = {
            str(tag).strip().casefold()
            for tags in tag_lists
            for tag in tags
            if str(tag).strip()
        }
        if not incoming:
            return [list(tags) for tags in tag_lists]
        user = Scope(user_id=scope.user_id)
        vocabulary = self.backend.topic_names(
            user,
            prefixes=None if merge_stored else {obvious_variant_prefix(tag) for tag in incoming},
        )
        retired = {name for name, active in vocabulary.items() if not active}
        groups = [
            group for group in obvious_canonical_merges(
                [{"category": name} for name in set(vocabulary) | incoming])
            if merge_stored or incoming.intersection(group["variants"])
        ]
        # only names merged away need resolving here; every other tag is
        # resolved once, where the backend files the column
        survivor = self.backend.tag_filing(sorted((incoming & retired) | {
            name for group in groups for name in group["variants"] if name in retired
        }), user)
        replacements: dict[str, str] = {}
        for group in groups:
            variants = list(group["variants"])
            gone = sorted((name for name in variants if name in retired),
                          key=lambda name: (name != group["canonical"], name))
            target = survivor.get(gone[0], gone[0]) if gone else group["canonical"]
            replacements.update({variant: target for variant in variants})
            stored = {name for name in variants if vocabulary.get(name) and name != target}
            if merge_stored and stored:
                self._merge_topics(scope.user_id, stored, target, exact_user=True)
        rewritten_lists: list[list[str]] = []
        for tags in tag_lists:
            rewritten: list[str] = []
            seen: set[str] = set()
            for raw in tags:
                normalized = str(raw).strip().casefold()
                canonical = replacements.get(normalized, normalized)
                canonical = survivor.get(canonical, canonical)  # merged away: its survivor
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
    ) -> list[AddAction]:
        """Reconcile candidates into the store (shared by add and distill).

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
            # the user's memories across runs; the save's own scope (run
            # included) decides what a match may do (``reconcile_candidate``)
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
                episode_ids=episode_ids,
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
                    owner=self._owner_for(scope, candidate.entities),
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

    def _open_proposals_to_recheck(self, scope: Scope) -> list[MergeProposal]:
        """Open proposals a save may compare again, or none when the provider
        is too slow to ask inside a save."""
        if not (self.decider.rejudges_on_new_evidence and self.decider.available):
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
                backend=self.backend, llm=self.llm, decider=self.decider,
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
        caller replaces the stored text and mentions. Without one, Memry can
        still retain or remove existing links by matching their known aliases;
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
            resolved = resolve_mentions(
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
            )
        except Exception as exc:
            raise ValueError(
                f"memory text was not changed because entity re-analysis failed: {exc}"
            ) from exc
        mentions = [
            EntityMention(
                entity_id=resolved[surface.lower()].id,
                memory_id=memory_id,
                surface=surface,
            )
            for surface in surfaces
            if surface.lower() in resolved
        ]
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
        default_uid = user_id or self.config.default_user_id
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
        """Restore a Memry backup exactly and transactionally."""
        return self.backend.import_backup(backup, owner_prefix=owner_prefix)

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

        Related saves in the same scope and with the same optional ``context``
        metadata are distilled together after the group has been quiet. Raw
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
            key = (
                memory.user_id,
                memory.agent_id,
                memory.run_id,
                _ingestion_context(memory.metadata).casefold(),
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
        if not self.llm.available:
            raise ValueError("no LLM configured; distillation needs one")

        messages = [
            {"role": "user", "content": memory.content} for memory in active
        ]
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
        candidates = extract_facts(
            self.llm,
            messages,
            now=now,
            vocabulary=self._tag_vocabulary(
                first_scope,
                text="\n".join(memory.content for memory in active),
            ),
            context=context or None,
            tag_hints=tag_hints,
            owner=self.owner_name(first_scope.user_id),
            entity_names=self._entity_vocabulary(
                first_scope, "\n".join(memory.content for memory in active)
            ),
        )
        self._confirm_candidate_whens(candidates, now=now)
        if not candidates:
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
        )
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
    ) -> list[SearchResult]:
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
        # Over-fetch when we will post-filter or fuse, so a full page survives.
        wide = (since or until or when_since or when_until) or (
            relational and not categories and not entity_id
        )
        fetch = limit if not wide else min(max(limit * 8, 40), 500)
        results = hybrid_search(
            backend=self.backend,
            embedder=self.embedder,
            query=query,
            scope=scope,
            limit=fetch,
            cfg=self.config.retrieval,
            include_invalid=include_invalid,
            categories=categories,
            entity_id=entity_id,
        )
        # The linked search: the memories of the entities linked to the query's
        # join the ranking (multi-hop answers hybrid alone scores at zero).
        if relational and not categories and not entity_id:
            results = self._search_linked(query, scope, results, include_invalid)
        if since or until:
            results = [r for r in results if _within(r.memory.created_at, since, until)]
        if when_since or when_until:
            results = [
                r for r in results if _when_within(r.memory, when_since, when_until)
            ]
        # a question needing several memories returns every member found
        members = sum(1 for r in results if r.signals.get("member"))
        return self._rerank(query, results)[:max(limit, min(members, SET_RESULT_CAP))]

    def _reranks(self) -> bool:
        """Whether the decision provider re-ranks. The setting decides where it
        is set; otherwise the provider's default stands. Either way a provider
        that was not measured to beat no re-ranking cannot be talked into it:
        through gpt-5-mini the same work scored below the baseline at ten
        seconds a search."""
        cfg = self.config.decision
        wanted = cfg.rerank if cfg.rerank is not None else self.decider.reranks_by_default
        return bool(wanted and self.decider.may_rerank)

    def relevance_mode(self) -> str:
        """What judges relevance in the linked search: ``retrieval.
        relational_relevance``, with "auto" read as "jev" where the decision
        provider re-ranks (``_reranks``: Jev by default, a text model measured
        to help when ``decision.rerank`` is on) and as "vector" elsewhere."""
        mode = self.config.retrieval.relational_relevance
        if mode != "auto":
            return mode
        return "jev" if self._reranks() else "vector"

    def _rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        """Let a relevance judgement adjust the hybrid order, not replace it.

        Hybrid ranking matches wording and carries recency, decayed importance,
        entity anchors and relation hops with it. Ordering purely by "does this
        text answer the question" measured worse than doing nothing, because it
        throws all of that away. Blending keeps it and adds what wording alone
        cannot see, and a floor lets an obvious non-answer be pushed back
        however well it matched.

        One call covers the whole shortlist. A provider that abstains or fails
        leaves the order exactly as it found it. It runs only where the linked
        search did not (no result carries its "about"): after the linked search
        its own order stands, judged or not.
        """
        cfg = self.config.decision
        if not self._reranks():
            return results
        if not self.decider.available or len(results) < 2:
            return results
        if any("about" in r.signals or "judged" in r.signals for r in results):
            return results  # the linked search ordered them, or already asked the same
        pool = results[: max(cfg.rerank_pool, 2)]
        answers = self.decider.decide(
            f"QUESTION: {query}",
            {f"m{i}": Noul(instructions="This memory helps answer the question. "
                                        f"Memory: {r.memory.content}")
             for i, r in enumerate(pool)},
        )
        if not any(answers[f"m{i}"].available for i in range(len(pool))):
            return results
        span = max(len(pool) - 1, 1)
        ordered = []
        for i, result in enumerate(pool):
            hybrid = 1.0 - (i / span)
            answer = answers[f"m{i}"]
            relevance = answer.value if answer.available else hybrid
            demoted = 1 if (answer.available and relevance < cfg.rerank_floor) else 0
            blended = cfg.rerank_weight * relevance + (1 - cfg.rerank_weight) * hybrid
            ordered.append((demoted, -blended, i, result))
        ordered.sort()
        return [r for _d, _s, _i, r in ordered] + results[len(pool):]

    def _search_linked(
        self, query: str, scope: Scope, results: list[SearchResult], include_invalid: bool,
    ) -> list[SearchResult]:
        """Score each candidate by how well it states the property asked
        (similarity of the question and the memory with the names of the
        entities the links reach replaced by "it") to the power ``relational_sharpness``, times how
        strongly it is about the entity the query names (``aboutness``). The
        links are followed directed and weighted, ``relational_depth`` deep
        (``graph_retrieval.activation_paths``). The candidates are the text
        ranking's and, for every entity linked at ``FAMILY_MIN`` or more, the
        ``FAMILY_TOP`` of its memories in the scope searched (its user, agent
        and run, as the text ranking's) that best state the property. A query
        naming no hub keeps the text ranking.

        With ``relational_relevance = "jev"`` (or "auto" with a provider that
        re-ranks, ``relevance_mode``) the decision provider then judges
        whether each of the first ``decision.rerank_pool`` answers the
        question, and that replaces the similarity for them
        (``_judge_in_rounds``). An answer from a thing the query's entity
        belongs to then counts only as far as none of the entity's own
        memories answers (a version's own change wins). Both count to the
        power of P(the question asks for one property): on "Show everything
        about it" aboutness alone orders the list."""
        cfg = self.config.retrieval
        seeds = [e for e in detect_query_entities(self.backend, scope, query, longest=True)
                 if self._is_hub(e)]
        first_person = False
        if not seeds and speaks_in_first_person(query):
            # "Where do I live?" names nobody: it is about the store's owner
            owner = self.owner_entity(scope.user_id)
            if owner is not None and self._is_hub(owner.id):
                seeds, first_person = [owner.id], True
        judges = self.relevance_mode() == "jev"
        if not seeds:
            if judges:
                return self._judge_in_rounds(query, results, scope, include_invalid)
            return results
        act, above = activation_paths(self.backend, seeds, depth=cfg.relational_depth,
                                      relation=LINKED_RELATION)
        names = [n for seed in seeds for n in self.backend.entity_aliases(seed)]
        question = mask_names(query, names)
        if first_person:
            question = mask_first_person(question)
        asked = self._asked_vector(question)

        pool: dict[str, SearchResult] = {r.memory.id: r for r in results}
        for entity_id, strength in act.items():
            if strength < FAMILY_MIN:
                continue
            # an entity's memories span runs: only those of the scope
            # searched, kept to in SQL before the newest FAMILY_SCAN are taken
            members = self.backend.entity_memories(
                entity_id, limit=FAMILY_SCAN, include_invalid=include_invalid, scope=scope)
            vectors = self._property_vectors([m.id for m in members])
            for memory in sorted(members,
                                 key=lambda m: -_similarity(asked, vectors.get(m.id)))[:FAMILY_TOP]:
                pool.setdefault(memory.id, SearchResult(memory=memory, score=0.0))
        entities: dict[str, list[Entity]] = {}
        scores = self._linked_scores(asked, list(pool), act, entities)
        scored = []
        for mid, result in pool.items():
            relevance, about = scores[mid]
            result.signals = {**result.signals, "property": round(relevance, 4),
                              "about": round(about, 3)}
            scored.append((relevance ** cfg.relational_sharpness * about, result.score, result))
        scored.sort(key=lambda item: (-item[0], -item[1]))
        ranked = [result for _, _, result in scored]
        if not judges:
            return ranked
        return self._judge_in_rounds(question if len(seeds) == 1 else query, ranked, scope,
                                     include_invalid, link={"act": act, "above": above,
                                                            "seeds": set(seeds),
                                                            "entities": entities,
                                                            "asked": asked})

    def _asked_vector(self, question: str) -> np.ndarray:
        """The question's vector as the property comparison reads it: cut to
        ``retrieval.property_dimensions`` and of length one."""
        asked = np.asarray(self.embedder.embed([question])[0], dtype=np.float32)
        asked = asked[: self.config.retrieval.property_dimensions or len(asked)]
        asked /= float(np.linalg.norm(asked)) or 1.0
        return asked

    def _linked_scores(
        self, asked: np.ndarray, memory_ids: list[str], act: dict[str, float],
        entities: dict[str, list[Entity]],
    ) -> dict[str, tuple[float, float]]:
        """(property similarity to ``asked``, aboutness) of each memory, as the
        linked search scores it. Only the names the links account for are
        masked: a memory naming an entity they reach is compared by its
        property vector, any other by its ordinary one, names kept ("Lena Blum
        works on Project Ekmibo" would otherwise read "It works on it", as
        empty as "What do I know about it?"). ``entities`` caches each memory's
        entities and is filled in, those not cached yet read at once."""
        missing = [mid for mid in dict.fromkeys(memory_ids) if mid not in entities]
        if missing:
            entities.update(self.backend.entities_of_memories(missing))
        reached = [mid for mid in memory_ids if any(e.id in act for e in entities[mid])]
        vectors = self._property_vectors(reached)
        vectors.update(self.backend.vectors_of([mid for mid in memory_ids if mid not in vectors],
                                               self.embedder.model_id))
        return {mid: (_similarity(asked, vectors.get(mid)),
                      aboutness([act.get(e.id) for e in entities[mid]]))
                for mid in memory_ids}

    def _judge_in_rounds(
        self, question: str, ranked: list[SearchResult], scope: Scope, include_invalid: bool,
        link: dict | None = None,
    ) -> list[SearchResult]:
        """The decision provider judges the first ``decision.rerank_pool`` of
        the ranking, and with them what kind of question it is, in one call:
        - about everything ("Show everything about X") or with one answer
          ("Where does Ada live?"): that is all;
        - several ("Which car is the cheapest?", "How much did I spend on
          groceries?"): one more call judges up to ``retrieval.set_pool``
          memories more (``_set_pool``), and the members of the set
          (``set_members`` over both calls' scores) come first.
        The "rounds" signal says how many calls were made (1 or 2), "pool" how
        many memories the second judged.
        ``link`` carries the linked search's activation: then "it" stands for
        each memory's own entity, aboutness weighs the order and a thing's
        answer yields to its version's own. Without it (a question naming
        nobody) memories are read as written and the judgement alone orders."""
        size = max(self.config.decision.rerank_pool, 2)
        act = link["act"] if link else {}
        above = link["above"] if link else set()
        seeds = link["seeds"] if link else set()
        entities: dict = link["entities"] if link else {}
        homes: dict[str, set[str]] = {}
        aliases: dict[str, list[str]] = {}

        def ents(mid: str) -> list:
            if mid not in entities:
                entities[mid] = self.backend.entities_of_memory(mid)
            return entities[mid]

        def subject(mid: str) -> str | None:
            linked = [e.id for e in ents(mid) if e.id in act]
            return max(linked, key=act.get) if linked else None

        # With one entity named, "it" stands for it in the question and the
        # memories; with several ("Did Ilva like Olive Kitchen?") the names
        # stay, or the question would read "Did it like it?".
        masking = bool(link) and len(seeds) == 1

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
            if link:
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
        rounds, pooled = 1, 0

        members: set[str] = set()
        if specific >= 0.5 and several >= SET_BAR:
            try:
                asked = link["asked"] if link else self._asked_vector(question)
            except Exception:  # embedding service down: the batch keeps its order
                asked = None
            batch = self._set_pool(ranked, size, judged, scope, include_invalid,
                                   asked=asked, act=act, entities=entities,
                                   members=set_members(judged))
            if batch:
                for result in batch:
                    if result.memory.id not in found:
                        found[result.memory.id] = result
                        extra.append(result)
                got, _, _ = judge(batch, False)
                rounds, pooled = 2, len(batch)
                judged.update(got)
            members = set_members(judged)
        # What is true of the thing a seed belongs to holds for the seed only
        # where the seed says nothing else: an answer reached by a step up
        # counts as far as none of the seed's own memories answers. Both
        # relevance and that override are per property, so they count as far
        # as the question asks for one ("Show everything about it" does not).
        overridden = max((value for mid, value in judged.items()
                          if seeds and any(e.id in seeds for e in ents(mid))), default=0.0)
        order = []
        for mid, value in judged.items():
            result = found[mid]
            if link and subject(mid) in above:
                value *= 1.0 - overridden
                result.signals = {**result.signals, "overridden": round(overridden, 4)}
            about = 1.0
            if link:
                about = result.signals.get("about") or aboutness([act.get(e.id) for e in ents(mid)])
                result.signals = {**result.signals, "about": round(about, 3)}
            result.signals = {**result.signals, "judged": round(value ** specific, 4),
                              "specific": round(specific, 4), "several": round(several, 4),
                              "rounds": rounds, "pool": pooled,
                              **({"member": True} if mid in members else {})}
            order.append((mid not in members, -(value ** specific) * about, result))
        order.sort(key=lambda item: (item[0], item[1]))
        return [result for *_, result in order] + [
            r for r in ranked + extra if r.memory.id not in judged]

    def _set_pool(
        self, ranked: list[SearchResult], size: int, judged: dict[str, float], scope: Scope,
        include_invalid: bool, *, asked: np.ndarray | None, act: dict[str, float],
        entities: dict[str, list[Entity]], members: set[str],
    ) -> list[SearchResult]:
        """What the second call of a question needing several memories judges:
        at most ``retrieval.set_pool`` memories not judged yet.

        The members of a set are the same kind of fact and filed under the same
        topics (tags). So the topics that at least ``SET_SHARED`` of the first
        ``size`` of the ranking carry are gathered, and every memory filed
        under one of them (its newest ``SET_SCAN``) scores, over those topics,
        how many of the first carry the topic over how many memories the topic
        has in the scope searched: a small topic most of them share counts
        most. The best are taken, a tie by the property ranking. Measured on
        the dense world's set questions, those held 85 to 100% of each set
        within 100 candidates.

        Where the topics give fewer than the budget (the first share none: an
        untagged store, memories saved with ``infer=False`` or imported
        verbatim), the rest are the unjudged memories nearest the ``members``
        found in the first call (``_nearest_unjudged``): measured, those held
        76 to 100% of the rest of a set, the ranking past the first 29 to 36%.
        Only with no member to start from is the ranking past the first taken.
        Either way the batch is ordered as the linked search orders (the
        property similarity to ``asked``, to the power
        ``relational_sharpness``, times aboutness)."""
        budget = max(self.config.retrieval.set_pool, 0)
        if not budget:
            return []
        sharpness = self.config.retrieval.relational_sharpness

        def linked_order(memory_ids: list[str]) -> dict[str, float]:
            if asked is None or not memory_ids:
                return dict.fromkeys(memory_ids, 0.0)
            scores = self._linked_scores(asked, memory_ids, act, entities)
            return {mid: relevance ** sharpness * about
                    for mid, (relevance, about) in scores.items()}

        first = [r.memory.id for r in ranked[:size]]
        done = set(first) | set(judged)
        topics = self.backend.entities_of_memories(first, kind="topic")
        carried = Counter(topic.id for mid in first for topic in topics.get(mid, []))
        shared = [topic_id for topic_id, count in carried.items() if count >= SET_SHARED]
        # how many memories each has where the search looks, as ``filed`` is read
        sizes = self.backend.entity_memory_counts(shared, scope=scope) if shared else {}
        walk: dict[str, float] = defaultdict(float)
        memories: dict[str, Memory] = {}
        for topic_id in shared:
            # the scope searched is kept to in SQL, before the newest SET_SCAN
            filed = self.backend.entity_memories(topic_id, limit=SET_SCAN,
                                                 include_invalid=include_invalid, scope=scope)
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
            best = sorted(walk, key=lambda mid: -walk[mid])
            edge = round(walk[best[min(budget, len(best)) - 1]], 9)
            # the candidates: all above the cut and all tied at it, each
            # scored once for the tie-break and the batch's order alike
            best = [mid for mid in best if round(walk[mid], 9) >= edge]
            order = linked_order(best)
            if len(best) > budget:
                chosen = [mid for mid in best if round(walk[mid], 9) > edge]
                tied = sorted((mid for mid in best if round(walk[mid], 9) == edge),
                              key=lambda mid: -order[mid])
                best = chosen + tied[: budget - len(chosen)]
            batch = [known.get(mid) or SearchResult(memory=memories[mid], score=0.0)
                     for mid in best]
        if len(batch) < budget:
            taken = done | {r.memory.id for r in batch}
            rest = [known.get(m.id) or SearchResult(memory=m, score=0.0)
                    for m in self._nearest_unjudged(members, taken, scope, include_invalid,
                                                    budget - len(batch))]
            if not batch and not rest:  # no member to start from
                rest = [r for r in ranked[size:] if r.memory.id not in done][:budget]
            order.update(linked_order([r.memory.id for r in rest]))
            batch += rest
        return sorted(batch, key=lambda r: -order[r.memory.id])

    def _nearest_unjudged(
        self, members: set[str], taken: set[str], scope: Scope, include_invalid: bool,
        count: int,
    ) -> list[Memory]:
        """The ``count`` memories of the scope searched nearest the members of
        a set found so far, none of ``taken``: the store's vector search from
        the centroid of the members' vectors. Members of a set are the same
        kind of fact ("It costs 21,000 euros"), so they sit closer to each
        other than to the question. Empty with no member, or none with a
        vector of the embedder in use."""
        if count <= 0 or not members:
            return []
        model = self.embedder.model_id
        vectors = list(self.backend.vectors_of(sorted(members), model).values())
        if not vectors or len({v.shape for v in vectors}) != 1:
            return []
        unit = [v / (float(np.linalg.norm(v)) or 1.0) for v in vectors]
        centre = np.mean(unit, axis=0)
        centre /= float(np.linalg.norm(centre)) or 1.0
        hits = self.backend.vector_search(centre.tolist(), model, scope,
                                          limit=count + len(taken),
                                          include_invalid=include_invalid)
        return [memory for memory, _ in hits if memory.id not in taken][:count]

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
        self, *, user_id: str | None = None, memory_ids: list[str] | None = None
    ) -> int:
        """Embed the property vector of each valid memory whose masked text is
        new, changed (a merge, a rename, a new home) or was embedded by another
        model, in batches of 64. A memory whose masked text is its text gets no
        row: search reads its ordinary vector, which is the same. With
        ``memory_ids`` only those memories (a save); otherwise every memory of
        the namespace that names an entity (the weekly upkeep, a backfill).
        Returns how many it embedded."""
        if not self.embedder.dimensions:
            return 0
        entities: dict[str, list[str]] = defaultdict(list)
        if memory_ids is None:
            scope = Scope(user_id=user_id)
            for entity_id, memory_id in self.backend.entity_memory_links(scope, kind="named"):
                entities[memory_id].append(entity_id)
            contents = {m.id: m.content
                        for m in self.backend.list_memories(scope, limit=10_000_000)
                        if m.id in entities}
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
        token_budget: int = 1200,
        limit: int = 20,
    ) -> ContextResult:
        scope = Scope(user_id=user_id, agent_id=agent_id, run_id=run_id)
        results = self.search(
            query, user_id=user_id, agent_id=agent_id, run_id=run_id, limit=limit
        )
        entity_text, entity_memory_ids = self._entity_context(
            scope, query, token_budget=min(300, max(80, token_budget // 4))
        )
        remaining = max(0, token_budget - estimate_tokens(entity_text))
        memory_context = build_context(results, token_budget=remaining)
        parts = [part for part in (entity_text, memory_context.text) if part]
        combined = "\n\n".join(parts)
        memory_ids = list(dict.fromkeys([*entity_memory_ids, *memory_context.memory_ids]))
        return ContextResult(
            text=combined,
            memory_ids=memory_ids,
            token_estimate=estimate_tokens(combined) if combined else 0,
        )

    def _entity_context(
        self, scope: Scope, query: str, *, token_budget: int
    ) -> tuple[str, list[str]]:
        entity_ids = detect_query_entities(self.backend, scope, query)[:3]
        if not entity_ids:
            return "", []
        header = "## Known entities (memry)\n"
        used = estimate_tokens(header)
        lines: list[str] = []
        memory_ids: list[str] = []
        for entity_id in entity_ids:
            entity = self._refresh_entity_description(entity_id)
            if entity is None or not entity.description:
                continue
            label = entity.name
            if entity.entity_type:
                label += f" ({entity.entity_type})"
            line = f"- {label}: {entity.description}"
            cost = estimate_tokens(line) + 1
            if used + cost > token_budget:
                continue
            lines.append(line)
            used += cost
            memory_ids.extend(
                memory.id for memory in self.backend.entity_memories(entity.id, limit=20)
            )
        if not lines:
            return "", []
        return header + "\n".join(lines), list(dict.fromkeys(memory_ids))

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
    ) -> dict[str, Any]:
        """Content-free aggregate graph over every active memory in scope.

        A planet is a hub that came up at least twice; a person is one from the
        first mention. On a real store the hubs alone were still 1,424 planets,
        610 of them things typed as a product or project and seen exactly once,
        so the map asks for a second sighting and the list does not. A part
        that has a home is not a planet of its own: it rides along on its home
        as one of its ``parts``. Nothing is hidden for good, since all of this
        is recomputed from the memories each time.
        """
        data = self.backend.knowledge_map(
            Scope(user_id=user_id, agent_id=agent_id, run_id=run_id)
        )
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
        data["entity_names"] = len(nodes)
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
        mention it. Counts are direct: synthetic parent tags are off, and a
        parent no longer rolls up the memories of the tags under it."""
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

    def direct_categories(
        self, *, user_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Histogram over tags attached straight to memories, no parent rollup.

        Abstraction must read this: if a system-generated parent appeared in
        its own input, the next run would cluster ``liver health`` and
        ``weekly gym`` into ``health`` and the useful level would decay one run
        at a time. ``categories()`` counts the same way now, since topic
        entities carry no hierarchy; this stays the name abstraction reads.
        """
        direct = self.backend.topic_mention_counts(Scope(user_id=user_id))
        return direct if direct is not None else self.categories(user_id=user_id)

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
    ) -> list[Memory]:
        scope = Scope(user_id=user_id, agent_id=agent_id, run_id=run_id)
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
        memory = self.backend.get_memory(memory_id)
        if not _owned(memory, owner_prefix):
            return False
        if hard:
            ok = self.backend.delete_memory(memory_id)
        else:
            ok = self.backend.invalidate_memory(memory_id) is not None
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
        threw out. Only records with nothing standing in for them belong here.
        """
        scope = Scope(user_id=user_id, agent_id=agent_id, run_id=run_id)
        out: list[dict[str, Any]] = []
        for memory in self.backend.list_memories(
            scope, include_invalid=True, limit=1_000_000
        ):
            if memory.invalid_at is None or memory.superseded_by:
                continue
            # Whatever ended it: a delete (yours, or the forgetting sweep), or
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
    ) -> bool:
        found = next(
            (row for row in self._open_conflicts(user_id) if row[0]["id"] == item_id),
            None,
        )
        if found is None:
            return False
        _, new, old = found
        if not (_owned(new, owner_prefix) and _owned(old, owner_prefix)):
            return False
        # held back from an UPDATE nobody wrote the merged text for: the new
        # one adds to the old one, so a confirmed replacement is an update's,
        # which the Archive's undo reverses keeping both (``undo_replacement``)
        update = _conflict_mark(new).get("kind") == "update"
        if decision == "accept":  # the new one is right
            self.backend.invalidate_memory(old.id, superseded_by=new.id)
            self.backend.add_event(MemoryEvent(
                memory_id=old.id, event="SUPERSEDE", old_content=old.content,
                new_content=new.content, actor="user",
                reason=(f"{UPDATE_SUPERSEDE_REASON}: you confirmed that memory {new.id} "
                        "replaces it" if update
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
                actor="user",
                reason=f"you kept it beside memory {old.id}: both are true",
            ))
        self._clear_conflict_mark(new)
        self._upkeep_set(
            "conflict:pending", user_id,
            [e for e in self._upkeep_get("conflict:pending", user_id, [])
             if e["id"] != item_id],
        )
        return True

    def replaced(
        self, *, user_id: str | None = None, limit: int = 200
    ) -> list[dict[str, Any]]:
        """Memories a contradiction, or an update kept and superseded, took
        out of use, newest first; ``contradiction`` says which.

        A memory that was consolidated or distilled lives on inside what
        replaced it, so there is nothing to undo. One that was contradicted is
        the opposite case - the store stopped believing it on one model's
        say-so - and that is the judgement worth a second look. One an update
        superseded, with no merged text written, holds what the newer memory
        does not say: its undo brings it back beside the newer one.
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
            if not (contradiction or _is_update_supersede(event)):
                continue
            out.append({
                "memory": memory,
                "replaced_at": memory.invalid_at,
                "replacement": self.backend.get_memory(memory.superseded_by),
                "reason": event.reason,
                "actor": event.actor,
                "contradiction": contradiction,
            })
        out.sort(key=lambda row: row["replaced_at"] or "", reverse=True)
        return out[:limit]

    def undo_replacement(
        self, memory_id: str, *, keep_new: bool = False,
        owner_prefix: str | None = None,
    ) -> bool:
        """Bring back a memory that a contradiction, or an update kept and
        superseded, replaced.

        ``keep_new`` leaves the replacement in use as well, for when both turn
        out to be true. Otherwise the replacement is forgotten - it goes to the
        Archive like any deleted memory, so this is itself undoable. The
        replacement of an update never contradicted the memory and is always
        kept.
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
        return self.backend.delete_memory(memory_id)

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
    ) -> list[Entity]:
        """Entities in scope: named things by default, tags (topic entities)
        with ``kind="topic"``, both with ``kind="any"``."""
        scope = Scope(user_id=user_id, agent_id=agent_id, run_id=run_id)
        return self.backend.list_entities(
            scope, include_merged=include_merged, limit=limit, kind=kind)

    def relations(self, *, user_id: str | None = None, limit: int = 1000) -> list[Relation]:
        return self.backend.list_relations(Scope(user_id=user_id), limit=limit)

    def restore_context_labels(
        self, *, user_id: str | None = None, dry_run: bool = False
    ) -> dict[str, Any]:
        """Give memories back the context label of the saves they came from.

        Facts extracted from a save did not keep the save's context label
        (fixed in the write path); the save's episode kept it, and every fact
        keeps its episode ids. A memory without a label takes the labels of its
        episodes, distinct ones joined as distillation joins them. Only
        memories without a label are looked at, so a second run changes
        nothing. Token-free. ``dry_run`` counts without writing."""
        missing = [
            m for m in self.get_all(user_id=user_id, limit=1_000_000)
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

    def repair_updated_at(self, *, user_id: str | None = None) -> dict[str, Any]:
        """Reconstruct each memory's updated_at from its audit trail.

        Housekeeping (tagging, relation backfill, re-embedding) used to bump
        updated_at; this recomputes the true value as the time of the last
        content-changing event (ADD/UPDATE/SUPERSEDE), or created_at if there was
        none. Token-free; idempotent. Fixes recency and decay after such a run.

        Times are compared as times (``later_ts``, ``same_ts``), as the write
        path compares them: a replayed save's "...T10:00:00Z" is earlier than
        a live "...T10:00:00.500000+00:00", though it sorts later as text.
        """
        fixed = 0
        for memory in self.get_all(user_id=user_id, include_invalid=True, limit=1_000_000):
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

    def backfill_relations(
        self, *, user_id: str | None = None, limit: int = 100_000
    ) -> dict[str, Any]:
        """One-time: extract typed relations from existing memories.

        Only memories with 2+ linked entities are considered (a relation needs
        two), each does one small focused LLM call, and each is marked done so a
        re-run spends no tokens. Cheap and resumable by design.
        """
        summary = {"scanned": 0, "processed": 0, "relations_added": 0, "skipped": 0}
        if not self.llm.available:
            summary["error"] = "no LLM configured"
            return summary
        for memory in self.get_all(user_id=user_id, limit=limit):
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
        self, *, user_id: str | None = None, batch: int = 40
    ) -> dict[str, Any]:
        """Classify entities that were linked before typing existed. Batched:
        one LLM call per ``batch`` entities, so a whole namespace is a handful of
        calls. Only untyped entities are touched, so re-runs cost nothing."""
        summary = {"typed": 0}
        if not self.llm.available:
            summary["skipped"] = "no LLM configured"
            return summary
        untyped = [
            e for e in self.backend.list_entities(Scope(user_id=user_id), limit=1_000_000)
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

    def _refresh_entity_description(
        self, entity_id: str, *, force: bool = False
    ) -> Entity | None:
        entity = self.backend.get_entity(entity_id)
        if entity is None or not entity.is_active:
            return None
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
        memories = self.backend.entity_memories(entity_id, limit=50)
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
        if entity_a != entity_b and not self.backend.merge_entities(entity_a, entity_b):
            return False
        self.backend.set_proposal_status(proposal_id, "confirmed")
        return True

    def reject_merge(
        self, proposal_id: str, *, owner_prefix: str | None = None
    ) -> bool:
        """User says: these are different entities. They stay separate for good."""
        proposal = self.backend.get_proposal(proposal_id)
        if not _owned(proposal, owner_prefix) or proposal.status != "proposed":
            return False
        self.backend.set_proposal_status(proposal_id, "rejected")
        return True

    def entity_junk(
        self, *, user_id: str | None = None, judge: bool = False
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
        entities = self.entities(user_id=user_id, limit=100_000)
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
        the user or by an automatic pass - can be taken back.
        """
        removed = 0
        for entity_id in entity_ids:
            entity = self.backend.get_entity(entity_id)
            # A tag is removed from its memories on the tag page (delete_tag);
            # retiring its entity alone would leave the memories filed under it.
            if entity is not None and entity.entity_type == TOPIC_TYPE:
                continue
            if _owned(entity, owner_prefix):
                removed += int(self.backend.retire_entity(entity_id, reason))
        return removed

    def retired_entities(
        self, *, user_id: str | None = None, limit: int = 200
    ) -> list[dict[str, Any]]:
        """Names that were removed and can still be brought back."""
        return self.backend.list_retired_entities(Scope(user_id=user_id), limit=limit)

    def restore_entities(
        self, entity_ids: list[str], *, owner_prefix: str | None = None
    ) -> int:
        """Bring retired entities back, with the evidence that still exists."""
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
            restored += int(self.backend.restore_entity(entity_id))
        return restored

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

        removed = int(self.backend.retire_entity(
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

        Two tags merge as tags (``merge_tags``), so the memories of the one
        folded in are filed under the one kept. A tag and a named thing fold
        into the thing (``MemoryBackend.merge_entities``)."""
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
            if keep.user_id != other.user_id:
                return False
            self._retag(keep.user_id, {other.normalized}, keep.normalized, exact_user=True)
            return self.backend.resolve_entity_id(merge_root) == keep_root
        return self.backend.merge_entities(keep_root, merge_root)

    # -- the store owner ----------------------------------------------------
    def set_owner_name(self, user_id: str | None, name: str) -> None:
        """Record the name of the person a namespace belongs to, from their
        account. The owner entity starts with it; the identity judge may later
        find the owner to be a named person in the store, whose name it keeps.
        """
        name = " ".join(str(name or "").split())[:80]
        if name and self._upkeep_get("owner_name", user_id, None) != name:
            self._upkeep_set("owner_name", user_id, name)

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

    def owner_name(self, user_id: str | None) -> str:
        """The name the extractor lists the owner under: the owner entity's,
        else the account's, else "the user"."""
        entity = self.owner_entity(user_id)
        if entity is not None:
            return entity.name
        return self._upkeep_get("owner_name", user_id, None) or "the user"

    def _owner_for(self, scope: Scope, surfaces: list[str]) -> Entity | None:
        """The owner entity, created when these extracted names first include
        the owner's. It belongs to the whole namespace, not to one run."""
        owner = self.owner_entity(scope.user_id)
        if owner is not None:
            return owner
        name = self.owner_name(scope.user_id)
        if not any(str(s).strip().casefold() == name.casefold() for s in surfaces):
            return None
        owner = self.backend.insert_entity(Entity(
            name=name, normalized=name.lower(), entity_type="person",
            user_id=scope.user_id, metadata={"owner": True},
        ))
        self._upkeep_set("owner_entity", scope.user_id, owner.id)
        return owner

    def resolve_entities(self, *, user_id: str | None = None) -> dict[str, int]:
        """Re-judge open proposals with accumulated evidence; auto-confirm only
        clear, high-confidence matches. Everything ambiguous stays proposed.

        Then drop entities nothing references. Extraction inevitably produces
        some records that never attach to anything, and without this they
        accumulate forever: a real store reached 206 such rows out of 519.
        """
        scope = Scope(user_id=user_id)
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
            backend=self.backend, llm=self.llm, decider=self.decider, scope=scope
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
        junk = self.entity_junk(user_id=user_id)["mechanical"]
        outcome["junk_removed"] = sum(
            self.remove_entities([item["id"]], reason=item["reason"])
            for item in junk
        )
        return outcome

    # ------------------------------------------------------------------
    # tag abstraction
    # ------------------------------------------------------------------
    def tags_to_topics(
        self, *, user_id: str | None = None, all_users: bool = True, dry_run: bool = False
    ) -> list[dict[str, Any]]:
        """Give every tag of the legacy ``topics`` table its topic entity and
        every ``memory_topics`` link its mention, user by user; see
        ``LocalBackend.tags_to_topics``. Idempotent; ``dry_run`` only counts."""
        return self.backend.tags_to_topics(
            user_id=user_id, all_users=all_users, dry_run=dry_run)

    def synthetic_tags(self, *, user_id: str | None = None) -> list[SyntheticTag]:
        """The higher-level tags the system invented for this namespace."""
        return self.backend.list_synthetic_tags(Scope(user_id=user_id))

    def abstract_tags(self, *, user_id: str | None = None) -> dict[str, Any]:
        """Create higher-level topic nodes and hierarchy edges.

        Parent labels are not copied onto memories. Query-time hierarchy
        expansion makes a parent filter include memories linked to its children.
        """
        cfg = self.config.tags
        summary: dict[str, Any] = {"user_id": user_id, "applied": []}
        if not self.llm.available:
            summary["skipped"] = "no LLM configured"
            return summary
        histogram = self.direct_categories(user_id=user_id)
        if len(histogram) < cfg.min_tags:
            summary["skipped"] = f"only {len(histogram)} tags (< {cfg.min_tags})"
            self._stamp_tag_run(user_id)
            return summary

        existing = [t.tag for t in self.synthetic_tags(user_id=user_id)]
        proposals = propose_synthetic_tags(
            self.llm, histogram,
            existing_synthetic=existing,
            max_new=cfg.max_new_tags,
            min_cluster=cfg.min_cluster_size,
        )
        all_topics = self.backend.list_topics(Scope(user_id=user_id), limit=100_000)
        for proposal in proposals:
            tag = proposal["tag"].strip().lower()
            members = [
                member.strip().lower()
                for member in proposal["members"]
                if member.strip()
            ]
            parent = self.backend.upsert_topic(
                Topic(name=tag, normalized=tag, user_id=user_id, provenance="synthetic")
            )
            relations_added = 0
            for member in members:
                children = [topic for topic in all_topics if topic.normalized == member]
                if not children:
                    children = [
                        self.backend.upsert_topic(
                            Topic(
                                name=member,
                                normalized=member,
                                user_id=user_id,
                                provenance="memory",
                            )
                        )
                    ]
                    all_topics.extend(children)
                for child in children:
                    if child.id == parent.id:
                        continue
                    self.backend.add_topic_relation(
                        TopicRelation(
                            broader_topic_id=parent.id,
                            narrower_topic_id=child.id,
                            user_id=user_id,
                            provenance="synthetic",
                        )
                    )
                    relations_added += 1
            self.backend.record_synthetic_tag(
                SyntheticTag(tag=tag, source_tags=members, user_id=user_id)
            )
            all_topics.append(parent)
            summary["applied"].append(
                {"tag": tag, "source_tags": members, "relations_added": relations_added}
            )
        self._stamp_tag_run(user_id)
        return summary

    def merge_obvious_topics(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        judge: bool = True,
    ) -> dict[str, Any]:
        """Collapse formatting and plural duplicates, then, with a calibrated
        judge and ``judge`` set, the tags it puts at its tag merge threshold
        (identity.py). Each tag pair's funnel step is stored, so a pair is
        compared when found and once more at 10 memories a tag, not on every
        pass.

        Tags are topic entities of one user, so a merge is theirs whole:
        ``agent_id`` and ``run_id`` narrow which memories are counted, not
        which are rewritten. Each merge folds the variant's topic entity into
        the canonical one and rewrites the ``categories`` column
        (``_merge_topics``)."""
        scope = Scope(user_id=user_id, agent_id=agent_id, run_id=run_id)

        def counted() -> list[dict[str, Any]]:
            rows = self.backend.topic_mention_counts(scope, exact_user=True)
            return rows if rows is not None else self.categories(
                user_id=user_id, agent_id=agent_id, run_id=run_id)

        groups = obvious_canonical_merges(counted())
        changed = 0
        for group in groups:
            remove = set(group["variants"]) - {group["canonical"]}
            changed += self._merge_topics(
                user_id, remove, group["canonical"], exact_user=True) or 0
        judged: list[dict[str, Any]] = []
        if judge and judges_pairs(self.decider):
            tags = counted()
            compared = self._upkeep_get("tag_pairs", user_id, {})
            judged = judged_tag_merges(
                self.decider, tags, self._entities_named(scope, tags),
                lambda tag: [m.content for m in self.get_all(
                    user_id=user_id, agent_id=agent_id, run_id=run_id,
                    categories=[tag], limit=TAG_EXAMPLES)],
                self._tag_vectors(tags),
                compared=compared,
            )
            self._upkeep_set("tag_pairs", user_id, compared)
            for group in judged:
                remove = set(group["variants"]) - {group["canonical"]}
                changed += self._merge_topics(
                    user_id, remove, group["canonical"], exact_user=True) or 0
        return {"groups_merged": len(groups) + len(judged), "memories_changed": changed}

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

    def _entities_named(
        self, scope: Scope, tags: list[dict[str, Any]]
    ) -> dict[str, tuple[str, str | None]]:
        """For each tag that is also the name of an entity: that entity's name
        and type, as evidence of what the tag means."""
        labels = {str(t["category"]).strip().casefold() for t in tags}
        named: dict[str, tuple[str, str | None]] = {}
        for entity in self.backend.find_entities_by_aliases(sorted(labels), scope, limit=10_000):
            label = entity.name.strip().casefold()
            if label in labels:
                named[label] = (entity.name, entity.entity_type)
        return named

    def _tag_vectors(self, tags: list[dict[str, Any]]):
        if not self.embedder.dimensions or self.embedder.name == "hash":
            return None
        labels = [str(t["category"]).strip().casefold() for t in tags]
        vectors = name_vectors(self.embedder.embed, [
            Entity(id=label, name=label, user_id=None) for label in labels
        ])
        return vectors

    def consolidate_memories(
        self,
        *,
        user_id: str | None = None,
        threshold: float = 0.90,
        max_groups: int = 25,
        apply: bool = True,
        only: list[list[str]] | None = None,
        exclude: set[frozenset[str]] | None = None,
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
        audit trail and time-travel still resolve.
        """
        scope = Scope(user_id=user_id)
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

    def repair_consolidated(self, *, user_id: str | None = None) -> dict[str, int]:
        """Memories consolidated before the merge kept their vector's model and
        their originals' mentions: re-embed those stored without a model (the
        model that made them is unknown) and give back the mentions."""
        scope = Scope(user_id=user_id)
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

    def semantic_tag_duplicates(
        self, *, user_id: str | None = None, threshold: float = 0.93
    ) -> list[dict[str, Any]]:
        """Tags that have split one subject, judged by the stored vectors.

        Needs no LLM and no new storage: a tag's centroid is the mean of its
        members' existing embeddings. This is the drift that matters over a long
        life - a fragmented tag silently caps recall, because filtering to it
        excludes memories the question needed.
        """
        scope = Scope(user_id=user_id)
        links = self.backend.topic_mention_links(scope)
        if not links:
            return []
        vectors = dict(self.backend.memory_vectors(scope, limit=1_000_000))
        if not vectors:
            return []

        members: dict[str, list[str]] = {}
        for tag, memory_id in links:
            if memory_id in vectors:
                members.setdefault(tag, []).append(memory_id)
        counts = {tag: len(ids) for tag, ids in members.items()}

        by_memory: dict[str, list[str]] = {}
        for tag, memory_id in links:
            by_memory.setdefault(memory_id, []).append(tag)
        cooccurrence: dict[tuple[str, str], int] = {}
        for tags in by_memory.values():
            unique = sorted(set(tags))
            for i, a in enumerate(unique):
                for b in unique[i + 1:]:
                    cooccurrence[(a, b)] = cooccurrence.get((a, b), 0) + 1

        centroids = {
            tag: np.mean([vectors[m] for m in ids], axis=0)
            for tag, ids in members.items()
            if len(ids) >= 2
        }
        # Embedding the tag names lets the detector require that the LABELS mean
        # the same thing, not just that the member memories sit close together.
        # On a real store, centroid similarity alone either found nothing or
        # proposed wrong merges; the conjunction found exactly the true split.
        # Only a semantic embedder can make that judgement - the hash embedder
        # scores "tech"/"technical" at 0.14, so with it the conjunction would
        # simply disable the detector. Zero-key mode keeps centroids alone.
        labels: dict[str, Any] | None = None
        if self.embedder.dimensions and centroids and self.embedder.name != "hash":
            names = sorted(centroids)
            try:
                labels = dict(zip(names, self.embedder.embed(names)))
            except Exception:
                labels = None
        return semantic_duplicate_tags(
            centroids, counts, cooccurrence, labels=labels, threshold=threshold
        )

    def tag_health(self, *, user_id: str | None = None) -> dict[str, Any]:
        """Cheap, deterministic signals on how well the tag vocabulary is holding.

        Fragmentation is the failure mode that costs recall silently: filtering
        to a tag that has split its subject drops the memories the question
        needed, before ranking ever runs. Nothing surfaces that today unless
        somebody presses "suggest merges", so a store can drift for months.

        No LLM, no writes. Everything here comes from counts already indexed and
        vectors already stored.
        """
        scope = Scope(user_id=user_id)
        counts = self.backend.topic_mention_counts(scope) or []
        total = len(self.get_all(user_id=user_id, limit=1_000_000))
        tagged = len({mid for _, mid in (self.backend.topic_mention_links(scope) or [])})
        singles = sum(1 for row in counts if row.get("count") == 1)
        splits = self.semantic_tag_duplicates(user_id=user_id)
        return {
            "memories": total,
            "untagged": max(0, total - tagged),
            "tags": len(counts),
            "single_use_tags": singles,
            "single_use_share": round(singles / len(counts), 3) if counts else 0.0,
            "suspected_splits": len(splits),
            "splits": splits[:10],
            "largest_tags": [
                {"tag": row["category"], "count": row["count"]} for row in counts[:5]
            ],
        }

    def suggest_tag_merges(self, *, user_id: str | None = None) -> list[dict[str, Any]]:
        """Suggest duplicate tags: spelling variants, then synonyms, then splits.

        Three detectors, cheapest first, each catching what the previous cannot:
        deterministic inflection, an LLM synonym pass, and vector-centroid
        overlap for the near-synonyms that share no words ("liver bloods" beside
        "liver lab results").
        """
        tags = self.categories(user_id=user_id)
        proposals = suggest_canonical_merges(self.llm, tags)
        seen = {v for group in proposals for v in group["variants"]}
        for pair in self.semantic_tag_duplicates(user_id=user_id):
            if not seen.intersection(pair["variants"]):
                proposals.append(pair)
                seen.update(pair["variants"])
        # A fourth pass for the synonyms the three above miss. Suggestion only:
        # every one of these still needs confirming under Upkeep.
        names = [str(t["category"]).strip().lower() for t in tags]
        names = [n for n in names if n and n not in seen]
        candidates = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
        if candidates and len(candidates) <= 200:
            for a, b in judge_tag_pairs(self.decider, candidates):
                if a in seen or b in seen:
                    continue
                proposals.append({"canonical": a, "variants": [a, b],
                                  "reason": f"{self.decider.name}: same meaning"})
                seen.update({a, b})
        return proposals

    # -- manual tag curation -----------------------------------------------
    def rename_tag(self, tag: str, to: str, *, user_id: str | None = None) -> int:
        """Rename one tag to another across every memory. Returns the count."""
        return self._retag(user_id, {tag.strip().lower()}, to.strip().lower())

    def merge_tags(self, tags: list[str], to: str, *, user_id: str | None = None) -> int:
        """Combine several tags into one across every memory."""
        remove = {t.strip().lower() for t in tags if t.strip()}
        return self._retag(user_id, remove, to.strip().lower())

    def delete_tag(self, tag: str, *, user_id: str | None = None) -> int:
        """Remove a tag from every memory (the memories stay)."""
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

        Once a tag is curated by hand its synthetic marker is dropped: the tag
        is now the user's, not the system's guess.
        """
        remove = {r for r in remove if r}
        if not remove:
            return 0
        if user_id is None and not exact_user:
            namespaces = self.backend.tag_namespaces(remove)
            if namespaces is not None:
                return sum(self._retag(namespace, remove, add, exact_user=True)
                           for namespace in namespaces)
        if add is not None:
            # The new name is a tag like any other, so it is held to the same
            # shape; one that cleans away to nothing is a plain removal.
            add = next(iter(clean_tags(add)), None)
        scope = Scope(user_id=user_id)
        indexed = self._merge_topics(user_id, remove, add, exact_user=exact_user)
        if indexed is not None:
            for tag in remove:
                self.backend.delete_synthetic_tag(scope, tag)
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
        scope = Scope(user_id=user_id)
        for tag in remove:
            self.backend.delete_synthetic_tag(scope, tag)
        return changed

    # -- maintenance switches ----------------------------------------------
    # The scheduler's passes were config-only, which meant turning one off was
    # an env-var edit and a restart. A runtime override lives in the meta table
    # so the dashboard toggle survives restarts; config stays the default when
    # no override was ever set.
    _MAINTENANCE_KEYS = (
        "dedup_entities", "tag_abstraction", "durability", "consolidation", "structure",
    )

    #: How many memories one durability pass scores. Jev answers 128 questions
    #: in a single call, so the batch is bounded by prudence, not by cost.
    DURABILITY_BATCH = 64

    def score_memory_durability(
        self, *, user_id: str | None = None, limit: int | None = None
    ) -> dict[str, Any]:
        """Record how long each memory is worth keeping, for memories missing it.

        Decay runs on a half-life per memory type, which treats "the train was
        delayed this morning" and "allergic to penicillin" the same because both
        are semantic. A per-fact estimate replaces that guess; anything still
        unscored keeps the old behaviour.

        Off unless ``decay.durability`` is set, whoever asks (the scheduler,
        "run now", the REST route). The score is housekeeping: it is written
        without moving the memory's ``updated_at``, which drives recency and
        decay age.
        """
        outcome: dict[str, Any] = {"scored": 0, "skipped": 0, "provider": self.decider.name}
        if not self.config.decay.durability:
            outcome["skipped"] = -1
            outcome["reason"] = self._pass_off_reason("durability")
            return outcome
        if not self.decider.available:
            outcome["skipped"] = -1
            log.info("durability: no decision provider configured, nothing scored")
            return outcome
        batch = limit or self.DURABILITY_BATCH
        pending = [
            m for m in self.get_all(user_id=user_id, limit=100_000)
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

    def _pass_off_reason(self, key: str) -> str:
        """Why a pass that is off (``maintenance_enabled``) is off."""
        if key == "durability" and not self.config.decay.durability:
            return "decay.durability is off (MEMRY_DURABILITY)"
        if key == "tag_abstraction" and not self.config.tags.enabled:
            return "tag abstraction is off (MEMRY_TAG_ABSTRACTION)"
        return "this pass is off; turn it on under Upkeep to run it"

    def maintenance_enabled(self, key: str) -> bool:
        if key == "tag_abstraction" and not self.config.tags.enabled:
            # Synthetic parent tags are off unless the config switches them on
            # (MEMRY_TAG_ABSTRACTION); a stored toggle alone cannot.
            return False
        if key == "durability" and not self.config.decay.durability:
            # Likewise the durability pass (MEMRY_DURABILITY).
            return False
        override = self.backend.get_meta(f"maintenance:{key}:enabled")
        if override is not None:
            return override == "true"
        if key == "dedup_entities":
            return self.config.dedup_entities
        if key == "tag_abstraction":
            return self.config.tags.enabled and self.llm.available
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

    def _stamp_tag_run(self, user_id: str | None) -> None:
        self.backend.set_meta(_tag_run_key(user_id), utcnow())

    def last_tag_run(self, user_id: str | None) -> str | None:
        return self.backend.get_meta(_tag_run_key(user_id))

    # ------------------------------------------------------------------
    # entity structure: hubs, homes and shared names (intelligence/structure.py)
    # ------------------------------------------------------------------
    #: Names screened per pass; one typed question each, asked in parallel.
    SCREEN_BATCH = 1000

    def _structure_inputs(self, user_id: str | None):
        scope = Scope(user_id=user_id)
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
        self, *, user_id: str | None = None, dry_run: bool = False
    ) -> dict[str, Any]:
        """Record where parts belong, and settle names that are shared.

        Nothing is deleted. A home is a note in the entity's metadata,
        recomputed every pass. A merge sets ``merged_into`` on the entity with
        less evidence, exactly as a confirmed proposal does. ``dry_run=True``
        changes nothing and returns the full plan instead.
        """
        entities, nodes, triples, judged = self._structure_inputs(user_id)
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

        plan = same_name_plan(nodes, homes, self._held_apart(Scope(user_id=user_id)))
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
        self, *, user_id: str | None = None, limit: int | None = None
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
        scope = Scope(user_id=user_id)
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
        at: datetime | None = None,
    ) -> dict[str, Any]:
        """Run one pass now, with the same code the scheduler uses.

        ``at`` is the tick the scheduler is working through, which is what the
        run is stamped with. Stamping the wall clock instead put the next run
        due at a different time than the tick that triggered it.

        A pass that is off (``maintenance_enabled``) runs nowhere: neither the
        scheduler nor "run now" starts it, and no run is recorded; the result
        says so (``ran`` False, with the reason).
        """
        if key not in self._MAINTENANCE_KEYS:
            raise ValueError(f"unknown pass: {key}")
        if not self.maintenance_enabled(key):
            return {"ran": False, "reason": self._pass_off_reason(key)}
        with self._pass_lock(key, user_id):
            stamp = at.isoformat(timespec="seconds") if at is not None else utcnow()
            if key == "dedup_entities":
                self.merge_obvious_topics(user_id=user_id)
                result = self.resolve_entities(user_id=user_id)
                # A calibrated provider judges names one by one in their memory; the
                # text model's batch review is the fallback when there is none.
                if self.decider.available:
                    result.update(self.run_name_screen(user_id=user_id))
                elif self.llm.available:
                    result.update(self.run_entity_review(user_id=user_id))
                self.backend.set_meta(_dedup_run_key(user_id), stamp)
            elif key == "structure":
                result = self.run_structure_pass(user_id=user_id)
            elif key == "tag_abstraction":
                result = self.abstract_tags(user_id=user_id)
            elif key == "durability":
                result = self.score_memory_durability(user_id=user_id)
            elif key == "consolidation":
                result = self.run_consolidation_pass(user_id=user_id)
                self.backend.set_meta(_consolidation_run_key(user_id), stamp)
            else:
                raise ValueError(f"unknown pass: {key}")
            if record:
                self._upkeep_set(f"last:{key}", user_id, {"at": stamp, "result": result})
            return result

    def run_upkeep_cycle(
        self, *, user_id: str | None = None, now: datetime | None = None
    ) -> dict[str, Any]:
        """One scheduler tick for one namespace: every pass that is on and due.

        Returns what ran, keyed by pass, so the scheduler can spread work over
        cycles on a many-account server.
        """
        now = now or datetime.now(timezone.utc)
        ran: dict[str, Any] = {}
        if self.upkeep_paused():
            return ran
        every = self.config.dedup_interval_days
        dedup_due = _due(self.backend.get_meta(_dedup_run_key(user_id)), every, now)
        if self.maintenance_enabled("structure") and dedup_due:
            ran["structure"] = self.run_upkeep_pass(
                "structure", user_id=user_id, at=now)
        if self.maintenance_enabled("dedup_entities") and dedup_due:
            ran["dedup_entities"] = self.run_upkeep_pass(
                "dedup_entities", user_id=user_id, at=now)
        if (
            self.maintenance_enabled("tag_abstraction") and self.llm.available
            and _due(self.last_tag_run(user_id), self.config.tags.interval_days, now)
        ):
            ran["tag_abstraction"] = self.run_upkeep_pass(
                "tag_abstraction", user_id=user_id, at=now)
        if (
            self.maintenance_enabled("consolidation") and self.llm.available
            and _due(self.backend.get_meta(_consolidation_run_key(user_id)), every, now)
        ):
            ran["consolidation"] = self.run_upkeep_pass(
                "consolidation", user_id=user_id, at=now)
        if self.maintenance_enabled("durability") and self.decider.available:
            # Cheap when nothing is unscored, so it runs every tick; only a
            # tick that scored something is worth remembering as a run.
            result = self.run_upkeep_pass("durability", user_id=user_id, record=False)
            if result.get("scored"):
                self._upkeep_set("last:durability", user_id,
                                 {"at": now.isoformat(timespec="seconds"), "result": result})
                ran["durability"] = result
        if dedup_due:
            # After this week's merges and new homes: re-embed the memories
            # whose masked names changed. Nothing to embed is no run.
            try:
                embedded = self.refresh_property_vectors(user_id=user_id)
            except Exception as exc:
                log.warning("property vector refresh failed: %s", exc)
                embedded = 0
            if embedded:
                ran["property_vectors"] = {"embedded": embedded}
        return ran

    def run_consolidation_pass(self, *, user_id: str | None = None) -> dict[str, Any]:
        """Merge exact duplicates on sight; queue what only a model vouched for.

        Entity identity was measured against a labelled set before it was
        allowed to merge on its own; a model's "same fact" has not been, so
        those merges wait for a person under Upkeep. Every judged group is
        remembered, so the model is asked about each one once.
        """
        with self._pass_lock("consolidation", user_id):
            repaired = self.repair_consolidated(user_id=user_id)
            seen_lists = self._upkeep_get("consolidation:seen", user_id, [])
            seen = {frozenset(ids) for ids in seen_lists}
            result = self.consolidate_memories(
                user_id=user_id, threshold=self.UPKEEP_CONSOLIDATION_THRESHOLD,
                apply=False, exclude=seen,
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

    def run_entity_review(self, *, user_id: str | None = None) -> dict[str, Any]:
        """Ask the model which concept names are not things; queue its verdicts.

        Nothing is removed here. A name the user chose to keep is never asked
        about again.
        """
        kept = set(self._upkeep_get("entity_review:kept", user_id, []))
        judged = self.entity_junk(user_id=user_id, judge=True)["judged"]
        pending = [
            {"id": j["id"], "name": j["name"]} for j in judged if j["id"] not in kept
        ]
        self._upkeep_set("entity_review:pending", user_id, pending)
        return {"reviewed": len(judged), "queued": len(pending)}

    def upkeep_queue(
        self, *, user_id: str | None = None, tag_health: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
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
                "detail": ("This updates a memory you already have, and no merged "
                           "text was written for the two." if update
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

        health = tag_health if tag_health is not None else self.tag_health(user_id=user_id)
        splits_listed = 0
        ignored = {
            tuple(sorted(pair)) for pair in self._upkeep_get("tag_split:ignored", user_id, [])
        }
        for split in health.get("splits", []):
            a, b = split["variants"]
            if tuple(sorted((a, b))) in ignored:
                continue
            splits_listed += 1
            items.append({
                "kind": "tag_split", "id": _group_id([a, b]),
                "title": f"#{a} and #{b}",
                "detail": "Look like one subject split in two, which caps what a "
                          f"search under either can find (similarity {split['similarity']}).",
                "accept": f"combine into #{split['canonical']}", "decline": "keep apart",
            })
        if self._upkeep_get("tag_split:count", user_id, None) != splits_listed:
            self._upkeep_set("tag_split:count", user_id, splits_listed)
        return items

    def proposals_for_a_person(self, user_id: str | None) -> list[MergeProposal]:
        """Entity pairs Upkeep asks a person about. With a calibrated judge,
        none: a pair it could not settle waits for new evidence and is
        compared again, since a person would be guessing from the same facts."""
        if judges_pairs(self.decider):
            return []
        return self.merge_proposals(user_id=user_id, limit=1000)

    def upkeep_count(self, *, user_id: str | None = None) -> int:
        """How many rows wait under Upkeep, without asking a model anything.

        The dashboard shows this as a badge on every load, so it must not cost
        what the full queue costs: tag health embeds every tag name. The split
        count is therefore the one the last full look at the queue found.
        """
        waiting = {entity.id for entity, _ in self._screen_rows(user_id)}
        waiting |= {p["id"] for p in self._upkeep_get("entity_review:pending", user_id, [])}
        return (
            len(self.proposals_for_a_person(user_id))
            + len(self._upkeep_get("conflict:pending", user_id, []))
            + len(self._upkeep_get("consolidation:pending", user_id, []))
            + len(waiting)
            + int(self._upkeep_get("tag_split:count", user_id, 0) or 0)
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
        if kind == "tag_split":
            for split in self.tag_health(user_id=user_id).get("splits", []):
                a, b = split["variants"]
                if _group_id([a, b]) != item_id:
                    continue
                if accept:
                    drop = [v for v in (a, b) if v != split["canonical"]]
                    self.merge_tags(drop, split["canonical"], user_id=user_id)
                    return True
                ignored = self._upkeep_get("tag_split:ignored", user_id, [])
                ignored.append(sorted((a, b)))
                self._upkeep_set("tag_split:ignored", user_id, ignored)
                return True
            return False
        return False

    # ------------------------------------------------------------------
    # maintenance
    # ------------------------------------------------------------------
    def decay_sweep(self, threshold: float = 0.1) -> list[str]:
        return decay_sweep(self.backend, self.config.decay, threshold=threshold)

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
        for user_id in self.backend.distinct_user_ids() or [None]:
            self.refresh_property_vectors(user_id=user_id)
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
        """The automatic-merge gate in force: the provider's own while it can
        answer, else the text model's. Above 1.0 means merges never happen on
        their own."""
        return _gate(self.decider, self.llm)

    def count_memories(self, *, owner_prefix: str | None = None) -> dict[str, int]:
        """Active, invalidated and forgotten counts; see ``MemoryBackend.count_memories``."""
        return self.backend.count_memories(owner_prefix)

    def reset(self) -> None:
        self.backend.reset()

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
