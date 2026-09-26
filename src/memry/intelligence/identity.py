"""Entity identity with a calibrated judge: which pairs to ask about, and how.

Two stages, and neither lists forms of names.

**Finding pairs.** Two entity names are worth a question when they share a word
that is rare among the store's own entity names, when they are spelled alike
(shared letter trigrams, or a small edit distance for a typo),
when one is the initial letters of the other, or when a semantic embedder puts
the two names close together. Rarity comes from the store: a word such as
"GmbH", "Ltd" or "Dr." that many names carry stops counting on its own, and
nobody has to list it. On 156 labelled pairs these four signals found all 32
true pairs whose names differ (legal forms from six countries, titles,
nicknames, acronyms, translations) and proposed 40 other pairs out of 8,385.

**Deciding a pair.** The judge sees both entities side by side, each with its
name, type, description and facts, and answers one question with three
options. The question is asked in both orders and the probabilities are
averaged, because the answer must not depend on which entity came first. Asked
in one order, the threshold that merged nothing wrong moved from 0.90 in one
run to 0.95 in the next, and with the first wording tried, swapping the two
entities left no safe threshold at all. Measured with Jev over the 156 pairs
in ``evals/identity_resolution_benchmark.py``, five runs:

* a pair merges from P(same) = 0.95. That merged 57-59 of 81 true pairs and no
  wrong one. No pair of two different things scored above 0.79; the pairs
  between 0.79 and 0.95 that are not true pairs are ones nobody could settle
  from the facts ("PR #92" twice with no repository, "R. Patel" twice). A
  deployment can lower it (``DecisionConfig.pair_merge_probability``): at
  0.85, 73-74 true pairs merged, and so did 4 of those unsettleable pairs;
* a pair is kept apart from P(different) = 0.5. No true pair scored above 0.43;
* anything else is compared a second time with all of both entities'
  memories instead of the most recent eight, and if that still settles
  nothing it waits for new evidence. Nobody is asked: the pair is compared
  again whenever a new memory mentions either side.

Every fact carries its date (when it became true where that is known, else
when it was recorded). Without dates, "lives in Munich" against "moved to
Amsterdam last month" read as two people (P(same) 0.54), and a promotion as
two roles (0.78); with dates, 0.97 and 1.00. Over the 156 pairs, two runs,
dates raised the true merges at 0.90 from 67-68 to 72-73, with no pair of
two things merged and one unsettleable pair instead of three.

The question Memry asked before (the existing entity's facts against one new
fact, decided on Jev's own confidence) merged nothing safely on the same pairs.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np

from ..backends.base import MemoryBackend
from ..models import Entity, Memory
from ..providers.decisions import Choice, Decider

PAIR_QUESTION = Choice(
    instructions=(
        "In one person's memory store, two entries that carry one name are "
        "usually one thing. Do entity A and entity B refer to the same "
        "real-world person, organization, place or thing?"
    ),
    criteria={
        "same": (
            "One thing: the names are one name, as written or written "
            "differently (legal form, title, abbreviation, acronym, nickname, "
            "translation, domain or spelling), and no fact contradicts one "
            "thing. Facts about unrelated topics are normal for one thing."
        ),
        "different": (
            "Two things: a fact contradicts one thing (different people, ages, "
            "roles, places, dates, sizes, owners, versions, containers or kinds "
            "of thing), or the names only share a word, a surname or a number."
        ),
        "unsure": (
            "Nothing settles it: the name alone does not pick out one thing (a "
            "first name, a surname, a role, or a number that is only unique "
            "inside a repository, tracker or file the facts do not name), and no "
            "fact links or separates them."
        ),
    },
)

#: A word carried by more than this share of the store's entity names (and by
#: more than two) is too common to pair two names on its own.
RARE_WORD_SHARE = 0.02
#: Character-trigram overlap from which two names count as spelled alike
#: ("OpenAI" and "Open AI", "PostgreSQL" and "Postgres").
SPELLING_SIMILARITY = 0.5
#: Share of characters that may change, by edit distance, for two names to
#: count as spelled alike ("colonge" and "cologne"). Trigrams miss a typo that
#: swaps two letters, since the swap breaks the trigrams around it.
EDIT_SIMILARITY = 0.8
#: Cosine between name embeddings from which two names count as close in
#: meaning ("Köln" and "Cologne" scored 0.68, unrelated names rarely 0.6).
MEANING_SIMILARITY = 0.6
#: Most candidates compared for one name, best first.
CANDIDATES_PER_NAME = 5
#: Facts per side in the first comparison, most recent first.
PROFILE_FACTS = 8
#: Facts per side in the second comparison, for a pair the first left waiting.
#: Jev reads the whole list: one contradicting fact placed last among 100 facts
#: still gave P(different) = 0.95, and last among 300 gave 0.86.
FULL_PROFILE_FACTS = 200


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.casefold())
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def name_tokens(name: str) -> list[str]:
    return re.findall(r"[^\W_]+", _fold(name or ""))


def _grams(name: str) -> set[str]:
    compact = "".join(name_tokens(name))
    return {compact[i:i + 3] for i in range(len(compact) - 2)} or {compact}


def edit_similarity(a: str, b: str) -> float:
    """1 minus the optimal-string-alignment distance over the longer length,
    on the names' letters and digits only."""
    a, b = "".join(name_tokens(a)), "".join(name_tokens(b))
    if not a or not b:
        return 0.0
    rows = [list(range(len(b) + 1))]
    for i in range(1, len(a) + 1):
        row = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            row[j] = min(rows[-1][j] + 1, row[j - 1] + 1,
                         rows[-1][j - 1] + (a[i - 1] != b[j - 1]))
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                row[j] = min(row[j], rows[-2][j - 2] + 1)
        rows = rows[-1:] + [row]
    return 1 - rows[-1][-1] / max(len(a), len(b))


def is_acronym_of(short: str, long: str) -> bool:
    """Whether ``short`` is built from the letters of ``long`` in order,
    starting with its first letter: "AWS" and "Amazon Web Services", "BSFZ"
    and "Bescheinigungsstelle Forschungszulage"."""
    parts, words = name_tokens(short), name_tokens(long)
    if len(parts) != 1 or len(words) < 2:
        return False
    letters, text = parts[0], "".join(words)
    if not (2 <= len(letters) <= 6) or len(letters) < len(words) or letters[0] != text[0]:
        return False
    rest = iter(text)
    return all(letter in rest for letter in letters)


class NameIndex:
    """The store's entity names, indexed for finding the ones worth comparing
    with a given name."""

    def __init__(
        self, entities: Iterable[Entity], vectors: dict[str, np.ndarray] | None = None
    ) -> None:
        self.entities = {e.id: e for e in entities if e.merged_into is None}
        self.vectors = vectors or {}
        self._tokens = {eid: set(name_tokens(e.name)) for eid, e in self.entities.items()}
        self._grams = {eid: _grams(e.name) for eid, e in self.entities.items()}
        counts = Counter(t for tokens in self._tokens.values() for t in tokens)
        self._rare_at = max(2, int(RARE_WORD_SHARE * len(self.entities)))
        self._by_token: dict[str, set[str]] = defaultdict(set)
        self._by_gram: dict[str, set[str]] = defaultdict(set)
        self._short: list[str] = []
        for eid, tokens in self._tokens.items():
            for token in tokens:
                if len(token) >= 3 and counts[token] <= self._rare_at:
                    self._by_token[token].add(eid)
            for gram in self._grams[eid]:
                self._by_gram[gram].add(eid)
            if len(tokens) == 1:
                self._short.append(eid)
        self._counts = counts
        self._vector_ids = [eid for eid in self.entities if vectors and eid in vectors]
        self._matrix = (
            np.vstack([vectors[eid] for eid in self._vector_ids]) if self._vector_ids else None
        )

    def candidates(
        self,
        name: str,
        *,
        vector: np.ndarray | None = None,
        exclude: Iterable[str] = (),
        limit: int = CANDIDATES_PER_NAME,
    ) -> list[Entity]:
        excluded = set(exclude)
        tokens, grams = set(name_tokens(name)), _grams(name)
        score: dict[str, float] = defaultdict(float)
        for token in tokens:
            if len(token) >= 3 and self._counts.get(token, 0) <= self._rare_at:
                for eid in self._by_token.get(token, ()):
                    score[eid] += 1.0
        overlap: Counter[str] = Counter(
            eid for gram in grams for eid in self._by_gram.get(gram, ())
        )
        for eid, shared in overlap.items():
            similarity = shared / len(grams | self._grams[eid])
            if similarity >= SPELLING_SIMILARITY:
                score[eid] += 1.0 + similarity
            elif edit_similarity(name, self.entities[eid].name) >= EDIT_SIMILARITY:
                score[eid] += 1.0
        if len(tokens) == 1:
            for eid, entity in self.entities.items():
                if is_acronym_of(name, entity.name):
                    score[eid] += 1.0
        else:
            for eid in self._short:
                if is_acronym_of(self.entities[eid].name, name):
                    score[eid] += 1.0
        if vector is not None and self._matrix is not None:
            cosine = self._matrix @ vector
            for i in np.flatnonzero(cosine >= MEANING_SIMILARITY):
                score[self._vector_ids[i]] += float(cosine[i])
        folded = _fold(name).strip()
        ranked = sorted(
            (eid for eid in score
             if eid not in excluded and _fold(self.entities[eid].name).strip() != folded),
            key=lambda eid: -score[eid],
        )
        return [self.entities[eid] for eid in ranked[:limit]]


@dataclass
class Source:
    """Where one fact came from, passed to the judge as it is stored."""

    recorded: str = ""   # when it was saved, "YYYY-MM-DD HH:MM"
    true_from: str = ""  # when the fact became true, if known
    text: str = ""       # the saved text it was extracted from
    session: str = ""    # the session that saved it
    client: str = ""     # the client that saved it
    context: str = ""    # the context label it was saved under


def source_of(memory: Memory) -> Source:
    return Source(
        recorded=(memory.created_at or "")[:16].replace("T", " "),
        true_from=(memory.valid_from or "")[:10],
        text=memory.source_episode_ids[0] if memory.source_episode_ids else "",
        session=memory.run_id or "",
        client=memory.agent_id or "",
        context=" ".join(str((memory.metadata or {}).get("context") or "").split())[:120],
    )


@dataclass
class Profile:
    """One side of a comparison: what the store knows about an entity."""

    name: str
    entity_type: str | None = None
    facts: list[str] = field(default_factory=list)
    description: str = ""
    home: str = ""
    #: The date of each fact (YYYY-MM-DD), parallel to ``facts``, for callers
    #: that only have dates (the benchmark). Without dates, "lives in Munich"
    #: against "moved to Amsterdam last month" read as two people (0.54).
    dates: list[str] = field(default_factory=list)
    #: Where each fact came from, parallel to ``facts``. Shown as stored; the
    #: judge decides what a shared saved text or session means.
    sources: list[Source] = field(default_factory=list)
    #: Whether this entity is the owner of the store (the user).
    owner: bool = False


def profile_of(
    backend: MemoryBackend, entity: Entity, limit: int = PROFILE_FACTS
) -> tuple[Profile, bool]:
    """The entity's most recent ``limit`` facts, and whether it has more."""
    memories = backend.entity_memories(entity.id, limit=limit + 1)
    home = (entity.metadata or {}).get("home")
    profile = Profile(
        name=entity.name,
        entity_type=entity.entity_type,
        facts=[m.content for m in memories[:limit]],
        description=entity.description or "",
        home=home.get("name", "") if isinstance(home, dict) else "",
        sources=[source_of(m) for m in memories[:limit]],
        owner=bool((entity.metadata or {}).get("owner")),
    )
    return profile, len(memories) > limit


def pair_state(a: Profile, b: Profile) -> str:
    texts: dict[str, int] = {}
    sessions: dict[str, int] = {}
    for profile in (a, b):
        for source in profile.sources:
            if source.text:
                texts.setdefault(source.text, len(texts) + 1)
            if source.session:
                sessions.setdefault(source.session, len(sessions) + 1)

    def origin(p: Profile, i: int) -> str:
        if i < len(p.sources):
            src = p.sources[i]
            parts = [f"recorded {src.recorded}"] if src.recorded else []
            if src.true_from and src.true_from != src.recorded[:10]:
                parts.append(f"true from {src.true_from}")
            if src.text:
                parts.append(f"saved text {texts[src.text]}")
            if src.session:
                parts.append(f"session {sessions[src.session]}")
            if src.client:
                parts.append(f"client {src.client}")
            if src.context:
                parts.append(f'context "{src.context}"')
            return "; ".join(parts)
        return p.dates[i] if i < len(p.dates) else ""

    def side(label: str, p: Profile) -> str:
        lines = [f'ENTITY {label}: "{p.name}" ({p.entity_type or "type unknown"})']
        if p.owner:
            lines.append("This entity is the owner of the memory store: the person the "
                         "memories belong to.")
        if p.home:
            lines.append(f"Part of: {p.home}")
        if p.description:
            lines.append(f"Description: {p.description}")
        lines.append("Facts:")
        for i, fact in enumerate(p.facts):
            where = origin(p, i)
            lines.append(f"- [{where}] {fact}" if where else f"- {fact}")
        lines += [] if p.facts else ["- (no facts)"]
        return "\n".join(lines)

    if any(p.sources for p in (a, b)):
        header = (" Each fact is shown with where it came from: when it was recorded, "
                  "when it became true if known, the saved text it was extracted from, "
                  "the session and client that saved it, and the context label it was "
                  "saved under. A saved text or session has the same number on both sides.")
    elif any(p.dates for p in (a, b)):
        header = " Each fact starts with the date it was recorded."
    else:
        header = ""
    return ("Two entries from one person's long-term memory store." + header
            + "\n\n" + side("A", a) + "\n\n" + side("B", b))


def judge_pair(decider: Decider, a: Profile, b: Profile) -> dict[str, float] | None:
    """P(same), P(different) and P(unsure), averaged over both orders. None
    when the judge did not answer both."""
    def ask(state: str):
        return decider.decide(state, {"pair": PAIR_QUESTION})["pair"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        answers = list(pool.map(ask, (pair_state(a, b), pair_state(b, a))))
    if not all(answer.available and answer.probabilities for answer in answers):
        return None
    return {
        option: sum(answer.probabilities.get(option, 0.0) for answer in answers) / 2
        for option in PAIR_QUESTION.criteria
    }


def decide_pair(probabilities: dict[str, float], decider: Decider) -> str:
    """"merge", "apart" or "wait", on the provider's measured thresholds."""
    if probabilities["same"] >= decider.pair_merge_probability:
        return "merge"
    if probabilities["different"] >= decider.pair_apart_probability:
        return "apart"
    return "wait"


def compare(
    decider: Decider, backend: MemoryBackend, a: Entity, b: Entity | Profile
) -> tuple[dict[str, float] | None, str]:
    """Decide a pair: the recent facts first, then, when that leaves the pair
    waiting and either side has more, all of both entities' memories.

    ``b`` is an entity, or the profile of a mention a save has not stored yet.
    Returns the probabilities the decision rests on and the decision.
    """
    first_a, more_a = profile_of(backend, a)
    if isinstance(b, Entity):
        first_b, more_b = profile_of(backend, b)
    else:
        first_b, more_b = b, False
    probabilities = judge_pair(decider, first_a, first_b)
    if probabilities is None:
        return None, "wait"
    action = decide_pair(probabilities, decider)
    if action == "wait" and (more_a or more_b):
        full_a = profile_of(backend, a, FULL_PROFILE_FACTS)[0] if more_a else first_a
        full_b = profile_of(backend, b, FULL_PROFILE_FACTS)[0] if more_b else first_b
        full = judge_pair(decider, full_a, full_b)
        if full is not None:
            probabilities, action = full, decide_pair(full, decider)
    return probabilities, action


def pair_reason(decider: Decider, probabilities: dict[str, float]) -> str:
    return f"{decider.name}: {max(probabilities, key=probabilities.get)}"


def judges_pairs(decider: Decider | None) -> bool:
    """Whether this provider decides identity pairs: it must answer with
    probabilities it computed. A text model's self-reported confidence was 0.9
    on its wrong answers too, and merged 8 of 81 true pairs at its safe
    threshold."""
    return decider is not None and decider.available and decider.calibrated


def name_vectors(
    embed: Callable[[list[str]], list[list[float]]] | None, entities: Iterable[Entity]
) -> dict[str, np.ndarray] | None:
    """Unit vectors for entity names, for pairing names close in meaning."""
    if embed is None:
        return None
    entities = list(entities)
    try:
        raw = embed([e.name for e in entities]) if entities else []
    except Exception:  # an embedding outage costs the meaning signal, nothing else
        return None
    out: dict[str, np.ndarray] = {}
    for entity, vector in zip(entities, raw):
        arr = np.asarray(vector, dtype=float)
        norm = np.linalg.norm(arr)
        if norm:
            out[entity.id] = arr / norm
    return out


def parallel(fn: Callable[[Any], Any], items: list[Any], workers: int = 8) -> list[Any]:
    if len(items) <= 1:
        return [fn(item) for item in items]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, items))


# -- tags ------------------------------------------------------------------
#: Most recent memories shown per tag. Measured on 379 candidate pairs from a
#: real store: with the names alone "memry" read as a typo of "memory" (0.98);
#: with two example memories "colonge" fell to 0.20 because one example cannot
#: show what a tag is used for; with 10 no pair of two subjects scored above
#: 0.46, and 25 were no better.
TAG_EXAMPLES = 10

TAG_QUESTION = Choice(
    instructions=(
        "Two tags that file memories in one person's memory store, each shown with "
        "memories filed under it. Do tag A and tag B name the same subject, so that "
        "every memory filed under one belongs under the other?"
    ),
    criteria={
        "same": (
            "One subject: the same tag written differently (spelling, typo, format, "
            "singular or plural, abbreviation, acronym, translation, legal form or web "
            "domain) or a synonym, and the memories under both are about that subject."
        ),
        "different": (
            "Two subjects: unrelated subjects, related subjects, or one tag is a part, "
            "kind, aspect or detail of the other, as \"insurance\" and \"insurance "
            "contract\"."
        ),
    },
)


def tag_state(a: str, b: str, counts: dict[str, int],
              known: dict[str, tuple[str, str | None]],
              examples: dict[str, list[str]]) -> str:
    """Both tags with how often each is used, the entity of that name where the
    store has one, and the memories most recently filed under each."""
    def side(label: str, tag: str) -> str:
        lines = [f'TAG {label}: "{tag}" (on {counts.get(tag, 0)} memories)']
        if tag in known:
            name, kind = known[tag]
            lines.append(f'This store has a {kind or "thing"} named "{name}".')
        shown = examples.get(tag, [])[:TAG_EXAMPLES]
        lines.append(f"The {len(shown)} most recent memories filed under it:")
        lines += [f"- {content}" for content in shown]
        return "\n".join(lines)

    return ("Two tags from one person's long-term memory store.\n\n"
            + side("A", a) + "\n\n" + side("B", b))


def judge_tag_pair(decider: Decider, a: str, b: str, counts: dict[str, int],
                   known: dict[str, tuple[str, str | None]],
                   examples: dict[str, list[str]]) -> float | None:
    """P(same subject), averaged over both orders."""
    def ask(state: str):
        return decider.decide(state, {"tag": TAG_QUESTION})["tag"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        answers = list(pool.map(ask, (tag_state(a, b, counts, known, examples),
                                      tag_state(b, a, counts, known, examples))))
    if not all(answer.available and answer.probabilities for answer in answers):
        return None
    return sum(answer.probabilities.get("same", 0.0) for answer in answers) / 2


def judged_tag_merges(
    decider: Decider,
    tags: list[dict[str, Any]],
    known: dict[str, tuple[str, str | None]],
    memories_of: Callable[[str], list[str]],
    vectors: dict[str, np.ndarray] | None = None,
    limit: int = 400,
) -> list[dict[str, Any]]:
    """Groups of tags the judge puts at ``decider.tag_merge_probability`` or
    higher, each kept under its most used tag. ``memories_of(tag)`` returns the
    tag's most recent memories, most recent first.

    Measured on the 379 candidate pairs of a real 417-tag store, 10 memories per
    tag, two runs: from 0.55 it merged 7-9 of the 16 pairs I labelled one
    subject ("fundation" and "fundation gmbh" at 0.86-0.88, "bildy" and
    "bildy.ai", "steuer" and "tax", "cologne" and "colonge") and none of the
    41 borderline or 322 two-subject pairs; the highest two-subject pair was
    "restart" and "shutdown" at 0.46.
    """
    counts = {str(t["category"]).strip().casefold(): int(t.get("count") or 0) for t in tags}
    labels = sorted(counts)
    nodes = [Entity(id=label, name=label, user_id=None) for label in labels]
    index = NameIndex(nodes, vectors)
    pairs = sorted({
        tuple(sorted((label, other.name)))
        for label in labels
        for other in index.candidates(label, vector=(vectors or {}).get(label))
    })[:limit]
    examples = {tag: memories_of(tag) for tag in sorted({t for pair in pairs for t in pair})}
    scores = parallel(
        lambda pair: judge_tag_pair(decider, *pair, counts, known, examples), pairs
    )
    parent = {label: label for label in labels}

    def root(label: str) -> str:
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = parent[label]
        return label

    for (a, b), score in zip(pairs, scores):
        if score is not None and score >= decider.tag_merge_probability:
            parent[root(a)] = root(b)
    groups: dict[str, list[str]] = defaultdict(list)
    for label in labels:
        groups[root(label)].append(label)
    return [
        {"canonical": max(members, key=lambda t: (counts[t], -len(t))), "variants": sorted(members),
         "reason": f"{decider.name}: same subject"}
        for members in groups.values() if len(members) > 1
    ]
