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

* a pair merges from a P(same) that falls with the evidence: the provider's
  ``pair_merge_by_step``, per step of the funnel below. On 18,885 comparisons
  of a new name against the entity it may belong to (synthetic stores with
  exact labels), keeping wrong merges at or under 2% of merges needed about
  0.97 with one memory on the smaller side, 0.85-0.96 with three, 0.79-0.84
  with eight and 0.78-0.79 with fifteen to thirty. At one memory Jev's
  P(same) is close to the real share; with more it is too cautious, by about
  half. A deployment can set one bar for all steps
  (``DecisionConfig.pair_merge_probability``);
* a pair is kept apart from P(different) = 0.5, once its smaller side has 10
  memories (``APART_STEP``); before, it waits. No true pair scored above 0.43;
* anything else waits, and nobody is asked. A waiting pair is compared again
  only when its smaller side reaches the next step of ``PAIR_STEPS``
  (3, 10, then 50 memories), and never after the last: at most four
  comparisons per pair. Each side is shown 10 memories (50 at the last step),
  chosen as the most recent few and the rest most similar to the other side's.

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
from datetime import datetime, timedelta, timezone
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
#: Most names looked at for one name on a shared word written in capitals
#: ("PR" in "PR #42" and "the Dutch address PR"), and most of those looks per
#: weekly pass. Such a word pairs too many names to compare them all: the judge
#: first rules out, on the two names alone, the ones that cannot be one thing.
LOOSE_PER_NAME = 10
NAME_CHECKS_PER_PASS = 20
#: P(different) from which the judge's look at two names alone rules the pair
#: out. Provisional: not measured yet.
NAME_CHECK_SKIP = 0.9
#: The comparison funnel. A pair is compared when it is found, and again only
#: when its smaller side reaches the next of these memory counts; after the
#: last, never. The smaller side is shown whole (up to ``memories_shown``), so
#: each step is new evidence about it. The larger side mostly grows with
#: memories about other things, so its count triggers nothing. Comparing again
#: on every memory that mentioned either side had no bound: an entity mentioned
#: in most saves had its waiting pairs compared on most saves.
PAIR_STEPS = (1, 3, 10, 50)
#: Memories per side that a comparison chooses from, most recent first.
PAIR_POOL = 200
#: Share of the memories shown per side that are the most recent ones; the rest
#: are those most similar to any of the other side's memories. A fact that links
#: the two or contradicts one of them ("lives in Munich", "moved to Amsterdam")
#: shares a topic with the other side. The most recent memories of a large
#: entity mostly do not, and they left the judge unsure: "Fundation" against
#: "Fundation GmbH" scored 0.76 with 3 recent facts per side and 0.59 with 10.
RECENT_SHARE = 0.3
#: A pair is kept apart for good only from this step on; before, "apart"
#: waits, since keeping apart ends all comparing. On 39 pairs from a real
#: store, "Fundation" (1 memory) against "Fundation GmbH" gave P(different)
#: 0.68-0.70 and at 10 memories P(same) 0.91-0.93; the store owner (1 or 3
#: memories) against "Cosmin Novac" gave P(different) 0.90-0.95 and at 10
#: P(same) 0.99. Keeping apart from the first step lost that pair for good;
#: from step 10 no true pair was lost, for 74-76 comparisons instead of 45-47.
APART_STEP = 10
#: A step between the first two of ``PAIR_STEPS``. A pair still waiting after
#: its first comparison, with a side of fewer than ``PAIR_STEPS[1]`` memories,
#: is compared once more with other memories from the conversations that saved
#: that side's ("the user is renovating the kitchen" beside "Johnny comes on
#: Tuesday"). It is merged on the bar of the first step until this step has
#: been measured on its own.
CONTEXT_STEP = 2
#: Memories from the same conversations shown per side with few memories.
CONTEXT_MEMORIES = 5
#: How old that side's newest memory must be before its conversation counts
#: as over. Memories a conversation is still adding would be missing.
CONTEXT_QUIET_HOURS = 1.0
#: Memories within this many hours of a memory, in its session (or, without
#: one, with its client and context label), count as the same conversation.
SESSION_HOURS = 3.0
#: People compared with the store owner on their memories alone, besides the
#: ones whose names are worth comparing: an account named "admin", or none,
#: shares no name with the owner's.
OWNER_CANDIDATES = 3


def pair_step(count: int) -> int:
    """The funnel step a pair whose smaller side has ``count`` memories is at."""
    return max((step for step in PAIR_STEPS if count >= step), default=0)


def memories_shown(step: int) -> int:
    """Memories shown per side at a step: 10, and 50 at the last step. Jev reads
    long profiles: one contradicting fact placed last among 100 facts still gave
    P(different) = 0.95."""
    return PAIR_STEPS[-1] if step >= PAIR_STEPS[-1] else PAIR_STEPS[-2]


def rounds(compared: int, smaller: int) -> list[tuple[int, int]]:
    """The comparisons owed to a pair last compared at step ``compared`` whose
    smaller side now has ``smaller`` memories, as (step, memories shown per
    side). Steps that would show the same memories collapse into one."""
    owed: list[tuple[int, int]] = []
    for step in PAIR_STEPS:
        if compared < step <= smaller:
            shown = memories_shown(step)
            if owed and owed[-1][1] == shown:
                owed[-1] = (step, shown)
            else:
                owed.append((step, shown))
    return owed


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.casefold())
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def name_tokens(name: str) -> list[str]:
    return re.findall(r"[^\W_]+", _fold(name or ""))


def upper_words(name: str) -> set[str]:
    """The words a name writes in capitals, 2 to 6 letters long ("PR",
    "ICAM", "AB"): identifiers that several names of one thing may share."""
    return {_fold(w) for w in re.findall(r"[^\W\d_]+", name or "")
            if 2 <= len(w) <= 6 and w.isupper()}


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
    starting with its first letter, and holds the first letter of every word
    of ``long`` in order: "AWS" and "Amazon Web Services", "BSFZ" and
    "Bescheinigungsstelle Forschungszulage", "KfW" and "Kreditanstalt für
    Wiederaufbau". Without the second condition "action" counted as an
    acronym of "ai applications". In a name that capitalizes its words, the
    words written in lower case ("de", "la", "für") may be left out: "ICAM" and
    "Ilustre Colegio de la Abogacía de Madrid"."""
    parts, words = name_tokens(short), name_tokens(long)
    if len(parts) != 1 or len(words) < 2:
        return False
    written = re.findall(r"[^\W_]+", long)
    needed = [w for w, as_written in zip(words, written) if not as_written[:1].islower()]
    if len(written) != len(words) or not needed:
        needed = words
    letters, text = parts[0], "".join(words)
    if not (2 <= len(letters) <= 6) or len(letters) < len(needed) or letters[0] != text[0]:
        return False
    rest = iter(text)
    if not all(letter in rest for letter in letters):
        return False
    rest = iter(letters)
    return all(word[0] in rest for word in needed)


class NameIndex:
    """The store's entity names, indexed for finding the ones worth comparing
    with a given name."""

    def __init__(
        self, entities: Iterable[Entity], vectors: dict[str, np.ndarray] | None = None,
        *, rare_words: bool = True,
    ) -> None:
        self.entities = {e.id: e for e in entities if e.merged_into is None}
        #: Whether a shared rare word pairs two names. Off for tags: tags are
        #: short phrases that share words across related subjects ("art assets"
        #: and "art direction"). On 417 real tags it raised 263 of 379 pairs
        #: and was the only signal for none of the 16 duplicates.
        self.rare_words = rare_words
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
        self._by_upper: dict[str, set[str]] = defaultdict(set)
        for eid, entity in self.entities.items():
            for word in upper_words(entity.name):
                self._by_upper[word].add(eid)
        self._vector_ids = [eid for eid in self.entities if vectors and eid in vectors]
        self._matrix = (
            np.vstack([vectors[eid] for eid in self._vector_ids]) if self._vector_ids else None
        )

    def named_in(self, text: str, limit: int = 60) -> list[Entity]:
        """Entities with a name word that appears in ``text`` and is rare among
        the store's names, most shared words first: the ones a text about
        "Fundation" may be naming, such as "Fundation GmbH"."""
        shared: Counter[str] = Counter()
        tokens = set(name_tokens(text))
        for token in tokens:
            if len(token) >= 3 and self._counts.get(token, 0) <= self._rare_at:
                shared.update(self._by_token.get(token, ()))
        short = [t for t in tokens if 2 <= len(t) <= 6]
        for eid, entity in self.entities.items():
            if eid not in shared and any(is_acronym_of(t, entity.name) for t in short):
                shared[eid] += 1
        ranked = sorted(shared, key=lambda eid: -shared[eid])
        return [self.entities[eid] for eid in ranked[:limit]]

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
        for token in tokens if self.rare_words else ():
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
        # Identical names are candidates too: two entities that carry one name
        # are the likeliest duplicates. The caller excludes the entity itself.
        ranked = sorted((eid for eid in score if eid not in excluded),
                        key=lambda eid: -score[eid])
        return [self.entities[eid] for eid in ranked[:limit]]

    def loose_candidates(
        self, name: str, *, exclude: Iterable[str] = (), limit: int = LOOSE_PER_NAME
    ) -> list[Entity]:
        """Entities whose names share a word written in capitals with ``name``
        ("PR #42" and "the Dutch address PR"): the rarest word first, then the
        most recently updated. Too loose to compare on without a look at the
        names first (``worth_comparing``); the caller excludes what
        ``candidates`` found."""
        excluded = set(exclude)
        found: list[str] = []
        for word in sorted(upper_words(name), key=lambda w: len(self._by_upper.get(w, ()))):
            members = sorted(self._by_upper.get(word, ()),
                             key=lambda eid: self.entities[eid].updated_at or "", reverse=True)
            found += [eid for eid in members if eid not in excluded and eid not in found]
        return [self.entities[eid] for eid in found[:limit]]


NAME_CHECK_CRITERIA = {
    "possible": (
        "It could: one name may be another way of writing or describing the "
        "other (shorter or longer, an abbreviation, a translation, a description "
        "of the thing), and nothing in the two names rules out one thing."
    ),
    "different": (
        "It cannot: the names carry different numbers, identifiers, people, "
        "organizations or places, or share only a common word or abbreviation "
        "such as a legal form."
    ),
}


def worth_comparing(
    decider: Decider, entity: Entity, others: list[Entity]
) -> tuple[list[Entity], list[tuple[Entity, float]]]:
    """Look at names alone before comparing on memories: one question per name
    in ``others``, in one call. Returns the names worth comparing with
    ``entity`` and the ones ruled out, with P(different). Without an answer a
    name is worth comparing."""
    if not others:
        return [], []

    def typed(e: Entity) -> str:
        return f'"{e.name}" ({e.entity_type or "type unknown"})'

    questions = {
        f"n{i}": Choice(instructions=f"Could {typed(other)} name the same thing as {typed(entity)}?",
                        criteria=NAME_CHECK_CRITERIA)
        for i, other in enumerate(others)
    }
    try:
        answers = decider.decide(
            f"A name from one person's long-term memory store: {typed(entity)}.", questions)
    except Exception:
        return list(others), []
    kept: list[Entity] = []
    ruled_out: list[tuple[Entity, float]] = []
    for i, other in enumerate(others):
        answer = answers[f"n{i}"]
        different = float((answer.probabilities or {}).get("different", 0.0)) if answer.available else 0.0
        if different >= NAME_CHECK_SKIP:
            ruled_out.append((other, different))
        else:
            kept.append(other)
    return kept, ruled_out


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
    #: Other memories from the conversations that saved these facts, naming
    #: neither side, with where each came from (``CONTEXT_STEP`` only).
    context: list[str] = field(default_factory=list)
    context_sources: list[Source] = field(default_factory=list)


@dataclass
class Mention:
    """A name a save has not attached yet, with the memory that carries it."""

    name: str
    entity_type: str | None
    memory: Memory


def profile_from(
    subject: Entity | Mention, memories: list[Memory], context: Iterable[Memory] = ()
) -> Profile:
    """One side of a comparison, showing ``memories`` as its facts and
    ``context`` as other memories of the conversations that saved them."""
    facts = [m.content for m in memories]
    sources = [source_of(m) for m in memories]
    around = list(context)
    extra = {"context": [m.content for m in around],
             "context_sources": [source_of(m) for m in around]}
    if isinstance(subject, Mention):
        return Profile(subject.name, subject.entity_type, facts, sources=sources, **extra)
    home = (subject.metadata or {}).get("home")
    return Profile(
        name=subject.name,
        entity_type=subject.entity_type,
        facts=facts,
        description=subject.description or "",
        home=home.get("name", "") if isinstance(home, dict) else "",
        sources=sources,
        owner=bool((subject.metadata or {}).get("owner")),
        **extra,
    )


def _unit_rows(vectors: dict[str, np.ndarray], ids: list[str]) -> np.ndarray | None:
    """The unit vectors of ``ids``, of the dimension most of them share (a
    store can hold vectors from more than one embedding model)."""
    found = [vectors[i] for i in ids if i in vectors]
    if not found:
        return None
    size = Counter(v.shape[0] for v in found).most_common(1)[0][0]
    rows = np.vstack([v for v in found if v.shape[0] == size]).astype(float)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    rows = rows[norms[:, 0] > 0] / norms[norms[:, 0] > 0]
    return rows if len(rows) else None


def choose(
    pool: list[Memory], shown: int, vectors: dict[str, np.ndarray], against: list[str]
) -> list[Memory]:
    """``shown`` memories of ``pool`` (most recent first) for a comparison: the
    most recent ``RECENT_SHARE`` of them, and the rest the ones most similar to
    any memory in ``against``, the other side's. Without vectors, the most
    recent. The result keeps the pool's order."""
    if len(pool) <= shown:
        return pool
    other = _unit_rows(vectors, against)
    if other is None:
        return pool[:shown]
    recent = max(1, round(shown * RECENT_SHARE))

    def closeness(memory: Memory) -> float:
        vector = vectors.get(memory.id)
        if vector is None or vector.shape[0] != other.shape[1]:
            return -2.0
        norm = np.linalg.norm(vector)
        return float((other @ (vector / norm)).max()) if norm else -2.0

    rest = pool[recent:]
    closest = set(sorted(range(len(rest)), key=lambda i: -closeness(rest[i]))[:shown - recent])
    return pool[:recent] + [m for i, m in enumerate(rest) if i in closest]


def pair_state(a: Profile, b: Profile) -> str:
    texts: dict[str, int] = {}
    sessions: dict[str, int] = {}
    for profile in (a, b):
        for source in profile.sources + profile.context_sources:
            if source.text:
                texts.setdefault(source.text, len(texts) + 1)
            if source.session:
                sessions.setdefault(source.session, len(sessions) + 1)

    def described(src: Source) -> str:
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

    def origin(p: Profile, i: int) -> str:
        if i < len(p.sources):
            return described(p.sources[i])
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
        if p.context:
            lines.append("Other memories from the same conversations (they name "
                         "neither entity):")
            for i, text in enumerate(p.context):
                where = described(p.context_sources[i]) if i < len(p.context_sources) else ""
                lines.append(f"- [{where}] {text}" if where else f"- {text}")
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
    if any(p.context for p in (a, b)):
        header += (" An entity with few facts may also show other memories from the "
                   "conversations that saved its facts; they name neither entity.")
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


def decide_pair(probabilities: dict[str, float], decider: Decider, step: int = 1) -> str:
    """"merge", "apart" or "wait", on the provider's measured thresholds for a
    comparison at this funnel step."""
    if probabilities["same"] >= decider.pair_merge_threshold(step):
        return "merge"
    if probabilities["different"] >= decider.pair_apart_probability:
        return "apart"
    return "wait"


@dataclass
class Verdict:
    """What comparing a pair came to."""

    #: The averaged answer of the last comparison made, None when none was.
    probabilities: dict[str, float] | None
    #: "merge", "apart" or "wait".
    action: str
    #: The funnel step the pair has now been compared at.
    step: int


def compare(
    decider: Decider, backend: MemoryBackend, a: Entity, b: Entity | Mention,
    compared: int = 0,
) -> Verdict:
    """Decide a pair at the funnel steps it has reached since it was last
    compared at step ``compared`` (0: never). Nothing is asked when it has
    reached no new step. Otherwise it is compared with 10 memories per side,
    and, when that leaves it waiting and its smaller side has 50 memories,
    once more with 50. Before ``APART_STEP`` a pair waits instead of being
    kept apart.

    A pair still waiting after its first comparison, with a side of fewer
    than ``PAIR_STEPS[1]`` memories, is compared once more at
    ``CONTEXT_STEP``: with other memories of the conversations that saved
    that side's, once they have been quiet for ``CONTEXT_QUIET_HOURS``.

    ``b`` is an entity, or a mention a save has not attached yet.
    """
    count_b = 1 if isinstance(b, Mention) else backend.count_entity_memories(b.id)
    smaller = min(backend.count_entity_memories(a.id), count_b)
    owed = rounds(compared, smaller)
    if not owed and not _context_owed(compared, smaller):
        return Verdict(None, "wait", compared)
    pool_a = backend.entity_memories(a.id, limit=PAIR_POOL)
    pool_b = ([b.memory] if isinstance(b, Mention)
              else backend.entity_memories(b.id, limit=PAIR_POOL))
    # A memory that names both entries says nothing about whether they are one
    # thing: the extractor listed two names for it. Shown on both sides it
    # read as the same fact twice, and "Michaela Neumann" merged with
    # "Dr. Neumann", named in one note, at P(same) 1.0. It is left out.
    shared = {m.id for m in pool_a} & {m.id for m in pool_b}
    if shared:
        pool_a = [m for m in pool_a if m.id not in shared]
        pool_b = [m for m in pool_b if m.id not in shared]
        smaller = min(len(pool_a), len(pool_b))
        owed = rounds(compared, smaller)
        if not owed and not _context_owed(compared, smaller):
            return Verdict(None, "wait", compared)
    vectors = backend.vectors_of([m.id for m in pool_a + pool_b])
    ids_a, ids_b = [m.id for m in pool_a], [m.id for m in pool_b]
    verdict = Verdict(None, "wait", compared)
    for step, shown in owed:
        probabilities = judge_pair(
            decider,
            profile_from(a, choose(pool_a, shown, vectors, ids_b)),
            profile_from(b, choose(pool_b, shown, vectors, ids_a)),
        )
        if probabilities is None:
            break
        verdict = Verdict(probabilities, decide_pair(probabilities, decider, step), step)
        if verdict.action == "apart" and step < APART_STEP:
            verdict.action = "wait"
        if verdict.action != "wait":
            break
    if verdict.action == "wait" and verdict.step == PAIR_STEPS[0]:
        return _in_context(decider, backend, a, b, pool_a, pool_b, vectors, verdict)
    return verdict


def _context_owed(compared: int, smaller: int) -> bool:
    return compared == PAIR_STEPS[0] and 1 <= smaller < PAIR_STEPS[1]


def _recorded(memory: Memory) -> datetime | None:
    try:
        at = datetime.fromisoformat(memory.created_at or "")
    except ValueError:
        return None
    return at if at.tzinfo else at.replace(tzinfo=timezone.utc)


def _in_context(
    decider: Decider, backend: MemoryBackend, a: Entity, b: Entity | Mention,
    pool_a: list[Memory], pool_b: list[Memory], vectors: dict[str, np.ndarray],
    verdict: Verdict,
) -> Verdict:
    """The ``CONTEXT_STEP`` comparison of a pair ``verdict`` left waiting at
    the first step. Nothing is asked while a conversation may still be adding
    memories; when there is nothing to add, the step counts as done."""
    thin = [pool if len(pool) < PAIR_STEPS[1] else [] for pool in (pool_a, pool_b)]
    quiet = datetime.now(timezone.utc) - timedelta(hours=CONTEXT_QUIET_HOURS)
    for memory in thin[0] + thin[1]:
        at = _recorded(memory)
        if at is None or at > quiet:
            return verdict
    named = {m.id for m in pool_a + pool_b}
    sides = {e.id for e in (a, b) if isinstance(e, Entity)}
    context: list[list[Memory]] = []
    for pool in thin:
        found: dict[str, Memory] = {}
        for memory in pool:
            for other in backend.session_memories(memory, hours=SESSION_HOURS):
                if other.id not in named and other.id not in found and not sides & {
                        e.id for e in backend.entities_of_memory(other.id)}:
                    found[other.id] = other
        around = list(found.values())
        if len(around) > CONTEXT_MEMORIES:
            near = backend.vectors_of(list(found) + [m.id for m in pool])
            keep = {m.id for m in choose(around, CONTEXT_MEMORIES, near, [m.id for m in pool])}
            around = [m for m in around if m.id in keep]
        context.append(around)
    if not any(context):
        return Verdict(verdict.probabilities, verdict.action, CONTEXT_STEP)
    ids_a, ids_b = [m.id for m in pool_a], [m.id for m in pool_b]
    shown = memories_shown(PAIR_STEPS[0])
    probabilities = judge_pair(
        decider,
        profile_from(a, choose(pool_a, shown, vectors, ids_b), context[0]),
        profile_from(b, choose(pool_b, shown, vectors, ids_a), context[1]),
    )
    if probabilities is None:
        return verdict
    action = decide_pair(probabilities, decider, CONTEXT_STEP)
    return Verdict(probabilities, "wait" if action == "apart" else action, CONTEXT_STEP)


def is_owner(entity: Entity | Mention) -> bool:
    return isinstance(entity, Entity) and bool((entity.metadata or {}).get("owner"))


def merge_pair(backend: MemoryBackend, a: Entity, b: Entity) -> bool:
    """Fold one entity of a pair the judge merged into the other. The store
    owner is folded into the person it was found to be, who keeps their name
    and becomes the owner."""
    keep, drop = (b, a) if is_owner(a) else (a, b)
    if not backend.merge_entities(keep.id, drop.id):
        return False
    if is_owner(drop):
        kept = backend.get_entity(keep.id) or keep
        backend.set_entity_metadata(kept.id, {**(kept.metadata or {}), "owner": True})
    return True


def closest_people(
    backend: MemoryBackend, scope: Any, owner: Entity, entities: Iterable[Entity],
    limit: int = OWNER_CANDIDATES,
) -> list[Entity]:
    """The people whose memories are closest, on average, to the owner's."""
    members: dict[str, list[str]] = defaultdict(list)
    for entity_id, memory_id in backend.entity_memory_links(scope):
        members[entity_id].append(memory_id)
    people = [e for e in entities if e.entity_type == "person" and e.id != owner.id
              and not is_owner(e) and members.get(e.id)]
    if not members.get(owner.id) or not people:
        return []
    vectors = backend.vectors_of(sorted({m for e in [owner, *people] for m in members[e.id]}))

    def centroid(entity: Entity) -> np.ndarray | None:
        rows = _unit_rows(vectors, members[entity.id])
        if rows is None:
            return None
        mean = rows.mean(axis=0)
        norm = np.linalg.norm(mean)
        return mean / norm if norm else None

    center = centroid(owner)
    if center is None:
        return []
    scored = []
    for person in people:
        other = centroid(person)
        if other is not None and other.shape == center.shape:
            scored.append((float(other @ center), person))
    return [person for _, person in sorted(scored, key=lambda item: -item[0])[:limit]]


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
#: A tag pair is compared when it is found, and once more when both tags are on
#: ``TAG_EXAMPLES`` memories and the judge sees all it will ever see; never
#: after. Comparing every candidate pair on every pass cost 758 judge calls a
#: pass on a 417-tag store, for the same answers.
TAG_STEPS = (1, TAG_EXAMPLES)


def tag_step(count: int) -> int:
    """The funnel step of a tag pair whose less used tag is on ``count`` memories."""
    return max((step for step in TAG_STEPS if count >= step), default=0)


def tag_pair_key(a: str, b: str) -> str:
    return "\n".join(sorted((a, b)))

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
    compared: dict[str, int] | None = None,
    limit: int = 400,
) -> list[dict[str, Any]]:
    """Groups of tags the judge puts at ``decider.tag_merge_probability`` or
    higher, each kept under its most used tag. ``memories_of(tag)`` returns the
    tag's most recent memories, most recent first.

    ``compared`` maps ``tag_pair_key`` to the step of ``TAG_STEPS`` the pair was
    last compared at, and is updated in place: only pairs that reached a new
    step are asked about, and pairs of tags that no longer exist are dropped.

    Measured on the 379 candidate pairs of a real 417-tag store, 10 memories per
    tag, two runs: from 0.55 it merged 7-9 of the 16 pairs I labelled one
    subject ("fundation" and "fundation gmbh" at 0.86-0.88, "bildy" and
    "bildy.ai", "steuer" and "tax", "cologne" and "colonge") and none of the
    41 borderline or 322 two-subject pairs; the highest two-subject pair was
    "restart" and "shutdown" at 0.46.
    """
    compared = {} if compared is None else compared
    counts = {str(t["category"]).strip().casefold(): int(t.get("count") or 0) for t in tags}
    labels = sorted(counts)
    nodes = [Entity(id=label, name=label, user_id=None) for label in labels]
    index = NameIndex(nodes, vectors, rare_words=False)
    step = {
        pair: tag_step(min(counts[pair[0]], counts[pair[1]]))
        for pair in {
            tuple(sorted((label, other.name)))
            for label in labels
            for other in index.candidates(label, vector=(vectors or {}).get(label),
                                          exclude={label})
        }
    }
    live = {tag_pair_key(*pair) for pair in step}
    for key in [key for key in compared if key not in live]:
        del compared[key]
    pairs = sorted(p for p in step if step[p] > compared.get(tag_pair_key(*p), 0))[:limit]
    examples = {tag: memories_of(tag) for tag in sorted({t for pair in pairs for t in pair})}
    scores = parallel(
        lambda pair: judge_tag_pair(decider, *pair, counts, known, examples), pairs
    )
    for pair, score in zip(pairs, scores):
        if score is not None:
            compared[tag_pair_key(*pair)] = step[pair]
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
