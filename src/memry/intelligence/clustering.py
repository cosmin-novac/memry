"""Topic vocabulary cleanup and higher-level abstraction.

Mechanical formatting and singular/plural duplicates are detected deterministically.
An optional LLM can propose exact synonym merges and broader synthetic topics. Broader
topics are stored as hierarchy edges rather than copied onto every member memory.

Both paths are conservative: merge candidates must be real stored labels, and a
synthetic cluster must contain enough existing topics to earn its place.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..providers.llm import LLM
from .extraction import parse_lenient_json

SYNTHETIC_TAG_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "clusters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tag": {"type": "string"},
                    "members": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["tag", "members"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["clusters"],
    "additionalProperties": False,
}

SYNTHETIC_TAG_SYSTEM = """You organize a personal memory system's tags.

You are given the full list of tags currently in use, each with how many
memories carry it. Think outside the box and propose up to {max_new} NEW
higher-level tags that each cluster several of the existing tags under a broader
theme - the kind of abstraction a librarian adds so specific labels roll up into
navigable topics (e.g. "health" over running/diet/sleep; "career" over
promotion/interview/salary).

Hard rules:
- Each new tag's "members" must be drawn ONLY from the existing tags listed
  below, spelled exactly as given. Never invent member tags.
- A cluster must group at least {min_cluster} existing tags. Skip weak groupings.
- The new tag name must be a short, lowercase, general theme, and must NOT
  duplicate an existing tag or one of the already-abstract tags listed.
- Prefer a few strong, genuinely useful clusters over many thin ones. It is fine
  to return fewer than {max_new}, or none if nothing clusters well.
- Do not force unrelated tags together. Coherence matters more than coverage.

Return JSON: {{"clusters": [{{"tag": "...", "members": ["...", "..."]}}]}}.
"""


CANONICALIZE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "groups": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "canonical": {"type": "string"},
                    "variants": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["canonical", "variants"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["groups"],
    "additionalProperties": False,
}

CANONICALIZE_SYSTEM = """You de-duplicate a tag vocabulary. Merging LOSES a
distinction forever, so only merge tags that are literally the SAME label written
differently. Merge ONLY these cases:
- spacing/hyphen/underscore/case: "writing preferences" = "writing-preference"
- singular/plural: "project" = "projects"
- an abbreviation and its full form: "org" = "organization"
- an exact synonym for the identical thing: "finance" = "financial"

NEVER merge tags that name different aspects, contexts, or scopes, even when they
are related. These are DISTINCT and must be LEFT ALONE:
- "writing-style" vs "response-style" (style of books vs style of replies)
- "tone" vs "style" (different attributes)
- "running" vs "diet" (both health, but different)
- "hardware" vs "hardware-limits" (a thing vs a constraint on it)
When two tags could be confused, that ambiguity is fixed by making them MORE
specific, not by collapsing them - so if in any doubt, do NOT merge.

For each merge group of two or more true duplicates, give the clearest canonical
name (prefer one already in the list). Return few, high-confidence groups, or an
empty list. JSON only: {"groups": [{"canonical": str, "variants": [str, ...]}]}."""


_TOPIC_SEPARATOR_RE = re.compile(r"[-_\s]+")
_UNINFLECTED_TOPICS = {
    "alias", "atlas", "bias", "business", "canvas", "chaos", "gas", "mathematics",
    "news", "physics", "series", "species", "status",
}


def _singular_topic_word(word: str) -> str:
    if len(word) <= 3 or word in _UNINFLECTED_TOPICS:
        return word
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith(("ches", "shes", "sses", "xes", "zes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


# A company is one subject whether or not its legal form is written: on a real
# store "fundation gmbh" (13 memories) sat beside "fundation" (31).
_TOPIC_LEGAL_FORMS = frozenset({
    "gmbh", "ug", "ag", "se", "kg", "inc", "ltd", "llc", "plc", "corp",
})
_TOPIC_TLDS = frozenset({
    "ai", "app", "co", "com", "de", "dev", "eu", "io", "me", "net", "org", "so", "xyz",
})
_TOPIC_DOMAIN_RE = re.compile(r"^([a-z0-9-]+)[.\s]([a-z]+)$")


def _obvious_topic_key(value: str, names: set[str] | frozenset[str] = frozenset()) -> str:
    """The form two tags share when they are one subject written two ways.

    ``names`` are the companies, products and projects the store knows. A
    domain ("bildy.ai", "bildy ai") only joins its name when the name is one of
    them: "character.ai" is not the tag "character".
    """
    value = value.casefold().strip()
    domain = _TOPIC_DOMAIN_RE.match(value)
    if domain and domain.group(2) in _TOPIC_TLDS and domain.group(1) in names:
        return domain.group(1)
    words = [word for word in _TOPIC_SEPARATOR_RE.split(value.replace(".", " ")) if word]
    while len(words) > 1 and words[-1] in _TOPIC_LEGAL_FORMS:
        words.pop()
    if not words:
        return ""
    words[-1] = _singular_topic_word(words[-1])
    return " ".join(words)


def domain_name(value: str) -> str | None:
    """The name in a tag written as a web domain ("bildy" in "bildy.ai" or
    "bildy ai"), so a caller looks up only those names in the store."""
    domain = _TOPIC_DOMAIN_RE.match(value.casefold().strip())
    return domain.group(1) if domain and domain.group(2) in _TOPIC_TLDS else None


def _one_swap_apart(a: str, b: str) -> bool:
    if len(a) != len(b):
        return False
    diff = [i for i in range(len(a)) if a[i] != b[i]]
    return (len(diff) == 2 and diff[1] == diff[0] + 1
            and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]])


def swapped_letter_typos(tags: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """``(typo, tag)`` pairs where the typo swaps two neighbouring letters of a
    tag used at least five times as often, and the typo is used at most twice.

    On a real store of 417 tags, "one letter apart" found five pairs and three
    were different subjects: "finance" and "yfinance", "memory" and "memry",
    "preference" and "reference". A swap found one pair, "colonge" and
    "cologne", and it was a typo. Words that are swaps of each other ("casual",
    "causal") still exist, so a caller confirms each pair before merging.
    """
    counts = {str(t["category"]).strip().casefold(): int(t.get("count") or 0) for t in tags}
    pairs = []
    for rare, rare_count in counts.items():
        if len(rare) < 6 or rare_count > 2:
            continue
        for common, common_count in counts.items():
            if common_count >= 5 * max(rare_count, 1) and _one_swap_apart(rare, common):
                pairs.append((rare, common))
    return sorted(pairs)


def obvious_canonical_merges(
    tags: list[dict[str, Any]], names: set[str] | frozenset[str] = frozenset()
) -> list[dict[str, Any]]:
    """Find deterministic formatting, singular/plural, legal-form and domain
    duplicates.

    A key is only actionable when two real stored labels map to it, so a lone
    word is never rewritten by a speculative inflection rule.
    """
    known = {str(tag["category"]).strip().casefold() for tag in tags}
    known.discard("")
    grouped: dict[str, list[str]] = {}
    for topic in sorted(known):
        key = _obvious_topic_key(topic, names)
        if key:
            grouped.setdefault(key, []).append(topic)
    merges: list[dict[str, Any]] = []
    for key, variants in grouped.items():
        if len(variants) < 2:
            continue
        canonical = key if key in variants else min(
            variants,
            key=lambda value: (value.count("-") + value.count("_"), len(value), value),
        )
        merges.append({"canonical": canonical, "variants": variants, "automatic": True})
    return merges

def judge_tag_pairs(decider, pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Which of these tag pairs mean the same thing, judged in one call.

    Only ever adds suggestions: on a labelled set this missed pairs a person
    would merge ("food" beside "diet") but never proposed an unrelated pair, so
    it is safe in front of a review queue and wrong as automation.
    """
    from ..providers.decisions import Noul

    if decider is None or not decider.available or not pairs:
        return []
    questions = {
        f"t{i}": Noul(instructions=f'Do the tags "{a}" and "{b}" mean the same thing '
                                   f"and should be merged into one?")
        for i, (a, b) in enumerate(pairs)
    }
    answers = decider.decide(
        "Tags used to file memories in a personal long-term memory store.", questions
    )
    out = []
    for i, pair in enumerate(pairs):
        answer = answers[f"t{i}"]
        if answer.available and answer.value >= 0.5:
            out.append(pair)
    return out


def suggest_canonical_merges(
    llm: LLM, tags: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """One cheap call proposing variant/synonym merges over the tag list.

    Returns validated ``{"canonical", "variants"}`` groups where every variant
    is a real existing tag and the group merges 2+ of them. Nothing is applied;
    the caller (Upkeep > Tags) shows these for one-click approval."""
    known = {str(t["category"]).strip().lower() for t in tags}
    known.discard("")
    if len(known) < 2:
        return []
    obvious = obvious_canonical_merges(tags)
    if not llm.available:
        return obvious
    listing = json.dumps(sorted(known), ensure_ascii=False)
    raw = llm.complete(
        CANONICALIZE_SYSTEM,
        f"Tags, as a JSON array with one tag per element: {listing}\n\n"
        "Propose the merge groups as JSON.",
        json_schema=CANONICALIZE_SCHEMA,
    )
    data = parse_lenient_json(raw)
    if not isinstance(data, dict):
        return obvious
    out: list[dict[str, Any]] = list(obvious)
    used: set[str] = {variant for group in obvious for variant in group["variants"]}
    for group in data.get("groups", []):
        if not isinstance(group, dict):
            continue
        variants = []
        for v in group.get("variants", []):
            v = str(v).strip().lower()
            if v in known and v not in used and v not in variants:
                variants.append(v)
        if len(variants) < 2:
            continue
        canonical = str(group.get("canonical", "")).strip().lower()
        if canonical not in variants:
            canonical = variants[0]  # canonical must be one of the real variants
        out.append({"canonical": canonical, "variants": variants})
        used.update(variants)
    return out


def semantic_duplicate_tags(
    centroids: dict[str, Any],
    counts: dict[str, int],
    cooccurrence: dict[tuple[str, str], int],
    *,
    labels: dict[str, Any] | None = None,
    threshold: float = 0.93,
    label_threshold: float = 0.62,
    paired_centroid_threshold: float = 0.75,
    max_pairs: int = 20,
) -> list[dict[str, Any]]:
    """Find tags that split one subject, using the vectors already stored.

    ``obvious_canonical_merges`` catches spelling and plural variants. It cannot
    catch "tech" beside "technical", which is the split that actually costs
    recall: a fragmented tag excludes the memories a question needs, and no
    ranking can recover them once the filter has dropped them.

    Signals, and why each is required:

    - **member centroids** (always): the two tags' memories occupy the same
      region of embedding space. Alone this is not enough. Measured on a real
      149-memory store, one centroid threshold either reported nothing (0.93)
      or proposed wrong merges like career+preference (0.80) while still
      missing the real split - personal stores are topically dense, so
      *different* tags about one life also sit close together.
    - **label similarity** (when ``labels`` is given): the tag names themselves
      mean the same thing. The conjunction was surgical on the same store:
      exactly the one true split (tech/technical, centroid 0.994) and nothing
      else. With labels available, the centroid requirement relaxes to
      ``paired_centroid_threshold``; without them, the strict ``threshold``
      applies alone.
    - **low co-occurrence** (always): complementary tags ("kitchen remodel" /
      "bathroom remodel") sit close in vector space but are genuinely distinct,
      and a user who applies both to one memory is telling us they mean
      different things.

    Returns ranked ``{"canonical", "variants", "similarity"}`` proposals. The
    caller decides whether to apply them; nothing here mutates the store.
    """
    import numpy as np

    names = [t for t in centroids if counts.get(t, 0) >= 2]
    if len(names) < 2:
        return []

    def unit_rows(vectors: list[Any]) -> "np.ndarray":
        matrix = np.array(vectors, dtype=float)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.where(norms == 0, 1.0, norms)

    sim = unit_rows([centroids[t] for t in names])
    sim = sim @ sim.T
    label_sim = None
    if labels and all(t in labels for t in names):
        lab = unit_rows([labels[t] for t in names])
        label_sim = lab @ lab.T

    pairs: list[dict[str, Any]] = []
    for i, a in enumerate(names):
        for j in range(i + 1, len(names)):
            b = names[j]
            score = float(sim[i, j])
            if label_sim is not None:
                if score < paired_centroid_threshold:
                    continue
                if float(label_sim[i, j]) < label_threshold:
                    continue
            elif score < threshold:
                continue
            together = cooccurrence.get((a, b), 0) + cooccurrence.get((b, a), 0)
            smaller = min(counts[a], counts[b])
            # Applied to the same memories = deliberate distinction, not a split.
            if smaller and together / smaller > 0.25:
                continue
            # The better-established label wins, ties broken for stability.
            canonical, variant = (a, b) if (counts[a], b) > (counts[b], a) else (b, a)
            pairs.append({
                "canonical": canonical,
                "variants": sorted([canonical, variant]),
                "similarity": round(score, 4),
            })
    pairs.sort(key=lambda p: -p["similarity"])
    return pairs[:max_pairs]


def propose_synthetic_tags(
    llm: LLM,
    tags: list[dict[str, Any]],
    *,
    existing_synthetic: list[str],
    max_new: int = 5,
    min_cluster: int = 2,
) -> list[dict[str, Any]]:
    """Ask the LLM for higher-level tags. Raises if the LLM is unavailable.

    ``tags`` is the category histogram ([{"category", "count"}]). Returns a list
    of validated ``{"tag", "members"}`` dicts: the tag is lowercased and unique,
    members are filtered to real existing tags, and clusters below
    ``min_cluster`` distinct members are dropped.
    """
    known = {str(t["category"]).strip().lower(): int(t.get("count", 0)) for t in tags}
    known.pop("", None)
    if len(known) < min_cluster:
        return []
    already = {t.strip().lower() for t in existing_synthetic}

    listing = "\n".join(f"- {tag} ({count})" for tag, count in known.items())
    raw = llm.complete(
        SYNTHETIC_TAG_SYSTEM.format(max_new=max_new, min_cluster=min_cluster),
        "Existing tags (tag (memory count)):\n"
        f"{listing}\n\n"
        f"Already-abstract tags to not repeat: {sorted(already) or 'none'}\n\n"
        "Propose the higher-level clusters as JSON.",
        json_schema=SYNTHETIC_TAG_SCHEMA,
    )
    return _validate(raw, known=set(known), already=already, min_cluster=min_cluster,
                     max_new=max_new)


def _validate(
    raw: str, *, known: set[str], already: set[str], min_cluster: int, max_new: int
) -> list[dict[str, Any]]:
    data = parse_lenient_json(raw)
    if not isinstance(data, dict):
        return []
    out: list[dict[str, Any]] = []
    seen_tags: set[str] = set()
    for cluster in data.get("clusters", []):
        if not isinstance(cluster, dict):
            continue
        tag = str(cluster.get("tag", "")).strip().lower()
        if not tag or tag in known or tag in already or tag in seen_tags:
            continue  # must be a genuinely new label
        # Members must be real existing tags, never the synthetic tag itself and
        # never another system-generated parent: abstracting an abstraction is
        # how "liver health" and "weekly gym" decay back into "health".
        members = []
        for m in cluster.get("members", []):
            m = str(m).strip().lower()
            if m in known and m != tag and m not in already and m not in members:
                members.append(m)
        if len(members) < min_cluster:
            continue
        out.append({"tag": tag, "members": members})
        seen_tags.add(tag)
        if len(out) >= max_new:
            break
    return out
