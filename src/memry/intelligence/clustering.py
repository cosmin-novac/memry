"""Topic vocabulary cleanup and higher-level abstraction.

Mechanical formatting and singular/plural duplicates are detected deterministically.
An optional LLM can propose exact synonym merges and broader synthetic topics. Broader
topics are stored as hierarchy edges rather than copied onto every member memory.

Both paths are conservative: merge candidates must be real stored labels, and a
synthetic cluster must contain enough existing topics to earn its place.
"""

from __future__ import annotations

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


def _obvious_topic_key(value: str) -> str:
    words = [word for word in _TOPIC_SEPARATOR_RE.split(value.casefold().strip()) if word]
    if not words:
        return ""
    words[-1] = _singular_topic_word(words[-1])
    return " ".join(words)


def obvious_canonical_merges(tags: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Find deterministic formatting and singular/plural duplicates.

    A key is only actionable when two real stored labels map to it, so a lone
    word is never rewritten by a speculative inflection rule.
    """
    known = {str(tag["category"]).strip().casefold() for tag in tags}
    known.discard("")
    grouped: dict[str, list[str]] = {}
    for topic in sorted(known):
        key = _obvious_topic_key(topic)
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

def suggest_canonical_merges(
    llm: LLM, tags: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """One cheap call proposing variant/synonym merges over the tag list.

    Returns validated ``{"canonical", "variants"}`` groups where every variant
    is a real existing tag and the group merges 2+ of them. Nothing is applied;
    the caller (Knowledge > Topics) shows these for one-click approval."""
    known = {str(t["category"]).strip().lower() for t in tags}
    known.discard("")
    if len(known) < 2:
        return []
    obvious = obvious_canonical_merges(tags)
    if not llm.available:
        return obvious
    listing = ", ".join(sorted(known))
    raw = llm.complete(
        CANONICALIZE_SYSTEM,
        f"Tags: {listing}\n\nPropose the merge groups as JSON.",
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
    threshold: float = 0.93,
    max_pairs: int = 20,
) -> list[dict[str, Any]]:
    """Find tags that split one subject, using the vectors already stored.

    ``obvious_canonical_merges`` catches spelling and plural variants. It cannot
    catch "liver bloods" beside "liver lab results", which is the split that
    actually costs recall: a fragmented tag excludes the memories a question
    needs, and no ranking can recover them once the filter has dropped them.

    Two signals must agree, because either alone is wrong:

    - the member memories occupy nearly the same region of embedding space;
    - the two tags rarely appear together on a memory. Complementary tags
      ("kitchen remodel" / "bathroom remodel") sit close in vector space but are
      genuinely distinct, and a user who applies both to one memory is telling
      us they mean different things.

    Returns ranked ``{"canonical", "variants", "similarity"}`` proposals. The
    caller decides whether to apply them; nothing here mutates the store.
    """
    import numpy as np

    names = [t for t in centroids if counts.get(t, 0) >= 2]
    if len(names) < 2:
        return []
    matrix = np.array([centroids[t] for t in names], dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.where(norms == 0, 1.0, norms)
    sim = matrix @ matrix.T

    pairs: list[dict[str, Any]] = []
    for i, a in enumerate(names):
        for j in range(i + 1, len(names)):
            b = names[j]
            score = float(sim[i, j])
            if score < threshold:
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
