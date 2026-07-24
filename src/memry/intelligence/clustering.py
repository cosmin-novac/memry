"""Tag abstraction: propose higher-level tags that cluster existing ones.

Given the flat list of tags a namespace has accumulated (each an ad-hoc label
from extraction or the user), an LLM is asked to step back and name a few
broader themes - a "synthetic" tag like ``health`` that groups ``running``,
``diet``, ``sleep``, ``doctor``. Each synthetic tag is then written onto every
memory carrying one of its member tags, giving a coarse index over a store that
would otherwise only have long-tail specifics.

This is deliberately conservative: the model may only cluster tags that already
exist (it cannot invent membership out of nothing), and a cluster must group at
least ``min_cluster_size`` of them, so a synthetic tag always earns its place.
"""

from __future__ import annotations

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


def suggest_canonical_merges(
    llm: LLM, tags: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """One cheap call proposing variant/synonym merges over the tag list.

    Returns validated ``{"canonical", "variants"}`` groups where every variant
    is a real existing tag and the group merges 2+ of them. Nothing is applied;
    the caller (Tag manager) shows these for one-click approval."""
    known = {str(t["category"]).strip().lower() for t in tags}
    known.discard("")
    if len(known) < 2:
        return []
    listing = ", ".join(sorted(known))
    raw = llm.complete(
        CANONICALIZE_SYSTEM,
        f"Tags: {listing}\n\nPropose the merge groups as JSON.",
        json_schema=CANONICALIZE_SCHEMA,
    )
    data = parse_lenient_json(raw)
    if not isinstance(data, dict):
        return []
    out: list[dict[str, Any]] = []
    used: set[str] = set()
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
        # members must be real existing tags, and not the synthetic tag itself
        members = []
        for m in cluster.get("members", []):
            m = str(m).strip().lower()
            if m in known and m != tag and m not in members:
                members.append(m)
        if len(members) < min_cluster:
            continue
        out.append({"tag": tag, "members": members})
        seen_tags.add(tag)
        if len(out) >= max_new:
            break
    return out
