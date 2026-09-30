"""Tag vocabulary cleanup: formatting and singular/plural duplicates, found
deterministically. Any other pair of tags is an entity pair
(``identity.compare_topics``)."""

from __future__ import annotations

import re
from typing import Any

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


def obvious_variant_prefix(value: str) -> str:
    """What every tag sharing ``value``'s obvious key (formatting and
    singular/plural, ``obvious_canonical_merges``) starts with once leading
    separators are dropped: the key's first word, less a final "y" when it
    is the only word ("companies" and "company" share "compan"). It narrows a
    lookup of one tag's obvious variants; the key decides which they are."""
    words = _obvious_topic_key(value).split(" ")
    first = words[0]
    return first[:-1] if len(words) == 1 and first.endswith("y") else first


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
