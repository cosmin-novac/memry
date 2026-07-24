"""RAPTOR-lite collection summaries: the coarse navigation layer.

Cluster the actual memory embeddings, then have the LLM title and summarize each
cluster from the memories themselves - never by abstracting tags, which was the
approach that produced vague, useless labels. Clustering is pure numpy (no
tokens); only a handful of the largest, most coherent clusters are summarized,
so the LLM cost per run is small and bounded.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..providers.llm import LLM
from .extraction import parse_lenient_json

SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"},
    },
    "required": ["title", "summary"],
    "additionalProperties": False,
}

SUMMARY_SYSTEM = """You name a cluster of a person's memories. Given several
related memories, return a short, SPECIFIC title (a project, topic, person, or
theme they actually share - not a generic word) and a one or two sentence
summary of what the cluster is about, grounded in the memories. JSON only:
{"title": str, "summary": str}."""


def cluster_vectors(
    ids: list[str],
    vectors: np.ndarray,
    *,
    threshold: float = 0.55,
    min_size: int = 4,
    max_clusters: int = 8,
) -> list[list[str]]:
    """Greedy cosine clustering, largest-first, capped. O(n^2) but only ever run
    on a bounded set, so it stays cheap; no LLM involved."""
    if len(ids) < min_size:
        return []
    norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9)
    sim = norm @ norm.T
    assigned = np.zeros(len(ids), dtype=bool)
    clusters: list[list[int]] = []
    # seed from the densest points first so clusters form around strong centers
    density = (sim >= threshold).sum(axis=1)
    for seed in np.argsort(-density):
        if assigned[seed]:
            continue
        members = [i for i in np.where(sim[seed] >= threshold)[0] if not assigned[i]]
        if len(members) < min_size:
            assigned[seed] = True
            continue
        for i in members:
            assigned[i] = True
        clusters.append(members)
        if len(clusters) >= max_clusters:
            break
    return [[ids[i] for i in c] for c in clusters]


def summarize_cluster(llm: LLM, contents: list[str]) -> dict[str, str]:
    """One LLM call to title + summarize a cluster. Caps how many memories are
    shown so the prompt stays small even for a big cluster."""
    shown = contents[:20]
    body = "\n".join(f"- {c}" for c in shown)
    raw = llm.complete(
        SUMMARY_SYSTEM,
        f"Memories:\n{body}\n\nTitle and summary as JSON.",
        json_schema=SUMMARY_SCHEMA,
    )
    data = parse_lenient_json(raw)
    if not isinstance(data, dict):
        return {}
    title = str(data.get("title", "")).strip()
    return {"title": title, "summary": str(data.get("summary", "")).strip()} if title else {}
