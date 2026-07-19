"""Fact extraction: turn raw conversation into candidate memories.

With an LLM configured, extraction distills discrete, self-contained facts
(the Mem0-paper phase 1). Without one, memry falls back to *verbatim*
mode - each message is stored as an episodic memory - so the system stays
useful with zero API keys.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from ..models import MEMORY_TYPES, CandidateFact
from ..providers.llm import LLM

EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "type": {
                        "type": "string",
                        "enum": ["semantic", "episodic", "procedural"],
                    },
                    "importance": {"type": "number"},
                    "categories": {"type": "array", "items": {"type": "string"}},
                    "entities": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["content", "type", "importance", "categories", "entities"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["facts"],
    "additionalProperties": False,
}

EXTRACTION_SYSTEM = """You are the long-term memory extraction system of an AI assistant.
Given a conversation, extract discrete facts worth remembering in future,
unrelated conversations. Today's date is {today}.

Extract:
- stable facts about the user (identity, role, location, relationships)
- preferences, opinions, goals, constraints
- decisions made and commitments/plans (convert relative dates to absolute)
- important entities and how they relate to the user
- procedural learnings (how the user wants things done)

Do NOT extract:
- small talk, transient context ("I'm tired today"), or assistant boilerplate
- secrets or credentials (passwords, API keys, tokens) - never store these
- information the user asked to keep out of memory

Rules:
- each fact must be fully self-contained: resolve pronouns and references
- one fact per item; keep each under ~200 characters
- importance in [0,1]: 0.9+ identity/hard constraints, ~0.7 preferences and
  decisions, ~0.4 minor details
- type: "semantic" (stable fact/preference), "episodic" (dated event/plan),
  "procedural" (how-to / workflow rule)

Respond with JSON only: {{"facts": [{{"content": str, "type": str,
"importance": number, "categories": [str], "entities": [str]}}]}}.
Return {{"facts": []}} if nothing is worth remembering."""


def extract_facts(
    llm: LLM,
    messages: list[dict[str, str]],
    *,
    now: datetime | None = None,
) -> list[CandidateFact]:
    """LLM extraction (phase 1). Raises if the LLM is unavailable."""
    now = now or datetime.now(timezone.utc)
    transcript = "\n".join(
        f"{m.get('role', 'user')}: {m['content'].strip()}"
        for m in messages
        if (m.get("content") or "").strip()
    )
    if not transcript:
        return []
    raw = llm.complete(
        EXTRACTION_SYSTEM.format(today=now.date().isoformat()),
        f"Conversation:\n{transcript}\n\nExtract the facts as JSON.",
        json_schema=EXTRACTION_SCHEMA,
    )
    return _parse_facts(raw)


def verbatim_candidates(messages: list[dict[str, str]]) -> list[CandidateFact]:
    """Zero-LLM fallback: store each message as an episodic memory."""
    out: list[CandidateFact] = []
    for m in messages:
        content = (m.get("content") or "").strip()
        if not content:
            continue
        role = m.get("role", "user")
        out.append(
            CandidateFact(
                content=content if role == "user" else f"{role}: {content}",
                memory_type="episodic",
                importance=0.5,
            )
        )
    return out


_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$", re.MULTILINE)


def _parse_facts(raw: str) -> list[CandidateFact]:
    data = parse_lenient_json(raw)
    if data is None:
        return []
    items = data.get("facts", []) if isinstance(data, dict) else data
    if not isinstance(items, list):
        return []
    facts: list[CandidateFact] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        mtype = item.get("type", "semantic")
        if mtype not in MEMORY_TYPES:
            mtype = "semantic"
        try:
            importance = float(item.get("importance", 0.5))
        except (TypeError, ValueError):
            importance = 0.5
        facts.append(
            CandidateFact(
                content=content,
                memory_type=mtype,  # type: ignore[arg-type]
                importance=min(max(importance, 0.0), 1.0),
                categories=[str(c) for c in item.get("categories", []) if c],
                entities=[str(e) for e in item.get("entities", []) if e],
            )
        )
    return facts


def parse_lenient_json(raw: str) -> Any:
    """Parse JSON out of LLM output: tolerates code fences and leading prose."""
    if not raw:
        return None
    text = _FENCE_RE.sub("", raw.strip()).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == opener:
                depth += 1
            elif text[i] == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    return None
