from __future__ import annotations

from datetime import datetime, timedelta, timezone

from conftest import FakeLLM, decision, fact, facts_response

from memry.config import DecayConfig
from memry.intelligence.context import build_context
from memry.intelligence.decay import effective_importance
from memry.intelligence.extraction import (
    extract_facts,
    parse_lenient_json,
    verbatim_candidates,
)
from memry.models import Memory, SearchResult


def test_parse_lenient_json_variants():
    assert parse_lenient_json('{"a": 1}') == {"a": 1}
    assert parse_lenient_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_lenient_json('Sure! Here you go: {"a": {"b": 2}} hope that helps') == {
        "a": {"b": 2}
    }
    assert parse_lenient_json("[1, 2]") == [1, 2]
    assert parse_lenient_json("no json here") is None
    assert parse_lenient_json("") is None


def test_extract_facts_parses_and_clamps():
    llm = FakeLLM(
        [
            facts_response(
                fact("User lives in Berlin", importance=1.7, categories=["location"]),
                {"content": "User prefers uv", "type": "bogus-type", "importance": "x",
                 "categories": [], "entities": []},
                {"content": "", "type": "semantic", "importance": 0.5, "categories": [], "entities": []},
            )
        ]
    )
    facts = extract_facts(llm, [{"role": "user", "content": "hi"}])
    assert len(facts) == 2
    assert facts[0].importance == 1.0  # clamped
    assert facts[1].memory_type == "semantic"  # bogus type falls back


def test_extract_facts_empty_conversation_skips_llm():
    llm = FakeLLM([])
    assert extract_facts(llm, [{"role": "user", "content": "  "}]) == []
    assert llm.calls == []


def test_verbatim_candidates():
    candidates = verbatim_candidates(
        [
            {"role": "user", "content": "I like tea"},
            {"role": "assistant", "content": "Noted!"},
            {"role": "user", "content": ""},
        ]
    )
    assert [c.content for c in candidates] == ["I like tea", "assistant: Noted!"]
    assert all(c.memory_type == "episodic" for c in candidates)


def test_effective_importance_decays_toward_floor():
    cfg = DecayConfig(enabled=True, half_life_days=30, floor=0.2)
    now = datetime.now(timezone.utc)
    fresh = Memory(content="x", importance=0.8)
    old = Memory(content="x", importance=0.8)
    old.updated_at = (now - timedelta(days=365)).isoformat(timespec="seconds")

    fresh_score = effective_importance(fresh, cfg, now)
    old_score = effective_importance(old, cfg, now)
    assert fresh_score > old_score
    assert old_score >= 0.8 * 0.2 - 1e-9  # never below floor * importance
    assert effective_importance(old, DecayConfig(enabled=False), now) == 0.8


def test_build_context_respects_budget():
    results = [
        SearchResult(memory=Memory(content=f"fact number {i} " + "x" * 80), score=1.0 - i * 0.01)
        for i in range(30)
    ]
    ctx = build_context(results, token_budget=200)
    assert ctx.text
    assert 0 < len(ctx.memory_ids) < 30
    assert ctx.token_estimate <= 200
    # highest-ranked memory is included first
    assert "fact number 0" in ctx.text


def test_build_context_empty():
    ctx = build_context([], token_budget=100)
    assert ctx.text == ""
    assert ctx.memory_ids == []
