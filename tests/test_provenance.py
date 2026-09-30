"""Provenance at the turn: extraction numbers the lines it shows, each fact
names the lines it rests on, and the store links the memory to the episodes
of exactly those lines. A fact that names none, or a line the transcript does
not have, rests on every episode of its save, as every fact did before."""

from __future__ import annotations

import json

from conftest import FakeLLM, fact, facts_response

from memry.intelligence.extraction import _transcript, extract_facts, verbatim_candidates

_TURNS = [
    {"role": "Ada", "content": "We finally went to the Lisbon aquarium."},
    {"role": "Bea", "content": "  "},
    {"role": "Bea", "content": "Did you see the otters?"},
    {"role": "Ada", "content": "Yes, two sea otters holding hands."},
    {"role": "user", "name": "Kai", "content": "They named one Mira."},
]


def _with_sources(content: str, sources) -> dict:
    return {**fact(content), "sources": sources}


def test_the_transcript_numbers_each_line_that_says_something():
    """An empty message gets no line and no number, as it gets no episode;
    the speaker is written as before, after the number."""
    assert _transcript(_TURNS, numbered=True) == (
        "[1] Ada: We finally went to the Lisbon aquarium.\n"
        "[2] Bea: Did you see the otters?\n"
        "[3] Ada: Yes, two sea otters holding hands.\n"
        "[4] Kai (user): They named one Mira.")
    # the coverage audit reads the lines unnumbered
    assert _transcript(_TURNS).startswith("Ada: We finally went")


def test_extraction_reads_the_lines_a_fact_rests_on():
    llm = FakeLLM([json.dumps({"facts": [
        _with_sources("Ada went to the Lisbon aquarium", [1]),
        _with_sources("Ada saw two sea otters holding hands", [2, "3", 3]),
        fact("Kai named an otter Mira"),  # an output written before sources
        _with_sources("Bea asked about otters", "2"),  # not a list: none given
        _with_sources("Ada liked it", [1, "x"]),  # not all numbers: none given
    ]})])
    facts = extract_facts(llm, _TURNS)
    assert [f.sources for f in facts] == [[1], [2, 3], [], [], []]
    system, user = llm.calls[0]
    assert "[3] Ada: Yes, two sea otters holding hands." in user
    assert '"sources": [int]' in system


def test_a_fact_is_linked_to_the_episodes_of_its_lines(store, fake_llm):
    fake_llm.queue(facts_response(
        _with_sources("Ada went to the Lisbon aquarium", [1]),
        _with_sources("Ada saw two sea otters holding hands", [2, 3]),
    ))
    result = store.add(_TURNS, user_id="ada")
    lines = result.episode_ids  # one episode per line that says something
    assert len(lines) == 4
    by_content = {m.content: m for m in store.get_all(user_id="ada")}
    assert by_content["Ada went to the Lisbon aquarium"].source_episode_ids == [lines[0]]
    assert by_content["Ada saw two sea otters holding hands"].source_episode_ids == lines[1:3]


def test_a_fact_without_lines_or_with_a_line_that_is_not_there_rests_on_the_whole_save(
        store, fake_llm):
    fake_llm.queue(facts_response(
        fact("Ada went to the Lisbon aquarium"),
        _with_sources("Ada saw two sea otters holding hands", [3, 9]),
        _with_sources("Kai named an otter Mira", [0]),
    ))
    result = store.add(_TURNS, user_id="ada")
    for memory in store.get_all(user_id="ada"):
        assert memory.source_episode_ids == result.episode_ids


def test_a_plain_user_and_assistant_save_is_linked_as_before(store, fake_llm):
    fake_llm.queue(facts_response(fact("User lives in Leeds")))
    result = store.add([{"role": "user", "content": "I moved to Leeds."},
                        {"role": "assistant", "content": "Noted."}], user_id="ada")
    [memory] = store.get_all(user_id="ada")
    assert memory.source_episode_ids == result.episode_ids


def test_a_verbatim_memory_rests_on_its_own_message(verbatim_store):
    assert [c.sources for c in verbatim_candidates(_TURNS)] == [[1], [2], [3], [4]]
    result = verbatim_store.add(_TURNS[:3], user_id="ada")
    memories = {m.content: m for m in verbatim_store.get_all(user_id="ada")}
    assert memories["Ada: We finally went to the Lisbon aquarium."].source_episode_ids == [
        result.episode_ids[0]]
    assert memories["Bea: Did you see the otters?"].source_episode_ids == [
        result.episode_ids[1]]


def test_a_distilled_fact_rests_on_the_raw_saves_of_its_lines(store, fake_llm):
    """A deferred group is one transcript line per raw save: a fact naming
    line 2 rests on the second save's episode only."""
    first = store.add_deferred("We went to the Lisbon aquarium.", user_id="ada", run_id="r")
    second = store.add_deferred("We saw two sea otters holding hands.", user_id="ada",
                                run_id="r")
    pending = [a.memory_id for a in first.actions + second.actions]
    fake_llm.queue(facts_response(
        _with_sources("Ada saw two sea otters holding hands", [2]),
        fact("Ada went to the Lisbon aquarium"),
    ))
    store._distill_pending_group(pending)
    by_content = {m.content: m for m in store.get_all(user_id="ada")}
    assert by_content["Ada saw two sea otters holding hands"].source_episode_ids == (
        second.episode_ids)
    assert by_content["Ada went to the Lisbon aquarium"].source_episode_ids == (
        first.episode_ids + second.episode_ids)
