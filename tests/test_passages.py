"""Passages (``intelligence.passages``): a message of more than 150 words is
split into passages of at most 150 words, at paragraph breaks first. Evidence
and turn search find, judge and show passages in place of whole messages, so
a long reply can be quoted in part. A message of 150 words or fewer is one
passage of its whole text, with the message's vector, so chats of short
messages are read as before. The memories take what the turns leave of the
context."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from memry.config import Config
from memry.intelligence.context import estimate_tokens, fitting, turn_line
from memry.intelligence.passages import PASSAGE_WORDS, split_passages
from memry.providers.embeddings import HashEmbedder
from memry.store import MemoryStore

from test_turn_search import _EXCERPT, ScriptedJudge, ScriptedLLM, WordEmbedder, _fact


def _filler(n: int, word: str) -> str:
    return " ".join(f"{word}{i}" for i in range(n)) + "."


def _words(texts) -> list[str]:
    return [w for text in texts for w in text.split()]


# -- the split ---------------------------------------------------------------

def test_a_message_of_150_words_or_fewer_is_one_passage_unchanged():
    text = "  Ada:\n\nwe went   to Tokyo.  " + _filler(PASSAGE_WORDS - 5, "w")
    assert len(text.split()) == PASSAGE_WORDS
    assert split_passages(text) == [text]


def test_paragraphs_are_packed_whole_while_they_fit():
    paragraphs = [_filler(60, w) for w in ("a", "b", "c", "d")]
    got = split_passages("\n\n".join(paragraphs))
    assert got == ["\n\n".join(paragraphs[:2]), "\n\n".join(paragraphs[2:])]


def test_a_long_paragraph_is_split_at_sentence_ends_then_lines_then_words():
    sentences = " ".join(_filler(40, f"s{i}_") for i in range(10))  # 400 words, 10 sentences
    code = "\n".join(f"line{i} = call({i}) + {i}" for i in range(120))  # one sentence, lines
    run = " ".join(f"x{i}" for i in range(400))  # no sentence end, no line break
    for text in (sentences, code, run):
        got = split_passages(text)
        assert len(got) > 1
        assert all(len(p.split()) <= PASSAGE_WORDS for p in got)
        assert _words(got) == text.split()  # every word, in order
    assert all(p.endswith(".") for p in split_passages(sentences))
    assert all("\n" in p for p in split_passages(code))


# -- storage -------------------------------------------------------------------

_LONG = "\n\n".join([
    _filler(100, "alpha"),
    _filler(100, "beta"),
    "Shibuya Crossing is like Tokyo's Times Square, the crowds never stop. "
    + _filler(85, "gamma"),
    _filler(100, "delta"),
])
_ASKED = "What did Ada say Shibuya Crossing is like?"


def _store(judge=None, db_path=":memory:") -> MemoryStore:
    llm = ScriptedLLM()
    store = MemoryStore(Config(db_path=db_path), llm=llm, embedder=WordEmbedder(),
                        decider=judge or ScriptedJudge())
    store.llm_script = llm
    store.config.retrieval.relational_relevance = "jev"
    return store


def _save_long(store, facts):
    store.llm_script.facts.append(facts)
    return store.add([{"role": "user", "content": "We went to Tokyo in March for the wedding."},
                      {"role": "assistant", "content": _LONG},
                      {"role": "user", "content": "Thanks, the kitchen is next."}],
                     user_id="ada", run_id="r1", created_at="2026-05-02T10:00:00+00:00")


def test_a_short_message_is_one_passage_with_its_vector_and_a_long_one_is_embedded_in_parts():
    store = _store()
    try:
        saved = _save_long(store, [_fact("Ada went to Tokyo in March for a wedding", 1)])
        short, long_, last = saved.episode_ids
        passages = store.backend.passages_of(saved.episode_ids)
        assert [p.id for p in passages[short]] == [f"{short}:0"]
        assert passages[short][0].content == "We went to Tokyo in March for the wedding."
        assert [(p.position, p.count) for p in passages[long_]] == [(i, 4) for i in range(4)]
        assert _words(p.content for p in passages[long_]) == _LONG.split()
        model = store.embedder.model_id
        episode = store.backend.episode_vectors_of([short], model)[short]
        passage = store.backend.passage_vectors_of([f"{short}:0"], model)[f"{short}:0"]
        assert episode.tolist() == passage.tolist()
        assert len(store.backend.passage_vectors_of([p.id for p in passages[long_]], model)) == 4
        assert store.backend.passages_to_embed(model) == []
    finally:
        store.close()


def test_a_store_saved_before_passages_gets_them_on_opening(tmp_path):
    path = str(tmp_path / "old.db")
    store = _store(db_path=path)
    saved = _save_long(store, [_fact("Ada went to Tokyo in March for a wedding", 1)])
    with store.backend._lock:
        store.backend._db.execute("DELETE FROM passages")
        store.backend._db.commit()
    assert store.backend.passages_of(saved.episode_ids) == {}
    store.close()

    store = _store(db_path=path)
    try:
        short, long_, _ = saved.episode_ids
        passages = store.backend.passages_of(saved.episode_ids)
        assert len(passages[short]) == 1 and len(passages[long_]) == 4
        model = store.embedder.model_id
        assert f"{short}:0" in store.backend.passage_vectors_of([f"{short}:0"], model)
        assert store.embed_passages(dry_run=True) == {
            "passages": 4, "words": len(_LONG.split()), "embedded": 0}
        assert store.embed_passages()["embedded"] == 4
        assert store.backend.passages_to_embed(model) == []
    finally:
        store.close()


def test_cli_backfill_passages(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("MEMRY_DB_PATH", str(tmp_path / "cli.db"))
    monkeypatch.setenv("MEMRY_CONFIG", str(tmp_path / "missing.json"))
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "VOYAGE_API_KEY",
                "MEMRY_LLM_PROVIDER", "MEMRY_EMBEDDING_PROVIDER"):
        monkeypatch.delenv(key, raising=False)
    from memry.cli import main

    def make():
        return MemoryStore(Config.load(), embedder=HashEmbedder(64))

    monkeypatch.setattr("memry.cli._store", make)
    store = make()
    store.add(_LONG, user_id="ada", infer=False)
    with store.backend._lock:
        store.backend._db.execute(
            "UPDATE passages SET embedding = NULL, embedding_model = NULL WHERE count > 1")
        store.backend._db.commit()
    store.close()
    assert main(["backfill-passages", "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "dry_run": True, "passages": 4, "words": len(_LONG.split()), "embedded": 0}
    assert main(["backfill-passages"]) == 0
    assert json.loads(capsys.readouterr().out)["embedded"] == 4


# -- reading -------------------------------------------------------------------

def test_evidence_shows_the_matching_passage_of_a_long_message():
    store = _store()
    try:
        saved = _save_long(store, [_fact("Ada said Shibuya Crossing is like Times Square", 2)])
        long_ = saved.episode_ids[1]
        results = store.search(_ASKED, user_id="ada", limit=5, evidence=False)
        whole = estimate_tokens(_LONG)
        turns = store.evidence(_ASKED, results, user_id="ada", token_budget=whole // 2)
        assert turns and all(t.episode_id == long_ for t in turns)
        best = max(turns, key=lambda t: t.score)  # taken first, shown in the order said
        assert best.passage_id == f"{long_}:2" and "Times Square" in best.content
        assert "(part 3 of 4): Shibuya Crossing" in turn_line(best)
        assert len(turns) < 4
        assert sum(estimate_tokens(turn_line(t)) + 1 for t in turns) <= whole // 2
    finally:
        store.close()


def test_turn_search_quotes_the_matching_passage_with_neighbours_that_fit():
    judge = ScriptedJudge(facts=0.2)
    store = _store(judge)
    try:
        saved = _save_long(store, [_fact("Ada went to Tokyo in March for a wedding", 1)])
        long_ = saved.episode_ids[1]
        budget = store.config.retrieval.turn_search_tokens
        assert estimate_tokens(_LONG) > budget  # the whole reply could never be shown
        results = store.search(_ASKED, user_id="ada", limit=5, evidence=False)
        shown = store.evidence(_ASKED, results, user_id="ada")
        note: dict = {}
        turns = store.turn_search(_ASKED, results, user_id="ada", shown=shown, note=note)
        assert f"{long_}:2" in [t.passage_id for t in turns]
        assert sum(estimate_tokens(turn_line(t)) + 1 for t in turns) <= budget
        # the judge was asked about exactly the lines of each excerpt
        asked = judge.excerpt_calls()[-1]
        best = next(item for item in note["excerpts"] if item["hit"] == f"{long_}:2")
        lines = [turn_line(t) for t in store._passage_turns(
            [store.backend.episodes_by_id([long_])[long_]]) if t.passage_id in best["passage_ids"]]
        assert any(text == _EXCERPT + " Conversation:\n" + "\n".join(lines) for text in asked)
        assert best["passage_ids"] == [f"{long_}:1", f"{long_}:2"]  # the next does not fit
    finally:
        store.close()


def test_short_messages_are_read_as_before():
    store = _store()
    try:
        saved = _save_long(store, [_fact("Ada went to Tokyo in March for a wedding", 1)])
        short = saved.episode_ids[0]
        results = store.search("When did Ada go to Tokyo?", user_id="ada", limit=5,
                               evidence=False)
        turns = store.evidence("When did Ada go to Tokyo?", results, user_id="ada")
        assert [t.passage_id for t in turns] == [f"{short}:0"]
        assert turn_line(turns[0]) == ("2 May 2026: user: "
                                       "We went to Tokyo in March for the wedding.")
    finally:
        store.close()


def test_the_memories_take_what_the_turns_leave():
    store = _store(ScriptedJudge(facts=0.2))  # under the bar: turn search is due
    try:
        for i in range(30):
            store.add(f"Tokyo note {i}: the trip had crowds", user_id="ada", infer=False)
        results = store.search("Tokyo trip crowds", user_id="ada", limit=30, evidence=False)
        budget = 500
        context = store.context_from("Tokyo trip crowds", results, user_id="ada",
                                     token_budget=budget, entities=[])
        assert context.turn_search["due"]
        assert not context.evidence and not context.excerpts  # each note says its turn whole
        share = min(store.config.retrieval.evidence_tokens, budget // 2)
        quoted = min(store.config.retrieval.turn_search_tokens, budget // 4)
        before = fitting(results, budget - share - quoted)
        assert len(context.results) > len(before)
        assert context.context.token_estimate <= budget
    finally:
        store.close()
