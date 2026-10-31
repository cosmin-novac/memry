"""Question keys: with ``retrieval.question_keys`` on, the extractor writes
2 or 3 questions each fact answers, the store keeps them beside the memory
(``memory_questions``) with their vectors, and the search reads them as two
more candidate lists (``question_keyword``, ``question_vector``). On by
default. Off, the prompt, the schema and the search are those without them."""

from __future__ import annotations

import json

import pytest

from memry.config import Config
from memry.intelligence.extraction import (
    EXTRACTION_SCHEMA,
    EXTRACTION_SYSTEM,
    extraction_schema,
    extraction_system,
)
from memry.intelligence.questions import (
    QUESTIONS_LIMIT,
    QUESTIONS_RULE,
    clean_questions,
    merge_questions,
)
from memry.models import Memory, Scope
from memry.providers.embeddings import HashEmbedder
from memry.store import MemoryStore

from conftest import FakeLLM, decision, fact, facts_response

NO_GAPS = json.dumps({"missing": []})


def _make_store(question_keys: bool = True, llm: FakeLLM | None = None) -> MemoryStore:
    cfg = Config(db_path=":memory:")
    cfg.decision.auto_confirm_confidence = 0.95
    cfg.retrieval.question_keys = question_keys
    return MemoryStore(cfg, llm=llm or FakeLLM(), embedder=HashEmbedder(64))


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def store(llm):
    s = _make_store(True, llm)
    yield s
    s.close()


@pytest.fixture
def plain_store(llm):
    s = _make_store(False, llm)
    yield s
    s.close()


def _asked(content: str, questions: list, **kw) -> dict:
    return {**fact(content, **kw), "questions": questions}


def _memory(store, content, user_id="ada", categories=()):
    return store.backend.insert_memory(
        Memory(content=content, user_id=user_id, embedding_model=store.embedder.model_id,
               categories=list(categories)),
        embedding=store.embedder.embed([content])[0])


def _texts(store, memory_id):
    return [q["text"] for q in store.backend.questions_of([memory_id]).get(memory_id, [])]


def _ids(found):
    return [memory.id for memory, _ in found]


# ------------------------------------------------------------ prompt, schema

def test_the_prompt_and_schema_without_questions_are_unchanged():
    """With the flag off the prompt and the schema are byte for byte the ones
    from before, so the provider's prompt cache keeps holding."""
    assert extraction_system() == EXTRACTION_SYSTEM
    assert extraction_system(ask_questions=False) == EXTRACTION_SYSTEM
    assert extraction_schema() == EXTRACTION_SCHEMA
    assert extraction_schema(ask_questions=False) is EXTRACTION_SCHEMA
    assert "questions" not in json.dumps(EXTRACTION_SCHEMA)
    assert QUESTIONS_RULE not in EXTRACTION_SYSTEM


def test_asking_for_questions_adds_the_rule_and_the_field():
    """Asked for questions, the prompt carries the rule before the sources
    rule and the field in the shape, and the schema requires a list of
    strings on each fact without changing the schema it was built from."""
    text = extraction_system(ask_questions=True)
    assert QUESTIONS_RULE in text
    assert text.index(QUESTIONS_RULE) < text.index("- sources: the numbers")
    assert '"questions": [str],\n"sources": [int]}}]}}.' in text
    assert text.replace(QUESTIONS_RULE, "", 1).replace('"questions": [str],\n', "", 1) \
        == EXTRACTION_SYSTEM
    schema = extraction_schema(ask_questions=True)
    items = schema["properties"]["facts"]["items"]
    assert items["properties"]["questions"] == {"type": "array", "items": {"type": "string"}}
    assert "questions" in items["required"]
    assert "questions" not in EXTRACTION_SCHEMA["properties"]["facts"]["items"]["properties"]


def test_questions_and_user_name_together_carry_both_additions():
    """Asked for questions and the user's name, the prompt and the schema
    carry both additions, and the shape still ends as the user name prompt
    ends."""
    both = extraction_system(ask_questions=True, ask_user_name=True)
    named = extraction_system(ask_user_name=True)
    assert QUESTIONS_RULE in both
    assert "- user_name:" in both
    assert '"questions": [str],\n"sources": [int]}}], "user_name": str|null}}.' in both
    assert 'Return {{"facts": [], "user_name": null}} if' in both
    assert both.replace(QUESTIONS_RULE, "", 1).replace('"questions": [str],\n', "", 1) == named
    tail = named[named.rindex('"sources": [int]'):]
    assert both.endswith(tail)
    schema = extraction_schema(ask_questions=True, ask_user_name=True)
    assert "user_name" in schema["required"]
    assert "questions" in schema["properties"]["facts"]["items"]["required"]


# ------------------------------------------------------------ cleaning

def test_clean_questions_keeps_strings_on_one_line_each_once():
    """Non strings are dropped, whitespace collapses, a repeat ignoring case
    is dropped, a question over 200 characters is dropped and the limit is
    kept. Anything but a list reads as none."""
    raw = [" Where does  Ada\nlive? ", 3, None, {"q": 1}, "where does ada live?",
           "", "   ", "x" * 201, "y" * 200, "What does Ada do?"]
    assert clean_questions(raw) == ["Where does Ada live?", "y" * 200, "What does Ada do?"]
    assert clean_questions("Where?") == []
    assert clean_questions(None) == []
    many = [f"Question {i}?" for i in range(20)]
    assert clean_questions(many) == many[:QUESTIONS_LIMIT]
    assert clean_questions(many, limit=2) == many[:2]


def test_merge_questions_keeps_the_first_lists_order_and_each_once():
    """A merged list is the first list's questions in order, then the new
    ones of the next list, each once ignoring case."""
    assert merge_questions(["B?", "A?"], ["a?", "C?", "b?"]) == ["B?", "A?", "C?"]
    assert merge_questions([], ["X?"]) == ["X?"]
    assert merge_questions([f"Q{i}?" for i in range(6)], [f"R{i}?" for i in range(6)],
                           limit=4) == ["Q0?", "Q1?", "Q2?", "Q3?"]


# ------------------------------------------------------------ the save

def test_a_save_writes_the_questions_of_each_new_memory(store, llm):
    """With the flag on the extractor is asked for questions and each added
    memory keeps its own, in order, embedded by the store's embedder."""
    llm.queue(facts_response(
        _asked("Ada moved to Amsterdam in March 2024",
               ["Which city does Ada live in?", "When did Ada move?"]),
        _asked("Ada works as a nurse", ["What is Ada's job?"])), NO_GAPS)
    actions = store.add("I moved to Amsterdam in March 2024 and I work as a nurse",
                        user_id="ada").actions
    assert [a.event for a in actions] == ["ADD", "ADD"]
    assert "questions:" in llm.calls[0][0]
    assert QUESTIONS_RULE.splitlines()[0] in llm.calls[0][0]
    rows = store.backend.questions_of([a.memory_id for a in actions])
    assert [q["text"] for q in rows[actions[0].memory_id]] == [
        "Which city does Ada live in?", "When did Ada move?"]
    assert [q["text"] for q in rows[actions[1].memory_id]] == ["What is Ada's job?"]
    for qs in rows.values():
        assert all(q["source"] == "save" for q in qs)
        assert all(q["embedding_model"] == store.embedder.model_id for q in qs)
    assert store.backend.questions_without_vectors(Scope(user_id="ada"),
                                                   store.embedder.model_id) == []
    assert llm.responses == []


def test_the_defaults_turn_question_keys_entity_questions_and_the_log_on(llm):
    """Question keys, entity questions and the search log are on by default,
    keys from traffic off; a store with the defaults asks the extractor for
    questions with the prompt and schema built for them."""
    cfg = Config(db_path=":memory:")
    retrieval = cfg.retrieval
    assert (retrieval.question_keys, retrieval.entity_questions, retrieval.search_log,
            retrieval.traffic_keys) == (True, True, True, False)
    store = MemoryStore(cfg, llm=llm, embedder=HashEmbedder(64))
    try:
        llm.queue(facts_response(_asked("Ada works as a nurse", ["What is Ada's job?"])),
                  NO_GAPS)
        actions = store.add("I work as a nurse", user_id="ada").actions
        system = llm.calls[0][0]
        assert QUESTIONS_RULE in system and '"questions": [str],' in system
        assert _texts(store, actions[0].memory_id) == ["What is Ada's job?"]
    finally:
        store.close()


def test_with_the_flag_off_nothing_is_asked_or_written(plain_store, llm):
    """With the flag off the prompt carries no questions rule, and questions
    a model sends anyway are not written."""
    llm.queue(facts_response(
        _asked("Ada moved to Amsterdam in March 2024", ["Which city does Ada live in?"])),
        NO_GAPS)
    actions = plain_store.add("I moved to Amsterdam in March 2024", user_id="ada").actions
    assert [a.event for a in actions] == ["ADD"]
    assert QUESTIONS_RULE not in llm.calls[0][0]
    assert "- questions:" not in llm.calls[0][0]
    assert plain_store.backend.questions_of([actions[0].memory_id]) == {}


def test_an_update_keeps_the_questions_of_both_texts(store, llm):
    """A MORE writes a merged memory that answers what both texts answered:
    its questions are the new candidate's first, then those of the memory it
    replaced, each once ignoring case."""
    llm.queue(facts_response(_asked("User likes green tea",
                                    ["What tea does the user like?",
                                     "Does the user drink tea?"])), NO_GAPS)
    first = store.add("I like green tea", user_id="ada", run_id="s1").actions[0]
    assert _texts(store, first.memory_id) == ["What tea does the user like?",
                                              "Does the user drink tea?"]
    merged = "User likes green tea, brewed at 80 degrees"
    llm.queue(facts_response(_asked("User brews green tea at 80 degrees",
                                    ["At what temperature does the user brew tea?",
                                     "what tea does the user like?"])),
              decision("MORE", target=0, content=merged), facts_response(), NO_GAPS)
    more = store.add("I brew it at 80 degrees", user_id="ada", run_id="s1").actions[0]
    assert more.event == "UPDATE" and more.memory_id != first.memory_id
    assert llm.responses == []
    assert _texts(store, more.memory_id) == [
        "At what temperature does the user brew tea?",
        "what tea does the user like?",
        "Does the user drink tea?",
    ]


# ------------------------------------------------------------ search

def test_a_stored_question_finds_a_memory_sharing_no_word_with_the_query(store):
    """With the flag on, a query that shares no word with a memory's text
    but shares words with its stored question finds it through the
    question_keyword list."""
    target = _memory(store, "Ada relocated to Amsterdam in March 2024")
    _memory(store, "Bob plays chess on Sundays")
    store.backend.set_questions(target.id, [("Which city does Ada live in now?", "save")])
    results = store.search("city live", user_id="ada", limit=5)
    hit = next(r for r in results if r.memory.id == target.id)
    assert "question_keyword" in hit.signals
    assert "keyword" not in hit.signals


def test_with_the_flag_off_the_question_lists_are_not_read(plain_store):
    """With the flag off the same search carries no question signal, even
    when a stored question matches."""
    target = _memory(plain_store, "Ada relocated to Amsterdam in March 2024")
    plain_store.backend.set_questions(target.id,
                                      [("Which city does Ada live in now?", "save")])
    results = plain_store.search("city live", user_id="ada", limit=5)
    assert any(r.memory.id == target.id for r in results)
    for r in results:
        assert "question_keyword" not in r.signals
        assert "question_vector" not in r.signals


def test_question_vector_search_returns_the_nearest_question_and_keeps_to_scope(store):
    """The memory whose question vector is nearest the query comes first,
    and a scope of another user finds nothing."""
    near = _memory(store, "Ada relocated to Amsterdam")
    far = _memory(store, "Ada plays chess on Sundays")
    store._write_questions(near.id, ["Which city does Ada live in?"], "save")
    store._write_questions(far.id, ["What game does Ada play on weekends?"], "save")
    query = store.embedder.embed(["Which city does Ada live in?"])[0]
    model = store.embedder.model_id
    found = store.backend.question_vector_search(query, model, Scope(user_id="ada"))
    assert _ids(found)[0] == near.id
    assert found[0][1] == pytest.approx(1.0, abs=1e-2)
    assert set(_ids(found)) == {near.id, far.id}
    assert store.backend.question_vector_search(query, model, Scope(user_id="bob")) == []
    assert store.backend.question_vector_search(query, "other-model",
                                                Scope(user_id="ada")) == []
    assert store.backend.question_keyword_search("city", Scope(user_id="bob")) == []


def test_question_keyword_search_keeps_to_the_filters(store):
    """question_keyword_search keeps to among, categories and validity: an
    invalidated memory is found only with include_invalid."""
    trip = _memory(store, "Ada flew to Lisbon", categories=["travel"])
    work = _memory(store, "Ada took a job in Lisbon", categories=["work"])
    store.backend.set_questions(trip.id, [("Where did Ada go in Portugal?", "save")])
    store.backend.set_questions(work.id, [("Where in Portugal does Ada work?", "save")])
    scope = Scope(user_id="ada")
    search = store.backend.question_keyword_search
    assert set(_ids(search("Portugal", scope))) == {trip.id, work.id}
    assert _ids(search("Portugal", scope, among={work.id})) == [work.id]
    assert _ids(search("Portugal", scope, among=set())) == []
    assert _ids(search("Portugal", scope, categories=["travel"])) == [trip.id]
    store.backend.invalidate_memory(trip.id)
    assert _ids(search("Portugal", scope)) == [work.id]
    assert set(_ids(search("Portugal", scope, include_invalid=True))) == {trip.id, work.id}
    assert search("", scope) == []
    assert search("Brazil", scope) == []


# ------------------------------------------------------------ lifecycle

def test_deleting_a_memory_removes_its_questions_and_invalidating_keeps_them(store):
    """A memory deleted for good takes its questions with it, from the table
    and from the word index; a memory only invalidated keeps them."""
    gone = _memory(store, "Ada owns a red bicycle")
    kept = _memory(store, "Ada owns a blue kayak")
    store._write_questions(gone.id, ["What colour is Ada's bicycle?"], "save")
    store._write_questions(kept.id, ["What boat does Ada own?"], "save")
    assert store.backend.delete_memory(gone.id)
    assert store.backend.questions_of([gone.id]) == {}
    assert store.backend.question_keyword_search(
        "bicycle", Scope(user_id="ada"), include_invalid=True) == []
    store.backend.invalidate_memory(kept.id)
    assert _texts(store, kept.id) == ["What boat does Ada own?"]


def test_a_backup_carries_the_question_texts_and_a_restore_embeds_them(store):
    """A backup carries the questions' texts and not their vectors; a
    backend restore brings the texts back without vectors, and the store's
    restore with the flag on embeds them again."""
    memory = _memory(store, "Ada relocated to Amsterdam")
    store._write_questions(memory.id, ["Which city does Ada live in?",
                                       "When did Ada move?"], "save")
    backup = store.export_backup(user_id="ada")
    rows = backup["tables"]["memory_questions"]
    assert [r["text"] for r in rows] == ["Which city does Ada live in?", "When did Ada move?"]
    assert all("embedding" not in r and "embedding_model" not in r for r in rows)
    json.dumps(backup)  # a backup is plain JSON

    fresh = _make_store(False)
    try:
        fresh.backend.import_backup(backup)
        assert [q["text"] for q in fresh.backend.questions_of([memory.id])[memory.id]] == [
            "Which city does Ada live in?", "When did Ada move?"]
        assert len(fresh.backend.questions_without_vectors(
            Scope(user_id="ada"), fresh.embedder.model_id)) == 2
        assert _ids(fresh.backend.question_keyword_search(
            "city", Scope(user_id="ada"))) == [memory.id]
    finally:
        fresh.close()

    keyed = _make_store(True)
    try:
        keyed.import_backup(backup)
        assert keyed.backend.questions_without_vectors(
            Scope(user_id="ada"), keyed.embedder.model_id) == []
        assert all(q["embedding_model"] == keyed.embedder.model_id
                   for q in keyed.backend.questions_of([memory.id])[memory.id])
    finally:
        keyed.close()


def test_an_older_backup_without_the_questions_table_restores(store):
    """A backup from before question keys has no memory_questions table and
    restores all the same, with no questions."""
    memory = _memory(store, "Ada relocated to Amsterdam")
    backup = store.export_backup(user_id="ada")
    tables = {k: v for k, v in backup["tables"].items() if k != "memory_questions"}
    older = {**backup, "tables": tables}
    fresh = _make_store(True)
    try:
        fresh.import_backup(older)
        assert fresh.get(memory.id).content == "Ada relocated to Amsterdam"
        assert fresh.backend.questions_of([memory.id]) == {}
    finally:
        fresh.close()


def test_a_backup_question_of_a_memory_outside_it_is_refused(store):
    """A question row whose memory is not in the backup is refused."""
    _memory(store, "Ada relocated to Amsterdam")
    backup = store.export_backup(user_id="ada")
    backup["tables"]["memory_questions"] = [
        {"memory_id": "nowhere", "n": 0, "text": "Where?", "source": "save"}]
    fresh = _make_store(True)
    try:
        with pytest.raises(ValueError, match="question key"):
            fresh.import_backup(backup)
    finally:
        fresh.close()


def test_refresh_question_vectors_embeds_what_has_no_vector_once(store):
    """refresh_question_vectors embeds the questions of valid memories with
    no vector from the store's embedder or one from another model, returns
    how many, and finds nothing the second time."""
    a = _memory(store, "Ada relocated to Amsterdam")
    b = _memory(store, "Ada plays chess")
    gone = _memory(store, "Ada had a cat")
    store.backend.set_questions(a.id, [("Which city does Ada live in?", "backfill"),
                                       ("When did Ada move?", "backfill")])
    store.backend.set_questions(b.id, [("What game does Ada play?", "agent")],
                                [[1.0] * 64], "old-model")
    store.backend.set_questions(gone.id, [("Did Ada have a pet?", "save")])
    store.backend.invalidate_memory(gone.id)
    assert store.refresh_question_vectors(user_id="ada") == 3
    assert store.refresh_question_vectors(user_id="ada") == 0
    rows = store.backend.questions_of([a.id, b.id, gone.id])
    assert all(q["embedding_model"] == store.embedder.model_id for q in rows[a.id] + rows[b.id])
    assert rows[gone.id][0]["embedding_model"] is None


def test_the_weekly_upkeep_embeds_question_keys(store):
    """With the flag on, the upkeep tick that is due embeds the question keys
    without a vector and reports how many under question_vectors; with
    nothing left it reports none."""
    memory = _memory(store, "Ada relocated to Amsterdam")
    store.backend.set_questions(memory.id, [("Which city does Ada live in?", "backfill")])
    ran = store.run_upkeep_cycle(user_id="ada")
    assert ran.get("question_vectors") == {"embedded": 1}
    assert store.backend.questions_without_vectors(Scope(user_id="ada"),
                                                   store.embedder.model_id) == []
