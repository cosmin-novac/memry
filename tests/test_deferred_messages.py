"""A conversation saved as a list of messages through the deferred path
(``POST /api/v1/memories`` with ``defer``, ``MemoryStore.add_deferred``) keeps
its speakers as a direct save does: one episode per message with its role, the
same extraction input once the worker distills it (the numbered lines, the
speaker instruction, the said day) and the same evidence afterwards. A text
saved through the deferred path is kept as it always was."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from starlette.testclient import TestClient

from conftest import FakeLLM, decision, facts_response

from memry.config import Config
from memry.providers.embeddings import HashEmbedder
from memry.rest import create_app
from memry.store import MemoryStore

SAID = "2023-05-08"

CONVERSATION = [
    {"role": "Ada", "content": "I got the job at the Lisbon aquarium."},
    {"role": "Bea", "content": "  "},  # says nothing: no episode, no line
    {"role": "Bea", "content": "Congratulations! When do you start at the aquarium?"},
    {"role": "user", "name": "Kai", "content": "She starts on Monday, I will drive her."},
    {"role": "Ada", "content": "Monday, and Kai drives me to the aquarium on the first day."},
]

#: What each message says, by the role its episode keeps, in the order said.
TURNS = [
    ("Ada", "I got the job at the Lisbon aquarium."),
    ("Bea", "Congratulations! When do you start at the aquarium?"),
    ("user", "She starts on Monday, I will drive her."),
    ("Ada", "Monday, and Kai drives me to the aquarium on the first day."),
]


class _Recording(FakeLLM):
    """Records every prompt. Extraction answers the queued facts; every
    reconcile question is NEW and the coverage audit finds nothing missing."""

    def complete(self, system: str, user: str, *, json_schema=None) -> str:
        self.calls.append((system, user))
        if system.startswith("You are the long-term memory extraction system"):
            return self.responses.pop(0)
        if system.startswith("You audit"):
            return json.dumps({"missing": []})
        return decision("NEW")

    def extractions(self) -> list[tuple[str, str]]:
        return [call for call in self.calls
                if call[0].startswith("You are the long-term memory extraction system")]


def _fact(content: str, *sources: int) -> dict:
    return {"content": content, "type": "episodic", "importance": 0.6, "categories": [],
            "entities": [], "relations": [], "when": None, "sources": list(sources)}


def _facts() -> str:
    return facts_response(
        _fact("Ada got a job at the Lisbon aquarium", 1),
        _fact("Ada starts her job at the Lisbon aquarium on Monday", 3, 4),
    )


def _store(llm: FakeLLM) -> MemoryStore:
    return MemoryStore(Config(db_path=":memory:"), llm=llm, embedder=HashEmbedder(64))


def _save(body: dict, *, defer: bool) -> tuple[MemoryStore, _Recording, dict]:
    """Save ``body`` over REST, with or without ``defer``; a deferred save is
    then distilled as the worker does once its group has been quiet."""
    llm = _Recording([_facts()])
    store = _store(llm)
    with TestClient(create_app(store)) as client:
        response = client.post("/api/v1/memories",
                               json={**body, "user_id": "ada", "defer": defer})
    assert response.status_code == (202 if defer else 201), response.text
    if defer:
        outcome = store.process_pending_enrichments(
            quiet_seconds=120, now=datetime.now(timezone.utc) + timedelta(seconds=300))
        assert outcome["succeeded"] == 1, outcome
    return store, llm, response.json()


def _said(store: MemoryStore, saved: dict) -> list[tuple[str, str, str]]:
    """The turns a save kept, in the order said: speaker, text, time."""
    episodes = store.backend.episodes_by_id(saved["episode_ids"])
    return [(episodes[i].role, episodes[i].content, episodes[i].created_at)
            for i in saved["episode_ids"]]


def _evidence(store: MemoryStore) -> set[tuple[str, str, str]]:
    found = store.search("When does Ada start at the Lisbon aquarium?", user_id="ada")
    return {(t.speaker, t.content, t.said_at) for r in found for t in r.evidence}


def _facts_of(store: MemoryStore) -> dict[str, tuple[list[tuple[str, str]], str]]:
    """Each fact in use: the turns it rests on (speaker, text) and its time."""
    episodes = {e.id: e for e in store.episodes(user_id="ada")}
    return {
        m.content: ([(episodes[i].role, episodes[i].content) for i in m.source_episode_ids],
                    m.created_at)
        for m in store.get_all(user_id="ada")
    }


# ---------------------------------------------------------------- REST, both paths


def test_a_deferred_conversation_keeps_its_speakers_as_a_direct_save_does():
    direct, direct_llm, saved = _save({"messages": CONVERSATION}, defer=False)
    deferred, deferred_llm, queued = _save({"messages": CONVERSATION}, defer=True)

    # one episode per message that says something, with its speaker
    assert [(role, text) for role, text, _ in _said(deferred, queued)] == TURNS
    assert [(role, text) for role, text, _ in _said(direct, saved)] == TURNS
    assert len(deferred.episodes(user_id="ada")) == len(TURNS)

    # the worker's extraction reads what the direct save's reads
    [direct_prompt] = direct_llm.extractions()
    [deferred_prompt] = deferred_llm.extractions()
    assert deferred_prompt == direct_prompt
    user = deferred_prompt[1]
    assert (
        "Conversation:\n"
        "[1] Ada: I got the job at the Lisbon aquarium.\n"
        "[2] Bea: Congratulations! When do you start at the aquarium?\n"
        "[3] Kai (user): She starts on Monday, I will drive her.\n"
        "[4] Ada: Monday, and Kai drives me to the aquarium on the first day.\n"
    ) in user
    assert "This conversation names its speakers" in user

    # the facts rest on the turns of their lines, and show them as evidence
    assert {c: turns for c, (turns, _) in _facts_of(deferred).items()} == {
        "Ada got a job at the Lisbon aquarium": [TURNS[0]],
        "Ada starts her job at the Lisbon aquarium on Monday": TURNS[2:4],
    }
    assert {c: turns for c, (turns, _) in _facts_of(direct).items()} == {
        c: turns for c, (turns, _) in _facts_of(deferred).items()}
    evidence = {(speaker, text) for speaker, text, _ in _evidence(deferred)}
    assert evidence == {(speaker, text) for speaker, text, _ in _evidence(direct)}
    assert ("Ada", TURNS[3][1]) in evidence
    assert not any(text.lstrip().startswith("[{") for _, text in evidence)
    direct.close()
    deferred.close()


def test_with_said_at_both_paths_date_the_conversation_the_same():
    body = {"messages": CONVERSATION, "said_at": SAID}
    direct, direct_llm, saved = _save(body, defer=False)
    deferred, deferred_llm, queued = _save(body, defer=True)

    day = f"{SAID}T00:00:00+00:00"
    assert _said(deferred, queued) == _said(direct, saved) == [
        (role, text, day) for role, text in TURNS]
    assert deferred_llm.extractions() == direct_llm.extractions()
    assert f"Today's date is {SAID}." in deferred_llm.extractions()[0][0]
    assert _facts_of(deferred) == _facts_of(direct)
    assert {created for _, created in _facts_of(deferred).values()} == {day}
    assert _evidence(deferred) == _evidence(direct)
    assert {said for _, _, said in _evidence(deferred)} == {day}
    direct.close()
    deferred.close()


def test_a_deferred_text_is_saved_and_extracted_as_before():
    llm = _Recording([facts_response(_fact("Ada got a job at the Lisbon aquarium", 1))])
    store = _store(llm)
    with TestClient(create_app(store)) as client:
        response = client.post("/api/v1/memories", json={
            "content": "  Ada: I got the job at the Lisbon aquarium.\n", "user_id": "ada",
            "defer": True, "metadata": {"context": "Ada's job"}})
    assert response.status_code == 202, response.text
    text = "Ada: I got the job at the Lisbon aquarium."
    pending = store.get(response.json()["actions"][0]["memory_id"])
    assert pending.content == text
    [episode] = store.episodes(user_id="ada")
    assert (episode.role, episode.content) == ("user", text)
    assert pending.source_episode_ids == [episode.id] == response.json()["episode_ids"]
    assert pending.metadata["context"] == "Ada's job"
    assert set(pending.metadata["_enrichment"]) == {"status", "attempts", "queued_at"}

    store.process_pending_enrichments(
        quiet_seconds=120, now=datetime.now(timezone.utc) + timedelta(seconds=300))
    [(_, user)] = llm.extractions()
    assert f"Conversation:\n[1] user: {text}\n" in user
    assert "This conversation names its speakers" not in user
    assert "Shared context for these related inputs:\nAda's job" in user
    [fact] = store.get_all(user_id="ada")
    assert fact.source_episode_ids == [episode.id]
    store.close()


@pytest.mark.parametrize("defer", [False, True])
@pytest.mark.parametrize("messages", [{"role": "Ada", "content": "Hi"}, ["Ada: Hi"], 42])
def test_rest_refuses_messages_that_are_not_a_list_of_objects(messages, defer):
    store = _store(FakeLLM())
    with TestClient(create_app(store)) as client:
        response = client.post("/api/v1/memories", json={
            "messages": messages, "user_id": "ada", "defer": defer})
    assert response.status_code == 400, response.text
    assert "messages" in response.json()["error"]
    assert store.get_all(user_id="ada") == [] and store.episodes(user_id="ada") == []
    store.close()


# ---------------------------------------------------------------- the store


def test_a_deferred_conversation_is_searchable_as_its_speakers_lines():
    store = _store(FakeLLM())
    result = store.add_deferred(CONVERSATION, user_id="ada", run_id="r")
    [action] = result.actions
    pending = store.get(action.memory_id)
    assert pending.content == action.content == (
        "Ada: I got the job at the Lisbon aquarium.\n"
        "Bea: Congratulations! When do you start at the aquarium?\n"
        "Kai: She starts on Monday, I will drive her.\n"
        "Ada: Monday, and Kai drives me to the aquarium on the first day.")
    assert pending.source_episode_ids == result.episode_ids
    assert pending.metadata["pending_distillation"] is True
    assert [r.memory.id for r in store.search("Kai drives", user_id="ada")] == [pending.id]
    assert store.llm.calls == []  # a deferred save asks no provider
    assert store.add_deferred([{"role": "Ada", "content": " "}], user_id="ada").actions == []
    store.close()


def _line(user: str, words: str) -> int:
    """The number the extraction prompt gives the line saying ``words``."""
    [number] = [int(line[1:line.index("]")]) for line in user.splitlines()
                if line.startswith("[") and words in line]
    return number


#: The fact each extraction answers for a line saying these words.
_STATED = {
    "I got the job": "Ada got a job at the Lisbon aquarium",
    "new bike": "Ada bought a new bike",
    "old car": "Ada sold her old car",
    "kitchen": "Ada painted her kitchen",
}


class _ByLines(_Recording):
    """Answers each extraction with a fact for each line it knows (``_STATED``),
    resting on that line."""

    def complete(self, system: str, user: str, *, json_schema=None) -> str:
        if not system.startswith("You are the long-term memory extraction system"):
            return super().complete(system, user, json_schema=json_schema)
        self.calls.append((system, user))
        return facts_response(*(_fact(fact, _line(user, words))
                                for words, fact in _STATED.items() if words in user))


def test_a_deferred_conversation_is_grouped_with_the_saves_it_was_grouped_with_before():
    """The quiet-period group is as before (scope and run, context label, said
    day): each message of a conversation is its own line, and a text is one
    line said by the user, numbered on across the group."""
    llm = _ByLines()
    store = _store(llm)
    week = {"context": "Ada's week"}
    conversation = store.add_deferred(CONVERSATION, user_id="ada", run_id="r", metadata=week)
    text = store.add_deferred("Ada: I bought a new bike.", user_id="ada", run_id="r",
                              metadata=week)
    other_run = store.add_deferred([{"role": "Ada", "content": "I sold my old car."}],
                                   user_id="ada", run_id="other", metadata=week)
    other_day = store.add_deferred([{"role": "Ada", "content": "I painted the kitchen."}],
                                   user_id="ada", run_id="r", metadata=week,
                                   created_at=f"{SAID}T00:00:00+00:00")
    other_label = store.add_deferred("Ada: I painted the kitchen.", user_id="ada",
                                     run_id="r", metadata={"context": "Ada's flat"})

    outcome = store.process_pending_enrichments(
        quiet_seconds=120, now=datetime.now(timezone.utc) + timedelta(seconds=300))

    assert outcome["succeeded"] == 5
    extractions = [user for _, user in llm.extractions()]
    assert len(extractions) == 4  # the other run, day and label each apart
    [grouped] = [user for user in extractions if "new bike" in user]
    assert "old car" not in grouped and "kitchen" not in grouped
    # the saves of one second are taken in either order
    assert "[1] user: Ada: I bought a new bike.\n[2] Ada: I got the job" in grouped or (
        "[4] Ada: Monday, and Kai drives me to the aquarium on the first day.\n"
        "[5] user: Ada: I bought a new bike.\n") in grouped
    assert "Shared context for these related inputs:\nAda's week" in grouped
    facts = {m.content: m for m in store.get_all(user_id="ada", run_id="r")}
    assert facts["Ada got a job at the Lisbon aquarium"].source_episode_ids == (
        conversation.episode_ids[:1])
    assert facts["Ada bought a new bike"].source_episode_ids == text.episode_ids
    for save in (conversation, text, other_run, other_day, other_label):
        assert store.get(save.actions[0].memory_id).invalid_at is not None
    store.close()


def test_a_distilled_group_numbers_a_conversation_and_a_text_in_the_order_given():
    llm = _ByLines()
    store = _store(llm)
    text = store.add_deferred("Ada: I bought a new bike.", user_id="ada")
    conversation = store.add_deferred(CONVERSATION, user_id="ada")
    store._distill_pending_group([text.actions[0].memory_id,
                                  conversation.actions[0].memory_id])
    [(_, user)] = llm.extractions()
    assert (
        "Conversation:\n"
        "[1] user: Ada: I bought a new bike.\n"
        "[2] Ada: I got the job at the Lisbon aquarium.\n"
        "[3] Bea: Congratulations! When do you start at the aquarium?\n"
        "[4] Kai (user): She starts on Monday, I will drive her.\n"
        "[5] Ada: Monday, and Kai drives me to the aquarium on the first day.\n") in user
    facts = {m.content: m for m in store.get_all(user_id="ada")}
    assert facts["Ada got a job at the Lisbon aquarium"].source_episode_ids == (
        conversation.episode_ids[:1])
    assert facts["Ada bought a new bike"].source_episode_ids == text.episode_ids
    store.close()


def test_a_conversation_edited_while_it_waits_is_extracted_as_edited():
    """An edit of the waiting text is what the person wants remembered: it is
    extracted as any edited pending text is, one line said by the user."""
    llm = _ByLines()
    store = _store(llm)
    conversation = store.add_deferred(CONVERSATION, user_id="ada")
    memory_id = conversation.actions[0].memory_id
    store.update(memory_id, content="Ada: I bought a new bike.")
    edited = len(llm.extractions())  # the edit's own analysis reads the new text
    store._distill_pending_group([memory_id])
    [(_, user)] = llm.extractions()[edited:]
    assert "Conversation:\n[1] user: Ada: I bought a new bike.\n" in user
    assert "aquarium" not in user
    [fact] = store.get_all(user_id="ada")
    assert fact.content == "Ada bought a new bike"
    assert fact.source_episode_ids == conversation.episode_ids
    store.close()


# ---------------------------------------------------------------- a speaker's name


def test_a_named_message_is_shown_by_its_name_and_keeps_its_role_on_both_paths():
    """A message's ``name`` is the speaker a turn shows as evidence, as
    extraction reads "Kai (user)"; the episode keeps the role, and the role
    decides what it decides (here: the speakers are named, so no owner is
    offered as "the user")."""
    direct, direct_llm, saved = _save({"messages": CONVERSATION}, defer=False)
    deferred, deferred_llm, queued = _save({"messages": CONVERSATION}, defer=True)
    for store, result, llm in ((direct, saved, direct_llm), (deferred, queued, deferred_llm)):
        episodes = store.backend.episodes_by_id(result["episode_ids"])
        kai = episodes[result["episode_ids"][2]]
        assert (kai.role, kai.name, kai.speaker) == ("user", "Kai", "Kai")
        assert [episodes[i].name for i in result["episode_ids"]] == [None, None, "Kai", None]
        speakers = {text: speaker for speaker, text, _ in _evidence(store)}
        assert speakers[TURNS[2][1]] == "Kai"
        assert speakers[TURNS[0][1]] == "Ada"
        [(_, user)] = llm.extractions()
        assert "[3] Kai (user): She starts on Monday" in user
        assert "The person these memories belong to" not in user
        store.close()


def test_a_plain_user_turn_is_still_shown_as_user():
    llm = _Recording([facts_response(_fact("Ada got a job at the Lisbon aquarium", 1))])
    store = _store(llm)
    saved = store.add([{"role": "user", "content": "I got the job at the Lisbon aquarium."}],
                      user_id="ada")
    [episode] = store.backend.episodes_by_id(saved.episode_ids).values()
    assert (episode.role, episode.name, episode.speaker) == ("user", None, "user")
    assert {t.speaker for r in store.search("Lisbon aquarium job", user_id="ada")
            for t in r.evidence} == {"user"}
    [(_, user)] = llm.extractions()
    assert 'belong to (the user) is the entity "the user"' in user  # the role decides
    store.close()


def test_a_store_from_before_names_shows_each_turn_by_its_role(tmp_path):
    """An episode table without the name column gains it, empty: every turn
    saved before is shown by its role, as it was."""
    import sqlite3

    path = str(tmp_path / "old.db")
    llm = _Recording([_facts()])
    store = MemoryStore(Config(db_path=path), llm=llm, embedder=HashEmbedder(64))
    store.add(CONVERSATION, user_id="ada")
    store.close()
    db = sqlite3.connect(path)
    db.execute("ALTER TABLE episodes DROP COLUMN name")
    db.commit()
    db.close()

    store = MemoryStore(Config(db_path=path), llm=_Recording(), embedder=HashEmbedder(64))
    assert {e.name for e in store.episodes(user_id="ada")} == {None}
    speakers = {text: speaker for speaker, text, _ in _evidence(store)}
    assert speakers[TURNS[2][1]] == "user" and speakers[TURNS[0][1]] == "Ada"
    store.close()
