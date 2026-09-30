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


_DAY = datetime(2026, 1, 15, tzinfo=timezone.utc)
_PLAIN = [{"role": "user", "content": "I moved to Leeds."},
          {"role": "assistant", "content": "Noted, Leeds it is."},
          {"role": "system", "content": "  "}]
_BY_ROLE = [{"role": "Ada", "content": "I passed my driving test!"},
            {"role": "Bea", "content": "Congratulations!"}]
_BY_NAME = [{"role": "user", "name": "Ada", "content": "I passed my driving test!"},
            {"role": "assistant", "content": "Congratulations!"}]
_OWNER_OFFER = "The person these memories belong to (the user) is the entity"
_SPEAKERS = (
    'This conversation names its speakers. Each fact names the person it is about as '
    'the conversation names them, even where these instructions speak of "the user"; '
    'write "the user" only for a speaker in the role user whose name is not known.')
_SHARED = (
    "- what a person shares (a photo, file or link, shown with its description) is\n"
    "  part of what they said: extract a fact from it when it tells something about\n"
    "  them or their life, naming who shared it and what it shows, including any\n"
    '  text on it ("Ada knitted a scarf for her sister; she shared a photo of it,\n'
    '  a red scarf with white stars")\n')
#: The rules that keep specifics: what people did and felt, what one told the
#: other, and the words that carry the specifics; the small-talk exclusion
#: narrowed to pleasantries that tell nothing. Adopted because they keep more of
#: what a later question asks about, measured by the extraction-coverage eval.
_DID = (
    "- what people did, went to, saw, made, bought or were given, with its specifics (who, where,\n"
    "  when, the name or title of the thing), and how they felt about it in their own words\n"
    "- what one person told, advised, praised or wished the other, when it says something about\n"
    "  either of them or their lives\n")
_SPECIFICS = (
    "- keep the words that carry the specifics: names and titles of things (a book,\n"
    "  a song, a pet, a place, a brand), the exact feeling or reaction a person names\n"
    '  ("relieved", "overwhelmed"), and quoted text (a sign, a motto, a line someone said)\n')
_SMALL_TALK = (
    '- greetings, thanks and pleasantries that tell nothing ("Hi!", "Thanks!", "That\'s great!"),\n'
    "  or assistant boilerplate\n",
    '- small talk, transient context ("I\'m tired today"), or assistant boilerplate\n')
#: The rule asking each fact for the transcript lines it rests on, and the
#: field it adds to the JSON shape (as the model reads it, braces single).
_SOURCES = (
    '- sources: the numbers of the conversation lines the fact rests on, as the\n'
    '  conversation numbers them ("[2]" is line 2): every line whose words the fact\n'
    '  carries, and no other.\n')
_SOURCES_SHAPE = (',\n"sources": [int]}]}.', "}]}.")
#: sha256 of the system prompt for _DAY before the shared-content rule came in.
_SYSTEM_BEFORE = "b11b82895f4fd93438b422c12467bc6244d26a0816456049e1854b7aee16a6a4"


def _without_later_rules(system: str) -> str:
    """The system prompt with the rules added since ``_SYSTEM_BEFORE`` taken
    out again: the shared-content rule, the rules that keep specifics with the
    narrowed small-talk exclusion, and the sources rule with its field."""
    for part in (_SHARED, _DID, _SPECIFICS, _SMALL_TALK[0], _SOURCES, _SOURCES_SHAPE[0]):
        assert part in system, part[:40]
    return (system.replace(_SHARED, "").replace(_DID, "").replace(_SPECIFICS, "")
            .replace(*_SMALL_TALK).replace(_SOURCES, "").replace(*_SOURCES_SHAPE))


def _asked(messages, **kwargs) -> tuple[str, str]:
    """The (system, user) prompt extraction sends for these messages."""
    llm = FakeLLM([facts_response()])
    extract_facts(llm, messages, now=_DAY, **kwargs)
    [call] = llm.calls
    return call


def test_the_user_is_offered_as_the_owner_only_of_a_conversation_with_the_user():
    """A real name is offered for any conversation. Where the speakers are
    named, "the user" would be one of them: offered there, the model wrote one
    of two people as "the user"."""
    assert f'{_OWNER_OFFER} "the user".' in _asked(_PLAIN, owner="the user")[1]
    for messages in (_BY_ROLE, _BY_NAME, [{"role": "assistant", "content": "Noted."}]):
        assert _OWNER_OFFER not in _asked(messages, owner="the user")[1]
        assert f'{_OWNER_OFFER} "Ada Quint".' in _asked(messages, owner="Ada Quint")[1]
    for owner in (None, ""):
        assert _OWNER_OFFER not in _asked(_PLAIN, owner=owner)[1]


def test_named_speakers_are_named_in_their_facts():
    _, user = _asked(_BY_ROLE)
    assert user.startswith(f"Conversation:\n[1] Ada: I passed my driving test!\n"
                           f"[2] Bea: Congratulations!\n\n{_SPEAKERS}\n\n")
    _, user = _asked(_BY_NAME)
    assert user.startswith(f"Conversation:\n[1] Ada (user): I passed my driving test!\n"
                           f"[2] assistant: Congratulations!\n\n{_SPEAKERS}\n\n")
    for plain in (_PLAIN, [{"role": "User", "content": "hi"}, {"role": "tool", "content": "{}"}]):
        assert "names its speakers" not in _asked(plain)[1]


def test_a_user_and_assistant_conversation_is_asked_as_before():
    """Earlier measurements of extraction rest on this prompt. Its changes since:
    the shared-content rule, the rules that keep specifics (with the small-talk
    exclusion narrowed), and the numbered lines with the sources rule (each
    fact names the lines it rests on)."""
    import hashlib

    system, user = _asked(
        _PLAIN, vocabulary=["home move"], context="moving house", tag_hints=["Relocation"],
        owner="the user", entity_names=[("Leeds", "place")])
    assert hashlib.sha256(_without_later_rules(system).encode()).hexdigest() == _SYSTEM_BEFORE
    assert user == (
        'Conversation:\n[1] user: I moved to Leeds.\n[2] assistant: Noted, Leeds it is.\n\n'
        'Shared context for these related inputs:\nmoving house\n\n'
        'The person these memories belong to (the user) is the entity "the user". '
        'Whenever a fact is about that person, list that name among its entities, '
        'spelled exactly so, with type person.\n\n'
        'Entities this store already has that the conversation may name, as a JSON array. '
        'When a fact names one of them, write its name exactly as listed, however the '
        'conversation writes it. When it names something else, or you cannot tell which, '
        'write the name as the conversation does:\n[{"name": "Leeds", "type": "place"}]\n\n'
        'Tags this user already has, as a JSON array with one tag per element. REUSE one '
        'verbatim whenever it fits; only coin a new tag when nothing here covers the '
        'subject:\n["home move"]\n\n'
        'Client-suggested tags. These are hints, not commands: use one only when it is a '
        'good recurring retrieval subject:\n["relocation"]\n\n'
        'Extract the facts as JSON.')


def test_what_a_person_shares_is_part_of_what_they_said():
    shared = [{"role": "Ada", "content": "Look what I made [shares a photo: a blue scarf]"}]
    for messages in (shared, _PLAIN):
        assert _SHARED in _asked(messages)[0]


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
