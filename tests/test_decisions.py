"""Decision providers: the typed-judgement layer and its Jev implementation.

The Jev tests drive a mock transport rather than the live API. What they pin is
the half Memry controls - the request it sends and its reading of the reply -
plus the part that matters most in production: every failure mode has to come
back as "no answer" so a save still completes.
"""

from __future__ import annotations

import json
import re

import httpx
import pytest

from conftest import FakeLLM
from memry.config import Config, DecisionConfig
from memry.intelligence.entities import IDENTITY_QUESTION, IDENTITY_SYSTEM
from memry.providers.decisions import (
    MEASURED_MERGE_GATES,
    NEVER_AUTO_MERGE,
    Answer,
    Choice,
    JevDecider,
    LLMDecider,
    Noul,
    NoneDecider,
    Score,
    build_decider,
    merge_gate_for,
)
from memry.models import Memory
from memry.providers.embeddings import HashEmbedder
from memry.providers.llm import NoneLLM
from memry.store import MemoryStore

QUESTIONS = {
    "identity": Choice(instructions="same person?",
                       criteria={"same": "clearly", "different": "clearly not",
                                 "unsure": "cannot tell"}),
    "urgency": Score(instructions="how urgent", levels=["low", "medium", "high"]),
    "is_question": Noul(instructions="the state asks a question"),
}


def jev(handler, **cfg) -> JevDecider:
    """A JevDecider whose HTTP client is backed by ``handler``."""
    decider = JevDecider(DecisionConfig(provider="jev", api_key="k", **cfg))
    decider._client = httpx.Client(transport=httpx.MockTransport(handler),
                                   headers={"authorization": "Bearer k"})
    return decider


# ---------------------------------------------------------------- selection
def test_the_flag_is_off_by_default():
    """Memry gives a store built in code without settings no decision
    provider, and sends none of its decision questions to the text model; that
    takes a setting. A server does not start without one
    (config.require_models)."""
    assert Config().decision.provider is None
    assert build_decider(DecisionConfig(), NoneLLM()).available is False
    assert build_decider(DecisionConfig(), FakeLLM()).name == "none"


def test_build_decider_picks_the_configured_provider():
    assert build_decider(DecisionConfig(provider="none"), FakeLLM()).name == "none"
    assert build_decider(DecisionConfig(provider="llm"), FakeLLM()).name == "llm"
    assert build_decider(
        DecisionConfig(provider="jev", api_key="k"), FakeLLM()
    ).name == "jev"


def test_jev_without_a_key_abstains_instead_of_calling():
    decider = JevDecider(DecisionConfig(provider="jev"))
    assert decider.available is False
    assert decider.decide("state", QUESTIONS)["identity"].available is False


# ---------------------------------------------------------------- jev wire
def test_jev_sends_one_call_carrying_every_question():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"answers": {}})

    jev(handler).decide("Ada lives in Amsterdam", QUESTIONS)

    assert seen["url"].endswith("/v1/systemone")
    assert seen["auth"] == "Bearer k"
    assert seen["model"] == "jev-latest"
    assert seen["state"] == "Ada lives in Amsterdam"
    # one request, all three questions, each carrying its own type
    assert set(seen["questions"]) == {"identity", "urgency", "is_question"}
    assert seen["questions"]["identity"]["type"] == "choice"
    assert seen["questions"]["identity"]["criteria"]["same"] == "clearly"
    assert seen["questions"]["urgency"]["criteria"] == ["low", "medium", "high"]
    assert seen["questions"]["is_question"] == {
        "type": "noul", "instructions": "the state asks a question"}


def test_jev_reads_back_each_answer_type():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"answers": {
            "identity": {"choice": "same",
                         "probabilities": {"same": 0.94, "different": 0.04, "unsure": 0.02},
                         "confidence": 0.94},
            "urgency": {"score": 2, "confidence": 0.7},
            "is_question": {"noul": 0.12},
        }})

    answers = jev(handler).decide("state", QUESTIONS)

    assert answers["identity"].value == "same"
    assert answers["identity"].confidence == pytest.approx(0.94)
    assert answers["identity"].probabilities["different"] == pytest.approx(0.04)
    assert answers["urgency"].value == 2          # index into the levels
    assert answers["is_question"].value == pytest.approx(0.12)
    assert all(answers[k].available for k in QUESTIONS)


def test_jev_derives_confidence_from_the_distribution_when_absent():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"answers": {"identity": {
            "choice": "unsure", "probabilities": {"same": 0.3, "different": 0.25,
                                                  "unsure": 0.45}}}})

    answer = jev(handler).decide("state", {"identity": QUESTIONS["identity"]})["identity"]
    assert answer.value == "unsure"
    assert answer.confidence == pytest.approx(0.45)


@pytest.mark.parametrize("handler, label", [
    (lambda r: httpx.Response(500, text="boom"), "server error"),
    (lambda r: httpx.Response(429, text="slow down"), "rate limited"),
    (lambda r: httpx.Response(200, text="not json"), "unparseable body"),
    (lambda r: httpx.Response(200, json={"answers": "nonsense"}), "wrong shape"),
    (lambda r: httpx.Response(200, json={"answers": {"identity": {"choice": "maybe"}}}),
     "option not in the schema"),
])
def test_every_jev_failure_abstains_rather_than_raising(handler, label):
    """A decision provider must never be able to fail a memory write."""
    answers = jev(handler).decide("state", {"identity": QUESTIONS["identity"]})
    assert answers["identity"].available is False, label
    assert answers["identity"].value is None, label


def test_jev_transport_error_abstains():
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    assert jev(handler).decide("s", QUESTIONS)["identity"].available is False


def test_base_url_is_configurable_for_self_hosted_or_proxied_endpoints():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"answers": {}})

    jev(handler, base_url="https://proxy.example/v9/").decide("s", QUESTIONS)
    assert seen["url"] == "https://proxy.example/v9/systemone"


# ---------------------------------------------------------------- llm decider
def test_llm_decider_coerces_answers_into_the_question_type():
    llm = FakeLLM([json.dumps({
        "identity": {"answer": "different", "confidence": 0.8},
        "urgency": {"answer": 1, "confidence": 0.6},
        "is_question": {"answer": 0.9},
    })])
    answers = LLMDecider(llm).decide("state", QUESTIONS)
    assert answers["identity"].value == "different"
    assert answers["urgency"].value == 1
    assert answers["is_question"].value == pytest.approx(0.9)


def test_llm_decider_rejects_an_answer_outside_the_schema():
    """The point of the typed layer: a text model cannot invent an option."""
    llm = FakeLLM([json.dumps({"identity": {"answer": "probably", "confidence": 0.99}})])
    answers = LLMDecider(llm).decide("state", {"identity": QUESTIONS["identity"]})
    assert answers["identity"].available is False


def test_llm_decider_rejects_an_out_of_range_score():
    llm = FakeLLM([json.dumps({"urgency": {"answer": 7, "confidence": 0.9}})])
    answers = LLMDecider(llm).decide("state", {"urgency": QUESTIONS["urgency"]})
    assert answers["urgency"].available is False


def test_unavailable_llm_decider_abstains():
    assert LLMDecider(NoneLLM()).decide("s", QUESTIONS)["identity"].available is False


def test_none_decider_abstains_for_every_question():
    answers = NoneDecider().decide("state", QUESTIONS)
    assert [answers[k].available for k in QUESTIONS] == [False, False, False]


def test_a_missing_key_reads_as_unavailable_not_as_a_verdict():
    assert Answer().available is False and Answer().value is None
    assert NoneDecider().decide("s", {})["anything"].available is False


# ---------------------------------------------------------------- integration
def test_identity_question_covers_the_three_verdicts():
    """The store gates merges on these exact strings."""
    assert set(IDENTITY_QUESTION.criteria) == {"same", "different", "unsure"}


def test_store_defaults_to_no_decision_provider():
    store = MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(),
                        embedder=HashEmbedder(64))
    assert store.decider.name == "none" and store.decider.available is False
    store.close()


def test_a_decider_that_is_not_calibrated_is_not_asked_about_identity_at_save():
    """Without a calibrated judge a save asks no identity question, of the
    decider or of the text model: a confident "different" from a decider
    whose answers carry no computed probabilities could not be used, and a
    name the store already has joins its entity by rule."""
    from conftest import fact, facts_response

    class AlwaysDifferent(NoneDecider):
        name = "stub"
        available = True

        def __init__(self) -> None:
            self.asked: list[str] = []

        def decide(self, state, questions):
            from memry.providers.decisions import Answers

            self.asked.extend(questions)
            return Answers({k: Answer("different", {"different": 0.96}, 0.96, True)
                            for k in questions})

    llm = FakeLLM()
    decider = AlwaysDifferent()
    store = MemoryStore(Config(db_path=":memory:"), llm=llm, embedder=HashEmbedder(64),
                        decider=decider)
    llm.queue(facts_response(fact("Jonas cooks Thai food", entities=["Jonas"])))
    store.add("my partner Jonas cooks Thai", user_id="ada")
    llm.queue(facts_response(fact("Jonas reviewed the design doc", entities=["Jonas"])),
              json.dumps({"action": "ADD", "target": None, "content": None, "reason": "new"}))
    store.add("a colleague named Jonas reviewed the doc", user_id="ada")

    [jonas] = store.entities(user_id="ada")
    assert store.backend.count_entity_memories(jonas.id) == 2
    assert "identity" not in decider.asked and "pair" not in decider.asked
    assert all(system != IDENTITY_SYSTEM for system, _ in llm.calls)
    store.close()


def test_stats_reports_the_decision_provider():
    """The dashboard's About panel reads this, and hides the row when it's off."""
    store = MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(),
                        embedder=HashEmbedder(64))
    assert store.stats()["decider"] == "none"
    store.close()

    store = MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(),
                        embedder=HashEmbedder(64),
                        decider=JevDecider(DecisionConfig(provider="jev", api_key="k")))
    assert store.stats()["decider"] == "jev:jev-latest"
    store.close()


def test_score_is_a_weighted_average_not_an_index():
    """The API returns the probability-weighted average of the levels, which
    falls between them. Truncating it to an int throws the signal away."""
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"answers": {"urgency": {
            "type": "score", "score": 1.7, "confidence": 0.9,
            "legend": {"0": "low", "1": "medium", "2": "high"},
            "probabilities": {"0": 0.1, "1": 0.1, "2": 0.8}}}})

    answer = jev(handler).decide("s", {"urgency": QUESTIONS["urgency"]})["urgency"]
    assert answer.value == pytest.approx(1.7)


def test_score_outside_the_rubric_abstains():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"answers": {
            "urgency": {"score": 9.0, "confidence": 0.9}}})

    assert jev(handler).decide("s", {"urgency": QUESTIONS["urgency"]})["urgency"].available is False


def test_noul_confidence_comes_from_distance_from_a_coin_flip():
    """A noul carries no confidence field, and 0.03 is a confident no."""
    def answer_for(value: float):
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"answers": {
                "is_question": {"type": "noul", "noul": value}}})
        return jev(handler).decide("s", {"is_question": QUESTIONS["is_question"]})["is_question"]

    assert answer_for(0.03).confidence == pytest.approx(0.94)   # confidently no
    assert answer_for(0.97).confidence == pytest.approx(0.94)   # confidently yes
    assert answer_for(0.50).confidence == pytest.approx(0.0)    # a shrug


def test_the_served_model_is_recorded_because_jev_latest_is_an_alias():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"model": "jev-1.13.0", "answers": {}})

    decider = jev(handler)
    assert decider.served_model is None
    decider.decide("s", QUESTIONS)
    assert decider.served_model == "jev-1.13.0"


# ---------------------------------------------------------------- merge gate
def test_each_provider_carries_its_own_merge_gate():
    """The gate is only meaningful relative to a provider's confidence spread.
    Measured over 56 labelled cases: gpt-5-mini's wrong answers score as high
    as its right ones, so its gate is 0.95; Jev's separate, so it can sit lower
    and still merge nothing it should not."""
    mini = FakeLLM(); mini.model = "gpt-5-mini"
    assert LLMDecider(mini).auto_confirm_confidence == 0.95
    assert JevDecider(DecisionConfig(provider="jev", api_key="k")).auto_confirm_confidence == 0.7


def test_a_text_model_nobody_measured_never_merges_on_its_own():
    """gpt-5.6-luna got more verdicts right than gpt-5-mini and put its worst
    wrong "same" at 0.98, above any gate. So there is no safe number for a model
    that has not been run through evals/identity_benchmark.py."""
    luna = FakeLLM(); luna.model = "gpt-5.6-luna"
    assert LLMDecider(luna).auto_confirm_confidence == NEVER_AUTO_MERGE > 1.0
    assert LLMDecider(FakeLLM()).auto_confirm_confidence == NEVER_AUTO_MERGE
    assert NoneDecider().auto_confirm_confidence == NEVER_AUTO_MERGE
    assert merge_gate_for("gpt-5-mini") == MEASURED_MERGE_GATES["gpt-5-mini"] == 0.95
    assert merge_gate_for(None) == NEVER_AUTO_MERGE


def test_the_gate_override_reaches_the_text_model_path_too():
    """MEMRY_DECISION_MERGE_CONFIDENCE used to apply to Jev only, so someone on
    the text-model path with a model of their own had no way to set the gate
    they measured."""
    from memry.intelligence.entities import _gate

    luna = FakeLLM(); luna.model = "gpt-5.6-luna"
    unset = build_decider(DecisionConfig(provider="none"), luna)
    assert _gate(unset, luna) == NEVER_AUTO_MERGE
    chosen = build_decider(DecisionConfig(provider="none", auto_confirm_confidence=0.8), luna)
    assert _gate(chosen, luna) == 0.8
    jev = build_decider(DecisionConfig(provider="jev", api_key="k"), luna)
    assert jev.auto_confirm_confidence == 0.7 and jev.fallback_gate == NEVER_AUTO_MERGE


def test_the_gate_can_be_overridden_per_deployment():
    d = JevDecider(DecisionConfig(provider="jev", api_key="k",
                                  auto_confirm_confidence=0.85))
    assert d.auto_confirm_confidence == 0.85


def test_a_decider_judgement_records_the_gate_it_should_be_measured_against():
    from memry.intelligence.entities import _gate, _judge_via_decider
    from memry.models import Entity

    class Stub(NoneDecider):
        name = "stub"
        available = True
        auto_confirm_confidence = 0.7

        def decide(self, state, questions):
            from memry.providers.decisions import Answers
            return Answers({k: Answer("same", {"same": 0.8}, 0.8, True) for k in questions})

    stub = Stub()
    assert _gate(stub) == 0.7
    mini = FakeLLM(); mini.model = "gpt-5-mini"
    assert _gate(None, mini) == 0.95                 # no provider: the text model's own gate
    assert _gate(None, FakeLLM()) == NEVER_AUTO_MERGE  # ...which an unmeasured model has none of
    unavailable = build_decider(DecisionConfig(provider="jev"), mini)   # no key
    assert not unavailable.available and _gate(unavailable, mini) == 0.95

    judged = _judge_via_decider(stub, Entity(id="e", name="Ada", user_id="ada"),
                                ["Ada works at Northwind"], "Ada lives in Amsterdam", "Ada")
    # 0.80 clears Jev's gate but would not clear the text model's 0.95
    assert judged["confidence"] == 0.8 and judged["gate"] == 0.7


# ------------------------------------------------- the other wired stages
def _stub(answers_by_key):
    class Stub(NoneDecider):
        name = "stub"
        available = True

        def decide(self, state, questions):
            from memry.providers.decisions import Answers
            self.last_state = state
            self.last_questions = questions
            return Answers({k: answers_by_key(k, questions[k]) for k in questions})
    return Stub()


def test_entity_typing_asks_one_question_per_name_in_a_single_call():
    """The names must travel in the questions. An early probe scored 2/16
    because they only lived in the state, and every answer came back 'person'
    at a confident-looking 0.75."""
    from memry.intelligence.entities import classify_entity_types

    want = {"n0": "person", "n1": "organization", "n2": "place"}
    stub = _stub(lambda k, q: Answer(want[k], {want[k]: 0.97}, 0.97, True))
    out = classify_entity_types(NoneLLM(), ["Ada Lindqvist", "Northwind", "Amsterdam"], stub)

    assert out == {"ada lindqvist": "person", "northwind": "organization",
                   "amsterdam": "place"}
    assert len(stub.last_questions) == 3          # one call, three questions
    for name in ("Ada Lindqvist", "Northwind", "Amsterdam"):
        assert any(name in q.instructions for q in stub.last_questions.values())


def test_entity_typing_falls_back_when_the_provider_abstains():
    from memry.intelligence.entities import classify_entity_types

    assert classify_entity_types(NoneLLM(), ["Ada"], NoneDecider()) == {}
    assert classify_entity_types(NoneLLM(), ["Ada"], _stub(lambda k, q: Answer())) == {}


def test_reconcile_asks_for_an_action_and_a_target():
    from memry.intelligence.reconcile import _decide_action

    stub = _stub(lambda k, q: Answer("CHANGED" if k == "action" else "1",
                                     {}, 0.93, True))
    out = _decide_action(stub, "EXISTING…\nNEW…", count=3)
    assert out["action"] == "CHANGED" and out["target"] == 1
    assert out["content"] is None       # writing a merged text is not its job
    assert set(stub.last_questions) == {"action", "target"}


def test_reconcile_with_one_candidate_skips_the_target_question():
    from memry.intelligence.reconcile import _decide_action

    stub = _stub(lambda k, q: Answer("SAME", {}, 0.99, True))
    out = _decide_action(stub, "state", count=1)
    assert out["action"] == "SAME" and out["target"] == 0
    assert set(stub.last_questions) == {"action"}


def test_reconcile_abstention_leaves_the_text_model_in_charge():
    from memry.intelligence.reconcile import _decide_action

    assert _decide_action(NoneDecider(), "state", count=2) is None
    assert _decide_action(_stub(lambda k, q: Answer()), "state", count=2) is None


def _updating(llm):
    """A store whose decision provider answers MORE of the first similar
    memory, holding "Ada works at Northwind"."""
    decider = _stub(lambda k, q: Answer("MORE" if k == "action" else "0", {}, 0.95, True))
    store = MemoryStore(Config(db_path=":memory:"), llm=llm, embedder=HashEmbedder(64),
                        decider=decider)
    first = store.add("Ada works at Northwind", user_id="u", infer=False).actions[0]
    return store, first.memory_id


def test_an_update_under_a_decision_provider_has_the_text_model_write_the_merge():
    """The provider chooses the answer; the merged text is the text model's,
    asked with the reconcile prompt, and is a new memory that supersedes the
    target."""
    from conftest import decision, facts_response
    from memry.intelligence.reconcile import MERGE_REQUEST, RECONCILE_SYSTEM

    llm = FakeLLM()
    store, target = _updating(llm)
    merged = "Ada works at Northwind as a data engineer since 2024"
    llm.queue(decision("MORE", target=0, content=merged), facts_response())
    result = store.add("Ada is a data engineer there since 2024", user_id="u", infer=False)

    [action] = result.actions
    assert action.event == "UPDATE" and action.memory_id != target
    assert store.get(action.memory_id).content == merged
    assert store.get(target).superseded_by == action.memory_id
    system, prompt = llm.calls[0]
    assert system == RECONCILE_SYSTEM and MERGE_REQUEST in prompt
    assert "[0] Ada works at Northwind" in prompt
    assert "Ada is a data engineer there since 2024" in prompt
    store.close()


@pytest.mark.parametrize("llm", ["none", "silent"])
def test_an_update_nobody_could_write_keeps_the_target_and_supersedes_it(llm):
    """Without a text model (or with one that writes nothing) the target is
    not overwritten with the new fact alone: it is kept, superseded by the
    new memory, and can be brought back."""
    from conftest import decision

    fake = FakeLLM([decision("ADD")]) if llm == "silent" else NoneLLM()
    store, target = _updating(fake)
    result = store.add("Ada is a data engineer there", user_id="u", infer=False)

    action = result.actions[0]
    assert action.event == "SUPERSEDE" and action.memory_id != target
    old = store.get(target)
    assert old.content == "Ada works at Northwind"
    assert old.invalid_at is not None and old.superseded_by == action.memory_id
    assert store.get(action.memory_id).content == "Ada is a data engineer there"
    assert [m.content for m in store.get_all(user_id="u")] == ["Ada is a data engineer there"]
    assert [e.event for e in store.history(target)] == ["ADD", "SUPERSEDE"]
    assert [row["memory"].id for row in store.replaced(user_id="u")] == [target]
    store.close()


def test_an_update_kept_and_superseded_is_no_contradiction_and_undo_keeps_both():
    """The newer memory of an update nobody could write adds to the old one,
    it does not contradict it: the Archive lists the old one as replaced, not
    contradicted, and its plain undo brings it back without forgetting the
    newer one."""
    from memry.store import _is_contradiction

    store, target = _updating(NoneLLM())
    newer = store.add("Ada is a data engineer there", user_id="u", infer=False).actions[0]
    assert newer.event == "SUPERSEDE"
    [event] = [e for e in store.history(target) if e.event == "SUPERSEDE"]
    assert event.kind == "update" and not _is_contradiction(event)
    [row] = store.replaced(user_id="u")
    assert (row["memory"].id, row["contradiction"]) == (target, False)
    assert store.undo_replacement(target)  # keep_new=False, as the plain undo sends
    assert store.get(target).invalid_at is None
    assert store.get(newer.memory_id).invalid_at is None
    assert sorted(m.content for m in store.get_all(user_id="u")) == [
        "Ada is a data engineer there", "Ada works at Northwind"]
    assert store.replaced(user_id="u") == []
    store.close()


def test_an_important_target_of_an_update_nobody_could_write_is_kept_and_asked_about():
    """A MORE with no merged text supersedes its target only where a
    contradiction could (``held_back``): a target rated important stays in
    use beside the new memory, which carries the conflict marker, and the
    pair waits under Upkeep."""
    from memry.intelligence.reconcile import CONFLICT_KEY

    decider = _stub(lambda k, q: Answer("MORE" if k == "action" else "0", {}, 0.95, True))
    store = MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(), embedder=HashEmbedder(64),
                        decider=decider)
    target = store.add("Ada works at Northwind", user_id="u", infer=False,
                       importance=0.9).actions[0].memory_id
    action = store.add("Ada is a data engineer there", user_id="u", infer=False).actions[0]

    assert action.event == "ADD" and action.conflicts_with == target
    old = store.get(target)
    assert old.invalid_at is None and old.superseded_by is None
    assert "rated important" in store.get(action.memory_id).metadata[CONFLICT_KEY]["held"]
    assert [e.event for e in store.history(target)] == ["ADD"]
    assert store.replaced(user_id="u") == []
    [item] = [i for i in store.upkeep_queue(user_id="u") if i["kind"] == "conflict"]
    assert item["id"] == action.memory_id and item["replaces"] == ["Ada works at Northwind"]
    assert "updates a memory" in item["detail"]
    # confirmed, the new one replaces the old as an update does: the undo keeps both
    assert store.decide_upkeep("conflict", action.memory_id, "accept", user_id="u")
    [event] = [e for e in store.history(target) if e.event == "SUPERSEDE"]
    assert event.kind == "update"
    store.close()


def test_an_update_the_text_model_chose_without_text_asks_it_for_the_merge():
    """The text model's own UPDATE with no content is completed as a decision
    provider's is: the text model is asked for the merged text, and the target
    is rewritten with it instead of being superseded."""
    from conftest import decision, facts_response
    from memry.intelligence.reconcile import MERGE_REQUEST

    llm = FakeLLM()
    store = MemoryStore(Config(db_path=":memory:"), llm=llm, embedder=HashEmbedder(64))
    target = store.add("Ada works at Northwind", user_id="u", infer=False).actions[0].memory_id
    merged = "Ada works at Northwind as a data engineer"
    llm.queue(decision("MORE", target=0, content=None),
              decision("MORE", target=0, content=merged), facts_response())
    result = store.add("Ada is a data engineer there", user_id="u", infer=False)

    [action] = result.actions
    assert action.event == "UPDATE" and store.get(action.memory_id).content == merged
    assert MERGE_REQUEST not in llm.calls[0][1] and MERGE_REQUEST in llm.calls[1][1]
    # superseded as an update: listed as replaced, no contradiction
    assert [(row["memory"].id, row["contradiction"])
            for row in store.replaced(user_id="u")] == [(target, False)]
    store.close()


def test_a_contradiction_is_listed_as_one_and_its_undo_forgets_the_newer():
    from conftest import decision, facts_response

    llm = FakeLLM()
    store = MemoryStore(Config(db_path=":memory:"), llm=llm, embedder=HashEmbedder(64))
    llm.queue(facts_response({"content": "Ada lives in Munich", "type": "semantic",
                              "importance": 0.5, "categories": [], "entities": []}))
    old = store.add("I live in Munich", user_id="u").actions[0].memory_id
    llm.queue(facts_response({"content": "Ada lives in Amsterdam", "type": "semantic",
                              "importance": 0.5, "categories": [], "entities": []}),
              decision("WRONG", target=0, reason="it was never Munich"))
    new = store.add("I moved to Amsterdam", user_id="u").actions[0]
    assert new.event == "DELETE"
    [row] = store.replaced(user_id="u")
    assert (row["memory"].id, row["contradiction"]) == (old, True)
    assert [e.kind for e in store.history(old) if e.event == "SUPERSEDE"] == ["contradiction"]
    assert store.undo_replacement(old)
    assert store.get(new.memory_id).invalid_at is not None
    store.close()


def test_a_supersedes_kind_is_recorded_and_read_before_its_reason(tmp_path):
    """What took a memory out of use is recorded in the event's ``kind``: a row
    of kind "update" is no contradiction whatever its reason says, and one of
    kind "contradiction" is one. A row from before the column (added at open)
    has no kind and is classified by its reason, as before."""
    import sqlite3

    from memry.backends.local import LocalBackend
    from memry.intelligence.reconcile import UPDATE_SUPERSEDE_REASON
    from memry.models import MemoryEvent
    from memry.store import _is_contradiction, _is_update_supersede

    path = tmp_path / "events.db"
    # the table as it was before the column, built as such rather than by
    # dropping the column (ALTER TABLE ... DROP COLUMN needs SQLite 3.35);
    # the open creates the rest of the schema and adds the column
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE memory_events (id TEXT PRIMARY KEY, memory_id TEXT NOT NULL, "
                   "event TEXT NOT NULL, old_content TEXT, new_content TEXT, reason TEXT, "
                   "actor TEXT NOT NULL DEFAULT 'system', created_at TEXT NOT NULL)")
        for event_id, reason in (("old-update", f"{UPDATE_SUPERSEDE_REASON}: kept and superseded."),
                                 ("old-contradiction", "moved cities"),
                                 ("old-merge", "consolidated into m2")):
            db.execute("INSERT INTO memory_events (id, memory_id, event, reason, actor, created_at) "
                       "VALUES (?, 'm', 'SUPERSEDE', ?, 'system', ?)",
                       (event_id, reason, "2023-01-01T00:00:00+00:00"))
    backend = LocalBackend(str(path))
    try:
        backend.add_event(MemoryEvent(memory_id="m", event="SUPERSEDE", id="new-update",
                                      reason="contradicted by new information", kind="update"))
        backend.add_event(MemoryEvent(memory_id="m", event="SUPERSEDE", id="new-contradiction",
                                      reason="consolidated into m3", kind="contradiction"))
        events = {e.id: e for e in backend.history("m")}
    finally:
        backend.close()
    assert {event_id: event.kind for event_id, event in events.items()} == {
        "old-update": None, "old-contradiction": None, "old-merge": None,
        "new-update": "update", "new-contradiction": "contradiction"}
    classified = {event_id: (_is_contradiction(event), _is_update_supersede(event))
                  for event_id, event in events.items()}
    assert classified == {
        "new-update": (False, True), "new-contradiction": (True, False),
        "old-update": (False, True), "old-contradiction": (True, False),
        "old-merge": (False, False)}


# ---------------------------------------------------------------- re-ranking
def _store_with(decider, **decision):
    from memry.config import Config
    cfg = Config(db_path=":memory:")
    cfg.decision = DecisionConfig(provider="jev", api_key="k", **decision)
    decider.reranks_by_default = True     # stand in for a provider that earned it
    decider.may_rerank = True
    return MemoryStore(cfg, llm=NoneLLM(), embedder=HashEmbedder(64), decider=decider)


SHED = "Where does the garden shed key hang?"


def _shed(i):
    return f"memory {i}: the garden shed key hangs by door {i}"


def _with_memories(store, n):
    """``n`` memories the question's words find, naming nothing the store
    knows, and their order as a search that is not judged returns them."""
    for i in range(n):
        store.backend.insert_memory(
            Memory(content=_shed(i), user_id="u", embedding_model=store.embedder.model_id),
            embedding=store.embedder.embed([_shed(i)])[0])
    judge, store.decider = store.decider, NoneDecider()
    ranking = _contents(store.search(SHED, user_id="u", limit=n))
    store.decider = judge
    return ranking


def _judging(score):
    """A stub that scores ``memory i`` ``score(i)``, reads the question as
    asking for one property with one answer, and counts its calls."""
    def answer(key, question):
        if key == "property":
            return Answer(1.0, {}, 0.9, True)
        if key == "several":
            return Answer(0.0, {}, 0.9, True)
        i = int(re.search(r"Memory: memory (\d+):", question.instructions).group(1))
        return Answer(score(i), {}, 0.9, True)
    stub = _stub(answer)
    stub.calls = 0
    decide = stub.decide

    def counted(state, questions):
        stub.calls += 1
        return decide(state, questions)
    stub.decide = counted
    return stub


def _contents(results):
    return [r.memory.content for r in results]


def _searched(store):
    return _contents(store.search(SHED, user_id="u", limit=10))


def test_rerank_orders_by_the_judgement_a_tie_keeping_the_ranking():
    """A search is ordered by the judgement, whether its question names
    anything or not: measured again in the wording every search now asks in
    (R-117), the judgement alone put an answer first on every question, and
    the blend with the text ranking's position no longer earned a rule of
    its own. A tie keeps the ranking's order."""
    rel = {0: 0.55, 1: 0.50, 2: 0.50, 3: 0.75}
    stub = _judging(rel.get)
    store = _store_with(stub)
    ranking = _with_memories(store, 4)
    assert _searched(store) == \
        [_shed(3), _shed(0)] + [text for text in ranking if text in (_shed(1), _shed(2))]
    assert stub.calls == 1 and sum(key.startswith("m") for key in stub.last_questions) == 4
    store.close()


def test_rerank_pushes_a_clear_non_answer_to_the_back():
    stub = _judging(lambda i: 0.9)
    store = _store_with(stub)
    ranking = _with_memories(store, 3)
    top = int(ranking[0].split(":")[0].split()[1])
    stub = _judging(lambda i: 0.02 if i == top else 0.9)
    store.decider = stub
    stub.reranks_by_default = stub.may_rerank = True
    assert _searched(store)[-1] == ranking[0]  # first in the text ranking, not an answer
    store.close()


def test_rerank_leaves_the_order_alone_when_the_provider_cannot_answer():
    for decider in (NoneDecider(), _stub(lambda k, q: Answer())):
        store = _store_with(decider)
        ranking = _with_memories(store, 3)
        assert _searched(store) == ranking
        store.close()


def test_rerank_follows_the_provider_unless_configured():
    """Re-ranking through a text model measured below not re-ranking at all, so
    it is on for the provider that earned it and off for the rest: a search
    asks the provider nothing and keeps its order."""
    from memry.config import Config
    from memry.providers.decisions import JevDecider

    assert Config().decision.rerank is None            # unset: ask the provider
    assert NoneDecider().reranks_by_default is False
    assert LLMDecider(FakeLLM()).reranks_by_default is False
    assert JevDecider(DecisionConfig(provider="jev", api_key="k")).reranks_by_default is True
    stub = _judging(lambda i: 0.01)
    store = MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(), embedder=HashEmbedder(64),
                        decider=stub)
    ranking = _with_memories(store, 3)
    assert store.relevance_mode() == "vector"
    assert _searched(store) == ranking and stub.calls == 0
    store.close()


# --------------------------------------------------- durability / decay
def test_durability_replaces_the_per_type_half_life():
    """Two semantic facts decay at the same rate today. One is a train delay
    and one is a penicillin allergy, so that rate is wrong for both."""
    from datetime import datetime, timedelta, timezone

    from memry.config import DecayConfig
    from memry.intelligence.decay import (
        DURABILITY_KEY, durability_factor, effective_importance,
    )
    from memry.models import Memory

    cfg = DecayConfig()
    now = datetime.now(timezone.utc)
    old = (now - timedelta(days=90)).isoformat()

    def mem(**meta):
        return Memory(id="m", user_id="u", content="x", importance=1.0,
                      memory_type="semantic", created_at=old, updated_at=old,
                      metadata=meta)

    fleeting = effective_importance(mem(**{DURABILITY_KEY: 0.0}), cfg, now)
    lasting = effective_importance(mem(**{DURABILITY_KEY: 2.0}), cfg, now)
    untyped = effective_importance(mem(), cfg, now)
    assert fleeting < untyped < lasting
    # an unscored memory keeps exactly the old behaviour
    assert durability_factor(mem()) is None
    assert durability_factor(mem(**{DURABILITY_KEY: "nonsense"})) is None
    # a score between levels interpolates rather than snapping
    assert 0.2 < durability_factor(mem(**{DURABILITY_KEY: 0.5})) < 1.0


def test_durability_pass_scores_and_says_what_it_did(caplog):
    import logging

    from memry.config import Config
    from memry.intelligence.decay import DURABILITY_KEY

    stub = _stub(lambda k, q: Answer(1.9, {}, 0.8, True))
    store = MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(),
                        embedder=HashEmbedder(64), decider=stub)
    store.config.decay.durability = True
    store.add("Ada is allergic to penicillin", user_id="u", infer=False)
    store.add("The train was delayed this morning", user_id="u", infer=False)

    with caplog.at_level(logging.INFO, logger="memry"):
        outcome = store.score_memory_durability(user_id="u")
    assert outcome["scored"] == 2 and outcome["provider"] == "stub"
    assert "durability: scored 2" in caplog.text
    assert all(DURABILITY_KEY in (m.metadata or {})
               for m in store.get_all(user_id="u", limit=10))
    # a second pass has nothing left to do
    assert store.score_memory_durability(user_id="u")["scored"] == 0
    store.close()


def test_durability_pass_without_a_provider_changes_nothing():
    from memry.config import Config
    from memry.intelligence.decay import DURABILITY_KEY

    store = MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(), embedder=HashEmbedder(64))
    store.config.decay.durability = True
    store.add("Ada is allergic to penicillin", user_id="u", infer=False)
    assert store.score_memory_durability(user_id="u")["scored"] == 0
    assert DURABILITY_KEY not in (store.get_all(user_id="u", limit=5)[0].metadata or {})
    store.close()


# --------------------------------------------------- consolidation gate
def test_consolidation_skips_the_expensive_call_when_the_facts_differ():
    """Most candidate groups are not the same fact, and each one currently
    costs a text-model call that also has to write the merged sentence."""
    from memry.intelligence.consolidate import judge_group
    from memry.models import Memory

    pair = [Memory(id=f"m{i}", user_id="u", content=c)
            for i, c in enumerate(["Ada lives in Amsterdam", "Ada works in Amsterdam"])]
    llm = FakeLLM()          # empty: running out would raise
    verdict = judge_group(llm, pair, _stub(lambda k, q: Answer(0.04, {}, 0.9, True)))
    assert verdict["same_fact"] is False
    assert "typed check" in verdict["reason"]
    assert llm.calls == []   # the text model was never asked


def test_consolidation_still_asks_the_text_model_to_write_the_merge():
    from memry.intelligence.consolidate import judge_group
    from memry.models import Memory

    pair = [Memory(id=f"m{i}", user_id="u", content=c)
            for i, c in enumerate(["Ada lives in Amsterdam", "Ada's home is Amsterdam"])]
    llm = FakeLLM([json.dumps({"same_fact": True, "content": "Ada lives in Amsterdam",
                               "reason": "same"})])
    verdict = judge_group(llm, pair, _stub(lambda k, q: Answer(0.9, {}, 0.9, True)))
    assert verdict["same_fact"] is True and verdict["content"] == "Ada lives in Amsterdam"
    assert len(llm.calls) == 1


def test_consolidation_abstention_leaves_the_text_model_in_charge():
    from memry.intelligence.consolidate import judge_group
    from memry.models import Memory

    pair = [Memory(id=f"m{i}", user_id="u", content="x") for i in range(2)]
    llm = FakeLLM([json.dumps({"same_fact": False, "reason": "no"})])
    assert judge_group(llm, pair, NoneDecider())["same_fact"] is False
    assert len(llm.calls) == 1


# --------------------------------------------------- tag drift
class _TagJudge(NoneDecider):
    """A calibrated judge at Jev's tag merge bar whose P(same subject) is
    ``same(state)``, recording what it is asked."""
    name, available, calibrated = "stub", True, True
    tag_merge_probability = JevDecider.tag_merge_probability

    def __init__(self, same):
        self.same, self.asked = same, []

    def decide(self, state, questions):
        from memry.providers.decisions import Answers

        self.asked.append((state, questions))
        p = self.same(state)
        return Answers({key: Answer("same" if p >= 0.5 else "different",
                                    {"same": p, "different": 1 - p}, 0.9, True)
                        for key in questions})


def _tag_store(tagged):
    store = MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(), embedder=HashEmbedder(64))
    for tag, contents in tagged.items():
        for content in contents:
            store.add(content, user_id="default", infer=False, categories=[tag])
    return store


def _weekly(store, judge):
    """The weekly pass's entity pairs, raised and compared by ``judge``: the
    tags each memory is filed under afterwards."""
    store.decider = judge
    store.resolve_entities(user_id="default")
    return {row["category"]: row["count"] for row in store.categories(user_id="default")}


def test_tags_merge_by_the_measured_question_at_its_measured_bar():
    """Two tags are an entity pair the funnel asks the tag question of: each
    tag shown with how many memories it is on and its 10 most recent
    memories, in both orders, merged from Jev's ``tag_merge_probability``,
    0.55, into the more used tag. Measured on the 379 candidate pairs of a
    real 417-tag store, labelled by hand (16 one subject, 41 borderline, 322
    two subjects), two runs: no pair of two subjects scored above 0.46, and
    from 0.55 the judge merged 7-9 of the 16 and nothing wrong; the entity
    pair question on the same pairs merged fewer and some of two subjects.
    The measurements are in the PhD repo, papers/memry-field-studies:
    findings/identity-obvious-merges.md ("Tags on the same store") with
    data/tag_pairs_jev.json, findings/identity-threshold-by-evidence.md ("Tag
    question"), notes/scenario-registry.md, T-4 and T-8, and
    data/topics-as-things/README.md. A judge with no measured bar is not asked."""
    from memry.intelligence.identity import TAG_EXAMPLES, TAG_QUESTION

    assert JevDecider.tag_merge_probability == 0.55
    assert TAG_EXAMPLES == 10
    assert TAG_QUESTION.instructions == (
        "Two tags that file memories in one person's memory store, each shown with "
        "memories filed under it. Do tag A and tag B name the same subject, so that "
        "every memory filed under one belongs under the other?")
    assert TAG_QUESTION.criteria == {
        "same": "One subject: the same tag written differently (spelling, typo, format, "
                "singular or plural, abbreviation, acronym, translation, legal form or web "
                "domain) or a synonym, and the memories under both are about that subject.",
        "different": "Two subjects: unrelated subjects, related subjects, or one tag is a "
                     "part, kind, aspect or detail of the other, as \"insurance\" and "
                     "\"insurance contract\".",
    }
    tags = {"quality assurance": [f"quality assurance note {i}" for i in range(12)],
            "qa": [f"qa note {i}" for i in range(11)]}
    for same, merged in ((0.55, True), (0.5499, False)):
        store = _tag_store(tags)
        judge = _TagJudge(lambda state: same)
        assert _weekly(store, judge) == ({"quality assurance": 23} if merged
                                         else {"quality assurance": 12, "qa": 11})
        assert [questions for _, questions in judge.asked] == [{"tag": TAG_QUESTION}] * 2
        first, second = sorted(state for state, _ in judge.asked)
        assert first.index('TAG A: "qa" (on 11 memories)') < first.index(
            'TAG B: "quality assurance" (on 12 memories)')
        assert second.index('TAG A: "quality assurance" (on 12 memories)') < second.index(
            'TAG B: "qa" (on 11 memories)')
        for state in (first, second):  # the 10 most recent of each tag's 11 and 12
            assert state.count("The 10 most recent memories filed under it:") == 2
            assert state.count("\n- qa note ") == 10
            assert state.count("\n- quality assurance note ") == 10
        store.close()
    store = _tag_store(tags)
    unmeasured = _TagJudge(lambda state: 0.99)
    unmeasured.calibrated = False  # no measured bar, as a text model: not asked
    assert _weekly(store, unmeasured) == {"quality assurance": 12, "qa": 11}
    assert unmeasured.asked == []
    store.close()


def test_tags_the_names_alone_would_join_stay_apart_when_their_memories_differ():
    """By their names "apple" and "apple inc" are one subject written with and
    without its legal form, and a judge shown the names alone joins them, as
    names-only judging put "memry" and "memory" at 0.98. The tag question
    shows the memories under each tag: where one tag files the fruit and the
    other the company, they stay two tags; where both file the company, the
    same two names merge."""
    from memry.intelligence.identity import TAG_QUESTION

    fruit = ["Picked apples at the orchard with Ada on Saturday",
             "Baked a pie with the Boskoop from the market",
             "Honeycrisp is my favourite variety to eat raw",
             "The tree in the garden gave twenty kilos this autumn",
             "Pressed cider from the windfalls",
             "Stored the harvest in the cellar in wooden crates"]
    company = ["Bought forty AAPL shares after the earnings call",
               "The new iPhone ships with a titanium frame",
               "Watched the WWDC keynote about the next macOS",
               "Cupertino headquarters is a ring-shaped campus",
               "Tim Cook announced the quarterly dividend",
               "The MacBook repair at the Genius Bar took two weeks"]

    def reads(state):
        """0.98 on the names; 0.1 when the memories are the fruit and the company."""
        two = any(m in state for m in fruit) and any(m in state for m in company)
        return 0.1 if two else 0.98

    apart = _tag_store({"apple": fruit, "apple inc": company})
    judge = _TagJudge(reads)
    assert _weekly(apart, judge) == {"apple": 6, "apple inc": 6}
    assert [questions for _, questions in judge.asked] == [{"tag": TAG_QUESTION}] * 2
    assert all(all(m in state for m in fruit + company) for state, _ in judge.asked)
    apart.close()

    one = _tag_store({"apple": company[:3], "apple inc": company[3:]})
    assert _weekly(one, _TagJudge(reads)) == {"apple": 6}
    one.close()


def test_rerank_cannot_be_forced_onto_a_provider_that_did_not_earn_it():
    """For a provider that was not measured to beat no re-ranking the
    setting is refused (R-118): no search asks it."""
    from memry.config import Config

    reversing = _judging(lambda i: i / 10.0)

    for provider, explicit in (("llm", True), ("llm", None), ("none", True)):
        cfg = Config(db_path=":memory:")
        cfg.decision = DecisionConfig(provider=provider, rerank=explicit)
        store = MemoryStore(cfg, llm=NoneLLM(), embedder=HashEmbedder(64),
                            decider=reversing)
        reversing.reranks_by_default = False
        ranking = _with_memories(store, 3)
        assert _searched(store) == ranking, (provider, explicit)
        assert reversing.calls == 0, (provider, explicit)
        store.close()


def test_rerank_can_be_turned_off_where_it_is_on():
    from memry.config import Config

    stub = _judging(lambda i: i / 10.0)
    stub.reranks_by_default = True
    stub.may_rerank = True

    cfg = Config(db_path=":memory:")
    cfg.decision = DecisionConfig(provider="jev", api_key="k", rerank=False)
    off = MemoryStore(cfg, llm=NoneLLM(), embedder=HashEmbedder(64), decider=stub)
    ranking = _with_memories(off, 3)
    assert _searched(off) == ranking and stub.calls == 0
    off.close()

    cfg2 = Config(db_path=":memory:")
    cfg2.decision = DecisionConfig(provider="jev", api_key="k")
    on = MemoryStore(cfg2, llm=NoneLLM(), embedder=HashEmbedder(64), decider=stub)
    _with_memories(on, 3)
    assert _searched(on) == [_shed(2), _shed(1), _shed(0)] and stub.calls == 1
    on.close()


def test_rerank_may_be_turned_on_for_a_text_model_measured_to_help():
    """A text model measured to help (R-118, measured again in the wording
    every search asks in: gpt-5.6-luna and gpt-5-mini) may be turned on by
    the setting; neither is on by default. A model not measured is refused."""
    from memry.config import Config

    for model in ("gpt-5.6-luna", "gpt-5-mini"):
        measured = FakeLLM(); measured.model = model
        assert LLMDecider(measured).may_rerank and not LLMDecider(measured).reranks_by_default
    unmeasured = FakeLLM(); unmeasured.model = "gpt-6-luna"
    assert not LLMDecider(unmeasured).may_rerank

    scores: dict[int, float] = {}
    reversing = _judging(scores.get)  # prefers the last of the text ranking
    reversing.may_rerank = True                     # measured to help...
    reversing.reranks_by_default = False            # ...but not on by itself

    for explicit, expect_reranked in ((None, False), (True, True), (False, False)):
        cfg = Config(db_path=":memory:")
        cfg.decision = DecisionConfig(provider="llm", rerank=explicit)
        store = MemoryStore(cfg, llm=NoneLLM(), embedder=HashEmbedder(64), decider=reversing)
        reversing.calls = 0
        ranking = _with_memories(store, 3)
        scores.update({int(text.split(":")[0].split()[1]): (k + 1) / 10
                       for k, text in enumerate(ranking)})
        assert (_searched(store) == ranking[::-1]) is expect_reranked, explicit
        assert reversing.calls == int(expect_reranked), explicit
        store.close()


def test_stats_reports_the_merge_gate_in_force():
    """Without a calibrated judge no model's answer merges two entities, a
    gate set for the text model included: the About panel and the Upkeep page
    read this to say so. A calibrated judge's own gate is in force while it
    answers."""
    from memry.config import Config

    luna = FakeLLM(); luna.model = "gpt-5.6-luna"
    quiet = MemoryStore(Config(db_path=":memory:"), llm=luna, embedder=HashEmbedder(64))
    assert quiet.stats()["merge_gate"] == NEVER_AUTO_MERGE
    quiet.close()

    cfg = Config(db_path=":memory:")
    cfg.decision.auto_confirm_confidence = 0.9
    chosen = MemoryStore(cfg, llm=luna, embedder=HashEmbedder(64))
    assert chosen.stats()["merge_gate"] == NEVER_AUTO_MERGE
    chosen.close()

    uncalibrated = MemoryStore(Config(db_path=":memory:"), llm=luna,
                               embedder=HashEmbedder(64), decider=_stub(lambda k, q: Answer()))
    uncalibrated.decider.auto_confirm_confidence = 0.7
    assert uncalibrated.stats()["merge_gate"] == NEVER_AUTO_MERGE
    uncalibrated.close()

    jev = MemoryStore(Config(db_path=":memory:"), llm=luna, embedder=HashEmbedder(64),
                      decider=_stub(lambda k, q: Answer()))
    jev.decider.calibrated = True
    jev.decider.auto_confirm_confidence = 0.7
    assert jev.stats()["merge_gate"] == 0.7      # the judge's own, while it answers
    jev.close()


# ------------------------------------------- open proposals and new evidence
class _Identity(NoneDecider):
    """Answers every identity question with one verdict at a set confidence and
    abstains on anything else, so only identity reaches the stub."""

    name = "stub"
    available = True
    auto_confirm_confidence = 0.7

    def __init__(self, confidence: float, *, verdict: str = "same",
                 rejudges: bool = True) -> None:
        self.confidence = confidence
        self.verdict = verdict
        self.rejudges_on_new_evidence = rejudges
        self.identity_calls = 0

    def decide(self, state, questions):
        from memry.providers.decisions import Answers

        if "identity" not in questions:
            return Answers({})
        self.identity_calls += 1
        return Answers({"identity": Answer(self.verdict, {self.verdict: self.confidence},
                                           self.confidence, True)})


def _jonas_store(decider, name: str = "Jonas", entity_type: str | None = None):
    """A store whose saves each mention one name, and a function that saves one.
    ``save(text, as_name, as_type)`` mentions another spelling or type."""
    from conftest import fact, facts_response

    llm = FakeLLM()
    store = MemoryStore(Config(db_path=":memory:"), llm=llm,
                        embedder=HashEmbedder(64), decider=decider)

    def save(text: str, as_name: str | None = None, as_type: str | None = None) -> None:
        mention = {"name": as_name or name, "type": as_type or entity_type}
        llm.queue(facts_response(fact(text, entities=[mention if mention["type"]
                                                      else mention["name"]])))
        if store.get_all(user_id="ada"):
            llm.queue(json.dumps({"action": "ADD", "target": None, "content": None,
                                  "reason": "new"}))
        store.add(text, user_id="ada")

    return store, save


def _open_pair(store) -> None:
    """Two "Jonas" entities of one memory each and the open pair between them.
    Without a calibrated judge a save joins a name the store has by rule, so
    it no longer raises such a pair itself."""
    from memry.models import Entity, EntityMention, Memory, MergeProposal

    backend = store.backend
    ids = []
    for text in ("Jonas cooks Thai food", "Jonas reviewed the design doc"):
        entity = backend.insert_entity(Entity(name="Jonas", normalized="jonas", user_id="ada"))
        memory = backend.insert_memory(
            Memory(content=text, user_id="ada", embedding_model=store.embedder.model_id),
            embedding=store.embedder.embed([text])[0])
        backend.add_mention(EntityMention(entity_id=entity.id, memory_id=memory.id,
                                          surface="Jonas"))
        ids.append(entity.id)
    backend.add_proposal(MergeProposal(entity_a=ids[0], entity_b=ids[1], user_id="ada",
                                       confidence=0.5, reason="not yet compared"))


@pytest.mark.parametrize("verdict", ["same", "different"])
def test_without_a_calibrated_judge_upkeep_asks_nothing_and_joins_one_name_by_rule(verdict):
    """A decider whose answers carry no computed probabilities is asked
    nothing about identity, at a save's re-check or in the weekly pass: its
    "same" could merge nothing, and a "different" at 0.96 kept a pair apart
    for good on a guess. Two entities of one name are joined by rule, as a
    save joins a name the store has; a pair of two names stays open for a
    person, with nothing written on it."""
    from memry.models import Entity, EntityMention, Memory, MergeProposal

    decider = _Identity(0.96, verdict=verdict)
    store, save = _jonas_store(decider)
    _open_pair(store)
    backend = store.backend
    jo = backend.insert_entity(Entity(name="Jo", normalized="jo", user_id="ada"))
    memory = backend.insert_memory(Memory(content="Jo called about the lease", user_id="ada"))
    backend.add_mention(EntityMention(entity_id=jo.id, memory_id=memory.id, surface="Jo"))
    jonas = next(e for e in store.entities(user_id="ada") if e.normalized == "jonas")
    other = backend.add_proposal(MergeProposal(entity_a=jonas.id, entity_b=jo.id,
                                               user_id="ada", reason="not yet compared"))

    save("Jonas booked a table for Friday")  # joins a Jonas by rule, re-checks nothing
    assert len(store.entities(user_id="ada")) == 3
    store.resolve_entities(user_id="ada")

    assert decider.identity_calls == 0
    [one] = [e for e in store.entities(user_id="ada") if e.normalized == "jonas"]
    assert backend.count_entity_memories(one.id) == 3
    [joined] = store.merge_proposals(user_id="ada", status="confirmed")
    assert joined.reason == "one name, joined by rule"
    [waiting] = store.merge_proposals(user_id="ada")
    assert (waiting.id, waiting.confidence, waiting.reason, waiting.different) == (
        other.id, 0.5, "not yet compared", None)
    store.close()


def test_a_slow_judge_leaves_open_pairs_for_the_weekly_pass():
    """A calibrated judge too slow to ask inside a save compares the name the
    save carries, and leaves the open pairs that name touches for the
    weekly pass."""
    judge = _PairJudge(lambda state: (0.6, 0.1))
    judge.rejudges_on_new_evidence = False
    store, save = _jonas_store(judge)
    _open_pair(store)
    save("Jonas booked the Thai restaurant for Friday")
    assert len(store.entities(user_id="ada")) == 2
    [proposal] = store.merge_proposals(user_id="ada")
    assert (proposal.confidence, proposal.reason, proposal.compared_step) == (
        0.5, "not yet compared", 0)
    store.close()


def test_only_jev_rechecks_on_every_save():
    """211 ms a question is cheap enough to ask on a save; 2.5 s is not."""
    assert JevDecider.rejudges_on_new_evidence is True
    assert LLMDecider.rejudges_on_new_evidence is False
    assert NoneDecider.rejudges_on_new_evidence is False


# ------------------------------------------- pairs decided by a calibrated judge
#: A belongs answer for a pair compared since the question exists: neither
#: entity is a version or a part of the other.
NEITHER = {"a_kind_of_b": 0.0, "a_part_of_b": 0.0, "b_kind_of_a": 0.0,
           "b_part_of_a": 0.0, "neither": 1.0}


class _PairJudge(NoneDecider):
    """A calibrated judge that answers the pair question from a function of the
    state it is shown, so a test can make the answer depend on the order of the
    two entities or on how many facts it sees."""

    name = "stub"
    available = True
    calibrated = True
    pair_merge_probability = 0.95
    rejudges_on_new_evidence = True

    def __init__(self, answer) -> None:
        self.answer = answer
        self.states: list[str] = []

    def decide(self, state, questions):
        from memry.providers.decisions import Answers

        if "pair" not in questions:
            return Answers({})
        self.states.append(state)
        same, different = self.answer(state)
        probabilities = {"same": same, "different": different,
                         "unsure": max(0.0, 1 - same - different)}
        return Answers({"pair": Answer(max(probabilities, key=probabilities.get),
                                       probabilities, 0.9, True)})


def _judged_store(answer, entity_type: str | None = None):
    judge = _PairJudge(answer)
    store, save = _jonas_store(judge, "Fundation GmbH", entity_type or "organization")
    return store, save, judge


def test_a_pair_merges_on_the_average_of_both_orders():
    """Asked in one order the judge said 1.0, in the other 0.85: the average,
    0.925, is under the 0.95 threshold, so the pair waits."""
    def answer(state):
        first = state.index("ENTITY A")
        return (1.0, 0.0) if "Finanzamt" in state[first:state.index("ENTITY B")] else (0.85, 0.0)

    store, save, judge = _judged_store(answer)
    save("Fundation GmbH's Finanzamt file number is 218/5713")
    save("Fundation has 150,000 euros in cash after taxes", as_name="Fundation")
    assert len(store.entities(user_id="ada")) == 2
    [proposal] = store.merge_proposals(user_id="ada")
    assert proposal.confidence == pytest.approx(0.925)
    assert len(judge.states) == 2  # one question per order
    store.close()


def test_a_known_name_is_asked_with_the_entity_first_only():
    """A mention of a name the store has is compared with the entity of that
    name in one call, the entity as A. Here the judge says "another" with the
    entity first (0.6) and "one thing" with the mention first: averaged
    (0.3), the mention would have joined. The check reads no merge bar, and
    one order did no worse on the namesakes it must keep apart (O-35)."""
    def answer(state):
        first = state.index("ENTITY A")
        entity_first = "Finanzamt" in state[first:state.index("ENTITY B")]
        return (0.3, 0.6) if entity_first else (0.9, 0.0)

    store, save, judge = _judged_store(answer)
    save("Fundation GmbH's Finanzamt file number is 218/5713")
    save("Fundation GmbH has 150,000 euros in cash after taxes")
    assert len(judge.states) == 1
    assert len(store.entities(user_id="ada")) == 2
    [proposal] = store.merge_proposals(user_id="ada")
    assert (proposal.confidence, proposal.different) == (pytest.approx(0.3), pytest.approx(0.6))
    store.close()


def test_one_order_keeps_the_answer_as_the_judge_gave_it():
    """``judge_pair_and_belongs(one_order=True)`` asks once, A first, and
    returns that answer unaveraged, the belongs answer keyed as it came."""
    from memry.intelligence.identity import Profile, judge_pair_and_belongs

    judge = _BelongsJudge(lambda state: (0.7, 0.2), _version_of(0.9))
    a = Profile("Kestrel planner", "product", ["Kestrel planner runs on Linux"])
    b = Profile("Kestrel planner v2", "product", ["Kestrel planner v2 added offline mode"])
    pair, belongs = judge_pair_and_belongs(judge, a, b, one_order=True)
    assert [_names(state) for state in judge.states] == [("Kestrel planner", "Kestrel planner v2")]
    assert pair == pytest.approx({"same": 0.7, "different": 0.2, "unsure": 0.1})
    assert (belongs["b_kind_of_a"], belongs["a_kind_of_b"]) == (0.9, 0.0)
    pair, belongs = judge_pair_and_belongs(judge, a, b)
    assert len(judge.states) == 3  # both orders
    assert (belongs["b_kind_of_a"], belongs["a_kind_of_b"]) == (pytest.approx(0.9), 0.0)


@pytest.mark.parametrize("same, different, entities, proposals", [
    (0.97, 0.0, 1, 0),   # merge
    (0.30, 0.60, 2, 1),  # "apart" on one memory waits: APART_STEP
    (0.80, 0.10, 2, 1),  # waits for evidence
])
def test_the_three_outcomes(same, different, entities, proposals):
    store, save, _ = _judged_store(lambda state: (same, different))
    save("Fundation GmbH's Finanzamt file number is 218/5713")
    save("Fundation has 150,000 euros in cash after taxes", as_name="Fundation")
    assert len(store.entities(user_id="ada")) == entities
    assert len(store.merge_proposals(user_id="ada")) == proposals
    store.close()


@pytest.mark.parametrize("same, different, entities, proposals", [
    (0.97, 0.00, 1, 0),
    (0.60, 0.20, 1, 0),  # under the merge bar, but nothing says it is another
    (0.30, 0.45, 1, 0),
    (0.30, 0.60, 2, 1),  # the judge says another: a new entity, compared again later
])
def test_a_known_name_attaches_unless_the_judge_says_it_is_another(
        same, different, entities, proposals):
    """A second mention of a name the store has joins that entity unless the
    judge says "different" at the apart bar. Held to the merge bar instead,
    most mentions of a known name became one-memory entities in a replayed
    store, and a one-memory entity never gains the evidence to be compared
    again."""
    store, save, _ = _judged_store(lambda state: (same, different))
    save("Fundation GmbH's Finanzamt file number is 218/5713")
    save("Fundation GmbH has 150,000 euros in cash after taxes")
    assert len(store.entities(user_id="ada")) == entities
    assert len(store.merge_proposals(user_id="ada")) == proposals
    store.close()


def test_one_memory_naming_an_entity_two_ways_makes_no_second_entity():
    """The memory names "Fundation" and "Fundation GmbH". "Fundation" merges
    into "Fundation GmbH" first; "Fundation GmbH" then joins it too, instead
    of becoming a second "Fundation GmbH" with nothing left to compare."""
    from conftest import fact, facts_response

    store, save, _ = _judged_store(lambda state: (0.99, 0.0))
    save("Fundation GmbH's Finanzamt file number is 218/5713")
    llm = store.llm
    llm.queue(facts_response(fact("Fundation (Fundation GmbH) has 150,000 euros in cash", entities=[
        {"name": "Fundation", "type": "organization"},
        {"name": "Fundation GmbH", "type": "organization"}])))
    llm.queue(json.dumps({"action": "ADD", "target": None, "content": None, "reason": "new"}))
    store.add("Fundation (Fundation GmbH) has 150,000 euros in cash", user_id="ada")
    assert [e.name for e in store.entities(user_id="ada")] == ["Fundation GmbH"]
    store.close()


def test_a_known_name_joins_the_likeliest_of_its_entities():
    """Two "Fundation GmbH" entities: the mention about Cologne joins the one
    whose memories are about Cologne, and no proposal is left behind."""
    def answer(state):
        return (0.9, 0.05) if state.count("Cologne") > 1 else (0.7, 0.1)

    store, save, _ = _judged_store(answer)
    _entity_with(store, "Fundation GmbH", ["Fundation GmbH paid invoice 12"])
    cologne = _entity_with(store, "Fundation GmbH", ["Fundation GmbH has an office in Cologne"])
    save("Fundation GmbH moved its Cologne office to Ehrenfeld")
    joined = [m.memory_id for m in store.backend.entity_mentions(cologne.id)]
    assert len(joined) == 2
    assert len(store.entities(user_id="ada")) == 2
    assert store.merge_proposals(user_id="ada") == []
    store.close()


def _names(state: str) -> tuple[str, str]:
    import re

    a, b = re.findall(r'ENTITY [AB]: "([^"]+)"', state)
    return a, b


def _entity_with(store, name: str, facts: list[str], entity_type: str = "organization"):
    from memry.models import Entity, EntityMention, Memory

    entity = store.backend.insert_entity(Entity(
        name=name, normalized=name.lower(), entity_type=entity_type, user_id="ada"))
    for text in facts:
        memory = store.backend.insert_memory(Memory(content=text, user_id="ada"))
        store.backend.add_mention(EntityMention(entity_id=entity.id, memory_id=memory.id,
                                                surface=name))
    return entity


def test_the_funnel_owes_a_pair_a_comparison_only_at_a_new_step():
    from memry.intelligence.identity import rounds

    assert rounds(0, 1) == [(1, 10)]               # found: compared with what there is
    assert rounds(1, 2) == []                       # nothing new at this step
    assert rounds(1, 3) == [(3, 10)]
    assert rounds(3, 9) == []
    assert rounds(3, 10) == [(10, 10)]
    assert rounds(0, 60) == [(10, 10), (50, 50)]    # 10 each first, then 50 if unsure
    assert rounds(10, 49) == []
    assert rounds(50, 5000) == []                   # after the last step, never again
    assert rounds(0, 0) == []                       # a side with no memories: nothing to show


def test_a_waiting_pair_is_compared_again_when_its_smaller_side_reaches_a_step():
    """"Fundation GmbH" has 12 memories and "Fundation" gains one per save. The
    pair is compared when found and when "Fundation" reaches 3 and 10
    memories, not on the saves in between."""
    def answer(state):
        a, b = _names(state)
        return (0.99, 0.0) if a == b else (0.8, 0.0)

    store, save, judge = _judged_store(answer)
    _entity_with(store, "Fundation GmbH", [f"Fundation GmbH invoice {i} was paid" for i in range(12)])
    asked = []
    for i in range(10):
        save(f"Fundation office note {i}", as_name="Fundation")
        asked.append(sum(1 for state in judge.states if set(_names(state)) == {
            "Fundation", "Fundation GmbH"}) // 2)
    assert asked == [1, 1, 2, 2, 2, 2, 2, 2, 2, 3]
    [proposal] = store.merge_proposals(user_id="ada")
    assert proposal.compared_step == 10
    store.close()


def test_a_pair_with_50_memories_a_side_is_compared_with_10_then_50():
    from memry.intelligence.identity import compare

    store, _, judge = _judged_store(
        lambda state: (0.99, 0.0) if state.count("\n- [") > 40 else (0.8, 0.0))
    a = _entity_with(store, "Fundation GmbH", [f"Fundation GmbH invoice {i}" for i in range(60)])
    b = _entity_with(store, "Fundation", [f"Fundation office note {i}" for i in range(55)])
    verdict = compare(judge, store.backend, a, b)
    assert (verdict.action, verdict.step) == ("merge", 50)
    assert [state.count("\n- [") for state in judge.states] == [20, 20, 100, 100]
    judge.states.clear()
    assert compare(judge, store.backend, a, b, compared=50).probabilities is None
    assert judge.states == []  # after the last step, never again
    store.close()


def test_a_pair_kept_apart_is_not_compared_again():
    """Kept apart once both sides have 10 memories, the pair is never asked
    about again."""
    def answer(state):
        a, b = _names(state)
        return (0.99, 0.0) if a == b else (0.2, 0.7)

    store, _, judge = _judged_store(answer)
    _entity_with(store, "Fundation GmbH", [f"Fundation GmbH invoice {i}" for i in range(10)])
    _entity_with(store, "Fundation Ventures", [f"Fundation Ventures deal {i}" for i in range(10)])
    assert store.resolve_entities(user_id="ada")["rejected"] == 1
    asked = len(judge.states)
    store.resolve_entities(user_id="ada")
    assert len(judge.states) == asked
    store.close()


def test_each_side_shows_its_recent_memories_and_those_closest_to_the_other_side():
    import numpy as np

    from memry.intelligence.identity import choose
    from memry.models import Memory

    pool = [Memory(id=f"m{i}", content=f"fact {i}") for i in range(30)]  # most recent first
    far, near = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    vectors = {m.id: far for m in pool}
    vectors.update({"m20": near, "m25": near, "m29": np.array([0.1, 0.9]),
                    "m28": np.array([0.0, 0.0, 1.0])})  # another model's vector is ignored
    vectors["other"] = near
    chosen = [m.id for m in choose(pool, 10, vectors, ["other"])]
    assert chosen[:3] == ["m0", "m1", "m2"]           # the most recent 3 of 10
    assert {"m20", "m25", "m29"} <= set(chosen)       # the closest to the other side
    assert len(chosen) == 10 and "m28" not in chosen
    assert [m.id for m in choose(pool, 10, {}, ["other"])] == [f"m{i}" for i in range(10)]
    assert choose(pool[:4], 10, vectors, ["other"]) == pool[:4]


def _state_side(state: str, name: str) -> list[str]:
    """The facts a pair state shows for the entity called ``name``, in order."""
    [block] = [b for b in state.split("\n\n") if b.startswith("ENTITY ") and f': "{name}"' in b]
    return [line.split("] ", 1)[1] for line in block.splitlines() if line.startswith("- [")]


def test_a_comparison_shows_each_side_from_its_200_most_recent_memories():
    """Jev reads long profiles: one contradicting fact placed last among 100
    or 300 facts still gave P(different) 0.95 and 0.86. A comparison chooses
    what it shows from each side's ``PAIR_POOL`` most recent memories: at
    the last step 50 of them, the 15 most recent (``RECENT_SHARE``) and the
    35 closest to the other side, in the pool's order. A memory older than
    the 200 is never shown, however close it is."""
    from datetime import datetime, timedelta, timezone

    from memry.intelligence.identity import PAIR_POOL, RECENT_SHARE, compare, memories_shown
    from memry.models import Entity, EntityMention, Memory

    assert (PAIR_POOL, RECENT_SHARE, memories_shown(50)) == (200, 0.3, 50)
    store, _, judge = _judged_store(lambda state: (0.8, 0.0))
    same, close, far = [0.0, 1.0, 0.0], [0.2, 1.0, 0.0], [1.0, 0.0, 0.0]
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def side(name: str, count: int, vector) -> Entity:
        entity = store.backend.insert_entity(Entity(
            name=name, normalized=name.lower(), entity_type="person", user_id="ada"))
        for i in range(count):  # the higher i, the more recent
            at = (start + timedelta(minutes=i)).isoformat(timespec="seconds")
            memory = store.backend.insert_memory(
                Memory(content=f"{name} note {i:03d}", user_id="ada", created_at=at, updated_at=at),
                embedding=vector(i))
            store.backend.add_mention(EntityMention(entity_id=entity.id, memory_id=memory.id,
                                                    surface=name))
        return entity

    # the 5 oldest of 205 are the closest to the other side, but not among its
    # 200 most recent; 40 others are close
    orrin = side("Orrin Vale", 205, lambda i: same if i < 5 else close if 20 <= i < 60 else far)
    other = side("O. Vale", 60, lambda i: same)
    verdict = compare(judge, store.backend, orrin, other)
    assert verdict.step == 50
    shown = [int(fact.rsplit(" ", 1)[1]) for fact in _state_side(judge.states[-1], "Orrin Vale")]
    assert shown == list(range(204, 189, -1)) + list(range(59, 24, -1))
    store.close()


def test_a_name_written_another_way_is_found_and_compared():
    store, save, _ = _judged_store(lambda state: (0.99, 0.0))
    save("Fundation GmbH's Finanzamt file number is 218/5713")
    save("Fundation builds an Office add-in", as_name="Fundation")
    assert [e.name for e in store.entities(user_id="ada")] == ["Fundation GmbH"]
    store.close()


def test_a_judged_pair_never_reaches_the_upkeep_queue():
    store, save, _ = _judged_store(lambda state: (0.8, 0.1))
    save("Fundation GmbH's Finanzamt file number is 218/5713")
    save("Fundation has 150,000 euros in cash after taxes", as_name="Fundation")
    assert len(store.merge_proposals(user_id="ada")) == 1
    assert [i for i in store.upkeep_queue(user_id="ada") if i["kind"] == "proposal"] == []
    assert store.upkeep_count(user_id="ada") == 0
    store.close()


def test_a_text_model_does_not_decide_pairs():
    """Its confidence is self-reported: gpt-5-mini said 0.9 on wrong answers."""
    from memry.intelligence.identity import judges_pairs
    from memry.providers.decisions import LLMDecider

    assert judges_pairs(_PairJudge(lambda s: (1.0, 0.0)))
    assert not judges_pairs(LLMDecider(FakeLLM()))
    assert not judges_pairs(None)
    assert JevDecider.calibrated and JevDecider.pair_merge_probability == 0.95


def test_the_weekly_pass_finds_and_merges_names_written_another_way():
    from memry.models import Entity, EntityMention, Memory

    store = MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(),
                        embedder=HashEmbedder(64),
                        decider=_PairJudge(lambda state: (0.99, 0.0)))
    for name, text in (("Nordlicht Robotics Oy", "Nordlicht Robotics Oy builds picking arms"),
                       ("Nordlicht Robotics", "Nordlicht Robotics quoted 38,000 euros"),
                       ("Northwind", "Northwind is a data company")):
        entity = store.backend.insert_entity(Entity(name=name, normalized=name.lower(),
                                                    entity_type="organization", user_id="ada"))
        memory = store.backend.insert_memory(Memory(content=text, user_id="ada"))
        store.backend.add_mention(EntityMention(entity_id=entity.id, memory_id=memory.id,
                                                surface=name))
    result = store.resolve_entities(user_id="ada")
    assert result["proposed"] == 1 and result["confirmed"] == 1
    names = sorted(e.name for e in store.entities(user_id="ada"))
    assert len(names) == 2 and names[0].startswith("Nordlicht") and names[1] == "Northwind"
    store.close()


def _index(*names, vectors=None):
    from memry.intelligence.identity import NameIndex
    from memry.models import Entity

    entities = [Entity(id=f"e{i}", name=n, user_id="ada") for i, n in enumerate(names)]
    return NameIndex(entities, vectors)


def test_names_worth_comparing_come_from_the_store_not_from_lists():
    fillers = [f"Firma{i} GmbH" for i in range(60)]  # "gmbh" is common in this store
    index = _index("Valmera S.à r.l.", "Kestrel UG (haftungsbeschränkt)", "Amazon Web Services",
                   "OpenAI", "Kestrel Holding GmbH", *fillers)
    found = lambda name: [e.name for e in index.candidates(name)]
    assert "Valmera S.à r.l." in found("Valmera")                        # rare shared word
    assert "Kestrel UG (haftungsbeschränkt)" in found("Kestrel GmbH")    # rare shared word
    assert "Amazon Web Services" in found("AWS")                         # initial letters
    assert "OpenAI" in found("Open AI")                                  # spelling
    assert found("Nordwind GmbH") == [] or all("Firma" not in n for n in found("Nordwind GmbH"))


def test_an_acronym_holds_the_first_letter_of_every_word():
    from memry.intelligence.identity import is_acronym_of

    for short, long in (("AWS", "Amazon Web Services"), ("KfW", "Kreditanstalt für Wiederaufbau"),
                        ("BSFZ", "Bescheinigungsstelle Forschungszulage"),
                        ("GTM", "Google Tag Manager"), ("qa", "quality assurance"),
                        ("ICAM", "Ilustre Colegio de la Abogacía de Madrid")):
        assert is_acronym_of(short, long), short
    assert not is_acronym_of("action", "ai applications")  # no second "a"
    assert not is_acronym_of("api", "ai applications")
    assert not is_acronym_of("icm", "Ilustre Colegio de la Abogacía de Madrid")  # no "A"


def _conversation(store, entity, texts, *, run_id="chat-1", hours_ago=2.0):
    """Memories saved in one conversation ``hours_ago``; the first names
    ``entity`` (when given), the others name nothing."""
    from datetime import datetime, timedelta, timezone

    from memry.models import EntityMention, Memory

    at = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat(timespec="seconds")
    saved = []
    for i, text in enumerate(texts):
        memory = store.backend.insert_memory(Memory(
            content=text, user_id="ada", run_id=run_id, created_at=at, updated_at=at))
        if i == 0 and entity is not None:
            store.backend.add_mention(EntityMention(entity_id=entity.id, memory_id=memory.id,
                                                    surface=entity.name))
        saved.append(memory)
    return saved


def _waiting_pair(store, hours_ago=2.0, others=("The user is renovating the kitchen",)):
    from memry.models import Entity, MergeProposal

    weber = _entity_with(store, "Johnny Weber",
                         [f"Johnny Weber rewired the kitchen socket {i}" for i in range(12)], "person")
    johnny = store.backend.insert_entity(Entity(
        name="Johnny", normalized="johnny", entity_type="person", user_id="ada"))
    _conversation(store, johnny, ["Johnny comes on Tuesday", *others], hours_ago=hours_ago)
    store.backend.add_proposal(MergeProposal(
        entity_a=weber.id, entity_b=johnny.id, user_id="ada", compared_step=1,
        belongs=NEITHER))
    return weber, johnny


def _kitchen(state):
    return (0.99, 0.0) if "renovating the kitchen" in state else (0.8, 0.1)


def test_a_thin_waiting_pair_is_compared_once_more_with_its_conversation():
    """"Johnny" (1 memory) waited against "Johnny Weber" at the first step. Once
    the conversation that saved it is over, the pair is compared once more with
    that conversation's other memories, and merges on them."""
    store, _, judge = _judged_store(_kitchen)
    _waiting_pair(store)
    assert store.resolve_entities(user_id="ada")["confirmed"] == 1
    [state, _] = judge.states
    assert "Other memories from the same conversations" in state
    assert "The user is renovating the kitchen" in state
    assert [e.name for e in store.entities(user_id="ada")] == ["Johnny Weber"]
    store.close()


def test_a_conversation_still_going_is_not_used_yet():
    store, _, judge = _judged_store(_kitchen)
    _waiting_pair(store, hours_ago=0.2)
    store.resolve_entities(user_id="ada")
    assert judge.states == []
    [proposal] = store.merge_proposals(user_id="ada")
    assert proposal.compared_step == 1  # still owed
    store.close()


def test_the_conversation_step_is_asked_once_and_shows_no_memory_naming_either_side():
    from memry.models import EntityMention, Scope

    store, _, judge = _judged_store(lambda state: (0.8, 0.1))
    weber, _ = _waiting_pair(store, others=("The user is renovating the kitchen",
                                            "Johnny Weber sent the invoice"))
    naming = [m for m in store.backend.list_memories(Scope(user_id="ada"), limit=100)
              if m.content == "Johnny Weber sent the invoice"][0]
    store.backend.add_mention(EntityMention(entity_id=weber.id, memory_id=naming.id,
                                            surface="Johnny Weber"))
    store.resolve_entities(user_id="ada")
    assert len(judge.states) == 2
    assert all("renovating the kitchen" in s for s in judge.states)
    assert not any(s.count("Johnny Weber sent the invoice") > 1 for s in judge.states)
    assert [p.compared_step for p in store.merge_proposals(user_id="ada")] == [2]
    store.resolve_entities(user_id="ada")
    assert len(judge.states) == 2  # not asked again
    store.close()


def test_the_conversation_step_with_nothing_to_add_asks_nothing():
    store, _, judge = _judged_store(_kitchen)
    _waiting_pair(store, others=())
    store.resolve_entities(user_id="ada")
    assert judge.states == []
    assert [p.compared_step for p in store.merge_proposals(user_id="ada")] == [2]
    store.close()


def test_a_thin_pair_whose_conversation_adds_nothing_keeps_its_belongs_answer():
    """A pair left waiting at the first step, whose conversation has nothing
    more to show, keeps the belongs answer of that comparison. It came back
    without one, so the pair was stored with none, and the weekly pass,
    which asks again a pair that has none, asked it again every week."""
    from memry.models import Entity, MergeProposal
    from memry.providers.decisions import Answer

    class PairAndBelongs(_PairJudge):
        def decide(self, state, questions):
            answers = super().decide(state, questions)
            if "belongs" in questions:
                answers.answers["belongs"] = Answer("neither", dict(NEITHER), 0.9, True)
            return answers

    store, _, _ = _judged_store(lambda state: (0.8, 0.1))
    judge = store.decider = PairAndBelongs(lambda state: (0.8, 0.1))
    weber = _entity_with(store, "Johnny Weber",
                         [f"Johnny Weber rewired the kitchen socket {i}" for i in range(12)], "person")
    johnny = store.backend.insert_entity(Entity(
        name="Johnny", normalized="johnny", entity_type="person", user_id="ada"))
    _conversation(store, johnny, ["Johnny comes on Tuesday"])
    store.backend.add_proposal(MergeProposal(entity_a=weber.id, entity_b=johnny.id,
                                             user_id="ada"))
    store.resolve_entities(user_id="ada")
    assert len(judge.states) == 2
    [proposal] = store.merge_proposals(user_id="ada")
    assert proposal.compared_step == 2 and proposal.belongs == NEITHER
    store.resolve_entities(user_id="ada")
    assert len(judge.states) == 2  # not asked again
    store.close()


def _trades(state):
    """The electrician and the plumber are two people; otherwise one."""
    if "socket" in state and "pipe" in state:
        return (0.05, 0.9)
    return (0.99, 0.0)


@pytest.mark.parametrize("count", [1, 3, 12])
def test_two_tradespeople_of_one_first_name_on_one_house_stay_two(count):
    """"Johnny" the electrician and "Johnny" the plumber both work on the
    kitchen. Entities of one name join unless the judge says different; it
    does, so they stay two at every step, and from ``APART_STEP`` on the pair
    is kept apart for good."""
    from memry.models import Scope

    store, _, judge = _judged_store(_trades)
    _entity_with(store, "Johnny", [f"Johnny rewired kitchen socket {i}" for i in range(count)],
                 "person")
    _entity_with(store, "Johnny", [f"Johnny fixed the kitchen sink pipe {i}" for i in range(count)],
                 "person")
    store.resolve_entities(user_id="ada")
    store.resolve_entities(user_id="ada")
    assert [e.name for e in store.entities(user_id="ada")] == ["Johnny", "Johnny"]
    assert judge.states  # the judge was asked
    rejected = store.backend.list_proposals(Scope(user_id="ada"), status="rejected")
    assert len(rejected) == (1 if count >= 10 else 0)
    store.close()


def test_the_namesakes_in_related_trades_load_and_are_scored(capsys):
    """``identity_v3`` holds pairs of two tradespeople of one first name on
    one house. The identity benchmark reads them with the rest, shows them
    to the judge with their dates, and scores them: here on answers written
    for the test in the shape of its cache of Jev's answers."""
    from evals import identity_resolution_benchmark as bench
    from memry.intelligence.identity import pair_state

    cases = bench.load_cases()
    by_id = {case["id"]: case for case in cases}
    trades = ["p31", "n21", "n22", "n23", "u06"]
    assert [by_id[i]["truth"] for i in trades] == [
        "same", "not-same", "not-same", "not-same", "ambiguous"]
    a, b = bench.profiles(by_id["n21"])
    assert (a.name, b.name, len(a.dates), len(b.dates)) == ("Johnny", "Johnny", 2, 2)
    assert "Johnny rewired the kitchen sockets during the renovation" in pair_state(a, b)

    def answered(truth):
        pair = {"same": {"same": 0.99, "different": 0.0, "unsure": 0.01},
                "not-same": {"same": 0.02, "different": 0.9, "unsure": 0.08},
                "ambiguous": {"same": 0.4, "different": 0.2, "unsure": 0.4}}[truth]
        return {"today": pair, "pair": pair, "pair_rev": pair, "name": {"one_name": 1.0},
                "name_kind": {"one_thing": 1.0}, "evidence": {"neutral": 1.0}}

    rows = {case["id"]: answered(case["truth"]) for case in cases}
    point = bench.operating_point([by_id[i] for i in trades], rows, "both orders", 0.95, 0.5)
    assert (point["merged"], point["apart"], point["left"]) == (
        ["p31"], ["n21", "n22", "n23"], ["u06"])
    bench.report(cases, rows)
    out = capsys.readouterr().out
    assert "all 161" in out and "v1+v2 (101)" in out and "v3 (60)" in out
    assert "merged 82 of 82 true pairs, kept apart 60 of 79, waiting 19" in out


def test_the_conversation_step_does_not_pull_a_first_name_into_two_people():
    """"Johnny comes on Tuesday", saved in a conversation about a dripping
    tap, waits against Johnny Weber the electrician and Johnny Brandt the
    plumber. Asked again with its conversation, the judge finds it the
    plumber and, just over the bar, the electrician too. Both answers were
    applied: Johnny joined one and then the other, and the two tradespeople,
    whom nothing had compared, became one entity. An answer speaks for the
    two entities it compared: the likelier merge is applied, and the other
    pair, now the electrician against the plumber, is compared on its own."""
    from memry.models import Entity, MergeProposal, Scope

    def answer(state):
        a, b = _names(state)
        if {a, b} == {"Johnny Weber", "Johnny Brandt"}:
            return (0.05, 0.9)
        if "tap is dripping" not in state:
            return (0.8, 0.1)
        return (0.99, 0.0) if "Johnny Brandt" in (a, b) else (0.96, 0.0)

    store, _, judge = _judged_store(answer)
    weber = _entity_with(store, "Johnny Weber",
                         [f"Johnny Weber rewired the kitchen socket {i}" for i in range(12)], "person")
    brandt = _entity_with(store, "Johnny Brandt",
                          [f"Johnny Brandt fixed the bathroom pipe {i}" for i in range(12)], "person")
    johnny = store.backend.insert_entity(Entity(
        name="Johnny", normalized="johnny", entity_type="person", user_id="ada"))
    _conversation(store, johnny, ["Johnny comes on Tuesday", "The bathroom tap is dripping again"])
    for person in (weber, brandt):
        store.backend.add_proposal(MergeProposal(
            entity_a=person.id, entity_b=johnny.id, user_id="ada", compared_step=1,
            belongs=NEITHER))
    store.resolve_entities(user_id="ada")
    assert store.backend.resolve_entity_id(johnny.id) == brandt.id
    assert store.backend.resolve_entity_id(weber.id) == weber.id
    store.resolve_entities(user_id="ada")
    assert sorted(e.name for e in store.entities(user_id="ada")) == ["Johnny Brandt", "Johnny Weber"]
    [apart] = store.backend.list_proposals(Scope(user_id="ada"), status="rejected")
    assert {apart.entity_a, apart.entity_b} == {weber.id, brandt.id}
    store.close()


def test_a_merge_records_the_pair_and_the_answer_that_decided_it():
    """A merge rewrote the pair it decided into the entity kept, on both
    sides, "confirmed" at whatever answer it held before: which two entities
    were found to be one, and on what, was lost. The pair keeps its two ends
    (the one merged away is a tombstone pointing at the other) with the
    judge's answer and the step it was given at."""
    from memry.models import MergeProposal

    store, _, _ = _judged_store(lambda state: (0.98, 0.01))
    kessler = _entity_with(store, "Kessler Bau GmbH", ["Kessler Bau GmbH poured the foundation"])
    short = _entity_with(store, "Kessler Bau", ["Kessler Bau sent the concrete invoice"])
    store.backend.add_proposal(MergeProposal(
        entity_a=kessler.id, entity_b=short.id, user_id="ada", reason="not yet compared"))
    assert store.resolve_entities(user_id="ada")["confirmed"] == 1
    assert store.backend.resolve_entity_id(short.id) == kessler.id
    [record] = store.merge_proposals(user_id="ada", status="confirmed")
    assert (record.entity_a, record.entity_b) == (kessler.id, short.id)
    assert (record.confidence, record.different, record.compared_step, record.reason) == (
        pytest.approx(0.98), pytest.approx(0.01), 1, "stub: same")
    assert record.decided_at
    store.close()


def test_a_merge_no_single_answer_decided_records_the_rule_or_the_person():
    """Entities of one name join unless the judge says different, a name that
    could be several people joins its clear favourite, and a person may
    confirm a pair: the pair records which, beside the answer it holds."""
    from memry.models import MergeProposal

    store, _, _ = _judged_store(lambda state: (0.6, 0.2))
    first = _entity_with(store, "Mira Holt", ["Mira Holt tiled the bathroom"], "person")
    second = _entity_with(store, "Mira Holt", ["Mira Holt quoted the hallway tiles"], "person")
    store.resolve_entities(user_id="ada")
    [record] = store.merge_proposals(user_id="ada", status="confirmed")
    assert {record.entity_a, record.entity_b} == {first.id, second.id}
    assert (record.confidence, record.different) == (pytest.approx(0.6), pytest.approx(0.2))
    assert record.reason == "one name, and the judge did not say different"
    store.close()

    store, _, _ = _judged_store(lambda state: (0.5, 0.2))
    sofia, people = _namesakes(store, {"Sofia Marin": 0.80, "Sofia Petrescu": 0.60})
    store.resolve_entities(user_id="ada")
    [record] = store.merge_proposals(user_id="ada", status="confirmed")
    assert {record.entity_a, record.entity_b} == {people["Sofia Marin"].id, sofia.id}
    assert record.confidence == pytest.approx(0.8)
    assert record.reason == "the likeliest of 2 entities this name may be, ahead by 0.20"
    store.close()

    store, _, _ = _judged_store(lambda state: (0.8, 0.1))
    tiler = _entity_with(store, "Mira Holt", ["Mira Holt tiled the bathroom"], "person")
    short = _entity_with(store, "Mira", ["Mira left the grout samples"], "person")
    proposal = store.backend.add_proposal(MergeProposal(
        entity_a=tiler.id, entity_b=short.id, user_id="ada", confidence=0.8, different=0.1,
        reason="stub: same", compared_step=1))
    assert store.confirm_merge(proposal.id)
    [record] = store.merge_proposals(user_id="ada", status="confirmed")
    assert (record.entity_a, record.entity_b, record.confidence) == (
        tiler.id, short.id, pytest.approx(0.8))
    assert record.reason == "confirmed by you"
    store.close()


@pytest.mark.parametrize("same, different, second, reason", [
    # a known name joins its likeliest entity unless the judge says different
    (0.6, 0.2, "Kessler Bau GmbH",
     "a name the store has: the likeliest of its entities, not said to be different"),
    # a name written another way joins on the merge bar
    (0.99, 0.0, "Kessler Bau", "stub: same"),
])
def test_a_name_joined_at_save_records_what_decided_it(same, different, second, reason):
    """The mention of a name joined at save time to an entity the store had
    keeps the rule or the answer that joined it, as a merge keeps it on its
    pair. The mention that made the entity keeps nothing."""
    judge = _PairJudge(lambda state: (same, different))
    store, save = _jonas_store(judge, "Kessler Bau GmbH", "organization")
    save("Kessler Bau GmbH poured the foundation")
    save("Kessler Bau sent the concrete invoice", as_name=second)
    [entity] = store.entities(user_id="ada")
    first, joined = store.backend.entity_mentions(entity.id)
    assert first.decided is None and joined.surface == second
    assert joined.decided == {"reason": reason, "same": pytest.approx(same),
                              "different": pytest.approx(different), "step": 1}
    store.close()


@pytest.mark.parametrize("open_pair", [False, True])
def test_a_merge_a_person_made_directly_is_recorded_as_theirs(open_pair):
    """Merged from the entity page, with no proposal between the two or with
    one still open, the pair records that a person merged them."""
    from memry.models import MergeProposal

    store, _, _ = _judged_store(lambda state: (0.8, 0.1))
    tiler = _entity_with(store, "Mira Holt", ["Mira Holt tiled the bathroom"], "person")
    short = _entity_with(store, "Mira", ["Mira left the grout samples"], "person")
    if open_pair:
        store.backend.add_proposal(MergeProposal(
            entity_a=short.id, entity_b=tiler.id, user_id="ada", confidence=0.8,
            different=0.1, reason="stub: same", compared_step=1))
    assert store.merge_entities(tiler.id, short.id)
    [record] = store.merge_proposals(user_id="ada", status="confirmed")
    assert {record.entity_a, record.entity_b} == {tiler.id, short.id}
    assert record.reason == "merged by you" and record.decided_at
    assert store.merge_proposals(user_id="ada") == []
    store.close()


def test_a_one_word_name_is_compared_with_the_few_names_that_carry_it():
    """Three names carrying "sofia" are too many for the word to count as rare
    in a small store, yet "Sofia" is most likely one of them."""
    from memry.intelligence.identity import NameIndex
    from memry.models import Entity

    def index(names):
        return NameIndex([Entity(id=n, name=n, entity_type="person", user_id=None) for n in names])

    sofias = index(["Sofia", "Sofia Marin", "Sofia Petrescu", "Carlos Ruiz", "Madrid"])
    assert {e.name for e in sofias.candidates("Sofia", exclude={"Sofia"})} >= {
        "Sofia Marin", "Sofia Petrescu"}
    assert "Sofia" in {e.name for e in sofias.candidates("Sofia Marin", exclude={"Sofia Marin"})}
    annas = index(["Anna"] + [f"Anna {s}" for s in ("Berg", "Cruz", "Dahl", "Egan", "Frey", "Gold")])
    assert annas.candidates("Anna", exclude={"Anna"}) == []  # six Annas: the name alone says nothing


def _namesakes(store, confidences, steps=None, hours_ago=2.0, different=None):
    """"Sofia" (one memory, saved ``hours_ago``) with an open pair to each of
    the people named in ``confidences`` (P(same) of each pair; P(different)
    from ``different``, else 0.1), compared at ``steps``."""
    from memry.models import Entity, MergeProposal

    sofia = store.backend.insert_entity(Entity(
        name="Sofia", normalized="sofia", entity_type="person", user_id="ada"))
    _conversation(store, sofia, ["Sofia can come by on Thursday afternoon"], hours_ago=hours_ago)
    people = {}
    for i, (name, confidence) in enumerate(confidences.items()):
        person = _entity_with(store, name, [f"{name} did job {j}" for j in range(10)], "person")
        store.backend.add_proposal(MergeProposal(
            entity_a=person.id, entity_b=sofia.id, user_id="ada", confidence=confidence,
            compared_step=(steps or [2] * len(confidences))[i],
            different=(different or {}).get(name, 0.1), belongs=NEITHER))
        people[name] = person
    return sofia, people


@pytest.mark.parametrize("confidences, steps, different, joins", [
    ({"Sofia Marin": 0.80, "Sofia Petrescu": 0.60}, None, None, "Sofia Marin"),
    ({"Sofia Marin": 0.80, "Sofia Petrescu": 0.75}, None, None, None),   # no clear lead
    ({"Sofia Marin": 0.80}, None, None, None),                           # may be a third Sofia
    ({"Sofia Marin": 0.80, "Sofia Petrescu": 0.60}, [2, 1], None, None),  # one not asked in context yet
    ({"Sofia Marin": 0.45, "Sofia Petrescu": 0.20}, None, None, None),   # likelier not her
    # the other is ruled out: one candidate left, as with one to begin with
    ({"Sofia Marin": 0.60, "Sofia Petrescu": 0.05}, None, {"Sofia Petrescu": 0.9}, None),
])
def test_a_name_that_could_be_several_people_joins_the_clear_favourite(
        confidences, steps, different, joins):
    store, _, judge = _judged_store(lambda state: (0.5, 0.2))
    # a pair still owed its conversation step waits for a conversation still going
    sofia, people = _namesakes(store, confidences, steps, hours_ago=0.2 if steps else 2.0,
                               different=different)
    store.resolve_entities(user_id="ada")
    merged_into = store.backend.resolve_entity_id(sofia.id)
    assert merged_into == (people[joins].id if joins else sofia.id)
    store.close()


def test_words_written_in_capitals():
    from memry.intelligence.identity import upper_words

    assert upper_words("PR #42") == {"pr"}
    assert upper_words("the Dutch address PR") == {"pr"}
    assert upper_words("ICAM") == {"icam"}
    assert upper_words("Fundation GmbH") == set()
    assert upper_words("A") == set()


def test_names_sharing_a_word_in_capitals_are_looked_at_before_they_are_compared():
    """"PR #42" shares only "PR" with "the Dutch address PR" and with "PR #43".
    The judge looks at the names first: "PR #43" is ruled out and recorded, so
    it is not looked at again; the other pair is compared on its memories."""
    from memry.providers.decisions import Answers

    class _NameJudge(_PairJudge):
        def __init__(self):
            super().__init__(lambda state: (0.5, 0.2))
            self.looks: list[tuple[str, str]] = []

        def decide(self, state, questions):
            if "pair" in questions:
                return super().decide(state, questions)
            out = {}
            for key, question in questions.items():
                if not key.startswith("n"):
                    continue
                self.looks.append((state, question.instructions))
                both = state + question.instructions
                different = 0.97 if "#42" in both and "#43" in both else 0.1
                out[key] = Answer("different" if different > 0.5 else "possible",
                                  {"different": different, "possible": 1 - different}, 0.9, True)
            return Answers(out)

    judge = _NameJudge()
    store, _ = _jonas_store(judge, "PR #42", "code")
    _entity_with(store, "PR #42", ["PR #42 changes the Dutch address form"], "code")
    _entity_with(store, "the Dutch address PR", ["The Dutch address PR was reviewed by María"], "code")
    _entity_with(store, "PR #43", ["PR #43 updates the footer"], "code")
    store.resolve_entities(user_id="ada")
    compared = {frozenset(_names(state)) for state in judge.states}
    assert frozenset({"PR #42", "the Dutch address PR"}) in compared
    assert frozenset({"PR #42", "PR #43"}) not in compared
    from memry.models import Scope

    [ruled_out] = store.backend.list_proposals(Scope(user_id="ada"), status="rejected")
    assert "names alone" in ruled_out.reason
    looks = len(judge.looks)
    store.resolve_entities(user_id="ada")
    assert len(judge.looks) == looks  # nothing new to look at
    store.close()


class _NameLook(_PairJudge):
    """A judge whose look at two names alone puts P(different) at
    ``different``; asked on memories, it waits."""

    def __init__(self, different: float) -> None:
        super().__init__(lambda state: (0.5, 0.2))
        self.different = different
        self.looks = 0

    def decide(self, state, questions):
        from memry.providers.decisions import Answers

        if "pair" in questions:
            return super().decide(state, questions)
        self.looks += 1
        probabilities = {"different": self.different, "possible": 1 - self.different}
        return Answers({key: Answer(max(probabilities, key=probabilities.get), probabilities,
                                    0.9, True) for key in questions})


@pytest.mark.parametrize("different, ruled_out", [(0.9, True), (0.8999, False)])
def test_a_look_at_the_names_rules_a_pair_out_from_the_skip_value(different, ruled_out):
    """``NAME_CHECK_SKIP`` is provisional: nobody has measured it. A look at
    two names that share only a word in capitals and that puts P(different)
    at 0.9 or more rules the pair out for good: it is recorded, never
    compared on its memories and never looked at again. Just under 0.9 the
    pair is compared."""
    from memry.intelligence.identity import NAME_CHECK_SKIP
    from memry.models import Scope

    assert NAME_CHECK_SKIP == 0.9
    judge = _NameLook(different)
    store, _ = _jonas_store(judge)
    _entity_with(store, "PR #57", ["PR #57 moves the invoice export to a queue"], "code")
    _entity_with(store, "the Harrow billing PR", ["The Harrow billing PR was merged on Friday"],
                 "code")
    store.resolve_entities(user_id="ada")
    compared = {frozenset(_names(state)) for state in judge.states}
    assert (frozenset({"PR #57", "the Harrow billing PR"}) in compared) is not ruled_out
    rejected = store.backend.list_proposals(Scope(user_id="ada"), status="rejected")
    assert [p.reason for p in rejected] == (
        [f"the names alone rule it out: P(different) {different:.2f}"] if ruled_out else [])
    looks, asked = judge.looks, len(judge.states)
    store.resolve_entities(user_id="ada")
    assert judge.looks == looks  # the names are not looked at again
    if ruled_out:
        assert len(judge.states) == asked  # nor compared
    store.close()


class _JevBars(_PairJudge):
    """The stub judge with Jev's merge bar per funnel step."""

    pair_merge_by_step = JevDecider.pair_merge_by_step


@pytest.mark.parametrize("same, merged", [(0.97, True), (0.965, False)])
def test_the_conversation_step_merges_on_the_bar_of_the_first_step(same, merged):
    """The conversation step (``CONTEXT_STEP``) has not been measured on its
    own, so a pair asked with its conversation merges on the bar of the first
    step, 0.97 with Jev's bars, not on the 0.96 of the step after it."""
    from memry.intelligence.identity import CONTEXT_STEP, PAIR_STEPS

    judge = _JevBars(lambda state: (same, 0.0) if "renovating the kitchen" in state else (0.8, 0.1))
    assert CONTEXT_STEP == 2 and PAIR_STEPS[:2] == (1, 3)
    assert judge.pair_merge_threshold(CONTEXT_STEP) == judge.pair_merge_threshold(1) == 0.97
    assert judge.pair_merge_threshold(3) == 0.96
    store, _ = _jonas_store(judge)
    _waiting_pair(store)
    store.resolve_entities(user_id="ada")
    assert any("Other memories from the same conversations" in s for s in judge.states)
    names = sorted(e.name for e in store.entities(user_id="ada"))
    assert names == (["Johnny Weber"] if merged else ["Johnny", "Johnny Weber"])
    store.close()


def test_names_close_in_meaning_are_compared_when_there_are_vectors():
    import numpy as np

    koeln, cologne = np.array([1.0, 0.0]), np.array([0.9, 0.436])
    index = _index("Köln", "Paris", vectors={"e0": koeln, "e1": np.array([0.0, 1.0])})
    assert [e.name for e in index.candidates("Cologne", vector=cologne)] == ["Köln"]


def test_the_pair_merge_threshold_can_be_set_per_deployment():
    jev = build_decider(DecisionConfig(provider="jev", api_key="k",
                                       pair_merge_probability=0.85), FakeLLM())
    assert jev.pair_merge_probability == 0.85
    text = build_decider(DecisionConfig(provider="llm", pair_merge_probability=0.85), FakeLLM())
    assert text.pair_merge_probability == NEVER_AUTO_MERGE  # a text model is not calibrated


def test_a_typo_that_swaps_two_letters_is_compared():
    from memry.intelligence.identity import edit_similarity

    assert edit_similarity("colonge", "cologne") > 0.8
    assert [e.name for e in _index("Cologne", "Berlin").candidates("Colonge")] == ["Cologne"]


def test_the_judge_sees_when_each_fact_was_recorded():
    """Without dates "lives in Munich" against "moved to Amsterdam last month"
    read as two people (0.54); with them as one (0.97)."""
    from memry.intelligence.identity import profile_from
    from memry.models import Entity, EntityMention, Memory

    store, save, judge = _judged_store(lambda state: (0.99, 0.0), "person")
    ada = store.backend.insert_entity(Entity(name="Ada Lindqvist", normalized="ada lindqvist",
                                             entity_type="person", user_id="ada"))
    memory = store.backend.insert_memory(Memory(content="Ada Lindqvist lives in Munich",
                                                user_id="ada", valid_from="2025-01-10T00:00:00+00:00"))
    store.backend.add_mention(EntityMention(entity_id=ada.id, memory_id=memory.id,
                                            surface="Ada Lindqvist"))
    profile = profile_from(ada, store.backend.entity_memories(ada.id))
    assert profile.sources[0].true_from == "2025-01-10"
    save("Ada Lindqvist moved to Amsterdam last month", as_name="Ada Lindqvist", as_type="person")
    munich = [line for s in judge.states for line in s.splitlines()
              if line.endswith("Ada Lindqvist lives in Munich")]
    assert munich and all("true from 2025-01-10" in line for line in munich)
    assert all("when it was recorded" in s for s in judge.states)
    store.close()


def test_the_judge_sees_where_each_fact_came_from():
    """Jev decides what a shared saved text or session means; Memry only shows
    it, numbered the same way on both sides."""
    from memry.intelligence.identity import Profile, Source, pair_state

    one = Source(recorded="2026-09-01 10:00", text="ep-x", session="run-7", client="claude",
                 context="grant application")
    other = Source(recorded="2026-09-20 18:30", text="ep-y", session="run-9")
    same_text = Source(recorded="2026-09-01 10:00", text="ep-x", session="run-7")
    state = pair_state(
        Profile("Andrei", "person", ["Andrei reviews the budget", "Andrei is on holiday"],
                sources=[one, other]),
        Profile("Andrei Dumitru", "person", ["Andrei Dumitru leads finance"],
                sources=[same_text]),
    )
    assert ('- [recorded 2026-09-01 10:00; saved text 1; session 1; client claude; '
            'context "grant application"] Andrei reviews the budget') in state
    assert "- [recorded 2026-09-20 18:30; saved text 2; session 2] Andrei is on holiday" in state
    assert ("- [recorded 2026-09-01 10:00; saved text 1; session 1] Andrei Dumitru leads finance"
            in state)
    assert "ep-x" not in state and "run-7" not in state
    assert "owner of the memory store" not in state
    owner = pair_state(Profile("the user", "person", ["User prefers tea"], owner=True),
                       Profile("Cosmin", "person", ["Cosmin drinks tea"]))
    assert "This entity is the owner of the memory store" in owner.split("ENTITY B")[0]


def test_one_name_saved_in_two_sessions_is_one_entity():
    """Looked up within the save's run, it became two entities that were never
    compared."""
    from conftest import fact, facts_response

    llm = FakeLLM()
    judge = _PairJudge(lambda state: (0.99, 0.0))
    store = MemoryStore(Config(db_path=":memory:"), llm=llm, embedder=HashEmbedder(64),
                        decider=judge)
    for text, run in (("Fundation GmbH's tax number is 218/5713", "r1"),
                      ("Fundation GmbH has 150,000 euros in cash", "r2")):
        llm.queue(facts_response(fact(text, entities=[
            {"name": "Fundation GmbH", "type": "organization"}])))
        if store.get_all(user_id="ada"):
            llm.queue(json.dumps({"action": "ADD", "target": None, "content": None,
                                  "reason": "new"}))
        store.add(text, user_id="ada", run_id=run)
    assert len(store.entities(user_id="ada")) == 1
    store.close()


def test_the_weekly_pass_compares_two_entities_that_carry_one_name():
    store = MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(), embedder=HashEmbedder(64),
                        decider=_PairJudge(lambda state: (0.99, 0.0)))
    _entity_with(store, "Fundation GmbH", ["Fundation GmbH's tax number is 218/5713"])
    _entity_with(store, "Fundation GmbH", ["Fundation GmbH has 150,000 euros in cash"])
    assert store.resolve_entities(user_id="ada")["confirmed"] == 1
    assert len(store.entities(user_id="ada")) == 1
    store.close()


def test_the_merge_bar_falls_as_the_smaller_side_gains_memories():
    """Measured: at one memory Jev's P(same) is close to exact, with more it is
    too cautious, so the P(same) a merge needs falls with evidence."""
    jev = build_decider(DecisionConfig(provider="jev", api_key="k"), FakeLLM())
    bars = [jev.pair_merge_threshold(step) for step in (1, 2, 3, 9, 10, 50, 200)]
    assert bars == sorted(bars, reverse=True) and bars[0] > bars[-1]
    assert jev.pair_merge_threshold(2) == jev.pair_merge_threshold(1)
    fixed = build_decider(DecisionConfig(provider="jev", api_key="k", pair_merge_probability=0.9),
                          FakeLLM())
    assert {fixed.pair_merge_threshold(step) for step in (1, 3, 10, 50)} == {0.9}


def test_a_comparison_merges_on_the_bar_for_its_step():
    from memry.intelligence.identity import decide_pair

    class Stepped(_PairJudge):
        pair_merge_by_step = {1: 0.96, 3: 0.85}

    judge = Stepped(lambda state: (0.9, 0.0))
    answer = {"same": 0.9, "different": 0.05, "unsure": 0.05}
    assert decide_pair(answer, judge, 1) == "wait"
    assert decide_pair(answer, judge, 3) == "merge"


def test_a_memory_naming_both_entities_is_not_evidence_they_are_one():
    """Two entities whose only memory is one note naming both were compared on
    that note twice and merged at P(same) 1.0."""
    from memry.models import Entity, EntityMention, Memory

    judge = _PairJudge(lambda state: (0.99, 0.0))
    store = MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(), embedder=HashEmbedder(64),
                        decider=judge)
    note = store.backend.insert_memory(Memory(
        content="The user met Michaela Neumann and wrote down Dr. Neumann's advice", user_id="ada"))
    for name in ("Michaela Neumann", "Dr. Neumann"):
        entity = store.backend.insert_entity(Entity(name=name, normalized=name.lower(),
                                                    entity_type="person", user_id="ada"))
        store.backend.add_mention(EntityMention(entity_id=entity.id, memory_id=note.id, surface=name))
    store.resolve_entities(user_id="ada")
    assert len(store.entities(user_id="ada")) == 2
    assert judge.states == []
    store.close()


def _one_name_pairs(store, names, pairs, compared_step=2):
    """Entities ``names`` (one memory each) and open pairs between them, as
    (index, index, P(same), P(different) or None when never stored)."""
    from memry.models import MergeProposal

    entities = [_entity_with(store, name, [f"{name} note {i}"]) for i, name in enumerate(names)]
    for first, second, same, different in pairs:
        store.backend.add_proposal(MergeProposal(
            entity_a=entities[first].id, entity_b=entities[second].id, user_id="ada",
            confidence=same, compared_step=compared_step, different=different,
            belongs=NEITHER))
    return entities


@pytest.mark.parametrize("same, different, joined", [
    (0.60, 0.20, True),    # the judge did not say "different": one entity
    (0.20, 0.70, False),   # it did: two
])
def test_entities_of_one_name_join_unless_the_judge_says_different(same, different, joined):
    """Two "FacTShirt" entities split before a known name joined its entity,
    and one memory each never reached the merge bar. Their pair never had
    P(different) stored, so it is asked once more, and it joins on the rule a
    save uses."""
    store, _, judge = _judged_store(lambda state: (same, different))
    first, second = _one_name_pairs(store, ["FacTShirt", "FacTShirt"],
                                    [(0, 1, 0.5, None)])
    store.resolve_entities(user_id="ada")
    assert judge.states  # asked again: nothing about "different" was stored
    together = store.backend.resolve_entity_id(first.id) == store.backend.resolve_entity_id(second.id)
    assert together is joined
    store.close()


def test_a_pair_of_one_name_uses_the_stored_answer():
    store, _, judge = _judged_store(lambda state: (0.5, 0.2))
    first, second = _one_name_pairs(store, ["FacTShirt", "FacTShirt"], [(0, 1, 0.6, 0.2)])
    store.resolve_entities(user_id="ada")
    assert not judge.states
    assert store.backend.resolve_entity_id(second.id) == first.id or \
        store.backend.resolve_entity_id(first.id) == second.id
    store.close()


def test_different_names_still_wait_for_the_merge_bar():
    store, _, _ = _judged_store(lambda state: (0.6, 0.2))
    camera, numbered = _one_name_pairs(store, ["camera", "Camera 92573"], [(0, 1, 0.6, 0.2)])
    store.resolve_entities(user_id="ada")
    assert store.backend.resolve_entity_id(numbered.id) == numbered.id
    assert store.backend.resolve_entity_id(camera.id) == camera.id
    store.close()


def test_two_entities_kept_apart_are_never_joined_through_a_third():
    """A and C were kept separate by a person; A~B and B~C are open. A and B
    join (the likelier pair), and B~C waits, since joining it would join A
    and C."""
    from memry.models import MergeProposal

    store, _, _ = _judged_store(lambda state: (0.5, 0.2))
    a, b, c = _one_name_pairs(store, ["FacTShirt"] * 3,
                              [(0, 1, 0.7, 0.1), (1, 2, 0.6, 0.2)])
    store.backend.add_proposal(MergeProposal(
        entity_a=a.id, entity_b=c.id, user_id="ada", confidence=0.5, status="rejected"))
    store.resolve_entities(user_id="ada")
    resolve = store.backend.resolve_entity_id
    assert resolve(a.id) == resolve(b.id)
    assert resolve(a.id) != resolve(c.id)
    store.close()


# ------------------------------------------------ one entity belongs to the other
class _BelongsJudge(_PairJudge):
    """Answers the pair question with ``pair(state)`` and the belongs question
    with ``belongs(state)``: how likely the entity shown first is a version of
    the one shown second, and the other way round."""

    def __init__(self, pair, belongs) -> None:
        super().__init__(pair)
        self.belongs = belongs
        self.asked: list[set[str]] = []

    def decide(self, state, questions):
        from memry.providers.decisions import Answers

        self.asked.append(set(questions))
        answers = super().decide(state, questions).answers
        if "belongs" in questions:
            a_of_b, b_of_a = self.belongs(state)
            probabilities = {"a_kind_of_b": a_of_b, "a_part_of_b": 0.0,
                             "b_kind_of_a": b_of_a, "b_part_of_a": 0.0,
                             "neither": 1.0 - a_of_b - b_of_a}
            answers["belongs"] = Answer(max(probabilities, key=probabilities.get),
                                        probabilities, 0.9, True)
        return Answers(answers)


def _version_of(p):
    """``p`` for "X v2" being a version of "X", 0 otherwise."""
    def answer(state):
        first, second = _names(state)
        return (p if first.endswith(" v2") else 0.0, p if second.endswith(" v2") else 0.0)
    return answer


def _version_pair(store, entity_type="product", belongs=None, step=0):
    from memry.models import MergeProposal

    thing = _entity_with(store, "Kestrel planner", ["Kestrel planner runs on Linux"], entity_type)
    version = _entity_with(store, "Kestrel planner v2", ["Kestrel planner v2 added offline mode"],
                           entity_type)
    store.backend.add_proposal(MergeProposal(
        entity_a=thing.id, entity_b=version.id, user_id="ada", confidence=0.5,
        compared_step=step, belongs=belongs))
    return thing, version


def test_the_belongs_question_is_asked_in_the_same_call_as_the_pair_question():
    judge = _BelongsJudge(lambda state: (0.5, 0.2), _version_of(0.9))
    store, _ = _jonas_store(judge, "Kestrel planner", "product")
    _version_pair(store)
    store.resolve_entities(user_id="ada")
    assert judge.asked and all(asked == {"pair", "belongs"} for asked in judge.asked
                               if "pair" in asked)
    store.close()


@pytest.mark.parametrize("entity_type, belongs, merged", [
    ("product", 0.9, False),   # a version: not merged at P(same) 0.99
    ("product", 0.7, True),    # under the bar: merged as before
    ("person", 0.9, True),     # a person is never a version or a part
])
def test_a_version_is_not_merged_into_its_thing(entity_type, belongs, merged):
    judge = _BelongsJudge(lambda state: (0.99, 0.0), _version_of(belongs))
    store, _ = _jonas_store(judge, "Kestrel planner", entity_type)
    thing, version = _version_pair(store, entity_type)
    store.resolve_entities(user_id="ada")
    resolve = store.backend.resolve_entity_id
    assert (resolve(thing.id) == resolve(version.id)) is merged
    if not merged:
        [proposal] = store.merge_proposals(user_id="ada")
        assert proposal.belongs["b_kind_of_a"] == pytest.approx(0.9)
        home = store.entity_structure(user_id="ada")[version.id]["home"]
        assert (home["id"], home["source"]) == (thing.id, "judged")
    store.close()


def test_a_pair_compared_before_the_belongs_question_is_asked_once_in_the_weekly_pass():
    judge = _BelongsJudge(lambda state: (0.5, 0.2), _version_of(0.0))
    store, _ = _jonas_store(judge, "Kestrel planner", "product")
    _version_pair(store, step=1)
    store.resolve_entities(user_id="ada")
    asked = len(judge.states)
    assert asked == 2  # both orders, at the step it had reached
    [proposal] = store.merge_proposals(user_id="ada")
    assert proposal.belongs is not None
    store.resolve_entities(user_id="ada")
    assert len(judge.states) == asked
    store.close()


def test_a_known_name_does_not_join_the_entity_it_is_a_part_of():
    """A save names "FacTShirt", and the store's "FacTShirt" is a shop the new
    mention is a product line of: the mention becomes its own entity with the
    shop as its home, instead of joining the shop."""
    def part_of(state):
        product_first = state.index("(product)") < state.index("(organization)")
        return (0.9, 0.0) if product_first else (0.0, 0.9)

    judge = _BelongsJudge(lambda state: (0.6, 0.2), part_of)
    store, save = _jonas_store(judge, "FacTShirt", "organization")
    save("FacTShirt opened its Etsy shop in 2025")
    save("FacTShirt sells a dinosaur shirt for 24 euros", as_type="product")
    shops = [e for e in store.entities(user_id="ada") if e.name == "FacTShirt"]
    assert len(shops) == 2
    [proposal] = store.merge_proposals(user_id="ada")
    assert max(proposal.belongs.values()) == pytest.approx(0.9)
    store.close()


def test_a_store_from_before_the_belongs_answer_gains_the_column(tmp_path):
    import sqlite3

    from memry.backends.local import LocalBackend
    from memry.models import MergeProposal, Scope

    path = tmp_path / "old.db"
    LocalBackend(str(path)).close()
    db = sqlite3.connect(path)
    db.execute("ALTER TABLE entity_proposals DROP COLUMN belongs")
    db.execute("INSERT INTO entity_proposals (id, entity_a, entity_b, user_id, created_at) "
               "VALUES ('old', 'a', 'b', 'ada', '2026-01-01T00:00:00+00:00')")
    db.commit()
    db.close()
    backend = LocalBackend(str(path))
    assert backend.get_proposal("old").belongs is None
    backend.add_proposal(MergeProposal(id="new", entity_a="c", entity_b="d", user_id="ada",
                                       belongs=NEITHER))
    backend.update_proposal_judgement("old", confidence=0.4, reason=None, belongs=NEITHER)
    assert {p.id: p.belongs for p in backend.list_proposals(Scope(user_id="ada"))} == {
        "old": NEITHER, "new": NEITHER}
    backend.close()
