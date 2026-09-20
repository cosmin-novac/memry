"""Decision providers: the typed-judgement layer and its Jev implementation.

The Jev tests drive a mock transport rather than the live API. What they pin is
the half Memry controls - the request it sends and its reading of the reply -
plus the part that matters most in production: every failure mode has to come
back as "no answer" so a save still completes.
"""

from __future__ import annotations

import json

import httpx
import pytest

from conftest import FakeLLM
from memry.config import Config, DecisionConfig
from memry.intelligence.entities import IDENTITY_QUESTION
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
    """A store that configures nothing must not acquire a decision provider."""
    assert Config().decision.provider == "none"
    assert build_decider(DecisionConfig(), NoneLLM()).available is False


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


def test_store_uses_a_configured_decider_for_entity_identity():
    """A confident "different" from the decider keeps two same-named entities
    apart, and the text model is never asked to judge identity."""
    from conftest import fact, facts_response

    class AlwaysDifferent(NoneDecider):
        name = "stub"
        available = True

        def decide(self, state, questions):
            from memry.providers.decisions import Answers

            return Answers({k: Answer("different", {"different": 0.96}, 0.96, True)
                            for k in questions})

    llm = FakeLLM()
    store = MemoryStore(Config(db_path=":memory:"), llm=llm, embedder=HashEmbedder(64),
                        decider=AlwaysDifferent())
    llm.queue(facts_response(fact("Jonas cooks Thai food", entities=["Jonas"])))
    store.add("my partner Jonas cooks Thai", user_id="ada")
    llm.queue(facts_response(fact("Jonas reviewed the design doc", entities=["Jonas"])),
              json.dumps({"action": "ADD", "target": None, "content": None, "reason": "new"}))
    store.add("a colleague named Jonas reviewed the doc", user_id="ada")

    assert len(store.entities(user_id="ada")) == 2
    assert llm.responses == []  # every queued response was consumed by extraction
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


def test_a_confident_different_still_blocks_an_obvious_merge_under_a_never_gate():
    """The same-name shortcut merges unless the model objects confidently. If
    that bar followed a never-merge gate it could never be cleared, and raising
    the gate would make merging *easier*. It is capped at 0.95."""
    from memry.intelligence.entities import _conflict_bar

    assert _conflict_bar({"gate": NEVER_AUTO_MERGE}) == 0.95
    assert _conflict_bar({"gate": 0.7}) == 0.7
    assert _conflict_bar({}) == 0.95


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

    stub = _stub(lambda k, q: Answer("UPDATE" if k == "action" else "1",
                                     {}, 0.93, True))
    out = _decide_action(stub, "EXISTING…\nNEW…", count=3)
    assert out["action"] == "UPDATE" and out["target"] == 1
    assert out["content"] is None       # writing the merged sentence is not its job
    assert set(stub.last_questions) == {"action", "target"}


def test_reconcile_with_one_candidate_skips_the_target_question():
    from memry.intelligence.reconcile import _decide_action

    stub = _stub(lambda k, q: Answer("NONE", {}, 0.99, True))
    out = _decide_action(stub, "state", count=1)
    assert out["action"] == "NONE" and out["target"] == 0
    assert set(stub.last_questions) == {"action"}


def test_reconcile_abstention_leaves_the_text_model_in_charge():
    from memry.intelligence.reconcile import _decide_action

    assert _decide_action(NoneDecider(), "state", count=2) is None
    assert _decide_action(_stub(lambda k, q: Answer()), "state", count=2) is None


# ---------------------------------------------------------------- re-ranking
def _store_with(decider, **decision):
    from memry.config import Config
    cfg = Config(db_path=":memory:")
    cfg.decision = DecisionConfig(provider="jev", api_key="k", **decision)
    decider.reranks_by_default = True     # stand in for a provider that earned it
    decider.may_rerank = True
    return MemoryStore(cfg, llm=NoneLLM(), embedder=HashEmbedder(64), decider=decider)


def test_rerank_blends_with_the_hybrid_order_rather_than_replacing_it():
    """Ordering purely by relevance measured worse than doing nothing: the
    hybrid rank carries recency, decay, anchors and relation hops with it."""
    results = [type("R", (), {"memory": type("M", (), {"content": f"memory {i}"})()})()
               for i in range(4)]
    # the model mildly prefers the last candidate; a 35% blend should not be
    # enough to drag it past the top of the hybrid order
    rel = {0: 0.55, 1: 0.50, 2: 0.50, 3: 0.75}
    stub = _stub(lambda k, q: Answer(rel[int(k[1:])], {}, 0.9, True))
    store = _store_with(stub)
    order = [r.memory.content for r in store._rerank("q", results)]
    assert len(stub.last_questions) == 4          # the judgement did run
    assert order == ["memory 0", "memory 1", "memory 2", "memory 3"]
    store.close()


def test_rerank_pushes_a_clear_non_answer_to_the_back():
    results = [type("R", (), {"memory": type("M", (), {"content": f"memory {i}"})()})()
               for i in range(3)]
    stub = _stub(lambda k, q: Answer(0.02 if k == "m0" else 0.9, {}, 0.9, True))
    store = _store_with(stub)
    order = [r.memory.content for r in store._rerank("q", results)]
    assert order[-1] == "memory 0"      # top of the hybrid order, but not an answer
    store.close()


def test_rerank_leaves_the_order_alone_when_the_provider_cannot_answer():
    results = [type("R", (), {"memory": type("M", (), {"content": f"memory {i}"})()})()
               for i in range(3)]
    for decider in (NoneDecider(), _stub(lambda k, q: Answer())):
        store = _store_with(decider)
        assert [r.memory.content for r in store._rerank("q", results)] == \
               ["memory 0", "memory 1", "memory 2"]
        store.close()


def test_rerank_follows_the_provider_unless_configured():
    """Re-ranking through a text model measured below not re-ranking at all, so
    it is on for the provider that earned it and off for the rest."""
    from memry.config import Config
    from memry.providers.decisions import JevDecider

    assert Config().decision.rerank is None            # unset: ask the provider
    assert NoneDecider().reranks_by_default is False
    assert LLMDecider(FakeLLM()).reranks_by_default is False
    assert JevDecider(DecisionConfig(provider="jev", api_key="k")).reranks_by_default is True
    store = MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(), embedder=HashEmbedder(64),
                        decider=_stub(lambda k, q: Answer(0.01, {}, 0.9, True)))
    results = [type("R", (), {"memory": type("M", (), {"content": f"memory {i}"})()})()
               for i in range(3)]
    assert store._rerank("q", results) == results
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
def test_tag_pairs_only_ever_add_suggestions():
    """On a labelled set this missed pairs a person would merge but never
    proposed an unrelated one, so it belongs in front of review, not automation."""
    from memry.intelligence.clustering import judge_tag_pairs

    pairs = [("work", "job"), ("food", "travel")]
    stub = _stub(lambda k, q: Answer(0.8 if k == "t0" else 0.05, {}, 0.9, True))
    assert judge_tag_pairs(stub, pairs) == [("work", "job")]
    assert judge_tag_pairs(NoneDecider(), pairs) == []
    assert judge_tag_pairs(stub, []) == []


def test_rerank_cannot_be_forced_onto_a_provider_that_did_not_earn_it():
    """Through gpt-5-mini the same re-ranking scored below no re-ranking at all,
    at ten seconds a query. For a provider that was not measured to beat the
    baseline the setting is refused."""
    from memry.config import Config

    results = [type("R", (), {"memory": type("M", (), {"content": f"memory {i}"})()})()
               for i in range(3)]
    reversing = _stub(lambda k, q: Answer(int(k[1:]) / 10.0, {}, 0.9, True))

    for provider, explicit in (("llm", True), ("llm", None), ("none", True)):
        cfg = Config(db_path=":memory:")
        cfg.decision = DecisionConfig(provider=provider, rerank=explicit)
        store = MemoryStore(cfg, llm=NoneLLM(), embedder=HashEmbedder(64),
                            decider=reversing)
        reversing.reranks_by_default = False
        assert store._rerank("q", results) == results, (provider, explicit)
        store.close()


def test_rerank_can_be_turned_off_where_it_is_on():
    from memry.config import Config

    results = [type("R", (), {"memory": type("M", (), {"content": f"memory {i}"})()})()
               for i in range(3)]
    stub = _stub(lambda k, q: Answer(int(k[1:]) / 10.0, {}, 0.9, True))
    stub.reranks_by_default = True
    stub.may_rerank = True

    cfg = Config(db_path=":memory:")
    cfg.decision = DecisionConfig(provider="jev", api_key="k", rerank=False)
    off = MemoryStore(cfg, llm=NoneLLM(), embedder=HashEmbedder(64), decider=stub)
    assert off._rerank("q", results) == results
    off.close()

    cfg2 = Config(db_path=":memory:")
    cfg2.decision = DecisionConfig(provider="jev", api_key="k")
    on = MemoryStore(cfg2, llm=NoneLLM(), embedder=HashEmbedder(64), decider=stub)
    assert on._rerank("q", results) != results
    on.close()


def test_rerank_may_be_turned_on_for_a_text_model_measured_to_help():
    """gpt-5.6-luna lifted recall@3 0.933 -> 0.956 and MRR 0.828 -> 0.933 at
    1.7 s a search, so the setting may turn it on; it is not on by default at
    that speed. gpt-5-mini scored below the baseline and stays refused."""
    from memry.config import Config

    luna = FakeLLM(); luna.model = "gpt-5.6-luna"
    mini = FakeLLM(); mini.model = "gpt-5-mini"
    assert LLMDecider(luna).may_rerank and not LLMDecider(luna).reranks_by_default
    assert not LLMDecider(mini).may_rerank

    results = [type("R", (), {"memory": type("M", (), {"content": f"memory {i}"})()})()
               for i in range(3)]
    reversing = _stub(lambda k, q: Answer(int(k[1:]) / 10.0, {}, 0.9, True))
    reversing.may_rerank = True                     # measured to help...
    reversing.reranks_by_default = False            # ...but not on by itself

    for explicit, expect_reranked in ((None, False), (True, True), (False, False)):
        cfg = Config(db_path=":memory:")
        cfg.decision = DecisionConfig(provider="llm", rerank=explicit)
        store = MemoryStore(cfg, llm=NoneLLM(), embedder=HashEmbedder(64), decider=reversing)
        assert (store._rerank("q", results) != results) is expect_reranked, explicit
        store.close()


def test_stats_reports_the_merge_gate_in_force():
    """Someone on an unmeasured text model will see merges stop happening on
    their own. The About panel and the Upkeep page read this to say why."""
    from memry.config import Config

    luna = FakeLLM(); luna.model = "gpt-5.6-luna"
    quiet = MemoryStore(Config(db_path=":memory:"), llm=luna, embedder=HashEmbedder(64))
    assert quiet.stats()["merge_gate"] == NEVER_AUTO_MERGE
    quiet.close()

    cfg = Config(db_path=":memory:")
    cfg.decision.auto_confirm_confidence = 0.9
    chosen = MemoryStore(cfg, llm=luna, embedder=HashEmbedder(64))
    assert chosen.stats()["merge_gate"] == 0.9
    chosen.close()

    jev = MemoryStore(Config(db_path=":memory:"), llm=luna, embedder=HashEmbedder(64),
                      decider=_stub(lambda k, q: Answer()))
    jev.decider.auto_confirm_confidence = 0.7
    assert jev.stats()["merge_gate"] == 0.7      # the provider's own, while it answers
    jev.close()
