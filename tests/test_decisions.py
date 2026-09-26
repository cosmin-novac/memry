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


def test_a_same_below_the_gate_waits_and_merges_once_new_evidence_clears_it():
    decider = _Identity(0.5)
    store, save = _jonas_store(decider)
    save("Jonas cooks Thai food")
    save("Jonas reviewed the design doc")
    [proposal] = store.merge_proposals(user_id="ada")
    assert (proposal.confidence, proposal.reason) == (0.5, "stub: same")
    assert len(store.entities(user_id="ada")) == 2
    assert decider.identity_calls == 1, "a pair raised by this save is not asked about again"

    decider.confidence = 0.9
    save("Jonas booked the Thai restaurant for Friday")
    assert len(store.entities(user_id="ada")) == 1
    assert store.merge_proposals(user_id="ada") == []
    store.close()


def test_a_slow_provider_leaves_open_pairs_for_the_weekly_pass():
    decider = _Identity(0.5, rejudges=False)
    store, save = _jonas_store(decider)
    save("Jonas cooks Thai food")
    save("Jonas reviewed the design doc")
    decider.confidence = 0.9
    save("Jonas booked the Thai restaurant for Friday")
    assert len(store.entities(user_id="ada")) == 2
    [proposal] = store.merge_proposals(user_id="ada")
    assert proposal.confidence == 0.5
    store.close()


def test_a_pair_that_stays_open_keeps_the_latest_answer():
    decider = _Identity(0.5)
    store, save = _jonas_store(decider)
    save("Jonas cooks Thai food")
    save("Jonas reviewed the design doc")
    decider.confidence = 0.62
    store.resolve_entities(user_id="ada")
    [proposal] = store.merge_proposals(user_id="ada")
    assert (proposal.confidence, proposal.reason) == (0.62, "stub: same")
    store.close()


def test_only_jev_rechecks_on_every_save():
    """211 ms a question is cheap enough to ask on a save; 2.5 s is not."""
    assert JevDecider.rejudges_on_new_evidence is True
    assert LLMDecider.rejudges_on_new_evidence is False
    assert NoneDecider.rejudges_on_new_evidence is False


# ------------------------------------------- pairs decided by a calibrated judge
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
        entity_a=weber.id, entity_b=johnny.id, user_id="ada", compared_step=1))
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
