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
    Answer,
    Choice,
    JevDecider,
    LLMDecider,
    Noul,
    NoneDecider,
    Score,
    build_decider,
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
