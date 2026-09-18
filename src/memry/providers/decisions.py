"""Decision providers: typed judgements, separate from text generation.

Memry's intelligence layer asks two different kinds of question. Some need
prose written ("distil these messages into facts"); others only pick from a
set of answers that is known before the call is made ("are these two Jonases
the same person?"). Until now both went through :class:`~memry.providers.llm.LLM`
and a JSON schema, which works but pays a text model to not write text, and
returns a self-reported confidence number that nothing calibrates.

This module is the second kind. A :class:`Decider` answers typed questions -
:class:`Choice`, :class:`Score`, :class:`Noul` - over a blob of state, and
returns the selected answer plus the probability it assigned to every option.
Several questions go in one call and are answered independently.

Three implementations ship:

``NoneDecider``
    Abstains. Every answer comes back unavailable, which callers already know
    how to treat as "not sure" - the same fallback a missing LLM gets today.

``LLMDecider``
    Renders the questions as a JSON prompt for the configured ``LLM``. This is
    what runs by default, so behaviour does not change when nothing is set.

``JevDecider``
    TypeSafe's Jev, a System One model that answers typed questions directly.
    Off unless ``MEMRY_DECISION_PROVIDER=jev``.

Confidence means the same thing in all three: how concentrated the probability
mass is. Callers gate on it (see ``AUTO_CONFIRM_CONFIDENCE``), so a provider
that reports honest uncertainty is worth more here than one that is merely
right on average.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..config import DecisionConfig
from .llm import LLM

log = logging.getLogger("memry")

JEV_BASE_URL = "https://api.typesafe.ai/v1"
JEV_DEFAULT_MODEL = "jev-latest"


# -- questions ---------------------------------------------------------------
@dataclass(frozen=True)
class Choice:
    """Pick one of a fixed set of options.

    ``criteria`` maps each option to the description that decides it. The
    options are the dict keys, so the answer is always one of them.
    """

    instructions: str
    criteria: dict[str, str]


@dataclass(frozen=True)
class Score:
    """Place the state on an ordered scale. ``levels`` runs low to high."""

    instructions: str
    levels: list[str]


@dataclass(frozen=True)
class Noul:
    """A truth value: how likely the statement in ``instructions`` holds."""

    instructions: str


Question = Choice | Score | Noul


# -- answers -----------------------------------------------------------------
@dataclass
class Answer:
    """One question's result.

    ``available`` is False when the provider could not answer - no provider
    configured, a transport error, or a malformed reply. Callers must check it
    rather than trusting ``value``, which is then meaningless.
    """

    value: Any = None
    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    available: bool = False

    @property
    def choice(self) -> Any:
        return self.value


@dataclass
class Answers:
    answers: dict[str, Answer] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Answer:
        return self.answers.get(key, Answer())

    def __contains__(self, key: str) -> bool:
        return key in self.answers


def _unavailable(questions: dict[str, Question]) -> Answers:
    return Answers({key: Answer() for key in questions})


# -- providers ---------------------------------------------------------------
class Decider(ABC):
    name: str = "decider"
    available: bool = False

    @abstractmethod
    def decide(self, state: str, questions: dict[str, Question]) -> Answers:
        """Answer every question against ``state``. Never raises: a provider
        that fails returns unavailable answers so ingestion keeps working."""

    def close(self) -> None:
        return None


class NoneDecider(Decider):
    """No decision provider. Callers fall back to their own conservative path."""

    name = "none"
    available = False

    def decide(self, state: str, questions: dict[str, Question]) -> Answers:
        return _unavailable(questions)


class LLMDecider(Decider):
    """Today's behaviour: ask the configured text model, parse the JSON back.

    Kept as its own provider so the decision call sites have one interface
    whether or not a System One model is configured, and so swapping providers
    is a config change rather than a code change.
    """

    name = "llm"

    _SYSTEM = (
        "You answer typed questions about a piece of state. Reply with JSON only, "
        "in the form {\"<key>\": {\"answer\": <answer>, \"confidence\": 0..1}}, one "
        "entry per question key you were given.\n"
        "- choice: \"answer\" is exactly one of the listed options.\n"
        "- score: \"answer\" is the 0-based index of the level that fits.\n"
        "- noul: \"answer\" is a number from 0 to 1, the probability the statement holds.\n"
        "Confidence is how sure you are, and should be low when the state does not "
        "settle the question."
    )

    def __init__(self, llm: LLM) -> None:
        self.llm = llm
        self.available = llm.available

    def decide(self, state: str, questions: dict[str, Question]) -> Answers:
        if not self.available or not questions:
            return _unavailable(questions)
        try:
            raw = self.llm.complete(self._SYSTEM, self._prompt(state, questions))
        except Exception as exc:  # provider/transport failure must not break a save
            log.warning("decision provider %s failed: %s", self.name, exc)
            return _unavailable(questions)
        return self._parse(raw, questions)

    @staticmethod
    def _prompt(state: str, questions: dict[str, Question]) -> str:
        spec: dict[str, Any] = {}
        for key, q in questions.items():
            if isinstance(q, Choice):
                spec[key] = {"type": "choice", "instructions": q.instructions,
                             "options": q.criteria}
            elif isinstance(q, Score):
                spec[key] = {"type": "score", "instructions": q.instructions,
                             "levels": q.levels}
            else:
                spec[key] = {"type": "noul", "instructions": q.instructions}
        return f"STATE:\n{state}\n\nQUESTIONS:\n{json.dumps(spec, indent=2)}\n\nJSON only."

    @staticmethod
    def _parse(raw: str, questions: dict[str, Question]) -> Answers:
        from ..intelligence.extraction import parse_lenient_json

        parsed = parse_lenient_json(raw)
        if not isinstance(parsed, dict):
            return _unavailable(questions)
        out: dict[str, Answer] = {}
        for key, q in questions.items():
            item = parsed.get(key)
            if not isinstance(item, dict) or "answer" not in item:
                out[key] = Answer()
                continue
            out[key] = _coerce(q, item.get("answer"), item.get("confidence"))
        return Answers(out)


def _clamp(value: Any, fallback: float = 0.0) -> float:
    try:
        return min(max(float(value), 0.0), 1.0)
    except (TypeError, ValueError):
        return fallback


def _coerce(question: Question, answer: Any, confidence: Any) -> Answer:
    """Force a provider's answer into the question's own type, or abstain.

    A text model can return anything; this is where "typed" stops being a
    promise and becomes true for every provider.
    """
    conf = _clamp(confidence, 0.5)
    if isinstance(question, Choice):
        if not isinstance(answer, str) or answer not in question.criteria:
            return Answer()
        probs = {option: (1.0 if option == answer else 0.0) for option in question.criteria}
        return Answer(value=answer, probabilities=probs, confidence=conf, available=True)
    if isinstance(question, Score):
        try:
            index = int(answer)
        except (TypeError, ValueError):
            return Answer()
        if not 0 <= index < len(question.levels):
            return Answer()
        return Answer(value=index, confidence=conf, available=True)
    truth = _clamp(answer, -1.0)
    if truth < 0.0:
        return Answer()
    return Answer(value=truth, confidence=conf, available=True)


class JevDecider(Decider):
    """TypeSafe Jev over HTTP.

    One POST carries every question; Jev answers them in parallel and returns a
    probability per option, so the confidence Memry gates merges on comes from
    the model's own distribution instead of a number it was asked to invent.
    """

    name = "jev"

    def __init__(self, cfg: DecisionConfig) -> None:
        self.cfg = cfg
        self.model = cfg.model or JEV_DEFAULT_MODEL
        self.base_url = (cfg.base_url or JEV_BASE_URL).rstrip("/")
        self.available = bool(cfg.api_key)
        if not self.available:
            log.warning(
                "MEMRY_DECISION_PROVIDER=jev but no API key is set; decisions fall "
                "back to the conservative path. Set MEMRY_DECISION_API_KEY."
            )
        self._client = httpx.Client(
            timeout=cfg.timeout,
            headers={
                "authorization": f"Bearer {cfg.api_key or ''}",
                "content-type": "application/json",
            },
        )

    def close(self) -> None:
        self._client.close()

    def decide(self, state: str, questions: dict[str, Question]) -> Answers:
        if not self.available or not questions:
            return _unavailable(questions)
        body = {
            "model": self.model,
            "state": state,
            "questions": {k: self._question(q) for k, q in questions.items()},
        }
        try:
            resp = self._client.post(f"{self.base_url}/systemone", json=body)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            # A save must survive a decision provider being down or rate limited.
            log.warning("jev decision call failed: %s", exc)
            return _unavailable(questions)
        return self._parse(payload, questions)

    @staticmethod
    def _question(q: Question) -> dict[str, Any]:
        if isinstance(q, Choice):
            return {"type": "choice", "instructions": q.instructions, "criteria": q.criteria}
        if isinstance(q, Score):
            return {"type": "score", "instructions": q.instructions, "criteria": q.levels}
        return {"type": "noul", "instructions": q.instructions}

    @staticmethod
    def _parse(payload: Any, questions: dict[str, Question]) -> Answers:
        answers = payload.get("answers") if isinstance(payload, dict) else None
        if not isinstance(answers, dict):
            return _unavailable(questions)
        out: dict[str, Answer] = {}
        for key, q in questions.items():
            item = answers.get(key)
            if not isinstance(item, dict):
                out[key] = Answer()
                continue
            probs = item.get("probabilities")
            probs = {str(k): _clamp(v) for k, v in probs.items()} if isinstance(probs, dict) else {}
            if isinstance(q, Choice):
                value = item.get("choice")
                if not isinstance(value, str) or value not in q.criteria:
                    out[key] = Answer()
                    continue
                # Jev reports confidence; derive it from the distribution when absent.
                conf = _clamp(item.get("confidence"), probs.get(value, 0.0))
                out[key] = Answer(value, probs, conf, available=True)
            elif isinstance(q, Score):
                try:
                    index = int(item.get("score"))
                except (TypeError, ValueError):
                    out[key] = Answer()
                    continue
                if not 0 <= index < len(q.levels):
                    out[key] = Answer()
                    continue
                out[key] = Answer(index, probs, _clamp(item.get("confidence")), available=True)
            else:
                truth = _clamp(item.get("noul"), -1.0)
                if truth < 0.0:
                    out[key] = Answer()
                    continue
                out[key] = Answer(truth, probs, _clamp(item.get("confidence"), truth), True)
        return Answers(out)


def build_decider(cfg: DecisionConfig, llm: LLM) -> Decider:
    """Pick the decision provider. ``llm`` backs the default, so a deployment
    that configures nothing keeps exactly the behaviour it has today."""
    if cfg.provider == "jev":
        return JevDecider(cfg)
    if cfg.provider == "none":
        return NoneDecider()
    return LLMDecider(llm)
