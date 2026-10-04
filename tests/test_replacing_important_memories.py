"""When an important memory may be replaced without asking.

Importance says how much a fact matters, not how risky replacing it is, and a
memory superseded as an update stays as history and the Archive undoes it.
Held for importance alone, every ordinary change of state to an important
memory went to the Upkeep queue: on 4 October 2026 a real store held 7, all for
importance 0.8-0.9. Six were a listing, task or document whose state had moved
on, and one was no conflict at all. The cases below are those 7, with judge
answers shaped like the decision provider's, and the cases that must still ask:
a lasting fact, a standing rule, an identity, and anything the judge is unsure
of (``reconcile.replacement_verdict``).
"""

from __future__ import annotations

import json

import pytest
from conftest import FakeLLM, fact, facts_response

from memry.config import Config
from memry.models import Memory, Scope
from memry.providers.decisions import Answer, Answers, NoneDecider
from memry.providers.embeddings import HashEmbedder
from memry.store import MemoryStore


class _Judge(NoneDecider):
    """Answers the reconcile questions as given: the action with its
    confidence and, where asked, what the old memory is."""

    name = "stub"
    available = True
    calibrated = True
    reconcile_bars = {"SAME": 0.85, "MORE": 0.8, "CHANGED": 0.5, "WRONG": 0.5}

    def __init__(self, action: str, confidence: float, standing: dict | None) -> None:
        self.action, self.confidence, self.standing = action, confidence, standing
        self.asked_standing: list[bool] = []

    def decide(self, state, questions):
        if "action" not in questions:
            return Answers({})
        self.asked_standing.append("standing" in questions)
        rest = (1 - self.confidence) / 4
        probabilities = {a: (self.confidence if a == self.action else rest)
                         for a in ("NEW", "SAME", "MORE", "CHANGED", "WRONG")}
        answers = {"action": Answer(self.action, probabilities, self.confidence, True)}
        if "standing" in questions and self.standing:
            best = max(self.standing, key=self.standing.get)
            answers["standing"] = Answer(best, self.standing, self.standing[best], True)
        return Answers(answers)


def _store(judge) -> tuple[MemoryStore, FakeLLM]:
    llm = FakeLLM()
    return MemoryStore(Config(db_path=":memory:"), llm=llm, embedder=HashEmbedder(64),
                       decider=judge), llm


def _saved(store, llm, old: str, importance: float, new: str, user: str = "ada"):
    old_memory = store.backend.insert_memory(Memory(
        content=old, user_id=user, importance=importance,
        created_at="2026-09-22T10:00:00+00:00", updated_at="2026-09-22T10:00:00+00:00"))
    llm.queue(facts_response(fact(new)))
    store.add(new, user_id=user, created_at="2026-10-04T10:00:00+00:00")
    [new_memory] = [m for m in store.backend.list_memories(Scope(user_id=user),
                                                           include_invalid=True)
                    if m.content == new]
    return store.backend.get_memory(old_memory.id), new_memory


STATE = {"state": 0.84, "still": 0.08, "stable": 0.08}
#: The 7 questions of the live queue: new, old, importance, the judge's
#: answer, what it read the old memory as, and the outcome the owner of that
#: store approved.
QUEUE = [
    ("Cosmin löschte das Umkirch-Inserat 170540982 am 30.09.2026 selbst über das "
     "Drei-Punkte-Menü.",
     "Das aktive ImmoScout-Inserat 170540982 in Umkirch hat 8 unbeantwortete Anfragen vom "
     "11.09.2026 bis 22.09.2026; Cosmin entscheidet, ob er antwortet, das Inserat pausiert "
     "oder es löscht.", 0.9, ("CHANGED", 0.86, STATE), "update"),
    ("Seit 03.10.2026 liefert das öffentliche Exposé von ImmoScout listing 170540982, "
     "Feldbergstraße 5, Umkirch einen 404-Fehler.",
     "Das ImmoScout-Inserat 170540982 in der Feldbergstraße 5, Umkirch ist aktiv: 840 €, "
     "3 Zimmer, 82 m², im Mieter-Netzwerk.", 0.8, ("CHANGED", 0.91, STATE), "update"),
    ("Die geplante Aufgabe „Miet-Alerts für Violeta filtern“ brach am 03.10.2026 nach "
     "wenigen Sekunden ab und ist auf claude-opus-5 festgelegt.",
     "Die Aufgabe „Miet-Alerts für Violeta filtern“ läuft seit 30.09.2026 mit 15 "
     "Stadtteilen, 40-70 m², bis etwa 950 € warm und ohne WBS.", 0.9,
     ("CHANGED", 0.82, {"state": 0.78, "still": 0.12, "stable": 0.10}), "update"),
    ("Die Aufgabe „Miet-Alerts für Violeta filtern“ hat seit dem Abend des 30.09.2026 "
     "nichts mehr ins Projekt geschrieben.",
     "Die Aufgabe „Miet-Alerts für Violeta filtern“ läuft seit 30.09.2026 mit 15 "
     "Stadtteilen, 40-70 m², bis etwa 950 € warm und ohne WBS.", 0.9,
     ("CHANGED", 0.83, {"state": 0.80, "still": 0.14, "stable": 0.06}), "update"),
    ("Cosmin Novacs ImmoScout-Bewerbermappe enthielt seine SCHUFA, seinen "
     "Einkommensnachweis, Mietzahlungsnachweis und Identitätsnachweis; eine "
     "Selbstauskunft war nicht angelegt.",
     "Bei den Bewerbungen wurden nie Dateien mitgeschickt. Wenn bei ImmoScout „Profil "
     "mitgeschickt“ ausgewählt ist, wird Cosmins Profil samt Bewerbermappe übermittelt: "
     "Selbstauskunft, SCHUFA-BonitätsCheck, Einkommensnachweis, Mietzahlungsnachweis und "
     "Identitätsnachweis; alle Unterlagen stammen von Cosmin.", 0.9,
     ("WRONG", 0.88, {"state": 0.72, "still": 0.10, "stable": 0.18}), "update"),
    ("Cosmins ImmoScout-Profil zeigte unter „Eigene Dokumente: 6 Dokumente hinzugefügt“ "
     "drei geschwärzte Gehaltsabrechnungen von Violeta Steguweit, eine doppelte "
     "Meldebescheinigung und eine Bürgschaftserklärung; die Dokumente wurden am "
     "30.09.2026 hochgeladen.",
     "Violetas eigene Unterlagen und die Bürgschaftserklärung sind nicht in Cosmins "
     "ImmoScout-Profil enthalten.", 0.9,
     ("WRONG", 0.90, {"state": 0.81, "still": 0.04, "stable": 0.15}), "update"),
    ("Cosmins Versuch, den Umzugsgrund im ImmoScout-Profil zu ändern, wurde am 04.10.2026 "
     "ohne Änderung abgebrochen, weil er die erforderlichen Angaben selbst eintragen muss.",
     "Cosmin lädt die Unterlagen aus bewerbungsmappe/fuer-upload/ selbst in ImmoScout "
     "„Eigene Dokumente“ hoch und ändert selbst den Umzugsgrund im Profil.", 0.9,
     ("CHANGED", 0.62, {"state": 0.22, "still": 0.70, "stable": 0.08}), "both"),
]
STABLE = {"state": 0.05, "still": 0.07, "stable": 0.88}
#: What must still ask: a lasting fact, a standing rule and an identity read as
#: such, a state the judge is not sure enough of, and a reading under its bar.
GUARDS = [
    ("Cosmin is not allergic to penicillin.", "Cosmin is allergic to penicillin.", 0.9,
     ("WRONG", 0.72, STABLE), "ask"),
    ("For this one commit, add a Co-Authored-By trailer.",
     "Never add Co-Authored-By trailers in memry commits.", 0.9,
     ("CHANGED", 0.66, STABLE), "ask"),
    ("Cosmin was born in Cluj.", "Cosmin was born in Bucharest.", 0.9,
     ("WRONG", 0.91, STABLE), "ask"),
    ("The listing 170540982 is paused.", "The listing 170540982 is active.", 0.9,
     ("CHANGED", 0.61, STATE), "ask"),
    ("The listing 170540982 is paused.", "The listing 170540982 is active.", 0.9,
     ("CHANGED", 0.92, {"state": 0.48, "still": 0.04, "stable": 0.48}), "ask"),
]


@pytest.mark.parametrize("new, old, importance, answer, outcome", QUEUE + GUARDS,
                         ids=[f"queue-{i + 1}" for i in range(len(QUEUE))]
                         + ["penicillin", "standing-rule", "birthplace", "unsure-state",
                            "unsure-reading"])
def test_what_an_important_memory_meets(new, old, importance, answer, outcome):
    judge = _Judge(*answer)
    store, llm = _store(judge)
    old_memory, new_memory = _saved(store, llm, old, importance, new)
    queue = store._upkeep_get("conflict:pending", "ada", [])
    assert judge.asked_standing == [True]  # asked in the one call, of a protected memory
    if outcome == "update":
        assert old_memory.invalid_at is not None and old_memory.superseded_by == new_memory.id
        [event] = [e for e in store.backend.history(old_memory.id) if e.event == "SUPERSEDE"]
        assert event.kind == "update"  # kept as history, even after a WRONG
        assert queue == [] and "conflict" not in (new_memory.metadata or {})
        assert any(r["memory"].id == old_memory.id for r in store.replaced(user_id="ada"))
    elif outcome == "both":
        assert old_memory.invalid_at is None and new_memory.invalid_at is None
        assert queue == [] and "conflict" not in (new_memory.metadata or {})
    else:
        assert old_memory.invalid_at is None and new_memory.invalid_at is None
        assert [q["with"] for q in queue] == [old_memory.id]
    store.close()


def test_a_memory_nothing_protects_asks_nothing_more():
    """The question of what the memory is costs tokens: it is asked only
    where one of the memories compared is protected."""
    judge = _Judge("CHANGED", 0.7, STATE)
    store, llm = _store(judge)
    old, new = _saved(store, llm, "The listing is active.", 0.5, "The listing is paused.")
    assert judge.asked_standing == [False] and old.superseded_by == new.id
    store.close()


def test_the_text_models_answer_still_holds_an_important_memory():
    """Its answers carry no probabilities, so nothing can raise the bar."""
    llm = FakeLLM()
    store = MemoryStore(Config(db_path=":memory:"), llm=llm, embedder=HashEmbedder(64))
    store.backend.insert_memory(Memory(content="The listing 170540982 is active.",
                                       user_id="ada", importance=0.9))
    llm.queue(facts_response(fact("The listing 170540982 was deleted.")),
              json.dumps({"action": "CHANGED", "target": 0, "content": None,
                          "reason": "deleted"}))
    store.add("The listing 170540982 was deleted.", user_id="ada")
    assert len(store._upkeep_get("conflict:pending", "ada", [])) == 1
    store.close()


def test_the_queue_is_decided_again_on_a_dry_run_then_applied():
    """``memry reconcile-queue``: the judge is asked again about each queued
    question; a dry run writes nothing, and ``apply`` acts on the new rule
    with a reason, undone under Archive like any replacement."""
    judge = _Judge("CHANGED", 0.86, None)  # held under the old rule: no reading
    store, llm = _store(judge)
    old, new = _saved(store, llm, QUEUE[0][1], 0.9, QUEUE[0][0])
    judge2 = _Judge("CHANGED", 0.62, {"state": 0.2, "still": 0.7, "stable": 0.1})
    other_old, other_new = _saved(store, llm, QUEUE[6][1], 0.9, QUEUE[6][0], user="bea")
    assert [len(store._upkeep_get("conflict:pending", u, [])) for u in ("ada", "bea")] == [1, 1]

    def answering(state, questions):
        chosen = judge2 if "Umzugsgrund" in state else _Judge("CHANGED", 0.86, STATE)
        return chosen.decide(state, questions)

    store.decider.decide = answering
    rows = {row["id"]: row for u in ("ada", "bea") for row in store.redecide_conflicts(user_id=u)}
    assert (rows[new.id]["before"], rows[new.id]["now"]) == ("ask", "replace")
    assert (rows[other_new.id]["before"], rows[other_new.id]["now"]) == ("ask", "both")
    assert not any(row["applied"] for row in rows.values())
    assert [len(store._upkeep_get("conflict:pending", u, [])) for u in ("ada", "bea")] == [1, 1]
    assert store.backend.get_memory(old.id).invalid_at is None

    rows = {row["id"]: row for u in ("ada", "bea")
            for row in store.redecide_conflicts(user_id=u, apply=True)}
    assert all(row["applied"] for row in rows.values())
    assert [store._upkeep_get("conflict:pending", u, []) for u in ("ada", "bea")] == [[], []]
    replaced = store.backend.get_memory(old.id)
    assert replaced.superseded_by == new.id
    [event] = [e for e in store.backend.history(old.id) if e.event == "SUPERSEDE"]
    assert event.kind == "update" and event.actor == "system"
    assert event.reason.startswith("Memry decided again")
    assert store.backend.get_memory(other_old.id).invalid_at is None
    assert store.undo_replacement(old.id)
    assert store.backend.get_memory(old.id).invalid_at is None
    store.close()
