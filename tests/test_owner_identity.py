"""Who the store owner is gets stated, never judged.

"the user" is a role. On a real store the identity judge read the owner (61
memories) and "Cosmin" (363), the person it was, as two people at P(different)
0.94-0.95, and the pair was kept apart for good. Here: extraction reports a
stated name, the owner takes it (folded into the person who carries it, or
renamed), the judge is never asked about the owner while it has no name, its
earlier answers on such pairs are opened again, one row stays per pair, and
``learn_owner`` finds the name in what a store already holds.
"""

from __future__ import annotations

import json

from conftest import FakeLLM, fact
from test_decisions import _entity_with, _names, _PairJudge

from memry.config import Config
from memry.intelligence.extraction import extract_facts
from memry.intelligence.owner import is_correction, person_for, statements_in
from memry.models import Entity, EntityMention, Memory, MergeProposal, Scope
from memry.providers.embeddings import HashEmbedder
from memry.providers.llm import NoneLLM
from memry.store import OWNER_PAIR_REOPENED, MemoryStore


def _store(answer=lambda state: (0.5, 0.1), llm=None):
    llm = llm or FakeLLM()
    judge = _PairJudge(answer)
    store = MemoryStore(Config(db_path=":memory:"), llm=llm, embedder=HashEmbedder(64),
                        decider=judge)
    return store, llm, judge


def _save(store, llm, text: str, *names: str, user_name: str | None = None) -> None:
    """Save ``text`` as one user message; extraction answers with one fact
    naming ``names`` and the ``user_name`` it reports."""
    llm.queue(json.dumps({"facts": [fact(text, entities=[
        {"name": name, "type": "person"} for name in names])], "user_name": user_name}))
    if store.get_all(user_id="ada"):
        llm.queue(json.dumps({"action": "ADD", "target": None, "content": None,
                              "reason": "new"}))
    store.add(text, user_id="ada")


def _owner(store, name: str = "the user", memories: int = 1) -> Entity:
    owner = _entity_with(store, name, [f"The user fact {i}" for i in range(memories)], "person")
    store.backend.set_entity_metadata(owner.id, {"owner": True})
    store._upkeep_set("owner_entity", "ada", owner.id)
    return store.backend.get_entity(owner.id)


# -- extraction ------------------------------------------------------------

def test_extraction_reports_a_stated_name_and_nothing_else():
    for answer, reported in (("Cos", ["Cos"]), (None, []), ("the user", []),
                             ("", []), ("The user's name is not given here at all", [])):
        llm = FakeLLM([json.dumps({"facts": [fact("The user's name is Cos.")],
                                   "user_name": answer})])
        identity: list[str] = []
        facts = extract_facts(llm, [{"role": "user", "content": "I'm Cos, hi"}],
                              identity=identity)
        assert identity == reported and [f.content for f in facts] == ["The user's name is Cos."]
    [(system, _)] = llm.calls
    assert "Never guess it" in system and '"user_name": str|null' in system


def test_an_output_without_the_field_reports_no_name():
    """An older output, or one that leaves the field out, states nothing."""
    llm = FakeLLM([json.dumps({"facts": [fact("The user sails")]})])
    identity: list[str] = []
    assert extract_facts(llm, [{"role": "user", "content": "I sail"}], identity=identity)
    assert identity == []


def test_a_save_that_states_no_name_names_nobody():
    store, llm, judge = _store()
    _save(store, llm, "The user sails", "the user")
    _entity_with(store, "Cosmin", ["Cosmin sails in Cluj"], "person")
    _save(store, llm, "The user bought a boat", "the user")
    assert store.owner_name("ada") == "the user"
    assert store._upkeep_get("owner_stated", "ada", None) is None
    store.close()


# -- a stated name names the owner -----------------------------------------

def test_a_stated_short_name_folds_the_owner_into_the_one_person_it_begins():
    """"Cos" begins "Cosmin" and no other person's name: the owner is folded
    into Cosmin, who keeps the name and becomes the owner, "the user" one of
    their names, recorded with the statement and undone like any merge."""
    store, llm, judge = _store()
    _save(store, llm, "The user sails", "the user")
    owner = store.owner_entity("ada")
    cosmin = _entity_with(store, "Cosmin", [f"Cosmin fact {i}" for i in range(5)], "person")
    _entity_with(store, "Bea", ["Bea paints"], "person")
    _save(store, llm, "The user's name is Cos.", "the user", user_name="Cos")

    person = store.owner_entity("ada")
    assert person.id == cosmin.id and person.name == "Cosmin"
    assert person.metadata["owner"] is True
    assert "the user" in store.backend.entity_aliases(cosmin.id)
    assert store.owner_name("ada") == "Cosmin"
    [merge] = store.merges(user_id="ada")
    assert merge["entity_id"] == owner.id and merge["keep_id"] == cosmin.id
    assert merge["decided"] == 'the owner\'s name was stated: "The user\'s name is Cos."'
    state = store._upkeep_get("owner_stated", "ada", None)
    assert state["name"] == "Cos" and state["outcome"]["action"] == "folded"
    assert state["evidence"][0]["memory_ids"]
    assert judge.states == []  # nobody judged anything

    # the extractor now lists the owner under its real name
    _save(store, llm, "The user bought a boat", "Cosmin")
    assert 'is the entity "Cosmin"' in [u for _, u in llm.calls
                                        if u.startswith("Conversation:")][-1]

    assert store.undo_merge(owner.id)["undone"]
    assert store.owner_entity("ada").id == owner.id
    assert not store.backend.get_entity(cosmin.id).metadata.get("owner")
    store.close()


def test_a_stated_identity_overrides_the_judges_different():
    """The judge answered a question it should not have been asked."""
    store, llm, _ = _store()
    owner = _owner(store)
    ada = _entity_with(store, "Ada Lindqvist", ["Ada Lindqvist sails"], "person")
    store.backend.add_proposal(MergeProposal(
        entity_a=owner.id, entity_b=ada.id, user_id="ada", status="rejected",
        reason="stub: different", different=0.95, decided_at="2026-09-27T19:02:24+00:00"))
    outcome = store.learn_owner_name("ada", "Ada Lindqvist", evidence={"text": "I'm Ada"})
    assert outcome["action"] == "folded" and store.owner_name("ada") == "Ada Lindqvist"
    [pair] = store.backend.list_proposals(Scope(user_id="ada"), status=None)
    assert pair.status == "confirmed" and pair.reason.startswith("the owner's name was stated")
    store.close()


def test_a_pair_a_person_kept_apart_is_not_folded_and_the_owner_takes_the_name():
    store, llm, _ = _store()
    owner = _owner(store)
    cosmin = _entity_with(store, "Cosmin", ["Cosmin sails"], "person")
    store.backend.add_proposal(MergeProposal(
        entity_a=owner.id, entity_b=cosmin.id, user_id="ada", status="rejected",
        reason="undone by you"))
    assert store.learn_owner_name("ada", "Cosmin")["action"] == "renamed"
    assert store.owner_entity("ada").id == owner.id and store.owner_name("ada") == "Cosmin"
    store.close()


def test_with_no_person_of_that_name_the_owner_is_renamed():
    store, llm, judge = _store()
    _save(store, llm, "The user sails", "the user")
    owner = store.owner_entity("ada")
    _save(store, llm, "The user's name is Ada.", "the user", user_name="Ada")
    renamed = store.owner_entity("ada")
    assert renamed.id == owner.id and renamed.name == "Ada"
    assert "the user" in store.backend.entity_aliases(owner.id)
    assert store.owner_name("ada") == "Ada"
    # named now, the owner is compared like any person
    _entity_with(store, "Ada Lindqvist", ["Ada Lindqvist sails"], "person")
    store.resolve_entities(user_id="ada")
    assert any(set(_names(state)) == {"Ada", "Ada Lindqvist"} for state in judge.states)
    store.close()


def test_a_named_turn_in_role_user_names_the_owner():
    store = MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(), embedder=HashEmbedder(64))
    store.add([{"role": "user", "name": "Ada", "content": "I sail on weekends"},
               {"role": "assistant", "content": "Nice"}], user_id="ada")
    assert store.owner_name("ada") == "Ada"
    state = store._upkeep_get("owner_stated", "ada", None)
    assert state["evidence"][0]["episode_ids"] and state["evidence"][0]["source"] == "turn"
    # speakers in other roles name nobody as the user
    store.add([{"role": "Bea", "content": "I'm Bea"}], user_id="bea")
    assert store.owner_name("bea") == "the user"
    store.close()


def test_the_accounts_name_wins():
    store, llm, _ = _store()
    store.set_owner_name("ada", "Ada Quint")
    _save(store, llm, "Ada Quint sails", "Ada Quint")
    _save(store, llm, "The user's name is Bea.", "Ada Quint", user_name="Bea")
    assert store.owner_name("ada") == "Ada Quint"
    state = store._upkeep_get("owner_stated", "ada", None)
    assert state["conflicts"][0]["name"] == "Bea" and "wins" in state["conflicts"][0]["why"]
    store.close()


def test_an_account_names_an_owner_made_before_it():
    """Made before the account named it, the owner took the account's name
    nowhere: an owner without a name is never compared, so it would have
    stayed "the user". The person who carries exactly that name is it."""
    store, llm, _ = _store()
    owner = _owner(store)
    cosmin = _entity_with(store, "Cosmin", ["Cosmin sails"], "person")
    store.set_owner_name("ada", "cosmin")
    assert store.owner_entity("ada").id == cosmin.id
    store.set_owner_name("bea", "Bea")  # no owner entity: the name waits for one
    assert store.owner_name("bea") == "Bea"
    assert store.backend.get_entity(owner.id).merged_into == cosmin.id
    store.close()


def test_the_first_name_stated_stays_unless_a_later_one_corrects_it():
    store, llm, _ = _store()
    owner = _owner(store)
    assert store.learn_owner_name("ada", "Ada", evidence={"text": "I'm Ada"})["action"] == "renamed"
    outcome = store.learn_owner_name("ada", "Bea", evidence={"text": "Bea says hi"})
    assert outcome["action"] == "conflict" and store.owner_name("ada") == "Ada"
    corrected = store.learn_owner_name(
        "ada", "Adda", evidence={"text": "My name is Adda, not Ada: it was misspelled"})
    assert corrected["action"] == "renamed" and store.owner_name("ada") == "Adda"
    state = store._upkeep_get("owner_stated", "ada", None)
    assert state["name"] == "Adda" and [c["name"] for c in state["conflicts"]] == ["Bea"]
    assert store.backend.get_entity(owner.id).name == "Adda"
    store.close()


def test_a_correction_after_a_fold_is_left_to_a_person():
    store, llm, _ = _store()
    _owner(store)
    _entity_with(store, "Cosmin", ["Cosmin sails"], "person")
    assert store.learn_owner_name("ada", "Cosmin")["action"] == "folded"
    outcome = store.learn_owner_name("ada", "Cosima",
                                     evidence={"text": "Actually I'm Cosima, not Cosmin"})
    assert outcome["action"] == "conflict" and "Merged names" in outcome["reason"]
    assert store.owner_name("ada") == "Cosmin"
    store.close()


def test_the_matching_rule():
    people = [("c", ["Cosmin", "Cosmin Novac"]), ("d", ["Dana"]), ("e", ["Dan Popescu"]),
              ("f", ["José"])]
    assert person_for("cosmin novac", people) == "c"
    assert person_for("Cos", people) == "c"           # begins one person's name
    assert person_for("Dan", people) == "e"           # a whole first name
    assert person_for("Da", people) is None           # too short to point at one
    assert person_for("Jose", people) == "f"          # accents left out
    assert person_for("Cos", people, short_names=False) is None
    assert person_for("Ana", [("a", ["Anabel"]), ("b", ["Anastasia"])]) is None  # two begin it
    assert person_for("Ana", [("a", ["Ana Maria"]), ("b", ["Anastasia"])]) == "a"
    assert statements_in("The user's name is Cos and he sails") == [("Cos", True)]
    assert statements_in("Hi Cos, welcome back", role="assistant") == [("Cos", False)]
    assert statements_in("my name is Ada", role="Bea") == []
    assert is_correction("My name is Cosima, not Cosmin", "Cosmin", "Cosima")
    assert not is_correction("My name is Cosima", "Cosmin", "Cosima")


# -- the judge is never asked about the owner without a name -----------------

def test_the_unnamed_owner_is_never_put_to_the_judge():
    store, llm, judge = _store()
    _save(store, llm, "the user sails a blue boat around Stockholm harbour", "the user")
    for name, text in (("Ada Lindqvist", "Ada Lindqvist sails a blue boat around Stockholm"),
                       ("Bob", "Bob repairs bicycles in Lyon"),
                       ("Chen", "Chen teaches chemistry in Taipei")):
        _save(store, llm, text, name)
    # a name sharing the word "user" is not compared with the owner at save;
    # "Ada" is compared with "Ada Lindqvist", the judge being there to ask
    _save(store, llm, "User Research meets on Fridays", "User Research")
    _save(store, llm, "Ada sails on Sundays", "Ada")
    owner = store.owner_entity("ada")
    # an open pair the judge answered before stays unasked
    store.backend.add_proposal(MergeProposal(
        entity_a=owner.id, entity_b=store.entities(user_id="ada")[1].id, user_id="ada",
        compared_step=1, different=0.4))
    store.resolve_entities(user_id="ada")
    store.run_structure_pass(user_id="ada")
    assert judge.states and not any("the user" in _names(state) for state in judge.states)
    assert not any(owner.id in (p.entity_a, p.entity_b)
                   for p in store.backend.list_proposals(Scope(user_id="ada"), status=None)
                   if p.compared_step == 0)
    store.close()


# -- existing stores ---------------------------------------------------------

def test_the_judges_answers_on_an_unnamed_owner_are_opened_again_once():
    store, _, _ = _store()
    owner = _owner(store, memories=3)
    cosmin = _entity_with(store, "Cosmin", ["Cosmin sails"], "person")
    bea = _entity_with(store, "Bea", ["Bea paints"], "person")
    dana = _entity_with(store, "Dana", ["Dana sings"], "person")
    acme = _entity_with(store, "Acme", ["Acme sells boats"])
    rows = {
        "judged": MergeProposal(entity_a=owner.id, entity_b=cosmin.id, user_id="ada",
                                status="rejected", reason="jev: different", different=0.94,
                                compared_step=10, decided_at="2026-09-27T19:02:24+00:00"),
        "person": MergeProposal(entity_a=owner.id, entity_b=bea.id, user_id="ada",
                                status="rejected", reason="kept apart by you"),
        "open": MergeProposal(entity_a=dana.id, entity_b=owner.id, user_id="ada",
                              reason="jev: unsure", compared_step=3, different=0.3),
        "thing": MergeProposal(entity_a=owner.id, entity_b=acme.id, user_id="ada",
                               status="rejected", reason="jev: different"),
    }
    for row in rows.values():
        store.backend.add_proposal(row)
    store.backend.set_meta("schema:owner-pairs:v1", "")  # as before the upgrade
    reopened = MemoryStore(Config(db_path=":memory:"), backend=store.backend,
                           llm=NoneLLM(), embedder=HashEmbedder(64))
    by_id = {p.id: p for p in store.backend.list_proposals(Scope(user_id="ada"), status=None)}
    for key in ("judged", "open"):
        row = by_id[rows[key].id]
        assert (row.status, row.reason, row.compared_step, row.different) == (
            "proposed", OWNER_PAIR_REOPENED, 0, None)
    assert by_id[rows["person"].id].status == "rejected"
    assert by_id[rows["thing"].id].status == "rejected"  # not a person
    marker = json.loads(store.backend.get_meta("schema:owner-pairs:v1"))
    assert {r["id"] for r in marker["reopened"]} == {rows["judged"].id, rows["open"].id}
    # once: a second open finds the marker; asked again, nothing is left to open
    again = MemoryStore(Config(db_path=":memory:"), backend=store.backend,
                        llm=NoneLLM(), embedder=HashEmbedder(64))
    assert store.backend.get_meta("schema:owner-pairs:v1") == json.dumps(marker)
    store.backend.set_meta("schema:owner-pairs:v1", "")
    again._settle_owner_pairs()
    assert json.loads(store.backend.get_meta("schema:owner-pairs:v1"))["reopened"] == []
    assert reopened.owner_name("ada") == "the user"
    # the owner is still not put to the judge until it has a name
    judge = _PairJudge(lambda state: (0.0, 0.99))
    judged = MemoryStore(Config(db_path=":memory:"), backend=store.backend,
                         llm=NoneLLM(), embedder=HashEmbedder(64), decider=judge)
    judged.resolve_entities(user_id="ada")
    assert not any("the user" in _names(state) for state in judge.states)
    judged.close()


def test_one_row_per_pair(tmp_path):
    from memry.backends.local import LocalBackend

    path = str(tmp_path / "pairs.db")
    backend = LocalBackend(path)
    a, b, c = (backend.insert_entity(Entity(name=n, normalized=n.lower(), entity_type="person",
                                            user_id="ada")) for n in ("A", "B", "C"))
    first = backend.add_proposal(MergeProposal(entity_a=a.id, entity_b=b.id, user_id="ada"))
    second = backend.add_proposal(MergeProposal(entity_a=b.id, entity_b=a.id, user_id="ada",
                                                status="rejected"))
    assert second.id == first.id and second.status == "proposed"
    assert len(backend.list_proposals(Scope(user_id="ada"), status=None)) == 1

    # rows written twice before: the more decided stays, a confirmed one stays
    rows = [
        ("p1", a.id, c.id, "proposed", "2026-09-27T19:02:24", None),
        ("p2", c.id, a.id, "rejected", "2026-09-27T19:02:24", "2026-09-28T10:00:00"),
        ("p3", b.id, c.id, "rejected", "2026-09-27T19:02:24", "2026-09-28T10:00:00"),
        ("p4", b.id, c.id, "confirmed", "2026-09-27T19:02:24", "2026-09-29T10:00:00"),
    ]
    backend._db.executemany(
        "INSERT INTO entity_proposals (id, entity_a, entity_b, user_id, status, confidence, "
        "created_at, decided_at) VALUES (?,?,?,'ada',?,0.5,?,?)", rows)
    backend._db.execute("DELETE FROM meta WHERE key = 'schema:one-row-per-pair:v1'")
    backend._db.commit()
    backend.close()
    backend = LocalBackend(path)
    left = {p.id: p.status for p in backend.list_proposals(Scope(user_id="ada"), status=None)}
    assert left == {first.id: "proposed", "p2": "rejected", "p4": "confirmed"}
    marker = json.loads(backend.get_meta("schema:one-row-per-pair:v1"))
    assert {row["id"] for row in marker["dropped"]} == {"p1", "p3"}
    backend.close()


def test_a_merge_keeps_one_row_for_the_owners_pair():
    """What left two rows for the owner and "Cosmin" on a real store: one pass
    paired the owner with "Cosmin" and "Cosmin Novac", and the two were merged
    later, by a version that moved the pair without looking for the other."""
    store, _, _ = _store()
    owner = _owner(store, "sailor42")
    cosmin = _entity_with(store, "Cosmin", ["Cosmin sails"], "person")
    novac = _entity_with(store, "Cosmin Novac", ["Cosmin Novac sails"], "person")
    for person in (cosmin, novac):
        store.backend.add_proposal(MergeProposal(entity_a=owner.id, entity_b=person.id,
                                                 user_id="ada"))
    store.backend.merge_entities(cosmin.id, novac.id)
    pairs = [p for p in store.backend.list_proposals(Scope(user_id="ada"), status=None)
             if owner.id in (p.entity_a, p.entity_b)]
    assert len(pairs) == 1
    store.close()


def _forgotten_cos(store) -> None:
    """The owner, "Cosmin" with many memories, and "The user's name is Cos."
    only in a forgotten duplicate."""
    owner = _owner(store, memories=3)
    _entity_with(store, "Cosmin", [f"Cosmin fact {i}" for i in range(12)], "person")
    _entity_with(store, "Constantin", ["Constantin fixes the roof"], "person")
    memory = store.backend.insert_memory(Memory(content="The user's name is Cos.",
                                                user_id="ada"))
    store.backend.add_mention(EntityMention(entity_id=owner.id, memory_id=memory.id,
                                            surface="the user"))
    store.backend.invalidate_memory(memory.id)


def test_learn_owner_reads_a_forgotten_statement_with_one_model_call():
    store, llm, _ = _store()
    _forgotten_cos(store)
    llm.queue(json.dumps({"person": "Cosmin", "name": "Cos"}))
    report = store.learn_owner(user_id="ada", dry_run=True)
    [(system, asked)] = llm.calls
    assert "The user's name is Cos. (forgotten)" in asked and '"Cosmin": 12 memories' in asked
    assert report["evidence"][0] == {"text": "The user's name is Cos.", "name": "Cos",
                                     "strong": True, "memory_id": report["evidence"][0]["memory_id"],
                                     "forgotten": True}
    assert report["decision"]["person"] == "Cosmin" and report["decision"]["by"] == "the text model"
    assert report["action"]["action"] == "would fold" and report["action"]["into"] == "Cosmin"
    assert report["action"]["owner_memories"] == 3 and report["action"]["into_memories"] == 12
    # the dry run wrote nothing
    assert store.owner_name("ada") == "the user"
    assert store._upkeep_get("owner_learned", "ada", None) is None
    assert store._upkeep_get("owner_stated", "ada", None) is None

    llm.queue(json.dumps({"person": "Cosmin", "name": "Cos"}))
    report = store.learn_owner(user_id="ada")
    assert report["action"]["action"] == "folded" and store.owner_name("ada") == "Cosmin"
    [merge] = store.merges(user_id="ada")
    assert "The user's name is Cos." in merge["decided"]
    assert store._upkeep_get("owner_learned", "ada", None)["action"] == "folded"
    store.close()


def test_learn_owner_without_a_text_model_counts_only_what_says_the_name_outright():
    store = MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(), embedder=HashEmbedder(64))
    _forgotten_cos(store)
    store.backend.insert_memory(Memory(content="I'm Constantin's neighbour", user_id="ada"))
    report = store.learn_owner(user_id="ada")
    assert report["decision"]["by"] == "the matching rule"
    assert report["action"]["action"] == "folded" and store.owner_name("ada") == "Cosmin"
    store.close()


def test_learn_owner_without_evidence_asks_nothing_and_changes_nothing():
    store, llm, _ = _store()
    _owner(store, memories=3)
    _entity_with(store, "Cosmin", ["Cosmin sails"], "person")
    report = store.learn_owner(user_id="ada")  # the FakeLLM has no answer to give
    assert report["action"] == {"action": "none", "reason": "nothing states who the user is"}
    assert store.owner_name("ada") == "the user" and llm.calls == []
    store.close()


def test_the_upkeep_cycle_learns_the_owner_once():
    store, llm, _ = _store()
    _forgotten_cos(store)
    llm.queue(json.dumps({"person": "Cosmin", "name": "Cos"}))
    ran = store.run_upkeep_cycle(user_id="ada")
    assert ran["learn_owner"]["action"] == "folded"
    asked = len(llm.calls)
    store.run_upkeep_cycle(user_id="ada")
    assert not any(u.startswith("Statements:") for _, u in llm.calls[asked:])
    store.close()


def test_memry_learn_owner(capsys, monkeypatch, tmp_path):
    from memry.cli import main

    path = tmp_path / "cli.db"
    monkeypatch.setenv("MEMRY_DB_PATH", str(path))
    monkeypatch.setenv("MEMRY_CONFIG", str(tmp_path / "missing.json"))
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "VOYAGE_API_KEY",
                "MEMRY_LLM_PROVIDER", "MEMRY_EMBEDDING_PROVIDER", "MEMRY_DECISION_PROVIDER"):
        monkeypatch.delenv(key, raising=False)
    store = MemoryStore(Config(db_path=str(path)), llm=NoneLLM(), embedder=HashEmbedder(64))
    _forgotten_cos(store)
    store.close()

    assert main(["learn-owner", "-u", "ada", "--dry-run"]) == 0
    [report] = json.loads(capsys.readouterr().out)
    assert report["dry_run"] and report["action"]["action"] == "would fold"
    assert main(["learn-owner", "-u", "ada"]) == 0
    [report] = json.loads(capsys.readouterr().out)
    assert report["action"]["action"] == "folded" and report["action"]["into"] == "Cosmin"
    store = MemoryStore(Config(db_path=str(path)), llm=NoneLLM(), embedder=HashEmbedder(64))
    assert store.owner_name("ada") == "Cosmin"
    store.close()
