"""The store owner: one entity for the person the memories belong to.

It starts from the account's name. Facts about the user collect on it, and the
identity judge decides, through the normal pair comparison, which named person
in the store it is.
"""

from __future__ import annotations

import json

from conftest import FakeLLM, fact, facts_response
from starlette.testclient import TestClient
from test_decisions import _entity_with, _names, _PairJudge

from memry.accounts import AccountStore
from memry.config import Config
from memry.models import EntityMention, Memory, MergeProposal, Scope
from memry.providers.embeddings import HashEmbedder
from memry.providers.llm import NoneLLM
from memry.rest import create_app
from memry.store import MemoryStore


def _store(answer=lambda state: (0.5, 0.1)):
    llm = FakeLLM()
    judge = _PairJudge(answer)
    store = MemoryStore(Config(db_path=":memory:"), llm=llm, embedder=HashEmbedder(64),
                        decider=judge)

    def save(text: str, *names: str) -> None:
        llm.queue(facts_response(fact(text, entities=[
            {"name": name, "type": "person"} for name in names])))
        if store.get_all(user_id="ada"):
            llm.queue(json.dumps({"action": "ADD", "target": None, "content": None,
                                  "reason": "new"}))
        store.add(text, user_id="ada")

    return store, save, llm, judge


def _extraction_prompts(llm) -> list[str]:
    return [user for _, user in llm.calls if user.startswith("Conversation:")]


def test_the_extractor_is_told_who_the_user_is():
    store, save, llm, _ = _store()
    save("The user prefers green tea", "the user")
    assert 'is the entity "the user"' in _extraction_prompts(llm)[-1]
    store.set_owner_name("ada", "ada")
    save("The user sails on weekends", "the user")
    # The owner entity exists now, and its name wins over the account's.
    assert 'is the entity "the user"' in _extraction_prompts(llm)[-1]
    store.close()


def test_facts_about_the_user_collect_on_one_owner_entity_without_asking_the_judge():
    store, save, _, judge = _store()
    store.set_owner_name("ada", "ada")
    save("The user prefers green tea", "ada")
    save("The user sails on weekends", "ada")
    [owner] = store.entities(user_id="ada")
    assert owner.name == "ada" and owner.metadata["owner"] is True
    assert store.owner_entity("ada").id == owner.id
    assert len(store.backend.entity_memories(owner.id)) == 2
    assert judge.states == []
    store.close()


def test_the_owner_is_folded_into_the_person_it_is_found_to_be():
    """Keeping the person's name: "Ada Lindqvist", not "ada"."""
    store, save, llm, judge = _store(lambda state: (0.99, 0.0))
    store.set_owner_name("ada", "ada")
    save("The user sails on weekends", "ada")
    _entity_with(store, "Ada Lindqvist", ["Ada Lindqvist sails in Stockholm"], "person")
    store.resolve_entities(user_id="ada")
    [person] = store.entities(user_id="ada")
    assert person.name == "Ada Lindqvist" and person.metadata["owner"] is True
    assert store.owner_name("ada") == "Ada Lindqvist"
    assert any("This entity is the owner of the memory store" in s for s in judge.states)
    save("The user bought a boat", "Ada Lindqvist")
    assert 'is the entity "Ada Lindqvist"' in _extraction_prompts(llm)[-1]
    assert len(store.backend.entity_memories(person.id)) == 3
    store.close()


def test_a_pair_is_kept_apart_only_from_10_memories():
    """Keeping apart ends all comparing. With a few memories the owner read as
    someone "Cosmin Novac" is not (P(different) 0.90-0.95 on a real store), and
    "Fundation" as a different thing from "Fundation GmbH" (0.68-0.70); at 10
    memories both read as one (P(same) 0.99 and 0.91-0.93)."""
    from memry.intelligence.identity import compare

    store, save, _, judge = _store(lambda state: (0.02, 0.95))
    owner = _entity_with(store, "the user", ["The user sails"], "person")
    store.backend.set_entity_metadata(owner.id, {"owner": True})
    owner = store.backend.get_entity(owner.id)
    ada = _entity_with(store, "Ada Lindqvist", [f"Ada Lindqvist fact {i}" for i in range(12)],
                       "person")
    bob = _entity_with(store, "Bob", ["Bob fixes bikes"], "person")
    assert compare(judge, store.backend, owner, ada).action == "wait"
    assert compare(judge, store.backend, bob, ada).action == "wait"
    for i in range(9):
        memory = store.backend.insert_memory(Memory(content=f"The user fact {i}", user_id="ada"))
        store.backend.add_mention(EntityMention(entity_id=owner.id, memory_id=memory.id,
                                                surface="the user"))
    verdict = compare(judge, store.backend, owner, ada, compared=1)
    assert (verdict.action, verdict.step) == ("apart", 10)
    store.close()


def test_the_owner_is_compared_with_the_people_whose_memories_are_closest():
    """An owner named "the user" shares no name with anyone."""
    store, save, _, judge = _store(lambda state: (0.5, 0.1))
    save("the user sails a blue boat around Stockholm harbour", "the user")
    for name, text in (("Ada Lindqvist", "Ada Lindqvist sails a blue boat around Stockholm harbour"),
                       ("Bob", "Bob repairs bicycles in Lyon"),
                       ("Chen", "Chen teaches chemistry at a school in Taipei"),
                       ("Dana", "Dana writes invoices for a bakery"),
                       ("Emil", "Emil plays chess online every night")):
        save(text, name)
    judge.states.clear()
    store.resolve_entities(user_id="ada")
    compared = {tuple(sorted(_names(s))) for s in judge.states}
    assert ("Ada Lindqvist", "the user") in compared
    assert len({pair for pair in compared if "the user" in pair}) == 3
    store.close()


def test_a_merge_starts_the_funnel_again_for_the_merged_entitys_open_pairs():
    store, _, _, _ = _store()
    a = _entity_with(store, "Ada", ["Ada sails"], "person")
    b = _entity_with(store, "Ada Lindqvist", ["Ada Lindqvist sails"], "person")
    c = _entity_with(store, "Ada L.", ["Ada L. sails"], "person")
    store.backend.add_proposal(MergeProposal(entity_a=b.id, entity_b=c.id, user_id="ada",
                                             compared_step=10))
    store.backend.merge_entities(a.id, b.id)
    [proposal] = store.backend.list_proposals(Scope(user_id="ada"))
    assert proposal.compared_step == 0 and {proposal.entity_a, proposal.entity_b} == {a.id, c.id}
    store.close()


def test_an_account_names_the_owner_of_its_namespace():
    accounts = AccountStore(":memory:")
    store = MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(), embedder=HashEmbedder(64))
    admin_key = accounts.issue_key(accounts.create("cosmin").name)
    member_key = accounts.issue_key(accounts.create("alice").name)
    client = TestClient(create_app(store, accounts=accounts))
    for key in (admin_key, member_key):
        client.get("/api/v1/memories", headers={"Authorization": f"Bearer {key}"})
    assert store.owner_name("default") == "cosmin"
    assert store.owner_name("alice::default") == "alice"
    assert store.owner_name("nobody") == "the user"
    store.close()
    accounts.close()


def test_the_named_person_survives_a_merge_with_the_owner_in_either_order():
    from memry.intelligence.identity import merge_pair

    for owner_first in (True, False):
        store, _, _, _ = _store()
        owner = _entity_with(store, "the user", ["The user sails"], "person")
        store.backend.set_entity_metadata(owner.id, {"owner": True})
        owner = store.backend.get_entity(owner.id)
        ada = _entity_with(store, "Ada Lindqvist", ["Ada Lindqvist sails"], "person")
        assert merge_pair(store.backend, *((owner, ada) if owner_first else (ada, owner)))
        [kept] = store.entities(user_id="ada")
        assert kept.id == ada.id and kept.metadata["owner"] is True
        store.close()


def test_the_name_screen_leaves_the_owner_alone():
    store, save, _, judge = _store()
    save("The user prefers green tea", "the user")
    asked = []
    judge.decide = lambda state, questions: asked.append(state) or __import__(
        "memry.providers.decisions", fromlist=["Answers"]).Answers({})
    store.run_name_screen(user_id="ada")
    assert asked == []
    store.close()


def test_the_extractor_is_offered_the_stored_names_a_text_may_mean():
    store, save, llm, _ = _store()
    for name in ("Fundation GmbH", "Amazon Web Services", "Raluca Novac"):
        _entity_with(store, name, [f"{name} is in the store"])
    save("Fundation paid the AWS bill", "Fundation")
    prompt = _extraction_prompts(llm)[-1]
    offered = json.loads(prompt.split("does:\n")[1].split("\n\n")[0])
    assert {e["name"] for e in offered} == {"Fundation GmbH", "Amazon Web Services"}
    store.close()
