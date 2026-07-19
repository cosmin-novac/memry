from __future__ import annotations

import json

from conftest import fact, facts_response, decision


def identity(verdict: str, confidence: float, reason: str = "test") -> str:
    return json.dumps({"verdict": verdict, "confidence": confidence, "reason": reason})


def add_fact_with_entity(store, fake_llm, content: str, entity: str, *identity_responses: str):
    """Queue extraction (one fact w/ entity) + optional reconcile-skip + identity calls."""
    fake_llm.queue(facts_response(fact(content, entities=[entity])))
    # a reconcile decision is needed once similar memories exist
    if store.get_all(user_id="ada"):
        fake_llm.queue(decision("ADD", reason="distinct fact"))
    fake_llm.queue(*identity_responses)
    return store.add(content, user_id="ada")


def test_first_mention_creates_entity(store, fake_llm):
    add_fact_with_entity(store, fake_llm, "User's partner Jonas loves Thai food", "Jonas")
    entities = store.entities(user_id="ada")
    assert len(entities) == 1
    assert entities[0].name == "Jonas"
    detail = store.entity(entities[0].id)
    assert detail["memories"][0].content == "User's partner Jonas loves Thai food"


def test_unsure_same_name_stays_separate_with_proposal(store, fake_llm):
    add_fact_with_entity(store, fake_llm, "User's partner Jonas loves Thai food", "Jonas")
    add_fact_with_entity(
        store, fake_llm, "A colleague named Jonas reviewed the phoenix design", "Jonas",
        identity("unsure", 0.5, "same name, unclear identity"),
    )
    entities = store.entities(user_id="ada")
    assert len(entities) == 2  # two Jonases, deliberately

    proposals = store.merge_proposals(user_id="ada")
    assert len(proposals) == 1
    assert proposals[0].status == "proposed"


def test_three_jonases_stay_three(store, fake_llm):
    add_fact_with_entity(store, fake_llm, "User's partner Jonas loves Thai food", "Jonas")
    add_fact_with_entity(
        store, fake_llm, "Colleague Jonas reviewed the phoenix design", "Jonas",
        identity("unsure", 0.5),
    )
    add_fact_with_entity(
        store, fake_llm, "Neighbor Jonas borrowed the ladder", "Jonas",
        identity("different", 0.95), identity("unsure", 0.4),
    )
    assert len(store.entities(user_id="ada")) == 3
    # different -> no proposal; unsure -> proposal
    assert len(store.merge_proposals(user_id="ada")) == 2


def test_confident_same_attaches_to_existing(store, fake_llm):
    add_fact_with_entity(store, fake_llm, "User's partner Jonas loves Thai food", "Jonas")
    add_fact_with_entity(
        store, fake_llm, "User's partner Jonas is allergic to shellfish", "Jonas",
        identity("same", 0.97, "both describe the user's partner"),
    )
    entities = store.entities(user_id="ada")
    assert len(entities) == 1
    detail = store.entity(entities[0].id)
    assert len(detail["memories"]) == 2
    assert store.merge_proposals(user_id="ada") == []


def test_user_confirms_merge(store, fake_llm):
    add_fact_with_entity(store, fake_llm, "Jonas plays guitar", "Jonas")
    add_fact_with_entity(
        store, fake_llm, "Jonas started guitar lessons in Berlin", "Jonas",
        identity("unsure", 0.6),
    )
    proposal = store.merge_proposals(user_id="ada")[0]
    assert store.confirm_merge(proposal.id)

    active = store.entities(user_id="ada")
    assert len(active) == 1
    merged = store.entities(user_id="ada", include_merged=True)
    assert len(merged) == 2
    loser = next(e for e in merged if e.merged_into)
    assert loser.merged_into == active[0].id
    # mentions were repointed: winner now carries both memories
    assert len(store.entity(active[0].id)["memories"]) == 2
    # proposal is settled
    assert store.merge_proposals(user_id="ada") == []
    assert not store.confirm_merge(proposal.id)  # can't decide twice


def test_user_rejects_merge(store, fake_llm):
    add_fact_with_entity(store, fake_llm, "Jonas the partner cooks", "Jonas")
    add_fact_with_entity(
        store, fake_llm, "Jonas from accounting emailed", "Jonas",
        identity("unsure", 0.5),
    )
    proposal = store.merge_proposals(user_id="ada")[0]
    assert store.reject_merge(proposal.id)
    assert len(store.entities(user_id="ada")) == 2
    rejected = store.merge_proposals(user_id="ada", status="rejected")
    assert len(rejected) == 1


def test_resolve_auto_confirms_only_clear_matches(store, fake_llm):
    add_fact_with_entity(store, fake_llm, "Jonas is the user's partner and a chef", "Jonas")
    add_fact_with_entity(
        store, fake_llm, "Jonas the chef cooked dinner with the user", "Jonas",
        identity("unsure", 0.6),
    )
    add_fact_with_entity(
        store, fake_llm, "A different Priya joined the team", "Priya",
    )
    # resolve: re-judge the open Jonas proposal -> now clearly the same
    fake_llm.queue(identity("same", 0.95, "both are the user's chef partner"))
    outcome = store.resolve_entities(user_id="ada")
    assert outcome["confirmed"] == 1
    assert len(store.entities(user_id="ada")) == 2  # merged Jonas + Priya


def test_no_llm_keeps_separate_and_proposes(verbatim_store):
    """Zero-LLM path: resolve_mentions is only reachable via explicit entities,
    but the policy must still be safe if invoked."""
    from memry.intelligence.entities import resolve_mentions
    from memry.models import Scope

    backend = verbatim_store.backend
    verbatim_store.add("Jonas one", user_id="ada", infer=False)
    memory_1 = verbatim_store.get_all(user_id="ada")[0]
    resolve_mentions(
        backend=backend, llm=verbatim_store.llm, scope=Scope(user_id="ada"),
        memory_id=memory_1.id, memory_content=memory_1.content, surfaces=["Jonas"],
    )
    verbatim_store.add("Jonas two", user_id="ada", infer=False)
    memory_2 = [m for m in verbatim_store.get_all(user_id="ada") if m.id != memory_1.id][0]
    resolve_mentions(
        backend=backend, llm=verbatim_store.llm, scope=Scope(user_id="ada"),
        memory_id=memory_2.id, memory_content=memory_2.content, surfaces=["Jonas"],
    )
    assert len(backend.list_entities(Scope(user_id="ada"))) == 2
    assert len(backend.list_proposals(Scope(user_id="ada"))) == 1


def test_entity_scoping_isolated(store, fake_llm):
    fake_llm.queue(facts_response(fact("Jonas fact", entities=["Jonas"])))
    store.add("about jonas", user_id="ada")
    fake_llm.queue(facts_response(fact("Jonas other-user fact", entities=["Jonas"])))
    # different user scope: no candidates, no identity call
    store.add("about another jonas", user_id="bob")
    assert len(store.entities(user_id="ada")) == 1
    assert len(store.entities(user_id="bob")) == 1
