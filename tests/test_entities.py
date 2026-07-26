from __future__ import annotations

import json

import pytest

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


def test_entity_memories_exclude_invalid_by_default(verbatim_store):
    from memry.models import Entity, EntityMention, Memory

    backend = verbatim_store.backend
    entity = backend.insert_entity(
        Entity(name="Marcus", user_id="ada", updated_at="2020-01-01T00:00:00+00:00")
    )
    memory = backend.insert_memory(Memory(content="Marcus is a good student", user_id="ada"))
    backend.add_mention(
        EntityMention(entity_id=entity.id, memory_id=memory.id, surface="Marcus")
    )

    assert [m.id for m in backend.entity_memories(entity.id)] == [memory.id]
    backend.invalidate_memory(memory.id)

    assert backend.entity_memories(entity.id) == []
    assert [m.id for m in backend.entity_memories(entity.id, include_invalid=True)] == [memory.id]
    assert backend.get_entity(entity.id).updated_at > "2020-01-01T00:00:00+00:00"


def test_hard_delete_removes_mentions_and_touches_entity(verbatim_store):
    from memry.models import Entity, EntityMention, Memory

    backend = verbatim_store.backend
    entity = backend.insert_entity(
        Entity(name="Marcus", user_id="ada", updated_at="2020-01-01T00:00:00+00:00")
    )
    memory = backend.insert_memory(Memory(content="Marcus studies physics", user_id="ada"))
    backend.add_mention(
        EntityMention(entity_id=entity.id, memory_id=memory.id, surface="Marcus")
    )

    assert backend.delete_memory(memory.id)
    assert backend.entity_mentions(entity.id) == []
    assert backend.entity_memories(entity.id, include_invalid=True) == []
    assert backend.get_entity(entity.id).updated_at > "2020-01-01T00:00:00+00:00"


def test_merge_touches_surviving_entity(verbatim_store):
    from memry.models import Entity

    backend = verbatim_store.backend
    keep = backend.insert_entity(
        Entity(name="Marcus", user_id="ada", updated_at="2020-01-01T00:00:00+00:00")
    )
    merge = backend.insert_entity(Entity(name="Cozmin", user_id="ada"))

    assert backend.merge_entities(keep.id, merge.id)
    assert backend.get_entity(keep.id).updated_at > "2020-01-01T00:00:00+00:00"

def test_aliases_are_derived_and_user_aliases_are_indexed(verbatim_store):
    from memry.models import Entity, EntityMention, Memory, Scope

    backend = verbatim_store.backend
    entity = backend.insert_entity(Entity(name="Marcus Popescu", user_id="ada"))
    memory = backend.insert_memory(
        Memory(content="C. Popescu teaches physics", user_id="ada")
    )
    backend.add_mention(
        EntityMention(entity_id=entity.id, memory_id=memory.id, surface="C. Popescu")
    )
    assert verbatim_store.add_entity_alias(entity.id, "Costi") is not None

    aliases = backend.entity_aliases(entity.id)
    assert aliases == ["Marcus Popescu", "C. Popescu", "Costi"]
    assert [candidate.id for candidate in backend.find_entity_candidates(
        "costi", Scope(user_id="ada")
    )] == [entity.id]
    assert [candidate.id for candidate in backend.find_entity_candidates(
        "c. popescu", Scope(user_id="ada")
    )] == [entity.id]


def test_entity_description_is_lazy_bounded_and_active_only(verbatim_store):
    from memry.models import Entity, EntityMention, Memory

    backend = verbatim_store.backend
    entity = backend.insert_entity(Entity(name="Marcus", entity_type="person", user_id="ada"))
    active = backend.insert_memory(
        Memory(content="Marcus is a strong physics student.", user_id="ada")
    )
    obsolete = backend.insert_memory(
        Memory(content="Marcus studies chemistry.", user_id="ada")
    )
    for memory in (active, obsolete):
        backend.add_mention(
            EntityMention(entity_id=entity.id, memory_id=memory.id, surface="Marcus")
        )

    first = verbatim_store.entity(entity.id)
    assert "physics" in first["entity"].description
    assert "chemistry" in first["entity"].description
    assert first["entity"].description_updated_at is not None

    backend.invalidate_memory(obsolete.id)
    stale = backend.get_entity(entity.id)
    assert stale.description_updated_at is None
    refreshed = verbatim_store.entity(entity.id)
    assert "physics" in refreshed["entity"].description
    assert "chemistry" not in refreshed["entity"].description
    assert [memory.id for memory in refreshed["memories"]] == [active.id]


def test_entity_description_is_in_reconstructed_context(verbatim_store):
    from memry.models import Entity, EntityMention, Memory

    backend = verbatim_store.backend
    entity = backend.insert_entity(Entity(name="Marcus", user_id="ada"))
    memory = backend.insert_memory(
        Memory(content="Marcus is a good student.", user_id="ada")
    )
    backend.add_mention(
        EntityMention(entity_id=entity.id, memory_id=memory.id, surface="Marcus")
    )

    context = verbatim_store.reconstruct_context("What do we know about Marcus?", user_id="ada")
    assert "## Known entities" in context.text
    assert "Marcus is a good student" in context.text
    assert memory.id in context.memory_ids

def test_full_name_with_overlapping_context_reuses_entity_without_llm(verbatim_store):
    from memry.intelligence.entities import resolve_mentions
    from memry.models import Memory, Scope

    backend = verbatim_store.backend
    scope = Scope(user_id="ada")
    first = backend.insert_memory(
        Memory(content="Marcus Vandenberg invests in Bitcoin.", user_id="ada")
    )
    resolve_mentions(
        backend=backend,
        llm=verbatim_store.llm,
        scope=scope,
        memory_id=first.id,
        memory_content=first.content,
        surfaces=["Marcus Vandenberg"],
        types={"marcus vandenberg": "person"},
    )
    second = backend.insert_memory(
        Memory(content="Marcus Vandenberg increased his Bitcoin investment.", user_id="ada")
    )
    resolve_mentions(
        backend=backend,
        llm=verbatim_store.llm,
        scope=scope,
        memory_id=second.id,
        memory_content=second.content,
        surfaces=["Marcus Vandenberg"],
        types={"marcus vandenberg": "person"},
    )

    entities = verbatim_store.entities(user_id="ada")
    assert len(entities) == 1
    assert {memory.id for memory in backend.entity_memories(entities[0].id)} == {
        first.id,
        second.id,
    }
    assert verbatim_store.merge_proposals(user_id="ada") == []


def test_same_full_name_without_shared_context_stays_separate(verbatim_store):
    from memry.intelligence.entities import resolve_mentions
    from memry.models import Memory, Scope

    backend = verbatim_store.backend
    scope = Scope(user_id="ada")
    for content in (
        "Marcus Vandenberg performs electronic music on stage.",
        "Marcus Vandenberg sold a Dacia car in Germany.",
    ):
        memory = backend.insert_memory(Memory(content=content, user_id="ada"))
        resolve_mentions(
            backend=backend,
            llm=verbatim_store.llm,
            scope=scope,
            memory_id=memory.id,
            memory_content=memory.content,
            surfaces=["Marcus Vandenberg"],
            types={"marcus vandenberg": "person"},
        )

    assert len(verbatim_store.entities(user_id="ada")) == 2
    assert len(verbatim_store.merge_proposals(user_id="ada")) == 1


def test_maintenance_auto_confirms_obvious_full_name_pair(verbatim_store):
    from memry.models import Entity, EntityMention, Memory, MergeProposal

    backend = verbatim_store.backend
    first = backend.insert_entity(
        Entity(name="Marcus Vandenberg", entity_type="person", user_id="ada")
    )
    second = backend.insert_entity(
        Entity(name="Marcus Vandenberg", entity_type="person", user_id="ada")
    )
    first_memory = backend.insert_memory(
        Memory(content="Marcus Vandenberg holds Bitcoin.", user_id="ada")
    )
    second_memory = backend.insert_memory(
        Memory(content="Marcus Vandenberg tracks his Bitcoin investment.", user_id="ada")
    )
    backend.add_mention(
        EntityMention(entity_id=first.id, memory_id=first_memory.id, surface=first.name)
    )
    backend.add_mention(
        EntityMention(entity_id=second.id, memory_id=second_memory.id, surface=second.name)
    )
    proposal = backend.add_proposal(
        MergeProposal(entity_a=first.id, entity_b=second.id, user_id="ada")
    )

    assert verbatim_store.resolve_entities(user_id="ada") == {
        "confirmed": 1,
        "rejected": 0,
        "kept": 0,
        "proposed": 0,
        "purged": 0,
    }
    assert len(verbatim_store.entities(user_id="ada")) == 1
    assert backend.get_proposal(proposal.id).status == "confirmed"


def test_confirm_merge_follows_already_merged_endpoint(verbatim_store):
    from memry.models import Entity, MergeProposal

    backend = verbatim_store.backend
    keep = backend.insert_entity(Entity(name="Marcus Vandenberg", user_id="ada"))
    old = backend.insert_entity(Entity(name="Marcus N.", user_id="ada"))
    assert backend.merge_entities(keep.id, old.id)
    legacy = backend.add_proposal(
        MergeProposal(entity_a=keep.id, entity_b=old.id, user_id="ada")
    )

    assert verbatim_store.confirm_merge(legacy.id)
    assert backend.get_proposal(legacy.id).status == "confirmed"
    assert backend.resolve_entity_id(old.id) == keep.id


def test_memory_edit_reanalyzes_and_replaces_entity_links(store, fake_llm):
    from memry.models import Entity, EntityMention, Memory

    backend = store.backend
    old_entity = backend.insert_entity(Entity(name="Marcus", user_id="ada"))
    memory = backend.insert_memory(
        Memory(content="Marcus is a good student.", entities=["Marcus"], user_id="ada")
    )
    backend.add_mention(
        EntityMention(entity_id=old_entity.id, memory_id=memory.id, surface="Marcus")
    )
    fake_llm.queue(
        facts_response(fact("Ada is a good student.", entities=["Ada"]))
    )

    updated = store.update(memory.id, content="Ada is a good student.")

    assert updated.entities == ["Ada"]
    assert [entity.name for entity in backend.entities_of_memory(memory.id)] == ["Ada"]
    assert backend.entity_memories(old_entity.id) == []


def test_memory_edit_analysis_failure_keeps_text_and_links(store):
    from memry.models import Entity, EntityMention, Memory

    backend = store.backend
    entity = backend.insert_entity(Entity(name="Marcus", user_id="ada"))
    memory = backend.insert_memory(
        Memory(content="Marcus is a good student.", entities=["Marcus"], user_id="ada")
    )
    backend.add_mention(
        EntityMention(entity_id=entity.id, memory_id=memory.id, surface="Marcus")
    )

    with pytest.raises(ValueError, match="entity re-analysis failed"):
        store.update(memory.id, content="Ada is a good student.")

    assert backend.get_memory(memory.id).content == "Marcus is a good student."
    assert [linked.id for linked in backend.entities_of_memory(memory.id)] == [entity.id]


# ------------------------------------------- empty records cannot disagree
from memry.intelligence.entities import resolve_mentions  # noqa: E402
from memry.models import (  # noqa: E402
    Entity, EntityMention, Memory, MergeProposal, Scope,
)
from memry.providers.llm import NoneLLM  # noqa: E402


def test_same_name_with_no_evidence_reuses_instead_of_forking(verbatim_store):
    """An identically-named record with no facts has no identity to differ from.

    Forking there produced the duplicate pile a real store accumulated: 4x
    "sehr geehrte", 3x "the father of photography", and 206 of 519 entities
    with zero mentions, each dragging along an unanswerable merge proposal.
    """
    backend = verbatim_store.backend
    empty = backend.insert_entity(
        Entity(name="Fundation GmbH", normalized="fundation gmbh", user_id="ada")
    )
    memory = backend.insert_memory(
        Memory(content="Fundation GmbH has VAT ID DE123.", user_id="ada")
    )
    resolved = resolve_mentions(
        backend=backend, llm=NoneLLM(), scope=Scope(user_id="ada"),
        memory_id=memory.id, memory_content=memory.content,
        surfaces=["Fundation GmbH"],
    )
    assert resolved["fundation gmbh"].id == empty.id
    assert len(verbatim_store.entities(user_id="ada")) == 1
    assert backend.list_proposals(Scope(user_id="ada"), status="proposed") == []


def test_a_record_with_evidence_is_still_judged_not_blindly_reused(verbatim_store):
    """The protection only applies to empty records; evidence still decides."""
    backend = verbatim_store.backend
    known = backend.insert_entity(
        Entity(name="Jonas", normalized="jonas", user_id="ada")
    )
    first = backend.insert_memory(
        Memory(content="Jonas is the plumber from Bremen.", user_id="ada")
    )
    backend.add_mention(
        EntityMention(entity_id=known.id, memory_id=first.id, surface="Jonas")
    )
    second = backend.insert_memory(
        Memory(content="Jonas is my nephew, born 2019.", user_id="ada")
    )
    resolve_mentions(
        backend=backend, llm=NoneLLM(), scope=Scope(user_id="ada"),
        memory_id=second.id, memory_content=second.content, surfaces=["Jonas"],
    )
    # single common name + real conflicting evidence -> kept separate
    assert len(verbatim_store.entities(user_id="ada")) == 2


def test_maintenance_settles_unanswerable_same_name_proposals(verbatim_store):
    backend = verbatim_store.backend
    with_facts = backend.insert_entity(
        Entity(name="Sehr geehrte", normalized="sehr geehrte", user_id="ada")
    )
    memory = backend.insert_memory(
        Memory(content="Sehr geehrte is used as a salutation.", user_id="ada")
    )
    backend.add_mention(
        EntityMention(entity_id=with_facts.id, memory_id=memory.id, surface="Sehr geehrte")
    )
    empty = backend.insert_entity(
        Entity(name="Sehr geehrte", normalized="sehr geehrte", user_id="ada")
    )
    backend.add_proposal(
        MergeProposal(entity_a=with_facts.id, entity_b=empty.id, user_id="ada")
    )
    result = verbatim_store.resolve_entities(user_id="ada")
    assert result["confirmed"] == 1 and result["kept"] == 0
    remaining = verbatim_store.entities(user_id="ada")
    assert [e.id for e in remaining] == [with_facts.id]  # the evidenced one survives


def test_orphan_entities_are_purged_but_referenced_ones_survive(verbatim_store):
    backend = verbatim_store.backend
    orphan = backend.insert_entity(
        Entity(name="bracketed placeholders", normalized="bracketed placeholders",
               user_id="ada")
    )
    kept = backend.insert_entity(
        Entity(name="Helios", normalized="helios", user_id="ada")
    )
    memory = backend.insert_memory(Memory(content="Helios ships Friday.", user_id="ada"))
    backend.add_mention(
        EntityMention(entity_id=kept.id, memory_id=memory.id, surface="Helios")
    )
    assert verbatim_store.resolve_entities(user_id="ada")["purged"] == 1
    ids = {e.id for e in verbatim_store.entities(user_id="ada")}
    assert ids == {kept.id}
    assert orphan.id not in ids
    # idempotent: nothing left to purge
    assert verbatim_store.resolve_entities(user_id="ada")["purged"] == 0


def test_preexisting_duplicates_get_a_proposal_so_they_can_be_reconsidered(verbatim_store):
    """Duplicates that predate a fix have nothing scheduled to look at them.

    Proposals are only created at write time, so two same-named entities that
    already exist would otherwise sit in the graph for good, with no open
    proposal and therefore no path to resolution.
    """
    backend = verbatim_store.backend
    scope = Scope(user_id="ada")
    first = backend.insert_entity(
        Entity(name="AI Flow", normalized="ai flow", user_id="ada")
    )
    second = backend.insert_entity(
        Entity(name="AI Flow", normalized="ai flow", user_id="ada")
    )
    for entity, text in ((first, "AI Flow ships weekly."),
                         (second, "AI Flow uses Postgres.")):
        memory = backend.insert_memory(Memory(content=text, user_id="ada"))
        backend.add_mention(
            EntityMention(entity_id=entity.id, memory_id=memory.id, surface="AI Flow")
        )
    assert backend.list_proposals(scope, status="proposed") == []

    result = verbatim_store.resolve_entities(user_id="ada")
    assert result["proposed"] == 1
    # both carry evidence, so with no LLM it stays for judgement rather than
    # being merged blind - but it is now visible and will be reconsidered
    assert result["confirmed"] + result["kept"] == 1


def test_entity_detail_carries_its_own_relations(verbatim_store):
    """Relations are shown under the entity they describe, not as a flat list.

    An edge only means something next to the thing it connects, and these are
    the same edges relational retrieval traverses.
    """
    from memry.models import Relation

    backend = verbatim_store.backend
    ada = backend.insert_entity(Entity(name="Ada", normalized="ada", user_id="ada"))
    helios = backend.insert_entity(
        Entity(name="Helios", normalized="helios", user_id="ada")
    )
    memory = backend.insert_memory(
        Memory(content="Ada works on Helios.", user_id="ada")
    )
    backend.add_relation(Relation(subject=ada.id, predicate="works_on",
                                  object=helios.id, user_id="ada",
                                  memory_id=memory.id))

    detail = verbatim_store.entity(ada.id)
    assert [r.predicate for r in detail["relations"]] == ["works_on"]
    # both endpoints are named, so the UI never has to render a bare id
    assert detail["relation_names"][helios.id] == "Helios"
    assert detail["relation_names"][ada.id] == "Ada"
    # and the edge is reachable from the other side too
    assert [r.predicate for r in verbatim_store.entity(helios.id)["relations"]] == [
        "works_on"
    ]
