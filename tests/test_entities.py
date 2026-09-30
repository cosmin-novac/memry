from __future__ import annotations

import json
import re

import pytest

from conftest import FakeLLM, decision, fact, facts_response
from memry.config import Config
from memry.intelligence.entities import IDENTITY_SYSTEM
from memry.intelligence.extraction import COVERAGE_SYSTEM
from memry.intelligence.reconcile import RECONCILE_SYSTEM
from memry.providers.decisions import Answer, Answers, NoneDecider
from memry.providers.embeddings import HashEmbedder
from memry.store import MemoryStore


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


def test_a_known_name_joins_its_entity_and_the_text_model_is_not_asked(store, fake_llm):
    """Without a calibrated judge a save asks no identity question, even of a
    text model whose gate was measured (the fixture pins gpt-5-mini's): a name
    the store already has joins its entity by rule. Two people of one name are
    told apart by a person, or by a type the extractor gives each."""
    add_fact_with_entity(store, fake_llm, "User's partner Jonas loves Thai food", "Jonas")
    add_fact_with_entity(
        store, fake_llm, "User's partner Jonas is allergic to shellfish", "Jonas")
    [jonas] = store.entities(user_id="ada")
    assert len(store.entity(jonas.id)["memories"]) == 2
    assert store.merge_proposals(user_id="ada") == []
    assert fake_llm.responses == []
    assert all(system != IDENTITY_SYSTEM for system, _ in fake_llm.calls)


def _two_jonases(store, first: str, second: str):
    """Two "Jonas" entities, one memory each, and the open pair between them."""
    from memry.models import Entity, EntityMention, Memory, MergeProposal

    backend = store.backend
    ids = []
    for text in (first, second):
        entity = backend.insert_entity(Entity(name="Jonas", normalized="jonas", user_id="ada"))
        memory = backend.insert_memory(Memory(content=text, user_id="ada"))
        backend.add_mention(EntityMention(entity_id=entity.id, memory_id=memory.id,
                                          surface="Jonas"))
        ids.append(entity.id)
    return backend.add_proposal(MergeProposal(entity_a=ids[0], entity_b=ids[1],
                                              user_id="ada", confidence=0.6))


def test_user_confirms_merge(store):
    proposal = _two_jonases(store, "Jonas plays guitar", "Jonas started guitar lessons in Berlin")
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


def test_user_rejects_merge(store):
    proposal = _two_jonases(store, "Jonas the partner cooks", "Jonas from accounting emailed")
    assert store.reject_merge(proposal.id)
    assert len(store.entities(user_id="ada")) == 2
    rejected = store.merge_proposals(user_id="ada", status="rejected")
    assert len(rejected) == 1


def test_resolve_auto_confirms_only_clear_matches(store, fake_llm):
    _two_jonases(store, "Jonas is the user's partner and a chef",
                 "Jonas the chef cooked dinner with the user")
    add_fact_with_entity(
        store, fake_llm, "A different Priya joined the team", "Priya",
    )
    # resolve: re-judge the open Jonas proposal -> now clearly the same
    fake_llm.queue(identity("same", 0.95, "both are the user's chef partner"))
    outcome = store.resolve_entities(user_id="ada")
    assert outcome["confirmed"] == 1
    assert len(store.entities(user_id="ada")) == 2  # merged Jonas + Priya


def test_no_llm_joins_a_known_name_by_rule(verbatim_store):
    """Zero-LLM path: resolve_mentions is only reachable via explicit entities,
    and a name the store already has joins its entity by rule, as it does with
    a text model and no calibrated judge."""
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
    [jonas] = backend.list_entities(Scope(user_id="ada"))
    assert [m.decided for m in backend.entity_mentions(jonas.id)] == [
        None, {"reason": "the one entity of this name"}]
    assert backend.list_proposals(Scope(user_id="ada")) == []


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
    """The protection only applies to empty records; with a calibrated judge,
    evidence still decides."""
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
        backend=backend, llm=NoneLLM(), decider=_Judge(same=0.05, different=0.9),
        scope=Scope(user_id="ada"),
        memory_id=second.id, memory_content=second.content, surfaces=["Jonas"],
    )
    # single common name + real conflicting evidence -> kept separate
    assert len(verbatim_store.entities(user_id="ada")) == 2


def test_maintenance_settles_unanswerable_same_name_proposals(verbatim_store):
    backend = verbatim_store.backend
    with_facts = backend.insert_entity(
        Entity(name="Fundation GmbH", normalized="fundation gmbh", user_id="ada")
    )
    memory = backend.insert_memory(
        Memory(content="Fundation GmbH is registered in Cologne.", user_id="ada")
    )
    backend.add_mention(
        EntityMention(entity_id=with_facts.id, memory_id=memory.id, surface="Fundation GmbH")
    )
    empty = backend.insert_entity(
        Entity(name="Fundation GmbH", normalized="fundation gmbh", user_id="ada")
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


# ------------------------------------------------- names that are not things
def test_mechanical_non_referents_match_the_real_junk():
    """Checked against the names an actual store accumulated."""
    from memry.intelligence.entities import non_referent_reason

    junk = ["2019", "2027", "2045", "2026-07-24", "July 2026", "$149",
            "0.33%", "192 GB", "256 GB RAM", "24 GB GPU", "[date]",
            "Sehr geehrte", "Sehr geehrte salutations",
            "https://forms.gle/37VX5yLXdZGFqWY26", "billing@example.com"]
    for name in junk:
        assert non_referent_reason(name), f"should be flagged: {name!r}"

    real = ["Abgeltungssteuer", "RAG", "MSCI World", "Bitcoin", "GmbH",
            "FZulG", "HRB 110232", "World Cup 2026", "Next.js", "SG Ready",
            "Marcus Vandenberg", "2013 master's degree", "6000i dual-split"]
    for name in real:
        assert non_referent_reason(name) is None, f"wrongly flagged: {name!r}"


def test_self_heal_removes_mechanical_junk_but_not_real_concepts(verbatim_store):
    backend = verbatim_store.backend
    memory = backend.insert_memory(
        Memory(content="Tax rules changed in 2027.", user_id="ada")
    )
    junk = backend.insert_entity(Entity(name="2027", normalized="2027",
                                        entity_type="other", user_id="ada"))
    keep = backend.insert_entity(Entity(name="Abgeltungssteuer",
                                        normalized="abgeltungssteuer",
                                        entity_type="concept", user_id="ada"))
    for entity in (junk, keep):
        backend.add_mention(EntityMention(entity_id=entity.id,
                                          memory_id=memory.id, surface=entity.name))

    result = verbatim_store.resolve_entities(user_id="ada")
    assert result["junk_removed"] == 1
    names = {e.name for e in verbatim_store.entities(user_id="ada")}
    assert names == {"Abgeltungssteuer"}
    # the memory the junk entity pointed at is untouched
    assert backend.get_memory(memory.id).invalid_at is None


def test_judged_junk_is_proposed_not_removed(verbatim_store):
    """Style instructions need a reader; the pass must only propose them."""
    import json as _json
    from conftest import FakeLLM

    backend = verbatim_store.backend
    memory = backend.insert_memory(Memory(content="style note", user_id="ada"))
    for name in ("avoid parentheses", "RAG"):
        entity = backend.insert_entity(Entity(name=name, normalized=name.lower(),
                                              entity_type="concept", user_id="ada"))
        backend.add_mention(EntityMention(entity_id=entity.id,
                                          memory_id=memory.id, surface=name))
    verbatim_store.llm = FakeLLM()
    verbatim_store.llm.queue(_json.dumps({"junk": ["avoid parentheses", "Everest"]}))

    result = verbatim_store.entity_junk(user_id="ada", judge=True)
    judged = {j["name"] for j in result["judged"]}
    assert judged == {"avoid parentheses"}  # 'Everest' was never offered
    # nothing was deleted by the review itself
    assert len(verbatim_store.entities(user_id="ada")) == 2

    removed = verbatim_store.remove_entities(
        [j["id"] for j in result["judged"]]
    )
    assert removed == 1
    assert {e.name for e in verbatim_store.entities(user_id="ada")} == {"RAG"}


# ------------------------------------------- two things that share a name
NOORD = "Invoice 2024-117 from Noord Legal B.V."
LEXNOVA = "Invoice 2024-117 from LexNova GmbH"


def test_the_extraction_prompt_names_two_things_that_share_a_name_apart():
    """A save naming two invoices numbered 2024-117 listed one "Invoice
    2024-117", so one entity stood for both and a Noord Legal memory later
    joined it. With this instruction, on the 13 saves of world 3 that mention
    the invoices, facts naming both kept them apart 6 of 6 times (1-2 of 6
    before) and facts naming one carried the sender 23 of 23 times (0 of 14):
    commit 973d0ca; the PhD repo's scenario registry, I-44
    (papers/memry-field-studies/notes/scenario-registry.md, with
    code/identity-replay/samename_test.py and
    data/identity_context/samename_extraction.json)."""
    from conftest import FakeLLM
    from memry.intelligence.extraction import extract_facts

    llm = FakeLLM([facts_response()])
    extract_facts(llm, [{"role": "user", "content": "Paid both invoices 2024-117."}])
    (system, _), = llm.calls
    assert (
        "When the conversation names two or more different things by the same name "
        "(two invoices numbered 2024-117 from different senders, a \"PR #42\" in two "
        "repositories), give each a name of its own: the shared name and what tells "
        "them apart in the conversation (\"Invoice 2024-117 from LexNova GmbH\"). Use "
        "that name for the thing in every fact, also in a fact that names only one."
    ) in " ".join(system.split())


def test_two_things_that_share_a_name_stay_two_entities_through_a_save(store, fake_llm):
    """Named apart by extraction, the two invoices are two entities: the fact
    naming both mentions each, and a fact naming one joins that one."""
    fake_llm.queue(facts_response(
        fact(f"{NOORD} and {LEXNOVA} were both paid in March.",
             entities=[{"name": NOORD, "type": "document"},
                       {"name": LEXNOVA, "type": "document"}]),
        fact(f"{LEXNOVA} was 1,200 euros.", entities=[{"name": LEXNOVA, "type": "document"}]),
    ))
    # the second fact's name is one the store now has: it joins that entity by
    # rule, and the text model is asked only for the coverage audit
    fake_llm.queue(json.dumps({"missing": []}))
    store.add("Paid both invoices numbered 2024-117 in March, Noord Legal's and "
              "LexNova's; LexNova's was 1,200 euros.", user_id="ada")

    entities = {e.name: e for e in store.entities(user_id="ada")}
    assert set(entities) == {NOORD, LEXNOVA}
    assert all(e.entity_type == "document" for e in entities.values())
    memories = {name: {m.content for m in store.backend.entity_memories(e.id)}
                for name, e in entities.items()}
    assert memories == {
        NOORD: {f"{NOORD} and {LEXNOVA} were both paid in March."},
        LEXNOVA: {f"{NOORD} and {LEXNOVA} were both paid in March.",
                  f"{LEXNOVA} was 1,200 euros."},
    }
    assert fake_llm.responses == []


# ------------------------------------------ names a save already knows
class _TextModel(FakeLLM):
    """A text model that answers by what it is asked: the scripted facts for
    an extraction; for a reconcile, a MORE of the memory holding a key of
    ``rewrite`` (its value is the merged text) and a NEW otherwise; "same"
    at 0.9 to an identity question, which it counts; nothing missing to the
    coverage audit."""

    def __init__(self, rewrite: dict[str, str] | None = None) -> None:
        super().__init__()
        self.rewrite = rewrite or {}
        self.identity_calls = 0

    def complete(self, system, user, *, json_schema=None):
        if system == RECONCILE_SYSTEM:
            for key, merged in self.rewrite.items():
                found = re.search(rf"\[(\d+)\] [^\n]*{re.escape(key)}", user)
                if found:
                    return decision("MORE", target=int(found.group(1)), content=merged)
            return decision("NEW")
        if system == IDENTITY_SYSTEM:
            self.identity_calls += 1
            return identity("same", 0.9)
        if system == COVERAGE_SYSTEM:
            return json.dumps({"missing": []})
        return super().complete(system, user, json_schema=json_schema)


class _Judge(NoneDecider):
    """A calibrated judge. Asked what a save does, it adds detail to the
    memory about mugs (MORE); asked about a pair, it keeps what it was shown and answers
    ``same`` and ``different`` (one thing, unless told otherwise)."""

    name = "stub"
    available = True
    calibrated = True
    pair_merge_probability = 0.95

    def __init__(self, same: float = 0.99, different: float = 0.0) -> None:
        self.same, self.different = same, different
        self.pairs: list[str] = []

    def decide(self, state, questions):
        answers = {}
        if "action" in questions:
            answers["action"] = Answer("MORE", {}, 0.95, True)
            found = re.search(r"\[(\d+)\] [^\n]*mugs", state)
            if found:
                answers["target"] = Answer(found.group(1), {}, 0.95, True)
        if "pair" in questions:
            self.pairs.append(state)
            probabilities = {"same": self.same, "different": self.different,
                             "unsure": max(0.0, 1 - self.same - self.different)}
            answers["pair"] = Answer(max(probabilities, key=probabilities.get),
                                     probabilities, 0.9, True)
        return Answers(answers)


SHOP = "The user sells ceramic mugs on Kettlebay"
ADDED = "The user now also sells teapots, glazed by Glazeworks"
REWRITTEN = "The user sells ceramic mugs and teapots on Kettlebay, glazed by Glazeworks"


def _shop(judge):
    """A store where "Kettlebay" is on two memories, one of them about mugs,
    and "Glazeworks" on one."""
    llm = _TextModel({"mugs": REWRITTEN})
    store = MemoryStore(Config(db_path=":memory:"), llm=llm, embedder=HashEmbedder(64),
                        decider=judge)
    backend = store.backend

    def named(name):
        return backend.insert_entity(Entity(name=name, normalized=name.lower(),
                                            entity_type="organization", user_id="ada"))

    def memory(content, entity):
        saved = backend.insert_memory(
            Memory(content=content, user_id="ada", embedding_model=store.embedder.model_id),
            embedding=store.embedder.embed([content])[0])
        backend.add_mention(EntityMention(entity_id=entity.id, memory_id=saved.id,
                                          surface=entity.name))
        return saved

    kettlebay, glazeworks = named("Kettlebay"), named("Glazeworks")
    shop = memory(SHOP, kettlebay)
    memory("Kettlebay charges a listing fee for every item", kettlebay)
    memory("Glazeworks glazes and ships on demand", glazeworks)
    return store, llm, kettlebay, glazeworks, shop


def _rewrite(store, llm, path, shop, extracted, distilled):
    """Rewrite the shop memory by ``path``; the id of the memory holding the
    rewritten text: the shop memory edited in place, or the new memory a
    MORE wrote (a save, or a distilled save), which supersedes it."""
    if path == "manual edit":
        llm.queue(extracted)
        store.update(shop.id, content=REWRITTEN)
        return shop.id
    if path == "reconcile update":
        llm.queue(extracted)
        actions = store.add(ADDED, user_id="ada", infer=False).actions
    else:
        pending = store.add_deferred(ADDED, user_id="ada").actions[0].memory_id
        llm.queue(distilled, extracted)
        actions = store.distill(pending).actions
    [action] = actions
    assert action.event == "UPDATE" and action.memory_id != shop.id
    assert store.get(shop.id).superseded_by == action.memory_id
    return action.memory_id


@pytest.mark.parametrize("judged", [True, False], ids=["calibrated judge", "text model only"])
@pytest.mark.parametrize("path", ["reconcile update", "manual edit", "distillation"])
def test_a_rewritten_memory_keeps_the_names_it_had(path, judged):
    """A memory naming "Kettlebay" is rewritten and still names it: it stays
    on the one "Kettlebay", and nobody is asked about "Kettlebay" and itself.
    Before, the rewritten memory was compared with the entity it already
    belonged to; a memory both sides share is left out of a comparison, so
    nothing was left to compare, and a second "Kettlebay" was made. A name
    new to the memory is still resolved, and on the rewritten text. A save
    that adds detail (MORE) writes the text as a new memory, read as an edit
    of the one it replaces."""
    judge = _Judge() if judged else None
    store, llm, kettlebay, glazeworks, shop = _shop(judge)
    extracted = facts_response(fact(REWRITTEN, entities=["Kettlebay", "Glazeworks"]))
    rewritten = _rewrite(store, llm, path, shop, extracted,
                         facts_response(fact(ADDED, entities=["Glazeworks"])))

    assert store.get(rewritten).content == REWRITTEN
    kettlebays = [e.id for e in store.entities(user_id="ada") if e.normalized == "kettlebay"]
    assert kettlebays == [kettlebay.id]
    linked = {e.id for e in store.backend.entities_of_memory(rewritten)}
    assert linked == {kettlebay.id, glazeworks.id}
    assert store.merge_proposals(user_id="ada") == []
    assert llm.identity_calls == 0
    if judge is not None:
        # asked about "Glazeworks" only, a name the store has: once, the
        # entity first, on the rewritten text
        [state] = judge.pairs
        assert '"Glazeworks"' in state and '"Kettlebay"' not in state
        entity, mention = state.split("ENTITY B")
        assert "Glazeworks glazes and ships on demand" in entity and REWRITTEN in mention
    assert llm.responses == []
    store.close()


@pytest.mark.parametrize("judged", [True, False], ids=["calibrated judge", "text model only"])
@pytest.mark.parametrize("path", ["reconcile update", "manual edit", "distillation"])
def test_a_rewritten_memorys_mentions_keep_what_decided_them(path, judged):
    """The mentions a rewritten memory's new text makes are written as a
    save writes them: with what decided each (``EntityMention.decided``) and
    the type extraction gave the name. Before, they were written with
    neither, so the merge history could not say why a name had joined."""
    judge = _Judge() if judged else None
    store, llm, kettlebay, glazeworks, shop = _shop(judge)
    extracted = facts_response(fact(REWRITTEN, entities=[
        {"name": "Kettlebay", "type": "organization"},
        {"name": "Glazeworks", "type": "organization"}]))
    rewritten = _rewrite(store, llm, path, shop, extracted,
                         facts_response(fact(ADDED, entities=["Glazeworks"])))

    mentions = {m.surface: m for entity in (kettlebay, glazeworks)
                for m in store.backend.entity_mentions(entity.id) if m.memory_id == rewritten}
    assert mentions["Kettlebay"].decided == {"reason": "the memory already names it"}
    joined = mentions["Glazeworks"].decided
    if judged:
        assert joined["reason"] == ("a name the store has: the likeliest of its entities, "
                                    "not said to be different") and joined["same"] == 0.99
    else:
        assert joined == {"reason": "the one entity of this name"}
    assert {m.entity_type for m in mentions.values()} == {"organization"}
    store.close()


def test_without_a_judge_a_known_name_joins_its_entity_and_asks_nothing():
    """Without a calibrated judge a text model's own confidence merges nothing
    (no gate was measured for it), so asking it whether a mention is a known
    entity bought nothing: every fact naming a known person made one more
    entity and an open pair. A name the store has joins its one entity by
    rule, the rule is kept on the mention, and no identity question is asked."""
    llm = _TextModel()
    store = MemoryStore(Config(db_path=":memory:"), llm=llm, embedder=HashEmbedder(64))
    doings = ["booked the venue", "sent the invitations", "chose the menu",
              "hired a photographer", "asked about parking"]
    for session in ("s1", "s2", "s3"):
        llm.queue(facts_response(*(
            fact(f"Livia {doing} for the {session} meeting",
                 entities=[{"name": "Livia", "type": "person"}])
            for doing in doings)))
        store.add(f"Livia's news from {session}", user_id="ada", run_id=session)

    [livia] = store.entities(user_id="ada")
    assert store.backend.count_entity_memories(livia.id) == 3 * len(doings)
    assert llm.identity_calls == 0
    assert store.merge_proposals(user_id="ada") == []
    decided = [m.decided for m in store.backend.entity_mentions(livia.id)]
    assert decided[0] is None  # the mention that made it
    assert decided[1:] == [{"reason": "the one entity of this name"}] * (len(decided) - 1)
    assert llm.responses == []
    store.close()


def test_without_a_judge_namesakes_are_chosen_by_rule():
    """Of two entities that share a name (two people a person kept apart), a
    mention joins the one its conversation already names, and otherwise the
    one with the most memories, whatever type the mention gives it. No
    identity question is asked and no new pair is raised."""
    llm = _TextModel()
    store = MemoryStore(Config(db_path=":memory:"), llm=llm, embedder=HashEmbedder(64))
    backend = store.backend
    partner, colleague = (
        backend.insert_entity(Entity(name="Jonas", normalized="jonas", entity_type="person",
                                     user_id="ada"))
        for _ in range(2))
    for entity, run, texts in (
        (partner, "home", ["Jonas cooks Thai food", "Jonas grows tomatoes",
                           "Jonas is allergic to shellfish"]),
        (colleague, "work", ["Jonas reviewed the phoenix design"]),
    ):
        for text in texts:
            memory = backend.insert_memory(Memory(content=text, user_id="ada", run_id=run))
            backend.add_mention(EntityMention(entity_id=entity.id, memory_id=memory.id,
                                              surface="Jonas"))
    backend.add_proposal(MergeProposal(entity_a=partner.id, entity_b=colleague.id,
                                       user_id="ada", status="rejected"))

    def save(text, run, kind="person"):
        llm.queue(facts_response(fact(text, entities=[{"name": "Jonas", "type": kind}])))
        memory_id = store.add(text, user_id="ada", run_id=run).actions[0].memory_id
        [entity] = backend.entities_of_memory(memory_id)
        [mention] = [m for m in backend.entity_mentions(entity.id) if m.memory_id == memory_id]
        return entity.id, mention.decided

    assert save("Jonas moved the design review to Monday", "work") == (
        colleague.id, {"reason": "of 2 entities of this name, the one this conversation names"})
    assert save("Jonas booked a table for Friday", "weekend") == (
        partner.id, {"reason": "of 2 entities of this name, the one with the most memories"})
    assert save("Jonas is also a note-taking app", "weekend", kind="product") == (
        partner.id, {"reason": "of 2 entities of this name, the one this conversation names"})
    assert backend.get_entity(partner.id).entity_type == "person"
    assert llm.identity_calls == 0
    assert store.merge_proposals(user_id="ada") == []
    store.close()


def test_without_a_judge_a_name_typed_otherwise_in_one_sentence_joins_its_entity():
    """An extractor types a name from one sentence: the shop is a project in
    most, a product in its listing's. That is no evidence of another thing.
    Before, each such mention made a second entity of the name, every week,
    until the weekly merge folded it back. It joins the entity of its name
    by rule, keeping the type it was given on the mention, and the entity's
    type is the one most of its mentions give, its own on a tie."""
    llm = _TextModel()
    store = MemoryStore(Config(db_path=":memory:"), llm=llm, embedder=HashEmbedder(64))
    backend = store.backend

    def save(text, kind):
        llm.queue(facts_response(fact(text, entities=[{"name": "Quirkwear", "type": kind}])))
        store.add(text, user_id="ada")
        [shop] = store.entities(user_id="ada")
        return shop

    save("Quirkwear ships orders every Tuesday", "project")
    shop = save("Quirkwear listing safety text must say wash cold", "product")
    assert shop.entity_type == "project"  # one each: it keeps its own
    assert [(m.entity_type, m.decided) for m in backend.entity_mentions(shop.id)] == [
        ("project", None), ("product", {"reason": "the one entity of this name"})]
    assert save("Quirkwear listing for the owl shirt passed the safety review",
                "product").entity_type == "product"
    assert store.merge_proposals(user_id="ada") == []
    assert llm.identity_calls == 0
    store.close()


@pytest.mark.parametrize("original, drifted, kept", [
    (["project"], ["product"], "project"),  # a tie: the type the store had first
    (["project", "project"], ["product"], "project"),
    (["project"], ["product", "product"], "product"),
])
def test_a_merge_keeps_the_type_most_mentions_give(verbatim_store, original, drifted, kept):
    """A merge of two entities of one name keeps the type most of both
    entities' mentions give, whichever entity the merge keeps; on a tie, the
    type of the one the store had first, since the second entity of a known
    name is the one a sentence typed otherwise. Before, the kept entity's
    type stayed, so a merge that kept the newer one kept the drifted type.
    Undone, each has its own type again."""
    backend = verbatim_store.backend

    def entity(kind, texts, at):
        made = backend.insert_entity(Entity(name="Quirkwear", normalized="quirkwear",
                                            entity_type=kind, user_id="ada", created_at=at))
        for text in texts:
            memory = backend.insert_memory(Memory(content=f"Quirkwear {text}", user_id="ada"))
            backend.add_mention(EntityMention(entity_id=made.id, memory_id=memory.id,
                                              surface="Quirkwear", entity_type=kind))
        return made

    first = entity("project", [f"ships on day {i}" for i in range(len(original))],
                   "2026-01-01T00:00:00+00:00")
    second = entity("product", [f"listing {i} passed" for i in range(len(drifted))],
                    "2026-02-01T00:00:00+00:00")
    assert verbatim_store.merge_entities(second.id, first.id)  # the newer one kept
    assert backend.get_entity(second.id).entity_type == kept
    assert verbatim_store.undo_merge(first.id)["undone"]
    assert (backend.get_entity(first.id).entity_type,
            backend.get_entity(second.id).entity_type) == ("project", "product")
