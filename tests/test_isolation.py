"""The ownership gate on id-addressed store operations.

Namespace filtering on list/search is not enough on its own: memory ids are
opaque but they travel (logs, tool output, an agent that saw them once), so
every by-id operation has to re-check who is asking. These tests pin that
behaviour for each such method, since a single gap is a cross-account read.
"""

from __future__ import annotations

import pytest

from memry.intelligence.entities import resolve_mentions
from memry.models import Scope
from memry.principal import ADMIN, Principal

ACME = Principal(name="acme")
GLOBEX = Principal(name="globex")


@pytest.fixture
def seeded(verbatim_store):
    """One memory owned by acme, one by globex."""
    store = verbatim_store
    store.add("acme roadmap", user_id=ACME.namespace(None), infer=False)
    store.add("globex pricing", user_id=GLOBEX.namespace(None), infer=False)
    acme_memory = next(
        m for m in store.get_all(limit=50) if m.content == "acme roadmap"
    )
    return store, acme_memory


def test_namespaces_are_derived_not_supplied():
    assert ACME.namespace(None) == "acme::default"
    assert ACME.namespace("u1") == "acme::u1"
    # the dangerous case: naming another principal's space just nests it
    assert ACME.namespace("globex::default") == "acme::globex::default"
    # admin is unconfined, including the "every namespace" None
    assert ADMIN.namespace(None) is None
    assert ADMIN.namespace("u1") == "u1"
    named_admin = Principal(
        name="owner", admin=True, fixed_user="default"
    )
    assert named_admin.is_admin is True
    assert named_admin.prefix == "default"
    assert named_admin.namespace(None) == "default"
    assert named_admin.namespace("someone-else") == "default"
    assert named_admin.owns("default") is True
    assert named_admin.owns("owner::default") is False


def test_get_is_gated(seeded):
    store, victim = seeded
    assert store.get(victim.id, owner_prefix=GLOBEX.prefix) is None
    assert store.get(victim.id, owner_prefix=ACME.prefix).content == "acme roadmap"
    assert store.get(victim.id, owner_prefix=ADMIN.prefix) is not None


def test_update_is_gated(seeded):
    store, victim = seeded
    assert store.update(victim.id, content="hijacked", owner_prefix=GLOBEX.prefix) is None
    assert store.get(victim.id).content == "acme roadmap"
    assert store.update(victim.id, content="revised", owner_prefix=ACME.prefix) is not None
    assert store.get(victim.id).content == "revised"


def test_delete_is_gated(seeded):
    store, victim = seeded
    assert store.delete(victim.id, owner_prefix=GLOBEX.prefix) is False
    assert store.get(victim.id).invalid_at is None
    assert store.delete(victim.id, owner_prefix=ACME.prefix) is True


def test_history_is_gated(seeded):
    store, victim = seeded
    assert store.history(victim.id, owner_prefix=GLOBEX.prefix) == []
    assert store.history(victim.id, owner_prefix=ACME.prefix)


def test_history_for_admin_survives_a_hard_delete(seeded):
    """Admin keeps the audit trail after the row is gone; the gate must not
    quietly turn that into an empty list."""
    store, victim = seeded
    store.delete(victim.id, hard=True)
    assert store.get(victim.id) is None
    assert [e.event for e in store.history(victim.id)] == ["ADD", "DELETE"]


def test_distill_is_gated(seeded):
    store, victim = seeded
    # gate is checked before the "no LLM configured" error, so a foreign caller
    # cannot even probe for existence
    assert store.distill(victim.id, owner_prefix=GLOBEX.prefix) is None
    with pytest.raises(ValueError):
        store.distill(victim.id, owner_prefix=ACME.prefix)


def _seed_entity(store, user_id: str, surface: str) -> str:
    store.add(f"{surface} fact", user_id=user_id, infer=False)
    memory = next(m for m in store.get_all(user_id=user_id) if surface in m.content)
    resolve_mentions(
        backend=store.backend,
        llm=store.llm,
        scope=Scope(user_id=user_id),
        memory_id=memory.id,
        memory_content=memory.content,
        surfaces=[surface],
    )
    return next(
        e for e in store.entities(user_id=user_id, limit=100) if e.name == surface
    ).id


def test_entity_is_gated(verbatim_store):
    store = verbatim_store
    entity_id = _seed_entity(store, ACME.namespace(None), "Jonas")
    assert store.entity(entity_id, owner_prefix=GLOBEX.prefix) is None
    assert store.entity(entity_id, owner_prefix=ACME.prefix)["entity"].name == "Jonas"


def test_merge_entities_is_gated(verbatim_store):
    store = verbatim_store
    keep = _seed_entity(store, ACME.namespace(None), "Jonas")
    other = _seed_entity(store, GLOBEX.namespace(None), "Mira")
    # a foreign caller cannot merge, and neither can a caller who owns only one
    # side of the pair
    assert store.merge_entities(keep, other, owner_prefix=GLOBEX.prefix) is False
    assert store.merge_entities(keep, other, owner_prefix=ACME.prefix) is False
    assert store.backend.get_entity(other).merged_into is None


def test_merge_proposals_are_gated(verbatim_store):
    store = verbatim_store
    from memry.models import MergeProposal

    a = _seed_entity(store, ACME.namespace(None), "Jonas")
    b = _seed_entity(store, ACME.namespace(None), "Jon")
    proposal = MergeProposal(entity_a=a, entity_b=b, user_id=ACME.namespace(None))
    store.backend.add_proposal(proposal)

    assert store.confirm_merge(proposal.id, owner_prefix=GLOBEX.prefix) is False
    assert store.reject_merge(proposal.id, owner_prefix=GLOBEX.prefix) is False
    assert store.confirm_merge(proposal.id, owner_prefix=ACME.prefix) is True
