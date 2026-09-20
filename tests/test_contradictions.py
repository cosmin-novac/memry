"""A contradiction may not quietly replace what the store has most reason to believe.

The case these pin happened on a real store: a rental form was misread as
saying a man's wife was his mother, reconciliation took that for a correction,
and the true memory was taken out of use with nobody asked and no way back.
"""

from __future__ import annotations

import pytest

from conftest import decision, fact, facts_response
from memry.config import SupersedeConfig
from memry.intelligence.reconcile import CONFLICT_KEY, held_back
from memry.models import Memory, clean_tags


def _mother_and_the_misreading(store, fake_llm):
    fake_llm.queue(facts_response(fact("Violeta is Cosmin's mother", importance=0.9)))
    store.add("My mother is Violeta", user_id="cos")
    old = store.get_all(user_id="cos")[0]
    fake_llm.queue(
        facts_response(fact("Raluca is Cosmin's mother", importance=0.9)),
        decision("DELETE", target=0, reason="replace outdated relationship"),
    )
    result = store.add("rental form: applicant Raluca, for his mother", user_id="cos")
    return old, result


def test_an_important_memory_is_not_replaced_without_asking(store, fake_llm):
    old, result = _mother_and_the_misreading(store, fake_llm)

    action = result.actions[0]
    assert action.event == "ADD"
    assert action.conflicts_with == old.id
    assert store.get(old.id).invalid_at is None, "the true memory stays in use"
    new = store.get(action.memory_id)
    assert new.metadata[CONFLICT_KEY]["with"] == old.id
    assert "important" in new.metadata[CONFLICT_KEY]["held"]

    assert store.upkeep_count(user_id="cos") == 1
    row = next(r for r in store.upkeep_queue(user_id="cos") if r["kind"] == "conflict")
    assert row["id"] == new.id
    assert row["replaces"] == ["Violeta is Cosmin's mother"]
    assert (row["accept"], row["decline"], row["other"]) == (
        "the new one is right", "the old one is right", "both are true",
    )


def test_saying_the_old_one_is_right_forgets_the_new_one(store, fake_llm):
    old, result = _mother_and_the_misreading(store, fake_llm)
    new_id = result.actions[0].memory_id

    assert store.decide_upkeep("conflict", new_id, "decline", user_id="cos")

    assert [m.content for m in store.get_all(user_id="cos")] == [old.content]
    forgotten = store.forgotten(user_id="cos")
    assert [row["memory"].id for row in forgotten] == [new_id], "and it can come back"
    assert store.upkeep_count(user_id="cos") == 0


def test_saying_the_new_one_is_right_replaces_the_old_one(store, fake_llm):
    old, result = _mother_and_the_misreading(store, fake_llm)
    new_id = result.actions[0].memory_id

    assert store.decide_upkeep("conflict", new_id, "accept", user_id="cos")

    assert store.get(old.id).superseded_by == new_id
    assert CONFLICT_KEY not in store.get(new_id).metadata
    event = store.history(old.id)[-1]
    assert (event.event, event.actor) == ("SUPERSEDE", "user")


def test_both_can_be_true(store, fake_llm):
    old, result = _mother_and_the_misreading(store, fake_llm)
    new_id = result.actions[0].memory_id

    assert store.decide_upkeep("conflict", new_id, "other", user_id="cos")

    assert {m.id for m in store.get_all(user_id="cos")} == {old.id, new_id}
    assert CONFLICT_KEY not in store.get(new_id).metadata
    assert store.upkeep_queue(user_id="cos") == []
    # "other" means nothing for the rows that only have a yes and a no
    assert not store.decide_upkeep("consolidation", "nope", "other", user_id="cos")


def test_a_conflict_settled_by_deleting_one_side_leaves_the_queue(store, fake_llm):
    _, result = _mother_and_the_misreading(store, fake_llm)
    store.delete(result.actions[0].memory_id)
    assert store.upkeep_queue(user_id="cos") == []
    assert store.upkeep_count(user_id="cos") == 0


def _moved_cities(store, fake_llm):
    fake_llm.queue(facts_response(fact("User lives in Munich")))
    store.add("I live in Munich", user_id="ada")
    old = store.get_all(user_id="ada")[0]
    fake_llm.queue(
        facts_response(fact("User lives in Amsterdam")),
        decision("DELETE", target=0, reason="moved cities"),
    )
    result = store.add("I moved to Amsterdam", user_id="ada")
    assert result.actions[0].event == "DELETE", "little at stake, so it goes ahead"
    return old, result.actions[0].memory_id


def test_a_replacement_is_listed_and_can_be_undone(store, fake_llm):
    old, new_id = _moved_cities(store, fake_llm)

    rows = store.replaced(user_id="ada")
    assert [(r["memory"].id, r["replacement"].id) for r in rows] == [(old.id, new_id)]
    assert rows[0]["reason"] == "moved cities"

    assert store.undo_replacement(old.id)
    assert [m.id for m in store.get_all(user_id="ada")] == [old.id]
    assert store.get(old.id).superseded_by is None
    assert [r["memory"].id for r in store.forgotten(user_id="ada")] == [new_id]
    assert store.replaced(user_id="ada") == []


def test_undoing_can_keep_both(store, fake_llm):
    old, new_id = _moved_cities(store, fake_llm)
    assert store.undo_replacement(old.id, keep_new=True)
    assert {m.id for m in store.get_all(user_id="ada")} == {old.id, new_id}


def test_a_merge_of_duplicates_is_not_a_replacement_to_undo(verbatim_store):
    store = verbatim_store
    store.add("likes tea", user_id="ada", infer=False)
    store.add("enjoys tea", user_id="ada", infer=False)
    memories = store.get_all(user_id="ada")
    store._merge_group(memories, "likes tea", user_id="ada")

    assert store.replaced(user_id="ada") == []
    with pytest.raises(ValueError):
        store.undo_replacement(memories[0].id)


def test_what_holds_a_replacement_back():
    cfg = SupersedeConfig()
    passing = Memory(content="x", importance=0.5, source_episode_ids=["a"])
    assert held_back(passing, {}, cfg) is None
    assert held_back(passing, {"confidence": 0.95}, cfg) is None
    assert "0.70 sure" in held_back(passing, {"confidence": 0.7}, cfg)
    assert "important" in held_back(
        Memory(content="x", importance=0.8), {"confidence": 0.99}, cfg
    )
    assert "2 separate saves" in held_back(
        Memory(content="x", importance=0.5, source_episode_ids=["a", "b"]), {}, cfg
    )


def test_tags_hold_one_subject_each():
    # the three broken tags found on a real store, and what they should have been
    assert clean_tags(["Steuernummer (TIN, Köln Vingst)"]) == ["Steuernummer"]
    assert clean_tags(["steuernummer (tin", "köln vingst)"]) == [
        "steuernummer", "köln vingst",
    ]
    assert clean_tags(["steuernummer (tin, stooq, system, tax, yfinance)", "person"]) == [
        "steuernummer", "person",
    ]
    assert clean_tags("tax, identity; Tax") == ["tax", "identity"]
    assert clean_tags(["x" * 65, "writing preferences", "", None]) == [
        "writing preferences"
    ]
    assert clean_tags(None) == []


def test_tags_are_cleaned_wherever_they_come_in(store, fake_llm, verbatim_store):
    fake_llm.queue(facts_response(
        fact("Fundation Steuernummer 218/5713/1681",
             categories=["Steuernummer (TIN, Köln Vingst)"])
    ))
    store.add("our tax number", user_id="ada")
    assert store.get_all(user_id="ada")[0].categories == ["steuernummer"]

    added = verbatim_store.add(
        "kept as is", user_id="ada", infer=False, categories=["home (narrow", "tax"]
    )
    memory_id = added.actions[0].memory_id
    assert verbatim_store.get(memory_id).categories == ["home", "tax"]
    updated = verbatim_store.update(memory_id, categories=["a, b", "c)"])
    assert updated.categories == ["a", "b", "c"]


def test_the_model_is_offered_tags_as_a_list_not_a_sentence(store, fake_llm):
    fake_llm.queue(facts_response(fact("one", categories=["tax", "writing preferences"])))
    store.add("one", user_id="ada")
    fake_llm.queue(facts_response(fact("two")), decision("ADD"))
    store.add("two", user_id="ada")
    offers = [user for _, user in fake_llm.calls if "Tags this user" in user]
    assert offers and '["tax", "writing preferences"]' in offers[-1]
