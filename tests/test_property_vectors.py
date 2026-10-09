"""Property vectors: each memory embedded with the names of its entities, and
of what those belong to, read as "it", so the linked search compares what a
memory says rather than whom it names (``store.refresh_property_vectors``)."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from memry.config import Config
from memry.models import CandidateFact, Entity, EntityMention, MergeProposal, Memory, Scope
from memry.providers.decisions import Answer, Answers, NoneDecider
from memry.providers.embeddings import HashEmbedder
from memry.providers.llm import NoneLLM
from memry.store import MemoryStore, _text_hash

from conftest import FakeLLM, fact, facts_response

ROOT = Path(__file__).resolve().parent.parent

NEITHER_BUT = {"a_kind_of_b": 0.9, "a_part_of_b": 0.0, "b_kind_of_a": 0.0,
               "b_part_of_a": 0.0, "neither": 0.1}


class _Same(NoneDecider):
    """A calibrated judge that finds a pair one thing unless it names one of
    ``apart``."""

    name = "stub"
    available = True
    calibrated = True
    pair_merge_probability = 0.95

    def __init__(self, apart=()) -> None:
        self.apart = apart

    def decide(self, state, questions):
        same = 0.0 if any(f'"{name}"' in state for name in self.apart) else 0.99
        probabilities = {"same": same, "different": 0.99 - same, "unsure": 0.01}
        return Answers({key: Answer(max(probabilities, key=probabilities.get), probabilities,
                                    0.9, True) for key in questions if key == "pair"})


class _Recording(HashEmbedder):
    def __init__(self) -> None:
        super().__init__(32)
        self.texts: list[str] = []

    def embed(self, texts):
        self.texts.extend(texts)
        return super().embed(texts)


@pytest.fixture
def store():
    s = MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(), embedder=_Recording())
    yield s
    s.close()


def _entity(store, name):
    return store.backend.insert_entity(Entity(name=name, normalized=name.lower(), user_id="ada"))


def _memory(store, content, entities):
    memory = store.backend.insert_memory(
        Memory(content=content, user_id="ada", embedding_model=store.embedder.model_id),
        embedding=store.embedder.embed([content])[0])
    for entity in entities:
        store.backend.add_mention(EntityMention(entity_id=entity.id, memory_id=memory.id,
                                                surface=entity.name))
    return memory


def test_a_saved_memory_gets_its_property_vector_once_its_mentions_are_attached(store):
    actions = store._apply_candidates(
        [CandidateFact(content="bildy stores its data in SQLite", entities=["bildy"])],
        Scope(user_id="ada"), [])
    memory_id = actions[0].memory_id
    assert "it stores its data in SQLite" in store.embedder.texts
    assert store.backend.property_vector_hashes([memory_id]) == {
        memory_id: (_text_hash("it stores its data in SQLite"), store._property_label())}


def test_a_memory_whose_text_names_no_entity_keeps_its_ordinary_vector(store):
    note = _memory(store, "The sprint review moved to Friday", [])
    assert store.refresh_property_vectors(user_id="ada") == 0
    assert store.backend.property_vector_hashes([note.id]) == {}


def test_a_new_home_masks_the_things_name_at_the_next_refresh(store):
    """A version's memory often names its product. Once the provider answers
    that Brisk 3 is a version of Brisk, "Brisk" reads "it" there too; a second
    refresh with nothing changed embeds nothing."""
    brisk, v3 = _entity(store, "Brisk"), _entity(store, "Brisk 3")
    memory = _memory(store, "The third release of Brisk added offline maps", [v3])
    assert store.refresh_property_vectors(user_id="ada") == 0  # nothing to mask yet
    store.backend.add_proposal(MergeProposal(
        entity_a=v3.id, entity_b=brisk.id, user_id="ada", confidence=0.7,
        different=0.3, belongs=NEITHER_BUT, compared_step=1))
    assert store.refresh_property_vectors(user_id="ada") == 1
    assert store.embedder.texts[-1] == "The third release of it added offline maps"
    assert store.refresh_property_vectors(user_id="ada") == 0
    assert memory.id in store.backend.property_vectors_of([memory.id], store._property_label())


class _VersionOf(_Same):
    """Finds "Brisk 3" a version of "Brisk", and waits on whether they are one."""

    def decide(self, state, questions):
        if "pair" not in questions:
            return Answers({})
        first = re.findall(r'ENTITY [AB]: "([^"]+)"', state)[0]
        child = "a" if first == "Brisk 3" else "b"
        pair = {"same": 0.3, "different": 0.1, "unsure": 0.6}
        belongs = {"a_kind_of_b": 0.0, "a_part_of_b": 0.0, "b_kind_of_a": 0.0,
                   "b_part_of_a": 0.0, "neither": 0.1}
        belongs[f"{child}_kind_of_{'b' if child == 'a' else 'a'}"] = 0.9
        return Answers({"pair": Answer("unsure", pair, 0.9, True),
                        "belongs": Answer(max(belongs, key=belongs.get), belongs, 0.9, True)})


@pytest.mark.parametrize("answered", ["by a save", "by the weekly pass"])
def test_a_new_home_the_judge_answers_masks_the_things_name_at_once(store, answered):
    """Where the judge's answer that Brisk 3 is a version of Brisk is stored,
    at a save naming Brisk for the first time or in the weekly pass, "Brisk"
    reads "it" in Brisk 3's memory at once rather than at the weekly
    refresh, which then has nothing to embed."""
    v3 = _entity(store, "Brisk 3")
    memory = _memory(store, "The third release of Brisk added offline maps", [v3])
    store.decider = _VersionOf()
    store.refresh_property_vectors(user_id="ada")
    if answered == "by a save":
        store._apply_candidates(
            [CandidateFact(content="Brisk runs on Android", entities=["Brisk"])],
            Scope(user_id="ada"), [])
    else:
        _memory(store, "Brisk runs on Android", [_entity(store, "Brisk")])
        assert _masked(store, memory) is None
        store.resolve_entities(user_id="ada")
    [pair] = store.merge_proposals(user_id="ada")
    assert pair.belongs is not None
    assert _masked(store, memory) == "The third release of it added offline maps"
    assert store.refresh_property_vectors(user_id="ada") == 0


def test_a_vector_from_another_embedding_model_is_not_read_and_is_replaced(store):
    bildy = _entity(store, "bildy")
    memory = _memory(store, "bildy runs on Linux", [bildy])
    store.backend.set_property_vectors({memory.id: [0.0] * 32}, "old-model",
                                       {memory.id: _text_hash("it runs on Linux")})
    assert store.backend.property_vectors_of([memory.id], store._property_label()) == {}
    assert store.refresh_property_vectors(user_id="ada") == 1
    assert memory.id in store.backend.property_vectors_of([memory.id], store._property_label())


def test_deleting_a_memory_deletes_its_property_vector(store):
    bildy = _entity(store, "bildy")
    memory = _memory(store, "bildy runs on Linux", [bildy])
    store.refresh_property_vectors(user_id="ada")
    assert store.backend.property_vector_hashes([memory.id])
    store.backend.delete_memory(memory.id)
    assert store.backend.property_vector_hashes([memory.id]) == {}


def test_a_reset_deletes_every_property_vector(store):
    quillon = _entity(store, "Quillon")
    memory = _memory(store, "Quillon runs on Linux", [quillon])
    store.refresh_property_vectors(user_id="ada")
    assert store.backend.property_vector_hashes([memory.id])
    store.reset()
    assert store.backend.property_vector_hashes([memory.id]) == {}


def test_editing_a_memory_re_embeds_its_property_vector(store):
    """A manual edit masks the new text; an edit that leaves no name to mask
    drops the row, and search reads the ordinary vector, which is the same."""
    quillon = _entity(store, "Quillon")
    memory = _memory(store, "Quillon runs on Linux", [quillon])
    store.refresh_property_vectors(user_id="ada")
    store.update(memory.id, content="Quillon runs on FreeBSD")
    assert store.embedder.texts[-1] == "it runs on FreeBSD"
    assert store.backend.property_vector_hashes([memory.id]) == {
        memory.id: (_text_hash("it runs on FreeBSD"), store._property_label())}
    store.update(memory.id, content="The build server runs on FreeBSD")
    assert store.backend.property_vector_hashes([memory.id]) == {}


def test_a_removed_entity_reads_as_a_name_again_at_once(store):
    """Once "Quillon" is removed its memory names nothing, so the row that
    read it as "it" goes, and the other memory reads it as a name again;
    brought back, the name is masked again. Both at once: left to the weekly
    refresh, search read the removed name as "it" for up to a week."""
    quillon, linux = _entity(store, "Quillon"), _entity(store, "Linux")
    memory = _memory(store, "Quillon runs on Linux", [quillon])
    both = _memory(store, "Quillon moved from Linux to BSD", [quillon, linux])
    store.refresh_property_vectors(user_id="ada")
    assert _masked(store, both) == "it moved from it to BSD"
    assert store.remove_entities([quillon.id]) == 1
    assert store.backend.property_vector_hashes([memory.id]) == {}
    assert _masked(store, both) == "Quillon moved from it to BSD"
    assert store.restore_entities([quillon.id]) == 1
    assert _masked(store, memory) == "it runs on Linux"
    assert _masked(store, both) == "it moved from it to BSD"
    assert store.refresh_property_vectors(user_id="ada") == 0  # nothing left for the week


def _masked(store, memory):
    """The masked text the stored property vector of ``memory`` was made from."""
    stored = store.backend.property_vector_hashes([memory.id]).get(memory.id)
    for text in reversed(store.embedder.texts):
        if stored and (_text_hash(text), store._property_label()) == stored:
            return text
    return None


def _tarnby(store):
    """ "Tarnby Labs" with a memory that also writes the short name, "Tarnby"
    with one of its own, and a version of Tarnby Labs whose memory writes the
    short name too, all refreshed. The version's pair was answered a day
    before, as on a real store: a pair raised later in the same second hid
    that its row lost to a newer one never compared."""
    labs, short = _entity(store, "Tarnby Labs"), _entity(store, "Tarnby")
    v2 = _entity(store, "Tarnby Labs v2")
    store.backend.add_proposal(MergeProposal(
        entity_a=v2.id, entity_b=labs.id, user_id="ada", confidence=0.7,
        different=0.3, belongs=NEITHER_BUT, compared_step=1,
        created_at="2026-01-01T00:00:00+00:00"))
    staff = _memory(store, "Tarnby Labs, called Tarnby by its staff, hired two engineers", [labs])
    office = _memory(store, "Tarnby moved to a bigger office", [short])
    release = _memory(store, "The second release of Tarnby added sync", [v2])
    store.refresh_property_vectors(user_id="ada")
    assert _masked(store, staff) == "it, called Tarnby by its staff, hired two engineers"
    assert _masked(store, release) is None  # nothing of its own to mask yet
    return labs, short, (staff, office, release)


def test_a_merge_masks_the_merged_name_at_once(store):
    """Before, a merged name read as a name in the memories of the entity it
    joined, and of that entity's versions and parts, until the weekly
    refresh."""
    labs, short, (staff, office, release) = _tarnby(store)
    assert store.merge_entities(labs.id, short.id)
    assert _masked(store, staff) == "it, called it by its staff, hired two engineers"
    assert _masked(store, office) == "it moved to a bigger office"
    assert _masked(store, release) == "The second release of it added sync"
    assert store.refresh_property_vectors(user_id="ada") == 0  # nothing left for the week


def test_a_merge_the_judge_decided_masks_the_merged_name_at_once(store):
    labs, short, (staff, _, release) = _tarnby(store)
    store.decider = _Same(apart=["Tarnby Labs v2"])
    store.backend.add_proposal(MergeProposal(
        entity_a=labs.id, entity_b=short.id, user_id="ada", reason="not yet compared"))
    assert store.resolve_entities(user_id="ada")["confirmed"] == 1
    assert _masked(store, staff) == "it, called it by its staff, hired two engineers"
    assert _masked(store, release) == "The second release of it added sync"


def test_a_rename_or_a_new_alias_masks_the_new_name_at_once(store):
    quillon = _entity(store, "Quillon")
    mobile = _memory(store, "Quillon Mobile added an offline mode", [quillon])
    nickname = _memory(store, "Quillon, Q-Mob to its users, dropped the web app", [quillon])
    store.refresh_property_vectors(user_id="ada")
    assert _masked(store, mobile) == "it Mobile added an offline mode"
    store.rename_entity(quillon.id, "Quillon Mobile")
    assert _masked(store, mobile) == "it added an offline mode"
    store.add_entity_alias(quillon.id, "Q-Mob")
    assert _masked(store, nickname) == "it, it to its users, dropped the web app"


def test_a_new_wording_joined_at_save_is_masked_in_the_entitys_other_memories(store):
    """A save that calls "Tarnby Labs" "Tarnby" gives it a new name, which the
    staff memory writes too: it reads "it" there at once. The refresh masks
    every memory of the entity, which is string work, and embeds only what
    changed: the saved memory and the staff memory, not the office one."""
    labs = _entity(store, "Tarnby Labs")
    staff = _memory(store, "Tarnby Labs, called Tarnby by its staff, hired two engineers", [labs])
    office = _memory(store, "Tarnby Labs opened a second office", [labs])
    store.refresh_property_vectors(user_id="ada")
    store.decider = _Same()
    before = len(store.embedder.texts)
    [action] = store._apply_candidates(
        [CandidateFact(content="Tarnby moved to a bigger office", entities=["Tarnby"])],
        Scope(user_id="ada"), [])
    assert [e.id for e in store.backend.entities_of_memory(action.memory_id)] == [labs.id]
    assert _masked(store, staff) == "it, called it by its staff, hired two engineers"
    assert _masked(store, office) == "it opened a second office"
    masked = [text for text in store.embedder.texts[before:]
              if text != "Tarnby moved to a bigger office"]  # the memory's own vector
    assert sorted(masked) == [
        "it moved to a bigger office", "it, called it by its staff, hired two engineers"]


def test_a_new_name_is_embedded_only_where_a_memory_writes_it(store):
    quillon = _entity(store, "Quillon")
    for text in ("Quillon runs on Linux", "Quillon, Q-Mob to its users, dropped the web app",
                 "Quillon added an offline mode"):
        _memory(store, text, [quillon])
    store.refresh_property_vectors(user_id="ada")
    before = len(store.embedder.texts)
    store.add_entity_alias(quillon.id, "Q-Mob")
    assert store.embedder.texts[before:] == ["it, it to its users, dropped the web app"]


def test_a_name_that_is_also_a_common_word_reads_it_in_its_own_memories(store):
    """An accepted limit. Masking matches a name as a whole word in any case,
    so the verb "go" reads "it" in the memories of the language "Go", which a
    mention wrote as "go". No rule without a list of common words tells the
    verb from the name there; the memories of other entities keep the word."""
    go, team = _entity(store, "Go"), _entity(store, "Harrow team")
    code = _memory(store, "The services are written in Go and the team wants to go faster", [go])
    store.backend.add_mention(EntityMention(entity_id=go.id, memory_id=code.id, surface="go"))
    other = _memory(store, "The Harrow team will go paperless in May", [team])
    store.refresh_property_vectors(user_id="ada")
    assert _masked(store, code) == "The services are written in it and the team wants to it faster"
    assert _masked(store, other) == "The it will go paperless in May"


def test_the_weekly_upkeep_refreshes_them(store):
    bildy = _entity(store, "bildy")
    _memory(store, "bildy runs on Linux", [bildy])
    ran = store.run_upkeep_cycle(user_id="ada")
    assert ran.get("property_vectors") == {"embedded": 1}


def test_property_vectors_can_be_stored_short(store):
    """With ``property_dimensions`` a property vector keeps its first numbers,
    at length 1; a vector stored at another length is not read and is
    re-embedded."""
    bildy = _entity(store, "bildy")
    memory = _memory(store, "bildy runs on Linux", [bildy])
    store.refresh_property_vectors(user_id="ada")
    store.config.retrieval.property_dimensions = 8
    assert store.backend.property_vectors_of([memory.id], store._property_label()) == {}
    assert store.refresh_property_vectors(user_id="ada") == 1
    vector = store.backend.property_vectors_of([memory.id], store._property_label())[memory.id]
    assert vector.shape == (8,)
    assert float((vector ** 2).sum()) == pytest.approx(1.0)


class _NamesWeigh(HashEmbedder):
    """Word counts in which a name's words weigh ten times any other word:
    the vector of "bildy v3 added offline mode" is mostly "bildy v3", as in
    an embedder that follows names. "added" counts as "add"."""

    NAMES = {"bildy", "v3"}

    def __init__(self) -> None:
        super().__init__(997)

    def _embed_one(self, text: str) -> list[float]:
        import re
        import zlib

        vector = [0.0] * self.dimensions
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            word = "add" if word == "added" else word
            vector[zlib.crc32(word.encode()) % self.dimensions] += (
                10.0 if word in self.NAMES else 1.0)
        return vector


def test_property_vectors_keep_a_versions_answer_among_the_first_twenty():
    """Why property vectors exist. "What did bildy v3 add?" is read "What did
    it add?"; its answer is v3's own memory. Compared by its ordinary vector,
    the answer is mostly its name and barely near the question (0.03 x 1.0),
    and 25 notes about adding things, linked to nothing, pass it
    (0.16 x 0.3): it falls out of the first 20, which the decision provider
    reads. With the names read "it" its property vector is the property
    alone, and it comes first."""
    store = MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(), embedder=_NamesWeigh())
    try:
        bildy, v3 = _entity(store, "bildy"), _entity(store, "bildy v3")
        answer = _memory(store, "bildy v3 added offline mode for travellers on long flights",
                         [v3])
        _memory(store, "bildy v3 was released in May", [v3])
        _memory(store, "bildy runs on Linux", [bildy])
        _memory(store, "bildy stores its data in SQLite", [bildy])
        for i in range(25):
            _memory(store, f"Remember to add note {i} to the list", [])
        store.backend.add_proposal(MergeProposal(
            entity_a=v3.id, entity_b=bildy.id, user_id="ada", confidence=0.3,
            belongs=NEITHER_BUT, compared_step=1))
        question = "What did bildy v3 add?"
        ordinary = [r.memory.id for r in store.search(question, user_id="ada", limit=20)]
        assert len(ordinary) == 20 and answer.id not in ordinary
        assert store.refresh_property_vectors(user_id="ada") == 4
        found = store.search(question, user_id="ada", limit=20)
        assert found[0].memory.id == answer.id and found[0].signals["about"] == 1.0
    finally:
        store.close()


def test_a_save_asks_no_text_model_for_its_property_vector():
    """The property vector is the saved text with the names read "it":
    string work and one embedding, no call to a text model. (A "says" written
    by a text model instead, the statement with its subject taken out, cost
    one call a memory on every save and left the design for it.) The save's
    text-model calls are the extraction and the coverage audit alone."""
    llm = FakeLLM([facts_response(fact("bildy stores its data in SQLite", entities=["bildy"])),
                   json.dumps({"missing": []})])
    store = MemoryStore(Config(db_path=":memory:"), llm=llm, embedder=_Recording())
    try:
        refresh = store.refresh_property_vectors
        during = {}

        def counted(**kwargs):
            calls, texts = len(llm.calls), len(store.embedder.texts)
            embedded = refresh(**kwargs)
            during.update(calls=len(llm.calls) - calls, texts=store.embedder.texts[texts:])
            return embedded

        store.refresh_property_vectors = counted
        store.add("bildy keeps everything in SQLite", user_id="ada")
        assert during == {"calls": 0, "texts": ["it stores its data in SQLite"]}
        assert len(llm.calls) == 2 and not llm.responses
    finally:
        store.close()


@pytest.mark.parametrize("dimensions", [None, 16])
def test_the_benchmarks_says_path_stores_the_masked_texts_as_memry_does(dimensions):
    """``relative_retrieval_benchmark.store_says`` stores a text given for
    each memory as its property vector. Given the masked texts Memry computes,
    it stores what ``refresh_property_vectors`` stores, row for row: the same
    memories, hashes and vectors, cut alike."""
    sys.path.insert(0, str(ROOT))
    from evals import relative_retrieval_benchmark as bench

    world = bench.build_world_dense(250)
    answers = json.loads((ROOT / "evals" / "datasets" / "belongs_answers.json").read_text())
    masked, ids = bench.build_store(world, HashEmbedder(64), "oracle", answers["answers"],
                                    property_dimensions=dimensions)
    entities = {mid: [e.id for e in masked.backend.entities_of_memory(mid, kind="named")]
                for mid in ids}
    texts = masked._masked_texts({mid: m["text"] for mid, m in zip(ids, world["memories"])},
                                 entities)
    says, _ = bench.build_store(world, HashEmbedder(64), "oracle", answers["answers"],
                                property_dimensions=dimensions,
                                says={str(k): texts[mid] for k, mid in enumerate(ids)})
    try:
        rows = masked.backend.property_vector_hashes(ids)
        assert rows and says.backend.property_vector_hashes(ids) == rows
        label = masked._property_label()
        ours, theirs = (s.backend.property_vectors_of(ids, label) for s in (masked, says))
        assert ours.keys() == theirs.keys() == rows.keys()
        assert all(np.array_equal(ours[mid], theirs[mid]) for mid in ours)
    finally:
        masked.close()
        says.close()


def test_the_benchmark_switches_the_vectors_the_linked_search_reads(monkeypatch, tmp_path, capsys):
    """``relative_retrieval_benchmark --vectors ordinary`` builds its stores
    with no property vectors, so what the linked search reads for every memory
    (``MemoryStore._property_vectors``) is its ordinary vector, names as
    written; ``--vectors property`` (the default) reads the masked ones.
    ``--property-dimensions`` reaches ``retrieval.property_dimensions``. Run
    offline: hash vectors, a small world, one family."""
    sys.path.insert(0, str(ROOT))
    from evals import relative_retrieval_benchmark as bench

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("TMPDIR", str(tmp_path))  # the embedder's cache
    read: dict[str, tuple] = {}
    score = bench.score

    def inspected(store, memory_ids, queries, mode):
        linked = store._property_vectors(memory_ids)
        ordinary = store.backend.vectors_of(memory_ids, store.embedder.model_id)
        read[store.config.retrieval.property_dimensions or "all"] = (
            len(store.backend.property_vector_hashes(memory_ids)),
            sum(np.array_equal(linked[mid], ordinary[mid]) for mid in memory_ids),
            len(memory_ids))
        return score(store, memory_ids, queries, mode)

    monkeypatch.setattr(bench, "score", inspected)
    common = ["bench", "--sizes", "200", "--links", "oracle", "--modes", "linked k1",
              "--families", "inherit"]
    monkeypatch.setattr(sys, "argv", common)
    bench.main()
    monkeypatch.setattr(sys, "argv", common + ["--vectors", "ordinary",
                                               "--property-dimensions", "16"])
    bench.main()
    (rows, same, n), (none, every, m) = read["all"], read[16]
    assert rows > 0 and same < n  # the masked memories are read by their property vectors
    assert none == 0 and every == m  # every memory by its ordinary vector
    assert "ordinary vectors, dimensions 16" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", common + ["--vectors", "ordinary", "--says", "x.json"])
    with pytest.raises(SystemExit):
        bench.main()
