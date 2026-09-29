"""Property vectors: each memory embedded with the names of its entities, and
of what those belong to, read as "it", so the linked search compares what a
memory says rather than whom it names (``store.refresh_property_vectors``)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from memry.config import Config
from memry.models import CandidateFact, Entity, EntityMention, MergeProposal, Memory, Scope
from memry.providers.embeddings import HashEmbedder
from memry.providers.llm import NoneLLM
from memry.store import MemoryStore, _text_hash

from conftest import FakeLLM, fact, facts_response

ROOT = Path(__file__).resolve().parent.parent

NEITHER_BUT = {"a_kind_of_b": 0.9, "a_part_of_b": 0.0, "b_kind_of_a": 0.0,
               "b_part_of_a": 0.0, "neither": 0.1}


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
    that bildy v3 is a version of bildy, "bildy" reads "it" there too; a second
    refresh with nothing changed embeds nothing."""
    bildy, v3 = _entity(store, "bildy"), _entity(store, "bildy v3")
    memory = _memory(store, "The third release of bildy added offline mode", [v3])
    assert store.refresh_property_vectors(user_id="ada") == 0  # nothing to mask yet
    store.backend.add_proposal(MergeProposal(
        entity_a=v3.id, entity_b=bildy.id, user_id="ada", confidence=0.7,
        different=0.3, belongs=NEITHER_BUT, compared_step=1))
    assert store.refresh_property_vectors(user_id="ada") == 1
    assert store.embedder.texts[-1] == "The third release of it added offline mode"
    assert store.refresh_property_vectors(user_id="ada") == 0
    assert memory.id in store.backend.property_vectors_of([memory.id], store._property_label())


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
