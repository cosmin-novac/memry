from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from memry.config import Config  # noqa: E402
from memry.providers.embeddings import HashEmbedder  # noqa: E402
from memry.providers.llm import LLM  # noqa: E402
from memry.store import MemoryStore  # noqa: E402


class FakeLLM(LLM):
    """Scripted LLM: pops queued responses; records prompts for assertions."""

    name = "fake"
    available = True

    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = list(responses or [])
        self.calls: list[tuple[str, str]] = []

    def queue(self, *responses: str) -> None:
        self.responses.extend(responses)

    def complete(self, system: str, user: str, *, json_schema=None) -> str:
        self.calls.append((system, user))
        if not self.responses:
            raise AssertionError("FakeLLM ran out of scripted responses")
        return self.responses.pop(0)


def facts_response(*facts: dict) -> str:
    return json.dumps({"facts": list(facts)})


def fact(content: str, type: str = "semantic", importance: float = 0.7, **kw) -> dict:
    return {
        "content": content,
        "type": type,
        "importance": importance,
        "categories": kw.get("categories", []),
        "entities": kw.get("entities", []),
    }


def decision(action: str, target=None, content=None, reason="test") -> str:
    return json.dumps({"action": action, "target": target, "content": content, "reason": reason})


@pytest.fixture
def config() -> Config:
    return Config(db_path=":memory:")


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def store(config, fake_llm) -> MemoryStore:
    s = MemoryStore(config, llm=fake_llm, embedder=HashEmbedder(128))
    yield s
    s.close()


@pytest.fixture
def verbatim_store(config) -> MemoryStore:
    """Store with no LLM (zero-key mode) and hash embeddings."""
    from memry.providers.llm import NoneLLM

    s = MemoryStore(config, llm=NoneLLM(), embedder=HashEmbedder(128))
    yield s
    s.close()
