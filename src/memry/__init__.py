"""Memry - the open, self-hostable memory layer for AI agents. https://memry.tech"""

from .config import (
    Config,
    DecayConfig,
    EmbeddingConfig,
    LLMConfig,
    RetrievalConfig,
)
from .models import (
    AddAction,
    AddResult,
    CandidateFact,
    ContextResult,
    Episode,
    Entity,
    EntityMention,
    Memory,
    MemoryEvent,
    Relation,
    Scope,
    Topic,
    SearchResult,
)
from .store import MemoryStore

__version__ = "0.2.42"

__all__ = [
    "MemoryStore",
    "Config",
    "LLMConfig",
    "EmbeddingConfig",
    "RetrievalConfig",
    "DecayConfig",
    "Memory",
    "Episode",
    "Entity",
    "EntityMention",
    "MemoryEvent",
    "Scope",
    "Topic",
    "SearchResult",
    "Relation",
    "AddResult",
    "AddAction",
    "CandidateFact",
    "ContextResult",
    "__version__",
]
