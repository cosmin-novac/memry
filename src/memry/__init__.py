"""Memry - the open, self-hostable memory layer for AI agents. https://memry.tech"""

from .config import (
    Config,
    DecayConfig,
    EmbeddingConfig,
    LLMConfig,
    RetrievalConfig,
    TagAbstractionConfig,
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
    TopicRelation,
    SearchResult,
    SyntheticTag,
)
from .store import MemoryStore

__version__ = "0.2.27"

__all__ = [
    "MemoryStore",
    "Config",
    "LLMConfig",
    "EmbeddingConfig",
    "RetrievalConfig",
    "DecayConfig",
    "TagAbstractionConfig",
    "Memory",
    "Episode",
    "Entity",
    "EntityMention",
    "MemoryEvent",
    "Scope",
    "Topic",
    "TopicRelation",
    "SearchResult",
    "SyntheticTag",
    "Relation",
    "AddResult",
    "AddAction",
    "CandidateFact",
    "ContextResult",
    "__version__",
]
