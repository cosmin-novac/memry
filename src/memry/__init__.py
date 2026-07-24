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
    Memory,
    MemoryEvent,
    Relation,
    Scope,
    SearchResult,
    SyntheticTag,
)
from .store import MemoryStore

__version__ = "0.2.12"

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
    "MemoryEvent",
    "Scope",
    "SearchResult",
    "SyntheticTag",
    "Relation",
    "AddResult",
    "AddAction",
    "CandidateFact",
    "ContextResult",
    "__version__",
]
