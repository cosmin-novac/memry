"""Memry - the open, self-hostable memory layer for AI agents. https://memry.tech"""

from .config import Config, DecayConfig, EmbeddingConfig, LLMConfig, RetrievalConfig
from .models import (
    AddAction,
    AddResult,
    CandidateFact,
    ContextResult,
    Episode,
    Memory,
    MemoryEvent,
    Scope,
    SearchResult,
)
from .store import MemoryStore

__version__ = "0.2.5"

__all__ = [
    "MemoryStore",
    "Config",
    "LLMConfig",
    "EmbeddingConfig",
    "RetrievalConfig",
    "DecayConfig",
    "Memory",
    "Episode",
    "MemoryEvent",
    "Scope",
    "SearchResult",
    "AddResult",
    "AddAction",
    "CandidateFact",
    "ContextResult",
    "__version__",
]
