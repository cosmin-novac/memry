"""Configuration for Memry.

Resolution order (later wins):
1. built-in defaults
2. config file (JSON) - ``MEMRY_CONFIG`` or ``~/.memry/config.json``
3. environment variables (``MEMRY_*``)
4. explicit kwargs / ``Config(...)`` construction

The servers (``memry serve``, ``memry mcp``) don't start until you set two
models (``require_models``): a text model for extraction
(``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY`` or ``MEMRY_LLM_PROVIDER``) and a
decision model that returns calibrated probabilities
(``MEMRY_DECISION_PROVIDER=jev`` with ``MEMRY_DECISION_API_KEY``). An operator
can send the decision questions to the text model, but only by setting
``MEMRY_DECISION_PROVIDER=llm`` (or ``none``). Memry does not run this check for
a store built directly in code; without keys, Memry then stores memories
verbatim and retrieves them over FTS5 BM25 and deterministic hash embeddings.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

DEFAULT_DIR = Path.home() / ".memry"

LLMProvider = Literal["anthropic", "openai", "ollama", "none"]
EmbeddingProvider = Literal["openai", "ollama", "voyage", "hash", "none"]
# "none" keeps the built-in prompt path; "llm" routes typed questions through
# the configured text model; "jev" uses TypeSafe's System One model.
DecisionProvider = Literal["none", "llm", "jev"]

#: The OpenAI default is gpt-6-luna, at half the price of gpt-5.6-luna. As the
#: extraction model it matched gpt-5.6-luna on every measure (details kept,
#: entities listed, same-name naming, coverage audit; two runs each on 118
#: saves) and listed fewer ordinary nouns as entities. gpt-6-luna has not been
#: measured for re-ranking (``MEASURED_RERANKERS``). Nobody has calibrated a text
#: model's confidence, so Memry never merges entities on it alone (see
#: providers/decisions.py); it sends those questions to the decision model.
DEFAULT_LLM_MODELS: dict[str, str] = {
    "anthropic": "claude-haiku-4-5",
    "openai": "gpt-6-luna",
    "ollama": "llama3.1",
}

DEFAULT_EMBEDDING_MODELS: dict[str, str] = {
    "openai": "text-embedding-3-small",
    "ollama": "nomic-embed-text",
    "voyage": "voyage-3.5-lite",
    "hash": "hash-v1",
}


class LLMConfig(BaseModel):
    provider: LLMProvider = "none"
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    max_tokens: int = 2000
    effort: str = "low"  # Anthropic effort level for extraction calls
    timeout: float = 120.0

    def resolved_model(self) -> str:
        return self.model or DEFAULT_LLM_MODELS.get(self.provider, "")


class DecisionConfig(BaseModel):
    """Provider for Memry's decision questions: entity identity, name screening,
    reconciliation, durability, dates and re-ranking.

    "jev" is TypeSafe's System One model, which returns a calibrated
    distribution over the answers. With "llm" Memry sends the same questions
    to the text model; with "none" it uses the text model's older prompts. An
    operator sets either one only on purpose: nobody has calibrated a text
    model's confidence, so Memry then merges entities only by fixed rules. Unset
    (None) means nobody chose; a server does not start with it
    (``require_models``), and Memry treats it as "none" in a store built in
    code.
    """

    provider: DecisionProvider | None = None
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    timeout: float = 30.0
    #: Override the provider's own automatic-merge gate. Leave unset to use the
    #: value measured for that provider.
    auto_confirm_confidence: float | None = None
    #: Override for the probability from which a calibrated judge merges two
    #: entities (0.95 for Jev). Measured on 156 labelled pairs: at 0.85, 73-74
    #: of 81 true pairs merged against 57-58 at 0.95, no pair of two different
    #: things merged, but 4 pairs that no fact settles did ("PR #92" twice,
    #: "Maria" twice). See evals/identity_resolution_benchmark.py.
    pair_merge_probability: float | None = None
    #: Turn re-ranking on or off. Unset leaves it as the provider has it: on
    #: with Jev, off otherwise. Turning it on only works for a provider that
    #: was measured to beat no re-ranking (Jev, and gpt-5.6-luna or
    #: gpt-5-mini as the text model); for any unmeasured model the setting is
    #: refused. It decides only what "auto" relevance resolves to
    #: (``MemoryStore.relevance_mode``).
    rerank: bool | None = None
    #: How many of the first candidates to judge: the first of the linked
    #: order with seeds, of the text ranking without (``MemoryStore.search``).
    rerank_pool: int = 20


class EmbeddingConfig(BaseModel):
    provider: EmbeddingProvider = "hash"
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    dimensions: int | None = None
    timeout: float = 60.0

    def resolved_model(self) -> str:
        return self.model or DEFAULT_EMBEDDING_MODELS.get(self.provider, "")


#: Link modes that search no longer has, per setting: a config naming one is
#: refused with a message saying so.
REMOVED_RELATIONAL: dict[str, tuple[str, ...]] = {
    "relational_mode": ("typed", "undirected"),
    "relational_fusion": ("rescue", "weighted", "gated", "inherit"),
}


class RetrievalConfig(BaseModel):
    # an assignment is validated too, so a removed mode cannot be set later
    model_config = ConfigDict(validate_assignment=True)

    rrf_k: int = 60
    vector_weight: float = 1.0
    keyword_weight: float = 1.0
    fused_weight: float = 0.70
    recency_weight: float = 0.15
    importance_weight: float = 0.15
    recency_half_life_days: float = 30.0
    candidate_multiplier: int = 3
    reconcile_similarity_limit: int = 5
    #: How search follows links from the entities a query names
    #: (``intelligence/graph_retrieval.py``): "directed" weighs every link by
    #: its kind, its direction and the judge's probability. The only value:
    #: "typed" and "undirected" were removed (``REMOVED_RELATIONAL``).
    relational_mode: Literal["directed"] = "directed"
    #: How many links a search follows from the query's entities.
    relational_depth: int = 1
    #: How the linked memories join the text ranking: "linked" scores every
    #: candidate, and the best of each linked entity's memories, by how well it
    #: states the property asked times how strongly it is about the entity the
    #: query names (``store._search_linked``). The only value: it beat the
    #: typed search on every family of the relative retrieval benchmark.
    relational_fusion: Literal["linked"] = "linked"
    #: "linked" fusion: the power the property similarity is raised to before
    #: it is multiplied by how strongly the memory is about the query's entity.
    #: 1 measured best: 2 and 3 lost the versions whose change is worded as one.
    relational_sharpness: float = 1.0
    #: What judges whether a memory answers. "jev": the decision provider
    #: judges the first ``decision.rerank_pool`` of every search in one call.
    #: "vector": no search is judged; the property vectors order a search
    #: whose question names a hub. "auto" (the default): "jev" where the
    #: decision provider re-ranks (Jev, unless ``decision.rerank`` is false,
    #: or a text model measured to help with ``decision.rerank`` true), else
    #: "vector" (``MemoryStore.relevance_mode``).
    relational_relevance: Literal["auto", "vector", "jev"] = "auto"
    #: "linked" fusion: how many leading numbers of each vector the property
    #: comparison keeps (None: all). The v3 OpenAI models are trained so a
    #: vector cut short still works; property vectors are stored this short.
    property_dimensions: int | None = None
    #: Question keys: with every fact the extractor writes 2 or 3 questions
    #: it answers, kept beside the memory as search keys (``memory_questions``),
    #: and every search matches the words and the meaning of the question
    #: against them too, as two more candidate lists fused with the memory
    #: text's (``retrieval.hybrid_search``). On LoCoMo turns three generated
    #: questions per record raised top-8 success by 11 to 15 points for every
    #: writing model (PhD notes, dreaming-questions). Off until measured on
    #: Memry's facts; ``memry backfill-questions`` writes them for a store
    #: saved before.
    question_keys: bool = False
    #: Entity questions (``intelligence.entity_questions``): the questions the
    #: owner would ask about an entity related to them, by its role ("Where
    #: does my sister work?"). A question that contains no hub's name is then
    #: about the entity whose questions alone contain its role word after "my",
    #: when its best cosine similarity with that entity's questions is at
    #: least ``entity_question_bar``; otherwise it is about the owner, as
    #: before. Off until measured (PhD notes, entity-questions).
    entity_questions: bool = False
    entity_question_bar: float = 0.5
    #: The search log (``search_log``): one row per search, with its time,
    #: namespace, run, the query's text, whether it was about a known entity
    #: (how many seeds), how it was ordered, how many results and how long it
    #: took. Kept ``SEARCH_LOG_DAYS`` days, deleted by upkeep, never in an
    #: export or a snapshot unless asked. ``memry search-stats`` counts it.
    #: Off by default: it keeps what people ask.
    search_log: bool = False
    #: Keys from traffic: a search followed by a save, whose order after the
    #: save puts one of the saved memories in its first 20, gives that memory
    #: the query as a question key (source "traffic"). Needs ``search_log``
    #: and ``question_keys``: with ``question_keys`` off the store warns once
    #: when it opens and keeps this off (alone, the keys from traffic made two
    #: question families worse; PhD notes, traffic-keys).
    traffic_keys: bool = False
    #: A question that needs several memories (a list, a total, a comparison)
    #: has at most this many more judged in one further call, after the first
    #: ``decision.rerank_pool``: the memories filed under the topics the first
    #: ones share, then those nearest the members found (``store._set_pool``).
    #: Measured, the topics' held 85 to 100% of each set within 100
    #: candidates.
    set_pool: int = 80
    #: The memories found are shown with the source turns they rest on that
    #: best match the query, up to this many tokens in all
    #: (``MemoryStore.evidence``); 0 shows none. A memory is a summary, and the
    #: turn it came from keeps what the summary left out.
    evidence_tokens: int = 600

    @field_validator("relational_mode", "relational_fusion", mode="before")
    @classmethod
    def _not_removed(cls, value: Any, info: ValidationInfo) -> Any:
        if value in REMOVED_RELATIONAL.get(info.field_name, ()):
            only = "directed" if info.field_name == "relational_mode" else "linked"
            raise ValueError(
                f"retrieval.{info.field_name} {value!r} was removed; the only value is "
                f"{only!r}")
        return value


class SupersedeConfig(BaseModel):
    """When a change or a correction may replace a stored memory without asking.

    Replacing takes a fact out of use (or leaves it only as history), and it
    rests on a single model judgement. It went wrong in the way that matters:
    a document was misread as saying someone's wife was their mother, and that
    "corrected" the true fact out of the store. So the judgement only acts on
    its own where little is at stake. Everything else keeps both memories in
    use and asks under Upkeep.
    """

    #: A memory at or above this importance is never replaced without asking.
    protect_importance: float = 0.8
    #: Nor is one that this many separate saves have stated: the saves behind
    #: its evidence, not its episodes (``reconcile.saves_of``).
    protect_sources: int = 2
    #: The confidence from which a typed reconcile answer acts, for a
    #: decision provider with no bars measured (``Decider.reconcile_bars``).
    #: The prompt path reports no confidence, so there only the two
    #: protections above apply.
    confidence: float = 0.9
    #: P(it no longer holds), CHANGED and WRONG together, from which a change
    #: replaces a protected memory the decision provider read as a changeable
    #: state (``reconcile.replacement_verdict``): importance raises the bar
    #: instead of blocking. Provisional; ``memry reconcile-queue`` shows the real judge's
    #: answers on a live queue.
    state_confidence: float = 0.8


class DecayConfig(BaseModel):
    enabled: bool = True
    #: Nothing forgets by decay now (the forgetting sweep was retired in
    #: 0.2.44); these settings shape ``decay.effective_importance``, a library
    #: function. The durability pass: the decision provider estimates, per
    #: memory, whether it matters for days, months or years, recorded and acted
    #: on by nothing yet. Off unless set (MEMRY_DURABILITY): the config is the
    #: only way to put it in the upkeep cycle or to run it now; a stored
    #: dashboard switch alone cannot.
    durability: bool = False
    half_life_days: float = 90.0
    floor: float = 0.15  # decayed importance never drops below floor * importance
    # Memory type shapes how fast a memory fades. Episodic memories are dated
    # events that lose relevance as they age; procedural rules ("always do X")
    # should persist; semantic facts sit in between. These multiply the base
    # half-life per type, so type is a real behaviour, not just a label.
    half_life_by_type: dict[str, float] = Field(
        default_factory=lambda: {
            "episodic": 0.5,     # fades about twice as fast
            "semantic": 1.0,
            "procedural": 3.0,   # persists about three times as long
            "working": 0.25,     # short-lived scratch
        }
    )


class AnnConfig(BaseModel):
    """Approximate-nearest-neighbor settings (usearch HNSW sidecar).

    Requires ``pip install memry[ann]``. Below ``min_rows`` exact brute-force
    search is used (it's faster there anyway); above it, the HNSW index
    over-fetches candidates that are then exact-rescored.
    """

    enabled: bool = True
    min_rows: int = 5000
    overfetch: int = 8


class SnapshotConfig(BaseModel):
    """The nightly copy of the database (``memry.snapshot``).

    ``dir`` unset means no snapshots. It must be a directory Memry never writes
    to otherwise: not the data directory, so a fault that damages the live file
    cannot reach the copy. ``at`` is a time of day on the server's clock (the
    container's TZ; UTC unless set). ``host_dir`` is only shown: where ``dir``
    lives on the host when it is a bind mount.

    The offsite copy is optional and off unless ``offsite_url`` and
    ``offsite_bucket`` are set: any S3-compatible store (Cloudflare R2,
    Backblaze B2, Contabo Object Storage). A local copy on the same disk does
    not survive losing that disk; the offsite copy does.
    """

    dir: str | None = None
    at: str = "03:30"
    host_dir: str | None = None
    offsite_url: str | None = None
    offsite_bucket: str | None = None
    offsite_key_id: str | None = None
    offsite_secret: str | None = None
    offsite_region: str = "auto"
    offsite_prefix: str = "memry/"
    #: Whether the copy keeps the search log (``retrieval.search_log``): the
    #: queries people asked. Off: the copy's log is emptied before it is
    #: checked, so neither the copy nor the offsite copy contains it.
    include_search_log: bool = False


class TenantConfig(BaseModel):
    """One tenant of a multi-tenant server: its own API key, its own
    transparently-namespaced memory space."""

    name: str
    api_key: str


class Config(BaseModel):
    db_path: str = str(DEFAULT_DIR / "memry.db")
    default_user_id: str = "default"
    api_key: str | None = None  # admin bearer token for the REST/MCP server
    tenants: list[TenantConfig] = Field(default_factory=list)
    # Accounts live in their own SQLite file (see memry.accounts); None derives
    # it from db_path. Unlike tenants these are created at runtime, not config.
    auth_db_path: str | None = None
    # Public base URL (https://memory.example.com). OAuth needs it: it is the
    # token issuer and the resource identifier clients discover. Unset means no
    # OAuth endpoints, which is the right default for a private single-user box.
    public_url: str | None = None
    # Periodic entity de-duplication: the maintenance autorun re-judges open
    # merge proposals and auto-confirms clear same-entity matches, so duplicate
    # entities collapse over time. Bounded by the number of open proposals.
    dedup_entities: bool = True
    dedup_interval_days: float = 7.0
    llm: LLMConfig = Field(default_factory=LLMConfig)
    decision: DecisionConfig = Field(default_factory=DecisionConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    decay: DecayConfig = Field(default_factory=DecayConfig)
    supersede: SupersedeConfig = Field(default_factory=SupersedeConfig)
    ann: AnnConfig = Field(default_factory=AnnConfig)
    snapshot: SnapshotConfig = Field(default_factory=SnapshotConfig)

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | os.PathLike[str] | None = None, **overrides: Any) -> "Config":
        data: dict[str, Any] = {}

        file_path = path or os.environ.get("MEMRY_CONFIG") or (DEFAULT_DIR / "config.json")
        try:
            raw = Path(file_path).read_text(encoding="utf-8")
            data = _deep_merge(data, json.loads(raw))
        except (FileNotFoundError, OSError):
            pass

        data = _deep_merge(data, _from_env())
        data = _deep_merge(data, overrides)
        cfg = cls.model_validate(data)
        return _autodetect_providers(
            cfg,
            llm_pinned="provider" in data.get("llm", {}),
            embedding_pinned="provider" in data.get("embedding", {}),
        )

    def redacted(self) -> dict[str, Any]:
        d = self.model_dump()
        for section in ("llm", "embedding", "decision"):
            if d[section].get("api_key"):
                d[section]["api_key"] = "***"
        if d.get("api_key"):
            d["api_key"] = "***"
        for tenant in d.get("tenants", []):
            tenant["api_key"] = "***"
        if d["snapshot"].get("offsite_secret"):
            d["snapshot"]["offsite_secret"] = "***"
        return d


def model_requirements(cfg: Config) -> list[str]:
    """The model settings still empty before a server can start: a text model,
    and a decision model an operator chose (Jev, or the text model on
    purpose). The list is empty once both are set."""
    missing = []
    if cfg.llm.provider == "none":
        missing.append(
            "a text model for extraction: set OPENAI_API_KEY (model gpt-6-luna), "
            "ANTHROPIC_API_KEY, or MEMRY_LLM_PROVIDER=ollama"
        )
    if cfg.decision.provider is None:
        missing.append(
            "a decision model for merges and Memry's other decision questions: set "
            "MEMRY_DECISION_PROVIDER=jev and MEMRY_DECISION_API_KEY to a TypeSafe "
            "key (https://typesafe.ai). To send these questions to the text model, "
            "set MEMRY_DECISION_PROVIDER=llm; Memry then merges entities only by fixed "
            "rules, and you confirm the other merges yourself"
        )
    elif cfg.decision.provider == "jev" and not cfg.decision.api_key:
        missing.append("MEMRY_DECISION_API_KEY: the TypeSafe key for MEMRY_DECISION_PROVIDER=jev")
    return missing


def require_models(cfg: Config) -> None:
    """Exit and list the empty model settings when someone starts a server
    without them, and log a warning when an operator sent the decision
    questions to the text model."""
    missing = model_requirements(cfg)
    if missing:
        raise SystemExit(
            "Set a text model and a decision model to run Memry. Not set yet:\n"
            + "\n".join(f"  - {item}" for item in missing)
            + "\nSee docs/self-hosting.md#the-two-models-needed-when-setting-up-the-server."
        )
    if cfg.decision.provider in ("llm", "none"):
        logging.getLogger("memry").warning(
            "MEMRY_DECISION_PROVIDER=%s: Memry sends its decision questions to the text "
            "model. A text model's confidence scores have not been calibrated, so Memry "
            "merges entities only by fixed rules; you confirm the other merges under "
            "Upkeep. With "
            "MEMRY_DECISION_PROVIDER=jev, Memry merges duplicates on its own.",
            cfg.decision.provider,
        )


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _from_env() -> dict[str, Any]:
    e = os.environ.get
    data: dict[str, Any] = {}

    def put(section: str | None, key: str, value: Any) -> None:
        if value is None or value == "":
            return
        if section is None:
            data[key] = value
        else:
            data.setdefault(section, {})[key] = value

    def _int(value: str | None) -> int | None:
        try:
            return int(value) if value else None
        except ValueError:
            return None

    def _bool(value: str | None) -> bool | None:
        if not value:
            return None
        return value.strip().lower() in ("1", "true", "yes", "on")

    def _float(value: str | None) -> float | None:
        try:
            return float(value) if value else None
        except ValueError:
            return None

    put(None, "db_path", e("MEMRY_DB_PATH"))
    put(None, "default_user_id", e("MEMRY_DEFAULT_USER"))
    put(None, "api_key", e("MEMRY_API_KEY"))
    put(None, "auth_db_path", e("MEMRY_AUTH_DB_PATH"))
    put(None, "public_url", e("MEMRY_PUBLIC_URL"))

    put("decay", "durability", _bool(e("MEMRY_DURABILITY")))
    put("snapshot", "dir", e("MEMRY_SNAPSHOT_DIR"))
    put("snapshot", "at", e("MEMRY_SNAPSHOT_AT"))
    put("snapshot", "host_dir", e("MEMRY_SNAPSHOT_HOST_DIR"))
    put("snapshot", "offsite_url", e("MEMRY_SNAPSHOT_OFFSITE_URL"))
    put("snapshot", "offsite_bucket", e("MEMRY_SNAPSHOT_OFFSITE_BUCKET"))
    put("snapshot", "offsite_key_id", e("MEMRY_SNAPSHOT_OFFSITE_KEY_ID"))
    put("snapshot", "offsite_secret", e("MEMRY_SNAPSHOT_OFFSITE_SECRET"))
    put("snapshot", "offsite_region", e("MEMRY_SNAPSHOT_OFFSITE_REGION"))
    put("snapshot", "offsite_prefix", e("MEMRY_SNAPSHOT_OFFSITE_PREFIX"))
    tenants_json = e("MEMRY_TENANTS")
    if tenants_json:
        try:
            data["tenants"] = json.loads(tenants_json)
        except json.JSONDecodeError:
            pass

    put("llm", "provider", e("MEMRY_LLM_PROVIDER"))
    put("llm", "model", e("MEMRY_LLM_MODEL"))
    put("llm", "api_key", e("MEMRY_LLM_API_KEY"))
    put("llm", "base_url", e("MEMRY_LLM_BASE_URL"))
    put("llm", "effort", e("MEMRY_LLM_EFFORT"))

    put("decision", "provider", e("MEMRY_DECISION_PROVIDER"))
    put("decision", "model", e("MEMRY_DECISION_MODEL"))
    put("decision", "api_key", e("MEMRY_DECISION_API_KEY"))
    put("decision", "base_url", e("MEMRY_DECISION_BASE_URL"))
    put("decision", "auto_confirm_confidence", _float(e("MEMRY_DECISION_MERGE_CONFIDENCE")))
    put("decision", "pair_merge_probability", _float(e("MEMRY_DECISION_PAIR_MERGE_PROBABILITY")))
    put("decision", "rerank", _bool(e("MEMRY_DECISION_RERANK")))

    put("supersede", "protect_importance", _float(e("MEMRY_SUPERSEDE_PROTECT_IMPORTANCE")))
    put("supersede", "protect_sources", _int(e("MEMRY_SUPERSEDE_PROTECT_SOURCES")))
    put("supersede", "state_confidence", _float(e("MEMRY_SUPERSEDE_STATE_CONFIDENCE")))
    put("supersede", "confidence", _float(e("MEMRY_SUPERSEDE_CONFIDENCE")))

    put("retrieval", "question_keys", _bool(e("MEMRY_QUESTION_KEYS")))
    put("retrieval", "search_log", _bool(e("MEMRY_SEARCH_LOG")))
    put("retrieval", "traffic_keys", _bool(e("MEMRY_TRAFFIC_KEYS")))

    put("embedding", "provider", e("MEMRY_EMBEDDING_PROVIDER"))
    put("embedding", "model", e("MEMRY_EMBEDDING_MODEL"))
    put("embedding", "api_key", e("MEMRY_EMBEDDING_API_KEY"))
    put("embedding", "base_url", e("MEMRY_EMBEDDING_BASE_URL"))
    put("embedding", "dimensions", _int(e("MEMRY_EMBEDDING_DIMENSIONS")))
    return data


def _autodetect_providers(
    cfg: Config, *, llm_pinned: bool = False, embedding_pinned: bool = False
) -> Config:
    """Upgrade the zero-config defaults when well-known API keys are present.

    A provider set explicitly (config file, env var, or override) is pinned:
    autodetection never replaces it, so e.g. MEMRY_EMBEDDING_PROVIDER=hash
    keeps the local embedder even when OPENAI_API_KEY exists.

    When several keys are present, prefer the one provider that can serve BOTH
    the LLM and the embeddings. Anthropic has no embeddings API, so preferring
    it for the LLM whenever its key exists silently splits a deployment across
    two vendors: Anthropic for extraction, OpenAI for vectors. That is two bills,
    two outage surfaces and two rate limits for no benefit. OpenAI covers both,
    so it wins when its key is available; Anthropic remains the choice when it
    is the only key present.
    """
    env = os.environ
    has_openai = bool(env.get("OPENAI_API_KEY"))
    if not llm_pinned and cfg.llm.provider == "none":
        if has_openai:
            cfg.llm.provider = "openai"
        elif env.get("ANTHROPIC_API_KEY"):
            # The Anthropic SDK is an optional extra. Autodetection must never
            # turn a working keyless install into one that fails to start, so
            # only upgrade when the SDK is importable; otherwise say why not
            # and stay keyless. An explicit MEMRY_LLM_PROVIDER=anthropic still
            # raises in build_llm, because then the user asked for it.
            if importlib.util.find_spec("anthropic") is not None:
                cfg.llm.provider = "anthropic"
            else:
                logging.getLogger("memry").warning(
                    "ANTHROPIC_API_KEY is set but the anthropic SDK is not "
                    "installed; running without an LLM (verbatim memories). "
                    "Install it with: pip install 'memry[anthropic]'"
                )
    if not embedding_pinned and cfg.embedding.provider == "hash":
        if has_openai:
            cfg.embedding.provider = "openai"
        elif env.get("VOYAGE_API_KEY"):
            cfg.embedding.provider = "voyage"
    return cfg
