"""Optional HNSW sidecar index (usearch) for approximate nearest-neighbor
search at scale.

Install with ``pip install memry[ann]``. Without it - or below the configured
row threshold - LocalBackend uses exact brute-force cosine, which is faster at
small sizes anyway. The sidecar is a cache, never the source of truth: it can
be deleted at any time and is rebuilt from SQLite (``memry reindex`` or lazily
on model mismatch). ANN results are always exact-rescored before ranking.

Keys are stable integers from the ``ann_keys`` table (SQLite rowids can be
renumbered by VACUUM, so they are not used directly).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

try:  # pragma: no cover - environment dependent
    from usearch.index import Index as _UsearchIndex

    HAS_USEARCH = True
except ImportError:  # pragma: no cover
    _UsearchIndex = None
    HAS_USEARCH = False


class HnswSidecar:
    """Persisted usearch index keyed by ann_keys.key."""

    def __init__(self, db_path: str, dimensions: int, model_id: str) -> None:
        self.dimensions = dimensions
        self.model_id = model_id
        self._persist = db_path != ":memory:"
        # The index file is per-model: a multiuser server can hold accounts on
        # different embedding models against one DB, and their indexes must not
        # share a file or they clobber each other on save. A short hash keeps
        # the name filesystem-safe whatever the model id contains.
        tag = hashlib.sha1(model_id.encode("utf-8")).hexdigest()[:12]
        self.index_path = Path(f"{db_path}.{tag}.usearch") if self._persist else None
        self.meta_path = (
            Path(f"{db_path}.{tag}.usearch.json") if self._persist else None
        )
        self.index = _UsearchIndex(ndim=dimensions, metric="cos")
        self._loaded_ok = self._try_load()

    @property
    def size(self) -> int:
        return len(self.index)

    @property
    def needs_rebuild(self) -> bool:
        return not self._loaded_ok

    def _try_load(self) -> bool:
        if not self._persist or self.index_path is None or not self.index_path.exists():
            return False
        try:
            meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
            if meta.get("model_id") != self.model_id or meta.get("dimensions") != self.dimensions:
                return False
            self.index.load(str(self.index_path))
            return True
        except Exception:
            return False

    def save(self) -> None:
        if not self._persist or self.index_path is None:
            return
        try:
            self.index.save(str(self.index_path))
            self.meta_path.write_text(
                json.dumps({"model_id": self.model_id, "dimensions": self.dimensions}),
                encoding="utf-8",
            )
        except Exception:
            pass  # the sidecar is a cache; persistence failures must not break writes

    def add(self, key: int, vector: list[float]) -> None:
        vec = np.asarray(vector, dtype=np.float32)
        try:
            if self.index.contains(key):
                self.index.remove(key)
            self.index.add(key, vec)
        except Exception:
            self._loaded_ok = False  # force rebuild on next opportunity

    def remove(self, key: int) -> None:
        try:
            if self.index.contains(key):
                self.index.remove(key)
        except Exception:
            self._loaded_ok = False

    def search(self, vector: list[float], k: int) -> list[int]:
        vec = np.asarray(vector, dtype=np.float32)
        matches = self.index.search(vec, k)
        keys = np.asarray(matches.keys).reshape(-1)
        return [int(key) for key in keys]

    def rebuild(self, items: list[tuple[int, bytes]]) -> None:
        """items: (key, float32-blob) pairs for every active memory."""
        self.index = _UsearchIndex(ndim=self.dimensions, metric="cos")
        for key, blob in items:
            self.index.add(key, np.frombuffer(blob, dtype=np.float32))
        self._loaded_ok = True
        self.save()
