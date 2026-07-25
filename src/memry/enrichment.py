"""Managed background enrichment for durable, pending MCP saves."""

from __future__ import annotations

import asyncio
import logging
from functools import partial
from typing import Any

import anyio.to_thread

from .store import MemoryStore


logger = logging.getLogger(__name__)


class EnrichmentWorker:
    """Drain restart-safe pending memories in bounded database batches."""

    def __init__(
        self,
        store: MemoryStore,
        *,
        batch_size: int = 8,
        poll_seconds: float = 2.0,
    ) -> None:
        self.store = store
        self.batch_size = max(1, batch_size)
        self.poll_seconds = max(0.05, poll_seconds)
        self._wake = asyncio.Event()

    def notify(self) -> None:
        """Wake the worker after a pending record has committed."""
        self._wake.set()

    async def run(self) -> None:
        """Run until cancelled; persisted pending records survive cancellation."""
        while True:
            # Clear before processing so a save committed during the provider
            # call remains signalled and cannot be lost between drain and wait.
            self._wake.clear()
            try:
                result: dict[str, Any] = await anyio.to_thread.run_sync(
                    partial(
                        self.store.process_pending_enrichments,
                        limit=self.batch_size,
                    )
                )
            except Exception:
                logger.exception("background enrichment batch failed")
                result = {"claimed": 0}
            if result.get("claimed", 0) >= self.batch_size:
                continue
            try:
                await asyncio.wait_for(
                    self._wake.wait(), timeout=self.poll_seconds
                )
            except TimeoutError:
                pass
