"""Interfaces with MemGPT for PKB management."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

try:
    from memgpt import MemGPT
except ImportError:  # pragma: no cover
    MemGPT = None

from .types import MemorySlice

LOGGER = logging.getLogger(__name__)


@dataclass
class MemGPTManager:
    """Wrapper around MemGPT client to sync working / recall / archival memories."""

    namespace: str = "psycho_world"

    def __post_init__(self) -> None:
        if MemGPT is None:
            LOGGER.warning("MemGPT is not installed; memory operations are no-ops.")
            self._client = None
        else:
            self._client = MemGPT()

    def load_context(self, user_id: str, top_k: int = 5) -> Dict[str, List[MemorySlice]]:
        if self._client is None:
            return {"working": [], "recall": [], "archival": []}
        try:
            conversations = self._client.recall_memories(user_id=user_id, namespace=self.namespace, top_k=top_k)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("MemGPT recall failed: %s", exc)
            conversations = []
        slices = [
            MemorySlice(role=item.get("role", "user"), content=item.get("content", ""), score=item.get("score", 0.0))
            for item in conversations
        ]
        try:
            archival = self._client.get_archival(user_id)
        except Exception:  # pragma: no cover
            archival = []
        return {"working": slices[:2], "recall": slices, "archival": archival}

    def update_core(self, user_id: str, fields: Dict[str, str]) -> None:
        if self._client is None:
            return
        try:
            self._client.update_core_memory(user_id=user_id, namespace=self.namespace, fields=fields)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("MemGPT update failed: %s", exc)
