"""Interfaces with MemGPT for PKB management."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib import error, request

try:  # pragma: no cover
    from memgpt import MemGPT
except ImportError:  # pragma: no cover
    MemGPT = None

from .config import settings
from .types import MemorySlice

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class MemoryFormatter:
    """Normalize free-form memories into a compact PKB-friendly template."""

    max_content_chars: int = 512

    def format_slice(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        score: float = 0.0,
    ) -> MemorySlice:
        metadata = metadata or {}
        cleaned = " ".join(content.strip().split())
        summary = metadata.get("summary") or cleaned[:120]
        tags = metadata.get("tags") or []
        topic = metadata.get("topic")
        header = f"[{role.upper()}] {summary}"
        if topic:
            header += f" | topic: {topic}"
        if tags:
            header += f" | tags: {', '.join(tags)}"
        detail = cleaned[: self.max_content_chars]
        formatted = f"{header}\nDetail: {detail}" if detail else header
        return MemorySlice(role=role, content=formatted, score=score)

    def format_archival(self, content: str) -> str:
        return self.format_slice("archival", content).content


class LettaRuntimeClient:
    """Minimal HTTP bridge to Letta runtime hooks."""

    def __init__(self) -> None:
        self._base_url = settings.letta.base_url.rstrip("/")
        self._headers = {"Content-Type": "application/json"}
        if settings.letta.api_key:
            self._headers["Authorization"] = f"Bearer {settings.letta.api_key}"
        self._timeout = settings.letta.timeout

    def core_memory_replace(self, user_id: str, namespace: str, fields: Dict[str, str]) -> None:
        if not fields:
            return
        payload = {"user_id": user_id, "namespace": namespace, "fields": fields}
        self._post("/mem/core/replace", payload)

    def recall_append(self, user_id: str, namespace: str, slices: List[MemorySlice]) -> None:
        if not slices:
            return
        payload = {
            "user_id": user_id,
            "namespace": namespace,
            "entries": [slice.__dict__ for slice in slices],
        }
        self._post("/mem/recall/append", payload)

    def _post(self, route: str, payload: Dict[str, Any]) -> None:
        url = f"{self._base_url}{route}"
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(url, data=data, headers=self._headers, method="POST")
        try:
            request.urlopen(req, timeout=self._timeout).read()
        except error.URLError as exc:  # pragma: no cover
            LOGGER.warning("Letta runtime call failed (%s): %s", route, exc)


@dataclass
class MemGPTManager:
    """Wrapper around MemGPT client to sync working / recall / archival memories."""

    namespace: str = "psycho_world"
    formatter: MemoryFormatter = field(default_factory=MemoryFormatter)

    def __post_init__(self) -> None:
        if MemGPT is None:
            LOGGER.warning("MemGPT is not installed; memory operations are no-ops.")
            self._client = None
        else:
            self._client = MemGPT()
        self._letta = LettaRuntimeClient() if settings.letta.enabled else None

    def load_context(
        self,
        user_id: str,
        top_k: int = 5,
        fallback_snippets: Optional[List[str]] = None,
    ) -> Dict[str, List[Any]]:
        conversations = self._recall_from_memgpt(user_id, top_k)
        slices = self._format_conversations(conversations)
        if not slices and fallback_snippets:
            slices = self._fallback_from_snippets(fallback_snippets, top_k)
        archival_entries = self._fetch_archival(user_id)
        archival = [self.formatter.format_archival(entry) for entry in archival_entries]
        return {"working": slices[:2], "recall": slices, "archival": archival}

    def core_memory_replace(self, user_id: str, fields: Dict[str, str]) -> None:
        if not fields:
            return
        if self._client is not None:
            hook = getattr(self._client, "core_memory_replace", None)
            try:
                if callable(hook):
                    hook(user_id=user_id, namespace=self.namespace, fields=fields)
                else:
                    self._client.update_core_memory(user_id=user_id, namespace=self.namespace, fields=fields)
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("MemGPT core memory update failed: %s", exc)
        if self._letta:
            self._letta.core_memory_replace(user_id, self.namespace, fields)

    def recall_append(self, user_id: str, entries: List[Dict[str, Any] | MemorySlice]) -> None:
        slices = []
        for entry in entries:
            if isinstance(entry, MemorySlice):
                slices.append(entry)
            else:
                slices.append(
                    self.formatter.format_slice(
                        role=entry.get("role", "user"),
                        content=entry.get("content", ""),
                        metadata=entry.get("metadata"),
                        score=float(entry.get("score", 0.0)),
                    )
                )
        if not slices:
            return
        if self._client is not None:
            hook = getattr(self._client, "recall_append", None)
            try:
                if callable(hook):
                    hook(user_id=user_id, namespace=self.namespace, entries=[slice.__dict__ for slice in slices])
                else:
                    store = getattr(self._client, "store_memory", None)
                    if callable(store):
                        for slice in slices:
                            store(user_id=user_id, namespace=self.namespace, memory=slice.__dict__)
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("MemGPT recall append failed: %s", exc)
        if self._letta:
            self._letta.recall_append(user_id, self.namespace, slices)

    # ------------------------------------------------------------------ #
    # Legacy alias to avoid breaking external imports.
    def update_core(self, user_id: str, fields: Dict[str, str]) -> None:
        self.core_memory_replace(user_id, fields)

    # ------------------------------------------------------------------ #

    def _recall_from_memgpt(self, user_id: str, top_k: int) -> List[Dict[str, Any]]:
        if self._client is None:
            return []
        try:
            return self._client.recall_memories(user_id=user_id, namespace=self.namespace, top_k=top_k)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("MemGPT recall failed: %s", exc)
            return []

    def _format_conversations(self, conversations: List[Dict[str, Any]]) -> List[MemorySlice]:
        formatted: List[MemorySlice] = []
        for item in conversations:
            formatted.append(
                self.formatter.format_slice(
                    role=item.get("role", "user"),
                    content=item.get("content", ""),
                    metadata=item.get("metadata"),
                    score=float(item.get("score", 0.0)),
                )
            )
        return formatted

    def _fallback_from_snippets(self, snippets: List[str], top_k: int) -> List[MemorySlice]:
        fallback = []
        for snippet in snippets[:top_k]:
            fallback.append(
                self.formatter.format_slice(
                    role="rag",
                    content=snippet,
                    metadata={"topic": "vector_fallback", "tags": ["rag", "fallback"]},
                )
            )
        return fallback

    def _fetch_archival(self, user_id: str) -> List[str]:
        if self._client is None:
            return []
        collected: List[str] = []
        pkb_getter = getattr(self._client, "get_personal_knowledge_base", None)
        if callable(pkb_getter):
            try:
                pkb = pkb_getter(user_id=user_id, namespace=self.namespace)
                collected.extend(_normalize_archival(pkb))
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("MemGPT PKB fetch failed: %s", exc)
        archival_getter = getattr(self._client, "get_archival", None)
        if callable(archival_getter):
            try:
                collected.extend(_normalize_archival(archival_getter(user_id)))
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("MemGPT archival fetch failed: %s", exc)
        return collected


def _normalize_archival(items: Any) -> List[str]:
    if items is None:
        return []
    if isinstance(items, list):
        normalized: List[str] = []
        for item in items:
            if isinstance(item, str):
                normalized.append(item)
            elif isinstance(item, dict):
                normalized.append(item.get("content", ""))
        return [entry for entry in normalized if entry]
    if isinstance(items, str):
        return [items]
    return []
