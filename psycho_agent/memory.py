"""Simplified agent memory system focused on lightweight conversation recall."""

from __future__ import annotations

import math
import re
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence

from .types import MemorySlice


# --------------------------------------------------------------------------- #
# Text utilities


STOP_WORDS = {
    "the",
    "and",
    "for",
    "with",
    "have",
    "that",
    "this",
    "then",
    "been",
    "but",
    "you",
    "are",
    "was",
    "just",
    "about",
    "from",
    "they",
    "will",
    "into",
    "them",
    "because",
    "like",
    "your",
    "when",
    "what",
    "how",
    "why",
}


def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z\u4e00-\u9fa5]{2,}", text.lower())
    return [token for token in tokens if token not in STOP_WORDS]


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _make_summary(content: str, keywords: Sequence[str]) -> str:
    content = " ".join(content.strip().split())
    if not content:
        return ""
    if keywords:
        headline = ", ".join(keywords[:4])
        return f"{headline}: {content[:160]}"
    return content[:180]


def _now() -> float:
    return time.time()


# --------------------------------------------------------------------------- #
# Data structures


@dataclass(slots=True)
class MemoryFormatter:
    """Create compact textual slices from structured notes."""

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
        summary = metadata.get("summary") or cleaned[:140]
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

    def format_archival(self, title: str, body: str) -> str:
        content = f"{title}: {body}"
        return self.format_slice("archival", content).content


@dataclass(slots=True)
class MemoryRecord:
    record_id: str
    role: str
    content: str
    summary: str
    keywords: List[str]
    metadata: Dict[str, Any]
    timestamp: float
    score: float = 0.0


class UserMemoryStore:
    """Per-user rolling memory buffer with optional core facts."""

    def __init__(self, max_history: int) -> None:
        self.history: Deque[MemoryRecord] = deque(maxlen=max_history)
        self.core_memory: Dict[str, str] = {}

    def append(self, record: MemoryRecord) -> None:
        self.history.append(record)

    def records(self) -> List[MemoryRecord]:
        return list(self.history)


# --------------------------------------------------------------------------- #
# Public manager


@dataclass
class AMemMemoryManager:
    """Lightweight memory orchestrator used by the workflow."""

    formatter: MemoryFormatter = field(default_factory=MemoryFormatter)
    max_history: int = 200

    def __post_init__(self) -> None:
        self._stores: Dict[str, UserMemoryStore] = {}

    def load_context(
        self,
        user_id: str,
        top_k: int = 5,
        fallback_snippets: Optional[List[str]] = None,
        query: Optional[str] = None,
    ) -> Dict[str, List[Any]]:
        store = self._get_store(user_id)
        records = store.records()
        if not records and fallback_snippets:
            formatted = self._fallback_from_snippets(fallback_snippets, top_k)
            return {"working": formatted[:2], "recall": formatted, "archival": []}
        selected = self._rank_records(records, top_k, query)
        slices = [self._record_to_slice(record) for record in selected]
        archival = self._format_core(store.core_memory)
        return {"working": slices[:2], "recall": slices, "archival": archival}

    def core_memory_replace(self, user_id: str, fields: Dict[str, str]) -> None:
        if not fields:
            return
        store = self._get_store(user_id)
        store.core_memory = dict(fields)

    def recall_append(self, user_id: str, entries: List[Dict[str, Any] | MemorySlice]) -> None:
        store = self._get_store(user_id)
        for entry in entries:
            record = self._build_record(entry)
            if record is None:
                continue
            store.append(record)

    def update_core(self, user_id: str, fields: Dict[str, str]) -> None:
        self.core_memory_replace(user_id, fields)

    def export_profile(self, user_id: str) -> Dict[str, Any]:
        store = self._get_store(user_id)
        return {
            "core_memory": dict(store.core_memory),
            "history": [
                {
                    "record_id": record.record_id,
                    "role": record.role,
                    "summary": record.summary,
                    "keywords": list(record.keywords),
                    "metadata": dict(record.metadata),
                    "timestamp": record.timestamp,
                }
                for record in store.records()
            ],
        }

    # Helpers ----------------------------------------------------------- #

    def _get_store(self, user_id: str) -> UserMemoryStore:
        store = self._stores.get(user_id)
        if store is None:
            store = UserMemoryStore(max_history=self.max_history)
            self._stores[user_id] = store
        return store

    def _rank_records(
        self,
        records: List[MemoryRecord],
        top_k: int,
        query: Optional[str],
    ) -> List[MemoryRecord]:
        if not records:
            return []
        query_tokens = _tokenize(query or "")
        now = _now()
        scored: List[tuple[float, MemoryRecord]] = []
        for record in records:
            recency = math.exp(-(now - record.timestamp) / 600.0)
            overlap = _jaccard(query_tokens, record.keywords) if query_tokens else 0.0
            score = 0.65 * overlap + 0.35 * recency + 0.15 * record.score
            scored.append((score, record))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [record for _, record in scored[:top_k]]

    def _record_to_slice(self, record: MemoryRecord) -> MemorySlice:
        detail = (
            f"{record.summary}\n"
            f"Keywords: {', '.join(record.keywords[:6]) or 'n/a'}\n"
            f"Tags: {', '.join(record.metadata.get('tags', [])[:8]) or 'n/a'}\n"
            f"Detail: {record.content[: self.formatter.max_content_chars]}"
        )
        metadata = {
            "summary": record.summary,
            "tags": record.metadata.get("tags", []),
            "topic": record.metadata.get("topic"),
        }
        return self.formatter.format_slice(
            role=record.role,
            content=detail,
            metadata=metadata,
            score=record.score,
        )

    def _fallback_from_snippets(self, snippets: List[str], top_k: int) -> List[MemorySlice]:
        slices = []
        for snippet in snippets[:top_k]:
            slices.append(
                self.formatter.format_slice(
                    role="rag",
                    content=snippet,
                    metadata={"topic": "vector_fallback", "tags": ["rag", "fallback"]},
                    score=0.1,
                )
            )
        return slices

    def _format_core(self, core: Dict[str, str]) -> List[str]:
        if not core:
            return []
        formatted = []
        for key, value in core.items():
            formatted.append(self.formatter.format_archival(key, value))
        return formatted

    def _build_record(self, entry: Dict[str, Any] | MemorySlice) -> Optional[MemoryRecord]:
        if isinstance(entry, MemorySlice):
            role = entry.role
            content = entry.content
            metadata: Dict[str, Any] = {
                "summary": entry.content[:120],
                "tags": ["legacy"],
                "topic": "memory_slice",
            }
        else:
            if not entry.get("content"):
                return None
            role = entry.get("role", "user")
            metadata = dict(entry.get("metadata") or {})
            content = entry["content"]
        keywords = metadata.get("keywords") or _tokenize(content)
        summary = metadata.get("summary") or _make_summary(content, keywords)
        metadata.setdefault("topic", metadata.get("context") or role)
        tags = metadata.get("tags")
        if not tags:
            metadata["tags"] = keywords[:4]
        score = float(metadata.get("score") or 0.0)
        record = MemoryRecord(
            record_id=str(metadata.get("note_id") or uuid.uuid4()),
            role=role,
            content=content,
            summary=summary,
            keywords=list(keywords),
            metadata=metadata,
            timestamp=_now(),
            score=score,
        )
        return record


# Backwards compatibility --------------------------------------------------- #
# External code may still import MemGPTManager, so keep the alias.
MemGPTManager = AMemMemoryManager
