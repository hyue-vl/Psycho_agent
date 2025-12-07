"""Lightweight text-based knowledge base for COKE CBT heuristics."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

from ..config import settings

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class _KBEntry:
    situation: str
    thought: str
    emotion: str
    core_beliefs: List[str]
    distortions: List[str]
    interventions: List[str]

    def as_chain(self) -> str:
        chain = f"Situation: {self.situation} -> Thought: {self.thought} -> Emotion: {self.emotion}"
        if self.core_beliefs:
            chain += f" | Core beliefs: {', '.join(self.core_beliefs)}"
        if self.distortions:
            chain += f" | Distortions: {', '.join(self.distortions)}"
        return chain


class COKEKGraph:
    """Text-backed retrieval surface that emulates the previous Neo4j API."""

    def __init__(self, kb_path: str | None = None) -> None:
        self._kb_path = Path(kb_path or settings.knowledge.coke_knowledge_path)
        self._entries: List[_KBEntry] = []
        self._load_kb()

    def _load_kb(self) -> None:
        if not self._kb_path.exists():
            LOGGER.warning("COKE knowledge file missing at %s", self._kb_path)
            self._entries = []
            return
        raw = self._kb_path.read_text(encoding="utf-8")
        blocks = [block.strip() for block in raw.split("\n---") if block.strip()]
        entries: List[_KBEntry] = []
        for block in blocks:
            payload: Dict[str, str] = {}
            for line in block.splitlines():
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                payload[key.strip().lower()] = value.strip()
            entries.append(
                _KBEntry(
                    situation=payload.get("situation", ""),
                    thought=payload.get("thought", ""),
                    emotion=payload.get("emotion", ""),
                    core_beliefs=_split(payload.get("core_beliefs")),
                    distortions=_split(payload.get("distortions")),
                    interventions=_split(payload.get("interventions")),
                )
            )
        self._entries = entries
        LOGGER.info("Loaded %d COKE KB entries from %s", len(entries), self._kb_path)

    def fetch_paths(self, situation: str, belief: str, limit: int = 5) -> List[str]:
        if not self._entries:
            return []
        scored = sorted(
            self._entries,
            key=lambda entry: _score_entry(entry, situation, belief),
            reverse=True,
        )
        filtered = [entry for entry in scored if _score_entry(entry, situation, belief) > 0]
        limit = min(limit, settings.knowledge.max_hits)
        return [entry.as_chain() for entry in filtered[:limit]]

    @lru_cache(maxsize=256)
    def interventions_for_distortion(self, distortion: str) -> List[str]:
        if not distortion:
            return []
        needle = distortion.lower()
        matches: List[str] = []
        for entry in self._entries:
            if any(needle in dist.lower() for dist in entry.distortions):
                matches.extend(entry.interventions)
        deduped: List[str] = []
        for item in matches:
            if item and item not in deduped:
                deduped.append(item)
        return deduped


def _split(value: str | None) -> List[str]:
    if not value:
        return []
    parts = [part.strip() for part in value.replace(";", ",").split(",")]
    return [part for part in parts if part]


def _score_entry(entry: _KBEntry, situation: str, belief: str) -> float:
    situation_score = _fuzzy_contains(entry.situation, situation)
    belief_score = max(_fuzzy_contains(" ".join(entry.core_beliefs), belief), _fuzzy_contains(" ".join(entry.distortions), belief))
    return situation_score * 0.6 + belief_score * 0.4


def _fuzzy_contains(haystack: str, needle: str) -> float:
    if not haystack or not needle:
        return 0.0
    hay_tokens = set(haystack.lower().split())
    needle_tokens = set(needle.lower().split())
    if not hay_tokens or not needle_tokens:
        return 0.0
    overlap = len(hay_tokens & needle_tokens)
    score = overlap / max(len(needle_tokens), 1)
    if score < settings.knowledge.fuzzy_match_threshold:
        return 0.0
    return score
