"""Agentic memory system inspired by A-MEM."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
import uuid
from datetime import datetime, timezone
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .config import settings
from .llm import QwenLLM
from .types import MemorySlice

LOGGER = logging.getLogger(__name__)

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


def _normalize_tuples(value: Any) -> List[Tuple[str, str, str]]:
    if not value:
        return []
    if isinstance(value, dict):
        items = value.get("tuples") or value.get("data") or value.get("memory") or []
    else:
        items = value
    tuples: List[Tuple[str, str, str]] = []
    for item in items:
        if isinstance(item, dict):
            subject = str(item.get("subject") or item.get("s") or "").strip()
            relation = str(item.get("relation") or item.get("r") or "").strip()
            obj = str(item.get("object") or item.get("o") or "").strip()
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            subject = str(item[0]).strip()
            relation = str(item[1]).strip()
            obj = str(item[2]).strip()
        else:
            continue
        triple = (subject, relation, obj)
        if all(triple):
            tuples.append(triple)
    return tuples


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
class MemoryNote:
    note_id: str
    role: str
    content: str
    context: str
    summary: str
    keywords: List[str]
    tags: List[str]
    links: List[str]
    metadata: Dict[str, Any]
    timestamp: float
    importance: float = 0.0
    tuples: List[Tuple[str, str, str]] = field(default_factory=list)


@dataclass(slots=True)
class MemoryProfile:
    notes: Dict[str, MemoryNote] = field(default_factory=dict)
    core_memory: Dict[str, str] = field(default_factory=dict)


_DEFAULT_HISTORY_DIR = Path(
    os.getenv("PSYCHO_AGENT_HISTORY_DIR")
    or (Path(__file__).resolve().parent / "conversation_logs")
)


class PersistentHistoryLogger:
    """Writes per-user interaction history to disk as JSONL + TXT."""

    def __init__(
        self,
        base_dir: Optional[str | Path] = None,
        *,
        create_text_mirror: bool = True,
    ) -> None:
        root = Path(base_dir) if base_dir is not None else _DEFAULT_HISTORY_DIR
        self._jsonl_dir = root / "jsonl"
        self._txt_dir = root / "txt"
        self._create_text_mirror = create_text_mirror
        self._jsonl_dir.mkdir(parents=True, exist_ok=True)
        if self._create_text_mirror:
            self._txt_dir.mkdir(parents=True, exist_ok=True)

    def append(self, user_id: str, event: Dict[str, Any]) -> None:
        """Persist the structured event and optionally a readable mirror."""
        payload = dict(event)
        payload.setdefault("logged_at", _now())
        self._write_jsonl(user_id, payload)
        if self._create_text_mirror:
            self._write_text(user_id, payload)

    # Internal helpers -------------------------------------------------- #

    def _jsonl_path(self, user_id: str) -> Path:
        return self._jsonl_dir / f"{user_id}.jsonl"

    def _txt_path(self, user_id: str) -> Path:
        return self._txt_dir / f"{user_id}.txt"

    def _write_jsonl(self, user_id: str, payload: Dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=False)
        try:
            with self._jsonl_path(user_id).open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError as exc:  # pragma: no cover - filesystem hiccups
            LOGGER.error("Unable to persist memory jsonl for %s: %s", user_id, exc)

    def _write_text(self, user_id: str, payload: Dict[str, Any]) -> None:
        timestamp = payload.get("timestamp") or payload.get("logged_at") or _now()
        iso = datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
        role = payload.get("role", "unknown").upper()
        content = payload.get("content", "")
        annotation = payload.get("annotation") or {}
        note_id = payload.get("note_id")
        annotation_tags = annotation.get("personalization_tags") or annotation.get("tags") or []
        summary = payload.get("summary") or ""
        lines = [
            f"[{iso}] ({note_id}) {role}: {content}",
            f"Summary: {summary}",
        ]
        if annotation:
            lines.append(
                "Annotation: "
                + json.dumps(
                    {
                        "intent": annotation.get("intent"),
                        "strategy": annotation.get("strategy"),
                        "risk_level": annotation.get("risk_level"),
                        "tags": annotation_tags,
                    },
                    ensure_ascii=False,
                )
            )
        text_line = "\n".join(lines) + "\n"
        try:
            with self._txt_path(user_id).open("a", encoding="utf-8") as fh:
                fh.write(text_line)
        except OSError as exc:  # pragma: no cover - filesystem hiccups
            LOGGER.error("Unable to persist memory txt for %s: %s", user_id, exc)


class InteractionAnnotator:
    """LLM-backed annotator that labels agent responses for future tuning."""

    SYSTEM_PROMPT = """你是一名对话数据标注助手，负责为治疗师的回复生成训练标签。\
输出 JSON，字段包含: \
intent (string), strategy (string), tone (string), risk_level (low|medium|high), \
personalization_tags (array of short tokens), follow_up (bool), training_notes (string)。\
根据用户输入与治疗师回复，提炼最重要的信息。"""

    _RISK_KEYWORDS = {"自杀", "suicide", "die", "kill", "伤害", "病危", "危机"}

    def __init__(
        self,
        *,
        llm: Optional[QwenLLM] = None,
        enabled: bool = True,
        temperature: float = 0.15,
        max_tokens: int = 320,
    ) -> None:
        self._enabled = enabled
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._llm = llm or (QwenLLM(
            model=settings.qwen.small_model,
            temperature=temperature,
            max_output_tokens=max_tokens,
            timeout_ms=min(settings.qwen.timeout, 4000),
        ) if enabled else None)

    def annotate(
        self,
        *,
        user_message: Optional[str],
        assistant_message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        text = assistant_message.strip()
        if not text:
            return None
        metadata = metadata or {}
        payload = self._build_payload(user_message, assistant_message, metadata)
        if not self._enabled or self._llm is None:
            return self._fallback(user_message, assistant_message, metadata)
        try:
            raw = self._llm.chat(
                [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": payload},
                ],
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
            return self._parse_response(raw, user_message, assistant_message, metadata)
        except Exception as exc:  # pragma: no cover - network paths
            LOGGER.debug("InteractionAnnotator fallback due to error: %s", exc)
            return self._fallback(user_message, assistant_message, metadata)

    # Internal helpers -------------------------------------------------- #

    def _build_payload(
        self,
        user_message: Optional[str],
        assistant_message: str,
        metadata: Dict[str, Any],
    ) -> str:
        tags = metadata.get("tags") or []
        topic = metadata.get("topic") or metadata.get("context") or "general"
        prompt = (
            f"来访者: {user_message or '（无最新输入）'}\n"
            f"治疗师: {assistant_message}\n"
            f"语境: {topic}; 标签: {', '.join(tags)}\n"
            "请给出 JSON 标签。"
        )
        return prompt

    def _parse_response(
        self,
        raw: str,
        user_message: Optional[str],
        assistant_message: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not raw:
            return self._fallback(user_message, assistant_message, metadata)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return self._fallback(user_message, assistant_message, metadata)
        return self._normalize(data, user_message, assistant_message, metadata)

    def _normalize(
        self,
        data: Dict[str, Any],
        user_message: Optional[str],
        assistant_message: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        tags = data.get("personalization_tags") or data.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]
        normalized = {
            "intent": str(data.get("intent") or data.get("label") or ""),
            "strategy": str(data.get("strategy") or data.get("plan") or ""),
            "tone": str(data.get("tone") or "supportive"),
            "risk_level": str(data.get("risk_level") or "low"),
            "personalization_tags": [str(tag) for tag in tags if tag],
            "follow_up": bool(data.get("follow_up") or data.get("follow_up_needed")),
            "training_notes": str(data.get("training_notes") or data.get("notes") or ""),
            "reference": {
                "user": user_message,
                "assistant": assistant_message,
                "context_tags": metadata.get("tags") or [],
            },
        }
        return normalized

    def _fallback(
        self,
        user_message: Optional[str],
        assistant_message: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        combined = f"{user_message or ''} {assistant_message}".lower()
        risk = "high" if any(keyword in combined for keyword in self._RISK_KEYWORDS) else "low"
        strategy = "crisis_management" if risk == "high" else "supportive_reflection"
        tags = list(metadata.get("tags") or [])
        if "plan" in combined:
            tags.append("action_plan")
        return {
            "intent": "assist_user",
            "strategy": strategy,
            "tone": "supportive",
            "risk_level": risk,
            "personalization_tags": sorted(set(tags)),
            "follow_up": risk != "low",
            "training_notes": "Auto-generated fallback annotation.",
            "reference": {
                "user": user_message,
                "assistant": assistant_message,
                "context_tags": metadata.get("tags") or [],
            },
        }

class TupleMemoryEncoder:
    """LLM-backed encoder that converts free text into (subject, relation, object) tuples."""

    SYSTEM_PROMPT = """你是会话记忆提取器。\
将输入压缩成若干三元组（subject, relation, object），用于长期记忆检索。\
输出 JSON，对象结构为 {"tuples": [{"subject": "...", "relation": "...", "object": "..."}]}。\
subject 可以是“来访者”“治疗师”或文本中的核心实体；relation 需要是动词或短语；object 写关键信息。\
如果信息不足，可以只返回 1-2 个高价值三元组，不要输出空字符串。"""

    def __init__(
        self,
        *,
        llm: Optional[QwenLLM] = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> None:
        self._llm = llm or QwenLLM(
            model=settings.qwen.small_model,
            temperature=temperature,
            max_output_tokens=max_tokens,
            timeout_ms=min(settings.qwen.timeout, 4000),
        )

    def encode(self, role: str, content: str, context: Optional[str] = None) -> List[Tuple[str, str, str]]:
        """Return tuple memories extracted from the content."""
        text = (content or "").strip()
        if not text:
            return []
        payload = (
            f"角色: {role}\n"
            f"场景: {context or 'general'}\n"
            f"原文: {text}\n"
            "请给出 1-3 个中文三元组。"
        )
        tuples: List[Tuple[str, str, str]] = []
        try:
            raw = self._llm.chat(
                [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": payload},
                ],
                max_tokens=512,
                temperature=0.2,
            )
            tuples = self._parse_response(raw)
        except Exception as exc:  # pragma: no cover - network paths
            LOGGER.debug("TupleMemoryEncoder fallback due to error: %s", exc)
        return tuples or self._fallback(role, text)

    def _parse_response(self, raw: str) -> List[Tuple[str, str, str]]:
        if not raw:
            return []
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return []
        if isinstance(data, dict):
            candidates = data.get("tuples") or data.get("memory") or data.get("data") or []
        elif isinstance(data, list):
            candidates = data
        else:
            return []
        triples: List[Tuple[str, str, str]] = []
        for item in candidates:
            subj: Optional[str]
            rel: Optional[str]
            obj: Optional[str]
            if isinstance(item, dict):
                subj = item.get("subject") or item.get("s")
                rel = item.get("relation") or item.get("r")
                obj = item.get("object") or item.get("o")
            elif isinstance(item, (list, tuple)) and len(item) >= 3:
                subj, rel, obj = item[:3]
            else:
                continue
            cleaned = [self._clean_field(value) for value in (subj, rel, obj)]
            if all(cleaned):
                triples.append(tuple(cleaned))  # type: ignore[arg-type]
        return triples

    @staticmethod
    def _clean_field(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _fallback(self, role: str, content: str) -> List[Tuple[str, str, str]]:
        keywords = _tokenize(content)
        triples: List[Tuple[str, str, str]] = []
        if keywords:
            triples.append((role or "speaker", "关注", keywords[0]))
            if len(keywords) > 1:
                triples.append((role or "speaker", "重复提到", keywords[1]))
        snippet = content[:80]
        if snippet:
            triples.append((role or "speaker", "陈述", snippet))
        # Deduplicate while preserving order
        seen = set()
        ordered: List[Tuple[str, str, str]] = []
        for triple in triples:
            if triple in seen:
                continue
            seen.add(triple)
            ordered.append(triple)
        return ordered[:3]

class KeywordExtractor:
    """Very small helper that emulates structured note taking."""

    def extract(self, text: str, limit: int = 8) -> List[str]:
        tokens = _tokenize(text)
        seen: Dict[str, int] = {}
        for token in tokens:
            seen[token] = seen.get(token, 0) + 1
        ranked = sorted(seen.items(), key=lambda item: item[1], reverse=True)
        return [token for token, _ in ranked[:limit]]


class MemoryGraph:
    """In-memory Zettelkasten-inspired network."""

    def __init__(self, link_threshold: float = 0.35) -> None:
        self._profiles: Dict[str, MemoryProfile] = defaultdict(MemoryProfile)
        self._extractor = KeywordExtractor()
        self._link_threshold = link_threshold

    def add_entry(
        self,
        user_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryNote:
        metadata = metadata or {}
        profile = self._profiles[user_id]
        note = self._build_note(role, content, metadata)
        profile.notes[note.note_id] = note
        related = self._find_related_notes(profile.notes.values(), note)
        self._link_notes(note, related)
        return note

    def replace_core(self, user_id: str, fields: Dict[str, str]) -> None:
        profile = self._profiles[user_id]
        profile.core_memory = dict(fields)

    def retrieve(
        self,
        user_id: str,
        top_k: int,
        query: Optional[str] = None,
    ) -> List[MemoryNote]:
        profile = self._profiles[user_id]
        notes = list(profile.notes.values())
        if not notes:
            return []
        query_tokens = _tokenize(query or "")
        now = _now()
        scored: List[Tuple[float, MemoryNote]] = []
        for note in notes:
            recency = math.exp(-(now - note.timestamp) / 600.0)
            overlap = _jaccard(query_tokens, note.keywords) if query_tokens else 0.0
            score = 0.6 * overlap + 0.3 * recency + 0.1 * note.importance
            scored.append((score, note))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [note for _, note in scored[:top_k]]

    def export_profile(self, user_id: str) -> Dict[str, Any]:
        profile = self._profiles[user_id]
        return {
            "core_memory": dict(profile.core_memory),
            "notes": [
                {
                    "note_id": note.note_id,
                    "role": note.role,
                    "context": note.context,
                    "summary": note.summary,
                    "keywords": list(note.keywords),
                    "tags": list(note.tags),
                    "links": list(note.links),
                    "metadata": dict(note.metadata),
                    "importance": note.importance,
                    "tuples": [tuple(item) for item in note.tuples],
                }
                for note in profile.notes.values()
            ],
        }

    # Internal helpers -------------------------------------------------- #

    def _build_note(self, role: str, content: str, metadata: Dict[str, Any]) -> MemoryNote:
        keywords = metadata.get("keywords") or self._extractor.extract(content)
        tags = sorted(set((metadata.get("tags") or []) + keywords[:4]))
        summary = metadata.get("summary") or _make_summary(content, keywords)
        context = metadata.get("context") or metadata.get("topic") or role
        importance = float(metadata.get("score", 0.0))
        importance = max(importance, min(1.0, len(content) / 400.0))
        metadata = dict(metadata)
        tuple_memory = _normalize_tuples(metadata.pop("tuples", None))
        return MemoryNote(
            note_id=str(metadata.get("note_id") or uuid.uuid4()),
            role=role,
            content=content,
            context=context,
            summary=summary,
            keywords=keywords,
            tags=tags,
            links=[],
            metadata=dict(metadata),
            timestamp=_now(),
            importance=importance,
            tuples=tuple_memory,
        )

    def _find_related_notes(
        self,
        candidates: Iterable[MemoryNote],
        note: MemoryNote,
        limit: int = 5,
    ) -> List[Tuple[float, MemoryNote]]:
        related: List[Tuple[float, MemoryNote]] = []
        for other in candidates:
            if other.note_id == note.note_id:
                continue
            similarity = 0.7 * _jaccard(note.keywords, other.keywords) + 0.3 * _jaccard(note.tags, other.tags)
            if similarity >= self._link_threshold:
                related.append((similarity, other))
        related.sort(key=lambda item: item[0], reverse=True)
        return related[:limit]

    def _link_notes(self, note: MemoryNote, related: List[Tuple[float, MemoryNote]]) -> None:
        for _, other in related:
            if other.note_id not in note.links:
                note.links.append(other.note_id)
            if note.note_id not in other.links:
                other.links.append(note.note_id)
            self._evolve_existing(other, note)
        if related:
            # Reflect cross-links in metadata
            note.metadata["linked_topics"] = sorted(
                {rel.context for _, rel in related} | set(note.metadata.get("linked_topics", []))
            )

    def _evolve_existing(self, target: MemoryNote, stimulus: MemoryNote) -> None:
        merged_tags = list(dict.fromkeys(target.tags + stimulus.tags))
        target.tags = merged_tags[:12]
        related_keywords = set(target.metadata.get("related_keywords", []))
        related_keywords.update(stimulus.keywords[:3])
        target.metadata["related_keywords"] = sorted(related_keywords)
        contexts = target.metadata.setdefault("context_history", [])
        if stimulus.context not in contexts:
            contexts.append(stimulus.context)
        target.summary = self._refine_summary(target.summary, stimulus.context)

    @staticmethod
    def _refine_summary(existing: str, stimulus_context: str) -> str:
        existing = existing.strip()
        if not existing:
            return stimulus_context
        if stimulus_context in existing:
            return existing
        return f"{existing} | context link: {stimulus_context}"


# --------------------------------------------------------------------------- #
# Public manager


@dataclass
class AMemMemoryManager:
    """Agentic memory orchestrator inspired by A-MEM."""

    formatter: MemoryFormatter = field(default_factory=MemoryFormatter)
    graph: MemoryGraph = field(default_factory=MemoryGraph)
    tuple_encoder: TupleMemoryEncoder = field(default_factory=TupleMemoryEncoder)
    history_logger: Optional[PersistentHistoryLogger] = field(default_factory=PersistentHistoryLogger)
    annotator: Optional[InteractionAnnotator] = field(default_factory=InteractionAnnotator)
    _recent_user_messages: Dict[str, str] = field(default_factory=dict, init=False, repr=False)

    def load_context(
        self,
        user_id: str,
        top_k: int = 5,
        fallback_snippets: Optional[List[str]] = None,
        query: Optional[str] = None,
    ) -> Dict[str, List[Any]]:
        notes = self.graph.retrieve(user_id, top_k, query=query)
        if not notes and fallback_snippets:
            formatted = self._fallback_from_snippets(fallback_snippets, top_k)
            return {"working": formatted[:2], "recall": formatted, "archival": []}
        slices = [self._note_to_slice(note) for note in notes]
        archival = self._format_core(user_id)
        return {"working": slices[:2], "recall": slices, "archival": archival}

    def core_memory_replace(self, user_id: str, fields: Dict[str, str]) -> None:
        if not fields:
            return
        self.graph.replace_core(user_id, fields)

    def recall_append(self, user_id: str, entries: List[Dict[str, Any] | MemorySlice]) -> None:
        for entry in entries:
            if isinstance(entry, MemorySlice):
                metadata = {"summary": entry.content[:120], "tags": ["legacy"], "context": entry.role}
                enriched = self._annotate_with_tuples(entry.role, entry.content, metadata)
                self.graph.add_entry(user_id, entry.role, entry.content, enriched)
                continue
            if not entry.get("content"):
                continue
            role = entry.get("role", "user")
            metadata = dict(entry.get("metadata") or {})
            content = entry["content"]
            enriched = self._annotate_with_tuples(role, content, metadata)
            note = self.graph.add_entry(
                user_id=user_id,
                role=role,
                content=content,
                metadata=enriched,
            )
            self._update_recent_messages(user_id, role, content)
            self._persist_interaction(
                user_id=user_id,
                role=role,
                content=content,
                metadata=metadata,
                enriched_metadata=enriched,
                note=note,
            )

    def update_core(self, user_id: str, fields: Dict[str, str]) -> None:
        self.core_memory_replace(user_id, fields)

    def export_profile(self, user_id: str) -> Dict[str, Any]:
        return self.graph.export_profile(user_id)

    # Helpers ----------------------------------------------------------- #

    def _note_to_slice(self, note: MemoryNote) -> MemorySlice:
        tuple_repr = "; ".join(" | ".join(triple) for triple in note.tuples[:3]) or "None"
        detail = (
            f"{note.summary}\n"
            f"Context: {note.context}\n"
            f"Keywords: {', '.join(note.keywords[:6])}\n"
            f"Tags: {', '.join(note.tags[:8])}\n"
            f"Tuples: {tuple_repr}\n"
            f"Links: {', '.join(note.links[:6]) or 'None'}\n"
            f"Detail: {note.content[: self.formatter.max_content_chars]}"
        )
        metadata = {
            "summary": note.summary,
            "tags": note.tags,
            "topic": note.context,
        }
        return self.formatter.format_slice(
            role=note.role,
            content=detail,
            metadata=metadata,
            score=note.importance,
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

    def _format_core(self, user_id: str) -> List[str]:
        profile = self.graph.export_profile(user_id)
        if not profile["core_memory"]:
            return []
        formatted = []
        for key, value in profile["core_memory"].items():
            formatted.append(self.formatter.format_archival(key, value))
        return formatted

    def _update_recent_messages(self, user_id: str, role: str, content: str) -> None:
        if role == "user" and content:
            self._recent_user_messages[user_id] = content

    def _persist_interaction(
        self,
        *,
        user_id: str,
        role: str,
        content: str,
        metadata: Dict[str, Any],
        enriched_metadata: Dict[str, Any],
        note: MemoryNote,
    ) -> None:
        if not self.history_logger:
            return
        annotation: Optional[Dict[str, Any]] = None
        if self.annotator and role == "assistant":
            annotation = self.annotator.annotate(
                user_message=self._recent_user_messages.get(user_id),
                assistant_message=content,
                metadata=enriched_metadata,
            )
        event = {
            "event_id": str(uuid.uuid4()),
            "user_id": user_id,
            "role": role,
            "content": content,
            "summary": note.summary,
            "context": note.context,
            "note_id": note.note_id,
            "timestamp": note.timestamp,
            "metadata": metadata,
            "enriched_metadata": enriched_metadata,
            "tuples": [list(item) for item in note.tuples],
            "annotation": annotation,
        }
        try:
            self.history_logger.append(user_id, event)
        except Exception as exc:  # pragma: no cover - log persistence errors
            LOGGER.error("Failed to persist interaction for %s: %s", user_id, exc)

    def _annotate_with_tuples(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        metadata = dict(metadata or {})
        if not self.tuple_encoder:
            return metadata
        tuples = self.tuple_encoder.encode(role, content, metadata.get("context") or metadata.get("topic"))
        if tuples:
            metadata["tuples"] = tuples
        return metadata


# Backwards compatibility --------------------------------------------------- #
# External code may still import MemGPTManager, so keep the alias.
MemGPTManager = AMemMemoryManager
