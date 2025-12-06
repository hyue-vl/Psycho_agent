"""Shared dataclasses and typed dictionaries for LangGraph state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, TypedDict


@dataclass(slots=True)
class MemorySlice:
    role: str
    content: str
    score: float = 0.0


@dataclass(slots=True)
class BeliefState:
    distortions: List[str]
    emotion_valence: int
    emotion_arousal: int
    communicative_intent: str
    risk_level: float
    rationale: str


@dataclass(slots=True)
class KnowledgeContext:
    coke_paths: List[str] = field(default_factory=list)
    rag_snippets: List[str] = field(default_factory=list)
    neo4j_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StrategyCandidate:
    label: str
    rationale: str
    draft_response: str
    projected_belief: Optional[BeliefState] = None
    reward_vector: Optional[Dict[str, float]] = None
    projected_reaction: Optional[str] = None


class GlobalState(TypedDict, total=False):
    user_input: str
    user_id: str
    working_context: List[Dict[str, str]]
    recall_memory: List[MemorySlice]
    archival_memory: List[str]
    belief_state: BeliefState
    knowledge_context: KnowledgeContext
    strategies: List[StrategyCandidate]
    selected_strategy: StrategyCandidate
    final_response: str
    risk_level: float
    diagnostics: Dict[str, Any]


class RewardVector(TypedDict):
    safety: float
    empathy: float
    adherence: float
    improvement: float


AgentNode = Literal["memory", "perception", "planning", "simulation", "action"]
