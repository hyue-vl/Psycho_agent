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
class AffectiveState:
    """Continuous latent state estimated by the affective state machine."""

    arousal: float = 0.5
    valence: float = 0.5
    trust: float = 0.5
    safety: float = 0.5

    def clamp(self) -> "AffectiveState":
        return AffectiveState(
            arousal=_clamp01(self.arousal),
            valence=_clamp01(self.valence),
            trust=_clamp01(self.trust),
            safety=_clamp01(self.safety),
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "arousal": self.arousal,
            "valence": self.valence,
            "trust": self.trust,
            "safety": self.safety,
        }


@dataclass(slots=True)
class UserProfile:
    """Compact encoding of θ_u personality parameters."""

    attachment_style: str = "secure"
    reactivity: float = 0.5
    trust_sensitivity: float = 0.5
    safety_bias: float = 0.5
    baseline_arousal: float = 0.5
    baseline_valence: float = 0.6


@dataclass(slots=True)
class KnowledgeContext:
    coke_paths: List[str] = field(default_factory=list)
    rag_snippets: List[str] = field(default_factory=list)
    kb_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StrategyCandidate:
    label: str
    rationale: str
    draft_response: str
    projected_belief: Optional[BeliefState] = None
    reward_vector: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class GlobalState(TypedDict, total=False):
    user_input: str
    user_id: str
    working_context: List[Dict[str, str]]
    recall_memory: List[MemorySlice]
    archival_memory: List[str]
    belief_state: BeliefState
    affective_state: AffectiveState
    user_profile: UserProfile
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


AgentNode = Literal["memory", "perception", "planning", "action"]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
