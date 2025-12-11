"""User world model based controller for strategy selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple

from .agents.state_machine import AffectiveStateMachine
from .types import (
    AffectiveState,
    BeliefState,
    GlobalState,
    StrategyCandidate,
    UserProfile,
    _clamp01,
)


@dataclass(slots=True)
class UserWorldModel:
    """Compact representation of the user's latent world model."""

    belief: BeliefState
    affective: AffectiveState
    profile: UserProfile
    topic_histogram: Dict[str, int] = field(default_factory=dict)
    desired_state: AffectiveState = field(default_factory=AffectiveState)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "belief": asdict(self.belief),
            "affective": self.affective.to_dict(),
            "profile": asdict(self.profile),
            "topics": self.topic_histogram,
            "desired_state": self.desired_state.to_dict(),
        }
        return payload


class WorldModelController:
    """Scores strategies by simulating affective dynamics under the user model."""

    def __init__(self, state_machine: AffectiveStateMachine, metadata_bias: float = 0.15) -> None:
        self._state_machine = state_machine
        self._metadata_bias = metadata_bias

    def build_model(self, state: GlobalState) -> UserWorldModel:
        belief = state.get("belief_state")
        if belief is None:
            raise ValueError("World model requires an inferred belief_state.")
        affective = state.get("affective_state") or self._state_machine.get_state(state.get("user_id", "default"))
        profile = state.get("user_profile") or UserProfile()
        topics = self._topic_histogram(state.get("recall_memory"))
        desired = self._desired_state(belief, profile)
        return UserWorldModel(
            belief=belief,
            affective=affective,
            profile=profile,
            topic_histogram=topics,
            desired_state=desired,
        )

    def select(
        self,
        user_id: str,
        candidates: List[StrategyCandidate],
        state: GlobalState,
    ) -> Tuple[StrategyCandidate, Dict[str, Any]]:
        if not candidates:
            raise ValueError("Cannot select strategy from empty candidate list.")
        model = self.build_model(state)
        scored: List[Dict[str, Any]] = []
        metadata_weight = self._metadata_bias * (1.0 - _clamp01(model.belief.risk_level))
        for candidate in candidates:
            predicted = self._state_machine.simulate_transition(
                user_id=user_id,
                action=candidate,
                observation=model.belief,
                current_state=model.affective,
                profile_override=model.profile,
            )
            alignment = self._alignment_score(predicted, model.desired_state)
            metadata_score = self._metadata_score(candidate)
            final_score = alignment + metadata_weight * metadata_score
            scored.append(
                {
                    "candidate": candidate,
                    "alignment": alignment,
                    "metadata_score": metadata_score,
                    "composite": final_score,
                    "predicted_state": predicted.to_dict(),
                }
            )
        scored.sort(key=lambda item: item["composite"], reverse=True)
        winner = scored[0]
        diagnostics = {
            "world_model": model.to_dict(),
            "weights": {
                "alignment": 1.0,
                "metadata": metadata_weight,
            },
            "rankings": [
                {
                    "label": item["candidate"].label,
                    "composite": item["composite"],
                    "alignment": item["alignment"],
                    "metadata": item["metadata_score"],
                    "predicted_state": item["predicted_state"],
                }
                for item in scored
            ],
            "selected": {
                "label": winner["candidate"].label,
                "score": winner["composite"],
            },
        }
        return winner["candidate"], diagnostics

    @staticmethod
    def _topic_histogram(recall_memory: Any) -> Dict[str, int]:
        histogram: Dict[str, int] = {}
        if not recall_memory:
            return histogram
        for entry in recall_memory:
            topic = None
            if isinstance(entry, dict):
                topic = entry.get("metadata", {}).get("topic")
            else:
                topic = getattr(getattr(entry, "metadata", None), "get", lambda _: None)("topic")
            if topic:
                histogram[topic] = histogram.get(topic, 0) + 1
        return histogram

    def _desired_state(self, belief: BeliefState, profile: UserProfile) -> AffectiveState:
        # Encourage trust/safety when risk is high, otherwise lean on profile baseline.
        risk = _clamp01(belief.risk_level)
        desired_valence = _clamp01(profile.baseline_valence + (0.6 - belief.emotion_valence / 10.0))
        desired_arousal = _clamp01(profile.baseline_arousal - (belief.emotion_arousal - 5) / 20.0)
        desired_trust = _clamp01(0.55 + (profile.trust_sensitivity - 0.5) * 0.6 + risk * 0.25)
        desired_safety = _clamp01(0.7 + (profile.safety_bias - 0.5) * 0.4 + risk * 0.3)
        return AffectiveState(
            arousal=desired_arousal,
            valence=desired_valence,
            trust=desired_trust,
            safety=desired_safety,
        )

    @staticmethod
    def _alignment_score(predicted: AffectiveState, target: AffectiveState) -> float:
        diffs = [
            abs(predicted.arousal - target.arousal),
            abs(predicted.valence - target.valence),
            abs(predicted.trust - target.trust),
            abs(predicted.safety - target.safety),
        ]
        mean_diff = sum(diffs) / len(diffs)
        return max(0.0, 1.0 - mean_diff * 1.5)

    @staticmethod
    def _metadata_score(candidate: StrategyCandidate) -> float:
        raw = candidate.metadata.get("score")
        try:
            return float(raw) if raw is not None else 0.0
        except (TypeError, ValueError):
            return 0.0
