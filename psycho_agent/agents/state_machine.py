"""Continuous affective state machine implementing belief transitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from ..types import AffectiveState, BeliefState, StrategyCandidate, UserProfile


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _scale_emotion(value: int) -> float:
    return _clamp(value / 10.0)


@dataclass(slots=True)
class TransitionDiagnostics:
    posterior: Optional[AffectiveState] = None
    prediction: Optional[AffectiveState] = None

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        payload: Dict[str, Dict[str, float]] = {}
        if self.posterior:
            payload["posterior"] = self.posterior.to_dict()
        if self.prediction:
            payload["predicted_next"] = self.prediction.to_dict()
        return payload


class AffectiveStateMachine:
    """Implements s_{t+1} = s_t + αΔ_react + βM_profile with filtering."""

    ACTION_SIGNATURES: Dict[str, Dict[str, float]] = {
        "Empathic Reflection": {"arousal": -0.06, "valence": 0.10, "trust": 0.08, "safety": 0.04},
        "Socratic Questioning": {"arousal": 0.00, "valence": 0.05, "trust": 0.02, "safety": 0.0},
        "Cognitive Restructuring": {"arousal": -0.02, "valence": 0.08, "trust": 0.04, "safety": 0.02},
        "Behavioural Activation": {"arousal": 0.04, "valence": 0.06, "trust": 0.03, "safety": 0.0},
        "Distress Tolerance": {"arousal": -0.08, "valence": 0.04, "trust": 0.05, "safety": 0.08},
        "Mindfulness Grounding": {"arousal": -0.1, "valence": 0.03, "trust": 0.04, "safety": 0.06},
        "Safety Planning": {"arousal": -0.02, "valence": 0.04, "trust": 0.06, "safety": 0.12},
        "__default__": {"arousal": -0.01, "valence": 0.04, "trust": 0.03, "safety": 0.03},
    }

    ATTACHMENT_BIASES: Dict[str, Dict[str, float]] = {
        "secure": {"trust": 0.02, "safety": 0.01},
        "anxious": {"arousal": 0.06, "trust": -0.03},
        "avoidant": {"valence": -0.02, "trust": -0.05, "safety": 0.02},
        "disorganized": {"arousal": 0.08, "safety": -0.05},
    }

    ACTION_PROFILE_GAIN: Dict[str, float] = {
        "Empathic Reflection": 0.7,
        "Socratic Questioning": 0.5,
        "Cognitive Restructuring": 0.6,
        "Behavioural Activation": 0.65,
        "Distress Tolerance": 0.55,
        "Mindfulness Grounding": 0.6,
        "Safety Planning": 0.8,
        "__default__": 0.5,
    }

    def __init__(
        self,
        alpha: float = 0.35,
        beta: float = 0.25,
        measurement_trust: float = 0.6,
        decay: float = 0.05,
    ) -> None:
        self._alpha = alpha
        self._beta = beta
        self._measurement_trust = measurement_trust
        self._decay = decay
        self._states: Dict[str, AffectiveState] = {}
        self._last_action: Dict[str, str] = {}
        self._profiles: Dict[str, UserProfile] = {}
        self._diagnostics: Dict[str, TransitionDiagnostics] = {}

    def set_profile(self, user_id: str, profile: Optional[UserProfile]) -> None:
        if profile is None:
            return
        self._profiles[user_id] = profile

    def get_state(self, user_id: str) -> AffectiveState:
        if user_id not in self._states:
            profile = self._profiles.get(user_id, UserProfile())
            self._states[user_id] = self._seed_state(profile)
        return self._states[user_id]

    def update_from_observation(self, user_id: str, observation: Optional[BeliefState]) -> AffectiveState:
        if observation is None:
            return self.get_state(user_id)
        prior = self.get_state(user_id)
        measurement = self._belief_to_state(observation)
        weight = self._measurement_weight(observation)
        posterior = self._blend(prior, measurement, weight).clamp()
        self._states[user_id] = posterior
        diag = self._diagnostics.setdefault(user_id, TransitionDiagnostics())
        diag.posterior = posterior
        return posterior

    def predict_transition(
        self,
        user_id: str,
        action: Optional[StrategyCandidate],
        observation: Optional[BeliefState] = None,
    ) -> AffectiveState:
        state = self.update_from_observation(user_id, observation)
        label = (action.label if action else self._last_action.get(user_id)) or "generic"
        profile = self._profiles.get(user_id, UserProfile())
        delta = self._delta_react(state, label, observation)
        profile_adjust = self._profile_matrix(profile, label)
        next_state = self._apply_delta(state, delta, profile_adjust).clamp()
        self._states[user_id] = next_state
        if action:
            self._last_action[user_id] = label
        diag = self._diagnostics.setdefault(user_id, TransitionDiagnostics())
        diag.prediction = next_state
        return next_state

    def diagnostics(self, user_id: str) -> Dict[str, Dict[str, float]]:
        diag = self._diagnostics.get(user_id)
        return diag.to_dict() if diag else {}

    def _seed_state(self, profile: UserProfile) -> AffectiveState:
        return AffectiveState(
            arousal=_clamp(profile.baseline_arousal),
            valence=_clamp(profile.baseline_valence),
            trust=_clamp(0.5 + (profile.trust_sensitivity - 0.5) * 0.2),
            safety=_clamp(0.5 + (profile.safety_bias - 0.5) * 0.2),
        )

    def _measurement_weight(self, observation: BeliefState) -> float:
        # Lower confidence when crisis risk is high.
        return _clamp(self._measurement_trust * (1.0 - 0.5 * observation.risk_level))

    def _blend(self, current: AffectiveState, measurement: AffectiveState, weight: float) -> AffectiveState:
        return AffectiveState(
            arousal=current.arousal + weight * (measurement.arousal - current.arousal),
            valence=current.valence + weight * (measurement.valence - current.valence),
            trust=current.trust + weight * (measurement.trust - current.trust),
            safety=current.safety + weight * (measurement.safety - current.safety),
        )

    def _belief_to_state(self, belief: BeliefState) -> AffectiveState:
        return AffectiveState(
            arousal=_scale_emotion(belief.emotion_arousal),
            valence=_scale_emotion(belief.emotion_valence),
            trust=_clamp(1.0 - 0.6 * belief.risk_level),
            safety=_clamp(1.0 - belief.risk_level),
        )

    def _delta_react(
        self,
        state: AffectiveState,
        action_label: str,
        observation: Optional[BeliefState],
    ) -> Dict[str, float]:
        signature = self.ACTION_SIGNATURES.get(action_label, self.ACTION_SIGNATURES["__default__"])
        arousal_target = _scale_emotion(observation.emotion_arousal) if observation else state.arousal
        valence_target = _scale_emotion(observation.emotion_valence) if observation else state.valence
        risk = observation.risk_level if observation else 0.0
        trust_penalty = _clamp(0.35 - 0.2 * signature["trust"], 0.1, 0.5)
        safety_penalty = _clamp(0.55 - 0.3 * signature["safety"], 0.15, 0.6)
        return {
            "arousal": (arousal_target - state.arousal) - self._decay * (state.arousal - 0.5) + signature["arousal"],
            "valence": (valence_target - state.valence) + signature["valence"],
            "trust": signature["trust"] - trust_penalty * risk,
            "safety": signature["safety"] - safety_penalty * risk,
        }

    def _profile_matrix(self, profile: UserProfile, action_label: str) -> Dict[str, float]:
        style = self.ATTACHMENT_BIASES.get(profile.attachment_style.lower(), {})
        gain = self.ACTION_PROFILE_GAIN.get(action_label, self.ACTION_PROFILE_GAIN["__default__"])
        return {
            "arousal": gain * ((profile.reactivity - 0.5) * 0.4 + style.get("arousal", 0.0)),
            "valence": gain * ((profile.baseline_valence - 0.5) * 0.3 + style.get("valence", 0.0)),
            "trust": gain * ((profile.trust_sensitivity - 0.5) * 0.5 + style.get("trust", 0.0)),
            "safety": gain * ((profile.safety_bias - 0.5) * 0.5 + style.get("safety", 0.0)),
        }

    def _apply_delta(
        self,
        state: AffectiveState,
        delta: Dict[str, float],
        profile_adjust: Dict[str, float],
    ) -> AffectiveState:
        return AffectiveState(
            arousal=state.arousal + self._alpha * delta["arousal"] + self._beta * profile_adjust["arousal"],
            valence=state.valence + self._alpha * delta["valence"] + self._beta * profile_adjust["valence"],
            trust=state.trust + self._alpha * delta["trust"] + self._beta * profile_adjust["trust"],
            safety=state.safety + self._alpha * delta["safety"] + self._beta * profile_adjust["safety"],
        )
