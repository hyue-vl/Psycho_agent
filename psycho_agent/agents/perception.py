"""Inference-only cognitive encoder using DoT prompting + self-consistency."""

from __future__ import annotations

from typing import List

from ..llm import QwenLLM, SelfConsistencyRunner
from ..types import BeliefState, GlobalState

DOT_PROMPT = """You are a clinical psychologist performing Diagnosis of Thought (DoT).
Analyse the user's latest message and the provided conversation summary.
Return a JSON object with the keys:
- cognitive_distortions: list of CBT distortion labels
- emotion_valence: integer 1-10 (10 positive)
- emotion_arousal: integer 1-10 (10 intense)
- communicative_intent: short phrase for latent intent
- risk_level: 0-1 float representing crisis likelihood
- rationale: 2-3 sentence explanation
Ensure valid JSON."""


class PerceptionAgent:
    """Runs multi-pass DoT extraction and updates the belief state."""

    def __init__(self, passes: int = 3) -> None:
        self._llm = QwenLLM()
        self._self_consistency = SelfConsistencyRunner(passes=passes)

    def __call__(self, state: GlobalState) -> GlobalState:
        user_input = state["user_input"]
        history = state.get("working_context", [])
        context_summary = "\n".join(f"{turn['role']}: {turn['content']}" for turn in history[-6:])

        def _run(_: int) -> str:
            messages = [
                {"role": "system", "content": DOT_PROMPT},
                {"role": "user", "content": f"History:\n{context_summary}\n\nCurrent:\n{user_input}"},
            ]
            return self._llm.chat(messages)

        result = self._self_consistency.run(_run)
        belief = BeliefState(
            distortions=result.get("cognitive_distortions", []),
            emotion_valence=int(result.get("emotion_valence", 5)),
            emotion_arousal=int(result.get("emotion_arousal", 5)),
            communicative_intent=result.get("communicative_intent", "unknown"),
            risk_level=float(result.get("risk_level", 0.0)),
            rationale=result.get("rationale", ""),
        )
        updated_state = dict(state)
        updated_state["belief_state"] = belief
        updated_state["risk_level"] = belief.risk_level
        diagnostics = updated_state.setdefault("diagnostics", {})
        diagnostics["perception"] = result
        return updated_state
