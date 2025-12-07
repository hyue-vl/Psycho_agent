"""Graph-constrained world simulator that predicts user reactions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List

from ..config import settings
from ..llm import QwenLLM
from ..types import BeliefState, GlobalState, StrategyCandidate

SIM_PROMPT = """You are a CBT user simulator that must obey the supplied therapy graph snippets.
Run a graph-constrained rollout with depth <= {depth}, narrating how the user moves through the chain.
Respond ONLY with JSON matching:
{{
  "thought_rollout": ["step explaining how graph edge is traversed", "..."],
  "projection": {{
    "projected_distortions": ["..."],
    "projected_emotion_valence": 1-10 integer,
    "projected_emotion_arousal": 1-10 integer,
    "projected_risk": 0-1 float,
    "reaction": "<first-person reply>",
    "justification": "<tie back to provided graph evidence>"
  }}
}}"""


@dataclass
class SimulationAgent:
    lookahead: int = 1

    def __post_init__(self) -> None:
        self._sim_llm = QwenLLM(
            model=settings.qwen.simulation_model,
            temperature=0.15,
            max_output_tokens=768,
        )

    def __call__(self, state: GlobalState) -> GlobalState:
        strategies: List[StrategyCandidate] = state.get("strategies", [])
        if not strategies:
            return state
        belief = state["belief_state"]
        knowledge = state.get("knowledge_context")
        enriched = []
        for strategy in strategies:
            prompt = self._build_prompt(belief, strategy, knowledge)
            system_prompt = SIM_PROMPT.format(depth=self.lookahead + 2)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            raw = self._sim_llm.chat(messages)
            enriched.append(self._project(strategy, raw))
        updated = dict(state)
        updated["strategies"] = enriched
        diagnostics = updated.setdefault("diagnostics", {})
        diagnostics["simulation_outputs"] = [s.reward_vector for s in enriched]
        diagnostics["simulation_rollouts"] = [
            s.metadata.get("simulation") for s in enriched if s.metadata.get("simulation")
        ]
        return updated

    def _build_prompt(self, belief: BeliefState, strategy: StrategyCandidate, knowledge) -> str:
        coke = "\n".join(getattr(knowledge, "coke_paths", [])) if knowledge else ""
        rag = "\n".join(getattr(knowledge, "rag_snippets", [])) if knowledge else ""
        thought_chain = " -> ".join(strategy.metadata.get("thought_chain", []))
        return (
            f"Current belief distortions: {belief.distortions}\n"
            f"Strategy: {strategy.label}\n"
            f"Draft reply: {strategy.draft_response}\n"
            f"Planned thought chain: {thought_chain or 'n/a'}\n"
            f"COKE knowledge constraints:\n{coke}\n"
            f"Supplemental evidence:\n{rag}\n"
            f"Simulate {self.lookahead} future turns and describe how the graph evidence shapes the user's reaction."
        )

    def _project(self, strategy: StrategyCandidate, raw: str) -> StrategyCandidate:
        data = self._safe_json(raw)
        projection = data.get("projection", data)
        rollout = _ensure_list(data.get("thought_rollout"))
        projected = BeliefState(
            distortions=projection.get("projected_distortions", strategy.rationale.split()),
            emotion_valence=int(projection.get("projected_emotion_valence", 5)),
            emotion_arousal=int(projection.get("projected_emotion_arousal", 5)),
            communicative_intent=projection.get("communicative_intent", "predicted"),
            risk_level=float(projection.get("projected_risk", 0.0)),
            rationale=projection.get("justification", ""),
        )
        reward = self._score_projection(projected)
        strategy.projected_belief = projected
        strategy.reward_vector = reward
        strategy.projected_reaction = projection.get("reaction")
        strategy.metadata.setdefault("simulation", {})
        strategy.metadata["simulation"].update(
            {
                "thought_rollout": rollout,
                "raw_projection": projection,
            }
        )
        return strategy

    @staticmethod
    def _safe_json(raw: str) -> dict:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _score_projection(self, projected: BeliefState) -> Dict[str, float]:
        improv = _clamp(1 - len(projected.distortions) / 5)
        empathy = _clamp(1 - abs(projected.emotion_valence - 7) / 7)
        adherence = _clamp(0.6 + 0.4 * (projected.emotion_arousal <= 7))
        safety = _clamp(1 - projected.risk_level)
        return {
            "safety": safety,
            "empathy": empathy,
            "adherence": adherence,
            "improvement": improv,
        }


def _ensure_list(value) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [value]
    return []


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))
