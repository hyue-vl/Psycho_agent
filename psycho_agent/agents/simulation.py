"""Graph-constrained world simulator that predicts user reactions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List

from ..config import settings
from ..knowledge import COKEKGraph
from ..llm import QwenLLM
from ..types import BeliefState, GlobalState, StrategyCandidate

SIM_PROMPT = """You are a user simulator. Given the therapist's strategy and draft reply,
simulate the user's inner monologue and outward response using first-person voice.
Return JSON with:
- projected_distortions: list
- projected_emotion_valence (1-10)
- projected_emotion_arousal (1-10)
- projected_risk (0-1)
- reaction: natural language reply
- justification: why this reaction follows the COKE knowledge graph edges provided."""


@dataclass
class SimulationAgent:
    lookahead: int = 1

    def __post_init__(self) -> None:
        self._llm = QwenLLM()
        self._graph = COKEKGraph()

    def __call__(self, state: GlobalState) -> GlobalState:
        strategies: List[StrategyCandidate] = state.get("strategies", [])
        if not strategies:
            return state
        belief = state["belief_state"]
        knowledge = state.get("knowledge_context")
        enriched = []
        for strategy in strategies:
            prompt = self._build_prompt(belief, strategy, knowledge)
            messages = [
                {"role": "system", "content": SIM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            raw = self._llm.chat(messages)
            enriched.append(self._project(strategy, raw))
        updated = dict(state)
        updated["strategies"] = enriched
        updated.setdefault("diagnostics", {})["simulation_outputs"] = [s.reward_vector for s in enriched]
        return updated

    def _build_prompt(self, belief: BeliefState, strategy: StrategyCandidate, knowledge) -> str:
        coke = "\n".join(getattr(knowledge, "coke_paths", [])) if knowledge else ""
        return (
            f"Current belief distortions: {belief.distortions}\n"
            f"Strategy: {strategy.label}\n"
            f"Draft reply: {strategy.draft_response}\n"
            f"COKE graph constraints:\n{coke}"
        )

    def _project(self, strategy: StrategyCandidate, raw: str) -> StrategyCandidate:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {}
        projected = BeliefState(
            distortions=data.get("projected_distortions", strategy.rationale.split()),
            emotion_valence=int(data.get("projected_emotion_valence", 5)),
            emotion_arousal=int(data.get("projected_emotion_arousal", 5)),
            communicative_intent="predicted",
            risk_level=float(data.get("projected_risk", 0.0)),
            rationale=data.get("justification", ""),
        )
        reward = {
            "safety": 1 - projected.risk_level,
            "empathy": 1 - abs(projected.emotion_valence - 7) / 7,
            "adherence": 0.8,
            "improvement": 1 - len(projected.distortions) / 5,
        }
        strategy.projected_belief = projected
        strategy.reward_vector = reward
        strategy.projected_reaction = data.get("reaction")
        return strategy
