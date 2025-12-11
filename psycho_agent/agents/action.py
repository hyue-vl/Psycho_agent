"""Constraint-based response generator selecting best planner strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..config import settings
from ..controller import WorldModelController
from ..llm import QwenLLM
from ..types import GlobalState, StrategyCandidate


ACTION_PROMPT = """你是负责输出治疗师回复的 Action Agent。
你会收到已选策略、草稿回复以及安全约束。
请润色草稿，使其更具同理心、安全合规，并符合 CBT 最佳实践。
务必明确共情地回应情绪、引用相关的应对工具，并邀请用户协作。"""


@dataclass
class ActionAgent:
    controller: WorldModelController | None = None

    def __post_init__(self) -> None:
        self._llm = QwenLLM()

    def __call__(self, state: GlobalState) -> GlobalState:
        strategies = state.get("strategies", [])
        if not strategies:
            return state
        user_id = state.get("user_id", "default")
        chosen, controller_diag = self._select_strategy(user_id, strategies, state)
        prompt = self._build_prompt(chosen, state)
        messages = [
            {"role": "system", "content": ACTION_PROMPT},
            {"role": "user", "content": prompt},
        ]
        response = self._llm.chat(messages)
        updated = dict(state)
        updated["selected_strategy"] = chosen
        updated["final_response"] = response.strip()
        diagnostics = updated.setdefault("diagnostics", {})
        diagnostics["action_prompt"] = prompt
        if controller_diag:
            diagnostics["controller"] = controller_diag
        return updated

    def _select_strategy(
        self,
        user_id: str,
        strategies: List[StrategyCandidate],
        state: GlobalState,
    ) -> tuple[StrategyCandidate, dict | None]:
        if self._controller:
            try:
                return self._controller.select(user_id, strategies, state)
            except ValueError:
                # Fallback to simple scoring if the world model cannot be constructed.
                pass
        return self._fallback_pick(strategies), None

    def _fallback_pick(self, strategies: List[StrategyCandidate]) -> StrategyCandidate:
        return max(strategies, key=self._combined_score)

    def _combined_score(self, strategy: StrategyCandidate) -> float:
        reward_score = self._score_reward(strategy.reward_vector)
        if reward_score is not None:
            return reward_score
        return self._metadata_score(strategy.metadata)

    def _score_reward(self, vector: Optional[Dict[str, float]]) -> Optional[float]:
        if not vector:
            return None
        weights = settings.reward_weights
        return (
            vector.get("safety", 0) * weights.safety
            + vector.get("empathy", 0) * weights.empathy
            + vector.get("adherence", 0) * weights.adherence
            + vector.get("improvement", 0) * weights.improvement
        )

    @staticmethod
    def _metadata_score(metadata: Dict[str, Any]) -> float:
        raw = metadata.get("score")
        try:
            return float(raw)
        except (TypeError, ValueError):
            return 0.0

    def _build_prompt(self, strategy: StrategyCandidate, state: GlobalState) -> str:
        risk = state.get("risk_level", 0.0)
        guardrail = "Activate safety planning and escalation protocols." if risk >= settings.risk_threshold else ""
        thought_chain = " -> ".join(strategy.metadata.get("thought_chain", []))
        planner_score = strategy.metadata.get("score")
        belief = state.get("belief_state")
        return (
            f"Strategy: {strategy.label}\n"
            f"Draft: {strategy.draft_response}\n"
            f"Planner rationale: {strategy.rationale}\n"
            f"Thought chain: {thought_chain or 'n/a'}\n"
            f"Planner score: {planner_score}\n"
            f"Observed belief: {belief}\n"
            f"Risk level: {risk}\n"
            f"{guardrail}"
        )
