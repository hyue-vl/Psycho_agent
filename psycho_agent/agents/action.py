"""Constraint-based response generator selecting best simulated strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from ..config import settings
from ..llm import QwenLLM
from ..types import GlobalState, StrategyCandidate


ACTION_PROMPT = """你是负责输出治疗师回复的 Action Agent。
你会收到已选策略、草稿回复以及安全约束。
请润色草稿，使其更具同理心、安全合规，并符合 CBT 最佳实践。
务必明确共情地回应情绪、引用相关的应对工具，并邀请用户协作。"""


@dataclass
class ActionAgent:
    def __post_init__(self) -> None:
        self._llm = QwenLLM()

    def __call__(self, state: GlobalState) -> GlobalState:
        strategies = state.get("strategies", [])
        if not strategies:
            return state
        chosen = self._pick(strategies)
        prompt = self._build_prompt(chosen, state)
        messages = [
            {"role": "system", "content": ACTION_PROMPT},
            {"role": "user", "content": prompt},
        ]
        response = self._llm.chat(messages)
        updated = dict(state)
        updated["selected_strategy"] = chosen
        updated["final_response"] = response.strip()
        updated.setdefault("diagnostics", {})["action_prompt"] = prompt
        return updated

    def _pick(self, strategies: List[StrategyCandidate]) -> StrategyCandidate:
        best = max(
            strategies,
            key=lambda s: self._score(s.reward_vector or {}),
        )
        return best

    def _score(self, vector: Dict[str, float]) -> float:
        weights = settings.reward_weights
        return (
            vector.get("safety", 0) * weights.safety
            + vector.get("empathy", 0) * weights.empathy
            + vector.get("adherence", 0) * weights.adherence
            + vector.get("improvement", 0) * weights.improvement
        )

    def _build_prompt(self, strategy: StrategyCandidate, state: GlobalState) -> str:
        risk = state.get("risk_level", 0.0)
        guardrail = "Activate safety planning and escalation protocols." if risk >= settings.risk_threshold else ""
        return (
            f"Strategy: {strategy.label}\n"
            f"Draft: {strategy.draft_response}\n"
            f"Projected belief: {strategy.projected_belief}\n"
            f"Simulated user reaction: {strategy.projected_reaction}\n"
            f"Risk level: {risk}\n"
            f"{guardrail}"
        )
