"""Tree-of-Thought planner that selects therapy strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..config import settings
from ..knowledge import COKEKGraph
from ..llm import QwenLLM
from ..types import BeliefState, GlobalState, StrategyCandidate


SYSTEM_PROMPT = """You are the Planning Agent inside a CBT therapy assistant.
Given the user's belief state and retrieved knowledge, output candidate strategies.
Strategies must be sampled from: Empathic Reflection, Socratic Questioning, Cognitive Restructuring,
Behavioural Activation, Distress Tolerance, Mindfulness Grounding, Safety Planning.
For each strategy supply rationale and a short draft response."""


@dataclass
class PlanningAgent:
    branches: int = settings.planner.max_branches
    depth: int = settings.planner.search_depth

    def __post_init__(self) -> None:
        self._llm = QwenLLM()
        self._graph = COKEKGraph()

    def __call__(self, state: GlobalState) -> GlobalState:
        belief: BeliefState = state["belief_state"]
        knowledge = state.get("knowledge_context")
        prompt = self._build_prompt(belief, knowledge)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        raw = self._llm.chat(messages)
        strategies = self._parse(raw, belief)
        updated = dict(state)
        updated["strategies"] = strategies
        updated.setdefault("diagnostics", {})["planning_prompt"] = prompt
        return updated

    def _build_prompt(self, belief: BeliefState, knowledge) -> str:
        rag = "\n".join(getattr(knowledge, "rag_snippets", [])) if knowledge else ""
        coke = "\n".join(getattr(knowledge, "coke_paths", [])) if knowledge else ""
        distortions = ", ".join(belief.distortions) or "unspecified"
        interventions = []
        if belief.distortions:
            try:
                interventions = self._graph.interventions_for_distortion(belief.distortions[0])
            except Exception:
                interventions = []
        interventions_text = ", ".join(interventions) if interventions else "Follow CBT playbook."
        return (
            f"Distortions: {distortions}\n"
            f"Intent: {belief.communicative_intent}\n"
            f"Valence: {belief.emotion_valence}, Arousal: {belief.emotion_arousal}\n"
            f"Risk: {belief.risk_level}\n"
            f"COKE paths:\n{coke}\n"
            f"Supporting RAG:\n{rag}\n"
            f"Recommended interventions: {interventions_text}\n"
            f"Please produce {self.branches} strategies with rationales and draft replies."
        )

    def _parse(self, raw: str, belief: BeliefState) -> List[StrategyCandidate]:
        blocks = [block.strip() for block in raw.split("\n\n") if block.strip()]
        candidates: List[StrategyCandidate] = []
        for block in blocks[: self.branches]:
            lines = block.splitlines()
            header = lines[0] if lines else "Strategy"
            label = header.split(":", 1)[-1].strip() if ":" in header else header.strip()
            rationale = next((line.split(":", 1)[1].strip() for line in lines if line.lower().startswith("rationale")), "")
            draft = next((line.split(":", 1)[1].strip() for line in lines if line.lower().startswith("draft")), "")
            candidates.append(StrategyCandidate(label=label, rationale=rationale, draft_response=draft))
        if not candidates:
            candidates.append(
                StrategyCandidate(
                    label="Empathic Reflection",
                    rationale="Fallback strategy when planner response cannot be parsed.",
                    draft_response="It sounds like this feels heavy; I'm here with you.",
                )
            )
        return candidates
