"""Tree-of-Thought planner that selects therapy strategies."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List

from ..config import settings
from ..knowledge import COKEKGraph
from ..llm import QwenLLM
from ..types import BeliefState, GlobalState, StrategyCandidate


PLAN_SYSTEM_PROMPT = """You are the Planning Agent inside a CBT therapy assistant.
Always run a Tree-of-Thought search with maximum depth {depth} to explore at least {branches} distinct strategies.
Valid strategies are limited to:
- Empathic Reflection
- Socratic Questioning
- Cognitive Restructuring
- Behavioural Activation
- Distress Tolerance
- Mindfulness Grounding
- Safety Planning
Respond ONLY with JSON. The top-level value must be a list where each item contains:
{{
  "strategy": "<one of the valid strategies>",
  "thought_chain": ["step 1", "..."],
  "rationale": "<why this branch is promising>",
  "draft_response": "<short therapist utterance>",
  "score": 0-1 float capturing expected reward
}}"""


@dataclass
class PlanningAgent:
    branches: int = settings.planner.max_branches
    depth: int = settings.planner.search_depth

    def __post_init__(self) -> None:
        self._planner_llm = QwenLLM(
            model=settings.qwen.planner_model,
            temperature=0.35,
            max_output_tokens=1024,
        )
        self._graph = COKEKGraph()

    def __call__(self, state: GlobalState) -> GlobalState:
        belief: BeliefState = state["belief_state"]
        knowledge = state.get("knowledge_context")
        prompt = self._build_prompt(belief, knowledge)
        messages = [
            {
                "role": "system",
                "content": PLAN_SYSTEM_PROMPT.format(depth=self.depth, branches=self.branches),
            },
            {"role": "user", "content": prompt},
        ]
        raw = self._planner_llm.chat(messages)
        tree = self._parse_tree(raw)
        strategies = self._to_candidates(tree)
        updated = dict(state)
        updated["strategies"] = strategies
        diagnostics = updated.setdefault("diagnostics", {})
        diagnostics["planning_prompt"] = prompt
        diagnostics["planning_tree"] = tree
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
            f"Tree-of-Thought search budget: depth<= {self.depth}, branches={self.branches}.\n"
            "Return detailed thought chains before selecting final drafts."
        )
 
    def _parse_tree(self, raw: str) -> List[Dict[str, Any]]:
        parsed = self._try_json(raw)
        if parsed:
            return parsed
        return self._fallback_nodes(raw)

    def _try_json(self, raw: str) -> List[Dict[str, Any]]:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return []
        if isinstance(data, dict) and "candidates" in data:
            data = data["candidates"]
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            return []
        nodes: List[Dict[str, Any]] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            nodes.append(
                {
                    "strategy": item.get("strategy") or item.get("label", ""),
                    "thought_chain": item.get("thought_chain") or item.get("chain") or item.get("steps") or [],
                    "rationale": item.get("rationale") or item.get("reasoning", ""),
                    "draft_response": item.get("draft_response") or item.get("draft", ""),
                    "score": _as_float(item.get("score")),
                }
            )
        return [node for node in nodes if node["strategy"]]

    def _fallback_nodes(self, raw: str) -> List[Dict[str, Any]]:
        blocks = [block.strip() for block in raw.split("\n\n") if block.strip()]
        nodes: List[Dict[str, Any]] = []
        for block in blocks[: self.branches]:
            lines = block.splitlines()
            if not lines:
                continue
            header = lines[0]
            label = header.split(":", 1)[-1].strip() if ":" in header else header.strip()
            rationale = next(
                (line.split(":", 1)[1].strip() for line in lines if line.lower().startswith("rationale")),
                "",
            )
            draft = next(
                (line.split(":", 1)[1].strip() for line in lines if line.lower().startswith("draft")),
                "",
            )
            chain = [line.strip("- •") for line in lines[1 : 1 + self.depth] if line.strip()]
            nodes.append(
                {
                    "strategy": label,
                    "thought_chain": chain,
                    "rationale": rationale or " -> ".join(chain),
                    "draft_response": draft,
                    "score": None,
                }
            )
        return nodes

    def _to_candidates(self, tree: List[Dict[str, Any]]) -> List[StrategyCandidate]:
        if not tree:
            return [
                StrategyCandidate(
                    label="Empathic Reflection",
                    rationale="Fallback strategy when planner response cannot be parsed.",
                    draft_response="It sounds like this feels heavy; I'm here with you.",
                    metadata={"thought_chain": ["Fallback branch"]},
                )
            ]
        candidates: List[StrategyCandidate] = []
        for node in tree[: self.branches]:
            label = node.get("strategy", "Strategy")
            rationale = node.get("rationale") or " -> ".join(node.get("thought_chain", []))
            draft = node.get("draft_response") or "I'm here with you."
            metadata = {
                "thought_chain": node.get("thought_chain", []),
                "score": node.get("score"),
            }
            candidates.append(
                StrategyCandidate(
                    label=label,
                    rationale=rationale,
                    draft_response=draft,
                    metadata=metadata,
                )
            )
        return candidates


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
