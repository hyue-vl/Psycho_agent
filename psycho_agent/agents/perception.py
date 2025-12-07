"""Inference-only cognitive encoder using DoT prompting + self-consistency."""

from __future__ import annotations

from typing import List

from ..llm import QwenLLM, SelfConsistencyRunner
from ..types import BeliefState, GlobalState

DOT_PROMPT = """你是一名执行思维诊断（DoT, Diagnosis of Thought）的临床心理学家。
请分析用户最新消息以及提供的对话摘要。
只返回一个包含以下键的 JSON：
- cognitive_distortions：列出对应的 CBT 认知扭曲标签
- emotion_valence：1-10 的整数（10 表示高度积极）
- emotion_arousal：1-10 的整数（10 表示情绪强烈）
- communicative_intent：一句话概括潜在意图
- risk_level：0-1 的浮点数，代表危机发生的可能性
- rationale：2-3 句中文解释，说明推理依据
务必输出有效 JSON。"""


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
