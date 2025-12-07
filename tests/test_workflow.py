from dataclasses import dataclass

from psycho_agent.types import BeliefState, StrategyCandidate
from psycho_agent.workflow import PsychoWorldGraph


@dataclass
class StubVectorStore:
    def add(self, text, metadata):
        return None

    def search(self, query, top_k=5):
        return []


@dataclass
class StubMemory:
    def load_context(self, user_id, top_k, fallback_snippets=None, query=None):
        return {"working": [], "recall": [], "archival": []}

    def recall_append(self, user_id, entries):
        return None

    def core_memory_replace(self, user_id, fields):
        return None


class StubGraph:
    def fetch_paths(self, situation, belief, limit=5):
        return []


class StubPerception:
    def __call__(self, state):
        state = dict(state)
        state["belief_state"] = BeliefState(
            distortions=["catastrophizing"],
            emotion_valence=3,
            emotion_arousal=8,
            communicative_intent="seek validation",
            risk_level=0.2,
            rationale="stub",
        )
        state["risk_level"] = 0.2
        return state


class StubPlanning:
    def __call__(self, state):
        state = dict(state)
        state["strategies"] = [
            StrategyCandidate(label="Empathic Reflection", rationale="stub", draft_response="draft reply")
        ]
        return state


class StubSimulation:
    def __call__(self, state):
        state = dict(state)
        for strategy in state["strategies"]:
            strategy.reward_vector = {"safety": 1.0, "empathy": 0.9, "adherence": 0.8, "improvement": 0.7}
            strategy.projected_belief = state["belief_state"]
        return state


class StubAction:
    def __call__(self, state):
        state = dict(state)
        state["final_response"] = "final"
        return state


def test_workflow_invocation_smoke():
    agent = PsychoWorldGraph(
        memory_manager=StubMemory(),
        vector_store=StubVectorStore(),
        knowledge_graph=StubGraph(),
        perception=StubPerception(),
        planning=StubPlanning(),
        simulation=StubSimulation(),
        action=StubAction(),
    )
    result = agent.invoke("I failed my exam")
    assert result["final_response"] == "final"
