"""LangGraph orchestration of the Psycho-World multi-agent workflow."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

try:  # pragma: no cover - optional dependency for tests
    from langgraph.graph import END, StateGraph
except ImportError:  # pragma: no cover
    END = "__END__"

    class _CompiledGraph:
        def __init__(self, nodes, edges, entry):
            self._nodes = nodes
            self._edges = edges
            self._entry = entry

        def invoke(self, state):
            if self._entry is None:
                raise RuntimeError("Entry point not set for fallback graph")
            current = self._entry
            payload = state
            visited = 0
            limit = len(self._nodes) + 2
            while current != END:
                node = self._nodes.get(current)
                if node is None:
                    raise RuntimeError(f"Undefined node '{current}' in fallback graph")
                payload = node(payload)
                current = self._edges.get(current, END)
                visited += 1
                if visited > limit:
                    raise RuntimeError("Fallback graph detected a cycle")
            return payload

    class StateGraph:  # type: ignore[override]
        def __init__(self, _state_type):
            self._nodes = {}
            self._edges = {}
            self._entry = None

        def add_node(self, name, fn):
            self._nodes[name] = fn

        def set_entry_point(self, name):
            self._entry = name

        def add_edge(self, start, end):
            self._edges[start] = end

        def compile(self):
            return _CompiledGraph(self._nodes, self._edges, self._entry)

from .agents import ActionAgent, AffectiveStateMachine, PerceptionAgent, PlanningAgent
from .config import settings
from .knowledge import COKEKGraph
from .memory import AMemMemoryManager, MemGPTManager
from .types import GlobalState, KnowledgeContext, UserProfile
from .vectorstore import BGEVectorStore

LOGGER = logging.getLogger(__name__)


class PsychoWorldGraph:
    """High-level façade that wires the MAS nodes together with LangGraph."""

    def __init__(
        self,
        *,
        memory_manager: AMemMemoryManager | None = None,
        vector_store: BGEVectorStore | None = None,
        knowledge_graph: COKEKGraph | None = None,
        perception: PerceptionAgent | None = None,
        planning: PlanningAgent | None = None,
        action: ActionAgent | None = None,
        state_machine: AffectiveStateMachine | None = None,
    ) -> None:
        self._memory_manager = memory_manager or AMemMemoryManager()
        self._vector_store = vector_store or BGEVectorStore()
        self._knowledge_graph = knowledge_graph or COKEKGraph()
        self._perception = perception or PerceptionAgent()
        self._planning = planning or PlanningAgent()
        self._action = action or ActionAgent()
        self._state_machine = state_machine or AffectiveStateMachine()
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(GlobalState)
        graph.add_node("memory", self._memory_node)
        graph.add_node("perception", self._perception_node)
        graph.add_node("planning", self._planning_node)
        graph.add_node("action", self._action_node)
        graph.set_entry_point("memory")
        graph.add_edge("memory", "perception")
        graph.add_edge("perception", "planning")
        graph.add_edge("planning", "action")
        graph.add_edge("action", END)
        return graph.compile()

    @property
    def memory_manager(self) -> MemGPTManager:
        """Expose the memory manager to allow external session orchestration."""
        return self._memory_manager

    def invoke(
        self,
        user_input: str,
        user_id: str = "default",
        user_profile: Optional[UserProfile] = None,
    ) -> Dict[str, Any]:
        initial_state: GlobalState = {
            "user_input": user_input,
            "user_id": user_id,
            "diagnostics": {},
        }
        if user_profile is not None:
            initial_state["user_profile"] = user_profile
            self._state_machine.set_profile(user_id, user_profile)
        initial_state["affective_state"] = self._state_machine.get_state(user_id)
        return self._graph.invoke(initial_state)

    # Node implementations -------------------------------------------------

    def _memory_node(self, state: GlobalState) -> GlobalState:
        user_id = state.get("user_id", "default")
        retrieved = self._vector_store.search(state["user_input"], settings.recall_top_k)
        rag_snippets = [doc.text for doc in retrieved]
        mem = self._memory_manager.load_context(
            user_id,
            settings.recall_top_k,
            fallback_snippets=rag_snippets,
            query=state["user_input"],
        )
        self._memory_manager.recall_append(
            user_id,
            [
                {
                    "role": "user",
                    "content": state["user_input"],
                    "metadata": {"topic": "latest_query"},
                }
            ],
        )
        working = [{"role": slice.role, "content": slice.content} for slice in mem["working"]]
        recall = mem["recall"]
        archival = mem["archival"]
        knowledge = KnowledgeContext(
            coke_paths=[],
            rag_snippets=rag_snippets,
            kb_metadata={},
        )
        updated = dict(state)
        updated["working_context"] = working
        updated["recall_memory"] = recall
        updated["archival_memory"] = archival
        updated["knowledge_context"] = knowledge
        profile = state.get("user_profile")
        if profile is not None:
            self._state_machine.set_profile(user_id, profile)
        updated.setdefault("affective_state", self._state_machine.get_state(user_id))
        return updated

    def _perception_node(self, state: GlobalState) -> GlobalState:
        updated = self._perception(state)
        belief = updated.get("belief_state")
        if belief:
            summary = {
                "last_intent": belief.communicative_intent,
                "last_distortions": ", ".join(belief.distortions),
                "risk_level": str(belief.risk_level),
            }
            self._memory_manager.core_memory_replace(state.get("user_id", "default"), summary)
        user_id = state.get("user_id", "default")
        if belief:
            posterior = self._state_machine.update_from_observation(user_id, belief)
            updated["affective_state"] = posterior
            diagnostics = updated.setdefault("diagnostics", {})
            sm_diag = diagnostics.setdefault("affective_state", {})
            sm_diag["posterior"] = posterior.to_dict()
        if updated["risk_level"] >= settings.risk_threshold:
            LOGGER.warning("High risk detected; ensure safety escalation outside the agent.")
        return updated

    def _planning_node(self, state: GlobalState) -> GlobalState:
        belief = state["belief_state"]
        knowledge = state.get("knowledge_context")
        if self._knowledge_graph and knowledge and belief.distortions:
            coke_paths = self._knowledge_graph.fetch_paths(
                situation=state["user_input"],
                belief=belief.distortions[0],
            )
            knowledge.coke_paths = coke_paths
        return self._planning(state)

    def _action_node(self, state: GlobalState) -> GlobalState:
        updated = self._action(state)
        final_response = updated.get("final_response")
        if final_response:
            user_id = state.get("user_id", "default")
            self._memory_manager.recall_append(
                user_id,
                [
                    {
                        "role": "assistant",
                        "content": final_response,
                        "metadata": {"topic": "agent_reply", "tags": ["assistant", "response"]},
                    }
                ],
            )
        user_id = state.get("user_id", "default")
        predicted = self._state_machine.predict_transition(
            user_id,
            updated.get("selected_strategy"),
            updated.get("belief_state"),
        )
        updated["affective_state"] = predicted
        diagnostics = updated.setdefault("diagnostics", {})
        sm_diag = diagnostics.setdefault("affective_state", {})
        sm_diag["predicted_next"] = predicted.to_dict()
        return updated

    # Optional utility -----------------------------------------------------

    def ingest_memory(self, text: str, metadata: Dict[str, Any]) -> None:
        """Add a document to the local vector store for recall."""
        self._vector_store.add(text, metadata)
