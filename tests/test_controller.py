from psycho_agent.agents.state_machine import AffectiveStateMachine
from psycho_agent.controller import WorldModelController
from psycho_agent.types import AffectiveState, BeliefState, StrategyCandidate, UserProfile


def _make_state(belief: BeliefState, affective: AffectiveState | None = None, profile: UserProfile | None = None):
    state = {
        "belief_state": belief,
        "affective_state": affective or AffectiveState(arousal=0.7, valence=0.3, trust=0.4, safety=0.4),
        "user_profile": profile or UserProfile(),
    }
    return state


def test_controller_prefers_safety_strategy_when_risk_high():
    sm = AffectiveStateMachine()
    controller = WorldModelController(sm)
    belief = BeliefState(
        distortions=["fortune telling"],
        emotion_valence=2,
        emotion_arousal=9,
        communicative_intent="cry for help",
        risk_level=0.85,
        rationale="",
    )
    strategies = [
        StrategyCandidate(
            label="Safety Planning",
            rationale="ensure safety",
            draft_response="focus on safety",
            metadata={"score": 0.2},
        ),
        StrategyCandidate(
            label="Behavioural Activation",
            rationale="increase activity",
            draft_response="plan activities",
            metadata={"score": 0.9},
        ),
    ]
    selected, diagnostics = controller.select("user", strategies, _make_state(belief))
    assert selected.label == "Safety Planning"
    assert diagnostics["selected"]["label"] == "Safety Planning"
    assert diagnostics["world_model"]["desired_state"]["safety"] > 0.8


def test_controller_respects_metadata_when_risk_low():
    sm = AffectiveStateMachine()
    controller = WorldModelController(sm)
    belief = BeliefState(
        distortions=["all or nothing"],
        emotion_valence=6,
        emotion_arousal=4,
        communicative_intent="seek advice",
        risk_level=0.1,
        rationale="",
    )
    strategies = [
        StrategyCandidate(
            label="Behavioural Activation",
            rationale="",
            draft_response="plan steps",
            metadata={"score": 0.8},
        ),
        StrategyCandidate(
            label="Mindfulness Grounding",
            rationale="",
            draft_response="mindfulness",
            metadata={"score": 0.1},
        ),
    ]
    selected, diagnostics = controller.select("user", strategies, _make_state(belief))
    assert selected.label == "Behavioural Activation"
    assert diagnostics["selected"]["label"] == "Behavioural Activation"
