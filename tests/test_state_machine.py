from psycho_agent.agents.state_machine import AffectiveStateMachine
from psycho_agent.types import BeliefState, StrategyCandidate, UserProfile


def _belief(valence: int = 8, arousal: int = 7, risk: float = 0.2) -> BeliefState:
    return BeliefState(
        distortions=["catastrophizing"],
        emotion_valence=valence,
        emotion_arousal=arousal,
        communicative_intent="seek reassurance",
        risk_level=risk,
        rationale="stub",
    )


def test_observation_update_moves_state_toward_measurement():
    machine = AffectiveStateMachine(measurement_trust=0.8)
    user_id = "u-observe"
    baseline = machine.get_state(user_id)
    posterior = machine.update_from_observation(user_id, _belief(valence=9, arousal=9))
    assert posterior.valence > baseline.valence
    assert posterior.arousal > baseline.arousal


def test_action_prediction_boosts_safety_planning_state():
    machine = AffectiveStateMachine()
    user_id = "u-safety"
    machine.set_profile(
        user_id,
        UserProfile(attachment_style="anxious", reactivity=0.8, trust_sensitivity=0.7, safety_bias=0.6),
    )
    belief = _belief(risk=0.4)
    posterior = machine.update_from_observation(user_id, belief)
    strategy = StrategyCandidate(label="Safety Planning", rationale="r", draft_response="d")
    predicted = machine.predict_transition(user_id, strategy, belief)
    assert predicted.safety >= posterior.safety
    for value in predicted.to_dict().values():
        assert 0.0 <= value <= 1.0
