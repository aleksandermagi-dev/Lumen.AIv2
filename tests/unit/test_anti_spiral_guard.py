from lumen.reasoning.anti_spiral_guard import AntiSpiralGuard
from lumen.reasoning.pipeline_models import (
    ConversationAwarenessResult,
    DialogueManagementResult,
    StateInterpretationResult,
)


def test_anti_spiral_guard_is_callable_independent_of_tone() -> None:
    guard = AntiSpiralGuard()

    active, reason = guard.evaluate(
        prompt="maybe this works, but I do not know yet",
        dialogue_management=DialogueManagementResult(
            interaction_mode="analytical",
            idea_state="uncertain",
            response_strategy="ask_question",
        ),
        conversation_awareness=ConversationAwarenessResult(
            recent_intent_pattern="hesitating",
            conversation_momentum="doubting",
            unresolved_thread_open=True,
        ),
        state_interpretation=StateInterpretationResult(
            trigger="confusion",
            trigger_signals=["hesitation"],
            repeated_failure_detected=False,
            uncertainty_stacking=True,
            humor_candidate=False,
        ),
    )

    assert active is True
    assert reason is not None
    assert "uncertainty" in reason.lower() or "grounded" in reason.lower()


def test_anti_spiral_guard_activates_from_pattern_not_emotion_label() -> None:
    guard = AntiSpiralGuard()

    active, reason = guard.evaluate(
        prompt="continue",
        dialogue_management=DialogueManagementResult(
            interaction_mode="analytical",
            idea_state="refining",
            response_strategy="answer",
        ),
        conversation_awareness=ConversationAwarenessResult(
            recent_intent_pattern="building",
            conversation_momentum="building",
            unresolved_thread_open=True,
        ),
        state_interpretation=StateInterpretationResult(
            trigger="repeated_failure",
            trigger_signals=["failure"],
            repeated_failure_detected=True,
            uncertainty_stacking=False,
            humor_candidate=False,
        ),
    )

    assert active is True
    assert reason == "Repeated blockers detected, so the response should slow down and return to clarity."
