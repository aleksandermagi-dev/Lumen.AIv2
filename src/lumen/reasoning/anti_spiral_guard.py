from __future__ import annotations

from lumen.reasoning.pipeline_models import (
    ConversationAwarenessResult,
    DialogueManagementResult,
    StateInterpretationResult,
)


class AntiSpiralGuard:
    """Standalone anti-spiral safety backbone for state-control decisions."""

    def evaluate(
        self,
        *,
        prompt: str,
        dialogue_management: DialogueManagementResult,
        conversation_awareness: ConversationAwarenessResult,
        state_interpretation: StateInterpretationResult,
    ) -> tuple[bool, str | None]:
        normalized = " ".join(str(prompt).lower().split())
        if state_interpretation.repeated_failure_detected:
            return True, "Repeated blockers detected, so the response should slow down and return to clarity."
        if state_interpretation.uncertainty_stacking:
            return True, "Uncertainty is stacking, so the response should pause escalation and clarify what is missing."
        if (
            conversation_awareness.recent_intent_pattern == "hesitating"
            and conversation_awareness.conversation_momentum == "doubting"
        ):
            return True, "The current turn is wobbling, so the response should shift into grounded problem-solving."
        if "not enough information" in normalized or "don't know" in normalized:
            return True, "The turn already points to missing information, so the response should stay explicit and non-escalatory."
        return False, None
