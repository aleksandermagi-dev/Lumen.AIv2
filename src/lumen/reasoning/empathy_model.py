from __future__ import annotations

from lumen.reasoning.conversation_policy import ConversationPolicy
from lumen.reasoning.pipeline_models import (
    ConversationAwarenessResult,
    EmpathyModelResult,
    StateControlResult,
)


class EmpathyModel:
    """Structural, non-mirroring empathy layer for interpreting emotional signal and response sensitivity."""

    HEAVY_CUES = ("overwhelmed", "heavy", "drained", "stressed", "upset", "sad", "frustrated")
    CONFUSION_CUES = ("confused", "lost", "not sure", "uncertain", "stuck")
    PRESSURE_CUES = ("urgent", "too much", "spiraling", "falling apart", "panic")

    def assess(
        self,
        *,
        prompt: str,
        conversation_awareness: ConversationAwarenessResult,
        state_control: StateControlResult | None,
    ) -> EmpathyModelResult:
        normalized = " ".join(str(prompt).strip().lower().split())
        feeling_label = None
        probable_cause = None
        response_sensitivity = "normal"

        if any(cue in normalized for cue in self.PRESSURE_CUES):
            feeling_label = "overloaded"
            probable_cause = "pressure or destabilization signal"
            response_sensitivity = "stabilizing"
        elif any(cue in normalized for cue in self.HEAVY_CUES):
            feeling_label = "heavy"
            probable_cause = "emotional or cognitive load"
            response_sensitivity = "gentle"
        elif any(cue in normalized for cue in self.CONFUSION_CUES):
            feeling_label = "uncertain"
            probable_cause = "confusion or ambiguity"
            response_sensitivity = "careful"
        elif conversation_awareness.recent_intent_pattern == "hesitating":
            feeling_label = "hesitant"
            probable_cause = "hesitation around the current line of thought"
            response_sensitivity = "careful"
        elif state_control is not None and state_control.anti_spiral_active:
            feeling_label = None
            probable_cause = "stacking uncertainty or blockers"
            response_sensitivity = "stabilizing"

        emotional_signal_detected = feeling_label is not None or probable_cause is not None
        grounded_acknowledgment = None
        if emotional_signal_detected:
            grounded_acknowledgment = ConversationPolicy.grounded_emotional_acknowledgment(
                feeling_label=feeling_label
            )

        return EmpathyModelResult(
            emotional_signal_detected=emotional_signal_detected,
            feeling_label=feeling_label,
            probable_cause=probable_cause,
            response_sensitivity=response_sensitivity,
            grounded_acknowledgment=grounded_acknowledgment,
        )
