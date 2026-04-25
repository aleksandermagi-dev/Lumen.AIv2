from __future__ import annotations

from lumen.reasoning.pipeline_models import (
    ConversationAwarenessResult,
    DialogueManagementResult,
    StateInterpretationResult,
)


class StateInterpreter:
    """Interprets posture triggers before the state controller chooses output posture."""

    ABSURDITY_CUES = (
        "absurd",
        "ridiculous",
        "wild",
        "lol",
        "lmao",
        "haha",
        "wtf",
    )
    BREAKTHROUGH_CUES = (
        "that worked",
        "we got it",
        "breakthrough",
        "nice",
        "perfect",
        "finally",
        "it clicks",
    )
    FAILURE_CUES = (
        "error",
        "failed",
        "broken",
        "stuck",
        "not working",
        "confused",
        "blocker",
    )

    def interpret(
        self,
        *,
        prompt: str,
        dialogue_management: DialogueManagementResult,
        conversation_awareness: ConversationAwarenessResult,
        recent_interactions: list[dict[str, object]],
    ) -> StateInterpretationResult:
        normalized = " ".join(str(prompt).lower().split())
        trigger_signals: list[str] = []
        if any(cue in normalized for cue in self.BREAKTHROUGH_CUES):
            trigger_signals.append("breakthrough")
        if any(cue in normalized for cue in self.FAILURE_CUES):
            trigger_signals.append("failure")
        if any(cue in normalized for cue in self.ABSURDITY_CUES):
            trigger_signals.append("absurdity")
        if conversation_awareness.recent_intent_pattern == "hesitating":
            trigger_signals.append("hesitation")
        if dialogue_management.idea_state == "validated":
            trigger_signals.append("clarity")
        if dialogue_management.idea_state in {"exploring", "branching"}:
            trigger_signals.append("exploration")

        repeated_failure_detected = self._repeated_failure(recent_interactions=recent_interactions)
        uncertainty_stacking = (
            dialogue_management.idea_state == "uncertain"
            and conversation_awareness.unresolved_thread_open
        )
        humor_candidate = (
            "absurdity" in trigger_signals and dialogue_management.interaction_mode == "social"
        )

        trigger = self._primary_trigger(
            trigger_signals=trigger_signals,
            repeated_failure_detected=repeated_failure_detected,
            uncertainty_stacking=uncertainty_stacking,
        )
        return StateInterpretationResult(
            trigger=trigger,
            trigger_signals=trigger_signals,
            repeated_failure_detected=repeated_failure_detected,
            uncertainty_stacking=uncertainty_stacking,
            humor_candidate=humor_candidate,
        )

    @staticmethod
    def _repeated_failure(*, recent_interactions: list[dict[str, object]]) -> bool:
        recent_frustration = 0
        for item in recent_interactions[:3]:
            response = item.get("response") or {}
            state_control = response.get("state_control") or {}
            if str(state_control.get("core_state") or "").strip() == "frustration":
                recent_frustration += 1
        return recent_frustration >= 2

    @staticmethod
    def _primary_trigger(
        *,
        trigger_signals: list[str],
        repeated_failure_detected: bool,
        uncertainty_stacking: bool,
    ) -> str | None:
        if "breakthrough" in trigger_signals:
            return "breakthrough"
        if repeated_failure_detected:
            return "repeated_failure"
        if "failure" in trigger_signals:
            return "confusion"
        if "absurdity" in trigger_signals:
            return "absurdity"
        if uncertainty_stacking or "hesitation" in trigger_signals:
            return "confusion"
        if "clarity" in trigger_signals:
            return "clarity_achieved"
        if "exploration" in trigger_signals:
            return "confusion"
        return "clarity_achieved"
