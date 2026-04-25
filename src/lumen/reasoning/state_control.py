from __future__ import annotations

from lumen.reasoning.anti_spiral_guard import AntiSpiralGuard
from lumen.reasoning.pipeline_models import (
    ConversationAwarenessResult,
    DialogueManagementResult,
    NLUExtraction,
    StateControlResult,
    StateInterpretationResult,
)


class StateControlLayer:
    """Regulated cognitive-affective posture layer with anti-spiral control."""

    def __init__(self, *, anti_spiral_guard: AntiSpiralGuard | None = None) -> None:
        self.anti_spiral_guard = anti_spiral_guard or AntiSpiralGuard()

    def infer(
        self,
        *,
        prompt: str,
        nlu: NLUExtraction,
        dialogue_management: DialogueManagementResult,
        conversation_awareness: ConversationAwarenessResult,
        state_interpretation: StateInterpretationResult,
        recent_interactions: list[dict[str, object]],
    ) -> StateControlResult:
        core_state = self._core_state(
            dialogue_management=dialogue_management,
            conversation_awareness=conversation_awareness,
            state_interpretation=state_interpretation,
            recent_interactions=recent_interactions,
        )
        anti_spiral_active, anti_spiral_reason = self.anti_spiral_guard.evaluate(
            prompt=prompt,
            dialogue_management=dialogue_management,
            conversation_awareness=conversation_awareness,
            state_interpretation=state_interpretation,
        )
        response_bias = self._response_bias(
            core_state=core_state,
            dialogue_management=dialogue_management,
            anti_spiral_active=anti_spiral_active,
        )
        humor_allowed = self._humor_allowed(
            core_state=core_state,
            dialogue_management=dialogue_management,
            nlu=nlu,
            anti_spiral_active=anti_spiral_active,
        )
        return StateControlResult(
            core_state=core_state,
            trigger=state_interpretation.trigger,
            anti_spiral_active=anti_spiral_active,
            anti_spiral_reason=anti_spiral_reason,
            response_bias=response_bias,
            humor_allowed=humor_allowed,
        )

    def _core_state(
        self,
        *,
        dialogue_management: DialogueManagementResult,
        conversation_awareness: ConversationAwarenessResult,
        state_interpretation: StateInterpretationResult,
        recent_interactions: list[dict[str, object]],
    ) -> str:
        if state_interpretation.humor_candidate:
            return "humor"
        if state_interpretation.trigger == "breakthrough":
            return "momentum"
        if state_interpretation.trigger == "repeated_failure":
            return "frustration"
        if state_interpretation.trigger == "confusion":
            return "curiosity"
        if dialogue_management.response_strategy in {"challenge", "ask_question"}:
            return "curiosity"
        if conversation_awareness.conversation_momentum == "building":
            if any(
                str((item.get("response") or {}).get("state_control", {}).get("core_state") or "").strip() == "momentum"
                for item in recent_interactions[:2]
            ):
                return "momentum"
        return "focus"

    @staticmethod
    def _response_bias(
        *,
        core_state: str,
        dialogue_management: DialogueManagementResult,
        anti_spiral_active: bool,
    ) -> str:
        if anti_spiral_active:
            return "stabilize"
        if core_state == "momentum":
            return "advance"
        if core_state == "curiosity":
            return "explore"
        if core_state == "frustration":
            return "repair"
        if core_state == "humor":
            return "lighten"
        if dialogue_management.response_strategy == "summarize":
            return "synthesize"
        return "clarify"

    @staticmethod
    def _humor_allowed(
        *,
        core_state: str,
        dialogue_management: DialogueManagementResult,
        nlu: NLUExtraction,
        anti_spiral_active: bool,
    ) -> bool:
        if anti_spiral_active:
            return False
        if nlu.ambiguity_flags:
            return False
        if dialogue_management.interaction_mode in {"clarification", "synthesis"}:
            return False
        return core_state == "humor" and dialogue_management.interaction_mode == "social"
