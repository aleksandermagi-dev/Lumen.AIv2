from __future__ import annotations

from lumen.reasoning.pipeline_models import (
    LowConfidenceRecoveryResult,
    NLUExtraction,
    RouteDecisionView,
    VibeCatcherResult,
)


class LowConfidenceRecoveryPolicy:
    """Owns partial-understanding recovery before the system commits to a misleading route."""

    @staticmethod
    def _clarifying_style(vibe_catcher: VibeCatcherResult) -> str:
        signals = set(vibe_catcher.directional_signals)
        if {"disagreement", "frustration"} & signals:
            return "targeted_directional_recovery"
        if "hesitation" in signals:
            return "hesitation_recovery"
        return "directional_recovery"

    def assess(
        self,
        *,
        nlu: NLUExtraction,
        route_decision: RouteDecisionView,
        vibe_catcher: VibeCatcherResult,
    ) -> LowConfidenceRecoveryResult:
        signals = set(vibe_catcher.directional_signals)
        hesitation_override = "hesitation" in signals and vibe_catcher.low_confidence
        if vibe_catcher.low_confidence and (route_decision.weak_route or hesitation_override):
            return LowConfidenceRecoveryResult(
                recovery_mode="soft_clarify",
                acknowledge_partial_understanding=True,
                clarifying_question_style=self._clarifying_style(vibe_catcher),
                rationale=(
                    vibe_catcher.recovery_hint
                    or "The prompt carries partial direction, but the route is too weak to treat as reliable."
                ),
            )
        if vibe_catcher.interpretation_confidence < 0.5 and route_decision.weak_route:
            return LowConfidenceRecoveryResult(
                recovery_mode="soft_clarify",
                acknowledge_partial_understanding=True,
                clarifying_question_style=self._clarifying_style(vibe_catcher),
                rationale=(
                    vibe_catcher.recovery_hint
                    or "The prompt seems directionally meaningful, but the structure is still too noisy to trust without a quick clarification."
                ),
            )
        if nlu.confidence_estimate < 0.35 and route_decision.weak_route:
            return LowConfidenceRecoveryResult(
                recovery_mode="hard_clarify",
                acknowledge_partial_understanding=True,
                clarifying_question_style="directional_recovery",
                rationale="Intent confidence is too low to continue honestly without clarification.",
            )
        if (
            vibe_catcher.low_confidence
            or vibe_catcher.interpretation_confidence < 0.8
            or nlu.confidence_estimate < 0.55
        ):
            return LowConfidenceRecoveryResult(
                recovery_mode="silent_recovery",
                acknowledge_partial_understanding=False,
                clarifying_question_style=None,
                rationale="Normalization and routing were sufficient to continue without explicit recovery.",
            )
        return LowConfidenceRecoveryResult(
            recovery_mode="none",
            acknowledge_partial_understanding=False,
            clarifying_question_style=None,
            rationale="No low-confidence recovery was needed.",
        )
