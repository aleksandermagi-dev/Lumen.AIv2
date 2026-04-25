from __future__ import annotations

from lumen.reasoning.pipeline_models import (
    ConversationAwarenessResult,
    DialogueManagementResult,
    LowConfidenceRecoveryResult,
    RouteDecisionView,
    SRDDiagnosticResult,
)


class SRDDiagnostic:
    """Structural response diagnostic for disruption, repairability, and agency preservation."""

    def diagnose(
        self,
        *,
        dialogue_management: DialogueManagementResult,
        conversation_awareness: ConversationAwarenessResult,
        route_decision: RouteDecisionView,
        low_confidence_recovery: LowConfidenceRecoveryResult,
    ) -> SRDDiagnosticResult:
        failure_types: list[str] = []
        if route_decision.weak_route:
            failure_types.append("logic_failure")
        if low_confidence_recovery.recovery_mode in {"soft_clarify", "hard_clarify"}:
            failure_types.append("coherence_failure")
        if dialogue_management.interaction_mode == "clarification":
            failure_types.append("agency_failure")
        if (
            conversation_awareness.unresolved_thread_open
            and conversation_awareness.conversation_momentum in {"doubting", "stalled"}
        ):
            failure_types.append("trust_failure")
        elif (
            low_confidence_recovery.recovery_mode == "hard_clarify"
            and conversation_awareness.conversation_momentum in {"doubting", "stalled"}
        ):
            failure_types.append("trust_failure")

        if not failure_types:
            return SRDDiagnosticResult(
                stage="baseline",
                failure_types=[],
                escalation_risk="low",
                repairable_here=True,
                preserve_agency=True,
                should_exit_early=False,
                rationale="No structural disruption signal was strong enough to trigger SRD escalation.",
            )

        repairable_here = low_confidence_recovery.recovery_mode != "hard_clarify"
        should_exit_early = not repairable_here and (
            low_confidence_recovery.acknowledge_partial_understanding
            or "trust_failure" in failure_types
        )
        escalation_risk = "medium"
        if "trust_failure" in failure_types or should_exit_early:
            escalation_risk = "high"
        stage = "repair_attempt" if repairable_here else "agency_block"
        rationale = (
            "Unexpected inconsistency and blocked agency were detected, so the response should preserve agency and avoid escalation."
            if should_exit_early
            else "Structural disruption was detected, but it is still repairable in the current turn."
        )
        return SRDDiagnosticResult(
            stage=stage,
            failure_types=failure_types,
            escalation_risk=escalation_risk,
            repairable_here=repairable_here,
            preserve_agency=True,
            should_exit_early=should_exit_early,
            rationale=rationale,
        )
