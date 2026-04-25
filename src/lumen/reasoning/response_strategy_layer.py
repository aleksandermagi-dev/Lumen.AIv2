from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ResponseBehaviorPosture:
    posture: str
    visible_uncertainty: bool
    reasoning_visibility: str
    tone_bias: str
    rationale: str

    def to_dict(self) -> dict[str, object]:
        return {
            "posture": self.posture,
            "visible_uncertainty": self.visible_uncertainty,
            "reasoning_visibility": self.reasoning_visibility,
            "tone_bias": self.tone_bias,
            "rationale": self.rationale,
        }


class ResponseStrategyLayer:
    """Collapses confidence and fallback signals into one final output posture."""

    def select(
        self,
        *,
        mode: str,
        route_mode: str,
        low_confidence_recovery=None,
        srd_diagnostic=None,
        state_control=None,
        response_payload: dict[str, object] | None = None,
    ) -> ResponseBehaviorPosture:
        payload = dict(response_payload or {})
        recovery_mode = str(getattr(low_confidence_recovery, "recovery_mode", "") or "").strip()
        escalation_risk = str(getattr(srd_diagnostic, "escalation_risk", "") or "").strip()
        should_exit_early = bool(getattr(srd_diagnostic, "should_exit_early", False))
        anti_spiral_active = bool(getattr(state_control, "anti_spiral_active", False))
        confidence_posture = str(payload.get("confidence_posture") or "").strip()
        local_context_assessment = str(payload.get("local_context_assessment") or "").strip()
        route_status = str(payload.get("route_status") or "").strip()
        reasoning_state = payload.get("reasoning_state") if isinstance(payload.get("reasoning_state"), dict) else {}
        ambiguity_status = str(reasoning_state.get("ambiguity_status") or "").strip()
        uncertainty_flags = {
            str(item).strip()
            for item in (reasoning_state.get("uncertainty_flags") or [])
            if str(item).strip()
        }
        provider_inference = payload.get("provider_inference") or {}
        hosted_fallback = bool(payload.get("hosted_fallback")) or (
            isinstance(provider_inference, dict) and bool(provider_inference.get("hosted_fallback"))
        )

        if ambiguity_status == "degraded_recovery" or "clarification_suppressed" in uncertainty_flags:
            return ResponseBehaviorPosture(
                posture="stabilize_and_narrow",
                visible_uncertainty=True,
                reasoning_visibility="minimal",
                tone_bias="grounded",
                rationale="Repeated clarification pressure was suppressed, so the response should narrow scope and proceed cautiously.",
            )
        if mode == "clarification" or recovery_mode in {"soft_clarify", "hard_clarify"}:
            return ResponseBehaviorPosture(
                posture="clarify_first",
                visible_uncertainty=True,
                reasoning_visibility="minimal",
                tone_bias="honest_directional",
                rationale="Low-confidence recovery or explicit clarification requires pinning down the route before committing.",
            )
        if mode == "safety" or should_exit_early:
            return ResponseBehaviorPosture(
                posture="boundary_stabilize",
                visible_uncertainty=False,
                reasoning_visibility="minimal",
                tone_bias="steady",
                rationale="Safety or structural exit conditions require a stabilized boundary response.",
            )
        if mode == "conversation":
            return ResponseBehaviorPosture(
                posture="social_direct",
                visible_uncertainty=False,
                reasoning_visibility="minimal",
                tone_bias="natural",
                rationale="Social turns should stay direct and avoid analytical scaffolding.",
            )
        if hosted_fallback:
            return ResponseBehaviorPosture(
                posture="fallback_answer",
                visible_uncertainty=True,
                reasoning_visibility="minimal",
                tone_bias="grounded",
                rationale="Hosted fallback was used because local context was too sparse to answer directly.",
            )
        if anti_spiral_active or escalation_risk == "high":
            return ResponseBehaviorPosture(
                posture="stabilize_and_narrow",
                visible_uncertainty=True,
                reasoning_visibility="minimal",
                tone_bias="steady",
                rationale="Anti-spiral or high escalation risk requires narrowing scope and keeping the reply grounded.",
            )
        if confidence_posture in {"tentative", "conflicted"} or route_status in {"weakened", "under_tension", "unresolved"}:
            return ResponseBehaviorPosture(
                posture="cautious_answer",
                visible_uncertainty=True,
                reasoning_visibility="minimal",
                tone_bias="grounded",
                rationale="Weak or conflicted reasoning signals require a cautious answer with visible uncertainty.",
            )
        if route_mode in {"planning", "research"} and local_context_assessment == "mixed":
            return ResponseBehaviorPosture(
                posture="reconcile_before_expand",
                visible_uncertainty=True,
                reasoning_visibility="minimal",
                tone_bias="grounded",
                rationale="Mixed local context should keep the response in a reconcile-first posture.",
            )
        return ResponseBehaviorPosture(
            posture="direct_answer",
            visible_uncertainty=bool(payload.get("uncertainty_note")),
            reasoning_visibility="minimal",
            tone_bias="clear",
            rationale="Current confidence and fallback signals support a direct user-facing answer.",
        )
