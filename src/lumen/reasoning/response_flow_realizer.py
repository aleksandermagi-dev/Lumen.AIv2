from __future__ import annotations

from lumen.reasoning.reasoning_language import ReasoningResponseLanguage
from lumen.reasoning.reasoning_state import ReasoningStateFrame


class ResponseFlowRealizer:
    """Adds light natural-flow framing to structured planning and research outputs."""

    @classmethod
    def intro_for_response(cls, *, response: dict[str, object], mode: str) -> str | None:
        if mode not in {"planning", "research"}:
            return None
        state = ReasoningStateFrame.from_mapping(
            response.get("reasoning_state") if isinstance(response.get("reasoning_state"), dict) else None
        )
        posture = (
            response.get("response_behavior_posture")
            if isinstance(response.get("response_behavior_posture"), dict)
            else {}
        )
        visible_uncertainty = bool(posture.get("visible_uncertainty"))
        confidence_tier = cls._confidence_tier(
            state=state,
            confidence_posture=str(response.get("confidence_posture") or "").strip(),
            visible_uncertainty=visible_uncertainty,
        )
        style = str(state.selected_mode or "default").strip().lower() or "default"
        intent_domain = str(state.intent_domain or response.get("intent_domain") or "").strip()
        response_depth = str(state.response_depth or response.get("response_depth") or "").strip()

        if mode == "planning":
            return cls._planning_intro(
                style=style,
                confidence_tier=confidence_tier,
                intent_domain=intent_domain,
                response_depth=response_depth,
            )
        return cls._research_intro(
            style=style,
            confidence_tier=confidence_tier,
            intent_domain=intent_domain,
            response_depth=response_depth,
        )

    @staticmethod
    def next_label_for_response(*, response: dict[str, object], mode: str) -> str:
        state = ReasoningStateFrame.from_mapping(
            response.get("reasoning_state") if isinstance(response.get("reasoning_state"), dict) else None
        )
        style = str(state.selected_mode or "default").strip().lower() or "default"
        posture = (
            response.get("response_behavior_posture")
            if isinstance(response.get("response_behavior_posture"), dict)
            else {}
        )
        confidence_tier = ResponseFlowRealizer._confidence_tier(
            state=state,
            confidence_posture=str(response.get("confidence_posture") or "").strip(),
            visible_uncertainty=bool(posture.get("visible_uncertainty")),
        )
        if style == "direct":
            if confidence_tier == "low":
                return "Next check:"
            return "Next:"
        if mode == "research":
            if confidence_tier == "low":
                return "Best next check:"
            return "Best next step:"
        if confidence_tier == "low":
            return "Next check:"
        return "Next step:"

    @staticmethod
    def _confidence_tier(
        *,
        state: ReasoningStateFrame,
        confidence_posture: str,
        visible_uncertainty: bool,
    ) -> str:
        tier = str(state.confidence_tier or "").strip().lower()
        if tier in {"low", "medium", "high"}:
            return tier
        posture = confidence_posture.lower()
        if posture:
            tier = ReasoningResponseLanguage.language_confidence_tier(
                confidence_posture=posture,
            )
            if tier == "high" and not visible_uncertainty:
                return "high"
            if tier == "medium":
                return "medium"
            if tier == "low":
                return "low"
        if posture == "strong" and not visible_uncertainty:
            return "high"
        if visible_uncertainty:
            return "low"
        return "low"

    @staticmethod
    def _planning_intro(
        *,
        style: str,
        confidence_tier: str,
        intent_domain: str,
        response_depth: str,
    ) -> str:
        if style == "direct":
            if confidence_tier == "high":
                return "Best plan from here:"
            if confidence_tier == "medium":
                return "Best starting plan:"
            return "Best provisional plan:"
        if intent_domain == "planning_strategy":
            if confidence_tier == "high":
                return "Here’s the roadmap I’d use from here."
            if confidence_tier == "low" or response_depth == "deep":
                return "Here’s the roadmap I’d start with, keeping the open assumptions visible."
            return "Here’s the roadmap I’d start with."
        if confidence_tier == "high":
            return "Here’s the plan I’d use from here."
        if confidence_tier == "medium":
            return "Here’s the plan I’d start with."
        return "Here’s the best starting plan, and I’d still keep it provisional."

    @staticmethod
    def _research_intro(
        *,
        style: str,
        confidence_tier: str,
        intent_domain: str,
        response_depth: str,
    ) -> str:
        if style == "direct":
            if confidence_tier == "high":
                return "Best grounded read:"
            if confidence_tier == "medium":
                return "Best current read:"
            return "Best provisional read:"
        if intent_domain == "learning_teaching":
            if confidence_tier == "high":
                return "Here’s the clearest explanation I can support right now."
            return "Here’s the clearest explanation so far, keeping the uncertainty visible."
        if intent_domain == "technical_engineering":
            if confidence_tier == "high":
                return "Here’s the clearest technical read I can support right now."
            return "Here’s the clearest technical read so far, keeping the uncertainty visible."
        if intent_domain == "decision_support":
            if confidence_tier == "high":
                return "Here’s the clearest read on the tradeoffs right now."
            return "Here’s the clearest tradeoff read so far, keeping the uncertainty visible."
        if confidence_tier == "high":
            return "Here’s the clearest read I can support right now."
        if confidence_tier == "medium":
            if response_depth == "deep":
                return "Here’s the clearest read so far, with the uncertainty kept visible."
            return "Here’s the clearest read so far, keeping the uncertainty visible."
        return "Here’s the best first read, and I’d still hold it provisionally."
