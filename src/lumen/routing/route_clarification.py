from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ClarificationThreshold:
    trigger: str
    threshold: float


class RouteClarificationPolicy:
    """Owns clarification thresholds and scope rules for routed prompts."""

    @staticmethod
    def is_ambiguous(route) -> bool:
        decision = route.decision_summary or {}
        return bool(decision.get("ambiguous"))

    @staticmethod
    def base_should_clarify(route) -> bool:
        if not RouteClarificationPolicy.is_ambiguous(route):
            return False
        if route.source == "active_thread_bias":
            return True
        if route.source in {"heuristic_planning", "heuristic_research", "fallback"}:
            return route.confidence <= 0.78
        return False

    @staticmethod
    def in_adaptive_scope(route) -> bool:
        return (
            RouteClarificationPolicy.is_ambiguous(route)
            and route.source in {"heuristic_planning", "heuristic_research"}
        )

    @staticmethod
    def select_adaptive_threshold(
        *,
        nlu_uncertainty_high: bool,
        retrieval_semantic_bias_high: bool,
    ) -> ClarificationThreshold:
        if nlu_uncertainty_high:
            return ClarificationThreshold(trigger="nlu_uncertainty", threshold=0.88)
        if retrieval_semantic_bias_high:
            return ClarificationThreshold(trigger="retrieval_semantic_bias", threshold=0.86)
        return ClarificationThreshold(trigger="adaptive_threshold", threshold=0.84)
