from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lumen.reasoning.reasoning_state import ReasoningStateFrame


@dataclass(slots=True)
class TrainabilityTrace:
    """Compact record of learnable decision surfaces for later offline labeling."""

    schema_version: str = "1"
    available_training_surfaces: list[str] = field(default_factory=list)
    deterministic_surfaces: list[str] = field(default_factory=list)
    intent_domain_classification: dict[str, object] = field(default_factory=dict)
    route_recommendation_support: dict[str, object] = field(default_factory=dict)
    memory_relevance_ranking: dict[str, object] = field(default_factory=dict)
    tool_use_decision_support: dict[str, object] = field(default_factory=dict)
    response_style_selection: dict[str, object] = field(default_factory=dict)
    confidence_calibration_support: dict[str, object] = field(default_factory=dict)
    supervised_decision_support: dict[str, object] = field(default_factory=dict)
    rationale_summary: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "available_training_surfaces": list(self.available_training_surfaces),
            "deterministic_surfaces": list(self.deterministic_surfaces),
            "intent_domain_classification": dict(self.intent_domain_classification),
            "route_recommendation_support": dict(self.route_recommendation_support),
            "memory_relevance_ranking": dict(self.memory_relevance_ranking),
            "tool_use_decision_support": dict(self.tool_use_decision_support),
            "response_style_selection": dict(self.response_style_selection),
            "confidence_calibration_support": dict(self.confidence_calibration_support),
            "supervised_decision_support": dict(self.supervised_decision_support),
            "rationale_summary": self.rationale_summary,
        }

    @classmethod
    def build(cls, *, response: dict[str, object]) -> "TrainabilityTrace":
        reasoning_state = ReasoningStateFrame.from_mapping(
            response.get("reasoning_state") if isinstance(response.get("reasoning_state"), dict) else None
        )
        route = response.get("route") if isinstance(response.get("route"), dict) else {}
        tool_threshold = (
            response.get("tool_threshold_decision")
            if isinstance(response.get("tool_threshold_decision"), dict)
            else dict(reasoning_state.tool_decision or {})
        )
        execution_outcome = (
            response.get("execution_outcome") if isinstance(response.get("execution_outcome"), dict) else {}
        )
        memory_classifier = cls._memory_classifier_diagnostics(response=response)
        retrieval_diagnostics = cls._retrieval_diagnostics(response=response)
        supervised_support = (
            response.get("supervised_support_trace")
            if isinstance(response.get("supervised_support_trace"), dict)
            else {}
        )
        response_style = dict(reasoning_state.response_style or {})
        confidence_score = float(reasoning_state.confidence or 0.0)
        intent_domain_confidence = response.get("intent_domain_confidence")

        return cls(
            available_training_surfaces=[
                "intent_domain_classification",
                "route_recommendation_support",
                "memory_relevance_ranking",
                "tool_use_decision_support",
                "response_style_selection",
                "confidence_calibration_support",
            ],
            deterministic_surfaces=[
                "system_invariants",
                "safety_boundaries",
                "hard_execution_constraints",
            ],
            intent_domain_classification={
                "intent_domain": str(response.get("intent_domain") or reasoning_state.intent_domain or "").strip() or None,
                "intent_domain_confidence": cls._float_or_none(intent_domain_confidence),
                "route_mode": str(response.get("mode") or reasoning_state.route_decision.get("mode") or "").strip() or None,
                "response_depth": str(response.get("response_depth") or reasoning_state.response_depth or "").strip() or None,
                "conversation_phase": str(response.get("conversation_phase") or reasoning_state.conversation_phase or "").strip() or None,
            },
            route_recommendation_support={
                "mode": str(response.get("mode") or route.get("mode") or reasoning_state.route_decision.get("mode") or "").strip() or None,
                "kind": str(response.get("kind") or route.get("kind") or reasoning_state.route_decision.get("kind") or "").strip() or None,
                "route_confidence": cls._float_or_none(route.get("confidence") or reasoning_state.route_decision.get("confidence")),
                "route_status": str(response.get("route_status") or "").strip() or None,
                "support_status": str(response.get("support_status") or "").strip() or None,
                "resolution_strategy": str(response.get("resolution_strategy") or "").strip() or None,
            },
            memory_relevance_ranking={
                "selected_count": len(memory_classifier.get("selected", [])),
                "rejected_count": len(memory_classifier.get("rejected", [])),
                "selected_labels": cls._labels(memory_classifier.get("selected")),
                "rejected_labels": cls._labels(memory_classifier.get("rejected")),
                "memory_context_used_labels": cls._memory_context_labels(reasoning_state.memory_context_used),
                "memory_context_used_count": len(reasoning_state.memory_context_used or []),
                "retrieval_selected_sources": [
                    str(item).strip()
                    for item in (retrieval_diagnostics.get("selected_sources") or [])
                    if str(item).strip()
                ],
                "retrieval_candidate_origins": dict(retrieval_diagnostics.get("candidate_origins") or {}),
                "retrieval_continuity_buckets": dict(retrieval_diagnostics.get("continuity_buckets") or {}),
                "retrieval_selected_reasons": [
                    str(item.get("reason") or "").strip()
                    for item in (retrieval_diagnostics.get("selected_reasons") or [])
                    if isinstance(item, dict) and str(item.get("reason") or "").strip()
                ],
                "semantic_statuses": [
                    str(item.get("semantic_status") or "").strip()
                    for item in (retrieval_diagnostics.get("selected_reasons") or [])
                    if isinstance(item, dict) and str(item.get("semantic_status") or "").strip()
                ],
            },
            tool_use_decision_support={
                "tool_usage_intent": dict(reasoning_state.tool_usage_intent or {}),
                "should_use_tool": bool(tool_threshold.get("should_use_tool", False)),
                "selected_tool": str(tool_threshold.get("selected_tool") or "").strip() or None,
                "selected_bundle": str(tool_threshold.get("selected_bundle") or "").strip() or None,
                "expected_confidence_gain": cls._float_or_none(tool_threshold.get("expected_confidence_gain")),
                "rationale": str(tool_threshold.get("rationale") or "").strip() or None,
                "execution_attempted": bool(execution_outcome.get("execution_attempted", False)),
                "execution_status": str(execution_outcome.get("execution_status") or "").strip() or None,
            },
            response_style_selection={
                "interaction_style": str(response_style.get("interaction_style") or reasoning_state.selected_mode or "").strip() or None,
                "intent_domain": str(response_style.get("intent_domain") or response.get("intent_domain") or reasoning_state.intent_domain or "").strip() or None,
                "response_depth": str(response_style.get("response_depth") or response.get("response_depth") or reasoning_state.response_depth or "").strip() or None,
                "conversation_phase": str(response_style.get("conversation_phase") or response.get("conversation_phase") or reasoning_state.conversation_phase or "").strip() or None,
                "structure": str(response_style.get("structure") or "").strip() or None,
                "next_steps_enabled": bool(response_style.get("next_steps_enabled", False)),
                "tool_suggestions_enabled": bool(response_style.get("tool_suggestions_enabled", False)),
            },
            confidence_calibration_support={
                "confidence_tier": str(reasoning_state.confidence_tier or "").strip() or "low",
                "confidence_score": confidence_score,
                "confidence_posture": str(response.get("confidence_posture") or "").strip() or None,
                "support_status": str(response.get("support_status") or "").strip() or None,
                "route_status": str(response.get("route_status") or "").strip() or None,
                "memory_signal_present": bool(reasoning_state.memory_context_used),
                "tool_verified": str(execution_outcome.get("execution_status") or "").strip().lower() == "ok",
            },
            supervised_decision_support={
                "enabled": bool(supervised_support.get("enabled", False)),
                "recommended_surfaces": sorted(
                    str(key).strip()
                    for key in (supervised_support.get("recommendations") or {}).keys()
                    if str(key).strip()
                ),
                "applied_surfaces": [
                    str(item).strip()
                    for item in (supervised_support.get("applied_surfaces") or [])
                    if str(item).strip()
                ],
                "deterministic_authority_preserved": bool(
                    supervised_support.get("deterministic_authority_preserved", True)
                ),
            },
            rationale_summary=str(reasoning_state.rationale_summary or "").strip() or None,
        )

    @staticmethod
    def _memory_classifier_diagnostics(*, response: dict[str, object]) -> dict[str, object]:
        retrieval = response.get("memory_retrieval")
        if not isinstance(retrieval, dict):
            return {}
        diagnostics = retrieval.get("diagnostics")
        if not isinstance(diagnostics, dict):
            return {}
        classifier = diagnostics.get("memory_context_classifier")
        if not isinstance(classifier, dict):
            return {}
        return classifier

    @staticmethod
    def _retrieval_diagnostics(*, response: dict[str, object]) -> dict[str, object]:
        retrieval = response.get("memory_retrieval")
        if not isinstance(retrieval, dict):
            return {}
        diagnostics = retrieval.get("diagnostics")
        if not isinstance(diagnostics, dict):
            return {}
        return diagnostics

    @staticmethod
    def _float_or_none(value: object) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _labels(items: object) -> list[str]:
        if not isinstance(items, list):
            return []
        labels: list[str] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or item.get("source") or "").strip()
            if label:
                labels.append(label)
        return labels

    @staticmethod
    def _memory_context_labels(items: object) -> list[str]:
        if not isinstance(items, list):
            return []
        labels: list[str] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or item.get("source") or item.get("kind") or "").strip()
            if label:
                labels.append(label)
        return labels
