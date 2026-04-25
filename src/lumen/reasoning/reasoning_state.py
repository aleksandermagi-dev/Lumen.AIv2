from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lumen.reasoning.confidence_gradient import ConfidenceGradient


VALID_EXECUTION_FAILURE_CLASSES = {
    "routing_failure",
    "input_failure",
    "runtime_dependency_failure",
    "execution_failure",
    "artifact_failure",
    "success",
}


VALID_REASONING_MODES = {"default", "collab", "direct"}


@dataclass(slots=True, frozen=True)
class ModeBehaviorProfile:
    mode: str
    posture: str
    explanation_style: str
    follow_up_style: str
    clarification_style: str

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "posture": self.posture,
            "explanation_style": self.explanation_style,
            "follow_up_style": self.follow_up_style,
            "clarification_style": self.clarification_style,
        }


@dataclass(slots=True, frozen=True)
class ExecutionOutcome:
    selected_tool_id: str | None = None
    selected_capability: str | None = None
    execution_attempted: bool = False
    execution_status: str = "idle"
    failure_class: str | None = None
    runtime_diagnostics: dict[str, object] = field(default_factory=dict)
    summary: str | None = None
    artifact_signals: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "selected_tool_id": self.selected_tool_id,
            "selected_capability": self.selected_capability,
            "execution_attempted": self.execution_attempted,
            "execution_status": self.execution_status,
            "failure_class": self.failure_class,
            "runtime_diagnostics": dict(self.runtime_diagnostics),
            "summary": self.summary,
            "artifact_signals": dict(self.artifact_signals),
        }


@dataclass(slots=True)
class CognitiveStateFrame:
    current_intent: str = "unknown"
    current_task: str | None = None
    current_path: str | None = None
    confidence: float = 0.0
    confidence_tier: str = "low"
    route_decision: dict[str, object] = field(default_factory=dict)
    intent_domain: str | None = None
    response_depth: str | None = None
    conversation_phase: str | None = None
    ambiguity_status: str = "unknown"
    selected_mode: str = "default"
    active_domain: str | None = None
    pending_followup: dict[str, object] = field(default_factory=dict)
    tool_candidate: dict[str, object] = field(default_factory=dict)
    memory_context_used: list[dict[str, object]] = field(default_factory=list)
    tool_usage_intent: dict[str, object] = field(default_factory=dict)
    tool_decision: dict[str, object] = field(default_factory=dict)
    response_style: dict[str, object] = field(default_factory=dict)
    uncertainty_flags: list[str] = field(default_factory=list)
    failure_flags: list[str] = field(default_factory=list)
    rationale_summary: str | None = None
    execution_status: str = "idle"
    known_context_summary: str | None = None
    resolved_prompt: str | None = None
    canonical_subject: str | None = None
    continuation_target: str | None = None
    comparison_targets: list[str] = field(default_factory=list)
    explanation_strategy: str = "direct_definition"
    tool_context: dict[str, object] = field(default_factory=dict)
    runtime_diagnostics: dict[str, object] = field(default_factory=dict)
    failure_class: str | None = None
    turn_status: str = "intake"
    mode_behavior: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "current_intent": self.current_intent,
            "current_task": self.current_task,
            "current_path": self.current_path,
            "confidence": self.confidence,
            "confidence_tier": self.confidence_tier,
            "route_decision": dict(self.route_decision),
            "intent_domain": self.intent_domain,
            "response_depth": self.response_depth,
            "conversation_phase": self.conversation_phase,
            "ambiguity_status": self.ambiguity_status,
            "selected_mode": self.selected_mode,
            "active_domain": self.active_domain,
            "pending_followup": dict(self.pending_followup),
            "tool_candidate": dict(self.tool_candidate),
            "memory_context_used": [dict(item) for item in self.memory_context_used],
            "tool_usage_intent": dict(self.tool_usage_intent),
            "tool_decision": dict(self.tool_decision),
            "response_style": dict(self.response_style),
            "uncertainty_flags": list(self.uncertainty_flags),
            "failure_flags": list(self.failure_flags),
            "rationale_summary": self.rationale_summary,
            "execution_status": self.execution_status,
            "known_context_summary": self.known_context_summary,
            "resolved_prompt": self.resolved_prompt,
            "canonical_subject": self.canonical_subject,
            "continuation_target": self.continuation_target,
            "comparison_targets": list(self.comparison_targets),
            "explanation_strategy": self.explanation_strategy,
            "tool_context": dict(self.tool_context),
            "runtime_diagnostics": dict(self.runtime_diagnostics),
            "failure_class": self.failure_class,
            "turn_status": self.turn_status,
            "mode_behavior": dict(self.mode_behavior),
        }

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "CognitiveStateFrame":
        if not isinstance(payload, dict):
            return cls()
        failure_class = str(payload.get("failure_class") or "").strip() or None
        if failure_class not in VALID_EXECUTION_FAILURE_CLASSES:
            failure_class = None
        selected_mode = _normalized_reasoning_mode(payload.get("selected_mode"))
        mode_behavior = _normalized_mode_behavior(
            selected_mode=selected_mode,
            raw_behavior=payload.get("mode_behavior"),
        )
        comparison_targets = payload.get("comparison_targets")
        confidence = float(payload.get("confidence") or 0.0)
        confidence_tier = str(payload.get("confidence_tier") or "").strip().lower()
        if confidence_tier not in {"low", "medium", "high"}:
            confidence_tier = ConfidenceGradient.tier_for_score(confidence)
        return cls(
            current_intent=str(payload.get("current_intent") or "unknown"),
            current_task=_optional_str(payload.get("current_task")),
            current_path=_optional_str(payload.get("current_path")),
            confidence=confidence,
            confidence_tier=confidence_tier,
            route_decision=dict(payload.get("route_decision") or {}),
            intent_domain=_optional_str(payload.get("intent_domain")),
            response_depth=_optional_str(payload.get("response_depth")),
            conversation_phase=_optional_str(payload.get("conversation_phase")),
            ambiguity_status=str(payload.get("ambiguity_status") or "unknown"),
            selected_mode=selected_mode,
            active_domain=_optional_str(payload.get("active_domain")),
            pending_followup=dict(payload.get("pending_followup") or {}),
            tool_candidate=dict(payload.get("tool_candidate") or {}),
            memory_context_used=[
                dict(item) for item in payload.get("memory_context_used") or []
                if isinstance(item, dict)
            ],
            tool_usage_intent=dict(payload.get("tool_usage_intent") or {}),
            tool_decision=dict(payload.get("tool_decision") or {}),
            response_style=dict(payload.get("response_style") or {}),
            uncertainty_flags=[
                str(item).strip() for item in payload.get("uncertainty_flags") or []
                if str(item).strip()
            ],
            failure_flags=[
                str(item).strip() for item in payload.get("failure_flags") or []
                if str(item).strip()
            ],
            rationale_summary=_optional_str(payload.get("rationale_summary")),
            execution_status=str(payload.get("execution_status") or "idle"),
            known_context_summary=_optional_str(payload.get("known_context_summary")),
            resolved_prompt=_optional_str(payload.get("resolved_prompt")),
            canonical_subject=_optional_str(payload.get("canonical_subject")),
            continuation_target=_optional_str(payload.get("continuation_target")),
            comparison_targets=[
                str(item).strip()
                for item in comparison_targets
                if str(item).strip()
            ]
            if isinstance(comparison_targets, list)
            else [],
            explanation_strategy=str(payload.get("explanation_strategy") or "direct_definition"),
            tool_context=dict(payload.get("tool_context") or {}),
            runtime_diagnostics=dict(payload.get("runtime_diagnostics") or {}),
            failure_class=failure_class,
            turn_status=str(payload.get("turn_status") or "intake"),
            mode_behavior=mode_behavior,
        )

    def with_updates(self, **updates: object) -> "CognitiveStateFrame":
        payload = self.to_dict()
        payload.update(updates)
        return self.from_mapping(payload)


ReasoningStateFrame = CognitiveStateFrame


def _optional_str(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _normalized_reasoning_mode(value: object) -> str:
    normalized = str(value or "default").strip().lower()
    if normalized == "conversational":
        return "collab"
    if normalized in VALID_REASONING_MODES:
        return normalized
    return "default"


def _default_mode_behavior(selected_mode: str) -> dict[str, object]:
    if selected_mode == "direct":
        return {
            "mode": "direct",
            "posture": "decisive",
            "explanation_style": "compressed",
            "follow_up_style": "minimal",
            "clarification_style": "minimal",
        }
    if selected_mode == "collab":
        return {
            "mode": "collab",
            "posture": "exploratory",
            "explanation_style": "guided",
            "follow_up_style": "cooperative",
            "clarification_style": "collaborative",
        }
    return {
        "mode": "default",
        "posture": "balanced",
        "explanation_style": "structured",
        "follow_up_style": "balanced",
        "clarification_style": "balanced",
    }


def _normalized_mode_behavior(
    *,
    selected_mode: str,
    raw_behavior: object,
) -> dict[str, object]:
    defaults = _default_mode_behavior(selected_mode)
    if not isinstance(raw_behavior, dict):
        return defaults
    normalized = {**defaults}
    for key, value in raw_behavior.items():
        text = str(value or "").strip()
        if text:
            normalized[str(key)] = text
    normalized["mode"] = _normalized_reasoning_mode(normalized.get("mode"))
    return normalized
