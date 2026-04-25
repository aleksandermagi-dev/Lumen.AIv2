from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lumen.reasoning.assistant_context import AssistantContext
from lumen.tools.registry_types import ToolResult


ASSISTANT_RESPONSE_SCHEMA_TYPE = "assistant_response"
ASSISTANT_RESPONSE_SCHEMA_VERSION = "1"


@dataclass(slots=True)
class RouteMetadata:
    confidence: float
    reason: str
    source: str | None = None
    strength: str | None = None
    caution: str | None = None
    evidence: list[dict[str, object]] = field(default_factory=list)
    decision_summary: dict[str, object] | None = None
    ambiguity: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "confidence": self.confidence,
            "reason": self.reason,
        }
        if self.source:
            payload["source"] = self.source
        if self.strength:
            payload["strength"] = self.strength
        if self.caution:
            payload["caution"] = self.caution
        if self.evidence:
            payload["evidence"] = list(self.evidence)
        if self.decision_summary:
            payload["decision_summary"] = dict(self.decision_summary)
        if self.ambiguity:
            payload["ambiguity"] = dict(self.ambiguity)
        return payload


@dataclass(slots=True)
class ToolExecutionDetails:
    tool_id: str
    capability: str
    input_path: Path | None = None
    params: dict[str, int | float | str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "tool_id": self.tool_id,
            "capability": self.capability,
            "input_path": str(self.input_path) if self.input_path else None,
            "params": dict(self.params),
        }


@dataclass(slots=True)
class AssistantResponse:
    mode: str
    kind: str
    summary: str
    context: AssistantContext | None = None
    route: RouteMetadata | None = None
    resolved_prompt: str | None = None
    resolution_strategy: str | None = None
    resolution_reason: str | None = None
    intent_domain: str | None = None
    intent_domain_confidence: float | None = None
    response_depth: str | None = None
    conversation_phase: str | None = None
    next_step_state: dict[str, object] | None = None
    tool_suggestion_state: dict[str, object] | None = None

    def _base_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_type": ASSISTANT_RESPONSE_SCHEMA_TYPE,
            "schema_version": ASSISTANT_RESPONSE_SCHEMA_VERSION,
            "mode": self.mode,
            "kind": self.kind,
            "summary": self.summary,
            "context": self.context.to_dict() if self.context else {},
        }
        if self.route is not None:
            payload["route"] = self.route.to_dict()
        if self.resolved_prompt:
            payload["resolved_prompt"] = self.resolved_prompt
        if self.resolution_strategy:
            payload["resolution_strategy"] = self.resolution_strategy
        if self.resolution_reason:
            payload["resolution_reason"] = self.resolution_reason
        if self.intent_domain:
            payload["intent_domain"] = self.intent_domain
        if self.intent_domain_confidence is not None:
            payload["intent_domain_confidence"] = self.intent_domain_confidence
        if self.response_depth:
            payload["response_depth"] = self.response_depth
        if self.conversation_phase:
            payload["conversation_phase"] = self.conversation_phase
        if self.next_step_state is not None:
            payload["next_step_state"] = dict(self.next_step_state)
        if self.tool_suggestion_state is not None:
            payload["tool_suggestion_state"] = dict(self.tool_suggestion_state)
        return payload


@dataclass(slots=True)
class PlanningResponse(AssistantResponse):
    evidence: list[str] = field(default_factory=list)
    best_evidence: str | None = None
    local_context_summary: str | None = None
    grounded_interpretation: str | None = None
    working_hypothesis: str | None = None
    uncertainty_note: str | None = None
    reasoning_frame: dict[str, str] = field(default_factory=dict)
    local_context_assessment: str | None = None
    grounding_strength: str | None = None
    confidence_posture: str | None = None
    closing_strategy: str | None = None
    steps: list[str] = field(default_factory=list)
    next_action: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload = self._base_payload()
        if self.evidence:
            payload["evidence"] = list(self.evidence)
        if self.best_evidence:
            payload["best_evidence"] = self.best_evidence
        if self.local_context_summary:
            payload["local_context_summary"] = self.local_context_summary
        if self.grounded_interpretation:
            payload["grounded_interpretation"] = self.grounded_interpretation
        if self.working_hypothesis:
            payload["working_hypothesis"] = self.working_hypothesis
        if self.uncertainty_note:
            payload["uncertainty_note"] = self.uncertainty_note
        if self.reasoning_frame:
            payload["reasoning_frame"] = dict(self.reasoning_frame)
        if self.local_context_assessment:
            payload["local_context_assessment"] = self.local_context_assessment
        if self.grounding_strength:
            payload["grounding_strength"] = self.grounding_strength
        if self.confidence_posture:
            payload["confidence_posture"] = self.confidence_posture
        if self.closing_strategy:
            payload["closing_strategy"] = self.closing_strategy
        payload["steps"] = list(self.steps)
        if self.next_action:
            payload["next_action"] = self.next_action
        return payload


@dataclass(slots=True)
class ResearchResponse(AssistantResponse):
    evidence: list[str] = field(default_factory=list)
    best_evidence: str | None = None
    local_context_summary: str | None = None
    grounded_interpretation: str | None = None
    working_hypothesis: str | None = None
    uncertainty_note: str | None = None
    reasoning_frame: dict[str, str] = field(default_factory=dict)
    local_context_assessment: str | None = None
    grounding_strength: str | None = None
    confidence_posture: str | None = None
    closing_strategy: str | None = None
    findings: list[str] = field(default_factory=list)
    recommendation: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload = self._base_payload()
        if self.evidence:
            payload["evidence"] = list(self.evidence)
        if self.best_evidence:
            payload["best_evidence"] = self.best_evidence
        if self.local_context_summary:
            payload["local_context_summary"] = self.local_context_summary
        if self.grounded_interpretation:
            payload["grounded_interpretation"] = self.grounded_interpretation
        if self.working_hypothesis:
            payload["working_hypothesis"] = self.working_hypothesis
        if self.uncertainty_note:
            payload["uncertainty_note"] = self.uncertainty_note
        if self.reasoning_frame:
            payload["reasoning_frame"] = dict(self.reasoning_frame)
        if self.local_context_assessment:
            payload["local_context_assessment"] = self.local_context_assessment
        if self.grounding_strength:
            payload["grounding_strength"] = self.grounding_strength
        if self.confidence_posture:
            payload["confidence_posture"] = self.confidence_posture
        if self.closing_strategy:
            payload["closing_strategy"] = self.closing_strategy
        payload["findings"] = list(self.findings)
        if self.recommendation:
            payload["recommendation"] = self.recommendation
        return payload


@dataclass(slots=True)
class ToolAssistantResponse(AssistantResponse):
    tool_execution: ToolExecutionDetails | None = None
    tool_result: ToolResult | None = None
    tool_route_origin: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload = self._base_payload()
        if self.tool_execution is not None:
            payload["tool_execution"] = self.tool_execution.to_dict()
        if self.tool_result is not None:
            payload["tool_result"] = self.tool_result
        if self.tool_route_origin:
            payload["tool_route_origin"] = self.tool_route_origin
        return payload


@dataclass(slots=True)
class ClarificationResponse(AssistantResponse):
    clarification_question: str | None = None
    options: list[str] = field(default_factory=list)
    clarification_context: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        payload = self._base_payload()
        if self.clarification_question:
            payload["clarification_question"] = self.clarification_question
        if self.options:
            payload["options"] = list(self.options)
        if self.clarification_context:
            payload["clarification_context"] = dict(self.clarification_context)
        return payload


@dataclass(slots=True)
class ConversationResponse(AssistantResponse):
    reply: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload = self._base_payload()
        if self.reply:
            payload["reply"] = self.reply
        return payload


@dataclass(slots=True)
class SafetyResponse(AssistantResponse):
    boundary_explanation: str | None = None
    safe_redirects: list[str] = field(default_factory=list)
    safety_decision: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        payload = self._base_payload()
        if self.boundary_explanation:
            payload["boundary_explanation"] = self.boundary_explanation
        if self.safe_redirects:
            payload["safe_redirects"] = list(self.safe_redirects)
        if self.safety_decision:
            payload["safety_decision"] = dict(self.safety_decision)
        return payload
