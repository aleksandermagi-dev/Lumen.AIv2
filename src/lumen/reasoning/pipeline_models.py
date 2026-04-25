from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lumen.reasoning.assistant_context import AssistantContext
from lumen.nlu.models import PromptUnderstanding, RouterInputView


@dataclass(slots=True)
class StageContract:
    stage_name: str
    required_inputs: list[str] = field(default_factory=list)
    produced_outputs: list[str] = field(default_factory=list)
    confidence_signal: str | None = None
    failure_state: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_name": self.stage_name,
            "required_inputs": list(self.required_inputs),
            "produced_outputs": list(self.produced_outputs),
            "confidence_signal": self.confidence_signal,
            "failure_state": self.failure_state,
        }


@dataclass(slots=True)
class InputIntake:
    raw_input: str
    cleaned_input: str
    detected_language: str
    session_id: str
    interaction_profile: dict[str, object] = field(default_factory=dict)
    session_context_snapshot: dict[str, Any] = field(default_factory=dict)
    active_thread: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "raw_input": self.raw_input,
            "cleaned_input": self.cleaned_input,
            "detected_language": self.detected_language,
            "session_id": self.session_id,
            "interaction_profile": dict(self.interaction_profile),
            "session_context_snapshot": dict(self.session_context_snapshot),
            "active_thread": self.active_thread,
        }


@dataclass(slots=True, frozen=True)
class CanonicalPromptUnderstandingView:
    original_text: str
    normalized_text: str
    canonical_text: str
    route_ready_text: str
    lookup_ready_text: str
    tool_ready_text: str
    detected_language: str
    normalized_topic: str | None
    dominant_intent: str
    intent_confidence: float
    extracted_entities: tuple[dict[str, object], ...] = field(default_factory=tuple)
    structure: dict[str, object] = field(default_factory=dict)
    profile_advice: dict[str, object] | None = None

    @classmethod
    def from_understanding(cls, understanding: PromptUnderstanding) -> "CanonicalPromptUnderstandingView":
        router_view: RouterInputView = understanding.router_view()
        return cls(
            original_text=understanding.original_text,
            normalized_text=understanding.normalized_text,
            canonical_text=understanding.canonical_text,
            route_ready_text=router_view.route_ready_text,
            lookup_ready_text=understanding.surface_views.lookup_ready_text,
            tool_ready_text=understanding.surface_views.tool_ready_text,
            detected_language=router_view.detected_language,
            normalized_topic=router_view.normalized_topic,
            dominant_intent=router_view.dominant_intent,
            intent_confidence=router_view.intent_confidence,
            extracted_entities=router_view.extracted_entities,
            structure=understanding.structure.to_dict(),
            profile_advice=understanding.profile_advice.to_dict() if understanding.profile_advice else None,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "original_text": self.original_text,
            "normalized_text": self.normalized_text,
            "canonical_text": self.canonical_text,
            "route_ready_text": self.route_ready_text,
            "lookup_ready_text": self.lookup_ready_text,
            "tool_ready_text": self.tool_ready_text,
            "detected_language": self.detected_language,
            "normalized_topic": self.normalized_topic,
            "dominant_intent": self.dominant_intent,
            "intent_confidence": self.intent_confidence,
            "extracted_entities": [dict(item) for item in self.extracted_entities],
            "structure": dict(self.structure),
            "profile_advice": dict(self.profile_advice) if self.profile_advice else None,
        }


@dataclass(slots=True)
class NLUExtraction:
    dominant_intent: str
    secondary_intents: list[str]
    topic: str | None
    entities: list[dict[str, object]]
    action_cues: dict[str, int]
    ambiguity_flags: list[str]
    confidence_estimate: float
    profile_advice: dict[str, object] | None = None
    profile_mismatch: bool = False
    canonical_understanding: CanonicalPromptUnderstandingView | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "dominant_intent": self.dominant_intent,
            "secondary_intents": list(self.secondary_intents),
            "topic": self.topic,
            "entities": list(self.entities),
            "action_cues": dict(self.action_cues),
            "ambiguity_flags": list(self.ambiguity_flags),
            "confidence_estimate": self.confidence_estimate,
            "profile_advice": dict(self.profile_advice) if self.profile_advice else None,
            "profile_mismatch": self.profile_mismatch,
            "canonical_understanding": (
                self.canonical_understanding.to_dict() if self.canonical_understanding else None
            ),
        }


@dataclass(slots=True, frozen=True)
class RouteAuthorityDecision:
    mode: str
    kind: str
    normalized_prompt: str
    confidence: float
    reason: str
    source: str
    route_metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "kind": self.kind,
            "normalized_prompt": self.normalized_prompt,
            "confidence": self.confidence,
            "reason": self.reason,
            "source": self.source,
            "route_metadata": dict(self.route_metadata),
        }


@dataclass(slots=True, frozen=True)
class RetrievalAdvisoryContext:
    query: str
    selected: tuple[dict[str, object], ...] = field(default_factory=tuple)
    memory_reply_hint: str | None = None
    recall_prompt: bool = False
    project_return_prompt: bool = False
    project_reply_hint: str | None = None
    diagnostics: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "selected": [dict(item) for item in self.selected],
            "memory_reply_hint": self.memory_reply_hint,
            "recall_prompt": self.recall_prompt,
            "project_return_prompt": self.project_return_prompt,
            "project_reply_hint": self.project_reply_hint,
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(slots=True, frozen=True)
class ResponsePackagingContext:
    mode: str
    kind: str
    client_surface: str
    route_mode: str
    route_kind: str
    allow_internal_scaffold: bool = False
    packaging_boundary: str = "interactive_reply"

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "kind": self.kind,
            "client_surface": self.client_surface,
            "route_mode": self.route_mode,
            "route_kind": self.route_kind,
            "allow_internal_scaffold": self.allow_internal_scaffold,
            "packaging_boundary": self.packaging_boundary,
        }


@dataclass(slots=True)
class DialogueManagementResult:
    interaction_mode: str
    idea_state: str
    response_strategy: str
    synthesis_checkpoint_due: bool = False
    checkpoint_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "interaction_mode": self.interaction_mode,
            "idea_state": self.idea_state,
            "response_strategy": self.response_strategy,
            "synthesis_checkpoint_due": self.synthesis_checkpoint_due,
            "checkpoint_reason": self.checkpoint_reason,
        }


@dataclass(slots=True)
class ConversationAwarenessResult:
    recent_intent_pattern: str | None = None
    conversation_momentum: str | None = None
    unresolved_thread_open: bool = False
    unresolved_thread_reason: str | None = None
    live_unresolved_question: str | None = None
    branch_state: str | None = None
    return_target: str | None = None
    return_requested: bool = False
    adaptive_posture: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "recent_intent_pattern": self.recent_intent_pattern,
            "conversation_momentum": self.conversation_momentum,
            "unresolved_thread_open": self.unresolved_thread_open,
            "unresolved_thread_reason": self.unresolved_thread_reason,
            "live_unresolved_question": self.live_unresolved_question,
            "branch_state": self.branch_state,
            "return_target": self.return_target,
            "return_requested": self.return_requested,
            "adaptive_posture": self.adaptive_posture,
        }


@dataclass(slots=True)
class VibeCatcherResult:
    normalized_prompt: str
    directional_signals: list[str] = field(default_factory=list)
    interpretation_confidence: float = 1.0
    low_confidence: bool = False
    recovery_hint: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "normalized_prompt": self.normalized_prompt,
            "directional_signals": list(self.directional_signals),
            "interpretation_confidence": self.interpretation_confidence,
            "low_confidence": self.low_confidence,
            "recovery_hint": self.recovery_hint,
        }


@dataclass(slots=True)
class LowConfidenceRecoveryResult:
    recovery_mode: str
    acknowledge_partial_understanding: bool = False
    clarifying_question_style: str | None = None
    rationale: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "recovery_mode": self.recovery_mode,
            "acknowledge_partial_understanding": self.acknowledge_partial_understanding,
            "clarifying_question_style": self.clarifying_question_style,
            "rationale": self.rationale,
        }


@dataclass(slots=True)
class SRDDiagnosticResult:
    stage: str
    failure_types: list[str] = field(default_factory=list)
    escalation_risk: str = "low"
    repairable_here: bool = True
    preserve_agency: bool = True
    should_exit_early: bool = False
    rationale: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "failure_types": list(self.failure_types),
            "escalation_risk": self.escalation_risk,
            "repairable_here": self.repairable_here,
            "preserve_agency": self.preserve_agency,
            "should_exit_early": self.should_exit_early,
            "rationale": self.rationale,
        }


@dataclass(slots=True)
class EmpathyModelResult:
    emotional_signal_detected: bool = False
    feeling_label: str | None = None
    probable_cause: str | None = None
    response_sensitivity: str = "normal"
    grounded_acknowledgment: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "emotional_signal_detected": self.emotional_signal_detected,
            "feeling_label": self.feeling_label,
            "probable_cause": self.probable_cause,
            "response_sensitivity": self.response_sensitivity,
            "grounded_acknowledgment": self.grounded_acknowledgment,
        }


@dataclass(slots=True)
class HumanLanguageLayerResult:
    flow_style: str = "loose"
    sentence_variation: bool = True
    allow_imperfection: bool = True
    context_continuity: bool = False
    emotional_alignment: str = "steady"
    user_energy: str = "casual"
    correction_detected: bool = False
    epistemic_stance: str = "exploratory"
    stance_confidence: str = "medium"
    humor_allowed: bool = False
    curiosity_signal: bool = False
    reaction_tokens_enabled: bool = False
    response_brevity: str = "balanced"

    def to_dict(self) -> dict[str, object]:
        return {
            "flow_style": self.flow_style,
            "sentence_variation": self.sentence_variation,
            "allow_imperfection": self.allow_imperfection,
            "context_continuity": self.context_continuity,
            "emotional_alignment": self.emotional_alignment,
            "user_energy": self.user_energy,
            "correction_detected": self.correction_detected,
            "epistemic_stance": self.epistemic_stance,
            "stance_confidence": self.stance_confidence,
            "humor_allowed": self.humor_allowed,
            "curiosity_signal": self.curiosity_signal,
            "reaction_tokens_enabled": self.reaction_tokens_enabled,
            "response_brevity": self.response_brevity,
        }


@dataclass(slots=True)
class StateInterpretationResult:
    trigger: str | None = None
    trigger_signals: list[str] = field(default_factory=list)
    repeated_failure_detected: bool = False
    uncertainty_stacking: bool = False
    humor_candidate: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "trigger": self.trigger,
            "trigger_signals": list(self.trigger_signals),
            "repeated_failure_detected": self.repeated_failure_detected,
            "uncertainty_stacking": self.uncertainty_stacking,
            "humor_candidate": self.humor_candidate,
        }


@dataclass(slots=True)
class StateControlResult:
    core_state: str
    trigger: str | None = None
    anti_spiral_active: bool = False
    anti_spiral_reason: str | None = None
    response_bias: str | None = None
    humor_allowed: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "core_state": self.core_state,
            "trigger": self.trigger,
            "anti_spiral_active": self.anti_spiral_active,
            "anti_spiral_reason": self.anti_spiral_reason,
            "response_bias": self.response_bias,
            "humor_allowed": self.humor_allowed,
        }


@dataclass(slots=True)
class ThoughtCheckpointSummary:
    current_direction: str
    strongest_point: str
    weakest_point: str
    open_questions: list[str] = field(default_factory=list)
    next_step: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "current_direction": self.current_direction,
            "strongest_point": self.strongest_point,
            "weakest_point": self.weakest_point,
            "open_questions": list(self.open_questions),
            "next_step": self.next_step,
        }


@dataclass(slots=True)
class ThoughtFramingResult:
    response_kind_label: str
    conversation_activity: str
    research_questions: list[str] = field(default_factory=list)
    checkpoint_summary: ThoughtCheckpointSummary | None = None
    branch_return_hint: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "response_kind_label": self.response_kind_label,
            "conversation_activity": self.conversation_activity,
            "research_questions": list(self.research_questions),
            "branch_return_hint": self.branch_return_hint,
            "checkpoint_summary": (
                self.checkpoint_summary.to_dict()
                if self.checkpoint_summary is not None
                else None
            ),
        }


@dataclass(slots=True)
class IntentDomainResult:
    domain: str
    confidence: float
    rationale: str | None = None
    signals: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "domain": self.domain,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "signals": list(self.signals),
        }


@dataclass(slots=True)
class ResponseDepthResult:
    level: str
    rationale: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "level": self.level,
            "rationale": self.rationale,
        }


@dataclass(slots=True)
class ConversationPhaseResult:
    phase: str
    rationale: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "phase": self.phase,
            "rationale": self.rationale,
        }


@dataclass(slots=True)
class RouteCandidateView:
    mode: str
    kind: str
    confidence: float
    reason: str
    source: str
    eligible: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "kind": self.kind,
            "confidence": self.confidence,
            "reason": self.reason,
            "source": self.source,
            "eligible": self.eligible,
        }


@dataclass(slots=True)
class RouteDecisionView:
    selected: dict[str, object]
    alternatives: list[dict[str, object]]
    normalized_scores: list[dict[str, object]]
    caution_notes: list[str]
    weak_route: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "selected": dict(self.selected),
            "alternatives": list(self.alternatives),
            "normalized_scores": list(self.normalized_scores),
            "caution_notes": list(self.caution_notes),
            "weak_route": self.weak_route,
        }


@dataclass(slots=True)
class FrontHalfPipelineResult:
    intake: InputIntake
    nlu: NLUExtraction
    route_candidates: list[RouteCandidateView]
    route_decision: RouteDecisionView
    resolved_prompt: str
    resolution_strategy: str
    resolution_reason: str
    resolution_changed: bool
    vibe_catcher: VibeCatcherResult | None = None
    dialogue_management: DialogueManagementResult | None = None
    conversation_awareness: ConversationAwarenessResult | None = None
    low_confidence_recovery: LowConfidenceRecoveryResult | None = None
    srd_diagnostic: SRDDiagnosticResult | None = None
    empathy_model: EmpathyModelResult | None = None
    human_language_layer: HumanLanguageLayerResult | None = None
    state_interpretation: StateInterpretationResult | None = None
    state_control: StateControlResult | None = None
    thought_framing: ThoughtFramingResult | None = None
    intent_domain: IntentDomainResult | None = None
    response_depth: ResponseDepthResult | None = None
    conversation_phase: ConversationPhaseResult | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "intake": self.intake.to_dict(),
            "nlu": self.nlu.to_dict(),
            "vibe_catcher": self.vibe_catcher.to_dict() if self.vibe_catcher else None,
            "dialogue_management": (
                self.dialogue_management.to_dict() if self.dialogue_management else None
            ),
            "conversation_awareness": (
                self.conversation_awareness.to_dict() if self.conversation_awareness else None
            ),
            "low_confidence_recovery": (
                self.low_confidence_recovery.to_dict() if self.low_confidence_recovery else None
            ),
            "srd_diagnostic": (
                self.srd_diagnostic.to_dict() if self.srd_diagnostic else None
            ),
            "empathy_model": (
                self.empathy_model.to_dict() if self.empathy_model else None
            ),
            "human_language_layer": (
                self.human_language_layer.to_dict() if self.human_language_layer else None
            ),
            "state_interpretation": (
                self.state_interpretation.to_dict() if self.state_interpretation else None
            ),
            "state_control": (
                self.state_control.to_dict() if self.state_control else None
            ),
            "thought_framing": (
                self.thought_framing.to_dict() if self.thought_framing else None
            ),
            "intent_domain": (
                self.intent_domain.to_dict() if self.intent_domain else None
            ),
            "response_depth": (
                self.response_depth.to_dict() if self.response_depth else None
            ),
            "conversation_phase": (
                self.conversation_phase.to_dict() if self.conversation_phase else None
            ),
            "route_candidates": [candidate.to_dict() for candidate in self.route_candidates],
            "route_decision": self.route_decision.to_dict(),
            "resolved_prompt": self.resolved_prompt,
            "resolution_strategy": self.resolution_strategy,
            "resolution_reason": self.resolution_reason,
            "resolution_changed": self.resolution_changed,
        }


@dataclass(slots=True)
class ClarificationGateDecision:
    action: str
    should_clarify: bool
    trigger: str | None
    repeated_pattern: bool
    question_style: str
    reason: str | None = None
    top_alternative: dict[str, object] | None = None
    notes: list[str] = field(default_factory=list)
    degradation_mode: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload = {
            "action": self.action,
            "should_clarify": self.should_clarify,
            "trigger": self.trigger,
            "repeated_pattern": self.repeated_pattern,
            "question_style": self.question_style,
            "notes": list(self.notes),
        }
        if self.reason:
            payload["reason"] = self.reason
        if self.top_alternative:
            payload["top_alternative"] = dict(self.top_alternative)
        if self.degradation_mode:
            payload["degradation_mode"] = self.degradation_mode
        return payload


@dataclass(slots=True)
class ValidationTargetView:
    source: str
    query: str | None
    record_count: int
    lead: str | None
    quality: str
    summary: str | None
    top_match: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        payload = {
            "source": self.source,
            "query": self.query,
            "record_count": self.record_count,
            "lead": self.lead,
            "quality": self.quality,
            "summary": self.summary,
        }
        if self.top_match:
            payload["top_match"] = dict(self.top_match)
        return payload


@dataclass(slots=True)
class LightweightEvidenceModel:
    evidence_sources: list[dict[str, object]] = field(default_factory=list)
    evidence_strength: str | None = None
    contradiction_flags: list[str] = field(default_factory=list)
    missing_sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "evidence_sources": list(self.evidence_sources),
            "evidence_strength": self.evidence_strength,
            "contradiction_flags": list(self.contradiction_flags),
            "missing_sources": list(self.missing_sources),
        }


@dataclass(slots=True)
class EvidenceUnit:
    evidence_id: str
    source: str
    summary: str
    strength: str
    authority_score: float = 0.0
    topic: str | None = None
    created_at: str | None = None
    age_bucket: str | None = None
    decay_factor: float = 1.0
    reaffirmed: bool = False
    supports: list[str] = field(default_factory=list)
    contradicts: list[str] = field(default_factory=list)
    matched_fields: list[str] = field(default_factory=list)
    score_breakdown: dict[str, int] | None = None
    selected_as_anchor: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "evidence_id": self.evidence_id,
            "source": self.source,
            "summary": self.summary,
            "strength": self.strength,
            "authority_score": self.authority_score,
            "topic": self.topic,
            "created_at": self.created_at,
            "age_bucket": self.age_bucket,
            "decay_factor": self.decay_factor,
            "reaffirmed": self.reaffirmed,
            "supports": list(self.supports),
            "contradicts": list(self.contradicts),
            "matched_fields": list(self.matched_fields),
            "score_breakdown": dict(self.score_breakdown) if self.score_breakdown else None,
            "selected_as_anchor": self.selected_as_anchor,
        }


@dataclass(slots=True)
class ValidationContextResult:
    assistant_context: AssistantContext
    evidence_quality_score: float
    retrieval_lead_summary: str | None
    missing_context_note: str | None
    contradiction_flags: list[str]
    failure_modes: dict[str, bool] = field(default_factory=dict)
    targets: list[ValidationTargetView] = field(default_factory=list)
    evidence_model: LightweightEvidenceModel = field(default_factory=LightweightEvidenceModel)
    evidence_ledger: list[EvidenceUnit] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "assistant_context": self.assistant_context.to_dict(),
            "evidence_quality_score": self.evidence_quality_score,
            "retrieval_lead_summary": self.retrieval_lead_summary,
            "missing_context_note": self.missing_context_note,
            "contradiction_flags": list(self.contradiction_flags),
            "failure_modes": dict(self.failure_modes),
            "targets": [target.to_dict() for target in self.targets],
            "evidence_model": self.evidence_model.to_dict(),
            "evidence_ledger": [unit.to_dict() for unit in self.evidence_ledger],
        }


@dataclass(slots=True)
class ReasoningFrameAssembly:
    frame_type: str
    problem_interpretation: str
    local_context_summary: str | None
    grounded_interpretation: str | None
    working_hypothesis: str | None
    validation_plan: list[str] = field(default_factory=list)
    reasoning_frame: dict[str, str] = field(default_factory=dict)
    interaction_profile: dict[str, object] = field(default_factory=dict)
    profile_advice: dict[str, object] | None = None
    local_context_assessment: str | None = None
    grounding_strength: str | None = None
    route_quality: str | None = None
    route_status: str | None = None
    support_status: str | None = None
    tension_status: str | None = None
    uncertainty_posture: str | None = None
    confidence_posture: str | None = None
    failure_modes: dict[str, bool] = field(default_factory=dict)
    evidence_model: LightweightEvidenceModel = field(default_factory=LightweightEvidenceModel)
    evidence_ledger: list[EvidenceUnit] = field(default_factory=list)
    anchor_evidence_id: str | None = None
    supporting_evidence_id: str | None = None
    tension_evidence_ids: list[str] = field(default_factory=list)
    tension_resolution: dict[str, object] | None = None
    status_snapshot: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_type": self.frame_type,
            "problem_interpretation": self.problem_interpretation,
            "interaction_profile": dict(self.interaction_profile),
            "profile_advice": dict(self.profile_advice) if self.profile_advice else None,
            "local_context_summary": self.local_context_summary,
            "grounded_interpretation": self.grounded_interpretation,
            "working_hypothesis": self.working_hypothesis,
            "validation_plan": list(self.validation_plan),
            "reasoning_frame": dict(self.reasoning_frame),
            "local_context_assessment": self.local_context_assessment,
            "grounding_strength": self.grounding_strength,
            "route_quality": self.route_quality,
            "route_status": self.route_status,
            "support_status": self.support_status,
            "tension_status": self.tension_status,
            "uncertainty_posture": self.uncertainty_posture,
            "confidence_posture": self.confidence_posture,
            "failure_modes": dict(self.failure_modes),
            "evidence_model": self.evidence_model.to_dict(),
            "evidence_ledger": [unit.to_dict() for unit in self.evidence_ledger],
            "anchor_evidence_id": self.anchor_evidence_id,
            "supporting_evidence_id": self.supporting_evidence_id,
            "tension_evidence_ids": list(self.tension_evidence_ids),
            "tension_resolution": dict(self.tension_resolution) if self.tension_resolution else None,
            "status_snapshot": dict(self.status_snapshot) if self.status_snapshot else None,
        }


@dataclass(slots=True)
class ReasoningStatusSnapshot:
    route_quality: str | None = None
    grounding_strength: str | None = None
    local_context_assessment: str | None = None
    confidence_posture: str | None = None
    uncertainty_posture: str | None = None
    route_status: str | None = None
    support_status: str | None = None
    tension_status: str | None = None
    failure_modes: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "route_quality": self.route_quality,
            "grounding_strength": self.grounding_strength,
            "local_context_assessment": self.local_context_assessment,
            "confidence_posture": self.confidence_posture,
            "uncertainty_posture": self.uncertainty_posture,
            "route_status": self.route_status,
            "support_status": self.support_status,
            "tension_status": self.tension_status,
            "failure_modes": dict(self.failure_modes),
        }


@dataclass(slots=True)
class TensionResolutionResult:
    tension_detected: bool
    category: str | None
    resolution_path: str | None
    rationale: str | None
    status: str | None = None
    anchor_status: str | None = None
    recommended_action: str | None = None
    leading_hypothesis_label: str | None = None
    anchor_evidence_id: str | None = None
    tension_evidence_ids: list[str] = field(default_factory=list)
    alternate_hypotheses: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "tension_detected": self.tension_detected,
            "category": self.category,
            "resolution_path": self.resolution_path,
            "rationale": self.rationale,
            "status": self.status,
            "anchor_status": self.anchor_status,
            "recommended_action": self.recommended_action,
            "leading_hypothesis_label": self.leading_hypothesis_label,
            "anchor_evidence_id": self.anchor_evidence_id,
            "tension_evidence_ids": list(self.tension_evidence_ids),
            "alternate_hypotheses": list(self.alternate_hypotheses),
        }


@dataclass(slots=True)
class ResponseSynthesis:
    mode: str
    kind: str
    grounded_interpretation: str | None
    response_body: list[str] = field(default_factory=list)
    route_evidence_distinction: str | None = None
    uncertainty_note: str | None = None
    validation_advice: list[str] = field(default_factory=list)
    suggested_next_step: str | None = None
    confidence_posture: str | None = None
    route_status: str | None = None
    support_status: str | None = None
    tension_status: str | None = None
    interaction_profile: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "kind": self.kind,
            "interaction_profile": dict(self.interaction_profile),
            "grounded_interpretation": self.grounded_interpretation,
            "response_body": list(self.response_body),
            "route_evidence_distinction": self.route_evidence_distinction,
            "uncertainty_note": self.uncertainty_note,
            "validation_advice": list(self.validation_advice),
            "suggested_next_step": self.suggested_next_step,
            "confidence_posture": self.confidence_posture,
            "route_status": self.route_status,
            "support_status": self.support_status,
            "tension_status": self.tension_status,
        }


@dataclass(slots=True)
class ExecutionStageResult:
    mode: str
    kind: str
    execution_type: str
    executed: bool
    execution_metadata: dict[str, object] = field(default_factory=dict)
    result_summary: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "kind": self.kind,
            "execution_type": self.execution_type,
            "executed": self.executed,
            "execution_metadata": dict(self.execution_metadata),
            "result_summary": self.result_summary,
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class ResponsePackaging:
    mode: str
    kind: str
    package_type: str
    answer: str | None
    confidence: str | None
    uncertainty: str | None
    evidence_summary: str | None
    support_status: str | None = None
    tension_status: str | None = None
    route_summary: dict[str, object] = field(default_factory=dict)
    follow_up_suggestions: list[str] = field(default_factory=list)
    interaction_profile: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "kind": self.kind,
            "package_type": self.package_type,
            "interaction_profile": dict(self.interaction_profile),
            "answer": self.answer,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "evidence_summary": self.evidence_summary,
            "support_status": self.support_status,
            "tension_status": self.tension_status,
            "route_summary": dict(self.route_summary),
            "follow_up_suggestions": list(self.follow_up_suggestions),
        }


@dataclass(slots=True)
class PersistenceObservation:
    session_id: str
    prompt: str
    route_summary: dict[str, object] = field(default_factory=dict)
    alternative_routes: list[dict[str, object]] = field(default_factory=list)
    clarification_event: dict[str, object] | None = None
    retrieval_summary: dict[str, object] = field(default_factory=dict)
    reasoning_summary: dict[str, object] = field(default_factory=dict)
    execution_summary: dict[str, object] = field(default_factory=dict)
    response_summary: dict[str, object] = field(default_factory=dict)
    follow_up_linkage: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        payload = {
            "session_id": self.session_id,
            "prompt": self.prompt,
            "route_summary": dict(self.route_summary),
            "alternative_routes": list(self.alternative_routes),
            "retrieval_summary": dict(self.retrieval_summary),
            "reasoning_summary": dict(self.reasoning_summary),
            "execution_summary": dict(self.execution_summary),
            "response_summary": dict(self.response_summary),
            "follow_up_linkage": dict(self.follow_up_linkage),
        }
        if self.clarification_event is not None:
            payload["clarification_event"] = dict(self.clarification_event)
        return payload


@dataclass(slots=True)
class PipelineTrace:
    intake_frame: InputIntake | None = None
    nlu_frame: NLUExtraction | None = None
    vibe_catcher: VibeCatcherResult | None = None
    dialogue_management: DialogueManagementResult | None = None
    conversation_awareness: ConversationAwarenessResult | None = None
    low_confidence_recovery: LowConfidenceRecoveryResult | None = None
    srd_diagnostic: SRDDiagnosticResult | None = None
    empathy_model: EmpathyModelResult | None = None
    human_language_layer: HumanLanguageLayerResult | None = None
    state_interpretation: StateInterpretationResult | None = None
    state_control: StateControlResult | None = None
    thought_framing: ThoughtFramingResult | None = None
    intent_domain: IntentDomainResult | None = None
    response_depth: ResponseDepthResult | None = None
    conversation_phase: ConversationPhaseResult | None = None
    route_candidates: list[RouteCandidateView] = field(default_factory=list)
    route_decision: RouteDecisionView | None = None
    clarification_decision: ClarificationGateDecision | None = None
    validation_context: ValidationContextResult | None = None
    reasoning_frame: ReasoningFrameAssembly | None = None
    synthesis_output: ResponseSynthesis | None = None
    execution_package: ExecutionStageResult | None = None
    response_package: ResponsePackaging | None = None
    persistence_observation: PersistenceObservation | None = None
    resolved_prompt: str | None = None
    resolution_strategy: str | None = None
    resolution_reason: str | None = None
    resolution_changed: bool = False
    stage_sequence: list[str] = field(default_factory=list)
    stage_contracts: dict[str, StageContract] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "intake_frame": self.intake_frame.to_dict() if self.intake_frame else None,
            "nlu_frame": self.nlu_frame.to_dict() if self.nlu_frame else None,
            "vibe_catcher": self.vibe_catcher.to_dict() if self.vibe_catcher else None,
            "dialogue_management": (
                self.dialogue_management.to_dict() if self.dialogue_management else None
            ),
            "conversation_awareness": (
                self.conversation_awareness.to_dict() if self.conversation_awareness else None
            ),
            "low_confidence_recovery": (
                self.low_confidence_recovery.to_dict() if self.low_confidence_recovery else None
            ),
            "srd_diagnostic": (
                self.srd_diagnostic.to_dict() if self.srd_diagnostic else None
            ),
            "empathy_model": (
                self.empathy_model.to_dict() if self.empathy_model else None
            ),
            "human_language_layer": (
                self.human_language_layer.to_dict() if self.human_language_layer else None
            ),
            "state_interpretation": (
                self.state_interpretation.to_dict() if self.state_interpretation else None
            ),
            "state_control": (
                self.state_control.to_dict() if self.state_control else None
            ),
            "thought_framing": (
                self.thought_framing.to_dict() if self.thought_framing else None
            ),
            "intent_domain": (
                self.intent_domain.to_dict() if self.intent_domain else None
            ),
            "response_depth": (
                self.response_depth.to_dict() if self.response_depth else None
            ),
            "conversation_phase": (
                self.conversation_phase.to_dict() if self.conversation_phase else None
            ),
            "route_candidates": [candidate.to_dict() for candidate in self.route_candidates],
            "route_decision": self.route_decision.to_dict() if self.route_decision else None,
            "clarification_decision": self.clarification_decision.to_dict() if self.clarification_decision else None,
            "validation_context": self.validation_context.to_dict() if self.validation_context else None,
            "reasoning_frame": self.reasoning_frame.to_dict() if self.reasoning_frame else None,
            "synthesis_output": self.synthesis_output.to_dict() if self.synthesis_output else None,
            "execution_package": self.execution_package.to_dict() if self.execution_package else None,
            "response_package": self.response_package.to_dict() if self.response_package else None,
            "persistence_observation": self.persistence_observation.to_dict() if self.persistence_observation else None,
            "resolved_prompt": self.resolved_prompt,
            "resolution_strategy": self.resolution_strategy,
            "resolution_reason": self.resolution_reason,
            "resolution_changed": self.resolution_changed,
            "stage_sequence": list(self.stage_sequence),
            "stage_contracts": {
                stage_name: contract.to_dict()
                for stage_name, contract in self.stage_contracts.items()
            },
        }
