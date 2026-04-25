from __future__ import annotations

from typing import Any

from lumen.app.models import InteractionProfile
from lumen.nlu.prompt_nlu import PromptNLU
from lumen.reasoning.anti_roleplay_policy import AntiRoleplayPolicy
from lumen.reasoning.assistant_context import AssistantContext
from lumen.reasoning.conversation_awareness import ConversationAwarenessLayer
from lumen.reasoning.dialogue_manager import DialogueManager
from lumen.reasoning.empathy_model import EmpathyModel
from lumen.reasoning.human_language_layer import HumanLanguageLayer
from lumen.reasoning.evidence_builder import EvidenceBuilder
from lumen.reasoning.evidence_ledger import EvidenceLedgerBuilder
from lumen.reasoning.interaction_style_policy import InteractionStylePolicy
from lumen.reasoning.intent_domain_policy import IntentDomainPolicy
from lumen.reasoning.low_confidence_recovery import LowConfidenceRecoveryPolicy
from lumen.reasoning.response_variation import ResponseVariationLayer
from lumen.reasoning.reasoning_status import ReasoningStatusPolicy
from lumen.reasoning.srd_diagnostic import SRDDiagnostic
from lumen.reasoning.state_control import StateControlLayer
from lumen.reasoning.state_interpreter import StateInterpreter
from lumen.reasoning.thought_framing import ThoughtFramingLayer
from lumen.reasoning.vibe_catcher import VibeCatcher
from lumen.reasoning.reasoning_vocabulary import display_status_label
from lumen.reasoning.pipeline_models import (
    CanonicalPromptUnderstandingView,
    ClarificationGateDecision,
    ConversationPhaseResult,
    ConversationAwarenessResult,
    DialogueManagementResult,
    EmpathyModelResult,
    ExecutionStageResult,
    FrontHalfPipelineResult,
    InputIntake,
    LightweightEvidenceModel,
    LowConfidenceRecoveryResult,
    NLUExtraction,
    IntentDomainResult,
    PipelineTrace,
    PersistenceObservation,
    ReasoningFrameAssembly,
    ReasoningStatusSnapshot,
    ResponseDepthResult,
    ResponsePackaging,
    ResponseSynthesis,
    RouteCandidateView,
    RouteDecisionView,
    SRDDiagnosticResult,
    StageContract,
    ThoughtFramingResult,
    TensionResolutionResult,
    ValidationContextResult,
    ValidationTargetView,
    VibeCatcherResult,
)
from lumen.reasoning.tension_resolver import TensionResolver
from lumen.routing.domain_router import DomainRoute, DomainRouter
from lumen.routing.prompt_resolution import PromptResolver
from lumen.routing.route_adaptation import RouteAdaptationPolicy
from lumen.routing.route_clarification import RouteClarificationPolicy


class ReasoningPipeline:
    """Front-half reasoning pipeline for intake, NLU, and route decision staging."""

    def __init__(
        self,
        *,
        prompt_resolver: PromptResolver,
        domain_router: DomainRouter,
        archive_service=None,
        interaction_history_service=None,
        session_context_service=None,
    ) -> None:
        self.prompt_resolver = prompt_resolver
        self.domain_router = domain_router
        self.archive_service = archive_service
        self.interaction_history_service = interaction_history_service
        self.session_context_service = session_context_service
        self.evidence_builder = EvidenceBuilder()
        self.evidence_ledger_builder = EvidenceLedgerBuilder(evidence_builder=self.evidence_builder)
        self.reasoning_status_policy = ReasoningStatusPolicy()
        self.prompt_nlu = PromptNLU()
        self.vibe_catcher = VibeCatcher()
        self.dialogue_manager = DialogueManager()
        self.conversation_awareness = ConversationAwarenessLayer()
        self.low_confidence_recovery = LowConfidenceRecoveryPolicy()
        self.srd_diagnostic = SRDDiagnostic()
        self.state_interpreter = StateInterpreter()
        self.state_control = StateControlLayer()
        self.empathy_model = EmpathyModel()
        self.human_language_layer = HumanLanguageLayer()
        self.thought_framing = ThoughtFramingLayer()
        self.tension_resolver = TensionResolver()
        self.intent_domain_policy = IntentDomainPolicy()

    def run_front_half(
        self,
        *,
        prompt: str,
        session_id: str,
        interaction_profile: InteractionProfile,
        interaction_summary: dict[str, object],
        recent_interactions: list[dict[str, Any]],
        active_thread: dict[str, Any] | None,
    ) -> tuple[FrontHalfPipelineResult, DomainRoute]:
        resolution = self.prompt_resolver.resolve(prompt, active_thread=active_thread)
        vibe_catcher = self.vibe_catcher.catch(resolution.resolved_prompt)
        understanding = self.prompt_nlu.analyze(vibe_catcher.normalized_prompt)
        route = self.domain_router.route(
            understanding,
            recent_interactions=recent_interactions,
            active_thread=active_thread,
        )
        route_analysis = self.domain_router.analyze(
            understanding,
            recent_interactions=recent_interactions,
            active_thread=active_thread,
        )
        intake = InputIntake(
            raw_input=prompt,
            cleaned_input=vibe_catcher.normalized_prompt,
            detected_language=str(route_analysis.signals.get("detected_language") or "en"),
            session_id=session_id,
            interaction_profile=interaction_profile.to_dict(),
            session_context_snapshot={
                "interaction_count": int(interaction_summary.get("interaction_count", 0)),
                "clarification_count": int(interaction_summary.get("clarification_count", 0)),
                "latest_posture": interaction_summary.get("latest_posture"),
                "recent_posture_mix": interaction_summary.get("recent_posture_mix"),
            },
            active_thread=active_thread,
        )
        nlu = NLUExtraction(
            dominant_intent=str(route_analysis.signals.get("dominant_intent") or "unknown"),
            secondary_intents=self._secondary_intents(route_analysis.signals),
            topic=self._optional_str(route_analysis.signals.get("normalized_topic")),
            entities=list(route_analysis.signals.get("extracted_entities") or []),
            action_cues={
                "planning_score": int(route_analysis.signals.get("planning_score") or 0),
                "research_score": int(route_analysis.signals.get("research_score") or 0),
                "action_score": int(route_analysis.signals.get("action_score") or 0),
                "answer_score": int(route_analysis.signals.get("answer_score") or 0),
            },
            ambiguity_flags=self._ambiguity_flags(route, route_analysis),
            confidence_estimate=float(route_analysis.signals.get("intent_confidence") or 0.0),
            # Advisory only: this may explain a mismatch in the trace, but it must
            # never become the active runtime profile for synthesis or packaging.
            profile_advice=(
                understanding.profile_advice.to_dict()
                if understanding and interaction_profile.allow_suggestions and understanding.profile_advice
                else None
            ),
            profile_mismatch=(
                self._profile_mismatch(
                    interaction_profile=interaction_profile,
                    profile_advice=understanding.profile_advice.to_dict() if understanding and understanding.profile_advice else None,
                )
                if understanding and interaction_profile.allow_suggestions
                else False
            ),
            canonical_understanding=CanonicalPromptUnderstandingView.from_understanding(understanding),
        )
        dialogue_management = self.dialogue_manager.manage(
            prompt=resolution.resolved_prompt,
            nlu=nlu,
            route=route,
            interaction_count=int(interaction_summary.get("interaction_count", 0)),
            recent_interactions=recent_interactions,
            active_thread=active_thread,
        )
        conversation_awareness = self.conversation_awareness.assess(
            prompt=resolution.resolved_prompt,
            dialogue_management=dialogue_management,
            recent_interactions=recent_interactions,
            active_thread=active_thread,
        )
        state_interpretation = self.state_interpreter.interpret(
            prompt=resolution.resolved_prompt,
            dialogue_management=dialogue_management,
            conversation_awareness=conversation_awareness,
            recent_interactions=recent_interactions,
        )
        state_control = self.state_control.infer(
            prompt=resolution.resolved_prompt,
            nlu=nlu,
            dialogue_management=dialogue_management,
            conversation_awareness=conversation_awareness,
            state_interpretation=state_interpretation,
            recent_interactions=recent_interactions,
        )
        empathy_model = self.empathy_model.assess(
            prompt=resolution.resolved_prompt,
            conversation_awareness=conversation_awareness,
            state_control=state_control,
        )
        human_language_layer = self.human_language_layer.assess(
            prompt=resolution.resolved_prompt,
            conversation_awareness=conversation_awareness,
            empathy_model=empathy_model,
            state_control=state_control,
            interaction_profile=interaction_profile,
            active_thread=active_thread,
        )
        thought_framing = self.thought_framing.frame(
            prompt=resolution.resolved_prompt,
            nlu=nlu,
            route=route,
            dialogue_management=dialogue_management,
            conversation_awareness=conversation_awareness,
            state_control=state_control,
            recent_interactions=recent_interactions,
            active_thread=active_thread,
        )
        intent_domain, response_depth, conversation_phase = self.intent_domain_policy.infer(
            prompt=resolution.resolved_prompt,
            route=route,
            nlu=nlu,
            interaction_profile=interaction_profile,
            dialogue_management=dialogue_management,
            conversation_awareness=conversation_awareness,
            empathy_model=empathy_model,
            state_control=state_control,
            active_thread=active_thread,
            recent_interactions=recent_interactions,
        )
        route_candidates = [
            RouteCandidateView(
                mode=candidate.mode,
                kind=candidate.kind,
                confidence=candidate.confidence,
                reason=candidate.reason,
                source=candidate.source,
            )
            for candidate in route_analysis.candidates
        ]
        route_decision = RouteDecisionView(
            selected=dict(route.comparison or route_analysis.decision_summary.selected.to_dict()),
            alternatives=list((route.decision_summary or {}).get("alternatives") or [item.to_dict() for item in route_analysis.decision_summary.alternatives]),
            normalized_scores=[
                {
                    "mode": item.candidate.mode,
                    "kind": item.candidate.kind,
                    "source": item.candidate.source,
                    "normalized_score": item.normalized_score,
                }
                for item in route_analysis.comparisons
            ],
            caution_notes=self._caution_notes(route),
            weak_route=self._is_weak_route(route),
        )
        low_confidence_recovery = self.low_confidence_recovery.assess(
            nlu=nlu,
            route_decision=route_decision,
            vibe_catcher=vibe_catcher,
        )
        srd_diagnostic = self.srd_diagnostic.diagnose(
            dialogue_management=dialogue_management,
            conversation_awareness=conversation_awareness,
            route_decision=route_decision,
            low_confidence_recovery=low_confidence_recovery,
        )
        return (
            FrontHalfPipelineResult(
                intake=intake,
                nlu=nlu,
                route_candidates=route_candidates,
                route_decision=route_decision,
                resolved_prompt=resolution.resolved_prompt,
                resolution_strategy=resolution.strategy,
                resolution_reason=resolution.reason,
                resolution_changed=resolution.changed,
                vibe_catcher=vibe_catcher,
                dialogue_management=dialogue_management,
                conversation_awareness=conversation_awareness,
                low_confidence_recovery=low_confidence_recovery,
                srd_diagnostic=srd_diagnostic,
                empathy_model=empathy_model,
                human_language_layer=human_language_layer,
                state_interpretation=state_interpretation,
                state_control=state_control,
                thought_framing=thought_framing,
                intent_domain=intent_domain,
                response_depth=response_depth,
                conversation_phase=conversation_phase,
            ),
            route,
        )

    @staticmethod
    def create_trace(front_half: FrontHalfPipelineResult) -> PipelineTrace:
        trace = PipelineTrace(
            intake_frame=front_half.intake,
            nlu_frame=front_half.nlu,
            vibe_catcher=front_half.vibe_catcher,
            dialogue_management=front_half.dialogue_management,
            conversation_awareness=front_half.conversation_awareness,
            low_confidence_recovery=front_half.low_confidence_recovery,
            srd_diagnostic=front_half.srd_diagnostic,
            empathy_model=front_half.empathy_model,
            human_language_layer=front_half.human_language_layer,
            state_interpretation=front_half.state_interpretation,
            state_control=front_half.state_control,
            thought_framing=front_half.thought_framing,
            intent_domain=front_half.intent_domain,
            response_depth=front_half.response_depth,
            conversation_phase=front_half.conversation_phase,
            route_candidates=list(front_half.route_candidates),
            route_decision=front_half.route_decision,
            resolved_prompt=front_half.resolved_prompt,
            resolution_strategy=front_half.resolution_strategy,
            resolution_reason=front_half.resolution_reason,
            resolution_changed=front_half.resolution_changed,
            stage_sequence=[
                "input_intake",
                "nlu_extraction",
                "vibe_catcher",
                "route_candidate_generation",
                "dialogue_management",
                "conversation_awareness",
                "low_confidence_recovery",
                "srd_diagnostic",
                "state_interpreter",
                "state_control",
                "empathy_model",
                "human_language_layer",
                "thought_framing",
                "intent_domain_policy",
                "route_scoring_and_comparison",
            ],
        )
        trace.stage_contracts["input_intake"] = ReasoningPipeline._stage_contract(
            stage_name="input_intake",
            required_inputs=["prompt", "session_id", "interaction_profile", "interaction_summary", "active_thread"],
            produced_outputs=[
                "raw_input",
                "cleaned_input",
                "detected_language",
                "interaction_profile",
                "session_context_snapshot",
                "active_thread",
            ],
            confidence_signal="detected_language",
            failure_state="missing_or_empty_input",
        )
        trace.stage_contracts["nlu_extraction"] = ReasoningPipeline._stage_contract(
            stage_name="nlu_extraction",
            required_inputs=["cleaned_input", "detected_language"],
            produced_outputs=[
                "dominant_intent",
                "secondary_intents",
                "topic",
                "entities",
                "action_cues",
                "ambiguity_flags",
                "confidence_estimate",
                "profile_advice",
                "profile_mismatch",
            ],
            confidence_signal="confidence_estimate",
            failure_state="unknown_or_low_confidence_intent",
        )
        trace.stage_contracts["vibe_catcher"] = ReasoningPipeline._stage_contract(
            stage_name="vibe_catcher",
            required_inputs=["resolved_prompt"],
            produced_outputs=[
                "normalized_prompt",
                "directional_signals",
                "interpretation_confidence",
                "low_confidence",
                "recovery_hint",
            ],
            confidence_signal=(
                str(front_half.vibe_catcher.interpretation_confidence)
                if front_half.vibe_catcher is not None
                else None
            ),
            failure_state=(
                "low_confidence_input"
                if front_half.vibe_catcher is not None and front_half.vibe_catcher.low_confidence
                else None
            ),
        )
        trace.stage_contracts["route_candidate_generation"] = ReasoningPipeline._stage_contract(
            stage_name="route_candidate_generation",
            required_inputs=["vibe_catcher", "nlu_extraction", "recent_interactions", "active_thread"],
            produced_outputs=["route_candidates"],
            confidence_signal="candidate_confidence",
            failure_state="no_eligible_route_candidates",
        )
        trace.stage_contracts["dialogue_management"] = ReasoningPipeline._stage_contract(
            stage_name="dialogue_management",
            required_inputs=["resolved_prompt", "nlu_extraction", "route_decision", "recent_interactions", "active_thread"],
            produced_outputs=[
                "interaction_mode",
                "idea_state",
                "response_strategy",
                "synthesis_checkpoint_due",
                "checkpoint_reason",
            ],
            confidence_signal=(
                front_half.dialogue_management.response_strategy
                if front_half.dialogue_management is not None
                else None
            ),
            failure_state=None,
        )
        trace.stage_contracts["conversation_awareness"] = ReasoningPipeline._stage_contract(
            stage_name="conversation_awareness",
            required_inputs=["resolved_prompt", "dialogue_management", "recent_interactions", "active_thread"],
            produced_outputs=[
                "recent_intent_pattern",
                "conversation_momentum",
                "unresolved_thread_open",
                "unresolved_thread_reason",
                "branch_state",
                "return_target",
                "adaptive_posture",
            ],
            confidence_signal=(
                front_half.conversation_awareness.adaptive_posture
                if front_half.conversation_awareness is not None
                else None
            ),
            failure_state=None,
        )
        trace.stage_contracts["low_confidence_recovery"] = ReasoningPipeline._stage_contract(
            stage_name="low_confidence_recovery",
            required_inputs=["vibe_catcher", "nlu_extraction", "route_decision"],
            produced_outputs=[
                "recovery_mode",
                "acknowledge_partial_understanding",
                "clarifying_question_style",
                "rationale",
            ],
            confidence_signal=(
                front_half.low_confidence_recovery.recovery_mode
                if front_half.low_confidence_recovery is not None
                else None
            ),
            failure_state=(
                "clarification_recovery"
                if front_half.low_confidence_recovery is not None
                and front_half.low_confidence_recovery.recovery_mode in {"soft_clarify", "hard_clarify"}
                else None
            ),
        )
        trace.stage_contracts["srd_diagnostic"] = ReasoningPipeline._stage_contract(
            stage_name="srd_diagnostic",
            required_inputs=["dialogue_management", "conversation_awareness", "route_decision", "low_confidence_recovery"],
            produced_outputs=[
                "stage",
                "failure_types",
                "escalation_risk",
                "repairable_here",
                "preserve_agency",
                "should_exit_early",
                "rationale",
            ],
            confidence_signal=(
                front_half.srd_diagnostic.escalation_risk
                if front_half.srd_diagnostic is not None
                else None
            ),
            failure_state=(
                "structural_disruption"
                if front_half.srd_diagnostic is not None and front_half.srd_diagnostic.failure_types
                else None
            ),
        )
        trace.stage_contracts["state_interpreter"] = ReasoningPipeline._stage_contract(
            stage_name="state_interpreter",
            required_inputs=["resolved_prompt", "dialogue_management", "conversation_awareness", "recent_interactions"],
            produced_outputs=[
                "trigger",
                "trigger_signals",
                "repeated_failure_detected",
                "uncertainty_stacking",
                "humor_candidate",
            ],
            confidence_signal=(
                front_half.state_interpretation.trigger
                if front_half.state_interpretation is not None
                else None
            ),
            failure_state=(
                "uncertainty_stacking"
                if front_half.state_interpretation is not None and front_half.state_interpretation.uncertainty_stacking
                else None
            ),
        )
        trace.stage_contracts["state_control"] = ReasoningPipeline._stage_contract(
            stage_name="state_control",
            required_inputs=["state_interpretation", "resolved_prompt", "nlu_extraction", "dialogue_management", "conversation_awareness", "recent_interactions"],
            produced_outputs=[
                "core_state",
                "trigger",
                "anti_spiral_active",
                "anti_spiral_reason",
                "response_bias",
                "humor_allowed",
            ],
            confidence_signal=(
                front_half.state_control.core_state
                if front_half.state_control is not None
                else None
            ),
            failure_state=(
                "anti_spiral_active"
                if front_half.state_control is not None and front_half.state_control.anti_spiral_active
                else None
            ),
        )
        trace.stage_contracts["empathy_model"] = ReasoningPipeline._stage_contract(
            stage_name="empathy_model",
            required_inputs=["resolved_prompt", "conversation_awareness", "state_control"],
            produced_outputs=[
                "emotional_signal_detected",
                "feeling_label",
                "probable_cause",
                "response_sensitivity",
                "grounded_acknowledgment",
            ],
            confidence_signal=(
                front_half.empathy_model.response_sensitivity
                if front_half.empathy_model is not None
                else None
            ),
            failure_state=None,
        )
        trace.stage_contracts["human_language_layer"] = ReasoningPipeline._stage_contract(
            stage_name="human_language_layer",
            required_inputs=[
                "resolved_prompt",
                "conversation_awareness",
                "empathy_model",
                "state_control",
                "interaction_profile",
                "active_thread",
            ],
            produced_outputs=[
                "flow_style",
                "context_continuity",
                "emotional_alignment",
                "user_energy",
                "correction_detected",
                "epistemic_stance",
                "stance_confidence",
                "response_brevity",
            ],
            confidence_signal=(
                front_half.human_language_layer.epistemic_stance
                if front_half.human_language_layer is not None
                else None
            ),
            failure_state=None,
        )
        trace.stage_contracts["thought_framing"] = ReasoningPipeline._stage_contract(
            stage_name="thought_framing",
            required_inputs=[
                "resolved_prompt",
                "nlu_extraction",
                "route_decision",
                "dialogue_management",
                "conversation_awareness",
                "state_control",
                "empathy_model",
                "human_language_layer",
                "recent_interactions",
                "active_thread",
            ],
            produced_outputs=[
                "response_kind_label",
                "conversation_activity",
                "research_questions",
                "branch_return_hint",
                "checkpoint_summary",
            ],
            confidence_signal=(
                front_half.thought_framing.response_kind_label
                if front_half.thought_framing is not None
                else None
            ),
            failure_state=None,
        )
        trace.stage_contracts["intent_domain_policy"] = ReasoningPipeline._stage_contract(
            stage_name="intent_domain_policy",
            required_inputs=[
                "resolved_prompt",
                "route_decision",
                "dialogue_management",
                "conversation_awareness",
                "interaction_profile",
            ],
            produced_outputs=[
                "intent_domain",
                "response_depth",
                "conversation_phase",
            ],
            confidence_signal=(
                str(front_half.intent_domain.confidence)
                if front_half.intent_domain is not None
                else None
            ),
            failure_state=None,
        )
        trace.stage_contracts["route_scoring_and_comparison"] = ReasoningPipeline._stage_contract(
            stage_name="route_scoring_and_comparison",
            required_inputs=["route_candidates", "session_continuity_signals", "route_scoring_signals"],
            produced_outputs=["route_decision", "normalized_scores", "caution_notes", "weak_route"],
            confidence_signal="normalized_score",
            failure_state="weak_route" if front_half.route_decision.weak_route else None,
        )
        return trace

    @staticmethod
    def record_clarification_decision(
        trace: PipelineTrace,
        decision: ClarificationGateDecision,
    ) -> ClarificationGateDecision:
        trace.clarification_decision = decision
        trace.stage_contracts["clarification_gate"] = ReasoningPipeline._stage_contract(
            stage_name="clarification_gate",
            required_inputs=["route_decision", "interaction_summary"],
            produced_outputs=["clarification_decision"],
            confidence_signal="should_clarify",
            failure_state=decision.trigger or ("high_ambiguity" if decision.should_clarify else None),
        )
        return decision

    @staticmethod
    def record_validation_context(
        trace: PipelineTrace,
        validation_context: ValidationContextResult,
    ) -> ValidationContextResult:
        trace.validation_context = validation_context
        trace.stage_contracts["context_retrieval_validation"] = ReasoningPipeline._stage_contract(
            stage_name="context_retrieval_validation",
            required_inputs=["route_decision", "session_context", "archive_context", "interaction_context"],
            produced_outputs=[
                "assistant_context",
                "evidence_quality_score",
                "retrieval_lead_summary",
                "missing_context_note",
                "contradiction_flags",
                "failure_modes",
                "targets",
                "evidence_model",
            ],
            confidence_signal="evidence_quality_score",
            failure_state=ReasoningPipeline._failure_state_label(validation_context.failure_modes),
        )
        return validation_context

    @staticmethod
    def record_reasoning_frame(
        trace: PipelineTrace,
        assembly: ReasoningFrameAssembly,
    ) -> ReasoningFrameAssembly:
        trace.reasoning_frame = assembly
        trace.stage_contracts["reasoning_frame_assembly"] = ReasoningPipeline._stage_contract(
            stage_name="reasoning_frame_assembly",
            required_inputs=["validation_context", "route_decision", "problem_kind"],
            produced_outputs=[
                "frame_type",
                "problem_interpretation",
                "local_context_summary",
                "grounded_interpretation",
                "working_hypothesis",
                "validation_plan",
                "uncertainty_posture",
                "confidence_posture",
                "failure_modes",
                "evidence_model",
            ],
            confidence_signal=assembly.confidence_posture,
            failure_state=ReasoningPipeline._failure_state_label(assembly.failure_modes),
        )
        return assembly

    @staticmethod
    def record_synthesis(
        trace: PipelineTrace,
        synthesis: ResponseSynthesis,
    ) -> ResponseSynthesis:
        trace.synthesis_output = synthesis
        trace.stage_contracts["synthesis"] = ReasoningPipeline._stage_contract(
            stage_name="synthesis",
            required_inputs=["reasoning_frame", "response_payload"],
            produced_outputs=[
                "grounded_interpretation",
                "response_body",
                "route_evidence_distinction",
                "uncertainty_note",
                "validation_advice",
                "suggested_next_step",
            ],
            confidence_signal=synthesis.confidence_posture,
            failure_state=synthesis.route_evidence_distinction,
        )
        return synthesis

    @staticmethod
    def record_execution_package(
        trace: PipelineTrace,
        execution: ExecutionStageResult,
    ) -> ExecutionStageResult:
        trace.execution_package = execution
        trace.stage_contracts["execution"] = ReasoningPipeline._stage_contract(
            stage_name="execution",
            required_inputs=["selected_route", "response_payload"],
            produced_outputs=[
                "execution_type",
                "executed",
                "execution_metadata",
                "result_summary",
                "warnings",
            ],
            confidence_signal="executed",
            failure_state="execution_warning" if execution.warnings else None,
        )
        return execution

    @staticmethod
    def record_response_package(
        trace: PipelineTrace,
        packaging: ResponsePackaging,
    ) -> ResponsePackaging:
        trace.response_package = packaging
        trace.stage_contracts["response_packaging"] = ReasoningPipeline._stage_contract(
            stage_name="response_packaging",
            required_inputs=["execution_package", "response_payload", "route_decision"],
            produced_outputs=[
                "package_type",
                "answer",
                "confidence",
                "uncertainty",
                "evidence_summary",
                "route_summary",
                "follow_up_suggestions",
            ],
            confidence_signal=packaging.confidence,
            failure_state=None,
        )
        return packaging

    @staticmethod
    def record_persistence_observation(
        trace: PipelineTrace,
        observation: PersistenceObservation,
    ) -> PersistenceObservation:
        trace.persistence_observation = observation
        trace.stage_contracts["persistence_observability"] = ReasoningPipeline._stage_contract(
            stage_name="persistence_observability",
            required_inputs=["response_package", "execution_package", "route_decision"],
            produced_outputs=[
                "route_summary",
                "alternative_routes",
                "clarification_event",
                "retrieval_summary",
                "reasoning_summary",
                "execution_summary",
                "response_summary",
                "follow_up_linkage",
            ],
            confidence_signal="response_summary",
            failure_state=None,
        )
        return observation

    def run_clarification_gate(
        self,
        *,
        route: DomainRoute,
        interaction_summary: dict[str, object],
        low_confidence_recovery: LowConfidenceRecoveryResult | None = None,
    ) -> ClarificationGateDecision:
        clarification_count = int(interaction_summary.get("clarification_count", 0))
        clarification_drift = str(interaction_summary.get("clarification_drift") or "")
        recent_clarification_mix = str(interaction_summary.get("recent_clarification_mix") or "")
        repeated_clarification = clarification_count >= 1

        if low_confidence_recovery is not None and low_confidence_recovery.recovery_mode in {"soft_clarify", "hard_clarify"}:
            if self._should_suppress_repeated_clarification(
                clarification_count=clarification_count,
                clarification_drift=clarification_drift,
                recent_clarification_mix=recent_clarification_mix,
            ):
                return ClarificationGateDecision(
                    action="degrade",
                    should_clarify=False,
                    trigger="clarification_suppressed",
                    repeated_pattern=True,
                    question_style="narrow_and_proceed",
                    reason=(
                        "Repeated clarification pressure was suppressed so the system can give a narrowed, low-confidence answer instead of looping."
                    ),
                    top_alternative=self._top_alternative(route),
                    notes=self._clarification_notes(interaction_summary),
                    degradation_mode="narrow_and_proceed",
                )
            return ClarificationGateDecision(
                action="clarify",
                should_clarify=True,
                trigger="low_confidence_recovery",
                repeated_pattern=repeated_clarification,
                question_style=str(low_confidence_recovery.clarifying_question_style or "directional_recovery"),
                reason=low_confidence_recovery.rationale or "Low-confidence recovery requested clarification before committing to a route.",
                top_alternative=self._top_alternative(route),
                notes=self._clarification_notes(interaction_summary),
            )
        if RouteClarificationPolicy.base_should_clarify(route):
            return ClarificationGateDecision(
                action="clarify",
                should_clarify=True,
                trigger="base_threshold",
                repeated_pattern=repeated_clarification,
                question_style="disambiguate_route",
                reason="Base clarification threshold was triggered by an ambiguous route.",
                top_alternative=self._top_alternative(route),
                notes=self._clarification_notes(interaction_summary),
            )

        if not RouteClarificationPolicy.is_ambiguous(route):
            return ClarificationGateDecision(
                action="proceed",
                should_clarify=False,
                trigger=None,
                repeated_pattern=False,
                question_style="none",
                reason="Route was not ambiguous enough to trigger clarification.",
            )

        if not RouteClarificationPolicy.in_adaptive_scope(route):
            return ClarificationGateDecision(
                action="proceed",
                should_clarify=False,
                trigger=None,
                repeated_pattern=False,
                question_style="none",
                reason="Ambiguity was present, but the route source is not in the heuristic clarification scope.",
                top_alternative=self._top_alternative(route),
            )

        nlu_uncertainty_high = self._session_nlu_uncertainty_high(interaction_summary)
        retrieval_semantic_bias_high = self._session_retrieval_semantic_bias_high(interaction_summary)

        if clarification_count < 1 and not nlu_uncertainty_high and not retrieval_semantic_bias_high:
            return ClarificationGateDecision(
                action="proceed",
                should_clarify=False,
                trigger=None,
                repeated_pattern=False,
                question_style="none",
                reason="No repeated clarification pattern or uncertainty signal was present.",
                top_alternative=self._top_alternative(route),
            )

        if (
            not nlu_uncertainty_high
            and not retrieval_semantic_bias_high
            and recent_clarification_mix not in {"clarification_heavy_mixed", "mixed"}
            and clarification_drift != "increasing"
        ):
            return ClarificationGateDecision(
                action="proceed",
                should_clarify=False,
                trigger=None,
                repeated_pattern=clarification_count >= 1,
                question_style="none",
                reason="Clarification history exists, but it has not become strong enough to adapt the threshold.",
                top_alternative=self._top_alternative(route),
            )

        threshold = RouteClarificationPolicy.select_adaptive_threshold(
            nlu_uncertainty_high=nlu_uncertainty_high,
            retrieval_semantic_bias_high=retrieval_semantic_bias_high,
        )

        should_clarify = route.confidence <= threshold.threshold
        return ClarificationGateDecision(
            action="clarify" if should_clarify else "proceed",
            should_clarify=should_clarify,
            trigger=threshold.trigger if should_clarify else None,
            repeated_pattern=clarification_count >= 1,
            question_style="adaptive_disambiguation" if should_clarify else "none",
            reason=(
                f"Adaptive clarification trigger '{threshold.trigger}' evaluated route confidence against threshold {threshold.threshold:.2f}."
            ),
            top_alternative=self._top_alternative(route),
            notes=self._clarification_notes(interaction_summary),
        )

    @staticmethod
    def _should_suppress_repeated_clarification(
        *,
        clarification_count: int,
        clarification_drift: str,
        recent_clarification_mix: str,
    ) -> bool:
        if clarification_count >= 2:
            return True
        if clarification_count < 1:
            return False
        if clarification_drift == "increasing":
            return True
        if recent_clarification_mix == "clarification_heavy_mixed":
            return True
        return False

    def build_validation_context(
        self,
        *,
        prompt: str,
        session_id: str,
        route: DomainRoute,
        interaction_summary: dict[str, object],
        interaction_profile: InteractionProfile | None = None,
    ) -> ValidationContextResult:
        if (
            self.archive_service is None
            or self.interaction_history_service is None
            or self.session_context_service is None
        ):
            raise RuntimeError("Validation context requires archive, interaction history, and session context services.")

        active_thread = self.session_context_service.get_active_thread(session_id)
        active_thread = self._filter_active_thread_for_route(
            route=route,
            prompt=prompt,
            active_thread=active_thread,
        )
        active_profile = interaction_profile or InteractionProfile.default()
        archive_context = self._build_archive_context(session_id=session_id)
        archive_context["active_thread"] = active_thread

        targeted = self._build_archive_validation_context(
            prompt=prompt,
            session_id=session_id,
            active_thread=active_thread,
            interaction_summary=interaction_summary,
            interaction_profile=active_profile,
        )
        if "record_count" in targeted:
            targeted = {
                **targeted,
                "matched_record_count": targeted["record_count"],
            }
            del targeted["record_count"]

        interaction_context = self._build_interaction_context(
            prompt,
            route=route,
            active_thread=active_thread,
            session_id=session_id,
            interaction_summary=interaction_summary,
        )

        archive_context.update(targeted)
        archive_context.update(interaction_context)
        archive_context["route"] = route.to_metadata().to_dict()
        assistant_context = AssistantContext.from_mapping(archive_context)

        targets = self._validation_targets(
            prompt=prompt,
            assistant_context=assistant_context,
        )
        missing_context_note = None
        if not any(target.record_count > 0 for target in targets if target.source != "active_thread"):
            if not active_thread:
                missing_context_note = "No archive, interaction, or active-thread context was available for validation."
            else:
                missing_context_note = "Validation is relying mostly on active-thread continuity because archive and interaction matches were sparse."

        contradiction_flags: list[str] = []
        if assistant_context.route and ((assistant_context.route.get("ambiguity") or {}).get("ambiguous")):
            contradiction_flags.append("ambiguous_route")
        if (
            assistant_context.top_matches
            and assistant_context.top_interaction_matches
            and self._retrieval_lead_label({"top_matches": assistant_context.top_matches})
            != self._retrieval_lead_label({"top_interaction_matches": assistant_context.top_interaction_matches})
        ):
            contradiction_flags.append("mixed_retrieval_leads")
        contradiction_flags.extend(
            self._deep_contradiction_flags(
                assistant_context=assistant_context,
                interaction_profile=active_profile,
            )
        )
        contradiction_flags = list(dict.fromkeys(contradiction_flags))

        retrieval_lead_summary = self._validation_retrieval_summary(assistant_context)
        evidence_quality_score = self._evidence_quality_score(assistant_context, active_thread=active_thread)
        failure_modes = self._failure_modes(
            assistant_context=assistant_context,
            evidence_quality_score=evidence_quality_score,
            contradiction_flags=contradiction_flags,
        )
        return ValidationContextResult(
            assistant_context=assistant_context,
            evidence_quality_score=evidence_quality_score,
            retrieval_lead_summary=retrieval_lead_summary,
            missing_context_note=missing_context_note,
            contradiction_flags=contradiction_flags,
            failure_modes=failure_modes,
            targets=targets,
            evidence_model=self._build_evidence_model(
                targets=targets,
                contradiction_flags=contradiction_flags,
                evidence_quality_score=evidence_quality_score,
            ),
            evidence_ledger=self.evidence_ledger_builder.build(
                assistant_context=assistant_context,
                targets=targets,
                contradiction_flags=contradiction_flags,
            ),
        )

    def assemble_reasoning_frame(
        self,
        *,
        prompt: str,
        route: DomainRoute,
        kind: str,
        interaction_profile: InteractionProfile,
        validation_context: ValidationContextResult,
    ) -> ReasoningFrameAssembly:
        context = validation_context.assistant_context
        mode = route.mode
        local_context_summary = self.evidence_builder.summarize_local_context(context=context)
        grounded_interpretation = self.evidence_builder.synthesize_interpretation(
            mode=mode,
            context=context,
        )
        working_hypothesis = self.evidence_builder.build_working_hypothesis(
            mode=mode,
            context=context,
        )
        reasoning_frame = self.evidence_builder.build_reasoning_frame(context=context)
        local_context_assessment = self.evidence_builder.assess_local_context(context=context)
        grounding_strength = self.evidence_builder.grounding_strength(context=context)
        route_quality = self.evidence_builder.route_quality_label(context=context)
        ledger_refs = self.evidence_ledger_builder.select_references(
            reasoning_frame=reasoning_frame,
            evidence_ledger=validation_context.evidence_ledger,
        )
        tension_resolution = self.tension_resolver.resolve(
            interaction_profile=interaction_profile,
            evidence_ledger=validation_context.evidence_ledger,
            anchor_evidence_id=ledger_refs["anchor_evidence_id"],
            tension_evidence_ids=ledger_refs["tension_evidence_ids"],
            contradiction_flags=validation_context.contradiction_flags,
            failure_modes=validation_context.failure_modes,
        )
        status_snapshot = self._build_reasoning_status_snapshot(
            route_strength=str((context.route or {}).get("strength") or ""),
            route_quality=route_quality,
            grounding_strength=grounding_strength,
            local_context_assessment=local_context_assessment,
            validation_context=validation_context,
            route_ambiguity=bool((context.route or {}).get("ambiguity")),
            tension_resolution=tension_resolution,
        )
        return ReasoningFrameAssembly(
            frame_type=self._frame_type(mode=mode, kind=kind),
            problem_interpretation=self._problem_interpretation(
                prompt=prompt,
                mode=mode,
                kind=kind,
                route_quality=route_quality,
            ),
            local_context_summary=local_context_summary,
            grounded_interpretation=grounded_interpretation,
            working_hypothesis=working_hypothesis,
            validation_plan=self._validation_plan(
                context=context,
                validation_context=validation_context,
                route_quality=route_quality,
                grounding_strength=grounding_strength,
                interaction_profile=interaction_profile,
            ),
            reasoning_frame=reasoning_frame,
            interaction_profile=interaction_profile.to_dict(),
            profile_advice=(
                dict(validation_context.assistant_context.active_thread.get("profile_advice"))
                if isinstance(validation_context.assistant_context.active_thread, dict)
                and isinstance(validation_context.assistant_context.active_thread.get("profile_advice"), dict)
                else None
            ),
            local_context_assessment=local_context_assessment,
            grounding_strength=grounding_strength,
            route_quality=route_quality,
            route_status=status_snapshot.route_status,
            support_status=status_snapshot.support_status,
            tension_status=status_snapshot.tension_status,
            uncertainty_posture=status_snapshot.uncertainty_posture,
            confidence_posture=status_snapshot.confidence_posture,
            failure_modes=dict(validation_context.failure_modes),
            evidence_model=validation_context.evidence_model,
            evidence_ledger=validation_context.evidence_ledger,
            anchor_evidence_id=ledger_refs["anchor_evidence_id"],
            supporting_evidence_id=ledger_refs["supporting_evidence_id"],
            tension_evidence_ids=ledger_refs["tension_evidence_ids"],
            tension_resolution=tension_resolution.to_dict(),
            status_snapshot=status_snapshot.to_dict(),
        )

    def synthesize_response(
        self,
        *,
        mode: str,
        kind: str,
        reasoning_frame_assembly: ReasoningFrameAssembly,
        response_payload: dict[str, object],
    ) -> ResponseSynthesis:
        response_body = list(
            response_payload.get("steps")
            if mode == "planning"
            else response_payload.get("findings")
            or []
        )
        suggested_next_step = (
            str(response_payload.get("next_action") or "").strip()
            if mode == "planning"
            else str(response_payload.get("recommendation") or "").strip()
        ) or None
        route_evidence_distinction = self._route_evidence_distinction(
            route_quality=reasoning_frame_assembly.route_quality,
            grounding_strength=reasoning_frame_assembly.grounding_strength,
        )
        validation_advice = [
            item
            for item in reasoning_frame_assembly.validation_plan
            if item and (
                "validate" in item.lower()
                or "resolve" in item.lower()
                or "exploratory" in item.lower()
            )
        ]
        validation_advice = self._augment_validation_advice_for_tension(
            validation_advice,
            tension_resolution=reasoning_frame_assembly.tension_resolution,
        )
        active_profile = dict(reasoning_frame_assembly.interaction_profile)
        response_body = self._prepend_ledger_context(
            response_body,
            reasoning_frame_assembly=reasoning_frame_assembly,
            interaction_profile=active_profile,
        )
        response_body = self._prepend_evidence_context(
            response_body,
            evidence_model=reasoning_frame_assembly.evidence_model,
            interaction_profile=active_profile,
        )
        response_body = self._prepend_tension_resolution_context(
            response_body,
            tension_resolution=reasoning_frame_assembly.tension_resolution,
            interaction_profile=active_profile,
        )
        response_body = self._prepend_evidence_strength_context(
            response_body,
            evidence_model=reasoning_frame_assembly.evidence_model,
            interaction_profile=active_profile,
        )
        return ResponseSynthesis(
            mode=mode,
            kind=kind,
            grounded_interpretation=reasoning_frame_assembly.grounded_interpretation,
            response_body=self._profile_adjusted_body(
                response_body,
                interaction_profile=active_profile,
                grounded_interpretation=reasoning_frame_assembly.grounded_interpretation,
                tension_resolution=reasoning_frame_assembly.tension_resolution,
                suggested_next_step=suggested_next_step,
                validation_advice=validation_advice,
                route_evidence_distinction=route_evidence_distinction,
            ),
            route_evidence_distinction=route_evidence_distinction,
            uncertainty_note=(
                str(response_payload.get("uncertainty_note") or "").strip()
                or reasoning_frame_assembly.uncertainty_posture
            ),
            validation_advice=self._profile_adjusted_validation_advice(
                validation_advice,
                interaction_profile=active_profile,
                suggested_next_step=suggested_next_step,
            ),
            suggested_next_step=suggested_next_step,
            confidence_posture=reasoning_frame_assembly.confidence_posture,
            route_status=reasoning_frame_assembly.route_status,
            support_status=reasoning_frame_assembly.support_status,
            tension_status=reasoning_frame_assembly.tension_status,
            interaction_profile=active_profile,
        )

    def package_execution_stage(
        self,
        *,
        mode: str,
        kind: str,
        response_payload: dict[str, object],
    ) -> ExecutionStageResult:
        if mode == "tool":
            tool_execution = response_payload.get("tool_execution")
            execution_metadata = dict(tool_execution) if isinstance(tool_execution, dict) else {}
            warnings: list[str] = []
            tool_result = response_payload.get("tool_result")
            if getattr(tool_result, "error", None):
                warnings.append(str(getattr(tool_result, "error")))
            if response_payload.get("tool_route_origin") == "nlu_hint_alias":
                warnings.append("Tool execution was reached through NLU hint alias resolution.")
            return ExecutionStageResult(
                mode=mode,
                kind=kind,
                execution_type="tool_call",
                executed=True,
                execution_metadata=execution_metadata,
                result_summary=str(response_payload.get("summary") or "").strip() or None,
                warnings=warnings,
            )
        return ExecutionStageResult(
            mode=mode,
            kind=kind,
            execution_type="reasoned_response",
            executed=True,
            execution_metadata={
                "response_mode": mode,
                "confidence_posture": response_payload.get("confidence_posture"),
            },
            result_summary=str(response_payload.get("summary") or "").strip() or None,
            warnings=[],
        )

    def package_response(
        self,
        *,
        mode: str,
        kind: str,
        route: DomainRoute,
        response_payload: dict[str, object],
        interaction_profile: InteractionProfile | None = None,
    ) -> ResponsePackaging:
        active_profile = interaction_profile or InteractionProfile.default()
        answer = self._packaged_answer(
            mode=mode,
            response_payload=response_payload,
            interaction_profile=active_profile,
        )
        support_status = str(
            response_payload.get("support_status")
            or response_payload.get("grounding_status")
            or response_payload.get("confidence_posture")
            or ""
        ).strip() or None
        tension_status = str(response_payload.get("tension_status") or "").strip() or None
        confidence = support_status
        uncertainty = tension_status if tension_status and tension_status != "stable" else (
            str(response_payload.get("uncertainty_note") or "").strip() or None
        )
        evidence_summary = self._packaged_evidence_summary(
            response_payload=response_payload,
            interaction_profile=active_profile,
        )
        route_metadata = route.to_metadata().to_dict()
        route_summary = self._packaged_route_summary(
            route_metadata=route_metadata,
            route_status=str(response_payload.get("route_status") or "").strip() or None,
            interaction_profile=active_profile,
        )
        return ResponsePackaging(
            mode=mode,
            kind=kind,
            package_type=self._package_type(mode=mode, interaction_profile=active_profile),
            answer=answer,
            confidence=confidence,
            uncertainty=uncertainty,
            evidence_summary=evidence_summary,
            support_status=support_status,
            tension_status=tension_status,
            route_summary=route_summary,
            follow_up_suggestions=self._follow_up_suggestions(
                mode=mode,
                response_payload=response_payload,
                interaction_profile=active_profile,
            ),
            interaction_profile=active_profile.to_dict(),
        )

    def package_persistence_observation(
        self,
        *,
        session_id: str,
        prompt: str,
        front_half: FrontHalfPipelineResult,
        route: DomainRoute,
        clarification_decision: ClarificationGateDecision | None,
        validation_context: ValidationContextResult | None,
        reasoning_frame_assembly: ReasoningFrameAssembly | None,
        execution_stage: ExecutionStageResult | None,
        response_packaging: ResponsePackaging | None,
    ) -> PersistenceObservation:
        route_summary = route.to_metadata().to_dict()
        decision_summary = route_summary.get("decision_summary") or {}
        alternatives = decision_summary.get("alternatives") or []
        retrieval_summary: dict[str, object] = {}
        dialogue_summary: dict[str, object] = {}
        thought_framing_summary: dict[str, object] = {}
        if validation_context is not None:
            retrieval_summary = {
                "evidence_quality_score": validation_context.evidence_quality_score,
                "retrieval_lead_summary": validation_context.retrieval_lead_summary,
                "missing_context_note": validation_context.missing_context_note,
                "contradiction_flags": list(validation_context.contradiction_flags),
                "deep_validation_used": bool(
                    reasoning_frame_assembly is not None
                    and str(
                        (reasoning_frame_assembly.interaction_profile or {}).get("reasoning_depth") or ""
                    ).strip()
                    == "deep"
                ),
            }
        if front_half.dialogue_management is not None:
            dialogue_summary = front_half.dialogue_management.to_dict()
        if front_half.thought_framing is not None:
            thought_framing_summary = front_half.thought_framing.to_dict()
        reasoning_summary: dict[str, object] = {}
        if reasoning_frame_assembly is not None:
            reasoning_summary = self._reasoning_summary_from_assembly(reasoning_frame_assembly)
        execution_summary = execution_stage.to_dict() if execution_stage is not None else {}
        response_summary = response_packaging.to_dict() if response_packaging is not None else {}
        follow_up_linkage = {
            "resolution_changed": front_half.resolution_changed,
            "resolution_strategy": front_half.resolution_strategy,
            "resolved_prompt": front_half.resolved_prompt,
        }
        clarification_event = None
        if clarification_decision is not None:
            clarification_event = clarification_decision.to_dict()
        return PersistenceObservation(
            session_id=session_id,
            prompt=prompt,
            route_summary=route_summary,
            alternative_routes=list(alternatives),
            clarification_event=clarification_event,
            retrieval_summary=retrieval_summary,
            reasoning_summary={
                **reasoning_summary,
                "dialogue_management": dialogue_summary,
                "thought_framing": thought_framing_summary,
            },
            execution_summary=execution_summary,
            response_summary=response_summary,
            follow_up_linkage=follow_up_linkage,
        )

    @staticmethod
    def _optional_str(value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _secondary_intents(signals: dict[str, object]) -> list[str]:
        intents: list[tuple[str, int]] = [
            ("planning", int(signals.get("planning_score") or 0)),
            ("research", int(signals.get("research_score") or 0)),
            ("action", int(signals.get("action_score") or 0)),
            ("answer", int(signals.get("answer_score") or 0)),
        ]
        dominant = str(signals.get("dominant_intent") or "unknown")
        return [label for label, score in intents if score > 0 and label != dominant]

    @staticmethod
    def _ambiguity_flags(route: DomainRoute, route_analysis) -> list[str]:
        flags: list[str] = []
        decision = route_analysis.decision_summary.to_dict()
        if decision.get("ambiguous"):
            flags.append("route_ambiguous")
        if route.source == "active_thread_bias":
            flags.append("active_thread_bias")
        ambiguity = route.to_metadata().ambiguity or {}
        if ambiguity.get("top_alternative"):
            flags.append("close_alternative_present")
        return flags

    @staticmethod
    def _caution_notes(route: DomainRoute) -> list[str]:
        metadata = route.to_metadata().to_dict()
        cautions: list[str] = []
        caution = metadata.get("caution")
        if isinstance(caution, str) and caution.strip():
            cautions.append(caution)
        ambiguity = metadata.get("ambiguity") or {}
        reason = ambiguity.get("reason")
        if isinstance(reason, str) and reason.strip():
            cautions.append(reason)
        return cautions

    @staticmethod
    def _is_weak_route(route: DomainRoute) -> bool:
        comparison = route.comparison or {}
        normalized_score = comparison.get("normalized_score")
        if normalized_score is None:
            return route.to_metadata().to_dict().get("strength") == "low"
        return float(normalized_score) < 1.45

    @staticmethod
    def _top_alternative(route: DomainRoute) -> dict[str, object] | None:
        ambiguity = route.to_metadata().ambiguity or {}
        top_alternative = ambiguity.get("top_alternative")
        return dict(top_alternative) if isinstance(top_alternative, dict) else None

    @staticmethod
    def _clarification_notes(interaction_summary: dict[str, object]) -> list[str]:
        notes: list[str] = []
        recent_mix = str(interaction_summary.get("recent_clarification_mix") or "").strip()
        if recent_mix:
            notes.append(f"recent_clarification_mix={recent_mix}")
        posture_mix = str(interaction_summary.get("recent_posture_mix") or "").strip()
        if posture_mix:
            notes.append(f"recent_posture_mix={posture_mix}")
        return notes

    def _build_archive_context(self, *, session_id: str) -> dict[str, object]:
        summary = self.archive_service.summary(session_id=session_id)
        return {
            "record_count": summary.get("record_count", 0),
            "status_counts": summary.get("status_counts", {}),
            "tool_counts": summary.get("tool_counts", {}),
        }

    def _build_interaction_context(
        self,
        prompt: str,
        *,
        route: DomainRoute,
        active_thread: dict[str, object] | None,
        session_id: str,
        interaction_summary: dict[str, object],
    ) -> dict[str, object]:
        interaction_context = self.interaction_history_service.retrieve_context(
            prompt,
            session_id=session_id,
        )
        interaction_context = self._adapt_retrieval_context(
            interaction_context,
            summary=interaction_summary,
            match_key="top_interaction_matches",
        )
        interaction_context = self._filter_interaction_context_for_route(
            context=interaction_context,
            route=route,
            prompt=prompt,
        )
        active_prompt = str((active_thread or {}).get("prompt") or "").strip()
        if (
            interaction_context.get("interaction_record_count", 0) > 0
            or not active_prompt
            or active_prompt == prompt
        ):
            return interaction_context

        fallback_context = self.interaction_history_service.retrieve_context(
            active_prompt,
            session_id=session_id,
        )
        fallback_context = self._adapt_retrieval_context(
            fallback_context,
            summary=interaction_summary,
            match_key="top_interaction_matches",
        )
        fallback_context = self._filter_interaction_context_for_route(
            context=fallback_context,
            route=route,
            prompt=prompt,
        )
        if fallback_context.get("interaction_record_count", 0) <= 0:
            active_topic = str((active_thread or {}).get("normalized_topic") or "").strip()
            if not active_topic or active_topic == prompt or active_topic == active_prompt:
                return interaction_context
            topic_context = self.interaction_history_service.retrieve_context(
                active_topic,
                session_id=session_id,
            )
            topic_context = self._adapt_retrieval_context(
                topic_context,
                summary=interaction_summary,
                match_key="top_interaction_matches",
            )
            topic_context = self._filter_interaction_context_for_route(
                context=topic_context,
                route=route,
                prompt=prompt,
            )
            if topic_context.get("interaction_record_count", 0) <= 0:
                return interaction_context
            return {
                **topic_context,
                "interaction_query": active_topic,
                "interaction_query_source": "active_topic",
            }

        return {
            **fallback_context,
            "interaction_query": active_prompt,
            "interaction_query_source": "active_thread",
        }

    def _filter_active_thread_for_route(
        self,
        *,
        route: DomainRoute,
        prompt: str,
        active_thread: dict[str, object] | None,
    ) -> dict[str, object] | None:
        if active_thread is None or route.mode not in {"planning", "research"}:
            return active_thread
        if str(active_thread.get("mode") or "").strip() != "conversation":
            return active_thread
        prompt_tokens = self._summary_tokens(prompt)
        thread_text = " ".join(
            str(active_thread.get(key) or "").strip()
            for key in ("prompt", "normalized_topic", "summary", "thread_summary", "objective")
        ).strip()
        if not thread_text:
            return None
        thread_tokens = self._summary_tokens(thread_text)
        return active_thread if (prompt_tokens & thread_tokens) else None

    def _filter_interaction_context_for_route(
        self,
        *,
        context: dict[str, object],
        route: DomainRoute,
        prompt: str,
    ) -> dict[str, object]:
        if route.mode not in {"planning", "research"}:
            return context
        matches = list(context.get("top_interaction_matches") or [])
        if not matches:
            return context
        filtered = [
            match
            for match in matches
            if self._interaction_match_relevant_for_route(match=match, route=route, prompt=prompt)
        ]
        if len(filtered) == len(matches):
            return context
        return {
            **context,
            "interaction_record_count": len(filtered),
            "top_interaction_matches": filtered,
        }

    def _interaction_match_relevant_for_route(
        self,
        *,
        match: dict[str, object],
        route: DomainRoute,
        prompt: str,
    ) -> bool:
        record = match.get("record")
        if not isinstance(record, dict):
            return True
        record_mode = str(record.get("mode") or "").strip()
        if record_mode == route.mode:
            return True
        if record_mode != "conversation":
            return True
        prompt_tokens = self._summary_tokens(prompt)
        match_text = " ".join(
            str(record.get(key) or "").strip()
            for key in ("summary", "prompt", "resolved_prompt")
        ).strip()
        entity_values = " ".join(str(value).strip() for value in (record.get("extracted_entities") or []) if str(value).strip())
        if entity_values:
            match_text = f"{match_text} {entity_values}".strip()
        if not match_text:
            return False
        return bool(prompt_tokens & self._summary_tokens(match_text))

    def _build_archive_validation_context(
        self,
        *,
        prompt: str,
        session_id: str,
        active_thread: dict[str, object] | None,
        interaction_summary: dict[str, object],
        interaction_profile: InteractionProfile,
    ) -> dict[str, object]:
        targeted = self.archive_service.retrieve_context(prompt, session_id=session_id)
        targeted = self._adapt_retrieval_context(
            targeted,
            summary=interaction_summary,
            match_key="top_matches",
        )
        if interaction_profile.reasoning_depth != "deep":
            return self._attach_archive_target_comparison(
                targeted,
                session_id=session_id,
            )

        additional_contexts: list[dict[str, object]] = []
        active_prompt = str((active_thread or {}).get("prompt") or "").strip()
        active_topic = str((active_thread or {}).get("normalized_topic") or "").strip()
        for query in (active_prompt, active_topic):
            if not query or query == prompt:
                continue
            extra = self.archive_service.retrieve_context(query, session_id=session_id)
            extra = self._adapt_retrieval_context(
                extra,
                summary=interaction_summary,
                match_key="top_matches",
            )
            additional_contexts.append(extra)
        if not additional_contexts:
            return self._attach_archive_target_comparison(
                targeted,
                session_id=session_id,
            )
        merged = self._merge_archive_contexts(targeted, additional_contexts)
        return self._attach_archive_target_comparison(
            merged,
            session_id=session_id,
        )

    def _adapt_retrieval_context(
        self,
        context: dict[str, object],
        *,
        summary: dict[str, object],
        match_key: str,
    ) -> dict[str, object]:
        matches = context.get(match_key)
        if not isinstance(matches, list) or len(matches) < 2:
            return context
        if not self._semantic_retrieval_dominates(summary):
            return context
        adapted_matches = sorted(
            matches,
            key=self._adaptive_retrieval_rank,
            reverse=True,
        )
        return {
            **context,
            match_key: adapted_matches,
        }

    @staticmethod
    def _merge_archive_contexts(
        primary: dict[str, object],
        extras: list[dict[str, object]],
    ) -> dict[str, object]:
        merged_matches: list[dict[str, object]] = []
        seen: set[tuple[str, str, str]] = set()

        def add_matches(matches: list[dict[str, object]]) -> None:
            for match in matches:
                if not isinstance(match, dict):
                    continue
                record = match.get("record")
                if not isinstance(record, dict):
                    continue
                key = (
                    str(record.get("tool_id") or ""),
                    str(record.get("capability") or ""),
                    str(record.get("summary") or record.get("prompt") or ""),
                )
                if key in seen:
                    continue
                seen.add(key)
                merged_matches.append(match)

        add_matches(list(primary.get("top_matches") or []))
        for extra in extras:
            add_matches(list(extra.get("top_matches") or []))

        return {
            **primary,
            "record_count": len(merged_matches),
            "top_matches": merged_matches[:5],
        }

    def _attach_archive_target_comparison(
        self,
        context: dict[str, object],
        *,
        session_id: str,
    ) -> dict[str, object]:
        top_matches = context.get("top_matches")
        if not isinstance(top_matches, list) or not top_matches:
            return context
        top_match = top_matches[0]
        if not isinstance(top_match, dict):
            return context
        record = top_match.get("record")
        if not isinstance(record, dict):
            return context
        capability = str(record.get("capability") or "").strip()
        target_label = str(record.get("target_label") or "").strip()
        tool_id = str(record.get("tool_id") or "").strip() or None
        if not capability or not target_label:
            return context
        comparison = self.archive_service.compare_runs_by_target(
            session_id=session_id,
            tool_id=tool_id,
            capability=capability,
        )
        target_groups = comparison.get("target_groups")
        if not isinstance(target_groups, list):
            return context
        matching_group = next(
            (
                group
                for group in target_groups
                if isinstance(group, dict) and str(group.get("target_label") or "").strip() == target_label
            ),
            None,
        )
        if not isinstance(matching_group, dict):
            return context
        if int(matching_group.get("run_count") or 0) < 2:
            return context
        return {
            **context,
            "archive_target_comparison": matching_group,
        }

    @staticmethod
    def _validation_targets(
        *,
        prompt: str,
        assistant_context: AssistantContext,
    ) -> list[ValidationTargetView]:
        targets: list[ValidationTargetView] = []
        active_thread = assistant_context.active_thread or {}
        if active_thread:
            summary = str(active_thread.get("summary") or active_thread.get("objective") or active_thread.get("prompt") or "").strip() or None
            targets.append(
                ValidationTargetView(
                    source="active_thread",
                    query=str(active_thread.get("prompt") or prompt).strip() or None,
                    record_count=1,
                    lead="continuity",
                    quality="medium",
                    summary=summary,
                    top_match=None,
                )
            )
        archive_top = assistant_context.top_matches[0] if assistant_context.top_matches else None
        targets.append(
            ValidationTargetView(
                source="archive",
                query=assistant_context.query,
                record_count=assistant_context.matched_record_count,
                lead=ReasoningPipeline._retrieval_lead_label({"top_matches": assistant_context.top_matches}),
                quality=ReasoningPipeline._quality_label(
                    assistant_context.matched_record_count,
                    bool(archive_top),
                ),
                summary=ReasoningPipeline._match_summary(archive_top),
                top_match=archive_top,
            )
        )
        interaction_top = assistant_context.top_interaction_matches[0] if assistant_context.top_interaction_matches else None
        targets.append(
            ValidationTargetView(
                source="interactions",
                query=assistant_context.interaction_query,
                record_count=assistant_context.interaction_record_count,
                lead=ReasoningPipeline._retrieval_lead_label({"top_interaction_matches": assistant_context.top_interaction_matches}),
                quality=ReasoningPipeline._quality_label(
                    assistant_context.interaction_record_count,
                    bool(interaction_top),
                ),
                summary=ReasoningPipeline._match_summary(interaction_top),
                top_match=interaction_top,
            )
        )
        return targets

    @staticmethod
    def _quality_label(record_count: int, has_top_match: bool) -> str:
        if record_count > 1 and has_top_match:
            return "strong"
        if record_count == 1 and has_top_match:
            return "supported"
        if has_top_match:
            return "light"
        return "missing"

    @staticmethod
    def _match_summary(match: dict[str, object] | None) -> str | None:
        if not isinstance(match, dict):
            return None
        record = match.get("record")
        if not isinstance(record, dict):
            return None
        return str(
            record.get("summary")
            or record.get("prompt")
            or record.get("resolved_prompt")
            or record.get("capability")
            or ""
        ).strip() or None

    @staticmethod
    def _summary_tokens(value: str) -> set[str]:
        words = {
            token.strip(".,:;!?()[]{}").lower()
            for token in value.split()
            if len(token.strip(".,:;!?()[]{}")) >= 4
        }
        return {word for word in words if word}

    @staticmethod
    def _evidence_quality_score(
        assistant_context: AssistantContext,
        *,
        active_thread: dict[str, object] | None,
    ) -> float:
        score = 0.0
        if active_thread:
            score += 0.2
        if assistant_context.top_matches:
            score += 0.45
        if assistant_context.top_interaction_matches:
            score += 0.35
        return round(min(score, 1.0), 2)

    @staticmethod
    def _validation_retrieval_summary(assistant_context: AssistantContext) -> str | None:
        leads: list[str] = []
        archive_lead = ReasoningPipeline._retrieval_lead_label({"top_matches": assistant_context.top_matches})
        interaction_lead = ReasoningPipeline._retrieval_lead_label(
            {"top_interaction_matches": assistant_context.top_interaction_matches}
        )
        if archive_lead:
            leads.append(f"archive={archive_lead}")
        if interaction_lead:
            leads.append(f"interactions={interaction_lead}")
        return ", ".join(leads) if leads else None

    @staticmethod
    def _build_evidence_model(
        *,
        targets: list[ValidationTargetView],
        contradiction_flags: list[str],
        evidence_quality_score: float,
    ) -> LightweightEvidenceModel:
        evidence_sources: list[dict[str, object]] = []
        missing_sources: list[str] = []
        for target in targets:
            evidence_sources.append(
                {
                    "source": target.source,
                    "quality": target.quality,
                    "record_count": target.record_count,
                    "lead": target.lead,
                    "summary": target.summary,
                }
            )
            if target.quality == "missing" or target.record_count <= 0:
                missing_sources.append(target.source)
        if evidence_quality_score >= 0.75:
            evidence_strength = "strong"
        elif evidence_quality_score >= 0.45:
            evidence_strength = "supported"
        elif evidence_quality_score > 0:
            evidence_strength = "light"
        else:
            evidence_strength = "missing"
        return LightweightEvidenceModel(
            evidence_sources=evidence_sources,
            evidence_strength=evidence_strength,
            contradiction_flags=list(contradiction_flags),
            missing_sources=missing_sources,
        )

    @staticmethod
    def _deep_contradiction_flags(
        *,
        assistant_context: AssistantContext,
        interaction_profile: InteractionProfile,
    ) -> list[str]:
        if interaction_profile.reasoning_depth != "deep":
            return []
        flags: list[str] = []
        archive_top = assistant_context.top_matches[0] if assistant_context.top_matches else None
        interaction_top = (
            assistant_context.top_interaction_matches[0]
            if assistant_context.top_interaction_matches
            else None
        )
        archive_summary = ReasoningPipeline._match_summary(archive_top) or ""
        interaction_summary = ReasoningPipeline._match_summary(interaction_top) or ""
        if archive_summary and interaction_summary:
            archive_tokens = ReasoningPipeline._summary_tokens(archive_summary)
            interaction_tokens = ReasoningPipeline._summary_tokens(interaction_summary)
            if archive_tokens and interaction_tokens and not (archive_tokens & interaction_tokens):
                flags.append("cross_source_topic_mismatch")
        route_ambiguity = bool((assistant_context.route or {}).get("ambiguity"))
        if route_ambiguity and not assistant_context.top_matches and not assistant_context.top_interaction_matches:
            flags.append("ambiguous_route_with_thin_support")
        return flags

    @staticmethod
    def _semantic_retrieval_dominates(summary: dict[str, object]) -> bool:
        retrieval_lead_counts = summary.get("retrieval_lead_counts", {})
        if not isinstance(retrieval_lead_counts, dict):
            return False
        observation_count = int(summary.get("retrieval_observation_count", 0))
        if observation_count < 3:
            return False
        semantic_count = int(retrieval_lead_counts.get("semantic", 0))
        keyword_count = int(retrieval_lead_counts.get("keyword", 0))
        blended_count = int(retrieval_lead_counts.get("blended", 0))
        return semantic_count > max(keyword_count + blended_count, 0)

    @staticmethod
    def _adaptive_retrieval_rank(match: dict[str, object]) -> tuple[int, int, int]:
        score = int(match.get("score") or 0)
        breakdown = match.get("score_breakdown")
        if not isinstance(breakdown, dict):
            return (score, 0, 0)
        keyword_score = int(breakdown.get("keyword_score") or 0)
        semantic_score = int(breakdown.get("semantic_score") or 0)
        if keyword_score > 0 and semantic_score > 0:
            adjusted_score = score + 1
        elif keyword_score > 0:
            adjusted_score = score + 1
        elif semantic_score > 0:
            adjusted_score = score - 2
        else:
            adjusted_score = score
        return (adjusted_score, keyword_score, semantic_score)

    @staticmethod
    def _session_nlu_uncertainty_high(interaction_summary: dict[str, object]) -> bool:
        interaction_count = int(interaction_summary.get("interaction_count", 0))
        dominant_intent_counts = dict(interaction_summary.get("dominant_intent_counts") or {})
        unknown_intent_count = int(dominant_intent_counts.get("unknown", 0))
        return interaction_count >= 3 and (unknown_intent_count / interaction_count) >= 0.34

    @staticmethod
    def _session_retrieval_semantic_bias_high(interaction_summary: dict[str, object]) -> bool:
        return RouteAdaptationPolicy.session_retrieval_semantic_bias_high(interaction_summary)

    @staticmethod
    def _retrieval_lead_label(context: dict[str, object]) -> str | None:
        candidates: list[dict[str, object]] = []
        for key in ("top_interaction_matches", "top_matches"):
            value = context.get(key)
            if isinstance(value, list) and value:
                top = value[0]
                if isinstance(top, dict):
                    candidates.append(top)
        if not candidates:
            return None
        best = max(candidates, key=lambda item: int(item.get("score") or 0))
        breakdown = best.get("score_breakdown")
        if not isinstance(breakdown, dict):
            return None
        keyword_score = int(breakdown.get("keyword_score") or 0)
        semantic_score = int(breakdown.get("semantic_score") or 0)
        if keyword_score > 0 and semantic_score > 0:
            return "blended"
        if semantic_score > 0:
            return "semantic"
        if keyword_score > 0:
            return "keyword"
        return "none"

    @staticmethod
    def _frame_type(*, mode: str, kind: str) -> str:
        if mode == "planning":
            return "plan-next-step"
        if kind == "research.comparison":
            return "compare"
        if kind == "research.summary":
            return "retrieve-and-summarize"
        return "explain"

    @staticmethod
    def _problem_interpretation(
        *,
        prompt: str,
        mode: str,
        kind: str,
        route_quality: str,
    ) -> str:
        if mode == "planning":
            if kind == "planning.migration":
                return f"Interpret this as a migration-planning request: {prompt}"
            if kind == "planning.architecture":
                return f"Interpret this as an architecture-planning request: {prompt}"
            return f"Interpret this as a planning request: {prompt}"
        if kind == "research.comparison":
            return f"Interpret this as a comparison request: {prompt}"
        if route_quality == "weak":
            return f"Interpret this as a provisional research request pending route validation: {prompt}"
        return f"Interpret this as an explanatory research request: {prompt}"

    @staticmethod
    def _validation_plan(
        *,
        context: AssistantContext,
        validation_context: ValidationContextResult,
        route_quality: str,
        grounding_strength: str,
        interaction_profile: InteractionProfile,
    ) -> list[str]:
        plan: list[str] = []
        active_thread = context.active_thread or {}
        reasoning_depth = str(interaction_profile.reasoning_depth or "normal")
        if active_thread:
            plan.append("Check active thread continuity before broadening the response.")
        top_archive = context.top_matches[0] if context.top_matches else None
        if top_archive:
            plan.append("Validate against the closest archive run.")
        top_interaction = context.top_interaction_matches[0] if context.top_interaction_matches else None
        if top_interaction:
            plan.append("Validate against the closest prior interaction.")
        if validation_context.missing_context_note:
            plan.append(validation_context.missing_context_note)
        if validation_context.contradiction_flags:
            plan.append("Resolve contradictory validation signals before sounding decisive.")
        elif route_quality == "weak":
            plan.append("Validate the route choice itself before leaning too hard on the result.")
        elif grounding_strength == "low":
            plan.append("Treat the first answer pass as exploratory until another local signal confirms it.")
        if reasoning_depth == "deep":
            plan.append("Cross-check the strongest local anchor against the other available local sources before finalizing the response.")
            return plan[:5]
        return plan[:4]

    def _failure_modes(
        self,
        *,
        assistant_context: AssistantContext,
        evidence_quality_score: float,
        contradiction_flags: list[str],
    ) -> dict[str, bool]:
        route_quality = self.evidence_builder.route_quality_label(context=assistant_context)
        route_ambiguity = bool((assistant_context.route or {}).get("ambiguity"))
        return {
            "weak_route": route_quality == "weak",
            "weak_context": evidence_quality_score < 0.35,
            "weak_evidence": self.evidence_builder.grounding_strength(context=assistant_context) == "low",
            "high_ambiguity": route_ambiguity or bool(contradiction_flags),
        }

    @staticmethod
    def _profile_mismatch(
        *,
        interaction_profile: InteractionProfile,
        profile_advice: dict[str, object] | None,
    ) -> bool:
        if not isinstance(profile_advice, dict):
            return False
        advice_confidence = float(profile_advice.get("confidence") or 0.0)
        if advice_confidence < 0.7:
            return False
        advised_style = str(profile_advice.get("interaction_style") or interaction_profile.interaction_style)
        advised_depth = str(profile_advice.get("reasoning_depth") or interaction_profile.reasoning_depth)
        return (
            advised_style != interaction_profile.interaction_style
            or advised_depth != interaction_profile.reasoning_depth
        )

    @staticmethod
    def _stage_contract(
        *,
        stage_name: str,
        required_inputs: list[str],
        produced_outputs: list[str],
        confidence_signal: str | None,
        failure_state: str | None,
    ) -> StageContract:
        return StageContract(
            stage_name=stage_name,
            required_inputs=required_inputs,
            produced_outputs=produced_outputs,
            confidence_signal=confidence_signal,
            failure_state=failure_state,
        )

    @staticmethod
    def _failure_state_label(failure_modes: dict[str, bool]) -> str | None:
        for label in ("weak_route", "weak_context", "weak_evidence", "high_ambiguity"):
            if failure_modes.get(label):
                return label
        return None

    @staticmethod
    def _route_evidence_distinction(
        *,
        route_quality: str | None,
        grounding_strength: str | None,
    ) -> str:
        if route_quality == "weak" and grounding_strength in {"medium", "high"}:
            return "weak_route_but_supported_evidence"
        if grounding_strength == "low" and route_quality in {"supported", "strong"}:
            return "right_route_but_weak_evidence"
        if route_quality == "weak" and grounding_strength == "low":
            return "weak_route_and_weak_evidence"
        return "route_and_evidence_generally_aligned"

    @staticmethod
    def _packaged_answer(
        *,
        mode: str,
        response_payload: dict[str, object],
        interaction_profile: InteractionProfile,
    ) -> str | None:
        if InteractionStylePolicy.is_direct(interaction_profile):
            if mode == "planning":
                return str(response_payload.get("next_action") or "").strip() or str(response_payload.get("summary") or "").strip() or None
            if mode == "research":
                return str(response_payload.get("recommendation") or "").strip() or str(response_payload.get("summary") or "").strip() or None
        if mode == "planning":
            next_action = str(response_payload.get("next_action") or "").strip()
            return next_action or str(response_payload.get("summary") or "").strip() or None
        if mode == "research":
            recommendation = str(response_payload.get("recommendation") or "").strip()
            return recommendation or str(response_payload.get("summary") or "").strip() or None
        return str(response_payload.get("summary") or "").strip() or None

    @staticmethod
    def _package_type(*, mode: str, interaction_profile: InteractionProfile) -> str:
        return InteractionStylePolicy.package_type(mode=mode, profile=interaction_profile)

    @staticmethod
    def _profile_adjusted_body(
        response_body: list[str],
        *,
        interaction_profile: dict[str, object],
        grounded_interpretation: str | None,
        tension_resolution: dict[str, object] | None,
        suggested_next_step: str | None,
        validation_advice: list[str],
        route_evidence_distinction: str,
    ) -> list[str]:
        style = str(interaction_profile.get("interaction_style") or "conversational")
        depth = InteractionStylePolicy.reasoning_depth(interaction_profile)
        if InteractionStylePolicy.is_direct(interaction_profile):
            tension_rationale = (
                str((tension_resolution or {}).get("rationale") or "").strip()
                if isinstance(tension_resolution, dict)
                else ""
            )
            why_line = tension_rationale or grounded_interpretation or next((item for item in response_body if item), None)
            action_line = suggested_next_step or (validation_advice[0] if validation_advice else None)
            body = []
            if suggested_next_step:
                body.append(f"Answer: {suggested_next_step}")
            elif response_body:
                body.append(f"Answer: {response_body[0]}")
            if why_line:
                body.append(f"Why: {why_line}")
            if action_line:
                body.append(f"Action: {action_line}")
            if depth == "deep" and route_evidence_distinction != "route_and_evidence_generally_aligned":
                body.append(
                    "Constraint: keep the answer narrow until the route and evidence are better aligned."
                )
            return body[: (4 if depth == "normal" else 5)]
        if not AntiRoleplayPolicy.grounded_warmth_allowed(interaction_style=style):
            return list(response_body)
        return list(response_body)

    @staticmethod
    def _profile_adjusted_validation_advice(
        validation_advice: list[str],
        *,
        interaction_profile: dict[str, object],
        suggested_next_step: str | None,
    ) -> list[str]:
        if InteractionStylePolicy.is_deep(interaction_profile):
            return list(validation_advice[: InteractionStylePolicy.validation_advice_limit(interaction_profile)])
        if InteractionStylePolicy.is_direct(interaction_profile):
            if validation_advice:
                return [validation_advice[0]]
            if suggested_next_step:
                return [f"Do this next: {suggested_next_step}"]
            return []
        return list(validation_advice[: InteractionStylePolicy.validation_advice_limit(interaction_profile)])

    @staticmethod
    def _augment_validation_advice_for_tension(
        validation_advice: list[str],
        *,
        tension_resolution: dict[str, object] | None,
    ) -> list[str]:
        advice = list(validation_advice)
        if not isinstance(tension_resolution, dict) or not tension_resolution.get("tension_detected"):
            return advice
        resolution_path = str(tension_resolution.get("resolution_path") or "").strip()
        recommended_action = str(tension_resolution.get("recommended_action") or "").strip()
        alternate_hypotheses = tension_resolution.get("alternate_hypotheses") or []
        if resolution_path == "clarification":
            advice.insert(0, "Clarify which conflicting assumption should remain authoritative before revising the answer.")
            return advice
        if resolution_path == "alternate_hypothesis":
            if isinstance(alternate_hypotheses, list) and len(alternate_hypotheses) >= 2:
                labels = [
                    str(item.get("label") or "").strip()
                    for item in alternate_hypotheses[:2]
                    if str(item.get("label") or "").strip()
                ]
                if labels:
                    advice.insert(0, f"Compare hypotheses {' vs '.join(labels)} before treating the current anchor as settled.")
            elif recommended_action.startswith("compare_hypotheses"):
                advice.insert(0, "Compare the competing hypotheses before committing to one interpretation.")
            elif recommended_action == "gather_missing_evidence":
                advice.insert(0, "Gather one more confirming source before resolving the tension.")
        elif resolution_path == "hypothesis_revision":
            advice.insert(0, "Revise the working hypothesis to reflect the stronger competing evidence.")
        return advice

    @staticmethod
    def _follow_up_suggestions(
        *,
        mode: str,
        response_payload: dict[str, object],
        interaction_profile: InteractionProfile,
    ) -> list[str]:
        suggestions: list[str] = []
        if mode == "planning":
            next_action = str(response_payload.get("next_action") or "").strip()
            if next_action:
                suggestions.append(next_action)
        elif mode == "research":
            recommendation = str(response_payload.get("recommendation") or "").strip()
            if recommendation:
                suggestions.append(recommendation)
        elif mode == "tool":
            summary = str(response_payload.get("summary") or "").strip()
            if summary:
                suggestions.append(f"Inspect the tool result: {summary}")
        if InteractionStylePolicy.is_direct(interaction_profile):
            return suggestions[:1]
        return suggestions[:2]

    @staticmethod
    def _packaged_evidence_summary(
        *,
        response_payload: dict[str, object],
        interaction_profile: InteractionProfile,
    ) -> str | None:
        summary = str(response_payload.get("best_evidence") or response_payload.get("summary") or "").strip() or None
        if summary is None:
            return None
        if InteractionStylePolicy.is_direct(interaction_profile):
            return summary.split(". ")[0].rstrip(".") + "."
        return summary

    @staticmethod
    def _packaged_route_summary(
        *,
        route_metadata: dict[str, object],
        route_status: str | None,
        interaction_profile: InteractionProfile,
    ) -> dict[str, object]:
        summary: dict[str, object] = {
            "source": route_metadata.get("source"),
            "strength": route_metadata.get("strength"),
        }
        if route_status:
            summary["status"] = route_status
        if InteractionStylePolicy.is_direct(interaction_profile):
            return dict(summary)
        return summary

    @staticmethod
    def _prepend_evidence_context(
        response_body: list[str],
        *,
        evidence_model: LightweightEvidenceModel,
        interaction_profile: dict[str, object],
    ) -> list[str]:
        body = list(response_body)
        if evidence_model.contradiction_flags:
            prefix = "Evidence conflict:" if InteractionStylePolicy.is_direct(interaction_profile) else "Evidence conflict detected:"
            body.insert(0, f"{prefix} {'; '.join(evidence_model.contradiction_flags)}.")
            return body
        if evidence_model.missing_sources:
            prefix = "Missing context:" if InteractionStylePolicy.is_direct(interaction_profile) else "Missing supporting context:"
            body.insert(0, f"{prefix} {', '.join(evidence_model.missing_sources)}.")
        return body

    def _prepend_ledger_context(
        self,
        response_body: list[str],
        *,
        reasoning_frame_assembly: ReasoningFrameAssembly,
        interaction_profile: dict[str, object],
    ) -> list[str]:
        interaction_style = InteractionStylePolicy.interaction_style(interaction_profile)
        anchor_line = self._anchor_context_line(reasoning_frame_assembly)
        support_line = self._support_context_line(reasoning_frame_assembly)
        if InteractionStylePolicy.is_direct(interaction_profile):
            return [anchor_line, *response_body] if anchor_line else list(response_body)
        lines = [line for line in (anchor_line, support_line) if line]
        return [*lines, *response_body] if lines else list(response_body)

    @staticmethod
    def _prepend_evidence_strength_context(
        response_body: list[str],
        *,
        evidence_model: LightweightEvidenceModel,
        interaction_profile: dict[str, object],
    ) -> list[str]:
        body = list(response_body)
        evidence_strength = str(evidence_model.evidence_strength or "").strip()
        if evidence_strength == "strong":
            leading_sources = [
                str(item.get("source") or "").strip()
                for item in evidence_model.evidence_sources[:2]
                if str(item.get("source") or "").strip()
            ]
            if leading_sources:
                prefix = (
                    f"Support status: {display_status_label('strongly_supported')} across"
                )
                body.insert(0, f"{prefix} {', '.join(leading_sources)}.")
        elif evidence_strength == "light":
            prefix = f"Support status: {display_status_label('moderately_supported')}."
            body.insert(0, f"{prefix} Keep this as a narrower first pass.")
        elif evidence_strength == "missing" and not evidence_model.missing_sources:
            prefix = f"Support status: {display_status_label('insufficiently_grounded')}."
            body.insert(0, prefix)
        return body

    @staticmethod
    def _prepend_tension_resolution_context(
        response_body: list[str],
        *,
        tension_resolution: dict[str, object] | None,
        interaction_profile: dict[str, object],
    ) -> list[str]:
        if not isinstance(tension_resolution, dict) or not tension_resolution.get("tension_detected"):
            return list(response_body)
        rationale = str(tension_resolution.get("rationale") or "").strip()
        tension_status = str(tension_resolution.get("status") or "under_tension").strip() or "under_tension"
        if InteractionStylePolicy.is_direct(interaction_profile):
            status_line = f"Tension status: {display_status_label(tension_status)}."
        else:
            if tension_status == "under_tension":
                status_line = ResponseVariationLayer.select_from_pool(
                    (
                        "This answer is still carrying real tension.",
                        "This answer is still under real tension.",
                        "There is still real tension in this answer.",
                    ),
                    seed_parts=[tension_status, rationale, "tension_status"],
                )
            elif tension_status == "unresolved":
                status_line = ResponseVariationLayer.select_from_pool(
                    (
                        "This answer is still unresolved.",
                        "This answer is still not resolved cleanly.",
                        "This answer is still open and unresolved.",
                    ),
                    seed_parts=[tension_status, rationale, "tension_status"],
                )
            elif tension_status == "revised":
                status_line = ResponseVariationLayer.select_from_pool(
                    (
                        "The original line no longer holds cleanly, so the answer has to be revised.",
                        "The original line no longer holds cleanly, so the answer needs revision.",
                        "The original line does not hold cleanly anymore, so the answer has to be revised.",
                    ),
                    seed_parts=[tension_status, rationale, "tension_status"],
                )
            else:
                status_line = f"Tension status: {display_status_label(tension_status)}."
        alternate_hypotheses = tension_resolution.get("alternate_hypotheses") or []
        if isinstance(alternate_hypotheses, list) and alternate_hypotheses:
            hypotheses_line = ReasoningPipeline._alternate_hypotheses_line(
                alternate_hypotheses=alternate_hypotheses,
                interaction_profile=interaction_profile,
            )
            weight_line = ReasoningPipeline._hypothesis_weight_line(
                alternate_hypotheses=alternate_hypotheses,
                interaction_profile=interaction_profile,
            )
        else:
            hypotheses_line = None
            weight_line = None
        if not rationale:
            body = list(response_body)
            if hypotheses_line:
                body.insert(0, hypotheses_line)
            if weight_line:
                body.insert(0, weight_line)
            body.insert(0, status_line)
            return body
        body = list(response_body)
        if hypotheses_line:
            body.insert(0, hypotheses_line)
        if weight_line:
            body.insert(0, weight_line)
        body.insert(0, rationale)
        body.insert(0, status_line)
        return body

    @staticmethod
    def _alternate_hypotheses_line(
        *,
        alternate_hypotheses: list[dict[str, object]],
        interaction_profile: dict[str, object],
    ) -> str | None:
        labels = []
        for item in alternate_hypotheses[:3]:
            label = str(item.get("label") or "").strip()
            status = str(item.get("status") or "").strip()
            summary = str(item.get("summary") or "").strip()
            if not label:
                continue
            if InteractionStylePolicy.is_direct(interaction_profile):
                labels.append(f"{label}({status or 'candidate'})")
            elif summary:
                labels.append(f"{label}: {summary}")
            else:
                labels.append(label)
        if not labels:
            return None
        if InteractionStylePolicy.is_direct(interaction_profile):
            return f"Hypotheses: {', '.join(labels)}."
        opener = ResponseVariationLayer.select_from_pool(
            (
                "The competing explanations are still live:",
                "The competing explanations are still in play:",
                "The competing explanations are still active:",
            ),
            seed_parts=[*labels, "alternate_hypotheses"],
        )
        return f"{opener} {' | '.join(labels)}."

    @staticmethod
    def _hypothesis_weight_line(
        *,
        alternate_hypotheses: list[dict[str, object]],
        interaction_profile: dict[str, object],
    ) -> str | None:
        if InteractionStylePolicy.is_direct(interaction_profile):
            return None
        leading = next(
            (item for item in alternate_hypotheses if str(item.get("status") or "").strip() == "leading"),
            None,
        )
        if not isinstance(leading, dict):
            return None
        leading_label = str(leading.get("label") or "").strip()
        if not leading_label:
            return None
        current_anchor = next(
            (item for item in alternate_hypotheses if str(item.get("status") or "").strip() == "current_anchor"),
            None,
        )
        if isinstance(current_anchor, dict) and str(current_anchor.get("label") or "").strip() != leading_label:
            opener = ResponseVariationLayer.select_from_pool(
                (
                    f"Right now, hypothesis {leading_label} is carrying more weight than the current anchor,",
                    f"At the moment, hypothesis {leading_label} is carrying more weight than the current anchor,",
                    f"Right now, hypothesis {leading_label} is out-weighting the current anchor,",
                ),
                seed_parts=[leading_label, "hypothesis_weight", "anchor_shift"],
            )
            closer = ResponseVariationLayer.select_from_pool(
                (
                    "but the competing line is still strong enough that the answer should stay provisional.",
                    "but the competing line is still strong enough that the answer should stay open.",
                    "but the competing line is still strong enough that the answer should remain provisional.",
                ),
                seed_parts=[leading_label, "hypothesis_weight", "anchor_shift_closer"],
            )
            return f"{opener} {closer}"
        opener = ResponseVariationLayer.select_from_pool(
            (
                f"Right now, hypothesis {leading_label} is carrying the most weight,",
                f"At the moment, hypothesis {leading_label} is carrying the most weight,",
                f"Right now, hypothesis {leading_label} is leading on weight,",
            ),
            seed_parts=[leading_label, "hypothesis_weight", "leading"],
        )
        closer = ResponseVariationLayer.select_from_pool(
            (
                "but the competing line is still strong enough that the answer should stay open.",
                "but the competing line is still strong enough that the answer should stay provisional.",
                "but the competing line is still strong enough that the answer should remain open.",
            ),
            seed_parts=[leading_label, "hypothesis_weight", "leading_closer"],
        )
        return f"{opener} {closer}"

    @staticmethod
    def _find_ledger_unit(
        evidence_ledger,
        evidence_id: str | None,
    ):
        if not evidence_id:
            return None
        for unit in evidence_ledger or []:
            if unit.evidence_id == evidence_id:
                return unit
        return None

    def _anchor_context_line(
        self,
        reasoning_frame_assembly: ReasoningFrameAssembly,
    ) -> str | None:
        anchor_unit = self._find_ledger_unit(
            reasoning_frame_assembly.evidence_ledger,
            reasoning_frame_assembly.anchor_evidence_id,
        )
        if anchor_unit is None:
            return None
        return (
            "Anchor evidence: "
            f"{anchor_unit.summary} "
            f"({display_status_label(anchor_unit.strength)} via {anchor_unit.source})."
        )

    def _support_context_line(
        self,
        reasoning_frame_assembly: ReasoningFrameAssembly,
    ) -> str | None:
        supporting_unit = self._find_ledger_unit(
            reasoning_frame_assembly.evidence_ledger,
            reasoning_frame_assembly.supporting_evidence_id,
        )
        if supporting_unit is None:
            return None
        return (
            "Supporting evidence: "
            f"{supporting_unit.summary} "
            f"({display_status_label(supporting_unit.strength)} via {supporting_unit.source})."
        )

    def _build_reasoning_status_snapshot(
        self,
        *,
        route_strength: str,
        route_quality: str,
        grounding_strength: str,
        local_context_assessment: str | None,
        validation_context: ValidationContextResult,
        route_ambiguity: bool,
        tension_resolution: TensionResolutionResult,
    ) -> ReasoningStatusSnapshot:
        return self.reasoning_status_policy.build_snapshot(
            route_strength=route_strength,
            route_quality=route_quality,
            grounding_strength=grounding_strength,
            local_context_assessment=local_context_assessment,
            route_ambiguity=route_ambiguity,
            contradiction_flags=validation_context.contradiction_flags,
            evidence_strength=validation_context.evidence_model.evidence_strength,
            failure_modes=validation_context.failure_modes,
            tension_resolution=tension_resolution,
        )

    def _reasoning_summary_from_assembly(
        self,
        reasoning_frame_assembly: ReasoningFrameAssembly,
    ) -> dict[str, object]:
        status_snapshot = (
            dict(reasoning_frame_assembly.status_snapshot)
            if isinstance(reasoning_frame_assembly.status_snapshot, dict)
            else {}
        )
        return {
            "frame_type": reasoning_frame_assembly.frame_type,
            "confidence_posture": reasoning_frame_assembly.confidence_posture,
            "uncertainty_posture": reasoning_frame_assembly.uncertainty_posture,
            "route_quality": reasoning_frame_assembly.route_quality,
            "route_status": reasoning_frame_assembly.route_status,
            "grounding_strength": reasoning_frame_assembly.grounding_strength,
            "support_status": reasoning_frame_assembly.support_status,
            "tension_status": reasoning_frame_assembly.tension_status,
            "failure_modes": dict(reasoning_frame_assembly.failure_modes),
            "status_snapshot": status_snapshot,
            "anchor_evidence_summary": self._ledger_unit_summary(
                reasoning_frame_assembly.evidence_ledger,
                reasoning_frame_assembly.anchor_evidence_id,
            ),
            "supporting_evidence_summary": self._ledger_unit_summary(
                reasoning_frame_assembly.evidence_ledger,
                reasoning_frame_assembly.supporting_evidence_id,
            ),
            "tension_evidence_summaries": [
                summary
                for summary in (
                    self._ledger_unit_summary(
                        reasoning_frame_assembly.evidence_ledger,
                        evidence_id,
                    )
                    for evidence_id in reasoning_frame_assembly.tension_evidence_ids
                )
                if summary
            ],
            "evidence_strength": reasoning_frame_assembly.evidence_model.evidence_strength,
            "evidence_sources": list(reasoning_frame_assembly.evidence_model.evidence_sources),
            "missing_sources": list(reasoning_frame_assembly.evidence_model.missing_sources),
            "contradiction_flags": list(reasoning_frame_assembly.evidence_model.contradiction_flags),
            "aged_evidence_count": sum(
                1
                for unit in reasoning_frame_assembly.evidence_ledger
                if (unit.age_bucket or "") in {"aging", "stale", "old"}
            ),
            "reaffirmed_evidence_count": sum(
                1 for unit in reasoning_frame_assembly.evidence_ledger if unit.reaffirmed
            ),
            "tension_resolution": dict(reasoning_frame_assembly.tension_resolution or {}),
            "anchor_evidence_id": reasoning_frame_assembly.anchor_evidence_id,
            "supporting_evidence_id": reasoning_frame_assembly.supporting_evidence_id,
            "tension_evidence_ids": list(reasoning_frame_assembly.tension_evidence_ids),
        }

    def _ledger_unit_summary(
        self,
        evidence_ledger,
        evidence_id: str | None,
    ) -> str | None:
        unit = self._find_ledger_unit(evidence_ledger, evidence_id)
        if unit is None:
            return None
        return f"{unit.source}: {unit.summary}"

