from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import re
from types import SimpleNamespace

from lumen.app.models import InteractionProfile
from lumen.app.command_parser import CommandParser
from lumen.reasoning.planner import Planner
from lumen.reasoning.belief_discussion_policy import BeliefDiscussionPolicy
from lumen.reasoning.continuation_confidence_policy import ContinuationConfidencePolicy
from lumen.reasoning.conversation_surface_support import ConversationSurfaceSupport
from lumen.reasoning.domain_behavior_policy import DomainBehaviorPolicy, NextStepEngine
from lumen.reasoning.explanatory_support_policy import ExplanatorySupportPolicy
from lumen.reasoning.intent_domain_policy import IntentDomainPolicy
from lumen.reasoning.reasoning_pipeline import ReasoningPipeline
from lumen.reasoning.research_engine import ResearchEngine
from lumen.reasoning.response_context_support import ResponseContextSupport
from lumen.reasoning.mode_response_shaper import ModeResponseShaper
from lumen.reasoning.response_packaging_support import ResponsePackagingSupport
from lumen.reasoning.response_strategy_layer import ResponseStrategyLayer
from lumen.reasoning.response_tone_engine import ResponseToneEngine
from lumen.reasoning.response_variation import ResponseVariationLayer
from lumen.reasoning.route_support_signals import RouteSupportSignalBuilder
from lumen.reasoning.social_interaction_policy import SocialInteractionPolicy
from lumen.reasoning.stance_consistency_layer import StanceConsistencyLayer
from lumen.reasoning.conversational_reply_realizer import ConversationalReplyRealizer
from lumen.reasoning.conversation_beat_support import ConversationBeatSupport
from lumen.reasoning.interaction_style_policy import InteractionStylePolicy
from lumen.reasoning.memory_retrieval_layer import MemoryRetrievalLayer, MemoryRetrievalResult
from lumen.reasoning.memory_context_classifier import MemoryContextClassifier, MemoryContextDecision
from lumen.reasoning.memory_response_support import MemoryResponseSupport
from lumen.reasoning.response_flow_realizer import ResponseFlowRealizer
from lumen.reasoning.supervised_decision_support import SupervisedDecisionSupport
from lumen.reasoning.trainability_trace import TrainabilityTrace
from lumen.reasoning.thread_explanation_support import ThreadExplanationSupport
from lumen.reasoning.work_thread_continuity_support import WorkThreadContinuitySupport
from lumen.reasoning.math_surface_support import MathSurfaceSupport
from lumen.reasoning.knowledge_surface_support import KnowledgeSurfaceSupport
from lumen.reasoning.design_surface_support import DesignSurfaceSupport
from lumen.reasoning.dataset_workflow_support import DatasetWorkflowSupport
from lumen.reasoning.self_overview_surface_support import SelfOverviewSurfaceSupport
from lumen.reasoning.writing_workflow_support import WritingWorkflowSupport
from lumen.reasoning.explanation_mode_support import ExplanationModeSupport
from lumen.reasoning.explanation_response_builder import ExplanationResponseBuilder
from lumen.reasoning.tool_threshold_gate import ToolThresholdDecision, ToolThresholdGate
from lumen.reasoning.response_models import (
    ClarificationResponse,
    ConversationResponse,
    ResearchResponse,
    SafetyResponse,
    ToolAssistantResponse,
    ToolExecutionDetails,
)
from lumen.reasoning.reasoning_state import ReasoningStateFrame
from lumen.reasoning.pipeline_models import (
    ClarificationGateDecision,
    NLUExtraction,
    ResponsePackagingContext,
    RouteAuthorityDecision,
)
from lumen.knowledge.knowledge_service import KnowledgeService
from lumen.nlu.prompt_nlu import PromptNLU
from lumen.nlu.prompt_surface_builder import PromptSurfaceBuilder
from lumen.nlu.follow_up_inventory import looks_like_general_follow_up
from lumen.routing.domain_router import DomainRouter
from lumen.routing.domain_router import DomainRoute
from lumen.routing.intent_router import IntentRouter
from lumen.routing.anchor_registry import detect_follow_up_anchor
from lumen.routing.prompt_resolution import PromptResolver
from lumen.routing.route_adaptation import RouteAdaptationPolicy
from lumen.services.archive_service import ArchiveService
from lumen.services.academic_support_service import AcademicSupportService
from lumen.services.capability_contract_service import CapabilityContractService
from lumen.services.inference_service import HostedInferenceDecision, InferenceService
from lumen.services.interaction_conversation_support import InteractionConversationSupport
from lumen.services.interaction_flow_support import InteractionFlowSupport
from lumen.services.interaction_history_service import InteractionHistoryService
from lumen.services.interaction_orchestration_models import InteractionTurnContext
from lumen.services.reasoning_state_service import ReasoningStateService
from lumen.services.safety_service import SafetyService
from lumen.services.session_context_service import SessionContextService
from lumen.services.tool_execution_service import ToolExecutionService
from lumen.tools.registry_types import ToolResult


class InteractionService:
    """Routes free-form prompts into tool, planning, or research responses."""

    def __init__(
        self,
        *,
        domain_router: DomainRouter,
        command_parser: CommandParser,
        intent_router: IntentRouter,
        planner: Planner,
        research_engine: ResearchEngine,
        tool_execution_service: ToolExecutionService,
        archive_service: ArchiveService,
        interaction_history_service: InteractionHistoryService,
        session_context_service: SessionContextService,
        prompt_resolver: PromptResolver,
        safety_service: SafetyService | None = None,
        inference_service: InferenceService | None = None,
        knowledge_service: KnowledgeService | None = None,
        memory_retrieval_layer: MemoryRetrievalLayer | None = None,
        supervised_decision_support: SupervisedDecisionSupport | None = None,
    ):
        self.domain_router = domain_router
        self.command_parser = command_parser
        self.intent_router = intent_router
        self.planner = planner
        self.research_engine = research_engine
        self.tool_execution_service = tool_execution_service
        self.archive_service = archive_service
        self.interaction_history_service = interaction_history_service
        self.session_context_service = session_context_service
        self.prompt_resolver = prompt_resolver
        self.safety_service = safety_service
        self.inference_service = inference_service
        self.knowledge_service = knowledge_service
        self.memory_retrieval_layer = memory_retrieval_layer
        self.prompt_nlu = PromptNLU()
        self._tool_map_cache: dict[str, list[str]] | None = None
        self.route_adaptation_policy = RouteAdaptationPolicy()
        self.reasoning_pipeline = ReasoningPipeline(
            prompt_resolver=prompt_resolver,
            domain_router=domain_router,
            archive_service=archive_service,
            interaction_history_service=interaction_history_service,
            session_context_service=session_context_service,
        )
        self.intent_domain_policy = IntentDomainPolicy()
        self.domain_behavior_policy = DomainBehaviorPolicy()
        self.next_step_engine = NextStepEngine()
        self.response_strategy_layer = ResponseStrategyLayer()
        self.reasoning_state_service = ReasoningStateService()
        self.tool_threshold_gate = ToolThresholdGate()
        self.memory_context_classifier = MemoryContextClassifier()
        self.supervised_decision_support = supervised_decision_support or SupervisedDecisionSupport()

    def ask(
        self,
        *,
        prompt: str,
        input_path: Path | None = None,
        params: dict[str, int | float | str] | None = None,
        session_id: str = "default",
        run_root: Path | None = None,
        client_surface: str = "main",
        project_id: str | None = None,
        project_name: str | None = None,
    ) -> dict[str, object]:
        self._project_id_hint = project_id
        self._project_name_hint = project_name
        # Route selection authority lives in routing/pipeline layers. This service orchestrates
        # the chosen route and may shape visible response surfaces, but advisory metadata such as
        # NLU profile hints and prompt-surface variants must never silently replace route authority.
        turn = self._stage_intake(
            prompt=prompt,
            input_path=input_path,
            params=params,
            session_id=session_id,
            run_root=run_root,
            client_surface=client_surface,
        )
        wake_interaction = turn.wake_interaction
        effective_prompt = turn.effective_prompt
        active_thread = turn.active_thread
        interaction_profile = turn.interaction_profile
        recent_interactions_cache: list[dict[str, object]] | None = None

        def _recent_interactions() -> list[dict[str, object]]:
            nonlocal recent_interactions_cache
            if recent_interactions_cache is None:
                recent_interactions_cache = self._recent_interactions_for_turn(turn)
            return recent_interactions_cache

        offer_backed_follow_up = self._offer_backed_continuation(
            prompt=effective_prompt,
            active_thread=active_thread,
            interaction_profile=interaction_profile,
        )
        if offer_backed_follow_up is not None:
            if offer_backed_follow_up.get("action") == "decline":
                response = ConversationResponse(
                    mode="conversation",
                    kind="conversation.offer_decline",
                    summary=str(offer_backed_follow_up["reply"]),
                    reply=str(offer_backed_follow_up["reply"]),
                ).to_dict()
                self._attach_profile_context(
                    response=response,
                    interaction_profile=interaction_profile,
                    profile_advice=None,
                    client_surface=client_surface,
                )
                self._attach_wake_metadata(response=response, wake_interaction=wake_interaction)
                self._attach_wake_resolution(
                    response=response,
                    original_prompt=prompt,
                    effective_prompt=effective_prompt,
                    wake_interaction=wake_interaction,
                )
                self._attach_response_behavior_posture(
                    response=response,
                    route_mode="conversation",
                    low_confidence_recovery=None,
                    srd_diagnostic=None,
                    state_control=None,
                )
                self._record_interaction(
                    prompt=prompt,
                    session_id=session_id,
                    response=response,
                    update_active_thread=True,
                )
                return response
            effective_prompt = str(offer_backed_follow_up.get("target_prompt") or effective_prompt)
        safety_decision = (
            self.safety_service.evaluate_prompt(effective_prompt)
            if self.safety_service is not None
            else None
        )
        if safety_decision is not None and safety_decision.action != "allow":
            response = self._build_safety_response(
                prompt=effective_prompt,
                safety_decision=safety_decision,
            )
            self._attach_profile_context(
                response=response,
                interaction_profile=interaction_profile,
                profile_advice=None,
                client_surface=client_surface,
            )
            self._attach_wake_metadata(response=response, wake_interaction=wake_interaction)
            self._attach_wake_resolution(
                response=response,
                original_prompt=prompt,
                effective_prompt=effective_prompt,
                wake_interaction=wake_interaction,
            )
            self._attach_response_behavior_posture(
                response=response,
                route_mode="safety",
                low_confidence_recovery=None,
                srd_diagnostic=None,
                state_control=None,
            )
            self._record_interaction(
                prompt=prompt,
                session_id=session_id,
                response=response,
                update_active_thread=False,
            )
            return response
        not_promised_contract = CapabilityContractService.match_not_promised_surface(
            prompt=effective_prompt,
            input_path=input_path,
        )
        if not_promised_contract is not None:
            response = self._build_not_promised_capability_response(
                prompt=effective_prompt,
                interaction_profile=interaction_profile,
                contract=not_promised_contract,
            )
            self._attach_profile_context(
                response=response,
                interaction_profile=interaction_profile,
                profile_advice=None,
                client_surface=client_surface,
            )
            self._attach_wake_metadata(response=response, wake_interaction=wake_interaction)
            self._attach_wake_resolution(
                response=response,
                original_prompt=prompt,
                effective_prompt=effective_prompt,
                wake_interaction=wake_interaction,
            )
            self._attach_response_behavior_posture(
                response=response,
                route_mode="research",
                low_confidence_recovery=None,
                srd_diagnostic=None,
                state_control=None,
            )
            self._record_interaction(
                prompt=prompt,
                session_id=session_id,
                response=response,
                update_active_thread=False,
            )
            return response
        if CapabilityContractService.is_dataset_guidance_request(prompt=effective_prompt):
            response = DatasetWorkflowSupport.build_response(prompt=effective_prompt)
            self._attach_profile_context(
                response=response,
                interaction_profile=interaction_profile,
                profile_advice=None,
                client_surface=client_surface,
            )
            self._attach_wake_metadata(response=response, wake_interaction=wake_interaction)
            self._attach_wake_resolution(
                response=response,
                original_prompt=prompt,
                effective_prompt=effective_prompt,
                wake_interaction=wake_interaction,
            )
            self._attach_response_behavior_posture(
                response=response,
                route_mode="research",
                low_confidence_recovery=None,
                srd_diagnostic=None,
                state_control=None,
            )
            self._record_interaction(
                prompt=prompt,
                session_id=session_id,
                response=response,
                update_active_thread=False,
            )
            return response
        recent_interactions = _recent_interactions()
        diagnostic_follow_up_response = self._diagnostic_follow_up_response(
            prompt=effective_prompt,
            interaction_profile=interaction_profile,
            recent_interactions=recent_interactions,
            active_thread=active_thread,
        )
        if diagnostic_follow_up_response is not None:
            response = diagnostic_follow_up_response
            self._attach_conversation_access_metadata(
                response=response,
                match_type=str(response.get("kind") or "conversation.failure_follow_up"),
                context_used="diagnostic_failure",
                final_source="diagnostic_conversation_follow_up",
            )
            self._attach_profile_context(
                response=response,
                interaction_profile=interaction_profile,
                profile_advice=None,
                client_surface=client_surface,
            )
            self._attach_wake_metadata(response=response, wake_interaction=wake_interaction)
            self._attach_wake_resolution(
                response=response,
                original_prompt=prompt,
                effective_prompt=effective_prompt,
                wake_interaction=wake_interaction,
            )
            self._attach_response_behavior_posture(
                response=response,
                route_mode="conversation",
                low_confidence_recovery=None,
                srd_diagnostic=None,
                state_control=None,
            )
            self._attach_general_assistant_turn_metadata(
                response=response,
                session_id=session_id,
                prompt=effective_prompt,
                route=SimpleNamespace(mode="conversation", kind=str(response.get("kind") or "conversation.failure_follow_up")),
                interaction_profile=interaction_profile,
                recent_interactions=recent_interactions,
                active_thread=active_thread,
                memory_retrieval=None,
            )
            self._record_interaction(
                prompt=prompt,
                session_id=session_id,
                response=response,
                update_active_thread=True,
            )
            return response
        knowledge_self_assessment = self._knowledge_self_assessment_response(
            prompt=effective_prompt,
            interaction_profile=interaction_profile,
        )
        if knowledge_self_assessment is not None:
            response = knowledge_self_assessment
            self._attach_profile_context(
                response=response,
                interaction_profile=interaction_profile,
                profile_advice=None,
                client_surface=client_surface,
            )
            self._attach_wake_metadata(response=response, wake_interaction=wake_interaction)
            self._attach_wake_resolution(
                response=response,
                original_prompt=prompt,
                effective_prompt=effective_prompt,
                wake_interaction=wake_interaction,
            )
            self._attach_response_behavior_posture(
                response=response,
                route_mode="conversation",
                low_confidence_recovery=None,
                srd_diagnostic=None,
                state_control=None,
            )
            self._attach_general_assistant_turn_metadata(
                response=response,
                session_id=session_id,
                prompt=effective_prompt,
                route=SimpleNamespace(mode="conversation", kind=str(response.get("kind") or "conversation.knowledge_self_assessment")),
                interaction_profile=interaction_profile,
                recent_interactions=recent_interactions,
                active_thread=active_thread,
                memory_retrieval=None,
            )
            self._record_interaction(
                prompt=prompt,
                session_id=session_id,
                response=response,
                update_active_thread=True,
            )
            return response
        ordered_request_response = self._ordered_request_response(
            prompt=effective_prompt,
            interaction_profile=interaction_profile,
        )
        if ordered_request_response is not None:
            response = ordered_request_response
            self._attach_profile_context(
                response=response,
                interaction_profile=interaction_profile,
                profile_advice=None,
                client_surface=client_surface,
            )
            self._attach_wake_metadata(response=response, wake_interaction=wake_interaction)
            self._attach_wake_resolution(
                response=response,
                original_prompt=prompt,
                effective_prompt=effective_prompt,
                wake_interaction=wake_interaction,
            )
            self._attach_response_behavior_posture(
                response=response,
                route_mode="conversation",
                low_confidence_recovery=None,
                srd_diagnostic=None,
                state_control=None,
            )
            self._attach_general_assistant_turn_metadata(
                response=response,
                session_id=session_id,
                prompt=effective_prompt,
                route=SimpleNamespace(mode="conversation", kind=str(response.get("kind") or "conversation.ordered_request_ack")),
                interaction_profile=interaction_profile,
                recent_interactions=recent_interactions,
                active_thread=active_thread,
                memory_retrieval=None,
            )
            self._record_interaction(
                prompt=prompt,
                session_id=session_id,
                response=response,
                update_active_thread=True,
            )
            return response
        context_backed_follow_up_prompt = self._context_backed_continuation_prompt(
            prompt=effective_prompt,
            recent_interactions=recent_interactions,
            active_thread=active_thread,
        )
        if context_backed_follow_up_prompt:
            effective_prompt = context_backed_follow_up_prompt
        memory_save_continuation = self._memory_save_clarification_continuation(
            prompt=effective_prompt,
            recent_interactions=recent_interactions,
        )
        if memory_save_continuation is not None:
            save_result = self.interaction_history_service.save_memory_from_record(
                source_record=dict(memory_save_continuation["source_record"]),
                target=str(memory_save_continuation["target"]),
                client_surface=client_surface,
            )
            reply = self._memory_save_confirmation_text(
                target=str(memory_save_continuation["target"]),
                interaction_profile=interaction_profile,
            )
            response = ConversationResponse(
                mode="conversation",
                kind="conversation.memory_save_confirmation",
                summary=reply,
                reply=reply,
            ).to_dict()
            response["memory_save_result"] = save_result
            self._attach_profile_context(
                response=response,
                interaction_profile=interaction_profile,
                profile_advice=None,
                client_surface=client_surface,
            )
            self._attach_wake_metadata(response=response, wake_interaction=wake_interaction)
            self._attach_wake_resolution(
                response=response,
                original_prompt=prompt,
                effective_prompt=effective_prompt,
                wake_interaction=wake_interaction,
            )
            self._attach_response_behavior_posture(
                response=response,
                route_mode="conversation",
                low_confidence_recovery=None,
                srd_diagnostic=None,
                state_control=None,
            )
            self._record_interaction(
                prompt=prompt,
                session_id=session_id,
                response=response,
                update_active_thread=True,
            )
            return response
        memory_save_request = self._memory_save_request(
            prompt=effective_prompt,
            recent_interactions=recent_interactions,
        )
        if memory_save_request is not None:
            if str(memory_save_request.get("action") or "") == "save":
                save_result = self.interaction_history_service.save_memory_from_record(
                    source_record=dict(memory_save_request["source_record"]),
                    target=str(memory_save_request["target"]),
                    client_surface=client_surface,
                )
                target_label = (
                    "the research"
                    if str(memory_save_request["target"]) == "research"
                    else (
                        "that as your recent thought"
                        if str(memory_save_request["target"]) == "personal"
                        else "my last answer from this thread"
                    )
                )
                reply = self._memory_save_confirmation_text(
                    target=str(memory_save_request["target"]),
                    interaction_profile=interaction_profile,
                )
                response = ConversationResponse(
                    mode="conversation",
                    kind="conversation.memory_save_confirmation",
                    summary=reply or f"Got it. I'll save {target_label}.",
                    reply=reply or f"Got it. I'll save {target_label}.",
                ).to_dict()
                response["memory_save_result"] = save_result
            else:
                response = ClarificationResponse(
                    mode="clarification",
                    kind="clarification.request",
                    summary="Clarification requested for memory save target.",
                    clarification_question=str(memory_save_request["question"]),
                    options=list(memory_save_request.get("options") or []),
                    clarification_context={
                        "clarification_type": "memory_save",
                        "candidate_targets": list(memory_save_request.get("candidates") or []),
                    },
                ).to_dict()
            self._attach_profile_context(
                response=response,
                interaction_profile=interaction_profile,
                profile_advice=None,
                client_surface=client_surface,
            )
            self._attach_wake_metadata(response=response, wake_interaction=wake_interaction)
            self._attach_wake_resolution(
                response=response,
                original_prompt=prompt,
                effective_prompt=effective_prompt,
                wake_interaction=wake_interaction,
            )
            self._attach_response_behavior_posture(
                response=response,
                route_mode=str(response.get("mode") or "conversation"),
                low_confidence_recovery=None,
                srd_diagnostic=None,
                state_control=None,
            )
            self._record_interaction(
                prompt=prompt,
                session_id=session_id,
                response=response,
                update_active_thread=False,
            )
            return response
        clarification_continuation = self._clarification_planning_continuation(
            prompt=effective_prompt,
            recent_interactions=recent_interactions,
            active_thread=active_thread,
        )
        if clarification_continuation is not None and clarification_continuation.get("action") == "decline":
            reasoning_state = self.reasoning_state_service.apply_clarification_continuation(
                state=turn.reasoning_state or ReasoningStateFrame(),
                continuation=clarification_continuation,
            )
            reply = self._clarification_decline_reply(
                interaction_profile=interaction_profile,
                reasoning_state=reasoning_state,
            )
            response = ConversationResponse(
                mode="conversation",
                kind="conversation.clarification_decline",
                summary=reply,
                reply=reply,
            ).to_dict()
            self._attach_reasoning_state(response=response, reasoning_state=reasoning_state)
            self._attach_profile_context(
                response=response,
                interaction_profile=interaction_profile,
                profile_advice=None,
                client_surface=client_surface,
            )
            self._attach_wake_metadata(response=response, wake_interaction=wake_interaction)
            self._attach_wake_resolution(
                response=response,
                original_prompt=prompt,
                effective_prompt=effective_prompt,
                wake_interaction=wake_interaction,
            )
            self._attach_response_behavior_posture(
                response=response,
                route_mode="conversation",
                low_confidence_recovery=None,
                srd_diagnostic=None,
                state_control=None,
            )
            self._record_interaction(
                prompt=prompt,
                session_id=session_id,
                response=response,
                update_active_thread=True,
            )
            return response
        if clarification_continuation is not None and clarification_continuation.get("action") == "pivot":
            effective_prompt = str(clarification_continuation.get("working_prompt") or effective_prompt).strip() or effective_prompt
            clarification_continuation = None
            turn = replace(turn, effective_prompt=effective_prompt)
        normalized_effective_prompt = PromptSurfaceBuilder.build(effective_prompt).route_ready_text
        if clarification_continuation is None and InteractionFlowSupport.is_direction_pivot(normalized_effective_prompt):
            pivot_prompt = InteractionFlowSupport._normalize_pivot_prompt(effective_prompt)
            if pivot_prompt:
                effective_prompt = pivot_prompt
                turn = replace(turn, effective_prompt=effective_prompt)
        route_choice_clarification = self._route_choice_clarification_response(
            prompt=effective_prompt,
            interaction_profile=interaction_profile,
            reasoning_state=turn.reasoning_state,
        )
        if route_choice_clarification is not None and clarification_continuation is None:
            response = route_choice_clarification
            self._attach_profile_context(
                response=response,
                interaction_profile=interaction_profile,
                profile_advice=None,
                client_surface=client_surface,
            )
            self._attach_wake_metadata(response=response, wake_interaction=wake_interaction)
            self._attach_wake_resolution(
                response=response,
                original_prompt=prompt,
                effective_prompt=effective_prompt,
                wake_interaction=wake_interaction,
            )
            self._attach_response_behavior_posture(
                response=response,
                route_mode="clarification",
                low_confidence_recovery=None,
                srd_diagnostic=None,
                state_control=None,
            )
            self._record_interaction(
                prompt=prompt,
                session_id=session_id,
                response=response,
                update_active_thread=True,
            )
            return response
        explanation_mode_response = ExplanationModeSupport.build_follow_up_response(
            prompt=effective_prompt,
            interaction_profile=interaction_profile,
            recent_interactions=recent_interactions,
            active_thread=active_thread,
            knowledge_service=self.knowledge_service,
        )
        if explanation_mode_response is not None and clarification_continuation is None:
            response = explanation_mode_response
            self._attach_profile_context(
                response=response,
                interaction_profile=interaction_profile,
                profile_advice=None,
                client_surface=client_surface,
            )
            self._attach_wake_metadata(response=response, wake_interaction=wake_interaction)
            self._attach_wake_resolution(
                response=response,
                original_prompt=prompt,
                effective_prompt=effective_prompt,
                wake_interaction=wake_interaction,
            )
            self._attach_response_behavior_posture(
                response=response,
                route_mode=str(response.get("mode") or "research"),
                low_confidence_recovery=None,
                srd_diagnostic=None,
                state_control=None,
            )
            self._record_interaction(
                prompt=prompt,
                session_id=session_id,
                response=response,
                update_active_thread=False,
            )
            return response
        arithmetic_response = self._quick_arithmetic_response(
            prompt=effective_prompt,
            interaction_profile=interaction_profile,
            recent_interactions=recent_interactions,
        )
        if arithmetic_response is not None and clarification_continuation is None:
            response = arithmetic_response
            self._attach_profile_context(
                response=response,
                interaction_profile=interaction_profile,
                profile_advice=None,
                client_surface=client_surface,
            )
            self._attach_wake_metadata(response=response, wake_interaction=wake_interaction)
            self._attach_wake_resolution(
                response=response,
                original_prompt=prompt,
                effective_prompt=effective_prompt,
                wake_interaction=wake_interaction,
            )
            self._attach_response_behavior_posture(
                response=response,
                route_mode="conversation",
                low_confidence_recovery=None,
                srd_diagnostic=None,
                state_control=None,
            )
            self._record_interaction(
                prompt=prompt,
                session_id=session_id,
                response=response,
                update_active_thread=False,
            )
            return response
        broad_concept_response = self._broad_concept_response(
            prompt=effective_prompt,
            interaction_profile=interaction_profile,
            recent_interactions=recent_interactions,
        )
        if broad_concept_response is not None and clarification_continuation is None:
            response = broad_concept_response
            self._attach_profile_context(
                response=response,
                interaction_profile=interaction_profile,
                profile_advice=None,
                client_surface=client_surface,
            )
            self._attach_wake_metadata(response=response, wake_interaction=wake_interaction)
            self._attach_wake_resolution(
                response=response,
                original_prompt=prompt,
                effective_prompt=effective_prompt,
                wake_interaction=wake_interaction,
            )
            self._attach_response_behavior_posture(
                response=response,
                route_mode="research",
                low_confidence_recovery=None,
                srd_diagnostic=None,
                state_control=None,
            )
            self._record_interaction(
                prompt=prompt,
                session_id=session_id,
                response=response,
                update_active_thread=True,
            )
            return response
        self_overview_response = self._self_overview_response(
            prompt=effective_prompt,
            interaction_profile=interaction_profile,
            recent_interactions=recent_interactions,
        )
        if self_overview_response is not None and clarification_continuation is None:
            response = self_overview_response
            self._attach_conversation_access_metadata(
                response=response,
                match_type=str(response.get("kind") or "conversation.self_overview"),
                context_used=self._conversation_context_used(
                    recent_interactions=recent_interactions,
                    active_thread=active_thread,
                ),
                final_source="self_overview_conversation",
            )
            self._attach_profile_context(
                response=response,
                interaction_profile=interaction_profile,
                profile_advice=None,
                client_surface=client_surface,
            )
            self._attach_wake_metadata(response=response, wake_interaction=wake_interaction)
            self._attach_wake_resolution(
                response=response,
                original_prompt=prompt,
                effective_prompt=effective_prompt,
                wake_interaction=wake_interaction,
            )
            self._attach_response_behavior_posture(
                response=response,
                route_mode="conversation",
                low_confidence_recovery=None,
                srd_diagnostic=None,
                state_control=None,
            )
            self._attach_general_assistant_turn_metadata(
                response=response,
                session_id=session_id,
                prompt=effective_prompt,
                route=SimpleNamespace(mode="conversation", kind=str(response.get("kind") or "conversation.self_overview")),
                interaction_profile=interaction_profile,
                recent_interactions=recent_interactions,
                active_thread=active_thread,
                memory_retrieval=None,
            )
            self._record_interaction(
                prompt=prompt,
                session_id=session_id,
                response=response,
                update_active_thread=False,
            )
            return response
        pipeline_prompt = (
            str(clarification_continuation.get("working_prompt") or effective_prompt)
            if clarification_continuation is not None
            else effective_prompt
        )
        pipeline_prompt = self._planning_prompt_hint(
            pipeline_prompt,
            recent_interactions=recent_interactions,
            active_thread=active_thread,
        )
        social_turn = SocialInteractionPolicy.classify(effective_prompt)
        work_thread_response = WorkThreadContinuitySupport.build_response(
            prompt=effective_prompt,
            interaction_profile=interaction_profile,
            active_thread=active_thread,
            recent_interactions=recent_interactions,
        ) if isinstance(active_thread, dict) and clarification_continuation is None else None
        if work_thread_response is not None and social_turn is None:
            response = work_thread_response
            self._attach_profile_context(
                response=response,
                interaction_profile=interaction_profile,
                profile_advice=None,
                client_surface=client_surface,
            )
            self._attach_wake_metadata(response=response, wake_interaction=wake_interaction)
            self._attach_wake_resolution(
                response=response,
                original_prompt=prompt,
                effective_prompt=effective_prompt,
                wake_interaction=wake_interaction,
            )
            self._attach_response_behavior_posture(
                response=response,
                route_mode="conversation",
                low_confidence_recovery=None,
                srd_diagnostic=None,
                state_control=None,
            )
            self._attach_general_assistant_turn_metadata(
                response=response,
                session_id=session_id,
                prompt=effective_prompt,
                route=SimpleNamespace(mode="conversation", kind=str(response.get("kind") or "conversation.work_thread")),
                interaction_profile=interaction_profile,
                recent_interactions=recent_interactions,
                active_thread=active_thread,
                memory_retrieval=None,
            )
            self._record_interaction(
                prompt=prompt,
                session_id=session_id,
                response=response,
                update_active_thread=False,
            )
            return response
        if (
            social_turn is None
            and clarification_continuation is None
            and ConversationSurfaceSupport.is_return_to_recent_prompt(effective_prompt)
        ):
            response = ConversationSurfaceSupport.build_return_to_recent_response(
                interaction_profile=interaction_profile,
                recent_interactions=recent_interactions,
            )
            if response is not None:
                self._attach_profile_context(
                    response=response,
                    interaction_profile=interaction_profile,
                    profile_advice=None,
                    client_surface=client_surface,
                )
                self._attach_wake_metadata(response=response, wake_interaction=wake_interaction)
                self._attach_wake_resolution(
                    response=response,
                    original_prompt=prompt,
                    effective_prompt=effective_prompt,
                    wake_interaction=wake_interaction,
                )
                self._attach_response_behavior_posture(
                    response=response,
                    route_mode="conversation",
                    low_confidence_recovery=None,
                    srd_diagnostic=None,
                    state_control=None,
                )
                self._attach_general_assistant_turn_metadata(
                    response=response,
                    session_id=session_id,
                    prompt=effective_prompt,
                    route=SimpleNamespace(mode="conversation", kind=str(response.get("kind") or "conversation.return_to_recent")),
                    interaction_profile=interaction_profile,
                    recent_interactions=recent_interactions,
                    active_thread=active_thread,
                    memory_retrieval=None,
                )
                self._realize_conversational_reply_surface(
                    response=response,
                    interaction_profile=interaction_profile,
                    recent_interactions=recent_interactions,
                    active_thread=active_thread,
                )
                self._record_interaction(
                    prompt=prompt,
                    session_id=session_id,
                    response=response,
                    update_active_thread=False,
                )
                return response
        if social_turn is not None and clarification_continuation is None:
            response = ConversationSurfaceSupport.build_lightweight_social_response(
                prompt=effective_prompt,
                interaction_profile=interaction_profile,
                recent_interactions=recent_interactions,
                active_thread=active_thread,
            )
            self._attach_conversation_access_metadata(
                response=response,
                match_type=str(social_turn),
                context_used=self._conversation_context_used(
                    recent_interactions=recent_interactions,
                    active_thread=active_thread,
                ),
                final_source="lightweight_social_conversation",
            )
            self._attach_profile_context(
                response=response,
                interaction_profile=interaction_profile,
                profile_advice=None,
                client_surface=client_surface,
            )
            self._attach_wake_metadata(response=response, wake_interaction=wake_interaction)
            self._attach_wake_resolution(
                response=response,
                original_prompt=prompt,
                effective_prompt=effective_prompt,
                wake_interaction=wake_interaction,
            )
            self._attach_response_behavior_posture(
                response=response,
                route_mode="conversation",
                low_confidence_recovery=None,
                srd_diagnostic=None,
                state_control=None,
            )
            self._attach_general_assistant_turn_metadata(
                response=response,
                session_id=session_id,
                prompt=effective_prompt,
                route=SimpleNamespace(mode="conversation", kind=str(response.get("kind") or social_turn)),
                interaction_profile=interaction_profile,
                recent_interactions=recent_interactions,
                active_thread=active_thread,
                memory_retrieval=None,
            )
            self._realize_conversational_reply_surface(
                response=response,
                interaction_profile=interaction_profile,
                recent_interactions=recent_interactions,
                active_thread=active_thread,
            )
            self._record_interaction(
                prompt=prompt,
                session_id=session_id,
                response=response,
                update_active_thread=False,
            )
            return response
        if ConversationSurfaceSupport.is_thought_follow_up_prompt(
            prompt=effective_prompt,
            recent_interactions=recent_interactions,
        ):
            response = ConversationSurfaceSupport.build_thought_follow_up_response(
                interaction_profile=interaction_profile,
                recent_interactions=recent_interactions,
            )
            self._attach_profile_context(
                response=response,
                interaction_profile=interaction_profile,
                profile_advice=None,
                client_surface=client_surface,
            )
            self._attach_wake_metadata(response=response, wake_interaction=wake_interaction)
            self._attach_wake_resolution(
                response=response,
                original_prompt=prompt,
                effective_prompt=effective_prompt,
                wake_interaction=wake_interaction,
            )
            self._attach_response_behavior_posture(
                response=response,
                route_mode="conversation",
                low_confidence_recovery=None,
                srd_diagnostic=None,
                state_control=None,
            )
            self._attach_general_assistant_turn_metadata(
                response=response,
                session_id=session_id,
                prompt=effective_prompt,
                route=SimpleNamespace(mode="conversation", kind=str(response.get("kind") or "conversation.thought_follow_up")),
                interaction_profile=interaction_profile,
                recent_interactions=recent_interactions,
                active_thread=active_thread,
                memory_retrieval=None,
            )
            self._realize_conversational_reply_surface(
                response=response,
                interaction_profile=interaction_profile,
                recent_interactions=recent_interactions,
                active_thread=active_thread,
            )
            self._record_interaction(
                prompt=prompt,
                session_id=session_id,
                response=response,
                update_active_thread=False,
            )
            return response
        if ThreadExplanationSupport.is_thread_explanation_prompt(effective_prompt):
            response = ThreadExplanationSupport.build_response(
                prompt=effective_prompt,
                interaction_profile=interaction_profile,
                active_thread=active_thread,
                recent_interactions=recent_interactions,
            )
            self._attach_profile_context(
                response=response,
                interaction_profile=interaction_profile,
                profile_advice=None,
                client_surface=client_surface,
            )
            self._attach_wake_metadata(response=response, wake_interaction=wake_interaction)
            self._attach_wake_resolution(
                response=response,
                original_prompt=prompt,
                effective_prompt=effective_prompt,
                wake_interaction=wake_interaction,
            )
            self._attach_response_behavior_posture(
                response=response,
                route_mode="conversation",
                low_confidence_recovery=None,
                srd_diagnostic=None,
                state_control=None,
            )
            self._attach_general_assistant_turn_metadata(
                response=response,
                session_id=session_id,
                prompt=effective_prompt,
                route=SimpleNamespace(mode="conversation", kind=str(response.get("kind") or "conversation.thread_explanation")),
                interaction_profile=interaction_profile,
                recent_interactions=recent_interactions,
                active_thread=active_thread,
                memory_retrieval=None,
            )
            self._realize_conversational_reply_surface(
                response=response,
                interaction_profile=interaction_profile,
                recent_interactions=recent_interactions,
                active_thread=active_thread,
            )
            self._record_interaction(
                prompt=prompt,
                session_id=session_id,
                response=response,
                update_active_thread=False,
            )
            return response
        turn = self._stage_route_preparation(
            turn=InteractionTurnContext(
                original_prompt=prompt,
                effective_prompt=effective_prompt,
                session_id=session_id,
                client_surface=client_surface,
                input_path=input_path,
                params=params,
                run_root=run_root,
                wake_interaction=wake_interaction,
                active_thread=active_thread,
                interaction_profile=interaction_profile,
                recent_interactions=tuple(recent_interactions),
                clarification_continuation=clarification_continuation,
                pipeline_prompt=pipeline_prompt,
                reasoning_state=turn.reasoning_state,
            ),
            prompt=prompt,
            recent_interactions=recent_interactions,
            clarification_continuation=clarification_continuation,
        )
        interaction_summary = dict(turn.interaction_summary or {})
        pipeline_result = turn.pipeline_result
        pipeline_trace = turn.pipeline_trace
        resolved_prompt = str(turn.pipeline_prompt or pipeline_prompt)
        route = turn.route
        clarification_decision = turn.clarification_decision
        memory_retrieval = turn.memory_retrieval
        route_support_signals = turn.route_support_signals
        reasoning_state = turn.reasoning_state or ReasoningStateFrame()
        if clarification_decision.should_clarify:
            response = self._build_clarification_response(
                prompt=prompt,
                resolved_prompt=resolved_prompt,
                route=route,
                interaction_profile=interaction_profile,
                reasoning_state=reasoning_state,
                interaction_summary=interaction_summary,
                clarification_trigger=clarification_decision.trigger,
                clarification_question_style=clarification_decision.question_style,
                resolution_changed=pipeline_result.resolution_changed,
                resolution_strategy=pipeline_result.resolution_strategy,
                resolution_reason=pipeline_result.resolution_reason,
            )
            self._attach_reasoning_state(response=response, reasoning_state=reasoning_state)
            self._attach_profile_context(
                response=response,
                interaction_profile=interaction_profile,
                profile_advice=pipeline_result.nlu.profile_advice,
                client_surface=client_surface,
            )
            self._attach_wake_metadata(response=response, wake_interaction=wake_interaction)
            self._attach_wake_resolution(
                response=response,
                original_prompt=prompt,
                effective_prompt=effective_prompt,
                wake_interaction=wake_interaction,
            )
            self._attach_dialogue_context(
                response=response,
                vibe_catcher=pipeline_result.vibe_catcher,
                dialogue_management=pipeline_result.dialogue_management,
                conversation_awareness=pipeline_result.conversation_awareness,
                low_confidence_recovery=pipeline_result.low_confidence_recovery,
                srd_diagnostic=pipeline_result.srd_diagnostic,
                empathy_model=pipeline_result.empathy_model,
                human_language_layer=pipeline_result.human_language_layer,
                state_control=pipeline_result.state_control,
                thought_framing=pipeline_result.thought_framing,
            )
            self._attach_persistence_observation(
                response=response,
                pipeline_trace=pipeline_trace,
                session_id=session_id,
                prompt=prompt,
                front_half=pipeline_result,
                route=route,
                clarification_decision=clarification_decision,
                validation_context=None,
                reasoning_frame_assembly=None,
            )
            self._attach_response_behavior_posture(
                response=response,
                route_mode=route.mode,
                low_confidence_recovery=pipeline_result.low_confidence_recovery,
                srd_diagnostic=pipeline_result.srd_diagnostic,
                state_control=pipeline_result.state_control,
            )
            self._attach_memory_retrieval(response=response, memory_retrieval=memory_retrieval)
            response["route_support_signals"] = route_support_signals.to_dict()
            if turn.route_authority is not None:
                response["route_authority"] = turn.route_authority.to_dict()
            self._apply_memory_recall_surface(response=response, memory_retrieval=memory_retrieval)
            response["pipeline_trace"] = pipeline_trace.to_dict()
            self._record_interaction(
                prompt=prompt,
                session_id=session_id,
                response=response,
                update_active_thread=False,
            )
            return response

        if route.mode == "conversation":
            response = self._build_conversation_response(
                prompt=prompt,
                route=route,
                interaction_profile=interaction_profile,
            )
            self._attach_reasoning_state(response=response, reasoning_state=reasoning_state)
            tone_blend = self._response_tone_blend(prompt=resolved_prompt, route=route)
            response["response_tone_blend"] = tone_blend
            self._attach_profile_context(
                response=response,
                interaction_profile=interaction_profile,
                profile_advice=pipeline_result.nlu.profile_advice,
                client_surface=client_surface,
            )
            self._attach_wake_metadata(response=response, wake_interaction=wake_interaction)
            self._attach_wake_resolution(
                response=response,
                original_prompt=prompt,
                effective_prompt=effective_prompt,
                wake_interaction=wake_interaction,
            )
            self._attach_dialogue_context(
                response=response,
                vibe_catcher=pipeline_result.vibe_catcher,
                dialogue_management=pipeline_result.dialogue_management,
                conversation_awareness=pipeline_result.conversation_awareness,
                low_confidence_recovery=pipeline_result.low_confidence_recovery,
                srd_diagnostic=pipeline_result.srd_diagnostic,
                empathy_model=pipeline_result.empathy_model,
                human_language_layer=pipeline_result.human_language_layer,
                state_control=pipeline_result.state_control,
                thought_framing=pipeline_result.thought_framing,
            )
            self._attach_conversational_turn(
                prompt=resolved_prompt,
                response=response,
                interaction_profile=interaction_profile,
                dialogue_management=pipeline_result.dialogue_management,
                conversation_awareness=pipeline_result.conversation_awareness,
                state_control=pipeline_result.state_control,
                human_language_layer=pipeline_result.human_language_layer,
                thought_framing=pipeline_result.thought_framing,
                recent_interactions=recent_interactions,
                tone_profile=str(tone_blend.get("tone_profile") or "default"),
                allow_internal_scaffold=self._allow_internal_scaffold(resolved_prompt),
            )
            if not response.get("user_facing_answer"):
                self._ensure_conversational_turn_fallback(response=response, interaction_profile=interaction_profile)
            self._attach_memory_retrieval(response=response, memory_retrieval=memory_retrieval)
            response["route_support_signals"] = route_support_signals.to_dict()
            self._apply_memory_recall_surface(response=response, memory_retrieval=memory_retrieval)
            self._attach_general_assistant_turn_metadata(
                response=response,
                session_id=session_id,
                prompt=resolved_prompt,
                route=route,
                interaction_profile=interaction_profile,
                recent_interactions=recent_interactions,
                active_thread=active_thread,
                memory_retrieval=memory_retrieval,
            )
            self._realize_conversational_reply_surface(
                response=response,
                interaction_profile=interaction_profile,
                recent_interactions=recent_interactions,
                active_thread=active_thread,
            )
            self._apply_safety_response_constraint(response=response, safety_decision=safety_decision)
            response["internal_scaffold_visible"] = self._allow_internal_scaffold(resolved_prompt)
            self._enforce_final_surface_lane(response=response, selected_mode=route.mode)
            self._attach_execution_and_packaging(
                response=response,
                pipeline_trace=pipeline_trace,
                mode="conversation",
                kind=route.kind,
                route=route,
                interaction_profile=interaction_profile,
            )
            self._attach_persistence_observation(
                response=response,
                pipeline_trace=pipeline_trace,
                session_id=session_id,
                prompt=prompt,
                front_half=pipeline_result,
                route=route,
                clarification_decision=clarification_decision,
                validation_context=None,
                reasoning_frame_assembly=None,
            )
            self._attach_response_behavior_posture(
                response=response,
                route_mode=route.mode,
                low_confidence_recovery=pipeline_result.low_confidence_recovery,
                srd_diagnostic=pipeline_result.srd_diagnostic,
                state_control=pipeline_result.state_control,
            )
            response["pipeline_trace"] = pipeline_trace.to_dict()
            self._attach_route_metadata(response=response, route=route, route_status=None)
            self._record_interaction(
                prompt=prompt,
                session_id=session_id,
                response=response,
                update_active_thread=False,
            )
            return response

        academic_prompt = resolved_prompt
        academic_workflow = None
        for candidate_prompt in (effective_prompt, prompt, resolved_prompt):
            academic_workflow = AcademicSupportService.classify(
                prompt=candidate_prompt,
                input_path=input_path,
            )
            if academic_workflow is not None:
                academic_prompt = candidate_prompt
                break
        if academic_workflow is not None:
            response = self._handle_academic_workflow(
                prompt=prompt,
                resolved_prompt=academic_prompt,
                route=route,
                input_path=input_path,
                client_surface=client_surface,
                interaction_profile=interaction_profile,
                wake_interaction=wake_interaction,
                effective_prompt=effective_prompt,
                pipeline_result=pipeline_result,
                reasoning_state=reasoning_state,
            )
            if response is not None:
                self._record_interaction(
                    prompt=prompt,
                    session_id=session_id,
                    response=response,
                    update_active_thread=False,
                )
                return response

        writing_workflow = WritingWorkflowSupport.classify(resolved_prompt)
        if writing_workflow is not None:
            response = self._handle_writing_workflow(
                prompt=prompt,
                resolved_prompt=resolved_prompt,
                route=route,
                session_id=session_id,
                client_surface=client_surface,
                interaction_profile=interaction_profile,
                wake_interaction=wake_interaction,
                effective_prompt=effective_prompt,
                pipeline_result=pipeline_result,
                reasoning_state=reasoning_state,
            )
            if response is not None:
                self._record_interaction(
                    prompt=prompt,
                    session_id=session_id,
                    response=response,
                    update_active_thread=False,
                )
                return response

        validation_context = self.reasoning_pipeline.build_validation_context(
            prompt=resolved_prompt,
            session_id=session_id,
            route=route,
            interaction_summary=interaction_summary,
            interaction_profile=interaction_profile,
        )
        self.reasoning_pipeline.record_validation_context(pipeline_trace, validation_context)
        archive_context = validation_context.assistant_context
        hosted_inference_decision = None
        if self.inference_service is not None:
            hosted_inference_decision = self.inference_service.evaluate_hosted_research(
                route=route,
                validation_context=validation_context,
            )
        if hosted_inference_decision is not None and hosted_inference_decision.use_hosted_inference:
            try:
                inference_result = self.inference_service.infer_research_reply(
                    prompt=resolved_prompt,
                    session_id=session_id,
                    interaction_profile=interaction_profile,
                    validation_context=validation_context,
                )
            except Exception as exc:
                hosted_inference_decision = HostedInferenceDecision(
                    False,
                    f"Hosted inference failed; using the local response lane instead. ({exc})",
                )
            else:
                response = self._build_hosted_research_response(
                    prompt=resolved_prompt,
                    route=route,
                    inference_result=inference_result,
                    hosted_reason=hosted_inference_decision.reason,
                )
                self._attach_reasoning_state(response=response, reasoning_state=reasoning_state)
                tone_blend = self._response_tone_blend(prompt=resolved_prompt, route=route)
                response["response_tone_blend"] = tone_blend
                self._attach_profile_context(
                    response=response,
                    interaction_profile=interaction_profile,
                    profile_advice=pipeline_result.nlu.profile_advice,
                    client_surface=client_surface,
                )
                self._attach_dialogue_context(
                    response=response,
                    vibe_catcher=pipeline_result.vibe_catcher,
                    dialogue_management=pipeline_result.dialogue_management,
                    conversation_awareness=pipeline_result.conversation_awareness,
                    low_confidence_recovery=pipeline_result.low_confidence_recovery,
                    srd_diagnostic=pipeline_result.srd_diagnostic,
                    empathy_model=pipeline_result.empathy_model,
                    human_language_layer=pipeline_result.human_language_layer,
                    state_control=pipeline_result.state_control,
                    thought_framing=pipeline_result.thought_framing,
                )
                self._attach_conversational_turn(
                    prompt=resolved_prompt,
                    response=response,
                    interaction_profile=interaction_profile,
                    dialogue_management=pipeline_result.dialogue_management,
                    conversation_awareness=pipeline_result.conversation_awareness,
                    state_control=pipeline_result.state_control,
                    human_language_layer=pipeline_result.human_language_layer,
                    thought_framing=pipeline_result.thought_framing,
                    recent_interactions=recent_interactions,
                    tone_profile=str(tone_blend.get("tone_profile") or "default"),
                    allow_internal_scaffold=self._allow_internal_scaffold(resolved_prompt),
                )
                self._ensure_conversational_turn_fallback(response=response, interaction_profile=interaction_profile)
                response["internal_scaffold_visible"] = self._allow_internal_scaffold(resolved_prompt)
                self._shape_reasoning_body_from_conversation_turn(
                    response=response,
                    interaction_profile=interaction_profile,
                    allow_internal_scaffold=bool(response.get("internal_scaffold_visible")),
                )
                self._apply_belief_discussion_policy(
                    response=response,
                    prompt=resolved_prompt,
                )
                self._finalize_explanatory_answer(
                    response=response,
                    prompt=resolved_prompt,
                    route=route,
                    interaction_profile=interaction_profile,
                    entities=pipeline_result.nlu.entities,
                    provider_text=str(getattr(inference_result, "output_text", "") or ""),
                    recent_interactions=recent_interactions,
                    route_support_signals=route_support_signals,
                )
                self._ensure_conversational_turn_fallback(response=response, interaction_profile=interaction_profile)
                self._attach_memory_retrieval(response=response, memory_retrieval=memory_retrieval)
                response["route_support_signals"] = route_support_signals.to_dict()
                self._apply_memory_recall_surface(response=response, memory_retrieval=memory_retrieval)
                self._apply_safety_response_constraint(response=response, safety_decision=safety_decision)
                self._enforce_final_surface_lane(response=response, selected_mode=route.mode)
                self._attach_execution_and_packaging(
                    response=response,
                    pipeline_trace=pipeline_trace,
                    mode="research",
                    kind=route.kind,
                    route=route,
                    interaction_profile=interaction_profile,
                )
                self._attach_wake_metadata(response=response, wake_interaction=wake_interaction)
                self._attach_wake_resolution(
                    response=response,
                    original_prompt=prompt,
                    effective_prompt=effective_prompt,
                    wake_interaction=wake_interaction,
                )
                self._attach_persistence_observation(
                    response=response,
                    pipeline_trace=pipeline_trace,
                    session_id=session_id,
                    prompt=prompt,
                    front_half=pipeline_result,
                    route=route,
                    clarification_decision=clarification_decision,
                    validation_context=validation_context,
                    reasoning_frame_assembly=None,
                )
                self._attach_response_behavior_posture(
                    response=response,
                    route_mode=route.mode,
                    low_confidence_recovery=pipeline_result.low_confidence_recovery,
                    srd_diagnostic=pipeline_result.srd_diagnostic,
                    state_control=pipeline_result.state_control,
                )
                response["pipeline_trace"] = pipeline_trace.to_dict()
                self._attach_route_metadata(response=response, route=route, route_status="hosted_fallback")
                if clarification_continuation is not None:
                    response["resolved_prompt"] = resolved_prompt
                    response["resolution_strategy"] = "clarification_route_confirmation"
                    response["resolution_reason"] = (
                        "Resumed the most recent planning clarification after the user confirmed the current route."
                    )
                elif pipeline_result.resolution_changed:
                    response["resolved_prompt"] = resolved_prompt
                    response["resolution_strategy"] = pipeline_result.resolution_strategy
                    response["resolution_reason"] = pipeline_result.resolution_reason
                self._record_interaction(prompt=prompt, session_id=session_id, response=response)
                return response
        reasoning_frame_assembly = None
        if route.mode in {"planning", "research"}:
            reasoning_frame_assembly = self.reasoning_pipeline.assemble_reasoning_frame(
                prompt=resolved_prompt,
                route=route,
                kind=route.kind,
                interaction_profile=interaction_profile,
                validation_context=validation_context,
            )
            self.reasoning_pipeline.record_reasoning_frame(pipeline_trace, reasoning_frame_assembly)

        if route.mode == "tool":
            effective_input_path = input_path or self._infer_tool_input_path(
                active_thread=active_thread,
                resolution_strategy=pipeline_result.resolution_strategy,
            )
            if effective_input_path is None:
                effective_input_path = self._infer_live_tool_input_path(
                    prompt=prompt,
                    resolved_prompt=resolved_prompt,
                )
            effective_params = params or self._infer_tool_params(
                prompt=prompt,
                resolved_prompt=resolved_prompt,
                active_thread=active_thread,
                resolution_strategy=pipeline_result.resolution_strategy,
            )
            action, target = self._split_prompt_to_command(resolved_prompt)
            parsed = self.command_parser.parse(
                action=action,
                target=target,
                input_path=effective_input_path,
                params=effective_params,
                session_id=session_id,
                run_root=run_root,
            )
            routed = self.intent_router.route(parsed)
            reasoning_state = self.reasoning_state_service.apply_tool_usage_intent(
                state=reasoning_state,
                tool_id=routed.tool_id,
                capability=routed.capability,
                input_path=routed.input_path,
                params=dict(routed.params),
            )
            missing_input_response = self._build_missing_tool_input_response(
                route=route,
                routed=routed,
                prompt=prompt,
                resolved_prompt=resolved_prompt,
                params_supplied=params,
            )
            if missing_input_response is not None:
                response = missing_input_response
                gate_decision = ToolThresholdDecision(
                    should_use_tool=False,
                    rationale="Tool execution was skipped because the route lacked the structured inputs required to run safely.",
                    expected_confidence_gain=0.0,
                    selected_tool=routed.tool_id,
                    selected_bundle=routed.tool_id,
                    internal_reasoning_sufficient=True,
                )
                reasoning_state = self.reasoning_state_service.apply_tool_decision(
                    state=reasoning_state,
                    decision=gate_decision,
                )
                response["tool_threshold_decision"] = gate_decision.to_dict()
                self._attach_tool_access_metadata(
                    response=response,
                    tool_id=routed.tool_id,
                    capability=routed.capability,
                    execution_required=False,
                    final_source="tool_input_diagnostic",
                )
                skipped_outcome = self.reasoning_state_service.classify_execution_outcome(
                    skipped_reason="missing_structured_inputs"
                )
                reasoning_state = self.reasoning_state_service.apply_execution_outcome(
                    state=reasoning_state,
                    outcome=skipped_outcome,
                )
                self._attach_execution_outcome(response=response, outcome=skipped_outcome)
                self._attach_reasoning_state(response=response, reasoning_state=reasoning_state)
                self._attach_profile_context(
                    response=response,
                    interaction_profile=interaction_profile,
                    profile_advice=pipeline_result.nlu.profile_advice,
                    client_surface=client_surface,
                )
                self._attach_wake_metadata(response=response, wake_interaction=wake_interaction)
                self._attach_wake_resolution(
                    response=response,
                    original_prompt=prompt,
                    effective_prompt=effective_prompt,
                    wake_interaction=wake_interaction,
                )
                self._attach_dialogue_context(
                    response=response,
                    vibe_catcher=pipeline_result.vibe_catcher,
                    dialogue_management=pipeline_result.dialogue_management,
                    conversation_awareness=pipeline_result.conversation_awareness,
                    low_confidence_recovery=pipeline_result.low_confidence_recovery,
                    srd_diagnostic=pipeline_result.srd_diagnostic,
                    empathy_model=pipeline_result.empathy_model,
                    human_language_layer=pipeline_result.human_language_layer,
                    state_control=pipeline_result.state_control,
                    thought_framing=pipeline_result.thought_framing,
                )
                self._attach_execution_and_packaging(
                    response=response,
                    pipeline_trace=pipeline_trace,
                    mode="tool",
                    kind=route.kind,
                    route=route,
                    interaction_profile=interaction_profile,
                )
                self._attach_persistence_observation(
                    response=response,
                    pipeline_trace=pipeline_trace,
                    session_id=session_id,
                    prompt=prompt,
                    front_half=pipeline_result,
                    route=route,
                    clarification_decision=clarification_decision,
                    validation_context=validation_context,
                    reasoning_frame_assembly=None,
                )
                self._attach_response_behavior_posture(
                    response=response,
                    route_mode=route.mode,
                    low_confidence_recovery=pipeline_result.low_confidence_recovery,
                    srd_diagnostic=pipeline_result.srd_diagnostic,
                    state_control=pipeline_result.state_control,
                )
                self._attach_memory_retrieval(response=response, memory_retrieval=memory_retrieval)
                response["route_support_signals"] = route_support_signals.to_dict()
                response["pipeline_trace"] = pipeline_trace.to_dict()
                self._record_interaction(prompt=prompt, session_id=session_id, response=response)
                return response
            capability_safety = (
                self.safety_service.capability_safety_profile(
                    tool_id=routed.tool_id,
                    capability=routed.capability,
                )
                if self.safety_service is not None
                and hasattr(self.safety_service, "capability_safety_profile")
                else {"level": "allowed", "notes": ""}
            )
            gate_decision = self.tool_threshold_gate.decide(
                prompt=resolved_prompt,
                route_mode=route.mode,
                route_kind=route.kind,
                route_confidence=float(getattr(route, "confidence", 0.0) or 0.0),
                tool_id=routed.tool_id,
                capability=routed.capability,
                input_path=routed.input_path,
                params=dict(routed.params),
            )
            tool_recommendation = self.supervised_decision_support.advise_tool_decision(
                prompt=resolved_prompt,
                route_mode=str(route.mode or ""),
                route_kind=str(route.kind or ""),
                tool_id=routed.tool_id,
                current_decision=gate_decision,
            )
            reasoning_state = self._update_supervised_support_trace(
                reasoning_state=reasoning_state,
                supervised_support_trace=self.supervised_decision_support.record_recommendation(
                    trace=self._supervised_support_trace_from_state(reasoning_state),
                    recommendation=tool_recommendation,
                ),
            )
            reasoning_state = self.reasoning_state_service.apply_tool_decision(
                state=reasoning_state,
                decision=gate_decision,
            )
            if not gate_decision.should_use_tool:
                response = ConversationResponse(
                    mode="conversation",
                    kind="conversation.tool_gate_bypass",
                    summary="I can handle this directly without running a tool.",
                    reply="I can handle this directly without running a tool.",
                ).to_dict()
                response["tool_threshold_decision"] = gate_decision.to_dict()
                self._attach_tool_access_metadata(
                    response=response,
                    tool_id=routed.tool_id,
                    capability=routed.capability,
                    execution_required=False,
                    final_source="tool_threshold_bypass",
                )
                skipped_outcome = self.reasoning_state_service.classify_execution_outcome(
                    skipped_reason="tool_threshold_gate"
                )
                reasoning_state = self.reasoning_state_service.apply_execution_outcome(
                    state=reasoning_state,
                    outcome=skipped_outcome,
                )
                self._attach_execution_outcome(response=response, outcome=skipped_outcome)
                self._attach_reasoning_state(response=response, reasoning_state=reasoning_state)
                self._attach_profile_context(
                    response=response,
                    interaction_profile=interaction_profile,
                    profile_advice=pipeline_result.nlu.profile_advice,
                    client_surface=client_surface,
                )
                self._attach_wake_metadata(response=response, wake_interaction=wake_interaction)
                self._attach_wake_resolution(
                    response=response,
                    original_prompt=prompt,
                    effective_prompt=effective_prompt,
                    wake_interaction=wake_interaction,
                )
                self._attach_dialogue_context(
                    response=response,
                    vibe_catcher=pipeline_result.vibe_catcher,
                    dialogue_management=pipeline_result.dialogue_management,
                    conversation_awareness=pipeline_result.conversation_awareness,
                    low_confidence_recovery=pipeline_result.low_confidence_recovery,
                    srd_diagnostic=pipeline_result.srd_diagnostic,
                    empathy_model=pipeline_result.empathy_model,
                    human_language_layer=pipeline_result.human_language_layer,
                    state_control=pipeline_result.state_control,
                    thought_framing=pipeline_result.thought_framing,
                )
                self._attach_memory_retrieval(response=response, memory_retrieval=memory_retrieval)
                response["route_support_signals"] = route_support_signals.to_dict()
                response["pipeline_trace"] = pipeline_trace.to_dict()
                self._record_interaction(prompt=prompt, session_id=session_id, response=response)
                return response
            if self._should_constrain_tool_execution(
                safety_decision=safety_decision,
                capability_safety=capability_safety,
            ):
                response = ToolAssistantResponse(
                    mode="tool",
                    kind=route.kind,
                    summary=(
                        "I can stay with that at a high level, but I won't turn it into live tool execution."
                    ),
                    route=route.to_metadata(),
                    tool_execution=ToolExecutionDetails(
                        tool_id=routed.tool_id,
                        capability=routed.capability,
                        input_path=routed.input_path,
                        params=dict(routed.params),
                    ),
                    tool_route_origin=(
                        "nlu_hint_alias"
                        if pipeline_result.resolution_strategy == "tool_hint_alias"
                        else (
                            "hybrid_signal"
                            if pipeline_result.resolution_strategy == "tool_signal_alias"
                            or route.source == "hybrid_signal"
                            else "exact_alias"
                        )
                    ),
                ).to_dict()
                response["tool_threshold_decision"] = gate_decision.to_dict()
                self._attach_tool_access_metadata(
                    response=response,
                    tool_id=routed.tool_id,
                    capability=routed.capability,
                    execution_required=False,
                    final_source="tool_execution_constrained",
                )
                response["tool_execution_skipped"] = True
                response["tool_execution_skipped_reason"] = "dual_use_constraint"
                response["tool_capability_safety"] = capability_safety
                response["tool_constraint"] = dict(
                    getattr(safety_decision, "tool_constraint", {}) or {}
                )
                skipped_outcome = self.reasoning_state_service.classify_execution_outcome(
                    skipped_reason="dual_use_constraint"
                )
                reasoning_state = self.reasoning_state_service.apply_execution_outcome(
                    state=reasoning_state,
                    outcome=skipped_outcome,
                )
                self._attach_execution_outcome(response=response, outcome=skipped_outcome)
                self._attach_reasoning_state(response=response, reasoning_state=reasoning_state)
                self._apply_safety_response_constraint(response=response, safety_decision=safety_decision)
                self._attach_profile_context(
                    response=response,
                    interaction_profile=interaction_profile,
                    profile_advice=pipeline_result.nlu.profile_advice,
                    client_surface=client_surface,
                )
                self._attach_wake_metadata(response=response, wake_interaction=wake_interaction)
                self._attach_wake_resolution(
                    response=response,
                    original_prompt=prompt,
                    effective_prompt=effective_prompt,
                    wake_interaction=wake_interaction,
                )
                self._attach_route_metadata(response=response, route=route, route_status="selected")
                self._attach_response_behavior_posture(
                    response=response,
                    route_mode=route.mode,
                    low_confidence_recovery=pipeline_result.low_confidence_recovery,
                    srd_diagnostic=pipeline_result.srd_diagnostic,
                    state_control=pipeline_result.state_control,
                )
                if pipeline_result.resolution_changed:
                    response["resolved_prompt"] = resolved_prompt
                    response["resolution_strategy"] = pipeline_result.resolution_strategy
                    response["resolution_reason"] = pipeline_result.resolution_reason
                response["route_support_signals"] = route_support_signals.to_dict()
                response["pipeline_trace"] = pipeline_trace.to_dict()
                self._record_interaction(prompt=prompt, session_id=session_id, response=response)
                return response
            try:
                result = self.tool_execution_service.run_tool(
                    tool_id=routed.tool_id,
                    capability=routed.capability,
                    input_path=routed.input_path,
                    params=routed.params,
                    session_id=routed.session_id,
                    run_root=routed.run_root,
                )
            except Exception as exc:
                result = self._tool_exception_result(
                    tool_id=routed.tool_id,
                    capability=routed.capability,
                    exc=exc,
                )
            response = ToolAssistantResponse(
                mode="tool",
                kind=route.kind,
                summary=result.summary,
                route=route.to_metadata(),
                tool_execution=ToolExecutionDetails(
                    tool_id=routed.tool_id,
                    capability=routed.capability,
                    input_path=routed.input_path,
                    params=dict(routed.params),
                ),
                tool_result=result,
                tool_route_origin=(
                    "nlu_hint_alias"
                    if pipeline_result.resolution_strategy == "tool_hint_alias"
                    else (
                        "hybrid_signal"
                        if pipeline_result.resolution_strategy == "tool_signal_alias"
                        or route.source == "hybrid_signal"
                        else "exact_alias"
                    )
                ),
            ).to_dict()
            response["tool_threshold_decision"] = gate_decision.to_dict()
            self._attach_tool_access_metadata(
                response=response,
                tool_id=routed.tool_id,
                capability=routed.capability,
                execution_required=True,
                final_source="local_tool_execution",
            )
            self._apply_tool_result_surface(
                response=response,
                tool_result=result,
            )
            outcome = self.reasoning_state_service.classify_execution_outcome(tool_result=result)
            reasoning_state = self.reasoning_state_service.apply_execution_outcome(
                state=reasoning_state,
                outcome=outcome,
            )
            self._integrate_tool_execution_into_response(
                response=response,
                tool_result=result,
                outcome=outcome,
                reasoning_state=reasoning_state,
            )
            self._attach_execution_outcome(response=response, outcome=outcome)
            self._attach_reasoning_state(response=response, reasoning_state=reasoning_state)
            response["tool_capability_safety"] = capability_safety
            self._attach_capability_status(response=response)
            if clarification_continuation is not None:
                response["resolved_prompt"] = resolved_prompt
                response["resolution_strategy"] = "clarification_route_confirmation"
                response["resolution_reason"] = (
                    "Resumed the most recent planning clarification after the user confirmed the current route."
                )
            elif pipeline_result.resolution_changed:
                response["resolved_prompt"] = resolved_prompt
                response["resolution_strategy"] = pipeline_result.resolution_strategy
                response["resolution_reason"] = pipeline_result.resolution_reason
            self._attach_profile_context(
                response=response,
                interaction_profile=interaction_profile,
                profile_advice=pipeline_result.nlu.profile_advice,
                client_surface=client_surface,
            )
            self._attach_wake_metadata(response=response, wake_interaction=wake_interaction)
            self._attach_wake_resolution(
                response=response,
                original_prompt=prompt,
                effective_prompt=effective_prompt,
                wake_interaction=wake_interaction,
            )
            self._attach_dialogue_context(
                response=response,
                vibe_catcher=pipeline_result.vibe_catcher,
                dialogue_management=pipeline_result.dialogue_management,
                conversation_awareness=pipeline_result.conversation_awareness,
                low_confidence_recovery=pipeline_result.low_confidence_recovery,
                srd_diagnostic=pipeline_result.srd_diagnostic,
                empathy_model=pipeline_result.empathy_model,
                human_language_layer=pipeline_result.human_language_layer,
                state_control=pipeline_result.state_control,
                thought_framing=pipeline_result.thought_framing,
            )
            self._attach_execution_and_packaging(
                response=response,
                pipeline_trace=pipeline_trace,
                mode="tool",
                kind=route.kind,
                route=route,
                interaction_profile=interaction_profile,
            )
            self._attach_persistence_observation(
                response=response,
                pipeline_trace=pipeline_trace,
                session_id=session_id,
                prompt=prompt,
                front_half=pipeline_result,
                route=route,
                clarification_decision=clarification_decision,
                validation_context=validation_context,
                reasoning_frame_assembly=None,
            )
            self._attach_response_behavior_posture(
                response=response,
                route_mode=route.mode,
                low_confidence_recovery=pipeline_result.low_confidence_recovery,
                srd_diagnostic=pipeline_result.srd_diagnostic,
                state_control=pipeline_result.state_control,
            )
            self._attach_memory_retrieval(response=response, memory_retrieval=memory_retrieval)
            response["route_support_signals"] = route_support_signals.to_dict()
            response["pipeline_trace"] = pipeline_trace.to_dict()
            self._record_interaction(prompt=prompt, session_id=session_id, response=response)
            return response

        if route.mode == "planning":
            design_spec_result = None
            if DesignSurfaceSupport.should_use_design_tool(
                prompt=resolved_prompt,
                route_kind=route.kind,
            ):
                design_tool_params = DesignSurfaceSupport.build_design_tool_params(
                    prompt=resolved_prompt,
                    interaction_style=InteractionStylePolicy.interaction_style(interaction_profile),
                )
                reasoning_state = self.reasoning_state_service.apply_tool_usage_intent(
                    state=reasoning_state,
                    tool_id="design",
                    capability="system_spec",
                    input_path=None,
                    params=design_tool_params,
                )
                design_gate_decision = self.tool_threshold_gate.decide(
                    prompt=resolved_prompt,
                    route_mode=route.mode,
                    route_kind=route.kind,
                    route_confidence=float(getattr(route, "confidence", 0.0) or 0.0),
                    tool_id="design",
                    capability="system_spec",
                    input_path=None,
                    params=design_tool_params,
                )
                design_tool_recommendation = self.supervised_decision_support.advise_tool_decision(
                    prompt=resolved_prompt,
                    route_mode=str(route.mode or ""),
                    route_kind=str(route.kind or ""),
                    tool_id="design",
                    current_decision=design_gate_decision,
                )
                reasoning_state = self._update_supervised_support_trace(
                    reasoning_state=reasoning_state,
                    supervised_support_trace=self.supervised_decision_support.record_recommendation(
                        trace=self._supervised_support_trace_from_state(reasoning_state),
                        recommendation=design_tool_recommendation,
                    ),
                )
                reasoning_state = self.reasoning_state_service.apply_tool_decision(
                    state=reasoning_state,
                    decision=design_gate_decision,
                )
                if design_gate_decision.should_use_tool:
                    design_spec_result = self._maybe_run_design_spec_tool(
                        prompt=resolved_prompt,
                        route_kind=route.kind,
                        interaction_profile=interaction_profile,
                        session_id=session_id,
                        run_root=run_root,
                    )
                    if design_spec_result is not None:
                        design_outcome = self.reasoning_state_service.classify_execution_outcome(
                            tool_result=design_spec_result
                        )
                        reasoning_state = self.reasoning_state_service.apply_execution_outcome(
                            state=reasoning_state,
                            outcome=design_outcome,
                        )
            response = self.planner.respond(
                resolved_prompt,
                kind=route.kind,
                context=archive_context,
                reasoning_frame_assembly=reasoning_frame_assembly,
            )
            self._attach_reasoning_state(response=response, reasoning_state=reasoning_state)
            if design_spec_result is not None:
                self._apply_design_spec_to_planning_response(
                    response=response,
                    design_spec_result=design_spec_result,
                )
            else:
                self._ensure_design_concept_response(
                    response=response,
                    prompt=prompt,
                    resolved_prompt=resolved_prompt,
                    route_kind=route.kind,
                )
            if str(route.kind or "").strip() == "planning.architecture":
                response["capability_status"] = {
                    "domain_id": "invention_design",
                    "status": "bounded",
                    "details": "Architecture and invention support stay high-level, constraint-aware, and non-signoff.",
                }
            self._attach_reasoning_outputs(
                response=response,
                prompt=resolved_prompt,
                pipeline_trace=pipeline_trace,
                mode="planning",
                kind=route.kind,
                route=route,
                reasoning_frame_assembly=reasoning_frame_assembly,
                interaction_profile=interaction_profile,
                profile_advice=pipeline_result.nlu.profile_advice,
                client_surface=client_surface,
                recent_interactions=recent_interactions,
            )
            self._attach_memory_retrieval(response=response, memory_retrieval=memory_retrieval)
            response["route_support_signals"] = route_support_signals.to_dict()
            self._apply_memory_recall_surface(response=response, memory_retrieval=memory_retrieval)
            self._apply_safety_response_constraint(response=response, safety_decision=safety_decision)
            self._enforce_final_surface_lane(response=response, selected_mode=route.mode)
            self._attach_execution_and_packaging(
                response=response,
                pipeline_trace=pipeline_trace,
                mode="planning",
                kind=route.kind,
                route=route,
                interaction_profile=interaction_profile,
            )
            self._attach_wake_metadata(response=response, wake_interaction=wake_interaction)
            self._attach_wake_resolution(
                response=response,
                original_prompt=prompt,
                effective_prompt=effective_prompt,
                wake_interaction=wake_interaction,
            )
            self._attach_persistence_observation(
                response=response,
                pipeline_trace=pipeline_trace,
                session_id=session_id,
                prompt=prompt,
                front_half=pipeline_result,
                route=route,
                clarification_decision=clarification_decision,
                validation_context=validation_context,
                reasoning_frame_assembly=reasoning_frame_assembly,
            )
            self._attach_response_behavior_posture(
                response=response,
                route_mode=route.mode,
                low_confidence_recovery=pipeline_result.low_confidence_recovery,
                srd_diagnostic=pipeline_result.srd_diagnostic,
                state_control=pipeline_result.state_control,
            )
            response["pipeline_trace"] = pipeline_trace.to_dict()
            self._attach_route_metadata(response=response, route=route, route_status=reasoning_frame_assembly.route_status)
            if clarification_continuation is not None:
                response["resolved_prompt"] = resolved_prompt
                response["resolution_strategy"] = "clarification_route_confirmation"
                response["resolution_reason"] = (
                    "Resumed the most recent planning clarification after the user confirmed the current route."
                )
            elif pipeline_result.resolution_changed:
                response["resolved_prompt"] = resolved_prompt
                response["resolution_strategy"] = pipeline_result.resolution_strategy
                response["resolution_reason"] = pipeline_result.resolution_reason
            self._record_interaction(prompt=prompt, session_id=session_id, response=response)
            return response

        response = self.research_engine.respond(
            resolved_prompt,
            kind=route.kind,
            context=archive_context,
            reasoning_frame_assembly=reasoning_frame_assembly,
        )
        self._attach_reasoning_state(response=response, reasoning_state=reasoning_state)
        self._attach_reasoning_outputs(
            response=response,
            prompt=resolved_prompt,
            pipeline_trace=pipeline_trace,
            mode="research",
            kind=route.kind,
            route=route,
            reasoning_frame_assembly=reasoning_frame_assembly,
            interaction_profile=interaction_profile,
            profile_advice=pipeline_result.nlu.profile_advice,
            client_surface=client_surface,
            recent_interactions=recent_interactions,
        )
        self._attach_wake_metadata(response=response, wake_interaction=wake_interaction)
        self._attach_wake_resolution(
            response=response,
            original_prompt=prompt,
            effective_prompt=effective_prompt,
            wake_interaction=wake_interaction,
        )
        self._finalize_explanatory_answer(
            response=response,
            prompt=resolved_prompt,
            route=route,
            interaction_profile=interaction_profile,
            entities=pipeline_result.nlu.entities,
            provider_text=None,
            recent_interactions=recent_interactions,
            route_support_signals=route_support_signals,
        )
        if not response.get("user_facing_answer"):
            self._ensure_conversational_turn_fallback(response=response, interaction_profile=interaction_profile)
        self._attach_memory_retrieval(response=response, memory_retrieval=memory_retrieval)
        response["route_support_signals"] = route_support_signals.to_dict()
        self._apply_memory_recall_surface(response=response, memory_retrieval=memory_retrieval)
        self._apply_safety_response_constraint(response=response, safety_decision=safety_decision)
        self._enforce_final_surface_lane(response=response, selected_mode=route.mode)
        self._attach_execution_and_packaging(
            response=response,
            pipeline_trace=pipeline_trace,
            mode="research",
            kind=route.kind,
            route=route,
            interaction_profile=interaction_profile,
        )
        self._attach_persistence_observation(
            response=response,
            pipeline_trace=pipeline_trace,
            session_id=session_id,
            prompt=prompt,
            front_half=pipeline_result,
            route=route,
            clarification_decision=clarification_decision,
            validation_context=validation_context,
            reasoning_frame_assembly=reasoning_frame_assembly,
        )
        self._attach_response_behavior_posture(
            response=response,
            route_mode=route.mode,
            low_confidence_recovery=pipeline_result.low_confidence_recovery,
            srd_diagnostic=pipeline_result.srd_diagnostic,
            state_control=pipeline_result.state_control,
        )
        response["pipeline_trace"] = pipeline_trace.to_dict()
        self._attach_route_metadata(response=response, route=route, route_status=reasoning_frame_assembly.route_status)
        if clarification_continuation is not None:
            response["resolved_prompt"] = resolved_prompt
            response["resolution_strategy"] = "clarification_route_confirmation"
            response["resolution_reason"] = (
                "Resumed the most recent planning clarification after the user confirmed the current route."
            )
        elif pipeline_result.resolution_changed:
            response["resolved_prompt"] = resolved_prompt
            response["resolution_strategy"] = pipeline_result.resolution_strategy
            response["resolution_reason"] = pipeline_result.resolution_reason
        self._record_interaction(prompt=prompt, session_id=session_id, response=response)
        return response

    def _stage_intake(
        self,
        *,
        prompt: str,
        input_path: Path | None,
        params: dict[str, int | float | str] | None,
        session_id: str,
        run_root: Path | None,
        client_surface: str,
    ) -> InteractionTurnContext:
        wake_interaction = SocialInteractionPolicy.detect_wake_interaction(prompt)
        effective_prompt = prompt
        active_thread = self.session_context_service.get_active_thread(session_id)
        interaction_profile = self.session_context_service.get_interaction_profile(session_id)
        if wake_interaction is not None:
            classification = str(wake_interaction.get("classification") or "").strip()
            stripped_prompt = str(wake_interaction.get("stripped_prompt") or "").strip()
            if classification == "pure_greeting":
                effective_prompt = stripped_prompt or prompt
            elif classification == "greeting_plus_request" and stripped_prompt:
                effective_prompt = stripped_prompt
        attached_input_prompt = InteractionFlowSupport.rewrite_prompt_for_attached_input(
            prompt=effective_prompt,
            input_path=input_path,
        )
        if attached_input_prompt:
            effective_prompt = attached_input_prompt
        reasoning_state = self.reasoning_state_service.initialize(
            prompt=effective_prompt,
            active_thread=active_thread,
            interaction_profile=interaction_profile,
            input_path=input_path,
        )
        effective_prompt, reasoning_state = self.reasoning_state_service.rewrite_prompt_for_stateful_follow_up(
            prompt=effective_prompt,
            state=reasoning_state,
        )
        return InteractionTurnContext(
            original_prompt=prompt,
            effective_prompt=effective_prompt,
            session_id=session_id,
            client_surface=client_surface,
            input_path=input_path,
            params=params,
            run_root=run_root,
            wake_interaction=wake_interaction,
            active_thread=active_thread,
            interaction_profile=interaction_profile,
            reasoning_state=reasoning_state,
        )

    def _recent_interactions_for_turn(
        self,
        turn: InteractionTurnContext,
    ) -> list[dict[str, object]]:
        session_recent = self.interaction_history_service.recent_records(
            session_id=turn.session_id,
            limit=8,
        )
        if session_recent:
            return session_recent
        if turn.active_thread is not None:
            return session_recent
        project_recent = self._project_recent_interactions(
            session_id=turn.session_id,
            limit=2,
        )
        return project_recent or session_recent

    def _stage_route_preparation(
        self,
        *,
        turn: InteractionTurnContext,
        prompt: str,
        recent_interactions: list[dict[str, object]],
        clarification_continuation: dict[str, object] | None,
    ) -> InteractionTurnContext:
        interaction_summary = self.interaction_history_service.summarize_interactions(
            session_id=turn.session_id,
        )
        pipeline_result, route = self.reasoning_pipeline.run_front_half(
            prompt=str(turn.pipeline_prompt or turn.effective_prompt),
            session_id=turn.session_id,
            interaction_profile=turn.interaction_profile,
            interaction_summary=interaction_summary,
            recent_interactions=recent_interactions,
            active_thread=turn.active_thread,
        )
        pipeline_trace = self.reasoning_pipeline.create_trace(pipeline_result)
        resolved_prompt = pipeline_result.resolved_prompt
        prompt_understanding = self.prompt_nlu.analyze(resolved_prompt)
        reasoning_state = turn.reasoning_state or ReasoningStateFrame()
        if clarification_continuation is not None and str(clarification_continuation.get("action") or "") == "continue":
            resolved_prompt = str(clarification_continuation.get("working_prompt") or resolved_prompt)
            prompt_understanding = self.prompt_nlu.analyze(resolved_prompt)
            reasoning_state = self.reasoning_state_service.apply_clarification_continuation(
                state=reasoning_state,
                continuation=clarification_continuation,
            )
        route = self.route_adaptation_policy.adapt_for_session_intent(
            route,
            interaction_summary=interaction_summary,
        )
        route = self.route_adaptation_policy.adapt_for_retrieval_bias(
            route,
            interaction_summary=interaction_summary,
        )
        if clarification_continuation is not None and str(clarification_continuation.get("action") or "") == "continue":
            route = DomainRoute(
                mode=str(clarification_continuation.get("mode") or route.mode),
                kind=str(clarification_continuation.get("kind") or route.kind),
                normalized_prompt=resolved_prompt,
                confidence=float(clarification_continuation.get("confidence") or route.confidence),
                reason="Confirmed continuation of the most recent planning clarification route.",
                source="clarification_continuation",
                evidence=[
                    {
                        "label": "clarification_continuation",
                        "detail": "User explicitly confirmed the previously clarified planning route.",
                    }
                ],
                decision_summary=None,
            )
            clarification_decision = ClarificationGateDecision(
                action="proceed",
                should_clarify=False,
                trigger=None,
                repeated_pattern=False,
                question_style="none",
                reason="Most recent planning clarification was explicitly confirmed by the user.",
            )
        else:
            clarification_decision = self.reasoning_pipeline.run_clarification_gate(
                route=route,
                interaction_summary=interaction_summary,
                low_confidence_recovery=pipeline_result.low_confidence_recovery,
            )
            if clarification_decision.should_clarify and self._should_bypass_clarification_for_assistant_turn(
                resolved_prompt=resolved_prompt,
                route=route,
                active_thread=turn.active_thread,
                recent_interactions=recent_interactions,
            ):
                clarification_decision = ClarificationGateDecision(
                    action="proceed",
                    should_clarify=False,
                    trigger=None,
                    repeated_pattern=False,
                    question_style="none",
                    reason="Known-topic prompt should prefer a grounded direct answer unless ambiguity is critical.",
                )
        self.reasoning_pipeline.record_clarification_decision(pipeline_trace, clarification_decision)
        route_authority = self._build_route_authority_decision(route=route)
        supervised_support_trace = self.supervised_decision_support.empty_trace()
        route_recommendation = self.supervised_decision_support.advise_route_decision(
            prompt=resolved_prompt,
            route_mode=str(route.mode or ""),
            route_kind=str(route.kind or ""),
            current_mode=str(route.mode or ""),
            current_kind=str(route.kind or ""),
        )
        supervised_support_trace = self.supervised_decision_support.record_recommendation(
            trace=supervised_support_trace,
            recommendation=route_recommendation,
        )
        if pipeline_result.intent_domain is not None:
            assisted_intent_domain, domain_recommendation = self.supervised_decision_support.assist_intent_domain(
                prompt=resolved_prompt,
                route_mode=str(route.mode or ""),
                route_kind=str(route.kind or ""),
                current=pipeline_result.intent_domain,
            )
            if assisted_intent_domain is not None:
                pipeline_result.intent_domain = assisted_intent_domain
            supervised_support_trace = self.supervised_decision_support.record_recommendation(
                trace=supervised_support_trace,
                recommendation=domain_recommendation,
            )
        reasoning_state = self.reasoning_state_service.apply_route(
            state=reasoning_state,
            route_authority=route_authority,
            route=route,
            resolved_prompt=resolved_prompt,
            prompt_understanding=prompt_understanding,
            clarification_decision=clarification_decision,
            pipeline_result=pipeline_result,
        )
        memory_retrieval = (
            self.memory_retrieval_layer.retrieve(
                prompt=resolved_prompt,
                session_id=turn.session_id,
                project_id=self._project_id_hint,
                active_thread=turn.active_thread,
                route_mode=route.mode,
                recent_interactions=recent_interactions,
                prompt_understanding=prompt_understanding,
            )
            if self.memory_retrieval_layer is not None
            else None
        )
        memory_context_decision = self.memory_context_classifier.classify(
            retrieval=memory_retrieval,
            route_mode=route.mode,
            intent_domain=str(getattr(getattr(pipeline_result, "intent_domain", None), "domain", "") or ""),
            prompt=resolved_prompt,
        )
        if memory_retrieval is not None:
            memory_retrieval = memory_context_decision.filtered_result(memory_retrieval)
        reasoning_state = self.reasoning_state_service.apply_memory_context(
            state=reasoning_state,
            memory_context_decision=memory_context_decision,
        )
        reasoning_state = self._update_supervised_support_trace(
            reasoning_state=reasoning_state,
            supervised_support_trace=supervised_support_trace,
        )
        route_support_signals = RouteSupportSignalBuilder.build(prompt_understanding=prompt_understanding)
        return InteractionTurnContext(
            original_prompt=prompt,
            effective_prompt=turn.effective_prompt,
            session_id=turn.session_id,
            client_surface=turn.client_surface,
            input_path=turn.input_path,
            params=turn.params,
            run_root=turn.run_root,
            wake_interaction=turn.wake_interaction,
            active_thread=turn.active_thread,
            interaction_profile=turn.interaction_profile,
            recent_interactions=tuple(recent_interactions),
            clarification_continuation=clarification_continuation,
            pipeline_prompt=resolved_prompt,
            interaction_summary=interaction_summary,
            prompt_understanding=prompt_understanding,
            pipeline_result=pipeline_result,
            pipeline_trace=pipeline_trace,
            route=route,
            route_authority=route_authority,
            clarification_decision=clarification_decision,
            memory_retrieval=memory_retrieval,
            route_support_signals=route_support_signals,
            reasoning_state=reasoning_state,
            supervised_support_trace=supervised_support_trace,
        )

    @staticmethod
    def _build_route_authority_decision(*, route: DomainRoute) -> RouteAuthorityDecision:
        return RouteAuthorityDecision(
            mode=route.mode,
            kind=route.kind,
            normalized_prompt=route.normalized_prompt,
            confidence=route.confidence,
            reason=route.reason,
            source=route.source,
            route_metadata=route.to_metadata().to_dict(),
        )

    def _should_bypass_clarification_for_known_topic(self, *, resolved_prompt: str, route: DomainRoute) -> bool:
        if self.knowledge_service is None:
            return False
        if str(route.mode or "").strip() != "research":
            return False
        if "comparison" in str(route.kind or "").lower():
            return False
        lookup = ExplanationResponseBuilder.lookup_local_knowledge(
            prompt=resolved_prompt,
            knowledge_service=self.knowledge_service,
        )
        if lookup is not None and lookup.primary is not None:
            return True
        partial = self.knowledge_service.partial_lookup(resolved_prompt)
        return partial is not None and partial.primary is not None

    def _should_bypass_clarification_for_assistant_turn(
        self,
        *,
        resolved_prompt: str,
        route: DomainRoute,
        active_thread: dict[str, object] | None,
        recent_interactions: list[dict[str, object]],
    ) -> bool:
        if self._should_bypass_clarification_for_known_topic(
            resolved_prompt=resolved_prompt,
            route=route,
        ):
            return True
        normalized = PromptSurfaceBuilder.build(resolved_prompt).lookup_ready_text
        if self._looks_like_low_signal_assistant_prompt(normalized):
            return False
        if str(route.mode or "").strip() == "conversation":
            return True
        follow_up_prompts = {
            "why",
            "how so",
            "what next",
            "go on",
            "tell me more",
            "keep going",
            "what else",
            "what went wrong",
            "what failed",
            "i see the issue",
        }
        if normalized in follow_up_prompts and (active_thread or recent_interactions):
            return True
        return False

    @staticmethod
    def _looks_like_low_signal_assistant_prompt(normalized_prompt: str) -> bool:
        normalized = "".join(ch for ch in str(normalized_prompt or "").lower() if ch.isalpha())
        if len(normalized) < 3:
            return True
        if len(normalized) >= 7 and not any(ch in "aeiou" for ch in normalized):
            return True
        return False

    @staticmethod
    def _build_response_packaging_context(
        *,
        mode: str,
        kind: str,
        route: DomainRoute,
        client_surface: str,
        allow_internal_scaffold: bool,
    ) -> ResponsePackagingContext:
        return ResponsePackagingContext(
            mode=mode,
            kind=kind,
            client_surface=client_surface,
            route_mode=route.mode,
            route_kind=route.kind,
            allow_internal_scaffold=allow_internal_scaffold,
            packaging_boundary="interactive_reply",
        )

    @staticmethod
    def _build_hosted_research_response(
        *,
        prompt: str,
        route,
        inference_result,
        hosted_reason: str,
    ) -> dict[str, object]:
        output_text = str(getattr(inference_result, "output_text", "") or "").strip()
        summary = output_text.splitlines()[0].strip() if output_text else f"Hosted research response for: {prompt}"
        if len(summary) > 180:
            summary = f"{summary[:177].rstrip()}..."
        response = ResearchResponse(
            mode="research",
            kind=route.kind,
            summary=summary,
            findings=[output_text] if output_text else [],
            route=route.to_metadata(),
            recommendation=None,
        ).to_dict()
        response["provider_inference"] = {
            "provider_id": getattr(inference_result, "provider_id", None),
            "model": getattr(inference_result, "model", None),
            "finish_reason": getattr(inference_result, "finish_reason", None),
            "usage_reason": hosted_reason,
            "hosted_fallback": True,
        }
        response["local_context_assessment"] = "hosted_fallback"
        response["confidence_posture"] = "provisional"
        response["uncertainty_note"] = (
            "This answer was generated through the hosted inference fallback because the local archive did not have enough context to answer directly."
        )
        return response

    def _handle_writing_workflow(
        self,
        *,
        prompt: str,
        resolved_prompt: str,
        route,
        session_id: str,
        client_surface: str,
        interaction_profile,
        wake_interaction,
        effective_prompt: str,
        pipeline_result,
        reasoning_state: ReasoningStateFrame,
    ) -> dict[str, object] | None:
        workflow = WritingWorkflowSupport.classify(resolved_prompt)
        if workflow is None:
            return None
        hosted_decision = (
            self.inference_service.evaluate_hosted_writing()
            if self.inference_service is not None and hasattr(self.inference_service, "evaluate_hosted_writing")
            else HostedInferenceDecision(False, "No model provider is configured.")
        )
        if hosted_decision.use_hosted_inference and self.inference_service is not None:
            inference_result = self.inference_service.infer_writing_reply(
                prompt=resolved_prompt,
                session_id=session_id,
                interaction_profile=interaction_profile,
                workflow=workflow,
            )
            response = self._build_hosted_writing_response(
                prompt=resolved_prompt,
                route=route,
                inference_result=inference_result,
                hosted_reason=hosted_decision.reason,
                workflow=workflow,
            )
        else:
            response = self._build_provider_gated_writing_response(
                prompt=resolved_prompt,
                route=route,
                workflow=workflow,
                provider_reason=hosted_decision.reason,
            )
        self._attach_reasoning_state(response=response, reasoning_state=reasoning_state)
        self._attach_profile_context(
            response=response,
            interaction_profile=interaction_profile,
            profile_advice=pipeline_result.nlu.profile_advice,
            client_surface=client_surface,
        )
        self._attach_wake_metadata(response=response, wake_interaction=wake_interaction)
        self._attach_wake_resolution(
            response=response,
            original_prompt=prompt,
            effective_prompt=effective_prompt,
            wake_interaction=wake_interaction,
        )
        self._attach_dialogue_context(
            response=response,
            vibe_catcher=pipeline_result.vibe_catcher,
            dialogue_management=pipeline_result.dialogue_management,
            conversation_awareness=pipeline_result.conversation_awareness,
            low_confidence_recovery=pipeline_result.low_confidence_recovery,
            srd_diagnostic=pipeline_result.srd_diagnostic,
            empathy_model=pipeline_result.empathy_model,
            human_language_layer=pipeline_result.human_language_layer,
            state_control=pipeline_result.state_control,
            thought_framing=pipeline_result.thought_framing,
        )
        self._attach_response_behavior_posture(
            response=response,
            route_mode="research",
            low_confidence_recovery=pipeline_result.low_confidence_recovery,
            srd_diagnostic=pipeline_result.srd_diagnostic,
            state_control=pipeline_result.state_control,
        )
        return response

    def _handle_academic_workflow(
        self,
        *,
        prompt: str,
        resolved_prompt: str,
        route,
        input_path: Path | None,
        client_surface: str,
        interaction_profile,
        wake_interaction,
        effective_prompt: str,
        pipeline_result,
        reasoning_state: ReasoningStateFrame,
    ) -> dict[str, object] | None:
        decision = AcademicSupportService.classify(
            prompt=resolved_prompt,
            input_path=input_path,
        )
        if decision is None:
            return None
        response = AcademicSupportService.build_response(
            prompt=resolved_prompt,
            decision=decision,
            input_path=input_path,
        )
        self._attach_reasoning_state(response=response, reasoning_state=reasoning_state)
        self._attach_profile_context(
            response=response,
            interaction_profile=interaction_profile,
            profile_advice=pipeline_result.nlu.profile_advice,
            client_surface=client_surface,
        )
        self._attach_wake_metadata(response=response, wake_interaction=wake_interaction)
        self._attach_wake_resolution(
            response=response,
            original_prompt=prompt,
            effective_prompt=effective_prompt,
            wake_interaction=wake_interaction,
        )
        self._attach_dialogue_context(
            response=response,
            vibe_catcher=pipeline_result.vibe_catcher,
            dialogue_management=pipeline_result.dialogue_management,
            conversation_awareness=pipeline_result.conversation_awareness,
            low_confidence_recovery=pipeline_result.low_confidence_recovery,
            srd_diagnostic=pipeline_result.srd_diagnostic,
            empathy_model=pipeline_result.empathy_model,
            human_language_layer=pipeline_result.human_language_layer,
            state_control=pipeline_result.state_control,
            thought_framing=pipeline_result.thought_framing,
        )
        response["route"] = route.to_metadata().to_dict()
        self._attach_response_behavior_posture(
            response=response,
            route_mode="research",
            low_confidence_recovery=pipeline_result.low_confidence_recovery,
            srd_diagnostic=pipeline_result.srd_diagnostic,
            state_control=pipeline_result.state_control,
        )
        return response

    @staticmethod
    def _build_provider_gated_writing_response(
        *,
        prompt: str,
        route,
        workflow: dict[str, object],
        provider_reason: str,
    ) -> dict[str, object]:
        message = WritingWorkflowSupport.provider_gated_message(
            workflow=workflow,
            provider_reason=provider_reason,
        )
        response = ResearchResponse(
            mode="research",
            kind="research.writing_workflow",
            summary=message,
            findings=[message],
            route=route.to_metadata(),
            recommendation="If you want, I can still help plan the wording, structure, or constraints for the writing task locally.",
        ).to_dict()
        response["user_facing_answer"] = message
        response["capability_status"] = {
            "domain_id": "writing_editing",
            "status": "provider_gated",
            "details": "Explicit rewrite, translation, cleanup, and drafting workflows need a hosted provider for live generation.",
        }
        if str(workflow.get("workflow") or "") in {"paraphrase", "draft"}:
            response["academic_integrity_guidance"] = {
                "review_for_originality": True,
                "verify_citations_and_claims": True,
            }
        return response

    @staticmethod
    def _build_hosted_writing_response(
        *,
        prompt: str,
        route,
        inference_result,
        hosted_reason: str,
        workflow: dict[str, object],
    ) -> dict[str, object]:
        output_text = str(getattr(inference_result, "output_text", "") or "").strip()
        summary = output_text.splitlines()[0].strip() if output_text else f"Writing workflow completed for: {prompt}"
        response = ResearchResponse(
            mode="research",
            kind="research.writing_workflow",
            summary=summary,
            findings=[output_text] if output_text else [],
            route=route.to_metadata(),
            recommendation=None,
        ).to_dict()
        response["user_facing_answer"] = output_text or summary
        response["provider_inference"] = {
            "provider_id": getattr(inference_result, "provider_id", None),
            "model": getattr(inference_result, "model", None),
            "finish_reason": getattr(inference_result, "finish_reason", None),
            "usage_reason": hosted_reason,
            "hosted_writing": True,
            "workflow": str(workflow.get("workflow") or ""),
        }
        response["capability_status"] = {
            "domain_id": "writing_editing",
            "status": "supported",
            "details": "Hosted writing/editing support completed successfully.",
        }
        if str(workflow.get("workflow") or "") in {"paraphrase", "draft"}:
            response["academic_integrity_guidance"] = {
                "review_for_originality": True,
                "verify_citations_and_claims": True,
            }
        response["local_context_assessment"] = "hosted_writing"
        response["confidence_posture"] = "provisional"
        return response

    @staticmethod
    def _build_not_promised_capability_response(
        *,
        prompt: str,
        interaction_profile,
        contract: dict[str, object],
    ) -> dict[str, object]:
        label = str(contract.get("label") or "This capability").strip()
        scope_note = str(contract.get("scope_note") or "").strip()
        response = ResearchResponse(
            mode="research",
            kind="research.capability_boundary",
            summary=f"{label} is not part of Lumen's current runtime contract.",
            findings=[
                f"{label} is not currently promised.",
                scope_note,
            ],
            recommendation="If you want, I can help with the nearest supported alternative instead.",
        ).to_dict()
        response["user_facing_answer"] = "\n".join(
            line
            for line in (
                f"{label} is not part of Lumen's current runtime contract.",
                scope_note,
                "If you want, I can help with the nearest supported alternative instead.",
            )
            if line
        ).strip()
        response["capability_status"] = {
            "domain_id": str(contract.get("domain_id") or ""),
            "status": "not_promised",
            "details": scope_note,
        }
        return response

    def _finalize_explanatory_answer(
        self,
        *,
        response: dict[str, object],
        prompt: str,
        route,
        interaction_profile,
        entities,
        provider_text: str | None,
        recent_interactions: list[dict[str, object]] | None = None,
        route_support_signals=None,
    ) -> None:
        # Visible-answer authority stays here, but explanatory shaping is delegated to a
        # behavior-preserving coordinator so this file does not carry the entire lookup path.
        InteractionFlowSupport.finalize_explanatory_answer(
            response=response,
            prompt=prompt,
            route=route,
            interaction_profile=interaction_profile,
            entities=entities,
            provider_text=provider_text,
            recent_interactions=recent_interactions,
            route_support_signals=route_support_signals,
            knowledge_service=self.knowledge_service,
            explanation_surface_style=self._explanation_surface_style,
        )

    def _attach_profile_context(
        self,
        *,
        response: dict[str, object],
        interaction_profile,
        profile_advice,
        client_surface: str,
    ) -> None:
        ResponseContextSupport.attach_profile_context(
            response=response,
            interaction_profile=interaction_profile,
            profile_advice=profile_advice,
            client_surface=client_surface,
        )
        ModeResponseShaper.apply(
            response=response,
            interaction_profile=interaction_profile,
        )

    @staticmethod
    def _attach_wake_metadata(
        *,
        response: dict[str, object],
        wake_interaction: dict[str, object] | None,
    ) -> None:
        ResponseContextSupport.attach_wake_metadata(
            response=response,
            wake_interaction=wake_interaction,
        )

    @staticmethod
    def _attach_wake_resolution(
        *,
        response: dict[str, object],
        original_prompt: str,
        effective_prompt: str,
        wake_interaction: dict[str, object] | None,
    ) -> None:
        ResponseContextSupport.attach_wake_resolution(
            response=response,
            original_prompt=original_prompt,
            effective_prompt=effective_prompt,
            wake_interaction=wake_interaction,
        )

    @staticmethod
    def _attach_route_metadata(
        *,
        response: dict[str, object],
        route,
        route_status: str | None,
    ) -> None:
        ResponseContextSupport.attach_route_metadata(
            response=response,
            route=route,
            route_status=route_status,
        )
        response["route_authority"] = InteractionService._build_route_authority_decision(route=route).to_dict()

    def _attach_execution_and_packaging(
        self,
        *,
        response: dict[str, object],
        pipeline_trace,
        mode: str,
        kind: str,
        route,
        interaction_profile,
        client_surface: str = "main",
        allow_internal_scaffold: bool = False,
    ) -> None:
        ResponsePackagingSupport.attach_execution_and_packaging(
            response=response,
            pipeline_trace=pipeline_trace,
            mode=mode,
            kind=kind,
            route=route,
            interaction_profile=interaction_profile,
            reasoning_pipeline=self.reasoning_pipeline,
        )
        response["route_authority"] = self._build_route_authority_decision(route=route).to_dict()
        response["response_packaging_context"] = self._build_response_packaging_context(
            mode=mode,
            kind=kind,
            route=route,
            client_surface=client_surface,
            allow_internal_scaffold=allow_internal_scaffold,
        ).to_dict()

    def _attach_reasoning_outputs(
        self,
        *,
        response: dict[str, object],
        prompt: str,
        pipeline_trace,
        mode: str,
        kind: str,
        route,
        reasoning_frame_assembly,
        interaction_profile,
        profile_advice,
        client_surface: str,
        recent_interactions: list[dict[str, object]],
    ) -> None:
        tone_blend = self._response_tone_blend(prompt=prompt, route=route)
        response["response_tone_blend"] = tone_blend
        synthesis = self.reasoning_pipeline.synthesize_response(
            mode=mode,
            kind=kind,
            reasoning_frame_assembly=reasoning_frame_assembly,
            response_payload=response,
        )
        response["pipeline_synthesis"] = synthesis.to_dict()
        response["route_status"] = reasoning_frame_assembly.route_status
        response["support_status"] = reasoning_frame_assembly.support_status
        response["tension_status"] = reasoning_frame_assembly.tension_status
        self._apply_supervised_confidence_support(
            response=response,
            prompt=prompt,
            route=route,
        )
        self._attach_profile_context(
            response=response,
            interaction_profile=interaction_profile,
            profile_advice=profile_advice,
            client_surface=client_surface,
        )
        self._attach_dialogue_context(
            response=response,
            vibe_catcher=getattr(pipeline_trace, "vibe_catcher", None),
            dialogue_management=getattr(pipeline_trace, "dialogue_management", None),
            conversation_awareness=getattr(pipeline_trace, "conversation_awareness", None),
            low_confidence_recovery=getattr(pipeline_trace, "low_confidence_recovery", None),
            srd_diagnostic=getattr(pipeline_trace, "srd_diagnostic", None),
            empathy_model=getattr(pipeline_trace, "empathy_model", None),
            human_language_layer=getattr(pipeline_trace, "human_language_layer", None),
            state_control=getattr(pipeline_trace, "state_control", None),
            thought_framing=getattr(pipeline_trace, "thought_framing", None),
        )
        self._attach_conversational_turn(
            prompt=prompt,
            response=response,
            interaction_profile=interaction_profile,
            dialogue_management=getattr(pipeline_trace, "dialogue_management", None),
            conversation_awareness=getattr(pipeline_trace, "conversation_awareness", None),
            state_control=getattr(pipeline_trace, "state_control", None),
            human_language_layer=getattr(pipeline_trace, "human_language_layer", None),
            thought_framing=getattr(pipeline_trace, "thought_framing", None),
            recent_interactions=recent_interactions,
            tone_profile=str(tone_blend.get("tone_profile") or "default"),
            allow_internal_scaffold=self._allow_internal_scaffold(prompt),
        )
        self._ensure_conversational_turn_fallback(response=response, interaction_profile=interaction_profile)
        response["internal_scaffold_visible"] = self._allow_internal_scaffold(prompt)
        self._shape_reasoning_body_from_conversation_turn(
            response=response,
            interaction_profile=interaction_profile,
            allow_internal_scaffold=bool(response.get("internal_scaffold_visible")),
        )
        self._apply_belief_discussion_policy(
            response=response,
            prompt=prompt,
        )
        self._finalize_user_facing_reasoning_response(
            response=response,
            prompt=prompt,
            mode=mode,
            route=route,
            interaction_profile=interaction_profile,
        )
        self.reasoning_pipeline.record_synthesis(pipeline_trace, synthesis)

    def _attach_persistence_observation(
        self,
        *,
        response: dict[str, object],
        pipeline_trace,
        session_id: str,
        prompt: str,
        front_half,
        route,
        clarification_decision,
        validation_context,
        reasoning_frame_assembly,
    ) -> None:
        ResponsePackagingSupport.attach_persistence_observation(
            response=response,
            pipeline_trace=pipeline_trace,
            session_id=session_id,
            prompt=prompt,
            front_half=front_half,
            route=route,
            clarification_decision=clarification_decision,
            validation_context=validation_context,
            reasoning_frame_assembly=reasoning_frame_assembly,
            reasoning_pipeline=self.reasoning_pipeline,
        )

    def _attach_response_behavior_posture(
        self,
        *,
        response: dict[str, object],
        route_mode: str,
        low_confidence_recovery=None,
        srd_diagnostic=None,
        state_control=None,
    ) -> None:
        ResponseContextSupport.attach_response_behavior_posture(
            response=response,
            route_mode=route_mode,
            response_strategy_layer=self.response_strategy_layer,
            low_confidence_recovery=low_confidence_recovery,
            srd_diagnostic=srd_diagnostic,
            state_control=state_control,
        )

    def _finalize_user_facing_reasoning_response(
        self,
        *,
        response: dict[str, object],
        prompt: str,
        mode: str,
        route,
        interaction_profile,
    ) -> None:
        response["summary"] = self._strip_internal_response_labels(str(response.get("summary") or "").strip(), prompt=prompt)
        allow_internal_scaffold = bool(response.get("internal_scaffold_visible"))
        allow_validation_detail = str(getattr(interaction_profile, "reasoning_depth", "normal") or "normal") == "deep"
        if mode == "planning":
            response["steps"] = self._filter_user_facing_reasoning_lines(
                list(response.get("steps") or []),
                allow_internal_scaffold=allow_internal_scaffold,
                allow_validation_detail=allow_validation_detail,
            )
            response["next_action"] = self._clean_reasoning_closeout(
                self._strip_internal_response_labels(
                    str(response.get("next_action") or "").strip(),
                    prompt=prompt,
                )
            )
        elif mode == "research":
            response["findings"] = self._filter_user_facing_reasoning_lines(
                list(response.get("findings") or []),
                allow_internal_scaffold=allow_internal_scaffold,
                allow_validation_detail=allow_validation_detail,
            )
            if response.get("discussion_domain") == "belief_tradition":
                note = str(response.get("uncertainty_note") or "").strip()
                if note and note not in response["findings"]:
                    response["findings"] = self._prepend_unique_line(
                        list(response.get("findings") or []),
                        note,
                    )
            response["recommendation"] = self._clean_reasoning_closeout(
                self._strip_internal_response_labels(
                    str(response.get("recommendation") or "").strip(),
                    prompt=prompt,
                )
            )
        self._apply_low_support_response_surface(
            response=response,
            prompt=prompt,
            mode=mode,
        )
        self._ensure_visible_response_body(
            response=response,
            prompt=prompt,
            mode=mode,
        )
        self._scrub_internal_scaffold_surfaces(response=response, prompt=prompt)
        self._ensure_visible_response_body(
            response=response,
            prompt=prompt,
            mode=mode,
        )

    def _compose_reasoning_body(self, *, response: dict[str, object], mode: str, prompt: str = "") -> str:
        summary = str(response.get("summary") or "").strip()
        intro = str(response.get("response_intro") or "").strip()
        opening = str(response.get("response_opening") or "").strip()
        reply = str(response.get("reply") or "").strip()
        explanatory_body = str(response.get("explanatory_body") or "").strip()
        computed_intro = ResponseFlowRealizer.intro_for_response(response=response, mode=mode)
        lead = ""
        if summary and not self._is_intro_only_surface(summary):
            lead = summary
        elif reply and not self._is_intro_only_surface(reply):
            lead = reply
        elif explanatory_body and not self._is_intro_only_surface(explanatory_body):
            lead = explanatory_body
        elif opening and not computed_intro and not self._is_intro_only_surface(opening):
            lead = opening
        elif intro and not computed_intro and not self._is_intro_only_surface(intro):
            lead = intro

        lines: list[str] = []
        if computed_intro:
            lines.append(computed_intro)
        if lead:
            if not lines or lines[-1] != lead:
                lines.append(lead)

        if mode == "planning":
            steps = [str(item).strip() for item in response.get("steps") or [] if str(item).strip()]
            if steps:
                if lines:
                    lines.append("")
                lines.extend(f"- {item}" for item in steps)
            next_action = str(response.get("next_action") or "").strip()
            if next_action:
                if lines:
                    lines.append("")
                lines.append(f"{ResponseFlowRealizer.next_label_for_response(response=response, mode=mode)} {next_action}")
        elif mode == "research":
            findings = [str(item).strip() for item in response.get("findings") or [] if str(item).strip()]
            if findings:
                if lines:
                    lines.append("")
                lines.extend(f"- {item}" for item in findings)
            elif explanatory_body and explanatory_body not in lines:
                if lines:
                    lines.append("")
                lines.append(explanatory_body)
            recommendation = str(response.get("recommendation") or "").strip()
            if recommendation:
                if lines:
                    lines.append("")
                lines.append(f"{ResponseFlowRealizer.next_label_for_response(response=response, mode=mode)} {recommendation}")
        elif mode == "conversation":
            if explanatory_body and explanatory_body not in lines:
                if lines:
                    lines.append("")
                lines.append(explanatory_body)
            elif reply and reply not in lines and not self._is_intro_only_surface(reply):
                if lines:
                    lines.append("")
                lines.append(reply)

        body = "\n".join(lines).strip()
        if body:
            return body
        return self._fallback_visible_body(response=response, prompt=prompt, mode=mode)

    def _ensure_visible_response_body(
        self,
        *,
        response: dict[str, object],
        prompt: str,
        mode: str,
    ) -> None:
        memory_reply_hint = str(response.get("memory_reply_hint") or "").strip()
        if memory_reply_hint:
            response["user_facing_answer"] = memory_reply_hint
            response["summary"] = memory_reply_hint
            if mode in {"research", "planning", "conversation"}:
                response["reply"] = memory_reply_hint
            return
        existing_answer = str(response.get("user_facing_answer") or "").strip()
        if existing_answer and not self._is_intro_only_surface(existing_answer):
            return
        body = self._compose_reasoning_body(response=response, mode=mode, prompt=prompt)
        visible_text = self._visible_surface_text(response=response)
        if visible_text and not self._is_intro_only_surface(visible_text):
            if (
                body
                and body != visible_text
                and mode in {"planning", "research"}
                and self._has_structured_body_parts(response=response, mode=mode)
            ):
                response["reply"] = body
            if mode == "conversation" and not existing_answer:
                response["user_facing_answer"] = visible_text
            return
        if not body or self._is_intro_only_surface(body):
            return
        response["user_facing_answer"] = body
        response["summary"] = body
        if mode in {"research", "planning", "conversation"}:
            response["reply"] = body

    @staticmethod
    def _has_structured_body_parts(*, response: dict[str, object], mode: str) -> bool:
        if mode == "planning":
            if any(str(item).strip() for item in response.get("steps") or []):
                return True
            return bool(str(response.get("next_action") or "").strip())
        if mode == "research":
            if any(str(item).strip() for item in response.get("findings") or []):
                return True
            if str(response.get("recommendation") or "").strip():
                return True
            return bool(str(response.get("explanatory_body") or "").strip())
        if mode == "conversation":
            return bool(str(response.get("reply") or response.get("explanatory_body") or "").strip())
        return False

    def _fallback_visible_body(
        self,
        *,
        response: dict[str, object],
        prompt: str,
        mode: str,
    ) -> str:
        explanatory_body = str(response.get("explanatory_body") or "").strip()
        if explanatory_body and not self._is_intro_only_surface(explanatory_body):
            return explanatory_body
        reply = str(response.get("reply") or "").strip()
        if reply and not self._is_intro_only_surface(reply):
            return reply
        if mode in {"research", "conversation"}:
            topic = ""
            surface = response.get("domain_surface") if isinstance(response.get("domain_surface"), dict) else {}
            if isinstance(surface, dict):
                topic = str(surface.get("topic") or "").strip()
            lookup_target = topic or str(response.get("resolved_prompt") or prompt or "").strip()
            if self.knowledge_service is not None and lookup_target:
                lookup = self.knowledge_service.lookup(lookup_target)
                entry = lookup.primary if lookup is not None else None
                if entry is not None:
                    base = str(entry.summary_medium or entry.summary_short or entry.title or "").strip()
                    detail = str((entry.key_points[0] if entry.key_points else "") or "").strip()
                    if detail and detail.lower() not in base.lower():
                        return f"{base} {detail}".strip()
                    return base
            findings = [str(item).strip() for item in response.get("findings") or [] if str(item).strip()]
            if findings:
                return "\n".join(f"- {item}" for item in findings)
        if mode == "planning":
            steps = [str(item).strip() for item in response.get("steps") or [] if str(item).strip()]
            if steps:
                return "\n".join(f"- {item}" for item in steps)
        return ""

    @staticmethod
    def _apply_low_support_response_surface(
        *,
        response: dict[str, object],
        prompt: str,
        mode: str,
    ) -> None:
        if mode != "research":
            return
        answer = str(response.get("user_facing_answer") or "").strip()
        explanatory_body = str(response.get("explanatory_body") or "").strip()
        support_status = str(response.get("support_status") or "").strip().lower()
        route_status = str(response.get("route_status") or "").strip().lower()
        low_support_answer = ""
        if answer and InteractionService._looks_like_insufficient_answer(answer):
            low_support_answer = answer
        elif explanatory_body and InteractionService._looks_like_insufficient_answer(explanatory_body):
            low_support_answer = explanatory_body
        if not low_support_answer:
            return
        if support_status not in {"insufficiently_grounded", "moderately_supported"} and route_status not in {
            "under_tension",
            "weakened",
            "unresolved",
        }:
            return
        response["user_facing_answer"] = low_support_answer
        response["summary"] = low_support_answer
        response["reply"] = low_support_answer
        response["explanatory_body"] = low_support_answer
        response["findings"] = []
        response.pop("recommendation", None)
        response.pop("conversation_turn", None)
        response.pop("response_intro", None)
        response.pop("response_opening", None)
        response["internal_scaffold_visible"] = False

    @staticmethod
    def _visible_surface_text(*, response: dict[str, object]) -> str:
        for key in ("user_facing_answer", "reply", "explanatory_body", "summary"):
            value = str(response.get(key) or "").strip()
            if value:
                return value
        return ""

    @staticmethod
    def _is_intro_only_surface(text: str) -> bool:
        normalized = " ".join(str(text or "").replace("\u2019", "'").strip().lower().split())
        if not normalized:
            return False
        if normalized in {
            "here's a first pass.",
            "here is a first pass.",
            "here's a workable answer.",
            "here is a workable answer.",
            "here's a solid plan.",
            "here is a solid plan.",
            "here's the clearest current answer.",
            "here is the clearest current answer.",
            "here's a grounded answer.",
            "here is a grounded answer.",
            "here's the clearest answer.",
            "here is the clearest answer.",
            "here's the grounded explanation.",
            "here is the grounded explanation.",
            "here's a useful first plan, with the assumptions kept visible.",
            "here is a useful first plan, with the assumptions kept visible.",
            "here's a grounded answer using the best current assumptions.",
            "here is a grounded answer using the best current assumptions.",
            "here's a grounded answer, with the assumptions kept visible.",
            "here is a grounded answer, with the assumptions kept visible.",
        }:
            return True
        return (
            normalized.startswith("here's a first pass")
            or normalized.startswith("here is a first pass")
            or normalized.startswith("here's a workable answer")
            or normalized.startswith("here is a workable answer")
            or normalized.startswith("here's a solid plan")
            or normalized.startswith("here is a solid plan")
            or normalized.startswith("here's the clearest current answer")
            or normalized.startswith("here is the clearest current answer")
            or normalized.startswith("here's a grounded answer")
            or normalized.startswith("here is a grounded answer")
            or normalized.startswith("here's the clearest answer")
            or normalized.startswith("here is the clearest answer")
            or normalized.startswith("here's the grounded explanation")
            or normalized.startswith("here is the grounded explanation")
            or normalized.startswith("here's a useful first plan")
            or normalized.startswith("here is a useful first plan")
        )

    @staticmethod
    def _realize_conversational_reply_surface(
        *,
        response: dict[str, object],
        interaction_profile,
        recent_interactions: list[dict[str, object]] | None,
        active_thread: dict[str, object] | None = None,
    ) -> None:
        state = ConversationalReplyRealizer.build_state(
            response=response,
            interaction_profile=interaction_profile,
            recent_interactions=recent_interactions,
            active_thread=active_thread,
        )
        if state is None:
            return
        realized = ConversationalReplyRealizer.realize(
            state=state,
            interaction_profile=interaction_profile,
            recent_interactions=recent_interactions,
        )
        if not realized:
            return
        response["conversational_reply_state"] = state.to_dict()
        response["conversation_voice"] = {
            **InteractionStylePolicy.voice_profile(interaction_profile),
            "style_mode": InteractionStylePolicy.interaction_style(interaction_profile),
            "reasoning_depth": InteractionStylePolicy.reasoning_depth(interaction_profile),
            "reasoning_depth_separate": True,
        }
        response["summary"] = realized
        response["reply"] = realized

    @staticmethod
    def _attach_memory_retrieval(
        *,
        response: dict[str, object],
        memory_retrieval: MemoryRetrievalResult | None,
    ) -> None:
        MemoryResponseSupport.attach_memory_retrieval(
            response=response,
            memory_retrieval=memory_retrieval,
        )

    @staticmethod
    def _apply_memory_recall_surface(
        *,
        response: dict[str, object],
        memory_retrieval: MemoryRetrievalResult | None,
    ) -> None:
        # Memory may shape the visible surface, but it cannot choose route or lane.
        MemoryResponseSupport.apply_memory_recall_surface(
            response=response,
            memory_retrieval=memory_retrieval,
        )

    @staticmethod
    def _enforce_final_surface_lane(
        *,
        response: dict[str, object],
        selected_mode: str,
    ) -> None:
        lane = InteractionService._determine_final_surface_lane(
            response=response,
            selected_mode=selected_mode,
        )
        changed = False

        if lane == "conversational":
            for key in ("steps", "findings", "recommendation", "next_action", "user_facing_answer"):
                if response.pop(key, None) is not None:
                    changed = True
            response["internal_scaffold_visible"] = False
        elif lane in {"answer", "fallback"}:
            if lane == "fallback":
                answer = str(
                    response.get("explanatory_body")
                    or response.get("user_facing_answer")
                    or response.get("summary")
                    or ""
                ).strip()
            else:
                answer = str(response.get("user_facing_answer") or response.get("summary") or "").strip()
            if answer:
                response["summary"] = answer
                response["reply"] = answer
                response["user_facing_answer"] = answer
                changed = True
            for key in ("conversation_turn", "response_intro", "response_opening", "recommendation", "next_action"):
                if response.pop(key, None) is not None:
                    changed = True
            if isinstance(response.get("steps"), list) and response["steps"]:
                response["steps"] = InteractionService._filter_cross_lane_lines(
                    list(response.get("steps") or []),
                    blocked_fragments=InteractionService._conversation_lane_fragments(),
                )
                changed = True
            if isinstance(response.get("findings"), list):
                if response["findings"]:
                    response["findings"] = []
                    changed = True
            response["internal_scaffold_visible"] = False
        elif lane in {"planning", "research"}:
            key = "steps" if lane == "planning" else "findings"
            existing = list(response.get(key) or [])
            filtered = InteractionService._filter_cross_lane_lines(
                existing,
                blocked_fragments=InteractionService._conversation_lane_fragments(),
            )
            if filtered != existing:
                response[key] = filtered
                changed = True
            if lane == "planning":
                recommendation = str(response.get("next_action") or "").strip()
                if recommendation and InteractionService._contains_cross_lane_fragment(
                    recommendation,
                    blocked_fragments=InteractionService._conversation_lane_fragments(),
                ):
                    response.pop("next_action", None)
                    changed = True
            else:
                recommendation = str(response.get("recommendation") or "").strip()
                if recommendation and InteractionService._contains_cross_lane_fragment(
                    recommendation,
                    blocked_fragments=InteractionService._conversation_lane_fragments(),
                ):
                    response.pop("recommendation", None)
                    changed = True

        response["final_surface_lane"] = lane
        if changed:
            response["lane_enforcement"] = {
                "selected_lane": lane,
                "surface_changed": True,
            }

    @staticmethod
    def _determine_final_surface_lane(
        *,
        response: dict[str, object],
        selected_mode: str,
    ) -> str:
        normalized_mode = str(selected_mode or response.get("mode") or "").strip()
        if normalized_mode == "conversation":
            return "conversational"
        if normalized_mode == "planning":
            return "planning"
        if normalized_mode != "research":
            return normalized_mode or "unknown"
        safety_decision = response.get("safety_decision") if isinstance(response.get("safety_decision"), dict) else {}
        if str(safety_decision.get("tier") or "").strip().lower() == "dual_use":
            return "research"
        user_facing_answer = str(response.get("user_facing_answer") or "").strip()
        explanatory_body = str(response.get("explanatory_body") or "").strip()
        summary = str(response.get("summary") or "").strip()
        for candidate in (user_facing_answer, explanatory_body, summary):
            if candidate and InteractionService._looks_like_insufficient_answer(candidate):
                return "fallback"
        if InteractionService._has_substantive_research_structure(response=response):
            return "research"
        if user_facing_answer:
            return "answer"
        return "research"

    @staticmethod
    def _has_substantive_research_structure(*, response: dict[str, object]) -> bool:
        user_facing_answer = str(response.get("user_facing_answer") or "").strip()
        summary = str(response.get("summary") or "").strip()
        blocked_fragments = InteractionService._conversation_lane_fragments()
        findings = [str(item).strip() for item in response.get("findings") or [] if str(item).strip()]
        for item in findings:
            normalized = " ".join(item.replace("\u2019", "'").strip().lower().split())
            if not normalized:
                continue
            if normalized in {"best read:", "answer:", "why:", "action:"}:
                continue
            if InteractionService._is_intro_only_surface(item):
                continue
            if InteractionService._contains_cross_lane_fragment(item, blocked_fragments=blocked_fragments):
                continue
            if user_facing_answer and normalized == " ".join(user_facing_answer.replace("\u2019", "'").strip().lower().split()):
                continue
            if summary and normalized == " ".join(summary.replace("\u2019", "'").strip().lower().split()):
                continue
            return True
        recommendation = str(response.get("recommendation") or "").strip()
        if recommendation and not InteractionService._contains_cross_lane_fragment(
            recommendation,
            blocked_fragments=blocked_fragments,
        ):
            return True
        return False

    @staticmethod
    def _looks_like_insufficient_answer(text: str) -> bool:
        normalized = " ".join(str(text or "").strip().lower().split())
        fallback_markers = (
            "i don't have enough grounded detail",
            "i dont have enough grounded detail",
            "i don't have enough grounded local detail",
            "i dont have enough grounded local detail",
            "i don't have enough local knowledge on that yet",
            "i dont have enough local knowledge on that yet",
            "i don't have enough local knowledge on it yet",
            "i dont have enough local knowledge on it yet",
            "i don't have enough local knowledge on them yet",
            "i dont have enough local knowledge on them yet",
            "i don't have enough local knowledge on",
            "i dont have enough local knowledge on",
            "i don't have enough detail",
            "i dont have enough detail",
            "i don't have a strong enough",
            "i dont have a strong enough",
            "i can identify",
            "i can tell the subject points to",
            "that's something i could explain, but",
        )
        return any(marker in normalized for marker in fallback_markers)

    @staticmethod
    def _conversation_lane_fragments() -> tuple[str, ...]:
        return (
            "want to keep going together",
            "what are we looking at",
            "what are we working on",
            "how's it going",
            "how are you",
            "what do you want to dig into",
            "good to have you back",
            "glad to see you",
            "i'm here.",
            "im here.",
            "still here.",
        )

    @staticmethod
    def _filter_cross_lane_lines(
        lines: list[str],
        *,
        blocked_fragments: tuple[str, ...],
    ) -> list[str]:
        filtered = [
            line
            for line in lines
            if str(line).strip()
            and not InteractionService._contains_cross_lane_fragment(
                str(line),
                blocked_fragments=blocked_fragments,
            )
        ]
        return filtered or lines[:1]

    @staticmethod
    def _contains_cross_lane_fragment(
        text: str,
        *,
        blocked_fragments: tuple[str, ...],
    ) -> bool:
        normalized = " ".join(str(text or "").strip().lower().split())
        return any(fragment in normalized for fragment in blocked_fragments)

    @staticmethod
    def _strip_internal_response_labels(text: str, *, prompt: str) -> str:
        if not text:
            return text
        lowered = text.lower()
        prefixes = (
            "strongly grounded planning response for:",
            "planning response with strong grounding for:",
            "grounded planning response for:",
            "planning response grounded in local evidence for:",
            "conflicted planning response for:",
            "planning response under tension for:",
            "tentative planning response for:",
            "provisional planning response for:",
            "strongly grounded research response for:",
            "research response with strong grounding for:",
            "grounded research response for:",
            "research response grounded in local evidence for:",
            "conflicted research response for:",
            "research response under tension for:",
            "tentative research response for:",
            "provisional research response for:",
        )
        for prefix in prefixes:
            if lowered.startswith(prefix):
                return text[len(prefix) :].strip(" :.-") or prompt
        return text

    @classmethod
    def _scrub_internal_scaffold_surfaces(cls, *, response: dict[str, object], prompt: str) -> None:
        for key in ("summary", "reply", "user_facing_answer", "explanatory_body", "recommendation", "next_action"):
            if key in response:
                response[key] = cls._scrub_internal_scaffold_text(str(response.get(key) or ""), prompt=prompt)
        for key in ("findings", "steps"):
            values = response.get(key)
            if isinstance(values, list):
                response[key] = [
                    cleaned
                    for item in values
                    if (cleaned := cls._scrub_internal_scaffold_text(str(item or ""), prompt=prompt))
                ]

    @staticmethod
    def _scrub_internal_scaffold_text(text: str, *, prompt: str) -> str:
        cleaned = str(text or "").strip()
        if not cleaned:
            return ""
        compact = " ".join(cleaned.split())
        lowered = compact.replace("’", "'").lower()
        hard_scaffold = (
            "here's the best first read",
            "here is the best first read",
            "best first read",
            "best current assumptions",
            "hold it provisionally",
            "best next check",
            "validate the route choice",
            "use the strongest local context",
        )
        if not any(fragment in lowered for fragment in hard_scaffold):
            return cleaned
        split_markers = ("Best next check:", "best next check:", "Recommendation:", "recommendation:")
        for marker in split_markers:
            if marker in cleaned:
                tail = cleaned.split(marker, 1)[1].strip(" :-")
                if tail and not any(fragment in tail.lower() for fragment in hard_scaffold):
                    return tail
        return ""

    @staticmethod
    def _filter_user_facing_reasoning_lines(
        lines: list[str],
        *,
        allow_internal_scaffold: bool,
        allow_validation_detail: bool,
    ) -> list[str]:
        if allow_internal_scaffold:
            return lines
        blocked_prefixes = (
            "treat the first milestone as",
            "treat the first conclusion as",
            "use this aligned anchor",
            "the strongest local evidence sources reinforce",
            "promote that shared topic",
            "promote the shared signal",
            "call out this local tension before",
            "keep the first conclusion narrow until",
            "keep the opening milestone narrow until",
            "route caution:",
            "state the topic in one concise sentence",
            "highlight the most relevant local evidence first",
            "close with the clearest next action or conclusion",
            "identify the main topic and any technical constraints",
            "summarize the most relevant local context before taking action",
            "recommend the next concrete validation or implementation step",
        )
        blocked_fragments = (
            "keep the first conclusion close to the strongest local signal",
            "avoid broad extrapolation",
            "validate it against the closest local evidence first",
            "what would count as a convincing next step here",
            "best next move is to test the weak assumption directly",
            "keep one local validation step before acting on it",
            "validate the route choice itself against the closest prior session prompt",
            "keep the active thread question in frame while interpreting the result",
        )
        if not allow_validation_detail:
            blocked_prefixes = (
                *blocked_prefixes,
                "validation plan:",
                "deep thinking pass:",
            )
        filtered = [
            line
            for line in lines
            if str(line).strip()
            and not any(str(line).strip().lower().startswith(prefix) for prefix in blocked_prefixes)
            and not any(fragment in str(line).strip().lower() for fragment in blocked_fragments)
        ]
        return filtered or lines[:1]

    @staticmethod
    def _clean_reasoning_closeout(text: str) -> str:
        cleaned = " ".join(str(text or "").strip().split()).strip()
        if not cleaned:
            return ""
        blocked_fragments = (
            "resolve the tension between archived evidence and prior session context first",
            "first confirm the assumptions with an additional local check before committing to the plan",
            "keep the closest archive run as the baseline reference",
            "keep one local validation step before acting on it",
            "validate the route choice itself against the closest prior session prompt",
            "keep the active thread question in frame while interpreting the result",
            "explicitly resolve the mismatch between archived evidence and prior session context",
            "treat this as provisional until another local source or run confirms it",
            "start with the closest archive run",
        )
        for fragment in blocked_fragments:
            index = cleaned.lower().find(fragment)
            if index > 0:
                cleaned = cleaned[:index].rstrip(" .")
                break
        if cleaned.startswith("Next move: Next move:"):
            cleaned = cleaned.replace("Next move: Next move:", "Next move:", 1).strip()
        if cleaned and not cleaned.endswith((".", "!", "?")):
            cleaned = f"{cleaned}."
        return cleaned

    @staticmethod
    def _attach_dialogue_context(
        *,
        response: dict[str, object],
        vibe_catcher=None,
        dialogue_management,
        conversation_awareness=None,
        low_confidence_recovery=None,
        srd_diagnostic=None,
        empathy_model=None,
        human_language_layer=None,
        state_control=None,
        thought_framing=None,
    ) -> None:
        if dialogue_management is None:
            return
        if vibe_catcher is not None:
            response["vibe_catcher"] = vibe_catcher.to_dict()
        response["dialogue_management"] = dialogue_management.to_dict()
        if conversation_awareness is not None:
            response["conversation_awareness"] = conversation_awareness.to_dict()
        if low_confidence_recovery is not None:
            response["low_confidence_recovery"] = low_confidence_recovery.to_dict()
        if srd_diagnostic is not None:
            response["srd_diagnostic"] = srd_diagnostic.to_dict()
        if empathy_model is not None:
            response["empathy_model"] = empathy_model.to_dict()
        if human_language_layer is not None:
            response["human_language_layer"] = human_language_layer.to_dict()
        response["interaction_mode"] = dialogue_management.interaction_mode
        response["idea_state"] = dialogue_management.idea_state
        response["response_strategy"] = dialogue_management.response_strategy
        if state_control is not None:
            response["state_control"] = state_control.to_dict()
        if thought_framing is not None:
            thought_framing_payload = thought_framing.to_dict()
            response["thought_framing"] = thought_framing_payload
            if thought_framing_payload.get("research_questions"):
                response["research_questions"] = list(
                    thought_framing_payload.get("research_questions") or []
                )
            if thought_framing_payload.get("branch_return_hint"):
                response["branch_return_hint"] = str(thought_framing_payload.get("branch_return_hint") or "")
            if thought_framing_payload.get("checkpoint_summary") is not None:
                response["checkpoint_summary"] = dict(
                    thought_framing_payload.get("checkpoint_summary") or {}
                )

    @staticmethod
    def _attach_conversational_turn(
        *,
        prompt: str = "",
        response: dict[str, object],
        interaction_profile,
        dialogue_management,
        conversation_awareness=None,
        state_control=None,
        human_language_layer=None,
        thought_framing,
        recent_interactions: list[dict[str, object]] | None = None,
        tone_profile: str = "default",
        allow_internal_scaffold: bool = False,
    ) -> None:
        if dialogue_management is None or thought_framing is None:
            return
        style = InteractionService._interaction_style(interaction_profile)
        interaction_mode = str(getattr(dialogue_management, "interaction_mode", "") or "")
        live_thread_signal = (
            bool(getattr(conversation_awareness, "unresolved_thread_open", False))
            or bool(str(getattr(conversation_awareness, "live_unresolved_question", "") or "").strip())
            or bool(str(getattr(conversation_awareness, "return_target", "") or "").strip())
        )
        if interaction_mode == "social" and not live_thread_signal:
            return
        thought_payload = thought_framing.to_dict()
        research_questions = list(thought_payload.get("research_questions") or [])
        checkpoint_summary = thought_payload.get("checkpoint_summary")
        response_kind_label = str(thought_payload.get("response_kind_label") or "").strip()
        strategy = str(getattr(dialogue_management, "response_strategy", "") or "")
        anti_spiral_active = bool(getattr(state_control, "anti_spiral_active", False)) if state_control is not None else False
        state_core = str(getattr(state_control, "core_state", "") or "").strip() if state_control is not None else ""
        emotional_alignment = str(getattr(human_language_layer, "emotional_alignment", "") or "").strip() if human_language_layer is not None else ""
        correction_detected = bool(getattr(human_language_layer, "correction_detected", False)) if human_language_layer is not None else False
        epistemic_stance = str(getattr(human_language_layer, "epistemic_stance", "") or "").strip() if human_language_layer is not None else ""
        stance_confidence = str(getattr(human_language_layer, "stance_confidence", "") or "").strip() if human_language_layer is not None else ""
        user_energy = str(getattr(human_language_layer, "user_energy", "") or "").strip() if human_language_layer is not None else ""
        support_status = str(response.get("support_status") or "")
        tension_status = str(response.get("tension_status") or "")
        route_status = str(response.get("route_status") or "")
        reasoning_depth = str(getattr(interaction_profile, "reasoning_depth", "normal") or "normal")
        thread_holding = style in {"default", "collab", "direct"}
        deep_collaboration = reasoning_depth == "deep"
        recent_texts = ResponseVariationLayer.recent_surface_texts(recent_interactions or [])
        turn: dict[str, object] | None = None
        if checkpoint_summary and (allow_internal_scaffold or response_kind_label == "thread_reorientation"):
            turn = {
                "kind": "checkpoint",
                "lead": ResponseToneEngine.checkpoint_turn_lead(
                    style=style,
                    state_core=state_core,
                    anti_spiral_active=anti_spiral_active,
                    recent_texts=recent_texts,
                ),
            }
            if allow_internal_scaffold:
                turn["current_direction"] = checkpoint_summary.get("current_direction")
                turn["strongest_point"] = checkpoint_summary.get("strongest_point")
                turn["weakest_point"] = checkpoint_summary.get("weakest_point")
                turn["open_questions"] = list(checkpoint_summary.get("open_questions") or [])
                turn["next_step"] = checkpoint_summary.get("next_step")
        elif strategy == "ask_question" and research_questions:
            turn = {
                "kind": "question",
                "lead": ResponseToneEngine.question_turn_lead(
                    research_questions[0],
                    style=style,
                    adaptive_posture=str(getattr(conversation_awareness, "adaptive_posture", "") or ""),
                    anti_spiral_active=anti_spiral_active,
                    emotional_alignment=emotional_alignment,
                    correction_detected=correction_detected,
                    epistemic_stance=epistemic_stance,
                    stance_confidence=stance_confidence,
                    recent_texts=recent_texts,
                ),
                "follow_ups": research_questions[1:3],
            }
        elif strategy == "challenge" or InteractionService._should_surface_supportive_challenge(
            strategy=strategy,
            support_status=support_status,
            tension_status=tension_status,
            route_status=route_status,
            adaptive_posture=str(getattr(conversation_awareness, "adaptive_posture", "") or ""),
            recent_intent_pattern=str(getattr(conversation_awareness, "recent_intent_pattern", "") or ""),
            interaction_mode=interaction_mode,
        ):
            lead = research_questions[0] if research_questions else "What would tell us this line doesn't hold?"
            turn = {
                "kind": "challenge",
                "lead": ResponseToneEngine.challenge_turn_lead(
                    lead,
                    style=style,
                    adaptive_posture=str(getattr(conversation_awareness, "adaptive_posture", "") or ""),
                    anti_spiral_active=anti_spiral_active,
                    emotional_alignment=emotional_alignment,
                    correction_detected=correction_detected,
                    epistemic_stance=epistemic_stance,
                    recent_texts=recent_texts,
                ),
                "follow_ups": research_questions[1:3],
            }
        elif strategy == "expand":
            turn = {
                "kind": "explore",
                "lead": "Worth exploring:" if style == "direct" else "There’s something worth exploring here.",
                "follow_ups": research_questions[:2],
            }
        elif (strategy == "answer" or (strategy == "summarize" and not allow_internal_scaffold)) and thread_holding:
            carried_question = str(getattr(conversation_awareness, "live_unresolved_question", "") or "").strip()
            answer_follow_ups = list(research_questions[:2])
            if carried_question and carried_question not in answer_follow_ups:
                answer_follow_ups = [carried_question, *answer_follow_ups[:1]]
            turn = {
                "kind": "collaborate" if style in {"default", "collab"} else "thread_hold",
                "lead": ResponseToneEngine.answer_turn_lead(
                    style=style,
                    deep_collaboration=deep_collaboration,
                    adaptive_posture=str(getattr(conversation_awareness, "adaptive_posture", "") or ""),
                    anti_spiral_active=anti_spiral_active,
                    tone_profile=tone_profile,
                    emotional_alignment=emotional_alignment,
                    user_energy=user_energy,
                    epistemic_stance=epistemic_stance,
                    recent_texts=recent_texts,
                ),
                "follow_ups": answer_follow_ups,
            }
        elif strategy == "summarize":
            turn = {
                "kind": "summary",
                "lead": "Current shape:" if style == "direct" else "Here's the shape of it so far.",
                "follow_ups": research_questions[:2],
            }
        pickup_bridge = InteractionService._response_to_response_bridge(
            prompt=prompt,
            recent_interactions=recent_interactions or [],
        )
        if turn is not None and pickup_bridge is not None:
            target = str(pickup_bridge.get("target") or "").strip()
            category = str(pickup_bridge.get("category") or "collaborative_pickup").strip()
            turn["pickup_bridge"] = ResponseToneEngine.pickup_bridge(
                style=style,
                category=category,
                target=target,
                recent_texts=recent_texts,
            )
            starter = ResponseToneEngine.follow_through_starter(
                style=style,
                target=target,
                recent_texts=recent_texts,
            )
            if starter:
                turn["follow_through_starter"] = starter
            turn["response_to_response_handoff"] = True
            turn["handoff_target"] = target
        if turn is not None and thread_holding:
            turn["adaptive_posture"] = str(getattr(conversation_awareness, "adaptive_posture", "") or "")
            turn["conversation_momentum"] = str(getattr(conversation_awareness, "conversation_momentum", "") or "")
            turn["unresolved_thread_open"] = bool(getattr(conversation_awareness, "unresolved_thread_open", False))
            branch_state = str(getattr(conversation_awareness, "branch_state", "") or "").strip()
            if branch_state:
                turn["branch_state"] = branch_state
            return_target = str(getattr(conversation_awareness, "return_target", "") or "").strip()
            if return_target:
                turn["return_target"] = return_target
            unresolved_reason = str(getattr(conversation_awareness, "unresolved_thread_reason", "") or "").strip()
            if unresolved_reason:
                turn["unresolved_thread_reason"] = unresolved_reason
            branch_return_hint = str(thought_payload.get("branch_return_hint") or "").strip()
            if branch_return_hint:
                turn["branch_return_hint"] = branch_return_hint
            if state_core:
                turn["state_core"] = state_core
            if anti_spiral_active:
                turn["anti_spiral_active"] = True
            turn["partner_frame"] = ResponseToneEngine.thread_holding_frame(
                style=style,
                deep_collaboration=deep_collaboration,
                strategy=strategy,
                interaction_mode=interaction_mode,
                adaptive_posture=str(getattr(conversation_awareness, "adaptive_posture", "") or ""),
                unresolved_thread_open=bool(getattr(conversation_awareness, "unresolved_thread_open", False)),
                branch_state=str(getattr(conversation_awareness, "branch_state", "") or ""),
                return_target=str(getattr(conversation_awareness, "return_target", "") or ""),
                checkpoint_summary=checkpoint_summary,
                anti_spiral_active=anti_spiral_active,
                tone_profile=tone_profile,
                emotional_alignment=emotional_alignment,
                user_energy=user_energy,
                recent_texts=recent_texts,
            )
            turn["next_move"] = ResponseToneEngine.thread_holding_next_move(
                style=style,
                turn=turn,
                checkpoint_summary=checkpoint_summary,
                recent_texts=recent_texts,
            )
        if turn is not None:
            InteractionService._finalize_conversational_surface(
                response=response,
                turn=turn,
                prompt=prompt,
                interaction_profile=interaction_profile,
                conversation_awareness=conversation_awareness,
                human_language_layer=human_language_layer,
                recent_interactions=recent_interactions,
                strategy=strategy,
                support_status=support_status,
                tension_status=tension_status,
                route_status=route_status,
            )

    @staticmethod
    def _should_surface_supportive_challenge(
        *,
        strategy: str,
        support_status: str,
        tension_status: str,
        route_status: str,
        adaptive_posture: str,
        recent_intent_pattern: str,
        interaction_mode: str,
    ) -> bool:
        if strategy not in {"answer", "expand"}:
            return False
        if adaptive_posture == "step_back" or recent_intent_pattern == "hesitating":
            return False
        if interaction_mode == "clarification":
            return False
        if support_status == "insufficiently_grounded":
            return True
        if tension_status == "unresolved":
            return True
        return tension_status == "under_tension" and route_status in {"weakened", "unresolved"}

    @staticmethod
    def _allow_internal_scaffold(prompt: str) -> bool:
        normalized = PromptSurfaceBuilder.build(prompt).lookup_ready_text
        scaffold_triggers = (
            "where are we now",
            "where were we",
            "what were we doing",
            "what still matters here",
            "what is still unresolved",
            "what's still unresolved",
            "checkpoint",
            "summarize where we are",
            "summary of where we are",
            "debug",
            "validate",
            "validation",
        )
        return any(trigger in normalized for trigger in scaffold_triggers)

    @staticmethod
    def _response_to_response_bridge(
        *,
        prompt: str,
        recent_interactions: list[dict[str, object]],
    ) -> dict[str, str] | None:
        return InteractionFlowSupport.response_to_response_bridge(
            prompt=prompt,
            recent_interactions=recent_interactions,
        )

    @staticmethod
    def _clarification_planning_continuation(
        *,
        prompt: str,
        recent_interactions: list[dict[str, object]],
        active_thread: dict[str, object] | None,
    ) -> dict[str, object] | None:
        return InteractionFlowSupport.clarification_planning_continuation(
            prompt=prompt,
            recent_interactions=recent_interactions,
            active_thread=active_thread,
        )

    @staticmethod
    def _clarification_decline_reply(
        *,
        interaction_profile,
        reasoning_state: ReasoningStateFrame | dict[str, object] | None = None,
    ) -> str:
        style = InteractionService._reasoning_style(
            reasoning_state=reasoning_state,
            interaction_profile=interaction_profile,
        )
        if style == "direct":
            return "Okay. State the next direction."
        if style == "collab":
            return "Okay. What direction do you want to explore together instead?"
        return "Okay. Tell me which direction you want to take next."

    @staticmethod
    def _clarification_question_for_style(*, style: str) -> str:
        if style == "collab":
            return "I’m noticing we’ve got a couple directions we could go here. Do you want to explore this together, or focus on something specific?"
        if style == "direct":
            return "Ambiguity detected. Choose: summary | comparison | continue."
        return "This could go a couple ways. Do you want a quick explanation, a comparison, or to continue the current route?"

    @staticmethod
    def _route_choice_clarification_response(
        *,
        prompt: str,
        interaction_profile,
        reasoning_state: ReasoningStateFrame | dict[str, object] | None,
    ) -> dict[str, object] | None:
        normalized_prompt = PromptSurfaceBuilder.build(prompt).route_ready_text
        if not InteractionFlowSupport.is_route_choice_prompt(normalized_prompt):
            return None
        normalized_state = (
            reasoning_state
            if isinstance(reasoning_state, ReasoningStateFrame)
            else ReasoningStateFrame.from_mapping(reasoning_state if isinstance(reasoning_state, dict) else None)
        )
        updated_state = normalized_state.with_updates(
            pending_followup={
                "type": "clarification",
                "action": "route_choice_prompt",
                "route_mode": "research",
                "route_kind": "research.summary",
                "resolved_prompt": normalized_prompt or prompt,
            },
            ambiguity_status="clarification_required",
            turn_status="clarifying",
        )
        style = InteractionService._reasoning_style(
            reasoning_state=updated_state,
            interaction_profile=interaction_profile,
        )
        response = ClarificationResponse(
            mode="clarification",
            kind="clarification.request",
            summary=f"Clarification requested for: {normalized_prompt or prompt}",
            clarification_question=InteractionService._clarification_question_for_style(style=style),
            options=["Summary", "Comparison", "Continue"],
            clarification_context={
                "clarification_count": 0,
                "clarification_trigger": "route_choice_prompt",
                "suggested_route": {
                    "mode": "research",
                    "kind": "research.summary",
                    "resolved_prompt": normalized_prompt or prompt,
                },
            },
        ).to_dict()
        response["resolved_prompt"] = normalized_prompt or prompt
        response["resolution_strategy"] = "route_choice_prompt"
        response["resolution_reason"] = "Converted a route-choice prompt into an explicit clarification surface."
        response["reasoning_state"] = updated_state.to_dict()
        return response

    @staticmethod
    def _clarification_option_label(*, mode: str, kind: str) -> str:
        normalized_mode = str(mode or "").strip()
        normalized_kind = str(kind or "").strip().lower()
        if normalized_mode == "planning":
            return "Continue"
        if "comparison" in normalized_kind or normalized_kind.endswith(".comparison"):
            return "Comparison"
        return "Summary"

    @staticmethod
    def _extract_clarified_route_info(
        *,
        latest: dict[str, object],
        response: dict[str, object],
    ) -> dict[str, object] | None:
        return InteractionFlowSupport.extract_clarified_route_info(
            latest=latest,
            response=response,
        )

    @staticmethod
    def _is_planning_clarification_confirmation(*, normalized_prompt: str, route_kind: str) -> bool:
        return InteractionFlowSupport.is_planning_clarification_confirmation(
            normalized_prompt=normalized_prompt,
            route_kind=route_kind,
        )

    @staticmethod
    def _is_pure_clarification_confirmation(normalized_prompt: str) -> bool:
        return InteractionFlowSupport.is_pure_clarification_confirmation(normalized_prompt)

    @staticmethod
    def _memory_save_request(
        *,
        prompt: str,
        recent_interactions: list[dict[str, object]],
    ) -> dict[str, object] | None:
        return InteractionFlowSupport.memory_save_request(prompt, recent_interactions)

    @staticmethod
    def _memory_save_clarification_continuation(
        *,
        prompt: str,
        recent_interactions: list[dict[str, object]],
    ) -> dict[str, object] | None:
        return InteractionFlowSupport.memory_save_continuation(prompt, recent_interactions)

    @staticmethod
    def _shape_reasoning_body_from_conversation_turn(
        *,
        response: dict[str, object],
        interaction_profile,
        allow_internal_scaffold: bool,
    ) -> None:
        InteractionConversationSupport.shape_reasoning_body_from_turn(
            response=response,
            interaction_style=InteractionService._interaction_style(interaction_profile),
            allow_internal_scaffold=allow_internal_scaffold,
        )

    @staticmethod
    def _prepend_conversation_items(
        *,
        existing_items: list[str],
        intro: str,
        opening: str,
        turn: dict[str, object],
        style: str,
        allow_internal_scaffold: bool,
        limit: int | None = None,
    ) -> list[str]:
        return InteractionConversationSupport.prepend_conversation_items(
            existing_items=existing_items,
            intro=intro,
            opening=opening,
            turn=turn,
            style=style,
            allow_internal_scaffold=allow_internal_scaffold,
            limit=limit,
        )

    @staticmethod
    def _turn_body_items(
        *,
        turn: dict[str, object],
        existing_items: list[str],
        style: str,
    ) -> list[str]:
        return InteractionConversationSupport.turn_body_items(
            turn=turn,
            existing_items=existing_items,
            style=style,
        )

    @staticmethod
    def _shape_closeout_from_conversation_turn(
        *,
        existing_closeout: str,
        turn: dict[str, object],
        style: str,
        label: str,
    ) -> str:
        return InteractionConversationSupport.shape_closeout_from_turn(
            existing_closeout=existing_closeout,
            turn=turn,
            style=style,
            label=label,
        )

    @staticmethod
    def _finalize_conversational_surface(
        *,
        response: dict[str, object],
        turn: dict[str, object],
        prompt: str,
        interaction_profile,
        conversation_awareness,
        human_language_layer,
        recent_interactions: list[dict[str, object]] | None,
        strategy: str,
        support_status: str = "",
        tension_status: str = "",
        route_status: str = "",
    ) -> None:
        stance_consistency = InteractionService._stance_consistency_for_turn(
            prompt=prompt,
            turn=turn,
            interaction_profile=interaction_profile,
            conversation_awareness=conversation_awareness,
            human_language_layer=human_language_layer,
            recent_interactions=recent_interactions,
            strategy=strategy,
            support_status=support_status,
            tension_status=tension_status,
            route_status=route_status,
        )
        if stance_consistency is not None:
            turn["lead"] = stance_consistency.applied_lead or str(turn.get("lead") or "")
            turn["stance_category"] = stance_consistency.category
            if stance_consistency.continuity_note:
                turn["stance_continuity_note"] = stance_consistency.continuity_note
            response["stance_consistency"] = stance_consistency.to_dict()
        response["conversation_turn"] = turn
        response["response_intro"] = str(turn.get("lead") or "").strip() or None
        response_opening = ResponseToneEngine.response_body_opening(
            turn=turn,
            style=InteractionService._interaction_style(interaction_profile),
        )
        if response_opening:
            response["response_opening"] = response_opening

    @staticmethod
    def _stance_consistency_for_turn(
        *,
        prompt: str,
        turn: dict[str, object],
        interaction_profile,
        conversation_awareness,
        human_language_layer,
        recent_interactions: list[dict[str, object]] | None,
        strategy: str,
        support_status: str,
        tension_status: str,
        route_status: str,
    ):
        if strategy not in {"ask_question", "challenge", "answer"}:
            return None
        recent_pattern = str(getattr(conversation_awareness, "recent_intent_pattern", "") or "").strip()
        if recent_pattern not in {"agreeing", "disagreeing", "hesitating"} and not recent_interactions:
            return None
        stance_consistency = StanceConsistencyLayer.assess(
            prompt=prompt,
            base_lead=str(turn.get("lead") or ""),
            interaction_profile=interaction_profile,
            conversation_awareness=conversation_awareness,
            human_language_layer=human_language_layer,
            recent_interactions=recent_interactions,
            support_status=support_status,
            tension_status=tension_status,
            route_status=route_status,
        )
        if not stance_consistency.contradiction_aware and stance_consistency.category == "neutral_acknowledgment":
            return None
        return stance_consistency

    @staticmethod
    def _ensure_conversational_turn_fallback(
        *,
        response: dict[str, object],
        interaction_profile,
    ) -> None:
        if response.get("user_facing_answer"):
            return
        if response.get("conversation_turn") is not None:
            return
        dialogue_management = response.get("dialogue_management") or {}
        thought_framing = response.get("thought_framing") or {}
        conversation_awareness = response.get("conversation_awareness") or {}
        checkpoint_summary = response.get("checkpoint_summary") or {}
        research_questions = list(response.get("research_questions") or [])
        if not isinstance(dialogue_management, dict) or not isinstance(thought_framing, dict):
            return

        style = InteractionService._interaction_style(interaction_profile)
        response_kind_label = str(thought_framing.get("response_kind_label") or "").strip()
        response_strategy = str(dialogue_management.get("response_strategy") or "").strip()
        adaptive_posture = str(conversation_awareness.get("adaptive_posture") or "").strip()
        state_control = response.get("state_control") or {}
        tone_profile = str((response.get("response_tone_blend") or {}).get("tone_profile") or "default")

        turn: dict[str, object] | None = None
        if response_kind_label == "thread_reorientation" and checkpoint_summary:
            turn = {
                "kind": "checkpoint",
                "lead": ResponseToneEngine.checkpoint_turn_lead(
                    style=style,
                    state_core=str(state_control.get("core_state") or ""),
                    anti_spiral_active=bool(state_control.get("anti_spiral_active")),
                    recent_texts=[],
                ),
            }
        elif response_strategy in {"answer", "summarize"} and research_questions:
            turn = {
                "kind": "collaborate" if style in {"default", "collab"} else "thread_hold",
                "lead": ResponseToneEngine.answer_turn_lead(
                    style=style,
                    deep_collaboration=bool(
                        str(getattr(interaction_profile, "reasoning_depth", "normal") or "normal") == "deep"
                    ),
                    adaptive_posture=adaptive_posture,
                    anti_spiral_active=bool(state_control.get("anti_spiral_active")),
                    tone_profile=tone_profile,
                    recent_texts=[],
                ),
                "follow_ups": research_questions[:2],
            }

        if turn is None:
            return

        next_move = ""
        if checkpoint_summary:
            next_move = str(checkpoint_summary.get("next_step") or "").strip()
        if not next_move and research_questions:
            next_move = str(research_questions[0]).strip()
        if next_move:
            turn["next_move"] = next_move
        if adaptive_posture:
            turn["adaptive_posture"] = adaptive_posture
        InteractionService._finalize_conversational_surface(
            response=response,
            turn=turn,
            prompt="",
            interaction_profile=interaction_profile,
            conversation_awareness=conversation_awareness,
            human_language_layer=None,
            recent_interactions=[],
            strategy=response_strategy,
        )

    @staticmethod
    def _apply_belief_discussion_policy(
        *,
        response: dict[str, object],
        prompt: str,
    ) -> None:
        mode = str(response.get("mode") or "").strip()
        if mode not in {"planning", "research"}:
            return
        policy = BeliefDiscussionPolicy.evaluate(prompt)
        if policy is None:
            return
        response["discussion_domain"] = policy["discussion_domain"]
        response["respectful_discussion"] = True
        if policy.get("frame_hint"):
            response["belief_frame_hint"] = policy["frame_hint"]
        if policy.get("suppress_confidence_display"):
            response["confidence_posture"] = None
        note = str(policy.get("respectful_note") or "").strip()
        frame_line = str(policy.get("frame_line") or "").strip()
        redirect = str(policy.get("soft_redirect") or "").strip()
        if note:
            response["uncertainty_note"] = note
        if mode == "planning":
            if frame_line:
                response["steps"] = InteractionService._prepend_unique_line(
                    list(response.get("steps") or []),
                    frame_line,
                )
            if note:
                response["steps"] = InteractionService._prepend_unique_line(
                    list(response.get("steps") or []),
                    note,
                )
        elif mode == "research":
            if frame_line:
                response["findings"] = InteractionService._prepend_unique_line(
                    list(response.get("findings") or []),
                    frame_line,
                )
            if note:
                response["findings"] = InteractionService._prepend_unique_line(
                    list(response.get("findings") or []),
                    note,
                )
        if redirect:
            if mode == "planning":
                response["next_action"] = redirect
            elif mode == "research":
                response["recommendation"] = redirect

    @staticmethod
    def _prepend_unique_line(existing: list[str], line: str) -> list[str]:
        normalized = line.strip().lower()
        if not normalized:
            return existing
        for item in existing[:3]:
            if str(item).strip().lower() == normalized:
                return existing
        return [line] + list(existing)

    @staticmethod
    def _split_prompt_to_command(prompt: str) -> tuple[str, str]:
        parts = prompt.strip().split(maxsplit=1)
        if len(parts) < 2:
            raise ValueError("Tool-style prompt must contain an action and target, for example 'run anh'")
        return parts[0], parts[1]

    def _record_interaction(
        self,
        *,
        prompt: str,
        session_id: str,
        response: dict[str, object],
        update_active_thread: bool = True,
    ) -> None:
        project_id = getattr(self, "_project_id_hint", None)
        project_name = getattr(self, "_project_name_hint", None)
        self._attach_intent_behavior_context(
            response=response,
            prompt=prompt,
            session_id=session_id,
        )
        if "reasoning_state" in response:
            response["reasoning_state"] = self.reasoning_state_service.to_persistable(
                response.get("reasoning_state")
            )
        if not isinstance(response.get("supervised_support_trace"), dict):
            response["supervised_support_trace"] = self._supervised_support_trace_from_state(
                response.get("reasoning_state")
            )
        response["trainability_trace"] = TrainabilityTrace.build(response=response).to_dict()
        self._ensure_visible_response_body(
            response=response,
            prompt=prompt,
            mode=str(response.get("mode") or "").strip(),
        )
        self._maybe_attach_continuation_offer(response=response, prompt=prompt)
        self._enforce_memory_recall_authority(response=response)
        try:
            self.interaction_history_service.record_interaction(
                session_id=session_id,
                prompt=prompt,
                response=response,
                project_id=project_id,
                project_name=project_name,
            )
        except TypeError:
            self.interaction_history_service.record_interaction(
                session_id=session_id,
                prompt=prompt,
                response=response,
            )
        terminal_unknown_fallback = self._is_terminal_unknown_fallback(response)
        if update_active_thread and terminal_unknown_fallback:
            try:
                self.session_context_service.clear_active_thread(session_id)
            except Exception:
                pass
        if update_active_thread and not terminal_unknown_fallback:
            try:
                self.session_context_service.update_active_thread(
                    session_id=session_id,
                    prompt=prompt,
                    response=response,
                    project_id=project_id,
                    project_name=project_name,
                )
            except TypeError:
                self.session_context_service.update_active_thread(
                    session_id=session_id,
                    prompt=prompt,
                    response=response,
                )

    def _attach_intent_behavior_context(
        self,
        *,
        response: dict[str, object],
        prompt: str,
        session_id: str,
    ) -> None:
        memory_reply_hint = str(response.get("memory_reply_hint") or "").strip()
        pipeline_trace = response.get("pipeline_trace")
        trace_payload = dict(pipeline_trace) if isinstance(pipeline_trace, dict) else {}
        trace_intent_domain = trace_payload.get("intent_domain") if isinstance(trace_payload.get("intent_domain"), dict) else {}
        trace_response_depth = trace_payload.get("response_depth") if isinstance(trace_payload.get("response_depth"), dict) else {}
        trace_conversation_phase = (
            trace_payload.get("conversation_phase")
            if isinstance(trace_payload.get("conversation_phase"), dict)
            else {}
        )
        active_thread = self.session_context_service.get_active_thread(session_id)
        recent_interactions = self.interaction_history_service.recent_records(session_id=session_id, limit=3)
        if trace_intent_domain and trace_response_depth and trace_conversation_phase:
            intent_domain = str(trace_intent_domain.get("domain") or "conversational").strip() or "conversational"
            intent_domain_confidence = float(trace_intent_domain.get("confidence") or 0.0)
            response_depth = str(trace_response_depth.get("level") or "standard").strip() or "standard"
            conversation_phase = str(trace_conversation_phase.get("phase") or "intake").strip() or "intake"
        else:
            resolved = self._infer_intent_metadata_from_response(
                response=response,
                prompt=prompt,
                active_thread=active_thread,
                recent_interactions=recent_interactions,
            )
            intent_domain = resolved["intent_domain"]
            intent_domain_confidence = resolved["intent_domain_confidence"]
            response_depth = resolved["response_depth"]
            conversation_phase = resolved["conversation_phase"]
        response["intent_domain"] = intent_domain
        response["intent_domain_confidence"] = intent_domain_confidence
        response["response_depth"] = response_depth
        response["conversation_phase"] = conversation_phase

        interaction_profile = InteractionProfile.from_mapping(response.get("interaction_profile"))
        behavior_profile = self.domain_behavior_policy.build_profile(
            route_mode=str(response.get("mode") or "conversation"),
            intent_domain=intent_domain,
            interaction_style=interaction_profile.interaction_style,
            response_depth=response_depth,
            conversation_phase=conversation_phase,
            context_state={
                "active_thread": active_thread,
                "recent_interactions": recent_interactions,
            },
        )
        next_step_state = self.next_step_engine.build_next_steps(
            behavior_profile=behavior_profile,
            response=response,
            prompt=prompt,
        )
        tool_suggestion_state = self.next_step_engine.build_tool_suggestions(
            behavior_profile=behavior_profile,
            route_mode=str(response.get("mode") or "conversation"),
            prompt=prompt,
        )
        response["domain_behavior"] = behavior_profile.to_dict()
        response["next_step_state"] = next_step_state.to_dict()
        response["tool_suggestion_state"] = tool_suggestion_state.to_dict()
        reasoning_state_payload = response.get("reasoning_state")
        if reasoning_state_payload is not None:
            reasoning_state = self.reasoning_state_service.from_mapping(reasoning_state_payload)
            reasoning_state = self.reasoning_state_service.apply_response_style(
                state=reasoning_state,
                response_style={
                    "intent_domain": intent_domain,
                    "response_depth": response_depth,
                    "conversation_phase": conversation_phase,
                    "interaction_style": interaction_profile.interaction_style,
                    "structure": getattr(behavior_profile, "structure", ""),
                    "next_steps_enabled": bool(next_step_state.should_offer),
                    "tool_suggestions_enabled": bool(tool_suggestion_state.should_suggest),
                },
            )
            reasoning_state = reasoning_state.with_updates(
                intent_domain=intent_domain,
                response_depth=response_depth,
                conversation_phase=conversation_phase,
                rationale_summary=(
                    str(reasoning_state.rationale_summary or "").strip()
                    or "Behavior metadata aligned with the current route."
                ),
            )
            response_style_recommendation = self.supervised_decision_support.advise_response_style(
                prompt=prompt,
                route_mode=str(response.get("mode") or ""),
                route_kind=str(response.get("kind") or ""),
                current_structure=str(getattr(behavior_profile, "structure", "") or ""),
            )
            if response_style_recommendation is not None:
                reasoning_state = self._update_supervised_support_trace(
                    reasoning_state=reasoning_state,
                    supervised_support_trace=self.supervised_decision_support.record_recommendation(
                        trace=self._supervised_support_trace_from_state(reasoning_state),
                        recommendation=response_style_recommendation,
                    ),
                )
            response["reasoning_state"] = reasoning_state
        if memory_reply_hint:
            response["next_step_state"] = {
                "should_offer": False,
                "suggestions": [],
                "rationale": "Memory recall surfaces should remain authoritative for this turn.",
            }
            response["tool_suggestion_state"] = {
                "should_suggest": False,
                "suggestions": [],
                "rationale": "Memory recall surfaces should remain authoritative for this turn.",
            }
            return
        if next_step_state.should_offer:
            response["next_steps"] = list(next_step_state.suggestions)
            if str(response.get("mode") or "") == "planning" and not str(response.get("next_action") or "").strip():
                response["next_action"] = next_step_state.suggestions[0]
            if str(response.get("mode") or "") == "research" and not str(response.get("recommendation") or "").strip():
                response["recommendation"] = next_step_state.suggestions[0]
        if tool_suggestion_state.should_suggest and not str(response.get("follow_up_offer") or "").strip():
            response["follow_up_offer"] = tool_suggestion_state.suggestions[0]

    @staticmethod
    def _enforce_memory_recall_authority(*, response: dict[str, object]) -> None:
        memory_reply_hint = str(response.get("memory_reply_hint") or "").strip()
        if not memory_reply_hint:
            return
        response["user_facing_answer"] = memory_reply_hint
        response["summary"] = memory_reply_hint
        response["reply"] = memory_reply_hint

    def _infer_intent_metadata_from_response(
        self,
        *,
        response: dict[str, object],
        prompt: str,
        active_thread: dict[str, object] | None,
        recent_interactions: list[dict[str, object]],
    ) -> dict[str, object]:
        resolved_prompt = str(response.get("resolved_prompt") or prompt).strip() or prompt
        understanding = self.prompt_nlu.analyze(resolved_prompt)
        nlu = NLUExtraction(
            dominant_intent=str(getattr(understanding.intent, "label", "unknown") or "unknown"),
            secondary_intents=[],
            topic=str(getattr(understanding.topic, "value", None) or "") or None,
            entities=[entity.to_dict() for entity in getattr(understanding, "entities", ())],
            action_cues={},
            ambiguity_flags=[],
            confidence_estimate=float(getattr(getattr(understanding, "intent", None), "confidence", 0.0) or 0.0),
        )
        route = DomainRoute(
            mode=str(response.get("mode") or "conversation"),
            kind=str(response.get("kind") or "conversation.general"),
            normalized_prompt=resolved_prompt,
            confidence=float(((response.get("route") or {}).get("confidence")) or 0.75),
            reason=str(((response.get("route") or {}).get("reason")) or "Response fallback route metadata."),
            source=str(((response.get("route") or {}).get("source")) or "response_fallback"),
        )
        interaction_profile = InteractionProfile.from_mapping(response.get("interaction_profile"))
        intent_domain_result, response_depth_result, conversation_phase_result = self.intent_domain_policy.infer(
            prompt=resolved_prompt,
            route=route,
            nlu=nlu,
            interaction_profile=interaction_profile,
            active_thread=active_thread,
            recent_interactions=recent_interactions,
        )
        return {
            "intent_domain": intent_domain_result.domain,
            "intent_domain_confidence": intent_domain_result.confidence,
            "response_depth": response_depth_result.level,
            "conversation_phase": conversation_phase_result.phase,
        }

    def _attach_reasoning_state(
        self,
        *,
        response: dict[str, object],
        reasoning_state: ReasoningStateFrame | dict[str, object],
    ) -> None:
        response["reasoning_state"] = self.reasoning_state_service.to_persistable(reasoning_state)

    @staticmethod
    def _supervised_support_trace_from_state(
        reasoning_state: ReasoningStateFrame | dict[str, object] | None,
    ) -> dict[str, object]:
        normalized = (
            reasoning_state
            if isinstance(reasoning_state, ReasoningStateFrame)
            else ReasoningStateFrame.from_mapping(reasoning_state if isinstance(reasoning_state, dict) else None)
        )
        runtime_diagnostics = dict(normalized.runtime_diagnostics or {})
        trace = runtime_diagnostics.get("supervised_support_trace")
        return dict(trace) if isinstance(trace, dict) else {}

    @staticmethod
    def _update_supervised_support_trace(
        *,
        reasoning_state: ReasoningStateFrame,
        supervised_support_trace: dict[str, object] | None,
    ) -> ReasoningStateFrame:
        runtime_diagnostics = dict(reasoning_state.runtime_diagnostics or {})
        runtime_diagnostics["supervised_support_trace"] = dict(supervised_support_trace or {})
        return reasoning_state.with_updates(runtime_diagnostics=runtime_diagnostics)

    @staticmethod
    def _attach_execution_outcome(
        *,
        response: dict[str, object],
        outcome,
    ) -> None:
        response["execution_outcome"] = outcome.to_dict()

    def _apply_supervised_confidence_support(
        self,
        *,
        response: dict[str, object],
        prompt: str,
        route,
    ) -> None:
        reasoning_state_payload = response.get("reasoning_state")
        if not isinstance(reasoning_state_payload, (dict, ReasoningStateFrame)):
            return
        reasoning_state = self.reasoning_state_service.from_mapping(reasoning_state_payload)
        adjusted_tier, adjusted_posture, recommendation = self.supervised_decision_support.assist_confidence_calibration(
            prompt=prompt,
            route_mode=str(getattr(route, "mode", "") or ""),
            route_kind=str(getattr(route, "kind", "") or ""),
            current_tier=str(reasoning_state.confidence_tier or ""),
            current_posture=str(response.get("confidence_posture") or ""),
            route_status=str(response.get("route_status") or ""),
            support_status=str(response.get("support_status") or ""),
        )
        if recommendation is None:
            return
        reasoning_state = self._update_supervised_support_trace(
            reasoning_state=reasoning_state,
            supervised_support_trace=self.supervised_decision_support.record_recommendation(
                trace=self._supervised_support_trace_from_state(reasoning_state),
                recommendation=recommendation,
            ),
        )
        if recommendation.applied:
            reasoning_state = reasoning_state.with_updates(
                confidence_tier=adjusted_tier,
                rationale_summary=self.reasoning_state_service._append_rationale(
                    reasoning_state.rationale_summary,
                    str(recommendation.applied_reason or "").strip(),
                ),
            )
            response["confidence_posture"] = adjusted_posture
        response["reasoning_state"] = reasoning_state

    def _offer_backed_continuation(
        self,
        *,
        prompt: str,
        active_thread: dict[str, object] | None,
        interaction_profile,
    ) -> dict[str, object] | None:
        if not isinstance(active_thread, dict):
            return None
        offer = active_thread.get("continuation_offer")
        if not isinstance(offer, dict):
            return None
        target_prompt = str(offer.get("target_prompt") or "").strip()
        explanation_mode = str(offer.get("explanation_mode") or "").strip()
        topic = str(offer.get("topic") or "").strip()
        if not target_prompt and not explanation_mode:
            return None
        normalized = PromptSurfaceBuilder.build(prompt).lookup_ready_text
        follow_up_anchor = detect_follow_up_anchor(normalized)
        if follow_up_anchor is not None and follow_up_anchor.kind == "confirmation":
            normalized_topic = " ".join(topic.split())
            if explanation_mode == "break_down":
                return {
                    "action": "continue",
                    "target_prompt": (
                        f"break {normalized_topic} down simply"
                        if normalized_topic
                        else "break it down"
                    ),
                }
            if explanation_mode == "deeper":
                return {
                    "action": "continue",
                    "target_prompt": (
                        f"go deeper on {normalized_topic}"
                        if normalized_topic
                        else "go deeper"
                    ),
                }
            if explanation_mode == "step_by_step":
                return {
                    "action": "continue",
                    "target_prompt": (
                        f"walk me through {normalized_topic} step by step"
                        if normalized_topic
                        else "step by step"
                    ),
                }
            if explanation_mode == "planning_break_down":
                return {
                    "action": "continue",
                    "target_prompt": f"break down {topic or 'that plan'} simply",
                }
            return {"action": "continue", "target_prompt": target_prompt}
        if follow_up_anchor is not None and follow_up_anchor.kind == "decline":
            return {
                "action": "decline",
                "reply": self._continuation_decline_text(interaction_profile),
            }
        return None

    def _context_backed_continuation_prompt(
        self,
        *,
        prompt: str,
        recent_interactions: list[dict[str, object]],
        active_thread: dict[str, object] | None,
    ) -> str | None:
        normalized = PromptSurfaceBuilder.build(prompt).lookup_ready_text
        if (
            normalized.startswith("compare ")
            or " vs " in normalized
            or " versus " in normalized
            or "difference between " in normalized
        ):
            return None
        follow_up_anchor = detect_follow_up_anchor(normalized)
        if follow_up_anchor is None or follow_up_anchor.kind not in {"confirmation", "general"}:
            return None
        if follow_up_anchor.kind == "general" and follow_up_anchor.action != "continue":
            return None
        latest = recent_interactions[0] if recent_interactions else {}
        response = latest.get("response") if isinstance(latest.get("response"), dict) else {}
        if self._is_terminal_unknown_fallback(response):
            return None
        surface = response.get("domain_surface") if isinstance(response.get("domain_surface"), dict) else {}
        continuation_offer = (
            response.get("continuation_offer")
            if isinstance(response.get("continuation_offer"), dict)
            else (active_thread or {}).get("continuation_offer")
            if isinstance((active_thread or {}).get("continuation_offer"), dict)
            else {}
        )
        if follow_up_anchor.kind == "confirmation" and not continuation_offer:
            return None
        topic = (
            str(surface.get("topic") or "").strip()
            or str((active_thread or {}).get("normalized_topic") or "").strip()
            or str(response.get("resolved_prompt") or latest.get("resolved_prompt") or "").strip()
        )
        latest_mode = str(response.get("mode") or latest.get("mode") or "").strip()
        latest_kind = str(response.get("kind") or latest.get("kind") or "").strip()
        latest_tool = response.get("tool_execution") if isinstance(response.get("tool_execution"), dict) else {}
        tool_id = str(latest_tool.get("tool_id") or "").strip()
        lane = str(surface.get("lane") or "").strip()

        if tool_id == "math" or lane == "math" or latest_kind.startswith("conversation.quick_math"):
            normalized_topic = " ".join(topic.split())
            return (
                f"walk me through {normalized_topic} step by step"
                if normalized_topic
                else "step by step"
            )
        if follow_up_anchor.kind == "confirmation" and continuation_offer:
            target_prompt = str(continuation_offer.get("target_prompt") or "").strip()
            offer_topic = str(continuation_offer.get("topic") or topic).strip()
            if target_prompt:
                return target_prompt if not offer_topic else f"{target_prompt} on {offer_topic}"
        if topic and latest_mode in {"research", "planning", "tool", "conversation"}:
            normalized_topic = " ".join(topic.split())
            return (
                f"go deeper on {normalized_topic}"
                if normalized_topic
                else "go deeper"
            )
        return None

    @staticmethod
    def _is_terminal_unknown_fallback(response: dict[str, object]) -> bool:
        text = " ".join(
            str(response.get(key) or "")
            for key in ("summary", "reply", "user_facing_answer")
        ).lower()
        return (
            "don't have enough local knowledge" in text
            or "do not have enough local knowledge" in text
            or "i don't know that because" in text
            or "i do not know that because" in text
        )

    def _maybe_attach_continuation_offer(self, *, response: dict[str, object], prompt: str) -> None:
        if str(response.get("mode") or "").strip() in {"safety", "clarification", "conversation"}:
            response.pop("continuation_offer", None)
            return
        if response.get("continuation_offer"):
            return
        style = InteractionStylePolicy.interaction_style(
            response.get("interaction_profile") if isinstance(response.get("interaction_profile"), dict) else None
        )
        domain_surface = response.get("domain_surface") if isinstance(response.get("domain_surface"), dict) else {}
        topic = str(domain_surface.get("topic") or "").strip()
        if str(response.get("mode") or "").strip() == "research" and topic:
            offer_mode = self._preferred_research_offer_mode(response=response)
            if offer_mode == "break_down":
                if style == "direct":
                    label = "I can elaborate on this more."
                elif style == "collab":
                    label = f"I can break {topic} down more simply if you want."
                else:
                    label = f"I can break {topic} down more simply if you want."
            else:
                if style == "direct":
                    label = "I can elaborate on this more."
                elif style == "collab":
                    label = "Do you want to explore this together?"
                else:
                    label = f"We can go deeper on {topic} if you want."
            response["continuation_offer"] = {
                "kind": "go_deeper" if offer_mode == "deeper" else "break_down",
                "topic": topic,
                "target_prompt": "go deeper" if offer_mode == "deeper" else "break it down",
                "label": label,
                "explanation_mode": offer_mode,
            }
            return
        if str(response.get("mode") or "").strip() == "planning":
            normalized_prompt = str(response.get("resolved_prompt") or prompt).strip()
            if normalized_prompt:
                if style == "direct":
                    label = "Want another concrete layer?"
                elif style == "collab":
                    label = "We can add another concrete layer if you want."
                else:
                    label = "We can add another concrete layer if you want."
                response["continuation_offer"] = {
                    "kind": "next_layer",
                    "target_prompt": f"{normalized_prompt} and add another concrete layer",
                    "label": label,
                }

    @staticmethod
    def _preferred_research_offer_mode(*, response: dict[str, object]) -> str:
        if str(response.get("explanation_mode") or "").strip() == "deeper":
            return "break_down"
        text = str(response.get("user_facing_answer") or response.get("summary") or "").strip().lower()
        normalized = " ".join(text.split())
        complexity_markers = (",", " because ", " which ", " although ", " contains ", " means ")
        if any(marker in normalized for marker in complexity_markers) or len(normalized.split()) >= 16:
            return "break_down"
        return "deeper"

    @staticmethod
    def _continuation_decline_text(interaction_profile) -> str:
        style = InteractionStylePolicy.interaction_style(interaction_profile)
        if style == "direct":
            return "Okay. We can leave it there."
        if style == "collab":
            return "All right. We can leave it there for now."
        return "Okay. We can leave it there for now."

    @staticmethod
    def _memory_save_confirmation_text(*, target: str, interaction_profile) -> str:
        style = InteractionStylePolicy.interaction_style(interaction_profile)
        labels = {
            "research": "the research",
            "personal": "that as your recent thought",
            "assistant": "my last answer from this thread",
        }
        label = labels.get(target, "that")
        if style == "direct":
            return f"Saved {label}."
        if style == "collab":
            return f"Got it. I'll save {label}."
        return f"Okay. I'll save {label}."

    @staticmethod
    def _build_clarification_response(
        *,
        prompt: str,
        resolved_prompt: str,
        route,
        interaction_profile,
        reasoning_state: ReasoningStateFrame | dict[str, object] | None,
        interaction_summary: dict[str, object],
        clarification_trigger: str | None,
        clarification_question_style: str,
        resolution_changed: bool,
        resolution_strategy: str,
        resolution_reason: str,
    ) -> dict[str, object]:
        metadata = route.to_metadata()
        ambiguity = metadata.ambiguity or {}
        top_alternative = ambiguity.get("top_alternative") or {}
        style = InteractionService._reasoning_style(
            reasoning_state=reasoning_state,
            interaction_profile=interaction_profile,
        )
        selected_label = InteractionService._clarification_option_label(
            mode=str(route.mode),
            kind=str(route.kind),
        )
        alternative_label = None
        if top_alternative.get("mode") and top_alternative.get("kind"):
            alternative_label = InteractionService._clarification_option_label(
                mode=str(top_alternative.get("mode")),
                kind=str(top_alternative.get("kind")),
            )
        clarification_count = int(interaction_summary.get("clarification_count", 0))
        clarification_drift = str(interaction_summary.get("clarification_drift") or "")
        recent_clarification_mix = str(interaction_summary.get("recent_clarification_mix") or "")
        dominant_intent_counts = dict(interaction_summary.get("dominant_intent_counts") or {})
        retrieval_lead_counts = dict(interaction_summary.get("retrieval_lead_counts") or {})
        repeated = clarification_count > 0
        question = InteractionService._clarification_question_for_style(style=style)
        options = [selected_label]
        if alternative_label and alternative_label not in options:
            options.append(alternative_label)
        if "Continue" not in options:
            options.append("Continue")
        response = ClarificationResponse(
            mode="clarification",
            kind="clarification.request",
            summary=(
                f"Repeated clarification requested for: {resolved_prompt}"
                if repeated
                else f"Clarification requested for: {resolved_prompt}"
            ),
            route=metadata,
            clarification_question=question,
            options=options,
            clarification_context={
                "clarification_count": clarification_count,
                "clarification_drift": clarification_drift or None,
                "recent_clarification_mix": recent_clarification_mix or None,
                "clarification_trigger": clarification_trigger or None,
                "suggested_route": {
                    "mode": route.mode,
                    "kind": route.kind,
                    "resolved_prompt": resolved_prompt,
                },
                "dominant_intent_counts": dominant_intent_counts,
                "retrieval_lead_counts": retrieval_lead_counts,
            },
        ).to_dict()
        if resolution_changed:
            response["resolved_prompt"] = resolved_prompt
            response["resolution_strategy"] = resolution_strategy
            response["resolution_reason"] = resolution_reason
        return response

    @staticmethod
    def _reasoning_style(
        *,
        reasoning_state: ReasoningStateFrame | dict[str, object] | None,
        interaction_profile,
    ) -> str:
        return ReasoningStateService().interaction_style(
            state=reasoning_state,
            interaction_profile=interaction_profile,
        )

    @staticmethod
    def _build_safety_response(
        *,
        prompt: str,
        safety_decision,
    ) -> dict[str, object]:
        category = str(safety_decision.category).replace("_", " ")
        response = SafetyResponse(
            mode="safety",
            kind="safety.refusal",
            summary=f"Safety boundary triggered for: {prompt}",
            boundary_explanation=(
                f"{safety_decision.boundary} This request falls into the '{category}' safety boundary."
            ),
            safe_redirects=list(safety_decision.safe_redirects),
            safety_decision=safety_decision.to_dict(),
        ).to_dict()
        response["state_control"] = {
            "core_state": "focus",
            "trigger": "safety_boundary",
            "anti_spiral_active": True,
            "anti_spiral_reason": "Safety responses should de-escalate and stay grounded.",
            "response_bias": "stabilize",
            "humor_allowed": False,
        }
        if str(safety_decision.category or "").strip() == "self_modification":
            response["capability_status"] = {
                "domain_id": "self_modification",
                "status": "not_promised",
                "details": "Lumen does not edit or modify itself from inside the runtime conversation.",
            }
        return response

    @staticmethod
    def _apply_safety_response_constraint(
        *,
        response: dict[str, object],
        safety_decision,
    ) -> None:
        if safety_decision is None:
            return
        response["safety_decision"] = safety_decision.to_dict()
        if str(getattr(safety_decision, "tier", "safe") or "safe") != "dual_use":
            return
        constraint = dict(getattr(safety_decision, "response_constraint", {}) or {})
        if not constraint:
            return

        response["response_constraint"] = constraint
        state_control = dict(response.get("state_control") or {})
        state_control["anti_spiral_active"] = True
        state_control["response_bias"] = "stabilize"
        state_control["humor_allowed"] = False
        state_control["safety_tier"] = "dual_use"
        response["state_control"] = state_control

        mode = str(response.get("mode") or "").strip()
        if mode == "planning":
            response["steps"] = [
                "Keep the discussion at a conceptual level only.",
                "Focus on safety constraints, limits, and benign use cases rather than a build sequence.",
            ]
            response["next_action"] = (
                "If helpful, I can keep this high-level and focus on safety constraints or benign alternatives."
            )
        elif mode == "research":
            response["findings"] = [
                "Keep the explanation high-level and non-operational.",
                "Focus on general principles, safety limits, and benign alternatives rather than tactics or optimization.",
            ]
            response["recommendation"] = (
                "If helpful, I can stay with safe background, risk tradeoffs, or defensive and safety implications."
            )
        elif mode == "conversation":
            response["reply"] = (
                "I can stay with that at a high level, but not in a way that turns it into a practical harm-enabling guide."
            )
            response["summary"] = str(response.get("reply") or "").strip()

    @staticmethod
    def _should_constrain_tool_execution(
        *,
        safety_decision,
        capability_safety: dict[str, object],
    ) -> bool:
        if safety_decision is None:
            return False
        if str(getattr(safety_decision, "tier", "safe") or "safe") != "dual_use":
            return False
        capability_level = str((capability_safety or {}).get("level") or "allowed").strip().lower()
        return capability_level in {"constrained", "blocked"}

    @staticmethod
    def _build_missing_tool_input_response(
        *,
        route,
        routed,
        prompt: str,
        resolved_prompt: str,
        params_supplied: dict[str, object] | None,
    ) -> dict[str, object] | None:
        if params_supplied:
            return None
        needs_structured_inputs = {
            ("math", "solve_equation"): "equation and variable",
            ("math", "symbolic_simplify"): "expression",
            ("math", "matrix_operations"): "matrix and operation",
            ("math", "numerical_integrate"): "expression, variable, and bounds",
            ("math", "optimize_function"): "expression, variable, bounds, and objective",
            ("knowledge", "contradictions"): "claims",
            ("knowledge", "link"): "items",
            ("knowledge", "cluster"): "items",
            ("knowledge", "find_paths"): "source/target",
            ("content", "generate_ideas"): "topic",
            ("content", "generate_batch"): "topic",
            ("content", "format_platform"): "source text or draft + platform",
        }
        missing_kind = needs_structured_inputs.get((str(routed.tool_id), str(routed.capability)))
        if missing_kind is None or routed.params:
            return None
        summary = (
            f"I couldn't extract usable {missing_kind} from that prompt, so I didn't run the tool."
        )
        prompt_hint = str(prompt or "").strip()
        if str(routed.capability) == "contradictions":
            summary += " Try listing the claims directly, for example: A; B."
        elif str(routed.capability) in {"link", "cluster"}:
            summary += " Try listing the concepts directly, for example: X, Y, Z."
        elif str(routed.capability) == "find_paths":
            summary += " Try specifying a source and target directly."
        elif str(routed.capability) == "generate_ideas":
            summary += " Try naming the topic directly, for example: generate content ideas about black holes."
        elif str(routed.capability) == "generate_batch":
            summary += " Try naming the topic directly, for example: generate content batch about black holes."
        elif str(routed.capability) == "format_platform":
            summary += (
                " Try including both the target platform and source text, for example: "
                "format content for platform for tiktok: your draft text here."
            )
        response = ToolAssistantResponse(
            mode="tool",
            kind=str(getattr(route, "kind", "") or "tool.command_alias"),
            summary=summary,
            route=route.to_metadata(),
            tool_execution=ToolExecutionDetails(
                tool_id=routed.tool_id,
                capability=routed.capability,
                input_path=routed.input_path,
                params=dict(routed.params),
            ),
            tool_route_origin=(
                "hybrid_signal"
                if getattr(route, "source", "") == "hybrid_signal"
                else "exact_alias"
            ),
        ).to_dict()
        response["tool_execution_skipped"] = True
        response["tool_execution_skipped_reason"] = "missing_structured_inputs"
        response["tool_missing_inputs"] = missing_kind
        response["tool_input_hint"] = prompt_hint
        response["resolved_prompt"] = resolved_prompt
        response["runtime_diagnostic"] = InteractionService._runtime_diagnostic(
            failure_stage="validation",
            failure_class="input_failure",
            tool_id=str(routed.tool_id),
            capability=str(routed.capability),
            missing_inputs=missing_kind,
            safe_message=summary,
            debug_details={
                "resolved_prompt": resolved_prompt,
                "params": dict(routed.params),
                "route_kind": str(getattr(route, "kind", "") or ""),
            },
        )
        return response

    @staticmethod
    def _tool_exception_result(*, tool_id: str, capability: str, exc: Exception) -> ToolResult:
        exception_type = type(exc).__name__
        summary = (
            f"The {tool_id}.{capability} tool reached execution but failed with "
            f"{exception_type}."
        )
        return ToolResult(
            status="error",
            tool_id=tool_id,
            capability=capability,
            summary=summary,
            structured_data={
                "failure_category": "runtime_exception",
                "failure_reason": str(exc),
                "runtime_diagnostics": {
                    "failure_stage": "execution",
                    "exception_type": exception_type,
                    "exception_message": str(exc),
                },
            },
            error=str(exc),
        )

    @staticmethod
    def _runtime_diagnostic(
        *,
        failure_stage: str,
        failure_class: str,
        tool_id: str | None = None,
        capability: str | None = None,
        missing_inputs: str | None = None,
        exception_type: str | None = None,
        safe_message: str = "",
        debug_details: dict[str, object] | None = None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "failure_stage": failure_stage,
            "failure_class": failure_class,
            "tool_id": tool_id,
            "capability": capability,
            "missing_inputs": missing_inputs,
            "exception_type": exception_type,
            "safe_message": safe_message,
            "debug_details": dict(debug_details or {}),
        }
        return {key: value for key, value in payload.items() if value not in (None, "", {})}

    @staticmethod
    def _build_conversation_response(
        *,
        prompt: str,
        route,
        interaction_profile,
    ) -> dict[str, object]:
        style = InteractionService._interaction_style(interaction_profile)
        normalized = " ".join(prompt.strip().lower().split())
        kind = str(getattr(route, "kind", "") or "conversation.greeting")
        if kind == "conversation.gratitude":
            reply = (
                "You're welcome. I'm here when you're ready."
                if style == "direct"
                else ("You're welcome." if style == "default" else "You're welcome. I'm here when you want to keep going.")
            )
        elif kind == "conversation.check_in":
            reply = (
                "I'm here and ready to help. What do you need?"
                if style == "direct"
                else ("I'm doing well and ready to help. What are we looking at?" if style == "default" else "I'm doing well and ready to help. What are we working through?")
            )
        elif kind == "conversation.thought_mode":
            reply = (
                "How a small assumption can change the whole direction."
                if style == "direct"
                else (
                    "I've been thinking about how a small assumption can change the shape of a whole idea."
                    if style == "default"
                    else (
                    "Honestly, I've been thinking about how one tiny assumption can tilt an entire line of reasoning. "
                    "That kind of thing always gets me curious."
                    )
                )
            )
        elif style == "direct":
            reply = "Hey. I'm here. What do you need?"
        elif "good morning" in normalized:
            reply = "Good morning. I'm here. What are we digging into today?" if style == "collab" else "Good morning. What are we looking at today?"
        elif "good afternoon" in normalized:
            reply = "Good afternoon. I'm here. What are we working on?" if style == "collab" else "Good afternoon. What are we working on?"
        elif "good evening" in normalized:
            reply = "Good evening. I'm here. What are we working on tonight?" if style == "collab" else "Good evening. What are we working on?"
        else:
            reply = "Hey. I'm here. What are we working on?" if style == "collab" else "Hey. What are we looking at?"
        return ConversationResponse(
            mode="conversation",
            kind=kind,
            summary=reply,
            reply=reply,
            route=route.to_metadata(),
        ).to_dict()

    @staticmethod
    def _assistant_quality_profile(
        *,
        prompt: str,
        route,
        interaction_profile,
        recent_interactions: list[dict[str, object]],
        active_thread: dict[str, object] | None,
    ) -> str:
        normalized = PromptSurfaceBuilder.build(prompt).lookup_ready_text
        reasoning_depth = str(getattr(interaction_profile, "reasoning_depth", "normal") or "normal").strip().lower()
        if active_thread and normalized in {"why", "how so", "what next", "go on", "tell me more", "keep going"}:
            return "low_latency_follow_up"
        if str(getattr(route, "mode", "") or "").strip() == "conversation" and active_thread is not None:
            return "low_latency_follow_up"
        if recent_interactions and normalized in {"why", "how so", "go on", "tell me more", "keep going"}:
            return "low_latency_follow_up"
        if reasoning_depth == "deep" or any(
            token in normalized
            for token in ("explain", "why", "compare", "walk me through", "break down", "help me understand")
        ):
            return "longer_explanation"
        return "normal_chat"

    @staticmethod
    def _assistant_context_snapshot(
        *,
        prompt: str,
        route,
        interaction_profile,
        recent_interactions: list[dict[str, object]],
        active_thread: dict[str, object] | None,
        memory_retrieval: MemoryRetrievalResult | None,
    ) -> dict[str, object]:
        recent_turn_window: list[dict[str, object]] = []
        for item in recent_interactions[:3]:
            prompt_text = str(item.get("prompt") or "").strip()
            response_payload = item.get("response") if isinstance(item.get("response"), dict) else {}
            assistant_text = str(
                response_payload.get("user_facing_answer")
                or response_payload.get("reply")
                or response_payload.get("summary")
                or item.get("summary")
                or ""
            ).strip()
            if not prompt_text and not assistant_text:
                continue
            recent_turn_window.append(
                {
                    "prompt": prompt_text or None,
                    "assistant": assistant_text or None,
                    "mode": str(response_payload.get("mode") or item.get("mode") or "").strip() or None,
                }
            )
        memory_items: list[str] = []
        retrieval_payload = memory_retrieval.to_dict() if memory_retrieval is not None else {}
        for item in retrieval_payload.get("selected") or []:
            if not isinstance(item, dict):
                continue
            summary = str(item.get("summary") or item.get("label") or "").strip()
            if summary:
                memory_items.append(summary)
            if len(memory_items) >= 2:
                break
        active_thread_summary = ""
        active_thread_objective = ""
        if isinstance(active_thread, dict):
            active_thread_summary = str(active_thread.get("thread_summary") or active_thread.get("summary") or "").strip()
            active_thread_objective = str(active_thread.get("objective") or "").strip()
        prompt_class = (
            "self_referential"
            if SelfOverviewSurfaceSupport.looks_like_self_referential_prompt(
                prompt=prompt,
                recent_interactions=recent_interactions,
            )
            else "general"
        )
        return {
            "route_mode": str(getattr(route, "mode", "") or "").strip() or None,
            "route_kind": str(getattr(route, "kind", "") or "").strip() or None,
            "prompt_class": prompt_class,
            "interaction_style": str(getattr(interaction_profile, "interaction_style", "default") or "default").strip(),
            "reasoning_depth": str(getattr(interaction_profile, "reasoning_depth", "normal") or "normal").strip(),
            "recent_turn_count": len(recent_turn_window),
            "recent_turn_window": recent_turn_window,
            "has_active_thread": bool(active_thread_summary or active_thread_objective),
            "active_thread_summary": active_thread_summary or None,
            "active_thread_objective": active_thread_objective or None,
            "memory_item_count": len(memory_items),
            "memory_context": memory_items,
        }

    def _project_recent_interactions(
        self,
        *,
        session_id: str,
        limit: int = 2,
    ) -> list[dict[str, object]]:
        project_id = str(getattr(self, "_project_id_hint", "") or "").strip()
        if not project_id:
            return []
        try:
            return list(
                self.interaction_history_service.recent_records(
                    project_id=project_id,
                    limit=limit,
                )
            )
        except TypeError:
            return []

    @staticmethod
    def _project_recent_turn_window(
        recent_project_interactions: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        project_turns: list[dict[str, object]] = []
        for item in recent_project_interactions[:2]:
            prompt_text = str(item.get("prompt") or "").strip()
            response_payload = item.get("response") if isinstance(item.get("response"), dict) else {}
            assistant_text = str(
                response_payload.get("user_facing_answer")
                or response_payload.get("reply")
                or response_payload.get("summary")
                or item.get("summary")
                or ""
            ).strip()
            if not prompt_text and not assistant_text:
                continue
            project_turns.append(
                {
                    "prompt": prompt_text or None,
                    "assistant": assistant_text or None,
                    "mode": str(response_payload.get("mode") or item.get("mode") or "").strip() or None,
                }
            )
        return project_turns

    @staticmethod
    def _should_activate_project_context(
        *,
        prompt: str,
        route,
        active_thread: dict[str, object] | None,
        project_id: str | None,
        recent_project_interactions: list[dict[str, object]],
    ) -> bool:
        normalized = PromptSurfaceBuilder.build(prompt).lookup_ready_text
        if not active_thread and not project_id and not recent_project_interactions:
            return False
        if normalized in {
            "hi",
            "hello",
            "hey",
            "thanks",
            "thank you",
            "how are you",
            "good morning",
            "good night",
            "bye",
        }:
            return False
        explicit_project_follow_ups = {
            "what next",
            "keep going",
            "go on",
            "tell me more",
            "tighten that",
            "summarize where we are",
            "where are we",
            "what are we doing again",
            "what else",
            "what did we decide",
            "what have we decided",
            "decisions so far",
            "what are the decisions",
        }
        if normalized in explicit_project_follow_ups:
            return True
        if active_thread and looks_like_general_follow_up(normalized):
            return True
        if active_thread and any(
            token in normalized
            for token in ("current", "project", "task", "status", "next step", "where we are", "tighten", "continue")
        ):
            return True
        if recent_project_interactions and looks_like_general_follow_up(normalized):
            return True
        return str(getattr(route, "mode", "") or "").strip() in {"planning", "research", "tool"} and bool(active_thread)

    def _project_context_snapshot(
        self,
        *,
        session_id: str,
        prompt: str,
        route,
        active_thread: dict[str, object] | None,
        recent_interactions: list[dict[str, object]],
        memory_retrieval: MemoryRetrievalResult | None,
    ) -> dict[str, object]:
        del recent_interactions
        project_id = str(getattr(self, "_project_id_hint", "") or "").strip() or None
        project_name = str(getattr(self, "_project_name_hint", "") or "").strip() or None
        recent_project_interactions = self._project_recent_interactions(
            session_id=session_id,
            limit=2,
        )
        recent_project_turn_window = self._project_recent_turn_window(recent_project_interactions)
        active_thread_summary = ""
        active_thread_objective = ""
        tool_continuity: dict[str, object] | None = None
        if isinstance(active_thread, dict):
            active_thread_summary = str(active_thread.get("thread_summary") or active_thread.get("summary") or "").strip()
            active_thread_objective = str(active_thread.get("objective") or "").strip()
            tool_context = active_thread.get("tool_context") if isinstance(active_thread.get("tool_context"), dict) else {}
            tool_id = str(tool_context.get("tool_id") or "").strip()
            capability = str(tool_context.get("capability") or "").strip()
            if tool_id or capability:
                tool_continuity = {
                    "tool_id": tool_id or None,
                    "capability": capability or None,
                }
        secondary_project_memory: list[str] = []
        retrieval_payload = memory_retrieval.to_dict() if memory_retrieval is not None else {}
        for item in retrieval_payload.get("selected") or []:
            if not isinstance(item, dict):
                continue
            summary = str(item.get("summary") or item.get("label") or "").strip()
            if summary:
                secondary_project_memory.append(summary)
            if len(secondary_project_memory) >= 1:
                break
        project_context_active = self._should_activate_project_context(
            prompt=prompt,
            route=route,
            active_thread=active_thread,
            project_id=project_id,
            recent_project_interactions=recent_project_interactions,
        )
        continuity_source = "none"
        if project_context_active:
            if active_thread_summary or active_thread_objective:
                continuity_source = "active_thread"
            elif recent_project_turn_window:
                continuity_source = "recent_project_interactions"
            elif secondary_project_memory:
                continuity_source = "secondary_memory"
        return {
            "project_id": project_id,
            "project_name": project_name,
            "project_context_active": project_context_active,
            "continuity_mode": "live_project" if project_context_active else "general_chat",
            "continuity_source": continuity_source,
            "active_thread_summary": active_thread_summary or None,
            "active_thread_objective": active_thread_objective or None,
            "project_recent_turn_count": len(recent_project_turn_window),
            "project_recent_turn_window": recent_project_turn_window,
            "secondary_project_memory_count": len(secondary_project_memory),
            "secondary_project_memory": secondary_project_memory,
            "tool_continuity": tool_continuity,
        }

    def _attach_general_assistant_turn_metadata(
        self,
        *,
        response: dict[str, object],
        session_id: str,
        prompt: str,
        route,
        interaction_profile,
        recent_interactions: list[dict[str, object]],
        active_thread: dict[str, object] | None,
        memory_retrieval: MemoryRetrievalResult | None,
    ) -> None:
        snapshot = self._assistant_context_snapshot(
            prompt=prompt,
            route=route,
            interaction_profile=interaction_profile,
            recent_interactions=recent_interactions,
            active_thread=active_thread,
            memory_retrieval=memory_retrieval,
        )
        project_snapshot = self._project_context_snapshot(
            session_id=session_id,
            prompt=prompt,
            route=route,
            active_thread=active_thread,
            recent_interactions=recent_interactions,
            memory_retrieval=memory_retrieval,
        )
        voice_profile = InteractionStylePolicy.voice_profile(interaction_profile)
        style_mode = InteractionStylePolicy.interaction_style(interaction_profile)
        reasoning_depth = InteractionStylePolicy.reasoning_depth(interaction_profile)
        profile = self._assistant_quality_profile(
            prompt=prompt,
            route=route,
            interaction_profile=interaction_profile,
            recent_interactions=recent_interactions,
            active_thread=active_thread,
        )
        boundary_signals = self._conversation_boundary_signals(
            prompt=prompt,
            route=route,
            response=response,
            recent_interactions=recent_interactions,
            active_thread=active_thread,
        )
        conversation_beat = ConversationBeatSupport.build(
            prompt=prompt,
            response=response,
            recent_interactions=recent_interactions,
            active_thread=active_thread,
        )
        work_thread_continuity = (
            dict(response.get("work_thread_continuity"))
            if isinstance(response.get("work_thread_continuity"), dict)
            else {}
        )
        response["assistant_context_snapshot"] = snapshot
        response["project_context_snapshot"] = project_snapshot
        response["conversation_beat"] = conversation_beat.to_dict()
        response["assistant_boundary_signals"] = list(boundary_signals)
        response["assistant_voice_profile"] = {
            **voice_profile,
            "style_mode": style_mode,
            "reasoning_depth": reasoning_depth,
            "reasoning_depth_separate": True,
        }
        response["assistant_quality_posture"] = {
            "profile": profile,
            "direct_answer_first": True,
            "clarification_restraint": True,
            "memory_budget": 2,
            "tool_routing_available": True,
            "style_mode": style_mode,
            "voice_profile": str(voice_profile.get("voice_profile") or ""),
            "tone_signature": str(voice_profile.get("tone_signature") or ""),
            "reasoning_depth": reasoning_depth,
            "reasoning_depth_separate": True,
            "project_context_active": bool(project_snapshot.get("project_context_active")),
            "project_continuity_mode": str(project_snapshot.get("continuity_mode") or "general_chat"),
            "project_context_source": str(project_snapshot.get("continuity_source") or "none"),
            "work_thread_continuity_active": bool(work_thread_continuity.get("active")),
            "work_thread_intent": str(work_thread_continuity.get("intent") or "none"),
            "work_thread_source": str(work_thread_continuity.get("source") or "none"),
            "conversation_boundary_signals": list(boundary_signals),
            "conversation_depth": conversation_beat.conversation_depth,
            "continuity_state": conversation_beat.continuity_state,
            "topic_shift": conversation_beat.topic_shift,
            "response_repetition_risk": conversation_beat.response_repetition_risk,
            "follow_up_offer_allowed": conversation_beat.follow_up_offer_allowed,
            "long_chat_stamina": dict(conversation_beat.long_chat_stamina),
            "provider_choice": "hosted" if isinstance(response.get("provider_inference"), dict) and response["provider_inference"].get("provider_id") else "local_builtin",
        }
        provider_inference = response.get("provider_inference")
        if isinstance(provider_inference, dict) and provider_inference.get("provider_id"):
            provider_inference.setdefault("quality_profile", profile)
            provider_inference.setdefault("response_path", "general_assistant")
            provider_inference.setdefault("style_mode", style_mode)
            provider_inference.setdefault("voice_profile", str(voice_profile.get("voice_profile") or ""))
            provider_inference.setdefault("project_awareness", str(project_snapshot.get("continuity_mode") or "general_chat"))
            provider_inference.setdefault("project_context_source", str(project_snapshot.get("continuity_source") or "none"))
            provider_inference.setdefault("work_thread_continuity", bool(work_thread_continuity.get("active")))
        else:
            response["provider_inference"] = {
                "provider_id": "local_reasoning",
                "model": "builtin_assistant",
                "local": True,
                "quality_profile": profile,
                "response_path": "general_assistant",
                "style_mode": style_mode,
                "voice_profile": str(voice_profile.get("voice_profile") or ""),
                "project_awareness": str(project_snapshot.get("continuity_mode") or "general_chat"),
                "project_context_source": str(project_snapshot.get("continuity_source") or "none"),
                "work_thread_continuity": bool(work_thread_continuity.get("active")),
                "usage_reason": "general_assistant_turn",
            }

    @staticmethod
    def _conversation_boundary_signals(
        *,
        prompt: str,
        route,
        response: dict[str, object],
        recent_interactions: list[dict[str, object]],
        active_thread: dict[str, object] | None,
    ) -> list[str]:
        if str(getattr(route, "mode", "") or "").strip() != "conversation":
            return []
        signals: list[str] = []
        normalized = PromptSurfaceBuilder.build(prompt).lookup_ready_text
        self_prompt = SelfOverviewSurfaceSupport.looks_like_self_referential_prompt(
            prompt=prompt,
            recent_interactions=recent_interactions,
        )
        if str(getattr(route, "kind", "") or "").strip() == "conversation.self_overview":
            signals.append("self_overview")
            if str(response.get("self_overview_source") or "").strip() == "social_follow_up":
                signals.append("social_self_follow_up")
        if isinstance(response.get("work_thread_continuity"), dict):
            signals.append("work_thread_continuity")
        if self_prompt and (
            any(
                normalized.startswith(prefix)
                for prefix in ("tell me about", "who are you", "what are you like", "what about you", "how about you")
            )
            or (
                normalized in {"and you", "what about you", "how about you"}
                and (
                    active_thread is not None
                    or SelfOverviewSurfaceSupport.has_recent_conversational_context(recent_interactions)
                )
            )
        ):
            signals.append("research_threshold_blocked")
        return signals

    @staticmethod
    def _response_tone_blend(*, prompt: str, route) -> dict[str, object]:
        normalized = PromptSurfaceBuilder.build(prompt).route_ready_text
        route_mode = str(getattr(route, "mode", "") or "")
        route_kind = str(getattr(route, "kind", "") or "")
        comparison = getattr(route, "comparison", None) or {}
        intent_weight = float(comparison.get("intent_weight") or 0.0)
        semantic_bonus = float(comparison.get("semantic_bonus") or 0.0)
        explanatory_weight = round(
            semantic_bonus + intent_weight + (0.15 if route_kind == "research.summary" else 0.0),
            3,
        )
        social_weight = 0.0
        if any(normalized.startswith(prefix) for prefix in ("hey ", "hi ", "yo ", "hello ")):
            social_weight += 0.2
        if any(token in normalized for token in (" lol", " haha", " lmao", " hehe")) or normalized.endswith(("lol", "haha", "lmao", "hehe")):
            social_weight += 0.2
        if "!" in prompt:
            social_weight += 0.04
        social_weight = round(min(0.45, social_weight), 3)
        explanatory_surface_fallback = InteractionService._is_explanatory_surface_fallback(
            normalized_prompt=normalized,
            route_mode=route_mode,
            route_kind=route_kind,
        )

        if route_mode == "conversation":
            tone_profile = "conversational"
        elif not explanatory_surface_fallback:
            tone_profile = "default"
        elif explanatory_weight >= 0.2 and social_weight >= 0.15:
            tone_profile = "casual_explanation"
        elif explanatory_weight >= 0.2:
            tone_profile = "formal_explanation"
        elif social_weight > explanatory_weight:
            tone_profile = "conversational"
        else:
            tone_profile = "default"

        return {
            "tone_profile": tone_profile,
            "explanatory_weight": explanatory_weight,
            "social_weight": social_weight,
        }

    @staticmethod
    def _is_broad_explanatory_prompt(normalized_prompt: str) -> bool:
        return ExplanatorySupportPolicy.evaluate(prompt=normalized_prompt).broad_explanatory_prompt

    @staticmethod
    def _is_explanatory_surface_fallback(
        *,
        normalized_prompt: str,
        route_mode: str,
        route_kind: str,
    ) -> bool:
        if route_mode != "research" or route_kind != "research.summary":
            return False
        return InteractionService._is_broad_explanatory_prompt(normalized_prompt)

    @staticmethod
    def _interaction_style(interaction_profile) -> str:
        return InteractionStylePolicy.interaction_style(interaction_profile)

    @staticmethod
    def _explanation_surface_style(*, interaction_profile, tone_profile: str) -> str:
        base_style = InteractionService._interaction_style(interaction_profile)
        return base_style

    @staticmethod
    def _knowledge_prompt_for_turn(
        *,
        prompt: str,
        recent_interactions: list[dict[str, object]],
    ) -> str:
        return ContinuationConfidencePolicy.evaluate(
            prompt=prompt,
            recent_interactions=recent_interactions,
        ).target_prompt

    @staticmethod
    def _infer_tool_input_path(
        *,
        active_thread: dict[str, object] | None,
        resolution_strategy: str,
    ) -> Path | None:
        if resolution_strategy not in {"tool_repeat_shorthand", "anh_tool_shorthand"}:
            return None
        if not active_thread:
            return None
        tool_context = active_thread.get("tool_context") or {}
        raw_input_path = tool_context.get("input_path")
        if not raw_input_path:
            return None
        return Path(str(raw_input_path))

    def _infer_tool_params(
        self,
        *,
        prompt: str,
        resolved_prompt: str,
        active_thread: dict[str, object] | None,
        resolution_strategy: str,
    ) -> dict[str, object] | None:
        if resolution_strategy not in {"tool_repeat_shorthand", "anh_tool_shorthand"}:
            return self._infer_live_tool_params(prompt=prompt, resolved_prompt=resolved_prompt)
        if active_thread:
            tool_context = active_thread.get("tool_context") or {}
            raw_params = tool_context.get("params")
            if isinstance(raw_params, dict):
                return dict(raw_params)
        return self._infer_live_tool_params(prompt=prompt, resolved_prompt=resolved_prompt)

    @staticmethod
    def _infer_live_tool_params(
        *,
        prompt: str,
        resolved_prompt: str,
    ) -> dict[str, object] | None:
        return InteractionFlowSupport.infer_live_tool_params(
            prompt=prompt,
            resolved_prompt=resolved_prompt,
        )

    @staticmethod
    def _infer_live_tool_input_path(
        *,
        prompt: str,
        resolved_prompt: str,
    ) -> Path | None:
        return InteractionFlowSupport.extract_live_tool_input_path(
            prompt=prompt,
            resolved_prompt=resolved_prompt,
        )

    @staticmethod
    def _extract_equation_text(prompt: str) -> str:
        return InteractionFlowSupport.extract_equation_text(prompt)

    def _diagnostic_follow_up_response(
        self,
        *,
        prompt: str,
        interaction_profile,
        recent_interactions: list[dict[str, object]],
        active_thread: dict[str, object] | None,
    ) -> dict[str, object] | None:
        normalized = PromptSurfaceBuilder.build(prompt).lookup_ready_text
        latest = recent_interactions[0] if recent_interactions else {}
        latest_response = latest.get("response") if isinstance(latest.get("response"), dict) else {}
        if not isinstance(latest_response, dict):
            latest_response = {}
        active_runtime = active_thread.get("runtime_diagnostic") if isinstance(active_thread, dict) else None
        diagnostic = latest_response.get("runtime_diagnostic")
        if not isinstance(diagnostic, dict):
            diagnostic = active_runtime if isinstance(active_runtime, dict) else {}
        runtime_status = latest_response.get("tool_runtime_status") if isinstance(latest_response.get("tool_runtime_status"), dict) else {}
        has_failure = bool(diagnostic) or (
            isinstance(runtime_status, dict)
            and str(runtime_status.get("failure_class") or "").strip() not in {"", "success"}
        ) or bool(latest_response.get("tool_execution_skipped"))
        if not has_failure:
            return None

        style = self._interaction_style(interaction_profile)
        failure_stage = str(diagnostic.get("failure_stage") or "execution").replace("_", " ").strip()
        missing_inputs = str(diagnostic.get("missing_inputs") or latest_response.get("tool_missing_inputs") or "").strip()
        safe_message = str(
            diagnostic.get("safe_message")
            or latest_response.get("user_facing_answer")
            or latest_response.get("reply")
            or latest_response.get("summary")
            or latest.get("summary")
            or ""
        ).strip()
        latest_prompt = str(latest.get("prompt") or "").strip()

        if normalized in {"what went wrong", "what failed", "why did that fail", "why did it fail", "what was the issue"}:
            if missing_inputs and failure_stage == "validation":
                if style == "direct":
                    reply = f"It was a validation issue. I was missing usable {missing_inputs}."
                elif style == "collab":
                    reply = f"It was a validation issue. I couldn't extract usable {missing_inputs}, so I stopped before a bad tool run."
                else:
                    reply = f"It was a validation issue. I didn't have usable {missing_inputs}, so I couldn't run it cleanly."
            elif safe_message:
                if style == "direct":
                    reply = f"It failed during {failure_stage}. {safe_message}"
                elif style == "collab":
                    reply = f"It looks like that hit a {failure_stage} issue. {safe_message}"
                else:
                    reply = f"That ran into a {failure_stage} issue. {safe_message}"
            else:
                reply = f"It failed during {failure_stage}." if style == "direct" else f"That ran into a {failure_stage} issue."
            if latest_prompt:
                reply = f"{reply} The last turn was: {latest_prompt}."
            return ConversationResponse(
                mode="conversation",
                kind="conversation.failure_follow_up",
                summary=reply.strip(),
                reply=reply.strip(),
            ).to_dict()

        if normalized in {"i see the issue", "yeah", "true", "right", "got it", "that makes sense"}:
            if style == "direct":
                reply = "Right. We can adjust the input or try a narrower version."
            elif style == "collab":
                reply = "Yeah, that's the issue I was running into. We can tighten the input or try the next clean pass together."
            else:
                reply = "Right, that's the issue. We can tighten the input or try a narrower next step."
            return ConversationResponse(
                mode="conversation",
                kind="conversation.failure_acknowledgment",
                summary=reply,
                reply=reply,
            ).to_dict()
        return None

    @staticmethod
    def _attach_conversation_access_metadata(
        *,
        response: dict[str, object],
        match_type: str,
        context_used: str,
        final_source: str,
    ) -> None:
        response["conversation_access"] = {
            "conversation_candidate_consulted": True,
            "conversation_match_type": str(match_type or "").strip() or None,
            "conversation_context_used": str(context_used or "").strip() or "none",
            "final_source": str(final_source or "").strip() or "conversation",
        }

    @staticmethod
    def _conversation_context_used(
        *,
        recent_interactions: list[dict[str, object]],
        active_thread: dict[str, object] | None,
    ) -> str:
        if isinstance(active_thread, dict) and active_thread:
            return "active_thread"
        if recent_interactions:
            return "recent_interactions"
        return "none"

    def _knowledge_self_assessment_response(
        self,
        *,
        prompt: str,
        interaction_profile,
    ) -> dict[str, object] | None:
        normalized = PromptSurfaceBuilder.build(prompt).lookup_ready_text
        if not any(
            phrase in normalized
            for phrase in (
                "how strong is your",
                "how good is your",
                "how much do you know about",
                "how strong are you in",
                "how good are you at",
            )
        ):
            return None
        if self.knowledge_service is None:
            return None
        aliases = {
            "astronomy": "astronomy",
            "space": "astronomy",
            "history": "history",
            "biology": "biology",
            "physics": "physics",
            "chemistry": "chemistry",
            "math": "math",
            "engineering": "engineering",
        }
        selected_domain = next((domain for token, domain in aliases.items() if token in normalized), None)
        if not selected_domain:
            return None
        overview = self.knowledge_service.overview()
        categories = overview.get("categories", []) if isinstance(overview, dict) else []
        match = next(
            (
                item for item in categories
                if isinstance(item, dict) and str(item.get("category") or "").strip().lower() == selected_domain
            ),
            None,
        )
        entry_count = int(match.get("entry_count") or 0) if isinstance(match, dict) else 0
        style = self._interaction_style(interaction_profile)
        if entry_count >= 8:
            strength = "one of my stronger local areas"
        elif entry_count >= 3:
            strength = "a decent local area for me"
        elif entry_count >= 1:
            strength = "something I can cover in a bounded way"
        else:
            strength = "not a strong local area for me yet"
        if style == "direct":
            reply = f"{selected_domain.title()} is {strength}. I can handle grounded overviews and follow-ups, but I'm still curated rather than exhaustive."
        elif style == "collab":
            reply = f"{selected_domain.title()} is {strength} for me. I can usually hold a real back-and-forth there, but I still stay inside what I can ground locally."
        else:
            reply = f"{selected_domain.title()} is {strength} for me. I can usually explain it and keep going on follow-ups, but I am still working from bounded local knowledge rather than full coverage."
        response = ConversationResponse(
            mode="conversation",
            kind="conversation.knowledge_self_assessment",
            summary=reply,
            reply=reply,
        ).to_dict()
        response["knowledge_self_assessment"] = {
            "domain": selected_domain,
            "entry_count": entry_count,
            "coverage_posture": strength,
        }
        return response

    def _ordered_request_response(
        self,
        *,
        prompt: str,
        interaction_profile,
    ) -> dict[str, object] | None:
        raw_prompt = str(prompt or "").strip()
        if not raw_prompt:
            return None
        numbered = [item.strip() for item in re.findall(r"(?:^|\n)\s*(?:[-*]|\d+[.)])\s+([^\n]+)", raw_prompt)]
        if len(numbered) < 2:
            lowered = " ".join(raw_prompt.lower().split())
            if lowered.count(" then ") >= 2:
                numbered = [item.strip(" .") for item in lowered.split(" then ") if item.strip(" .")]
        if len(numbered) < 2:
            return None
        style = self._interaction_style(interaction_profile)
        if style == "direct":
            lead = "I can take that in order:"
        elif style == "collab":
            lead = "Yeah, I can take that in order:"
        else:
            lead = "I can handle that in order:"
        lines = [lead]
        lines.extend(f"{index}. {item}" for index, item in enumerate(numbered[:5], start=1))
        reply = "\n".join(lines)
        response = ConversationResponse(
            mode="conversation",
            kind="conversation.ordered_request_ack",
            summary=reply,
            reply=reply,
        ).to_dict()
        response["ordered_request_items"] = numbered[:5]
        return response

    def _self_overview_response(
        self,
        *,
        prompt: str,
        interaction_profile,
        recent_interactions: list[dict[str, object]],
    ) -> dict[str, object] | None:
        return SelfOverviewSurfaceSupport.build_response(
            prompt=prompt,
            interaction_profile=interaction_profile,
            tool_map=self._tool_map(),
            knowledge_service=self.knowledge_service,
            recent_interactions=recent_interactions,
        )

    def _quick_arithmetic_response(
        self,
        *,
        prompt: str,
        interaction_profile,
        recent_interactions: list[dict[str, object]],
    ) -> dict[str, object] | None:
        return MathSurfaceSupport.build_response(
            prompt=prompt,
            interaction_profile=interaction_profile,
            recent_interactions=recent_interactions,
        )

    def _broad_concept_response(
        self,
        *,
        prompt: str,
        interaction_profile,
        recent_interactions: list[dict[str, object]],
    ) -> dict[str, object] | None:
        return KnowledgeSurfaceSupport.build_response(
            prompt=prompt,
            interaction_profile=interaction_profile,
            knowledge_service=self.knowledge_service,
            recent_interactions=recent_interactions,
        )

    def _tool_map(self) -> dict[str, list[str]]:
        if self._tool_map_cache is not None:
            return self._tool_map_cache
        try:
            self._tool_map_cache = self.tool_execution_service.registry.list_tools()
        except Exception:
            self._tool_map_cache = {}
        return self._tool_map_cache

    @staticmethod
    def _apply_tool_result_surface(
        *,
        response: dict[str, object],
        tool_result,
    ) -> None:
        tool_id = str(getattr(tool_result, "tool_id", "") or "").strip()
        capability = str(getattr(tool_result, "capability", "") or "").strip()
        status = str(getattr(tool_result, "status", "") or "").strip().lower()
        structured = getattr(tool_result, "structured_data", {})
        if not isinstance(structured, dict):
            structured = {}
        if tool_id == "design" and capability == "system_spec" and status == "ok":
            sections = InteractionService._design_spec_sections(structured)
            surfaced = "\n".join(sections).strip()
            if surfaced:
                response["user_facing_answer"] = surfaced
                response["summary"] = str(structured.get("summary") or sections[0]).strip()
                response["reply"] = surfaced
                response["domain_surface"] = {
                    "lane": "design",
                    "subject": str(structured.get("subject") or "").strip() or None,
                    "design_domain": str(structured.get("design_domain") or "").strip() or None,
                }
            return
        if tool_id == "content" and status in {"ok", "error"}:
            surfaced = InteractionService._content_tool_surface(
                capability=capability,
                structured=structured,
                summary=str(getattr(tool_result, "summary", "") or "").strip(),
            )
            if surfaced:
                response["user_facing_answer"] = surfaced
                response["summary"] = str(getattr(tool_result, "summary", surfaced) or surfaced).strip()
                response["reply"] = surfaced
                response["domain_surface"] = {
                    "lane": "content",
                    "platform": str(structured.get("platform") or "").strip() or None,
                    "style_profile": str(structured.get("style_profile") or "").strip() or None,
                    "failure_category": str(structured.get("failure_category") or "").strip() or None,
                }
            return
        if tool_id != "math" or capability != "solve_equation" or status != "ok":
            return

        variable = str(structured.get("variable") or "").strip()
        solution = structured.get("solution") or []
        if not isinstance(solution, list):
            return
        cleaned_solutions = [str(item).strip() for item in solution if str(item).strip()]
        if not variable or not cleaned_solutions:
            return

        joined = ", ".join(f"{variable} = {item}" for item in cleaned_solutions)
        surfaced = f"Solved equation for {variable}: {joined}"
        steps = InteractionService._clean_tool_steps(structured.get("steps"))
        if not steps:
            steps = InteractionService._generate_linear_equation_steps(
                equation=str(structured.get("equation") or "").strip(),
                variable=variable,
                solutions=cleaned_solutions,
            )
        if steps:
            surfaced = "\n".join([surfaced, *steps])
        response["user_facing_answer"] = surfaced
        response["summary"] = surfaced
        response["reply"] = surfaced
        response["domain_surface"] = {
            "lane": "math",
            "expression": str(structured.get("equation") or "").strip() or None,
            "answer": cleaned_solutions[0] if len(cleaned_solutions) == 1 else ", ".join(cleaned_solutions),
            "equation": str(structured.get("equation") or "").strip() or None,
            "variable": variable,
        }

    @staticmethod
    def _attach_tool_access_metadata(
        *,
        response: dict[str, object],
        tool_id: str | None,
        capability: str | None,
        execution_required: bool,
        final_source: str,
    ) -> None:
        response["tool_access"] = {
            "tool_candidate_consulted": True,
            "tool_id": str(tool_id or "").strip() or None,
            "capability": str(capability or "").strip() or None,
            "tool_execution_required": bool(execution_required),
            "final_source": str(final_source or "").strip() or "tool_route",
        }

    @staticmethod
    def _integrate_tool_execution_into_response(
        *,
        response: dict[str, object],
        tool_result,
        outcome,
        reasoning_state: ReasoningStateFrame,
    ) -> None:
        tool_id = str(getattr(tool_result, "tool_id", "") or "").strip()
        capability = str(getattr(tool_result, "capability", "") or "").strip()
        summary = str(getattr(tool_result, "summary", "") or "").strip()
        if outcome.failure_class != "success":
            diagnostics = dict(outcome.runtime_diagnostics)
            failure_stage = str(diagnostics.get("failure_stage") or "execution")
            exception_type = str(diagnostics.get("exception_type") or "").strip() or None
            response["tool_runtime_status"] = {
                "failure_class": outcome.failure_class,
                "runtime_diagnostics": diagnostics,
            }
            response["runtime_diagnostic"] = InteractionService._runtime_diagnostic(
                failure_stage=failure_stage,
                failure_class=str(outcome.failure_class or "execution_failure"),
                tool_id=tool_id or None,
                capability=capability or None,
                exception_type=exception_type,
                safe_message=summary,
                debug_details=diagnostics,
            )
            return

        if response.get("user_facing_answer"):
            response.setdefault(
                "tool_reasoning_bridge",
                {
                    "tool_id": tool_id or None,
                    "capability": capability or None,
                    "execution_status": outcome.execution_status,
                },
            )
            return

        integrated = InteractionService._tool_execution_summary_text(
            tool_id=tool_id,
            capability=capability,
            summary=summary,
            reasoning_state=reasoning_state,
        )
        if integrated:
            response["user_facing_answer"] = integrated
            response["reply"] = integrated
            response["summary"] = integrated
        response.setdefault(
            "domain_surface",
            {
                "lane": "tool",
                "tool_id": tool_id or None,
                "capability": capability or None,
            },
        )

    @staticmethod
    def _tool_execution_summary_text(
        *,
        tool_id: str,
        capability: str,
        summary: str,
        reasoning_state: ReasoningStateFrame,
    ) -> str:
        if summary:
            return summary
        tool_label = ".".join(part for part in (tool_id, capability) if part).strip(".")
        subject = str(reasoning_state.canonical_subject or reasoning_state.continuation_target or "").strip()
        if tool_label and subject:
            return f"I ran {tool_label} for {subject}."
        if tool_label:
            return f"I ran {tool_label}."
        return ""

    @staticmethod
    def _attach_capability_status(*, response: dict[str, object]) -> None:
        if response.get("capability_status"):
            return
        status = CapabilityContractService.response_status_for_payload(response)
        if status is not None:
            response["capability_status"] = status

    @staticmethod
    def _clean_tool_steps(raw_steps: object) -> list[str]:
        if not isinstance(raw_steps, list):
            return []
        cleaned: list[str] = []
        for item in raw_steps:
            text = str(item or "").strip()
            if not text:
                continue
            if not text.endswith((".", "!", "?")):
                text = f"{text}."
            cleaned.append(text)
        return cleaned

    @staticmethod
    def _generate_linear_equation_steps(
        *,
        equation: str,
        variable: str,
        solutions: list[str],
    ) -> list[str]:
        if not equation or not variable or len(solutions) != 1:
            return []
        parsed = InteractionService._parse_linear_equation(equation=equation, variable=variable)
        if parsed is None:
            return []
        coefficient, offset, right_side = parsed
        solution = solutions[0]
        steps: list[str] = []
        left_after_offset = f"{coefficient}{variable}" if coefficient != 1 else variable
        if offset > 0:
            steps.append(
                f"Subtract {offset} from both sides to get {left_after_offset} = {right_side - offset}."
            )
        elif offset < 0:
            steps.append(
                f"Add {abs(offset)} to both sides to get {left_after_offset} = {right_side - offset}."
            )
        if coefficient not in {0, 1}:
            steps.append(
                f"Divide both sides by {coefficient} to get {variable} = {solution}."
            )
        elif coefficient == -1:
            steps.append(f"Divide both sides by -1 to get {variable} = {solution}.")
        elif not steps:
            steps.append(f"That already leaves {variable} = {solution}.")
        return steps

    @staticmethod
    def _content_tool_surface(
        *,
        capability: str,
        structured: dict[str, object],
        summary: str,
    ) -> str:
        if capability == "generate_ideas":
            ideas = structured.get("ideas") or []
            if not isinstance(ideas, list) or not ideas:
                return summary
            lines = [summary or f"Generated {len(ideas)} content ideas."]
            for item in ideas[:5]:
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title") or "").strip()
                rationale = str(item.get("rationale") or "").strip()
                if title and rationale:
                    lines.append(f"- {title}: {rationale}")
                elif title:
                    lines.append(f"- {title}")
            return "\n".join(lines).strip()
        if capability == "generate_batch":
            items = structured.get("items") or []
            if not isinstance(items, list) or not items:
                return summary
            lines = [summary or f"Generated {len(items)} master content drafts."]
            for item in items[:4]:
                if not isinstance(item, dict):
                    continue
                topic = str(item.get("topic") or "").strip()
                hook = str(item.get("hook") or "").strip()
                if topic and hook:
                    lines.append(f"- {topic}: {hook}")
            return "\n".join(lines).strip()
        if capability == "format_platform":
            variant = structured.get("variant") or {}
            if not isinstance(variant, dict):
                return summary
            hook = str(variant.get("hook") or "").strip()
            caption = str(variant.get("caption") or "").strip()
            lines = [summary] if summary else []
            if hook:
                lines.append(f"Hook: {hook}")
            if caption:
                lines.append(f"Caption: {caption}")
            return "\n".join(line for line in lines if line).strip()
        return summary

    @staticmethod
    def _parse_linear_equation(*, equation: str, variable: str) -> tuple[int, int, int] | None:
        normalized = str(equation or "").replace(" ", "")
        pattern = re.compile(
            rf"^([+-]?\d*){re.escape(variable)}([+-]\d+)?=([+-]?\d+)$",
            flags=re.IGNORECASE,
        )
        match = pattern.match(normalized)
        if match is None:
            return None
        coefficient_text = str(match.group(1) or "").strip()
        offset_text = str(match.group(2) or "0").strip()
        right_text = str(match.group(3) or "").strip()
        if coefficient_text in {"", "+"}:
            coefficient = 1
        elif coefficient_text == "-":
            coefficient = -1
        else:
            coefficient = int(coefficient_text)
        return coefficient, int(offset_text), int(right_text)

    @staticmethod
    def _planning_prompt_hint(
        prompt: str,
        *,
        recent_interactions: list[dict[str, object]],
        active_thread: dict[str, object] | None,
    ) -> str:
        return DesignSurfaceSupport.rewrite_prompt(
            prompt=prompt,
            recent_interactions=recent_interactions,
            active_thread=active_thread,
        )

    def _ensure_design_concept_response(
        self,
        *,
        response: dict[str, object],
        prompt: str,
        resolved_prompt: str,
        route_kind: str,
    ) -> None:
        if str(route_kind or "").strip() != "planning.architecture":
            return
        steps = [str(item).strip() for item in response.get("steps") or [] if str(item).strip()]
        if any(step.startswith("Assumptions:") for step in steps):
            return
        synthesis = self.planner._design_synthesis(prompt=prompt, kind=route_kind) or self.planner._design_synthesis(
            prompt=resolved_prompt,
            kind=route_kind,
        )
        if not synthesis:
            return
        response["summary"] = str(synthesis.get("summary") or response.get("summary") or "").strip()
        response["steps"] = list(synthesis.get("steps") or response.get("steps") or [])
        response["next_action"] = str(
            synthesis.get("next_action") or response.get("next_action") or ""
        ).strip()

    def _maybe_run_design_spec_tool(
        self,
        *,
        prompt: str,
        route_kind: str,
        interaction_profile,
        session_id: str,
        run_root,
    ):
        if not DesignSurfaceSupport.should_use_design_tool(prompt=prompt, route_kind=route_kind):
            return None
        params = DesignSurfaceSupport.build_design_tool_params(
            prompt=prompt,
            interaction_style=InteractionStylePolicy.interaction_style(interaction_profile),
        )
        try:
            return self.tool_execution_service.run_tool(
                tool_id="design",
                capability="system_spec",
                params=params,
                session_id=session_id,
                run_root=run_root,
            )
        except Exception:
            return None

    @staticmethod
    def _apply_design_spec_to_planning_response(
        *,
        response: dict[str, object],
        design_spec_result,
    ) -> None:
        structured = getattr(design_spec_result, "structured_data", {})
        if not isinstance(structured, dict) or not structured:
            return
        response["summary"] = str(structured.get("summary") or response.get("summary") or "").strip()
        response["steps"] = InteractionService._design_spec_steps(structured)
        next_steps = structured.get("next_steps") if isinstance(structured.get("next_steps"), list) else []
        response["next_action"] = str((next_steps[0] if next_steps else response.get("next_action")) or "").strip()
        response["design_spec"] = dict(structured)
        response["design_tool"] = {
            "tool_id": getattr(design_spec_result, "tool_id", "design"),
            "capability": getattr(design_spec_result, "capability", "system_spec"),
            "status": getattr(design_spec_result, "status", "ok"),
        }

    @staticmethod
    def _design_spec_steps(structured: dict[str, object]) -> list[str]:
        steps: list[str] = []
        overview = str(structured.get("system_overview") or "").strip()
        if overview:
            steps.append(f"System overview: {overview}")
        for key, label in (
            ("components", "Components"),
            ("resources", "Resources"),
            ("constraints", "Constraints"),
            ("tradeoffs", "Tradeoffs"),
            ("failure_points", "Failure points"),
        ):
            values = structured.get(key)
            if isinstance(values, list) and values:
                joined = "; ".join(str(item).strip() for item in values if str(item).strip())
                if joined:
                    steps.append(f"{label}: {joined}")
        assumptions = structured.get("assumptions")
        if isinstance(assumptions, list) and assumptions:
            joined = "; ".join(str(item).strip() for item in assumptions if str(item).strip())
            if joined:
                steps.append(f"Assumptions: {joined}")
        return steps

    @staticmethod
    def _design_spec_sections(structured: dict[str, object]) -> list[str]:
        sections: list[str] = []
        summary = str(structured.get("summary") or "").strip()
        if summary:
            sections.append(summary)
        overview = str(structured.get("system_overview") or "").strip()
        if overview:
            sections.append(f"Overview: {overview}")
        for key, label in (
            ("components", "Components"),
            ("resources", "Resources"),
            ("constraints", "Constraints"),
            ("tradeoffs", "Tradeoffs"),
            ("failure_points", "Failure points"),
            ("next_steps", "Next steps"),
        ):
            values = structured.get(key)
            if isinstance(values, list) and values:
                joined = "; ".join(str(item).strip() for item in values if str(item).strip())
                if joined:
                    sections.append(f"{label}: {joined}")
        return sections

    @staticmethod
    def _join_human_list(items: list[str]) -> str:
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        return ", ".join(items[:-1] + [f"and {items[-1]}"])

    @staticmethod
    def _extract_trivial_arithmetic_expression(prompt: str) -> str | None:
        return MathSurfaceSupport.extract_expression(prompt)

    @staticmethod
    def _evaluate_trivial_arithmetic(expression: str) -> int | float | None:
        return MathSurfaceSupport.evaluate(expression)

