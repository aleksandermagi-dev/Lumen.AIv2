from __future__ import annotations

from collections import Counter
from typing import Any

from lumen.evaluation.evaluation_models import (
    EvaluationSurfaceReview,
    InteractionDecisionEvaluation,
)
from lumen.reasoning.reasoning_state import ReasoningStateFrame


class DecisionEvaluation:
    """Offline heuristic review of persisted decision surfaces."""

    SURFACES = (
        "route_quality",
        "intent_domain_quality",
        "memory_relevance_quality",
        "tool_use_justification_quality",
        "confidence_calibration_quality",
        "assistant_reply_quality",
        "supervised_support_quality",
    )

    def evaluate_record(self, record: dict[str, Any]) -> InteractionDecisionEvaluation:
        response = record.get("response") if isinstance(record.get("response"), dict) else {}
        reasoning_state = ReasoningStateFrame.from_mapping(
            response.get("reasoning_state") if isinstance(response.get("reasoning_state"), dict) else None
        )
        trainability_trace = (
            record.get("trainability_trace") if isinstance(record.get("trainability_trace"), dict) else {}
        )
        supervised_support_trace = (
            record.get("supervised_support_trace")
            if isinstance(record.get("supervised_support_trace"), dict)
            else {}
        )

        reviews = (
            self._evaluate_route_quality(record=record, trace=trainability_trace, reasoning_state=reasoning_state),
            self._evaluate_intent_domain_quality(record=record, trace=trainability_trace, reasoning_state=reasoning_state),
            self._evaluate_memory_relevance_quality(record=record, trace=trainability_trace, reasoning_state=reasoning_state),
            self._evaluate_tool_use_quality(trace=trainability_trace),
            self._evaluate_confidence_calibration(record=record, trace=trainability_trace, reasoning_state=reasoning_state),
            self._evaluate_assistant_reply_quality(record=record, response=response),
            self._evaluate_supervised_support(
                record=record,
                trace=trainability_trace,
                supervised_support_trace=supervised_support_trace,
            ),
        )
        counts = Counter(review.judgment for review in reviews)
        return InteractionDecisionEvaluation(
            session_id=self._clean_text(record.get("session_id")),
            interaction_path=self._clean_text(record.get("interaction_path")),
            created_at=self._clean_text(record.get("created_at")),
            mode=self._clean_text(record.get("mode")),
            kind=self._clean_text(record.get("kind")),
            summary=self._clean_text(record.get("summary")),
            overall_judgment=self._overall_judgment(reviews),
            surface_reviews=reviews,
            judgment_counts=dict(counts),
        )

    def _evaluate_route_quality(
        self,
        *,
        record: dict[str, Any],
        trace: dict[str, Any],
        reasoning_state: ReasoningStateFrame,
    ) -> EvaluationSurfaceReview:
        route_trace = (
            trace.get("route_recommendation_support")
            if isinstance(trace.get("route_recommendation_support"), dict)
            else {}
        )
        mode = self._clean_text(route_trace.get("mode")) or self._clean_text(record.get("mode"))
        kind = self._clean_text(route_trace.get("kind")) or self._clean_text(record.get("kind"))
        route_confidence = self._float_or_none(route_trace.get("route_confidence"))
        route_status = self._clean_text(route_trace.get("route_status")) or self._clean_text(record.get("route_status"))
        support_status = self._clean_text(route_trace.get("support_status")) or self._clean_text(record.get("support_status"))
        state_mode = self._clean_text(reasoning_state.route_decision.get("mode"))
        state_kind = self._clean_text(reasoning_state.route_decision.get("kind"))
        if not mode and not kind and not state_mode and not state_kind:
            return self._review(
                surface="route_quality",
                judgment="insufficient_evidence",
                score=None,
                rationale="No stable route metadata was persisted for offline review.",
            )
        aligned = self._route_pair_aligned(mode, kind) and self._route_pair_aligned(state_mode or mode, state_kind or kind)
        if mode and record.get("mode") and mode != record.get("mode"):
            aligned = False
        if kind and record.get("kind") and kind != record.get("kind"):
            aligned = False
        if aligned and route_status in {"grounded", "stable", "revised"} and support_status in {"supported", "strongly_supported", "moderately_supported"} and (route_confidence is None or route_confidence >= 0.65):
            return self._review(
                surface="route_quality",
                judgment="correct",
                score=0.9 if route_confidence is None else max(0.8, route_confidence),
                rationale="Route metadata, route status, and downstream mode/kind stayed aligned.",
                evidence={
                    "mode": mode,
                    "kind": kind,
                    "route_status": route_status,
                    "support_status": support_status,
                    "route_confidence": route_confidence,
                },
            )
        if not aligned or route_status in {"under_tension", "unresolved"}:
            return self._review(
                surface="route_quality",
                judgment="incorrect",
                score=0.2,
                rationale="Persisted route metadata conflicted with the selected route or remained unresolved.",
                evidence={
                    "mode": mode,
                    "kind": kind,
                    "state_mode": state_mode,
                    "state_kind": state_kind,
                    "route_status": route_status,
                    "support_status": support_status,
                },
            )
        return self._review(
            surface="route_quality",
            judgment="weak",
            score=0.55,
            rationale="A route was present, but support or confidence signals were not strong enough to treat it as clearly correct.",
            evidence={
                "mode": mode,
                "kind": kind,
                "route_status": route_status,
                "support_status": support_status,
                "route_confidence": route_confidence,
            },
        )

    def _evaluate_intent_domain_quality(
        self,
        *,
        record: dict[str, Any],
        trace: dict[str, Any],
        reasoning_state: ReasoningStateFrame,
    ) -> EvaluationSurfaceReview:
        intent_trace = (
            trace.get("intent_domain_classification")
            if isinstance(trace.get("intent_domain_classification"), dict)
            else {}
        )
        style_trace = (
            trace.get("response_style_selection")
            if isinstance(trace.get("response_style_selection"), dict)
            else {}
        )
        domain = self._clean_text(intent_trace.get("intent_domain")) or self._clean_text(record.get("intent_domain")) or self._clean_text(reasoning_state.intent_domain)
        style_domain = self._clean_text(style_trace.get("intent_domain"))
        route_mode = self._clean_text(intent_trace.get("route_mode")) or self._clean_text(record.get("mode"))
        confidence = self._float_or_none(intent_trace.get("intent_domain_confidence"))
        response_depth = self._clean_text(intent_trace.get("response_depth"))
        if not domain:
            return self._review(
                surface="intent_domain_quality",
                judgment="insufficient_evidence",
                score=None,
                rationale="No persisted intent-domain label was available for review.",
            )
        if style_domain and style_domain != domain:
            return self._review(
                surface="intent_domain_quality",
                judgment="incorrect",
                score=0.25,
                rationale="Intent-domain classification conflicted with the response-style surface.",
                evidence={
                    "intent_domain": domain,
                    "response_style_domain": style_domain,
                    "route_mode": route_mode,
                },
            )
        if confidence is not None and confidence >= 0.7 and (style_domain == domain or style_domain is None):
            return self._review(
                surface="intent_domain_quality",
                judgment="correct",
                score=max(0.8, confidence),
                rationale="Intent-domain classification was internally aligned and had strong confidence.",
                evidence={
                    "intent_domain": domain,
                    "route_mode": route_mode,
                    "response_depth": response_depth,
                    "intent_domain_confidence": confidence,
                },
            )
        return self._review(
            surface="intent_domain_quality",
            judgment="weak",
            score=0.55 if confidence is None else min(0.69, max(confidence, 0.45)),
            rationale="Intent-domain classification was present, but alignment or confidence was not strong enough for a clear pass.",
            evidence={
                "intent_domain": domain,
                "response_style_domain": style_domain,
                "route_mode": route_mode,
                "intent_domain_confidence": confidence,
            },
        )

    def _evaluate_memory_relevance_quality(
        self,
        *,
        record: dict[str, Any],
        trace: dict[str, Any],
        reasoning_state: ReasoningStateFrame,
    ) -> EvaluationSurfaceReview:
        memory_trace = (
            trace.get("memory_relevance_ranking")
            if isinstance(trace.get("memory_relevance_ranking"), dict)
            else {}
        )
        selected_count = self._int_or_zero(memory_trace.get("selected_count"))
        rejected_count = self._int_or_zero(memory_trace.get("rejected_count"))
        used_count = self._int_or_zero(memory_trace.get("memory_context_used_count"))
        selected_labels = self._string_list(memory_trace.get("selected_labels"))
        rejected_labels = self._string_list(memory_trace.get("rejected_labels"))
        used_labels = self._string_list(memory_trace.get("memory_context_used_labels"))
        if selected_count == 0 and rejected_count == 0 and used_count == 0:
            return self._review(
                surface="memory_relevance_quality",
                judgment="insufficient_evidence",
                score=None,
                rationale="No memory selection diagnostics were persisted for this interaction.",
            )
        if used_count > selected_count or (selected_count > 0 and rejected_count > selected_count * 2):
            return self._review(
                surface="memory_relevance_quality",
                judgment="incorrect",
                score=0.2,
                rationale="Memory usage suggests noise bleed or inconsistent retrieval selection.",
                evidence={
                    "selected_count": selected_count,
                    "rejected_count": rejected_count,
                    "memory_context_used_count": used_count,
                    "selected_labels": selected_labels,
                    "rejected_labels": rejected_labels,
                    "used_labels": used_labels,
                },
            )
        if selected_count > 0 and used_count > 0 and rejected_count <= selected_count:
            return self._review(
                surface="memory_relevance_quality",
                judgment="correct",
                score=0.85,
                rationale="Selected memory was actually used and rejection volume stayed bounded.",
                evidence={
                    "selected_count": selected_count,
                    "rejected_count": rejected_count,
                    "memory_context_used_count": used_count,
                    "selected_labels": selected_labels,
                    "used_labels": used_labels,
                    "active_memory_context_labels": [
                        self._clean_text(item.get("label"))
                        for item in reasoning_state.memory_context_used
                        if isinstance(item, dict) and self._clean_text(item.get("label"))
                    ],
                },
            )
        return self._review(
            surface="memory_relevance_quality",
            judgment="weak",
            score=0.5,
            rationale="Memory retrieval showed some signal, but not enough evidence to call the ranking clearly strong.",
            evidence={
                "selected_count": selected_count,
                "rejected_count": rejected_count,
                "memory_context_used_count": used_count,
                "selected_labels": selected_labels,
                "rejected_labels": rejected_labels,
            },
        )

    def _evaluate_tool_use_quality(
        self,
        *,
        trace: dict[str, Any],
    ) -> EvaluationSurfaceReview:
        tool_trace = (
            trace.get("tool_use_decision_support")
            if isinstance(trace.get("tool_use_decision_support"), dict)
            else {}
        )
        should_use_tool = tool_trace.get("should_use_tool")
        selected_tool = self._clean_text(tool_trace.get("selected_tool"))
        execution_attempted = bool(tool_trace.get("execution_attempted", False))
        execution_status = self._clean_text(tool_trace.get("execution_status"))
        rationale = self._clean_text(tool_trace.get("rationale"))
        if should_use_tool is None and not selected_tool and not execution_attempted and not rationale:
            return self._review(
                surface="tool_use_justification_quality",
                judgment="insufficient_evidence",
                score=None,
                rationale="No tool-threshold trace was available for review.",
            )
        if should_use_tool is False and execution_attempted:
            return self._review(
                surface="tool_use_justification_quality",
                judgment="incorrect",
                score=0.2,
                rationale="A tool execution happened even though the persisted tool decision said not to use one.",
                evidence={
                    "should_use_tool": should_use_tool,
                    "selected_tool": selected_tool,
                    "execution_attempted": execution_attempted,
                    "execution_status": execution_status,
                },
            )
        if should_use_tool and (not selected_tool or not rationale):
            return self._review(
                surface="tool_use_justification_quality",
                judgment="incorrect",
                score=0.25,
                rationale="Tool use was recommended without enough persisted justification or target selection.",
                evidence={
                    "should_use_tool": should_use_tool,
                    "selected_tool": selected_tool,
                    "rationale": rationale,
                },
            )
        if should_use_tool and execution_attempted and execution_status == "ok":
            return self._review(
                surface="tool_use_justification_quality",
                judgment="correct",
                score=0.9,
                rationale="Tool use was justified, executed, and verified successfully.",
                evidence={
                    "should_use_tool": should_use_tool,
                    "selected_tool": selected_tool,
                    "execution_status": execution_status,
                },
            )
        if should_use_tool is False and not execution_attempted:
            return self._review(
                surface="tool_use_justification_quality",
                judgment="correct",
                score=0.85,
                rationale="The system correctly stayed local when the tool threshold did not justify execution.",
                evidence={
                    "should_use_tool": should_use_tool,
                    "execution_attempted": execution_attempted,
                },
            )
        return self._review(
            surface="tool_use_justification_quality",
            judgment="weak",
            score=0.5,
            rationale="Tool-threshold metadata was present, but execution outcome did not provide a strong confirmation.",
            evidence={
                "should_use_tool": should_use_tool,
                "selected_tool": selected_tool,
                "execution_attempted": execution_attempted,
                "execution_status": execution_status,
            },
        )

    def _evaluate_confidence_calibration(
        self,
        *,
        record: dict[str, Any],
        trace: dict[str, Any],
        reasoning_state: ReasoningStateFrame,
    ) -> EvaluationSurfaceReview:
        confidence_trace = (
            trace.get("confidence_calibration_support")
            if isinstance(trace.get("confidence_calibration_support"), dict)
            else {}
        )
        confidence_tier = self._clean_text(confidence_trace.get("confidence_tier")) or self._clean_text(reasoning_state.confidence_tier)
        confidence_score = self._float_or_none(confidence_trace.get("confidence_score"))
        posture = self._clean_text(confidence_trace.get("confidence_posture")) or self._clean_text(record.get("confidence_posture"))
        support_status = self._clean_text(confidence_trace.get("support_status")) or self._clean_text(record.get("support_status"))
        route_status = self._clean_text(confidence_trace.get("route_status")) or self._clean_text(record.get("route_status"))
        memory_signal = bool(confidence_trace.get("memory_signal_present", False))
        tool_verified = bool(confidence_trace.get("tool_verified", False))
        if not confidence_tier and confidence_score is None and not posture:
            return self._review(
                surface="confidence_calibration_quality",
                judgment="insufficient_evidence",
                score=None,
                rationale="Confidence calibration metadata was not available for this interaction.",
            )
        strong_signals = support_status in {"supported", "strongly_supported"} or route_status in {"grounded", "stable"} or tool_verified
        weak_signals = support_status in {"insufficiently_grounded"} or route_status in {"under_tension", "unresolved"}
        if confidence_tier == "high" and weak_signals and not memory_signal and not tool_verified:
            return self._review(
                surface="confidence_calibration_quality",
                judgment="incorrect",
                score=0.2,
                rationale="The visible confidence remained high despite weak route and evidence signals.",
                evidence={
                    "confidence_tier": confidence_tier,
                    "confidence_score": confidence_score,
                    "confidence_posture": posture,
                    "support_status": support_status,
                    "route_status": route_status,
                },
            )
        if confidence_tier == "low" and strong_signals:
            return self._review(
                surface="confidence_calibration_quality",
                judgment="incorrect",
                score=0.25,
                rationale="Confidence was understated relative to the available support and verification signals.",
                evidence={
                    "confidence_tier": confidence_tier,
                    "confidence_score": confidence_score,
                    "confidence_posture": posture,
                    "support_status": support_status,
                    "route_status": route_status,
                    "tool_verified": tool_verified,
                },
            )
        if (confidence_tier == "high" and strong_signals) or (confidence_tier == "low" and weak_signals):
            return self._review(
                surface="confidence_calibration_quality",
                judgment="correct",
                score=0.86,
                rationale="Confidence tier and support signals moved in the same direction.",
                evidence={
                    "confidence_tier": confidence_tier,
                    "confidence_score": confidence_score,
                    "confidence_posture": posture,
                    "support_status": support_status,
                    "route_status": route_status,
                    "memory_signal_present": memory_signal,
                    "tool_verified": tool_verified,
                },
            )
        return self._review(
            surface="confidence_calibration_quality",
            judgment="weak",
            score=0.55,
            rationale="Confidence cues were present, but the supporting signals were mixed.",
            evidence={
                "confidence_tier": confidence_tier,
                "confidence_score": confidence_score,
                "confidence_posture": posture,
                "support_status": support_status,
                "route_status": route_status,
                "memory_signal_present": memory_signal,
                "tool_verified": tool_verified,
            },
        )

    def _evaluate_supervised_support(
        self,
        *,
        record: dict[str, Any],
        trace: dict[str, Any],
        supervised_support_trace: dict[str, Any],
    ) -> EvaluationSurfaceReview:
        supervised_trace = (
            trace.get("supervised_decision_support")
            if isinstance(trace.get("supervised_decision_support"), dict)
            else {}
        )
        enabled = bool(supervised_trace.get("enabled", supervised_support_trace.get("enabled", False)))
        recommended_surfaces = self._string_list(
            supervised_trace.get("recommended_surfaces") or (supervised_support_trace.get("recommendations") or {}).keys()
        )
        applied_surfaces = self._string_list(
            supervised_trace.get("applied_surfaces") or supervised_support_trace.get("applied_surfaces")
        )
        authority_preserved = bool(
            supervised_trace.get(
                "deterministic_authority_preserved",
                supervised_support_trace.get("deterministic_authority_preserved", True),
            )
        )
        if not enabled and not recommended_surfaces and not applied_surfaces:
            return self._review(
                surface="supervised_support_quality",
                judgment="insufficient_evidence",
                score=None,
                rationale="No supervised-support recommendations were persisted for review.",
            )
        if not authority_preserved or any(surface not in recommended_surfaces for surface in applied_surfaces):
            return self._review(
                surface="supervised_support_quality",
                judgment="incorrect",
                score=0.2,
                rationale="Supervised support appears to have exceeded its bounded advisory role.",
                evidence={
                    "enabled": enabled,
                    "recommended_surfaces": recommended_surfaces,
                    "applied_surfaces": applied_surfaces,
                    "deterministic_authority_preserved": authority_preserved,
                },
            )
        if authority_preserved and recommended_surfaces:
            return self._review(
                surface="supervised_support_quality",
                judgment="correct",
                score=0.88,
                rationale="Supervised support stayed bounded and preserved deterministic authority.",
                evidence={
                    "enabled": enabled,
                    "recommended_surfaces": recommended_surfaces,
                    "applied_surfaces": applied_surfaces,
                    "deterministic_authority_preserved": authority_preserved,
                },
            )
        return self._review(
            surface="supervised_support_quality",
            judgment="weak",
            score=0.5,
            rationale="Supervised support remained bounded, but there was too little recommendation activity for a stronger judgment.",
            evidence={
                "enabled": enabled,
                "recommended_surfaces": recommended_surfaces,
                "applied_surfaces": applied_surfaces,
                "deterministic_authority_preserved": authority_preserved,
                "record_mode": self._clean_text(record.get("mode")),
            },
        )

    def _evaluate_assistant_reply_quality(
        self,
        *,
        record: dict[str, Any],
        response: dict[str, Any],
    ) -> EvaluationSurfaceReview:
        assistant_posture = (
            response.get("assistant_quality_posture")
            if isinstance(response.get("assistant_quality_posture"), dict)
            else {}
        )
        context_snapshot = (
            response.get("assistant_context_snapshot")
            if isinstance(response.get("assistant_context_snapshot"), dict)
            else {}
        )
        voice_profile_payload = (
            response.get("assistant_voice_profile")
            if isinstance(response.get("assistant_voice_profile"), dict)
            else {}
        )
        project_snapshot = (
            response.get("project_context_snapshot")
            if isinstance(response.get("project_context_snapshot"), dict)
            else {}
        )
        provider_inference = (
            response.get("provider_inference")
            if isinstance(response.get("provider_inference"), dict)
            else {}
        )
        memory_retrieval = (
            response.get("memory_retrieval")
            if isinstance(response.get("memory_retrieval"), dict)
            else {}
        )
        conversation_beat = (
            response.get("conversation_beat")
            if isinstance(response.get("conversation_beat"), dict)
            else {}
        )
        profile = self._clean_text(assistant_posture.get("profile"))
        direct_answer_first = assistant_posture.get("direct_answer_first")
        clarification_restraint = assistant_posture.get("clarification_restraint")
        memory_budget = self._int_or_zero(assistant_posture.get("memory_budget"))
        style_mode = self._clean_text(assistant_posture.get("style_mode")) or self._clean_text(voice_profile_payload.get("style_mode"))
        voice_profile = self._clean_text(assistant_posture.get("voice_profile")) or self._clean_text(voice_profile_payload.get("voice_profile"))
        tone_signature = self._clean_text(assistant_posture.get("tone_signature")) or self._clean_text(voice_profile_payload.get("tone_signature"))
        reasoning_depth = self._clean_text(assistant_posture.get("reasoning_depth")) or self._clean_text(voice_profile_payload.get("reasoning_depth"))
        reasoning_depth_separate = assistant_posture.get("reasoning_depth_separate", voice_profile_payload.get("reasoning_depth_separate"))
        memory_item_count = self._int_or_zero(context_snapshot.get("memory_item_count"))
        recent_turn_count = self._int_or_zero(context_snapshot.get("recent_turn_count"))
        project_context_active = bool(project_snapshot.get("project_context_active", False))
        project_continuity_mode = self._clean_text(project_snapshot.get("continuity_mode")) or "general_chat"
        project_context_source = self._clean_text(project_snapshot.get("continuity_source")) or "none"
        work_thread_active = bool(assistant_posture.get("work_thread_continuity_active", False))
        work_thread_intent = self._clean_text(assistant_posture.get("work_thread_intent")) or "none"
        work_thread_source = self._clean_text(assistant_posture.get("work_thread_source")) or "none"
        project_recent_turn_count = self._int_or_zero(project_snapshot.get("project_recent_turn_count"))
        secondary_project_memory_count = self._int_or_zero(project_snapshot.get("secondary_project_memory_count"))
        route_mode = self._clean_text(context_snapshot.get("route_mode")) or self._clean_text(record.get("mode"))
        prompt_class = self._clean_text(context_snapshot.get("prompt_class")) or "general"
        response_path = self._clean_text(provider_inference.get("response_path"))
        provider_id = self._clean_text(provider_inference.get("provider_id"))
        selected_memory = memory_retrieval.get("selected") if isinstance(memory_retrieval.get("selected"), list) else []
        personal_memory_count = sum(
            1
            for item in selected_memory
            if isinstance(item, dict)
            and (
                self._clean_text(item.get("source")) == "personal_memory"
                or self._clean_text(item.get("memory_kind")) in {"durable_user_memory", "profile", "preference"}
            )
        )
        explicit_memory_recall = bool(memory_retrieval.get("recall_prompt"))
        boundary_signals = self._string_list(
            assistant_posture.get("conversation_boundary_signals")
            or response.get("assistant_boundary_signals")
        )
        conversation_depth = self._int_or_zero(
            conversation_beat.get("conversation_depth") or assistant_posture.get("conversation_depth")
        )
        repetition_risk = self._clean_text(
            conversation_beat.get("response_repetition_risk")
            or assistant_posture.get("response_repetition_risk")
        )
        follow_up_offer_allowed = conversation_beat.get(
            "follow_up_offer_allowed",
            assistant_posture.get("follow_up_offer_allowed"),
        )
        long_chat_stamina = (
            conversation_beat.get("long_chat_stamina")
            if isinstance(conversation_beat.get("long_chat_stamina"), dict)
            else assistant_posture.get("long_chat_stamina")
            if isinstance(assistant_posture.get("long_chat_stamina"), dict)
            else {}
        )
        if (
            not profile
            and direct_answer_first is None
            and clarification_restraint is None
            and not context_snapshot
            and not voice_profile_payload
            and not project_snapshot
            and not provider_inference
        ):
            return self._review(
                surface="assistant_reply_quality",
                judgment="insufficient_evidence",
                score=None,
                rationale="No assistant-quality posture was persisted for this interaction.",
            )
        if (memory_budget > 0 and memory_item_count > memory_budget) or recent_turn_count > 3:
            return self._review(
                surface="assistant_reply_quality",
                judgment="incorrect",
                score=0.2,
                rationale="Assistant context assembly exceeded the bounded chat posture.",
                evidence={
                    "profile": profile,
                    "memory_budget": memory_budget,
                    "memory_item_count": memory_item_count,
                    "recent_turn_count": recent_turn_count,
                    "route_mode": route_mode,
                },
            )
        if (
            project_context_active
            and (
                project_continuity_mode != "live_project"
                or project_context_source == "none"
                or project_recent_turn_count > 2
                or secondary_project_memory_count > 1
            )
        ):
            return self._review(
                surface="assistant_reply_quality",
                judgment="incorrect",
                score=0.25,
                rationale="Project-aware assistant continuity was active, but the persisted project context signals were not bounded or source-grounded.",
                evidence={
                    "profile": profile,
                    "project_context_active": project_context_active,
                    "project_continuity_mode": project_continuity_mode,
                    "project_context_source": project_context_source,
                    "project_recent_turn_count": project_recent_turn_count,
                    "secondary_project_memory_count": secondary_project_memory_count,
                },
            )
        if work_thread_active and (
            project_continuity_mode != "live_project"
            or project_context_source != "active_thread"
            or work_thread_source != "active_thread"
            or work_thread_intent == "none"
        ):
            return self._review(
                surface="assistant_reply_quality",
                judgment="incorrect",
                score=0.25,
                rationale="Work-thread continuity was active, but it was not anchored to the live active thread metadata.",
                evidence={
                    "work_thread_active": work_thread_active,
                    "work_thread_intent": work_thread_intent,
                    "work_thread_source": work_thread_source,
                    "project_continuity_mode": project_continuity_mode,
                    "project_context_source": project_context_source,
                },
            )
        if route_mode == "research" and prompt_class == "self_referential":
            return self._review(
                surface="assistant_reply_quality",
                judgment="incorrect",
                score=0.2,
                rationale="Casual self-referential assistant chat escalated into research instead of staying conversational.",
                evidence={
                    "profile": profile,
                    "prompt_class": prompt_class,
                    "route_mode": route_mode,
                    "route_kind": self._clean_text(context_snapshot.get("route_kind")) or self._clean_text(record.get("kind")),
                    "boundary_signals": boundary_signals,
                },
            )
        if (
            route_mode == "conversation"
            and conversation_depth >= 5
            and repetition_risk == "high"
            and follow_up_offer_allowed is not False
        ):
            return self._review(
                surface="assistant_reply_quality",
                judgment="incorrect",
                score=0.25,
                rationale="Long-chat repetition risk was high, but follow-up offers were still allowed.",
                evidence={
                    "conversation_depth": conversation_depth,
                    "response_repetition_risk": repetition_risk,
                    "follow_up_offer_allowed": follow_up_offer_allowed,
                    "long_chat_stamina": dict(long_chat_stamina),
                },
            )
        if (
            route_mode == "conversation"
            and not explicit_memory_recall
            and personal_memory_count > 1
        ):
            return self._review(
                surface="assistant_reply_quality",
                judgment="incorrect",
                score=0.25,
                rationale="Ordinary conversation over-injected personal memory instead of keeping long-term familiarity bounded.",
                evidence={
                    "personal_memory_count": personal_memory_count,
                    "explicit_memory_recall": explicit_memory_recall,
                    "memory_item_count": memory_item_count,
                    "memory_budget": memory_budget,
                },
            )
        if not style_mode or not voice_profile or not tone_signature or reasoning_depth_separate is not True:
            return self._review(
                surface="assistant_reply_quality",
                judgment="weak",
                score=0.45,
                rationale="Assistant reply metadata did not cleanly separate personality style from reasoning depth.",
                evidence={
                    "style_mode": style_mode,
                    "voice_profile": voice_profile,
                    "tone_signature": tone_signature,
                    "reasoning_depth": reasoning_depth,
                    "reasoning_depth_separate": reasoning_depth_separate,
                },
            )
        if (
            route_mode == "conversation"
            and profile
            and direct_answer_first is True
            and clarification_restraint is True
            and response_path == "general_assistant"
            and provider_id
            and (prompt_class != "self_referential" or bool(boundary_signals))
        ):
            return self._review(
                surface="assistant_reply_quality",
                judgment="correct",
                score=0.88,
                rationale="Conversation reply posture stayed explicit, bounded, provider-aware, and project continuity stayed controlled when active.",
                evidence={
                    "profile": profile,
                    "memory_budget": memory_budget,
                    "memory_item_count": memory_item_count,
                    "personal_memory_count": personal_memory_count,
                    "recent_turn_count": recent_turn_count,
                    "provider_id": provider_id,
                    "response_path": response_path,
                    "project_context_active": project_context_active,
                    "project_continuity_mode": project_continuity_mode,
                    "project_context_source": project_context_source,
                    "work_thread_active": work_thread_active,
                    "work_thread_intent": work_thread_intent,
                    "work_thread_source": work_thread_source,
                    "style_mode": style_mode,
                    "voice_profile": voice_profile,
                    "tone_signature": tone_signature,
                    "reasoning_depth": reasoning_depth,
                    "prompt_class": prompt_class,
                    "boundary_signals": boundary_signals,
                    "conversation_depth": conversation_depth,
                    "response_repetition_risk": repetition_risk,
                    "follow_up_offer_allowed": follow_up_offer_allowed,
                },
            )
        return self._review(
            surface="assistant_reply_quality",
            judgment="weak",
            score=0.55,
            rationale="Assistant-quality metadata was present, but the persisted posture was not strong enough for a clear pass.",
            evidence={
                "profile": profile,
                "direct_answer_first": direct_answer_first,
                "clarification_restraint": clarification_restraint,
                "memory_budget": memory_budget,
                "memory_item_count": memory_item_count,
                "recent_turn_count": recent_turn_count,
                "provider_id": provider_id,
                "response_path": response_path,
                "project_context_active": project_context_active,
                "project_continuity_mode": project_continuity_mode,
                "project_context_source": project_context_source,
                "style_mode": style_mode,
                "voice_profile": voice_profile,
                "tone_signature": tone_signature,
                "reasoning_depth": reasoning_depth,
                "prompt_class": prompt_class,
                "boundary_signals": boundary_signals,
            },
        )

    @staticmethod
    def _overall_judgment(reviews: tuple[EvaluationSurfaceReview, ...]) -> str:
        counts = Counter(review.judgment for review in reviews)
        if counts.get("incorrect", 0) >= 2:
            return "incorrect"
        if counts.get("correct", 0) >= 4 and counts.get("incorrect", 0) == 0:
            return "correct"
        if counts.get("insufficient_evidence", 0) >= len(reviews) // 2:
            return "insufficient_evidence"
        return "weak"

    @staticmethod
    def _route_pair_aligned(mode: str | None, kind: str | None) -> bool:
        if not mode or not kind:
            return True
        prefix = kind.split(".", 1)[0]
        return prefix == mode

    @staticmethod
    def _review(
        *,
        surface: str,
        judgment: str,
        score: float | None,
        rationale: str,
        evidence: dict[str, object] | None = None,
    ) -> EvaluationSurfaceReview:
        return EvaluationSurfaceReview(
            surface=surface,
            judgment=judgment,
            score=score,
            rationale=rationale,
            evidence=dict(evidence or {}),
        )

    @staticmethod
    def _clean_text(value: object) -> str | None:
        text = str(value or "").strip()
        return text or None

    @staticmethod
    def _float_or_none(value: object) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _int_or_zero(value: object) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _string_list(value: object) -> list[str]:
        items = value if isinstance(value, (list, tuple, set)) else []
        return [item for item in (DecisionEvaluation._clean_text(entry) for entry in items) if item]
