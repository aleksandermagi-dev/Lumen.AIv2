from __future__ import annotations

from pathlib import Path
from typing import Any

from lumen.app.models import InteractionProfile
from lumen.nlu.focus_resolution import FocusResolutionSupport
from lumen.nlu.prompt_surface_builder import PromptSurfaceBuilder
from lumen.reasoning.confidence_gradient import ConfidenceGradient
from lumen.reasoning.reasoning_state import (
    ExecutionOutcome,
    ModeBehaviorProfile,
    ReasoningStateFrame,
)


class ReasoningStateService:
    """Builds and incrementally updates the lightweight reasoning spine for a turn."""

    _SIMPLE_FOLLOW_UP_PROMPTS = {
        "break it down",
        "break that down",
        "break this down",
        "explain it simply",
        "explain that simply",
        "explain this simply",
        "simplify it",
        "simplify that",
        "simplify this",
        "in simple terms",
    }
    _DEEP_FOLLOW_UP_PROMPTS = {
        "go deeper",
        "go deeper on that",
        "go deeper on this",
        "explain it deeply",
        "explain that deeply",
        "explain this deeply",
        "explain it in depth",
        "explain that in depth",
        "explain this in depth",
        "more detail",
        "more details",
        "in more detail",
    }

    def initialize(
        self,
        *,
        prompt: str,
        active_thread: dict[str, object] | None,
        interaction_profile: InteractionProfile | None,
        input_path: Path | None,
    ) -> ReasoningStateFrame:
        prior = ReasoningStateFrame.from_mapping(
            active_thread.get("reasoning_state") if isinstance(active_thread, dict) else None
        )
        profile = interaction_profile or InteractionProfile.default()
        normalized = PromptSurfaceBuilder.build(prompt).route_ready_text
        subject = FocusResolutionSupport.subject_focus(normalized).focus
        if self._is_explanation_follow_up_prompt(normalized):
            subject = prior.canonical_subject or subject
        return self._ensure_mode_state(prior.with_updates(
            current_task=normalized or prior.current_task,
            resolved_prompt=normalized or prior.resolved_prompt,
            canonical_subject=subject or prior.canonical_subject,
            selected_mode=profile.normalize_interaction_style(profile.interaction_style),
            mode_behavior=self._mode_behavior(profile).to_dict(),
            active_domain=prior.active_domain,
            ambiguity_status="pending",
            turn_status="intake",
            current_path=prior.current_path,
            current_intent=prior.current_intent,
            tool_context=dict(prior.tool_context or {}),
            known_context_summary=self._known_context_summary(active_thread=active_thread, input_path=input_path),
            explanation_strategy=self._infer_explanation_strategy(normalized),
            comparison_targets=self._comparison_targets(normalized),
        ))

    def rewrite_prompt_for_stateful_follow_up(
        self,
        *,
        prompt: str,
        state: ReasoningStateFrame,
    ) -> tuple[str, ReasoningStateFrame]:
        normalized = PromptSurfaceBuilder.build(prompt).route_ready_text
        if not normalized:
            return prompt, state
        rewritten = self._stateful_explanation_prompt(normalized=normalized, state=state)
        if not rewritten:
            return prompt, state
        updated_state = state.with_updates(
            current_task=rewritten,
            resolved_prompt=rewritten,
            explanation_strategy=self._infer_explanation_strategy(rewritten),
            turn_status="stateful_follow_up",
        )
        return rewritten, updated_state

    def apply_route(
        self,
        *,
        state: ReasoningStateFrame,
        route_authority,
        route,
        resolved_prompt: str,
        prompt_understanding,
        clarification_decision,
        pipeline_result=None,
    ) -> ReasoningStateFrame:
        subject = FocusResolutionSupport.subject_focus(resolved_prompt).focus
        if state.canonical_subject and self._is_explanation_transform_prompt(resolved_prompt):
            subject = state.canonical_subject
        prior_pending = dict(state.pending_followup or {})
        prior_followup_type = str(prior_pending.get("type") or "").strip()
        pending_followup = dict(state.pending_followup or {})
        if clarification_decision is not None and getattr(clarification_decision, "should_clarify", False):
            pending_followup = {
                "type": "clarification",
                "action": getattr(clarification_decision, "action", None),
                "route_mode": getattr(route, "mode", None),
                "route_kind": getattr(route, "kind", None),
                "resolved_prompt": resolved_prompt,
            }
        route_confidence = float(getattr(route_authority, "confidence", getattr(route, "confidence", 0.0)) or 0.0)
        weak_route = bool(getattr(getattr(pipeline_result, "route_decision", None), "weak_route", False))
        confidence_assessment = ConfidenceGradient.from_route(
            score=route_confidence,
            weak_route=weak_route,
            rationale=self._confidence_rationale(
                route_reason=str(getattr(route_authority, "reason", "") or ""),
                weak_route=weak_route,
            ),
        )
        uncertainty_flags = list(state.uncertainty_flags or [])
        failure_flags = list(state.failure_flags or [])
        if weak_route:
            uncertainty_flags.append("weak_route")
        if clarification_decision is not None and getattr(clarification_decision, "should_clarify", False):
            uncertainty_flags.append("clarification_required")
        if clarification_decision is not None and getattr(clarification_decision, "action", "") == "degrade":
            uncertainty_flags.append("clarification_suppressed")
            failure_flags.append("degraded_recovery")
        response_style = dict(state.response_style or {})
        if pipeline_result is not None:
            response_style.update(
                {
                    "selected_mode": str(state.selected_mode or ""),
                    "flow_style": str(getattr(getattr(pipeline_result, "human_language_layer", None), "flow_style", "") or ""),
                    "response_brevity": str(getattr(getattr(pipeline_result, "human_language_layer", None), "response_brevity", "") or ""),
                    "interaction_mode": str(getattr(getattr(pipeline_result, "dialogue_management", None), "interaction_mode", "") or ""),
                }
            )
        return self._ensure_mode_state(state.with_updates(
            current_intent=str(getattr(prompt_understanding, "intent", None).label if getattr(prompt_understanding, "intent", None) else state.current_intent),
            current_task=resolved_prompt,
            current_path=f"{getattr(route, 'mode', '')}:{getattr(route, 'kind', '')}",
            confidence=confidence_assessment.score,
            confidence_tier=confidence_assessment.tier,
            route_decision=(route_authority.to_dict() if route_authority is not None else {}),
            intent_domain=str(getattr(getattr(pipeline_result, "intent_domain", None), "domain", state.intent_domain or "") or "") or None,
            response_depth=str(getattr(getattr(pipeline_result, "response_depth", None), "level", state.response_depth or "") or "") or None,
            conversation_phase=str(getattr(getattr(pipeline_result, "conversation_phase", None), "phase", state.conversation_phase or "") or "") or None,
            ambiguity_status=(
                "clarification_required"
                if clarification_decision is not None and getattr(clarification_decision, "should_clarify", False)
                else "degraded_recovery"
                if clarification_decision is not None and getattr(clarification_decision, "action", "") == "degrade"
                else "resolved"
                if prior_followup_type == "clarification_resolved"
                else "grounded"
                if getattr(route, "mode", "") in {"planning", "research", "tool"}
                else "clear"
            ),
            active_domain=str(getattr(route, "mode", "") or state.active_domain or ""),
            pending_followup=pending_followup,
            resolved_prompt=resolved_prompt,
            canonical_subject=subject or state.canonical_subject,
            continuation_target=resolved_prompt,
            comparison_targets=self._comparison_targets(resolved_prompt) or state.comparison_targets,
            explanation_strategy=self._infer_explanation_strategy(resolved_prompt),
            tool_candidate=self._tool_candidate(route),
            response_style=response_style,
            uncertainty_flags=self._dedupe_flags(uncertainty_flags),
            failure_flags=self._dedupe_flags(failure_flags),
            rationale_summary=self._summarize_route_alignment(
                route_reason=str(getattr(route_authority, "reason", "") or ""),
                intent_domain=str(getattr(getattr(pipeline_result, "intent_domain", None), "domain", "") or ""),
                response_depth=str(getattr(getattr(pipeline_result, "response_depth", None), "level", "") or ""),
            ),
            turn_status=(
                "clarifying"
                if clarification_decision is not None and getattr(clarification_decision, "should_clarify", False)
                else "recovered"
                if clarification_decision is not None and getattr(clarification_decision, "action", "") == "degrade"
                else "clarification_resumed"
                if prior_followup_type == "clarification_resolved"
                else "routed"
            ),
        ))

    def apply_clarification_continuation(
        self,
        *,
        state: ReasoningStateFrame,
        continuation: dict[str, object] | None,
    ) -> ReasoningStateFrame:
        if not continuation:
            return self._ensure_mode_state(state)
        action = str(continuation.get("action") or "").strip()
        if action == "decline":
            return self._ensure_mode_state(state.with_updates(
                pending_followup={},
                ambiguity_status="declined",
                turn_status="awaiting_direction",
            ))
        return self._ensure_mode_state(state.with_updates(
            pending_followup={
                "type": "clarification_resolved",
                "action": action,
                "route_mode": continuation.get("mode"),
                "route_kind": continuation.get("kind"),
            },
            current_path=(
                f"{continuation.get('mode')}:{continuation.get('kind')}"
                if continuation.get("mode") and continuation.get("kind")
                else state.current_path
            ),
            continuation_target=str(continuation.get("resolved_prompt") or continuation.get("working_prompt") or state.continuation_target or ""),
            ambiguity_status="resolved",
            turn_status="clarification_resumed",
        ))

    def apply_memory_context(
        self,
        *,
        state: ReasoningStateFrame,
        memory_context_decision,
    ) -> ReasoningStateFrame:
        selected = list(getattr(memory_context_decision, "selected", ()) or ())
        if not selected:
            return self._ensure_mode_state(state.with_updates(
                memory_context_used=[],
                rationale_summary=self._append_rationale(
                    state.rationale_summary,
                    "No retrieved memory met the relevance threshold for this turn.",
                ),
            ))
        avg_signal = sum(
            (
                float(item.relevance_score)
                + float(item.domain_match)
                + float(item.recency_weight)
                + float(item.confidence)
                + float(item.intent_alignment)
            ) / 5.0
            for item in selected
        ) / max(len(selected), 1)
        confidence_assessment = ConfidenceGradient.with_memory(
            state.confidence,
            memory_signal=avg_signal,
            rationale="Admitted only high-signal memory context into the current turn.",
        )
        return self._ensure_mode_state(state.with_updates(
            confidence=confidence_assessment.score,
            confidence_tier=confidence_assessment.tier,
            memory_context_used=[item.to_dict() for item in selected],
            rationale_summary=self._append_rationale(
                state.rationale_summary,
                str(getattr(memory_context_decision, "rationale", "") or "").strip(),
            ),
        ))

    def apply_tool_usage_intent(
        self,
        *,
        state: ReasoningStateFrame,
        tool_id: str | None,
        capability: str | None,
        input_path: Path | None,
        params: dict[str, object] | None,
    ) -> ReasoningStateFrame:
        return self._ensure_mode_state(state.with_updates(
            tool_usage_intent={
                "tool_id": tool_id,
                "capability": capability,
                "input_path": str(input_path) if input_path else None,
                "has_params": bool(params),
            },
        ))

    def apply_tool_decision(
        self,
        *,
        state: ReasoningStateFrame,
        decision,
    ) -> ReasoningStateFrame:
        confidence_assessment = ConfidenceGradient.with_tool_outcome(
            state.confidence,
            expected_confidence_gain=float(getattr(decision, "expected_confidence_gain", 0.0) or 0.0),
            verified=False,
            failed=not bool(getattr(decision, "should_use_tool", False)) and not bool(getattr(decision, "internal_reasoning_sufficient", False)),
            rationale=str(getattr(decision, "rationale", "") or "").strip() or None,
        )
        uncertainty_flags = list(state.uncertainty_flags or [])
        if not bool(getattr(decision, "should_use_tool", False)):
            uncertainty_flags.append("tool_gate_declined")
        return self._ensure_mode_state(state.with_updates(
            confidence=confidence_assessment.score,
            confidence_tier=confidence_assessment.tier,
            tool_decision=decision.to_dict(),
            uncertainty_flags=self._dedupe_flags(uncertainty_flags),
            rationale_summary=self._append_rationale(
                state.rationale_summary,
                str(getattr(decision, "rationale", "") or "").strip(),
            ),
        ))

    def apply_response_style(
        self,
        *,
        state: ReasoningStateFrame,
        response_style: dict[str, object],
    ) -> ReasoningStateFrame:
        merged = dict(state.response_style or {})
        merged.update({str(key): value for key, value in (response_style or {}).items()})
        return self._ensure_mode_state(state.with_updates(response_style=merged))

    def apply_execution_outcome(
        self,
        *,
        state: ReasoningStateFrame,
        outcome: ExecutionOutcome,
    ) -> ReasoningStateFrame:
        context_summary = self._merged_known_context_summary(
            existing=state.known_context_summary,
            outcome=outcome,
        )
        confidence_assessment = ConfidenceGradient.with_tool_outcome(
            state.confidence,
            expected_confidence_gain=float((state.tool_decision or {}).get("expected_confidence_gain") or 0.0),
            verified=bool(outcome.execution_attempted and outcome.failure_class == "success"),
            failed=bool(outcome.execution_attempted and outcome.failure_class not in {None, "success"}),
            rationale="Updated confidence after the tool execution outcome was observed.",
        )
        failure_flags = list(state.failure_flags or [])
        if outcome.failure_class and outcome.failure_class != "success":
            failure_flags.append(str(outcome.failure_class))
        runtime_diagnostics = dict(state.runtime_diagnostics or {})
        runtime_diagnostics.update(dict(outcome.runtime_diagnostics))
        return self._ensure_mode_state(state.with_updates(
            execution_status=outcome.execution_status,
            failure_class=outcome.failure_class,
            confidence=confidence_assessment.score,
            confidence_tier=confidence_assessment.tier,
            runtime_diagnostics=runtime_diagnostics,
            tool_context={
                "tool_id": outcome.selected_tool_id,
                "capability": outcome.selected_capability,
                **({"summary": outcome.summary} if outcome.summary else {}),
            },
            known_context_summary=context_summary,
            failure_flags=self._dedupe_flags(failure_flags),
            pending_followup={},
            turn_status="executed" if outcome.execution_attempted else "execution_skipped",
        ))

    def to_persistable(self, state: ReasoningStateFrame | dict[str, object] | None) -> dict[str, object]:
        if isinstance(state, ReasoningStateFrame):
            return self._ensure_mode_state(state).to_dict()
        if isinstance(state, dict):
            return self._ensure_mode_state(ReasoningStateFrame.from_mapping(state)).to_dict()
        return ReasoningStateFrame().to_dict()

    def from_mapping(self, state: ReasoningStateFrame | dict[str, object] | None) -> ReasoningStateFrame:
        if isinstance(state, ReasoningStateFrame):
            return self._ensure_mode_state(state)
        if isinstance(state, dict):
            return self._ensure_mode_state(ReasoningStateFrame.from_mapping(state))
        return self._ensure_mode_state(ReasoningStateFrame())

    def interaction_style(
        self,
        *,
        state: ReasoningStateFrame | dict[str, object] | None,
        interaction_profile: InteractionProfile | None = None,
    ) -> str:
        normalized_state = self._ensure_mode_state(
            state
            if isinstance(state, ReasoningStateFrame)
            else ReasoningStateFrame.from_mapping(state if isinstance(state, dict) else None)
        )
        if normalized_state.selected_mode:
            return str(normalized_state.selected_mode)
        profile = interaction_profile or InteractionProfile.default()
        return InteractionProfile.normalize_interaction_style(profile.interaction_style)

    @staticmethod
    def classify_execution_outcome(*, tool_result=None, skipped_reason: str | None = None) -> ExecutionOutcome:
        if skipped_reason:
            failure_class = "input_failure" if skipped_reason == "missing_structured_inputs" else "routing_failure"
            return ExecutionOutcome(
                execution_attempted=False,
                execution_status="skipped",
                failure_class=failure_class,
                runtime_diagnostics={"skipped_reason": skipped_reason},
            )
        if tool_result is None:
            return ExecutionOutcome(execution_attempted=False, execution_status="idle")
        structured = getattr(tool_result, "structured_data", {})
        if not isinstance(structured, dict):
            structured = {}
        tool_id = str(getattr(tool_result, "tool_id", "") or "").strip() or None
        capability = str(getattr(tool_result, "capability", "") or "").strip() or None
        status = str(getattr(tool_result, "status", "") or "").strip().lower()
        failure_class = ReasoningStateService._failure_class_from_result(status=status, structured=structured)
        diagnostics = {}
        runtime = structured.get("runtime_diagnostics")
        if isinstance(runtime, dict):
            diagnostics.update(runtime)
        analysis_status = structured.get("analysis_status")
        if isinstance(analysis_status, dict):
            nested_runtime = analysis_status.get("runtime_diagnostics")
            if isinstance(nested_runtime, dict):
                diagnostics.update({f"analysis_{key}": value for key, value in nested_runtime.items()})
            for key in ("failure_reason", "result_quality"):
                value = analysis_status.get(key)
                if value is not None and value != "":
                    diagnostics[f"analysis_{key}"] = value
        for key in ("failure_category", "failure_reason", "result_quality"):
            value = structured.get(key)
            if value is not None and value != "":
                diagnostics[key] = value
        artifact_signals = {
            "artifact_count": len(getattr(tool_result, "artifacts", []) or []),
            "status": status,
        }
        domain_payload = structured.get("domain_payload")
        if isinstance(domain_payload, dict):
            accepted_files = domain_payload.get("accepted_files")
            if isinstance(accepted_files, list):
                missing_artifacts = 0
                for item in accepted_files:
                    if not isinstance(item, dict):
                        continue
                    artifact_state = item.get("artifact_generation_status")
                    if isinstance(artifact_state, dict) and any(
                        expected is True and created is False
                        for expected, created in (
                            (artifact_state.get("overview_plot_expected"), artifact_state.get("overview_plot_created")),
                            (artifact_state.get("window_plot_expected"), artifact_state.get("window_plot_created")),
                        )
                    ):
                        missing_artifacts += 1
                if missing_artifacts:
                    artifact_signals["missing_artifact_files"] = missing_artifacts
        return ExecutionOutcome(
            selected_tool_id=tool_id,
            selected_capability=capability,
            execution_attempted=True,
            execution_status=status or "ok",
            failure_class=failure_class,
            runtime_diagnostics=diagnostics,
            summary=str(getattr(tool_result, "summary", "") or "").strip() or None,
            artifact_signals=artifact_signals,
        )

    @staticmethod
    def _failure_class_from_result(*, status: str, structured: dict[str, object]) -> str:
        if status == "ok":
            return "success"
        analysis_status = structured.get("analysis_status")
        analysis_failure_reason = ""
        analysis_quality = ""
        analysis_plot_generated = None
        if isinstance(analysis_status, dict):
            analysis_failure_reason = str(analysis_status.get("failure_reason") or "").strip().lower()
            analysis_quality = str(analysis_status.get("result_quality") or "").strip().lower()
            if "plot_generated" in analysis_status:
                analysis_plot_generated = bool(analysis_status.get("plot_generated"))
            if (
                analysis_quality == "candidate_dips_detected"
                and analysis_status.get("line_detected") is True
                and analysis_plot_generated is not False
            ):
                return "success"
        category = str(structured.get("failure_category") or "").strip().lower()
        reason = str(structured.get("failure_reason") or "").strip().lower()
        quality = str(structured.get("result_quality") or "").strip().lower()
        text = " ".join(
            part for part in (category, reason, quality, analysis_failure_reason, analysis_quality) if part
        )
        if analysis_quality == "partial_artifacts" or analysis_plot_generated is False:
            return "artifact_failure"
        if "dependency" in text or "provider" in text or "runtime" in text:
            return "runtime_dependency_failure"
        if "artifact" in text or "plot" in text:
            return "artifact_failure"
        if "unsupported_operation" in text or "unsupported operation" in text:
            return "unsupported_operation"
        if "input" in text or "invalid" in text or "missing" in text:
            return "input_failure"
        return "execution_failure"

    @staticmethod
    def _mode_behavior(profile: InteractionProfile) -> ModeBehaviorProfile:
        style = InteractionProfile.normalize_interaction_style(profile.interaction_style)
        if style == "direct":
            return ModeBehaviorProfile(
                mode="direct",
                posture="decisive",
                explanation_style="compressed",
                follow_up_style="minimal",
                clarification_style="minimal",
            )
        if style == "collab":
            return ModeBehaviorProfile(
                mode="collab",
                posture="exploratory",
                explanation_style="guided",
                follow_up_style="cooperative",
                clarification_style="collaborative",
            )
        return ModeBehaviorProfile(
            mode="default",
            posture="balanced",
            explanation_style="structured",
            follow_up_style="balanced",
            clarification_style="balanced",
        )

    @staticmethod
    def _known_context_summary(*, active_thread: dict[str, object] | None, input_path: Path | None) -> str | None:
        parts: list[str] = []
        if isinstance(active_thread, dict):
            summary = str(active_thread.get("thread_summary") or active_thread.get("summary") or "").strip()
            if summary:
                parts.append(summary)
        if input_path is not None:
            parts.append(f"attached_input={input_path}")
        return " | ".join(parts) if parts else None

    @staticmethod
    def _confidence_rationale(*, route_reason: str, weak_route: bool) -> str:
        if weak_route:
            return f"Route confidence is provisional because the selected route is weak. {route_reason}".strip()
        return f"Route confidence follows the selected route authority. {route_reason}".strip()

    @staticmethod
    def _summarize_route_alignment(*, route_reason: str, intent_domain: str, response_depth: str) -> str | None:
        parts = [part for part in (route_reason, intent_domain, response_depth) if str(part).strip()]
        if not parts:
            return None
        return " | ".join(str(part).strip() for part in parts if str(part).strip())

    @staticmethod
    def _append_rationale(existing: str | None, addition: str | None) -> str | None:
        current = str(existing or "").strip()
        extra = str(addition or "").strip()
        if not extra:
            return current or None
        if not current:
            return extra
        if extra.lower() in current.lower():
            return current
        return f"{current} | {extra}"

    @staticmethod
    def _dedupe_flags(flags: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for flag in flags:
            normalized = str(flag).strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            ordered.append(normalized)
        return ordered

    @staticmethod
    def _merged_known_context_summary(
        *,
        existing: str | None,
        outcome: ExecutionOutcome,
    ) -> str | None:
        parts = [str(existing or "").strip()] if str(existing or "").strip() else []
        tool_id = str(outcome.selected_tool_id or "").strip()
        capability = str(outcome.selected_capability or "").strip()
        execution_status = str(outcome.execution_status or "").strip()
        if tool_id or capability:
            tool_label = ".".join(part for part in (tool_id, capability) if part)
            status_label = execution_status or ("attempted" if outcome.execution_attempted else "idle")
            parts.append(f"tool={tool_label or 'unknown'} status={status_label}")
        if outcome.summary:
            parts.append(str(outcome.summary).strip())
        return " | ".join(part for part in parts if part) or None

    @staticmethod
    def _infer_explanation_strategy(prompt: str) -> str:
        normalized = " ".join(str(prompt or "").strip().lower().split())
        if not normalized:
            return "direct_definition"
        if "compare " in normalized or " vs " in normalized or " versus " in normalized:
            return "compare_contrast"
        if " in relation to " in normalized or " related to " in normalized:
            return "compare_contrast"
        if any(phrase in normalized for phrase in ("break it down", "step by step")):
            return "step_by_step"
        if any(phrase in normalized for phrase in ("analogy", "like a ", "like an ")):
            return "analogy"
        if any(phrase in normalized for phrase in ("simply", "simple", "simply but correctly")):
            return "concrete_example"
        return "direct_definition"

    @staticmethod
    def _is_explanation_transform_prompt(prompt: str) -> bool:
        normalized = " ".join(str(prompt or "").strip().lower().split())
        if not normalized:
            return False
        if normalized.startswith("explain ") and any(
            marker in normalized
            for marker in (" simply", " deeply", " in depth", " step by step", " in relation to ")
        ):
            return True
        if normalized.startswith("break ") and " down" in normalized:
            return True
        return False

    @classmethod
    def _stateful_explanation_prompt(
        cls,
        *,
        normalized: str,
        state: ReasoningStateFrame,
    ) -> str | None:
        if not cls._is_explanation_follow_up_prompt(normalized):
            return None
        subject = cls._stateful_subject(state)
        if not subject:
            return None
        if normalized in cls._SIMPLE_FOLLOW_UP_PROMPTS:
            return f"explain {subject} simply"
        if normalized in cls._DEEP_FOLLOW_UP_PROMPTS:
            return f"explain {subject} deeply"
        if normalized.startswith("explain it ") or normalized.startswith("explain that ") or normalized.startswith("explain this "):
            suffix = normalized.split(" ", maxsplit=2)[-1].strip()
            return f"explain {subject} {suffix}".strip()
        return None

    @classmethod
    def _is_explanation_follow_up_prompt(cls, normalized: str) -> bool:
        return (
            normalized in cls._SIMPLE_FOLLOW_UP_PROMPTS
            or normalized in cls._DEEP_FOLLOW_UP_PROMPTS
            or normalized.startswith(("explain it ", "explain that ", "explain this "))
        )

    @staticmethod
    def _stateful_subject(state: ReasoningStateFrame) -> str | None:
        for candidate in (
            state.canonical_subject,
            state.continuation_target,
            state.resolved_prompt,
            state.current_task,
        ):
            text = " ".join(str(candidate or "").strip().split())
            if not text:
                continue
            if text.lower() in {
                "it",
                "this",
                "that",
                "break it down",
                "break that down",
                "break this down",
            }:
                continue
            return text
        return None

    @staticmethod
    def _comparison_targets(prompt: str) -> list[str]:
        normalized = " ".join(str(prompt or "").strip().lower().split())
        separators = (" vs ", " versus ", " in relation to ", " related to ")
        for separator in separators:
            if separator in normalized:
                left, right = normalized.split(separator, maxsplit=1)
                return [FocusResolutionSupport.subject_focus(left).focus, FocusResolutionSupport.subject_focus(right).focus]
        if normalized.startswith("compare ") and " and " in normalized:
            body = normalized[len("compare ") :]
            left, right = body.split(" and ", maxsplit=1)
            return [FocusResolutionSupport.subject_focus(left).focus, FocusResolutionSupport.subject_focus(right).focus]
        return []

    @staticmethod
    def _tool_candidate(route) -> dict[str, object]:
        if str(getattr(route, "mode", "") or "").strip() != "tool":
            return {}
        return {
            "route_kind": str(getattr(route, "kind", "") or "").strip() or None,
            "route_source": str(getattr(route, "source", "") or "").strip() or None,
            "confidence": float(getattr(route, "confidence", 0.0) or 0.0),
        }

    @staticmethod
    def _ensure_mode_state(state: ReasoningStateFrame) -> ReasoningStateFrame:
        normalized = ReasoningStateFrame.from_mapping(state.to_dict())
        if not normalized.confidence_tier:
            normalized.confidence_tier = ConfidenceGradient.tier_for_score(normalized.confidence)
        return normalized
