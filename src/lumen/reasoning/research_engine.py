from __future__ import annotations

from lumen.reasoning.assistant_context import AssistantContext
from lumen.reasoning.evidence_builder import EvidenceBuilder
from lumen.reasoning.interaction_style_policy import InteractionStylePolicy
from lumen.reasoning.reasoning_language import ReasoningResponseLanguage
from lumen.reasoning.response_variation import ResponseVariationLayer
from lumen.reasoning.reasoning_status import ReasoningStatusPolicy
from lumen.reasoning.pipeline_models import ReasoningFrameAssembly
from lumen.reasoning.response_models import ResearchResponse


class ResearchEngine:
    """Builds a lightweight research-style response."""

    def __init__(self) -> None:
        self.evidence_builder = EvidenceBuilder()
        self.reasoning_status_policy = ReasoningStatusPolicy()

    def respond(
        self,
        prompt: str,
        *,
        kind: str = "research.general",
        context: AssistantContext | dict[str, object] | None = None,
        reasoning_frame_assembly: ReasoningFrameAssembly | None = None,
    ) -> dict[str, object]:
        if context is None:
            context_obj = None
            context_payload: dict[str, object] = {}
        else:
            context_obj = (
                context
                if isinstance(context, AssistantContext)
                else AssistantContext.from_mapping(context)
            )
            context_payload = context_obj.to_dict()

        if kind == "research.comparison":
            findings = [
                "Identify the two or more options being compared.",
                "Call out the most meaningful tradeoffs first.",
                "End with the clearest context-dependent recommendation.",
            ]
            recommendation = "Compare the options on their most meaningful tradeoffs and choose the clearest context-fit."
        elif kind == "research.summary":
            findings = [
                "State the topic in one concise sentence.",
                "Highlight the most relevant local evidence first.",
                "Close with the clearest next action or conclusion.",
            ]
            recommendation = "Summarize the strongest local evidence first, then end with one concrete next step."
        else:
            findings = [
                "Identify the main topic and any technical constraints.",
                "Summarize the most relevant local context before taking action.",
                "Recommend the next concrete validation or implementation step.",
            ]
            recommendation = "Use the strongest local context to decide the next concrete validation step."

        evidence = self.evidence_builder.build(mode="research", context=context_obj)
        route = (context_obj.route if context_obj is not None else {}) or {}
        route_caution = str(route.get("caution") or "").strip()
        route_ambiguity = bool(route.get("ambiguity"))
        route_quality = (
            reasoning_frame_assembly.route_quality
            if reasoning_frame_assembly and reasoning_frame_assembly.route_quality
            else self.evidence_builder.route_quality_label(context=context_obj)
        )
        if route_caution:
            evidence.append(f"Route caution: {route_caution}")
        best_evidence = self.evidence_builder.summarize_best_evidence(evidence)
        local_context_summary = (
            reasoning_frame_assembly.local_context_summary
            if reasoning_frame_assembly
            else self.evidence_builder.summarize_local_context(context=context_obj)
        )
        grounded_interpretation = (
            reasoning_frame_assembly.grounded_interpretation
            if reasoning_frame_assembly
            else self.evidence_builder.synthesize_interpretation(
                mode="research",
                context=context_obj,
            )
        )
        working_hypothesis = (
            reasoning_frame_assembly.working_hypothesis
            if reasoning_frame_assembly
            else self.evidence_builder.build_working_hypothesis(
                mode="research",
                context=context_obj,
            )
        )
        reasoning_frame = (
            reasoning_frame_assembly.reasoning_frame
            if reasoning_frame_assembly
            else self.evidence_builder.build_reasoning_frame(context=context_obj)
        )
        local_context_assessment = (
            reasoning_frame_assembly.local_context_assessment
            if reasoning_frame_assembly
            else self.evidence_builder.assess_local_context(context=context_obj)
        )
        validation_plan = list(reasoning_frame_assembly.validation_plan) if reasoning_frame_assembly else []
        interaction_profile = (
            dict(reasoning_frame_assembly.interaction_profile)
            if reasoning_frame_assembly
            else {}
        )
        grounding_strength = (
            reasoning_frame_assembly.grounding_strength
            if reasoning_frame_assembly and reasoning_frame_assembly.grounding_strength
            else self.evidence_builder.grounding_strength(context=context_obj)
        )
        findings = self._structure_findings(
            findings=findings,
            grounding_strength=grounding_strength,
            route_quality=route_quality,
            local_context_assessment=local_context_assessment,
            reasoning_frame=reasoning_frame,
            route_ambiguity=route_ambiguity,
            tension_resolution=(
                dict(reasoning_frame_assembly.tension_resolution)
                if reasoning_frame_assembly and reasoning_frame_assembly.tension_resolution
                else None
            ),
        )
        if validation_plan:
            validation_limit = InteractionStylePolicy.validation_advice_limit(interaction_profile)
            findings.extend(f"Validation plan: {item}" for item in validation_plan[:validation_limit])
        if InteractionStylePolicy.is_deep(interaction_profile):
            findings.append("Deep thinking pass: cross-check the main anchor against the other local sources before closing the answer.")
        if InteractionStylePolicy.is_direct(interaction_profile):
            findings = findings[:5]
        confidence_posture = self.reasoning_status_policy.confidence_posture(
            route_strength=str(route.get("strength") or ""),
            route_quality=route_quality,
            grounding_strength=grounding_strength,
            local_context_assessment=local_context_assessment,
            route_ambiguity=route_ambiguity,
        )
        uncertainty_note = ReasoningResponseLanguage.uncertainty_note(
            confidence_posture=confidence_posture,
            local_context_assessment=local_context_assessment,
            reasoning_frame=reasoning_frame,
            route_caution=route_caution,
            route_ambiguity=route_ambiguity,
            subject_label="answer",
        )
        closing_strategy = self._closing_strategy(
            grounding_strength=grounding_strength,
            route_caution=route_caution,
            route_quality=route_quality,
            local_context_assessment=local_context_assessment,
            route_ambiguity=route_ambiguity,
        )
        summary = ReasoningResponseLanguage.summary_for_prompt(
            prompt=prompt,
            confidence_posture=confidence_posture,
            response_kind="research",
            intent_domain=None,
        )
        recommendation = self._grounded_recommendation(
            base_recommendation=recommendation,
            grounding_strength=grounding_strength,
            route_caution=route_caution,
            route_quality=route_quality,
            local_context_summary=local_context_summary,
            local_context_assessment=local_context_assessment,
            reasoning_frame=reasoning_frame,
            route_ambiguity=route_ambiguity,
        )
        if InteractionStylePolicy.is_direct(interaction_profile):
            recommendation = recommendation.split(". ")[0].rstrip(".") + "."

        return ResearchResponse(
            mode="research",
            kind=kind,
            summary=summary,
            context=context_obj or AssistantContext.from_mapping(context_payload),
            evidence=evidence,
            best_evidence=best_evidence,
            local_context_summary=local_context_summary,
            grounded_interpretation=grounded_interpretation,
            working_hypothesis=working_hypothesis,
            uncertainty_note=uncertainty_note,
            reasoning_frame=reasoning_frame,
            local_context_assessment=local_context_assessment,
            grounding_strength=grounding_strength,
            confidence_posture=confidence_posture,
            closing_strategy=closing_strategy,
            findings=findings,
            recommendation=recommendation,
        ).to_dict()

    @staticmethod
    def _structure_findings(
        *,
        findings: list[str],
        grounding_strength: str,
        route_quality: str,
        local_context_assessment: str | None,
        reasoning_frame: dict[str, str],
        route_ambiguity: bool,
        tension_resolution: dict[str, object] | None,
    ) -> list[str]:
        structured = list(findings)
        insert_at = 1 if structured else 0
        if local_context_assessment == "mixed":
            structured.insert(
                insert_at,
                (
                    f"Call out this local tension before drawing conclusions: {reasoning_frame.get('tension')}."
                    if reasoning_frame.get("tension")
                    else "Call out the tension between archived evidence and prior session context before drawing conclusions."
                ),
            )
            structured.insert(
                insert_at + 1,
                ReasoningResponseLanguage.tension_guidance(
                    tension_resolution=tension_resolution,
                    fallback="Separate what is corroborated locally from what still depends on interpretation.",
                    subject_label="conclusion",
                ),
            )
            return structured
        if local_context_assessment == "aligned":
            structured.insert(
                insert_at,
                (
                    f"Use this aligned anchor as the strongest signal: {reasoning_frame.get('primary_anchor')}."
                    if reasoning_frame.get("primary_anchor")
                    else "Use the agreement between archived evidence and prior session context as the strongest anchor."
                ),
            )
            if reasoning_frame.get("coherence_topic"):
                structured.insert(
                    insert_at + 1,
                    f"The strongest local evidence sources reinforce the same topic: {reasoning_frame.get('coherence_topic')}.",
                )
                structured.insert(
                    insert_at + 2,
                    "Promote that shared topic to the main conclusion before adding secondary detail.",
                )
                return structured
            structured.insert(
                insert_at + 1,
                "Promote the shared signal to the main conclusion before adding secondary detail.",
            )
            return structured
        if grounding_strength == "low":
            structured.insert(
                insert_at,
                "Treat the early conclusion as exploratory until another local source confirms it.",
            )
            return structured
        if grounding_strength == "high" and route_quality == "weak":
            structured.insert(
                insert_at,
                ResearchEngine._route_validation_finding(reasoning_frame=reasoning_frame, strength_label="strong"),
            )
            return structured
        if grounding_strength == "medium":
            if route_quality == "weak":
                structured.insert(
                    insert_at,
                    ResearchEngine._route_validation_finding(reasoning_frame=reasoning_frame, strength_label="fairly strong"),
                )
                return structured
            structured.insert(
                insert_at,
                (
                    f"Keep the first conclusion close to this local anchor and avoid broad extrapolation: {reasoning_frame.get('primary_anchor')}."
                    if reasoning_frame.get("primary_anchor")
                    else "Keep the first conclusion close to the strongest local signal and avoid broad extrapolation."
                ),
            )
            return structured
        if route_ambiguity:
            structured.insert(
                insert_at,
                ReasoningResponseLanguage.tension_guidance(
                    tension_resolution=tension_resolution,
                    fallback="Keep the first conclusion narrow until the prompt intent is clearer, because the route decision was close.",
                    subject_label="conclusion",
                ),
            )
            return structured
        return structured

    @staticmethod
    def _grounded_recommendation(
        *,
        base_recommendation: str,
        grounding_strength: str,
        route_caution: str,
        route_quality: str,
        local_context_summary: str | None,
        local_context_assessment: str | None,
        reasoning_frame: dict[str, str],
        route_ambiguity: bool,
    ) -> str:
        strategy = ResearchEngine._closing_strategy(
            grounding_strength=grounding_strength,
            route_caution=route_caution,
            route_quality=route_quality,
            local_context_assessment=local_context_assessment,
            route_ambiguity=route_ambiguity,
        )
        if strategy == "reconcile_first":
            base_recommendation = (
                f"{base_recommendation} Explicitly resolve the mismatch between archived evidence and prior session context."
            )
            if route_caution:
                base_recommendation = (
                    f"{base_recommendation} Keep in mind that the route choice was somewhat tentative."
                )
        elif strategy == "anchor_and_execute":
            base_recommendation = (
                f"{base_recommendation} Use the agreement between archived evidence and prior session context as the main anchor."
            )
            coherence_topic = local_context_summary or ""
            if "semantically aligned" in coherence_topic:
                base_recommendation = (
                    f"{base_recommendation} Keep the shared local topic explicit while interpreting the result."
                )
        if strategy == "proceed_with_context":
            return ResearchEngine._contextualize_recommendation(base_recommendation, local_context_summary)
        if strategy == "validate_before_commit":
            return ResearchEngine._contextualize_recommendation(
                f"{base_recommendation} Keep one local validation step before acting on it.",
                local_context_summary,
            )
        if strategy == "cautious_execution":
            return ResearchEngine._contextualize_recommendation(
                (
                    f"{base_recommendation} Keep the route caution in mind while checking the first conclusion."
                    if route_quality != "weak"
                    else f"{base_recommendation} Keep the route caution in mind and {ResearchEngine._route_validation_recommendation(reasoning_frame=reasoning_frame)}"
                ),
                local_context_summary,
            )
        if strategy == "cautious_validation":
            return ResearchEngine._contextualize_recommendation(
                (
                (
                    f"{base_recommendation} Keep one local validation step before acting on it, "
                    "especially because the route choice was somewhat tentative."
                )
                if route_quality != "weak"
                else (
                    f"{base_recommendation} Keep one local validation step before acting on it, "
                    f"and {ResearchEngine._route_validation_recommendation(reasoning_frame=reasoning_frame)}"
                )
                ),
                local_context_summary,
            )
        if strategy == "exploratory_validation":
            return ResearchEngine._contextualize_recommendation(
                (
                    f"{base_recommendation} Treat this as provisional until another local source or run "
                    "confirms it."
                ),
                local_context_summary,
            )
        return ResearchEngine._contextualize_recommendation(
            (
                f"{base_recommendation} Treat this as provisional until another local source or run "
                "confirms it."
            ),
            local_context_summary,
        )

    @staticmethod
    def _closing_strategy(
        *,
        grounding_strength: str,
        route_caution: str,
        route_quality: str,
        local_context_assessment: str | None,
        route_ambiguity: bool,
    ) -> str:
        if local_context_assessment == "mixed":
            return "reconcile_first"
        if local_context_assessment == "aligned":
            return "anchor_and_execute"
        if grounding_strength == "low":
            return "exploratory_validation"
        if route_ambiguity:
            if grounding_strength == "high":
                return "cautious_execution"
            return "cautious_validation"
        if route_quality == "weak":
            return "cautious_validation"
        if grounding_strength == "high" and not route_caution:
            return "proceed_with_context"
        if grounding_strength == "medium" and not route_caution:
            return "validate_before_commit"
        if grounding_strength == "high":
            return "cautious_execution"
        if grounding_strength == "medium":
            return "cautious_validation"
        return "exploratory_validation"

    @staticmethod
    def _contextualize_recommendation(
        base_recommendation: str,
        local_context_summary: str | None,
    ) -> str:
        if not local_context_summary:
            return base_recommendation
        if "Closest archive run:" in local_context_summary:
            tail = ResponseVariationLayer.select_from_pool(
                (
                    "Start with the closest archive run.",
                    "Begin with the closest archive run.",
                    "Anchor it in the closest archive run first.",
                ),
                seed_parts=[base_recommendation, local_context_summary, "research", "archive_tail"],
            )
            return f"{base_recommendation} {tail}"
        if "Closest prior session prompt:" in local_context_summary:
            tail = ResponseVariationLayer.select_from_pool(
                (
                    "Reconcile it with the closest prior session prompt first.",
                    "First reconcile it with the closest prior session prompt.",
                    "Anchor it against the closest prior session prompt first.",
                ),
                seed_parts=[base_recommendation, local_context_summary, "research", "interaction_tail"],
            )
            return f"{base_recommendation} {tail}"
        if "Active thread:" in local_context_summary:
            tail = ResponseVariationLayer.select_from_pool(
                (
                    "Keep the active thread question in view while interpreting the result.",
                    "Keep the active thread question visible while interpreting the result.",
                    "Keep the active thread question in frame while interpreting the result.",
                ),
                seed_parts=[base_recommendation, local_context_summary, "research", "thread_tail"],
            )
            return f"{base_recommendation} {tail}"
        return base_recommendation

    @staticmethod
    def _route_validation_finding(*, reasoning_frame: dict[str, str], strength_label: str) -> str:
        source = str(reasoning_frame.get("primary_anchor_source") or "").strip()
        if source == "archive":
            return (
                f"Treat the first conclusion as a route-validation checkpoint, because the local evidence is {strength_label} "
                "and the chosen route should be validated against the closest archive evidence."
            )
        if source == "interaction":
            return (
                f"Treat the first conclusion as a route-validation checkpoint, because the local evidence is {strength_label} "
                "and the chosen route should be validated against the closest prior session prompt."
            )
        if source == "active_thread":
            return (
                f"Treat the first conclusion as a route-validation checkpoint, because the local evidence is {strength_label} "
                "and the chosen route should be validated against active thread continuity."
            )
        return (
            f"Treat the first conclusion as a route-validation checkpoint, because the local evidence is {strength_label} "
            "but the chosen route remains comparatively weak."
        )

    @staticmethod
    def _route_validation_recommendation(*, reasoning_frame: dict[str, str]) -> str:
        source = str(reasoning_frame.get("primary_anchor_source") or "").strip()
        if source == "archive":
            return "validate the route choice itself against the closest archive evidence before leaning on the conclusion."
        if source == "interaction":
            return "validate the route choice itself against the closest prior session prompt before leaning on the conclusion."
        if source == "active_thread":
            return "validate the route choice itself against active thread continuity before leaning on the conclusion."
        return "validate the route choice itself before leaning on the conclusion."

