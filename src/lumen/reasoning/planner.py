from __future__ import annotations

from lumen.reasoning.assistant_context import AssistantContext
from lumen.reasoning.evidence_builder import EvidenceBuilder
from lumen.reasoning.interaction_style_policy import InteractionStylePolicy
from lumen.reasoning.reasoning_language import ReasoningResponseLanguage
from lumen.reasoning.response_variation import ResponseVariationLayer
from lumen.reasoning.reasoning_status import ReasoningStatusPolicy
from lumen.reasoning.pipeline_models import ReasoningFrameAssembly
from lumen.reasoning.response_models import PlanningResponse


class Planner:
    """Builds a lightweight local planning response."""

    def __init__(self) -> None:
        self.evidence_builder = EvidenceBuilder()
        self.reasoning_status_policy = ReasoningStatusPolicy()

    def respond(
        self,
        prompt: str,
        *,
        kind: str = "planning.general",
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

        design_synthesis = self._design_synthesis(prompt=prompt, kind=kind)
        summary_override: str | None = None
        if design_synthesis is not None:
            steps = list(design_synthesis["steps"])
            next_action = str(design_synthesis["next_action"])
            summary_override = str(design_synthesis["summary"])
        elif kind == "planning.migration":
            steps = [
                "Define the current state, target state, and non-negotiable migration constraints.",
                "Identify the smallest safe migration slice that delivers value without breaking active flows.",
                "Sequence rollout, validation, and fallback checkpoints before broadening the migration.",
            ]
            next_action = "Define the smallest safe migration slice and its validation checkpoint."
        elif kind == "planning.architecture":
            steps = [
                "Define the target architecture boundaries and invariants.",
                "Map current modules to the desired end-state structure.",
                "Sequence migration work into low-risk phases with checkpoints.",
            ]
            next_action = "Document the target boundaries and map current modules to them."
        else:
            steps = [
                "Clarify the main objective and constraints.",
                "Break the work into small executable milestones.",
                "Validate the next milestone before expanding scope.",
            ]
            next_action = "Clarify the next milestone and its validation step."

        evidence = self.evidence_builder.build(mode="planning", context=context_obj)
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
                mode="planning",
                context=context_obj,
            )
        )
        working_hypothesis = (
            reasoning_frame_assembly.working_hypothesis
            if reasoning_frame_assembly
            else self.evidence_builder.build_working_hypothesis(
                mode="planning",
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
        first_pass_step = None if design_synthesis is not None else self._first_pass_attempt_step(prompt=prompt, kind=kind)
        if first_pass_step:
            steps = [first_pass_step, *steps]
        steps = self._structure_steps(
            steps=steps,
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
            steps.extend(f"Validation plan: {item}" for item in validation_plan[:validation_limit])
        if InteractionStylePolicy.is_deep(interaction_profile):
            steps.append("Deep thinking pass: cross-check the main anchor against the other local sources before closing the plan.")
        if InteractionStylePolicy.is_direct(interaction_profile):
            steps = steps[:5]
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
            subject_label="plan",
        )
        closing_strategy = self._closing_strategy(
            grounding_strength=grounding_strength,
            route_caution=route_caution,
            route_quality=route_quality,
            local_context_assessment=local_context_assessment,
            route_ambiguity=route_ambiguity,
        )
        summary = summary_override or ReasoningResponseLanguage.summary_for_prompt(
            prompt=prompt,
            confidence_posture=confidence_posture,
            response_kind="planning",
        )
        next_action = self._grounded_next_action(
            base_next_action=next_action,
            grounding_strength=grounding_strength,
            route_caution=route_caution,
            route_quality=route_quality,
            local_context_summary=local_context_summary,
            local_context_assessment=local_context_assessment,
            reasoning_frame=reasoning_frame,
            route_ambiguity=route_ambiguity,
        )
        if InteractionStylePolicy.is_direct(interaction_profile):
            next_action = next_action.split(". ")[0].rstrip(".") + "."

        return PlanningResponse(
            mode="planning",
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
            steps=steps,
            next_action=next_action,
        ).to_dict()

    @staticmethod
    def _design_synthesis(*, prompt: str, kind: str) -> dict[str, object] | None:
        subject = Planner._design_subject(prompt)
        if not Planner._should_synthesize_design(prompt=prompt, kind=kind, subject=subject):
            return None
        subject_label = subject or "the system"
        engine_like = any(token in subject_label for token in ("engine", "propulsion", "thruster", "motor"))
        if engine_like:
            high_level = (
                f"High-level system: treat {subject_label} as a modular energy-to-output chain with a power section, "
                "a controlled conversion core, and an output stage."
            )
            components = (
                "Key components: power source, fuel or working-fluid feed, conversion chamber, control and sensor loop, "
                "thermal management, structural housing, and nozzle or drive stage."
            )
            interaction = (
                "Interaction: the feed system meters input into the conversion core, the power section sustains the reaction or "
                "energized flow, the controller keeps the chamber stable, and the output stage turns that controlled flow into usable thrust or motion."
            )
            next_refinement = (
                "Next refinement: lock the operating medium, thrust target, and duty cycle so the chamber, cooling, and control logic can be sized cleanly."
            )
        else:
            high_level = (
                f"High-level system: treat {subject_label} as a modular system with an input layer, a core mechanism, "
                "a control loop, and an output interface."
            )
            components = (
                "Key components: input source, conversion or processing core, control and sensing layer, structural frame, safety envelope, and output interface."
            )
            interaction = (
                "Interaction: the input layer feeds the core mechanism, the control loop keeps the operating state stable, "
                "and the output interface exposes the useful behavior without overloading the core."
            )
            next_refinement = (
                "Next refinement: choose the operating environment, target performance, and failure constraints so each module can be sized with less guesswork."
            )
        assumptions = (
            f"Assumptions: treat {subject_label} as a concept-level design and assume we want a clear mechanism and module layout before manufacturing details."
        )
        summary = (
            f"Here’s a first-pass design concept for {subject_label}. "
            "I’m making a few reasonable assumptions so the structure can move forward without stalling."
        )
        return {
            "summary": summary,
            "steps": [
                assumptions,
                high_level,
                components,
                interaction,
                next_refinement,
            ],
            "next_action": next_refinement,
        }

    @staticmethod
    def _should_synthesize_design(*, prompt: str, kind: str, subject: str) -> bool:
        normalized = " ".join(str(prompt).lower().split())
        if not normalized:
            return False
        starters = (
            "design ",
            "design me ",
            "build ",
            "build me ",
            "invent ",
            "invent me ",
            "draft ",
            "propose ",
            "sketch ",
        )
        design_nouns = (
            "engine",
            "propulsion",
            "thruster",
            "motor",
            "device",
            "system",
            "mechanism",
            "prototype",
            "reactor",
        )
        if kind == "planning.architecture" and any(normalized.startswith(prefix) for prefix in starters):
            return any(token in subject for token in design_nouns)
        if not any(normalized.startswith(prefix) for prefix in starters):
            return False
        return any(token in subject for token in design_nouns)

    @staticmethod
    def _design_subject(prompt: str) -> str:
        normalized = " ".join(str(prompt).strip().lower().split())
        starters = (
            "design me an ",
            "design me a ",
            "design me ",
            "design an ",
            "design a ",
            "design ",
            "build me an ",
            "build me a ",
            "build me ",
            "build an ",
            "build a ",
            "build ",
            "invent an ",
            "invent a ",
            "invent ",
            "draft an ",
            "draft a ",
            "draft ",
            "propose an ",
            "propose a ",
            "propose ",
            "sketch an ",
            "sketch a ",
            "sketch ",
        )
        for prefix in starters:
            if normalized.startswith(prefix):
                return normalized[len(prefix) :].strip(" .!?")
        return normalized.strip(" .!?")

    @staticmethod
    def _first_pass_attempt_step(*, prompt: str, kind: str) -> str | None:
        if kind == "planning.migration":
            return None
        normalized = " ".join(str(prompt).lower().split())
        starters = (
            "create ",
            "design ",
            "build ",
            "invent ",
            "draft ",
            "propose ",
            "sketch ",
        )
        starter = next((prefix for prefix in starters if normalized.startswith(prefix)), None)
        if starter is None:
            return None
        subject = normalized[len(starter) :].strip(" .!?")
        if not subject:
            return "Start with a concrete first-pass concept that makes the core mechanism, constraints, and tradeoffs explicit."
        if any(token in subject for token in ("engine", "design", "system", "prototype", "device", "propulsion")):
            return (
                f"Start with a first-pass concept for {subject}, making the core mechanism, constraints, and tradeoffs explicit."
            )
        return f"Start with a first-pass version of {subject} that is concrete enough to test and refine."

    @staticmethod
    def _structure_steps(
        *,
        steps: list[str],
        grounding_strength: str,
        route_quality: str,
        local_context_assessment: str | None,
        reasoning_frame: dict[str, str],
        route_ambiguity: bool,
        tension_resolution: dict[str, object] | None,
    ) -> list[str]:
        structured = list(steps)
        insert_at = 1 if structured else 0
        if local_context_assessment == "mixed":
            mixed_guidance = ReasoningResponseLanguage.tension_guidance(
                tension_resolution=tension_resolution,
                fallback="Treat the first milestone as a reconciliation checkpoint instead of a rollout step.",
                subject_label="milestone",
            )
            structured.insert(
                insert_at,
                reasoning_frame.get("tension")
                or "Resolve the mismatch between the closest archive evidence and prior session context before sequencing work.",
            )
            structured.insert(insert_at + 1, mixed_guidance)
            return structured
        if local_context_assessment == "aligned":
            structured.insert(
                insert_at,
                (
                    f"Use this aligned anchor to keep the rollout focused: {reasoning_frame.get('primary_anchor')}."
                    if reasoning_frame.get("primary_anchor")
                    else "Use the agreement between archived evidence and prior session context to keep the rollout focused."
                ),
            )
            if reasoning_frame.get("coherence_topic"):
                structured.insert(
                    insert_at + 1,
                    f"The strongest local evidence sources reinforce the same topic: {reasoning_frame.get('coherence_topic')}.",
                )
                structured.insert(
                    insert_at + 2,
                    "Turn that shared topic into the first concrete milestone rather than reopening the problem definition.",
                )
                return structured
            structured.insert(
                insert_at + 1,
                "Turn that agreement into the first concrete milestone rather than reopening the problem definition.",
            )
            return structured
        if grounding_strength == "low":
            structured.insert(
                insert_at,
                "Treat the opening milestone as evidence-gathering and validation, not commitment.",
            )
            return structured
        if grounding_strength == "high" and route_quality == "weak":
            structured.insert(
                insert_at,
                Planner._route_validation_step(reasoning_frame=reasoning_frame, strength_label="strong"),
            )
            return structured
        if grounding_strength == "medium":
            if route_quality == "weak":
                structured.insert(
                    insert_at,
                    Planner._route_validation_step(reasoning_frame=reasoning_frame, strength_label="fairly strong"),
                )
                return structured
            structured.insert(
                insert_at,
                (
                    f"Use the first milestone to validate this local anchor before broadening scope: {reasoning_frame.get('primary_anchor')}."
                    if reasoning_frame.get("primary_anchor")
                    else "Use the first milestone to validate the strongest local signal before broadening scope."
                ),
            )
            return structured
        if route_ambiguity:
            structured.insert(
                insert_at,
                ReasoningResponseLanguage.tension_guidance(
                    tension_resolution=tension_resolution,
                    fallback="Keep the opening milestone narrow until the prompt intent is clearer, because the route decision was close.",
                    subject_label="milestone",
                ),
            )
            return structured
        return structured

    @staticmethod
    def _grounded_next_action(
        *,
        base_next_action: str,
        grounding_strength: str,
        route_caution: str,
        route_quality: str,
        local_context_summary: str | None,
        local_context_assessment: str | None,
        reasoning_frame: dict[str, str],
        route_ambiguity: bool,
    ) -> str:
        strategy = Planner._closing_strategy(
            grounding_strength=grounding_strength,
            route_caution=route_caution,
            route_quality=route_quality,
            local_context_assessment=local_context_assessment,
            route_ambiguity=route_ambiguity,
        )
        if strategy == "reconcile_first":
            base_next_action = f"{base_next_action} Resolve the tension between archived evidence and prior session context first."
            if route_caution:
                base_next_action = (
                    f"{base_next_action} Keep in mind that the route choice was somewhat tentative."
                )
        elif strategy == "anchor_and_execute":
            base_next_action = f"{base_next_action} Lean on the aligned local context to keep the plan tight."
            coherence_topic = local_context_summary or ""
            if "semantically aligned" in coherence_topic:
                base_next_action = f"{base_next_action} Keep the shared local topic explicit while sequencing the work."
        if strategy == "proceed_with_context":
            return Planner._contextualize_next_action(base_next_action, local_context_summary)
        if strategy == "validate_before_commit":
            return Planner._contextualize_next_action(
                f"{base_next_action} Validate it against the closest local evidence first.",
                local_context_summary,
            )
        if strategy == "cautious_execution":
            return Planner._contextualize_next_action(
                (
                    f"{base_next_action} Keep the route caution in mind while validating the first step."
                    if route_quality != "weak"
                    else f"{base_next_action} Keep the route caution in mind and {Planner._route_validation_action(reasoning_frame=reasoning_frame)}"
                ),
                local_context_summary,
            )
        if strategy == "cautious_validation":
            return Planner._contextualize_next_action(
                (
                (
                    f"{base_next_action} Validate it against the closest local evidence first, "
                    "especially because the route choice was somewhat tentative."
                )
                if route_quality != "weak"
                else (
                    f"{base_next_action} Validate it against the closest local evidence first, "
                    f"and {Planner._route_validation_action(reasoning_frame=reasoning_frame)}"
                )
                ),
                local_context_summary,
            )
        if strategy == "exploratory_validation":
            return Planner._contextualize_next_action(
                (
                    f"{base_next_action} First confirm the assumptions with an additional local check "
                    "before committing to the plan."
                ),
                local_context_summary,
            )
        return Planner._contextualize_next_action(
            (
                f"{base_next_action} First confirm the assumptions with an additional local check "
                "before committing to the plan."
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
    def _contextualize_next_action(base_next_action: str, local_context_summary: str | None) -> str:
        if not local_context_summary:
            return base_next_action
        if "Closest archive run:" in local_context_summary:
            tail = ResponseVariationLayer.select_from_pool(
                (
                    "Use the closest archive run as the baseline.",
                    "Use the closest archive run as the starting baseline.",
                    "Keep the closest archive run as the baseline reference.",
                ),
                seed_parts=[base_next_action, local_context_summary, "planning", "archive_tail"],
            )
            return f"{base_next_action} {tail}"
        if "Closest prior session prompt:" in local_context_summary:
            tail = ResponseVariationLayer.select_from_pool(
                (
                    "Keep the prior session prompt alignment in view.",
                    "Keep the prior session prompt alignment visible.",
                    "Keep the prior session prompt alignment in frame.",
                ),
                seed_parts=[base_next_action, local_context_summary, "planning", "interaction_tail"],
            )
            return f"{base_next_action} {tail}"
        if "Active thread:" in local_context_summary:
            tail = ResponseVariationLayer.select_from_pool(
                (
                    "Keep the active thread goal in view while sequencing the work.",
                    "Keep the active thread goal visible while sequencing the work.",
                    "Keep the active thread goal in frame while sequencing the work.",
                ),
                seed_parts=[base_next_action, local_context_summary, "planning", "thread_tail"],
            )
            return f"{base_next_action} {tail}"
        return base_next_action

    @staticmethod
    def _route_validation_step(*, reasoning_frame: dict[str, str], strength_label: str) -> str:
        source = str(reasoning_frame.get("primary_anchor_source") or "").strip()
        if source == "archive":
            return (
                f"Treat the first milestone as a route-validation checkpoint, because the local evidence is {strength_label} "
                "and the route should be validated against the closest archive evidence before broadening scope."
            )
        if source == "interaction":
            return (
                f"Treat the first milestone as a route-validation checkpoint, because the local evidence is {strength_label} "
                "and the route should be validated against the closest prior session prompt before broadening scope."
            )
        if source == "active_thread":
            return (
                f"Treat the first milestone as a route-validation checkpoint, because the local evidence is {strength_label} "
                "and the route should be validated against the active thread continuity before broadening scope."
            )
        return (
            f"Treat the first milestone as a route-validation checkpoint, because the local evidence is {strength_label} "
            "but the route choice itself is still comparatively weak."
        )

    @staticmethod
    def _route_validation_action(*, reasoning_frame: dict[str, str]) -> str:
        source = str(reasoning_frame.get("primary_anchor_source") or "").strip()
        if source == "archive":
            return "validate the route choice itself against the closest archive evidence before broadening scope."
        if source == "interaction":
            return "validate the route choice itself against the closest prior session prompt before broadening scope."
        if source == "active_thread":
            return "validate the route choice itself against the active thread continuity before broadening scope."
        return "validate the route choice itself before broadening scope."

