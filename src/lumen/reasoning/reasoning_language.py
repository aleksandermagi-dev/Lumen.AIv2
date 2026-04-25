from __future__ import annotations

from lumen.reasoning.conversation_policy import ConversationPolicy


class ReasoningResponseLanguage:
    """Shared user-facing phrasing for planning and research responses."""

    @staticmethod
    def summary_for_prompt(
        *,
        prompt: str,
        confidence_posture: str,
        response_kind: str,
        intent_domain: str | None = None,
    ) -> str:
        seed = f"{prompt}|{confidence_posture}|{response_kind}|{intent_domain or ''}"
        if response_kind == "planning":
            if confidence_posture == "strong":
                return ReasoningResponseLanguage._select_variant(
                    seed=seed,
                    options=(
                        "Here’s the grounded plan.",
                        "Here’s a grounded plan to start from.",
                    ),
                )
            if confidence_posture == "supported":
                return ReasoningResponseLanguage._select_variant(
                    seed=seed,
                    options=(
                        "Here’s a solid plan.",
                        "Here’s a solid first plan.",
                    ),
                )
            if confidence_posture == "conflicted":
                return ReasoningResponseLanguage._select_variant(
                    seed=seed,
                    options=(
                        "Here’s a grounded plan, with the tension kept visible.",
                        "Here’s the best current plan, with the tension kept visible.",
                    ),
                )
            return ReasoningResponseLanguage._select_variant(
                seed=seed,
                options=(
                    "Here’s a starting plan using the best current assumptions.",
                    "Here’s a useful first plan, with the assumptions kept visible.",
                ),
            )
        if confidence_posture == "strong":
            if intent_domain == "technical_engineering":
                return ReasoningResponseLanguage._select_variant(
                    seed=seed,
                    options=(
                        "Here’s the grounded technical answer.",
                        "Here’s the grounded technical read.",
                    ),
                )
            if intent_domain == "decision_support":
                return ReasoningResponseLanguage._select_variant(
                    seed=seed,
                    options=(
                        "Here’s the grounded tradeoff read.",
                        "Here’s the grounded decision read.",
                    ),
                )
            return ReasoningResponseLanguage._select_variant(
                seed=seed,
                options=(
                    "Here’s the grounded answer.",
                    "Here’s the grounded explanation.",
                ),
            )
        if confidence_posture == "supported":
            if intent_domain == "technical_engineering":
                return ReasoningResponseLanguage._select_variant(
                    seed=seed,
                    options=(
                        "Here’s the clearest current technical answer.",
                        "Here’s the clearest current technical read.",
                    ),
                )
            if intent_domain == "decision_support":
                return ReasoningResponseLanguage._select_variant(
                    seed=seed,
                    options=(
                        "Here’s the clearest current tradeoff read.",
                        "Here’s the clearest current decision read.",
                    ),
                )
            return ReasoningResponseLanguage._select_variant(
                seed=seed,
                options=(
                    "Here’s a workable answer.",
                    "Here’s the clearest current answer.",
                ),
            )
        if confidence_posture == "conflicted":
            return ReasoningResponseLanguage._select_variant(
                seed=seed,
                options=(
                    "Here’s the best grounded answer, with the tension kept visible.",
                    "Here’s a grounded answer, with the unresolved tension kept visible.",
                ),
            )
        return ReasoningResponseLanguage._select_variant(
            seed=seed,
            options=(
                "Here’s a useful first pass using the best current assumptions.",
                "Here’s a first pass, with the assumptions kept visible.",
            ),
        )

    @staticmethod
    def uncertainty_note(
        *,
        confidence_posture: str,
        local_context_assessment: str | None,
        reasoning_frame: dict[str, str],
        route_caution: str,
        route_ambiguity: bool,
        subject_label: str,
    ) -> str | None:
        tier = ReasoningResponseLanguage.language_confidence_tier(
            confidence_posture=confidence_posture,
        )
        if tier == "high":
            return None
        if local_context_assessment == "mixed":
            tension = reasoning_frame.get("tension")
            return ConversationPolicy.conflicted_evidence_note(
                subject_label=subject_label,
                tension=tension,
            )
        if tier == "low":
            anchor = reasoning_frame.get("primary_anchor")
            if route_ambiguity:
                return ConversationPolicy.route_ambiguity_note(subject_label=subject_label)
            if anchor:
                return ConversationPolicy.insufficient_evidence_note(
                    subject_label=subject_label,
                    anchor=anchor,
                )
            if route_caution:
                if subject_label == "plan":
                    return (
                        "We do not have enough evidence yet to settle this plan with confidence. "
                        "A clearer version of the request or a bit more grounded detail would help."
                    )
                return (
                    "We do not have enough evidence yet to settle this answer with confidence. "
                    "A clearer version of the question or a bit more grounded detail would help."
                )
            return ConversationPolicy.insufficient_evidence_note(subject_label=subject_label)
        if route_caution:
            return ConversationPolicy.route_caution_note(subject_label=subject_label)
        if route_ambiguity:
            return ConversationPolicy.route_ambiguity_note(subject_label=subject_label)
        return None

    @staticmethod
    def language_confidence_tier(*, confidence_posture: str) -> str:
        posture = str(confidence_posture or "").strip().lower()
        if posture == "strong":
            return "high"
        if posture == "supported":
            return "medium"
        if posture in {"tentative", "conflicted"}:
            return "low"
        return "medium"

    @staticmethod
    def tension_guidance(
        *,
        tension_resolution: dict[str, object] | None,
        fallback: str,
        subject_label: str,
    ) -> str:
        if not isinstance(tension_resolution, dict) or not tension_resolution.get("tension_detected"):
            return fallback
        resolution_path = str(tension_resolution.get("resolution_path") or "").strip()
        recommended_action = str(tension_resolution.get("recommended_action") or "").strip()
        leading_label = str(tension_resolution.get("leading_hypothesis_label") or "").strip()
        if resolution_path == "clarification":
            return (
                f"Treat the first {subject_label} as a clarification checkpoint, because conflicting evidence should be resolved before the anchor changes."
            )
        if resolution_path == "hypothesis_revision":
            return (
                f"Treat the first {subject_label} as a hypothesis-revision checkpoint, because the stronger competing evidence now outweighs the current anchor."
            )
        if resolution_path == "alternate_hypothesis":
            if leading_label:
                return (
                    f"Keep the first {subject_label} centered on the leading hypothesis ({leading_label}) while keeping the competing explanation explicit."
                )
            if recommended_action == "gather_missing_evidence":
                return (
                    f"Keep the first {subject_label} narrow until one more confirming source separates the competing explanations."
                )
            return (
                f"Keep the first {subject_label} narrow while the competing explanations remain unresolved."
            )
        return fallback

    @staticmethod
    def _select_variant(*, seed: str, options: tuple[str, ...]) -> str:
        if len(options) == 1:
            return options[0]
        index = sum(ord(char) for char in seed) % len(options)
        return options[index]
