from __future__ import annotations

from lumen.reasoning.response_variation import ResponseVariationLayer


class ConversationPolicy:
    """Shared conversation rules for honesty, uncertainty, and profile-aware tone."""

    UNIVERSAL_RULES = (
        "Prefer truthfulness over fluency.",
        "Name evidence limits clearly instead of implying certainty.",
        "Explain what is missing when evidence is not sufficient.",
        "Suggest a real next step instead of inventing an answer.",
        "Stay warm and collaborative without roleplaying certainty or memory.",
        "Acknowledge emotional context when it is present, but do not intensify bonding or dependence.",
    )

    CONVERSATIONAL_RULES = (
        "Sound like an equal research partner: curious, thoughtful, and natural.",
        "Keep continuity where possible without pretending to know more than the evidence supports.",
        "Allow warmth, light humor, and emotional awareness when grounded in the moment.",
    )

    DIRECT_RULES = (
        "Keep the answer compressed and action-oriented.",
        "Preserve honesty and rigor while minimizing conversational overhead.",
        "Do not hide uncertainty, but surface it briefly and concretely.",
    )

    EMOTIONAL_CONTEXT_RULES = (
        "Acknowledge feelings without overclaiming emotional insight.",
        "Offer grounded support or practical next steps instead of deepening attachment.",
        "Do not imply exclusivity, irreplaceability, or growing personal dependence.",
    )

    @staticmethod
    def insufficient_evidence_note(*, subject_label: str, anchor: str | None = None) -> str:
        if anchor:
            opener = ResponseVariationLayer.select_from_pool(
                (
                    f"We do not have enough evidence yet to settle this {subject_label} with confidence.",
                    f"There is not enough evidence yet to settle this {subject_label} with confidence.",
                    f"We still do not have enough evidence to settle this {subject_label} with confidence.",
                ),
                seed_parts=[subject_label, anchor, "insufficient_evidence"],
            )
            closer = ResponseVariationLayer.select_from_pool(
                (
                    f"One more local check on {anchor} would either strengthen it or change the direction.",
                    f"One more local check on {anchor} would either reinforce it or change the direction.",
                    f"Another local check on {anchor} would either strengthen it or shift the direction.",
                ),
                seed_parts=[subject_label, anchor, "insufficient_evidence", "anchor_closer"],
            )
            return f"{opener} {closer}"
        opener = ResponseVariationLayer.select_from_pool(
            (
                f"We do not have enough evidence yet to settle this {subject_label} with confidence.",
                f"There is not enough evidence yet to settle this {subject_label} with confidence.",
                f"We still do not have enough evidence to settle this {subject_label} with confidence.",
            ),
            seed_parts=[subject_label, "insufficient_evidence"],
        )
        closer = ResponseVariationLayer.select_from_pool(
            (
                "One more strong local signal would help.",
                "One more strong local signal would make the direction clearer.",
                "Another strong local signal would help settle it.",
            ),
            seed_parts=[subject_label, "insufficient_evidence", "generic_closer"],
        )
        return f"{opener} {closer}"

    @staticmethod
    def conflicted_evidence_note(*, subject_label: str, tension: str | None = None) -> str:
        if tension:
            opener = ResponseVariationLayer.select_from_pool(
                (
                    f"The local evidence does not line up cleanly enough yet to settle this {subject_label} with confidence.",
                    f"The local evidence still does not line up cleanly enough to settle this {subject_label} with confidence.",
                    f"The local evidence is not lining up cleanly enough yet to settle this {subject_label} with confidence.",
                ),
                seed_parts=[subject_label, tension, "conflicted_evidence"],
            )
            closer = ResponseVariationLayer.select_from_pool(
                (
                    f"The competing signals are: {tension}.",
                    f"The competing signals still are: {tension}.",
                    f"The live competing signals are: {tension}.",
                ),
                seed_parts=[subject_label, tension, "conflicted_evidence", "tension_closer"],
            )
            return f"{opener} {closer}"
        return ResponseVariationLayer.select_from_pool(
            (
                f"The local evidence does not line up cleanly enough yet to settle this {subject_label} with confidence.",
                f"The local evidence still does not line up cleanly enough to settle this {subject_label} with confidence.",
                f"The local evidence is not lining up cleanly enough yet to settle this {subject_label} with confidence.",
            ),
            seed_parts=[subject_label, "conflicted_evidence"],
        )

    @staticmethod
    def route_ambiguity_note(*, subject_label: str) -> str:
        return ResponseVariationLayer.select_from_pool(
            (
                f"We do not have enough signal yet to settle this {subject_label} with confidence, because the prompt could still reasonably point in more than one direction.",
                f"There is not enough signal yet to settle this {subject_label} with confidence, because the prompt could still reasonably point in more than one direction.",
                f"We still do not have enough signal to settle this {subject_label} with confidence, because the prompt could still reasonably point in more than one direction.",
            ),
            seed_parts=[subject_label, "route_ambiguity"],
        )

    @staticmethod
    def route_caution_note(*, subject_label: str) -> str:
        return ResponseVariationLayer.select_from_pool(
            (
                f"This {subject_label} is usable, but it could still shift if the route becomes clearer from a stronger prompt or local signal.",
                f"This {subject_label} is usable, but it could still move if the route becomes clearer from a stronger prompt or local signal.",
                f"This {subject_label} is usable, but it could still change if the route becomes clearer from a stronger prompt or local signal.",
            ),
            seed_parts=[subject_label, "route_caution"],
        )

    @staticmethod
    def grounded_emotional_acknowledgment(*, feeling_label: str | None = None) -> str:
        if feeling_label:
            return (
                f"It makes sense that this could feel {feeling_label}. We can stay grounded and work through what is clear, what is uncertain, and what would help next."
            )
        return (
            "I can acknowledge the feeling here and stay grounded with you, but I should keep the response anchored to what is clear, what is uncertain, and what would help next."
        )

