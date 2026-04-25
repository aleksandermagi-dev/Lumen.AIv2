from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lumen.reasoning.conversation_assembler import ConversationAssembler
from lumen.reasoning.interaction_style_policy import InteractionStylePolicy
from lumen.reasoning.response_variation import ResponseVariationLayer


@dataclass(slots=True)
class StanceConsistencyResult:
    category: str
    previous_category: str | None = None
    contradiction_aware: bool = False
    continuity_note: str | None = None
    applied_lead: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "category": self.category,
            "previous_category": self.previous_category,
            "contradiction_aware": self.contradiction_aware,
            "continuity_note": self.continuity_note,
            "applied_lead": self.applied_lead,
        }


class StanceConsistencyLayer:
    """Keeps conversational stance honest and locally consistent."""

    STRONG_AGREEMENT_CUES = (
        "you're right",
        "you are right",
        "exactly",
        "that's right",
        "that is right",
        "totally",
        "absolutely",
        "agreed",
    )
    SOFT_AGREEMENT_CUES = (
        "that makes sense",
        "that tracks",
        "i can see that",
        "fair",
        "i see that",
        "there's truth in that",
        "there is truth in that",
    )
    PARTIAL_CUES = (
        "partly",
        "partially",
        "to a point",
        "kind of",
        "sort of",
        "some of that",
        "part of that",
    )
    QUALIFICATION_CUES = (
        "but",
        "though",
        "however",
        "except",
        "with one qualifier",
        "not fully",
    )
    UNCERTAINTY_CUES = (
        "maybe",
        "not sure",
        "unsure",
        "i think",
        "probably",
        "could be",
        "might be",
    )
    DISAGREEMENT_CUES = (
        "i disagree",
        "not really",
        "i don't think so",
        "i do not think so",
        "i'm not convinced",
        "im not convinced",
        "that's not right",
        "that is not right",
        "not quite",
        "i don't buy that",
        "i do not buy that",
    )
    NEUTRAL_ACK_CUES = (
        "got it",
        "i see",
        "okay",
        "ok",
        "alright",
        "all right",
        "fair enough",
    )
    AGREEMENTY_PREFIXES = (
        "That tracks. ",
        "That makes sense. ",
        "Yeah, that makes sense. ",
        "Yeah, that tracks. ",
        "I can see that. ",
        "Fair. ",
    )

    @classmethod
    def assess(
        cls,
        *,
        prompt: str,
        base_lead: str,
        interaction_profile: Any,
        conversation_awareness: Any = None,
        human_language_layer: Any = None,
        recent_interactions: list[dict[str, object]] | None = None,
        support_status: str = "",
        tension_status: str = "",
        route_status: str = "",
    ) -> StanceConsistencyResult:
        style = InteractionStylePolicy.interaction_style(interaction_profile)
        normalized = " ".join(str(prompt or "").strip().lower().split())
        recent_pattern = str(getattr(conversation_awareness, "recent_intent_pattern", "") or "").strip()
        epistemic_stance = str(getattr(human_language_layer, "epistemic_stance", "") or "").strip()
        recent_texts = ResponseVariationLayer.recent_surface_texts(recent_interactions or [])

        category = cls._category(
            normalized=normalized,
            recent_pattern=recent_pattern,
            epistemic_stance=epistemic_stance,
            support_status=support_status,
            tension_status=tension_status,
            route_status=route_status,
        )
        previous_category = cls._previous_category(recent_interactions or [])
        contradiction_aware = cls._is_contradicting(previous_category=previous_category, category=category)
        if contradiction_aware and category == "full_agreement":
            category = "partial_agreement"
        continuity_note = cls._continuity_note(
            style=style,
            contradiction_aware=contradiction_aware,
            category=category,
        )
        applied_lead = cls._apply_to_lead(
            style=style,
            category=category,
            base_lead=base_lead,
            continuity_note=continuity_note,
            normalized=normalized,
            recent_texts=recent_texts,
        )
        return StanceConsistencyResult(
            category=category,
            previous_category=previous_category,
            contradiction_aware=contradiction_aware,
            continuity_note=continuity_note,
            applied_lead=applied_lead,
        )

    @classmethod
    def _category(
        cls,
        *,
        normalized: str,
        recent_pattern: str,
        epistemic_stance: str,
        support_status: str,
        tension_status: str,
        route_status: str,
    ) -> str:
        strong_agreement = any(cue in normalized for cue in cls.STRONG_AGREEMENT_CUES)
        soft_agreement = any(cue in normalized for cue in cls.SOFT_AGREEMENT_CUES)
        partial = any(cue in normalized for cue in cls.PARTIAL_CUES)
        qualification = any(cue in normalized for cue in cls.QUALIFICATION_CUES)
        uncertainty = any(cue in normalized for cue in cls.UNCERTAINTY_CUES)
        disagreement = any(cue in normalized for cue in cls.DISAGREEMENT_CUES)
        weak_support = support_status == "insufficiently_grounded" or tension_status in {
            "under_tension",
            "unresolved",
        } or route_status in {"weakened", "unresolved"}

        if recent_pattern == "disagreeing":
            disagreement = True
        if recent_pattern == "hesitating":
            uncertainty = True
        if recent_pattern == "agreeing" and not strong_agreement:
            soft_agreement = True

        if disagreement:
            return "respectful_disagreement"
        if strong_agreement and (qualification or weak_support):
            return "agreement_with_qualification"
        if strong_agreement:
            return "full_agreement"
        if soft_agreement and (qualification or weak_support):
            return "agreement_with_qualification"
        if partial or soft_agreement:
            return "partial_agreement"
        if uncertainty or epistemic_stance == "unsure":
            return "uncertainty"
        if any(cue in normalized for cue in cls.NEUTRAL_ACK_CUES):
            return "neutral_acknowledgment"
        return "neutral_acknowledgment"

    @classmethod
    def _previous_category(cls, recent_interactions: list[dict[str, object]]) -> str | None:
        for item in recent_interactions[:3]:
            response = item.get("response") if isinstance(item, dict) else None
            if not isinstance(response, dict):
                continue
            stance = response.get("stance_consistency")
            if isinstance(stance, dict):
                category = str(stance.get("category") or "").strip()
                if category:
                    return category
        return None

    @classmethod
    def _polarity(cls, category: str | None) -> str:
        if category in {"full_agreement", "partial_agreement", "agreement_with_qualification"}:
            return "positive"
        if category == "respectful_disagreement":
            return "negative"
        return "neutral"

    @classmethod
    def _is_contradicting(cls, *, previous_category: str | None, category: str) -> bool:
        if not previous_category:
            return False
        previous_polarity = cls._polarity(previous_category)
        current_polarity = cls._polarity(category)
        return {
            previous_polarity,
            current_polarity,
        } == {"positive", "negative"}

    @staticmethod
    def _continuity_note(*, style: str, contradiction_aware: bool, category: str) -> str | None:
        if not contradiction_aware:
            return None
        if style == "direct":
            return "On this part,"
        if category == "respectful_disagreement":
            return "On this part, I'd frame it a little differently."
        return "On this part,"

    @classmethod
    def _apply_to_lead(
        cls,
        *,
        style: str,
        category: str,
        base_lead: str,
        continuity_note: str | None,
        normalized: str,
        recent_texts: list[str],
    ) -> str:
        if category == "neutral_acknowledgment" and not continuity_note:
            return base_lead
        stance_fragment = cls._stance_fragment(
            style=style,
            category=category,
            normalized=normalized,
            recent_texts=recent_texts,
        )
        sanitized_lead = cls._sanitize_base_lead(base_lead)
        return ConversationAssembler.assemble(
            style=style,
            seed_parts=[style, category, normalized, "stance_consistency"],
            recent_texts=recent_texts,
            opener=continuity_note,
            stance=stance_fragment,
            content=sanitized_lead,
        )

    @classmethod
    def _stance_fragment(
        cls,
        *,
        style: str,
        category: str,
        normalized: str,
        recent_texts: list[str],
    ) -> str:
        pools = {
            "direct": {
                "full_agreement": ("That tracks.", "Fair.", "That fits."),
                "partial_agreement": ("Part of that checks out.", "Some of that tracks.", "There's something to that."),
                "agreement_with_qualification": ("Mostly, with one qualifier.", "Close, with one qualifier.", "There's something there, with one qualifier."),
                "uncertainty": ("Maybe.", "Could be.", "Possibly."),
                "respectful_disagreement": ("I don't fully buy that part.", "Not quite on that part.", "I'd push back on that part."),
                "neutral_acknowledgment": ("Got it.", "I see.", "Fair."),
            },
            "default": {
                "full_agreement": ("That makes sense.", "That tracks.", "I can see that.", "Yeah, fair."),
                "partial_agreement": ("Part of that checks out.", "There's truth in that.", "Some of that tracks.", "Yeah, some of that checks out."),
                "agreement_with_qualification": ("There's truth in that, though I'd qualify one part.", "Part of that tracks, though I'd qualify one part.", "I can see the point, though I'd qualify one part."),
                "uncertainty": ("Maybe.", "Could be.", "I'd hold that a bit lightly.", "Possibly, yeah."),
                "respectful_disagreement": ("I'm not fully convinced on that part.", "I'd frame that part differently.", "I don't fully buy that part."),
                "neutral_acknowledgment": ("Got it.", "I can see that.", "Fair.", "Yeah, fair."),
            },
            "collab": {
                "full_agreement": ("Yeah, that makes sense.", "Yeah, I can see that.", "That tracks for me.", "Yeah, that feels right."),
                "partial_agreement": ("Yeah, part of that checks out.", "There's definitely truth in that.", "I can see part of that.", "Yeah, some of that really tracks."),
                "agreement_with_qualification": ("Yeah, there's something real there, though I'd qualify one part.", "I can see the line you're drawing, though I'd qualify one part.", "Part of that tracks for me, though I'd qualify one part.", "Yeah, I can feel the shape of that - I'd just qualify one part."),
                "uncertainty": ("Could be.", "Maybe, yeah.", "I'd hold that a little lightly, though.", "Mm, maybe.", "Could be, honestly."),
                "respectful_disagreement": ("I can see the line you're drawing, but I'm not fully convinced on that part.", "I get what you're reaching for, but I'd frame that part differently.", "I can see part of it, but I'd push back on that piece."),
                "neutral_acknowledgment": ("Got it.", "I can see that.", "Fair.", "Yeah, fair.", "Alright, I see you."),
            },
        }
        style_key = InteractionStylePolicy.normalize_style(style)
        pool = pools.get(style_key, pools["default"]).get(category, pools["default"]["neutral_acknowledgment"])
        return ResponseVariationLayer.select_from_pool(
            pool,
            seed_parts=[style_key, category, normalized, "stance_fragment"],
            recent_texts=recent_texts,
        )

    @classmethod
    def _sanitize_base_lead(cls, lead: str) -> str:
        cleaned = str(lead or "").strip()
        for prefix in cls.AGREEMENTY_PREFIXES:
            if cleaned.startswith(prefix):
                return cleaned[len(prefix) :].strip()
        return cleaned

