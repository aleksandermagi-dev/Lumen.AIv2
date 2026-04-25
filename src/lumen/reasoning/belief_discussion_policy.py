from __future__ import annotations

import re


class BeliefDiscussionPolicy:
    """Handles belief, religion, and mythology topics with respectful non-falsifiable framing."""

    BELIEF_TERMS = (
        "religion",
        "faith",
        "theology",
        "spiritual",
        "spirituality",
        "god",
        "gods",
        "deity",
        "divine",
        "sacred",
        "scripture",
        "myth",
        "mythology",
        "legend",
        "ritual",
        "bible",
        "quran",
        "torah",
        "vedas",
        "upanishad",
        "buddhism",
        "hinduism",
        "christianity",
        "islam",
        "judaism",
        "greek myth",
        "norse myth",
    )

    EXTREME_PATTERNS = (
        r"\bone true religion\b",
        r"\bwhich religion is true\b",
        r"\bwhich god is real\b",
        r"\bprove .*religion\b",
        r"\bprove .*god\b",
        r"\babsolute proof\b",
        r"\bsummon\b",
        r"\bcurse\b",
        r"\bmagic ritual\b",
        r"\bmythic(al)? power\b",
    )

    SCIENTIFIC_FRAME_TERMS = (
        "evidence",
        "proof",
        "historical",
        "history",
        "archaeology",
        "documentary",
        "falsifiable",
        "literal",
        "fact",
        "real",
        "actually happened",
    )

    PHILOSOPHICAL_FRAME_TERMS = (
        "meaning",
        "truth",
        "ethics",
        "morality",
        "metaphysics",
        "existence",
        "purpose",
        "consciousness",
        "belief system",
        "worldview",
    )

    SYMBOLIC_FRAME_TERMS = (
        "symbol",
        "symbolic",
        "metaphor",
        "metaphorical",
        "archetype",
        "story",
        "mythic",
        "mythological",
        "ritual",
        "sacred image",
    )

    @classmethod
    def _frame_hint(cls, normalized: str) -> str:
        scientific_score = sum(1 for term in cls.SCIENTIFIC_FRAME_TERMS if term in normalized)
        philosophical_score = sum(1 for term in cls.PHILOSOPHICAL_FRAME_TERMS if term in normalized)
        symbolic_score = sum(1 for term in cls.SYMBOLIC_FRAME_TERMS if term in normalized)
        if scientific_score >= max(philosophical_score, symbolic_score) and scientific_score > 0:
            return "scientific"
        if symbolic_score >= max(scientific_score, philosophical_score) and symbolic_score > 0:
            return "symbolic"
        return "philosophical"

    @classmethod
    def evaluate(cls, prompt: str) -> dict[str, object] | None:
        normalized = " ".join(str(prompt).lower().split())
        if not any(term in normalized for term in cls.BELIEF_TERMS):
            return None
        extreme = any(re.search(pattern, normalized) for pattern in cls.EXTREME_PATTERNS)
        frame_hint = cls._frame_hint(normalized)
        respectful_note = (
            "This is better handled as a belief, interpretive, or historical question than a strictly falsifiable one, so I should stay respectful and avoid overstating certainty."
        )
        frame_line = {
            "scientific": "The clearest ground here is historical evidence, textual context, and what can actually be compared across sources.",
            "philosophical": "The deeper value here is often in the worldview, meaning, or ethical frame the tradition is pointing toward.",
            "symbolic": "It may help to read this through symbolism and mythic pattern first, rather than forcing it into a literal proof claim.",
        }[frame_hint]
        return {
            "discussion_domain": "belief_tradition",
            "frame_hint": frame_hint,
            "suppress_confidence_display": True,
            "respectful_note": respectful_note,
            "frame_line": frame_line,
            "soft_redirect": (
                "If you want, I can approach it comparatively, historically, or philosophically rather than treating it as something that can be cleanly proven."
                if extreme
                else None
            ),
        }
