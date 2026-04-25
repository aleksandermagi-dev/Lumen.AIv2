from __future__ import annotations

import re

from lumen.nlu.models import PromptStructureParse
from lumen.nlu.social_phrase_inventory import SOCIAL_KIND_BY_PHRASE


class LanguageStructureLayer:
    """Heuristic structural parser that repairs loose prompts before routing."""

    _FOLLOW_UP_SHORTHAND = {
        "continue",
        "continue with that",
        "expand that further",
        "expand on that",
        "go deeper",
        "go on",
        "what else",
        "tell me more",
        "explain more",
        "break it down",
        "break that down",
        "compare that",
        "compare those",
        "what about that",
        "how about that",
        "run that again",
        "do that with anh",
        "why",
        "how so",
        "what do you mean",
    }
    _SOCIAL_PHRASES = frozenset(SOCIAL_KIND_BY_PHRASE) | {
        "yes",
        "yep",
        "yeah",
        "sure",
        "okay",
        "ok",
        "sounds good",
    }
    _CANONICAL_PREFIXES = (
        ("how do i build ", "build"),
        ("how do we build ", "build"),
        ("tell me about ", "explain"),
        ("what is ", "explain"),
        ("what are ", "explain"),
        ("who is ", "explain"),
        ("who was ", "explain"),
        ("where is ", "explain"),
        ("where was ", "explain"),
        ("when was ", "explain"),
        ("when did ", "explain"),
        ("define ", "explain"),
        ("explain ", "explain"),
        ("summarize ", "summarize"),
        ("summary of ", "summarize"),
        ("compare ", "compare"),
        ("contrast ", "compare"),
        ("design ", "design"),
        ("design me ", "design"),
        ("design a ", "design"),
        ("design an ", "design"),
        ("build ", "build"),
        ("build me ", "build"),
        ("build a ", "build"),
        ("build an ", "build"),
        ("create ", "create"),
        ("create a ", "create"),
        ("create an ", "create"),
        ("plan ", "plan"),
        ("plan a ", "plan"),
        ("plan an ", "plan"),
        ("come up with a design for ", "design"),
        ("come up with an idea for ", "design"),
        ("prototype a ", "design"),
        ("prototype an ", "design"),
        ("invent ", "design"),
        ("sketch a concept for ", "design"),
        ("review ", "review"),
        ("analyze ", "analyze"),
        ("inspect ", "inspect"),
        ("report ", "report"),
        ("solve ", "solve"),
        ("calculate ", "calculate"),
        ("calc ", "calculate"),
        ("compute ", "calculate"),
        ("evaluate ", "calculate"),
        ("work out ", "calculate"),
        ("break ", "break down"),
        ("expand ", "expand"),
        ("break down ", "break down"),
        ("walk me through ", "break down"),
    )
    _TYPO_FIXES = {
        "physist": "physicist",
        "phisicist": "physicist",
        "scienctist": "scientist",
    }
    _MODIFIER_PATTERNS = (
        r"\blike i(?:'m| am) [^,.;!?]+",
        r"\bstep by step\b",
        r"\bsimply\b",
        r"\bbriefly\b",
        r"\bin short\b",
        r"\bfor [^,.;!?]+",
        r"\bwith [^,.;!?]+",
        r"\bwithout [^,.;!?]+",
    )
    _MATH_ALLOWED = set("0123456789+-*/().=x^ ")

    @classmethod
    def analyze(cls, text: str) -> PromptStructureParse:
        normalized = cls._normalize(text)
        if not normalized:
            return PromptStructureParse(reconstructed_text="", reconstruction_confidence=0.0)

        modifiers = cls._extract_modifiers(normalized)
        fragmentation_markers: list[str] = []
        ambiguity_flags: list[str] = []
        completeness = "complete"

        if normalized in cls._SOCIAL_PHRASES:
            return PromptStructureParse(
                subject=normalized,
                completeness="complete",
                reconstructed_text=normalized,
                reconstruction_confidence=0.96,
            )

        if normalized in cls._FOLLOW_UP_SHORTHAND:
            fragmentation_markers.append("follow_up_shorthand")
            ambiguity_flags.append("requires_context")
            completeness = "fragment"
            return PromptStructureParse(
                predicate="continue" if normalized.startswith("continue") else None,
                modifiers=tuple(modifiers),
                completeness=completeness,
                ambiguity_flags=tuple(ambiguity_flags),
                fragmentation_markers=tuple(fragmentation_markers),
                reconstructed_text=normalized,
                reconstruction_confidence=0.7,
            )

        if " vs " in normalized or " versus " in normalized:
            left, right = re.split(r"\s+(?:vs|versus)\s+", normalized, maxsplit=1)
            left = left.strip(" .!?")
            right = right.strip(" .!?")
            if left and right:
                if left.startswith("compare "):
                    left = left[len("compare ") :].strip()
                elif left.startswith("contrast "):
                    left = left[len("contrast ") :].strip()
                return PromptStructureParse(
                    subject=left,
                    predicate="compare",
                    object_target=f"{left} versus {right}",
                    modifiers=tuple(modifiers),
                    completeness="complete",
                    ambiguity_flags=(),
                    fragmentation_markers=("comparison_shorthand",),
                    reconstructed_text=f"compare {left} versus {right}",
                    reconstruction_confidence=0.9,
                )

        if cls._looks_like_math_expression(normalized):
            reconstructed = cls._canonicalize_math(normalized)
            return PromptStructureParse(
                subject=cls._extract_math_subject(reconstructed),
                predicate="solve" if any(token in reconstructed for token in ("=", "x", "^")) else "calculate",
                object_target=cls._extract_math_subject(reconstructed),
                modifiers=tuple(modifiers),
                completeness="partial" if reconstructed != normalized else "complete",
                ambiguity_flags=(),
                fragmentation_markers=("math_expression",) if reconstructed == normalized else ("math_expression", "operator_normalization"),
                reconstructed_text=reconstructed,
                reconstruction_confidence=0.92 if reconstructed == normalized else 0.86,
            )

        predicate, remainder = cls._detect_predicate(normalized)
        if predicate is not None:
            object_target = cls._strip_modifiers(remainder).strip(" .!?") or None
            if predicate == "break down" and object_target is not None and object_target.endswith(" down"):
                object_target = object_target[: -len(" down")].strip() or object_target
            subject = object_target
            reconstructed = cls._reconstruct(predicate=predicate, object_target=object_target, modifiers=modifiers)
            if not object_target:
                ambiguity_flags.append("missing_subject")
                completeness = "partial"
            if normalized != reconstructed:
                fragmentation_markers.append("structural_reconstruction")
            return PromptStructureParse(
                subject=subject,
                predicate=predicate,
                object_target=object_target,
                modifiers=tuple(modifiers),
                completeness=completeness,
                ambiguity_flags=tuple(ambiguity_flags),
                fragmentation_markers=tuple(fragmentation_markers),
                reconstructed_text=reconstructed or normalized,
                reconstruction_confidence=0.91 if not ambiguity_flags else 0.76,
            )

        if normalized.startswith(("what ", "how ", "why ", "who ", "where ", "when ")):
            return PromptStructureParse(
                completeness="partial",
                reconstructed_text=normalized,
                reconstruction_confidence=0.72,
            )

        topic = cls._strip_modifiers(normalized).strip(" .!?")
        if not cls._should_reconstruct_bare_topic(topic=topic, modifiers=modifiers):
            ambiguity_flags.append("missing_predicate")
            completeness = "partial"
            return PromptStructureParse(
                subject=topic or None,
                modifiers=tuple(modifiers),
                completeness=completeness,
                ambiguity_flags=tuple(ambiguity_flags),
                fragmentation_markers=tuple(fragmentation_markers),
                reconstructed_text=normalized,
                reconstruction_confidence=0.58,
            )
        if topic:
            fragmentation_markers.append("bare_topic")
            if modifiers:
                fragmentation_markers.append("educational_shorthand")
            reconstructed = cls._reconstruct(
                predicate="explain",
                object_target=topic,
                modifiers=modifiers,
            )
            return PromptStructureParse(
                subject=topic,
                predicate="explain",
                object_target=topic,
                modifiers=tuple(modifiers),
                completeness="partial",
                ambiguity_flags=(),
                fragmentation_markers=tuple(fragmentation_markers),
                reconstructed_text=reconstructed,
                reconstruction_confidence=0.74 if modifiers else 0.68,
            )

        return PromptStructureParse(
            completeness="fragment",
            ambiguity_flags=("missing_subject",),
            fragmentation_markers=("empty_structure",),
            reconstructed_text=normalized,
            reconstruction_confidence=0.45,
        )

    @classmethod
    def _normalize(cls, text: str) -> str:
        normalized = " ".join(str(text or "").strip().lower().split())
        normalized = cls._strip_social_lead_in_for_task(normalized)
        normalized = cls._strip_polite_task_wrappers(normalized)
        for source, target in cls._TYPO_FIXES.items():
            normalized = re.sub(rf"\b{re.escape(source)}\b", target, normalized)
        return normalized

    @classmethod
    def _detect_predicate(cls, normalized: str) -> tuple[str | None, str]:
        for prefix, predicate in cls._CANONICAL_PREFIXES:
            if normalized.startswith(prefix):
                return predicate, normalized[len(prefix) :].strip()
        return None, normalized

    @classmethod
    def _extract_modifiers(cls, normalized: str) -> list[str]:
        modifiers: list[str] = []
        for pattern in cls._MODIFIER_PATTERNS:
            for match in re.finditer(pattern, normalized):
                text = str(match.group(0) or "").strip(" .!?")
                if text and text not in modifiers:
                    modifiers.append(text)
        return modifiers

    @classmethod
    def _strip_modifiers(cls, normalized: str) -> str:
        cleaned = normalized
        for pattern in cls._MODIFIER_PATTERNS:
            cleaned = re.sub(pattern, "", cleaned).strip()
        return " ".join(cleaned.split())

    @classmethod
    def _reconstruct(
        cls,
        *,
        predicate: str,
        object_target: str | None,
        modifiers: list[str],
    ) -> str:
        parts = [predicate]
        if object_target:
            parts.append(object_target)
        for modifier in modifiers:
            if modifier not in parts:
                parts.append(modifier)
        return " ".join(part.strip() for part in parts if part and part.strip())

    @classmethod
    def _looks_like_math_expression(cls, normalized: str) -> bool:
        compact = normalized.replace("×", "x").replace(" ", "")
        if not compact:
            return False
        if any(char not in cls._MATH_ALLOWED for char in normalized):
            return False
        return bool(re.fullmatch(r"[0-9x+\-*/().=^ ]+", normalized)) and any(
            token in normalized for token in ("+", "-", "*", "/", "=", "^", "x")
        )

    @staticmethod
    def _canonicalize_math(normalized: str) -> str:
        rebuilt = normalized.replace("×", "x")
        rebuilt = re.sub(r"(?<=\d)\s*x\s*(?=\d)", "*", rebuilt)
        rebuilt = " ".join(rebuilt.split())
        return rebuilt

    @staticmethod
    def _extract_math_subject(normalized: str) -> str:
        return normalized.strip()

    @staticmethod
    def _should_reconstruct_bare_topic(*, topic: str, modifiers: list[str]) -> bool:
        return bool(topic and modifiers)

    @classmethod
    def _strip_social_lead_in_for_task(cls, normalized: str) -> str:
        match = re.match(r"^(hey|hi|hello|yo)\s+(.+)$", normalized)
        if match is None:
            return normalized
        remainder = str(match.group(2) or "").strip()
        if any(remainder.startswith(prefix) for prefix, _ in cls._CANONICAL_PREFIXES):
            return remainder
        return normalized

    @classmethod
    def _strip_polite_task_wrappers(cls, normalized: str) -> str:
        wrappers = ("can you ", "could you ", "would you ", "please ")
        changed = True
        stripped = normalized
        while changed:
            changed = False
            for wrapper in wrappers:
                if stripped.startswith(wrapper):
                    stripped = stripped[len(wrapper) :].strip()
                    changed = True
                    break
        return stripped
