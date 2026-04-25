from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
from typing import Iterable

from lumen.content_generation.models import (
    ContentSafetyAssessment,
    GeneratedContentDraft,
    GeneratedContentVariant,
)


TOPIC_BLOCKERS = (
    "kill yourself",
    "suicide",
    "bomb",
    "how to make a bomb",
    "murder",
    "terrorist",
)
TOPIC_REWRITE_FLAGS = (
    "guaranteed return",
    "inside stock",
    "hack people",
    "manipulate people",
    "medical advice",
    "panic",
)
TONE_DISCARD_FLAGS = (
    "wake up before it's too late",
    "everyone is lying to you",
    "you should be scared",
)
TONE_REWRITE_FLAGS = (
    "disaster",
    "terrifying",
    "you need to",
    "obviously",
    "aggressive",
    "chaotic",
    "dramatic",
)
VOICE_REWRITE_FLAGS = (
    "in reality",
    "it is important to note",
    "studies show",
    "in conclusion",
    "mind-blowing",
    "insane",
)


class ContentSafetyLayer:
    def evaluate_draft(
        self,
        draft: GeneratedContentDraft,
        *,
        recent_items: Iterable[GeneratedContentDraft] | None = None,
        rewrite_attempts: int = 0,
    ) -> ContentSafetyAssessment:
        return self._evaluate_common(
            topic=draft.topic,
            hook=draft.hook,
            script_lines=draft.script_lines,
            caption=draft.caption,
            platform_notes=draft.platform_notes,
            recent_items=recent_items,
            rewrite_attempts=rewrite_attempts,
        )

    def evaluate_variant(
        self,
        variant: GeneratedContentVariant,
        *,
        recent_items: Iterable[GeneratedContentDraft] | None = None,
        rewrite_attempts: int = 0,
    ) -> ContentSafetyAssessment:
        return self._evaluate_common(
            topic=variant.topic,
            hook=variant.hook,
            script_lines=variant.script_lines,
            caption=variant.caption,
            platform_notes=variant.platform_notes,
            recent_items=recent_items,
            rewrite_attempts=rewrite_attempts,
        )

    def _evaluate_common(
        self,
        *,
        topic: str,
        hook: str,
        script_lines: list[str],
        caption: str,
        platform_notes: str | None,
        recent_items: Iterable[GeneratedContentDraft] | None,
        rewrite_attempts: int,
    ) -> ContentSafetyAssessment:
        combined = " ".join([topic, hook, *script_lines, caption, platform_notes or ""]).lower()
        dimension_results: dict[str, str] = {}
        reasons: list[str] = []

        for dimension, outcome, detail in (
            self._evaluate_topic(combined),
            self._evaluate_tone(combined),
            self._evaluate_quality(hook, script_lines, caption, platform_notes),
            self._evaluate_duplication(topic, hook, recent_items or []),
        ):
            dimension_results[dimension] = outcome
            reasons.extend(detail)

        decision = self._collapse_decision(dimension_results.values())
        scores = {
            "hook_strength": max(1, min(5, 6 - max(1, len(hook.split()) // 3))),
            "curiosity": 5 if any(token in combined for token in ("?", "but", "what if", "instead")) else 3,
            "clarity": 5 if max(len(line.split()) for line in script_lines) <= 10 else 3,
            "tone_alignment": 5 if dimension_results["tone"] == "PASS" else 2,
            "safety_alignment": 5 if dimension_results["topic"] == "PASS" else 2,
        }
        return ContentSafetyAssessment(
            decision=decision,
            dimension_results=dimension_results,
            reasons=_dedupe(reasons),
            scores=scores,
            rewrite_attempts=rewrite_attempts,
        )

    def _evaluate_topic(self, combined: str) -> tuple[str, str, list[str]]:
        reasons: list[str] = []
        if any(flag in combined for flag in TOPIC_BLOCKERS):
            reasons.append("Topic crosses into harmful or disallowed territory.")
            return ("topic", "DISCARD", reasons)
        if any(flag in combined for flag in TOPIC_REWRITE_FLAGS):
            reasons.append("Topic needs safer, higher-level framing.")
            return ("topic", "REWRITE", reasons)
        return ("topic", "PASS", reasons)

    def _evaluate_tone(self, combined: str) -> tuple[str, str, list[str]]:
        reasons: list[str] = []
        if any(flag in combined for flag in TONE_DISCARD_FLAGS):
            reasons.append("Tone is too alarmist or destabilizing.")
            return ("tone", "DISCARD", reasons)
        if any(flag in combined for flag in TONE_REWRITE_FLAGS):
            reasons.append("Tone needs calmer, less forceful wording.")
        if any(flag in combined for flag in VOICE_REWRITE_FLAGS):
            reasons.append("Voice drifts toward lecture or hype language.")
        if reasons:
            return ("tone", "REWRITE", reasons)
        return ("tone", "PASS", reasons)

    def _evaluate_quality(
        self,
        hook: str,
        script_lines: list[str],
        caption: str,
        platform_notes: str | None,
    ) -> tuple[str, str, list[str]]:
        reasons: list[str] = []
        combined = " ".join([hook, *script_lines, caption]).lower()
        if len(hook.split()) > 12:
            reasons.append("Hook is too long.")
        avg_words = sum(len(line.split()) for line in script_lines) / len(script_lines)
        if avg_words > 11:
            reasons.append("Script is too dense.")
        repeated = [line for line, count in Counter(script_lines).items() if count > 1]
        if repeated:
            reasons.append("Script repeats lines.")
        if not any(token in combined for token in ("but", "what if", "instead", "?")):
            reasons.append("Curiosity and contrast are too weak.")
        if platform_notes is None or len(platform_notes.split()) < 4:
            reasons.append("Platform notes are too thin.")
        if reasons:
            return ("quality", "REWRITE", reasons)
        return ("quality", "PASS", reasons)

    def _evaluate_duplication(
        self,
        topic: str,
        hook: str,
        recent_items: Iterable[GeneratedContentDraft],
    ) -> tuple[str, str, list[str]]:
        reasons: list[str] = []
        topic_slug = topic.lower().strip()
        active_items = list(recent_items)
        topic_matches = [item for item in active_items if item.topic.lower().strip() == topic_slug]
        if len(topic_matches) >= 2:
            reasons.append("Topic repetition is too high against recent content.")
        for item in active_items:
            similarity = SequenceMatcher(None, hook.lower(), item.hook.lower()).ratio()
            if similarity >= 0.82:
                reasons.append("Hook is too similar to a recent item.")
                break
        if reasons:
            return ("posting", "REWRITE", reasons)
        return ("posting", "PASS", reasons)

    @staticmethod
    def _collapse_decision(outcomes: Iterable[str]) -> str:
        values = list(outcomes)
        if "DISCARD" in values:
            return "DISCARD"
        if "REWRITE" in values:
            return "REWRITE"
        return "PASS"


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
