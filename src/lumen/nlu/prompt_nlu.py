from __future__ import annotations

from lumen.nlu.language_structure_layer import LanguageStructureLayer
from lumen.nlu.models import ProfileAdvice
from lumen.nlu.language_detector import LanguageDetector
from lumen.nlu.intent_extractor import IntentExtractor
from lumen.nlu.models import PromptUnderstanding
from lumen.nlu.models import PromptSurfaceViews
from lumen.nlu.prompt_surface_builder import PromptSurfaceBuilder
from lumen.nlu.topic_normalizer import TopicNormalizer


class PromptNLU:
    """Builds the canonical prompt-understanding contract for downstream layers."""

    def __init__(self) -> None:
        self.language_detector = LanguageDetector()
        self.language_structure_layer = LanguageStructureLayer()
        self.topic_normalizer = TopicNormalizer()
        self.intent_extractor = IntentExtractor()

    def analyze(self, text: str) -> PromptUnderstanding:
        base_surface_views = PromptSurfaceBuilder.build(text)
        structure = self.language_structure_layer.analyze(base_surface_views.route_ready_text)
        surface_views = self._structured_surface_views(
            original=base_surface_views,
            reconstructed_text=structure.reconstructed_text or base_surface_views.intent_ready_text,
        )
        topic = self.topic_normalizer.normalize(surface_views.reconstructed_text)
        profile_advice = self._suggest_profile(surface_views.reconstructed_text)
        return PromptUnderstanding(
            original_text=text,
            normalized_text=surface_views.normalized_text,
            intent_ready_text=surface_views.intent_ready_text,
            surface_views=surface_views,
            structure=structure,
            language=self.language_detector.detect(surface_views.normalized_text),
            topic=topic,
            intent=self.intent_extractor.extract_intent(surface_views.reconstructed_text, topic=topic),
            entities=self.intent_extractor.extract_entities(
                surface_views.reconstructed_text,
                topic=topic,
                original_text=text,
            ),
            profile_advice=profile_advice,
        )

    @staticmethod
    def _structured_surface_views(
        *,
        original: PromptSurfaceViews,
        reconstructed_text: str,
    ) -> PromptSurfaceViews:
        reconstructed = PromptSurfaceBuilder._strip_leading_address(reconstructed_text).strip()
        lookup_text = reconstructed.rstrip("?.!").strip()
        return PromptSurfaceViews(
            raw_text=original.raw_text,
            normalized_text=original.normalized_text,
            intent_ready_text=original.intent_ready_text,
            reconstructed_text=reconstructed or original.intent_ready_text,
            route_ready_text=reconstructed or original.route_ready_text,
            lookup_ready_text=lookup_text or original.lookup_ready_text,
            tool_ready_text=reconstructed or original.tool_ready_text,
            tool_source_text=original.tool_source_text,
        )

    @staticmethod
    def _suggest_profile(text: str) -> ProfileAdvice | None:
        normalized = text.lower()
        direct_cues = (
            "brief",
            "concise",
            "short",
            "quick",
            "direct",
            "straight",
            "just answer",
            "just give me",
        )
        conversational_cues = (
            "brainstorm",
            "explore",
            "walk me through",
            "talk through",
            "let's",
            "thinking out loud",
            "work through",
        )
        deep_cues = (
            "deep",
            "thorough",
            "detailed",
            "analyze",
            "reason",
            "research",
            "compare",
            "tradeoff",
            "architecture",
            "hypothesis",
            "why",
        )
        normal_cues = (
            "high-level",
            "quick",
            "brief",
            "summary",
            "simple answer",
        )

        style = "conversational"
        depth = "normal"
        confidence = 0.55

        if any(cue in normalized for cue in direct_cues):
            style = "direct"
            confidence = 0.78
        elif any(cue in normalized for cue in conversational_cues):
            style = "conversational"
            confidence = 0.72

        if any(cue in normalized for cue in deep_cues):
            depth = "deep"
            confidence = max(confidence, 0.8)
        elif any(cue in normalized for cue in normal_cues):
            depth = "normal"
            confidence = max(confidence, 0.65)

        if confidence <= 0.56:
            return None

        return ProfileAdvice(
            interaction_style=style,
            reasoning_depth=depth,
            confidence=round(confidence, 2),
        )
