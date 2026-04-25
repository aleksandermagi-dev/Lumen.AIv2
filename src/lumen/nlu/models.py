from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class DetectedLanguage:
    code: str
    confidence: float
    source: str = "heuristic"

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "confidence": self.confidence,
            "source": self.source,
        }


@dataclass(slots=True)
class NormalizedTopic:
    value: str | None
    tokens: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return {
            "value": self.value,
            "tokens": list(self.tokens),
        }


@dataclass(slots=True)
class ExtractedEntity:
    label: str
    value: str
    confidence: float

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "value": self.value,
            "confidence": self.confidence,
        }


@dataclass(slots=True)
class IntentParse:
    label: str
    confidence: float
    source: str = "heuristic"

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "source": self.source,
        }


@dataclass(slots=True)
class ProfileAdvice:
    interaction_style: str
    reasoning_depth: str
    selection_source: str = "suggested"
    confidence: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "interaction_style": self.interaction_style,
            "reasoning_depth": self.reasoning_depth,
            "selection_source": self.selection_source,
            "confidence": self.confidence,
        }


@dataclass(slots=True)
class PromptSurfaceViews:
    raw_text: str
    normalized_text: str
    intent_ready_text: str
    reconstructed_text: str
    route_ready_text: str
    lookup_ready_text: str
    tool_ready_text: str
    tool_source_text: str

    def to_dict(self) -> dict[str, object]:
        return {
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "intent_ready_text": self.intent_ready_text,
            "reconstructed_text": self.reconstructed_text,
            "route_ready_text": self.route_ready_text,
            "lookup_ready_text": self.lookup_ready_text,
            "tool_ready_text": self.tool_ready_text,
            "tool_source_text": self.tool_source_text,
        }


@dataclass(slots=True)
class PromptStructureParse:
    subject: str | None = None
    predicate: str | None = None
    object_target: str | None = None
    modifiers: tuple[str, ...] = field(default_factory=tuple)
    completeness: str = "complete"
    ambiguity_flags: tuple[str, ...] = field(default_factory=tuple)
    fragmentation_markers: tuple[str, ...] = field(default_factory=tuple)
    reconstructed_text: str = ""
    reconstruction_confidence: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object_target": self.object_target,
            "modifiers": list(self.modifiers),
            "completeness": self.completeness,
            "ambiguity_flags": list(self.ambiguity_flags),
            "fragmentation_markers": list(self.fragmentation_markers),
            "reconstructed_text": self.reconstructed_text,
            "reconstruction_confidence": self.reconstruction_confidence,
        }


@dataclass(slots=True, frozen=True)
class RouterInputView:
    original_text: str
    normalized_text: str
    canonical_text: str
    route_ready_text: str
    detected_language: str
    normalized_topic: str | None
    dominant_intent: str
    intent_confidence: float
    extracted_entities: tuple[dict[str, object], ...] = field(default_factory=tuple)
    structure_subject: str | None = None
    structure_predicate: str | None = None
    structure_object: str | None = None
    structure_modifiers: tuple[str, ...] = field(default_factory=tuple)
    structure_completeness: str = "complete"
    structure_ambiguity_flags: tuple[str, ...] = field(default_factory=tuple)
    structure_fragmentation_markers: tuple[str, ...] = field(default_factory=tuple)
    structure_reconstruction_confidence: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "original_text": self.original_text,
            "normalized_text": self.normalized_text,
            "canonical_text": self.canonical_text,
            "route_ready_text": self.route_ready_text,
            "detected_language": self.detected_language,
            "normalized_topic": self.normalized_topic,
            "dominant_intent": self.dominant_intent,
            "intent_confidence": self.intent_confidence,
            "extracted_entities": [dict(entity) for entity in self.extracted_entities],
            "structure_subject": self.structure_subject,
            "structure_predicate": self.structure_predicate,
            "structure_object": self.structure_object,
            "structure_modifiers": list(self.structure_modifiers),
            "structure_completeness": self.structure_completeness,
            "structure_ambiguity_flags": list(self.structure_ambiguity_flags),
            "structure_fragmentation_markers": list(self.structure_fragmentation_markers),
            "structure_reconstruction_confidence": self.structure_reconstruction_confidence,
        }


@dataclass(slots=True)
class PromptUnderstanding:
    original_text: str
    normalized_text: str
    intent_ready_text: str
    surface_views: PromptSurfaceViews
    structure: PromptStructureParse
    language: DetectedLanguage
    topic: NormalizedTopic
    intent: IntentParse
    entities: tuple[ExtractedEntity, ...] = field(default_factory=tuple)
    profile_advice: ProfileAdvice | None = None

    @property
    def canonical_text(self) -> str:
        reconstructed = str(self.surface_views.reconstructed_text or "").strip()
        if reconstructed:
            return reconstructed
        return str(self.intent_ready_text or self.normalized_text).strip()

    def router_view(self) -> RouterInputView:
        return RouterInputView(
            original_text=self.original_text,
            normalized_text=self.normalized_text,
            canonical_text=self.canonical_text,
            route_ready_text=self.surface_views.route_ready_text,
            detected_language=self.language.code,
            normalized_topic=self.topic.value,
            dominant_intent=self.intent.label,
            intent_confidence=self.intent.confidence,
            extracted_entities=tuple(entity.to_dict() for entity in self.entities),
            structure_subject=self.structure.subject,
            structure_predicate=self.structure.predicate,
            structure_object=self.structure.object_target,
            structure_modifiers=self.structure.modifiers,
            structure_completeness=self.structure.completeness,
            structure_ambiguity_flags=self.structure.ambiguity_flags,
            structure_fragmentation_markers=self.structure.fragmentation_markers,
            structure_reconstruction_confidence=self.structure.reconstruction_confidence,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "original_text": self.original_text,
            "normalized_text": self.normalized_text,
            "canonical_text": self.canonical_text,
            "intent_ready_text": self.intent_ready_text,
            "surface_views": self.surface_views.to_dict(),
            "structure": self.structure.to_dict(),
            "language": self.language.to_dict(),
            "topic": self.topic.to_dict(),
            "intent": self.intent.to_dict(),
            "entities": [entity.to_dict() for entity in self.entities],
            "profile_advice": self.profile_advice.to_dict() if self.profile_advice else None,
            "router_view": self.router_view().to_dict(),
        }


CanonicalPromptUnderstanding = PromptUnderstanding
StructureInterpretation = PromptStructureParse
