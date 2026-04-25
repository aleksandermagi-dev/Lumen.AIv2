from __future__ import annotations

from dataclasses import dataclass

from lumen.knowledge.models import KnowledgeEntry


@dataclass(frozen=True, slots=True)
class GlossaryConcept:
    entry_id: str
    title: str
    category: str
    aliases: tuple[str, ...]
    definition: str
    related_topics: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GlossaryAliasResolution:
    candidates: tuple[str, ...]
    default: str
    canonical_title: str
    domain_bias: str | None = None
    confidence_boost: str | None = None


class KnowledgeGlossary:
    def __init__(
        self,
        *,
        alias_to_resolution: dict[str, GlossaryAliasResolution],
        title_to_concept: dict[str, GlossaryConcept],
    ) -> None:
        self._alias_to_resolution = alias_to_resolution
        self._title_to_concept = title_to_concept

    @classmethod
    def from_entries(cls, entries: tuple[KnowledgeEntry, ...]) -> "KnowledgeGlossary":
        alias_to_resolution: dict[str, GlossaryAliasResolution] = {}
        title_to_concept: dict[str, GlossaryConcept] = {}
        for entry in entries:
            concept = GlossaryConcept(
                entry_id=entry.id,
                title=entry.title,
                category=entry.category,
                aliases=tuple(dict.fromkeys((entry.title, *entry.aliases))),
                definition=entry.summary_short or entry.summary_medium or entry.title,
                related_topics=entry.related_topics,
            )
            title_to_concept[cls._normalize(entry.title)] = concept
            for alias in concept.aliases:
                normalized = cls._normalize(alias)
                if normalized and normalized not in alias_to_resolution:
                    alias_to_resolution[normalized] = GlossaryAliasResolution(
                        candidates=(entry.id,),
                        default=entry.id,
                        canonical_title=entry.title,
                    )
            if entry.id == "astronomy.great_attractor":
                alias_to_resolution["ga"] = GlossaryAliasResolution(
                    candidates=("astronomy.great_attractor",),
                    default="astronomy.great_attractor",
                    canonical_title=entry.title,
                    domain_bias="astronomy",
                    confidence_boost="short_token",
                )
        return cls(alias_to_resolution=alias_to_resolution, title_to_concept=title_to_concept)

    def candidate_subjects(self, subject: str) -> list[str]:
        normalized_subject = self._normalize(subject)
        if not normalized_subject:
            return []
        candidates: list[str] = [normalized_subject]
        resolution = self.resolve_subject(subject)
        if resolution is not None:
            canonical_normalized = self._normalize(resolution.canonical_title)
            if canonical_normalized not in candidates:
                candidates.append(canonical_normalized)
            concept = self._title_to_concept.get(canonical_normalized)
            if concept is not None:
                for alias in concept.aliases:
                    normalized_alias = self._normalize(alias)
                    if normalized_alias and normalized_alias not in candidates:
                        candidates.append(normalized_alias)
        return candidates

    def resolve_subject(self, subject: str) -> GlossaryAliasResolution | None:
        normalized_subject = self._normalize(subject)
        if not normalized_subject:
            return None
        direct = self._alias_to_resolution.get(normalized_subject)
        if direct is not None:
            return direct
        tokens = normalized_subject.split()
        token_set = set(tokens)
        for alias, resolution in self._alias_to_resolution.items():
            if resolution.confidence_boost != "short_token":
                continue
            if alias not in token_set:
                continue
            if resolution.domain_bias and resolution.domain_bias not in token_set:
                continue
            return resolution
        return None

    @staticmethod
    def _normalize(text: str) -> str:
        normalized = str(text or "").strip().lower()
        for dash in ("-", "\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2212"):
            normalized = normalized.replace(dash, " ")
        return " ".join(normalized.split())
