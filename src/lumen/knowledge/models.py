from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class KnowledgeFormula:
    formula_text: str
    variable_meanings: dict[str, str] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)
    interpretation: str | None = None
    example_usage: str | None = None


@dataclass(slots=True)
class KnowledgeEntry:
    id: str
    title: str
    entry_type: str
    category: str
    subcategory: str | None = None
    aliases: tuple[str, ...] = ()
    summary_short: str = ""
    summary_medium: str = ""
    summary_deep: str | None = None
    key_points: tuple[str, ...] = ()
    common_questions: tuple[str, ...] = ()
    related_topics: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    difficulty_level: str = "general"
    examples: tuple[str, ...] = ()
    source_type: str = "curated_local"
    confidence: float = 0.8
    last_updated: str = ""
    formula: KnowledgeFormula | None = None


@dataclass(slots=True)
class KnowledgeRelationship:
    source_entry_id: str
    relation_type: str
    target_entry_id: str


@dataclass(slots=True)
class KnowledgeMatch:
    entry: KnowledgeEntry
    score: float
    matched_alias: str | None = None


@dataclass(slots=True)
class KnowledgeLookupResult:
    mode: str
    primary: KnowledgeEntry | None = None
    secondary: KnowledgeEntry | None = None
    score: float = 0.0
    matched_alias: str | None = None
    comparison_relation: str | None = None
    partial: bool = False
