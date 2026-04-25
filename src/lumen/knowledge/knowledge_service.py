from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from lumen.knowledge.knowledge_db import KnowledgeDB
from lumen.knowledge.glossary import GlossaryAliasResolution, KnowledgeGlossary
from lumen.knowledge.models import KnowledgeEntry, KnowledgeLookupResult, KnowledgeMatch
from lumen.nlu.focus_resolution import FocusResolutionSupport
from lumen.nlu.prompt_surface_builder import PromptSurfaceBuilder
from lumen.knowledge.seed_data import SEED_ENTRIES, SEED_RELATIONSHIPS


@dataclass(slots=True)
class KnowledgeService:
    db: KnowledgeDB
    last_lookup_diagnostics: dict[str, object] = field(default_factory=dict, init=False)
    glossary: KnowledgeGlossary | None = field(default=None, init=False)

    @classmethod
    def from_path(cls, db_path: Path | str) -> "KnowledgeService":
        service = cls(db=KnowledgeDB(db_path))
        service.initialize()
        return service

    @classmethod
    def in_memory(cls) -> "KnowledgeService":
        return cls.from_path(":memory:")

    def initialize(self) -> None:
        self.db.initialize()
        # Re-apply curated seed data on startup so existing user databases pick up
        # newly shipped local knowledge entries and alias improvements.
        self.db.seed(entries=SEED_ENTRIES, relationships=SEED_RELATIONSHIPS)
        self.glossary = KnowledgeGlossary.from_entries(SEED_ENTRIES)

    def lookup(self, query: str) -> KnowledgeLookupResult | None:
        normalized = self.db.normalize(PromptSurfaceBuilder.build(query).lookup_ready_text)
        if not normalized:
            self.last_lookup_diagnostics = {"reason": "no_subject_resolution", "query": query}
            return None
        comparison_parts = self._comparison_parts(normalized)
        if comparison_parts is not None:
            left = self._best_match(comparison_parts[0])
            right = self._best_match(comparison_parts[1])
            combined_subject = self._combined_comparison_subject(normalized)
            combined_match = self._best_match(combined_subject) if combined_subject else None
            if left is None or right is None:
                if combined_match is not None:
                    threshold = 0.68 if self._is_common_subject(combined_subject, combined_match.entry) else 0.72
                    if combined_match.score >= threshold:
                        self.last_lookup_diagnostics = {
                            "reason": "entry_match",
                            "query": query,
                            "subject": combined_subject,
                            "matched_alias": combined_match.matched_alias,
                            "score": combined_match.score,
                        }
                        return KnowledgeLookupResult(
                            mode="entry",
                            primary=combined_match.entry,
                            score=combined_match.score,
                            matched_alias=combined_match.matched_alias,
                        )
                self.last_lookup_diagnostics = {
                    "reason": "candidate_conflict",
                    "query": query,
                    "comparison_parts": list(comparison_parts),
                }
                return None
            left_threshold = 0.68 if self._is_common_subject(comparison_parts[0], left.entry) else 0.75
            right_threshold = 0.68 if self._is_common_subject(comparison_parts[1], right.entry) else 0.75
            if left.score < left_threshold or right.score < right_threshold:
                if combined_match is not None:
                    combined_threshold = 0.68 if self._is_common_subject(combined_subject, combined_match.entry) else 0.72
                    if combined_match.score >= combined_threshold:
                        self.last_lookup_diagnostics = {
                            "reason": "entry_match",
                            "query": query,
                            "subject": combined_subject,
                            "matched_alias": combined_match.matched_alias,
                            "score": combined_match.score,
                        }
                        return KnowledgeLookupResult(
                            mode="entry",
                            primary=combined_match.entry,
                            score=combined_match.score,
                            matched_alias=combined_match.matched_alias,
                        )
                self.last_lookup_diagnostics = {
                    "reason": "insufficient_dominance",
                    "query": query,
                    "comparison_parts": list(comparison_parts),
                    "scores": {"left": left.score, "right": right.score},
                }
                return None
            relation = self._relation_between(left.entry.id, right.entry.id)
            self.last_lookup_diagnostics = {
                "reason": "comparison_match",
                "query": query,
                "comparison_parts": list(comparison_parts),
                "relation": relation,
            }
            return KnowledgeLookupResult(
                mode="comparison",
                primary=left.entry,
                secondary=right.entry,
                score=min(left.score, right.score),
                comparison_relation=relation,
            )

        subject = self._subject_key(normalized)
        match = self._best_match(subject)
        if match is None:
            self.last_lookup_diagnostics = {
                "reason": FocusResolutionSupport.subject_focus(query).reason(),
                "query": query,
                "subject": subject,
            }
            return None
        threshold = 0.68 if self._is_common_subject(subject, match.entry) else 0.72
        short_alias_resolution = self._short_token_resolution(subject)
        if match.score < threshold and not self._should_accept_short_alias_match(match, short_alias_resolution):
            self.last_lookup_diagnostics = {
                "reason": "insufficient_dominance",
                "query": query,
                "subject": subject,
                "score": match.score,
                "matched_alias": match.matched_alias,
            }
            return None
        self.last_lookup_diagnostics = {
            "reason": "entry_match",
            "query": query,
            "subject": subject,
            "matched_alias": match.matched_alias,
            "score": match.score,
        }
        return KnowledgeLookupResult(
            mode="entry",
            primary=match.entry,
            score=match.score,
            matched_alias=match.matched_alias,
        )

    def partial_lookup(self, query: str) -> KnowledgeLookupResult | None:
        normalized = self.db.normalize(PromptSurfaceBuilder.build(query).lookup_ready_text)
        if not normalized:
            self.last_lookup_diagnostics = {"reason": "no_subject_resolution", "query": query}
            return None
        subject = self._subject_key(normalized)
        relaxed_subject = subject.replace("'s ", " ").replace("’s ", " ")
        match = self._best_match(relaxed_subject)
        if match is None:
            self.last_lookup_diagnostics = {
                "reason": "weak_focus_overlap",
                "query": query,
                "subject": relaxed_subject,
            }
            return None
        matched_alias = self.db.normalize(match.matched_alias or match.entry.title)
        normalized_subject = self.db.normalize(relaxed_subject)
        if match.score < 0.65 or match.score >= 0.92:
            self.last_lookup_diagnostics = {
                "reason": "insufficient_dominance",
                "query": query,
                "subject": relaxed_subject,
                "score": match.score,
            }
            return None
        if matched_alias not in normalized_subject and self.db.normalize(match.entry.title) not in normalized_subject:
            self.last_lookup_diagnostics = {
                "reason": "candidate_conflict",
                "query": query,
                "subject": relaxed_subject,
                "matched_alias": matched_alias,
            }
            return None
        self.last_lookup_diagnostics = {
            "reason": "partial_match",
            "query": query,
            "subject": relaxed_subject,
            "matched_alias": matched_alias,
            "score": match.score,
        }
        return KnowledgeLookupResult(
            mode="entry",
            primary=match.entry,
            score=match.score,
            matched_alias=match.matched_alias,
            partial=True,
        )

    def related_connections(self, entry_id: str) -> list[tuple[str, str]]:
        with self.db.connect() as connection:
            rows = connection.execute(
                """
                SELECT r.relation_type, e.title
                FROM knowledge_relationships r
                JOIN knowledge_entries e ON e.id = r.target_entry_id
                WHERE r.source_entry_id = ?
                ORDER BY
                    CASE r.relation_type
                        WHEN 'part_of' THEN 0
                        WHEN 'related_to' THEN 1
                        WHEN 'causes' THEN 2
                        WHEN 'opposite_of' THEN 3
                        WHEN 'compares_with' THEN 4
                        ELSE 5
                    END,
                    e.title
                """,
                (entry_id,),
            ).fetchall()
        return [(str(row["relation_type"]), str(row["title"])) for row in rows]

    def overview(self) -> dict[str, object]:
        with self.db.connect() as connection:
            rows = connection.execute(
                """
                SELECT category, title
                FROM knowledge_entries
                ORDER BY category, title
                """
            ).fetchall()
        grouped: dict[str, list[str]] = {}
        for row in rows:
            category = str(row["category"] or "uncategorized").strip() or "uncategorized"
            grouped.setdefault(category, []).append(str(row["title"]))
        categories = [
            {
                "category": category,
                "entry_count": len(titles),
                "titles": list(titles),
            }
            for category, titles in grouped.items()
        ]
        return {
            "entry_count": sum(item["entry_count"] for item in categories),
            "category_count": len(categories),
            "categories": categories,
        }

    def _best_match(self, normalized_subject: str) -> KnowledgeMatch | None:
        subject_variants = self._subject_candidates(normalized_subject)
        if not subject_variants:
            return None
        with self.db.connect() as connection:
            short_alias_match = self._glossary_short_alias_match(connection, normalized_subject)
            if short_alias_match is not None:
                return short_alias_match
            for subject in subject_variants:
                alias_row = connection.execute(
                    """
                    SELECT e.*, a.alias
                    FROM knowledge_aliases a
                    JOIN knowledge_entries e ON e.id = a.entry_id
                    WHERE a.normalized_alias = ?
                    LIMIT 1
                    """,
                    (subject,),
                ).fetchone()
                if alias_row is not None:
                    return KnowledgeMatch(
                        entry=self._load_entry(connection, str(alias_row["id"])),
                        score=1.0,
                        matched_alias=str(alias_row["alias"]),
                    )

                title_row = connection.execute(
                    "SELECT * FROM knowledge_entries WHERE lower(title) = ? LIMIT 1",
                    (subject,),
                ).fetchone()
                if title_row is not None:
                    return KnowledgeMatch(
                        entry=self._load_entry(connection, str(title_row["id"])),
                        score=0.96,
                        matched_alias=str(title_row["title"]),
                    )

            rows = connection.execute("SELECT id, title FROM knowledge_entries").fetchall()

        best: KnowledgeMatch | None = None
        for row in rows:
            with self.db.connect() as connection:
                entry = self._load_entry(connection, str(row["id"]))
            normalized_names = [
                self.db.normalize(entry.title),
                *[self.db.normalize(alias) for alias in entry.aliases],
            ]
            best_score = 0.0
            for subject in subject_variants:
                for candidate_name in normalized_names:
                    if not candidate_name:
                        continue
                    candidate_tokens = set(candidate_name.split())
                    subject_tokens = set(subject.split())
                    overlap = len(subject_tokens & candidate_tokens)
                    if not overlap:
                        continue
                    score = overlap / max(len(subject_tokens), len(candidate_tokens))
                    coverage = overlap / max(1, len(subject_tokens))
                    if subject in candidate_name or candidate_name in subject:
                        score += 0.2
                    if coverage >= 0.75:
                        score += 0.08
                    if len(subject_tokens) == len(candidate_tokens) and coverage == 1.0:
                        score += 0.06
                    best_score = max(best_score, score)
            if best_score < 0.6:
                continue
            candidate = KnowledgeMatch(entry=entry, score=min(best_score, 0.89), matched_alias=entry.title)
            if best is None or candidate.score > best.score:
                best = candidate
        return best

    def _glossary_short_alias_match(self, connection, normalized_subject: str) -> KnowledgeMatch | None:
        resolution = self._short_token_resolution(normalized_subject)
        if resolution is None:
            return None
        try:
            entry = self._load_entry(connection, resolution.default)
        except Exception:
            return None
        return KnowledgeMatch(
            entry=entry,
            score=0.99,
            matched_alias=normalized_subject,
        )

    def _load_entry(self, connection, entry_id: str) -> KnowledgeEntry:
        row = connection.execute(
            "SELECT * FROM knowledge_entries WHERE id = ?",
            (entry_id,),
        ).fetchone()
        formula_row = connection.execute(
            "SELECT * FROM knowledge_formulas WHERE entry_id = ?",
            (entry_id,),
        ).fetchone()
        return self.db.entry_from_row(row, formula_row)

    def _relation_between(self, left_id: str, right_id: str) -> str | None:
        with self.db.connect() as connection:
            row = connection.execute(
                """
                SELECT relation_type FROM knowledge_relationships
                WHERE source_entry_id = ? AND target_entry_id = ?
                LIMIT 1
                """,
                (left_id, right_id),
            ).fetchone()
        return str(row["relation_type"]) if row is not None else None

    @classmethod
    def _comparison_parts(cls, normalized_query: str) -> tuple[str, str] | None:
        if " vs " in normalized_query:
            left, right = normalized_query.split(" vs ", maxsplit=1)
            return cls._subject_key(left), cls._subject_key(right)
        if " versus " in normalized_query:
            left, right = normalized_query.split(" versus ", maxsplit=1)
            return cls._subject_key(left), cls._subject_key(right)
        if " in relation to " in normalized_query:
            left, right = normalized_query.split(" in relation to ", maxsplit=1)
            return cls._subject_key(left), cls._subject_key(right)
        if " related to " in normalized_query:
            left, right = normalized_query.split(" related to ", maxsplit=1)
            return cls._subject_key(left), cls._subject_key(right)
        if normalized_query.startswith("compare "):
            body = normalized_query[len("compare ") :].strip()
            if " and " in body:
                left, right = body.split(" and ", maxsplit=1)
                return cls._subject_key(left), cls._subject_key(right)
        return None

    @staticmethod
    def _combined_comparison_subject(normalized_query: str) -> str:
        combined = normalized_query.replace(" versus ", " and ").replace(" vs ", " and ").strip()
        return FocusResolutionSupport.subject_focus(combined).focus

    @staticmethod
    def _subject_key(normalized_prompt: str) -> str:
        return FocusResolutionSupport.subject_focus(normalized_prompt).focus

    def _subject_candidates(self, normalized_subject: str) -> list[str]:
        subject = self.db.normalize(normalized_subject)
        if not subject:
            return []
        variants: list[str] = []
        descriptor_tokens = {
            "concept",
            "concepts",
            "event",
            "events",
            "formula",
            "formulas",
            "object",
            "objects",
            "person",
            "people",
            "place",
            "places",
            "planet",
            "planets",
            "process",
            "processes",
            "system",
            "systems",
            "topic",
            "topics",
        }

        def add(candidate: str) -> None:
            normalized_candidate = self.db.normalize(candidate)
            if normalized_candidate and normalized_candidate not in variants:
                variants.append(normalized_candidate)

        if self.glossary is not None:
            for candidate in self.glossary.candidate_subjects(subject):
                add(candidate)
        add(subject)
        cleaned_subject = self._trim_subject_noise(subject)
        if cleaned_subject != subject:
            if self.glossary is not None:
                for candidate in self.glossary.candidate_subjects(cleaned_subject):
                    add(candidate)
            add(cleaned_subject)
        if subject.startswith("the "):
            add(subject[4:])
        if "(" in subject and ")" in subject:
            add(subject.split("(", maxsplit=1)[0].strip())
        tokens = subject.split()
        if len(tokens) >= 2:
            add(" ".join(token for token in tokens if token not in {"the", "a", "an"}))
            singularized_tokens = [self._singularize_token(token) for token in tokens]
            add(" ".join(singularized_tokens))
            trimmed_tokens = [token for token in tokens if token not in descriptor_tokens]
            if trimmed_tokens and trimmed_tokens != tokens:
                add(" ".join(trimmed_tokens))
                add(" ".join(self._singularize_token(token) for token in trimmed_tokens))
            if len(tokens) == 2:
                add(" ".join(reversed(tokens)))
                add(" ".join(reversed(singularized_tokens)))
                if trimmed_tokens and len(trimmed_tokens) == 2:
                    add(" ".join(reversed(trimmed_tokens)))
        else:
            add(self._singularize_token(subject))
        return variants

    def _short_token_resolution(self, subject: str) -> GlossaryAliasResolution | None:
        if self.glossary is None:
            return None
        resolution = self.glossary.resolve_subject(subject)
        if resolution is None or resolution.confidence_boost != "short_token":
            return None
        normalized = self.db.normalize(subject)
        tokens = normalized.split()
        short_tokens = [token for token in tokens if len(token) <= 3]
        if not short_tokens:
            return None
        if normalized in short_tokens:
            return resolution
        if any(token in tokens for token in short_tokens):
            return resolution
        return None

    @staticmethod
    def _should_accept_short_alias_match(
        match: KnowledgeMatch,
        resolution: GlossaryAliasResolution | None,
    ) -> bool:
        if resolution is None:
            return False
        return match.entry.id in set(resolution.candidates)

    @staticmethod
    def _singularize_token(token: str) -> str:
        if len(token) <= 3:
            return token
        if token.endswith("ies") and len(token) > 4:
            return token[:-3] + "y"
        if token.endswith("es") and len(token) > 4 and not token.endswith(("ses", "xes")):
            return token[:-2]
        if token.endswith("s") and not token.endswith("ss"):
            return token[:-1]
        return token

    @staticmethod
    def _trim_subject_noise(subject: str) -> str:
        cleaned = " ".join(str(subject or "").strip().lower().split())
        if not cleaned:
            return ""
        leading_phrases = (
            "causes of the ",
            "causes of ",
            "cause of the ",
            "cause of ",
            "let's explore ",
            "lets explore ",
            "explore ",
            "explain to me ",
            "explain ",
            "break down ",
            "break it down ",
            "to me what a ",
            "to me what an ",
            "to me what ",
            "what a ",
            "what an ",
            "what ",
        )
        trailing_phrases = (
            " using an analogy",
            " with an analogy",
            " with examples",
            " with example",
            " in simple terms",
            " step by step",
            " break it down",
            " in depth",
            " in detail",
            " more deeply",
            " deep",
            " deeply",
            " briefly",
            " clearly",
            " simply but correctly",
            " simply correctly",
            " but correctly",
            " correctly",
            " instead",
            " simply",
            " is",
            " are",
        )
        changed = True
        while changed and cleaned:
            changed = False
            for prefix in leading_phrases:
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix) :].strip()
                    changed = True
            for suffix in trailing_phrases:
                if cleaned.endswith(suffix):
                    cleaned = cleaned[: -len(suffix)].strip()
                    changed = True
        return cleaned

    @staticmethod
    def _is_common_subject(subject: str, entry: KnowledgeEntry) -> bool:
        normalized = " ".join(str(subject or "").strip().lower().split())
        if not normalized:
            return False
        tokens = normalized.split()
        if len(tokens) > 2:
            return False
        if entry.entry_type not in {"concept", "formula", "system", "object"}:
            return False
        common_tokens = {
            "power",
            "watt",
            "watts",
            "voltage",
            "current",
            "resistance",
            "ohm",
            "ohms",
            "formula",
            "quadratic",
            "saturn",
            "black",
            "hole",
        }
        return any(token in common_tokens for token in tokens)
