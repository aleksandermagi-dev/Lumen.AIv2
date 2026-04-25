from __future__ import annotations

from dataclasses import dataclass

from lumen.knowledge.models import KnowledgeEntry


@dataclass(frozen=True, slots=True)
class SeedHygieneIssue:
    entry_id: str
    alias: str
    reason: str


class SeedHygiene:
    _BROAD_ALIAS_TOKENS = {
        "ai",
        "ml",
        "os",
        "cpu",
        "ram",
        "system",
        "engine",
        "physics",
        "history",
    }

    @classmethod
    def lint_entries(cls, entries: tuple[KnowledgeEntry, ...]) -> list[SeedHygieneIssue]:
        issues: list[SeedHygieneIssue] = []
        seen_aliases: dict[str, str] = {}
        for entry in entries:
            for alias in entry.aliases:
                normalized = " ".join(str(alias or "").strip().lower().split())
                if not normalized:
                    issues.append(
                        SeedHygieneIssue(entry_id=entry.id, alias=str(alias), reason="empty_alias")
                    )
                    continue
                if normalized in seen_aliases and seen_aliases[normalized] != entry.id:
                    issues.append(
                        SeedHygieneIssue(
                            entry_id=entry.id,
                            alias=alias,
                            reason=f"duplicate_alias:{seen_aliases[normalized]}",
                        )
                    )
                seen_aliases.setdefault(normalized, entry.id)
                if normalized in cls._BROAD_ALIAS_TOKENS or len(normalized) <= 2:
                    issues.append(
                        SeedHygieneIssue(entry_id=entry.id, alias=alias, reason="overly_broad_alias")
                    )
        return issues
