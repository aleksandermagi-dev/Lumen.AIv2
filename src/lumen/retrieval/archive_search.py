from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ArchiveSearchMatch:
    score: int
    matched_fields: list[str] = field(default_factory=list)
    score_breakdown: dict[str, int] = field(default_factory=dict)
    record: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ArchiveSearchResult:
    query: str
    record_count: int
    matches: list[ArchiveSearchMatch] = field(default_factory=list)
