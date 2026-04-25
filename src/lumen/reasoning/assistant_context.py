from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class AssistantContext:
    record_count: int = 0
    status_counts: dict[str, int] = field(default_factory=dict)
    tool_counts: dict[str, int] = field(default_factory=dict)
    active_thread: dict[str, Any] | None = None
    route: dict[str, Any] | None = None
    query: str | None = None
    top_matches: list[dict[str, Any]] = field(default_factory=list)
    matched_record_count: int = 0
    interaction_record_count: int = 0
    top_interaction_matches: list[dict[str, Any]] = field(default_factory=list)
    interaction_query: str | None = None
    interaction_query_source: str | None = None
    archive_target_comparison: dict[str, Any] | None = None

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "AssistantContext":
        return cls(
            record_count=int(payload.get("record_count", 0) or 0),
            status_counts=dict(payload.get("status_counts") or {}),
            tool_counts=dict(payload.get("tool_counts") or {}),
            active_thread=dict(payload.get("active_thread")) if isinstance(payload.get("active_thread"), dict) else payload.get("active_thread"),
            route=dict(payload.get("route")) if isinstance(payload.get("route"), dict) else payload.get("route"),
            query=str(payload.get("query")) if payload.get("query") is not None else None,
            top_matches=list(payload.get("top_matches") or []),
            matched_record_count=int(payload.get("matched_record_count", 0) or 0),
            interaction_record_count=int(payload.get("interaction_record_count", 0) or 0),
            top_interaction_matches=list(payload.get("top_interaction_matches") or []),
            interaction_query=str(payload.get("interaction_query")) if payload.get("interaction_query") is not None else None,
            interaction_query_source=str(payload.get("interaction_query_source")) if payload.get("interaction_query_source") is not None else None,
            archive_target_comparison=(
                dict(payload.get("archive_target_comparison"))
                if isinstance(payload.get("archive_target_comparison"), dict)
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
