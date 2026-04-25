from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True, frozen=True)
class IndexedMemoryRecord:
    source: str
    storage_layer: str
    source_category: str
    memory_kind: str
    label: str
    summary: str
    session_id: str | None = None
    source_path: str | None = None
    created_at: str | None = None
    domain_tags: tuple[str, ...] = ()
    confidence_hint: float = 0.5
    relevance_hint: float = 0.5
    metadata: dict[str, object] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, object]:
        payload = {
            "storage_layer": self.storage_layer,
            "source_category": self.source_category,
            "domain_tags": list(self.domain_tags),
            "confidence_hint": round(float(self.confidence_hint), 4),
            "relevance_hint": round(float(self.relevance_hint), 4),
            "session_id": self.session_id,
            "source_path": self.source_path,
            "created_at": self.created_at,
        }
        payload.update(dict(self.metadata))
        return payload


class MemoryIndexBuilder:
    """Normalize raw persisted memory records into ranking-friendly indexed views."""

    _CONTRADICTION_MARKERS = ("contradiction", "conflict", "inconsistency", "disagree", "mismatch")

    @staticmethod
    def _extract_personal_memory_summary(entry: dict[str, Any]) -> str:
        source_prompt = str(entry.get("source_interaction_prompt") or "").strip()
        if ":" in source_prompt:
            prefix, remainder = source_prompt.split(":", 1)
            normalized_prefix = " ".join(prefix.lower().split())
            if normalized_prefix in {"remember this about me", "remember this", "save this"}:
                cleaned = remainder.strip()
                if cleaned:
                    return cleaned
        source_summary = str(entry.get("source_interaction_summary") or "").strip()
        if source_summary:
            return source_summary
        content = str(entry.get("content") or "").strip()
        if content:
            return content
        return str(entry.get("title") or "").strip()

    @staticmethod
    def from_personal_entry(entry: dict[str, Any]) -> IndexedMemoryRecord | None:
        title = str(entry.get("title") or "").strip()
        summary = MemoryIndexBuilder._extract_personal_memory_summary(entry)
        if not title and not summary:
            return None
        normalized_topic = str(entry.get("normalized_topic") or "").strip()
        return IndexedMemoryRecord(
            source="personal_memory",
            storage_layer="indexed_layer",
            source_category="personal_memory",
            memory_kind="durable_user_memory",
            label=title or normalized_topic or "personal memory",
            summary=summary,
            session_id=str(entry.get("session_id") or "").strip() or None,
            source_path=str(entry.get("entry_path") or "").strip() or None,
            created_at=str(entry.get("created_at") or "").strip() or None,
            domain_tags=tuple(
                token
                for token in (
                    "personal",
                    "preference",
                    normalized_topic,
                    str(((entry.get("memory_classification") or {}).get("candidate_type")) or "").strip(),
                )
                if token
            ),
            confidence_hint=0.82,
            relevance_hint=0.72,
            metadata={
                "client_surface": str(entry.get("client_surface") or "").strip() or None,
                "memory_origin": str(entry.get("memory_origin") or "").strip() or None,
                "source_reliability": 0.92,
                "contradiction_status": MemoryIndexBuilder._contradiction_status(
                    title=title,
                    summary=summary,
                ),
            },
        )

    @staticmethod
    def from_research_note(note: dict[str, Any]) -> IndexedMemoryRecord | None:
        title = str(note.get("title") or "").strip()
        content = str(note.get("content") or "").strip()
        if not title and not content:
            return None
        normalized_topic = str(note.get("normalized_topic") or "").strip()
        dominant_intent = str(note.get("dominant_intent") or "").strip()
        return IndexedMemoryRecord(
            source="research_notes",
            storage_layer="indexed_layer",
            source_category="research_notes",
            memory_kind="durable_project_memory",
            label=title or normalized_topic or "research note",
            summary=content or title,
            session_id=str(note.get("session_id") or "").strip() or None,
            source_path=str(note.get("note_path") or "").strip() or None,
            created_at=str(note.get("created_at") or "").strip() or None,
            domain_tags=tuple(
                token
                for token in (
                    "research",
                    dominant_intent,
                    normalized_topic,
                    str(note.get("note_type") or "").strip(),
                )
                if token
            ),
            confidence_hint=0.78,
            relevance_hint=0.76,
            metadata={
                "client_surface": str(note.get("client_surface") or "").strip() or None,
                "source_interaction_mode": str(note.get("source_interaction_mode") or "").strip() or None,
                "source_interaction_kind": str(note.get("source_interaction_kind") or "").strip() or None,
                "source_reliability": 0.84,
                "contradiction_status": MemoryIndexBuilder._contradiction_status(
                    title=title,
                    summary=content or title,
                ),
            },
        )

    @classmethod
    def _contradiction_status(cls, *, title: str, summary: str) -> str:
        text = " ".join((title, summary)).lower()
        if any(marker in text for marker in cls._CONTRADICTION_MARKERS):
            return "potential_conflict"
        return "clear"
