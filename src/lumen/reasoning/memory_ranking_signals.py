from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from lumen.memory.indexing import IndexedMemoryRecord


class MemoryRankingSignals:
    """Shared scoring helpers for durable-memory retrieval ranking."""

    _CONTRADICTION_CUES = ("contradiction", "conflict", "inconsistency", "disagree", "mismatch")
    _GENERIC_LABEL_MARKERS = ("summary", "notes", "note", "project", "idea", "finding", "response")

    @classmethod
    def score_indexed_memory(
        cls,
        *,
        indexed: IndexedMemoryRecord,
        recall_focus: str,
        recall_prompt: bool,
        project_return_prompt: bool,
        normalized_prompt: str,
    ) -> tuple[float, dict[str, object]]:
        focus_overlap = cls.focus_overlap(
            recall_focus,
            indexed.summary,
            indexed.label,
            *indexed.domain_tags,
        )
        age_bucket, decay_factor = cls.age_profile(
            created_at=indexed.created_at,
            source=indexed.source,
        )
        reaffirmed = focus_overlap >= 2
        source_reliability = cls.source_reliability(
            source=indexed.source,
            metadata=indexed.metadata,
        )
        contradiction_penalty = cls.contradiction_penalty(
            indexed=indexed,
            normalized_prompt=normalized_prompt,
        )
        generic_penalty = cls.generic_label_penalty(indexed.label)
        relevance = float(indexed.relevance_hint) * decay_factor
        if recall_prompt:
            relevance += 0.1
        if project_return_prompt and indexed.memory_kind == "durable_project_memory":
            relevance += 0.1
        relevance += focus_overlap * 0.08
        relevance += source_reliability * 0.06
        if reaffirmed:
            relevance += 0.08
        relevance -= contradiction_penalty
        relevance -= generic_penalty
        relevance = max(0.0, min(0.97, relevance))
        metadata = indexed.to_metadata()
        metadata["age_bucket"] = age_bucket
        metadata["decay_factor"] = round(decay_factor, 4)
        metadata["reaffirmed"] = reaffirmed
        metadata["source_reliability"] = round(source_reliability, 4)
        metadata["contradiction_penalty"] = round(contradiction_penalty, 4)
        metadata["generic_label_penalty"] = round(generic_penalty, 4)
        metadata["focus_overlap"] = focus_overlap
        return relevance, metadata

    @classmethod
    def score_retrieved_memory(
        cls,
        *,
        source: str,
        label: str,
        summary: str,
        base_relevance: float,
        metadata: dict[str, Any] | None,
        recall_focus: str,
        recall_prompt: bool,
        project_return_prompt: bool,
        normalized_prompt: str,
    ) -> tuple[float, dict[str, object]]:
        payload = dict(metadata or {})
        focus_overlap = cls.focus_overlap(
            recall_focus,
            summary,
            label,
            *list(payload.get("domain_tags") or []),
        )
        age_bucket, decay_factor = cls.age_profile(
            created_at=str(payload.get("created_at") or "").strip() or None,
            source=source,
        )
        reaffirmed = bool(payload.get("reaffirmed")) or focus_overlap >= 2
        source_reliability = cls.source_reliability(source=source, metadata=payload)
        contradiction_penalty = cls.contradiction_penalty_from_fields(
            label=label,
            summary=summary,
            metadata=payload,
            normalized_prompt=normalized_prompt,
        )
        generic_penalty = cls.generic_label_penalty(label)
        stale_penalty = cls.staleness_penalty(age_bucket=age_bucket, reaffirmed=reaffirmed)
        relevance = float(base_relevance) * decay_factor
        if recall_prompt:
            relevance += 0.08
        if project_return_prompt and source in {"archive", "research_notes", "active_thread"}:
            relevance += 0.1
        relevance += focus_overlap * 0.06
        relevance += source_reliability * 0.04
        if reaffirmed:
            relevance += 0.06
        relevance -= contradiction_penalty
        relevance -= generic_penalty
        relevance -= stale_penalty
        relevance = max(0.0, min(0.97, relevance))
        payload["age_bucket"] = age_bucket
        payload["decay_factor"] = round(decay_factor, 4)
        payload["reaffirmed"] = reaffirmed
        payload["source_reliability"] = round(source_reliability, 4)
        payload["contradiction_penalty"] = round(contradiction_penalty, 4)
        payload["generic_label_penalty"] = round(generic_penalty, 4)
        payload["staleness_penalty"] = round(stale_penalty, 4)
        payload["focus_overlap"] = focus_overlap
        return relevance, payload

    @staticmethod
    def focus_overlap(focus: str, *haystacks: str) -> int:
        tokens = [token for token in str(focus or "").split() if len(token) > 2]
        if not tokens:
            return 0
        score = 0
        lowered = [str(item or "").lower() for item in haystacks]
        for token in tokens:
            if any(token in haystack for haystack in lowered):
                score += 1
        return score

    @staticmethod
    def age_profile(*, created_at: str | None, source: str) -> tuple[str, float]:
        normalized_source = str(source or "").strip().lower()
        if normalized_source == "active_thread":
            return ("active", 1.0)
        if not created_at:
            return ("unknown", 0.88)
        try:
            created_dt = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
        except ValueError:
            return ("unknown", 0.88)
        if created_dt.tzinfo is None:
            created_dt = created_dt.replace(tzinfo=UTC)
        age_days = max((datetime.now(UTC) - created_dt.astimezone(UTC)).days, 0)
        if normalized_source == "personal_memory":
            if age_days >= 365:
                return ("old", 0.7)
            if age_days >= 120:
                return ("stale", 0.82)
            if age_days >= 30:
                return ("aging", 0.92)
            return ("recent", 1.0)
        if normalized_source in {"interaction", "interaction_history"}:
            if age_days >= 365:
                return ("old", 0.55)
            if age_days >= 120:
                return ("stale", 0.75)
            if age_days >= 30:
                return ("aging", 0.9)
            return ("recent", 1.0)
        if normalized_source == "archive":
            if age_days >= 180:
                return ("old", 0.45)
            if age_days >= 60:
                return ("stale", 0.65)
            if age_days >= 14:
                return ("aging", 0.85)
            return ("recent", 1.0)
        if age_days >= 365:
            return ("old", 0.58)
        if age_days >= 120:
            return ("stale", 0.72)
        if age_days >= 30:
            return ("aging", 0.88)
        return ("recent", 1.0)

    @staticmethod
    def source_reliability(*, source: str, metadata: dict[str, Any] | None = None) -> float:
        payload = dict(metadata or {})
        explicit = payload.get("source_reliability")
        if isinstance(explicit, (int, float)):
            return float(explicit)
        if source == "personal_memory":
            return 0.92
        if source == "research_notes":
            return 0.84
        if source == "active_thread":
            return 0.96
        if source == "interaction_history":
            return 0.72
        if source == "archive":
            return 0.68
        return 0.7

    @classmethod
    def contradiction_penalty(cls, *, indexed: IndexedMemoryRecord, normalized_prompt: str) -> float:
        return cls.contradiction_penalty_from_fields(
            label=indexed.label,
            summary=indexed.summary,
            metadata=indexed.metadata,
            normalized_prompt=normalized_prompt,
        )

    @classmethod
    def contradiction_penalty_from_fields(
        cls,
        *,
        label: str,
        summary: str,
        metadata: dict[str, Any] | None,
        normalized_prompt: str,
    ) -> float:
        prompt = str(normalized_prompt or "").lower()
        if any(cue in prompt for cue in cls._CONTRADICTION_CUES):
            return 0.0
        payload = dict(metadata or {})
        contradiction_status = str(payload.get("contradiction_status") or "").strip().lower()
        if contradiction_status == "potential_conflict":
            return 0.12
        text = " ".join((label, summary)).lower()
        if any(cue in text for cue in cls._CONTRADICTION_CUES):
            return 0.08
        return 0.0

    @classmethod
    def generic_label_penalty(cls, label: str) -> float:
        cleaned = " ".join(str(label or "").strip().lower().split())
        if not cleaned:
            return 0.0
        if any(marker in cleaned for marker in cls._GENERIC_LABEL_MARKERS):
            return 0.08
        if len(cleaned.split()) >= 8:
            return 0.05
        return 0.0

    @staticmethod
    def staleness_penalty(*, age_bucket: str, reaffirmed: bool) -> float:
        if reaffirmed:
            return 0.0
        if age_bucket == "old":
            return 0.12
        if age_bucket == "stale":
            return 0.08
        if age_bucket == "aging":
            return 0.03
        return 0.0
