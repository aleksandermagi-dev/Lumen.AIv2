from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class CompactPromptView:
    canonical_prompt: str | None
    original_prompt: str | None
    resolved_prompt: str | None
    rewritten: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class CompactArchiveContextRecord:
    session_id: str | None
    tool_id: str | None
    capability: str | None
    status: str | None
    run_id: str | None
    target_label: str | None
    result_quality: str | None
    summary: str | None
    created_at: str | None
    archive_path: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class CompactInteractionContextRecord:
    session_id: str | None
    prompt: str | None
    resolved_prompt: str | None
    mode: str | None
    kind: str | None
    summary: str | None
    created_at: str | None
    interaction_path: str | None
    resolution_strategy: str | None
    resolution_reason: str | None
    dominant_intent: str | None
    extracted_entities: tuple[str, ...]
    prompt_view: CompactPromptView

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["prompt_view"] = self.prompt_view.to_dict()
        payload["extracted_entities"] = list(self.extracted_entities)
        return payload


@dataclass(slots=True)
class CompactContextMatch:
    score: int | None
    record: CompactArchiveContextRecord | CompactInteractionContextRecord
    matched_fields: tuple[str, ...] = ()
    score_breakdown: dict[str, int] | None = None

    def to_dict(self) -> dict[str, object]:
        payload = {
            "score": self.score,
            "record": self.record.to_dict(),
            "matched_fields": list(self.matched_fields),
        }
        if self.score_breakdown:
            payload["score_breakdown"] = dict(self.score_breakdown)
        return payload
