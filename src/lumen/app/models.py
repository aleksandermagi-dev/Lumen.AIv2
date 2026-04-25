from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class InteractionProfile:
    interaction_style: str = "conversational"
    reasoning_depth: str = "normal"
    selection_source: str = "user"
    confidence: float | None = None
    allow_suggestions: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "interaction_style": self.normalize_interaction_style(self.interaction_style),
            "reasoning_depth": self.reasoning_depth,
            "selection_source": self.selection_source,
            "confidence": self.confidence,
            "allow_suggestions": self.allow_suggestions,
        }

    @classmethod
    def default(cls) -> "InteractionProfile":
        return cls(interaction_style="collab")

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "InteractionProfile":
        if not isinstance(payload, dict):
            return cls.default()
        return cls(
            interaction_style=cls.normalize_interaction_style(
                str(payload.get("interaction_style") or "collab")
            ),
            reasoning_depth=str(payload.get("reasoning_depth") or "normal"),
            selection_source=str(payload.get("selection_source") or "user"),
            confidence=(
                float(payload["confidence"])
                if payload.get("confidence") is not None
                else None
            ),
            allow_suggestions=bool(payload.get("allow_suggestions", True)),
        )

    @staticmethod
    def normalize_interaction_style(style: str) -> str:
        normalized = str(style or "collab").strip().lower()
        if normalized == "conversational":
            return "collab"
        if normalized in {"default", "collab", "direct"}:
            return normalized
        return "collab"


@dataclass(slots=True)
class SessionState:
    session_id: str
    repo_root: Path
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(slots=True)
class ActiveThreadState:
    session_id: str
    mode: str
    kind: str
    prompt: str
    objective: str
    thread_summary: str
    summary: str
    confidence_posture: str | None = None
    route_status: str | None = None
    support_status: str | None = None
    tension_status: str | None = None
    tool_route_origin: str | None = None
    local_context_assessment: str | None = None
    coherence_topic: str | None = None
    interaction_profile: dict[str, Any] = field(default_factory=dict)
    pipeline_observability: dict[str, Any] = field(default_factory=dict)
    pipeline_trace: dict[str, Any] = field(default_factory=dict)
    original_prompt: str | None = None
    detected_language: str | None = None
    normalized_topic: str | None = None
    dominant_intent: str | None = None
    intent_domain: str | None = None
    intent_domain_confidence: float | None = None
    response_depth: str | None = None
    conversation_phase: str | None = None
    next_step_state: dict[str, Any] = field(default_factory=dict)
    tool_suggestion_state: dict[str, Any] = field(default_factory=dict)
    trainability_trace: dict[str, Any] = field(default_factory=dict)
    supervised_support_trace: dict[str, Any] = field(default_factory=dict)
    extracted_entities: tuple[dict[str, object], ...] = field(default_factory=tuple)
    tool_context: dict[str, Any] = field(default_factory=dict)
    continuation_offer: dict[str, Any] = field(default_factory=dict)
    reasoning_state: dict[str, Any] = field(default_factory=dict)
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(slots=True)
class ArchivedRunRecord:
    session_id: str
    tool_id: str
    capability: str
    status: str
    summary: str
    run_dir: Path | None
    archive_path: Path
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
