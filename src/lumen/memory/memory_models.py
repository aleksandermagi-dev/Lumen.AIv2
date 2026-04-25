from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class MemoryClassification:
    candidate_type: str
    classification_confidence: float
    save_eligible: bool
    requires_explicit_user_consent: bool
    explicit_save_requested: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_type": self.candidate_type,
            "classification_confidence": round(float(self.classification_confidence), 4),
            "save_eligible": self.save_eligible,
            "requires_explicit_user_consent": self.requires_explicit_user_consent,
            "explicit_save_requested": self.explicit_save_requested,
            "reason": self.reason,
        }

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "MemoryClassification":
        if not isinstance(payload, dict):
            return cls.ephemeral(
                reason="No memory classification was recorded, so the interaction remains unsaved by default."
            )
        return cls(
            candidate_type=str(payload.get("candidate_type") or "ephemeral_conversation_context"),
            classification_confidence=float(payload.get("classification_confidence") or 0.0),
            save_eligible=bool(payload.get("save_eligible")),
            requires_explicit_user_consent=bool(payload.get("requires_explicit_user_consent")),
            explicit_save_requested=bool(payload.get("explicit_save_requested")),
            reason=str(payload.get("reason") or "").strip(),
        )

    @classmethod
    def research_candidate(cls, *, confidence: float, reason: str, save_eligible: bool = True) -> "MemoryClassification":
        return cls(
            candidate_type="research_memory_candidate",
            classification_confidence=confidence,
            save_eligible=save_eligible,
            requires_explicit_user_consent=False,
            explicit_save_requested=False,
            reason=reason,
        )

    @classmethod
    def personal_candidate(
        cls,
        *,
        confidence: float,
        reason: str,
        explicit_save_requested: bool = False,
    ) -> "MemoryClassification":
        return cls(
            candidate_type="personal_context_candidate",
            classification_confidence=confidence,
            save_eligible=False,
            requires_explicit_user_consent=True,
            explicit_save_requested=explicit_save_requested,
            reason=reason,
        )

    @classmethod
    def ephemeral(cls, *, reason: str, confidence: float = 0.0) -> "MemoryClassification":
        return cls(
            candidate_type="ephemeral_conversation_context",
            classification_confidence=confidence,
            save_eligible=False,
            requires_explicit_user_consent=False,
            explicit_save_requested=False,
            reason=reason,
        )


@dataclass(slots=True)
class MemoryWriteDecision:
    action: str
    save_research_note: bool
    save_personal_memory: bool
    blocked_by_surface_policy: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "save_research_note": self.save_research_note,
            "save_personal_memory": self.save_personal_memory,
            "blocked_by_surface_policy": self.blocked_by_surface_policy,
            "reason": self.reason,
        }

    @classmethod
    def skip(cls, *, reason: str, blocked_by_surface_policy: bool = False) -> "MemoryWriteDecision":
        return cls(
            action="skip",
            save_research_note=False,
            save_personal_memory=False,
            blocked_by_surface_policy=blocked_by_surface_policy,
            reason=reason,
        )

    @classmethod
    def research_note(cls, *, reason: str) -> "MemoryWriteDecision":
        return cls(
            action="save_research_note",
            save_research_note=True,
            save_personal_memory=False,
            blocked_by_surface_policy=False,
            reason=reason,
        )

    @classmethod
    def personal_memory(cls, *, reason: str) -> "MemoryWriteDecision":
        return cls(
            action="save_personal_memory",
            save_research_note=False,
            save_personal_memory=True,
            blocked_by_surface_policy=False,
            reason=reason,
        )
