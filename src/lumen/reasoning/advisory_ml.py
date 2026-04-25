from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True, frozen=True)
class AdvisoryMLRequest:
    surface: str
    session_id: str | None
    project_id: str | None
    message_id: str | None
    context: dict[str, object] = field(default_factory=dict)
    features: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "surface": self.surface,
            "session_id": self.session_id,
            "project_id": self.project_id,
            "message_id": self.message_id,
            "context": dict(self.context),
            "features": dict(self.features),
        }


@dataclass(slots=True, frozen=True)
class AdvisoryMLRecommendation:
    surface: str
    recommended_label: str | None
    confidence: float | None
    provenance: dict[str, object] = field(default_factory=dict)
    enabled: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "surface": self.surface,
            "recommended_label": self.recommended_label,
            "confidence": self.confidence,
            "provenance": dict(self.provenance),
            "enabled": self.enabled,
        }


class AdvisoryMLAdapter:
    """Advisory-only interface for future offline-trained recommenders."""

    def recommend(self, request: AdvisoryMLRequest) -> AdvisoryMLRecommendation:
        return AdvisoryMLRecommendation(
            surface=request.surface,
            recommended_label=None,
            confidence=None,
            provenance={
                "adapter": "noop",
                "authority": "advisory_only",
                "deterministic_authority_preserved": True,
            },
            enabled=False,
        )


def build_advisory_ml_request(
    *,
    surface: str,
    session_id: str | None,
    project_id: str | None,
    message_id: str | None,
    context: dict[str, Any] | None = None,
    features: dict[str, Any] | None = None,
) -> AdvisoryMLRequest:
    return AdvisoryMLRequest(
        surface=surface,
        session_id=session_id,
        project_id=project_id,
        message_id=message_id,
        context=dict(context or {}),
        features=dict(features or {}),
    )
