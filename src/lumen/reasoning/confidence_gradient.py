from __future__ import annotations

from dataclasses import dataclass, field


CONFIDENCE_TIERS = ("low", "medium", "high")


@dataclass(slots=True, frozen=True)
class ConfidenceAssessment:
    score: float
    tier: str
    rationale: str | None = None
    contributors: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return {
            "score": round(float(self.score), 4),
            "tier": self.tier,
            "rationale": self.rationale,
            "contributors": list(self.contributors),
        }


class ConfidenceGradient:
    """Map existing numeric confidence into stable cognitive confidence bands."""

    @staticmethod
    def normalize_score(score: float | int | None) -> float:
        if score is None:
            return 0.0
        try:
            normalized = float(score)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, normalized))

    @staticmethod
    def tier_for_score(score: float | int | None) -> str:
        normalized = ConfidenceGradient.normalize_score(score)
        if normalized >= 0.78:
            return "high"
        if normalized >= 0.45:
            return "medium"
        return "low"

    @classmethod
    def from_score(
        cls,
        score: float | int | None,
        *,
        rationale: str | None = None,
        contributors: list[str] | tuple[str, ...] | None = None,
    ) -> ConfidenceAssessment:
        normalized = cls.normalize_score(score)
        return ConfidenceAssessment(
            score=normalized,
            tier=cls.tier_for_score(normalized),
            rationale=rationale,
            contributors=tuple(str(item).strip() for item in (contributors or ()) if str(item).strip()),
        )

    @classmethod
    def from_route(
        cls,
        *,
        score: float | int | None,
        weak_route: bool = False,
        rationale: str | None = None,
    ) -> ConfidenceAssessment:
        normalized = cls.normalize_score(score)
        if weak_route:
            normalized = max(0.0, normalized - 0.15)
        contributors = ["route_decision"]
        if weak_route:
            contributors.append("weak_route_penalty")
        return cls.from_score(normalized, rationale=rationale, contributors=contributors)

    @classmethod
    def with_memory(
        cls,
        base_score: float | int | None,
        *,
        memory_signal: float,
        rationale: str | None = None,
    ) -> ConfidenceAssessment:
        normalized = cls.normalize_score(base_score)
        adjusted = max(0.0, min(1.0, normalized + (memory_signal - 0.5) * 0.20))
        return cls.from_score(
            adjusted,
            rationale=rationale,
            contributors=["memory_context", "relevance_filter"],
        )

    @classmethod
    def with_tool_outcome(
        cls,
        base_score: float | int | None,
        *,
        expected_confidence_gain: float = 0.0,
        verified: bool = False,
        failed: bool = False,
        rationale: str | None = None,
    ) -> ConfidenceAssessment:
        normalized = cls.normalize_score(base_score)
        adjusted = normalized
        contributors = ["tool_threshold_gate"]
        if verified:
            adjusted = min(1.0, adjusted + max(0.12, expected_confidence_gain))
            contributors.append("tool_verified")
        elif failed:
            adjusted = max(0.0, adjusted - 0.18)
            contributors.append("tool_failed")
        return cls.from_score(adjusted, rationale=rationale, contributors=contributors)
