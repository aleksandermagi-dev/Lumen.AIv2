from __future__ import annotations

from dataclasses import dataclass, field


VALID_EVALUATION_JUDGMENTS = {
    "correct",
    "weak",
    "incorrect",
    "insufficient_evidence",
}


@dataclass(slots=True, frozen=True)
class EvaluationSurfaceReview:
    surface: str
    judgment: str
    score: float | None
    rationale: str
    evidence: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.judgment not in VALID_EVALUATION_JUDGMENTS:
            raise ValueError(f"Unsupported evaluation judgment: {self.judgment}")

    def to_dict(self) -> dict[str, object]:
        return {
            "surface": self.surface,
            "judgment": self.judgment,
            "score": round(float(self.score), 4) if self.score is not None else None,
            "rationale": self.rationale,
            "evidence": dict(self.evidence),
        }


@dataclass(slots=True, frozen=True)
class InteractionDecisionEvaluation:
    session_id: str | None
    interaction_path: str | None
    created_at: str | None
    mode: str | None
    kind: str | None
    summary: str | None
    overall_judgment: str
    surface_reviews: tuple[EvaluationSurfaceReview, ...]
    judgment_counts: dict[str, int] = field(default_factory=dict)
    schema_version: str = "1"

    def __post_init__(self) -> None:
        if self.overall_judgment not in VALID_EVALUATION_JUDGMENTS:
            raise ValueError(f"Unsupported overall evaluation judgment: {self.overall_judgment}")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "interaction_path": self.interaction_path,
            "created_at": self.created_at,
            "mode": self.mode,
            "kind": self.kind,
            "summary": self.summary,
            "overall_judgment": self.overall_judgment,
            "judgment_counts": dict(self.judgment_counts),
            "surface_reviews": [review.to_dict() for review in self.surface_reviews],
        }


@dataclass(slots=True, frozen=True)
class EvaluationAggregate:
    surface: str
    reviewed_count: int
    judgment_counts: dict[str, int]
    average_score: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "surface": self.surface,
            "reviewed_count": self.reviewed_count,
            "judgment_counts": dict(self.judgment_counts),
            "average_score": round(float(self.average_score), 4) if self.average_score is not None else None,
        }


@dataclass(slots=True, frozen=True)
class DecisionEvaluationBatch:
    session_id: str | None
    evaluated_count: int
    judgment_counts: dict[str, int]
    surface_aggregates: tuple[EvaluationAggregate, ...]
    evaluations: tuple[InteractionDecisionEvaluation, ...]
    behavior_slices: tuple[dict[str, object], ...] = ()
    schema_version: str = "1"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "evaluated_count": self.evaluated_count,
            "judgment_counts": dict(self.judgment_counts),
            "surface_aggregates": [aggregate.to_dict() for aggregate in self.surface_aggregates],
            "evaluations": [evaluation.to_dict() for evaluation in self.evaluations],
            "behavior_slices": [dict(item) for item in self.behavior_slices],
        }
