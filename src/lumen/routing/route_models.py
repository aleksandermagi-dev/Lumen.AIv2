from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class RouteEvidence:
    label: str
    detail: str
    weight: float

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "detail": self.detail,
            "weight": self.weight,
        }


@dataclass(slots=True)
class RouteCandidate:
    mode: str
    kind: str
    confidence: float
    reason: str
    source: str
    evidence: list[RouteEvidence] = field(default_factory=list)

    def to_summary(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "kind": self.kind,
            "confidence": self.confidence,
            "reason": self.reason,
            "source": self.source,
            "evidence": [item.to_dict() for item in self.evidence],
        }


@dataclass(slots=True)
class RouteComparison:
    candidate: RouteCandidate
    source_priority: int
    mode_priority: int
    semantic_bonus: float = 0.0
    intent_weight: float = 0.0
    context_decay: float = 0.0
    normalized_score: float = 0.0

    @property
    def rank_tuple(self) -> tuple[float, int, float, float, float, int]:
        return (
            self.normalized_score,
            self.source_priority,
            self.intent_weight,
            self.semantic_bonus,
            self.candidate.confidence,
            self.mode_priority,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate": self.candidate.to_summary(),
            "source_priority": self.source_priority,
            "mode_priority": self.mode_priority,
            "semantic_bonus": self.semantic_bonus,
            "intent_weight": self.intent_weight,
            "context_decay": self.context_decay,
            "normalized_score": self.normalized_score,
            "rank_tuple": list(self.rank_tuple),
        }


@dataclass(slots=True)
class RouteDecisionSummary:
    selected: RouteComparison
    alternatives: list[RouteComparison] = field(default_factory=list)
    ambiguous: bool = False
    ambiguity_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload = {
            "selected": self.selected.to_dict(),
            "alternatives": [item.to_dict() for item in self.alternatives],
        }
        if self.ambiguous:
            payload["ambiguous"] = True
        if self.ambiguity_reason:
            payload["ambiguity_reason"] = self.ambiguity_reason
        return payload


@dataclass(slots=True)
class RouteAnalysis:
    normalized_prompt: str
    signals: dict[str, object]
    candidates: list[RouteCandidate]
    comparisons: list[RouteComparison]
    decision_summary: RouteDecisionSummary

    def selected_route(self) -> RouteCandidate:
        return self.decision_summary.selected.candidate

    def to_dict(self) -> dict[str, object]:
        return {
            "normalized_prompt": self.normalized_prompt,
            "signals": dict(self.signals),
            "candidates": [candidate.to_summary() for candidate in self.candidates],
            "comparisons": [comparison.to_dict() for comparison in self.comparisons],
            "decision_summary": self.decision_summary.to_dict(),
        }
