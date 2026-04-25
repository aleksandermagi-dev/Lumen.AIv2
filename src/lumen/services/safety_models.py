from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class PromptSafetyDecision:
    action: str
    category: str
    severity: str
    rationale: str
    boundary: str
    tier: str = "safe"
    outcome_risk: str = "low"
    response_constraint: dict[str, object] = field(default_factory=dict)
    tool_constraint: dict[str, object] = field(default_factory=dict)
    safe_redirects: list[str] = field(default_factory=list)
    matched_signals: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "category": self.category,
            "severity": self.severity,
            "rationale": self.rationale,
            "boundary": self.boundary,
            "tier": self.tier,
            "outcome_risk": self.outcome_risk,
            "response_constraint": dict(self.response_constraint),
            "tool_constraint": dict(self.tool_constraint),
            "safe_redirects": list(self.safe_redirects),
            "matched_signals": list(self.matched_signals),
        }
