from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class SafetyOutcomeAssessment:
    tier: str
    outcome_risk: str
    rationale: str
    category: str = "allowed"
    boundary: str = ""
    safe_redirects: list[str] = field(default_factory=list)
    matched_signals: list[str] = field(default_factory=list)
    response_constraint: dict[str, object] = field(default_factory=dict)


class SafetyOutcomeEvaluator:
    """Evaluates whether a requested output would materially lower the barrier to harm."""

    _PUBLIC_DISCUSSION_SIGNALS = (
        "declassified",
        "historical",
        "history",
        "ethics",
        "public record",
        "critical analysis",
        "defensive",
        "prevention",
        "harm prevention",
        "risk analysis",
    )
    _HIGH_RISK_PATTERNS = (
        "disable a car",
        "disable brakes",
        "track someone without",
        "follow someone without",
        "surveil someone without",
        "without them knowing",
        "bypass a lock",
        "break into",
        "disable an alarm",
        "make a jammer",
        "signal jammer",
        "knock someone out",
        "restrain someone",
        "poison someone",
        "sabotage a vehicle",
        "evade law enforcement",
        "hide evidence",
    )
    _MEDIUM_RISK_TOPICS = (
        "gps tracker",
        "tracking device",
        "surveillance drone",
        "covert camera",
        "counter-surveillance",
        "lockpick",
        "jammer",
        "signal jammer",
        "disable detection",
        "evade detection",
    )
    _OPERATIONAL_VERBS = (
        "build",
        "make",
        "create",
        "design",
        "optimize",
        "improve",
        "deploy",
        "use",
        "install",
        "set up",
        "configure",
        "track",
        "monitor",
        "follow",
        "surveil",
        "disable",
        "bypass",
        "evade",
    )

    def evaluate(self, prompt: str) -> SafetyOutcomeAssessment:
        normalized = " ".join(str(prompt).lower().split())

        if self._contains_any(normalized, self._PUBLIC_DISCUSSION_SIGNALS) and not self._contains_any(
            normalized, self._OPERATIONAL_VERBS
        ):
            return SafetyOutcomeAssessment(
                tier="safe",
                outcome_risk="low",
                rationale="Public or ethical discussion signals are present without an operational request.",
            )

        high_signals = self._matched_signals(normalized, self._HIGH_RISK_PATTERNS)
        if high_signals:
            return SafetyOutcomeAssessment(
                tier="restricted",
                outcome_risk="high",
                category="outcome_barrier_reduction",
                rationale="The request would materially lower the barrier to real-world harm or covert targeting.",
                boundary="I can't help with instructions that would make harming, coercing, or covertly targeting someone easier.",
                safe_redirects=[
                    "I can help with safety, ethics, prevention, or defensive analysis instead.",
                ],
                matched_signals=high_signals,
            )

        medium_topic_signals = self._matched_signals(normalized, self._MEDIUM_RISK_TOPICS)
        if medium_topic_signals and (
            self._contains_any(normalized, self._OPERATIONAL_VERBS)
            or normalized.startswith(("how do", "how does", "explain how", "what is the best way to"))
        ):
            return SafetyOutcomeAssessment(
                tier="dual_use",
                outcome_risk="medium",
                rationale="The request has dual-use potential and should stay high-level and non-operational.",
                matched_signals=medium_topic_signals,
                response_constraint={
                    "level": "high_level_only",
                    "allow_steps": False,
                    "allow_optimization": False,
                    "allow_sourcing": False,
                    "allow_tactical_next_steps": False,
                },
            )

        return SafetyOutcomeAssessment(
            tier="safe",
            outcome_risk="low",
            rationale="No meaningful barrier-reduction outcome signal was detected.",
        )

    @staticmethod
    def _contains_any(text: str, patterns: tuple[str, ...]) -> bool:
        return any(pattern in text for pattern in patterns)

    @staticmethod
    def _matched_signals(text: str, patterns: tuple[str, ...]) -> list[str]:
        return [pattern for pattern in patterns if pattern in text]
