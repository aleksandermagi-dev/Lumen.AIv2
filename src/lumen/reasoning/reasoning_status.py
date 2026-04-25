from __future__ import annotations

from lumen.reasoning.pipeline_models import ReasoningStatusSnapshot, TensionResolutionResult


class ReasoningStatusPolicy:
    """Owns qualitative reasoning-status interpretation for the pipeline."""

    def build_snapshot(
        self,
        *,
        route_strength: str,
        route_quality: str,
        grounding_strength: str,
        local_context_assessment: str | None,
        route_ambiguity: bool,
        contradiction_flags: list[str],
        evidence_strength: str | None,
        failure_modes: dict[str, bool],
        tension_resolution: TensionResolutionResult,
    ) -> ReasoningStatusSnapshot:
        confidence_posture = self.confidence_posture(
            route_strength=route_strength,
            route_quality=route_quality,
            grounding_strength=grounding_strength,
            local_context_assessment=local_context_assessment,
            route_ambiguity=route_ambiguity,
        )
        uncertainty_posture = self.uncertainty_posture(
            confidence_posture=confidence_posture,
            local_context_assessment=local_context_assessment,
            contradiction_flags=contradiction_flags,
        )
        tension_status = self.tension_status(tension_resolution=tension_resolution)
        snapshot = ReasoningStatusSnapshot(
            route_quality=route_quality,
            grounding_strength=grounding_strength,
            local_context_assessment=local_context_assessment,
            confidence_posture=confidence_posture,
            uncertainty_posture=uncertainty_posture,
            route_status=self.route_status(
                route_quality=route_quality,
                route_ambiguity=route_ambiguity,
                tension_status=tension_status,
            ),
            support_status=self.support_status(
                grounding_strength=grounding_strength,
                evidence_strength=evidence_strength,
            ),
            tension_status=tension_status,
            failure_modes=dict(failure_modes),
        )
        return self.normalize_snapshot(snapshot)

    @staticmethod
    def normalize_snapshot(snapshot: ReasoningStatusSnapshot) -> ReasoningStatusSnapshot:
        failure_modes = dict(snapshot.failure_modes)
        contradiction_present = bool(
            failure_modes.get("high_ambiguity")
            or failure_modes.get("weak_context")
            or failure_modes.get("weak_evidence")
        )

        if snapshot.tension_status in {"under_tension", "unresolved", "revised"}:
            if snapshot.route_status not in {"under_tension", "unresolved", "revised"}:
                snapshot.route_status = snapshot.tension_status

        if snapshot.support_status == "strongly_supported" and failure_modes.get("weak_context"):
            snapshot.support_status = "insufficiently_grounded"
        elif snapshot.support_status == "strongly_supported" and failure_modes.get("weak_evidence"):
            snapshot.support_status = "moderately_supported"

        if snapshot.tension_status == "stable" and contradiction_present:
            snapshot.tension_status = "under_tension"
            if snapshot.route_status == "stable":
                snapshot.route_status = "under_tension"

        if snapshot.tension_status == "revised" and not contradiction_present:
            snapshot.tension_status = "under_tension"
            if snapshot.route_status == "revised":
                snapshot.route_status = "under_tension"

        return snapshot

    @staticmethod
    def confidence_posture(
        *,
        route_strength: str,
        route_quality: str,
        grounding_strength: str,
        local_context_assessment: str | None,
        route_ambiguity: bool,
    ) -> str:
        if (
            route_strength == "high"
            and grounding_strength == "high"
            and local_context_assessment == "aligned"
            and not route_ambiguity
        ):
            return "strong"
        if local_context_assessment == "mixed":
            return "conflicted"
        if route_strength == "low" or grounding_strength == "low" or route_quality == "weak":
            return "tentative"
        if route_ambiguity:
            if grounding_strength == "high":
                return "supported"
            return "tentative"
        if grounding_strength == "high" or route_strength == "high":
            return "supported"
        return "tentative"

    @staticmethod
    def uncertainty_posture(
        *,
        confidence_posture: str,
        local_context_assessment: str | None,
        contradiction_flags: list[str],
    ) -> str:
        if contradiction_flags or local_context_assessment == "mixed":
            return "conflicted"
        if confidence_posture == "tentative":
            return "tentative"
        if confidence_posture == "supported":
            return "guarded"
        return "stable"

    @staticmethod
    def support_status(*, grounding_strength: str, evidence_strength: str | None) -> str:
        if grounding_strength == "low" or evidence_strength == "missing":
            return "insufficiently_grounded"
        if grounding_strength == "high" or evidence_strength == "strong":
            return "strongly_supported"
        return "moderately_supported"

    @staticmethod
    def tension_status(*, tension_resolution: TensionResolutionResult) -> str:
        status = str(tension_resolution.status or "").strip()
        return status or "stable"

    @staticmethod
    def route_status(
        *,
        route_quality: str,
        route_ambiguity: bool,
        tension_status: str,
    ) -> str:
        if tension_status == "revised":
            return "revised"
        if tension_status == "unresolved":
            return "unresolved"
        if tension_status == "under_tension":
            return "under_tension"
        if route_ambiguity:
            return "under_tension"
        if route_quality == "weak":
            return "weakened"
        return "stable"
