from __future__ import annotations

from lumen.app.models import InteractionProfile
from lumen.reasoning.interaction_style_policy import InteractionStylePolicy
from lumen.reasoning.pipeline_models import EvidenceUnit, TensionResolutionResult


class TensionResolver:
    """Classifies and resolves tension without destabilizing anchor selection."""

    def resolve(
        self,
        *,
        interaction_profile: InteractionProfile,
        evidence_ledger: list[EvidenceUnit],
        anchor_evidence_id: str | None,
        tension_evidence_ids: list[str],
        contradiction_flags: list[str],
        failure_modes: dict[str, bool],
    ) -> TensionResolutionResult:
        if not tension_evidence_ids and not contradiction_flags:
            return TensionResolutionResult(
                tension_detected=False,
                category=None,
                resolution_path=None,
                rationale=None,
                status="stable",
                anchor_status="stable",
            )

        category = self._category(
            contradiction_flags=contradiction_flags,
            failure_modes=failure_modes,
        )
        anchor_unit = self._find_unit(evidence_ledger, anchor_evidence_id)
        tension_units = [
            unit
            for unit in (
                self._find_unit(evidence_ledger, evidence_id)
                for evidence_id in tension_evidence_ids
            )
            if unit is not None
        ]
        resolution_path = self._resolution_path(
            category=category,
            anchor_unit=anchor_unit,
            tension_units=tension_units,
        )
        alternate_hypotheses = []
        leading_hypothesis_label = None
        if resolution_path == "alternate_hypothesis":
            alternate_hypotheses = self._alternate_hypotheses(
                anchor_unit=anchor_unit,
                tension_units=tension_units,
            )
            leading_hypothesis_label = self._leading_hypothesis_label(alternate_hypotheses)
        rationale = self._rationale(
            category=category,
            resolution_path=resolution_path,
            anchor_unit=anchor_unit,
            tension_units=tension_units,
            interaction_profile=interaction_profile,
            alternate_hypotheses=alternate_hypotheses,
        )
        return TensionResolutionResult(
            tension_detected=True,
            category=category,
            resolution_path=resolution_path,
            rationale=rationale,
            status=self._status_for_resolution_path(resolution_path),
            anchor_status=self._anchor_status(
                resolution_path=resolution_path,
                anchor_unit=anchor_unit,
                tension_units=tension_units,
            ),
            recommended_action=self._recommended_action(
                category=category,
                resolution_path=resolution_path,
                leading_hypothesis_label=leading_hypothesis_label,
            ),
            leading_hypothesis_label=leading_hypothesis_label,
            anchor_evidence_id=anchor_evidence_id,
            tension_evidence_ids=list(tension_evidence_ids),
            alternate_hypotheses=alternate_hypotheses,
        )

    @staticmethod
    def _find_unit(evidence_ledger: list[EvidenceUnit], evidence_id: str | None) -> EvidenceUnit | None:
        if not evidence_id:
            return None
        for unit in evidence_ledger:
            if unit.evidence_id == evidence_id:
                return unit
        return None

    @staticmethod
    def _category(*, contradiction_flags: list[str], failure_modes: dict[str, bool]) -> str:
        if "ambiguous_route" in contradiction_flags:
            return "clarification_tension"
        if failure_modes.get("weak_context"):
            return "missing_information_tension"
        return "evidence_tension"

    @staticmethod
    def _resolution_path(
        *,
        category: str,
        anchor_unit: EvidenceUnit | None,
        tension_units: list[EvidenceUnit],
    ) -> str:
        if category in {"clarification_tension", "missing_information_tension"}:
            return "clarification"
        anchor_score = float(anchor_unit.authority_score if anchor_unit else 0.0)
        tension_score = max((float(unit.authority_score) for unit in tension_units), default=0.0)
        if tension_score > anchor_score + 0.75:
            return "hypothesis_revision"
        return "alternate_hypothesis"

    @staticmethod
    def _alternate_hypotheses(
        *,
        anchor_unit: EvidenceUnit | None,
        tension_units: list[EvidenceUnit],
    ) -> list[dict[str, object]]:
        hypotheses: list[dict[str, object]] = []
        if anchor_unit is not None:
            hypotheses.append(
                {
                    "label": "A",
                    "anchor_evidence_id": anchor_unit.evidence_id,
                    "source": anchor_unit.source,
                    "authority_score": anchor_unit.authority_score,
                    "summary": anchor_unit.summary,
                    "status": "current_anchor",
                }
            )
        for index, unit in enumerate(tension_units, start=1):
            hypotheses.append(
                {
                    "label": chr(ord("A") + index),
                    "anchor_evidence_id": unit.evidence_id,
                    "source": unit.source,
                    "authority_score": unit.authority_score,
                    "summary": unit.summary,
                    "status": "competing",
                }
            )
        ranked = sorted(hypotheses, key=lambda item: float(item.get("authority_score") or 0.0), reverse=True)[:3]
        if ranked:
            top_score = float(ranked[0].get("authority_score") or 0.0)
            for item in ranked:
                score = float(item.get("authority_score") or 0.0)
                if score >= top_score - 0.35:
                    item["status"] = "leading"
                elif item.get("status") != "current_anchor":
                    item["status"] = "competing"
        return ranked

    @staticmethod
    def _leading_hypothesis_label(alternate_hypotheses: list[dict[str, object]]) -> str | None:
        for hypothesis in alternate_hypotheses:
            if str(hypothesis.get("status") or "") == "leading":
                return str(hypothesis.get("label") or "").strip() or None
        return None

    @staticmethod
    def _status_for_resolution_path(resolution_path: str) -> str:
        if resolution_path == "hypothesis_revision":
            return "revised"
        if resolution_path == "clarification":
            return "unresolved"
        if resolution_path == "alternate_hypothesis":
            return "under_tension"
        return "stable"

    @staticmethod
    def _anchor_status(
        *,
        resolution_path: str,
        anchor_unit: EvidenceUnit | None,
        tension_units: list[EvidenceUnit],
    ) -> str:
        if resolution_path == "hypothesis_revision":
            return "replaced"
        if resolution_path == "clarification":
            return "uncertain"
        if resolution_path == "alternate_hypothesis":
            anchor_score = float(anchor_unit.authority_score if anchor_unit else 0.0)
            tension_score = max((float(unit.authority_score) for unit in tension_units), default=0.0)
            if tension_score >= anchor_score:
                return "weakened"
            return "stable"
        return "stable"

    @staticmethod
    def _recommended_action(
        *,
        category: str,
        resolution_path: str,
        leading_hypothesis_label: str | None,
    ) -> str:
        if resolution_path == "clarification":
            return "clarify_conflict"
        if resolution_path == "hypothesis_revision":
            return "revise_hypothesis"
        if resolution_path == "alternate_hypothesis":
            if category == "missing_information_tension":
                return "gather_missing_evidence"
            if leading_hypothesis_label:
                return f"compare_hypotheses_{leading_hypothesis_label.lower()}"
            return "compare_hypotheses"
        return "none"

    @staticmethod
    def _rationale(
        *,
        category: str,
        resolution_path: str,
        anchor_unit: EvidenceUnit | None,
        tension_units: list[EvidenceUnit],
        interaction_profile: InteractionProfile,
        alternate_hypotheses: list[dict[str, object]],
    ) -> str:
        anchor_summary = anchor_unit.summary if anchor_unit else "current anchor"
        tension_summary = tension_units[0].summary if tension_units else "new evidence"
        leading_hypothesis = next(
            (item for item in alternate_hypotheses if str(item.get("status") or "") == "leading"),
            None,
        )
        if InteractionStylePolicy.is_direct(interaction_profile):
            if resolution_path == "clarification":
                return f"Conflict detected: {anchor_summary} contradicts {tension_summary}. Clarification required."
            if resolution_path == "hypothesis_revision":
                return f"Conflict detected: {tension_summary} is stronger than {anchor_summary}. Re-evaluation required."
            if leading_hypothesis:
                return (
                    f"Conflict detected: {anchor_summary} competes with {tension_summary}. "
                    f"Hypothesis {leading_hypothesis.get('label')} is currently leading."
                )
            return f"Conflict detected: {anchor_summary} competes with {tension_summary}. Alternate hypotheses required."
        if resolution_path == "clarification":
            return (
                f"There is a tension between {anchor_summary} and {tension_summary}. "
                f"This looks like {category.replace('_', ' ')}, so clarification should come before anchor revision."
            )
        if resolution_path == "hypothesis_revision":
            return (
                f"Tension evidence from {tension_summary} is stronger than the current anchor {anchor_summary}. "
                "Revising the working hypothesis is justified after re-evaluation."
            )
        if leading_hypothesis:
            return (
                f"There is a meaningful conflict between the current anchor {anchor_summary} and competing evidence "
                f"from {tension_summary}. Hypothesis {leading_hypothesis.get('label')} is currently leading, "
                "but the alternatives should remain explicit until the evidence separates more clearly."
            )
        return (
            f"There is a meaningful conflict between the current anchor {anchor_summary} and competing evidence "
            f"from {tension_summary}, so alternate hypotheses should stay explicit until one becomes clearly stronger."
        )

