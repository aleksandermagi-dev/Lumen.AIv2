from __future__ import annotations

from typing import Any
from pathlib import Path

from lumen.evaluation.evaluation_models import InteractionDecisionEvaluation
from lumen.labeling.labeling_models import LabeledExample


SURFACE_TO_LABEL_CATEGORY = {
    "route_quality": "route_decision_quality",
    "intent_domain_quality": "intent_domain_quality",
    "memory_relevance_quality": "memory_relevance_quality",
    "tool_use_justification_quality": "tool_use_justification_quality",
    "confidence_calibration_quality": "confidence_calibration_quality",
    "supervised_support_quality": "supervised_support_recommendation_quality",
}


class LabelingSupport:
    """Converts evaluated interactions into labeled, provenance-rich examples."""

    def examples_from_evaluation(
        self,
        *,
        record: dict[str, Any],
        evaluation: InteractionDecisionEvaluation,
    ) -> list[LabeledExample]:
        prompt = self._clean_text(record.get("prompt"))
        source_summary = self._clean_text(record.get("summary"))
        route = record.get("route") if isinstance(record.get("route"), dict) else {}
        trainability_trace = (
            record.get("trainability_trace") if isinstance(record.get("trainability_trace"), dict) else {}
        )
        examples: list[LabeledExample] = []
        for review in evaluation.surface_reviews:
            label_category = SURFACE_TO_LABEL_CATEGORY.get(review.surface, review.surface)
            example_id = self._example_id(
                interaction_path=evaluation.interaction_path,
                created_at=evaluation.created_at,
                surface=review.surface,
            )
            examples.append(
                LabeledExample(
                    schema_version="1",
                    example_id=example_id,
                    session_id=evaluation.session_id,
                    project_id=self._clean_text(record.get("project_id")),
                    interaction_path=evaluation.interaction_path,
                    message_id=self._stable_message_id(record=record),
                    created_at=evaluation.created_at,
                    source_prompt=prompt,
                    source_summary=source_summary,
                    label_category=label_category,
                    label_value=review.judgment,
                    trainable=True,
                    provenance={
                        "schema_version": "1",
                        "surface": review.surface,
                        "mode": evaluation.mode,
                        "kind": evaluation.kind,
                        "route_mode": self._clean_text(route.get("mode")) or evaluation.mode,
                        "route_kind": self._clean_text(route.get("kind")) or evaluation.kind,
                        "evaluation_overall_judgment": evaluation.overall_judgment,
                        "source_session_id": evaluation.session_id,
                        "source_project_id": self._clean_text(record.get("project_id")),
                        "source_interaction_path": evaluation.interaction_path,
                        "source_message_id": self._stable_message_id(record=record),
                        "evaluation_surface": review.surface,
                    },
                    metadata={
                        "evaluation_score": review.score,
                        "evaluation_rationale": review.rationale,
                        "evidence": dict(review.evidence),
                        "available_training_surfaces": list(trainability_trace.get("available_training_surfaces") or []),
                        "deterministic_surfaces": list(trainability_trace.get("deterministic_surfaces") or []),
                    },
                )
            )
        return examples

    @staticmethod
    def _example_id(
        *,
        interaction_path: str | None,
        created_at: str | None,
        surface: str,
    ) -> str:
        base = interaction_path or created_at or "interaction"
        normalized = "".join(character if character.isalnum() else "_" for character in base)
        normalized = normalized.strip("_") or "interaction"
        return f"{normalized}__{surface}"

    @staticmethod
    def _clean_text(value: object) -> str | None:
        text = str(value or "").strip()
        return text or None

    @staticmethod
    def _stable_message_id(*, record: dict[str, Any]) -> str | None:
        session_id = str(record.get("session_id") or "").strip()
        interaction_path = str(record.get("interaction_path") or "").strip()
        created_at = str(record.get("created_at") or "").strip()
        if session_id and interaction_path:
            return f"{session_id}:{Path(interaction_path).stem}:assistant"
        if session_id and created_at:
            normalized = "".join(character if character.isalnum() else "_" for character in created_at).strip("_")
            if normalized:
                return f"{session_id}:{normalized}:assistant"
        return None
