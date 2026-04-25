from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from lumen.evaluation.decision_evaluation import DecisionEvaluation
from lumen.evaluation.evaluation_models import (
    DecisionEvaluationBatch,
    EvaluationAggregate,
)
from lumen.evaluation.long_conversation_evaluation import (
    LongConversationEvaluator,
    ScriptedConversationEvaluation,
    ScriptedConversationScenario,
)


class EvaluationRunner:
    """Runs offline decision evaluation across persisted interaction records."""

    def __init__(
        self,
        evaluator: DecisionEvaluation | None = None,
        long_conversation_evaluator: LongConversationEvaluator | None = None,
    ) -> None:
        self.evaluator = evaluator or DecisionEvaluation()
        self.long_conversation_evaluator = long_conversation_evaluator or LongConversationEvaluator()

    def evaluate_records(
        self,
        records: list[dict[str, Any]],
        *,
        session_id: str | None = None,
    ) -> DecisionEvaluationBatch:
        evaluations = tuple(self.evaluator.evaluate_record(record) for record in records)
        overall_counts = Counter(evaluation.overall_judgment for evaluation in evaluations)
        aggregate_counts: dict[str, Counter[str]] = defaultdict(Counter)
        aggregate_scores: dict[str, list[float]] = defaultdict(list)
        for evaluation in evaluations:
            for review in evaluation.surface_reviews:
                aggregate_counts[review.surface][review.judgment] += 1
                if review.score is not None:
                    aggregate_scores[review.surface].append(float(review.score))
        surface_aggregates = tuple(
            EvaluationAggregate(
                surface=surface,
                reviewed_count=sum(counter.values()),
                judgment_counts=dict(counter),
                average_score=(
                    sum(aggregate_scores.get(surface, [])) / len(aggregate_scores[surface])
                    if aggregate_scores.get(surface)
                    else None
                ),
            )
            for surface, counter in sorted(aggregate_counts.items())
        )
        return DecisionEvaluationBatch(
            session_id=session_id,
            evaluated_count=len(evaluations),
            judgment_counts=dict(overall_counts),
            surface_aggregates=surface_aggregates,
            evaluations=evaluations,
            behavior_slices=tuple(self._behavior_slices(records)),
        )

    def evaluate_scripted_conversation(
        self,
        *,
        scenario: ScriptedConversationScenario,
        records: list[dict[str, Any]],
    ) -> ScriptedConversationEvaluation:
        return self.long_conversation_evaluator.evaluate(scenario=scenario, records=records)

    def _behavior_slices(self, records: list[dict[str, Any]]) -> list[dict[str, object]]:
        slices: list[dict[str, object]] = []
        slices.append(
            self._slice_summary(
                "same_project_continuity",
                records,
                lambda record: self._selected_reason_contains(record, "recent_summary_window")
                or self._selected_reason_contains(record, "message_window_continuity")
                or self._selected_reason_contains(record, "recent_interaction_window"),
            )
        )
        slices.append(
            self._slice_summary(
                "restart_continuity",
                records,
                lambda record: self._selected_source_in(record, {"session_summaries", "message_window"}),
            )
        )
        slices.append(
            self._slice_summary(
                "cross_project_semantic_dampening",
                records,
                lambda record: self._rejected_semantic_status_contains(record, "cross_project_dampened")
                or self._selected_semantic_status_contains(record, "cross_project_dampened"),
            )
        )
        slices.append(
            self._slice_summary(
                "low_confidence_caution",
                records,
                lambda record: str(record.get("confidence_posture") or "").strip().lower() in {"tentative", "conflicted"},
            )
        )
        slices.append(
            self._slice_summary(
                "retrieval_reason_stability",
                records,
                lambda record: bool(self._retrieval_diagnostics(record).get("selected_reasons"))
                and bool(self._retrieval_diagnostics(record).get("rejected_reasons")),
            )
        )
        return slices

    @staticmethod
    def _slice_summary(
        name: str,
        records: list[dict[str, Any]],
        predicate,
    ) -> dict[str, object]:
        matched = [record for record in records if predicate(record)]
        sample_paths = [
            str(record.get("interaction_path") or "").strip()
            for record in matched[:3]
            if str(record.get("interaction_path") or "").strip()
        ]
        total = len(records)
        matched_count = len(matched)
        return {
            "name": name,
            "matched_count": matched_count,
            "total_count": total,
            "ratio": round((matched_count / total), 4) if total else 0.0,
            "status": "ok" if matched_count > 0 else "insufficient_evidence",
            "sample_interaction_paths": sample_paths,
        }

    @staticmethod
    def _retrieval_diagnostics(record: dict[str, Any]) -> dict[str, Any]:
        retrieval = record.get("memory_retrieval")
        if isinstance(retrieval, dict) and isinstance(retrieval.get("diagnostics"), dict):
            return dict(retrieval.get("diagnostics") or {})
        response = record.get("response")
        if isinstance(response, dict):
            nested = response.get("memory_retrieval")
            if isinstance(nested, dict) and isinstance(nested.get("diagnostics"), dict):
                return dict(nested.get("diagnostics") or {})
        return {}

    @classmethod
    def _selected_reason_contains(cls, record: dict[str, Any], expected: str) -> bool:
        diagnostics = cls._retrieval_diagnostics(record)
        for item in diagnostics.get("selected_reasons") or []:
            if isinstance(item, dict) and str(item.get("reason") or "").strip() == expected:
                return True
        return False

    @classmethod
    def _selected_source_in(cls, record: dict[str, Any], expected: set[str]) -> bool:
        diagnostics = cls._retrieval_diagnostics(record)
        return any(str(item).strip() in expected for item in (diagnostics.get("selected_sources") or []))

    @classmethod
    def _rejected_semantic_status_contains(cls, record: dict[str, Any], expected: str) -> bool:
        diagnostics = cls._retrieval_diagnostics(record)
        for item in diagnostics.get("rejected_reasons") or []:
            if isinstance(item, dict) and expected in str(item.get("semantic_status") or ""):
                return True
        return False

    @classmethod
    def _selected_semantic_status_contains(cls, record: dict[str, Any], expected: str) -> bool:
        diagnostics = cls._retrieval_diagnostics(record)
        for item in diagnostics.get("selected_reasons") or []:
            if isinstance(item, dict) and expected in str(item.get("semantic_status") or ""):
                return True
        return False
