from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path

from lumen.app.context_policy import ContextPolicy
from lumen.app.settings import AppSettings
from lumen.evaluation.evaluation_runner import EvaluationRunner
from lumen.labeling.dataset_exporter import DatasetExporter
from lumen.labeling.labeling_support import LabelingSupport
from lumen.nlu.follow_up_inventory import looks_like_reference_follow_up
from lumen.nlu.prompt_surface_builder import PromptSurfaceBuilder
from lumen.memory.interaction_log_manager import InteractionLogManager
from lumen.reporting.report_models import InteractionPatternsReport, InteractionSummaryReport
from lumen.reporting.report_models import (
    InteractionEvaluationReport,
    InteractionListReport,
    InteractionSearchReport,
    LabeledDatasetExportReport,
)
from lumen.retrieval.context_models import CompactContextMatch
from lumen.memory.memory_models import MemoryClassification


class InteractionHistoryService:
    """Handles persistence and inspection of assistant interactions."""

    def __init__(self, interaction_log_manager: InteractionLogManager, settings: AppSettings):
        self.interaction_log_manager = interaction_log_manager
        self.settings = settings
        self.context_policy = ContextPolicy.from_settings(settings)
        self.evaluation_runner = EvaluationRunner()
        self.labeling_support = LabelingSupport()
        self.dataset_exporter = DatasetExporter(settings)

    def record_interaction(
        self,
        *,
        session_id: str,
        prompt: str,
        response: dict[str, object],
        project_id: str | None = None,
        project_name: str | None = None,
    ) -> dict[str, object]:
        return self.interaction_log_manager.record_interaction(
            session_id=session_id,
            prompt=prompt,
            response=response,
            project_id=project_id,
            project_name=project_name,
        )

    def inspect_session(self, session_id: str) -> dict[str, object]:
        report = self.interaction_log_manager.inspect_session(session_id)
        return {
            **report,
            "records": [self._decorate_record(record) for record in report["records"]],
        }

    def list_records(
        self,
        *,
        session_id: str | None = None,
        project_id: str | None = None,
        resolution_strategy: str | None = None,
    ) -> dict[str, object]:
        records = self.interaction_log_manager.list_records(session_id=session_id, project_id=project_id)
        records = self._filter_by_resolution_strategy(records, resolution_strategy)
        return InteractionListReport(
            repo_root=str(self.interaction_log_manager.settings.repo_root),
            session_id=session_id,
            resolution_strategy=resolution_strategy,
            interaction_count=len(records),
            interaction_records=[self._decorate_record(record) for record in records],
        ).to_dict()

    def search_interactions(
        self,
        query: str,
        *,
        session_id: str | None = None,
        project_id: str | None = None,
        resolution_strategy: str | None = None,
        limit: int | None = None,
    ) -> dict[str, object]:
        matches = self.interaction_log_manager.search_records(
            query,
            session_id=session_id,
            project_id=project_id,
        )
        if resolution_strategy:
            matches = [
                match
                for match in matches
                if str(match["record"].get("resolution_strategy") or "").strip().lower()
                == resolution_strategy.strip().lower()
            ]
        if limit is not None:
            matches = matches[:limit]
        return InteractionSearchReport(
            repo_root=str(self.interaction_log_manager.settings.repo_root),
            session_id=session_id,
            query=query,
            resolution_strategy=resolution_strategy,
            interaction_count=len(matches),
            matches=[self._decorate_match(match) for match in matches],
        ).to_dict()

    def summarize_interactions(
        self,
        *,
        session_id: str | None = None,
        project_id: str | None = None,
    ) -> dict[str, object]:
        records = self.interaction_log_manager.list_records(session_id=session_id, project_id=project_id)
        decorated_records = [self._decorate_record(record) for record in records]
        mode_counts: dict[str, int] = {}
        kind_counts: dict[str, int] = {}
        posture_counts: dict[str, int] = {}
        detected_language_counts: dict[str, int] = {}
        dominant_intent_counts: dict[str, int] = {}
        local_context_assessment_counts: dict[str, int] = {}
        tool_route_origin_counts: dict[str, int] = {}
        resolution_counts: dict[str, int] = {}
        retrieval_lead_counts: dict[str, int] = {}
        evidence_strength_counts: dict[str, int] = {}
        evidence_source_counts: dict[str, int] = {}
        missing_source_counts: dict[str, int] = {}
        contradiction_flag_counts: dict[str, int] = {}
        memory_classification_counts: dict[str, int] = {}
        memory_write_action_counts: dict[str, int] = {}
        clarification_count = 0
        coherence_topic_count = 0
        semantic_route_count = 0
        deep_validation_count = 0
        contradiction_signal_count = 0
        memory_save_eligible_count = 0
        explicit_memory_consent_count = 0
        memory_surface_block_count = 0
        personal_memory_saved_count = 0
        research_note_count = 0
        research_artifact_count = 0
        research_artifact_type_counts: dict[str, int] = {}
        route_normalized_scores: list[float] = []
        route_intent_bias_count = 0
        route_intent_caution_count = 0
        retrieval_route_caution_count = 0
        retrieval_observation_count = 0

        for record in decorated_records:
            mode = str(record.get("mode", "unknown"))
            kind = str(record.get("kind", "unknown"))
            posture = str(record.get("confidence_posture") or "unknown")
            detected_language = str(record.get("detected_language") or "unknown")
            dominant_intent = str(record.get("dominant_intent") or "unknown")
            local_context_assessment = str(record.get("local_context_assessment") or "none")
            tool_route_origin = str(record.get("tool_route_origin") or "none")
            resolution = str(record.get("resolution_strategy") or "none")
            if mode == "clarification":
                clarification_count += 1
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
            kind_counts[kind] = kind_counts.get(kind, 0) + 1
            posture_counts[posture] = posture_counts.get(posture, 0) + 1
            detected_language_counts[detected_language] = detected_language_counts.get(detected_language, 0) + 1
            dominant_intent_counts[dominant_intent] = dominant_intent_counts.get(dominant_intent, 0) + 1
            local_context_assessment_counts[local_context_assessment] = (
                local_context_assessment_counts.get(local_context_assessment, 0) + 1
            )
            tool_route_origin_counts[tool_route_origin] = tool_route_origin_counts.get(tool_route_origin, 0) + 1
            resolution_counts[resolution] = resolution_counts.get(resolution, 0) + 1
            evidence_strength = str(record.get("evidence_strength") or "none")
            evidence_strength_counts[evidence_strength] = evidence_strength_counts.get(evidence_strength, 0) + 1
            memory_classification = record.get("memory_classification") or {}
            memory_candidate_type = str(memory_classification.get("candidate_type") or "unknown")
            memory_classification_counts[memory_candidate_type] = (
                memory_classification_counts.get(memory_candidate_type, 0) + 1
            )
            memory_write_decision = record.get("memory_write_decision") or {}
            memory_write_action = str(memory_write_decision.get("action") or "unknown")
            memory_write_action_counts[memory_write_action] = (
                memory_write_action_counts.get(memory_write_action, 0) + 1
            )
            if bool(memory_classification.get("save_eligible")):
                memory_save_eligible_count += 1
            if bool(memory_classification.get("requires_explicit_user_consent")):
                explicit_memory_consent_count += 1
            if bool(memory_write_decision.get("blocked_by_surface_policy")):
                memory_surface_block_count += 1
            if isinstance(record.get("personal_memory"), dict):
                personal_memory_saved_count += 1
            if isinstance(record.get("research_note"), dict):
                research_note_count += 1
                promoted_artifacts = record["research_note"].get("promoted_artifacts") or []
                research_artifact_count += len(promoted_artifacts)
                for artifact in promoted_artifacts:
                    if not isinstance(artifact, dict):
                        continue
                    artifact_type = str(artifact.get("artifact_type") or "").strip()
                    if artifact_type:
                        research_artifact_type_counts[artifact_type] = (
                            research_artifact_type_counts.get(artifact_type, 0) + 1
                        )
            for source in record.get("evidence_sources") or []:
                source_name = str(source.get("source") or "").strip()
                if source_name:
                    evidence_source_counts[source_name] = evidence_source_counts.get(source_name, 0) + 1
            for source_name in record.get("missing_sources") or []:
                normalized_source = str(source_name).strip()
                if normalized_source:
                    missing_source_counts[normalized_source] = missing_source_counts.get(normalized_source, 0) + 1
            if str(record.get("coherence_topic") or "").strip():
                coherence_topic_count += 1
            if float(record.get("route_semantic_bonus") or 0.0) > 0:
                semantic_route_count += 1
            if bool(record.get("deep_validation_used")):
                deep_validation_count += 1
            contradiction_flags = record.get("contradiction_flags") or []
            contradiction_signal_count += len(contradiction_flags)
            for flag in contradiction_flags:
                normalized_flag = str(flag).strip()
                if normalized_flag:
                    contradiction_flag_counts[normalized_flag] = contradiction_flag_counts.get(normalized_flag, 0) + 1
            if record.get("route_normalized_score") is not None:
                route_normalized_scores.append(float(record.get("route_normalized_score") or 0.0))
            if bool(record.get("route_intent_bias")):
                route_intent_bias_count += 1
            if bool(record.get("route_intent_caution")):
                route_intent_caution_count += 1
            if bool(record.get("route_retrieval_caution")):
                retrieval_route_caution_count += 1
            retrieval_context = record.get("context") if isinstance(record.get("context"), dict) else {}
            lead = self._retrieval_lead_label(retrieval_context)
            if lead is not None:
                retrieval_lead_counts[lead] = retrieval_lead_counts.get(lead, 0) + 1
                retrieval_observation_count += 1

        recent_records = decorated_records[:5]
        posture_trend = [
            str(record.get("confidence_posture") or "unknown")
            for record in recent_records
        ]
        clarification_trend = [
            "clarified" if str(record.get("mode") or "").strip() == "clarification" else "clear"
            for record in recent_records
        ]
        latest_posture = posture_trend[0] if posture_trend else None
        latest_clarification = clarification_trend[0] if clarification_trend else None
        recent_topics = [
            topic
            for topic in (
                str(record.get("normalized_topic") or "").strip()
                for record in recent_records
            )
            if topic
        ]

        return InteractionSummaryReport(
            repo_root=str(self.interaction_log_manager.settings.repo_root),
            session_id=session_id,
            interaction_count=len(decorated_records),
            clarification_count=clarification_count,
            clarification_ratio=round((clarification_count / len(decorated_records)), 4) if decorated_records else 0.0,
            clarification_trend=clarification_trend,
            recent_clarification_mix=self._recent_clarification_mix(clarification_trend),
            latest_clarification=latest_clarification,
            clarification_drift=self._clarification_drift(clarification_trend),
            mode_counts=mode_counts,
            kind_counts=kind_counts,
            posture_counts=posture_counts,
            posture_trend=posture_trend,
            recent_posture_mix=self._recent_posture_mix(posture_trend),
            latest_posture=latest_posture,
            posture_drift=self._posture_drift(posture_trend),
            detected_language_counts=detected_language_counts,
            dominant_intent_counts=dominant_intent_counts,
            local_context_assessment_counts=local_context_assessment_counts,
            coherence_topic_count=coherence_topic_count,
            semantic_route_count=semantic_route_count,
            semantic_route_ratio=round((semantic_route_count / len(decorated_records)), 4) if decorated_records else 0.0,
            route_normalized_score_count=len(route_normalized_scores),
            route_normalized_score_avg=round((sum(route_normalized_scores) / len(route_normalized_scores)), 4)
            if route_normalized_scores
            else 0.0,
            route_normalized_score_max=round(max(route_normalized_scores), 4) if route_normalized_scores else 0.0,
            route_intent_bias_count=route_intent_bias_count,
            route_intent_bias_ratio=round((route_intent_bias_count / len(decorated_records)), 4) if decorated_records else 0.0,
            route_intent_caution_count=route_intent_caution_count,
            route_intent_caution_ratio=(
                round((route_intent_caution_count / len(decorated_records)), 4)
                if decorated_records
                else 0.0
            ),
            retrieval_route_caution_count=retrieval_route_caution_count,
            retrieval_route_caution_ratio=(
                round((retrieval_route_caution_count / len(decorated_records)), 4)
                if decorated_records
                else 0.0
            ),
            retrieval_lead_counts=retrieval_lead_counts,
            retrieval_observation_count=retrieval_observation_count,
            evidence_strength_counts=evidence_strength_counts,
            evidence_source_counts=evidence_source_counts,
            missing_source_counts=missing_source_counts,
            deep_validation_count=deep_validation_count,
            deep_validation_ratio=round((deep_validation_count / len(decorated_records)), 4) if decorated_records else 0.0,
            contradiction_signal_count=contradiction_signal_count,
            contradiction_flag_counts=contradiction_flag_counts,
            memory_classification_counts=memory_classification_counts,
            memory_write_action_counts=memory_write_action_counts,
            memory_save_eligible_count=memory_save_eligible_count,
            explicit_memory_consent_count=explicit_memory_consent_count,
            memory_surface_block_count=memory_surface_block_count,
            personal_memory_saved_count=personal_memory_saved_count,
            research_note_count=research_note_count,
            research_artifact_count=research_artifact_count,
            research_artifact_type_counts=research_artifact_type_counts,
            recent_topics=recent_topics,
            tool_route_origin_counts=tool_route_origin_counts,
            resolution_counts=resolution_counts,
            recent_interactions=recent_records,
        ).to_dict()

    def evaluate_interactions(
        self,
        *,
        session_id: str | None = None,
        project_id: str | None = None,
    ) -> dict[str, object]:
        records = self.interaction_log_manager.list_records(session_id=session_id, project_id=project_id)
        batch = self.evaluation_runner.evaluate_records(records, session_id=session_id)
        return InteractionEvaluationReport(
            repo_root=str(self.interaction_log_manager.settings.repo_root),
            session_id=session_id,
            evaluated_count=batch.evaluated_count,
            judgment_counts=batch.judgment_counts,
            surface_aggregates=[aggregate.to_dict() for aggregate in batch.surface_aggregates],
            evaluations=[evaluation.to_dict() for evaluation in batch.evaluations],
            behavior_slices=list(batch.behavior_slices),
        ).to_dict()

    def export_labeled_examples(
        self,
        *,
        session_id: str | None = None,
        project_id: str | None = None,
    ) -> dict[str, object]:
        records = self.interaction_log_manager.list_records(session_id=session_id, project_id=project_id)
        batch = self.evaluation_runner.evaluate_records(records, session_id=session_id)
        examples = []
        for record, evaluation in zip(records, batch.evaluations, strict=False):
            examples.extend(
                self.labeling_support.examples_from_evaluation(
                    record=record,
                    evaluation=evaluation,
                )
            )
        export_result = self.dataset_exporter.export_examples(
            examples=examples,
            session_id=session_id,
            project_id=project_id,
        )
        return LabeledDatasetExportReport(
            repo_root=str(self.interaction_log_manager.settings.repo_root),
            session_id=session_id,
            project_id=project_id,
            dataset_path=export_result.dataset_path,
            manifest_path=export_result.manifest_path,
            export_batch_id=export_result.export_batch_id,
            example_count=export_result.example_count,
            split_counts=export_result.split_counts,
            label_category_counts=export_result.label_category_counts,
            label_value_counts=export_result.label_value_counts,
        ).to_dict()

    def summarize_patterns(
        self,
        *,
        session_id: str | None = None,
        project_id: str | None = None,
    ) -> dict[str, object]:
        records = self.interaction_log_manager.list_records(session_id=session_id, project_id=project_id)
        resolution_counts: dict[str, int] = {}
        follow_up_count = 0
        ambiguous_follow_up_count = 0
        rewritten_count = 0

        for record in records:
            resolution = str(record.get("resolution_strategy") or "none")
            resolution_counts[resolution] = resolution_counts.get(resolution, 0) + 1
            if resolution.strip().lower() != "none" and not self._ignore_pattern_rewrite(record):
                rewritten_count += 1

            prompt = str(record.get("prompt") or "")
            is_follow_up = self._looks_like_follow_up(prompt)
            if is_follow_up:
                follow_up_count += 1
                if resolution == "none":
                    ambiguous_follow_up_count += 1

        interaction_count = len(records)
        rewrite_ratio = (rewritten_count / interaction_count) if interaction_count else 0.0
        follow_up_ratio = (follow_up_count / interaction_count) if interaction_count else 0.0
        ambiguity_ratio = (ambiguous_follow_up_count / follow_up_count) if follow_up_count else 0.0

        status = "ok"
        observations: list[str] = []
        if interaction_count >= 3 and rewrite_ratio >= 0.5:
            status = "warn"
            observations.append("Prompt-resolution rewrites are common in this interaction history.")
        if follow_up_count >= 2 and ambiguity_ratio >= 0.5:
            status = "warn"
            observations.append("Several follow-up prompts remained unresolved by shorthand interpretation.")
        if not observations:
            observations.append("Interaction patterns look stable.")

        return InteractionPatternsReport(
            repo_root=str(self.interaction_log_manager.settings.repo_root),
            session_id=session_id,
            interaction_count=interaction_count,
            follow_up_count=follow_up_count,
            ambiguous_follow_up_count=ambiguous_follow_up_count,
            rewrite_ratio=round(rewrite_ratio, 4),
            follow_up_ratio=round(follow_up_ratio, 4),
            ambiguity_ratio=round(ambiguity_ratio, 4),
            resolution_counts=resolution_counts,
            observations=observations,
            status=status,
            recent_interactions=[self._decorate_record(record) for record in records[:5]],
        ).to_dict()

    @staticmethod
    def _ignore_pattern_rewrite(record: dict[str, object]) -> bool:
        resolution = str(record.get("resolution_strategy") or "").strip().lower()
        if resolution != "wake_phrase_strip":
            return False
        prompt = str(record.get("prompt") or "").strip()
        wake_interaction = record.get("wake_interaction") or {}
        stripped_prompt = str(wake_interaction.get("stripped_prompt") or "").strip()
        if InteractionHistoryService._looks_like_command_like_prompt(prompt):
            return True
        if stripped_prompt and InteractionHistoryService._looks_like_command_like_prompt(stripped_prompt):
            return True
        return False

    def retrieve_context(
        self,
        query: str,
        *,
        session_id: str | None = None,
        project_id: str | None = None,
        limit: int | None = None,
    ) -> dict[str, object]:
        effective_limit = limit if limit is not None else self.settings.context_match_limit
        matches = self.interaction_log_manager.search_records(
            query,
            session_id=session_id,
            project_id=project_id,
        )[:effective_limit]
        return {
            "interaction_record_count": len(matches),
            "top_interaction_matches": [self._compact_context_match(match) for match in matches],
        }

    def recent_records(
        self,
        *,
        session_id: str | None = None,
        project_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, object]]:
        effective_limit = limit if limit is not None else self.settings.context_match_limit
        records = self.interaction_log_manager.list_records(
            session_id=session_id,
            project_id=project_id,
        )[:effective_limit]
        return [self._decorate_record(record) for record in records]

    def save_memory_from_record(
        self,
        *,
        source_record: dict[str, object],
        target: str,
        client_surface: str = "main",
    ) -> dict[str, object]:
        timestamp = datetime.now(UTC)
        source = dict(source_record)
        session_id = str(source.get("session_id") or "").strip()
        if not session_id:
            return {"status": "error", "reason": "missing_session_id"}
        source_path = str(source.get("interaction_path") or "").strip() or None

        if target == "research":
            source["memory_classification"] = MemoryClassification.research_candidate(
                confidence=0.95,
                reason="The user explicitly selected the prior research context for saving.",
            ).to_dict()
            note = self.interaction_log_manager.research_note_manager.record_note(
                session_id=session_id,
                timestamp=timestamp,
                record=source,
                client_surface=client_surface,
                source_interaction_path=source_path,
            )
            source["research_note"] = note
            if self.interaction_log_manager.graph_memory_manager is not None:
                self.interaction_log_manager.graph_memory_manager.ingest_interaction_memory(record=source)
            return {
                "status": "ok",
                "target": "research",
                "research_note": note,
            }

        if target == "personal":
            source["memory_classification"] = MemoryClassification.personal_candidate(
                confidence=0.95,
                reason="The user explicitly selected the prior personal or reflective context for saving.",
                explicit_save_requested=True,
            ).to_dict()
            entry = self.interaction_log_manager.personal_memory_manager.record_entry(
                session_id=session_id,
                timestamp=timestamp,
                record=source,
                client_surface=client_surface,
                source_interaction_path=source_path,
            )
            source["personal_memory"] = entry
            if self.interaction_log_manager.graph_memory_manager is not None:
                self.interaction_log_manager.graph_memory_manager.ingest_interaction_memory(record=source)
            return {
                "status": "ok",
                "target": "personal",
                "personal_memory": entry,
            }

        if target == "assistant":
            source["memory_classification"] = MemoryClassification.personal_candidate(
                confidence=0.95,
                reason="The user explicitly asked Lumen to remember its own prior answer.",
                explicit_save_requested=True,
            ).to_dict()
            prompt_text = str(source.get("prompt") or "").strip()
            summary_text = str(source.get("summary") or "").strip()
            response_payload = source.get("response") if isinstance(source.get("response"), dict) else {}
            assistant_text = (
                str(response_payload.get("user_facing_answer") or "").strip()
                or str(response_payload.get("reply") or "").strip()
                or summary_text
            )
            topic = str(source.get("normalized_topic") or "").strip() or "assistant_answer"
            title = prompt_text or f"assistant answer about {topic}"
            content_lines = [f"Assistant answer: {assistant_text}"]
            if prompt_text:
                content_lines.append(f"Source prompt: {prompt_text}")
            entry = self.interaction_log_manager.personal_memory_manager.record_entry(
                session_id=session_id,
                timestamp=timestamp,
                record=source,
                client_surface=client_surface,
                source_interaction_path=source_path,
                title_override=title,
                content_override="\n".join(line for line in content_lines if line),
                memory_origin="assistant",
            )
            source["personal_memory"] = entry
            if self.interaction_log_manager.graph_memory_manager is not None:
                self.interaction_log_manager.graph_memory_manager.ingest_interaction_memory(record=source)
            return {
                "status": "ok",
                "target": "assistant",
                "personal_memory": entry,
            }

        return {"status": "error", "reason": f"unsupported_target:{target}"}

    @staticmethod
    def _filter_by_resolution_strategy(
        records: list[dict[str, object]],
        resolution_strategy: str | None,
    ) -> list[dict[str, object]]:
        if not resolution_strategy:
            return records
        needle = resolution_strategy.strip().lower()
        return [
            record
            for record in records
            if str(record.get("resolution_strategy") or "").strip().lower() == needle
        ]

    @staticmethod
    def _looks_like_follow_up(prompt: str) -> bool:
        normalized = PromptSurfaceBuilder.build(prompt).lookup_ready_text
        return looks_like_reference_follow_up(normalized)

    @staticmethod
    def _looks_like_command_like_prompt(prompt: str) -> bool:
        normalized = PromptSurfaceBuilder.build(prompt).tool_ready_text
        if not normalized:
            return False
        command_prefixes = (
            "doctor",
            "session ",
            "interaction ",
            "archive ",
            "memory ",
            "config ",
            "tool ",
            "bundle ",
            "profile ",
        )
        return normalized == "doctor" or any(normalized.startswith(prefix) for prefix in command_prefixes)

    @staticmethod
    def _decorate_record(record: dict[str, object]) -> dict[str, object]:
        decorated = dict(record)
        original_prompt = str(record.get("prompt") or "").strip() or None
        resolved_prompt = str(record.get("resolved_prompt") or "").strip() or None
        decorated["prompt_view"] = {
            "canonical_prompt": resolved_prompt or original_prompt,
            "original_prompt": original_prompt,
            "resolved_prompt": resolved_prompt,
            "rewritten": bool(resolved_prompt),
        }
        decorated["confidence_posture"] = str(
            record.get("confidence_posture")
            or (
                (record.get("response") or {}).get("confidence_posture")
                if isinstance(record.get("response"), dict)
                else ""
            )
            or ""
        ).strip() or None
        decorated["tool_route_origin"] = str(
            record.get("tool_route_origin")
            or (
                (record.get("response") or {}).get("tool_route_origin")
                if isinstance(record.get("response"), dict)
                else ""
            )
            or ""
        ).strip() or None
        decorated["local_context_assessment"] = str(
            record.get("local_context_assessment")
            or (
                (record.get("response") or {}).get("local_context_assessment")
                if isinstance(record.get("response"), dict)
                else ""
            )
            or ""
        ).strip() or None
        decorated["coherence_topic"] = str(
            record.get("coherence_topic")
            or (
                ((record.get("response") or {}).get("reasoning_frame") or {}).get("coherence_topic")
                if isinstance(record.get("response"), dict)
                else ""
            )
            or ""
        ).strip() or None
        route = record.get("route")
        route_semantic_bonus = None
        route_normalized_score = None
        if isinstance(route, dict):
            decision_summary = route.get("decision_summary") or {}
            selected = decision_summary.get("selected") or {}
            route_semantic_bonus = selected.get("semantic_bonus")
            route_normalized_score = selected.get("normalized_score")
        if route_semantic_bonus is None and isinstance(record.get("response"), dict):
            response_route = (record.get("response") or {}).get("route") or {}
            if isinstance(response_route, dict):
                decision_summary = response_route.get("decision_summary") or {}
                selected = decision_summary.get("selected") or {}
                route_semantic_bonus = selected.get("semantic_bonus")
                route_normalized_score = selected.get("normalized_score")
        decorated["route_semantic_bonus"] = float(route_semantic_bonus or 0.0)
        decorated["route_normalized_score"] = (
            round(float(route_normalized_score), 4)
            if route_normalized_score is not None
            else None
        )
        route_evidence = []
        if isinstance(route, dict):
            route_evidence = route.get("evidence") or []
        if not route_evidence and isinstance(record.get("response"), dict):
            response_route = (record.get("response") or {}).get("route") or {}
            if isinstance(response_route, dict):
                route_evidence = response_route.get("evidence") or []
        decorated["route_retrieval_caution"] = any(
            isinstance(item, dict) and str(item.get("label") or "").strip() == "retrieval_bias_caution"
            for item in route_evidence
        )
        decorated["route_intent_bias"] = any(
            isinstance(item, dict) and str(item.get("label") or "").strip() == "session_intent_bias"
            for item in route_evidence
        )
        decorated["route_intent_caution"] = any(
            isinstance(item, dict) and str(item.get("label") or "").strip() == "session_intent_caution"
            for item in route_evidence
        )
        pipeline_observability = record.get("pipeline_observability") if isinstance(record.get("pipeline_observability"), dict) else {}
        retrieval_summary = (
            pipeline_observability.get("retrieval_summary")
            if isinstance(pipeline_observability.get("retrieval_summary"), dict)
            else {}
        )
        reasoning_summary = (
            pipeline_observability.get("reasoning_summary")
            if isinstance(pipeline_observability.get("reasoning_summary"), dict)
            else {}
        )
        decorated["route_status"] = str(
            record.get("route_status")
            or (
                (record.get("response") or {}).get("route_status")
                if isinstance(record.get("response"), dict)
                else ""
            )
            or reasoning_summary.get("route_status")
            or ""
        ).strip() or None
        decorated["support_status"] = str(
            record.get("support_status")
            or (
                (record.get("response") or {}).get("support_status")
                if isinstance(record.get("response"), dict)
                else ""
            )
            or reasoning_summary.get("support_status")
            or ""
        ).strip() or None
        decorated["tension_status"] = str(
            record.get("tension_status")
            or (
                (record.get("response") or {}).get("tension_status")
                if isinstance(record.get("response"), dict)
                else ""
            )
            or reasoning_summary.get("tension_status")
            or ""
        ).strip() or None
        decorated["deep_validation_used"] = bool(retrieval_summary.get("deep_validation_used"))
        decorated["evidence_strength"] = str(reasoning_summary.get("evidence_strength") or "").strip() or None
        decorated["evidence_sources"] = list(reasoning_summary.get("evidence_sources") or [])
        decorated["missing_sources"] = list(reasoning_summary.get("missing_sources") or [])
        decorated["contradiction_flags"] = list(reasoning_summary.get("contradiction_flags") or [])
        decorated["memory_classification"] = dict(record.get("memory_classification") or {})
        decorated["memory_write_decision"] = dict(record.get("memory_write_decision") or {})
        decorated["research_note"] = (
            dict(record.get("research_note") or {})
            if isinstance(record.get("research_note"), dict)
            else None
        )
        if isinstance(decorated.get("research_note"), dict):
            decorated["research_note"]["promoted_artifacts"] = list(
                decorated["research_note"].get("promoted_artifacts") or []
            )
        decorated["personal_memory"] = (
            dict(record.get("personal_memory") or {})
            if isinstance(record.get("personal_memory"), dict)
            else None
        )
        decorated["client_surface"] = str(record.get("client_surface") or "main")
        return decorated

    def _decorate_match(self, match: dict[str, object]) -> dict[str, object]:
        return {
            **match,
            "record": self._decorate_record(match["record"]),
        }

    def _compact_context_match(self, match: dict[str, object]) -> dict[str, object]:
        record = self._decorate_record(match["record"])
        return CompactContextMatch(
            score=match.get("score"),
            record=self.context_policy.compact_interaction_context_record(record),
            matched_fields=tuple(
                str(field)
                for field in (match.get("matched_fields") or [])
                if str(field).strip()
            ),
            score_breakdown={
                str(key): int(value)
                for key, value in ((match.get("score_breakdown") or {}).items())
            }
            if isinstance(match.get("score_breakdown"), dict)
            else None,
        ).to_dict()

    @staticmethod
    def _retrieval_lead_label(context: dict[str, object]) -> str | None:
        candidates: list[dict[str, object]] = []
        for key in ("top_interaction_matches", "top_matches"):
            value = context.get(key)
            if isinstance(value, list) and value:
                top = value[0]
                if isinstance(top, dict):
                    candidates.append(top)
        if not candidates:
            return None
        best = max(candidates, key=lambda item: int(item.get("score") or 0))
        breakdown = best.get("score_breakdown")
        if not isinstance(breakdown, dict):
            return None
        keyword_score = int(breakdown.get("keyword_score") or 0)
        semantic_score = int(breakdown.get("semantic_score") or 0)
        if keyword_score > 0 and semantic_score > 0:
            return "blended"
        if semantic_score > 0:
            return "semantic"
        if keyword_score > 0:
            return "keyword"
        return "none"

    @staticmethod
    def _recent_posture_mix(posture_trend: list[str]) -> str | None:
        if not posture_trend:
            return None
        unique = {item for item in posture_trend if item}
        if len(unique) == 1:
            only = next(iter(unique))
            return f"stable:{only}"
        if "conflicted" in unique:
            return "mixed_with_conflict"
        if "tentative" in unique and ("supported" in unique or "strong" in unique):
            return "improving_or_mixed"
        return "mixed"

    @staticmethod
    def _posture_drift(posture_trend: list[str]) -> str | None:
        if not posture_trend:
            return None
        if len(posture_trend) == 1:
            return "insufficient_data"

        scores = {
            "conflicted": 0,
            "tentative": 1,
            "supported": 2,
            "strong": 3,
            "unknown": -1,
        }
        latest_score = scores.get(posture_trend[0], -1)
        baseline_scores = [scores.get(item, -1) for item in posture_trend[1:] if scores.get(item, -1) >= 0]
        if latest_score < 0 or not baseline_scores:
            return "insufficient_data"

        baseline = sum(baseline_scores) / len(baseline_scores)
        if latest_score >= baseline + 0.75:
            return "strengthening"
        if latest_score <= baseline - 0.75:
            return "weakening"
        if len(set(posture_trend)) > 1:
            return "steady_with_variation"
        return "steady"

    @staticmethod
    def _recent_clarification_mix(clarification_trend: list[str]) -> str | None:
        if not clarification_trend:
            return None
        unique = {item for item in clarification_trend if item}
        if len(unique) == 1:
            return f"stable:{next(iter(unique))}"
        if clarification_trend.count("clarified") >= clarification_trend.count("clear"):
            return "clarification_heavy_mixed"
        return "mixed"

    @staticmethod
    def _clarification_drift(clarification_trend: list[str]) -> str | None:
        if not clarification_trend:
            return None
        if len(clarification_trend) == 1:
            return "insufficient_data"

        latest = clarification_trend[0]
        baseline = clarification_trend[1:]
        if not baseline:
            return "insufficient_data"
        latest_score = 1 if latest == "clarified" else 0
        baseline_scores = [1 if item == "clarified" else 0 for item in baseline]
        baseline_average = sum(baseline_scores) / len(baseline_scores)
        if latest_score > baseline_average:
            return "increasing"
        if latest_score < baseline_average:
            return "decreasing"
        if len(set(clarification_trend)) > 1:
            return "steady_with_variation"
        return "steady"
