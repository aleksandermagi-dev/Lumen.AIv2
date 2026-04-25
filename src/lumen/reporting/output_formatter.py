from __future__ import annotations

import json
from typing import Any

from lumen.reasoning.reasoning_vocabulary import display_status_label
from lumen.reporting.report_models import ArchiveListReport
from lumen.reporting.report_models import ActiveThreadReport, SessionReport, SessionResetReport
from lumen.schemas.output_schema import OutputSchema
from lumen.schemas.retrieval_schema import RetrievalResultSchema
from lumen.tools.registry_types import ToolResult


class OutputFormatter:
    """Formats durable/exportable payloads for CLI and reporting surfaces.

    Interactive turn shaping belongs to the response pipeline. This formatter is
    the reporting boundary for rendered payloads, archive views, and export-like
    outputs after a response has already been composed.
    """

    def render_json(self, payload: dict[str, Any]) -> str:
        return json.dumps(payload, indent=2)

    def render_text(self, payload: dict[str, Any]) -> str:
        if "semantic_model_name" in payload and "total_memory_items" in payload:
            return self._render_semantic_status_text(payload)
        if "processed" in payload and "runtime_available" in payload and "model_name" in payload:
            return self._render_semantic_backfill_text(payload)
        if "db_path" in payload and "table_counts" in payload and "migrations" in payload:
            return self._render_persistence_status_text(payload)
        if "db_path" in payload and "db_counts" in payload and "legacy_counts" in payload:
            return self._render_persistence_coverage_text(payload)
        if "db_path" in payload and "missing_rows" in payload and "orphan_counts" in payload:
            return self._render_persistence_doctor_text(payload)
        if "bundle_id" in payload and "manifest_path" in payload:
            return self._render_bundle_text(payload)
        if "research_notes" in payload and "note_count" in payload:
            return self._render_research_notes_text(payload)
        if "research_artifacts" in payload and "artifact_count" in payload:
            return self._render_research_artifacts_text(payload)
        if payload.get("status") == "ok" and isinstance(payload.get("artifact"), dict):
            return self._render_research_artifact_promotion_text(payload)
        if payload.get("schema_type") == "assistant_response":
            return self._render_assistant_response_text(payload)
        if "active_thread" in payload and "cleared" in payload:
            return self._render_session_reset_text(payload)
        if "interaction_profile" in payload and "active_thread" not in payload and "records" not in payload:
            return self._render_session_profile_text(payload)
        if "active_thread" in payload and "records" not in payload and "interaction_records" not in payload:
            return self._render_active_thread_text(payload)
        if "dataset_path" in payload and "label_category_counts" in payload and "example_count" in payload:
            return self._render_labeled_dataset_export_text(payload)
        if payload.get("schema_type") == "dataset_jsonl_export":
            return self._render_dataset_jsonl_export_text(payload)
        if payload.get("schema_type") == "dataset_review_batch":
            return self._render_dataset_review_batch_text(payload)
        if payload.get("schema_type") == "dataset_run_comparison":
            return self._render_dataset_run_comparison_text(payload)
        if payload.get("schema_type") == "dataset_example_label_update":
            return self._render_dataset_label_update_text(payload)
        if payload.get("schema_type") == "dataset_example_update":
            return self._render_dataset_example_update_text(payload)
        if "surface_aggregates" in payload and "evaluations" in payload and "evaluated_count" in payload:
            return self._render_interaction_evaluation_text(payload)
        if "follow_up_count" in payload and "ambiguity_ratio" in payload:
            return self._render_interaction_patterns_text(payload)
        if "resolution_counts" in payload and "mode_counts" in payload:
            return self._render_interaction_summary_text(payload)
        if "interaction_records" in payload and "records" not in payload:
            return self._render_interactions_text(payload)
        if "matches" in payload and "interaction_count" in payload and "query" in payload and "record_count" not in payload:
            return self._render_interaction_search_text(payload)
        if "target_groups" in payload and "target_count" in payload and payload.get("schema_type") == "archive_compare_result":
            return self._render_archive_compare_text(payload)
        if "latest_by_capability" in payload and "status_counts" in payload:
            return self._render_archive_summary_text(payload)
        if "record" in payload and "found" in payload:
            return self._render_latest_text(payload)
        if "matches" in payload and "query" in payload:
            return self._render_search_text(payload)
        if "created_paths" in payload or "existing_paths" in payload:
            return self._render_init_text(payload)
        if "tools" in payload:
            return self._render_tools_text(payload)
        if payload and all(isinstance(value, dict) and "tool_id" in value for value in payload.values()):
            return self._render_capabilities_text(payload)
        if "tool_id" in payload and "capability" in payload:
            return self._render_tool_result_text(payload)
        if "checks" in payload:
            return self._render_doctor_text(payload)
        if "records" in payload and "session_id" in payload:
            return self._render_records_text(payload)
        if payload.get("status") == "error":
            return self._render_error_text(payload)
        return self.render_json(payload)

    def tool_result_payload(self, result: ToolResult) -> dict[str, Any]:
        return OutputSchema.build_tool_result_payload(result)

    def bundle_inspection_payload(
        self,
        *,
        bundle_id: str,
        name: str,
        version: str,
        schema_version: str,
        description: str,
        manifest_path: str | None,
        entrypoint: str,
        capabilities: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return OutputSchema.build_bundle_inspection_payload(
            bundle_id=bundle_id,
            name=name,
            version=version,
            schema_version=schema_version,
            description=description,
            manifest_path=manifest_path,
            entrypoint=entrypoint,
            capabilities=capabilities,
        )

    def error_payload(
        self,
        *,
        exc: Exception,
        available_tools: dict[str, list[str]],
    ) -> dict[str, Any]:
        return {
            "status": "error",
            "error_type": exc.__class__.__name__,
            "message": str(exc),
            "available_tools": available_tools,
        }

    def archive_records_payload(
        self,
        *,
        repo_root: str,
        session_id: str | None,
        tool_id: str | None,
        capability: str | None,
        records: list[dict[str, Any]],
        query: str | None = None,
        status: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> dict[str, Any]:
        return ArchiveListReport(
            repo_root=repo_root,
            session_id=session_id,
            tool_id=tool_id,
            capability=capability,
            query=query,
            status_filter=status,
            date_from=date_from,
            date_to=date_to,
            record_count=len(records),
            records=records,
        ).to_dict()

    def session_payload(
        self,
        *,
        repo_root: str,
        session_id: str,
        records: list[dict[str, Any]],
        interaction_records: list[dict[str, Any]] | None = None,
        interaction_profile: dict[str, Any] | None = None,
        active_thread: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        posture_counts: dict[str, int] = {}
        detected_language_counts: dict[str, int] = {}
        dominant_intent_counts: dict[str, int] = {}
        local_context_assessment_counts: dict[str, int] = {}
        tool_route_origin_counts: dict[str, int] = {}
        posture_trend: list[str] = []
        recent_topics: list[str] = []
        retrieval_lead_counts: dict[str, int] = {}
        evidence_strength_counts: dict[str, int] = {}
        evidence_source_counts: dict[str, int] = {}
        missing_source_counts: dict[str, int] = {}
        contradiction_flag_counts: dict[str, int] = {}
        memory_classification_counts: dict[str, int] = {}
        memory_write_action_counts: dict[str, int] = {}
        clarification_count = 0
        clarification_trend: list[str] = []
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
        for record in interaction_records or []:
            posture = str(record.get("confidence_posture") or "unknown")
            detected_language = str(record.get("detected_language") or "unknown")
            dominant_intent = str(record.get("dominant_intent") or "unknown")
            local_context_assessment = str(record.get("local_context_assessment") or "none")
            tool_route_origin = str(record.get("tool_route_origin") or "none")
            posture_counts[posture] = posture_counts.get(posture, 0) + 1
            detected_language_counts[detected_language] = detected_language_counts.get(detected_language, 0) + 1
            dominant_intent_counts[dominant_intent] = dominant_intent_counts.get(dominant_intent, 0) + 1
            local_context_assessment_counts[local_context_assessment] = (
                local_context_assessment_counts.get(local_context_assessment, 0) + 1
            )
            tool_route_origin_counts[tool_route_origin] = tool_route_origin_counts.get(tool_route_origin, 0) + 1
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
            context = record.get("context")
            if isinstance(context, dict):
                lead = self._retrieval_lead_label(context)
                if lead is not None:
                    retrieval_lead_counts[lead] = retrieval_lead_counts.get(lead, 0) + 1
                    retrieval_observation_count += 1
            topic = str(record.get("normalized_topic") or "").strip()
            if topic and len(recent_topics) < 5:
                recent_topics.append(topic)
            is_clarification = str(record.get("mode") or "").strip() == "clarification"
            if is_clarification:
                clarification_count += 1
            if len(posture_trend) < 5:
                posture_trend.append(posture)
            if len(clarification_trend) < 5:
                clarification_trend.append("clarified" if is_clarification else "clear")
        latest_posture = posture_trend[0] if posture_trend else None
        interaction_count = len(interaction_records or [])
        return SessionReport(
            repo_root=repo_root,
            session_id=session_id,
            record_count=len(records),
            records=records,
            interaction_count=interaction_count,
            clarification_count=clarification_count,
            clarification_ratio=round((clarification_count / interaction_count), 4) if interaction_count else 0.0,
            clarification_trend=clarification_trend,
            recent_clarification_mix=self._recent_clarification_mix(clarification_trend),
            latest_clarification=clarification_trend[0] if clarification_trend else None,
            clarification_drift=self._clarification_drift(clarification_trend),
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
            semantic_route_ratio=round((semantic_route_count / interaction_count), 4) if interaction_count else 0.0,
            route_normalized_score_count=len(route_normalized_scores),
            route_normalized_score_avg=round((sum(route_normalized_scores) / len(route_normalized_scores)), 4)
            if route_normalized_scores
            else 0.0,
            route_normalized_score_max=round(max(route_normalized_scores), 4) if route_normalized_scores else 0.0,
            route_intent_bias_count=route_intent_bias_count,
            route_intent_bias_ratio=round((route_intent_bias_count / interaction_count), 4) if interaction_count else 0.0,
            route_intent_caution_count=route_intent_caution_count,
            route_intent_caution_ratio=round((route_intent_caution_count / interaction_count), 4) if interaction_count else 0.0,
            retrieval_route_caution_count=retrieval_route_caution_count,
            retrieval_route_caution_ratio=(
                round((retrieval_route_caution_count / interaction_count), 4) if interaction_count else 0.0
            ),
            retrieval_lead_counts=retrieval_lead_counts,
            retrieval_observation_count=retrieval_observation_count,
            evidence_strength_counts=evidence_strength_counts,
            evidence_source_counts=evidence_source_counts,
            missing_source_counts=missing_source_counts,
            deep_validation_count=deep_validation_count,
            deep_validation_ratio=round((deep_validation_count / interaction_count), 4) if interaction_count else 0.0,
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
            interaction_records=interaction_records or [],
            interaction_profile=interaction_profile,
            active_thread=active_thread,
        ).to_dict()

    def active_thread_payload(
        self,
        *,
        repo_root: str,
        session_id: str,
        interaction_profile: dict[str, Any] | None,
        active_thread: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return ActiveThreadReport(
            repo_root=repo_root,
            session_id=session_id,
            interaction_profile=interaction_profile,
            active_thread=active_thread,
        ).to_dict()

    def session_reset_payload(
        self,
        *,
        repo_root: str,
        session_id: str,
        cleared: bool,
        state_path: str,
        interaction_profile: dict[str, Any] | None,
        active_thread: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return SessionResetReport(
            repo_root=repo_root,
            session_id=session_id,
            cleared=cleared,
            state_path=state_path,
            interaction_profile=interaction_profile,
            active_thread=active_thread,
        ).to_dict()

    def session_profile_payload(
        self,
        *,
        repo_root: str,
        session_id: str,
        interaction_profile: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "repo_root": repo_root,
            "session_id": session_id,
            "interaction_profile": interaction_profile,
        }

    def archive_search_payload(
        self,
        *,
        repo_root: str,
        session_id: str | None,
        tool_id: str | None,
        capability: str | None,
        query: str,
        matches: list[dict[str, Any]],
        status: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> dict[str, Any]:
        return RetrievalResultSchema.build_search_payload(
            repo_root=repo_root,
            session_id=session_id,
            tool_id=tool_id,
            capability=capability,
            query=query,
            matches=matches,
            status=status,
            date_from=date_from,
            date_to=date_to,
        )

    def latest_record_payload(
        self,
        *,
        repo_root: str,
        session_id: str | None,
        tool_id: str | None,
        capability: str | None,
        status: str | None,
        record: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return RetrievalResultSchema.build_latest_payload(
            repo_root=repo_root,
            session_id=session_id,
            tool_id=tool_id,
            capability=capability,
            status=status,
            record=record,
        )

    def archive_summary_payload(
        self,
        *,
        repo_root: str,
        session_id: str | None,
        tool_id: str | None,
        capability: str | None,
        date_from: str | None,
        date_to: str | None,
        record_count: int,
        status_counts: dict[str, int],
        tool_counts: dict[str, int],
        capability_counts: dict[str, int],
        target_label_counts: dict[str, int],
        result_quality_counts: dict[str, int],
        latest_by_capability: dict[str, dict[str, Any]],
        recent_records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return RetrievalResultSchema.build_summary_payload(
            repo_root=repo_root,
            session_id=session_id,
            tool_id=tool_id,
            capability=capability,
            date_from=date_from,
            date_to=date_to,
            record_count=record_count,
            status_counts=status_counts,
            tool_counts=tool_counts,
            capability_counts=capability_counts,
            target_label_counts=target_label_counts,
            result_quality_counts=result_quality_counts,
            latest_by_capability=latest_by_capability,
            recent_records=recent_records,
        )

    def archive_compare_payload(
        self,
        *,
        repo_root: str,
        session_id: str | None,
        tool_id: str | None,
        capability: str,
        date_from: str | None,
        date_to: str | None,
        record_count: int,
        target_count: int,
        target_groups: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return RetrievalResultSchema.build_compare_payload(
            repo_root=repo_root,
            session_id=session_id,
            tool_id=tool_id,
            capability=capability,
            date_from=date_from,
            date_to=date_to,
            record_count=record_count,
            target_count=target_count,
            target_groups=target_groups,
        )

    def _render_tool_result_text(self, payload: dict[str, Any]) -> str:
        lines = [
            f"status: {payload.get('status')}",
            f"tool: {payload.get('tool_id')}",
            f"capability: {payload.get('capability')}",
            f"summary: {payload.get('summary')}",
        ]
        if payload.get("run_dir"):
            lines.append(f"run_dir: {payload['run_dir']}")
        if payload.get("archive_path"):
            lines.append(f"archive_path: {payload['archive_path']}")
        if payload.get("error"):
            lines.append(f"error: {payload['error']}")

        artifacts = payload.get("artifacts") or []
        if artifacts:
            lines.append("artifacts:")
            for artifact in artifacts:
                lines.append(f"- {artifact['name']}: {artifact['path']}")

        structured_data = payload.get("structured_data") or {}
        parsed_results = structured_data.get("parsed_results") or {}
        if parsed_results:
            lines.append("results:")
            for key, value in parsed_results.items():
                lines.append(f"- {key}: {value}")
        elif structured_data:
            posture_counts = structured_data.get("posture_counts") or {}
            if structured_data.get("latest_posture"):
                lines.append(f"latest_posture: {structured_data.get('latest_posture')}")
            if structured_data.get("recent_posture_mix"):
                lines.append(f"recent_posture_mix: {structured_data.get('recent_posture_mix')}")
            posture_trend = structured_data.get("posture_trend") or []
            if posture_trend:
                lines.append(f"posture_trend: {' -> '.join(str(item) for item in posture_trend)}")
            if structured_data.get("posture_drift"):
                lines.append(f"posture_drift: {structured_data.get('posture_drift')}")
            if posture_counts:
                lines.append("posture_counts:")
                for key, value in posture_counts.items():
                    lines.append(f"- {key}: {value}")

        return "\n".join(lines)

    def _render_assistant_response_text(self, payload: dict[str, Any]) -> str:
        mode = payload.get("mode")
        suppress_confidence_display = payload.get("discussion_domain") == "belief_tradition"
        internal_scaffold_visible = bool(payload.get("internal_scaffold_visible"))
        user_facing_answer = str(payload.get("user_facing_answer") or "").strip()
        lines = [
            f"mode: {mode}",
            f"kind: {payload.get('kind')}",
            f"summary: {payload.get('summary')}",
        ]
        diagnostic_lines: list[str] = []
        if user_facing_answer:
            lines.append(f"answer: {user_facing_answer}")
        if payload.get("discussion_domain"):
            diagnostic_lines.append(f"discussion_domain: {payload.get('discussion_domain')}")
        if internal_scaffold_visible and payload.get("response_intro"):
            diagnostic_lines.append(f"response_intro: {payload.get('response_intro')}")
        if internal_scaffold_visible and payload.get("response_opening"):
            diagnostic_lines.append(f"response_opening: {payload.get('response_opening')}")
        if payload.get("resolved_prompt"):
            diagnostic_lines.append(f"resolved_prompt: {payload.get('resolved_prompt')}")
        if payload.get("resolution_strategy"):
            diagnostic_lines.append(f"resolution_strategy: {payload.get('resolution_strategy')}")
        if payload.get("resolution_reason"):
            diagnostic_lines.append(f"resolution_reason: {payload.get('resolution_reason')}")
        route = payload.get("route") or {}
        if route:
            if route.get("strength"):
                diagnostic_lines.append(f"route_strength: {route.get('strength')}")
            if route.get("status") or payload.get("route_status"):
                diagnostic_lines.append(
                    "route_status: "
                    f"{display_status_label(route.get('status') or payload.get('route_status'))}"
                )
            diagnostic_lines.append(f"route_reason: {route.get('reason')}")
            if route.get("caution"):
                diagnostic_lines.append(f"route_caution: {route.get('caution')}")
            ambiguity = route.get("ambiguity") or {}
            if ambiguity.get("ambiguous"):
                diagnostic_lines.append(f"route_ambiguity: {ambiguity.get('reason') or 'ambiguous'}")
            decision_summary = route.get("decision_summary") or {}
            alternatives = decision_summary.get("alternatives") or []
            if alternatives:
                diagnostic_lines.append("route_alternatives:")
                for item in alternatives:
                    candidate = item.get("candidate") or {}
                    diagnostic_lines.append(
                        f"- {candidate.get('mode')} | {candidate.get('kind')} | "
                        f"{candidate.get('source')}"
                    )

        if mode == "planning":
            conversation_turn = payload.get("conversation_turn") or {}
            if internal_scaffold_visible and conversation_turn.get("lead"):
                diagnostic_lines.append(f"conversation_turn: {conversation_turn.get('lead')}")
            if internal_scaffold_visible and conversation_turn.get("partner_frame"):
                diagnostic_lines.append(f"conversation_frame: {conversation_turn.get('partner_frame')}")
            if internal_scaffold_visible and conversation_turn.get("next_move"):
                diagnostic_lines.append(f"conversation_next: {conversation_turn.get('next_move')}")
            if payload.get("grounding_strength"):
                lines.append(f"grounding_strength: {payload.get('grounding_strength')}")
            if payload.get("confidence_posture") and not suppress_confidence_display:
                lines.append(
                    f"confidence_posture: {display_status_label(payload.get('confidence_posture'))}"
                )
            if payload.get("support_status"):
                lines.append(
                    f"support_status: {display_status_label(payload.get('support_status'))}"
                )
            if payload.get("tension_status"):
                lines.append(
                    f"tension_status: {display_status_label(payload.get('tension_status'))}"
                )
            if payload.get("closing_strategy"):
                lines.append(f"closing_strategy: {payload.get('closing_strategy')}")
            if payload.get("best_evidence"):
                lines.append(f"best_evidence: {payload.get('best_evidence')}")
            if payload.get("local_context_summary"):
                lines.append(f"local_context_summary: {payload.get('local_context_summary')}")
            if payload.get("grounded_interpretation"):
                lines.append(f"grounded_interpretation: {payload.get('grounded_interpretation')}")
            if payload.get("working_hypothesis"):
                lines.append(f"working_hypothesis: {payload.get('working_hypothesis')}")
            if payload.get("uncertainty_note"):
                lines.append(f"uncertainty_note: {payload.get('uncertainty_note')}")
            reasoning_frame = payload.get("reasoning_frame") or {}
            if reasoning_frame:
                lines.append("reasoning_frame:")
                for key, value in reasoning_frame.items():
                    lines.append(f"- {key}: {value}")
            if payload.get("local_context_assessment"):
                lines.append(f"local_context_assessment: {payload.get('local_context_assessment')}")
            evidence = payload.get("evidence", [])
            if evidence:
                lines.append("evidence:")
                for item in evidence:
                    lines.append(f"- {item}")
            steps = payload.get("steps", [])
            if steps:
                lines.append("steps:")
                for step in steps:
                    lines.append(f"- {step}")
            if payload.get("next_action"):
                lines.append(f"next_action: {payload.get('next_action')}")
            return self._append_diagnostics(lines=lines, diagnostic_lines=diagnostic_lines)

        if mode == "research":
            conversation_turn = payload.get("conversation_turn") or {}
            if internal_scaffold_visible and conversation_turn.get("lead"):
                diagnostic_lines.append(f"conversation_turn: {conversation_turn.get('lead')}")
            if internal_scaffold_visible and conversation_turn.get("partner_frame"):
                diagnostic_lines.append(f"conversation_frame: {conversation_turn.get('partner_frame')}")
            if internal_scaffold_visible and conversation_turn.get("next_move"):
                diagnostic_lines.append(f"conversation_next: {conversation_turn.get('next_move')}")
            if payload.get("grounding_strength"):
                lines.append(f"grounding_strength: {payload.get('grounding_strength')}")
            if payload.get("confidence_posture") and not suppress_confidence_display:
                lines.append(
                    f"confidence_posture: {display_status_label(payload.get('confidence_posture'))}"
                )
            if payload.get("support_status"):
                lines.append(
                    f"support_status: {display_status_label(payload.get('support_status'))}"
                )
            if payload.get("tension_status"):
                lines.append(
                    f"tension_status: {display_status_label(payload.get('tension_status'))}"
                )
            if payload.get("closing_strategy"):
                lines.append(f"closing_strategy: {payload.get('closing_strategy')}")
            if payload.get("best_evidence"):
                lines.append(f"best_evidence: {payload.get('best_evidence')}")
            if payload.get("local_context_summary"):
                lines.append(f"local_context_summary: {payload.get('local_context_summary')}")
            if payload.get("grounded_interpretation"):
                lines.append(f"grounded_interpretation: {payload.get('grounded_interpretation')}")
            if payload.get("working_hypothesis"):
                lines.append(f"working_hypothesis: {payload.get('working_hypothesis')}")
            if payload.get("uncertainty_note"):
                lines.append(f"uncertainty_note: {payload.get('uncertainty_note')}")
            reasoning_frame = payload.get("reasoning_frame") or {}
            if reasoning_frame:
                lines.append("reasoning_frame:")
                for key, value in reasoning_frame.items():
                    lines.append(f"- {key}: {value}")
            if payload.get("local_context_assessment"):
                lines.append(f"local_context_assessment: {payload.get('local_context_assessment')}")
            evidence = payload.get("evidence", [])
            if evidence:
                lines.append("evidence:")
                for item in evidence:
                    lines.append(f"- {item}")
            findings = payload.get("findings", [])
            if findings:
                lines.append("findings:")
                for item in findings:
                    lines.append(f"- {item}")
            if payload.get("recommendation"):
                lines.append(f"recommendation: {payload.get('recommendation')}")
            return self._append_diagnostics(lines=lines, diagnostic_lines=diagnostic_lines)

        if mode == "clarification":
            if payload.get("clarification_question"):
                lines.append(f"clarification_question: {payload.get('clarification_question')}")
            options = payload.get("options", [])
            if options:
                lines.append("options:")
                for option in options:
                    lines.append(f"- {option}")
            clarification_context = payload.get("clarification_context") or {}
            if clarification_context.get("clarification_count") is not None:
                lines.append(
                    f"clarification_count: {clarification_context.get('clarification_count')}"
                )
            if clarification_context.get("recent_clarification_mix"):
                lines.append(
                    "recent_clarification_mix: "
                    f"{clarification_context.get('recent_clarification_mix')}"
                )
            if clarification_context.get("clarification_drift"):
                lines.append(
                    f"clarification_drift: {clarification_context.get('clarification_drift')}"
                )
            if clarification_context.get("clarification_trigger"):
                lines.append(
                    f"clarification_trigger: {clarification_context.get('clarification_trigger')}"
                )
            return self._append_diagnostics(lines=lines, diagnostic_lines=diagnostic_lines)

        if mode == "tool":
            if payload.get("tool_route_origin"):
                lines.append(f"tool_route_origin: {payload.get('tool_route_origin')}")
            tool_result = payload.get("tool_result")
            if tool_result:
                lines.append(self.render_text(self.tool_result_payload(tool_result)))
                return self._append_diagnostics(lines=lines, diagnostic_lines=diagnostic_lines)
        return self._append_diagnostics(lines=lines, diagnostic_lines=diagnostic_lines)

    @staticmethod
    def _append_diagnostics(*, lines: list[str], diagnostic_lines: list[str]) -> str:
        if not diagnostic_lines:
            return "\n".join(lines)
        return "\n".join([*lines, "diagnostics:", *diagnostic_lines])

    def _render_doctor_text(self, payload: dict[str, Any]) -> str:
        lines = [
            f"status: {payload.get('status')}",
            f"repo_root: {payload.get('repo_root')}",
            "checks:",
        ]
        for item in payload.get("checks", []):
            lines.append(f"- {item.get('name')}: {item.get('status')} ({item.get('details')})")
        return "\n".join(lines)

    def _render_records_text(self, payload: dict[str, Any]) -> str:
        lines = [
            f"repo_root: {payload.get('repo_root')}",
            f"session_id: {payload.get('session_id')}",
            f"query: {payload.get('query')}",
            f"status_filter: {payload.get('status_filter')}",
            f"date_from: {payload.get('date_from')}",
            f"date_to: {payload.get('date_to')}",
            f"record_count: {payload.get('record_count')}",
        ]
        records = payload.get("records", [])
        if records:
            lines.append("records:")
            for record in records:
                lines.append(
                    f"- {record.get('created_at', '<unknown>')} | "
                    f"{record.get('tool_id')} | {record.get('capability')} | "
                    f"{record.get('status')} | {record.get('archive_path')}"
                    + (f" | run_id={record.get('run_id')}" if record.get("run_id") else "")
                    + (f" | target={record.get('target_label')}" if record.get("target_label") else "")
                    + (
                        f" | quality={record.get('result_quality')}"
                        if record.get("result_quality")
                        else ""
                    )
                )
        interaction_records = payload.get("interaction_records", [])
        if interaction_records:
            lines.append(f"interaction_count: {payload.get('interaction_count')}")
            if payload.get("clarification_count") is not None:
                lines.append(f"clarification_count: {payload.get('clarification_count')}")
                lines.append(f"clarification_ratio: {payload.get('clarification_ratio')}")
            if payload.get("latest_clarification"):
                lines.append(f"latest_clarification: {payload.get('latest_clarification')}")
            clarification_trend = payload.get("clarification_trend", [])
            if clarification_trend:
                lines.append(f"clarification_trend: {' -> '.join(clarification_trend)}")
            if payload.get("recent_clarification_mix"):
                lines.append(f"recent_clarification_mix: {payload.get('recent_clarification_mix')}")
            if payload.get("clarification_drift"):
                lines.append(f"clarification_drift: {payload.get('clarification_drift')}")
            posture_counts = payload.get("posture_counts", {})
            if posture_counts:
                lines.append("posture_counts:")
                for key, value in posture_counts.items():
                    lines.append(f"- {key}: {value}")
            for label, key in (
                ("detected_language_counts", "detected_language_counts"),
                ("dominant_intent_counts", "dominant_intent_counts"),
                ("tool_route_origin_counts", "tool_route_origin_counts"),
            ):
                counts = payload.get(key, {})
                if counts:
                    lines.append(f"{label}:")
                    for count_key, value in counts.items():
                        lines.append(f"- {count_key}: {value}")
            if payload.get("latest_posture"):
                lines.append(f"latest_posture: {payload.get('latest_posture')}")
            posture_trend = payload.get("posture_trend", [])
            if posture_trend:
                lines.append(f"posture_trend: {' -> '.join(posture_trend)}")
            if payload.get("recent_posture_mix"):
                lines.append(f"recent_posture_mix: {payload.get('recent_posture_mix')}")
            if payload.get("posture_drift"):
                lines.append(f"posture_drift: {payload.get('posture_drift')}")
            retrieval_lead_counts = payload.get("retrieval_lead_counts", {})
            if retrieval_lead_counts:
                lines.append("retrieval_lead_counts:")
                for count_key, value in retrieval_lead_counts.items():
                    lines.append(f"- {count_key}: {value}")
            if payload.get("retrieval_observation_count") is not None:
                lines.append(f"retrieval_observation_count: {payload.get('retrieval_observation_count')}")
            evidence_strength_counts = payload.get("evidence_strength_counts", {})
            if evidence_strength_counts:
                lines.append("evidence_strength_counts:")
                for count_key, value in evidence_strength_counts.items():
                    lines.append(f"- {count_key}: {value}")
            evidence_source_counts = payload.get("evidence_source_counts", {})
            if evidence_source_counts:
                lines.append("evidence_source_counts:")
                for count_key, value in evidence_source_counts.items():
                    lines.append(f"- {count_key}: {value}")
            missing_source_counts = payload.get("missing_source_counts", {})
            if missing_source_counts:
                lines.append("missing_source_counts:")
                for count_key, value in missing_source_counts.items():
                    lines.append(f"- {count_key}: {value}")
            if payload.get("deep_validation_count") is not None:
                lines.append(f"deep_validation_count: {payload.get('deep_validation_count')}")
            if payload.get("deep_validation_ratio") is not None:
                lines.append(f"deep_validation_ratio: {payload.get('deep_validation_ratio')}")
            if payload.get("contradiction_signal_count") is not None:
                lines.append(f"contradiction_signal_count: {payload.get('contradiction_signal_count')}")
            contradiction_flag_counts = payload.get("contradiction_flag_counts", {})
            if contradiction_flag_counts:
                lines.append("contradiction_flag_counts:")
                for count_key, value in contradiction_flag_counts.items():
                    lines.append(f"- {count_key}: {value}")
            memory_classification_counts = payload.get("memory_classification_counts", {})
            if memory_classification_counts:
                lines.append("memory_classification_counts:")
                for count_key, value in memory_classification_counts.items():
                    lines.append(f"- {count_key}: {value}")
            memory_write_action_counts = payload.get("memory_write_action_counts", {})
            if memory_write_action_counts:
                lines.append("memory_write_action_counts:")
                for count_key, value in memory_write_action_counts.items():
                    lines.append(f"- {count_key}: {value}")
            if payload.get("memory_save_eligible_count") is not None:
                lines.append(f"memory_save_eligible_count: {payload.get('memory_save_eligible_count')}")
            if payload.get("explicit_memory_consent_count") is not None:
                lines.append(f"explicit_memory_consent_count: {payload.get('explicit_memory_consent_count')}")
            if payload.get("memory_surface_block_count") is not None:
                lines.append(f"memory_surface_block_count: {payload.get('memory_surface_block_count')}")
            if payload.get("personal_memory_saved_count") is not None:
                lines.append(f"personal_memory_saved_count: {payload.get('personal_memory_saved_count')}")
            if payload.get("research_note_count") is not None:
                lines.append(f"research_note_count: {payload.get('research_note_count')}")
            if payload.get("research_artifact_count") is not None:
                lines.append(f"research_artifact_count: {payload.get('research_artifact_count')}")
            research_artifact_type_counts = payload.get("research_artifact_type_counts", {})
            if research_artifact_type_counts:
                lines.append("research_artifact_type_counts:")
                for count_key, value in research_artifact_type_counts.items():
                    lines.append(f"- {count_key}: {value}")
            recent_topics = payload.get("recent_topics", [])
            if recent_topics:
                lines.append(f"recent_topics: {' | '.join(recent_topics)}")
            lines.append("interactions:")
            for record in interaction_records:
                detail = (
                    f"- {record.get('created_at', '<unknown>')} | "
                    f"{record.get('mode')} | {record.get('kind')} | "
                    f"{self._display_interaction_summary(record)}"
                )
                if record.get("confidence_posture"):
                    detail += f" | posture={self._display_status(record.get('confidence_posture'))}"
                if record.get("route_status"):
                    detail += f" | route_status={self._display_status(record.get('route_status'))}"
                if record.get("support_status"):
                    detail += f" | support_status={self._display_status(record.get('support_status'))}"
                if record.get("tension_status"):
                    detail += f" | tension_status={self._display_status(record.get('tension_status'))}"
                if record.get("local_context_assessment"):
                    detail += f" | context={record.get('local_context_assessment')}"
                if record.get("coherence_topic"):
                    detail += f" | coherence={record.get('coherence_topic')}"
                if float(record.get("route_semantic_bonus") or 0.0) > 0:
                    detail += f" | route_semantic_bonus={record.get('route_semantic_bonus')}"
                if record.get("route_intent_bias"):
                    detail += " | route_intent_bias=true"
                if record.get("route_intent_caution"):
                    detail += " | route_intent_caution=true"
                if record.get("route_retrieval_caution"):
                    detail += " | route_retrieval_caution=true"
                if record.get("deep_validation_used"):
                    detail += " | deep_validation_used=true"
                if record.get("evidence_strength"):
                    detail += f" | evidence_strength={record.get('evidence_strength')}"
                memory_classification = record.get("memory_classification") or {}
                if memory_classification.get("candidate_type"):
                    detail += f" | memory={memory_classification.get('candidate_type')}"
                if memory_classification.get("save_eligible"):
                    detail += " | memory_save_eligible=true"
                if memory_classification.get("requires_explicit_user_consent"):
                    detail += " | memory_requires_consent=true"
                memory_write_decision = record.get("memory_write_decision") or {}
                if memory_write_decision.get("action"):
                    detail += f" | memory_write_action={memory_write_decision.get('action')}"
                if memory_write_decision.get("blocked_by_surface_policy"):
                    detail += " | memory_surface_blocked=true"
                if isinstance(record.get("personal_memory"), dict):
                    detail += " | personal_memory_saved=true"
                if isinstance(record.get("research_note"), dict):
                    detail += " | research_note_saved=true"
                    promoted_artifacts = record["research_note"].get("promoted_artifacts") or []
                    if promoted_artifacts:
                        detail += f" | promoted_artifacts={len(promoted_artifacts)}"
                if record.get("client_surface"):
                    detail += f" | surface={record.get('client_surface')}"
                evidence_sources = record.get("evidence_sources") or []
                if evidence_sources:
                    source_names = [
                        str(item.get("source") or "").strip()
                        for item in evidence_sources
                        if str(item.get("source") or "").strip()
                    ]
                    if source_names:
                        detail += f" | evidence_sources={','.join(source_names[:3])}"
                missing_sources = record.get("missing_sources") or []
                if missing_sources:
                    detail += f" | missing_sources={','.join(str(item) for item in missing_sources)}"
                contradiction_flags = record.get("contradiction_flags") or []
                if contradiction_flags:
                    detail += f" | contradictions={','.join(str(item) for item in contradiction_flags)}"
                if record.get("tool_route_origin"):
                    detail += f" | tool_route_origin={record.get('tool_route_origin')}"
                detail += self._interaction_prompt_suffix(record)
                if record.get("resolution_strategy"):
                    detail += f" | resolution={record.get('resolution_strategy')}"
                lines.append(detail)
        active_thread = payload.get("active_thread")
        if active_thread:
            lines.append("active_thread:")
            lines.append(
                f"- {active_thread.get('mode')} | {active_thread.get('kind')} | "
                f"{active_thread.get('prompt')}"
            )
            if active_thread.get("original_prompt"):
                lines.append(f"original_prompt: {active_thread.get('original_prompt')}")
            if active_thread.get("confidence_posture"):
                lines.append(f"confidence_posture: {self._display_status(active_thread.get('confidence_posture'))}")
            if active_thread.get("route_status"):
                lines.append(f"route_status: {self._display_status(active_thread.get('route_status'))}")
            if active_thread.get("support_status"):
                lines.append(f"support_status: {self._display_status(active_thread.get('support_status'))}")
            if active_thread.get("tension_status"):
                lines.append(f"tension_status: {self._display_status(active_thread.get('tension_status'))}")
            if active_thread.get("local_context_assessment"):
                lines.append(f"local_context_assessment: {active_thread.get('local_context_assessment')}")
            if active_thread.get("coherence_topic"):
                lines.append(f"coherence_topic: {active_thread.get('coherence_topic')}")
            if active_thread.get("tool_route_origin"):
                lines.append(f"tool_route_origin: {active_thread.get('tool_route_origin')}")
            if active_thread.get("objective"):
                lines.append(f"active_objective: {active_thread.get('objective')}")
            if active_thread.get("thread_summary"):
                lines.append(f"thread_summary: {active_thread.get('thread_summary')}")
            self._append_tool_context_lines(lines, active_thread)
        self._append_interaction_profile_lines(lines, payload.get("interaction_profile"))
        return "\n".join(lines)

    def _render_active_thread_text(self, payload: dict[str, Any]) -> str:
        lines = [
            f"repo_root: {payload.get('repo_root')}",
            f"session_id: {payload.get('session_id')}",
        ]
        active_thread = payload.get("active_thread")
        if not active_thread:
            lines.append("active_thread: <none>")
            self._append_interaction_profile_lines(lines, payload.get("interaction_profile"))
            return "\n".join(lines)

        lines.append("active_thread:")
        lines.append(
            f"- {active_thread.get('mode')} | {active_thread.get('kind')} | "
            f"{active_thread.get('prompt')}"
        )
        if active_thread.get("original_prompt"):
            lines.append(f"original_prompt: {active_thread.get('original_prompt')}")
        if active_thread.get("confidence_posture"):
            lines.append(f"confidence_posture: {self._display_status(active_thread.get('confidence_posture'))}")
        if active_thread.get("route_status"):
            lines.append(f"route_status: {self._display_status(active_thread.get('route_status'))}")
        if active_thread.get("support_status"):
            lines.append(f"support_status: {self._display_status(active_thread.get('support_status'))}")
        if active_thread.get("tension_status"):
            lines.append(f"tension_status: {self._display_status(active_thread.get('tension_status'))}")
        if active_thread.get("local_context_assessment"):
            lines.append(f"local_context_assessment: {active_thread.get('local_context_assessment')}")
        if active_thread.get("coherence_topic"):
            lines.append(f"coherence_topic: {active_thread.get('coherence_topic')}")
        if active_thread.get("tool_route_origin"):
            lines.append(f"tool_route_origin: {active_thread.get('tool_route_origin')}")
        if active_thread.get("objective"):
            lines.append(f"active_objective: {active_thread.get('objective')}")
        if active_thread.get("thread_summary"):
            lines.append(f"thread_summary: {active_thread.get('thread_summary')}")
        self._append_tool_context_lines(lines, active_thread)
        self._append_interaction_profile_lines(lines, payload.get("interaction_profile"))
        return "\n".join(lines)

    def _render_session_reset_text(self, payload: dict[str, Any]) -> str:
        lines = [
            f"repo_root: {payload.get('repo_root')}",
            f"session_id: {payload.get('session_id')}",
            f"cleared: {payload.get('cleared')}",
            f"state_path: {payload.get('state_path')}",
        ]
        lines.append("active_thread: <none>")
        self._append_interaction_profile_lines(lines, payload.get("interaction_profile"))
        return "\n".join(lines)

    def _render_session_profile_text(self, payload: dict[str, Any]) -> str:
        lines = [
            f"repo_root: {payload.get('repo_root')}",
            f"session_id: {payload.get('session_id')}",
        ]
        self._append_interaction_profile_lines(lines, payload.get("interaction_profile"))
        return "\n".join(lines)

    def _render_interactions_text(self, payload: dict[str, Any]) -> str:
        lines = [
            f"repo_root: {payload.get('repo_root')}",
            f"session_id: {payload.get('session_id')}",
            f"resolution_strategy: {payload.get('resolution_strategy')}",
            f"interaction_count: {payload.get('interaction_count')}",
        ]
        records = payload.get("interaction_records", [])
        if records:
            lines.append("interactions:")
            for record in records:
                detail = (
                    f"- {record.get('created_at', '<unknown>')} | "
                    f"{record.get('mode')} | {record.get('kind')} | "
                    f"{self._display_interaction_summary(record)}"
                )
                if record.get("confidence_posture"):
                    detail += f" | posture={self._display_status(record.get('confidence_posture'))}"
                if record.get("local_context_assessment"):
                    detail += f" | context={record.get('local_context_assessment')}"
                if record.get("coherence_topic"):
                    detail += f" | coherence={record.get('coherence_topic')}"
                if float(record.get("route_semantic_bonus") or 0.0) > 0:
                    detail += f" | route_semantic_bonus={record.get('route_semantic_bonus')}"
                if record.get("tool_route_origin"):
                    detail += f" | tool_route_origin={record.get('tool_route_origin')}"
                detail += self._interaction_prompt_suffix(record)
                if record.get("resolution_strategy"):
                    detail += f" | resolution={record.get('resolution_strategy')}"
                lines.append(detail)
        return "\n".join(lines)

    def _render_interaction_search_text(self, payload: dict[str, Any]) -> str:
        lines = [
            f"repo_root: {payload.get('repo_root')}",
            f"session_id: {payload.get('session_id')}",
            f"query: {payload.get('query')}",
            f"resolution_strategy: {payload.get('resolution_strategy')}",
            f"interaction_count: {payload.get('interaction_count')}",
        ]
        matches = payload.get("matches", [])
        if matches:
            lines.append("matches:")
            for match in matches:
                record = match.get("record", {})
                detail = (
                    f"- score={match.get('score')} | "
                    f"{record.get('created_at', '<unknown>')} | "
                    f"{record.get('mode')} | {record.get('kind')} | "
                    f"{record.get('prompt')}"
                )
                score_breakdown = match.get("score_breakdown") or {}
                if score_breakdown:
                    detail += (
                        " | keyword="
                        f"{score_breakdown.get('keyword_score', 0)}"
                        " | semantic="
                        f"{score_breakdown.get('semantic_score', 0)}"
                    )
                resolved_prompt = str(record.get("resolved_prompt") or "").strip()
                if resolved_prompt:
                    detail += f" | resolved={resolved_prompt}"
                if record.get("resolution_strategy"):
                    detail += f" | resolution={record.get('resolution_strategy')}"
                lines.append(detail)
        return "\n".join(lines)

    def _render_interaction_evaluation_text(self, payload: dict[str, Any]) -> str:
        lines = [
            f"repo_root: {payload.get('repo_root')}",
            f"session_id: {payload.get('session_id')}",
            f"evaluated_count: {payload.get('evaluated_count')}",
        ]
        judgment_counts = payload.get("judgment_counts") or {}
        if judgment_counts:
            lines.append("judgment_counts:")
            for key, value in judgment_counts.items():
                lines.append(f"- {key}: {value}")
        surface_aggregates = payload.get("surface_aggregates") or []
        if surface_aggregates:
            lines.append("surface_aggregates:")
            for aggregate in surface_aggregates:
                lines.append(f"- surface: {aggregate.get('surface')}")
                lines.append(f"  reviewed_count: {aggregate.get('reviewed_count')}")
                if aggregate.get("average_score") is not None:
                    lines.append(f"  average_score: {aggregate.get('average_score')}")
                counts = aggregate.get("judgment_counts") or {}
                for key, value in counts.items():
                    lines.append(f"  {key}: {value}")
        evaluations = payload.get("evaluations") or []
        if evaluations:
            lines.append("evaluations:")
            for evaluation in evaluations:
                lines.append(f"- overall_judgment: {evaluation.get('overall_judgment')}")
                if evaluation.get("mode"):
                    lines.append(f"  mode: {evaluation.get('mode')}")
                if evaluation.get("kind"):
                    lines.append(f"  kind: {evaluation.get('kind')}")
                if evaluation.get("interaction_path"):
                    lines.append(f"  interaction_path: {evaluation.get('interaction_path')}")
                for review in evaluation.get("surface_reviews") or []:
                    lines.append(f"  {review.get('surface')}: {review.get('judgment')}")
        return "\n".join(lines)

    def _render_labeled_dataset_export_text(self, payload: dict[str, Any]) -> str:
        lines = [
            f"repo_root: {payload.get('repo_root')}",
            f"session_id: {payload.get('session_id')}",
            f"dataset_path: {payload.get('dataset_path')}",
            f"example_count: {payload.get('example_count')}",
        ]
        category_counts = payload.get("label_category_counts") or {}
        if category_counts:
            lines.append("label_category_counts:")
            for key, value in category_counts.items():
                lines.append(f"- {key}: {value}")
        value_counts = payload.get("label_value_counts") or {}
        if value_counts:
            lines.append("label_value_counts:")
            for key, value in value_counts.items():
                lines.append(f"- {key}: {value}")
        return "\n".join(lines)

    def _render_dataset_jsonl_export_text(self, payload: dict[str, Any]) -> str:
        lines = [
            f"status: {payload.get('status')}",
            f"dataset_name: {payload.get('dataset_name')}",
            f"export_name: {payload.get('export_name')}",
            f"export_dir: {payload.get('export_dir')}",
            f"example_count: {payload.get('example_count')}",
        ]
        split_counts = payload.get("split_counts") or {}
        if split_counts:
            lines.append("split_counts:")
            for key, value in split_counts.items():
                lines.append(f"- {key}: {value}")
        quality = payload.get("quality_report") or {}
        if quality:
            lines.append(f"duplicate_group_count: {quality.get('duplicate_group_count')}")
            lines.append(f"split_leakage_count: {quality.get('split_leakage_count')}")
            lines.append(f"missing_canonical_label_count: {quality.get('missing_canonical_label_count')}")
        return "\n".join(lines)

    def _render_dataset_review_batch_text(self, payload: dict[str, Any]) -> str:
        lines = [
            f"status: {payload.get('status')}",
            f"dataset_name: {payload.get('dataset_name')}",
            f"import_run_id: {payload.get('import_run_id')}",
            f"review_count: {payload.get('review_count')}",
            f"prioritize: {payload.get('prioritize')}",
        ]
        examples = payload.get("examples") or []
        if examples:
            lines.append("examples:")
            for example in examples[:10]:
                lines.append(
                    f"- {example.get('id')} | {example.get('example_type')} | split={example.get('split_assignment')} | label_source={example.get('label_source')}"
                )
        if payload.get("output_path"):
            lines.append(f"output_path: {payload.get('output_path')}")
        return "\n".join(lines)

    def _render_dataset_run_comparison_text(self, payload: dict[str, Any]) -> str:
        left = payload.get("left_run") or {}
        right = payload.get("right_run") or {}
        overlap = payload.get("overlap") or {}
        lines = [
            f"status: {payload.get('status')}",
            f"left_run: {left.get('id')} ({left.get('dataset_name')})",
            f"right_run: {right.get('id')} ({right.get('dataset_name')})",
            f"left_example_count: {left.get('example_count')}",
            f"right_example_count: {right.get('example_count')}",
            f"shared_example_signature_count: {overlap.get('shared_example_signature_count')}",
            f"shared_ratio_vs_smaller_run: {overlap.get('shared_ratio_vs_smaller_run')}",
        ]
        return "\n".join(lines)

    def _render_dataset_label_update_text(self, payload: dict[str, Any]) -> str:
        label = payload.get("label") or {}
        lines = [
            f"status: {payload.get('status')}",
            f"dataset_example_id: {payload.get('dataset_example_id')}",
            f"label_role: {label.get('label_role')}",
            f"label_value: {label.get('label_value')}",
            f"is_canonical: {label.get('is_canonical')}",
        ]
        return "\n".join(lines)

    def _render_dataset_example_update_text(self, payload: dict[str, Any]) -> str:
        example = payload.get("example") or {}
        lines = [
            f"status: {payload.get('status')}",
            f"example_id: {example.get('id')}",
            f"trainable: {example.get('trainable')}",
            f"ingestion_state: {example.get('ingestion_state')}",
            f"split_assignment: {example.get('split_assignment')}",
            f"label_source: {example.get('label_source')}",
        ]
        return "\n".join(lines)

    def _render_research_notes_text(self, payload: dict[str, Any]) -> str:
        lines = [
            f"repo_root: {payload.get('repo_root')}",
            f"session_id: {payload.get('session_id')}",
            f"note_count: {payload.get('note_count')}",
        ]
        notes = payload.get("research_notes", [])
        if notes:
            lines.append("research_notes:")
            for note in notes:
                detail = (
                    f"- {note.get('created_at', '<unknown>')} | "
                    f"{note.get('note_type')} | {note.get('title')} | "
                    f"{note.get('note_path')}"
                )
                promoted = note.get("promoted_artifacts") or []
                if promoted:
                    detail += f" | promoted={len(promoted)}"
                lines.append(detail)
        return "\n".join(lines)

    def _render_research_artifacts_text(self, payload: dict[str, Any]) -> str:
        lines = [
            f"repo_root: {payload.get('repo_root')}",
            f"session_id: {payload.get('session_id')}",
            f"artifact_count: {payload.get('artifact_count')}",
        ]
        artifacts = payload.get("research_artifacts", [])
        if artifacts:
            lines.append("research_artifacts:")
            for artifact in artifacts:
                lines.append(
                    f"- {artifact.get('created_at', '<unknown>')} | "
                    f"{artifact.get('artifact_type')} | {artifact.get('title')} | "
                    f"{artifact.get('artifact_path')}"
                )
        return "\n".join(lines)

    def _render_research_artifact_promotion_text(self, payload: dict[str, Any]) -> str:
        artifact = payload.get("artifact") or {}
        lines = [
            f"status: {payload.get('status')}",
            f"repo_root: {payload.get('repo_root')}",
            f"artifact_type: {artifact.get('artifact_type')}",
            f"title: {artifact.get('title')}",
            f"artifact_path: {artifact.get('artifact_path')}",
            f"source_note_path: {artifact.get('source_note_path')}",
        ]
        if artifact.get("promotion_reason"):
            lines.append(f"promotion_reason: {artifact.get('promotion_reason')}")
        return "\n".join(lines)

    def _render_interaction_summary_text(self, payload: dict[str, Any]) -> str:
        lines = [
            f"repo_root: {payload.get('repo_root')}",
            f"session_id: {payload.get('session_id')}",
            f"interaction_count: {payload.get('interaction_count')}",
        ]
        if payload.get("clarification_count") is not None:
            lines.append(f"clarification_count: {payload.get('clarification_count')}")
            lines.append(f"clarification_ratio: {payload.get('clarification_ratio')}")
        if payload.get("latest_clarification"):
            lines.append(f"latest_clarification: {payload.get('latest_clarification')}")
        clarification_trend = payload.get("clarification_trend", [])
        if clarification_trend:
            lines.append(f"clarification_trend: {' -> '.join(clarification_trend)}")
        if payload.get("recent_clarification_mix"):
            lines.append(f"recent_clarification_mix: {payload.get('recent_clarification_mix')}")
        if payload.get("clarification_drift"):
            lines.append(f"clarification_drift: {payload.get('clarification_drift')}")
        for label, key in (
            ("mode_counts", "mode_counts"),
            ("kind_counts", "kind_counts"),
            ("posture_counts", "posture_counts"),
            ("detected_language_counts", "detected_language_counts"),
            ("dominant_intent_counts", "dominant_intent_counts"),
            ("local_context_assessment_counts", "local_context_assessment_counts"),
            ("tool_route_origin_counts", "tool_route_origin_counts"),
            ("resolution_counts", "resolution_counts"),
        ):
            counts = payload.get(key, {})
            if counts:
                lines.append(f"{label}:")
                for count_key, value in counts.items():
                    lines.append(f"- {count_key}: {value}")
        if payload.get("latest_posture"):
            lines.append(f"latest_posture: {payload.get('latest_posture')}")
        posture_trend = payload.get("posture_trend", [])
        if posture_trend:
            lines.append(f"posture_trend: {' -> '.join(posture_trend)}")
        if payload.get("recent_posture_mix"):
            lines.append(f"recent_posture_mix: {payload.get('recent_posture_mix')}")
        if payload.get("posture_drift"):
            lines.append(f"posture_drift: {payload.get('posture_drift')}")
        retrieval_lead_counts = payload.get("retrieval_lead_counts", {})
        if retrieval_lead_counts:
            lines.append("retrieval_lead_counts:")
            for count_key, value in retrieval_lead_counts.items():
                lines.append(f"- {count_key}: {value}")
        if payload.get("retrieval_observation_count") is not None:
            lines.append(f"retrieval_observation_count: {payload.get('retrieval_observation_count')}")
        evidence_strength_counts = payload.get("evidence_strength_counts", {})
        if evidence_strength_counts:
            lines.append("evidence_strength_counts:")
            for count_key, value in evidence_strength_counts.items():
                lines.append(f"- {count_key}: {value}")
        evidence_source_counts = payload.get("evidence_source_counts", {})
        if evidence_source_counts:
            lines.append("evidence_source_counts:")
            for count_key, value in evidence_source_counts.items():
                lines.append(f"- {count_key}: {value}")
        missing_source_counts = payload.get("missing_source_counts", {})
        if missing_source_counts:
            lines.append("missing_source_counts:")
            for count_key, value in missing_source_counts.items():
                lines.append(f"- {count_key}: {value}")
        if payload.get("deep_validation_count") is not None:
            lines.append(f"deep_validation_count: {payload.get('deep_validation_count')}")
        if payload.get("deep_validation_ratio") is not None:
            lines.append(f"deep_validation_ratio: {payload.get('deep_validation_ratio')}")
        if payload.get("contradiction_signal_count") is not None:
            lines.append(f"contradiction_signal_count: {payload.get('contradiction_signal_count')}")
        contradiction_flag_counts = payload.get("contradiction_flag_counts", {})
        if contradiction_flag_counts:
            lines.append("contradiction_flag_counts:")
            for count_key, value in contradiction_flag_counts.items():
                lines.append(f"- {count_key}: {value}")
        memory_classification_counts = payload.get("memory_classification_counts", {})
        if memory_classification_counts:
            lines.append("memory_classification_counts:")
            for count_key, value in memory_classification_counts.items():
                lines.append(f"- {count_key}: {value}")
        memory_write_action_counts = payload.get("memory_write_action_counts", {})
        if memory_write_action_counts:
            lines.append("memory_write_action_counts:")
            for count_key, value in memory_write_action_counts.items():
                lines.append(f"- {count_key}: {value}")
        if payload.get("memory_save_eligible_count") is not None:
            lines.append(f"memory_save_eligible_count: {payload.get('memory_save_eligible_count')}")
        if payload.get("explicit_memory_consent_count") is not None:
            lines.append(f"explicit_memory_consent_count: {payload.get('explicit_memory_consent_count')}")
        if payload.get("memory_surface_block_count") is not None:
            lines.append(f"memory_surface_block_count: {payload.get('memory_surface_block_count')}")
        if payload.get("personal_memory_saved_count") is not None:
            lines.append(f"personal_memory_saved_count: {payload.get('personal_memory_saved_count')}")
        if payload.get("research_note_count") is not None:
            lines.append(f"research_note_count: {payload.get('research_note_count')}")
        if payload.get("research_artifact_count") is not None:
            lines.append(f"research_artifact_count: {payload.get('research_artifact_count')}")
        research_artifact_type_counts = payload.get("research_artifact_type_counts", {})
        if research_artifact_type_counts:
            lines.append("research_artifact_type_counts:")
            for count_key, value in research_artifact_type_counts.items():
                lines.append(f"- {count_key}: {value}")
        recent_topics = payload.get("recent_topics", [])
        if recent_topics:
            lines.append(f"recent_topics: {' | '.join(recent_topics)}")
        if payload.get("coherence_topic_count") is not None:
            lines.append(f"coherence_topic_count: {payload.get('coherence_topic_count')}")
        if payload.get("semantic_route_count") is not None:
            lines.append(f"semantic_route_count: {payload.get('semantic_route_count')}")
        if payload.get("semantic_route_ratio") is not None:
            lines.append(f"semantic_route_ratio: {payload.get('semantic_route_ratio')}")
        if payload.get("route_intent_bias_count") is not None:
            lines.append(f"route_intent_bias_count: {payload.get('route_intent_bias_count')}")
        if payload.get("route_intent_bias_ratio") is not None:
            lines.append(f"route_intent_bias_ratio: {payload.get('route_intent_bias_ratio')}")
        if payload.get("route_intent_caution_count") is not None:
            lines.append(f"route_intent_caution_count: {payload.get('route_intent_caution_count')}")
        if payload.get("route_intent_caution_ratio") is not None:
            lines.append(f"route_intent_caution_ratio: {payload.get('route_intent_caution_ratio')}")
        if payload.get("retrieval_route_caution_count") is not None:
            lines.append(f"retrieval_route_caution_count: {payload.get('retrieval_route_caution_count')}")
        if payload.get("retrieval_route_caution_ratio") is not None:
            lines.append(f"retrieval_route_caution_ratio: {payload.get('retrieval_route_caution_ratio')}")
        recent = payload.get("recent_interactions", [])
        if recent:
            lines.append("recent_interactions:")
            for record in recent:
                detail = (
                    f"- {record.get('created_at', '<unknown>')} | "
                    f"{record.get('mode')} | {record.get('kind')} | {self._display_interaction_summary(record)}"
                )
                if record.get("confidence_posture"):
                    detail += f" | posture={self._display_status(record.get('confidence_posture'))}"
                if record.get("route_status"):
                    detail += f" | route_status={self._display_status(record.get('route_status'))}"
                if record.get("support_status"):
                    detail += f" | support_status={self._display_status(record.get('support_status'))}"
                if record.get("tension_status"):
                    detail += f" | tension_status={self._display_status(record.get('tension_status'))}"
                if record.get("local_context_assessment"):
                    detail += f" | context={record.get('local_context_assessment')}"
                if record.get("coherence_topic"):
                    detail += f" | coherence={record.get('coherence_topic')}"
                if float(record.get("route_semantic_bonus") or 0.0) > 0:
                    detail += f" | route_semantic_bonus={record.get('route_semantic_bonus')}"
                if record.get("route_intent_bias"):
                    detail += " | route_intent_bias=true"
                if record.get("route_intent_caution"):
                    detail += " | route_intent_caution=true"
                if record.get("route_retrieval_caution"):
                    detail += " | route_retrieval_caution=true"
                if record.get("deep_validation_used"):
                    detail += " | deep_validation_used=true"
                if record.get("evidence_strength"):
                    detail += f" | evidence_strength={record.get('evidence_strength')}"
                memory_classification = record.get("memory_classification") or {}
                if memory_classification.get("candidate_type"):
                    detail += f" | memory={memory_classification.get('candidate_type')}"
                if memory_classification.get("save_eligible"):
                    detail += " | memory_save_eligible=true"
                if memory_classification.get("requires_explicit_user_consent"):
                    detail += " | memory_requires_consent=true"
                memory_write_decision = record.get("memory_write_decision") or {}
                if memory_write_decision.get("action"):
                    detail += f" | memory_write_action={memory_write_decision.get('action')}"
                if memory_write_decision.get("blocked_by_surface_policy"):
                    detail += " | memory_surface_blocked=true"
                if isinstance(record.get("personal_memory"), dict):
                    detail += " | personal_memory_saved=true"
                if isinstance(record.get("research_note"), dict):
                    detail += " | research_note_saved=true"
                    promoted_artifacts = record["research_note"].get("promoted_artifacts") or []
                    if promoted_artifacts:
                        detail += f" | promoted_artifacts={len(promoted_artifacts)}"
                if record.get("client_surface"):
                    detail += f" | surface={record.get('client_surface')}"
                contradiction_flags = record.get("contradiction_flags") or []
                if contradiction_flags:
                    detail += f" | contradictions={','.join(str(item) for item in contradiction_flags)}"
                if record.get("tool_route_origin"):
                    detail += f" | tool_route_origin={record.get('tool_route_origin')}"
                detail += self._interaction_prompt_suffix(record)
                if record.get("resolution_strategy"):
                    detail += f" | resolution={record.get('resolution_strategy')}"
                lines.append(detail)
        return "\n".join(lines)

    def _render_interaction_patterns_text(self, payload: dict[str, Any]) -> str:
        lines = [
            f"repo_root: {payload.get('repo_root')}",
            f"session_id: {payload.get('session_id')}",
            f"status: {payload.get('status')}",
            f"interaction_count: {payload.get('interaction_count')}",
            f"follow_up_count: {payload.get('follow_up_count')}",
            f"ambiguous_follow_up_count: {payload.get('ambiguous_follow_up_count')}",
            f"rewrite_ratio: {payload.get('rewrite_ratio')}",
            f"follow_up_ratio: {payload.get('follow_up_ratio')}",
            f"ambiguity_ratio: {payload.get('ambiguity_ratio')}",
        ]
        resolution_counts = payload.get("resolution_counts", {})
        if resolution_counts:
            lines.append("resolution_counts:")
            for key, value in resolution_counts.items():
                lines.append(f"- {key}: {value}")
        observations = payload.get("observations", [])
        if observations:
            lines.append("observations:")
            for observation in observations:
                lines.append(f"- {observation}")
        recent = payload.get("recent_interactions", [])
        if recent:
            lines.append("recent_interactions:")
            for record in recent:
                detail = (
                    f"- {record.get('created_at', '<unknown>')} | "
                    f"{record.get('mode')} | {record.get('kind')} | {self._display_interaction_summary(record)}"
                )
                if record.get("confidence_posture"):
                    detail += f" | posture={self._display_status(record.get('confidence_posture'))}"
                if record.get("route_status"):
                    detail += f" | route_status={self._display_status(record.get('route_status'))}"
                if record.get("support_status"):
                    detail += f" | support_status={self._display_status(record.get('support_status'))}"
                if record.get("tension_status"):
                    detail += f" | tension_status={self._display_status(record.get('tension_status'))}"
                if record.get("tool_route_origin"):
                    detail += f" | tool_route_origin={record.get('tool_route_origin')}"
                detail += self._interaction_prompt_suffix(record)
                if record.get("resolution_strategy"):
                    detail += f" | resolution={record.get('resolution_strategy')}"
                lines.append(detail)
        return "\n".join(lines)

    @staticmethod
    def _display_interaction_summary(record: dict[str, Any]) -> str:
        summary = str(record.get("summary") or "").strip()
        mode = str(record.get("mode") or "").strip()
        summary_lower = summary.lower()
        if mode == "planning" and "planning response" in summary_lower and "for:" in summary_lower:
            return "Planning response"
        if mode == "research" and "research response" in summary_lower and "for:" in summary_lower:
            return "Research response"
        return summary

    @staticmethod
    def _interaction_prompt_suffix(record: dict[str, Any]) -> str:
        resolved_prompt = str(record.get("resolved_prompt") or "").strip()
        original_prompt = str(record.get("prompt") or "").strip()
        if resolved_prompt:
            return f" | prompt={resolved_prompt} | original={original_prompt}"
        return ""

    @staticmethod
    def _display_status(value: object) -> str:
        return display_status_label(None if value is None else str(value)) or ""

    @staticmethod
    def _append_tool_context_lines(lines: list[str], active_thread: dict[str, Any]) -> None:
        tool_context = active_thread.get("tool_context") or {}
        if not tool_context:
            return
        tool_id = tool_context.get("tool_id")
        capability = tool_context.get("capability")
        input_path = tool_context.get("input_path")
        params = tool_context.get("params")

        if tool_id or capability:
            lines.append(
                "active_tool: "
                f"{tool_id or '<unknown>'}"
                + (f".{capability}" if capability else "")
            )
        if input_path:
            lines.append(f"active_tool_input: {input_path}")
        if params:
            lines.append(f"active_tool_params: {json.dumps(params, sort_keys=True)}")

    @staticmethod
    def _append_interaction_profile_lines(
        lines: list[str],
        interaction_profile: dict[str, Any] | None,
    ) -> None:
        if not isinstance(interaction_profile, dict) or not interaction_profile:
            return
        lines.append(f"interaction_style: {interaction_profile.get('interaction_style')}")
        lines.append(f"reasoning_depth: {interaction_profile.get('reasoning_depth')}")
        lines.append(f"profile_selection_source: {interaction_profile.get('selection_source')}")
        lines.append(f"allow_suggestions: {interaction_profile.get('allow_suggestions')}")

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

    def _render_error_text(self, payload: dict[str, Any]) -> str:
        lines = [
            f"status: {payload.get('status')}",
            f"error_type: {payload.get('error_type')}",
            f"message: {payload.get('message')}",
        ]
        available_tools = payload.get("available_tools") or {}
        if available_tools:
            lines.append("available_tools:")
            for tool_id, capabilities in available_tools.items():
                lines.append(f"- {tool_id}: {', '.join(capabilities)}")
        return "\n".join(lines)

    def _render_tools_text(self, payload: dict[str, Any]) -> str:
        lines = ["tools:"]
        for tool_id, capabilities in payload.get("tools", {}).items():
            lines.append(f"- {tool_id}: {', '.join(capabilities)}")
        return "\n".join(lines)

    def _render_capabilities_text(self, payload: dict[str, Any]) -> str:
        lines = ["capabilities:"]
        for capability_key, spec in payload.items():
            lines.append(
                f"- {capability_key}: {spec.get('tool_id')}.{spec.get('tool_capability')} "
                f"({spec.get('description')})"
            )
        return "\n".join(lines)

    def _render_init_text(self, payload: dict[str, Any]) -> str:
        lines = [
            f"status: {payload.get('status')}",
            f"repo_root: {payload.get('repo_root')}",
        ]
        created = payload.get("created_paths") or []
        existing = payload.get("existing_paths") or []
        if created:
            lines.append("created:")
            for item in created:
                lines.append(f"- {item}")
        if existing:
            lines.append("already_present:")
            for item in existing:
                lines.append(f"- {item}")
        if payload.get("config_path"):
            lines.append(f"config_path: {payload.get('config_path')}")
        return "\n".join(lines)

    def _render_bundle_text(self, payload: dict[str, Any]) -> str:
        lines = [
            f"bundle_id: {payload.get('bundle_id')}",
            f"name: {payload.get('name')}",
            f"version: {payload.get('version')}",
            f"schema_version: {payload.get('bundle_schema_version')}",
            f"manifest_path: {payload.get('manifest_path')}",
        ]
        description = payload.get("description")
        if description:
            lines.append(f"description: {description}")

        capabilities = payload.get("capabilities") or []
        if capabilities:
            lines.append("capabilities:")
            for capability in capabilities:
                detail = f"- {capability.get('id')}"
                if capability.get("app_capability_key"):
                    detail += f" | app={capability.get('app_capability_key')}"
                aliases = capability.get("command_aliases") or []
                if aliases:
                    detail += f" | aliases={', '.join(aliases)}"
                keywords = capability.get("trigger_keywords") or []
                if keywords:
                    detail += f" | keywords={', '.join(keywords)}"
                lines.append(detail)
        return "\n".join(lines)

    def _render_search_text(self, payload: dict[str, Any]) -> str:
        lines = [
            f"repo_root: {payload.get('repo_root')}",
            f"session_id: {payload.get('session_id')}",
            f"query: {payload.get('query')}",
            f"record_count: {payload.get('record_count')}",
        ]
        matches = payload.get("matches", [])
        if matches:
            lines.append("matches:")
            for match in matches:
                record = match.get("record", {})
                score_breakdown = match.get("score_breakdown") or {}
                breakdown_text = ""
                if score_breakdown:
                    breakdown_text = (
                        " | keyword="
                        f"{score_breakdown.get('keyword_score', 0)}"
                        " | semantic="
                        f"{score_breakdown.get('semantic_score', 0)}"
                    )
                lines.append(
                    f"- score={match.get('score')} | fields={', '.join(match.get('matched_fields', []))} | "
                    f"{record.get('tool_id')} | {record.get('capability')} | {record.get('summary')}"
                    + (f" | run_id={record.get('run_id')}" if record.get("run_id") else "")
                    + (f" | target={record.get('target_label')}" if record.get("target_label") else "")
                    + (
                        f" | quality={record.get('result_quality')}"
                        if record.get("result_quality")
                        else ""
                    )
                    + f"{breakdown_text}"
                )
        return "\n".join(lines)

    def _render_latest_text(self, payload: dict[str, Any]) -> str:
        lines = [
            f"repo_root: {payload.get('repo_root')}",
            f"session_id: {payload.get('session_id')}",
            f"tool_id: {payload.get('tool_id')}",
            f"capability: {payload.get('capability')}",
            f"status_filter: {payload.get('status_filter')}",
            f"found: {payload.get('found')}",
        ]
        record = payload.get("record")
        if record:
            lines.append("record:")
            lines.append(
                f"- {record.get('created_at', '<unknown>')} | {record.get('tool_id')} | "
                f"{record.get('capability')} | {record.get('status')} | {record.get('summary')}"
                + (f" | run_id={record.get('run_id')}" if record.get("run_id") else "")
                + (f" | target={record.get('target_label')}" if record.get("target_label") else "")
                + (
                    f" | quality={record.get('result_quality')}"
                    if record.get("result_quality")
                    else ""
                )
            )
        return "\n".join(lines)

    def _render_archive_summary_text(self, payload: dict[str, Any]) -> str:
        lines = [
            f"repo_root: {payload.get('repo_root')}",
            f"session_id: {payload.get('session_id')}",
            f"tool_id: {payload.get('tool_id')}",
            f"capability: {payload.get('capability')}",
            f"date_from: {payload.get('date_from')}",
            f"date_to: {payload.get('date_to')}",
            f"record_count: {payload.get('record_count')}",
        ]

        status_counts = payload.get("status_counts", {})
        if status_counts:
            lines.append("status_counts:")
            for key, value in status_counts.items():
                lines.append(f"- {key}: {value}")

        tool_counts = payload.get("tool_counts", {})
        if tool_counts:
            lines.append("tool_counts:")
            for key, value in tool_counts.items():
                lines.append(f"- {key}: {value}")

        capability_counts = payload.get("capability_counts", {})
        if capability_counts:
            lines.append("capability_counts:")
            for key, value in capability_counts.items():
                lines.append(f"- {key}: {value}")

        target_label_counts = payload.get("target_label_counts", {})
        if target_label_counts:
            lines.append("target_label_counts:")
            for key, value in target_label_counts.items():
                lines.append(f"- {key}: {value}")

        result_quality_counts = payload.get("result_quality_counts", {})
        if result_quality_counts:
            lines.append("result_quality_counts:")
            for key, value in result_quality_counts.items():
                lines.append(f"- {key}: {value}")

        latest_by_capability = payload.get("latest_by_capability", {})
        if latest_by_capability:
            lines.append("latest_by_capability:")
            for key, record in latest_by_capability.items():
                lines.append(
                    f"- {key}: {record.get('created_at', '<unknown>')} | "
                    f"{record.get('status')} | {record.get('summary')}"
                    + (f" | run_id={record.get('run_id')}" if record.get("run_id") else "")
                    + (f" | target={record.get('target_label')}" if record.get("target_label") else "")
                    + (
                        f" | quality={record.get('result_quality')}"
                        if record.get("result_quality")
                        else ""
                    )
                )

        recent_records = payload.get("recent_records", [])
        if recent_records:
            lines.append("recent_records:")
            for record in recent_records:
                lines.append(
                    f"- {record.get('created_at', '<unknown>')} | {record.get('tool_id')} | "
                    f"{record.get('capability')} | {record.get('status')} | {record.get('summary')}"
                    + (f" | run_id={record.get('run_id')}" if record.get("run_id") else "")
                    + (f" | target={record.get('target_label')}" if record.get("target_label") else "")
                    + (
                        f" | quality={record.get('result_quality')}"
                        if record.get("result_quality")
                        else ""
                    )
                )

        return "\n".join(lines)

    def _render_archive_compare_text(self, payload: dict[str, Any]) -> str:
        lines = [
            f"repo_root: {payload.get('repo_root')}",
            f"session_id: {payload.get('session_id')}",
            f"tool_id: {payload.get('tool_id')}",
            f"capability: {payload.get('capability')}",
            f"date_from: {payload.get('date_from')}",
            f"date_to: {payload.get('date_to')}",
            f"record_count: {payload.get('record_count')}",
            f"target_count: {payload.get('target_count')}",
        ]
        target_groups = payload.get("target_groups", [])
        if target_groups:
            lines.append("targets:")
            for group in target_groups:
                lines.append(f"- {group.get('target_label')} | runs={group.get('run_count')}")
                result_quality_counts = group.get("result_quality_counts") or {}
                if result_quality_counts:
                    counts = ", ".join(f"{key}={value}" for key, value in result_quality_counts.items())
                    lines.append(f"  quality_distribution: {counts}")
                if group.get("trend_summary"):
                    lines.append(f"  trend: {group.get('trend_summary')}")
                recent_runs = group.get("recent_runs") or []
                if recent_runs:
                    lines.append("  recent_runs:")
                    for run in recent_runs:
                        lines.append(
                            "  - "
                            f"{run.get('created_at', '<unknown>')} | "
                            f"{run.get('result_quality') or 'unknown'} | "
                            f"{run.get('status')} | {run.get('summary')}"
                            + (f" | run_id={run.get('run_id')}" if run.get("run_id") else "")
                        )
        return "\n".join(lines)

    def _render_persistence_status_text(self, payload: dict[str, Any]) -> str:
        lines = [
            f"repo_root: {payload.get('repo_root')}",
            f"db_path: {payload.get('db_path')}",
        ]
        migrations = payload.get("migrations") or []
        if migrations:
            lines.append("migrations:")
            for migration in migrations:
                lines.append(
                    f"- {migration.get('version')} | {migration.get('name')} | {migration.get('applied_at')}"
                )
        import_runs = payload.get("import_runs") or []
        if import_runs:
            lines.append("import_runs:")
            for run in import_runs:
                lines.append(f"- {run.get('name')} | {run.get('completed_at')}")
        table_counts = payload.get("table_counts") or {}
        if table_counts:
            lines.append("table_counts:")
            for key, value in table_counts.items():
                lines.append(f"- {key}: {value}")
        semantic_status = payload.get("semantic_status") or {}
        if semantic_status:
            lines.append("semantic_status:")
            lines.extend(self._semantic_status_lines(semantic_status, bullet_prefix="- "))
        lines.append(f"fallback_read_count: {payload.get('fallback_read_count')}")
        fallback_breakdown = payload.get("fallback_read_breakdown") or {}
        if fallback_breakdown:
            lines.append("fallback_read_breakdown:")
            for key, value in fallback_breakdown.items():
                lines.append(f"- {key}: {value}")
        return "\n".join(lines)

    def _render_persistence_coverage_text(self, payload: dict[str, Any]) -> str:
        lines = [
            f"repo_root: {payload.get('repo_root')}",
            f"db_path: {payload.get('db_path')}",
        ]
        db_counts = payload.get("db_counts") or {}
        if db_counts:
            lines.append("db_counts:")
            for key, value in db_counts.items():
                lines.append(f"- {key}: {value}")
        legacy_counts = payload.get("legacy_counts") or {}
        if legacy_counts:
            lines.append("legacy_counts:")
            for key, value in legacy_counts.items():
                lines.append(f"- {key}: {value}")
        semantic_status = payload.get("semantic_status") or {}
        if semantic_status:
            lines.append("semantic_status:")
            lines.extend(self._semantic_status_lines(semantic_status, bullet_prefix="- "))
        fallback_breakdown = payload.get("fallback_read_breakdown") or {}
        if fallback_breakdown:
            lines.append("fallback_read_breakdown:")
            for key, value in fallback_breakdown.items():
                lines.append(f"- {key}: {value}")
        return "\n".join(lines)

    def _render_persistence_doctor_text(self, payload: dict[str, Any]) -> str:
        lines = [
            f"repo_root: {payload.get('repo_root')}",
            f"db_path: {payload.get('db_path')}",
        ]
        missing_rows = payload.get("missing_rows") or {}
        if missing_rows:
            lines.append("missing_rows:")
            for key, value in missing_rows.items():
                lines.append(f"- {key}: {value}")
        orphan_counts = payload.get("orphan_counts") or {}
        if orphan_counts:
            lines.append("orphan_counts:")
            for key, value in orphan_counts.items():
                lines.append(f"- {key}: {value}")
        duplicates = payload.get("duplicate_import_suspicions") or {}
        if duplicates:
            lines.append("duplicate_import_suspicions:")
            for key, value in duplicates.items():
                lines.append(f"- {key}: {value}")
        semantic_status = payload.get("semantic_status") or {}
        if semantic_status:
            lines.append("semantic_status:")
            lines.extend(self._semantic_status_lines(semantic_status, bullet_prefix="- "))
        lines.append(f"fallback_read_count: {payload.get('fallback_read_count')}")
        fallback_surfaces = payload.get("fallback_read_surfaces") or []
        if fallback_surfaces:
            lines.append("fallback_read_surfaces:")
            for surface in fallback_surfaces:
                lines.append(f"- {surface}")
        fallback_breakdown = payload.get("fallback_read_breakdown") or {}
        if fallback_breakdown:
            lines.append("fallback_read_breakdown:")
            for key, value in fallback_breakdown.items():
                lines.append(f"- {key}: {value}")
        return "\n".join(lines)

    def _render_semantic_status_text(self, payload: dict[str, Any]) -> str:
        lines = [f"db_path: {payload.get('db_path')}"]
        lines.extend(self._semantic_status_lines(payload))
        return "\n".join(lines)

    @staticmethod
    def _semantic_status_lines(payload: dict[str, Any], *, bullet_prefix: str = "") -> list[str]:
        lines = [
            f"{bullet_prefix}semantic_model_name: {payload.get('semantic_model_name')}",
            f"{bullet_prefix}semantic_runtime_available: {payload.get('semantic_runtime_available')}",
            f"{bullet_prefix}total_memory_items: {payload.get('total_memory_items')}",
            f"{bullet_prefix}embedded_memory_items: {payload.get('embedded_memory_items')}",
            f"{bullet_prefix}ready_embeddings: {payload.get('ready_embeddings')}",
            f"{bullet_prefix}pending_embeddings: {payload.get('pending_embeddings')}",
            f"{bullet_prefix}failed_embeddings: {payload.get('failed_embeddings')}",
            f"{bullet_prefix}stale_embeddings: {payload.get('stale_embeddings')}",
        ]
        if payload.get("semantic_runtime_error"):
            lines.append(f"{bullet_prefix}semantic_runtime_error: {payload.get('semantic_runtime_error')}")
        return lines

    @staticmethod
    def _render_semantic_backfill_text(payload: dict[str, Any]) -> str:
        lines = [
            f"db_path: {payload.get('db_path')}",
            f"model_name: {payload.get('model_name')}",
            f"runtime_available: {payload.get('runtime_available')}",
            f"processed: {payload.get('processed')}",
            f"ready: {payload.get('ready')}",
            f"failed: {payload.get('failed')}",
            f"pending: {payload.get('pending')}",
            f"skipped: {payload.get('skipped')}",
        ]
        return "\n".join(lines)
