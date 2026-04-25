from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class DoctorCheck:
    name: str
    status: str
    details: str
    extra: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        payload = {
            "name": self.name,
            "status": self.status,
            "details": self.details,
        }
        payload.update(self.extra)
        return payload


@dataclass(slots=True)
class DoctorReport:
    status: str
    repo_root: str
    checks: list[DoctorCheck]

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "repo_root": self.repo_root,
            "checks": [check.to_dict() for check in self.checks],
        }


@dataclass(slots=True)
class ArchiveListReport:
    repo_root: str
    session_id: str | None
    tool_id: str | None
    capability: str | None
    query: str | None
    status_filter: str | None
    date_from: str | None
    date_to: str | None
    record_count: int
    records: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return {
            "repo_root": self.repo_root,
            "session_id": self.session_id,
            "tool_id": self.tool_id,
            "capability": self.capability,
            "query": self.query,
            "status_filter": self.status_filter,
            "date_from": self.date_from,
            "date_to": self.date_to,
            "record_count": self.record_count,
            "records": list(self.records),
        }


@dataclass(slots=True)
class InteractionSummaryReport:
    repo_root: str
    session_id: str | None
    interaction_count: int
    clarification_count: int
    clarification_ratio: float
    clarification_trend: list[str]
    recent_clarification_mix: str | None
    latest_clarification: str | None
    clarification_drift: str | None
    mode_counts: dict[str, int]
    kind_counts: dict[str, int]
    posture_counts: dict[str, int]
    posture_trend: list[str]
    recent_posture_mix: str | None
    latest_posture: str | None
    posture_drift: str | None
    detected_language_counts: dict[str, int]
    dominant_intent_counts: dict[str, int]
    local_context_assessment_counts: dict[str, int]
    coherence_topic_count: int
    semantic_route_count: int
    semantic_route_ratio: float
    route_normalized_score_count: int
    route_normalized_score_avg: float
    route_normalized_score_max: float
    route_intent_bias_count: int
    route_intent_bias_ratio: float
    route_intent_caution_count: int
    route_intent_caution_ratio: float
    retrieval_route_caution_count: int
    retrieval_route_caution_ratio: float
    retrieval_lead_counts: dict[str, int]
    retrieval_observation_count: int
    evidence_strength_counts: dict[str, int]
    evidence_source_counts: dict[str, int]
    missing_source_counts: dict[str, int]
    deep_validation_count: int
    deep_validation_ratio: float
    contradiction_signal_count: int
    contradiction_flag_counts: dict[str, int]
    memory_classification_counts: dict[str, int]
    memory_write_action_counts: dict[str, int]
    memory_save_eligible_count: int
    explicit_memory_consent_count: int
    memory_surface_block_count: int
    personal_memory_saved_count: int
    research_note_count: int
    research_artifact_count: int
    research_artifact_type_counts: dict[str, int]
    recent_topics: list[str]
    tool_route_origin_counts: dict[str, int]
    resolution_counts: dict[str, int]
    recent_interactions: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return {
            "repo_root": self.repo_root,
            "session_id": self.session_id,
            "interaction_count": self.interaction_count,
            "clarification_count": self.clarification_count,
            "clarification_ratio": self.clarification_ratio,
            "clarification_trend": list(self.clarification_trend),
            "recent_clarification_mix": self.recent_clarification_mix,
            "latest_clarification": self.latest_clarification,
            "clarification_drift": self.clarification_drift,
            "mode_counts": dict(self.mode_counts),
            "kind_counts": dict(self.kind_counts),
            "posture_counts": dict(self.posture_counts),
            "posture_trend": list(self.posture_trend),
            "recent_posture_mix": self.recent_posture_mix,
            "latest_posture": self.latest_posture,
            "posture_drift": self.posture_drift,
            "detected_language_counts": dict(self.detected_language_counts),
            "dominant_intent_counts": dict(self.dominant_intent_counts),
            "local_context_assessment_counts": dict(self.local_context_assessment_counts),
            "coherence_topic_count": self.coherence_topic_count,
            "semantic_route_count": self.semantic_route_count,
            "semantic_route_ratio": self.semantic_route_ratio,
            "route_normalized_score_count": self.route_normalized_score_count,
            "route_normalized_score_avg": self.route_normalized_score_avg,
            "route_normalized_score_max": self.route_normalized_score_max,
            "route_intent_bias_count": self.route_intent_bias_count,
            "route_intent_bias_ratio": self.route_intent_bias_ratio,
            "route_intent_caution_count": self.route_intent_caution_count,
            "route_intent_caution_ratio": self.route_intent_caution_ratio,
            "retrieval_route_caution_count": self.retrieval_route_caution_count,
            "retrieval_route_caution_ratio": self.retrieval_route_caution_ratio,
            "retrieval_lead_counts": dict(self.retrieval_lead_counts),
            "retrieval_observation_count": self.retrieval_observation_count,
            "evidence_strength_counts": dict(self.evidence_strength_counts),
            "evidence_source_counts": dict(self.evidence_source_counts),
            "missing_source_counts": dict(self.missing_source_counts),
            "deep_validation_count": self.deep_validation_count,
            "deep_validation_ratio": self.deep_validation_ratio,
            "contradiction_signal_count": self.contradiction_signal_count,
            "contradiction_flag_counts": dict(self.contradiction_flag_counts),
            "memory_classification_counts": dict(self.memory_classification_counts),
            "memory_write_action_counts": dict(self.memory_write_action_counts),
            "memory_save_eligible_count": self.memory_save_eligible_count,
            "explicit_memory_consent_count": self.explicit_memory_consent_count,
            "memory_surface_block_count": self.memory_surface_block_count,
            "personal_memory_saved_count": self.personal_memory_saved_count,
            "research_note_count": self.research_note_count,
            "research_artifact_count": self.research_artifact_count,
            "research_artifact_type_counts": dict(self.research_artifact_type_counts),
            "recent_topics": list(self.recent_topics),
            "tool_route_origin_counts": dict(self.tool_route_origin_counts),
            "resolution_counts": dict(self.resolution_counts),
            "recent_interactions": list(self.recent_interactions),
        }


@dataclass(slots=True)
class InteractionListReport:
    repo_root: str
    session_id: str | None
    resolution_strategy: str | None
    interaction_count: int
    interaction_records: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return {
            "repo_root": self.repo_root,
            "session_id": self.session_id,
            "resolution_strategy": self.resolution_strategy,
            "interaction_count": self.interaction_count,
            "interaction_records": list(self.interaction_records),
        }


@dataclass(slots=True)
class InteractionSearchReport:
    repo_root: str
    session_id: str | None
    query: str
    resolution_strategy: str | None
    interaction_count: int
    matches: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return {
            "repo_root": self.repo_root,
            "session_id": self.session_id,
            "query": self.query,
            "resolution_strategy": self.resolution_strategy,
            "interaction_count": self.interaction_count,
            "matches": list(self.matches),
        }


@dataclass(slots=True)
class InteractionEvaluationReport:
    repo_root: str
    session_id: str | None
    evaluated_count: int
    judgment_counts: dict[str, int]
    surface_aggregates: list[dict[str, object]]
    evaluations: list[dict[str, object]]
    behavior_slices: list[dict[str, object]] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "repo_root": self.repo_root,
            "session_id": self.session_id,
            "evaluated_count": self.evaluated_count,
            "judgment_counts": dict(self.judgment_counts),
            "surface_aggregates": list(self.surface_aggregates),
            "evaluations": list(self.evaluations),
            "behavior_slices": list(self.behavior_slices or []),
        }


@dataclass(slots=True)
class LabeledDatasetExportReport:
    repo_root: str
    session_id: str | None
    project_id: str | None
    dataset_path: str
    manifest_path: str
    export_batch_id: str
    example_count: int
    split_counts: dict[str, int]
    label_category_counts: dict[str, int]
    label_value_counts: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "repo_root": self.repo_root,
            "session_id": self.session_id,
            "project_id": self.project_id,
            "dataset_path": self.dataset_path,
            "manifest_path": self.manifest_path,
            "export_batch_id": self.export_batch_id,
            "example_count": self.example_count,
            "split_counts": dict(self.split_counts),
            "label_category_counts": dict(self.label_category_counts),
            "label_value_counts": dict(self.label_value_counts),
        }


@dataclass(slots=True)
class InteractionPatternsReport:
    repo_root: str
    session_id: str | None
    interaction_count: int
    follow_up_count: int
    ambiguous_follow_up_count: int
    rewrite_ratio: float
    follow_up_ratio: float
    ambiguity_ratio: float
    resolution_counts: dict[str, int]
    observations: list[str]
    status: str
    recent_interactions: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return {
            "repo_root": self.repo_root,
            "session_id": self.session_id,
            "interaction_count": self.interaction_count,
            "follow_up_count": self.follow_up_count,
            "ambiguous_follow_up_count": self.ambiguous_follow_up_count,
            "rewrite_ratio": self.rewrite_ratio,
            "follow_up_ratio": self.follow_up_ratio,
            "ambiguity_ratio": self.ambiguity_ratio,
            "resolution_counts": dict(self.resolution_counts),
            "observations": list(self.observations),
            "status": self.status,
            "recent_interactions": list(self.recent_interactions),
        }


@dataclass(slots=True)
class SessionReport:
    repo_root: str
    session_id: str
    record_count: int
    records: list[dict[str, object]]
    interaction_count: int
    clarification_count: int
    clarification_ratio: float
    clarification_trend: list[str]
    recent_clarification_mix: str | None
    latest_clarification: str | None
    clarification_drift: str | None
    posture_counts: dict[str, int]
    posture_trend: list[str]
    recent_posture_mix: str | None
    latest_posture: str | None
    posture_drift: str | None
    detected_language_counts: dict[str, int]
    dominant_intent_counts: dict[str, int]
    local_context_assessment_counts: dict[str, int]
    coherence_topic_count: int
    semantic_route_count: int
    semantic_route_ratio: float
    route_normalized_score_count: int
    route_normalized_score_avg: float
    route_normalized_score_max: float
    route_intent_bias_count: int
    route_intent_bias_ratio: float
    route_intent_caution_count: int
    route_intent_caution_ratio: float
    retrieval_route_caution_count: int
    retrieval_route_caution_ratio: float
    retrieval_lead_counts: dict[str, int]
    retrieval_observation_count: int
    evidence_strength_counts: dict[str, int]
    evidence_source_counts: dict[str, int]
    missing_source_counts: dict[str, int]
    deep_validation_count: int
    deep_validation_ratio: float
    contradiction_signal_count: int
    contradiction_flag_counts: dict[str, int]
    memory_classification_counts: dict[str, int]
    memory_write_action_counts: dict[str, int]
    memory_save_eligible_count: int
    explicit_memory_consent_count: int
    memory_surface_block_count: int
    personal_memory_saved_count: int
    research_note_count: int
    research_artifact_count: int
    research_artifact_type_counts: dict[str, int]
    recent_topics: list[str]
    tool_route_origin_counts: dict[str, int]
    interaction_records: list[dict[str, object]]
    interaction_profile: dict[str, object] | None
    active_thread: dict[str, object] | None

    def to_dict(self) -> dict[str, object]:
        return {
            "repo_root": self.repo_root,
            "session_id": self.session_id,
            "record_count": self.record_count,
            "records": list(self.records),
            "interaction_count": self.interaction_count,
            "clarification_count": self.clarification_count,
            "clarification_ratio": self.clarification_ratio,
            "clarification_trend": list(self.clarification_trend),
            "recent_clarification_mix": self.recent_clarification_mix,
            "latest_clarification": self.latest_clarification,
            "clarification_drift": self.clarification_drift,
            "posture_counts": dict(self.posture_counts),
            "posture_trend": list(self.posture_trend),
            "recent_posture_mix": self.recent_posture_mix,
            "latest_posture": self.latest_posture,
            "posture_drift": self.posture_drift,
            "detected_language_counts": dict(self.detected_language_counts),
            "dominant_intent_counts": dict(self.dominant_intent_counts),
            "local_context_assessment_counts": dict(self.local_context_assessment_counts),
            "coherence_topic_count": self.coherence_topic_count,
            "semantic_route_count": self.semantic_route_count,
            "semantic_route_ratio": self.semantic_route_ratio,
            "route_normalized_score_count": self.route_normalized_score_count,
            "route_normalized_score_avg": self.route_normalized_score_avg,
            "route_normalized_score_max": self.route_normalized_score_max,
            "route_intent_bias_count": self.route_intent_bias_count,
            "route_intent_bias_ratio": self.route_intent_bias_ratio,
            "route_intent_caution_count": self.route_intent_caution_count,
            "route_intent_caution_ratio": self.route_intent_caution_ratio,
            "retrieval_route_caution_count": self.retrieval_route_caution_count,
            "retrieval_route_caution_ratio": self.retrieval_route_caution_ratio,
            "retrieval_lead_counts": dict(self.retrieval_lead_counts),
            "retrieval_observation_count": self.retrieval_observation_count,
            "evidence_strength_counts": dict(self.evidence_strength_counts),
            "evidence_source_counts": dict(self.evidence_source_counts),
            "missing_source_counts": dict(self.missing_source_counts),
            "deep_validation_count": self.deep_validation_count,
            "deep_validation_ratio": self.deep_validation_ratio,
            "contradiction_signal_count": self.contradiction_signal_count,
            "contradiction_flag_counts": dict(self.contradiction_flag_counts),
            "memory_classification_counts": dict(self.memory_classification_counts),
            "memory_write_action_counts": dict(self.memory_write_action_counts),
            "memory_save_eligible_count": self.memory_save_eligible_count,
            "explicit_memory_consent_count": self.explicit_memory_consent_count,
            "memory_surface_block_count": self.memory_surface_block_count,
            "personal_memory_saved_count": self.personal_memory_saved_count,
            "research_note_count": self.research_note_count,
            "research_artifact_count": self.research_artifact_count,
            "research_artifact_type_counts": dict(self.research_artifact_type_counts),
            "recent_topics": list(self.recent_topics),
            "tool_route_origin_counts": dict(self.tool_route_origin_counts),
            "interaction_records": list(self.interaction_records),
            "interaction_profile": self.interaction_profile,
            "active_thread": self.active_thread,
        }


@dataclass(slots=True)
class ActiveThreadReport:
    repo_root: str
    session_id: str
    interaction_profile: dict[str, object] | None
    active_thread: dict[str, object] | None

    def to_dict(self) -> dict[str, object]:
        return {
            "repo_root": self.repo_root,
            "session_id": self.session_id,
            "interaction_profile": self.interaction_profile,
            "active_thread": self.active_thread,
        }


@dataclass(slots=True)
class SessionResetReport:
    repo_root: str
    session_id: str
    cleared: bool
    state_path: str
    interaction_profile: dict[str, object] | None
    active_thread: dict[str, object] | None

    def to_dict(self) -> dict[str, object]:
        return {
            "repo_root": self.repo_root,
            "session_id": self.session_id,
            "cleared": self.cleared,
            "state_path": self.state_path,
            "interaction_profile": self.interaction_profile,
            "active_thread": self.active_thread,
        }
