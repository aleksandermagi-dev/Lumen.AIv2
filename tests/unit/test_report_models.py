from lumen.reporting.report_models import (
    ActiveThreadReport,
    ArchiveListReport,
    DoctorCheck,
    DoctorReport,
    InteractionListReport,
    InteractionPatternsReport,
    InteractionSearchReport,
    InteractionSummaryReport,
    SessionReport,
    SessionResetReport,
)


def test_doctor_report_model_serializes_checks() -> None:
    payload = DoctorReport(
        status="ok",
        repo_root="C:/repo",
        checks=[
            DoctorCheck(
                name="tool_registry",
                status="ok",
                details="Discovered bundles",
                extra={"bundles": {"anh": ["spectral_dip_scan"]}},
            )
        ],
    ).to_dict()

    assert payload["status"] == "ok"
    assert payload["repo_root"] == "C:/repo"
    assert payload["checks"][0]["name"] == "tool_registry"
    assert payload["checks"][0]["bundles"]["anh"] == ["spectral_dip_scan"]


def test_archive_list_report_model_serializes_payload() -> None:
    payload = ArchiveListReport(
        repo_root="C:/repo",
        session_id="default",
        tool_id="anh",
        capability="spectral_dip_scan",
        query=None,
        status_filter="ok",
        date_from=None,
        date_to=None,
        record_count=1,
        records=[{"tool_id": "anh", "summary": "indexed run"}],
    ).to_dict()

    assert payload["tool_id"] == "anh"
    assert payload["status_filter"] == "ok"
    assert payload["record_count"] == 1
    assert payload["records"][0]["summary"] == "indexed run"


def test_interaction_summary_report_model_serializes_payload() -> None:
    payload = InteractionSummaryReport(
        repo_root="C:/repo",
        session_id="default",
        interaction_count=2,
        clarification_count=1,
        clarification_ratio=0.5,
        clarification_trend=["clarified", "clear"],
        recent_clarification_mix="mixed",
        latest_clarification="clarified",
        clarification_drift="increasing",
        mode_counts={"planning": 1, "research": 1},
        kind_counts={"planning.migration": 1, "research.summary": 1},
        posture_counts={"supported": 1, "tentative": 1},
        posture_trend=["supported", "tentative"],
        recent_posture_mix="improving_or_mixed",
        latest_posture="supported",
        posture_drift="strengthening",
        detected_language_counts={"en": 2},
        dominant_intent_counts={"planning": 1, "research": 1},
        local_context_assessment_counts={"aligned": 1, "partial": 1},
        coherence_topic_count=1,
        semantic_route_count=1,
        semantic_route_ratio=0.5,
        route_normalized_score_count=1,
        route_normalized_score_avg=1.42,
        route_normalized_score_max=1.42,
        route_intent_bias_count=1,
        route_intent_bias_ratio=0.5,
        route_intent_caution_count=0,
        route_intent_caution_ratio=0.0,
        retrieval_route_caution_count=1,
        retrieval_route_caution_ratio=0.5,
        retrieval_lead_counts={"blended": 1, "keyword": 1},
        retrieval_observation_count=2,
        evidence_strength_counts={"supported": 1, "light": 1},
        evidence_source_counts={"archive": 1, "interaction": 1},
        missing_source_counts={"workspace": 1},
        deep_validation_count=1,
        deep_validation_ratio=0.5,
        contradiction_signal_count=1,
        contradiction_flag_counts={"topic_mismatch": 1},
        memory_classification_counts={"research_memory_candidate": 1, "ephemeral_conversation_context": 1},
        memory_write_action_counts={"save_research_note": 1, "skip": 1},
        memory_save_eligible_count=1,
        explicit_memory_consent_count=0,
        memory_surface_block_count=0,
        personal_memory_saved_count=0,
        research_note_count=1,
        research_artifact_count=1,
        research_artifact_type_counts={"decision": 1},
        recent_topics=["migration plan", "archive structure"],
        tool_route_origin_counts={"none": 2},
        resolution_counts={"none": 1, "compare_shorthand": 1},
        recent_interactions=[{"prompt": "create a migration plan for lumen"}],
    ).to_dict()

    assert payload["interaction_count"] == 2
    assert payload["clarification_count"] == 1
    assert payload["clarification_ratio"] == 0.5
    assert payload["clarification_trend"] == ["clarified", "clear"]
    assert payload["recent_clarification_mix"] == "mixed"
    assert payload["latest_clarification"] == "clarified"
    assert payload["clarification_drift"] == "increasing"
    assert payload["mode_counts"]["planning"] == 1
    assert payload["posture_counts"]["supported"] == 1
    assert payload["posture_trend"] == ["supported", "tentative"]
    assert payload["recent_posture_mix"] == "improving_or_mixed"
    assert payload["latest_posture"] == "supported"
    assert payload["posture_drift"] == "strengthening"
    assert payload["detected_language_counts"]["en"] == 2
    assert payload["dominant_intent_counts"]["planning"] == 1
    assert payload["local_context_assessment_counts"]["aligned"] == 1
    assert payload["coherence_topic_count"] == 1
    assert payload["semantic_route_count"] == 1
    assert payload["semantic_route_ratio"] == 0.5
    assert payload["route_normalized_score_count"] == 1
    assert payload["route_normalized_score_avg"] == 1.42
    assert payload["route_normalized_score_max"] == 1.42
    assert payload["route_intent_bias_count"] == 1
    assert payload["route_intent_bias_ratio"] == 0.5
    assert payload["route_intent_caution_count"] == 0
    assert payload["route_intent_caution_ratio"] == 0.0
    assert payload["retrieval_route_caution_count"] == 1
    assert payload["retrieval_route_caution_ratio"] == 0.5
    assert payload["retrieval_lead_counts"]["blended"] == 1
    assert payload["retrieval_observation_count"] == 2
    assert payload["memory_classification_counts"]["research_memory_candidate"] == 1
    assert payload["memory_write_action_counts"]["save_research_note"] == 1
    assert payload["memory_save_eligible_count"] == 1
    assert payload["explicit_memory_consent_count"] == 0
    assert payload["memory_surface_block_count"] == 0
    assert payload["personal_memory_saved_count"] == 0
    assert payload["research_note_count"] == 1
    assert payload["research_artifact_count"] == 1
    assert payload["research_artifact_type_counts"]["decision"] == 1
    assert payload["recent_interactions"][0]["prompt"] == "create a migration plan for lumen"


def test_interaction_patterns_report_model_serializes_payload() -> None:
    payload = InteractionPatternsReport(
        repo_root="C:/repo",
        session_id="default",
        interaction_count=3,
        follow_up_count=2,
        ambiguous_follow_up_count=0,
        rewrite_ratio=0.6667,
        follow_up_ratio=0.6667,
        ambiguity_ratio=0.0,
        resolution_counts={"compare_shorthand": 1, "reference_follow_up": 1},
        observations=["Interaction patterns look stable."],
        status="ok",
        recent_interactions=[{"prompt": "what about that"}],
    ).to_dict()

    assert payload["status"] == "ok"
    assert payload["follow_up_count"] == 2
    assert payload["resolution_counts"]["compare_shorthand"] == 1
    assert payload["recent_interactions"][0]["prompt"] == "what about that"


def test_interaction_list_and_search_report_models_serialize_payloads() -> None:
    listing = InteractionListReport(
        repo_root="C:/repo",
        session_id="default",
        resolution_strategy="compare_shorthand",
        interaction_count=1,
        interaction_records=[{"prompt": "what about that"}],
    ).to_dict()
    search = InteractionSearchReport(
        repo_root="C:/repo",
        session_id="default",
        query="migration",
        resolution_strategy=None,
        interaction_count=1,
        matches=[{"score": 4, "record": {"prompt": "what about that"}}],
    ).to_dict()

    assert listing["interaction_count"] == 1
    assert listing["interaction_records"][0]["prompt"] == "what about that"
    assert search["query"] == "migration"
    assert search["matches"][0]["record"]["prompt"] == "what about that"


def test_session_report_models_serialize_payloads() -> None:
    session = SessionReport(
        repo_root="C:/repo",
        session_id="default",
        record_count=1,
        records=[{"tool_id": "anh"}],
        interaction_count=1,
        clarification_count=1,
        clarification_ratio=1.0,
        clarification_trend=["clarified"],
        recent_clarification_mix="stable:clarified",
        latest_clarification="clarified",
        clarification_drift="insufficient_data",
        posture_counts={"supported": 1},
        posture_trend=["supported"],
        recent_posture_mix="stable:supported",
        latest_posture="supported",
        posture_drift="insufficient_data",
        detected_language_counts={"en": 1},
        dominant_intent_counts={"planning": 1},
        local_context_assessment_counts={"aligned": 1},
        coherence_topic_count=1,
        semantic_route_count=1,
        semantic_route_ratio=1.0,
        route_normalized_score_count=1,
        route_normalized_score_avg=1.61,
        route_normalized_score_max=1.61,
        route_intent_bias_count=1,
        route_intent_bias_ratio=1.0,
        route_intent_caution_count=0,
        route_intent_caution_ratio=0.0,
        retrieval_route_caution_count=1,
        retrieval_route_caution_ratio=1.0,
        retrieval_lead_counts={"blended": 1},
        retrieval_observation_count=1,
        evidence_strength_counts={"supported": 1},
        evidence_source_counts={"archive": 1},
        missing_source_counts={},
        deep_validation_count=1,
        deep_validation_ratio=1.0,
        contradiction_signal_count=0,
        contradiction_flag_counts={},
        memory_classification_counts={"research_memory_candidate": 1},
        memory_write_action_counts={"save_research_note": 1},
        memory_save_eligible_count=1,
        explicit_memory_consent_count=0,
        memory_surface_block_count=0,
        personal_memory_saved_count=0,
        research_note_count=1,
        research_artifact_count=1,
        research_artifact_type_counts={"decision": 1},
        recent_topics=["migration plan for lumen"],
        tool_route_origin_counts={"none": 1},
        interaction_records=[{"prompt": "create a migration plan for lumen"}],
        interaction_profile=None,
        active_thread={
            "prompt": "create a migration plan for lumen",
            "confidence_posture": "supported",
            "local_context_assessment": "aligned",
            "coherence_topic": "migration lumen routing",
        },
    ).to_dict()
    current = ActiveThreadReport(
        repo_root="C:/repo",
        session_id="default",
        interaction_profile=None,
        active_thread={"prompt": "create a migration plan for lumen"},
    ).to_dict()
    reset = SessionResetReport(
        repo_root="C:/repo",
        session_id="default",
        cleared=True,
        state_path="C:/repo/data/sessions/default/thread_state.json",
        interaction_profile=None,
        active_thread=None,
    ).to_dict()

    assert session["record_count"] == 1
    assert session["posture_counts"]["supported"] == 1
    assert session["clarification_count"] == 1
    assert session["clarification_ratio"] == 1.0
    assert session["clarification_trend"] == ["clarified"]
    assert session["recent_clarification_mix"] == "stable:clarified"
    assert session["latest_clarification"] == "clarified"
    assert session["clarification_drift"] == "insufficient_data"
    assert session["posture_trend"] == ["supported"]
    assert session["recent_posture_mix"] == "stable:supported"
    assert session["latest_posture"] == "supported"
    assert session["posture_drift"] == "insufficient_data"
    assert session["detected_language_counts"]["en"] == 1
    assert session["dominant_intent_counts"]["planning"] == 1
    assert session["local_context_assessment_counts"]["aligned"] == 1
    assert session["coherence_topic_count"] == 1
    assert session["semantic_route_count"] == 1
    assert session["semantic_route_ratio"] == 1.0
    assert session["route_normalized_score_count"] == 1
    assert session["route_normalized_score_avg"] == 1.61
    assert session["route_normalized_score_max"] == 1.61
    assert session["route_intent_bias_count"] == 1
    assert session["route_intent_bias_ratio"] == 1.0
    assert session["route_intent_caution_count"] == 0
    assert session["route_intent_caution_ratio"] == 0.0
    assert session["retrieval_route_caution_count"] == 1
    assert session["retrieval_route_caution_ratio"] == 1.0
    assert session["retrieval_lead_counts"]["blended"] == 1
    assert session["retrieval_observation_count"] == 1
    assert session["memory_classification_counts"]["research_memory_candidate"] == 1
    assert session["memory_write_action_counts"]["save_research_note"] == 1
    assert session["memory_save_eligible_count"] == 1
    assert session["memory_surface_block_count"] == 0
    assert session["personal_memory_saved_count"] == 0
    assert session["research_note_count"] == 1
    assert session["research_artifact_count"] == 1
    assert session["research_artifact_type_counts"]["decision"] == 1
    assert session["active_thread"]["prompt"] == "create a migration plan for lumen"
    assert session["active_thread"]["confidence_posture"] == "supported"
    assert session["active_thread"]["local_context_assessment"] == "aligned"
    assert session["active_thread"]["coherence_topic"] == "migration lumen routing"
    assert current["active_thread"]["prompt"] == "create a migration plan for lumen"
    assert reset["cleared"] is True

