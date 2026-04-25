from pathlib import Path
from datetime import UTC, datetime
import json

from lumen.app.controller import AppController
from lumen.app.settings import AppSettings
from lumen.memory.interaction_log_manager import InteractionLogManager
from lumen.routing.domain_router import DomainRoute


def test_controller_can_list_and_search_interactions(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)

    controller.ask(prompt="create a roadmap for developing lumen further", session_id="default")
    controller.ask(prompt="summarize the current archive structure", session_id="default")

    listing = controller.list_interactions(session_id="default")
    search = controller.search_interactions("roadmap", session_id="default")

    assert listing["interaction_count"] == 2
    assert len(listing["interaction_records"]) == 2
    assert listing["interaction_records"][0]["prompt_view"]["canonical_prompt"] is not None
    assert listing["interaction_records"][0]["detected_language"] == "en"
    assert listing["interaction_records"][0]["normalized_topic"] is not None
    assert listing["interaction_records"][0]["dominant_intent"] in {"planning", "research"}
    assert isinstance(listing["interaction_records"][0]["extracted_entities"], list)
    assert listing["interaction_records"][0]["memory_classification"]["candidate_type"] == "research_memory_candidate"
    assert listing["interaction_records"][0]["memory_classification"]["save_eligible"] is True
    assert isinstance(listing["interaction_records"][0]["research_note"], dict)
    assert listing["interaction_records"][0]["research_note"]["note_type"] == "chronological_research_note"

    assert search["interaction_count"] >= 1
    assert search["matches"][0]["record"]["prompt"] == "create a roadmap for developing lumen further"
    assert search["matches"][0]["record"]["detected_language"] == "en"
    assert search["matches"][0]["record"]["normalized_topic"] == "roadmap for developing lumen further"
    assert search["matches"][0]["record"]["dominant_intent"] == "planning"
    assert any(entity["value"] == "roadmap" for entity in search["matches"][0]["record"]["extracted_entities"])
    assert search["matches"][0]["score"] > 0
    assert search["matches"][0]["score_breakdown"]["keyword_score"] > 0
    assert (tmp_path / "data" / "interactions" / "default" / "_index.json").exists()


def test_interaction_index_writes_semantic_signature(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)

    controller.ask(prompt="create a migration plan for lumen routing", session_id="default")

    index_path = tmp_path / "data" / "interactions" / "default" / "_index.json"
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))

    assert len(index_payload) == 1
    signature = index_payload[0]["semantic_signature"]
    assert "migration" in signature["prompt_tokens"]
    assert "routing" in signature["topic_tokens"]
    assert signature["dominant_intent"] == "planning"
    assert "routing" in signature["entities"]


def test_interaction_index_search_caps_candidates_by_setting(tmp_path: Path) -> None:
    (tmp_path / "lumen.toml").write_text(
        "\n".join(
            [
                "[app]",
                "search_candidate_limit = 2",
            ]
        ),
        encoding="utf-8",
    )
    controller = AppController(repo_root=tmp_path)

    controller.ask(prompt="create migration routing plan alpha", session_id="default")
    controller.ask(prompt="create migration routing plan beta", session_id="default")
    controller.ask(prompt="create migration routing plan gamma", session_id="default")

    query_understanding = controller.interaction_history_service.interaction_log_manager.prompt_nlu.analyze(
        "migration routing plan"
    )
    matches = controller.interaction_history_service.interaction_log_manager._search_index(
        "migration",
        query_understanding=query_understanding,
        session_id="default",
    )

    assert len(matches) == 2


def test_controller_searches_persisted_resolution_metadata(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)

    controller.ask(prompt="create a migration plan for lumen", session_id="default")
    controller.ask(prompt="now compare that", session_id="default")

    search = controller.search_interactions("compare_shorthand", session_id="default")

    assert search["interaction_count"] >= 1
    assert search["matches"][0]["record"]["resolution_strategy"] == "compare_shorthand"
    assert search["matches"][0]["record"]["resolved_prompt"] == "compare the migration plan for lumen"
    assert search["matches"][0]["record"]["prompt_view"]["canonical_prompt"] == "compare the migration plan for lumen"
    assert search["matches"][0]["record"]["prompt_view"]["original_prompt"] == "now compare that"
    assert search["matches"][0]["record"]["prompt_view"]["rewritten"] is True


def test_controller_search_prefers_canonical_prompt_meaning(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)

    controller.ask(prompt="create a migration plan for lumen", session_id="default")
    controller.ask(prompt="what about that", session_id="default")

    search = controller.search_interactions("what about the migration plan", session_id="default")

    assert search["interaction_count"] >= 1
    top_record = search["matches"][0]["record"]
    assert top_record["prompt"] == "what about that"
    assert top_record["prompt_view"]["canonical_prompt"] == "what about the migration plan for lumen"


def test_controller_can_filter_and_summarize_interactions_by_resolution_strategy(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)

    controller.ask(prompt="create a migration plan for lumen", session_id="default")
    controller.ask(prompt="now compare that", session_id="default")
    controller.ask(prompt="what about that", session_id="default")

    listing = controller.list_interactions(
        session_id="default",
        resolution_strategy="compare_shorthand",
    )
    summary = controller.summarize_interactions(session_id="default")

    assert listing["interaction_count"] == 1
    assert listing["interaction_records"][0]["resolution_strategy"] == "compare_shorthand"

    assert summary["interaction_count"] == 3
    assert summary["clarification_count"] == 0
    assert summary["clarification_ratio"] == 0.0
    assert summary["clarification_trend"] == ["clear", "clear", "clear"]
    assert summary["recent_clarification_mix"] == "stable:clear"
    assert summary["latest_clarification"] == "clear"
    assert summary["clarification_drift"] == "steady"
    assert sum(summary["posture_counts"].values()) == 3
    assert summary["detected_language_counts"]["en"] == 3
    assert sum(summary["dominant_intent_counts"].values()) == 3
    assert summary["dominant_intent_counts"].get("planning", 0) >= 1
    assert summary["dominant_intent_counts"].get("research", 0) >= 1
    assert "local_context_assessment_counts" in summary
    assert "coherence_topic_count" in summary
    assert "semantic_route_count" in summary
    assert "semantic_route_ratio" in summary
    assert "route_normalized_score_count" in summary
    assert "route_normalized_score_avg" in summary
    assert "route_normalized_score_max" in summary
    assert "route_intent_bias_count" in summary
    assert "route_intent_bias_ratio" in summary
    assert "route_intent_caution_count" in summary
    assert "route_intent_caution_ratio" in summary
    assert "retrieval_route_caution_count" in summary
    assert "retrieval_route_caution_ratio" in summary
    assert "retrieval_lead_counts" in summary
    assert "retrieval_observation_count" in summary
    assert "evidence_strength_counts" in summary
    assert "evidence_source_counts" in summary
    assert "missing_source_counts" in summary
    assert "deep_validation_count" in summary
    assert "deep_validation_ratio" in summary
    assert "contradiction_signal_count" in summary
    assert "contradiction_flag_counts" in summary
    assert "memory_classification_counts" in summary
    assert "memory_write_action_counts" in summary
    assert "memory_save_eligible_count" in summary
    assert "explicit_memory_consent_count" in summary
    assert "memory_surface_block_count" in summary
    assert "research_note_count" in summary
    assert "research_artifact_count" in summary
    assert "research_artifact_type_counts" in summary
    assert len(summary["recent_topics"]) >= 1
    assert summary["tool_route_origin_counts"]["none"] == 3
    assert len(summary["posture_trend"]) <= 3
    assert summary["recent_posture_mix"] is not None
    assert summary["latest_posture"] in {"supported", "conflicted", "tentative", "strong"}
    assert summary["posture_drift"] in {"strengthening", "weakening", "steady", "steady_with_variation", "insufficient_data"}
    assert summary["resolution_counts"]["compare_shorthand"] == 1
    assert summary["resolution_counts"]["reference_follow_up"] == 1
    assert listing["interaction_records"][0]["confidence_posture"] in {"supported", "conflicted", "tentative", "strong"}


def test_controller_can_evaluate_interactions_offline(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)

    controller.ask(prompt="create a migration plan for lumen routing", session_id="default")

    report = controller.evaluate_interactions(session_id="default")

    assert report["evaluated_count"] == 1
    assert report["surface_aggregates"]
    assert report["evaluations"][0]["overall_judgment"] in {
        "correct",
        "weak",
        "incorrect",
        "insufficient_evidence",
    }
    surfaces = {item["surface"] for item in report["evaluations"][0]["surface_reviews"]}
    assert "route_quality" in surfaces
    assert "supervised_support_quality" in surfaces


def test_controller_can_export_labeled_examples(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)

    controller.ask(prompt="create a migration plan for lumen routing", session_id="default")

    report = controller.export_labeled_examples(session_id="default")

    assert report["example_count"] > 0
    assert Path(report["dataset_path"]).exists()
    assert "route_decision_quality" in report["label_category_counts"]


def test_controller_does_not_auto_mark_personal_context_as_save_eligible(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)

    controller.ask(
        prompt="I feel anxious about my health",
        session_id="default",
    )

    listing = controller.list_interactions(session_id="default")
    summary = controller.summarize_interactions(session_id="default")

    memory_classification = listing["interaction_records"][0]["memory_classification"]
    assert memory_classification["candidate_type"] == "personal_context_candidate"
    assert memory_classification["save_eligible"] is False
    assert memory_classification["requires_explicit_user_consent"] is True
    assert memory_classification["explicit_save_requested"] is False
    assert listing["interaction_records"][0]["memory_write_decision"]["action"] == "skip"
    assert summary["memory_classification_counts"]["personal_context_candidate"] == 1
    assert summary["memory_write_action_counts"]["skip"] == 1
    assert summary["memory_save_eligible_count"] == 0
    assert summary["explicit_memory_consent_count"] == 1
    assert summary["memory_surface_block_count"] == 0
    assert summary["personal_memory_saved_count"] == 0
    assert summary["research_note_count"] == 0
    assert summary["research_artifact_count"] == 0
    assert listing["interaction_records"][0]["personal_memory"] is None


def test_controller_auto_saves_chronological_research_note_for_research_candidate(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)

    controller.ask(prompt="create a migration plan for lumen routing", session_id="default")

    listing = controller.list_interactions(session_id="default")
    summary = controller.summarize_interactions(session_id="default")
    notes_root = tmp_path / "data" / "research_notes" / "default"
    note_files = list(notes_root.glob("*.json"))

    assert len(note_files) == 1
    note_payload = json.loads(note_files[0].read_text(encoding="utf-8"))
    assert note_payload["note_type"] == "chronological_research_note"
    assert note_payload["source_interaction_prompt"] == "create a migration plan for lumen routing"
    assert note_payload["source_interaction_mode"] == "planning"
    assert note_payload["memory_classification"]["candidate_type"] == "research_memory_candidate"
    assert listing["interaction_records"][0]["memory_write_decision"]["action"] == "save_research_note"
    assert summary["memory_write_action_counts"]["save_research_note"] == 1
    assert listing["interaction_records"][0]["research_note"]["note_path"] == str(note_files[0])
    assert summary["research_note_count"] == 1
    assert summary["research_artifact_count"] == 0


def test_controller_does_not_auto_save_lightweight_thread_continuation(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)

    controller.ask(prompt="create a migration plan for lumen routing", session_id="default")
    controller.ask(prompt="keep going", session_id="default")

    listing = controller.list_interactions(session_id="default")
    latest = listing["interaction_records"][0]

    assert latest["prompt"] == "keep going"
    assert latest["memory_classification"]["candidate_type"] == "ephemeral_conversation_context"
    assert latest["memory_write_decision"]["action"] == "skip"


def test_controller_does_not_auto_save_research_note_for_personal_context(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)

    controller.ask(
        prompt="remember this about me: I feel anxious about my health",
        session_id="default",
    )

    notes_root = tmp_path / "data" / "research_notes" / "default"
    assert not notes_root.exists() or list(notes_root.glob("*.json")) == []


def test_controller_explicitly_saves_personal_context_separately(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)

    controller.ask(
        prompt="remember this about me: I feel anxious about my health",
        session_id="default",
    )

    listing = controller.list_interactions(session_id="default")
    summary = controller.summarize_interactions(session_id="default")
    personal_root = tmp_path / "data" / "personal_memory" / "default"
    entry_files = list(personal_root.glob("*.json"))

    assert len(entry_files) == 1
    entry_payload = json.loads(entry_files[0].read_text(encoding="utf-8"))
    assert entry_payload["source_interaction_prompt"] == "remember this about me: I feel anxious about my health"
    assert entry_payload["memory_classification"]["candidate_type"] == "personal_context_candidate"
    assert listing["interaction_records"][0]["memory_write_decision"]["action"] == "save_personal_memory"
    assert summary["memory_write_action_counts"]["save_personal_memory"] == 1
    assert listing["interaction_records"][0]["personal_memory"]["entry_path"] == str(entry_files[0])
    assert listing["interaction_records"][0]["personal_memory"]["client_surface"] == "main"
    assert listing["interaction_records"][0]["research_note"] is None
    assert summary["personal_memory_saved_count"] == 1
    assert summary["research_note_count"] == 0
    assert summary["research_artifact_count"] == 0


def test_controller_saves_durable_conversational_preference_separately(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)

    controller.ask(
        prompt="from now on keep it brief with me",
        session_id="default",
    )

    listing = controller.list_interactions(session_id="default")
    summary = controller.summarize_interactions(session_id="default")

    assert listing["interaction_records"][0]["memory_classification"]["candidate_type"] == "personal_context_candidate"
    assert listing["interaction_records"][0]["memory_classification"]["explicit_save_requested"] is True
    assert listing["interaction_records"][0]["memory_write_decision"]["action"] == "save_personal_memory"
    assert summary["personal_memory_saved_count"] == 1


def test_controller_summarizes_mobile_surface_memory_blocking(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)

    controller.ask(
        prompt="create a migration plan for lumen routing",
        session_id="default",
        client_surface="mobile",
    )

    listing = controller.list_interactions(session_id="default")
    summary = controller.summarize_interactions(session_id="default")

    assert listing["interaction_records"][0]["memory_write_decision"]["action"] == "skip"
    assert listing["interaction_records"][0]["memory_write_decision"]["blocked_by_surface_policy"] is True
    assert summary["memory_write_action_counts"]["skip"] == 1
    assert summary["memory_surface_block_count"] == 1


def test_research_note_manager_skips_mobile_auto_save_by_default(tmp_path: Path) -> None:
    settings = AppSettings.from_repo_root(tmp_path)
    controller = AppController(repo_root=tmp_path)

    record = {
        "prompt": "create a migration plan for lumen routing",
        "mode": "planning",
        "kind": "planning.migration",
        "summary": "Grounded planning response for: create a migration plan for lumen routing",
        "normalized_topic": "migration plan for lumen routing",
        "dominant_intent": "planning",
        "interaction_profile": {},
        "memory_classification": {
            "candidate_type": "research_memory_candidate",
            "classification_confidence": 0.82,
            "save_eligible": True,
            "requires_explicit_user_consent": False,
            "reason": "Research-oriented interaction.",
        },
    }

    result = controller.interaction_log_manager.research_note_manager.maybe_record_note(
        session_id="default",
        timestamp=datetime.now(UTC),
        record=record,
        client_surface="mobile",
    )

    assert settings.mobile_research_note_auto_save is False
    assert result is None


def test_controller_summarizes_deep_validation_and_evidence_strength(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)
    controller.set_session_profile(
        "default",
        reasoning_depth="deep",
    )

    controller.ask(prompt="create a migration plan for lumen", session_id="default")

    summary = controller.summarize_interactions(session_id="default")
    listing = controller.list_interactions(session_id="default")

    assert summary["deep_validation_count"] == 1
    assert summary["deep_validation_ratio"] == 1.0
    assert summary["contradiction_signal_count"] >= 0
    assert summary["evidence_strength_counts"]
    assert summary["evidence_source_counts"]
    assert "missing_source_counts" in summary
    assert "contradiction_flag_counts" in summary
    assert listing["interaction_records"][0]["deep_validation_used"] is True
    assert listing["interaction_records"][0]["evidence_strength"] in {"supported", "strong", "light", "missing"}
    assert isinstance(listing["interaction_records"][0]["evidence_sources"], list)


def test_controller_summarizes_tool_route_origins(tmp_path: Path, monkeypatch) -> None:
    import shutil

    source_root = Path(__file__).resolve().parents[2]
    for relative in [
        Path("tool_bundles"),
        Path("tools"),
        Path("data") / "examples",
        Path("lumen.toml.example"),
    ]:
        src = source_root / relative
        dest = tmp_path / relative
        if src.is_dir():
            shutil.copytree(src, dest)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)

    controller = AppController(repo_root=tmp_path)

    def fake_run_tool(**kwargs):
        from lumen.tools.registry_types import ToolResult

        return ToolResult(
            status="ok",
            tool_id=str(kwargs["tool_id"]),
            capability=str(kwargs["capability"]),
            summary="Tool run completed",
        )

    monkeypatch.setattr(controller.tool_execution_service, "run_tool", fake_run_tool)

    controller.ask(prompt="report session confidence", session_id="default")
    controller.ask(prompt="confidence report for this session", session_id="default")

    summary = controller.summarize_interactions(session_id="default")
    listing = controller.list_interactions(session_id="default")

    assert summary["tool_route_origin_counts"]["exact_alias"] == 1
    assert summary["tool_route_origin_counts"]["nlu_hint_alias"] == 1
    assert {record["tool_route_origin"] for record in listing["interaction_records"]} == {
        "exact_alias",
        "nlu_hint_alias",
    }


def test_controller_summary_reports_clarification_frequency(tmp_path: Path, monkeypatch) -> None:
    controller = AppController(repo_root=tmp_path)

    controller.ask(prompt="create a migration plan for lumen", session_id="default")
    monkeypatch.setattr(
        controller.interaction_service.domain_router,
        "route",
        lambda *args, **kwargs: DomainRoute(
            mode="planning",
            kind="planning.migration",
            normalized_prompt="review the migration summary",
            confidence=0.74,
            reason="Planning and research cues were too close to resolve cleanly.",
            source="heuristic_planning",
            evidence=[],
            decision_summary={
                "selected": {},
                "alternatives": [
                    {
                        "candidate": {
                            "mode": "research",
                            "kind": "research.summary",
                            "source": "heuristic_research",
                            "confidence": 0.72,
                        }
                    }
                ],
                "ambiguous": True,
                "ambiguity_reason": "Top route candidates had very similar confidence and closely ranked source priority.",
            },
        ),
    )
    controller.ask(prompt="review the migration summary", session_id="default")

    summary = controller.summarize_interactions(session_id="default")
    session_report = controller.inspect_session("default")

    assert summary["clarification_count"] == 1
    assert summary["clarification_ratio"] == 0.5
    assert summary["clarification_trend"] == ["clarified", "clear"]
    assert summary["recent_clarification_mix"] == "clarification_heavy_mixed"
    assert summary["latest_clarification"] == "clarified"
    assert summary["clarification_drift"] == "increasing"
    assert session_report["clarification_count"] == 1
    assert session_report["clarification_ratio"] == 0.5
    assert session_report["clarification_trend"] == ["clarified", "clear"]
    assert session_report["recent_clarification_mix"] == "clarification_heavy_mixed"
    assert session_report["latest_clarification"] == "clarified"
    assert session_report["clarification_drift"] == "increasing"


def test_controller_can_report_interaction_patterns(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)

    controller.ask(prompt="create a migration plan for lumen", session_id="default")
    controller.ask(prompt="now compare that", session_id="default")
    controller.ask(prompt="what about that", session_id="default")

    report = controller.interaction_patterns(session_id="default")

    assert report["interaction_count"] == 3
    assert report["follow_up_count"] == 2
    assert report["ambiguous_follow_up_count"] == 0
    assert report["resolution_counts"]["compare_shorthand"] == 1
    assert report["status"] == "warn"


def test_interaction_patterns_ignore_command_like_prompts(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)

    controller.ask(prompt="lumen doctor", session_id="default")
    controller.ask(prompt="Hey Lumen, doctor", session_id="default")
    controller.ask(prompt="lumen session inspect default", session_id="default")
    controller.ask(prompt="lumen interaction summary --session-id default", session_id="default")

    report = controller.interaction_patterns(session_id="default")

    assert report["interaction_count"] == 4
    assert report["follow_up_count"] == 0
    assert report["ambiguous_follow_up_count"] == 0
    assert report["status"] == "ok"


def test_interaction_log_manager_sanitizes_legacy_records_on_load(tmp_path: Path) -> None:
    settings = AppSettings.from_repo_root(tmp_path)
    manager = InteractionLogManager(settings=settings)
    session_dir = settings.interactions_root / "default"
    session_dir.mkdir(parents=True, exist_ok=True)
    record_path = session_dir / "legacy.json"
    record_path.write_text(
        json.dumps(
            {
                "schema_type": "interaction_record",
                "schema_version": "1",
                "session_id": "default",
                "prompt": "what about that",
                "mode": "planning",
                "kind": "planning.migration",
                "summary": "Planning response for: what about the migration plan for lumen",
                "resolved_prompt": "what about the migration plan for lumen",
                "created_at": "2026-03-15T00:00:00+00:00",
                "context": {
                    "top_interaction_matches": [
                        {
                            "score": 1,
                            "record": {
                                "prompt": "create a migration plan for lumen",
                                "context": {
                                    "top_interaction_matches": [{"score": 2, "record": {"prompt": "nested"}}]
                                },
                                "response": {"schema_type": "assistant_response"},
                            },
                        }
                    ]
                },
                "response": {
                    "schema_type": "assistant_response",
                    "context": {
                        "top_interaction_matches": [
                            {
                                "score": 1,
                                "record": {
                                    "prompt": "create a migration plan for lumen",
                                    "response": {"schema_type": "assistant_response"},
                                },
                            }
                        ]
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    record = manager.load_record(record_path)

    nested = record["context"]["top_interaction_matches"][0]["record"]
    assert "response" not in nested
    assert "top_interaction_matches" not in nested.get("context", {})
    response_nested = record["response"]["context"]["top_interaction_matches"][0]["record"]
    assert "response" not in response_nested


def test_interaction_log_manager_skips_empty_partial_records_in_list_records(tmp_path: Path) -> None:
    settings = AppSettings.from_repo_root(tmp_path)
    manager = InteractionLogManager(settings=settings)
    session_dir = settings.interactions_root / "default"
    session_dir.mkdir(parents=True, exist_ok=True)

    valid_record = {
        "schema_type": "interaction_record",
        "schema_version": "3",
        "session_id": "default",
        "prompt": "what is entropy",
        "resolved_prompt": "what is entropy",
        "mode": "research",
        "kind": "research.summary",
        "summary": "Entropy is a measure of energy dispersion in a system.",
        "created_at": "2026-04-01T00:00:00+00:00",
        "context": {},
        "response": {
            "schema_type": "assistant_response",
            "mode": "research",
            "kind": "research.summary",
            "summary": "Entropy is a measure of energy dispersion in a system.",
        },
    }
    (session_dir / "valid.json").write_text(json.dumps(valid_record), encoding="utf-8")
    (session_dir / "partial.json").write_text("", encoding="utf-8")

    records = manager.list_records(session_id="default")

    assert len(records) == 1
    assert records[0]["prompt"] == "what is entropy"


def test_interaction_log_manager_skips_records_that_raise_memory_error(tmp_path: Path, monkeypatch) -> None:
    settings = AppSettings.from_repo_root(tmp_path)
    manager = InteractionLogManager(settings=settings)
    session_dir = settings.interactions_root / "default"
    session_dir.mkdir(parents=True, exist_ok=True)
    bad_path = session_dir / "too_large.json"
    bad_path.write_text("{}", encoding="utf-8")

    original_load_record = manager.load_record

    def _fake_load_record(path: Path) -> dict[str, object]:
        if path == bad_path:
            raise MemoryError("simulated oversized record")
        return original_load_record(path)

    monkeypatch.setattr(manager, "load_record", _fake_load_record)

    records = manager.list_records(session_id="default")

    assert records == []


def test_interaction_log_manager_skips_oversized_records_before_reading(tmp_path: Path, monkeypatch) -> None:
    settings = AppSettings.from_repo_root(tmp_path)
    manager = InteractionLogManager(settings=settings)
    session_dir = settings.interactions_root / "default"
    session_dir.mkdir(parents=True, exist_ok=True)
    oversized_path = session_dir / "oversized.json"
    oversized_path.write_text("{}", encoding="utf-8")

    original_stat = Path.stat
    original_load_record = manager.load_record

    def _fake_stat(path: Path, *args, **kwargs):
        result = original_stat(path, *args, **kwargs)
        if path == oversized_path:
            from os import stat_result

            values = list(result)
            values[6] = manager.max_record_bytes + 1
            return stat_result(values)
        return result

    def _guarded_load_record(path: Path) -> dict[str, object]:
        if path == oversized_path:
            raise AssertionError("oversized record should be skipped before read")
        return original_load_record(path)

    monkeypatch.setattr(Path, "stat", _fake_stat)
    monkeypatch.setattr(manager, "load_record", _guarded_load_record)

    records = manager.list_records(session_id="default")

    assert records == []


def test_interaction_log_manager_compacts_oversized_records_for_storage(tmp_path: Path, monkeypatch) -> None:
    settings = AppSettings.from_repo_root(tmp_path)
    manager = InteractionLogManager(settings=settings)
    oversized_json = "x" * (manager.max_record_bytes + 1)
    call_count = {"value": 0}

    def _fake_safe_json_dumps(payload: object) -> str | None:
        call_count["value"] += 1
        if call_count["value"] < 3:
            return oversized_json
        return json.dumps(payload, indent=2)

    monkeypatch.setattr(manager, "_safe_json_dumps", _fake_safe_json_dumps)

    prepared, serialized = manager._prepare_record_for_storage(
        {
            "session_id": "default",
            "prompt": "run anh file.fits",
            "mode": "tool",
            "kind": "tool.command_alias",
            "summary": "ANH completed",
            "context": {"huge": ["value"] * 100},
            "pipeline_observability": {"trace": ["value"] * 100},
            "pipeline_trace": {"steps": ["value"] * 100},
            "response": {
                "schema_type": "assistant_response",
                "mode": "tool",
                "kind": "tool.command_alias",
                "summary": "ANH completed",
                "tool_execution": {"tool_id": "anh", "capability": "spectral_dip_scan"},
                "tool_result": {"huge": ["value"] * 100},
            },
        }
    )

    assert prepared["context"] == {}
    assert prepared["pipeline_observability"] == {}
    assert prepared["pipeline_trace"] == {}
    assert prepared["response"]["summary"] == "ANH completed"
    assert isinstance(serialized, str)


def test_interaction_log_manager_reloads_compacted_trace_summaries_from_minimal_record(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = AppSettings.from_repo_root(tmp_path)
    manager = InteractionLogManager(settings=settings)
    oversized_json = "x" * (manager.max_record_bytes + 1)
    call_count = {"value": 0}

    def _fake_safe_json_dumps(payload: object) -> str | None:
        call_count["value"] += 1
        if call_count["value"] < 3:
            return oversized_json
        return json.dumps(payload, indent=2)

    monkeypatch.setattr(manager, "_safe_json_dumps", _fake_safe_json_dumps)

    manager.record_interaction(
        session_id="default",
        prompt="create a migration plan for lumen routing",
        response={
            "mode": "planning",
            "kind": "planning.migration",
            "summary": "Here is a workable plan for the routing work.",
            "trainability_trace": {
                "schema_version": "1",
                "rationale_summary": "x" * 25_000,
            },
            "supervised_support_trace": {
                "schema_version": "1",
                "enabled": True,
                "recommendations": {
                    "route_recommendation_support": {
                        "surface": "route_recommendation_support",
                        "recommended_mode": "planning",
                        "confidence": 0.82,
                        "applied": False,
                        "applied_reason": "x" * 25_000,
                    }
                },
            },
        },
    )

    records = manager.list_records(session_id="default")

    assert len(records) == 1
    assert records[0]["trainability_trace"]["compacted"] is True
    assert records[0]["trainability_trace"]["label"] == "trainability_trace"
    assert records[0]["supervised_support_trace"]["compacted"] is True
    assert records[0]["supervised_support_trace"]["label"] == "supervised_support_trace"


def test_interaction_records_do_not_persist_recursive_active_thread_snapshots(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)
    session_id = "qa-record-size"
    prompts = [
        "Hey Lumen!",
        "likewise, ive been thinking about mars",
        "space",
        "biology",
        "tell me about ww2",
        "im feeling sad today",
        "how was your day?",
        "tell me about the moon",
    ]

    for prompt in prompts:
        controller.ask(prompt=prompt, session_id=session_id)

    files = sorted((tmp_path / "data" / "interactions" / session_id).glob("*.json"))
    records = [path for path in files if path.name != "_index.json"]

    assert records
    assert max(path.stat().st_size for path in records) < 500 * 1024
    for path in records:
        text = path.read_text(encoding="utf-8")
        assert '"top_interaction_matches"' not in text
        assert text.count('"active_thread"') <= 1


def test_interaction_history_service_compacts_retrieval_context_matches(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)

    controller.ask(prompt="create a migration plan for lumen", session_id="default")
    controller.ask(prompt="what about that", session_id="default")

    context = controller.interaction_history_service.retrieve_context(
        "what about the migration plan",
        session_id="default",
    )
    record = context["top_interaction_matches"][0]["record"]

    assert record["prompt"] == "what about that"
    assert record["prompt_view"]["canonical_prompt"] == "what about the migration plan for lumen"
    assert record["dominant_intent"] == "research"
    assert "migration" in record["extracted_entities"]
    assert "context" not in record
    assert "response" not in record
    assert "matched_fields" in context["top_interaction_matches"][0]
    assert "score_breakdown" in context["top_interaction_matches"][0]


def test_interaction_search_can_use_nlu_metadata_in_results(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)

    controller.ask(prompt="design the routing architecture", session_id="default")

    search = controller.search_interactions("planning routing", session_id="default")

    assert search["interaction_count"] >= 1
    top_record = search["matches"][0]["record"]
    assert top_record["prompt"] == "design the routing architecture"
    assert top_record["dominant_intent"] == "planning"
    assert any(entity["value"] == "routing" for entity in top_record["extracted_entities"])


def test_interaction_search_can_use_semantic_overlap_for_paraphrased_query(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)

    controller.ask(prompt="create a migration plan for lumen routing", session_id="default")

    search = controller.search_interactions("plan lumen routing migration", session_id="default")

    assert search["interaction_count"] >= 1
    top_record = search["matches"][0]["record"]
    assert top_record["prompt"] == "create a migration plan for lumen routing"
    assert top_record["dominant_intent"] == "planning"
    assert top_record["normalized_topic"] == "migration plan for lumen routing"
    assert search["matches"][0]["score_breakdown"]["semantic_score"] > 0

    context = controller.interaction_history_service.retrieve_context(
        "plan lumen routing migration",
        session_id="default",
    )
    assert "semantic" in context["top_interaction_matches"][0]["matched_fields"]
    assert context["top_interaction_matches"][0]["score_breakdown"]["semantic_score"] > 0


def test_interaction_search_prefers_stronger_blended_match(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)

    controller.ask(prompt="create a migration plan for lumen routing", session_id="default")
    controller.ask(prompt="routing logs", session_id="default")

    search = controller.search_interactions("plan lumen routing migration", session_id="default")

    assert search["interaction_count"] >= 2
    assert search["matches"][0]["record"]["prompt"] == "create a migration plan for lumen routing"


def test_interaction_history_service_truncates_compact_context_fields(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)
    long_prompt = "what about " + ("the migration plan for lumen " * 20)

    controller.ask(prompt="create a migration plan for lumen", session_id="default")
    controller.ask(prompt=long_prompt, session_id="default")

    context = controller.interaction_history_service.retrieve_context(
        "migration plan",
        session_id="default",
    )
    record = context["top_interaction_matches"][0]["record"]

    assert len(record["prompt"]) <= 160
    assert len(record["summary"]) <= 160
    assert len(record["prompt_view"]["canonical_prompt"]) <= 160


def test_interaction_log_manager_search_uses_recursive_indexes_across_root(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)

    controller.ask(prompt="create a migration plan for lumen", session_id="default")

    search = controller.search_interactions("migration", session_id="default")

    assert search["interaction_count"] >= 1
    assert search["matches"][0]["record"]["prompt"] == "create a migration plan for lumen"


def test_interaction_log_manager_reports_index_status(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)

    controller.ask(prompt="create a migration plan for lumen", session_id="default")

    status = controller.interaction_log_manager.index_status(session_id="default")

    assert status["record_file_count"] == 1
    assert status["indexed_record_count"] == 1
    assert status["legacy_record_count"] == 0
    assert status["coverage_ratio"] == 1.0
