from pathlib import Path
import shutil

from lumen.app.controller import AppController
from lumen.routing.domain_router import DomainRoute


def _copy_runtime_assets(repo_root: Path) -> None:
    source_root = Path(__file__).resolve().parents[2]
    for relative in [
        Path("tool_bundles"),
        Path("tools"),
        Path("data") / "examples",
        Path("lumen.toml.example"),
    ]:
        src = source_root / relative
        dest = repo_root / relative
        if src.is_dir():
            shutil.copytree(src, dest)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)


def test_doctor_reports_interaction_patterns(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)

    provider_check = next(
        item for item in controller.build_doctor_report()["checks"] if item["name"] == "model_provider"
    )
    assert provider_check["deployment_mode"] == "local_only"
    assert provider_check["provider_id"] == "local"
    assert provider_check["status"] == "ok"

    controller.ask(prompt="create a migration plan for lumen", session_id="default")
    controller.ask(prompt="now compare that", session_id="default")
    controller.ask(prompt="what about that", session_id="default")

    report = controller.build_doctor_report()
    interaction_check = next(
        item for item in report["checks"] if item["name"] == "interaction_patterns"
    )

    assert interaction_check["interaction_count"] == 3
    assert interaction_check["follow_up_count"] == 2
    assert interaction_check["ambiguous_follow_up_count"] == 0
    assert interaction_check["resolution_counts"]["compare_shorthand"] == 1
    assert interaction_check["resolution_counts"]["reference_follow_up"] == 1
    assert interaction_check["status"] == "warn"

    posture_check = next(
        item for item in report["checks"] if item["name"] == "confidence_posture"
    )
    assert posture_check["interaction_count"] == 3
    assert sum(posture_check["posture_counts"].values()) == 3
    assert posture_check["recent_posture_mix"] is not None
    assert posture_check["status"] in {"ok", "warn"}

    tool_route_check = next(
        item for item in report["checks"] if item["name"] == "tool_route_origins"
    )
    assert tool_route_check["tool_route_origin_counts"]["none"] == 3
    assert tool_route_check["status"] == "ok"

    nlu_check = next(
        item for item in report["checks"] if item["name"] == "nlu_signals"
    )
    assert nlu_check["detected_language_counts"]["en"] == 3
    assert nlu_check["dominant_intent_counts"]["planning"] == 1
    assert nlu_check["dominant_intent_counts"]["research"] == 2
    assert nlu_check["status"] == "ok"

    semantic_route_check = next(
        item for item in report["checks"] if item["name"] == "route_semantic_signals"
    )
    assert semantic_route_check["interaction_count"] == 3
    assert "semantic route reinforcement" in semantic_route_check["details"]
    assert semantic_route_check["status"] == "ok"

    normalized_score_check = next(
        item for item in report["checks"] if item["name"] == "route_normalized_scores"
    )
    assert normalized_score_check["interaction_count"] == 3
    assert normalized_score_check["route_normalized_score_count"] >= 1
    assert normalized_score_check["status"] == "ok"

    session_intent_check = next(
        item for item in report["checks"] if item["name"] == "route_session_intent"
    )
    assert session_intent_check["interaction_count"] == 3
    assert session_intent_check["status"] in {"ok", "warn"}

    retrieval_check = next(
        item for item in report["checks"] if item["name"] == "retrieval_ranking"
    )
    assert retrieval_check["retrieval_observation_count"] >= 0
    assert retrieval_check["status"] in {"ok", "warn"}

    retrieval_caution_check = next(
        item for item in report["checks"] if item["name"] == "route_retrieval_caution"
    )
    assert retrieval_caution_check["interaction_count"] == 3
    assert retrieval_caution_check["status"] in {"ok", "warn"}


def test_doctor_warns_when_hosted_provider_is_missing_api_key(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "lumen.toml").write_text(
        "\n".join(
            [
                "[app]",
                'deployment_mode = "hybrid"',
                'inference_provider = "openai_responses"',
                'openai_api_base = "https://api.openai.com/v1"',
                'openai_responses_model = "gpt-5"',
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    controller = AppController(repo_root=tmp_path)

    report = controller.build_doctor_report()
    provider_check = next(item for item in report["checks"] if item["name"] == "model_provider")

    assert provider_check["deployment_mode"] == "hybrid"
    assert provider_check["provider_id"] == "openai_responses"
    assert provider_check["default_model"] == "gpt-5"
    assert provider_check["api_key_present"] is False
    assert provider_check["status"] == "warn"


def test_doctor_reports_index_coverage(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)

    controller.ask(prompt="create a migration plan for lumen", session_id="default")

    report = controller.build_doctor_report()
    archive_index = next(item for item in report["checks"] if item["name"] == "archive_index")
    interaction_index = next(item for item in report["checks"] if item["name"] == "interaction_index")

    assert archive_index["indexed_record_count"] == archive_index["record_file_count"]
    assert archive_index["legacy_record_count"] == 0
    assert archive_index["status"] == "ok"

    assert interaction_index["indexed_record_count"] == interaction_index["record_file_count"]
    assert interaction_index["legacy_record_count"] == 0
    assert interaction_index["status"] == "ok"

    safety_check = next(item for item in report["checks"] if item["name"] == "safety_policy")
    assert safety_check["status"] == "ok"
    assert safety_check["bundle_policy"] == "registry_declared_only"
    assert safety_check["input_policy"] == "existing_file_when_provided"
    assert safety_check["prompt_policy_version"] == "v3"
    assert "weapons_explosives" in safety_check["hard_refuse_categories"]
    assert "self_modification" in safety_check["hard_refuse_categories"]

    human_check = next(item for item in report["checks"] if item["name"] == "human_thinking_layer_readiness")
    assert human_check["status"] == "warn"
    assert human_check["confirmed_gap_count"] >= 1
    assert human_check["targeted_implementation_count"] >= 1
    assert human_check["backlog_status_counts"]["future_roadmap"] >= 1

    capability_check = next(item for item in report["checks"] if item["name"] == "capability_contracts")
    assert capability_check["status"] == "ok"
    assert capability_check["status_counts"]["not_promised"] >= 1
    assert capability_check["status_counts"]["bounded"] >= 1

    overlap_check = next(item for item in report["checks"] if item["name"] == "behavioral_overlap_audit")
    assert overlap_check["status"] == "ok"
    assert overlap_check["gaps"] == []
    assert overlap_check["drift_findings"] == []
    assert overlap_check["contract_statuses"]["academic_writing"] == "bounded"
    assert overlap_check["contract_statuses"]["writing_editing"] == "provider_gated"


def test_human_thinking_layer_report_maps_dimensions_and_backlog(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)

    report = controller.human_thinking_layer_report()

    assert report["status"] == "warn"
    assert report["note_sources"]["audit_reference"]["path"].endswith("new additions and test\\test.md")
    assert report["note_sources"]["backlog_reference"]["path"].endswith("new additions and test\\plan ideas.md")
    dimensions = {item["dimension_id"]: item for item in report["audit_dimensions"]}
    assert dimensions["response_shaping_looseness"]["implementation_status"] == "present_but_weak"
    assert dimensions["context_continuity"]["implementation_status"] == "present"
    assert dimensions["srd_disruption_agency_trust"]["implementation_status"] == "partial"
    assert dimensions["intentional_tool_invocation"]["recommended_action"] == "leave"
    assert report["confirmed_gap_list"]
    assert report["targeted_implementation_list"]

    backlog = {item["domain_id"]: item for item in report["backlog_appendix"]["domains"]}
    assert backlog["content_generation"]["backlog_status"] == "partially_present"
    assert backlog["explainability_transparency"]["backlog_status"] == "already_present"
    assert backlog["speech_audio"]["backlog_status"] == "future_roadmap"


def test_capability_contract_report_marks_self_edit_as_not_promised(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)

    report = controller.capability_contract_report()

    contracts = {item["domain_id"]: item for item in report["contracts"]}
    assert contracts["self_modification"]["status"] == "not_promised"
    assert contracts["writing_editing"]["status"] == "provider_gated"
    assert contracts["dataset_analysis"]["status"] == "bounded"
    assert contracts["academic_writing"]["status"] == "bounded"
    assert contracts["citation_support"]["status"] == "bounded"
    assert contracts["supervised_ml_data_support"]["status"] == "bounded"


def test_academic_support_report_surfaces_bounded_workflows(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)

    report = controller.academic_support_report()

    workflows = {item["workflow"]: item for item in report["workflows"]}
    assert workflows["brainstorm"]["domain_id"] == "academic_writing"
    assert workflows["citation_help"]["domain_id"] == "citation_support"
    assert workflows["dataset_readiness"]["domain_id"] == "supervised_ml_data_support"


def test_behavioral_overlap_audit_detects_contract_drift(tmp_path: Path, monkeypatch) -> None:
    controller = AppController(repo_root=tmp_path)

    original_report = controller.diagnostics_service.build_capability_contract_report

    def _drifted_report():
        report = original_report()
        for item in report["contracts"]:
            if item["domain_id"] == "writing_editing":
                item["status"] = "bounded"
        return report

    monkeypatch.setattr(controller.diagnostics_service, "build_capability_contract_report", _drifted_report)

    audit = controller.diagnostics_service.build_behavioral_overlap_audit()

    assert audit["status"] == "error"
    assert audit["drift_findings"]


def test_doctor_reports_runtime_layout_and_core_bundle_readiness(tmp_path: Path) -> None:
    _copy_runtime_assets(tmp_path)
    data_root = tmp_path / "custom-data"
    controller = AppController(repo_root=tmp_path, data_root=data_root, execution_mode="frozen")

    report = controller.build_doctor_report()

    tool_registry_check = next(item for item in report["checks"] if item["name"] == "tool_registry")
    runtime_resources_check = next(item for item in report["checks"] if item["name"] == "runtime_resources")
    runtime_layout_check = next(item for item in report["checks"] if item["name"] == "runtime_layout")

    assert tool_registry_check["status"] == "ok"
    assert runtime_resources_check["status"] == "ok"
    assert runtime_resources_check["missing_required_resources"] == []
    assert runtime_resources_check["required_resources"]["tool_bundles"].endswith("tool_bundles")
    assert sorted(tool_registry_check["missing_bundles"]) == []
    assert set(tool_registry_check["required_bundles"]) == {
        "workspace",
        "report",
        "memory",
        "math",
        "system",
        "knowledge",
    }
    assert runtime_layout_check["execution_mode"] == "frozen"
    assert runtime_layout_check["runtime_root"] == str(tmp_path.resolve())
    assert runtime_layout_check["data_root"] == str(data_root.resolve())
    assert runtime_layout_check["bundle_count"] >= 6


def test_doctor_errors_when_required_runtime_bundles_are_missing(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path, execution_mode="frozen")

    report = controller.build_doctor_report()

    tool_registry_check = next(item for item in report["checks"] if item["name"] == "tool_registry")
    runtime_resources_check = next(item for item in report["checks"] if item["name"] == "runtime_resources")
    runtime_layout_check = next(item for item in report["checks"] if item["name"] == "runtime_layout")

    assert tool_registry_check["status"] == "error"
    assert runtime_resources_check["status"] == "error"
    assert runtime_resources_check["missing_required_resources"] == ["tool_bundles"]
    assert "math" in tool_registry_check["missing_bundles"]
    assert runtime_layout_check["status"] == "error"
    assert "workspace" in runtime_layout_check["missing_required_bundles"]


def test_doctor_warns_when_confidence_posture_is_mostly_tentative(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)

    controller.ask(prompt="hello there", session_id="default")
    controller.ask(prompt="hello again", session_id="default")
    controller.ask(prompt="what does this mean", session_id="default")

    report = controller.build_doctor_report()
    posture_check = next(
        item for item in report["checks"] if item["name"] == "confidence_posture"
    )

    assert posture_check["interaction_count"] == 3
    assert posture_check["posture_counts"]["tentative"] >= 2
    assert posture_check["status"] == "warn"


def test_doctor_reports_route_clarification_frequency(tmp_path: Path, monkeypatch) -> None:
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
    controller.ask(prompt="review the migration summary", session_id="default")

    report = controller.build_doctor_report()
    clarification_check = next(
        item for item in report["checks"] if item["name"] == "route_clarifications"
    )

    assert clarification_check["interaction_count"] == 3
    assert clarification_check["clarification_count"] == 2
    assert clarification_check["clarification_ratio"] == 0.6667
    assert clarification_check["clarification_trend"] == ["clarified", "clarified", "clear"]
    assert clarification_check["recent_clarification_mix"] == "clarification_heavy_mixed"
    assert clarification_check["latest_clarification"] == "clarified"
    assert clarification_check["clarification_drift"] == "increasing"
    assert clarification_check["status"] == "warn"


def test_doctor_warns_when_nlu_hint_tool_routing_outnumbers_exact_aliases(tmp_path: Path, monkeypatch) -> None:
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
    controller.ask(prompt="inspect the workspace structure", session_id="default")

    report = controller.build_doctor_report()
    tool_route_check = next(
        item for item in report["checks"] if item["name"] == "tool_route_origins"
    )

    assert tool_route_check["tool_route_origin_counts"]["exact_alias"] == 1
    assert tool_route_check["tool_route_origin_counts"]["nlu_hint_alias"] == 2
    assert tool_route_check["status"] == "warn"


def test_doctor_reports_memory_behavior_and_warns_on_unsaved_personal_context(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)

    controller.ask(prompt="I feel anxious about my health", session_id="default")

    report = controller.build_doctor_report()
    memory_check = next(
        item for item in report["checks"] if item["name"] == "memory_behavior"
    )

    assert memory_check["memory_classification_counts"]["personal_context_candidate"] == 1
    assert memory_check["memory_write_action_counts"]["skip"] == 1
    assert memory_check["personal_memory_saved_count"] == 0
    assert memory_check["explicit_memory_consent_count"] == 1
    assert memory_check["memory_surface_block_count"] == 0
    assert memory_check["status"] == "warn"


def test_doctor_reports_memory_behavior_for_explicit_personal_save(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)

    controller.ask(
        prompt="remember this about me: I feel anxious about my health",
        session_id="default",
    )

    report = controller.build_doctor_report()
    memory_check = next(
        item for item in report["checks"] if item["name"] == "memory_behavior"
    )

    assert memory_check["memory_classification_counts"]["personal_context_candidate"] == 1
    assert memory_check["memory_write_action_counts"]["save_personal_memory"] == 1
    assert memory_check["personal_memory_saved_count"] == 1
    assert memory_check["explicit_memory_consent_count"] == 1
    assert memory_check["memory_surface_block_count"] == 0
    assert memory_check["research_note_count"] == 0
    assert memory_check["status"] == "ok"


def test_doctor_reports_memory_behavior_for_promoted_research_artifacts(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)

    controller.ask(prompt="create a migration plan for lumen routing", session_id="default")
    notes_report = controller.list_research_notes(session_id="default")
    note_path = Path(notes_report["research_notes"][0]["note_path"])
    controller.promote_research_note(
        note_path=note_path,
        artifact_type="decision",
        title="Routing migration decision",
    )

    report = controller.build_doctor_report()
    memory_check = next(
        item for item in report["checks"] if item["name"] == "memory_behavior"
    )

    assert memory_check["research_note_count"] == 1
    assert memory_check["memory_write_action_counts"]["save_research_note"] == 1
    assert memory_check["research_artifact_count"] == 1
    assert memory_check["research_artifact_type_counts"]["decision"] == 1
    assert memory_check["status"] == "ok"


def test_doctor_reports_memory_behavior_for_mobile_surface_block(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)

    controller.ask(
        prompt="create a migration plan for lumen routing",
        session_id="default",
        client_surface="mobile",
    )

    report = controller.build_doctor_report()
    memory_check = next(
        item for item in report["checks"] if item["name"] == "memory_behavior"
    )

    assert memory_check["memory_classification_counts"]["research_memory_candidate"] == 1
    assert memory_check["memory_write_action_counts"]["skip"] == 1
    assert memory_check["memory_surface_block_count"] == 1
    assert memory_check["research_note_count"] == 0
    assert memory_check["status"] == "ok"


def test_doctor_report_survives_interaction_history_schema_errors(tmp_path: Path, monkeypatch) -> None:
    controller = AppController(repo_root=tmp_path)

    monkeypatch.setattr(
        controller.interaction_history_service,
        "summarize_patterns",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("Unsupported interaction_record schema_version '5'. Expected '3' or a known migration path.")
        ),
    )

    report = controller.build_doctor_report()
    interaction_check = next(
        item for item in report["checks"] if item["name"] == "interaction_patterns"
    )

    assert interaction_check["status"] == "error"
    assert "startup can continue" in interaction_check["details"]
    assert "Unsupported interaction_record schema_version '5'" in interaction_check["error"]
