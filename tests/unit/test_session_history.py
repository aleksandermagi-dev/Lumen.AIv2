from pathlib import Path

from lumen.app.controller import AppController
from lumen.tools.registry_types import ToolResult


def test_session_inspect_includes_interaction_records(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)

    controller.ask(prompt="create a roadmap for developing lumen further", session_id="default")

    report = controller.inspect_session("default")

    assert report["session_id"] == "default"
    assert report["record_count"] == 0
    assert report["interaction_count"] == 1
    assert len(report["interaction_records"]) == 1
    assert report["interaction_records"][0]["mode"] == "planning"
    assert report["interaction_records"][0]["kind"].startswith("planning.")
    assert report["interaction_records"][0]["confidence_posture"] in {"supported", "conflicted", "tentative", "strong"}
    assert sum(report["posture_counts"].values()) == 1
    assert report["posture_trend"] == [report["interaction_records"][0]["confidence_posture"]]
    assert report["recent_posture_mix"] == f"stable:{report['interaction_records'][0]['confidence_posture']}"
    assert report["latest_posture"] == report["interaction_records"][0]["confidence_posture"]
    assert report["posture_drift"] == "insufficient_data"
    assert report["interaction_profile"]["interaction_style"] == "collab"
    assert report["interaction_profile"]["reasoning_depth"] == "normal"
    assert report["active_thread"]["mode"] == "planning"
    assert report["active_thread"]["prompt"] == "create a roadmap for developing lumen further"
    assert report["active_thread"]["objective"] == "Plan work for: create a roadmap for developing lumen further"
    assert report["active_thread"]["thread_summary"] in {
        "Here’s a workable first pass.",
        "Here’s a solid first plan.",
    }
    assert report["active_thread"]["confidence_posture"] == report["interaction_records"][0]["confidence_posture"]
    assert report["interaction_records"][0]["route_status"] in {"stable", "weakened", "under_tension", "revised", "unresolved", None}
    assert report["interaction_records"][0]["support_status"] in {"strongly_supported", "moderately_supported", "insufficiently_grounded", None}
    assert report["interaction_records"][0]["tension_status"] in {"stable", "under_tension", "revised", "unresolved", None}
    assert report["active_thread"]["local_context_assessment"] == report["interaction_records"][0]["local_context_assessment"]
    assert report["interaction_records"][0]["pipeline_observability"]["response_summary"]["package_type"] == "structured"
    assert report["active_thread"]["pipeline_observability"]["compacted"] is True
    assert report["active_thread"]["pipeline_observability"]["response_summary"]["package_type"] == "structured"
    assert report["interaction_records"][0]["pipeline_trace"]["reasoning_frame"]["frame_type"] == "plan-next-step"
    assert report["interaction_records"][0]["detected_language"] == "en"
    assert report["interaction_records"][0]["normalized_topic"] == "roadmap for developing lumen further"
    assert report["interaction_records"][0]["dominant_intent"] == "planning"
    assert report["interaction_records"][0]["local_context_assessment"] in {"partial", None}
    assert sum(report["local_context_assessment_counts"].values()) == 1
    assert report["coherence_topic_count"] == 0
    assert any(entity["value"] == "roadmap" for entity in report["interaction_records"][0]["extracted_entities"])
    assert report["interaction_records"][0]["interaction_profile"]["interaction_style"] == "collab"
    assert report["interaction_records"][0]["evidence_strength"] in {"supported", "strong", "light", "missing", None}
    assert isinstance(report["interaction_records"][0]["evidence_sources"], list)
    assert report["interaction_records"][0]["memory_classification"]["candidate_type"] == "research_memory_candidate"
    assert report["memory_classification_counts"]["research_memory_candidate"] == 1
    assert report["memory_save_eligible_count"] == 1
    assert report["explicit_memory_consent_count"] == 0
    assert report["personal_memory_saved_count"] == 0
    assert report["research_note_count"] == 1
    assert report["research_artifact_count"] == 0
    assert report["interaction_records"][0]["research_note"]["note_type"] == "chronological_research_note"
    assert report["interaction_records"][0]["deep_validation_used"] in {True, False}
    assert report["active_thread"]["detected_language"] == "en"
    assert report["active_thread"]["normalized_topic"] == "roadmap for developing lumen further"
    assert report["active_thread"]["dominant_intent"] == "planning"
    assert report["active_thread"]["interaction_profile"]["allow_suggestions"] is True
    assert any(entity["value"] == "roadmap" for entity in report["active_thread"]["extracted_entities"])
    assert report["active_thread"]["pipeline_trace"] == {}


def test_session_inspect_uses_session_profile_after_reset(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)

    controller.ask(prompt="create a roadmap for developing lumen further", session_id="default")
    controller.set_session_profile(
        "default",
        interaction_style="direct",
        reasoning_depth="deep",
        allow_suggestions=False,
    )
    controller.reset_session_thread("default")

    report = controller.inspect_session("default")

    assert report["interaction_profile"]["interaction_style"] == "direct"
    assert report["interaction_profile"]["reasoning_depth"] == "deep"
    assert report["interaction_profile"]["allow_suggestions"] is False
    assert report["active_thread"] is None


def test_session_inspect_surfaces_deep_validation_summary(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)
    controller.set_session_profile("default", reasoning_depth="deep")

    controller.ask(prompt="create a roadmap for developing lumen further", session_id="default")

    report = controller.inspect_session("default")

    assert report["deep_validation_count"] == 1
    assert report["deep_validation_ratio"] == 1.0
    assert report["evidence_strength_counts"]
    assert report["evidence_source_counts"]
    assert "missing_source_counts" in report
    assert report["contradiction_signal_count"] >= 0
    assert "contradiction_flag_counts" in report


def test_session_inspect_reports_promoted_research_artifacts(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)

    controller.ask(prompt="create a migration plan for lumen routing", session_id="default")
    notes_report = controller.list_research_notes(session_id="default")
    note_path = Path(notes_report["research_notes"][0]["note_path"])
    controller.promote_research_note(
        note_path=note_path,
        artifact_type="decision",
        title="Routing migration decision",
    )

    report = controller.inspect_session("default")

    assert report["research_note_count"] == 1
    assert report["research_artifact_count"] == 1
    assert report["research_artifact_type_counts"]["decision"] == 1
    assert len(report["interaction_records"][0]["research_note"]["promoted_artifacts"]) == 1


def test_session_inspect_persists_resolution_metadata(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)

    controller.ask(prompt="create a migration plan for lumen", session_id="default")
    controller.ask(prompt="now compare that", session_id="default")

    report = controller.inspect_session("default")
    latest = report["interaction_records"][0]

    assert latest["prompt"] == "now compare that"
    assert latest["resolved_prompt"].startswith("compare ")
    assert latest["resolution_strategy"] == "compare_shorthand"
    assert "comparison shorthand" in latest["resolution_reason"]
    assert latest["prompt_view"]["canonical_prompt"] == "compare the migration plan for lumen"
    assert latest["prompt_view"]["original_prompt"] == "now compare that"

    assert report["active_thread"]["prompt"] == "now compare that"
    assert report["active_thread"]["original_prompt"] == "now compare that"
    assert report["active_thread"]["objective"] == "Research topic: compare the migration plan for lumen"


def test_session_inspect_trims_nested_interaction_response_context(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)

    controller.ask(prompt="create a migration plan for lumen", session_id="default")
    controller.ask(prompt="what about that", session_id="default")
    controller.ask(prompt="summarize the current archive structure", session_id="default")

    report = controller.inspect_session("default")
    latest = report["interaction_records"][0]
    top_match = latest["context"]["top_interaction_matches"][0]["record"]

    assert latest["context"]["interaction_query_source"] is None
    assert "response" not in top_match
    assert top_match["prompt_view"]["canonical_prompt"] == "what about the migration plan for lumen"


def test_controller_ask_preserves_tool_resolution_metadata(tmp_path: Path, monkeypatch) -> None:
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
    sample_csv = tmp_path / "data" / "examples" / "cf4_ga_cone_template.csv"

    def fake_run_tool(**kwargs):
        from lumen.tools.registry_types import ToolResult

        return ToolResult(
            status="ok",
            tool_id="anh",
            capability="spectral_dip_scan",
            summary="GA Local Analysis Kit run completed",
        )

    monkeypatch.setattr(controller.tool_execution_service, "run_tool", fake_run_tool)

    controller.ask(prompt="run anh", session_id="default", input_path=sample_csv)
    response = controller.ask(prompt="run that again", session_id="default")

    assert response["mode"] == "tool"
    assert response["resolved_prompt"] == "run anh"
    assert response["resolution_strategy"] == "tool_repeat_shorthand"
    assert "repeat-style follow-up" in response["resolution_reason"]


def test_controller_ask_preserves_live_tool_answer_over_generic_tool_summary(
    tmp_path: Path, monkeypatch
) -> None:
    controller = AppController(repo_root=tmp_path)

    tool_result = ToolResult(
        status="ok",
        tool_id="math",
        capability="solve_equation",
        summary="Solved equation for x",
        structured_data={"variable": "x", "solution": ["8/3"]},
    )

    monkeypatch.setattr(
        controller.interaction_service,
        "ask",
        lambda **_: {
            "schema_type": "assistant_response",
            "schema_version": "1.0",
            "mode": "tool",
            "kind": "tool.math.solve_equation",
            "summary": "Solved equation for x: x = 8/3",
            "user_facing_answer": "Solved equation for x: x = 8/3",
            "result_summary": "Solved equation for x",
            "evidence_summary": "Solved equation for x",
            "tool_execution": {"tool_id": "math", "capability": "solve_equation"},
            "tool_result": tool_result,
        },
    )

    response = controller.ask(prompt="solve 3x = 8", session_id="default")

    assert response["summary"] == "Solved equation for x: x = 8/3"
    assert response["user_facing_answer"] == "Solved equation for x: x = 8/3"
    assert response["result_summary"] == "Solved equation for x"
    assert response["evidence_summary"] == "Solved equation for x"


def test_controller_ask_falls_back_to_tool_summary_when_no_live_tool_answer_exists(
    tmp_path: Path, monkeypatch
) -> None:
    controller = AppController(repo_root=tmp_path)

    tool_result = ToolResult(
        status="ok",
        tool_id="math",
        capability="solve_equation",
        summary="Solved equation for x",
        structured_data={"variable": "x", "solution": []},
    )

    monkeypatch.setattr(
        controller.interaction_service,
        "ask",
        lambda **_: {
            "schema_type": "assistant_response",
            "schema_version": "1.0",
            "mode": "tool",
            "kind": "tool.math.solve_equation",
            "summary": "",
            "tool_execution": {"tool_id": "math", "capability": "solve_equation"},
            "tool_result": tool_result,
        },
    )

    response = controller.ask(prompt="solve x + 2 = 11", session_id="default")

    assert response["summary"] == "Solved equation for x"
    assert "user_facing_answer" not in response


def test_session_inspect_includes_active_tool_context(tmp_path: Path, monkeypatch) -> None:
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
    sample_csv = tmp_path / "data" / "examples" / "cf4_ga_cone_template.csv"

    def fake_run_tool(**kwargs):
        from lumen.tools.registry_types import ToolResult

        return ToolResult(
            status="ok",
            tool_id="anh",
            capability="spectral_dip_scan",
            summary="GA Local Analysis Kit run completed",
        )

    monkeypatch.setattr(controller.tool_execution_service, "run_tool", fake_run_tool)

    controller.ask(prompt="run anh", session_id="default", input_path=sample_csv)

    report = controller.inspect_session("default")

    assert report["active_thread"]["prompt"] == "run anh"
    assert report["active_thread"]["tool_route_origin"] == "exact_alias"
    assert report["active_thread"]["tool_context"]["tool_id"] == "anh"
    assert report["active_thread"]["tool_context"]["capability"] == "spectral_dip_scan"
    assert report["active_thread"]["tool_context"]["input_path"] == str(sample_csv)


def test_session_inspect_reports_tool_route_origin_counts(tmp_path: Path, monkeypatch) -> None:
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

    report = controller.inspect_session("default")

    assert report["tool_route_origin_counts"]["exact_alias"] == 1
    assert report["tool_route_origin_counts"]["nlu_hint_alias"] == 1
    assert report["interaction_records"][0]["tool_route_origin"] in {"exact_alias", "nlu_hint_alias"}


def test_session_inspect_canonicalizes_tool_repeat_prompt(tmp_path: Path, monkeypatch) -> None:
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
    sample_csv = tmp_path / "data" / "examples" / "cf4_ga_cone_template.csv"

    def fake_run_tool(**kwargs):
        from lumen.tools.registry_types import ToolResult

        return ToolResult(
            status="ok",
            tool_id="anh",
            capability="spectral_dip_scan",
            summary="GA Local Analysis Kit run completed",
        )

    monkeypatch.setattr(controller.tool_execution_service, "run_tool", fake_run_tool)

    controller.ask(prompt="run anh", session_id="default", input_path=sample_csv)
    controller.ask(prompt="run that again", session_id="default")

    report = controller.current_session_thread("default")

    assert report["active_thread"]["prompt"] == "run that again"
    assert report["active_thread"]["objective"] == "Execute tool task: run anh"


