from pathlib import Path
import json

from lumen.app.controller import AppController


def test_controller_lists_research_notes(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)

    controller.ask(prompt="create a migration plan for lumen routing", session_id="default")

    report = controller.list_research_notes(session_id="default")

    assert report["note_count"] == 1
    note = report["research_notes"][0]
    assert note["note_type"] == "chronological_research_note"
    assert note["source_interaction_prompt"] == "create a migration plan for lumen routing"


def test_controller_promotes_research_note_to_structured_artifact(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)

    controller.ask(prompt="create a migration plan for lumen routing", session_id="default")
    notes_report = controller.list_research_notes(session_id="default")
    note_path = Path(notes_report["research_notes"][0]["note_path"])

    promotion = controller.promote_research_note(
        note_path=note_path,
        artifact_type="decision",
        title="Routing migration decision",
        promotion_reason="This note captures a stable implementation decision.",
    )
    artifacts_report = controller.list_research_artifacts(session_id="default")

    assert promotion["status"] == "ok"
    artifact = promotion["artifact"]
    assert artifact["artifact_type"] == "decision"
    assert artifact["title"] == "Routing migration decision"
    assert Path(artifact["artifact_path"]).exists()
    assert artifacts_report["artifact_count"] == 1
    assert artifacts_report["research_artifacts"][0]["source_note_path"] == str(note_path)

    refreshed_note = json.loads(note_path.read_text(encoding="utf-8"))
    assert len(refreshed_note["promoted_artifacts"]) == 1
    assert refreshed_note["promoted_artifacts"][0]["artifact_type"] == "decision"


def test_controller_summary_counts_promoted_research_artifacts(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)

    controller.ask(prompt="create a migration plan for lumen routing", session_id="default")
    notes_report = controller.list_research_notes(session_id="default")
    note_path = Path(notes_report["research_notes"][0]["note_path"])
    controller.promote_research_note(
        note_path=note_path,
        artifact_type="decision",
        title="Routing migration decision",
    )

    summary = controller.summarize_interactions(session_id="default")
    session_report = controller.inspect_session("default")
    listing = controller.list_interactions(session_id="default")

    assert summary["research_note_count"] == 1
    assert summary["research_artifact_count"] == 1
    assert summary["research_artifact_type_counts"]["decision"] == 1
    assert session_report["research_artifact_count"] == 1
    assert session_report["research_artifact_type_counts"]["decision"] == 1
    assert len(listing["interaction_records"][0]["research_note"]["promoted_artifacts"]) == 1


def test_controller_promote_research_note_rejects_unknown_artifact_type(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)

    controller.ask(prompt="create a migration plan for lumen routing", session_id="default")
    notes_report = controller.list_research_notes(session_id="default")
    note_path = Path(notes_report["research_notes"][0]["note_path"])

    try:
        controller.promote_research_note(
            note_path=note_path,
            artifact_type="unknown",
        )
    except ValueError as exc:
        assert "Unsupported artifact_type" in str(exc)
    else:
        raise AssertionError("Expected promote_research_note to reject unknown artifact types")
