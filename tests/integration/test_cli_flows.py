from __future__ import annotations

from io import StringIO
from pathlib import Path
import contextlib
import shutil
import sys

from lumen.cli.main import main


class _FakeANHAnalysisModule:
    def load_spectrum(self, path: str):
        print(f"Coverage: loaded {Path(path).name}")
        return [1392.0, 1393.0, 1402.0, 1403.0], [1.0, 0.9, 0.85, 1.0]

    @staticmethod
    def smooth(values, window: int):
        return values

    @staticmethod
    def zoom(wavelengths, flux, rest_wavelength, width=1.0, title=None, smooth_win=5):
        print(f"{title}: checked")
        return (rest_wavelength - 0.01, 0.12, -219.0)

    @staticmethod
    def plot_si_iv_window(wavelengths, flux):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 3))
        plt.plot(wavelengths, flux)
        plt.title("Si IV window")


def test_cli_init_and_doctor_text_output(tmp_path: Path) -> None:
    repo_root = tmp_path
    _copy_project_assets(repo_root)

    init_exit, init_stdout, init_stderr = _run_cli(
        ["--repo-root", str(repo_root), "--format", "text", "init"]
    )
    doctor_exit, doctor_stdout, doctor_stderr = _run_cli(
        ["--repo-root", str(repo_root), "--format", "text", "doctor"]
    )

    assert init_exit == 0
    assert "status: ok" in init_stdout
    assert str(repo_root) in init_stdout
    assert init_stderr == ""

    assert doctor_exit in {0, 1}
    assert "checks:" in doctor_stdout
    assert "tool_registry: ok" in doctor_stdout
    assert doctor_stderr == ""


def test_cli_run_outputs_json_result(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path
    _copy_project_assets(repo_root)
    sample_fits = repo_root / "data" / "examples" / "m31_x1d.fits"
    sample_fits.write_bytes(b"FAKEFITS")

    from tool_bundles.anh.adapters.anh_spectral_scan_adapter import ANHSpectralDipScanAdapter

    monkeypatch.setattr(ANHSpectralDipScanAdapter, "_load_analysis_module", lambda self: _FakeANHAnalysisModule())

    exit_code, stdout, stderr = _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "run",
            "anh.spectral_dip_scan",
            "--csv",
            str(sample_fits),
        ]
    )

    assert exit_code == 0
    assert '"status": "ok"' in stdout
    assert '"tool_id": "anh"' in stdout
    assert '"capability": "spectral_dip_scan"' in stdout
    assert stderr == ""


def test_cli_run_workspace_bundle_outputs_json_result(tmp_path: Path) -> None:
    repo_root = tmp_path
    _copy_project_assets(repo_root)

    exit_code, stdout, stderr = _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "run",
            "workspace.inspect.structure",
        ]
    )

    assert exit_code == 0
    assert '"status": "ok"' in stdout
    assert '"tool_id": "workspace"' in stdout
    assert '"capability": "inspect.structure"' in stdout
    assert stderr == ""


def test_cli_run_report_bundle_outputs_json_result(tmp_path: Path) -> None:
    repo_root = tmp_path
    _copy_project_assets(repo_root)

    _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "ask",
            "create a roadmap for developing lumen further",
            "--session-id",
            "default",
        ]
    )

    exit_code, stdout, stderr = _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "run",
            "report.session.confidence",
            "--session-id",
            "default",
        ]
    )

    assert exit_code == 0
    assert '"status": "ok"' in stdout
    assert '"tool_id": "report"' in stdout
    assert '"capability": "session.confidence"' in stdout
    assert stderr == ""


def test_cli_run_memory_bundle_outputs_json_result(tmp_path: Path) -> None:
    repo_root = tmp_path
    _copy_project_assets(repo_root)

    _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "ask",
            "create a roadmap for developing lumen further",
            "--session-id",
            "default",
        ]
    )

    exit_code, stdout, stderr = _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "run",
            "memory.session.timeline",
            "--session-id",
            "default",
        ]
    )

    assert exit_code == 0
    assert '"status": "ok"' in stdout
    assert '"tool_id": "memory"' in stdout
    assert '"capability": "session.timeline"' in stdout
    assert stderr == ""


def test_cli_command_outputs_text_result(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path
    _copy_project_assets(repo_root)
    sample_fits = repo_root / "data" / "examples" / "m31_x1d.fits"
    sample_fits.write_bytes(b"FAKEFITS")

    from tool_bundles.anh.adapters.anh_spectral_scan_adapter import ANHSpectralDipScanAdapter

    monkeypatch.setattr(ANHSpectralDipScanAdapter, "_load_analysis_module", lambda self: _FakeANHAnalysisModule())

    exit_code, stdout, stderr = _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "--format",
            "text",
            "command",
            "run",
            "anh",
            "--csv",
            str(sample_fits),
        ]
    )

    assert exit_code == 0
    assert "status: ok" in stdout
    assert "tool: anh" in stdout
    assert "capability: spectral_dip_scan" in stdout
    assert "ANH analyzed 1 file" in stdout
    assert stderr == ""


def test_cli_command_workspace_outputs_text_result(tmp_path: Path) -> None:
    repo_root = tmp_path
    _copy_project_assets(repo_root)

    exit_code, stdout, stderr = _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "--format",
            "text",
            "command",
            "inspect",
            "workspace",
        ]
    )

    assert exit_code == 0
    assert "status: ok" in stdout
    assert "tool: workspace" in stdout
    assert "capability: inspect.structure" in stdout
    assert stderr == ""


def test_cli_command_report_outputs_text_result(tmp_path: Path) -> None:
    repo_root = tmp_path
    _copy_project_assets(repo_root)

    _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "ask",
            "create a roadmap for developing lumen further",
            "--session-id",
            "default",
        ]
    )

    exit_code, stdout, stderr = _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "--format",
            "text",
            "command",
            "report",
            "session confidence",
            "--session-id",
            "default",
        ]
    )

    assert exit_code == 0
    assert "status: ok" in stdout
    assert "tool: report" in stdout
    assert "capability: session.confidence" in stdout
    assert stderr == ""


def test_cli_command_memory_outputs_text_result(tmp_path: Path) -> None:
    repo_root = tmp_path
    _copy_project_assets(repo_root)

    _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "ask",
            "create a roadmap for developing lumen further",
            "--session-id",
            "default",
        ]
    )

    exit_code, stdout, stderr = _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "--format",
            "text",
            "command",
            "inspect",
            "session timeline",
            "--session-id",
            "default",
        ]
    )

    assert exit_code == 0
    assert "status: ok" in stdout
    assert "tool: memory" in stdout
    assert "capability: session.timeline" in stdout
    assert stderr == ""


def test_cli_ask_workspace_routes_through_tool_bundle(tmp_path: Path) -> None:
    repo_root = tmp_path
    _copy_project_assets(repo_root)

    exit_code, stdout, stderr = _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "--format",
            "text",
            "ask",
            "inspect workspace",
            "--session-id",
            "default",
        ]
    )

    assert exit_code == 0
    assert "mode: tool" in stdout
    assert "summary: Workspace structure inspection completed" in stdout
    assert stderr == ""


def test_cli_ask_report_routes_through_tool_bundle(tmp_path: Path) -> None:
    repo_root = tmp_path
    _copy_project_assets(repo_root)

    _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "ask",
            "create a roadmap for developing lumen further",
            "--session-id",
            "default",
        ]
    )

    exit_code, stdout, stderr = _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "--format",
            "text",
            "ask",
            "report session confidence",
            "--session-id",
            "default",
        ]
    )

    assert exit_code == 0
    assert "mode: tool" in stdout
    assert "tool_route_origin: exact_alias" in stdout
    assert "tool: report" in stdout
    assert "capability: session.confidence" in stdout
    assert stderr == ""


def test_cli_ask_report_hint_alias_surfaces_tool_route_origin(tmp_path: Path) -> None:
    repo_root = tmp_path
    _copy_project_assets(repo_root)

    _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "ask",
            "create a roadmap for developing lumen further",
            "--session-id",
            "default",
        ]
    )

    exit_code, stdout, stderr = _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "--format",
            "text",
            "ask",
            "confidence report for this session",
            "--session-id",
            "default",
        ]
    )

    assert exit_code == 0
    assert "mode: tool" in stdout
    assert "tool_route_origin: nlu_hint_alias" in stdout
    assert "resolved_prompt: report session confidence" in stdout
    assert "tool: report" in stdout
    assert "capability: session.confidence" in stdout
    assert stderr == ""


def test_cli_ask_memory_routes_through_tool_bundle(tmp_path: Path) -> None:
    repo_root = tmp_path
    _copy_project_assets(repo_root)

    _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "ask",
            "create a roadmap for developing lumen further",
            "--session-id",
            "default",
        ]
    )

    exit_code, stdout, stderr = _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "--format",
            "text",
            "ask",
            "inspect session timeline",
            "--session-id",
            "default",
        ]
    )

    assert exit_code == 0
    assert "mode: tool" in stdout
    assert "tool: memory" in stdout
    assert "capability: session.timeline" in stdout
    assert stderr == ""


def test_cli_memory_note_and_promotion_flow(tmp_path: Path) -> None:
    repo_root = tmp_path
    _copy_project_assets(repo_root)

    _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "ask",
            "create a migration plan for lumen routing",
            "--session-id",
            "default",
        ]
    )

    notes_exit, notes_stdout, notes_stderr = _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "--format",
            "text",
            "memory",
            "notes",
            "--session-id",
            "default",
        ]
    )

    note_files = list((repo_root / "data" / "research_notes" / "default").glob("*.json"))
    assert len(note_files) == 1

    promote_exit, promote_stdout, promote_stderr = _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "--format",
            "text",
            "memory",
            "promote-note",
            str(note_files[0]),
            "--type",
            "decision",
            "--title",
            "Routing migration decision",
            "--reason",
            "Stable implementation checkpoint",
        ]
    )

    artifacts_exit, artifacts_stdout, artifacts_stderr = _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "--format",
            "text",
            "memory",
            "artifacts",
            "--session-id",
            "default",
        ]
    )

    assert notes_exit == 0
    assert "note_count: 1" in notes_stdout
    assert notes_stderr == ""

    assert promote_exit == 0
    assert "artifact_type: decision" in promote_stdout
    assert "title: Routing migration decision" in promote_stdout
    assert promote_stderr == ""

    assert artifacts_exit == 0
    assert "artifact_count: 1" in artifacts_stdout
    assert "decision | Routing migration decision" in artifacts_stdout
    assert artifacts_stderr == ""


def test_cli_repl_without_subcommand_routes_through_ask(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path
    _copy_project_assets(repo_root)

    prompts = iter(
        [
            "/status",
            "create a roadmap for developing lumen further",
            "/exit",
        ]
    )
    monkeypatch.setattr("builtins.input", lambda _: next(prompts))

    exit_code, stdout, stderr = _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "--format",
            "text",
        ]
    )

    assert exit_code == 0
    assert "Lumen interactive mode" in stdout
    assert "session_id: default" in stdout
    assert "mode: planning" in stdout
    assert stderr == ""


def test_cli_repl_status_shows_confidence_snapshot(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path
    _copy_project_assets(repo_root)

    prompts = iter(
        [
            "create a roadmap for developing lumen further",
            "/status",
            "/exit",
        ]
    )
    monkeypatch.setattr("builtins.input", lambda _: next(prompts))

    exit_code, stdout, stderr = _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "--format",
            "text",
        ]
    )

    assert exit_code == 0
    assert "interaction_style: collab" in stdout
    assert "reasoning_depth: normal" in stdout
    assert "allow_suggestions: True" in stdout
    assert "active_confidence_posture: supported" in stdout
    assert "latest_clarification:" in stdout
    assert "recent_clarification_mix:" in stdout
    assert "clarification_trend:" in stdout
    assert "clarification_drift:" in stdout
    assert "latest_posture:" in stdout
    assert "recent_posture_mix:" in stdout
    assert "posture_trend:" in stdout
    assert "posture_drift:" in stdout
    assert stderr == ""


def test_cli_can_list_and_search_interactions(tmp_path: Path) -> None:
    repo_root = tmp_path
    _copy_project_assets(repo_root)

    ask_exit, ask_stdout, ask_stderr = _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "ask",
            "create a roadmap for developing lumen further",
            "--session-id",
            "default",
        ]
    )
    list_exit, list_stdout, list_stderr = _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "--format",
            "text",
            "interaction",
            "list",
            "--session-id",
            "default",
        ]
    )
    search_exit, search_stdout, search_stderr = _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "--format",
            "text",
            "interaction",
            "search",
            "roadmap",
            "--session-id",
            "default",
        ]
    )

    assert ask_exit == 0
    assert '"mode": "planning"' in ask_stdout
    assert ask_stderr == ""

    assert list_exit == 0
    assert "interaction_count: 1" in list_stdout
    assert "create a roadmap for developing lumen further" not in list_stdout
    assert "Here’s a " in list_stdout
    assert list_stderr == ""

    assert search_exit == 0
    assert "query: roadmap" in search_stdout
    assert "create a roadmap for developing lumen further" in search_stdout
    assert search_stderr == ""


def test_cli_can_show_current_session_thread(tmp_path: Path) -> None:
    repo_root = tmp_path
    _copy_project_assets(repo_root)

    ask_exit, _, ask_stderr = _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "ask",
            "create a roadmap for developing lumen further",
            "--session-id",
            "default",
        ]
    )
    current_exit, current_stdout, current_stderr = _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "--format",
            "text",
            "session",
            "current",
            "default",
        ]
    )

    assert ask_exit == 0
    assert ask_stderr == ""

    assert current_exit == 0
    assert "session_id: default" in current_stdout
    assert "active_thread:" in current_stdout
    assert "interaction_style: collab" in current_stdout
    assert "reasoning_depth: normal" in current_stdout
    assert "confidence_posture: supported" in current_stdout
    assert "active_objective: Plan work for: create a roadmap for developing lumen further" in current_stdout
    assert current_stderr == ""


def test_cli_current_thread_shows_active_tool_context(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path
    _copy_project_assets(repo_root)
    sample_fits = repo_root / "data" / "examples" / "m31_x1d.fits"
    sample_fits.write_bytes(b"FAKEFITS")

    from tool_bundles.anh.adapters.anh_spectral_scan_adapter import ANHSpectralDipScanAdapter

    monkeypatch.setattr(ANHSpectralDipScanAdapter, "_load_analysis_module", lambda self: _FakeANHAnalysisModule())

    _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "ask",
            "run anh",
            "--csv",
            str(sample_fits),
            "--session-id",
            "default",
        ]
    )
    current_exit, current_stdout, current_stderr = _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "--format",
            "text",
            "session",
            "current",
            "default",
        ]
    )

    assert current_exit == 0
    assert "tool_route_origin: exact_alias" in current_stdout
    assert "active_tool: anh.spectral_dip_scan" in current_stdout
    assert f"active_tool_input: {sample_fits}" in current_stdout
    assert current_stderr == ""


def test_cli_can_reset_current_session_thread(tmp_path: Path) -> None:
    repo_root = tmp_path
    _copy_project_assets(repo_root)

    _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "ask",
            "create a roadmap for developing lumen further",
            "--session-id",
            "default",
        ]
    )
    reset_exit, reset_stdout, reset_stderr = _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "--format",
            "text",
            "session",
            "reset",
            "default",
        ]
    )

    assert reset_exit == 0
    assert "cleared: True" in reset_stdout
    assert "active_thread: <none>" in reset_stdout
    assert "interaction_style: collab" in reset_stdout
    assert reset_stderr == ""


def test_cli_can_show_and_update_session_profile(tmp_path: Path) -> None:
    repo_root = tmp_path
    _copy_project_assets(repo_root)

    show_exit, show_stdout, show_stderr = _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "--format",
            "text",
            "session",
            "profile",
            "default",
        ]
    )
    update_exit, update_stdout, update_stderr = _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "--format",
            "text",
            "session",
            "profile",
            "default",
            "--style",
            "direct",
            "--depth",
            "deep",
            "--allow-suggestions",
            "false",
        ]
    )
    current_exit, current_stdout, current_stderr = _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "--format",
            "text",
            "session",
            "current",
            "default",
        ]
    )

    assert show_exit == 0
    assert "interaction_style: collab" in show_stdout
    assert "reasoning_depth: normal" in show_stdout
    assert show_stderr == ""

    assert update_exit == 0
    assert "interaction_style: direct" in update_stdout
    assert "reasoning_depth: deep" in update_stdout
    assert "allow_suggestions: False" in update_stdout
    assert update_stderr == ""

    assert current_exit == 0
    assert "active_thread: <none>" in current_stdout
    assert "interaction_style: direct" in current_stdout
    assert "reasoning_depth: deep" in current_stdout
    assert "allow_suggestions: False" in current_stdout
    assert current_stderr == ""


def test_cli_can_summarize_interactions(tmp_path: Path) -> None:
    repo_root = tmp_path
    _copy_project_assets(repo_root)

    _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "ask",
            "create a migration plan for lumen",
            "--session-id",
            "default",
        ]
    )
    _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "ask",
            "now compare that",
            "--session-id",
            "default",
        ]
    )

    summary_exit, summary_stdout, summary_stderr = _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "--format",
            "text",
            "interaction",
            "summary",
            "--session-id",
            "default",
        ]
    )

    assert summary_exit == 0
    assert "clarification_count:" in summary_stdout
    assert "latest_clarification:" in summary_stdout
    assert "clarification_trend:" in summary_stdout
    assert "recent_clarification_mix:" in summary_stdout
    assert "clarification_drift:" in summary_stdout
    assert "posture_counts:" in summary_stdout
    assert "latest_posture:" in summary_stdout
    assert "posture_trend:" in summary_stdout
    assert "recent_posture_mix:" in summary_stdout
    assert "posture_drift:" in summary_stdout
    assert "detected_language_counts:" in summary_stdout
    assert "dominant_intent_counts:" in summary_stdout
    assert "evidence_strength_counts:" in summary_stdout
    assert "deep_validation_count:" in summary_stdout
    assert "deep_validation_ratio:" in summary_stdout
    assert "contradiction_signal_count:" in summary_stdout
    assert "recent_topics:" in summary_stdout
    assert "tool_route_origin_counts:" in summary_stdout
    assert "resolution_counts:" in summary_stdout
    assert "compare_shorthand: 1" in summary_stdout
    assert summary_stderr == ""


def test_cli_can_evaluate_interactions(tmp_path: Path) -> None:
    repo_root = tmp_path
    _copy_project_assets(repo_root)

    _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "ask",
            "create a migration plan for lumen",
            "--session-id",
            "default",
        ]
    )

    eval_exit, eval_stdout, eval_stderr = _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "--format",
            "text",
            "interaction",
            "evaluate",
            "--session-id",
            "default",
        ]
    )

    assert eval_exit == 0
    assert "evaluated_count:" in eval_stdout
    assert "surface_aggregates:" in eval_stdout
    assert "route_quality:" in eval_stdout
    assert "supervised_support_quality:" in eval_stdout
    assert eval_stderr == ""


def test_cli_can_export_labeled_examples(tmp_path: Path) -> None:
    repo_root = tmp_path
    _copy_project_assets(repo_root)

    _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "ask",
            "create a migration plan for lumen",
            "--session-id",
            "default",
        ]
    )

    export_exit, export_stdout, export_stderr = _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "--format",
            "text",
            "interaction",
            "export-labels",
            "--session-id",
            "default",
        ]
    )

    assert export_exit == 0
    assert "dataset_path:" in export_stdout
    assert "example_count:" in export_stdout
    assert "label_category_counts:" in export_stdout
    assert export_stderr == ""


def test_cli_can_derive_review_and_export_dataset_rows(tmp_path: Path) -> None:
    repo_root = tmp_path
    _copy_project_assets(repo_root)

    _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "ask",
            "what is entropy?",
            "--session-id",
            "default",
        ]
    )

    derive_exit, derive_stdout, derive_stderr = _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "--format",
            "text",
            "dataset",
            "derive-runtime-dataset",
            "lumen_conversation_v1",
            "--strategy",
            "derived_instruction_response",
            "--session-id",
            "default",
        ]
    )

    review_exit, review_stdout, review_stderr = _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "--format",
            "text",
            "dataset",
            "sample-dataset-review",
            "--dataset-name",
            "lumen_conversation_v1",
            "--limit",
            "5",
        ]
    )

    export_exit, export_stdout, export_stderr = _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "--format",
            "text",
            "dataset",
            "export-dataset-jsonl",
            "lumen_conversation_v1",
        ]
    )

    assert derive_exit == 0
    assert "example_count" in derive_stdout
    assert derive_stderr == ""
    assert review_exit == 0
    assert "review_count:" in review_stdout
    assert review_stderr == ""
    assert export_exit == 0
    assert "export_dir:" in export_stdout
    assert "example_count:" in export_stdout
    assert export_stderr == ""


def test_cli_can_report_persistence_status_and_coverage(tmp_path: Path) -> None:
    repo_root = tmp_path
    _copy_project_assets(repo_root)

    _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "ask",
            "create a roadmap for developing lumen further",
            "--session-id",
            "default",
        ]
    )

    status_exit, status_stdout, status_stderr = _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "--format",
            "text",
            "persistence",
            "status",
        ]
    )
    coverage_exit, coverage_stdout, coverage_stderr = _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "--format",
            "text",
            "persistence",
            "coverage",
        ]
    )
    doctor_exit, doctor_stdout, doctor_stderr = _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "--format",
            "text",
            "persistence",
            "doctor",
        ]
    )
    semantic_exit, semantic_stdout, semantic_stderr = _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "--format",
            "text",
            "persistence",
            "semantic-status",
        ]
    )
    backfill_exit, backfill_stdout, backfill_stderr = _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "--format",
            "text",
            "persistence",
            "backfill-embeddings",
            "--limit",
            "2",
        ]
    )

    assert status_exit == 0
    assert "db_path:" in status_stdout
    assert "migrations:" in status_stdout
    assert "table_counts:" in status_stdout
    assert status_stderr == ""

    assert coverage_exit == 0
    assert "db_counts:" in coverage_stdout
    assert "legacy_counts:" in coverage_stdout
    assert coverage_stderr == ""

    assert doctor_exit == 0
    assert "missing_rows:" in doctor_stdout
    assert "orphan_counts:" in doctor_stdout
    assert doctor_stderr == ""

    assert semantic_exit == 0
    assert "semantic_model_name:" in semantic_stdout
    assert "total_memory_items:" in semantic_stdout
    assert semantic_stderr == ""

    assert backfill_exit == 0
    assert "model_name:" in backfill_stdout
    assert "processed:" in backfill_stdout
    assert backfill_stderr == ""


def test_cli_can_filter_interactions_by_resolution_strategy(tmp_path: Path) -> None:
    repo_root = tmp_path
    _copy_project_assets(repo_root)

    _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "ask",
            "create a migration plan for lumen",
            "--session-id",
            "default",
        ]
    )
    _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "ask",
            "now compare that",
            "--session-id",
            "default",
        ]
    )

    list_exit, list_stdout, list_stderr = _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "--format",
            "text",
            "interaction",
            "list",
            "--session-id",
            "default",
            "--resolution-strategy",
            "compare_shorthand",
        ]
    )

    assert list_exit == 0
    assert "resolution_strategy: compare_shorthand" in list_stdout
    assert "prompt=compare the migration plan for lumen" in list_stdout
    assert "original=now compare that" in list_stdout
    assert "resolution=compare_shorthand" in list_stdout
    assert list_stderr == ""


def test_cli_can_report_interaction_patterns(tmp_path: Path) -> None:
    repo_root = tmp_path
    _copy_project_assets(repo_root)

    _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "ask",
            "create a migration plan for lumen",
            "--session-id",
            "default",
        ]
    )
    _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "ask",
            "now compare that",
            "--session-id",
            "default",
        ]
    )
    _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "ask",
            "what about that",
            "--session-id",
            "default",
        ]
    )

    patterns_exit, patterns_stdout, patterns_stderr = _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "--format",
            "text",
            "interaction",
            "patterns",
            "--session-id",
            "default",
        ]
    )

    assert patterns_exit == 0
    assert "follow_up_count: 2" in patterns_stdout
    assert "ambiguity_ratio: 0.0" in patterns_stdout
    assert "resolution_counts:" in patterns_stdout
    assert patterns_stderr == ""


def test_cli_repl_supports_current_thread_command(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path
    _copy_project_assets(repo_root)

    prompts = iter(
        [
            "create a roadmap for developing lumen further",
            "/current",
            "/exit",
        ]
    )
    monkeypatch.setattr("builtins.input", lambda _: next(prompts))

    exit_code, stdout, stderr = _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "--format",
            "text",
        ]
    )

    assert exit_code == 0
    assert "active_thread:" in stdout
    assert "active_objective: Plan work for: create a roadmap for developing lumen further" in stdout
    assert stderr == ""


def test_cli_repl_supports_reset_command(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path
    _copy_project_assets(repo_root)

    prompts = iter(
        [
            "create a roadmap for developing lumen further",
            "/reset",
            "/current",
            "/exit",
        ]
    )
    monkeypatch.setattr("builtins.input", lambda _: next(prompts))

    exit_code, stdout, stderr = _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "--format",
            "text",
        ]
    )

    assert exit_code == 0
    assert "cleared: True" in stdout
    assert "active_thread: <none>" in stdout
    assert stderr == ""


def test_cli_doctor_reports_interaction_patterns(tmp_path: Path) -> None:
    repo_root = tmp_path
    _copy_project_assets(repo_root)

    _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "ask",
            "create a migration plan for lumen",
            "--session-id",
            "default",
        ]
    )
    _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "ask",
            "now compare that",
            "--session-id",
            "default",
        ]
    )
    _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "ask",
            "what about that",
            "--session-id",
            "default",
        ]
    )

    doctor_exit, doctor_stdout, doctor_stderr = _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "--format",
            "text",
            "doctor",
        ]
    )

    assert doctor_exit in {0, 1}
    assert "interaction_patterns: warn" in doctor_stdout
    assert "confidence_posture:" in doctor_stdout
    assert "nlu_signals:" in doctor_stdout
    assert "tool_route_origins:" in doctor_stdout
    assert doctor_stderr == ""


def test_cli_archive_summary_surfaces_research_run_fields(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path
    _copy_project_assets(repo_root)
    sample_fits = repo_root / "data" / "examples" / "m31_x1d.fits"
    sample_fits.write_bytes(b"FAKEFITS")

    from tool_bundles.anh.adapters.anh_spectral_scan_adapter import ANHSpectralDipScanAdapter

    monkeypatch.setattr(ANHSpectralDipScanAdapter, "_load_analysis_module", lambda self: _FakeANHAnalysisModule())

    _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "ask",
            "run anh",
            "--csv",
            str(sample_fits),
            "--session-id",
            "default",
        ]
    )

    summary_exit, summary_stdout, summary_stderr = _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "--format",
            "text",
            "archive",
            "summary",
            "--session-id",
            "default",
        ]
    )

    assert summary_exit == 0
    assert "target_label_counts:" in summary_stdout
    assert "ANH Si IV Spectral Scan: 1" in summary_stdout
    assert "result_quality_counts:" in summary_stdout
    assert "candidate_dips_detected: 1" in summary_stdout
    assert summary_stderr == ""


def test_cli_archive_compare_groups_runs_within_capability(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path
    _copy_project_assets(repo_root)
    sample_fits = repo_root / "data" / "examples" / "m31_x1d.fits"
    sample_fits.write_bytes(b"FAKEFITS")

    from tool_bundles.anh.adapters.anh_spectral_scan_adapter import ANHSpectralDipScanAdapter

    monkeypatch.setattr(ANHSpectralDipScanAdapter, "_load_analysis_module", lambda self: _FakeANHAnalysisModule())

    _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "ask",
            "run anh",
            "--csv",
            str(sample_fits),
            "--session-id",
            "default",
        ]
    )
    _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "ask",
            "run anh",
            "--csv",
            str(sample_fits),
            "--session-id",
            "default",
        ]
    )

    compare_exit, compare_stdout, compare_stderr = _run_cli(
        [
            "--repo-root",
            str(repo_root),
            "--format",
            "text",
            "archive",
            "compare",
            "--session-id",
            "default",
            "--capability",
            "spectral_dip_scan",
        ]
    )

    assert compare_exit == 0
    assert "capability: spectral_dip_scan" in compare_stdout
    assert "target_count: 1" in compare_stdout
    assert "targets:" in compare_stdout
    assert "ANH Si IV Spectral Scan | runs=2" in compare_stdout
    assert "quality_distribution:" in compare_stdout
    assert "trend:" in compare_stdout
    assert "recent_runs:" in compare_stdout
    assert compare_stderr == ""


def _run_cli(args: list[str]) -> tuple[int, str, str]:
    stdout_buffer = StringIO()
    stderr_buffer = StringIO()

    original_argv = sys.argv
    try:
        sys.argv = ["lumen", *args]
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            exit_code = main()
    finally:
        sys.argv = original_argv

    return exit_code, stdout_buffer.getvalue(), stderr_buffer.getvalue()


def _fake_completed_run(amplitude: str):
    def fake_run(command, cwd, capture_output, text, check):
        out_dir = Path(command[command.index("--out") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "results.txt").write_text(
            "\n".join(
                [
                    f"Bulk-flow amplitude: {amplitude}",
                    "Apex (RA, Dec): (200.00°, -49.00°)",
                ]
            ),
            encoding="utf-8",
        )
        (out_dir / "apex_map.png").write_text("fake image", encoding="utf-8")

        class CompletedProcess:
            returncode = 0
            stdout = "Done.\nOutputs in: fake"
            stderr = ""

        return CompletedProcess()

    return fake_run


def _copy_project_assets(repo_root: Path) -> None:
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

