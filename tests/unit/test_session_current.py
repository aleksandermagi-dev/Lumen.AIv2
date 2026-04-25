from pathlib import Path
import shutil

from lumen.app.controller import AppController
from lumen.app.models import InteractionProfile


def test_controller_returns_current_session_thread(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)

    controller.ask(prompt="create a migration plan for lumen", session_id="default")

    report = controller.current_session_thread("default")

    assert report["session_id"] == "default"
    assert report["active_thread"]["mode"] == "planning"
    assert report["active_thread"]["kind"] == "planning.migration"
    assert report["active_thread"]["objective"] == "Plan work for: create a migration plan for lumen"
    assert report["active_thread"]["original_prompt"] is None
    assert report["interaction_profile"]["interaction_style"] == "collab"
    assert report["interaction_profile"]["reasoning_depth"] == "normal"
    assert report["active_thread"]["pipeline_observability"]["compacted"] is True
    assert report["active_thread"]["pipeline_observability"]["response_summary"]["package_type"] == "structured"
    assert report["active_thread"]["pipeline_trace"] == {}


def test_current_session_thread_includes_tool_context(tmp_path: Path, monkeypatch) -> None:
    _copy_project_assets(tmp_path)
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

    report = controller.current_session_thread("default")

    assert report["active_thread"]["tool_route_origin"] == "exact_alias"
    assert report["active_thread"]["tool_context"]["tool_id"] == "anh"
    assert report["active_thread"]["tool_context"]["capability"] == "spectral_dip_scan"
    assert report["active_thread"]["tool_context"]["input_path"] == str(sample_csv)


def test_current_session_thread_preserves_original_prompt_when_resolved(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)

    controller.ask(prompt="create a migration plan for lumen", session_id="default")
    controller.ask(prompt="expand that further", session_id="default")

    report = controller.current_session_thread("default")

    assert report["active_thread"]["prompt"] == "expand that further"
    assert report["active_thread"]["original_prompt"] == "expand that further"


def test_current_session_thread_does_not_treat_loose_connective_prompt_as_follow_up(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)

    controller.ask(prompt="create a migration plan for lumen", session_id="default")
    controller.ask(prompt="also tell me about black holes", session_id="default")

    report = controller.current_session_thread("default")

    assert report["active_thread"]["mode"] == "research"
    assert report["active_thread"]["prompt"] == "also tell me about black holes"
    assert report["active_thread"]["objective"] == "Research topic: also tell me about black holes"
    assert report["active_thread"]["original_prompt"] is None


def test_current_session_thread_treats_addressed_continuation_as_follow_up(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)

    controller.ask(prompt="create a migration plan for lumen", session_id="default")
    controller.ask(prompt="Hey Lumen, expand that further", session_id="default")

    report = controller.current_session_thread("default")

    assert report["active_thread"]["original_prompt"] == "Hey Lumen, expand that further"
    assert "Plan work for: create a migration plan for lumen" in report["active_thread"]["objective"]


def test_session_profile_persists_without_active_thread(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)

    controller.session_context_service.set_interaction_profile(
        "default",
        InteractionProfile(
            interaction_style="direct",
            reasoning_depth="deep",
            selection_source="user",
            allow_suggestions=False,
        ),
    )

    report = controller.current_session_thread("default")

    assert report["active_thread"] is None
    assert report["interaction_profile"]["interaction_style"] == "direct"
    assert report["interaction_profile"]["reasoning_depth"] == "deep"
    assert report["interaction_profile"]["allow_suggestions"] is False


def test_session_reset_preserves_session_profile(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)

    controller.ask(prompt="create a migration plan for lumen", session_id="default")
    controller.set_session_profile(
        "default",
        interaction_style="direct",
        reasoning_depth="deep",
        allow_suggestions=False,
    )

    report = controller.reset_session_thread("default")

    assert report["active_thread"] is None
    assert report["interaction_profile"]["interaction_style"] == "direct"
    assert report["interaction_profile"]["reasoning_depth"] == "deep"
    assert report["interaction_profile"]["allow_suggestions"] is False


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


