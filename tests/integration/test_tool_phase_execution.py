from pathlib import Path
import shutil

from lumen.app.controller import AppController


def _copy_project_assets(repo_root: Path) -> None:
    source_root = Path(__file__).resolve().parents[2]
    for relative in [Path("tool_bundles"), Path("tools"), Path("src"), Path("lumen.toml.example")]:
        src = source_root / relative
        dest = repo_root / relative
        if src.is_dir():
            shutil.copytree(src, dest)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)


def test_controller_run_command_routes_new_bundle_aliases(tmp_path: Path) -> None:
    _copy_project_assets(tmp_path)
    controller = AppController(repo_root=tmp_path)

    result = controller.run_command(
        action="analyze",
        target="architecture",
        params={"target_path": "src/lumen/services", "depth": 1},
        session_id="tool-phase",
    )

    assert result.status == "ok"
    assert result.tool_id == "system"
    assert result.capability == "analyze.architecture"
    assert result.archive_path is not None
    assert result.archive_path.exists()


def test_controller_run_tool_executes_math_and_knowledge_with_structured_params(tmp_path: Path) -> None:
    _copy_project_assets(tmp_path)
    controller = AppController(repo_root=tmp_path)

    math_result = controller.run_tool(
        "math",
        "optimize_function",
        params={
            "expression": "-(x^2) + 4x",
            "variable": "x",
            "bounds": {"min": 0, "max": 5},
            "objective": "maximize",
            "samples": 21,
        },
        session_id="tool-phase",
    )
    knowledge_result = controller.run_tool(
        "knowledge",
        "contradictions",
        params={
            "claims": [
                "The system is safe for human use.",
                "The system is not safe for human use.",
            ],
            "strictness": "high",
        },
        session_id="tool-phase",
    )

    assert math_result.status == "ok"
    assert math_result.structured_data["best_input"] is not None
    assert knowledge_result.status == "ok"
    assert knowledge_result.structured_data["contradictions"]
