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


def test_controller_lists_new_math_system_knowledge_and_design_bundles(tmp_path: Path) -> None:
    _copy_project_assets(tmp_path)
    controller = AppController(repo_root=tmp_path)

    tools = controller.list_tools()

    assert "math" in tools
    assert "system" in tools
    assert "knowledge" in tools
    assert "design" in tools
    assert "solve_equation" in tools["math"]
    assert "analyze.architecture" in tools["system"]
    assert "find_paths" in tools["knowledge"]
    assert "system_spec" in tools["design"]


def test_math_bundle_executes_equation_and_matrix_tools(tmp_path: Path) -> None:
    _copy_project_assets(tmp_path)
    controller = AppController(repo_root=tmp_path)

    solve_result = controller.run_tool(
        "math",
        "solve_equation",
        params={"equation": "2x + 3 = 11", "variable": "x"},
    )
    matrix_result = controller.run_tool(
        "math",
        "matrix_operations",
        params={"operation": "determinant", "matrix_a": [[1, 2], [3, 4]]},
    )

    assert solve_result.status == "ok"
    assert solve_result.structured_data["solution"] == ["4"]
    assert matrix_result.status == "ok"
    assert matrix_result.structured_data["result"] == -2


def test_math_bundle_executes_quadratic_equations_with_human_notation(tmp_path: Path) -> None:
    _copy_project_assets(tmp_path)
    controller = AppController(repo_root=tmp_path)

    caret_result = controller.run_tool(
        "math",
        "solve_equation",
        params={"equation": "3x^2 + 2x - 5 = 0", "variable": "x"},
    )
    superscript_result = controller.run_tool(
        "math",
        "solve_equation",
        params={"equation": "3x² + 2x - 5 = 0", "variable": "x"},
    )

    assert caret_result.status == "ok"
    assert caret_result.structured_data["solution"] == ["-1.66667", "1"]
    assert superscript_result.status == "ok"
    assert superscript_result.structured_data["solution"] == ["-1.66667", "1"]


def test_system_bundle_executes_architecture_analysis_and_refactor_suggestions(tmp_path: Path) -> None:
    _copy_project_assets(tmp_path)
    controller = AppController(repo_root=tmp_path)

    analysis = controller.run_tool(
        "system",
        "analyze.architecture",
        params={"target_path": "src/lumen/services", "depth": 1},
    )
    suggestions = controller.run_tool(
        "system",
        "suggest.refactor",
        params={"target_path": "src/lumen/services", "goal": "improve_tests"},
    )

    assert analysis.status == "ok"
    assert analysis.structured_data["file_count"] > 0
    assert suggestions.status == "ok"
    assert "safe_sequence" in suggestions.structured_data


def test_knowledge_bundle_executes_link_and_path_search(tmp_path: Path) -> None:
    _copy_project_assets(tmp_path)
    controller = AppController(repo_root=tmp_path)
    controller.graph_memory_manager.create_entities(
        [
            {"name": "Lumen", "entity_type": "project"},
            {"name": "Routing", "entity_type": "system"},
        ]
    )
    controller.graph_memory_manager.create_relations(
        [
            {"source": "Lumen", "source_type": "project", "relation_type": "depends_on", "target": "Routing", "target_type": "system"}
        ]
    )

    linked = controller.run_tool(
        "knowledge",
        "link",
        params={"items": ["Voltage", "Current", "Routing"]},
    )
    path_result = controller.run_tool(
        "knowledge",
        "find_paths",
        params={"source": "Lumen", "target": "Routing", "max_hops": 2},
    )

    assert linked.status == "ok"
    assert "links" in linked.structured_data
    assert path_result.status == "ok"
    assert path_result.structured_data["path_count"] >= 1


def test_design_bundle_generates_structured_system_spec(tmp_path: Path) -> None:
    _copy_project_assets(tmp_path)
    controller = AppController(repo_root=tmp_path)

    result = controller.run_tool(
        "design",
        "system_spec",
        params={"brief": "design an api workflow for lumen", "interaction_style": "default"},
    )

    assert result.status == "ok"
    assert result.structured_data["design_domain"] == "software_system"
    assert result.structured_data["components"]
    assert result.structured_data["tradeoffs"]
