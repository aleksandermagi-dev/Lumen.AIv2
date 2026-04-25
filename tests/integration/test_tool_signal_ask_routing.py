from pathlib import Path
import shutil

import pytest

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


def test_controller_ask_routes_hybrid_math_prompt_through_tool_mode(tmp_path: Path) -> None:
    _copy_project_assets(tmp_path)
    controller = AppController(repo_root=tmp_path)

    response = controller.ask(prompt="solve 2x + 5 = 13", session_id="hybrid-tool")

    assert response["mode"] == "tool"
    assert response["tool_execution"]["tool_id"] == "math"
    assert response["tool_execution"]["capability"] == "solve_equation"
    assert response["tool_route_origin"] == "hybrid_signal"


def test_controller_ask_reports_missing_math_inputs_without_adapter_crash(tmp_path: Path) -> None:
    _copy_project_assets(tmp_path)
    controller = AppController(repo_root=tmp_path)

    response = controller.ask(prompt="solve equation", session_id="math-missing-inputs")

    assert response["mode"] == "tool"
    assert response["tool_execution_skipped"] is True
    assert response["tool_execution_skipped_reason"] == "missing_structured_inputs"
    assert response["runtime_diagnostic"]["failure_stage"] == "validation"
    assert response["runtime_diagnostic"]["failure_class"] == "input_failure"
    assert response["runtime_diagnostic"]["missing_inputs"] == "equation and variable"


def test_controller_ask_keeps_conceptual_derivative_prompt_out_of_tool_execution(tmp_path: Path) -> None:
    _copy_project_assets(tmp_path)
    controller = AppController(repo_root=tmp_path)

    response = controller.ask(prompt="what is the derivative of x^2", session_id="math-concept")

    assert response["mode"] == "research"
    assert response["kind"] == "research.academic_math_support"
    assert response.get("tool_execution") is None
    assert "derivative" in str(response.get("user_facing_answer") or response.get("summary") or "").lower()


def test_controller_ask_reports_true_math_runtime_failure_with_diagnostic(tmp_path: Path, monkeypatch) -> None:
    _copy_project_assets(tmp_path)
    controller = AppController(repo_root=tmp_path)

    def _boom(**kwargs):
        raise RuntimeError("simulated adapter failure")

    monkeypatch.setattr(controller.interaction_service.tool_execution_service, "run_tool", _boom)

    response = controller.ask(prompt="solve 2x + 5 = 13", session_id="math-runtime-failure")

    assert response["mode"] == "tool"
    assert response["runtime_diagnostic"]["failure_stage"] == "execution"
    assert response["runtime_diagnostic"]["failure_class"] == "runtime_dependency_failure"
    assert response["runtime_diagnostic"]["exception_type"] == "RuntimeError"
    assert "execution" in str(response.get("user_facing_answer") or response.get("summary") or "").lower()


def test_controller_ask_solves_natural_quadratic_prompt_end_to_end(tmp_path: Path) -> None:
    _copy_project_assets(tmp_path)
    controller = AppController(repo_root=tmp_path)

    response = controller.ask(prompt="Hey Lumen, solve 3x² + 2x - 5 = 0.", session_id="hybrid-tool")

    assert response["mode"] == "tool"
    assert response["summary"].startswith("Solved equation for x:")
    assert "Apply the quadratic formula." in response["summary"]
    assert response["tool_execution"]["tool_id"] == "math"
    assert response["tool_execution"]["capability"] == "solve_equation"
    assert response["tool_execution"]["params"]["equation"] == "3x² + 2x - 5 = 0"
    assert response["tool_execution"]["params"]["variable"] == "x"


def test_controller_ask_solves_natural_quadratic_prompt_without_trailing_punctuation(tmp_path: Path) -> None:
    _copy_project_assets(tmp_path)
    controller = AppController(repo_root=tmp_path)

    response = controller.ask(prompt="Hey Lumen, solve 3x² + 2x - 5 = 0", session_id="hybrid-tool")

    assert response["mode"] == "tool"
    assert response["summary"].startswith("Solved equation for x:")
    assert "Apply the quadratic formula." in response["summary"]
    assert response["tool_execution"]["tool_id"] == "math"
    assert response["tool_execution"]["capability"] == "solve_equation"
    assert response["tool_execution"]["params"]["equation"] == "3x² + 2x - 5 = 0"
    assert response["tool_execution"]["params"]["variable"] == "x"


def test_controller_ask_solves_high_degree_math_with_symbolic_fallback(tmp_path: Path) -> None:
    _copy_project_assets(tmp_path)
    controller = AppController(repo_root=tmp_path)

    response = controller.ask(prompt="solve x^4 - 10x^2 + 9 = 0", session_id="math-symbolic-degree")

    assert response["mode"] == "tool"
    assert response["tool_execution"]["tool_id"] == "math"
    assert response["tool_execution"]["capability"] == "solve_equation"
    assert "x = -3" in str(response.get("user_facing_answer") or response.get("summary") or "")
    assert "x = 3" in str(response.get("user_facing_answer") or response.get("summary") or "")


@pytest.mark.parametrize(
    ("prompt", "expected_fragment"),
    [
        ("solve (x+3)/(x-2) + (x-2)/(x+3) = 25/6", "sqrt(481)"),
        ("solve sqrt(x+6) + sqrt(3x-2) = 2x", "2.71584"),
    ],
)
def test_controller_ask_solves_symbolic_math_extensions(
    tmp_path: Path,
    prompt: str,
    expected_fragment: str,
) -> None:
    _copy_project_assets(tmp_path)
    controller = AppController(repo_root=tmp_path)

    response = controller.ask(prompt=prompt, session_id="math-symbolic-extension")

    assert response["mode"] == "tool"
    assert response["tool_execution"]["tool_id"] == "math"
    assert response["tool_execution"]["capability"] == "solve_equation"
    assert expected_fragment in str(response.get("user_facing_answer") or response.get("summary") or "")


def test_bundle_inspection_exposes_routing_signal_metadata(tmp_path: Path) -> None:
    _copy_project_assets(tmp_path)
    controller = AppController(repo_root=tmp_path)

    report = controller.inspect_bundle("math")

    capability = report["capabilities"][0]
    assert "trigger_keywords" in capability
    assert "structural_patterns" in capability
    assert "intent_hints" in capability


def test_controller_ask_extracts_claims_for_knowledge_contradictions(tmp_path: Path) -> None:
    _copy_project_assets(tmp_path)
    controller = AppController(repo_root=tmp_path)

    response = controller.ask(
        prompt="find inconsistencies in these claims: gravity is a force; gravity does not affect mass",
        session_id="hybrid-tool",
    )

    assert response["mode"] == "tool"
    assert response["tool_execution"]["tool_id"] == "knowledge"
    assert response["tool_execution"]["capability"] == "contradictions"
    assert response["tool_execution"]["params"]["claims"] == [
        "gravity is a force",
        "gravity does not affect mass",
    ]
    assert response.get("tool_execution_skipped") is not True


def test_controller_ask_extracts_items_for_knowledge_links(tmp_path: Path) -> None:
    _copy_project_assets(tmp_path)
    controller = AppController(repo_root=tmp_path)

    response = controller.ask(
        prompt="how do these relate: voltage, current, resistance",
        session_id="hybrid-tool",
    )

    assert response["mode"] == "tool"
    assert response["tool_execution"]["tool_id"] == "knowledge"
    assert response["tool_execution"]["capability"] == "link"
    assert response["tool_execution"]["params"]["items"] == [
        "voltage",
        "current",
        "resistance",
    ]
    assert response.get("tool_execution_skipped") is not True


def test_controller_ask_skips_knowledge_tool_when_structured_inputs_are_missing(tmp_path: Path) -> None:
    _copy_project_assets(tmp_path)
    controller = AppController(repo_root=tmp_path)

    response = controller.ask(
        prompt="find inconsistencies in this",
        session_id="hybrid-tool",
    )

    assert response["mode"] == "tool"
    assert response["tool_execution"]["tool_id"] == "knowledge"
    assert response["tool_execution"]["capability"] == "contradictions"
    assert response["tool_execution_skipped"] is True
    assert response["tool_execution_skipped_reason"] == "missing_structured_inputs"
    assert response["tool_missing_inputs"] == "claims"
    assert "didn't run the tool" in response["summary"].lower()


def test_controller_ask_uses_local_research_fallback_when_hosted_inference_is_unavailable(tmp_path: Path) -> None:
    _copy_project_assets(tmp_path)
    controller = AppController(repo_root=tmp_path)

    response = controller.ask(
        prompt="What is a black hole?",
        session_id="fallback-research",
    )

    assert response["mode"] == "research"
    assert response["kind"] == "research.summary"
    assert "black hole" in str(response["summary"]).lower()


def test_controller_ask_answers_expanded_astronomy_topic_from_local_knowledge(tmp_path: Path) -> None:
    _copy_project_assets(tmp_path)
    controller = AppController(repo_root=tmp_path)

    response = controller.ask(
        prompt="Tell me about the Great Attractor.",
        session_id="fallback-research",
    )

    assert response["mode"] == "research"
    assert "great attractor" in str(response["summary"]).lower()
    assert "gravitational influence" in str(response["summary"]).lower() or "galax" in str(response["summary"]).lower()


def test_controller_ask_answers_entropy_from_local_knowledge(tmp_path: Path) -> None:
    _copy_project_assets(tmp_path)
    controller = AppController(repo_root=tmp_path)

    response = controller.ask(
        prompt="what is entropy",
        session_id="fallback-research",
    )

    assert response["mode"] == "research"
    assert "entropy" in str(response["summary"]).lower()
    assert "spread out" in str(response["summary"]).lower()


def test_controller_ask_persists_reasoning_state_for_grounded_research_turn(tmp_path: Path) -> None:
    _copy_project_assets(tmp_path)
    controller = AppController(repo_root=tmp_path)

    controller.ask(
        prompt="what is entropy",
        session_id="reasoning-state",
    )

    active_thread = controller.current_session_thread("reasoning-state")["active_thread"]
    reasoning_state = active_thread["reasoning_state"]

    assert reasoning_state["current_path"] == "research:research.summary"
    assert reasoning_state["canonical_subject"] == "entropy"
    assert reasoning_state["resolved_prompt"] == "what is entropy"
    assert reasoning_state["selected_mode"] == "collab"
    assert reasoning_state["mode_behavior"]["mode"] == "collab"


def test_controller_ask_answers_entropy_deep_prompt_from_local_knowledge(tmp_path: Path) -> None:
    _copy_project_assets(tmp_path)
    controller = AppController(repo_root=tmp_path)

    response = controller.ask(
        prompt="explain entropy deeply",
        session_id="fallback-research",
    )

    assert response["mode"] == "research"
    assert "entropy" in str(response["summary"]).lower()
    assert "deeply" not in str(response["summary"]).lower()
    assert "don't have enough local knowledge on deeply" not in str(response["summary"]).lower()


def test_controller_ask_resolves_ga_short_alias_with_domain_hint(tmp_path: Path) -> None:
    _copy_project_assets(tmp_path)
    controller = AppController(repo_root=tmp_path)

    response = controller.ask(
        prompt="what is GA in astronomy",
        session_id="fallback-research",
    )

    assert response["mode"] == "research"
    assert "great attractor" in str(response["summary"]).lower()


def test_controller_ask_answers_expanded_chemistry_topic_from_local_knowledge(tmp_path: Path) -> None:
    _copy_project_assets(tmp_path)
    controller = AppController(repo_root=tmp_path)

    response = controller.ask(
        prompt="Explain the periodic table.",
        session_id="fallback-research",
    )

    assert response["mode"] == "research"
    assert "periodic table" in str(response["summary"]).lower()


def test_controller_ask_answers_expanded_biology_topic_from_local_knowledge(tmp_path: Path) -> None:
    _copy_project_assets(tmp_path)
    controller = AppController(repo_root=tmp_path)

    response = controller.ask(
        prompt="What is photosynthesis?",
        session_id="fallback-research",
    )

    assert response["mode"] == "research"
    assert "photosynthesis" in str(response["summary"]).lower()


def test_controller_ask_prefers_direct_answer_for_known_history_prompt_without_clarifying(tmp_path: Path) -> None:
    _copy_project_assets(tmp_path)
    controller = AppController(repo_root=tmp_path)

    response = controller.ask(
        prompt="Causes of the French Revolution",
        session_id="fallback-research",
    )

    assert response["mode"] == "research"
    assert response["kind"] == "research.summary"
    assert "french revolution" in str(response["summary"]).lower()


def test_controller_ask_answers_glossary_backed_combined_topic_from_local_knowledge(tmp_path: Path) -> None:
    _copy_project_assets(tmp_path)
    controller = AppController(repo_root=tmp_path)

    response = controller.ask(
        prompt="Weather vs Climate",
        session_id="fallback-research",
    )

    assert response["mode"] == "research"
    assert "weather" in str(response["summary"]).lower()
    assert "climate" in str(response["summary"]).lower()
    assert "tradeoff" not in str(response["summary"]).lower()


def test_controller_ask_keeps_grounded_comparison_in_research_lane(tmp_path: Path) -> None:
    _copy_project_assets(tmp_path)
    controller = AppController(repo_root=tmp_path)

    response = controller.ask(
        prompt="compare black hole and neutron star",
        session_id="fallback-research",
    )

    assert response["mode"] == "research"
    assert response["kind"] == "research.comparison"
    assert "black hole" in str(response["summary"]).lower()
    assert "neutron star" in str(response["summary"]).lower()
    assert "first pass" not in str(response["summary"]).lower()
    assert "next step" not in str(response["summary"]).lower()


def test_controller_ask_answers_relational_prompt_with_grounded_known_concepts(tmp_path: Path) -> None:
    _copy_project_assets(tmp_path)
    controller = AppController(repo_root=tmp_path)

    response = controller.ask(
        prompt="entropy in relation to black holes",
        session_id="fallback-research",
    )

    assert response["mode"] == "research"
    summary = str(response["summary"]).lower()
    assert "entropy" in summary
    assert "black hole" in summary
    assert "thermodynamic object" in summary or "broader physical theory" in summary
    assert "first pass" not in summary
    assert "next step" not in summary


def test_controller_ask_keeps_explicit_comparison_isolated_from_prior_topic(tmp_path: Path) -> None:
    _copy_project_assets(tmp_path)
    controller = AppController(repo_root=tmp_path)

    controller.ask(
        prompt="explain entropy deeply",
        session_id="comparison-isolation",
    )
    response = controller.ask(
        prompt="compare black hole and neutron star",
        session_id="comparison-isolation",
    )

    assert response["mode"] == "research"
    assert response["kind"] == "research.comparison"
    summary = str(response["summary"]).lower()
    assert "black hole" in summary
    assert "neutron star" in summary
    assert "entropy" not in summary
    assert "first pass" not in summary
    assert "next step" not in summary


def test_controller_ask_skips_content_batch_when_topic_is_missing(tmp_path: Path) -> None:
    _copy_project_assets(tmp_path)
    controller = AppController(repo_root=tmp_path)

    response = controller.ask(
        prompt="generate content batch",
        session_id="content-tools",
    )

    assert response["mode"] == "tool"
    assert response["tool_execution"]["tool_id"] == "content"
    assert response["tool_execution"]["capability"] == "generate_batch"
    assert response["tool_execution_skipped"] is True
    assert response["tool_missing_inputs"] == "topic"


def test_controller_ask_skips_content_ideas_when_topic_is_missing(tmp_path: Path) -> None:
    _copy_project_assets(tmp_path)
    controller = AppController(repo_root=tmp_path)

    response = controller.ask(
        prompt="generate content ideas",
        session_id="content-tools",
    )

    assert response["mode"] == "tool"
    assert response["tool_execution"]["tool_id"] == "content"
    assert response["tool_execution"]["capability"] == "generate_ideas"
    assert response["tool_execution_skipped"] is True
    assert response["tool_missing_inputs"] == "topic"


def test_controller_ask_surfaces_content_runtime_diagnosis_when_provider_is_missing(tmp_path: Path) -> None:
    _copy_project_assets(tmp_path)
    controller = AppController(repo_root=tmp_path)

    response = controller.ask(
        prompt="Generate me 5 content ideas on topic black holes",
        session_id="content-tools",
    )

    assert response["mode"] == "tool"
    assert response["tool_execution"]["tool_id"] == "content"
    assert response["tool_execution"]["capability"] == "generate_ideas"
    assert "provider" in str(response["summary"]).lower() or "configured" in str(response["summary"]).lower()
    structured = getattr(response.get("tool_result"), "structured_data", {})
    assert structured["failure_category"] == "missing_provider_config"


def test_controller_ask_routes_format_prompt_to_content_tool_and_surfaces_missing_input(tmp_path: Path) -> None:
    _copy_project_assets(tmp_path)
    controller = AppController(repo_root=tmp_path)

    response = controller.ask(
        prompt="format content for platform",
        session_id="content-tools",
    )

    assert response["mode"] == "tool"
    assert response["tool_execution"]["tool_id"] == "content"
    assert response["tool_execution"]["capability"] == "format_platform"
    assert response["tool_execution_skipped"] is True
    assert response["tool_missing_inputs"] == "source text or draft + platform"


def test_controller_ask_routes_explicit_anh_prompt_with_absolute_path(tmp_path: Path) -> None:
    _copy_project_assets(tmp_path)
    controller = AppController(repo_root=tmp_path)
    session_id = "anh-path-routing-explicit-path"
    sample = tmp_path / "fixtures" / "lb6f07nrq_x1d.fits"
    sample.parent.mkdir(parents=True, exist_ok=True)
    sample.write_bytes(b"FAKEFITS")

    response = controller.ask(
        prompt=f"run anh {sample}",
        session_id=session_id,
    )

    assert response["mode"] == "tool"
    assert response["tool_execution"]["tool_id"] == "anh"
    assert response["tool_execution"]["capability"] == "spectral_dip_scan"
    assert str(response["tool_execution"]["input_path"]).endswith("lb6f07nrq_x1d.fits")


def test_controller_ask_routes_generic_attached_spectral_file_prompt_to_anh(tmp_path: Path) -> None:
    _copy_project_assets(tmp_path)
    controller = AppController(repo_root=tmp_path)
    attached = tmp_path / "sample_x1d.fits"
    attached.write_text("placeholder", encoding="utf-8")

    response = controller.ask(
        prompt="Analyze this attached file and tell me what it is",
        input_path=attached,
        session_id="attached-file-routing",
    )

    assert response["mode"] == "tool"
    assert response["tool_execution"]["tool_id"] == "anh"
    assert response["tool_execution"]["capability"] == "spectral_dip_scan"


def test_controller_ask_routes_generic_attached_csv_prompt_to_data_describe(tmp_path: Path) -> None:
    _copy_project_assets(tmp_path)
    controller = AppController(repo_root=tmp_path)
    attached = tmp_path / "sample.csv"
    attached.write_text("mass,velocity\n1,2\n2,4\n", encoding="utf-8")

    response = controller.ask(
        prompt="Analyze this attached file and tell me what it is",
        input_path=attached,
        session_id="attached-data-routing",
    )

    assert response["mode"] == "tool"
    assert response["tool_execution"]["tool_id"] == "data"
    assert response["tool_execution"]["capability"] == "describe"


def test_controller_ask_routes_paper_search_prompt_to_paper_bundle(tmp_path: Path) -> None:
    _copy_project_assets(tmp_path)
    controller = AppController(repo_root=tmp_path)

    response = controller.ask(
        prompt="search papers on dark matter halos",
        session_id="paper-search-routing",
    )

    assert response["mode"] == "tool"
    assert response["tool_execution"]["tool_id"] == "paper"
    assert response["tool_execution"]["capability"] == "search"
    assert response["tool_execution"]["params"]["query"] == "on dark matter halos"


def test_controller_ask_routes_simulation_prompt_to_simulate_bundle(tmp_path: Path) -> None:
    _copy_project_assets(tmp_path)
    controller = AppController(repo_root=tmp_path)

    response = controller.ask(
        prompt="simulate orbit with semi-major axis 3 and eccentricity 0.2",
        session_id="simulation-routing",
    )

    assert response["mode"] == "tool"
    assert response["tool_execution"]["tool_id"] == "simulate"
    assert response["tool_execution"]["capability"] == "orbit"
    assert response["tool_execution"]["params"]["semi_major_axis"] == 3.0
    assert response["tool_execution"]["params"]["eccentricity"] == 0.2


def test_controller_ask_routes_experiment_prompt_to_experiment_bundle(tmp_path: Path) -> None:
    _copy_project_assets(tmp_path)
    controller = AppController(repo_root=tmp_path)

    response = controller.ask(
        prompt="design experiment to test whether light affects plant growth",
        session_id="experiment-routing",
    )

    assert response["mode"] == "tool"
    assert response["tool_execution"]["tool_id"] == "experiment"
    assert response["tool_execution"]["capability"] == "design"
    assert "light affects plant growth" in str(response["tool_execution"]["params"]["topic"]).lower()


def test_controller_ask_routes_invent_prompt_to_invent_bundle(tmp_path: Path) -> None:
    _copy_project_assets(tmp_path)
    controller = AppController(repo_root=tmp_path)

    response = controller.ask(
        prompt="generate concept for a lightweight propulsion system under these constraints: low mass, easy maintenance",
        session_id="invent-routing",
    )

    assert response["mode"] == "tool"
    assert response["tool_execution"]["tool_id"] == "invent"
    assert response["tool_execution"]["capability"] == "generate_concepts"
    assert "lightweight propulsion system" in str(response["tool_execution"]["params"]["brief"]).lower()


def test_controller_ask_routes_physics_prompt_to_physics_wrapper(tmp_path: Path) -> None:
    _copy_project_assets(tmp_path)
    controller = AppController(repo_root=tmp_path)

    response = controller.ask(
        prompt="model energy with mass 2 velocity 3 height 5",
        session_id="physics-routing",
    )

    assert response["mode"] == "tool"
    assert response["tool_execution"]["tool_id"] == "physics"
    assert response["tool_execution"]["capability"] == "energy_model"
    assert response["tool_execution"]["params"]["mass"] == 2.0


def test_controller_ask_routes_astronomy_wrapper_prompt_to_orbit_profile(tmp_path: Path) -> None:
    _copy_project_assets(tmp_path)
    controller = AppController(repo_root=tmp_path)

    response = controller.ask(
        prompt="analyze orbit profile with semi-major axis 3 and eccentricity 0.2",
        session_id="astronomy-routing",
    )

    assert response["mode"] == "tool"
    assert response["tool_execution"]["tool_id"] == "astronomy"
    assert response["tool_execution"]["capability"] == "orbit_profile"
    assert response["tool_execution"]["params"]["semi_major_axis"] == 3.0


def test_controller_ask_routes_natural_refactor_prompt_to_system_tool(tmp_path: Path) -> None:
    _copy_project_assets(tmp_path)
    controller = AppController(repo_root=tmp_path)

    response = controller.ask(
        prompt="suggest a refactor for this architecture",
        session_id="system-tools",
    )

    assert response["mode"] == "tool"
    assert response["tool_execution"]["tool_id"] == "system"
    assert response["tool_execution"]["capability"] == "suggest.refactor"
    assert response["tool_execution"]["params"]["target_path"] == "src"
    assert "refactor suggestions" in str(response["summary"]).lower()


def test_controller_ask_routes_natural_docs_prompt_to_system_tool(tmp_path: Path) -> None:
    _copy_project_assets(tmp_path)
    controller = AppController(repo_root=tmp_path)

    response = controller.ask(
        prompt="generate docs for this architecture",
        session_id="system-tools",
    )

    assert response["mode"] == "tool"
    assert response["tool_execution"]["tool_id"] == "system"
    assert response["tool_execution"]["capability"] == "generate.docs"
    assert response["tool_execution"]["params"]["target_path"] == "src"
    assert "generated" in str(response["summary"]).lower()


def test_controller_ask_routes_project_structure_prompt_to_workspace_tool(tmp_path: Path) -> None:
    _copy_project_assets(tmp_path)
    controller = AppController(repo_root=tmp_path)

    response = controller.ask(
        prompt="show me the project structure",
        session_id="workspace-tools",
    )

    assert response["mode"] == "tool"
    assert response["tool_execution"]["tool_id"] == "workspace"
    assert response["tool_execution"]["capability"] == "inspect.structure"


def test_controller_ask_routes_format_prompt_with_source_text_to_content_tool(tmp_path: Path) -> None:
    _copy_project_assets(tmp_path)
    controller = AppController(repo_root=tmp_path)

    response = controller.ask(
        prompt="format content for platform for tiktok: Routing drift starts with one small mismatch.",
        session_id="content-tools",
    )

    assert response["mode"] == "tool"
    assert response["tool_execution"]["tool_id"] == "content"
    assert response["tool_execution"]["capability"] == "format_platform"
    assert response["tool_execution"]["params"]["platform"] == "tiktok"
    assert "routing drift starts with one small mismatch" in str(
        response["tool_execution"]["params"]["source_text"]
    ).lower()


def test_controller_ask_preserves_black_hole_research_substance_across_modes(tmp_path: Path) -> None:
    _copy_project_assets(tmp_path)
    responses = []
    for style in ("default", "collab", "direct"):
        controller = AppController(repo_root=tmp_path)
        controller.set_session_profile("mode-check", interaction_style=style)
        responses.append(controller.ask(prompt="What is a black hole?", session_id="mode-check"))

    assert all(response["mode"] == "research" for response in responses)
    assert all(response["kind"] == "research.summary" for response in responses)
    assert all("collapsed region of spacetime" in str(response["summary"]).lower() for response in responses)
    assert all("escape speed exceeds the speed of light" in str(response["summary"]).lower() for response in responses)


def test_controller_ask_preserves_low_support_posture_across_modes(tmp_path: Path) -> None:
    _copy_project_assets(tmp_path)
    responses = []
    for style in ("default", "collab", "direct"):
        controller = AppController(repo_root=tmp_path)
        controller.set_session_profile("mode-check", interaction_style=style)
        responses.append(
            controller.ask(prompt="Tell me about hyperbolic lemon engines.", session_id="mode-check")
        )

    assert all(response["mode"] == "research" for response in responses)
    assert all(response["kind"] == "research.summary" for response in responses)
    assert all(
        "don't have enough local knowledge" in str(response["summary"]).lower()
        and "because" in str(response["summary"]).lower()
        for response in responses
    )
