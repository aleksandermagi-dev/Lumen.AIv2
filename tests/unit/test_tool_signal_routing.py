from pathlib import Path
import shutil

from lumen.app.controller import AppController
from lumen.routing.capability_manager import CapabilityManager
from lumen.routing.domain_router import DomainRouter
from lumen.routing.prompt_resolution import PromptResolver
from lumen.routing.tool_signal_catalog import DORMANT_TOOL_SIGNAL_CATALOG


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


def _capability_manager(tmp_path: Path) -> CapabilityManager:
    _copy_project_assets(tmp_path)
    controller = AppController(repo_root=tmp_path)
    return controller.capability_manager


def test_capability_manager_infers_math_signal_route(tmp_path: Path) -> None:
    manager = _capability_manager(tmp_path)

    match = manager.infer_by_signals("solve 2x + 5 = 13")

    assert match is not None
    assert match.capability_key == "math.solve_equation"
    assert match.match_source == "hybrid_signal"
    assert "contains_equation" in match.matched_patterns
    assert match.tool_intent_gate_passed is True


def test_capability_manager_infers_system_and_knowledge_signal_routes(tmp_path: Path) -> None:
    manager = _capability_manager(tmp_path)

    system_match = manager.infer_by_signals("analyze this system structure")
    refactor_match = manager.infer_by_signals("suggest a refactor for this architecture")
    docs_match = manager.infer_by_signals("generate docs for this architecture")
    knowledge_match = manager.infer_by_signals("find inconsistencies in these claims")
    content_match = manager.infer_by_signals("generate content batch for tiktok about black holes")
    data_match = manager.infer_by_signals("describe this dataset")
    paper_match = manager.infer_by_signals("search papers on dark matter")
    timeline_match = manager.infer_by_signals("render a timeline of these events")
    simulation_match = manager.infer_by_signals("simulate orbit with eccentricity 0.2")
    experiment_match = manager.infer_by_signals("design experiment to test whether light affects plant growth")
    invent_match = manager.infer_by_signals("generate concept for a lightweight propulsion system under these constraints: low mass, easy maintenance")
    physics_match = manager.infer_by_signals("model energy with mass 2 velocity 3 height 5")
    astronomy_match = manager.infer_by_signals("analyze orbit profile with semi-major axis 3 and eccentricity 0.2")

    assert system_match is not None
    assert system_match.capability_key == "system.analyze_architecture"
    assert refactor_match is not None
    assert refactor_match.capability_key == "system.suggest_refactor"
    assert docs_match is not None
    assert docs_match.capability_key == "system.generate_docs"
    assert knowledge_match is not None
    assert knowledge_match.capability_key == "knowledge.contradictions"
    assert content_match is not None
    assert content_match.capability_key == "content.generate_batch"
    assert data_match is not None
    assert data_match.capability_key == "data.describe"
    assert paper_match is not None
    assert paper_match.capability_key == "paper.search"
    assert timeline_match is not None
    assert timeline_match.capability_key == "viz.timeline"
    assert simulation_match is not None
    assert simulation_match.capability_key == "simulate.orbit"
    assert experiment_match is not None
    assert experiment_match.capability_key == "experiment.design"
    assert invent_match is not None
    assert invent_match.capability_key == "invent.generate_concepts"
    assert physics_match is not None
    assert physics_match.capability_key == "physics.energy_model"
    assert astronomy_match is not None
    assert astronomy_match.capability_key == "astronomy.orbit_profile"


def test_capability_manager_does_not_route_weak_or_reasoning_prompts(tmp_path: Path) -> None:
    manager = _capability_manager(tmp_path)

    assert manager.infer_by_signals("system") is None
    assert manager.infer_by_signals("matrix") is None
    assert manager.infer_by_signals("design me an engine") is None
    assert manager.infer_by_signals("explain black holes") is None
    assert manager.infer_by_signals("generate ideas for my engine startup") is None


def test_capability_manager_can_route_explicit_design_spec_alias_without_capturing_generic_design(tmp_path: Path) -> None:
    manager = _capability_manager(tmp_path)

    explicit_match = manager.infer_by_signals("generate system spec for lumen api workflow")
    generic_match = manager.infer_by_signals("design a workflow for lumen")

    assert explicit_match is not None
    assert explicit_match.capability_key == "design.system_spec"
    assert generic_match is None


def test_prompt_resolver_can_rewrite_prompt_from_hybrid_tool_signals(tmp_path: Path) -> None:
    manager = _capability_manager(tmp_path)
    resolver = PromptResolver(capability_manager=manager)

    resolution = resolver.resolve("solve 2x + 5 = 13", active_thread=None)

    assert resolution.changed is True
    assert resolution.strategy == "tool_signal_alias"
    assert resolution.resolved_prompt == "solve equation"


def test_domain_router_routes_hybrid_tool_signals_but_preserves_reasoning_guards(tmp_path: Path) -> None:
    manager = _capability_manager(tmp_path)
    router = DomainRouter(capability_manager=manager)

    math_route = router.route("solve 2x + 5 = 13")
    system_route = router.route("analyze this system structure")
    content_route = router.route("generate content ideas for tiktok about black holes")
    design_route = router.route("design me an engine")
    explain_route = router.route("explain black holes")
    workspace_route = router.route("show me the project structure")

    assert math_route.mode == "tool"
    assert math_route.source == "hybrid_signal"
    assert system_route.mode == "tool"
    assert system_route.source == "hybrid_signal"
    assert content_route.mode == "tool"
    assert content_route.source in {"hybrid_signal", "manifest_alias"}
    assert workspace_route.mode == "tool"
    assert workspace_route.source == "hybrid_signal"
    assert design_route.mode != "tool"
    assert explain_route.mode == "research"


def test_domain_router_prefers_exact_tool_alias_over_generic_summary_prompt_shape(tmp_path: Path) -> None:
    manager = _capability_manager(tmp_path)
    router = DomainRouter(capability_manager=manager)

    route = router.route("describe data")

    assert route.mode == "tool"
    assert route.kind == "tool.command_alias"
    assert route.source == "manifest_alias"


def test_dormant_signal_catalog_is_present_but_non_routable() -> None:
    assert DORMANT_TOOL_SIGNAL_CATALOG == {}
