from pathlib import Path

from lumen.tools.registry_types import BundleManifest, CapabilityManifest, ToolRequest

from tool_bundles.simulate.adapters.diffusion_adapter import DiffusionAdapter
from tool_bundles.simulate.adapters.orbit_adapter import OrbitAdapter
from tool_bundles.simulate.adapters.population_adapter import PopulationAdapter
from tool_bundles.simulate.adapters.system_adapter import SystemAdapter


def _manifest() -> BundleManifest:
    return BundleManifest(
        id="simulate",
        name="Simulation Tools",
        version="0.1.0",
        entrypoint="bundle.py",
        capabilities=[
            CapabilityManifest(id="system", adapter="system_adapter"),
            CapabilityManifest(id="orbit", adapter="orbit_adapter"),
            CapabilityManifest(id="population", adapter="population_adapter"),
            CapabilityManifest(id="diffusion", adapter="diffusion_adapter"),
        ],
    )


def test_system_adapter_runs_default_system_simulation(tmp_path: Path) -> None:
    adapter = SystemAdapter(manifest=_manifest(), repo_root=tmp_path)

    result = adapter.execute(ToolRequest(tool_id="simulate", capability="system", session_id="sim-tests"))

    assert result.status == "ok"
    assert result.structured_data["simulation_type"] == "system"
    assert len(result.structured_data["series"]) >= 3


def test_orbit_adapter_generates_orbit_points(tmp_path: Path) -> None:
    adapter = OrbitAdapter(manifest=_manifest(), repo_root=tmp_path)

    result = adapter.execute(
        ToolRequest(
            tool_id="simulate",
            capability="orbit",
            params={"semi_major_axis": 2.0, "eccentricity": 0.2, "samples": 24},
            session_id="sim-tests",
        )
    )

    assert result.status == "ok"
    assert result.structured_data["simulation_type"] == "orbit"
    assert len(result.structured_data["points"]) == 24


def test_population_adapter_runs_logistic_growth(tmp_path: Path) -> None:
    adapter = PopulationAdapter(manifest=_manifest(), repo_root=tmp_path)

    result = adapter.execute(
        ToolRequest(
            tool_id="simulate",
            capability="population",
            params={"initial_population": 50, "growth_rate": 0.2, "carrying_capacity": 500, "steps": 12},
            session_id="sim-tests",
        )
    )

    assert result.status == "ok"
    assert result.structured_data["simulation_type"] == "population"
    assert result.structured_data["final_population"] > 50


def test_diffusion_adapter_writes_svg_artifact(tmp_path: Path) -> None:
    adapter = DiffusionAdapter(manifest=_manifest(), repo_root=tmp_path)

    result = adapter.execute(ToolRequest(tool_id="simulate", capability="diffusion", session_id="sim-tests"))

    assert result.status == "ok"
    assert result.structured_data["simulation_type"] == "diffusion"
    assert any(artifact.name == "diffusion_simulation.svg" for artifact in result.artifacts)
