from pathlib import Path

from lumen.tools.registry_types import BundleManifest, CapabilityManifest, ToolRequest

from tool_bundles.astronomy.adapters.orbit_profile_adapter import OrbitProfileAdapter
from tool_bundles.physics.adapters.energy_model_adapter import EnergyModelAdapter


def _physics_manifest() -> BundleManifest:
    return BundleManifest(
        id="physics",
        name="Physics Wrappers",
        version="0.1.0",
        entrypoint="bundle.py",
        capabilities=[CapabilityManifest(id="energy_model", adapter="energy_model_adapter")],
    )


def _astronomy_manifest() -> BundleManifest:
    return BundleManifest(
        id="astronomy",
        name="Astronomy Wrappers",
        version="0.1.0",
        entrypoint="bundle.py",
        capabilities=[CapabilityManifest(id="orbit_profile", adapter="orbit_profile_adapter")],
    )


def test_physics_energy_model_adapter_returns_energy_terms(tmp_path: Path) -> None:
    adapter = EnergyModelAdapter(manifest=_physics_manifest(), repo_root=tmp_path)

    result = adapter.execute(
        ToolRequest(
            tool_id="physics",
            capability="energy_model",
            params={"mass": 2, "velocity": 3, "height": 5},
            session_id="domain-wrapper-tests",
        )
    )

    assert result.status == "ok"
    assert result.structured_data["domain_type"] == "physics.energy_model"
    assert result.structured_data["kinetic_energy"] == 9.0


def test_astronomy_orbit_profile_adapter_returns_orbit_interpretation(tmp_path: Path) -> None:
    adapter = OrbitProfileAdapter(manifest=_astronomy_manifest(), repo_root=tmp_path)

    result = adapter.execute(
        ToolRequest(
            tool_id="astronomy",
            capability="orbit_profile",
            params={"semi_major_axis": 3, "eccentricity": 0.2},
            session_id="domain-wrapper-tests",
        )
    )

    assert result.status == "ok"
    assert result.structured_data["domain_type"] == "astronomy.orbit_profile"
    assert "periapsis" in result.structured_data["interpretation"]
