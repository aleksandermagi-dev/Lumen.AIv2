from pathlib import Path

from lumen.tools.registry_types import BundleManifest, CapabilityManifest, ToolRequest

from tool_bundles.experiment.adapters.analysis_plan_adapter import AnalysisPlanAdapter
from tool_bundles.experiment.adapters.controls_adapter import ControlsAdapter
from tool_bundles.experiment.adapters.design_adapter import DesignAdapter
from tool_bundles.experiment.adapters.variables_adapter import VariablesAdapter


def _manifest() -> BundleManifest:
    return BundleManifest(
        id="experiment",
        name="Experiment Tools",
        version="0.1.0",
        entrypoint="bundle.py",
        capabilities=[
            CapabilityManifest(id="design", adapter="design_adapter"),
            CapabilityManifest(id="variables", adapter="variables_adapter"),
            CapabilityManifest(id="controls", adapter="controls_adapter"),
            CapabilityManifest(id="analysis_plan", adapter="analysis_plan_adapter"),
        ],
    )


def test_design_adapter_builds_experiment_outline(tmp_path: Path) -> None:
    adapter = DesignAdapter(manifest=_manifest(), repo_root=tmp_path)

    result = adapter.execute(
        ToolRequest(
            tool_id="experiment",
            capability="design",
            params={"topic": "whether light affects plant growth"},
            session_id="experiment-tests",
        )
    )

    assert result.status == "ok"
    assert result.structured_data["experiment_type"] == "design"
    assert "light affects plant growth" in result.structured_data["topic"]


def test_variables_adapter_identifies_variable_structure(tmp_path: Path) -> None:
    adapter = VariablesAdapter(manifest=_manifest(), repo_root=tmp_path)

    result = adapter.execute(
        ToolRequest(
            tool_id="experiment",
            capability="variables",
            params={"topic": "whether light affects plant growth"},
            session_id="experiment-tests",
        )
    )

    assert result.status == "ok"
    assert result.structured_data["experiment_type"] == "variables"
    assert "dependent_variable" in result.structured_data


def test_controls_adapter_returns_control_guidance(tmp_path: Path) -> None:
    adapter = ControlsAdapter(manifest=_manifest(), repo_root=tmp_path)

    result = adapter.execute(
        ToolRequest(
            tool_id="experiment",
            capability="controls",
            params={"topic": "whether light affects plant growth"},
            session_id="experiment-tests",
        )
    )

    assert result.status == "ok"
    assert result.structured_data["experiment_type"] == "controls"
    assert len(result.structured_data["controls"]) >= 1


def test_analysis_plan_adapter_returns_analysis_steps(tmp_path: Path) -> None:
    adapter = AnalysisPlanAdapter(manifest=_manifest(), repo_root=tmp_path)

    result = adapter.execute(
        ToolRequest(
            tool_id="experiment",
            capability="analysis_plan",
            params={"topic": "whether light affects plant growth"},
            session_id="experiment-tests",
        )
    )

    assert result.status == "ok"
    assert result.structured_data["experiment_type"] == "analysis_plan"
    assert len(result.structured_data["analysis_steps"]) >= 3
