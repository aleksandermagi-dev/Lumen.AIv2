from pathlib import Path

from lumen.tools.registry_types import BundleManifest, CapabilityManifest, ToolRequest

from tool_bundles.invent.adapters.constraint_check_adapter import ConstraintCheckAdapter
from tool_bundles.invent.adapters.failure_modes_adapter import FailureModesAdapter
from tool_bundles.invent.adapters.generate_concepts_adapter import GenerateConceptsAdapter
from tool_bundles.invent.adapters.material_suggestions_adapter import MaterialSuggestionsAdapter


def _manifest() -> BundleManifest:
    return BundleManifest(
        id="invent",
        name="Invent Tools",
        version="0.1.0",
        entrypoint="bundle.py",
        capabilities=[
            CapabilityManifest(id="generate_concepts", adapter="generate_concepts_adapter"),
            CapabilityManifest(id="constraint_check", adapter="constraint_check_adapter"),
            CapabilityManifest(id="material_suggestions", adapter="material_suggestions_adapter"),
            CapabilityManifest(id="failure_modes", adapter="failure_modes_adapter"),
        ],
    )


def test_generate_concepts_adapter_returns_concept_set(tmp_path: Path) -> None:
    adapter = GenerateConceptsAdapter(manifest=_manifest(), repo_root=tmp_path)

    result = adapter.execute(
        ToolRequest(
            tool_id="invent",
            capability="generate_concepts",
            params={"brief": "a lightweight propulsion concept", "constraints": ["low mass", "easy maintenance"]},
            session_id="invent-tests",
        )
    )

    assert result.status == "ok"
    assert result.structured_data["invent_type"] == "generate_concepts"
    assert len(result.structured_data["concepts"]) >= 3


def test_constraint_check_adapter_returns_assessments(tmp_path: Path) -> None:
    adapter = ConstraintCheckAdapter(manifest=_manifest(), repo_root=tmp_path)

    result = adapter.execute(
        ToolRequest(
            tool_id="invent",
            capability="constraint_check",
            params={"brief": "a lightweight propulsion concept", "constraints": ["low mass"]},
            session_id="invent-tests",
        )
    )

    assert result.status == "ok"
    assert result.structured_data["invent_type"] == "constraint_check"
    assert len(result.structured_data["assessments"]) >= 1


def test_material_suggestions_adapter_returns_material_classes(tmp_path: Path) -> None:
    adapter = MaterialSuggestionsAdapter(manifest=_manifest(), repo_root=tmp_path)

    result = adapter.execute(
        ToolRequest(
            tool_id="invent",
            capability="material_suggestions",
            params={"brief": "a lightweight propulsion concept"},
            session_id="invent-tests",
        )
    )

    assert result.status == "ok"
    assert result.structured_data["invent_type"] == "material_suggestions"
    assert len(result.structured_data["materials"]) >= 3


def test_failure_modes_adapter_returns_failure_mode_list(tmp_path: Path) -> None:
    adapter = FailureModesAdapter(manifest=_manifest(), repo_root=tmp_path)

    result = adapter.execute(
        ToolRequest(
            tool_id="invent",
            capability="failure_modes",
            params={"brief": "a lightweight propulsion concept"},
            session_id="invent-tests",
        )
    )

    assert result.status == "ok"
    assert result.structured_data["invent_type"] == "failure_modes"
    assert len(result.structured_data["failure_modes"]) >= 3
