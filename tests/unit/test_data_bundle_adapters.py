from pathlib import Path
import json

from lumen.tools.registry_types import BundleManifest, CapabilityManifest, ToolRequest

from tool_bundles.data.adapters.cluster_adapter import ClusterAdapter
from tool_bundles.data.adapters.correlate_adapter import CorrelateAdapter
from tool_bundles.data.adapters.describe_adapter import DescribeAdapter
from tool_bundles.data.adapters.regression_adapter import RegressionAdapter
from tool_bundles.data.adapters.visualize_adapter import VisualizeAdapter


def _manifest() -> BundleManifest:
    return BundleManifest(
        id="data",
        name="Data Analysis Tools",
        version="0.1.0",
        entrypoint="bundle.py",
        capabilities=[
            CapabilityManifest(id="describe", adapter="describe_adapter"),
            CapabilityManifest(id="correlate", adapter="correlate_adapter"),
            CapabilityManifest(id="regression", adapter="regression_adapter"),
            CapabilityManifest(id="cluster", adapter="cluster_adapter"),
            CapabilityManifest(id="visualize", adapter="visualize_adapter"),
        ],
    )


def _write_dataset(path: Path) -> None:
    path.write_text("mass,velocity,label\n1,2,a\n2,4,b\n3,6,c\n4,8,d\n", encoding="utf-8")


def test_describe_adapter_summarizes_csv_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / "sample.csv"
    _write_dataset(dataset)
    adapter = DescribeAdapter(manifest=_manifest(), repo_root=tmp_path)

    result = adapter.execute(ToolRequest(tool_id="data", capability="describe", input_path=dataset, session_id="data-tests"))

    assert result.status == "ok"
    assert result.structured_data["row_count"] == 4
    assert "mass" in result.structured_data["numeric_columns"]


def test_correlate_adapter_finds_numeric_correlation(tmp_path: Path) -> None:
    dataset = tmp_path / "sample.csv"
    _write_dataset(dataset)
    adapter = CorrelateAdapter(manifest=_manifest(), repo_root=tmp_path)

    result = adapter.execute(ToolRequest(tool_id="data", capability="correlate", input_path=dataset, session_id="data-tests"))

    assert result.status == "ok"
    assert result.structured_data["pairs"][0]["left"] == "mass"
    assert result.structured_data["pairs"][0]["right"] == "velocity"


def test_regression_adapter_runs_linear_fit(tmp_path: Path) -> None:
    dataset = tmp_path / "sample.csv"
    _write_dataset(dataset)
    adapter = RegressionAdapter(manifest=_manifest(), repo_root=tmp_path)

    result = adapter.execute(ToolRequest(tool_id="data", capability="regression", input_path=dataset, session_id="data-tests"))

    assert result.status == "ok"
    assert round(float(result.structured_data["slope"]), 2) == 2.0
    assert result.structured_data["x_column"] == "mass"


def test_cluster_adapter_returns_clusters(tmp_path: Path) -> None:
    dataset = tmp_path / "sample.csv"
    _write_dataset(dataset)
    adapter = ClusterAdapter(manifest=_manifest(), repo_root=tmp_path)

    result = adapter.execute(ToolRequest(tool_id="data", capability="cluster", input_path=dataset, session_id="data-tests"))

    assert result.status == "ok"
    assert result.structured_data["cluster_count"] == 2
    assert len(result.structured_data["clusters"]) == 2


def test_visualize_adapter_writes_svg_artifact(tmp_path: Path) -> None:
    dataset = tmp_path / "sample.csv"
    _write_dataset(dataset)
    adapter = VisualizeAdapter(manifest=_manifest(), repo_root=tmp_path)

    result = adapter.execute(ToolRequest(tool_id="data", capability="visualize", input_path=dataset, session_id="data-tests"))

    assert result.status == "ok"
    assert any(artifact.name == "data_visualization.svg" for artifact in result.artifacts)

