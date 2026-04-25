from pathlib import Path
import json

from lumen.tools.registry_types import BundleManifest, CapabilityManifest, ToolRequest

from tool_bundles.paper.adapters.compare_adapter import CompareAdapter
from tool_bundles.paper.adapters.extract_methods_adapter import ExtractMethodsAdapter
from tool_bundles.paper.adapters.search_adapter import SearchAdapter
from tool_bundles.paper.adapters.summary_adapter import SummaryAdapter


def _manifest() -> BundleManifest:
    return BundleManifest(
        id="paper",
        name="Paper Research Tools",
        version="0.1.0",
        entrypoint="bundle.py",
        capabilities=[
            CapabilityManifest(id="search", adapter="search_adapter"),
            CapabilityManifest(id="summary", adapter="summary_adapter"),
            CapabilityManifest(id="compare", adapter="compare_adapter"),
            CapabilityManifest(id="extract.methods", adapter="extract_methods_adapter"),
        ],
    )


def test_paper_summary_adapter_summarizes_text_file(tmp_path: Path) -> None:
    paper = tmp_path / "paper.txt"
    paper.write_text(
        "Abstract: This paper studies dark matter halos. Methods: We fit a simulation model to local observations. Results: The fit matches the strongest trend.",
        encoding="utf-8",
    )
    adapter = SummaryAdapter(manifest=_manifest(), repo_root=tmp_path)

    result = adapter.execute(ToolRequest(tool_id="paper", capability="summary", input_path=paper, session_id="paper-tests"))

    assert result.status == "ok"
    assert result.structured_data["abstract_excerpt"]


def test_paper_extract_methods_adapter_finds_methods_text(tmp_path: Path) -> None:
    paper = tmp_path / "paper.txt"
    paper.write_text(
        "Introduction: Context. Methods: We sampled 20 spectra and compared line profiles. Results: The method was stable.",
        encoding="utf-8",
    )
    adapter = ExtractMethodsAdapter(manifest=_manifest(), repo_root=tmp_path)

    result = adapter.execute(ToolRequest(tool_id="paper", capability="extract.methods", input_path=paper, session_id="paper-tests"))

    assert result.status == "ok"
    assert "sampled 20 spectra" in result.structured_data["methods_excerpt"].lower()


def test_paper_compare_adapter_compares_supplied_texts(tmp_path: Path) -> None:
    adapter = CompareAdapter(manifest=_manifest(), repo_root=tmp_path)
    request = ToolRequest(
        tool_id="paper",
        capability="compare",
        params={
            "papers": [
                "Abstract: Paper one. Methods: Method one. Results: Result one.",
                "Abstract: Paper two. Methods: Method two. Results: Result two.",
            ]
        },
        session_id="paper-tests",
    )

    result = adapter.execute(request)

    assert result.status == "ok"
    assert result.structured_data["paper_count"] == 2


def test_paper_search_adapter_reports_missing_source_cleanly(tmp_path: Path) -> None:
    adapter = SearchAdapter(manifest=_manifest(), repo_root=tmp_path)

    result = adapter.execute(ToolRequest(tool_id="paper", capability="search", params={"query": "dark matter"}, session_id="paper-tests"))

    assert result.status == "error"
    assert result.structured_data["failure_category"] == "runtime_dependency_failure"
    assert result.structured_data["runtime_diagnostics"]["provider_status"] == "source_unavailable"
