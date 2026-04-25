from pathlib import Path

from lumen.schemas.output_schema import OUTPUT_SCHEMA_VERSION, OutputSchema
from lumen.tools.registry_types import Artifact, ToolResult


def test_tool_result_output_schema_includes_version_and_type() -> None:
    result = ToolResult(
        status="ok",
        tool_id="anh",
        capability="spectral_dip_scan",
        summary="done",
        artifacts=[
            Artifact(
                name="results.txt",
                path=Path("C:/tmp/results.txt"),
                media_type="text/plain",
            )
        ],
    )

    payload = OutputSchema.build_tool_result_payload(result)

    assert payload["schema_type"] == "tool_result"
    assert payload["schema_version"] == OUTPUT_SCHEMA_VERSION
    assert payload["tool_id"] == "anh"


def test_bundle_inspection_output_schema_includes_version_and_type() -> None:
    payload = OutputSchema.build_bundle_inspection_payload(
        bundle_id="anh",
        name="Astronomical Node Heuristics",
        version="0.1.0",
        schema_version="1",
        description="desc",
        manifest_path="C:/repo/tool_bundles/anh/manifest.json",
        entrypoint="bundle.py",
        capabilities=[],
    )

    assert payload["schema_type"] == "bundle_inspection"
    assert payload["schema_version"] == OUTPUT_SCHEMA_VERSION
    assert payload["bundle_schema_version"] == "1"

