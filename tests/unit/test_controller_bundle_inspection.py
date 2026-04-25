from pathlib import Path

from lumen.app.controller import AppController


def test_controller_can_inspect_bundle_manifest(tmp_path: Path) -> None:
    source_root = Path(__file__).resolve().parents[2]
    tool_bundles_src = source_root / "tool_bundles"
    tools_src = source_root / "tools"

    import shutil

    shutil.copytree(tool_bundles_src, tmp_path / "tool_bundles")
    shutil.copytree(tools_src, tmp_path / "tools")

    controller = AppController(repo_root=tmp_path)
    report = controller.inspect_bundle("anh")

    assert report["bundle_id"] == "anh"
    assert report["name"] == "Astronomical Node Heuristics"
    assert report["capabilities"][0]["id"] == "spectral_dip_scan"
    assert report["capabilities"][0]["app_capability_key"] == "astronomy.anh_spectral_scan"


def test_controller_can_inspect_content_bundle_manifest(tmp_path: Path) -> None:
    source_root = Path(__file__).resolve().parents[2]
    tool_bundles_src = source_root / "tool_bundles"
    tools_src = source_root / "tools"
    src_root = source_root / "src"

    import shutil

    shutil.copytree(tool_bundles_src, tmp_path / "tool_bundles")
    shutil.copytree(tools_src, tmp_path / "tools")
    shutil.copytree(src_root, tmp_path / "src")

    controller = AppController(repo_root=tmp_path)
    report = controller.inspect_bundle("content")

    assert report["bundle_id"] == "content"
    capability_ids = {capability["id"] for capability in report["capabilities"]}
    assert capability_ids == {"generate_ideas", "generate_batch", "format_platform"}
