from pathlib import Path
import json
import zipfile

from tool_bundles.anh.adapters.anh_spectral_scan_adapter import ANHSpectralDipScanAdapter
from lumen.tools.registry_types import BundleManifest, CapabilityManifest, ToolRequest


class _FakeAnalysisModule:
    def load_spectrum(self, path: str):
        print(f"Coverage: loaded {Path(path).name}")
        return [1392.0, 1393.0, 1402.0, 1403.0], [1.0, 0.9, 0.85, 1.0]

    @staticmethod
    def smooth(values, window: int):
        return values

    @staticmethod
    def zoom(wavelengths, flux, rest_wavelength, width=1.0, title=None, smooth_win=5):
        print(f"{title}: checked")
        return (rest_wavelength - 0.01, 0.12, -219.0)

    @staticmethod
    def plot_si_iv_window(wavelengths, flux):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 3))
        plt.plot(wavelengths, flux)
        plt.title("Si IV window")


class _BrokenAnalysisModule(_FakeAnalysisModule):
    def load_spectrum(self, path: str):
        raise RuntimeError("failed to load spectrum")


class _IndexErrorThenSparseAnalysisModule(_FakeAnalysisModule):
    def analyze_siiv_targets(self, path: str, target_lines):
        raise IndexError("index 4 is out of bounds for axis 0 with size 3")

    def load_spectrum(self, path: str):
        return [1393.76, 1393.77], [1.0, 0.98]


def _manifest() -> BundleManifest:
    return BundleManifest(
        id="anh",
        name="Astronomical Node Heuristics",
        version="1.0.0",
        entrypoint="bundle.py",
        capabilities=[
            CapabilityManifest(
                id="spectral_dip_scan",
                adapter="anh_spectral_scan_adapter",
                app_capability_key="astronomy.anh_spectral_scan",
            )
        ],
    )


def test_anh_adapter_rejects_unsupported_input(tmp_path: Path) -> None:
    adapter = ANHSpectralDipScanAdapter(manifest=_manifest(), repo_root=tmp_path)
    source = tmp_path / "random.txt"
    source.write_text("not a fits file", encoding="utf-8")

    result = adapter.execute(
        ToolRequest(
            tool_id="anh",
            capability="spectral_dip_scan",
            input_path=source,
        )
    )

    assert result.status == "error"
    assert result.structured_data["analysis_status"]["result_quality"] == "invalid_input"


def test_anh_adapter_runs_single_file_flow(tmp_path: Path, monkeypatch) -> None:
    adapter = ANHSpectralDipScanAdapter(manifest=_manifest(), repo_root=tmp_path)
    monkeypatch.setattr(adapter, "_load_analysis_module", lambda: _FakeAnalysisModule())
    source = tmp_path / "m31_x1d.fits"
    source.write_bytes(b"FAKEFITS")

    result = adapter.execute(
        ToolRequest(
            tool_id="anh",
            capability="spectral_dip_scan",
            input_path=source,
            session_id="unit",
        )
    )

    assert result.status == "ok"
    assert result.summary.startswith("ANH analyzed 1 file")
    assert result.structured_data["analysis_status"]["result_quality"] == "candidate_dips_detected"
    assert result.structured_data["analysis_status"]["runtime_diagnostics"]["numpy"]["available"] is True
    assert result.structured_data["batch_record"]["files_analyzed"] == 1
    assert result.structured_data["batch_record"]["candidate_files"][0]["filename"] == "m31_x1d.fits"
    artifact_status = result.structured_data["domain_payload"]["accepted_files"][0]["artifact_generation_status"]
    assert artifact_status["overview_plot_expected"] is True
    assert artifact_status["window_plot_expected"] is True
    assert artifact_status["overview_plot_created"] is True
    assert artifact_status["window_plot_created"] is True
    assert any(artifact.name == "candidate_rankings.json" for artifact in result.artifacts)
    assert any("Coverage:" in line for line in result.logs)

    summary_payload = json.loads((result.run_dir / "outputs" / "analysis_summary.json").read_text(encoding="utf-8"))
    assert summary_payload["analysis_status"]["validated"] is True
    assert summary_payload["provenance"]["baseline_script_path"].endswith("anh_andromeda_v6_3.py")
    assert summary_payload["intake"]["category"] == "raw_spectral_input"


def test_anh_adapter_supports_directory_batch_and_partial_failure(tmp_path: Path, monkeypatch) -> None:
    adapter = ANHSpectralDipScanAdapter(manifest=_manifest(), repo_root=tmp_path)
    monkeypatch.setattr(adapter, "_load_analysis_module", lambda: _BrokenAnalysisModule())
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    (batch_dir / "a_x1d.fits").write_bytes(b"A")
    (batch_dir / "skip.txt").write_text("bad", encoding="utf-8")

    result = adapter.execute(
        ToolRequest(
            tool_id="anh",
            capability="spectral_dip_scan",
            input_path=batch_dir,
            session_id="unit",
        )
    )

    assert result.status == "error"
    assert result.structured_data["analysis_status"]["result_quality"] == "analysis_failed"
    assert result.structured_data["batch_record"]["files_analyzed"] == 1
    assert result.structured_data["batch_record"]["skipped_files"]
    assert any("skip.txt" in item["path"] for item in result.structured_data["batch_record"]["skipped_files"])


def test_anh_adapter_summarizes_processed_results_without_running_scan(tmp_path: Path, monkeypatch) -> None:
    adapter = ANHSpectralDipScanAdapter(manifest=_manifest(), repo_root=tmp_path)
    monkeypatch.setattr(adapter, "_load_analysis_module", lambda: (_ for _ in ()).throw(AssertionError("scan should not run")))
    source = tmp_path / "Full_Si_IV_Absorption_Dataset.csv"
    source.write_text(
        "\n".join(
            [
                "File,Si IV 1393 λ (Å),Si IV 1393 v (km/s),Si IV 1402 λ (Å),Si IV 1402 v (km/s)",
                "1,1394.004,52,1403.013,51",
            ]
        ),
        encoding="utf-8",
    )

    result = adapter.execute(
        ToolRequest(
            tool_id="anh",
            capability="spectral_dip_scan",
            input_path=source,
            session_id="unit",
        )
    )

    assert result.status == "ok"
    assert result.structured_data["analysis_status"]["result_quality"] == "processed_results_summarized"
    assert result.structured_data["intake"]["category"] == "processed_results_input"
    assert result.structured_data["intake"]["execution_mode"] == "summarize_results"
    assert any(artifact.name.endswith("_normalized_results.json") for artifact in result.artifacts)


def test_anh_adapter_supports_manifest_batch_with_mixed_inputs(tmp_path: Path, monkeypatch) -> None:
    adapter = ANHSpectralDipScanAdapter(manifest=_manifest(), repo_root=tmp_path)
    monkeypatch.setattr(adapter, "_load_analysis_module", lambda: _FakeAnalysisModule())
    raw_source = tmp_path / "m31_x1d.fits"
    raw_source.write_bytes(b"FAKEFITS")
    processed_source = tmp_path / "si_iv_summary.csv"
    processed_source.write_text(
        "\n".join(
            [
                "File,Si IV 1393 λ (Å),Si IV 1393 v (km/s),Si IV 1402 λ (Å),Si IV 1402 v (km/s)",
                "1,1394.004,52,1403.013,51",
            ]
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "batch_manifest.txt"
    manifest.write_text(
        "\n".join([raw_source.name, processed_source.name, "missing_x1d.fits"]),
        encoding="utf-8",
    )

    result = adapter.execute(
        ToolRequest(
            tool_id="anh",
            capability="spectral_dip_scan",
            input_path=manifest,
            session_id="unit",
        )
    )

    assert result.status == "partial"
    assert result.structured_data["intake"]["category"] == "batch_input"
    assert result.structured_data["intake"]["execution_mode"] == "batch_mixed"
    assert result.structured_data["analysis_status"]["partial_success"] is True
    assert any("missing_x1d.fits" in item["path"] for item in result.structured_data["batch_record"]["skipped_files"])


def test_anh_adapter_stages_archive_input_and_runs_scan(tmp_path: Path, monkeypatch) -> None:
    adapter = ANHSpectralDipScanAdapter(manifest=_manifest(), repo_root=tmp_path)
    monkeypatch.setattr(adapter, "_load_analysis_module", lambda: _FakeAnalysisModule())
    archive_path = tmp_path / "mast_bundle.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("raw/lb6f07nrq_x1d.fits", b"FAKEFITS")
        archive.writestr("raw/notes.txt", "ignore me")

    result = adapter.execute(
        ToolRequest(
            tool_id="anh",
            capability="spectral_dip_scan",
            input_path=archive_path,
            session_id="unit",
        )
    )

    assert result.status == "ok"
    assert result.structured_data["intake"]["category"] == "archive_input"
    assert result.structured_data["analysis_status"]["analysis_ran"] is True
    assert result.structured_data["analysis_status"]["partial_success"] is False
    assert result.structured_data["batch_record"]["files_extracted"] >= 2
    assert result.structured_data["batch_record"]["staged_good_count"] == 1
    assert result.structured_data["batch_record"]["staged_failed_count"] == 0
    assert result.structured_data["batch_record"]["manifest_paths"]["good_manifest"].endswith("good_manifest.txt")
    assert not any("notes.txt" in item["path"] for item in result.structured_data["batch_record"]["skipped_files"])
    assert any("notes.txt" in item["path"] for item in result.structured_data["batch_record"]["skipped_non_spectral"])


def test_anh_adapter_applies_archive_limit_after_valid_member_classification(tmp_path: Path, monkeypatch) -> None:
    adapter = ANHSpectralDipScanAdapter(manifest=_manifest(), repo_root=tmp_path)
    monkeypatch.setattr(adapter, "_load_analysis_module", lambda: _FakeAnalysisModule())
    archive_path = tmp_path / "mast_bundle.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("000_MANIFEST.html", "<html>support</html>")
        for index in range(4):
            archive.writestr(f"raw/lb6f07n{index:02d}_x1d.fits", b"FAKEFITS")

    result = adapter.execute(
        ToolRequest(
            tool_id="anh",
            capability="spectral_dip_scan",
            input_path=archive_path,
            params={"max_files": 3},
            session_id="unit",
        )
    )

    assert result.status == "partial"
    assert result.structured_data["batch_record"]["staged_good_count"] == 3
    assert result.structured_data["batch_record"]["staged_failed_count"] == 1
    assert result.structured_data["batch_record"]["skipped_non_spectral_count"] == 1
    assert any("MANIFEST.html" in item["path"] for item in result.structured_data["batch_record"]["skipped_non_spectral"])
    assert not any("MANIFEST.html" in item["path"] for item in result.structured_data["batch_record"]["skipped_files"])


def test_anh_adapter_reports_sparse_index_error_as_insufficient_data(tmp_path: Path, monkeypatch) -> None:
    adapter = ANHSpectralDipScanAdapter(manifest=_manifest(), repo_root=tmp_path)
    monkeypatch.setattr(adapter, "_load_analysis_module", lambda: _IndexErrorThenSparseAnalysisModule())
    source = tmp_path / "short_x1d.fits"
    source.write_bytes(b"FAKEFITS")

    result = adapter.execute(
        ToolRequest(
            tool_id="anh",
            capability="spectral_dip_scan",
            input_path=source,
            session_id="unit",
        )
    )

    assert result.status == "error"
    assert result.structured_data["analysis_status"]["result_quality"] == "analysis_failed"
    skipped = result.structured_data["batch_record"]["skipped_files"]
    assert any("too few usable samples" in item["reason"] for item in skipped)
    assert not any("IndexError" in item["reason"] for item in skipped)


def test_anh_adapter_reports_missing_runtime_dependency_for_raw_inputs(tmp_path: Path, monkeypatch) -> None:
    adapter = ANHSpectralDipScanAdapter(manifest=_manifest(), repo_root=tmp_path)
    monkeypatch.setattr(adapter, "_load_analysis_module", lambda: (_ for _ in ()).throw(ModuleNotFoundError("astropy")))
    source = tmp_path / "m31_x1d.fits"
    source.write_bytes(b"FAKEFITS")

    result = adapter.execute(
        ToolRequest(
            tool_id="anh",
            capability="spectral_dip_scan",
            input_path=source,
            session_id="unit",
        )
    )

    assert result.status == "error"
    assert result.summary.startswith("ANH could not run raw spectral analysis")
    assert result.structured_data["analysis_status"]["result_quality"] == "missing_runtime_dependency"
    assert "astropy" in result.structured_data["analysis_status"]["failure_reason"]
    diagnostics = result.structured_data["analysis_status"]["runtime_diagnostics"]
    assert "astropy" in diagnostics
    assert "numpy" in diagnostics


def test_anh_adapter_reports_archive_with_no_valid_spectra(tmp_path: Path) -> None:
    adapter = ANHSpectralDipScanAdapter(manifest=_manifest(), repo_root=tmp_path)
    archive_path = tmp_path / "mast_bundle.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("raw/readme.txt", "not spectral")
        archive.writestr("raw/metadata.json", "{}")

    result = adapter.execute(
        ToolRequest(
            tool_id="anh",
            capability="spectral_dip_scan",
            input_path=archive_path,
            session_id="unit",
        )
    )

    assert result.status == "error"
    assert result.structured_data["analysis_status"]["result_quality"] == "archive_staged_no_valid_spectra"
    assert result.summary.startswith("ANH staged the archive input")
