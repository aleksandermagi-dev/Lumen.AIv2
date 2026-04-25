from __future__ import annotations

from pathlib import Path
import shutil

from lumen.app.controller import AppController


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


def test_controller_run_command_routes_run_anh_into_anh_bundle(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = tmp_path
    _copy_project_assets(repo_root)

    from tool_bundles.anh.adapters.anh_spectral_scan_adapter import ANHSpectralDipScanAdapter

    monkeypatch.setattr(ANHSpectralDipScanAdapter, "_load_analysis_module", lambda self: _FakeAnalysisModule())

    controller = AppController(repo_root=repo_root)
    sample_fits = repo_root / "data" / "examples" / "m31_x1d.fits"
    sample_fits.write_bytes(b"FAKEFITS")

    result = controller.run_command(
        action="run",
        target="anh",
        input_path=sample_fits,
        params={"rank_limit": 5},
        session_id="command-route",
    )

    assert result.status == "ok"
    assert result.tool_id == "anh"
    assert result.capability == "spectral_dip_scan"
    assert result.archive_path is not None
    assert result.archive_path.exists()
    assert result.structured_data["bundle_standard"] == "lumen_research_bundle_v1"
    assert result.structured_data["analysis_status"]["validated"] is True
    assert result.structured_data["analysis_status"]["analysis_ran"] is True
    assert result.structured_data["domain_payload"]["selected_params"]["rank_limit"] == 5
    assert result.structured_data["batch_record"]["candidate_files"][0]["filename"] == "m31_x1d.fits"


def _copy_project_assets(repo_root: Path) -> None:
    source_root = Path(__file__).resolve().parents[2]
    for relative in [
        Path("tool_bundles"),
        Path("tools"),
        Path("data") / "examples",
        Path("lumen.toml.example"),
    ]:
        src = source_root / relative
        dest = repo_root / relative
        if src.is_dir():
            shutil.copytree(src, dest)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
