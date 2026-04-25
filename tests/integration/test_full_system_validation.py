from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

from lumen.desktop.main import main as desktop_main
from lumen.validation import (
    collect_readiness_truth_table,
    packaged_smoke_fallback_report_path,
    prepare_validation_audit_data_root,
    render_system_sweep_markdown,
    run_full_system_sweep,
    run_source_full_system_validation,
)


class _FakeANHAnalysisModule:
    def load_spectrum(self, path: str):
        return [1392.0, 1393.0, 1402.0, 1403.0], [1.0, 0.9, 0.85, 1.0]

    @staticmethod
    def smooth(values, window: int):
        return values

    @staticmethod
    def zoom(wavelengths, flux, rest_wavelength, width=1.0, title=None, smooth_win=5):
        return (rest_wavelength - 0.01, 0.12, -219.0)

    @staticmethod
    def plot_si_iv_window(wavelengths, flux):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(4, 2))
        plt.plot(wavelengths, flux)


def _copy_validation_assets(repo_root: Path, data_root: Path) -> None:
    source_root = Path(__file__).resolve().parents[2]
    for relative in [
        Path("tool_bundles"),
        Path("tools"),
        Path("src"),
        Path("README.md"),
        Path("lumen.toml.example"),
    ]:
        src = source_root / relative
        dest = repo_root / relative
        if src.is_dir():
            shutil.copytree(src, dest)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)

    for relative in [
        Path("data") / "knowledge",
        Path("data") / "examples",
    ]:
        src = source_root / relative
        dest = data_root / relative.relative_to("data")
        if src.exists():
            shutil.copytree(src, dest)


def test_collect_readiness_truth_table_reports_current_surface(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    data_root = tmp_path / "runtime-data"
    repo_root.mkdir()
    data_root.mkdir()
    _copy_validation_assets(repo_root, data_root)

    sample_fits = data_root / "examples" / "probe_x1d.fits"
    sample_fits.parent.mkdir(parents=True, exist_ok=True)
    sample_fits.write_bytes(b"FAKEFITS")

    from tool_bundles.anh.adapters.anh_spectral_scan_adapter import ANHSpectralDipScanAdapter

    monkeypatch.setattr(ANHSpectralDipScanAdapter, "_load_analysis_module", lambda self: _FakeANHAnalysisModule())

    report = collect_readiness_truth_table(
        repo_root=repo_root,
        data_root=data_root,
        execution_mode="source",
        anh_probe_path=sample_fits,
    )

    assert report["source_surface"]["required_bundles_present"] is True
    assert report["source_surface"]["required_capabilities_present"] is True
    assert report["source_surface"]["reasoning_spine"] is True
    assert report["source_surface"]["state_aware_routing"] is True
    assert report["runtime_truth"]["content"]["label"] in {"fully live", "runtime/provider gated"}
    assert report["runtime_truth"]["anh"]["label"] == "real execution verified"


def test_desktop_hidden_validation_smoke_writes_report(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    data_root = tmp_path / "runtime-data"
    repo_root.mkdir()
    data_root.mkdir()
    _copy_validation_assets(repo_root, data_root)

    sample_fits = data_root / "examples" / "probe_x1d.fits"
    sample_fits.parent.mkdir(parents=True, exist_ok=True)
    sample_fits.write_bytes(b"FAKEFITS")
    report_path = tmp_path / "packaged_smoke_report.json"

    from tool_bundles.anh.adapters.anh_spectral_scan_adapter import ANHSpectralDipScanAdapter

    monkeypatch.setattr(ANHSpectralDipScanAdapter, "_load_analysis_module", lambda self: _FakeANHAnalysisModule())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lumen-desktop",
            "--repo-root",
            str(repo_root),
            "--data-root",
            str(data_root),
            "--validation-smoke-report",
            str(report_path),
            "--validation-anh-probe",
            str(sample_fits),
        ],
    )

    exit_code = desktop_main()

    assert exit_code == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["label"] == "packaged_smoke_validation"
    assert payload["execution_mode"] == "source"
    assert payload["summary"]["blocker_count"] == 0
    assert payload["report_artifacts"]["requested_report_written"] is True
    assert payload["report_artifacts"]["report_authority"] == "requested_path"
    assert payload["report_artifacts"]["audit_data_mode"] == "clean"
    assert payload["report_artifacts"]["audit_data_root"] != str(data_root.resolve())
    fallback_path = packaged_smoke_fallback_report_path(data_root=data_root)
    assert fallback_path.exists()
    fallback_payload = json.loads(fallback_path.read_text(encoding="utf-8"))
    assert fallback_payload["label"] == "packaged_smoke_validation"
    assert fallback_payload["report_artifacts"]["fallback_report_written"] is True


def test_source_full_system_validation_returns_structured_report(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    data_root = tmp_path / "runtime-data"
    repo_root.mkdir()
    data_root.mkdir()
    _copy_validation_assets(repo_root, data_root)

    sample_fits = data_root / "examples" / "probe_x1d.fits"
    sample_fits.parent.mkdir(parents=True, exist_ok=True)
    sample_fits.write_bytes(b"FAKEFITS")

    from tool_bundles.anh.adapters.anh_spectral_scan_adapter import ANHSpectralDipScanAdapter

    monkeypatch.setattr(ANHSpectralDipScanAdapter, "_load_analysis_module", lambda self: _FakeANHAnalysisModule())

    report = run_source_full_system_validation(
        repo_root=repo_root,
        data_root=data_root,
        anh_probe_path=sample_fits,
    )

    assert report["label"] == "source_full_validation"
    assert report["audit_context"]["audit_data_mode"] == "clean"
    assert report["audit_context"]["audit_data_root"] != str(data_root.resolve())
    assert "readiness" in report
    assert isinstance(report["checks"], list)
    assert report["summary"]["passed_count"] > 0


def test_prepare_validation_audit_data_root_supports_clean_and_isolated_copy(tmp_path: Path) -> None:
    source_data_root = tmp_path / "runtime-data"
    (source_data_root / "examples").mkdir(parents=True)
    (source_data_root / "examples" / "seed.txt").write_text("ok", encoding="utf-8")
    (source_data_root / "sessions").mkdir(parents=True)
    (source_data_root / "sessions" / "state.json").write_text("{}", encoding="utf-8")

    clean_context = prepare_validation_audit_data_root(source_data_root=source_data_root, audit_data_mode="clean")
    isolated_copy_context = prepare_validation_audit_data_root(
        source_data_root=source_data_root,
        audit_data_mode="isolated_copy",
    )

    clean_root = Path(clean_context["audit_data_root"])
    copied_root = Path(isolated_copy_context["audit_data_root"])
    assert clean_context["audit_data_mode"] == "clean"
    assert isolated_copy_context["audit_data_mode"] == "isolated_copy"
    assert (clean_root / "examples" / "seed.txt").exists()
    assert not (clean_root / "sessions" / "state.json").exists()
    assert (copied_root / "sessions" / "state.json").exists()


def test_full_system_sweep_consolidates_runtime_and_slice_results(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    data_root = tmp_path / "runtime-data"
    repo_root.mkdir()
    data_root.mkdir()

    source_report = {
        "label": "source_full_validation",
        "summary": {"blocker_count": 0},
        "checks": [],
        "refinement_qa": [],
    }
    packaged_report = {
        "label": "packaged_smoke_validation",
        "summary": {"blocker_count": 0},
        "checks": [],
        "refinement_qa": [],
    }

    monkeypatch.setattr(
        "lumen.validation.system_sweep.run_source_full_system_validation",
        lambda **kwargs: {
            **source_report,
            "audit_context": {
                "audit_data_root": str(data_root / "validation" / "audit"),
                "audit_data_mode": "clean",
            },
        },
    )
    monkeypatch.setattr(
        "lumen.validation.system_sweep.run_packaged_executable_smoke_validation",
        lambda **kwargs: {
            "artifact": {"exists": True, "stale": False},
            "report": packaged_report,
            "boot_status": "ok",
            "timing_ms": 12.0,
            "report_authority": "requested_path",
        },
    )
    monkeypatch.setattr(
        "lumen.validation.system_sweep.run_pytest_regression_slice",
        lambda **kwargs: {
            "slice_id": kwargs["slice_def"].slice_id,
            "title": kwargs["slice_def"].title,
            "category": kwargs["slice_def"].category,
            "owner_subsystem": kwargs["slice_def"].owner_subsystem,
            "runtime_scope": kwargs["slice_def"].runtime_scope,
            "targets": list(kwargs["slice_def"].targets),
            "passed": True,
            "returncode": 0,
            "timing_ms": 8.5,
            "summary": "10 passed in 0.11s",
            "stdout_tail": "10 passed in 0.11s",
            "stderr_tail": "",
            "rerun": False,
        },
    )

    report = run_full_system_sweep(repo_root=repo_root, data_root=data_root)

    assert report["schema_version"] == "full_system_sweep.v1"
    assert report["audit_mode"] == "fast"
    assert report["audit_data_mode"] == "clean"
    assert report["runtime_summary"]["source_verdict"] == "pass"
    assert report["runtime_summary"]["packaged_verdict"] == "pass"
    assert report["runtime_summary"]["parity_verdict"] == "aligned"
    assert report["runtime_summary"]["blocker_count"] == 0
    assert report["release_readiness"]["ui_shell_status"] == "pass"
    assert report["release_readiness"]["tool_routing_status"] == "pass"
    assert report["release_readiness"]["anh_live_probe_status"] == "deferred_final_gate"
    assert report["release_readiness"]["intentional_skips"] == [
        "ANH live MAST probe deferred to final release-candidate gate."
    ]
    assert report["findings"] == []
    assert report["systems_verified_stable"]
    markdown = render_system_sweep_markdown(report)
    assert "## Release Readiness" in markdown
    assert "ANH live probe: `deferred_final_gate`" in markdown
    assert "## Regressions Found" in markdown
    assert "## Systems Verified Stable" in markdown
    assert "## Weaknesses and Gaps" in markdown
    assert "## Intentionally Unchanged" in markdown
    assert "## System Coherence" in markdown
    assert "Audit mode" in markdown


def test_full_system_sweep_reports_packaged_runtime_gap(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    data_root = tmp_path / "runtime-data"
    repo_root.mkdir()
    data_root.mkdir()

    monkeypatch.setattr(
        "lumen.validation.system_sweep.run_source_full_system_validation",
        lambda **kwargs: {
            "label": "source_full_validation",
            "summary": {"blocker_count": 0},
            "checks": [],
            "refinement_qa": [],
            "audit_context": {"audit_data_root": str(data_root / "validation" / "audit"), "audit_data_mode": "clean"},
        },
    )
    monkeypatch.setattr(
        "lumen.validation.system_sweep.run_packaged_executable_smoke_validation",
        lambda **kwargs: {
            "artifact": {"exists": False, "stale": False},
            "report": None,
            "boot_status": "missing",
            "timing_ms": 0.0,
            "stderr_tail": "",
            "stdout_tail": "",
        },
    )
    monkeypatch.setattr(
        "lumen.validation.system_sweep.run_pytest_regression_slice",
        lambda **kwargs: {
            "slice_id": kwargs["slice_def"].slice_id,
            "title": kwargs["slice_def"].title,
            "category": kwargs["slice_def"].category,
            "owner_subsystem": kwargs["slice_def"].owner_subsystem,
            "runtime_scope": kwargs["slice_def"].runtime_scope,
            "targets": list(kwargs["slice_def"].targets),
            "passed": True,
            "returncode": 0,
            "timing_ms": 5.0,
            "summary": "ok",
            "stdout_tail": "ok",
            "stderr_tail": "",
            "rerun": False,
        },
    )

    report = run_full_system_sweep(repo_root=repo_root, data_root=data_root)

    assert report["runtime_summary"]["packaged_verdict"] == "fail"
    assert report["runtime_summary"]["parity_verdict"] == "diverged"
    assert any(item["category"] == "packaged/runtime parity" for item in report["findings"])


def test_full_system_sweep_recovers_packaged_report_from_fallback_path(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    data_root = tmp_path / "runtime-data"
    repo_root.mkdir()
    data_root.mkdir()

    fallback_path = packaged_smoke_fallback_report_path(data_root=data_root)
    fallback_path.parent.mkdir(parents=True, exist_ok=True)
    fallback_path.write_text(
        json.dumps(
            {
                "label": "packaged_smoke_validation",
                "summary": {"blocker_count": 0},
                "checks": [],
                "refinement_qa": [],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "lumen.validation.system_sweep.subprocess.run",
        lambda *args, **kwargs: type(
            "Completed",
            (),
            {"returncode": 0, "stdout": "", "stderr": ""},
        )(),
    )
    monkeypatch.setattr(
        "lumen.validation.system_sweep.inspect_packaged_artifact",
        lambda **kwargs: {"exists": True, "stale": False, "path": str(repo_root / "dist" / "lumen.exe")},
    )

    from lumen.validation.system_sweep import run_packaged_executable_smoke_validation

    result = run_packaged_executable_smoke_validation(
        repo_root=repo_root,
        data_root=data_root,
        packaged_executable=repo_root / "dist" / "lumen.exe",
    )

    assert result["boot_status"] == "ok"
    assert result["report_source"] == "fallback_recovered"
    assert result["requested_report_missing"] is True


def test_full_system_sweep_fast_mode_uses_subset_of_slices(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    data_root = tmp_path / "runtime-data"
    repo_root.mkdir()
    data_root.mkdir()

    monkeypatch.setattr(
        "lumen.validation.system_sweep.run_source_full_system_validation",
        lambda **kwargs: {
            "label": "source_full_validation",
            "summary": {"blocker_count": 0},
            "checks": [],
            "refinement_qa": [],
            "audit_context": {"audit_data_root": str(data_root / "validation" / "audit"), "audit_data_mode": "clean"},
        },
    )
    monkeypatch.setattr(
        "lumen.validation.system_sweep.run_packaged_executable_smoke_validation",
        lambda **kwargs: {
            "artifact": {"exists": True, "stale": False},
            "report": {"label": "packaged_smoke_validation", "summary": {"blocker_count": 0}, "checks": [], "refinement_qa": []},
            "boot_status": "ok",
            "timing_ms": 5.0,
            "report_authority": "requested_path",
        },
    )

    seen_slice_ids: list[str] = []

    def _fake_slice(**kwargs):
        seen_slice_ids.append(kwargs["slice_def"].slice_id)
        return {
            "slice_id": kwargs["slice_def"].slice_id,
            "title": kwargs["slice_def"].title,
            "category": kwargs["slice_def"].category,
            "owner_subsystem": kwargs["slice_def"].owner_subsystem,
            "runtime_scope": kwargs["slice_def"].runtime_scope,
            "targets": list(kwargs["slice_def"].targets),
            "passed": True,
            "returncode": 0,
            "timing_ms": 1.0,
            "summary": "ok",
            "stdout_tail": "ok",
            "stderr_tail": "",
            "rerun": False,
        }

    monkeypatch.setattr("lumen.validation.system_sweep.run_pytest_regression_slice", _fake_slice)

    report = run_full_system_sweep(repo_root=repo_root, data_root=data_root, mode="fast")

    assert report["audit_mode"] == "fast"
    assert "chat_state_memory_persistence" not in seen_slice_ids
    assert "validation_contract" in seen_slice_ids
    routing_slice = next(item for item in report["targeted_slices"] if item["slice_id"] == "routing_reasoning_regression")
    assert routing_slice["targets"] == ["tests/unit/test_reasoning_pipeline.py"]


def test_full_system_sweep_full_mode_includes_stability_hardening_slices(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    data_root = tmp_path / "runtime-data"
    repo_root.mkdir()
    data_root.mkdir()

    monkeypatch.setattr(
        "lumen.validation.system_sweep.run_source_full_system_validation",
        lambda **kwargs: {
            "label": "source_full_validation",
            "summary": {"blocker_count": 0},
            "checks": [],
            "refinement_qa": [],
            "audit_context": {"audit_data_root": str(data_root / "validation" / "audit"), "audit_data_mode": "clean"},
        },
    )
    monkeypatch.setattr(
        "lumen.validation.system_sweep.run_packaged_executable_smoke_validation",
        lambda **kwargs: {
            "artifact": {"exists": True, "stale": False},
            "report": {"label": "packaged_smoke_validation", "summary": {"blocker_count": 0}, "checks": [], "refinement_qa": []},
            "boot_status": "ok",
            "timing_ms": 5.0,
            "report_authority": "requested_path",
        },
    )

    seen_slice_ids: list[str] = []

    def _fake_slice(**kwargs):
        seen_slice_ids.append(kwargs["slice_def"].slice_id)
        return {
            "slice_id": kwargs["slice_def"].slice_id,
            "title": kwargs["slice_def"].title,
            "category": kwargs["slice_def"].category,
            "owner_subsystem": kwargs["slice_def"].owner_subsystem,
            "runtime_scope": kwargs["slice_def"].runtime_scope,
            "targets": list(kwargs["slice_def"].targets),
            "passed": True,
            "returncode": 0,
            "timing_ms": 1.0,
            "summary": "ok",
            "stdout_tail": "ok",
            "stderr_tail": "",
            "rerun": False,
        }

    monkeypatch.setattr("lumen.validation.system_sweep.run_pytest_regression_slice", _fake_slice)

    report = run_full_system_sweep(repo_root=repo_root, data_root=data_root, mode="full")

    assert report["audit_mode"] == "full"
    assert "runtime_hardening" in seen_slice_ids
    assert "conversation_qa_stability" in seen_slice_ids
    assert report["release_readiness"]["persistence_status"] == "pass"


def test_packaged_smoke_validation_can_reuse_cached_fallback_report(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    data_root = tmp_path / "runtime-data"
    dist_dir = repo_root / "dist"
    dist_dir.mkdir(parents=True)
    data_root.mkdir()
    packaged_executable = dist_dir / "lumen.exe"
    packaged_executable.write_text("stub", encoding="utf-8")

    fallback_report = data_root / "validation" / "packaged_smoke_latest.json"
    fallback_report.parent.mkdir(parents=True, exist_ok=True)
    fallback_report.write_text(
        json.dumps(
            {
                "label": "packaged_smoke_validation",
                "generated_at": "2026-04-13T01:23:45+00:00",
                "summary": {"blocker_count": 0},
                "checks": [],
                "refinement_qa": [],
                "report_artifacts": {
                    "requested_report_written": True,
                    "fallback_report_written": True,
                    "report_authority": "requested_path",
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "lumen.validation.system_sweep.inspect_packaged_artifact",
        lambda **kwargs: {
            "path": str(packaged_executable),
            "exists": True,
            "stale": False,
            "built_at": "2026-04-13T01:00:00+00:00",
            "latest_source_at": "2026-04-13T00:55:00+00:00",
        },
    )

    def _should_not_run(*args, **kwargs):
        raise AssertionError("fresh packaged executable launch should not happen when a fresh cached report is reusable")

    monkeypatch.setattr("lumen.validation.system_sweep.subprocess.run", _should_not_run)

    from lumen.validation.system_sweep import run_packaged_executable_smoke_validation

    result = run_packaged_executable_smoke_validation(
        repo_root=repo_root,
        data_root=data_root,
        packaged_executable=packaged_executable,
        allow_cached_report=True,
    )

    assert result["boot_status"] == "ok"
    assert result["cache_used"] is True
    assert result["validation_origin"] == "cached_report"
    assert result["report_authority"] == "requested_path"
    assert result["requested_report_written"] is True


def test_packaged_smoke_validation_prefers_lumen_executable_by_default(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    data_root = tmp_path / "runtime-data"
    dist_dir = repo_root / "dist"
    dist_dir.mkdir(parents=True)
    data_root.mkdir()
    packaged_executable = dist_dir / "lumen.exe"
    packaged_executable.write_text("stub", encoding="utf-8")

    fallback_report = data_root / "validation" / "packaged_smoke_latest.json"
    fallback_report.parent.mkdir(parents=True, exist_ok=True)
    fallback_report.write_text(
        json.dumps(
            {
                "label": "packaged_smoke_validation",
                "generated_at": "2026-04-13T01:23:45+00:00",
                "summary": {"blocker_count": 0},
                "checks": [],
                "refinement_qa": [],
                "report_artifacts": {
                    "requested_report_written": True,
                    "fallback_report_written": True,
                    "report_authority": "requested_path",
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "lumen.validation.system_sweep.inspect_packaged_artifact",
        lambda **kwargs: {
            "path": str(kwargs["packaged_executable"]),
            "exists": True,
            "stale": False,
            "built_at": "2026-04-13T01:00:00+00:00",
            "latest_source_at": "2026-04-13T00:55:00+00:00",
        },
    )

    def _should_not_run(*args, **kwargs):
        raise AssertionError("fresh packaged executable launch should not happen when a fresh cached report is reusable")

    monkeypatch.setattr("lumen.validation.system_sweep.subprocess.run", _should_not_run)

    from lumen.validation.system_sweep import run_packaged_executable_smoke_validation

    result = run_packaged_executable_smoke_validation(
        repo_root=repo_root,
        data_root=data_root,
        allow_cached_report=True,
    )

    assert result["artifact"]["path"] == str(packaged_executable)
    assert result["boot_status"] == "ok"
    assert result["cache_used"] is True


def test_packaged_smoke_validation_skips_stale_artifact_in_cached_mode(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    data_root = tmp_path / "runtime-data"
    dist_dir = repo_root / "dist"
    dist_dir.mkdir(parents=True)
    data_root.mkdir()
    packaged_executable = dist_dir / "lumen.exe"
    packaged_executable.write_text("stub", encoding="utf-8")

    monkeypatch.setattr(
        "lumen.validation.system_sweep.inspect_packaged_artifact",
        lambda **kwargs: {
            "path": str(packaged_executable),
            "exists": True,
            "stale": True,
            "built_at": "2026-04-12T23:55:26+00:00",
            "latest_source_at": "2026-04-13T21:59:24+00:00",
        },
    )

    def _should_not_run(*args, **kwargs):
        raise AssertionError("stale packaged artifacts should be skipped in cached sweep mode")

    monkeypatch.setattr("lumen.validation.system_sweep.subprocess.run", _should_not_run)

    from lumen.validation.system_sweep import run_packaged_executable_smoke_validation

    result = run_packaged_executable_smoke_validation(
        repo_root=repo_root,
        data_root=data_root,
        packaged_executable=packaged_executable,
        allow_cached_report=True,
    )

    assert result["boot_status"] == "stale_skipped"
    assert result["validation_origin"] == "stale_artifact_skipped"
    assert result["report_authority"] == "stale_skipped"


def test_full_system_sweep_can_force_fresh_packaged_validation(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    data_root = tmp_path / "runtime-data"
    repo_root.mkdir()
    data_root.mkdir()

    seen_allow_cached: list[bool] = []

    monkeypatch.setattr(
        "lumen.validation.system_sweep.run_source_full_system_validation",
        lambda **kwargs: {
            "label": "source_full_validation",
            "summary": {"blocker_count": 0},
            "checks": [],
            "refinement_qa": [],
            "audit_context": {"audit_data_root": str(data_root / "validation" / "audit"), "audit_data_mode": "clean"},
        },
    )

    def _fake_packaged(**kwargs):
        seen_allow_cached.append(bool(kwargs.get("allow_cached_report")))
        return {
            "artifact": {"exists": True, "stale": False},
            "report": {"label": "packaged_smoke_validation", "summary": {"blocker_count": 0}, "checks": [], "refinement_qa": []},
            "boot_status": "ok",
            "timing_ms": 5.0,
            "report_authority": "requested_path",
            "validation_origin": "fresh_run",
            "cache_used": False,
        }

    monkeypatch.setattr(
        "lumen.validation.system_sweep.run_packaged_executable_smoke_validation",
        _fake_packaged,
    )
    monkeypatch.setattr(
        "lumen.validation.system_sweep.run_pytest_regression_slice",
        lambda **kwargs: {
            "slice_id": kwargs["slice_def"].slice_id,
            "title": kwargs["slice_def"].title,
            "category": kwargs["slice_def"].category,
            "owner_subsystem": kwargs["slice_def"].owner_subsystem,
            "runtime_scope": kwargs["slice_def"].runtime_scope,
            "targets": list(kwargs["slice_def"].targets),
            "passed": True,
            "returncode": 0,
            "timing_ms": 1.0,
            "summary": "ok",
            "stdout_tail": "ok",
            "stderr_tail": "",
            "rerun": False,
        },
    )

    run_full_system_sweep(repo_root=repo_root, data_root=data_root)
    run_full_system_sweep(repo_root=repo_root, data_root=data_root, force_fresh_packaged=True)

    assert seen_allow_cached == [True, False]


def test_system_sweep_release_gate_uses_full_fresh_profile(tmp_path: Path, monkeypatch) -> None:
    from lumen.validation import system_sweep

    repo_root = tmp_path / "repo"
    data_root = tmp_path / "runtime-data"
    json_report = tmp_path / "release.json"
    markdown_report = tmp_path / "release.md"
    repo_root.mkdir()
    data_root.mkdir()

    captured: dict[str, object] = {}

    def _fake_sweep(**kwargs):
        captured.update(kwargs)
        return {
            "schema_version": "full_system_sweep.v1",
            "generated_at": "2026-04-22T00:00:00+00:00",
            "audit_mode": "full",
            "audit_data_mode": "clean",
            "runtime_summary": {
                "source_verdict": "pass",
                "packaged_verdict": "pass",
                "parity_verdict": "aligned",
                "blocker_count": 0,
                "hotfix_count": 0,
            },
            "packaged_runtime": {"validation_origin": "fresh_run", "report_authority": "requested_path"},
            "release_readiness": {"anh_live_probe_status": "deferred_final_gate"},
            "regressions_found": {},
            "root_causes": [],
            "fixes_applied": [],
            "systems_verified_stable": [],
            "remaining_risks": [],
            "weaknesses_and_gaps": [],
            "intentionally_unchanged": [],
            "recommendations": {"v2_stabilization": [], "v3_improvements": []},
            "system_coherence": {"coherent": True, "conclusion": "ok"},
        }

    monkeypatch.setattr(system_sweep, "run_full_system_sweep", _fake_sweep)

    exit_code = system_sweep.main(
        [
            "--repo-root",
            str(repo_root),
            "--data-root",
            str(data_root),
            "--json-report",
            str(json_report),
            "--markdown-report",
            str(markdown_report),
            "--release-gate",
        ]
    )

    assert exit_code == 0
    assert captured["mode"] == "full"
    assert captured["force_fresh_packaged"] is True
    assert json_report.exists()
    assert markdown_report.exists()


def test_source_validation_defers_anh_when_probe_not_supplied(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    data_root = tmp_path / "runtime-data"
    repo_root.mkdir()
    data_root.mkdir()
    _copy_validation_assets(repo_root, data_root)

    report = run_source_full_system_validation(
        repo_root=repo_root,
        data_root=data_root,
        anh_probe_path=None,
    )

    anh_checks = [item for item in report["checks"] if item["check_id"] == "source_anh_runtime"]
    assert anh_checks
    assert anh_checks[0]["passed"] is True
    assert anh_checks[0]["evidence"]["status"] == "deferred_final_gate"
    assert report["summary"]["blocker_count"] == 0
    assert not any("ANH runtime status" in item for item in report["refinement_qa"])
