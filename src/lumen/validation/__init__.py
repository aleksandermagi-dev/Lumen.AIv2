from importlib import import_module
from typing import Any

from lumen.validation.full_system_validation import (
    collect_readiness_truth_table,
    packaged_smoke_fallback_report_path,
    prepare_validation_audit_data_root,
    render_qa_readout_markdown,
    run_packaged_smoke_validation,
    run_source_full_system_validation,
    write_validation_report,
)

__all__ = [
    "SYSTEM_SWEEP_TEST_SLICES",
    "collect_readiness_truth_table",
    "packaged_smoke_fallback_report_path",
    "prepare_validation_audit_data_root",
    "inspect_packaged_artifact",
    "render_system_sweep_markdown",
    "render_qa_readout_markdown",
    "run_full_system_sweep",
    "run_packaged_executable_smoke_validation",
    "run_packaged_smoke_validation",
    "run_pytest_regression_slice",
    "run_source_full_system_validation",
    "write_system_sweep_artifacts",
    "write_validation_report",
]

_SYSTEM_SWEEP_EXPORTS = {
    "SYSTEM_SWEEP_TEST_SLICES",
    "inspect_packaged_artifact",
    "render_system_sweep_markdown",
    "run_full_system_sweep",
    "run_packaged_executable_smoke_validation",
    "run_pytest_regression_slice",
    "write_system_sweep_artifacts",
}


def __getattr__(name: str) -> Any:
    if name in _SYSTEM_SWEEP_EXPORTS:
        module = import_module("lumen.validation.system_sweep")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
