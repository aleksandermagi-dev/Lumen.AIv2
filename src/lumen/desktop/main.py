from __future__ import annotations

import argparse
from pathlib import Path
import tkinter as tk

from lumen.app.runtime_paths import resolve_desktop_runtime_paths
from lumen.desktop.chat_app import LumenDesktopApp
from lumen.desktop.desktop_crash_support import (
    append_crash_record,
    build_crash_record,
    desktop_crash_log_path,
)
from lumen.desktop.startup_diagnostics import StartupCheckpointLogger, startup_log_path
from lumen.validation import (
    packaged_smoke_fallback_report_path,
    prepare_validation_audit_data_root,
    run_packaged_smoke_validation,
    write_validation_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lumen-desktop")
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--validation-smoke-report", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--validation-anh-probe", type=Path, help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    runtime_data_root = (args.data_root or (Path.cwd() / "data")).resolve()
    execution_mode = "source"
    try:
        runtime_paths = resolve_desktop_runtime_paths(
            repo_root=args.repo_root,
            data_root=args.data_root,
        )
        runtime_data_root = runtime_paths.data_root
        execution_mode = runtime_paths.execution_mode
    except Exception as exc:
        append_crash_record(
            log_path=desktop_crash_log_path(data_root=runtime_data_root),
            record=build_crash_record(
                execution_mode=execution_mode,
                source="desktop_main.runtime_paths",
                exc=exc,
            ),
        )
        raise
    startup_logger = StartupCheckpointLogger(
        log_path=startup_log_path(data_root=runtime_paths.data_root),
        execution_mode=runtime_paths.execution_mode,
    )
    startup_logger.checkpoint("desktop_main", "before", details="desktop entrypoint entered")
    startup_logger.checkpoint(
        "runtime_paths_resolved",
        "after",
        details=f"runtime_root={runtime_paths.runtime_root} data_root={runtime_paths.data_root}",
    )
    if args.validation_smoke_report is not None:
        requested_report_path = args.validation_smoke_report.resolve()
        fallback_report_path = packaged_smoke_fallback_report_path(data_root=runtime_paths.data_root)
        startup_logger.checkpoint("validation_smoke", "before", details=f"requested={requested_report_path}")
        try:
            audit_context = prepare_validation_audit_data_root(
                source_data_root=runtime_paths.data_root,
                audit_data_mode="clean",
            )
            audit_data_root = Path(audit_context["audit_data_root"])
            startup_logger.checkpoint(
                "validation_smoke_audit_data",
                "after",
                details=f"audit_data_root={audit_data_root} mode={audit_context['audit_data_mode']}",
            )
            report = run_packaged_smoke_validation(
                repo_root=runtime_paths.runtime_root,
                data_root=audit_data_root,
                execution_mode=runtime_paths.execution_mode,
                anh_probe_path=args.validation_anh_probe,
            )
            report = dict(report)
            report_artifacts = {
                "requested_report_path": str(requested_report_path),
                "fallback_report_path": str(fallback_report_path),
                "requested_report_written": True,
                "fallback_report_written": False,
                "report_authority": "requested_path",
                "audit_data_root": str(audit_data_root),
                "audit_data_mode": audit_context["audit_data_mode"],
            }
            report["report_artifacts"] = report_artifacts
            write_validation_report(report, requested_report_path)

            try:
                report_artifacts["fallback_report_written"] = True
                write_validation_report(report, fallback_report_path)
                write_validation_report(report, requested_report_path)
            except Exception as mirror_exc:
                report_artifacts["fallback_report_written"] = False
                startup_logger.checkpoint("validation_smoke_mirror", "error", details=str(mirror_exc))
                write_validation_report(report, requested_report_path)

            startup_logger.checkpoint(
                "validation_smoke",
                "after",
                details=(
                    f"requested={requested_report_path} "
                    f"fallback={fallback_report_path} "
                    f"requested_written={report_artifacts['requested_report_written']} "
                    f"fallback_written={report_artifacts['fallback_report_written']}"
                ),
            )
            return 0 if not report.get("blockers") else 1
        except Exception as exc:
            startup_logger.checkpoint("validation_smoke", "error", details=str(exc))
            append_crash_record(
                log_path=desktop_crash_log_path(data_root=runtime_paths.data_root),
                record=build_crash_record(
                    execution_mode=runtime_paths.execution_mode,
                    source="desktop_main.validation_smoke",
                    exc=exc,
                ),
            )
            raise
    try:
        startup_logger.checkpoint("tk_root", "before", details="create Tk root")
        root = tk.Tk()
        startup_logger.checkpoint("tk_root", "after", details="Tk root created")
        LumenDesktopApp(
            root,
            repo_root=runtime_paths.runtime_root,
            data_root=runtime_paths.data_root,
            execution_mode=runtime_paths.execution_mode,
            startup_logger=startup_logger,
        )
        startup_logger.checkpoint("mainloop", "before", details="enter Tk mainloop")
        root.mainloop()
        startup_logger.checkpoint("mainloop", "after", details="Tk mainloop exited")
        return 0
    except Exception as exc:
        append_crash_record(
            log_path=desktop_crash_log_path(data_root=runtime_paths.data_root),
            record=build_crash_record(
                execution_mode=runtime_paths.execution_mode,
                source="desktop_main",
                exc=exc,
            ),
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
