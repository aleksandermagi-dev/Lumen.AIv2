from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

from lumen.validation.full_system_validation import (
    packaged_smoke_fallback_report_path,
    run_source_full_system_validation,
    write_validation_report,
)


SWEEP_SCHEMA_VERSION = "full_system_sweep.v1"
SOURCE_VALIDATION_SCHEMA_VERSION = "validation_report.v2"
PYTEST_SLICE_TIMEOUT_SECONDS = 600
PACKAGED_SMOKE_TIMEOUT_SECONDS = 600
DEFAULT_SWEEP_MODE = "fast"


@dataclass(frozen=True, slots=True)
class SweepTestSlice:
    slice_id: str
    title: str
    category: str
    owner_subsystem: str
    runtime_scope: str
    targets: tuple[str, ...]


SYSTEM_SWEEP_TEST_SLICES: tuple[SweepTestSlice, ...] = (
    SweepTestSlice(
        slice_id="ui_shell_regression",
        title="UI and desktop shell regression",
        category="ui/shell",
        owner_subsystem="desktop_shell",
        runtime_scope="source-only",
        targets=(
            "tests/unit/test_chat_ui_support.py",
            "tests/unit/test_chat_presenter.py",
        ),
    ),
    SweepTestSlice(
        slice_id="routing_reasoning_regression",
        title="Input, routing, reasoning, and interaction flow",
        category="routing/reasoning",
        owner_subsystem="interaction_service",
        runtime_scope="shared",
        targets=(
            "tests/unit/test_interaction_service.py",
            "tests/unit/test_reasoning_pipeline.py",
            "tests/integration/test_tool_signal_ask_routing.py",
        ),
    ),
    SweepTestSlice(
        slice_id="chat_state_memory_persistence",
        title="Chat state, memory, persistence, and diagnostics",
        category="memory/retrieval",
        owner_subsystem="persistence_and_memory",
        runtime_scope="shared",
        targets=(
            "tests/unit/test_interaction_history_service.py",
            "tests/unit/test_archive_manager.py",
            "tests/unit/test_session_context_service.py",
            "tests/unit/test_session_current.py",
            "tests/unit/test_session_history.py",
            "tests/unit/test_memory_classification.py",
            "tests/unit/test_memory_indexing.py",
            "tests/unit/test_memory_loop.py",
            "tests/unit/test_memory_promotion.py",
            "tests/unit/test_memory_write_policy.py",
            "tests/unit/test_diagnostics_service.py",
        ),
    ),
    SweepTestSlice(
        slice_id="tools_and_safety",
        title="Tool routing, execution, CLI, and safety",
        category="tools/execution",
        owner_subsystem="tooling_and_safety",
        runtime_scope="shared",
        targets=(
            "tests/unit/test_tool_signal_routing.py",
            "tests/unit/test_tool_phase_bundles.py",
            "tests/unit/test_safety_service.py",
            "tests/integration/test_tool_phase_execution.py",
            "tests/integration/test_cli_flows.py",
        ),
    ),
    SweepTestSlice(
        slice_id="validation_contract",
        title="Full validation contract and packaged smoke hook",
        category="packaged/runtime parity",
        owner_subsystem="validation",
        runtime_scope="shared",
        targets=("tests/integration/test_full_system_validation.py",),
    ),
    SweepTestSlice(
        slice_id="runtime_hardening",
        title="Runtime durability, persistence, startup, and desktop recovery",
        category="persistence/session-state",
        owner_subsystem="persistence_and_memory",
        runtime_scope="shared",
        targets=(
            "tests/unit/test_controller_init.py",
            "tests/unit/test_persistence_manager.py",
            "tests/unit/test_desktop_startup.py",
            "tests/unit/test_chat_experience_support.py",
        ),
    ),
    SweepTestSlice(
        slice_id="conversation_qa_stability",
        title="Long-conversation routing, tone, memory restraint, and scaffold QA",
        category="routing/reasoning",
        owner_subsystem="interaction_service",
        runtime_scope="shared",
        targets=(
            "tests/unit/test_long_conversation_evaluation.py",
            "tests/unit/test_decision_evaluation.py",
            "tests/unit/test_domain_router.py",
            "tests/unit/test_knowledge_service.py",
        ),
    ),
)

FAST_SWEEP_SLICE_IDS: tuple[str, ...] = (
    "ui_shell_regression",
    "routing_reasoning_regression",
    "tools_and_safety",
    "validation_contract",
)

FAST_SYSTEM_SWEEP_TEST_SLICES: tuple[SweepTestSlice, ...] = (
    SweepTestSlice(
        slice_id="ui_shell_regression",
        title="UI and desktop shell regression",
        category="ui/shell",
        owner_subsystem="desktop_shell",
        runtime_scope="source-only",
        targets=("tests/unit/test_chat_ui_support.py",),
    ),
    SweepTestSlice(
        slice_id="routing_reasoning_regression",
        title="Input, routing, reasoning, and interaction flow",
        category="routing/reasoning",
        owner_subsystem="interaction_service",
        runtime_scope="shared",
        targets=("tests/unit/test_reasoning_pipeline.py",),
    ),
    SweepTestSlice(
        slice_id="tools_and_safety",
        title="Tool routing, execution, CLI, and safety",
        category="tools/execution",
        owner_subsystem="tooling_and_safety",
        runtime_scope="shared",
        targets=(
            "tests/unit/test_tool_signal_routing.py",
            "tests/unit/test_safety_service.py",
        ),
    ),
    SweepTestSlice(
        slice_id="validation_contract",
        title="Full validation contract and packaged smoke hook",
        category="packaged/runtime parity",
        owner_subsystem="validation",
        runtime_scope="shared",
        targets=("tests/integration/test_full_system_validation.py",),
    ),
)


def run_full_system_sweep(
    *,
    repo_root: Path,
    data_root: Path,
    mode: str = DEFAULT_SWEEP_MODE,
    packaged_executable: Path | None = None,
    anh_probe_path: Path | None = None,
    fixes_applied: list[dict[str, Any]] | None = None,
    rerun_slice_ids: list[str] | None = None,
    force_fresh_packaged: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    repo_root = Path(repo_root).resolve()
    data_root = Path(data_root).resolve()
    fixes_applied = list(fixes_applied or [])
    normalized_mode = "full" if str(mode or "").strip().lower() == "full" else DEFAULT_SWEEP_MODE

    source_started = time.perf_counter()
    source_report = run_source_full_system_validation(
        repo_root=repo_root,
        data_root=data_root,
        anh_probe_path=_resolve_anh_probe(repo_root=repo_root, data_root=data_root, anh_probe_path=anh_probe_path),
        use_isolated_audit_data=True,
        audit_data_mode="clean",
    )
    source_duration_ms = round((time.perf_counter() - source_started) * 1000, 2)

    packaged_started = time.perf_counter()
    packaged_result = run_packaged_executable_smoke_validation(
        repo_root=repo_root,
        data_root=data_root,
        packaged_executable=packaged_executable,
        anh_probe_path=_resolve_anh_probe(repo_root=repo_root, data_root=data_root, anh_probe_path=anh_probe_path),
        allow_cached_report=not force_fresh_packaged,
    )
    packaged_duration_ms = round((time.perf_counter() - packaged_started) * 1000, 2)

    slices: list[dict[str, Any]] = []
    rerun_set = {item for item in (rerun_slice_ids or []) if item}
    for slice_def in _select_sweep_slices(mode=normalized_mode):
        slices.append(run_pytest_regression_slice(repo_root=repo_root, slice_def=slice_def))
        if slice_def.slice_id in rerun_set:
            rerun_result = run_pytest_regression_slice(repo_root=repo_root, slice_def=slice_def)
            rerun_result["rerun"] = True
            slices.append(rerun_result)

    findings = _collect_findings(
        source_report=source_report,
        packaged_result=packaged_result,
        slices=slices,
    )
    runtime_summary = _build_runtime_summary(
        source_report=source_report,
        packaged_result=packaged_result,
        findings=findings,
        fixes_applied=fixes_applied,
    )
    report = {
        "schema_version": SWEEP_SCHEMA_VERSION,
        "generated_at": _timestamp_now(),
        "label": "full_lumen_system_sweep",
        "audit_mode": normalized_mode,
        "repo_root": str(repo_root),
        "data_root": str(data_root),
        "audit_data_root": source_report.get("audit_context", {}).get("audit_data_root"),
        "audit_data_mode": source_report.get("audit_context", {}).get("audit_data_mode"),
        "source_validation": source_report,
        "packaged_validation": packaged_result.get("report"),
        "packaged_runtime": packaged_result,
        "targeted_slices": slices,
        "release_readiness": _build_release_readiness(
            source_report=source_report,
            packaged_result=packaged_result,
            slices=slices,
            runtime_summary=runtime_summary,
            anh_probe_path=_resolve_anh_probe(repo_root=repo_root, data_root=data_root, anh_probe_path=anh_probe_path),
        ),
        "findings": findings,
        "regressions_found": _group_findings_by_category(findings),
        "root_causes": _root_causes(findings),
        "fixes_applied": fixes_applied,
        "systems_verified_stable": _systems_verified_stable(source_report=source_report, packaged_result=packaged_result, slices=slices),
        "remaining_risks": _remaining_risks(source_report=source_report, packaged_result=packaged_result, slices=slices),
        "weaknesses_and_gaps": _weaknesses_and_gaps(source_report=source_report, packaged_result=packaged_result, slices=slices),
        "intentionally_unchanged": _intentionally_unchanged(findings=findings),
        "recommendations": {
            "v2_stabilization": _v2_recommendations(packaged_result=packaged_result, findings=findings),
            "v3_improvements": _v3_recommendations(findings=findings),
        },
        "runtime_summary": runtime_summary,
        "performance": {
            "source_validation_ms": source_duration_ms,
            "packaged_validation_ms": packaged_duration_ms,
            "targeted_slice_total_ms": round(sum(float(item.get("timing_ms") or 0.0) for item in slices), 2),
            "total_ms": round((time.perf_counter() - started) * 1000, 2),
        },
        "system_coherence": {
            "coherent": bool(runtime_summary.get("blocker_count", 0) == 0),
            "conclusion": _coherence_conclusion(runtime_summary),
        },
    }
    return report


def run_packaged_executable_smoke_validation(
    *,
    repo_root: Path,
    data_root: Path,
    packaged_executable: Path | None = None,
    anh_probe_path: Path | None = None,
    timeout_seconds: int = PACKAGED_SMOKE_TIMEOUT_SECONDS,
    allow_cached_report: bool = False,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    data_root = Path(data_root).resolve()
    packaged_executable = (
        Path(packaged_executable).resolve()
        if packaged_executable
        else _resolve_default_packaged_executable(repo_root=repo_root)
    )
    artifact = inspect_packaged_artifact(repo_root=repo_root, packaged_executable=packaged_executable)
    fallback_report_path = packaged_smoke_fallback_report_path(data_root=data_root)

    if not artifact["exists"]:
        return {
            "schema_version": SOURCE_VALIDATION_SCHEMA_VERSION,
            "artifact": artifact,
            "report": None,
            "returncode": None,
            "timing_ms": 0.0,
            "stdout_tail": "",
            "stderr_tail": "",
            "boot_status": "missing",
            "validation_origin": "missing_artifact",
            "cache_used": False,
        }

    if allow_cached_report and artifact.get("stale"):
        return {
            "schema_version": SOURCE_VALIDATION_SCHEMA_VERSION,
            "artifact": artifact,
            "report": None,
            "returncode": None,
            "timing_ms": 0.0,
            "stdout_tail": "",
            "stderr_tail": "",
            "boot_status": "stale_skipped",
            "report_source": "stale_skipped",
            "requested_report_missing": True,
            "requested_report_written": False,
            "fallback_report_written": False,
            "report_authority": "stale_skipped",
            "fallback_report_path": str(fallback_report_path),
            "validation_origin": "stale_artifact_skipped",
            "cache_used": False,
        }

    if allow_cached_report:
        cached_result = _load_cached_packaged_smoke_result(
            artifact=artifact,
            fallback_report_path=fallback_report_path,
        )
        if cached_result is not None:
            return cached_result

    with tempfile.TemporaryDirectory(prefix="lumen-packaged-smoke-") as temp_dir:
        report_path = Path(temp_dir) / "packaged_smoke_report.json"
        fallback_before_mtime = fallback_report_path.stat().st_mtime if fallback_report_path.exists() else None
        command = [
            str(packaged_executable),
            "--repo-root",
            str(repo_root),
            "--data-root",
            str(data_root),
            "--validation-smoke-report",
            str(report_path),
        ]
        if anh_probe_path is not None:
            command.extend(["--validation-anh-probe", str(anh_probe_path)])

        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            timing_ms = round((time.perf_counter() - started) * 1000, 2)
            report_source = "requested_path"
            requested_report_missing = False
            requested_report_written = report_path.exists()
            fallback_report_written = bool(
                fallback_report_path.exists()
                and (
                    fallback_before_mtime is None
                    or fallback_report_path.stat().st_mtime > fallback_before_mtime
                )
            )
            if report_path.exists():
                report = json.loads(report_path.read_text(encoding="utf-8"))
            elif fallback_report_path.exists():
                report = json.loads(fallback_report_path.read_text(encoding="utf-8"))
                report_source = "fallback_recovered"
                requested_report_missing = True
            else:
                report = None
                report_source = "missing"
                requested_report_missing = True
            if isinstance(report, dict):
                artifacts = report.get("report_artifacts")
                if isinstance(artifacts, dict):
                    requested_report_written = bool(artifacts.get("requested_report_written", requested_report_written))
                    fallback_report_written = bool(artifacts.get("fallback_report_written", fallback_report_written))
                    report_source = str(artifacts.get("report_authority") or report_source)
        except subprocess.TimeoutExpired as exc:
            return {
                "schema_version": SOURCE_VALIDATION_SCHEMA_VERSION,
                "artifact": artifact,
                "report": None,
                "returncode": None,
                "timing_ms": round((time.perf_counter() - started) * 1000, 2),
                "stdout_tail": _tail_text(exc.stdout or ""),
                "stderr_tail": _tail_text(exc.stderr or ""),
                "boot_status": "timed_out",
            }

    return {
        "schema_version": SOURCE_VALIDATION_SCHEMA_VERSION,
        "artifact": artifact,
        "report": report,
        "returncode": completed.returncode,
        "timing_ms": timing_ms,
        "stdout_tail": _tail_text(completed.stdout),
        "stderr_tail": _tail_text(completed.stderr),
        "boot_status": "ok" if report is not None else "failed_before_report",
        "report_source": report_source,
        "requested_report_missing": requested_report_missing,
        "requested_report_written": requested_report_written,
        "fallback_report_written": fallback_report_written,
        "report_authority": report_source,
        "fallback_report_path": str(fallback_report_path),
        "validation_origin": "fresh_run",
        "cache_used": False,
    }


def _load_cached_packaged_smoke_result(
    *,
    artifact: dict[str, Any],
    fallback_report_path: Path,
) -> dict[str, Any] | None:
    if artifact.get("stale") or not fallback_report_path.exists():
        return None
    try:
        report = json.loads(fallback_report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(report, dict) or str(report.get("label") or "") != "packaged_smoke_validation":
        return None

    executable_path = Path(str(artifact.get("path") or ""))
    if not executable_path.exists():
        return None
    if fallback_report_path.stat().st_mtime < executable_path.stat().st_mtime:
        return None

    artifacts = report.get("report_artifacts")
    requested_report_written = False
    fallback_report_written = True
    report_authority = "fallback_recovered"
    if isinstance(artifacts, dict):
        requested_report_written = bool(artifacts.get("requested_report_written", False))
        fallback_report_written = bool(artifacts.get("fallback_report_written", True))
        report_authority = str(artifacts.get("report_authority") or report_authority)

    return {
        "schema_version": SOURCE_VALIDATION_SCHEMA_VERSION,
        "artifact": artifact,
        "report": report,
        "returncode": 0,
        "timing_ms": 0.0,
        "stdout_tail": "",
        "stderr_tail": "",
        "boot_status": "ok",
        "report_source": report_authority,
        "requested_report_missing": not requested_report_written,
        "requested_report_written": requested_report_written,
        "fallback_report_written": fallback_report_written,
        "report_authority": report_authority,
        "fallback_report_path": str(fallback_report_path),
        "validation_origin": "cached_report",
        "cache_used": True,
        "cached_report_path": str(fallback_report_path),
        "cached_report_generated_at": str(report.get("generated_at") or ""),
    }


def run_pytest_regression_slice(
    *,
    repo_root: Path,
    slice_def: SweepTestSlice,
    timeout_seconds: int = PYTEST_SLICE_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", *slice_def.targets, "-q"]
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    timing_ms = round((time.perf_counter() - started) * 1000, 2)
    return {
        **asdict(slice_def),
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "timing_ms": timing_ms,
        "summary": _pytest_summary(completed.stdout, completed.stderr),
        "stdout_tail": _tail_text(completed.stdout),
        "stderr_tail": _tail_text(completed.stderr),
        "rerun": False,
    }


def inspect_packaged_artifact(*, repo_root: Path, packaged_executable: Path) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    packaged_executable = Path(packaged_executable).resolve()
    exists = packaged_executable.exists()
    latest_source = _latest_source_timestamp(repo_root)
    built_at = datetime.fromtimestamp(packaged_executable.stat().st_mtime, tz=UTC).isoformat() if exists else None
    stale = bool(exists and latest_source and packaged_executable.stat().st_mtime < latest_source.timestamp())
    return {
        "path": str(packaged_executable),
        "exists": exists,
        "built_at": built_at,
        "stale": stale,
        "latest_source_at": latest_source.isoformat() if latest_source else None,
    }


def _resolve_default_packaged_executable(*, repo_root: Path) -> Path:
    dist_dir = (repo_root / "dist").resolve()
    preferred = (
        dist_dir / "lumen.exe",
        dist_dir / "keep.exe",
        dist_dir / "main.exe",
    )
    for candidate in preferred:
        if candidate.exists():
            return candidate.resolve()
    exe_candidates = sorted(dist_dir.glob("*.exe"), key=lambda path: path.stat().st_mtime, reverse=True)
    if exe_candidates:
        return exe_candidates[0].resolve()
    return preferred[0].resolve()


def render_system_sweep_markdown(report: dict[str, Any]) -> str:
    runtime_summary = report.get("runtime_summary", {})
    packaged_runtime = report.get("packaged_runtime", {})
    recommendations = report.get("recommendations", {})
    lines = [
        "# Lumen Full System Sweep",
        "",
        f"- Schema version: `{report.get('schema_version')}`",
        f"- Generated at: `{report.get('generated_at')}`",
        f"- Audit mode: `{report.get('audit_mode', DEFAULT_SWEEP_MODE)}`",
        f"- Audit data mode: `{report.get('audit_data_mode', 'unknown')}`",
        f"- Source verdict: `{runtime_summary.get('source_verdict', 'unknown')}`",
        f"- Packaged verdict: `{runtime_summary.get('packaged_verdict', 'unknown')}`",
        f"- Parity verdict: `{runtime_summary.get('parity_verdict', 'unknown')}`",
        f"- Packaged validation origin: `{packaged_runtime.get('validation_origin', 'unknown')}`",
        f"- Report authority: `{packaged_runtime.get('report_authority', 'unknown')}`",
        f"- Blockers: `{runtime_summary.get('blocker_count', 0)}`",
        f"- Hotfixes applied: `{len(report.get('fixes_applied', []))}`",
    ]
    release = report.get("release_readiness") or {}
    if release:
        lines.extend(
            [
                "",
                "## Release Readiness",
                f"- Source verdict: `{release.get('source_verdict', 'unknown')}`",
                f"- Packaged verdict: `{release.get('packaged_verdict', 'unknown')}`",
                f"- Parity verdict: `{release.get('parity_verdict', 'unknown')}`",
                f"- Blockers: `{release.get('blocker_count', 'unknown')}`",
                f"- UI shell: `{release.get('ui_shell_status', 'unknown')}`",
                f"- Persistence/runtime: `{release.get('persistence_status', 'unknown')}`",
                f"- Tool routing: `{release.get('tool_routing_status', 'unknown')}`",
                f"- ANH live probe: `{release.get('anh_live_probe_status', 'unknown')}`",
            ]
        )
        intentional_skips = release.get("intentional_skips") or []
        if intentional_skips:
            lines.append(f"- Intentional skips: `{', '.join(str(item) for item in intentional_skips)}`")
    lines.extend(["", "## Regressions Found"])
    grouped = report.get("regressions_found", {})
    if grouped:
        for category, items in grouped.items():
            lines.append(f"### {category}")
            for item in items:
                lines.append(f"- {item['title']} [{item['severity']}] ({item['runtime_scope']})")
                if item.get("details"):
                    lines.append(f"  - {item['details']}")
    else:
        lines.append("- No regressions were recorded.")
    lines.extend(["", "## Root Causes"])
    root_causes = report.get("root_causes", [])
    if root_causes:
        lines.extend(f"- {item}" for item in root_causes)
    else:
        lines.append("- No likely root causes were identified.")
    lines.extend(["", "## Fixes Applied"])
    fixes_applied = report.get("fixes_applied", [])
    if fixes_applied:
        for item in fixes_applied:
            lines.append(f"- {item.get('summary') or item.get('title') or 'Unnamed fix'}")
    else:
        lines.append("- No hotfixes were applied during this sweep.")
    lines.extend(["", "## Systems Verified Stable"])
    stable = report.get("systems_verified_stable", [])
    if stable:
        lines.extend(f"- {item}" for item in stable)
    else:
        lines.append("- No systems were verified as stable.")
    lines.extend(["", "## Remaining Risks"])
    risks = report.get("remaining_risks", [])
    if risks:
        lines.extend(f"- {item}" for item in risks)
    else:
        lines.append("- No remaining risks were recorded.")
    lines.extend(["", "## Weaknesses and Gaps"])
    weaknesses = report.get("weaknesses_and_gaps", [])
    if weaknesses:
        lines.extend(f"- {item}" for item in weaknesses)
    else:
        lines.append("- No material weaknesses or gaps were recorded.")
    lines.extend(["", "## Intentionally Unchanged"])
    unchanged = report.get("intentionally_unchanged", [])
    if unchanged:
        lines.extend(f"- {item}" for item in unchanged)
    else:
        lines.append("- No intentionally unchanged items were recorded.")
    lines.extend(["", "## Recommendations", "### v2 Stabilization"])
    v2_items = recommendations.get("v2_stabilization", [])
    if v2_items:
        lines.extend(f"- {item}" for item in v2_items)
    else:
        lines.append("- No v2 stabilization recommendations.")
    lines.append("")
    lines.append("### v3 Improvements")
    v3_items = recommendations.get("v3_improvements", [])
    if v3_items:
        lines.extend(f"- {item}" for item in v3_items)
    else:
        lines.append("- No v3 improvement recommendations.")
    lines.extend(
        [
            "",
            "## System Coherence",
            f"- Coherent: `{report.get('system_coherence', {}).get('coherent', False)}`",
            f"- Conclusion: {report.get('system_coherence', {}).get('conclusion', 'No conclusion available.')}",
        ]
    )
    return "\n".join(lines) + "\n"


def write_system_sweep_artifacts(
    *,
    report: dict[str, Any],
    json_destination: Path,
    markdown_destination: Path | None = None,
) -> dict[str, Path]:
    paths = {"json": write_validation_report(report, json_destination)}
    if markdown_destination is not None:
        markdown_destination.parent.mkdir(parents=True, exist_ok=True)
        markdown_destination.write_text(render_system_sweep_markdown(report), encoding="utf-8")
        paths["markdown"] = markdown_destination
    return paths


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m lumen.validation.system_sweep")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--data-root", type=Path, default=Path.cwd() / "data")
    parser.add_argument("--json-report", type=Path, required=True)
    parser.add_argument("--markdown-report", type=Path)
    parser.add_argument("--packaged-executable", type=Path)
    parser.add_argument("--anh-probe", type=Path)
    parser.add_argument("--mode", choices=("fast", "full"), default=DEFAULT_SWEEP_MODE)
    parser.add_argument("--force-fresh-packaged", action="store_true")
    parser.add_argument(
        "--release-gate",
        action="store_true",
        help="Run the shippable release guardrail profile: full sweep plus fresh packaged validation.",
    )
    args = parser.parse_args(argv)

    report = run_full_system_sweep(
        repo_root=args.repo_root,
        data_root=args.data_root,
        mode="full" if args.release_gate else args.mode,
        packaged_executable=args.packaged_executable,
        anh_probe_path=args.anh_probe,
        force_fresh_packaged=bool(args.force_fresh_packaged or args.release_gate),
    )
    write_system_sweep_artifacts(
        report=report,
        json_destination=args.json_report,
        markdown_destination=args.markdown_report,
    )
    return 0 if report.get("runtime_summary", {}).get("blocker_count", 0) == 0 else 1


def _collect_findings(
    *,
    source_report: dict[str, Any],
    packaged_result: dict[str, Any],
    slices: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for report, runtime_scope in (
        (source_report, "source-only"),
        (packaged_result.get("report"), "packaged-only"),
    ):
        if not isinstance(report, dict):
            continue
        for check in report.get("checks", []):
            if check.get("passed"):
                continue
            findings.append(
                {
                    "title": str(check.get("title") or "Validation check failed"),
                    "category": _category_for_check(check),
                    "severity": _severity_label(check.get("severity")),
                    "runtime_scope": runtime_scope,
                    "details": str(check.get("details") or "").strip(),
                    "trigger": str(check.get("check_id") or "").strip(),
                    "likely_root_cause": _likely_root_cause(check),
                    "owner_subsystem": _owner_for_category(_category_for_check(check)),
                    "fix_status": "open",
                    "parity_status": "diverged" if runtime_scope != "shared" else "aligned",
                }
            )
    if packaged_result.get("artifact", {}).get("stale"):
        findings.append(
            {
                "title": "Packaged executable may be stale relative to source",
                "category": "packaged/runtime parity",
                "severity": "medium",
                "runtime_scope": "packaged-only",
                "details": "The packaged desktop artifact predates newer source files and should be treated as a parity risk.",
                "trigger": "packaged_artifact_staleness",
                "likely_root_cause": "The packaged executable was not rebuilt after source changes.",
                "owner_subsystem": "desktop_packaging",
                "fix_status": "open",
                "parity_status": "diverged",
            }
        )
    if packaged_result.get("boot_status") == "missing":
        missing_path = str(packaged_result.get("artifact", {}).get("path") or "dist/lumen.exe")
        findings.append(
            {
                "title": "Packaged desktop artifact is missing",
                "category": "packaged/runtime parity",
                "severity": "high",
                "runtime_scope": "packaged-only",
                "details": f"The sweep could not run packaged smoke validation because {missing_path} was not present.",
                "trigger": "missing_packaged_artifact",
                "likely_root_cause": "The desktop binary has not been built for the current workspace.",
                "owner_subsystem": "desktop_packaging",
                "fix_status": "open",
                "parity_status": "diverged",
            }
        )
    elif packaged_result.get("boot_status") == "timed_out":
        findings.append(
            {
                "title": "Packaged desktop validation timed out before report emission",
                "category": "packaged/runtime parity",
                "severity": "high",
                "runtime_scope": "packaged-only",
                "details": "The packaged smoke exceeded the configured timeout before writing a validation report.",
                "trigger": "packaged_boot_timeout",
                "likely_root_cause": "Packaged startup or validation is slower than the current harness timeout budget.",
                "owner_subsystem": "validation",
                "fix_status": "open",
                "parity_status": "diverged",
            }
        )
    elif packaged_result.get("boot_status") == "failed_before_report":
        findings.append(
            {
                "title": "Packaged desktop failed before validation report emission",
                "category": "packaged/runtime parity",
                "severity": "high",
                "runtime_scope": "packaged-only",
                "details": _tail_text(packaged_result.get("stderr_tail") or packaged_result.get("stdout_tail") or ""),
                "trigger": "packaged_boot_failure",
                "likely_root_cause": "Desktop runtime startup failed before the hidden smoke report could be written.",
                "owner_subsystem": "desktop_shell",
                "fix_status": "open",
                "parity_status": "diverged",
            }
        )
    if packaged_result.get("report_source") == "fallback_recovered":
        findings.append(
            {
                "title": "Packaged smoke report was recovered from fallback path",
                "category": "packaged/runtime parity",
                "severity": "medium",
                "runtime_scope": "packaged-only",
                "details": "The packaged runtime produced a fallback validation report under the data root because the requested report path was missing.",
                "trigger": "packaged_report_fallback_recovery",
                "likely_root_cause": "Frozen desktop report emission to the requested path remains unreliable in some runs, so fallback recovery is still needed.",
                "owner_subsystem": "validation",
                "fix_status": "open",
                "parity_status": "conditional",
            }
        )
    overlap_check = source_report.get("readiness", {}).get("doctor_checks", {}).get("behavioral_overlap_audit", {})
    if isinstance(overlap_check, dict):
        for item in overlap_check.get("drift_findings", []):
            findings.append(
                {
                    "title": f"Behavioral overlap drift: {item.get('pair') or item.get('surface') or 'unknown overlap'}",
                    "category": "routing/reasoning",
                    "severity": "medium",
                    "runtime_scope": "shared",
                    "details": str(item.get("details") or item.get("note") or "").strip(),
                    "trigger": "overlap_drift_detected",
                    "likely_root_cause": "Overlapping support surfaces no longer agree on status, routing, or user-visible transparency wording.",
                    "owner_subsystem": "diagnostics",
                    "fix_status": "open",
                    "parity_status": "shared-risk",
                }
            )
        for item in overlap_check.get("intentional_overlaps", []):
            findings.append(
                {
                    "title": f"Intentional overlap tracked: {item.get('pair') or 'overlap'}",
                    "category": "routing/reasoning",
                    "severity": "low",
                    "runtime_scope": "shared",
                    "details": str(item.get("note") or "").strip(),
                    "trigger": "intentional_overlap",
                    "likely_root_cause": "This overlap is deliberate and only needs consolidation if it starts drifting.",
                    "owner_subsystem": "diagnostics",
                    "fix_status": "deferred",
                    "parity_status": "aligned",
                }
            )
    for slice_result in slices:
        if slice_result.get("passed"):
            continue
        findings.append(
            {
                "title": slice_result.get("title"),
                "category": slice_result.get("category"),
                "severity": "high",
                "runtime_scope": slice_result.get("runtime_scope"),
                "details": slice_result.get("summary") or slice_result.get("stderr_tail") or "Pytest slice failed.",
                "trigger": slice_result.get("slice_id"),
                "likely_root_cause": "A targeted regression slice failed; inspect the captured pytest output for the first failing assertion.",
                "owner_subsystem": slice_result.get("owner_subsystem"),
                "fix_status": "open",
                "parity_status": "diverged" if slice_result.get("runtime_scope") != "shared" else "shared-risk",
            }
        )
    return findings


def _build_runtime_summary(
    *,
    source_report: dict[str, Any],
    packaged_result: dict[str, Any],
    findings: list[dict[str, Any]],
    fixes_applied: list[dict[str, Any]],
) -> dict[str, Any]:
    source_blockers = int(source_report.get("summary", {}).get("blocker_count", 0))
    packaged_report = packaged_result.get("report") or {}
    packaged_blockers = int(packaged_report.get("summary", {}).get("blocker_count", 0))
    blocker_count = sum(1 for item in findings if item.get("severity") == "high")
    source_verdict = "pass" if source_blockers == 0 else "fail"
    packaged_verdict = "pass"
    if packaged_result.get("boot_status") == "stale_skipped":
        packaged_verdict = "conditional"
    elif packaged_result.get("boot_status") != "ok":
        packaged_verdict = "fail"
    elif packaged_blockers > 0:
        packaged_verdict = "fail"
    elif packaged_result.get("report_source") == "fallback_recovered":
        packaged_verdict = "conditional"
    elif packaged_result.get("artifact", {}).get("stale"):
        packaged_verdict = "conditional"
    parity_verdict = "aligned"
    if packaged_result.get("boot_status") == "stale_skipped":
        parity_verdict = "conditional"
    elif packaged_result.get("boot_status") != "ok":
        parity_verdict = "diverged"
    elif source_verdict != "pass" or packaged_verdict == "fail":
        parity_verdict = "diverged"
    elif packaged_result.get("report_source") == "fallback_recovered":
        parity_verdict = "conditional"
    elif packaged_result.get("artifact", {}).get("stale"):
        parity_verdict = "conditional"
    return {
        "source_verdict": source_verdict,
        "packaged_verdict": packaged_verdict,
        "parity_verdict": parity_verdict,
        "blocker_count": blocker_count,
        "hotfix_count": len(fixes_applied),
    }


def _build_release_readiness(
    *,
    source_report: dict[str, Any],
    packaged_result: dict[str, Any],
    slices: list[dict[str, Any]],
    runtime_summary: dict[str, Any],
    anh_probe_path: Path | None,
) -> dict[str, Any]:
    source_anh = _check_details(source_report, "source_anh_runtime")
    packaged_report = packaged_result.get("report") or {}
    packaged_anh = _check_details(packaged_report, "packaged_anh_runtime")
    anh_status = "verified" if anh_probe_path is not None else "deferred_final_gate"
    if anh_probe_path is None:
        intentional_skips = ["ANH live MAST probe deferred to final release-candidate gate."]
    else:
        intentional_skips = []
    return {
        "source_verdict": runtime_summary.get("source_verdict", "unknown"),
        "packaged_verdict": runtime_summary.get("packaged_verdict", "unknown"),
        "parity_verdict": runtime_summary.get("parity_verdict", "unknown"),
        "blocker_count": runtime_summary.get("blocker_count", 0),
        "ui_shell_status": _slice_status(slices, "ui_shell_regression"),
        "persistence_status": _slice_status(slices, "chat_state_memory_persistence", "runtime_hardening"),
        "tool_routing_status": _slice_status(slices, "tools_and_safety"),
        "anh_live_probe_status": anh_status,
        "anh_source_details": source_anh,
        "anh_packaged_details": packaged_anh,
        "intentional_skips": intentional_skips,
        "fresh_packaged_validation": packaged_result.get("validation_origin") == "fresh_run",
        "packaged_report_authority": packaged_result.get("report_authority", "unknown"),
    }


def _slice_status(slices: list[dict[str, Any]], *slice_ids: str) -> str:
    matched = [item for item in slices if item.get("slice_id") in slice_ids]
    if not matched:
        return "not_run"
    return "pass" if all(item.get("passed") for item in matched) else "fail"


def _check_details(report: dict[str, Any], check_id: str) -> str:
    for check in report.get("checks", []) if isinstance(report, dict) else []:
        if check.get("check_id") == check_id:
            return str(check.get("details") or "")
    return ""


def _group_findings_by_category(findings: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in findings:
        grouped.setdefault(str(item.get("category") or "uncategorized"), []).append(item)
    return grouped


def _root_causes(findings: list[dict[str, Any]]) -> list[str]:
    seen: list[str] = []
    for item in findings:
        cause = str(item.get("likely_root_cause") or "").strip()
        if cause and cause not in seen:
            seen.append(cause)
    return seen


def _systems_verified_stable(
    *,
    source_report: dict[str, Any],
    packaged_result: dict[str, Any],
    slices: list[dict[str, Any]],
) -> list[str]:
    stable: list[str] = []
    if int(source_report.get("summary", {}).get("blocker_count", 0)) == 0:
        stable.append("Source full-system validation passed without blockers.")
    packaged_report = packaged_result.get("report") or {}
    if packaged_result.get("boot_status") == "ok" and int(packaged_report.get("summary", {}).get("blocker_count", 0)) == 0:
        stable.append("Packaged smoke validation reached the desktop runtime and completed without blockers.")
    for item in slices:
        if item.get("passed"):
            stable.append(f"{item.get('title')} passed.")
    return stable


def _remaining_risks(
    *,
    source_report: dict[str, Any],
    packaged_result: dict[str, Any],
    slices: list[dict[str, Any]],
) -> list[str]:
    risks: list[str] = []
    for item in source_report.get("refinement_qa", []):
        risks.append(str(item))
    packaged_report = packaged_result.get("report") or {}
    for item in packaged_report.get("refinement_qa", []):
        risks.append(str(item))
    if packaged_result.get("artifact", {}).get("stale"):
        risks.append("Packaged executable is older than the latest source changes, so parity confidence is conditional until rebuild.")
    for item in slices:
        if not item.get("passed"):
            risks.append(f"{item.get('title')} is still failing and needs follow-up.")
    return risks


def _weaknesses_and_gaps(
    *,
    source_report: dict[str, Any],
    packaged_result: dict[str, Any],
    slices: list[dict[str, Any]],
) -> list[str]:
    weaknesses: list[str] = []
    if packaged_result.get("report_source") == "fallback_recovered":
        weaknesses.append(
            "Packaged validation report emission still needs fallback recovery, so requested-path report writing in frozen mode is not fully trustworthy yet."
        )
    overlap_check = source_report.get("readiness", {}).get("doctor_checks", {}).get("behavioral_overlap_audit", {})
    if isinstance(overlap_check, dict) and overlap_check.get("drift_findings"):
        weaknesses.append("Behavioral overlap drift was detected across intentionally shared capability surfaces.")
    if packaged_result.get("artifact", {}).get("stale"):
        weaknesses.append("Packaged parity remains sensitive to rebuild discipline after runtime or shell changes.")
    source_report_refinement = list(source_report.get("refinement_qa", []) or [])
    packaged_report = packaged_result.get("report") or {}
    packaged_refinement = list(packaged_report.get("refinement_qa", []) or [])
    weaknesses.extend(str(item) for item in source_report_refinement)
    weaknesses.extend(str(item) for item in packaged_refinement)
    if any(not item.get("passed") for item in slices):
        weaknesses.append("At least one targeted regression slice still needs follow-up, so the audit is not fully closed.")
    return weaknesses


def _intentionally_unchanged(*, findings: list[dict[str, Any]]) -> list[str]:
    unchanged: list[str] = [
        "Harmless duplicate or overlapping capability surfaces were left in place when they did not create conflicting runtime behavior.",
        "No architecture-wide refactor was performed; only audit-blocking or regression-causing behavior should be changed in this pass.",
    ]
    if not any(item.get("category") == "ui/shell" for item in findings):
        unchanged.append("The desktop UI architecture and current interaction design were preserved because no release-blocking shell redesign issue was confirmed in this pass.")
    return unchanged


def _v2_recommendations(*, packaged_result: dict[str, Any], findings: list[dict[str, Any]]) -> list[str]:
    recommendations: list[str] = []
    if packaged_result.get("artifact", {}).get("stale"):
        recommendations.append("Rebuild the packaged desktop artifact after shell or runtime changes so packaged parity stays trustworthy.")
    if any(item.get("category") == "memory/retrieval" for item in findings):
        recommendations.append("Keep watching SQLite-first memory and persistence diagnostics for fallback drift after regression fixes.")
    if any(item.get("category") == "packaged/runtime parity" for item in findings):
        recommendations.append("Keep source and packaged validation in the same release checklist so parity regressions are caught before shipment.")
    return recommendations


def _v3_recommendations(*, findings: list[dict[str, Any]]) -> list[str]:
    recommendations: list[str] = []
    if any(item.get("category") == "performance/stability" for item in findings):
        recommendations.append("Add deeper performance instrumentation only if latency regressions become recurrent, not as a default subsystem.")
    recommendations.append("If the UI sweep grows further, add a lightweight scripted desktop smoke layer for navigation and state transitions.")
    return recommendations


def _coherence_conclusion(runtime_summary: dict[str, Any]) -> str:
    if runtime_summary.get("blocker_count", 0) == 0 and runtime_summary.get("parity_verdict") == "aligned":
        return "Lumen currently behaves like one coherent system across source and packaged runtimes."
    if runtime_summary.get("blocker_count", 0) == 0:
        return "Lumen appears coherent in the validated paths, but packaged parity is still conditional."
    return "Lumen still shows cross-subsystem regressions or parity gaps and should not be treated as fully unified yet."


def _resolve_anh_probe(*, repo_root: Path, data_root: Path, anh_probe_path: Path | None) -> Path | None:
    if anh_probe_path is not None and Path(anh_probe_path).exists():
        return Path(anh_probe_path).resolve()
    return None


def _latest_source_timestamp(repo_root: Path) -> datetime | None:
    candidates = [
        *repo_root.glob("src/**/*.py"),
        *repo_root.glob("tool_bundles/**/*.py"),
        *repo_root.glob("*.spec"),
        repo_root / "pyproject.toml",
        repo_root / "README.md",
    ]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return None
    latest = max(existing, key=lambda path: path.stat().st_mtime)
    return datetime.fromtimestamp(latest.stat().st_mtime, tz=UTC)


def _timestamp_now() -> str:
    return datetime.now(tz=UTC).isoformat()


def _tail_text(text: str, *, max_lines: int = 12) -> str:
    lines = [line.rstrip() for line in str(text or "").splitlines() if line.strip()]
    return "\n".join(lines[-max_lines:])


def _pytest_summary(stdout: str, stderr: str) -> str:
    combined = "\n".join([line for line in [stdout, stderr] if line]).strip()
    if not combined:
        return "No pytest output captured."
    lines = [line.strip() for line in combined.splitlines() if line.strip()]
    for line in reversed(lines):
        if " passed" in line or " failed" in line or " error" in line or " skipped" in line:
            return line
    return lines[-1]


def _severity_label(raw: Any) -> str:
    normalized = str(raw or "").strip().lower()
    if normalized == "blocker":
        return "high"
    if normalized == "refinement":
        return "medium"
    return "low"


def _category_for_check(check: dict[str, Any]) -> str:
    check_id = str(check.get("check_id") or "")
    title = str(check.get("title") or "").lower()
    if "greeting" in check_id or "mode identity" in title:
        return "interaction-mode identity"
    if "clarification" in check_id or "followup" in check_id:
        return "chat-state lifecycle"
    if "anh" in check_id or "tool" in title or "math_smoke" in check_id or "content_status" in check_id:
        return "tools/execution"
    if "known_topics" in check_id or "explanation" in title or "failure honesty" in title:
        return "routing/reasoning"
    return "packaged/runtime parity" if str(check.get("phase")) == "packaged" else "routing/reasoning"


def _owner_for_category(category: str) -> str:
    owners = {
        "ui/shell": "desktop_shell",
        "routing/reasoning": "interaction_service",
        "interaction-mode identity": "response_shaping",
        "chat-state lifecycle": "desktop_shell",
        "memory/retrieval": "persistence_and_memory",
        "tools/execution": "tooling_and_safety",
        "persistence/session-state": "persistence_and_memory",
        "performance/stability": "validation",
        "safety/boundaries": "tooling_and_safety",
        "packaged/runtime parity": "validation",
    }
    return owners.get(category, "validation")


def _select_sweep_slices(*, mode: str) -> tuple[SweepTestSlice, ...]:
    if mode == "full":
        return SYSTEM_SWEEP_TEST_SLICES
    return FAST_SYSTEM_SWEEP_TEST_SLICES


def _likely_root_cause(check: dict[str, Any]) -> str:
    category = _category_for_check(check)
    if category == "tools/execution":
        return "Tool routing or execution no longer matches the expected end-to-end contract for this prompt."
    if category == "chat-state lifecycle":
        return "Follow-up or clarification state handling drifted from the expected conversation-thread behavior."
    if category == "interaction-mode identity":
        return "Mode-specific response shaping no longer preserves clear delivery differences."
    return "Reasoning or response composition drifted from the expected validation contract."


if __name__ == "__main__":
    raise SystemExit(main())
