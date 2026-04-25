from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
from uuid import uuid4
from typing import Any

from lumen.app.controller import AppController
from lumen.app.settings import AUDIT_DATA_ROOT_OVERRIDE_ENV


REQUIRED_BUNDLES = {
    "anh",
    "content",
    "data",
    "experiment",
    "invent",
    "knowledge",
    "math",
    "paper",
    "physics",
    "simulate",
    "system",
    "viz",
    "astronomy",
    "workspace",
}

REQUIRED_CAPABILITIES = {
    "data.describe",
    "data.visualize",
    "paper.search",
    "paper.summary",
    "paper.compare",
    "paper.extract_methods",
    "simulate.system",
    "simulate.orbit",
    "experiment.design",
    "invent.generate_concepts",
    "physics.energy_model",
    "astronomy.orbit_profile",
}

SCAFFOLD_RESIDUE = (
    "first pass",
    "next: summarize",
    "state the topic in one concise sentence",
    "identify the main topic",
    "identify the two or more options being compared",
)


@dataclass(slots=True)
class ValidationCheck:
    phase: str
    check_id: str
    title: str
    passed: bool
    severity: str
    details: str
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "check_id": self.check_id,
            "title": self.title,
            "passed": self.passed,
            "severity": self.severity,
            "details": self.details,
            "evidence": self.evidence,
        }


def collect_readiness_truth_table(
    *,
    repo_root: Path,
    data_root: Path,
    execution_mode: str = "source",
    anh_probe_path: Path | None = None,
) -> dict[str, Any]:
    controller = AppController(repo_root=repo_root, data_root=data_root, execution_mode=execution_mode)
    doctor = controller.build_doctor_report()
    bundles = controller.list_tools()
    bundle_ids = set(bundles)
    capabilities = controller.list_app_capabilities()
    capability_ids = set(capabilities)
    readme_text = _read_text(repo_root / "README.md")

    source_surface = {
        "reasoning_spine": _probe_reasoning_spine(controller),
        "state_aware_routing": _probe_state_aware_followup(controller),
        "mode_integrated_behavior": _probe_mode_integration(controller),
        "tool_as_reasoning_step": _probe_tool_reasoning_integration(controller),
        "nlg_response_shaping": _probe_nlg_modes(controller),
        "required_bundles_present": REQUIRED_BUNDLES.issubset(bundle_ids),
        "required_capabilities_present": REQUIRED_CAPABILITIES.issubset(capability_ids),
        "simulate_bundle_present": "simulate" in bundle_ids,
        "experiment_bundle_present": "experiment" in bundle_ids,
        "invent_bundle_present": "invent" in bundle_ids,
        "domain_wrappers_present": {
            "physics.energy_model": "physics.energy_model" in capability_ids,
            "astronomy.orbit_profile": "astronomy.orbit_profile" in capability_ids,
        },
    }

    docs_alignment = {
        "readme_present": bool(readme_text),
        "readme_mentions_required_bundles": all(f"`{bundle}`" in readme_text for bundle in REQUIRED_BUNDLES),
        "readme_mentions_modes": all(token in readme_text for token in ("`collab`", "`default`", "`direct`")),
        "readme_mentions_soft_stop": "soft `Stop`" in readme_text or "soft stop" in readme_text.lower(),
        "readme_mentions_content_caveat": "provider" in readme_text.lower() and "content" in readme_text.lower(),
        "readme_mentions_anh_live": "ANH is part of the active `1.1` runtime surface" in readme_text,
    }

    provider = controller.model_provider
    provider_status = _content_runtime_status(controller)
    anh_status = _anh_runtime_status(controller, anh_probe_path=anh_probe_path)

    return {
        "execution_mode": execution_mode,
        "repo_root": str(repo_root),
        "data_root": str(data_root),
        "bundle_ids": sorted(bundle_ids),
        "capability_ids": sorted(capability_ids),
        "source_surface": source_surface,
        "docs_alignment": docs_alignment,
        "runtime_truth": {
            "content": provider_status,
            "anh": anh_status,
            "provider": {
                "provider_id": getattr(provider, "provider_id", None),
                "deployment_mode": controller.settings.deployment_mode,
            },
        },
        "doctor_checks": {
            item["name"]: {
                "status": item["status"],
                "details": item["details"],
                "extra": item.get("extra", {}),
            }
            for item in doctor.get("checks", [])
        },
    }


def write_validation_report(report: dict[str, Any], destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return destination


def packaged_smoke_fallback_report_path(*, data_root: Path) -> Path:
    return Path(data_root).resolve() / "validation" / "packaged_smoke_latest.json"


def prepare_validation_audit_data_root(
    *,
    source_data_root: Path,
    audit_workspace_root: Path | None = None,
    audit_data_mode: str = "clean",
) -> dict[str, str]:
    source_data_root = Path(source_data_root).resolve()
    mode = "isolated_copy" if str(audit_data_mode or "").strip().lower() == "isolated_copy" else "clean"
    workspace_root = (
        Path(audit_workspace_root).resolve()
        if audit_workspace_root is not None
        else source_data_root / "validation" / "audit_workspaces"
    )
    audit_root = workspace_root / f"audit-{mode}-{uuid4().hex[:12]}"
    audit_root.mkdir(parents=True, exist_ok=True)

    _copy_optional_tree(source_data_root / "examples", audit_root / "examples")

    if mode == "isolated_copy":
        for relative in (
            "sessions",
            "interactions",
            "archive",
            "personal_memory",
            "research_notes",
            "research_artifacts",
            "labeled_datasets",
            "tool_runs",
        ):
            _copy_optional_tree(source_data_root / relative, audit_root / relative)

    return {
        "audit_data_root": str(audit_root),
        "audit_data_mode": mode,
        "source_data_root": str(source_data_root),
    }


def render_qa_readout_markdown(report: dict[str, Any]) -> str:
    readiness = report.get("readiness", {})
    runtime = readiness.get("runtime_truth", {})
    audit_context = report.get("audit_context", {})
    stable = report.get("stable_ready", [])
    refinement = report.get("refinement_qa", [])
    blockers = report.get("blockers", [])
    coherent = "yes" if not blockers else "not yet"
    submission = "needs blocker fixes first" if blockers else "ready for refinement QA and submission framing"
    lines = [
        "# Lumen v2 QA Readout",
        "",
        f"- Validation label: `{report.get('label')}`",
        f"- Execution mode: `{report.get('execution_mode')}`",
        f"- Audit data mode: `{audit_context.get('audit_data_mode', 'live')}`",
        f"- Content status: `{runtime.get('content', {}).get('label', 'unknown')}`",
        f"- ANH status: `{runtime.get('anh', {}).get('label', 'unknown')}`",
        "",
        "## Stable / Ready",
    ]
    if stable:
        lines.extend(f"- {item}" for item in stable)
    else:
        lines.append("- No stable areas were recorded.")
    lines.extend(["", "## Refinement QA"])
    if refinement:
        lines.extend(f"- {item}" for item in refinement)
    else:
        lines.append("- No refinement-only issues were recorded.")
    lines.extend(["", "## Blockers"])
    if blockers:
        lines.extend(f"- {item}" for item in blockers)
    else:
        lines.append("- No blockers were recorded.")
    lines.extend(
        [
            "",
            "## Final Answer",
            f"- Does Lumen behave like one coherent system? `{coherent}`",
            f"- Does it still need anything else before submission? `{submission}`",
            f"- Is the advertised surface honest in this runtime? `{'yes' if readiness.get('docs_alignment', {}).get('readme_mentions_required_bundles') else 'partially'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def run_source_full_system_validation(
    *,
    repo_root: Path,
    data_root: Path,
    anh_probe_path: Path | None = None,
    use_isolated_audit_data: bool = True,
    audit_data_mode: str = "clean",
    audit_workspace_root: Path | None = None,
) -> dict[str, Any]:
    effective_data_root = Path(data_root).resolve()
    audit_context = {
        "audit_data_root": str(effective_data_root),
        "audit_data_mode": "live",
        "source_data_root": str(Path(data_root).resolve()),
    }
    if use_isolated_audit_data:
        audit_context = prepare_validation_audit_data_root(
            source_data_root=effective_data_root,
            audit_workspace_root=audit_workspace_root,
            audit_data_mode=audit_data_mode,
        )
        effective_data_root = Path(audit_context["audit_data_root"])

    with _temporary_audit_data_root(effective_data_root):
        controller = AppController(repo_root=repo_root, data_root=effective_data_root, execution_mode="source")
        fixtures = _build_fixture_inputs(effective_data_root)
        readiness = collect_readiness_truth_table(
            repo_root=repo_root,
            data_root=effective_data_root,
            execution_mode="source",
            anh_probe_path=anh_probe_path,
        )
        checks: list[ValidationCheck] = []

        _run_conversational_core_checks(controller, checks)
        _run_explanation_checks(controller, checks)
        _run_tool_integration_checks(controller, checks, fixtures)
        _run_anh_check(controller, checks, anh_probe_path=anh_probe_path)
        _run_failure_honesty_checks(controller, checks)
        _run_mode_identity_checks(controller, checks)

    return _finalize_report(
        label="source_full_validation",
        execution_mode="source",
        readiness=readiness,
        checks=checks,
        audit_context=audit_context,
    )


def run_packaged_smoke_validation(
    *,
    repo_root: Path,
    data_root: Path,
    execution_mode: str,
    anh_probe_path: Path | None = None,
) -> dict[str, Any]:
    controller = AppController(repo_root=repo_root, data_root=data_root, execution_mode=execution_mode)
    readiness = collect_readiness_truth_table(
        repo_root=repo_root,
        data_root=data_root,
        execution_mode=execution_mode,
        anh_probe_path=anh_probe_path,
    )
    checks: list[ValidationCheck] = []

    _record_mode_greeting_checks(controller, checks, phase="packaged")
    _record_known_topic_check(controller, checks, phase="packaged")
    _record_followup_continuity_check(controller, checks, phase="packaged")
    _record_clarification_check(controller, checks, phase="packaged")
    _record_math_tool_check(controller, checks, phase="packaged")
    _record_failure_case_check(controller, checks, phase="packaged")
    _record_content_status_check(controller, checks, phase="packaged")
    _record_anh_runtime_check(controller, checks, phase="packaged", anh_probe_path=anh_probe_path)

    return _finalize_report(
        label="packaged_smoke_validation",
        execution_mode=execution_mode,
        readiness=readiness,
        checks=checks,
    )


def _run_conversational_core_checks(controller: AppController, checks: list[ValidationCheck]) -> None:
    _record_mode_greeting_checks(controller, checks, phase="source")
    _record_known_topic_check(controller, checks, phase="source")
    _record_followup_continuity_check(controller, checks, phase="source")
    _record_clarification_check(controller, checks, phase="source")


def _run_explanation_checks(controller: AppController, checks: list[ValidationCheck]) -> None:
    cases = [
        ("explain_simple", "explain entropy simply", ("entropy", "spread"), ()),
        ("explain_relational", "explain entropy in relation to black holes", ("entropy", "black hole"), ("first pass",)),
        ("compare_grounded", "compare black holes and neutron stars", ("black hole", "neutron star"), ("first pass",)),
        ("step_by_step", "break entropy down step by step", ("entropy",), ("first pass",)),
    ]
    for check_id, prompt, anchors, forbidden in cases:
        response = controller.ask(prompt=prompt, session_id=f"source-{check_id}")
        summary = _summary_text(response)
        passed = (
            response.get("mode") == "research"
            and all(anchor in summary.lower() for anchor in anchors)
            and not any(token in summary.lower() for token in forbidden)
        )
        _append_check(
            checks,
            phase="source",
            check_id=check_id,
            title=f"Explanation quality: {prompt}",
            passed=passed,
            severity="blocker" if not passed else "info",
            details=f"mode={response.get('mode')} kind={response.get('kind')}",
            evidence={"prompt": prompt, "response": _snapshot_response(response)},
        )


def _run_tool_integration_checks(
    controller: AppController,
    checks: list[ValidationCheck],
    fixtures: dict[str, Path],
) -> None:
    scenarios = [
        ("knowledge_links", "how do these relate: voltage, current, resistance", None, "knowledge", "link"),
        ("system_refactor", "suggest a refactor for this architecture", None, "system", "suggest.refactor"),
        ("math_solve", "solve 2x + 5 = 13", None, "math", "solve_equation"),
        ("math_natural", "solve 3x^2 + 2x - 5 = 0", None, "math", "solve_equation"),
        ("data_describe", "describe data", fixtures["sample_csv"], "data", "describe"),
        ("viz_graph", "render graph", fixtures["graph_json"], "viz", "graph"),
        ("paper_search", "search papers black holes", fixtures["paper_catalog"], "paper", "search"),
        ("paper_summary", "summarize paper", fixtures["paper_one"], "paper", "summary"),
        ("paper_compare", "compare papers Methods: A telescope survey. Results: One. ; Methods: A simulation study. Results: Two.", None, "paper", "compare"),
        ("paper_methods", "extract methods", fixtures["paper_one"], "paper", "extract.methods"),
        ("simulate_system", "simulate system with initial value 10 growth rate 0.2 damping rate 0.05 for 12 steps", None, "simulate", "system"),
        ("experiment_design", "design experiment for whether light intensity changes photosynthesis", None, "experiment", "design"),
        ("invent_concepts", "generate concepts for a propulsion concept under these constraints: low mass; high durability", None, "invent", "generate_concepts"),
        ("physics_wrapper", "model energy with mass 2 velocity 3 height 5", None, "physics", "energy_model"),
        ("astronomy_wrapper", "analyze orbit profile with semi-major axis 3 and eccentricity 0.2", None, "astronomy", "orbit_profile"),
    ]

    for check_id, prompt, input_path, tool_id, capability in scenarios:
        response = controller.ask(prompt=prompt, input_path=input_path, session_id=f"source-{check_id}")
        summary = _summary_text(response)
        execution = response.get("tool_execution") or {}
        integrated = bool(summary) and not any(token in summary.lower() for token in ("traceback", "exception"))
        passed = (
            response.get("mode") == "tool"
            and execution.get("tool_id") == tool_id
            and str(execution.get("capability")) == capability
            and integrated
        )
        severity = "blocker" if not passed else "info"
        if passed and _contains_scaffold_residue(summary):
            passed = False
            severity = "refinement"
        _append_check(
            checks,
            phase="source",
            check_id=check_id,
            title=f"Tool integration: {prompt}",
            passed=passed,
            severity=severity,
            details=f"mode={response.get('mode')} tool={execution.get('tool_id')} capability={execution.get('capability')}",
            evidence={"prompt": prompt, "input_path": str(input_path) if input_path else None, "response": _snapshot_response(response)},
        )

    _record_content_status_check(controller, checks, phase="source")


def _run_anh_check(
    controller: AppController,
    checks: list[ValidationCheck],
    *,
    anh_probe_path: Path | None,
) -> None:
    _record_anh_runtime_check(controller, checks, phase="source", anh_probe_path=anh_probe_path)


def _run_failure_honesty_checks(controller: AppController, checks: list[ValidationCheck]) -> None:
    cases = [
        ("fake_concept", "tell me about hyperdimensional thermal lattice theory", ("don't have enough", "because")),
        ("unknown_prompt", "help me with this", ("don't have enough", "because")),
        ("missing_runtime", "Generate me 5 content ideas on topic black holes", ("unavailable", "configured")),
        ("missing_structured_input", "find inconsistencies in this", ("didn't run the tool",)),
    ]
    for check_id, prompt, anchors in cases:
        response = controller.ask(prompt=prompt, session_id=f"source-{check_id}")
        summary = _summary_text(response).lower()
        passed = all(anchor in summary for anchor in (item.lower() for item in anchors))
        _append_check(
            checks,
            phase="source",
            check_id=check_id,
            title=f"Failure honesty: {prompt}",
            passed=passed,
            severity="blocker" if not passed else "info",
            details=f"mode={response.get('mode')} kind={response.get('kind')}",
            evidence={"prompt": prompt, "response": _snapshot_response(response)},
        )


def _run_mode_identity_checks(controller: AppController, checks: list[ValidationCheck]) -> None:
    _record_mode_invariance(
        controller,
        checks,
        phase="source",
        check_id="mode_identity_knowledge",
        prompt="what is entropy?",
        anchors=("entropy", "spread"),
        require_distinct_delivery=True,
    )
    _record_mode_invariance(
        controller,
        checks,
        phase="source",
        check_id="mode_identity_tool",
        prompt="solve 2x + 5 = 13",
        anchors=("solved",),
        expect_mode="tool",
        require_distinct_delivery=False,
    )
    _record_mode_invariance(
        controller,
        checks,
        phase="source",
        check_id="mode_identity_clarification",
        prompt="what route makes sense here",
        anchors=("summary", "comparison", "continue"),
        expect_mode="clarification",
        require_distinct_delivery=True,
    )
    _record_mode_invariance(
        controller,
        checks,
        phase="source",
        check_id="mode_identity_failure",
        prompt="Generate me 5 content ideas on topic black holes",
        anchors=("unavailable", "configured"),
        expect_mode="tool",
        require_distinct_delivery=True,
    )


def _record_mode_greeting_checks(controller: AppController, checks: list[ValidationCheck], *, phase: str) -> None:
    results = []
    for style in ("collab", "default", "direct"):
        session_id = f"{phase}-greeting-{style}"
        controller.set_session_profile(session_id, interaction_style=style)
        response = controller.ask(prompt="hello", session_id=session_id)
        results.append((style, response))
    passed = all(_summary_text(response) for _, response in results) and len({_summary_text(r) for _, r in results}) >= 2
    _append_check(
        checks,
        phase=phase,
        check_id=f"{phase}_greeting_modes",
        title="Greeting / first contact across modes",
        passed=passed,
        severity="refinement" if not passed else "info",
        details="Modes should feel distinct without changing core intent.",
        evidence={style: _snapshot_response(response) for style, response in results},
    )


def _record_known_topic_check(controller: AppController, checks: list[ValidationCheck], *, phase: str) -> None:
    prompts = [
        "what is entropy?",
        "what is the Great Attractor?",
        "who was George Washington?",
    ]
    snapshots: dict[str, Any] = {}
    passed = True
    for index, prompt in enumerate(prompts, start=1):
        response = controller.ask(prompt=prompt, session_id=f"{phase}-known-{index}")
        summary = _summary_text(response).lower()
        snapshots[prompt] = _snapshot_response(response)
        if response.get("mode") != "research" or response.get("mode") == "clarification":
            passed = False
        if prompt == "what is entropy?" and "entropy" not in summary:
            passed = False
        if prompt == "what is the Great Attractor?" and "great attractor" not in summary:
            passed = False
        if prompt == "who was George Washington?" and "washington" not in summary:
            passed = False
    _append_check(
        checks,
        phase=phase,
        check_id=f"{phase}_known_topics",
        title="Known-topic direct answers",
        passed=passed,
        severity="blocker" if not passed else "info",
        details="Known topics should answer directly without unnecessary clarification.",
        evidence=snapshots,
    )


def _record_followup_continuity_check(controller: AppController, checks: list[ValidationCheck], *, phase: str) -> None:
    session_id = f"{phase}-followup"
    controller.set_session_profile(session_id, interaction_style="collab")
    first = controller.ask(prompt="what is entropy?", session_id=session_id)
    simple = controller.ask(prompt="explain that simply", session_id=session_id)
    deep = controller.ask(prompt="go deeper", session_id=session_id)
    compare = controller.ask(prompt="compare that to black holes", session_id=session_id)

    pivot_session = f"{phase}-pivot"
    controller.ask(prompt="what is entropy?", session_id=pivot_session)
    pivot = controller.ask(prompt="no, let's explore black holes instead", session_id=pivot_session)

    plan_session = f"{phase}-planning-followup"
    controller.ask(prompt="create a migration plan for lumen", session_id=plan_session)
    reference = controller.ask(prompt="what about that", session_id=plan_session)

    passed = (
        "entropy" in _summary_text(first).lower()
        and "entropy" in _summary_text(simple).lower()
        and "entropy" in _summary_text(deep).lower()
        and "black hole" in _summary_text(compare).lower()
        and "black hole" in _summary_text(pivot).lower()
        and reference.get("mode") == "planning"
        and reference.get("resolution_strategy") == "reference_follow_up"
    )
    _append_check(
        checks,
        phase=phase,
        check_id=f"{phase}_followup_continuity",
        title="Follow-up continuity, continuation, and pivot handling",
        passed=passed,
        severity="blocker" if not passed else "info",
        details="Follow-ups should stay anchored to the current reasoning state rather than resetting.",
        evidence={
            "first": _snapshot_response(first),
            "simple": _snapshot_response(simple),
            "deep": _snapshot_response(deep),
            "compare": _snapshot_response(compare),
            "pivot": _snapshot_response(pivot),
            "reference": _snapshot_response(reference),
        },
    )


def _record_clarification_check(controller: AppController, checks: list[ValidationCheck], *, phase: str) -> None:
    snapshots: dict[str, Any] = {}
    passed = True
    for style in ("collab", "default", "direct"):
        session_id = f"{phase}-clarification-{style}"
        controller.set_session_profile(session_id, interaction_style=style)
        response = controller.ask(prompt="what route makes sense here", session_id=session_id)
        snapshots[style] = _snapshot_response(response)
        summary = _summary_text(response).lower()
        if response.get("mode") != "clarification":
            passed = False
            continue
        options = [str(item) for item in response.get("options") or []]
        if options != ["Summary", "Comparison", "Continue"]:
            passed = False
        if "research." in summary or "planning." in summary:
            passed = False
    _append_check(
        checks,
        phase=phase,
        check_id=f"{phase}_clarification_surface",
        title="Meaningful clarification behavior",
        passed=passed,
        severity="blocker" if not passed else "info",
        details="Clarification should still appear when useful and remain mode-specific and human-readable.",
        evidence=snapshots,
    )


def _record_math_tool_check(controller: AppController, checks: list[ValidationCheck], *, phase: str) -> None:
    response = controller.ask(prompt="solve 2x + 5 = 13", session_id=f"{phase}-math-smoke")
    execution = response.get("tool_execution") or {}
    passed = (
        response.get("mode") == "tool"
        and execution.get("tool_id") == "math"
        and str(execution.get("capability")) == "solve_equation"
    )
    _append_check(
        checks,
        phase=phase,
        check_id=f"{phase}_math_smoke",
        title="Math tool smoke",
        passed=passed,
        severity="blocker" if not passed else "info",
        details="Packaged/runtime smoke should still reach the math tool correctly.",
        evidence={"response": _snapshot_response(response)},
    )


def _record_failure_case_check(controller: AppController, checks: list[ValidationCheck], *, phase: str) -> None:
    response = controller.ask(prompt="Generate me 5 content ideas on topic black holes", session_id=f"{phase}-failure-smoke")
    summary = _summary_text(response).lower()
    passed = "unavailable" in summary or "configured" in summary
    _append_check(
        checks,
        phase=phase,
        check_id=f"{phase}_failure_smoke",
        title="Failure honesty smoke",
        passed=passed,
        severity="blocker" if not passed else "info",
        details="Failure output should stay honest and mode-consistent.",
        evidence={"response": _snapshot_response(response)},
    )


def _record_content_status_check(controller: AppController, checks: list[ValidationCheck], *, phase: str) -> None:
    response = controller.ask(
        prompt="Generate me 5 content ideas on topic black holes",
        session_id=f"{phase}-content-status",
    )
    summary = _summary_text(response).lower()
    execution = response.get("tool_execution") or {}
    passed = (
        response.get("mode") == "tool"
        and execution.get("tool_id") == "content"
        and str(execution.get("capability")) == "generate_ideas"
        and ("unavailable" in summary or "configured" in summary or "generated" in summary)
    )
    _append_check(
        checks,
        phase=phase,
        check_id=f"{phase}_content_status",
        title="Astra/content runtime status",
        passed=passed,
        severity="blocker" if not passed else "info",
        details="Content should be either fully operational or clearly runtime/provider gated.",
        evidence={"response": _snapshot_response(response)},
    )


def _record_anh_runtime_check(
    controller: AppController,
    checks: list[ValidationCheck],
    *,
    phase: str,
    anh_probe_path: Path | None,
) -> None:
    if anh_probe_path is None or not anh_probe_path.exists():
        _append_check(
            checks,
            phase=phase,
            check_id=f"{phase}_anh_runtime",
            title="ANH runtime status",
            passed=True,
            severity="info",
            details="ANH live MAST probe deferred to final release-candidate gate.",
            evidence={
                "anh_probe_path": str(anh_probe_path) if anh_probe_path else None,
                "status": "deferred_final_gate",
            },
        )
        return
    response = controller.ask(prompt=f"run anh {anh_probe_path}", session_id=f"{phase}-anh-runtime")
    execution = response.get("tool_execution") or {}
    tool_result = response.get("tool_result")
    status = getattr(tool_result, "status", None)
    passed = (
        response.get("mode") == "tool"
        and execution.get("tool_id") == "anh"
        and str(execution.get("capability")) == "spectral_dip_scan"
        and status in {"ok", "partial"}
    )
    _append_check(
        checks,
        phase=phase,
        check_id=f"{phase}_anh_runtime",
        title="ANH real runtime execution",
        passed=passed,
        severity="blocker" if not passed else "info",
        details=f"ANH tool_result.status={status}",
        evidence={"anh_probe_path": str(anh_probe_path), "response": _snapshot_response(response)},
    )


def _record_mode_invariance(
    controller: AppController,
    checks: list[ValidationCheck],
    *,
    phase: str,
    check_id: str,
    prompt: str,
    anchors: tuple[str, ...],
    expect_mode: str | None = None,
    require_distinct_delivery: bool = True,
) -> None:
    results: dict[str, dict[str, Any]] = {}
    passed = True
    for style in ("collab", "default", "direct"):
        session_id = f"{phase}-{check_id}-{style}"
        controller.set_session_profile(session_id, interaction_style=style)
        response = controller.ask(prompt=prompt, session_id=session_id)
        results[style] = _snapshot_response(response)
        summary = _summary_text(response).lower()
        if expect_mode and response.get("mode") != expect_mode:
            passed = False
        if not all(anchor.lower() in summary or anchor.lower() in json.dumps(results[style]).lower() for anchor in anchors):
            passed = False
    rendered_variants = {
        " || ".join(
            str(results[style].get(key) or "").strip()
            for key in ("summary", "reply", "user_facing_answer", "clarification_question", "follow_up_offer")
        ).strip()
        for style in ("collab", "default", "direct")
    }
    if require_distinct_delivery and len(rendered_variants) < 2:
        passed = False
    _append_check(
        checks,
        phase=phase,
        check_id=check_id,
        title=f"Mode identity: {prompt}",
        passed=passed,
        severity="blocker" if not passed else "info",
        details="Modes should differ in delivery but keep the same core facts or outcome.",
        evidence=results,
    )


def _finalize_report(
    *,
    label: str,
    execution_mode: str,
    readiness: dict[str, Any],
    checks: list[ValidationCheck],
    audit_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    blocker_checks = [item for item in checks if not item.passed and item.severity == "blocker"]
    refinement_checks = [item for item in checks if not item.passed and item.severity == "refinement"]
    stable_checks = [item for item in checks if item.passed]
    return {
        "label": label,
        "execution_mode": execution_mode,
        "audit_context": dict(audit_context or {}),
        "readiness": readiness,
        "checks": [item.to_dict() for item in checks],
        "stable_ready": [f"{item.title}: {item.details}" for item in stable_checks],
        "refinement_qa": [f"{item.title}: {item.details}" for item in refinement_checks],
        "blockers": [f"{item.title}: {item.details}" for item in blocker_checks],
        "summary": {
            "passed_count": len(stable_checks),
            "refinement_count": len(refinement_checks),
            "blocker_count": len(blocker_checks),
            "coherent_system": len(blocker_checks) == 0,
        },
    }


def _append_check(
    checks: list[ValidationCheck],
    *,
    phase: str,
    check_id: str,
    title: str,
    passed: bool,
    severity: str,
    details: str,
    evidence: dict[str, Any],
) -> None:
    checks.append(
        ValidationCheck(
            phase=phase,
            check_id=check_id,
            title=title,
            passed=passed,
            severity=severity if not passed else "info",
            details=details,
            evidence=evidence,
        )
    )


def _build_fixture_inputs(data_root: Path) -> dict[str, Path]:
    fixture_root = data_root / "validation_inputs"
    fixture_root.mkdir(parents=True, exist_ok=True)

    sample_csv = fixture_root / "sample_data.csv"
    sample_csv.write_text(
        "time,value,temperature\n0,10,21\n1,12,22\n2,15,24\n3,14,23\n4,18,27\n",
        encoding="utf-8",
    )

    graph_json = fixture_root / "graph.json"
    graph_json.write_text(
        json.dumps(
            {
                "nodes": [
                    {"id": "entropy", "label": "Entropy"},
                    {"id": "black_holes", "label": "Black Holes"},
                ],
                "edges": [
                    {"source": "entropy", "target": "black_holes", "label": "relates_to"},
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    paper_one = fixture_root / "paper_one.txt"
    paper_one.write_text(
        "Abstract: We observed entropy changes in accretion disks. "
        "Methods: We used a telescope survey and spectral fitting. "
        "Results: The disk entropy profile rose with heating.",
        encoding="utf-8",
    )

    paper_catalog = fixture_root / "paper_catalog.json"
    paper_catalog.write_text(
        json.dumps(
            [
                {
                    "title": "Entropy and Black Holes",
                    "source": "local_catalog",
                    "abstract": "A study of black hole entropy and thermodynamics.",
                },
                {
                    "title": "Neutron Star Cooling",
                    "source": "local_catalog",
                    "abstract": "A study of neutron star evolution and cooling.",
                },
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "sample_csv": sample_csv,
        "graph_json": graph_json,
        "paper_one": paper_one,
        "paper_catalog": paper_catalog,
    }


def _copy_optional_tree(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


@contextmanager
def _temporary_audit_data_root(data_root: Path):
    previous = os.environ.get(AUDIT_DATA_ROOT_OVERRIDE_ENV)
    os.environ[AUDIT_DATA_ROOT_OVERRIDE_ENV] = str(Path(data_root).resolve())
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(AUDIT_DATA_ROOT_OVERRIDE_ENV, None)
        else:
            os.environ[AUDIT_DATA_ROOT_OVERRIDE_ENV] = previous


def _probe_reasoning_spine(controller: AppController) -> bool:
    session_id = "readiness-spine"
    controller.ask(prompt="what is entropy?", session_id=session_id)
    active = (controller.current_session_thread(session_id) or {}).get("active_thread") or {}
    reasoning_state = active.get("reasoning_state") or {}
    return bool(reasoning_state.get("current_path") and reasoning_state.get("selected_mode"))


def _probe_state_aware_followup(controller: AppController) -> bool:
    session_id = "readiness-followup"
    controller.ask(prompt="create a migration plan for lumen", session_id=session_id)
    response = controller.ask(prompt="what about that", session_id=session_id)
    return (
        response.get("mode") == "planning"
        and response.get("resolution_strategy") == "reference_follow_up"
        and "migration plan" in str(response.get("resolved_prompt") or "").lower()
    )


def _probe_mode_integration(controller: AppController) -> bool:
    results = []
    for style in ("collab", "direct"):
        session_id = f"readiness-mode-{style}"
        controller.set_session_profile(session_id, interaction_style=style)
        results.append(_summary_text(controller.ask(prompt="what is entropy?", session_id=session_id)))
    return len(set(results)) == 2 and all("entropy" in item.lower() for item in results)


def _probe_tool_reasoning_integration(controller: AppController) -> bool:
    response = controller.ask(prompt="solve 2x + 5 = 13", session_id="readiness-tool")
    execution = response.get("tool_execution") or {}
    return (
        response.get("mode") == "tool"
        and execution.get("tool_id") == "math"
        and bool(_summary_text(response))
    )


def _probe_nlg_modes(controller: AppController) -> bool:
    collab_id = "readiness-nlg-collab"
    direct_id = "readiness-nlg-direct"
    controller.set_session_profile(collab_id, interaction_style="collab")
    controller.set_session_profile(direct_id, interaction_style="direct")
    collab = _summary_text(controller.ask(prompt="hello", session_id=collab_id))
    direct = _summary_text(controller.ask(prompt="hello", session_id=direct_id))
    return bool(collab) and bool(direct) and collab != direct


def _content_runtime_status(controller: AppController) -> dict[str, Any]:
    provider = controller.model_provider
    if provider is None:
        return {
            "status": "runtime_provider_gated",
            "label": "runtime/provider gated",
            "details": "No model provider is configured.",
        }
    if getattr(provider, "provider_id", "") == "local":
        return {
            "status": "runtime_provider_gated",
            "label": "runtime/provider gated",
            "details": "Configured provider is local-only.",
        }
    ready, reason = controller.inference_service._provider_is_ready()
    probe = controller.ask(
        prompt="Generate me 5 content ideas on topic black holes",
        session_id="readiness-content",
    )
    summary = _summary_text(probe).lower()
    execution = probe.get("tool_execution") or {}
    if (
        probe.get("mode") == "tool"
        and execution.get("tool_id") == "content"
        and ("unavailable" in summary or "configured" in summary or "missing" in summary)
    ):
        return {
            "status": "runtime_provider_gated",
            "label": "runtime/provider gated",
            "details": _summary_text(probe) or reason,
        }
    if (
        probe.get("mode") == "tool"
        and execution.get("tool_id") == "content"
        and ("generated" in summary or "formatted" in summary)
    ):
        return {
            "status": "fully_live",
            "label": "fully live",
            "details": _summary_text(probe),
        }
    return {
        "status": "fully_live" if ready else "runtime_provider_gated",
        "label": "fully live" if ready else "runtime/provider gated",
        "details": reason,
    }


def _anh_runtime_status(controller: AppController, *, anh_probe_path: Path | None) -> dict[str, Any]:
    if anh_probe_path is None or not anh_probe_path.exists():
        return {
            "status": "deferred_final_gate",
            "label": "deferred to final release gate",
            "details": "No ANH probe path was supplied; live MAST validation is intentionally deferred.",
        }
    response = controller.ask(prompt=f"run anh {anh_probe_path}", session_id="readiness-anh")
    tool_result = response.get("tool_result")
    status = getattr(tool_result, "status", None)
    if status in {"ok", "partial"}:
        return {
            "status": "real_execution_verified",
            "label": "real execution verified",
            "details": _summary_text(response),
        }
    return {
        "status": "runtime_dependency_gated",
        "label": "runtime/dependency gated",
        "details": _summary_text(response) or str(getattr(tool_result, "error", "") or "ANH probe failed."),
    }


def _snapshot_response(response: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": response.get("mode"),
        "kind": response.get("kind"),
        "summary": _summary_text(response),
        "reply": response.get("reply"),
        "user_facing_answer": response.get("user_facing_answer"),
        "follow_up_offer": response.get("follow_up_offer"),
        "resolved_prompt": response.get("resolved_prompt"),
        "resolution_strategy": response.get("resolution_strategy"),
        "tool_execution": response.get("tool_execution"),
        "tool_execution_skipped": response.get("tool_execution_skipped"),
        "clarification_question": response.get("clarification_question"),
        "options": response.get("options"),
    }


def _summary_text(response: dict[str, Any]) -> str:
    for key in ("summary", "reply", "answer", "user_facing_answer", "result_summary"):
        value = response.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _contains_scaffold_residue(text: str) -> bool:
    lowered = " ".join(str(text or "").lower().split())
    return any(token in lowered for token in SCAFFOLD_RESIDUE)


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")
