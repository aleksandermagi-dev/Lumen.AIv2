from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json


@dataclass(frozen=True, slots=True)
class ResearchArtifactRecord:
    role: str
    path: Path
    media_type: str | None = None
    description: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        payload = {
            "role": self.role,
            "path": str(self.path),
            "filename": self.path.name,
            "exists": self.path.exists(),
        }
        if self.media_type:
            payload["media_type"] = self.media_type
        if self.description:
            payload["description"] = self.description
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


def sha256_or_none(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_standard_summary(
    *,
    bundle_id: str,
    capability: str,
    run_id: str,
    target_label: str,
    input_files: list[str],
    provenance: dict[str, object],
    analysis_status: dict[str, object],
    batch_record: dict[str, object],
    domain_payload: dict[str, object],
    produced_artifacts: list[ResearchArtifactRecord],
) -> dict[str, object]:
    return {
        "summary_version": "1",
        "bundle_standard": "lumen_research_bundle_v1",
        "bundle_id": bundle_id,
        "capability": capability,
        "run_id": run_id,
        "target_label": target_label,
        "input_files": list(input_files),
        "provenance": provenance,
        "analysis_status": analysis_status,
        "batch_record": batch_record,
        "domain_payload": domain_payload,
        "produced_artifacts": [artifact.to_dict() for artifact in produced_artifacts],
    }


def write_standard_artifacts(output_dir: Path, summary_payload: dict[str, object]) -> dict[str, Path]:
    json_path = output_dir / "analysis_summary.json"
    json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    text_path = output_dir / "analysis_summary.txt"
    text_path.write_text(render_standard_summary_text(summary_payload), encoding="utf-8")

    produced_artifacts = list(summary_payload.get("produced_artifacts", []))
    produced_artifacts.extend(
        [
            ResearchArtifactRecord(
                role="summary_json",
                path=json_path,
                media_type="application/json",
                description="Structured machine-readable research run summary.",
            ).to_dict(),
            ResearchArtifactRecord(
                role="summary_text",
                path=text_path,
                media_type="text/plain",
                description="Human-readable research run summary.",
            ).to_dict(),
        ]
    )
    summary_payload["produced_artifacts"] = produced_artifacts

    manifest_path = output_dir / "artifact_manifest.json"
    produced_artifacts.append(
        ResearchArtifactRecord(
            role="artifact_manifest",
            path=manifest_path,
            media_type="application/json",
            description="Manifest describing all research run artifacts.",
        ).to_dict()
    )
    manifest_payload = {
        "manifest_version": "1",
        "bundle_standard": summary_payload.get("bundle_standard"),
        "bundle_id": summary_payload.get("bundle_id"),
        "capability": summary_payload.get("capability"),
        "run_id": summary_payload.get("run_id"),
        "target_label": summary_payload.get("target_label"),
        "input_files": summary_payload.get("input_files", []),
        "provenance": summary_payload.get("provenance", {}),
        "analysis_status": summary_payload.get("analysis_status", {}),
        "artifacts": list(produced_artifacts),
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")

    return {
        "summary_json": json_path,
        "summary_text": text_path,
        "artifact_manifest": manifest_path,
    }


def render_standard_summary_text(summary_payload: dict[str, object]) -> str:
    provenance = summary_payload.get("provenance") or {}
    analysis_status = summary_payload.get("analysis_status") or {}
    lines = [
        f"bundle_standard: {summary_payload.get('bundle_standard')}",
        f"bundle_id: {summary_payload.get('bundle_id')}",
        f"capability: {summary_payload.get('capability')}",
        f"run_id: {summary_payload.get('run_id')}",
        f"target_label: {summary_payload.get('target_label')}",
        f"timestamp_utc: {provenance.get('timestamp_utc')}",
        f"wrapper_version: {provenance.get('wrapper_version')}",
        f"baseline_script_path: {provenance.get('baseline_script_path')}",
        f"baseline_script_sha256: {provenance.get('baseline_script_sha256')}",
        f"validated: {analysis_status.get('validated')}",
        f"analysis_ran: {analysis_status.get('analysis_ran')}",
        f"plot_generated: {analysis_status.get('plot_generated')}",
        f"line_detected: {analysis_status.get('line_detected')}",
        f"result_quality: {analysis_status.get('result_quality')}",
        f"failure_reason: {analysis_status.get('failure_reason')}",
        "input_files:",
    ]
    for input_file in summary_payload.get("input_files", []):
        lines.append(f"- {input_file}")

    lines.append("produced_artifacts:")
    for artifact in summary_payload.get("produced_artifacts", []):
        if isinstance(artifact, dict):
            lines.append(f"- {artifact.get('role')} | {artifact.get('path')}")

    domain_payload = summary_payload.get("domain_payload") or {}
    parsed_results = domain_payload.get("parsed_results") or {}
    targeted_checks = domain_payload.get("targeted_checks") or []
    if parsed_results:
        lines.append("parsed_results:")
        for key, value in parsed_results.items():
            lines.append(f"- {key}: {value}")
    if targeted_checks:
        lines.append("targeted_checks:")
        for item in targeted_checks:
            if not isinstance(item, dict):
                continue
            result = item.get("result") or {}
            lines.append(
                "- "
                f"{item.get('title')} | expected_velocity_kms={item.get('expected_velocity_kms')} | "
                f"lambda_min={result.get('lambda_min')} | depth={result.get('depth')} | "
                f"velocity_kms={result.get('velocity_kms')}"
            )
    return "\n".join(lines)
