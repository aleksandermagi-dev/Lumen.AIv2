from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any


def build_run_dir(*, repo_root: Path, request) -> Path:
    root = request.run_root or (repo_root / "data" / "tool_runs")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = root / request.session_id / request.tool_id / request.capability / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def ensure_outputs_dir(run_dir: Path) -> Path:
    outputs_dir = run_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return outputs_dir


def write_json_artifact(outputs_dir: Path, name: str, payload: dict[str, Any]) -> Path:
    artifact_path = outputs_dir / name
    artifact_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return artifact_path


def write_text_artifact(outputs_dir: Path, name: str, content: str) -> Path:
    artifact_path = outputs_dir / name
    artifact_path.write_text(content, encoding="utf-8")
    return artifact_path


def write_svg_artifact(outputs_dir: Path, name: str, content: str) -> Path:
    artifact_path = outputs_dir / name
    artifact_path.write_text(content, encoding="utf-8")
    return artifact_path


def write_artifact_manifest(outputs_dir: Path, payload: dict[str, Any]) -> Path:
    manifest_path = outputs_dir / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path
