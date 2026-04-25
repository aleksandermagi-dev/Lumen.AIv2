from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path

from lumen.app.settings import AppSettings
from lumen.memory.interaction_log_manager import InteractionLogManager
from lumen.memory.session_state_manager import SessionStateManager
from lumen.services.interaction_history_service import InteractionHistoryService
from lumen.tools.registry_types import Artifact, BundleManifest, ToolRequest, ToolResult


class SessionConfidenceAdapter:
    """Build a local confidence report from persisted interaction and session state."""

    def __init__(self, *, manifest: BundleManifest, repo_root: Path):
        self.manifest = manifest
        self.repo_root = repo_root
        self.settings = AppSettings.from_repo_root(repo_root)
        self.interaction_history_service = InteractionHistoryService(
            interaction_log_manager=InteractionLogManager(settings=self.settings),
            settings=self.settings,
        )
        self.session_state_manager = SessionStateManager(settings=self.settings)

    def execute(self, request: ToolRequest) -> ToolResult:
        run_dir = self._build_run_dir(request)
        outputs_dir = run_dir / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)

        summary = self.interaction_history_service.summarize_interactions(
            session_id=request.session_id
        )
        patterns = self.interaction_history_service.summarize_patterns(
            session_id=request.session_id
        )
        active_thread = self.session_state_manager.get_active_thread(request.session_id)
        report = {
            "session_id": request.session_id,
            "summary": summary,
            "patterns": patterns,
            "active_thread": active_thread,
        }

        artifact_path = outputs_dir / "session_confidence_report.json"
        artifact_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        posture_counts = summary.get("posture_counts") or {}
        latest_posture = summary.get("latest_posture") or "unknown"
        summary_text = (
            f"Session confidence report generated for '{request.session_id}' "
            f"with latest posture '{latest_posture}'"
        )

        return ToolResult(
            status="ok",
            tool_id=request.tool_id,
            capability=request.capability,
            summary=summary_text,
            structured_data={
                "session_id": request.session_id,
                "posture_counts": posture_counts,
                "latest_posture": latest_posture,
                "recent_posture_mix": summary.get("recent_posture_mix"),
                "posture_trend": summary.get("posture_trend"),
                "posture_drift": summary.get("posture_drift"),
                "active_thread": active_thread,
            },
            artifacts=[
                Artifact(
                    name="session_confidence_report.json",
                    path=artifact_path,
                    media_type="application/json",
                    description="Structured session confidence and posture report",
                )
            ],
            logs=[
                f"Collected interaction summary for session '{request.session_id}'.",
                f"Collected interaction pattern report for session '{request.session_id}'.",
            ],
            provenance={
                "repo_root": str(self.repo_root),
                "session_id": request.session_id,
            },
            run_dir=run_dir,
        )

    def _build_run_dir(self, request: ToolRequest) -> Path:
        root = request.run_root or (self.repo_root / "data" / "tool_runs")
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        run_dir = root / request.session_id / request.tool_id / request.capability / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
