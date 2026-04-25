from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path

from lumen.app.settings import AppSettings
from lumen.memory.archive_manager import ArchiveManager
from lumen.memory.interaction_log_manager import InteractionLogManager
from lumen.memory.session_state_manager import SessionStateManager
from lumen.tools.registry_types import Artifact, BundleManifest, ToolRequest, ToolResult


class SessionTimelineAdapter:
    """Build a local cross-surface session timeline."""

    def __init__(self, *, manifest: BundleManifest, repo_root: Path):
        self.manifest = manifest
        self.repo_root = repo_root
        self.settings = AppSettings.from_repo_root(repo_root)
        self.archive_manager = ArchiveManager(settings=self.settings)
        self.interaction_log_manager = InteractionLogManager(settings=self.settings)
        self.session_state_manager = SessionStateManager(settings=self.settings)

    def execute(self, request: ToolRequest) -> ToolResult:
        run_dir = self._build_run_dir(request)
        outputs_dir = run_dir / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)

        archive_records = self.archive_manager.list_records(session_id=request.session_id)
        interaction_records = self.interaction_log_manager.list_records(session_id=request.session_id)
        active_thread = self.session_state_manager.get_active_thread(request.session_id)
        timeline = self._build_timeline(archive_records, interaction_records)

        report = {
            "session_id": request.session_id,
            "timeline_count": len(timeline),
            "timeline": timeline,
            "archive_record_count": len(archive_records),
            "interaction_record_count": len(interaction_records),
            "active_thread": active_thread,
        }

        artifact_path = outputs_dir / "session_timeline_report.json"
        artifact_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        return ToolResult(
            status="ok",
            tool_id=request.tool_id,
            capability=request.capability,
            summary=(
                f"Session timeline generated for '{request.session_id}' "
                f"with {len(timeline)} combined events"
            ),
            structured_data={
                "session_id": request.session_id,
                "timeline_count": len(timeline),
                "archive_record_count": len(archive_records),
                "interaction_record_count": len(interaction_records),
                "timeline": timeline[:10],
                "active_thread": active_thread,
            },
            artifacts=[
                Artifact(
                    name="session_timeline_report.json",
                    path=artifact_path,
                    media_type="application/json",
                    description="Combined local session timeline report",
                )
            ],
            logs=[
                f"Collected {len(archive_records)} archive records for session '{request.session_id}'.",
                f"Collected {len(interaction_records)} interaction records for session '{request.session_id}'.",
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

    @staticmethod
    def _build_timeline(
        archive_records: list[dict[str, object]],
        interaction_records: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        events: list[dict[str, object]] = []

        for record in archive_records:
            events.append(
                {
                    "event_type": "tool_run",
                    "created_at": record.get("created_at"),
                    "tool_id": record.get("tool_id"),
                    "capability": record.get("capability"),
                    "status": record.get("status"),
                    "summary": record.get("summary"),
                }
            )

        for record in interaction_records:
            prompt_view = record.get("prompt_view") or {}
            events.append(
                {
                    "event_type": "interaction",
                    "created_at": record.get("created_at"),
                    "mode": record.get("mode"),
                    "kind": record.get("kind"),
                    "summary": record.get("summary"),
                    "confidence_posture": record.get("confidence_posture"),
                    "prompt": prompt_view.get("canonical_prompt") or record.get("prompt"),
                    "original_prompt": prompt_view.get("original_prompt"),
                }
            )

        events.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        return events
