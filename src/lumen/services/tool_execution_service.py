from __future__ import annotations

from pathlib import Path
from typing import Any

from lumen.app.models import SessionState
from lumen.app.settings import AppSettings
from lumen.memory.archive_manager import ArchiveManager
from lumen.routing.tool_registry import ToolRegistry
from lumen.services.safety_service import SafetyService
from lumen.tools.registry_types import ToolRequest, ToolResult


class ToolExecutionService:
    """Handles explicit tool execution authority and archive writeback.

    Tool execution is an active capability invocation, not passive retrieval.
    Routing may choose whether to invoke a tool, but retrieval/memory layers must
    never implicitly execute tools through this service.
    """

    def __init__(
        self,
        *,
        settings: AppSettings,
        registry: ToolRegistry,
        archive_manager: ArchiveManager,
        safety_service: SafetyService,
    ):
        self.settings = settings
        self.registry = registry
        self.archive_manager = archive_manager
        self.safety_service = safety_service

    def run_tool(
        self,
        *,
        tool_id: str,
        capability: str,
        input_path: Path | None = None,
        params: dict[str, Any] | None = None,
        session_id: str = "default",
        run_root: Path | None = None,
    ) -> ToolResult:
        self.safety_service.validate_tool_request(
            tool_id=tool_id,
            capability=capability,
            input_path=input_path,
            params=params or {},
            session_id=session_id,
            run_root=run_root,
        )
        session = SessionState(session_id=session_id, repo_root=self.settings.repo_root)
        request = ToolRequest(
            tool_id=tool_id,
            capability=capability,
            input_path=input_path.resolve() if input_path else None,
            params=params or {},
            session_id=session.session_id,
            run_root=run_root.resolve() if run_root else self.settings.tool_runs_root,
        )
        result = self.registry.execute(request)
        record = self.archive_manager.record_tool_run(session_id=session.session_id, result=result)
        result.archive_path = record.archive_path
        return result
