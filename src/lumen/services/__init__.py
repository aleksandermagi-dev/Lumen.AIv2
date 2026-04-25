"""Focused application services for Lumen orchestration."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "ArchiveService",
    "BundleService",
    "DiagnosticsService",
    "InteractionService",
    "ToolExecutionService",
    "WorkspaceService",
]


def __getattr__(name: str):
    module_map = {
        "ArchiveService": "lumen.services.archive_service",
        "BundleService": "lumen.services.bundle_service",
        "DiagnosticsService": "lumen.services.diagnostics_service",
        "InteractionService": "lumen.services.interaction_service",
        "ToolExecutionService": "lumen.services.tool_execution_service",
        "WorkspaceService": "lumen.services.workspace_service",
    }
    module_name = module_map.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = import_module(module_name)
    return getattr(module, name)
