from __future__ import annotations

from abc import ABC, abstractmethod

from .registry_types import BundleManifest, ToolRequest, ToolResult


class ToolBundle(ABC):
    """Base interface for a manifest-backed tool bundle."""

    def __init__(self, manifest: BundleManifest):
        self.manifest = manifest

    @property
    def id(self) -> str:
        return self.manifest.id

    @property
    def capabilities(self) -> list[str]:
        return [item.id for item in self.manifest.capabilities]

    @abstractmethod
    def execute(self, request: ToolRequest) -> ToolResult:
        raise NotImplementedError
