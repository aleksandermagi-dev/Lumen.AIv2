from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from types import ModuleType

from lumen.tools.base import ToolBundle
from lumen.tools.registry_types import BundleManifest, ToolRequest, ToolResult


class ToolRegistry:
    """Loads manifest-backed local bundles and routes tool requests."""

    def __init__(self, repo_root: Path, bundles_dir: Path | None = None):
        self.repo_root = repo_root
        self.bundles_dir = bundles_dir or repo_root / "tool_bundles"
        self._manifests: dict[str, BundleManifest] = {}
        self._bundles: dict[str, ToolBundle] = {}

    def discover(self) -> dict[str, BundleManifest]:
        manifests: dict[str, BundleManifest] = {}
        if not self.bundles_dir.exists():
            self._manifests = manifests
            return manifests

        for manifest_path in sorted(self.bundles_dir.glob("*/manifest.json")):
            try:
                manifest = BundleManifest.from_file(manifest_path)
            except Exception as exc:
                raise ValueError(f"Invalid bundle manifest at '{manifest_path}': {exc}") from exc
            if manifest.id in manifests:
                other = manifests[manifest.id]
                raise ValueError(
                    f"Duplicate bundle id '{manifest.id}' in manifests "
                    f"'{other.manifest_path}' and '{manifest.manifest_path}'"
                )
            manifests[manifest.id] = manifest

        self._validate_cross_bundle_manifests(manifests)

        self._manifests = manifests
        return manifests

    def list_tools(self) -> dict[str, list[str]]:
        if not self._manifests:
            self.discover()
        return {
            manifest.id: [capability.id for capability in manifest.capabilities]
            for manifest in self._manifests.values()
        }

    def get_manifests(self) -> dict[str, BundleManifest]:
        if not self._manifests:
            self.discover()
        return dict(self._manifests)

    def execute(self, request: ToolRequest) -> ToolResult:
        bundle = self.get_bundle(request.tool_id)
        return bundle.execute(request)

    def get_bundle(self, tool_id: str) -> ToolBundle:
        if tool_id in self._bundles:
            return self._bundles[tool_id]

        if not self._manifests:
            self.discover()

        manifest = self._manifests.get(tool_id)
        if manifest is None:
            raise KeyError(f"Tool bundle '{tool_id}' is not registered")

        bundle = self._load_bundle(manifest)
        self._bundles[tool_id] = bundle
        return bundle

    def _load_bundle(self, manifest: BundleManifest) -> ToolBundle:
        entrypoint = manifest.bundle_root / manifest.entrypoint
        module = self._load_module(
            module_name=f"lumen_bundle_{manifest.id}",
            module_path=entrypoint,
            search_root=self.repo_root,
        )

        if not hasattr(module, "create_bundle"):
            raise AttributeError(f"Bundle entrypoint '{entrypoint}' is missing create_bundle()")

        bundle = module.create_bundle(manifest, self.repo_root)
        if not isinstance(bundle, ToolBundle):
            raise TypeError(f"Bundle '{manifest.id}' did not return a ToolBundle instance")
        return bundle

    @staticmethod
    def _load_module(module_name: str, module_path: Path, search_root: Path) -> ModuleType:
        search_root_str = str(search_root.resolve())
        if search_root_str not in sys.path:
            sys.path.insert(0, search_root_str)

        spec = spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load module from {module_path}")

        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _validate_cross_bundle_manifests(manifests: dict[str, BundleManifest]) -> None:
        app_capability_keys: dict[str, tuple[str, str]] = {}
        command_aliases: dict[str, tuple[str, str]] = {}

        for manifest in manifests.values():
            for capability in manifest.capabilities:
                if capability.app_capability_key:
                    key = capability.app_capability_key
                    if key in app_capability_keys:
                        other_bundle, other_capability = app_capability_keys[key]
                        raise ValueError(
                            f"Duplicate app capability key '{key}' across bundles: "
                            f"'{other_bundle}.{other_capability}' and '{manifest.id}.{capability.id}'"
                        )
                    app_capability_keys[key] = (manifest.id, capability.id)

                for alias in capability.command_aliases:
                    normalized = alias.strip().lower()
                    if normalized in command_aliases:
                        other_bundle, other_capability = command_aliases[normalized]
                        raise ValueError(
                            f"Duplicate command alias '{alias}' across bundles: "
                            f"'{other_bundle}.{other_capability}' and '{manifest.id}.{capability.id}'"
                        )
                    command_aliases[normalized] = (manifest.id, capability.id)
