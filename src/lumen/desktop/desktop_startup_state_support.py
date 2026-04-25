from __future__ import annotations

from lumen.desktop.shell_transition_support import DesktopCapabilityState


def capability_state_from_startup_health(result: dict[str, object]) -> DesktopCapabilityState:
    missing = [str(item) for item in result.get("missing", []) if str(item).strip()]
    missing_resources = [str(item) for item in result.get("missing_resources", []) if str(item).strip()]
    capabilities = result.get("capabilities") if isinstance(result.get("capabilities"), dict) else {}
    surface_runtime_ready = (
        result.get("surface_runtime_ready")
        if isinstance(result.get("surface_runtime_ready"), dict)
        else {}
    )
    return DesktopCapabilityState.from_runtime(
        missing_bundles=missing,
        missing_resources=missing_resources,
        capabilities=capabilities,
        surface_runtime_ready=surface_runtime_ready,
    )
