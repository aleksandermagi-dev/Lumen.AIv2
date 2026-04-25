import pytest

from lumen.routing.capability_manager import CapabilityManager
from lumen.routing.intent_router import IntentRouter
from lumen.routing.intents import AppIntent, CapabilityRequestIntent
from lumen.tools.registry_types import BundleManifest, CapabilityManifest


def test_capability_manager_returns_ga_capability() -> None:
    manager = CapabilityManager(
        manifests={
            "anh": BundleManifest(
                id="anh",
                name="Astronomical Node Heuristics",
                version="0.1.0",
                entrypoint="bundle.py",
                capabilities=[
                    CapabilityManifest(
                        id="spectral_dip_scan",
                        adapter="anh_spectral_scan_adapter",
                        app_capability_key="astronomy.anh_spectral_scan",
                        app_description="Great Attractor local bulk-flow analysis via the ANH bundle.",
                        command_aliases=["run anh"],
                    )
                ],
            )
        }
    )

    capability = manager.get("astronomy.anh_spectral_scan")

    assert capability.tool_id == "anh"
    assert capability.tool_capability == "spectral_dip_scan"


def test_capability_manager_raises_for_unknown_key() -> None:
    manager = CapabilityManager(manifests={})

    with pytest.raises(KeyError):
        manager.get("missing.capability")


def test_intent_router_routes_ga_intent_to_tool_execution() -> None:
    manager = CapabilityManager(
        manifests={
            "anh": BundleManifest(
                id="anh",
                name="Astronomical Node Heuristics",
                version="0.1.0",
                entrypoint="bundle.py",
                capabilities=[
                    CapabilityManifest(
                        id="spectral_dip_scan",
                        adapter="anh_spectral_scan_adapter",
                        app_capability_key="astronomy.anh_spectral_scan",
                        command_aliases=["run anh"],
                    )
                ],
            )
        }
    )
    router = IntentRouter(capability_manager=manager)
    intent = CapabilityRequestIntent(
        intent_type="capability_request",
        action="analyze",
        target="ga",
        capability_key="astronomy.anh_spectral_scan",
    )

    routed = router.route(intent)

    assert routed.tool_id == "anh"
    assert routed.capability == "spectral_dip_scan"


def test_intent_router_raises_for_unmapped_generic_intent() -> None:
    manager = CapabilityManager(manifests={})
    router = IntentRouter(capability_manager=manager)
    intent = AppIntent(intent_type="generic", action="plan", target="research")

    with pytest.raises(ValueError):
        router.route(intent)


def test_capability_manager_can_find_capability_by_command_alias() -> None:
    manager = CapabilityManager(
        manifests={
            "anh": BundleManifest(
                id="anh",
                name="Astronomical Node Heuristics",
                version="0.1.0",
                entrypoint="bundle.py",
                capabilities=[
                    CapabilityManifest(
                        id="spectral_dip_scan",
                        adapter="anh_spectral_scan_adapter",
                        app_capability_key="astronomy.anh_spectral_scan",
                        command_aliases=["run anh"],
                    )
                ],
            )
        }
    )

    capability = manager.find_by_command("run", "anh")

    assert capability is not None
    assert capability.capability_key == "astronomy.anh_spectral_scan"

