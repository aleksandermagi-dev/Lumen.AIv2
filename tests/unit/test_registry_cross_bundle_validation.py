import pytest

from lumen.routing.tool_registry import ToolRegistry
from lumen.tools.registry_types import BundleManifest, CapabilityManifest


def test_registry_rejects_duplicate_app_capability_keys_across_bundles() -> None:
    manifests = {
        "anh": BundleManifest(
            id="anh",
            name="ANH",
            version="0.1.0",
            entrypoint="bundle.py",
            capabilities=[
                CapabilityManifest(
                    id="spectral_dip_scan",
                    adapter="anh_spectral_scan_adapter",
                    app_capability_key="astronomy.anh_spectral_scan",
                )
            ],
        ),
        "other": BundleManifest(
            id="other",
            name="Other",
            version="0.1.0",
            entrypoint="bundle.py",
            capabilities=[
                CapabilityManifest(
                    id="bulk_flow.other",
                    adapter="other_adapter",
                    app_capability_key="astronomy.anh_spectral_scan",
                )
            ],
        ),
    }

    with pytest.raises(ValueError, match="Duplicate app capability key"):
        ToolRegistry._validate_cross_bundle_manifests(manifests)


def test_registry_rejects_duplicate_command_aliases_across_bundles() -> None:
    manifests = {
        "anh": BundleManifest(
            id="anh",
            name="ANH",
            version="0.1.0",
            entrypoint="bundle.py",
            capabilities=[
                CapabilityManifest(
                    id="spectral_dip_scan",
                    adapter="anh_spectral_scan_adapter",
                    command_aliases=["run anh"],
                )
            ],
        ),
        "other": BundleManifest(
            id="other",
            name="Other",
            version="0.1.0",
            entrypoint="bundle.py",
                capabilities=[
                    CapabilityManifest(
                        id="bulk_flow.other",
                        adapter="other_adapter",
                        command_aliases=["run anh"],
                    )
                ],
            ),
        }

    with pytest.raises(ValueError, match="Duplicate command alias"):
        ToolRegistry._validate_cross_bundle_manifests(manifests)


def test_registry_accepts_distinct_cross_bundle_capabilities() -> None:
    manifests = {
        "anh": BundleManifest(
            id="anh",
            name="ANH",
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
        ),
        "other": BundleManifest(
            id="other",
            name="Other",
            version="0.1.0",
            entrypoint="bundle.py",
            capabilities=[
                CapabilityManifest(
                    id="spectral.scan",
                    adapter="spectral_adapter",
                    app_capability_key="astronomy.spectral_scan",
                    command_aliases=["analyze spectrum"],
                )
            ],
        ),
    }

    ToolRegistry._validate_cross_bundle_manifests(manifests)

