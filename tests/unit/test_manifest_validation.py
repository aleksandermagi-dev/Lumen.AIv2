import pytest

from lumen.tools.registry_types import BundleManifest, CapabilityManifest


def test_bundle_manifest_validation_accepts_unique_capabilities() -> None:
    manifest = BundleManifest(
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

    manifest.validate()


def test_bundle_manifest_validation_rejects_duplicate_capability_ids() -> None:
    manifest = BundleManifest(
        id="anh",
        name="Astronomical Node Heuristics",
        version="0.1.0",
        entrypoint="bundle.py",
        capabilities=[
            CapabilityManifest(id="spectral_dip_scan", adapter="a"),
            CapabilityManifest(id="spectral_dip_scan", adapter="b"),
        ],
    )

    with pytest.raises(ValueError, match="Duplicate capability id"):
        manifest.validate()


def test_bundle_manifest_validation_rejects_duplicate_app_capability_keys() -> None:
    manifest = BundleManifest(
        id="anh",
        name="Astronomical Node Heuristics",
        version="0.1.0",
        entrypoint="bundle.py",
        capabilities=[
            CapabilityManifest(
                id="spectral_dip_scan",
                adapter="a",
                app_capability_key="astronomy.anh_spectral_scan",
            ),
            CapabilityManifest(
                id="bulk_flow.alt",
                adapter="b",
                app_capability_key="astronomy.anh_spectral_scan",
            ),
        ],
    )

    with pytest.raises(ValueError, match="Duplicate app capability key"):
        manifest.validate()


def test_bundle_manifest_validation_rejects_duplicate_command_aliases() -> None:
    manifest = BundleManifest(
        id="anh",
        name="Astronomical Node Heuristics",
        version="0.1.0",
        entrypoint="bundle.py",
        capabilities=[
            CapabilityManifest(
                id="spectral_dip_scan",
                adapter="a",
                command_aliases=["run anh"],
            ),
            CapabilityManifest(
                id="bulk_flow.alt",
                adapter="b",
                command_aliases=["run anh"],
            ),
        ],
    )

    with pytest.raises(ValueError, match="Duplicate command alias"):
        manifest.validate()

