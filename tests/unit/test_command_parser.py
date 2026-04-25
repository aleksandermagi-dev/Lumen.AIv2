from pathlib import Path

from lumen.routing.capability_manager import CapabilityManager
from lumen.app.command_parser import CommandParser
from lumen.routing.intents import AppIntent, CapabilityRequestIntent
from lumen.tools.registry_types import BundleManifest, CapabilityManifest


def make_capability_manager() -> CapabilityManager:
    return CapabilityManager(
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
                        command_aliases=[
                            "run anh",
                            "scan si iv dips",
                            "find spectral dips",
                        ],
                    )
                ],
            )
        }
    )


def test_parse_anh_command_returns_typed_intent() -> None:
    parser = CommandParser(capability_manager=make_capability_manager())

    intent = parser.parse(
        action="run",
        target="anh",
        input_path=Path("data/examples/cf4_ga_cone_template.csv"),
        params={"h0": 70},
        session_id="s1",
    )

    assert isinstance(intent, CapabilityRequestIntent)
    assert intent.intent_type == "capability_request"
    assert intent.capability_key == "astronomy.anh_spectral_scan"
    assert intent.target == "anh"
    assert intent.params["h0"] == 70
    assert intent.session_id == "s1"


def test_parse_unknown_command_returns_generic_intent() -> None:
    parser = CommandParser(capability_manager=make_capability_manager())

    intent = parser.parse(action="plan", target="research")

    assert isinstance(intent, AppIntent)
    assert intent.intent_type == "generic"
    assert intent.action == "plan"
    assert intent.target == "research"

