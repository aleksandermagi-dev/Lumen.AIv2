from __future__ import annotations

from pathlib import Path
from typing import Any

from lumen.tools.simulation_tools import simulate_orbit, simulation_svg


def load_domain_params(*, input_path: Path | None, params: dict[str, Any]) -> dict[str, Any]:
    return dict(params)


def energy_model_payload(params: dict[str, Any]) -> dict[str, Any]:
    mass = _coerce_float(params.get("mass"), 1.0)
    velocity = _coerce_float(params.get("velocity"), 0.0)
    height = _coerce_float(params.get("height"), 0.0)
    gravity = _coerce_float(params.get("gravity"), 9.81)

    kinetic_energy = 0.5 * mass * (velocity**2)
    potential_energy = mass * gravity * height
    total_energy = kinetic_energy + potential_energy

    energy_points = [
        {"x": 0.0, "y": round(kinetic_energy, 6), "label": "kinetic"},
        {"x": 1.0, "y": round(potential_energy, 6), "label": "potential"},
        {"x": 2.0, "y": round(total_energy, 6), "label": "total"},
    ]
    return {
        "status": "ok",
        "domain_type": "physics.energy_model",
        "parameters": {
            "mass": mass,
            "velocity": velocity,
            "height": height,
            "gravity": gravity,
        },
        "kinetic_energy": round(kinetic_energy, 6),
        "potential_energy": round(potential_energy, 6),
        "total_energy": round(total_energy, 6),
        "assumptions": [
            "Classical mechanics only.",
            "Potential energy is measured relative to the chosen height reference.",
            "No dissipative losses are included.",
        ],
        "interpretation": (
            "The kinetic term captures motion, while the potential term captures stored gravitational energy at the chosen height."
        ),
        "points": energy_points,
        "runtime_diagnostics": {"runtime_ready": True, "input_ready": True},
    }


def energy_model_svg(payload: dict[str, Any]) -> str:
    return simulation_svg(
        title="Physics energy model",
        points=list(payload.get("points") or []),
        x_label="energy component",
        y_label="energy",
    )


def orbit_profile_payload(params: dict[str, Any]) -> dict[str, Any]:
    orbit = simulate_orbit(params)
    periapsis = float(orbit.get("periapsis") or 0.0)
    apoapsis = float(orbit.get("apoapsis") or 0.0)
    eccentricity = float(((orbit.get("parameters") or {}).get("eccentricity")) or 0.0)
    classification = "nearly circular" if eccentricity < 0.1 else ("moderately elliptical" if eccentricity < 0.5 else "highly elliptical")
    payload = {
        "status": "ok",
        "domain_type": "astronomy.orbit_profile",
        "orbit": orbit,
        "classification": classification,
        "interpretation": (
            f"This orbit is {classification}, with periapsis {periapsis} and apoapsis {apoapsis}."
        ),
        "runtime_diagnostics": {"runtime_ready": True, "input_ready": True},
    }
    return payload


def orbit_profile_svg(payload: dict[str, Any]) -> str:
    orbit = payload.get("orbit") or {}
    points = orbit.get("points") if isinstance(orbit, dict) else []
    return simulation_svg(
        title="Astronomy orbit profile",
        points=list(points or []),
        x_label="x",
        y_label="y",
    )


def extract_energy_request(prompt: str) -> dict[str, object]:
    text = " ".join(str(prompt or "").strip().split())
    lowered = text.lower()
    for prefix in ("model energy", "energy model", "physics energy model"):
        if lowered.startswith(prefix):
            text = text[len(prefix) :].strip(" :,-")
            lowered = text.lower()
            break
    params: dict[str, object] = {}
    for label, key in (("mass", "mass"), ("velocity", "velocity"), ("height", "height"), ("gravity", "gravity")):
        value = _extract_float_after_token(lowered, text, label)
        if value is not None:
            params[key] = value
    return params


def extract_orbit_profile_request(prompt: str) -> dict[str, object]:
    text = " ".join(str(prompt or "").strip().split())
    lowered = text.lower()
    for prefix in ("analyze orbit profile", "astronomy orbit profile", "orbit profile"):
        if lowered.startswith(prefix):
            text = text[len(prefix) :].strip(" :,-")
            lowered = text.lower()
            break
    params: dict[str, object] = {}
    for label, key in (
        ("semi-major axis", "semi_major_axis"),
        ("radius", "semi_major_axis"),
        ("eccentricity", "eccentricity"),
        ("samples", "samples"),
    ):
        value = _extract_float_after_token(lowered, text, label)
        if value is not None:
            params[key] = int(value) if key == "samples" else value
    return params


def _extract_float_after_token(lowered: str, original: str, token: str) -> float | None:
    index = lowered.find(token)
    if index == -1:
        return None
    fragment = original[index + len(token) :].strip(" :,-")
    if not fragment:
        return None
    first = fragment.split()[0].strip(",;")
    return _coerce_float(first, None)


def _coerce_float(value: object, default: float | None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
