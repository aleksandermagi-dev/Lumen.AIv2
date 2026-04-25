from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from lumen.tools.structured_data_tools import build_chart_svg


def _coerce_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _coerce_int(value: object, default: int, *, minimum: int = 1) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = int(default)
    return max(minimum, parsed)


def load_simulation_params(*, input_path: Path | None, params: dict[str, Any]) -> dict[str, Any]:
    if input_path is not None and input_path.exists() and input_path.suffix.lower() == ".json":
        try:
            payload = json.loads(input_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            payload = {}
        if isinstance(payload, dict):
            merged = dict(payload)
            merged.update(params)
            return merged
    return dict(params)


def simulate_system(params: dict[str, Any]) -> dict[str, Any]:
    initial_value = _coerce_float(params.get("initial_value"), 1.0)
    growth_rate = _coerce_float(params.get("growth_rate"), 0.08)
    damping_rate = _coerce_float(params.get("damping_rate"), 0.02)
    forcing = _coerce_float(params.get("forcing"), 0.0)
    steps = _coerce_int(params.get("steps"), 24, minimum=2)
    label = str(params.get("label") or "system_state")

    value = initial_value
    series: list[dict[str, float]] = [{"step": 0.0, "value": round(value, 6)}]
    for step in range(1, steps + 1):
        value = value + (growth_rate * value) - (damping_rate * value) + forcing
        series.append({"step": float(step), "value": round(value, 6)})

    points = [{"x": item["step"], "y": item["value"], "label": str(int(item["step"]))} for item in series]
    return {
        "status": "ok",
        "simulation_type": "system",
        "assumptions": [
            "Discrete first-order update rule.",
            "Growth and damping rates stay constant across the run.",
            "External forcing is applied uniformly each step.",
        ],
        "parameters": {
            "initial_value": initial_value,
            "growth_rate": growth_rate,
            "damping_rate": damping_rate,
            "forcing": forcing,
            "steps": steps,
            "label": label,
        },
        "series": series,
        "points": points,
        "final_value": round(value, 6),
        "runtime_diagnostics": {"runtime_ready": True, "input_ready": True},
    }


def simulate_population(params: dict[str, Any]) -> dict[str, Any]:
    initial_population = _coerce_float(params.get("initial_population"), 100.0)
    growth_rate = _coerce_float(params.get("growth_rate"), 0.12)
    carrying_capacity = max(_coerce_float(params.get("carrying_capacity"), 1000.0), 1.0)
    steps = _coerce_int(params.get("steps"), 24, minimum=2)

    population = max(initial_population, 0.0)
    series: list[dict[str, float]] = [{"step": 0.0, "population": round(population, 6)}]
    for step in range(1, steps + 1):
        population = population + (growth_rate * population * (1.0 - (population / carrying_capacity)))
        population = max(population, 0.0)
        series.append({"step": float(step), "population": round(population, 6)})

    points = [{"x": item["step"], "y": item["population"], "label": str(int(item["step"]))} for item in series]
    return {
        "status": "ok",
        "simulation_type": "population",
        "assumptions": [
            "Logistic growth with a fixed carrying capacity.",
            "No migration or stochastic shocks are applied.",
        ],
        "parameters": {
            "initial_population": initial_population,
            "growth_rate": growth_rate,
            "carrying_capacity": carrying_capacity,
            "steps": steps,
        },
        "series": series,
        "points": points,
        "final_population": round(population, 6),
        "runtime_diagnostics": {"runtime_ready": True, "input_ready": True},
    }


def simulate_diffusion(params: dict[str, Any]) -> dict[str, Any]:
    cells = _coerce_int(params.get("cells"), 21, minimum=5)
    steps = _coerce_int(params.get("steps"), 20, minimum=1)
    diffusion_rate = min(max(_coerce_float(params.get("diffusion_rate"), 0.12), 0.001), 0.24)
    peak_value = _coerce_float(params.get("peak_value"), 1.0)

    profile = [0.0 for _ in range(cells)]
    center = cells // 2
    profile[center] = peak_value
    history = [profile.copy()]

    for _ in range(steps):
        next_profile = profile.copy()
        for index in range(1, cells - 1):
            next_profile[index] = profile[index] + diffusion_rate * (
                profile[index - 1] - (2 * profile[index]) + profile[index + 1]
            )
        next_profile[0] = 0.0
        next_profile[-1] = 0.0
        profile = next_profile
        history.append(profile.copy())

    final_profile = [
        {"position": float(index), "value": round(value, 6)}
        for index, value in enumerate(profile)
    ]
    points = [{"x": item["position"], "y": item["value"], "label": str(int(item["position"]))} for item in final_profile]
    return {
        "status": "ok",
        "simulation_type": "diffusion",
        "assumptions": [
            "One-dimensional discrete diffusion with fixed zero boundaries.",
            "The diffusion coefficient stays constant across the run.",
        ],
        "parameters": {
            "cells": cells,
            "steps": steps,
            "diffusion_rate": diffusion_rate,
            "peak_value": peak_value,
        },
        "profiles": history,
        "final_profile": final_profile,
        "points": points,
        "center_value": round(profile[center], 6),
        "runtime_diagnostics": {"runtime_ready": True, "input_ready": True},
    }


def simulate_orbit(params: dict[str, Any]) -> dict[str, Any]:
    semi_major_axis = max(_coerce_float(params.get("semi_major_axis", params.get("radius")), 1.0), 0.01)
    eccentricity = min(max(_coerce_float(params.get("eccentricity"), 0.15), 0.0), 0.95)
    samples = _coerce_int(params.get("samples"), 72, minimum=12)

    points: list[dict[str, float | str]] = []
    for index in range(samples):
        theta = (2 * math.pi * index) / samples
        radius = (semi_major_axis * (1 - (eccentricity**2))) / (1 + eccentricity * math.cos(theta))
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        points.append({"x": round(x, 6), "y": round(y, 6), "label": str(index)})

    periapsis = semi_major_axis * (1 - eccentricity)
    apoapsis = semi_major_axis * (1 + eccentricity)
    return {
        "status": "ok",
        "simulation_type": "orbit",
        "assumptions": [
            "Two-body orbital geometry with no perturbations.",
            "The orbit is represented in a 2D plane.",
        ],
        "parameters": {
            "semi_major_axis": semi_major_axis,
            "eccentricity": eccentricity,
            "samples": samples,
        },
        "points": points,
        "periapsis": round(periapsis, 6),
        "apoapsis": round(apoapsis, 6),
        "runtime_diagnostics": {"runtime_ready": True, "input_ready": True},
    }


def simulation_svg(*, title: str, points: list[dict[str, Any]], x_label: str, y_label: str) -> str:
    normalized = [
        {
            "x": float(point.get("x", 0.0)),
            "y": float(point.get("y", 0.0)),
            "label": str(point.get("label") or ""),
        }
        for point in points
    ]
    return build_chart_svg(title=title, points=normalized, x_label=x_label, y_label=y_label)
