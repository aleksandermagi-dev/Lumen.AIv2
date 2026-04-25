from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class AppIntent:
    intent_type: str
    action: str
    target: str
    input_path: Path | None = None
    params: dict[str, Any] = field(default_factory=dict)
    session_id: str = "default"
    run_root: Path | None = None


@dataclass(slots=True)
class ToolExecutionIntent:
    intent_type: str
    tool_id: str
    capability: str
    input_path: Path | None = None
    params: dict[str, Any] = field(default_factory=dict)
    session_id: str = "default"
    run_root: Path | None = None


@dataclass(slots=True)
class AnalyzeGreatAttractorIntent(AppIntent):
    region: str = "great_attractor"


@dataclass(slots=True)
class CapabilityRequestIntent(AppIntent):
    capability_key: str = ""
