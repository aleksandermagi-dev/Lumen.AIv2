from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys


@dataclass(slots=True, frozen=True)
class DesktopRuntimePaths:
    runtime_root: Path
    data_root: Path
    execution_mode: str


def is_frozen_app() -> bool:
    return bool(getattr(sys, "frozen", False))


def detect_runtime_root(*, repo_root: Path | None = None) -> Path:
    if repo_root is not None:
        return repo_root.resolve()
    if is_frozen_app():
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(str(meipass)).resolve()
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[3]


def default_data_root(*, runtime_root: Path) -> Path:
    if is_frozen_app():
        local_appdata = os.environ.get("LOCALAPPDATA")
        if local_appdata:
            return (Path(local_appdata) / "Lumen" / "data").resolve()
        return (Path.home() / ".lumen" / "data").resolve()
    return (runtime_root / "data").resolve()


def resolve_desktop_runtime_paths(
    *,
    repo_root: Path | None = None,
    data_root: Path | None = None,
) -> DesktopRuntimePaths:
    runtime_root = detect_runtime_root(repo_root=repo_root)
    resolved_data_root = data_root.resolve() if data_root is not None else default_data_root(runtime_root=runtime_root)
    return DesktopRuntimePaths(
        runtime_root=runtime_root,
        data_root=resolved_data_root,
        execution_mode="frozen" if is_frozen_app() else "source",
    )
