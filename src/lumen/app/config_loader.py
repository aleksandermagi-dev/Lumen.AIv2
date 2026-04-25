from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None


class ConfigLoader:
    """Loads optional repo-local configuration for Lumen."""

    CONFIG_FILENAME = "lumen.toml"

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root.resolve()
        self.config_path = self.repo_root / self.CONFIG_FILENAME

    def exists(self) -> bool:
        return self.config_path.exists()

    def load(self) -> dict[str, Any]:
        if not self.exists():
            return {}
        if tomllib is None:
            raise RuntimeError("tomllib is unavailable; Python 3.11+ is required for lumen.toml support")
        with self.config_path.open("rb") as handle:
            data = tomllib.load(handle)
        if not isinstance(data, dict):
            raise ValueError("lumen.toml must contain a top-level table")
        return data
