from __future__ import annotations

from pathlib import Path

from lumen.app.settings import AppSettings
from lumen.db.persistence_manager import PersistenceManager


class WorkspaceService:
    """Handles local workspace bootstrap and scaffolding tasks."""

    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.repo_root = settings.repo_root
        self.persistence_manager = PersistenceManager(settings)

    def initialize_workspace(self) -> dict[str, object]:
        created_paths: list[str] = []
        existing_paths: list[str] = []

        for path in [
            self.settings.data_root,
            self.settings.persistence_db_path.parent,
            self.settings.archive_root,
            self.settings.interactions_root,
            self.settings.personal_memory_root,
            self.settings.research_notes_root,
            self.settings.research_artifacts_root,
            self.settings.labeled_datasets_root,
            self.settings.graph_memory_db_path.parent,
            self.settings.sessions_root,
            self.settings.tool_runs_root,
            self.settings.examples_root,
        ]:
            if path.exists():
                existing_paths.append(str(path))
            else:
                path.mkdir(parents=True, exist_ok=True)
                created_paths.append(str(path))

        config_path = self.repo_root / "lumen.toml"
        example_path = self.repo_root / "lumen.toml.example"
        if config_path.exists():
            existing_paths.append(str(config_path))
        elif example_path.exists():
            config_path.write_text(example_path.read_text(encoding="utf-8"), encoding="utf-8")
            created_paths.append(str(config_path))

        self.persistence_manager.bootstrap(run_imports=True)
        if self.settings.persistence_db_path.exists():
            existing_paths.append(str(self.settings.persistence_db_path))

        return {
            "status": "ok",
            "repo_root": str(self.repo_root),
            "config_path": str(config_path) if config_path.exists() else None,
            "created_paths": created_paths,
            "existing_paths": existing_paths,
        }
