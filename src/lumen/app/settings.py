from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from lumen.app.config_loader import ConfigLoader

AUDIT_DATA_ROOT_OVERRIDE_ENV = "LUMEN_DATA_ROOT_OVERRIDE"


@dataclass(slots=True)
class AppSettings:
    repo_root: Path
    data_root: Path
    persistence_db_path: Path
    knowledge_root: Path
    knowledge_db_path: Path
    archive_root: Path
    interactions_root: Path
    personal_memory_root: Path
    research_notes_root: Path
    research_artifacts_root: Path
    labeled_datasets_root: Path
    graph_memory_db_path: Path
    sessions_root: Path
    tool_runs_root: Path
    examples_root: Path
    default_output_format: str = "json"
    default_session_id: str = "default"
    deployment_mode: str = "local_only"
    inference_provider: str = "local"
    openai_api_base: str | None = None
    openai_responses_model: str | None = None
    provider_timeout_seconds: int = 30
    mobile_research_note_auto_save: bool = False
    context_match_limit: int = 3
    search_candidate_limit: int = 25
    context_prompt_max_length: int = 160
    context_summary_max_length: int = 160
    session_objective_max_length: int = 200
    session_thread_summary_max_length: int = 280
    max_interaction_record_bytes: int = 32 * 1024 * 1024
    max_session_state_bytes: int = 16 * 1024 * 1024
    tool_run_retention_per_capability: int = 25
    config_path: Path | None = None

    @classmethod
    def from_repo_root(
        cls,
        repo_root: Path,
        *,
        data_root_override: Path | None = None,
    ) -> "AppSettings":
        resolved = repo_root.resolve()
        config_loader = ConfigLoader(resolved)
        raw_config = config_loader.load() if config_loader.exists() else {}
        app_config = raw_config.get("app", {})
        env_has_openai_key = bool(os.environ.get("OPENAI_API_KEY"))
        deployment_mode = app_config.get("deployment_mode")
        inference_provider = app_config.get("inference_provider")
        openai_model = (
            app_config.get("openai_responses_model")
            or os.environ.get("OPENAI_RESPONSES_MODEL")
            or os.environ.get("OPENAI_MODEL")
        )

        if deployment_mode is None and inference_provider is None and env_has_openai_key:
            deployment_mode = "hybrid"
            inference_provider = "openai_responses"

        if data_root_override is not None:
            data_root = data_root_override.resolve()
        else:
            env_override = os.environ.get(AUDIT_DATA_ROOT_OVERRIDE_ENV)
            if env_override:
                data_root = Path(env_override).resolve()
            else:
                data_root = cls._resolve_path(
                    app_config.get("data_root"),
                    base=resolved,
                    fallback=resolved / "data",
                )
        return cls(
            repo_root=resolved,
            data_root=data_root,
            persistence_db_path=cls._resolve_path(
                app_config.get("persistence_db_path"),
                base=resolved,
                fallback=(data_root / "persistence" / "lumen.sqlite3"),
            ),
            knowledge_root=cls._resolve_path(
                app_config.get("knowledge_root"),
                base=resolved,
                fallback=data_root / "knowledge",
            ),
            knowledge_db_path=cls._resolve_path(
                app_config.get("knowledge_db_path"),
                base=resolved,
                fallback=(data_root / "knowledge" / "lumen_knowledge.sqlite3"),
            ),
            archive_root=cls._resolve_path(
                app_config.get("archive_root"),
                base=resolved,
                fallback=data_root / "archive",
            ),
            interactions_root=cls._resolve_path(
                app_config.get("interactions_root"),
                base=resolved,
                fallback=data_root / "interactions",
            ),
            personal_memory_root=cls._resolve_path(
                app_config.get("personal_memory_root"),
                base=resolved,
                fallback=data_root / "personal_memory",
            ),
            research_notes_root=cls._resolve_path(
                app_config.get("research_notes_root"),
                base=resolved,
                fallback=data_root / "research_notes",
            ),
            research_artifacts_root=cls._resolve_path(
                app_config.get("research_artifacts_root"),
                base=resolved,
                fallback=data_root / "research_artifacts",
            ),
            labeled_datasets_root=cls._resolve_path(
                app_config.get("labeled_datasets_root"),
                base=resolved,
                fallback=data_root / "labeled_datasets",
            ),
            graph_memory_db_path=cls._resolve_path(
                app_config.get("graph_memory_db_path"),
                base=resolved,
                fallback=(data_root / "graph_memory" / "lumen_memory.sqlite3"),
            ),
            sessions_root=cls._resolve_path(
                app_config.get("sessions_root"),
                base=resolved,
                fallback=data_root / "sessions",
            ),
            tool_runs_root=cls._resolve_path(
                app_config.get("tool_runs_root"),
                base=resolved,
                fallback=data_root / "tool_runs",
            ),
            examples_root=cls._resolve_path(
                app_config.get("examples_root"),
                base=resolved,
                fallback=data_root / "examples",
            ),
            default_output_format=app_config.get("default_output_format", "json"),
            default_session_id=app_config.get("default_session_id", "default"),
            deployment_mode=deployment_mode or "local_only",
            inference_provider=inference_provider or "local",
            openai_api_base=app_config.get("openai_api_base"),
            openai_responses_model=openai_model,
            provider_timeout_seconds=int(app_config.get("provider_timeout_seconds", 30)),
            mobile_research_note_auto_save=bool(app_config.get("mobile_research_note_auto_save", False)),
            context_match_limit=int(app_config.get("context_match_limit", 3)),
            search_candidate_limit=int(app_config.get("search_candidate_limit", 25)),
            context_prompt_max_length=int(app_config.get("context_prompt_max_length", 160)),
            context_summary_max_length=int(app_config.get("context_summary_max_length", 160)),
            session_objective_max_length=int(app_config.get("session_objective_max_length", 200)),
            session_thread_summary_max_length=int(app_config.get("session_thread_summary_max_length", 280)),
            max_interaction_record_bytes=int(app_config.get("max_interaction_record_bytes", 32 * 1024 * 1024)),
            max_session_state_bytes=int(app_config.get("max_session_state_bytes", 16 * 1024 * 1024)),
            tool_run_retention_per_capability=int(app_config.get("tool_run_retention_per_capability", 25)),
            config_path=config_loader.config_path if config_loader.exists() else None,
        )

    @staticmethod
    def _resolve_path(value: str | None, *, base: Path, fallback: Path) -> Path:
        if not value:
            return fallback.resolve()
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = base / candidate
        return candidate.resolve()
