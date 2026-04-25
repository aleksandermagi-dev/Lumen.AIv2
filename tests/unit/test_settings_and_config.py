from pathlib import Path

from lumen.app.runtime_paths import default_data_root, detect_runtime_root, resolve_desktop_runtime_paths
from lumen.app.settings import AppSettings


def test_settings_use_repo_defaults_without_config(tmp_path: Path) -> None:
    settings = AppSettings.from_repo_root(tmp_path)

    assert settings.repo_root == tmp_path.resolve()
    assert settings.data_root == (tmp_path / "data").resolve()
    assert settings.persistence_db_path == (tmp_path / "data" / "persistence" / "lumen.sqlite3").resolve()
    assert settings.knowledge_root == (tmp_path / "data" / "knowledge").resolve()
    assert settings.knowledge_db_path == (tmp_path / "data" / "knowledge" / "lumen_knowledge.sqlite3").resolve()
    assert settings.graph_memory_db_path == (tmp_path / "data" / "graph_memory" / "lumen_memory.sqlite3").resolve()
    assert settings.archive_root == (tmp_path / "data" / "archive").resolve()
    assert settings.interactions_root == (tmp_path / "data" / "interactions").resolve()
    assert settings.personal_memory_root == (tmp_path / "data" / "personal_memory").resolve()
    assert settings.research_notes_root == (tmp_path / "data" / "research_notes").resolve()
    assert settings.research_artifacts_root == (tmp_path / "data" / "research_artifacts").resolve()
    assert settings.sessions_root == (tmp_path / "data" / "sessions").resolve()
    assert settings.tool_runs_root == (tmp_path / "data" / "tool_runs").resolve()
    assert settings.examples_root == (tmp_path / "data" / "examples").resolve()
    assert settings.default_output_format == "json"
    assert settings.deployment_mode == "local_only"
    assert settings.inference_provider == "local"
    assert settings.openai_api_base is None
    assert settings.openai_responses_model is None
    assert settings.provider_timeout_seconds == 30
    assert settings.mobile_research_note_auto_save is False
    assert settings.context_match_limit == 3
    assert settings.search_candidate_limit == 25
    assert settings.context_prompt_max_length == 160
    assert settings.context_summary_max_length == 160
    assert settings.session_objective_max_length == 200
    assert settings.session_thread_summary_max_length == 280
    assert settings.config_path is None


def test_settings_load_overrides_from_lumen_toml(tmp_path: Path) -> None:
    (tmp_path / "lumen.toml").write_text(
        "\n".join(
            [
                "[app]",
                'default_output_format = "text"',
                'default_session_id = "lab"',
                'deployment_mode = "hybrid"',
                'inference_provider = "openai_responses"',
                'openai_api_base = "https://api.openai.com/v1"',
                'openai_responses_model = "gpt-5"',
                "provider_timeout_seconds = 45",
                'data_root = "workspace_data"',
                'persistence_db_path = "workspace_data/persistence/lumen.sqlite3"',
                'knowledge_root = "workspace_knowledge"',
                'knowledge_db_path = "workspace_knowledge/lumen.sqlite3"',
                'graph_memory_db_path = "workspace_graph/lumen_memory.sqlite3"',
                'archive_root = "workspace_archive"',
                'interactions_root = "workspace_interactions"',
                'personal_memory_root = "workspace_personal_memory"',
                'research_notes_root = "workspace_research_notes"',
                'research_artifacts_root = "workspace_research_artifacts"',
                'sessions_root = "workspace_sessions"',
                'tool_runs_root = "workspace_runs"',
                'examples_root = "workspace_examples"',
                "mobile_research_note_auto_save = true",
                "context_match_limit = 5",
                "search_candidate_limit = 7",
                "context_prompt_max_length = 120",
                "context_summary_max_length = 140",
                "session_objective_max_length = 180",
                "session_thread_summary_max_length = 240",
            ]
        ),
        encoding="utf-8",
    )

    settings = AppSettings.from_repo_root(tmp_path)

    assert settings.default_output_format == "text"
    assert settings.default_session_id == "lab"
    assert settings.deployment_mode == "hybrid"
    assert settings.inference_provider == "openai_responses"
    assert settings.openai_api_base == "https://api.openai.com/v1"
    assert settings.openai_responses_model == "gpt-5"
    assert settings.provider_timeout_seconds == 45
    assert settings.data_root == (tmp_path / "workspace_data").resolve()
    assert settings.persistence_db_path == (tmp_path / "workspace_data" / "persistence" / "lumen.sqlite3").resolve()
    assert settings.knowledge_root == (tmp_path / "workspace_knowledge").resolve()
    assert settings.knowledge_db_path == (tmp_path / "workspace_knowledge" / "lumen.sqlite3").resolve()
    assert settings.graph_memory_db_path == (tmp_path / "workspace_graph" / "lumen_memory.sqlite3").resolve()
    assert settings.archive_root == (tmp_path / "workspace_archive").resolve()
    assert settings.interactions_root == (tmp_path / "workspace_interactions").resolve()
    assert settings.personal_memory_root == (tmp_path / "workspace_personal_memory").resolve()
    assert settings.research_notes_root == (tmp_path / "workspace_research_notes").resolve()
    assert settings.research_artifacts_root == (tmp_path / "workspace_research_artifacts").resolve()
    assert settings.sessions_root == (tmp_path / "workspace_sessions").resolve()
    assert settings.tool_runs_root == (tmp_path / "workspace_runs").resolve()
    assert settings.examples_root == (tmp_path / "workspace_examples").resolve()
    assert settings.mobile_research_note_auto_save is True
    assert settings.context_match_limit == 5
    assert settings.search_candidate_limit == 7
    assert settings.context_prompt_max_length == 120
    assert settings.context_summary_max_length == 140
    assert settings.session_objective_max_length == 180
    assert settings.session_thread_summary_max_length == 240
    assert settings.config_path == (tmp_path / "lumen.toml").resolve()


def test_settings_use_explicit_data_root_override(tmp_path: Path) -> None:
    explicit_data_root = tmp_path / "runtime-data"

    settings = AppSettings.from_repo_root(tmp_path, data_root_override=explicit_data_root)

    assert settings.repo_root == tmp_path.resolve()
    assert settings.data_root == explicit_data_root.resolve()
    assert settings.persistence_db_path == (explicit_data_root / "persistence" / "lumen.sqlite3").resolve()
    assert settings.archive_root == (explicit_data_root / "archive").resolve()
    assert settings.interactions_root == (explicit_data_root / "interactions").resolve()


def test_detect_runtime_root_prefers_explicit_repo_root(tmp_path: Path) -> None:
    resolved = detect_runtime_root(repo_root=tmp_path)

    assert resolved == tmp_path.resolve()


def test_default_data_root_uses_runtime_root_in_source_mode(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("lumen.app.runtime_paths.is_frozen_app", lambda: False)

    data_root = default_data_root(runtime_root=tmp_path)

    assert data_root == (tmp_path / "data").resolve()


def test_resolve_desktop_runtime_paths_returns_source_defaults(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("lumen.app.runtime_paths.is_frozen_app", lambda: False)

    paths = resolve_desktop_runtime_paths(repo_root=tmp_path)

    assert paths.runtime_root == tmp_path.resolve()
    assert paths.data_root == (tmp_path / "data").resolve()
    assert paths.execution_mode == "source"
