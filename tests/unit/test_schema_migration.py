from pathlib import Path
import json

from lumen.app.settings import AppSettings
from lumen.app.models import InteractionProfile
from lumen.memory.archive_manager import ArchiveManager
from lumen.memory.interaction_log_manager import InteractionLogManager
from lumen.memory.session_state_manager import SessionStateManager
from lumen.schemas.migration import SchemaMigration


def test_archive_record_migrates_legacy_version_on_load(tmp_path: Path) -> None:
    settings = AppSettings.from_repo_root(tmp_path)
    manager = ArchiveManager(settings=settings)
    archive_dir = settings.archive_root / "default" / "anh" / "spectral_dip_scan"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / "legacy.json"
    archive_path.write_text(
        json.dumps(
            {
                "schema_version": "0",
                "session_id": "default",
                "tool_id": "anh",
                "capability": "spectral_dip_scan",
                "status": "ok",
                "summary": "legacy run",
                "created_at": "2026-03-15T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    record = manager.load_record(archive_path)

    assert record["schema_type"] == "archive_record"
    assert record["schema_version"] == "1"


def test_interaction_record_migrates_legacy_version_on_load(tmp_path: Path) -> None:
    settings = AppSettings.from_repo_root(tmp_path)
    manager = InteractionLogManager(settings=settings)
    interaction_dir = settings.interactions_root / "default"
    interaction_dir.mkdir(parents=True, exist_ok=True)
    interaction_path = interaction_dir / "legacy.json"
    interaction_path.write_text(
        json.dumps(
            {
                "schema_version": "0",
                "session_id": "default",
                "prompt": "create a migration plan for lumen",
                "mode": "planning",
                "kind": "planning.migration",
                "summary": "Planning response for: create a migration plan for lumen",
                "created_at": "2026-03-15T00:00:00+00:00",
                "response": {"mode": "planning"},
            }
        ),
        encoding="utf-8",
    )

    record = manager.load_record(interaction_path)

    assert record["schema_type"] == "interaction_record"
    assert record["schema_version"] == "5"
    assert record["memory_classification"]["candidate_type"] == "ephemeral_conversation_context"
    assert record["memory_write_decision"]["action"] == "skip"


def test_schema_migration_can_tolerate_newer_known_versions_when_enabled() -> None:
    migration = SchemaMigration(
        schema_type="interaction_record",
        current_version="3",
        allow_newer_versions=True,
        migrations={
            "1": lambda payload: {**payload, "schema_version": "2"},
            "2": lambda payload: {**payload, "schema_version": "3"},
        },
    )

    payload = migration.migrate(
        {
            "schema_type": "interaction_record",
            "schema_version": "5",
            "session_id": "default",
        }
    )

    assert payload["schema_version"] == "5"


def test_interaction_record_load_tolerates_future_schema_version(tmp_path: Path) -> None:
    settings = AppSettings.from_repo_root(tmp_path)
    manager = InteractionLogManager(settings=settings)
    interaction_dir = settings.interactions_root / "default"
    interaction_dir.mkdir(parents=True, exist_ok=True)
    interaction_path = interaction_dir / "future.json"
    interaction_path.write_text(
        json.dumps(
            {
                "schema_type": "interaction_record",
                "schema_version": "6",
                "session_id": "default",
                "prompt": "create a migration plan for lumen",
                "mode": "planning",
                "kind": "planning.migration",
                "summary": "Planning response for: create a migration plan for lumen",
                "created_at": "2026-03-15T00:00:00+00:00",
                "response": {"mode": "planning", "summary": "Planning response"},
            }
        ),
        encoding="utf-8",
    )

    record = manager.load_record(interaction_path)

    assert record["schema_type"] == "interaction_record"
    assert record["schema_version"] == "6"


def test_session_thread_migrates_legacy_version_on_load(tmp_path: Path) -> None:
    settings = AppSettings.from_repo_root(tmp_path)
    manager = SessionStateManager(settings=settings)
    state_dir = settings.sessions_root / "default"
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "thread_state.json"
    state_path.write_text(
        json.dumps(
            {
                "schema_version": "0",
                "session_id": "default",
                "mode": "planning",
                "kind": "planning.migration",
                "prompt": "create a migration plan for lumen",
                "objective": "Plan work for: create a migration plan for lumen",
                "thread_summary": "Planning response for: create a migration plan for lumen",
                "summary": "Planning response for: create a migration plan for lumen",
                "updated_at": "2026-03-15T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    record = manager.get_active_thread("default")

    assert record is not None
    assert record["schema_type"] == "session_thread_state"
    assert record["schema_version"] == "1"


def test_session_state_manager_prefers_saved_session_profile_over_response_profile(tmp_path: Path) -> None:
    settings = AppSettings.from_repo_root(tmp_path)
    manager = SessionStateManager(settings=settings)
    manager.set_interaction_profile(
        "default",
        InteractionProfile(
            interaction_style="collab",
            reasoning_depth="normal",
            selection_source="user",
            allow_suggestions=True,
        ),
    )

    payload = manager.update_active_thread(
        session_id="default",
        prompt="summarize the archive structure",
        response={
            "mode": "research",
            "kind": "research.summary",
            "summary": "Archive structure summary",
            "interaction_profile": {
                "interaction_style": "direct",
                "reasoning_depth": "normal",
                "selection_source": "suggested",
                "allow_suggestions": True,
            },
        },
    )

    assert payload["interaction_profile"]["interaction_style"] == "collab"
    assert manager.get_active_thread("default")["interaction_profile"]["interaction_style"] == "collab"


def test_session_state_manager_defaults_new_profiles_to_canonical_style(tmp_path: Path) -> None:
    settings = AppSettings.from_repo_root(tmp_path)
    manager = SessionStateManager(settings=settings)

    payload = manager.set_interaction_profile("default", InteractionProfile.default())

    assert payload["interaction_style"] == "collab"
    assert manager.get_interaction_profile("default").interaction_style == "collab"


def test_session_state_manager_persists_continuation_offer_explanation_mode(tmp_path: Path) -> None:
    settings = AppSettings.from_repo_root(tmp_path)
    manager = SessionStateManager(settings=settings)

    payload = manager.update_active_thread(
        session_id="default",
        prompt="what is a galaxy",
        response={
            "mode": "research",
            "kind": "research.summary",
            "summary": "A galaxy is a huge system of stars.",
            "continuation_offer": {
                "kind": "break_down",
                "topic": "galaxy",
                "target_prompt": "break it down",
                "label": "I can break that down more simply if you want.",
                "explanation_mode": "break_down",
            },
        },
    )

    assert payload["continuation_offer"]["explanation_mode"] == "break_down"
    assert manager.get_active_thread("default")["continuation_offer"]["explanation_mode"] == "break_down"

