from pathlib import Path

from lumen.app.controller import AppController


def test_initialize_workspace_creates_expected_directories_and_config(tmp_path: Path) -> None:
    (tmp_path / "lumen.toml.example").write_text(
        "[app]\ndefault_output_format = \"json\"\n",
        encoding="utf-8",
    )

    controller = AppController(repo_root=tmp_path)
    report = controller.initialize_workspace()

    assert report["status"] == "ok"
    assert (tmp_path / "data").exists()
    assert (tmp_path / "data" / "archive").exists()
    assert (tmp_path / "data" / "graph_memory").exists()
    assert (tmp_path / "data" / "interactions").exists()
    assert (tmp_path / "data" / "personal_memory").exists()
    assert (tmp_path / "data" / "research_notes").exists()
    assert (tmp_path / "data" / "research_artifacts").exists()
    assert (tmp_path / "data" / "labeled_datasets").exists()
    assert (tmp_path / "data" / "sessions").exists()
    assert (tmp_path / "data" / "tool_runs").exists()
    assert (tmp_path / "data" / "examples").exists()
    assert (tmp_path / "lumen.toml").exists()


def test_initialize_workspace_is_non_destructive_when_re_run(tmp_path: Path) -> None:
    (tmp_path / "lumen.toml.example").write_text(
        "[app]\ndefault_output_format = \"json\"\n",
        encoding="utf-8",
    )
    (tmp_path / "lumen.toml").write_text(
        "[app]\ndefault_output_format = \"text\"\n",
        encoding="utf-8",
    )

    controller = AppController(repo_root=tmp_path)
    report = controller.initialize_workspace()

    assert str((tmp_path / "lumen.toml").resolve()) in report["existing_paths"]
    assert "text" in (tmp_path / "lumen.toml").read_text(encoding="utf-8")


def test_list_recent_sessions_reads_latest_record_per_session(tmp_path: Path) -> None:
    (tmp_path / "lumen.toml.example").write_text(
        "[app]\ndefault_output_format = \"json\"\n",
        encoding="utf-8",
    )
    controller = AppController(repo_root=tmp_path)
    controller.initialize_workspace()

    session_a = controller.interaction_log_manager.interactions_root / "session-a"
    session_b = controller.interaction_log_manager.interactions_root / "session-b"
    session_a.mkdir(parents=True, exist_ok=True)
    session_b.mkdir(parents=True, exist_ok=True)

    (session_a / "20260323T100000000000Z.json").write_text(
        '{"session_id":"session-a","prompt":"old prompt","summary":"old summary","mode":"conversation","kind":"conversation.reply","created_at":"2026-03-23T10:00:00+00:00"}',
        encoding="utf-8",
    )
    (session_a / "20260323T110000000000Z.json").write_text(
        '{"session_id":"session-a","prompt":"new prompt","summary":"new summary","mode":"research","kind":"research.summary","created_at":"2026-03-23T11:00:00+00:00"}',
        encoding="utf-8",
    )
    (session_b / "20260323T120000000000Z.json").write_text(
        '{"session_id":"session-b","prompt":"another prompt","summary":"another summary","mode":"tool","kind":"tool.command_alias","created_at":"2026-03-23T12:00:00+00:00"}',
        encoding="utf-8",
    )
    (session_b / "index.json").write_text("[]", encoding="utf-8")

    report = controller.list_recent_sessions(limit=2)

    assert report["session_count"] == 2
    assert [session["session_id"] for session in report["sessions"]] == ["session-b", "session-a"]
    assert report["sessions"][1]["prompt"] == "new prompt"
    assert report["sessions"][1]["summary"] == "new summary"


def test_list_recent_sessions_skips_invalid_newest_json_and_falls_back(tmp_path: Path) -> None:
    (tmp_path / "lumen.toml.example").write_text(
        "[app]\ndefault_output_format = \"json\"\n",
        encoding="utf-8",
    )
    controller = AppController(repo_root=tmp_path)
    controller.initialize_workspace()

    session_dir = controller.interaction_log_manager.interactions_root / "session-a"
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "20260323T110000000000Z.json").write_text(
        '{"session_id":"session-a","prompt":"valid prompt","summary":"valid summary","mode":"research","kind":"research.summary","created_at":"2026-03-23T11:00:00+00:00"}',
        encoding="utf-8",
    )
    (session_dir / "20260323T120000000000Z.json").write_text(
        '["not", "an", "interaction", "record"]',
        encoding="utf-8",
    )

    report = controller.list_recent_sessions(limit=1)

    assert report["session_count"] == 1
    assert report["sessions"][0]["session_id"] == "session-a"
    assert report["sessions"][0]["prompt"] == "valid prompt"
    assert report["sessions"][0]["summary"] == "valid summary"


def test_archive_session_hides_it_from_recent_sessions(tmp_path: Path) -> None:
    (tmp_path / "lumen.toml.example").write_text(
        "[app]\ndefault_output_format = \"json\"\n",
        encoding="utf-8",
    )
    controller = AppController(repo_root=tmp_path)
    controller.initialize_workspace()

    session_dir = controller.interaction_log_manager.interactions_root / "session-a"
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "20260323T110000000000Z.json").write_text(
        '{"session_id":"session-a","prompt":"valid prompt","summary":"valid summary","mode":"research","kind":"research.summary","created_at":"2026-03-23T11:00:00+00:00"}',
        encoding="utf-8",
    )

    controller.archive_session("session-a")
    report = controller.list_recent_sessions(limit=5)

    assert report["session_count"] == 0
    assert report["sessions"] == []

    archived_report = controller.list_recent_sessions(limit=5, include_archived=True, archived_only=True)
    assert archived_report["session_count"] == 1
    assert archived_report["sessions"][0]["session_id"] == "session-a"


def test_list_recent_sessions_filters_db_rows_without_restorable_history(tmp_path: Path) -> None:
    (tmp_path / "lumen.toml.example").write_text(
        "[app]\ndefault_output_format = \"json\"\n",
        encoding="utf-8",
    )
    controller = AppController(repo_root=tmp_path)
    controller.initialize_workspace()
    controller.persistence_manager.bootstrap()

    controller.persistence_manager.sessions.upsert(
        session_id="session-dead",
        project_id=None,
        title="Dead session",
        mode="conversation",
        started_at="2026-03-23T10:00:00+00:00",
        updated_at="2026-03-23T10:00:00+00:00",
        status="active",
        metadata=None,
    )
    controller.persistence_manager.sessions.upsert(
        session_id="session-live",
        project_id=None,
        title="Live session",
        mode="research",
        started_at="2026-03-23T11:00:00+00:00",
        updated_at="2026-03-23T11:00:00+00:00",
        status="active",
        metadata=None,
    )

    live_dir = controller.interaction_log_manager.interactions_root / "session-live"
    live_dir.mkdir(parents=True, exist_ok=True)
    (live_dir / "20260323T110000000000Z.json").write_text(
        '{"session_id":"session-live","prompt":"valid prompt","summary":"valid summary","mode":"research","kind":"research.summary","created_at":"2026-03-23T11:00:00+00:00"}',
        encoding="utf-8",
    )

    report = controller.list_recent_sessions(limit=5)

    assert [session["session_id"] for session in report["sessions"]] == ["session-live"]


def test_list_recent_sessions_hides_internal_validation_sessions(tmp_path: Path) -> None:
    (tmp_path / "lumen.toml.example").write_text(
        "[app]\ndefault_output_format = \"json\"\n",
        encoding="utf-8",
    )
    controller = AppController(repo_root=tmp_path)
    controller.initialize_workspace()

    controller.interaction_log_manager.record_interaction(
        session_id="packaged-greeting-default",
        prompt="hello",
        response={
            "mode": "conversation",
            "kind": "conversation.greeting",
            "summary": "Hi.",
        },
    )
    controller.interaction_log_manager.record_interaction(
        session_id="desktop-user-chat",
        prompt="real user chat",
        response={
            "mode": "conversation",
            "kind": "conversation.reply",
            "summary": "Real reply.",
        },
    )

    report = controller.list_recent_sessions(limit=10)

    assert [session["session_id"] for session in report["sessions"]] == ["desktop-user-chat"]


def test_first_chat_turn_creates_single_session_with_ordered_messages(tmp_path: Path) -> None:
    (tmp_path / "lumen.toml.example").write_text(
        "[app]\ndefault_output_format = \"json\"\n",
        encoding="utf-8",
    )
    controller = AppController(repo_root=tmp_path)

    controller.set_session_profile("desktop-new", interaction_style="direct")
    assert controller.persistence_manager.sessions.get("desktop-new") is None

    controller.ask(prompt="hello lumen", session_id="desktop-new")

    sessions = controller.persistence_manager.sessions.list_recent(limit=5)
    messages = controller.persistence_manager.messages.list_by_session("desktop-new")
    assert [session["id"] for session in sessions] == ["desktop-new"]
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert messages[0]["content"] == "hello lumen"


def test_active_thread_state_does_not_replace_session_title_with_assistant_summary(tmp_path: Path) -> None:
    (tmp_path / "lumen.toml.example").write_text(
        "[app]\ndefault_output_format = \"json\"\n",
        encoding="utf-8",
    )
    controller = AppController(repo_root=tmp_path)

    controller.ask(prompt="hey buddy", session_id="desktop-title-check")

    session = controller.persistence_manager.sessions.get("desktop-title-check")

    assert session is not None
    assert session["title"] == "hey buddy"
    assert "best first read" not in str(session["title"]).lower()


def test_archive_and_delete_memory_change_controller_lists(tmp_path: Path) -> None:
    (tmp_path / "lumen.toml.example").write_text(
        "[app]\ndefault_output_format = \"json\"\n",
        encoding="utf-8",
    )
    controller = AppController(repo_root=tmp_path)
    controller.initialize_workspace()

    session_id = "session-a"
    personal_dir = controller.settings.personal_memory_root / session_id
    personal_dir.mkdir(parents=True, exist_ok=True)
    personal_path = personal_dir / "20260323T110000000000Z.json"
    personal_path.write_text(
        '{"schema_type":"personal_memory","schema_version":"1","session_id":"session-a","created_at":"2026-03-23T11:00:00+00:00","title":"Preference","content":"Summary: keep it brief","source_interaction_prompt":"remember this","source_interaction_mode":"conversation"}',
        encoding="utf-8",
    )

    research_dir = controller.settings.research_notes_root / session_id
    research_dir.mkdir(parents=True, exist_ok=True)
    research_path = research_dir / "20260323T120000000000Z.json"
    research_path.write_text(
        '{"schema_type":"research_note","schema_version":"1","session_id":"session-a","created_at":"2026-03-23T12:00:00+00:00","note_type":"chronological_research_note","title":"Gravity","content":"Prompt: gravity","source_interaction_prompt":"gravity","source_interaction_mode":"research","source_interaction_kind":"research.summary"}',
        encoding="utf-8",
    )

    controller.archive_memory(kind="personal_memory", path=str(personal_path))
    assert controller.list_personal_memory(session_id=session_id)["entry_count"] == 0

    controller.delete_memory(kind="research_note", path=str(research_path))
    assert controller.list_research_notes(session_id=session_id)["note_count"] == 0


def test_archive_and_delete_db_backed_memory_by_memory_id(tmp_path: Path) -> None:
    (tmp_path / "lumen.toml.example").write_text(
        "[app]\ndefault_output_format = \"json\"\n",
        encoding="utf-8",
    )
    controller = AppController(repo_root=tmp_path)
    controller.initialize_workspace()

    personal = controller.persistence_manager.record_memory_item(
        source_type="personal_memory",
        payload={
            "session_id": "session-a",
            "created_at": "2026-03-23T11:00:00+00:00",
            "title": "Preference",
            "content": "Summary: keep it brief",
            "source_interaction_prompt": "remember this",
            "source_interaction_mode": "conversation",
        },
    )
    note = controller.persistence_manager.record_memory_item(
        source_type="research_note",
        payload={
            "session_id": "session-a",
            "created_at": "2026-03-23T12:00:00+00:00",
            "title": "Gravity",
            "content": "Prompt: gravity",
            "source_interaction_prompt": "gravity",
            "source_interaction_mode": "research",
            "source_interaction_kind": "research.summary",
        },
    )

    controller.archive_memory(kind="personal_memory", path=str(personal["id"]))
    assert controller.persistence_manager.memory_items.get(str(personal["id"]))["status"] == "archived"

    controller.delete_memory(kind="research_note", path=str(note["id"]))
    assert controller.persistence_manager.memory_items.get(str(note["id"])) is None
