from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import time
from types import SimpleNamespace
import tkinter as tk
from collections import Counter

import pytest

import lumen.desktop.chat_app as chat_app_module
from lumen.desktop.chat_app import LumenDesktopApp
from lumen.desktop.chat_layout_support import bubble_side_padding, bubble_wraplength
from lumen.desktop.desktop_crash_support import read_crash_records
from lumen.desktop.desktop_startup_snapshot_support import build_startup_snapshot
from lumen.desktop.desktop_startup_state_support import capability_state_from_startup_health
from lumen.desktop.desktop_style_support import (
    resolve_composer_button_palette,
    resolve_control_availability,
    resolve_input_palette,
    resolve_load_more_palette,
    resolve_nav_button_visual,
    resolve_top_icon_palette,
)
from lumen.desktop.desktop_view_state_support import resolve_view_refresh_decision
from lumen.desktop.memory_archive_support import build_memory_row_cache, memory_entry_key
from lumen.desktop.shell_transition_support import DebugTraceSession, DesktopCapabilityState
from lumen.desktop.chat_ui_support import (
    DARK_PALETTE,
    DesktopChatMessage,
    LIGHT_PALETTE,
    LUMEN_PURPLE_PALETTE,
    PALETTE_KEYS,
    THEME_TOKENS,
    THEME_PALETTES,
    build_pending_message,
    custom_accent_palette,
    custom_accent_theme,
    day_group_label,
    empty_state_text,
    grouped_session_rows,
    human_date_label,
    message_role_style,
    nav_button_style,
    neutral_pending_phrase,
    palette_from_theme,
    resolve_theme_tokens,
    select_cognitive_indicator,
    validate_palette,
)


class _FakeController:
    def __init__(
        self,
        *,
        repo_root: Path,
        data_root: Path | None = None,
        execution_mode: str = "source",
    ):
        self.repo_root = repo_root
        self.execution_mode = execution_mode
        self.settings = SimpleNamespace(data_root=(data_root or (repo_root / "data")))
        self.model_provider = SimpleNamespace(provider_id="local")
        self.research_note_calls = 0
        self.personal_memory_calls = 0
        self.knowledge_overview_calls = 0
        self.recent_session_calls = 0
        self.ask_calls: list[dict[str, object]] = []
        self.profile_calls: list[tuple[str, str]] = []
        self._recent_sessions = [
            {
                "session_id": "desktop-1",
                "summary": "Black holes overview",
                "prompt": "tell me about black holes",
                "mode": "research",
                "kind": "research.summary",
                "created_at": "2026-03-23T12:00:00+00:00",
            }
        ]
        self.renamed_sessions: list[tuple[str, str | None]] = []
        self.archived_sessions: set[str] = set()
        self.deleted_sessions: set[str] = set()
        self.archived_memory_paths: list[str] = []
        self.deleted_memory_paths: list[str] = []

    def set_session_profile(self, session_id: str, *, interaction_style: str) -> None:
        self.profile_calls.append((session_id, interaction_style))
        return None

    def reset_session_thread(self, session_id: str) -> None:
        return None

    def list_research_notes(
        self,
        *,
        session_id: str | None = None,
        include_archived: bool = False,
        archived_only: bool = False,
        limit: int | None = None,
    ) -> dict[str, object]:
        self.research_note_calls += 1
        notes = [
                {
                    "title": "Gravity note",
                    "created_at": "2026-03-23T12:00:00+00:00",
                    "note_path": str(self.settings.data_root / "research_notes" / "desktop-1" / "gravity.json"),
                },
                {
                    "title": "Orbit note",
                    "created_at": "2026-03-22T12:00:00+00:00",
                    "note_path": str(self.settings.data_root / "research_notes" / "desktop-1" / "orbit.json"),
                },
            ]
        visible: list[dict[str, object]] = []
        for note in notes:
            note_path = str(note.get("note_path") or "")
            if note_path in self.deleted_memory_paths:
                continue
            is_archived = note_path in self.archived_memory_paths
            if archived_only and not is_archived:
                continue
            if not include_archived and is_archived:
                continue
            visible.append(note)
            if limit is not None and len(visible) >= max(int(limit), 1):
                break
        return {"research_notes": visible}

    def list_personal_memory(
        self,
        *,
        session_id: str | None = None,
        include_archived: bool = False,
        archived_only: bool = False,
        limit: int | None = None,
    ) -> dict[str, object]:
        self.personal_memory_calls += 1
        entries = [
                {
                    "title": "Preference",
                    "content": "Summary: keep it brief",
                    "created_at": "2026-03-23T09:30:00+00:00",
                    "entry_path": str(self.settings.data_root / "personal_memory" / "desktop-1" / "pref.json"),
                },
                {
                    "title": "Project note",
                    "content": "Summary: keep the UI minimal",
                    "created_at": "2026-03-20T09:30:00+00:00",
                    "entry_path": str(self.settings.data_root / "personal_memory" / "desktop-1" / "project.json"),
                },
            ]
        visible: list[dict[str, object]] = []
        for entry in entries:
            entry_path = str(entry.get("entry_path") or "")
            if entry_path in self.deleted_memory_paths:
                continue
            is_archived = entry_path in self.archived_memory_paths
            if archived_only and not is_archived:
                continue
            if not include_archived and is_archived:
                continue
            visible.append(entry)
            if limit is not None and len(visible) >= max(int(limit), 1):
                break
        return {"personal_memory": visible}

    def list_archived_memory(self, *, session_id: str | None = None) -> dict[str, object]:
        personal = self.list_personal_memory(session_id=session_id, include_archived=True, archived_only=True).get("personal_memory", [])
        research = self.list_research_notes(session_id=session_id, include_archived=True, archived_only=True).get("research_notes", [])
        return {
            "archived_memory": [
                *[
                    {
                        "title": entry["title"],
                        "content": entry["content"],
                        "created_at": entry["created_at"],
                        "kind": "personal_memory",
                        "entry_path": entry["entry_path"],
                    }
                    for entry in personal
                ],
                *[
                    {
                        "title": note["title"],
                        "content": "",
                        "created_at": note["created_at"],
                        "kind": "research_note",
                        "note_path": note["note_path"],
                    }
                    for note in research
                ],
            ]
        }

    def knowledge_overview(self) -> dict[str, object]:
        self.knowledge_overview_calls += 1
        return {
            "categories": [
                {"category": "astronomy", "entry_count": 2, "titles": ["Black Hole", "Saturn"]},
                {"category": "physics", "entry_count": 1, "titles": ["Gravity"]},
            ]
        }

    def list_memory_topics(self, *, session_id: str | None = None) -> dict[str, object]:
        personal = self.list_personal_memory(session_id=session_id).get("personal_memory", [])
        research = self.list_research_notes(session_id=session_id).get("research_notes", [])
        return {
            "topics": [
                {
                    "topic": "general",
                    "count": len(personal) + len(research),
                    "entries": [
                        *[
                            {
                                "title": item["title"],
                                "content": item["content"],
                                "created_at": item["created_at"],
                                "kind": "personal_memory",
                            }
                            for item in personal
                        ],
                        *[
                            {
                                "title": item["title"],
                                "content": "",
                                "created_at": item["created_at"],
                                "kind": "research_note",
                            }
                            for item in research
                        ],
                    ],
                }
            ]
        }

    def list_recent_sessions(
        self,
        *,
        limit: int = 10,
        include_archived: bool = False,
        archived_only: bool = False,
    ) -> dict[str, object]:
        self.recent_session_calls += 1
        visible = []
        for session in self._recent_sessions:
            session_id = str(session.get("session_id") or "")
            is_archived = session_id in self.archived_sessions
            if session_id in self.deleted_sessions:
                continue
            if archived_only and not is_archived:
                continue
            if not include_archived and is_archived:
                continue
            visible.append(session)
        return {"sessions": list(visible[:limit])}

    def rename_session(self, session_id: str, *, title: str | None) -> dict[str, object]:
        self.renamed_sessions.append((session_id, title))
        for session in self._recent_sessions:
            if session.get("session_id") == session_id:
                session["title"] = title
                break
        return {"session_id": session_id, "title": title}

    def archive_session(self, session_id: str) -> dict[str, object]:
        self.archived_sessions.add(session_id)
        return {"session_id": session_id, "archived": True}

    def delete_session(self, session_id: str) -> dict[str, object]:
        self.deleted_sessions.add(session_id)
        self._recent_sessions = [
            session for session in self._recent_sessions if session.get("session_id") != session_id
        ]
        return {"session_id": session_id, "deleted": True}

    def archive_memory(self, *, kind: str, path: str) -> dict[str, object]:
        self.archived_memory_paths.append(path)
        return {"kind": kind, "path": path, "archived": True}

    def delete_memory(self, *, kind: str, path: str) -> dict[str, object]:
        self.deleted_memory_paths.append(path)
        return {"kind": kind, "path": path, "deleted": True}

    def list_interactions(self, *, session_id: str | None = None) -> dict[str, object]:
        return {
            "interaction_records": [
                {
                    "prompt": "tell me about black holes",
                    "summary": "Black holes overview",
                    "mode": "research",
                    "created_at": "2026-03-23T12:00:00+00:00",
                    "response": {"mode": "research", "summary": "Black holes overview"},
                }
            ]
        }

    def get_session_profile(self, session_id: str) -> dict[str, object]:
        return {"interaction_profile": {"interaction_style": "default"}}

    def ask(
        self,
        *,
        prompt: str,
        input_path: Path | None = None,
        session_id: str,
        interaction_style: str | None = None,
        client_surface: str | None = None,
    ) -> dict[str, object]:
        self.ask_calls.append(
            {
                "prompt": prompt,
                "input_path": input_path,
                "session_id": session_id,
                "interaction_style": interaction_style,
                "client_surface": client_surface,
            }
        )
        return {
            "mode": "conversation",
            "summary": f"Full answer for: {prompt}",
            "reply": f"Full answer for: {prompt}",
            "user_facing_answer": f"Full answer for: {prompt}",
        }

    def list_tools(self) -> list[str]:
        return ["workspace", "report", "memory", "math", "system", "knowledge"]

    def build_doctor_report(self) -> dict[str, object]:
        return {
            "checks": [
                {
                    "name": "tool_registry",
                    "missing_bundles": [],
                },
                {
                    "name": "runtime_resources",
                    "missing_required_resources": [],
                },
                {
                    "name": "runtime_layout",
                    "runtime_root": str(self.repo_root),
                    "data_root": str(self.settings.data_root),
                },
            ]
        }

    def list_app_capabilities(self) -> dict[str, dict[str, str]]:
        return {
            "workspace.inspect": {
                "tool_id": "workspace",
                "tool_capability": "inspect",
                "description": "Inspect workspace state.",
            },
            "memory.read": {
                "tool_id": "memory",
                "tool_capability": "read",
                "description": "Read memory surfaces.",
            },
        }

    def capability_contract_report(self) -> dict[str, object]:
        return {"contracts": []}


class _ImmediateThread:
    def __init__(self, *, target=None, args=(), daemon=None):
        self._target = target
        self._args = args
        self.daemon = daemon

    def start(self) -> None:
        if self._target is not None:
            self._target(*self._args)


class _DeferredThread:
    def __init__(self, *, target=None, args=(), daemon=None):
        self._target = target
        self._args = args
        self.daemon = daemon

    def start(self) -> None:
        return None


def _install_immediate_ui_task_runner(monkeypatch) -> None:
    def _run(self, task_name: str, worker, *, timing_label: str | None = None) -> int:
        token = self._next_async_token(task_name)
        try:
            payload = worker()
        except Exception as exc:
            self.ui_task_queue.put(("error", task_name, token, exc))
        else:
            self.ui_task_queue.put(
                (
                    "result",
                    task_name,
                    token,
                    {
                        "payload": payload,
                        "elapsed_ms": 0.0,
                        "timing_label": timing_label or task_name,
                    },
                )
            )
        return token

    monkeypatch.setattr(LumenDesktopApp, "_start_ui_background_task", _run)


def _destroy_app_root(root: tk.Tk, app: LumenDesktopApp | None = None) -> None:
    if app is None:
        candidate = getattr(root, "_lumen_app", None)
        if isinstance(candidate, LumenDesktopApp):
            app = candidate
    if app is not None:
        try:
            app._on_root_destroy()
        except Exception:
            pass
    try:
        root.quit()
        root.withdraw()
        root.update_idletasks()
        root.destroy()
    except tk.TclError:
        return


def test_destroy_app_root_is_idempotent(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    app = LumenDesktopApp(root, repo_root=Path.cwd())
    _destroy_app_root(root, app)
    _destroy_app_root(root, app)


def test_dark_palette_uses_dark_foundation() -> None:
    assert DARK_PALETTE["app_bg"] == "#0f1117"
    assert DARK_PALETTE["sidebar_bg"] == "#0c0e13"
    assert DARK_PALETTE["text_primary"] != "#ffffff"
    assert DARK_PALETTE["panel_bg"].startswith("#")
    assert set(PALETTE_KEYS).issubset(DARK_PALETTE.keys())
    validate_palette(DARK_PALETTE)


def test_light_palette_is_available_for_theme_toggle() -> None:
    assert THEME_PALETTES["dark"]["app_bg"] == DARK_PALETTE["app_bg"]
    assert THEME_PALETTES["light"]["app_bg"] == LIGHT_PALETTE["app_bg"]
    assert THEME_PALETTES["custom"]["nav_active_border"].startswith("#")
    assert LIGHT_PALETTE["app_bg"] != DARK_PALETTE["app_bg"]
    validate_palette(LIGHT_PALETTE)


def test_theme_palettes_are_isolated_and_light_uses_accent_not_gray() -> None:
    light = dict(THEME_PALETTES["light"])
    dark_before = dict(THEME_PALETTES["dark"])
    purple_before = dict(THEME_PALETTES["custom"])

    light["app_bg"] = "#123456"

    assert THEME_PALETTES["dark"] == dark_before
    assert THEME_PALETTES["custom"] == purple_before
    assert LIGHT_PALETTE["nav_active_border"] == THEME_TOKENS["light"]["accent"]
    assert LIGHT_PALETTE["nav_active_border"] == "#8d56ee"
    assert LUMEN_PURPLE_PALETTE["app_bg"] != LIGHT_PALETTE["app_bg"]


def test_pending_indicator_selector_stays_in_mode_pool() -> None:
    conversation = select_cognitive_indicator(mode="conversation", key="hello")
    research = select_cognitive_indicator(mode="research", key="what is gravity")
    engineering = select_cognitive_indicator(mode="planning", key="implement routing")

    assert conversation.startswith("Lumen is ")
    assert conversation in {
        "Lumen is reasoning...",
        "Lumen is following the thread...",
        "Lumen is working through it...",
    }
    assert research in {
        "Lumen is analyzing...",
        "Lumen is examining the question...",
        "Lumen is tracing the evidence...",
    }
    assert engineering in {
        "Lumen is tracing...",
        "Lumen is analyzing the structure...",
        "Lumen is checking the system...",
    }


def test_pending_message_uses_neutral_safe_phrasing() -> None:
    message = build_pending_message(key="tell me about black holes")
    assert message.message_type == "pending"
    assert message.mode == "conversation"
    assert message.text in {
        "Lumen is reasoning...",
        "Lumen is working through it...",
        "Lumen is with you on this...",
    }
    assert message.text not in {
        "Lumen is analyzing...",
        "Lumen is examining the question...",
    }
    assert isinstance(message.timestamp, str)
    assert ":" in message.timestamp


def test_message_role_style_distinguishes_user_and_assistant_alignment() -> None:
    user_style = message_role_style("user")
    assistant_style = message_role_style("assistant")
    pending_style = message_role_style("pending")
    light_user_style = message_role_style("user", palette=LIGHT_PALETTE)

    assert user_style["anchor"] == "e"
    assert assistant_style["anchor"] == "w"
    assert pending_style["anchor"] == "w"
    assert user_style["bubble_bg"] != assistant_style["bubble_bg"]
    assert light_user_style["bubble_bg"] == LIGHT_PALETTE["user_bg"]


def test_neutral_pending_phrase_does_not_use_mode_specific_research_copy() -> None:
    phrase = neutral_pending_phrase(key="architecture")

    assert phrase in {
        "Lumen is reasoning...",
        "Lumen is working through it...",
        "Lumen is with you on this...",
    }
    assert phrase not in {
        "Lumen is analyzing...",
        "Lumen is examining the question...",
    }


def test_pending_indicator_is_inserted_and_removed(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())

        app._set_pending(True, prompt="tell me about propulsion")

        assert app.pending_row is not None
        pending_widgets = app.pending_row.winfo_children()[0].winfo_children()
        assert any(isinstance(widget, tk.Text) for widget in pending_widgets)
        assert any(
            str(widget.get("1.0", tk.END)).strip().endswith("...")
            for widget in pending_widgets
            if isinstance(widget, tk.Text)
        )

        app._clear_pending_indicator()

        assert app.pending_row is None
    finally:
        _destroy_app_root(root, app if "app" in locals() else None)


def test_desktop_app_surfaces_startup_warning_when_core_bundles_are_missing(monkeypatch) -> None:
    class _BrokenController(_FakeController):
        def list_tools(self) -> list[str]:
            return ["workspace"]

        def build_doctor_report(self) -> dict[str, object]:
            return {
                "checks": [
                    {"name": "tool_registry", "missing_bundles": ["report", "memory", "math", "system", "knowledge"]},
                    {"name": "runtime_resources", "missing_required_resources": ["tool_bundles"]},
                    {
                        "name": "runtime_layout",
                        "runtime_root": str(self.repo_root),
                        "data_root": str(self.settings.data_root),
                    },
                ]
            }

    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _BrokenController)
    monkeypatch.setenv("LUMEN_DEBUG_UI", "1")
    _install_immediate_ui_task_runner(monkeypatch)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        root.update()
        app._drain_queues_once()

        assert app.status_var.get() == "Startup issue detected"
        assert app.add_button.cget("state") == tk.DISABLED
        assert str(app.mode_selector.cget("state")) == "disabled"
        assert app.nav_buttons["recent"].cget("state") == tk.DISABLED
        assert app.nav_buttons["archived"].cget("state") == tk.DISABLED
        assert app.nav_buttons["memory"].cget("state") == tk.DISABLED
        assert app.nav_buttons["archived_memory"].cget("state") == tk.DISABLED
        assert any("Missing resources: tool_bundles" in message.text for message in app.messages)
        assert any("Runtime root:" in message.text and "Data root:" in message.text for message in app.messages)
    finally:
        _destroy_app_root(root, app if "app" in locals() else None)


def test_selecting_gated_view_leaves_current_view_stable(monkeypatch) -> None:
    class _BrokenController(_FakeController):
        list_personal_memory = None
        list_research_notes = None

        def build_doctor_report(self) -> dict[str, object]:
            return {
                "checks": [
                    {"name": "tool_registry", "missing_bundles": ["memory"]},
                    {"name": "runtime_resources", "missing_required_resources": []},
                    {
                        "name": "runtime_layout",
                        "runtime_root": str(self.repo_root),
                        "data_root": str(self.settings.data_root),
                    },
                ]
            }

    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _BrokenController)
    monkeypatch.setenv("LUMEN_DEBUG_UI", "1")
    _install_immediate_ui_task_runner(monkeypatch)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        root.update()
        app._drain_queues_once()
        current_view = app.current_view
        surfaced: list[str] = []
        monkeypatch.setattr(app, "_surface_runtime_failure", lambda message, **kwargs: surfaced.append(message))
        monkeypatch.setattr(app, "_refresh_memory_view", lambda: surfaced.append("refresh_memory"))

        app._show_view("memory")

        assert app.current_view == current_view
        assert surfaced
        assert surfaced[0] == "Memory tools are unavailable in this runtime right now."
        assert "refresh_memory" not in surfaced
    finally:
        _destroy_app_root(root, app if "app" in locals() else None)


def test_memory_refresh_shows_inline_unavailable_state_when_capability_gated(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        app._show_view("memory")
        app._desktop_capability_state = DesktopCapabilityState.from_runtime(
            missing_bundles=["memory"],
            missing_resources=[],
            capabilities={"workspace.inspect": {"tool_id": "workspace"}},
            surface_runtime_ready={"memory": False},
        )

        app._refresh_memory_view()

        preview = app.memory_preview.get("1.0", tk.END)
        labels = [child for child in app.memory_list_inner.winfo_children() if isinstance(child, tk.Label)]
        assert "Memory tools are unavailable" in preview
        assert labels
        assert "Memory tools are unavailable" in labels[0].cget("text")
    finally:
        _destroy_app_root(root, app if "app" in locals() else None)


def test_archived_memory_refresh_shows_inline_unavailable_state_when_capability_gated(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        app._show_view("archived_memory")
        app._desktop_capability_state = DesktopCapabilityState.from_runtime(
            missing_bundles=["memory"],
            missing_resources=[],
            capabilities={"workspace.inspect": {"tool_id": "workspace"}},
            surface_runtime_ready={"memory": False},
        )

        app._refresh_archived_memory_view()

        preview = app.archived_memory_preview.get("1.0", tk.END)
        labels = [child for child in app.archived_memory_list_inner.winfo_children() if isinstance(child, tk.Label)]
        assert "Memory tools are unavailable" in preview
        assert labels
        assert "Memory tools are unavailable" in labels[0].cget("text")
    finally:
        _destroy_app_root(root, app if "app" in locals() else None)


def test_archived_memory_top_bar_title_is_visible(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")

        app._show_view("archived_memory")

        assert app.screen_title_var.get() == "Archived Memory"
    finally:
        _destroy_app_root(root, app if "app" in locals() else None)


def test_memory_apply_replaces_loading_placeholder_with_rows(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        app._show_view("memory")
        app._render_memory_surface_state(
            archived=False,
            preview_text="Loading memory...",
            list_text="Loading memory...",
        )

        app._apply_memory_view_result(
            {
                "entries": [
                    {
                        "title": f"Memory {index}",
                        "content": f"Note {index}",
                        "created_at": "2026-03-23T12:00:00+00:00",
                        "kind": "personal_memory",
                        "entry_path": f"memory-{index}.json",
                    }
                    for index in range(4)
                ],
                "signature": tuple(
                    (f"Memory {index}", "2026-03-23T12:00:00+00:00", "personal_memory", f"memory-{index}.json")
                    for index in range(4)
                ),
                "has_more_available": False,
                "fetch_limit": 4,
                "fetch_reason": "bounded",
            }
        )

        labels = [child for child in app.memory_list_inner.winfo_children() if isinstance(child, tk.Label)]
        assert all("Loading memory..." not in label.cget("text") for label in labels)
        assert app.memory_rendered_count == 4
    finally:
        _destroy_app_root(root, app if "app" in locals() else None)


def test_archived_memory_apply_replaces_loading_placeholder_with_rows(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        app._show_view("archived_memory")
        app._render_memory_surface_state(
            archived=True,
            preview_text="Loading archived memory...",
            list_text="Loading archived memory...",
        )

        app._apply_archived_memory_view_result(
            {
                "entries": [
                    {
                        "title": f"Archived Memory {index}",
                        "content": f"Note {index}",
                        "created_at": "2026-03-23T12:00:00+00:00",
                        "kind": "personal_memory",
                        "entry_path": f"archived-memory-{index}.json",
                    }
                    for index in range(4)
                ],
                "signature": tuple(
                    (
                        f"Archived Memory {index}",
                        "2026-03-23T12:00:00+00:00",
                        "personal_memory",
                        f"archived-memory-{index}.json",
                    )
                    for index in range(4)
                ),
                "has_more_available": False,
                "fetch_limit": 4,
                "fetch_reason": "bounded",
            }
        )

        labels = [child for child in app.archived_memory_list_inner.winfo_children() if isinstance(child, tk.Label)]
        assert all("Loading archived memory..." not in label.cget("text") for label in labels)
        assert app.archived_memory_rendered_count == 4
    finally:
        _destroy_app_root(root, app if "app" in locals() else None)


def test_startup_health_reconciles_active_memory_surface_when_runtime_becomes_unavailable(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        app._show_view("memory")
        calls: list[str] = []
        monkeypatch.setattr(app, "_refresh_memory_view", lambda: calls.append("refresh_memory"))

        app._apply_startup_health_result(
            {
                "missing": ["memory"],
                "missing_resources": ["tool_bundles"],
                "runtime_root": str(tmp_path),
                "data_root": str(tmp_path / "data"),
                "execution_mode": "frozen",
                "debug_ui": False,
                "capabilities": {"workspace.inspect": {"tool_id": "workspace"}},
                "surface_runtime_ready": {"memory": True},
            }
        )

        assert app._desktop_capability_state.memory_surfaces_ready is False
        assert calls == ["refresh_memory"]
    finally:
        _destroy_app_root(root, app if "app" in locals() else None)


def test_booting_capability_state_suppresses_heavy_view_refresh(monkeypatch) -> None:
    app = object.__new__(LumenDesktopApp)
    app._desktop_capability_state = DesktopCapabilityState.booting()
    app.recent_sessions_view_dirty = True
    app._debug_ui_event_counts = {}
    app._debug_event = LumenDesktopApp._debug_event.__get__(app, LumenDesktopApp)
    app._view_capability_available = LumenDesktopApp._view_capability_available.__get__(app, LumenDesktopApp)

    result = app._refresh_recent_sessions_view()

    assert result is False
    assert app.recent_sessions_view_dirty is True
    assert app._debug_ui_event_counts["surface_capability_gated"] == 1


def test_desktop_app_hides_runtime_paths_during_healthy_startup(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("LUMEN_DEBUG_UI", raising=False)
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data", execution_mode="frozen")
        root.update()

        assert not any("Runtime root:" in message.text for message in app.messages)
        assert not any("Data root:" in message.text for message in app.messages)
        greeting = app.greeting_label.cget("text")
        assert greeting
        assert greeting.startswith(
            (
                "Good morning.",
                "Morning.",
                "Good afternoon.",
                "Afternoon.",
                "Good evening.",
                "Evening.",
                "Hello.",
                "Here and ready.",
                "Here when you're ready.",
            )
        )
    finally:
        _destroy_app_root(root, app if "app" in locals() else None)


def test_desktop_app_can_surface_runtime_paths_in_debug_ui_mode(monkeypatch) -> None:
    monkeypatch.setenv("LUMEN_DEBUG_UI", "1")
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    _install_immediate_ui_task_runner(monkeypatch)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=Path.cwd() / "data", execution_mode="frozen")
        root.update()
        app._drain_queues_once()

        assert any("Runtime root:" in message.text and "Data root:" in message.text for message in app.messages)
    finally:
        _destroy_app_root(root, app if "app" in locals() else None)


def test_debug_ui_events_write_to_log_file_in_debug_mode(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LUMEN_DEBUG_UI", "1")
    app = object.__new__(LumenDesktopApp)
    app._debug_ui_event_counts = {}
    app.debug_ui_log_path = tmp_path / "desktop_ui" / "ui_debug_20260416_224246_pid1234.log"
    app._debug_ui_log_failed = False
    app._emit_debug_ui_line = LumenDesktopApp._emit_debug_ui_line.__get__(app, LumenDesktopApp)
    app._debug_event = LumenDesktopApp._debug_event.__get__(app, LumenDesktopApp)

    app._debug_event("theme_change", theme="Light", current_view="chat")

    assert app.debug_ui_log_path.exists()
    payload = app.debug_ui_log_path.read_text(encoding="utf-8")
    assert "[lumen-ui] theme_change#1 theme=Light current_view=chat" in payload


def test_debug_trace_session_uses_unique_per_launch_log_paths(tmp_path: Path) -> None:
    first = DebugTraceSession.create(data_root=tmp_path, enabled=True)
    second = DebugTraceSession.create(data_root=tmp_path, enabled=True)

    assert first.log_path is not None
    assert second.log_path is not None
    assert first.log_path.name.startswith("ui_debug_")
    assert second.log_path.name.startswith("ui_debug_")
    assert first.log_path != second.log_path or first.session_label != second.session_label


def test_control_availability_resolves_state_without_style_decisions() -> None:
    availability = resolve_control_availability(
        shell_ready=True,
        pending=False,
        capability_phase="ready",
        chat_send_ready=True,
    )

    assert availability.chat_ready is True
    assert availability.nav_enabled is True
    assert availability.selector_state == "readonly"
    assert availability.chat_state == tk.NORMAL


def test_control_availability_keeps_hamburger_interactive_while_booting() -> None:
    availability = resolve_control_availability(
        shell_ready=True,
        pending=False,
        capability_phase="booting",
        chat_send_ready=False,
    )

    assert availability.nav_enabled is False
    assert availability.chat_state == tk.DISABLED
    assert availability.top_level_state == tk.NORMAL


def test_nav_visual_uses_muted_foreground_when_disabled() -> None:
    visual = resolve_nav_button_visual(
        name="memory",
        current_view="chat",
        hovered_nav="memory",
        enabled=False,
        palette=DARK_PALETTE,
        use_accented_hover=False,
    )

    assert visual["fg"] == DARK_PALETTE["text_muted"]
    assert visual["disabledforeground"] == DARK_PALETTE["text_muted"]


def test_input_and_composer_palette_helpers_preserve_lumen_purple_accent() -> None:
    input_palette = resolve_input_palette(palette=LUMEN_PURPLE_PALETTE, placeholder_active=False)
    button_palette = resolve_composer_button_palette(palette=LUMEN_PURPLE_PALETTE, primary=True)

    assert input_palette["bg"] == LUMEN_PURPLE_PALETTE["input_bg"]
    assert button_palette["fg"] == LUMEN_PURPLE_PALETTE["nav_active_border"]


def test_view_refresh_decision_separates_clear_hold_and_queue() -> None:
    assert resolve_view_refresh_decision(
        view_name="chat",
        view_enabled=True,
        hotbar_animation_active=False,
        hotbar_open=False,
    ).should_clear is True
    assert resolve_view_refresh_decision(
        view_name="memory",
        view_enabled=True,
        hotbar_animation_active=True,
        hotbar_open=False,
    ).should_hold_for_hotbar is True
    assert resolve_view_refresh_decision(
        view_name="recent",
        view_enabled=True,
        hotbar_animation_active=False,
        hotbar_open=False,
    ).should_queue is True


def test_startup_health_helper_builds_degraded_capability_state() -> None:
    state = capability_state_from_startup_health(
        {
            "missing": ["memory"],
            "missing_resources": ["tool_bundles"],
            "capabilities": {"workspace.inspect": {"tool_id": "workspace"}},
        }
    )

    assert state.phase == "degraded"
    assert state.memory_surfaces_ready is False


def test_startup_health_helper_keeps_memory_surfaces_ready_when_runtime_is_available() -> None:
    state = capability_state_from_startup_health(
        {
            "missing": ["memory"],
            "missing_resources": [],
            "capabilities": {"workspace.inspect": {"tool_id": "workspace"}},
            "surface_runtime_ready": {"memory": True},
        }
    )

    assert state.phase == "degraded"
    assert state.memory_surfaces_ready is True


def test_startup_snapshot_resolves_final_palette_once() -> None:
    snapshot = build_startup_snapshot(
        theme_name="Custom",
        custom_theme_name="Lumen Purple",
        custom_accent_color=None,
    )

    assert snapshot.theme_name == "custom"
    assert snapshot.palette["nav_active_border"] == LUMEN_PURPLE_PALETTE["nav_active_border"]


def test_optional_startup_followup_skips_diagnostics_and_background_tasks_by_default() -> None:
    app = object.__new__(LumenDesktopApp)
    calls: list[str] = []
    app._debug_event = lambda *args, **kwargs: calls.append(str(args[0]))
    app._schedule_startup_followup = lambda **kwargs: calls.append(f"schedule:{kwargs}")

    app._run_optional_startup_followup()

    assert calls == ["startup_followup_skipped"]


def test_bubble_wraplength_and_session_reset_stay_bounded(monkeypatch) -> None:
    class _FakeCanvas:
        @staticmethod
        def winfo_width() -> int:
            return 900

    class _FakeChild:
        def __init__(self) -> None:
            self.destroyed = False

        def destroy(self) -> None:
            self.destroyed = True

    class _FakeFrame:
        def __init__(self, children: list[_FakeChild]) -> None:
            self._children = children

        def winfo_children(self) -> list[_FakeChild]:
            return list(self._children)

    app = object.__new__(LumenDesktopApp)
    app.chat_canvas = _FakeCanvas()
    app.pending_row = object()
    app.message_labels = [object()]
    children = [_FakeChild(), _FakeChild()]
    app.chat_frame = _FakeFrame(children)

    wraplength = app._bubble_wraplength()
    padding = app._bubble_side_padding()

    assert 280 <= wraplength <= 620
    assert 48 <= padding <= 140

    app._set_chat_text("")

    assert all(child.destroyed for child in children)
    assert app.pending_row is None
    assert app.message_labels == []


def test_chat_layout_support_keeps_wrap_and_padding_bounded() -> None:
    assert 280 <= bubble_wraplength(0) <= 620
    assert 280 <= bubble_wraplength(1400) <= 620
    assert 48 <= bubble_side_padding(0) <= 140
    assert 48 <= bubble_side_padding(1400) <= 140


def test_theme_change_swaps_current_palette_without_touching_messages() -> None:
    app = object.__new__(LumenDesktopApp)
    app.theme_var = SimpleNamespace(get=lambda: "Light")
    app.custom_theme_var = SimpleNamespace(get=lambda: "Lumen Purple")
    app.custom_accent_color = None
    app.current_theme = dict(THEME_TOKENS["dark"])
    app.current_palette = dict(DARK_PALETTE)
    app.messages = []
    calls: list[str] = []
    app.root = SimpleNamespace(configure=lambda **kwargs: calls.append(str(kwargs.get("bg"))))
    app._configure_styles = lambda: calls.append("styles")
    app._apply_palette_to_shell = lambda **kwargs: calls.append(f"shell:{kwargs}")
    app._refresh_message_styles = lambda **kwargs: calls.append(f"messages:{kwargs}")
    app._persist_desktop_preferences_safe = lambda: calls.append("prefs")

    app._on_theme_changed()

    assert app.current_theme == THEME_TOKENS["light"]
    assert app.current_palette["app_bg"] == LIGHT_PALETTE["app_bg"]
    assert calls == [
        LIGHT_PALETTE["app_bg"],
        "styles",
        "shell:{'reflow_messages': False, 'include_assets': False, 'include_cache': False}",
        "prefs",
    ]
    assert app._debug_ui_event_counts["theme_change"] == 1
    assert app._debug_ui_event_counts["theme_apply_flush"] == 1


def test_initial_chat_view_commit_does_not_schedule_refresh(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    scheduled: list[str] = []
    original_schedule = LumenDesktopApp._schedule_view_refresh

    def _wrapped_schedule(self, view_name: str) -> None:
        scheduled.append(view_name)
        return original_schedule(self, view_name)

    monkeypatch.setattr(LumenDesktopApp, "_schedule_view_refresh", _wrapped_schedule)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        _ = LumenDesktopApp(root, repo_root=Path.cwd())
        assert scheduled == []
    finally:
        root.destroy()


def test_theme_change_writes_desktop_prefs_once_per_flush_and_not_pending() -> None:
    app = object.__new__(LumenDesktopApp)
    calls: list[str] = []
    app._debug_ui_event_counts = {}
    app.theme_var = SimpleNamespace(get=lambda: "Light")
    app.custom_theme_var = SimpleNamespace(get=lambda: "Lumen Purple")
    app.custom_accent_color = None
    app.current_theme = dict(THEME_TOKENS["dark"])
    app.current_palette = dict(DARK_PALETTE)
    app.current_view = "chat"
    app.pending = False
    app.deferred_view_refresh_job = None
    app.pending_hotbar_refresh_target = None
    app.root = SimpleNamespace(configure=lambda **kwargs: calls.append(str(kwargs.get("bg"))))
    app._configure_styles = lambda: calls.append("styles")
    app._apply_palette_to_shell = lambda **kwargs: calls.append(f"shell:{kwargs}")
    app._persist_desktop_preferences_safe = lambda: calls.append("prefs")
    app._show_pending_indicator = lambda *args, **kwargs: calls.append("pending")
    app._cancel_deferred_view_refresh = lambda: calls.append("cancel")
    app._schedule_theme_apply = LumenDesktopApp._schedule_theme_apply.__get__(app, LumenDesktopApp)
    app._stabilize_theme_transition_state = lambda: calls.append("stabilize")
    app._flush_theme_apply = LumenDesktopApp._flush_theme_apply.__get__(app, LumenDesktopApp)
    app._debug_event = LumenDesktopApp._debug_event.__get__(app, LumenDesktopApp)

    app._on_theme_changed()

    assert calls.count("prefs") == 1
    assert "pending" not in calls


def test_theme_change_does_not_force_identity_asset_reload(monkeypatch) -> None:
    app = object.__new__(LumenDesktopApp)
    calls: list[str] = []
    app._debug_ui_event_counts = {}
    app.theme_var = SimpleNamespace(get=lambda: "Light")
    app.custom_theme_var = SimpleNamespace(get=lambda: "Lumen Purple")
    app.custom_accent_color = None
    app.current_theme = dict(THEME_TOKENS["dark"])
    app.current_palette = dict(DARK_PALETTE)
    app.current_view = "chat"
    app.root = SimpleNamespace(configure=lambda **kwargs: calls.append(str(kwargs.get("bg"))))
    app._configure_styles = lambda: calls.append("styles")
    app._apply_palette_to_shell = lambda **kwargs: calls.append(f"shell:{kwargs}")
    app._persist_desktop_preferences_safe = lambda: calls.append("prefs")
    app._cancel_deferred_view_refresh = lambda: calls.append("cancel")
    app._schedule_theme_apply = LumenDesktopApp._schedule_theme_apply.__get__(app, LumenDesktopApp)
    app._stabilize_theme_transition_state = lambda: calls.append("stabilize")
    app._flush_theme_apply = LumenDesktopApp._flush_theme_apply.__get__(app, LumenDesktopApp)
    app._debug_event = LumenDesktopApp._debug_event.__get__(app, LumenDesktopApp)

    app._on_theme_changed()

    assert "shell:{'reflow_messages': False, 'include_assets': False, 'include_cache': False}" in calls


def test_resolve_theme_tokens_keeps_dark_light_and_purple_isolated() -> None:
    dark = resolve_theme_tokens("dark")
    light = resolve_theme_tokens("light")
    purple = resolve_theme_tokens("custom")

    assert dark == THEME_TOKENS["dark"]
    assert light == THEME_TOKENS["light"]
    assert purple == THEME_TOKENS["custom"]
    assert dark["background"] != light["background"]
    assert purple["background"] != light["background"]


def test_custom_accent_theme_derives_palette_without_mutating_base_themes() -> None:
    accent_theme = custom_accent_theme("#ab66ff")
    accent_palette = palette_from_theme(accent_theme)
    dark_accent = THEME_TOKENS["dark"]["accent"]

    assert accent_theme["accent"] != dark_accent
    assert THEME_TOKENS["custom"]["background"] != accent_theme["background"]
    assert accent_palette["nav_active_border"] == accent_theme["accent"]


def test_display_name_persists_to_desktop_preferences(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        app.display_name_var.set("Alek")
        app._confirm_display_name()

        prefs_path = tmp_path / "data" / "desktop_ui" / "preferences.json"
        assert prefs_path.exists()
        payload = json.loads(prefs_path.read_text(encoding="utf-8"))
        assert payload["display_name"] == "Alek"

        try:
            second_root = tk.Tk()
        except tk.TclError:
            pytest.skip("Tk could not be reinitialized in this test environment")
        second_root.withdraw()
        try:
            app2 = LumenDesktopApp(second_root, repo_root=Path.cwd(), data_root=tmp_path / "data")
            assert app2.display_name_var.get() == "Alek"
        finally:
            second_root.destroy()
    finally:
        _destroy_app_root(root, app if "app" in locals() else None)


def test_archived_sessions_view_lists_archived_chats(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    _install_immediate_ui_task_runner(monkeypatch)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        session = app.controller._recent_sessions[0]
        app.controller.archive_session(str(session["session_id"]))

        app._show_view("archived")
        root.update()
        app._drain_queues_once()

        rendered = [
            child for child in app.archived_list_inner.winfo_children()
            if isinstance(child, tk.Frame)
        ]
        assert rendered
    finally:
        _destroy_app_root(root, app if "app" in locals() else None)


def test_sidebar_views_are_lazy_loaded_at_startup(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    _install_immediate_ui_task_runner(monkeypatch)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")

        assert "memory" not in app.views
        assert "recent" not in app.views
        assert "settings" not in app.views
        assert app.controller.research_note_calls == 0
        assert app.controller.personal_memory_calls == 0
        assert app.controller.knowledge_overview_calls == 0
        assert app.controller.recent_session_calls == 0

        app._show_view("memory")
        root.update()
        app._drain_queues_once()

        assert "memory" in app.views
        assert app.controller.research_note_calls == 1
        assert app.controller.personal_memory_calls == 1
        assert app.controller.knowledge_overview_calls == 0
        assert app.controller.recent_session_calls == 0

        app._show_view("recent")
        root.update()
        app._drain_queues_once()

        assert "recent" in app.views
        assert app.controller.recent_session_calls == 1
    finally:
        _destroy_app_root(root, app if "app" in locals() else None)


def test_leaving_heavy_surface_tears_down_live_widgets_but_keeps_cached_data(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    _install_immediate_ui_task_runner(monkeypatch)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")

        app._show_view("memory")
        root.update()
        app._drain_queues_once()

        assert "memory" in app.views
        assert hasattr(app, "memory_list_inner")
        assert app.memory_entries

        cached_entries = list(app.memory_entries)

        app._show_view("chat")
        root.update()

        assert "memory" not in app.views
        assert not hasattr(app, "memory_list_inner")
        assert app.memory_entries == cached_entries

        app._show_view("memory")
        root.update()

        assert "memory" in app.views
        assert hasattr(app, "memory_list_inner")
        assert app.memory_entries == cached_entries
    finally:
        _destroy_app_root(root, app if "app" in locals() else None)


def test_context_bar_updates_with_mode_switch_without_loading_side_views(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")

        assert app.context_bar_var.get() == "Mode: Default • Task: Open Reasoning"

        app.mode_var.set("Collab")
        app._on_mode_changed()

        assert app.context_bar_var.get() == "Mode: Collab • Task: Open Reasoning"
        assert app.controller.research_note_calls == 0
        assert app.controller.recent_session_calls == 0
    finally:
        _destroy_app_root(root, app if "app" in locals() else None)


def test_mode_change_persists_session_profile_and_desktop_default(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        prefs: list[str] = []
        pending: list[str] = []
        monkeypatch.setattr(app, "_persist_desktop_preferences_safe", lambda: prefs.append("prefs"))
        monkeypatch.setattr(app, "_show_pending_indicator", lambda *args, **kwargs: pending.append("pending"))
        app.controller.profile_calls.clear()
        app._debug_ui_event_counts.clear()

        app.mode_var.set("Collab")
        app._on_mode_changed()

        assert app.controller.profile_calls == [(app.session_id, "collab")]
        assert prefs == ["prefs"]
        assert pending == []
        assert app._debug_ui_event_counts["mode_change"] == 1
        assert app._debug_ui_event_counts["apply_mode_to_session"] == 1
    finally:
        _destroy_app_root(root, app if "app" in locals() else None)


def test_mode_change_off_chat_does_not_refresh_chat_landing(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        app.current_view = "recent"
        landing_updates: list[str] = []
        monkeypatch.setattr(app, "_update_landing_state", lambda: landing_updates.append("landing"))

        app.mode_var.set("Collab")
        app._on_mode_changed()

        assert landing_updates == []
    finally:
        _destroy_app_root(root, app if "app" in locals() else None)


def test_nav_button_style_distinguishes_active_hover_and_idle_states() -> None:
    active = nav_button_style(active=True, hovered=False, palette=DARK_PALETTE)
    hovered = nav_button_style(active=False, hovered=True, palette=DARK_PALETTE)
    idle = nav_button_style(active=False, hovered=False, palette=DARK_PALETTE)

    assert active["bg"] == DARK_PALETTE["nav_active_bg"]
    assert active["highlightbackground"] == DARK_PALETTE["nav_active_border"]
    assert hovered["bg"] == DARK_PALETTE["nav_hover_bg"]
    assert hovered["fg"] == DARK_PALETTE["text_primary"]
    assert idle["bg"] == DARK_PALETTE["sidebar_bg"]
    assert idle["fg"] == DARK_PALETTE["text_secondary"]


def test_dark_family_hover_helper_only_targets_dark_and_custom() -> None:
    app = object.__new__(LumenDesktopApp)
    app.theme_var = SimpleNamespace(get=lambda: "Dark")
    app.current_palette = dict(DARK_PALETTE)
    assert app._use_accented_dark_family_hover() is True
    assert app._browser_row_hover_bg() == DARK_PALETTE["nav_active_bg"]

    app.theme_var = SimpleNamespace(get=lambda: "Custom")
    app.current_palette = dict(LUMEN_PURPLE_PALETTE)
    assert app._use_accented_dark_family_hover() is True
    assert app._browser_row_hover_bg() == LUMEN_PURPLE_PALETTE["nav_active_bg"]

    app.theme_var = SimpleNamespace(get=lambda: "Light")
    app.current_palette = dict(LIGHT_PALETTE)
    assert app._use_accented_dark_family_hover() is False
    assert app._browser_row_hover_bg() == LIGHT_PALETTE["list_hover_bg"]


def test_day_grouping_helpers_produce_human_buckets_and_labels() -> None:
    now = datetime.fromisoformat("2026-03-26T15:00:00+00:00")

    assert day_group_label("2026-03-26T12:00:00+00:00", now=now) == "Today"
    assert day_group_label("2026-03-25T12:00:00+00:00", now=now) == "Yesterday"
    assert day_group_label("2026-03-24T12:00:00+00:00", now=now) == "Earlier This Week"
    assert human_date_label("2026-03-26T12:00:00+00:00", now=now) == "12:00 PM"


def test_grouped_session_rows_inserts_day_headers() -> None:
    now = datetime.fromisoformat("2026-03-23T15:00:00+00:00")
    rows = grouped_session_rows(
        [
            {"session_id": "a", "prompt": "today prompt", "created_at": "2026-03-23T12:00:00+00:00", "mode": "research"},
            {"session_id": "b", "prompt": "yesterday prompt", "created_at": "2026-03-22T12:00:00+00:00", "mode": "default"},
        ],
        now=now,
    )

    assert rows[0]["kind"] == "header"
    assert rows[0]["label"] == "Today"
    assert rows[1]["kind"] == "session"
    assert rows[2]["label"] == "Yesterday"


def test_empty_state_copy_is_intentional_for_memory_and_recent_views() -> None:
    memory = empty_state_text("memory")
    recent = empty_state_text("recent")

    assert "Saved memory" in memory
    assert "research results" in memory
    assert "Completed chat sessions" in recent
    assert "open it" in recent.lower()


def test_transcript_and_recent_preview_use_selectable_text_widgets(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        message = build_pending_message(key="math prompt")
        app._append_chat_message(message)
        app._show_view("recent")

        assert app.message_text_widgets
        transcript_widget = app.message_text_widgets[-1]
        assert isinstance(transcript_widget, tk.Text)
        assert str(transcript_widget.cget("state")) == tk.DISABLED
        assert hasattr(app, "recent_list_canvas")
        assert isinstance(app.recent_list_canvas, tk.Canvas)
    finally:
        _destroy_app_root(root, app if "app" in locals() else None)


def test_placeholder_clears_when_using_starter_prompt(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())

        assert app.placeholder_active is True

        app.starter_prompt_var.set("What is entropy, simply but correctly?")
        app._use_starter_prompt()

        assert app.placeholder_active is False
        assert app._input_value() == "What is entropy, simply but correctly?"
        assert app.controller.ask_calls == []
    finally:
        _destroy_app_root(root, app if "app" in locals() else None)


def test_starter_prompt_submission_respects_current_mode(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    _install_immediate_ui_task_runner(monkeypatch)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        app.mode_var.set("Direct")
        app._on_mode_changed()

        app._activate_starter_prompt("Compare a black hole and a neutron star clearly")
        app._send_message()
        app._drain_queues_once()

        assert app.mode_var.get() == "Direct"
        assert app.controller.profile_calls[-1] == (app.session_id, "direct")
        assert app.messages[-1].text == "Full answer for: Compare a black hole and a neutron star clearly"
    finally:
        _destroy_app_root(root, app if "app" in locals() else None)


def test_attachment_file_selection_updates_status_and_send_passes_input_path(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    _install_immediate_ui_task_runner(monkeypatch)
    selected = tmp_path / "example.zip"
    selected.write_text("zip placeholder")
    monkeypatch.setattr("lumen.desktop.chat_app.filedialog.askopenfilename", lambda **kwargs: str(selected))
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())

        app._select_attachment_file()

        assert app.attached_input_path == selected
        assert "Zip:" in app.attachment_var.get()

        app._hide_input_placeholder()
        app.input_box.insert("1.0", "Run this")
        app._send_message()
        app._drain_queues_once()

        assert app.controller.ask_calls[-1]["input_path"] == selected
        assert app.attached_input_path is None
        assert app.attachment_var.get() == "No file, folder, or zip selected."
    finally:
        root.destroy()


def test_attachment_folder_selection_updates_status(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    selected = tmp_path / "sample-folder"
    selected.mkdir()
    monkeypatch.setattr("lumen.desktop.chat_app.filedialog.askdirectory", lambda **kwargs: str(selected))
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())

        app._select_attachment_folder()

        assert app.attached_input_path == selected
        assert app.attached_input_kind == "folder"
        assert "Folder:" in app.attachment_var.get()
        assert str(app.clear_attachment_button.cget("state")) == tk.NORMAL
    finally:
        root.destroy()


def test_new_session_clears_unsent_attachment(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    selected = tmp_path / "input.txt"
    selected.write_text("hello")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        app._set_attachment(selected, kind="file")

        assert app.attached_input_path == selected

        app._start_new_session()

        assert app.attached_input_path is None
        assert app.attachment_var.get() == "No file, folder, or zip selected."
    finally:
        root.destroy()


def test_stop_button_appears_only_while_pending_and_reenables_input(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    monkeypatch.setattr("lumen.desktop.chat_app.messagebox.askyesno", lambda *args, **kwargs: True)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())

        assert app.stop_button.winfo_manager() == ""

        app._set_pending(True, prompt="tell me about entropy")

        assert app.stop_button.winfo_manager() == "grid"
        assert str(app.stop_button.cget("state")) == tk.NORMAL

        app.pending_request_id = 7
        app._stop_current_task()

        assert app.pending is False
        assert app.pending_request_id == 0
        assert 7 in app.ignored_request_ids
        assert app.stop_button.winfo_manager() == ""
        assert str(app.send_button.cget("state")) == tk.NORMAL
        assert app.status_var.get() == "Stopped. Waiting for your next direction."
    finally:
        root.destroy()


def test_stopped_request_result_is_ignored_when_it_arrives_late(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    monkeypatch.setattr("lumen.desktop.chat_app.Thread", _DeferredThread)
    monkeypatch.setattr("lumen.desktop.chat_app.messagebox.askyesno", lambda *args, **kwargs: True)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        starting_messages = len(app.messages)

        app._hide_input_placeholder()
        app.input_box.insert("1.0", "Explain entropy.")
        app._send_message()

        assert app.pending_request_id == 1
        assert app.pending is True

        app._stop_current_task()
        app.result_queue.put(
            (
                "response",
                1,
                {
                    "mode": "research",
                    "summary": "Late response should not appear.",
                    "reply": "Late response should not appear.",
                    "user_facing_answer": "Late response should not appear.",
                },
            )
        )
        app._drain_queues_once()

        assert len(app.messages) == starting_messages + 1
        assert any(message.sender == "You" and message.text == "Explain entropy." for message in app.messages)
        assert all(message.message_type != "pending" for message in app.messages)
        assert all("Late response should not appear." not in message.text for message in app.messages)
        assert app.pending is False
        assert app.pending_request_id == 0
        assert 1 not in app.ignored_request_ids
    finally:
        root.destroy()


def test_light_theme_uses_palette_borders_on_primary_surfaces(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        app.theme_var.set("Light")
        app._on_theme_changed()

        assert app.chat_surface.cget("highlightbackground") == LIGHT_PALETTE["app_bg"]
        assert app.chat_surface.cget("highlightcolor") == LIGHT_PALETTE["app_bg"]
        assert app.input_frame.cget("highlightbackground") == LIGHT_PALETTE["app_bg"]
        assert app.input_frame.cget("highlightcolor") == LIGHT_PALETTE["app_bg"]
    finally:
        root.destroy()


def test_placeholder_clears_on_click_and_stays_hidden_while_focused(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())

        assert app.placeholder_active is True

        app._on_input_pointer_down()

        assert app.placeholder_active is False
        assert app._input_value() == ""

        app.input_box.insert("1.0", "Hello")
        app.input_box.delete("1.0", tk.END)
        app._on_input_key_release()

        assert app.placeholder_active is False
        assert app._input_value() == ""
    finally:
        root.destroy()


def test_backspace_from_placeholder_clears_hint_without_sticking(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())

        outcome = app._on_input_key_press(SimpleNamespace(keysym="BackSpace"))

        assert outcome == "break"
        assert app.placeholder_active is False
        assert app._input_value() == ""
    finally:
        root.destroy()


def test_selected_text_returns_full_selection(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        widget = tk.Text(root)
        widget.insert("1.0", "Hello world")
        widget.tag_add(tk.SEL, "1.0", "1.5")

        assert app._selected_text(widget) == "Hello"
    finally:
        root.destroy()


def test_copy_selected_text_uses_full_selection(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        widget = tk.Text(root)
        widget.insert("1.0", "Hello world")
        widget.tag_add(tk.SEL, "1.0", "1.11")

        app._copy_selected_text(widget)

        assert root.clipboard_get() == "Hello world"
    finally:
        root.destroy()


def test_copy_selected_text_falls_back_to_full_disabled_widget_text(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        widget = tk.Text(root)
        widget.insert("1.0", "Transcript body")
        widget.configure(state=tk.DISABLED)

        app._copy_selected_text(widget)

        assert root.clipboard_get() == "Transcript body"
    finally:
        root.destroy()


def test_paste_text_inserts_clipboard_contents(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        widget = tk.Text(root)
        root.clipboard_clear()
        root.clipboard_append("Hello paste")

        app._paste_text(widget)

        assert widget.get("1.0", tk.END).strip() == "Hello paste"
    finally:
        root.destroy()


def test_read_clipboard_falls_back_to_selection_when_clipboard_get_fails(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())

        monkeypatch.setattr(root, "clipboard_get", lambda: (_ for _ in ()).throw(tk.TclError("no clipboard")))
        monkeypatch.setattr(root, "selection_get", lambda **kwargs: "Selection fallback")

        assert app._read_clipboard() == "Selection fallback"
    finally:
        root.destroy()


def test_recent_sessions_view_groups_rows_by_day(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    _install_immediate_ui_task_runner(monkeypatch)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        app._show_view("recent")
        root.update()
        app._drain_queues_once()

        rows = app.recent_sessions_rows
        assert rows
        assert rows[0]["kind"] == "header"
        assert rows[0]["label"] in {"Today", "Yesterday", "Earlier This Week", "Older"}
        assert any(row["kind"] == "session" for row in rows)
    finally:
        root.destroy()


def test_loading_saved_chat_recovers_when_one_record_cannot_render(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        original_helper = app._saved_chat_assistant_text

        def _broken_saved_text(record, *, response):
            if str(record.get("prompt") or "") == "tell me about black holes":
                raise RuntimeError("bad saved record")
            return original_helper(record, response=response)

        monkeypatch.setattr(app, "_saved_chat_assistant_text", _broken_saved_text)

        app._load_session("desktop-1")

        assert app.status_var.get() == "Loaded desktop-1"
        assert any(
            "could not be rendered" in message.text.lower()
            for message in app.messages
            if message.message_type == "system"
        )
        assert app.current_view == "chat"
    finally:
        root.destroy()


def test_memory_view_includes_date_grouped_sections(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    _install_immediate_ui_task_runner(monkeypatch)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        app._show_view("memory")
        root.update()
        app._drain_queues_once()
        descendants = list(app.memory_list_inner.winfo_children())
        labels: list[tk.Label] = []
        while descendants:
            widget = descendants.pop()
            if isinstance(widget, tk.Label):
                labels.append(widget)
            descendants.extend(widget.winfo_children())
        row_texts = [
            widget.cget("text")
            for widget in labels
        ]
        preview = app.memory_preview.get("1.0", tk.END)
        assert any("Preference" in text or "Project note" in text for text in row_texts)
        assert "Select a memory" in preview
    finally:
        root.destroy()


def test_memory_view_mounts_with_loading_placeholder_before_fetch(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        app.memory_entries = []
        app.memory_view_dirty = True

        app._show_view("memory")

        preview = app.memory_preview.get("1.0", tk.END)
        labels = [child for child in app.memory_list_inner.winfo_children() if isinstance(child, tk.Label)]
        assert "Loading memory" in preview
        assert labels
        assert "Loading memory" in labels[0].cget("text")
    finally:
        root.destroy()


def test_archived_memory_view_lists_archived_entries(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    _install_immediate_ui_task_runner(monkeypatch)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        app.controller.archive_memory(
            kind="personal_memory",
            path=str(app.controller.settings.data_root / "personal_memory" / "desktop-1" / "pref.json"),
        )

        app._show_view("archived_memory")
        root.update()
        app._drain_queues_once()

        descendants = list(app.archived_memory_list_inner.winfo_children())
        labels: list[tk.Label] = []
        while descendants:
            widget = descendants.pop()
            if isinstance(widget, tk.Label):
                labels.append(widget)
            descendants.extend(widget.winfo_children())
        row_texts = [widget.cget("text") for widget in labels]
        assert any("Preference" in text for text in row_texts)
    finally:
        root.destroy()


def test_archived_memory_view_mounts_with_loading_placeholder_before_fetch(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        app.archived_memory_entries = []
        app.archived_memory_view_dirty = True

        app._show_view("archived_memory")

        preview = app.archived_memory_preview.get("1.0", tk.END)
        labels = [child for child in app.archived_memory_list_inner.winfo_children() if isinstance(child, tk.Label)]
        assert "Loading archived memory" in preview
        assert labels
        assert "Loading archived memory" in labels[0].cget("text")
    finally:
        root.destroy()


def test_archived_memory_view_renders_from_cache_before_refresh(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        app._show_view("archived_memory")
        app.archived_memory_entries = [
            {
                "title": "Preference",
                "created_at": "2026-03-23T12:00:00+00:00",
                "kind": "personal_memory",
                "entry_path": "pref.json",
                "content": "Remember this.",
            }
        ]
        app.archived_memory_render_signature = app._memory_entries_signature(app.archived_memory_entries)
        rendered: list[str] = []
        monkeypatch.setattr(app, "_start_ui_background_task", lambda *args, **kwargs: 1)
        monkeypatch.setattr(app, "_rebuild_memory_rows", lambda **kwargs: rendered.append("rows"))

        app._refresh_archived_memory_view()

        assert rendered == ["rows"]
    finally:
        root.destroy()


def test_tk_callback_exception_writes_desktop_crash_record(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        surfaced: list[str] = []
        monkeypatch.setattr(app, "_surface_runtime_failure", lambda message, **kwargs: surfaced.append(message))

        try:
            raise RuntimeError("callback boom")
        except RuntimeError as exc:
            app._report_tk_callback_exception(RuntimeError, exc, exc.__traceback__)

        records = read_crash_records(app.desktop_crash_log_path)
        assert records
        assert records[-1]["source"] == "tk_callback"
        assert records[-1]["exception_type"] == "RuntimeError"
        assert surfaced
    finally:
        root.destroy()


def test_surface_runtime_failure_logs_silently_to_text_file(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        visible: list[str] = []
        monkeypatch.setattr(app, "_append_system_line", lambda message: visible.append(message))

        app._surface_runtime_failure(
            "Lumen couldn't refresh archived chats.",
            source="ui_task.archived",
            category="refresh_failure",
            context={"surface": "archived", "reason": "test"},
        )

        assert visible == []
        assert app.status_var.get() == "Lumen couldn't refresh archived chats."
        log_text = app.desktop_runtime_failure_log_path.read_text(encoding="utf-8")
        assert "source: ui_task.archived" in log_text
        assert "category: refresh_failure" in log_text
        assert "message: Lumen couldn't refresh archived chats." in log_text
        assert "context.surface: archived" in log_text
        assert "context.reason: test" in log_text
    finally:
        root.destroy()


def test_capability_gated_surface_failure_logs_distinct_category(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        monkeypatch.setattr(app, "_view_capability_available", lambda view_name: False)
        monkeypatch.setattr(app, "_view_capability_reason", lambda view_name: f"{view_name} unavailable")
        monkeypatch.setattr(app, "_clear_hotbar_pending_state", lambda **kwargs: None)
        monkeypatch.setattr(app, "_cancel_deferred_view_refresh", lambda: None)

        result = app._gate_view_activation("memory")

        assert result is False
        log_text = app.desktop_runtime_failure_log_path.read_text(encoding="utf-8")
        assert "source: view_gate.memory" in log_text
        assert "category: capability_gated" in log_text
        assert "message: memory unavailable" in log_text
    finally:
        root.destroy()


def test_archived_memory_reentry_does_not_start_duplicate_refresh(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        app._show_view("archived_memory")
        app.archived_memory_entries = [
            {
                "title": "Preference",
                "created_at": "2026-03-23T12:00:00+00:00",
                "kind": "personal_memory",
                "entry_path": "pref.json",
                "content": "Remember this.",
            }
        ]
        signature = app._memory_entries_signature(app.archived_memory_entries)
        app.archived_memory_render_signature = signature
        app.archived_memory_cached_signature = signature
        start_calls: list[str] = []

        monkeypatch.setattr(app, "_start_ui_background_task", lambda *args, **kwargs: start_calls.append("start") or 1)

        app._refresh_archived_memory_view()
        app._refresh_archived_memory_view()

        assert start_calls == ["start"]
    finally:
        root.destroy()


def test_archived_sessions_view_renders_from_cache_before_refresh(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        app._show_view("archived")
        session = {
            "session_id": "arch-1",
            "title": "Archived chat",
            "prompt": "Tell me more",
            "mode": "research",
            "created_at": "2026-03-23T12:00:00+00:00",
            "archived": True,
        }
        app.archived_sessions_cache = [session]
        app.archived_sessions_rows = grouped_session_rows([session])
        signature = app._session_signature([session])
        app.archived_sessions_signature = signature
        app.archived_sessions_render_signature = ()
        rendered: list[str] = []
        monkeypatch.setattr(app, "_start_ui_background_task", lambda *args, **kwargs: 1)
        monkeypatch.setattr(app, "_render_archived_sessions_from_cache", lambda: rendered.append("render"))

        app._refresh_archived_sessions_view()

        assert rendered == ["render"]
    finally:
        root.destroy()


def test_archived_sessions_reentry_does_not_start_duplicate_refresh(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        app._show_view("archived")
        session = {
            "session_id": "arch-1",
            "title": "Archived chat",
            "prompt": "Tell me more",
            "mode": "research",
            "created_at": "2026-03-23T12:00:00+00:00",
            "archived": True,
        }
        app.archived_sessions_cache = [session]
        app.archived_sessions_rows = grouped_session_rows([session])
        signature = app._session_signature([session])
        app.archived_sessions_signature = signature
        app.archived_sessions_render_signature = signature
        start_calls: list[str] = []

        monkeypatch.setattr(app, "_start_ui_background_task", lambda *args, **kwargs: start_calls.append("start") or 1)

        app._refresh_archived_sessions_view()
        app._refresh_archived_sessions_view()

        assert start_calls == []
    finally:
        root.destroy()


def test_archived_sessions_deferred_refresh_skips_when_same_refresh_in_flight(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        app.current_view = "archived"
        app.deferred_view_refresh_target = "archived"
        app.archived_sessions_view_dirty = True
        app.archived_sessions_fetch_in_flight = True
        app.archived_sessions_requested_signature = app.archived_sessions_signature
        calls: list[str] = []

        monkeypatch.setattr(app, "_apply_palette_to_view", lambda view_name: calls.append(f"palette:{view_name}"))
        monkeypatch.setattr(app, "_start_ui_background_task", lambda *args, **kwargs: calls.append("start") or 1)

        app._run_deferred_view_refresh()

        assert calls == ["palette:archived"]
    finally:
        root.destroy()


def test_archived_sessions_refresh_suppresses_duplicate_in_flight_request(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        app._show_view("archived")
        session = {
            "session_id": "arch-1",
            "title": "Archived chat",
            "prompt": "Tell me more",
            "mode": "research",
            "created_at": "2026-03-23T12:00:00+00:00",
            "archived": True,
        }
        app.archived_sessions_cache = [session]
        app.archived_sessions_rows = grouped_session_rows([session])
        app.archived_sessions_signature = ()
        app.archived_sessions_render_signature = ()
        app.archived_sessions_fetch_in_flight = True
        app.archived_sessions_requested_signature = ()
        start_calls: list[str] = []

        monkeypatch.setattr(app, "_start_ui_background_task", lambda *args, **kwargs: start_calls.append("start") or 1)

        app._refresh_archived_sessions_view()

        assert start_calls == []
    finally:
        root.destroy()


def test_archived_sessions_apply_after_surface_teardown_drops_ui_work(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        app._show_view("archived")
        session = {
            "session_id": "arch-1",
            "title": "Archived chat",
            "prompt": "Tell me more",
            "mode": "research",
            "created_at": "2026-03-23T12:00:00+00:00",
            "archived": True,
        }
        app._teardown_heavy_surface("archived")
        app.current_view = "archived"
        calls: list[str] = []
        monkeypatch.setattr(app, "_render_archived_sessions_from_cache", lambda: calls.append("render"))

        app._apply_archived_sessions_result(
            {
                "available": True,
                "sessions": [session],
                "signature": app._session_signature([session]),
                "rows": grouped_session_rows([session]),
            }
        )

        assert calls == []
        assert app.archived_sessions_cache == [session]
        assert app.archived_sessions_view_dirty is True
        log_text = app.desktop_runtime_failure_log_path.read_text(encoding="utf-8")
        assert "source: archived.apply" in log_text
        assert "category: stale_result_drop" in log_text
        assert "context.reason: surface_missing" in log_text
    finally:
        root.destroy()


def test_archived_sessions_remount_renders_cached_rows_without_refetch(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        app._show_view("archived")
        sessions = [
            {
                "session_id": f"arch-{index}",
                "title": f"Archived chat {index}",
                "prompt": "Tell me more",
                "mode": "research",
                "created_at": "2026-03-23T12:00:00+00:00",
                "archived": True,
            }
            for index in range(12)
        ]
        app.archived_sessions_cache = sessions
        app.archived_sessions_rows = grouped_session_rows(sessions)
        signature = app._session_signature(sessions)
        app.archived_sessions_signature = signature
        app.archived_sessions_render_signature = signature
        app.archived_sessions_render_limit = 8
        app.archived_sessions_rendered_count = 8

        app._show_view("chat")
        root.update()
        app._drain_queues_once()

        start_calls: list[str] = []
        monkeypatch.setattr(app, "_start_ui_background_task", lambda *args, **kwargs: start_calls.append("start") or 1)

        app._show_view("archived")
        root.update()
        app._drain_queues_once()

        assert hasattr(app, "archived_list_inner")
        assert app.archived_list_inner.winfo_children()
        assert app.archived_sessions_render_limit == 4
        assert app.archived_sessions_rendered_count == 4
        assert app.archived_sessions_load_more_button is not None
        assert start_calls == []
    finally:
        root.destroy()

def test_archived_memory_deferred_refresh_skips_when_same_refresh_in_flight(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        app.current_view = "archived_memory"
        app.deferred_view_refresh_target = "archived_memory"
        app.archived_memory_view_dirty = True
        app.archived_memory_fetch_in_flight = True
        app.archived_memory_requested_version = app.archived_memory_state_version
        calls: list[str] = []

        monkeypatch.setattr(app, "_apply_palette_to_view", lambda view_name: calls.append(f"palette:{view_name}"))
        monkeypatch.setattr(app, "_start_ui_background_task", lambda *args, **kwargs: calls.append("start") or 1)

        app._run_deferred_view_refresh()

        assert calls == []
    finally:
        root.destroy()


def test_archived_memory_apply_failure_clears_in_flight_and_logs(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        app.current_view = "archived_memory"
        app.archived_memory_fetch_in_flight = True
        app.archived_memory_requested_version = 3
        app.archived_memory_entries = [
            {
                "title": "Preference",
                "created_at": "2026-03-23T12:00:00+00:00",
                "kind": "personal_memory",
                "entry_path": "pref.json",
                "content": "Remember this.",
            }
        ]
        app.archived_memory_rendered_count = 1
        surfaced: list[str] = []
        monkeypatch.setattr(app, "_surface_runtime_failure", lambda message, **kwargs: surfaced.append(message))
        monkeypatch.setattr(
            app,
            "_render_archived_memory_from_cache",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("render fail")),
        )

        app._apply_archived_memory_view_result({"entries": [], "signature": ()})

        assert app.archived_memory_fetch_in_flight is False
        assert app.archived_memory_requested_version is None
        assert app.archived_memory_view_dirty is True
        assert surfaced
        records = read_crash_records(app.desktop_crash_log_path)
        assert records[-1]["source"] == "archived_memory.apply"
        assert records[-1]["context"]["surface"] == "archived_memory"
        assert records[-1]["context"]["rendered_count"] == 1
    finally:
        root.destroy()


def test_delete_memory_failure_surfaces_status_and_logs(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    monkeypatch.setattr("lumen.desktop.chat_app.messagebox.askyesno", lambda *args, **kwargs: True)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        surfaced: list[str] = []
        monkeypatch.setattr(app, "_surface_runtime_failure", lambda message, **kwargs: surfaced.append(message))
        monkeypatch.setattr(app.controller, "delete_memory", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("delete fail")))

        app._delete_memory_entry({"kind": "personal_memory", "entry_path": "pref.json"})

        assert surfaced == ["Lumen couldn't delete that memory item."]
        records = read_crash_records(app.desktop_crash_log_path)
        assert records[-1]["source"] == "memory.delete_action"
    finally:
        root.destroy()


def test_recent_view_reentry_does_not_start_duplicate_refresh(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        app._show_view("recent")
        app.recent_sessions_cache = [
            {
                "session_id": "desktop-1",
                "title": "Chat",
                "prompt": "hello",
                "mode": "conversation",
                "created_at": "2026-03-23T12:00:00+00:00",
            }
        ]
        app.recent_sessions_rows = grouped_session_rows(app.recent_sessions_cache)
        signature = app._session_signature(app.recent_sessions_cache)
        app.recent_sessions_signature = signature
        app.recent_sessions_render_signature = signature
        start_calls: list[str] = []

        monkeypatch.setattr(app, "_start_ui_background_task", lambda *args, **kwargs: start_calls.append("start") or 1)

        app._refresh_recent_sessions_view()
        app._refresh_recent_sessions_view()

        assert start_calls == []
    finally:
        root.destroy()


def test_recent_deferred_refresh_skips_when_cache_is_current(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        app.current_view = "recent"
        app.deferred_view_refresh_target = "recent"
        app.recent_sessions_view_dirty = True
        app.recent_sessions_cache = [
            {
                "session_id": "desktop-1",
                "title": "Chat",
                "prompt": "hello",
                "mode": "conversation",
                "created_at": "2026-03-23T12:00:00+00:00",
            }
        ]
        app.recent_sessions_rows = grouped_session_rows(app.recent_sessions_cache)
        signature = app._session_signature(app.recent_sessions_cache)
        app.recent_sessions_signature = signature
        app.recent_sessions_render_signature = signature
        calls: list[str] = []

        monkeypatch.setattr(app, "_apply_palette_to_view", lambda view_name: calls.append(f"palette:{view_name}"))
        monkeypatch.setattr(app, "_start_ui_background_task", lambda *args, **kwargs: calls.append("start") or 1)

        app._run_deferred_view_refresh()

        assert calls == []
    finally:
        root.destroy()


def test_session_row_descriptors_include_session_identity_for_visually_identical_rows(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        sessions = [
            {
                "session_id": "desktop-1",
                "title": "Same Chat",
                "summary": "same summary",
                "prompt": "",
                "mode": "research",
                "created_at": "2026-03-23T12:00:00+00:00",
            },
            {
                "session_id": "desktop-2",
                "title": "Same Chat",
                "summary": "same summary",
                "prompt": "",
                "mode": "research",
                "created_at": "2026-03-23T12:00:00+00:00",
            },
        ]

        descriptors = app._build_session_row_descriptors(
            rows=grouped_session_rows(sessions),
            target_session_count=2,
            already_rendered=0,
        )

        session_descriptors = [descriptor for descriptor in descriptors if descriptor[0] == "session"]
        assert session_descriptors[0][1] == "desktop-1"
        assert session_descriptors[1][1] == "desktop-2"
        assert session_descriptors[0] != session_descriptors[1]
    finally:
        root.destroy()


def test_recent_reused_rows_keep_distinct_session_targets_for_identical_labels(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        app._show_view("recent")
        sessions = [
            {
                "session_id": "desktop-1",
                "title": "Same Chat",
                "summary": "same summary",
                "prompt": "",
                "mode": "research",
                "created_at": "2026-03-23T12:00:00+00:00",
            },
            {
                "session_id": "desktop-2",
                "title": "Same Chat",
                "summary": "same summary",
                "prompt": "",
                "mode": "research",
                "created_at": "2026-03-23T12:00:00+00:00",
            },
        ]
        app.recent_sessions_cache = sessions
        app.recent_sessions_rows = grouped_session_rows(sessions)
        app.recent_sessions_signature = app._session_signature(sessions)

        opened: list[str] = []
        monkeypatch.setattr(app, "_load_session", lambda session_id: opened.append(str(session_id)))

        app._render_recent_sessions_from_cache()
        app.recent_sessions_rendered_count = 1
        monkeypatch.setattr(app, "_browser_rows_match_descriptors", lambda widgets, descriptors: True)
        app._render_recent_sessions_from_cache()

        session_rows = [
            widget
            for widget in app.recent_list_inner.winfo_children()
            if getattr(widget, "_browser_descriptor", ("",))[0] == "session"
        ]
        assert len(session_rows) == 2
        app._invoke_browser_row_command(session_rows[0])
        app._invoke_browser_row_command(session_rows[1])

        assert opened == ["desktop-1", "desktop-2"]
        assert getattr(session_rows[0], "_browser_command_target_id", None) == "desktop-1"
        assert getattr(session_rows[1], "_browser_command_target_id", None) == "desktop-2"
    finally:
        _destroy_app_root(root, app if "app" in locals() else None)


def test_archived_reused_rows_keep_distinct_session_targets_for_identical_labels(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        app._show_view("archived")
        sessions = [
            {
                "session_id": "archived-1",
                "title": "Same Chat",
                "summary": "same summary",
                "prompt": "",
                "mode": "research",
                "created_at": "2026-03-23T12:00:00+00:00",
            },
            {
                "session_id": "archived-2",
                "title": "Same Chat",
                "summary": "same summary",
                "prompt": "",
                "mode": "research",
                "created_at": "2026-03-23T12:00:00+00:00",
            },
        ]
        app.archived_sessions_cache = sessions
        app.archived_sessions_rows = grouped_session_rows(sessions)
        app.archived_sessions_signature = app._session_signature(sessions)

        opened: list[str] = []
        monkeypatch.setattr(app, "_load_session", lambda session_id: opened.append(str(session_id)))

        app._render_archived_sessions_from_cache()
        app.archived_sessions_rendered_count = 1
        monkeypatch.setattr(app, "_browser_rows_match_descriptors", lambda widgets, descriptors: True)
        app._render_archived_sessions_from_cache()

        session_rows = [
            widget
            for widget in app.archived_list_inner.winfo_children()
            if getattr(widget, "_browser_descriptor", ("",))[0] == "session"
        ]
        assert len(session_rows) == 2
        app._invoke_browser_row_command(session_rows[0])
        app._invoke_browser_row_command(session_rows[1])

        assert opened == ["archived-1", "archived-2"]
        assert getattr(session_rows[0], "_browser_command_target_id", None) == "archived-1"
        assert getattr(session_rows[1], "_browser_command_target_id", None) == "archived-2"
    finally:
        _destroy_app_root(root, app if "app" in locals() else None)


def test_recent_session_cache_snapshot_preserves_summary_field(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")

        snapshot = app._session_metadata_snapshot(
            {
                "session_id": "desktop-1",
                "title": "",
                "summary": "summary fallback",
                "prompt": "",
                "created_at": "2026-03-23T12:00:00+00:00",
                "mode": "research",
            },
            archived=False,
        )

        assert snapshot["summary"] == "summary fallback"
    finally:
        root.destroy()


def test_conversation_cache_restore_hides_internal_sessions(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    cache_path = tmp_path / "data" / "desktop_ui" / "conversation_cache.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "recent": [
                    {
                        "session_id": "packaged-greeting-default",
                        "title": "hello",
                        "created_at": "2026-03-23T12:00:00+00:00",
                    },
                    {
                        "session_id": "desktop-1",
                        "title": "Real Chat",
                        "created_at": "2026-03-23T12:01:00+00:00",
                    },
                ],
                "archived": [
                    {
                        "session_id": "source-known-1",
                        "title": "what is entropy?",
                        "created_at": "2026-03-23T12:00:00+00:00",
                    },
                    {
                        "session_id": "desktop-archived",
                        "title": "Archived Chat",
                        "created_at": "2026-03-23T12:01:00+00:00",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")

        assert [session["session_id"] for session in app.recent_sessions_cache] == ["desktop-1"]
        assert [session["session_id"] for session in app.archived_sessions_cache] == ["desktop-archived"]
    finally:
        _destroy_app_root(root, app if "app" in locals() else None)


def test_conversation_cache_restore_ignores_corrupt_file_without_clobbering_runtime_state(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    cache_path = tmp_path / "data" / "desktop_ui" / "conversation_cache.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("{not valid json", encoding="utf-8")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=tmp_path, data_root=tmp_path / "data")
        app.recent_sessions_cache = [{"session_id": "desktop-live", "title": "Live"}]
        app.archived_sessions_cache = [{"session_id": "desktop-archived", "title": "Archived"}]

        app._load_conversation_cache()

        assert [session["session_id"] for session in app.recent_sessions_cache] == ["desktop-live"]
        assert [session["session_id"] for session in app.archived_sessions_cache] == ["desktop-archived"]
        assert app._conversation_cache_dirty is False
    finally:
        _destroy_app_root(root, locals().get("app"))


def test_recent_cache_placeholder_does_not_suppress_live_fetch(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    cache_path = tmp_path / "data" / "desktop_ui" / "conversation_cache.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "recent": [
                    {
                        "session_id": "desktop-stale",
                        "title": "Stale cached row",
                        "created_at": "2026-03-23T12:00:00+00:00",
                    }
                ],
                "archived": [],
            }
        ),
        encoding="utf-8",
    )
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        app.current_view = "recent"
        app.recent_sessions_render_signature = app.recent_sessions_signature
        app.recent_sessions_requested_signature = None
        app.recent_sessions_restored_from_disk = True
        scheduled: list[str] = []
        monkeypatch.setattr(app, "_start_ui_background_task", lambda task_name, *args, **kwargs: scheduled.append(task_name))

        app._refresh_recent_sessions_view()

        assert scheduled == ["recent"]
    finally:
        _destroy_app_root(root, locals().get("app"))


def test_duplicate_show_view_is_suppressed_when_surface_is_already_stable(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        app.current_view = "recent"
        app.recent_sessions_view_dirty = False
        app.hotbar_open = False
        app.hotbar_transition_in_progress = False
        calls: list[str] = []
        monkeypatch.setattr(app, "_ensure_view_built", lambda view_name: calls.append(f"ensure:{view_name}") or True)
        monkeypatch.setattr(app, "_apply_view_visibility", lambda view_name, schedule_refresh=True: calls.append(f"apply:{view_name}"))

        app._show_view("recent")
        root.update_idletasks()

        assert calls == []
    finally:
        _destroy_app_root(root, locals().get("app"))


def test_session_row_display_title_uses_summary_before_generic_chat(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        session = {
            "session_id": "desktop-summary-only",
            "title": "",
            "prompt": "",
            "summary": "summary fallback",
            "created_at": "2026-03-23T12:00:00+00:00",
            "mode": "conversation",
        }

        descriptors = app._build_session_row_descriptors(
            rows=grouped_session_rows([session]),
            target_session_count=1,
            already_rendered=0,
        )

        session_descriptor = next(descriptor for descriptor in descriptors if descriptor[0] == "session")
        assert session_descriptor[2] == "summary fallback"
        assert app._session_signature([session])[0][1] == "summary fallback"
    finally:
        _destroy_app_root(root, app if "app" in locals() else None)


def test_loaded_chat_title_does_not_use_summary_as_top_title() -> None:
    session = {
        "session_id": "desktop-summary-only",
        "title": "",
        "prompt": "",
        "summary": "Assistant last reply summary",
        "created_at": "2026-03-23T12:00:00+00:00",
    }

    assert LumenDesktopApp._session_title_fallback(session) == "Assistant last reply summary"
    assert LumenDesktopApp._session_open_title_fallback(session) == "Assistant last reply summary"


def test_rename_current_chat_persists_through_controller(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        app.recent_sessions_cache = [{"session_id": app.session_id, "title": "Chat"}]
        monkeypatch.setattr("lumen.desktop.chat_app.simpledialog.askstring", lambda *args, **kwargs: "Deep Space")

        app._rename_current_chat()

        assert app.chat_title_var.get() == "Deep Space"
        assert (app.session_id, "Deep Space") in app.controller.renamed_sessions
    finally:
        root.destroy()


def test_landing_greeting_varies_by_mode_and_avoids_quiet_collab_phrase(monkeypatch) -> None:
    class FakeDatetime(datetime):
        @classmethod
        def now(cls):
            return cls(2026, 4, 23, 22, 0)

    monkeypatch.setattr(chat_app_module, "datetime", FakeDatetime)
    fake = SimpleNamespace(mode_var=SimpleNamespace(get=lambda: "Collab"), session_id="desktop-test")

    greeting = LumenDesktopApp._landing_greeting(fake)

    assert greeting
    assert "take this quietly" not in greeting.lower()
    assert "quietly" not in greeting.lower()


def test_message_text_height_expands_for_long_assistant_replies() -> None:
    app = LumenDesktopApp.__new__(LumenDesktopApp)
    app._estimated_text_widget_width = lambda: 20
    long_text = " ".join(["Lumen can keep a long explanation visible."] * 80)

    height = app._estimate_text_widget_height(text=long_text)

    assert height > 24


def test_landing_mode_change_does_not_create_recent_row(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        app.recent_sessions_cache = []
        app.recent_sessions_rows = []

        app.mode_var.set("Direct")
        app._on_mode_changed()

        assert app.controller.profile_calls[-1] == (app.session_id, "direct")
        assert app.recent_sessions_cache == []
        assert app.recent_sessions_rows == []
    finally:
        _destroy_app_root(root, app if "app" in locals() else None)


def test_rename_current_chat_falls_back_locally_if_persistence_fails(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        monkeypatch.setattr("lumen.desktop.chat_app.simpledialog.askstring", lambda *args, **kwargs: "Recovered Title")
        monkeypatch.setattr(
            app.controller,
            "rename_session",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("rename failed")),
        )

        app._rename_current_chat()

        assert app.chat_title_var.get() == "Recovered Title"
        assert app.status_var.get() == "Chat renamed locally"
    finally:
        root.destroy()


def test_custom_theme_option_is_available_in_settings(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        app._show_view("settings")

        assert app.theme_value_label.cget("text") == "Dark"
        assert "Font" in {label.cget("text") for label in app.settings_row_labels}

        app.theme_var.set("Custom")
        app._on_theme_changed()

        assert app.theme_value_label.cget("text") in {
            "Custom • Lumen Purple",
            "Custom • Color Wheel",
        }
    finally:
        root.destroy()


def test_mode_selector_is_in_composer_not_hotbar(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())

        assert hasattr(app, "mode_selector")
        assert app.mode_selector.master == app.mode_pill
        assert not hasattr(app, "hotbar_mode_selector")
        assert int(app.mode_selector.grid_info()["column"]) == 0
        assert int(app.input_box.grid_info()["column"]) == 2
        assert int(app.add_button.grid_info()["column"]) == 3
        assert int(app.mic_button.grid_info()["column"]) == 4
    finally:
        root.destroy()


def test_clear_attachment_button_is_hidden_without_attachment(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        root.update_idletasks()

        assert app.clear_attachment_button.winfo_manager() == ""

        selected = tmp_path / "input.txt"
        selected.write_text("hello")
        app._set_attachment(selected, kind="file")
        root.update_idletasks()

        assert app.clear_attachment_button.winfo_manager() == "grid"

        app._clear_attachment()
        root.update_idletasks()

        assert app.clear_attachment_button.winfo_manager() == ""
    finally:
        root.destroy()


def test_settings_help_starts_hidden_and_toggles(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        app._show_view("settings")
        root.update_idletasks()

        assert app.help_text.winfo_manager() == ""
        assert app.allow_emojis_var.get() is False
        assert not hasattr(app, "help_toggle_button")
        assert app.help_row_indicator.cget("text") == "Show"

        app._toggle_settings_help()
        root.update_idletasks()

        assert app.help_text.winfo_manager() == "grid"
        assert app.help_row_indicator.cget("text") == "Hide"
        assert app.help_text.cget("bg") == app.current_palette["app_bg"]
        assert app.help_text.cget("fg") == app.current_palette["text_primary"]
    finally:
        root.destroy()


def test_landing_screen_uses_identity_icon_when_available(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        root.update_idletasks()

        assert app.identity_label.cget("text") == ""
        assert app.identity_image is not None
        assert app.identity_label.master == app.landing_orb_frame
        assert app.greeting_label.master == app.landing_greeting_frame
        assert app.landing_orb_frame.winfo_height() >= 0
    finally:
        root.destroy()


def test_landing_icon_scales_with_available_width(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        monkeypatch.setattr(app.chat_canvas, "winfo_width", lambda: 2000)

        icon_size = app._current_identity_icon_size()
        app._refresh_landing_icon_geometry(2000)

        assert app.IDENTITY_ICON_MIN_SIZE <= icon_size <= app.IDENTITY_ICON_MAX_SIZE
        assert icon_size == 200
        assert int(app.landing_orb_frame.cget("width")) == icon_size + 40
        assert int(app.landing_orb_frame.cget("height")) == icon_size + 40
    finally:
        root.destroy()


def test_custom_color_wheel_applies_runtime_accent(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    monkeypatch.setattr("lumen.desktop.chat_app.colorchooser.askcolor", lambda **kwargs: ((171, 102, 255), "#ab66ff"))
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        app._show_view("settings")
        app.theme_var.set("Custom")
        app.custom_theme_var.set("Color Wheel")

        app._on_custom_theme_changed()

        assert app.custom_accent_color == "#ab66ff"
        assert app.current_palette["input_focus_border"] == "#ab66ff"
        assert app.current_palette["app_bg"] != DARK_PALETTE["app_bg"]
    finally:
        root.destroy()


def test_color_wheel_selection_closes_popup_before_chooser(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        app._show_view("settings")
        app.theme_var.set("Custom")
        app._show_settings_popup("custom_theme")
        states: list[tuple[bool, bool]] = []
        original_on_theme_changed = app._on_theme_changed
        theme_changes: list[str] = []

        def _askcolor(**kwargs):
            states.append((app.settings_popup is None, app._custom_color_chooser_active))
            return ((171, 102, 255), "#ab66ff")

        def _wrapped_on_theme_changed(event=None):
            theme_changes.append("theme")
            return original_on_theme_changed(event)

        monkeypatch.setattr("lumen.desktop.chat_app.colorchooser.askcolor", _askcolor)
        monkeypatch.setattr(app, "_on_theme_changed", _wrapped_on_theme_changed)

        app._select_custom_theme_option("Color Wheel")

        assert states == [(True, True)]
        assert theme_changes == ["theme"]
        assert app.settings_popup is None
        assert app._custom_color_chooser_active is False
        assert app.custom_theme_var.get() == "Color Wheel"
        assert app.custom_accent_color == "#ab66ff"
    finally:
        root.destroy()


def test_color_wheel_cancel_restores_prior_custom_theme_state(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        app._show_view("settings")
        app.theme_var.set("Custom")
        app.custom_theme_var.set("Lumen Purple")
        app.custom_accent_color = None
        app._show_settings_popup("custom_theme")
        theme_changes: list[str] = []

        monkeypatch.setattr("lumen.desktop.chat_app.colorchooser.askcolor", lambda **kwargs: (None, None))
        monkeypatch.setattr(app, "_on_theme_changed", lambda event=None: theme_changes.append("theme"))

        app._select_custom_theme_option("Color Wheel")

        assert theme_changes == []
        assert app.settings_popup is None
        assert app._custom_color_chooser_active is False
        assert app.custom_theme_var.get() == "Lumen Purple"
        assert app.custom_accent_color is None
    finally:
        root.destroy()


def test_font_setting_persists_to_desktop_preferences(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        app.font_family_var.set("Arial")

        app._on_font_changed()

        prefs_path = tmp_path / "data" / "desktop_ui" / "preferences.json"
        payload = json.loads(prefs_path.read_text(encoding="utf-8"))
        assert payload["font_family"] == "Arial"
    finally:
        root.destroy()


def test_font_change_updates_live_landing_and_starter_fonts(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        app.font_family_var.set("Arial")

        app._on_font_changed()

        assert "Arial" in str(app.greeting_label.cget("font"))
        assert "Arial" in str(app.greeting_subtitle.cget("font"))
        assert all("Arial" in str(button.cget("font")) for button in app.starter_prompt_buttons)
    finally:
        root.destroy()


def test_mode_setting_persists_to_desktop_preferences(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        app.mode_var.set("Direct")

        app._on_mode_changed()

        payload = json.loads(app.desktop_prefs_path.read_text(encoding="utf-8"))
        assert payload["interaction_style"] == "direct"
    finally:
        root.destroy()


def test_desktop_preferences_restore_saved_mode(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    prefs_path = tmp_path / "data" / "desktop_ui" / "preferences.json"
    prefs_path.parent.mkdir(parents=True, exist_ok=True)
    prefs_path.write_text(json.dumps({"interaction_style": "collab"}), encoding="utf-8")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")

        assert app.mode_var.get() == "Collab"
    finally:
        root.destroy()


def test_profile_image_path_persists_to_desktop_preferences(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        profile_image = tmp_path / "avatar.png"
        profile_image.write_text("not-a-real-image", encoding="utf-8")
        app.profile_avatar_path = profile_image

        app._save_desktop_preferences()

        prefs_path = tmp_path / "data" / "desktop_ui" / "preferences.json"
        payload = json.loads(prefs_path.read_text(encoding="utf-8"))
        assert payload["profile_avatar_path"] == str(profile_image)
    finally:
        root.destroy()


def test_settings_theme_switch_keeps_text_visible(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        app._show_view("settings")

        app.theme_var.set("Light")
        app._on_theme_changed()
        light_colors = {label.cget("fg") for label in app.settings_row_labels}

        app.theme_var.set("Dark")
        app._on_theme_changed()
        dark_colors = {label.cget("fg") for label in app.settings_row_labels}

        assert LIGHT_PALETTE["text_primary"] in light_colors
        assert DARK_PALETTE["text_primary"] in dark_colors
    finally:
        root.destroy()


def test_language_style_selector_updates_emoji_preference(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        app._show_view("settings")

        app._select_language_style_option("Emoji Friendly")

        assert app.language_style_var.get() == "Emoji Friendly"
        assert app.allow_emojis_var.get() is True
        assert str(app.language_style_value_label.cget("textvariable")) == str(app.language_style_var)
    finally:
        root.destroy()


def test_reset_button_restores_theme_defaults_without_touching_custom_wheel_accent(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        app._show_view("settings")
        app.theme_var.set("Custom")
        app.custom_theme_var.set("Color Wheel")
        app.custom_accent_color = "#ab66ff"
        app.font_family_var.set("Impact")
        app.text_size_var.set(14)
        app.language_style_var.set("Emoji Friendly")
        app.allow_emojis_var.set(True)
        app.custom_colors["user_bg"] = "#111111"
        app.custom_colors["assistant_text"] = "#ffffff"

        app._reset_style_overrides()

        assert app.theme_var.get() == "Custom"
        assert app.custom_theme_var.get() == "Color Wheel"
        assert app.custom_accent_color == "#ab66ff"
        assert app.font_family_var.get() == app.DEFAULT_FONT_FAMILY
        assert app.text_size_var.get() == 14
        assert app.language_style_var.get() == "Emoji Friendly"
        assert app.allow_emojis_var.get() is True
        assert all(value is None for value in app.custom_colors.values())
        assert app.current_palette["input_focus_border"] == "#ab66ff"

        prefs_path = tmp_path / "data" / "desktop_ui" / "preferences.json"
        payload = json.loads(prefs_path.read_text(encoding="utf-8"))
        assert payload["font_family"] == app.DEFAULT_FONT_FAMILY
        assert payload["custom_accent_color"] == "#ab66ff"
        assert all(not value for value in payload["custom_colors"].values())
    finally:
        _destroy_app_root(root, app if "app" in locals() else None)


def test_settings_reset_button_uses_lumen_purple_style(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        app._show_view("settings")

        assert app.settings_reset_button.winfo_manager() == "place"
        assert app.settings_reset_button.cget("text") == "Reset"
        assert app.settings_reset_button.cget("bg") == THEME_PALETTES["custom"]["nav_active_border"]
    finally:
        _destroy_app_root(root, app if "app" in locals() else None)


def test_settings_reset_button_uses_color_wheel_accent_when_active(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        app._show_view("settings")
        app.theme_var.set("Custom")
        app.custom_theme_var.set("Color Wheel")
        app.custom_accent_color = "#2277ee"

        app._style_settings_reset_button(hovered=False)

        assert app.settings_reset_button.cget("bg") == custom_accent_palette("#2277ee")["nav_active_border"]
    finally:
        _destroy_app_root(root, app if "app" in locals() else None)


def test_light_mode_top_icon_style_uses_light_theme_tokens() -> None:
    app = object.__new__(LumenDesktopApp)
    app.theme_var = SimpleNamespace(get=lambda: "Light")
    app.current_palette = dict(LIGHT_PALETTE)

    idle = app._top_icon_style(hovered=False, primary=False)
    hovered = app._top_icon_style(hovered=True, primary=False)
    primary = app._top_icon_style(hovered=False, primary=True)

    assert idle["fg"] == LIGHT_PALETTE["nav_active_border"]
    assert primary["fg"] == LIGHT_PALETTE["nav_active_border"]
    assert idle == resolve_top_icon_palette(
        palette=LIGHT_PALETTE,
        theme_name="light",
        hovered=False,
        primary=False,
        enabled=True,
    )
    assert hovered == resolve_top_icon_palette(
        palette=LIGHT_PALETTE,
        theme_name="light",
        hovered=True,
        primary=False,
        enabled=True,
    )
    assert primary == resolve_top_icon_palette(
        palette=LIGHT_PALETTE,
        theme_name="light",
        hovered=False,
        primary=True,
        enabled=True,
    )


def test_dark_family_top_icons_keep_accent_when_disabled() -> None:
    style = resolve_top_icon_palette(
        palette=LUMEN_PURPLE_PALETTE,
        theme_name="custom",
        hovered=False,
        primary=False,
        enabled=False,
    )

    assert style["fg"] == LUMEN_PURPLE_PALETTE["nav_active_border"]
    assert style["disabledforeground"] == LUMEN_PURPLE_PALETTE["nav_active_border"]


def test_theme_change_cancels_deferred_refresh_and_requeues_current_view() -> None:
    app = object.__new__(LumenDesktopApp)
    calls: list[str] = []
    canceled: list[str] = []
    app.theme_var = SimpleNamespace(get=lambda: "Light")
    app.custom_theme_var = SimpleNamespace(get=lambda: "Lumen Purple")
    app.custom_accent_color = None
    app.current_theme = dict(THEME_TOKENS["dark"])
    app.current_palette = dict(DARK_PALETTE)
    app.current_view = "recent"
    app.deferred_view_refresh_job = "refresh-job"
    app.deferred_view_refresh_target = "recent"
    app.pending_hotbar_refresh_target = "recent"
    app.hotbar_transition_in_progress = False
    app.hotbar_animation_job = None
    app.hotbar_open = False
    app.root = SimpleNamespace(
        configure=lambda **kwargs: calls.append(str(kwargs.get("bg"))),
        after_idle=lambda callback: callback(),
        after_cancel=lambda job: canceled.append(str(job)),
    )
    app._configure_styles = lambda: calls.append("styles")
    app._apply_palette_to_shell = lambda **kwargs: calls.append(f"shell:{kwargs}")
    app._schedule_view_refresh = lambda view_name: calls.append(f"refresh:{view_name}")
    app._persist_desktop_preferences_safe = lambda: calls.append("prefs")

    app._on_theme_changed()

    assert canceled == ["refresh-job"]
    assert app.current_theme == THEME_TOKENS["light"]
    assert app.current_palette["app_bg"] == LIGHT_PALETTE["app_bg"]
    assert calls == [
        LIGHT_PALETTE["app_bg"],
        "styles",
        "shell:{'reflow_messages': False, 'include_assets': False, 'include_cache': False}",
        "refresh:recent",
        "prefs",
    ]


def test_theme_change_stabilizes_hotbar_transition_before_restyle() -> None:
    app = object.__new__(LumenDesktopApp)
    calls: list[str] = []
    canceled: list[str] = []
    app.theme_var = SimpleNamespace(get=lambda: "Light")
    app.custom_theme_var = SimpleNamespace(get=lambda: "Lumen Purple")
    app.custom_accent_color = None
    app.current_theme = dict(THEME_TOKENS["dark"])
    app.current_palette = dict(DARK_PALETTE)
    app.current_view = "chat"
    app.deferred_view_refresh_job = None
    app.deferred_view_refresh_target = None
    app.pending_hotbar_refresh_target = "recent"
    app.hotbar_transition_in_progress = True
    app.hotbar_animation_job = "hotbar-job"
    app.hotbar_open = True
    app.hotbar_target_width = 236
    app.hotbar_current_width = 104
    app.root = SimpleNamespace(
        configure=lambda **kwargs: calls.append(str(kwargs.get("bg"))),
        after_idle=lambda callback: callback(),
        after_cancel=lambda job: canceled.append(str(job)),
    )
    app.hotbar = SimpleNamespace(
        winfo_exists=lambda: True,
        configure=lambda **kwargs: calls.append(f"hotbar:{kwargs.get('width')}"),
        grid=lambda: calls.append("hotbar:grid"),
        grid_remove=lambda: calls.append("hotbar:grid_remove"),
    )
    app._unlock_transition_layout = lambda: calls.append("unlock")
    app._configure_styles = lambda: calls.append("styles")
    app._apply_palette_to_shell = lambda **kwargs: calls.append(f"shell:{kwargs}")
    app._persist_desktop_preferences_safe = lambda: calls.append("prefs")

    app._on_theme_changed()

    assert canceled == ["hotbar-job"]
    assert app.pending_hotbar_refresh_target == "recent"
    assert app.hotbar_transition_in_progress is False
    assert app.hotbar_current_width == 236
    assert calls == [
        LIGHT_PALETTE["app_bg"],
        "hotbar:236",
        "hotbar:grid",
        "unlock",
        "styles",
        "shell:{'reflow_messages': False, 'include_assets': False, 'include_cache': False}",
        "prefs",
    ]


def test_hotbar_toggle_queues_reopen_during_close_transition() -> None:
    app = object.__new__(LumenDesktopApp)
    app.hotbar_open = True
    app.hotbar_transition_in_progress = True
    app.pending_hotbar_open_state = None
    app.hotbar_current_width = 236
    app.hotbar_target_width = 236

    app._toggle_hotbar()

    assert app.pending_hotbar_open_state is False

    app.hotbar_open = False
    app._toggle_hotbar()

    assert app.pending_hotbar_open_state is True


def test_schedule_view_refresh_waits_for_hotbar_to_settle() -> None:
    app = object.__new__(LumenDesktopApp)
    app.current_view = "recent"
    app.hotbar_open = True
    app.hotbar_animation_job = "job-1"
    app.hotbar_navigation_generation = 3
    app.deferred_view_refresh_job = None
    app.deferred_view_refresh_target = None
    app.deferred_view_refresh_generation = 0
    app.pending_hotbar_refresh_target = None
    app.pending_refresh_generation = 0
    app.root = SimpleNamespace(after_idle=lambda callback: "idle-job", after_cancel=lambda job: None)

    app._schedule_view_refresh("recent")

    assert app.deferred_view_refresh_job is None
    assert app.deferred_view_refresh_target == "recent"
    assert app.pending_hotbar_refresh_target == "recent"
    assert app.pending_refresh_generation == 3


def test_queue_deferred_view_refresh_coalesces_duplicate_schedule_requests() -> None:
    app = object.__new__(LumenDesktopApp)
    queued: list[str] = []
    app.deferred_view_refresh_job = None
    app.root = SimpleNamespace(after_idle=lambda callback: queued.append("idle") or "job-1")

    app._queue_deferred_view_refresh()
    app._queue_deferred_view_refresh()

    assert queued == ["idle"]
    assert app.deferred_view_refresh_job == "job-1"


def test_bind_global_mousewheel_attaches_once() -> None:
    app = object.__new__(LumenDesktopApp)
    binds: list[str] = []
    app._mousewheel_bound_globally = False
    app.root = SimpleNamespace(bind_all=lambda pattern, callback, add=None: binds.append(str(pattern)))

    app._bind_global_mousewheel()
    app._bind_global_mousewheel()

    assert binds == ["<MouseWheel>", "<Button-4>", "<Button-5>"]


def test_hotbar_finalize_queues_single_deferred_refresh_after_close() -> None:
    app = object.__new__(LumenDesktopApp)
    queued: list[str] = []
    app.hotbar_target_width = 236
    app.hotbar_open = True
    app.hotbar_current_width = 236
    app.hotbar = SimpleNamespace(winfo_exists=lambda: True, configure=lambda **kwargs: None, grid=lambda: None, grid_remove=lambda: None)
    app._unlock_transition_layout = lambda: None
    app._debug_timing = lambda label, elapsed_ms: None
    app._timed_ui_call = lambda label, func: func()
    app.pending_hotbar_open_state = None
    app.pending_view_name = None
    app.pending_hotbar_refresh_target = "recent"
    app.deferred_view_refresh_target = "recent"
    app.hotbar_navigation_generation = 1
    app.pending_refresh_generation = 1
    app.deferred_view_refresh_generation = 0
    app.deferred_view_refresh_job = None
    app.hotbar_animation_job = "job-1"
    app.hotbar_transition_in_progress = True
    app._hotbar_transition_started_at = None
    app.root = SimpleNamespace(after_idle=lambda callback: queued.append("idle") or "job-1")

    app._finalize_hotbar_transition(opening=False)

    assert queued == ["idle"]
    assert app.deferred_view_refresh_target == "recent"
    assert app.deferred_view_refresh_job == "job-1"
    assert app.deferred_view_refresh_generation == 1


def test_hotbar_animation_skips_redundant_width_reconfigure(monkeypatch) -> None:
    app = object.__new__(LumenDesktopApp)
    configured: list[int] = []
    scheduled: list[int] = []
    app.HOTBAR_ANIMATION_DURATION = 0.10
    app.HOTBAR_ANIMATION_INTERVAL_MS = 16
    app.hotbar_transition_in_progress = False
    app.hotbar_animation_job = None
    app.hotbar_current_width = 0
    app.hotbar_target_width = 236
    app.hotbar_open = False
    app.root = SimpleNamespace(after=lambda delay, callback: scheduled.append(int(delay)) or "job-1", after_cancel=lambda job: None)
    app.hotbar = SimpleNamespace(
        winfo_exists=lambda: True,
        configure=lambda **kwargs: configured.append(int(kwargs["width"])),
        grid=lambda: None,
        grid_remove=lambda: None,
    )
    app._lock_transition_layout = lambda: None
    app._finalize_hotbar_transition = lambda *, opening: configured.append(999)

    timings = iter([0.0, 0.0, 0.00001])
    monkeypatch.setattr("lumen.desktop.chat_app.perf_counter", lambda: next(timings))

    app._animate_hotbar(opening=True)

    assert configured == []
    assert scheduled == [16]


def test_hotbar_open_close_on_chat_does_not_reload_identity_icon(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        root.update_idletasks()
        loads: list[str] = []

        monkeypatch.setattr(app, "_load_identity_icon", lambda: loads.append("icon"))

        def _settle_hotbar() -> None:
            for _ in range(40):
                if app.hotbar_animation_job is None:
                    break
                time.sleep(0.01)
                root.update()
            root.update_idletasks()
            assert app.hotbar_animation_job is None

        app._toggle_hotbar()
        _settle_hotbar()
        app._handle_global_click(SimpleNamespace(widget=app.content_container))
        _settle_hotbar()

        assert loads == []
    finally:
        root.destroy()


def test_pure_hotbar_toggle_does_not_create_navigation_or_refresh_intent(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")

        def _settle_hotbar() -> None:
            for _ in range(80):
                if app.hotbar_animation_job is None:
                    break
                time.sleep(0.01)
                root.update()
            assert app.hotbar_animation_job is None

        app._toggle_hotbar()
        _settle_hotbar()
        app._toggle_hotbar()
        _settle_hotbar()

        assert app.pending_view_name is None
        assert app.pending_hotbar_refresh_target is None
        assert app.deferred_view_refresh_target is None
    finally:
        root.destroy()


def test_global_click_inside_hotbar_closes_settings_popup_without_closing_hotbar(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        app._show_view("settings")
        app._toggle_hotbar()

        for _ in range(80):
            if app.hotbar_animation_job is None:
                break
            time.sleep(0.01)
            root.update()

        app._show_settings_popup("theme")
        root.update()
        root.update_idletasks()
        assert app.settings_popup is not None
        assert app.hotbar_open is True

        app._handle_global_click(SimpleNamespace(widget=app.hotbar_hint))
        root.update()
        root.update_idletasks()

        assert app.settings_popup is None
        assert app.hotbar_open is True
    finally:
        root.destroy()


def test_global_click_inside_settings_row_closes_hotbar_without_closing_popup(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        app._show_view("settings")
        app._toggle_hotbar()

        for _ in range(80):
            if app.hotbar_animation_job is None:
                break
            time.sleep(0.01)
            root.update()

        app._show_settings_popup("theme")
        root.update()
        root.update_idletasks()
        assert app.settings_popup is not None
        assert app.hotbar_open is True

        theme_row = app.settings_row_map["Theme"]
        app._handle_global_click(SimpleNamespace(widget=theme_row))

        for _ in range(80):
            if app.hotbar_animation_job is None:
                break
            time.sleep(0.01)
            root.update()

        root.update_idletasks()
        assert app.settings_popup is not None
        assert app.hotbar_open is False
    finally:
        root.destroy()


def test_theme_change_during_hotbar_transition_clears_stale_hotbar_follow_up_work() -> None:
    app = object.__new__(LumenDesktopApp)
    calls: list[str] = []
    canceled: list[str] = []
    app.theme_var = SimpleNamespace(get=lambda: "Light")
    app.custom_theme_var = SimpleNamespace(get=lambda: "Lumen Purple")
    app.custom_accent_color = None
    app.current_theme = dict(THEME_TOKENS["dark"])
    app.current_palette = dict(DARK_PALETTE)
    app.current_view = "chat"
    app.deferred_view_refresh_job = None
    app.deferred_view_refresh_target = "recent"
    app.pending_hotbar_refresh_target = "recent"
    app.pending_hotbar_open_state = True
    app.pending_view_name = "recent"
    app.pending_view_job = "pending-view-job"
    app._deferred_refresh_from_hotbar_close = True
    app.hotbar_transition_in_progress = True
    app.hotbar_animation_job = "hotbar-job"
    app.hotbar_open = True
    app.root = SimpleNamespace(
        configure=lambda **kwargs: calls.append(str(kwargs.get("bg"))),
        after_idle=lambda callback: callback(),
        after_cancel=lambda job: canceled.append(str(job)),
    )
    app.hotbar_target_width = 236
    app.hotbar_current_width = 112
    app.hotbar = SimpleNamespace(
        winfo_exists=lambda: True,
        configure=lambda **kwargs: calls.append(f"hotbar:{kwargs.get('width')}"),
        grid=lambda: calls.append("hotbar:grid"),
        grid_remove=lambda: calls.append("hotbar:grid_remove"),
    )
    app._unlock_transition_layout = lambda: calls.append("unlock")
    app._configure_styles = lambda: calls.append("styles")
    app._apply_palette_to_shell = lambda **kwargs: calls.append(f"shell:{kwargs}")
    app._persist_desktop_preferences_safe = lambda: calls.append("prefs")

    app._on_theme_changed()

    assert canceled == ["hotbar-job"]
    assert app.pending_hotbar_refresh_target == "recent"
    assert app.pending_hotbar_open_state is None
    assert app.pending_view_name == "recent"
    assert app.pending_view_job == "pending-view-job"
    assert app.hotbar_transition_in_progress is False
    assert app.hotbar_current_width == 236
    assert "unlock" in calls


def test_control_availability_restyles_top_icons_without_clearing_hotbar_navigation_state() -> None:
    app = object.__new__(LumenDesktopApp)
    restyles: list[str] = []
    app._shell_ready_flag = False
    app.pending = False
    app.hotbar_navigation_generation = 7
    app.pending_hotbar_refresh_target = "recent"
    app.pending_view_name = "recent"
    app._desktop_capability_state = DesktopCapabilityState.booting()
    app.nav_buttons = {}
    app._refresh_top_icon_styles = lambda: restyles.append("top-icons")
    app._style_nav_buttons = lambda: restyles.append("nav")

    app._apply_control_availability()

    assert restyles == ["top-icons", "nav"]
    assert app.pending_hotbar_refresh_target == "recent"
    assert app.pending_view_name == "recent"
    assert app.hotbar_navigation_generation == 7


def test_non_chat_view_visibility_hides_landing_without_refreshing_chat_landing(monkeypatch) -> None:
    app = object.__new__(LumenDesktopApp)
    actions: list[str] = []
    chat_frame = SimpleNamespace(grid=lambda **kwargs: actions.append("chat:grid"), grid_forget=lambda: actions.append("chat:forget"))
    recent_frame = SimpleNamespace(grid=lambda **kwargs: actions.append("recent:grid"), grid_forget=lambda: actions.append("recent:forget"))
    app.views = {"chat": chat_frame, "recent": recent_frame}
    app.input_frame = SimpleNamespace(grid=lambda: actions.append("input:grid"), grid_remove=lambda: actions.append("input:remove"))
    app.landing_frame = SimpleNamespace(place_forget=lambda: actions.append("landing:forget"))
    app._style_nav_buttons = lambda: actions.append("nav")
    app._refresh_top_bar_title = lambda: actions.append("title")
    app._update_landing_state = lambda: actions.append("landing:update")
    app._schedule_view_refresh = lambda view_name: actions.append(f"refresh:{view_name}")

    app._apply_view_visibility("recent")

    assert actions == [
        "chat:forget",
        "recent:grid",
        "input:remove",
        "landing:forget",
        "title",
        "nav",
        "refresh:recent",
    ]


def test_hotbar_destination_navigation_schedules_single_view_change_and_refresh(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        app._toggle_hotbar()

        for _ in range(80):
            if app.hotbar_animation_job is None:
                break
            time.sleep(0.01)
            root.update()

        shown: list[str] = []
        queued: list[str] = []
        original_show_view = app._show_view
        monkeypatch.setattr(app, "_queue_deferred_view_refresh", lambda: queued.append(str(app.deferred_view_refresh_target)))

        def _wrapped_show_view(view_name: str) -> None:
            shown.append(view_name)
            original_show_view(view_name)

        monkeypatch.setattr(app, "_show_view", _wrapped_show_view)

        app._handle_hotbar_destination("recent")
        root.update()
        root.update_idletasks()

        assert shown == ["recent"]
        assert queued == ["recent"]
        assert app.pending_view_name is None
        assert app.current_view == "recent"
        assert app._debug_ui_event_counts["apply_pending_view"] == 1
        assert app._debug_ui_event_counts["show_view"] >= 1
    finally:
        root.destroy()


def test_new_hotbar_destination_clears_stale_pending_view_job_and_refresh(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        canceled: list[str] = []
        monkeypatch.setattr(root, "after_cancel", lambda job: canceled.append(str(job)))
        finalized: list[bool] = []
        monkeypatch.setattr(app, "_finalize_hotbar_transition", lambda *, opening: finalized.append(bool(opening)))

        app.hotbar_open = True
        app.pending_view_job = "stale-job"
        app.pending_view_name = "memory"
        app.pending_hotbar_refresh_target = "memory"
        app.deferred_view_refresh_target = "memory"

        app._handle_hotbar_destination("recent")

        assert canceled == ["stale-job"]
        assert app.pending_view_name == "recent"
        assert app.pending_hotbar_refresh_target is None
        assert app.deferred_view_refresh_target is None
        assert finalized == [False]
    finally:
        root.destroy()


def test_apply_pending_view_ignores_stale_generation() -> None:
    app = object.__new__(LumenDesktopApp)
    app.pending_view_job = "job"
    app.hotbar_navigation_generation = 4
    calls: list[str] = []
    app._debug_event = lambda *args, **kwargs: calls.append(str(args[0]))
    app._show_view = lambda view_name: calls.append(f"show:{view_name}")

    app._apply_pending_view("recent", generation=3)

    assert calls == ["hotbar_navigation_discarded"]


def test_run_deferred_view_refresh_ignores_stale_generation() -> None:
    app = object.__new__(LumenDesktopApp)
    calls: list[str] = []
    app.deferred_view_refresh_target = "memory"
    app.deferred_view_refresh_generation = 2
    app.deferred_view_refresh_job = "job"
    app.current_view = "memory"
    app.hotbar_navigation_generation = 3
    app._deferred_refresh_from_hotbar_close = False
    app._debug_event = lambda *args, **kwargs: calls.append(str(args[0]))
    app._timed_ui_call = lambda label, func: func()
    app._set_deferred_view_refresh_target = lambda view_name, generation=None: setattr(app, "deferred_view_refresh_target", view_name)

    app._run_deferred_view_refresh()

    assert calls[:2] == ["run_deferred_view_refresh", "hotbar_navigation_discarded"]


def test_view_change_does_not_create_pending_indicator(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        pending: list[str] = []
        monkeypatch.setattr(app, "_show_pending_indicator", lambda *args, **kwargs: pending.append("pending"))

        app._show_view("recent")
        root.update()
        app._drain_queues_once()

        assert pending == []
    finally:
        root.destroy()


def test_text_size_row_opens_settings_popup(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        app._show_view("settings")

        app._activate_text_size_row()
        root.update_idletasks()

        assert app.settings_popup is not None
        assert app.settings_popup_kind == "text_size"
    finally:
        root.destroy()


def test_input_row_is_built_inside_shiftable_main_column(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())

        assert app.input_frame.master == app.main_column
        assert app.content_container.master == app.main_column
    finally:
        root.destroy()


def test_top_bar_shows_chat_title_only_for_named_chat(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())

        assert app.screen_title_var.get() == ""

        app.chat_title_var.set("Deep Space")
        app._refresh_top_bar_title()

        assert app.screen_title_var.get() == "Deep Space"

        app.chat_title_var.set("Chat")
        app._refresh_top_bar_title()

        assert app.screen_title_var.get() == ""
    finally:
        root.destroy()


def test_recent_sessions_view_uses_bounded_rendering_with_load_more(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    _install_immediate_ui_task_runner(monkeypatch)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        app.controller._recent_sessions = [
            {
                "session_id": f"desktop-{index}",
                "summary": f"Summary {index}",
                "prompt": f"Prompt {index}",
                "mode": "conversation",
                "created_at": "2026-03-23T12:00:00+00:00",
            }
            for index in range(40)
        ]

        app._show_view("recent")
        root.update()
        app._drain_queues_once()

        assert app.recent_sessions_load_more_button is not None
        assert app.recent_sessions_render_limit == 4
        assert app.recent_sessions_load_more_button.cget("text") == "Load More"
        assert app.recent_sessions_load_more_button.pack_info()["anchor"] == "center"

        app._load_more_recent_sessions()

        assert app.recent_sessions_render_limit == 8
    finally:
        root.destroy()


def test_recent_sessions_reopen_resets_to_first_four_items(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    _install_immediate_ui_task_runner(monkeypatch)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        app.controller._recent_sessions = [
            {
                "session_id": f"desktop-{index}",
                "summary": f"Summary {index}",
                "prompt": f"Prompt {index}",
                "mode": "conversation",
                "created_at": "2026-03-23T12:00:00+00:00",
            }
            for index in range(20)
        ]

        app._show_view("recent")
        root.update()
        app._drain_queues_once()
        app._load_more_recent_sessions()

        assert app.recent_sessions_render_limit == 8

        app._show_view("chat")
        root.update()
        app._drain_queues_once()

        app._show_view("recent")
        root.update()
        app._drain_queues_once()

        assert app.recent_sessions_render_limit == 4
        assert app.recent_sessions_rendered_count == 4
        assert app.recent_sessions_load_more_button is not None
    finally:
        root.destroy()


def test_memory_view_uses_bounded_rendering_with_load_more(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    _install_immediate_ui_task_runner(monkeypatch)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        entries = [
            {
                "title": f"Memory {index}",
                "content": f"Note {index}",
                "created_at": "2026-03-23T12:00:00+00:00",
                "entry_path": f"memory-{index}.json",
            }
            for index in range(40)
        ]
        monkeypatch.setattr(app.controller, "list_personal_memory", lambda **kwargs: {"personal_memory": list(entries)})
        monkeypatch.setattr(app.controller, "list_research_notes", lambda **kwargs: {"research_notes": []})

        app._show_view("memory")
        root.update()
        app._drain_queues_once()

        assert app.memory_load_more_button is not None
        assert app.memory_render_limit == 4
        assert app.memory_rendered_count == 4
        assert app.memory_load_more_button.cget("text") == "Load More"
        assert app.memory_load_more_button.pack_info()["anchor"] == "center"

        app._load_more_memory_entries()
        root.update()
        app._drain_queues_once()

        assert app.memory_render_limit == 8
        assert app.memory_rendered_count == 8
    finally:
        root.destroy()


def test_memory_view_restores_first_window_from_disk_cache(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    _install_immediate_ui_task_runner(monkeypatch)
    cache_path = tmp_path / "data" / "desktop_ui" / "conversation_cache.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_entries = [
        {
            "title": f"Memory {index}",
            "content": f"Cached {index}",
            "created_at": f"2026-03-{30 - index:02d}T12:00:00+00:00",
            "kind": "personal_memory",
            "entry_path": f"memory-{index}.json",
        }
        for index in range(5)
    ]
    cache_path.write_text(
        json.dumps(
            {
                "recent": [],
                "archived": [],
                "memory": {
                    "entries": cache_entries,
                    "signature": [
                        [entry["title"], entry["created_at"], entry["kind"], entry["entry_path"]]
                        for entry in cache_entries
                    ],
                    "has_more": True,
                },
                "archived_memory": {"entries": [], "signature": [], "has_more": False},
            }
        ),
        encoding="utf-8",
    )
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        app._show_view("memory")
        root.update()

        assert app.memory_rendered_count == 4
        assert app.memory_load_more_button is not None
        assert "Loading memory..." not in app.memory_preview.get("1.0", "end-1c")
        first_row = next(
            child
            for child in app.memory_list_inner.winfo_children()
            if getattr(child, "_browser_descriptor", ("",))[0] == "entry"
        )
        assert getattr(first_row, "_browser_descriptor")[1] == "Memory 0"
    finally:
        root.destroy()


def test_memory_view_reopen_resets_to_first_four_items(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    _install_immediate_ui_task_runner(monkeypatch)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        fetch_limits: list[int | None] = []

        def _personal_memory(**kwargs):
            fetch_limits.append(kwargs.get("limit"))
            entries = [
                {
                    "title": f"Memory {index}",
                    "content": f"Note {index}",
                    "created_at": "2026-03-23T12:00:00+00:00",
                    "entry_path": f"memory-{index}.json",
                }
                for index in range(40)
            ]
            limit = kwargs.get("limit")
            return {"personal_memory": entries[:limit] if limit is not None else entries}

        monkeypatch.setattr(app.controller, "list_personal_memory", _personal_memory)
        monkeypatch.setattr(app.controller, "list_research_notes", lambda **kwargs: {"research_notes": []})

        app._show_view("memory")
        root.update()
        app._drain_queues_once()
        app._load_more_memory_entries()
        root.update()
        app._drain_queues_once()

        assert app.memory_render_limit == 8
        assert fetch_limits == [5, 9]

        app._show_view("chat")
        root.update()
        app._drain_queues_once()

        app._show_view("memory")
        root.update()
        app._drain_queues_once()

        assert app.memory_render_limit == 4
        assert app.memory_rendered_count == 4
        assert app.memory_load_more_button is not None
        assert fetch_limits == [5, 9]
    finally:
        root.destroy()


def test_memory_collect_fetches_both_sources_in_order(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    requested_sources: list[str] = []
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        monkeypatch.setattr(
            app.controller,
            "list_personal_memory",
            lambda **kwargs: requested_sources.append("personal_memory") or {
                "personal_memory": [
                    {
                        "title": "Memory",
                        "content": "Note",
                        "created_at": "2026-03-23T12:00:00+00:00",
                        "entry_path": "memory.json",
                    }
                ]
            },
        )
        monkeypatch.setattr(
            app.controller,
            "list_research_notes",
            lambda **kwargs: requested_sources.append("research_notes") or {
                "research_notes": [
                    {
                        "title": "Note",
                        "content": "Research",
                        "created_at": "2026-03-22T12:00:00+00:00",
                        "note_path": "note.json",
                    }
                ]
            },
        )

        entries, has_more = app._collect_memory_entries(archived_only=False, fetch_limit=4)

        assert requested_sources == ["personal_memory", "research_notes"]
        assert has_more is False
        assert [str(item.get("title")) for item in entries] == ["Memory", "Note"]
    finally:
        root.destroy()


def test_memory_collect_passes_bounded_limit_to_both_sources(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        personal_limits: list[int | None] = []
        note_limits: list[int | None] = []

        def _personal_memory(**kwargs):
            personal_limits.append(kwargs.get("limit"))
            return {"personal_memory": []}

        def _research_notes(**kwargs):
            note_limits.append(kwargs.get("limit"))
            return {"research_notes": []}

        monkeypatch.setattr(app.controller, "list_personal_memory", _personal_memory)
        monkeypatch.setattr(app.controller, "list_research_notes", _research_notes)

        entries, has_more = app._collect_memory_entries(archived_only=False, fetch_limit=4)

        assert entries == []
        assert has_more is False
        assert personal_limits == [5]
        assert note_limits == [5]
    finally:
        root.destroy()


def test_memory_view_first_entry_fetches_only_bounded_initial_slice(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    _install_immediate_ui_task_runner(monkeypatch)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        personal_calls: list[int | None] = []
        note_calls: list[int | None] = []

        def _personal_memory(**kwargs):
            limit = kwargs.get("limit")
            personal_calls.append(limit)
            entries = [
                {
                    "title": f"Memory {index}",
                    "content": f"Note {index}",
                    "created_at": f"2026-03-{30 - (index % 20):02d}T12:00:00+00:00",
                    "entry_path": f"memory-{index}.json",
                }
                for index in range(60)
            ]
            return {"personal_memory": entries[:limit] if limit is not None else entries}

        def _research_notes(**kwargs):
            limit = kwargs.get("limit")
            note_calls.append(limit)
            return {"research_notes": []}

        monkeypatch.setattr(app.controller, "list_personal_memory", _personal_memory)
        monkeypatch.setattr(app.controller, "list_research_notes", _research_notes)

        app._show_view("memory")
        root.update()
        app._drain_queues_once()

        assert personal_calls == [5]
        assert note_calls == [5]
        assert len(app.memory_entries) == 4
        assert app.memory_entries_has_more is True
        assert app.memory_rendered_count == 4
    finally:
        root.destroy()


def test_archived_memory_view_uses_bounded_rendering_with_load_more(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    _install_immediate_ui_task_runner(monkeypatch)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        entries = [
            {
                "title": f"Archived {index}",
                "content": f"Note {index}",
                "created_at": "2026-03-23T12:00:00+00:00",
                "kind": "personal_memory",
                "entry_path": f"archived-{index}.json",
            }
            for index in range(38)
        ]
        monkeypatch.setattr(app.controller, "list_personal_memory", lambda **kwargs: {"personal_memory": list(entries)})
        monkeypatch.setattr(app.controller, "list_research_notes", lambda **kwargs: {"research_notes": []})

        app._show_view("archived_memory")
        root.update()
        app._drain_queues_once()

        assert app.archived_memory_load_more_button is not None
        assert app.archived_memory_render_limit == 4
        assert app.archived_memory_rendered_count == 4
        assert app.archived_memory_load_more_button.cget("text") == "Load More"
        assert app.archived_memory_load_more_button.pack_info()["anchor"] == "center"

        app._load_more_archived_memory_entries()
        root.update()
        app._drain_queues_once()

        assert app.archived_memory_render_limit == 8
        assert app.archived_memory_rendered_count == 8
    finally:
        root.destroy()


def test_archived_memory_view_restores_first_window_from_disk_cache(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    _install_immediate_ui_task_runner(monkeypatch)
    cache_path = tmp_path / "data" / "desktop_ui" / "conversation_cache.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_entries = [
        {
            "title": f"Archived Memory {index}",
            "content": f"Cached {index}",
            "created_at": f"2026-03-{30 - index:02d}T12:00:00+00:00",
            "kind": "research_note",
            "note_path": f"archived-note-{index}.json",
            "archived": True,
        }
        for index in range(5)
    ]
    cache_path.write_text(
        json.dumps(
            {
                "recent": [],
                "archived": [],
                "memory": {"entries": [], "signature": [], "has_more": False},
                "archived_memory": {
                    "entries": cache_entries,
                    "signature": [
                        [entry["title"], entry["created_at"], entry["kind"], entry["note_path"]]
                        for entry in cache_entries
                    ],
                    "has_more": True,
                },
            }
        ),
        encoding="utf-8",
    )
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        monkeypatch.setattr(app.controller, "list_personal_memory", lambda **kwargs: {"personal_memory": []})
        monkeypatch.setattr(
            app.controller,
            "list_research_notes",
            lambda **kwargs: {
                "research_notes": [
                    {
                        "title": entry["title"],
                        "created_at": entry["created_at"],
                        "note_path": entry["note_path"],
                    }
                    for entry in cache_entries
                ]
            },
        )
        app._show_view("archived_memory")
        root.update()
        app._drain_queues_once()

        assert app.archived_memory_rendered_count == 4
        assert app.archived_memory_load_more_button is not None
        assert "Loading archived memory..." not in app.archived_memory_preview.get("1.0", "end-1c")
    finally:
        root.destroy()


def test_archived_memory_view_reopen_resets_to_first_four_items(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    _install_immediate_ui_task_runner(monkeypatch)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        fetch_limits: list[int | None] = []

        def _personal_memory(**kwargs):
            fetch_limits.append(kwargs.get("limit"))
            entries = [
                {
                    "title": f"Archived {index}",
                    "content": f"Note {index}",
                    "created_at": "2026-03-23T12:00:00+00:00",
                    "kind": "personal_memory",
                    "entry_path": f"archived-{index}.json",
                }
                for index in range(40)
            ]
            limit = kwargs.get("limit")
            return {"personal_memory": entries[:limit] if limit is not None else entries}

        monkeypatch.setattr(app.controller, "list_personal_memory", _personal_memory)
        monkeypatch.setattr(app.controller, "list_research_notes", lambda **kwargs: {"research_notes": []})

        app._show_view("archived_memory")
        root.update()
        app._drain_queues_once()
        app._load_more_archived_memory_entries()
        root.update()
        app._drain_queues_once()

        assert app.archived_memory_render_limit == 8
        assert fetch_limits == [5, 9]

        app._show_view("chat")
        root.update()
        app._drain_queues_once()

        app._show_view("archived_memory")
        root.update()
        app._drain_queues_once()

        assert app.archived_memory_render_limit == 4
        assert app.archived_memory_rendered_count == 4
        assert app.archived_memory_load_more_button is not None
        assert fetch_limits == [5, 9]
    finally:
        root.destroy()


def test_archived_memory_view_first_entry_fetches_only_bounded_initial_slice(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    _install_immediate_ui_task_runner(monkeypatch)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        personal_calls: list[int | None] = []
        note_calls: list[int | None] = []

        def _personal_memory(**kwargs):
            limit = kwargs.get("limit")
            personal_calls.append(limit)
            entries = [
                {
                    "title": f"Archived memory {index}",
                    "content": f"Note {index}",
                    "created_at": f"2026-03-{30 - (index % 20):02d}T12:00:00+00:00",
                    "entry_path": f"archived-memory-{index}.json",
                }
                for index in range(60)
            ]
            return {"personal_memory": entries[:limit] if limit is not None else entries}

        def _research_notes(**kwargs):
            limit = kwargs.get("limit")
            note_calls.append(limit)
            return {"research_notes": []}

        monkeypatch.setattr(app.controller, "list_personal_memory", _personal_memory)
        monkeypatch.setattr(app.controller, "list_research_notes", _research_notes)

        app._show_view("archived_memory")
        root.update()
        app._drain_queues_once()

        assert personal_calls == [5]
        assert note_calls == [5]
        assert len(app.archived_memory_entries) == 4
        assert app.archived_memory_entries_has_more is True
        assert app.archived_memory_rendered_count == 4
    finally:
        root.destroy()


def test_archived_memory_load_more_suppresses_duplicate_target_window() -> None:
    app = object.__new__(LumenDesktopApp)
    events: list[tuple[str, dict[str, object]]] = []
    starts: list[tuple[bool, int, str]] = []
    app.archived_memory_render_limit = 8
    app.archived_memory_render_step = 4
    app.archived_memory_entries_has_more = True
    app.archived_memory_entries = [{"title": "Archived 1"}]
    app.archived_memory_fetch_in_flight = True
    app.archived_memory_requested_fetch_limit = 12
    app._debug_event = lambda name, **kwargs: events.append((str(name), kwargs))
    app._start_memory_surface_fetch = lambda *, archived, fetch_limit, fetch_reason: starts.append(
        (bool(archived), int(fetch_limit), str(fetch_reason))
    )

    app._load_more_archived_memory_entries()

    assert app.archived_memory_render_limit == 12
    assert starts == []
    assert events == [
        (
            "memory_load_more_suppressed",
            {"view": "archived_memory", "reason": "duplicate_target_window", "target_fetch_limit": 12},
        )
    ]


def test_archived_memory_apply_after_leave_updates_cache_without_visible_render() -> None:
    app = object.__new__(LumenDesktopApp)
    calls: list[str] = []
    events: list[tuple[str, dict[str, object]]] = []
    app.current_view = "chat"
    app.archived_memory_fetch_in_flight = True
    app.archived_memory_requested_version = 5
    app.archived_memory_requested_fetch_limit = 48
    app.archived_memory_state_version = 5
    app.archived_memory_view_dirty = False
    app.archived_memory_entries = []
    app.archived_memory_cached_signature = ()
    app.archived_memory_entries_has_more = False
    app.archived_memory_loaded_version = -1
    app.archived_memory_render_signature = ()
    app._render_archived_memory_from_cache = lambda **kwargs: calls.append("render")
    app._debug_event = lambda name, **kwargs: events.append((str(name), kwargs))
    app._timed_ui_call = lambda label, func: func()

    app._apply_archived_memory_view_result(
        {
            "entries": [{"title": "Archived memory", "entry_path": "archived.json", "created_at": "2026-03-23T12:00:00+00:00"}],
            "signature": (("Archived memory", "archived.json", "2026-03-23T12:00:00+00:00", ""),),
            "has_more_available": False,
            "fetch_limit": 48,
            "fetch_reason": "extended",
        }
    )

    assert calls == []
    assert app.archived_memory_fetch_in_flight is False
    assert app.archived_memory_requested_version is None
    assert app.archived_memory_requested_fetch_limit is None
    assert app.archived_memory_view_dirty is True
    assert len(app.archived_memory_entries) == 1
    assert events == [
        (
            "continuation_result_dropped_after_leave",
            {"view": "archived_memory", "fetch_reason": "extended", "fetch_limit": 48, "fetched_count": 1},
        )
    ]


def test_archived_memory_render_uses_row_cache_without_rebuilding_descriptors(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        app._show_view("archived_memory")
        entries = [
            {
                "title": f"Archived {index}",
                "created_at": "2026-03-23T12:00:00+00:00",
                "kind": "personal_memory",
                "entry_path": f"archived-{index}.json",
                "content": f"Note {index}",
            }
            for index in range(6)
        ]
        (
            app.archived_memory_row_descriptors,
            app.archived_memory_row_entry_map,
            app.archived_memory_row_descriptor_offsets,
            app.archived_memory_row_group_counts,
        ) = build_memory_row_cache(entries)
        app.archived_memory_entries = entries
        app.archived_memory_cached_signature = app._memory_entries_signature(entries)
        app.archived_memory_render_signature = ()
        app.archived_memory_entries_has_more = False
        app.archived_memory_render_limit = 4
        app.archived_memory_rendered_count = 0
        monkeypatch.setattr(
            app,
            "_memory_row_descriptors",
            lambda entries: (_ for _ in ()).throw(AssertionError("row descriptors should come from cache")),
        )

        app._render_archived_memory_from_cache(render_mode="render_from_cached_slice")

        assert app.archived_memory_rendered_count == 4
        assert app.archived_memory_list_inner.winfo_children()
    finally:
        root.destroy()


def test_recent_session_payload_uses_smaller_initial_fetch_limit(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        limits: list[int] = []

        def _list_recent_sessions(*, limit=10, **kwargs):
            limits.append(int(limit))
            return {"sessions": []}

        monkeypatch.setattr(app.controller, "list_recent_sessions", _list_recent_sessions)

        app._build_session_view_payload(archived=False)

        assert limits == [app.RECENT_SESSIONS_FETCH_LIMIT]
    finally:
        root.destroy()


def test_recent_sessions_load_more_appends_without_rebuilding_existing_rows(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    _install_immediate_ui_task_runner(monkeypatch)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        app.controller._recent_sessions = [
            {
                "session_id": f"desktop-{index}",
                "summary": f"Summary {index}",
                "prompt": f"Prompt {index}",
                "mode": "conversation",
                "created_at": "2026-03-23T12:00:00+00:00",
            }
            for index in range(40)
        ]
        calls: list[str] = []
        original_build = app._build_browser_row
        monkeypatch.setattr(
            app,
            "_build_browser_row",
            lambda *args, **kwargs: (calls.append(str(kwargs.get("title") or "")), original_build(*args, **kwargs))[1],
        )

        app._show_view("recent")
        root.update()
        app._drain_queues_once()
        initial_count = len(calls)

        app._load_more_recent_sessions()
        root.update()
        app._drain_queues_once()

        assert initial_count == 4
        assert len(calls) == 8
    finally:
        root.destroy()


def test_pressing_enter_on_display_name_confirms_locally(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        app.display_name_var.set("A" * 30)

        result = app._confirm_display_name()

        assert result == "break"
        assert app.display_name_var.get() == "A" * 20
        assert app.status_var.get() == "Profile name updated"
    finally:
        root.destroy()


def test_input_row_is_hidden_outside_chat_view(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        root.update_idletasks()
        assert app.input_frame.winfo_manager() == "grid"

        app._show_view("recent")
        root.update_idletasks()
        assert app.input_frame.winfo_manager() == ""

        app._show_view("chat")
        root.update_idletasks()
        assert app.input_frame.winfo_manager() == "grid"
    finally:
        root.destroy()


def test_hotbar_can_open_close_and_reopen_without_sticking(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        root.update_idletasks()

        def _settle_hotbar() -> None:
            for _ in range(80):
                if app.hotbar_animation_job is None:
                    break
                time.sleep(0.01)
                root.update()
            assert app.hotbar_animation_job is None

        app._toggle_hotbar()
        _settle_hotbar()
        assert app.hotbar_open is True
        assert app.hotbar_current_width == app.hotbar_target_width
        assert app.hotbar.winfo_manager() == "grid"

        app._handle_global_click(SimpleNamespace(widget=app.content_container))
        _settle_hotbar()
        assert app.hotbar_open is False
        assert app.hotbar_current_width == 0
        assert app.hotbar_transition_in_progress is False
        assert app.hotbar.winfo_manager() == ""

        app._toggle_hotbar()
        _settle_hotbar()
        assert app.hotbar_open is True
        assert app.hotbar_current_width == app.hotbar_target_width
    finally:
        root.destroy()


def test_hotbar_navigation_defers_view_build_until_close_settles(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")

        def _settle_hotbar() -> None:
            for _ in range(80):
                if app.hotbar_animation_job is None:
                    break
                time.sleep(0.01)
                root.update()
            root.update_idletasks()
            assert app.hotbar_animation_job is None

        app._toggle_hotbar()
        _settle_hotbar()
        assert app.hotbar_open is True
        assert "recent" not in app.views

        app._show_view("recent")

        assert app.pending_view_name == "recent"
        assert "recent" not in app.views

        _settle_hotbar()
        root.update()
        root.update_idletasks()

        assert app.pending_view_name is None
        assert app.hotbar_open is False
        assert app.current_view == "recent"
        assert "recent" in app.views
    finally:
        root.destroy()


def test_hotbar_destination_click_closes_immediately_before_switch(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        app._toggle_hotbar()

        for _ in range(80):
            if app.hotbar_animation_job is None:
                break
            time.sleep(0.01)
            root.update()

        assert app.hotbar_open is True

        app._handle_hotbar_destination("settings")
        root.update()
        root.update_idletasks()

        assert app.hotbar_animation_job is None
        assert app.hotbar_open is False
        assert app.current_view == "settings"
    finally:
        root.destroy()


def test_deferred_view_refresh_applies_palette_once_after_dirty_refresh(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        app.current_view = "recent"
        app.deferred_view_refresh_target = "recent"
        app.recent_sessions_view_dirty = True
        calls: list[str] = []

        monkeypatch.setattr(app, "_apply_palette_to_view", lambda view_name: calls.append(f"palette:{view_name}"))

        def _refresh() -> None:
            calls.append("refresh")
            app.recent_sessions_view_dirty = False

        monkeypatch.setattr(app, "_refresh_recent_sessions_view", _refresh)

        app._run_deferred_view_refresh()

        assert calls == ["refresh", "palette:recent"]
    finally:
        root.destroy()


def test_toggle_scrollbar_visibility_defers_without_forcing_layout() -> None:
    app = object.__new__(LumenDesktopApp)
    scheduled: list[str] = []
    app.root = SimpleNamespace(after_idle=lambda callback: scheduled.append("idle") or "job-1")
    canvas = SimpleNamespace()
    scrollbar = SimpleNamespace(winfo_exists=lambda: True)

    app._toggle_scrollbar_visibility(canvas, scrollbar, defer=True)
    app._toggle_scrollbar_visibility(canvas, scrollbar, defer=True)

    assert scheduled == ["idle"]
    assert getattr(canvas, "_lumen_scrollbar_job", None) == "job-1"


def test_chat_canvas_configure_coalesces_follow_up_layout_work() -> None:
    app = object.__new__(LumenDesktopApp)
    callbacks: list[callable] = []
    calls: list[str] = []
    app._chat_canvas_layout_job = None
    app._chat_canvas_layout_width = None
    app._chat_canvas_layout_needs_scrollregion = False
    app.root = SimpleNamespace(after_idle=lambda callback: callbacks.append(callback) or "job-1")
    app.chat_canvas = SimpleNamespace(
        itemconfigure=lambda *args, **kwargs: calls.append("item"),
        winfo_width=lambda: 900,
        configure=lambda **kwargs: calls.append("scrollregion"),
        bbox=lambda *args, **kwargs: (0, 0, 0, 0),
    )
    app.chat_window = object()
    app.chat_scrollbar = SimpleNamespace(winfo_exists=lambda: True, winfo_manager=lambda: "", grid=lambda **kwargs: calls.append("grid"), grid_remove=lambda: calls.append("remove"))
    app._toggle_scrollbar_visibility = lambda *args, **kwargs: calls.append("scrollbar")
    app._refresh_message_wraps = lambda: calls.append("wraps")
    app._refresh_landing_icon_geometry = lambda width=None: calls.append(f"icon:{width}")

    app._on_chat_canvas_configure(SimpleNamespace(width=800))
    app._on_chat_canvas_configure(SimpleNamespace(width=900))

    assert calls == []
    assert len(callbacks) == 1

    callbacks[0]()

    assert calls == ["item", "scrollbar", "wraps", "icon:900"]


def test_non_chat_canvas_layout_coalesces_to_one_shell_flush() -> None:
    app = object.__new__(LumenDesktopApp)
    callbacks: list[callable] = []
    calls: list[str] = []
    app.root = SimpleNamespace(after_idle=lambda callback: callbacks.append(callback) or "job-1")
    app._debug_event = lambda *args, **kwargs: None
    app.chat_canvas = None
    canvas = SimpleNamespace(
        winfo_exists=lambda: True,
        itemconfigure=lambda *args, **kwargs: calls.append("item"),
        configure=lambda **kwargs: calls.append("scrollregion"),
        bbox=lambda *args, **kwargs: (0, 0, 0, 0),
    )
    canvas._lumen_layout_surface = "recent"
    canvas._lumen_layout_window_id = object()
    canvas._lumen_layout_scrollbar = SimpleNamespace(
        winfo_exists=lambda: True,
        winfo_manager=lambda: "",
        grid=lambda **kwargs: calls.append("grid"),
        grid_remove=lambda: calls.append("remove"),
    )
    app._toggle_scrollbar_visibility = lambda *args, **kwargs: calls.append("scrollbar")

    app._request_canvas_layout(canvas, width=700, needs_scrollregion=True)
    app._request_canvas_layout(canvas, width=820, needs_scrollregion=True)

    assert len(callbacks) == 1
    assert calls == []

    callbacks[0]()

    assert calls == ["item", "scrollregion", "scrollbar"]


def test_startup_followup_coalesces_multiple_requests() -> None:
    app = object.__new__(LumenDesktopApp)
    callbacks: list[callable] = []
    calls: list[str] = []
    app.root = SimpleNamespace(after_idle=lambda callback: callbacks.append(callback) or "job-1")
    app._startup_followup_job = None
    app._startup_followup_needs_post_bootstrap = False
    app._startup_followup_needs_view_refresh = False
    app._startup_followup_needs_background_tasks = False
    app._debug_event = lambda *args, **kwargs: None
    app._post_startup_bootstrap = lambda: calls.append("post")
    app._refresh_loaded_views_after_startup = lambda: calls.append("refresh")
    app._start_background_startup_tasks = lambda: calls.append("background")

    app._schedule_startup_followup(post_bootstrap=True)
    app._schedule_startup_followup(refresh_views=True, background_tasks=True)

    assert len(callbacks) == 1
    callbacks[0]()

    assert calls == ["post", "refresh", "background"]
    assert app._startup_followup_job is None


def test_assistant_reveal_uses_incremental_inserts_without_repeated_delete() -> None:
    app = object.__new__(LumenDesktopApp)
    callbacks: list[callable] = []
    configure_calls: list[tuple[object, ...]] = []
    delete_calls: list[str] = []
    inserts: list[str] = []
    app.current_palette = {"text_secondary": "#999", "text_primary": "#fff", "text_muted": "#888"}
    app.root = SimpleNamespace(after=lambda delay, callback: callbacks.append(callback) or f"job-{len(callbacks)}")
    app.message_reveal_jobs = set()
    app._bubble_pady = lambda: 10
    app._debug_event = lambda *args, **kwargs: None
    bubble = SimpleNamespace(
        winfo_exists=lambda: True,
        configure=lambda **kwargs: configure_calls.append(("bubble", kwargs)),
    )
    meta_label = SimpleNamespace(
        winfo_exists=lambda: True,
        configure=lambda **kwargs: configure_calls.append(("meta", kwargs)),
    )
    text_widget = SimpleNamespace(
        winfo_exists=lambda: True,
        configure=lambda **kwargs: configure_calls.append(("text", kwargs)),
        delete=lambda *args, **kwargs: delete_calls.append("delete"),
        insert=lambda *args, **kwargs: inserts.append(str(args[1])),
        _lumen_text_fg="#fff",
    )
    message = SimpleNamespace(sender="Lumen", text="x" * 200, message_type="assistant", timestamp="")

    app._animate_message_presence(message=message, bubble=bubble, text_widget=text_widget, meta_label=meta_label)

    while callbacks:
        callback = callbacks.pop(0)
        callback()

    assert delete_calls == ["delete"]
    assert len(inserts) > 1
    assert "".join(inserts) == "x" * 200
    assert app.message_reveal_jobs == set()


def test_assistant_reveal_requests_live_scroll_as_text_expands() -> None:
    app = object.__new__(LumenDesktopApp)
    callbacks: list[callable] = []
    scroll_calls: list[str] = []
    app.current_palette = {"text_secondary": "#999", "text_primary": "#fff", "text_muted": "#888"}
    app.root = SimpleNamespace(after=lambda delay, callback: callbacks.append(callback) or f"job-{len(callbacks)}")
    app.message_reveal_jobs = set()
    app.chat_canvas = SimpleNamespace(winfo_exists=lambda: True)
    app._scroll_chat_to_bottom = lambda: scroll_calls.append("scroll")
    app._bubble_wraplength = lambda: 420
    app._bubble_pady = lambda: 10
    app._debug_event = lambda *args, **kwargs: None
    bubble = SimpleNamespace(winfo_exists=lambda: True, configure=lambda **kwargs: None)
    meta_label = SimpleNamespace(winfo_exists=lambda: True, configure=lambda **kwargs: None)
    text_widget = SimpleNamespace(
        winfo_exists=lambda: True,
        configure=lambda **kwargs: None,
        delete=lambda *args, **kwargs: None,
        insert=lambda *args, **kwargs: None,
        _lumen_text_fg="#fff",
    )
    message = SimpleNamespace(sender="Lumen", text="y" * 260, message_type="assistant", timestamp="")

    app._animate_message_presence(message=message, bubble=bubble, text_widget=text_widget, meta_label=meta_label)

    while callbacks:
        callbacks.pop(0)()

    assert len(scroll_calls) >= 2


def test_assistant_message_without_animation_renders_text_immediately() -> None:
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    app = object.__new__(LumenDesktopApp)
    try:
        app.messages = []
        app.current_palette = DARK_PALETTE
        app.custom_colors = {}
        app.chat_frame = tk.Frame(root, bg=DARK_PALETTE["app_bg"])
        app.chat_frame.pack()
        app.message_text_widgets = []
        app.message_labels = []
        app.pending_row = None
        app.pending_text_widget = None
        app._row_pady = lambda: 0
        app._bubble_padx = lambda: 8
        app._bubble_pady = lambda: 8
        app._bubble_side_padding = lambda: 0
        app._bubble_wraplength = lambda: 420
        app._meta_font = lambda: ("Arial", 8)
        app._message_font = lambda: ("Arial", 10)
        app._scroll_chat_to_bottom = lambda: None

        app._render_chat_message(
            DesktopChatMessage(
                sender="Lumen",
                text="Natural assistant reply.",
                message_type="assistant",
                timestamp="",
            ),
            store_message=True,
            auto_scroll=False,
            animate=False,
        )

        assert app.message_text_widgets
        rendered = app.message_text_widgets[-1].get("1.0", "end").strip()
        assert rendered == "Natural assistant reply."
    finally:
        try:
            root.destroy()
        except tk.TclError:
            pass


def test_scroll_chat_to_bottom_defers_without_root_layout() -> None:
    app = object.__new__(LumenDesktopApp)
    callbacks: list[callable] = []
    calls: list[str] = []
    app._chat_scroll_to_bottom_job = None
    app._debug_event = lambda *args, **kwargs: None
    app.root = SimpleNamespace(after_idle=lambda callback: callbacks.append(callback) or "job-1")
    app.chat_canvas = SimpleNamespace(
        winfo_exists=lambda: True,
        yview_moveto=lambda value: calls.append(f"scroll:{value}"),
    )

    app._scroll_chat_to_bottom()
    app._scroll_chat_to_bottom()

    assert len(callbacks) == 1
    assert calls == []

    callbacks[0]()

    assert calls == ["scroll:1.0"]


def test_mousewheel_scroll_coalesces_per_scrollable() -> None:
    app = object.__new__(LumenDesktopApp)
    callbacks: list[callable] = []
    calls: list[str] = []
    app._mousewheel_flush_jobs = {}
    app._debug_event = lambda *args, **kwargs: None
    app.root = SimpleNamespace(after_idle=lambda callback: callbacks.append(callback) or "job-1")
    scrollable = SimpleNamespace(
        _lumen_layout_surface="recent",
        _lumen_pending_scroll_units=0,
        winfo_exists=lambda: True,
        yview_scroll=lambda units, kind: calls.append(f"{units}:{kind}"),
    )

    app._queue_mousewheel_scroll(scrollable, 1)
    app._queue_mousewheel_scroll(scrollable, 2)

    assert len(callbacks) == 1
    assert calls == []

    callbacks[0]()

    assert calls == ["3:units"]
    assert app._mousewheel_flush_jobs == {}


def test_browser_row_pointer_containment_keeps_hover_active() -> None:
    app = object.__new__(LumenDesktopApp)
    child = SimpleNamespace(master=None)
    row = SimpleNamespace(
        winfo_exists=lambda: True,
        winfo_pointerxy=lambda: (10, 12),
        winfo_containing=lambda x, y: child,
    )
    child.master = row

    assert app._browser_row_contains_pointer(row) is True


def test_browser_hover_ownership_allows_only_one_active_row_per_surface() -> None:
    app = object.__new__(LumenDesktopApp)
    app._active_browser_hover_rows = {"recent": None}
    app._debug_event = lambda *args, **kwargs: None
    styles: list[tuple[str, bool]] = []
    app._style_browser_row = lambda row, hovered: styles.append((row.name, hovered))
    row_a = SimpleNamespace(name="a", winfo_exists=lambda: True)
    row_b = SimpleNamespace(name="b", winfo_exists=lambda: True)

    app._set_active_browser_hover_row("recent", row_a)
    app._set_active_browser_hover_row("recent", row_b)

    assert app._active_browser_hover_rows["recent"] is row_b
    assert styles == [("a", True), ("a", False), ("b", True)]


def test_clear_stale_browser_hover_clears_only_when_pointer_left() -> None:
    app = object.__new__(LumenDesktopApp)
    row = SimpleNamespace(winfo_exists=lambda: True)
    app._active_browser_hover_rows = {"memory": row}
    app._debug_event = lambda *args, **kwargs: None
    cleared: list[object | None] = []
    app._set_active_browser_hover_row = lambda surface, selected: cleared.append(selected)
    app._browser_row_contains_pointer = lambda current: False

    app._clear_stale_browser_hover("memory")

    assert cleared == [None]


def test_write_clipboard_does_not_force_root_update(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        monkeypatch.setattr(root, "update", lambda: (_ for _ in ()).throw(AssertionError("root.update should not run")))

        app._write_clipboard("clipboard text")

        assert root.clipboard_get() == "clipboard text"
    finally:
        root.destroy()


def test_load_session_restores_transcript_without_assistant_reveal_churn(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        reveal_calls: list[str] = []
        scroll_calls: list[str] = []
        original_render = app._render_chat_message
        monkeypatch.setattr(app, "_animate_message_presence", lambda **kwargs: reveal_calls.append("animate"))
        monkeypatch.setattr(app, "_scroll_chat_to_bottom", lambda: scroll_calls.append("scroll"))
        monkeypatch.setattr(app, "_scroll_restored_chat_to_reading_position", lambda: scroll_calls.append("restore-scroll"))

        def _render_with_trace(message, *, store_message, auto_scroll=True, animate=True):
            scroll_calls.append(f"render:{message.message_type}:{auto_scroll}:{animate}")
            return original_render(message, store_message=store_message, auto_scroll=auto_scroll, animate=animate)

        monkeypatch.setattr(app, "_render_chat_message", _render_with_trace)

        app._load_session("desktop-1")

        assert reveal_calls == []
        assert any(item == "render:user:False:False" for item in scroll_calls)
        assert any(item == "render:assistant:False:False" for item in scroll_calls)
        assert scroll_calls[-1] == "restore-scroll"
    finally:
        root.destroy()


def test_load_session_restores_full_transcript_without_chat_history_load_more(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        records = [
            {
                "prompt": f"question {index}",
                "summary": f"answer {index}",
                "mode": "research",
                "created_at": f"2026-03-{10 + index:02d}T12:00:00+00:00",
                "response": {"mode": "research", "summary": f"answer {index}"},
            }
            for index in range(6)
        ]
        monkeypatch.setattr(app.controller, "list_interactions", lambda **kwargs: {"interaction_records": records})

        app._load_session("desktop-1")

        assert len(app.messages) == 12
        rendered_texts = [message.text for message in app.messages]
        assert rendered_texts[:2] == ["question 0", "answer 0"]
        assert "answer 5" in rendered_texts
    finally:
        root.destroy()


def test_load_session_restores_transcript_from_decorated_runtime_shape(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        records = [
            {
                "prompt": "",
                "prompt_view": {
                    "canonical_prompt": "decorated question",
                    "original_prompt": "decorated question",
                    "resolved_prompt": "",
                    "rewritten": False,
                },
                "summary": "decorated answer",
                "mode": "research",
                "created_at": "2026-03-10T12:00:00+00:00",
                "response": {},
            }
        ]
        monkeypatch.setattr(app.controller, "list_interactions", lambda **kwargs: {"interaction_records": records})

        app._load_session("desktop-1")

        rendered_texts = [message.text for message in app.messages if message.message_type in {"user", "assistant"}]
        assert rendered_texts == ["decorated question", "decorated answer"]
    finally:
        root.destroy()


def test_load_session_keeps_summary_fallback_out_of_open_chat_title(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        app.recent_sessions_cache = [
            {
                "session_id": "desktop-1",
                "title": "",
                "prompt": "",
                "summary": "Summary only title",
                "created_at": "2026-03-23T12:00:00+00:00",
            }
        ]
        records = [
            {
                "prompt": "decorated question",
                "summary": "decorated answer",
                "mode": "research",
                "created_at": "2026-03-10T12:00:00+00:00",
                "response": {"summary": "decorated answer"},
            }
        ]
        monkeypatch.setattr(app.controller, "list_interactions", lambda **kwargs: {"interaction_records": records})

        app._load_session("desktop-1")

        assert app.chat_title_var.get() == "Summary only title"
    finally:
        root.destroy()


def test_load_session_partial_restore_keeps_usable_messages_out_of_landing_state(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        records = [
            {
                "prompt": "good question",
                "summary": "good answer",
                "mode": "research",
                "created_at": "2026-03-10T12:00:00+00:00",
                "response": {"summary": "good answer"},
            },
            {
                "prompt": "broken question",
                "mode": "research",
                "created_at": "2026-03-11T12:00:00+00:00",
                "response": {"summary": "broken answer"},
            },
        ]
        monkeypatch.setattr(app.controller, "list_interactions", lambda **kwargs: {"interaction_records": records})

        original_helper = app._saved_chat_assistant_text

        def _patched_saved_text(record, *, response):
            if str(record.get("prompt") or "") == "broken question":
                raise RuntimeError("bad saved record")
            return original_helper(record, response=response)

        monkeypatch.setattr(app, "_saved_chat_assistant_text", _patched_saved_text)

        app._load_session("desktop-1")

        rendered_texts = [message.text for message in app.messages if message.message_type in {"user", "assistant"}]
        assert rendered_texts == ["good question", "good answer", "broken question"]
        assert any(
            "could not be rendered" in message.text.lower()
            for message in app.messages
            if message.message_type == "system"
        )
        assert not bool(app.landing_frame.place_info())
    finally:
        root.destroy()


def test_load_session_rejects_nonrestorable_recent_row_instead_of_showing_landing_screen(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        app.recent_sessions_cache = [
            {
                "session_id": "desktop-1",
                "title": "",
                "prompt": "",
                "summary": "ghost row",
                "created_at": "2026-03-23T12:00:00+00:00",
            }
        ]
        app.recent_sessions_rows = grouped_session_rows(app.recent_sessions_cache)
        views: list[str] = []
        monkeypatch.setattr(app.controller, "list_interactions", lambda **kwargs: {"interaction_records": []})
        monkeypatch.setattr(app, "_show_view", lambda view_name: views.append(view_name))

        app._load_session("desktop-1")

        assert app.status_var.get() == "Saved chat is no longer available"
        assert app.recent_sessions_cache == []
        assert views[-1] == "recent"
    finally:
        root.destroy()


def test_rerender_messages_uses_batch_history_path_without_animation() -> None:
    app = object.__new__(LumenDesktopApp)
    children = [SimpleNamespace(destroy=lambda: None), SimpleNamespace(destroy=lambda: None)]
    rendered: list[tuple[str, bool, bool]] = []
    debug_events: list[str] = []
    scroll_calls: list[str] = []
    app.messages = [
        SimpleNamespace(message_type="user", text="Hi"),
        SimpleNamespace(message_type="assistant", text="Hello"),
    ]
    app.chat_frame = SimpleNamespace(winfo_children=lambda: children)
    app.pending_row = object()
    app.message_labels = [object()]
    app.message_text_widgets = [object()]
    app._debug_event = lambda *args, **kwargs: debug_events.append(str(args[0]))
    app._scroll_chat_to_bottom = lambda: scroll_calls.append("scroll")
    app._render_chat_message = lambda message, *, store_message, auto_scroll=True, animate=True: rendered.append(
        (str(message.message_type), bool(auto_scroll), bool(animate))
    )
    app._timed_ui_call = lambda label, func: func()

    app._rerender_messages()

    assert debug_events == ["message_rerender_batch"]
    assert rendered == [("user", False, False), ("assistant", False, False)]
    assert scroll_calls == ["scroll"]


def test_memory_rows_reuse_widgets_when_entries_unchanged(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        app._show_view("memory")
        entries = [
            {
                "title": "Preference",
                "created_at": "2026-03-23T12:00:00+00:00",
                "kind": "personal_memory",
                "entry_path": "pref.json",
                "content": "Remember this.",
            }
        ]

        app._rebuild_memory_rows(
            entries=entries,
            inner=app.memory_list_inner,
            command_builder=app._show_memory_entry,
            archive_enabled=True,
        )
        first_widgets = list(app.memory_list_inner.winfo_children())

        app._rebuild_memory_rows(
            entries=entries,
            inner=app.memory_list_inner,
            command_builder=app._show_memory_entry,
            archive_enabled=True,
        )
        second_widgets = list(app.memory_list_inner.winfo_children())

        assert second_widgets == first_widgets
    finally:
        root.destroy()


def test_memory_load_more_appends_without_rebuilding_unchanged_rows(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    _install_immediate_ui_task_runner(monkeypatch)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        entries = [
            {
                "title": f"Memory {index}",
                "content": f"Note {index}",
                "created_at": "2026-03-23T12:00:00+00:00",
                "entry_path": f"memory-{index}.json",
            }
            for index in range(40)
        ]
        monkeypatch.setattr(app.controller, "list_personal_memory", lambda **kwargs: {"personal_memory": list(entries)})
        monkeypatch.setattr(app.controller, "list_research_notes", lambda **kwargs: {"research_notes": []})

        app._show_view("memory")
        root.update()
        app._drain_queues_once()
        first_widgets = [
            widget
            for widget in app.memory_list_inner.winfo_children()
            if widget is not app.memory_load_more_button
        ]

        app._load_more_memory_entries()
        root.update()
        app._drain_queues_once()
        second_widgets = list(app.memory_list_inner.winfo_children())

        assert second_widgets[: len(first_widgets)] == first_widgets
        assert len(second_widgets) > len(first_widgets)
    finally:
        root.destroy()


def test_memory_load_more_extends_fetch_window_progressively(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    _install_immediate_ui_task_runner(monkeypatch)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        limits: list[int | None] = []

        def _personal_memory(**kwargs):
            limit = kwargs.get("limit")
            limits.append(limit)
            entries = [
                {
                    "title": f"Memory {index}",
                    "content": f"Note {index}",
                    "created_at": f"2026-03-{30 - (index % 20):02d}T12:00:00+00:00",
                    "entry_path": f"memory-{index}.json",
                }
                for index in range(60)
            ]
            return {"personal_memory": entries[:limit] if limit is not None else entries}

        monkeypatch.setattr(app.controller, "list_personal_memory", _personal_memory)
        monkeypatch.setattr(app.controller, "list_research_notes", lambda **kwargs: {"research_notes": []})

        app._show_view("memory")
        root.update()
        app._drain_queues_once()

        assert limits == [5]
        assert app.memory_rendered_count == 4

        app._load_more_memory_entries()
        root.update()
        app._drain_queues_once()

        assert limits == [5, 9]
        assert len(app.memory_entries) == 8
        assert app.memory_entries_has_more is True
        assert app.memory_rendered_count == 8
    finally:
        root.destroy()


def test_memory_load_more_suppresses_duplicate_target_window() -> None:
    app = object.__new__(LumenDesktopApp)
    events: list[tuple[str, dict[str, object]]] = []
    starts: list[tuple[bool, int, str]] = []
    app.memory_render_limit = 8
    app.memory_render_step = 4
    app.memory_entries_has_more = True
    app.memory_entries = [{"title": "Memory 1"}]
    app.memory_fetch_in_flight = True
    app.memory_requested_fetch_limit = 12
    app._debug_event = lambda name, **kwargs: events.append((str(name), kwargs))
    app._start_memory_surface_fetch = lambda *, archived, fetch_limit, fetch_reason: starts.append(
        (bool(archived), int(fetch_limit), str(fetch_reason))
    )

    app._load_more_memory_entries()

    assert app.memory_render_limit == 12
    assert starts == []
    assert events == [
        (
            "memory_load_more_suppressed",
            {"view": "memory", "reason": "duplicate_target_window", "target_fetch_limit": 12},
        )
    ]


def test_memory_surface_reentry_renders_cached_slice_before_refresh(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    _install_immediate_ui_task_runner(monkeypatch)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        entries = [
            {
                "title": f"Memory {index}",
                "content": f"Note {index}",
                "created_at": "2026-03-23T12:00:00+00:00",
                "entry_path": f"memory-{index}.json",
            }
            for index in range(30)
        ]
        monkeypatch.setattr(
            app.controller,
            "list_personal_memory",
            lambda **kwargs: {"personal_memory": entries[: kwargs.get("limit")]},
        )
        monkeypatch.setattr(app.controller, "list_research_notes", lambda **kwargs: {"research_notes": []})

        events: list[str] = []
        app._debug_event = lambda *args, **kwargs: events.append(str(args[0]))

        app._show_view("memory")
        root.update()
        app._drain_queues_once()

        app._show_view("recent")
        root.update()
        app._drain_queues_once()

        app._show_view("memory")
        root.update()
        app._drain_queues_once()

        assert "surface_teardown" in events
        assert "surface_reenter_from_cache" in events
        assert "render_from_cached_slice" in events
    finally:
        root.destroy()


def test_memory_render_uses_row_cache_without_rebuilding_descriptors(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        app._show_view("memory")
        entries = [
            {
                "title": f"Memory {index}",
                "created_at": "2026-03-23T12:00:00+00:00",
                "kind": "personal_memory",
                "entry_path": f"memory-{index}.json",
                "content": f"Note {index}",
            }
            for index in range(6)
        ]
        (
            app.memory_row_descriptors,
            app.memory_row_entry_map,
            app.memory_row_descriptor_offsets,
            app.memory_row_group_counts,
        ) = build_memory_row_cache(entries)
        app.memory_entries = entries
        app.memory_cached_signature = app._memory_entries_signature(entries)
        app.memory_render_signature = ()
        app.memory_entries_has_more = False
        app.memory_render_limit = 4
        app.memory_rendered_count = 0
        monkeypatch.setattr(
            app,
            "_memory_row_descriptors",
            lambda entries: (_ for _ in ()).throw(AssertionError("row descriptors should come from cache")),
        )

        app._render_memory_entries_from_cache(archived=False, render_mode="render_from_cached_slice")

        assert app.memory_rendered_count == 4
        assert app.memory_list_inner.winfo_children()
    finally:
        root.destroy()


def test_memory_apply_after_leave_updates_cache_without_visible_render() -> None:
    app = object.__new__(LumenDesktopApp)
    calls: list[str] = []
    events: list[tuple[str, dict[str, object]]] = []
    app.current_view = "chat"
    app.memory_fetch_in_flight = True
    app.memory_requested_fetch_limit = 48
    app.memory_view_dirty = False
    app.memory_entries = []
    app.memory_cached_signature = ()
    app.memory_entries_has_more = False
    app._render_memory_entries_from_cache = lambda **kwargs: calls.append("render")
    app._debug_event = lambda name, **kwargs: events.append((str(name), kwargs))
    app._timed_ui_call = lambda label, func: func()

    app._apply_memory_view_result(
        {
            "entries": [{"title": "Memory", "entry_path": "memory.json", "created_at": "2026-03-23T12:00:00+00:00"}],
            "signature": (("Memory", "memory.json", "2026-03-23T12:00:00+00:00", ""),),
            "has_more_available": False,
            "fetch_limit": 48,
            "fetch_reason": "extended",
        }
    )

    assert calls == []
    assert app.memory_fetch_in_flight is False
    assert app.memory_requested_fetch_limit is None
    assert app.memory_view_dirty is True
    assert len(app.memory_entries) == 1
    assert events == [
        (
            "continuation_result_dropped_after_leave",
            {"view": "memory", "fetch_reason": "extended", "fetch_limit": 48, "fetched_count": 1},
        )
    ]


def test_memory_apply_builds_row_cache_for_reuse() -> None:
    app = object.__new__(LumenDesktopApp)
    calls: list[str] = []
    app.current_view = "memory"
    app.memory_fetch_in_flight = True
    app.memory_requested_fetch_limit = 4
    app.memory_view_dirty = True
    app.memory_entries = []
    app.memory_cached_signature = ()
    app.memory_entries_has_more = False
    app.memory_render_signature = ()
    app._timed_ui_call = lambda label, func: func()
    app._render_memory_entries_from_cache = lambda **kwargs: calls.append("render")

    app._apply_memory_view_result(
        {
            "entries": [
                {
                    "title": "Memory",
                    "entry_path": "memory.json",
                    "created_at": "2026-03-23T12:00:00+00:00",
                    "kind": "personal_memory",
                }
            ],
            "signature": (("Memory", "2026-03-23T12:00:00+00:00", "personal_memory", "memory.json"),),
            "has_more_available": False,
            "fetch_limit": 4,
            "fetch_reason": "bounded",
        }
    )

    assert calls == ["render"]
    assert app.memory_row_descriptors
    assert app.memory_row_descriptor_offsets == (0, 2)
    assert app.memory_row_group_counts == (0, 1)


def test_load_more_button_uses_active_accent_palette() -> None:
    palette = custom_accent_palette("#33aa77")

    style = resolve_load_more_palette(palette=palette)

    assert style["bg"] == palette["nav_active_border"]
    assert style["fg"] == palette["text_primary"]
    assert style["activebackground"] == palette["nav_active_border"]


def test_archived_memory_rows_rebuild_when_entries_change(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        app._show_view("archived_memory")
        first_entries = [
            {
                "title": "Preference",
                "created_at": "2026-03-23T12:00:00+00:00",
                "kind": "personal_memory",
                "entry_path": "pref.json",
                "content": "Remember this.",
            }
        ]
        second_entries = [
            {
                "title": "Preference",
                "created_at": "2026-03-23T12:00:00+00:00",
                "kind": "personal_memory",
                "entry_path": "pref.json",
                "content": "Remember this.",
            },
            {
                "title": "Research note",
                "created_at": "2026-03-22T12:00:00+00:00",
                "kind": "research_note",
                "note_path": "note.md",
                "content": "Investigate this.",
            },
        ]

        app._rebuild_memory_rows(
            entries=first_entries,
            inner=app.archived_memory_list_inner,
            command_builder=app._show_archived_memory_entry,
            archive_enabled=False,
        )
        first_widgets = list(app.archived_memory_list_inner.winfo_children())

        app._rebuild_memory_rows(
            entries=second_entries,
            inner=app.archived_memory_list_inner,
            command_builder=app._show_archived_memory_entry,
            archive_enabled=False,
        )
        second_widgets = list(app.archived_memory_list_inner.winfo_children())

        assert second_widgets != first_widgets
        assert len(second_widgets) > len(first_widgets)
    finally:
        root.destroy()


def test_recent_resize_layout_does_not_rerender_rows_when_signature_unchanged(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    _install_immediate_ui_task_runner(monkeypatch)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        app.controller._recent_sessions = [
            {
                "session_id": f"desktop-{index}",
                "summary": f"Summary {index}",
                "prompt": f"Prompt {index}",
                "mode": "conversation",
                "created_at": "2026-03-23T12:00:00+00:00",
            }
            for index in range(10)
        ]
        app._show_view("recent")
        root.update()
        app._drain_queues_once()

        calls: list[str] = []
        monkeypatch.setattr(app, "_render_recent_sessions_from_cache", lambda: calls.append("render"))

        app._sync_canvas_layout(app.recent_list_canvas, app.recent_sessions_scrollbar, window_id=app.recent_list_window, width=700)
        root.update_idletasks()

        assert calls == []
    finally:
        root.destroy()


def test_mic_button_uses_accent_for_disabled_foreground(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())

        app.theme_var.set("Light")
        app._on_theme_changed()
        assert app.mic_button.cget("disabledforeground") == app.current_palette["nav_active_border"]

        app.theme_var.set("Dark")
        app._on_theme_changed()
        assert app.mic_button.cget("disabledforeground") == app.current_palette["nav_active_border"]

        app.theme_var.set("Custom")
        app._on_theme_changed()
        assert app.mic_button.cget("disabledforeground") == app.current_palette["nav_active_border"]
    finally:
        root.destroy()


def test_light_mode_side_views_use_panel_surfaces_for_empty_states(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    _install_immediate_ui_task_runner(monkeypatch)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        app.controller._recent_sessions = []
        app.controller.archived_memory_paths = [
            str(app.controller.settings.data_root / "personal_memory" / "desktop-1" / "pref.json"),
            str(app.controller.settings.data_root / "personal_memory" / "desktop-1" / "project.json"),
            str(app.controller.settings.data_root / "research_notes" / "desktop-1" / "gravity.json"),
            str(app.controller.settings.data_root / "research_notes" / "desktop-1" / "orbit.json"),
        ]

        app.theme_var.set("Light")
        app._on_theme_changed()
        app._show_view("recent")
        root.update()
        app._drain_queues_once()

        recent_empty = [
            child for child in app.recent_list_inner.winfo_children()
            if isinstance(child, tk.Label)
        ][0]
        assert recent_empty.cget("bg") == app.current_palette["panel_bg"]
        assert recent_empty.cget("fg") == app.current_palette["text_muted"]

        app._show_view("archived_memory")
        root.update_idletasks()
        assert app.archived_memory_list_canvas.cget("bg") == app.current_palette["panel_bg"]
        assert app.archived_memory_list_inner.cget("bg") == app.current_palette["panel_bg"]
        assert app.archived_memory_preview.cget("bg") == app.current_palette["panel_bg"]
    finally:
        root.destroy()


def test_settings_row_click_triggers_help_toggle(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        app._show_view("settings")
        help_label = next(
            label for label in app.settings_row_labels
            if label.cget("text") == "Help"
        )

        assert help_label.bind("<Button-1>")
        app._toggle_settings_help()
        root.update_idletasks()

        assert app.help_text.winfo_manager() == "grid"
    finally:
        root.destroy()


def test_settings_theme_row_click_opens_theme_control(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        app._show_view("settings")
        opened: list[str] = []
        monkeypatch.setattr(app, "_show_settings_popup", lambda kind, anchor=None: opened.append(kind))
        theme_label = next(
            label for label in app.settings_row_labels
            if label.cget("text") == "Theme"
        )

        assert theme_label.bind("<Button-1>")
        app._activate_theme_row()

        assert opened == ["theme"]
    finally:
        root.destroy()


def test_settings_theme_row_masks_right_side_value_artifacts(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        app._show_view("settings")

        assert app.theme_value_mask.winfo_manager() == "grid"
        assert app.theme_value_mask.cget("bg") == app.current_palette["app_bg"]
        assert app.theme_value_label.winfo_manager() == ""
    finally:
        root.destroy()


def test_settings_popup_reuses_single_instance_for_same_kind(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        app._show_view("settings")

        app._show_settings_popup("theme")
        first_popup = app.settings_popup
        assert first_popup is not None

        app._show_settings_popup("theme")

        assert app.settings_popup is first_popup
        assert app.settings_popup_kind == "theme"
    finally:
        root.destroy()


def test_settings_help_and_lumen_text_color_rows_close_popup_before_owning_control(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    monkeypatch.setattr("lumen.desktop.chat_app.colorchooser.askcolor", lambda *args, **kwargs: ((17, 34, 51), "#112233"))
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        app._show_view("settings")

        app._show_settings_popup("theme")
        assert app.settings_popup is not None

        app._toggle_settings_help()
        root.update_idletasks()

        assert app.settings_popup is None
        assert app.help_text.winfo_manager() == "grid"

        app._show_settings_popup("theme")
        assert app.settings_popup is not None

        app._pick_color("assistant_text")
        root.update_idletasks()

        assert app.settings_popup is None
        assert app.color_value_labels["assistant_text"].cget("text") == "#112233"
    finally:
        root.destroy()


def test_deferred_view_build_uses_targeted_palette_application(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        calls: list[str] = []
        original_apply_view = app._apply_palette_to_view
        monkeypatch.setattr(app, "_apply_palette_to_shell", lambda: calls.append("shell"))
        monkeypatch.setattr(app, "_apply_palette_to_view", lambda view_name: (calls.append(view_name), original_apply_view(view_name))[1])

        app._show_view("recent")
        root.update_idletasks()

        assert "shell" not in calls
        assert "recent" in calls
    finally:
        root.destroy()


def test_theme_change_does_not_write_conversation_cache(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        writes: list[str] = []
        monkeypatch.setattr(app, "_write_conversation_cache_safe", lambda: writes.append("cache"))

        app.theme_var.set("Light")
        app._on_theme_changed()

        assert writes == []
    finally:
        root.destroy()


def test_recent_sessions_listbox_styling_can_target_specific_rows() -> None:
    app = object.__new__(LumenDesktopApp)
    app.current_palette = dict(DARK_PALETTE)
    app.hovered_recent_index = 2
    app.recent_sessions_rows = [
        {"kind": "header"},
        {"kind": "session"},
        {"kind": "session"},
    ]

    class _FakeListbox:
        def __init__(self) -> None:
            self.calls: list[int] = []

        @staticmethod
        def size() -> int:
            return 3

        @staticmethod
        def curselection() -> tuple[int]:
            return (1,)

        def itemconfig(self, index: int, **kwargs) -> None:
            self.calls.append(index)

    app.recent_sessions_listbox = _FakeListbox()

    app._style_recent_sessions_listbox(indices=[1, 2])

    assert app.recent_sessions_listbox.calls == [1, 2]


def test_nav_hover_only_restyles_changed_rows() -> None:
    app = object.__new__(LumenDesktopApp)
    app.current_view = "chat"
    app.hovered_nav = None
    app.current_palette = dict(DARK_PALETTE)
    app.nav_buttons = {"chat": object(), "recent": object(), "memory": object()}
    touched: list[str] = []
    app._use_accented_dark_family_hover = lambda: False
    app._apply_nav_button_style = lambda name, nav_palette=None: touched.append(str(name))

    app._set_nav_hover("recent", True)
    app._set_nav_hover("recent", True)
    app._set_nav_hover("memory", True)

    assert Counter(touched) == {"recent": 2, "chat": 2, "memory": 1}


def test_apply_settings_row_style_targets_direct_row_widgets(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        section = tk.Frame(root)
        caption = tk.Label(section, text="Theme")
        control = tk.Label(section, text="Dark")
        divider = tk.Frame(section, height=1)
        section._lumen_settings_caption = caption  # type: ignore[attr-defined]
        section._lumen_settings_control = control  # type: ignore[attr-defined]
        section._lumen_settings_divider = divider  # type: ignore[attr-defined]
        touched: list[tuple[str, bool]] = []

        monkeypatch.setattr(
            app,
            "_style_settings_control_surface",
            lambda widget, *, bg, hovered: touched.append((widget.__class__.__name__, bool(hovered))),
        )

        app._apply_settings_row_style(section, hovered=True)

        assert touched == [("Label", True), ("Label", True)]
    finally:
        _destroy_app_root(root, app if "app" in locals() else None)


def test_settings_row_leave_ignores_child_to_child_motion(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        section = tk.Frame(root)
        caption = tk.Label(section, text="Theme")
        caption.pack()
        control = tk.Label(section, text="Light")
        control.pack()
        calls: list[bool] = []

        monkeypatch.setattr(
            app.root,
            "winfo_containing",
            lambda _x, _y: control,
        )

        app._handle_settings_row_leave(
            section,
            lambda hovered: calls.append(bool(hovered)),
            SimpleNamespace(x_root=10, y_root=10),
        )

        assert calls == []
    finally:
        _destroy_app_root(root, app if "app" in locals() else None)


def test_custom_theme_recolors_hotbar_footer_surface(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        app.theme_var.set("Custom")
        app._on_theme_changed()

        assert app.hotbar_footer.cget("bg") == app.current_palette["sidebar_bg"]
        assert app.hotbar_hint.cget("bg") == app.current_palette["sidebar_bg"]
    finally:
        _destroy_app_root(root, app if "app" in locals() else None)


def test_top_icon_hover_skips_redundant_configures() -> None:
    app = object.__new__(LumenDesktopApp)
    app.current_palette = dict(DARK_PALETTE)
    app.current_theme = {"background": DARK_PALETTE["app_bg"]}

    class _FakeButton:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def winfo_exists(self) -> bool:
            return True

        def configure(self, **kwargs) -> None:
            self.calls.append(str(kwargs.get("bg")))

    button = _FakeButton()
    app._set_top_icon_hover(button, True)
    app._set_top_icon_hover(button, True)
    app._set_top_icon_hover(button, False)

    assert button.calls == [DARK_PALETTE["button_hover_bg"], DARK_PALETTE["app_bg"]]


def test_light_theme_applies_panel_and_app_backgrounds_to_built_views(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        app.theme_var.set("Light")
        app._on_theme_changed()

        app._show_view("memory")
        memory_canvas = app.memory_list_canvas
        memory_inner = app.memory_list_inner
        memory_preview = app.memory_preview
        app._show_view("archived")
        root.update_idletasks()

        assert app.landing_frame.cget("bg") == app.current_palette["app_bg"]
        assert not hasattr(app, "memory_list_canvas")
        assert memory_canvas.winfo_exists() == 0
        assert memory_inner.winfo_exists() == 0
        assert memory_preview.winfo_exists() == 0
        assert app.views["archived"].cget("bg") == app.current_palette["panel_bg"]
        assert app.archived_list_canvas.cget("bg") == app.current_palette["panel_bg"]
        assert app.archived_list_inner.cget("bg") == app.current_palette["panel_bg"]
    finally:
        root.destroy()


def test_archive_session_removes_it_from_recent_list(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    monkeypatch.setattr("lumen.desktop.chat_app.messagebox.askyesno", lambda *args, **kwargs: True)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        session = app.controller._recent_sessions[0]

        app._archive_session(session)

        assert session["session_id"] in app.controller.archived_sessions
        assert app.status_var.get() == "Chat archived"
        assert all(
            str(item.get("session_id") or "") != str(session["session_id"])
            for item in app.recent_sessions_cache
        )
    finally:
        root.destroy()


def test_archiving_current_session_keeps_chat_open_and_marks_archived(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    monkeypatch.setattr("lumen.desktop.chat_app.messagebox.askyesno", lambda *args, **kwargs: True)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        session = {
            "session_id": app.session_id,
            "title": "Current",
            "prompt": "Current",
            "mode": "conversation",
            "created_at": "2026-03-23T12:00:00+00:00",
        }
        old_session_id = app.session_id

        app._archive_session(session)

        assert old_session_id in app.controller.archived_sessions
        assert app.session_id == old_session_id
        assert app.current_view == "chat"
    finally:
        root.destroy()


def test_delete_session_removes_it_from_recent_list(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    monkeypatch.setattr("lumen.desktop.chat_app.messagebox.askyesno", lambda *args, **kwargs: True)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        session = app.controller._recent_sessions[0]

        app._delete_session(session)

        assert session["session_id"] in app.controller.deleted_sessions
        assert app.status_var.get() == "Chat deleted"
    finally:
        root.destroy()


def test_archive_memory_calls_controller(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    monkeypatch.setattr("lumen.desktop.chat_app.messagebox.askyesno", lambda *args, **kwargs: True)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        app._show_view("memory")
        entry = {
            "title": "Preference",
            "kind": "personal_memory",
            "entry_path": str(app.controller.settings.data_root / "personal_memory" / "desktop-1" / "pref.json"),
        }

        app._archive_memory_entry(entry)

        assert entry["entry_path"] in app.controller.archived_memory_paths
        assert app.status_var.get() == "Memory archived"
    finally:
        root.destroy()


def test_delete_memory_calls_controller(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    monkeypatch.setattr("lumen.desktop.chat_app.messagebox.askyesno", lambda *args, **kwargs: True)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        note = {
            "title": "Gravity note",
            "kind": "research_note",
            "note_path": str(app.controller.settings.data_root / "research_notes" / "desktop-1" / "gravity.json"),
        }

        app._delete_memory_entry(note)

        assert note["note_path"] in app.controller.deleted_memory_paths
        assert app.status_var.get() == "Memory deleted"
    finally:
        root.destroy()


def test_memory_entry_key_distinguishes_duplicate_visible_rows() -> None:
    first = {
        "kind": "personal_memory",
        "title": "Same",
        "created_at": "2026-03-23T12:00:00+00:00",
        "memory_item_id": "memory-a",
    }
    second = {
        "kind": "personal_memory",
        "title": "Same",
        "created_at": "2026-03-23T12:00:00+00:00",
        "memory_item_id": "memory-b",
    }

    assert memory_entry_key(first) == "personal_memory:memory-a"
    assert memory_entry_key(second) == "personal_memory:memory-b"


def test_memory_entry_action_path_uses_db_memory_id_when_file_path_missing() -> None:
    entry = {
        "kind": "research_note",
        "title": "DB backed note",
        "memory_item_id": "note-db-id",
    }

    assert LumenDesktopApp._memory_entry_action_path(entry) == "note-db-id"


def test_memory_entry_snapshot_preserves_db_identity_fields() -> None:
    snapshot = LumenDesktopApp._memory_entry_snapshot(
        {
            "title": "DB backed note",
            "kind": "research_note",
            "memory_item_id": "note-db-id",
            "id": "row-id",
            "source_id": "source-id",
        }
    )

    assert snapshot["memory_item_id"] == "note-db-id"
    assert snapshot["id"] == "row-id"
    assert snapshot["source_id"] == "source-id"


def test_reused_memory_row_preserves_descriptor_identity_and_action_targets(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        app._show_view("memory")
        command_calls: list[str] = []
        archive_calls: list[str] = []
        delete_calls: list[str] = []
        monkeypatch.setattr(app, "_archive_memory_entry", lambda entry: archive_calls.append(str(entry.get("memory_item_id") or "")))
        monkeypatch.setattr(app, "_delete_memory_entry", lambda entry: delete_calls.append(str(entry.get("memory_item_id") or "")))

        first = {
            "title": "Same row",
            "kind": "personal_memory",
            "created_at": "2026-03-23T12:00:00+00:00",
            "memory_item_id": "memory-a",
        }
        descriptors, entry_map = app._memory_row_descriptors([first])
        app._rebuild_memory_rows(
            descriptors=descriptors,
            entry_map=entry_map,
            inner=app.memory_list_inner,
            command_builder=lambda entry: command_calls.append(str(entry.get("memory_item_id") or "")),
            archive_enabled=True,
        )

        row = next(widget for widget in app.memory_list_inner.winfo_children() if getattr(widget, "_browser_descriptor", ("",))[0] == "entry")
        assert getattr(row, "_browser_descriptor", None) == descriptors[1]

        second = {
            "title": "Same row",
            "kind": "personal_memory",
            "created_at": "2026-03-23T12:00:00+00:00",
            "memory_item_id": "memory-b",
        }
        descriptors, entry_map = app._memory_row_descriptors([second])
        app._rebuild_memory_rows(
            descriptors=descriptors,
            entry_map=entry_map,
            inner=app.memory_list_inner,
            command_builder=lambda entry: command_calls.append(str(entry.get("memory_item_id") or "")),
            archive_enabled=True,
        )

        row = next(widget for widget in app.memory_list_inner.winfo_children() if getattr(widget, "_browser_descriptor", ("",))[0] == "entry")
        assert getattr(row, "_browser_descriptor", None) == descriptors[1]
        assert getattr(row, "_browser_command_target_id", "") == "personal_memory:memory-b"

        app._invoke_browser_row_command(row)
        assert command_calls[-1] == "memory-b"

        actions = getattr(row, "_browser_context_actions", [])
        actions[0][1]()
        actions[1][1]()
        assert archive_calls[-1] == "memory-b"
        assert delete_calls[-1] == "memory-b"
    finally:
        _destroy_app_root(root, locals().get("app"))


def test_loading_session_restores_saved_mode(monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd())
        monkeypatch.setattr(
            app.controller,
            "get_session_profile",
            lambda session_id: {"interaction_profile": {"interaction_style": "collab"}},
        )

        app._load_session("desktop-1")

        assert app.mode_var.get() == "Collab"
    finally:
        root.destroy()
