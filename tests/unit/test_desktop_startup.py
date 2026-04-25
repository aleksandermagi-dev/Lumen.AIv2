from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tkinter as tk

import pytest

from lumen.desktop.chat_app import LumenDesktopApp
from lumen.desktop.desktop_crash_support import desktop_crash_log_path, read_crash_records
from lumen.desktop.main import main as desktop_main
from lumen.desktop.chat_ui_support import LIGHT_PALETTE
from lumen.desktop.memory_archive_support import build_memory_row_cache
from lumen.desktop.shell_transition_support import DesktopCapabilityState
from lumen.desktop.startup_diagnostics import (
    StartupCheckpointLogger,
    read_startup_checkpoints,
    startup_log_path,
    summarize_startup_checkpoints,
)
from tests.unit.test_chat_ui_support import _FakeController, _destroy_app_root


def test_frozen_startup_defers_controller_until_scheduled_boot(tmp_path: Path) -> None:
    logger = StartupCheckpointLogger(
        log_path=startup_log_path(data_root=tmp_path / "data"),
        execution_mode="frozen",
    )
    app = object.__new__(LumenDesktopApp)
    scheduled: list[tuple[str, object]] = []
    statuses: list[str] = []

    class _StatusVar:
        def set(self, value: str) -> None:
            statuses.append(value)

        def get(self) -> str:
            return statuses[-1] if statuses else ""

    class _FakeRoot:
        def after(self, _delay: int, func) -> None:
            scheduled.append(("after", func))

        def after_idle(self, func) -> None:
            scheduled.append(("after_idle", func))

    app.root = _FakeRoot()
    app.startup_logger = logger
    app.status_var = _StatusVar()
    app.controller = None
    app._controller_ready = False
    app._controller_bootstrapped = False
    app.current_view = "chat"
    app.memory_view_dirty = False
    app.recent_sessions_view_dirty = False
    app.archived_sessions_view_dirty = False
    app.archived_memory_view_dirty = False
    app._checkpoint = lambda checkpoint_id, phase, details=None: logger.checkpoint(
        checkpoint_id, phase, details=details
    )
    app._set_shell_ready_state = lambda ready: statuses.append(f"ready:{ready}")
    app._apply_mode_to_session = lambda: statuses.append("mode_applied")
    app._post_startup_bootstrap = lambda: statuses.append("startup_health_scheduled")
    app._refresh_loaded_views_after_startup = lambda: statuses.append("view_refresh_scheduled")

    def _bootstrap() -> None:
        logger.checkpoint("controller_bootstrap", "before", details="construct app controller")
        app.controller = SimpleNamespace(model_provider=SimpleNamespace(provider_id="local"))
        app._controller_ready = True
        logger.checkpoint("controller_bootstrap", "after", details="app controller ready")

    app._bootstrap_controller = _bootstrap
    app._initial_status = lambda: "Local mode"

    logger.checkpoint("window_show", "after", details="main window surfaced")
    logger.checkpoint("first_render_complete", "after", details="first render callback completed")
    app._begin_deferred_startup()

    assert app.controller is None
    assert statuses[:2] == ["Starting Lumen...", "ready:False"]
    assert scheduled and scheduled[0][0] == "after"

    scheduled[0][1]()

    assert app.controller is not None
    assert "Local mode" in statuses
    assert "ready:True" in statuses
    assert "mode_applied" in statuses

    records = read_startup_checkpoints(logger.log_path)
    ids = [str(item.get("checkpoint_id") or "") for item in records]
    assert "controller_bootstrap" in ids
    assert ids.index("first_render_complete") < ids.index("controller_bootstrap")

    summary = summarize_startup_checkpoints(logger.log_path)
    assert summary["first_render_seen"] is True
    assert "window_show" in summary["completed_before_first_render"]


def test_frozen_startup_failure_keeps_window_open_and_logs_error(tmp_path: Path) -> None:
    logger = StartupCheckpointLogger(
        log_path=startup_log_path(data_root=tmp_path / "data"),
        execution_mode="frozen",
    )
    app = object.__new__(LumenDesktopApp)
    app.startup_logger = logger
    app.messages = []
    app.pending = False
    app.controller = None
    statuses: list[str] = []

    class _StatusVar:
        def set(self, value: str) -> None:
            statuses.append(value)

        def get(self) -> str:
            return statuses[-1] if statuses else ""

    app.status_var = _StatusVar()
    app._set_shell_ready_state = lambda ready: statuses.append(f"ready:{ready}")
    app._append_system_line = lambda text: app.messages.append(SimpleNamespace(text=text))

    app._handle_startup_failure(RuntimeError("boom"))

    assert app.status_var.get() == "Startup issue detected"
    assert "part of startup did not finish" in app.messages[-1].text

    records = read_startup_checkpoints(logger.log_path)
    assert any(
        str(item.get("checkpoint_id") or "") == "deferred_startup"
        and str(item.get("phase") or "") == "error"
        for item in records
    )


def test_startup_health_result_applies_capability_snapshot_and_logs_it(tmp_path: Path) -> None:
    logger = StartupCheckpointLogger(
        log_path=startup_log_path(data_root=tmp_path / "data"),
        execution_mode="frozen",
    )
    app = object.__new__(LumenDesktopApp)
    statuses: list[str] = []
    messages: list[str] = []
    debug_events: list[tuple[str, dict[str, object]]] = []
    app._desktop_capability_state = DesktopCapabilityState.booting()
    app._shell_ready_flag = True
    app.pending = False
    app.startup_logger = logger
    app.current_view = "chat"
    app.status_var = SimpleNamespace(set=lambda value: statuses.append(value), get=lambda: statuses[-1] if statuses else "")
    app._append_system_line = lambda text: messages.append(text)
    app._apply_control_availability = lambda: statuses.append("controls_applied")
    app._debug_event = lambda label, **fields: debug_events.append((label, fields))

    app._apply_startup_health_result(
        {
            "missing": ["memory"],
            "missing_resources": ["tool_bundles"],
            "runtime_root": str(tmp_path),
            "data_root": str(tmp_path / "data"),
            "execution_mode": "frozen",
            "debug_ui": False,
            "capabilities": {"workspace.inspect": {"tool_id": "workspace"}},
        }
    )

    assert app._desktop_capability_state.phase == "degraded"
    assert app._desktop_capability_state.memory_surfaces_ready is False
    assert statuses[0] == "controls_applied"
    assert "Startup issue detected" in statuses
    records = read_startup_checkpoints(logger.log_path)
    assert any(str(item.get("checkpoint_id") or "") == "capability_snapshot" for item in records)
    assert any(label == "capability_snapshot_applied" for label, _fields in debug_events)


def test_startup_health_result_keeps_memory_surfaces_enabled_when_runtime_is_usable(tmp_path: Path) -> None:
    logger = StartupCheckpointLogger(
        log_path=startup_log_path(data_root=tmp_path / "data"),
        execution_mode="frozen",
    )
    app = object.__new__(LumenDesktopApp)
    statuses: list[str] = []
    debug_events: list[tuple[str, dict[str, object]]] = []
    app._desktop_capability_state = DesktopCapabilityState.booting()
    app._shell_ready_flag = True
    app.pending = False
    app.startup_logger = logger
    app.current_view = "chat"
    app.status_var = SimpleNamespace(set=lambda value: statuses.append(value), get=lambda: statuses[-1] if statuses else "")
    app._append_system_line = lambda text: None
    app._apply_control_availability = lambda: statuses.append("controls_applied")
    app._debug_event = lambda label, **fields: debug_events.append((label, fields))

    app._apply_startup_health_result(
        {
            "missing": ["memory"],
            "missing_resources": [],
            "runtime_root": str(tmp_path),
            "data_root": str(tmp_path / "data"),
            "execution_mode": "frozen",
            "debug_ui": False,
            "capabilities": {"workspace.inspect": {"tool_id": "workspace"}},
            "surface_runtime_ready": {"memory": True},
        }
    )

    assert app._desktop_capability_state.phase == "degraded"
    assert app._desktop_capability_state.memory_surfaces_ready is True
    payload = next(fields for label, fields in debug_events if label == "capability_snapshot_applied")
    assert "memory" in str(payload["enabled_surfaces"])
    assert "archived_memory" in str(payload["enabled_surfaces"])


def test_startup_health_result_keeps_memory_surfaces_enabled_when_bundle_is_reported_missing_but_runtime_is_otherwise_usable(
    tmp_path: Path,
) -> None:
    logger = StartupCheckpointLogger(
        log_path=startup_log_path(data_root=tmp_path / "data"),
        execution_mode="frozen",
    )
    app = object.__new__(LumenDesktopApp)
    statuses: list[str] = []
    debug_events: list[tuple[str, dict[str, object]]] = []
    app._desktop_capability_state = DesktopCapabilityState.booting()
    app._shell_ready_flag = True
    app.pending = False
    app.startup_logger = logger
    app.current_view = "chat"
    app.status_var = SimpleNamespace(set=lambda value: statuses.append(value), get=lambda: statuses[-1] if statuses else "")
    app._append_system_line = lambda text: None
    app._apply_control_availability = lambda: statuses.append("controls_applied")
    app._debug_event = lambda label, **fields: debug_events.append((label, fields))

    app._apply_startup_health_result(
        {
            "missing": ["memory"],
            "missing_resources": [],
            "runtime_root": str(tmp_path),
            "data_root": str(tmp_path / "data"),
            "execution_mode": "frozen",
            "debug_ui": False,
            "capabilities": {"workspace.inspect": {"tool_id": "workspace"}},
        }
    )

    assert app._desktop_capability_state.phase == "degraded"
    assert app._desktop_capability_state.memory_surfaces_ready is True
    payload = next(fields for label, fields in debug_events if label == "capability_snapshot_applied")
    assert payload["memory_surfaces_ready"] is True
    assert str(payload["memory_runtime_ready"]) == "unknown"
    assert "memory" in str(payload["enabled_surfaces"])
    assert "archived_memory" in str(payload["enabled_surfaces"])


def test_startup_health_result_reconciles_even_when_memory_surfaces_remain_disabled_in_degraded_mode(
    tmp_path: Path,
) -> None:
    logger = StartupCheckpointLogger(
        log_path=startup_log_path(data_root=tmp_path / "data"),
        execution_mode="frozen",
    )
    app = object.__new__(LumenDesktopApp)
    statuses: list[str] = []
    debug_events: list[tuple[str, dict[str, object]]] = []
    app._desktop_capability_state = DesktopCapabilityState.booting()
    app._shell_ready_flag = True
    app.pending = False
    app.startup_logger = logger
    app.current_view = "chat"
    app.status_var = SimpleNamespace(set=lambda value: statuses.append(value), get=lambda: statuses[-1] if statuses else "")
    app._append_system_line = lambda text: None
    app._apply_control_availability = lambda: statuses.append("controls_applied")
    app._debug_event = lambda label, **fields: debug_events.append((label, fields))
    app._reconcile_active_surface_after_capability_snapshot = lambda: statuses.append("reconciled")

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

    assert app._desktop_capability_state.phase == "degraded"
    assert app._desktop_capability_state.memory_surfaces_ready is False
    assert "reconciled" in statuses
    payload = next(fields for label, fields in debug_events if label == "capability_snapshot_applied")
    assert "memory" in str(payload["disabled_surfaces"])
    assert "archived_memory" in str(payload["disabled_surfaces"])


def test_ready_startup_reconciles_active_memory_surface_into_rows(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _FakeController)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=tmp_path, data_root=tmp_path / "data")
        app._show_view("memory")
        app._render_memory_surface_state(
            archived=False,
            preview_text="Loading memory...",
            list_text="Loading memory...",
        )
        entries = [
            {
                "title": f"Memory {index}",
                "content": f"Note {index}",
                "created_at": "2026-03-23T12:00:00+00:00",
                "kind": "personal_memory",
                "entry_path": f"memory-{index}.json",
            }
            for index in range(4)
        ]
        app.memory_entries = entries
        app.memory_cached_signature = app._memory_entries_signature(entries)
        (
            app.memory_row_descriptors,
            app.memory_row_entry_map,
            app.memory_row_descriptor_offsets,
            app.memory_row_group_counts,
        ) = build_memory_row_cache(entries)
        app.memory_view_dirty = False
        app.memory_fetch_in_flight = False
        app.memory_entries_has_more = False

        app._apply_startup_health_result(
            {
                "missing": [],
                "missing_resources": [],
                "runtime_root": str(tmp_path),
                "data_root": str(tmp_path / "data"),
                "execution_mode": "frozen",
                "debug_ui": False,
                "capabilities": {"workspace.inspect": {"tool_id": "workspace"}},
                "surface_runtime_ready": {"memory": True},
            }
        )

        labels = [child for child in app.memory_list_inner.winfo_children() if isinstance(child, tk.Label)]
        assert all("Loading memory..." not in label.cget("text") for label in labels)
        assert app.memory_rendered_count == 4
    finally:
        _destroy_app_root(root, app if "app" in locals() else None)


def test_window_geometry_is_clamped_onscreen() -> None:
    class _FakeRoot:
        def __init__(self) -> None:
            self._geometry = "1080x760+99999+99999"

        def geometry(self, value: str | None = None) -> str:
            if value is not None:
                self._geometry = value
            return self._geometry

        @staticmethod
        def winfo_screenwidth() -> int:
            return 1920

        @staticmethod
        def winfo_screenheight() -> int:
            return 1080

    app = object.__new__(LumenDesktopApp)
    app.root = _FakeRoot()

    app._ensure_window_geometry_visible()

    geometry = app.root.geometry()
    _, x_text, y_text = geometry.split("+", 2)
    x = int(x_text)
    y = int(y_text)
    assert x < app.root.winfo_screenwidth()
    assert y < app.root.winfo_screenheight()


def test_source_startup_with_saved_light_theme_initializes_light_palette(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", lambda **kwargs: SimpleNamespace(
        model_provider=SimpleNamespace(provider_id="local"),
        settings=SimpleNamespace(data_root=kwargs.get("data_root")),
        set_session_profile=lambda *args, **kw: None,
        reset_session_thread=lambda *args, **kw: None,
        run_deferred_startup_tasks=lambda: None,
    ))
    prefs_path = tmp_path / "data" / "desktop_ui" / "preferences.json"
    prefs_path.parent.mkdir(parents=True, exist_ok=True)
    prefs_path.write_text(json.dumps({"theme": "Light"}), encoding="utf-8")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        app = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        assert app.theme_var.get() == "Light"
        assert app.current_palette["app_bg"] == LIGHT_PALETTE["app_bg"]
        assert app.hamburger_button.cget("bg") == LIGHT_PALETTE["app_bg"]
    finally:
        _destroy_app_root(root, app if "app" in locals() else None)


def test_source_startup_restores_memory_browser_cache(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", lambda **kwargs: SimpleNamespace(
        model_provider=SimpleNamespace(provider_id="local"),
        settings=SimpleNamespace(data_root=kwargs.get("data_root")),
        set_session_profile=lambda *args, **kw: None,
        reset_session_thread=lambda *args, **kw: None,
        run_deferred_startup_tasks=lambda: None,
        list_personal_memory=lambda **kwargs: {"personal_memory": []},
        list_research_notes=lambda **kwargs: {"research_notes": []},
    ))
    cache_path = tmp_path / "data" / "desktop_ui" / "conversation_cache.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "recent": [],
                "archived": [],
                "memory": {
                    "entries": [
                        {
                            "title": "Cached Memory",
                            "content": "Remember this.",
                            "created_at": "2026-03-23T12:00:00+00:00",
                            "kind": "personal_memory",
                            "entry_path": "memory.json",
                        }
                    ],
                    "signature": [["Cached Memory", "2026-03-23T12:00:00+00:00", "personal_memory", "memory.json"]],
                    "has_more": False,
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
        assert app.memory_entries
        assert app.memory_entries[0]["title"] == "Cached Memory"
    finally:
        _destroy_app_root(root, app if "app" in locals() else None)


def test_source_startup_applies_theme_and_bootstrap_once(monkeypatch, tmp_path: Path) -> None:
    configure_calls: list[str] = []
    bootstrap_calls: list[str] = []
    startup_calls: list[str] = []

    class _Controller:
        def __init__(self, **kwargs):
            self.model_provider = SimpleNamespace(provider_id="local")
            self.settings = SimpleNamespace(data_root=kwargs.get("data_root"))

        def set_session_profile(self, *args, **kwargs) -> None:
            return None

        def reset_session_thread(self, *args, **kwargs) -> None:
            return None

        def run_deferred_startup_tasks(self) -> None:
            return None

    original_configure_styles = LumenDesktopApp._configure_styles
    original_bootstrap_controller = LumenDesktopApp._bootstrap_controller
    original_post_startup_bootstrap = LumenDesktopApp._post_startup_bootstrap

    def _configure_styles(self):
        configure_calls.append("styles")
        return original_configure_styles(self)

    def _bootstrap_controller(self):
        bootstrap_calls.append("bootstrap")
        return original_bootstrap_controller(self)

    def _post_startup_bootstrap(self):
        startup_calls.append("startup")
        return original_post_startup_bootstrap(self)

    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _Controller)
    monkeypatch.setattr(LumenDesktopApp, "_configure_styles", _configure_styles)
    monkeypatch.setattr(LumenDesktopApp, "_bootstrap_controller", _bootstrap_controller)
    monkeypatch.setattr(LumenDesktopApp, "_post_startup_bootstrap", _post_startup_bootstrap)

    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        _ = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")

        assert configure_calls == ["styles"]
        assert bootstrap_calls == ["bootstrap"]
        assert startup_calls == []
    finally:
        _destroy_app_root(root, app if "app" in locals() else None)


def test_source_startup_runs_diagnostics_only_in_debug_mode(monkeypatch, tmp_path: Path) -> None:
    startup_calls: list[str] = []

    class _Controller:
        def __init__(self, **kwargs):
            self.model_provider = SimpleNamespace(provider_id="local")
            self.settings = SimpleNamespace(data_root=kwargs.get("data_root"))

        def set_session_profile(self, *args, **kwargs) -> None:
            return None

        def reset_session_thread(self, *args, **kwargs) -> None:
            return None

        def run_deferred_startup_tasks(self) -> None:
            return None

    original_post_startup_bootstrap = LumenDesktopApp._post_startup_bootstrap

    def _post_startup_bootstrap(self):
        startup_calls.append("startup")
        return original_post_startup_bootstrap(self)

    monkeypatch.setenv("LUMEN_DEBUG_UI", "1")
    monkeypatch.setattr("lumen.desktop.chat_app.AppController", _Controller)
    monkeypatch.setattr(LumenDesktopApp, "_post_startup_bootstrap", _post_startup_bootstrap)

    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this test environment")
    root.withdraw()
    try:
        _ = LumenDesktopApp(root, repo_root=Path.cwd(), data_root=tmp_path / "data")
        root.update()
        assert startup_calls == ["startup"]
    finally:
        _destroy_app_root(root, app if "app" in locals() else None)


def test_capability_state_fallback_inference_is_stable_without_explicit_runtime_ready() -> None:
    state = DesktopCapabilityState.from_runtime(
        missing_bundles=["memory"],
        missing_resources=[],
        capabilities={"workspace.inspect": {"tool_id": "workspace"}},
        surface_runtime_ready=None,
    )

    payload = state.summary_payload()
    assert state.phase == "degraded"
    assert state.memory_surfaces_ready is True
    assert payload["degraded_reason"] == "missing_bundles"
    assert payload["surface_ready_sources"]["memory"] == "capability_fallback"


def test_main_runtime_failure_writes_desktop_crash_record(tmp_path: Path, monkeypatch) -> None:
    args = SimpleNamespace(
        repo_root=tmp_path,
        data_root=tmp_path / "data",
        validation_smoke_report=None,
        validation_anh_probe=None,
    )
    monkeypatch.setattr("lumen.desktop.main.build_parser", lambda: SimpleNamespace(parse_args=lambda: args))
    monkeypatch.setattr(
        "lumen.desktop.main.resolve_desktop_runtime_paths",
        lambda **kwargs: SimpleNamespace(
            runtime_root=tmp_path,
            data_root=tmp_path / "data",
            execution_mode="source",
        ),
    )
    monkeypatch.setattr("lumen.desktop.main.tk.Tk", lambda: (_ for _ in ()).throw(RuntimeError("tk boom")))

    with pytest.raises(RuntimeError, match="tk boom"):
        desktop_main()

    records = read_crash_records(desktop_crash_log_path(data_root=tmp_path / "data"))
    assert records
    assert records[-1]["source"] == "desktop_main"
    assert records[-1]["exception_type"] == "RuntimeError"
