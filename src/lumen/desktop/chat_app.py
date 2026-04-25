from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
from queue import Empty, Queue
import sys
from threading import Thread
from time import perf_counter
import tkinter as tk
from tkinter import colorchooser, filedialog, font as tkfont, messagebox, simpledialog, ttk
from typing import Callable

try:
    from PIL import Image, ImageTk
except Exception:  # pragma: no cover - optional desktop image fallback
    Image = None
    ImageTk = None

from lumen.app.controller import AppController
from lumen.app.settings import AppSettings
from lumen.desktop.chat_experience_support import build_context_bar, build_pending_label
from lumen.desktop.chat_layout_support import bubble_side_padding, bubble_wraplength
from lumen.desktop.chat_presenter import ChatPresenter
from lumen.desktop.desktop_crash_support import (
    append_crash_record,
    append_runtime_failure_text,
    build_crash_record,
    desktop_crash_log_path,
    desktop_runtime_failure_log_path,
)
from lumen.desktop.desktop_startup_snapshot_support import build_startup_snapshot
from lumen.desktop.desktop_startup_state_support import capability_state_from_startup_health
from lumen.desktop.desktop_style_support import (
    DesktopControlAvailability,
    resolve_composer_button_palette,
    resolve_control_availability,
    resolve_input_palette,
    resolve_load_more_palette,
    resolve_nav_button_visual,
    resolve_top_icon_palette,
)
from lumen.desktop.desktop_view_state_support import resolve_view_refresh_decision
from lumen.desktop.memory_archive_support import (
    bounded_entries_slice,
    build_memory_row_cache,
    build_memory_row_descriptors,
    memory_group_count,
    memory_entries_signature,
    memory_entry_key,
    memory_row_cache_slice,
    memory_rows_match_descriptors,
    should_fetch_archived_memory,
    should_render_archived_memory_cache,
)
from lumen.desktop.shell_transition_support import (
    DebugTraceSession,
    DesktopCapabilityState,
    ShellTransitionState,
)
from lumen.desktop.startup_diagnostics import StartupCheckpointLogger
from lumen.desktop.chat_ui_support import (
    DARK_PALETTE,
    DesktopChatMessage,
    THEME_PALETTES,
    darkTheme,
    custom_accent_palette,
    day_group_label,
    empty_state_text,
    grouped_entry_sections,
    grouped_session_rows,
    human_date_label,
    knowledge_category_lines,
    message_role_style,
    message_timestamp,
    nav_button_style,
    palette_from_theme,
    resolve_theme_tokens,
    validate_palette,
)
from lumen.nlu.starter_prompt_layer import StarterPromptLayer
from lumen.session_visibility import is_user_visible_session


class LumenDesktopApp:
    """Small local desktop chat shell over the existing AppController."""

    MODE_OPTIONS = {
        "Direct": "direct",
        "Default": "default",
        "Collab": "collab",
    }
    MODE_DESCRIPTORS = {
        "Direct": "concise",
        "Default": "balanced",
        "Collab": "exploratory",
    }
    STYLE_TO_MODE = {value: key for key, value in MODE_OPTIONS.items()}
    STARTER_PROMPT_OPTIONS = tuple(
        prompt
        for category in StarterPromptLayer.starter_prompts()
        for prompt in category.prompts
    )
    LANGUAGE_STYLE_OPTIONS = ("Standard", "Emoji Friendly")
    FONT_OPTIONS = ("Arial", "Times New Roman", "Impact", "Pacifico")
    DEFAULT_FONT_FAMILY = "Segoe UI"
    IDENTITY_ICON_SIZE = 124
    IDENTITY_ICON_MIN_SIZE = 160
    IDENTITY_ICON_MAX_SIZE = 220
    IDENTITY_ICON_VIEWPORT_RATIO = 0.10
    DEBUG_UI_ENV = "LUMEN_DEBUG_UI"
    DEBUG_UI_LOG_PREFIX = "ui_debug"
    DEFAULT_DISPLAY_NAME = "You"
    CONVERSATION_CACHE_FILE = "conversation_cache.json"
    HOTBAR_ANIMATION_DURATION = 0.10
    HOTBAR_ANIMATION_INTERVAL_MS = 16
    RECENT_SESSIONS_FETCH_LIMIT = 60
    HEAVY_SURFACE_RENDER_STEP = 4
    HEAVY_SURFACE_VIEWS = ("recent", "archived", "memory", "archived_memory")

    def __init__(
        self,
        root: tk.Tk,
        *,
        repo_root: Path,
        data_root: Path | None = None,
        execution_mode: str = "source",
        startup_logger: StartupCheckpointLogger | None = None,
    ):
        self.root = root
        self.repo_root = repo_root
        self.execution_mode = str(execution_mode or "source").strip() or "source"
        self.settings = AppSettings.from_repo_root(repo_root, data_root_override=data_root)
        self.startup_logger = startup_logger
        self.controller: AppController | None = None
        self._controller_bootstrapped = False
        self._controller_ready = False
        self._launch_first_boot = self.execution_mode == "frozen"
        self.presenter = ChatPresenter()
        self.session_id = self._new_session_id()
        self.pending = False
        self.result_queue: Queue[tuple[str, int, object]] = Queue()
        self.ui_task_queue: Queue[tuple[str, str, int, object]] = Queue()
        self.pending_row: tk.Widget | None = None
        self.pending_text_widget: tk.Text | None = None
        self.pending_base_text = "Lumen is reasoning"
        self.pending_animation_job: str | None = None
        self.pending_dot_step = 0
        self.active_request_id = 0
        self.pending_request_id = 0
        self.ignored_request_ids: set[int] = set()
        self.attached_input_path: Path | None = None
        self.attached_input_kind = ""
        self.message_labels: list[tk.Label] = []
        self.message_text_widgets: list[tk.Text] = []
        self.messages: list[DesktopChatMessage] = []
        self.recent_sessions_cache: list[dict[str, object]] = []
        self.recent_sessions_rows: list[dict[str, object]] = []
        self.recent_sessions_signature: tuple[tuple[str, str, str, str, bool], ...] = ()
        self.recent_sessions_render_signature: tuple[tuple[str, str, str, str, bool], ...] = ()
        self.recent_sessions_rendered_count = 0
        self.recent_sessions_requested_signature: tuple[tuple[str, str, str, str, bool], ...] | None = None
        self.recent_sessions_fetch_in_flight = False
        self.recent_sessions_restored_from_disk = False
        self.archived_sessions_cache: list[dict[str, object]] = []
        self.archived_sessions_rows: list[dict[str, object]] = []
        self.archived_sessions_signature: tuple[tuple[str, str, str, str, bool], ...] = ()
        self.archived_sessions_render_signature: tuple[tuple[str, str, str, str, bool], ...] = ()
        self.archived_sessions_rendered_count = 0
        self.archived_sessions_requested_signature: tuple[tuple[str, str, str, str, bool], ...] | None = None
        self.archived_sessions_fetch_in_flight = False
        self.archived_sessions_restored_from_disk = False
        self.archived_memory_entries: list[dict[str, object]] = []
        self.memory_entries: list[dict[str, object]] = []
        self.current_view = "chat"
        self.memory_view_dirty = True
        self.recent_sessions_view_dirty = True
        self.archived_sessions_view_dirty = True
        self.archived_memory_view_dirty = True
        self.memory_cached_signature: tuple[tuple[str, str, str, str], ...] = ()
        self.memory_render_signature: tuple[tuple[str, str, str, str], ...] = ()
        self.memory_row_descriptors: list[tuple[object, ...]] = []
        self.memory_row_entry_map: dict[str, dict[str, object]] = {}
        self.memory_row_descriptor_offsets: tuple[int, ...] = (0,)
        self.memory_row_group_counts: tuple[int, ...] = (0,)
        self.memory_entries_has_more = False
        self.memory_fetch_in_flight = False
        self.memory_requested_fetch_limit: int | None = None
        self.archived_memory_render_signature: tuple[tuple[str, str, str, str], ...] = ()
        self.archived_memory_cached_signature: tuple[tuple[str, str, str, str], ...] = ()
        self.archived_memory_row_descriptors: list[tuple[object, ...]] = []
        self.archived_memory_row_entry_map: dict[str, dict[str, object]] = {}
        self.archived_memory_row_descriptor_offsets: tuple[int, ...] = (0,)
        self.archived_memory_row_group_counts: tuple[int, ...] = (0,)
        self.archived_memory_entries_has_more = False
        self.archived_memory_state_version = 0
        self.archived_memory_loaded_version = -1
        self.archived_memory_requested_version: int | None = None
        self.archived_memory_requested_fetch_limit: int | None = None
        self.archived_memory_fetch_in_flight = False
        self.placeholder_active = False
        self.hovered_nav: str | None = None
        self.hovered_recent_index: int | None = None
        self._last_recent_selected_index: int | None = None
        self.selected_memory_topic_index: int | None = None
        self._async_task_tokens: dict[str, int] = {
            "recent": 0,
            "archived": 0,
            "memory": 0,
            "archived_memory": 0,
            "startup_health": 0,
        }
        self._async_task_started_at: dict[str, float] = {}
        self.active_scrollable: tk.Widget | None = None
        self._mousewheel_bound_globally = False
        self.deferred_view_builders: dict[str, object] = {}
        self.starter_prompt_buttons: list[tk.Button] = []
        self._startup_health_applied = False
        self._shell_ready_flag = False
        self._desktop_capability_state = (
            DesktopCapabilityState.booting()
            if self._launch_first_boot and self._startup_diagnostics_enabled()
            else DesktopCapabilityState.from_runtime(
                missing_bundles=[],
                missing_resources=[],
                capabilities={"bootstrap": {}},
            )
        )
        self.hotbar_open = False
        self.identity_image: tk.PhotoImage | None = None
        self.profile_avatar_image: tk.PhotoImage | None = None
        self.profile_avatar_path: Path | None = None
        self.identity_icon_path = self._resolve_identity_icon_path(repo_root)
        self.desktop_prefs_path = self.settings.data_root / "desktop_ui" / "preferences.json"
        self.conversation_cache_path = self.settings.data_root / "desktop_ui" / self.CONVERSATION_CACHE_FILE
        self.desktop_crash_log_path = desktop_crash_log_path(data_root=self.settings.data_root)
        self.desktop_runtime_failure_log_path = desktop_runtime_failure_log_path(data_root=self.settings.data_root)
        self.debug_ui_log_path: Path | None = None
        self._debug_ui_log_failed = False
        self.screen_title_var = tk.StringVar(value="")
        self.date_var = tk.StringVar(value="")
        self.time_var = tk.StringVar(value="")
        self.daylight_var = tk.StringVar(value="")
        self.display_name_var = tk.StringVar(value=self.DEFAULT_DISPLAY_NAME)
        self.chat_title_var = tk.StringVar(value="Chat")
        self.font_family_var = tk.StringVar(value=self.DEFAULT_FONT_FAMILY)
        self.text_size_var = tk.IntVar(value=11)
        self.custom_theme_var = tk.StringVar(value="Lumen Purple")
        self.custom_accent_color: str | None = None
        self.allow_emojis_var = tk.BooleanVar(value=False)
        self.settings_help_visible = tk.BooleanVar(value=False)
        self.custom_colors = {
            "user_bg": None,
            "user_text": None,
            "assistant_bg": None,
            "assistant_text": None,
        }
        self.hotbar_target_width = 236
        self.hotbar_current_width = 0
        self.hotbar_animation_job: str | None = None
        self.hotbar_transition_in_progress = False
        self.pending_hotbar_open_state: bool | None = None
        self.pending_hotbar_refresh_target: str | None = None
        self._deferred_refresh_from_hotbar_close = False
        self.recent_sessions_render_limit = self.HEAVY_SURFACE_RENDER_STEP
        self.recent_sessions_render_step = self.HEAVY_SURFACE_RENDER_STEP
        self.recent_sessions_load_more_button: tk.Button | None = None
        self.archived_sessions_render_limit = self.HEAVY_SURFACE_RENDER_STEP
        self.archived_sessions_render_step = self.HEAVY_SURFACE_RENDER_STEP
        self.archived_sessions_load_more_button: tk.Button | None = None
        self.memory_render_limit = self.HEAVY_SURFACE_RENDER_STEP
        self.memory_render_step = self.HEAVY_SURFACE_RENDER_STEP
        self.memory_rendered_count = 0
        self.memory_load_more_button: tk.Button | None = None
        self.archived_memory_render_limit = self.HEAVY_SURFACE_RENDER_STEP
        self.archived_memory_render_step = self.HEAVY_SURFACE_RENDER_STEP
        self.archived_memory_rendered_count = 0
        self.archived_memory_load_more_button: tk.Button | None = None
        self.hotbar_navigation_generation = 0
        self.pending_view_generation = 0
        self.pending_refresh_generation = 0
        self._active_browser_hover_rows: dict[str, tk.Widget | None] = {
            "recent": None,
            "archived": None,
            "memory": None,
            "archived_memory": None,
        }
        self.message_reveal_jobs: set[str] = set()
        self.pending_row_request_id = 0
        self.deferred_view_refresh_job: str | None = None
        self.deferred_view_refresh_target: str | None = None
        self.deferred_view_refresh_generation = 0
        self.pending_view_name: str | None = None
        self.pending_view_job: str | None = None
        self._theme_apply_job: str | None = None
        self._theme_apply_in_progress = False
        self._theme_apply_requested = False
        self._chat_canvas_layout_job: str | None = None
        self._chat_canvas_layout_width: int | None = None
        self._chat_canvas_layout_needs_scrollregion = False
        self._chat_scroll_to_bottom_job: str | None = None
        self._mousewheel_flush_jobs: dict[int, str] = {}
        self._startup_followup_job: str | None = None
        self._startup_followup_needs_post_bootstrap = False
        self._startup_followup_needs_view_refresh = False
        self._startup_followup_needs_background_tasks = False
        self._debug_ui_flag = self._read_debug_ui_enabled_env()
        self._shell_transition_state = ShellTransitionState(
            debug_session=DebugTraceSession.create(
                data_root=self.settings.data_root,
                enabled=self._debug_ui_enabled(),
                prefix=self.DEBUG_UI_LOG_PREFIX,
            )
        )
        self.debug_ui_log_path = self._shell_transition_state.debug_session.log_path
        self._custom_color_chooser_active = False
        self.settings_popup: tk.Toplevel | None = None
        self.settings_popup_kind: str | None = None
        self.settings_popup_opening = False
        self._hotbar_transition_started_at: float | None = None
        self.queue_drain_job: str | None = None
        self.clock_job: str | None = None
        self.conversation_cache_write_job: str | None = None
        self._destroying = False
        self.settings_help_sections = (
            "Starter Prompts",
            "What Lumen Can Do",
            "How to Use Lumen Better",
            "Mode Meanings",
            "Stop / Control Info",
            "Report Issue / Feedback",
        )
        self.settings_help_copy = {
            "Starter Prompts": "Use the visible thinking-tool chips on the landing screen to seed a prompt quickly without opening a menu.",
            "What Lumen Can Do": "Lumen can reason conversationally, use math and knowledge tools, analyze structured data, inspect papers, simulate systems, design experiments, explore invention constraints, and work through domain wrappers.",
            "How to Use Lumen Better": "Ask directly, follow up naturally, attach one file or folder when needed, and use mode changes when you want Lumen to be more concise or more exploratory.",
            "Mode Meanings": "Default stays balanced. Direct keeps the answer concise. Collab stays exploratory and partner-like while preserving the same facts and conclusions.",
            "Stop / Control Info": "While Lumen is actively responding, the plus and mic controls become Stop. Stopping keeps the partial conversation and ignores the late result safely.",
            "Report Issue / Feedback": "If the shell looks wrong, a screen feels slow, or a response path feels brittle, capture the prompt or view you were in so the issue can be reproduced cleanly.",
        }

        self.theme_var = tk.StringVar(value="Dark")
        self.mode_var = tk.StringVar(value="Default")
        self.starter_prompt_var = tk.StringVar(
            value=self.STARTER_PROMPT_OPTIONS[0] if self.STARTER_PROMPT_OPTIONS else ""
        )
        self.show_starter_prompts_var = tk.BooleanVar(value=True)
        self.chat_density_var = tk.StringVar(value="Comfortable")
        self.language_style_var = tk.StringVar(value="Standard")
        self.current_theme = dict(darkTheme)
        self.current_palette = dict(DARK_PALETTE)
        self._conversation_cache_signature = ""
        self._conversation_cache_dirty = False
        self._conversation_cache_write_scheduled = False
        self._identity_icon_signature: tuple[str, int] | None = None
        self._debug_ui_event_counts: dict[str, int] = {}
        self._control_availability = DesktopControlAvailability(
            shell_interactive=False,
            chat_ready=False,
            nav_enabled=False,
            selector_state="disabled",
            chat_state=tk.DISABLED,
            top_level_state=tk.DISABLED,
        )

        self.root.title("Lumen")
        self.root.geometry("1080x760")
        self.root.configure(bg=self.current_palette["app_bg"])
        self._checkpoint("root_configured", "after", "title and base geometry configured")
        self._load_desktop_preferences()
        self._load_conversation_cache()
        startup_snapshot = build_startup_snapshot(
            theme_name=str(self.theme_var.get() or "Dark"),
            custom_theme_name=str(self.custom_theme_var.get() or "Lumen Purple"),
            custom_accent_color=self.custom_accent_color,
        )
        self.current_theme = dict(startup_snapshot.theme_tokens)
        self.current_palette = dict(startup_snapshot.palette)
        self.root.configure(bg=self.current_palette["app_bg"])
        self._configure_styles()

        if not self._launch_first_boot:
            self._bootstrap_controller()
            self._shell_ready_flag = True
        self.status_var = tk.StringVar(value=self._initial_status())
        self.context_bar_var = tk.StringVar(
            value=build_context_bar(mode_label=self.mode_var.get(), prompt="")
        )
        self.mode_descriptor_var = tk.StringVar(
            value=self.MODE_DESCRIPTORS.get(self.mode_var.get(), "balanced")
        )
        self.attachment_var = tk.StringVar(value="No file, folder, or zip selected.")
        self.attachment_hint_var = tk.StringVar(
            value="Attach one file, folder, or zip for the next send."
        )
        self._apply_mode_to_session()

        self._checkpoint("shell_layout", "before", "build desktop shell")
        self._build_layout()
        self._checkpoint("shell_layout", "after", "desktop shell built")
        self._apply_palette_to_shell()
        self._on_starter_visibility_changed()
        self._show_input_placeholder()
        self._bind_global_mousewheel()
        self.root.bind_all("<Button-1>", self._handle_global_click, add="+")
        self.root.bind("<Destroy>", self._on_root_destroy, add="+")
        self.root.report_callback_exception = self._report_tk_callback_exception
        setattr(self.root, "_lumen_app", self)
        self._start_debug_ui_session()
        self._apply_control_availability()
        if self._launch_first_boot:
            self._set_shell_ready_state(False)
            self._show_window_for_startup()
            self._checkpoint("first_idle_scheduled", "after", "startup idle callbacks scheduled")
            self.root.after_idle(self._mark_first_render_complete)
            self.root.after_idle(self._begin_deferred_startup)
        else:
            self._run_optional_startup_followup()
        self.queue_drain_job = self.root.after(100, self._drain_queue)
        self._update_clock()

    def _build_layout(self) -> None:
        frame = tk.Frame(self.root, bg=self.current_palette["app_bg"], padx=14, pady=14)
        frame.pack(fill=tk.BOTH, expand=True)
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        self.container = frame

        self._build_top_bar(frame)
        self._build_shell_body(frame)
        self._build_chat_view()
        self.deferred_view_builders = {
            "memory": self._build_memory_view,
            "archived_memory": self._build_archived_memory_view,
            "recent": self._build_recent_sessions_view,
            "archived": self._build_archived_sessions_view,
            "settings": self._build_settings_view,
        }
        self._build_input_row(self.main_column)
        self._apply_view_visibility("chat", schedule_refresh=False)

    def _build_top_bar(self, parent: tk.Widget) -> None:
        top_bar = tk.Frame(parent, bg=self.current_palette["app_bg"], pady=2)
        top_bar.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        top_bar.grid_columnconfigure(1, weight=1)
        self.top_bar = top_bar

        left_group = tk.Frame(top_bar, bg=self.current_palette["app_bg"])
        left_group.grid(row=0, column=0, sticky="w")
        self.top_bar_left = left_group

        self.hamburger_button = tk.Button(
            left_group,
            text="\u2630",
            command=self._toggle_hotbar,
            relief=tk.FLAT,
            bd=0,
            padx=8,
            pady=6,
            cursor="hand2",
            highlightthickness=0,
        )
        self.hamburger_button.grid(row=0, column=0, sticky="w")
        self.hamburger_button.bind("<Enter>", lambda event: self._set_top_icon_hover(self.hamburger_button, True))
        self.hamburger_button.bind("<Leave>", lambda event: self._set_top_icon_hover(self.hamburger_button, False))

        self.top_new_session_button = tk.Button(
            left_group,
            text="+",
            command=self._start_new_session,
            relief=tk.FLAT,
            bd=0,
            padx=10,
            pady=6,
            cursor="hand2",
            highlightthickness=0,
            font=("Segoe UI Semibold", 13),
        )
        self.top_new_session_button.grid(row=0, column=1, sticky="w", padx=(6, 0))
        self.top_new_session_button.bind("<Enter>", lambda event: self._set_top_icon_hover(self.top_new_session_button, True))
        self.top_new_session_button.bind("<Leave>", lambda event: self._set_top_icon_hover(self.top_new_session_button, False))

        center_group = tk.Frame(top_bar, bg=self.current_palette["app_bg"])
        center_group.grid(row=0, column=1, sticky="n")
        self.top_bar_center = center_group
        self.screen_title_label = tk.Label(
            center_group,
            textvariable=self.screen_title_var,
            font=("Segoe UI Semibold", 11),
            bg=self.current_palette["app_bg"],
            fg=self.current_palette["text_primary"],
        )
        self.screen_title_label.grid(row=0, column=0)
        self.date_label = tk.Label(
            center_group,
            textvariable=self.date_var,
            font=("Segoe UI", 10),
            bg=self.current_palette["app_bg"],
            fg=self.current_palette["text_secondary"],
        )
        self.date_label.grid(row=1, column=0)

        right_group = tk.Frame(top_bar, bg=self.current_palette["app_bg"])
        right_group.grid(row=0, column=2, sticky="e")
        self.top_bar_right = right_group
        self.daylight_label = tk.Label(
            right_group,
            textvariable=self.daylight_var,
            font=("Segoe UI Symbol", 10),
            bg=self.current_palette["app_bg"],
            fg=self.current_palette["text_secondary"],
        )
        self.daylight_label.grid(row=0, column=0, padx=(0, 6))
        self.time_label = tk.Label(
            right_group,
            textvariable=self.time_var,
            font=("Segoe UI Semibold", 10),
            bg=self.current_palette["app_bg"],
            fg=self.current_palette["text_primary"],
        )
        self.time_label.grid(row=0, column=1)

    def _build_shell_body(self, parent: tk.Widget) -> None:
        body = tk.Frame(parent, bg=self.current_palette["app_bg"])
        body.grid(row=1, column=0, sticky="nsew")
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)
        self.body_container = body

        self.hotbar = tk.Frame(body, bg=self.current_palette["sidebar_bg"], width=236, bd=0, highlightthickness=0)
        self.hotbar.grid(row=0, column=0, sticky="nsw", padx=(0, 14))
        self.hotbar.grid_remove()
        self.hotbar.grid_propagate(False)
        self.hotbar.configure(width=0)
        self._build_hotbar_content(self.hotbar)

        main_column = tk.Frame(body, bg=self.current_palette["app_bg"])
        main_column.grid(row=0, column=1, sticky="nsew")
        main_column.grid_rowconfigure(0, weight=1)
        main_column.grid_columnconfigure(0, weight=1)
        self.main_column = main_column

        self._build_content_container(main_column)

    def _build_hotbar_content(self, parent: tk.Widget) -> None:
        parent.grid_rowconfigure(2, weight=1)
        parent.grid_columnconfigure(0, weight=1)

        profile_block = tk.Frame(parent, bg=self.current_palette["sidebar_bg"], padx=12, pady=10)
        profile_block.grid(row=0, column=0, sticky="ew")
        self.hotbar_profile_block = profile_block

        self.profile_avatar = tk.Canvas(
            profile_block,
            width=50,
            height=50,
            bd=0,
            highlightthickness=0,
            bg=self.current_palette["sidebar_bg"],
        )
        self.profile_avatar.grid(row=0, column=0, sticky="w")
        self.profile_avatar.bind("<Button-1>", self._change_profile_avatar)

        self.display_name_entry = tk.Entry(
            profile_block,
            textvariable=self.display_name_var,
            relief=tk.FLAT,
            bd=0,
            width=20,
            font=("Segoe UI Semibold", 11),
        )
        self.display_name_entry.grid(row=1, column=0, sticky="ew", pady=(10, 4))
        self.display_name_entry.bind("<KeyRelease>", self._on_display_name_changed)
        self.display_name_entry.bind("<Return>", self._confirm_display_name)

        nav_block = tk.Frame(parent, bg=self.current_palette["sidebar_bg"], padx=8, pady=8)
        nav_block.grid(row=1, column=0, sticky="nsew")
        self.hotbar_nav_block = nav_block

        self.nav_buttons = {}
        self.nav_button_frames = {}
        self.nav_accents = {}
        nav_items = (
            ("chat", self.chat_title_var),
            ("recent", "All Conversations"),
            ("archived", "Archived Chats"),
            ("memory", "Memory"),
            ("archived_memory", "Archived Memory"),
            ("settings", "Settings"),
            ("quit", "Exit"),
        )
        for index, (view_name, label) in enumerate(nav_items):
            row = tk.Frame(nav_block, bg=self.current_palette["sidebar_bg"], bd=0, highlightthickness=0)
            row.grid(row=index, column=0, sticky="ew", pady=(0, 6))
            row.grid_columnconfigure(1, weight=1)
            accent = tk.Frame(row, width=3, bg=self.current_palette["sidebar_bg"])
            accent.grid(row=0, column=0, sticky="ns", padx=(0, 8))
            text = label if isinstance(label, str) else label.get()
            button = tk.Button(
                row,
                text=text,
                command=(self.root.destroy if view_name == "quit" else lambda name=view_name: self._handle_hotbar_destination(name)),
                relief=tk.FLAT,
                bd=0,
                anchor="w",
                padx=10,
                pady=10,
                cursor="hand2",
                highlightthickness=0,
            )
            button.grid(row=0, column=1, sticky="ew")
            button.bind("<Enter>", lambda event, name=view_name: self._set_nav_hover(name, True))
            button.bind("<Leave>", lambda event, name=view_name: self._set_nav_hover(name, False))
            if view_name == "chat":
                button.bind("<Double-Button-1>", lambda event: self._rename_current_chat())
            self.nav_buttons[view_name] = button
            self.nav_button_frames[view_name] = row
            self.nav_accents[view_name] = accent

        footer = tk.Frame(parent, bg=self.current_palette["sidebar_bg"], padx=12, pady=10)
        footer.grid(row=3, column=0, sticky="ew")
        self.hotbar_footer = footer
        self.hotbar_hint = tk.Label(
            footer,
            text="Double-click Chat to rename it.",
            anchor="w",
            justify=tk.LEFT,
            font=("Segoe UI", 9),
            bg=self.current_palette["sidebar_bg"],
            fg=self.current_palette["text_muted"],
        )
        self.hotbar_hint.grid(row=0, column=0, sticky="ew")

    def _build_sidebar(self, parent: tk.Widget) -> None:
        sidebar = tk.Frame(
            parent,
            bg=self.current_palette["sidebar_bg"],
            highlightbackground=self.current_palette["sidebar_border"],
            highlightthickness=1,
            bd=0,
            width=220,
        )
        sidebar.grid(row=0, column=0, sticky="nsw", padx=(0, 12))
        sidebar.grid_rowconfigure(1, weight=1)
        self.sidebar = sidebar

        title = tk.Label(
            sidebar,
            text="Lumen",
            bg=self.current_palette["sidebar_bg"],
            fg=self.current_palette["text_primary"],
            font=("Segoe UI Semibold", 16),
            anchor="w",
            padx=16,
            pady=14,
        )
        title.grid(row=0, column=0, sticky="ew")
        self.sidebar_title = title

        nav_surface = tk.Frame(sidebar, bg=self.current_palette["sidebar_bg"])
        nav_surface.grid(row=1, column=0, sticky="nsew")
        nav_surface.grid_rowconfigure(0, weight=1)
        nav_surface.grid_columnconfigure(0, weight=1)
        self.sidebar_surface = nav_surface

        self.sidebar_canvas = tk.Canvas(
            nav_surface,
            bg=self.current_palette["sidebar_bg"],
            highlightthickness=0,
            bd=0,
            width=208,
        )
        self.sidebar_canvas._lumen_layout_surface = "sidebar"  # type: ignore[attr-defined]
        self.sidebar_canvas.grid(row=0, column=0, sticky="nsew")
        self.sidebar_scrollbar = ttk.Scrollbar(
            nav_surface,
            style="Dark.Vertical.TScrollbar",
            orient="vertical",
            command=self.sidebar_canvas.yview,
        )
        self.sidebar_scrollbar.grid(row=0, column=1, sticky="ns")
        self.sidebar_canvas.configure(yscrollcommand=self.sidebar_scrollbar.set)

        self.sidebar_inner = tk.Frame(self.sidebar_canvas, bg=self.current_palette["sidebar_bg"], padx=10, pady=8)
        self.sidebar_window = self.sidebar_canvas.create_window((0, 0), window=self.sidebar_inner, anchor="nw")
        self._register_scroll_owner(self.sidebar_canvas, self.sidebar_scrollbar)
        self._bind_canvas_layout(
            inner=self.sidebar_inner,
            canvas=self.sidebar_canvas,
            scrollbar=self.sidebar_scrollbar,
            window_id=self.sidebar_window,
        )
        self._bind_mousewheel(self.sidebar_canvas, self.sidebar_canvas)

        self.nav_buttons: dict[str, tk.Button] = {}
        self.nav_button_frames: dict[str, tk.Frame] = {}
        self.nav_accents: dict[str, tk.Frame] = {}
        for view_name, label in (
            ("chat", "Chat"),
            ("memory", "Memory"),
            ("recent", "Last Chats"),
            ("settings", "Settings"),
        ):
            row = tk.Frame(
                self.sidebar_inner,
                bg=self.current_palette["sidebar_bg"],
                highlightthickness=0,
                bd=0,
            )
            row.pack(fill=tk.X, pady=4)
            row.grid_columnconfigure(1, weight=1)
            accent = tk.Frame(row, width=4, bg=self.current_palette["sidebar_bg"])
            accent.grid(row=0, column=0, sticky="ns")
            button = tk.Button(
                row,
                text=label,
                command=lambda name=view_name: self._show_view(name),
                relief=tk.FLAT,
                bd=0,
                anchor="w",
                padx=12,
                pady=10,
                cursor="hand2",
                highlightthickness=0,
            )
            button.grid(row=0, column=1, sticky="ew")
            button.bind("<Enter>", lambda event, name=view_name: self._set_nav_hover(name, True))
            button.bind("<Leave>", lambda event, name=view_name: self._set_nav_hover(name, False))
            self.nav_button_frames[view_name] = row
            self.nav_accents[view_name] = accent
            self.nav_buttons[view_name] = button

    def _build_content_container(self, parent: tk.Widget) -> None:
        content = tk.Frame(parent, bg=self.current_palette["app_bg"])
        content.grid(row=0, column=0, sticky="nsew")
        content.grid_rowconfigure(0, weight=1)
        content.grid_columnconfigure(0, weight=1)
        self.content_container = content
        self.views: dict[str, tk.Frame] = {}

    def _build_chat_view(self) -> None:
        frame = tk.Frame(self.content_container, bg=self.current_palette["app_bg"])
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        self.views["chat"] = frame

        self.chat_surface = tk.Frame(
            frame,
            bg=self.current_palette["app_bg"],
            highlightbackground=self.current_palette["app_bg"],
            highlightthickness=0,
            bd=0,
        )
        self.chat_surface.grid(row=0, column=0, sticky="nsew")
        self.chat_surface.grid_rowconfigure(0, weight=1)
        self.chat_surface.grid_columnconfigure(0, weight=1)

        self.chat_canvas = tk.Canvas(
            self.chat_surface,
            bg=self.current_palette["app_bg"],
            highlightthickness=0,
            bd=0,
        )
        self.chat_canvas._lumen_layout_surface = "chat"  # type: ignore[attr-defined]
        self.chat_canvas.grid(row=0, column=0, sticky="nsew")
        self._bind_mousewheel(self.chat_canvas, self.chat_canvas)

        self.chat_scrollbar = ttk.Scrollbar(
            self.chat_surface,
            style="Dark.Vertical.TScrollbar",
            orient="vertical",
            command=self.chat_canvas.yview,
        )
        self.chat_scrollbar.grid(row=0, column=1, sticky="ns")
        self.chat_canvas.configure(yscrollcommand=self.chat_scrollbar.set)

        self.chat_frame = tk.Frame(self.chat_canvas, bg=self.current_palette["app_bg"], padx=0, pady=18)
        self.chat_window = self.chat_canvas.create_window((0, 0), window=self.chat_frame, anchor="nw")
        self.chat_frame.bind("<Configure>", self._on_chat_frame_configure)
        self.chat_canvas.bind("<Configure>", self._on_chat_canvas_configure)
        self._register_scroll_owner(self.chat_canvas, self.chat_scrollbar)

        self.landing_frame = tk.Frame(self.chat_surface, bg=self.current_palette["app_bg"])
        self.landing_frame.place(relx=0.5, rely=0.5, anchor="center")
        self.landing_orb_frame = tk.Frame(
            self.landing_frame,
            bg=self.current_palette["app_bg"],
            width=180,
            height=180,
        )
        self.landing_orb_frame.pack(pady=(0, 18))
        self.landing_orb_frame.pack_propagate(False)
        self.identity_label = tk.Label(
            self.landing_orb_frame,
            bg=self.current_palette["app_bg"],
            anchor="center",
            justify=tk.CENTER,
        )
        self.identity_label.pack(expand=True)
        self.landing_greeting_frame = tk.Frame(self.landing_frame, bg=self.current_palette["app_bg"])
        self.landing_greeting_frame.pack(pady=(2, 0))
        self.greeting_label = tk.Label(
            self.landing_greeting_frame,
            bg=self.current_palette["app_bg"],
            fg=self.current_palette["text_primary"],
            font=("Segoe UI Semibold", 20),
            text="Lumen is ready.",
        )
        self.greeting_label.pack()
        self.greeting_subtitle = tk.Label(
            self.landing_greeting_frame,
            bg=self.current_palette["app_bg"],
            fg=self.current_palette["text_secondary"],
            font=("Segoe UI", 11),
            text="What are we working through today?",
        )
        self.greeting_subtitle.pack(pady=(10, 18))
        self.starter_frame = tk.Frame(self.landing_frame, bg=self.current_palette["app_bg"])
        self.starter_frame.pack()
        starter_header = tk.Label(
            self.starter_frame,
            text="Thinking tools",
            bg=self.current_palette["app_bg"],
            fg=self.current_palette["text_muted"],
            font=("Segoe UI Semibold", 9),
        )
        starter_header.pack(anchor="center", pady=(0, 10))
        self.starter_dropdown_frame = tk.Frame(self.starter_frame, bg=self.current_palette["app_bg"])
        self.starter_dropdown_frame.pack()
        self.starter_prompt_picker = ttk.Combobox(
            self.starter_dropdown_frame,
            style="Dark.TCombobox",
            state="readonly",
            width=44,
            textvariable=self.starter_prompt_var,
            values=list(self.STARTER_PROMPT_OPTIONS),
        )
        self.starter_prompt_picker.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.starter_prompt_apply_button = tk.Button(
            self.starter_dropdown_frame,
            text="Use",
            command=self._use_starter_prompt,
            relief=tk.FLAT,
            bd=0,
            padx=10,
            pady=6,
            cursor="hand2",
            highlightthickness=0,
        )
        self.starter_prompt_apply_button.grid(row=0, column=1, sticky="e")

    def _build_input_row(self, parent: tk.Widget) -> None:
        input_frame = tk.Frame(parent, bg=self.current_palette["app_bg"])
        input_frame.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        input_frame.grid_columnconfigure(0, weight=1)
        self.input_frame = input_frame

        self.context_bar_label = tk.Label(
            input_frame,
            textvariable=self.context_bar_var,
            bg=self.current_palette["app_bg"],
            fg=self.current_palette["text_secondary"],
            font=("Segoe UI", 9),
            anchor="w",
        )
        self.context_bar_label.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        self.attachment_row = tk.Frame(input_frame, bg=self.current_palette["app_bg"])
        self.attachment_row.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        self.attachment_row.grid_columnconfigure(0, weight=1)
        self.attachment_label = tk.Label(
            self.attachment_row,
            textvariable=self.attachment_var,
            bg=self.current_palette["app_bg"],
            fg=self.current_palette["text_muted"],
            font=("Segoe UI", 9),
            anchor="w",
        )
        self.attachment_label.grid(row=0, column=0, sticky="ew")
        self.clear_attachment_button = tk.Button(
            self.attachment_row,
            text="Clear",
            command=self._clear_attachment,
            relief=tk.FLAT,
            bd=0,
            padx=8,
            pady=4,
            cursor="hand2",
            highlightthickness=0,
        )
        self.clear_attachment_button.grid(row=0, column=1, sticky="e")

        composer = tk.Frame(input_frame, bg=self.current_palette["panel_bg"], padx=12, pady=10)
        composer.grid(row=2, column=0, sticky="ew")
        composer.grid_columnconfigure(2, weight=1)
        self.composer_frame = composer

        self.mode_pill = tk.Frame(composer, bg=self.current_palette["panel_bg"])
        self.mode_pill.grid(row=0, column=0, sticky="w", padx=(0, 10))
        self.mode_selector = ttk.Combobox(
            self.mode_pill,
            style="Dark.TCombobox",
            state="readonly",
            width=10,
            textvariable=self.mode_var,
            values=list(self.MODE_OPTIONS.keys()),
        )
        self.mode_selector.grid(row=0, column=0, sticky="w")
        self.mode_selector.bind("<<ComboboxSelected>>", self._on_mode_changed)
        self.mode_descriptor_label = tk.Label(
            self.mode_pill,
            textvariable=self.mode_descriptor_var,
            bg=self.current_palette["panel_bg"],
            fg=self.current_palette["text_muted"],
            font=("Segoe UI", 9),
            padx=8,
        )
        self.mode_descriptor_label.grid(row=0, column=1, sticky="w")

        self.input_box = tk.Text(
            composer,
            height=self._input_box_height(),
            wrap=tk.WORD,
            font=self._message_font(),
            bg=self.current_palette["input_bg"],
            fg=self.current_palette["text_primary"],
            insertbackground=self.current_palette["text_primary"],
            relief=tk.FLAT,
            highlightthickness=0,
            padx=12,
            pady=10,
        )
        self.input_box.grid(row=0, column=2, sticky="ew", padx=(0, 12))
        self.input_box.bind("<Return>", self._on_return_pressed)
        self.input_box.bind("<Shift-Return>", self._on_shift_return_pressed)
        self.input_box.bind("<Button-1>", self._on_input_pointer_down)
        self.input_box.bind("<KeyPress>", self._on_input_key_press)
        self.input_box.bind("<FocusIn>", self._on_input_focus_in)
        self.input_box.bind("<FocusOut>", self._on_input_focus_out)
        self.input_box.bind("<KeyRelease>", self._on_input_key_release)
        self._bind_text_context_menu(self.input_box, editable=True)

        self.add_button = tk.Button(
            composer,
            relief=tk.FLAT,
            bd=0,
            padx=10,
            pady=8,
            cursor="hand2",
            highlightthickness=0,
            font=("Segoe UI Semibold", 14),
            text="+",
            command=self._show_attach_menu,
        )
        self.add_button.grid(row=0, column=3, sticky="e", padx=(0, 8))

        self.mic_button = tk.Button(
            composer,
            text="\U0001F3A4",
            relief=tk.FLAT,
            bd=0,
            padx=10,
            pady=8,
            state=tk.DISABLED,
            highlightthickness=0,
            font=("Segoe UI Symbol", 12),
        )
        self.mic_button.grid(row=0, column=4, sticky="e")

        self.stop_button = ttk.Button(
            composer,
            style="Dark.TButton",
            text="Stop",
            command=self._stop_current_task,
        )
        self.stop_button.grid(row=0, column=3, columnspan=2, sticky="e")
        self.stop_button.grid_remove()

        self.send_button = self.add_button
        self.add_file_button = self.add_button
        self.add_folder_button = self.add_button
        self.action_frame = composer
        self.attachment_hint_label = tk.Label(
            input_frame,
            textvariable=self.attachment_hint_var,
            bg=self.current_palette["app_bg"],
            fg=self.current_palette["text_muted"],
            font=("Segoe UI", 8),
            anchor="w",
        )
        self.attachment_hint_label.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        self._refresh_attachment_state()

    def _build_memory_view(self) -> None:
        frame = self._build_panel_view(title="Memory")
        frame.grid_rowconfigure(0, weight=0)
        frame.grid_rowconfigure(1, weight=1)

        self.memory_preview = tk.Text(
            frame,
            wrap=tk.WORD,
            state=tk.DISABLED,
            relief=tk.FLAT,
            highlightthickness=0,
            bd=0,
            padx=16,
            pady=16,
            height=7,
        )
        self.memory_preview.grid(row=0, column=0, sticky="ew", pady=(0, 18))
        self._bind_text_context_menu(self.memory_preview, editable=False)

        list_host = tk.Frame(frame, bg=self.current_palette["panel_bg"])
        list_host.grid(row=1, column=0, sticky="nsew")
        list_host.grid_rowconfigure(0, weight=1)
        list_host.grid_columnconfigure(0, weight=1)

        self.memory_list_canvas = tk.Canvas(
            list_host,
            highlightthickness=0,
            bd=0,
            bg=self.current_palette["panel_bg"],
        )
        self.memory_list_canvas._lumen_layout_surface = "memory"  # type: ignore[attr-defined]
        self.memory_list_canvas.grid(row=0, column=0, sticky="nsew")
        self.memory_list_scrollbar = ttk.Scrollbar(
            list_host,
            style="Dark.Vertical.TScrollbar",
            orient="vertical",
            command=self.memory_list_canvas.yview,
        )
        self.memory_list_scrollbar.grid(row=0, column=1, sticky="ns")
        self.memory_list_canvas.configure(yscrollcommand=self.memory_list_scrollbar.set)
        self.memory_list_inner = tk.Frame(self.memory_list_canvas, bg=self.current_palette["panel_bg"])
        self.memory_list_window = self.memory_list_canvas.create_window((0, 0), window=self.memory_list_inner, anchor="nw")
        self._register_scroll_owner(self.memory_list_canvas, self.memory_list_scrollbar)
        self._bind_canvas_layout(
            inner=self.memory_list_inner,
            canvas=self.memory_list_canvas,
            scrollbar=self.memory_list_scrollbar,
            window_id=self.memory_list_window,
        )
        self._bind_mousewheel(self.memory_list_canvas, self.memory_list_canvas)
        self.views["memory"] = frame
        restored = self._restore_session_cache_from_disk("memory")
        if self.memory_entries:
            self._debug_event(
                "surface_reenter_from_cache",
                view="memory",
                fetched_count=len(self.memory_entries),
                render_limit=self.memory_render_limit,
                cache_source="disk" if restored else "memory",
            )
            self._render_memory_entries_from_cache(archived=False, render_mode="render_from_cached_slice")
        else:
            self._render_memory_surface_first_paint(archived=False)

    def _build_recent_sessions_view(self) -> None:
        frame = self._build_panel_view(title="All Conversations")
        list_host = tk.Frame(frame, bg=self.current_palette["panel_bg"])
        list_host.grid(row=0, column=0, sticky="nsew")
        list_host.grid_rowconfigure(0, weight=1)
        list_host.grid_columnconfigure(0, weight=1)
        self.recent_list_canvas = tk.Canvas(
            list_host,
            highlightthickness=0,
            bd=0,
            bg=self.current_palette["panel_bg"],
        )
        self.recent_list_canvas._lumen_layout_surface = "recent"  # type: ignore[attr-defined]
        self.recent_list_canvas.grid(row=0, column=0, sticky="nsew")
        self.recent_sessions_scrollbar = ttk.Scrollbar(
            list_host,
            style="Dark.Vertical.TScrollbar",
            orient="vertical",
            command=self.recent_list_canvas.yview,
        )
        self.recent_sessions_scrollbar.grid(row=0, column=1, sticky="ns")
        self.recent_list_canvas.configure(yscrollcommand=self.recent_sessions_scrollbar.set)
        self.recent_list_inner = tk.Frame(self.recent_list_canvas, bg=self.current_palette["panel_bg"])
        self.recent_list_window = self.recent_list_canvas.create_window((0, 0), window=self.recent_list_inner, anchor="nw")
        self._register_scroll_owner(self.recent_list_canvas, self.recent_sessions_scrollbar)
        self._bind_canvas_layout(
            inner=self.recent_list_inner,
            canvas=self.recent_list_canvas,
            scrollbar=self.recent_sessions_scrollbar,
            window_id=self.recent_list_window,
        )
        self._bind_mousewheel(self.recent_list_canvas, self.recent_list_canvas)
        self.views["recent"] = frame
        if self.recent_sessions_rows:
            self._debug_event(
                "surface_reenter_from_cache",
                view="recent",
                fetched_count=len(self.recent_sessions_cache),
                render_limit=self.recent_sessions_render_limit,
            )
            self._render_recent_sessions_from_cache()

    def _build_archived_sessions_view(self) -> None:
        frame = self._build_panel_view(title="Archived Chats")
        list_host = tk.Frame(frame, bg=self.current_palette["panel_bg"])
        list_host.grid(row=0, column=0, sticky="nsew")
        list_host.grid_rowconfigure(0, weight=1)
        list_host.grid_columnconfigure(0, weight=1)
        self.archived_list_canvas = tk.Canvas(
            list_host,
            highlightthickness=0,
            bd=0,
            bg=self.current_palette["panel_bg"],
        )
        self.archived_list_canvas._lumen_layout_surface = "archived"  # type: ignore[attr-defined]
        self.archived_list_canvas.grid(row=0, column=0, sticky="nsew")
        self.archived_sessions_scrollbar = ttk.Scrollbar(
            list_host,
            style="Dark.Vertical.TScrollbar",
            orient="vertical",
            command=self.archived_list_canvas.yview,
        )
        self.archived_sessions_scrollbar.grid(row=0, column=1, sticky="ns")
        self.archived_list_canvas.configure(yscrollcommand=self.archived_sessions_scrollbar.set)
        self.archived_list_inner = tk.Frame(self.archived_list_canvas, bg=self.current_palette["panel_bg"])
        self.archived_list_window = self.archived_list_canvas.create_window((0, 0), window=self.archived_list_inner, anchor="nw")
        self._register_scroll_owner(self.archived_list_canvas, self.archived_sessions_scrollbar)
        self._bind_canvas_layout(
            inner=self.archived_list_inner,
            canvas=self.archived_list_canvas,
            scrollbar=self.archived_sessions_scrollbar,
            window_id=self.archived_list_window,
        )
        self._bind_mousewheel(self.archived_list_canvas, self.archived_list_canvas)
        self.views["archived"] = frame
        if self.archived_sessions_rows:
            self._debug_event(
                "archived_surface_reentered_from_cache",
                cached=bool(self.archived_sessions_signature),
                signature_size=len(self.archived_sessions_signature),
            )
            self._render_archived_sessions_from_cache()

    def _build_archived_memory_view(self) -> None:
        frame = self._build_panel_view(title="Archived Memory")
        frame.grid_rowconfigure(0, weight=0)
        frame.grid_rowconfigure(1, weight=1)

        self.archived_memory_preview = tk.Text(
            frame,
            wrap=tk.WORD,
            state=tk.DISABLED,
            relief=tk.FLAT,
            bd=0,
            highlightthickness=0,
            height=10,
            padx=12,
            pady=12,
        )
        self.archived_memory_preview.grid(row=0, column=0, sticky="ew", pady=(0, 18))
        self._bind_text_context_menu(self.archived_memory_preview, editable=False)

        list_host = tk.Frame(frame, bg=self.current_palette["panel_bg"])
        list_host.grid(row=1, column=0, sticky="nsew")
        list_host.grid_rowconfigure(0, weight=1)
        list_host.grid_columnconfigure(0, weight=1)
        self.archived_memory_list_canvas = tk.Canvas(
            list_host,
            highlightthickness=0,
            bd=0,
            bg=self.current_palette["panel_bg"],
        )
        self.archived_memory_list_canvas._lumen_layout_surface = "archived_memory"  # type: ignore[attr-defined]
        self.archived_memory_list_canvas.grid(row=0, column=0, sticky="nsew")
        self.archived_memory_list_scrollbar = ttk.Scrollbar(
            list_host,
            style="Dark.Vertical.TScrollbar",
            orient="vertical",
            command=self.archived_memory_list_canvas.yview,
        )
        self.archived_memory_list_scrollbar.grid(row=0, column=1, sticky="ns")
        self.archived_memory_list_canvas.configure(yscrollcommand=self.archived_memory_list_scrollbar.set)
        self.archived_memory_list_inner = tk.Frame(self.archived_memory_list_canvas, bg=self.current_palette["panel_bg"])
        self.archived_memory_list_window = self.archived_memory_list_canvas.create_window((0, 0), window=self.archived_memory_list_inner, anchor="nw")
        self._register_scroll_owner(self.archived_memory_list_canvas, self.archived_memory_list_scrollbar)
        self._bind_canvas_layout(
            inner=self.archived_memory_list_inner,
            canvas=self.archived_memory_list_canvas,
            scrollbar=self.archived_memory_list_scrollbar,
            window_id=self.archived_memory_list_window,
        )
        self._bind_mousewheel(self.archived_memory_list_canvas, self.archived_memory_list_canvas)
        self.views["archived_memory"] = frame
        restored = self._restore_session_cache_from_disk("archived_memory")
        if self.archived_memory_entries:
            self._debug_event(
                "surface_reenter_from_cache",
                view="archived_memory",
                fetched_count=len(self.archived_memory_entries),
                render_limit=self.archived_memory_render_limit,
                cache_source="disk" if restored else "memory",
            )
            self._render_archived_memory_from_cache(render_mode="render_from_cached_slice")
        else:
            self._render_memory_surface_first_paint(archived=True)

    def _build_settings_view(self) -> None:
        frame = self._build_panel_view(title="Settings")
        scroll_host = tk.Frame(frame, bg=self.current_palette["app_bg"])
        scroll_host.grid(row=0, column=0, sticky="nsew")
        scroll_host.grid_rowconfigure(0, weight=1)
        scroll_host.grid_columnconfigure(0, weight=1)
        self.settings_scroll_host = scroll_host
        settings_canvas = tk.Canvas(scroll_host, highlightthickness=0, bd=0, bg=self.current_palette["app_bg"])
        settings_canvas._lumen_layout_surface = "settings"  # type: ignore[attr-defined]
        settings_canvas.grid(row=0, column=0, sticky="nsew")
        self.settings_canvas = settings_canvas
        settings_scrollbar = ttk.Scrollbar(
            scroll_host,
            style="Dark.Vertical.TScrollbar",
            orient="vertical",
            command=settings_canvas.yview,
        )
        settings_scrollbar.grid(row=0, column=1, sticky="ns")
        self.settings_scrollbar = settings_scrollbar
        settings_canvas.configure(yscrollcommand=settings_scrollbar.set)
        self.settings_scroll_inner = tk.Frame(settings_canvas, bg=self.current_palette["app_bg"], padx=8, pady=8)
        self.settings_scroll_window = settings_canvas.create_window((0, 0), window=self.settings_scroll_inner, anchor="nw")
        self._register_scroll_owner(settings_canvas, self.settings_scrollbar)
        self._bind_canvas_layout(
            inner=self.settings_scroll_inner,
            canvas=settings_canvas,
            scrollbar=self.settings_scrollbar,
            window_id=self.settings_scroll_window,
        )
        self._bind_mousewheel(settings_canvas, settings_canvas)

        self.theme_value_label = tk.Label(
            self.settings_scroll_inner,
            text=str(self.theme_var.get() or "Dark"),
            bg=self.current_palette["app_bg"],
            fg=self.current_palette["text_primary"],
            font=("Segoe UI", 10),
            anchor="e",
            justify=tk.RIGHT,
            cursor="hand2",
        )
        self.theme_value_mask = tk.Frame(
            self.settings_scroll_inner,
            bg=self.current_palette["app_bg"],
            width=112,
            height=20,
            highlightthickness=0,
            bd=0,
        )
        self.theme_value_mask.grid_propagate(False)
        self.density_value_label = tk.Label(
            self.settings_scroll_inner,
            textvariable=self.chat_density_var,
            bg=self.current_palette["app_bg"],
            fg=self.current_palette["text_primary"],
            font=("Segoe UI", 10),
            anchor="e",
            justify=tk.RIGHT,
            cursor="hand2",
        )
        self.font_value_label = tk.Label(
            self.settings_scroll_inner,
            textvariable=self.font_family_var,
            bg=self.current_palette["app_bg"],
            fg=self.current_palette["text_primary"],
            font=("Segoe UI", 10),
            anchor="e",
            justify=tk.RIGHT,
            cursor="hand2",
        )
        self.text_size_scale = tk.Scale(
            self.root,
            from_=10,
            to=16,
            orient=tk.HORIZONTAL,
            variable=self.text_size_var,
            command=lambda value: self._on_text_size_changed(),
            relief=tk.FLAT,
            bd=0,
            highlightthickness=0,
        )
        self.text_size_value_label = tk.Label(
            self.settings_scroll_inner,
            text=str(self.text_size_var.get()),
            bg=self.current_palette["app_bg"],
            fg=self.current_palette["text_primary"],
            font=("Segoe UI", 10),
            anchor="e",
            justify=tk.RIGHT,
            cursor="hand2",
        )
        self.language_style_value_label = tk.Label(
            self.settings_scroll_inner,
            textvariable=self.language_style_var,
            bg=self.current_palette["app_bg"],
            fg=self.current_palette["text_primary"],
            font=("Segoe UI", 10),
            anchor="e",
            justify=tk.RIGHT,
            cursor="hand2",
        )
        self.color_value_labels: dict[str, tk.Label] = {}
        self.settings_popup: tk.Toplevel | None = None
        self.settings_popup_kind: str | None = None
        self.settings_popup_opening = False

        self._build_settings_rows()
        self.settings_reset_button = tk.Button(
            frame,
            text="Reset",
            command=self._reset_style_overrides,
            relief=tk.FLAT,
            bd=0,
            padx=22,
            pady=8,
            cursor="hand2",
            highlightthickness=0,
            font=("Segoe UI Semibold", 9),
        )
        self.settings_reset_button.place(relx=1.0, x=-12, y=10, anchor="ne")
        self.settings_reset_button.bind(
            "<Enter>",
            lambda event: self._style_settings_reset_button(hovered=True),
            add="+",
        )
        self.settings_reset_button.bind(
            "<Leave>",
            lambda event: self._style_settings_reset_button(hovered=False),
            add="+",
        )
        self._style_settings_reset_button(hovered=False)
        self.views["settings"] = frame

    def _build_panel_view(self, *, title: str) -> tk.Frame:
        frame = tk.Frame(
            self.content_container,
            bg=self.current_palette["app_bg"],
            highlightbackground=self.current_palette["app_bg"],
            highlightthickness=0,
            bd=0,
            padx=6,
            pady=6,
        )
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        return frame

    def _build_settings_rows(self) -> None:
        self.settings_row_sections: list[tk.Frame] = []
        self.settings_row_labels: list[tk.Label] = []
        self.settings_row_map: dict[str, tk.Frame] = {}
        self.settings_row_controls: dict[str, tk.Widget] = {}
        row = 0
        row = self._add_settings_row("Theme", self.theme_value_mask, row, action=self._activate_theme_row)
        row = self._add_settings_row("Chat Density", self.density_value_label, row, action=self._activate_density_row)
        row = self._add_settings_row("Text Size", self.text_size_value_label, row, action=self._activate_text_size_row)
        row = self._add_settings_row("Language Style", self.language_style_value_label, row, action=self._activate_emoji_row)
        row = self._add_settings_row("Font", self.font_value_label, row, action=self._activate_font_row)

        for label, key in (
            ("User Bubble Color", "user_bg"),
            ("User Text Color", "user_text"),
            ("Lumen Bubble Color", "assistant_bg"),
            ("Lumen Text Color", "assistant_text"),
        ):
            value_label = tk.Label(
                self.settings_scroll_inner,
                text=self._format_color_choice(key),
                bg=self.current_palette["app_bg"],
                fg=self.current_palette["text_primary"],
                font=("Segoe UI", 10),
                anchor="e",
                justify=tk.RIGHT,
                cursor="hand2",
            )
            self.color_value_labels[key] = value_label
            row = self._add_settings_row(label, value_label, row, action=lambda color_key=key: self._pick_color(color_key))

        self.help_row_indicator = tk.Label(
            self.settings_scroll_inner,
            text="Show",
            anchor="e",
            justify=tk.RIGHT,
            font=("Segoe UI", 9),
            bg=self.current_palette["app_bg"],
            fg=self.current_palette["text_secondary"],
            cursor="hand2",
        )
        row = self._add_settings_row("Help", self.help_row_indicator, row, action=self._toggle_settings_help)
        self.help_text = tk.Text(
            self.settings_scroll_inner,
            wrap=tk.WORD,
            state=tk.DISABLED,
            relief=tk.FLAT,
            bd=0,
            highlightthickness=0,
            height=12,
            padx=12,
            pady=12,
        )
        self.help_text.grid(row=row, column=0, columnspan=2, sticky="ew")
        self._bind_text_context_menu(self.help_text, editable=False)
        self._set_text_widget(
            self.help_text,
            "\n\n".join(
                f"{section}\n{self.settings_help_copy.get(section, 'Keep this section clean and integrated into Lumen.')}"
                for section in self.settings_help_sections
            ),
        )
        self.help_text.grid_remove()
        self.settings_scroll_inner.grid_columnconfigure(1, weight=1)

    def _add_settings_row(
        self,
        label: str,
        control: tk.Widget,
        row: int,
        *,
        action: object | None = None,
    ) -> int:
        section = tk.Frame(self.settings_scroll_inner, bg=self.current_palette["app_bg"], pady=8)
        section.grid(row=row, column=0, columnspan=2, sticky="ew")
        section.grid_columnconfigure(1, weight=1)
        divider = tk.Frame(section, height=1, bg=self.current_palette["panel_divider"])
        divider.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        caption = tk.Label(
            section,
            text=label,
            anchor="w",
            justify=tk.LEFT,
            font=("Segoe UI", 10),
            bg=self.current_palette["app_bg"],
            fg=self.current_palette["text_primary"],
        )
        caption.grid(row=0, column=0, sticky="w")
        control.grid(row=0, column=1, sticky="e")
        if isinstance(control, tk.Button):
            control.configure(
                bg=self.current_palette["panel_alt_bg"],
                fg=self.current_palette["text_primary"],
                activebackground=self.current_palette["button_hover_bg"],
                activeforeground=self.current_palette["text_primary"],
            )
        self._bind_settings_row_hover(section, caption, control)
        if action is not None:
            self._bind_settings_row_action(section, caption, control, action)
        section._lumen_settings_caption = caption  # type: ignore[attr-defined]
        section._lumen_settings_control = control  # type: ignore[attr-defined]
        section._lumen_settings_divider = divider  # type: ignore[attr-defined]
        section._lumen_settings_hovered = False  # type: ignore[attr-defined]
        self.settings_row_sections.append(section)
        self.settings_row_labels.append(caption)
        self.settings_row_map[label] = section
        self.settings_row_controls[label] = control
        return row + 1

    def _bind_settings_row_action(self, section: tk.Frame, caption: tk.Label, control: tk.Widget, action: object) -> None:
        def _invoke(_: tk.Event | None = None) -> str:
            action()
            return "break"

        for widget in (section, caption):
            widget.bind("<Button-1>", _invoke, add="+")
        if isinstance(control, tk.Label):
            control.bind("<Button-1>", _invoke, add="+")

    def _bind_settings_row_hover(self, section: tk.Frame, caption: tk.Label, control: tk.Widget) -> None:
        def _apply(active: bool) -> None:
            current_state = bool(getattr(section, "_lumen_settings_hovered", False))
            if current_state == active:
                return
            section._lumen_settings_hovered = active  # type: ignore[attr-defined]
            self._apply_settings_row_style(section, hovered=active)

        for widget in (section, caption, control):
            widget.bind("<Enter>", lambda event: _apply(True), add="+")
            widget.bind("<Leave>", lambda event, row=section: self._handle_settings_row_leave(row, _apply, event), add="+")

    def _widget_within_section(self, section: tk.Frame, widget: object | None) -> bool:
        current = widget
        while current is not None:
            if current is section:
                return True
            current = getattr(current, "master", None)
        return False

    def _handle_settings_row_leave(
        self,
        section: tk.Frame,
        apply_hover: Callable[[bool], None],
        event: tk.Event | None = None,
    ) -> None:
        next_widget = None
        if event is not None:
            try:
                next_widget = self.root.winfo_containing(event.x_root, event.y_root)
            except tk.TclError:
                next_widget = None
        if self._widget_within_section(section, next_widget):
            return
        apply_hover(False)

    def _apply_settings_row_style(self, section: tk.Frame, *, hovered: bool) -> None:
        bg = self.current_palette["nav_hover_bg"] if hovered else self.current_palette["app_bg"]
        section.configure(bg=bg)
        caption = getattr(section, "_lumen_settings_caption", None)
        control = getattr(section, "_lumen_settings_control", None)
        divider = getattr(section, "_lumen_settings_divider", None)
        if divider is not None and divider.winfo_exists():
            divider.configure(bg=self.current_palette["panel_divider"])
        if isinstance(caption, tk.Label) and caption.winfo_exists():
            self._style_settings_control_surface(caption, bg=bg, hovered=hovered)
        if control is not None and control.winfo_exists():
            self._style_settings_control_surface(control, bg=bg, hovered=hovered)

    def _style_settings_control_surface(self, widget: tk.Widget, *, bg: str, hovered: bool) -> None:
        if isinstance(widget, tk.Frame):
            widget.configure(bg=bg)
            return
        if isinstance(widget, tk.Button):
            widget.configure(
                bg=self.current_palette["button_hover_bg"] if hovered else self.current_palette["panel_alt_bg"],
                fg=self.current_palette["text_primary"],
                activebackground=self.current_palette["button_hover_bg"],
                activeforeground=self.current_palette["text_primary"],
            )
            return
        if isinstance(widget, tk.Label):
            default_fg = self.current_palette["text_secondary"] if widget is getattr(self, "help_row_indicator", None) else self.current_palette["text_primary"]
            widget.configure(
                bg=bg,
                fg=self.current_palette["text_primary"] if hovered else default_fg,
            )
            return
        if isinstance(widget, tk.Checkbutton):
            widget.configure(
                bg=bg,
                fg=self.current_palette["text_primary"],
                activebackground=bg,
                activeforeground=self.current_palette["text_primary"],
                selectcolor=self.current_palette["panel_alt_bg"],
            )
            return
        if isinstance(widget, tk.Scale):
            widget.configure(
                bg=bg,
                fg=self.current_palette["text_primary"],
                highlightbackground=bg,
                troughcolor=self.current_palette["panel_alt_bg"],
                activebackground=self.current_palette["nav_active_bg"],
            )
            return

    def _settings_reset_palette(self) -> dict[str, str]:
        theme_name = str(self.theme_var.get() or "Dark").strip()
        custom_name = str(self.custom_theme_var.get() or "Lumen Purple").strip()
        if theme_name == "Custom" and custom_name == "Color Wheel" and self.custom_accent_color:
            return custom_accent_palette(self.custom_accent_color)
        return dict(THEME_PALETTES["custom"])

    def _style_settings_reset_button(self, *, hovered: bool) -> None:
        if not hasattr(self, "settings_reset_button") or not self.settings_reset_button.winfo_exists():
            return
        self.settings_reset_button._lumen_hovered = hovered  # type: ignore[attr-defined]
        purple_palette = self._settings_reset_palette()
        background = purple_palette["button_hover_bg"] if hovered else purple_palette["nav_active_border"]
        self.settings_reset_button.configure(
            bg=background,
            fg=purple_palette["text_primary"],
            activebackground=purple_palette["button_hover_bg"],
            activeforeground=purple_palette["text_primary"],
            highlightbackground=purple_palette["nav_active_border"],
            highlightcolor=purple_palette["nav_active_border"],
            disabledforeground=purple_palette["text_secondary"],
        )

    def _top_icon_style(self, *, hovered: bool, primary: bool = False) -> dict[str, str]:
        theme_var = getattr(self, "theme_var", None)
        theme_name = str(theme_var.get() if theme_var is not None else "Dark").strip().lower()
        return resolve_top_icon_palette(
            palette=self.current_palette,
            theme_name=theme_name,
            hovered=hovered,
            primary=primary,
            enabled=True,
        )

    def _apply_top_icon_style(self, button: tk.Button, *, hovered: bool, primary: bool = False) -> None:
        if not button.winfo_exists():
            return
        theme_var = getattr(self, "theme_var", None)
        theme_name = str(theme_var.get() if theme_var is not None else "Dark").strip().lower()
        style = resolve_top_icon_palette(
            palette=self.current_palette,
            theme_name=theme_name,
            hovered=hovered,
            primary=primary,
            enabled=str(button.cget("state")) != str(tk.DISABLED) if hasattr(button, "cget") else True,
        )
        button.configure(
            bg=style["bg"],
            fg=style["fg"],
            activebackground=style["activebackground"],
            activeforeground=style["activeforeground"],
            disabledforeground=style["disabledforeground"],
        )

    def _refresh_top_icon_styles(self) -> None:
        if hasattr(self, "hamburger_button") and self.hamburger_button.winfo_exists():
            self._apply_top_icon_style(
                self.hamburger_button,
                hovered=bool(getattr(self.hamburger_button, "_lumen_hovered", False)),
            )
        if hasattr(self, "top_new_session_button") and self.top_new_session_button.winfo_exists():
            self._apply_top_icon_style(
                self.top_new_session_button,
                hovered=bool(getattr(self.top_new_session_button, "_lumen_hovered", False)),
                primary=True,
            )

    def _reset_style_overrides(self) -> None:
        self.font_family_var.set(self.DEFAULT_FONT_FAMILY)
        for key in self.custom_colors:
            self.custom_colors[key] = None
        self.current_theme = self._resolve_theme_tokens()
        self.current_palette = palette_from_theme(self.current_theme)
        validate_palette(self.current_palette)
        self.root.configure(bg=self.current_palette["app_bg"])
        self._configure_styles()
        self._apply_palette_to_shell(reflow_messages=True, include_assets=True, include_cache=False)
        self.input_box.configure(font=self._message_font(), height=self._input_box_height())
        if hasattr(self, "font_value_label"):
            self.font_value_label.configure(text=self.DEFAULT_FONT_FAMILY)
        self._persist_desktop_preferences_safe()
        self.status_var.set(f"Style reset to {self.theme_var.get()} defaults")

    def _pick_color(self, key: str) -> None:
        self._close_settings_popup()
        _, color = colorchooser.askcolor(parent=self.root)
        if not color:
            return
        self.custom_colors[key] = color
        label = self.color_value_labels.get(key)
        if label is not None and label.winfo_exists():
            label.configure(text=self._format_color_choice(key))
        self._apply_palette_to_shell(reflow_messages=False, include_assets=False, include_cache=False)
        self._persist_desktop_preferences_safe()

    def _format_color_choice(self, key: str) -> str:
        value = str(self.custom_colors.get(key) or "").strip()
        return value.upper() if value else "Choose"

    def _toggle_settings_help(self) -> None:
        self._close_settings_popup()
        visible = not self.settings_help_visible.get()
        self.settings_help_visible.set(visible)
        if visible:
            self.help_text.grid()
            if hasattr(self, "help_row_indicator"):
                self.help_row_indicator.configure(text="Hide")
        else:
            self.help_text.grid_remove()
            if hasattr(self, "help_row_indicator"):
                self.help_row_indicator.configure(text="Show")

    def _show_settings_popup(self, kind: str, *, anchor: tk.Widget | None = None) -> None:
        if self.pending:
            return
        if self.settings_popup_opening:
            return
        target = anchor
        if target is None:
            target = self.settings_row_map.get(
                {
                    "theme": "Theme",
                    "custom_theme": "Theme",
                    "density": "Chat Density",
                    "text_size": "Text Size",
                    "language_style": "Language Style",
                    "font": "Font",
                }.get(kind, ""),
            )
        if target is None or not target.winfo_exists():
            return
        if self.settings_popup is not None and self.settings_popup.winfo_exists():
            if self.settings_popup_kind == kind:
                self.settings_popup.lift()
                self._style_settings_popup()
                return
            self.settings_popup.destroy()
        self.settings_popup_opening = True
        popup = tk.Toplevel(self.root)
        popup.withdraw()
        popup.overrideredirect(True)
        popup.transient(self.root)
        popup.configure(bg=self.current_palette["panel_border"])
        shell = tk.Frame(popup, bg=self.current_palette["panel_bg"], padx=10, pady=10)
        shell.pack(fill=tk.BOTH, expand=True)
        self.settings_popup = popup
        self.settings_popup_kind = kind
        if kind == "theme":
            self._populate_option_popup(
                shell,
                options=["Dark", "Light", "Custom"],
                current=str(self.theme_var.get()),
                on_select=self._select_theme_option,
            )
        elif kind == "custom_theme":
            self._populate_option_popup(
                shell,
                options=["Lumen Purple", "Color Wheel"],
                current=str(self.custom_theme_var.get()),
                on_select=self._select_custom_theme_option,
            )
        elif kind == "density":
            self._populate_option_popup(
                shell,
                options=["Comfortable", "Compact"],
                current=str(self.chat_density_var.get()),
                on_select=self._select_density_option,
            )
        elif kind == "language_style":
            self._populate_option_popup(
                shell,
                options=list(self.LANGUAGE_STYLE_OPTIONS),
                current=str(self.language_style_var.get()),
                on_select=self._select_language_style_option,
            )
        elif kind == "font":
            self._populate_option_popup(
                shell,
                options=list(self.FONT_OPTIONS),
                current=str(self.font_family_var.get()),
                on_select=self._select_font_option,
            )
        elif kind == "text_size":
            self._populate_text_size_popup(shell)
        else:
            popup.destroy()
            self.settings_popup = None
            self.settings_popup_kind = None
            self.settings_popup_opening = False
            return
        self._style_settings_popup()
        anchor_x = target.winfo_rootx()
        anchor_y = target.winfo_rooty()
        anchor_width = target.winfo_width()
        anchor_height = target.winfo_height()
        popup_width = popup.winfo_reqwidth()
        popup_height = popup.winfo_reqheight()
        x = anchor_x + max(0, (anchor_width - popup_width) // 2)
        y = anchor_y + anchor_height + 6
        popup.geometry(f"+{x}+{y}")
        popup.deiconify()
        popup.lift()
        self.root.after_idle(lambda: setattr(self, "settings_popup_opening", False))

    def _populate_option_popup(
        self,
        parent: tk.Frame,
        *,
        options: list[str],
        current: str,
        on_select,
    ) -> None:
        for option in options:
            button = tk.Button(
                parent,
                text=option,
                command=lambda value=option: on_select(value),
                relief=tk.FLAT,
                bd=0,
                padx=12,
                pady=8,
                anchor="w",
                justify=tk.LEFT,
                cursor="hand2",
                highlightthickness=0,
            )
            button.pack(fill=tk.X)
            button._settings_selected = option == current  # type: ignore[attr-defined]
            button.bind("<Enter>", lambda event, widget=button: self._style_popup_option(widget, hovered=True), add="+")
            button.bind("<Leave>", lambda event, widget=button: self._style_popup_option(widget, hovered=False), add="+")
            self._style_popup_option(button, hovered=False)

    def _populate_text_size_popup(self, parent: tk.Frame) -> None:
        value_label = tk.Label(
            parent,
            text=f"Text Size: {self.text_size_var.get()}",
            anchor="center",
            justify=tk.CENTER,
            font=("Segoe UI Semibold", 10),
            bg=self.current_palette["panel_bg"],
            fg=self.current_palette["text_primary"],
        )
        value_label.pack(fill=tk.X, pady=(0, 8))
        scale = tk.Scale(
            parent,
            from_=10,
            to=16,
            orient=tk.HORIZONTAL,
            variable=self.text_size_var,
            command=lambda value: self._on_popup_text_size_changed(value, value_label),
            relief=tk.FLAT,
            bd=0,
            highlightthickness=0,
            showvalue=False,
        )
        scale.pack(fill=tk.X)
        self.popup_text_size_scale = scale
        self._style_settings_control_surface(scale, bg=self.current_palette["panel_bg"], hovered=False)

    def _on_popup_text_size_changed(self, value: object, label: tk.Label) -> None:
        label.configure(text=f"Text Size: {int(float(value))}")
        self._on_text_size_changed()

    def _style_popup_option(self, widget: tk.Button, *, hovered: bool) -> None:
        selected = bool(getattr(widget, "_settings_selected", False))
        bg = self.current_palette["nav_active_bg"] if selected else (
            self.current_palette["button_hover_bg"] if hovered else self.current_palette["panel_bg"]
        )
        widget.configure(
            bg=bg,
            fg=self.current_palette["text_primary"],
            activebackground=self.current_palette["button_hover_bg"],
            activeforeground=self.current_palette["text_primary"],
        )

    def _style_settings_popup(self) -> None:
        if getattr(self, "_custom_color_chooser_active", False):
            return
        popup = getattr(self, "settings_popup", None)
        if popup is None or not popup.winfo_exists():
            return
        popup.configure(bg=self.current_palette["panel_border"])
        for child in popup.winfo_children():
            if isinstance(child, tk.Frame):
                child.configure(bg=self.current_palette["panel_bg"])
                for grandchild in child.winfo_children():
                    if isinstance(grandchild, tk.Button):
                        self._style_popup_option(grandchild, hovered=False)
                    elif isinstance(grandchild, tk.Label):
                        grandchild.configure(bg=self.current_palette["panel_bg"], fg=self.current_palette["text_primary"])
                    elif isinstance(grandchild, tk.Scale):
                        self._style_settings_control_surface(grandchild, bg=self.current_palette["panel_bg"], hovered=False)

    def _close_settings_popup(self) -> None:
        popup = getattr(self, "settings_popup", None)
        if popup is not None and popup.winfo_exists():
            popup.destroy()
        self.settings_popup = None
        self.settings_popup_kind = None
        self.settings_popup_opening = False

    def _activate_theme_row(self) -> None:
        self._show_settings_popup("theme")

    def _activate_density_row(self) -> None:
        self._show_settings_popup("density")

    def _activate_text_size_row(self) -> None:
        self._show_settings_popup("text_size")

    def _activate_emoji_row(self) -> None:
        self._show_settings_popup("language_style")

    def _activate_font_row(self) -> None:
        self._show_settings_popup("font")

    def _select_theme_option(self, value: str) -> None:
        self.theme_var.set(value)
        self._on_theme_changed()
        if value == "Custom":
            self._show_settings_popup("custom_theme")

    def _select_custom_theme_option(self, value: str) -> None:
        self.custom_theme_var.set(value)
        self._on_custom_theme_changed()

    def _select_density_option(self, value: str) -> None:
        self.chat_density_var.set(value)
        self._on_density_changed()

    def _select_language_style_option(self, value: str) -> None:
        self.language_style_var.set(value)
        self.allow_emojis_var.set(value == "Emoji Friendly")
        if hasattr(self, "language_style_value_label"):
            self.language_style_value_label.configure(text=value)
        self._persist_desktop_preferences_safe()

    def _select_font_option(self, value: str) -> None:
        self.font_family_var.set(value)
        self._on_font_changed()

    def _on_text_size_changed(self) -> None:
        self._refresh_message_styles(reflow=True)
        self.input_box.configure(font=self._message_font(), height=self._input_box_height())
        if hasattr(self, "text_size_value_label"):
            self.text_size_value_label.configure(text=str(self.text_size_var.get()))
        self._persist_desktop_preferences_safe()

    def _on_font_changed(self) -> None:
        self._refresh_message_styles(reflow=True)
        self.input_box.configure(font=self._message_font(), height=self._input_box_height())
        self._refresh_live_text_surfaces()
        if hasattr(self, "font_value_label"):
            self.font_value_label.configure(text=str(self.font_family_var.get() or self.DEFAULT_FONT_FAMILY))
        self._persist_desktop_preferences_safe()

    def _checkpoint(self, checkpoint_id: str, phase: str, details: str | None = None) -> None:
        if self.startup_logger is None:
            return
        self.startup_logger.checkpoint(checkpoint_id, phase, details=details)

    def _debug_timing(self, label: str, elapsed_ms: float) -> None:
        if not self._debug_ui_enabled():
            return
        self._emit_debug_ui_line(f"[lumen-ui] {label}: {elapsed_ms:.2f}ms")

    def _debug_event(self, label: str, **fields: object) -> None:
        counts = getattr(self, "_debug_ui_event_counts", None)
        if not isinstance(counts, dict):
            counts = {}
            self._debug_ui_event_counts = counts
        counts[label] = int(counts.get(label, 0)) + 1
        if not self._debug_ui_enabled():
            return
        detail = " ".join(
            f"{key}={value}"
            for key, value in fields.items()
        ).strip()
        suffix = f" {detail}" if detail else ""
        self._emit_debug_ui_line(f"[lumen-ui] {label}#{counts[label]}{suffix}")

    def _emit_debug_ui_line(self, line: str) -> None:
        try:
            sys.stderr.write(f"{line}\n")
        except Exception:
            pass
        log_path = getattr(self, "debug_ui_log_path", None)
        if not isinstance(log_path, Path):
            return
        if bool(getattr(self, "_debug_ui_log_failed", False)):
            return
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{line}\n")
        except Exception:
            self._debug_ui_log_failed = True

    def _start_debug_ui_session(self) -> None:
        if not self._debug_ui_enabled():
            return
        session_label = str(getattr(self._transition_state().debug_session, "session_label", "") or "")
        session_marker = (
            f"[lumen-ui] session_start "
            f"time={datetime.now().isoformat(timespec='seconds')} "
            f"session={session_label} "
            f"mode={self.execution_mode} "
            f"data_root={self.settings.data_root}"
        )
        self._emit_debug_ui_line(session_marker)

    def _record_desktop_crash(
        self,
        *,
        source: str,
        exc: BaseException,
        traceback_obj=None,
        details: str | None = None,
        context: dict[str, object] | None = None,
    ) -> None:
        append_crash_record(
            log_path=self.desktop_crash_log_path,
            record=build_crash_record(
                execution_mode=self.execution_mode,
                source=source,
                exc=exc,
                traceback_obj=traceback_obj,
                current_view=str(getattr(self, "current_view", "") or ""),
                pending_view=str(getattr(self, "pending_view_name", "") or ""),
                deferred_target=str(getattr(self, "deferred_view_refresh_target", "") or ""),
                details=details,
                context=context,
            ),
        )

    def _heavy_surface_context(
        self,
        *,
        view_name: str,
        fetched_count: int,
        rendered_count: int,
        grouped_sections: int,
        has_more: bool,
        render_mode: str,
    ) -> dict[str, object]:
        return {
            "active_view": str(getattr(self, "current_view", "") or ""),
            "surface": str(view_name),
            "fetched_count": int(fetched_count),
            "rendered_count": int(rendered_count),
            "grouped_sections": int(grouped_sections),
            "has_more": bool(has_more),
            "render_mode": str(render_mode),
        }

    def _describe_heavy_surface_context(self, context: dict[str, object]) -> str:
        return ", ".join(f"{key}={value}" for key, value in context.items())

    def _append_runtime_failure_log(
        self,
        *,
        message: str,
        source: str = "runtime",
        category: str = "refresh_failure",
        context: dict[str, object] | None = None,
    ) -> None:
        capability_state = getattr(self, "_desktop_capability_state", None)
        capability_phase = str(getattr(capability_state, "phase", "") or "")
        current_view = str(getattr(self, "current_view", "") or "")
        lines = [
            f"timestamp: {datetime.now().astimezone().isoformat()}",
            f"execution_mode: {self.execution_mode}",
            f"current_view: {current_view or 'unknown'}",
            f"source: {str(source or 'runtime').strip() or 'runtime'}",
            f"category: {str(category or 'refresh_failure').strip() or 'refresh_failure'}",
            f"capability_phase: {capability_phase or 'unknown'}",
            f"message: {str(message or '').strip()}",
        ]
        if context:
            for key, value in context.items():
                lines.append(f"context.{key}: {value}")
        append_runtime_failure_text(
            log_path=self.desktop_runtime_failure_log_path,
            lines=lines,
        )

    def _surface_runtime_failure(
        self,
        message: str,
        *,
        source: str = "runtime",
        category: str = "refresh_failure",
        context: dict[str, object] | None = None,
    ) -> None:
        text = str(message or "").strip()
        if not text:
            return
        self._append_runtime_failure_log(
            message=text,
            source=source,
            category=category,
            context=context,
        )
        self._debug_event(
            "surface_runtime_failure",
            source=source,
            category=category,
            current_view=str(getattr(self, "current_view", "") or ""),
        )
        try:
            self.status_var.set(text)
        except Exception:
            pass

    def _report_tk_callback_exception(self, exc_type, exc_value, exc_traceback) -> None:
        if not isinstance(exc_value, BaseException):
            exc_value = RuntimeError(str(exc_value))
        self._record_desktop_crash(
            source="tk_callback",
            exc=exc_value,
            traceback_obj=exc_traceback,
        )
        self._surface_runtime_failure(
            "Lumen hit a desktop UI issue, but kept the window open. Check desktop_crash.log for details.",
            source="tk_callback",
            category="desktop_ui_issue",
        )

    def _timed_ui_call(self, label: str, func: Callable[[], object]) -> object:
        started = perf_counter()
        result = func()
        self._debug_timing(label, (perf_counter() - started) * 1000.0)
        return result

    def _next_async_token(self, task_name: str) -> int:
        token = int(self._async_task_tokens.get(task_name, 0)) + 1
        self._async_task_tokens[task_name] = token
        self._async_task_started_at[task_name] = perf_counter()
        return token

    def _async_token_current(self, task_name: str, token: int) -> bool:
        return int(self._async_task_tokens.get(task_name, 0)) == int(token)

    def _invalidate_archived_memory_state(self) -> None:
        self.archived_memory_view_dirty = True
        self.archived_memory_state_version = int(getattr(self, "archived_memory_state_version", 0)) + 1
        self.archived_memory_loaded_version = -1
        self.archived_memory_requested_version = None
        self.archived_memory_requested_fetch_limit = None
        self.archived_memory_fetch_in_flight = False
        self._async_task_tokens["archived_memory"] = int(self._async_task_tokens.get("archived_memory", 0)) + 1
        self._async_task_started_at.pop("archived_memory", None)

    def _start_ui_background_task(
        self,
        task_name: str,
        worker: Callable[[], object],
        *,
        timing_label: str | None = None,
    ) -> int:
        token = self._next_async_token(task_name)

        def _run() -> None:
            started = perf_counter()
            try:
                payload = worker()
            except Exception as exc:  # pragma: no cover - background UI path
                if self._destroying:
                    return
                self.ui_task_queue.put(("error", task_name, token, exc))
                return
            elapsed_ms = (perf_counter() - started) * 1000.0
            if self._destroying:
                return
            self.ui_task_queue.put(
                (
                    "result",
                    task_name,
                    token,
                    {
                        "payload": payload,
                        "elapsed_ms": elapsed_ms,
                        "timing_label": timing_label or task_name,
                    },
                )
            )

        Thread(target=_run, daemon=True).start()
        return token

    def _bootstrap_controller(self) -> None:
        if self._controller_bootstrapped:
            return
        self._controller_bootstrapped = True
        self._checkpoint("controller_bootstrap", "before", "construct app controller")
        try:
            self.controller = self._build_controller()
        except Exception as exc:
            self._controller_bootstrapped = False
            if self.startup_logger is not None:
                self.startup_logger.error("controller_bootstrap", exc)
            raise
        self._controller_ready = True
        self._checkpoint("controller_bootstrap", "after", "app controller ready")

    def _build_controller(self) -> AppController:
        startup_hook: Callable[[str, str, str | None], None] = (
            lambda checkpoint_id, phase, details=None: self._checkpoint(checkpoint_id, phase, details)
        )
        try:
            return AppController(
                repo_root=self.repo_root,
                data_root=self.settings.data_root,
                execution_mode=self.execution_mode,
                settings=self.settings,
                startup_hook=startup_hook,
            )
        except TypeError:
            return AppController(
                repo_root=self.repo_root,
                data_root=self.settings.data_root,
                execution_mode=self.execution_mode,
            )

    def _begin_deferred_startup(self) -> None:
        if self._controller_ready:
            self._run_optional_startup_followup()
            return
        self.status_var.set("Starting Lumen...")
        self._set_shell_ready_state(False)
        self.root.after(25, self._finish_deferred_startup)

    def _finish_deferred_startup(self) -> None:
        self._checkpoint("deferred_startup", "before", "complete deferred desktop startup")
        try:
            self._bootstrap_controller()
            self.status_var.set(self._initial_status())
            self._apply_mode_to_session()
            self._set_shell_ready_state(True)
        except Exception as exc:
            self._handle_startup_failure(exc)
            return
        self._checkpoint("deferred_startup", "after", "deferred desktop startup complete")
        self._run_optional_startup_followup()

    def _startup_diagnostics_enabled(self) -> bool:
        return self._debug_ui_enabled()

    def _background_startup_tasks_enabled(self) -> bool:
        return self._debug_ui_enabled()

    def _run_optional_startup_followup(self) -> None:
        post_bootstrap = self._startup_diagnostics_enabled()
        background_tasks = self._background_startup_tasks_enabled()
        if not post_bootstrap and not background_tasks:
            self._debug_event("startup_followup_skipped", diagnostics=False, background_tasks=False)
            return
        self._schedule_startup_followup(
            post_bootstrap=post_bootstrap,
            refresh_views=False,
            background_tasks=background_tasks,
        )

    def _schedule_startup_followup(
        self,
        *,
        post_bootstrap: bool = False,
        refresh_views: bool = False,
        background_tasks: bool = False,
    ) -> None:
        self._startup_followup_needs_post_bootstrap = bool(
            getattr(self, "_startup_followup_needs_post_bootstrap", False)
        ) or post_bootstrap
        self._startup_followup_needs_view_refresh = bool(
            getattr(self, "_startup_followup_needs_view_refresh", False)
        ) or refresh_views
        self._startup_followup_needs_background_tasks = (
            bool(getattr(self, "_startup_followup_needs_background_tasks", False)) or background_tasks
        )
        self._debug_event(
            "startup_followup_requested",
            post_bootstrap=bool(self._startup_followup_needs_post_bootstrap),
            refresh_views=bool(self._startup_followup_needs_view_refresh),
            background_tasks=bool(self._startup_followup_needs_background_tasks),
        )
        if getattr(self, "_startup_followup_job", None) is not None:
            return

        def _apply() -> None:
            self._startup_followup_job = None
            post_bootstrap_needed = self._startup_followup_needs_post_bootstrap
            refresh_views_needed = self._startup_followup_needs_view_refresh
            background_tasks_needed = self._startup_followup_needs_background_tasks
            self._startup_followup_needs_post_bootstrap = False
            self._startup_followup_needs_view_refresh = False
            self._startup_followup_needs_background_tasks = False
            self._debug_event(
                "startup_followup_flushed",
                post_bootstrap=bool(post_bootstrap_needed),
                refresh_views=bool(refresh_views_needed),
                background_tasks=bool(background_tasks_needed),
            )
            if post_bootstrap_needed:
                self._post_startup_bootstrap()
            if refresh_views_needed:
                self._refresh_loaded_views_after_startup()
            if background_tasks_needed:
                self._start_background_startup_tasks()

        self._startup_followup_job = self.root.after_idle(_apply)

    def _refresh_loaded_views_after_startup(self) -> None:
        self._checkpoint("view_refreshes", "before", "refresh startup-dependent views")
        if self.current_view == "chat":
            self._debug_event("startup_view_refresh_skipped", current_view="chat")
            self._checkpoint("view_refreshes", "after", "current_view=chat skipped")
            return
        self.memory_view_dirty = True
        self.recent_sessions_view_dirty = True
        self.archived_sessions_view_dirty = True
        self._invalidate_archived_memory_state()
        if self._view_capability_available(self.current_view):
            self._schedule_view_refresh(self.current_view)
        self._checkpoint("view_refreshes", "after", f"current_view={self.current_view}")

    def _handle_startup_failure(self, exc: Exception) -> None:
        self._set_shell_ready_state(False)
        detail = f"{exc.__class__.__name__}: {exc}"
        self.status_var.set("Startup issue detected")
        self._append_system_line(
            "Lumen opened, but part of startup did not finish. "
            f"Details: {detail}"
        )
        if self.startup_logger is not None:
            self.startup_logger.error("deferred_startup", exc)

    def _start_background_startup_tasks(self) -> None:
        if not self._background_startup_tasks_enabled():
            self._debug_event("background_startup_skipped")
            return
        if self.controller is None or not hasattr(self.controller, "run_deferred_startup_tasks"):
            return
        self._debug_event("background_startup_invoked")
        worker = Thread(target=self._run_background_startup_tasks, daemon=True)
        worker.start()

    def _run_background_startup_tasks(self) -> None:
        controller = self.controller
        if controller is None or not hasattr(controller, "run_deferred_startup_tasks"):
            return
        try:
            controller.run_deferred_startup_tasks()
        except Exception as exc:
            if self.startup_logger is not None:
                self.startup_logger.error("legacy_imports", exc)

    def _set_shell_ready_state(self, ready: bool) -> None:
        self._shell_ready_flag = bool(ready)
        self._apply_control_availability()

    def _apply_control_availability(self) -> None:
        capability_state = getattr(
            self,
            "_desktop_capability_state",
            DesktopCapabilityState.from_runtime(
                missing_bundles=[],
                missing_resources=[],
                capabilities={"bootstrap": {}},
            ),
        )
        availability = resolve_control_availability(
            shell_ready=bool(self._shell_ready_flag),
            pending=bool(self.pending),
            capability_phase=str(capability_state.phase or ""),
            chat_send_ready=bool(capability_state.chat_send_ready),
        )
        self._control_availability = availability
        if hasattr(self, "input_box") and self.input_box.winfo_exists():
            self.input_box.configure(state=availability.chat_state)
        if hasattr(self, "mode_selector") and self.mode_selector.winfo_exists():
            self.mode_selector.configure(state=availability.selector_state)
        if hasattr(self, "display_name_entry") and self.display_name_entry.winfo_exists():
            self.display_name_entry.configure(state="normal" if availability.nav_enabled else "disabled")
        if hasattr(self, "add_button") and self.add_button.winfo_exists():
            self.add_button.configure(state=availability.chat_state)
        if hasattr(self, "hamburger_button") and self.hamburger_button.winfo_exists():
            self.hamburger_button.configure(state=availability.top_level_state)
        if hasattr(self, "top_new_session_button") and self.top_new_session_button.winfo_exists():
            self.top_new_session_button.configure(state=availability.chat_state)
        self._refresh_top_icon_styles()
        for button in getattr(self, "starter_prompt_buttons", []):
            if button.winfo_exists():
                button.configure(state=availability.chat_state)
        for name, button in getattr(self, "nav_buttons", {}).items():
            if button.winfo_exists():
                state = tk.NORMAL if availability.nav_enabled and self._view_capability_available(name) else tk.DISABLED
                button.configure(state=state)
        if hasattr(self, "nav_buttons"):
            self._style_nav_buttons()

    def _view_capability_available(self, view_name: str) -> bool:
        capability_state = getattr(self, "_desktop_capability_state", None)
        if not isinstance(capability_state, DesktopCapabilityState):
            return True
        return bool(capability_state.is_view_enabled(view_name))

    def _view_capability_reason(self, view_name: str) -> str:
        capability_state = getattr(self, "_desktop_capability_state", None)
        if not isinstance(capability_state, DesktopCapabilityState):
            return "That surface is unavailable in this runtime right now."
        return capability_state.reason_for_view(view_name)

    def _gate_view_activation(self, view_name: str) -> bool:
        if self._view_capability_available(view_name):
            return True
        self._clear_hotbar_pending_state(clear_view_name=True)
        self._cancel_deferred_view_refresh()
        self._debug_event("surface_capability_gated", view_name=view_name, phase=self._desktop_capability_state.phase)
        self._surface_runtime_failure(
            self._view_capability_reason(view_name),
            source=f"view_gate.{view_name}",
            category="capability_gated",
        )
        return False

    def _show_window_for_startup(self) -> None:
        self._checkpoint("window_show", "before", "prepare and surface main window")
        try:
            self._ensure_window_geometry_visible()
            if str(self.root.state()).lower() in {"iconic", "withdrawn"}:
                self.root.state("normal")
            self.root.deiconify()
            self.root.lift()
        finally:
            self._checkpoint("window_show", "after", "main window surfaced")

    def _ensure_window_geometry_visible(self) -> None:
        try:
            geometry = str(self.root.geometry() or "")
        except tk.TclError:
            geometry = ""
        width = 1080
        height = 760
        x = None
        y = None
        if "x" in geometry:
            size_part, *position_parts = geometry.split("+")
            try:
                width_text, height_text = size_part.split("x", 1)
                width = max(640, int(width_text))
                height = max(480, int(height_text))
            except (TypeError, ValueError):
                width = 1080
                height = 760
            if len(position_parts) >= 2:
                try:
                    x = int(position_parts[0])
                    y = int(position_parts[1])
                except ValueError:
                    x = None
                    y = None
        screen_width = max(1, int(self.root.winfo_screenwidth() or 1))
        screen_height = max(1, int(self.root.winfo_screenheight() or 1))
        width = min(width, screen_width)
        height = min(height, screen_height)
        if x is None:
            x = max(0, (screen_width - width) // 2)
        if y is None:
            y = max(0, (screen_height - height) // 3)
        x = min(max(0, x), max(0, screen_width - width))
        y = min(max(0, y), max(0, screen_height - height))
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def _mark_first_render_complete(self) -> None:
        self._checkpoint("first_render_complete", "after", "first render callback completed")

    def _initial_status(self) -> str:
        if self.controller is None:
            return "Starting Lumen..."
        provider = self.controller.model_provider
        if provider.provider_id == "local":
            return "Local mode"
        return f"{provider.provider_id} ready"

    def _post_startup_bootstrap(self) -> None:
        if not self._startup_diagnostics_enabled():
            self._debug_event("startup_diagnostics_skipped")
            return
        if self._startup_health_applied:
            return
        if self.controller is None:
            return
        self._startup_health_applied = True
        self._apply_startup_health()

    def _update_clock(self) -> None:
        if self._destroying:
            return
        now = datetime.now()
        self.date_var.set(now.strftime("%B %d, %Y"))
        self.time_var.set(now.strftime("%I:%M %p").lstrip("0"))
        self.daylight_var.set("\u2600" if 6 <= now.hour < 18 else "\u263D")
        self._update_landing_state()
        self.clock_job = self.root.after(30000, self._update_clock)

    def _on_root_destroy(self, event: tk.Event | None = None) -> None:
        if event is not None and event.widget is not self.root:
            return
        if self._destroying:
            return
        self._destroying = True
        for job_name in (
            "queue_drain_job",
            "clock_job",
            "conversation_cache_write_job",
            "deferred_view_refresh_job",
            "hotbar_animation_job",
            "_chat_canvas_layout_job",
            "_startup_followup_job",
            "pending_view_job",
            "pending_animation_job",
        ):
            job = getattr(self, job_name, None)
            if not job:
                continue
            try:
                self.root.after_cancel(job)
            except tk.TclError:
                pass
            setattr(self, job_name, None)
        for canvas_name in (
            "chat_canvas",
            "recent_list_canvas",
            "archived_list_canvas",
            "sidebar_canvas",
            "memory_list_canvas",
            "archived_memory_list_canvas",
            "settings_canvas",
        ):
            canvas = getattr(self, canvas_name, None)
            if canvas is not None:
                self._cancel_scrollbar_visibility_update(canvas)
                self._cancel_canvas_layout_update(canvas)
        if self._chat_scroll_to_bottom_job is not None:
            try:
                self.root.after_cancel(self._chat_scroll_to_bottom_job)
            except tk.TclError:
                pass
            self._chat_scroll_to_bottom_job = None
        for scrollable_id, job in list(getattr(self, "_mousewheel_flush_jobs", {}).items()):
            try:
                self.root.after_cancel(job)
            except tk.TclError:
                pass
            self._mousewheel_flush_jobs.pop(scrollable_id, None)
        for job in list(self.message_reveal_jobs):
            try:
                self.root.after_cancel(job)
            except tk.TclError:
                pass
        self.message_reveal_jobs.clear()
        self._release_tk_resources_for_shutdown()
        try:
            if getattr(self.root, "_lumen_app", None) is self:
                setattr(self.root, "_lumen_app", None)
        except Exception:
            pass

    def _release_tk_resources_for_shutdown(self) -> None:
        for image_attr in ("identity_image", "profile_avatar_image"):
            if hasattr(self, image_attr):
                setattr(self, image_attr, None)
        for widget_attr in ("pending_row", "settings_popup"):
            if hasattr(self, widget_attr):
                setattr(self, widget_attr, None)
        for collection_attr in ("message_labels", "message_text_widgets"):
            collection = getattr(self, collection_attr, None)
            if isinstance(collection, list):
                collection.clear()
        for surface_name in ("recent", "archived", "memory", "archived_memory"):
            self._set_active_browser_hover_row(surface_name, None)
        for variable_attr in (
            "screen_title_var",
            "date_var",
            "time_var",
            "daylight_var",
            "display_name_var",
            "chat_title_var",
            "font_family_var",
            "text_size_var",
            "custom_theme_var",
            "allow_emojis_var",
            "settings_help_visible",
            "theme_var",
            "mode_var",
            "starter_prompt_var",
            "show_starter_prompts_var",
            "chat_density_var",
            "language_style_var",
            "status_var",
            "context_bar_var",
            "mode_descriptor_var",
            "attachment_var",
            "attachment_hint_var",
        ):
            variable = getattr(self, variable_attr, None)
            if variable is None:
                continue
            try:
                variable.set("")
            except Exception:
                pass
            setattr(self, variable_attr, None)

    def _toggle_hotbar(self) -> None:
        self._timed_ui_call("toggle_hotbar", lambda: self._request_hotbar_state(not self.hotbar_open))

    def _request_hotbar_state(self, opening: bool) -> None:
        target_state = bool(opening)
        if self.hotbar_transition_in_progress:
            self._set_pending_hotbar_open_state(target_state)
            return
        if self.hotbar_open == target_state and self.hotbar_current_width == (self.hotbar_target_width if target_state else 0):
            return
        self._set_pending_hotbar_open_state(None)
        self.hotbar_open = target_state
        self._animate_hotbar(opening=target_state)

    def _clear_hotbar_pending_state(self, *, clear_view_name: bool) -> None:
        self._set_pending_hotbar_open_state(None)
        self._set_pending_hotbar_refresh_target(None)
        self._deferred_refresh_from_hotbar_close = False
        if clear_view_name:
            self._set_pending_view_name(None)
            job = getattr(self, "pending_view_job", None)
            if job is not None:
                try:
                    self.root.after_cancel(job)
                except (AttributeError, tk.TclError):
                    pass
                self.pending_view_job = None

    def _collect_click_context(self, widget: tk.Widget | None) -> tuple[bool, bool, bool]:
        current: object | None = widget
        inside_hotbar = False
        inside_settings_popup = False
        inside_settings_rows = False
        while current is not None:
            if current in {self.hotbar, self.hamburger_button}:
                inside_hotbar = True
            if self.settings_popup is not None and current is self.settings_popup:
                inside_settings_popup = True
            if current in getattr(self, "settings_row_sections", []):
                inside_settings_rows = True
            current = getattr(current, "master", None)
        return inside_hotbar, inside_settings_popup, inside_settings_rows

    def _dismiss_settings_popup_for_click(self, *, inside_settings_popup: bool, inside_settings_rows: bool) -> None:
        if self.settings_popup is not None and self.settings_popup.winfo_exists() and not inside_settings_popup and not inside_settings_rows:
            self._close_settings_popup()

    def _dismiss_hotbar_for_click(self, *, inside_hotbar: bool) -> None:
        if self.hotbar_open and not self.hotbar_transition_in_progress and not inside_hotbar:
            self._clear_hotbar_pending_state(clear_view_name=True)
            self._request_hotbar_state(False)

    def _set_top_icon_hover(self, button: tk.Button, hovered: bool) -> None:
        if not button.winfo_exists():
            return
        if bool(getattr(button, "_lumen_hovered", False)) == hovered:
            return
        button._lumen_hovered = hovered  # type: ignore[attr-defined]
        self._apply_top_icon_style(
            button,
            hovered=hovered,
            primary=button is getattr(self, "top_new_session_button", None),
        )

    def _handle_hotbar_destination(self, view_name: str) -> None:
        if view_name != "quit" and not self._gate_view_activation(view_name):
            return
        if self.hotbar_open or self.hotbar_transition_in_progress:
            if self.hotbar_animation_job is not None:
                try:
                    self.root.after_cancel(self.hotbar_animation_job)
                except tk.TclError:
                    pass
                self.hotbar_animation_job = None
            self.hovered_nav = None
            self._clear_hotbar_pending_state(clear_view_name=True)
            self._set_deferred_view_refresh_target(None)
            if self.deferred_view_refresh_job is not None:
                try:
                    self.root.after_cancel(self.deferred_view_refresh_job)
                except tk.TclError:
                    pass
                self.deferred_view_refresh_job = None
            if view_name == "archived_memory":
                self.archived_memory_view_dirty = True
            generation = self._begin_hotbar_navigation(view_name)
            self._debug_event("hotbar_navigation_requested", view_name=view_name, generation=generation)
            self._finalize_hotbar_transition(opening=False)
            return
        self._show_view(view_name)

    def _handle_global_click(self, event: tk.Event) -> None:
        widget = getattr(event, "widget", None)
        if widget is None:
            return
        inside_hotbar, inside_settings_popup, inside_settings_rows = self._collect_click_context(widget)
        self._dismiss_settings_popup_for_click(
            inside_settings_popup=inside_settings_popup,
            inside_settings_rows=inside_settings_rows,
        )
        self._dismiss_hotbar_for_click(inside_hotbar=inside_hotbar)

    def _animate_hotbar(self, *, opening: bool) -> None:
        if self.hotbar_transition_in_progress:
            return
        self.hotbar_transition_in_progress = True
        self._hotbar_transition_started_at = perf_counter()
        if self.hotbar_animation_job is not None:
            try:
                self.root.after_cancel(self.hotbar_animation_job)
            except tk.TclError:
                pass
            self.hotbar_animation_job = None
        start = int(self.hotbar_current_width)
        target = self.hotbar_target_width if opening else 0
        if start == target:
            self._finalize_hotbar_transition(opening=opening)
            return
        if opening:
            self.hotbar.grid()
        self._lock_transition_layout()
        started_at = perf_counter()
        duration = self.HOTBAR_ANIMATION_DURATION
        last_width = start

        def _ease(progress: float) -> float:
            return 1 - (1 - progress) ** 3

        def _step() -> None:
            if not self.hotbar.winfo_exists():
                self._finalize_hotbar_transition(opening=opening)
                return
            progress = min(1.0, (perf_counter() - started_at) / duration)
            eased = _ease(progress)
            next_width = round(start + (target - start) * eased)
            self.hotbar_current_width = next_width
            nonlocal last_width
            if next_width != last_width:
                self.hotbar.configure(width=next_width)
                last_width = next_width
            if progress >= 1.0 or self.hotbar_current_width == target:
                self._finalize_hotbar_transition(opening=opening)
                return
            self.hotbar_animation_job = self.root.after(self.HOTBAR_ANIMATION_INTERVAL_MS, _step)

        self.hotbar_current_width = start
        _step()

    def _finalize_hotbar_transition(self, *, opening: bool) -> None:
        def _apply() -> None:
            target = self.hotbar_target_width if opening else 0
            self.hotbar_open = opening
            self.hotbar_current_width = target
            if hasattr(self, "hotbar") and self.hotbar.winfo_exists():
                self.hotbar.configure(width=target)
                if opening:
                    self.hotbar.grid()
                else:
                    self.hotbar.grid_remove()
            self._unlock_transition_layout()
            self.hotbar_animation_job = None
            self.hotbar_transition_in_progress = False
            started = self._hotbar_transition_started_at
            self._hotbar_transition_started_at = None
            if started is not None:
                self._debug_timing("hotbar_transition_total", (perf_counter() - started) * 1000.0)
            queued_state = self.pending_hotbar_open_state
            self._set_pending_hotbar_open_state(None)
            if queued_state is not None and queued_state != opening:
                self.root.after_idle(lambda state=queued_state: self._request_hotbar_state(state))
                return
            if not opening and self.pending_view_name:
                pending_view, generation = self._take_pending_view_name()
                if pending_view is not None and self.pending_view_job is None:
                    self.pending_view_job = self.root.after_idle(
                        lambda target_view=pending_view, target_generation=generation: self._apply_pending_view(
                            target_view,
                            generation=target_generation,
                        )
                    )
                return
            refresh_target, generation = self._coordinator_take_refresh_target()
            if refresh_target:
                self._set_deferred_view_refresh_target(refresh_target, generation=generation)
                self._deferred_refresh_from_hotbar_close = not opening
                self._queue_deferred_view_refresh()

        self._timed_ui_call(f"finalize_hotbar:{'open' if opening else 'close'}", _apply)

    def _queue_deferred_view_refresh(self) -> None:
        if self.deferred_view_refresh_job is not None:
            return
        self.deferred_view_refresh_job = self.root.after_idle(self._run_deferred_view_refresh)

    def _apply_pending_view(self, view_name: str, *, generation: int = 0) -> None:
        self.pending_view_job = None
        if generation and generation != int(getattr(self, "hotbar_navigation_generation", 0) or 0):
            self._debug_event("hotbar_navigation_discarded", view_name=view_name, generation=generation)
            return
        if generation:
            self._debug_event("hotbar_navigation_consumed", view_name=view_name, generation=generation)
        self._debug_event("apply_pending_view", view_name=view_name)
        self._show_view(view_name)

    def _lock_transition_layout(self) -> None:
        if not hasattr(self, "main_column") or not self.main_column.winfo_exists():
            return
        width = max(1, int(self.main_column.winfo_width() or 0))
        if width <= 1 and hasattr(self, "_transition_locked_width"):
            width = int(getattr(self, "_transition_locked_width", 0) or 0)
        if width <= 1:
            return
        self._transition_locked_width = width
        self.main_column.grid_propagate(False)
        self.main_column.configure(width=width)

    def _unlock_transition_layout(self) -> None:
        if hasattr(self, "main_column") and self.main_column.winfo_exists():
            self.main_column.grid_propagate(True)

    def _show_attach_menu(self) -> None:
        if self.pending:
            return
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label="Add File", command=self._select_attachment_file)
        menu.add_command(label="Add Folder", command=self._select_attachment_folder)
        x = self.add_button.winfo_rootx()
        y = self.add_button.winfo_rooty() - 4
        try:
            menu.tk_popup(x, y)
        finally:
            menu.grab_release()

    def _on_display_name_changed(self, event: tk.Event | None = None) -> None:
        value = str(self.display_name_var.get() or "")[:20]
        if value != self.display_name_var.get():
            self.display_name_var.set(value)
        self._draw_profile_avatar()

    def _confirm_display_name(self, event: tk.Event | None = None) -> str:
        value = (str(self.display_name_var.get() or "").strip() or self.DEFAULT_DISPLAY_NAME)[:20]
        self.display_name_var.set(value)
        try:
            self._save_desktop_preferences()
        except OSError:
            self.status_var.set("Profile name updated for this session")
            self._draw_profile_avatar()
            return "break"
        self._draw_profile_avatar()
        self.status_var.set("Profile name updated")
        return "break"

    def _change_profile_avatar(self, event: tk.Event | None = None) -> None:
        selection = filedialog.askopenfilename(
            parent=self.root,
            title="Choose a Lumen profile image",
            filetypes=(
                ("Image files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp"),
                ("PNG files", "*.png"),
                ("All files", "*.*"),
            ),
        )
        if not selection:
            return
        path = Path(selection)
        image = self._load_circular_avatar_image(path, size=42)
        if image is None:
            self.profile_avatar_path = None
            self.profile_avatar_image = None
            self._draw_profile_avatar()
            self.status_var.set("Profile image could not be loaded")
            return
        self.profile_avatar_path = path
        self.profile_avatar_image = image
        self._draw_profile_avatar()
        self._persist_desktop_preferences_safe()
        self.status_var.set("Profile image updated")

    def _rename_current_chat(self) -> None:
        current_title = str(self.chat_title_var.get() or "Chat").strip() or "Chat"
        new_title = simpledialog.askstring(
            "Rename Chat",
            "Chat title:",
            initialvalue=current_title,
            parent=self.root,
        )
        if new_title is None:
            return
        normalized = str(new_title).strip() or "Chat"
        title = normalized[:40]
        if hasattr(self.controller, "rename_session"):
            try:
                self.controller.rename_session(self.session_id, title=title)
            except Exception:
                self.status_var.set("Chat renamed locally")
        for session in self.recent_sessions_cache:
            if isinstance(session, dict) and str(session.get("session_id") or "") == self.session_id:
                session["title"] = title
                break
        self.chat_title_var.set(title)
        chat_button = self.nav_buttons.get("chat")
        if chat_button is not None:
            chat_button.configure(text=self.chat_title_var.get())
        self._refresh_top_bar_title()
        self.recent_sessions_view_dirty = True

    def _apply_startup_health(self) -> None:
        if self.controller is None:
            return
        self._checkpoint("startup_health", "before", "run desktop startup health check")
        self._debug_event("capability_snapshot_requested", current_view=getattr(self, "current_view", "chat"))

        def _worker() -> dict[str, object]:
            controller = self.controller
            if controller is None:
                return {}
            capabilities: dict[str, object] = {}
            report = controller.build_doctor_report()
            if hasattr(controller, "list_app_capabilities"):
                try:
                    listed = controller.list_app_capabilities()
                    if isinstance(listed, dict):
                        capabilities = listed
                except Exception:
                    capabilities = {}
            check_map = {
                str(item.get("name") or ""): item
                for item in report.get("checks", [])
                if isinstance(item, dict)
            }
            registry_check = check_map.get("tool_registry", {})
            runtime_resources_check = check_map.get("runtime_resources", {})
            runtime_layout_check = check_map.get("runtime_layout", {})
            missing = [
                str(item)
                for item in registry_check.get("missing_bundles", [])
                if str(item).strip()
            ]
            missing_resources = [
                str(item)
                for item in runtime_resources_check.get("missing_required_resources", [])
                if str(item).strip()
            ]
            return {
                "missing": missing,
                "missing_resources": missing_resources,
                "runtime_root": str(runtime_layout_check.get("runtime_root", controller.repo_root)),
                "data_root": str(runtime_layout_check.get("data_root", controller.settings.data_root)),
                "execution_mode": str(controller.execution_mode or ""),
                "debug_ui": self._debug_ui_enabled(),
                "capabilities": capabilities,
                "surface_runtime_ready": {
                    "memory": all(
                        callable(getattr(controller, attribute_name, None))
                        for attribute_name in ("list_personal_memory", "list_research_notes")
                    ),
                },
            }

        self._start_ui_background_task("startup_health", _worker, timing_label="startup_health_fetch")

    @classmethod
    def _read_debug_ui_enabled_env(cls) -> bool:
        return str(os.environ.get(cls.DEBUG_UI_ENV) or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def _debug_ui_enabled(self) -> bool:
        flag = getattr(self, "_debug_ui_flag", None)
        if flag is None:
            return self._read_debug_ui_enabled_env()
        return bool(flag)

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        palette = self.current_palette
        style.configure(
            ".",
            background=palette["panel_bg"],
            foreground=palette["text_primary"],
            bordercolor=palette["panel_border"],
            lightcolor=palette["panel_border"],
            darkcolor=palette["panel_border"],
        )
        style.configure(
            "Dark.TButton",
            background=palette["panel_alt_bg"],
            foreground=palette["text_primary"],
            bordercolor=palette["panel_alt_bg"],
            lightcolor=palette["panel_alt_bg"],
            darkcolor=palette["panel_alt_bg"],
            borderwidth=0,
            relief="flat",
            focusthickness=0,
            focustcolor=palette["panel_alt_bg"],
            padding=(12, 7),
            font=("Segoe UI Semibold", 10),
        )
        style.map(
            "Dark.TButton",
            background=[("active", palette["button_hover_bg"]), ("disabled", palette["system_bg"])],
            foreground=[("disabled", palette["text_secondary"])],
        )
        style.configure(
            "Dark.TCombobox",
            fieldbackground=palette["input_bg"],
            background=palette["panel_alt_bg"],
            foreground=palette["text_primary"],
            arrowcolor=palette["text_secondary"],
            bordercolor=palette["panel_alt_bg"],
            lightcolor=palette["panel_alt_bg"],
            darkcolor=palette["panel_alt_bg"],
            padding=6,
        )
        style.map(
            "Dark.TCombobox",
            fieldbackground=[("readonly", palette["input_bg"])],
            foreground=[("readonly", palette["text_primary"])],
            selectbackground=[("readonly", palette["input_bg"])],
            selectforeground=[("readonly", palette["text_primary"])],
        )
        self.root.option_add("*TCombobox*Listbox*Background", palette["panel_bg"])
        self.root.option_add("*TCombobox*Listbox*Foreground", palette["text_primary"])
        self.root.option_add("*TCombobox*Listbox*selectBackground", palette["nav_active_bg"])
        self.root.option_add("*TCombobox*Listbox*selectForeground", palette["text_primary"])
        self.root.option_add("*Menu.background", palette["panel_bg"])
        self.root.option_add("*Menu.foreground", palette["text_primary"])
        self.root.option_add("*Menu.activeBackground", palette["button_hover_bg"])
        self.root.option_add("*Menu.activeForeground", palette["text_primary"])
        style.configure(
            "Dark.Vertical.TScrollbar",
            background=palette["panel_alt_bg"],
            troughcolor=palette["panel_bg"],
            bordercolor=palette["panel_bg"],
            lightcolor=palette["panel_bg"],
            darkcolor=palette["panel_bg"],
            arrowcolor=palette["text_secondary"],
            arrowsize=11,
            width=10,
        )

    def _start_new_session(self) -> None:
        if self.pending:
            return
        self.controller.reset_session_thread(self.session_id)
        self.session_id = self._new_session_id()
        self.recent_sessions_render_limit = self.recent_sessions_render_step
        self.chat_title_var.set("Chat")
        if "chat" in self.nav_buttons:
            self.nav_buttons["chat"].configure(text="Chat")
        self._apply_mode_to_session()
        self._clear_attachment()
        self._set_chat_text("")
        self.status_var.set(self._initial_status())
        self.context_bar_var.set(build_context_bar(mode_label=self.mode_var.get(), prompt=""))
        self.memory_view_dirty = True
        self.recent_sessions_view_dirty = True
        self.archived_sessions_view_dirty = True
        self._invalidate_archived_memory_state()
        if self.current_view == "recent":
            self._refresh_recent_sessions_view()
        self._show_view("chat")

    def _load_desktop_preferences(self) -> None:
        try:
            if not self.desktop_prefs_path.exists():
                return
            payload = json.loads(self.desktop_prefs_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            return
        if not isinstance(payload, dict):
            return
        display_name = " ".join(str(payload.get("display_name") or "").split()).strip()
        if display_name:
            self.display_name_var.set(display_name[:20])
        stored_theme = str(payload.get("theme") or "").strip()
        if stored_theme in {"Dark", "Light", "Custom"}:
            self.theme_var.set(stored_theme)
        stored_custom_theme = str(payload.get("custom_theme") or "").strip()
        if stored_custom_theme in {"Lumen Purple", "Color Wheel"}:
            self.custom_theme_var.set(stored_custom_theme)
        stored_density = str(payload.get("chat_density") or "").strip()
        if stored_density in {"Comfortable", "Compact"}:
            self.chat_density_var.set(stored_density)
        stored_style = str(payload.get("language_style") or "").strip()
        if stored_style in self.LANGUAGE_STYLE_OPTIONS:
            self.language_style_var.set(stored_style)
            self.allow_emojis_var.set(stored_style == "Emoji Friendly")
        elif isinstance(payload.get("allow_emojis"), bool):
            self.allow_emojis_var.set(bool(payload.get("allow_emojis")))
            self.language_style_var.set("Emoji Friendly" if self.allow_emojis_var.get() else "Standard")
        stored_font = str(payload.get("font_family") or "").strip()
        if stored_font:
            self.font_family_var.set(stored_font)
        stored_interaction_style = str(payload.get("interaction_style") or "").strip().lower()
        if stored_interaction_style in self.STYLE_TO_MODE:
            self.mode_var.set(self.STYLE_TO_MODE[stored_interaction_style])
        try:
            stored_size = int(payload.get("text_size", self.text_size_var.get()))
        except (TypeError, ValueError):
            stored_size = self.text_size_var.get()
        self.text_size_var.set(max(10, min(16, stored_size)))
        stored_accent = str(payload.get("custom_accent_color") or "").strip()
        self.custom_accent_color = stored_accent or None
        avatar_path = str(payload.get("profile_avatar_path") or "").strip()
        if avatar_path:
            self.profile_avatar_path = Path(avatar_path)
            self.profile_avatar_image = None
        stored_colors = payload.get("custom_colors")
        if isinstance(stored_colors, dict):
            for key in self.custom_colors:
                color = stored_colors.get(key)
                self.custom_colors[key] = str(color).strip() if isinstance(color, str) and color else None

    def _save_desktop_preferences(self) -> None:
        self._debug_event("save_desktop_preferences")
        payload = {
            "display_name": str(self.display_name_var.get() or "").strip() or self.DEFAULT_DISPLAY_NAME,
            "profile_avatar_path": str(self.profile_avatar_path) if self.profile_avatar_path is not None else "",
            "font_family": str(self.font_family_var.get() or "").strip() or self.DEFAULT_FONT_FAMILY,
            "text_size": int(self.text_size_var.get() or 11),
            "theme": str(self.theme_var.get() or "Dark").strip() or "Dark",
            "custom_theme": str(self.custom_theme_var.get() or "Lumen Purple").strip() or "Lumen Purple",
            "custom_accent_color": str(self.custom_accent_color or "").strip(),
            "allow_emojis": bool(self.allow_emojis_var.get()),
            "language_style": str(self.language_style_var.get() or "Standard").strip() or "Standard",
            "chat_density": str(self.chat_density_var.get() or "Comfortable").strip() or "Comfortable",
            "interaction_style": self.MODE_OPTIONS.get(self.mode_var.get(), "default"),
            "custom_colors": {key: value for key, value in self.custom_colors.items()},
        }
        self.desktop_prefs_path.parent.mkdir(parents=True, exist_ok=True)
        self.desktop_prefs_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _persist_desktop_preferences_safe(self) -> None:
        if not hasattr(self, "display_name_var") or not hasattr(self, "desktop_prefs_path"):
            return
        try:
            self._save_desktop_preferences()
        except OSError:
            return

    def _send_message(self) -> None:
        if self.pending:
            return
        prompt = self._input_value()
        if not prompt:
            return
        self._submit_prompt(prompt)

    def _submit_prompt(self, prompt: str) -> None:
        if self.controller is None:
            self._append_system_line("Lumen is still starting up. Give it a moment and try again.")
            self.status_var.set("Starting Lumen...")
            return
        attachment_path = self.attached_input_path
        if attachment_path is not None and not attachment_path.exists():
            self._append_system_line(f"Attachment not found: {attachment_path}")
            self.status_var.set("Selected attachment was not found.")
            self._clear_attachment()
            return
        self._hide_input_placeholder()
        self.input_box.delete("1.0", tk.END)
        self._append_chat_message(
            DesktopChatMessage(
                sender="You",
                text=prompt,
                message_type="user",
                timestamp=message_timestamp(),
            )
        )
        self.context_bar_var.set(
            build_context_bar(mode_label=self.mode_var.get(), prompt=prompt)
        )
        self.active_request_id += 1
        request_id = self.active_request_id
        self.pending_request_id = request_id
        self._set_pending(True, prompt=prompt, request_id=request_id)
        self._clear_attachment()
        worker = Thread(target=self._run_ask, args=(request_id, prompt, attachment_path), daemon=True)
        worker.start()

    def _run_ask(self, request_id: int, prompt: str, input_path: Path | None = None) -> None:
        if self.controller is None:
            self.result_queue.put(("error", request_id, RuntimeError("Lumen is still starting up.")))
            return
        try:
            response = self.controller.ask(
                prompt=prompt,
                input_path=input_path,
                session_id=self.session_id,
                client_surface="main",
            )
            self.result_queue.put(("response", request_id, response))
        except Exception as exc:  # pragma: no cover - UI runtime path
            self.result_queue.put(("error", request_id, exc))

    def _drain_queue(self) -> None:
        self._drain_queues_once()
        if self.root.winfo_exists() and not self._destroying:
            self.queue_drain_job = self.root.after(100, self._drain_queue)

    def _drain_queues_once(self) -> None:
        try:
            while True:
                kind, request_id, payload = self.result_queue.get_nowait()
                self._handle_chat_result(kind=kind, request_id=request_id, payload=payload)
        except Empty:
            pass
        try:
            while True:
                kind, task_name, token, payload = self.ui_task_queue.get_nowait()
                self._handle_ui_task_result(kind=kind, task_name=task_name, token=token, payload=payload)
        except Empty:
            pass

    def _handle_chat_result(self, *, kind: str, request_id: int, payload: object) -> None:
        if request_id in self.ignored_request_ids:
            self.ignored_request_ids.discard(request_id)
            self._clear_pending_indicator(request_id=request_id)
            return
        if request_id != self.pending_request_id:
            return
        self._clear_pending_indicator(request_id=request_id)
        if kind == "response":
            response = payload
            message = self.presenter.render(
                response,
                decorate=True,
                style=self.MODE_OPTIONS.get(self.mode_var.get(), "default"),
                recent_assistant_texts=self._recent_assistant_texts(),
            )
            self._append_chat_message(
                DesktopChatMessage(
                    sender="Lumen",
                    text=message,
                    message_type="assistant",
                    mode=str(response.get("mode") or ""),
                    timestamp=message_timestamp(),
                )
            )
            self.status_var.set(self.presenter.build_status(response))
            self.context_bar_var.set(
                build_context_bar(
                    mode_label=self.mode_var.get(),
                    response=response,
                )
            )
        else:
            error = payload
            self._append_chat_message(
                DesktopChatMessage(
                    sender="Lumen",
                    text=f"Error: {error}",
                    message_type="assistant",
                    timestamp=message_timestamp(),
                )
            )
            self.status_var.set(f"Error: {error}")
            self.context_bar_var.set(
                build_context_bar(mode_label=self.mode_var.get(), prompt="")
            )
        self._debug_chat_save_state()
        self._set_pending(False)
        self.pending_request_id = 0
        self.memory_view_dirty = True
        self.recent_sessions_view_dirty = True
        self.archived_sessions_view_dirty = True
        self._invalidate_archived_memory_state()
        if self.current_view == "memory":
            self._refresh_memory_view()
        elif self.current_view == "recent":
            self._refresh_recent_sessions_view()
        elif self.current_view == "archived":
            self._refresh_archived_sessions_view()
        elif self.current_view == "archived_memory":
            self._refresh_archived_memory_view()

    def _debug_chat_save_state(self) -> None:
        session_id = str(getattr(self, "session_id", "") or "")
        controller = getattr(self, "controller", None)
        data_root = str(getattr(getattr(controller, "settings", None), "data_root", "") or "")
        session_exists = False
        message_count: int | None = None
        interaction_record_count: int | None = None
        error: str | None = None
        try:
            persistence = getattr(controller, "persistence_manager", None)
            sessions = getattr(persistence, "sessions", None)
            messages = getattr(persistence, "messages", None)
            if sessions is not None and session_id:
                session_exists = sessions.get(session_id) is not None
            if messages is not None and session_id:
                message_count = len(messages.list_by_session(session_id))
            if controller is not None and hasattr(controller, "list_interactions") and session_id:
                report = controller.list_interactions(session_id=session_id)
                records = report.get("interaction_records", []) if isinstance(report, dict) else []
                interaction_record_count = len(records) if isinstance(records, list) else 0
        except Exception as exc:  # pragma: no cover - diagnostics only
            error = str(exc)
        self._debug_event(
            "chat_save_state",
            session_id=session_id,
            data_root=data_root,
            session_exists=session_exists,
            message_count=message_count,
            interaction_record_count=interaction_record_count,
            error=error,
        )

    def _handle_ui_task_result(self, *, kind: str, task_name: str, token: int, payload: object) -> None:
        if not self._async_token_current(task_name, token):
            return
        started = self._async_task_started_at.get(task_name)
        if kind == "error":
            if task_name == "startup_health":
                self._checkpoint("startup_health", "after", "desktop startup health check failed")
                if self.startup_logger is not None and isinstance(payload, Exception):
                    self.startup_logger.error("startup_health", payload)
                self.status_var.set("Startup issue detected")
                self._append_system_line(
                    "Startup warning: health checks could not complete. "
                    f"Details: {payload}"
                )
                self._record_desktop_crash(source="ui_task.startup_health", exc=payload if isinstance(payload, BaseException) else RuntimeError(str(payload)))
            elif task_name in {"memory", "archived_memory", "recent", "archived"}:
                if task_name == "archived_memory":
                    self.archived_memory_fetch_in_flight = False
                    self.archived_memory_requested_version = None
                    self.archived_memory_requested_fetch_limit = None
                    self.archived_memory_view_dirty = True
                elif task_name == "memory":
                    self.memory_fetch_in_flight = False
                    self.memory_requested_fetch_limit = None
                    self.memory_view_dirty = True
                elif task_name == "recent":
                    self.recent_sessions_fetch_in_flight = False
                    self.recent_sessions_view_dirty = True
                elif task_name == "archived":
                    self.archived_sessions_view_dirty = True
                self._record_desktop_crash(
                    source=f"ui_task.{task_name}",
                    exc=payload if isinstance(payload, BaseException) else RuntimeError(str(payload)),
                )
                self._surface_runtime_failure(
                    f"Lumen couldn't refresh {task_name.replace('_', ' ')}. The current view was left in place.",
                    source=f"ui_task.{task_name}",
                    category="refresh_failure",
                    context={
                        "task_name": task_name,
                        "current_view": str(getattr(self, "current_view", "") or ""),
                    },
                )
            return
        if not isinstance(payload, dict):
            return
        elapsed_ms = float(payload.get("elapsed_ms", 0.0) or 0.0)
        timing_label = str(payload.get("timing_label") or task_name).strip() or task_name
        self._debug_timing(timing_label, elapsed_ms)
        if started is not None:
            self._debug_timing(f"{task_name}_total_latency", (perf_counter() - started) * 1000.0)
        result_payload = payload.get("payload")
        if task_name == "startup_health" and isinstance(result_payload, dict):
            self._apply_startup_health_result(result_payload)
            self._checkpoint("startup_health", "after", "desktop startup health check complete")
        elif task_name == "recent" and isinstance(result_payload, dict):
            self._apply_recent_sessions_result(result_payload)
        elif task_name == "archived" and isinstance(result_payload, dict):
            self._apply_archived_sessions_result(result_payload)
        elif task_name == "memory" and isinstance(result_payload, dict):
            self._apply_memory_view_result(result_payload)
        elif task_name == "archived_memory" and isinstance(result_payload, dict):
            self._apply_archived_memory_view_result(result_payload)

    def _apply_startup_health_result(self, result: dict[str, object]) -> None:
        missing = [str(item) for item in result.get("missing", []) if str(item).strip()]
        missing_resources = [str(item) for item in result.get("missing_resources", []) if str(item).strip()]
        runtime_root = str(result.get("runtime_root") or "")
        data_root = str(result.get("data_root") or "")
        capability_state = capability_state_from_startup_health(result)
        self._desktop_capability_state = capability_state
        self._apply_control_availability()
        snapshot = capability_state.summary_payload()
        self._debug_event(
            "capability_snapshot_applied",
            phase=str(snapshot.get("phase") or ""),
            degraded_reason=str(snapshot.get("degraded_reason") or ""),
            missing_bundles="|".join(snapshot.get("missing_bundles", [])),
            missing_resources="|".join(snapshot.get("missing_resources", [])),
            enabled_surfaces="|".join(snapshot.get("enabled_surfaces", [])),
            disabled_surfaces="|".join(snapshot.get("disabled_surfaces", [])),
            memory_runtime_ready=str(
                (
                    result.get("surface_runtime_ready", {}).get("memory")
                    if isinstance(result.get("surface_runtime_ready"), dict)
                    else "unknown"
                )
            ),
            memory_surfaces_ready=capability_state.memory_surfaces_ready,
            memory_ready_source=str(snapshot.get("surface_ready_sources", {}).get("memory") or ""),
            recent_ready_source=str(snapshot.get("surface_ready_sources", {}).get("recent") or ""),
            chat_ready_source=str(snapshot.get("surface_ready_sources", {}).get("chat") or ""),
        )
        try:
            if self.startup_logger is not None:
                self.startup_logger.checkpoint(
                    "capability_snapshot",
                    "after",
                    details=(
                        f"phase={snapshot.get('phase')} "
                        f"enabled={','.join(snapshot.get('enabled_surfaces', []))} "
                        f"disabled={','.join(snapshot.get('disabled_surfaces', []))} "
                        f"degraded_reason={snapshot.get('degraded_reason', '')} "
                        f"missing_bundles={','.join(snapshot.get('missing_bundles', []))} "
                        f"missing_resources={','.join(snapshot.get('missing_resources', []))} "
                        f"memory_runtime_ready="
                        f"{(result.get('surface_runtime_ready', {}).get('memory') if isinstance(result.get('surface_runtime_ready'), dict) else 'unknown')} "
                        f"memory_surfaces_ready={capability_state.memory_surfaces_ready} "
                        f"memory_ready_source={snapshot.get('surface_ready_sources', {}).get('memory', '')}"
                    ),
                )
        except Exception:
            pass
        self._reconcile_active_surface_after_capability_snapshot()
        if missing or missing_resources:
            self.status_var.set("Startup issue detected")
            details: list[str] = []
            if missing:
                details.append(f"Missing bundles: {', '.join(missing)}")
            if missing_resources:
                details.append(f"Missing resources: {', '.join(missing_resources)}")
            self._append_system_line(
                "Startup warning: core tools are unavailable. "
                + " ".join(details)
                + ". Check the runtime root and packaged resources. "
                + f"Runtime root: {runtime_root} | Data root: {data_root}"
            )
            return
        if str(result.get("execution_mode") or "") == "frozen" and bool(result.get("debug_ui")):
            self._append_system_line(
                f"Runtime root: {runtime_root} | Data root: {data_root}"
            )

    def _reconcile_active_surface_after_capability_snapshot(self) -> None:
        current_view = str(getattr(self, "current_view", "") or "")
        if current_view == "memory" and hasattr(self, "memory_list_inner"):
            self._debug_event("memory_surface_state_reconcile", view="memory")
            self._refresh_memory_view()
        elif current_view == "archived_memory" and hasattr(self, "archived_memory_list_inner"):
            self._debug_event("memory_surface_state_reconcile", view="archived_memory")
            self._refresh_archived_memory_view()

    def _set_pending(self, pending: bool, *, prompt: str = "", request_id: int | None = None) -> None:
        self.pending = pending
        ready = self.controller is not None
        self._apply_control_availability()
        self.stop_button.configure(state=tk.NORMAL if pending else tk.DISABLED)
        if pending:
            self.stop_button.grid()
        else:
            self.stop_button.grid_remove()
        self.mic_button.configure(state=tk.DISABLED if not pending else tk.DISABLED)
        if pending:
            self._close_settings_popup()
        self._refresh_attachment_state()
        if pending:
            self.status_var.set("Lumen is working...")
            self.add_button.grid_remove()
            self.mic_button.grid_remove()
            if request_id is not None:
                self.pending_row_request_id = request_id
            self._show_pending_indicator(prompt, request_id=request_id)
        else:
            self._clear_pending_indicator(request_id=request_id, force=request_id is None)
            self.pending_row_request_id = 0
            self.add_button.grid()
            self.mic_button.grid()
            availability = getattr(self, "_control_availability", None)
            chat_ready = bool(ready and isinstance(availability, DesktopControlAvailability) and availability.chat_ready)
            self.input_box.configure(
                state=tk.NORMAL if chat_ready else tk.DISABLED
            )
            if chat_ready and not self._input_has_user_text():
                self._show_input_placeholder()
            if chat_ready:
                self.input_box.focus_set()

    def _stop_current_task(self) -> None:
        if not self.pending or self.pending_request_id <= 0:
            return
        confirmed = messagebox.askyesno(
            "Stop Lumen",
            "Stop the current task? Lumen will keep the partial conversation and wait for your next direction.",
            parent=self.root,
        )
        if not confirmed:
            return
        self.ignored_request_ids.add(self.pending_request_id)
        stopped_request_id = self.pending_request_id
        self.pending_request_id = 0
        self._clear_pending_indicator(request_id=stopped_request_id, force=True)
        self._set_pending(False, request_id=stopped_request_id)
        self.status_var.set("Stopped. Waiting for your next direction.")
        self.context_bar_var.set(build_context_bar(mode_label=self.mode_var.get(), prompt=""))

    def _select_attachment_file(self) -> None:
        if self.pending:
            return
        selection = filedialog.askopenfilename(
            parent=self.root,
            title="Select a file or zip for Lumen",
            filetypes=(
                ("All files", "*.*"),
                ("Zip archives", "*.zip"),
            ),
        )
        if not selection:
            return
        self._set_attachment(Path(selection), kind="file")

    def _select_attachment_folder(self) -> None:
        if self.pending:
            return
        selection = filedialog.askdirectory(
            parent=self.root,
            title="Select a folder for Lumen",
            mustexist=True,
        )
        if not selection:
            return
        self._set_attachment(Path(selection), kind="folder")

    def _set_attachment(self, path: Path, *, kind: str) -> None:
        self.attached_input_path = path.expanduser()
        self.attached_input_kind = kind
        self._refresh_attachment_state()

    def _clear_attachment(self) -> None:
        self.attached_input_path = None
        self.attached_input_kind = ""
        self._refresh_attachment_state()

    def _refresh_attachment_state(self) -> None:
        path = self.attached_input_path
        if path is None:
            self.attachment_var.set("No file, folder, or zip selected.")
            self.attachment_hint_var.set("Attach one file, folder, or zip for the next send.")
        else:
            self.attachment_var.set(
                f"{self._attachment_kind_label(path)}: {self._short_attachment_label(path)}"
            )
            self.attachment_hint_var.set(
                "This attachment will be used for the next send, then cleared automatically."
            )
        if hasattr(self, "attachment_label"):
            self.attachment_label.configure(
                fg=self.current_palette["text_secondary"] if path else self.current_palette["text_muted"]
            )
        if hasattr(self, "clear_attachment_button"):
            if self.pending or path is None:
                self.clear_attachment_button.grid_remove()
            else:
                self.clear_attachment_button.grid(row=0, column=1, sticky="e")
                self.clear_attachment_button.configure(state=tk.NORMAL)

    def _attachment_kind_label(self, path: Path) -> str:
        if self.attached_input_kind == "folder" or path.is_dir():
            return "Folder"
        if path.suffix.lower() == ".zip":
            return "Zip"
        return "File"

    @staticmethod
    def _short_attachment_label(path: Path, *, max_length: int = 72) -> str:
        text = str(path)
        if len(text) <= max_length:
            return text
        head = max(8, max_length // 2 - 2)
        tail = max(8, max_length - head - 3)
        return f"{text[:head]}...{text[-tail:]}"

    def _show_pending_indicator(self, prompt: str, *, request_id: int | None = None) -> None:
        if self.pending_row is not None:
            return
        self.pending_base_text = build_pending_label(mode_label=self.mode_var.get(), prompt=prompt).rstrip(".")
        self.pending_dot_step = 3
        if request_id is not None:
            self.pending_row_request_id = request_id
        pending_message = DesktopChatMessage(
            sender="Lumen",
            text=f"{self.pending_base_text}...",
            message_type="pending",
            mode=self.MODE_OPTIONS.get(self.mode_var.get(), "default"),
            timestamp=message_timestamp(),
        )
        self.pending_row = self._append_chat_message(pending_message)
        self._schedule_pending_animation()

    def _clear_pending_indicator(self, *, request_id: int | None = None, force: bool = False) -> None:
        if not force and request_id is not None and self.pending_row_request_id not in {0, request_id}:
            return
        if self.pending_animation_job is not None:
            try:
                self.root.after_cancel(self.pending_animation_job)
            except tk.TclError:
                pass
            self.pending_animation_job = None
        self.pending_text_widget = None
        if self.pending_row is None:
            return
        self.pending_row.destroy()
        self.pending_row = None
        self.messages = [message for message in self.messages if message.message_type != "pending"]
        self._scroll_chat_to_bottom()

    def _append_chat_message(
        self,
        message: DesktopChatMessage,
        *,
        auto_scroll: bool = True,
        animate: bool = True,
    ) -> tk.Widget:
        row = self._render_chat_message(message, store_message=True, auto_scroll=auto_scroll, animate=animate)
        self._update_landing_state()
        return row

    def _render_chat_message(
        self,
        message: DesktopChatMessage,
        *,
        store_message: bool,
        auto_scroll: bool = True,
        animate: bool = True,
    ) -> tk.Widget:
        if store_message:
            self.messages.append(message)
        role = message_role_style(message.message_type, palette=self.current_palette)
        if message.message_type == "user":
            role["bubble_bg"] = str(self.custom_colors.get("user_bg") or role["bubble_bg"])
            role["text_fg"] = str(self.custom_colors.get("user_text") or role["text_fg"])
        elif message.message_type == "assistant":
            role["bubble_bg"] = str(self.custom_colors.get("assistant_bg") or role["bubble_bg"])
            role["text_fg"] = str(self.custom_colors.get("assistant_text") or role["text_fg"])
        row = tk.Frame(self.chat_frame, bg=self.current_palette["app_bg"])
        row.pack(fill=tk.X, pady=self._row_pady())

        bubble = tk.Frame(
            row,
            bg=role["bubble_bg"],
            highlightbackground=role["bubble_border"],
            highlightcolor=role["bubble_border"],
            highlightthickness=1,
            bd=0,
            padx=self._bubble_padx(),
            pady=self._bubble_pady(),
        )
        width_pad = self._bubble_side_padding()
        if role["anchor"] == "e":
            bubble.pack(anchor="e", padx=(width_pad, 0))
        elif role["anchor"] == "w":
            bubble.pack(anchor="w", padx=(0, width_pad))
        else:
            bubble.pack(anchor="center", padx=width_pad)

        meta_label: tk.Label | None = None
        if message.message_type != "system":
            meta_text = message.sender
            if message.timestamp:
                meta_text = f"{message.sender}  {message.timestamp}"
            meta_label = tk.Label(
                bubble,
                text=meta_text,
                bg=role["bubble_bg"],
                fg=role["sender_fg"],
                font=self._meta_font(),
                anchor="w",
            )
            meta_label.pack(anchor="w", pady=(0, 4))

        text_widget = tk.Text(
            bubble,
            wrap=tk.WORD,
            bg=role["bubble_bg"],
            fg=role["text_fg"],
            font=self._message_font(),
            relief=tk.FLAT,
            highlightthickness=0,
            bd=0,
            padx=0,
            pady=0,
            cursor="xterm",
            takefocus=0,
            undo=False,
        )
        text_widget.pack(anchor="w", fill=tk.X)
        self._configure_readonly_text(
            text_widget,
            text="" if message.message_type == "assistant" and animate else message.text,
            bubble_bg=role["bubble_bg"],
            text_fg=role["text_fg"],
        )
        text_widget._lumen_bubble_widget = bubble  # type: ignore[attr-defined]
        text_widget._lumen_row_widget = row  # type: ignore[attr-defined]
        text_widget._lumen_meta_label = meta_label  # type: ignore[attr-defined]
        text_widget._lumen_message_type = message.message_type  # type: ignore[attr-defined]
        self.message_text_widgets.append(text_widget)
        if meta_label is not None:
            self.message_labels.append(meta_label)

        if auto_scroll:
            self._scroll_chat_to_bottom()
        if animate:
            self._animate_message_presence(
                message=message,
                bubble=bubble,
                text_widget=text_widget,
                meta_label=meta_label,
            )
        if message.message_type == "pending":
            self.pending_row = row
            self.pending_text_widget = text_widget
        return row

    def _schedule_pending_animation(self) -> None:
        if not self.pending or self.pending_text_widget is None:
            self.pending_animation_job = None
            return

        def _tick() -> None:
            if not self.pending or self.pending_text_widget is None or not self.pending_text_widget.winfo_exists():
                self.pending_animation_job = None
                return
            self.pending_dot_step = (self.pending_dot_step % 3) + 1
            animated_text = f"{self.pending_base_text}{'.' * self.pending_dot_step}"
            self.pending_text_widget.configure(state=tk.NORMAL)
            self.pending_text_widget.delete("1.0", tk.END)
            self.pending_text_widget.insert("1.0", animated_text)
            self.pending_text_widget.configure(state=tk.DISABLED)
            self.pending_animation_job = self.root.after(420, _tick)

        self.pending_animation_job = self.root.after(420, _tick)

    def _animate_message_presence(
        self,
        *,
        message: DesktopChatMessage,
        bubble: tk.Frame,
        text_widget: tk.Text,
        meta_label: tk.Label | None,
    ) -> None:
        if message.message_type != "assistant":
            return
        self._debug_event("assistant_reveal_start", text_length=len(str(message.text or "")))
        muted_fg = self.current_palette["text_secondary"]
        final_fg = str(getattr(text_widget, "_lumen_text_fg", self.current_palette["text_primary"]))
        full_text = str(message.text or "")
        if not full_text:
            return
        text_widget.configure(state=tk.NORMAL, fg=muted_fg)
        text_widget.delete("1.0", tk.END)
        text_widget.configure(state=tk.DISABLED)
        text_widget._lumen_reveal_index = 0  # type: ignore[attr-defined]
        if meta_label is not None:
            meta_label.configure(fg=self.current_palette["text_muted"])
        bubble.configure(pady=max(4, self._bubble_pady() - 2))

        step = max(24, min(120, len(full_text) // 5 or 24))

        def _request_reveal_scroll() -> None:
            if hasattr(self, "chat_canvas") and callable(getattr(self, "_scroll_chat_to_bottom", None)):
                self._scroll_chat_to_bottom()

        prior_job = getattr(text_widget, "_lumen_reveal_job", None)
        if prior_job:
            try:
                self.root.after_cancel(prior_job)
            except tk.TclError:
                pass
            self.message_reveal_jobs.discard(prior_job)
            text_widget._lumen_reveal_job = None  # type: ignore[attr-defined]

        def _finish() -> None:
            if not bubble.winfo_exists() or not text_widget.winfo_exists():
                return
            job = getattr(text_widget, "_lumen_reveal_job", None)
            if job:
                self.message_reveal_jobs.discard(job)
                text_widget._lumen_reveal_job = None  # type: ignore[attr-defined]
            bubble.configure(pady=self._bubble_pady())
            text_widget.configure(state=tk.NORMAL)
            text_widget._lumen_text = full_text  # type: ignore[attr-defined]
            text_widget.configure(fg=final_fg)
            text_widget.configure(height=self._estimate_text_widget_height(text=full_text))
            text_widget.configure(state=tk.DISABLED)
            if meta_label is not None and meta_label.winfo_exists():
                meta_label.configure(fg=self.current_palette["text_secondary"])
            _request_reveal_scroll()
            self._debug_event("assistant_reveal_flush", text_length=len(full_text))

        def _tick(index: int = 0) -> None:
            if not bubble.winfo_exists() or not text_widget.winfo_exists():
                job = getattr(text_widget, "_lumen_reveal_job", None)
                if job:
                    self.message_reveal_jobs.discard(job)
                    text_widget._lumen_reveal_job = None  # type: ignore[attr-defined]
                self._debug_event("assistant_reveal_cancel", reason="widget_destroyed")
                return
            prior_index = int(getattr(text_widget, "_lumen_reveal_index", 0) or 0)
            next_index = min(len(full_text), index + step)
            text_widget.configure(state=tk.NORMAL)
            if next_index > prior_index:
                text_widget.insert(tk.END, full_text[prior_index:next_index])
            current_text = full_text[:next_index]
            text_widget._lumen_text = current_text  # type: ignore[attr-defined]
            text_widget.configure(height=self._estimate_text_widget_height(text=current_text))
            text_widget.configure(state=tk.DISABLED)
            text_widget._lumen_reveal_index = next_index  # type: ignore[attr-defined]
            _request_reveal_scroll()
            if next_index >= len(full_text):
                _finish()
                return
            job = self.root.after(28, lambda: _tick(next_index))
            text_widget._lumen_reveal_job = job  # type: ignore[attr-defined]
            self.message_reveal_jobs.add(job)

        _tick()

    def _append_system_line(self, message: str) -> None:
        self._append_chat_message(
            DesktopChatMessage(
                sender="System",
                text=message,
                message_type="system",
                timestamp=message_timestamp(),
            )
        )

    def _set_chat_text(self, value: str) -> None:
        for child in self.chat_frame.winfo_children():
            child.destroy()
        self.pending_row = None
        self.message_labels = []
        self.message_text_widgets = []
        self.messages = []
        if value:
            self._append_system_line(value)
        self._update_landing_state()

    def _build_saved_chat_messages(
        self,
        ordered_records: list[dict[str, object]],
    ) -> tuple[list[DesktopChatMessage], bool, int, int]:
        messages: list[DesktopChatMessage] = []
        had_render_issue = False
        restored_user_count = 0
        restored_assistant_count = 0
        for record in ordered_records:
            try:
                prompt = self._saved_chat_prompt_text(record)
                timestamp = self._timestamp_from_record(record)
                if prompt:
                    messages.append(
                        DesktopChatMessage(
                            sender="You",
                            text=prompt,
                            message_type="user",
                            timestamp=timestamp,
                        )
                    )
                    restored_user_count += 1
                response = self._saved_chat_response_payload(record)
                assistant_text = self._saved_chat_assistant_text(record, response=response)
                if assistant_text:
                    messages.append(
                        DesktopChatMessage(
                            sender="Lumen",
                            text=assistant_text,
                            message_type="assistant",
                            mode=str(record.get("mode") or ""),
                            timestamp=timestamp,
                        )
                    )
                    restored_assistant_count += 1
            except Exception:
                had_render_issue = True
                continue
        return messages, had_render_issue, restored_user_count, restored_assistant_count

    @staticmethod
    def _saved_chat_prompt_text(record: dict[str, object]) -> str:
        prompt = str(record.get("prompt") or "").strip()
        if prompt:
            return prompt
        prompt_view = record.get("prompt_view") if isinstance(record.get("prompt_view"), dict) else {}
        return str(
            prompt_view.get("canonical_prompt")
            or prompt_view.get("resolved_prompt")
            or prompt_view.get("original_prompt")
            or ""
        ).strip()

    @staticmethod
    def _saved_chat_response_payload(record: dict[str, object]) -> dict[str, object]:
        return dict(record.get("response") or {}) if isinstance(record.get("response"), dict) else {}

    def _saved_chat_assistant_text(
        self,
        record: dict[str, object],
        *,
        response: dict[str, object],
    ) -> str:
        for candidate in (
            str(response.get("user_facing_answer") or "").strip(),
            str(response.get("reply") or "").strip(),
            str(response.get("summary") or "").strip(),
            str(record.get("user_facing_answer") or "").strip(),
            str(record.get("reply") or "").strip(),
            str(record.get("summary") or "").strip(),
        ):
            if candidate:
                return candidate
        return self.presenter.render(response or record)

    def _scroll_chat_to_bottom(self) -> None:
        self._debug_event("chat_scroll_request")
        if self._chat_scroll_to_bottom_job is not None:
            return

        def _apply() -> None:
            self._chat_scroll_to_bottom_job = None
            if not hasattr(self, "chat_canvas") or not self.chat_canvas.winfo_exists():
                return
            self._debug_event("chat_scroll_flush")
            self.chat_canvas.yview_moveto(1.0)

        self._chat_scroll_to_bottom_job = self.root.after_idle(_apply)

    def _scroll_restored_chat_to_reading_position(self) -> None:
        """Keep short restored transcripts readable from the top; long threads open latest."""
        if self._chat_scroll_to_bottom_job is not None:
            try:
                self.root.after_cancel(self._chat_scroll_to_bottom_job)
            except tk.TclError:
                pass
            self._chat_scroll_to_bottom_job = None

        def _apply() -> None:
            if not hasattr(self, "chat_canvas") or not self.chat_canvas.winfo_exists():
                return
            try:
                self.root.update_idletasks()
                bbox = self.chat_canvas.bbox("all")
                content_height = int((bbox[3] - bbox[1]) if bbox else 0)
                viewport_height = max(1, int(self.chat_canvas.winfo_height() or 1))
                if content_height <= viewport_height + 24:
                    self.chat_canvas.yview_moveto(0.0)
                else:
                    self.chat_canvas.yview_moveto(1.0)
                self._debug_event(
                    "saved_session_scroll_positioned",
                    content_height=content_height,
                    viewport_height=viewport_height,
                    at_top=content_height <= viewport_height + 24,
                )
            except tk.TclError:
                return

        self.root.after_idle(_apply)

    @staticmethod
    def _register_scroll_owner(canvas: tk.Canvas, scrollbar: ttk.Scrollbar) -> None:
        setattr(canvas, "_lumen_scrollbar", scrollbar)

    def _bind_canvas_layout(
        self,
        *,
        inner: tk.Widget,
        canvas: tk.Canvas,
        scrollbar: ttk.Scrollbar | None,
        window_id: int | None,
    ) -> None:
        self._register_scroll_owner(canvas, scrollbar)
        setattr(canvas, "_lumen_layout_window_id", window_id)
        setattr(canvas, "_lumen_layout_scrollbar", scrollbar)
        setattr(canvas, "_lumen_layout_job", None)
        setattr(canvas, "_lumen_layout_width", None)
        setattr(canvas, "_lumen_layout_needs_scrollregion", False)
        inner.bind(
            "<Configure>",
            lambda event, target=canvas: self._request_canvas_layout(target, needs_scrollregion=True),
        )
        canvas.bind(
            "<Configure>",
            lambda event, target=canvas: self._request_canvas_layout(
                target,
                width=int(getattr(event, "width", 0) or 0),
                needs_scrollregion=False,
            ),
        )

    def _sync_canvas_layout(
        self,
        canvas: tk.Canvas,
        scrollbar: ttk.Scrollbar | None,
        *,
        window_id: int | None = None,
        width: int | None = None,
    ) -> None:
        if scrollbar is not None:
            setattr(canvas, "_lumen_layout_scrollbar", scrollbar)
        if window_id is not None:
            setattr(canvas, "_lumen_layout_window_id", window_id)
        self._request_canvas_layout(canvas, width=width, needs_scrollregion=True)

    def _request_canvas_layout(
        self,
        canvas: tk.Canvas,
        *,
        width: int | None = None,
        needs_scrollregion: bool = False,
        after_callback=None,
    ) -> None:
        if width is not None:
            setattr(canvas, "_lumen_layout_width", int(width))
        if needs_scrollregion:
            setattr(canvas, "_lumen_layout_needs_scrollregion", True)
        if after_callback is not None:
            setattr(canvas, "_lumen_layout_after_callback", after_callback)
        if getattr(canvas, "_lumen_layout_job", None) is not None:
            return
        surface = str(getattr(canvas, "_lumen_layout_surface", canvas))
        self._debug_event("shell_layout_requested", surface=surface)

        def _apply() -> None:
            setattr(canvas, "_lumen_layout_job", None)
            self._flush_canvas_layout(canvas)

        job = self.root.after_idle(_apply)
        setattr(canvas, "_lumen_layout_job", job)
        if canvas is getattr(self, "chat_canvas", None):
            self._chat_canvas_layout_job = job

    def _flush_canvas_layout(self, canvas: tk.Canvas) -> None:
        exists = getattr(canvas, "winfo_exists", None)
        if callable(exists) and not exists():
            return
        width = getattr(canvas, "_lumen_layout_width", None)
        setattr(canvas, "_lumen_layout_width", None)
        window_id = getattr(canvas, "_lumen_layout_window_id", None)
        if window_id is not None and width is not None and int(width) > 0:
            canvas.itemconfigure(window_id, width=int(width))
        if bool(getattr(canvas, "_lumen_layout_needs_scrollregion", False)):
            canvas.configure(scrollregion=canvas.bbox("all"))
            setattr(canvas, "_lumen_layout_needs_scrollregion", False)
        scrollbar = getattr(canvas, "_lumen_layout_scrollbar", None)
        self._toggle_scrollbar_visibility(canvas, scrollbar, defer=False)
        after_callback = getattr(canvas, "_lumen_layout_after_callback", None)
        setattr(canvas, "_lumen_layout_after_callback", None)
        if callable(after_callback):
            after_callback(int(width or 0))
        if canvas is getattr(self, "chat_canvas", None):
            self._chat_canvas_layout_job = None
        surface = str(getattr(canvas, "_lumen_layout_surface", canvas))
        self._debug_event("shell_layout_flushed", surface=surface)

    def _cancel_canvas_layout_update(self, canvas: tk.Canvas) -> None:
        job = getattr(canvas, "_lumen_layout_job", None)
        if not job:
            return
        try:
            self.root.after_cancel(job)
        except (AttributeError, tk.TclError):
            pass
        setattr(canvas, "_lumen_layout_job", None)
        if canvas is getattr(self, "chat_canvas", None):
            self._chat_canvas_layout_job = None

    def _schedule_scrollbar_visibility_update(self, canvas: tk.Canvas, scrollbar: ttk.Scrollbar | None) -> None:
        if scrollbar is None or not scrollbar.winfo_exists():
            return
        if getattr(canvas, "_lumen_scrollbar_job", None) is not None:
            return

        def _apply() -> None:
            canvas._lumen_scrollbar_job = None  # type: ignore[attr-defined]
            self._toggle_scrollbar_visibility(canvas, scrollbar, defer=False)

        canvas._lumen_scrollbar_job = self.root.after_idle(_apply)  # type: ignore[attr-defined]

    def _cancel_scrollbar_visibility_update(self, canvas: tk.Canvas) -> None:
        job = getattr(canvas, "_lumen_scrollbar_job", None)
        if not job:
            return
        try:
            self.root.after_cancel(job)
        except (AttributeError, tk.TclError):
            pass
        canvas._lumen_scrollbar_job = None  # type: ignore[attr-defined]

    def _toggle_scrollbar_visibility(
        self,
        canvas: tk.Canvas,
        scrollbar: ttk.Scrollbar | None,
        *,
        defer: bool = False,
    ) -> None:
        if scrollbar is None or not scrollbar.winfo_exists():
            return
        if defer:
            self._schedule_scrollbar_visibility_update(canvas, scrollbar)
            return
        bbox = canvas.bbox("all")
        needs_scroll = bool(bbox and (bbox[3] - bbox[1]) > int(canvas.winfo_height() or 0))
        if needs_scroll:
            if not scrollbar.winfo_manager():
                scrollbar.grid(row=0, column=1, sticky="ns")
        elif scrollbar.winfo_manager():
            scrollbar.grid_remove()

    def _on_chat_frame_configure(self, event: tk.Event) -> None:
        self._chat_canvas_layout_needs_scrollregion = True
        self._schedule_chat_canvas_layout()

    def _on_chat_canvas_configure(self, event: tk.Event) -> None:
        self._chat_canvas_layout_width = int(getattr(event, "width", 0) or 0)
        self._schedule_chat_canvas_layout()

    def _schedule_chat_canvas_layout(self) -> None:
        width = int(self._chat_canvas_layout_width or self.chat_canvas.winfo_width() or 0)
        needs_scrollregion = bool(self._chat_canvas_layout_needs_scrollregion)
        self._chat_canvas_layout_width = None
        self._chat_canvas_layout_needs_scrollregion = False

        def _after_layout(applied_width: int) -> None:
            width_value = applied_width if applied_width > 0 else width
            self._refresh_message_wraps()
            self._refresh_landing_icon_geometry(width_value if width_value > 0 else None)

        setattr(self.chat_canvas, "_lumen_layout_window_id", self.chat_window)
        setattr(self.chat_canvas, "_lumen_layout_scrollbar", self.chat_scrollbar)
        self._request_canvas_layout(
            self.chat_canvas,
            width=width if width > 0 else None,
            needs_scrollregion=needs_scrollregion,
            after_callback=_after_layout,
        )
        self._chat_canvas_layout_job = getattr(self.chat_canvas, "_lumen_layout_job", None)

    def _bubble_wraplength(self) -> int:
        return bubble_wraplength(self.chat_canvas.winfo_width())

    def _bubble_side_padding(self) -> int:
        return bubble_side_padding(self.chat_canvas.winfo_width())

    def _refresh_message_wraps(self) -> None:
        live_labels: list[tk.Label] = []
        for label in self.message_labels:
            if label.winfo_exists():
                label.configure(font=self._meta_font())
                live_labels.append(label)
        self.message_labels = live_labels
        live_text_widgets: list[tk.Text] = []
        for widget in self.message_text_widgets:
            if widget.winfo_exists():
                self._refresh_readonly_text_widget(widget)
                live_text_widgets.append(widget)
        self.message_text_widgets = live_text_widgets

    def _refresh_live_text_surfaces(self) -> None:
        def _apply() -> None:
            resolved_family = self._resolved_font_family()
            if hasattr(self, "greeting_label") and self.greeting_label.winfo_exists():
                self.greeting_label.configure(font=(resolved_family, 20))
            if hasattr(self, "greeting_subtitle") and self.greeting_subtitle.winfo_exists():
                self.greeting_subtitle.configure(font=(resolved_family, 11))
            if hasattr(self, "screen_title_label") and self.screen_title_label.winfo_exists():
                self.screen_title_label.configure(font=(resolved_family, 16))
            if hasattr(self, "context_bar_label") and self.context_bar_label.winfo_exists():
                self.context_bar_label.configure(font=(resolved_family, 9))
            if hasattr(self, "mode_descriptor_label") and self.mode_descriptor_label.winfo_exists():
                self.mode_descriptor_label.configure(font=(resolved_family, 8))
            if hasattr(self, "status_label") and self.status_label.winfo_exists():
                self.status_label.configure(font=(resolved_family, 9))
            for button in getattr(self, "starter_prompt_buttons", []):
                if button.winfo_exists():
                    button.configure(font=(resolved_family, 9))
            self._refresh_message_wraps()
            self._update_landing_state()
            try:
                self.root.update_idletasks()
            except tk.TclError:
                pass
        self._timed_ui_call("refresh_live_text_surfaces", _apply)

    def _refresh_message_styles(self, *, reflow: bool) -> None:
        def _apply() -> None:
            if reflow:
                self._rerender_messages()
                self._refresh_live_text_surfaces()
                return
            for widget in self.message_text_widgets:
                if not widget.winfo_exists():
                    continue
                message_type = str(getattr(widget, "_lumen_message_type", "assistant") or "assistant")
                role = message_role_style(message_type, palette=self.current_palette)
                if message_type == "user":
                    role["bubble_bg"] = str(self.custom_colors.get("user_bg") or role["bubble_bg"])
                    role["text_fg"] = str(self.custom_colors.get("user_text") or role["text_fg"])
                elif message_type == "assistant":
                    role["bubble_bg"] = str(self.custom_colors.get("assistant_bg") or role["bubble_bg"])
                    role["text_fg"] = str(self.custom_colors.get("assistant_text") or role["text_fg"])
                widget._lumen_bubble_bg = role["bubble_bg"]  # type: ignore[attr-defined]
                widget._lumen_text_fg = role["text_fg"]  # type: ignore[attr-defined]
                bubble = getattr(widget, "_lumen_bubble_widget", None)
                if bubble is not None and bubble.winfo_exists():
                    bubble.configure(
                        bg=role["bubble_bg"],
                        highlightbackground=role["bubble_border"],
                        highlightcolor=role["bubble_border"],
                    )
                row = getattr(widget, "_lumen_row_widget", None)
                if row is not None and row.winfo_exists():
                    row.configure(bg=self.current_palette["app_bg"])
                meta_label = getattr(widget, "_lumen_meta_label", None)
                if meta_label is not None and meta_label.winfo_exists():
                    meta_label.configure(
                        bg=role["bubble_bg"],
                        fg=role["sender_fg"],
                        font=self._meta_font(),
                    )
                self._refresh_readonly_text_widget(widget)
            self._refresh_live_text_surfaces()
        self._timed_ui_call("message_styles" if not reflow else "message_rerender", _apply)

    def _on_theme_changed(self, event: tk.Event | None = None) -> None:
        theme_name = str(self.theme_var.get() or "Dark").strip().lower()
        self._debug_event("theme_change", theme=theme_name, current_view=getattr(self, "current_view", "chat"))
        self.current_theme = self._resolve_theme_tokens(theme_name)
        self.current_palette = palette_from_theme(self.current_theme)
        validate_palette(self.current_palette)
        self.root.configure(bg=self.current_palette["app_bg"])
        self._cancel_deferred_view_refresh()
        self._schedule_theme_apply()

    def _resolve_theme_tokens(self, theme_name: str | None = None) -> dict[str, str]:
        normalized = str(theme_name or self.theme_var.get() or "Dark").strip().lower()
        accent = self.custom_accent_color if normalized == "custom" and self.custom_theme_var.get() == "Color Wheel" else None
        return resolve_theme_tokens(normalized, custom_accent_hex=accent)

    def _resolve_theme_palette(self, theme_name: str | None = None) -> dict[str, str]:
        return palette_from_theme(self._resolve_theme_tokens(theme_name))

    def _resolved_custom_palette(self) -> dict[str, str]:
        return palette_from_theme(self._resolve_theme_tokens("custom"))

    def _on_custom_theme_changed(self, event: tk.Event | None = None) -> None:
        choice = str(self.custom_theme_var.get() or "Lumen Purple").strip() or "Lumen Purple"
        if choice == "Color Wheel":
            prior_choice = "Color Wheel" if self.custom_accent_color else "Lumen Purple"
            prior_accent = self.custom_accent_color
            self._close_settings_popup()
            self._custom_color_chooser_active = True
            try:
                _, color = colorchooser.askcolor(
                    color=self.custom_accent_color or self.current_palette["nav_active_border"],
                    parent=self.root,
                    title="Choose a Lumen accent",
                )
            finally:
                self._custom_color_chooser_active = False
            if not color:
                self.custom_theme_var.set(prior_choice)
                self.custom_accent_color = prior_accent
                return
            self.custom_accent_color = color
        else:
            self.custom_accent_color = None
        if self.theme_var.get() != "Custom":
            self.theme_var.set("Custom")
        self._on_theme_changed()

    def _cancel_deferred_view_refresh(self) -> None:
        job = getattr(self, "deferred_view_refresh_job", None)
        if not job:
            return
        try:
            self.root.after_cancel(job)
        except (AttributeError, tk.TclError):
            pass
        self.deferred_view_refresh_job = None

    def _schedule_theme_apply(self) -> None:
        self._set_theme_apply_requested(True)
        if self._theme_apply_in_progress or getattr(self, "_theme_apply_job", None) is not None:
            return
        should_defer = bool(getattr(self, "_launch_first_boot", False)) or bool(
            getattr(self, "hotbar_transition_in_progress", False)
        ) or getattr(self, "hotbar_animation_job", None) is not None
        scheduler = getattr(self.root, "after_idle", None)
        if should_defer and callable(scheduler):
            self._theme_apply_job = scheduler(self._flush_theme_apply)
            return
        self._flush_theme_apply()

    def _stabilize_theme_transition_state(self) -> None:
        job = getattr(self, "hotbar_animation_job", None)
        if job is not None:
            try:
                self.root.after_cancel(job)
            except (AttributeError, tk.TclError):
                pass
            self.hotbar_animation_job = None
        if getattr(self, "hotbar_transition_in_progress", False):
            target_width = int(getattr(self, "hotbar_target_width", 0) or 0)
            target = target_width if self.hotbar_open else 0
            self.hotbar_current_width = target
            if hasattr(self, "hotbar") and self.hotbar.winfo_exists():
                self.hotbar.configure(width=target)
                if self.hotbar_open:
                    self.hotbar.grid()
                else:
                    self.hotbar.grid_remove()
            self._unlock_transition_layout()
            self.hotbar_transition_in_progress = False
            if self.pending_hotbar_open_state == self.hotbar_open:
                self._set_pending_hotbar_open_state(None)
            self._debug_event("hotbar_transition_theme_settled", opening=bool(self.hotbar_open))

    def _flush_theme_apply(self) -> None:
        self._theme_apply_job = None
        if not self._theme_apply_requested or self._theme_apply_in_progress:
            return
        self._set_theme_apply_in_progress(True)
        try:
            while self._theme_apply_requested:
                self._set_theme_apply_requested(False)
                self._debug_event("theme_apply_flush", current_view=str(getattr(self, "current_view", "chat") or "chat"))
                self._stabilize_theme_transition_state()
                self._configure_styles()
                self._apply_palette_to_shell(
                    reflow_messages=False,
                    include_assets=False,
                    include_cache=False,
                )
                current_view = str(getattr(self, "current_view", "chat") or "chat")
                if current_view != "chat":
                    self._schedule_view_refresh(current_view)
                self._persist_desktop_preferences_safe()
        finally:
            self._set_theme_apply_in_progress(False)

    def _apply_palette_to_shell(
        self,
        *,
        reflow_messages: bool = False,
        include_assets: bool = False,
        include_cache: bool = False,
    ) -> None:
        self._debug_event(
            "apply_palette_to_shell",
            reflow_messages=reflow_messages,
            include_assets=include_assets,
            include_cache=include_cache,
        )
        palette = self.current_palette
        self.container.configure(bg=palette["app_bg"])
        self.top_bar.configure(bg=palette["app_bg"])
        self.top_bar_left.configure(bg=palette["app_bg"])
        self.top_bar_center.configure(bg=palette["app_bg"])
        self.top_bar_right.configure(bg=palette["app_bg"])
        self.screen_title_label.configure(bg=palette["app_bg"], fg=palette["text_primary"])
        self.date_label.configure(bg=palette["app_bg"], fg=palette["text_secondary"])
        self.time_label.configure(bg=palette["app_bg"], fg=palette["text_primary"])
        self.daylight_label.configure(bg=palette["app_bg"], fg=palette["text_secondary"])
        self._refresh_top_icon_styles()
        self.body_container.configure(bg=palette["app_bg"])
        self.hotbar.configure(bg=palette["sidebar_bg"])
        self.hotbar_profile_block.configure(bg=palette["sidebar_bg"])
        self.hotbar_nav_block.configure(bg=palette["sidebar_bg"])
        if hasattr(self, "hotbar_footer"):
            self.hotbar_footer.configure(bg=palette["sidebar_bg"])
        self.profile_avatar.configure(bg=palette["sidebar_bg"])
        self.hotbar_hint.configure(bg=palette["sidebar_bg"], fg=palette["text_muted"])
        self.display_name_entry.configure(
            bg=palette["panel_bg"],
            fg=palette["text_primary"],
            insertbackground=palette["text_primary"],
            disabledbackground=palette["panel_bg"],
            disabledforeground=palette["text_secondary"],
        )
        self._draw_profile_avatar()
        self.content_container.configure(bg=palette["app_bg"])
        if hasattr(self, "main_column"):
            self.main_column.configure(bg=palette["app_bg"])
        if hasattr(self, "theme_value_label"):
            theme_value = str(self.theme_var.get() or "Dark")
            if theme_value == "Custom":
                theme_value = f"Custom • {self.custom_theme_var.get() or 'Lumen Purple'}"
            self.theme_value_label.configure(bg=palette["app_bg"], fg=palette["text_primary"], text=theme_value)
        if hasattr(self, "theme_value_mask"):
            self.theme_value_mask.configure(bg=palette["app_bg"])
        if hasattr(self, "density_value_label"):
            self.density_value_label.configure(bg=palette["app_bg"], fg=palette["text_primary"])
        if hasattr(self, "font_value_label"):
            self.font_value_label.configure(bg=palette["app_bg"], fg=palette["text_primary"])
        if hasattr(self, "text_size_value_label"):
            self.text_size_value_label.configure(bg=palette["app_bg"], fg=palette["text_primary"])
        if hasattr(self, "language_style_value_label"):
            self.language_style_value_label.configure(bg=palette["app_bg"], fg=palette["text_primary"])
        for key, label in getattr(self, "color_value_labels", {}).items():
            if label.winfo_exists():
                label.configure(
                    bg=palette["app_bg"],
                    fg=palette["text_primary"],
                    text=self._format_color_choice(key),
                )
        self._style_nav_buttons()
        for button in self.starter_prompt_buttons:
            self._style_starter_prompt_chip(button)
        self.input_frame.configure(
            bg=palette["app_bg"],
            highlightbackground=palette["app_bg"],
            highlightcolor=palette["app_bg"],
        )
        self.context_bar_label.configure(bg=palette["app_bg"], fg=palette["text_secondary"])
        self.attachment_row.configure(bg=palette["app_bg"])
        input_palette = resolve_input_palette(palette=palette, placeholder_active=bool(self.placeholder_active))
        self.input_box.configure(
            bg=input_palette["bg"],
            fg=input_palette["fg"],
            insertbackground=input_palette["insertbackground"],
            font=self._message_font(),
            height=self._input_box_height(),
        )
        self.composer_frame.configure(bg=palette["panel_bg"])
        self.mode_pill.configure(bg=palette["panel_bg"])
        self.mode_descriptor_label.configure(bg=palette["panel_bg"], fg=palette["text_muted"])
        for button in (self.add_button, self.mic_button, self.clear_attachment_button, self.starter_prompt_apply_button):
            button_palette = resolve_composer_button_palette(
                palette=palette,
                primary=button in {self.add_button, self.mic_button},
            )
            button.configure(**button_palette)
        self.stop_button.configure(style="Dark.TButton")
        self.attachment_label.configure(
            bg=palette["app_bg"],
            fg=palette["text_secondary"] if self.attached_input_path else palette["text_muted"],
        )
        self.attachment_hint_label.configure(
            bg=palette["app_bg"],
            fg=palette["text_muted"],
        )
        for view_name in ("chat", "memory", "archived_memory", "recent", "archived", "settings"):
            if view_name in self.views:
                self._apply_palette_to_view(view_name)
        self._style_settings_popup()
        self._style_settings_reset_button(hovered=bool(getattr(getattr(self, "settings_reset_button", None), "_lumen_hovered", False)))
        self._refresh_message_styles(reflow=reflow_messages)
        if include_assets:
            self._load_identity_icon()
        if include_cache:
            self._schedule_conversation_cache_write()

    def _rerender_messages(self) -> None:
        def _apply() -> None:
            preserved = list(self.messages)
            self._debug_event("message_rerender_batch", message_count=len(preserved))
            for child in self.chat_frame.winfo_children():
                child.destroy()
            self.pending_row = None
            self.message_labels = []
            self.message_text_widgets = []
            self.messages = []
            for message in preserved:
                self._render_chat_message(
                    message,
                    store_message=True,
                    auto_scroll=False,
                    animate=False,
                )
            if preserved:
                self._scroll_chat_to_bottom()
        self._timed_ui_call("message_rerender", _apply)

    def _on_return_pressed(self, event: tk.Event) -> str:
        self._send_message()
        return "break"

    @staticmethod
    def _on_shift_return_pressed(event: tk.Event) -> None:
        return None

    @staticmethod
    def _new_session_id() -> str:
        return datetime.now().strftime("desktop-%Y%m%d-%H%M%S-%f")

    def _on_mode_changed(self, event: tk.Event | None = None) -> None:
        self._debug_event("mode_change", mode=self.mode_var.get(), pending=bool(getattr(self, "pending", False)))
        if not self._begin_mode_apply():
            return
        try:
            self._apply_mode_to_session()
            self.status_var.set(f"{self.mode_var.get()} mode")
            self.mode_descriptor_var.set(self.MODE_DESCRIPTORS.get(self.mode_var.get(), "balanced"))
            self.context_bar_var.set(build_context_bar(mode_label=self.mode_var.get(), prompt=""))
            self._persist_desktop_preferences_safe()
            if str(getattr(self, "current_view", "chat") or "chat") == "chat":
                self._update_landing_state()
        finally:
            self._finish_mode_apply()

    def _apply_mode_to_session(self) -> None:
        if self.controller is None:
            return
        interaction_style = self.MODE_OPTIONS.get(self.mode_var.get(), "default")
        self._debug_event("apply_mode_to_session", interaction_style=interaction_style, session_id=self.session_id)
        self.controller.set_session_profile(
            self.session_id,
            interaction_style=interaction_style,
        )

    def _use_starter_prompt(self) -> None:
        prompt = str(self.starter_prompt_var.get() or "").strip()
        if not prompt or self.pending:
            return
        self._hide_input_placeholder()
        self.input_box.delete("1.0", tk.END)
        self.input_box.insert("1.0", prompt)
        self.input_box.focus_set()
        self.input_box.mark_set(tk.INSERT, tk.END)

    def _show_view(self, view_name: str) -> None:
        def _apply() -> None:
            self._debug_event(
                "show_view",
                view_name=view_name,
                current_view=getattr(self, "current_view", "chat"),
                hotbar_open=bool(getattr(self, "hotbar_open", False)),
                hotbar_transition=bool(getattr(self, "hotbar_transition_in_progress", False)),
            )
            if view_name == "quit":
                self.root.destroy()
                return
            if view_name != "chat" and not self._gate_view_activation(view_name):
                return
            self._close_settings_popup()
            if (
                view_name != "chat"
                and view_name != self.current_view
                and (self.hotbar_open or self.hotbar_transition_in_progress)
            ):
                generation = self._begin_hotbar_navigation(view_name)
                self._debug_event("hotbar_navigation_requested", view_name=view_name, generation=generation)
                if self.hotbar_open and not self.hotbar_transition_in_progress:
                    self._toggle_hotbar()
                return
            if self._should_skip_duplicate_view_show(view_name):
                self._debug_event("show_view_suppressed", view_name=view_name, reason="already_current_and_stable")
                return
            if not self._ensure_view_built(view_name):
                return
            self._apply_view_visibility(view_name)
        self._timed_ui_call(f"show_view:{view_name}", _apply)

    def _should_skip_duplicate_view_show(self, view_name: str) -> bool:
        normalized_view = str(view_name or "")
        current_view = str(getattr(self, "current_view", "chat") or "chat")
        if normalized_view != current_view:
            return False
        if normalized_view == "chat":
            return not bool(getattr(self, "pending", False))
        if bool(getattr(self, "hotbar_transition_in_progress", False)):
            return False
        dirty_by_view = {
            "recent": bool(getattr(self, "recent_sessions_view_dirty", False)),
            "archived": bool(getattr(self, "archived_sessions_view_dirty", False)),
            "memory": bool(getattr(self, "memory_view_dirty", False)),
            "archived_memory": bool(getattr(self, "archived_memory_view_dirty", False)),
            "settings": False,
        }
        return not dirty_by_view.get(normalized_view, False)

    def _ensure_view_built(self, view_name: str) -> bool:
        if view_name in self.views:
            self._debug_event("ensure_view_built", view_name=view_name, built=True, remount=False)
            return True
        builder = self.deferred_view_builders.get(view_name)
        if builder is None:
            return False
        self._debug_event("ensure_view_built", view_name=view_name, built=False, remount=True)
        builder()
        if view_name in self.views:
            self._apply_palette_to_view(view_name)
        return view_name in self.views

    def _apply_view_visibility(self, view_name: str, *, schedule_refresh: bool = True) -> None:
        previous_view = getattr(self, "current_view", "chat")
        self._debug_event("apply_view_visibility", view_name=view_name, previous_view=previous_view)
        for name, frame in self.views.items():
            if name == view_name:
                frame.grid(row=0, column=0, sticky="nsew")
            else:
                frame.grid_forget()
        for surface in tuple(getattr(self, "_active_browser_hover_rows", {}).keys()):
            if surface != view_name:
                self._set_active_browser_hover_row(surface, None)
        if previous_view != view_name:
            self._teardown_inactive_heavy_views(active_view=view_name)
        self.current_view = view_name
        if view_name == "chat":
            self.input_frame.grid()
            self._update_landing_state()
        else:
            self.input_frame.grid_remove()
            if hasattr(self, "landing_frame"):
                self.landing_frame.place_forget()
            self._refresh_top_bar_title()
        self._style_nav_buttons()
        if schedule_refresh:
            self._schedule_view_refresh(view_name)

    def _surface_async_task_name(self, view_name: str) -> str | None:
        normalized = str(view_name or "").strip().lower()
        if normalized in self.HEAVY_SURFACE_VIEWS:
            return normalized
        return None

    def _cancel_surface_runtime_work(self, view_name: str) -> None:
        task_name = self._surface_async_task_name(view_name)
        if task_name:
            self._async_task_tokens[task_name] = int(self._async_task_tokens.get(task_name, 0)) + 1
            self._async_task_started_at.pop(task_name, None)
        if view_name == "recent":
            self.recent_sessions_fetch_in_flight = False
            self.recent_sessions_requested_signature = None
            self.recent_sessions_view_dirty = bool(self.recent_sessions_view_dirty) or not bool(self.recent_sessions_cache)
        elif view_name == "archived":
            self.archived_sessions_fetch_in_flight = False
            self.archived_sessions_requested_signature = None
            self.archived_sessions_view_dirty = bool(self.archived_sessions_view_dirty) or not bool(self.archived_sessions_cache)
        elif view_name == "memory":
            self.memory_fetch_in_flight = False
            self.memory_requested_fetch_limit = None
            self.memory_view_dirty = bool(self.memory_view_dirty) or not bool(self.memory_entries)
        elif view_name == "archived_memory":
            self.archived_memory_fetch_in_flight = False
            self.archived_memory_requested_version = None
            self.archived_memory_requested_fetch_limit = None
            self.archived_memory_view_dirty = bool(self.archived_memory_view_dirty) or not bool(self.archived_memory_entries)
        if getattr(self, "deferred_view_refresh_target", None) == view_name:
            self._cancel_deferred_view_refresh()
            self._set_deferred_view_refresh_target(None)
        if getattr(self, "pending_hotbar_refresh_target", None) == view_name:
            self._set_pending_hotbar_refresh_target(None)

    def _cancel_mousewheel_flush_for_widget(self, widget: tk.Widget | None) -> None:
        if widget is None:
            return
        scrollable_id = id(widget)
        job = self._mousewheel_flush_jobs.pop(scrollable_id, None)
        if not job:
            return
        try:
            self.root.after_cancel(job)
        except tk.TclError:
            pass

    def _reset_heavy_surface_window_state(self, view_name: str) -> None:
        step = self.HEAVY_SURFACE_RENDER_STEP
        if view_name == "recent":
            self.recent_sessions_render_limit = step
            self.recent_sessions_rendered_count = 0
            self.recent_sessions_load_more_button = None
        elif view_name == "archived":
            self.archived_sessions_render_limit = step
            self.archived_sessions_rendered_count = 0
            self.archived_sessions_load_more_button = None
        elif view_name == "memory":
            self.memory_render_limit = step
            self.memory_rendered_count = 0
            self.memory_load_more_button = None
        elif view_name == "archived_memory":
            self.archived_memory_render_limit = step
            self.archived_memory_rendered_count = 0
            self.archived_memory_load_more_button = None

    def _teardown_heavy_surface(self, view_name: str) -> None:
        if view_name not in self.HEAVY_SURFACE_VIEWS:
            return
        frame = self.views.pop(view_name, None)
        if frame is None:
            return
        self._cancel_surface_runtime_work(view_name)
        self._set_active_browser_hover_row(view_name, None)
        attr_names: tuple[str, ...]
        if view_name == "recent":
            attr_names = (
                "recent_list_canvas",
                "recent_sessions_scrollbar",
                "recent_list_inner",
                "recent_list_window",
                "recent_sessions_load_more_button",
            )
        elif view_name == "archived":
            attr_names = (
                "archived_list_canvas",
                "archived_sessions_scrollbar",
                "archived_list_inner",
                "archived_list_window",
                "archived_sessions_load_more_button",
            )
        elif view_name == "memory":
            attr_names = (
                "memory_preview",
                "memory_list_canvas",
                "memory_list_scrollbar",
                "memory_list_inner",
                "memory_list_window",
                "memory_load_more_button",
            )
        else:
            attr_names = (
                "archived_memory_preview",
                "archived_memory_list_canvas",
                "archived_memory_list_scrollbar",
                "archived_memory_list_inner",
                "archived_memory_list_window",
                "archived_memory_load_more_button",
            )
        for attr_name in attr_names:
            widget = getattr(self, attr_name, None)
            if isinstance(widget, tk.Canvas):
                self._cancel_scrollbar_visibility_update(widget)
                self._cancel_canvas_layout_update(widget)
                self._cancel_mousewheel_flush_for_widget(widget)
                if getattr(self, "active_scrollable", None) is widget:
                    self.active_scrollable = None
            if hasattr(widget, "winfo_exists") and widget.winfo_exists():
                try:
                    widget.destroy()
                except tk.TclError:
                    pass
            if attr_name.endswith("load_more_button"):
                setattr(self, attr_name, None)
            elif hasattr(self, attr_name):
                delattr(self, attr_name)
        if hasattr(frame, "winfo_exists") and frame.winfo_exists():
            try:
                frame.destroy()
            except tk.TclError:
                pass
        self._reset_heavy_surface_window_state(view_name)
        self._debug_event("surface_teardown", view_name=view_name)

    def _teardown_inactive_heavy_views(self, *, active_view: str) -> None:
        for view_name in self.HEAVY_SURFACE_VIEWS:
            if view_name != active_view:
                self._teardown_heavy_surface(view_name)

    def _schedule_view_refresh(self, view_name: str) -> None:
        if self.deferred_view_refresh_job is not None:
            try:
                self.root.after_cancel(self.deferred_view_refresh_job)
            except tk.TclError:
                pass
            self.deferred_view_refresh_job = None
        generation = int(getattr(self, "hotbar_navigation_generation", 0) or 0)
        decision = resolve_view_refresh_decision(
            view_name=view_name,
            view_enabled=self._view_capability_available(view_name),
            hotbar_animation_active=bool(self.hotbar_animation_job is not None),
            hotbar_open=bool(self.hotbar_open),
        )
        self._deferred_refresh_from_hotbar_close = False
        if decision.should_clear:
            if not self._view_capability_available(view_name) and view_name != "chat":
                self._debug_event("surface_capability_gated", view_name=view_name, phase=self._desktop_capability_state.phase)
            self._set_deferred_view_refresh_target(None)
            self._set_pending_hotbar_refresh_target(None)
            return
        self._set_deferred_view_refresh_target(view_name, generation=generation if generation else None)
        if decision.should_hold_for_hotbar:
            self._set_pending_hotbar_refresh_target(view_name, generation=generation if generation else None)
            return
        if decision.should_queue:
            self._queue_deferred_view_refresh()

    def _run_deferred_view_refresh(self) -> None:
        def _apply() -> None:
            target = self.deferred_view_refresh_target
            generation = int(getattr(self, "deferred_view_refresh_generation", 0) or 0)
            self.deferred_view_refresh_job = None
            self._set_deferred_view_refresh_target(None)
            self._deferred_refresh_from_hotbar_close = False
            self._debug_event(
                "run_deferred_view_refresh",
                target=target,
                current_view=getattr(self, "current_view", "chat"),
                generation=generation,
            )
            if generation and generation != int(getattr(self, "hotbar_navigation_generation", 0) or 0):
                self._debug_event("hotbar_navigation_discarded", target=target, generation=generation)
                return
            if not target or target != self.current_view:
                return
            if not self._view_capability_available(target):
                self._debug_event("surface_capability_gated", view_name=target, phase=self._desktop_capability_state.phase)
                return
            refreshed = False
            if target == "memory" and self.memory_view_dirty:
                self._refresh_memory_view()
                refreshed = True
            elif target == "recent" and self.recent_sessions_view_dirty:
                was_dirty = bool(self.recent_sessions_view_dirty)
                recent_result = self._refresh_recent_sessions_view()
                refreshed = bool(recent_result)
                if recent_result is None and was_dirty and not self.recent_sessions_view_dirty:
                    refreshed = True
            elif target == "archived" and self.archived_sessions_view_dirty:
                self._refresh_archived_sessions_view()
                refreshed = True
            elif target == "archived_memory" and self.archived_memory_view_dirty:
                prior_token = int(getattr(self, "_async_task_tokens", {}).get("archived_memory", 0))
                self._refresh_archived_memory_view()
                refreshed = int(getattr(self, "_async_task_tokens", {}).get("archived_memory", 0)) != prior_token
            if refreshed:
                self._apply_palette_to_view(target)

        label = "deferred_view_refresh_after_hotbar" if bool(getattr(self, "_deferred_refresh_from_hotbar_close", False)) else "deferred_view_refresh"
        self._timed_ui_call(label, _apply)

    def _transition_state(self) -> ShellTransitionState:
        state = getattr(self, "_shell_transition_state", None)
        if isinstance(state, ShellTransitionState):
            return state
        state = ShellTransitionState(
            pending_hotbar_open_state=getattr(self, "pending_hotbar_open_state", None),
            pending_hotbar_refresh_target=getattr(self, "pending_hotbar_refresh_target", None),
            deferred_view_refresh_target=getattr(self, "deferred_view_refresh_target", None),
            pending_view_name=getattr(self, "pending_view_name", None),
            hotbar_navigation_generation=int(getattr(self, "hotbar_navigation_generation", 0) or 0),
            pending_view_generation=int(getattr(self, "pending_view_generation", 0) or 0),
            pending_refresh_generation=int(getattr(self, "pending_refresh_generation", 0) or 0),
            theme_apply_requested=bool(getattr(self, "_theme_apply_requested", False)),
            theme_apply_in_progress=bool(getattr(self, "_theme_apply_in_progress", False)),
            debug_session=DebugTraceSession(
                log_path=getattr(self, "debug_ui_log_path", None),
            ),
        )
        self._shell_transition_state = state
        self._sync_transition_state_to_attrs(state)
        return state

    def _sync_transition_state_to_attrs(self, state: ShellTransitionState | None = None) -> None:
        state = state or self._transition_state()
        self.pending_hotbar_open_state = state.pending_hotbar_open_state
        self.pending_hotbar_refresh_target = state.pending_hotbar_refresh_target
        self.deferred_view_refresh_target = state.deferred_view_refresh_target
        self.pending_view_name = state.pending_view_name
        self.hotbar_navigation_generation = state.hotbar_navigation_generation
        self.pending_view_generation = state.pending_view_generation
        self.pending_refresh_generation = state.pending_refresh_generation
        self._theme_apply_requested = state.theme_apply_requested
        self._theme_apply_in_progress = state.theme_apply_in_progress

    def _set_pending_hotbar_open_state(self, target_state: bool | None) -> None:
        state = self._transition_state()
        state.set_pending_hotbar_open_state(target_state)
        self._sync_transition_state_to_attrs(state)

    def _set_pending_hotbar_refresh_target(self, view_name: str | None, *, generation: int | None = None) -> None:
        state = self._transition_state()
        state.set_pending_hotbar_refresh_target(view_name, generation=generation)
        self._sync_transition_state_to_attrs(state)

    def _set_deferred_view_refresh_target(self, view_name: str | None, *, generation: int | None = None) -> None:
        state = self._transition_state()
        state.set_deferred_view_refresh_target(view_name, generation=generation)
        self._sync_transition_state_to_attrs(state)
        self.deferred_view_refresh_generation = self.pending_refresh_generation if view_name else 0

    def _set_pending_view_name(self, view_name: str | None, *, generation: int | None = None) -> None:
        state = self._transition_state()
        state.set_pending_view_name(view_name, generation=generation)
        self._sync_transition_state_to_attrs(state)

    def _begin_hotbar_navigation(self, view_name: str) -> int:
        state = self._transition_state()
        generation = state.begin_hotbar_navigation(view_name)
        self._sync_transition_state_to_attrs(state)
        return generation

    def _take_pending_view_name(self) -> tuple[str | None, int]:
        state = self._transition_state()
        pending_view = state.consume_pending_view_name()
        self._sync_transition_state_to_attrs(state)
        return pending_view

    def _coordinator_take_refresh_target(self) -> tuple[str | None, int]:
        state = self._transition_state()
        refresh_target = state.consume_refresh_target()
        self._sync_transition_state_to_attrs(state)
        return refresh_target

    def _set_theme_apply_requested(self, requested: bool) -> None:
        state = self._transition_state()
        state.set_theme_apply_requested(requested)
        self._sync_transition_state_to_attrs(state)

    def _set_theme_apply_in_progress(self, in_progress: bool) -> None:
        state = self._transition_state()
        state.set_theme_apply_in_progress(in_progress)
        self._sync_transition_state_to_attrs(state)

    def _begin_mode_apply(self) -> bool:
        state = self._transition_state()
        if not state.begin_mode_apply():
            return False
        self._sync_transition_state_to_attrs(state)
        return True

    def _finish_mode_apply(self) -> None:
        state = self._transition_state()
        state.finish_mode_apply()
        self._sync_transition_state_to_attrs(state)

    def _build_starter_prompt_chips(self) -> None:
        for child in self.starter_chips_frame.winfo_children():
            child.destroy()
        self.starter_prompt_buttons = []
        visible_prompts = list(self.STARTER_PROMPT_OPTIONS[:5])
        for index, prompt in enumerate(visible_prompts):
            button = tk.Button(
                self.starter_chips_frame,
                text=prompt,
                command=lambda value=prompt: self._activate_starter_prompt(value),
                relief=tk.FLAT,
                bd=0,
                padx=12,
                pady=7,
                cursor="hand2",
                highlightthickness=1,
                wraplength=220,
                justify=tk.LEFT,
                anchor="w",
            )
            row = index // 3
            column = index % 3
            button.grid(row=row, column=column, sticky="w", padx=(0, 8), pady=(0, 8))
            button.bind("<Enter>", lambda event, widget=button: self._style_starter_prompt_chip(widget, hovered=True))
            button.bind("<Leave>", lambda event, widget=button: self._style_starter_prompt_chip(widget))
            self.starter_prompt_buttons.append(button)
            self._style_starter_prompt_chip(button)

    def _activate_starter_prompt(self, prompt: str) -> None:
        self.starter_prompt_var.set(prompt)
        for button in self.starter_prompt_buttons:
            self._style_starter_prompt_chip(button)
        self._use_starter_prompt()

    def _style_starter_prompt_chip(self, button: tk.Button, *, hovered: bool = False) -> None:
        prompt = str(button.cget("text") or "").strip()
        selected = prompt == str(self.starter_prompt_var.get() or "").strip()
        bg = self.current_palette["chip_active_bg"] if selected else (
            self.current_palette["chip_hover_bg"] if hovered else self.current_palette["chip_bg"]
        )
        button.configure(
            bg=bg,
            fg=self.current_palette["text_primary"],
            activebackground=self.current_palette["chip_hover_bg"],
            activeforeground=self.current_palette["text_primary"],
            highlightbackground=self.current_palette["chip_border"],
            highlightcolor=self.current_palette["chip_border"],
            disabledforeground=self.current_palette["text_secondary"],
            font=(self._resolved_font_family(), 9),
        )

    def _style_nav_buttons(self) -> None:
        nav_palette = dict(self.current_palette)
        if self._use_accented_dark_family_hover():
            nav_palette["nav_hover_bg"] = self.current_palette["nav_active_bg"]
        for name, button in self.nav_buttons.items():
            self._apply_nav_button_style(name, nav_palette=nav_palette)

    def _apply_nav_button_style(self, name: str, *, nav_palette: dict[str, str] | None = None) -> None:
        if name not in self.nav_buttons:
            return
        nav_palette = dict(nav_palette or self.current_palette)
        button = self.nav_buttons[name]
        enabled = self._view_capability_available(name) and str(button.cget("state")) != str(tk.DISABLED)
        visual = resolve_nav_button_visual(
            name=name,
            current_view=self.current_view,
            hovered_nav=self.hovered_nav,
            enabled=enabled,
            palette=nav_palette,
            use_accented_hover=self._use_accented_dark_family_hover(),
        )
        self.nav_button_frames[name].configure(
            bg=str(visual["bg"]),
            highlightbackground=str(visual["highlightbackground"]),
            highlightcolor=str(visual["highlightbackground"]),
        )
        self.nav_accents[name].configure(
            bg=str(visual["accent_bg"])
        )
        button.configure(
            bg=str(visual["bg"]),
            fg=str(visual["fg"]),
            activebackground=str(visual["activebackground"]),
            activeforeground=str(visual["activeforeground"]),
            disabledforeground=str(visual["disabledforeground"]),
            font=("Segoe UI Semibold", 10) if bool(visual["active"]) else ("Segoe UI", 10),
        )

    def _set_nav_hover(self, view_name: str, hovered: bool) -> None:
        if hovered and not self._view_capability_available(view_name):
            return
        previous = self.hovered_nav
        next_hover = view_name if hovered else (None if self.hovered_nav == view_name else self.hovered_nav)
        if previous == next_hover:
            return
        self.hovered_nav = next_hover
        nav_palette = dict(self.current_palette)
        if self._use_accented_dark_family_hover():
            nav_palette["nav_hover_bg"] = self.current_palette["nav_active_bg"]
        touched = [name for name in {previous, next_hover, self.current_view} if name]
        for name in touched:
            self._apply_nav_button_style(name, nav_palette=nav_palette)

    def _style_panel_view(self, frame: tk.Frame) -> None:
        frame.configure(
            bg=self.current_palette["panel_bg"],
            highlightbackground=self.current_palette["panel_bg"],
            highlightcolor=self.current_palette["panel_bg"],
        )
        for child in frame.winfo_children():
            if isinstance(child, tk.Label):
                child.configure(bg=self.current_palette["panel_bg"], fg=self.current_palette["text_primary"])
            elif isinstance(child, tk.Frame):
                child.configure(bg=self.current_palette["panel_bg"])
            elif isinstance(child, tk.Text):
                child.configure(
                    bg=self.current_palette["panel_bg"],
                    fg=self.current_palette["text_primary"],
                    insertbackground=self.current_palette["text_primary"],
                    selectbackground=self.current_palette["user_bg"],
                    selectforeground=self.current_palette["text_primary"],
                    font=self._message_font(),
                )

    def _restyle_surface_tree(self, widget: tk.Widget, *, bg: str) -> None:
        if not widget.winfo_exists():
            return
        if isinstance(widget, tk.Frame):
            try:
                widget.configure(bg=bg)
            except tk.TclError:
                pass
        elif isinstance(widget, tk.Canvas):
            try:
                widget.configure(bg=bg, highlightbackground=bg, highlightcolor=bg)
            except tk.TclError:
                pass
        elif isinstance(widget, tk.Label):
            try:
                widget.configure(bg=bg)
            except tk.TclError:
                pass
        elif isinstance(widget, tk.Text):
            try:
                widget.configure(
                    bg=bg,
                    insertbackground=self.current_palette["text_primary"],
                    selectbackground=self.current_palette["user_bg"],
                    selectforeground=self.current_palette["text_primary"],
                )
            except tk.TclError:
                pass
        for child in widget.winfo_children():
            self._restyle_surface_tree(child, bg=bg)

    def _use_accented_dark_family_hover(self) -> bool:
        theme_name = str(self.theme_var.get() or "Dark").strip().lower()
        return theme_name in {"dark", "custom"}

    def _browser_row_hover_bg(self) -> str:
        if self._use_accented_dark_family_hover():
            return self.current_palette["nav_active_bg"]
        return self.current_palette["list_hover_bg"]

    @staticmethod
    def _browser_surface_name(widget: tk.Widget | None) -> str | None:
        if widget is None:
            return None
        return str(getattr(widget, "_browser_surface", "") or "").strip() or None

    def _set_active_browser_hover_row(self, surface: str, row: tk.Widget | None) -> None:
        hover_rows = getattr(self, "_active_browser_hover_rows", None)
        if not isinstance(hover_rows, dict):
            hover_rows = {}
            self._active_browser_hover_rows = hover_rows
        current = hover_rows.get(surface)
        if current is row:
            return
        if current is not None and getattr(current, "winfo_exists", lambda: False)():
            current._browser_row_hovered = False  # type: ignore[attr-defined]
            self._style_browser_row(current, hovered=False)
        hover_rows[surface] = None
        if row is None or not row.winfo_exists():
            self._debug_event("browser_hover_cleared", surface=surface)
            return
        row._browser_row_hovered = True  # type: ignore[attr-defined]
        hover_rows[surface] = row
        self._style_browser_row(row, hovered=True)
        self._debug_event("browser_hover_set", surface=surface)

    def _clear_stale_browser_hover(self, surface: str) -> None:
        hover_rows = getattr(self, "_active_browser_hover_rows", None)
        if not isinstance(hover_rows, dict):
            return
        current = hover_rows.get(surface)
        if current is None:
            return
        if self._browser_row_contains_pointer(current):
            return
        self._debug_event("browser_hover_stale_cleared", surface=surface)
        self._set_active_browser_hover_row(surface, None)

    def _style_browser_row(self, row: tk.Widget, *, hovered: bool) -> None:
        meta = getattr(row, "_browser_row_meta", None)
        if not meta:
            return
        base_bg = self.current_palette["panel_bg"]
        hover_bg = self._browser_row_hover_bg()
        bg = hover_bg if hovered else base_bg
        try:
            row.configure(bg=base_bg)
        except tk.TclError:
            pass
        divider = meta.get("divider")
        if divider is not None and divider.winfo_exists():
            divider.configure(bg=self.current_palette["panel_divider"])
        card = meta.get("card")
        if card is not None and card.winfo_exists():
            card.configure(bg=bg)
        title_label = meta.get("title")
        if title_label is not None and title_label.winfo_exists():
            title_label.configure(bg=bg, fg=self.current_palette["text_primary"])
        detail_label = meta.get("detail")
        if detail_label is not None and detail_label.winfo_exists():
            detail_label.configure(bg=bg, fg=self.current_palette["text_secondary"])
        secondary_label = meta.get("secondary")
        if secondary_label is not None and secondary_label.winfo_exists():
            secondary_label.configure(bg=bg, fg=self.current_palette["text_muted"])

    def _restyle_browser_rows(self, widget: tk.Widget) -> None:
        if not widget.winfo_exists():
            return
        if getattr(widget, "_browser_row_meta", None):
            self._style_browser_row(widget, hovered=bool(getattr(widget, "_browser_row_hovered", False)))
        for child in widget.winfo_children():
            self._restyle_browser_rows(child)

    def _apply_palette_to_view(self, view_name: str) -> None:
        palette = self.current_palette
        if view_name == "chat":
            self.chat_surface.configure(
                bg=palette["app_bg"],
                highlightbackground=palette["app_bg"],
                highlightcolor=palette["app_bg"],
            )
            self.chat_canvas.configure(bg=palette["app_bg"])
            self.chat_frame.configure(bg=palette["app_bg"])
            self._restyle_surface_tree(self.landing_frame, bg=palette["app_bg"])
            if hasattr(self, "landing_orb_frame"):
                self.landing_orb_frame.configure(bg=palette["app_bg"])
            if hasattr(self, "landing_greeting_frame"):
                self.landing_greeting_frame.configure(bg=palette["app_bg"])
            self.identity_label.configure(bg=palette["app_bg"])
            self.greeting_label.configure(bg=palette["app_bg"], fg=palette["text_primary"])
            self.greeting_subtitle.configure(bg=palette["app_bg"], fg=palette["text_secondary"])
            self.starter_frame.configure(bg=palette["app_bg"])
            self.starter_dropdown_frame.configure(bg=palette["app_bg"])
            return

        frame = self.views.get(view_name)
        if frame is None or not frame.winfo_exists():
            return
        if view_name == "settings":
            self._style_panel_view(frame)
            if hasattr(self, "settings_scroll_host"):
                self.settings_scroll_host.configure(bg=palette["app_bg"])
            if hasattr(self, "settings_canvas"):
                self.settings_canvas.configure(bg=palette["app_bg"])
            if hasattr(self, "settings_scroll_inner"):
                self.settings_scroll_inner.configure(bg=palette["app_bg"])
            for section in getattr(self, "settings_row_sections", []):
                if section.winfo_exists():
                    section.configure(bg=palette["app_bg"])
                    for child in section.winfo_children():
                        if isinstance(child, tk.Frame) and str(child.cget("height") or "") == "1":
                            child.configure(bg=palette["panel_divider"])
                        else:
                            self._style_settings_control_surface(child, bg=palette["app_bg"], hovered=False)
            if hasattr(self, "help_text"):
                self.help_text.configure(
                    bg=palette["app_bg"],
                    fg=palette["text_primary"],
                    insertbackground=palette["text_primary"],
                    selectbackground=palette["user_bg"],
                    selectforeground=palette["text_primary"],
                )
            self._style_settings_reset_button(
                hovered=bool(getattr(getattr(self, "settings_reset_button", None), "_lumen_hovered", False))
            )
            return

        self._style_panel_view(frame)
        self._restyle_surface_tree(frame, bg=palette["panel_bg"])
        if view_name == "memory":
            self.memory_list_canvas.configure(bg=palette["panel_bg"])
            self.memory_list_inner.configure(bg=palette["panel_bg"])
            self.memory_preview.configure(
                bg=palette["panel_bg"],
                fg=palette["text_primary"],
                insertbackground=palette["text_primary"],
                selectbackground=palette["user_bg"],
                selectforeground=palette["text_primary"],
            )
        elif view_name == "archived_memory":
            self.archived_memory_list_canvas.configure(bg=palette["panel_bg"])
            self.archived_memory_list_inner.configure(bg=palette["panel_bg"])
            self.archived_memory_preview.configure(
                bg=palette["panel_bg"],
                fg=palette["text_primary"],
                insertbackground=palette["text_primary"],
                selectbackground=palette["user_bg"],
                selectforeground=palette["text_primary"],
            )
        elif view_name == "recent":
            self.recent_list_canvas.configure(bg=palette["panel_bg"])
            self.recent_list_inner.configure(bg=palette["panel_bg"])
        elif view_name == "archived":
            self.archived_list_canvas.configure(bg=palette["panel_bg"])
            self.archived_list_inner.configure(bg=palette["panel_bg"])
        if view_name in {"memory", "archived_memory", "recent", "archived"}:
            button_attr = {
                "memory": "memory_load_more_button",
                "archived_memory": "archived_memory_load_more_button",
                "recent": "recent_sessions_load_more_button",
                "archived": "archived_sessions_load_more_button",
            }[view_name]
            button = getattr(self, button_attr, None)
            if button is not None and button.winfo_exists():
                button.configure(**resolve_load_more_palette(palette=palette))
        self._restyle_browser_rows(frame)

    def _build_browser_row(
        self,
        parent: tk.Widget,
        *,
        title: str,
        detail: str,
        secondary: str = "",
        command,
        context_actions: list[tuple[str, object]] | None = None,
        surface: str | None = None,
        descriptor_identity: tuple[object, ...] | None = None,
        command_target_id: str | None = None,
        render_state: str = "rebuilt",
    ) -> tk.Frame:
        base_bg = self.current_palette["panel_bg"]
        row = tk.Frame(parent, bg=base_bg, padx=4, pady=2)
        row.pack(fill=tk.X, pady=(0, 2))
        row.grid_columnconfigure(0, weight=1)

        card = tk.Frame(row, bg=base_bg, cursor="hand2")
        card.grid(row=0, column=0, sticky="ew")
        card.grid_columnconfigure(0, weight=1)

        title_label = tk.Label(
            card,
            text=title,
            anchor="w",
            justify=tk.LEFT,
            font=("Segoe UI Semibold", 11),
            bg=base_bg,
            fg=self.current_palette["text_primary"],
            padx=8,
            pady=4,
        )
        title_label.grid(row=0, column=0, sticky="w")
        detail_label = tk.Label(
            card,
            text=detail,
            anchor="e",
            justify=tk.RIGHT,
            font=("Segoe UI", 9),
            bg=base_bg,
            fg=self.current_palette["text_secondary"],
            padx=8,
            pady=4,
        )
        detail_label.grid(row=0, column=1, sticky="e")
        if secondary:
            secondary_label = tk.Label(
                card,
                text=secondary,
                anchor="w",
                justify=tk.LEFT,
                font=("Segoe UI", 9),
                bg=base_bg,
                fg=self.current_palette["text_muted"],
                padx=8,
                pady=0,
            )
            secondary_label.grid(row=1, column=0, columnspan=2, sticky="ew")
            widgets = (card, title_label, detail_label, secondary_label)
        else:
            widgets = (card, title_label, detail_label)

        divider = tk.Frame(row, height=1, bg=self.current_palette["panel_divider"])
        divider.grid(row=1, column=0, sticky="ew", padx=8, pady=(4, 8))
        row._browser_row_meta = {  # type: ignore[attr-defined]
            "card": card,
            "title": title_label,
            "detail": detail_label,
            "secondary": secondary_label if secondary else None,
            "divider": divider,
        }
        row._browser_row_hovered = False  # type: ignore[attr-defined]
        row._browser_descriptor = descriptor_identity or ("session", title, detail, secondary)  # type: ignore[attr-defined]
        row._browser_command = command  # type: ignore[attr-defined]
        row._browser_context_actions = context_actions or []  # type: ignore[attr-defined]
        row._browser_surface = surface or self._browser_surface_name(parent) or "browser"  # type: ignore[attr-defined]
        row._browser_command_target_id = command_target_id or ""  # type: ignore[attr-defined]
        row._browser_render_state = render_state  # type: ignore[attr-defined]

        def _set_hover(active: bool) -> None:
            surface_name = self._browser_surface_name(row) or "browser"
            if active:
                self._set_active_browser_hover_row(surface_name, row)
                return
            if self._active_browser_hover_rows.get(surface_name) is row:
                self._set_active_browser_hover_row(surface_name, None)

        def _clear_hover_if_pointer_left() -> None:
            if self._browser_row_contains_pointer(row):
                return
            _set_hover(False)

        for widget in widgets:
            widget.bind("<Enter>", lambda event: _set_hover(True))
            widget.bind("<Leave>", lambda event: _clear_hover_if_pointer_left())
            widget.bind("<Button-1>", lambda event, target=row: self._invoke_browser_row_command(target))
            if context_actions:
                widget.bind(
                    "<Button-3>",
                    lambda event, target=row: self._show_browser_row_context_menu(target, event),
                )
        self._style_browser_row(row, hovered=False)
        return row

    def _update_browser_row(
        self,
        row: tk.Frame,
        *,
        title: str,
        detail: str,
        secondary: str = "",
        command,
        context_actions: list[tuple[str, object]] | None = None,
        surface: str | None = None,
        descriptor_identity: tuple[object, ...] | None = None,
        command_target_id: str | None = None,
        render_state: str = "reused",
    ) -> None:
        meta = getattr(row, "_browser_row_meta", {})
        title_label = meta.get("title")
        detail_label = meta.get("detail")
        secondary_label = meta.get("secondary")
        if title_label is not None and title_label.winfo_exists():
            title_label.configure(text=title)
        if detail_label is not None and detail_label.winfo_exists():
            detail_label.configure(text=detail)
        if secondary_label is not None and secondary_label.winfo_exists():
            secondary_label.configure(text=secondary)
            if secondary:
                secondary_label.grid()
            else:
                secondary_label.grid_remove()
        row._browser_descriptor = descriptor_identity or ("session", title, detail, secondary)  # type: ignore[attr-defined]
        row._browser_command = command  # type: ignore[attr-defined]
        row._browser_context_actions = context_actions or []  # type: ignore[attr-defined]
        row._browser_surface = surface or getattr(row, "_browser_surface", "browser")  # type: ignore[attr-defined]
        row._browser_command_target_id = command_target_id or ""  # type: ignore[attr-defined]
        row._browser_render_state = render_state  # type: ignore[attr-defined]
        self._style_browser_row(row, hovered=bool(getattr(row, "_browser_row_hovered", False)))

    def _invoke_browser_row_command(self, row: tk.Widget) -> None:
        command = getattr(row, "_browser_command", None)
        if not callable(command):
            return
        descriptor = getattr(row, "_browser_descriptor", ())
        surface = str(getattr(row, "_browser_surface", "") or "")
        callback_target_session_id = str(getattr(row, "_browser_command_target_id", "") or "")
        if (
            surface in {"recent", "archived"}
            and isinstance(descriptor, tuple)
            and descriptor
            and descriptor[0] == "session"
        ):
            rendered_session_id = str(descriptor[1] if len(descriptor) > 1 else "" or "")
            self._debug_event(
                "recent_row_open",
                surface=surface,
                clicked_session_id=rendered_session_id,
                descriptor_identity=repr(descriptor),
                row_render_state=str(getattr(row, "_browser_render_state", "") or "unknown"),
                callback_target_session_id=callback_target_session_id or rendered_session_id,
            )
        command()

    def _show_browser_row_context_menu(self, row: tk.Widget, event: tk.Event) -> None:
        actions = getattr(row, "_browser_context_actions", None)
        if actions:
            self._show_context_menu(event, actions)

    def _browser_row_contains_pointer(self, row: tk.Widget) -> bool:
        try:
            if not row.winfo_exists():
                return False
            pointer_x, pointer_y = row.winfo_pointerxy()
            hovered_widget = row.winfo_containing(pointer_x, pointer_y)
        except tk.TclError:
            return False
        current: object | None = hovered_widget
        while current is not None:
            if current is row:
                return True
            current = getattr(current, "master", None)
        return False

    def _build_browser_header(self, parent: tk.Widget, label: str, *, top_padding: int = 14) -> tk.Label:
        header = tk.Label(
            parent,
            text=label,
            anchor="w",
            font=("Segoe UI Semibold", 10),
            bg=self.current_palette["panel_bg"],
            fg=self.current_palette["text_muted"],
        )
        header.pack(fill=tk.X, pady=(top_padding if parent.winfo_children() else 0, 8))
        header._browser_descriptor = ("header", label, top_padding)  # type: ignore[attr-defined]
        return header

    def _update_browser_header(self, header: tk.Label, label: str, *, first: bool, top_padding: int = 14) -> None:
        header.configure(
            text=label,
            bg=self.current_palette["panel_bg"],
            fg=self.current_palette["text_muted"],
        )
        header.pack_configure(pady=(top_padding if not first else 0, 8))
        header._browser_descriptor = ("header", label, top_padding)  # type: ignore[attr-defined]

    def _show_context_menu(self, event: tk.Event, actions: list[tuple[str, object]]) -> None:
        menu = tk.Menu(self.root, tearoff=0)
        for label, callback in actions:
            menu.add_command(label=label, command=callback)
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    @staticmethod
    def _resolve_identity_icon_path(repo_root: Path) -> Path:
        bundled_root = getattr(sys, "_MEIPASS", "")
        bundled_path = Path(str(bundled_root)) / "assets" / "lumenicon.png" if bundled_root else None
        if bundled_path is not None and bundled_path.exists():
            return bundled_path
        return repo_root / "New UI" / "instructions and pics" / "refrence pics" / "Pic" / "lumenicon.png"

    def _build_identity_fallback_badge(self, *, size: int | None = None) -> tk.PhotoImage:
        if size is None:
            size = self._current_identity_icon_size()
        image = tk.PhotoImage(master=self.root, width=size, height=size)
        background = self.current_palette["app_bg"]
        accent = self.current_palette["nav_active_border"]
        inner = self.current_palette["nav_active_bg"]
        for x in range(size):
            for y in range(size):
                dx = x - size / 2
                dy = y - size / 2
                distance = (dx * dx + dy * dy) ** 0.5
                if distance <= size * 0.48:
                    color = accent if distance >= size * 0.36 else inner
                else:
                    color = background
                image.put(color, (x, y))
        return image

    def _current_identity_icon_size(self) -> int:
        available_width = 0
        if hasattr(self, "chat_canvas") and self.chat_canvas.winfo_exists():
            available_width = int(self.chat_canvas.winfo_width() or 0)
        if available_width <= 1 and hasattr(self, "chat_surface") and self.chat_surface.winfo_exists():
            available_width = int(self.chat_surface.winfo_width() or 0)
        if available_width <= 1 and self.root.winfo_exists():
            available_width = int(self.root.winfo_width() or 0)
        if available_width <= 1:
            available_width = self.IDENTITY_ICON_MIN_SIZE * 10
        proposed = int(available_width * self.IDENTITY_ICON_VIEWPORT_RATIO)
        return max(self.IDENTITY_ICON_MIN_SIZE, min(self.IDENTITY_ICON_MAX_SIZE, proposed))

    def _refresh_landing_icon_geometry(self, width: int | None = None) -> None:
        if not hasattr(self, "landing_orb_frame"):
            return
        icon_size = self._current_identity_icon_size() if width is None else max(
            self.IDENTITY_ICON_MIN_SIZE,
            min(self.IDENTITY_ICON_MAX_SIZE, int(width * self.IDENTITY_ICON_VIEWPORT_RATIO)),
        )
        frame_size = icon_size + 40
        self.landing_orb_frame.configure(width=frame_size, height=frame_size)

    def _load_identity_icon(self) -> None:
        def _apply() -> None:
            icon_size = self._current_identity_icon_size()
            self._refresh_landing_icon_geometry()
            signature = (str(self.identity_icon_path), icon_size)
            if self._identity_icon_signature == signature and self.identity_image is not None:
                return
            if not self.identity_icon_path.exists():
                self.identity_image = self._build_identity_fallback_badge(size=icon_size)
                self.identity_label.configure(image=self.identity_image, text="", compound="center")
                self._identity_icon_signature = signature
                return
            image = self._load_resized_image(self.identity_icon_path, max_size=icon_size)
            if image is None:
                self.identity_image = self._build_identity_fallback_badge(size=icon_size)
                self.identity_label.configure(image=self.identity_image, text="", compound="center")
                self._identity_icon_signature = signature
                return
            self.identity_image = image
            self.identity_label.configure(
                image=self.identity_image,
                text="",
                compound="center",
                font=("Segoe UI Semibold", 34),
            )
            self._identity_icon_signature = signature

        self._timed_ui_call("identity_icon_load", _apply)

    def _load_resized_image(self, path: Path, *, max_size: int) -> tk.PhotoImage | None:
        try:
            image = tk.PhotoImage(master=self.root, file=str(path))
        except tk.TclError:
            image = None
        if image is not None:
            scale = max(1, max(image.width(), image.height()) // max_size)
            if scale > 1:
                image = image.subsample(scale, scale)
            return image
        if Image is None or ImageTk is None:
            return None
        try:
            loaded = Image.open(path)
            loaded.thumbnail((max_size, max_size))
            return ImageTk.PhotoImage(loaded, master=self.root)
        except Exception:
            return None

    def _load_circular_avatar_image(self, path: Path, *, size: int) -> tk.PhotoImage | None:
        if Image is None or ImageTk is None:
            return self._load_resized_image(path, max_size=size)
        try:
            loaded = Image.open(path).convert("RGBA")
            width, height = loaded.size
            crop = min(width, height)
            left = (width - crop) // 2
            top = (height - crop) // 2
            loaded = loaded.crop((left, top, left + crop, top + crop))
            resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", getattr(Image, "LANCZOS", 1))
            loaded = loaded.resize((size, size), resample)
            mask = Image.new("L", (size, size), 0)
            from PIL import ImageDraw
            ImageDraw.Draw(mask).ellipse((0, 0, size - 1, size - 1), fill=255)
            loaded.putalpha(mask)
            return ImageTk.PhotoImage(loaded, master=self.root)
        except Exception:
            return self._load_resized_image(path, max_size=size)

    def _draw_profile_avatar(self) -> None:
        self.profile_avatar.delete("all")
        self.profile_avatar.create_oval(4, 4, 46, 46, fill=self.current_palette["nav_active_bg"], outline="")
        if self.profile_avatar_path is not None and self.profile_avatar_image is None:
            self.profile_avatar_image = self._load_circular_avatar_image(self.profile_avatar_path, size=42)
        if self.profile_avatar_image is not None:
            self.profile_avatar.create_image(25, 25, image=self.profile_avatar_image)
            return
        initials = (str(self.display_name_var.get() or self.DEFAULT_DISPLAY_NAME).strip()[:1] or "Y").upper()
        self.profile_avatar.create_text(
            25,
            25,
            text=initials,
            fill=self.current_palette["text_primary"],
            font=("Segoe UI Semibold", 16),
        )

    def _update_landing_state(self) -> None:
        if not hasattr(self, "landing_frame"):
            return
        visible_messages = [
            message for message in self.messages if message.message_type in {"user", "assistant"}
        ]
        show_landing = getattr(self, "current_view", "chat") == "chat" and not visible_messages
        if show_landing:
            self._load_identity_icon()
            self.greeting_label.configure(text=self._landing_greeting())
            self.greeting_subtitle.configure(text="What are we working through today?")
            self.landing_frame.place(relx=0.5, rely=0.48, anchor="center")
        else:
            self.landing_frame.place_forget()
        self._refresh_top_bar_title()

    def _landing_greeting(self) -> str:
        now = datetime.now()
        hour = now.hour
        if 5 <= hour < 12:
            period = "morning"
        elif 12 <= hour < 18:
            period = "afternoon"
        elif 18 <= hour < 20:
            period = "evening"
        else:
            period = "night"
        greeting_pools = {
            "Direct": {
                "morning": ("Good morning.", "Morning. Ready.", "Good morning. What's next?"),
                "afternoon": ("Good afternoon.", "Afternoon. Ready.", "Good afternoon. What's next?"),
                "evening": ("Good evening.", "Evening. Ready.", "Good evening. What's next?"),
                "night": ("Hello.", "Ready when you are.", "Here when you're ready."),
            },
            "Collab": {
                "morning": (
                    "Good morning. I'm here with you.",
                    "Morning. What are we picking up together?",
                    "Good morning. We can start wherever feels right.",
                ),
                "afternoon": (
                    "Good afternoon. Let’s explore it together.",
                    "Afternoon. What thread are we pulling on?",
                    "Good afternoon. I'm with you; where should we start?",
                    "Good afternoon. Feels good to have you here. What are we opening up?",
                ),
                "evening": (
                    "Good evening. Ready when you are.",
                    "Evening. What feels worth opening up?",
                    "Good evening. We can keep it easy or go deep.",
                    "Good evening. Glad to see you. What are we picking up?",
                ),
                "night": (
                    "Hello. I'm here with you.",
                    "Hey. We can keep this gentle and clear.",
                    "Here with you. What are we picking up?",
                    "Hey. Good to see you. Where do you want to start?",
                ),
            },
            "Default": {
                "morning": (
                    "Good morning. Lumen is ready.",
                    "Morning. Lumen is ready.",
                    "Good morning. What are we working through today?",
                    "Good morning. What should we open up first?",
                ),
                "afternoon": (
                    "Good afternoon. Lumen is ready.",
                    "Afternoon. Lumen is ready.",
                    "Good afternoon. What should we open up?",
                    "Good afternoon. Where do you want to begin?",
                ),
                "evening": (
                    "Good evening. Lumen is ready.",
                    "Evening. Lumen is ready.",
                    "Good evening. What are we working through?",
                    "Good evening. What would you like to dig into?",
                ),
                "night": (
                    "Hello. Lumen is ready.",
                    "Hello. Ready when you are.",
                    "Here and ready. What are we picking up?",
                    "Here when you're ready. What should we open first?",
                ),
            },
        }
        mode = self.mode_var.get() if self.mode_var.get() in greeting_pools else "Default"
        choices = greeting_pools[mode][period]
        seed_text = f"{mode}:{period}:{getattr(self, 'session_id', '')}:{now.toordinal()}:{now.hour // 2}"
        index = sum(ord(char) for char in seed_text) % len(choices)
        return choices[index]

    def _refresh_top_bar_title(self) -> None:
        if self.current_view != "chat":
            self.screen_title_var.set(
                {
                    "recent": "All Conversations",
                    "archived": "Archived Chats",
                    "memory": "Memory",
                    "archived_memory": "Archived Memory",
                    "settings": "Settings",
                }.get(self.current_view, "")
            )
            return
        current_title = str(self.chat_title_var.get() or "Chat").strip() or "Chat"
        self.screen_title_var.set("" if current_title == "Chat" else current_title)

    def _on_starter_visibility_changed(self) -> None:
        if self.show_starter_prompts_var.get():
            self.starter_frame.pack()
        else:
            self.starter_frame.pack_forget()

    def _on_density_changed(self, event: tk.Event | None = None) -> None:
        self.input_box.configure(height=self._input_box_height(), font=self._message_font())
        self._refresh_message_styles(reflow=True)
        self._persist_desktop_preferences_safe()

    def _refresh_memory_view(self) -> None:
        resolved_state = self._resolve_memory_surface_visible_state(archived=False)
        if resolved_state["state"] == "unavailable":
            self.memory_view_dirty = True
            self._debug_event("surface_capability_gated", view_name="memory", phase=self._desktop_capability_state.phase)
            if hasattr(self, "memory_list_inner") and hasattr(self, "memory_preview"):
                self._render_memory_surface_resolved_state(
                    archived=False,
                    render_mode="render_from_cached_slice",
                )
            return
        was_dirty = bool(self.memory_view_dirty)
        cache_rendered = False
        if hasattr(self, "memory_list_inner"):
            if resolved_state["state"] == "cache_rows":
                self._debug_event(
                    "surface_reenter_from_cache",
                    view="memory",
                    fetched_count=len(self.memory_entries),
                    render_limit=self.memory_render_limit,
                )
                cache_rendered = (
                    self._render_memory_surface_resolved_state(
                        archived=False,
                        render_mode="render_from_cached_slice",
                    )
                    == "cache_rows"
                )
            else:
                self._render_memory_surface_resolved_state(
                    archived=False,
                    render_mode="render_from_cached_slice",
                )
        self.memory_view_dirty = False
        target_fetch_limit = max(self.memory_render_limit, self.memory_render_step)
        if (
            cache_rendered
            and not was_dirty
            and (len(self.memory_entries) >= target_fetch_limit or not self.memory_entries_has_more)
        ):
            self._debug_event(
                "memory_load_more_suppressed",
                view="memory",
                reason="cache_current_reentry",
                target_fetch_limit=target_fetch_limit,
            )
            return
        if self.memory_fetch_in_flight and self.memory_requested_fetch_limit == target_fetch_limit:
            self._debug_event(
                "memory_load_more_suppressed",
                view="memory",
                reason="in_flight_same_target",
                target_fetch_limit=target_fetch_limit,
            )
            return
        self._start_memory_surface_fetch(
            archived=False,
            fetch_limit=target_fetch_limit,
            fetch_reason="extended" if self.memory_render_limit > self.memory_render_step else "bounded",
        )

    def _refresh_archived_memory_view(self) -> None:
        resolved_state = self._resolve_memory_surface_visible_state(archived=True)
        if resolved_state["state"] == "unavailable":
            self.archived_memory_view_dirty = True
            self._debug_event(
                "surface_capability_gated",
                view_name="archived_memory",
                phase=self._desktop_capability_state.phase,
            )
            if hasattr(self, "archived_memory_list_inner") and hasattr(self, "archived_memory_preview"):
                self._render_memory_surface_resolved_state(
                    archived=True,
                    render_mode="render_from_cached_slice",
                )
            return
        state_version = int(getattr(self, "archived_memory_state_version", 0))
        loaded_version = int(getattr(self, "archived_memory_loaded_version", -1))
        requested_version = getattr(self, "archived_memory_requested_version", None)
        cached_signature = tuple(getattr(self, "archived_memory_cached_signature", self.archived_memory_render_signature))
        painted_signature = tuple(getattr(self, "archived_memory_render_signature", ()))
        has_archived_surface = hasattr(self, "archived_memory_list_inner")
        cache_rendered = False
        self.archived_memory_view_dirty = False
        if resolved_state["state"] != "cache_rows" and has_archived_surface:
            self._render_memory_surface_resolved_state(
                archived=True,
                render_mode="render_from_cached_slice",
            )
        if should_render_archived_memory_cache(
            entries=self.archived_memory_entries,
            cached_signature=cached_signature,
            painted_signature=painted_signature,
            loaded_version=loaded_version,
            state_version=state_version,
            has_children=bool(self.archived_memory_list_inner.winfo_children()) if has_archived_surface else False,
        ):
            try:
                self._debug_event(
                    "surface_reenter_from_cache",
                    view="archived_memory",
                    fetched_count=len(self.archived_memory_entries),
                    render_limit=self.archived_memory_render_limit,
                )
                self._timed_ui_call(
                    "archived_memory_cache_render",
                    lambda: self._render_archived_memory_from_cache(render_mode="render_from_cached_slice"),
                )
                cache_rendered = True
            except Exception as exc:
                self.archived_memory_fetch_in_flight = False
                self.archived_memory_requested_version = None
                self.archived_memory_view_dirty = True
                context = self._memory_surface_context(archived=True, render_mode="cache-only")
                self._record_desktop_crash(
                    source="archived_memory.cache_render",
                    exc=exc,
                    details=self._describe_heavy_surface_context(context),
                    context=context,
                )
                self._surface_runtime_failure(
                    "Lumen couldn't render archived memory right now.",
                    source="archived_memory.cache_render",
                    category="render_failure",
                    context=context,
                )
                return
        target_fetch_limit = max(self.archived_memory_render_limit, self.archived_memory_render_step)
        if (
            cache_rendered
            and loaded_version == state_version
            and (len(self.archived_memory_entries) >= target_fetch_limit or not self.archived_memory_entries_has_more)
        ):
            self._debug_event(
                "memory_load_more_suppressed",
                view="archived_memory",
                reason="cache_current_reentry",
                target_fetch_limit=target_fetch_limit,
            )
            return
        if not should_fetch_archived_memory(
            current_view=str(getattr(self, "current_view", "") or ""),
            loaded_version=loaded_version,
            state_version=state_version,
            fetch_in_flight=bool(getattr(self, "archived_memory_fetch_in_flight", False)),
            requested_version=requested_version if isinstance(requested_version, int) else None,
        ):
            return
        self.archived_memory_fetch_in_flight = True
        self.archived_memory_requested_version = state_version
        self._start_memory_surface_fetch(
            archived=True,
            fetch_limit=target_fetch_limit,
            fetch_reason="extended" if self.archived_memory_render_limit > self.archived_memory_render_step else "bounded",
        )

    def _start_memory_surface_fetch(self, *, archived: bool, fetch_limit: int, fetch_reason: str) -> None:
        normalized_limit = max(int(fetch_limit or 0), 1)
        if archived:
            self.archived_memory_fetch_in_flight = True
            self.archived_memory_requested_version = int(getattr(self, "archived_memory_state_version", 0))
            self.archived_memory_requested_fetch_limit = normalized_limit
        else:
            self.memory_fetch_in_flight = True
            self.memory_requested_fetch_limit = normalized_limit
        self._debug_event(
            "memory_load_more_extended_fetch" if fetch_reason == "extended" else "payload_fetch_bounded",
            view="archived_memory" if archived else "memory",
            fetch_limit=normalized_limit,
            fetch_reason=fetch_reason,
        )
        self._start_ui_background_task(
            "archived_memory" if archived else "memory",
            lambda: self._build_memory_view_payload(
                archived_only=archived,
                fetch_limit=normalized_limit,
                fetch_reason=fetch_reason,
            ),
            timing_label="archived_memory_fetch" if archived else "memory_fetch",
        )

    def _build_memory_view_payload(
        self,
        *,
        archived_only: bool = False,
        fetch_limit: int | None = None,
        fetch_reason: str = "bounded",
    ) -> dict[str, object]:
        entries, has_more_available = self._collect_memory_entries(
            archived_only=archived_only,
            fetch_limit=fetch_limit,
        )
        self._debug_event(
            "payload_fetch_extended" if fetch_reason == "extended" else "payload_fetch_bounded",
            view="archived_memory" if archived_only else "memory",
            fetched_count=len(entries),
            grouped_sections=memory_group_count(entries),
            render_limit=max(int(fetch_limit or len(entries) or 0), 0),
            has_more=has_more_available,
        )
        return {
            "entries": entries,
            "signature": self._memory_entries_signature(entries),
            "archived_only": archived_only,
            "has_more_available": has_more_available,
            "fetch_limit": fetch_limit,
            "row_cache": build_memory_row_cache(entries),
        }

    def _collect_memory_entries(
        self,
        *,
        archived_only: bool,
        fetch_limit: int | None = None,
    ) -> tuple[list[dict[str, object]], bool]:
        normalized_limit = max(int(fetch_limit), 1) if fetch_limit is not None else None
        per_source_limit = normalized_limit + 1 if normalized_limit is not None else None
        entries: list[dict[str, object]] = []

        def _fetch_personal_memory() -> list[dict[str, object]]:
            if not hasattr(self.controller, "list_personal_memory"):
                return []
            report = self.controller.list_personal_memory(
                session_id=None,
                include_archived=archived_only,
                archived_only=archived_only,
                limit=per_source_limit,
            )
            entries = [dict(item) for item in report.get("personal_memory", []) if isinstance(item, dict)]
            self._debug_event(
                "memory_source_fetch",
                source="personal_memory",
                archived_only=archived_only,
                requested_limit=per_source_limit,
                returned_count=len(entries),
            )
            return entries

        def _fetch_research_notes() -> list[dict[str, object]]:
            if not hasattr(self.controller, "list_research_notes"):
                return []
            report = self.controller.list_research_notes(
                session_id=None,
                include_archived=archived_only,
                archived_only=archived_only,
                limit=per_source_limit,
            )
            notes: list[dict[str, object]] = []
            for item in report.get("research_notes", []):
                if isinstance(item, dict):
                    notes.append(
                        {
                            "title": item.get("title"),
                            "content": item.get("content", ""),
                            "created_at": item.get("created_at"),
                            "kind": "research_note",
                            "note_path": item.get("note_path"),
                            "memory_item_id": item.get("memory_item_id") or item.get("id"),
                            "id": item.get("id"),
                            "source_id": item.get("source_id"),
                        }
                    )
            self._debug_event(
                "memory_source_fetch",
                source="research_notes",
                archived_only=archived_only,
                requested_limit=per_source_limit,
                returned_count=len(notes),
            )
            return notes

        personal_started = perf_counter()
        personal_entries = _fetch_personal_memory()
        personal_elapsed_ms = round((perf_counter() - personal_started) * 1000, 2)
        self._debug_event(
            "memory_source_timing",
            source="personal_memory",
            archived_only=archived_only,
            requested_limit=per_source_limit,
            returned_count=len(personal_entries),
            elapsed_ms=personal_elapsed_ms,
        )
        entries.extend(personal_entries)

        notes_started = perf_counter()
        note_entries = _fetch_research_notes()
        notes_elapsed_ms = round((perf_counter() - notes_started) * 1000, 2)
        self._debug_event(
            "memory_source_timing",
            source="research_notes",
            archived_only=archived_only,
            requested_limit=per_source_limit,
            returned_count=len(note_entries),
            elapsed_ms=notes_elapsed_ms,
        )
        entries.extend(note_entries)
        sorted_entries = sorted(entries, key=lambda item: str(item.get("created_at") or ""), reverse=True)
        if normalized_limit is None:
            return sorted_entries, False
        has_more_available = len(sorted_entries) > normalized_limit
        return sorted_entries[:normalized_limit], has_more_available

    def _on_memory_topic_selected(self, event: tk.Event | None = None) -> None:
        selection = self.memory_topics_listbox.curselection()
        if not selection:
            return
        selected_index = selection[0]
        self.selected_memory_topic_index = selected_index
        knowledge = self.controller.knowledge_overview() if hasattr(self.controller, "knowledge_overview") else {}
        self._update_memory_detail(selected_index, knowledge=knowledge)

    def _update_memory_detail(self, index: int, *, knowledge: dict[str, object] | None = None) -> None:
        rows = getattr(self, "memory_topic_rows", [])
        if not (0 <= index < len(rows)):
            self._set_text_widget(self.memory_detail, "Select a topic to inspect the saved entries.")
            return
        row = rows[index]
        topic = str(row.get("topic") or "").strip()
        if topic == "__knowledge__":
            lines = ["What Lumen Knows", ""]
            lines.extend(f"- {line}" for line in knowledge_category_lines(knowledge or {}))
            if not (knowledge or {}).get("categories"):
                lines.append("No local knowledge entries found.")
            self._set_text_widget(self.memory_detail, "\n".join(lines).strip())
            return

        entries = row.get("entries") if isinstance(row.get("entries"), list) else []
        lines = [topic.replace("_", " ").title(), ""]
        if not entries:
            lines.append(empty_state_text("memory"))
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            title = str(entry.get("title") or topic).strip()
            created = human_date_label(entry.get("created_at"))
            kind = str(entry.get("kind") or "").replace("_", " ").title()
            content = str(entry.get("content") or "").strip()
            lines.append(f"- {title}")
            meta = " • ".join(item for item in (kind, created) if item)
            if meta:
                lines.append(f"  {meta}")
            for detail_line in content.splitlines():
                if detail_line.strip():
                    lines.append(f"  {detail_line}")
            lines.append("")
        self._set_text_widget(self.memory_detail, "\n".join(lines).strip())

    def _refresh_recent_sessions_view(self) -> bool:
        if not self._view_capability_available("recent"):
            self.recent_sessions_view_dirty = True
            self._debug_event("surface_capability_gated", view_name="recent", phase=self._desktop_capability_state.phase)
            return False
        cached_signature = tuple(getattr(self, "recent_sessions_signature", ()))
        painted_signature = tuple(getattr(self, "recent_sessions_render_signature", ()))
        has_recent_surface = hasattr(self, "recent_list_inner")
        cache_rendered = False
        self.recent_sessions_view_dirty = False
        restored = self._restore_session_cache_from_disk("recent")
        if (
            (restored or self.recent_sessions_rows or cached_signature)
            and (
                not has_recent_surface
                or not self.recent_list_inner.winfo_children()
                or cached_signature != painted_signature
            )
        ):
            if has_recent_surface:
                self._timed_ui_call("recent_cache_render", self._render_recent_sessions_from_cache)
                cache_rendered = True
        if (
            getattr(self, "current_view", "") == "recent"
            and cached_signature
            and painted_signature == cached_signature
            and (
                getattr(self, "recent_sessions_requested_signature", None) == cached_signature
                or not getattr(self, "recent_sessions_restored_from_disk", False)
            )
            and not getattr(self, "recent_sessions_fetch_in_flight", False)
        ):
            self._debug_event("recent_refresh_suppressed", reason="cache_current")
            return cache_rendered
        if (
            getattr(self, "recent_sessions_fetch_in_flight", False)
            and getattr(self, "recent_sessions_requested_signature", None) == cached_signature
        ):
            self._debug_event("recent_refresh_suppressed", reason="in_flight_same_signature")
            return cache_rendered
        self.recent_sessions_fetch_in_flight = True
        self.recent_sessions_restored_from_disk = False
        self.recent_sessions_requested_signature = cached_signature
        self._debug_event(
            "recent_refresh_scheduled",
            cached=bool(cached_signature),
            signature_size=len(cached_signature),
        )
        self._start_ui_background_task("recent", lambda: self._build_session_view_payload(archived=False), timing_label="recent_fetch")
        return True

    def _render_recent_sessions_from_cache(self) -> None:
        self._render_session_rows_from_cache(archived=False)

    def _apply_recent_sessions_result(self, result: dict[str, object]) -> None:
        def _apply() -> None:
            self.recent_sessions_fetch_in_flight = False
            if not bool(result.get("available", False)):
                if self.current_view != "recent":
                    self.recent_sessions_view_dirty = True
                    return
                for child in self.recent_list_inner.winfo_children():
                    child.destroy()
                tk.Label(
                    self.recent_list_inner,
                    text=empty_state_text("recent"),
                    anchor="w",
                    justify=tk.LEFT,
                    bg=self.current_palette["panel_bg"],
                    fg=self.current_palette["text_muted"],
                ).pack(fill=tk.X)
                return
            sessions = [
                dict(item)
                for item in result.get("sessions", [])
                if isinstance(item, dict) and is_user_visible_session(item)
            ]
            signature = tuple(result.get("signature", ()))
            if signature != self.recent_sessions_signature:
                self.recent_sessions_cache = sessions
                self.recent_sessions_rows = list(result.get("rows", []))
                self.recent_sessions_signature = signature
                self.recent_sessions_requested_signature = signature
                self.recent_sessions_restored_from_disk = False
                self._conversation_cache_dirty = True
                self._schedule_conversation_cache_write()
            else:
                self.recent_sessions_requested_signature = signature
            if self.current_view != "recent":
                self.recent_sessions_view_dirty = True
                return
            self._render_recent_sessions_from_cache()

        self._timed_ui_call("recent_apply", _apply)

    def _refresh_archived_sessions_view(self) -> None:
        if not self._view_capability_available("archived"):
            self.archived_sessions_view_dirty = True
            self._debug_event("surface_capability_gated", view_name="archived", phase=self._desktop_capability_state.phase)
            return
        cached_signature = tuple(getattr(self, "archived_sessions_signature", ()))
        painted_signature = tuple(getattr(self, "archived_sessions_render_signature", ()))
        has_archived_surface = hasattr(self, "archived_list_inner")
        cache_rendered = False
        self.archived_sessions_view_dirty = False
        if not has_archived_surface:
            return
        restored = self._restore_session_cache_from_disk("archived")
        if (
            (restored or self.archived_sessions_rows or cached_signature)
            and (
                not has_archived_surface
                or not self.archived_list_inner.winfo_children()
                or cached_signature != painted_signature
            )
        ):
            self._debug_event(
                "archived_surface_reentered_from_cache",
                cached=bool(cached_signature),
                signature_size=len(cached_signature),
            )
            self._timed_ui_call("archived_cache_render", self._render_archived_sessions_from_cache)
            cache_rendered = True
        if (
            getattr(self, "current_view", "") == "archived"
            and cached_signature
            and painted_signature == cached_signature
            and (
                getattr(self, "archived_sessions_requested_signature", None) == cached_signature
                or not getattr(self, "archived_sessions_restored_from_disk", False)
            )
            and not getattr(self, "archived_sessions_fetch_in_flight", False)
        ):
            self._debug_event("archived_fetch_suppressed", reason="cache_current")
            return
        if (
            getattr(self, "archived_sessions_fetch_in_flight", False)
            and getattr(self, "archived_sessions_requested_signature", None) == cached_signature
        ):
            self._debug_event("archived_fetch_suppressed", reason="in_flight_same_signature")
            return
        self.archived_sessions_fetch_in_flight = True
        self.archived_sessions_restored_from_disk = False
        self.archived_sessions_requested_signature = cached_signature
        self._debug_event(
            "archived_fetch_scheduled",
            cached=bool(cached_signature),
            signature_size=len(cached_signature),
            cache_rendered=cache_rendered,
        )
        self._start_ui_background_task(
            "archived",
            lambda: self._build_session_view_payload(archived=True),
            timing_label="archived_fetch",
        )

    def _build_session_view_payload(self, *, archived: bool) -> dict[str, object]:
        if not hasattr(self.controller, "list_recent_sessions"):
            return {"available": False, "archived": archived, "sessions": [], "signature": ()}
        if archived:
            report = self.controller.list_recent_sessions(limit=200, include_archived=True, archived_only=True)
        else:
            report = self.controller.list_recent_sessions(limit=self.RECENT_SESSIONS_FETCH_LIMIT)
        sessions = report.get("sessions", []) if isinstance(report, dict) else []
        cached_sessions = [
            dict(item)
            for item in sessions
            if isinstance(item, dict) and is_user_visible_session(item)
        ]
        grouped_rows = grouped_session_rows(cached_sessions)
        self._debug_event(
            "heavy_surface_payload",
            view="archived" if archived else "recent",
            fetched_count=len(cached_sessions),
            grouped_sections=sum(1 for row in grouped_rows if row.get("kind") == "header"),
        )
        return {
            "available": True,
            "archived": archived,
            "sessions": cached_sessions,
            "signature": self._session_signature(cached_sessions),
            "rows": grouped_rows,
        }

    def _render_archived_sessions_from_cache(self) -> None:
        if not hasattr(self, "archived_list_inner") or not self.archived_list_inner.winfo_exists():
            self._debug_event("archived_result_dropped_after_teardown", reason="surface_missing_render")
            self._append_runtime_failure_log(
                message="Archived chats render was skipped because the surface was already torn down.",
                source="archived.render",
                category="stale_result_drop",
                context={
                    "surface": "archived",
                    "reason": "surface_missing_render",
                    "current_view": str(getattr(self, "current_view", "") or ""),
                },
            )
            return
        self._render_session_rows_from_cache(archived=True)

    def _apply_archived_sessions_result(self, result: dict[str, object]) -> None:
        def _apply() -> None:
            self.archived_sessions_fetch_in_flight = False
            if not bool(result.get("available", False)):
                if self.current_view != "archived":
                    self.archived_sessions_view_dirty = True
                    self._append_runtime_failure_log(
                        message="Archived chats result arrived after the user had already left the surface.",
                        source="archived.apply",
                        category="stale_result_drop",
                        context={
                            "surface": "archived",
                            "reason": "view_switched",
                            "signature_size": len(signature),
                            "current_view": str(getattr(self, "current_view", "") or ""),
                        },
                    )
                    return
                if not hasattr(self, "archived_list_inner") or not self.archived_list_inner.winfo_exists():
                    self._debug_event("archived_result_dropped_after_teardown", reason="surface_missing", available=False)
                    self.archived_sessions_view_dirty = True
                    self._append_runtime_failure_log(
                        message="Archived chats surface was unavailable while applying an empty archived result.",
                        source="archived.apply",
                        category="stale_result_drop",
                        context={
                            "surface": "archived",
                            "reason": "surface_missing",
                            "available": False,
                            "current_view": str(getattr(self, "current_view", "") or ""),
                        },
                    )
                    return
                for child in self.archived_list_inner.winfo_children():
                    child.destroy()
                tk.Label(
                    self.archived_list_inner,
                    text="No archived chats yet.",
                    anchor="w",
                    justify=tk.LEFT,
                    bg=self.current_palette["panel_bg"],
                    fg=self.current_palette["text_muted"],
                ).pack(fill=tk.X)
                return
            sessions = [
                dict(item)
                for item in result.get("sessions", [])
                if isinstance(item, dict) and is_user_visible_session(item)
            ]
            signature = tuple(result.get("signature", ()))
            if signature != self.archived_sessions_signature:
                self.archived_sessions_cache = sessions
                self.archived_sessions_rows = list(result.get("rows", []))
                self.archived_sessions_signature = signature
                self.archived_sessions_requested_signature = signature
                self.archived_sessions_restored_from_disk = False
                self._conversation_cache_dirty = True
                self._schedule_conversation_cache_write()
            else:
                self.archived_sessions_requested_signature = signature
            if self.current_view != "archived":
                self._debug_event("archived_result_dropped_after_teardown", reason="view_switched", signature_size=len(signature))
                self.archived_sessions_view_dirty = True
                self._append_runtime_failure_log(
                    message="Archived chats result arrived after the user switched away.",
                    source="archived.apply",
                    category="stale_result_drop",
                    context={
                        "surface": "archived",
                        "reason": "view_switched",
                        "signature_size": len(signature),
                        "session_count": len(sessions),
                        "current_view": str(getattr(self, "current_view", "") or ""),
                    },
                )
                return
            if not hasattr(self, "archived_list_inner") or not self.archived_list_inner.winfo_exists():
                self._debug_event("archived_result_dropped_after_teardown", reason="surface_missing", signature_size=len(signature))
                self.archived_sessions_view_dirty = True
                self._append_runtime_failure_log(
                    message="Archived chats surface was unavailable while applying a refreshed archived result.",
                    source="archived.apply",
                    category="stale_result_drop",
                    context={
                        "surface": "archived",
                        "reason": "surface_missing",
                        "signature_size": len(signature),
                        "session_count": len(sessions),
                        "current_view": str(getattr(self, "current_view", "") or ""),
                    },
                )
                return
            self._debug_event(
                "archived_result_applied",
                signature_size=len(signature),
                session_count=len(sessions),
            )
            self._render_archived_sessions_from_cache()

        self._timed_ui_call("archived_apply", _apply)

    def _apply_memory_view_result(self, result: dict[str, object]) -> None:
        def _apply() -> None:
            try:
                entries = [dict(item) for item in result.get("entries", []) if isinstance(item, dict)]
                signature = tuple(result.get("signature", ()))
                fetch_reason = str(result.get("fetch_reason") or "bounded")
                requested_limit = int(result.get("fetch_limit") or len(entries) or 0)
                row_cache = result.get("row_cache") if isinstance(result.get("row_cache"), tuple) else None
                prior_signature = self.memory_cached_signature
                prior_has_more = bool(self.memory_entries_has_more)
                self.memory_fetch_in_flight = False
                self.memory_requested_fetch_limit = None
                self.memory_entries = entries
                self.memory_cached_signature = signature
                if isinstance(row_cache, tuple) and len(row_cache) == 4:
                    (
                        self.memory_row_descriptors,
                        self.memory_row_entry_map,
                        self.memory_row_descriptor_offsets,
                        self.memory_row_group_counts,
                    ) = row_cache
                else:
                    (
                        self.memory_row_descriptors,
                        self.memory_row_entry_map,
                        self.memory_row_descriptor_offsets,
                        self.memory_row_group_counts,
                    ) = build_memory_row_cache(entries)
                self.memory_entries_has_more = bool(result.get("has_more_available", False))
                if prior_signature != signature or prior_has_more != self.memory_entries_has_more:
                    self._mark_conversation_cache_dirty()
                if self.current_view != "memory":
                    self.memory_view_dirty = True
                    self._debug_event(
                        "continuation_result_dropped_after_leave",
                        view="memory",
                        fetch_reason=fetch_reason,
                        fetch_limit=requested_limit,
                        fetched_count=len(entries),
                    )
                    return
                self._render_memory_entries_from_cache(archived=False, render_mode="cache+fetch")
            except Exception as exc:
                self.memory_fetch_in_flight = False
                self.memory_requested_fetch_limit = None
                self.memory_view_dirty = True
                context = self._memory_surface_context(archived=False, render_mode="cache+fetch")
                self._record_desktop_crash(
                    source="memory.apply",
                    exc=exc,
                    details=self._describe_heavy_surface_context(context),
                    context=context,
                )
                self._surface_runtime_failure(
                    "Lumen couldn't apply the memory refresh.",
                    source="memory.apply",
                    category="apply_failure",
                    context=context,
                )

        self._timed_ui_call("memory_apply", _apply)

    def _apply_archived_memory_view_result(self, result: dict[str, object]) -> None:
        def _apply() -> None:
            try:
                entries = [dict(item) for item in result.get("entries", []) if isinstance(item, dict)]
                signature = tuple(result.get("signature", ()))
                requested_version = getattr(self, "archived_memory_requested_version", None)
                requested_limit = int(result.get("fetch_limit") or len(entries) or 0)
                fetch_reason = str(result.get("fetch_reason") or "bounded")
                row_cache = result.get("row_cache") if isinstance(result.get("row_cache"), tuple) else None
                prior_cached_signature = self.archived_memory_cached_signature
                prior_has_more = bool(self.archived_memory_entries_has_more)
                if requested_version is None:
                    requested_version = int(getattr(self, "archived_memory_state_version", 0))
                self.archived_memory_fetch_in_flight = False
                self.archived_memory_requested_version = None
                self.archived_memory_requested_fetch_limit = None
                self.archived_memory_entries = entries
                self.archived_memory_cached_signature = signature
                if isinstance(row_cache, tuple) and len(row_cache) == 4:
                    (
                        self.archived_memory_row_descriptors,
                        self.archived_memory_row_entry_map,
                        self.archived_memory_row_descriptor_offsets,
                        self.archived_memory_row_group_counts,
                    ) = row_cache
                else:
                    (
                        self.archived_memory_row_descriptors,
                        self.archived_memory_row_entry_map,
                        self.archived_memory_row_descriptor_offsets,
                        self.archived_memory_row_group_counts,
                    ) = build_memory_row_cache(entries)
                self.archived_memory_entries_has_more = bool(result.get("has_more_available", False))
                self.archived_memory_loaded_version = int(requested_version)
                if prior_cached_signature != signature or prior_has_more != self.archived_memory_entries_has_more:
                    self._mark_conversation_cache_dirty()
                if self.current_view != "archived_memory":
                    self.archived_memory_view_dirty = True
                    self._debug_event(
                        "continuation_result_dropped_after_leave",
                        view="archived_memory",
                        fetch_reason=fetch_reason,
                        fetch_limit=requested_limit,
                        fetched_count=len(entries),
                    )
                    return
                self._render_archived_memory_from_cache(render_mode="cache+fetch")
            except Exception as exc:
                self.archived_memory_fetch_in_flight = False
                self.archived_memory_requested_version = None
                self.archived_memory_requested_fetch_limit = None
                self.archived_memory_view_dirty = True
                context = self._memory_surface_context(archived=True, render_mode="cache+fetch")
                self._record_desktop_crash(
                    source="archived_memory.apply",
                    exc=exc,
                    details=self._describe_heavy_surface_context(context),
                    context=context,
                )
                self._surface_runtime_failure(
                    "Lumen couldn't apply the archived memory refresh.",
                    source="archived_memory.apply",
                    category="apply_failure",
                    context=context,
                )

        self._timed_ui_call("archived_memory_apply", _apply)

    def _render_archived_memory_from_cache(self, *, render_mode: str = "cache-only") -> None:
        try:
            self._render_memory_entries_from_cache(archived=True, render_mode=render_mode)
        except Exception as exc:
            self.archived_memory_fetch_in_flight = False
            self.archived_memory_requested_version = None
            self.archived_memory_requested_fetch_limit = None
            self.archived_memory_view_dirty = True
            context = self._memory_surface_context(archived=True, render_mode=render_mode)
            self._record_desktop_crash(
                source="archived_memory.render",
                exc=exc,
                details=self._describe_heavy_surface_context(context),
                context=context,
            )
            self._surface_runtime_failure(
                "Lumen couldn't render archived memory.",
                source="archived_memory.render",
                category="render_failure",
                context=context,
            )

    def _memory_surface_context(self, *, archived: bool, render_mode: str) -> dict[str, object]:
        entries = self.archived_memory_entries if archived else self.memory_entries
        rendered_count = (
            int(getattr(self, "archived_memory_rendered_count", 0))
            if archived
            else int(getattr(self, "memory_rendered_count", 0))
        )
        group_counts = (
            getattr(self, "archived_memory_row_group_counts", (0,)) if archived else getattr(self, "memory_row_group_counts", (0,))
        )
        grouped_sections = (
            int(group_counts[min(rendered_count, len(group_counts) - 1)])
            if group_counts
            else memory_group_count(entries[:rendered_count])
        )
        return self._heavy_surface_context(
            view_name="archived_memory" if archived else "memory",
            fetched_count=len(entries),
            rendered_count=rendered_count,
            grouped_sections=grouped_sections,
            has_more=(rendered_count < len(entries))
            or (self.archived_memory_entries_has_more if archived else self.memory_entries_has_more),
            render_mode=render_mode,
        )

    def _render_memory_entries_from_cache(self, *, archived: bool, render_mode: str) -> None:
        entries = self.archived_memory_entries if archived else self.memory_entries
        inner = self.archived_memory_list_inner if archived else self.memory_list_inner
        render_limit = self.archived_memory_render_limit if archived else self.memory_render_limit
        rendered_count = self.archived_memory_rendered_count if archived else self.memory_rendered_count
        full_signature = self.archived_memory_render_signature if archived else self.memory_render_signature
        current_signature = self.archived_memory_cached_signature if archived else self.memory_cached_signature
        row_descriptors = self.archived_memory_row_descriptors if archived else self.memory_row_descriptors
        row_entry_map = self.archived_memory_row_entry_map if archived else self.memory_row_entry_map
        row_descriptor_offsets = (
            self.archived_memory_row_descriptor_offsets if archived else self.memory_row_descriptor_offsets
        )
        row_group_counts = self.archived_memory_row_group_counts if archived else self.memory_row_group_counts
        has_more_available = self.archived_memory_entries_has_more if archived else self.memory_entries_has_more
        visible_entries, target_count, has_more = bounded_entries_slice(
            entries,
            render_limit=render_limit,
            has_more_available=has_more_available,
        )
        button_attr = "archived_memory_load_more_button" if archived else "memory_load_more_button"
        existing_button = getattr(self, button_attr, None)
        if existing_button is not None and existing_button.winfo_exists():
            existing_button.destroy()
        setattr(self, button_attr, None)
        signature_extends_rendered = (
            rendered_count > 0
            and len(current_signature) >= rendered_count
            and tuple(current_signature[:rendered_count]) == tuple(full_signature[:rendered_count])
        )
        if full_signature != current_signature and not signature_extends_rendered:
            for child in inner.winfo_children():
                child.destroy()
            rendered_count = 0
        if not entries:
            for child in inner.winfo_children():
                child.destroy()
            if archived:
                self.archived_memory_rendered_count = 0
                self.archived_memory_render_signature = current_signature
                self._set_text_widget(self.archived_memory_preview, "No archived memory yet.")
                self._set_active_browser_hover_row("archived_memory", None)
            else:
                self.memory_rendered_count = 0
                self.memory_render_signature = current_signature
                self._set_memory_preview_empty(empty_state_text("memory"))
                self._set_active_browser_hover_row("memory", None)
            tk.Label(
                inner,
                text="No archived memory yet." if archived else empty_state_text("memory"),
                anchor="w",
                justify=tk.LEFT,
                bg=self.current_palette["panel_bg"],
                fg=self.current_palette["text_muted"],
            ).pack(fill=tk.X)
            return
        if archived:
            self._set_text_widget(self.archived_memory_preview, "Select archived memory to inspect it.")
        else:
            self._set_memory_preview_empty("Select a memory to inspect it.")
        if not row_descriptors or len(row_descriptor_offsets) <= target_count:
            (
                row_descriptors,
                row_entry_map,
                row_descriptor_offsets,
                row_group_counts,
            ) = build_memory_row_cache(entries)
            if archived:
                self.archived_memory_row_descriptors = row_descriptors
                self.archived_memory_row_entry_map = row_entry_map
                self.archived_memory_row_descriptor_offsets = row_descriptor_offsets
                self.archived_memory_row_group_counts = row_group_counts
            else:
                self.memory_row_descriptors = row_descriptors
                self.memory_row_entry_map = row_entry_map
                self.memory_row_descriptor_offsets = row_descriptor_offsets
                self.memory_row_group_counts = row_group_counts
        target_descriptors = memory_row_cache_slice(
            row_descriptors,
            row_descriptor_offsets,
            visible_count=target_count,
        )
        grouped_sections = (
            int(row_group_counts[min(target_count, len(row_group_counts) - 1)])
            if row_group_counts
            else memory_group_count(visible_entries)
        )
        if rendered_count >= target_count and inner.winfo_children():
            self._debug_event(
                "memory_rows_reused",
                archived=archived,
                reason="visible_slice_current",
                fetched_count=len(entries),
                rendered_count=target_count,
            )
            if archived:
                self.archived_memory_render_signature = current_signature
                self.archived_memory_rendered_count = target_count
            else:
                self.memory_render_signature = current_signature
                self.memory_rendered_count = target_count
            if render_mode == "render_from_cached_slice":
                self._debug_event(
                    "render_from_cached_slice",
                    view="archived_memory" if archived else "memory",
                    fetched_count=len(entries),
                    rendered_count=target_count,
                )
            self._update_memory_load_more_button(
                archived=archived,
                rendered_count=target_count,
                total_count=len(entries),
                has_more_available=has_more_available,
            )
            self._debug_event(
                "heavy_surface_render",
                view="archived_memory" if archived else "memory",
                fetched_count=len(entries),
                rendered_count=target_count,
                grouped_sections=grouped_sections,
                has_more=has_more,
                render_mode=render_mode,
            )
            self._clear_stale_browser_hover("archived_memory" if archived else "memory")
            return
        self._rebuild_memory_rows(
            descriptors=target_descriptors,
            entry_map=row_entry_map,
            inner=inner,
            command_builder=self._show_archived_memory_entry if archived else self._show_memory_entry,
            archive_enabled=not archived,
            already_rendered=rendered_count,
        )
        if archived:
            self.archived_memory_render_signature = current_signature
            self.archived_memory_rendered_count = target_count
        else:
            self.memory_render_signature = current_signature
            self.memory_rendered_count = target_count
        if render_mode == "render_from_cached_slice":
            self._debug_event(
                "render_from_cached_slice",
                view="archived_memory" if archived else "memory",
                fetched_count=len(entries),
                rendered_count=target_count,
            )
        self._update_memory_load_more_button(
            archived=archived,
            rendered_count=target_count,
            total_count=len(entries),
            has_more_available=has_more_available,
        )
        self._debug_event(
            "heavy_surface_render",
            view="archived_memory" if archived else "memory",
            fetched_count=len(entries),
            rendered_count=target_count,
            grouped_sections=grouped_sections,
            has_more=has_more,
            render_mode=render_mode,
        )
        self._clear_stale_browser_hover("archived_memory" if archived else "memory")

    def _show_memory_entry(self, entry: dict[str, object]) -> None:
        title = str(entry.get("title") or "Memory").strip()
        kind = str(entry.get("kind") or "memory").replace("_", " ").title()
        created = human_date_label(entry.get("created_at")) or "unknown"
        content = str(entry.get("content") or "").strip()
        body = [title, "", f"{kind} • {created}"]
        if content:
            body.extend(["", content])
        self._set_text_widget(self.memory_preview, "\n".join(body).strip())

    def _show_archived_memory_entry(self, entry: dict[str, object]) -> None:
        title = str(entry.get("title") or "Memory").strip()
        kind = str(entry.get("kind") or "memory").replace("_", " ").title()
        created = human_date_label(entry.get("created_at")) or "unknown"
        content = str(entry.get("content") or "").strip()
        body = [title, "", f"{kind} • {created}"]
        if content:
            body.extend(["", content])
        self._set_text_widget(self.archived_memory_preview, "\n".join(body).strip())

    @staticmethod
    def _memory_entries_signature(entries: list[dict[str, object]]) -> tuple[tuple[str, str, str, str], ...]:
        return memory_entries_signature(entries)

    def _memory_row_descriptors(
        self,
        entries: list[dict[str, object]],
    ) -> tuple[list[tuple[object, ...]], dict[str, dict[str, object]]]:
        return build_memory_row_descriptors(entries)

    def _memory_rows_match_descriptors(self, widgets: list[tk.Widget], descriptors: list[tuple[object, ...]]) -> bool:
        return memory_rows_match_descriptors(widgets, descriptors)

    def _rebuild_memory_rows(
        self,
        *,
        descriptors: list[tuple[object, ...]] | None = None,
        entry_map: dict[str, dict[str, object]] | None = None,
        entries: list[dict[str, object]] | None = None,
        inner: tk.Frame,
        command_builder: Callable[[dict[str, object]], object],
        archive_enabled: bool,
        already_rendered: int = 0,
    ) -> None:
        if descriptors is None or entry_map is None:
            cache_descriptors, cache_entry_map, _, _ = build_memory_row_cache(entries or [])
            if descriptors is None:
                descriptors = cache_descriptors
            if entry_map is None:
                entry_map = cache_entry_map
        existing_widgets = list(inner.winfo_children())
        if self._memory_rows_match_descriptors(existing_widgets, descriptors):
            self._debug_event(
                "memory_rows_reused",
                archived=not archive_enabled,
                reason="descriptors_match",
            )
            for index, descriptor in enumerate(descriptors):
                widget = existing_widgets[index]
                if descriptor[0] == "header":
                    self._update_browser_header(
                        widget,
                        str(descriptor[1]),
                        first=index == 0,
                        top_padding=int(descriptor[2]),
                    )  # type: ignore[arg-type]
                    continue
                entry = entry_map.get(str(descriptor[4]))
                if not isinstance(entry, dict):
                    continue
                actions: list[tuple[str, object]] = []
                if archive_enabled:
                    actions.append(("Archive", lambda selected=entry: self._archive_memory_entry(selected)))
                actions.append(("Delete", lambda selected=entry: self._delete_memory_entry(selected)))
                self._update_browser_row(
                    widget,  # type: ignore[arg-type]
                    title=str(descriptor[1]),
                    detail=str(descriptor[2]),
                    secondary=str(descriptor[3]),
                    command=lambda selected=entry: command_builder(selected),
                    context_actions=actions,
                    surface="archived_memory" if not archive_enabled else "memory",
                    descriptor_identity=descriptor,
                    command_target_id=str(descriptor[4]),
                )
            return

        prefix_descriptors = descriptors[: len(existing_widgets)]
        if (
            existing_widgets
            and len(prefix_descriptors) == len(existing_widgets)
            and self._memory_rows_match_descriptors(existing_widgets, prefix_descriptors)
        ):
            self._debug_event(
                "memory_rows_reused",
                archived=not archive_enabled,
                reason="prefix_append",
            )
            for descriptor in descriptors[len(existing_widgets) :]:
                if descriptor[0] == "header":
                    self._build_browser_header(inner, str(descriptor[1]), top_padding=int(descriptor[2]))
                    continue
                entry = entry_map.get(str(descriptor[4]))
                if not isinstance(entry, dict):
                    continue
                actions: list[tuple[str, object]] = []
                if archive_enabled:
                    actions.append(("Archive", lambda selected=entry: self._archive_memory_entry(selected)))
                actions.append(("Delete", lambda selected=entry: self._delete_memory_entry(selected)))
                row = self._build_browser_row(
                    inner,
                    title=str(descriptor[1]),
                    detail=str(descriptor[2]),
                    secondary=str(descriptor[3]),
                    command=lambda selected=entry: command_builder(selected),
                    context_actions=actions,
                    surface="archived_memory" if not archive_enabled else "memory",
                    command_target_id=str(descriptor[4]),
                )
                row._browser_descriptor = descriptor  # type: ignore[attr-defined]
            return

        self._debug_event(
            "memory_rows_rebuilt",
            archived=not archive_enabled,
        )
        for child in existing_widgets:
            child.destroy()
        for descriptor in descriptors:
            if descriptor[0] == "header":
                self._build_browser_header(inner, str(descriptor[1]), top_padding=int(descriptor[2]))
                continue
            entry = entry_map.get(str(descriptor[4]))
            if not isinstance(entry, dict):
                continue
            actions: list[tuple[str, object]] = []
            if archive_enabled:
                actions.append(("Archive", lambda selected=entry: self._archive_memory_entry(selected)))
            actions.append(("Delete", lambda selected=entry: self._delete_memory_entry(selected)))
            row = self._build_browser_row(
                inner,
                title=str(descriptor[1]),
                detail=str(descriptor[2]),
                secondary=str(descriptor[3]),
                command=lambda selected=entry: command_builder(selected),
                context_actions=actions,
                surface="archived_memory" if not archive_enabled else "memory",
                command_target_id=str(descriptor[4]),
            )
            row._browser_descriptor = descriptor  # type: ignore[attr-defined]

    def _open_recent_session_by_data(self, session: dict[str, object]) -> None:
        session_id = str(session.get("session_id") or "").strip()
        if not session_id:
            return
        if not any(
            isinstance(item, dict) and str(item.get("session_id") or "") == session_id
            for item in self.recent_sessions_cache
        ):
            self.recent_sessions_cache.append(dict(session))
        self._load_session(session_id)

    def _render_session_rows_from_cache(self, *, archived: bool) -> None:
        inner = self.archived_list_inner if archived else self.recent_list_inner
        rows = self.archived_sessions_rows if archived else self.recent_sessions_rows
        render_limit = self.archived_sessions_render_limit if archived else self.recent_sessions_render_limit
        render_signature = (
            self.archived_sessions_render_signature if archived else self.recent_sessions_render_signature
        )
        rendered_count = self.archived_sessions_rendered_count if archived else self.recent_sessions_rendered_count
        total_sessions = sum(1 for row in rows if row.get("kind") == "session")
        target_session_count = min(
            render_limit,
            total_sessions,
            200 if archived else self.RECENT_SESSIONS_FETCH_LIMIT,
        )
        if render_signature != self._session_signature(
            self.archived_sessions_cache if archived else self.recent_sessions_cache
        ):
            for child in inner.winfo_children():
                child.destroy()
            rendered_count = 0
        if not rows:
            for child in inner.winfo_children():
                child.destroy()
            self._set_active_browser_hover_row("archived" if archived else "recent", None)
            tk.Label(
                inner,
                text="No archived chats yet." if archived else empty_state_text("recent"),
                anchor="w",
                justify=tk.LEFT,
                bg=self.current_palette["panel_bg"],
                fg=self.current_palette["text_muted"],
            ).pack(fill=tk.X)
            if archived:
                self.archived_sessions_render_signature = self.archived_sessions_signature
                self.archived_sessions_rendered_count = 0
            else:
                self.recent_sessions_render_signature = self.recent_sessions_signature
                self.recent_sessions_rendered_count = 0
            return
        if rendered_count == 0:
            for child in inner.winfo_children():
                child.destroy()
        if rendered_count >= target_session_count and render_signature == (
            self.archived_sessions_signature if archived else self.recent_sessions_signature
        ) and inner.winfo_children():
            self._debug_event("browser_rows_reused", view="archived" if archived else "recent", reason="signature_current")
            self._update_session_load_more_button(archived=archived, rendered_count=rendered_count)
            self._debug_event(
                "heavy_surface_render",
                view="archived" if archived else "recent",
                fetched_count=total_sessions,
                rendered_count=target_session_count,
                grouped_sections=sum(1 for row in rows if row.get("kind") == "header"),
                has_more=target_session_count < total_sessions,
                render_mode="render-only",
            )
            self._clear_stale_browser_hover("archived" if archived else "recent")
            return
        already_rendered = rendered_count
        target_descriptors = self._build_session_row_descriptors(
            rows=rows,
            target_session_count=target_session_count,
            already_rendered=already_rendered,
        )
        existing_widgets = list(inner.winfo_children())
        if self._browser_rows_match_descriptors(existing_widgets, target_descriptors):
            self._debug_event("browser_rows_reused", view="archived" if archived else "recent", reason="descriptors_match")
            visible_sessions: dict[str, dict[str, object]] = {}
            session_index = already_rendered
            for row in rows:
                if row.get("kind") != "session":
                    continue
                session_index += 1
                if session_index <= already_rendered:
                    continue
                if session_index > target_session_count:
                    break
                session = row.get("session")
                if not isinstance(session, dict):
                    continue
                session_id = str(session.get("session_id") or "").strip()
                if session_id:
                    visible_sessions[session_id] = session
            for index, descriptor in enumerate(target_descriptors):
                widget = existing_widgets[index]
                if descriptor[0] == "header":
                    self._update_browser_header(
                        widget,
                        str(descriptor[1]),
                        first=index == 0,
                        top_padding=int(descriptor[2]),
                    )  # type: ignore[arg-type]
                    continue
                session_row = widget
                session_id = str(descriptor[1] if len(descriptor) > 1 else "" or "").strip()
                session = visible_sessions.get(session_id)
                if not isinstance(session, dict):
                    continue
                title = self._session_display_title(session)
                detail = human_date_label(session.get("created_at"))
                secondary = str(session.get("mode") or "conversation").replace("_", " ").title()
                descriptor_identity = (
                    "session",
                    str(session.get("session_id") or ""),
                    title,
                    detail,
                    secondary,
                )
                actions = [("Delete", lambda selected=session: self._delete_session(selected))]
                if not archived:
                    actions.insert(0, ("Archive", lambda selected=session: self._archive_session(selected)))
                self._update_browser_row(
                    session_row,  # type: ignore[arg-type]
                    title=title,
                    detail=detail,
                    secondary=secondary,
                    command=lambda selected=session: self._open_recent_session_by_data(selected),
                    context_actions=actions,
                    surface="archived" if archived else "recent",
                    descriptor_identity=descriptor_identity,
                    command_target_id=str(session.get("session_id") or ""),
                    render_state="reused",
                )
            if archived:
                self.archived_sessions_render_signature = self.archived_sessions_signature
                self.archived_sessions_rendered_count = target_session_count
            else:
                self.recent_sessions_render_signature = self.recent_sessions_signature
                self.recent_sessions_rendered_count = target_session_count
            self._update_session_load_more_button(archived=archived, rendered_count=target_session_count)
            return
        self._debug_event("browser_rows_rebuilt", view="archived" if archived else "recent")
        session_count = 0
        current_group = None
        if already_rendered:
            for row in rows:
                if row.get("kind") == "header":
                    current_group = str(row.get("label") or current_group or "")
                elif row.get("kind") == "session":
                    session_count += 1
                    if session_count >= already_rendered:
                        break
        rendered_sessions = session_count
        for row in rows:
            if row.get("kind") == "header":
                group_label = str(row.get("label") or "")
                if rendered_sessions < already_rendered:
                    current_group = group_label
                    continue
                if rendered_sessions >= target_session_count:
                    break
                if current_group == group_label:
                    continue
                self._build_browser_header(inner, group_label, top_padding=14)
                current_group = group_label
                continue
            session = row.get("session")
            if not isinstance(session, dict):
                continue
            rendered_sessions += 1
            if rendered_sessions <= already_rendered:
                continue
            if rendered_sessions > target_session_count:
                break
            title = self._session_display_title(session)
            detail = human_date_label(session.get("created_at"))
            secondary = str(session.get("mode") or "conversation").replace("_", " ").title()
            descriptor_identity = (
                "session",
                str(session.get("session_id") or ""),
                title,
                detail,
                secondary,
            )
            actions = [("Delete", lambda selected=session: self._delete_session(selected))]
            if not archived:
                actions.insert(0, ("Archive", lambda selected=session: self._archive_session(selected)))
            self._build_browser_row(
                inner,
                title=title,
                detail=detail,
                secondary=secondary,
                command=lambda selected=session: self._open_recent_session_by_data(selected),
                context_actions=actions,
                surface="archived" if archived else "recent",
                descriptor_identity=descriptor_identity,
                command_target_id=str(session.get("session_id") or ""),
                render_state="rebuilt",
            )
        if archived:
            self.archived_sessions_render_signature = self.archived_sessions_signature
            self.archived_sessions_rendered_count = target_session_count
        else:
            self.recent_sessions_render_signature = self.recent_sessions_signature
            self.recent_sessions_rendered_count = target_session_count
        self._update_session_load_more_button(archived=archived, rendered_count=target_session_count)
        self._debug_event(
            "heavy_surface_render",
            view="archived" if archived else "recent",
            fetched_count=total_sessions,
            rendered_count=target_session_count,
            grouped_sections=sum(1 for row in rows if row.get("kind") == "header"),
            has_more=target_session_count < total_sessions,
            render_mode="render-only",
        )
        self._clear_stale_browser_hover("archived" if archived else "recent")

    def _update_session_load_more_button(self, *, archived: bool, rendered_count: int) -> None:
        rows = self.archived_sessions_rows if archived else self.recent_sessions_rows
        total_sessions = sum(1 for row in rows if row.get("kind") == "session")
        self._update_heavy_surface_load_more_button(
            view_name="archived" if archived else "recent",
            parent=self.archived_list_inner if archived else self.recent_list_inner,
            rendered_count=rendered_count,
            total_count=total_sessions,
            command=self._load_more_archived_sessions if archived else self._load_more_recent_sessions,
        )

    def _update_memory_load_more_button(
        self,
        *,
        archived: bool,
        rendered_count: int,
        total_count: int,
        has_more_available: bool = False,
    ) -> None:
        self._update_heavy_surface_load_more_button(
            view_name="archived_memory" if archived else "memory",
            parent=self.archived_memory_list_inner if archived else self.memory_list_inner,
            rendered_count=rendered_count,
            total_count=total_count,
            command=self._load_more_archived_memory_entries if archived else self._load_more_memory_entries,
            has_more_available=has_more_available,
        )

    def _update_heavy_surface_load_more_button(
        self,
        *,
        view_name: str,
        parent: tk.Widget,
        rendered_count: int,
        total_count: int,
        command: Callable[[], None],
        has_more_available: bool = False,
    ) -> None:
        button_attr_map = {
            "recent": "recent_sessions_load_more_button",
            "archived": "archived_sessions_load_more_button",
            "memory": "memory_load_more_button",
            "archived_memory": "archived_memory_load_more_button",
        }
        button_attr = button_attr_map[view_name]
        existing_button = getattr(self, button_attr, None)
        if existing_button is not None and existing_button.winfo_exists():
            existing_button.destroy()
        setattr(self, button_attr, None)
        if rendered_count >= total_count and not has_more_available:
            return
        style = resolve_load_more_palette(palette=self.current_palette)
        button = tk.Button(
            parent,
            text="Load More",
            command=command,
            relief=tk.FLAT,
            bd=0,
            padx=18,
            pady=8,
            cursor="hand2",
            highlightthickness=0,
            width=12,
            **style,
        )
        button.pack(anchor="center", pady=(14, 8))
        setattr(self, button_attr, button)

    @staticmethod
    def _session_signature(sessions: list[dict[str, object]]) -> tuple[tuple[str, str, str, str, bool], ...]:
        return tuple(
            (
                str(item.get("session_id") or ""),
                str(item.get("title") or item.get("prompt") or item.get("summary") or ""),
                str(item.get("created_at") or ""),
                str(item.get("mode") or ""),
                bool(item.get("archived", False)),
            )
            for item in sessions
            if isinstance(item, dict)
        )

    def _browser_rows_match_descriptors(self, widgets: list[tk.Widget], descriptors: list[tuple[object, ...]]) -> bool:
        if len(widgets) != len(descriptors):
            return False
        return all(getattr(widget, "_browser_descriptor", None) == descriptor for widget, descriptor in zip(widgets, descriptors))

    def _build_session_row_descriptors(
        self,
        *,
        rows: list[dict[str, object]],
        target_session_count: int,
        already_rendered: int,
    ) -> list[tuple[object, ...]]:
        descriptors: list[tuple[object, ...]] = []
        current_group = None
        rendered_sessions = 0
        for row in rows:
            if row.get("kind") == "header":
                group_label = str(row.get("label") or "")
                if rendered_sessions < already_rendered:
                    current_group = group_label
                    continue
                if rendered_sessions >= target_session_count:
                    break
                if current_group == group_label:
                    continue
                descriptors.append(("header", group_label, 14))
                current_group = group_label
                continue
            session = row.get("session")
            if not isinstance(session, dict):
                continue
            rendered_sessions += 1
            if rendered_sessions <= already_rendered:
                continue
            if rendered_sessions > target_session_count:
                break
            title = self._session_display_title(session)
            descriptors.append(
                (
                    "session",
                    str(session.get("session_id") or ""),
                    title,
                    human_date_label(session.get("created_at")),
                    str(session.get("mode") or "conversation").replace("_", " ").title(),
                )
            )
        return descriptors

    @staticmethod
    def _session_display_title(session: dict[str, object]) -> str:
        return (
            str(
                session.get("title")
                or session.get("prompt")
                or session.get("summary")
                or session.get("session_id")
                or "Chat"
            ).strip()
            or "Chat"
        )

    def _remove_session_from_caches(self, session_id: str) -> None:
        self.recent_sessions_cache = [
            item for item in self.recent_sessions_cache
            if isinstance(item, dict) and str(item.get("session_id") or "") != session_id
        ]
        self.recent_sessions_rows = grouped_session_rows(self.recent_sessions_cache)
        self.recent_sessions_signature = self._session_signature(self.recent_sessions_cache)
        self.archived_sessions_cache = [
            item for item in self.archived_sessions_cache
            if isinstance(item, dict) and str(item.get("session_id") or "") != session_id
        ]
        self.archived_sessions_rows = grouped_session_rows(self.archived_sessions_cache)
        self.archived_sessions_signature = self._session_signature(self.archived_sessions_cache)
        self._conversation_cache_dirty = True
        self._schedule_conversation_cache_write()

    def _mark_current_session_archived(self) -> None:
        self._debug_event("active_session_marked_archived", session_id=self.session_id)
        self.recent_sessions_view_dirty = True
        self.archived_sessions_view_dirty = True
        self._conversation_cache_dirty = True
        self._schedule_conversation_cache_write()

    def _archive_session(self, session: dict[str, object]) -> None:
        session_id = str(session.get("session_id") or "").strip()
        if not session_id:
            return
        if not messagebox.askyesno("Archive Chat", "Archive this chat? It will be removed from the main conversations list.", parent=self.root):
            return
        try:
            if hasattr(self.controller, "archive_session"):
                self.controller.archive_session(session_id)
        except Exception:
            self.status_var.set("Couldn't archive chat")
            self._append_system_line("Lumen couldn't archive that chat.")
            return
        self.status_var.set("Chat archived")
        archived_session = dict(session)
        self._remove_session_from_caches(session_id)
        self.archived_sessions_cache.insert(0, archived_session)
        archived_session["archived"] = True
        self.archived_sessions_rows = grouped_session_rows(self.archived_sessions_cache)
        self.archived_sessions_signature = self._session_signature(self.archived_sessions_cache)
        self.recent_sessions_view_dirty = True
        self.archived_sessions_view_dirty = True
        if session_id == self.session_id:
            self._mark_current_session_archived()
        if hasattr(self, "recent_list_inner"):
            self._render_recent_sessions_from_cache()
        if hasattr(self, "archived_list_inner"):
            self._render_archived_sessions_from_cache()

    def _delete_session(self, session: dict[str, object]) -> None:
        session_id = str(session.get("session_id") or "").strip()
        if not session_id:
            return
        if not messagebox.askyesno("Delete Chat", "Delete this chat permanently? This cannot be undone.", parent=self.root):
            return
        if hasattr(self.controller, "delete_session"):
            self.controller.delete_session(session_id)
        if session_id == self.session_id:
            self._start_new_session()
        self.status_var.set("Chat deleted")
        self._remove_session_from_caches(session_id)
        self.recent_sessions_view_dirty = True
        self.archived_sessions_view_dirty = True
        if hasattr(self, "recent_list_inner"):
            self._render_recent_sessions_from_cache()
        if hasattr(self, "archived_list_inner"):
            self._render_archived_sessions_from_cache()

    def _archive_memory_entry(self, entry: dict[str, object]) -> None:
        path = self._memory_entry_action_path(entry)
        if not path:
            return
        if not messagebox.askyesno("Archive Memory", "Archive this memory item? It will be removed from the main memory list.", parent=self.root):
            return
        try:
            if hasattr(self.controller, "archive_memory"):
                self.controller.archive_memory(kind=str(entry.get("kind") or "personal_memory"), path=path)
        except Exception as exc:
            self._record_desktop_crash(source="memory.archive_action", exc=exc)
            self._surface_runtime_failure(
                "Lumen couldn't archive that memory item.",
                source="memory.archive_action",
                category="action_failure",
                context={
                    "surface": "memory",
                    "action": "archive",
                    "path": path,
                },
            )
            self.memory_view_dirty = True
            self.archived_memory_view_dirty = True
            return
        self.status_var.set("Memory archived")
        self.memory_view_dirty = True
        self._invalidate_archived_memory_state()
        if hasattr(self, "memory_list_inner"):
            self._refresh_memory_view()
        if hasattr(self, "archived_memory_list_inner"):
            self._refresh_archived_memory_view()

    def _delete_memory_entry(self, entry: dict[str, object]) -> None:
        path = self._memory_entry_action_path(entry)
        if not path:
            return
        if not messagebox.askyesno("Delete Memory", "Delete this memory item permanently? This cannot be undone.", parent=self.root):
            return
        try:
            if hasattr(self.controller, "delete_memory"):
                self.controller.delete_memory(kind=str(entry.get("kind") or "personal_memory"), path=path)
        except Exception as exc:
            self._record_desktop_crash(source="memory.delete_action", exc=exc)
            self._surface_runtime_failure(
                "Lumen couldn't delete that memory item.",
                source="memory.delete_action",
                category="action_failure",
                context={
                    "surface": "memory",
                    "action": "delete",
                    "path": path,
                },
            )
            self.memory_view_dirty = True
            self.archived_memory_view_dirty = True
            return
        self.status_var.set("Memory deleted")
        self.memory_view_dirty = True
        self._invalidate_archived_memory_state()
        if hasattr(self, "memory_list_inner"):
            self._refresh_memory_view()
        if hasattr(self, "archived_memory_list_inner"):
            self._refresh_archived_memory_view()

    @staticmethod
    def _memory_entry_action_path(entry: dict[str, object]) -> str:
        return str(
            entry.get("memory_item_id")
            or entry.get("id")
            or entry.get("source_id")
            or entry.get("note_path")
            or entry.get("entry_path")
            or entry.get("artifact_path")
            or ""
        ).strip()

    def _load_more_recent_sessions(self) -> None:
        self.recent_sessions_render_limit = min(
            self.RECENT_SESSIONS_FETCH_LIMIT,
            self.recent_sessions_render_limit + self.recent_sessions_render_step,
        )
        self._render_recent_sessions_from_cache()

    def _load_more_archived_sessions(self) -> None:
        self.archived_sessions_render_limit += self.archived_sessions_render_step
        self._render_archived_sessions_from_cache()

    def _load_more_memory_entries(self) -> None:
        self.memory_render_limit += self.memory_render_step
        if self.memory_entries_has_more and len(self.memory_entries) < self.memory_render_limit:
            if self.memory_fetch_in_flight and self.memory_requested_fetch_limit == self.memory_render_limit:
                self._debug_event(
                    "memory_load_more_suppressed",
                    view="memory",
                    reason="duplicate_target_window",
                    target_fetch_limit=self.memory_render_limit,
                )
                return
            self._start_memory_surface_fetch(
                archived=False,
                fetch_limit=self.memory_render_limit,
                fetch_reason="extended",
            )
            return
        self._render_memory_entries_from_cache(archived=False, render_mode="render-only")

    def _load_more_archived_memory_entries(self) -> None:
        self.archived_memory_render_limit += self.archived_memory_render_step
        if self.archived_memory_entries_has_more and len(self.archived_memory_entries) < self.archived_memory_render_limit:
            if (
                self.archived_memory_fetch_in_flight
                and self.archived_memory_requested_fetch_limit == self.archived_memory_render_limit
            ):
                self._debug_event(
                    "memory_load_more_suppressed",
                    view="archived_memory",
                    reason="duplicate_target_window",
                    target_fetch_limit=self.archived_memory_render_limit,
                )
                return
            self._start_memory_surface_fetch(
                archived=True,
                fetch_limit=self.archived_memory_render_limit,
                fetch_reason="extended",
            )
            return
        self._render_archived_memory_from_cache(render_mode="render-only")

    def _conversation_cache_payload(self) -> dict[str, object]:
        return {
            "recent": [self._session_metadata_snapshot(item, archived=False) for item in self.recent_sessions_cache],
            "archived": [self._session_metadata_snapshot(item, archived=True) for item in self.archived_sessions_cache],
            "memory": self._memory_cache_snapshot(archived=False),
            "archived_memory": self._memory_cache_snapshot(archived=True),
        }

    @staticmethod
    def _session_metadata_snapshot(session: dict[str, object], *, archived: bool) -> dict[str, object]:
        return {
            "session_id": str(session.get("session_id") or ""),
            "title": str(session.get("title") or ""),
            "summary": str(session.get("summary") or ""),
            "prompt": str(session.get("prompt") or ""),
            "created_at": str(session.get("created_at") or ""),
            "mode": str(session.get("mode") or ""),
            "archived": archived or bool(session.get("archived", False)),
        }

    def _memory_cache_snapshot(self, *, archived: bool) -> dict[str, object]:
        entries = self.archived_memory_entries if archived else self.memory_entries
        signature = self.archived_memory_cached_signature if archived else self.memory_cached_signature
        has_more = self.archived_memory_entries_has_more if archived else self.memory_entries_has_more
        return {
            "entries": [self._memory_entry_snapshot(item) for item in entries if isinstance(item, dict)],
            "signature": [list(item) for item in signature],
            "has_more": bool(has_more),
        }

    @staticmethod
    def _memory_entry_snapshot(entry: dict[str, object]) -> dict[str, object]:
        return {
            "title": str(entry.get("title") or ""),
            "content": str(entry.get("content") or ""),
            "created_at": str(entry.get("created_at") or ""),
            "kind": str(entry.get("kind") or ""),
            "entry_path": str(entry.get("entry_path") or ""),
            "note_path": str(entry.get("note_path") or ""),
            "artifact_path": str(entry.get("artifact_path") or ""),
            "memory_item_id": str(entry.get("memory_item_id") or ""),
            "id": str(entry.get("id") or ""),
            "source_id": str(entry.get("source_id") or ""),
            "archived": bool(entry.get("archived", False)),
        }

    def _restore_memory_cache_payload(self, *, archived: bool, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        entries = [dict(item) for item in payload.get("entries", []) if isinstance(item, dict)]
        signature_payload = payload.get("signature", [])
        signature = tuple(
            tuple(str(part) for part in item[:4])
            for item in signature_payload
            if isinstance(item, (list, tuple)) and len(item) >= 4
        )
        (
            row_descriptors,
            row_entry_map,
            row_descriptor_offsets,
            row_group_counts,
        ) = build_memory_row_cache(entries)
        has_more = bool(payload.get("has_more", False))
        if archived:
            self.archived_memory_entries = entries
            self.archived_memory_cached_signature = signature
            self.archived_memory_row_descriptors = row_descriptors
            self.archived_memory_row_entry_map = row_entry_map
            self.archived_memory_row_descriptor_offsets = row_descriptor_offsets
            self.archived_memory_row_group_counts = row_group_counts
            self.archived_memory_entries_has_more = has_more
        else:
            self.memory_entries = entries
            self.memory_cached_signature = signature
            self.memory_row_descriptors = row_descriptors
            self.memory_row_entry_map = row_entry_map
            self.memory_row_descriptor_offsets = row_descriptor_offsets
            self.memory_row_group_counts = row_group_counts
            self.memory_entries_has_more = has_more

    def _load_conversation_cache(self) -> None:
        def _apply() -> None:
            try:
                if not self.conversation_cache_path.exists():
                    return
                payload = json.loads(self.conversation_cache_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, TypeError):
                return
            if not isinstance(payload, dict):
                return
            recent = [
                dict(item)
                for item in payload.get("recent", [])
                if isinstance(item, dict) and is_user_visible_session(item)
            ]
            archived = [
                dict(item)
                for item in payload.get("archived", [])
                if isinstance(item, dict) and is_user_visible_session(item)
            ]
            self.recent_sessions_cache = recent
            self.recent_sessions_rows = grouped_session_rows(recent)
            self.recent_sessions_signature = self._session_signature(recent)
            self.recent_sessions_restored_from_disk = bool(recent)
            self.archived_sessions_cache = archived
            self.archived_sessions_rows = grouped_session_rows(archived)
            self.archived_sessions_signature = self._session_signature(archived)
            self.archived_sessions_restored_from_disk = bool(archived)
            self._restore_memory_cache_payload(archived=False, payload=payload.get("memory"))
            self._restore_memory_cache_payload(archived=True, payload=payload.get("archived_memory"))
            self._conversation_cache_signature = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            self._conversation_cache_dirty = False

        self._timed_ui_call("conversation_cache_load", _apply)

    def _write_conversation_cache_safe(self) -> None:
        self._debug_event("write_conversation_cache")
        if not hasattr(self, "conversation_cache_path"):
            return
        payload = self._conversation_cache_payload()
        payload_signature = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        if not self._conversation_cache_dirty and payload_signature == self._conversation_cache_signature:
            return
        try:
            self.conversation_cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.conversation_cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError:
            return
        self._conversation_cache_signature = payload_signature
        self._conversation_cache_dirty = False

    def _mark_conversation_cache_dirty(self) -> None:
        self._conversation_cache_dirty = True
        if hasattr(self, "_conversation_cache_write_scheduled"):
            self._schedule_conversation_cache_write()

    def _schedule_conversation_cache_write(self) -> None:
        if self._conversation_cache_write_scheduled:
            return
        self._conversation_cache_write_scheduled = True

        def _write() -> None:
            self.conversation_cache_write_job = None
            self._conversation_cache_write_scheduled = False
            self._timed_ui_call("conversation_cache_write", self._write_conversation_cache_safe)

        self.conversation_cache_write_job = self.root.after_idle(_write)

    def _restore_session_cache_from_disk(self, view_name: str) -> bool:
        if view_name == "recent" and self.recent_sessions_rows:
            return True
        if view_name == "archived" and self.archived_sessions_rows:
            return True
        if view_name == "memory" and self.memory_entries:
            return True
        if view_name == "archived_memory" and self.archived_memory_entries:
            return True
        self._load_conversation_cache()
        return (view_name == "recent" and bool(self.recent_sessions_rows)) or (
            view_name == "archived" and bool(self.archived_sessions_rows)
        ) or (view_name == "memory" and bool(self.memory_entries)) or (
            view_name == "archived_memory" and bool(self.archived_memory_entries)
        )

    def _render_recent_sessions_from_cache_if_visible(self) -> None:
        if getattr(self, "current_view", "") == "recent" and hasattr(self, "recent_list_inner"):
            self._render_recent_sessions_from_cache()

    def _render_archived_sessions_from_cache_if_visible(self) -> None:
        if getattr(self, "current_view", "") == "archived" and hasattr(self, "archived_list_inner"):
            self._render_archived_sessions_from_cache()

    def _refresh_memory_view_if_visible(self) -> None:
        if getattr(self, "current_view", "") == "memory" and hasattr(self, "memory_list_inner"):
            self._refresh_memory_view()

    def _refresh_archived_memory_view_if_visible(self) -> None:
        if getattr(self, "current_view", "") == "archived_memory" and hasattr(self, "archived_memory_list_inner"):
            self._refresh_archived_memory_view()

    def _on_recent_session_selected(self, event: tk.Event | None = None) -> None:
        selection = self.recent_sessions_listbox.curselection()
        if not selection:
            return
        selected_index = selection[0]
        if not (0 <= selected_index < len(self.recent_sessions_rows)):
            return
        if self.recent_sessions_rows[selected_index].get("kind") != "session":
            self.recent_sessions_listbox.selection_clear(selected_index)
            self._style_recent_sessions_listbox()
            return
        previous_index = self._last_recent_selected_index
        self._last_recent_selected_index = selected_index
        self._style_recent_sessions_listbox(indices=[selected_index, previous_index])
        self._update_recent_session_detail(selected_index)

    def _update_recent_session_detail(self, index: int) -> None:
        if not (0 <= index < len(self.recent_sessions_rows)):
            self._set_text_widget(self.recent_session_detail, "Select a recent session to preview it.")
            return
        row = self.recent_sessions_rows[index]
        session = row.get("session")
        if not isinstance(session, dict):
            self._set_text_widget(self.recent_session_detail, "Select a recent session to preview it.")
            return
        lines = [
            f"Session: {session.get('session_id')}",
            f"Title: {session.get('title') or 'Untitled'}",
            f"Mode: {session.get('mode') or 'unknown'}",
            f"When: {human_date_label(session.get('created_at')) or 'unknown'}",
            "",
            f"Last prompt: {session.get('prompt') or 'n/a'}",
        ]
        summary = str(session.get("summary") or "").strip()
        if summary:
            lines.extend(["", f"Last reply: {summary}"])
        self._set_text_widget(self.recent_session_detail, "\n".join(lines))

    def _rename_selected_recent_session(self) -> None:
        selection = self.recent_sessions_listbox.curselection()
        if not selection:
            return
        index = selection[0]
        if not (0 <= index < len(self.recent_sessions_rows)):
            return
        session = self.recent_sessions_rows[index].get("session")
        if not isinstance(session, dict):
            return
        session_id = str(session.get("session_id") or "").strip()
        if not session_id:
            return
        current_title = self._session_display_title(session)
        new_title = simpledialog.askstring(
            "Rename Chat",
            "Chat title:",
            initialvalue=current_title,
            parent=self.root,
        )
        if new_title is None:
            return
        if hasattr(self.controller, "rename_session"):
            self.controller.rename_session(session_id, title=new_title)
        self.recent_sessions_view_dirty = True
        self._refresh_recent_sessions_view()

    def _on_recent_sessions_hover(self, event: tk.Event) -> None:
        widget = self.recent_sessions_listbox
        if widget.size() <= 0:
            return
        index = int(widget.nearest(event.y))
        if self.hovered_recent_index == index:
            return
        previous_index = self.hovered_recent_index
        self.hovered_recent_index = index
        self._style_recent_sessions_listbox(indices=[previous_index, index])

    def _clear_recent_sessions_hover(self, event: tk.Event | None = None) -> None:
        previous_index = self.hovered_recent_index
        self.hovered_recent_index = None
        self._style_recent_sessions_listbox(indices=[previous_index])

    def _style_recent_sessions_listbox(self, *, indices: list[int | None] | None = None) -> None:
        if indices is None:
            targets = list(range(self.recent_sessions_listbox.size()))
        else:
            targets = sorted(
                {
                    int(index)
                    for index in indices
                    if isinstance(index, int) and 0 <= int(index) < self.recent_sessions_listbox.size()
                }
            )
        selected_indices = set(self.recent_sessions_listbox.curselection())
        for index in targets:
            row = self.recent_sessions_rows[index] if index < len(self.recent_sessions_rows) else {"kind": "session"}
            is_header = row.get("kind") == "header"
            selected = index in selected_indices and not is_header
            hovered = index == self.hovered_recent_index and not selected and not is_header
            bg = self.current_palette["panel_bg"] if is_header else (
                self.current_palette["user_bg"] if selected else (
                    self.current_palette["list_hover_bg"] if hovered else self.current_palette["input_bg"]
                )
            )
            fg = self.current_palette["text_muted"] if is_header else self.current_palette["text_primary"]
            try:
                self.recent_sessions_listbox.itemconfig(index, background=bg, foreground=fg)
            except tk.TclError:
                break

    def _open_selected_recent_session(self, event: tk.Event | None = None) -> None:
        selection = self.recent_sessions_listbox.curselection()
        if not selection:
            return
        index = selection[0]
        if not (0 <= index < len(self.recent_sessions_rows)):
            return
        session = self.recent_sessions_rows[index].get("session")
        if not isinstance(session, dict):
            return
        session_id = str(session.get("session_id") or "").strip()
        if not session_id:
            return
        self._load_session(session_id)

    @staticmethod
    def _session_title_fallback(session: dict[str, object] | None) -> str:
        if not isinstance(session, dict):
            return "Chat"
        return (
            str(session.get("title") or "").strip()
            or str(session.get("prompt") or "").strip()
            or str(session.get("summary") or "").strip()
            or "Chat"
        )

    @staticmethod
    def _session_open_title_fallback(session: dict[str, object] | None) -> str:
        if not isinstance(session, dict):
            return "Chat"
        title = str(session.get("title") or "").strip()
        prompt = str(session.get("prompt") or "").strip()
        summary = str(session.get("summary") or "").strip()
        if title and title.lower() != "chat":
            return title
        if prompt:
            return prompt
        if summary:
            return summary
        created = human_date_label(session.get("created_at"))
        return f"Chat • {created}" if created else "Chat"

    def _cached_session_match(self, session_id: str) -> tuple[dict[str, object] | None, str | None]:
        for surface, cache in (("recent", self.recent_sessions_cache), ("archived", self.archived_sessions_cache)):
            match = next(
                (item for item in cache if isinstance(item, dict) and str(item.get("session_id") or "") == session_id),
                None,
            )
            if isinstance(match, dict):
                return match, surface
        return None, None

    def _reject_nonrestorable_recent_session(self, session_id: str, *, source_surface: str | None) -> None:
        self._remove_session_from_caches(session_id)
        self.recent_sessions_view_dirty = True
        self.archived_sessions_view_dirty = True
        self.status_var.set("Saved chat is no longer available")
        self._debug_event(
            "saved_session_open_rejected",
            session_id=session_id,
            source_surface=source_surface or "unknown",
            reason="nonrestorable_recent_row",
        )
        if hasattr(self, "recent_list_inner"):
            self._render_recent_sessions_from_cache()
        if hasattr(self, "archived_list_inner"):
            self._render_archived_sessions_from_cache()
        fallback_view = source_surface if source_surface in {"recent", "archived"} else "recent"
        self._show_view(fallback_view)

    def _load_session(self, session_id: str) -> None:
        self._debug_event("saved_session_load_requested", session_id=session_id)
        try:
            report = self.controller.list_interactions(session_id=session_id)
            records = report.get("interaction_records", []) if isinstance(report, dict) else []
        except Exception:
            self.status_var.set(f"Couldn't load {session_id}")
            self._append_system_line("Saved chat could not be loaded.")
            self._show_view("chat")
            return
        try:
            profile = self.controller.get_session_profile(session_id).get("interaction_profile", {})
        except Exception:
            profile = {}
        recent_match, source_surface = self._cached_session_match(session_id)
        title = self._session_open_title_fallback(recent_match)
        ordered_records = sorted(
            [record for record in records if isinstance(record, dict)],
            key=lambda item: str(item.get("created_at") or ""),
        )
        restored_messages, had_render_issue, restored_user_count, restored_assistant_count = self._build_saved_chat_messages(
            ordered_records
        )
        self._debug_event(
            "saved_session_restore_result",
            session_id=session_id,
            interaction_record_count=len(ordered_records),
            restored_user_count=restored_user_count,
            restored_assistant_count=restored_assistant_count,
            restored_message_count=len(restored_messages),
            empty_restore=not bool(restored_messages),
            had_render_issue=had_render_issue,
            source_surface=source_surface or "unknown",
        )
        if not restored_messages and isinstance(recent_match, dict):
            self._reject_nonrestorable_recent_session(session_id, source_surface=source_surface)
            return
        self.chat_title_var.set(title)
        if "chat" in self.nav_buttons:
            self.nav_buttons["chat"].configure(text=title)
        self.session_id = session_id
        self._refresh_top_bar_title()
        self.mode_var.set(self.STYLE_TO_MODE.get(str(profile.get("interaction_style") or "default"), "Default"))
        self._set_chat_text("")
        for message in restored_messages:
            self._render_chat_message(
                message,
                store_message=True,
                auto_scroll=False,
                animate=False,
            )
        if restored_messages:
            self._scroll_restored_chat_to_reading_position()
        self._update_landing_state()
        if had_render_issue:
            self._append_system_line("Some saved messages could not be rendered, but the chat was recovered.")
        self.status_var.set(f"Loaded {session_id}")
        self.context_bar_var.set(build_context_bar(mode_label=self.mode_var.get(), prompt=""))
        self.memory_view_dirty = True
        self.recent_sessions_view_dirty = True
        self.archived_sessions_view_dirty = True
        self._show_view("chat")

    def _recent_assistant_texts(self) -> list[str]:
        return [
            message.text
            for message in self.messages
            if message.message_type == "assistant" and str(message.text).strip()
        ][-3:]

    def _message_font(self) -> tuple[str, int]:
        base_size = int(self.text_size_var.get() or 11)
        size = base_size - 1 if self.chat_density_var.get() == "Compact" else base_size
        return (self._resolved_font_family(), size)

    def _meta_font(self) -> tuple[str, int]:
        base_size = max(8, int(self.text_size_var.get() or 11) - 2)
        size = base_size - 1 if self.chat_density_var.get() == "Compact" else base_size
        return (self._resolved_font_family(), size)

    def _resolved_font_family(self) -> str:
        preferred = str(self.font_family_var.get() or "").strip() or "Segoe UI"
        try:
            families = set(tkfont.families(self.root))
        except tk.TclError:
            families = set()
        if preferred in families or not families:
            return preferred
        fallback_candidates = ("Segoe UI", "Arial", "Helvetica", "TkDefaultFont")
        for candidate in fallback_candidates:
            if candidate in families or not families:
                return candidate
        return preferred

    def _bubble_padx(self) -> int:
        return 10 if self.chat_density_var.get() == "Compact" else 12

    def _bubble_pady(self) -> int:
        return 6 if self.chat_density_var.get() == "Compact" else 8

    def _row_pady(self) -> int:
        return 4 if self.chat_density_var.get() == "Compact" else 6

    def _input_box_height(self) -> int:
        return 3 if self.chat_density_var.get() == "Compact" else 4

    @staticmethod
    def _timestamp_from_record(record: dict[str, object]) -> str | None:
        value = str(record.get("created_at") or "").strip()
        if not value:
            return None
        try:
            return message_timestamp(datetime.fromisoformat(value.replace("Z", "+00:00")))
        except ValueError:
            return None

    @staticmethod
    def _set_text_widget(widget: tk.Text, text: str) -> None:
        widget.configure(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert("1.0", text)
        widget.configure(state=tk.DISABLED)

    def _set_memory_preview_empty(self, text: str) -> None:
        self.memory_preview.configure(state=tk.NORMAL)
        self.memory_preview.delete("1.0", tk.END)
        self.memory_preview.tag_configure("memory_empty_center", justify="center")
        self.memory_preview.insert("1.0", text, ("memory_empty_center",))
        self.memory_preview.tag_add("memory_empty_center", "1.0", "end-1c")
        self.memory_preview.configure(state=tk.DISABLED)

    def _render_memory_surface_first_paint(self, *, archived: bool) -> None:
        self._render_memory_surface_resolved_state(
            archived=archived,
            render_mode="render_from_cached_slice",
            event_name="memory_surface_first_paint_state",
        )

    def _resolve_memory_surface_visible_state(self, *, archived: bool) -> dict[str, str]:
        view_name = "archived_memory" if archived else "memory"
        if not self._view_capability_available(view_name):
            reason = self._view_capability_reason(view_name)
            return {
                "state": "unavailable",
                "source": "unavailable",
                "preview_text": reason,
                "list_text": reason,
            }
        entries = self.archived_memory_entries if archived else self.memory_entries
        fetch_in_flight = self.archived_memory_fetch_in_flight if archived else self.memory_fetch_in_flight
        is_dirty = self.archived_memory_view_dirty if archived else self.memory_view_dirty
        if entries:
            return {
                "state": "cache_rows",
                "source": "memory_cache",
                "preview_text": "",
                "list_text": "",
            }
        if fetch_in_flight or is_dirty:
            loading_text = "Loading archived memory..." if archived else "Loading memory..."
            return {
                "state": "loading",
                "source": "fetch_pending",
                "preview_text": loading_text,
                "list_text": loading_text,
            }
        empty_text = "No archived memory yet." if archived else empty_state_text("memory")
        return {
            "state": "empty",
            "source": "empty",
            "preview_text": empty_text,
            "list_text": empty_text,
        }

    def _render_memory_surface_resolved_state(
        self,
        *,
        archived: bool,
        render_mode: str,
        event_name: str = "memory_surface_state_resolved",
    ) -> str:
        state = self._resolve_memory_surface_visible_state(archived=archived)
        view_name = "archived_memory" if archived else "memory"
        self._debug_event(
            event_name,
            view=view_name,
            state=state["state"],
            source=state["source"],
            runtime_phase=self._desktop_capability_state.phase,
        )
        if state["state"] == "cache_rows":
            inner = self.archived_memory_list_inner if archived else self.memory_list_inner
            if inner.winfo_children():
                self._debug_event(
                    "memory_placeholder_clear_before_render",
                    view=view_name,
                    child_count=len(inner.winfo_children()),
                )
            self._render_memory_entries_from_cache(
                archived=archived,
                render_mode=render_mode,
            )
            return state["state"]
        self._render_memory_surface_state(
            archived=archived,
            preview_text=state["preview_text"],
            list_text=state["list_text"],
        )
        return state["state"]

    def _render_memory_surface_state(
        self,
        *,
        archived: bool,
        preview_text: str,
        list_text: str,
    ) -> None:
        inner = self.archived_memory_list_inner if archived else self.memory_list_inner
        entries = self.archived_memory_entries if archived else self.memory_entries
        if entries and ("Loading" in preview_text or "Loading" in list_text):
            self._debug_event(
                "memory_placeholder_persisted_after_usable_cache",
                view="archived_memory" if archived else "memory",
                entry_count=len(entries),
            )
        button_attr = "archived_memory_load_more_button" if archived else "memory_load_more_button"
        existing_button = getattr(self, button_attr, None)
        if existing_button is not None and existing_button.winfo_exists():
            existing_button.destroy()
        setattr(self, button_attr, None)
        for child in inner.winfo_children():
            child.destroy()
        if archived:
            self.archived_memory_rendered_count = 0
            self._set_text_widget(self.archived_memory_preview, preview_text)
            self._set_active_browser_hover_row("archived_memory", None)
        else:
            self.memory_rendered_count = 0
            self._set_memory_preview_empty(preview_text)
            self._set_active_browser_hover_row("memory", None)
        tk.Label(
            inner,
            text=list_text,
            anchor="w",
            justify=tk.LEFT,
            bg=self.current_palette["panel_bg"],
            fg=self.current_palette["text_muted"],
        ).pack(fill=tk.X)

    def _configure_readonly_text(
        self,
        widget: tk.Text,
        *,
        text: str,
        bubble_bg: str,
        text_fg: str,
    ) -> None:
        widget._lumen_text = text  # type: ignore[attr-defined]
        widget._lumen_bubble_bg = bubble_bg  # type: ignore[attr-defined]
        widget._lumen_text_fg = text_fg  # type: ignore[attr-defined]
        widget.configure(
            bg=bubble_bg,
            fg=text_fg,
            insertbackground=text_fg,
            selectbackground=self.current_palette["user_bg"],
            selectforeground=self.current_palette["text_primary"],
            relief=tk.FLAT,
            bd=0,
            highlightthickness=0,
            highlightbackground=bubble_bg,
            highlightcolor=bubble_bg,
            height=self._estimate_text_widget_height(text=text),
            width=self._estimated_text_widget_width(),
        )
        widget.delete("1.0", tk.END)
        widget.insert("1.0", text)
        widget.configure(state=tk.DISABLED)
        self._bind_text_context_menu(widget, editable=False)
        widget.bind("<Button-1>", lambda event, text_widget=widget: text_widget.focus_set(), add="+")

    def _refresh_readonly_text_widget(self, widget: tk.Text) -> None:
        widget.configure(state=tk.NORMAL)
        widget.configure(
            font=self._message_font(),
            height=self._estimate_text_widget_height(text=str(getattr(widget, "_lumen_text", "") or "")),
            width=self._estimated_text_widget_width(),
            bg=str(getattr(widget, "_lumen_bubble_bg", self.current_palette["assistant_bg"])),
            fg=str(getattr(widget, "_lumen_text_fg", self.current_palette["text_primary"])),
            relief=tk.FLAT,
            bd=0,
            highlightthickness=0,
            highlightbackground=str(getattr(widget, "_lumen_bubble_bg", self.current_palette["assistant_bg"])),
            highlightcolor=str(getattr(widget, "_lumen_bubble_bg", self.current_palette["assistant_bg"])),
        )
        widget.configure(state=tk.DISABLED)

    def _estimated_text_widget_width(self) -> int:
        if not hasattr(self, "chat_canvas"):
            return 64
        return max(28, min(84, self._bubble_wraplength() // 7))

    def _estimate_text_widget_height(self, *, text: str) -> int:
        width = self._estimated_text_widget_width()
        segments = str(text or "").splitlines() or [""]
        line_count = 0
        for segment in segments:
            normalized = max(len(segment), 1)
            line_count += max(1, (normalized // max(width, 1)) + (1 if normalized % max(width, 1) else 0))
        return max(1, min(160, line_count))

    def _bind_text_context_menu(self, widget: tk.Text, *, editable: bool) -> None:
        def _show_menu(event: tk.Event) -> str:
            if not editable:
                widget.focus_set()
            self._set_active_scrollable(widget)
            menu = tk.Menu(self.root, tearoff=0)
            if editable:
                menu.add_command(label="Cut", command=lambda: self._cut_text(widget))
                menu.add_command(label="Copy", command=lambda: self._copy_selected_text(widget))
                menu.add_command(label="Paste", command=lambda: self._paste_text(widget))
            else:
                menu.add_command(label="Copy", command=lambda: self._copy_selected_text(widget))
            menu.add_separator()
            menu.add_command(
                label="Select All",
                command=lambda: self._select_all_text(widget),
            )
            menu.tk_popup(event.x_root, event.y_root)
            return "break"

        widget.bind("<Button-3>", _show_menu)
        widget.bind("<Control-Button-1>", _show_menu)
        widget.bind("<Control-a>", lambda event, text_widget=widget: self._select_all_text(text_widget))
        widget.bind("<Control-c>", lambda event, text_widget=widget: self._copy_selected_text(text_widget))
        widget.bind("<Control-v>", lambda event, text_widget=widget: self._paste_text(text_widget) if editable else "break")
        widget.bind("<Control-x>", lambda event, text_widget=widget: self._cut_text(text_widget) if editable else "break")
        widget.bind("<Button-1>", lambda event, scroll_target=widget: self._set_active_scrollable(scroll_target), add="+")
        widget.bind("<FocusIn>", lambda event, scroll_target=widget: self._set_active_scrollable(scroll_target), add="+")

    @staticmethod
    def _select_all_text(widget: tk.Text) -> str:
        widget.tag_add(tk.SEL, "1.0", tk.END)
        widget.mark_set(tk.INSERT, "1.0")
        widget.see(tk.INSERT)
        return "break"

    @staticmethod
    def _selected_text(widget: tk.Text) -> str:
        try:
            return str(widget.get(tk.SEL_FIRST, tk.SEL_LAST))
        except tk.TclError:
            return ""

    def _copy_selected_text(self, widget: tk.Text) -> str:
        selected = self._selected_text(widget)
        if not selected and str(widget.cget("state")) == tk.DISABLED:
            selected = str(widget.get("1.0", "end-1c"))
        if not selected:
            return "break"
        self._write_clipboard(selected)
        return "break"

    def _paste_text(self, widget: tk.Text) -> str:
        if str(widget.cget("state")) == tk.DISABLED:
            return "break"
        text = self._read_clipboard()
        if not text:
            return "break"
        if widget is self.input_box and self.placeholder_active:
            self._hide_input_placeholder()
        try:
            if widget.tag_ranges(tk.SEL):
                widget.delete(tk.SEL_FIRST, tk.SEL_LAST)
            widget.insert(tk.INSERT, text)
            widget.see(tk.INSERT)
            widget.focus_set()
        except tk.TclError:
            try:
                widget.event_generate("<<Paste>>")
            except tk.TclError:
                return "break"
        return "break"

    def _cut_text(self, widget: tk.Text) -> str:
        if str(widget.cget("state")) == tk.DISABLED:
            return "break"
        selected = self._selected_text(widget)
        if not selected:
            return "break"
        self._write_clipboard(selected)
        try:
            widget.delete(tk.SEL_FIRST, tk.SEL_LAST)
        except tk.TclError:
            return "break"
        return "break"

    def _write_clipboard(self, text: str) -> None:
        self.root.clipboard_clear()
        self.root.clipboard_append(text)

    def _read_clipboard(self) -> str:
        readers = (
            lambda: self.root.clipboard_get(),
            lambda: self.root.selection_get(selection="CLIPBOARD"),
            lambda: str(self.root.tk.call("clipboard", "get")),
        )
        for reader in readers:
            try:
                text = str(reader() or "")
            except (tk.TclError, RuntimeError):
                continue
            if text:
                return text
        return ""

    def _show_input_placeholder(self) -> None:
        if self.placeholder_active or self._input_has_user_text():
            return
        self.placeholder_active = True
        self.input_box.configure(fg=self.current_palette["text_muted"])
        self.input_box.delete("1.0", tk.END)
        self.input_box.insert("1.0", "Message Lumen...")

    def _hide_input_placeholder(self) -> None:
        if not self.placeholder_active:
            return
        self.placeholder_active = False
        self.input_box.configure(fg=self.current_palette["text_primary"])
        self.input_box.delete("1.0", tk.END)

    def _input_has_user_text(self) -> bool:
        if self.placeholder_active:
            return False
        return bool(self.input_box.get("1.0", tk.END).strip())

    def _input_value(self) -> str:
        if self.placeholder_active:
            return ""
        return self.input_box.get("1.0", tk.END).strip()

    def _on_input_focus_in(self, event: tk.Event | None = None) -> None:
        self.input_frame.configure(highlightbackground=self.current_palette["input_focus_border"])
        self._hide_input_placeholder()

    def _on_input_focus_out(self, event: tk.Event | None = None) -> None:
        self.input_frame.configure(highlightbackground=self.current_palette["input_border"])
        if not self._input_has_user_text():
            self._show_input_placeholder()

    def _on_input_pointer_down(self, event: tk.Event | None = None) -> None:
        if self.placeholder_active:
            self._hide_input_placeholder()

    def _on_input_key_press(self, event: tk.Event | None = None) -> str | None:
        if not self.placeholder_active:
            return None
        if event is None:
            self._hide_input_placeholder()
            return None
        keysym = str(getattr(event, "keysym", "") or "")
        if keysym in {"Shift_L", "Shift_R", "Control_L", "Control_R", "Alt_L", "Alt_R"}:
            return None
        self._hide_input_placeholder()
        if keysym in {"BackSpace", "Delete"}:
            return "break"
        return None

    def _on_input_key_release(self, event: tk.Event | None = None) -> None:
        if self.placeholder_active:
            return

    @staticmethod
    def _scroll_units(event: tk.Event) -> int:
        delta = getattr(event, "delta", 0)
        if delta:
            return -1 * int(delta / 120) if delta % 120 == 0 else (-1 if delta > 0 else 1)
        num = getattr(event, "num", 0)
        if num == 4:
            return -1
        if num == 5:
            return 1
        return 0

    def _bind_mousewheel(self, widget: tk.Widget, scrollable: tk.Widget) -> None:
        widget.bind(
            "<Enter>",
            lambda event, scroll_target=scrollable: self._set_active_scrollable(scroll_target),
        )
        widget.bind(
            "<Button-1>",
            lambda event, scroll_target=scrollable: self._set_active_scrollable(scroll_target),
            add="+",
        )
        widget.bind(
            "<FocusIn>",
            lambda event, scroll_target=scrollable: self._set_active_scrollable(scroll_target),
            add="+",
        )

    def _set_active_scrollable(self, scrollable: tk.Widget | None) -> None:
        self.active_scrollable = scrollable

    def _bind_global_mousewheel(self) -> None:
        if self._mousewheel_bound_globally:
            return

        def _dispatch(event: tk.Event) -> str | None:
            scrollable = self.active_scrollable
            if scrollable is None:
                return None
            units = self._scroll_units(event)
            if not units:
                return None
            self._queue_mousewheel_scroll(scrollable, units)
            return "break"

        self.root.bind_all("<MouseWheel>", _dispatch, add="+")
        self.root.bind_all("<Button-4>", _dispatch, add="+")
        self.root.bind_all("<Button-5>", _dispatch, add="+")
        self._mousewheel_bound_globally = True

    def _queue_mousewheel_scroll(self, scrollable: tk.Widget, units: int) -> None:
        if not units:
            return
        scrollable_id = id(scrollable)
        pending_units = int(getattr(scrollable, "_lumen_pending_scroll_units", 0) or 0) + int(units)
        setattr(scrollable, "_lumen_pending_scroll_units", pending_units)
        self._debug_event(
            "mousewheel_scroll_request",
            surface=str(getattr(scrollable, "_lumen_layout_surface", getattr(scrollable, "_name", "widget")) or "widget"),
            units=pending_units,
        )
        if scrollable_id in self._mousewheel_flush_jobs:
            return

        def _flush(target: tk.Widget = scrollable, target_id: int = scrollable_id) -> None:
            self._mousewheel_flush_jobs.pop(target_id, None)
            total_units = int(getattr(target, "_lumen_pending_scroll_units", 0) or 0)
            setattr(target, "_lumen_pending_scroll_units", 0)
            if not total_units:
                return
            try:
                if not target.winfo_exists():
                    return
                target.yview_scroll(total_units, "units")
                self._debug_event(
                    "mousewheel_scroll_flush",
                    surface=str(getattr(target, "_lumen_layout_surface", getattr(target, "_name", "widget")) or "widget"),
                    units=total_units,
                )
                surface = str(getattr(target, "_lumen_layout_surface", "") or "").strip()
                if surface in getattr(self, "_active_browser_hover_rows", {}):
                    self._clear_stale_browser_hover(surface)
            except tk.TclError:
                return

        self._mousewheel_flush_jobs[scrollable_id] = self.root.after_idle(_flush)
