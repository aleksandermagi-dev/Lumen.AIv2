from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import os
from pathlib import Path


@dataclass
class DebugTraceSession:
    log_path: Path | None = None
    session_label: str = ""

    @classmethod
    def create(cls, *, data_root: Path, enabled: bool, prefix: str = "ui_debug") -> "DebugTraceSession":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        session_label = f"{timestamp}_pid{os.getpid()}"
        log_path = data_root / "desktop_ui" / f"{prefix}_{session_label}.log"
        if not enabled:
            return cls(log_path=log_path, session_label=session_label)
        return cls(log_path=log_path, session_label=session_label)


@dataclass
class ShellTransitionState:
    pending_hotbar_open_state: bool | None = None
    pending_hotbar_refresh_target: str | None = None
    deferred_view_refresh_target: str | None = None
    pending_view_name: str | None = None
    hotbar_navigation_generation: int = 0
    pending_view_generation: int = 0
    pending_refresh_generation: int = 0
    theme_apply_requested: bool = False
    theme_apply_in_progress: bool = False
    mode_apply_in_progress: bool = False
    debug_session: DebugTraceSession = field(default_factory=DebugTraceSession)

    def set_pending_hotbar_open_state(self, target_state: bool | None) -> None:
        self.pending_hotbar_open_state = target_state

    def set_pending_hotbar_refresh_target(self, view_name: str | None, *, generation: int | None = None) -> None:
        self.pending_hotbar_refresh_target = view_name
        if generation is not None:
            self.pending_refresh_generation = int(generation)
        elif view_name is None:
            self.pending_refresh_generation = 0

    def set_deferred_view_refresh_target(self, view_name: str | None, *, generation: int | None = None) -> None:
        self.deferred_view_refresh_target = view_name
        if generation is not None:
            self.pending_refresh_generation = int(generation)
        elif view_name is None:
            self.pending_refresh_generation = 0

    def set_pending_view_name(self, view_name: str | None, *, generation: int | None = None) -> None:
        self.pending_view_name = view_name
        if generation is not None:
            self.pending_view_generation = int(generation)
        elif view_name is None:
            self.pending_view_generation = 0

    def begin_hotbar_navigation(self, view_name: str) -> int:
        self.hotbar_navigation_generation += 1
        generation = self.hotbar_navigation_generation
        self.pending_view_name = view_name
        self.pending_view_generation = generation
        self.pending_hotbar_refresh_target = None
        self.deferred_view_refresh_target = None
        self.pending_refresh_generation = generation
        return generation

    def consume_pending_view_name(self) -> tuple[str | None, int]:
        pending_view = self.pending_view_name
        generation = self.pending_view_generation
        self.pending_view_name = None
        self.pending_view_generation = 0
        return pending_view, generation

    def consume_refresh_target(self) -> tuple[str | None, int]:
        refresh_target = self.pending_hotbar_refresh_target or self.deferred_view_refresh_target
        generation = self.pending_refresh_generation
        self.pending_hotbar_refresh_target = None
        if refresh_target:
            self.deferred_view_refresh_target = None
        self.pending_refresh_generation = 0
        return refresh_target, generation

    def set_theme_apply_requested(self, requested: bool) -> None:
        self.theme_apply_requested = bool(requested)

    def set_theme_apply_in_progress(self, in_progress: bool) -> None:
        self.theme_apply_in_progress = bool(in_progress)

    def begin_mode_apply(self) -> bool:
        if self.mode_apply_in_progress:
            return False
        self.mode_apply_in_progress = True
        return True

    def finish_mode_apply(self) -> None:
        self.mode_apply_in_progress = False


@dataclass
class DesktopCapabilityState:
    phase: str = "booting"
    shell_ready: bool = False
    chat_send_ready: bool = False
    history_surfaces_ready: bool = False
    memory_surfaces_ready: bool = False
    settings_ready: bool = True
    degraded_warning: bool = False
    missing_bundles: tuple[str, ...] = ()
    missing_resources: tuple[str, ...] = ()
    capability_count: int = 0
    surface_reasons: dict[str, str] = field(default_factory=dict)
    surface_ready_sources: dict[str, str] = field(default_factory=dict)

    @classmethod
    def booting(cls) -> "DesktopCapabilityState":
        reason = "Lumen is still checking runtime capabilities."
        return cls(
            phase="booting",
            surface_reasons={
                "chat": reason,
                "recent": reason,
                "archived": reason,
                "memory": reason,
                "archived_memory": reason,
                "settings": reason,
            },
        )

    @classmethod
    def from_runtime(
        cls,
        *,
        missing_bundles: list[str] | tuple[str, ...],
        missing_resources: list[str] | tuple[str, ...],
        capabilities: dict[str, object] | None,
        surface_runtime_ready: dict[str, bool] | None = None,
    ) -> "DesktopCapabilityState":
        normalized_missing_bundles = tuple(sorted({str(item).strip() for item in missing_bundles if str(item).strip()}))
        normalized_missing_resources = tuple(
            sorted({str(item).strip() for item in missing_resources if str(item).strip()})
        )
        capability_count = len(capabilities or {})
        runtime_ready = {
            str(key).strip().lower(): bool(value)
            for key, value in (surface_runtime_ready or {}).items()
            if str(key).strip()
        }
        surface_ready_sources: dict[str, str] = {}
        shell_ready = not normalized_missing_resources
        if "chat" in runtime_ready:
            chat_send_ready = shell_ready and bool(runtime_ready.get("chat"))
            surface_ready_sources["chat"] = "explicit_runtime_ready"
        else:
            chat_send_ready = shell_ready and capability_count > 0
            surface_ready_sources["chat"] = "capability_count" if chat_send_ready else "no_capabilities"
        history_runtime_keys = [key for key in ("history", "recent", "archived") if key in runtime_ready]
        if history_runtime_keys:
            history_surfaces_ready = shell_ready and all(bool(runtime_ready.get(key)) for key in history_runtime_keys)
            surface_ready_sources["recent"] = "explicit_runtime_ready"
            surface_ready_sources["archived"] = "explicit_runtime_ready"
        else:
            history_surfaces_ready = shell_ready
            history_source = "shell_ready" if history_surfaces_ready else "missing_resources"
            surface_ready_sources["recent"] = history_source
            surface_ready_sources["archived"] = history_source
        if "memory" in runtime_ready:
            memory_surfaces_ready = shell_ready and bool(runtime_ready.get("memory"))
            surface_ready_sources["memory"] = "explicit_runtime_ready"
            surface_ready_sources["archived_memory"] = "explicit_runtime_ready"
        elif "memory" in normalized_missing_bundles:
            # In packaged/runtime doctor reports, "memory" can be flagged as missing even
            # when the desktop runtime still exposes usable memory methods. If the shell is
            # otherwise healthy and the runtime has capabilities, keep memory enabled unless
            # an explicit runtime readiness signal says otherwise.
            memory_surfaces_ready = shell_ready and capability_count > 0
            memory_source = "capability_fallback" if memory_surfaces_ready else "missing_bundle"
            surface_ready_sources["memory"] = memory_source
            surface_ready_sources["archived_memory"] = memory_source
        else:
            memory_surfaces_ready = shell_ready
            memory_source = "shell_ready" if memory_surfaces_ready else "missing_resources"
            surface_ready_sources["memory"] = memory_source
            surface_ready_sources["archived_memory"] = memory_source
        degraded_warning = bool(normalized_missing_bundles or normalized_missing_resources)
        phase = "degraded" if degraded_warning else "ready"
        surface_reasons: dict[str, str] = {}
        surface_ready_sources["settings"] = "always_enabled"

        if normalized_missing_resources:
            shared_reason = (
                "This runtime is missing required desktop resources, so Lumen is keeping some surfaces disabled."
            )
            for view_name in ("chat", "recent", "archived", "memory", "archived_memory"):
                surface_reasons[view_name] = shared_reason
        elif capability_count <= 0:
            surface_reasons["chat"] = "No app capabilities are available in this runtime right now."

        if "memory" in normalized_missing_bundles and not memory_surfaces_ready:
            memory_reason = "Memory tools are unavailable in this runtime right now."
            surface_reasons["memory"] = memory_reason
            surface_reasons["archived_memory"] = memory_reason

        return cls(
            phase=phase,
            shell_ready=shell_ready,
            chat_send_ready=chat_send_ready,
            history_surfaces_ready=history_surfaces_ready,
            memory_surfaces_ready=memory_surfaces_ready,
            settings_ready=True,
            degraded_warning=degraded_warning,
            missing_bundles=normalized_missing_bundles,
            missing_resources=normalized_missing_resources,
            capability_count=capability_count,
            surface_reasons=surface_reasons,
            surface_ready_sources=surface_ready_sources,
        )

    def is_view_enabled(self, view_name: str) -> bool:
        normalized = str(view_name or "").strip().lower()
        if normalized == "quit":
            return True
        if self.phase == "booting":
            return normalized == "chat"
        if normalized == "chat":
            return self.chat_send_ready
        if normalized in {"recent", "archived"}:
            return self.history_surfaces_ready
        if normalized in {"memory", "archived_memory"}:
            return self.memory_surfaces_ready
        if normalized == "settings":
            return self.settings_ready
        return self.shell_ready

    def reason_for_view(self, view_name: str) -> str:
        normalized = str(view_name or "").strip().lower()
        reason = str(self.surface_reasons.get(normalized) or "").strip()
        if reason:
            return reason
        if self.phase == "booting":
            return "Lumen is still checking runtime capabilities."
        if normalized == "chat" and not self.chat_send_ready:
            return "Chat is unavailable in this runtime right now."
        if normalized in {"recent", "archived"} and not self.history_surfaces_ready:
            return "Conversation history is unavailable in this runtime right now."
        if normalized in {"memory", "archived_memory"} and not self.memory_surfaces_ready:
            return "Memory is unavailable in this runtime right now."
        if normalized == "settings" and not self.settings_ready:
            return "Settings are unavailable in this runtime right now."
        return "That surface is unavailable in this runtime right now."

    def summary_payload(self) -> dict[str, object]:
        disabled_surfaces = [
            name
            for name in ("chat", "recent", "archived", "memory", "archived_memory", "settings")
            if not self.is_view_enabled(name)
        ]
        enabled_surfaces = [
            name
            for name in ("chat", "recent", "archived", "memory", "archived_memory", "settings")
            if self.is_view_enabled(name)
        ]
        return {
            "phase": self.phase,
            "degraded_reason": "missing_resources" if self.missing_resources else ("missing_bundles" if self.missing_bundles else ""),
            "shell_ready": self.shell_ready,
            "chat_send_ready": self.chat_send_ready,
            "history_surfaces_ready": self.history_surfaces_ready,
            "memory_surfaces_ready": self.memory_surfaces_ready,
            "settings_ready": self.settings_ready,
            "degraded_warning": self.degraded_warning,
            "missing_bundles": list(self.missing_bundles),
            "missing_resources": list(self.missing_resources),
            "capability_count": self.capability_count,
            "enabled_surfaces": enabled_surfaces,
            "disabled_surfaces": disabled_surfaces,
            "surface_ready_sources": dict(self.surface_ready_sources),
        }
