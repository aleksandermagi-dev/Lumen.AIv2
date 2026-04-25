from __future__ import annotations

from dataclasses import dataclass
import tkinter as tk

from lumen.desktop.chat_ui_support import nav_button_style


@dataclass(frozen=True)
class DesktopControlAvailability:
    shell_interactive: bool
    chat_ready: bool
    nav_enabled: bool
    selector_state: str
    chat_state: str
    top_level_state: str


def resolve_control_availability(
    *,
    shell_ready: bool,
    pending: bool,
    capability_phase: str,
    chat_send_ready: bool,
) -> DesktopControlAvailability:
    shell_interactive = bool(shell_ready) and str(capability_phase or "").strip().lower() != "booting"
    chat_ready = shell_interactive and bool(chat_send_ready)
    top_level_interactive = bool(shell_ready) and not pending
    return DesktopControlAvailability(
        shell_interactive=shell_interactive,
        chat_ready=chat_ready,
        nav_enabled=shell_interactive and not pending,
        selector_state="readonly" if chat_ready and not pending else "disabled",
        chat_state=tk.NORMAL if chat_ready and not pending else tk.DISABLED,
        top_level_state=tk.NORMAL if top_level_interactive else tk.DISABLED,
    )


def resolve_nav_button_visual(
    *,
    name: str,
    current_view: str,
    hovered_nav: str | None,
    enabled: bool,
    palette: dict[str, str],
    use_accented_hover: bool,
) -> dict[str, str | bool]:
    nav_palette = dict(palette)
    if use_accented_hover:
        nav_palette["nav_hover_bg"] = palette["nav_active_bg"]
    active = name == current_view
    hovered = hovered_nav == name and not active and enabled
    nav_style = nav_button_style(active=active, hovered=hovered, palette=nav_palette)
    return {
        "bg": nav_style["bg"],
        "fg": nav_style["fg"] if enabled else palette["text_muted"],
        "highlightbackground": nav_style["highlightbackground"],
        "accent_bg": palette["nav_active_border"] if active else nav_style["bg"],
        "activebackground": nav_palette["nav_hover_bg"],
        "activeforeground": palette["text_primary"],
        "disabledforeground": palette["text_muted"],
        "active": active,
        "hovered": hovered,
    }


def resolve_input_palette(*, palette: dict[str, str], placeholder_active: bool) -> dict[str, str]:
    return {
        "bg": palette["panel_bg"] if placeholder_active else palette["input_bg"],
        "fg": palette["text_muted"] if placeholder_active else palette["text_primary"],
        "insertbackground": palette["text_primary"],
    }


def resolve_composer_button_palette(*, palette: dict[str, str], primary: bool) -> dict[str, str]:
    return {
        "bg": palette["panel_bg"] if primary else palette["panel_alt_bg"],
        "fg": palette["nav_active_border"] if primary else palette["text_primary"],
        "activebackground": palette["button_hover_bg"],
        "activeforeground": palette["nav_active_border"] if primary else palette["text_primary"],
        "disabledforeground": palette["nav_active_border"] if primary else palette["text_secondary"],
    }


def resolve_load_more_palette(*, palette: dict[str, str]) -> dict[str, str]:
    accent_bg = palette["nav_active_border"]
    return {
        "bg": accent_bg,
        "fg": palette["text_primary"],
        "activebackground": accent_bg,
        "activeforeground": palette["text_primary"],
        "disabledforeground": palette["text_muted"],
    }


def resolve_top_icon_palette(
    *,
    palette: dict[str, str],
    theme_name: str,
    hovered: bool,
    primary: bool,
    enabled: bool,
) -> dict[str, str]:
    normalized = str(theme_name or "").strip().lower()
    light_theme = normalized == "light"
    if light_theme:
        idle_fg = palette["nav_active_border"]
        disabled_fg = palette["nav_active_border"] if primary else palette["text_muted"]
        active_bg = palette["nav_hover_bg"]
        active_fg = palette["nav_active_border"]
        return {
            "bg": active_bg if hovered and enabled else palette["app_bg"],
            "fg": active_fg if hovered and enabled else (idle_fg if enabled else disabled_fg),
            "activebackground": active_bg,
            "activeforeground": active_fg,
            "disabledforeground": disabled_fg,
        }
    accent_fg = palette["nav_active_border"]
    active_bg = palette["button_hover_bg"]
    return {
        "bg": active_bg if hovered and enabled else palette["app_bg"],
        "fg": accent_fg,
        "activebackground": active_bg,
        "activeforeground": accent_fg,
        "disabledforeground": accent_fg,
    }
