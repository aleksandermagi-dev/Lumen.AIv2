from __future__ import annotations

from dataclasses import dataclass

from lumen.desktop.chat_ui_support import palette_from_theme, resolve_theme_tokens, validate_palette


@dataclass(frozen=True)
class DesktopStartupSnapshot:
    theme_name: str
    theme_tokens: dict[str, str]
    palette: dict[str, str]
    initial_view: str = "chat"


def build_startup_snapshot(
    *,
    theme_name: str,
    custom_theme_name: str,
    custom_accent_color: str | None,
) -> DesktopStartupSnapshot:
    normalized_theme = str(theme_name or "Dark").strip().lower() or "dark"
    accent = custom_accent_color if normalized_theme == "custom" and str(custom_theme_name or "").strip() == "Color Wheel" else None
    theme_tokens = resolve_theme_tokens(normalized_theme, custom_accent_hex=accent)
    palette = palette_from_theme(theme_tokens)
    validate_palette(palette)
    return DesktopStartupSnapshot(
        theme_name=normalized_theme,
        theme_tokens=theme_tokens,
        palette=palette,
    )
