from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any


THEME_TOKEN_KEYS: tuple[str, ...] = (
    "background",
    "sidebarBackground",
    "surface",
    "surfaceAlt",
    "border",
    "divider",
    "assistantBubble",
    "assistantBorder",
    "userBubble",
    "userBorder",
    "pendingBubble",
    "pendingBorder",
    "systemBubble",
    "systemBorder",
    "textPrimary",
    "textSecondary",
    "textMuted",
    "inputBackground",
    "inputBorder",
    "inputFocusBorder",
    "navHover",
    "navActive",
    "accent",
    "accentHover",
    "listHover",
    "chipBackground",
    "chipHover",
    "chipActive",
    "chipBorder",
)

PALETTE_KEYS: tuple[str, ...] = (
    "app_bg",
    "sidebar_bg",
    "sidebar_border",
    "panel_bg",
    "panel_alt_bg",
    "panel_border",
    "panel_divider",
    "assistant_bg",
    "assistant_border",
    "user_bg",
    "user_border",
    "pending_bg",
    "pending_border",
    "system_bg",
    "system_border",
    "text_primary",
    "text_secondary",
    "text_muted",
    "input_bg",
    "input_border",
    "input_focus_border",
    "nav_hover_bg",
    "nav_active_bg",
    "nav_active_border",
    "button_hover_bg",
    "list_hover_bg",
    "chip_bg",
    "chip_hover_bg",
    "chip_active_bg",
    "chip_border",
)


def _palette_from_theme(theme: dict[str, str]) -> dict[str, str]:
    return {
        "app_bg": theme["background"],
        "sidebar_bg": theme["sidebarBackground"],
        "sidebar_border": theme["border"],
        "panel_bg": theme["surface"],
        "panel_alt_bg": theme["surfaceAlt"],
        "panel_border": theme["border"],
        "panel_divider": theme["divider"],
        "assistant_bg": theme["assistantBubble"],
        "assistant_border": theme["assistantBorder"],
        "user_bg": theme["userBubble"],
        "user_border": theme["userBorder"],
        "pending_bg": theme["pendingBubble"],
        "pending_border": theme["pendingBorder"],
        "system_bg": theme["systemBubble"],
        "system_border": theme["systemBorder"],
        "text_primary": theme["textPrimary"],
        "text_secondary": theme["textSecondary"],
        "text_muted": theme["textMuted"],
        "input_bg": theme["inputBackground"],
        "input_border": theme["inputBorder"],
        "input_focus_border": theme["inputFocusBorder"],
        "nav_hover_bg": theme["navHover"],
        "nav_active_bg": theme["navActive"],
        "nav_active_border": theme["accent"],
        "button_hover_bg": theme["accentHover"],
        "list_hover_bg": theme["listHover"],
        "chip_bg": theme["chipBackground"],
        "chip_hover_bg": theme["chipHover"],
        "chip_active_bg": theme["chipActive"],
        "chip_border": theme["chipBorder"],
    }


darkTheme: dict[str, str] = {
    "background": "#0f1117",
    "sidebarBackground": "#0c0e13",
    "surface": "#12151c",
    "surfaceAlt": "#181c26",
    "border": "#232938",
    "divider": "#202636",
    "assistantBubble": "#181b25",
    "assistantBorder": "#2d3550",
    "userBubble": "#1d1930",
    "userBorder": "#57428f",
    "pendingBubble": "#161924",
    "pendingBorder": "#554089",
    "systemBubble": "#141720",
    "systemBorder": "#262c3d",
    "textPrimary": "#e6edf3",
    "textSecondary": "#9aa4b2",
    "textMuted": "#7f8a99",
    "inputBackground": "#10131a",
    "inputBorder": "#303850",
    "inputFocusBorder": "#9c67ff",
    "navHover": "#151924",
    "navActive": "#1e2130",
    "accent": "#a566ff",
    "accentHover": "#252c3d",
    "listHover": "#171c27",
    "chipBackground": "#141823",
    "chipHover": "#1c2230",
    "chipActive": "#2c2545",
    "chipBorder": "#43386d",
}

lumenPurpleTheme: dict[str, str] = {
    "background": "#120f1d",
    "sidebarBackground": "#100c18",
    "surface": "#1a1527",
    "surfaceAlt": "#231a37",
    "border": "#3e2c63",
    "divider": "#33284f",
    "assistantBubble": "#211835",
    "assistantBorder": "#6e4ec0",
    "userBubble": "#2a1f46",
    "userBorder": "#9b67ff",
    "pendingBubble": "#1b152b",
    "pendingBorder": "#a566ff",
    "systemBubble": "#171222",
    "systemBorder": "#433065",
    "textPrimary": "#f1eaff",
    "textSecondary": "#b7a9d1",
    "textMuted": "#9586b0",
    "inputBackground": "#151120",
    "inputBorder": "#533a87",
    "inputFocusBorder": "#c286ff",
    "navHover": "#1f1730",
    "navActive": "#2d2147",
    "accent": "#bf82ff",
    "accentHover": "#38295b",
    "listHover": "#241a39",
    "chipBackground": "#1b152b",
    "chipHover": "#2a1f44",
    "chipActive": "#3d2d64",
    "chipBorder": "#6548a3",
}

lightTheme: dict[str, str] = {
    "background": "#f4f6f9",
    "sidebarBackground": "#eef2f6",
    "surface": "#ffffff",
    "surfaceAlt": "#f8fafc",
    "border": "#e1e6ee",
    "divider": "#e8edf3",
    "assistantBubble": "#f8fafc",
    "assistantBorder": "#e2e8f0",
    "userBubble": "#f0ebfb",
    "userBorder": "#d8cbf7",
    "pendingBubble": "#f2f6fa",
    "pendingBorder": "#d6e0eb",
    "systemBubble": "#f1f4f8",
    "systemBorder": "#dde4ec",
    "textPrimary": "#17202c",
    "textSecondary": "#5f6b7a",
    "textMuted": "#7d8795",
    "inputBackground": "#ffffff",
    "inputBorder": "#d4dce7",
    "inputFocusBorder": "#8d56ee",
    "navHover": "#ece8fa",
    "navActive": "#e7defb",
    "accent": "#8d56ee",
    "accentHover": "#ede6fb",
    "listHover": "#f2ecff",
    "chipBackground": "#f7f2ff",
    "chipHover": "#efe6ff",
    "chipActive": "#e6d9ff",
    "chipBorder": "#d9c9fb",
}

def _validate_theme(theme: dict[str, str]) -> None:
    for key in THEME_TOKEN_KEYS:
        value = theme.get(key, "")
        if not isinstance(value, str) or not value.startswith("#") or len(value) != 7:
            raise ValueError(f"Invalid theme token for {key!r}: {value!r}")


for _theme in (darkTheme, lightTheme, lumenPurpleTheme):
    _validate_theme(_theme)


DARK_PALETTE: dict[str, str] = _palette_from_theme(darkTheme)
LIGHT_PALETTE: dict[str, str] = _palette_from_theme(lightTheme)
LUMEN_PURPLE_PALETTE: dict[str, str] = _palette_from_theme(lumenPurpleTheme)

THEME_TOKENS: dict[str, dict[str, str]] = {
    "dark": dict(darkTheme),
    "light": dict(lightTheme),
    "custom": dict(lumenPurpleTheme),
}

THEME_PALETTES: dict[str, dict[str, str]] = {
    name: _palette_from_theme(tokens)
    for name, tokens in THEME_TOKENS.items()
}


def _clamp_channel(value: int) -> int:
    return max(0, min(255, value))


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    color = str(value or "").strip().lstrip("#")
    if len(color) != 6:
        raise ValueError(f"Unsupported color value: {value!r}")
    return tuple(int(color[index:index + 2], 16) for index in (0, 2, 4))


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*(_clamp_channel(channel) for channel in rgb))


def _blend(color: str, other: str, *, ratio: float) -> str:
    base = _hex_to_rgb(color)
    target = _hex_to_rgb(other)
    return _rgb_to_hex(
        tuple(
            round(base[index] + (target[index] - base[index]) * ratio)
            for index in range(3)
        )
    )


def custom_accent_palette(accent_hex: str) -> dict[str, str]:
    return _palette_from_theme(custom_accent_theme(accent_hex))


def custom_accent_theme(accent_hex: str) -> dict[str, str]:
    accent = str(accent_hex or "").strip() or "#9c67ff"
    base = dict(darkTheme)
    base["background"] = _blend(accent, darkTheme["background"], ratio=0.88)
    base["sidebarBackground"] = _blend(accent, darkTheme["sidebarBackground"], ratio=0.90)
    base["surface"] = _blend(accent, darkTheme["surface"], ratio=0.84)
    base["surfaceAlt"] = _blend(accent, darkTheme["surfaceAlt"], ratio=0.76)
    base["border"] = _blend(accent, darkTheme["border"], ratio=0.70)
    base["divider"] = _blend(accent, darkTheme["divider"], ratio=0.72)
    base["assistantBubble"] = _blend(accent, darkTheme["assistantBubble"], ratio=0.78)
    base["assistantBorder"] = _blend(accent, darkTheme["assistantBorder"], ratio=0.46)
    base["pendingBubble"] = _blend(accent, darkTheme["pendingBubble"], ratio=0.82)
    base["userBubble"] = _blend(accent, "#120f1d", ratio=0.42)
    base["userBorder"] = _blend(accent, "#ffffff", ratio=0.12)
    base["pendingBorder"] = _blend(accent, "#ffffff", ratio=0.08)
    base["systemBubble"] = _blend(accent, darkTheme["systemBubble"], ratio=0.84)
    base["systemBorder"] = _blend(accent, darkTheme["systemBorder"], ratio=0.58)
    base["inputBackground"] = _blend(accent, darkTheme["inputBackground"], ratio=0.86)
    base["inputBorder"] = _blend(accent, darkTheme["inputBorder"], ratio=0.56)
    base["inputFocusBorder"] = accent
    base["navHover"] = _blend(accent, darkTheme["navHover"], ratio=0.78)
    base["navActive"] = _blend(accent, "#171320", ratio=0.28)
    base["accent"] = _blend(accent, "#ffffff", ratio=0.14)
    base["accentHover"] = _blend(accent, "#141923", ratio=0.18)
    base["listHover"] = _blend(accent, darkTheme["listHover"], ratio=0.72)
    base["chipBackground"] = _blend(accent, darkTheme["chipBackground"], ratio=0.82)
    base["chipHover"] = _blend(accent, darkTheme["chipHover"], ratio=0.58)
    base["chipActive"] = _blend(accent, "#171320", ratio=0.30)
    base["chipBorder"] = _blend(accent, "#ffffff", ratio=0.10)
    _validate_theme(base)
    return base


def resolve_theme_tokens(theme_name: str | None, *, custom_accent_hex: str | None = None) -> dict[str, str]:
    normalized = str(theme_name or "dark").strip().lower()
    if normalized == "custom" and custom_accent_hex:
        return custom_accent_theme(custom_accent_hex)
    fallback = THEME_TOKENS.get(normalized, THEME_TOKENS["dark"])
    return dict(fallback)


def palette_from_theme(theme: dict[str, str]) -> dict[str, str]:
    return _palette_from_theme(theme)


COGNITIVE_INDICATOR_POOLS: dict[str, tuple[str, ...]] = {
    "conversation": (
        "Lumen is reasoning...",
        "Lumen is following the thread...",
        "Lumen is working through it...",
    ),
    "research": (
        "Lumen is analyzing...",
        "Lumen is examining the question...",
        "Lumen is tracing the evidence...",
    ),
    "engineering": (
        "Lumen is tracing...",
        "Lumen is analyzing the structure...",
        "Lumen is checking the system...",
    ),
}


@dataclass(frozen=True)
class DesktopChatMessage:
    sender: str
    text: str
    message_type: str
    mode: str | None = None
    timestamp: str | None = None


def session_list_label(session: dict[str, Any]) -> str:
    title = str(session.get("title") or "").strip()
    prompt = str(session.get("prompt") or "").strip()
    summary = str(session.get("summary") or "").strip()
    created_at = str(session.get("created_at") or "").strip()
    lead = title or prompt or summary or str(session.get("session_id") or "Session").strip()
    compact = lead if len(lead) <= 56 else f"{lead[:53].rstrip()}..."
    return f"{compact}  {created_at}" if created_at else compact


def parse_iso_datetime(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def day_group_label(value: object, *, now: datetime | None = None) -> str:
    stamp = parse_iso_datetime(value)
    if stamp is None:
        return "Older"
    reference = now or datetime.now(stamp.tzinfo)
    stamp_date = stamp.date()
    reference_date = reference.date()
    delta_days = (reference_date - stamp_date).days
    if delta_days <= 0:
        return "Today"
    if delta_days == 1:
        return "Yesterday"
    week_start = reference_date - timedelta(days=reference_date.weekday())
    if stamp_date >= week_start:
        return "Earlier This Week"
    return "Older"


def human_date_label(value: object, *, now: datetime | None = None) -> str:
    stamp = parse_iso_datetime(value)
    if stamp is None:
        return ""
    reference = now or datetime.now(stamp.tzinfo)
    time_label = stamp.strftime("%I:%M %p").lstrip("0")
    if stamp.date() == reference.date():
        return time_label
    if stamp.year == reference.year:
        return f"{stamp.strftime('%b %d')} • {time_label}"
    return stamp.strftime("%b %d, %Y")


def session_browser_line(session: dict[str, Any], *, now: datetime | None = None) -> str:
    title = str(session.get("title") or "").strip()
    prompt = str(session.get("prompt") or "").strip()
    summary = str(session.get("summary") or "").strip()
    session_id = str(session.get("session_id") or "Session").strip()
    lead = title or prompt or summary or session_id
    compact = lead if len(lead) <= 46 else f"{lead[:43].rstrip()}..."
    parts = [compact]
    date_label = human_date_label(session.get("created_at"), now=now)
    mode = str(session.get("mode") or "").strip().title()
    trailing = " • ".join(part for part in (date_label, mode) if part)
    if trailing:
        parts.append(trailing)
    return "  ".join(parts)


def grouped_session_rows(
    sessions: list[dict[str, Any]],
    *,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    current_group = None
    for session in sessions:
        group = day_group_label(session.get("created_at"), now=now)
        if group != current_group:
            rows.append({"kind": "header", "label": group, "session": None})
            current_group = group
        rows.append(
            {
                "kind": "session",
                "label": session_browser_line(session, now=now),
                "session": session,
            }
        )
    return rows


def grouped_entry_sections(
    entries: list[dict[str, Any]],
    *,
    now: datetime | None = None,
) -> list[tuple[str, list[dict[str, Any]]]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for entry in entries:
        group = day_group_label(entry.get("created_at"), now=now)
        if group not in buckets:
            buckets[group] = []
            order.append(group)
        buckets[group].append(entry)
    return [(group, buckets[group]) for group in order]


def knowledge_category_lines(overview: dict[str, Any]) -> list[str]:
    categories = overview.get("categories") if isinstance(overview, dict) else []
    if not isinstance(categories, list):
        return []
    lines: list[str] = []
    for category in categories:
        if not isinstance(category, dict):
            continue
        name = str(category.get("category") or "uncategorized").replace("_", "/").title()
        count = int(category.get("entry_count") or 0)
        titles = [str(title).strip() for title in category.get("titles") or [] if str(title).strip()]
        preview = ", ".join(titles[:3])
        line = f"{name} ({count})"
        if preview:
            line += f": {preview}"
        lines.append(line)
    return lines


def validate_palette(palette: dict[str, str]) -> None:
    for key in PALETTE_KEYS:
        value = palette.get(key, "")
        if not isinstance(value, str) or not value.startswith("#") or len(value) != 7:
            raise ValueError(f"Invalid palette value for {key!r}: {value!r}")


def normalize_cognitive_mode(mode: str | None) -> str:
    normalized = str(mode or "").strip().lower()
    if normalized in {"planning", "tool", "engineering"}:
        return "engineering"
    if normalized == "research":
        return "research"
    return "conversation"


def select_cognitive_indicator(*, mode: str | None, key: str = "") -> str:
    normalized = normalize_cognitive_mode(mode)
    pool = COGNITIVE_INDICATOR_POOLS[normalized]
    selector = key.strip() or normalized
    index = sum(ord(char) for char in selector) % len(pool)
    return pool[index]


def neutral_pending_phrase(*, key: str = "") -> str:
    pool = (
        "Lumen is reasoning...",
        "Lumen is working through it...",
        "Lumen is with you on this...",
    )
    selector = key.strip() or "neutral"
    index = sum(ord(char) for char in selector) % len(pool)
    return pool[index]


def build_pending_message(*, key: str = "") -> DesktopChatMessage:
    return DesktopChatMessage(
        sender="Lumen",
        text=neutral_pending_phrase(key=key),
        message_type="pending",
        mode="conversation",
        timestamp=message_timestamp(),
    )


def message_role_style(message_type: str, *, palette: dict[str, str] | None = None) -> dict[str, str]:
    active_palette = palette or DARK_PALETTE
    if message_type == "user":
        return {
            "anchor": "e",
            "bubble_bg": active_palette["user_bg"],
            "bubble_border": active_palette["user_border"],
            "sender_fg": active_palette["text_secondary"],
            "text_fg": active_palette["text_primary"],
        }
    if message_type == "assistant":
        return {
            "anchor": "w",
            "bubble_bg": active_palette["assistant_bg"],
            "bubble_border": active_palette["assistant_border"],
            "sender_fg": active_palette["text_secondary"],
            "text_fg": active_palette["text_primary"],
        }
    if message_type == "pending":
        return {
            "anchor": "w",
            "bubble_bg": active_palette["pending_bg"],
            "bubble_border": active_palette["pending_border"],
            "sender_fg": active_palette["text_secondary"],
            "text_fg": active_palette["text_primary"],
        }
    return {
        "anchor": "center",
        "bubble_bg": active_palette["system_bg"],
        "bubble_border": active_palette["system_border"],
        "sender_fg": active_palette["text_secondary"],
        "text_fg": active_palette["text_secondary"],
    }


def message_timestamp(moment: datetime | None = None) -> str:
    current = moment or datetime.now()
    return current.strftime("%I:%M %p").lstrip("0")


def nav_button_style(*, active: bool, hovered: bool, palette: dict[str, str]) -> dict[str, str]:
    if active:
        return {
            "bg": palette["nav_active_bg"],
            "fg": palette["text_primary"],
            "highlightbackground": palette["nav_active_border"],
        }
    if hovered:
        return {
            "bg": palette["nav_hover_bg"],
            "fg": palette["text_primary"],
            "highlightbackground": palette["nav_hover_bg"],
        }
    return {
        "bg": palette["sidebar_bg"],
        "fg": palette["text_secondary"],
        "highlightbackground": palette["sidebar_bg"],
    }


def empty_state_text(kind: str) -> str:
    normalized = str(kind or "").strip().lower()
    if normalized == "memory":
        return (
            "Saved memory and research notes will show up here after memory-worthy turns "
            "or saved research results."
        )
    if normalized == "recent":
        return (
            "Completed chat sessions show up here. Pick one to preview it, then open it "
            "back into the chat view."
        )
    return "Nothing to show here yet."
