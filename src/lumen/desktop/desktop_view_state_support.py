from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ViewRefreshDecision:
    should_queue: bool
    should_clear: bool
    should_hold_for_hotbar: bool


def resolve_view_refresh_decision(
    *,
    view_name: str,
    view_enabled: bool,
    hotbar_animation_active: bool,
    hotbar_open: bool,
) -> ViewRefreshDecision:
    normalized = str(view_name or "").strip().lower()
    if normalized == "chat":
        return ViewRefreshDecision(
            should_queue=False,
            should_clear=True,
            should_hold_for_hotbar=False,
        )
    if not view_enabled:
        return ViewRefreshDecision(
            should_queue=False,
            should_clear=True,
            should_hold_for_hotbar=False,
        )
    if hotbar_animation_active or hotbar_open:
        return ViewRefreshDecision(
            should_queue=False,
            should_clear=False,
            should_hold_for_hotbar=True,
        )
    return ViewRefreshDecision(
        should_queue=True,
        should_clear=False,
        should_hold_for_hotbar=False,
    )
