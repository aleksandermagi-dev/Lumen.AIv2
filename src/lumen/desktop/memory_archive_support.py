from __future__ import annotations

from typing import Any

from lumen.desktop.chat_ui_support import day_group_label, human_date_label


def memory_entry_key(entry: dict[str, object]) -> str:
    kind = str(entry.get("kind") or "memory").strip() or "memory"
    raw_key = str(
        entry.get("entry_path")
        or entry.get("note_path")
        or entry.get("artifact_path")
        or entry.get("memory_item_id")
        or entry.get("id")
        or entry.get("source_id")
        or ""
    ).strip()
    if raw_key:
        return f"{kind}:{raw_key}"
    title = str(entry.get("title") or "").strip()
    created_at = str(entry.get("created_at") or "").strip()
    content = str(entry.get("content") or "").strip()
    return f"{kind}:{title}:{created_at}:{content[:80]}"


def memory_entries_signature(entries: list[dict[str, object]]) -> tuple[tuple[str, str, str, str], ...]:
    return tuple(
        (
            str(item.get("title") or ""),
            str(item.get("created_at") or ""),
            str(item.get("kind") or ""),
            memory_entry_key(item),
        )
        for item in entries
        if isinstance(item, dict)
    )


def build_memory_row_descriptors(
    entries: list[dict[str, object]],
) -> tuple[list[tuple[object, ...]], dict[str, dict[str, object]]]:
    descriptors: list[tuple[object, ...]] = []
    entry_map: dict[str, dict[str, object]] = {}
    current_group = None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        group = day_group_label(entry.get("created_at"))
        if group != current_group:
            descriptors.append(("header", group, 12))
            current_group = group
        entry_key = memory_entry_key(entry)
        descriptor = (
            "entry",
            str(entry.get("title") or "Memory").strip(),
            human_date_label(entry.get("created_at")),
            str(entry.get("kind") or "memory").replace("_", " ").title(),
            entry_key,
        )
        descriptors.append(descriptor)
        entry_map[entry_key] = entry
    return descriptors, entry_map


def build_memory_row_cache(
    entries: list[dict[str, object]],
) -> tuple[
    list[tuple[object, ...]],
    dict[str, dict[str, object]],
    tuple[int, ...],
    tuple[int, ...],
]:
    descriptors: list[tuple[object, ...]] = []
    entry_map: dict[str, dict[str, object]] = {}
    descriptor_offsets: list[int] = [0]
    group_counts: list[int] = [0]
    current_group = None
    group_count = 0
    visible_entries = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        group = day_group_label(entry.get("created_at"))
        if group != current_group:
            descriptors.append(("header", group, 12))
            current_group = group
            group_count += 1
        entry_key = memory_entry_key(entry)
        descriptors.append(
            (
                "entry",
                str(entry.get("title") or "Memory").strip(),
                human_date_label(entry.get("created_at")),
                str(entry.get("kind") or "memory").replace("_", " ").title(),
                entry_key,
            )
        )
        entry_map[entry_key] = entry
        visible_entries += 1
        descriptor_offsets.append(len(descriptors))
        group_counts.append(group_count)
    return descriptors, entry_map, tuple(descriptor_offsets), tuple(group_counts)


def memory_row_cache_slice(
    descriptors: list[tuple[object, ...]],
    descriptor_offsets: tuple[int, ...],
    *,
    visible_count: int,
) -> list[tuple[object, ...]]:
    if not descriptor_offsets:
        return []
    normalized_visible = max(0, min(int(visible_count), len(descriptor_offsets) - 1))
    return descriptors[: descriptor_offsets[normalized_visible]]


def memory_rows_match_descriptors(widgets: list[Any], descriptors: list[tuple[object, ...]]) -> bool:
    if len(widgets) != len(descriptors):
        return False
    return all(getattr(widget, "_browser_descriptor", None) == descriptor for widget, descriptor in zip(widgets, descriptors))


def bounded_entries_slice(
    entries: list[dict[str, object]],
    *,
    render_limit: int,
    has_more_available: bool = False,
) -> tuple[list[dict[str, object]], int, bool]:
    total_count = len(entries)
    visible_count = max(0, min(int(render_limit), total_count))
    has_more = visible_count < total_count or (bool(has_more_available) and visible_count >= total_count)
    return entries[:visible_count], visible_count, has_more


def memory_group_count(entries: list[dict[str, object]]) -> int:
    current_group = None
    count = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        group = day_group_label(entry.get("created_at"))
        if group != current_group:
            current_group = group
            count += 1
    return count


def should_render_archived_memory_cache(
    *,
    entries: list[dict[str, object]],
    cached_signature: tuple[tuple[str, str, str, str], ...],
    painted_signature: tuple[tuple[str, str, str, str], ...],
    loaded_version: int,
    state_version: int,
    has_children: bool,
) -> bool:
    return (
        (entries or cached_signature or loaded_version == state_version)
        and (not has_children or cached_signature != painted_signature)
    )


def should_fetch_archived_memory(
    *,
    current_view: str,
    loaded_version: int,
    state_version: int,
    fetch_in_flight: bool,
    requested_version: int | None,
) -> bool:
    if current_view == "archived_memory" and loaded_version == state_version:
        return False
    if current_view == "archived_memory" and fetch_in_flight and requested_version == state_version:
        return False
    return True
