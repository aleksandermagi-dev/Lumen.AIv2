from __future__ import annotations


def display_status_label(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    explicit = {
        "under_tension": "under tension",
        "strongly_supported": "strongly supported",
        "moderately_supported": "moderately supported",
        "insufficiently_grounded": "insufficiently grounded",
    }
    return explicit.get(normalized, normalized.replace("_", " "))
