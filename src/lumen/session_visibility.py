from __future__ import annotations


INTERNAL_SESSION_PREFIXES: tuple[str, ...] = (
    "packaged-",
    "source-",
    "readiness-",
    "sweep-",
    "release-",
    "qa-",
    "validation-",
    "audit-",
    "anh-path-routing-",
)

INTERNAL_SESSION_IDS: frozenset[str] = frozenset(
    {
        "codex-save-restore-check",
    }
)


def is_internal_session_id(session_id: object) -> bool:
    text = str(session_id or "").strip().lower()
    if not text:
        return False
    return text in INTERNAL_SESSION_IDS or text.startswith(INTERNAL_SESSION_PREFIXES)


def is_user_visible_session(session: dict[str, object] | object) -> bool:
    if not isinstance(session, dict):
        return False
    return not is_internal_session_id(session.get("session_id") or session.get("id"))
