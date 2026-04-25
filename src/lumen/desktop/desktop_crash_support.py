from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import traceback
from typing import Any


def desktop_crash_log_path(*, data_root: Path) -> Path:
    return data_root / "desktop_ui" / "desktop_crash.log"


def desktop_runtime_failure_log_path(*, data_root: Path) -> Path:
    return data_root / "desktop_ui" / "desktop_runtime_failures.txt"


def build_crash_record(
    *,
    execution_mode: str,
    source: str,
    exc: BaseException,
    traceback_obj=None,
    current_view: str | None = None,
    pending_view: str | None = None,
    deferred_target: str | None = None,
    details: str | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "execution_mode": str(execution_mode or ""),
        "source": str(source or "").strip() or "desktop",
        "exception_type": exc.__class__.__name__,
        "exception_message": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, traceback_obj or exc.__traceback__)).strip(),
    }
    if current_view:
        record["current_view"] = str(current_view)
    if pending_view:
        record["pending_view"] = str(pending_view)
    if deferred_target:
        record["deferred_target"] = str(deferred_target)
    if details:
        record["details"] = str(details)
    if context:
        record["context"] = dict(context)
    return record


def append_crash_record(*, log_path: Path, record: dict[str, Any]) -> None:
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    except OSError:
        return


def append_runtime_failure_text(*, log_path: Path, lines: list[str]) -> None:
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines).rstrip() + "\n\n")
    except OSError:
        return


def read_crash_records(log_path: Path) -> list[dict[str, Any]]:
    if not log_path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records
