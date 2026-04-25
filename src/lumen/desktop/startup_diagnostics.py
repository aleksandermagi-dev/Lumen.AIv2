from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any


@dataclass(slots=True)
class StartupCheckpointLogger:
    log_path: Path
    execution_mode: str
    pid: int = os.getpid()
    _started_at: float = 0.0

    def __post_init__(self) -> None:
        self._started_at = perf_counter()

    def checkpoint(self, checkpoint_id: str, phase: str, *, details: str | None = None) -> None:
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "pid": self.pid,
            "execution_mode": self.execution_mode,
            "checkpoint_id": str(checkpoint_id or "").strip(),
            "phase": str(phase or "").strip(),
            "elapsed_ms": round((perf_counter() - self._started_at) * 1000.0, 3),
        }
        if details:
            record["details"] = str(details)
        self._write_record(record)

    def error(self, checkpoint_id: str, exc: BaseException, *, details: str | None = None) -> None:
        detail = str(details or exc).strip() or exc.__class__.__name__
        self.checkpoint(checkpoint_id, "error", details=detail)

    def _write_record(self, record: dict[str, Any]) -> None:
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        except OSError:
            return


def startup_log_path(*, data_root: Path) -> Path:
    return data_root / "desktop_ui" / "startup_checkpoints.log"


def read_startup_checkpoints(log_path: Path) -> list[dict[str, Any]]:
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


def summarize_startup_checkpoints(log_path: Path) -> dict[str, Any]:
    records = read_startup_checkpoints(log_path)
    first_render_index = next(
        (
            index
            for index, item in enumerate(records)
            if str(item.get("checkpoint_id") or "") == "first_render_complete"
            and str(item.get("phase") or "") == "after"
        ),
        None,
    )
    before_first_render = records if first_render_index is None else records[: first_render_index + 1]
    completed_before_first_render = [
        str(item.get("checkpoint_id") or "")
        for item in before_first_render
        if str(item.get("phase") or "") == "after" and str(item.get("checkpoint_id") or "").strip()
    ]
    last_record = records[-1] if records else None
    stall_checkpoint = None
    if first_render_index is None and isinstance(last_record, dict):
        stall_checkpoint = str(last_record.get("checkpoint_id") or "").strip() or None
    return {
        "log_path": str(log_path),
        "record_count": len(records),
        "first_render_seen": first_render_index is not None,
        "completed_before_first_render": completed_before_first_render,
        "first_stall_checkpoint": stall_checkpoint,
        "last_checkpoint": last_record,
    }
