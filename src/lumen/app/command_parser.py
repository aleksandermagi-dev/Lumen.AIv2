from __future__ import annotations

from pathlib import Path
from typing import Any

from lumen.routing.capability_manager import CapabilityManager
from lumen.routing.intents import AppIntent, CapabilityRequestIntent


class CommandParser:
    """Parses lightweight high-level Lumen commands into structured app intents."""

    def __init__(self, capability_manager: CapabilityManager):
        self.capability_manager = capability_manager

    def parse(
        self,
        *,
        action: str,
        target: str,
        input_path: Path | None = None,
        params: dict[str, Any] | None = None,
        session_id: str = "default",
        run_root: Path | None = None,
    ) -> AppIntent:
        normalized_action = action.strip().lower()
        normalized_target = target.strip().lower()
        resolved_input = input_path.resolve() if input_path else None
        resolved_run_root = run_root.resolve() if run_root else None
        normalized_params = params or {}

        capability = self.capability_manager.find_by_command(
            action=normalized_action,
            target=normalized_target,
        )
        if capability is not None:
            return CapabilityRequestIntent(
                intent_type="capability_request",
                action=normalized_action,
                target=normalized_target,
                capability_key=capability.capability_key,
                input_path=resolved_input,
                params=normalized_params,
                session_id=session_id,
                run_root=resolved_run_root,
            )

        return AppIntent(
            intent_type="generic",
            action=normalized_action,
            target=normalized_target,
            input_path=resolved_input,
            params=normalized_params,
            session_id=session_id,
            run_root=resolved_run_root,
        )
