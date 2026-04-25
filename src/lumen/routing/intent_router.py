from __future__ import annotations

from lumen.routing.capability_manager import CapabilityManager
from lumen.routing.intents import AppIntent, CapabilityRequestIntent, ToolExecutionIntent


class IntentRouter:
    """Minimal keyword router for high-level local Lumen commands."""

    def __init__(self, capability_manager: CapabilityManager):
        self.capability_manager = capability_manager

    def route(self, intent: AppIntent) -> ToolExecutionIntent:
        if isinstance(intent, CapabilityRequestIntent):
            capability = self.capability_manager.get(intent.capability_key)
            return ToolExecutionIntent(
                intent_type="tool_execution",
                tool_id=capability.tool_id,
                capability=capability.tool_capability,
                input_path=intent.input_path,
                params=intent.params,
                session_id=intent.session_id,
                run_root=intent.run_root,
            )

        raise ValueError(
            f"No route is defined for intent_type='{intent.intent_type}', "
            f"action='{intent.action}', target='{intent.target}'"
        )
