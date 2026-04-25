from __future__ import annotations

from lumen.app.models import InteractionProfile
from lumen.memory.session_state_manager import SessionStateManager


class SessionContextService:
    """Handles persisted session-level thread context."""

    def __init__(self, session_state_manager: SessionStateManager):
        self.session_state_manager = session_state_manager

    def get_active_thread(self, session_id: str) -> dict[str, object] | None:
        return self.session_state_manager.get_active_thread(session_id)

    def get_interaction_profile(self, session_id: str):
        return self.session_state_manager.get_interaction_profile(session_id)

    def set_interaction_profile(
        self,
        session_id: str,
        profile: InteractionProfile,
    ) -> dict[str, object]:
        return self.session_state_manager.set_interaction_profile(session_id, profile)

    def get_session_metadata(self, session_id: str) -> dict[str, object]:
        return self.session_state_manager.get_session_metadata(session_id)

    def set_session_title(self, session_id: str, title: str | None) -> dict[str, object]:
        return self.session_state_manager.set_session_title(session_id, title)

    def set_session_archived(self, session_id: str, archived: bool) -> dict[str, object]:
        return self.session_state_manager.set_session_archived(session_id, archived)

    def delete_session(self, session_id: str) -> dict[str, object]:
        return self.session_state_manager.delete_session(session_id)

    def update_active_thread(
        self,
        *,
        session_id: str,
        prompt: str,
        response: dict[str, object],
        project_id: str | None = None,
        project_name: str | None = None,
    ) -> dict[str, object]:
        return self.session_state_manager.update_active_thread(
            session_id=session_id,
            prompt=prompt,
            response=response,
            project_id=project_id,
            project_name=project_name,
        )

    def clear_active_thread(self, session_id: str) -> dict[str, object]:
        return self.session_state_manager.clear_active_thread(session_id)
