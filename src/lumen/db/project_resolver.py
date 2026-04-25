from __future__ import annotations

from lumen.db.repositories import ProjectRepository, SessionRepository


class ProjectResolver:
    """Resolves project assignment for new structured persistence rows."""

    def __init__(
        self,
        *,
        project_repository: ProjectRepository,
        session_repository: SessionRepository,
    ):
        self.project_repository = project_repository
        self.session_repository = session_repository

    def resolve_project_id(
        self,
        *,
        session_id: str,
        prompt: str | None = None,
        title: str | None = None,
        active_topic: str | None = None,
        project_id: str | None = None,
        project_name: str | None = None,
    ) -> str:
        if project_id:
            existing = self.project_repository.get(project_id)
            if existing is not None:
                return str(existing["id"])
        if project_name:
            existing = self.project_repository.get_by_name(project_name)
            if existing is not None:
                return str(existing["id"])
            normalized = ProjectRepository.normalize_name(project_name)
            created = self.project_repository.upsert(
                project_id=normalized.replace(" ", "_") or "general",
                name=project_name,
                description=f"Auto-created project for {project_name}.",
                status="active",
            )
            return str(created["id"])

        existing_session = self.session_repository.get(session_id)
        if existing_session is not None and existing_session.get("project_id"):
            return str(existing_session["project_id"])

        candidates = self.project_repository.list_projects()
        search_text = " ".join(item for item in [prompt or "", title or "", active_topic or ""] if item)
        normalized_text = ProjectRepository.normalize_name(search_text)
        for candidate in candidates:
            normalized_name = str(candidate.get("normalized_name") or "").strip()
            if normalized_name and normalized_name in normalized_text:
                return str(candidate["id"])

        general = self.project_repository.ensure_general_project()
        return str(general["id"])
