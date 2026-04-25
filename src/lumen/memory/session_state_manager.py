from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
from typing import Any

from lumen.app.context_policy import ContextPolicy
from lumen.app.models import ActiveThreadState, InteractionProfile
from lumen.app.settings import AppSettings
from lumen.db.persistence_manager import PersistenceManager
from lumen.nlu.follow_up_inventory import looks_like_general_follow_up, looks_like_reference_follow_up
from lumen.nlu.prompt_nlu import PromptNLU
from lumen.nlu.prompt_surface_builder import PromptSurfaceBuilder
from lumen.services.reasoning_state_service import ReasoningStateService
from lumen.routing.anchor_registry import detect_follow_up_anchor
from lumen.schemas.interaction_profile_schema import InteractionProfileSchema
from lumen.schemas.session_thread_schema import SessionThreadSchema


class SessionStateManager:
    """Persists lightweight active thread state per session."""

    STATE_FILENAME = "thread_state.json"
    PROFILE_FILENAME = "interaction_profile.json"
    METADATA_FILENAME = "session_metadata.json"
    TARGET_STATE_BYTES = 100 * 1024

    def __init__(self, settings: AppSettings, persistence_manager: PersistenceManager | None = None):
        self.settings = settings
        self.sessions_root = settings.sessions_root
        self.max_state_bytes = settings.max_session_state_bytes
        self.context_policy = ContextPolicy.from_settings(settings)
        self.prompt_nlu = PromptNLU()
        self.reasoning_state_service = ReasoningStateService()
        self.persistence_manager = persistence_manager or PersistenceManager(settings)

    def get_active_thread(self, session_id: str) -> dict[str, Any] | None:
        payload = self._load_active_thread_payload(session_id)
        if payload is None:
            return None
        payload["interaction_profile"] = self.get_interaction_profile(session_id).to_dict()
        return payload

    def get_interaction_profile(self, session_id: str) -> InteractionProfile:
        session = self._persisted_session(session_id)
        if session is not None and isinstance(session.get("metadata_json"), dict):
            metadata_payload = dict(session["metadata_json"]).get("interaction_profile")
            if isinstance(metadata_payload, dict):
                return InteractionProfile.from_mapping(metadata_payload)
        profile_path = self._profile_path(session_id)
        if profile_path.exists():
            self.persistence_manager.record_fallback_read("session profile fallback")
            payload = InteractionProfileSchema.normalize(
                json.loads(profile_path.read_text(encoding="utf-8"))
            )
            InteractionProfileSchema.validate(payload)
            return InteractionProfile.from_mapping(payload)
        active_thread = self._load_active_thread_payload(session_id)
        if not active_thread:
            return InteractionProfile.default()
        return InteractionProfile.from_mapping(active_thread.get("interaction_profile"))

    def get_session_metadata(self, session_id: str) -> dict[str, Any]:
        session = self._persisted_session(session_id)
        if session is not None:
            metadata = session.get("metadata_json") if isinstance(session.get("metadata_json"), dict) else {}
            return {
                "session_id": session_id,
                "title": str(session.get("title") or "").strip() or None,
                "archived": str(session.get("status") or "").strip() == "archived" or bool(metadata.get("archived", False)),
                "title_source": str(metadata.get("title_source") or "").strip() or None,
                "title_locked": bool(metadata.get("title_locked", False)),
                "metadata_path": str(self._metadata_path(session_id)),
            }
        path = self._metadata_path(session_id)
        if not path.exists():
            return {"session_id": session_id, "title": None, "archived": False, "metadata_path": str(path)}
        self.persistence_manager.record_fallback_read("session metadata fallback")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        return {
            "session_id": session_id,
            "title": str(payload.get("title") or "").strip() or None,
            "archived": bool(payload.get("archived", False)),
            "title_source": str(payload.get("title_source") or "").strip() or None,
            "title_locked": bool(payload.get("title_locked", False)),
            "metadata_path": str(path),
        }

    def set_session_title(self, session_id: str, title: str | None) -> dict[str, Any]:
        current = self.get_session_metadata(session_id)
        path = self._metadata_path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        normalized_title = " ".join(str(title or "").split()).strip() or None
        payload = {
            "session_id": session_id,
            "title": normalized_title,
            "archived": current.get("archived", False),
            "title_source": "manual_user" if normalized_title else None,
            "title_locked": bool(normalized_title),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        current_session = self._persisted_session(session_id)
        existing_metadata = (
            dict(current_session.get("metadata_json") or {})
            if isinstance(current_session, dict)
            else {}
        )
        self.persistence_manager.ensure_session(
            session_id=session_id,
            prompt=normalized_title,
            created_at=datetime.now(UTC).isoformat(),
            mode=None,
            title=normalized_title,
            status="archived" if current.get("archived", False) else "active",
            metadata={
                **existing_metadata,
                "archived": current.get("archived", False),
                "title_source": "manual_user" if normalized_title else None,
                "title_locked": bool(normalized_title),
            },
        )
        return {
            **payload,
            "metadata_path": str(path),
        }

    def set_session_archived(self, session_id: str, archived: bool) -> dict[str, Any]:
        current = self.get_session_metadata(session_id)
        path = self._metadata_path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "session_id": session_id,
            "title": current.get("title"),
            "archived": bool(archived),
            "title_source": current.get("title_source"),
            "title_locked": bool(current.get("title_locked", False)),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        current_session = self._persisted_session(session_id)
        existing_metadata = (
            dict(current_session.get("metadata_json") or {})
            if isinstance(current_session, dict)
            else {}
        )
        self.persistence_manager.ensure_session(
            session_id=session_id,
            prompt=current.get("title"),
            created_at=datetime.now(UTC).isoformat(),
            mode=None,
            title=current.get("title"),
            status="archived" if archived else "active",
            metadata={
                **existing_metadata,
                "archived": archived,
                "title_source": current.get("title_source") or existing_metadata.get("title_source"),
                "title_locked": bool(current.get("title_locked", existing_metadata.get("title_locked", False))),
            },
        )
        return {
            **payload,
            "metadata_path": str(path),
        }

    def set_interaction_profile(
        self,
        session_id: str,
        profile: InteractionProfile,
    ) -> dict[str, Any]:
        payload = InteractionProfileSchema.normalize(profile.to_dict())
        InteractionProfileSchema.validate(payload)
        path = self._profile_path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        current = self._persisted_session(session_id)
        existing_metadata = (
            dict(current.get("metadata_json") or {})
            if isinstance(current, dict)
            else {}
        )
        if isinstance(current, dict):
            self.persistence_manager.ensure_session(
                session_id=session_id,
                prompt=None,
                created_at=datetime.now(UTC).isoformat(),
                mode=None,
                title=str(current.get("title") or "").strip() or None,
                project_id=str(current.get("project_id")) if current.get("project_id") else None,
                status=str(current.get("status") or "active"),
                metadata={**existing_metadata, "interaction_profile": payload},
            )
        payload["profile_path"] = str(path)
        return payload

    def update_active_thread(
        self,
        *,
        session_id: str,
        prompt: str,
        response: dict[str, object],
        project_id: str | None = None,
        project_name: str | None = None,
    ) -> dict[str, Any]:
        previous = self.get_active_thread(session_id)
        active_profile = self._build_interaction_profile(
            session_id=session_id,
            response=response,
            previous=previous,
        )
        session_prompt = self._build_session_prompt(prompt=prompt, response=response)
        analysis_prompt = self._build_analysis_prompt(prompt=prompt, response=response)
        understanding = self.prompt_nlu.analyze(analysis_prompt)
        reasoning_state = self._build_reasoning_state(response=response, previous=previous)
        state = ActiveThreadState(
            session_id=session_id,
            mode=str(response.get("mode", "unknown")),
            kind=str(response.get("kind", "unknown")),
            prompt=session_prompt,
            original_prompt=self._build_original_prompt(prompt=prompt, response=response),
            objective=self._build_objective(prompt=analysis_prompt, response=response, previous=previous),
            thread_summary=self._build_thread_summary(prompt=analysis_prompt, response=response, previous=previous),
            summary=str(response.get("summary", "")),
            confidence_posture=str(response.get("confidence_posture") or "").strip() or None,
            tool_route_origin=str(response.get("tool_route_origin") or "").strip() or None,
            local_context_assessment=str(response.get("local_context_assessment") or "").strip() or None,
            coherence_topic=str(((response.get("reasoning_frame") or {}).get("coherence_topic")) or "").strip() or None,
            route_status=str(response.get("route_status") or "").strip() or None,
            support_status=str(response.get("support_status") or "").strip() or None,
            tension_status=str(response.get("tension_status") or "").strip() or None,
            interaction_profile=active_profile.to_dict(),
            pipeline_observability=self._compact_pipeline_observability(response=response),
            pipeline_trace={},
            detected_language=understanding.language.code,
            normalized_topic=understanding.topic.value,
            dominant_intent=understanding.intent.label,
            intent_domain=self._resolve_cognitive_text(
                response=response,
                reasoning_state=reasoning_state,
                key="intent_domain",
            ),
            intent_domain_confidence=(
                self._resolve_cognitive_float(
                    response=response,
                    reasoning_state=reasoning_state,
                    key="intent_domain_confidence",
                )
            ),
            response_depth=self._resolve_cognitive_text(
                response=response,
                reasoning_state=reasoning_state,
                key="response_depth",
            ),
            conversation_phase=self._resolve_cognitive_text(
                response=response,
                reasoning_state=reasoning_state,
                key="conversation_phase",
            ),
            next_step_state=dict(response.get("next_step_state") or {}),
            tool_suggestion_state=dict(response.get("tool_suggestion_state") or {}),
            trainability_trace=self._compact_value(
                response.get("trainability_trace"),
                label="trainability_trace",
                max_chars=12_000,
            ),
            supervised_support_trace=self._compact_value(
                response.get("supervised_support_trace"),
                label="supervised_support_trace",
                max_chars=12_000,
            ),
            extracted_entities=tuple(entity.to_dict() for entity in understanding.entities),
            tool_context=self._build_tool_context(response=response, previous=previous),
            continuation_offer=self._build_continuation_offer(response=response),
            reasoning_state=reasoning_state,
            updated_at=datetime.now(UTC),
        )
        payload = SessionThreadSchema.normalize(
            {
                "session_id": state.session_id,
                "mode": state.mode,
                "kind": state.kind,
                "prompt": state.prompt,
                "original_prompt": state.original_prompt,
                "objective": state.objective,
                "thread_summary": state.thread_summary,
                "summary": state.summary,
                "confidence_posture": state.confidence_posture,
                "tool_route_origin": state.tool_route_origin,
                "local_context_assessment": state.local_context_assessment,
                "coherence_topic": state.coherence_topic,
                "route_status": state.route_status,
                "support_status": state.support_status,
                "tension_status": state.tension_status,
                "interaction_profile": state.interaction_profile,
                "pipeline_observability": state.pipeline_observability,
                "pipeline_trace": state.pipeline_trace,
                "detected_language": state.detected_language,
                "normalized_topic": state.normalized_topic,
                "dominant_intent": state.dominant_intent,
                "intent_domain": state.intent_domain,
                "intent_domain_confidence": state.intent_domain_confidence,
                "response_depth": state.response_depth,
                "conversation_phase": state.conversation_phase,
                "next_step_state": state.next_step_state,
                "tool_suggestion_state": state.tool_suggestion_state,
                "trainability_trace": state.trainability_trace,
                "supervised_support_trace": state.supervised_support_trace,
                "extracted_entities": list(state.extracted_entities),
                "tool_context": state.tool_context,
                "continuation_offer": state.continuation_offer,
                "reasoning_state": state.reasoning_state,
                "updated_at": state.updated_at.isoformat(),
            }
        )
        SessionThreadSchema.validate(payload)
        path = self._state_path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload, serialized = self._prepare_payload_for_storage(payload)
        path.write_text(serialized, encoding="utf-8")
        payload["state_path"] = str(path)
        self.persistence_manager.record_session_state(
            session_id=session_id,
            payload=payload,
            project_id=project_id,
            project_name=project_name,
        )
        return payload

    def clear_active_thread(self, session_id: str) -> dict[str, Any]:
        path = self._state_path(session_id)
        existed = path.exists()
        if existed:
            path.unlink()
        current = self._persisted_session(session_id)
        if isinstance(current, dict):
            metadata = dict(current.get("metadata_json") or {})
            for key in (
                "session_id",
                "mode",
                "kind",
                "prompt",
                "original_prompt",
                "objective",
                "thread_summary",
                "summary",
                "confidence_posture",
                "tool_route_origin",
                "local_context_assessment",
                "coherence_topic",
                "route_status",
                "support_status",
                "tension_status",
                "interaction_profile",
                "pipeline_observability",
                "pipeline_trace",
                "detected_language",
                "normalized_topic",
                "dominant_intent",
                "intent_domain",
                "intent_domain_confidence",
                "response_depth",
                "conversation_phase",
                "next_step_state",
                "tool_suggestion_state",
                "trainability_trace",
                "supervised_support_trace",
                "extracted_entities",
                "tool_context",
                "continuation_offer",
                "reasoning_state",
                "updated_at",
            ):
                metadata.pop(key, None)
            self.persistence_manager.sessions.upsert(
                session_id=session_id,
                project_id=str(current.get("project_id")) if current.get("project_id") else None,
                title=str(current.get("title") or "").strip() or None,
                mode=str(current.get("mode") or "").strip() or None,
                started_at=str(current.get("started_at") or datetime.now(UTC).isoformat()),
                updated_at=datetime.now(UTC).isoformat(),
                status=str(current.get("status") or "active"),
                summary_id=str(current.get("summary_id")) if current.get("summary_id") else None,
                metadata=metadata or None,
            )
        return {
            "session_id": session_id,
            "cleared": existed,
            "state_path": str(path),
        }

    def delete_session(self, session_id: str) -> dict[str, Any]:
        session_root = self.sessions_root / session_id
        existed = session_root.exists()
        if existed:
            shutil.rmtree(session_root, ignore_errors=True)
        self.persistence_manager.delete_session(session_id)
        return {
            "session_id": session_id,
            "deleted": existed,
            "session_root": str(session_root),
        }

    def _state_path(self, session_id: str) -> Path:
        return self.sessions_root / session_id / self.STATE_FILENAME

    def _profile_path(self, session_id: str) -> Path:
        return self.sessions_root / session_id / self.PROFILE_FILENAME

    def _metadata_path(self, session_id: str) -> Path:
        return self.sessions_root / session_id / self.METADATA_FILENAME

    def _load_active_thread_payload(self, session_id: str) -> dict[str, Any] | None:
        session = self._persisted_session(session_id)
        if session is not None and isinstance(session.get("metadata_json"), dict):
            metadata_payload = dict(session["metadata_json"])
            if metadata_payload.get("session_id") == session_id and metadata_payload.get("updated_at"):
                latest_summary = self._latest_session_summary(session_id)
                if latest_summary and not str(metadata_payload.get("thread_summary") or "").strip():
                    metadata_payload["thread_summary"] = str(latest_summary.get("summary_text") or "").strip() or None
                if latest_summary and not str(metadata_payload.get("summary") or "").strip():
                    metadata_payload["summary"] = str(latest_summary.get("summary_text") or "").strip() or None
                metadata_payload["state_path"] = str(self._state_path(session_id))
                return metadata_payload
        path = self._state_path(session_id)
        if not path.exists():
            return None
        self.persistence_manager.record_fallback_read("session thread fallback")
        try:
            if path.stat().st_size > self.max_state_bytes:
                return None
            payload = SessionThreadSchema.normalize(
                json.loads(path.read_text(encoding="utf-8"))
            )
            SessionThreadSchema.validate(payload)
        except (json.JSONDecodeError, MemoryError, OSError, ValueError):
            return None
        payload["state_path"] = str(path)
        return payload

    def _persisted_session(self, session_id: str) -> dict[str, Any] | None:
        try:
            self.persistence_manager.bootstrap()
            return self.persistence_manager.sessions.get(session_id)
        except Exception:
            return None

    def _latest_session_summary(self, session_id: str) -> dict[str, Any] | None:
        try:
            self.persistence_manager.bootstrap()
            return self.persistence_manager.session_summaries.latest_by_session(session_id)
        except Exception:
            return None

    def _prepare_payload_for_storage(self, payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
        prepared = payload
        serialized = self._safe_json_dumps(prepared)
        target_bytes = min(self.max_state_bytes, self.TARGET_STATE_BYTES)
        if serialized is None or len(serialized.encode("utf-8")) > target_bytes:
            prepared = self._compact_state_payload(prepared)
            serialized = self._safe_json_dumps(prepared)
        if serialized is None or len(serialized.encode("utf-8")) > target_bytes:
            prepared = self._minimal_state_payload(prepared)
            serialized = self._safe_json_dumps(prepared)
        if serialized is None:
            raise MemoryError("Unable to serialize session thread state safely.")
        return prepared, serialized

    @staticmethod
    def _safe_json_dumps(payload: object) -> str | None:
        try:
            return json.dumps(payload, indent=2)
        except (MemoryError, TypeError, ValueError):
            return None

    def _compact_state_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        compact = dict(payload)
        for key in ("pipeline_observability", "pipeline_trace", "trainability_trace", "supervised_support_trace", "tool_context", "reasoning_state"):
            compact[key] = self._compact_value(compact.get(key), label=key)
        return compact

    def _minimal_state_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        minimal = dict(payload)
        minimal["pipeline_observability"] = {}
        minimal["pipeline_trace"] = {}
        minimal["trainability_trace"] = self._compact_value(
            minimal.get("trainability_trace"),
            label="trainability_trace",
            max_chars=20_000,
        )
        minimal["supervised_support_trace"] = self._compact_value(
            minimal.get("supervised_support_trace"),
            label="supervised_support_trace",
            max_chars=20_000,
        )
        minimal["tool_context"] = self._compact_value(minimal.get("tool_context"), label="tool_context", max_chars=20_000)
        minimal["reasoning_state"] = self._compact_value(minimal.get("reasoning_state"), label="reasoning_state", max_chars=20_000)
        return minimal

    @staticmethod
    def _compact_pipeline_observability(*, response: dict[str, object]) -> dict[str, Any]:
        observability = response.get("pipeline_observability")
        if not isinstance(observability, dict):
            observability = {}
        route = response.get("route") if isinstance(response.get("route"), dict) else {}
        local_knowledge = (
            response.get("local_knowledge_access")
            if isinstance(response.get("local_knowledge_access"), dict)
            else {}
        )
        conversation = (
            response.get("conversation_access")
            if isinstance(response.get("conversation_access"), dict)
            else {}
        )
        tool_access = response.get("tool_access") if isinstance(response.get("tool_access"), dict) else {}
        diagnostic = (
            response.get("runtime_diagnostic")
            if isinstance(response.get("runtime_diagnostic"), dict)
            else response.get("diagnostic") if isinstance(response.get("diagnostic"), dict) else {}
        )
        response_summary = (
            observability.get("response_summary")
            if isinstance(observability.get("response_summary"), dict)
            else {}
        )
        compact: dict[str, Any] = {
            "compacted": True,
            "route_summary": {
                "source": route.get("source"),
                "reason": route.get("reason"),
                "strength": route.get("strength"),
                "confidence": route.get("confidence"),
            },
            "response_summary": {
                "package_type": response_summary.get("package_type"),
                "answer": str(response.get("summary") or "")[:500] or None,
                "mode": response.get("mode"),
                "kind": response.get("kind"),
            },
        }
        if local_knowledge:
            compact["local_knowledge_access"] = {
                "local_knowledge_match": local_knowledge.get("local_knowledge_match"),
                "knowledge_entry_id": local_knowledge.get("knowledge_entry_id"),
                "knowledge_match_type": local_knowledge.get("knowledge_match_type"),
                "final_source": local_knowledge.get("final_source"),
            }
        if conversation:
            compact["conversation_access"] = {
                "conversation_match_type": conversation.get("conversation_match_type"),
                "conversation_context_used": conversation.get("conversation_context_used"),
                "final_source": conversation.get("final_source"),
            }
        if tool_access:
            compact["tool_access"] = {
                "tool_id": tool_access.get("tool_id"),
                "capability": tool_access.get("capability"),
                "tool_execution_required": tool_access.get("tool_execution_required"),
                "final_source": tool_access.get("final_source"),
            }
        if diagnostic:
            compact["diagnostic"] = {
                "failure_stage": diagnostic.get("failure_stage"),
                "failure_class": diagnostic.get("failure_class"),
                "tool_id": diagnostic.get("tool_id"),
                "capability": diagnostic.get("capability"),
            }
        return compact

    @staticmethod
    def _compact_value(value: object, *, label: str, max_chars: int = 50_000) -> object:
        try:
            serialized = json.dumps(value, sort_keys=True)
        except (MemoryError, TypeError, ValueError):
            serialized = None
        if serialized is not None and len(serialized) <= max_chars:
            return value
        summary: dict[str, object] = {
            "compacted": True,
            "label": label,
            "detail": "Value omitted from session-thread storage because it was too large to persist safely.",
        }
        if serialized is not None:
            summary["approx_chars"] = len(serialized)
        if isinstance(value, dict):
            summary["top_level_keys"] = list(value.keys())[:20]
        elif isinstance(value, list):
            summary["item_count"] = len(value)
        elif value is not None:
            summary["value_type"] = type(value).__name__
        return summary

    @staticmethod
    def _build_session_prompt(
        *,
        prompt: str,
        response: dict[str, object],
    ) -> str:
        return " ".join(str(prompt or "").strip().split())

    @staticmethod
    def _build_analysis_prompt(
        *,
        prompt: str,
        response: dict[str, object],
    ) -> str:
        resolved_prompt = " ".join(str(response.get("resolved_prompt") or "").strip().split())
        if resolved_prompt:
            return resolved_prompt
        return " ".join(str(prompt or "").strip().split())

    @staticmethod
    def _build_original_prompt(
        *,
        prompt: str,
        response: dict[str, object],
    ) -> str | None:
        normalized_original = " ".join(str(prompt or "").strip().split())
        normalized_resolved = " ".join(str(response.get("resolved_prompt") or "").strip().split())
        if not normalized_resolved or normalized_original == normalized_resolved:
            return None
        return normalized_original

    def _build_objective(
        self,
        *,
        prompt: str,
        response: dict[str, object],
        previous: dict[str, Any] | None,
    ) -> str:
        mode = str(response.get("mode", "")).strip()
        normalized_prompt = " ".join(prompt.strip().split())
        if previous and SessionStateManager._looks_like_follow_up(normalized_prompt):
            previous_objective = str(previous.get("objective", "")).strip()
            previous_mode = str(previous.get("mode", "")).strip()
            if previous_objective and previous_mode == mode:
                return previous_objective
        if mode == "planning":
            return self.context_policy.truncate_text(
                f"Plan work for: {normalized_prompt}",
                self.context_policy.session_objective_max_length,
            )
        if mode == "research":
            return self.context_policy.truncate_text(
                f"Research topic: {normalized_prompt}",
                self.context_policy.session_objective_max_length,
            )
        if mode == "tool":
            return self.context_policy.truncate_text(
                f"Execute tool task: {normalized_prompt}",
                self.context_policy.session_objective_max_length,
            )
        return self.context_policy.truncate_text(
            normalized_prompt,
            self.context_policy.session_objective_max_length,
        )

    def _build_thread_summary(
        self,
        *,
        prompt: str,
        response: dict[str, object],
        previous: dict[str, Any] | None,
    ) -> str:
        normalized_prompt = " ".join(prompt.strip().split())
        current_summary = str(response.get("summary", "")).strip() or normalized_prompt
        if previous and SessionStateManager._looks_like_follow_up(normalized_prompt):
            prior = str(previous.get("thread_summary") or previous.get("summary") or "").strip()
            if prior:
                return self.context_policy.truncate_text(
                    f"{prior} | latest: {normalized_prompt}",
                    self.context_policy.session_thread_summary_max_length,
                )
        return self.context_policy.truncate_text(
            current_summary,
            self.context_policy.session_thread_summary_max_length,
        )

    @staticmethod
    def _build_tool_context(
        *,
        response: dict[str, object],
        previous: dict[str, Any] | None,
    ) -> dict[str, Any]:
        mode = str(response.get("mode", "")).strip()
        if mode == "tool":
            tool_execution = response.get("tool_execution")
            if isinstance(tool_execution, dict):
                return dict(tool_execution)
            return {}
        if previous:
            return dict(previous.get("tool_context") or {})
        return {}

    def _build_reasoning_state(
        self,
        *,
        response: dict[str, object],
        previous: dict[str, Any] | None,
    ) -> dict[str, Any]:
        state = response.get("reasoning_state")
        if isinstance(state, dict) and state:
            return self.reasoning_state_service.from_mapping(state).to_dict()
        if previous:
            previous_state = previous.get("reasoning_state")
            if isinstance(previous_state, dict) and previous_state:
                return self.reasoning_state_service.from_mapping(previous_state).to_dict()
        return {}

    @staticmethod
    def _resolve_cognitive_text(
        *,
        response: dict[str, object],
        reasoning_state: dict[str, Any],
        key: str,
    ) -> str | None:
        raw = response.get(key)
        if raw is None and reasoning_state:
            raw = reasoning_state.get(key)
        value = str(raw or "").strip()
        return value or None

    @staticmethod
    def _resolve_cognitive_float(
        *,
        response: dict[str, object],
        reasoning_state: dict[str, Any],
        key: str,
    ) -> float | None:
        raw = response.get(key)
        if raw is None and reasoning_state:
            raw = reasoning_state.get(key)
        if raw is None:
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _build_continuation_offer(*, response: dict[str, object]) -> dict[str, Any]:
        offer = response.get("continuation_offer")
        if not isinstance(offer, dict):
            return {}
        target_prompt = str(offer.get("target_prompt") or "").strip()
        label = str(offer.get("label") or "").strip()
        if not target_prompt:
            return {}
        payload: dict[str, Any] = {"target_prompt": target_prompt}
        if label:
            payload["label"] = label
        kind = str(offer.get("kind") or "").strip()
        if kind:
            payload["kind"] = kind
        topic = str(offer.get("topic") or "").strip()
        if topic:
            payload["topic"] = topic
        explanation_mode = str(offer.get("explanation_mode") or "").strip()
        if explanation_mode:
            payload["explanation_mode"] = explanation_mode
        return payload

    def _build_interaction_profile(
        self,
        *,
        session_id: str,
        response: dict[str, object],
        previous: dict[str, Any] | None,
    ) -> InteractionProfile:
        profile_path = self._profile_path(session_id)
        if profile_path.exists():
            payload = InteractionProfileSchema.normalize(
                json.loads(profile_path.read_text(encoding="utf-8"))
            )
            InteractionProfileSchema.validate(payload)
            return InteractionProfile.from_mapping(payload)
        profile = response.get("interaction_profile")
        if isinstance(profile, dict):
            return InteractionProfile.from_mapping(profile)
        if previous:
            return InteractionProfile.from_mapping(previous.get("interaction_profile"))
        return InteractionProfile.default()

    @staticmethod
    def _looks_like_follow_up(normalized_prompt: str) -> bool:
        normalized = PromptSurfaceBuilder.build(normalized_prompt).lookup_ready_text
        if not normalized:
            return False
        anchor = detect_follow_up_anchor(normalized)
        if anchor is not None:
            return True
        if looks_like_reference_follow_up(normalized):
            return True
        return looks_like_general_follow_up(normalized) and normalized.startswith(("continue", "expand", "go deeper", "tell me more"))
