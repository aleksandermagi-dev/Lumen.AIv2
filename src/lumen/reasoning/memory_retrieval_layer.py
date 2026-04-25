from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lumen.memory.indexing import IndexedMemoryRecord, MemoryIndexBuilder
from lumen.memory.graph_memory import GraphMemoryManager
from lumen.nlu.focus_resolution import FocusResolutionSupport
from lumen.nlu.follow_up_inventory import looks_like_reference_follow_up
from lumen.nlu.models import PromptUnderstanding
from lumen.nlu.prompt_surface_builder import PromptSurfaceBuilder
from lumen.reasoning.continuation_confidence_policy import ContinuationConfidencePolicy
from lumen.reasoning.memory_ranking_signals import MemoryRankingSignals
from lumen.reasoning.pipeline_models import RetrievalAdvisoryContext


@dataclass(slots=True)
class RetrievedMemory:
    source: str
    memory_kind: str
    label: str
    summary: str
    relevance: float
    metadata: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "memory_kind": self.memory_kind,
            "label": self.label,
            "summary": self.summary,
            "relevance": round(float(self.relevance), 4),
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class MemoryRetrievalResult:
    query: str
    selected: list[RetrievedMemory]
    memory_reply_hint: str | None = None
    recall_prompt: bool = False
    project_return_prompt: bool = False
    project_reply_hint: str | None = None
    diagnostics: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "selected": [item.to_dict() for item in self.selected],
            "memory_reply_hint": self.memory_reply_hint,
            "recall_prompt": self.recall_prompt,
            "project_return_prompt": self.project_return_prompt,
            "project_reply_hint": self.project_reply_hint,
            "diagnostics": dict(self.diagnostics or {}),
        }

    def to_advisory_context(self) -> RetrievalAdvisoryContext:
        return RetrievalAdvisoryContext(
            query=self.query,
            selected=tuple(item.to_dict() for item in self.selected),
            memory_reply_hint=self.memory_reply_hint,
            recall_prompt=self.recall_prompt,
            project_return_prompt=self.project_return_prompt,
            project_reply_hint=self.project_reply_hint,
            diagnostics=dict(self.diagnostics or {}),
        )


class MemoryRetrievalLayer:
    """Reads current memory sources, ranks them, and prepares reply-safe memory context."""

    EXPLICIT_KNOWLEDGE_PREFIXES = (
        "tell me about ",
        "what is ",
        "what are ",
        "what's ",
        "whats ",
        "explain ",
        "describe ",
        "define ",
        "could you explain ",
        "can you explain ",
        "what do you know about ",
    )

    def __init__(
        self,
        *,
        interaction_history_service,
        archive_service,
        graph_memory_manager: GraphMemoryManager,
        personal_memory_manager=None,
        research_note_manager=None,
        persistence_manager=None,
        semantic_embedding_service=None,
    ) -> None:
        self.interaction_history_service = interaction_history_service
        self.archive_service = archive_service
        self.graph_memory_manager = graph_memory_manager
        self.personal_memory_manager = personal_memory_manager
        self.research_note_manager = research_note_manager
        self.persistence_manager = persistence_manager
        self.semantic_embedding_service = semantic_embedding_service

    def retrieve(
        self,
        *,
        prompt: str,
        session_id: str,
        project_id: str | None = None,
        active_thread: dict[str, object] | None,
        route_mode: str,
        recent_interactions: list[dict[str, object]] | None = None,
        prompt_understanding: PromptUnderstanding | None = None,
    ) -> MemoryRetrievalResult:
        # Retrieval may inherit a continuation target, but this layer must never alter the chosen route.
        continuation = ContinuationConfidencePolicy.evaluate(
            prompt=prompt,
            recent_interactions=list(recent_interactions or []),
        )
        retrieval_prompt = continuation.target_prompt
        normalized = (
            prompt_understanding.surface_views.lookup_ready_text
            if prompt_understanding is not None
            and str(prompt_understanding.original_text or "").strip() == str(retrieval_prompt or "").strip()
            else PromptSurfaceBuilder.build(retrieval_prompt).lookup_ready_text
        )
        recall_prompt = self._is_memory_recall_prompt(normalized)
        project_return_prompt = self._is_project_return_prompt(normalized)
        diagnostics: dict[str, object] = {"normalized_prompt": normalized}
        if recall_prompt:
            recall_resolution = FocusResolutionSupport.recall_focus(retrieval_prompt)
            recall_focus = recall_resolution.focus
            diagnostics["focus_resolution"] = recall_resolution.diagnostics()
        elif project_return_prompt:
            project_resolution = FocusResolutionSupport.project_return_focus(retrieval_prompt)
            recall_focus = project_resolution.focus
            diagnostics["focus_resolution"] = project_resolution.diagnostics()
        else:
            recall_focus = normalized
            diagnostics["focus_resolution"] = {
                "normalized_prompt": normalized,
                "focus": recall_focus,
                "wrappers_removed": [],
                "reason": "direct_focus",
            }
        require_focus_overlap = self._requires_focus_overlap(
            normalized_prompt=normalized,
            recall_prompt=recall_prompt,
            project_return_prompt=project_return_prompt,
        )
        effective_project_id = project_id or self._project_id_for_session(session_id)
        diagnostics["project_id"] = effective_project_id
        diagnostics["continuity_window_used"] = min(len(list(recent_interactions or [])), 5)
        candidates: list[RetrievedMemory] = []

        active_thread_candidate = self._active_thread_candidate(
            normalized_prompt=normalized,
            recall_focus=recall_focus,
            active_thread=active_thread,
            recall_prompt=recall_prompt,
            project_return_prompt=project_return_prompt,
            require_focus_overlap=require_focus_overlap,
        )
        if active_thread_candidate is not None:
            candidates.append(active_thread_candidate)

        candidates.extend(
            self._recent_interaction_candidates(
                recall_focus=recall_focus,
                recall_prompt=recall_prompt,
                project_return_prompt=project_return_prompt,
                require_focus_overlap=require_focus_overlap,
                recent_interactions=list(recent_interactions or []),
                session_id=session_id,
                current_project_id=effective_project_id,
            )
        )
        candidates.extend(
            self._recent_summary_candidates(
                recall_focus=recall_focus,
                recall_prompt=recall_prompt,
                project_return_prompt=project_return_prompt,
                require_focus_overlap=require_focus_overlap,
                session_id=session_id,
                project_id=effective_project_id,
            )
        )
        candidates.extend(
            self._message_window_candidates(
                recall_focus=recall_focus,
                recall_prompt=recall_prompt,
                project_return_prompt=project_return_prompt,
                require_focus_overlap=require_focus_overlap,
                session_id=session_id,
                project_id=effective_project_id,
            )
        )

        interaction_context = self.interaction_history_service.retrieve_context(
            retrieval_prompt,
            session_id=session_id,
            project_id=effective_project_id,
            limit=3,
        )
        candidates.extend(
            self._interaction_candidates(
                normalized_prompt=normalized,
                recall_focus=recall_focus,
                recall_prompt=recall_prompt,
                project_return_prompt=project_return_prompt,
                require_focus_overlap=require_focus_overlap,
                context=interaction_context,
                current_project_id=effective_project_id,
            )
        )

        archive_context = self.archive_service.retrieve_context(
            retrieval_prompt,
            session_id=session_id,
            project_id=effective_project_id,
            limit=3,
        )
        candidates.extend(
            self._archive_candidates(
                normalized_prompt=normalized,
                recall_focus=recall_focus,
                recall_prompt=recall_prompt,
                project_return_prompt=project_return_prompt,
                require_focus_overlap=require_focus_overlap,
                context=archive_context,
                current_project_id=effective_project_id,
            )
        )

        # Retrieval may inherit the already-answered topic, but it still cannot change the selected route.
        graph_matches = self.graph_memory_manager.search_nodes(recall_focus or retrieval_prompt, limit=5)
        candidates.extend(
            self._graph_candidates(
                recall_focus=recall_focus,
                recall_prompt=recall_prompt,
                project_return_prompt=project_return_prompt,
                require_focus_overlap=require_focus_overlap,
                matches=graph_matches,
            )
        )
        candidates.extend(
            self._personal_memory_candidates(
                recall_focus=recall_focus,
                recall_prompt=recall_prompt,
                require_focus_overlap=require_focus_overlap,
                session_id=session_id,
                project_id=effective_project_id,
            )
        )
        candidates.extend(
            self._research_note_candidates(
                recall_focus=recall_focus,
                recall_prompt=recall_prompt,
                project_return_prompt=project_return_prompt,
                require_focus_overlap=require_focus_overlap,
                session_id=session_id,
                project_id=effective_project_id,
            )
        )

        pre_semantic_order = [
            (item.source, item.label.lower())
            for item in self._ranked_candidates(
                candidates,
                recall_prompt=recall_prompt,
                project_return_prompt=project_return_prompt,
            )
        ]
        self._apply_semantic_scoring(
            candidates,
            query=retrieval_prompt,
            normalized_prompt=normalized,
            project_id=effective_project_id,
            recall_prompt=recall_prompt,
            project_return_prompt=project_return_prompt,
        )

        selected = self._select(
            candidates,
            route_mode=route_mode,
            recall_prompt=recall_prompt,
            project_return_prompt=project_return_prompt,
        )
        diagnostics["candidate_count"] = len(candidates)
        diagnostics["selected_count"] = len(selected)
        diagnostics["selected_sources"] = [item.source for item in selected]
        diagnostics["candidate_origins"] = self._candidate_origin_counts(candidates)
        diagnostics["continuity_buckets"] = self._continuity_bucket_counts(candidates)
        diagnostics["semantic_changed_ordering"] = pre_semantic_order != [
            (item.source, item.label.lower())
            for item in self._ranked_candidates(
                candidates,
                recall_prompt=recall_prompt,
                project_return_prompt=project_return_prompt,
            )
        ]
        diagnostics["selected_reasons"] = [
            {
                "source": item.source,
                "label": item.label,
                "reason": str(item.metadata.get("selection_reason") or "selected"),
                "continuity_bucket": str(item.metadata.get("continuity_bucket") or "unclassified"),
                "project_match": bool(item.metadata.get("project_match")),
                "session_continuity": bool(item.metadata.get("session_continuity")),
                "semantic_similarity": round(float(item.metadata.get("semantic_similarity") or 0.0), 4),
                "semantic_bonus": round(float(item.metadata.get("semantic_bonus") or 0.0), 4),
                "semantic_used": bool(item.metadata.get("semantic_used")),
                "semantic_status": str(item.metadata.get("semantic_status") or "not_used"),
            }
            for item in selected
        ]
        diagnostics["rejected_reasons"] = self._rejected_reason_summary(candidates=candidates, selected=selected)
        if not selected:
            diagnostics["reason"] = (
                "no_subject_resolution"
                if not recall_focus
                else "weak_focus_overlap"
                if candidates
                else "insufficient_dominance"
            )
        elif len(selected) > 1:
            diagnostics["reason"] = "candidate_conflict"
        else:
            diagnostics["reason"] = "selected"
        return MemoryRetrievalResult(
            query=prompt,
            selected=selected,
            memory_reply_hint=self._reply_hint(
                selected,
                recall_prompt=recall_prompt,
                project_return_prompt=project_return_prompt,
            ),
            recall_prompt=recall_prompt,
            project_return_prompt=project_return_prompt,
            project_reply_hint=self._project_reply_hint(selected) if project_return_prompt else None,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _recent_interaction_candidates(
        *,
        recall_focus: str,
        recall_prompt: bool,
        project_return_prompt: bool,
        require_focus_overlap: bool,
        recent_interactions: list[dict[str, object]],
        session_id: str,
        current_project_id: str | None,
    ) -> list[RetrievedMemory]:
        results: list[RetrievedMemory] = []
        for item in recent_interactions[:5]:
            if not isinstance(item, dict):
                continue
            prompt = str(item.get("prompt") or "").strip()
            summary = str(item.get("summary") or "").strip()
            candidate_text = summary or prompt
            if not candidate_text:
                continue
            focus_overlap = MemoryRetrievalLayer._focus_overlap(
                recall_focus,
                candidate_text,
                str(item.get("normalized_topic") or ""),
            )
            if require_focus_overlap and recall_focus and focus_overlap <= 0:
                continue
            if (recall_prompt or project_return_prompt) and recall_focus and focus_overlap <= 0:
                continue
            project_match = bool(
                current_project_id and str(item.get("project_id") or "").strip() == current_project_id
            )
            relevance = 0.48 + (focus_overlap * 0.09)
            if str(item.get("session_id") or "").strip() == session_id:
                relevance += 0.1
            if project_match:
                relevance += 0.08
            results.append(
                RetrievedMemory(
                    source="recent_interactions",
                    memory_kind="ephemeral_context",
                    label=str(item.get("normalized_topic") or item.get("kind") or "recent interaction").strip()
                    or "recent interaction",
                    summary=candidate_text,
                    relevance=min(relevance, 0.9),
                    metadata={
                        "created_at": str(item.get("created_at") or "").strip() or None,
                        "project_match": project_match,
                        "session_continuity": str(item.get("session_id") or "").strip() == session_id,
                        "selection_reason": "recent_interaction_window",
                        "continuity_bucket": "recent_interaction_window",
                        "continuity_rank": 5.0,
                        "source_reliability": 0.78,
                        "focus_overlap": focus_overlap,
                    },
                )
            )
        return results

    def _recent_summary_candidates(
        self,
        *,
        recall_focus: str,
        recall_prompt: bool,
        project_return_prompt: bool,
        require_focus_overlap: bool,
        session_id: str,
        project_id: str | None,
    ) -> list[RetrievedMemory]:
        persistence = self.persistence_manager
        summaries = getattr(persistence, "session_summaries", None) if persistence is not None else None
        if summaries is None:
            return []
        rows: list[dict[str, object]] = []
        seen: set[str] = set()
        loaders = [lambda: summaries.list_recent_by_session(session_id, limit=2)]
        if project_id:
            loaders.append(lambda: summaries.list_recent_by_project(project_id, limit=3))
        for loader in loaders:
            try:
                items = loader()
            except Exception:
                items = []
            for row in items:
                if not isinstance(row, dict):
                    continue
                row_id = str(row.get("id") or "")
                if row_id and row_id in seen:
                    continue
                if row_id:
                    seen.add(row_id)
                rows.append(row)
        results: list[RetrievedMemory] = []
        for row in rows[:4]:
            summary = str(row.get("summary_text") or "").strip()
            if not summary:
                continue
            metadata_payload = row.get("metadata_json") if isinstance(row.get("metadata_json"), dict) else {}
            label = str(
                metadata_payload.get("objective")
                or metadata_payload.get("prompt")
                or row.get("summary_scope")
                or "recent session summary"
            ).strip()
            focus_overlap = self._focus_overlap(recall_focus, summary, label)
            if require_focus_overlap and recall_focus and focus_overlap <= 0:
                continue
            if (recall_prompt or project_return_prompt) and recall_focus and focus_overlap <= 0:
                continue
            session_continuity = str(row.get("session_id") or "").strip() == session_id
            relevance = 0.54 + (focus_overlap * 0.08)
            if session_continuity:
                relevance += 0.12
            elif project_id:
                relevance += 0.06
            if project_return_prompt:
                relevance += 0.08
            results.append(
                RetrievedMemory(
                    source="session_summaries",
                    memory_kind="active_project_memory",
                    label=label or "recent session summary",
                    summary=summary,
                    relevance=min(relevance, 0.92),
                    metadata={
                        "created_at": str(row.get("created_at") or "").strip() or None,
                        "project_match": bool(project_id),
                        "session_continuity": session_continuity,
                        "selection_reason": "recent_summary_window",
                        "continuity_bucket": "recent_summary_window",
                        "continuity_rank": 4.0 if session_continuity else 3.4,
                        "source_reliability": 0.82,
                        "focus_overlap": focus_overlap,
                        "summary_scope": str(row.get("summary_scope") or "").strip() or None,
                    },
                )
            )
        return results

    def _message_window_candidates(
        self,
        *,
        recall_focus: str,
        recall_prompt: bool,
        project_return_prompt: bool,
        require_focus_overlap: bool,
        session_id: str,
        project_id: str | None,
    ) -> list[RetrievedMemory]:
        persistence = self.persistence_manager
        messages = getattr(persistence, "messages", None) if persistence is not None else None
        if messages is None:
            return []
        rows: list[dict[str, object]] = []
        seen: set[str] = set()
        loaders = [lambda: messages.list_message_window_by_session(session_id, limit=4)]
        if project_id:
            loaders.append(lambda: messages.list_message_window_by_project(project_id, limit=6))
        for loader in loaders:
            try:
                items = loader()
            except Exception:
                items = []
            for row in items:
                if not isinstance(row, dict):
                    continue
                row_id = str(row.get("id") or "")
                if row_id and row_id in seen:
                    continue
                if row_id:
                    seen.add(row_id)
                rows.append(row)
        results: list[RetrievedMemory] = []
        for row in rows[:5]:
            content = str(row.get("content") or "").strip()
            if not content:
                continue
            metadata_payload = row.get("message_metadata_json") if isinstance(row.get("message_metadata_json"), dict) else {}
            label = str(
                metadata_payload.get("resolved_prompt")
                or metadata_payload.get("original_prompt")
                or row.get("intent_domain")
                or "recent message window"
            ).strip()
            focus_overlap = self._focus_overlap(recall_focus, content, label)
            if require_focus_overlap and recall_focus and focus_overlap <= 0:
                continue
            if (recall_prompt or project_return_prompt) and recall_focus and focus_overlap <= 0:
                continue
            session_continuity = str(row.get("session_id") or "").strip() == session_id
            relevance = 0.47 + (focus_overlap * 0.07)
            if session_continuity:
                relevance += 0.11
            elif project_id:
                relevance += 0.04
            if project_return_prompt:
                relevance += 0.05
            results.append(
                RetrievedMemory(
                    source="message_window",
                    memory_kind="active_project_memory",
                    label=label or "recent message window",
                    summary=content,
                    relevance=min(relevance, 0.88),
                    metadata={
                        "created_at": str(row.get("created_at") or "").strip() or None,
                        "project_match": bool(project_id),
                        "session_continuity": session_continuity,
                        "selection_reason": "message_window_continuity",
                        "continuity_bucket": "recent_message_window",
                        "continuity_rank": 3.8 if session_continuity else 3.1,
                        "source_reliability": 0.76,
                        "focus_overlap": focus_overlap,
                        "role": str(row.get("role") or "").strip() or None,
                    },
                )
            )
        return results

    @staticmethod
    def _supports_semantic_scoring(item: RetrievedMemory) -> bool:
        if item.source not in {"personal_memory", "research_notes"}:
            return False
        metadata = item.metadata or {}
        return bool(metadata.get("source_path"))

    @staticmethod
    def _source_type_for_item(item: RetrievedMemory) -> str | None:
        if item.source == "personal_memory":
            return "personal_memory"
        if item.source == "research_notes":
            return "research_note"
        return None

    @staticmethod
    def _broader_recall_requested(normalized_prompt: str) -> bool:
        text = str(normalized_prompt or "").lower()
        cues = ("across projects", "any project", "in general", "broader recall", "globally")
        return any(cue in text for cue in cues)

    def _apply_semantic_scoring(
        self,
        candidates: list[RetrievedMemory],
        *,
        query: str,
        normalized_prompt: str,
        project_id: str | None,
        recall_prompt: bool,
        project_return_prompt: bool,
    ) -> None:
        if not candidates or self.persistence_manager is None:
            return
        service = self.semantic_embedding_service or getattr(self.persistence_manager, "semantic_embedding_service", None)
        embeddings = getattr(self.persistence_manager, "memory_embeddings", None)
        if service is None or embeddings is None:
            return
        if not service.is_available():
            for item in candidates:
                if self._supports_semantic_scoring(item):
                    item.metadata.setdefault("semantic_skipped_reason", "runtime_unavailable")
                    item.metadata.setdefault("semantic_used", False)
                    item.metadata.setdefault("semantic_status", "suppressed:runtime_unavailable")
            return
        eligible = [item for item in candidates if self._supports_semantic_scoring(item)]
        if not eligible:
            return
        try:
            query_vector = service.embed_text(query)
        except Exception:
            for item in eligible:
                item.metadata.setdefault("semantic_skipped_reason", "query_embedding_failed")
                item.metadata.setdefault("semantic_used", False)
                item.metadata.setdefault("semantic_status", "suppressed:query_embedding_failed")
            return
        broader_recall = self._broader_recall_requested(normalized_prompt)
        for item in eligible:
            metadata = item.metadata
            source_type = self._source_type_for_item(item)
            source_id = str(metadata.get("source_path") or "").strip()
            if not source_type or not source_id:
                metadata.setdefault("semantic_skipped_reason", "missing_source_reference")
                metadata.setdefault("semantic_used", False)
                metadata.setdefault("semantic_status", "suppressed:missing_source_reference")
                continue
            embedding_row = embeddings.get_by_source(source_type=source_type, source_id=source_id)
            if not isinstance(embedding_row, dict):
                metadata.setdefault("semantic_skipped_reason", "embedding_missing")
                metadata.setdefault("semantic_used", False)
                metadata.setdefault("semantic_status", "suppressed:embedding_missing")
                continue
            if str(embedding_row.get("status") or "") != "ready" or embedding_row.get("embedding_blob") is None:
                metadata["semantic_skipped_reason"] = str(embedding_row.get("status") or "embedding_unavailable")
                metadata["semantic_used"] = False
                metadata["semantic_status"] = f"suppressed:{metadata['semantic_skipped_reason']}"
                continue
            candidate_vector = service.unpack_embedding(embedding_row.get("embedding_blob"))
            similarity = float(service.cosine_similarity(query_vector, candidate_vector))
            bonus = self._semantic_bonus(
                similarity=similarity,
                project_match=bool(metadata.get("project_match")),
                session_continuity=bool(metadata.get("session_continuity")),
                broader_recall=broader_recall,
                recall_prompt=recall_prompt,
                project_return_prompt=project_return_prompt,
                source=item.source,
                active_project_id=project_id,
            )
            metadata["semantic_similarity"] = round(similarity, 4)
            metadata["semantic_bonus"] = round(bonus, 4)
            metadata["semantic_model"] = str(embedding_row.get("model_name") or service.model_name)
            metadata["semantic_used"] = bonus > 0
            if bonus <= 0:
                if similarity < 0.35:
                    metadata["semantic_skipped_reason"] = "below_similarity_floor"
                elif active_project_id := project_id:
                    if not bool(metadata.get("project_match")) and not broader_recall:
                        metadata["semantic_skipped_reason"] = "cross_project_dampened"
                    else:
                        metadata["semantic_skipped_reason"] = "no_bonus_band"
                else:
                    metadata["semantic_skipped_reason"] = "no_bonus_band"
                metadata["semantic_status"] = f"suppressed:{metadata['semantic_skipped_reason']}"
                continue
            metadata["base_relevance"] = round(float(item.relevance), 4)
            metadata["selection_reason"] = str(metadata.get("selection_reason") or "memory_match")
            item.relevance = min(0.99, float(item.relevance) + bonus)
            metadata["final_relevance"] = round(float(item.relevance), 4)
            metadata["semantic_status"] = "applied:capped" if bonus >= 0.12 else "applied"

    @staticmethod
    def _semantic_bonus(
        *,
        similarity: float,
        project_match: bool,
        session_continuity: bool,
        broader_recall: bool,
        recall_prompt: bool,
        project_return_prompt: bool,
        source: str,
        active_project_id: str | None,
    ) -> float:
        if similarity < 0.35:
            return 0.0
        if similarity >= 0.8:
            bonus = 0.12
        elif similarity >= 0.65:
            bonus = 0.08
        elif similarity >= 0.5:
            bonus = 0.05
        else:
            bonus = 0.02
        if session_continuity:
            bonus += 0.02
        if project_match:
            bonus += 0.015
        elif active_project_id and not broader_recall:
            bonus *= 0.35
        if recall_prompt and source == "personal_memory":
            bonus += 0.01
        if project_return_prompt and source == "research_notes":
            bonus += 0.01
        if not project_match and active_project_id and not broader_recall:
            bonus = min(bonus, 0.04)
        return max(0.0, min(0.12, bonus))

    @staticmethod
    def _rank_tuple(
        item: RetrievedMemory,
        *,
        recall_prompt: bool,
        project_return_prompt: bool,
    ) -> tuple[float, ...]:
        return (
            item.relevance,
            1 if bool(item.metadata.get("project_match")) else 0,
            1 if bool(item.metadata.get("session_continuity")) else 0,
            float(item.metadata.get("continuity_rank") or 0.0),
            1 if bool(item.metadata.get("reaffirmed")) else 0,
            float(item.metadata.get("semantic_bonus") or 0.0),
            float(item.metadata.get("source_reliability") or 0.0),
            float(item.metadata.get("decay_factor") or 0.0),
            1.0 - float(item.metadata.get("contradiction_penalty") or 0.0),
            1.0 - float(item.metadata.get("generic_label_penalty") or 0.0),
            1 if project_return_prompt and item.source == "active_thread" else 0,
            1 if item.source == "research_notes" and project_return_prompt else 0,
            1 if item.source == "recent_interactions" else 0,
            1 if item.source == "personal_memory" and recall_prompt else 0,
            1 if item.memory_kind == "durable_user_memory" and recall_prompt else 0,
            1 if item.memory_kind == "durable_project_memory" and project_return_prompt else 0,
            1 if item.memory_kind == "active_project_memory" else 0,
        )

    @classmethod
    def _ranked_candidates(
        cls,
        candidates: list[RetrievedMemory],
        *,
        recall_prompt: bool,
        project_return_prompt: bool,
    ) -> list[RetrievedMemory]:
        return sorted(
            candidates,
            key=lambda item: cls._rank_tuple(
                item,
                recall_prompt=recall_prompt,
                project_return_prompt=project_return_prompt,
            ),
            reverse=True,
        )

    @staticmethod
    def _is_memory_recall_prompt(normalized_prompt: str) -> bool:
        if not normalized_prompt:
            return False
        recall_cues = (
            "what do you remember",
            "what do we remember",
            "what do you know about my preference",
            "what do you remember about",
            "what do you have on",
            "remember about",
        )
        return any(cue in normalized_prompt for cue in recall_cues)

    @staticmethod
    def _is_project_return_prompt(normalized_prompt: str) -> bool:
        if not normalized_prompt:
            return False
        project_return_cues = (
            "back to ",
            "where were we on ",
            "continue the ",
            "continue with ",
            "what was our last take on ",
            "pick back up ",
            "pick back up on ",
            "return to ",
            "resume the ",
        )
        return any(cue in normalized_prompt for cue in project_return_cues)

    @staticmethod
    def _project_return_focus(normalized_prompt: str) -> str:
        return FocusResolutionSupport.project_return_focus(normalized_prompt).focus

    def _active_thread_candidate(
        self,
        *,
        normalized_prompt: str,
        recall_focus: str,
        active_thread: dict[str, object] | None,
        recall_prompt: bool,
        project_return_prompt: bool,
        require_focus_overlap: bool,
    ) -> RetrievedMemory | None:
        if active_thread is None:
            return None
        thread_summary = str(
            active_thread.get("thread_summary")
            or active_thread.get("summary")
            or active_thread.get("objective")
            or ""
        ).strip()
        if not thread_summary:
            return None
        topic = str(active_thread.get("normalized_topic") or "").strip()
        overlap_bonus = 0.0
        if topic and any(token in thread_summary.lower() for token in topic.lower().split()):
            overlap_bonus += 0.15
        if looks_like_reference_follow_up(normalized_prompt):
            overlap_bonus += 0.25
        focus_overlap = self._focus_overlap(recall_focus, thread_summary, topic)
        if require_focus_overlap and focus_overlap <= 0 and not looks_like_reference_follow_up(normalized_prompt):
            return None
        if recall_prompt and focus_overlap <= 0 and not looks_like_reference_follow_up(normalized_prompt):
            return None
        if project_return_prompt and focus_overlap <= 0 and not looks_like_reference_follow_up(normalized_prompt):
            return None
        if recall_prompt:
            overlap_bonus += 0.2
        if project_return_prompt:
            overlap_bonus += 0.2
        relevance = 0.6 + overlap_bonus + (focus_overlap * 0.08)
        return RetrievedMemory(
            source="active_thread",
            memory_kind="active_project_memory",
            label=topic or str(active_thread.get("kind") or "active thread"),
            summary=thread_summary,
            relevance=min(relevance, 1.0),
            metadata={
                "mode": str(active_thread.get("mode") or "").strip(),
                "kind": str(active_thread.get("kind") or "").strip(),
                "created_at": str(active_thread.get("updated_at") or active_thread.get("created_at") or "").strip() or None,
                "source_reliability": 0.96,
                "reaffirmed": focus_overlap >= 2,
                "focus_overlap": focus_overlap,
                "continuity_bucket": "active_thread",
                "continuity_rank": 6.0,
                "project_match": True,
                "session_continuity": True,
            },
        )

    @staticmethod
    def _interaction_candidates(
        *,
        normalized_prompt: str,
        recall_focus: str,
        recall_prompt: bool,
        project_return_prompt: bool,
        require_focus_overlap: bool,
        context: dict[str, object],
        current_project_id: str | None,
    ) -> list[RetrievedMemory]:
        results: list[RetrievedMemory] = []
        for match in list(context.get("top_interaction_matches") or [])[:3]:
            record = match.get("record")
            if not isinstance(record, dict):
                continue
            summary = str(record.get("summary") or record.get("prompt_view", {}).get("canonical_prompt") or "").strip()
            if not summary:
                continue
            score = int(match.get("score") or 0)
            focus_overlap = MemoryRetrievalLayer._focus_overlap(
                recall_focus,
                summary,
                str(record.get("normalized_topic") or ""),
            )
            if require_focus_overlap and recall_focus and focus_overlap <= 0:
                continue
            if recall_prompt and recall_focus and focus_overlap <= 0:
                continue
            if project_return_prompt and recall_focus and focus_overlap <= 0:
                continue
            relevance = min(0.8, 0.25 + (score * 0.08) + (0.12 if recall_prompt else 0.0))
            if project_return_prompt and focus_overlap > 0:
                relevance = min(0.9, relevance + 0.18)
            project_match = bool(current_project_id and str(record.get("project_id") or "").strip() == current_project_id)
            if project_match:
                relevance = min(0.96, relevance + 0.1)
            elif current_project_id:
                relevance = max(0.0, relevance - 0.12)
            relevance = min(0.9, relevance + (focus_overlap * 0.07))
            relevance, metadata = MemoryRankingSignals.score_retrieved_memory(
                source="interaction_history",
                label=str(record.get("normalized_topic") or record.get("kind") or "interaction").strip() or "interaction",
                summary=summary,
                base_relevance=relevance,
                metadata={
                    "mode": str(record.get("mode") or "").strip(),
                    "kind": str(record.get("kind") or "").strip(),
                    "score": score,
                    "created_at": str(record.get("created_at") or "").strip() or None,
                    "source_reliability": 0.72,
                    "project_match": project_match,
                    "session_continuity": True,
                    "selection_reason": "session_or_project_history",
                    "continuity_bucket": "historical_interaction",
                    "continuity_rank": 2.8 if project_match else 1.8,
                },
                recall_focus=recall_focus,
                recall_prompt=recall_prompt,
                project_return_prompt=project_return_prompt,
                normalized_prompt=normalized_prompt,
            )
            results.append(
                RetrievedMemory(
                    source="interaction_history",
                    memory_kind="ephemeral_context",
                    label=str(record.get("normalized_topic") or record.get("kind") or "interaction").strip() or "interaction",
                    summary=summary,
                    relevance=relevance,
                    metadata=metadata,
                )
            )
        return results

    @staticmethod
    def _archive_candidates(
        *,
        normalized_prompt: str,
        recall_focus: str,
        recall_prompt: bool,
        project_return_prompt: bool,
        require_focus_overlap: bool,
        context: dict[str, object],
        current_project_id: str | None,
    ) -> list[RetrievedMemory]:
        results: list[RetrievedMemory] = []
        for match in list(context.get("top_matches") or [])[:3]:
            record = match.get("record")
            if not isinstance(record, dict):
                continue
            summary = str(record.get("summary") or "").strip()
            if not summary:
                continue
            score = int(match.get("score") or 0)
            focus_overlap = MemoryRetrievalLayer._focus_overlap(
                recall_focus,
                summary,
                str(record.get("capability") or record.get("tool_id") or ""),
            )
            if require_focus_overlap and recall_focus and focus_overlap <= 0:
                continue
            if recall_prompt and recall_focus and focus_overlap <= 0:
                continue
            if project_return_prompt and recall_focus and focus_overlap <= 0:
                continue
            relevance = min(0.82, 0.2 + (score * 0.07) + (0.1 if recall_prompt else 0.0) + (focus_overlap * 0.06))
            if project_return_prompt and focus_overlap > 0:
                relevance = min(0.94, relevance + 0.16)
            project_match = bool(current_project_id and str(record.get("project_id") or "").strip() == current_project_id)
            if project_match:
                relevance = min(0.97, relevance + 0.1)
            elif current_project_id:
                relevance = max(0.0, relevance - 0.14)
            relevance, metadata = MemoryRankingSignals.score_retrieved_memory(
                source="archive",
                label=str(record.get("capability") or record.get("tool_id") or "archive").strip() or "archive",
                summary=summary,
                base_relevance=relevance,
                metadata={
                    "tool_id": str(record.get("tool_id") or "").strip(),
                    "capability": str(record.get("capability") or "").strip(),
                    "score": score,
                    "created_at": str(record.get("created_at") or "").strip() or None,
                    "source_reliability": 0.68,
                    "project_match": project_match,
                    "selection_reason": "archived_tool_context",
                    "continuity_bucket": "archived_tool_context",
                    "continuity_rank": 2.4 if project_match else 1.4,
                },
                recall_focus=recall_focus,
                recall_prompt=recall_prompt,
                project_return_prompt=project_return_prompt,
                normalized_prompt=normalized_prompt,
            )
            results.append(
                RetrievedMemory(
                    source="archive",
                    memory_kind="active_project_memory",
                    label=str(record.get("capability") or record.get("tool_id") or "archive").strip() or "archive",
                    summary=summary,
                    relevance=relevance,
                    metadata=metadata,
                )
            )
        return results

    def _graph_candidates(
        self,
        *,
        recall_focus: str,
        recall_prompt: bool,
        project_return_prompt: bool,
        require_focus_overlap: bool,
        matches: list[dict[str, object]],
    ) -> list[RetrievedMemory]:
        results: list[RetrievedMemory] = []
        for match in matches[:4]:
            opened = self.graph_memory_manager.open_nodes(ids=[int(match["id"])])
            if not opened:
                continue
            node = opened[0]
            observations = list(node.get("observations") or [])
            relations_out = list(node.get("relations_out") or [])
            relations_in = list(node.get("relations_in") or [])
            summary = str(match.get("observation_preview") or "").strip()
            if not summary and observations:
                summary = str(observations[0].get("content") or "").strip()
            if not summary:
                summary = str(node.get("name") or "").strip()
            related = self._related_memory_hint(relations_out=relations_out, relations_in=relations_in)
            if related:
                summary = f"{summary} {related}".strip()
            entity_type = str(node.get("entity_type") or "note").strip()
            memory_kind = "durable_user_memory" if entity_type in {"person", "preference"} else "durable_project_memory"
            score = float(match.get("score") or 0.0)
            label = str(node.get("name") or "").strip() or entity_type
            focus_overlap = self._focus_overlap(
                recall_focus,
                summary,
                label,
                entity_type,
            )
            if require_focus_overlap and recall_focus and focus_overlap <= 0:
                continue
            if (recall_prompt or project_return_prompt) and recall_focus and focus_overlap <= 0:
                continue
            relevance = min(0.98, 0.3 + (score * 0.08) + (0.12 if recall_prompt else 0.0) + (focus_overlap * 0.08))
            if project_return_prompt and memory_kind == "durable_project_memory":
                relevance = min(0.98, relevance + 0.12)
                relevance = max(0.0, relevance - self._generic_project_label_penalty(label))
            results.append(
                RetrievedMemory(
                    source="graph_memory",
                    memory_kind=memory_kind,
                    label=label,
                    summary=summary,
                    relevance=relevance,
                    metadata={
                        "entity_type": entity_type,
                        "node_id": int(node.get("id") or 0),
                    },
                )
            )
        return results

    def _personal_memory_candidates(
        self,
        *,
        recall_focus: str,
        recall_prompt: bool,
        require_focus_overlap: bool,
        session_id: str,
        project_id: str | None,
    ) -> list[RetrievedMemory]:
        if self.personal_memory_manager is None:
            return []
        entries = self.personal_memory_manager.list_entries(
            session_id=session_id,
            project_id=project_id,
            include_archived=False,
        )
        results: list[RetrievedMemory] = []
        for entry in entries[:8]:
            indexed = MemoryIndexBuilder.from_personal_entry(entry)
            if indexed is None:
                continue
            focus_overlap = self._focus_overlap(
                recall_focus,
                indexed.summary,
                indexed.label,
                *indexed.domain_tags,
            )
            if require_focus_overlap and recall_focus and focus_overlap <= 0:
                continue
            if recall_prompt and recall_focus and focus_overlap <= 0:
                continue
            relevance = min(
                0.94,
                float(indexed.relevance_hint),
            )
            relevance, metadata = MemoryRankingSignals.score_indexed_memory(
                indexed=indexed,
                recall_focus=recall_focus,
                recall_prompt=recall_prompt,
                project_return_prompt=False,
                normalized_prompt=recall_focus,
            )
            entry_project_id = str(entry.get("project_id") or "").strip()
            metadata["project_match"] = bool(project_id and entry_project_id and entry_project_id == project_id)
            metadata["session_continuity"] = bool(
                session_id and str(entry.get("session_id") or "").strip() == session_id
            )
            metadata.setdefault("continuity_bucket", "personal_memory")
            metadata.setdefault("continuity_rank", 2.2 if metadata["project_match"] else 1.6)
            results.append(indexed_to_memory(indexed=indexed, relevance=relevance, metadata=metadata))
        return results

    def _research_note_candidates(
        self,
        *,
        recall_focus: str,
        recall_prompt: bool,
        project_return_prompt: bool,
        require_focus_overlap: bool,
        session_id: str,
        project_id: str | None,
    ) -> list[RetrievedMemory]:
        if self.research_note_manager is None:
            return []
        notes = self.research_note_manager.list_notes(
            session_id=session_id,
            project_id=project_id,
            include_archived=False,
        )
        results: list[RetrievedMemory] = []
        for note in notes[:10]:
            indexed = MemoryIndexBuilder.from_research_note(note)
            if indexed is None:
                continue
            focus_overlap = self._focus_overlap(
                recall_focus,
                indexed.summary,
                indexed.label,
                *indexed.domain_tags,
            )
            if require_focus_overlap and recall_focus and focus_overlap <= 0:
                continue
            if (recall_prompt or project_return_prompt) and recall_focus and focus_overlap <= 0:
                continue
            relevance = min(
                0.95,
                float(indexed.relevance_hint),
            )
            relevance, metadata = MemoryRankingSignals.score_indexed_memory(
                indexed=indexed,
                recall_focus=recall_focus,
                recall_prompt=recall_prompt,
                project_return_prompt=project_return_prompt,
                normalized_prompt=recall_focus,
            )
            note_project_id = str(note.get("project_id") or "").strip()
            metadata["project_match"] = bool(project_id and note_project_id and note_project_id == project_id)
            metadata["session_continuity"] = bool(
                session_id and str(note.get("session_id") or "").strip() == session_id
            )
            metadata.setdefault("continuity_bucket", "research_note")
            metadata.setdefault("continuity_rank", 2.6 if metadata["project_match"] else 1.7)
            results.append(indexed_to_memory(indexed=indexed, relevance=relevance, metadata=metadata))
        return results

    @classmethod
    def _requires_focus_overlap(
        cls,
        *,
        normalized_prompt: str,
        recall_prompt: bool,
        project_return_prompt: bool,
    ) -> bool:
        if recall_prompt or project_return_prompt:
            return False
        if looks_like_reference_follow_up(normalized_prompt):
            return False
        return any(
            normalized_prompt.startswith(prefix)
            for prefix in cls.EXPLICIT_KNOWLEDGE_PREFIXES
        )

    @staticmethod
    def _related_memory_hint(
        *,
        relations_out: list[dict[str, object]],
        relations_in: list[dict[str, object]],
    ) -> str | None:
        if relations_out:
            item = relations_out[0]
            relation_type = str(item.get("relation_type") or "").replace("_", " ").strip()
            target = str(item.get("target_name") or "").strip()
            if relation_type and target:
                return f"It {relation_type} {target}."
        if relations_in:
            item = relations_in[0]
            relation_type = str(item.get("relation_type") or "").replace("_", " ").strip()
            source = str(item.get("source_name") or "").strip()
            if relation_type and source:
                return f"It's linked from {source} via {relation_type}."
        return None

    @staticmethod
    def _select(
        candidates: list[RetrievedMemory],
        *,
        route_mode: str,
        recall_prompt: bool,
        project_return_prompt: bool,
    ) -> list[RetrievedMemory]:
        ranked = MemoryRetrievalLayer._ranked_candidates(
            candidates,
            recall_prompt=recall_prompt,
            project_return_prompt=project_return_prompt,
        )
        selected: list[RetrievedMemory] = []
        seen: set[tuple[str, str]] = set()
        for item in ranked:
            key = (item.source, item.label.lower())
            if key in seen:
                continue
            age_bucket = str(item.metadata.get("age_bucket") or "").strip()
            reaffirmed = bool(item.metadata.get("reaffirmed"))
            contradiction_penalty = float(item.metadata.get("contradiction_penalty") or 0.0)
            generic_penalty = float(item.metadata.get("generic_label_penalty") or 0.0)
            if recall_prompt and item.relevance < 0.5:
                continue
            if age_bucket in {"stale", "old"} and not reaffirmed and item.relevance < 0.72:
                continue
            if contradiction_penalty >= 0.1:
                continue
            if project_return_prompt and generic_penalty >= 0.08 and not reaffirmed and item.relevance < 0.8:
                continue
            if not recall_prompt and route_mode == "conversation" and item.relevance < 0.45:
                continue
            if (
                not recall_prompt
                and route_mode != "conversation"
                and item.memory_kind == "ephemeral_context"
            ):
                continue
            if project_return_prompt and item.memory_kind == "ephemeral_context" and not selected:
                continue
            selected.append(item)
            seen.add(key)
            if len(selected) >= 3:
                break
        return selected

    @staticmethod
    def _rejected_reason_summary(
        *,
        candidates: list[RetrievedMemory],
        selected: list[RetrievedMemory],
    ) -> list[dict[str, object]]:
        chosen = {(item.source, item.label.lower()) for item in selected}
        rejected: list[dict[str, object]] = []
        for item in candidates:
            key = (item.source, item.label.lower())
            if key in chosen:
                continue
            rejected.append(
                {
                    "source": item.source,
                    "label": item.label,
                    "reason": str(item.metadata.get("selection_reason") or "ranked_lower"),
                    "continuity_bucket": str(item.metadata.get("continuity_bucket") or "unclassified"),
                    "project_match": bool(item.metadata.get("project_match")),
                    "session_continuity": bool(item.metadata.get("session_continuity")),
                    "semantic_similarity": round(float(item.metadata.get("semantic_similarity") or 0.0), 4),
                    "semantic_bonus": round(float(item.metadata.get("semantic_bonus") or 0.0), 4),
                    "semantic_used": bool(item.metadata.get("semantic_used")),
                    "semantic_status": str(item.metadata.get("semantic_status") or "not_used"),
                    "semantic_skipped_reason": item.metadata.get("semantic_skipped_reason"),
                }
            )
            if len(rejected) >= 5:
                break
        return rejected

    @staticmethod
    def _candidate_origin_counts(candidates: list[RetrievedMemory]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in candidates:
            counts[item.source] = counts.get(item.source, 0) + 1
        return dict(sorted(counts.items()))

    @staticmethod
    def _continuity_bucket_counts(candidates: list[RetrievedMemory]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in candidates:
            bucket = str(item.metadata.get("continuity_bucket") or "unclassified")
            counts[bucket] = counts.get(bucket, 0) + 1
        return dict(sorted(counts.items()))

    @staticmethod
    def _reply_hint(
        selected: list[RetrievedMemory],
        *,
        recall_prompt: bool,
        project_return_prompt: bool,
    ) -> str | None:
        if not recall_prompt:
            return None
        if not selected:
            return "I don't have a strong memory match for that yet."
        user_memories = [item for item in selected if item.memory_kind == "durable_user_memory"]
        project_memories = [
            item for item in selected if item.memory_kind in {"durable_project_memory", "active_project_memory"}
        ]
        ephemeral = [item for item in selected if item.memory_kind == "ephemeral_context"]
        lines: list[str] = []
        if user_memories:
            item = user_memories[0]
            lines.append(f"You've told me {item.summary.rstrip('.')}.")
        if project_memories:
            item = project_memories[0]
            if item.source == "active_thread":
                label = item.label.strip()
                if label:
                    lines.append(f"We were working on {label}: {item.summary.rstrip('.')}.")
                else:
                    lines.append(f"We were working on {item.summary.rstrip('.')}.")
            else:
                lines.append(f"I have {item.label} in memory: {item.summary.rstrip('.')}.")
        elif ephemeral:
            item = ephemeral[0]
            lines.append(f"The closest earlier thread I have is: {item.summary.rstrip('.')}.")
        if not lines:
            return "I don't have a strong memory match for that yet."
        return " ".join(lines[:2]).strip()

    @staticmethod
    def _project_reply_hint(selected: list[RetrievedMemory]) -> str | None:
        project_memories = [
            item for item in selected if item.memory_kind in {"durable_project_memory", "active_project_memory"}
        ]
        if not project_memories:
            return None
        lead = project_memories[0]
        if lead.relevance < 0.72:
            return None
        if len(project_memories) > 1:
            runner_up = project_memories[1]
            if runner_up.label.lower() != lead.label.lower() and abs(lead.relevance - runner_up.relevance) < 0.08:
                return None
        label = lead.label.strip()
        summary = lead.summary.rstrip(".")
        if lead.source == "active_thread":
            if label:
                return f"We were still in {label}, mainly around {summary}."
            return f"We were still on that thread, mainly around {summary}."
        if label:
            return f"The closest thread I have is {label}, especially {summary}."
        return f"The closest thread I have is {summary}."

    @staticmethod
    def _generic_project_label_penalty(label: str) -> float:
        cleaned = " ".join(str(label or "").strip().lower().split())
        if not cleaned:
            return 0.0
        generic_markers = (
            "first pass",
            "assumptions",
            "summary",
            "response",
            "note",
            "finding",
            "idea",
            "project",
        )
        if any(marker in cleaned for marker in generic_markers):
            return 0.18
        if len(cleaned.split()) >= 8:
            return 0.12
        return 0.0

    @staticmethod
    def _recall_focus(normalized_prompt: str) -> str:
        return FocusResolutionSupport.recall_focus(normalized_prompt).focus

    @staticmethod
    def _focus_overlap(focus: str, *haystacks: str) -> int:
        return MemoryRankingSignals.focus_overlap(focus, *haystacks)

    def _project_id_for_session(self, session_id: str) -> str | None:
        manager = getattr(self.interaction_history_service, "interaction_log_manager", None)
        persistence_manager = getattr(manager, "persistence_manager", None)
        sessions = getattr(persistence_manager, "sessions", None)
        if sessions is None:
            return None
        try:
            session = sessions.get(session_id)
        except Exception:
            return None
        if not isinstance(session, dict):
            return None
        project_id = str(session.get("project_id") or "").strip()
        return project_id or None


def indexed_to_memory(
    *,
    indexed: IndexedMemoryRecord,
    relevance: float,
    metadata: dict[str, object] | None = None,
) -> RetrievedMemory:
    return RetrievedMemory(
        source=indexed.source,
        memory_kind=indexed.memory_kind,
        label=indexed.label,
        summary=indexed.summary,
        relevance=relevance,
        metadata=dict(metadata or indexed.to_metadata()),
    )
