from __future__ import annotations

from lumen.memory.memory_models import MemoryClassification, MemoryWriteDecision


class MemoryWritePolicy:
    """Owns the decision about whether an interaction may write memory."""

    def decide(
        self,
        *,
        classification: MemoryClassification,
        client_surface: str,
        mobile_research_note_auto_save: bool,
    ) -> MemoryWriteDecision:
        if classification.candidate_type == "personal_context_candidate":
            if classification.explicit_save_requested:
                return MemoryWriteDecision.personal_memory(
                    reason="Personal context was explicitly user-directed for saving, so it may be stored separately."
                )
            return MemoryWriteDecision.skip(
                reason="Personal context requires explicit user consent and remains unsaved by default."
            )

        if classification.candidate_type == "research_memory_candidate" and classification.save_eligible:
            if client_surface == "mobile" and not mobile_research_note_auto_save:
                return MemoryWriteDecision.skip(
                    reason="Research-note auto-save is blocked on the mobile surface by default.",
                    blocked_by_surface_policy=True,
                )
            return MemoryWriteDecision.research_note(
                reason="Research-oriented context is eligible for chronological research-note saving."
            )

        return MemoryWriteDecision.skip(
            reason="The interaction is not eligible for automatic memory writing."
        )
