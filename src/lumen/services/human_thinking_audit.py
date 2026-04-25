from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AuditDimension:
    dimension_id: str
    label: str
    implementation_status: str
    owner_modules: tuple[str, ...]
    current_behavior_summary: str
    repo_evidence: tuple[str, ...]
    observed_weakness: str | None
    recommended_action: str

    def to_dict(self) -> dict[str, object]:
        return {
            "dimension_id": self.dimension_id,
            "label": self.label,
            "implementation_status": self.implementation_status,
            "owner_modules": list(self.owner_modules),
            "current_behavior_summary": self.current_behavior_summary,
            "repo_evidence": list(self.repo_evidence),
            "observed_weakness": self.observed_weakness,
            "recommended_action": self.recommended_action,
        }


@dataclass(frozen=True, slots=True)
class BacklogDomain:
    domain_id: str
    label: str
    backlog_status: str
    current_state_summary: str
    repo_evidence: tuple[str, ...]
    handoff_note: str

    def to_dict(self) -> dict[str, object]:
        return {
            "domain_id": self.domain_id,
            "label": self.label,
            "backlog_status": self.backlog_status,
            "current_state_summary": self.current_state_summary,
            "repo_evidence": list(self.repo_evidence),
            "handoff_note": self.handoff_note,
        }


class HumanThinkingAuditService:
    """Builds a findings-first audit of Lumen's human/thinking layer."""

    NOTE_DIR = "new additions and test"

    AUDIT_DIMENSIONS: tuple[AuditDimension, ...] = (
        AuditDimension(
            dimension_id="response_shaping_looseness",
            label="Response shaping / looseness",
            implementation_status="present_but_weak",
            owner_modules=(
                "src/lumen/reasoning/human_language_layer.py",
                "src/lumen/reasoning/interaction_style_policy.py",
                "src/lumen/reasoning/response_tone_engine.py",
            ),
            current_behavior_summary="Mode-specific structure and looseness already exist, and collab is intentionally looser than default/direct.",
            repo_evidence=(
                "tests/unit/test_interaction_service.py::test_interaction_service_personality_surfaces_change_visible_wording",
                "tests/unit/test_interaction_service.py::test_interaction_service_preserves_planning_substance_across_modes",
            ),
            observed_weakness="Collab already differs, but it can still read a little templated on analytical turns and refinement prompts.",
            recommended_action="tighten",
        ),
        AuditDimension(
            dimension_id="context_continuity",
            label="Context continuity",
            implementation_status="present",
            owner_modules=(
                "src/lumen/reasoning/conversation_awareness.py",
                "src/lumen/reasoning/stance_consistency_layer.py",
                "src/lumen/reasoning/memory_retrieval_layer.py",
            ),
            current_behavior_summary="The system already carries active-thread and retrieval continuity through follow-up resolution and stance shaping.",
            repo_evidence=(
                "src/lumen/validation/full_system_validation.py::_record_followup_continuity_check",
                "tests/unit/test_interaction_history_service.py::test_controller_can_report_interaction_patterns",
            ),
            observed_weakness=None,
            recommended_action="leave",
        ),
        AuditDimension(
            dimension_id="emotional_mirroring",
            label="Emotional mirroring",
            implementation_status="present_but_weak",
            owner_modules=(
                "src/lumen/reasoning/empathy_model.py",
                "src/lumen/reasoning/human_language_layer.py",
                "src/lumen/reasoning/response_tone_engine.py",
            ),
            current_behavior_summary="Grounded emotional alignment already exists and is bounded by anti-roleplay and safety layers.",
            repo_evidence=(
                "tests/unit/test_reasoning_response_models.py::test_emotional_support_limits_allow_warmth_without_dependency",
                "tests/unit/test_interaction_service.py::test_interaction_service_uses_warmer_collab_greeting_surface",
            ),
            observed_weakness="Frustration and correction signals are recognized, but the cue set is still a little narrow for natural user language.",
            recommended_action="tighten",
        ),
        AuditDimension(
            dimension_id="epistemic_awareness",
            label="Epistemic awareness",
            implementation_status="present_but_weak",
            owner_modules=(
                "src/lumen/reasoning/human_language_layer.py",
                "src/lumen/reasoning/stance_consistency_layer.py",
                "src/lumen/reasoning/response_tone_engine.py",
            ),
            current_behavior_summary="The pipeline already tags exploratory, assertive, and unsure posture and uses it to shape leads and disagreement framing.",
            repo_evidence=(
                "tests/unit/test_stance_consistency_layer.py::test_stance_consistency_uses_livelier_collab_uncertainty_surface",
                "tests/unit/test_reasoning_pipeline.py::test_reasoning_pipeline_human_language_layer_distinguishes_epistemic_stances",
            ),
            observed_weakness="The stance cues are present, but the prompt classifier is simple enough that mixed cues can collapse into the wrong stance bucket.",
            recommended_action="tighten",
        ),
        AuditDimension(
            dimension_id="correction_handling",
            label="Correction handling",
            implementation_status="present_but_weak",
            owner_modules=(
                "src/lumen/reasoning/human_language_layer.py",
                "src/lumen/reasoning/response_tone_engine.py",
                "src/lumen/services/interaction_service.py",
            ),
            current_behavior_summary="Correction-aware phrasing already exists and is threaded through the response tone engine.",
            repo_evidence=(
                "src/lumen/reasoning/response_tone_engine.py::question_turn_lead",
                "tests/unit/test_interaction_service.py::test_interaction_service_suppresses_repeated_low_confidence_clarification_loops",
            ),
            observed_weakness="Some common refinement phrasings are still not caught cleanly enough, which can make the system sound more reset-heavy than intended.",
            recommended_action="tighten",
        ),
        AuditDimension(
            dimension_id="energy_adaptation",
            label="Energy adaptation",
            implementation_status="present",
            owner_modules=(
                "src/lumen/reasoning/human_language_layer.py",
                "src/lumen/reasoning/response_tone_engine.py",
            ),
            current_behavior_summary="The current human-language layer already maps user energy into calmer, engaged, or focused response bias.",
            repo_evidence=(
                "src/lumen/reasoning/human_language_layer.py::_user_energy",
                "tests/unit/test_interaction_service.py::test_interaction_service_micro_turn_collab_is_slightly_warmer_than_default",
            ),
            observed_weakness=None,
            recommended_action="leave",
        ),
        AuditDimension(
            dimension_id="intentional_tool_invocation",
            label="Intentional tool invocation",
            implementation_status="present",
            owner_modules=(
                "src/lumen/reasoning/tool_threshold_gate.py",
                "src/lumen/services/interaction_service.py",
            ),
            current_behavior_summary="Tool routing is bounded by the threshold gate and existing regression coverage already checks that core reasoning survives across modes.",
            repo_evidence=(
                "tests/integration/test_tool_signal_ask_routing.py",
                "tests/unit/test_interaction_service.py::test_interaction_service_math_answer_stays_invariant_across_modes",
            ),
            observed_weakness=None,
            recommended_action="leave",
        ),
        AuditDimension(
            dimension_id="thinking_layer_quality_across_modes",
            label="Thinking-layer quality across modes",
            implementation_status="present",
            owner_modules=(
                "src/lumen/reasoning/reasoning_pipeline.py",
                "src/lumen/services/interaction_service.py",
            ),
            current_behavior_summary="The reasoning spine remains active across modes, and existing tests already verify that mode changes affect expression instead of planning substance.",
            repo_evidence=(
                "tests/unit/test_interaction_service.py::test_interaction_service_preserves_planning_substance_across_modes",
                "tests/unit/test_interaction_service.py::test_interaction_service_math_answer_stays_invariant_across_modes",
            ),
            observed_weakness=None,
            recommended_action="leave",
        ),
        AuditDimension(
            dimension_id="srd_disruption_agency_trust",
            label="SRD-style disruption / agency / trust handling",
            implementation_status="partial",
            owner_modules=(
                "src/lumen/reasoning/srd_diagnostic.py",
                "src/lumen/reasoning/response_strategy_layer.py",
            ),
            current_behavior_summary="The repo already contains an SRD diagnostic and uses it to stabilize or narrow responses under disruption.",
            repo_evidence=(
                "tests/unit/test_reasoning_pipeline.py::test_reasoning_pipeline_srd_marks_agency_block_for_hard_clarify_case",
                "tests/unit/test_interaction_service.py::test_interaction_service_sets_clarify_first_behavior_posture_for_low_confidence_clarification",
            ),
            observed_weakness="SRD is real, but the current trigger set is narrower and more structural than the richer trust/agency framing described in the note.",
            recommended_action="tighten",
        ),
        AuditDimension(
            dimension_id="self_edit_disabled_policy",
            label="Self-edit disabled policy",
            implementation_status="present",
            owner_modules=(
                "src/lumen/services/safety_service.py",
                "src/lumen/services/interaction_service.py",
            ),
            current_behavior_summary="Runtime self-edit behavior is disabled and should be surfaced as unavailable rather than conditionally supervised.",
            repo_evidence=(
                "src/lumen/services/safety_service.py",
                "tests/unit/test_interaction_service.py::test_interaction_service_returns_safety_refusal_without_updating_active_thread",
            ),
            observed_weakness=None,
            recommended_action="leave",
        ),
    )

    BACKLOG_DOMAINS: tuple[BacklogDomain, ...] = (
        BacklogDomain(
            domain_id="content_generation",
            label="Content generation and editing",
            backlog_status="partially_present",
            current_state_summary="Lumen already has content-generation runtime surfaces, but they are provider-gated and not the focus of this audit phase.",
            repo_evidence=(
                "src/lumen/validation/full_system_validation.py::_content_runtime_status",
                "main.spec",
            ),
            handoff_note="Keep as roadmap input only unless human-language findings require a small content-surface honesty fix.",
        ),
        BacklogDomain(
            domain_id="analytics_business",
            label="Analytics / business / forecasting",
            backlog_status="future_roadmap",
            current_state_summary="Math, data, and experiment tooling exist, but broad business analytics and forecasting are not a current runtime promise.",
            repo_evidence=(
                "src/lumen/validation/full_system_validation.py",
                "tool_bundles",
            ),
            handoff_note="Defer to a later domain-expansion phase.",
        ),
        BacklogDomain(
            domain_id="speech_audio",
            label="Speech / audio understanding",
            backlog_status="future_roadmap",
            current_state_summary="No first-class speech/audio understanding layer is currently part of the advertised runtime.",
            repo_evidence=(
                "src/lumen/validation/full_system_validation.py",
            ),
            handoff_note="Treat as future multimodal expansion, not current human-layer scope.",
        ),
        BacklogDomain(
            domain_id="vision_imaging",
            label="Vision / imaging",
            backlog_status="future_roadmap",
            current_state_summary="Desktop image handling exists for the shell, but there is no broad production computer-vision reasoning surface here.",
            repo_evidence=(
                "src/lumen/desktop/chat_app.py",
            ),
            handoff_note="Leave out of this phase unless later roadmap work explicitly targets multimodal reasoning.",
        ),
        BacklogDomain(
            domain_id="assistants_automation",
            label="Assistants / automation",
            backlog_status="partially_present",
            current_state_summary="Lumen already behaves like an assistant and has tooling boundaries, but recurring automation is not part of this audit’s main implementation scope.",
            repo_evidence=(
                "src/lumen/services/interaction_service.py",
                "src/lumen/reasoning/tool_threshold_gate.py",
            ),
            handoff_note="Keep as backlog; only revisit if conversational audit findings reveal boundary confusion.",
        ),
        BacklogDomain(
            domain_id="explainability_transparency",
            label="Explainability / transparency",
            backlog_status="already_present",
            current_state_summary="The repo already contains substantial reasoning, retrieval, and diagnostics transparency surfaces.",
            repo_evidence=(
                "src/lumen/services/diagnostics_service.py",
                "src/lumen/reasoning/trainability_trace.py",
            ),
            handoff_note="This phase should refine and audit the existing surfaces rather than adding a new explainability layer.",
        ),
        BacklogDomain(
            domain_id="invention_schematic_support",
            label="Invention / schematic support",
            backlog_status="partially_present",
            current_state_summary="Design, planning, and invention-adjacent tooling exists, but broad schematic generation is not a stable public contract yet.",
            repo_evidence=(
                "src/lumen/reasoning/design_surface_support.py",
                "src/lumen/tools/design_spec_builder.py",
            ),
            handoff_note="Keep for later roadmap work unless the human-language audit uncovers wording that overclaims this capability.",
        ),
        BacklogDomain(
            domain_id="investing_news_health_world_knowledge",
            label="Investing / news / health / world knowledge",
            backlog_status="future_roadmap",
            current_state_summary="There is general knowledge and reasoning support, but no validated real-time news, health, or investing authority surface should be implied here.",
            repo_evidence=(
                "src/lumen/knowledge/knowledge_service.py",
                "src/lumen/services/safety_service.py",
            ),
            handoff_note="Keep explicitly out of scope for this phase and treat as later domain-specific roadmap work.",
        ),
    )

    def __init__(self, *, repo_root: Path) -> None:
        self.repo_root = Path(repo_root).resolve()

    def build_report(self) -> dict[str, object]:
        note_dir = self.repo_root / self.NOTE_DIR
        note_sources = {
            "audit_reference": self._note_status(note_dir / "test.md"),
            "backlog_reference": self._note_status(note_dir / "plan ideas.md"),
        }
        dimensions = [item.to_dict() for item in self.AUDIT_DIMENSIONS]
        backlog = [item.to_dict() for item in self.BACKLOG_DOMAINS]
        confirmed_gaps = [
            item for item in dimensions
            if str(item.get("implementation_status") or "") in {"present_but_weak", "partial", "missing"}
        ]
        targeted_implementation = [
            {
                "dimension_id": item["dimension_id"],
                "label": item["label"],
                "recommended_action": item["recommended_action"],
                "observed_weakness": item["observed_weakness"],
            }
            for item in confirmed_gaps
            if str(item.get("recommended_action") or "") in {"tighten", "add_tests"}
        ]
        action_counts: dict[str, int] = {}
        for item in dimensions:
            action = str(item.get("recommended_action") or "defer")
            action_counts[action] = action_counts.get(action, 0) + 1
        backlog_counts: dict[str, int] = {}
        for item in backlog:
            status = str(item.get("backlog_status") or "future_roadmap")
            backlog_counts[status] = backlog_counts.get(status, 0) + 1
        return {
            "status": self._status_from_dimensions(dimensions),
            "repo_root": str(self.repo_root),
            "note_sources": note_sources,
            "audit_dimensions": dimensions,
            "confirmed_gap_list": confirmed_gaps,
            "targeted_implementation_list": targeted_implementation,
            "action_counts": action_counts,
            "backlog_appendix": {
                "domains": backlog,
                "status_counts": backlog_counts,
            },
        }

    @staticmethod
    def _status_from_dimensions(dimensions: list[dict[str, object]]) -> str:
        if any(str(item.get("implementation_status") or "") == "missing" for item in dimensions):
            return "warn"
        if any(str(item.get("implementation_status") or "") in {"present_but_weak", "partial"} for item in dimensions):
            return "warn"
        return "ok"

    @staticmethod
    def _note_status(path: Path) -> dict[str, object]:
        return {
            "path": str(path),
            "exists": path.exists(),
        }
