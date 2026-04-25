from pathlib import Path
from types import SimpleNamespace

import pytest

from lumen.app.models import InteractionProfile
from lumen.app.command_parser import CommandParser
from lumen.reasoning.pipeline_models import (
    ConversationAwarenessResult,
    DialogueManagementResult,
    ThoughtFramingResult,
)
from lumen.reasoning.memory_retrieval_layer import MemoryRetrievalResult, RetrievedMemory
from lumen.reasoning.mode_response_shaper import ModeResponseShaper
from lumen.reasoning.reasoning_state import ExecutionOutcome, ReasoningStateFrame
from lumen.reasoning.supervised_decision_support import SupervisedDecisionSupport, SupervisedExample
from lumen.reasoning.planner import Planner
from lumen.reasoning.research_engine import ResearchEngine
from lumen.routing.capability_manager import CapabilityManager
from lumen.routing.domain_router import DomainRoute, DomainRouter
from lumen.routing.intent_router import IntentRouter
from lumen.routing.prompt_resolution import PromptResolver
from lumen.knowledge.knowledge_service import KnowledgeService
from lumen.services.interaction_service import InteractionService
from lumen.services.interaction_orchestration_models import InteractionTurnContext
from lumen.services.reasoning_state_service import ReasoningStateService
from lumen.services.safety_models import PromptSafetyDecision
from lumen.tools.registry_types import BundleManifest, CapabilityManifest


class FakeArchiveService:
    def summary(self, **kwargs):
        return {
            "record_count": 3,
            "status_counts": {"ok": 2, "partial": 1},
            "tool_counts": {"anh": 3},
        }

    def retrieve_context(self, query, **kwargs):
        return {
            "query": query,
            "record_count": 2,
            "top_matches": [
                {
                    "score": 5,
                    "score_breakdown": {"keyword_score": 3, "semantic_score": 2},
                    "matched_fields": ["summary"],
                    "record": {
                        "tool_id": "anh",
                        "capability": "spectral_dip_scan",
                        "summary": "Great Attractor confirmation candidate",
                    },
                }
            ],
        }


class FakeToolExecutionService:
    def __init__(self, capability_manager: CapabilityManager):
        class _Registry:
            def __init__(self, manager: CapabilityManager):
                self._manager = manager

            def list_tools(self) -> dict[str, list[str]]:
                grouped: dict[str, list[str]] = {}
                for spec in self._manager.list_capabilities().values():
                    grouped.setdefault(spec.tool_id, []).append(spec.tool_capability)
                return grouped

        self.registry = _Registry(capability_manager)

    def run_tool(self, **kwargs):
        if kwargs.get("tool_id") == "design":
            class Result:
                status = "ok"
                tool_id = "design"
                capability = "system_spec"
                summary = "Generated bounded system spec."
                run_dir = None
                archive_path = None
                error = None
                structured_data = {
                    "subject": str((kwargs.get("params") or {}).get("brief") or "the system"),
                    "design_domain": "software_system",
                    "summary": "Here is a structured first-pass system spec for the design target.",
                    "system_overview": "Treat it as layered input, orchestration, execution, and output surfaces.",
                    "components": [
                        "Entry surface for requests",
                        "Coordinator for system decisions",
                        "Typed execution layer",
                    ],
                    "resources": ["Requirements brief", "Typed contracts"],
                    "constraints": ["Keep the first version modular", "Preserve route authority"],
                    "tradeoffs": ["More structure improves reliability but raises implementation cost"],
                    "failure_points": ["Weak boundaries can leak scaffolding"],
                    "next_steps": ["Lock the main user flow", "Build the thinnest end-to-end slice"],
                    "assumptions": ["Assume a bounded v1 target"],
                }
                artifacts = []
                logs = []
                provenance = {}

            return Result()

        class Result:
            status = "ok"
            tool_id = "anh"
            capability = "spectral_dip_scan"
            summary = "GA Local Analysis Kit run completed"
            run_dir = None
            archive_path = None
            error = None
            structured_data = {}
            artifacts = []
            logs = []
            provenance = {}

        return Result()


class FakeInteractionHistoryService:
    def __init__(self):
        self.records = []

    def record_interaction(self, **kwargs):
        self.records.append(kwargs)
        return kwargs

    def retrieve_context(self, query, **kwargs):
        return {
            "interaction_record_count": 1,
            "top_interaction_matches": [
                {
                    "score": 4,
                    "score_breakdown": {"keyword_score": 2, "semantic_score": 2},
                    "record": {
                        "prompt": "create a migration plan for lumen",
                        "prompt_view": {
                            "canonical_prompt": "create a migration plan for lumen",
                            "original_prompt": "create a migration plan for lumen",
                            "resolved_prompt": None,
                            "rewritten": False,
                        },
                        "summary": "Planning response for: create a migration plan for lumen",
                    },
                }
            ],
        }

    def recent_records(self, **kwargs):
        return [
            {
                "mode": "planning",
                "kind": "planning.migration",
                "summary": "Planning response for: create a migration plan for lumen",
            }
        ]

    def summarize_interactions(self, **kwargs):
        clarification_count = sum(
            1
            for item in self.records
            if str((item.get("response") or {}).get("mode") or "").strip() == "clarification"
        )
        interaction_count = len(self.records)
        dominant_intent_counts: dict[str, int] = {}
        for item in self.records:
            response = item.get("response") or {}
            dominant_intent = str(response.get("dominant_intent") or "unknown").strip() or "unknown"
            dominant_intent_counts[dominant_intent] = dominant_intent_counts.get(dominant_intent, 0) + 1
        clarification_trend = [
            "clarified"
            if str((item.get("response") or {}).get("mode") or "").strip() == "clarification"
            else "clear"
            for item in reversed(self.records[-5:])
        ]
        if not clarification_trend:
            clarification_drift = None
            recent_clarification_mix = None
            latest_clarification = None
        elif len(clarification_trend) == 1:
            clarification_drift = "insufficient_data"
            recent_clarification_mix = f"stable:{clarification_trend[0]}"
            latest_clarification = clarification_trend[0]
        else:
            latest_clarification = clarification_trend[0]
            if clarification_trend.count("clarified") >= clarification_trend.count("clear"):
                recent_clarification_mix = "clarification_heavy_mixed"
            else:
                recent_clarification_mix = "mixed"
            clarification_drift = "increasing" if latest_clarification == "clarified" else "decreasing"
        return {
            "interaction_count": interaction_count,
            "clarification_count": clarification_count,
            "clarification_ratio": round((clarification_count / interaction_count), 4) if interaction_count else 0.0,
            "clarification_trend": clarification_trend,
            "recent_clarification_mix": recent_clarification_mix,
            "latest_clarification": latest_clarification,
            "clarification_drift": clarification_drift,
            "dominant_intent_counts": dominant_intent_counts,
            "retrieval_lead_counts": {},
            "retrieval_observation_count": 0,
        }


class FakeSessionContextService:
    def __init__(self):
        self.active_thread = None
        self.interaction_profile = InteractionProfile.default()

    def get_active_thread(self, session_id):
        return self.active_thread

    def get_interaction_profile(self, session_id):
        return self.interaction_profile

    def update_active_thread(self, *, session_id, prompt, response):
        if self.active_thread is None:
            objective = f"Objective for: {prompt}"
            thread_summary = response.get("summary")
            tool_context = dict(response.get("tool_execution") or {})
        else:
            objective = self.active_thread["objective"]
            thread_summary = f"{self.active_thread['thread_summary']} | latest: {prompt}"
            tool_context = dict(response.get("tool_execution") or self.active_thread.get("tool_context") or {})
        self.active_thread = {
            "session_id": session_id,
            "mode": response.get("mode"),
            "kind": response.get("kind"),
            "prompt": prompt,
            "objective": objective,
            "thread_summary": thread_summary,
            "summary": response.get("summary"),
            "interaction_profile": self.interaction_profile.to_dict(),
            "intent_domain": response.get("intent_domain"),
            "intent_domain_confidence": response.get("intent_domain_confidence"),
            "response_depth": response.get("response_depth"),
            "conversation_phase": response.get("conversation_phase"),
            "next_step_state": dict(response.get("next_step_state") or {}),
            "tool_suggestion_state": dict(response.get("tool_suggestion_state") or {}),
            "trainability_trace": dict(response.get("trainability_trace") or {}),
            "supervised_support_trace": dict(response.get("supervised_support_trace") or {}),
            "tool_context": tool_context,
            "continuation_offer": dict(response.get("continuation_offer") or {}),
            "normalized_topic": str(
                ((response.get("domain_surface") or {}).get("topic") if isinstance(response.get("domain_surface"), dict) else "")
            ).strip().lower()
            or None,
            "reasoning_state": dict(response.get("reasoning_state") or {}),
        }
        return self.active_thread


class FakeSafetyService:
    def evaluate_prompt(self, prompt: str) -> PromptSafetyDecision:
        normalized = " ".join(str(prompt).lower().split())
        if any(
            token in normalized
            for token in ("edit yourself", "modify yourself", "rewrite your own code", "patch yourself")
        ):
            return PromptSafetyDecision(
                action="refuse",
                category="self_modification",
                severity="medium",
                rationale="Runtime self-modification is disabled.",
                boundary="I can't edit or modify my own code or runtime behavior from inside the conversation.",
                tier="restricted",
                outcome_risk="medium",
                safe_redirects=[
                    "I can help explain what change you want, draft a plan, or review a supervised edit you explicitly direct.",
                ],
            )
        return PromptSafetyDecision(
            action="allow",
            category="allowed",
            severity="none",
            rationale="safe",
            boundary="",
        )


class SparseArchiveService(FakeArchiveService):
    def retrieve_context(self, query, **kwargs):
        return {
            "query": query,
            "record_count": 0,
            "top_matches": [],
        }


class SparseInteractionHistoryService(FakeInteractionHistoryService):
    def retrieve_context(self, query, **kwargs):
        return {
            "interaction_record_count": 0,
            "top_interaction_matches": [],
        }

    def recent_records(self, **kwargs):
        return []


class ProjectAwareInteractionHistoryService(FakeInteractionHistoryService):
    def __init__(
        self,
        *,
        session_recent: list[dict[str, object]] | None = None,
        project_recent: list[dict[str, object]] | None = None,
    ):
        super().__init__()
        self._session_recent = list(session_recent or [])
        self._project_recent = list(project_recent or [])

    def recent_records(self, **kwargs):
        if kwargs.get("project_id"):
            return list(self._project_recent)
        return list(self._session_recent)


class MemoryClarificationHistoryService(FakeInteractionHistoryService):
    def __init__(self, recent_records: list[dict[str, object]]):
        super().__init__()
        self._recent_records = recent_records
        self.saved: list[dict[str, object]] = []

    def recent_records(self, **kwargs):
        return list(self._recent_records)

    def save_memory_from_record(self, *, source_record, target, client_surface="main"):
        result = {
            "status": "ok",
            "target": target,
            "source_prompt": source_record.get("prompt"),
            "client_surface": client_surface,
        }
        self.saved.append(result)
        return result


class FakeInferenceService:
    def evaluate_hosted_research(self, *, route, validation_context):
        return type(
            "Decision",
            (),
            {
                "use_hosted_inference": route.mode == "research" and route.kind in {"research.general", "research.summary"},
                "reason": "Sparse local context for a general or explanatory research turn.",
            },
        )()

    def evaluate_hosted_writing(self):
        return type(
            "Decision",
            (),
            {
                "use_hosted_inference": True,
                "reason": "Hosted provider is available for bounded writing/editing workflows.",
            },
        )()

    def infer_research_reply(self, *, prompt, session_id, interaction_profile, validation_context):
        normalized = " ".join(str(prompt).lower().split())
        if "george washington" in normalized:
            output_text = "George Washington was the first president of the United States and a central figure in the American founding."
        elif "black hole" in normalized:
            output_text = "Black holes are regions where gravity is so strong that even light cannot escape."
        else:
            output_text = "Here's a grounded explanation based on the current question."
        return type(
            "Result",
            (),
            {
                "provider_id": "openai_responses",
                "model": "gpt-test",
                "output_text": output_text,
                "finish_reason": "completed",
            },
        )()

    def infer_writing_reply(self, *, prompt, session_id, interaction_profile, workflow):
        label = str(workflow.get("label") or "")
        if label == "translation":
            output_text = "Hola. Esta es una traduccion de trabajo."
        elif label == "rewrite":
            output_text = "Here is a cleaner rewrite in a more professional tone."
        elif label == "cleanup":
            output_text = "Here is the cleaned-up text with grammar corrected."
        else:
            output_text = "Subject: Migration update\n\nHere is a draft message you can refine."
        return type(
            "Result",
            (),
            {
                "provider_id": "openai_responses",
                "model": "gpt-test",
                "output_text": output_text,
                "finish_reason": "completed",
            },
        )()


def make_interaction_service(
    *,
    archive_service=None,
    interaction_history_service=None,
    session_context_service=None,
    inference_service=None,
    supervised_decision_support=None,
) -> InteractionService:
    capability_manager = CapabilityManager(
        manifests={
            "anh": BundleManifest(
                id="anh",
                name="Astronomical Node Heuristics",
                version="0.1.0",
                entrypoint="bundle.py",
                capabilities=[
                    CapabilityManifest(
                        id="spectral_dip_scan",
                        adapter="anh_spectral_scan_adapter",
                        app_capability_key="astronomy.anh_spectral_scan",
                        command_aliases=[
                            "run anh",
                            "scan si iv dips",
                            "find spectral dips",
                        ],
                    )
                ],
            ),
            "report": BundleManifest(
                id="report",
                name="Reporting Tools",
                version="0.1.0",
                entrypoint="bundle.py",
                capabilities=[
                    CapabilityManifest(
                        id="session.confidence",
                        adapter="session_confidence_adapter",
                        app_capability_key="report.session_confidence",
                        command_aliases=[
                            "report session confidence",
                            "summarize confidence",
                            "summarize session confidence",
                        ],
                    )
                ],
            ),
            "workspace": BundleManifest(
                id="workspace",
                name="Workspace Tools",
                version="0.1.0",
                entrypoint="bundle.py",
                capabilities=[
                    CapabilityManifest(
                        id="inspect.structure",
                        adapter="workspace_inspect_adapter",
                        app_capability_key="workspace.inspect_structure",
                        command_aliases=[
                            "inspect workspace",
                            "summarize workspace",
                        ],
                    )
                ],
            ),
            "design": BundleManifest(
                id="design",
                name="Design Tools",
                version="0.1.0",
                entrypoint="bundle.py",
                capabilities=[
                    CapabilityManifest(
                        id="system_spec",
                        adapter="system_spec_adapter",
                        app_capability_key="design.system_spec",
                        command_aliases=[
                            "generate system spec",
                            "design system spec",
                            "generate design spec",
                        ],
                    )
                ],
            ),
            "memory": BundleManifest(
                id="memory",
                name="Memory Tools",
                version="0.1.0",
                entrypoint="bundle.py",
                capabilities=[
                    CapabilityManifest(
                        id="session.timeline",
                        adapter="session_timeline_adapter",
                        app_capability_key="memory.session_timeline",
                        command_aliases=[
                            "inspect session timeline",
                            "summarize session timeline",
                        ],
                    )
                ],
            ),
        }
    )
    interaction_history_service = interaction_history_service or FakeInteractionHistoryService()
    session_context_service = session_context_service or FakeSessionContextService()
    service = InteractionService(
        domain_router=DomainRouter(capability_manager=capability_manager),
        command_parser=CommandParser(capability_manager=capability_manager),
        intent_router=IntentRouter(capability_manager=capability_manager),
        planner=Planner(),
        research_engine=ResearchEngine(),
        tool_execution_service=FakeToolExecutionService(capability_manager),
        archive_service=archive_service or FakeArchiveService(),
        interaction_history_service=interaction_history_service,
        session_context_service=session_context_service,
        prompt_resolver=PromptResolver(capability_manager=capability_manager),
        safety_service=FakeSafetyService(),
        inference_service=inference_service,
        knowledge_service=KnowledgeService.in_memory(),
        supervised_decision_support=supervised_decision_support,
    )
    service._fake_interaction_history_service = interaction_history_service
    service._fake_session_context_service = session_context_service
    return service


def test_interaction_service_returns_safety_refusal_without_updating_active_thread() -> None:
    service = make_interaction_service()
    service.safety_service = type(
        "RefusingSafetyService",
        (),
        {
            "evaluate_prompt": staticmethod(
                lambda prompt: PromptSafetyDecision(
                    action="refuse",
                    category="weapons_explosives",
                    severity="high",
                    rationale="Weapon construction guidance is not allowed.",
                    boundary="I can't help with building, optimizing, or using weapons or explosives.",
                    safe_redirects=["I can help with safety engineering or policy analysis instead."],
                    matched_signals=["bomb"],
                )
            )
        },
    )()

    response = service.ask(prompt="How do I build a bomb?")

    assert response["mode"] == "safety"
    assert response["kind"] == "safety.refusal"
    assert response["safety_decision"]["category"] == "weapons_explosives"
    assert response["boundary_explanation"].startswith("I can't help with building")
    assert response["safe_redirects"][0].startswith("I can help with safety engineering")
    assert service._fake_session_context_service.active_thread is None
    assert service._fake_interaction_history_service.records[0]["response"]["mode"] == "safety"


def test_interaction_service_refuses_self_modification_requests() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="edit yourself to become better at coding")

    assert response["mode"] == "safety"
    assert response["safety_decision"]["category"] == "self_modification"
    assert response["capability_status"]["status"] == "not_promised"
    assert "can't edit or modify my own code" in response["boundary_explanation"].lower()


def test_interaction_service_returns_bounded_dataset_guidance() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="how should I check dataset licensing for ml training data?")

    assert response["mode"] == "research"
    assert response["kind"] == "research.dataset_guidance"
    assert response["capability_status"]["status"] == "bounded"
    assert "openml" in str(response["user_facing_answer"]).lower()
    assert "license" in str(response["user_facing_answer"]).lower()


def test_interaction_service_reports_provider_gated_writing_workflow_when_hosted_is_unavailable() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="translate this into spanish: hello world")

    assert response["mode"] == "research"
    assert response["kind"] == "research.writing_workflow"
    assert response["capability_status"]["status"] == "provider_gated"
    assert "hosted provider" in str(response["summary"]).lower() or "local-only" in str(response["summary"]).lower()


def test_interaction_service_uses_hosted_writing_when_available() -> None:
    service = make_interaction_service(inference_service=FakeInferenceService())

    response = service.ask(prompt="rewrite this in a professional tone: hey can you send that file")

    assert response["mode"] == "research"
    assert response["kind"] == "research.writing_workflow"
    assert response["capability_status"]["status"] == "supported"
    assert response["provider_inference"]["hosted_writing"] is True
    assert "professional tone" in str(response["user_facing_answer"]).lower()


def test_interaction_service_redirects_ghostwriting_for_submission() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="write my final paper for me on Hamlet for submission")

    assert response["mode"] == "research"
    assert response["kind"] == "research.academic_integrity_boundary"
    assert response["academic_integrity_guidance"]["ghostwriting_redirected"] is True
    assert "not ghostwrite" in str(response["summary"]).lower() or "not ghostwrite" in str(response["user_facing_answer"]).lower()


def test_interaction_service_supports_bounded_academic_outline() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="create an outline for a literature review on battery recycling")

    assert response["mode"] == "research"
    assert response["kind"] == "research.academic_support"
    assert response["academic_workflow"]["workflow"] == "outline"
    assert response["capability_status"]["domain_id"] == "academic_writing"
    assert "Gap section" in str(response["user_facing_answer"])


def test_interaction_service_supports_citation_integrity_flags() -> None:
    service = make_interaction_service()

    response = service.ask(
        prompt="format this citation in APA: author Jane Doe; title Neural Feedback; year 2024; url https://example.com/paper"
    )

    assert response["mode"] == "research"
    assert response["kind"] == "research.academic_citation"
    assert response["citation_integrity_status"] == "formatted_from_supplied_metadata"
    assert response["capability_status"]["domain_id"] == "citation_support"


def test_interaction_service_supports_bounded_math_bridge_help() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="explain Bayes theorem from the basics")

    assert response["mode"] == "research"
    assert response["kind"] == "research.academic_math_support"
    assert response["math_support_level"] == "bridge"
    assert response["capability_status"]["domain_id"] == "college_math_science_support"


def test_interaction_service_routes_conceptual_derivative_to_academic_math_support() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="what is the derivative of x^2")

    assert response["mode"] == "research"
    assert response["kind"] == "research.academic_math_support"
    assert response["academic_workflow"]["workflow"] == "math_science_support"
    assert "derivative" in str(response["user_facing_answer"]).lower()


def test_interaction_service_routes_simple_greeting_as_conversation() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="hello lumen")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.greeting"
    assert response["reply"]
    assert "validation plan" not in response["reply"].lower()
    assert response["reply"].endswith("?") or "ready" in response["reply"].lower() or "here" in response["reply"].lower()
    assert response["summary"] == response["reply"]
    assert response["dialogue_management"]["interaction_mode"] == "social"
    assert response["intent_domain"] == "conversational"
    assert response["response_depth"] in {"concise", "standard"}
    assert response["conversation_phase"] == "intake"
    assert response["wake_interaction"] == {
        "wake_phrase": "hello lumen",
        "classification": "pure_greeting",
        "stripped_prompt": "",
    }
    assert response["thought_framing"]["response_kind_label"] == "lightweight_social"
    assert response["interaction_mode"] == "social"
    assert response["idea_state"] == "refining"
    assert response["response_strategy"] == "answer"
    assert response["state_control"]["core_state"] == "focus"
    assert response["reasoning_depth"] == "low"
    assert response["tools_enabled"] is False
    assert response["lightweight_social"] is True
    assert "pipeline_execution" not in response
    assert "pipeline_trace" not in response
    assert service._fake_session_context_service.active_thread is None
    assert service._fake_interaction_history_service.records[0]["response"]["mode"] == "conversation"


def test_interaction_service_attaches_general_assistant_metadata_to_conversation_turn() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="hello lumen")

    assert response["assistant_quality_posture"]["profile"] == "normal_chat"
    assert response["assistant_quality_posture"]["direct_answer_first"] is True
    assert response["assistant_quality_posture"]["clarification_restraint"] is True
    assert response["assistant_quality_posture"]["memory_budget"] == 2
    assert response["assistant_context_snapshot"]["route_mode"] == "conversation"
    assert response["assistant_context_snapshot"]["recent_turn_count"] <= 3
    assert response["assistant_context_snapshot"]["memory_item_count"] == 0
    assert response["assistant_voice_profile"]["style_mode"] == "collab"
    assert response["assistant_voice_profile"]["voice_profile"] == "warm_partner"
    assert response["assistant_voice_profile"]["reasoning_depth"] == "normal"
    assert response["assistant_voice_profile"]["reasoning_depth_separate"] is True
    assert response["assistant_quality_posture"]["style_mode"] == "collab"
    assert response["assistant_quality_posture"]["voice_profile"] == "warm_partner"
    assert response["assistant_quality_posture"]["reasoning_depth_separate"] is True
    assert response["provider_inference"]["provider_id"] == "local_reasoning"
    assert response["provider_inference"]["model"] == "builtin_assistant"
    assert response["provider_inference"]["response_path"] == "general_assistant"
    assert response["provider_inference"]["style_mode"] == "collab"
    assert response["provider_inference"]["voice_profile"] == "warm_partner"


def test_interaction_service_assistant_context_snapshot_stays_bounded() -> None:
    route = SimpleNamespace(mode="conversation", kind="conversation.check_in")
    interaction_profile = InteractionProfile.default()
    recent_interactions = [
        {"prompt": f"prompt {index}", "response": {"summary": f"reply {index}", "mode": "conversation"}}
        for index in range(5)
    ]
    memory_retrieval = MemoryRetrievalResult(
        query="assistant context",
        selected=[
            RetrievedMemory(
                source="memory",
                memory_kind="personal_memory",
                label=f"Memory {index}",
                summary=f"Summary {index}",
                relevance=0.9 - (index * 0.1),
                metadata={},
            )
            for index in range(4)
        ],
    )

    snapshot = InteractionService._assistant_context_snapshot(
        prompt="how are you",
        route=route,
        interaction_profile=interaction_profile,
        recent_interactions=recent_interactions,
        active_thread={"thread_summary": "Current thread summary", "objective": "Stay on topic"},
        memory_retrieval=memory_retrieval,
    )

    assert snapshot["recent_turn_count"] == 3
    assert len(snapshot["recent_turn_window"]) == 3
    assert snapshot["memory_item_count"] == 2
    assert snapshot["memory_context"] == ["Summary 0", "Summary 1"]
    assert snapshot["has_active_thread"] is True


def test_interaction_service_bypasses_clarification_for_grounded_assistant_follow_up() -> None:
    service = make_interaction_service()

    assert service._should_bypass_clarification_for_assistant_turn(
        resolved_prompt="what next",
        route=SimpleNamespace(mode="planning", kind="planning.migration"),
        active_thread={"thread_summary": "Migration cleanup", "objective": "Finish the cleanup"},
        recent_interactions=[],
    ) is True
    assert service._should_bypass_clarification_for_assistant_turn(
        resolved_prompt="xqzvbnm",
        route=SimpleNamespace(mode="conversation", kind="conversation.check_in"),
        active_thread=None,
        recent_interactions=[],
    ) is False


def test_interaction_service_marks_live_project_continuity_from_active_thread() -> None:
    service = make_interaction_service()
    service._fake_session_context_service.active_thread = {
        "thread_summary": "Routing cleanup for the desktop chat shell",
        "objective": "Finish the routing cleanup",
        "tool_context": {"tool_id": "workspace", "capability": "inspect.structure"},
    }

    response = service.ask(
        prompt="what are we doing again?",
        project_id="project-lumen",
        project_name="Lumen",
    )

    assert response["project_context_snapshot"]["project_context_active"] is True
    assert response["project_context_snapshot"]["continuity_mode"] == "live_project"
    assert response["project_context_snapshot"]["continuity_source"] == "active_thread"
    assert response["project_context_snapshot"]["project_id"] == "project-lumen"
    assert response["project_context_snapshot"]["project_name"] == "Lumen"
    assert response["project_context_snapshot"]["tool_continuity"]["tool_id"] == "workspace"
    assert response["assistant_quality_posture"]["project_context_active"] is True
    assert response["assistant_quality_posture"]["project_context_source"] == "active_thread"
    assert response["provider_inference"]["project_awareness"] == "live_project"


@pytest.mark.parametrize(
    ("style", "prompt", "expected_intent"),
    [
        ("default", "what next", "next_step"),
        ("collab", "summarize where we are", "status_summary"),
        ("direct", "keep going", "continue_work"),
        ("default", "what did we decide", "decision_recap"),
    ],
)
def test_interaction_service_answers_work_thread_followups_from_active_thread(
    style: str,
    prompt: str,
    expected_intent: str,
) -> None:
    service = make_interaction_service()
    service._fake_session_context_service.interaction_profile = InteractionProfile(
        interaction_style=style,
        reasoning_depth="normal",
        selection_source="user",
    )
    service._fake_session_context_service.active_thread = {
        "mode": "planning",
        "kind": "planning.release",
        "prompt": "prepare the release checklist",
        "normalized_topic": "release checklist",
        "thread_summary": "We are tightening Lumen's release checklist and QA pass.",
        "objective": "Plan work for: prepare the release checklist",
        "tool_context": {"tool_id": "workspace", "capability": "tests"},
    }

    response = service.ask(
        prompt=prompt,
        project_id="project-lumen",
        project_name="Lumen",
    )

    assert response["mode"] == "conversation"
    assert response["kind"] == f"conversation.work_thread_{expected_intent}"
    assert response["work_thread_continuity"]["active"] is True
    assert response["work_thread_continuity"]["intent"] == expected_intent
    assert response["work_thread_continuity"]["source"] == "active_thread"
    assert response["assistant_quality_posture"]["work_thread_continuity_active"] is True
    assert response["assistant_quality_posture"]["work_thread_intent"] == expected_intent
    assert response["project_context_snapshot"]["project_context_active"] is True
    assert response["project_context_snapshot"]["continuity_source"] == "active_thread"
    assert response["provider_inference"]["work_thread_continuity"] is True
    assert "release" in response["reply"].lower() or "checklist" in response["reply"].lower()


def test_interaction_service_keeps_general_knowledge_prompt_out_of_work_thread_surface() -> None:
    service = make_interaction_service()
    service._fake_session_context_service.active_thread = {
        "mode": "planning",
        "kind": "planning.release",
        "thread_summary": "Release checklist work",
        "objective": "Plan work for: release checklist",
    }

    response = service.ask(prompt="tell me about chemistry")

    assert response["kind"] != "conversation.work_thread_next_step"
    assert not response.get("work_thread_continuity")
    assert "chem" in str(response.get("summary") or response.get("reply") or "").lower()


def test_interaction_service_attaches_long_chat_conversation_beat_and_restrains_memory() -> None:
    recent = [
        {
            "prompt": "one",
            "mode": "conversation",
            "kind": "conversation.reply",
            "summary": "We can keep pulling on this if you want.",
            "response": {
                "mode": "conversation",
                "kind": "conversation.reply",
                "summary": "We can keep pulling on this if you want.",
                "reply": "We can keep pulling on this if you want.",
            },
        },
        {
            "prompt": "two",
            "mode": "conversation",
            "kind": "conversation.reply",
            "summary": "We could stay with it if you want.",
            "response": {
                "mode": "conversation",
                "kind": "conversation.reply",
                "summary": "We could stay with it if you want.",
                "reply": "We could stay with it if you want.",
            },
        },
        {
            "prompt": "three",
            "mode": "conversation",
            "kind": "conversation.reply",
            "summary": "That makes sense.",
            "response": {"mode": "conversation", "kind": "conversation.reply", "summary": "That makes sense."},
        },
        {
            "prompt": "four",
            "mode": "conversation",
            "kind": "conversation.reply",
            "summary": "I am with you.",
            "response": {"mode": "conversation", "kind": "conversation.reply", "summary": "I am with you."},
        },
    ]
    history = ProjectAwareInteractionHistoryService(session_recent=recent)
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=history,
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )

    response = service.ask(prompt="yeah")

    assert response["mode"] == "conversation"
    assert response["conversation_beat"]["continuity_state"] == "continuing"
    assert response["conversation_beat"]["conversation_depth"] == 5
    assert response["conversation_beat"]["response_repetition_risk"] == "high"
    assert response["conversation_beat"]["follow_up_offer_allowed"] is False
    assert response["assistant_context_snapshot"]["recent_turn_count"] == 3
    assert response["assistant_context_snapshot"]["memory_item_count"] == 0
    assert response["assistant_quality_posture"]["follow_up_offer_allowed"] is False


def test_interaction_service_returns_to_recent_conversation_thread_without_research_scaffold() -> None:
    recent = [
        {
            "prompt": "we were talking about trust",
            "mode": "conversation",
            "kind": "conversation.reply",
            "summary": "Trust was the main thread.",
            "response": {
                "mode": "conversation",
                "kind": "conversation.reply",
                "summary": "Trust was the main thread.",
                "reply": "Trust was the main thread.",
            },
        }
    ]
    history = ProjectAwareInteractionHistoryService(session_recent=recent)
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=history,
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )

    response = service.ask(prompt="go back to what we were saying")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.return_to_recent"
    assert response["conversation_beat"]["continuity_state"] == "returning"
    assert response["conversation_beat"]["topic_shift"] == "return_to_recent"
    assert "best first read" not in response["summary"].lower()
    assert "trust" in response["summary"].lower()


def test_interaction_service_keeps_generic_social_chat_out_of_project_mode() -> None:
    service = make_interaction_service()
    service._fake_session_context_service.active_thread = {
        "thread_summary": "Desktop routing cleanup",
        "objective": "Finish the cleanup",
    }

    response = service.ask(
        prompt="hello lumen",
        project_id="project-lumen",
        project_name="Lumen",
    )

    assert response["project_context_snapshot"]["project_context_active"] is False
    assert response["project_context_snapshot"]["continuity_mode"] == "general_chat"
    assert response["assistant_quality_posture"]["project_context_active"] is False
    assert response["provider_inference"]["project_awareness"] == "general_chat"


def test_interaction_service_prefers_recent_project_interactions_over_secondary_memory() -> None:
    history = ProjectAwareInteractionHistoryService(
        session_recent=[],
        project_recent=[
            {
                "prompt": "tighten the sidebar routing",
                "summary": "We should lock row identity to session ids first.",
                "response": {"mode": "planning", "summary": "We should lock row identity to session ids first."},
            }
        ],
    )
    service = make_interaction_service(interaction_history_service=history)
    service._project_id_hint = "project-lumen"

    snapshot = service._project_context_snapshot(
        session_id="default",
        prompt="keep going",
        route=SimpleNamespace(mode="conversation", kind="conversation.check_in"),
        active_thread=None,
        recent_interactions=[],
        memory_retrieval=MemoryRetrievalResult(
            query="routing cleanup",
            selected=[
                RetrievedMemory(
                    source="memory",
                    memory_kind="research_note",
                    label="Older routing note",
                    summary="Older routing memory should stay secondary.",
                    relevance=0.82,
                    metadata={},
                )
            ],
        ),
    )

    assert snapshot["project_context_active"] is True
    assert snapshot["continuity_source"] == "recent_project_interactions"
    assert snapshot["project_recent_turn_count"] == 1
    assert snapshot["secondary_project_memory_count"] == 1


def test_interaction_service_uses_project_recent_records_when_session_recent_is_empty() -> None:
    history = ProjectAwareInteractionHistoryService(
        session_recent=[],
        project_recent=[
            {
                "prompt": "summarize the migration cleanup",
                "summary": "We were tightening the desktop routing behavior.",
                "response": {"mode": "planning", "summary": "We were tightening the desktop routing behavior."},
            }
        ],
    )
    service = make_interaction_service(interaction_history_service=history)
    service._project_id_hint = "project-lumen"

    recent = service._recent_interactions_for_turn(
        InteractionTurnContext(
            original_prompt="keep going",
            effective_prompt="keep going",
            session_id="default",
            client_surface="main",
            active_thread=None,
        )
    )

    assert recent
    assert recent[0]["prompt"] == "summarize the migration cleanup"


def test_interaction_service_uses_briefer_greeting_for_direct_profile() -> None:
    service = make_interaction_service()
    service._fake_session_context_service.interaction_profile = InteractionProfile(
        interaction_style="direct",
        reasoning_depth="normal",
        selection_source="user",
    )

    response = service.ask(prompt="hello lumen")

    assert response["mode"] == "conversation"
    assert len(str(response["reply"]).split()) <= 7
    assert "glad you're here" not in response["reply"].lower()
    assert response["dialogue_management"]["interaction_mode"] == "social"
    assert response["interaction_mode"] == "social"
    assert response["tools_enabled"] is False


def test_interaction_service_uses_balanced_greeting_for_default_profile() -> None:
    service = make_interaction_service()
    service._fake_session_context_service.interaction_profile = InteractionProfile(
        interaction_style="default",
        reasoning_depth="normal",
        selection_source="user",
    )

    response = service.ask(prompt="hello lumen")

    assert response["reply"]
    assert len(str(response["reply"]).split()) <= 12
    assert "validation plan" not in response["reply"].lower()


def test_interaction_service_uses_warmer_collab_greeting_surface() -> None:
    service = make_interaction_service()
    service._fake_session_context_service.interaction_profile = InteractionProfile(
        interaction_style="collab",
        reasoning_depth="normal",
        selection_source="user",
    )
    service._fake_interaction_history_service.records = []

    response = service.ask(prompt="hi")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.greeting"
    assert response["reply"]
    assert "glad" in response["reply"].lower() or "with you" in response["reply"].lower() or "?" in response["reply"]
    assert response["reply"].endswith("?") or "glad" in response["reply"].lower() or "!" in response["reply"]


@pytest.mark.parametrize("style", ["default", "collab", "direct"])
@pytest.mark.parametrize("prompt", ["hey buddy", "hey lumen", "hi friend"])
def test_interaction_service_routes_affectionate_greetings_as_conversation_in_all_styles(
    style: str,
    prompt: str,
) -> None:
    service = make_interaction_service()
    service._fake_session_context_service.interaction_profile = InteractionProfile(
        interaction_style=style,
        reasoning_depth="deep",
        selection_source="user",
    )

    response = service.ask(prompt=prompt)

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.greeting"
    assert response["interaction_mode"] == "social"
    assert response["tools_enabled"] is False
    assert "best first read" not in str(response.get("reply") or "").lower()
    assert "validation plan" not in str(response.get("reply") or "").lower()


def test_interaction_service_explains_live_thread_conversationally() -> None:
    session_context_service = FakeSessionContextService()
    session_context_service.active_thread = {
        "session_id": "default",
        "mode": "planning",
        "kind": "planning.migration",
        "prompt": "create a migration plan for lumen routing",
        "summary": "Focus on keeping the routing layers modular.",
        "thread_summary": "keeping the routing layers modular",
        "normalized_topic": "routing",
        "objective": "keep the routing layers modular",
    }
    service = make_interaction_service(session_context_service=session_context_service)

    response = service.ask(prompt="what thread are we on?")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.thread_explanation"
    assert "routing" in response["summary"].lower()
    assert "route support" not in response["summary"].lower()
    assert "thread state" not in response["summary"].lower()


def test_interaction_service_explains_why_it_is_continuing_without_internal_jargon() -> None:
    session_context_service = FakeSessionContextService()
    session_context_service.active_thread = {
        "session_id": "default",
        "mode": "research",
        "kind": "research.summary",
        "prompt": "black holes",
        "summary": "Black holes are regions where gravity is so strong that not even light can escape.",
        "thread_summary": "black holes and the event horizon question",
        "normalized_topic": "black holes",
        "objective": "keep unpacking the event horizon question",
    }
    history = FakeInteractionHistoryService()
    history.recent_records = lambda **kwargs: [
        {"prompt": "tell me more", "mode": "research", "kind": "research.summary", "summary": "We went deeper on event horizons."}
    ]
    service = make_interaction_service(
        session_context_service=session_context_service,
        interaction_history_service=history,
    )

    response = service.ask(prompt="why are you answering like that?")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.thread_explanation"
    assert "black holes" in response["summary"].lower() or "continuing" in response["summary"].lower()
    assert "strategy" not in response["summary"].lower()
    assert "route" not in response["summary"].lower()


def test_interaction_service_softens_thread_explanation_when_active_thread_is_weak() -> None:
    session_context_service = FakeSessionContextService()
    session_context_service.active_thread = {
        "session_id": "default",
        "mode": "planning",
        "kind": "planning.migration",
        "prompt": "routing",
        "summary": "routing",
        "thread_summary": "routing",
        "normalized_topic": "routing",
        "objective": "",
    }
    history = FakeInteractionHistoryService()
    history.recent_records = lambda **kwargs: [
        {"prompt": "hello lumen", "mode": "conversation", "kind": "conversation.greeting", "summary": "Hey there."}
    ]
    service = make_interaction_service(
        session_context_service=session_context_service,
        interaction_history_service=history,
    )

    response = service.ask(prompt="what are we doing again?")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.thread_explanation"
    assert "might" in response["summary"].lower() or "don't have" in response["summary"].lower()


def test_interaction_service_handles_addressed_thread_explanation_prompt() -> None:
    session_context_service = FakeSessionContextService()
    session_context_service.active_thread = {
        "mode": "planning",
        "kind": "planning.architecture",
        "normalized_topic": "routing architecture",
        "thread_summary": "We were working through the routing architecture and its authority seams.",
        "objective": "Plan work for: design the routing architecture",
    }
    history = FakeInteractionHistoryService()
    history.recent_records = lambda **kwargs: [
        {"prompt": "tell me more", "mode": "research", "kind": "research.summary", "summary": "We kept going on routing."}
    ]
    service = make_interaction_service(
        session_context_service=session_context_service,
        interaction_history_service=history,
    )

    response = service.ask(prompt="Hey Lumen, what are we doing again?")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.thread_explanation"
    assert "routing" in response["summary"].lower()


def test_interaction_service_routes_thanks_as_conversation() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="thanks lumen")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.gratitude"
    assert response["reply"]
    assert "welcome" in response["reply"].lower() or "glad" in response["reply"].lower() or "any time" in response["reply"].lower()
    assert response["dialogue_management"]["response_strategy"] == "answer"
    assert response["interaction_mode"] == "social"
    assert response["state_control"]["core_state"] == "focus"
    assert service._fake_session_context_service.active_thread is None


def test_interaction_service_routes_slang_gratitude_as_conversation() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="thx lumen")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.gratitude"


def test_interaction_service_routes_broader_slang_gratitude_as_conversation() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="tysm lumin")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.gratitude"


def test_interaction_service_routes_soft_prefixed_greeting_as_conversation() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="yo lumen")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.greeting"


def test_interaction_service_routes_acknowledgment_with_momentum_state() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="perfect")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.acknowledgment"
    assert response["state_control"]["core_state"] == "momentum"
    assert response["state_control"]["response_bias"] == "advance"


@pytest.mark.parametrize("prompt", ["I see the issue", "i understand", "i see it"])
def test_interaction_service_routes_issue_acknowledgment_without_context_as_conversation(prompt: str) -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )

    response = service.ask(prompt=prompt)

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.acknowledgment"
    assert response["conversation_access"]["conversation_candidate_consulted"] is True
    assert response["conversation_access"]["conversation_match_type"] == "conversation.acknowledgment"
    assert response["conversation_access"]["final_source"] == "lightweight_social_conversation"
    assert "don't know" not in str(response["summary"]).lower()


def test_interaction_service_routes_how_are_you_as_conversation() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="how are you")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.check_in"
    assert response["reply"]
    assert "doing" in response["reply"].lower() or "good" in response["reply"].lower()


def test_interaction_service_routes_micro_turns_as_small_conversation() -> None:
    for prompt in ["yeah", "true", "maybe", "idk", "fair", "hmm", "nah", "okay", "lol"]:
        service = make_interaction_service()
        response = service.ask(prompt=prompt)

        assert response["mode"] == "conversation"
        assert response["kind"] == "conversation.micro_turn"
        assert response["summary"] == response["reply"]
        assert len(str(response["reply"]).split()) <= 4
        assert "next move:" not in response["reply"].lower()
        assert "validation plan" not in response["reply"].lower()
        assert "here's the clearest read" not in response["reply"].lower()


def test_interaction_service_lol_is_mode_locked_micro_turn() -> None:
    default_service = make_interaction_service()
    default_service._fake_session_context_service.interaction_profile = InteractionProfile(
        interaction_style="default",
        reasoning_depth="normal",
        selection_source="user",
    )

    direct_service = make_interaction_service()
    direct_service._fake_session_context_service.interaction_profile = InteractionProfile(
        interaction_style="direct",
        reasoning_depth="normal",
        selection_source="user",
    )

    collab_service = make_interaction_service()
    collab_service._fake_session_context_service.interaction_profile = InteractionProfile(
        interaction_style="collab",
        reasoning_depth="normal",
        selection_source="user",
    )

    default_response = default_service.ask(prompt="lol")
    direct_response = direct_service.ask(prompt="lol")
    collab_response = collab_service.ask(prompt="lol")

    assert default_response["kind"] == "conversation.micro_turn"
    assert direct_response["kind"] == "conversation.micro_turn"
    assert collab_response["kind"] == "conversation.micro_turn"
    assert len(str(direct_response["reply"]).split()) <= 2
    assert default_response["reply"] != collab_response["reply"] or "yeah" in str(collab_response["reply"]).lower()


def test_interaction_service_micro_turn_collab_is_slightly_warmer_than_default() -> None:
    default_service = make_interaction_service()
    default_service._fake_session_context_service.interaction_profile = InteractionProfile(
        interaction_style="default",
        reasoning_depth="normal",
        selection_source="user",
    )

    collab_service = make_interaction_service()
    collab_service._fake_session_context_service.interaction_profile = InteractionProfile(
        interaction_style="collab",
        reasoning_depth="normal",
        selection_source="user",
    )

    default_response = default_service.ask(prompt="fair")
    collab_response = collab_service.ask(prompt="fair")

    assert default_response["kind"] == "conversation.micro_turn"
    assert collab_response["kind"] == "conversation.micro_turn"
    assert default_response["reply"]
    assert collab_response["reply"]
    assert len(str(collab_response["reply"]).split()) >= len(str(default_response["reply"]).split()) or "yeah" in str(collab_response["reply"]).lower()


def test_interaction_service_micro_turn_direct_stays_most_minimal() -> None:
    service = make_interaction_service()
    service._fake_session_context_service.interaction_profile = InteractionProfile(
        interaction_style="direct",
        reasoning_depth="normal",
        selection_source="user",
    )

    response = service.ask(prompt="maybe")

    assert response["kind"] == "conversation.micro_turn"
    assert len(str(response["reply"]).split()) <= 2
    assert response["dialogue_management"]["idea_state"] == "refining"
    assert response["interaction_mode"] == "social"
    assert response["tools_enabled"] is False
    assert service._fake_session_context_service.active_thread is None


def test_interaction_service_routes_socially_prefixed_check_in_as_conversation() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="hey how are you?")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.check_in"


def test_interaction_service_research_black_holes_uses_local_knowledge_surface() -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        inference_service=FakeInferenceService(),
    )

    response = service.ask(prompt="research black holes")

    assert response["mode"] == "research"


def test_interaction_service_handles_multiplied_quick_math_with_x_notation() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="5+5x6")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.quick_math"
    assert response["domain_surface"]["answer"] == "35"


def test_interaction_service_handles_fragmented_educational_prompt_via_structure_layer() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="black holes like I'm smart but not a physist")

    assert response["mode"] == "research"
    assert response["kind"] == "research.summary"
    assert response.get("internal_scaffold_visible") is not True


def test_interaction_service_cleans_wrapped_black_hole_prompt_into_grounded_surface() -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        inference_service=FakeInferenceService(),
    )

    response = service.ask(prompt="Explain black holes like I'm smart but not a physist")

    assert response["mode"] == "research"
    assert "black hole" in response["summary"].lower()
    assert response.get("internal_scaffold_visible") is not True


def test_interaction_service_uses_design_tool_to_enrich_planning_architecture_response() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="design an api workflow for lumen")

    assert response["mode"] == "planning"
    assert response["kind"] == "planning.architecture"
    assert response["design_tool"]["tool_id"] == "design"
    assert response["design_spec"]["design_domain"] == "software_system"
    assert any("Components:" in step for step in response["steps"])
    assert "system spec" in response["summary"].lower() or "first-pass" in response["summary"].lower()


def test_interaction_service_answers_what_can_you_do_as_self_overview() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="what can you do")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.self_overview"
    assert "math" in response["summary"].lower()
    assert "knowledge" in response["summary"].lower()


def test_interaction_service_answers_what_all_do_you_know_as_self_overview() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="what all do you know")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.self_overview"
    assert "astronomy" in response["summary"].lower()


def test_interaction_service_answers_what_can_you_help_with_as_self_overview() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="what can you help with")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.self_overview"
    assert "help" in response["summary"].lower() or "can" in response["summary"].lower()


def test_interaction_service_answers_tell_me_about_yourself_as_self_overview() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="tell me about yourself")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.self_overview"
    assert response["self_overview_focus"] == "identity"
    assert "self_overview" in response["assistant_boundary_signals"]
    assert "research_threshold_blocked" in response["assistant_boundary_signals"]


def test_interaction_service_answers_who_are_you_as_self_overview() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="who are you")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.self_overview"
    assert response["self_overview_focus"] == "identity"


def test_interaction_service_keeps_social_self_follow_up_conversational() -> None:
    service = make_interaction_service()
    service._fake_interaction_history_service.recent_records = lambda **kwargs: [
        {
            "mode": "conversation",
            "kind": "conversation.check_in",
            "interaction_mode": "social",
            "summary": "I'm doing well and ready to help. What are we working through?",
        }
    ]

    response = service.ask(prompt="what about you?")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.self_overview"
    assert response["self_overview_source"] == "social_follow_up"
    assert "social_self_follow_up" in response["assistant_boundary_signals"]
    assert "research_threshold_blocked" in response["assistant_boundary_signals"]


def test_interaction_service_explains_recent_runtime_failure_conversationally() -> None:
    history = SparseInteractionHistoryService()
    history.recent_records = lambda **kwargs: [
        {
            "prompt": "solve 2x + 5 = 13",
            "mode": "tool",
            "kind": "tool.math.solve_equation",
            "summary": "The math.solve_equation tool reached execution but failed with RuntimeError.",
            "response": {
                "mode": "tool",
                "kind": "tool.math.solve_equation",
                "summary": "The math.solve_equation tool reached execution but failed with RuntimeError.",
                "runtime_diagnostic": {
                    "failure_stage": "execution",
                    "failure_class": "runtime_dependency_failure",
                    "tool_id": "math",
                    "capability": "solve_equation",
                    "exception_type": "RuntimeError",
                    "safe_message": "The math.solve_equation tool reached execution but failed with RuntimeError.",
                },
                "tool_runtime_status": {"failure_class": "runtime_dependency_failure"},
            },
        }
    ]
    service = make_interaction_service(interaction_history_service=history)

    response = service.ask(prompt="what went wrong")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.failure_follow_up"
    assert "execution" in response["summary"].lower()
    assert "runtimeerror" in response["summary"].lower()
    assert response["conversation_access"]["conversation_context_used"] == "diagnostic_failure"
    assert response["conversation_access"]["final_source"] == "diagnostic_conversation_follow_up"


def test_interaction_service_acknowledges_failure_follow_up_without_research_drift() -> None:
    history = SparseInteractionHistoryService()
    history.recent_records = lambda **kwargs: [
        {
            "prompt": "solve equation",
            "mode": "tool",
            "kind": "tool.math.solve_equation",
            "summary": "Need usable equation and variable before tool run.",
            "response": {
                "mode": "tool",
                "kind": "tool.math.solve_equation",
                "summary": "Need usable equation and variable before tool run.",
                "tool_execution_skipped": True,
                "tool_missing_inputs": "equation and variable",
                "runtime_diagnostic": {
                    "failure_stage": "validation",
                    "failure_class": "input_failure",
                    "tool_id": "math",
                    "capability": "solve_equation",
                    "missing_inputs": "equation and variable",
                    "safe_message": "Need usable equation and variable before tool run.",
                },
            },
        }
    ]
    service = make_interaction_service(interaction_history_service=history)

    response = service.ask(prompt="I see the issue")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.failure_acknowledgment"
    assert "tighten" in response["summary"].lower() or "narrower" in response["summary"].lower()
    assert response["conversation_access"]["conversation_context_used"] == "diagnostic_failure"
    assert response["conversation_access"]["final_source"] == "diagnostic_conversation_follow_up"


def test_interaction_service_answers_history_knowledge_self_assessment() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="how strong is your history knowledge?")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.knowledge_self_assessment"
    assert response["knowledge_self_assessment"]["domain"] == "history"
    assert "history" in response["summary"].lower()


def test_interaction_service_acknowledges_ordered_request_sequence() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="Do this, then that, then summarize it.")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.ordered_request_ack"
    assert response["ordered_request_items"][:3] == ["do this,", "that,", "summarize it"]


def test_interaction_service_keeps_what_are_you_like_conversational() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="what are you like?")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.self_overview"
    assert response["self_overview_focus"] == "identity"


def test_interaction_service_answers_generic_memory_prompt_as_bounded_memory_overview() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="what do you remember")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.memory_overview"
    assert "remember" in response["summary"].lower() or "saved details" in response["summary"].lower()


def test_interaction_service_keeps_explicit_memory_prompts_out_of_self_overview() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="what do you remember about me")

    assert response["kind"] != "conversation.self_overview"


def test_interaction_service_answers_trivial_arithmetic() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="what is 2+2?")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.quick_math"
    assert "4" in response["summary"]


def test_interaction_service_answers_trivial_arithmetic_with_trailing_question_phrase() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="4*4 is what?")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.quick_math"
    assert "16" in response["summary"]


def test_interaction_service_answers_trivial_arithmetic_with_equals_question_phrase() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="10/2 equals what?")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.quick_math"
    assert "5" in response["summary"]


def test_interaction_service_answers_trivial_arithmetic_with_operator_words() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="what's 9 minus 3?")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.quick_math"
    assert "6" in response["summary"]


def test_interaction_service_answers_trivial_arithmetic_with_casual_compute_phrase() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="can you do 12/4?")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.quick_math"
    assert "3" in response["summary"]


def test_interaction_service_carries_quick_math_follow_up() -> None:
    history = SparseInteractionHistoryService()
    history.recent_records = lambda **kwargs: [
        {
            "prompt": "what is 2+2?",
            "mode": "conversation",
            "kind": "conversation.quick_math",
            "summary": "It's 4.",
            "response": {
                "mode": "conversation",
                "kind": "conversation.quick_math",
                "summary": "It's 4.",
                "domain_surface": {"lane": "math", "expression": "2+2", "answer": "4"},
            },
        }
    ]
    service = make_interaction_service(interaction_history_service=history)

    response = service.ask(prompt="why")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.quick_math_follow_up"
    assert "2+2" in response["summary"] or "4" in response["summary"]


def test_interaction_service_explains_why_after_live_math_solve() -> None:
    history = SparseInteractionHistoryService()
    history.recent_records = lambda **kwargs: [
        {
            "prompt": "solve 3x + 2 = 11",
            "mode": "tool",
            "kind": "tool.math.solve_equation",
            "summary": "Solved equation for x: x = 3\nSubtract 2 from both sides to get 3x = 9.\nDivide both sides by 3 to get x = 3.",
            "response": {
                "mode": "tool",
                "kind": "tool.math.solve_equation",
                "summary": "Solved equation for x: x = 3\nSubtract 2 from both sides to get 3x = 9.\nDivide both sides by 3 to get x = 3.",
                "user_facing_answer": "Solved equation for x: x = 3\nSubtract 2 from both sides to get 3x = 9.\nDivide both sides by 3 to get x = 3.",
                "tool_execution": {
                    "tool_id": "math",
                    "capability": "solve_equation",
                    "params": {"equation": "3x + 2 = 11", "variable": "x"},
                },
                "domain_surface": {
                    "lane": "math",
                    "equation": "3x + 2 = 11",
                    "variable": "x",
                    "answer": "3",
                },
            },
        }
    ]
    service = make_interaction_service(interaction_history_service=history)

    response = service.ask(prompt="why")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.quick_math_follow_up"
    assert "3x + 2 = 11" in response["summary"]
    assert "subtract 2" in response["summary"].lower()
    assert "x = 3" in response["summary"].lower()


def test_interaction_service_breaks_down_knowledge_follow_up_on_current_topic() -> None:
    history = SparseInteractionHistoryService()
    history.recent_records = lambda **kwargs: [
        {
            "prompt": "What is the Milky Way?",
            "mode": "research",
            "kind": "research.summary",
            "summary": "The Milky Way is the galaxy that contains our Solar System.",
            "response": {
                "mode": "research",
                "kind": "research.summary",
                "summary": "The Milky Way is the galaxy that contains our Solar System.",
                "user_facing_answer": "The Milky Way is the galaxy that contains our Solar System.",
                "domain_surface": {
                    "lane": "knowledge",
                    "topic": "Milky Way",
                    "entry_id": "milky_way",
                },
            },
        }
    ]
    service = make_interaction_service(interaction_history_service=history)

    response = service.ask(prompt="break it down")

    assert response["mode"] == "research"
    assert response["explanation_mode"] == "break_down"
    assert "milky way" in response["summary"].lower()
    assert "plain-english version" in response["summary"].lower() or "simpler version" in response["summary"].lower()
    assert response["summary"] != "The Milky Way is the galaxy that contains our Solar System."


def test_interaction_service_goes_deeper_without_resetting_topic() -> None:
    history = SparseInteractionHistoryService()
    history.recent_records = lambda **kwargs: [
        {
            "prompt": "What is the Milky Way?",
            "mode": "research",
            "kind": "research.summary",
            "summary": "The Milky Way is the galaxy that contains our Solar System.",
            "response": {
                "mode": "research",
                "kind": "research.summary",
                "summary": "The Milky Way is the galaxy that contains our Solar System.",
                "user_facing_answer": "The Milky Way is the galaxy that contains our Solar System.",
                "domain_surface": {"lane": "knowledge", "topic": "Milky Way"},
            },
        }
    ]
    service = make_interaction_service(interaction_history_service=history)

    response = service.ask(prompt="go deeper")

    assert response["mode"] == "research"
    assert response["explanation_mode"] == "deeper"
    assert "milky way" in response["summary"].lower()
    assert "deeper" in response["summary"].lower()


def test_interaction_service_breaks_down_math_tool_result_from_current_context() -> None:
    history = SparseInteractionHistoryService()
    history.recent_records = lambda **kwargs: [
        {
            "prompt": "solve 3x + 2 = 11",
            "mode": "tool",
            "kind": "tool.math.solve_equation",
            "summary": "Solved equation for x: x = 3",
            "response": {
                "mode": "tool",
                "kind": "tool.math.solve_equation",
                "summary": "Solved equation for x: x = 3",
                "user_facing_answer": "Solved equation for x: x = 3",
                "tool_execution": {
                    "tool_id": "math",
                    "capability": "solve_equation",
                    "params": {"equation": "3x + 2 = 11", "variable": "x"},
                },
            },
        }
    ]
    session_context_service = FakeSessionContextService()
    session_context_service.active_thread = {
        "mode": "tool",
        "kind": "tool.math.solve_equation",
        "summary": "Solved equation for x: x = 3",
        "tool_context": {
            "tool_id": "math",
            "capability": "solve_equation",
            "params": {"equation": "3x + 2 = 11", "variable": "x"},
        },
    }
    service = make_interaction_service(
        interaction_history_service=history,
        session_context_service=session_context_service,
    )

    response = service.ask(prompt="break it down")

    assert response["mode"] == "conversation"
    assert response["explanation_mode"] == "break_down"
    assert "3x + 2 = 11" in response["summary"]
    assert "x = 3" in response["summary"]
    assert "subtract 2" in response["summary"].lower() or "remove the extra 2" in response["summary"].lower()


def test_interaction_service_explains_math_follow_up_step_by_step() -> None:
    history = SparseInteractionHistoryService()
    history.recent_records = lambda **kwargs: [
        {
            "prompt": "solve 3x + 2 = 11",
            "mode": "tool",
            "kind": "tool.math.solve_equation",
            "summary": "Solved equation for x: x = 3",
            "response": {
                "mode": "tool",
                "kind": "tool.math.solve_equation",
                "summary": "Solved equation for x: x = 3",
                "user_facing_answer": "Solved equation for x: x = 3",
                "tool_execution": {
                    "tool_id": "math",
                    "capability": "solve_equation",
                    "params": {"equation": "3x + 2 = 11", "variable": "x"},
                },
            },
        }
    ]
    session_context_service = FakeSessionContextService()
    session_context_service.active_thread = {
        "mode": "tool",
        "kind": "tool.math.solve_equation",
        "summary": "Solved equation for x: x = 3",
        "tool_context": {
            "tool_id": "math",
            "capability": "solve_equation",
            "params": {"equation": "3x + 2 = 11", "variable": "x"},
        },
    }
    service = make_interaction_service(
        interaction_history_service=history,
        session_context_service=session_context_service,
    )

    response = service.ask(prompt="step by step")

    assert response["mode"] == "conversation"
    assert response["explanation_mode"] == "step_by_step"
    assert "1." in response["summary"]
    assert "2." in response["summary"]
    assert "x = 3" in response["summary"]


def test_interaction_service_handles_addressed_break_down_follow_up() -> None:
    history = SparseInteractionHistoryService()
    history.recent_records = lambda **kwargs: [
        {
            "prompt": "What is gravity?",
            "mode": "research",
            "kind": "research.summary",
            "summary": "Gravity is the attraction between masses.",
            "response": {
                "mode": "research",
                "kind": "research.summary",
                "summary": "Gravity is the attraction between masses.",
                "user_facing_answer": "Gravity is the attraction between masses.",
                "domain_surface": {"lane": "knowledge", "topic": "gravity"},
            },
        }
    ]
    service = make_interaction_service(interaction_history_service=history)

    response = service.ask(prompt="Hey Lumen, break that down")

    assert response["mode"] == "research"
    assert response["explanation_mode"] == "break_down"
    assert "gravity" in response["summary"].lower()


def test_interaction_service_go_deeper_prefers_latest_topic_over_stale_math_context() -> None:
    history = SparseInteractionHistoryService()
    history.recent_records = lambda **kwargs: [
        {
            "prompt": "what is a galaxy",
            "mode": "research",
            "kind": "research.summary",
            "summary": "A galaxy is a huge collection of stars, gas, dust, and dark matter held together by gravity.",
            "response": {
                "mode": "research",
                "kind": "research.summary",
                "summary": "A galaxy is a huge collection of stars, gas, dust, and dark matter held together by gravity.",
                "user_facing_answer": "A galaxy is a huge collection of stars, gas, dust, and dark matter held together by gravity.",
                "domain_surface": {"lane": "knowledge", "topic": "galaxy"},
            },
        },
        {
            "prompt": "solve 3x + 2 = 11",
            "mode": "tool",
            "kind": "tool.math.solve_equation",
            "summary": "Solved equation for x: x = 3",
            "response": {
                "mode": "tool",
                "kind": "tool.math.solve_equation",
                "summary": "Solved equation for x: x = 3",
                "user_facing_answer": "Solved equation for x: x = 3",
                "tool_execution": {
                    "tool_id": "math",
                    "capability": "solve_equation",
                    "params": {"equation": "3x + 2 = 11", "variable": "x"},
                },
            },
        },
    ]
    session_context_service = FakeSessionContextService()
    session_context_service.active_thread = {
        "mode": "tool",
        "kind": "tool.math.solve_equation",
        "summary": "Solved equation for x: x = 3",
        "normalized_topic": "3x + 2 = 11",
        "tool_context": {
            "tool_id": "math",
            "capability": "solve_equation",
            "params": {"equation": "3x + 2 = 11", "variable": "x"},
        },
    }
    service = make_interaction_service(
        interaction_history_service=history,
        session_context_service=session_context_service,
    )

    response = service.ask(prompt="go deeper")

    assert response["mode"] == "research"
    assert response["explanation_mode"] == "deeper"
    assert "galaxy" in response["summary"].lower()
    assert "3x + 2 = 11" not in response["summary"]


def test_interaction_service_break_down_prefers_latest_topic_over_stale_math_context() -> None:
    history = SparseInteractionHistoryService()
    history.recent_records = lambda **kwargs: [
        {
            "prompt": "What is gravity?",
            "mode": "research",
            "kind": "research.summary",
            "summary": "Gravity is the attraction between masses.",
            "response": {
                "mode": "research",
                "kind": "research.summary",
                "summary": "Gravity is the attraction between masses.",
                "user_facing_answer": "Gravity is the attraction between masses.",
                "domain_surface": {"lane": "knowledge", "topic": "gravity"},
            },
        }
    ]
    session_context_service = FakeSessionContextService()
    session_context_service.active_thread = {
        "mode": "tool",
        "kind": "tool.math.solve_equation",
        "summary": "Solved equation for x: x = 3",
        "normalized_topic": "3x + 2 = 11",
        "tool_context": {
            "tool_id": "math",
            "capability": "solve_equation",
            "params": {"equation": "3x + 2 = 11", "variable": "x"},
        },
    }
    service = make_interaction_service(
        interaction_history_service=history,
        session_context_service=session_context_service,
    )

    response = service.ask(prompt="break it down")

    assert response["mode"] == "research"
    assert response["explanation_mode"] == "break_down"
    assert "gravity" in response["summary"].lower()
    assert "3x + 2 = 11" not in response["summary"]


def test_interaction_service_yes_consumes_pending_break_down_offer() -> None:
    history = SparseInteractionHistoryService()
    history.recent_records = lambda **kwargs: [
        {
            "prompt": "What is a galaxy?",
            "mode": "research",
            "kind": "research.summary",
            "summary": "A galaxy is a huge collection of stars, gas, dust, and dark matter held together by gravity.",
            "response": {
                "mode": "research",
                "kind": "research.summary",
                "summary": "A galaxy is a huge collection of stars, gas, dust, and dark matter held together by gravity.",
                "user_facing_answer": "A galaxy is a huge collection of stars, gas, dust, and dark matter held together by gravity.",
                "domain_surface": {"lane": "knowledge", "topic": "galaxy"},
            },
        }
    ]
    session_context_service = FakeSessionContextService()
    session_context_service.active_thread = {
        "mode": "research",
        "kind": "research.summary",
        "normalized_topic": "galaxy",
        "continuation_offer": {
            "kind": "break_down",
            "topic": "galaxy",
            "target_prompt": "break it down",
            "label": "I can break galaxy down more simply if you want.",
            "explanation_mode": "break_down",
        },
    }
    service = make_interaction_service(
        interaction_history_service=history,
        session_context_service=session_context_service,
    )

    response = service.ask(prompt="yes")

    assert response["mode"] == "research"
    assert response["explanation_mode"] == "break_down"
    assert "galaxy" in response["summary"].lower()
    assert "plain-english version" in response["summary"].lower() or "simpler" in response["summary"].lower()


def test_interaction_service_yes_does_not_continue_latest_topic_without_pending_offer() -> None:
    history = SparseInteractionHistoryService()
    history.recent_records = lambda **kwargs: [
        {
            "prompt": "What is a galaxy?",
            "mode": "research",
            "kind": "research.summary",
            "summary": "A galaxy is a huge collection of stars, gas, dust, and dark matter held together by gravity.",
            "response": {
                "mode": "research",
                "kind": "research.summary",
                "summary": "A galaxy is a huge collection of stars, gas, dust, and dark matter held together by gravity.",
                "user_facing_answer": "A galaxy is a huge collection of stars, gas, dust, and dark matter held together by gravity.",
                "domain_surface": {"lane": "knowledge", "topic": "galaxy"},
            },
        }
    ]
    session_context_service = FakeSessionContextService()
    session_context_service.active_thread = {
        "mode": "research",
        "kind": "research.summary",
        "normalized_topic": "galaxy",
    }
    service = make_interaction_service(
        interaction_history_service=history,
        session_context_service=session_context_service,
    )

    response = service.ask(prompt="yes")

    assert not (
        response["mode"] == "research"
        and response.get("explanation_mode") == "deeper"
        and "galaxy" in response["summary"].lower()
    )


def test_interaction_service_go_deeper_expands_black_holes() -> None:
    history = SparseInteractionHistoryService()
    history.recent_records = lambda **kwargs: [
        {
            "prompt": "Explain black holes",
            "mode": "research",
            "kind": "research.summary",
            "summary": "Black holes are regions where gravity is so strong that even light cannot escape.",
            "response": {
                "mode": "research",
                "kind": "research.summary",
                "summary": "Black holes are regions where gravity is so strong that even light cannot escape.",
                "user_facing_answer": "Black holes are regions where gravity is so strong that even light cannot escape.",
                "domain_surface": {"lane": "knowledge", "topic": "black holes"},
            },
        }
    ]
    service = make_interaction_service(interaction_history_service=history)

    response = service.ask(prompt="go deeper")

    assert response["mode"] == "research"
    assert response["explanation_mode"] == "deeper"
    assert "black hole" in response["summary"].lower()
    assert "deeper" in response["summary"].lower()


def test_interaction_service_surfaces_full_body_for_comet_prompt() -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
    )

    response = service.ask(prompt="explain to me what a comet is")

    visible = str(response.get("user_facing_answer") or response.get("summary") or "").strip()
    assert visible
    assert visible != "Here's a grounded answer."
    assert "comet" in visible.lower()
    assert "tail" in visible.lower() or "sun" in visible.lower()


def test_interaction_service_yes_consumes_live_star_break_down_offer_with_full_body() -> None:
    history = SparseInteractionHistoryService()
    session_context_service = FakeSessionContextService()
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=history,
        session_context_service=session_context_service,
    )

    first = service.ask(prompt="tell me about a star")

    assert first["mode"] == "research"
    assert isinstance(session_context_service.active_thread, dict)
    history.recent_records = lambda **kwargs: [
        {
            "prompt": "tell me about a star",
            "mode": first["mode"],
            "kind": first["kind"],
            "summary": first["summary"],
            "response": first,
        }
    ]
    session_context_service.active_thread["continuation_offer"] = {
        "kind": "break_down",
        "topic": "star",
        "target_prompt": "break it down",
        "label": "I can break that down more simply if you want.",
        "explanation_mode": "break_down",
    }

    response = service.ask(prompt="yes")

    visible = str(response.get("user_facing_answer") or response.get("summary") or "").strip()
    assert response["mode"] == "research"
    assert response["explanation_mode"] == "break_down"
    assert visible
    assert visible != "Here's a grounded answer using the best current assumptions."
    assert "star" in visible.lower()
    assert "simple" in visible.lower() or "plain-english" in visible.lower() or "main thing" in visible.lower()


def test_interaction_service_yes_consumes_pending_deeper_offer() -> None:
    history = SparseInteractionHistoryService()
    history.recent_records = lambda **kwargs: [
        {
            "prompt": "What is gravity?",
            "mode": "research",
            "kind": "research.summary",
            "summary": "Gravity is the attraction between masses.",
            "response": {
                "mode": "research",
                "kind": "research.summary",
                "summary": "Gravity is the attraction between masses.",
                "user_facing_answer": "Gravity is the attraction between masses.",
                "domain_surface": {"lane": "knowledge", "topic": "gravity"},
            },
        }
    ]
    session_context_service = FakeSessionContextService()
    session_context_service.active_thread = {
        "mode": "research",
        "kind": "research.summary",
        "normalized_topic": "gravity",
        "continuation_offer": {
            "kind": "go_deeper",
            "topic": "gravity",
            "target_prompt": "go deeper",
            "label": "Want me to go deeper?",
            "explanation_mode": "deeper",
        },
    }
    service = make_interaction_service(
        interaction_history_service=history,
        session_context_service=session_context_service,
    )

    response = service.ask(prompt="yes")

    assert response["mode"] == "research"
    assert response["explanation_mode"] == "deeper"
    assert "gravity" in response["summary"].lower()
    assert "deeper" in response["summary"].lower()


def test_interaction_service_history_follow_up_goes_deeper_on_current_topic() -> None:
    history = SparseInteractionHistoryService()
    history.recent_records = lambda **kwargs: [
        {
            "prompt": "tell me about George Washington",
            "mode": "research",
            "kind": "research.summary",
            "summary": "George Washington was a central leader in the American founding period.",
            "response": {
                "mode": "research",
                "kind": "research.summary",
                "summary": "George Washington was a central leader in the American founding period.",
                "user_facing_answer": "George Washington was a central leader in the American founding period.",
                "domain_surface": {"lane": "knowledge", "topic": "George Washington"},
            },
        }
    ]
    service = make_interaction_service(
        interaction_history_service=history,
        session_context_service=FakeSessionContextService(),
    )

    response = service.ask(prompt="go deeper")

    assert response["mode"] == "research"
    assert response["explanation_mode"] == "deeper"
    assert "george washington" in response["summary"].lower()


def test_interaction_service_answers_trivial_arithmetic_in_direct_mode_minimally() -> None:
    service = make_interaction_service()
    service._fake_session_context_service.interaction_profile = InteractionProfile(
        interaction_style="direct",
        reasoning_depth="normal",
        selection_source="user",
    )

    response = service.ask(prompt="what is 2+2?")

    assert "4" in response["summary"]
    assert len(str(response["summary"]).split()) <= 2


def test_interaction_service_routes_slang_check_in_as_conversation() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="how r u")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.check_in"


def test_interaction_service_routes_whats_up_as_presence_check_in() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="what's up")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.check_in"
    assert "ready" not in response["reply"].lower()


def test_interaction_service_routes_howve_you_been_as_presence_check_in() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="how've you been")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.check_in"
    assert "ready" not in response["reply"].lower()


def test_interaction_service_keeps_ready_prompt_in_transition_lane() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="ready?")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.transition"


def test_interaction_service_routes_what_is_up_as_conversation() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="sup")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.check_in"


def test_interaction_service_routes_broader_open_question_slang_as_conversation() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="whatcha got on ur mind")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.thought_mode"


def test_interaction_service_uses_low_confidence_recovery_for_collapsed_messy_input() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="aintnobodygonnadoitlikethat")

    assert response["mode"] == "clarification"
    assert response["clarification_context"]["clarification_trigger"] == "low_confidence_recovery"
    assert response["low_confidence_recovery"]["recovery_mode"] == "soft_clarify"
    assert response["low_confidence_recovery"]["clarifying_question_style"] == "targeted_directional_recovery"
    assert response["vibe_catcher"]["low_confidence"] is True
    assert "disagreement" in response["vibe_catcher"]["directional_signals"]
    assert response["srd_diagnostic"]["stage"] == "repair_attempt"
    assert "coherence_failure" in response["srd_diagnostic"]["failure_types"]
    assert (
        response["clarification_question"]
        == "I’m noticing we’ve got a couple directions we could go here. Do you want to explore this together, or focus on something specific?"
    )


def test_interaction_service_uses_hesitation_recovery_for_loose_messy_input() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="idk kinda not sure bout that plan")

    assert response["mode"] == "clarification"
    assert response["clarification_context"]["clarification_trigger"] == "low_confidence_recovery"
    assert response["low_confidence_recovery"]["clarifying_question_style"] == "hesitation_recovery"
    assert response["vibe_catcher"]["low_confidence"] is True
    assert "hesitation" in response["vibe_catcher"]["directional_signals"]
    assert (
        response["clarification_question"]
        == "I’m noticing we’ve got a couple directions we could go here. Do you want to explore this together, or focus on something specific?"
    )


def test_interaction_service_normalizes_messy_research_input_without_tripping() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="yo can u plz explain wtf happened with ga rn")

    assert response["mode"] == "research"
    assert response["vibe_catcher"]["normalized_prompt"] == "yo can you please explain wtf happened with ga right now"
    assert "frustration" in response["vibe_catcher"]["directional_signals"]
    assert response["low_confidence_recovery"]["recovery_mode"] in {"none", "silent_recovery"}


def test_interaction_service_handles_messy_summary_input_without_bad_clarification() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="plz summarize whats going on w the archive rn")

    assert response["mode"] == "research"
    assert response["low_confidence_recovery"]["recovery_mode"] == "silent_recovery"
    assert response["vibe_catcher"]["normalized_prompt"] == "please summarize what is going on with the archive right now"


def test_interaction_service_handles_filler_heavy_planning_input_without_bad_clarification() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="umm can ya real quick create a migration plan for lumen")

    assert response["mode"] == "planning"
    assert response["low_confidence_recovery"]["recovery_mode"] in {"none", "silent_recovery"}


def test_interaction_service_surfaces_empathy_model_for_emotionally_loaded_prompt() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="I'm overwhelmed by this migration plan and stressed")

    assert response["empathy_model"]["emotional_signal_detected"] is True
    assert response["empathy_model"]["response_sensitivity"] in {"gentle", "stabilizing"}
    assert response["empathy_model"]["grounded_acknowledgment"] is not None


def test_interaction_service_routes_thought_mode_prompt_as_conversation() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="what is on your mind")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.thought_mode"
    assert response["reply"]
    assert "thinking" in response["reply"].lower() or "mind" in response["reply"].lower() or "how " in response["reply"].lower()
    assert response["dialogue_management"]["interaction_mode"] == "social"
    assert response["interaction_mode"] == "social"
    assert response["state_control"]["core_state"] == "curiosity"
    assert response["state_control"]["response_bias"] == "explore"


def test_interaction_service_routes_thought_mode_in_direct_mode() -> None:
    service = make_interaction_service()
    service._fake_session_context_service.interaction_profile = InteractionProfile(
        interaction_style="direct",
        reasoning_depth="normal",
        selection_source="user",
    )

    response = service.ask(prompt="what's on your mind")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.thought_mode"
    assert len(str(response["reply"]).split()) <= 14
    assert "how " in response["reply"].lower() or "thread" in response["reply"].lower()


def test_interaction_service_routes_thought_mode_without_apostrophe_as_conversation() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="whats on your mind")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.thought_mode"


def test_interaction_service_routes_slangy_thought_mode_as_conversation() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="wuts on ur mind")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.thought_mode"


def test_interaction_service_routes_thinking_prompt_to_thought_mode() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="what are you thinking about")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.thought_mode"


def test_interaction_service_explains_its_own_thought_follow_up() -> None:
    service = make_interaction_service()
    service._fake_interaction_history_service.recent_records = lambda **kwargs: [
        {
            "mode": "conversation",
            "kind": "conversation.thought_mode",
            "summary": "I've been thinking about how a small assumption can change the shape of a whole idea.",
        }
    ]

    response = service.ask(prompt="what do you mean?")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.thought_follow_up"
    assert "assumption" in response["reply"].lower()
    assert "not enough grounded detail" not in response["reply"].lower()


def test_interaction_service_explains_addressed_thought_follow_up() -> None:
    service = make_interaction_service()
    service._fake_interaction_history_service.recent_records = lambda **kwargs: [
        {
            "mode": "conversation",
            "kind": "conversation.thought_mode",
            "summary": "I've been thinking about how a small assumption can change the shape of a whole idea.",
        }
    ]

    response = service.ask(prompt="Hey Lumen, what do you mean?")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.thought_follow_up"
    assert "not enough grounded detail" not in response["reply"].lower()


def test_interaction_service_uses_thought_seed_for_follow_up_explanation() -> None:
    service = make_interaction_service()
    service._fake_interaction_history_service.recent_records = lambda **kwargs: [
        {
            "mode": "conversation",
            "kind": "conversation.thought_mode",
            "summary": "I've been thinking about how framing changes what answers even become visible.",
            "response": {
                "thought_topic": "framing",
                "thought_explanation": "The framing changes what answers even become visible, so the question shape ends up doing more work than people usually notice.",
            },
        }
    ]

    response = service.ask(prompt="why?")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.thought_follow_up"
    assert "framing" in response["reply"].lower() or "question" in response["reply"].lower()
    assert "not enough grounded detail" not in response["reply"].lower()


def test_interaction_service_routes_topic_suggestion_prompt_as_conversation() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="what do you want to talk about?")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.topic_suggestion"
    assert "pick one" in response["reply"].lower() or "your call" in response["reply"].lower()
    assert "routing" not in response["reply"].lower()


def test_interaction_service_routes_socially_prefixed_topic_suggestion_as_conversation() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="hey what should we do?")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.topic_suggestion"


def test_interaction_service_routes_praise_prefixed_topic_suggestion_as_conversation() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="great job what should we do?")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.topic_suggestion"


def test_interaction_service_does_not_swallow_explicit_task_after_social_lead_in() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="hey create a migration plan for lumen")

    assert response["mode"] == "planning"


def test_interaction_service_routes_topic_suggestion_in_direct_mode() -> None:
    service = make_interaction_service()
    service._fake_session_context_service.interaction_profile = InteractionProfile(
        interaction_style="direct",
        reasoning_depth="normal",
        selection_source="user",
    )

    response = service.ask(prompt="got any ideas?")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.topic_suggestion"
    assert response["reply"].endswith("Pick one.")


def test_interaction_service_biases_topic_suggestion_with_active_thread() -> None:
    service = make_interaction_service()
    service._fake_session_context_service.active_thread = {
        "mode": "planning",
        "kind": "planning.architecture",
        "normalized_topic": "lumen routing cleanup",
    }

    response = service.ask(prompt="what should we do?")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.topic_suggestion"
    assert "lumen routing cleanup" in response["reply"].lower()


def test_interaction_service_stays_chatty_without_pushing_into_work() -> None:
    service = make_interaction_service()
    service._fake_interaction_history_service.recent_records = lambda **kwargs: [
        {
            "mode": "conversation",
            "kind": "conversation.greeting",
            "summary": "Hey. I'm here.",
        },
        {
            "mode": "conversation",
            "kind": "conversation.check_in",
            "summary": "I'm doing well.",
        },
    ]

    response = service.ask(prompt="hello lumen")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.greeting"
    assert response["reply"]
    assert "validation plan" not in response["reply"].lower()


def test_interaction_service_avoids_reusing_recent_social_reply_when_alternative_exists() -> None:
    service = make_interaction_service()
    service._fake_interaction_history_service.recent_records = lambda **kwargs: [
        {
            "mode": "conversation",
            "kind": "conversation.greeting",
            "summary": "Hey. What are we working on?",
            "reply": "Hey. What are we working on?",
        }
    ]

    response = service.ask(prompt="hello lumen")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.greeting"
    assert response["reply"] != "Hey. What are we working on?"


def test_interaction_service_uses_chatty_check_in_reply_in_social_context() -> None:
    service = make_interaction_service()
    service._fake_interaction_history_service.recent_records = lambda **kwargs: [
        {
            "mode": "conversation",
            "kind": "conversation.greeting",
            "summary": "Hey. I'm here.",
        }
    ]

    response = service.ask(prompt="how are you")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.check_in"
    assert response["reply"]
    assert "how" in response["reply"].lower() or "what" in response["reply"].lower() or "good" in response["reply"].lower()


def test_interaction_service_routes_acknowledgment_as_conversation() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="sounds good")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.acknowledgment"
    assert response["reply"]
    assert "good" in response["reply"].lower() or "alright" in response["reply"].lower() or "tracks" in response["reply"].lower()
    assert response["dialogue_management"]["interaction_mode"] == "social"
    assert response["interaction_mode"] == "social"
    assert response["tools_enabled"] is False


def test_interaction_service_routes_transition_as_conversation() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="let's do it")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.transition"
    assert response["reply"]
    assert "let" in response["reply"].lower() or "ready" in response["reply"].lower() or "start" in response["reply"].lower()
    assert response["dialogue_management"]["interaction_mode"] == "social"
    assert response["interaction_mode"] == "social"
    assert response["tools_enabled"] is False


def test_interaction_service_routes_farewell_as_conversation() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="see you later")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.farewell"
    assert response["reply"] in {
        "See you.",
        "Talk soon.",
        "Take care.",
        "Alright, talk soon.",
        "Sounds good, talk soon.",
        "Alright, we'll pick this up later.",
        "Let me know when you're back.",
        "See you later. I got you when you're back.",
        "Talk soon. I'll be here.",
        "Take care. We'll pick this up later.",
        "Alright, I'll be here.",
        "Cool, I'll be here when you're ready.",
        "Glad we got somewhere today. Talk soon.",
        "Take care. We'll keep going when you're back.",
        "Absolutely. See you later.",
    }
    assert response["dialogue_management"]["interaction_mode"] == "social"
    assert response["interaction_mode"] == "social"
    assert response["tools_enabled"] is False


def test_interaction_service_routes_terminal_gratitude_farewell_as_conversation() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="thanks, see you later")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.farewell"
    assert "see you later" in response["reply"].lower() or "talk soon" in response["reply"].lower()
    assert response["interaction_mode"] == "social"


def test_interaction_service_routes_brb_as_conversation_farewell() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="brb")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.farewell"
    assert response["interaction_mode"] == "social"


def test_interaction_service_avoids_repeating_farewell_reply_across_turns() -> None:
    service = make_interaction_service()

    first = service.ask(prompt="see ya")
    second = service.ask(prompt="talk later")

    assert first["kind"] == "conversation.farewell"
    assert second["kind"] == "conversation.farewell"
    assert first["reply"] != second["reply"]


def test_interaction_service_uses_direct_farewell_tone() -> None:
    service = make_interaction_service()
    service._fake_session_context_service.interaction_profile = InteractionProfile(
        interaction_style="direct",
        reasoning_depth="normal",
        selection_source="user",
        allow_suggestions=True,
    )

    response = service.ask(prompt="goodnight")

    assert response["kind"] == "conversation.farewell"
    assert response["reply"] in {"Okay.", "Later.", "Talk later.", "Talk soon."}


def test_interaction_service_uses_collab_farewell_tone() -> None:
    service = make_interaction_service()
    service._fake_session_context_service.interaction_profile = InteractionProfile(
        interaction_style="conversational",
        reasoning_depth="normal",
        selection_source="user",
        allow_suggestions=True,
    )

    response = service.ask(prompt="talk later")

    assert response["kind"] == "conversation.farewell"
    assert response["reply"] in {
        "See you later. I got you when you're back.",
        "Talk soon. I'll be here.",
        "Take care. We'll pick this up later.",
        "Alright, I'll be here.",
        "Cool, I'll be here when you're ready.",
        "Let me know when you're back.",
        "Sounds good, talk soon.",
        "Glad we got somewhere today. Talk soon.",
        "Take care. We'll keep going when you're back.",
        "Absolutely. See you later.",
    }


def test_interaction_service_routes_bare_lumen_as_social_greeting() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="lumen")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.greeting"
    assert response["dialogue_management"]["interaction_mode"] == "social"
    assert response["interaction_mode"] == "social"


def test_interaction_service_routes_goodbye_lumen_as_social_farewell() -> None:
    service = make_interaction_service()
    service._fake_session_context_service.active_thread = {
        "mode": "research",
        "kind": "research.summary",
        "thread_summary": "Researching cosmology",
    }

    response = service.ask(prompt="Goodbye Lumen!")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.farewell"
    assert response["dialogue_management"]["interaction_mode"] == "social"
    assert response["interaction_mode"] == "social"


def test_interaction_service_adds_conversational_turn_for_exploratory_response() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="what do you think about this migration idea?")

    assert response["mode"] in {"planning", "research"}
    assert response["dialogue_management"]["interaction_mode"] == "hybrid"
    assert response["response_strategy"] == "expand"
    assert response["conversation_turn"]["kind"] == "explore"
    assert response["conversation_turn"]["lead"] == "There’s something worth exploring here."
    assert response["response_intro"] == "There’s something worth exploring here."
    assert response["response_opening"] == (
        "Let's keep the thread open long enough to see which direction earns more confidence."
    )
    assert response["conversation_turn"]["follow_ups"] == [
        "What assumption are we actually testing here?",
        "Have you considered the strongest alternative explanation?",
    ]


def test_interaction_service_adds_checkpoint_turn_when_synthesis_checkpoint_is_due() -> None:
    service = make_interaction_service()
    service._fake_interaction_history_service.records = [
        {"response": {"mode": "research", "dominant_intent": "research"}},
        {"response": {"mode": "research", "dominant_intent": "research"}},
        {"response": {"mode": "research", "dominant_intent": "research"}},
    ]

    response = service.ask(prompt="what do you think about this migration idea?")

    assert response["dialogue_management"]["synthesis_checkpoint_due"] is True
    assert response["conversation_turn"]["kind"] == "collaborate"
    assert "Here's my read so far" in response["conversation_turn"]["lead"]
    assert "Let me pull the threads together so we can see what still holds and what still needs work." in response["conversation_turn"]["partner_frame"]
    assert response["checkpoint_summary"]["next_step"] == "Capture the current synthesis, then decide whether to refine or branch."


def test_interaction_service_reorients_cleanly_on_where_are_we_now() -> None:
    service = make_interaction_service()
    service._fake_session_context_service.active_thread = {
        "mode": "planning",
        "kind": "planning.architecture",
        "thread_summary": "Migration plan focused on routing cleanup",
        "objective": "Tighten the migration plan",
    }
    service._fake_interaction_history_service.recent_records = lambda **kwargs: [
        {
            "mode": "planning",
            "summary": "Migration plan focused on routing cleanup",
            "response": {
                "research_questions": ["What assumption should we tighten first?"],
            },
        }
    ]

    response = service.ask(prompt="where are we now?")

    assert response["dialogue_management"]["interaction_mode"] == "synthesis"
    assert response["thought_framing"]["response_kind_label"] == "thread_reorientation"
    assert response["checkpoint_summary"]["current_direction"] == "Migration plan focused on routing cleanup"
    assert response["checkpoint_summary"]["weakest_point"] == (
        "The main unresolved point is still: What assumption should we tighten first?"
    )
    assert response["checkpoint_summary"]["next_step"] == (
        "Re-anchor on the live thread, then either resolve the main open question or tighten the next step."
    )


def test_interaction_service_shapes_question_turn_as_curious_exploration() -> None:
    response: dict[str, object] = {}

    InteractionService._attach_conversational_turn(
        response=response,
        interaction_profile=InteractionProfile.default(),
        dialogue_management=DialogueManagementResult(
            interaction_mode="clarification",
            idea_state="uncertain",
            response_strategy="ask_question",
        ),
        conversation_awareness=ConversationAwarenessResult(
            adaptive_posture="step_back",
        ),
        thought_framing=ThoughtFramingResult(
            response_kind_label="research_question",
            conversation_activity="reducing ambiguity before committing to a stronger answer",
            research_questions=[
                "What do you mean by the core assumption here?",
                "Do you want me to expand the idea or critique it?",
            ],
        ),
    )

    assert response["conversation_turn"]["kind"] == "question"
    assert response["conversation_turn"]["lead"] in {
        "Before we push this further, I'd want to ask: What do you mean by the core assumption here?",
        "Before we move too fast, I'd want to pin this down: What do you mean by the core assumption here?",
        "Before we lean on this too hard, I'd want to clarify this first: What do you mean by the core assumption here?",
        "Before we move too fast, I'd want to ask: What do you mean by the core assumption here?",
        "Before we push this further, I'd want to pin this down: What do you mean by the core assumption here?",
        "Before we lean on this too hard, I'd want to ask: What do you mean by the core assumption here?",
    }
    assert response["conversation_turn"]["follow_ups"] == [
        "Do you want me to expand the idea or critique it?",
    ]
    assert response["conversation_turn"]["adaptive_posture"] == "step_back"


def test_interaction_service_surfaces_supportive_challenge_when_support_is_weak() -> None:
    response: dict[str, object] = {
        "support_status": "insufficiently_grounded",
        "tension_status": "under_tension",
        "route_status": "weakened",
    }

    InteractionService._attach_conversational_turn(
        response=response,
        interaction_profile=InteractionProfile.default(),
        dialogue_management=DialogueManagementResult(
            interaction_mode="hybrid",
            idea_state="exploring",
            response_strategy="expand",
        ),
        thought_framing=ThoughtFramingResult(
            response_kind_label="exploratory_expansion",
            conversation_activity="exploring an idea while keeping analytical structure",
            research_questions=[
                "What assumption are we actually testing here?",
                "Have you considered the strongest alternative explanation?",
                "Are you trying to explain the idea, test it, or turn it into something usable?",
            ],
        ),
    )

    assert response["conversation_turn"]["kind"] == "challenge"
    assert response["conversation_turn"]["lead"] in {
        "I like the direction, but let's test one assumption: What assumption are we actually testing here?",
        "I like the line of thought, but let's test one assumption: What assumption are we actually testing here?",
        "I like where this is going, but let's test one assumption: What assumption are we actually testing here?",
    }
    assert response["conversation_turn"]["follow_ups"] == [
        "Have you considered the strongest alternative explanation?",
        "Are you trying to explain the idea, test it, or turn it into something usable?",
    ]


def test_interaction_service_adds_qualified_stance_in_agreement_heavy_turns() -> None:
    response: dict[str, object] = {
        "support_status": "insufficiently_grounded",
        "tension_status": "under_tension",
        "route_status": "weakened",
    }

    InteractionService._attach_conversational_turn(
        prompt="you're right",
        response=response,
        interaction_profile=InteractionProfile.default(),
        dialogue_management=DialogueManagementResult(
            interaction_mode="hybrid",
            idea_state="exploring",
            response_strategy="answer",
        ),
        conversation_awareness=ConversationAwarenessResult(
            adaptive_posture="acknowledge",
            recent_intent_pattern="agreeing",
        ),
        human_language_layer=SimpleNamespace(
            emotional_alignment="steady",
            correction_detected=False,
            epistemic_stance="exploratory",
            stance_confidence="medium",
            user_energy="casual",
        ),
        thought_framing=ThoughtFramingResult(
            response_kind_label="analysis",
            conversation_activity="working through the idea",
            research_questions=["What assumption should we tighten first?"],
        ),
    )

    assert response["stance_consistency"]["category"] == "agreement_with_qualification"
    assert "qualify" in response["conversation_turn"]["lead"].lower()
    assert "you're right" not in response["conversation_turn"]["lead"].lower()


def test_interaction_service_marks_stance_reversal_without_silent_flip() -> None:
    response: dict[str, object] = {}

    InteractionService._attach_conversational_turn(
        prompt="i disagree with that part",
        response=response,
        interaction_profile=InteractionProfile.default(),
        dialogue_management=DialogueManagementResult(
            interaction_mode="hybrid",
            idea_state="exploring",
            response_strategy="answer",
        ),
        conversation_awareness=ConversationAwarenessResult(
            adaptive_posture="acknowledge",
            recent_intent_pattern="disagreeing",
        ),
        human_language_layer=SimpleNamespace(
            emotional_alignment="steady",
            correction_detected=False,
            epistemic_stance="assertive",
            stance_confidence="high",
            user_energy="focused",
        ),
        thought_framing=ThoughtFramingResult(
            response_kind_label="analysis",
            conversation_activity="working through the idea",
            research_questions=["What assumption should we tighten first?"],
        ),
        recent_interactions=[
            {
                "response": {
                    "stance_consistency": {
                        "category": "full_agreement",
                    }
                }
            }
        ],
    )

    assert response["stance_consistency"]["category"] == "respectful_disagreement"
    assert response["stance_consistency"]["contradiction_aware"] is True
    assert "On this part" in response["conversation_turn"]["lead"]


def test_interaction_service_prefers_question_over_supportive_challenge_when_user_is_hesitating() -> None:
    response: dict[str, object] = {
        "support_status": "insufficiently_grounded",
        "tension_status": "under_tension",
        "route_status": "weakened",
    }

    InteractionService._attach_conversational_turn(
        response=response,
        interaction_profile=InteractionProfile.default(),
        dialogue_management=DialogueManagementResult(
            interaction_mode="hybrid",
            idea_state="exploring",
            response_strategy="expand",
        ),
        conversation_awareness=ConversationAwarenessResult(
            adaptive_posture="step_back",
            recent_intent_pattern="hesitating",
        ),
        thought_framing=ThoughtFramingResult(
            response_kind_label="exploratory_expansion",
            conversation_activity="exploring an idea while keeping analytical structure",
            research_questions=[
                "What assumption are we actually testing here?",
                "Have you considered the strongest alternative explanation?",
            ],
        ),
    )

    assert response["conversation_turn"]["kind"] == "explore"
    assert response["conversation_turn"]["lead"] == "There’s something worth exploring here."


def test_interaction_service_detects_response_to_response_bridge_from_topic_suggestion() -> None:
    bridge = InteractionService._response_to_response_bridge(
        prompt="let's do black holes",
        recent_interactions=[
            {
                "mode": "conversation",
                "kind": "conversation.topic_suggestion",
                "summary": "We could go a few directions: black holes, a systems question, or propulsion concepts. Your call.",
                "response": {
                    "reply": "We could go a few directions: black holes, a systems question, or propulsion concepts. Your call.",
                },
            }
        ],
    )

    assert bridge == {
        "category": "direct_acceptance",
        "target": "black holes",
    }


def test_interaction_service_detects_soft_direction_pickup_from_lumens_prompt() -> None:
    bridge = InteractionService._response_to_response_bridge(
        prompt="probably the architecture thing",
        recent_interactions=[
            {
                "mode": "conversation",
                "kind": "conversation.topic_suggestion",
                "summary": "We could go a few directions: astronomy, the architecture thing, or propulsion concepts. Pick one.",
                "response": {
                    "reply": "We could go a few directions: astronomy, the architecture thing, or propulsion concepts. Pick one.",
                },
            }
        ],
    )

    assert bridge == {
        "category": "hesitant_acceptance",
        "target": "the architecture thing",
    }


def test_interaction_service_adds_pickup_bridge_and_follow_through_starter() -> None:
    response: dict[str, object] = {}

    InteractionService._attach_conversational_turn(
        prompt="let's do black holes",
        response=response,
        interaction_profile=InteractionProfile(
            interaction_style="collab",
            reasoning_depth="normal",
            selection_source="user",
        ),
        dialogue_management=DialogueManagementResult(
            interaction_mode="hybrid",
            idea_state="exploring",
            response_strategy="answer",
        ),
        conversation_awareness=ConversationAwarenessResult(
            adaptive_posture="acknowledge",
        ),
        human_language_layer=SimpleNamespace(
            emotional_alignment="steady",
            correction_detected=False,
            epistemic_stance="exploratory",
            stance_confidence="medium",
            user_energy="casual",
        ),
        thought_framing=ThoughtFramingResult(
            response_kind_label="analysis",
            conversation_activity="following the user's chosen direction",
            research_questions=["What part of black holes should we start with?"],
        ),
        recent_interactions=[
            {
                "mode": "conversation",
                "kind": "conversation.topic_suggestion",
                "summary": "We could go a few directions: black holes, a systems question, or propulsion concepts. Your call.",
                "response": {
                    "reply": "We could go a few directions: black holes, a systems question, or propulsion concepts. Your call.",
                },
            }
        ],
    )

    assert response["conversation_turn"]["response_to_response_handoff"] is True
    assert response["conversation_turn"]["handoff_target"] == "black holes"
    assert response["conversation_turn"]["pickup_bridge"] in {
        "Nice, yeah - let's go there.",
        "Okay, I'm with you.",
        "Yeah, let's do that.",
        "That works - let's start there.",
        "Perfect, that's a good direction.",
    }
    assert response["conversation_turn"]["follow_through_starter"] in {
        "So with black holes, the key thing is this.",
        "Alright, first thing with black holes is this.",
        "Okay, let's stay with black holes for a second.",
    }


def test_interaction_service_adds_deep_collaboration_frame_for_exploratory_turns() -> None:
    service = make_interaction_service()
    service._fake_session_context_service.interaction_profile = InteractionProfile(
        interaction_style="conversational",
        reasoning_depth="deep",
        selection_source="user",
    )

    response = service.ask(prompt="what do you think about this migration idea?")

    assert response["conversation_turn"]["kind"] == "explore"
    assert response["conversation_turn"]["partner_frame"] == (
        "Let's widen the idea a little, then decide which line is actually worth keeping."
    )
    assert response["conversation_turn"]["next_move"] == (
        "What assumption are we actually testing here?"
    )


def test_interaction_service_keeps_main_thread_visible_during_side_branch() -> None:
    service = make_interaction_service()
    service._fake_session_context_service.active_thread = {
        "mode": "planning",
        "kind": "planning.architecture",
        "thread_summary": "Migration plan focused on routing cleanup",
        "objective": "Tighten the migration plan",
    }

    response = service.ask(prompt="what about a plugin route instead")

    assert response["dialogue_management"]["idea_state"] == "branching"
    assert response["conversation_awareness"]["branch_state"] == "side_branch_open"
    assert response["conversation_awareness"]["return_target"] == "Migration plan focused on routing cleanup"
    assert response["thought_framing"]["branch_return_hint"] == (
        "We can follow this branch, but the main thread to return to is: Migration plan focused on routing cleanup"
    )
    assert response["conversation_turn"]["branch_return_hint"] == (
        "We can follow this branch, but the main thread to return to is: Migration plan focused on routing cleanup"
    )
    assert not any(
        "main thread to return to is: Migration plan focused on routing cleanup" in item
        for item in response.get("steps", []) + response.get("findings", [])
    )


def test_interaction_service_reanchors_cleanly_when_user_returns_to_main_thread() -> None:
    service = make_interaction_service()
    service._fake_session_context_service.active_thread = {
        "mode": "planning",
        "kind": "planning.architecture",
        "thread_summary": "Migration plan focused on routing cleanup",
        "objective": "Tighten the migration plan",
    }

    response = service.ask(prompt="go back to the main thread")

    assert response["conversation_awareness"]["branch_state"] == "returning_to_main"
    assert response["conversation_awareness"]["return_requested"] is True
    assert response["thought_framing"]["branch_return_hint"] == (
        "We're back on the main thread: Migration plan focused on routing cleanup"
    )
    assert response["conversation_turn"]["branch_return_hint"] == (
        "We're back on the main thread: Migration plan focused on routing cleanup"
    )
    assert response["conversation_turn"]["partner_frame"] == (
        "Let's re-anchor on the main thread, keep Migration plan focused on routing cleanup in view, and move it forward cleanly."
    )
    assert not any(
        "We're back on the main thread: Migration plan focused on routing cleanup" == item
        for item in response.get("steps", []) + response.get("findings", [])
    )


def test_interaction_service_reorients_cleanly_on_where_are_we_now() -> None:
    service = make_interaction_service()
    service._fake_session_context_service.active_thread = {
        "mode": "planning",
        "kind": "planning.architecture",
        "thread_summary": "Migration plan focused on routing cleanup",
        "objective": "Tighten the migration plan",
    }
    service._fake_interaction_history_service.recent_records = lambda **kwargs: [
        {
            "mode": "planning",
            "summary": "Migration plan focused on routing cleanup",
            "response": {
                "research_questions": ["What assumption should we tighten first?"],
            },
        }
    ]

    response = service.ask(prompt="where are we now?")

    assert response["dialogue_management"]["interaction_mode"] == "synthesis"
    assert response["thought_framing"]["response_kind_label"] == "thread_reorientation"
    if response.get("conversation_turn") is not None:
        assert response["conversation_turn"]["kind"] == "checkpoint"
    assert response["checkpoint_summary"]["current_direction"] == "Migration plan focused on routing cleanup"
    assert response["checkpoint_summary"]["weakest_point"] == (
        "The main unresolved point is still: What assumption should we tighten first?"
    )
    assert response["checkpoint_summary"]["next_step"] == (
        "Re-anchor on the live thread, then either resolve the main open question or tighten the next step."
    )


def test_interaction_service_carries_live_unresolved_question_forward() -> None:
    service = make_interaction_service()
    service._fake_session_context_service.active_thread = {
        "mode": "planning",
        "kind": "planning.architecture",
        "thread_summary": "Migration plan focused on routing cleanup",
        "objective": "Tighten the migration plan",
    }
    service._fake_interaction_history_service.recent_records = lambda **kwargs: [
        {
            "mode": "planning",
            "summary": "Planning response for routing cleanup",
            "response": {
                "research_questions": ["What assumption should we tighten first?"],
                "conversation_turn": {
                    "next_move": "What assumption should we tighten first?",
                },
            },
        }
    ]

    response = service.ask(prompt="keep going")

    assert response["conversation_awareness"]["live_unresolved_question"] == "What assumption should we tighten first?"
    assert response["research_questions"][0] == "What assumption should we tighten first?"
    if response.get("conversation_turn") is not None:
        assert response["conversation_turn"]["next_move"] == "What assumption should we tighten first?"


def test_interaction_service_keeps_follow_through_turns_converged_on_live_question() -> None:
    service = make_interaction_service()
    service._fake_session_context_service.active_thread = {
        "mode": "planning",
        "kind": "planning.architecture",
        "thread_summary": "Migration plan focused on routing cleanup",
        "objective": "Tighten the migration plan",
    }
    service._fake_interaction_history_service.recent_records = lambda **kwargs: [
        {
            "mode": "planning",
            "summary": "Planning response for routing cleanup",
            "response": {
                "research_questions": ["What assumption should we tighten first?"],
                "conversation_turn": {
                    "next_move": "What assumption should we tighten first?",
                },
            },
        }
    ]

    response = service.ask(prompt="what else")

    assert response["conversation_awareness"]["recent_intent_pattern"] == "following_through"
    assert response["conversation_awareness"]["live_unresolved_question"] == "What assumption should we tighten first?"
    if response.get("conversation_turn") is not None:
        assert response["conversation_turn"]["next_move"] == "What assumption should we tighten first?"


def test_interaction_service_adds_deep_collaboration_turn_for_analytical_answers() -> None:
    service = make_interaction_service()
    service._fake_session_context_service.interaction_profile = InteractionProfile(
        interaction_style="conversational",
        reasoning_depth="deep",
        selection_source="user",
    )

    response = service.ask(prompt="create a migration plan for lumen")

    assert response["dialogue_management"]["response_strategy"] == "answer"
    assert response["conversation_turn"]["kind"] == "collaborate"
    assert "Here's my read so far" in response["conversation_turn"]["lead"]
    assert response["conversation_turn"]["partner_frame"] == (
        "I'll give you my read so far, but we should keep the live uncertainty in view."
    )
    assert response["conversation_turn"]["next_move"] == "What assumption should we tighten first?"


def test_interaction_service_adds_thread_holding_turn_for_normal_conversational_answers() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="create a migration plan for lumen")

    assert response["dialogue_management"]["response_strategy"] == "answer"
    assert response["conversation_turn"]["kind"] == "collaborate"
    assert "Here's my read so far" in response["conversation_turn"]["lead"]
    assert response["response_intro"] == response["conversation_turn"]["lead"]
    assert "I'll give you my read, keep the thread in view, and push it one step further." in response["response_opening"]
    assert "I'll give you my read, keep the thread in view, and push it one step further." in response["conversation_turn"]["partner_frame"]
    assert response["conversation_turn"]["next_move"] == "What assumption should we tighten first?"
    assert response["conversation_turn"]["adaptive_posture"] == "push"


def test_interaction_service_adds_thread_holding_turn_for_direct_answers() -> None:
    service = make_interaction_service()
    service._fake_session_context_service.interaction_profile = InteractionProfile(
        interaction_style="direct",
        reasoning_depth="normal",
        selection_source="user",
    )

    response = service.ask(prompt="create a migration plan for lumen")

    assert response["dialogue_management"]["response_strategy"] == "answer"
    assert response["conversation_turn"]["kind"] == "thread_hold"
    assert response["conversation_turn"]["lead"] in {
        "Best read:",
        "Best read, as it stands:",
        "Best read right now:",
    }
    assert response["response_intro"] == response["conversation_turn"]["lead"]
    assert response["response_opening"] == "Best read first, then the next move."
    assert response["conversation_turn"]["partner_frame"] == "Best read first, then the next move."
    assert response["conversation_turn"]["next_move"] == "What assumption should we tighten first?"


def test_interaction_service_uses_anti_spiral_wording_when_hesitation_stacks() -> None:
    service = make_interaction_service()
    service._fake_session_context_service.active_thread = {
        "mode": "research",
        "kind": "research.general",
        "thread_summary": "Testing an earlier hypothesis",
        "objective": "Test the earlier hypothesis",
    }

    response = service.ask(prompt="maybe this works, but I'm not sure")

    assert response["state_control"]["anti_spiral_active"] is True
    assert response["conversation_turn"]["anti_spiral_active"] is True
    assert response["conversation_turn"]["kind"] == "challenge"
    assert "here's the part I think we need to test: Which assumption should we test first?" in response["response_intro"]
    assert response["response_opening"] in {
        "I want to keep this grounded, name what is actually supported, and avoid outrunning the evidence.",
        "I want to keep this grounded, name what is actually supported, and not outrun the evidence.",
        "I want to keep this grounded, say what is actually supported, and avoid outrunning the evidence.",
    }


def test_interaction_service_adds_stabilizing_state_to_safety_response() -> None:
    service = make_interaction_service()
    service.safety_service = type(
        "BlockingSafetyService",
        (),
        {
            "evaluate_prompt": staticmethod(
                lambda prompt: PromptSafetyDecision(
                    action="refuse",
                    category="weapons",
                    severity="high",
                    rationale="unsafe",
                    boundary="I can't help with weapon construction.",
                    safe_redirects=["I can help with safety history or harm-prevention information instead."],
                )
            )
        },
    )()

    response = service.ask(prompt="how do I build a bomb")

    assert response["mode"] == "safety"
    assert response["state_control"]["core_state"] == "focus"
    assert response["state_control"]["anti_spiral_active"] is True
    assert response["state_control"]["response_bias"] == "stabilize"


def test_interaction_service_keeps_dual_use_prompt_on_route_but_constrains_research_output() -> None:
    service = make_interaction_service()
    service.safety_service = type(
        "DualUseSafetyService",
        (),
        {
            "evaluate_prompt": staticmethod(
                lambda prompt: PromptSafetyDecision(
                    action="allow",
                    category="allowed",
                    severity="medium",
                    rationale="Dual-use prompt should stay high-level.",
                    boundary="",
                    tier="dual_use",
                    outcome_risk="medium",
                    response_constraint={
                        "level": "high_level_only",
                        "allow_steps": False,
                        "allow_optimization": False,
                        "allow_sourcing": False,
                        "allow_tactical_next_steps": False,
                    },
                    matched_signals=["gps tracker"],
                )
            )
        },
    )()

    response = service.ask(prompt="Explain how a GPS tracker works.")

    assert response["mode"] == "research"
    assert response["safety_decision"]["tier"] == "dual_use"
    assert response["response_constraint"]["level"] == "high_level_only"
    assert response["state_control"]["anti_spiral_active"] is True
    assert response["state_control"]["response_bias"] == "stabilize"
    assert response["findings"] == [
        "Keep the explanation high-level and non-operational.",
        "Focus on general principles, safety limits, and benign alternatives rather than tactics or optimization.",
    ]
    assert "safe background" in response["recommendation"].lower()


def test_interaction_service_skips_constrained_tool_execution_for_dual_use_prompt() -> None:
    service = make_interaction_service()
    service.safety_service = type(
        "DualUseToolSafetyService",
        (),
        {
            "evaluate_prompt": staticmethod(
                lambda prompt: PromptSafetyDecision(
                    action="allow",
                    category="allowed",
                    severity="medium",
                    rationale="Dual-use prompt should stay high-level.",
                    boundary="",
                    tier="dual_use",
                    outcome_risk="medium",
                    response_constraint={
                        "level": "high_level_only",
                        "allow_steps": False,
                        "allow_optimization": False,
                        "allow_sourcing": False,
                        "allow_tactical_next_steps": False,
                    },
                    tool_constraint={
                        "level": "constrained",
                        "reason": "dual_use_prompt",
                        "allow_execution": False,
                    },
                    matched_signals=["gps tracker"],
                )
            ),
            "capability_safety_profile": staticmethod(
                lambda **kwargs: {
                    "level": "constrained",
                    "notes": "Tool execution stays blocked for dual-use prompts.",
                }
            ),
        },
    )()

    response = service.ask(prompt="inspect workspace")

    assert response["mode"] == "tool"
    assert response["tool_execution_skipped"] is True
    assert response["tool_execution_skipped_reason"] == "dual_use_constraint"
    assert response["tool_capability_safety"]["level"] == "constrained"
    assert response["tool_constraint"]["allow_execution"] is False
    assert response["summary"].startswith("I can stay with that at a high level")


def test_interaction_service_final_surface_enforcer_cleans_conversational_lane() -> None:
    response = {
        "mode": "conversation",
        "summary": "Hey. What are we looking at?",
        "reply": "Hey. What are we looking at?",
        "steps": ["Validation plan: tighten the route first."],
        "recommendation": "Next move: tighten the route first.",
        "user_facing_answer": "Here's the clearest explanation.",
    }

    InteractionService._enforce_final_surface_lane(
        response=response,
        selected_mode="conversation",
    )

    assert response["final_surface_lane"] == "conversational"
    assert "steps" not in response
    assert "recommendation" not in response
    assert "user_facing_answer" not in response


def test_interaction_service_final_surface_enforcer_cleans_answer_lane() -> None:
    response = {
        "mode": "research",
        "summary": "Black holes are regions where gravity is so strong that even light cannot escape.",
        "user_facing_answer": "Black holes are regions where gravity is so strong that even light cannot escape.",
        "conversation_turn": {"lead": "Hey. What are we looking at?"},
        "response_intro": "Hey. What are we looking at?",
        "response_opening": "Still here. Want to keep going together?",
        "findings": ["Best read:", "Black holes are regions where gravity is so strong that even light cannot escape."],
        "recommendation": "Want to keep going together?",
    }

    InteractionService._enforce_final_surface_lane(
        response=response,
        selected_mode="research",
    )

    assert response["final_surface_lane"] == "answer"
    assert response["summary"] == response["user_facing_answer"]
    assert "conversation_turn" not in response
    assert "response_intro" not in response
    assert "response_opening" not in response
    assert response["findings"] == []


def test_interaction_service_final_surface_enforcer_keeps_fallback_lane_honest() -> None:
    response = {
        "mode": "research",
        "summary": "I don't have enough grounded detail to answer that confidently yet.",
        "user_facing_answer": "I don't have enough grounded detail to answer that confidently yet.",
        "conversation_turn": {"lead": "Hey. What are we looking at?"},
        "response_intro": "Hey. What are we looking at?",
        "findings": ["Here's the clearest explanation."],
    }

    InteractionService._enforce_final_surface_lane(
        response=response,
        selected_mode="research",
    )

    assert response["final_surface_lane"] == "fallback"
    assert response["summary"] == "I don't have enough grounded detail to answer that confidently yet."
    assert "conversation_turn" not in response
    assert response["findings"] == []


def test_interaction_service_final_surface_enforcer_keeps_planning_lane_structured() -> None:
    response = {
        "mode": "planning",
        "steps": [
            "Glad to see you. What are we working on?",
            "Start by tightening the route arbitration.",
        ],
        "next_action": "Good to have you back. Next move: tighten the route arbitration.",
    }

    InteractionService._enforce_final_surface_lane(
        response=response,
        selected_mode="planning",
    )

    assert response["final_surface_lane"] == "planning"
    assert response["steps"] == ["Start by tightening the route arbitration."]
    assert "next_action" not in response


def test_interaction_service_final_surface_enforcer_does_not_change_route_metadata() -> None:
    response = {
        "mode": "research",
        "kind": "research.summary",
        "summary": "Black holes are regions where gravity is so strong that even light cannot escape.",
        "user_facing_answer": "Black holes are regions where gravity is so strong that even light cannot escape.",
        "route": {"mode": "research", "kind": "research.summary", "source": "explicit_summary"},
        "conversation_turn": {"lead": "Hey. What are we looking at?"},
    }

    InteractionService._enforce_final_surface_lane(
        response=response,
        selected_mode="research",
    )

    assert response["mode"] == "research"
    assert response["kind"] == "research.summary"
    assert response["route"] == {"mode": "research", "kind": "research.summary", "source": "explicit_summary"}


def test_interaction_service_strips_wake_phrase_before_planning_route() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="hello lumen, create a migration plan for lumen")

    assert response["mode"] == "planning"
    assert response["kind"] == "planning.migration"
    assert response["pipeline_trace"]["intake_frame"]["raw_input"] == "create a migration plan for lumen"
    assert response["resolved_prompt"] == "create a migration plan for lumen"
    assert response["wake_interaction"] == {
        "wake_phrase": "hello lumen",
        "classification": "greeting_plus_request",
        "stripped_prompt": "create a migration plan for lumen",
    }


def test_interaction_service_strips_wake_phrase_before_tool_route() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="hey lumen, run anh", input_path=Path("data/examples/cf4_ga_cone_template.csv"))

    assert response["mode"] == "tool"
    assert response["kind"] == "tool.command_alias"
    assert response["resolved_prompt"] == "run anh"
    assert response["wake_interaction"] == {
        "wake_phrase": "hey lumen",
        "classification": "greeting_plus_request",
        "stripped_prompt": "run anh",
    }


def test_interaction_service_uses_fresh_social_greeting_when_no_recent_context() -> None:
    service = make_interaction_service()
    service._fake_interaction_history_service.recent_records = lambda **kwargs: []

    response = service.ask(prompt="hello lumen")

    assert response["reply"]
    assert "?" in response["reply"] or "glad" in response["reply"].lower()


def test_interaction_service_routes_planning_prompt() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="create a migration plan for lumen")

    assert response["mode"] == "planning"
    assert response["kind"] == "planning.migration"
    assert response["schema_type"] == "assistant_response"
    assert response["context"]["record_count"] == 3
    assert response["context"]["top_matches"][0]["record"]["tool_id"] == "anh"
    assert response["context"]["top_interaction_matches"][0]["record"]["prompt"] == "create a migration plan for lumen"
    assert response["context"]["active_thread"] is None
    assert response["thought_framing"]["response_kind_label"] == "direct_answer"
    assert response["thought_framing"]["conversation_activity"] == "turning the current idea into a more workable plan"
    assert response["research_questions"] == [
        "What assumption should we tighten first?",
        "What evidence would make this sharper?",
    ]
    assert response["conversation_turn"]["kind"] == "collaborate"
    assert "Here's my read so far" in response["conversation_turn"]["lead"]
    assert response["response_intro"] == response["conversation_turn"]["lead"]
    assert "I'll give you my read, keep the thread in view, and push it one step further." in response["response_opening"]
    assert response["evidence"][0].startswith("Routing selected planning because")
    assert response["best_evidence"].startswith("Routing selected planning because")
    assert response["grounding_strength"] == "high"
    assert "Closest archive run: anh/spectral_dip_scan" in response["local_context_summary"]
    assert response["grounded_interpretation"].startswith("Local evidence is mixed, so this planning response should first reconcile competing signals:")
    assert response["local_context_assessment"] == "mixed"
    assert response["steps"][0] == "Define the current state, target state, and non-negotiable migration constraints."
    assert response["steps"][1].startswith("Archive evidence emphasizes anh/spectral_dip_scan")
    assert all(
        not step.lower().startswith("treat the first milestone as")
        for step in response["steps"]
    )
    assert all(not step.startswith("Working hypothesis:") for step in response["steps"])
    assert all("Here's my read so far" not in step for step in response["steps"])
    assert all("I'll give you my read" not in step for step in response["steps"])
    assert response["pipeline_synthesis"]["route_evidence_distinction"] == "route_and_evidence_generally_aligned"
    assert response["pipeline_synthesis"]["response_body"][0].startswith("Support status: strongly supported")
    assert response["pipeline_synthesis"]["response_body"][1] in {
        "This answer is still carrying real tension.",
        "This answer is still under real tension.",
        "There is still real tension in this answer.",
    }
    assert "alternatives should remain explicit" in response["pipeline_synthesis"]["response_body"][2].lower()
    assert "hypothesis a is carrying" in response["pipeline_synthesis"]["response_body"][3].lower()
    assert any(
        phrase in response["pipeline_synthesis"]["response_body"][4].lower()
        for phrase in {
            "competing explanations are still live",
            "competing explanations are still in play",
            "competing explanations are still active",
        }
    )
    assert response["pipeline_synthesis"]["response_body"][5].startswith("Anchor evidence:")
    assert response["pipeline_synthesis"]["response_body"][6] == "Define the current state, target state, and non-negotiable migration constraints."
    assert response["pipeline_execution"]["execution_type"] == "reasoned_response"
    assert response["pipeline_packaging"]["package_type"] == "structured"
    assert response["pipeline_observability"]["response_summary"]["package_type"] == "structured"
    assert response["pipeline_trace"]["intake_frame"]["raw_input"] == "create a migration plan for lumen"
    assert response["pipeline_trace"]["dialogue_management"]["interaction_mode"] == "analytical"
    assert response["pipeline_trace"]["thought_framing"]["response_kind_label"] == "direct_answer"
    assert response["pipeline_trace"]["intake_frame"]["interaction_profile"]["interaction_style"] == "collab"
    assert response["pipeline_trace"]["validation_context"]["failure_modes"]["weak_evidence"] is False
    assert response["pipeline_trace"]["stage_contracts"]["synthesis"]["stage_name"] == "synthesis"
    assert response["pipeline_trace"]["stage_contracts"]["persistence_observability"]["produced_outputs"]
    assert response["next_action"] == "Define the smallest safe migration slice and its validation checkpoint."
    assert response["dialogue_management"]["interaction_mode"] == "analytical"
    assert response["dialogue_management"]["response_strategy"] == "answer"
    assert "route" in response
    assert response["route"]["confidence"] > 0
    assert "Identify the smallest safe migration slice that delivers value without breaking active flows." in response["steps"]
    assert len(service._fake_interaction_history_service.records) == 1
    assert service._fake_session_context_service.active_thread["kind"] == "planning.migration"
    assert service._fake_session_context_service.active_thread["interaction_profile"]["reasoning_depth"] == "normal"
    assert service._fake_session_context_service.active_thread["thread_summary"] == response["summary"]


def test_interaction_service_routes_research_prompt() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="summarize the current archive structure")

    assert response["mode"] == "research"
    assert response["kind"] == "research.summary"
    assert response["schema_type"] == "assistant_response"
    assert response["context"]["status_counts"]["ok"] == 2
    assert response["context"]["top_matches"][0]["score"] == 5
    assert response["context"]["active_thread"] is None
    assert response["evidence"][0].startswith("Routing selected research because")
    assert response["best_evidence"].startswith("Routing selected research because")
    assert response["grounding_strength"] == "high"
    assert "Closest archive run: anh/spectral_dip_scan" in response["local_context_summary"]
    assert response["grounded_interpretation"].startswith("Local evidence is mixed, so this research response should first reconcile competing signals:")
    assert response["local_context_assessment"] == "mixed"
    assert response["findings"][0] == "Keep the first conclusion centered on the leading hypothesis (A) while keeping the competing explanation explicit."
    assert all("Here's my read so far" not in item for item in response["findings"])
    assert all("Let me pull the threads together" not in item for item in response["findings"])
    assert all(
        not item.startswith(prefix)
        for item in response["findings"]
        for prefix in (
            "The strongest point so far is ",
            "The strongest point right now is ",
            "The strongest signal so far is ",
            "The weak point is ",
            "The weak point right now is ",
            "The pressure point is ",
            "The open question is ",
            "The main open question is ",
            "The unresolved question is ",
            "Working hypothesis:",
        )
    )
    assert "State the topic in one concise sentence." not in response["findings"]
    assert all(
        not item.startswith("Call out this local tension before drawing conclusions:")
        for item in response["findings"]
    )
    assert any(
        "leading hypothesis" in item.lower() or "competing explanation explicit" in item.lower()
        for item in response["findings"]
    )
    assert response["pipeline_synthesis"]["response_body"][0].startswith("Support status: strongly supported")
    assert response["pipeline_synthesis"]["response_body"][1] in {
        "This answer is still carrying real tension.",
        "This answer is still under real tension.",
        "There is still real tension in this answer.",
    }
    assert "alternatives should remain explicit" in response["pipeline_synthesis"]["response_body"][2].lower()
    assert "hypothesis a is carrying" in response["pipeline_synthesis"]["response_body"][3].lower()
    assert any(
        phrase in response["pipeline_synthesis"]["response_body"][4].lower()
        for phrase in {
            "competing explanations are still live",
            "competing explanations are still in play",
            "competing explanations are still active",
        }
    )
    assert "user_facing_answer" not in response
    assert response["summary"] != "I can't answer that cleanly from local knowledge alone yet."
    assert response["pipeline_synthesis"]["response_body"][5].startswith("Anchor evidence:")
    assert response["pipeline_synthesis"]["response_body"][6] == "State the topic in one concise sentence."
    assert response["pipeline_execution"]["execution_type"] == "reasoned_response"
    assert response["pipeline_packaging"]["package_type"] == "structured"
    assert response["pipeline_observability"]["reasoning_summary"]["frame_type"] == "retrieve-and-summarize"
    assert response["pipeline_trace"]["reasoning_frame"]["frame_type"] == "retrieve-and-summarize"
    assert response["pipeline_trace"]["stage_contracts"]["reasoning_frame_assembly"]["confidence_signal"]
    assert response["recommendation"].startswith("Summarize the strongest local evidence first")
    assert "closest archive run" not in response["recommendation"].lower()
    assert response["recommendation"] == "Summarize the strongest local evidence first, then end with one concrete next step."
    assert "route" in response
    assert response["route"]["confidence"] > 0
    assert len(service._fake_interaction_history_service.records) == 1
    assert service._fake_session_context_service.active_thread["kind"] == "research.summary"
    assert service._fake_session_context_service.active_thread["thread_summary"] == response["summary"]


def test_interaction_service_preserves_client_surface_placeholder() -> None:
    service = make_interaction_service()

    response = service.ask(
        prompt="create a migration plan for lumen",
        client_surface="mobile",
    )

    assert response["client_surface"] == "mobile"
    assert service._fake_interaction_history_service.records[0]["response"]["client_surface"] == "mobile"


def test_interaction_service_surfaces_route_caution_for_fallback_prompt() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="hello there")

    assert response["mode"] == "research"
    assert response["route"]["strength"] == "low"
    assert "fell back to a general research response" in response["route"]["caution"]
    assert any(item.startswith("Route caution:") for item in response["evidence"])
    assert response["recommendation"] == "Validate the strongest assumption with another source."


def test_interaction_service_routes_comparison_prompt() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="compare local archive retrieval versus indexed retrieval")

    assert response["mode"] == "research"
    assert response["kind"] == "research.comparison"
    assert response["local_context_assessment"] == "mixed"
    assert response["findings"][0] == "Identify the two or more options being compared."
    assert all(response["response_intro"] != item for item in response["findings"])
    assert all("I'll give you my read" not in item for item in response["findings"])
    assert all(
        not item.startswith("Call out this local tension before drawing conclusions:")
        for item in response["findings"]
    )
    assert any(
        "leading hypothesis" in item.lower() or "competing explanation explicit" in item.lower()
        for item in response["findings"]
    )


def test_interaction_service_allows_tool_prompt_without_input_when_tool_can_handle_it() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="run anh")

    assert response["mode"] == "tool"
    assert response["kind"] == "tool.command_alias"


def test_interaction_service_routes_tool_prompt_when_input_exists() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="run anh", input_path=Path("data/examples/cf4_ga_cone_template.csv"))

    assert response["mode"] == "tool"
    assert response["kind"] == "tool.command_alias"
    assert response["schema_type"] == "assistant_response"
    assert "route" in response
    assert response["route"]["confidence"] > 0
    assert response["tool_route_origin"] == "exact_alias"
    assert response["tool_execution"]["input_path"].endswith("cf4_ga_cone_template.csv")
    assert response["pipeline_execution"]["execution_type"] == "tool_call"
    assert response["pipeline_packaging"]["package_type"] == "brief"
    assert response["pipeline_observability"]["execution_summary"]["execution_type"] == "tool_call"
    assert response["pipeline_trace"]["execution_package"]["execution_type"] == "tool_call"
    assert response["pipeline_trace"]["stage_contracts"]["execution"]["failure_state"] in {None, "execution_warning"}
    assert len(service._fake_interaction_history_service.records) == 1
    assert service._fake_session_context_service.active_thread["kind"] == "tool.command_alias"


def test_interaction_service_uses_recent_session_follow_up_context() -> None:
    service = make_interaction_service()

    first = service.ask(prompt="create a migration plan for lumen")
    second = service.ask(prompt="expand that further")

    assert first["mode"] == "planning"
    assert second["mode"] == "planning"
    assert second["kind"] == "planning.migration"
    assert "Follow-up prompt" in second["route"]["reason"]
    assert second["context"]["active_thread"]["objective"] == "Objective for: create a migration plan for lumen"
    assert second["evidence"][0].startswith("Routing selected planning because")
    assert "Current active prompt: create a migration plan for lumen." in second["evidence"]
    assert service._fake_session_context_service.active_thread["prompt"] == "expand that further"
    assert service._fake_session_context_service.active_thread["thread_summary"].endswith("latest: expand that further")


def test_interaction_service_resolves_compare_shorthand_from_active_thread() -> None:
    service = make_interaction_service()

    service.ask(prompt="create a migration plan for lumen")
    response = service.ask(prompt="now compare that")

    assert response["mode"] == "research"
    assert response["kind"] == "research.comparison"
    assert response["resolved_prompt"] == "compare the migration plan for lumen"
    assert response["resolution_strategy"] == "compare_shorthand"
    assert "comparison shorthand" in response["resolution_reason"]


def test_interaction_service_resolves_reference_follow_up_from_active_thread() -> None:
    service = make_interaction_service()

    service.ask(prompt="create a migration plan for lumen")
    response = service.ask(prompt="what about that")

    assert response["mode"] == "planning"
    assert response["kind"] == "planning.migration"
    assert response["resolved_prompt"] == "what about the migration plan for lumen"
    assert response["resolution_strategy"] == "reference_follow_up"


def test_interaction_service_returns_clarification_for_ambiguous_route() -> None:
    service = make_interaction_service()

    first = service.ask(prompt="create a migration plan for lumen")
    response = service.ask(prompt="review the migration summary")

    assert first["mode"] == "planning"
    assert response["mode"] == "clarification"
    assert response["kind"] == "clarification.request"
    assert response["route"]["ambiguity"]["ambiguous"] is True
    assert (
        response["clarification_question"]
        == "I’m noticing we’ve got a couple directions we could go here. Do you want to explore this together, or focus on something specific?"
    )
    assert response["options"] == ["Continue", "Summary"]
    assert response["clarification_context"]["clarification_count"] == 0
    assert response["clarification_context"]["clarification_trigger"] == "base_threshold"
    assert service._fake_session_context_service.active_thread["prompt"] == "create a migration plan for lumen"
    assert service._fake_session_context_service.active_thread["kind"] == "planning.migration"


def test_interaction_service_uses_mode_specific_clarification_wording() -> None:
    collab = make_interaction_service()
    collab._fake_session_context_service.interaction_profile = InteractionProfile(
        interaction_style="collab",
        reasoning_depth="normal",
        selection_source="user",
        allow_suggestions=True,
    )
    collab.ask(prompt="create a migration plan for lumen")
    collab_response = collab.ask(prompt="review the migration summary")

    default = make_interaction_service()
    default._fake_session_context_service.interaction_profile = InteractionProfile(
        interaction_style="default",
        reasoning_depth="normal",
        selection_source="user",
        allow_suggestions=True,
    )
    default.ask(prompt="create a migration plan for lumen")
    default_response = default.ask(prompt="review the migration summary")

    direct = make_interaction_service()
    direct._fake_session_context_service.interaction_profile = InteractionProfile(
        interaction_style="direct",
        reasoning_depth="normal",
        selection_source="user",
        allow_suggestions=True,
    )
    direct.ask(prompt="create a migration plan for lumen")
    direct_response = direct.ask(prompt="review the migration summary")

    assert (
        collab_response["clarification_question"]
        == "I’m noticing we’ve got a couple directions we could go here. Do you want to explore this together, or focus on something specific?"
    )
    assert (
        default_response["clarification_question"]
        == "This could go a couple ways. Do you want a quick explanation, a comparison, or to continue the current route?"
    )
    assert direct_response["clarification_question"] == "Ambiguity detected. Choose: summary | comparison | continue."
    assert collab_response["options"] == ["Continue", "Summary"]
    assert default_response["options"] == ["Continue", "Summary"]
    assert direct_response["options"] == ["Continue", "Summary"]


def test_interaction_service_prefers_direct_answer_for_known_history_topic() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="Causes of the French Revolution")

    assert response["mode"] == "research"
    assert response["kind"] == "research.summary"
    assert "french revolution" in str(response["summary"]).lower()
    assert response["mode"] != "clarification"


def test_interaction_service_surfaces_profile_advice_without_overriding_active_profile() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="give me a brief direct answer about the archive structure")

    assert response["interaction_profile"]["interaction_style"] == "collab"
    assert response["interaction_profile"]["reasoning_depth"] == "normal"
    assert response["profile_advice"]["interaction_style"] == "direct"
    assert response["pipeline_trace"]["nlu_frame"]["profile_mismatch"] is True


def test_interaction_service_keeps_collab_surface_when_nlu_suggests_direct() -> None:
    service = make_interaction_service()
    service._fake_session_context_service.interaction_profile = InteractionProfile(
        interaction_style="collab",
        reasoning_depth="normal",
        selection_source="user",
        allow_suggestions=True,
    )

    response = service.ask(prompt="give me a brief direct answer about the archive structure")

    assert response["interaction_profile"]["interaction_style"] == "collab"
    assert response["profile_advice"]["interaction_style"] == "direct"
    assert response["response_intro"] != "Best read:"
    assert response["response_opening"] != "Best read first, then the next move."


def test_interaction_service_keeps_direct_mode_across_turns_despite_non_direct_follow_up() -> None:
    service = make_interaction_service()
    service._fake_session_context_service.interaction_profile = InteractionProfile(
        interaction_style="direct",
        reasoning_depth="normal",
        selection_source="user",
        allow_suggestions=True,
    )

    first = service.ask(prompt="summarize the archive structure")
    second = service.ask(prompt="tell me more")

    assert first["interaction_profile"]["interaction_style"] == "direct"
    assert second["interaction_profile"]["interaction_style"] == "direct"
    assert service._fake_session_context_service.active_thread["interaction_profile"]["interaction_style"] == "direct"


def test_interaction_service_uses_deep_profile_to_expand_validation_and_packaging() -> None:
    service = make_interaction_service()
    service._fake_session_context_service.interaction_profile = InteractionProfile(
        interaction_style="conversational",
        reasoning_depth="deep",
        selection_source="user",
        allow_suggestions=True,
    )

    response = service.ask(prompt="create a migration plan for lumen")

    assert response["interaction_profile"]["reasoning_depth"] == "deep"
    assert response["pipeline_packaging"]["package_type"] == "deep"
    assert any(item.startswith("Validation plan:") for item in response["steps"])
    assert response["steps"][-1].startswith("Deep thinking pass:")
    assert len(response["pipeline_synthesis"]["validation_advice"]) >= 2


def test_interaction_service_uses_direct_profile_to_tighten_output() -> None:
    service = make_interaction_service()
    service._fake_session_context_service.interaction_profile = InteractionProfile(
        interaction_style="direct",
        reasoning_depth="normal",
        selection_source="user",
        allow_suggestions=True,
    )

    response = service.ask(prompt="summarize the current archive structure")

    assert response["interaction_profile"]["interaction_style"] == "direct"
    assert response["pipeline_packaging"]["package_type"] == "structured"
    assert len(response["pipeline_synthesis"]["response_body"]) <= 4
    assert response["pipeline_synthesis"]["response_body"][0].startswith("Answer:")
    assert response["pipeline_synthesis"]["response_body"][1].startswith("Why:")
    assert response["pipeline_synthesis"]["response_body"][2].startswith("Action:")
    assert len(response["pipeline_synthesis"]["validation_advice"]) <= 1
    assert len(response["findings"]) <= 5
    assert response["recommendation"] == "Summarize the strongest local evidence first, then end with one concrete next step."


def test_interaction_service_makes_repeated_clarification_more_explicit() -> None:
    service = make_interaction_service()

    service.ask(prompt="create a migration plan for lumen")
    first = service.ask(prompt="review the migration summary")
    second = service.ask(prompt="review the migration summary")

    assert first["mode"] == "clarification"
    assert second["mode"] == "clarification"
    assert second["summary"] == "Repeated clarification requested for: review the migration summary"
    assert (
        second["clarification_question"]
        == "I’m noticing we’ve got a couple directions we could go here. Do you want to explore this together, or focus on something specific?"
    )
    assert second["clarification_context"]["clarification_count"] == 1
    assert second["clarification_context"]["recent_clarification_mix"] == "clarification_heavy_mixed"
    assert second["clarification_context"]["clarification_drift"] == "increasing"
    assert second["clarification_context"]["clarification_trigger"] == "base_threshold"


def test_interaction_service_clarifies_earlier_after_clarification_heavy_session() -> None:
    service = make_interaction_service()

    service.ask(prompt="create a migration plan for lumen")
    service.ask(prompt="review the migration summary")

    service.domain_router.route = lambda *args, **kwargs: DomainRoute(
        mode="planning",
        kind="planning.architecture",
        normalized_prompt="sketch the migration summary",
        confidence=0.82,
        reason="Planning cues narrowly outranked research cues",
        source="heuristic_planning",
        evidence=[],
        decision_summary={
            "selected": {},
            "alternatives": [
                {
                    "candidate": {
                        "mode": "research",
                        "kind": "research.summary",
                        "source": "heuristic_research",
                        "confidence": 0.79,
                    }
                }
            ],
            "ambiguous": True,
            "ambiguity_reason": "Top route candidates had very similar confidence and closely ranked source priority.",
        },
    )

    response = service.ask(prompt="sketch the migration summary")

    assert response["mode"] == "clarification"
    assert response["summary"] == "Repeated clarification requested for: sketch the migration summary"
    assert (
        response["clarification_question"]
        == "I’m noticing we’ve got a couple directions we could go here. Do you want to explore this together, or focus on something specific?"
    )
    assert response["clarification_context"]["clarification_count"] == 1
    assert response["clarification_context"]["clarification_drift"] == "increasing"
    assert response["clarification_context"]["clarification_trigger"] == "adaptive_threshold"


def test_interaction_service_resumes_design_after_keep_current_route_confirmation() -> None:
    service = make_interaction_service()
    service._fake_interaction_history_service.recent_records = lambda **kwargs: [
        {
            "prompt": "design me an engine",
            "mode": "clarification",
            "kind": "clarification.request",
            "response": {
                "mode": "clarification",
                "kind": "clarification.request",
                "route": {"confidence": 0.84, "reason": "Planning cues narrowly outranked research cues"},
                "options": [
                    "planning (planning.architecture)",
                    "research (research.summary)",
                    "keep current route",
                ],
                "clarification_context": {
                    "suggested_route": {
                        "mode": "planning",
                        "kind": "planning.architecture",
                        "resolved_prompt": "design me an engine",
                    }
                },
            },
        }
    ]

    response = service.ask(prompt="keep current route")

    assert response["mode"] == "planning"
    assert response["kind"] == "planning.architecture"
    assert response["resolution_strategy"] == "clarification_route_confirmation"
    assert "first-pass design concept" in response["summary"].lower()
    assert any(step.startswith("Assumptions:") for step in response["steps"])
    assert any(step.startswith("High-level system:") for step in response["steps"])
    assert any(step.startswith("Key components:") for step in response["steps"])
    assert any(step.startswith("Interaction:") for step in response["steps"])
    assert any(step.startswith("Next refinement:") for step in response["steps"])
    assert "not enough grounded detail" not in response["summary"].lower()


def test_interaction_service_declines_clarified_route_and_requests_new_direction() -> None:
    service = make_interaction_service()
    service._fake_session_context_service.interaction_profile = InteractionProfile(
        interaction_style="default",
        reasoning_depth="normal",
        selection_source="user",
        allow_suggestions=True,
    )
    service._fake_interaction_history_service.recent_records = lambda **kwargs: [
        {
            "prompt": "design me an engine",
            "mode": "clarification",
            "kind": "clarification.request",
            "response": {
                "mode": "clarification",
                "kind": "clarification.request",
                "clarification_context": {
                    "suggested_route": {
                        "mode": "planning",
                        "kind": "planning.architecture",
                        "resolved_prompt": "design me an engine",
                    }
                },
            },
        }
    ]

    response = service.ask(prompt="no")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.clarification_decline"
    assert response["summary"] == "Okay. Tell me which direction you want to take next."


def test_interaction_service_resumes_design_after_other_confirmation_variants() -> None:
    confirmations = ["let's do that", "go with that", "proceed"]
    for confirmation in confirmations:
        service = make_interaction_service()
        service._fake_interaction_history_service.recent_records = lambda **kwargs: [
            {
                "prompt": "design me an engine",
                "mode": "clarification",
                "kind": "clarification.request",
                "response": {
                    "mode": "clarification",
                    "kind": "clarification.request",
                    "route": {"confidence": 0.84, "reason": "Planning cues narrowly outranked research cues"},
                    "options": [
                        "planning (planning.architecture)",
                        "research (research.summary)",
                        "keep current route",
                    ],
                    "clarification_context": {
                        "suggested_route": {
                            "mode": "planning",
                            "kind": "planning.architecture",
                            "resolved_prompt": "design me an engine",
                        }
                    },
                },
            }
        ]

        response = service.ask(prompt=confirmation)

        assert response["mode"] == "planning"
        assert response["kind"] == "planning.architecture"
        assert response["resolution_strategy"] == "clarification_route_confirmation"
        assert any(step.startswith("High-level system:") for step in response["steps"])


def test_interaction_service_pivots_cleanly_when_user_changes_direction_after_clarification() -> None:
    service = make_interaction_service()
    service._fake_interaction_history_service.recent_records = lambda **kwargs: [
        {
            "prompt": "design me an engine",
            "mode": "clarification",
            "kind": "clarification.request",
            "response": {
                "mode": "clarification",
                "kind": "clarification.request",
                "clarification_context": {
                    "suggested_route": {
                        "mode": "planning",
                        "kind": "planning.architecture",
                        "resolved_prompt": "design me an engine",
                    }
                },
            },
        }
    ]

    response = service.ask(prompt="let's explore black holes instead")

    assert response["mode"] == "research"
    assert "black hole" in str(response["summary"]).lower()
    assert response.get("resolution_strategy") != "clarification_route_confirmation"


def test_interaction_service_resumes_design_when_follow_up_adds_concrete_engine_direction() -> None:
    service = make_interaction_service()
    service._fake_interaction_history_service.recent_records = lambda **kwargs: [
        {
            "prompt": "design me an engine",
            "mode": "clarification",
            "kind": "clarification.request",
            "response": {
                "mode": "clarification",
                "kind": "clarification.request",
                "route": {"confidence": 0.84, "reason": "Planning cues narrowly outranked research cues"},
                "options": [
                    "planning (planning.architecture)",
                    "research (research.summary)",
                    "keep current route",
                ],
                "clarification_context": {
                    "suggested_route": {
                        "mode": "planning",
                        "kind": "planning.architecture",
                        "resolved_prompt": "design me an engine",
                    }
                },
            },
        }
    ]

    response = service.ask(prompt="compact reusable engine")

    assert response["mode"] == "planning"
    assert response["resolution_strategy"] == "clarification_route_confirmation"
    assert "additional design direction" in response["resolved_prompt"].lower()


def test_interaction_service_does_not_force_planning_when_user_chooses_explanation_after_design_clarification() -> None:
    service = make_interaction_service()
    service._fake_interaction_history_service.recent_records = lambda **kwargs: [
        {
            "prompt": "design me an engine",
            "mode": "clarification",
            "kind": "clarification.request",
            "response": {
                "mode": "clarification",
                "kind": "clarification.request",
                "route": {"confidence": 0.84, "reason": "Planning cues narrowly outranked research cues"},
                "options": [
                    "planning (planning.architecture)",
                    "research (research.summary)",
                    "keep current route",
                ],
                "clarification_context": {
                    "suggested_route": {
                        "mode": "planning",
                        "kind": "planning.architecture",
                        "resolved_prompt": "design me an engine",
                    }
                },
            },
        }
    ]

    response = service.ask(prompt="explain it first")

    assert response["mode"] != "planning" or response.get("resolution_strategy") != "clarification_route_confirmation"


def test_interaction_service_answers_relational_prompt_with_grounded_known_concepts() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="entropy in relation to black holes")

    assert response["mode"] == "research"
    summary = str(response["summary"]).lower()
    assert "entropy" in summary
    assert "black hole" in summary
    assert "first pass" not in summary
    assert "next step" not in summary


def test_interaction_service_clarifies_earlier_when_recent_intent_detection_is_uncertain() -> None:
    service = make_interaction_service()

    for prompt in ["hello there", "hmm", "can you help"]:
        service.ask(prompt=prompt)

    service.domain_router.route = lambda *args, **kwargs: DomainRoute(
        mode="planning",
        kind="planning.architecture",
        normalized_prompt="sketch the migration summary",
        confidence=0.87,
        reason="Planning cues narrowly outranked research cues",
        source="heuristic_planning",
        evidence=[],
        decision_summary={
            "selected": {},
            "alternatives": [
                {
                    "candidate": {
                        "mode": "research",
                        "kind": "research.summary",
                        "source": "heuristic_research",
                        "confidence": 0.85,
                    }
                }
            ],
            "ambiguous": True,
            "ambiguity_reason": "Top route candidates had very similar confidence and closely ranked source priority.",
        },
    )

    response = service.ask(prompt="sketch the migration summary")

    assert response["mode"] == "clarification"
    assert (
        response["clarification_question"]
        == "I’m noticing we’ve got a couple directions we could go here. Do you want to explore this together, or focus on something specific?"
    )
    assert response["clarification_context"]["clarification_trigger"] == "nlu_uncertainty"
    assert response["clarification_context"]["dominant_intent_counts"]["unknown"] == 3


def test_interaction_service_clarifies_earlier_when_retrieval_is_too_semantic_led() -> None:
    service = make_interaction_service()

    service._fake_interaction_history_service.summarize_interactions = lambda **kwargs: {
        "interaction_count": 4,
        "clarification_count": 0,
        "clarification_ratio": 0.0,
        "clarification_trend": ["clear"],
        "recent_clarification_mix": "stable:clear",
        "latest_clarification": "clear",
        "clarification_drift": "steady",
        "dominant_intent_counts": {"planning": 2},
        "retrieval_lead_counts": {"semantic": 3, "keyword": 0, "blended": 1},
        "retrieval_observation_count": 4,
    }
    service.domain_router.route = lambda *args, **kwargs: DomainRoute(
        mode="planning",
        kind="planning.architecture",
        normalized_prompt="sketch the migration summary",
        confidence=0.85,
        reason="Planning cues narrowly outranked research cues",
        source="heuristic_planning",
        evidence=[],
        decision_summary={
            "selected": {},
            "alternatives": [
                {
                    "candidate": {
                        "mode": "research",
                        "kind": "research.summary",
                        "source": "heuristic_research",
                        "confidence": 0.83,
                    }
                }
            ],
            "ambiguous": True,
            "ambiguity_reason": "Top route candidates had very similar confidence and closely ranked source priority.",
        },
    )

    response = service.ask(prompt="sketch the migration summary")

    assert response["mode"] == "clarification"
    assert (
        response["clarification_question"]
        == "I’m noticing we’ve got a couple directions we could go here. Do you want to explore this together, or focus on something specific?"
    )
    assert response["clarification_context"]["clarification_trigger"] == "retrieval_semantic_bias"
    assert response["clarification_context"]["retrieval_lead_counts"]["semantic"] == 3


def test_interaction_service_softens_ambiguous_route_confidence_when_retrieval_is_semantic_led() -> None:
    service = make_interaction_service()

    service._fake_interaction_history_service.summarize_interactions = lambda **kwargs: {
        "interaction_count": 4,
        "clarification_count": 0,
        "clarification_ratio": 0.0,
        "clarification_trend": ["clear"],
        "recent_clarification_mix": "stable:clear",
        "latest_clarification": "clear",
        "clarification_drift": "steady",
        "dominant_intent_counts": {"planning": 2},
        "retrieval_lead_counts": {"semantic": 3, "keyword": 0, "blended": 1},
        "retrieval_observation_count": 4,
    }
    service.domain_router.route = lambda *args, **kwargs: DomainRoute(
        mode="planning",
        kind="planning.architecture",
        normalized_prompt="draft the migration architecture",
        confidence=0.9,
        reason="Planning cues narrowly outranked research cues",
        source="heuristic_planning",
        evidence=[],
        decision_summary={
            "selected": {},
            "alternatives": [
                {
                    "candidate": {
                        "mode": "research",
                        "kind": "research.summary",
                        "source": "heuristic_research",
                        "confidence": 0.88,
                    }
                }
            ],
            "ambiguous": True,
            "ambiguity_reason": "Top route candidates had very similar confidence and closely ranked source priority.",
        },
    )

    response = service.ask(prompt="draft the migration architecture")

    assert response["mode"] == "planning"
    assert response["route"]["confidence"] == pytest.approx(0.87)
    assert "Retrieval bias caution lowered route confidence slightly" in response["route"]["reason"]
    assert any(
        item.get("label") == "retrieval_bias_caution"
        for item in response["route"]["evidence"]
    )


def test_interaction_service_reinforces_ambiguous_route_when_session_intent_is_consistently_planning() -> None:
    service = make_interaction_service()

    service._fake_interaction_history_service.summarize_interactions = lambda **kwargs: {
        "interaction_count": 5,
        "clarification_count": 0,
        "clarification_ratio": 0.0,
        "clarification_trend": ["clear"],
        "recent_clarification_mix": "stable:clear",
        "latest_clarification": "clear",
        "clarification_drift": "steady",
        "dominant_intent_counts": {"planning": 4, "research": 1},
        "retrieval_lead_counts": {"keyword": 2, "semantic": 1, "blended": 1},
        "retrieval_observation_count": 4,
    }
    service.domain_router.route = lambda *args, **kwargs: DomainRoute(
        mode="planning",
        kind="planning.architecture",
        normalized_prompt="sketch the migration summary",
        confidence=0.78,
        reason="Planning cues narrowly outranked research cues",
        source="heuristic_planning",
        evidence=[],
        decision_summary={
            "selected": {},
            "alternatives": [
                {
                    "candidate": {
                        "mode": "research",
                        "kind": "research.summary",
                        "source": "heuristic_research",
                        "confidence": 0.76,
                    }
                }
            ],
            "ambiguous": True,
            "ambiguity_reason": "Top route candidates had very similar confidence and closely ranked source priority.",
        },
    )

    response = service.ask(prompt="sketch the migration summary")

    assert response["mode"] == "planning"
    assert response["route"]["confidence"] == pytest.approx(0.81)
    assert "Session intent continuity reinforced this route" in response["route"]["reason"]
    assert any(
        item.get("label") == "session_intent_bias"
        for item in response["route"]["evidence"]
    )


def test_interaction_service_softens_ambiguous_route_when_session_intent_conflicts() -> None:
    service = make_interaction_service()

    service._fake_interaction_history_service.summarize_interactions = lambda **kwargs: {
        "interaction_count": 5,
        "clarification_count": 0,
        "clarification_ratio": 0.0,
        "clarification_trend": ["clear"],
        "recent_clarification_mix": "stable:clear",
        "latest_clarification": "clear",
        "clarification_drift": "steady",
        "dominant_intent_counts": {"research": 4, "planning": 1},
        "retrieval_lead_counts": {"keyword": 2, "semantic": 1, "blended": 1},
        "retrieval_observation_count": 4,
    }
    service.domain_router.route = lambda *args, **kwargs: DomainRoute(
        mode="planning",
        kind="planning.architecture",
        normalized_prompt="sketch the migration summary",
        confidence=0.82,
        reason="Planning cues narrowly outranked research cues",
        source="heuristic_planning",
        evidence=[],
        decision_summary={
            "selected": {},
            "alternatives": [
                {
                    "candidate": {
                        "mode": "research",
                        "kind": "research.summary",
                        "source": "heuristic_research",
                        "confidence": 0.8,
                    }
                }
            ],
            "ambiguous": True,
            "ambiguity_reason": "Top route candidates had very similar confidence and closely ranked source priority.",
        },
    )

    response = service.ask(prompt="sketch the migration summary")

    assert response["mode"] == "planning"
    assert response["route"]["confidence"] == pytest.approx(0.79)
    assert "Session intent caution lowered route confidence slightly" in response["route"]["reason"]
    assert any(
        item.get("label") == "session_intent_caution"
        for item in response["route"]["evidence"]
    )


def test_interaction_service_uses_canonical_prompt_view_in_research_guidance() -> None:
    service = make_interaction_service()
    service._fake_interaction_history_service.retrieve_context = lambda query, **kwargs: {
        "interaction_record_count": 1,
        "top_interaction_matches": [
            {
                "score": 4,
                "record": {
                    "prompt": "what about that",
                    "resolved_prompt": "what about the migration plan for lumen",
                    "prompt_view": {
                        "canonical_prompt": "what about the migration plan for lumen",
                        "original_prompt": "what about that",
                        "resolved_prompt": "what about the migration plan for lumen",
                        "rewritten": True,
                    },
                    "summary": "Planning response for: what about the migration plan for lumen",
                },
            }
        ],
    }

    response = service.ask(prompt="summarize the current archive structure")

    assert (
        "Closest prior session prompt: what about the migration plan for lumen (from 'what about that')."
        in response["evidence"]
    )


def test_interaction_service_uses_canonical_prompt_view_in_planning_guidance() -> None:
    service = make_interaction_service()
    service._fake_interaction_history_service.retrieve_context = lambda query, **kwargs: {
        "interaction_record_count": 1,
        "top_interaction_matches": [
            {
                "score": 4,
                "record": {
                    "prompt": "what about that",
                    "resolved_prompt": "what about the migration plan for lumen",
                    "prompt_view": {
                        "canonical_prompt": "what about the migration plan for lumen",
                        "original_prompt": "what about that",
                        "resolved_prompt": "what about the migration plan for lumen",
                        "rewritten": True,
                    },
                    "summary": "Planning response for: what about the migration plan for lumen",
                },
            }
        ],
    }

    response = service.ask(prompt="create a migration plan for lumen")

    assert (
        "Closest prior session prompt: what about the migration plan for lumen (from 'what about that')."
        in response["evidence"]
    )


def test_interaction_service_mentions_resolved_active_prompt_in_research_guidance() -> None:
    service = make_interaction_service()
    service._fake_session_context_service.active_thread = {
        "session_id": "default",
        "mode": "planning",
        "kind": "planning.migration",
        "prompt": "what about the migration plan for lumen",
        "original_prompt": "what about that",
        "objective": "Plan work for: create a migration plan for lumen",
        "thread_summary": "Planning response for: create a migration plan for lumen | latest: what about the migration plan for lumen",
        "summary": "Planning response for: what about the migration plan for lumen",
        "tool_context": {},
    }

    response = service.ask(prompt="summarize the current archive structure")

    assert (
        "Current active prompt resolves 'what about that' to 'what about the migration plan for lumen'."
        in response["evidence"]
    )


def test_interaction_service_falls_back_to_active_thread_for_interaction_context() -> None:
    service = make_interaction_service()
    calls: list[str] = []

    def retrieve_context(query, **kwargs):
        calls.append(query)
        if query == "summarize the current archive structure":
            return {
                "interaction_record_count": 0,
                "top_interaction_matches": [],
            }
        if query == "what about the migration plan for lumen":
            return {
                "interaction_record_count": 1,
                "top_interaction_matches": [
                    {
                        "score": 4,
                        "record": {
                            "prompt": "what about that",
                            "resolved_prompt": "what about the migration plan for lumen",
                            "prompt_view": {
                                "canonical_prompt": "what about the migration plan for lumen",
                                "original_prompt": "what about that",
                                "resolved_prompt": "what about the migration plan for lumen",
                                "rewritten": True,
                            },
                            "summary": "Planning response for: what about the migration plan for lumen",
                        },
                    }
                ],
            }
        return {
            "interaction_record_count": 0,
            "top_interaction_matches": [],
        }

    service._fake_interaction_history_service.retrieve_context = retrieve_context
    service._fake_session_context_service.active_thread = {
        "session_id": "default",
        "mode": "planning",
        "kind": "planning.migration",
        "prompt": "what about the migration plan for lumen",
        "original_prompt": "what about that",
        "objective": "Plan work for: create a migration plan for lumen",
        "thread_summary": "Planning response for: create a migration plan for lumen | latest: what about the migration plan for lumen",
        "summary": "Planning response for: what about the migration plan for lumen",
        "tool_context": {},
    }

    response = service.ask(prompt="summarize the current archive structure")

    assert calls == [
        "summarize the current archive structure",
        "what about the migration plan for lumen",
    ]
    assert response["context"]["interaction_query"] == "what about the migration plan for lumen"
    assert response["context"]["interaction_query_source"] == "active_thread"
    assert (
        "Closest prior session prompt: what about the migration plan for lumen (from 'what about that')."
        in response["evidence"]
    )


def test_interaction_service_falls_back_to_active_topic_for_interaction_context() -> None:
    service = make_interaction_service()
    calls: list[str] = []

    def retrieve_context(query, **kwargs):
        calls.append(query)
        if query in {
            "summarize the current archive structure",
            "review the latest state",
        }:
            return {
                "interaction_record_count": 0,
                "top_interaction_matches": [],
            }
        if query == "migration plan for lumen":
            return {
                "interaction_record_count": 1,
                "top_interaction_matches": [
                    {
                        "score": 4,
                        "record": {
                            "prompt": "create a migration plan for lumen",
                            "resolved_prompt": None,
                            "prompt_view": {
                                "canonical_prompt": "create a migration plan for lumen",
                                "original_prompt": "create a migration plan for lumen",
                                "resolved_prompt": None,
                                "rewritten": False,
                            },
                            "summary": "Planning response for: create a migration plan for lumen",
                            "dominant_intent": "planning",
                            "extracted_entities": [{"label": "domain", "value": "migration", "confidence": 0.8}],
                        },
                    }
                ],
            }
        return {
            "interaction_record_count": 0,
            "top_interaction_matches": [],
        }

    service._fake_interaction_history_service.retrieve_context = retrieve_context
    service._fake_session_context_service.active_thread = {
        "session_id": "default",
        "mode": "planning",
        "kind": "planning.migration",
        "prompt": "review the latest state",
        "normalized_topic": "migration plan for lumen",
        "original_prompt": None,
        "objective": "Plan work for: create a migration plan for lumen",
        "thread_summary": "Planning response for: create a migration plan for lumen",
        "summary": "Planning response for: create a migration plan for lumen",
        "tool_context": {},
    }

    response = service.ask(prompt="summarize the current archive structure")

    assert calls == [
        "summarize the current archive structure",
        "review the latest state",
        "migration plan for lumen",
    ]
    assert response["context"]["interaction_query"] == "migration plan for lumen"
    assert response["context"]["interaction_query_source"] == "active_topic"
    assert (
        "Closest prior session prompt: create a migration plan for lumen."
        in response["evidence"]
    )


def test_interaction_service_dampens_semantic_only_matches_in_semantic_heavy_session() -> None:
    service = make_interaction_service()

    service._fake_interaction_history_service.summarize_interactions = lambda **kwargs: {
        "interaction_count": 4,
        "clarification_count": 0,
        "clarification_ratio": 0.0,
        "clarification_trend": ["clear"],
        "recent_clarification_mix": "stable:clear",
        "latest_clarification": "clear",
        "clarification_drift": "steady",
        "dominant_intent_counts": {"planning": 2},
        "retrieval_lead_counts": {"semantic": 3, "keyword": 0, "blended": 1},
        "retrieval_observation_count": 4,
    }
    service.archive_service.retrieve_context = lambda query, **kwargs: {
        "query": query,
        "record_count": 2,
        "top_matches": [
            {
                "score": 8,
                "score_breakdown": {"keyword_score": 0, "semantic_score": 8},
                "matched_fields": ["semantic"],
                "record": {
                    "tool_id": "anh",
                    "capability": "spectral_dip_scan",
                    "summary": "routing logs",
                },
            },
            {
                "score": 7,
                "score_breakdown": {"keyword_score": 5, "semantic_score": 2},
                "matched_fields": ["summary", "semantic"],
                "record": {
                    "tool_id": "anh",
                    "capability": "spectral_dip_scan",
                    "summary": "Great Attractor routing analysis candidate",
                },
            },
        ],
    }

    response = service.ask(prompt="summarize the current archive structure")

    assert response["context"]["top_matches"][0]["record"]["summary"] == "Great Attractor routing analysis candidate"


def test_interaction_service_resolves_thread_follow_up_from_active_thread() -> None:
    service = make_interaction_service()

    service.ask(prompt="create a migration plan for lumen")
    response = service.ask(prompt="expand that further")

    assert response["mode"] == "planning"
    assert response["kind"] == "planning.migration"
    assert response["resolved_prompt"] == "expand the migration plan for lumen"
    assert response["resolution_strategy"] == "thread_follow_up"
    assert "active thread subject" in response["resolution_reason"]


def test_interaction_service_resolves_anh_tool_shorthand_from_active_thread() -> None:
    service = make_interaction_service()

    service.ask(prompt="run anh", input_path=Path("data/examples/cf4_ga_cone_template.csv"))
    response = service.ask(prompt="do that with anh", input_path=Path("data/examples/cf4_ga_cone_template.csv"))

    assert response["mode"] == "tool"
    assert response["kind"] == "tool.command_alias"
    assert response["resolved_prompt"] == "run anh"
    assert response["resolution_strategy"] == "anh_tool_shorthand"
    assert "active ANH tool command" in response["resolution_reason"]


def test_interaction_service_resolves_tool_repeat_shorthand_from_active_thread() -> None:
    service = make_interaction_service()

    service.ask(prompt="run anh", input_path=Path("data/examples/cf4_ga_cone_template.csv"))
    response = service.ask(prompt="run that again")

    assert response["mode"] == "tool"
    assert response["kind"] == "tool.command_alias"
    assert response["resolved_prompt"] == "run anh"
    assert response["resolution_strategy"] == "tool_repeat_shorthand"
    assert "repeat-style follow-up" in response["resolution_reason"]
    assert response["tool_execution"]["input_path"].endswith("cf4_ga_cone_template.csv")


def test_interaction_service_resolves_tool_hint_alias_for_report_bundle() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="confidence report for this session")

    assert response["mode"] == "tool"
    assert response["kind"] == "tool.command_alias"
    assert response["resolved_prompt"] == "report session confidence"
    assert response["resolution_strategy"] == "tool_hint_alias"
    assert "closest manifest-declared tool alias" in response["resolution_reason"]
    assert response["tool_route_origin"] == "nlu_hint_alias"


def test_interaction_service_resolves_tool_hint_alias_for_workspace_bundle() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="inspect the workspace structure")

    assert response["mode"] == "tool"
    assert response["kind"] == "tool.command_alias"
    assert response["resolved_prompt"] == "inspect workspace"
    assert response["resolution_strategy"] == "tool_hint_alias"


def test_interaction_service_resolves_tool_hint_alias_for_memory_bundle() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="review the session timeline")

    assert response["mode"] == "tool"
    assert response["kind"] == "tool.command_alias"
    assert response["resolved_prompt"] == "inspect session timeline"
    assert response["resolution_strategy"] == "tool_hint_alias"


def test_interaction_service_does_not_force_tool_hint_alias_for_non_request_text() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="session confidence is improving")

    assert response["mode"] != "tool"
    assert "resolved_prompt" not in response or response["resolved_prompt"] != "report session confidence"


def test_interaction_service_handles_belief_discussion_respectfully_without_confidence_posture() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="summarize how greek myth and norse myth differ as belief systems")

    assert response["mode"] == "research"
    assert response["discussion_domain"] == "belief_tradition"
    assert response["respectful_discussion"] is True
    assert response["confidence_posture"] is None
    assert "strictly falsifiable" in response["uncertainty_note"]
    assert any("strictly falsifiable" in item for item in response["findings"])


def test_interaction_service_soft_redirects_extreme_belief_prompt() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="prove which religion is true")

    assert response["mode"] == "research"
    assert response["discussion_domain"] == "belief_tradition"
    assert response["confidence_posture"] is None
    assert "comparatively, historically, or philosophically" in response["recommendation"]


def test_interaction_service_subtly_leans_historical_on_evidence_style_belief_prompt() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="what historical evidence exists for early christianity")

    assert response["discussion_domain"] == "belief_tradition"
    assert response["belief_frame_hint"] == "scientific"
    assert any("historical evidence" in item.lower() for item in response["findings"])


def test_interaction_service_subtly_leans_symbolic_on_myth_prompt() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="what does odin symbolize in norse myth")

    assert response["discussion_domain"] == "belief_tradition"
    assert response["belief_frame_hint"] == "symbolic"
    assert any("symbolism" in item.lower() or "mythic pattern" in item.lower() for item in response["findings"])


def test_interaction_service_uses_hosted_fallback_for_sparse_general_research() -> None:
    interaction_history_service = SparseInteractionHistoryService()
    session_context_service = FakeSessionContextService()
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=interaction_history_service,
        session_context_service=session_context_service,
        inference_service=FakeInferenceService(),
    )

    response = service.ask(prompt="Tell me about black holes")

    assert response["mode"] == "research"
    assert response["kind"] == "research.summary"
    assert "gravity" in response["summary"].lower()
    assert "black hole" in response["summary"].lower()
    assert response["user_facing_answer"] == response["summary"]
    assert response["findings"] == []
    assert response["local_knowledge_access"]["final_source"] == "curated_local_knowledge"
    assert "provider_inference" not in response
    assert response["route"]["source"] == "curated_local_knowledge"
    assert response["local_knowledge_access"]["local_knowledge_match"] is True
    assert response["response_behavior_posture"]["posture"] == "direct_answer"
    assert response["response_behavior_posture"]["visible_uncertainty"] is False


def test_interaction_service_routes_simple_factual_prompt_as_explanatory_research() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="tell me about black holes")

    assert response["mode"] == "research"
    assert response["kind"] == "research.summary"
    assert response["route"]["source"] == "curated_local_knowledge"


def test_interaction_service_routes_informal_factual_prompt_as_explanatory_research() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="hey can you explain black holes")

    assert response["mode"] == "research"
    assert response["kind"] == "research.summary"
    assert "black hole" in response["summary"].lower()


def test_interaction_service_keeps_trailing_humor_on_black_holes_prompt_substantive() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="tell me about black holes lol")

    assert response["mode"] == "research"
    assert response["kind"] == "research.summary"
    assert "black hole" in response["summary"].lower()


def test_interaction_service_routes_topic_only_prompt_as_explanatory_research() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="black holes")

    assert response["mode"] == "research"
    assert response["kind"] == "research.summary"


def test_interaction_service_uses_formal_explanatory_tone_for_plain_factual_prompt() -> None:
    interaction_history_service = SparseInteractionHistoryService()
    session_context_service = FakeSessionContextService()
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=interaction_history_service,
        session_context_service=session_context_service,
        inference_service=FakeInferenceService(),
    )

    response = service.ask(prompt="Tell me about black holes")

    assert response["response_tone_blend"]["tone_profile"] == "formal_explanation"
    assert "gravity" in response["summary"].lower()
    assert "black hole" in response["summary"].lower()
    assert response["user_facing_answer"] == response["summary"]
    assert response["findings"] == []


def test_interaction_service_uses_casual_explanatory_tone_when_social_signal_mixes_in() -> None:
    interaction_history_service = SparseInteractionHistoryService()
    session_context_service = FakeSessionContextService()
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=interaction_history_service,
        session_context_service=session_context_service,
        inference_service=FakeInferenceService(),
    )

    response = service.ask(prompt="hey what's a black hole lol")

    assert response["mode"] == "research"
    assert response["kind"] == "research.summary"
    assert response["response_tone_blend"]["tone_profile"] == "casual_explanation"
    assert "gravity" in response["summary"]
    assert "response_intro" not in response


def test_interaction_service_keeps_default_mode_surface_when_explanatory_tone_is_casual() -> None:
    interaction_history_service = SparseInteractionHistoryService()
    session_context_service = FakeSessionContextService()
    session_context_service.interaction_profile = InteractionProfile(
        interaction_style="default",
        reasoning_depth="normal",
        selection_source="user",
    )
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=interaction_history_service,
        session_context_service=session_context_service,
        inference_service=FakeInferenceService(),
    )

    response = service.ask(prompt="hey what's a black hole lol")

    assert response["response_tone_blend"]["tone_profile"] == "casual_explanation"
    assert not response["summary"].startswith("Sure.")


def test_interaction_service_routes_praise_as_social_affirmation() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="great job")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.affirmation"
    assert response["reply"] in {
        "Thank you. That means a lot.",
        "Thanks. I appreciate that.",
        "Thank you. I'm glad it helped.",
        "I appreciate that.",
        "Thanks. I'm glad that landed well.",
    }


def test_interaction_service_routes_you_are_awesome_as_social_affirmation() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="you are awesome")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.affirmation"
    assert "thank" in response["reply"].lower() or "appreciate" in response["reply"].lower()


def test_interaction_service_routes_relational_greeting_as_conversation() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="good to see you too")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.greeting"


def test_interaction_service_realizes_social_check_in_as_clean_utterance() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="how are you?")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.check_in"
    assert response["summary"] == response["reply"]
    assert response["conversational_reply_state"]["lane"] == "conversational"
    assert "next move:" not in response["summary"].lower()
    assert "validation plan" not in response["summary"].lower()


def test_interaction_service_collab_check_in_can_use_warmer_punctuation() -> None:
    service = make_interaction_service()
    service._fake_session_context_service.interaction_profile = InteractionProfile(
        interaction_style="collab",
        reasoning_depth="normal",
        selection_source="user",
    )

    response = service.ask(prompt="how are you?")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.check_in"
    assert "next move:" not in response["summary"].lower()
    assert "validation plan" not in response["summary"].lower()
    assert "?" in response["summary"] or "what" in response["summary"].lower() or "how about you" in response["summary"].lower()


def test_interaction_service_realizes_topic_suggestion_as_clean_single_reply() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="what do you want to talk about?")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.topic_suggestion"
    assert response["summary"] == response["reply"]
    assert response["conversational_reply_state"]["intent"] == "conversation.topic_suggestion"
    assert "pick one" in response["summary"].lower() or "your call" in response["summary"].lower()
    assert "here's the clearest read" not in response["summary"].lower()


def test_interaction_service_realizes_thought_mode_without_scaffold_leakage() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="what's on your mind?")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.thought_mode"
    assert response["summary"] == response["reply"]
    assert response["conversational_reply_state"]["intent"] == "conversation.thought_mode"
    assert "next move:" not in response["summary"].lower()
    assert "summarize the strongest" not in response["summary"].lower()


def test_interaction_service_routes_named_subject_to_explanatory_research() -> None:
    interaction_history_service = SparseInteractionHistoryService()
    session_context_service = FakeSessionContextService()
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=interaction_history_service,
        session_context_service=session_context_service,
        inference_service=FakeInferenceService(),
    )

    response = service.ask(prompt="George Washington")

    assert response["mode"] == "research"
    assert response["kind"] == "research.summary"
    assert "george washington" in response["summary"].lower()
    assert "first president" in response["summary"].lower()
    assert response["findings"] == []


def test_interaction_service_produces_local_fallback_answer_when_api_is_unavailable() -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )

    response = service.ask(prompt="Black Hole")

    assert response["mode"] == "research"
    assert response["kind"] == "research.summary"
    assert "black hole" in response["summary"].lower()
    assert "gravity" in response["summary"].lower()
    assert response["findings"] == []
    assert response["response_behavior_posture"]["posture"] == "cautious_answer"


def test_interaction_service_answers_formula_prompt_from_local_knowledge() -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )

    response = service.ask(prompt="What is the quadratic formula?")

    assert response["mode"] == "research"
    assert response["kind"] == "research.summary"
    assert "quadratic formula" in response["summary"].lower()
    assert "sqrt" in response["summary"].lower()
    assert response["findings"] == []
    assert response["reply"] == response["summary"]
    assert response["user_facing_answer"] == response["summary"]


def test_interaction_service_routes_topic_only_formula_prompt_as_explanatory_research() -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )

    response = service.ask(prompt="quadratic formula")

    assert response["mode"] == "research"
    assert response["kind"] == "research.summary"
    assert "quadratic formula" in response["summary"].lower()


def test_interaction_service_routes_topic_only_system_prompt_as_explanatory_research() -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )

    response = service.ask(prompt="feedback loop")

    assert response["mode"] == "research"
    assert response["kind"] == "research.summary"
    assert "feedback loop" in response["summary"].lower()


def test_interaction_service_answers_comparison_prompt_from_local_knowledge() -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )

    response = service.ask(prompt="Black hole vs neutron star")

    assert response["mode"] == "research"
    assert response["kind"] == "research.comparison"
    assert "black hole" in response["summary"].lower()
    assert "neutron star" in response["summary"].lower()
    assert "main difference" in response["summary"].lower()
    assert "surface" in response["summary"].lower() or "event horizon" in response["summary"].lower()
    assert response["findings"] == []


def test_interaction_service_answers_basic_unit_comparison_from_local_knowledge() -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )

    response = service.ask(prompt="ohms vs watts")

    assert response["mode"] == "research"
    assert response["kind"] == "research.comparison"
    assert "resistance" in response["summary"].lower()
    assert "power" in response["summary"].lower()
    assert "main difference" in response["summary"].lower()
    assert "not the same thing" in response["summary"].lower() or "different" in response["summary"].lower()
    assert response["reply"] == response["summary"]
    assert response["user_facing_answer"] == response["summary"]


def test_interaction_service_direct_mode_keeps_full_explanation_depth() -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )
    service._fake_session_context_service.interaction_profile = InteractionProfile(
        interaction_style="direct",
        reasoning_depth="normal",
        selection_source="user",
    )

    response = service.ask(prompt="Tell me about black holes")

    assert response["mode"] == "research"
    assert "black hole" in response["summary"].lower()
    assert len(str(response["summary"]).split(". ")) >= 2


def test_interaction_service_tell_me_more_inherits_local_knowledge_confidence() -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )
    service._fake_interaction_history_service.recent_records = lambda **kwargs: [
        {
            "mode": "research",
            "kind": "research.summary",
            "prompt": "Black Hole",
            "summary": "A black hole is a region of space where gravity is so strong that not even light can escape.",
            "response": {
                "user_facing_answer": "A black hole is a region of space where gravity is so strong that not even light can escape.",
            },
        }
    ]

    response = service.ask(prompt="tell me more")

    assert response["mode"] == "research"
    assert "black hole" in response["summary"].lower()
    assert "event horizon" in response["summary"].lower()
    assert "not enough grounded detail" not in response["summary"].lower()


def test_interaction_service_addressed_tell_me_more_inherits_local_knowledge_confidence() -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )
    service._fake_interaction_history_service.recent_records = lambda **kwargs: [
        {
            "mode": "research",
            "kind": "research.summary",
            "prompt": "Black Hole",
            "summary": "A black hole is a region of space where gravity is so strong that not even light can escape.",
            "response": {
                "user_facing_answer": "A black hole is a region of space where gravity is so strong that not even light can escape.",
                "resolved_prompt": "Black Hole",
            },
        }
    ]

    response = service.ask(prompt="Hey Lumen, tell me more")

    assert response["mode"] == "research"
    assert "black hole" in response["summary"].lower()
    assert "not enough grounded detail" not in response["summary"].lower()


def test_interaction_service_go_on_inherits_local_knowledge_confidence() -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )
    service._fake_interaction_history_service.recent_records = lambda **kwargs: [
        {
            "mode": "research",
            "kind": "research.general",
            "prompt": "Black Hole",
            "resolved_prompt": "Black Hole",
            "summary": "A black hole is a region of space where gravity is so strong that not even light can escape.",
            "response": {
                "user_facing_answer": "A black hole is a region of space where gravity is so strong that not even light can escape.",
                "resolved_prompt": "Black Hole",
            },
        }
    ]

    response = service.ask(prompt="go on")

    assert response["mode"] == "research"
    assert "black hole" in response["summary"].lower()
    assert "event horizon" in response["summary"].lower()
    assert "not enough grounded detail" not in response["summary"].lower()


def test_interaction_service_does_not_overwrite_substantive_research_answer_with_fallback_text() -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )
    response = {
        "mode": "research",
        "kind": "research.general",
        "summary": "We could use stored energy in principle, but the useful answer depends on conversion efficiency and mass constraints.",
        "findings": ["Stored energy can work in principle, but the bottleneck is usually conversion efficiency."],
        "response_tone_blend": {"tone_profile": "default"},
    }

    service._finalize_explanatory_answer(
        response=response,
        prompt="what is hyperbolic lemon engine",
        route=SimpleNamespace(mode="research", kind="research.summary"),
        interaction_profile=service._fake_session_context_service.interaction_profile,
        entities=(),
        provider_text=None,
        recent_interactions=[],
        route_support_signals=None,
    )

    assert response["summary"].startswith("We could use stored energy in principle")
    assert response.get("user_facing_answer") is None
    assert response["explanation_answer_source"] == "fallback"


def test_interaction_service_answers_reversed_comparison_prompt_from_local_knowledge() -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )

    response = service.ask(prompt="neutron star vs black hole")

    assert response["mode"] == "research"
    assert response["kind"] == "research.comparison"
    assert "neutron star" in response["summary"].lower()
    assert "black hole" in response["summary"].lower()
    assert "main difference" in response["summary"].lower()


def test_interaction_service_answers_voltage_prompt_from_local_knowledge() -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )

    response = service.ask(prompt="What is voltage?")

    assert response["mode"] == "research"
    assert response["kind"] == "research.summary"
    assert "voltage" in response["summary"].lower()
    assert "potential" in response["summary"].lower() or "circuit" in response["summary"].lower()
    assert response["findings"] == []


def test_interaction_service_answers_new_history_prompt_from_local_knowledge() -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )

    response = service.ask(prompt="Abraham Lincoln")

    assert response["mode"] == "research"
    assert response["kind"] == "research.summary"
    assert "abraham lincoln" in response["summary"].lower()
    assert "civil war" in response["summary"].lower() or "abolition" in response["summary"].lower()


def test_interaction_service_answers_broad_history_prompt_substantively() -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )

    response = service.ask(prompt="tell me about history")

    assert response["mode"] == "research"
    assert response["kind"] == "research.summary"
    assert "history" in response["summary"].lower()
    assert "past" in response["summary"].lower()


@pytest.mark.parametrize(
    ("prompt", "entry_id"),
    [
        ("tell me about physics", "physics.physics"),
        ("teach me biology", "biology.biology"),
        ("how does chemistry work", "chemistry.chemistry"),
        ("what is earth science", "earth_science.earth_science"),
        ("explain computer science", "systems.computing"),
        ("what is history", "history.history"),
    ],
)
def test_interaction_service_prefers_curated_knowledge_over_weak_research(
    prompt: str,
    entry_id: str,
) -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )

    response = service.ask(prompt=prompt)

    assert response["mode"] == "research"
    assert response["kind"] == "research.summary"
    assert response["domain_surface"]["lane"] == "knowledge"
    assert response["domain_surface"]["entry_id"] == entry_id
    assert response["local_knowledge_access"]["local_knowledge_match"] is True
    assert response["local_knowledge_access"]["final_source"] == "curated_local_knowledge"


def test_interaction_service_answers_conceptual_tool_question_from_local_knowledge() -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )

    response = service.ask(prompt="what is ANH")

    assert response["mode"] == "research"
    assert response["kind"] == "research.summary"
    assert response["domain_surface"]["lane"] == "knowledge"
    assert response["domain_surface"]["entry_id"] == "astronomy.anh"
    assert response["local_knowledge_access"]["final_source"] == "curated_local_knowledge"
    assert "tool_access" not in response


@pytest.mark.parametrize("style", ["default", "collab", "direct"])
def test_interaction_service_conversational_boundary_is_tone_stable(style: str) -> None:
    session_context_service = FakeSessionContextService()
    session_context_service.interaction_profile = InteractionProfile(
        interaction_style=style,
        reasoning_depth="normal",
        selection_source="user",
    )
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=session_context_service,
        inference_service=None,
    )

    for prompt in ("hey buddy", "just chill vibes", "what about you?", "that makes sense"):
        response = service.ask(prompt=prompt)
        assert response["mode"] == "conversation"
        assert response["conversation_access"]["conversation_candidate_consulted"] is True


def test_interaction_service_explicit_tool_execution_reports_tool_access() -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )

    response = service.ask(prompt="run anh")

    assert response["mode"] == "tool"
    assert response["kind"] == "tool.command_alias"
    assert response["tool_access"]["tool_candidate_consulted"] is True
    assert response["tool_access"]["tool_id"] == "anh"
    assert response["tool_access"]["capability"] == "spectral_dip_scan"
    assert response["tool_access"]["tool_execution_required"] is True
    assert response["tool_access"]["final_source"] == "local_tool_execution"


def test_interaction_service_knowledge_follow_up_keeps_curated_source_metadata() -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )
    service._fake_interaction_history_service.recent_records = lambda **kwargs: [
        {
            "mode": "research",
            "kind": "research.summary",
            "prompt": "Galaxy",
            "summary": "A galaxy is a vast gravitationally bound collection of stars, gas, dust, and dark matter.",
            "response": {
                "mode": "research",
                "kind": "research.summary",
                "summary": "A galaxy is a vast gravitationally bound collection of stars, gas, dust, and dark matter.",
                "domain_surface": {
                    "lane": "knowledge",
                    "topic": "Galaxy",
                    "entry_id": "astronomy.galaxy",
                },
            },
        }
    ]

    response = service.ask(prompt="examples?")

    assert response["mode"] == "research"
    assert response["kind"] == "research.summary"
    assert response["domain_surface"]["lane"] == "knowledge"
    assert response["domain_surface"]["topic"] == "Galaxy"
    assert response["local_knowledge_access"]["knowledge_entry_id"] == "astronomy.galaxy"
    assert response["local_knowledge_access"]["knowledge_match_type"] == "follow_up_continuity"


@pytest.mark.parametrize(
    "prompt",
    [
        "what organ pumps blood through out the body?",
        "what is the zodiac",
        "tell me about astrology",
    ],
)
def test_interaction_service_answers_phase90_broad_prompts_without_scaffold(prompt: str) -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )

    response = service.ask(prompt=prompt)
    text = str(response.get("user_facing_answer") or response.get("summary") or "")

    assert response["mode"] == "research"
    assert response["kind"] == "research.summary"
    assert text
    assert "best first read" not in text.lower()
    assert "best current assumptions" not in text.lower()
    assert "best next check" not in text.lower()
    if "zodiac" in prompt or "astrology" in prompt:
        assert "scientific prediction" in text.lower() or "not the same as astronomy" in text.lower()


def test_interaction_service_answers_explain_history_prompt_substantively() -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )

    response = service.ask(prompt="explain history")

    assert response["mode"] == "research"
    assert response["kind"] == "research.summary"
    assert "history" in response["summary"].lower()
    assert "past" in response["summary"].lower()


def test_interaction_service_answers_milky_way_prompt_from_local_knowledge() -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )

    response = service.ask(prompt="tell me about the Milky Way")

    assert response["mode"] == "research"
    assert response["kind"] == "research.summary"
    assert "milky way" in response["summary"].lower()
    assert "galaxy" in response["summary"].lower()


@pytest.mark.parametrize("prompt", ["tell me about space", "space itself", "outer space", "universe", "cosmos"])
def test_interaction_service_answers_broad_space_prompts_from_local_knowledge(prompt: str) -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )

    response = service.ask(prompt=prompt)

    assert response["mode"] == "research"
    assert response["kind"] == "research.summary"
    assert response["domain_surface"]["lane"] == "knowledge"
    assert response["domain_surface"]["entry_id"] == "astronomy.space"
    assert "space" in response["summary"].lower()


@pytest.mark.parametrize("style", ["default", "collab", "direct"])
def test_interaction_service_picks_up_space_after_social_starter(style: str) -> None:
    history = ProjectAwareInteractionHistoryService(
        session_recent=[
            {
                "mode": "conversation",
                "kind": "conversation.greeting",
                "summary": "Good to see you. What are we picking up?",
                "response": {
                    "mode": "conversation",
                    "kind": "conversation.greeting",
                    "reply": "Good to see you. What are we picking up?",
                },
            }
        ]
    )
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=history,
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )
    service._fake_session_context_service.interaction_profile = InteractionProfile(
        interaction_style=style,
        reasoning_depth="normal",
        selection_source="user",
    )

    response = service.ask(prompt="let's pick up space")

    assert response["mode"] == "research"
    assert response["kind"] == "research.summary"
    assert response["domain_surface"]["lane"] == "knowledge"
    assert response["domain_surface"]["entry_id"] == "astronomy.space"
    assert "space" in response["summary"].lower()


@pytest.mark.parametrize("style", ["default", "collab", "direct"])
def test_interaction_service_handles_chill_vibes_as_conversation(style: str) -> None:
    history = ProjectAwareInteractionHistoryService(
        session_recent=[
            {
                "mode": "conversation",
                "kind": "conversation.check_in",
                "summary": "I'm doing well. What's the vibe today?",
                "response": {
                    "mode": "conversation",
                    "kind": "conversation.check_in",
                    "reply": "I'm doing well. What's the vibe today?",
                },
            }
        ]
    )
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=history,
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )
    service._fake_session_context_service.interaction_profile = InteractionProfile(
        interaction_style=style,
        reasoning_depth="normal",
        selection_source="user",
    )

    response = service.ask(prompt="just chill vibes")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.vibe_reply"
    assert response["interaction_mode"] == "social"
    assert "local knowledge" not in response["summary"].lower()


@pytest.mark.parametrize(
    ("prompt", "entry_id"),
    [
        ("tell me about biology", "biology.biology"),
        ("what is chemistry", "chemistry.chemistry"),
        ("physics", "physics.physics"),
        ("teach me about math", "math.mathematics"),
        ("what is computer science", "systems.computing"),
        ("tell me about earth science", "earth_science.earth_science"),
        ("what is engineering", "engineering.engineering"),
    ],
)
@pytest.mark.parametrize("style", ["default", "collab", "direct"])
def test_interaction_service_answers_broad_domain_prompts_from_local_knowledge_in_all_styles(
    style: str,
    prompt: str,
    entry_id: str,
) -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )
    service._fake_session_context_service.interaction_profile = InteractionProfile(
        interaction_style=style,
        reasoning_depth="deep",
        selection_source="user",
    )

    response = service.ask(prompt=prompt)

    assert response["mode"] == "research"
    assert response["kind"] == "research.summary"
    assert response["domain_surface"]["lane"] == "knowledge"
    assert response["domain_surface"]["entry_id"] == entry_id
    assert response["interaction_profile"]["interaction_style"] == style
    assert "don't have enough local knowledge" not in response["summary"].lower()


@pytest.mark.parametrize(
    ("prompt", "entry_id"),
    [
        ("what is motion", "physics.motion"),
        ("tell me about waves", "physics.waves"),
        ("how does chemistry work", "chemistry.chemistry"),
        ("teach me algebra", "math.algebra"),
        ("what is climate", "earth.weather_and_climate"),
        ("explain the engineering design process", "engineering.design_process"),
        ("tell me about ancient civilizations", "history.ancient_civilizations"),
        ("what are scientific theories", "science.models_laws_theories"),
        ("explain cybersecurity", "systems.cybersecurity"),
    ],
)
@pytest.mark.parametrize("style", ["default", "collab", "direct"])
def test_interaction_service_answers_second_ring_domain_prompts_in_all_styles(
    style: str,
    prompt: str,
    entry_id: str,
) -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )
    service._fake_session_context_service.interaction_profile = InteractionProfile(
        interaction_style=style,
        reasoning_depth="deep",
        selection_source="user",
    )

    response = service.ask(prompt=prompt)

    assert response["mode"] == "research"
    assert response["kind"] == "research.summary"
    assert response["domain_surface"]["lane"] == "knowledge"
    assert response["domain_surface"]["entry_id"] == entry_id
    assert response["interaction_profile"]["interaction_style"] == style
    assert "don't have enough local knowledge" not in response["summary"].lower()


@pytest.mark.parametrize(
    ("prompt", "entry_id"),
    [
        ("what is ANH", "astronomy.anh"),
        ("what is MAST data", "astronomy.mast_data"),
        ("what does HST/COS mean", "astronomy.hst_cos"),
        ("what is a spectral dip scan", "astronomy.anh_spectral_dip_scan"),
    ],
)
def test_interaction_service_answers_anh_explanation_prompts_without_running_tool(
    prompt: str,
    entry_id: str,
) -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )

    response = service.ask(prompt=prompt)

    assert response["mode"] == "research"
    assert response["kind"] == "research.summary"
    assert response["domain_surface"]["lane"] == "knowledge"
    assert response["domain_surface"]["entry_id"] == entry_id
    assert "tool_result" not in response


def test_interaction_service_routes_casual_lead_in_with_real_knowledge_intent_to_local_knowledge() -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )

    response = service.ask(prompt="hey buddy explain biology")

    assert response["mode"] == "research"
    assert response["kind"] == "research.summary"
    assert response["domain_surface"]["entry_id"] == "biology.biology"


def test_interaction_service_routes_casual_lead_in_with_cross_domain_intent_to_local_knowledge() -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )

    response = service.ask(prompt="hey buddy explain chemistry")

    assert response["mode"] == "research"
    assert response["kind"] == "research.summary"
    assert response["domain_surface"]["entry_id"] == "chemistry.chemistry"


def test_interaction_service_routes_what_do_you_know_about_broad_topic_to_local_knowledge() -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )

    response = service.ask(prompt="what do you know about biology")

    assert response["mode"] == "research"
    assert response["kind"] == "research.summary"
    assert response["domain_surface"]["entry_id"] == "biology.biology"


def test_interaction_service_answers_informal_milky_way_prompt_from_local_knowledge() -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )

    response = service.ask(prompt="hey what's the Milky Way")

    assert response["mode"] == "research"
    assert response["kind"] == "research.summary"
    assert "milky way" in response["summary"].lower()


def test_interaction_service_answers_new_engineering_prompt_from_local_knowledge() -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )

    response = service.ask(prompt="What is a rocket engine?")

    assert response["mode"] == "research"
    assert response["kind"] == "research.summary"
    assert "rocket engine" in response["summary"].lower()
    assert "thrust" in response["summary"].lower()
    assert "propulsion system" in response["summary"].lower()


def test_interaction_service_design_me_an_engine_recovers_to_planning() -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )

    response = service.ask(prompt="design me an engine")

    assert response["mode"] == "planning"
    assert response["kind"] == "planning.architecture"
    assert any("engine" in step.lower() or "first-pass concept" in step.lower() for step in response["steps"])
    assert "reply" in response
    assert "Assumptions:" in response["reply"]
    assert "Key components:" in response["reply"]


def test_interaction_service_routes_broader_design_entry_prompt_into_planning() -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )

    response = service.ask(prompt="come up with a design for a propulsion engine")

    assert response["mode"] == "planning"
    assert any("propulsion engine" in step.lower() or "first-pass concept" in step.lower() for step in response["steps"])


def test_interaction_service_carries_design_follow_up_into_planning() -> None:
    history = SparseInteractionHistoryService()
    history.recent_records = lambda **kwargs: [
        {
            "prompt": "design me an engine",
            "mode": "planning",
            "kind": "planning.architecture",
            "summary": "Here’s a first-pass design concept for engine.",
        }
    ]
    service = make_interaction_service(interaction_history_service=history)

    response = service.ask(prompt="what else")

    assert response["mode"] == "planning"
    assert any("design" in step.lower() or "engine" in step.lower() for step in response["steps"])


def test_interaction_service_uses_related_connection_in_system_answer() -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )

    response = service.ask(prompt="What is a control system?")

    assert response["mode"] == "research"
    assert response["kind"] == "research.summary"
    assert "control system" in response["summary"].lower()
    assert "feedback loop" in response["summary"].lower()


def test_interaction_service_expands_default_knowledge_answer_with_light_partner_tail() -> None:
    session_context_service = FakeSessionContextService()
    session_context_service.interaction_profile = InteractionProfile(
        interaction_style="default",
        reasoning_depth="normal",
        selection_source="user",
    )
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=session_context_service,
        inference_service=None,
    )

    response = service.ask(prompt="What is voltage?")

    assert response["continuation_offer"]["kind"] in {"break_down", "go_deeper"}


def test_interaction_service_uses_more_expressive_knowledge_surface_in_collab_mode() -> None:
    session_context_service = FakeSessionContextService()
    session_context_service.interaction_profile = InteractionProfile(
        interaction_style="collab",
        reasoning_depth="normal",
        selection_source="user",
    )
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=session_context_service,
        inference_service=None,
    )

    response = service.ask(prompt="What is voltage?")

    assert not response["summary"].startswith("Sure. ")
    assert "voltage" in response["summary"].lower()
    assert response["continuation_offer"]["kind"] in {"break_down", "go_deeper"}


def test_interaction_service_keeps_direct_knowledge_answer_concise() -> None:
    session_context_service = FakeSessionContextService()
    session_context_service.interaction_profile = InteractionProfile(
        interaction_style="direct",
        reasoning_depth="normal",
        selection_source="user",
    )
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=session_context_service,
        inference_service=None,
    )

    response = service.ask(prompt="What is voltage?")

    assert "If you want" not in response["summary"]
    assert "voltage" in response["summary"].lower()
    assert len(str(response["summary"]).split(". ")) >= 2


def test_interaction_service_keeps_research_route_under_mode_changes() -> None:
    for style in ("default", "collab", "direct"):
        session_context_service = FakeSessionContextService()
        session_context_service.interaction_profile = InteractionProfile(
            interaction_style=style,
            reasoning_depth="normal",
            selection_source="user",
        )
        service = make_interaction_service(
            archive_service=SparseArchiveService(),
            interaction_history_service=SparseInteractionHistoryService(),
            session_context_service=session_context_service,
            inference_service=None,
        )

        response = service.ask(prompt="What is voltage?")

        assert response["mode"] == "research"
        assert response["kind"] == "research.summary"


def test_interaction_service_keeps_conversation_route_under_mode_changes() -> None:
    for style in ("default", "collab", "direct"):
        session_context_service = FakeSessionContextService()
        session_context_service.interaction_profile = InteractionProfile(
            interaction_style=style,
            reasoning_depth="normal",
            selection_source="user",
        )
        service = make_interaction_service(session_context_service=session_context_service)

        response = service.ask(prompt="hello lumen")

        assert response["mode"] == "conversation"
        assert response["kind"] == "conversation.greeting"


def test_interaction_service_explanatory_finalization_preserves_route_and_attaches_support_signals() -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )

    response = service.ask(prompt="What is voltage?")

    assert response["mode"] == "research"
    assert response["kind"] == "research.summary"
    assert response["route"]["source"] == "curated_local_knowledge"
    assert response["user_facing_answer"] == response["summary"]
    assert response["route_support_signals"]["broad_explanatory_prompt"] is True
    assert response["route_support_signals"]["blocked_knowledge_prompt"] is False


def test_interaction_service_memory_retrieval_can_influence_surface_without_changing_mode() -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )
    service.memory_retrieval_layer = type(
        "RecallLayer",
        (),
        {
            "retrieve": staticmethod(
                lambda **kwargs: MemoryRetrievalResult(
                    query=str(kwargs.get("prompt") or ""),
                    selected=[
                        RetrievedMemory(
                            source="graph_memory",
                            memory_kind="durable_user_memory",
                            label="routing preference",
                            summary="You prefer direct architecture summaries.",
                            relevance=0.92,
                            metadata={},
                        )
                    ],
                    memory_reply_hint="You prefer direct architecture summaries.",
                    recall_prompt=True,
                )
            )
        },
    )()

    response = service.ask(prompt="what do you remember about my preferences?")

    assert response["mode"] == "research"
    assert response["kind"] == "research.general"
    assert response["summary"] == "You prefer direct architecture summaries."


def test_interaction_service_extracted_packaging_helpers_preserve_pipeline_payloads() -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )

    planning = service.ask(prompt="create a migration plan for lumen")
    research = service.ask(prompt="What is voltage?")

    for response in (planning, research):
        assert "pipeline_execution" in response
        assert "pipeline_packaging" in response
        assert "pipeline_trace" in response


def test_interaction_service_extracted_memory_surface_helpers_preserve_recall_behavior() -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )
    service.memory_retrieval_layer = type(
        "RecallLayer",
        (),
        {
            "retrieve": staticmethod(
                lambda **kwargs: MemoryRetrievalResult(
                    query=str(kwargs.get("prompt") or ""),
                    selected=[
                        RetrievedMemory(
                            source="graph_memory",
                            memory_kind="durable_user_memory",
                            label="project note",
                            summary="The routing project should stay modular.",
                            relevance=0.91,
                            metadata={},
                        )
                    ],
                    memory_reply_hint="The routing project should stay modular.",
                    recall_prompt=True,
                )
            )
        },
    )()

    response = service.ask(prompt="what do you remember about the routing project?")

    assert response["mode"] == "research"
    assert response["summary"] == "The routing project should stay modular."
    assert response["memory_retrieval"]["selected"][0]["source"] == "graph_memory"


def test_interaction_service_uses_partial_match_fallback_for_known_entity_weak_subtopic() -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )

    response = service.ask(prompt="Tell me about George Washington's leadership style")

    assert response["mode"] == "research"
    assert response["kind"] == "research.summary"
    assert "i know who george washington is" in response["summary"].lower()
    assert "leadership style" in response["summary"].lower()
    assert response["findings"] == []


def test_interaction_service_keeps_archive_structure_summary_out_of_knowledge_finalizer() -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )

    response = service.ask(prompt="summarize the current archive structure")

    assert response["mode"] == "research"
    assert response["kind"] == "research.summary"
    assert response.get("findings") is not None
    assert "if you want, i can unpack it a bit further" not in str(response["summary"]).lower()
    assert "grounded detail" not in str(response["summary"]).lower()


def test_interaction_service_uses_clean_weak_match_fallback_for_unknown_general_subject() -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )

    response = service.ask(prompt="hyperbolic lemon engine")

    assert response["mode"] == "research"
    assert response["kind"] == "research.summary"
    assert "grounded detail" not in response["summary"].lower()
    assert "concept or subject worth explaining" not in response["summary"].lower()
    assert "because" in response["summary"].lower()
    assert "don't have enough local knowledge" in response["summary"].lower()
    assert "general concept" not in response["summary"].lower()


def test_interaction_service_answers_what_are_prompt_from_local_knowledge() -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )

    response = service.ask(prompt="what are watts")

    assert response["mode"] == "research"
    assert response["kind"] == "research.summary"
    assert "power" in response["summary"].lower()
    assert "grounded detail" not in response["summary"].lower()
    assert response["reply"] == response["summary"]
    assert response["user_facing_answer"] == response["summary"]
    assert response["findings"] == []


def test_interaction_service_answers_what_do_watts_mean_again_from_local_knowledge() -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )

    response = service.ask(prompt="what do watts mean again")

    assert response["mode"] == "research"
    assert response["kind"] == "research.summary"
    assert "power" in response["summary"].lower()


def test_interaction_service_answers_what_do_mean_prompt_from_local_knowledge() -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )

    response = service.ask(prompt="what do watts mean")

    assert response["mode"] == "research"
    assert response["kind"] == "research.summary"
    assert "power" in response["summary"].lower()
    assert "grounded detail" not in response["summary"].lower()
    assert response["reply"] == response["summary"]
    assert response["user_facing_answer"] == response["summary"]
    assert response["findings"] == []


def test_interaction_service_answers_what_do_ohms_mean_from_local_knowledge() -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )

    response = service.ask(prompt="what do ohms mean")

    assert response["mode"] == "research"
    assert response["kind"] == "research.summary"
    assert "resistance" in response["summary"].lower()
    assert "grounded detail" not in response["summary"].lower()
    assert response["reply"] == response["summary"]
    assert response["user_facing_answer"] == response["summary"]
    assert response["findings"] == []


def test_interaction_service_answers_expanded_astronomy_prompt_from_local_knowledge() -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )

    response = service.ask(prompt="Tell me about the Great Attractor.")

    assert response["mode"] == "research"
    assert response["kind"] == "research.summary"
    assert "great attractor" in response["summary"].lower()
    assert "gravitational influence" in response["summary"].lower() or "galax" in response["summary"].lower()
    assert response["reply"] == response["summary"]
    assert response["user_facing_answer"] == response["summary"]
    assert response["findings"] == []


def test_interaction_service_answers_entropy_prompt_from_local_knowledge() -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )

    response = service.ask(prompt="explain entropy simply but correctly")

    assert response["mode"] == "research"
    assert response["kind"] == "research.summary"
    assert "entropy" in response["summary"].lower()
    assert "spread" in response["summary"].lower()
    assert "for example" in response["summary"].lower() or "picture it" in response["summary"].lower()
    assert "don't have enough local knowledge" not in response["summary"].lower()


def test_interaction_service_answers_entropy_deep_prompt_from_local_knowledge() -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )

    response = service.ask(prompt="explain entropy deeply")

    assert response["mode"] == "research"
    assert response["kind"] == "research.summary"
    assert "entropy" in str(response["summary"]).lower()
    assert "deeply" not in str(response["summary"]).lower()
    assert "don't have enough local knowledge on deeply" not in str(response["summary"]).lower()


def test_interaction_service_answers_relational_prompt_with_grounded_bridge() -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )

    response = service.ask(prompt="entropy in relation to black holes")

    summary = str(response["summary"]).lower()
    assert response["mode"] == "research"
    assert "entropy" in summary
    assert "black hole" in summary
    assert "thermodynamic object" in summary or "broader physical theory" in summary


def test_interaction_service_answers_ga_prompt_with_domain_hint_from_local_knowledge() -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )

    response = service.ask(prompt="what is GA in astronomy")

    assert response["mode"] == "research"
    assert response["kind"] == "research.summary"
    assert "great attractor" in response["summary"].lower()


def test_interaction_service_answers_expanded_chemistry_prompt_from_local_knowledge() -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )

    response = service.ask(prompt="Explain the periodic table")

    assert response["mode"] == "research"
    assert response["kind"] == "research.summary"
    assert "periodic table" in response["summary"].lower()


def test_interaction_service_answers_expanded_biology_prompt_from_local_knowledge() -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )

    response = service.ask(prompt="What is photosynthesis?")

    assert response["mode"] == "research"
    assert response["kind"] == "research.summary"
    assert "photosynthesis" in response["summary"].lower()


def test_interaction_service_prevents_prior_thread_contamination_for_fresh_topic_knowledge_prompt() -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )

    first = service.ask(prompt="What is a black hole?")
    second = service.ask(prompt="Tell me about the Great Attractor.")

    assert first["mode"] == "research"
    assert second["mode"] == "research"
    assert "black hole" not in str(second["summary"]).lower()
    assert "event horizon" not in str(second["summary"]).lower()
    assert "great attractor" in str(second["summary"]).lower()


def test_interaction_service_routes_explicit_anh_prompt_with_absolute_path(tmp_path: Path) -> None:
    service = make_interaction_service()
    sample = tmp_path / "fixtures" / "lb6f07nrq_x1d.fits"
    sample.parent.mkdir(parents=True, exist_ok=True)
    sample.write_bytes(b"FAKEFITS")

    response = service.ask(
        prompt=f"run anh {sample}"
    )

    assert response["mode"] == "tool"
    assert response["kind"] == "tool.command_alias"
    assert response["tool_execution"]["tool_id"] == "anh"
    assert response["tool_execution"]["capability"] == "spectral_dip_scan"
    assert response["tool_execution"]["input_path"].endswith("lb6f07nrq_x1d.fits")


def test_interaction_service_uses_updated_low_support_wording() -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )

    response = service.ask(prompt="explain hyperdimensional thermal lattice theory")

    assert response["mode"] == "research"
    assert "because" in str(response["summary"]).lower()
    assert "don't have enough local knowledge" in str(response["summary"]).lower()
    assert "general concept" not in str(response["summary"]).lower()


@pytest.mark.parametrize("style", ["default", "collab", "direct"])
def test_interaction_service_math_answer_stays_invariant_across_modes(style: str) -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )
    service._fake_session_context_service.interaction_profile = InteractionProfile(
        interaction_style=style,
        reasoning_depth="normal",
        selection_source="user",
    )

    response = service.ask(prompt="5+5x6")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.quick_math"
    assert "35" in str(response["summary"])


def test_interaction_service_preserves_planning_substance_across_modes() -> None:
    direct_service = make_interaction_service()
    direct_service._fake_session_context_service.interaction_profile = InteractionProfile(
        interaction_style="direct",
        reasoning_depth="normal",
        selection_source="user",
    )
    collab_service = make_interaction_service()
    collab_service._fake_session_context_service.interaction_profile = InteractionProfile(
        interaction_style="collab",
        reasoning_depth="normal",
        selection_source="user",
    )

    direct = direct_service.ask(prompt="can you help me think through this?")
    collab = collab_service.ask(prompt="can you help me think through this?")

    assert direct["mode"] == collab["mode"] == "planning"
    assert direct["kind"] == collab["kind"] == "planning.architecture"
    assert direct.get("steps") == collab.get("steps")
    assert direct.get("next_action") == collab.get("next_action")


def test_interaction_service_surfaces_solved_math_value_for_tool_execution() -> None:
    response = {"summary": "Solved equation for x"}

    InteractionService._apply_tool_result_surface(
        response=response,
        tool_result=SimpleNamespace(
            tool_id="math",
            capability="solve_equation",
            status="ok",
            structured_data={"variable": "x", "solution": ["3"]},
        ),
    )

    assert response["user_facing_answer"] == "Solved equation for x: x = 3"
    assert response["summary"] == "Solved equation for x: x = 3"


def test_interaction_service_surfaces_math_tool_steps_when_available() -> None:
    response = {"summary": "Solved equation for x"}

    InteractionService._apply_tool_result_surface(
        response=response,
        tool_result=SimpleNamespace(
            tool_id="math",
            capability="solve_equation",
            status="ok",
            structured_data={
                "equation": "3x + 2 = 11",
                "variable": "x",
                "solution": ["3"],
                "steps": [
                    "Subtract 2 from both sides to get 3x = 9",
                    "Divide both sides by 3 to get x = 3",
                ],
            },
        ),
    )

    assert response["user_facing_answer"].startswith("Solved equation for x: x = 3")
    assert "Subtract 2 from both sides to get 3x = 9." in response["user_facing_answer"]
    assert "Divide both sides by 3 to get x = 3." in response["user_facing_answer"]


def test_interaction_service_generates_simple_math_steps_when_missing() -> None:
    response = {"summary": "Solved equation for x"}

    InteractionService._apply_tool_result_surface(
        response=response,
        tool_result=SimpleNamespace(
            tool_id="math",
            capability="solve_equation",
            status="ok",
            structured_data={
                "equation": "3x + 2 = 11",
                "variable": "x",
                "solution": ["3"],
                "steps": [],
            },
        ),
    )

    assert response["user_facing_answer"].startswith("Solved equation for x: x = 3")
    assert "Subtract 2 from both sides to get 3x = 9." in response["user_facing_answer"]
    assert "Divide both sides by 3 to get x = 3." in response["user_facing_answer"]


def test_interaction_service_surfaces_multiple_solutions_for_math_tool_execution() -> None:
    response = {"summary": "Solved equation for x"}

    InteractionService._apply_tool_result_surface(
        response=response,
        tool_result=SimpleNamespace(
            tool_id="math",
            capability="solve_equation",
            status="ok",
            structured_data={"variable": "x", "solution": ["1", "-1"]},
        ),
    )

    assert response["user_facing_answer"] == "Solved equation for x: x = 1, x = -1"
    assert response["summary"] == "Solved equation for x: x = 1, x = -1"


def test_interaction_service_keeps_generic_math_tool_summary_when_solution_missing() -> None:
    response = {"summary": "Solved equation for x"}

    InteractionService._apply_tool_result_surface(
        response=response,
        tool_result=SimpleNamespace(
            tool_id="math",
            capability="solve_equation",
            status="ok",
            structured_data={"variable": "x", "solution": []},
        ),
    )

    assert "user_facing_answer" not in response
    assert response["summary"] == "Solved equation for x"


def test_interaction_service_builds_user_facing_body_when_planning_summary_is_intro_only() -> None:
    service = make_interaction_service()
    response = {
        "summary": "Here's a grounded answer using the best current assumptions.",
        "steps": [
            "Define the ship's mission and operating environment.",
            "Choose a propulsion approach that fits the mass and range goals.",
        ],
        "next_action": "Lock the top-level constraints before refining the structure.",
    }

    service._finalize_user_facing_reasoning_response(
        response=response,
        prompt="how do I build a ship",
        mode="planning",
        route=SimpleNamespace(mode="planning", kind="planning.design"),
        interaction_profile=InteractionProfile.default(),
    )

    assert "user_facing_answer" in response
    assert "Define the ship's mission" in response["user_facing_answer"]
    assert "Choose a propulsion approach" in response["user_facing_answer"]
    assert response["summary"] != "Here's a grounded answer using the best current assumptions."


def test_interaction_service_builds_user_facing_body_when_research_summary_is_intro_only() -> None:
    service = make_interaction_service()
    response = {
        "mode": "research",
        "summary": "Here's a grounded answer.",
        "findings": ["A comet is a small icy body that can grow a glowing coma and tail near the Sun."],
        "domain_surface": {"lane": "knowledge", "topic": "comet"},
    }

    service._ensure_visible_response_body(
        response=response,
        prompt="what is a comet",
        mode="research",
    )

    assert response["user_facing_answer"] != "Here's a grounded answer."
    assert "comet" in response["user_facing_answer"].lower()
    assert "tail" in response["user_facing_answer"].lower()


def test_interaction_service_filters_unrelated_social_active_thread_from_planning_response() -> None:
    archive_service = SparseArchiveService()
    interaction_history_service = SparseInteractionHistoryService()
    session_context_service = FakeSessionContextService()
    session_context_service.active_thread = {
        "session_id": "default",
        "mode": "conversation",
        "kind": "conversation.greeting",
        "prompt": "good to see you too",
        "summary": "Good to see you too.",
        "thread_summary": "Good to see you too.",
        "normalized_topic": "social greeting",
        "objective": "Keep the conversation warm.",
    }
    service = make_interaction_service(
        archive_service=archive_service,
        interaction_history_service=interaction_history_service,
        session_context_service=session_context_service,
    )

    response = service.ask(prompt="create a migration plan for lumen")

    assert response["mode"] == "planning"
    assert all("good to see you" not in step.lower() for step in response.get("steps", []))
    assert "social greeting" not in str(response.get("local_context_summary") or "").lower()


def test_interaction_service_still_attempts_first_pass_on_tentative_design_prompt() -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )

    response = service.ask(prompt="create me a propulsion engine design")

    assert response["mode"] == "planning"
    assert any("first-pass concept" in step.lower() or "propulsion engine design" in step.lower() for step in response["steps"])
    assert not any(step.lower().startswith("validation plan:") for step in response["steps"])


def test_interaction_service_strips_internal_labels_from_user_facing_planning_output() -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
    )

    response = service.ask(prompt="create me a propulsion engine design")

    summary = str(response.get("summary") or "")
    assert not summary.startswith("Tentative")
    assert not summary.startswith("Planning response")
    assert not summary.startswith("Grounded planning response")


def test_interaction_service_sets_clarify_first_behavior_posture_for_low_confidence_clarification() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="aintnobodygonnadoitlikethat")

    assert response["mode"] == "clarification"
    assert response["response_behavior_posture"]["posture"] == "clarify_first"
    assert response["response_behavior_posture"]["visible_uncertainty"] is True


def test_interaction_service_suppresses_repeated_low_confidence_clarification_loops() -> None:
    service = make_interaction_service()

    first = service.ask(prompt="aintnobodygonnadoitlikethat")
    second = service.ask(prompt="aintnobodygonnadoitlikethat")
    third = service.ask(prompt="aintnobodygonnadoitlikethat")

    assert first["mode"] == "clarification"
    assert second["mode"] == "clarification"
    assert third["mode"] != "clarification"
    assert third["reasoning_state"]["ambiguity_status"] == "degraded_recovery"
    assert "clarification_suppressed" in third["reasoning_state"]["uncertainty_flags"]
    assert "degraded_recovery" in third["reasoning_state"]["failure_flags"]
    assert third["response_behavior_posture"]["posture"] == "stabilize_and_narrow"


def test_interaction_service_personality_surfaces_change_visible_wording() -> None:
    prompt = "create a migration plan for lumen"

    direct_service = make_interaction_service()
    direct_service._fake_session_context_service.interaction_profile = InteractionProfile(
        interaction_style="direct",
        reasoning_depth="normal",
        selection_source="user",
    )
    direct_response = direct_service.ask(prompt=prompt)

    default_service = make_interaction_service()
    default_service._fake_session_context_service.interaction_profile = InteractionProfile(
        interaction_style="default",
        reasoning_depth="normal",
        selection_source="user",
    )
    default_response = default_service.ask(prompt=prompt)

    collab_service = make_interaction_service()
    collab_service._fake_session_context_service.interaction_profile = InteractionProfile(
        interaction_style="collab",
        reasoning_depth="normal",
        selection_source="user",
    )
    collab_response = collab_service.ask(prompt=prompt)

    assert direct_response["response_intro"] != default_response["response_intro"]
    assert default_response["response_intro"] != collab_response["response_intro"]
    assert direct_response["response_opening"] == "Best read first, then the next move."
    assert "I'll keep it clear, grounded, and easy to follow." in default_response["response_opening"]
    assert "I'll give you my read, keep the thread in view, and push it one step further." in collab_response["response_opening"]
    assert direct_response["mode_nlg_profile"]["voice_profile"] == "crisp_focused"
    assert default_response["mode_nlg_profile"]["voice_profile"] == "calm_grounded"
    assert collab_response["mode_nlg_profile"]["voice_profile"] == "warm_partner"
    assert direct_response["mode_nlg_profile"]["reasoning_depth_separate"] is True
    assert default_response["mode_nlg_profile"]["reasoning_depth_separate"] is True
    assert collab_response["mode_nlg_profile"]["reasoning_depth_separate"] is True


def test_interaction_service_style_changes_tone_not_substance() -> None:
    prompt = "create a migration plan for lumen"

    responses = {}
    for style in ("default", "collab", "direct"):
        service = make_interaction_service()
        service._fake_session_context_service.interaction_profile = InteractionProfile(
            interaction_style=style,
            reasoning_depth="normal",
            selection_source="user",
        )
        responses[style] = service.ask(prompt=prompt)

    assert responses["default"]["mode"] == responses["collab"]["mode"] == responses["direct"]["mode"] == "planning"
    assert responses["default"]["kind"] == responses["collab"]["kind"] == responses["direct"]["kind"] == "planning.migration"
    assert responses["default"]["steps"] == responses["collab"]["steps"] == responses["direct"]["steps"]
    assert responses["default"]["next_action"] == responses["collab"]["next_action"] == responses["direct"]["next_action"]
    assert responses["default"]["mode_nlg_profile"]["voice_profile"] != responses["collab"]["mode_nlg_profile"]["voice_profile"]
    assert responses["default"]["mode_nlg_profile"]["voice_profile"] != responses["direct"]["mode_nlg_profile"]["voice_profile"]
    assert responses["collab"]["mode_nlg_profile"]["voice_profile"] != responses["direct"]["mode_nlg_profile"]["voice_profile"]


def test_interaction_service_reasoning_depth_stays_separate_from_voice_mode() -> None:
    service = make_interaction_service()
    service._fake_session_context_service.interaction_profile = InteractionProfile(
        interaction_style="direct",
        reasoning_depth="deep",
        selection_source="user",
    )

    response = service.ask(prompt="explain entropy simply")

    assert response["mode_nlg_profile"]["mode"] == "direct"
    assert response["mode_nlg_profile"]["voice_profile"] == "crisp_focused"
    assert response["mode_nlg_profile"]["reasoning_depth_separate"] is True


def test_interaction_service_clarifies_ambiguous_memory_save_target() -> None:
    interaction_history_service = MemoryClarificationHistoryService(
        recent_records=[
            {
                "session_id": "default",
                "prompt": "tell me about black holes",
                "summary": "Black holes overview",
                "mode": "research",
                "kind": "research.summary",
                "response": {"mode": "research", "summary": "Black holes overview"},
                "interaction_path": "fake/research.json",
            },
            {
                "session_id": "default",
                "prompt": "i think i want to explore this more personally",
                "summary": "Let's stay with that thought.",
                "mode": "conversation",
                "kind": "conversation.thought_follow_up",
                "response": {"mode": "conversation", "summary": "Let's stay with that thought."},
                "interaction_path": "fake/thought.json",
            },
        ]
    )
    service = make_interaction_service(interaction_history_service=interaction_history_service)

    response = service.ask(prompt="save this")

    assert response["mode"] == "clarification"
    assert response["clarification_context"]["clarification_type"] == "memory_save"
    assert response["options"] == ["Save the research", "Save your last thought"]


def test_interaction_service_resolves_memory_save_clarification_into_save() -> None:
    interaction_history_service = MemoryClarificationHistoryService(
        recent_records=[
            {
                "session_id": "default",
                "prompt": "save this",
                "summary": "Clarification requested for memory save target.",
                "mode": "clarification",
                "kind": "clarification.request",
                "response": {
                    "mode": "clarification",
                    "kind": "clarification.request",
                    "clarification_context": {"clarification_type": "memory_save"},
                },
            },
            {
                "session_id": "default",
                "prompt": "tell me about black holes",
                "summary": "Black holes overview",
                "mode": "research",
                "kind": "research.summary",
                "response": {"mode": "research", "summary": "Black holes overview"},
                "interaction_path": "fake/research.json",
            },
        ]
    )
    service = make_interaction_service(interaction_history_service=interaction_history_service)

    response = service.ask(prompt="save the research")

    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.memory_save_confirmation"
    assert response["memory_save_result"]["target"] == "research"
    assert interaction_history_service.saved[0]["source_prompt"] == "tell me about black holes"


def test_interaction_service_uses_explicit_orchestration_stages_for_nontrivial_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    service = make_interaction_service()
    stage_order: list[str] = []

    original_stage_intake = service._stage_intake
    original_recent_interactions = service._recent_interactions_for_turn
    original_stage_route_preparation = service._stage_route_preparation

    def record_stage_intake(**kwargs) -> InteractionTurnContext:
        stage_order.append("intake")
        return original_stage_intake(**kwargs)

    def record_recent_interactions(turn: InteractionTurnContext) -> list[dict[str, object]]:
        stage_order.append("continuation_context")
        return original_recent_interactions(turn)

    def record_route_preparation(**kwargs) -> InteractionTurnContext:
        stage_order.append("route_preparation")
        return original_stage_route_preparation(**kwargs)

    monkeypatch.setattr(service, "_stage_intake", record_stage_intake)
    monkeypatch.setattr(service, "_recent_interactions_for_turn", record_recent_interactions)
    monkeypatch.setattr(service, "_stage_route_preparation", record_route_preparation)

    response = service.ask(prompt="create a migration plan for lumen routing")

    assert response["mode"] == "planning"
    assert stage_order[:3] == ["intake", "continuation_context", "route_preparation"]


def test_interaction_service_attaches_reasoning_state_to_grounded_research_answer() -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
    )

    response = service.ask(prompt="what is entropy")

    reasoning_state = response["reasoning_state"]
    assert response["mode"] == "research"
    assert reasoning_state["current_path"] == "research:research.summary"
    assert reasoning_state["canonical_subject"] == "entropy"
    assert reasoning_state["selected_mode"] == "collab"
    assert reasoning_state["mode_behavior"]["mode"] == "collab"
    assert reasoning_state["turn_status"] == "routed"


def test_interaction_service_attaches_execution_outcome_for_successful_tool_turn(tmp_path: Path) -> None:
    service = make_interaction_service()
    sample_input = tmp_path / "sample.csv"
    sample_input.write_text("x,y\n1,2\n", encoding="utf-8")

    response = service.ask(prompt="run anh", input_path=sample_input)

    execution_outcome = response["execution_outcome"]
    reasoning_state = response["reasoning_state"]

    assert response["mode"] == "tool"
    assert response["tool_execution"]["tool_id"] == "anh"
    assert execution_outcome["execution_attempted"] is True
    assert execution_outcome["execution_status"] == "ok"
    assert execution_outcome["failure_class"] == "success"
    assert reasoning_state["execution_status"] == "ok"
    assert reasoning_state["failure_class"] == "success"
    assert reasoning_state["turn_status"] == "executed"
    assert "tool=anh.spectral_dip_scan status=ok" in str(reasoning_state["known_context_summary"])
    assert "GA Local Analysis Kit run completed" in str(reasoning_state["known_context_summary"])


def test_interaction_service_integrates_generic_tool_result_into_response_surface() -> None:
    response = {
        "mode": "tool",
        "kind": "tool.command_alias",
        "summary": "",
    }
    outcome = ExecutionOutcome(
        selected_tool_id="workspace",
        selected_capability="inspect.structure",
        execution_attempted=True,
        execution_status="ok",
        failure_class="success",
        summary="Workspace structure inspection completed",
    )

    InteractionService._integrate_tool_execution_into_response(
        response=response,
        tool_result=SimpleNamespace(
            tool_id="workspace",
            capability="inspect.structure",
            summary="Workspace structure inspection completed",
        ),
        outcome=outcome,
        reasoning_state=ReasoningStateFrame(canonical_subject="workspace structure"),
    )

    assert response["summary"] == "Workspace structure inspection completed"
    assert response["reply"] == "Workspace structure inspection completed"
    assert response["user_facing_answer"] == "Workspace structure inspection completed"
    assert response["domain_surface"]["lane"] == "tool"
    assert response["domain_surface"]["tool_id"] == "workspace"


def test_reasoning_state_service_classifies_content_runtime_dependency_failure() -> None:
    outcome = ReasoningStateService.classify_execution_outcome(
        tool_result=SimpleNamespace(
            tool_id="content",
            capability="generate_ideas",
            status="error",
            summary="Content idea generation is unavailable until a hosted content provider is configured.",
            structured_data={
                "result_quality": "capability_unavailable",
                "failure_category": "missing_provider_config",
                "failure_reason": "Hosted content generation is not configured for this runtime.",
                "runtime_diagnostics": {
                    "provider_status": "missing_provider_config",
                    "runtime_ready": False,
                },
            },
            artifacts=[],
        )
    )

    assert outcome.failure_class == "runtime_dependency_failure"
    assert outcome.runtime_diagnostics["provider_status"] == "missing_provider_config"
    assert outcome.runtime_diagnostics["failure_reason"] == "Hosted content generation is not configured for this runtime."


def test_reasoning_state_service_classifies_anh_artifact_failure_from_analysis_status() -> None:
    outcome = ReasoningStateService.classify_execution_outcome(
        tool_result=SimpleNamespace(
            tool_id="anh",
            capability="spectral_dip_scan",
            status="partial",
            summary="ANH analyzed 1 file(s) but some expected plots were not created.",
            structured_data={
                "analysis_status": {
                    "result_quality": "partial_artifacts",
                    "failure_reason": "Some files were skipped, failed inspection, or only partially analyzed.",
                    "plot_generated": False,
                    "runtime_diagnostics": {
                        "astropy": {"available": True, "version": "1.0"},
                    },
                },
                "domain_payload": {
                    "accepted_files": [
                        {
                            "filename": "sample_x1d.fits",
                            "artifact_generation_status": {
                                "overview_plot_expected": True,
                                "overview_plot_created": True,
                                "window_plot_expected": True,
                                "window_plot_created": False,
                            },
                        }
                    ]
                },
            },
            artifacts=[],
        )
    )

    assert outcome.failure_class == "artifact_failure"
    assert outcome.runtime_diagnostics["analysis_result_quality"] == "partial_artifacts"
    assert outcome.artifact_signals["missing_artifact_files"] == 1


def test_reasoning_state_service_treats_candidate_detected_partial_anh_as_success() -> None:
    outcome = ReasoningStateService.classify_execution_outcome(
        tool_result=SimpleNamespace(
            tool_id="anh",
            capability="spectral_dip_scan",
            status="partial",
            summary="ANH analyzed 24 file(s) and found 23 candidate file(s).",
            structured_data={
                "analysis_status": {
                    "result_quality": "candidate_dips_detected",
                    "failure_reason": "Some files were skipped, failed inspection, or only partially analyzed.",
                    "plot_generated": True,
                    "line_detected": True,
                },
                "batch_record": {
                    "files_discovered": 34,
                    "files_analyzed": 24,
                    "candidate_files": [{"filename": "sample_x1d.fits"}],
                },
            },
            artifacts=[SimpleNamespace(name="candidate_rankings.json")],
        )
    )

    assert outcome.execution_status == "partial"
    assert outcome.failure_class == "success"


def test_interaction_service_uses_reasoning_state_for_keep_current_route_follow_up() -> None:
    session_context_service = FakeSessionContextService()
    session_context_service.active_thread = {
        "session_id": "default",
        "mode": "planning",
        "kind": "planning.migration",
        "prompt": "review the migration summary",
        "summary": "Planning response for: review the migration summary",
        "thread_summary": "Planning thread",
        "objective": "Plan work for: create a migration plan for lumen",
        "reasoning_state": {
            "current_path": "planning:planning.migration",
            "pending_followup": {
                "type": "clarification",
                "action": "continue",
                "route_mode": "planning",
                "route_kind": "planning.migration",
                "resolved_prompt": "review the migration summary",
            },
            "resolved_prompt": "review the migration summary",
            "continuation_target": "review the migration summary",
            "selected_mode": "default",
            "turn_status": "clarifying",
        },
    }
    interaction_history_service = FakeInteractionHistoryService()
    interaction_history_service.recent_records = lambda **kwargs: []
    service = make_interaction_service(
        session_context_service=session_context_service,
        interaction_history_service=interaction_history_service,
    )

    response = service.ask(prompt="keep current route")

    reasoning_state = response["reasoning_state"]
    assert response["mode"] == "planning"
    assert response.get("resolution_strategy") == "clarification_route_confirmation"
    assert reasoning_state["ambiguity_status"] == "resolved"
    assert reasoning_state["current_path"] == "planning:planning.migration"
    assert reasoning_state["turn_status"] == "clarification_resumed"


def test_interaction_service_pivots_cleanly_from_reasoning_state_backed_follow_up() -> None:
    session_context_service = FakeSessionContextService()
    session_context_service.active_thread = {
        "session_id": "default",
        "mode": "planning",
        "kind": "planning.migration",
        "prompt": "review the migration summary",
        "summary": "Planning response for: review the migration summary",
        "thread_summary": "Planning thread",
        "objective": "Plan work for: create a migration plan for lumen",
        "reasoning_state": {
            "current_path": "planning:planning.migration",
            "pending_followup": {
                "type": "clarification",
                "action": "continue",
                "route_mode": "planning",
                "route_kind": "planning.migration",
                "resolved_prompt": "review the migration summary",
            },
            "resolved_prompt": "review the migration summary",
            "continuation_target": "review the migration summary",
            "selected_mode": "default",
            "turn_status": "clarifying",
        },
    }
    interaction_history_service = FakeInteractionHistoryService()
    interaction_history_service.recent_records = lambda **kwargs: []
    service = make_interaction_service(
        session_context_service=session_context_service,
        interaction_history_service=interaction_history_service,
    )

    response = service.ask(prompt="let's explore black holes instead")

    reasoning_state = response["reasoning_state"]
    assert response["mode"] == "research"
    assert "black hole" in str(response["summary"]).lower()
    assert "black hole" in reasoning_state["canonical_subject"]
    assert reasoning_state["current_path"] == "research:research.summary"


def test_interaction_service_uses_reasoning_state_for_break_it_down_follow_up() -> None:
    session_context_service = FakeSessionContextService()
    session_context_service.active_thread = {
        "session_id": "default",
        "mode": "research",
        "kind": "research.summary",
        "prompt": "what is entropy",
        "summary": "Entropy is about energy spreading out.",
        "thread_summary": "Entropy explanation thread",
        "objective": "Research topic: what is entropy",
        "reasoning_state": {
            "current_path": "research:research.summary",
            "canonical_subject": "entropy",
            "resolved_prompt": "what is entropy",
            "continuation_target": "what is entropy",
            "selected_mode": "default",
            "turn_status": "routed",
        },
    }
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=session_context_service,
        inference_service=None,
    )

    response = service.ask(prompt="break it down")

    reasoning_state = response["reasoning_state"]
    assert response["mode"] == "research"
    assert "entropy" in str(response["summary"]).lower()
    assert reasoning_state["resolved_prompt"] == "explain entropy simply"
    assert reasoning_state["explanation_strategy"] in {"concrete_example", "direct_definition"}


def test_interaction_service_persists_declined_clarification_state() -> None:
    session_context_service = FakeSessionContextService()
    session_context_service.active_thread = {
        "session_id": "default",
        "mode": "planning",
        "kind": "planning.migration",
        "prompt": "review the migration summary",
        "summary": "Planning response for: review the migration summary",
        "thread_summary": "Planning thread",
        "objective": "Plan work for: create a migration plan for lumen",
        "reasoning_state": {
            "current_path": "planning:planning.migration",
            "pending_followup": {
                "type": "clarification",
                "action": "continue",
                "route_mode": "planning",
                "route_kind": "planning.migration",
                "resolved_prompt": "review the migration summary",
            },
            "resolved_prompt": "review the migration summary",
            "continuation_target": "review the migration summary",
            "selected_mode": "default",
            "turn_status": "clarifying",
        },
    }
    interaction_history_service = FakeInteractionHistoryService()
    interaction_history_service.recent_records = lambda **kwargs: []
    service = make_interaction_service(
        session_context_service=session_context_service,
        interaction_history_service=interaction_history_service,
    )

    response = service.ask(prompt="no")

    reasoning_state = response["reasoning_state"]
    active_thread_state = service._fake_session_context_service.active_thread["reasoning_state"]
    assert response["mode"] == "conversation"
    assert response["kind"] == "conversation.clarification_decline"
    assert reasoning_state["ambiguity_status"] == "declined"
    assert reasoning_state["pending_followup"] == {}
    assert active_thread_state["ambiguity_status"] == "declined"
    assert active_thread_state["pending_followup"] == {}


def test_interaction_service_clarification_uses_reasoning_mode_profile() -> None:
    service = make_interaction_service()
    service._fake_session_context_service.interaction_profile = InteractionProfile(
        interaction_style="collab",
        reasoning_depth="normal",
        selection_source="user",
    )

    response = service.ask(prompt="aintnobodygonnadoitlikethat")

    reasoning_state = response["reasoning_state"]
    assert response["mode"] == "clarification"
    assert response["clarification_question"] == (
        "I’m noticing we’ve got a couple directions we could go here. "
        "Do you want to explore this together, or focus on something specific?"
    )
    assert reasoning_state["selected_mode"] == "collab"
    assert reasoning_state["mode_behavior"]["mode"] == "collab"
    assert reasoning_state["mode_behavior"]["clarification_style"] == "collaborative"


def test_mode_response_shaper_shapes_tool_missing_input_surface_for_collab_mode() -> None:
    profile = InteractionProfile(
        interaction_style="collab",
        reasoning_depth="normal",
        selection_source="user",
    )
    response = {
        "mode": "tool",
        "summary": "I couldn't extract usable topic from that prompt, so I didn't run the tool. Try naming the topic directly, for example: generate content ideas about black holes.",
        "tool_execution_skipped": True,
        "tool_missing_inputs": "topic",
    }

    ModeResponseShaper.apply(response=response, interaction_profile=profile)

    assert response["mode_nlg_profile"]["mode"] == "collab"
    assert str(response["user_facing_answer"]).startswith("I'm ready to do that, but I need usable topic first.")
    assert "generate content ideas about black holes" in str(response["user_facing_answer"]).lower()


def test_mode_response_shaper_shapes_tool_missing_input_surface_for_direct_mode() -> None:
    profile = InteractionProfile(
        interaction_style="direct",
        reasoning_depth="normal",
        selection_source="user",
    )
    response = {
        "mode": "tool",
        "summary": "I couldn't extract usable topic from that prompt, so I didn't run the tool. Try naming the topic directly, for example: generate content batch about black holes.",
        "tool_execution_skipped": True,
        "tool_missing_inputs": "topic",
    }

    ModeResponseShaper.apply(response=response, interaction_profile=profile)

    assert response["mode_nlg_profile"]["mode"] == "direct"
    assert str(response["user_facing_answer"]).startswith("Need usable topic before tool run.")


def test_interaction_service_attaches_mode_specific_follow_up_offer_without_changing_research_substance() -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
    )
    service._fake_session_context_service.interaction_profile = InteractionProfile(
        interaction_style="direct",
        reasoning_depth="normal",
        selection_source="user",
    )

    response = service.ask(prompt="what is entropy")

    assert response["mode"] == "research"
    assert "entropy" in str(response["summary"]).lower()
    assert response["follow_up_offer"] == "I can go one layer deeper."
    assert response["mode_nlg_profile"]["mode"] == "direct"


def test_interaction_service_attaches_intent_domain_behavior_metadata_to_planning_response() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="create a roadmap and prioritize the first steps for migrating lumen")

    assert response["mode"] == "planning"
    assert response["intent_domain"] == "planning_strategy"
    assert response["intent_domain_confidence"] > 0.0
    assert response["response_depth"] in {"standard", "deep"}


def test_interaction_service_realizes_more_natural_planning_body_from_cognitive_state() -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )

    response = service.ask(prompt="create a roadmap and prioritize the first steps for migrating lumen")

    reply = str(response.get("reply") or "")

    assert response["mode"] == "planning"
    assert reply.startswith("Here’s the roadmap I’d start with") or reply.startswith("Here’s the roadmap I’d use")
    assert "\n- " in reply
    assert "\n\nNext step:" in reply or "\n\nNext:" in reply


def test_interaction_service_realizes_more_natural_research_body_from_cognitive_state() -> None:
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        inference_service=None,
    )

    response = {
        "mode": "research",
        "summary": "Here’s a grounded answer.",
        "findings": [
            "Entropy tracks how widely energy is spread through the available states of a system.",
            "Higher entropy usually means the energy is distributed across more possible arrangements.",
        ],
        "recommendation": "Connect it to thermodynamics next so the intuition stays anchored.",
        "intent_domain": "learning_teaching",
        "response_depth": "deep",
        "conversation_phase": "intake",
        "reasoning_state": {
            "selected_mode": "default",
            "intent_domain": "learning_teaching",
            "response_depth": "deep",
            "conversation_phase": "intake",
            "confidence_tier": "high",
        },
    }

    service._ensure_visible_response_body(
        response=response,
        prompt="explain entropy simply but correctly",
        mode="research",
    )

    reply = str(response.get("reply") or "")

    assert response["mode"] == "research"
    assert (
        reply.startswith("Here’s the clearest explanation")
        or reply.startswith("Here’s the clearest read")
        or reply.startswith("Best current read:")
    )
    assert "\n- " in reply or "entropy" in reply.lower()
    assert "\n\nBest next step:" in reply or "\n\nNext:" in reply
    assert response["conversation_phase"] == "intake"


def test_interaction_service_keeps_tone_mode_separate_from_domain_classification() -> None:
    service = make_interaction_service()
    service._fake_session_context_service.interaction_profile = InteractionProfile(
        interaction_style="direct",
        reasoning_depth="normal",
        selection_source="user",
    )

    response = service.ask(prompt="teach me step by step what a black hole is")

    assert response["mode_nlg_profile"]["mode"] == "direct"
    assert response["intent_domain"] == "learning_teaching"
    assert response["tool_suggestion_state"]["should_suggest"] is False


def test_interaction_service_mode_switch_updates_clarification_without_settling_turn() -> None:
    session_context_service = FakeSessionContextService()
    session_context_service.interaction_profile = InteractionProfile(
        interaction_style="direct",
        reasoning_depth="normal",
        selection_source="user",
    )
    session_context_service.active_thread = {
        "session_id": "default",
        "mode": "research",
        "kind": "research.summary",
        "prompt": "tell me about black holes",
        "summary": "Black holes are regions of extreme gravity.",
        "thread_summary": "Black hole thread",
        "objective": "Research black holes",
        "reasoning_state": {
            "current_path": "research:research.summary",
            "canonical_subject": "black holes",
            "resolved_prompt": "tell me about black holes",
            "continuation_target": "tell me about black holes",
            "selected_mode": "collab",
            "turn_status": "routed",
        },
    }
    service = make_interaction_service(
        session_context_service=session_context_service,
        interaction_history_service=SparseInteractionHistoryService(),
        archive_service=SparseArchiveService(),
    )

    response = service.ask(prompt="aintnobodygonnadoitlikethat")

    reasoning_state = response["reasoning_state"]
    assert response["mode"] == "clarification"
    assert response["clarification_question"] == "Ambiguity detected. Choose: summary | comparison | continue."
    assert reasoning_state["selected_mode"] == "direct"
    assert reasoning_state["mode_behavior"]["mode"] == "direct"
    assert reasoning_state["mode_behavior"]["follow_up_style"] == "minimal"


def test_interaction_service_persists_behavior_metadata_into_reasoning_state() -> None:
    service = make_interaction_service()

    response = service.ask(prompt="create a roadmap for stabilizing lumen cognition")

    reasoning_state = response["reasoning_state"]
    persisted_state = service._fake_session_context_service.active_thread["reasoning_state"]
    recorded_state = service._fake_interaction_history_service.records[-1]["response"]["reasoning_state"]
    trainability_trace = response["trainability_trace"]
    persisted_trace = service._fake_session_context_service.active_thread["trainability_trace"]
    recorded_trace = service._fake_interaction_history_service.records[-1]["response"]["trainability_trace"]

    assert reasoning_state["intent_domain"] == "planning_strategy"
    assert reasoning_state["response_depth"] in {"standard", "deep"}
    assert reasoning_state["conversation_phase"] in {"intake", "execution"}
    assert reasoning_state["response_style"]["intent_domain"] == "planning_strategy"
    assert persisted_state["response_style"]["intent_domain"] == "planning_strategy"
    assert recorded_state["response_style"]["intent_domain"] == "planning_strategy"
    assert "tool_use_decision_support" in trainability_trace["available_training_surfaces"]
    assert trainability_trace["intent_domain_classification"]["intent_domain"] == "planning_strategy"
    assert trainability_trace["route_recommendation_support"]["mode"] == "planning"
    assert trainability_trace["response_style_selection"]["intent_domain"] == "planning_strategy"
    assert persisted_trace["response_style_selection"]["intent_domain"] == "planning_strategy"
    assert recorded_trace["response_style_selection"]["intent_domain"] == "planning_strategy"


def test_interaction_service_filters_irrelevant_memory_for_technical_turn() -> None:
    class StubMemoryRetrievalLayer:
        def retrieve(self, **kwargs):
            return MemoryRetrievalResult(
                query=str(kwargs.get("prompt") or ""),
                selected=[
                    RetrievedMemory(
                        source="personal_memory",
                        memory_kind="profile",
                        label="Favorite tea",
                        summary="The user likes chamomile tea before bed.",
                        relevance=0.21,
                        metadata={"category": "preference", "confidence": 0.35},
                    ),
                    RetrievedMemory(
                        source="bug_log",
                        memory_kind="bug_log",
                        label="Parser regression",
                        summary="The parser failed after a schema mismatch in the workspace loader.",
                        relevance=0.93,
                        metadata={"category": "debug", "confidence": 0.9, "recency_days": 2},
                    ),
                ],
                diagnostics={},
            )

    service = make_interaction_service()
    service.memory_retrieval_layer = StubMemoryRetrievalLayer()

    response = service.ask(prompt="debug the workspace parser failure")

    reasoning_state = response["reasoning_state"]
    memory_context = reasoning_state["memory_context_used"]
    memory_diagnostics = response["memory_retrieval"]["diagnostics"]["memory_context_classifier"]

    assert response["mode"] in {"planning", "research", "conversation"}
    assert len(memory_context) == 1
    assert memory_context[0]["label"] == "Parser regression"
    assert len(memory_diagnostics["selected"]) == 1
    assert memory_diagnostics["selected"][0]["label"] == "Parser regression"
    assert any(item["label"] == "Favorite tea" for item in memory_diagnostics["rejected"])


def test_interaction_service_records_tool_gate_and_confidence_for_successful_tool_turn(tmp_path: Path) -> None:
    service = make_interaction_service()
    sample_input = tmp_path / "sample.csv"
    sample_input.write_text("x,y\n1,2\n", encoding="utf-8")

    response = service.ask(prompt="run anh", input_path=sample_input)

    reasoning_state = response["reasoning_state"]

    assert response["mode"] == "tool"
    assert response["tool_threshold_decision"]["should_use_tool"] is True
    assert reasoning_state["tool_usage_intent"]["tool_id"] == "anh"
    assert reasoning_state["tool_decision"]["should_use_tool"] is True
    assert reasoning_state["confidence_tier"] in {"medium", "high"}


def test_interaction_service_records_supervised_support_trace_without_overriding_authority() -> None:
    support = SupervisedDecisionSupport(
        examples_by_surface={
            "intent_domain_classification": [
                SupervisedExample(
                    surface="intent_domain_classification",
                    label="learning_teaching",
                    prompt_terms=("teach", "step", "black", "hole"),
                    route_modes=("research",),
                )
            ]
        }
    )
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        supervised_decision_support=support,
    )

    response = service.ask(prompt="teach me step by step what a black hole is")

    supervised_trace = response["supervised_support_trace"]
    trainability_trace = response["trainability_trace"]
    persisted_trace = service._fake_session_context_service.active_thread["supervised_support_trace"]

    assert response["mode"] == "research"
    assert response["local_knowledge_access"]["knowledge_entry_id"] == "astronomy.black_hole"
    assert response.get("intent_domain") in {"learning_teaching", None}
    assert isinstance(supervised_trace, dict)
    assert isinstance(trainability_trace, dict)
    assert isinstance(persisted_trace, dict)


def test_interaction_service_records_tool_support_trace_for_tool_turn(tmp_path: Path) -> None:
    support = SupervisedDecisionSupport(
        examples_by_surface={
            "tool_use_decision_support": [
                SupervisedExample(
                    surface="tool_use_decision_support",
                    label="use_tool",
                    prompt_terms=("run", "anh"),
                    route_modes=("tool",),
                    tool_id="anh",
                )
            ]
        }
    )
    service = make_interaction_service(supervised_decision_support=support)
    sample_input = tmp_path / "sample.csv"
    sample_input.write_text("x,y\n1,2\n", encoding="utf-8")

    response = service.ask(prompt="run anh", input_path=sample_input)

    supervised_trace = response["supervised_support_trace"]

    assert response["mode"] == "tool"
    assert supervised_trace["enabled"] is True
    assert "tool_use_decision_support" in supervised_trace["recommendations"]
    assert supervised_trace["deterministic_authority_preserved"] is True


def test_interaction_service_expands_supervised_support_to_route_and_response_style_traces() -> None:
    support = SupervisedDecisionSupport(
        examples_by_surface={
            "route_recommendation_support": [
                SupervisedExample(
                    surface="route_recommendation_support",
                    label="planning:planning.migration",
                    prompt_terms=("migration", "plan"),
                    route_modes=("planning",),
                )
            ],
            "response_style_selection": [
                SupervisedExample(
                    surface="response_style_selection",
                    label="systematic",
                    prompt_terms=("migration", "plan"),
                    route_modes=("planning",),
                )
            ],
        }
    )
    service = make_interaction_service(
        archive_service=SparseArchiveService(),
        interaction_history_service=SparseInteractionHistoryService(),
        session_context_service=FakeSessionContextService(),
        supervised_decision_support=support,
    )

    response = service.ask(prompt="create a migration plan for lumen")

    supervised_trace = response["supervised_support_trace"]

    assert "route_recommendation_support" in supervised_trace["recommendations"]
    assert "response_style_selection" in supervised_trace["recommendations"]
    assert supervised_trace["recommendations"]["route_recommendation_support"]["applied"] is False
    assert supervised_trace["recommendations"]["response_style_selection"]["applied"] is False


def test_interaction_service_can_apply_more_cautious_supervised_confidence_calibration() -> None:
    support = SupervisedDecisionSupport(
        examples_by_surface={
            "confidence_calibration_support": [
                SupervisedExample(
                    surface="confidence_calibration_support",
                    label="tentative",
                    prompt_terms=("migration", "summary"),
                    route_modes=("research",),
                )
            ]
        }
    )
    service = make_interaction_service(supervised_decision_support=support)
    route = DomainRoute(
        mode="research",
        kind="research.summary",
        normalized_prompt="review the migration summary",
        confidence=0.62,
        reason="Research cues narrowly outranked planning cues.",
    )
    response = {
        "mode": "research",
        "kind": "research.summary",
        "confidence_posture": "supported",
        "route_status": "weakened",
        "support_status": "moderately_supported",
        "reasoning_state": {
            "confidence": 0.61,
            "confidence_tier": "medium",
            "selected_mode": "default",
            "runtime_diagnostics": {},
        },
    }

    service._apply_supervised_confidence_support(
        response=response,
        prompt="review the migration summary",
        route=route,
    )

    reasoning_state = response["reasoning_state"]
    supervised_trace = reasoning_state.runtime_diagnostics["supervised_support_trace"]
    recommendation = supervised_trace["recommendations"]["confidence_calibration_support"]

    assert recommendation["applied"] is True
    assert response["confidence_posture"] == "tentative"
    assert reasoning_state.confidence_tier == "low"


def test_interaction_service_recent_qa_flow_does_not_stick_to_stale_research_context() -> None:
    service = make_interaction_service(interaction_history_service=SparseInteractionHistoryService())

    expected = [
        ("Hey Lumen!", "conversation", "conversation.greeting", None),
        ("likewise, ive been thinking about mars", "research", "research.summary", "astronomy.mars"),
        ("I just seem to catch edge cases all day lol", "conversation", "conversation.acknowledgment", None),
        ("how are you lumen?", "conversation", "conversation.check_in", None),
        ("space", "research", "research.summary", "astronomy.space"),
        ("physics", "research", "research.summary", "physics.physics"),
        ("biology", "research", "research.summary", "biology.biology"),
        ("tell me about ww2", "research", "research.summary", "history.world_war_ii"),
        ("thats fair thank you for your honesty", "conversation", "conversation.gratitude", None),
        ("im feeling sad today", "conversation", "conversation.emotional_state", None),
        ("im feeling happy", "conversation", "conversation.emotional_state", None),
        ("how was your day?", "conversation", "conversation.check_in", None),
        ("tell me about the moon", "research", "research.summary", "astronomy.moon"),
    ]

    for prompt, mode, kind, entry_id in expected:
        response = service.ask(prompt=prompt, session_id="qa-flow")
        assert response["mode"] == mode, prompt
        assert response["kind"] == kind, prompt
        assert "best first read" not in str(response.get("summary", "")).lower()
        assert "hold it provisionally" not in str(response.get("summary", "")).lower()
        if entry_id is not None:
            assert response["local_knowledge_access"]["knowledge_entry_id"] == entry_id


def test_interaction_service_confirmation_requires_explicit_continuation_offer_after_fallback() -> None:
    service = make_interaction_service(interaction_history_service=SparseInteractionHistoryService())

    fallback = service.ask(prompt="tell me about a completely unknown flarble topic", session_id="fallback-flow")
    assert "don't know" in fallback["summary"].lower() or "don't have enough local knowledge" in fallback["summary"].lower()

    response = service.ask(prompt="yes", session_id="fallback-flow")

    assert not str(response.get("summary") or "").lower().startswith("let's go one layer deeper")
    assert "completely unknown flarble topic" not in str(response.get("summary") or "").lower()


def test_interaction_service_confirmation_follows_explicit_knowledge_offer() -> None:
    service = make_interaction_service(interaction_history_service=SparseInteractionHistoryService())

    first = service.ask(prompt="biology", session_id="knowledge-flow")
    assert first["local_knowledge_access"]["knowledge_entry_id"] == "biology.biology"

    response = service.ask(prompt="yes", session_id="knowledge-flow")

    assert response["mode"] == "research"
    assert response["kind"] == "research.summary"
    assert "biology" in response["summary"].lower()
    assert "what is biology" not in response["summary"].lower()


