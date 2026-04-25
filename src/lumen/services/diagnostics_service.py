from __future__ import annotations

import importlib.util
import sys

from lumen.app.settings import AppSettings
from lumen.memory.archive_manager import ArchiveManager
from lumen.desktop.chat_experience_support import build_capability_transparency_line
from lumen.providers.base import ModelProvider
from lumen.providers.openai_responses_provider import OpenAIResponsesProvider
from lumen.reporting.report_models import DoctorCheck, DoctorReport
from lumen.routing.tool_registry import ToolRegistry
from lumen.services.academic_support_service import AcademicSupportService
from lumen.services.capability_contract_service import CapabilityContractService
from lumen.services.human_thinking_audit import HumanThinkingAuditService
from lumen.services.interaction_history_service import InteractionHistoryService
from lumen.services.safety_service import SafetyService
from lumen.routing.tool_signal_catalog import DORMANT_TOOL_SIGNAL_CATALOG


class DiagnosticsService:
    """Builds local readiness and diagnostics reports."""

    def __init__(
        self,
        settings: AppSettings,
        registry: ToolRegistry,
        model_provider: ModelProvider | None = None,
        archive_manager: ArchiveManager | None = None,
        interaction_history_service: InteractionHistoryService | None = None,
        safety_service: SafetyService | None = None,
        execution_mode: str = "source",
    ):
        self.settings = settings
        self.registry = registry
        self.repo_root = settings.repo_root
        self.model_provider = model_provider
        self.archive_manager = archive_manager
        self.interaction_history_service = interaction_history_service
        self.safety_service = safety_service
        self.execution_mode = str(execution_mode or "source").strip() or "source"

    def build_doctor_report(self) -> dict[str, object]:
        checks: list[DoctorCheck] = []

        def add_check(
            name: str,
            ok: bool,
            details: str,
            extra: dict[str, object] | None = None,
        ) -> None:
            checks.append(
                DoctorCheck(
                    name=name,
                    status="ok" if ok else "error",
                    details=details,
                    extra=extra or {},
                )
            )

        bundles = self.registry.list_tools()
        required_bundles = {"workspace", "report", "memory", "math", "system", "knowledge"}
        discovered_bundle_ids = set(bundles)
        missing_bundles = sorted(required_bundles - discovered_bundle_ids)
        add_check(
            "tool_registry",
            not missing_bundles,
            (
                f"Discovered bundles: {bundles}"
                if not missing_bundles
                else f"Missing required bundles: {', '.join(missing_bundles)}; discovered bundles: {bundles}"
            ),
            {
                "bundles": bundles,
                "required_bundles": sorted(required_bundles),
                "missing_bundles": missing_bundles,
                "bundle_count": len(discovered_bundle_ids),
            },
        )
        bundles_dir = getattr(self.registry, "bundles_dir", self.repo_root / "tool_bundles")
        required_runtime_resources = {
            "tool_bundles": bundles_dir,
        }
        optional_runtime_resources = {
            "examples_root": self.settings.examples_root,
            "config_template": self.repo_root / "lumen.toml.example",
        }
        required_missing_resources = [
            name for name, path in required_runtime_resources.items() if not path.exists()
        ]
        checks.append(
            DoctorCheck(
                name="runtime_resources",
                status="ok" if not required_missing_resources else "error",
                details=(
                    "Runtime resources are present"
                    if not required_missing_resources
                    else "Missing required runtime resources: "
                    + ", ".join(sorted(required_missing_resources))
                ),
                extra={
                    "required_resources": {
                        name: str(path)
                        for name, path in required_runtime_resources.items()
                    },
                    "optional_resources": {
                        name: str(path)
                        for name, path in optional_runtime_resources.items()
                    },
                    "missing_required_resources": required_missing_resources,
                    "optional_resource_presence": {
                        name: path.exists()
                        for name, path in optional_runtime_resources.items()
                    },
                },
            )
        )
        checks.append(
            DoctorCheck(
                name="runtime_layout",
                status="ok" if not missing_bundles else "error",
                details=(
                    f"execution_mode={self.execution_mode}; runtime_root={self.repo_root}; data_root={self.settings.data_root}"
                ),
                extra={
                    "execution_mode": self.execution_mode,
                    "runtime_root": str(self.repo_root),
                    "data_root": str(self.settings.data_root),
                    "bundle_count": len(discovered_bundle_ids),
                    "missing_required_bundles": missing_bundles,
                },
            )
        )
        checks.append(
            DoctorCheck(
                name="dormant_tool_signals",
                status="ok",
                details=f"Dormant future-ready tool signal catalogs: {', '.join(sorted(DORMANT_TOOL_SIGNAL_CATALOG))}",
                extra={"dormant_tool_signals": DORMANT_TOOL_SIGNAL_CATALOG},
            )
        )

        example_csv = self.settings.examples_root / "cf4_ga_cone_template.csv"
        add_check(
            "example_dataset",
            example_csv.exists(),
            f"Expected example dataset at {example_csv}",
            {"path": str(example_csv)},
        )

        add_check(
            "python_runtime",
            bool(sys.executable),
            f"Current interpreter: {sys.executable or '<unavailable>'}",
            {"executable": sys.executable},
        )

        add_check(
            "config",
            True,
            (
                f"Using config file at {self.settings.config_path}"
                if self.settings.config_path
                else "No lumen.toml found; using built-in defaults"
            ),
            {"config_path": str(self.settings.config_path) if self.settings.config_path else None},
        )

        if self.model_provider is not None:
            capabilities = self.model_provider.capabilities()
            provider_extra = {
                "deployment_mode": self.settings.deployment_mode,
                "provider_id": self.model_provider.provider_id,
                "capabilities": capabilities.to_dict(),
            }
            provider_status = "ok"
            provider_details = (
                f"Deployment mode={self.settings.deployment_mode}; "
                f"provider={self.model_provider.provider_id}"
            )
            if isinstance(self.model_provider, OpenAIResponsesProvider):
                provider_extra["api_base"] = self.model_provider.api_base
                provider_extra["default_model"] = self.model_provider.default_model
                provider_extra["api_key_env"] = self.model_provider.api_key_env
                provider_extra["api_key_present"] = self.model_provider.has_api_key()
                if self.settings.deployment_mode in {"hybrid", "hosted"} and not self.model_provider.has_api_key():
                    provider_status = "warn"
                    provider_details += f"; missing {self.model_provider.api_key_env}"
                elif self.model_provider.default_model:
                    provider_details += f"; model={self.model_provider.default_model}"
            elif self.settings.deployment_mode != "local_only":
                provider_status = "warn"
                provider_details += "; provider does not match deployment mode"

            checks.append(
                DoctorCheck(
                    name="model_provider",
                    status=provider_status,
                    details=provider_details,
                    extra=provider_extra,
                )
            )

        if self.safety_service is not None:
            checks.append(
                DoctorCheck(
                    name="safety_policy",
                    status="ok",
                    details="Prompt and execution safety policy is active",
                    extra=self.safety_service.policy_report(),
                )
            )

        for module_name in ["numpy", "pandas", "matplotlib"]:
            ok = importlib.util.find_spec(module_name) is not None
            add_check(
                f"dependency:{module_name}",
                ok,
                f"Python import {'available' if ok else 'missing'} for '{module_name}'",
            )

        reportlab_ok = importlib.util.find_spec("reportlab") is not None
        add_check(
            "dependency_optional:reportlab",
            reportlab_ok,
            (
                "Optional Python import available for 'reportlab'"
                if reportlab_ok
                else "Optional Python import missing for 'reportlab'; PDF generation may be skipped"
            ),
        )

        interaction_summary: dict[str, object] | None = None
        if self.interaction_history_service is not None:
            try:
                interaction_patterns = self.interaction_history_service.summarize_patterns()
                checks.append(
                    DoctorCheck(
                        name="interaction_patterns",
                        status=str(interaction_patterns["status"]),
                        details=(
                            f"{interaction_patterns['follow_up_count']} follow-up prompts, "
                            f"{interaction_patterns['ambiguous_follow_up_count']} ambiguous follow-ups, "
                            f"{interaction_patterns['rewrite_ratio']:.0%} rewritten overall"
                        ),
                        extra={
                            "interaction_count": interaction_patterns["interaction_count"],
                            "resolution_counts": interaction_patterns["resolution_counts"],
                            "follow_up_count": interaction_patterns["follow_up_count"],
                            "ambiguous_follow_up_count": interaction_patterns["ambiguous_follow_up_count"],
                        },
                    )
                )
                interaction_summary = self.interaction_history_service.summarize_interactions()
            except Exception as exc:
                checks.append(
                    DoctorCheck(
                        name="interaction_patterns",
                        status="error",
                        details="Interaction history diagnostics are unavailable, but startup can continue.",
                        extra={
                            "error": str(exc),
                        },
                    )
                )
                interaction_summary = None

        if isinstance(interaction_summary, dict):
            posture_counts = interaction_summary.get("posture_counts", {})
            interaction_count = int(interaction_summary.get("interaction_count", 0))
            recent_posture_mix = str(interaction_summary.get("recent_posture_mix") or "")
            tentative_count = int(posture_counts.get("tentative", 0))
            conflicted_count = int(posture_counts.get("conflicted", 0))

            confidence_status = "ok"
            if interaction_count >= 3:
                if tentative_count / interaction_count >= 0.5:
                    confidence_status = "warn"
                elif recent_posture_mix in {"mixed_with_conflict", "mixed"} and conflicted_count > 0:
                    confidence_status = "warn"

            posture_parts: list[str] = []
            if posture_counts:
                posture_parts = [
                    f"{key}={value}"
                    for key, value in sorted(posture_counts.items())
                ]
            posture_summary = ", ".join(posture_parts) if posture_parts else "no interaction posture data"
            details = posture_summary
            if recent_posture_mix:
                details += f"; recent mix={recent_posture_mix}"

            checks.append(
                DoctorCheck(
                    name="confidence_posture",
                    status=confidence_status,
                    details=details,
                    extra={
                        "interaction_count": interaction_count,
                        "posture_counts": posture_counts,
                        "posture_trend": interaction_summary.get("posture_trend", []),
                        "recent_posture_mix": interaction_summary.get("recent_posture_mix"),
                    },
                )
            )
            detected_language_counts = interaction_summary.get("detected_language_counts", {})
            dominant_intent_counts = interaction_summary.get("dominant_intent_counts", {})
            recent_topics = interaction_summary.get("recent_topics", [])
            unknown_intent_count = int(dominant_intent_counts.get("unknown", 0))
            nlu_status = "ok"
            if interaction_count >= 3 and unknown_intent_count / interaction_count >= 0.34:
                nlu_status = "warn"
            language_parts = [
                f"{key}={value}"
                for key, value in sorted(detected_language_counts.items())
                if value
            ]
            intent_parts = [
                f"{key}={value}"
                for key, value in sorted(dominant_intent_counts.items())
                if value
            ]
            details = (
                f"languages[{', '.join(language_parts) or 'none'}]; "
                f"intents[{', '.join(intent_parts) or 'none'}]"
            )
            if recent_topics:
                details += f"; recent topics={', '.join(recent_topics[:3])}"
            checks.append(
                DoctorCheck(
                    name="nlu_signals",
                    status=nlu_status,
                    details=details,
                    extra={
                        "interaction_count": interaction_count,
                        "detected_language_counts": detected_language_counts,
                        "dominant_intent_counts": dominant_intent_counts,
                        "recent_topics": recent_topics,
                    },
                )
            )
            semantic_route_count = int(interaction_summary.get("semantic_route_count", 0))
            semantic_route_ratio = float(interaction_summary.get("semantic_route_ratio", 0.0))
            route_normalized_score_count = int(interaction_summary.get("route_normalized_score_count", 0))
            route_normalized_score_avg = float(interaction_summary.get("route_normalized_score_avg", 0.0))
            route_normalized_score_max = float(interaction_summary.get("route_normalized_score_max", 0.0))
            semantic_route_status = "ok"
            if interaction_count >= 3 and semantic_route_ratio >= 0.5:
                semantic_route_status = "ok"
            checks.append(
                DoctorCheck(
                    name="route_semantic_signals",
                    status=semantic_route_status,
                    details=(
                        f"{semantic_route_count}/{interaction_count} recent interactions used semantic route reinforcement"
                    ),
                    extra={
                        "interaction_count": interaction_count,
                        "semantic_route_count": semantic_route_count,
                        "semantic_route_ratio": semantic_route_ratio,
                    },
                )
            )
            checks.append(
                DoctorCheck(
                    name="route_normalized_scores",
                    status="ok",
                    details=(
                        f"{route_normalized_score_count} scored routes; avg={route_normalized_score_avg}; max={route_normalized_score_max}"
                    ),
                    extra={
                        "interaction_count": interaction_count,
                        "route_normalized_score_count": route_normalized_score_count,
                        "route_normalized_score_avg": route_normalized_score_avg,
                        "route_normalized_score_max": route_normalized_score_max,
                    },
                )
            )
            route_intent_bias_count = int(interaction_summary.get("route_intent_bias_count", 0))
            route_intent_bias_ratio = float(interaction_summary.get("route_intent_bias_ratio", 0.0))
            route_intent_caution_count = int(interaction_summary.get("route_intent_caution_count", 0))
            route_intent_caution_ratio = float(interaction_summary.get("route_intent_caution_ratio", 0.0))
            route_intent_status = "ok"
            if interaction_count >= 3 and route_intent_caution_ratio >= 0.34:
                route_intent_status = "warn"
            checks.append(
                DoctorCheck(
                    name="route_session_intent",
                    status=route_intent_status,
                    details=(
                        f"{route_intent_bias_count}/{interaction_count} recent interactions reinforced ambiguous routes from session intent; "
                        f"{route_intent_caution_count}/{interaction_count} softened them"
                    ),
                    extra={
                        "interaction_count": interaction_count,
                        "route_intent_bias_count": route_intent_bias_count,
                        "route_intent_bias_ratio": route_intent_bias_ratio,
                        "route_intent_caution_count": route_intent_caution_count,
                        "route_intent_caution_ratio": route_intent_caution_ratio,
                    },
                )
            )
            retrieval_route_caution_count = int(interaction_summary.get("retrieval_route_caution_count", 0))
            retrieval_route_caution_ratio = float(interaction_summary.get("retrieval_route_caution_ratio", 0.0))
            retrieval_route_caution_status = "ok"
            if interaction_count >= 3 and retrieval_route_caution_ratio >= 0.34:
                retrieval_route_caution_status = "warn"
            checks.append(
                DoctorCheck(
                    name="route_retrieval_caution",
                    status=retrieval_route_caution_status,
                    details=(
                        f"{retrieval_route_caution_count}/{interaction_count} recent interactions softened route confidence due to retrieval bias"
                    ),
                    extra={
                        "interaction_count": interaction_count,
                        "retrieval_route_caution_count": retrieval_route_caution_count,
                        "retrieval_route_caution_ratio": retrieval_route_caution_ratio,
                    },
                )
            )
            retrieval_lead_counts = interaction_summary.get("retrieval_lead_counts", {})
            retrieval_observation_count = int(interaction_summary.get("retrieval_observation_count", 0))
            semantic_led = int(retrieval_lead_counts.get("semantic", 0))
            keyword_led = int(retrieval_lead_counts.get("keyword", 0))
            retrieval_status = "ok"
            if retrieval_observation_count >= 3 and semantic_led > keyword_led * 2 and keyword_led == 0:
                retrieval_status = "warn"
            retrieval_parts = [
                f"{key}={value}"
                for key, value in sorted(retrieval_lead_counts.items())
                if value
            ]
            checks.append(
                DoctorCheck(
                    name="retrieval_ranking",
                    status=retrieval_status,
                    details=(
                        ", ".join(retrieval_parts)
                        if retrieval_parts
                        else "no retrieval ranking observations"
                    ),
                    extra={
                        "retrieval_lead_counts": retrieval_lead_counts,
                        "retrieval_observation_count": retrieval_observation_count,
                    },
                )
            )
            tool_route_origin_counts = interaction_summary.get("tool_route_origin_counts", {})
            exact_alias_count = int(tool_route_origin_counts.get("exact_alias", 0))
            nlu_hint_alias_count = int(tool_route_origin_counts.get("nlu_hint_alias", 0))
            tool_route_status = "ok"
            if nlu_hint_alias_count > exact_alias_count and nlu_hint_alias_count > 0:
                tool_route_status = "warn"
            tool_route_parts = [
                f"{key}={value}"
                for key, value in sorted(tool_route_origin_counts.items())
                if value
            ]
            checks.append(
                DoctorCheck(
                    name="tool_route_origins",
                    status=tool_route_status,
                    details=", ".join(tool_route_parts) if tool_route_parts else "no tool-route origin data",
                    extra={
                        "tool_route_origin_counts": tool_route_origin_counts,
                    },
                )
            )
            clarification_count = int(interaction_summary.get("clarification_count", 0))
            clarification_ratio = float(interaction_summary.get("clarification_ratio", 0.0))
            recent_clarification_mix = str(interaction_summary.get("recent_clarification_mix") or "")
            clarification_drift = str(interaction_summary.get("clarification_drift") or "")
            clarification_status = "ok"
            if interaction_count >= 3 and clarification_ratio >= 0.34:
                clarification_status = "warn"
            elif clarification_drift == "increasing" and clarification_count > 0:
                clarification_status = "warn"
            checks.append(
                DoctorCheck(
                    name="route_clarifications",
                    status=clarification_status,
                    details=(
                        f"{clarification_count}/{interaction_count} recent interactions required clarification"
                        + (f"; recent mix={recent_clarification_mix}" if recent_clarification_mix else "")
                    ),
                    extra={
                        "interaction_count": interaction_count,
                        "clarification_count": clarification_count,
                        "clarification_ratio": clarification_ratio,
                        "clarification_trend": interaction_summary.get("clarification_trend", []),
                        "recent_clarification_mix": interaction_summary.get("recent_clarification_mix"),
                        "latest_clarification": interaction_summary.get("latest_clarification"),
                        "clarification_drift": interaction_summary.get("clarification_drift"),
                    },
                )
            )
            memory_classification_counts = interaction_summary.get("memory_classification_counts", {})
            memory_write_action_counts = interaction_summary.get("memory_write_action_counts", {})
            memory_save_eligible_count = int(interaction_summary.get("memory_save_eligible_count", 0))
            explicit_memory_consent_count = int(interaction_summary.get("explicit_memory_consent_count", 0))
            memory_surface_block_count = int(interaction_summary.get("memory_surface_block_count", 0))
            personal_memory_saved_count = int(interaction_summary.get("personal_memory_saved_count", 0))
            research_note_count = int(interaction_summary.get("research_note_count", 0))
            research_artifact_count = int(interaction_summary.get("research_artifact_count", 0))
            research_artifact_type_counts = interaction_summary.get("research_artifact_type_counts", {})
            personal_context_count = int(memory_classification_counts.get("personal_context_candidate", 0))
            memory_status = "ok"
            if personal_context_count > 0 and personal_memory_saved_count == 0:
                memory_status = "warn"
            if personal_memory_saved_count > explicit_memory_consent_count:
                memory_status = "warn"
            checks.append(
                DoctorCheck(
                    name="memory_behavior",
                    status=memory_status,
                    details=(
                        f"research_notes={research_note_count}; "
                        f"research_artifacts={research_artifact_count}; "
                        f"personal_saves={personal_memory_saved_count}; "
                        f"consent_required={explicit_memory_consent_count}; "
                        f"surface_blocked={memory_surface_block_count}"
                    ),
                    extra={
                        "interaction_count": interaction_count,
                        "memory_classification_counts": memory_classification_counts,
                        "memory_write_action_counts": memory_write_action_counts,
                        "memory_save_eligible_count": memory_save_eligible_count,
                        "explicit_memory_consent_count": explicit_memory_consent_count,
                        "memory_surface_block_count": memory_surface_block_count,
                        "personal_memory_saved_count": personal_memory_saved_count,
                        "research_note_count": research_note_count,
                        "research_artifact_count": research_artifact_count,
                        "research_artifact_type_counts": research_artifact_type_counts,
                    },
                )
            )

        if self.archive_manager is not None:
            archive_index = self.archive_manager.index_status()
            checks.append(
                DoctorCheck(
                    name="archive_index",
                    status="ok" if archive_index["legacy_record_count"] == 0 else "warn",
                    details=(
                        f"{archive_index['indexed_record_count']}/{archive_index['record_file_count']} "
                        f"archive records indexed"
                    ),
                    extra=archive_index,
                )
            )

        if self.interaction_history_service is not None:
            interaction_index = self.interaction_history_service.interaction_log_manager.index_status()
            checks.append(
                DoctorCheck(
                    name="interaction_index",
                    status="ok" if interaction_index["legacy_record_count"] == 0 else "warn",
                    details=(
                        f"{interaction_index['indexed_record_count']}/{interaction_index['record_file_count']} "
                        f"interaction records indexed"
                    ),
                    extra=interaction_index,
                )
            )

        human_audit = self.build_human_thinking_layer_report()
        checks.append(
            DoctorCheck(
                name="human_thinking_layer_readiness",
                status=str(human_audit.get("status") or "warn"),
                details=(
                    f"{len(human_audit.get('audit_dimensions', []))} audited dimensions; "
                    f"{len(human_audit.get('confirmed_gap_list', []))} confirmed gaps; "
                    f"{len(human_audit.get('targeted_implementation_list', []))} targeted actions"
                ),
                extra={
                    "note_sources": human_audit.get("note_sources", {}),
                    "action_counts": human_audit.get("action_counts", {}),
                    "confirmed_gap_count": len(human_audit.get("confirmed_gap_list", [])),
                    "targeted_implementation_count": len(human_audit.get("targeted_implementation_list", [])),
                    "backlog_status_counts": (
                        (human_audit.get("backlog_appendix") or {}).get("status_counts", {})
                        if isinstance(human_audit.get("backlog_appendix"), dict)
                        else {}
                    ),
                },
            )
        )
        capability_contracts = self.build_capability_contract_report()
        status_counts = capability_contracts.get("status_counts", {})
        checks.append(
            DoctorCheck(
                name="capability_contracts",
                status="ok",
                details=(
                    f"supported={status_counts.get('supported', 0)}, "
                    f"bounded={status_counts.get('bounded', 0)}, "
                    f"provider_gated={status_counts.get('provider_gated', 0)}, "
                    f"not_promised={status_counts.get('not_promised', 0)}"
                ),
                extra=capability_contracts,
            )
        )
        academic_support = self.build_academic_support_report()
        checks.append(
            DoctorCheck(
                name="academic_support",
                status="ok",
                details=f"{len(list(academic_support.get('workflows') or []))} bounded academic workflows available",
                extra=academic_support,
            )
        )
        overlap_audit = self.build_behavioral_overlap_audit()
        checks.append(
            DoctorCheck(
                name="behavioral_overlap_audit",
                status=str(overlap_audit.get("status") or "ok"),
                details=str(overlap_audit.get("summary") or "Behavioral overlap audit completed."),
                extra=overlap_audit,
            )
        )

        overall = "ok" if all(
            item.status in {"ok", "warn"}
            for item in checks
            if not item.name.startswith("dependency_optional:")
        ) else "error"
        return DoctorReport(
            status=overall,
            repo_root=str(self.repo_root),
            checks=checks,
        ).to_dict()

    def build_human_thinking_layer_report(self) -> dict[str, object]:
        return HumanThinkingAuditService(repo_root=self.repo_root).build_report()

    def build_capability_contract_report(self) -> dict[str, object]:
        return CapabilityContractService.build_report()

    def build_academic_support_report(self) -> dict[str, object]:
        return AcademicSupportService.build_report()

    def build_behavioral_overlap_audit(self) -> dict[str, object]:
        contracts = self.build_capability_contract_report()
        contract_map = {str(item.get("domain_id") or ""): item for item in contracts.get("contracts", [])}
        workflows = self.build_academic_support_report().get("workflows", [])
        workflow_domain_ids = {str(item.get("domain_id") or "") for item in workflows}
        gaps: list[str] = []
        drift_findings: list[dict[str, str]] = []
        if "academic_writing" not in workflow_domain_ids:
            gaps.append("academic_writing workflows missing from academic support report")
        if "citation_support" not in workflow_domain_ids:
            gaps.append("citation_support workflows missing from academic support report")
        if "supervised_ml_data_support" not in workflow_domain_ids:
            gaps.append("supervised_ml_data_support workflows missing from academic support report")

        intentional_overlaps = [
            {
                "pair": "writing_editing <-> academic_writing",
                "status": "expected_split",
                "note": "Hosted live generation remains provider-gated while local academic support stays bounded.",
            },
            {
                "pair": "dataset_analysis <-> supervised_ml_data_support",
                "status": "expected_split",
                "note": "General data tooling remains separate from academic supervised-data readiness guidance.",
            },
            {
                "pair": "paper tooling <-> literature_synthesis",
                "status": "expected_split",
                "note": "Paper tooling and academic synthesis overlap intentionally but serve different user-facing support lanes.",
            },
        ]
        expected_statuses = {
            "writing_editing": "provider_gated",
            "academic_writing": "bounded",
            "dataset_analysis": "bounded",
            "supervised_ml_data_support": "bounded",
            "literature_synthesis": "bounded",
        }
        contract_statuses = {
            key: str(value.get("status") or "")
            for key, value in contract_map.items()
            if key in expected_statuses
        }
        for domain_id, expected_status in expected_statuses.items():
            actual_status = contract_statuses.get(domain_id)
            if actual_status != expected_status:
                drift_findings.append(
                    {
                        "surface": domain_id,
                        "details": f"Expected {expected_status}, found {actual_status or 'missing'}.",
                    }
                )

        transparency_expectations = {
            "dataset_analysis": "Capability status: bounded.",
            "writing_editing": "Capability status: provider-gated.",
            "vision_imaging": "Capability status: not promised.",
        }
        for domain_id, expected_prefix in transparency_expectations.items():
            contract = contract_map.get(domain_id)
            if not isinstance(contract, dict):
                continue
            line = build_capability_transparency_line(
                {
                    "capability_status": {
                        "status": contract.get("status"),
                        "details": contract.get("scope_note"),
                    }
                }
            )
            if not isinstance(line, str) or not line.startswith(expected_prefix):
                drift_findings.append(
                    {
                        "pair": f"desktop transparency <-> {domain_id}",
                        "details": f"Expected transparency prefix {expected_prefix!r}, found {line!r}.",
                    }
                )

        status = "error" if gaps or drift_findings else "ok"
        summary = (
            "No harmful duplicate-path conflicts detected across academic, writing, paper, and dataset support surfaces."
            if not gaps and not drift_findings
            else "Behavioral overlap audit found missing or conflicting support surfaces."
        )
        return {
            "status": status,
            "summary": summary,
            "gaps": gaps,
            "drift_findings": drift_findings,
            "intentional_overlaps": intentional_overlaps,
            "contract_statuses": contract_statuses,
        }
