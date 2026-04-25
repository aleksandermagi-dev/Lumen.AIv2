from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any
from typing import Callable

from lumen.app.command_parser import CommandParser
from lumen.app.models import InteractionProfile
from lumen.app.settings import AppSettings
from lumen.db.persistence_manager import PersistenceManager
from lumen.memory.archive_manager import ArchiveManager
from lumen.memory.graph_memory import GraphMemoryManager
from lumen.memory.interaction_log_manager import InteractionLogManager
from lumen.memory.personal_memory import PersonalMemoryManager
from lumen.memory.research_artifacts import ResearchArtifactManager
from lumen.memory.research_notes import ResearchNoteManager
from lumen.memory.session_state_manager import SessionStateManager
from lumen.knowledge.knowledge_service import KnowledgeService
from lumen.providers.factory import build_model_provider
from lumen.reporting.output_formatter import OutputFormatter
from lumen.reasoning.planner import Planner
from lumen.reasoning.research_engine import ResearchEngine
from lumen.reasoning.memory_retrieval_layer import MemoryRetrievalLayer
from lumen.services.archive_service import ArchiveService
from lumen.services.bundle_service import BundleService
from lumen.services.diagnostics_service import DiagnosticsService
from lumen.services.dataset_ingestion_service import DatasetIngestionService
from lumen.services.dataset_curation_service import DatasetCurationService
from lumen.services.interaction_history_service import InteractionHistoryService
from lumen.services.interaction_service import InteractionService
from lumen.services.inference_service import InferenceService
from lumen.services.session_context_service import SessionContextService
from lumen.services.storage_hygiene_service import StorageHygieneService
from lumen.services.safety_service import SafetyService
from lumen.services.tool_execution_service import ToolExecutionService
from lumen.services.workspace_service import WorkspaceService
from lumen.session_visibility import is_internal_session_id
from lumen.routing.capability_manager import CapabilityManager
from lumen.routing.domain_router import DomainRouter
from lumen.routing.intent_router import IntentRouter
from lumen.routing.prompt_resolution import PromptResolver
from lumen.routing.tool_registry import ToolRegistry
from lumen.tools.registry_types import ToolResult


class AppController:
    """Thin application-layer facade over Lumen's registry-backed tool execution."""

    def __init__(
        self,
        repo_root: Path,
        *,
        data_root: Path | None = None,
        execution_mode: str = "source",
        settings: AppSettings | None = None,
        startup_hook: Callable[[str, str, str | None], None] | None = None,
    ):
        self._startup_hook = startup_hook
        self._emit_startup("settings", "before", "resolve app settings")
        self.settings = settings or AppSettings.from_repo_root(repo_root, data_root_override=data_root)
        self._emit_startup("settings", "after", f"data_root={self.settings.data_root}")
        self.repo_root = self.settings.repo_root
        self.execution_mode = str(execution_mode or "source").strip() or "source"
        self._deferred_legacy_imports_pending = self.execution_mode == "frozen"
        self._initialize_persistence_and_memory()
        self._initialize_runtime_services()
        self._initialize_reasoning_spine()

    def _emit_startup(self, checkpoint_id: str, phase: str, details: str | None = None) -> None:
        if self._startup_hook is None:
            return
        self._startup_hook(checkpoint_id, phase, details)

    def _initialize_persistence_and_memory(self) -> None:
        self._emit_startup("persistence_bootstrap", "before", "bootstrap sqlite persistence")
        self.persistence_manager = PersistenceManager(self.settings)
        self.persistence_manager.bootstrap(run_imports=not self._deferred_legacy_imports_pending)
        self._emit_startup("persistence_bootstrap", "after", "sqlite persistence ready")

        self._emit_startup("memory_managers", "before", "initialize memory managers")
        self.archive_manager = ArchiveManager(settings=self.settings, persistence_manager=self.persistence_manager)
        self.graph_memory_manager = GraphMemoryManager(settings=self.settings)
        self.personal_memory_manager = PersonalMemoryManager(settings=self.settings, persistence_manager=self.persistence_manager)
        self.research_note_manager = ResearchNoteManager(settings=self.settings, persistence_manager=self.persistence_manager)
        self.interaction_log_manager = InteractionLogManager(
            settings=self.settings,
            graph_memory_manager=self.graph_memory_manager,
            persistence_manager=self.persistence_manager,
        )
        self.research_artifact_manager = ResearchArtifactManager(settings=self.settings, persistence_manager=self.persistence_manager)
        self.session_state_manager = SessionStateManager(settings=self.settings, persistence_manager=self.persistence_manager)
        self._emit_startup("memory_managers", "after", "memory managers ready")

    def _initialize_runtime_services(self) -> None:
        self._emit_startup("knowledge_service", "before", "initialize knowledge service")
        self.knowledge_service = KnowledgeService.from_path(self.settings.persistence_db_path)
        self._emit_startup("knowledge_service", "after", "knowledge service ready")

        self._emit_startup("provider_registry", "before", "initialize provider and tool registry")
        self.output_formatter = OutputFormatter()
        self.model_provider = build_model_provider(self.settings)
        self.inference_service = InferenceService(self.model_provider)
        self.registry = ToolRegistry(repo_root=self.repo_root)
        self.registry.discover()
        self._emit_startup("provider_registry", "after", "provider and tool registry ready")

        self._emit_startup("service_wiring", "before", "wire archive, diagnostics, and runtime services")
        self.bundle_service = BundleService(self.registry, self.output_formatter)
        self.archive_service = ArchiveService(
            archive_manager=self.archive_manager,
            formatter=self.output_formatter,
            repo_root=str(self.repo_root),
            settings=self.settings,
        )
        self.interaction_history_service = InteractionHistoryService(
            interaction_log_manager=self.interaction_log_manager,
            settings=self.settings,
        )
        self.session_context_service = SessionContextService(
            session_state_manager=self.session_state_manager
        )
        self.memory_retrieval_layer = MemoryRetrievalLayer(
            interaction_history_service=self.interaction_history_service,
            archive_service=self.archive_service,
            graph_memory_manager=self.graph_memory_manager,
            personal_memory_manager=self.personal_memory_manager,
            research_note_manager=self.research_note_manager,
            persistence_manager=self.persistence_manager,
        )
        self.safety_service = SafetyService(
            settings=self.settings,
            registry=self.registry,
        )
        self.tool_execution_service = ToolExecutionService(
            settings=self.settings,
            registry=self.registry,
            archive_manager=self.archive_manager,
            safety_service=self.safety_service,
        )
        self.workspace_service = WorkspaceService(self.settings)
        self.dataset_ingestion_service = DatasetIngestionService(
            settings=self.settings,
            persistence_manager=self.persistence_manager,
        )
        self.dataset_curation_service = DatasetCurationService(
            settings=self.settings,
            persistence_manager=self.persistence_manager,
        )
        self.storage_hygiene_service = StorageHygieneService(self.settings)
        self.diagnostics_service = DiagnosticsService(
            self.settings,
            self.registry,
            model_provider=self.model_provider,
            archive_manager=self.archive_manager,
            interaction_history_service=self.interaction_history_service,
            safety_service=self.safety_service,
            execution_mode=self.execution_mode,
        )
        self.capability_manager = CapabilityManager(manifests=self.registry.get_manifests())
        self.command_parser = CommandParser(capability_manager=self.capability_manager)
        self.intent_router = IntentRouter(capability_manager=self.capability_manager)
        self.domain_router = DomainRouter(capability_manager=self.capability_manager)
        self.prompt_resolver = PromptResolver(capability_manager=self.capability_manager)
        self._emit_startup("service_wiring", "after", "runtime services ready")

    def _initialize_reasoning_spine(self) -> None:
        self._emit_startup("interaction_service_wiring", "before", "initialize reasoning and interaction services")
        self.interaction_service = InteractionService(
            domain_router=self.domain_router,
            command_parser=self.command_parser,
            intent_router=self.intent_router,
            planner=Planner(),
            research_engine=ResearchEngine(),
            tool_execution_service=self.tool_execution_service,
            archive_service=self.archive_service,
            interaction_history_service=self.interaction_history_service,
            session_context_service=self.session_context_service,
            prompt_resolver=self.prompt_resolver,
            safety_service=self.safety_service,
            inference_service=self.inference_service,
            knowledge_service=self.knowledge_service,
            memory_retrieval_layer=self.memory_retrieval_layer,
        )
        self._emit_startup("interaction_service_wiring", "after", "interaction service ready")

    def run_deferred_startup_tasks(self) -> None:
        if not self._deferred_legacy_imports_pending:
            return
        self._emit_startup("legacy_imports", "before", "run deferred legacy persistence imports")
        self.persistence_manager.run_legacy_imports()
        self._deferred_legacy_imports_pending = False
        self._emit_startup("legacy_imports", "after", "deferred legacy persistence imports complete")

    def list_tools(self) -> dict[str, list[str]]:
        return self.registry.list_tools()

    def inspect_bundle(self, bundle_id: str) -> dict[str, object]:
        return self.bundle_service.inspect_bundle(bundle_id)

    def initialize_workspace(self) -> dict[str, object]:
        return self.workspace_service.initialize_workspace()

    def list_app_capabilities(self) -> dict[str, dict[str, str]]:
        return {
            key: {
                "tool_id": spec.tool_id,
                "tool_capability": spec.tool_capability,
                "description": spec.description,
            }
            for key, spec in self.capability_manager.list_capabilities().items()
        }

    def run_tool(
        self,
        tool_id: str,
        capability: str,
        *,
        input_path: Path | None = None,
        params: dict[str, Any] | None = None,
        session_id: str = "default",
        run_root: Path | None = None,
    ) -> ToolResult:
        return self.tool_execution_service.run_tool(
            tool_id=tool_id,
            capability=capability,
            input_path=input_path,
            params=params or {},
            session_id=session_id,
            run_root=run_root,
        )

    def run_command(
        self,
        *,
        action: str,
        target: str,
        input_path: Path | None = None,
        params: dict[str, Any] | None = None,
        session_id: str = "default",
        run_root: Path | None = None,
    ) -> ToolResult:
        command = self.command_parser.parse(
            action=action,
            target=target,
            input_path=input_path,
            params=params,
            session_id=session_id,
            run_root=run_root,
        )
        routed = self.intent_router.route(command)
        return self.tool_execution_service.run_tool(
            tool_id=routed.tool_id,
            capability=routed.capability,
            input_path=routed.input_path,
            params=routed.params,
            session_id=routed.session_id,
            run_root=routed.run_root,
        )

    def ask(
        self,
        *,
        prompt: str,
        input_path: Path | None = None,
        params: dict[str, Any] | None = None,
        session_id: str = "default",
        run_root: Path | None = None,
        client_surface: str = "main",
        project_id: str | None = None,
        project_name: str | None = None,
    ) -> dict[str, object]:
        response = self.interaction_service.ask(
            prompt=prompt,
            input_path=input_path,
            params=params,
            session_id=session_id,
            run_root=run_root,
            client_surface=client_surface,
            project_id=project_id,
            project_name=project_name,
        )
        if response.get("mode") == "tool" and response.get("tool_result") is not None:
            tool_result = response["tool_result"]
            live_summary = str(response.get("summary") or "").strip()
            payload = {
                "schema_type": response["schema_type"],
                "schema_version": response["schema_version"],
                "mode": response["mode"],
                "kind": response.get("kind"),
                "summary": live_summary or tool_result.summary,
                "route": response.get("route"),
                "tool_execution": response.get("tool_execution"),
                "tool_result": tool_result,
            }
            for key in (
                "user_facing_answer",
                "reply",
                "answer",
                "result_text",
                "result_summary",
                "evidence_summary",
            ):
                value = response.get(key)
                if isinstance(value, str) and value.strip():
                    payload[key] = value
            for key in (
                "resolved_prompt",
                "resolution_strategy",
                "resolution_reason",
                "tool_route_origin",
                "tool_execution_skipped",
                "tool_execution_skipped_reason",
                "tool_missing_inputs",
                "tool_input_hint",
                "tool_runtime_status",
                "runtime_diagnostic",
                "execution_outcome",
            ):
                if key in response:
                    payload[key] = response[key]
            return payload
        return response

    def build_doctor_report(self) -> dict[str, object]:
        return self.diagnostics_service.build_doctor_report()

    def human_thinking_layer_report(self) -> dict[str, object]:
        return self.diagnostics_service.build_human_thinking_layer_report()

    def capability_contract_report(self) -> dict[str, object]:
        return self.diagnostics_service.build_capability_contract_report()

    def academic_support_report(self) -> dict[str, object]:
        return self.diagnostics_service.build_academic_support_report()

    def list_archive_records(
        self,
        *,
        session_id: str | None = None,
        project_id: str | None = None,
        tool_id: str | None = None,
        capability: str | None = None,
        status: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> dict[str, object]:
        return self.archive_service.list_records(
            session_id=session_id,
            project_id=project_id,
            tool_id=tool_id,
            capability=capability,
            status=status,
            date_from=date_from,
            date_to=date_to,
        )

    def inspect_session(self, session_id: str) -> dict[str, object]:
        archive_report = self.archive_service.inspect_session(session_id)
        interaction_report = self.interaction_history_service.inspect_session(session_id)
        return self.output_formatter.session_payload(
            repo_root=str(self.repo_root),
            session_id=session_id,
            records=archive_report["records"],
            interaction_records=interaction_report["records"],
            interaction_profile=self.session_context_service.get_interaction_profile(session_id).to_dict(),
            active_thread=self.session_context_service.get_active_thread(session_id),
        )

    def current_session_thread(self, session_id: str) -> dict[str, object]:
        return self.output_formatter.active_thread_payload(
            repo_root=str(self.repo_root),
            session_id=session_id,
            interaction_profile=self.session_context_service.get_interaction_profile(session_id).to_dict(),
            active_thread=self.session_context_service.get_active_thread(session_id),
        )

    def reset_session_thread(self, session_id: str) -> dict[str, object]:
        result = self.session_context_service.clear_active_thread(session_id)
        return self.output_formatter.session_reset_payload(
            repo_root=str(self.repo_root),
            session_id=session_id,
            cleared=result["cleared"],
            state_path=result["state_path"],
            interaction_profile=self.session_context_service.get_interaction_profile(session_id).to_dict(),
            active_thread=None,
        )

    def get_session_profile(self, session_id: str) -> dict[str, object]:
        return self.output_formatter.session_profile_payload(
            repo_root=str(self.repo_root),
            session_id=session_id,
            interaction_profile=self.session_context_service.get_interaction_profile(session_id).to_dict(),
        )

    def set_session_profile(
        self,
        session_id: str,
        *,
        interaction_style: str | None = None,
        reasoning_depth: str | None = None,
        allow_suggestions: bool | None = None,
    ) -> dict[str, object]:
        current = self.session_context_service.get_interaction_profile(session_id)
        normalized_style = interaction_style or current.interaction_style
        profile = InteractionProfile(
            interaction_style=normalized_style,
            reasoning_depth=reasoning_depth or current.reasoning_depth,
            selection_source="user",
            confidence=None,
            allow_suggestions=(
                allow_suggestions if allow_suggestions is not None else current.allow_suggestions
            ),
        )
        self.session_context_service.set_interaction_profile(session_id, profile)
        return self.get_session_profile(session_id)

    def list_interactions(
        self,
        *,
        session_id: str | None = None,
        project_id: str | None = None,
        resolution_strategy: str | None = None,
    ) -> dict[str, object]:
        return self.interaction_history_service.list_records(
            session_id=session_id,
            project_id=project_id,
            resolution_strategy=resolution_strategy,
        )

    def search_interactions(
        self,
        query: str,
        *,
        session_id: str | None = None,
        project_id: str | None = None,
        resolution_strategy: str | None = None,
        limit: int | None = None,
    ) -> dict[str, object]:
        return self.interaction_history_service.search_interactions(
            query,
            session_id=session_id,
            project_id=project_id,
            resolution_strategy=resolution_strategy,
            limit=limit,
        )

    def summarize_interactions(
        self,
        *,
        session_id: str | None = None,
        project_id: str | None = None,
    ) -> dict[str, object]:
        return self.interaction_history_service.summarize_interactions(session_id=session_id, project_id=project_id)

    def evaluate_interactions(
        self,
        *,
        session_id: str | None = None,
        project_id: str | None = None,
    ) -> dict[str, object]:
        return self.interaction_history_service.evaluate_interactions(session_id=session_id, project_id=project_id)

    def export_labeled_examples(
        self,
        *,
        session_id: str | None = None,
        project_id: str | None = None,
    ) -> dict[str, object]:
        return self.interaction_history_service.export_labeled_examples(session_id=session_id, project_id=project_id)

    def interaction_patterns(
        self,
        *,
        session_id: str | None = None,
        project_id: str | None = None,
    ) -> dict[str, object]:
        return self.interaction_history_service.summarize_patterns(session_id=session_id, project_id=project_id)

    def create_dataset_import_run(
        self,
        *,
        dataset_name: str,
        source_format: str,
        dataset_kind: str,
        import_strategy: str,
        dataset_version: str | None = None,
        source_path: str | None = None,
        ingestion_status: str = "staged",
        notes: dict[str, object] | None = None,
    ) -> dict[str, object]:
        return self.dataset_ingestion_service.create_import_run(
            dataset_name=dataset_name,
            source_format=source_format,
            dataset_kind=dataset_kind,
            import_strategy=import_strategy,
            dataset_version=dataset_version,
            source_path=source_path,
            ingestion_status=ingestion_status,
            notes=notes,
        )

    def import_dataset(
        self,
        *,
        dataset_name: str,
        source_format: str,
        dataset_kind: str,
        source_path: Path,
        dataset_version: str | None = None,
        import_strategy: str = "external_file",
        csv_mapping: dict[str, str] | None = None,
        notes: dict[str, object] | None = None,
    ) -> dict[str, object]:
        return self.dataset_ingestion_service.import_dataset(
            dataset_name=dataset_name,
            source_format=source_format,
            dataset_kind=dataset_kind,
            source_path=source_path,
            dataset_version=dataset_version,
            import_strategy=import_strategy,
            csv_mapping=csv_mapping,
            notes=notes,
        )

    def import_runtime_dataset_examples(
        self,
        *,
        dataset_name: str,
        import_strategy: str,
        session_id: str | None = None,
        project_id: str | None = None,
        limit: int | None = None,
    ) -> dict[str, object]:
        return self.dataset_ingestion_service.import_runtime_examples(
            dataset_name=dataset_name,
            import_strategy=import_strategy,
            session_id=session_id,
            project_id=project_id,
            limit=limit,
        )

    def list_dataset_import_runs(self, *, dataset_name: str | None = None, limit: int = 100) -> list[dict[str, object]]:
        return self.dataset_ingestion_service.list_dataset_import_runs(dataset_name=dataset_name, limit=limit)

    def list_dataset_examples(
        self,
        *,
        import_run_id: str | None = None,
        example_type: str | None = None,
        split_assignment: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, object]]:
        return self.dataset_ingestion_service.list_dataset_examples(
            import_run_id=import_run_id,
            example_type=example_type,
            split_assignment=split_assignment,
            limit=limit,
        )

    def list_dataset_labels(self, *, dataset_example_id: str | None = None, limit: int = 200) -> list[dict[str, object]]:
        return self.dataset_ingestion_service.list_dataset_labels(dataset_example_id=dataset_example_id, limit=limit)

    def sample_dataset_review(
        self,
        *,
        dataset_name: str | None = None,
        import_run_id: str | None = None,
        example_type: str | None = None,
        limit: int = 50,
        prioritize: str = "programmatic_first",
        output_path: Path | None = None,
    ) -> dict[str, object]:
        return self.dataset_curation_service.sample_review_batch(
            dataset_name=dataset_name,
            import_run_id=import_run_id,
            example_type=example_type,
            limit=limit,
            prioritize=prioritize,
            output_path=output_path,
        )

    def update_dataset_example(
        self,
        *,
        example_id: str,
        trainable: bool | None = None,
        ingestion_state: str | None = None,
        split_assignment: str | None = None,
        label_source: str | None = None,
        review_note: str | None = None,
    ) -> dict[str, object]:
        return self.dataset_curation_service.update_dataset_example(
            example_id=example_id,
            trainable=trainable,
            ingestion_state=ingestion_state,
            split_assignment=split_assignment,
            label_source=label_source,
            review_note=review_note,
        )

    def label_dataset_example(
        self,
        *,
        dataset_example_id: str,
        label_role: str,
        label_value: str,
        label_category: str | None = None,
        reviewer: str | None = None,
        reason: str | None = None,
        is_canonical: bool = False,
    ) -> dict[str, object]:
        return self.dataset_curation_service.label_dataset_example(
            dataset_example_id=dataset_example_id,
            label_role=label_role,
            label_value=label_value,
            label_category=label_category,
            reviewer=reviewer,
            reason=reason,
            is_canonical=is_canonical,
        )

    def export_dataset_jsonl(
        self,
        *,
        dataset_name: str,
        import_run_ids: list[str] | None = None,
        split_assignments: list[str] | None = None,
        example_types: list[str] | None = None,
        label_sources: list[str] | None = None,
        canonical_only: bool = False,
        trainable_only: bool = True,
        evaluation_only: bool = False,
        export_name: str | None = None,
        output_root: Path | None = None,
    ) -> dict[str, object]:
        return self.dataset_curation_service.export_dataset_jsonl(
            dataset_name=dataset_name,
            import_run_ids=import_run_ids,
            split_assignments=split_assignments,
            example_types=example_types,
            label_sources=label_sources,
            canonical_only=canonical_only,
            trainable_only=trainable_only,
            evaluation_only=evaluation_only,
            export_name=export_name,
            output_root=output_root,
        )

    def compare_dataset_runs(self, *, left_import_run_id: str, right_import_run_id: str) -> dict[str, object]:
        return self.dataset_curation_service.compare_dataset_runs(
            left_import_run_id=left_import_run_id,
            right_import_run_id=right_import_run_id,
        )

    def search_archive_records(
        self,
        query: str,
        *,
        session_id: str | None = None,
        project_id: str | None = None,
        tool_id: str | None = None,
        capability: str | None = None,
        status: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> dict[str, object]:
        return self.archive_service.search_records(
            query,
            session_id=session_id,
            project_id=project_id,
            tool_id=tool_id,
            capability=capability,
            status=status,
            date_from=date_from,
            date_to=date_to,
        )

    def latest_archive_record(
        self,
        *,
        session_id: str | None = None,
        project_id: str | None = None,
        tool_id: str | None = None,
        capability: str | None = None,
        status: str | None = "ok",
    ) -> dict[str, object]:
        return self.archive_service.latest_record(
            session_id=session_id,
            project_id=project_id,
            tool_id=tool_id,
            capability=capability,
            status=status,
        )

    def archive_summary(
        self,
        *,
        session_id: str | None = None,
        project_id: str | None = None,
        tool_id: str | None = None,
        capability: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> dict[str, object]:
        return self.archive_service.summary(
            session_id=session_id,
            project_id=project_id,
            tool_id=tool_id,
            capability=capability,
            date_from=date_from,
            date_to=date_to,
        )

    def compare_archive_runs(
        self,
        *,
        capability: str,
        session_id: str | None = None,
        project_id: str | None = None,
        tool_id: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> dict[str, object]:
        return self.archive_service.compare_runs_by_target(
            capability=capability,
            session_id=session_id,
            project_id=project_id,
            tool_id=tool_id,
            date_from=date_from,
            date_to=date_to,
        )

    def list_research_notes(
        self,
        *,
        session_id: str | None = None,
        project_id: str | None = None,
        include_archived: bool = False,
        archived_only: bool = False,
        limit: int | None = None,
    ) -> dict[str, object]:
        notes = self.research_artifact_manager.list_notes(
            session_id=session_id,
            project_id=project_id,
            include_archived=include_archived,
            archived_only=archived_only,
            limit=limit,
        )
        return {
            "repo_root": str(self.repo_root),
            "session_id": session_id,
            "note_count": len(notes),
            "research_notes": notes,
        }

    def list_research_artifacts(
        self,
        *,
        session_id: str | None = None,
        project_id: str | None = None,
    ) -> dict[str, object]:
        artifacts = self.research_artifact_manager.list_artifacts(session_id=session_id, project_id=project_id)
        return {
            "repo_root": str(self.repo_root),
            "session_id": session_id,
            "artifact_count": len(artifacts),
            "research_artifacts": artifacts,
        }

    def list_personal_memory(
        self,
        *,
        session_id: str | None = None,
        project_id: str | None = None,
        include_archived: bool = False,
        archived_only: bool = False,
        limit: int | None = None,
    ) -> dict[str, object]:
        entries = self.interaction_log_manager.personal_memory_manager.list_entries(
            session_id=session_id,
            project_id=project_id,
            include_archived=include_archived,
            archived_only=archived_only,
            limit=limit,
        )
        return {
            "repo_root": str(self.repo_root),
            "session_id": session_id,
            "entry_count": len(entries),
            "personal_memory": entries,
        }

    def list_archived_memory(self, *, session_id: str | None = None) -> dict[str, object]:
        personal = self.list_personal_memory(
            session_id=session_id,
            include_archived=True,
            archived_only=True,
        ).get("personal_memory", [])
        notes = self.list_research_notes(
            session_id=session_id,
            include_archived=True,
            archived_only=True,
        ).get("research_notes", [])
        entries: list[dict[str, object]] = []
        for entry in personal:
            if isinstance(entry, dict):
                entries.append(
                    {
                        "title": entry.get("title"),
                        "content": entry.get("content", ""),
                        "created_at": entry.get("created_at"),
                        "kind": "personal_memory",
                        "entry_path": entry.get("entry_path"),
                    }
                )
        for note in notes:
            if isinstance(note, dict):
                entries.append(
                    {
                        "title": note.get("title"),
                        "content": note.get("content", ""),
                        "created_at": note.get("created_at"),
                        "kind": "research_note",
                        "note_path": note.get("note_path"),
                    }
                )
        entries.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        return {
            "repo_root": str(self.repo_root),
            "session_id": session_id,
            "entry_count": len(entries),
            "archived_memory": entries,
        }

    def list_memory_topics(
        self,
        *,
        session_id: str | None = None,
    ) -> dict[str, object]:
        personal = self.list_personal_memory(session_id=session_id).get("personal_memory", [])
        notes = self.list_research_notes(session_id=session_id).get("research_notes", [])
        grouped: dict[str, list[dict[str, object]]] = {}
        for entry in personal:
            if not isinstance(entry, dict):
                continue
            topic = str(entry.get("normalized_topic") or "").strip() or "general"
            kind = (
                "assistant_memory"
                if str(entry.get("memory_origin") or "").strip() == "assistant"
                else "personal_memory"
            )
            grouped.setdefault(topic, []).append(
                {
                    "title": str(entry.get("title") or "").strip() or topic,
                    "content": str(entry.get("content") or "").strip(),
                    "created_at": entry.get("created_at"),
                    "kind": kind,
                    "normalized_topic": topic,
                }
            )
        for entry in notes:
            if not isinstance(entry, dict):
                continue
            topic = str(entry.get("normalized_topic") or "").strip() or "general"
            grouped.setdefault(topic, []).append(
                {
                    "title": str(entry.get("title") or "").strip() or topic,
                    "content": str(entry.get("content") or "").strip(),
                    "created_at": entry.get("created_at"),
                    "kind": "research_note",
                    "normalized_topic": topic,
                }
            )
        topics = [
            {
                "topic": topic,
                "count": len(items),
                "latest_at": max((str(item.get("created_at") or "") for item in items), default=""),
                "entries": sorted(items, key=lambda item: str(item.get("created_at") or ""), reverse=True),
            }
            for topic, items in grouped.items()
        ]
        topics.sort(key=lambda item: (str(item.get("latest_at") or ""), str(item.get("topic") or "")), reverse=True)
        return {
            "repo_root": str(self.repo_root),
            "session_id": session_id,
            "topic_count": len(topics),
            "topics": topics,
        }

    def list_recent_sessions(
        self,
        *,
        limit: int = 10,
        project_id: str | None = None,
        include_archived: bool = False,
        archived_only: bool = False,
    ) -> dict[str, object]:
        session_limit = max(int(limit), 1)
        sessions = []
        try:
            self.persistence_manager.bootstrap()
            session_rows = self.persistence_manager.sessions.list_recent(
                limit=max(session_limit * 8, session_limit),
                include_archived=include_archived,
                archived_only=archived_only,
                project_id=project_id,
            )
        except Exception:
            session_rows = []
        if session_rows:
            for row in session_rows:
                session_id = str(row.get("id") or "").strip()
                if (
                    not session_id
                    or is_internal_session_id(session_id)
                    or not self._recent_session_has_restorable_history(session_id)
                ):
                    continue
                sessions.append(self._recent_session_payload_from_db_row(row))
                if len(sessions) >= session_limit:
                    break
        else:
            interactions_root = self.interaction_log_manager.interactions_root
            if interactions_root.exists():
                candidates: list[tuple[str, str, dict[str, object]]] = []
                for session_dir in interactions_root.iterdir():
                    if not session_dir.is_dir():
                        continue
                    latest_record = self._latest_interaction_record(session_dir)
                    if latest_record is None:
                        continue
                    candidates.append(
                        (
                            str(latest_record.get("created_at") or ""),
                            session_dir.name,
                            latest_record,
                        )
                    )
                candidates.sort(reverse=True)
                for _, session_id, record in candidates:
                    if is_internal_session_id(session_id):
                        continue
                    metadata = self.session_context_service.get_session_metadata(session_id)
                    is_archived = bool(metadata.get("archived", False))
                    if archived_only and not is_archived:
                        continue
                    if not include_archived and is_archived:
                        continue
                    sessions.append(
                        {
                            "session_id": session_id,
                            "title": metadata.get("title"),
                            "summary": str(record.get("summary") or "").strip(),
                            "prompt": str(record.get("prompt") or "").strip(),
                            "mode": str(record.get("mode") or "").strip(),
                            "kind": str(record.get("kind") or "").strip(),
                            "created_at": str(record.get("created_at") or "").strip(),
                        }
                    )
                    if len(sessions) >= session_limit:
                        break
        return {
            "repo_root": str(self.repo_root),
            "session_count": len(sessions),
            "sessions": sessions,
        }

    def _recent_session_payload_from_db_row(self, row: dict[str, object]) -> dict[str, object]:
        prompt = str(row.get("latest_user_message_content") or "").strip()
        assistant_summary = str(row.get("summary_text") or row.get("latest_message_content") or "").strip()
        summary = "" if self._looks_like_internal_scaffold(assistant_summary) else assistant_summary
        if not prompt:
            latest_record = self._latest_interaction_record(self.interaction_log_manager.interactions_root / str(row.get("id") or ""))
            if isinstance(latest_record, dict):
                prompt = str(latest_record.get("prompt") or "").strip()
                record_summary = str(latest_record.get("summary") or "").strip()
                if not summary and not self._looks_like_internal_scaffold(record_summary):
                    summary = record_summary
        title = str(row.get("title") or "").strip()
        if self._looks_like_internal_scaffold(title):
            title = prompt
        return {
            "session_id": row.get("id"),
            "title": title or row.get("title"),
            "summary": summary,
            "prompt": prompt,
            "mode": str(row.get("mode") or "").strip(),
            "kind": None,
            "created_at": str(row.get("latest_user_message_at") or row.get("latest_message_at") or row.get("updated_at") or "").strip(),
            "project_id": row.get("project_id"),
        }

    @staticmethod
    def _looks_like_internal_scaffold(text: str) -> bool:
        normalized = " ".join(str(text or "").replace("’", "'").strip().lower().split())
        if not normalized:
            return False
        fragments = (
            "best first read",
            "best current assumptions",
            "hold it provisionally",
            "best next check",
            "route caution:",
            "validation plan:",
            "validate the route choice",
            "use the strongest local context",
        )
        return any(fragment in normalized for fragment in fragments)

    def _recent_session_has_restorable_history(self, session_id: str) -> bool:
        try:
            records = self.interaction_log_manager.list_records(session_id=session_id)
        except Exception:
            return False
        return any(self._interaction_record_has_restorable_content(record) for record in records if isinstance(record, dict))

    @staticmethod
    def _interaction_record_has_restorable_content(record: dict[str, object]) -> bool:
        prompt = str(record.get("prompt") or "").strip()
        prompt_view = record.get("prompt_view")
        if not prompt and isinstance(prompt_view, dict):
            prompt = (
                str(prompt_view.get("canonical_prompt") or "").strip()
                or str(prompt_view.get("original_prompt") or "").strip()
                or str(prompt_view.get("resolved_prompt") or "").strip()
            )
        response = record.get("response")
        assistant_text = ""
        if isinstance(response, dict):
            assistant_text = (
                str(response.get("user_facing_answer") or "").strip()
                or str(response.get("reply") or "").strip()
                or str(response.get("summary") or "").strip()
            )
        if not assistant_text:
            assistant_text = (
                str(record.get("user_facing_answer") or "").strip()
                or str(record.get("reply") or "").strip()
                or str(record.get("summary") or "").strip()
            )
        return bool(prompt or assistant_text)

    def persistence_status(self) -> dict[str, object]:
        return self.persistence_manager.status_report()

    def persistence_coverage(self) -> dict[str, object]:
        return self.persistence_manager.coverage_report()

    def persistence_doctor(self) -> dict[str, object]:
        return self.persistence_manager.doctor_report()

    def semantic_status(self) -> dict[str, object]:
        return self.persistence_manager.semantic_status_report()

    def backfill_memory_item_embeddings(self, *, limit: int | None = None) -> dict[str, object]:
        return self.persistence_manager.backfill_memory_item_embeddings(limit=limit)

    def rename_session(self, session_id: str, *, title: str | None) -> dict[str, object]:
        return self.session_context_service.set_session_title(session_id, title)

    def archive_session(self, session_id: str) -> dict[str, object]:
        return self.session_context_service.set_session_archived(session_id, True)

    def delete_session(self, session_id: str) -> dict[str, object]:
        paths = [
            self.interaction_log_manager.interactions_root / session_id,
            self.settings.personal_memory_root / session_id,
            self.settings.research_notes_root / session_id,
            self.settings.research_artifacts_root / session_id,
            self.settings.archive_root / session_id,
            self.settings.tool_runs_root / session_id,
        ]
        for path in paths:
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)
        profile_report = self.session_context_service.delete_session(session_id)
        return {
            "session_id": session_id,
            "deleted": True,
            "session_state": profile_report,
        }

    def archive_memory(self, *, kind: str, path: str) -> dict[str, object]:
        normalized_kind = str(kind or "").strip().lower()
        if normalized_kind == "research_note":
            return self.research_artifact_manager.archive_note(note_path=path)
        return self.interaction_log_manager.personal_memory_manager.archive_entry(entry_path=path)

    def delete_memory(self, *, kind: str, path: str) -> dict[str, object]:
        normalized_kind = str(kind or "").strip().lower()
        if normalized_kind == "research_note":
            return self.research_artifact_manager.delete_note(note_path=path)
        return self.interaction_log_manager.personal_memory_manager.delete_entry(entry_path=path)

    def _latest_interaction_record(self, session_dir: Path) -> dict[str, object] | None:
        for candidate in sorted(session_dir.glob("*.json"), reverse=True):
            if candidate.name == "index.json":
                continue
            try:
                record = self.interaction_log_manager.load_record(candidate)
            except (OSError, ValueError, TypeError):
                continue
            if isinstance(record, dict):
                return record
        return None

    def knowledge_overview(self) -> dict[str, object]:
        overview = self.knowledge_service.overview()
        return {
            "repo_root": str(self.repo_root),
            **overview,
        }

    def storage_hygiene_report(self) -> dict[str, object]:
        return self.storage_hygiene_service.report()

    def cleanup_storage(
        self,
        *,
        prune_oversized: bool = True,
        prune_tool_runs: bool = True,
        retain_per_capability: int | None = None,
    ) -> dict[str, object]:
        return self.storage_hygiene_service.cleanup(
            prune_oversized=prune_oversized,
            prune_tool_runs=prune_tool_runs,
            retain_per_capability=retain_per_capability,
        )

    def promote_research_note(
        self,
        *,
        note_path: Path,
        artifact_type: str,
        title: str | None = None,
        promotion_reason: str | None = None,
    ) -> dict[str, object]:
        artifact = self.research_artifact_manager.promote_note(
            note_path=note_path,
            artifact_type=artifact_type,
            title=title,
            promotion_reason=promotion_reason,
        )
        self.graph_memory_manager.ingest_research_artifact(artifact)
        return {
            "status": "ok",
            "repo_root": str(self.repo_root),
            "artifact": artifact,
        }

    def read_memory_graph(self, *, limit: int = 50) -> dict[str, object]:
        return self.graph_memory_manager.read_graph(limit=limit)

    def search_memory_graph_nodes(self, query: str, *, limit: int = 5) -> list[dict[str, object]]:
        return self.graph_memory_manager.search_nodes(query, limit=limit)

    def open_memory_graph_nodes(
        self,
        *,
        names: list[str] | None = None,
        ids: list[int] | None = None,
    ) -> list[dict[str, object]]:
        return self.graph_memory_manager.open_nodes(names=names, ids=ids)
