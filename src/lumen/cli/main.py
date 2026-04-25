from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from lumen.app.controller import AppController
from lumen.routing.tool_registry import ToolRegistry


def configure_console_output() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name)
        encoding = getattr(stream, "encoding", None)
        reconfigure = getattr(stream, "reconfigure", None)
        if encoding is None or reconfigure is None:
            continue
        if encoding.lower() != "utf-8":
            reconfigure(encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lumen")
    parser.add_argument("--repo-root", default=Path.cwd(), type=Path)
    parser.add_argument("--format", choices=["json", "text"], default=None)
    subparsers = parser.add_subparsers(dest="command")

    list_parser = subparsers.add_parser("list-tools", help="List registered tool bundles")
    list_parser.set_defaults(command="list-tools")

    capabilities_parser = subparsers.add_parser("list-capabilities", help="List app-level capabilities")
    capabilities_parser.set_defaults(command="list-capabilities")

    bundle_parser = subparsers.add_parser("bundle", help="Inspect bundle manifests")
    bundle_subparsers = bundle_parser.add_subparsers(dest="bundle_command")

    bundle_inspect_parser = bundle_subparsers.add_parser("inspect", help="Inspect a bundle manifest")
    bundle_inspect_parser.add_argument("bundle_id")
    bundle_inspect_parser.set_defaults(command="bundle", bundle_command="inspect")

    init_parser = subparsers.add_parser("init", help="Initialize local Lumen workspace scaffolding")
    init_parser.set_defaults(command="init")

    doctor_parser = subparsers.add_parser("doctor", help="Check local tool and runtime readiness")
    doctor_parser.set_defaults(command="doctor")

    ask_parser = subparsers.add_parser("ask", help="Route a free-form prompt through Lumen")
    ask_parser.add_argument("prompt")
    ask_parser.add_argument("--csv", dest="csv_path", type=Path, help="Optional dataset or directory input path")
    ask_parser.add_argument("--session-id", default="default")
    ask_parser.add_argument("--surface", choices=["main", "mobile"], default="main")
    ask_parser.add_argument("--run-root", type=Path, help="Optional override for tool run output root")
    ask_parser.add_argument("--h0", type=float, help="Optional Hubble constant override for GA flow")
    ask_parser.add_argument("--boot", type=int, help="Optional bootstrap iteration count")
    ask_parser.set_defaults(command="ask")

    repl_parser = subparsers.add_parser("repl", help="Start an interactive Lumen prompt loop")
    repl_parser.add_argument("--csv", dest="csv_path", type=Path, help="Optional default dataset or directory input path")
    repl_parser.add_argument("--session-id", default="default")
    repl_parser.add_argument("--run-root", type=Path, help="Optional override for tool run output root")
    repl_parser.add_argument("--h0", type=float, help="Optional Hubble constant override for GA flow")
    repl_parser.add_argument("--boot", type=int, help="Optional bootstrap iteration count")
    repl_parser.set_defaults(command="repl")

    command_parser = subparsers.add_parser("command", help="Execute a high-level Lumen command")
    command_parser.add_argument("action", help="High-level action, for example 'analyze'")
    command_parser.add_argument("target", help="High-level target, for example 'ga'")
    command_parser.add_argument("--csv", dest="csv_path", type=Path, help="Dataset or directory input path")
    command_parser.add_argument("--session-id", default="default")
    command_parser.add_argument("--run-root", type=Path, help="Optional override for tool run output root")
    command_parser.add_argument("--h0", type=float, help="Optional Hubble constant override for GA flow")
    command_parser.add_argument("--boot", type=int, help="Optional bootstrap iteration count")
    command_parser.set_defaults(command="command")

    archive_parser = subparsers.add_parser("archive", help="Inspect archived tool runs")
    archive_subparsers = archive_parser.add_subparsers(dest="archive_command")

    archive_list_parser = archive_subparsers.add_parser("list", help="List archived run records")
    archive_list_parser.add_argument("--session-id")
    archive_list_parser.add_argument("--tool-id")
    archive_list_parser.add_argument("--capability")
    archive_list_parser.add_argument("--status")
    archive_list_parser.add_argument("--date-from")
    archive_list_parser.add_argument("--date-to")
    archive_list_parser.set_defaults(command="archive", archive_command="list")

    archive_search_parser = archive_subparsers.add_parser("search", help="Search archived run records")
    archive_search_parser.add_argument("query")
    archive_search_parser.add_argument("--session-id")
    archive_search_parser.add_argument("--tool-id")
    archive_search_parser.add_argument("--capability")
    archive_search_parser.add_argument("--status")
    archive_search_parser.add_argument("--date-from")
    archive_search_parser.add_argument("--date-to")
    archive_search_parser.set_defaults(command="archive", archive_command="search")

    archive_latest_parser = archive_subparsers.add_parser("latest", help="Get the latest archived run record")
    archive_latest_parser.add_argument("--session-id")
    archive_latest_parser.add_argument("--tool-id")
    archive_latest_parser.add_argument("--capability")
    archive_latest_parser.add_argument("--status", default="ok")
    archive_latest_parser.set_defaults(command="archive", archive_command="latest")

    archive_summary_parser = archive_subparsers.add_parser("summary", help="Summarize archived run records")
    archive_summary_parser.add_argument("--session-id")
    archive_summary_parser.add_argument("--tool-id")
    archive_summary_parser.add_argument("--capability")
    archive_summary_parser.add_argument("--date-from")
    archive_summary_parser.add_argument("--date-to")
    archive_summary_parser.set_defaults(command="archive", archive_command="summary")

    archive_compare_parser = archive_subparsers.add_parser(
        "compare",
        help="Compare archived research runs by target within one capability",
    )
    archive_compare_parser.add_argument("--session-id")
    archive_compare_parser.add_argument("--tool-id")
    archive_compare_parser.add_argument("--capability", required=True)
    archive_compare_parser.add_argument("--date-from")
    archive_compare_parser.add_argument("--date-to")
    archive_compare_parser.set_defaults(command="archive", archive_command="compare")

    interaction_parser = subparsers.add_parser("interaction", help="Inspect persisted assistant interactions")
    interaction_subparsers = interaction_parser.add_subparsers(dest="interaction_command")

    interaction_list_parser = interaction_subparsers.add_parser("list", help="List assistant interactions")
    interaction_list_parser.add_argument("--session-id")
    interaction_list_parser.add_argument("--project-id")
    interaction_list_parser.add_argument("--resolution-strategy")
    interaction_list_parser.set_defaults(command="interaction", interaction_command="list")

    interaction_search_parser = interaction_subparsers.add_parser("search", help="Search assistant interactions")
    interaction_search_parser.add_argument("query")
    interaction_search_parser.add_argument("--session-id")
    interaction_search_parser.add_argument("--project-id")
    interaction_search_parser.add_argument("--resolution-strategy")
    interaction_search_parser.add_argument("--limit", type=int)
    interaction_search_parser.set_defaults(command="interaction", interaction_command="search")

    interaction_summary_parser = interaction_subparsers.add_parser("summary", help="Summarize assistant interactions")
    interaction_summary_parser.add_argument("--session-id")
    interaction_summary_parser.add_argument("--project-id")
    interaction_summary_parser.set_defaults(command="interaction", interaction_command="summary")

    interaction_evaluate_parser = interaction_subparsers.add_parser(
        "evaluate",
        help="Run an offline evaluation pass over persisted assistant interactions",
    )
    interaction_evaluate_parser.add_argument("--session-id")
    interaction_evaluate_parser.add_argument("--project-id")
    interaction_evaluate_parser.set_defaults(command="interaction", interaction_command="evaluate")

    interaction_export_labels_parser = interaction_subparsers.add_parser(
        "export-labels",
        help="Export evaluated interaction traces as labeled local dataset examples",
    )
    interaction_export_labels_parser.add_argument("--session-id")
    interaction_export_labels_parser.add_argument("--project-id")
    interaction_export_labels_parser.set_defaults(command="interaction", interaction_command="export-labels")

    interaction_patterns_parser = interaction_subparsers.add_parser(
        "patterns",
        help="Inspect shorthand, follow-up, and ambiguity patterns in assistant interactions",
    )
    interaction_patterns_parser.add_argument("--session-id")
    interaction_patterns_parser.add_argument("--project-id")
    interaction_patterns_parser.set_defaults(command="interaction", interaction_command="patterns")

    persistence_parser = subparsers.add_parser("persistence", help="Inspect SQLite persistence health and coverage")
    persistence_subparsers = persistence_parser.add_subparsers(dest="persistence_command")

    persistence_status_parser = persistence_subparsers.add_parser("status", help="Show DB path, migrations, and row counts")
    persistence_status_parser.set_defaults(command="persistence", persistence_command="status")

    persistence_coverage_parser = persistence_subparsers.add_parser("coverage", help="Compare DB row counts with legacy file roots")
    persistence_coverage_parser.set_defaults(command="persistence", persistence_command="coverage")

    persistence_doctor_parser = persistence_subparsers.add_parser("doctor", help="Inspect missing rows and orphaned references")
    persistence_doctor_parser.set_defaults(command="persistence", persistence_command="doctor")

    persistence_semantic_parser = persistence_subparsers.add_parser(
        "semantic-status",
        help="Inspect memory embedding coverage and semantic runtime availability",
    )
    persistence_semantic_parser.set_defaults(command="persistence", persistence_command="semantic-status")

    persistence_backfill_parser = persistence_subparsers.add_parser(
        "backfill-embeddings",
        help="Generate or refresh memory-item embeddings",
    )
    persistence_backfill_parser.add_argument("--limit", type=int)
    persistence_backfill_parser.set_defaults(command="persistence", persistence_command="backfill-embeddings")

    dataset_parser = subparsers.add_parser("dataset", help="Curate, export, and compare supervised dataset rows")
    dataset_subparsers = dataset_parser.add_subparsers(dest="dataset_command")

    dataset_derive_parser = dataset_subparsers.add_parser(
        "derive-runtime-dataset",
        help="Derive a dataset import run from runtime SQLite interactions",
    )
    dataset_derive_parser.add_argument("dataset_name")
    dataset_derive_parser.add_argument(
        "--strategy",
        required=True,
        choices=["derived_trainability", "derived_instruction_response", "derived_tool_runs"],
    )
    dataset_derive_parser.add_argument("--session-id")
    dataset_derive_parser.add_argument("--project-id")
    dataset_derive_parser.add_argument("--limit", type=int)
    dataset_derive_parser.set_defaults(command="dataset", dataset_command="derive-runtime-dataset")

    dataset_import_parser = dataset_subparsers.add_parser(
        "import-dataset",
        help="Import an external JSON, JSONL, or CSV dataset into SQLite",
    )
    dataset_import_parser.add_argument("dataset_name")
    dataset_import_parser.add_argument("--source-format", required=True, choices=["json", "jsonl", "csv"])
    dataset_import_parser.add_argument(
        "--dataset-kind",
        required=True,
        choices=["qa_pairs", "instruction_response", "reasoning_explanation_pairs", "classification_examples"],
    )
    dataset_import_parser.add_argument("--source-path", required=True, type=Path)
    dataset_import_parser.add_argument("--dataset-version")
    dataset_import_parser.add_argument("--csv-question-column")
    dataset_import_parser.add_argument("--csv-answer-column")
    dataset_import_parser.add_argument("--csv-instruction-column")
    dataset_import_parser.add_argument("--csv-response-column")
    dataset_import_parser.add_argument("--csv-prompt-column")
    dataset_import_parser.add_argument("--csv-explanation-column")
    dataset_import_parser.add_argument("--csv-text-column")
    dataset_import_parser.add_argument("--csv-input-column")
    dataset_import_parser.add_argument("--csv-label-column")
    dataset_import_parser.add_argument("--csv-label-category")
    dataset_import_parser.set_defaults(command="dataset", dataset_command="import-dataset")

    dataset_review_parser = dataset_subparsers.add_parser(
        "sample-dataset-review",
        help="Generate a review batch from curated dataset rows",
    )
    dataset_review_parser.add_argument("--dataset-name")
    dataset_review_parser.add_argument("--import-run-id")
    dataset_review_parser.add_argument("--example-type")
    dataset_review_parser.add_argument("--limit", type=int, default=50)
    dataset_review_parser.add_argument("--prioritize", default="programmatic_first")
    dataset_review_parser.add_argument("--output", type=Path)
    dataset_review_parser.set_defaults(command="dataset", dataset_command="sample-dataset-review")

    dataset_label_parser = dataset_subparsers.add_parser(
        "label-dataset-example",
        help="Add a correction or canonical label to a dataset example",
    )
    dataset_label_parser.add_argument("dataset_example_id")
    dataset_label_parser.add_argument("--label-role", required=True, choices=["target_label", "correction_label", "canonical_label"])
    dataset_label_parser.add_argument("--label-value", required=True)
    dataset_label_parser.add_argument("--label-category")
    dataset_label_parser.add_argument("--reviewer")
    dataset_label_parser.add_argument("--reason")
    dataset_label_parser.add_argument("--canonical", action="store_true")
    dataset_label_parser.add_argument("--trainable", choices=["true", "false"])
    dataset_label_parser.add_argument("--ingestion-state", choices=["staged", "ready", "rejected", "archived"])
    dataset_label_parser.add_argument("--split-assignment", choices=["train", "validation", "test"])
    dataset_label_parser.add_argument("--review-note")
    dataset_label_parser.set_defaults(command="dataset", dataset_command="label-dataset-example")

    dataset_export_parser = dataset_subparsers.add_parser(
        "export-dataset-jsonl",
        help="Export curated dataset rows to JSONL training or evaluation files",
    )
    dataset_export_parser.add_argument("dataset_name")
    dataset_export_parser.add_argument("--import-run-id", action="append")
    dataset_export_parser.add_argument("--split", action="append", choices=["train", "validation", "test"])
    dataset_export_parser.add_argument("--example-type", action="append", choices=["instruction_response", "classification", "explanation_pair", "qa_pair"])
    dataset_export_parser.add_argument("--label-source", action="append")
    dataset_export_parser.add_argument("--canonical-only", action="store_true")
    dataset_export_parser.add_argument("--include-non-trainable", action="store_true")
    dataset_export_parser.add_argument("--evaluation-only", action="store_true")
    dataset_export_parser.add_argument("--export-name")
    dataset_export_parser.add_argument("--output-root", type=Path)
    dataset_export_parser.set_defaults(command="dataset", dataset_command="export-dataset-jsonl")

    dataset_compare_parser = dataset_subparsers.add_parser(
        "compare-dataset-runs",
        help="Compare two dataset import runs",
    )
    dataset_compare_parser.add_argument("left_import_run_id")
    dataset_compare_parser.add_argument("right_import_run_id")
    dataset_compare_parser.set_defaults(command="dataset", dataset_command="compare-dataset-runs")

    memory_parser = subparsers.add_parser("memory", help="Inspect research notes and promoted artifacts")
    memory_subparsers = memory_parser.add_subparsers(dest="memory_command")

    memory_notes_parser = memory_subparsers.add_parser("notes", help="List chronological research notes")
    memory_notes_parser.add_argument("--session-id")
    memory_notes_parser.set_defaults(command="memory", memory_command="notes")

    memory_artifacts_parser = memory_subparsers.add_parser("artifacts", help="List promoted research artifacts")
    memory_artifacts_parser.add_argument("--session-id")
    memory_artifacts_parser.set_defaults(command="memory", memory_command="artifacts")

    memory_promote_parser = memory_subparsers.add_parser("promote-note", help="Promote a research note into a structured artifact")
    memory_promote_parser.add_argument("note_path", type=Path)
    memory_promote_parser.add_argument("--type", required=True, choices=["hypothesis", "finding", "experiment", "decision", "milestone"])
    memory_promote_parser.add_argument("--title")
    memory_promote_parser.add_argument("--reason")
    memory_promote_parser.set_defaults(command="memory", memory_command="promote-note")

    session_parser = subparsers.add_parser("session", help="Inspect session history")
    session_subparsers = session_parser.add_subparsers(dest="session_command")

    session_inspect_parser = session_subparsers.add_parser("inspect", help="Inspect a session archive")
    session_inspect_parser.add_argument("session_id")
    session_inspect_parser.set_defaults(command="session", session_command="inspect")

    session_current_parser = session_subparsers.add_parser("current", help="Show the active thread for a session")
    session_current_parser.add_argument("session_id")
    session_current_parser.set_defaults(command="session", session_command="current")

    session_profile_parser = session_subparsers.add_parser(
        "profile",
        help="Show or update the session interaction profile",
    )
    session_profile_parser.add_argument("session_id")
    session_profile_parser.add_argument("--style", choices=["conversational", "direct"])
    session_profile_parser.add_argument("--depth", choices=["normal", "deep"])
    session_profile_parser.add_argument("--allow-suggestions", choices=["true", "false"])
    session_profile_parser.set_defaults(command="session", session_command="profile")

    session_reset_parser = session_subparsers.add_parser("reset", help="Clear the active thread for a session")
    session_reset_parser.add_argument("session_id")
    session_reset_parser.set_defaults(command="session", session_command="reset")

    run_parser = subparsers.add_parser("run", help="Execute a tool capability")
    run_parser.add_argument("tool_spec", help="Tool spec in the form '<tool_id>.<capability>'")
    run_parser.add_argument("--csv", dest="csv_path", type=Path, help="Dataset or directory input path")
    run_parser.add_argument("--session-id", default="default")
    run_parser.add_argument("--run-root", type=Path, help="Optional override for tool run output root")
    run_parser.add_argument("--h0", type=float, help="Optional Hubble constant override for GA flow")
    run_parser.add_argument("--boot", type=int, help="Optional bootstrap iteration count")
    run_parser.set_defaults(command="run")

    return parser


def main() -> int:
    configure_console_output()
    parser = build_parser()
    args = parser.parse_args()

    controller = AppController(repo_root=args.repo_root)
    registry = controller.registry
    formatter = controller.output_formatter
    output_format = args.format or controller.settings.default_output_format

    try:
        if args.command is None:
            return run_repl(
                controller=controller,
                formatter=formatter,
                output_format=output_format,
                session_id=controller.settings.default_session_id,
                input_path=None,
                run_root=None,
                params={},
            )

        if args.command == "list-tools":
            payload = {"tools": controller.list_tools()}
            print(render_output(formatter, payload, output_format))
            return 0

        if args.command == "list-capabilities":
            print(render_output(formatter, controller.list_app_capabilities(), output_format))
            return 0

        if args.command == "bundle" and args.bundle_command == "inspect":
            report = controller.inspect_bundle(args.bundle_id)
            print(render_output(formatter, report, output_format))
            return 0

        if args.command == "init":
            report = controller.initialize_workspace()
            print(render_output(formatter, report, output_format))
            return 0

        if args.command == "doctor":
            report = controller.build_doctor_report()
            print(render_output(formatter, report, output_format))
            return 0 if report["status"] == "ok" else 1

        if args.command == "ask":
            response = controller.ask(
                prompt=args.prompt,
                input_path=args.csv_path,
                params=build_params(args),
                session_id=args.session_id,
                run_root=args.run_root,
                client_surface=args.surface,
            )
            print(render_output(formatter, response, output_format))
            return 0

        if args.command == "repl":
            if args.csv_path is not None:
                validate_input_path(args.csv_path)
            validate_optional_run_params(args)
            return run_repl(
                controller=controller,
                formatter=formatter,
                output_format=output_format,
                session_id=args.session_id,
                input_path=args.csv_path,
                run_root=args.run_root,
                params=build_params(args),
            )

        if args.command == "command":
            validate_run_inputs(args)
            result = controller.run_command(
                action=args.action,
                target=args.target,
                input_path=args.csv_path,
                params=build_params(args),
                session_id=args.session_id,
                run_root=args.run_root,
            )
            print(render_output(formatter, formatter.tool_result_payload(result), output_format))
            return 0 if result.status == "ok" else 1

        if args.command == "archive" and args.archive_command == "list":
            report = controller.list_archive_records(
                session_id=args.session_id,
                tool_id=args.tool_id,
                capability=args.capability,
                status=args.status,
                date_from=args.date_from,
                date_to=args.date_to,
            )
            print(render_output(formatter, report, output_format))
            return 0

        if args.command == "archive" and args.archive_command == "search":
            report = controller.search_archive_records(
                args.query,
                session_id=args.session_id,
                tool_id=args.tool_id,
                capability=args.capability,
                status=args.status,
                date_from=args.date_from,
                date_to=args.date_to,
            )
            print(render_output(formatter, report, output_format))
            return 0

        if args.command == "archive" and args.archive_command == "latest":
            report = controller.latest_archive_record(
                session_id=args.session_id,
                tool_id=args.tool_id,
                capability=args.capability,
                status=args.status,
            )
            print(render_output(formatter, report, output_format))
            return 0

        if args.command == "archive" and args.archive_command == "summary":
            report = controller.archive_summary(
                session_id=args.session_id,
                tool_id=args.tool_id,
                capability=args.capability,
                date_from=args.date_from,
                date_to=args.date_to,
            )
            print(render_output(formatter, report, output_format))
            return 0

        if args.command == "archive" and args.archive_command == "compare":
            report = controller.compare_archive_runs(
                session_id=args.session_id,
                tool_id=args.tool_id,
                capability=args.capability,
                date_from=args.date_from,
                date_to=args.date_to,
            )
            print(render_output(formatter, report, output_format))
            return 0

        if args.command == "interaction" and args.interaction_command == "list":
            report = controller.list_interactions(
                session_id=args.session_id,
                project_id=args.project_id,
                resolution_strategy=args.resolution_strategy,
            )
            print(render_output(formatter, report, output_format))
            return 0

        if args.command == "interaction" and args.interaction_command == "search":
            report = controller.search_interactions(
                args.query,
                session_id=args.session_id,
                project_id=args.project_id,
                resolution_strategy=args.resolution_strategy,
                limit=args.limit,
            )
            print(render_output(formatter, report, output_format))
            return 0

        if args.command == "interaction" and args.interaction_command == "summary":
            report = controller.summarize_interactions(session_id=args.session_id, project_id=args.project_id)
            print(render_output(formatter, report, output_format))
            return 0

        if args.command == "interaction" and args.interaction_command == "evaluate":
            report = controller.evaluate_interactions(session_id=args.session_id, project_id=args.project_id)
            print(render_output(formatter, report, output_format))
            return 0

        if args.command == "interaction" and args.interaction_command == "export-labels":
            report = controller.export_labeled_examples(session_id=args.session_id, project_id=args.project_id)
            print(render_output(formatter, report, output_format))
            return 0

        if args.command == "interaction" and args.interaction_command == "patterns":
            report = controller.interaction_patterns(session_id=args.session_id, project_id=args.project_id)
            print(render_output(formatter, report, output_format))
            return 0

        if args.command == "persistence" and args.persistence_command == "status":
            print(render_output(formatter, controller.persistence_status(), output_format))
            return 0

        if args.command == "persistence" and args.persistence_command == "coverage":
            print(render_output(formatter, controller.persistence_coverage(), output_format))
            return 0

        if args.command == "persistence" and args.persistence_command == "doctor":
            print(render_output(formatter, controller.persistence_doctor(), output_format))
            return 0

        if args.command == "persistence" and args.persistence_command == "semantic-status":
            print(render_output(formatter, controller.semantic_status(), output_format))
            return 0

        if args.command == "persistence" and args.persistence_command == "backfill-embeddings":
            print(
                render_output(
                    formatter,
                    controller.backfill_memory_item_embeddings(limit=args.limit),
                    output_format,
                )
            )
            return 0

        if args.command == "dataset" and args.dataset_command == "derive-runtime-dataset":
            report = controller.import_runtime_dataset_examples(
                dataset_name=args.dataset_name,
                import_strategy=args.strategy,
                session_id=args.session_id,
                project_id=args.project_id,
                limit=args.limit,
            )
            print(render_output(formatter, report, output_format))
            return 0

        if args.command == "dataset" and args.dataset_command == "import-dataset":
            csv_mapping = {
                key: value
                for key, value in {
                    "question_column": args.csv_question_column,
                    "answer_column": args.csv_answer_column,
                    "instruction_column": args.csv_instruction_column,
                    "response_column": args.csv_response_column,
                    "prompt_column": args.csv_prompt_column,
                    "explanation_column": args.csv_explanation_column,
                    "text_column": args.csv_text_column,
                    "input_column": args.csv_input_column,
                    "label_column": args.csv_label_column,
                    "label_category": args.csv_label_category,
                }.items()
                if value
            }
            report = controller.import_dataset(
                dataset_name=args.dataset_name,
                source_format=args.source_format,
                dataset_kind=args.dataset_kind,
                source_path=args.source_path,
                dataset_version=args.dataset_version,
                csv_mapping=csv_mapping or None,
            )
            print(render_output(formatter, report, output_format))
            return 0

        if args.command == "dataset" and args.dataset_command == "sample-dataset-review":
            report = controller.sample_dataset_review(
                dataset_name=args.dataset_name,
                import_run_id=args.import_run_id,
                example_type=args.example_type,
                limit=args.limit,
                prioritize=args.prioritize,
                output_path=args.output,
            )
            print(render_output(formatter, report, output_format))
            return 0

        if args.command == "dataset" and args.dataset_command == "label-dataset-example":
            report = controller.label_dataset_example(
                dataset_example_id=args.dataset_example_id,
                label_role=args.label_role,
                label_value=args.label_value,
                label_category=args.label_category,
                reviewer=args.reviewer,
                reason=args.reason,
                is_canonical=args.canonical,
            )
            if (
                args.trainable is not None
                or args.ingestion_state is not None
                or args.split_assignment is not None
                or args.review_note is not None
            ):
                controller.update_dataset_example(
                    example_id=args.dataset_example_id,
                    trainable=parse_bool_option(args.trainable),
                    ingestion_state=args.ingestion_state,
                    split_assignment=args.split_assignment,
                    review_note=args.review_note,
                )
            print(render_output(formatter, report, output_format))
            return 0

        if args.command == "dataset" and args.dataset_command == "export-dataset-jsonl":
            report = controller.export_dataset_jsonl(
                dataset_name=args.dataset_name,
                import_run_ids=args.import_run_id,
                split_assignments=args.split,
                example_types=args.example_type,
                label_sources=args.label_source,
                canonical_only=args.canonical_only,
                trainable_only=not args.include_non_trainable,
                evaluation_only=args.evaluation_only,
                export_name=args.export_name,
                output_root=args.output_root,
            )
            print(render_output(formatter, report, output_format))
            return 0

        if args.command == "dataset" and args.dataset_command == "compare-dataset-runs":
            report = controller.compare_dataset_runs(
                left_import_run_id=args.left_import_run_id,
                right_import_run_id=args.right_import_run_id,
            )
            print(render_output(formatter, report, output_format))
            return 0

        if args.command == "memory" and args.memory_command == "notes":
            report = controller.list_research_notes(session_id=args.session_id)
            print(render_output(formatter, report, output_format))
            return 0

        if args.command == "memory" and args.memory_command == "artifacts":
            report = controller.list_research_artifacts(session_id=args.session_id)
            print(render_output(formatter, report, output_format))
            return 0

        if args.command == "memory" and args.memory_command == "promote-note":
            report = controller.promote_research_note(
                note_path=args.note_path,
                artifact_type=args.type,
                title=args.title,
                promotion_reason=args.reason,
            )
            print(render_output(formatter, report, output_format))
            return 0

        if args.command == "session" and args.session_command == "inspect":
            report = controller.inspect_session(args.session_id)
            print(render_output(formatter, report, output_format))
            return 0

        if args.command == "session" and args.session_command == "current":
            report = controller.current_session_thread(args.session_id)
            print(render_output(formatter, report, output_format))
            return 0

        if args.command == "session" and args.session_command == "profile":
            if args.style is None and args.depth is None and args.allow_suggestions is None:
                report = controller.get_session_profile(args.session_id)
            else:
                report = controller.set_session_profile(
                    args.session_id,
                    interaction_style=args.style,
                    reasoning_depth=args.depth,
                    allow_suggestions=parse_bool_option(args.allow_suggestions),
                )
            print(render_output(formatter, report, output_format))
            return 0

        if args.command == "session" and args.session_command == "reset":
            report = controller.reset_session_thread(args.session_id)
            print(render_output(formatter, report, output_format))
            return 0

        if args.command == "run":
            tool_id, capability = parse_tool_spec(args.tool_spec)
            validate_requested_capability(registry, tool_id, capability)
            validate_run_inputs(args)

            result = controller.run_tool(
                tool_id=tool_id,
                capability=capability,
                input_path=args.csv_path,
                params=build_params(args),
                session_id=args.session_id,
                run_root=args.run_root,
            )
            print(render_output(formatter, formatter.tool_result_payload(result), output_format))
            return 0 if result.status == "ok" else 1
    except Exception as exc:
        payload = formatter.error_payload(exc=exc, available_tools=registry.list_tools())
        print(
            render_output(formatter, payload, output_format),
            file=sys.stderr,
        )
        return 1

    parser.print_help()
    return 0


def parse_tool_spec(tool_spec: str) -> tuple[str, str]:
    parts = tool_spec.split(".")
    if len(parts) < 2:
        raise ValueError("Tool spec must look like '<tool_id>.<capability>'")
    return parts[0], ".".join(parts[1:])


def build_params(args: argparse.Namespace) -> dict[str, int | float]:
    params: dict[str, int | float] = {}
    if args.h0 is not None:
        params["h0"] = args.h0
    if args.boot is not None:
        params["boot"] = args.boot
    return params


def parse_bool_option(value: str | None) -> bool | None:
    if value is None:
        return None
    return value == "true"


def render_output(formatter, payload: dict, output_format: str) -> str:
    if output_format == "text":
        return formatter.render_text(payload)
    return formatter.render_json(payload)


def run_repl(
    *,
    controller: AppController,
    formatter,
    output_format: str,
    session_id: str,
    input_path: Path | None,
    run_root: Path | None,
    params: dict[str, int | float],
) -> int:
    active_csv = input_path
    print("Lumen interactive mode")
    print("Type /help for commands, /exit to quit.")
    if active_csv is not None:
        print(f"default_csv: {active_csv}")

    while True:
        try:
            raw = input("lumen> ")
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print()
            return 0

        prompt = raw.strip()
        if not prompt:
            continue

        if prompt in {"/exit", "/quit"}:
            return 0

        if prompt == "/help":
            print("Commands:")
            print("- /help")
            print("- /exit")
            print("- /quit")
            print("- /csv <path>    Set a default CSV for tool prompts")
            print("- /clear csv     Clear the default CSV")
            print("- /status        Show the current REPL context")
            print("- /current       Show the active session thread")
            print("- /inspect       Show the full session snapshot")
            print("- /reset         Clear the active session thread")
            continue

        if prompt.startswith("/csv "):
            candidate = Path(prompt[len("/csv ") :].strip())
            validate_input_path(candidate)
            active_csv = candidate
            print(f"default_csv: {active_csv}")
            continue

        if prompt == "/clear csv":
            active_csv = None
            print("default_csv: <none>")
            continue

        if prompt == "/status":
            print(f"session_id: {session_id}")
            print(f"default_csv: {active_csv or '<none>'}")
            print(f"run_root: {run_root or '<default>'}")
            print(f"format: {output_format}")
            if params:
                print(f"params: {json.dumps(params, indent=2)}")
            active_thread_report = controller.current_session_thread(session_id)
            active_thread = active_thread_report.get("active_thread") or {}
            interaction_profile = active_thread_report.get("interaction_profile") or {}
            if interaction_profile:
                print(f"interaction_style: {interaction_profile.get('interaction_style')}")
                print(f"reasoning_depth: {interaction_profile.get('reasoning_depth')}")
                print(f"allow_suggestions: {interaction_profile.get('allow_suggestions')}")
            if active_thread.get("confidence_posture"):
                print(f"active_confidence_posture: {active_thread.get('confidence_posture')}")
            summary = controller.summarize_interactions(session_id=session_id)
            if summary.get("latest_clarification"):
                print(f"latest_clarification: {summary.get('latest_clarification')}")
            if summary.get("recent_clarification_mix"):
                print(f"recent_clarification_mix: {summary.get('recent_clarification_mix')}")
            clarification_trend = summary.get("clarification_trend") or []
            if clarification_trend:
                print(f"clarification_trend: {' -> '.join(str(item) for item in clarification_trend)}")
            if summary.get("clarification_drift"):
                print(f"clarification_drift: {summary.get('clarification_drift')}")
            if summary.get("latest_posture"):
                print(f"latest_posture: {summary.get('latest_posture')}")
            if summary.get("recent_posture_mix"):
                print(f"recent_posture_mix: {summary.get('recent_posture_mix')}")
            posture_trend = summary.get("posture_trend") or []
            if posture_trend:
                print(f"posture_trend: {' -> '.join(str(item) for item in posture_trend)}")
            if summary.get("posture_drift"):
                print(f"posture_drift: {summary.get('posture_drift')}")
            continue

        if prompt == "/current":
            report = controller.current_session_thread(session_id)
            print(render_output(formatter, report, output_format))
            continue

        if prompt == "/inspect":
            report = controller.inspect_session(session_id)
            print(render_output(formatter, report, output_format))
            continue

        if prompt == "/reset":
            report = controller.reset_session_thread(session_id)
            print(render_output(formatter, report, output_format))
            continue

        try:
            response = controller.ask(
                prompt=prompt,
                input_path=active_csv,
                params=params,
                session_id=session_id,
                run_root=run_root,
            )
            print(render_output(formatter, response, output_format))
        except Exception as exc:
            payload = formatter.error_payload(exc=exc, available_tools=controller.registry.list_tools())
            print(render_output(formatter, payload, output_format), file=sys.stderr)


def validate_requested_capability(registry: ToolRegistry, tool_id: str, capability: str) -> None:
    available = registry.list_tools()
    if tool_id not in available:
        known_tools = ", ".join(sorted(available)) or "<none>"
        raise ValueError(f"Unknown tool bundle '{tool_id}'. Available bundles: {known_tools}")

    if capability not in available[tool_id]:
        known_capabilities = ", ".join(sorted(available[tool_id])) or "<none>"
        raise ValueError(
            f"Unknown capability '{capability}' for bundle '{tool_id}'. "
            f"Available capabilities: {known_capabilities}"
        )


def validate_run_inputs(args: argparse.Namespace) -> None:
    if args.csv_path is not None:
        validate_input_path(args.csv_path)
    validate_optional_run_params(args)


def validate_input_path(input_path: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")


def validate_optional_run_params(args: argparse.Namespace) -> None:
    if args.h0 is not None and args.h0 <= 0:
        raise ValueError("--h0 must be greater than 0")

    if args.boot is not None and args.boot <= 0:
        raise ValueError("--boot must be greater than 0")
if __name__ == "__main__":
    raise SystemExit(main())

