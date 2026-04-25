from pathlib import Path
import json

from lumen.app.settings import AppSettings
from lumen.memory.archive_manager import ArchiveManager
from lumen.reporting.output_formatter import OutputFormatter
from lumen.services.archive_service import ArchiveService
from lumen.tools.registry_types import ToolResult


def test_archive_manager_records_and_lists_tool_runs(tmp_path: Path) -> None:
    settings = AppSettings.from_repo_root(tmp_path)
    manager = ArchiveManager(settings=settings)
    run_dir = tmp_path / "data" / "tool_runs" / "default" / "anh" / "spectral_dip_scan" / "run1"
    run_dir.mkdir(parents=True, exist_ok=True)

    result = ToolResult(
        status="ok",
        tool_id="anh",
        capability="spectral_dip_scan",
        summary="test run",
        structured_data={"parsed_results": {"bulk_flow_amplitude": "123 km/s"}},
        run_dir=run_dir,
    )

    record = manager.record_tool_run(session_id="default", result=result)
    records = manager.list_records(session_id="default")
    session_report = manager.inspect_session("default")

    assert record.archive_path.exists()
    assert (record.archive_path.parent / "_index.json").exists()
    assert len(records) == 1
    assert records[0]["tool_id"] == "anh"
    assert records[0]["capability"] == "spectral_dip_scan"
    assert session_report["record_count"] == 1


def test_archive_manager_extracts_research_run_contract_fields(tmp_path: Path) -> None:
    settings = AppSettings.from_repo_root(tmp_path)
    manager = ArchiveManager(settings=settings)
    run_dir = tmp_path / "data" / "tool_runs" / "default" / "anh" / "spectral_dip_scan" / "run1"
    run_dir.mkdir(parents=True, exist_ok=True)

    result = ToolResult(
        status="ok",
        tool_id="anh",
        capability="spectral_dip_scan",
        summary="GA Local Analysis Kit run completed",
        structured_data={
            "bundle_standard": "lumen_research_bundle_v1",
            "bundle_id": "anh",
            "capability": "spectral_dip_scan",
            "run_id": "run_2026_03_16_213045",
            "target_label": "GA Local Analysis Kit",
            "input_files": ["dataset.csv"],
            "analysis_status": {
                "validated": True,
                "analysis_ran": True,
                "plot_generated": True,
                "line_detected": None,
                "result_quality": "scientific_output_present",
                "failure_reason": None,
            },
            "batch_record": {"target_label": "GA Local Analysis Kit"},
            "domain_payload": {"parsed_results": {"bulk_flow_amplitude": "123 km/s"}},
            "provenance": {"run_id": "run_2026_03_16_213045"},
        },
        run_dir=run_dir,
    )

    manager.record_tool_run(session_id="default", result=result)
    records = manager.list_records(session_id="default")

    assert records[0]["run_id"] == "run_2026_03_16_213045"
    assert records[0]["target_label"] == "GA Local Analysis Kit"
    assert records[0]["result_quality"] == "scientific_output_present"
    assert records[0]["research_run"]["analysis_status"]["validated"] is True


def test_archive_manager_writes_semantic_signature_to_index(tmp_path: Path) -> None:
    settings = AppSettings.from_repo_root(tmp_path)
    manager = ArchiveManager(settings=settings)
    run_dir = tmp_path / "data" / "tool_runs" / "default" / "anh" / "spectral_dip_scan" / "run1"
    run_dir.mkdir(parents=True, exist_ok=True)

    manager.record_tool_run(
        session_id="default",
        result=ToolResult(
            status="ok",
            tool_id="anh",
            capability="spectral_dip_scan",
            summary="Great Attractor confidence report",
            run_dir=run_dir,
        ),
    )

    index_path = settings.archive_root / "default" / "anh" / "spectral_dip_scan" / "_index.json"
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))

    assert len(index_payload) == 1
    signature = index_payload[0]["semantic_signature"]
    assert "confidence" in signature["prompt_tokens"]
    assert "report" in signature["prompt_tokens"]
    assert signature["dominant_intent"] is None


def test_archive_index_search_caps_candidates_by_setting(tmp_path: Path) -> None:
    (tmp_path / "lumen.toml").write_text(
        "\n".join(
            [
                "[app]",
                "search_candidate_limit = 2",
            ]
        ),
        encoding="utf-8",
    )
    settings = AppSettings.from_repo_root(tmp_path)
    manager = ArchiveManager(settings=settings)

    for run_name, summary in [
        ("run1", "Great Attractor routing analysis alpha"),
        ("run2", "Great Attractor routing analysis beta"),
        ("run3", "Great Attractor routing analysis gamma"),
    ]:
        run_dir = tmp_path / "data" / "tool_runs" / "default" / "anh" / "spectral_dip_scan" / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        manager.record_tool_run(
            session_id="default",
            result=ToolResult(
                status="ok",
                tool_id="anh",
                capability="spectral_dip_scan",
                summary=summary,
                run_dir=run_dir,
            ),
        )

    query_understanding = manager.prompt_nlu.analyze("great attractor routing analysis")
    matches = manager._search_index(
        "routing",
        query_understanding=query_understanding,
        session_id="default",
        tool_id=None,
        capability=None,
        status=None,
    )

    assert len(matches) == 2


def test_archive_manager_returns_empty_for_missing_session(tmp_path: Path) -> None:
    settings = AppSettings.from_repo_root(tmp_path)
    manager = ArchiveManager(settings=settings)

    assert manager.list_records(session_id="missing") == []
    assert manager.inspect_session("missing")["record_count"] == 0


def test_archive_manager_can_search_record_content(tmp_path: Path) -> None:
    settings = AppSettings.from_repo_root(tmp_path)
    manager = ArchiveManager(settings=settings)
    run_dir = tmp_path / "data" / "tool_runs" / "default" / "anh" / "spectral_dip_scan" / "run1"
    run_dir.mkdir(parents=True, exist_ok=True)

    result = ToolResult(
        status="ok",
        tool_id="anh",
        capability="spectral_dip_scan",
        summary="Great Attractor confirmation candidate",
        structured_data={"parsed_results": {"angle_to_ga_dir_(~200°,-49°)": "1.10°"}},
        run_dir=run_dir,
    )
    manager.record_tool_run(session_id="default", result=result)

    result = manager.search_records("confirmation", session_id="default")

    assert result.record_count == 1
    assert result.matches[0].record["summary"] == "Great Attractor confirmation candidate"
    assert "summary" in result.matches[0].matched_fields
    assert result.matches[0].score_breakdown["keyword_score"] > 0
    assert result.matches[0].score_breakdown["semantic_score"] >= 0


def test_archive_manager_can_use_semantic_overlap_for_capability_style_query(tmp_path: Path) -> None:
    settings = AppSettings.from_repo_root(tmp_path)
    manager = ArchiveManager(settings=settings)
    run_dir = tmp_path / "data" / "tool_runs" / "default" / "anh" / "spectral_dip_scan" / "run1"
    run_dir.mkdir(parents=True, exist_ok=True)

    result = ToolResult(
        status="ok",
        tool_id="anh",
        capability="spectral_dip_scan",
        summary="Great Attractor confirmation candidate",
        run_dir=run_dir,
    )
    manager.record_tool_run(session_id="default", result=result)

    search = manager.search_records("ga local bulk flow analysis", session_id="default")

    assert search.record_count == 1
    assert search.matches[0].record["capability"] == "spectral_dip_scan"
    assert "semantic" in search.matches[0].matched_fields
    assert search.matches[0].score_breakdown["semantic_score"] > 0


def test_archive_search_prefers_stronger_blended_match(tmp_path: Path) -> None:
    settings = AppSettings.from_repo_root(tmp_path)
    manager = ArchiveManager(settings=settings)

    for run_name, summary in [
        ("run1", "Great Attractor routing analysis candidate"),
        ("run2", "routing logs"),
    ]:
        run_dir = tmp_path / "data" / "tool_runs" / "default" / "anh" / "spectral_dip_scan" / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        manager.record_tool_run(
            session_id="default",
            result=ToolResult(
                status="ok",
                tool_id="anh",
                capability="spectral_dip_scan",
                summary=summary,
                run_dir=run_dir,
            ),
        )

    search = manager.search_records("great attractor routing analysis", session_id="default")

    assert search.record_count == 1
    assert search.matches[0].record["summary"] == "Great Attractor routing analysis candidate"


def test_archive_manager_can_filter_by_status_and_latest(tmp_path: Path) -> None:
    settings = AppSettings.from_repo_root(tmp_path)
    manager = ArchiveManager(settings=settings)
    run_dir = tmp_path / "data" / "tool_runs" / "default" / "anh" / "spectral_dip_scan" / "run1"
    run_dir.mkdir(parents=True, exist_ok=True)

    ok_result = ToolResult(
        status="ok",
        tool_id="anh",
        capability="spectral_dip_scan",
        summary="successful run",
        run_dir=run_dir,
    )
    partial_result = ToolResult(
        status="partial",
        tool_id="anh",
        capability="spectral_dip_scan",
        summary="partial run",
        run_dir=run_dir,
    )

    manager.record_tool_run(session_id="default", result=ok_result)
    manager.record_tool_run(session_id="default", result=partial_result)

    ok_records = manager.list_records(session_id="default", status="ok")
    latest_partial = manager.latest_record(session_id="default", status="partial")

    assert len(ok_records) == 1
    assert ok_records[0]["summary"] == "successful run"
    assert latest_partial is not None
    assert latest_partial["summary"] == "partial run"


def test_archive_manager_can_summarize_records(tmp_path: Path) -> None:
    settings = AppSettings.from_repo_root(tmp_path)
    manager = ArchiveManager(settings=settings)
    run_dir = tmp_path / "data" / "tool_runs" / "default" / "anh" / "spectral_dip_scan" / "run1"
    run_dir.mkdir(parents=True, exist_ok=True)

    manager.record_tool_run(
        session_id="default",
        result=ToolResult(
            status="ok",
            tool_id="anh",
            capability="spectral_dip_scan",
            summary="first success",
            run_dir=run_dir,
        ),
    )
    manager.record_tool_run(
        session_id="default",
        result=ToolResult(
            status="partial",
            tool_id="anh",
            capability="spectral_dip_scan",
            summary="partial follow-up",
            run_dir=run_dir,
        ),
    )

    summary = manager.summarize_records(session_id="default")

    assert summary["record_count"] == 2
    assert summary["status_counts"]["ok"] == 1
    assert summary["status_counts"]["partial"] == 1
    assert summary["tool_counts"]["anh"] == 2
    assert "spectral_dip_scan" in summary["latest_by_capability"]
    assert "target_label_counts" in summary
    assert "result_quality_counts" in summary


def test_archive_manager_can_compare_runs_by_target_with_trend_summary(tmp_path: Path) -> None:
    settings = AppSettings.from_repo_root(tmp_path)
    manager = ArchiveManager(settings=settings)

    for run_name, created_target, quality, summary in [
        ("run1", "GA Local Analysis Kit", "partial_artifacts", "initial pass"),
        ("run2", "GA Local Analysis Kit", "scientific_output_present", "improved pass"),
        ("run3", "Cluster Window", "scientific_output_present", "cluster pass"),
    ]:
        run_dir = tmp_path / "data" / "tool_runs" / "default" / "anh" / "spectral_dip_scan" / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        manager.record_tool_run(
            session_id="default",
            result=ToolResult(
                status="ok",
                tool_id="anh",
                capability="spectral_dip_scan",
                summary=summary,
                structured_data={
                    "bundle_standard": "lumen_research_bundle_v1",
                    "bundle_id": "anh",
                    "capability": "spectral_dip_scan",
                    "run_id": f"run_2026_03_16_21304{run_name[-1]}",
                    "target_label": created_target,
                    "analysis_status": {
                        "validated": True,
                        "analysis_ran": True,
                        "plot_generated": True,
                        "line_detected": None,
                        "result_quality": quality,
                        "failure_reason": None,
                    },
                    "batch_record": {"target_label": created_target},
                    "domain_payload": {},
                    "provenance": {"run_id": f"run_2026_03_16_21304{run_name[-1]}"},
                },
                run_dir=run_dir,
            ),
        )

    comparison = manager.compare_runs_by_target(
        session_id="default",
        capability="spectral_dip_scan",
    )

    assert comparison["record_count"] == 3
    assert comparison["target_count"] == 2
    first_group = comparison["target_groups"][0]
    assert first_group["target_label"] == "GA Local Analysis Kit"
    assert first_group["run_count"] == 2
    assert first_group["result_quality_counts"]["partial_artifacts"] == 1
    assert first_group["result_quality_counts"]["scientific_output_present"] == 1
    assert "earlier runs were mixed" in first_group["trend_summary"]
    assert len(first_group["recent_runs"]) == 2


def test_archive_manager_trims_nested_response_context_from_payloads(tmp_path: Path) -> None:
    settings = AppSettings.from_repo_root(tmp_path)
    manager = ArchiveManager(settings=settings)
    run_dir = tmp_path / "data" / "tool_runs" / "default" / "anh" / "spectral_dip_scan" / "run1"
    run_dir.mkdir(parents=True, exist_ok=True)

    result = ToolResult(
        status="ok",
        tool_id="anh",
        capability="spectral_dip_scan",
        summary="sanitized run",
        structured_data={
            "parsed_results": {"bulk_flow_amplitude": "123 km/s"},
            "top_matches": [{"record": {"summary": "nested archive match"}}],
        },
        provenance={
            "source": "adapter",
            "context": {
                "response": {"schema_type": "assistant_response"},
                "top_interaction_matches": [{"record": {"prompt": "what about that"}}],
                "kept": {"query": "run anh"},
            },
        },
        run_dir=run_dir,
    )

    manager.record_tool_run(session_id="default", result=result)
    records = manager.list_records(session_id="default")

    assert len(records) == 1
    record = records[0]
    assert "top_matches" not in record["structured_data"]
    assert "response" not in record["provenance"]["context"]
    assert "top_interaction_matches" not in record["provenance"]["context"]
    assert record["provenance"]["context"]["kept"]["query"] == "run anh"


def test_archive_manager_sanitizes_legacy_records_on_load(tmp_path: Path) -> None:
    settings = AppSettings.from_repo_root(tmp_path)
    manager = ArchiveManager(settings=settings)
    archive_dir = settings.archive_root / "default" / "anh" / "spectral_dip_scan"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / "legacy.json"
    archive_path.write_text(
        json.dumps(
            {
                "schema_type": "archive_record",
                "schema_version": "1",
                "session_id": "default",
                "tool_id": "anh",
                "capability": "spectral_dip_scan",
                "status": "ok",
                "summary": "legacy run",
                "created_at": "2026-03-15T00:00:00+00:00",
                "structured_data": {
                    "parsed_results": {"bulk_flow_amplitude": "123 km/s"},
                    "top_matches": [{"record": {"summary": "nested"}}],
                },
                "provenance": {
                    "context": {
                        "response": {"schema_type": "assistant_response"},
                        "top_interaction_matches": [{"record": {"prompt": "what about that"}}],
                        "kept": {"query": "run anh"},
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    record = manager.load_record(archive_path)

    assert "top_matches" not in record["structured_data"]
    assert "response" not in record["provenance"]["context"]
    assert "top_interaction_matches" not in record["provenance"]["context"]
    assert record["provenance"]["context"]["kept"]["query"] == "run anh"


def test_archive_service_compacts_retrieval_context_records(tmp_path: Path) -> None:
    settings = AppSettings.from_repo_root(tmp_path)
    manager = ArchiveManager(settings=settings)
    service = ArchiveService(manager, OutputFormatter(), str(tmp_path), settings)
    run_dir = tmp_path / "data" / "tool_runs" / "default" / "anh" / "spectral_dip_scan" / "run1"
    run_dir.mkdir(parents=True, exist_ok=True)

    manager.record_tool_run(
        session_id="default",
        result=ToolResult(
            status="ok",
            tool_id="anh",
            capability="spectral_dip_scan",
            summary="Great Attractor confirmation candidate",
            structured_data={"parsed_results": {"bulk_flow_amplitude": "123 km/s"}},
            provenance={"source": "adapter", "details": {"kept": True}},
            logs=["line 1", "line 2"],
            run_dir=run_dir,
        ),
    )

    context = service.retrieve_context("confirmation", session_id="default")
    record = context["top_matches"][0]["record"]

    assert record["tool_id"] == "anh"
    assert record["summary"] == "Great Attractor confirmation candidate"
    assert "structured_data" not in record
    assert "logs" not in record
    assert "provenance" not in record
    assert context["top_matches"][0]["score_breakdown"]["keyword_score"] > 0


def test_archive_service_preserves_research_run_fields_in_compact_context(tmp_path: Path) -> None:
    settings = AppSettings.from_repo_root(tmp_path)
    manager = ArchiveManager(settings=settings)
    service = ArchiveService(manager, OutputFormatter(), str(tmp_path), settings)
    run_dir = tmp_path / "data" / "tool_runs" / "default" / "anh" / "spectral_dip_scan" / "run1"
    run_dir.mkdir(parents=True, exist_ok=True)

    manager.record_tool_run(
        session_id="default",
        result=ToolResult(
            status="ok",
            tool_id="anh",
            capability="spectral_dip_scan",
            summary="GA Local Analysis Kit run completed",
            structured_data={
                "bundle_standard": "lumen_research_bundle_v1",
                "bundle_id": "anh",
                "capability": "spectral_dip_scan",
                "run_id": "run_2026_03_16_213045",
                "target_label": "GA Local Analysis Kit",
                "input_files": ["dataset.csv"],
                "analysis_status": {
                    "validated": True,
                    "analysis_ran": True,
                    "plot_generated": True,
                    "line_detected": None,
                    "result_quality": "scientific_output_present",
                    "failure_reason": None,
                },
                "batch_record": {"target_label": "GA Local Analysis Kit"},
                "domain_payload": {"parsed_results": {"bulk_flow_amplitude": "123 km/s"}},
                "provenance": {"run_id": "run_2026_03_16_213045"},
            },
            run_dir=run_dir,
        ),
    )

    context = service.retrieve_context("ga local", session_id="default")
    record = context["top_matches"][0]["record"]

    assert record["run_id"] == "run_2026_03_16_213045"
    assert record["target_label"] == "GA Local Analysis Kit"
    assert record["result_quality"] == "scientific_output_present"


def test_archive_service_truncates_compact_context_summary(tmp_path: Path) -> None:
    settings = AppSettings.from_repo_root(tmp_path)
    manager = ArchiveManager(settings=settings)
    service = ArchiveService(manager, OutputFormatter(), str(tmp_path), settings)
    run_dir = tmp_path / "data" / "tool_runs" / "default" / "anh" / "spectral_dip_scan" / "run1"
    run_dir.mkdir(parents=True, exist_ok=True)

    manager.record_tool_run(
        session_id="default",
        result=ToolResult(
            status="ok",
            tool_id="anh",
            capability="spectral_dip_scan",
            summary="Great Attractor " + ("confirmation candidate " * 20),
            run_dir=run_dir,
        ),
    )

    context = service.retrieve_context("confirmation", session_id="default")
    summary = context["top_matches"][0]["record"]["summary"]

    assert len(summary) <= 160
    assert summary.endswith("...")


def test_archive_manager_search_uses_recursive_indexes_across_session_scope(tmp_path: Path) -> None:
    settings = AppSettings.from_repo_root(tmp_path)
    manager = ArchiveManager(settings=settings)
    run_dir = tmp_path / "data" / "tool_runs" / "default" / "anh" / "spectral_dip_scan" / "run1"
    run_dir.mkdir(parents=True, exist_ok=True)

    manager.record_tool_run(
        session_id="default",
        result=ToolResult(
            status="ok",
            tool_id="anh",
            capability="spectral_dip_scan",
            summary="Great Attractor confirmation candidate",
            run_dir=run_dir,
        ),
    )

    result = manager.search_records("confirmation", session_id="default")

    assert result.record_count == 1
    assert result.matches[0].record["summary"] == "Great Attractor confirmation candidate"


def test_archive_manager_reports_index_status(tmp_path: Path) -> None:
    settings = AppSettings.from_repo_root(tmp_path)
    manager = ArchiveManager(settings=settings)
    run_dir = tmp_path / "data" / "tool_runs" / "default" / "anh" / "spectral_dip_scan" / "run1"
    run_dir.mkdir(parents=True, exist_ok=True)

    manager.record_tool_run(
        session_id="default",
        result=ToolResult(
            status="ok",
            tool_id="anh",
            capability="spectral_dip_scan",
            summary="indexed run",
            run_dir=run_dir,
        ),
    )

    status = manager.index_status(session_id="default")

    assert status["record_file_count"] == 1
    assert status["indexed_record_count"] == 1
    assert status["legacy_record_count"] == 0
    assert status["coverage_ratio"] == 1.0

