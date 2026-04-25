from pathlib import Path

from lumen.schemas.archive_schema import ARCHIVE_RECORD_SCHEMA_VERSION, ArchiveRecordSchema
from lumen.schemas.manifest_schema import BUNDLE_MANIFEST_SCHEMA_VERSION, BundleManifestSchema
from lumen.schemas.retrieval_schema import RETRIEVAL_RESULT_SCHEMA_VERSION, RetrievalResultSchema
from lumen.tools.registry_types import BundleManifest


def test_bundle_manifest_schema_defaults_and_round_trip() -> None:
    payload = {
        "id": "anh",
        "name": "Astronomical Node Heuristics",
        "entrypoint": "bundle.py",
        "capabilities": [],
    }

    normalized = BundleManifestSchema.normalize(payload)
    BundleManifestSchema.validate(normalized)

    assert normalized["schema_version"] == BUNDLE_MANIFEST_SCHEMA_VERSION


def test_archive_record_schema_defaults_and_validation() -> None:
    payload = {
        "session_id": "default",
        "tool_id": "anh",
        "capability": "spectral_dip_scan",
        "status": "ok",
        "summary": "done",
        "created_at": "2026-01-01T00:00:00+00:00",
    }

    normalized = ArchiveRecordSchema.normalize(payload)
    ArchiveRecordSchema.validate(normalized)

    assert normalized["schema_type"] == "archive_record"
    assert normalized["schema_version"] == ARCHIVE_RECORD_SCHEMA_VERSION


def test_retrieval_result_schema_builders_include_versions() -> None:
    search_payload = RetrievalResultSchema.build_search_payload(
        repo_root="C:/repo",
        session_id="default",
        tool_id="anh",
        capability="spectral_dip_scan",
        query="ga",
        matches=[],
    )
    latest_payload = RetrievalResultSchema.build_latest_payload(
        repo_root="C:/repo",
        session_id="default",
        tool_id="anh",
        capability="spectral_dip_scan",
        status="ok",
        record=None,
    )

    assert search_payload["schema_version"] == RETRIEVAL_RESULT_SCHEMA_VERSION
    assert search_payload["schema_type"] == "archive_search_result"
    assert latest_payload["schema_version"] == RETRIEVAL_RESULT_SCHEMA_VERSION
    assert latest_payload["schema_type"] == "archive_latest_result"


def test_bundle_manifest_loads_schema_version_from_file(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        '{"schema_version":"1","id":"anh","name":"ANH","entrypoint":"bundle.py","capabilities":[]}',
        encoding="utf-8",
    )

    manifest = BundleManifest.from_file(manifest_path)

    assert manifest.schema_version == "1"

