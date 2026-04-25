from pathlib import Path

from lumen.app.settings import AppSettings
from lumen.services.storage_hygiene_service import StorageHygieneService


def test_storage_hygiene_report_detects_oversized_files(tmp_path: Path) -> None:
    settings = AppSettings.from_repo_root(tmp_path)
    service = StorageHygieneService(settings)

    interaction_dir = settings.interactions_root / "default"
    interaction_dir.mkdir(parents=True, exist_ok=True)
    oversized_interaction = interaction_dir / "huge.json"
    oversized_interaction.write_text("{}", encoding="utf-8")

    session_dir = settings.sessions_root / "default"
    session_dir.mkdir(parents=True, exist_ok=True)
    oversized_session = session_dir / "thread_state.json"
    oversized_session.write_text("{}", encoding="utf-8")

    oversized_interaction.write_bytes(b"x" * (settings.max_interaction_record_bytes + 1))
    oversized_session.write_bytes(b"x" * (settings.max_session_state_bytes + 1))

    report = service.report()

    assert report["counts"]["oversized_interaction_files"] == 1
    assert report["counts"]["oversized_session_files"] == 1


def test_storage_hygiene_cleanup_prunes_stale_tool_runs(tmp_path: Path) -> None:
    settings = AppSettings.from_repo_root(tmp_path)
    service = StorageHygieneService(settings)
    capability_root = settings.tool_runs_root / "default" / "simulate" / "orbit"
    for name in ("20260101T000000000000Z", "20260102T000000000000Z", "20260103T000000000000Z"):
        run_dir = capability_root / name / "outputs"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "artifact.json").write_text("{}", encoding="utf-8")

    result = service.cleanup(prune_oversized=False, prune_tool_runs=True, retain_per_capability=1)

    assert result["removed_count"] == 2
    remaining = sorted(path.name for path in capability_root.iterdir() if path.is_dir())
    assert remaining == ["20260103T000000000000Z"]


def test_storage_hygiene_cleanup_removes_oversized_runtime_files(tmp_path: Path) -> None:
    settings = AppSettings.from_repo_root(tmp_path)
    service = StorageHygieneService(settings)

    interaction_dir = settings.interactions_root / "default"
    interaction_dir.mkdir(parents=True, exist_ok=True)
    oversized_interaction = interaction_dir / "huge.json"
    oversized_interaction.write_bytes(b"x" * (settings.max_interaction_record_bytes + 1))

    session_dir = settings.sessions_root / "default"
    session_dir.mkdir(parents=True, exist_ok=True)
    oversized_session = session_dir / "thread_state.json"
    oversized_session.write_bytes(b"x" * (settings.max_session_state_bytes + 1))

    result = service.cleanup(prune_oversized=True, prune_tool_runs=False)

    assert result["removed_count"] == 2
    assert not oversized_interaction.exists()
    assert not oversized_session.exists()
