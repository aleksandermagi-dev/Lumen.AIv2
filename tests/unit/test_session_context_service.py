from pathlib import Path
import shutil
import json

from lumen.app.controller import AppController
from lumen.app.models import InteractionProfile
from lumen.app.settings import AppSettings
from lumen.memory.session_state_manager import SessionStateManager


def test_controller_persists_active_thread_state(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)

    controller.ask(prompt="create a migration plan for lumen", session_id="default")
    controller.ask(prompt="expand that further", session_id="default")

    report = controller.inspect_session("default")

    assert report["active_thread"]["mode"] == "planning"
    assert report["active_thread"]["kind"] == "planning.migration"
    assert report["active_thread"]["prompt"] == "expand that further"
    assert report["active_thread"]["objective"] == "Plan work for: create a migration plan for lumen"
    assert report["active_thread"]["thread_summary"]
    assert (tmp_path / "data" / "sessions" / "default" / "thread_state.json").exists()


def test_controller_can_reset_active_thread_state(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)

    controller.ask(prompt="create a migration plan for lumen", session_id="default")
    report = controller.reset_session_thread("default")

    assert report["cleared"] is True
    assert report["active_thread"] is None
    assert controller.current_session_thread("default")["active_thread"] is None
    assert not (tmp_path / "data" / "sessions" / "default" / "thread_state.json").exists()


def test_controller_persists_tool_context_in_active_thread(tmp_path: Path, monkeypatch) -> None:
    _copy_project_assets(tmp_path)
    controller = AppController(repo_root=tmp_path)
    sample_csv = tmp_path / "data" / "examples" / "cf4_ga_cone_template.csv"

    def fake_run_tool(**kwargs):
        from lumen.tools.registry_types import ToolResult

        return ToolResult(
            status="ok",
            tool_id="anh",
            capability="spectral_dip_scan",
            summary="GA Local Analysis Kit run completed",
        )

    monkeypatch.setattr(controller.tool_execution_service, "run_tool", fake_run_tool)

    controller.ask(
        prompt="run anh",
        session_id="default",
        input_path=sample_csv,
    )

    active_thread = controller.current_session_thread("default")["active_thread"]
    assert active_thread["tool_context"]["tool_id"] == "anh"
    assert active_thread["tool_context"]["capability"] == "spectral_dip_scan"
    assert str(sample_csv) == active_thread["tool_context"]["input_path"]


def test_controller_persists_reasoning_state_in_active_thread(tmp_path: Path) -> None:
    _copy_project_assets(tmp_path)
    controller = AppController(repo_root=tmp_path)

    controller.ask(prompt="what is entropy", session_id="default")

    active_thread = controller.current_session_thread("default")["active_thread"]
    reasoning_state = active_thread["reasoning_state"]

    assert reasoning_state["current_path"] == "research:research.summary"
    assert reasoning_state["selected_mode"] == "collab"
    assert reasoning_state["mode_behavior"]["mode"] == "collab"
    assert reasoning_state["canonical_subject"] == "entropy"
    assert reasoning_state["turn_status"] == "routed"
    assert reasoning_state["resolved_prompt"] == "what is entropy"


def test_session_state_manager_derives_cognitive_fields_from_reasoning_state(tmp_path: Path) -> None:
    settings = AppSettings.from_repo_root(tmp_path)
    manager = SessionStateManager(settings=settings)

    payload = manager.update_active_thread(
        session_id="default",
        prompt="teach me entropy",
        response={
            "mode": "research",
            "kind": "research.summary",
            "summary": "Entropy is a measure of how many microscopic configurations fit a macroscopic state.",
            "reasoning_state": {
                "current_path": "research:research.summary",
                "intent_domain": "learning_teaching",
                "response_depth": "deep",
                "conversation_phase": "exploration",
                "confidence": 0.82,
                "confidence_tier": "high",
                "response_style": {"intent_domain": "learning_teaching"},
            },
        },
    )

    assert payload["intent_domain"] == "learning_teaching"
    assert payload["response_depth"] == "deep"
    assert payload["conversation_phase"] == "exploration"
    assert payload["reasoning_state"]["confidence_tier"] == "high"
    assert payload["reasoning_state"]["response_style"]["intent_domain"] == "learning_teaching"


def test_session_state_manager_round_trips_cognitive_frame_fields(tmp_path: Path) -> None:
    settings = AppSettings.from_repo_root(tmp_path)
    manager = SessionStateManager(settings=settings)

    manager.update_active_thread(
        session_id="default",
        prompt="debug the workspace parser",
        response={
            "mode": "planning",
            "kind": "planning.debug",
            "summary": "Let’s isolate the parser regression first.",
            "reasoning_state": {
                "current_path": "planning:planning.debug",
                "intent_domain": "technical_engineering",
                "response_depth": "standard",
                "conversation_phase": "execution",
                "confidence": 0.67,
                "confidence_tier": "medium",
                "route_decision": {"mode": "planning", "kind": "planning.debug"},
                "memory_context_used": [
                    {"source": "bug_log", "label": "Workspace parser issue"}
                ],
                "tool_usage_intent": {"tool_id": "workspace", "capability": "inspect.structure"},
                "tool_decision": {"should_use_tool": False, "selected_tool": "workspace"},
                "response_style": {"intent_domain": "technical_engineering"},
            },
        },
    )

    restored = manager.get_active_thread("default")

    assert restored is not None
    assert restored["intent_domain"] == "technical_engineering"
    assert restored["response_depth"] == "standard"
    assert restored["conversation_phase"] == "execution"
    assert restored["reasoning_state"]["confidence_tier"] == "medium"
    assert restored["reasoning_state"]["route_decision"]["kind"] == "planning.debug"
    assert restored["reasoning_state"]["memory_context_used"][0]["label"] == "Workspace parser issue"
    assert restored["reasoning_state"]["tool_decision"]["selected_tool"] == "workspace"


def test_session_state_manager_round_trips_trainability_trace(tmp_path: Path) -> None:
    settings = AppSettings.from_repo_root(tmp_path)
    manager = SessionStateManager(settings=settings)

    manager.update_active_thread(
        session_id="default",
        prompt="debug the workspace parser",
        response={
            "mode": "planning",
            "kind": "planning.debug",
            "summary": "Let’s isolate the parser regression first.",
            "reasoning_state": {
                "current_path": "planning:planning.debug",
                "intent_domain": "technical_engineering",
                "response_depth": "standard",
                "conversation_phase": "execution",
                "confidence": 0.67,
                "confidence_tier": "medium",
            },
            "trainability_trace": {
                "schema_version": "1",
                "available_training_surfaces": [
                    "intent_domain_classification",
                    "tool_use_decision_support",
                ],
                "deterministic_surfaces": ["system_invariants"],
                "intent_domain_classification": {
                    "intent_domain": "technical_engineering",
                    "route_mode": "planning",
                },
                "route_recommendation_support": {"mode": "planning", "kind": "planning.debug"},
                "memory_relevance_ranking": {"selected_count": 1},
                "tool_use_decision_support": {"should_use_tool": False, "selected_tool": "workspace"},
                "response_style_selection": {"intent_domain": "technical_engineering"},
                "confidence_calibration_support": {"confidence_tier": "medium"},
                "rationale_summary": "Trace prepared for later offline labeling.",
            },
        },
    )

    restored = manager.get_active_thread("default")

    assert restored is not None
    assert restored["trainability_trace"]["schema_version"] == "1"
    assert restored["trainability_trace"]["intent_domain_classification"]["intent_domain"] == "technical_engineering"
    assert restored["trainability_trace"]["tool_use_decision_support"]["selected_tool"] == "workspace"


def test_session_profile_write_does_not_create_empty_db_session(tmp_path: Path) -> None:
    settings = AppSettings.from_repo_root(tmp_path)
    manager = SessionStateManager(settings=settings)

    payload = manager.set_interaction_profile(
        "desktop-empty",
        InteractionProfile(interaction_style="direct"),
    )

    assert payload["interaction_style"] == "direct"
    assert manager.get_interaction_profile("desktop-empty").interaction_style == "direct"
    assert manager.persistence_manager.sessions.get("desktop-empty") is None


def test_session_state_manager_round_trips_supervised_support_trace(tmp_path: Path) -> None:
    settings = AppSettings.from_repo_root(tmp_path)
    manager = SessionStateManager(settings=settings)

    manager.update_active_thread(
        session_id="default",
        prompt="create a roadmap for the parser work",
        response={
            "mode": "planning",
            "kind": "planning.migration",
            "summary": "Here is the roadmap I would use.",
            "reasoning_state": {
                "current_path": "planning:planning.migration",
                "intent_domain": "planning_strategy",
                "response_depth": "standard",
                "conversation_phase": "intake",
                "confidence": 0.73,
                "confidence_tier": "medium",
            },
            "supervised_support_trace": {
                "schema_version": "1",
                "enabled": True,
                "surfaces_with_examples": ["intent_domain_classification"],
                "recommendations": {
                    "intent_domain_classification": {
                        "surface": "intent_domain_classification",
                        "recommended_label": "planning_strategy",
                        "confidence": 0.84,
                        "applied": False,
                        "applied_reason": "Deterministic result already matched.",
                    }
                },
                "applied_surfaces": [],
                "deterministic_authority_preserved": True,
            },
        },
    )

    restored = manager.get_active_thread("default")

    assert restored is not None
    assert restored["supervised_support_trace"]["enabled"] is True
    assert (
        restored["supervised_support_trace"]["recommendations"]["intent_domain_classification"]["recommended_label"]
        == "planning_strategy"
    )


def test_controller_truncates_long_active_thread_fields(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)
    long_prompt = "create a migration plan for " + ("lumen " * 80)

    controller.ask(prompt=long_prompt, session_id="default")
    controller.ask(prompt="expand that further", session_id="default")

    active_thread = controller.current_session_thread("default")["active_thread"]

    assert len(active_thread["objective"]) <= 200
    assert len(active_thread["thread_summary"]) <= 280
    assert active_thread["thread_summary"]


def test_session_state_manager_skips_oversized_thread_state_on_load(tmp_path: Path, monkeypatch) -> None:
    settings = AppSettings.from_repo_root(tmp_path)
    manager = SessionStateManager(settings=settings)
    session_dir = settings.sessions_root / "default"
    session_dir.mkdir(parents=True, exist_ok=True)
    state_path = session_dir / manager.STATE_FILENAME
    state_path.write_text("{}", encoding="utf-8")

    original_stat = Path.stat

    def _fake_stat(path: Path, *args, **kwargs):
        result = original_stat(path, *args, **kwargs)
        if path == state_path:
            from os import stat_result

            values = list(result)
            values[6] = manager.max_state_bytes + 1
            return stat_result(values)
        return result

    monkeypatch.setattr(Path, "stat", _fake_stat)

    assert manager.get_active_thread("default") is None


def test_session_state_manager_compacts_oversized_payload_for_storage(tmp_path: Path, monkeypatch) -> None:
    settings = AppSettings.from_repo_root(tmp_path)
    manager = SessionStateManager(settings=settings)
    oversized_json = "x" * (manager.max_state_bytes + 1)
    call_count = {"value": 0}

    def _fake_safe_json_dumps(payload: object) -> str | None:
        call_count["value"] += 1
        if call_count["value"] < 3:
            return oversized_json
        return json.dumps(payload, indent=2)

    monkeypatch.setattr(manager, "_safe_json_dumps", _fake_safe_json_dumps)

    prepared, serialized = manager._prepare_payload_for_storage(
        {
            "session_id": "default",
            "mode": "tool",
            "kind": "tool.command_alias",
            "prompt": "run anh",
            "objective": "Execute tool task: run anh",
            "thread_summary": "ANH run",
            "summary": "ANH run",
            "interaction_profile": {},
            "pipeline_observability": {"trace": ["x"] * 100},
            "pipeline_trace": {"steps": ["x"] * 100},
            "detected_language": "en",
            "normalized_topic": "anh",
            "dominant_intent": "tool",
            "extracted_entities": [],
            "tool_context": {"huge": ["x"] * 100},
            "continuation_offer": {},
            "reasoning_state": {"huge": ["x"] * 100},
            "updated_at": "2026-04-02T00:00:00+00:00",
        }
    )

    assert prepared["pipeline_observability"] == {}
    assert prepared["pipeline_trace"] == {}
    assert isinstance(serialized, str)


def test_session_state_manager_reloads_compacted_trace_summaries_after_minimal_storage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = AppSettings.from_repo_root(tmp_path)
    manager = SessionStateManager(settings=settings)
    oversized_json = "x" * (manager.max_state_bytes + 1)
    call_count = {"value": 0}

    def _fake_safe_json_dumps(payload: object) -> str | None:
        call_count["value"] += 1
        if call_count["value"] < 3:
            return oversized_json
        return json.dumps(payload, indent=2)

    monkeypatch.setattr(manager, "_safe_json_dumps", _fake_safe_json_dumps)

    manager.update_active_thread(
        session_id="default",
        prompt="create a roadmap for the parser work",
        response={
            "mode": "planning",
            "kind": "planning.migration",
            "summary": "Here is the roadmap I would use.",
            "reasoning_state": {
                "current_path": "planning:planning.migration",
                "intent_domain": "planning_strategy",
                "response_depth": "standard",
                "conversation_phase": "intake",
                "confidence": 0.73,
                "confidence_tier": "medium",
            },
            "trainability_trace": {
                "schema_version": "1",
                "rationale_summary": "x" * 25_000,
            },
            "supervised_support_trace": {
                "schema_version": "1",
                "enabled": True,
                "recommendations": {
                    "intent_domain_classification": {
                        "surface": "intent_domain_classification",
                        "recommended_label": "planning_strategy",
                        "confidence": 0.84,
                        "applied": False,
                        "applied_reason": "x" * 25_000,
                    }
                },
            },
        },
    )

    restored = manager.get_active_thread("default")

    assert restored is not None
    assert restored["trainability_trace"]["compacted"] is True
    assert restored["trainability_trace"]["label"] == "trainability_trace"
    assert restored["supervised_support_trace"]["compacted"] is True
    assert restored["supervised_support_trace"]["label"] == "supervised_support_trace"


def test_active_thread_storage_stays_lightweight_and_non_recursive(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)
    session_id = "qa-storage"
    prompts = [
        "Hey Lumen!",
        "likewise, ive been thinking about mars",
        "I just seem to catch edge cases all day lol",
        "space",
        "biology",
        "tell me about ww2",
        "im feeling sad today",
        "how was your day?",
        "tell me about the moon",
    ]

    for prompt in prompts:
        controller.ask(prompt=prompt, session_id=session_id)

    state_path = tmp_path / "data" / "sessions" / session_id / "thread_state.json"
    payload = json.loads(state_path.read_text(encoding="utf-8"))

    assert state_path.stat().st_size < 100 * 1024
    assert payload["pipeline_trace"] == {}
    assert payload["pipeline_observability"]["compacted"] is True
    assert "active_thread" not in json.dumps(payload)
    assert "top_interaction_matches" not in json.dumps(payload)


def test_controller_restores_active_thread_continuity_and_traces_after_restart(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)
    controller.ask(prompt="create a migration plan for lumen", session_id="default")

    reloaded = AppController(repo_root=tmp_path)
    restored = reloaded.current_session_thread("default")["active_thread"]

    assert restored is not None
    assert restored["prompt"] == "create a migration plan for lumen"
    assert restored["trainability_trace"]["schema_version"] == "1"
    assert restored["supervised_support_trace"]["schema_version"] == "1"

    follow_up = reloaded.ask(prompt="expand that further", session_id="default")
    current = reloaded.current_session_thread("default")["active_thread"]

    assert follow_up["mode"] == "planning"
    assert current["objective"] == "Plan work for: create a migration plan for lumen"
    assert current["trainability_trace"]["schema_version"] == "1"
    assert current["supervised_support_trace"]["schema_version"] == "1"


def test_controller_restores_tool_repeat_continuity_after_restart(tmp_path: Path, monkeypatch) -> None:
    _copy_project_assets(tmp_path)
    first = AppController(repo_root=tmp_path)
    sample_csv = tmp_path / "data" / "examples" / "cf4_ga_cone_template.csv"

    def fake_run_tool(**kwargs):
        from lumen.tools.registry_types import ToolResult

        return ToolResult(
            status="ok",
            tool_id="anh",
            capability="spectral_dip_scan",
            summary="GA Local Analysis Kit run completed",
        )

    monkeypatch.setattr(first.tool_execution_service, "run_tool", fake_run_tool)
    first.ask(
        prompt="run anh",
        session_id="default",
        input_path=sample_csv,
    )

    reloaded = AppController(repo_root=tmp_path)
    monkeypatch.setattr(reloaded.tool_execution_service, "run_tool", fake_run_tool)
    restored = reloaded.current_session_thread("default")["active_thread"]

    assert restored is not None
    assert restored["tool_context"]["tool_id"] == "anh"
    assert restored["tool_context"]["capability"] == "spectral_dip_scan"
    assert restored["tool_context"]["input_path"] == str(sample_csv)

    follow_up = reloaded.ask(prompt="run that again", session_id="default")

    assert follow_up["mode"] == "tool"
    assert follow_up["resolved_prompt"] == "run anh"
    assert follow_up["resolution_strategy"] == "tool_repeat_shorthand"


def _copy_project_assets(repo_root: Path) -> None:
    source_root = Path(__file__).resolve().parents[2]
    for relative in [
        Path("tool_bundles"),
        Path("tools"),
        Path("data") / "examples",
        Path("lumen.toml.example"),
    ]:
        src = source_root / relative
        dest = repo_root / relative
        if src.is_dir():
            shutil.copytree(src, dest)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)

