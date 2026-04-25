from pathlib import Path

from lumen.app.controller import AppController


def _visible_text(response: dict[str, object]) -> str:
    return str(
        response.get("reply")
        or response.get("user_facing_answer")
        or response.get("summary")
        or ""
    ).strip()


def test_memory_loop_recalls_personal_preference_naturally(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)

    controller.ask(
        prompt="remember this about me: from now on keep it brief with me",
        session_id="default",
    )

    response = controller.ask(
        prompt="what do you remember about my preferences?",
        session_id="default",
    )

    text = _visible_text(response).lower()

    assert response["memory_retrieval"]["selected"]
    assert "brief" in text
    assert "local knowledge" not in text
    assert "validation plan" not in text


def test_memory_loop_retrieves_project_memory_for_recall_prompt(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)

    controller.ask(prompt="create a migration plan for lumen routing", session_id="default")

    response = controller.ask(
        prompt="what do you remember about the lumen routing project?",
        session_id="default",
    )

    text = _visible_text(response).lower()

    assert response["memory_retrieval"]["selected"]
    assert "routing" in text or "migration" in text


def test_memory_loop_uses_relation_backed_graph_recall(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)

    controller.ask(prompt="design the routing architecture", session_id="default")
    notes = controller.list_research_notes(session_id="default")
    note_path = Path(notes["research_notes"][0]["note_path"])

    controller.promote_research_note(
        note_path=note_path,
        artifact_type="finding",
        title="routing decision trail",
        promotion_reason="keep a durable routing record",
    )

    response = controller.ask(
        prompt="what do you remember about routing decisions?",
        session_id="default",
    )

    text = _visible_text(response).lower()
    graph_matches = controller.search_memory_graph_nodes("routing decision", limit=5)

    assert graph_matches
    assert any(match["name"] == "routing decision trail" for match in graph_matches)
    assert "routing" in text


def test_memory_loop_filters_irrelevant_memory_when_no_good_match_exists(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)

    controller.ask(
        prompt="remember this about me: from now on keep it brief with me",
        session_id="default",
    )
    controller.ask(prompt="create a migration plan for lumen routing", session_id="default")

    response = controller.ask(
        prompt="what do you remember about black holes?",
        session_id="default",
    )

    text = _visible_text(response).lower()

    assert "brief" not in text
    assert "routing" not in text
    assert "migration" not in text
    assert "don't have a strong memory match" in text
    assert response["memory_retrieval"]["diagnostics"]["reason"] in {
        "insufficient_dominance",
        "weak_focus_overlap",
    }


def test_memory_loop_reuses_project_memory_for_return_prompt(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)

    controller.ask(prompt="create a migration plan for lumen routing", session_id="default")

    response = controller.ask(
        prompt="where were we on the lumen routing project?",
        session_id="default",
    )

    text = _visible_text(response).lower()

    assert response["memory_retrieval"]["project_return_prompt"] is True
    assert response["memory_retrieval"]["selected"]
    assert response["memory_retrieval"]["project_reply_hint"]
    assert "routing" in text or "modular" in text or "migration" in text


def test_memory_loop_reuses_project_memory_for_addressed_return_prompt(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)

    controller.ask(prompt="create a migration plan for lumen routing", session_id="default")

    response = controller.ask(
        prompt="Hey Lumen, where were we on the lumen routing project?",
        session_id="default",
    )

    text = _visible_text(response).lower()

    assert response["memory_retrieval"]["project_return_prompt"] is True
    assert response["memory_retrieval"]["selected"]
    assert response["memory_retrieval"]["diagnostics"]["focus_resolution"]["reason"] == "wrapper_stripped"
    assert "routing" in text or "migration" in text


def test_memory_loop_withholds_project_return_hint_when_nearby_graph_matches_are_ambiguous(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)

    controller.graph_memory_manager.create_entities(
        [
            {"name": "routing review notes from the first pass", "entity_type": "project"},
            {"name": "routing assumptions summary for the first pass", "entity_type": "project"},
        ]
    )
    controller.graph_memory_manager.add_observations(
        [
            {
                "entity_name": "routing review notes from the first pass",
                "entity_type": "project",
                "content": "routing cleanup notes focused on module boundaries",
            },
            {
                "entity_name": "routing assumptions summary for the first pass",
                "entity_type": "project",
                "content": "routing cleanup notes focused on helper boundaries",
            },
        ]
    )

    response = controller.ask(
        prompt="where were we on routing?",
        session_id="default",
    )

    assert response["memory_retrieval"]["project_return_prompt"] is True
    assert response["memory_retrieval"]["project_reply_hint"] is None
    assert "closest thread i have" not in _visible_text(response).lower()
