from pathlib import Path

from lumen.app.settings import AppSettings
from lumen.memory.graph_memory import GraphMemoryManager


def test_graph_memory_manager_creates_searches_and_opens_nodes(tmp_path: Path) -> None:
    settings = AppSettings.from_repo_root(tmp_path)
    manager = GraphMemoryManager(settings=settings)

    created = manager.create_entities(
        [
            {"name": "Lumen", "entity_type": "project"},
            {"name": "Routing", "entity_type": "system"},
        ]
    )
    assert len(created) == 2

    relations = manager.create_relations(
        [
            {"source": "Lumen", "source_type": "project", "relation_type": "depends_on", "target": "Routing", "target_type": "system"}
        ]
    )
    assert len(relations) == 1

    observations = manager.add_observations(
        [
            {
                "node_name": "Lumen",
                "entity_type": "project",
                "content": "Lumen is an active research-partner project.",
                "source_type": "test",
            }
        ]
    )
    assert len(observations) == 1

    matches = manager.search_nodes("research partner lumen", limit=5)
    assert matches
    assert matches[0]["name"] == "Lumen"

    opened = manager.open_nodes(names=["Lumen"])
    assert len(opened) == 1
    assert opened[0]["name"] == "Lumen"
    assert opened[0]["relations_out"][0]["relation_type"] == "depends_on"
    assert opened[0]["observations"][0]["content"] == "Lumen is an active research-partner project."

    graph = manager.read_graph(limit=10)
    assert graph["node_count"] == 2
    assert graph["relation_count"] == 1
    assert graph["observation_count"] == 1
