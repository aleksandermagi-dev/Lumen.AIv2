from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lumen.app.settings import AppSettings
from lumen.knowledge.knowledge_service import KnowledgeService
from lumen.memory.graph_memory import GraphMemoryManager


@dataclass(slots=True)
class KnowledgeOps:
    knowledge_service: KnowledgeService
    graph_memory_manager: GraphMemoryManager

    @classmethod
    def from_repo_root(cls, repo_root: Path) -> "KnowledgeOps":
        settings = AppSettings.from_repo_root(repo_root)
        return cls(
            knowledge_service=KnowledgeService.from_path(settings.persistence_db_path),
            graph_memory_manager=GraphMemoryManager(settings=settings),
        )

    def link(self, items: list[str], relation_hint: str | None = None) -> dict[str, Any]:
        normalized = [item.strip() for item in items if str(item).strip()]
        links: list[dict[str, object]] = []
        unlinked: list[str] = []
        for index, left in enumerate(normalized):
            left_lookup = self.knowledge_service.lookup(left)
            linked = False
            for right in normalized[index + 1 :]:
                right_lookup = self.knowledge_service.lookup(right)
                if left_lookup and right_lookup:
                    relation = self.knowledge_service._relation_between(  # noqa: SLF001
                        left_lookup.primary.id, right_lookup.primary.id
                    )
                    if relation and (relation_hint is None or relation == relation_hint):
                        links.append(
                            {
                                "source": left_lookup.primary.title,
                                "target": right_lookup.primary.title,
                                "relation_type": relation,
                                "evidence": "knowledge_db",
                            }
                        )
                        linked = True
                        continue
                overlap = _token_overlap(left, right)
                if overlap >= 0.5:
                    links.append(
                        {
                            "source": left,
                            "target": right,
                            "relation_type": relation_hint or "related_to",
                            "evidence": "inferred_overlap",
                        }
                    )
                    linked = True
            if not linked:
                unlinked.append(left)
        return {"links": links, "unlinked_items": unlinked, "confidence": "high" if links else "low"}

    def find_paths(self, source: str, target: str, *, max_hops: int) -> dict[str, Any]:
        adjacency = self._build_adjacency()
        source_key = self._resolve_node_key(source, adjacency)
        target_key = self._resolve_node_key(target, adjacency)
        if source_key is None or target_key is None:
            return {"paths": [], "best_path": [], "path_count": 0, "confidence": "low"}
        queue: deque[tuple[str, list[dict[str, object]]]] = deque([(source_key, [{"node": source_key}])])
        paths: list[list[dict[str, object]]] = []
        while queue:
            current, path = queue.popleft()
            if len(path) - 1 > max_hops:
                continue
            if current == target_key:
                paths.append(path)
                continue
            seen_nodes = {str(step["node"]) for step in path}
            for edge in adjacency.get(current, []):
                next_node = str(edge["target"])
                if next_node in seen_nodes:
                    continue
                queue.append(
                    (
                        next_node,
                        path + [{"relation_type": edge["relation_type"], "node": next_node, "evidence": edge["evidence"]}],
                    )
                )
        best = min(paths, key=len) if paths else []
        return {"paths": paths, "best_path": best, "path_count": len(paths), "confidence": "high" if paths else "low"}

    def cluster(self, items: list[str], strategy: str) -> dict[str, Any]:
        normalized = [item.strip() for item in items if str(item).strip()]
        buckets: dict[str, list[str]] = defaultdict(list)
        outliers: list[str] = []
        for item in normalized:
            tokens = _tokens(item)
            if not tokens:
                outliers.append(item)
                continue
            key = sorted(tokens)[0] if strategy == "topic" else max(tokens, key=len)
            buckets[key].append(item)
        clusters = [{"cluster_key": key, "items": members} for key, members in buckets.items() if len(members) >= 2]
        for members in buckets.values():
            if len(members) == 1:
                outliers.extend(members)
        return {"clusters": clusters, "outliers": outliers, "confidence": "medium" if clusters else "low"}

    def contradictions(self, claims: list[str], strictness: str) -> dict[str, Any]:
        contradictions: list[dict[str, object]] = []
        tensions: list[dict[str, object]] = []
        unresolved: list[str] = []
        for index, left in enumerate(claims):
            found = False
            for right in claims[index + 1 :]:
                if _is_contradiction(left, right):
                    contradictions.append({"left": left, "right": right, "confidence": strictness})
                    found = True
                elif _token_overlap(left, right) >= 0.4:
                    tensions.append({"left": left, "right": right, "reason": "Competing overlap without direct contradiction."})
                    found = True
            if not found:
                unresolved.append(left)
        return {"contradictions": contradictions, "tensions": tensions, "unresolved": unresolved}

    def _build_adjacency(self) -> dict[str, list[dict[str, str]]]:
        adjacency: dict[str, list[dict[str, str]]] = defaultdict(list)
        with self.knowledge_service.db.connect() as connection:
            rows = connection.execute(
                """
                SELECT s.title AS source_title, r.relation_type, t.title AS target_title
                FROM knowledge_relationships r
                JOIN knowledge_entries s ON s.id = r.source_entry_id
                JOIN knowledge_entries t ON t.id = r.target_entry_id
                """
            ).fetchall()
        for row in rows:
            adjacency[str(row["source_title"])].append(
                {"target": str(row["target_title"]), "relation_type": str(row["relation_type"]), "evidence": "knowledge_db"}
            )
        graph = self.graph_memory_manager.read_graph(limit=200)
        for node in graph.get("nodes") or []:
            source = str(node["name"])
            for relation in node.get("relations_out") or []:
                adjacency[source].append(
                    {
                        "target": str(relation["target_name"]),
                        "relation_type": str(relation["relation_type"]),
                        "evidence": "graph_memory",
                    }
                )
        return adjacency

    def _resolve_node_key(self, query: str, adjacency: dict[str, list[dict[str, str]]]) -> str | None:
        lookup = self.knowledge_service.lookup(query)
        if lookup is not None:
            return lookup.primary.title
        matches = self.graph_memory_manager.search_nodes(query, limit=1)
        if matches:
            return str(matches[0]["name"])
        lowered = query.strip().lower()
        for candidate in adjacency:
            if candidate.lower() == lowered:
                return candidate
        return None


def _tokens(text: str) -> set[str]:
    return {token for token in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if len(token) > 2}


def _token_overlap(left: str, right: str) -> float:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(len(left_tokens), len(right_tokens))


def _is_contradiction(left: str, right: str) -> bool:
    left_lower = left.lower()
    right_lower = right.lower()
    negations = (" not ", " never ", " no ", " cannot ", " can't ")
    if any(token in f" {left_lower} " for token in negations) != any(token in f" {right_lower} " for token in negations):
        return bool(_tokens(left) & _tokens(right))
    opposite_pairs = [("increase", "decrease"), ("possible", "impossible"), ("safe", "dangerous")]
    for left_word, right_word in opposite_pairs:
        if (left_word in left_lower and right_word in right_lower) or (right_word in left_lower and left_word in right_lower):
            return True
    return False
