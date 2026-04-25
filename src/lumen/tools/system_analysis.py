from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class AnalyzedFile:
    path: str
    file_type: str
    line_count: int
    import_count: int
    function_count: int
    class_count: int
    dependency_names: list[str]


def analyze_repo_slice(repo_root: Path, *, target_path: str | None, depth: int | None) -> dict[str, object]:
    target = (repo_root / target_path).resolve() if target_path else (repo_root / "src").resolve()
    if not target.exists():
        raise FileNotFoundError(f"Target path does not exist: {target}")
    files = _collect_files(target, max_depth=depth)
    analyzed = [_analyze_file(path, repo_root) for path in files]
    components = [
        {
            "path": item.path,
            "file_type": item.file_type,
            "line_count": item.line_count,
            "function_count": item.function_count,
            "class_count": item.class_count,
        }
        for item in analyzed
    ]
    dependency_counter = Counter()
    for item in analyzed:
        dependency_counter.update(item.dependency_names)
    dependencies = [
        {"module": module, "count": count}
        for module, count in dependency_counter.most_common(20)
    ]
    hotspots = [
        {
            "path": item.path,
            "line_count": item.line_count,
            "import_count": item.import_count,
            "reason": _hotspot_reason(item),
        }
        for item in analyzed
        if _hotspot_reason(item)
    ]
    risks = _derive_risks(analyzed)
    return {
        "target_path": str(target),
        "file_count": len(analyzed),
        "components": components,
        "dependencies": dependencies,
        "hotspots": hotspots,
        "risks": risks,
    }


def suggest_refactors(analysis: dict[str, object], *, goal: str) -> dict[str, object]:
    recommendations: list[dict[str, object]] = []
    components = list(analysis.get("components") or [])
    hotspots = list(analysis.get("hotspots") or [])
    risks = list(analysis.get("risks") or [])
    if goal == "extract_helpers":
        for hotspot in hotspots[:3]:
            recommendations.append(
                {
                    "path": hotspot["path"],
                    "change": "extract support helpers",
                    "reason": "Large file or high import pressure increases change risk.",
                }
            )
    elif goal == "reduce_coupling":
        for dependency in list(analysis.get("dependencies") or [])[:3]:
            recommendations.append(
                {
                    "module": dependency["module"],
                    "change": "reduce fan-in or isolate shared helpers",
                    "reason": "Frequently imported modules are coupling hotspots.",
                }
            )
    elif goal == "clarify_authority":
        for risk in risks:
            if "authority" in str(risk.get("reason", "")).lower():
                recommendations.append(
                    {
                        "path": risk["path"],
                        "change": "document or isolate authority seam",
                        "reason": risk["reason"],
                    }
                )
    elif goal == "improve_tests":
        for component in sorted(components, key=lambda item: int(item["line_count"]), reverse=True)[:3]:
            recommendations.append(
                {
                    "path": component["path"],
                    "change": "add seam-focused regression tests",
                    "reason": "Large components are likely to benefit from contract coverage.",
                }
            )
    return {
        "recommendations": recommendations,
        "rationale": f"Generated bounded refactor suggestions for goal '{goal}'.",
        "risk_level": "medium" if recommendations else "low",
        "safe_sequence": [
            "Lock behavior with targeted tests.",
            "Extract or isolate one bounded seam at a time.",
            "Re-run focused regression tests after each change.",
        ],
    }


def generate_docs(repo_root: Path, *, target_path: str, doc_type: str, include_public_interfaces: bool) -> dict[str, object]:
    target = (repo_root / target_path).resolve()
    if not target.exists():
        raise FileNotFoundError(f"Target path does not exist: {target}")
    if doc_type == "bundle_manifest" and target.name == "manifest.json":
        import json

        payload = json.loads(target.read_text(encoding="utf-8"))
        sections = [
            {"heading": "Bundle", "content": payload.get("name") or payload.get("id")},
            {"heading": "Description", "content": payload.get("description", "")},
            {
                "heading": "Capabilities",
                "content": [
                    {
                        "id": capability.get("id"),
                        "description": capability.get("description", ""),
                    }
                    for capability in payload.get("capabilities", [])
                ],
            },
        ]
    else:
        analysis = analyze_repo_slice(repo_root, target_path=target_path, depth=2)
        sections = [
            {"heading": "Overview", "content": f"Analyzed {analysis['file_count']} files under {target_path}."},
            {"heading": "Hotspots", "content": analysis["hotspots"]},
            {"heading": "Risks", "content": analysis["risks"]},
        ]
        if include_public_interfaces:
            public_items = []
            for file_path in _collect_files(target, max_depth=2):
                if file_path.suffix != ".py":
                    continue
                source = file_path.read_text(encoding="utf-8", errors="ignore")
                tree = ast.parse(source)
                exports = [
                    node.name
                    for node in tree.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                    and not node.name.startswith("_")
                ]
                if exports:
                    public_items.append({"path": str(file_path.relative_to(repo_root)), "exports": exports})
            sections.append({"heading": "Public Interfaces", "content": public_items})
    summary = f"Generated {doc_type} documentation for {target_path}."
    return {"title": target.name, "sections": sections, "summary": summary}


def render_architecture_map(analysis: dict[str, object]) -> str:
    lines = ["Architecture Map", f"Files analyzed: {analysis.get('file_count', 0)}"]
    for hotspot in analysis.get("hotspots") or []:
        lines.append(f"- {hotspot['path']}: {hotspot['reason']}")
    return "\n".join(lines)


def render_docs_markdown(doc: dict[str, object]) -> str:
    lines = [f"# {doc.get('title', 'Generated Docs')}", "", str(doc.get("summary", "")), ""]
    for section in doc.get("sections") or []:
        lines.append(f"## {section['heading']}")
        lines.append(str(section.get("content")))
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _collect_files(root: Path, *, max_depth: int | None) -> list[Path]:
    if root.is_file():
        return [root]
    files: list[Path] = []
    root_depth = len(root.parts)
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix == ".pyc":
            continue
        if max_depth is not None and len(path.parts) - root_depth > max_depth:
            continue
        files.append(path)
    return files


def _analyze_file(path: Path, repo_root: Path) -> AnalyzedFile:
    source = path.read_text(encoding="utf-8", errors="ignore")
    line_count = len(source.splitlines())
    import_count = function_count = class_count = 0
    dependency_names: list[str] = []
    if path.suffix == ".py":
        try:
            tree = ast.parse(source)
        except SyntaxError:
            tree = None
        if tree is not None:
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    import_count += len(node.names)
                    dependency_names.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    import_count += len(node.names)
                    dependency_names.append(node.module or "")
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    function_count += 1
                elif isinstance(node, ast.ClassDef):
                    class_count += 1
    return AnalyzedFile(
        path=str(path.relative_to(repo_root)),
        file_type=path.suffix or "unknown",
        line_count=line_count,
        import_count=import_count,
        function_count=function_count,
        class_count=class_count,
        dependency_names=[name for name in dependency_names if name],
    )


def _hotspot_reason(item: AnalyzedFile) -> str | None:
    reasons: list[str] = []
    if item.line_count >= 250:
        reasons.append("large file")
    if item.import_count >= 12:
        reasons.append("high import fan-in")
    if "interaction_service" in item.path or "router" in item.path:
        reasons.append("authority-sensitive seam")
    return ", ".join(reasons) if reasons else None


def _derive_risks(analyzed: list[AnalyzedFile]) -> list[dict[str, object]]:
    risks: list[dict[str, object]] = []
    for item in analyzed:
        if item.line_count >= 250:
            risks.append({"path": item.path, "reason": "Large module is harder to change safely."})
        if "router" in item.path or "interaction_service" in item.path:
            risks.append({"path": item.path, "reason": "Authority overlap or orchestration drift risk."})
    return risks
