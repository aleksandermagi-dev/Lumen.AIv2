from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def load_paper_text(*, input_path: Path | None = None, params: dict[str, Any] | None = None) -> str:
    params = params or {}
    direct_text = str(params.get("paper_text") or params.get("text") or "").strip()
    if direct_text:
        return direct_text
    if input_path is None or not input_path.exists():
        return ""
    suffix = input_path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return input_path.read_text(encoding="utf-8")
    if suffix == ".json":
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for key in ("paper_text", "text", "abstract", "content"):
                candidate = str(payload.get(key) or "").strip()
                if candidate:
                    return candidate
        if isinstance(payload, list):
            return "\n\n".join(str(item) for item in payload[:5])
    return ""


def summarize_paper(text: str) -> dict[str, Any]:
    cleaned = _clean_text(text)
    if not cleaned:
        return {
            "status": "error",
            "failure_category": "input_failure",
            "failure_reason": "Need paper text or an attached readable text file to summarize.",
            "runtime_diagnostics": {"runtime_ready": True, "input_ready": False},
        }
    sentences = _sentences(cleaned)
    abstract = _section(cleaned, ("abstract",))
    methods = _section(cleaned, ("methods", "methodology", "approach"))
    findings = _section(cleaned, ("results", "findings", "discussion"))
    summary_points = [item for item in [abstract, methods, findings] if item][:3]
    if not summary_points:
        summary_points = sentences[:3]
    return {
        "status": "ok",
        "summary_points": summary_points,
        "abstract_excerpt": abstract,
        "methods_excerpt": methods,
        "findings_excerpt": findings,
        "runtime_diagnostics": {"runtime_ready": True, "input_ready": True},
    }


def extract_methods(text: str) -> dict[str, Any]:
    cleaned = _clean_text(text)
    if not cleaned:
        return {
            "status": "error",
            "failure_category": "input_failure",
            "failure_reason": "Need paper text or an attached readable text file to extract methods.",
            "runtime_diagnostics": {"runtime_ready": True, "input_ready": False},
        }
    methods = _section(cleaned, ("methods", "methodology", "approach", "procedure"))
    if not methods:
        methods = " ".join(_sentences(cleaned)[:3])
    return {
        "status": "ok",
        "methods_excerpt": methods,
        "runtime_diagnostics": {"runtime_ready": True, "input_ready": True},
    }


def compare_papers(papers: list[str]) -> dict[str, Any]:
    usable = [_clean_text(paper) for paper in papers if _clean_text(paper)]
    if len(usable) < 2:
        return {
            "status": "error",
            "failure_category": "input_failure",
            "failure_reason": "Need at least two paper texts to compare.",
            "runtime_diagnostics": {"runtime_ready": True, "input_ready": False},
        }
    summaries = [summarize_paper(paper) for paper in usable[:2]]
    return {
        "status": "ok",
        "paper_count": 2,
        "comparison": [
            {
                "paper_index": index + 1,
                "summary_points": summary.get("summary_points", []),
                "methods_excerpt": summary.get("methods_excerpt") or "",
                "findings_excerpt": summary.get("findings_excerpt") or "",
            }
            for index, summary in enumerate(summaries)
        ],
        "runtime_diagnostics": {"runtime_ready": True, "input_ready": True},
    }


def search_papers(query: str, *, input_path: Path | None = None, params: dict[str, Any] | None = None) -> dict[str, Any]:
    query = str(query or "").strip()
    params = params or {}
    if input_path is None and not params.get("catalog"):
        return {
            "status": "error",
            "failure_category": "runtime_dependency_failure",
            "failure_reason": "No paper source is configured for this runtime.",
            "runtime_diagnostics": {
                "runtime_ready": False,
                "provider_status": "source_unavailable",
                "failure_hint": "Attach a local paper catalog or configure a paper source for search.",
            },
        }
    catalog: list[dict[str, Any]] = []
    if isinstance(params.get("catalog"), list):
        catalog = [item for item in params["catalog"] if isinstance(item, dict)]
    elif input_path is not None and input_path.exists() and input_path.suffix.lower() == ".json":
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            catalog = [item for item in payload if isinstance(item, dict)]
        elif isinstance(payload, dict):
            for key in ("papers", "items", "results"):
                candidate = payload.get(key)
                if isinstance(candidate, list):
                    catalog = [item for item in candidate if isinstance(item, dict)]
                    break
    if not catalog:
        return {
            "status": "error",
            "failure_category": "input_failure",
            "failure_reason": "Attached paper source could not be parsed into a catalog.",
            "runtime_diagnostics": {"runtime_ready": True, "input_ready": False},
        }
    query_terms = {term for term in re.findall(r"[a-z0-9]+", query.lower()) if len(term) > 2}
    ranked: list[dict[str, Any]] = []
    for item in catalog:
        haystack = " ".join(str(item.get(key) or "") for key in ("title", "abstract", "authors", "source")).lower()
        tokens = set(re.findall(r"[a-z0-9]+", haystack))
        overlap = len(query_terms & tokens)
        if overlap <= 0:
            continue
        ranked.append(
            {
                "title": str(item.get("title") or "Untitled paper").strip(),
                "source": str(item.get("source") or item.get("venue") or "local_catalog").strip(),
                "abstract": str(item.get("abstract") or "").strip(),
                "score": overlap,
            }
        )
    ranked.sort(key=lambda item: (-int(item["score"]), item["title"]))
    return {
        "status": "ok",
        "query": query,
        "results": ranked[:5],
        "runtime_diagnostics": {"runtime_ready": True, "input_ready": True},
    }


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [part.strip() for part in parts if part.strip()]


def _section(text: str, headings: tuple[str, ...]) -> str:
    lowered = text.lower()
    for heading in headings:
        match = re.search(rf"\b{re.escape(heading)}\b[:\s-]*", lowered)
        if not match:
            continue
        start = match.end()
        tail = text[start:]
        next_heading = re.search(r"\b(?:abstract|introduction|background|methods|methodology|approach|results|discussion|conclusion)\b[:\s-]*", tail, flags=re.IGNORECASE)
        excerpt = tail[: next_heading.start()] if next_heading else tail[:500]
        cleaned = _clean_text(excerpt)
        if cleaned:
            return cleaned[:500]
    return ""
