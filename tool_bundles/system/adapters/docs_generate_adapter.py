from __future__ import annotations

from pathlib import Path

from lumen.tools.registry_types import Artifact, BundleManifest, ToolRequest, ToolResult
from lumen.tools.run_utils import build_run_dir, ensure_outputs_dir, write_json_artifact, write_text_artifact
from lumen.tools.system_analysis import generate_docs, render_docs_markdown


class DocsGenerateAdapter:
    def __init__(self, *, manifest: BundleManifest, repo_root: Path):
        self.manifest = manifest
        self.repo_root = repo_root

    def execute(self, request: ToolRequest) -> ToolResult:
        target_path = str(request.params.get("target_path") or "").strip()
        doc_type = str(request.params.get("doc_type") or "module_overview").strip()
        include_public_interfaces = bool(request.params.get("include_public_interfaces"))
        if not target_path:
            raise ValueError("generate.docs requires 'target_path'")
        run_dir = build_run_dir(repo_root=self.repo_root, request=request)
        outputs_dir = ensure_outputs_dir(run_dir)
        payload = generate_docs(
            self.repo_root,
            target_path=target_path,
            doc_type=doc_type,
            include_public_interfaces=include_public_interfaces,
        )
        json_artifact = write_json_artifact(outputs_dir, "generated_docs.json", payload)
        md_artifact = write_text_artifact(outputs_dir, "generated_docs.md", render_docs_markdown(payload))
        return ToolResult(
            status="ok",
            tool_id=request.tool_id,
            capability=request.capability,
            summary=payload["summary"],
            structured_data=payload,
            artifacts=[
                Artifact(name="generated_docs.json", path=json_artifact, media_type="application/json"),
                Artifact(name="generated_docs.md", path=md_artifact, media_type="text/markdown"),
            ],
            logs=[f"Generated {doc_type} docs for {target_path}."],
            provenance={"repo_root": str(self.repo_root), "session_id": request.session_id},
            run_dir=run_dir,
        )
