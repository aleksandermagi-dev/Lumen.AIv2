from __future__ import annotations

import json
from pathlib import Path

from lumen.content_generation.formatters import format_draft_text, format_drafts_markdown, format_ideas_text
from lumen.content_generation.models import ContentArtifactPackage, GeneratedContentDraft, GeneratedContentVariant, GeneratedIdea


class ContentArtifactWriter:
    def write_ideas_package(self, *, outputs_dir: Path, ideas: list[GeneratedIdea]) -> ContentArtifactPackage:
        outputs_dir.mkdir(parents=True, exist_ok=True)
        text_path = outputs_dir / "ideas.txt"
        json_path = outputs_dir / "ideas.json"
        manifest_path = outputs_dir / "artifact_manifest.json"
        text_path.write_text(format_ideas_text(ideas), encoding="utf-8")
        json_path.write_text(json.dumps([item.to_mapping() for item in ideas], indent=2), encoding="utf-8")
        manifest = {
            "package_type": "ideas",
            "artifacts": [
                {"name": "ideas.txt", "path": str(text_path)},
                {"name": "ideas.json", "path": str(json_path)},
            ],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return ContentArtifactPackage(
            root_dir=str(outputs_dir),
            artifact_paths=[str(text_path), str(json_path)],
            manifest_path=str(manifest_path),
            summary_path=str(text_path),
        )

    def write_batch_package(
        self,
        *,
        outputs_dir: Path,
        items: list[GeneratedContentDraft],
        variants: list[GeneratedContentVariant] | None = None,
    ) -> ContentArtifactPackage:
        outputs_dir.mkdir(parents=True, exist_ok=True)
        items_dir = outputs_dir / "items"
        items_dir.mkdir(parents=True, exist_ok=True)
        artifact_paths: list[str] = []
        manifest_records: list[dict[str, str]] = []

        for item in items:
            target_dir = items_dir / item.id
            target_dir.mkdir(parents=True, exist_ok=True)
            script_path = target_dir / "script.txt"
            caption_path = target_dir / "caption.txt"
            json_path = target_dir / "item.json"
            script_path.write_text("\n".join([item.hook, *item.script_lines]).strip() + "\n", encoding="utf-8")
            caption_text = item.caption.strip()
            if item.hashtags:
                caption_text = f"{caption_text}\n\n{' '.join(item.hashtags)}"
            caption_path.write_text(caption_text.strip() + "\n", encoding="utf-8")
            json_path.write_text(json.dumps(item.to_mapping(), indent=2), encoding="utf-8")
            for path in (script_path, caption_path, json_path):
                artifact_paths.append(str(path))
                manifest_records.append({"name": path.name, "path": str(path)})

        if variants:
            variants_dir = outputs_dir / "variants"
            variants_dir.mkdir(parents=True, exist_ok=True)
            for item in variants:
                target_dir = variants_dir / f"{item.platform}_{item.id}"
                target_dir.mkdir(parents=True, exist_ok=True)
                text_path = target_dir / "variant.txt"
                json_path = target_dir / "variant.json"
                text_path.write_text(format_draft_text(item), encoding="utf-8")
                json_path.write_text(json.dumps(item.to_mapping(), indent=2), encoding="utf-8")
                for path in (text_path, json_path):
                    artifact_paths.append(str(path))
                    manifest_records.append({"name": path.name, "path": str(path)})

        markdown_path = outputs_dir / "batch.md"
        summary_path = outputs_dir / "batch_summary.txt"
        payload_path = outputs_dir / "batch.json"
        manifest_path = outputs_dir / "artifact_manifest.json"
        markdown_path.write_text(format_drafts_markdown(items + list(variants or [])), encoding="utf-8")
        summary_path.write_text(self._summary_text(items=items, variants=variants or []), encoding="utf-8")
        payload_path.write_text(
            json.dumps(
                {
                    "items": [item.to_mapping() for item in items],
                    "variants": [item.to_mapping() for item in variants or []],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        artifact_paths.extend([str(markdown_path), str(summary_path), str(payload_path)])
        manifest_records.extend(
            [
                {"name": "batch.md", "path": str(markdown_path)},
                {"name": "batch_summary.txt", "path": str(summary_path)},
                {"name": "batch.json", "path": str(payload_path)},
            ]
        )
        manifest_path.write_text(
            json.dumps({"package_type": "content_batch", "artifacts": manifest_records}, indent=2),
            encoding="utf-8",
        )
        return ContentArtifactPackage(
            root_dir=str(outputs_dir),
            artifact_paths=artifact_paths,
            manifest_path=str(manifest_path),
            summary_path=str(summary_path),
        )

    @staticmethod
    def _summary_text(
        *,
        items: list[GeneratedContentDraft],
        variants: list[GeneratedContentVariant],
    ) -> str:
        lines = [f"Generated {len(items)} master drafts."]
        if variants:
            lines.append(f"Generated {len(variants)} platform variants.")
        for item in items[:5]:
            lines.append(f"- {item.id}: {item.topic} [{item.safety.decision}]")
        return "\n".join(lines) + "\n"
