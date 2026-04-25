from __future__ import annotations

from lumen.reasoning.response_models import ResearchResponse


class DatasetWorkflowSupport:
    """Provides bounded local guidance for dataset licensing and source workflow questions."""

    @staticmethod
    def build_response(*, prompt: str) -> dict[str, object]:
        summary = "Here’s the safest bounded dataset workflow I’d use."
        findings = [
            "Check the dataset license first and confirm it matches your intended training or publication use.",
            "Prefer established dataset repositories such as OpenML, Hugging Face Datasets, or Kaggle when they clearly state provenance and usage terms.",
            "Use Google Dataset Search as a discovery layer, but still verify the original source, documentation, and license before using the data.",
        ]
        recommendation = (
            "If you already have a local dataset or file, I can help inspect it or reason about a bounded analysis workflow next."
        )
        response = ResearchResponse(
            mode="research",
            kind="research.dataset_guidance",
            summary=summary,
            findings=findings,
            recommendation=recommendation,
        ).to_dict()
        response["user_facing_answer"] = "\n".join(
            [
                summary,
                "",
                "- Check the dataset license before using it for training, publication, or redistribution.",
                "- Prefer repositories with clear provenance, docs, and usage terms such as OpenML, Hugging Face Datasets, and Kaggle.",
                "- Treat Google Dataset Search as discovery help only, then verify the original source directly.",
                "",
                recommendation,
            ]
        ).strip()
        response["capability_status"] = {
            "domain_id": "dataset_analysis",
            "status": "bounded",
            "details": "Dataset workflow guidance is advisory and local-data bounded, not a guaranteed external crawler or licensing authority.",
        }
        response["capability_guidance"] = {
            "guidance_type": "dataset_license_and_source",
            "advisory_only": True,
            "source_prompt": prompt,
        }
        return response
