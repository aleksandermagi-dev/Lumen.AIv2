from __future__ import annotations

from dataclasses import dataclass

from lumen.providers.base import ModelProvider
from lumen.providers.models import InferenceRequest, InferenceResult


@dataclass(slots=True)
class HostedInferenceDecision:
    use_hosted_inference: bool
    reason: str


class InferenceService:
    """Thin adapter between runtime routing decisions and the configured model provider."""

    def __init__(self, model_provider: ModelProvider | None = None):
        self.model_provider = model_provider

    def evaluate_hosted_research(
        self,
        *,
        route,
        validation_context,
    ) -> HostedInferenceDecision:
        if self.model_provider is None:
            return HostedInferenceDecision(False, "No model provider is configured.")
        if self.model_provider.provider_id == "local":
            return HostedInferenceDecision(False, "Configured provider is local-only.")
        provider_ready, provider_reason = self._provider_is_ready()
        if not provider_ready:
            return HostedInferenceDecision(False, provider_reason)
        if route.mode != "research" or route.kind not in {"research.general", "research.summary"}:
            return HostedInferenceDecision(False, "Hosted inference is only used for general or explanatory research turns.")
        if route.source in {"active_thread", "recent_interaction", "active_thread_bias", "active_intent", "active_topic", "active_entities"}:
            return HostedInferenceDecision(False, "A stronger local continuity route exists, so hosted fallback is not used.")
        assistant_context = getattr(validation_context, "assistant_context", None)
        top_matches = list(getattr(assistant_context, "top_matches", []) or []) if assistant_context is not None else []
        top_interaction_matches = list(getattr(assistant_context, "top_interaction_matches", []) or []) if assistant_context is not None else []
        active_thread = getattr(assistant_context, "active_thread", None) if assistant_context is not None else None
        if top_matches or top_interaction_matches or active_thread:
            return HostedInferenceDecision(False, "Local context exists, so hosted fallback is not needed.")
        return HostedInferenceDecision(True, "Sparse local context for a general or explanatory research turn.")

    def infer_research_reply(
        self,
        *,
        prompt: str,
        session_id: str,
        interaction_profile,
        validation_context,
    ) -> InferenceResult:
        if self.model_provider is None:
            raise RuntimeError("No model provider is configured.")
        instructions = self._instructions(interaction_profile=interaction_profile)
        assistant_context = getattr(validation_context, "assistant_context", None)
        local_context_summary = None
        if assistant_context is not None:
            local_context_summary = str(getattr(assistant_context, "local_context_summary", "") or "").strip() or None
        input_text = prompt
        if local_context_summary:
            input_text = (
                f"User question: {prompt}\n\n"
                f"Local context summary: {local_context_summary}\n\n"
                "Use the local context when relevant, but answer the user directly."
            )
        request = InferenceRequest(
            model=None,
            instructions=instructions,
            input_text=input_text,
            metadata={
                "session_id": session_id,
                "interaction_style": getattr(interaction_profile, "interaction_style", "conversational"),
                "reasoning_depth": getattr(interaction_profile, "reasoning_depth", "normal"),
            },
            temperature=0.4,
            max_output_tokens=900,
        )
        return self.model_provider.infer(request)

    def evaluate_hosted_writing(self) -> HostedInferenceDecision:
        if self.model_provider is None:
            return HostedInferenceDecision(False, "No model provider is configured.")
        if self.model_provider.provider_id == "local":
            return HostedInferenceDecision(False, "Configured provider is local-only.")
        provider_ready, provider_reason = self._provider_is_ready()
        if not provider_ready:
            return HostedInferenceDecision(False, provider_reason)
        return HostedInferenceDecision(True, "Hosted provider is available for bounded writing/editing workflows.")

    def infer_writing_reply(
        self,
        *,
        prompt: str,
        session_id: str,
        interaction_profile,
        workflow: dict[str, object],
    ) -> InferenceResult:
        if self.model_provider is None:
            raise RuntimeError("No model provider is configured.")
        from lumen.reasoning.writing_workflow_support import WritingWorkflowSupport

        request = InferenceRequest(
            model=None,
            instructions=WritingWorkflowSupport.hosted_instructions(
                workflow=workflow,
                interaction_style=getattr(interaction_profile, "interaction_style", "default"),
                reasoning_depth=getattr(interaction_profile, "reasoning_depth", "normal"),
            ),
            input_text=prompt,
            metadata={
                "session_id": session_id,
                "workflow": str(workflow.get("workflow") or ""),
                "interaction_style": getattr(interaction_profile, "interaction_style", "conversational"),
            },
            temperature=0.35,
            max_output_tokens=900,
        )
        return self.model_provider.infer(request)

    def _provider_is_ready(self) -> tuple[bool, str]:
        provider = self.model_provider
        if provider is None:
            return False, "No model provider is configured."
        has_api_key = getattr(provider, "has_api_key", None)
        if callable(has_api_key) and not bool(has_api_key()):
            return False, "Hosted provider is missing its API key."
        default_model = getattr(provider, "default_model", None)
        if default_model is not None and not str(default_model or "").strip():
            return False, "Hosted provider is missing its default model."
        return True, "Hosted provider is configured."

    @staticmethod
    def _instructions(*, interaction_profile) -> str:
        style = str(getattr(interaction_profile, "interaction_style", "default") or "default").strip().lower()
        if style == "conversational":
            style = "collab"
        depth = str(getattr(interaction_profile, "reasoning_depth", "normal") or "normal").strip().lower()
        base = (
            "You are Lumen. Answer clearly, truthfully, and directly. "
            "State uncertainty plainly when needed. Do not roleplay or pretend personal knowledge. "
            "Keep the answer useful for a local private desktop chat."
        )
        if style == "direct":
            style_instruction = "Keep the response concise and action-forward."
        elif style == "collab":
            style_instruction = "Keep the tone warm, natural, and collaborative."
        else:
            style_instruction = "Keep the tone balanced, natural, and grounded."
        if depth == "light":
            depth_instruction = "Prefer a light answer unless the user asks for more."
        elif depth == "deep":
            depth_instruction = "Go deeper when the question benefits from it, but stay clear."
        else:
            depth_instruction = "Keep the answer informative without becoming overlong."
        return f"{base} {style_instruction} {depth_instruction}"
