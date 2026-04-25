from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib import error, request as urllib_request

from lumen.providers.base import ModelProvider
from lumen.providers.models import InferenceRequest, InferenceResult, ProviderCapabilities


@dataclass(slots=True)
class OpenAIResponsesProvider(ModelProvider):
    deployment_mode_value: str
    api_base: str | None = None
    default_model: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    timeout_seconds: int = 30

    @property
    def provider_id(self) -> str:
        return "openai_responses"

    @property
    def deployment_mode(self) -> str:
        return self.deployment_mode_value

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_sync=True,
            supports_streaming=True,
            supports_async=True,
            supports_background=True,
        )

    def has_api_key(self) -> bool:
        return bool(os.environ.get(self.api_key_env))

    def build_payload(self, request: InferenceRequest) -> dict[str, Any]:
        return {
            "model": request.model or self.default_model,
            "input": request.input_text,
            "instructions": request.instructions,
            "metadata": dict(request.metadata),
            "temperature": request.temperature,
            "max_output_tokens": request.max_output_tokens,
        }

    def infer(self, request: InferenceRequest) -> InferenceResult:
        if not self.has_api_key():
            raise RuntimeError(f"{self.api_key_env} is required for hosted Responses API inference")
        payload = self.build_payload(request)
        if not payload.get("model"):
            raise RuntimeError(
                "No OpenAI model is configured for hosted inference. "
                "Set app.openai_responses_model or the OPENAI_RESPONSES_MODEL environment variable."
            )
        body = json.dumps(payload).encode("utf-8")
        api_base = (self.api_base or "https://api.openai.com/v1").rstrip("/")
        endpoint = f"{api_base}/responses"
        http_request = urllib_request.Request(
            endpoint,
            data=body,
            headers={
                "Authorization": f"Bearer {os.environ[self.api_key_env]}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib_request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                raw_payload = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI Responses API request failed with HTTP {exc.code}: {details}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"OpenAI Responses API request failed: {exc.reason}") from exc
        output_text = self._extract_output_text(raw_payload)
        if not output_text:
            raise RuntimeError("OpenAI Responses API returned no text output.")
        return InferenceResult(
            provider_id=self.provider_id,
            model=str(raw_payload.get("model") or request.model or self.default_model or ""),
            output_text=output_text,
            finish_reason=str(raw_payload.get("status") or raw_payload.get("finish_reason") or "") or None,
            raw_response=raw_payload,
        )

    @staticmethod
    def _extract_output_text(payload: dict[str, Any]) -> str:
        direct = str(payload.get("output_text") or "").strip()
        if direct:
            return direct
        collected: list[str] = []
        for item in payload.get("output") or []:
            if not isinstance(item, dict):
                continue
            for content in item.get("content") or []:
                if not isinstance(content, dict):
                    continue
                text_value = str(content.get("text") or "").strip()
                if text_value:
                    collected.append(text_value)
        return "\n\n".join(part for part in collected if part).strip()
