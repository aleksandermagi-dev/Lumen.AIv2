from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ProviderCapabilities:
    supports_sync: bool = True
    supports_streaming: bool = False
    supports_async: bool = False
    supports_background: bool = False

    def to_dict(self) -> dict[str, bool]:
        return {
            "supports_sync": self.supports_sync,
            "supports_streaming": self.supports_streaming,
            "supports_async": self.supports_async,
            "supports_background": self.supports_background,
        }


@dataclass(slots=True)
class InferenceRequest:
    model: str | None
    instructions: str | None
    input_text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    temperature: float | None = None
    max_output_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "instructions": self.instructions,
            "input_text": self.input_text,
            "metadata": dict(self.metadata),
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
        }


@dataclass(slots=True)
class InferenceResult:
    provider_id: str
    model: str | None
    output_text: str
    finish_reason: str | None = None
    raw_response: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "model": self.model,
            "output_text": self.output_text,
            "finish_reason": self.finish_reason,
            "raw_response": dict(self.raw_response),
        }
