from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lumen.app.models import InteractionProfile
from lumen.nlu.models import PromptUnderstanding
from lumen.reasoning.memory_retrieval_layer import MemoryRetrievalResult
from lumen.reasoning.reasoning_state import ReasoningStateFrame
from lumen.reasoning.pipeline_models import (
    ClarificationGateDecision,
    ResponsePackagingContext,
    RouteAuthorityDecision,
)
from lumen.reasoning.route_support_signals import RouteSupportSignals
from lumen.routing.domain_router import DomainRoute


@dataclass(slots=True, frozen=True)
class InteractionTurnContext:
    original_prompt: str
    effective_prompt: str
    session_id: str
    client_surface: str
    input_path: Path | None = None
    params: dict[str, int | float | str] | None = None
    run_root: Path | None = None
    wake_interaction: dict[str, object] | None = None
    active_thread: dict[str, Any] | None = None
    interaction_profile: InteractionProfile | None = None
    recent_interactions: tuple[dict[str, object], ...] = ()
    clarification_continuation: dict[str, object] | None = None
    pipeline_prompt: str | None = None
    interaction_summary: dict[str, object] | None = None
    prompt_understanding: PromptUnderstanding | None = None
    pipeline_result: Any | None = None
    pipeline_trace: Any | None = None
    route: DomainRoute | None = None
    route_authority: RouteAuthorityDecision | None = None
    clarification_decision: ClarificationGateDecision | None = None
    memory_retrieval: MemoryRetrievalResult | None = None
    route_support_signals: RouteSupportSignals | None = None
    response_packaging: ResponsePackagingContext | None = None
    reasoning_state: ReasoningStateFrame | None = None
    supervised_support_trace: dict[str, object] | None = None
    response: dict[str, object] | None = None
    update_active_thread: bool = False
