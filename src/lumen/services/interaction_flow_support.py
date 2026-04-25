from __future__ import annotations

from pathlib import Path
import re
from typing import Callable

from lumen.nlu.prompt_surface_builder import PromptSurfaceBuilder
from lumen.reasoning.explanation_response_builder import ExplanationAnswerResult
from lumen.reasoning.continuation_confidence_policy import ContinuationConfidencePolicy
from lumen.reasoning.explanation_response_builder import ExplanationResponseBuilder
from lumen.reasoning.explanatory_support_policy import ExplanatorySupportPolicy
from lumen.tools.experiment_tools import extract_topic_from_prompt
from lumen.tools.invent_tools import extract_brief_from_prompt, extract_constraints_from_prompt
from lumen.tools.domain_tools import extract_energy_request, extract_orbit_profile_request


class InteractionFlowSupport:
    """Pure helper seams for interaction-flow shaping and continuation logic."""

    _WINDOWS_ABSOLUTE_PATH_PATTERN = re.compile(
        r"([A-Za-z]:(?:[\\/][^<>:\"|?*\r\n]+)+\.(?:fits?|x1d|csv|json|txt|zip))",
        flags=re.IGNORECASE,
    )

    _EXPLANATORY_PREFIXES = (
        "research ",
        "research on ",
        "tell me about ",
        "explain ",
        "explain to me ",
        "explain to me what ",
        "what is ",
        "what's ",
        "whats ",
    )
    _GENERIC_ATTACHED_INPUT_PHRASES = (
        "analyze this attached file",
        "analyze this file",
        "inspect this file",
        "read this file",
        "what is this file",
        "what's this file",
        "tell me what this file is",
        "analyze this attached folder",
        "analyze this folder",
        "inspect this folder",
        "what is this folder",
        "analyze this zip",
        "inspect this zip",
        "what is this zip",
        "analyze this attached zip",
    )

    @staticmethod
    def finalize_explanatory_answer(
        *,
        response: dict[str, object],
        prompt: str,
        route,
        interaction_profile,
        entities,
        provider_text: str | None,
        recent_interactions: list[dict[str, object]] | None,
        route_support_signals,
        knowledge_service,
        explanation_surface_style: Callable[..., str],
    ) -> None:
        route_mode = str(getattr(route, "mode", "") or "")
        route_kind = str(getattr(route, "kind", "") or "")
        explanatory_signals = (
            getattr(route_support_signals, "explanatory", None)
            if route_support_signals is not None
            else ExplanatorySupportPolicy.evaluate(prompt=prompt, entities=entities)
        )
        knowledge_eligible = ExplanationResponseBuilder.should_consult_local_knowledge(
            prompt=prompt,
            route_mode=route_mode,
            route_kind=route_kind,
            entities=entities,
            support_signals=explanatory_signals,
        )
        if not knowledge_eligible:
            return
        if route_kind == "research.comparison":
            local_lookup = ExplanationResponseBuilder.lookup_local_knowledge(
                prompt=prompt,
                knowledge_service=knowledge_service,
            )
            if local_lookup is None:
                return
        tone_profile = str(
            (
                (response.get("response_tone_blend") or {})
                if isinstance(response.get("response_tone_blend"), dict)
                else {}
            ).get("tone_profile")
            or ""
        ).strip()
        surface_style = explanation_surface_style(
            interaction_profile=interaction_profile,
            tone_profile=tone_profile,
        )
        continuation = ContinuationConfidencePolicy.evaluate(
            prompt=prompt,
            recent_interactions=recent_interactions or [],
        )
        knowledge_prompt = continuation.target_prompt
        reasoning_state = response.get("reasoning_state") if isinstance(response.get("reasoning_state"), dict) else {}
        explanation_strategy = str(reasoning_state.get("explanation_strategy") or "").strip() or None
        answer_result = ExplanationAnswerResult(answer="", source="fallback")
        fallback_answer = ""
        for candidate_prompt in InteractionFlowSupport._knowledge_prompt_candidates(knowledge_prompt):
            answer_result = ExplanationResponseBuilder.build_answer(
                prompt=candidate_prompt,
                interaction_style=surface_style,
                explanation_strategy=explanation_strategy,
                continuation=continuation.inherits_confidence,
                entities=entities,
                provider_text=provider_text,
                knowledge_service=knowledge_service,
            )
            answer_text = str(answer_result.answer or "").strip()
            if (
                answer_text
                and not InteractionFlowSupport._looks_like_missing_detail_answer(answer_text)
                and not fallback_answer
            ):
                fallback_answer = answer_text
            if answer_result.should_replace_surface:
                break
        answer = str(answer_result.answer or "").strip()
        project_memory_hint = str(response.get("project_memory_hint") or "").strip()
        if (
            project_memory_hint
            and answer_result.should_replace_surface
            and project_memory_hint.lower() not in answer.lower()
        ):
            answer = f"{project_memory_hint} {answer}".strip()
        response["explanation_answer_source"] = answer_result.source
        existing_summary = str(response.get("summary") or "").strip()
        existing_findings = [str(item).strip() for item in response.get("findings") or [] if str(item).strip()]
        existing_recommendation = str(response.get("recommendation") or response.get("next_action") or "").strip()
        existing_substantive_surface = (
            any(InteractionFlowSupport._is_substantive_reasoning_line(item) for item in existing_findings)
            or InteractionFlowSupport._is_substantive_reasoning_line(existing_recommendation)
            or InteractionFlowSupport._is_substantive_surface(existing_summary)
        )
        should_surface_low_support = (
            route_mode == "research"
            and (
                answer_result.source in {"partial", "generic", "fallback"}
                or InteractionFlowSupport._looks_like_missing_detail_answer(answer)
            )
            and not existing_substantive_surface
        )
        if not answer_result.should_replace_surface and not should_surface_low_support:
            if fallback_answer:
                response["explanatory_body"] = fallback_answer
            return
        response["user_facing_answer"] = answer
        response["summary"] = answer
        response["reply"] = answer
        response["explanatory_body"] = answer
        response["findings"] = []
        response.pop("recommendation", None)
        response.pop("next_action", None)
        response.pop("conversation_turn", None)
        response.pop("response_intro", None)
        response.pop("response_opening", None)
        response["internal_scaffold_visible"] = False

    @staticmethod
    def _is_substantive_surface(text: str) -> bool:
        normalized = " ".join(str(text or "").replace("\u2019", "'").strip().lower().split())
        if not normalized:
            return False
        if normalized in {
            "here's a first pass.",
            "here is a first pass.",
            "here's a workable answer.",
            "here is a workable answer.",
            "here's the clearest current answer.",
            "here is the clearest current answer.",
            "here's a grounded answer.",
            "here is a grounded answer.",
            "here's the clearest answer.",
            "here is the clearest answer.",
            "here's the grounded explanation.",
            "here is the grounded explanation.",
            "here's a useful first plan, with the assumptions kept visible.",
            "here is a useful first plan, with the assumptions kept visible.",
            "here's a grounded answer using the best current assumptions.",
            "here is a grounded answer using the best current assumptions.",
            "here's a grounded answer, with the assumptions kept visible.",
            "here is a grounded answer, with the assumptions kept visible.",
        }:
            return False
        return True

    @staticmethod
    def _is_substantive_reasoning_line(text: str) -> bool:
        normalized = " ".join(str(text or "").strip().lower().split())
        if not normalized:
            return False
        blocked_prefixes = (
            "state the topic in one concise sentence",
            "highlight the most relevant local evidence first",
            "close with the clearest next action or conclusion",
            "summarize the strongest local evidence first",
            "use the strongest local context to decide the next concrete validation step",
            "keep the first conclusion close to the strongest local signal",
        )
        return not any(normalized.startswith(prefix) for prefix in blocked_prefixes)

    @staticmethod
    def _knowledge_prompt_candidates(prompt: str) -> list[str]:
        normalized = " ".join(str(prompt or "").strip().split())
        if not normalized:
            return [""]
        candidates = [normalized]
        lowered = normalized.lower()
        for prefix in InteractionFlowSupport._EXPLANATORY_PREFIXES:
            if lowered.startswith(prefix):
                stripped = normalized[len(prefix) :].strip()
                if stripped and stripped not in candidates:
                    candidates.append(stripped)
        article_stripped: list[str] = []
        for candidate in list(candidates):
            cleaned = InteractionFlowSupport._strip_leading_article(candidate)
            if cleaned and cleaned not in candidates and cleaned not in article_stripped:
                article_stripped.append(cleaned)
            subject = InteractionFlowSupport._strip_trailing_is(cleaned or candidate)
            if subject and subject not in candidates and subject not in article_stripped:
                article_stripped.append(subject)
            article_cleaned_subject = InteractionFlowSupport._strip_leading_article(subject)
            if (
                article_cleaned_subject
                and article_cleaned_subject not in candidates
                and article_cleaned_subject not in article_stripped
            ):
                article_stripped.append(article_cleaned_subject)
        candidates.extend(article_stripped)
        return candidates

    @staticmethod
    def _strip_leading_article(text: str) -> str:
        return re.sub(r"^(?:a|an|the)\s+", "", str(text or "").strip(), count=1, flags=re.IGNORECASE).strip()

    @staticmethod
    def _strip_trailing_is(text: str) -> str:
        normalized = str(text or "").strip()
        return re.sub(r"\bis\s*$", "", normalized, flags=re.IGNORECASE).strip()

    @staticmethod
    def _looks_like_missing_detail_answer(text: str) -> bool:
        normalized = " ".join(str(text or "").strip().lower().split())
        return any(
            phrase in normalized
            for phrase in (
                "don't have enough grounded detail",
                "don't have enough detail",
                "answer confidently yet",
                "explain it properly yet",
            )
        )

    @staticmethod
    def response_to_response_bridge(
        *,
        prompt: str,
        recent_interactions: list[dict[str, object]],
    ) -> dict[str, str] | None:
        if not recent_interactions:
            return None
        latest = recent_interactions[0]
        latest_kind = str(latest.get("kind") or "").strip()
        latest_summary = str(latest.get("summary") or "").strip().lower()
        response = latest.get("response") if isinstance(latest.get("response"), dict) else {}
        latest_reply = str(response.get("reply") or response.get("summary") or "").strip().lower()
        latest_text = " ".join(part for part in (latest_summary, latest_reply) if part).strip()
        if not latest_text:
            return None
        prompt_shape = PromptSurfaceBuilder.build(prompt).route_ready_text
        if not prompt_shape:
            return None
        invited_pickup = (
            latest_kind
            in {
                "conversation.topic_suggestion",
                "conversation.greeting",
                "conversation.check_in",
                "conversation.thought_mode",
            }
            or "pick one" in latest_text
            or "your call" in latest_text
            or "what do you want to" in latest_text
            or "what direction do you want" in latest_text
            or "what are you thinking" in latest_text
            or "what's on your mind" in latest_text
        )
        if not invited_pickup:
            return None

        direct_prefixes = (
            "let's do ",
            "lets do ",
            "let's get into ",
            "lets get into ",
            "let's go with ",
            "lets go with ",
            "let's run with ",
            "lets run with ",
            "let's talk about ",
            "lets talk about ",
            "let's pick up ",
            "lets pick up ",
            "pick up ",
            "i want to talk about ",
            "i want to get into ",
            "i want to do ",
        )
        soft_prefixes = ("maybe ", "probably ")
        if prompt_shape in {"the first one", "first one", "the second one", "second one", "that one"}:
            return {"category": "directional_acceptance", "target": prompt_shape}
        if prompt_shape.startswith("probably the first one") or prompt_shape.startswith(
            "maybe the first one"
        ):
            return {"category": "hesitant_acceptance", "target": "the first one"}
        for prefix in direct_prefixes:
            if prompt_shape.startswith(prefix):
                return {
                    "category": "direct_acceptance",
                    "target": prompt_shape[len(prefix) :].strip() or "that",
                }
        for prefix in soft_prefixes:
            if prompt_shape.startswith(prefix):
                return {
                    "category": "hesitant_acceptance",
                    "target": prompt_shape[len(prefix) :].strip() or "that",
                }
        if prompt_shape.startswith("let's ") or prompt_shape.startswith("lets "):
            return {
                "category": "collaborative_pickup",
                "target": prompt_shape.split(" ", 1)[1].strip() if " " in prompt_shape else "that",
            }
        if prompt_shape.startswith("i'm thinking ") or prompt_shape.startswith("im thinking "):
            prefix = "i'm thinking " if prompt_shape.startswith("i'm thinking ") else "im thinking "
            return {
                "category": "soft_acceptance",
                "target": prompt_shape[len(prefix) :].strip() or "that",
            }
        token_count = len(prompt_shape.split())
        if 0 < token_count <= 5 and not any(
            prompt_shape.startswith(prefix)
            for prefix in ("what ", "why ", "how ", "can ", "could ", "should ", "would ", "do ", "did ", "is ", "are ")
        ):
            return {
                "category": "soft_acceptance",
                "target": prompt_shape,
            }
        return None

    @classmethod
    def clarification_planning_continuation(
        cls,
        *,
        prompt: str,
        recent_interactions: list[dict[str, object]],
        active_thread: dict[str, object] | None = None,
    ) -> dict[str, object] | None:
        normalized_prompt = PromptSurfaceBuilder.build(prompt).route_ready_text
        if not normalized_prompt:
            return None
        if cls.is_direction_pivot(normalized_prompt):
            return {
                "action": "pivot",
                "working_prompt": cls._normalize_pivot_prompt(prompt),
            }
        if cls.is_clarification_decline(normalized_prompt):
            return {"action": "decline"}
        route_info = cls._clarified_route_info_from_state(active_thread=active_thread)
        if route_info is None:
            if not recent_interactions:
                return None
            latest = recent_interactions[0]
            latest_mode = str(latest.get("mode") or "").strip()
            latest_kind = str(latest.get("kind") or "").strip()
            response = latest.get("response") if isinstance(latest.get("response"), dict) else {}
            response_mode = str(response.get("mode") or "").strip()
            response_kind = str(response.get("kind") or "").strip()
            if {latest_mode, response_mode}.isdisjoint({"clarification"}) and {
                latest_kind,
                response_kind,
            }.isdisjoint({"clarification.request"}):
                return None
            route_info = cls.extract_clarified_route_info(latest=latest, response=response)
            if route_info is None:
                return None
        if not cls.is_clarification_confirmation(
            normalized_prompt=normalized_prompt,
            route_mode=str(route_info.get("mode") or ""),
            route_kind=str(route_info.get("kind") or ""),
        ):
            return None
        base_prompt = (
            str(route_info.get("resolved_prompt") or "").strip()
            or str(latest.get("resolved_prompt") or "").strip()
            or str(response.get("resolved_prompt") or "").strip()
            or str(latest.get("prompt") or "").strip()
        )
        if not base_prompt:
            return None
        working_prompt = base_prompt
        if not cls.is_pure_clarification_confirmation(normalized_prompt):
            suffix = (
                "Additional design direction"
                if str(route_info.get("mode") or "").strip() == "planning"
                else "Additional clarification detail"
            )
            working_prompt = f"{base_prompt}. {suffix}: {prompt.strip()}"
        return {
            "action": "continue",
            "mode": str(route_info.get("mode") or "planning"),
            "kind": str(route_info.get("kind") or "planning.architecture"),
            "resolved_prompt": base_prompt,
            "working_prompt": working_prompt,
            "confidence": float(route_info.get("confidence") or 0.84),
        }

    @staticmethod
    def _clarified_route_info_from_state(
        *,
        active_thread: dict[str, object] | None,
    ) -> dict[str, object] | None:
        if not isinstance(active_thread, dict):
            return None
        reasoning_state = active_thread.get("reasoning_state")
        if not isinstance(reasoning_state, dict):
            return None
        pending = reasoning_state.get("pending_followup")
        if not isinstance(pending, dict):
            return None
        if str(pending.get("type") or "").strip() not in {"clarification", "clarification_resolved"}:
            return None
        mode = str(pending.get("route_mode") or "").strip()
        kind = str(pending.get("route_kind") or "").strip()
        resolved_prompt = str(pending.get("resolved_prompt") or reasoning_state.get("continuation_target") or reasoning_state.get("resolved_prompt") or "").strip()
        if not mode or not kind or not resolved_prompt:
            return None
        return {
            "mode": mode,
            "kind": kind,
            "resolved_prompt": resolved_prompt,
            "confidence": float(reasoning_state.get("confidence") or 0.84),
        }

    @staticmethod
    def extract_clarified_route_info(
        *,
        latest: dict[str, object],
        response: dict[str, object],
    ) -> dict[str, object] | None:
        clarification_context = (
            response.get("clarification_context")
            if isinstance(response.get("clarification_context"), dict)
            else {}
        )
        suggested_route = (
            clarification_context.get("suggested_route")
            if isinstance(clarification_context.get("suggested_route"), dict)
            else {}
        )
        mode = str(suggested_route.get("mode") or "").strip()
        kind = str(suggested_route.get("kind") or "").strip()
        if (not mode or not kind) and isinstance(response.get("options"), list):
            option = next(
                (
                    str(item).strip()
                    for item in response["options"]
                    if "(" in str(item) and ")" in str(item)
                ),
                "",
            )
            if option:
                label, _, kind_part = option.partition("(")
                mode = mode or label.strip().lower()
                kind = kind or kind_part.rstrip(")").strip()
        if not mode or not kind:
            return None
        route_metadata = response.get("route") if isinstance(response.get("route"), dict) else {}
        return {
            "mode": mode,
            "kind": kind,
            "resolved_prompt": str(suggested_route.get("resolved_prompt") or "").strip(),
            "confidence": route_metadata.get("confidence"),
        }

    @staticmethod
    def is_clarification_confirmation(
        *,
        normalized_prompt: str,
        route_mode: str,
        route_kind: str,
    ) -> bool:
        explicit_confirmations = {
            "keep current route",
            "go with that",
            "lets go with that",
            "let's go with that",
            "let's do that",
            "lets do that",
            "yes",
            "yes please",
            "yeah",
            "yep",
            "go ahead",
            "proceed",
            "continue",
            "keep going",
            "a plan",
            "the plan",
            "plan",
            "planning",
            "design",
            "design it",
            "build it",
            "summary",
            "comparison",
        }
        design_detail_tokens = {
            "engine",
            "propulsion",
            "thruster",
            "motor",
            "system",
            "device",
            "mechanism",
            "prototype",
            "reusable",
            "compact",
            "lightweight",
            "modular",
        }
        if normalized_prompt in explicit_confirmations:
            return True
        if route_mode == "research" and normalized_prompt.startswith(("explain ", "summarize ", "compare ")):
            return True
        if normalized_prompt.startswith(("plan ", "design ", "build ", "sketch ", "propose ")):
            return True
        if any(phrase in normalized_prompt for phrase in ("want a plan", "want the plan", "go with the plan")):
            return True
        if route_mode == "planning" and ("explain" in normalized_prompt or "explanation" in normalized_prompt):
            return False
        if route_kind == "planning.architecture" and any(
            token in normalized_prompt.split() for token in design_detail_tokens
        ):
            return True
        return False

    @staticmethod
    def is_planning_clarification_confirmation(*, normalized_prompt: str, route_kind: str) -> bool:
        return InteractionFlowSupport.is_clarification_confirmation(
            normalized_prompt=normalized_prompt,
            route_mode="planning",
            route_kind=route_kind,
        )

    @staticmethod
    def is_pure_clarification_confirmation(normalized_prompt: str) -> bool:
        return normalized_prompt in {
            "keep current route",
            "go with that",
            "lets go with that",
            "let's go with that",
            "let's do that",
            "lets do that",
            "yes",
            "yes please",
            "yeah",
            "yep",
            "go ahead",
            "proceed",
            "continue",
            "keep going",
            "a plan",
            "the plan",
            "plan",
            "planning",
            "design",
            "design it",
            "build it",
            "summary",
            "comparison",
        }

    @staticmethod
    def is_clarification_decline(normalized_prompt: str) -> bool:
        return normalized_prompt in {
            "no",
            "nope",
            "not that",
            "dont continue",
            "don't continue",
            "cancel",
            "stop",
        }

    @staticmethod
    def is_direction_pivot(normalized_prompt: str) -> bool:
        normalized = " ".join(str(normalized_prompt or "").strip().lower().split())
        return normalized.startswith(
            (
                "let's explore ",
                "lets explore ",
                "explore ",
                "instead ",
                "switch to ",
                "pivot to ",
                "no let's explore ",
                "no lets explore ",
            )
        )

    @staticmethod
    def is_route_choice_prompt(normalized_prompt: str) -> bool:
        normalized = " ".join(str(normalized_prompt or "").strip().lower().split())
        if not normalized:
            return False
        exact_matches = {
            "what route makes sense here",
            "which route should we take",
            "which direction should we go",
            "what direction makes sense here",
            "help me decide the route",
            "which route fits this best",
        }
        if normalized in exact_matches:
            return True
        return normalized.startswith(
            (
                "what route makes sense",
                "which route should we take",
                "which direction should we go",
                "what direction makes sense",
                "help me decide the route",
                "which route fits this best",
            )
        )

    @staticmethod
    def _normalize_pivot_prompt(prompt: str) -> str:
        normalized = " ".join(str(prompt or "").strip().split())
        if not normalized:
            return ""
        patterns = (
            r"^no,\s*",
            r"^no\s+",
            r"^(?:let's|lets)\s+explore\s+",
            r"^explore\s+",
            r"^instead\s+",
            r"^switch\s+to\s+",
            r"^pivot\s+to\s+",
        )
        candidate = normalized
        for pattern in patterns:
            candidate = re.sub(pattern, "", candidate, flags=re.IGNORECASE).strip()
        candidate = re.sub(r"\s+instead$", "", candidate, flags=re.IGNORECASE).strip()
        return candidate or normalized

    @classmethod
    def infer_live_tool_params(
        cls,
        *,
        prompt: str,
        resolved_prompt: str,
    ) -> dict[str, object] | None:
        surfaces = PromptSurfaceBuilder.build(prompt)
        normalized_prompt = surfaces.tool_ready_text
        normalized_resolved = " ".join(str(resolved_prompt or "").strip().lower().split())

        if normalized_resolved == "solve equation" and "=" in prompt:
            equation_text = cls.extract_equation_text(surfaces.tool_source_text)
            if not equation_text:
                return None
            normalized_equation = " ".join(equation_text.lower().split())
            variable_matches = re.findall(r"\b([xyz])\b", normalized_equation)
            compact_matches = re.findall(r"(?<![A-Za-z])\d*([xyz])(?![A-Za-z])", normalized_equation)
            variables = variable_matches or compact_matches
            variable = next((item for item in variables if item in {"x", "y", "z"}), "")
            if variable:
                return {
                    "equation": equation_text,
                    "variable": variable,
                }

        if normalized_resolved == "simplify expression":
            expression = str(surfaces.tool_source_text or "").strip()
            lowered = normalized_prompt
            for prefix in ("simplify ", "expand ", "factor "):
                if lowered.startswith(prefix):
                    target_form = prefix.strip()
                    raw_expression = expression[len(prefix) :].strip()
                    if raw_expression:
                        normalized_target_form = "canonical" if target_form == "simplify" else target_form
                        return {
                            "expression": raw_expression,
                            "target_form": normalized_target_form,
                        }

        if normalized_resolved == "check contradictions":
            claims = cls.extract_claims(surfaces.tool_source_text)
            if claims:
                return {
                    "claims": claims,
                    "strictness": "medium",
                }

        if normalized_resolved in {"link knowledge", "cluster knowledge"}:
            items = cls.extract_items(surfaces.tool_source_text)
            if items:
                params = {"items": items}
                if normalized_resolved == "cluster knowledge":
                    params["strategy"] = "relation"
                else:
                    relation_hint = cls.extract_relation_hint(surfaces.tool_ready_text)
                    if relation_hint:
                        params["relation_hint"] = relation_hint
                return params

        if normalized_resolved == "find knowledge":
            path_params = cls.extract_source_target_path(surfaces.tool_source_text)
            if path_params is not None:
                return path_params

        if normalized_resolved in {"generate system spec", "design system spec", "generate design spec"}:
            brief = cls.extract_design_brief(surfaces.tool_source_text)
            if brief:
                return {"brief": brief}

        if normalized_resolved == "analyze architecture":
            request = cls.extract_system_tool_request(surfaces.tool_source_text, mode="analyze_architecture")
            if request is not None:
                return request

        if normalized_resolved == "suggest refactor":
            request = cls.extract_system_tool_request(surfaces.tool_source_text, mode="suggest_refactor")
            if request is not None:
                return request

        if normalized_resolved == "generate docs":
            request = cls.extract_system_tool_request(surfaces.tool_source_text, mode="generate_docs")
            if request is not None:
                return request

        if normalized_resolved == "search papers":
            query = cls._strip_command_prefix(
                surfaces.tool_source_text,
                prefixes=("search papers", "find papers", "paper search", "literature search"),
            )
            if query:
                return {"query": query}

        if normalized_resolved == "compare papers":
            papers = cls._extract_paper_comparison_inputs(surfaces.tool_source_text)
            if papers:
                return {"papers": papers}

        if normalized_resolved == "summarize paper":
            paper_text = cls._strip_command_prefix(
                surfaces.tool_source_text,
                prefixes=("summarize paper", "summarize this paper", "paper summary"),
            )
            if paper_text:
                return {"paper_text": paper_text}

        if normalized_resolved == "extract paper methods":
            paper_text = cls._strip_command_prefix(
                surfaces.tool_source_text,
                prefixes=("extract paper methods", "extract methods", "paper methods"),
            )
            if paper_text:
                return {"paper_text": paper_text}

        if normalized_resolved == "simulate system":
            return cls.extract_simulation_request(surfaces.tool_source_text, mode="system")

        if normalized_resolved == "simulate orbit":
            return cls.extract_simulation_request(surfaces.tool_source_text, mode="orbit")

        if normalized_resolved == "simulate population":
            return cls.extract_simulation_request(surfaces.tool_source_text, mode="population")

        if normalized_resolved == "simulate diffusion":
            return cls.extract_simulation_request(surfaces.tool_source_text, mode="diffusion")

        if normalized_resolved == "design experiment":
            return cls.extract_experiment_request(surfaces.tool_source_text, mode="design")

        if normalized_resolved == "identify variables":
            return cls.extract_experiment_request(surfaces.tool_source_text, mode="variables")

        if normalized_resolved == "identify controls":
            return cls.extract_experiment_request(surfaces.tool_source_text, mode="controls")

        if normalized_resolved == "plan experiment analysis":
            return cls.extract_experiment_request(surfaces.tool_source_text, mode="analysis_plan")

        if normalized_resolved in {"generate concept", "generate concepts"}:
            return cls.extract_invent_request(surfaces.tool_source_text, mode="generate_concepts")

        if normalized_resolved in {"check concept constraints", "check constraints"}:
            return cls.extract_invent_request(surfaces.tool_source_text, mode="constraint_check")

        if normalized_resolved == "suggest materials":
            return cls.extract_invent_request(surfaces.tool_source_text, mode="material_suggestions")

        if normalized_resolved == "analyze failure modes":
            return cls.extract_invent_request(surfaces.tool_source_text, mode="failure_modes")

        if normalized_resolved in {"model energy", "energy model", "physics energy model"}:
            return extract_energy_request(surfaces.tool_source_text)

        if normalized_resolved in {"analyze orbit profile", "astronomy orbit profile", "orbit profile"}:
            return extract_orbit_profile_request(surfaces.tool_source_text)

        if normalized_resolved == "generate content ideas":
            request = cls.extract_content_request(surfaces.tool_source_text, mode="ideas")
            if request is not None:
                return request

        if normalized_resolved == "generate content batch":
            request = cls.extract_content_request(surfaces.tool_source_text, mode="batch")
            if request is not None:
                return request

        if normalized_resolved in {"format content for platform", "adapt content for platform"}:
            request = cls.extract_platform_format_request(surfaces.tool_source_text)
            if request is not None:
                return request

        return None

    @classmethod
    def extract_live_tool_input_path(
        cls,
        *,
        prompt: str,
        resolved_prompt: str,
    ) -> Path | None:
        text = str(prompt or "").strip()
        if not text:
            return None
        lowered = " ".join(text.lower().split())
        normalized_resolved = " ".join(str(resolved_prompt or "").strip().lower().split())
        for alias in (normalized_resolved, "run anh", "scan si iv dips", "find spectral dips"):
            if alias and lowered.startswith(alias):
                text = text[len(alias) :].strip(" :,-")
                break
        match = cls._WINDOWS_ABSOLUTE_PATH_PATTERN.search(text)
        if match is None:
            return None
        return Path(match.group(1).strip().rstrip(".?!,;"))

    @classmethod
    def rewrite_prompt_for_attached_input(
        cls,
        *,
        prompt: str,
        input_path: Path | None,
    ) -> str | None:
        if input_path is None:
            return None
        normalized = PromptSurfaceBuilder.build(prompt).route_ready_text
        if not normalized:
            return None
        if normalized in {"run anh", "scan si iv dips", "find spectral dips"}:
            return None
        if not cls._looks_like_generic_attached_input_request(normalized):
            return None
        lowered_name = input_path.name.lower()
        suffix = input_path.suffix.lower()
        if input_path.is_dir() or suffix in {".fit", ".fits", ".x1d", ".zip"}:
            if any(token in lowered_name for token in ("mast", "x1d", "fits", "spectrum", "spectra")) or input_path.is_dir() or suffix in {".fit", ".fits", ".x1d", ".zip"}:
                return "run anh"
        if suffix in {".csv", ".tsv", ".json"}:
            return "describe data"
        if suffix in {".txt", ".md"}:
            return "summarize paper"
        return None

    @staticmethod
    def _strip_command_prefix(text: str, *, prefixes: tuple[str, ...]) -> str:
        normalized = " ".join(str(text or "").strip().split())
        lowered = normalized.lower()
        for prefix in prefixes:
            if lowered.startswith(prefix):
                return normalized[len(prefix) :].strip(" :.-")
        return ""

    @staticmethod
    def _extract_paper_comparison_inputs(text: str) -> list[str]:
        normalized = " ".join(str(text or "").strip().split())
        if not normalized:
            return []
        lowered = normalized.lower()
        for prefix in ("compare papers", "compare these papers"):
            if lowered.startswith(prefix):
                normalized = normalized[len(prefix) :].strip(" :.-")
                break
        lowered_normalized = normalized.lower()
        for separator in (" vs ", " versus ", " || ", "; "):
            if separator in lowered_normalized:
                parts = [part.strip() for part in re.split(re.escape(separator), normalized, flags=re.IGNORECASE) if part.strip()]
                if len(parts) >= 2:
                    return parts[:2]
        return []

    @staticmethod
    def extract_equation_text(prompt: str) -> str:
        text = str(prompt or "").strip()
        if "=" not in text:
            return ""
        lowered = text.lower()
        for marker in ("solve for ", "solve ", "equation "):
            index = lowered.rfind(marker)
            if index != -1:
                text = text[index + len(marker) :].strip()
                break
        text = text.strip().rstrip(".?!;,")
        if "=" not in text:
            return ""
        match = re.search(
            r"([0-9a-z_\(\)\[\]\+\-\*/\^²³⁰¹⁴⁵⁶⁷⁸⁹\s=.,]+)$",
            text,
            flags=re.IGNORECASE,
        )
        if match is not None:
            candidate = match.group(1).strip().rstrip(".?!;,")
            if "=" in candidate:
                return candidate
        return text

    @staticmethod
    def extract_claims(prompt: str) -> list[str]:
        text = str(prompt or "").strip().rstrip(".?!")
        if not text:
            return []
        lowered = text.lower()
        for marker in (
            "these claims:",
            "the following claims:",
            "claims:",
            "inconsistencies in",
            "contradictions in",
            "check contradictions in",
        ):
            index = lowered.find(marker)
            if index != -1:
                text = text[index + len(marker) :].strip()
                break
        text = text.strip().rstrip(".?!")
        if not text:
            return []
        separators = (";", "\n", " | ")
        parts: list[str] = [text]
        for separator in separators:
            if separator in text:
                parts = [item.strip(" -\t") for item in text.split(separator)]
                break
        if parts == [text] and "," in text and ":" in str(prompt):
            comma_parts = [item.strip(" -\t") for item in text.split(",")]
            if len(comma_parts) > 1:
                parts = comma_parts
        claims = [part.rstrip(".?!") for part in parts if part.strip(" -\t").strip()]
        return claims if len(claims) >= 2 else []

    @staticmethod
    def extract_items(prompt: str) -> list[str]:
        text = str(prompt or "").strip().rstrip(".?!")
        if not text:
            return []
        lowered = text.lower()
        for marker in (
            "how do these relate:",
            "how are these connected:",
            "link knowledge:",
            "cluster knowledge:",
            "these concepts:",
            "these ideas:",
            "items:",
        ):
            index = lowered.find(marker)
            if index != -1:
                text = text[index + len(marker) :].strip()
                break
        text = text.strip().rstrip(".?!")
        if not text:
            return []
        separator = None
        for candidate in (";", ",", "\n", " | "):
            if candidate in text:
                separator = candidate
                break
        if separator is None:
            return []
        items = [item.strip(" -\t").rstrip(".?!") for item in text.split(separator)]
        items = [item for item in items if item]
        return items if len(items) >= 2 else []

    @staticmethod
    def extract_relation_hint(normalized_prompt: str) -> str | None:
        prompt = " ".join(str(normalized_prompt or "").strip().lower().split())
        if "relate" in prompt or "relationship" in prompt:
            return "relationship"
        if "connect" in prompt or "connected" in prompt:
            return "connection"
        if "link" in prompt:
            return "link"
        return None

    @staticmethod
    def extract_source_target_path(prompt: str) -> dict[str, object] | None:
        text = str(prompt or "").strip().rstrip(".?!")
        if not text:
            return None
        lowered = text.lower()
        if "path between" in lowered and " and " in lowered:
            after = text[lowered.find("path between") + len("path between") :].strip()
            source, _, target = after.partition(" and ")
            if source.strip() and target.strip():
                return {"source": source.strip(), "target": target.strip(), "max_hops": 4}
        source_match = re.search(r"source\s*:\s*([^;,\n]+)", text, flags=re.IGNORECASE)
        target_match = re.search(r"target\s*:\s*([^;,\n]+)", text, flags=re.IGNORECASE)
        if source_match and target_match:
            source = source_match.group(1).strip().rstrip(".?!")
            target = target_match.group(1).strip().rstrip(".?!")
            if source and target:
                return {"source": source, "target": target, "max_hops": 4}
        return None

    @staticmethod
    def extract_design_brief(prompt: str) -> str:
        text = str(prompt or "").strip().rstrip(".?!")
        if not text:
            return ""
        lowered = text.lower()
        for marker in (
            "generate system spec for ",
            "design system spec for ",
            "generate design spec for ",
        ):
            if lowered.startswith(marker):
                return text[len(marker) :].strip()
        return text

    @staticmethod
    def extract_system_tool_request(
        prompt: str,
        *,
        mode: str,
    ) -> dict[str, object] | None:
        text = str(prompt or "").strip()
        lowered = " ".join(text.lower().split())
        target_path = ""
        explicit_path = InteractionFlowSupport.extract_live_tool_input_path(prompt=text, resolved_prompt=text)
        if explicit_path is not None:
            target_path = str(explicit_path)
        elif any(token in lowered for token in ("this architecture", "the architecture", "this repo", "this codebase", "this project")):
            target_path = "src"

        if mode == "analyze_architecture":
            if not target_path:
                target_path = "src"
            focus = ""
            for prefix in ("analyze architecture", "analyze this architecture", "analyze this system", "inspect architecture"):
                if lowered.startswith(prefix):
                    focus = text[len(prefix) :].strip(" :,-")
                    break
            return {"target_path": target_path, "focus": focus or None, "depth": 2}

        if mode == "suggest_refactor":
            goal = "extract_helpers"
            if "performance" in lowered:
                goal = "performance"
            elif any(token in lowered for token in ("clarity", "readability", "clean up", "cleanup")):
                goal = "clarity"
            if not target_path:
                target_path = "src"
            return {"target_path": target_path, "goal": goal}

        if mode == "generate_docs":
            if not target_path:
                target_path = "src"
            doc_type = "module_overview"
            if "catalog" in lowered:
                doc_type = "capability_catalog"
            return {
                "target_path": target_path,
                "doc_type": doc_type,
                "include_public_interfaces": True,
            }
        return None

    @classmethod
    def _looks_like_generic_attached_input_request(cls, normalized_prompt: str) -> bool:
        normalized = " ".join(str(normalized_prompt or "").strip().lower().split())
        if normalized in cls._GENERIC_ATTACHED_INPUT_PHRASES:
            return True
        if "attached file" in normalized or "attached zip" in normalized or "attached folder" in normalized:
            return any(token in normalized for token in ("analyze", "inspect", "read", "what is", "tell me"))
        if any(token in normalized for token in ("this file", "this folder", "this zip")):
            return any(token in normalized for token in ("analyze", "inspect", "read", "what is", "tell me"))
        return False

    @staticmethod
    def extract_content_request(prompt: str, *, mode: str) -> dict[str, object] | None:
        text = str(prompt or "").strip()
        lowered = " ".join(text.lower().split())
        aliases = {
            "ideas": ("generate content ideas", "brainstorm content ideas"),
            "batch": ("generate content batch", "create content drafts"),
        }
        for alias in aliases.get(mode, ()):
            if lowered.startswith(alias):
                text = text[len(alias) :].strip(" :,-")
                lowered = " ".join(text.lower().split())
                break
        if mode == "ideas" and lowered.startswith("generate me ") and "idea" in lowered:
            text = text[len("generate me ") :].strip(" :,-")
            lowered = " ".join(text.lower().split())
        elif mode == "batch" and lowered.startswith("generate me ") and any(
            token in lowered for token in ("draft", "drafts", "post", "posts", "script", "scripts")
        ):
            text = text[len("generate me ") :].strip(" :,-")
            lowered = " ".join(text.lower().split())
        platform = "all"
        for token, normalized in (
            ("youtube shorts", "youtube_shorts"),
            ("youtube_shorts", "youtube_shorts"),
            ("shorts", "youtube_shorts"),
            ("tiktok", "tiktok"),
        ):
            if token in lowered:
                platform = normalized
                break
        topic = text
        for phrase in (" on topic ", " about topic "):
            index = lowered.find(phrase)
            if index != -1:
                topic = text[index + len(phrase) :].strip(" :,-")
                break
        for prefix in ("for ", "about ", "on "):
            marker = f"{prefix}"
            index = lowered.find(marker)
            if index != -1:
                topic = text[index + len(prefix) :].strip(" :,-")
                break
        for noisy_prefix in (
            "topic ",
            "ideas on ",
            "ideas about ",
            "ideas for ",
            "content ideas on ",
            "content ideas about ",
            "content ideas for ",
            "drafts on ",
            "drafts about ",
            "posts on ",
            "posts about ",
        ):
            if topic.lower().startswith(noisy_prefix):
                topic = topic[len(noisy_prefix) :].strip(" :,-")
                break
        topic = topic.strip(" :,-")
        if not topic:
            return None
        if mode == "ideas":
            return {"topic": topic, "platform": platform, "count": 5}
        return {"topic": topic, "count": 3}

    @staticmethod
    def extract_platform_format_request(prompt: str) -> dict[str, object] | None:
        text = str(prompt or "").strip()
        lowered = " ".join(text.lower().split())
        aliases = ("format content for platform", "adapt content for platform")
        for alias in aliases:
            if lowered.startswith(alias):
                text = text[len(alias) :].strip(" :,-")
                lowered = " ".join(text.lower().split())
                break

        platform = ""
        for token, normalized in (
            ("youtube shorts", "youtube_shorts"),
            ("youtube_shorts", "youtube_shorts"),
            ("shorts", "youtube_shorts"),
            ("tiktok", "tiktok"),
        ):
            if token in lowered:
                platform = normalized
                for prefix in ("for ", "to ", "into "):
                    phrase = f"{prefix}{token}"
                    index = lowered.find(phrase)
                    if index != -1:
                        text = (text[:index] + text[index + len(phrase) :]).strip(" :,-")
                        lowered = " ".join(text.lower().split())
                        break
                break

        source_text = text.strip(" :,-")
        if not platform and not source_text:
            return None
        params: dict[str, object] = {}
        if platform:
            params["platform"] = platform
        if source_text:
            params["source_text"] = source_text
        return params or None

    @staticmethod
    def extract_simulation_request(prompt: str, *, mode: str) -> dict[str, object]:
        text = " ".join(str(prompt or "").strip().split())
        lowered = text.lower()
        prefixes = {
            "system": ("simulate system", "model system"),
            "orbit": ("simulate orbit", "model orbit"),
            "population": ("simulate population", "model population"),
            "diffusion": ("simulate diffusion", "model diffusion"),
        }
        for prefix in prefixes.get(mode, ()):
            if lowered.startswith(prefix):
                text = text[len(prefix) :].strip(" :,-")
                lowered = text.lower()
                break

        def _find_float(*patterns: str) -> float | None:
            for pattern in patterns:
                match = re.search(pattern, lowered, flags=re.IGNORECASE)
                if match is not None:
                    try:
                        return float(match.group(1))
                    except ValueError:
                        return None
            return None

        def _find_int(*patterns: str) -> int | None:
            value = _find_float(*patterns)
            if value is None:
                return None
            return int(value)

        params: dict[str, object] = {}
        steps = _find_int(r"\bsteps?\s+(\d+)\b", r"\bover\s+(\d+)\s+steps?\b", r"\bfor\s+(\d+)\s+steps?\b")
        if steps is not None:
            params["steps"] = steps

        if mode == "system":
            initial_value = _find_float(r"\binitial(?: value)?\s+([\-]?\d+(?:\.\d+)?)\b")
            growth_rate = _find_float(r"\bgrowth(?: rate)?\s+([\-]?\d+(?:\.\d+)?)\b")
            damping_rate = _find_float(r"\bdamping(?: rate)?\s+([\-]?\d+(?:\.\d+)?)\b")
            forcing = _find_float(r"\bforcing\s+([\-]?\d+(?:\.\d+)?)\b")
            if initial_value is not None:
                params["initial_value"] = initial_value
            if growth_rate is not None:
                params["growth_rate"] = growth_rate
            if damping_rate is not None:
                params["damping_rate"] = damping_rate
            if forcing is not None:
                params["forcing"] = forcing
            return params

        if mode == "orbit":
            semi_major_axis = _find_float(
                r"\bsemi[- ]major axis\s+([\-]?\d+(?:\.\d+)?)\b",
                r"\bradius\s+([\-]?\d+(?:\.\d+)?)\b",
            )
            eccentricity = _find_float(r"\beccentricity\s+([\-]?\d+(?:\.\d+)?)\b")
            samples = _find_int(r"\bsamples?\s+(\d+)\b", r"\bpoints?\s+(\d+)\b")
            if semi_major_axis is not None:
                params["semi_major_axis"] = semi_major_axis
            if eccentricity is not None:
                params["eccentricity"] = eccentricity
            if samples is not None:
                params["samples"] = samples
            return params

        if mode == "population":
            initial_population = _find_float(r"\binitial population\s+([\-]?\d+(?:\.\d+)?)\b", r"\bstart(?:ing)?\s+([\-]?\d+(?:\.\d+)?)\b")
            growth_rate = _find_float(r"\bgrowth(?: rate)?\s+([\-]?\d+(?:\.\d+)?)\b")
            carrying_capacity = _find_float(r"\bcarrying capacity\s+([\-]?\d+(?:\.\d+)?)\b", r"\bcapacity\s+([\-]?\d+(?:\.\d+)?)\b")
            if initial_population is not None:
                params["initial_population"] = initial_population
            if growth_rate is not None:
                params["growth_rate"] = growth_rate
            if carrying_capacity is not None:
                params["carrying_capacity"] = carrying_capacity
            return params

        if mode == "diffusion":
            cells = _find_int(r"\bcells?\s+(\d+)\b", r"\bgrid\s+(\d+)\b")
            diffusion_rate = _find_float(r"\bdiffusion(?: rate)?\s+([\-]?\d+(?:\.\d+)?)\b")
            peak_value = _find_float(r"\bpeak(?: value)?\s+([\-]?\d+(?:\.\d+)?)\b")
            if cells is not None:
                params["cells"] = cells
            if diffusion_rate is not None:
                params["diffusion_rate"] = diffusion_rate
            if peak_value is not None:
                params["peak_value"] = peak_value
            return params

        return params

    @staticmethod
    def extract_experiment_request(prompt: str, *, mode: str) -> dict[str, object]:
        text = " ".join(str(prompt or "").strip().split())
        topic = extract_topic_from_prompt(text)
        params: dict[str, object] = {}
        if topic:
            params["topic"] = topic
            if mode == "design":
                params["hypothesis"] = f"Changing the main condition in {topic} should produce a measurable response."

        lowered = text.lower()
        if mode in {"design", "variables", "analysis_plan"}:
            if "independent variable" in lowered:
                fragment = text[lowered.find("independent variable") + len("independent variable") :].strip(" :,-")
                if fragment:
                    params["independent_variable"] = fragment
            if "dependent variable" in lowered:
                fragment = text[lowered.find("dependent variable") + len("dependent variable") :].strip(" :,-")
                if fragment:
                    params["dependent_variable"] = fragment
        if mode == "controls" and "control" in lowered:
            fragment = text[lowered.find("control") + len("control") :].strip(" :,-")
            if fragment:
                params["controls"] = [fragment]
        return params

    @staticmethod
    def extract_invent_request(prompt: str, *, mode: str) -> dict[str, object]:
        text = " ".join(str(prompt or "").strip().split())
        brief = extract_brief_from_prompt(text)
        constraints = extract_constraints_from_prompt(text)
        params: dict[str, object] = {}
        if brief:
            params["brief"] = brief
        if constraints:
            params["constraints"] = constraints
        lowered = text.lower()
        if mode in {"constraint_check", "failure_modes"}:
            params["concept"] = brief or text
        if mode == "material_suggestions" and "for " in lowered and not brief:
            fragment = text[lowered.find("for ") + len("for ") :].strip(" :,-")
            if fragment:
                params["brief"] = fragment
        return params

    @staticmethod
    def memory_save_request(prompt: str, recent_interactions: list[dict[str, object]]) -> dict[str, object] | None:
        normalized = PromptSurfaceBuilder.build(prompt).lookup_ready_text
        if not normalized:
            return None
        explicit_prefixes = (
            "remember this",
            "save this",
            "remember that",
            "save that",
            "please remember",
            "please save",
            "remember what you just said",
            "save that explanation",
            "remember your answer",
            "remember your answer about",
        )
        if not any(normalized.startswith(prefix) for prefix in explicit_prefixes):
            return None
        if "about me" in normalized or "my preference" in normalized:
            return None
        explicit_assistant_target = InteractionFlowSupport._prompt_explicitly_targets_assistant_memory(
            normalized
        )
        candidates = InteractionFlowSupport._memory_save_candidates(
            recent_interactions,
            include_assistant=explicit_assistant_target,
        )
        if not candidates:
            return None
        explicit_target = InteractionFlowSupport._memory_save_target_from_prompt(normalized)
        if explicit_target is not None and explicit_target in candidates:
            return {
                "action": "save",
                "target": explicit_target,
                "source_record": candidates[explicit_target],
            }
        if len(candidates) == 1:
            target = next(iter(candidates))
            return {
                "action": "save",
                "target": target,
                "source_record": candidates[target],
            }
        options = []
        if "research" in candidates:
            options.append("Save the research")
        if "personal" in candidates:
            options.append("Save your last thought")
        if "assistant" in candidates:
            options.append("Save your last answer")
        return {
            "action": "clarify",
            "options": options,
            "candidates": tuple(candidates.keys()),
            "question": "Did you want the research, your last thought, or my last answer saved?",
        }

    @staticmethod
    def memory_save_continuation(
        prompt: str,
        recent_interactions: list[dict[str, object]],
    ) -> dict[str, object] | None:
        if len(recent_interactions) < 2:
            return None
        latest = recent_interactions[0]
        response = latest.get("response") if isinstance(latest.get("response"), dict) else {}
        clarification_context = (
            response.get("clarification_context")
            if isinstance(response.get("clarification_context"), dict)
            else {}
        )
        if str(clarification_context.get("clarification_type") or "").strip() != "memory_save":
            return None
        normalized = PromptSurfaceBuilder.build(prompt).lookup_ready_text
        target = InteractionFlowSupport._memory_save_target_from_prompt(normalized)
        if target is None:
            return None
        source_candidates = InteractionFlowSupport._memory_save_candidates(
            recent_interactions[1:],
            include_assistant=(target == "assistant"),
        )
        source_record = source_candidates.get(target)
        if source_record is None:
            return None
        return {
            "target": target,
            "source_record": source_record,
            "reply": (
                "Got it. I'll save the research from the recent thread."
                if target == "research"
                else (
                    "Got it. I'll save that as your recent thought."
                    if target == "personal"
                    else "Got it. I'll save my last answer from this thread."
                )
            ),
        }

    @staticmethod
    def _memory_save_target_from_prompt(normalized_prompt: str) -> str | None:
        normalized = " ".join(str(normalized_prompt or "").strip().lower().split())
        if any(token in normalized for token in ("research", "result", "note", "summary")):
            return "research"
        if any(token in normalized for token in ("thought", "idea", "personal", "about me")):
            return "personal"
        if any(
            token in normalized
            for token in ("what you just said", "your answer", "your last answer", "that explanation")
        ):
            return "assistant"
        if normalized in {
            "save the research",
            "remember the research",
            "save my last thought",
            "remember my last thought",
            "save your last answer",
            "remember your last answer",
        }:
            if "research" in normalized:
                return "research"
            if "thought" in normalized:
                return "personal"
            return "assistant"
        return None

    @staticmethod
    def _prompt_explicitly_targets_assistant_memory(normalized_prompt: str) -> bool:
        normalized = " ".join(str(normalized_prompt or "").strip().lower().split())
        return any(
            token in normalized
            for token in ("what you just said", "your answer", "your last answer", "that explanation")
        )

    @staticmethod
    def _memory_save_candidates(
        recent_interactions: list[dict[str, object]],
        *,
        include_assistant: bool = False,
    ) -> dict[str, dict[str, object]]:
        candidates: dict[str, dict[str, object]] = {}
        for record in recent_interactions:
            if not isinstance(record, dict):
                continue
            mode = str(record.get("mode") or "").strip()
            kind = str(record.get("kind") or "").strip()
            prompt = PromptSurfaceBuilder.build(str(record.get("prompt") or "")).lookup_ready_text
            if "research" not in candidates and mode in {"research", "planning", "tool"}:
                candidates["research"] = record
            if "personal" not in candidates:
                if kind in {"conversation.thought_mode", "conversation.thought_follow_up"}:
                    candidates["personal"] = record
                elif any(
                    cue in prompt
                    for cue in ("i think", "i feel", "my thought", "my idea", "for me", "about me")
                ):
                    candidates["personal"] = record
                elif mode == "conversation" and str(record.get("summary") or "").strip():
                    candidates["personal"] = record
            if include_assistant and "assistant" not in candidates:
                response = record.get("response") if isinstance(record.get("response"), dict) else {}
                assistant_text = (
                    str(response.get("user_facing_answer") or "").strip()
                    or str(response.get("reply") or "").strip()
                    or str(record.get("summary") or "").strip()
                )
                if assistant_text:
                    candidates["assistant"] = record
            if {"research", "personal"}.issubset(candidates.keys()) and (
                not include_assistant or "assistant" in candidates
            ):
                break
        return candidates
