from __future__ import annotations

from dataclasses import dataclass
import re

from lumen.nlu.prompt_nlu import PromptNLU
from lumen.tools.registry_types import BundleManifest
from lumen.routing.tool_signal_catalog import DORMANT_TOOL_SIGNAL_CATALOG


@dataclass(slots=True)
class CapabilitySpec:
    capability_key: str
    tool_id: str
    tool_capability: str
    description: str = ""
    command_aliases: list[str] | None = None
    trigger_keywords: list[str] | None = None
    structural_patterns: list[str] | None = None
    intent_hints: list[str] | None = None
    routing_priority: int = 0
    tool_intent_required: bool = True


@dataclass(slots=True)
class ToolSignalMatch:
    capability_key: str
    tool_id: str
    tool_capability: str
    confidence: float
    match_source: str
    matched_keywords: list[str]
    matched_patterns: list[str]
    matched_intent_hints: list[str]
    tool_intent_gate_passed: bool
    suppressed_reason: str | None = None


class CapabilityManager:
    """Resolves app-level capabilities to concrete tool bundle capabilities."""

    def __init__(self, manifests: dict[str, BundleManifest] | None = None) -> None:
        self._capabilities: dict[str, CapabilitySpec] = {}
        self._prompt_nlu = PromptNLU()
        if manifests:
            self.load_from_manifests(manifests)

    def load_from_manifests(self, manifests: dict[str, BundleManifest]) -> None:
        capabilities: dict[str, CapabilitySpec] = {}
        for manifest in manifests.values():
            for capability in manifest.capabilities:
                if not capability.app_capability_key:
                    continue
                capabilities[capability.app_capability_key] = CapabilitySpec(
                    capability_key=capability.app_capability_key,
                    tool_id=manifest.id,
                    tool_capability=capability.id,
                    description=capability.app_description or capability.description,
                    command_aliases=capability.command_aliases,
                    trigger_keywords=capability.trigger_keywords,
                    structural_patterns=capability.structural_patterns,
                    intent_hints=capability.intent_hints,
                    routing_priority=capability.routing_priority,
                    tool_intent_required=capability.tool_intent_required,
                )
        self._capabilities = capabilities

    def get(self, capability_key: str) -> CapabilitySpec:
        if capability_key not in self._capabilities:
            known = ", ".join(sorted(self._capabilities)) or "<none>"
            raise KeyError(
                f"Unknown capability key '{capability_key}'. Available capabilities: {known}"
            )
        return self._capabilities[capability_key]

    def list_capabilities(self) -> dict[str, CapabilitySpec]:
        return dict(self._capabilities)

    def find_by_command(self, action: str, target: str) -> CapabilitySpec | None:
        needle = f"{action.strip().lower()} {target.strip().lower()}"
        for spec in self._capabilities.values():
            aliases = spec.command_aliases or []
            if needle in {alias.strip().lower() for alias in aliases}:
                return spec
        return None

    def infer_command_alias(self, prompt: str) -> tuple[CapabilitySpec, str] | None:
        understanding = self._prompt_nlu.analyze(prompt)
        if not self._looks_like_tool_request(understanding.normalized_text):
            return None
        prompt_tokens = self._meaningful_tokens(understanding.normalized_text)
        entity_values = {
            entity.value.strip().lower()
            for entity in understanding.entities
            if entity.value
        }
        best_match: tuple[int, CapabilitySpec, str] | None = None
        for spec in self._capabilities.values():
            for alias in spec.command_aliases or []:
                normalized_alias = alias.strip().lower()
                alias_tokens = self._meaningful_tokens(normalized_alias)
                token_overlap = prompt_tokens & alias_tokens
                score = len(token_overlap) * 2
                if entity_values & alias_tokens:
                    score += len(entity_values & alias_tokens)
                if normalized_alias in understanding.normalized_text:
                    score += 4
                if score < 4:
                    continue
                if best_match is None or score > best_match[0]:
                    best_match = (score, spec, normalized_alias)
        if best_match is None:
            return None
        return best_match[1], best_match[2]

    def infer_by_signals(self, prompt: str) -> ToolSignalMatch | None:
        understanding = self._prompt_nlu.analyze(prompt)
        normalized = understanding.intent_ready_text
        tool_intent_gate = self._passes_tool_intent_gate(normalized)
        best_match: ToolSignalMatch | None = None
        for spec in self._capabilities.values():
            matched_keywords = self._matched_keywords(normalized, spec.trigger_keywords or [])
            matched_patterns = self._matched_patterns(normalized, spec.structural_patterns or [])
            matched_intent_hints = self._matched_keywords(normalized, spec.intent_hints or [])
            signal_strength = (
                (len(matched_patterns) * 0.34)
                + (len(matched_keywords) * 0.12)
                + (len(matched_intent_hints) * 0.18)
                + min(0.08, spec.routing_priority * 0.01)
            )
            if signal_strength <= 0:
                continue
            if spec.tool_intent_required and not tool_intent_gate:
                suppressed = "tool_intent_gate"
            elif signal_strength < 0.72:
                suppressed = "low_confidence"
            else:
                suppressed = None
            confidence = round(min(0.94, 0.22 + signal_strength), 3)
            candidate = ToolSignalMatch(
                capability_key=spec.capability_key,
                tool_id=spec.tool_id,
                tool_capability=spec.tool_capability,
                confidence=confidence,
                match_source="hybrid_signal",
                matched_keywords=matched_keywords,
                matched_patterns=matched_patterns,
                matched_intent_hints=matched_intent_hints,
                tool_intent_gate_passed=tool_intent_gate,
                suppressed_reason=suppressed,
            )
            if candidate.suppressed_reason is not None:
                continue
            if best_match is None:
                best_match = candidate
                continue
            current_score = (
                len(candidate.matched_patterns) * 3,
                candidate.confidence,
                self._capabilities[candidate.capability_key].routing_priority,
            )
            best_score = (
                len(best_match.matched_patterns) * 3,
                best_match.confidence,
                self._capabilities[best_match.capability_key].routing_priority,
            )
            if current_score > best_score:
                best_match = candidate
        return best_match

    def dormant_signal_catalog(self) -> dict[str, dict[str, object]]:
        return DORMANT_TOOL_SIGNAL_CATALOG

    @staticmethod
    def _meaningful_tokens(text: str) -> set[str]:
        stopwords = {"the", "a", "an", "for", "of", "and", "to", "with", "this", "that", "latest"}
        return {
            token
            for token in text.split()
            if token not in stopwords and len(token) > 2
        }

    @staticmethod
    def _looks_like_tool_request(text: str) -> bool:
        normalized = " ".join(text.strip().lower().split())
        tool_verbs = {
            "analyze",
            "adapt",
            "compare",
            "format",
            "inspect",
            "extract",
            "plot",
            "render",
            "search",
            "summarize",
            "report",
            "generate",
            "show",
            "review",
            "simulate",
            "model",
            "build",
            "create",
            "run",
        }
        tokens = normalized.split()
        if tokens and tokens[0] in tool_verbs:
            return True
        request_phrases = (
            "report ",
            "summarize ",
            "inspect ",
            "extract ",
            "plot ",
            "render ",
            "search ",
            "compare ",
            "show ",
            "generate ",
            "format ",
            "adapt ",
            "simulate ",
            "model ",
            "run ",
        )
        return any(phrase in normalized for phrase in request_phrases)

    @staticmethod
    def _matched_keywords(normalized: str, keywords: list[str]) -> list[str]:
        return [keyword for keyword in keywords if keyword in normalized]

    def _matched_patterns(self, normalized: str, patterns: list[str]) -> list[str]:
        matched: list[str] = []
        for pattern in patterns:
            if self._matches_pattern(normalized, pattern):
                matched.append(pattern)
        return matched

    @staticmethod
    def _matches_pattern(normalized: str, pattern: str) -> bool:
        prompt = normalized.lower()
        if pattern == "contains_equation":
            return "=" in prompt
        if pattern == "algebraic_variable_form":
            return bool(re.search(r"\b\d*[xyz]\b", prompt))
        if pattern == "polynomial_form":
            return "x^" in prompt or "x²" in prompt or "y^" in prompt or "z^" in prompt
        if pattern == "numeric_variable_expression":
            return bool(re.search(r"\b\d+[xyz]\b|\b[xyz]\s*[\+\-\*/]\s*\d", prompt))
        if pattern == "matrix_literal_or_operation":
            return "[" in prompt and "]" in prompt or "matrix" in prompt
        if pattern == "integration_bounds":
            return any(token in prompt for token in ("from ", "between ", "lower_bound", "upper_bound"))
        if pattern == "optimization_bounds":
            return any(token in prompt for token in ("maximize", "minimize", "bounds", "between "))
        if pattern == "file_path_mention":
            return "/" in prompt or "\\" in prompt or ".py" in prompt or ".json" in prompt
        if pattern == "repo_structure_phrase":
            return any(
                token in prompt
                for token in ("codebase", "module", "service", "package", "repo", "architecture", "workflow", "pipeline")
            )
        if pattern == "structured_data_file":
            return any(token in prompt for token in (".csv", ".tsv", ".json", "dataset", "table", "spreadsheet"))
        if pattern == "dataset_analysis_phrase":
            return any(token in prompt for token in ("describe this dataset", "analyze this dataset", "describe the data", "analyze the data", "inspect this csv", "inspect this json"))
        if pattern == "trend_analysis_phrase":
            return any(token in prompt for token in ("regression", "trend", "correlate", "scatter", "plot"))
        if pattern == "network_graph_phrase":
            return any(token in prompt for token in ("graph", "network", "nodes", "edges", "relationships"))
        if pattern == "timeline_phrase":
            return any(token in prompt for token in ("timeline", "chronology", "sequence of events"))
        if pattern == "parameter_space_phrase":
            return any(token in prompt for token in ("parameter space", "phase space", "map parameters"))
        if pattern == "orbit_phrase":
            return any(token in prompt for token in ("simulate orbit", "orbital path", "orbit path", "model orbit"))
        if pattern == "population_growth_phrase":
            return any(
                token in prompt
                for token in ("population growth", "simulate population", "logistic growth", "population dynamics")
            )
        if pattern == "diffusion_phrase":
            return any(token in prompt for token in ("simulate diffusion", "diffusion model", "spread over time"))
        if pattern == "paper_search_phrase":
            return any(token in prompt for token in ("search papers", "find papers", "paper search", "literature search"))
        if pattern == "paper_summary_phrase":
            return any(token in prompt for token in ("summarize paper", "paper summary", "summarize this paper"))
        if pattern == "paper_compare_phrase":
            return any(token in prompt for token in ("compare papers", "paper comparison", "compare these papers"))
        if pattern == "methods_extraction_phrase":
            return any(token in prompt for token in ("extract methods", "paper methods", "methodology from this paper"))
        if pattern == "paper_text_file":
            return any(token in prompt for token in (".txt", ".md", "paper text", "paper file"))
        if pattern == "system_analysis_phrase":
            return any(
                token in prompt
                for token in (
                    "how is this structured",
                    "analyze this system",
                    "system structure",
                    "analyze this architecture",
                    "inspect this architecture",
                )
            )
        if pattern == "docs_generation_phrase":
            return any(
                token in prompt
                for token in (
                    "generate docs",
                    "document this",
                    "module overview",
                    "capability catalog",
                    "generate documentation",
                    "document this architecture",
                )
            )
        if pattern == "multi_concept_relation":
            return any(token in prompt for token in (" these ", " concepts ", " ideas ", " relate", "connect"))
        if pattern == "relation_question":
            return any(token in prompt for token in ("how do these relate", "how are these connected"))
        if pattern == "inconsistency_claims":
            return any(token in prompt for token in ("inconsisten", "contradiction", "claims"))
        if pattern == "source_target_path":
            return "source" in prompt and "target" in prompt or "path between" in prompt
        if pattern == "workspace_inspection":
            return any(
                token in prompt
                for token in (
                    "inspect workspace",
                    "summarize workspace",
                    "project tree",
                    "top level",
                    "workspace structure",
                    "repo structure",
                    "project structure",
                )
            )
        if pattern == "session_confidence_phrase":
            return any(
                token in prompt
                for token in (
                    "session confidence",
                    "confidence report",
                    "posture summary",
                    "how confident are you",
                    "confidence for this session",
                )
            )
        if pattern == "session_timeline_phrase":
            return any(
                token in prompt
                for token in (
                    "session timeline",
                    "history timeline",
                    "interaction timeline",
                    "what have we done",
                    "what did we do",
                )
            )
        if pattern == "numeric_array":
            return bool(re.search(r"\[\s*\d+(\.\d+)?(?:\s*,\s*\d+(\.\d+)?)+\s*\]", prompt))
        if pattern == "trend_analysis_phrase":
            return any(token in prompt for token in ("find trends", "analyze this data", "correlate"))
        if pattern == "dataset_analysis_phrase":
            return "dataset" in prompt or "data set" in prompt
        if pattern == "spectral_file_phrase":
            return any(token in prompt for token in ("fits", "x1d", "spectrum", "spectra", "spectral file"))
        if pattern == "absorption_candidate_phrase":
            return any(token in prompt for token in ("si iv", "absorption", "dip", "candidate dip", "gas-line"))
        if pattern == "time_based_change":
            return any(token in prompt for token in ("over time", "evolve", "time step"))
        if pattern == "system_with_variables":
            return bool(re.search(r"\b[A-Za-z]\b", prompt)) and "system" in prompt
        if pattern == "constraint_framing":
            return "under these constraints" in prompt or "within these constraints" in prompt
        if pattern == "concept_generation_phrase":
            return any(
                token in prompt
                for token in ("generate a concept", "create a concept", "generate concept", "generate concepts", "create concepts")
            )
        if pattern == "constraint_check_phrase":
            return any(
                token in prompt
                for token in ("check concept constraints", "check constraints", "constraint check", "verify constraints")
            )
        if pattern == "material_selection_phrase":
            return any(token in prompt for token in ("suggest materials", "material suggestions", "what materials should", "choose materials"))
        if pattern == "failure_modes_phrase":
            return any(token in prompt for token in ("failure modes", "analyze failure modes", "likely failures", "what could fail"))
        if pattern == "energy_model_phrase":
            return any(
                token in prompt
                for token in ("model energy", "energy model", "kinetic energy", "potential energy")
            )
        if pattern == "orbit_profile_phrase":
            return any(
                token in prompt
                for token in ("orbit profile", "analyze orbit profile", "astronomy orbit")
            )
        if pattern == "design_spec_phrase":
            return any(token in prompt for token in ("generate system spec", "design system spec", "generate design spec"))
        if pattern == "experiment_design_phrase":
            return any(
                token in prompt
                for token in (
                    "design experiment",
                    "design an experiment",
                    "experiment to test",
                    "test whether",
                )
            )
        if pattern == "experiment_analysis_phrase":
            return any(
                token in prompt
                for token in (
                    "analysis plan",
                    "plan experiment analysis",
                    "analyze experiment results",
                    "analysis of experiment results",
                )
            )
        if pattern == "scientific_method_framing":
            return "hypothesis" in prompt or "controls" in prompt or "variables" in prompt
        if pattern == "variables_controls_phrase":
            return "design an experiment" in prompt or ("variables" in prompt and "controls" in prompt)
        if pattern == "short_form_content_phrase":
            return any(
                token in prompt
                for token in (
                    "short-form",
                    "short form",
                    "content batch",
                    "script lines",
                    "caption",
                    "hashtags",
                    "hook",
                )
            )
        if pattern == "platform_target_mention":
            return any(token in prompt for token in ("tiktok", "youtube shorts", "youtube_shorts", "shorts"))
        if pattern == "idea_generation_phrase":
            return any(token in prompt for token in ("content ideas", "brainstorm", "hooks for", "idea list"))
        if pattern == "batch_generation_phrase":
            return any(
                token in prompt
                for token in ("content batch", "batch of posts", "batch drafts", "content drafts", "draft a batch")
            )
        return False

    @staticmethod
    def _passes_tool_intent_gate(normalized: str) -> bool:
        imperative_prefixes = (
            "solve",
            "analyze",
            "inspect",
            "generate",
            "find",
            "check",
            "cluster",
            "map",
            "plot",
            "render",
            "search",
            "extract",
            "simplify",
            "integrate",
            "optimize",
            "report",
            "summarize",
            "link",
            "show",
            "review",
            "document",
            "adapt",
            "simulate",
            "model",
        )
        object_framing = ("this ", "these ", "this system", "these claims", "this data")
        if normalized.startswith(imperative_prefixes):
            return True
        if any(token in normalized for token in object_framing):
            return True
        if any(
            token in normalized
            for token in ("content ideas", "content batch", "content drafts", "format content", "adapt content")
        ) and any(verb in normalized for verb in ("generate", "brainstorm", "create", "format", "adapt")):
            return True
        if "experiment" in normalized and any(
            phrase in normalized for phrase in ("design experiment", "identify variables", "identify controls", "analysis plan")
        ):
            return True
        if any(
            phrase in normalized
            for phrase in (
                "generate concept",
                "generate concepts",
                "check concept constraints",
                "check constraints",
                "suggest materials",
                "analyze failure modes",
            )
        ):
            return True
        if any(
            phrase in normalized
            for phrase in (
                "model energy",
                "energy model",
                "physics energy model",
                "analyze orbit profile",
                "astronomy orbit profile",
                "orbit profile",
            )
        ):
            return True
        if "=" in normalized:
            return True
        if re.search(r"\b\d*[xyz]\b", normalized) and any(ch.isdigit() for ch in normalized):
            return True
        return False
