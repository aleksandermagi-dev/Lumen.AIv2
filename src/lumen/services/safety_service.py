from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from lumen.app.settings import AppSettings
from lumen.routing.tool_registry import ToolRegistry
from lumen.services.safety_models import PromptSafetyDecision
from lumen.tools.registry_types import CapabilityManifest
from lumen.services.safety_outcome_evaluator import SafetyOutcomeEvaluator


class SafetyService:
    """Applies prompt and execution safety checks."""

    SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
    _WEAPON_PATTERNS = (
        "gun",
        "firearm",
        "pistol",
        "rifle",
        "shotgun",
        "make a gun",
        "build a gun",
        "make a firearm",
        "build a firearm",
        "bomb",
        "explosive",
        "detonator",
        "molotov",
        "weapon",
        "silencer",
        "ghost gun",
    )
    _DRUG_PATTERNS = (
        "cook meth",
        "make meth",
        "synthesize cocaine",
        "make fentanyl",
        "drug lab",
    )
    _BIO_PATTERNS = (
        "weaponize pathogen",
        "spread disease intentionally",
        "modify pathogen",
        "bioweapon",
    )
    _CYBER_PATTERNS = (
        "hack",
        "malware",
        "ransomware",
        "phishing",
        "password theft",
        "credential stuffing",
        "ddos",
        "steal account",
        "crack wifi",
    )
    _VIOLENCE_PATTERNS = (
        "kill",
        "murder",
        "poison",
        "sabotage",
        "hurt them",
        "attack someone",
    )
    _SELF_HARM_PATTERNS = (
        "kill myself",
        "suicide",
        "hurt myself",
        "self harm",
        "end my life",
    )
    _RESTRICTED_ACCESS_PATTERNS = (
        "classified",
        "bypass security",
        "unauthorized access",
        "infiltrate",
        "secure facility",
        "covert operation",
    )
    _PUBLIC_DISCUSSION_SIGNALS = (
        "declassified",
        "historical",
        "history",
        "ethics",
        "public record",
        "critical analysis",
    )
    _SEXUAL_EXPLOITATION_PATTERNS = (
        "child porn",
        "underage sex",
        "minor sex",
        "minor",
        "underage",
        "rape",
        "sexual assault",
        "incest",
        "forced sex",
    )
    _EXPLICIT_SEXUAL_PATTERNS = (
        "erotic roleplay",
        "write porn",
        "explicit sex story",
        "graphic sexual",
        "sexual roleplay",
    )
    _SEXUAL_HEALTH_SIGNALS = (
        "sexual health",
        "std",
        "sti",
        "contraception",
        "consent",
        "pregnancy",
        "anatomy",
    )
    _DEPENDENCY_PATTERNS = (
        "be my girlfriend",
        "be my boyfriend",
        "tell me you love me",
        "i only need you",
        "make someone depend on me",
        "isolate them from friends",
        "manipulate my partner",
    )
    _SELF_MODIFICATION_PATTERNS = (
        "edit yourself",
        "modify yourself",
        "rewrite your own code",
        "patch yourself",
        "change your own code",
        "update your own code",
        "upgrade yourself",
        "self edit",
        "self-edit",
    )

    def __init__(self, *, settings: AppSettings, registry: ToolRegistry):
        self.settings = settings
        self.registry = registry
        self.outcome_evaluator = SafetyOutcomeEvaluator()

    def validate_tool_request(
        self,
        *,
        tool_id: str,
        capability: str,
        input_path: Path | None,
        params: dict[str, Any] | None,
        session_id: str,
        run_root: Path | None,
    ) -> None:
        manifests = self.registry.get_manifests()
        manifest = manifests.get(tool_id)
        if manifest is None:
            raise ValueError(f"Tool bundle '{tool_id}' is not registered")

        capability_map = manifest.capability_map()
        if capability not in capability_map:
            raise ValueError(
                f"Capability '{capability}' is not registered for bundle '{tool_id}'"
            )

        normalized_session_id = session_id.strip()
        if not self.SESSION_ID_PATTERN.match(normalized_session_id):
            raise ValueError(
                "Session id must start with an alphanumeric character and only contain "
                "letters, numbers, '.', '_' or '-'"
            )

        if input_path is not None:
            resolved_input = input_path.resolve()
            if not resolved_input.exists():
                raise FileNotFoundError(f"Tool input does not exist: {resolved_input}")
            if not (resolved_input.is_file() or resolved_input.is_dir()):
                raise ValueError(f"Tool input must be a file or directory: {resolved_input}")

        if run_root is not None:
            resolved_run_root = run_root.resolve()
            try:
                resolved_run_root.relative_to(self.settings.repo_root)
            except ValueError as exc:
                raise ValueError(
                    f"Run root must stay inside the repo root: {resolved_run_root}"
                ) from exc

        if params is None:
            return

        for key, value in params.items():
            if not isinstance(key, str):
                raise ValueError("Tool params must use string keys")
            if not self._is_safe_param_value(value):
                raise ValueError(
                    f"Tool param '{key}' must be JSON-like data using strings, numbers, bools, null, lists, or dicts"
                )

    def capability_safety_profile(self, *, tool_id: str, capability: str) -> dict[str, object]:
        manifests = self.registry.get_manifests()
        manifest = manifests.get(tool_id)
        if manifest is None:
            raise ValueError(f"Tool bundle '{tool_id}' is not registered")
        capability_spec = manifest.capability_map().get(capability)
        if capability_spec is None:
            raise ValueError(
                f"Capability '{capability}' is not registered for bundle '{tool_id}'"
            )
        return self._capability_safety_profile(capability_spec)

    def evaluate_prompt(self, prompt: str) -> PromptSafetyDecision:
        normalized = " ".join(str(prompt).lower().split())

        if self._contains_any(normalized, self._SELF_HARM_PATTERNS):
            return PromptSafetyDecision(
                action="refuse",
                category="self_harm",
                severity="high",
                rationale="Self-harm instructions are not allowed.",
                boundary="I can't help with instructions for harming yourself.",
                tier="restricted",
                outcome_risk="high",
                safe_redirects=[
                    "If you are in immediate danger, contact local emergency services now.",
                    "If you want, I can help you write a message to a trusted person or find crisis support resources.",
                ],
                matched_signals=self._matched_signals(normalized, self._SELF_HARM_PATTERNS),
            )

        if self._contains_any(normalized, self._SEXUAL_EXPLOITATION_PATTERNS):
            return PromptSafetyDecision(
                action="refuse",
                category="sexual_exploitation",
                severity="high",
                rationale="Exploitative or abusive sexual content is not allowed.",
                boundary="I can't help with exploitative, coercive, or abusive sexual content.",
                tier="restricted",
                outcome_risk="high",
                safe_redirects=[
                    "I can help with consent, sexual health, or relationship-safety information instead.",
                ],
                matched_signals=self._matched_signals(normalized, self._SEXUAL_EXPLOITATION_PATTERNS),
            )

        if self._contains_any(normalized, self._EXPLICIT_SEXUAL_PATTERNS) and not self._contains_any(
            normalized, self._SEXUAL_HEALTH_SIGNALS
        ):
            return PromptSafetyDecision(
                action="refuse",
                category="explicit_sexual_content",
                severity="medium",
                rationale="Explicit sexual content requests are outside Lumen's scope.",
                boundary="I can't help produce explicit sexual content.",
                tier="restricted",
                outcome_risk="high",
                safe_redirects=[
                    "I can help with sexual health, consent, or relationship questions in a non-explicit way.",
                ],
                matched_signals=self._matched_signals(normalized, self._EXPLICIT_SEXUAL_PATTERNS),
            )

        if self._contains_any(normalized, self._DEPENDENCY_PATTERNS):
            return PromptSafetyDecision(
                action="refuse",
                category="manipulative_dependency",
                severity="high",
                rationale="Manipulative emotional dependency or coercive intimacy is not allowed.",
                boundary="I can't help with coercive intimacy or manipulating someone's dependence on you.",
                tier="restricted",
                outcome_risk="high",
                safe_redirects=[
                    "I can help you communicate honestly, set boundaries, or think through a healthier relationship approach.",
                ],
                matched_signals=self._matched_signals(normalized, self._DEPENDENCY_PATTERNS),
            )

        if self._contains_any(normalized, self._SELF_MODIFICATION_PATTERNS):
            return PromptSafetyDecision(
                action="refuse",
                category="self_modification",
                severity="medium",
                rationale="Runtime self-modification is disabled.",
                boundary="I can't edit or modify my own code or runtime behavior from inside the conversation.",
                tier="restricted",
                outcome_risk="medium",
                safe_redirects=[
                    "I can help explain what change you want, draft a plan, or review a supervised edit you explicitly direct.",
                ],
                matched_signals=self._matched_signals(normalized, self._SELF_MODIFICATION_PATTERNS),
            )

        if self._contains_any(normalized, self._WEAPON_PATTERNS):
            return self._refusal(
                normalized,
                category="weapons_explosives",
                severity="high",
                rationale="Weapon or explosive construction and optimization are not allowed.",
                boundary="I can't help with building, optimizing, or using weapons or explosives.",
                safe_redirects=[
                    "I can help with safety engineering, materials science history, or public-policy discussion instead.",
                ],
                patterns=self._WEAPON_PATTERNS,
            )

        if self._contains_any(normalized, self._DRUG_PATTERNS):
            return self._refusal(
                normalized,
                category="illegal_drug_production",
                severity="high",
                rationale="Illegal drug production guidance is not allowed.",
                boundary="I can't help with producing illegal drugs.",
                safe_redirects=[
                    "I can help with public-health information, addiction science, or legal policy discussion instead.",
                ],
                patterns=self._DRUG_PATTERNS,
            )

        if self._contains_any(normalized, self._BIO_PATTERNS):
            return self._refusal(
                normalized,
                category="biological_weaponization",
                severity="high",
                rationale="Biological weaponization guidance is not allowed.",
                boundary="I can't help with modifying or spreading biological agents for harm.",
                safe_redirects=[
                    "I can help with biosafety, public health, or the history and ethics of biological risk instead.",
                ],
                patterns=self._BIO_PATTERNS,
            )

        if self._contains_any(normalized, self._CYBER_PATTERNS):
            return self._refusal(
                normalized,
                category="cybercrime_fraud",
                severity="high",
                rationale="Cybercrime, fraud, and unauthorized access guidance are not allowed.",
                boundary="I can't help with hacking, malware, fraud, or credential theft.",
                safe_redirects=[
                    "I can help with defensive security, incident response, or account-hardening guidance instead.",
                ],
                patterns=self._CYBER_PATTERNS,
            )

        if self._contains_any(normalized, self._VIOLENCE_PATTERNS):
            return self._refusal(
                normalized,
                category="violent_wrongdoing",
                severity="high",
                rationale="Violent wrongdoing guidance is not allowed.",
                boundary="I can't help with harming, killing, or sabotaging people or property.",
                safe_redirects=[
                    "I can help you de-escalate a situation, think through safety options, or discuss the issue at a non-operational level instead.",
                ],
                patterns=self._VIOLENCE_PATTERNS,
            )

        if self._contains_any(normalized, self._RESTRICTED_ACCESS_PATTERNS) and not self._contains_any(
            normalized, self._PUBLIC_DISCUSSION_SIGNALS
        ):
            return self._refusal(
                normalized,
                category="restricted_access_or_covert_misuse",
                severity="high",
                rationale="Unauthorized access or covert misuse guidance is not allowed.",
                boundary="I can't help with bypassing security, infiltrating protected systems, or covert harmful operations.",
                safe_redirects=[
                    "I can help discuss declassified history, security ethics, or high-level institutional analysis instead.",
                ],
                patterns=self._RESTRICTED_ACCESS_PATTERNS,
            )

        outcome = self.outcome_evaluator.evaluate(prompt)
        if outcome.tier == "restricted":
            return PromptSafetyDecision(
                action="refuse",
                category=outcome.category,
                severity="high",
                rationale=outcome.rationale,
                boundary=outcome.boundary,
                tier=outcome.tier,
                outcome_risk=outcome.outcome_risk,
                response_constraint=dict(outcome.response_constraint),
                tool_constraint={"level": "blocked", "reason": "restricted_prompt"},
                safe_redirects=list(outcome.safe_redirects),
                matched_signals=list(outcome.matched_signals),
            )

        tool_constraint = (
            {
                "level": "constrained",
                "reason": "dual_use_prompt",
                "allow_execution": False,
                "allow_high_level_only": True,
            }
            if outcome.tier == "dual_use"
            else {}
        )
        return PromptSafetyDecision(
            action="allow",
            category="allowed",
            severity="medium" if outcome.tier == "dual_use" else "none",
            rationale=outcome.rationale if outcome.tier == "dual_use" else "No prompt safety restriction triggered.",
            boundary="",
            tier=outcome.tier,
            outcome_risk=outcome.outcome_risk,
            response_constraint=dict(outcome.response_constraint),
            tool_constraint=tool_constraint,
            matched_signals=list(outcome.matched_signals),
        )

    def policy_report(self) -> dict[str, object]:
        return {
            "session_id_pattern": self.SESSION_ID_PATTERN.pattern,
            "run_root_scope": str(self.settings.repo_root),
            "param_policy": "json_like",
            "input_policy": "existing_file_when_provided",
            "bundle_policy": "registry_declared_only",
            "prompt_policy_version": "v3",
            "outcome_policy_version": "v1",
            "prompt_actions": ["allow", "refuse"],
            "tier_model": ["safe", "dual_use", "restricted"],
            "tool_capability_levels": ["allowed", "constrained", "blocked"],
            "governing_rule": "If the response would meaningfully reduce the barrier to real-world harm, block or constrain it.",
            "hard_refuse_categories": [
                "weapons_explosives",
                "illegal_drug_production",
                "biological_weaponization",
                "cybercrime_fraud",
                "violent_wrongdoing",
                "self_harm",
                "sexual_exploitation",
                "manipulative_dependency",
                "self_modification",
                "restricted_access_or_covert_misuse",
                "explicit_sexual_content",
            ],
            "contextual_allowances": [
                "declassified_history",
                "public_ethics_discussion",
                "sexual_health",
            ],
        }

    @staticmethod
    def _contains_any(text: str, patterns: tuple[str, ...]) -> bool:
        return any(pattern in text for pattern in patterns)

    @staticmethod
    def _matched_signals(text: str, patterns: tuple[str, ...]) -> list[str]:
        return [pattern for pattern in patterns if pattern in text]

    @staticmethod
    def _capability_safety_profile(capability: CapabilityManifest) -> dict[str, object]:
        level = str(getattr(capability, "safety_level", "allowed") or "allowed").strip().lower()
        if level not in {"allowed", "constrained", "blocked"}:
            level = "allowed"
        return {
            "level": level,
            "notes": str(getattr(capability, "safety_notes", "") or "").strip(),
        }

    @classmethod
    def _is_safe_param_value(cls, value: Any) -> bool:
        if value is None or isinstance(value, (str, int, float, bool)):
            return True
        if isinstance(value, list):
            return all(cls._is_safe_param_value(item) for item in value)
        if isinstance(value, dict):
            return all(
                isinstance(key, str) and cls._is_safe_param_value(item)
                for key, item in value.items()
            )
        return False

    def _refusal(
        self,
        text: str,
        *,
        category: str,
        severity: str,
        rationale: str,
        boundary: str,
        safe_redirects: list[str],
        patterns: tuple[str, ...],
    ) -> PromptSafetyDecision:
        return PromptSafetyDecision(
            action="refuse",
            category=category,
            severity=severity,
            rationale=rationale,
            boundary=boundary,
            tier="restricted",
            outcome_risk="high",
            safe_redirects=safe_redirects,
            matched_signals=self._matched_signals(text, patterns),
        )
