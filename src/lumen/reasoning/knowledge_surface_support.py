from __future__ import annotations

from lumen.knowledge.knowledge_service import KnowledgeService
from lumen.knowledge.models import KnowledgeLookupResult
from lumen.nlu.focus_resolution import FocusResolutionSupport
from lumen.nlu.prompt_nlu import PromptNLU
from lumen.reasoning.explanation_response_builder import ExplanationResponseBuilder
from lumen.reasoning.interaction_style_policy import InteractionStylePolicy
from lumen.reasoning.response_models import ResearchResponse
from lumen.reasoning.response_variation import ResponseVariationLayer


class KnowledgeSurfaceSupport:
    """Owns broad concept / seeded knowledge surfaces and lightweight continuations."""

    _prompt_nlu = PromptNLU()

    GENERIC_TOPIC_ANCHORS = {
        "anh",
        "astronomical node heuristics",
        "ai",
        "algebra",
        "ancient civilizations",
        "ancient history",
        "artificial intelligence",
        "astrology",
        "atoms and molecules",
        "biology",
        "bonding",
        "calculus",
        "chemistry",
        "chemical bonding",
        "civil engineering",
        "climate",
        "computer science",
        "computer security",
        "computers",
        "computing",
        "cyber security",
        "cybersecurity",
        "database",
        "databases",
        "design process",
        "earth science",
        "electricity",
        "electrical engineering",
        "engineering",
        "engineering design process",
        "evidence",
        "experiments",
        "fits spectrum",
        "fits spectra",
        "geometry",
        "geology",
        "how chemistry works",
        "hst cos",
        "hst/cos",
        "history",
        "horoscope",
        "logic",
        "mathematical logic",
        "math",
        "mathematics",
        "mast",
        "mast data",
        "measurement",
        "moon",
        "the moon",
        "lunar",
        "mars",
        "the mars",
        "mechanical engineering",
        "modern history",
        "modern world",
        "motion",
        "networks",
        "ocean",
        "oceans",
        "operating systems",
        "physics",
        "probability",
        "relativity",
        "reliability",
        "science",
        "scientific evidence",
        "scientific laws",
        "scientific method",
        "scientific models",
        "scientific theories",
        "si iv absorption",
        "software engineering",
        "space",
        "space itself",
        "outer space",
        "spectral dip scan",
        "spectral dip scans",
        "statistics",
        "system reliability",
        "systems",
        "the engineering design process",
        "thermodynamics",
        "universe",
        "the universe",
        "water cycle",
        "wave",
        "waves",
        "cosmos",
        "the cosmos",
        "cosmology",
        "milky way",
        "the milky way",
        "galaxy",
        "galaxies",
        "star",
        "stars",
        "gravity",
        "energy",
        "atom",
        "atoms",
        "watts",
        "watt",
        "what organ pumps blood",
        "what organ pumps blood throughout the body",
        "what organ pumps blood through out the body",
        "ohms",
        "ohm",
        "zodiac",
        "the zodiac",
        "zodiac signs",
        "ww2",
        "wwii",
        "world war 2",
        "world war ii",
        "world war two",
        "greece",
        "ancient greece",
    }

    BROAD_TOPIC_LOOKUP_ALIASES = {
        "anh": "ANH",
        "astronomical node heuristics": "ANH",
        "ai": "Artificial Intelligence",
        "algebra": "Algebra",
        "ancient civilizations": "Ancient Civilizations",
        "ancient history": "Ancient Civilizations",
        "artificial intelligence": "Artificial Intelligence",
        "astrology": "Astrology",
        "atoms and molecules": "Molecules and Compounds",
        "biology": "Biology",
        "bonding": "Chemical Bonding",
        "calculus": "Derivative",
        "chemistry": "Chemistry",
        "chemical bonding": "Chemical Bonding",
        "civil engineering": "Civil Engineering",
        "climate": "Weather and Climate",
        "computer science": "Computing",
        "computer security": "Cybersecurity",
        "computers": "Computing",
        "computing": "Computing",
        "cyber security": "Cybersecurity",
        "cybersecurity": "Cybersecurity",
        "database": "Database",
        "databases": "Database",
        "design process": "Engineering Design Process",
        "earth science": "Earth Science",
        "electricity": "Electricity",
        "electrical engineering": "Electrical Engineering",
        "engineering": "Engineering",
        "engineering design process": "Engineering Design Process",
        "evidence": "Evidence and Experiments",
        "experiments": "Evidence and Experiments",
        "fits spectrum": "FITS Spectrum",
        "fits spectra": "FITS Spectrum",
        "geometry": "Geometry",
        "geology": "Geology",
        "how chemistry works": "Chemistry",
        "hst cos": "HST/COS",
        "hst/cos": "HST/COS",
        "history": "History",
        "horoscope": "Horoscope",
        "logic": "Mathematical Logic",
        "mathematical logic": "Mathematical Logic",
        "math": "Mathematics",
        "mathematics": "Mathematics",
        "mast": "MAST Data",
        "mast data": "MAST Data",
        "measurement": "Measurement",
        "moon": "Moon",
        "the moon": "Moon",
        "lunar": "Moon",
        "mars": "Mars",
        "the mars": "Mars",
        "mechanical engineering": "Mechanical Engineering",
        "modern history": "Modern World Context",
        "modern world": "Modern World Context",
        "motion": "Motion",
        "networks": "Network Topology",
        "ocean": "Oceans",
        "oceans": "Oceans",
        "operating systems": "Operating System",
        "physics": "Physics",
        "probability": "Probability",
        "relativity": "General Relativity",
        "reliability": "Systems Reliability",
        "science": "Science",
        "scientific evidence": "Evidence and Experiments",
        "scientific laws": "Scientific Models, Laws, and Theories",
        "scientific method": "Scientific Method",
        "scientific models": "Scientific Models, Laws, and Theories",
        "scientific theories": "Scientific Models, Laws, and Theories",
        "si iv absorption": "Si IV Absorption",
        "software engineering": "Software Engineering",
        "spectral dip scan": "ANH Spectral Dip Scan",
        "spectral dip scans": "ANH Spectral Dip Scan",
        "a spectral dip scan": "ANH Spectral Dip Scan",
        "statistics": "Statistics",
        "system reliability": "Systems Reliability",
        "systems": "Computing",
        "the engineering design process": "Engineering Design Process",
        "thermodynamics": "Thermodynamics",
        "water cycle": "Water Cycle",
        "wave": "Waves",
        "waves": "Waves",
        "what organ pumps blood": "Heart",
        "what organ pumps blood throughout the body": "Heart",
        "what organ pumps blood through out the body": "Heart",
        "zodiac": "Zodiac",
        "the zodiac": "Zodiac",
        "zodiac signs": "Zodiac",
        "ww2": "World War II",
        "wwii": "World War II",
        "world war 2": "World War II",
        "world war ii": "World War II",
        "world war two": "World War II",
        "greece": "Ancient Greece and Democracy",
        "ancient greece": "Ancient Greece and Democracy",
    }

    FOLLOW_UP_PROMPTS = {
        "examples",
        "examples?",
        "go deeper",
        "go on",
        "tell me more",
        "what about that",
        "what about that?",
        "what else",
        "why",
        "what do you mean",
        "how so",
        "explain more",
    }

    @classmethod
    def build_response(
        cls,
        *,
        prompt: str,
        interaction_profile,
        knowledge_service: KnowledgeService | None,
        recent_interactions: list[dict[str, object]],
    ) -> dict[str, object] | None:
        follow_up = cls.build_follow_up_response(
            prompt=prompt,
            interaction_profile=interaction_profile,
            recent_interactions=recent_interactions,
        )
        if follow_up is not None:
            return follow_up

        cleaned_prompt = cls.clean_prompt(prompt)
        if not cleaned_prompt:
            return None
        if knowledge_service is None:
            return None

        lookup = cls._lookup_candidates(
            prompt=prompt,
            cleaned_prompt=cleaned_prompt,
            knowledge_service=knowledge_service,
        )
        if lookup is None or lookup.primary is None:
            return None
        if not cls._should_use_local_knowledge_response(
            prompt=prompt,
            cleaned_prompt=cleaned_prompt,
            lookup=lookup,
        ):
            return None
        return cls._entry_response(
            prompt=prompt,
            interaction_profile=interaction_profile,
            lookup=lookup,
            recent_interactions=recent_interactions,
        )

    @classmethod
    def build_follow_up_response(
        cls,
        *,
        prompt: str,
        interaction_profile,
        recent_interactions: list[dict[str, object]],
    ) -> dict[str, object] | None:
        if not recent_interactions:
            return None
        normalized = cls._lookup_ready_text(prompt)
        if normalized not in cls.FOLLOW_UP_PROMPTS:
            return None
        latest = recent_interactions[0]
        latest_response = latest.get("response") if isinstance(latest.get("response"), dict) else {}
        latest_mode = str(latest_response.get("mode") or latest.get("mode") or "").strip()
        if latest_mode != "research":
            return None
        surface = latest_response.get("domain_surface") if isinstance(latest_response.get("domain_surface"), dict) else {}
        lane = str(surface.get("lane") or "").strip()
        if lane != "knowledge":
            return None

        topic = str(surface.get("topic") or latest.get("prompt") or "that topic").strip()
        summary = str(latest_response.get("summary") or latest.get("summary") or "").strip()
        secondary = str(surface.get("secondary") or "").strip()
        style = InteractionStylePolicy.interaction_style(interaction_profile)
        recent_texts = ResponseVariationLayer.recent_surface_texts(recent_interactions)

        if style == "direct":
            pool = (
                f"The short version is that {summary.rstrip('.')}.",
                f"The main point is still {summary.rstrip('.')}.",
                f"For {topic}, the core idea is {summary.rstrip('.')}.",
                f"{summary.rstrip('.')}. That is the useful anchor.",
            )
        elif style == "collab":
            pool = (
                f"A deeper look at {topic}: {summary.rstrip('.')}.",
                f"Another useful angle on {topic}: {summary.rstrip('.')}.",
                f"Staying with {topic}, the cleanest continuation is {summary.rstrip('.')}.",
                f"The next useful layer is {summary.rstrip('.')}.",
            )
        else:
            pool = (
                f"Another way to frame {topic} is this: {summary.rstrip('.')}.",
                f"The clearest continuation is still {summary.rstrip('.')}.",
                f"For {topic}, the main takeaway remains {summary.rstrip('.')}.",
                f"A useful next angle is {summary.rstrip('.')}.",
            )
        if secondary:
            pool = pool + (
                f"It also helps to connect {topic} to {secondary}.",
                f"A useful comparison point here is {secondary}.",
            )

        reply = ResponseVariationLayer.select_from_pool(
            pool,
            seed_parts=[normalized, topic, secondary, style, "knowledge_follow_up"],
            recent_texts=recent_texts,
        )
        response = ResearchResponse(
            mode="research",
            kind="research.summary",
            summary=reply,
            findings=[],
        ).to_dict()
        response["reply"] = reply
        response["user_facing_answer"] = reply
        response["domain_surface"] = {
            "lane": "knowledge",
            "topic": topic,
            "secondary": secondary or None,
        }
        response["explanation_mode"] = "deeper"
        response["local_knowledge_access"] = {
            "local_knowledge_consulted": True,
            "local_knowledge_match": True,
            "knowledge_entry_id": str(surface.get("entry_id") or "").strip() or None,
            "knowledge_match_type": "follow_up_continuity",
            "final_source": "curated_local_knowledge",
        }
        return response

    @classmethod
    def _entry_response(
        cls,
        *,
        prompt: str,
        interaction_profile,
        lookup: KnowledgeLookupResult,
        recent_interactions: list[dict[str, object]],
    ) -> dict[str, object]:
        if lookup.mode == "comparison" and lookup.primary is not None and lookup.secondary is not None:
            return cls._comparison_response(
                prompt=prompt,
                interaction_profile=interaction_profile,
                lookup=lookup,
                recent_interactions=recent_interactions,
            )
        entry = lookup.primary
        assert entry is not None
        style = InteractionStylePolicy.interaction_style(interaction_profile)
        recent_texts = ResponseVariationLayer.recent_surface_texts(recent_interactions)
        shape = ResponseVariationLayer.response_shape(
            prompt=prompt,
            interaction_style=style,
            route_mode="research",
        )
        lead_verb = ResponseVariationLayer.choose_shape_verb(
            prompt=prompt,
            interaction_style=style,
            route_mode="research",
            recent_texts=recent_texts,
        )
        base = entry.summary_medium or entry.summary_short or entry.title
        topic = entry.title
        related = list(entry.related_topics[:2])
        key_point = entry.key_points[0] if entry.key_points else ""

        if style == "direct":
            lead_pool = (
                f"{topic}: {base}",
                base,
                f"{topic} is {base[0].lower() + base[1:]}" if base and base[0].isupper() else base,
            )
            if key_point:
                key_pool = (key_point, f"Key point: {key_point}")
            else:
                key_pool = ()
        elif style == "collab":
            lead_pool = (
                f"A clean way to {lead_verb} {topic} is this: {base}",
                f"Here is a grounded way to frame {topic}: {base}",
                f"{topic} is easiest to frame this way: {base}",
            )
            if key_point:
                key_pool = (
                    f"The part I would keep in view is {key_point}.",
                    f"A strong anchor is {key_point}.",
                    f"The useful thread here is {key_point}.",
                )
            else:
                key_pool = ()
        else:
            lead_pool = (
                f"{topic} is best summarized this way: {base}",
                f"A clear summary of {topic} is: {base}",
                f"For {topic}, a grounded starting point is: {base}",
            )
            if key_point:
                key_pool = (
                    f"The main point is {key_point}.",
                    f"A helpful anchor is {key_point}.",
                    f"The key thing to keep in mind is {key_point}.",
                )
            else:
                key_pool = ()

        parts = [("lead", lead_pool)]
        if key_pool:
            parts.append(("key", key_pool))
        if entry.formula is not None and entry.formula.formula_text:
            parts.append(("formula", (f"Formula: {entry.formula.formula_text}",)))
        if "simply" in cls._lookup_ready_text(prompt):
            example = entry.examples[0] if entry.examples else "for example, the useful pattern is easier to see when you compare a concentrated state with a more spread-out one"
            parts.append(("example", (f"For example: {example}",)))
        if related:
            related_text = " and ".join(related)
            if shape == "analogy":
                close_pool = (
                    f"You can think of it next to {related_text} to keep the comparison grounded.",
                    f"A useful analogy anchor here is how it relates to {related_text}.",
                )
            else:
                close_pool = (
                    f"It also relates to {related_text}.",
                    f"It helps to place it next to {related_text}.",
                    f"It connects to {related_text} as well.",
                )
            parts.append(("close", close_pool))

        reply = ResponseVariationLayer.realize(
            parts=parts,
            seed_parts=[prompt, style, topic, "knowledge_entry"],
            recent_texts=recent_texts,
        )
        response = ResearchResponse(
            mode="research",
            kind="research.summary",
            summary=reply,
            findings=[],
        ).to_dict()
        response["reply"] = reply
        response["user_facing_answer"] = reply
        response["domain_surface"] = {
            "lane": "knowledge",
            "topic": topic,
            "secondary": related[0] if related else None,
            "entry_id": entry.id,
        }
        response["local_knowledge_access"] = {
            "local_knowledge_consulted": True,
            "local_knowledge_match": True,
            "knowledge_entry_id": entry.id,
            "knowledge_match_type": cls._match_type(lookup=lookup),
            "matched_alias": lookup.matched_alias,
            "score": lookup.score,
            "category": entry.category,
            "final_source": "curated_local_knowledge",
        }
        response["route"] = {
            "source": "curated_local_knowledge",
            "reason": "Curated local knowledge matched before weak research routing.",
            "strength": "high",
            "confidence": lookup.score,
        }
        response["route_support_signals"] = {
            "broad_explanatory_prompt": cls._looks_like_knowledge_prompt(prompt),
            "blocked_knowledge_prompt": False,
            "local_knowledge_consulted": True,
            "local_knowledge_match": True,
        }
        response["response_tone_blend"] = {
            "tone_profile": "casual_explanation" if " lol" in cls._lookup_ready_text(prompt) else "formal_explanation",
            "interaction_style": style,
        }
        response["reasoning_state"] = {
            "current_path": "research:research.summary",
            "canonical_subject": entry.title.lower(),
            "resolved_prompt": prompt,
            "continuation_target": entry.title,
            "selected_mode": style,
            "turn_status": "routed",
        }
        response["pipeline_execution"] = {"execution_type": "local_knowledge", "executed": True}
        response["pipeline_packaging"] = {"package_type": "structured", "final_source": "curated_local_knowledge"}
        response["pipeline_trace"] = {
            "reasoning_frame": {"frame_type": "curated-local-knowledge"},
            "stage_contracts": {},
        }
        return response

    @classmethod
    def _comparison_response(
        cls,
        *,
        prompt: str,
        interaction_profile,
        lookup: KnowledgeLookupResult,
        recent_interactions: list[dict[str, object]],
    ) -> dict[str, object]:
        left = lookup.primary
        right = lookup.secondary
        assert left is not None
        assert right is not None
        style = InteractionStylePolicy.interaction_style(interaction_profile)
        recent_texts = ResponseVariationLayer.recent_surface_texts(recent_interactions)
        base = ExplanationResponseBuilder._comparison_answer(
            left,
            right,
            relation=lookup.comparison_relation,
            prompt=prompt,
            strategy="compare_contrast",
        )
        if style == "direct":
            lead_pool = (
                base,
                f"{left.title} and {right.title}: {base}",
            )
        elif style == "collab":
            lead_pool = (
                f"Here is the grounded connection: {base}",
                f"A clean way to connect them is this: {base}",
            )
        else:
            lead_pool = (
                f"A grounded way to connect {left.title} and {right.title} is this: {base}",
                base,
            )
        reply = ResponseVariationLayer.select_from_pool(
            lead_pool,
            seed_parts=[prompt, style, left.id, right.id, "knowledge_comparison"],
            recent_texts=recent_texts,
        )
        response = ResearchResponse(
            mode="research",
            kind="research.summary",
            summary=reply,
            findings=[],
        ).to_dict()
        response["reply"] = reply
        response["user_facing_answer"] = reply
        response["domain_surface"] = {
            "lane": "knowledge",
            "topic": left.title,
            "secondary": right.title,
            "entry_id": left.id,
            "secondary_entry_id": right.id,
        }
        response["local_knowledge_access"] = {
            "local_knowledge_consulted": True,
            "local_knowledge_match": True,
            "knowledge_entry_id": left.id,
            "secondary_entry_id": right.id,
            "knowledge_match_type": "comparison",
            "matched_alias": lookup.matched_alias,
            "score": lookup.score,
            "category": left.category,
            "final_source": "curated_local_knowledge",
        }
        response["route"] = {
            "source": "curated_local_knowledge",
            "reason": "Curated local knowledge matched a relation/comparison before weak research routing.",
            "strength": "high",
            "confidence": lookup.score,
        }
        response["route_support_signals"] = {
            "broad_explanatory_prompt": cls._looks_like_knowledge_prompt(prompt),
            "blocked_knowledge_prompt": False,
            "local_knowledge_consulted": True,
            "local_knowledge_match": True,
            "knowledge_match_type": "comparison",
        }
        response["response_tone_blend"] = {
            "tone_profile": "formal_explanation",
            "interaction_style": style,
        }
        response["reasoning_state"] = {
            "current_path": "research:research.summary",
            "canonical_subject": left.title.lower(),
            "secondary_subject": right.title.lower(),
            "resolved_prompt": prompt,
            "continuation_target": left.title,
            "selected_mode": style,
            "turn_status": "routed",
        }
        response["pipeline_execution"] = {"execution_type": "local_knowledge", "executed": True}
        response["pipeline_packaging"] = {"package_type": "structured", "final_source": "curated_local_knowledge"}
        response["pipeline_trace"] = {
            "reasoning_frame": {"frame_type": "curated-local-knowledge-comparison"},
            "stage_contracts": {},
        }
        return response

    @staticmethod
    def _lookup_candidates(
        *,
        prompt: str,
        cleaned_prompt: str,
        knowledge_service: KnowledgeService,
    ) -> KnowledgeLookupResult | None:
        mapped = KnowledgeSurfaceSupport.BROAD_TOPIC_LOOKUP_ALIASES.get(
            KnowledgeSurfaceSupport._lookup_ready_text(cleaned_prompt)
        )
        candidates = tuple(
            dict.fromkeys(
                candidate
                for candidate in (mapped, cleaned_prompt, prompt)
                if str(candidate or "").strip()
            )
        )
        for candidate in candidates:
            lookup = knowledge_service.lookup(candidate)
            if lookup is not None:
                return lookup
        for candidate in candidates:
            lookup = knowledge_service.partial_lookup(candidate)
            if lookup is not None:
                return lookup
        return None

    @classmethod
    def _should_use_local_knowledge_response(
        cls,
        *,
        prompt: str,
        cleaned_prompt: str,
        lookup: KnowledgeLookupResult,
    ) -> bool:
        normalized = cls._strip_social_lead_in(cls._lookup_ready_text(prompt))
        cleaned = cls._lookup_ready_text(cleaned_prompt)
        if not normalized or not cleaned:
            return False
        if cls._should_intercept_surface(prompt=prompt, cleaned_prompt=cleaned_prompt):
            return True
        if cleaned in cls.GENERIC_TOPIC_ANCHORS or normalized in cls.GENERIC_TOPIC_ANCHORS:
            return True
        if cls._looks_like_knowledge_prompt(prompt) and lookup.score >= 0.8:
            return True
        return False

    @staticmethod
    def _match_type(*, lookup: KnowledgeLookupResult) -> str:
        if lookup.mode == "comparison":
            return "comparison"
        if lookup.partial:
            return "partial"
        if lookup.matched_alias:
            return "alias"
        return "entry"

    @staticmethod
    def _looks_like_knowledge_prompt(prompt: str) -> bool:
        normalized = KnowledgeSurfaceSupport._strip_social_lead_in(KnowledgeSurfaceSupport._lookup_ready_text(prompt))
        starters = (
            "can you tell me about ",
            "also tell me about ",
            "teach me about ",
            "teach me step by step what ",
            "teach me step by step ",
            "teach me what ",
            "teach me ",
            "teach ",
            "go deeper on ",
            "go deeper into ",
            "go deeper about ",
            "let's pick up ",
            "lets pick up ",
            "pick up ",
            "likewise ive been thinking about ",
            "likewise i've been thinking about ",
            "likewise i have been thinking about ",
            "ive been thinking about ",
            "i've been thinking about ",
            "i have been thinking about ",
            "tell me about ",
            "what do you know about ",
            "what does ",
            "what organ ",
            "what is ",
            "what are ",
            "what's ",
            "whats ",
            "how does ",
            "explain ",
            "research ",
            "hey can you explain ",
            "can you explain ",
            "so what's the deal with ",
            "so whats the deal with ",
            "what do ",
        )
        return any(normalized.startswith(prefix) for prefix in starters)

    @classmethod
    def _should_intercept_surface(cls, *, prompt: str, cleaned_prompt: str) -> bool:
        normalized = cls._strip_social_lead_in(cls._lookup_ready_text(prompt))
        cleaned = cls._lookup_ready_text(cleaned_prompt)
        if not normalized or not cleaned:
            return False

        generic_anchor = (
            cleaned in cls.GENERIC_TOPIC_ANCHORS
            or normalized in cls.GENERIC_TOPIC_ANCHORS
            or cleaned in cls.BROAD_TOPIC_LOOKUP_ALIASES
            or normalized in cls.BROAD_TOPIC_LOOKUP_ALIASES
        )

        if normalized.startswith("research "):
            return generic_anchor

        wrapper_prefixes = (
            "can you tell me about ",
            "also tell me about ",
            "teach me about ",
            "teach me step by step what ",
            "teach me step by step ",
            "teach me what ",
            "teach me ",
            "go deeper on ",
            "go deeper into ",
            "go deeper about ",
            "let's pick up ",
            "lets pick up ",
            "pick up ",
            "likewise ive been thinking about ",
            "likewise i've been thinking about ",
            "likewise i have been thinking about ",
            "ive been thinking about ",
            "i've been thinking about ",
            "i have been thinking about ",
            "hey can you explain ",
            "can you explain ",
            "so what's the deal with ",
            "so whats the deal with ",
            "what do you know about ",
            "how does ",
            "what does ",
        )
        if generic_anchor and any(normalized.startswith(prefix) for prefix in wrapper_prefixes):
            return True

        if generic_anchor and any(normalized.endswith(suffix) for suffix in (" lol", " again", " please", " for me")):
            return True

        if normalized.startswith("what do ") and cleaned in cls.GENERIC_TOPIC_ANCHORS:
            return True

        if generic_anchor:
            return True

        return False

    @staticmethod
    def clean_prompt(prompt: str) -> str:
        normalized = KnowledgeSurfaceSupport._strip_social_lead_in(
            KnowledgeSurfaceSupport._lookup_ready_text(prompt)
        )
        if not normalized:
            return ""
        subject_focus = FocusResolutionSupport.subject_focus(normalized).focus
        prefixes = (
            "can you tell me about ",
            "also tell me about ",
            "teach me about ",
            "teach me step by step what ",
            "teach me step by step ",
            "teach me what ",
            "teach me ",
            "teach ",
            "go deeper on ",
            "go deeper into ",
            "go deeper about ",
            "let's pick up ",
            "lets pick up ",
            "pick up ",
            "likewise ive been thinking about ",
            "likewise i've been thinking about ",
            "likewise i have been thinking about ",
            "ive been thinking about ",
            "i've been thinking about ",
            "i have been thinking about ",
            "hey can you explain ",
            "can you explain ",
            "so what's the deal with ",
            "so whats the deal with ",
            "tell me about ",
            "what do you know about ",
            "what does ",
            "what organ ",
            "what is ",
            "what's the deal with ",
            "whats the deal with ",
            "what's ",
            "whats ",
            "how does ",
            "explain ",
            "research ",
            "what do ",
        )
        cleaned = normalized
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix) :].strip()
                    changed = True
                    break
        for suffix in (" again", " lol", " please", " for me"):
            if cleaned.endswith(suffix):
                cleaned = cleaned[: -len(suffix)].strip()
        if cleaned.endswith(" mean"):
            cleaned = cleaned[: -len(" mean")].strip()
        if cleaned.endswith(" work"):
            cleaned = cleaned[: -len(" work")].strip()
        if cleaned.endswith(" works"):
            cleaned = cleaned[: -len(" works")].strip()
        if cleaned.endswith(" step by step"):
            cleaned = cleaned[: -len(" step by step")].strip()
        if cleaned.startswith("what ") and cleaned.endswith(" is"):
            cleaned = cleaned[len("what ") : -len(" is")].strip()
        if cleaned.startswith("a ") and cleaned.endswith(" is"):
            cleaned = cleaned[len("a ") : -len(" is")].strip()
        if normalized.startswith("what do ") and cleaned:
            return cleaned
        for marker in (
            " like i'm ",
            " like im ",
            " like i am ",
            " without dumbing it down",
            " but not ",
        ):
            if marker in cleaned:
                cleaned = cleaned.split(marker, 1)[0].strip()
        return cleaned or subject_focus or normalized

    @staticmethod
    def _strip_social_lead_in(text: str) -> str:
        normalized = str(text or "").strip()
        for prefix in (
            "hey buddy ",
            "hi buddy ",
            "hello buddy ",
            "hey friend ",
            "hi friend ",
            "hello friend ",
            "hey pal ",
            "hi pal ",
            "hey lumen ",
            "hi lumen ",
            "hello lumen ",
        ):
            if normalized.startswith(prefix):
                return normalized[len(prefix) :].strip()
        return normalized

    @classmethod
    def _lookup_ready_text(cls, prompt: str) -> str:
        return cls._prompt_nlu.analyze(prompt).surface_views.lookup_ready_text
