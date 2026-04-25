from lumen.nlu.language_detector import LanguageDetector
from lumen.nlu.prompt_nlu import PromptNLU
from lumen.nlu.text_normalizer import TextNormalizer
from lumen.nlu.topic_normalizer import TopicNormalizer


def test_language_detector_defaults_to_english_for_common_prompts() -> None:
    language = LanguageDetector().detect("create a migration plan for lumen")

    assert language.code == "en"
    assert language.confidence >= 0.55


def test_language_detector_can_pick_up_basic_spanish_prompt() -> None:
    language = LanguageDetector().detect("crear un plan y comparar el resumen")

    assert language.code == "es"
    assert language.confidence >= 0.55


def test_topic_normalizer_extracts_stable_topic_phrase() -> None:
    topic = TopicNormalizer().normalize("create a migration plan for lumen")

    assert topic.value == "migration plan for lumen"
    assert "migration" in topic.tokens
    assert "lumen" in topic.tokens


def test_prompt_nlu_returns_language_and_topic() -> None:
    understanding = PromptNLU().analyze("summarize the current archive structure")

    assert understanding.language.code == "en"
    assert understanding.topic.value == "current archive structure"
    assert understanding.intent.label == "research"
    assert "archive" in understanding.topic.tokens
    assert any(entity.value == "archive" for entity in understanding.entities)


def test_prompt_nlu_extracts_planning_intent_and_entities() -> None:
    understanding = PromptNLU().analyze("create a migration plan for lumen routing")

    assert understanding.intent.label == "planning"
    assert understanding.topic.value == "migration plan for lumen routing"
    assert any(entity.value == "migration" for entity in understanding.entities)
    assert any(entity.value == "routing" for entity in understanding.entities)


def test_prompt_nlu_keeps_surface_normalization_separate_from_intent_ready_text() -> None:
    understanding = PromptNLU().analyze("yo can you summarize the archive")

    assert understanding.normalized_text == "yo can you summarize the archive"
    assert understanding.intent_ready_text == "summarize the archive"
    assert understanding.intent.label == "research"


def test_text_normalizer_handles_broader_slang_and_short_forms() -> None:
    normalized = TextNormalizer.normalize("tysm lumin, how r u rn")

    assert normalized == "thank you so much lumen how are you right now"


def test_prompt_nlu_survives_messy_planning_phrase() -> None:
    understanding = PromptNLU().analyze("plz help me figure out the migration plan cuz im stuck")

    assert understanding.intent.label == "planning"
    assert "migration" in understanding.topic.tokens


def test_text_normalizer_handles_fillers_and_short_connection_words() -> None:
    normalized = TextNormalizer.normalize("umm can ya summarize w the archive abt ga real quick")

    assert normalized == "can you summarize with the archive about ga quickly"


def test_prompt_nlu_extracts_named_subject_entity() -> None:
    understanding = PromptNLU().analyze("George Washington")

    assert understanding.intent.label == "research"
    assert any(entity.label == "person" for entity in understanding.entities)


def test_prompt_nlu_treats_topic_only_concept_as_research() -> None:
    understanding = PromptNLU().analyze("black holes")

    assert understanding.intent.label == "research"
    assert any(entity.label in {"concept", "object"} for entity in understanding.entities)


def test_prompt_nlu_extracts_formula_subject_as_explanatory_entity() -> None:
    understanding = PromptNLU().analyze("quadratic formula")

    assert understanding.intent.label == "research"
    assert any(entity.label == "formula" for entity in understanding.entities)


def test_prompt_nlu_extracts_system_subject_as_explanatory_entity() -> None:
    understanding = PromptNLU().analyze("feedback loop")

    assert understanding.intent.label == "research"
    assert any(entity.label == "system" for entity in understanding.entities)


def test_prompt_nlu_builds_shared_surface_views_for_noisy_math_prompt() -> None:
    understanding = PromptNLU().analyze("Hey Lumen, solve 3x² + 2x - 5 = 0")

    assert understanding.surface_views.raw_text == "Hey Lumen, solve 3x² + 2x - 5 = 0"
    assert understanding.surface_views.lookup_ready_text == "solve 3x² + 2x - 5 = 0"
    assert understanding.surface_views.tool_ready_text == "solve 3x² + 2x - 5 = 0"
    assert understanding.surface_views.tool_source_text == "solve 3x² + 2x - 5 = 0"


def test_prompt_nlu_builds_shared_surface_views_for_direct_lumen_address() -> None:
    understanding = PromptNLU().analyze("Lumen, tell me more about black holes")

    assert understanding.surface_views.route_ready_text == "tell me more about black holes"
    assert understanding.surface_views.lookup_ready_text == "tell me more about black holes"
    assert understanding.surface_views.tool_ready_text == "tell me more about black holes"
    assert understanding.surface_views.tool_source_text == "tell me more about black holes"


def test_prompt_nlu_adds_structure_and_repairs_educational_shorthand() -> None:
    understanding = PromptNLU().analyze("black holes like I'm smart but not a physist")

    assert understanding.structure.predicate == "explain"
    assert understanding.structure.subject == "black holes"
    assert "educational_shorthand" in understanding.structure.fragmentation_markers
    assert "physicist" in understanding.structure.reconstructed_text
    assert understanding.surface_views.route_ready_text == understanding.structure.reconstructed_text


def test_prompt_nlu_marks_follow_up_shorthand_as_fragment() -> None:
    understanding = PromptNLU().analyze("go deeper")

    assert understanding.structure.completeness == "fragment"
    assert "requires_context" in understanding.structure.ambiguity_flags
    assert "follow_up_shorthand" in understanding.structure.fragmentation_markers


def test_prompt_understanding_exposes_canonical_router_view() -> None:
    understanding = PromptNLU().analyze("black holes like I'm smart but not a physist")
    router_view = understanding.router_view()

    assert understanding.canonical_text == understanding.surface_views.reconstructed_text
    assert router_view.canonical_text == understanding.canonical_text
    assert router_view.route_ready_text == understanding.surface_views.route_ready_text
    assert router_view.structure_predicate == "explain"
    assert router_view.structure_subject == "black holes"
