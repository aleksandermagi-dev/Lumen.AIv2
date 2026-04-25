from lumen.nlu.starter_prompt_layer import StarterPromptLayer


def test_starter_prompt_layer_exposes_categories_and_prompts() -> None:
    categories = StarterPromptLayer.starter_prompts()

    assert categories
    assert any(category.key == "science" for category in categories)
    assert any("Explain black holes" in prompt for category in categories for prompt in category.prompts)


def test_starter_prompt_layer_detects_overlapping_word_pools() -> None:
    analysis = StarterPromptLayer.analyze("help me design and refine this system idea")

    assert "builder" in analysis.matched_categories
    assert "system" in analysis.matched_categories
    assert analysis.category_scores["builder"] >= 1
