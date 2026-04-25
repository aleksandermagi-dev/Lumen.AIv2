from lumen.nlu.explanatory_intent_policy import ExplanatoryIntentPolicy
from lumen.reasoning.explanatory_support_policy import ExplanatorySupportPolicy


def test_explanatory_intent_policy_detects_topic_only_subject() -> None:
    assert ExplanatoryIntentPolicy.looks_like_topic_only_query("george washington") is True
    assert ExplanatoryIntentPolicy.looks_like_topic_only_query("summarize archive structure") is False


def test_explanatory_intent_policy_handles_social_prefixed_explanatory_prompt() -> None:
    assert ExplanatoryIntentPolicy.looks_like_broad_explanatory_prompt("hey what is voltage lol") is True
    assert ExplanatoryIntentPolicy.looks_like_broad_explanatory_prompt("hello there") is False


def test_explanatory_intent_policy_blocks_system_summary_prompts() -> None:
    assert ExplanatoryIntentPolicy.looks_like_broad_explanatory_prompt("summarize the current archive structure") is False
    assert ExplanatoryIntentPolicy.is_blocked_knowledge_explanatory_prompt("report session confidence") is True
    assert ExplanatoryIntentPolicy.looks_like_broad_explanatory_prompt("prove which religion is true") is False


def test_explanatory_intent_policy_detects_explanatory_entities() -> None:
    assert ExplanatoryIntentPolicy.has_explanatory_entities(
        [
            {"label": "formula", "value": "quadratic formula"},
        ]
    ) is True
    assert ExplanatoryIntentPolicy.has_explanatory_entities(
        [
            {"label": "tool", "value": "anh"},
        ]
    ) is False


def test_explanatory_support_policy_handles_addressed_explanatory_prompt() -> None:
    signals = ExplanatorySupportPolicy.evaluate(prompt="Hey Lumen, what are watts?")

    assert signals.normalized_prompt == "what are watts"
    assert signals.broad_explanatory_prompt is True
