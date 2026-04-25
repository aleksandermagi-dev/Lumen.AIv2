from lumen.knowledge.seed_data import SEED_ENTRIES
from lumen.knowledge.seed_hygiene import SeedHygiene
from lumen.knowledge.models import KnowledgeEntry


def test_seed_hygiene_accepts_current_seed_entries() -> None:
    assert SeedHygiene.lint_entries(SEED_ENTRIES) == []


def test_seed_hygiene_flags_broad_and_duplicate_aliases() -> None:
    issues = SeedHygiene.lint_entries(
        (
            KnowledgeEntry(
                id="test.one",
                title="One",
                entry_type="concept",
                category="test",
                aliases=("os", "duplicate alias"),
            ),
            KnowledgeEntry(
                id="test.two",
                title="Two",
                entry_type="concept",
                category="test",
                aliases=("duplicate alias",),
            ),
        )
    )

    reasons = {issue.reason for issue in issues}
    assert "overly_broad_alias" in reasons
    assert any(reason.startswith("duplicate_alias:") for reason in reasons)
