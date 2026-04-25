from lumen.knowledge.knowledge_service import KnowledgeService
from pathlib import Path

import pytest


def test_knowledge_service_seeds_entries_and_exact_lookup() -> None:
    service = KnowledgeService.in_memory()

    result = service.lookup("George Washington")

    assert result is not None
    assert result.mode == "entry"
    assert result.primary is not None
    assert result.primary.title == "George Washington"
    assert result.score >= 0.9


def test_knowledge_service_handles_formula_lookup() -> None:
    service = KnowledgeService.in_memory()

    result = service.lookup("What is the quadratic formula?")

    assert result is not None
    assert result.primary is not None
    assert result.primary.title == "Quadratic Formula"
    assert result.primary.formula is not None
    assert "sqrt" in result.primary.formula.formula_text


def test_knowledge_service_handles_reordered_alias_lookup() -> None:
    service = KnowledgeService.in_memory()

    result = service.lookup("saturn planet")

    assert result is not None
    assert result.primary is not None
    assert result.primary.title == "Saturn"


def test_knowledge_service_handles_simple_plural_subject_lookup() -> None:
    service = KnowledgeService.in_memory()

    result = service.lookup("quadratic formulas")

    assert result is not None
    assert result.primary is not None
    assert result.primary.title == "Quadratic Formula"


def test_knowledge_service_handles_comparison_lookup() -> None:
    service = KnowledgeService.in_memory()

    result = service.lookup("Black hole vs neutron star")

    assert result is not None
    assert result.mode == "comparison"
    assert result.primary is not None
    assert result.secondary is not None
    assert result.primary.title == "Black Hole"
    assert result.secondary.title == "Neutron Star"


def test_knowledge_service_handles_reversed_comparison_lookup() -> None:
    service = KnowledgeService.in_memory()

    result = service.lookup("neutron star vs black hole")

    assert result is not None
    assert result.mode == "comparison"
    assert result.primary is not None
    assert result.secondary is not None
    assert result.primary.title == "Neutron Star"
    assert result.secondary.title == "Black Hole"


def test_knowledge_service_handles_noisy_comparison_lookup() -> None:
    service = KnowledgeService.in_memory()

    result = service.lookup("hey what's a neutron star versus a black hole lol")

    assert result is not None
    assert result.mode == "comparison"
    assert result.primary is not None
    assert result.secondary is not None
    assert result.primary.title == "Neutron Star"
    assert result.secondary.title == "Black Hole"


def test_knowledge_service_handles_basic_unit_comparison_lookup() -> None:
    service = KnowledgeService.in_memory()

    result = service.lookup("ohms vs watts")

    assert result is not None
    assert result.mode == "comparison"
    assert result.primary is not None
    assert result.secondary is not None
    assert result.primary.title == "Resistance"
    assert result.secondary.title == "Power"


def test_knowledge_service_handles_new_history_lookup() -> None:
    service = KnowledgeService.in_memory()

    result = service.lookup("Abraham Lincoln")

    assert result is not None
    assert result.primary is not None
    assert result.primary.title == "Abraham Lincoln"


def test_knowledge_service_handles_expanded_history_lookup() -> None:
    service = KnowledgeService.in_memory()

    result = service.lookup("French Revolution")

    assert result is not None
    assert result.primary is not None
    assert result.primary.title == "French Revolution"


def test_knowledge_service_handles_new_astronomy_lookup() -> None:
    service = KnowledgeService.in_memory()

    result = service.lookup("event horizon")

    assert result is not None
    assert result.primary is not None
    assert result.primary.title == "Event Horizon"


@pytest.mark.parametrize("query", ["space", "space itself", "outer space", "universe", "cosmos"])
def test_knowledge_service_handles_broad_space_aliases(query: str) -> None:
    service = KnowledgeService.in_memory()

    result = service.lookup(query)

    assert result is not None
    assert result.primary is not None
    assert result.primary.title == "Space"


@pytest.mark.parametrize(
    ("query", "title"),
    [
        ("biology", "Biology"),
        ("chemistry", "Chemistry"),
        ("physics", "Physics"),
        ("math", "Mathematics"),
        ("mathematics", "Mathematics"),
        ("engineering", "Engineering"),
        ("earth science", "Earth Science"),
        ("computer science", "Computing"),
        ("computers", "Computing"),
    ],
)
def test_knowledge_service_handles_broad_domain_anchors(query: str, title: str) -> None:
    service = KnowledgeService.in_memory()

    result = service.lookup(query)

    assert result is not None
    assert result.primary is not None
    assert result.primary.title == title


@pytest.mark.parametrize(
    ("query", "title"),
    [
        ("what organ pumps blood throughout the body", "Heart"),
        ("what organ pumps blood through out the body", "Heart"),
        ("what is the zodiac", "Zodiac"),
        ("tell me about astrology", "Astrology"),
        ("horoscope", "Horoscope"),
    ],
)
def test_knowledge_service_handles_phase90_broad_and_cultural_aliases(query: str, title: str) -> None:
    service = KnowledgeService.in_memory()

    result = service.lookup(query)

    assert result is not None
    assert result.primary is not None
    assert result.primary.title == title


@pytest.mark.parametrize(
    ("query", "title"),
    [
        ("motion", "Motion"),
        ("waves", "Waves"),
        ("electricity", "Electricity"),
        ("thermodynamics", "Thermodynamics"),
        ("quantum basics", "Quantum Mechanics"),
        ("relativity basics", "General Relativity"),
        ("chemical bonding", "Chemical Bonding"),
        ("atoms and molecules", "Molecules and Compounds"),
        ("climate", "Weather and Climate"),
        ("geology", "Geology"),
        ("oceans", "Oceans"),
        ("water cycle", "Water Cycle"),
        ("engineering design process", "Engineering Design Process"),
        ("mechanical engineering", "Mechanical Engineering"),
        ("electrical engineering", "Electrical Engineering"),
        ("civil engineering", "Civil Engineering"),
        ("software engineering", "Software Engineering"),
        ("systems reliability", "Systems Reliability"),
        ("ancient civilizations", "Ancient Civilizations"),
        ("major wars", "Wars and Revolutions"),
        ("modern history", "Modern World Context"),
        ("algebra", "Algebra"),
        ("geometry", "Geometry"),
        ("logic", "Mathematical Logic"),
        ("scientific evidence", "Evidence and Experiments"),
        ("scientific models", "Scientific Models, Laws, and Theories"),
        ("measurement", "Measurement"),
        ("database", "Database"),
        ("cybersecurity", "Cybersecurity"),
        ("artificial intelligence", "Artificial Intelligence"),
    ],
)
def test_knowledge_service_handles_second_ring_domain_breadth(query: str, title: str) -> None:
    service = KnowledgeService.in_memory()

    result = service.lookup(query)

    assert result is not None
    assert result.primary is not None
    assert result.primary.title == title


@pytest.mark.parametrize(
    ("query", "title"),
    [
        ("what is ANH", "ANH"),
        ("astronomical node heuristics", "ANH"),
        ("spectral dip scan", "ANH Spectral Dip Scan"),
        ("MAST data", "MAST Data"),
        ("HST/COS", "HST/COS"),
        ("FITS spectrum", "FITS Spectrum"),
        ("Si IV absorption", "Si IV Absorption"),
        ("candidate velocity", "Candidate Velocity and Depth"),
        ("great attractor bulk flow", "Great Attractor Bulk-Flow Context"),
    ],
)
def test_knowledge_service_handles_anh_context_entries(query: str, title: str) -> None:
    service = KnowledgeService.in_memory()

    result = service.lookup(query)

    assert result is not None
    assert result.primary is not None
    assert result.primary.title == title


@pytest.mark.parametrize(
    ("query", "title"),
    [
        ("u.s. civil war", "American Civil War"),
        ("a squared plus b squared equals c squared", "Pythagorean Theorem"),
        ("speed velocity acceleration", "Speed, Velocity, and Acceleration"),
    ],
)
def test_knowledge_service_handles_brittle_taught_fact_aliases(query: str, title: str) -> None:
    service = KnowledgeService.in_memory()

    result = service.lookup(query)

    assert result is not None
    assert result.primary is not None
    assert result.primary.title == title


def test_knowledge_service_handles_great_attractor_lookup() -> None:
    service = KnowledgeService.in_memory()

    result = service.lookup("Great Attractor")

    assert result is not None
    assert result.primary is not None
    assert result.primary.title == "Great Attractor"


def test_knowledge_service_handles_short_astronomy_alias_lookup() -> None:
    service = KnowledgeService.in_memory()

    result = service.lookup("GA")

    assert result is not None
    assert result.primary is not None
    assert result.primary.title == "Great Attractor"


def test_knowledge_service_handles_entropy_lookup() -> None:
    service = KnowledgeService.in_memory()

    result = service.lookup("what is entropy")

    assert result is not None
    assert result.primary is not None
    assert result.primary.id == "physics.entropy"
    assert "spread out" in result.primary.summary_medium.lower()


def test_knowledge_service_handles_thermodynamic_entropy_lookup() -> None:
    service = KnowledgeService.in_memory()

    result = service.lookup("thermodynamic entropy")

    assert result is not None
    assert result.primary is not None
    assert result.primary.id == "physics.entropy"


def test_knowledge_service_handles_short_alias_with_domain_hint() -> None:
    service = KnowledgeService.in_memory()

    result = service.lookup("what is ga in astronomy")

    assert result is not None
    assert result.primary is not None
    assert result.primary.id == "astronomy.great_attractor"


def test_knowledge_service_handles_hyphenated_astronomy_lookup() -> None:
    service = KnowledgeService.in_memory()

    result = service.lookup("Hydra-Centaurus")

    assert result is not None
    assert result.primary is not None
    assert result.primary.title == "Hydra-Centaurus Supercluster"


def test_knowledge_service_handles_expanded_chemistry_lookup() -> None:
    service = KnowledgeService.in_memory()

    result = service.lookup("Periodic Table")

    assert result is not None
    assert result.primary is not None
    assert result.primary.title == "Periodic Table"


def test_knowledge_service_handles_expanded_biology_lookup() -> None:
    service = KnowledgeService.in_memory()

    result = service.lookup("Photosynthesis")

    assert result is not None
    assert result.primary is not None
    assert result.primary.title == "Photosynthesis"


def test_knowledge_service_handles_expanded_earth_science_lookup() -> None:
    service = KnowledgeService.in_memory()

    result = service.lookup("Weather vs Climate")

    assert result is not None
    assert result.primary is not None
    assert result.primary.title == "Weather and Climate"


def test_knowledge_service_handles_expanded_math_lookup() -> None:
    service = KnowledgeService.in_memory()

    result = service.lookup("Linear Algebra")

    assert result is not None
    assert result.primary is not None
    assert result.primary.title == "Linear Algebra"


def test_knowledge_service_handles_glossary_backed_phrase_lookup() -> None:
    service = KnowledgeService.in_memory()

    result = service.lookup("the scientific method")

    assert result is not None
    assert result.primary is not None
    assert result.primary.title == "Scientific Method"


def test_knowledge_service_reapplies_seed_data_for_existing_db(tmp_path: Path) -> None:
    db_path = tmp_path / "knowledge.sqlite3"
    service = KnowledgeService.from_path(db_path)
    with service.db.connect() as connection:
        connection.execute("DELETE FROM knowledge_aliases WHERE entry_id = ?", ("astronomy.milky_way",))
        connection.execute("DELETE FROM knowledge_entries WHERE id = ?", ("astronomy.milky_way",))
        connection.commit()

    reloaded = KnowledgeService.from_path(db_path)
    result = reloaded.lookup("Milky Way")

    assert result is not None
    assert result.primary is not None
    assert result.primary.title == "Milky Way"


def test_knowledge_service_handles_new_engineering_lookup() -> None:
    service = KnowledgeService.in_memory()

    result = service.lookup("rocket engine")

    assert result is not None
    assert result.primary is not None
    assert result.primary.title == "Rocket Engine"


def test_knowledge_service_exposes_related_connections() -> None:
    service = KnowledgeService.in_memory()
    result = service.lookup("rocket engine")

    assert result is not None
    assert result.primary is not None
    assert ("part_of", "Propulsion System") in service.related_connections(result.primary.id)


def test_knowledge_service_returns_none_for_weak_unknown_match() -> None:
    service = KnowledgeService.in_memory()

    result = service.lookup("hyperbolic lemon engine")

    assert result is None


def test_knowledge_service_handles_new_astronomy_fundamental_lookup() -> None:
    service = KnowledgeService.in_memory()

    result = service.lookup("escape velocity")

    assert result is not None
    assert result.primary is not None
    assert result.primary.title == "Escape Velocity"


def test_knowledge_service_handles_new_propulsion_metric_lookup() -> None:
    service = KnowledgeService.in_memory()

    result = service.lookup("specific impulse")

    assert result is not None
    assert result.primary is not None
    assert result.primary.title == "Specific Impulse"


def test_knowledge_service_handles_new_systems_lookup() -> None:
    service = KnowledgeService.in_memory()

    result = service.lookup("data structures")

    assert result is not None
    assert result.primary is not None
    assert result.primary.title == "Data Structure"


def test_knowledge_service_rejects_overly_ambiguous_short_aliases() -> None:
    service = KnowledgeService.in_memory()

    assert service.lookup("os") is None
    assert service.lookup("isp") is None
    assert service.lookup("ht") is None


def test_knowledge_service_keeps_generic_neighbor_prompt_as_no_match() -> None:
    service = KnowledgeService.in_memory()

    result = service.lookup("force field")

    assert result is None


def test_knowledge_service_handles_what_are_prompt_shape() -> None:
    service = KnowledgeService.in_memory()

    result = service.lookup("what are watts")

    assert result is not None
    assert result.primary is not None
    assert result.primary.title == "Power"


def test_knowledge_service_handles_what_do_mean_prompt_shape() -> None:
    service = KnowledgeService.in_memory()

    result = service.lookup("what do watts mean")

    assert result is not None
    assert result.primary is not None
    assert result.primary.title == "Power"


def test_knowledge_service_handles_what_does_mean_prompt_shape() -> None:
    service = KnowledgeService.in_memory()

    result = service.lookup("what does resistance mean")

    assert result is not None
    assert result.primary is not None
    assert result.primary.title == "Resistance"


def test_knowledge_service_exposes_category_overview() -> None:
    service = KnowledgeService.in_memory()

    overview = service.overview()

    assert overview["entry_count"] >= 1
    categories = {item["category"] for item in overview["categories"]}
    assert "astronomy" in categories
    assert "physics" in categories


def test_knowledge_service_handles_meaning_of_prompt_shape() -> None:
    service = KnowledgeService.in_memory()

    result = service.lookup("what is the meaning of voltage")

    assert result is not None
    assert result.primary is not None
    assert result.primary.title == "Voltage"


def test_knowledge_service_handles_could_you_explain_prompt_shape() -> None:
    service = KnowledgeService.in_memory()

    result = service.lookup("could you explain watts")

    assert result is not None
    assert result.primary is not None
    assert result.primary.title == "Power"


def test_knowledge_service_handles_noisy_addressed_lookup_prompt() -> None:
    service = KnowledgeService.in_memory()

    result = service.lookup("Hey Lumen, what do watts mean?")

    assert result is not None
    assert result.primary is not None
    assert result.primary.title == "Power"


def test_knowledge_service_records_lookup_diagnostics_for_miss() -> None:
    service = KnowledgeService.in_memory()

    result = service.lookup("hyperbolic lemon engine")

    assert result is None
    assert service.last_lookup_diagnostics["subject"] == "hyperbolic lemon engine"
    assert service.last_lookup_diagnostics["reason"] in {
        "direct_focus",
        "insufficient_dominance",
    }
