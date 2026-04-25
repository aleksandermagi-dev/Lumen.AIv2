from pathlib import Path

from lumen.services.academic_support_service import AcademicSupportService


def test_academic_support_service_classifies_citation_and_math_workflows() -> None:
    citation = AcademicSupportService.classify(prompt="format this citation in APA: author Jane Doe; title Testing; year 2024")
    math = AcademicSupportService.classify(prompt="explain Bayes theorem from the basics")

    assert citation is not None
    assert citation.workflow == "citation_help"
    assert math is not None
    assert math.workflow == "math_science_support"
    assert math.support_level == "bridge"


def test_academic_support_service_formats_citation_from_supplied_metadata() -> None:
    decision = AcademicSupportService.classify(
        prompt="format this citation in APA: author Jane Doe; title Neural Feedback; year 2024; url https://example.com/paper"
    )

    response = AcademicSupportService.build_response(
        prompt="format this citation in APA: author Jane Doe; title Neural Feedback; year 2024; url https://example.com/paper",
        decision=decision,
    )

    assert response["citation_integrity_status"] == "formatted_from_supplied_metadata"
    assert "Jane Doe." in response["user_facing_answer"]
    assert response["capability_status"]["domain_id"] == "citation_support"


def test_academic_support_service_flags_incomplete_citation_metadata() -> None:
    decision = AcademicSupportService.classify(prompt='format this citation in MLA: "Only a Title"')

    response = AcademicSupportService.build_response(
        prompt='format this citation in MLA: "Only a Title"',
        decision=decision,
    )

    assert response["citation_integrity_status"] == "incomplete_source_metadata"
    assert "Missing fields" in response["user_facing_answer"]


def test_academic_support_service_builds_dataset_readiness_flags_from_local_csv(tmp_path: Path) -> None:
    dataset = tmp_path / "training.csv"
    dataset.write_text("feature_a,feature_b,label\n1,2,yes\n3,4,no\n", encoding="utf-8")
    decision = AcademicSupportService.classify(
        prompt="review this supervised learning dataset for label leakage and train/validation/test split",
        input_path=dataset,
    )

    response = AcademicSupportService.build_response(
        prompt="review this supervised learning dataset for label leakage and train/validation/test split",
        decision=decision,
        input_path=dataset,
    )

    assert response["dataset_readiness_flags"]["schema_visible"] is True
    assert response["dataset_readiness_flags"]["leakage_check_required"] is True
    assert "Visible columns" in response["user_facing_answer"]


def test_academic_support_service_reports_workflow_inventory() -> None:
    report = AcademicSupportService.build_report()

    workflows = {item["workflow"]: item for item in report["workflows"]}
    assert workflows["citation_help"]["domain_id"] == "citation_support"
    assert workflows["dataset_readiness"]["domain_id"] == "supervised_ml_data_support"
