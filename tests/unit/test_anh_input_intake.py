from pathlib import Path
import json
import zipfile

from tool_bundles.anh.adapters.anh_input_intake import ANHInputIntake


def test_anh_input_intake_recognizes_processed_csv_summary(tmp_path: Path) -> None:
    csv_path = tmp_path / "Full_Si_IV_Absorption_Dataset.csv"
    csv_path.write_text(
        "\n".join(
            [
                "File,Si IV 1393 λ (Å),Si IV 1393 v (km/s),Si IV 1402 λ (Å),Si IV 1402 v (km/s)",
                "1,1394.004,52,1403.013,51",
            ]
        ),
        encoding="utf-8",
    )

    result = ANHInputIntake().recognize(csv_path)

    assert result.category == "processed_results_input"
    assert result.execution_mode == "summarize_results"
    assert result.recognized_format == "csv_summary"
    assert result.processed_inputs == (csv_path.resolve(),)


def test_anh_input_intake_recognizes_text_manifest_and_expands_members(tmp_path: Path) -> None:
    raw_path = tmp_path / "m31_x1d.fits"
    raw_path.write_bytes(b"FAKEFITS")
    processed_path = tmp_path / "si_iv_summary.csv"
    processed_path.write_text(
        "\n".join(
            [
                "File,Si IV 1393 λ (Å),Si IV 1393 v (km/s),Si IV 1402 λ (Å),Si IV 1402 v (km/s)",
                "1,1394.004,52,1403.013,51",
            ]
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "inputs_manifest.txt"
    manifest_path.write_text(
        "\n".join(
            [
                raw_path.name,
                processed_path.name,
                "missing_x1d.fits",
            ]
        ),
        encoding="utf-8",
    )

    result = ANHInputIntake().recognize(manifest_path)

    assert result.category == "batch_input"
    assert result.execution_mode == "batch_mixed"
    assert raw_path.resolve() in result.raw_inputs
    assert processed_path.resolve() in result.processed_inputs
    assert any("missing_x1d.fits" in item["path"] for item in result.skipped_inputs)


def test_anh_input_intake_recognizes_json_processed_summary(tmp_path: Path) -> None:
    json_path = tmp_path / "candidate_results.json"
    json_path.write_text(
        json.dumps(
            {
                "candidate_files": [
                    {"filename": "a_x1d.fits", "velocity_kms": -219.0, "lambda_min": 1393.75}
                ]
            }
        ),
        encoding="utf-8",
    )

    result = ANHInputIntake().recognize(json_path)

    assert result.category == "processed_results_input"
    assert result.recognized_format == "json_summary"


def test_anh_input_intake_recognizes_raw_archive_zip(tmp_path: Path) -> None:
    zip_path = tmp_path / "mast_bundle.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("mast/lb6f07nrq_x1d.fits", b"FAKEFITS")
        archive.writestr("mast/notes.txt", "hello")

    result = ANHInputIntake().recognize(zip_path)

    assert result.category == "archive_input"
    assert result.execution_mode == "archive_scan"
    assert result.recognized_format == "mast_archive_zip"
    assert result.archive_inputs == (zip_path.resolve(),)
