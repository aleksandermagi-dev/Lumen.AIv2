from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
import importlib.util
import json
import math
import shutil

from lumen.tools.registry_types import Artifact, BundleManifest, ToolRequest, ToolResult
from tool_bundles.anh.adapters.anh_input_intake import ANHInputIntake, ANHInputRecognitionResult
from tool_bundles.anh.adapters.anh_raw_intake_staging import ANHRawIntakeStaging, ANHRawStagingResult
from tool_bundles.anh.adapters.anh_result_summary import ANHResultSummary
from tool_bundles.anh.adapters.research_bundle_standard import (
    ResearchArtifactRecord,
    build_standard_summary,
    sha256_or_none,
    write_standard_artifacts,
)


DEFAULT_TARGET_LINES = (
    {"rest_wavelength": 1393.760, "title": "Si IV 1393 @ -219", "width": 0.8, "smooth_window": 5},
    {"rest_wavelength": 1402.773, "title": "Si IV 1402 @ -219", "width": 0.8, "smooth_window": 5},
    {"rest_wavelength": 1393.760, "title": "Si IV 1393 @ -300", "width": 1.0, "smooth_window": 5},
    {"rest_wavelength": 1402.773, "title": "Si IV 1402 @ -300", "width": 1.0, "smooth_window": 5},
    {"rest_wavelength": 1393.760, "title": "Si IV 1393 @ 0", "width": 1.0, "smooth_window": 5},
    {"rest_wavelength": 1402.773, "title": "Si IV 1402 @ 0", "width": 1.0, "smooth_window": 5},
)


@dataclass
class ANHSpectralDipScanAdapter:
    manifest: BundleManifest
    repo_root: Path
    WRAPPER_VERSION = "1.2.0"
    DEFAULT_MAX_FILES = 25
    DEFAULT_ARCHIVE_MAX_FILES = 100

    def execute(self, request: ToolRequest) -> ToolResult:
        run_dir = self._create_run_dir(request)
        output_dir = run_dir / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        run_id = run_dir.name
        target_lines = self._target_lines(request.params)

        if request.input_path is None:
            return self._build_invalid_input_result(
                request=request,
                run_dir=run_dir,
                output_dir=output_dir,
                run_id=run_id,
                reason="ANH spectral scan requires an input_path pointing to a FITS/x1d file, processed summary, manifest, directory, or raw archive.",
                target_lines=target_lines,
                intake=None,
            )

        intake = ANHInputIntake().recognize(
            request.input_path,
            max_files=self._effective_max_files(request=request, archive=False),
        )
        if intake.category == "invalid_input":
            return self._build_invalid_input_result(
                request=request,
                run_dir=run_dir,
                output_dir=output_dir,
                run_id=run_id,
                reason="; ".join(intake.reasons) or "No supported ANH-compatible inputs were found.",
                target_lines=target_lines,
                skipped_inputs=list(intake.skipped_inputs),
                intake=intake,
            )

        staging_result = self._stage_inputs(
            request=request,
            run_dir=run_dir,
            intake=intake,
        )
        staged_raw = list(staging_result.staged_good_raw)
        staged_processed = list(staging_result.staged_processed)

        file_results: list[dict[str, object]] = []
        logs: list[str] = list(staging_result.logs)
        artifact_records: list[ResearchArtifactRecord] = []
        analysis_module = None
        dependency_failure_reason: str | None = None
        if staged_raw:
            try:
                analysis_module = self._load_analysis_module()
                self._configure_headless_matplotlib()
            except Exception as exc:
                dependency_failure_reason = self._runtime_dependency_failure(exc)
                logs.append(dependency_failure_reason)
                for staged_input in staged_raw:
                    file_results.append(
                        {
                            "status": "error",
                            "filename": staged_input.name,
                            "staged_path": str(staged_input),
                            "source_kind": "raw_spectral_input",
                            "recognized_format": "mast_x1d_fits",
                            "line_detected": False,
                            "strongest_candidate": None,
                            "line_results": [],
                            "artifacts": [],
                            "failure_reason": dependency_failure_reason,
                        }
                    )

        if analysis_module is not None:
            for staged_input in staged_raw:
                result = self._run_single_file_analysis(
                    staged_input=staged_input,
                    output_dir=output_dir,
                    target_lines=target_lines,
                    smooth_window=int(request.params.get("smooth_window", 11)) if request.params else 11,
                    analysis_module=analysis_module,
                )
                file_results.append({"status": result["status"], **dict(result["summary"])})
                logs.extend(result["logs"])

        summary_adapter = ANHResultSummary()
        for staged_input in staged_processed:
            file_output_dir = output_dir / staged_input.stem
            file_output_dir.mkdir(parents=True, exist_ok=True)
            summarized = summary_adapter.summarize(staged_input=staged_input, output_dir=file_output_dir)
            file_results.append({"status": summarized.status, **dict(summarized.summary)})
            logs.extend(summarized.logs)
            artifact_records.extend(summarized.artifact_records)

        skipped_inputs = list(intake.skipped_inputs) + list(staging_result.staged_failed)
        skipped_non_spectral = list(staging_result.skipped_non_spectral)
        for item in file_results:
            if str(item.get("status")) != "ok":
                skipped_inputs.append(
                    {
                        "path": str(item.get("staged_path") or item.get("filename") or request.input_path),
                        "reason": str(item.get("failure_reason") or "analysis_failed"),
                    }
                )

        candidate_rankings = self._rank_candidates(
            file_results=file_results,
            rank_limit=int(request.params.get("rank_limit", 10)) if request.params else 10,
        )
        artifacts, generated_records = self._collect_output_artifacts(output_dir)
        artifact_records.extend(generated_records)
        ranking_artifact = output_dir / "candidate_rankings.json"
        ranking_artifact.write_text(json.dumps(candidate_rankings, indent=2), encoding="utf-8")
        artifacts.append(
            Artifact(
                name=ranking_artifact.name,
                path=ranking_artifact,
                media_type="application/json",
                description="Ranked ANH candidate files by strongest detected dip or reported velocity.",
            )
        )
        artifact_records.append(
            ResearchArtifactRecord(
                role="candidate_rankings",
                path=ranking_artifact,
                media_type="application/json",
                description="Ranked ANH candidate files by strongest detected dip or reported velocity.",
            )
        )

        analyzed_count = len(file_results)
        success_count = sum(1 for item in file_results if str(item.get("status")) == "ok")
        candidate_count = sum(1 for item in file_results if bool(item.get("line_detected")))
        partial_success = success_count > 0 and (success_count < analyzed_count or bool(skipped_inputs))
        status = "ok" if success_count > 0 and not partial_success else "partial" if success_count > 0 else "error"
        result_quality = self._result_quality(
            status=status,
            candidate_count=candidate_count,
            analyzed_count=analyzed_count,
            execution_mode=intake.execution_mode,
            dependency_failure_reason=dependency_failure_reason,
            staged_raw_count=len(staged_raw),
            staged_processed_count=len(staged_processed),
        )

        summary_payload = build_standard_summary(
            bundle_id=self.manifest.id,
            capability=request.capability,
            run_id=run_id,
            target_label="ANH Si IV Spectral Scan",
            input_files=[str(request.input_path)],
            provenance=self._provenance(
                request=request,
                run_id=run_id,
                target_lines=target_lines,
                intake=intake,
            ),
            analysis_status={
                "validated": intake.category != "invalid_input",
                "analysis_ran": bool(staged_raw),
                "plot_generated": any(artifact.name.endswith(".png") for artifact in artifacts),
                "line_detected": candidate_count > 0,
                "partial_success": partial_success,
                "result_quality": result_quality,
                "runtime_diagnostics": self._runtime_environment_diagnostics(),
                "failure_reason": self._failure_reason(
                    status=status,
                    dependency_failure_reason=dependency_failure_reason,
                    result_quality=result_quality,
                    staged_raw_count=len(staged_raw),
                    staged_processed_count=len(staged_processed),
                ),
            },
            batch_record={
                "files_discovered": len(staging_result.discovered_files),
                "files_extracted": len(staging_result.extracted_files),
                "files_analyzed": analyzed_count,
                "accepted_files": [str(item["staged_path"]) for item in file_results],
                "staged_good_count": len(staging_result.staged_good_raw),
                "staged_failed_count": len(staging_result.staged_failed),
                "skipped_non_spectral_count": len(staging_result.skipped_non_spectral),
                "processed_summary_count": len(staging_result.staged_processed),
                "manifest_paths": dict(staging_result.manifest_paths),
                "skipped_files": skipped_inputs,
                "skipped_non_spectral": skipped_non_spectral,
                "candidate_files": candidate_rankings,
                "strongest_candidate": candidate_rankings[0] if candidate_rankings else None,
                "member_classifications": [item.to_dict() for item in (staging_result.member_classifications or intake.member_classifications)],
            },
            domain_payload={
                "staging": staging_result.to_dict(),
                "files_analyzed": analyzed_count,
                "accepted_files": [self._file_summary(item) for item in file_results],
                "skipped_files": skipped_inputs,
                "skipped_non_spectral": skipped_non_spectral,
                "candidate_files": candidate_rankings,
                "strongest_candidate": candidate_rankings[0] if candidate_rankings else None,
                "selected_params": {
                    "target_lines": target_lines,
                    "smooth_window": int(request.params.get("smooth_window", 11)) if request.params else 11,
                    "scan_width": request.params.get("scan_width") if request.params else None,
                    "max_files": self._effective_max_files(request=request, archive=bool(intake.archive_inputs)),
                    "rank_limit": int(request.params.get("rank_limit", 10)) if request.params else 10,
                },
                "runtime_diagnostics": self._runtime_environment_diagnostics(),
            },
            produced_artifacts=artifact_records,
        )
        summary_payload["intake"] = intake.to_dict()

        generated_contract_artifacts = write_standard_artifacts(output_dir, summary_payload)
        for role, path in generated_contract_artifacts.items():
            artifacts.append(
                Artifact(
                    name=path.name,
                    path=path,
                    media_type="application/json" if path.suffix == ".json" else "text/plain",
                    description=f"Standardized ANH output: {role}",
                )
            )

        return ToolResult(
            status=status,
            tool_id=request.tool_id,
            capability=request.capability,
            summary=self._summary_text(
                status=status,
                analyzed_count=analyzed_count,
                candidate_count=candidate_count,
                candidate_rankings=candidate_rankings,
                execution_mode=intake.execution_mode,
                result_quality=result_quality,
            ),
            structured_data=summary_payload,
            artifacts=artifacts,
            logs=logs,
            provenance=summary_payload["provenance"],
            run_dir=run_dir,
            error=None if status == "ok" else str(summary_payload["analysis_status"]["failure_reason"]),
        )

    def _build_invalid_input_result(
        self,
        *,
        request: ToolRequest,
        run_dir: Path,
        output_dir: Path,
        run_id: str,
        reason: str,
        target_lines: list[dict[str, object]],
        skipped_inputs: list[dict[str, object]] | None = None,
        intake: ANHInputRecognitionResult | None,
    ) -> ToolResult:
        payload = build_standard_summary(
            bundle_id=self.manifest.id,
            capability=request.capability,
            run_id=run_id,
            target_label="ANH Si IV Spectral Scan",
            input_files=[str(request.input_path)] if request.input_path else [],
            provenance=self._provenance(request=request, run_id=run_id, target_lines=target_lines, intake=intake),
            analysis_status={
                "validated": False,
                "analysis_ran": False,
                "plot_generated": False,
                "line_detected": False,
                "partial_success": False,
                "result_quality": "invalid_input",
                "failure_reason": reason,
            },
            batch_record={
                "files_analyzed": 0,
                "accepted_files": [],
                "skipped_files": list(skipped_inputs or []),
                "candidate_files": [],
                "strongest_candidate": None,
                "member_classifications": [item.to_dict() for item in (intake.member_classifications if intake else ())],
            },
            domain_payload={
                "files_analyzed": 0,
                "accepted_files": [],
                "skipped_files": list(skipped_inputs or []),
                "candidate_files": [],
                "strongest_candidate": None,
                "selected_params": {
                    "target_lines": target_lines,
                },
            },
            produced_artifacts=[],
        )
        payload["intake"] = intake.to_dict() if intake is not None else {
            "category": "invalid_input",
            "execution_mode": "reject",
            "recognized_format": "missing_input",
            "confidence": "high",
            "reasons": [reason],
            "archive_inputs": [],
            "raw_inputs": [],
            "processed_inputs": [],
            "skipped_inputs": list(skipped_inputs or []),
            "member_classifications": [],
        }
        generated_contract_artifacts = write_standard_artifacts(output_dir, payload)
        artifacts = [
            Artifact(
                name=path.name,
                path=path,
                media_type="application/json" if path.suffix == ".json" else "text/plain",
                description=f"Standardized ANH output: {role}",
            )
            for role, path in generated_contract_artifacts.items()
        ]
        return ToolResult(
            status="error",
            tool_id=request.tool_id,
            capability=request.capability,
            summary=f"ANH could not run: {reason}",
            structured_data=payload,
            artifacts=artifacts,
            logs=[reason],
            provenance=payload["provenance"],
            run_dir=run_dir,
            error=reason,
        )

    @staticmethod
    def _stage_input_group(*, paths: tuple[Path, ...], staged_dir: Path) -> list[Path]:
        staged_dir.mkdir(parents=True, exist_ok=True)
        staged_paths: list[Path] = []
        for index, source in enumerate(paths, start=1):
            staged_name = source.name if index == 1 else f"{index:03d}_{source.name}"
            destination = staged_dir / staged_name
            shutil.copy2(source, destination)
            staged_paths.append(destination)
        return staged_paths

    def _stage_inputs(
        self,
        *,
        request: ToolRequest,
        run_dir: Path,
        intake: ANHInputRecognitionResult,
    ) -> ANHRawStagingResult:
        max_files = self._effective_max_files(request=request, archive=bool(intake.archive_inputs))
        if intake.archive_inputs:
            return ANHRawIntakeStaging().stage(
                source=intake.archive_inputs[0],
                run_dir=run_dir,
                max_files=max_files,
            )
        if request.input_path is not None and request.input_path.is_dir():
            return ANHRawIntakeStaging().stage(
                source=request.input_path.resolve(),
                run_dir=run_dir,
                max_files=max_files,
            )

        staged_raw = tuple(
            self._stage_input_group(
                paths=intake.raw_inputs,
                staged_dir=run_dir / "inputs" / "raw",
            )
        )
        staged_processed = tuple(
            self._stage_input_group(
                paths=intake.processed_inputs,
                staged_dir=run_dir / "inputs" / "processed",
            )
        )
        return ANHRawStagingResult(
            discovered_files=tuple(str(path) for path in intake.raw_inputs + intake.processed_inputs),
            extracted_files=(),
            staged_good_raw=staged_raw,
            staged_processed=staged_processed,
            staged_failed=tuple(intake.skipped_inputs),
            skipped_non_spectral=(),
            manifest_paths={},
            member_classifications=intake.member_classifications,
            logs=(),
        )

    def _effective_max_files(self, *, request: ToolRequest, archive: bool) -> int:
        if request.params and request.params.get("max_files") is not None:
            try:
                return max(1, int(request.params["max_files"]))
            except (TypeError, ValueError):
                return self.DEFAULT_ARCHIVE_MAX_FILES if archive else self.DEFAULT_MAX_FILES
        return self.DEFAULT_ARCHIVE_MAX_FILES if archive else self.DEFAULT_MAX_FILES

    def _run_single_file_analysis(
        self,
        *,
        staged_input: Path,
        output_dir: Path,
        target_lines: list[dict[str, object]],
        smooth_window: int,
        analysis_module,
    ) -> dict[str, object]:
        import matplotlib.pyplot as plt

        file_output_dir = output_dir / staged_input.stem
        file_output_dir.mkdir(parents=True, exist_ok=True)
        logs: list[str] = []
        try:
            runtime_result: dict[str, object] | None = None
            if hasattr(analysis_module, "analyze_siiv_targets"):
                capture = StringIO()
                try:
                    with redirect_stdout(capture):
                        runtime_result = analysis_module.analyze_siiv_targets(str(staged_input), target_lines=target_lines)
                except Exception as exc:
                    logs.extend(self._captured_lines(capture))
                    logs.append(
                        f"{staged_input.name}: primary ANH analysis raised {exc.__class__.__name__}; "
                        "retrying with safe per-line scan."
                    )

            if runtime_result is not None:
                logs.extend(self._captured_lines(capture))
                wavelengths = runtime_result["wavelengths"]
                flux = runtime_result["flux"]
                line_results = list(runtime_result.get("line_results") or [])
                strongest = runtime_result.get("strongest_candidate")
                line_detected = bool(runtime_result.get("line_detected"))
            else:
                capture = StringIO()
                with redirect_stdout(capture):
                    wavelengths, flux = analysis_module.load_spectrum(str(staged_input))
                logs.extend(self._captured_lines(capture))
                insufficient_reason = self._insufficient_spectrum_reason(wavelengths, flux, target_lines)
                if insufficient_reason is not None:
                    logs.append(f"{staged_input.name}: {insufficient_reason}")
                    return self._insufficient_data_result(
                        staged_input=staged_input,
                        reason=insufficient_reason,
                        logs=logs,
                    )
                line_results = self._scan_target_lines_safely(
                    wavelengths=wavelengths,
                    flux=flux,
                    target_lines=target_lines,
                    analysis_module=analysis_module,
                    logs=logs,
                )
                if line_results and all(item.get("failure_reason") for item in line_results):
                    reason = "Insufficient usable Si IV-window coverage for every configured target line."
                    logs.append(f"{staged_input.name}: {reason}")
                    return self._insufficient_data_result(
                        staged_input=staged_input,
                        reason=reason,
                        logs=logs,
                        line_results=line_results,
                    )
                strongest = self._strongest_candidate(line_results)
                line_detected = strongest is not None

            overview_path = file_output_dir / f"{staged_input.stem}_overview.png"
            if hasattr(analysis_module, "create_overview_figure"):
                figure = analysis_module.create_overview_figure(
                    wavelengths,
                    flux,
                    smooth_window=smooth_window,
                    title=f"{staged_input.name} Spectrum",
                )
            else:
                figure = plt.figure(figsize=(11, 4))
                plt.plot(wavelengths, flux, lw=0.4, label="raw")
                plt.plot(wavelengths, analysis_module.smooth(flux, smooth_window), lw=1.0, label="smoothed")
                plt.xlabel("Wavelength (A)")
                plt.ylabel("Flux")
                plt.title(f"{staged_input.name} Spectrum")
                plt.legend()
                plt.tight_layout()
            figure.savefig(overview_path)
            plt.close(figure)

            capture = StringIO()
            with redirect_stdout(capture):
                if hasattr(analysis_module, "plot_si_iv_window"):
                    try:
                        figure = analysis_module.plot_si_iv_window(wavelengths, flux, show=False)
                    except TypeError:
                        figure = analysis_module.plot_si_iv_window(wavelengths, flux)
                else:
                    figure = None
            logs.extend(self._captured_lines(capture))
            if figure is None:
                figure = plt.gcf()
            window_path = file_output_dir / f"{staged_input.stem}_si_iv_window.png"
            figure.savefig(window_path)
            plt.close("all")
            artifact_generation_status = {
                "overview_plot_expected": True,
                "overview_plot_created": overview_path.exists(),
                "window_plot_expected": True,
                "window_plot_created": window_path.exists(),
            }
            if not all(artifact_generation_status.values()):
                logs.append(
                    f"{staged_input.name}: artifact generation incomplete "
                    f"(overview={overview_path.exists()}, window={window_path.exists()})."
                )

            return {
                "status": "ok",
                "logs": logs,
                "summary": {
                    "filename": staged_input.name,
                    "staged_path": str(staged_input),
                    "source_kind": "raw_spectral_input",
                    "recognized_format": "mast_x1d_fits",
                    "line_detected": line_detected,
                    "strongest_candidate": strongest,
                    "line_results": line_results,
                    "artifacts": [str(overview_path), str(window_path)],
                    "artifact_generation_status": artifact_generation_status,
                    "failure_reason": None,
                },
            }
        except Exception as exc:
            plt.close("all")
            logs.append(f"{staged_input.name}: {exc.__class__.__name__}: {exc}")
            return {
                "status": "error",
                "logs": logs,
                "summary": {
                    "filename": staged_input.name,
                    "staged_path": str(staged_input),
                    "source_kind": "raw_spectral_input",
                    "recognized_format": "mast_x1d_fits",
                    "line_detected": False,
                    "strongest_candidate": None,
                    "line_results": [],
                    "artifacts": [],
                    "artifact_generation_status": {
                        "overview_plot_expected": True,
                        "overview_plot_created": False,
                        "window_plot_expected": True,
                        "window_plot_created": False,
                    },
                    "failure_reason": f"{exc.__class__.__name__}: {exc}",
                },
            }

    def _scan_target_lines_safely(
        self,
        *,
        wavelengths: object,
        flux: object,
        target_lines: list[dict[str, object]],
        analysis_module,
        logs: list[str],
    ) -> list[dict[str, object]]:
        line_results: list[dict[str, object]] = []
        for line in target_lines:
            capture = StringIO()
            normalized = None
            failure_reason = None
            try:
                with redirect_stdout(capture):
                    result = analysis_module.zoom(
                        wavelengths,
                        flux,
                        float(line["rest_wavelength"]),
                        width=float(line["width"]),
                        title=str(line["title"]),
                        smooth_win=int(line["smooth_window"]),
                    )
                normalized = self._normalize_zoom_result(result)
            except Exception as exc:
                failure_reason = f"{exc.__class__.__name__}: {exc}"
            logs.extend(self._captured_lines(capture))
            if failure_reason:
                logs.append(f"{line.get('title')}: {failure_reason}")
            line_results.append(
                {
                    "title": str(line["title"]),
                    "rest_wavelength": float(line["rest_wavelength"]),
                    "width": float(line["width"]),
                    "smooth_window": int(line["smooth_window"]),
                    "line_detected": normalized is not None,
                    "result": normalized,
                    "failure_reason": failure_reason,
                }
            )
        return line_results

    @staticmethod
    def _insufficient_spectrum_reason(
        wavelengths: object,
        flux: object,
        target_lines: list[dict[str, object]],
    ) -> str | None:
        try:
            pairs = [
                (float(wave), float(value))
                for wave, value in zip(wavelengths, flux)
                if math.isfinite(float(wave)) and math.isfinite(float(value))
            ]
        except Exception as exc:
            return f"Spectrum could not be converted into finite wavelength/flux samples ({exc.__class__.__name__}: {exc})."
        if len(pairs) < 4:
            return f"Spectrum has too few usable samples for dip scanning ({len(pairs)} finite sample(s))."
        has_target_coverage = any(
            abs(wave - float(line["rest_wavelength"])) <= float(line["width"])
            for wave, _value in pairs
            for line in target_lines
        )
        if not has_target_coverage:
            return "Spectrum has no usable samples inside the configured Si IV scan windows."
        return None

    @staticmethod
    def _insufficient_data_result(
        *,
        staged_input: Path,
        reason: str,
        logs: list[str],
        line_results: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        return {
            "status": "skipped",
            "logs": logs,
            "summary": {
                "filename": staged_input.name,
                "staged_path": str(staged_input),
                "source_kind": "raw_spectral_input",
                "recognized_format": "mast_x1d_fits",
                "line_detected": False,
                "strongest_candidate": None,
                "line_results": list(line_results or []),
                "artifacts": [],
                "artifact_generation_status": {
                    "overview_plot_expected": False,
                    "overview_plot_created": False,
                    "window_plot_expected": False,
                    "window_plot_created": False,
                },
                "failure_reason": reason,
            },
        }

    @staticmethod
    def _normalize_zoom_result(result: object) -> dict[str, float] | None:
        if not isinstance(result, tuple) or len(result) != 3:
            return None
        lam_min, depth, velocity_kms = result
        return {
            "lambda_min": round(float(lam_min), 6),
            "depth": round(float(depth), 6),
            "velocity_kms": round(float(velocity_kms), 3),
        }

    @staticmethod
    def _strongest_candidate(line_results: list[dict[str, object]]) -> dict[str, object] | None:
        ranked = [
            {
                "title": item["title"],
                "rest_wavelength": item["rest_wavelength"],
                **dict(item["result"] or {}),
            }
            for item in line_results
            if item.get("result") is not None
        ]
        if not ranked:
            return None
        ranked.sort(key=lambda item: float(item.get("depth") or 0.0), reverse=True)
        return ranked[0]

    @staticmethod
    def _rank_candidates(*, file_results: list[dict[str, object]], rank_limit: int) -> list[dict[str, object]]:
        ranked = [
            {
                "filename": item["filename"],
                "source_kind": item.get("source_kind"),
                **dict(item["strongest_candidate"] or {}),
            }
            for item in file_results
            if item.get("strongest_candidate") is not None
        ]
        ranked.sort(
            key=lambda item: abs(float(item.get("velocity_kms") or 0.0))
            if item.get("velocity_kms") is not None
            else float(item.get("depth") or 0.0),
            reverse=True,
        )
        return ranked[: max(1, rank_limit)]

    @staticmethod
    def _file_summary(item: dict[str, object]) -> dict[str, object]:
        return {
            "filename": item["filename"],
            "staged_path": item["staged_path"],
            "source_kind": item.get("source_kind"),
            "recognized_format": item.get("recognized_format"),
            "line_detected": bool(item.get("line_detected")),
            "strongest_candidate": item.get("strongest_candidate"),
            "line_results": list(item.get("line_results") or []),
            "parsed_rows": list(item.get("parsed_rows") or []),
            "summary_statistics": dict(item.get("summary_statistics") or {}),
            "artifacts": list(item.get("artifacts") or []),
            "artifact_generation_status": dict(item.get("artifact_generation_status") or {}),
            "failure_reason": item.get("failure_reason"),
        }

    @staticmethod
    def _summary_text(
        *,
        status: str,
        analyzed_count: int,
        candidate_count: int,
        candidate_rankings: list[dict[str, object]],
        execution_mode: str,
        result_quality: str,
    ) -> str:
        if status == "error":
            if result_quality == "missing_runtime_dependency":
                return "ANH could not run raw spectral analysis because required scientific runtime dependencies are missing."
            if result_quality == "archive_staged_no_valid_spectra":
                return "ANH staged the archive input but found no valid raw spectra to analyze."
            return "ANH analysis failed for all accepted inputs."
        if execution_mode == "summarize_results" and analyzed_count > 0:
            return f"ANH summarized processed results from {analyzed_count} file(s)."
        if candidate_count <= 0:
            return f"ANH analyzed {analyzed_count} file(s) and found no candidate Si IV dips."
        strongest = candidate_rankings[0]
        return (
            f"ANH analyzed {analyzed_count} file(s) and found {candidate_count} candidate file(s). "
            f"Strongest candidate: {strongest.get('filename')} at "
            f"{strongest.get('velocity_kms')} km/s with depth {strongest.get('depth')}."
        )

    @staticmethod
    def _result_quality(
        *,
        status: str,
        candidate_count: int,
        analyzed_count: int,
        execution_mode: str,
        dependency_failure_reason: str | None,
        staged_raw_count: int,
        staged_processed_count: int,
    ) -> str:
        if dependency_failure_reason and staged_raw_count > 0 and analyzed_count == staged_raw_count and staged_processed_count == 0:
            return "missing_runtime_dependency"
        if execution_mode == "archive_scan" and analyzed_count <= 0:
            return "archive_staged_no_valid_spectra"
        if analyzed_count <= 0 and staged_raw_count > 0 and staged_processed_count <= 0:
            return "archive_staged_no_valid_spectra"
        if analyzed_count <= 0:
            return "invalid_input"
        if status == "error":
            return "analysis_failed"
        if (
            execution_mode == "summarize_results"
            or (staged_processed_count > 0 and staged_raw_count == 0 and status in {"ok", "partial"})
        ) and status in {"ok", "partial"}:
            return "processed_results_summarized"
        if candidate_count > 0:
            return "candidate_dips_detected"
        if status == "partial":
            return "partial_artifacts"
        return "no_candidate_dips"

    @staticmethod
    def _runtime_dependency_failure(exc: Exception) -> str:
        if isinstance(exc, ModuleNotFoundError):
            missing_name = getattr(exc, "name", None) or str(exc).strip() or "required scientific dependency"
            return (
                f"Missing runtime dependency for ANH raw spectral analysis: {missing_name}. "
                "Install the ANH scientific stack (astropy, numpy, scipy, matplotlib)."
            )
        return f"ANH raw spectral runtime setup failed: {exc.__class__.__name__}: {exc}"

    @staticmethod
    def _failure_reason(
        *,
        status: str,
        dependency_failure_reason: str | None,
        result_quality: str,
        staged_raw_count: int,
        staged_processed_count: int,
    ) -> str | None:
        if status == "ok":
            return None
        if dependency_failure_reason and staged_raw_count > 0 and staged_processed_count == 0:
            return dependency_failure_reason
        if result_quality == "archive_staged_no_valid_spectra":
            return "Archive extraction and staging completed, but no valid ANH raw spectra were available for analysis."
        if status == "partial":
            return "Some files were skipped, failed inspection, or only partially analyzed."
        return "ANH analysis failed for all accepted inputs."

    def _collect_output_artifacts(self, output_dir: Path) -> tuple[list[Artifact], list[ResearchArtifactRecord]]:
        artifacts: list[Artifact] = []
        records: list[ResearchArtifactRecord] = []
        for path in sorted(output_dir.rglob("*")):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix == ".png":
                media_type = "image/png"
            elif suffix == ".json":
                media_type = "application/json"
            elif suffix == ".txt":
                media_type = "text/plain"
            else:
                media_type = "application/octet-stream"
            description = "ANH generated artifact"
            artifacts.append(Artifact(name=path.name, path=path, media_type=media_type, description=description))
            records.append(
                ResearchArtifactRecord(
                    role=path.stem,
                    path=path,
                    media_type=media_type,
                    description=description,
                )
            )
        return artifacts, records

    @staticmethod
    def _captured_lines(capture: StringIO) -> list[str]:
        return [line for line in capture.getvalue().splitlines() if line.strip()]

    @staticmethod
    def _configure_headless_matplotlib() -> None:
        import matplotlib

        matplotlib.use("Agg", force=True)

    @staticmethod
    def _runtime_environment_diagnostics() -> dict[str, object]:
        import importlib.util
        from importlib.metadata import PackageNotFoundError, version

        diagnostics: dict[str, object] = {}
        for package_name in ("astropy", "numpy", "scipy", "matplotlib"):
            available = importlib.util.find_spec(package_name) is not None
            package_version = None
            if available:
                try:
                    package_version = version(package_name)
                except PackageNotFoundError:
                    package_version = None
            diagnostics[package_name] = {
                "available": available,
                "version": package_version,
            }
        return diagnostics

    def _load_analysis_module(self):
        script_path = self.repo_root / "tools" / "anh" / "anh_andromeda_v6_3.py"
        spec = importlib.util.spec_from_file_location("anh_andromeda_v6_3", script_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load ANH script at {script_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _provenance(
        self,
        *,
        request: ToolRequest,
        run_id: str,
        target_lines: list[dict[str, object]],
        intake: ANHInputRecognitionResult | None,
    ) -> dict[str, object]:
        baseline_script = self.repo_root / "tools" / "anh" / "anh_andromeda_v6_3.py"
        return {
            "run_id": run_id,
            "wrapper_version": self.WRAPPER_VERSION,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "bundle_id": self.manifest.id,
            "bundle_version": self.manifest.version,
            "baseline_script_path": str(baseline_script),
            "baseline_script_sha256": sha256_or_none(baseline_script),
            "input_dataset": str(request.input_path) if request.input_path else None,
            "intake_execution_mode": intake.execution_mode if intake is not None else "reject",
            "selected_params": {
                "target_lines": target_lines,
                "smooth_window": request.params.get("smooth_window", 11),
                "scan_width": request.params.get("scan_width"),
                "max_files": self._effective_max_files(
                    request=request,
                    archive=bool(intake.archive_inputs) if intake is not None else False,
                ),
                "rank_limit": request.params.get("rank_limit", 10),
            },
            "runtime_diagnostics": self._runtime_environment_diagnostics(),
        }

    @staticmethod
    def _target_lines(params: dict[str, object]) -> list[dict[str, object]]:
        custom = params.get("target_lines")
        if isinstance(custom, list) and custom:
            normalized: list[dict[str, object]] = []
            for item in custom:
                if not isinstance(item, dict):
                    continue
                if "rest_wavelength" not in item:
                    continue
                normalized.append(
                    {
                        "rest_wavelength": float(item["rest_wavelength"]),
                        "title": str(item.get("title") or f"Line @ {item['rest_wavelength']}"),
                        "width": float(item.get("width", params.get("scan_width", 1.0))),
                        "smooth_window": int(item.get("smooth_window", params.get("smooth_window", 5))),
                    }
                )
            if normalized:
                return normalized
        scan_width = float(params.get("scan_width", 1.0))
        smooth_window = int(params.get("smooth_window", 5))
        return [
            {
                **item,
                "width": float(item.get("width", scan_width)),
                "smooth_window": int(item.get("smooth_window", smooth_window)),
            }
            for item in DEFAULT_TARGET_LINES
        ]

    def _create_run_dir(self, request: ToolRequest) -> Path:
        root = request.run_root or self.repo_root / "data" / "tool_runs"
        run_id = datetime.now().strftime("run_%Y_%m_%d_%H%M%S")
        run_dir = root / request.session_id / request.tool_id / request.capability / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
