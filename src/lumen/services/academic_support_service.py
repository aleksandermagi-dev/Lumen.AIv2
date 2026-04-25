from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import re

from lumen.reasoning.response_models import ResearchResponse


@dataclass(frozen=True, slots=True)
class AcademicWorkflowDecision:
    workflow: str
    domain_id: str
    status: str
    kind: str
    support_level: str | None = None


class AcademicSupportService:
    """Bounded local academic-support helpers layered over existing research flows."""

    _GHOSTWRITING_PATTERNS: tuple[str, ...] = (
        "write my paper for me",
        "write my essay for me",
        "write my research paper",
        "write my final paper",
        "write my assignment for me",
        "do my homework",
        "complete my assignment",
        "write this for submission",
        "publish this paper for me",
    )
    _CITATION_PATTERNS: tuple[str, ...] = (
        "format this citation",
        "fix this citation",
        "clean up this citation",
        "works cited",
        "references page",
        "apa",
        "mla",
        "chicago",
        "bibliography",
    )
    _BRAINSTORM_PATTERNS: tuple[str, ...] = (
        "brainstorm",
        "topic ideas",
        "thesis ideas",
        "argument ideas",
    )
    _OUTLINE_PATTERNS: tuple[str, ...] = (
        "outline",
        "create an outline",
        "make an outline",
        "literature review structure",
        "lab report structure",
    )
    _REVISION_PATTERNS: tuple[str, ...] = (
        "review this draft",
        "give feedback on this draft",
        "revise this draft",
        "revision feedback",
        "improve coherence",
        "improve clarity",
    )
    _RHETORICAL_PATTERNS: tuple[str, ...] = (
        "rhetorical analysis",
        "analyze the rhetoric",
        "analyze tone and audience",
        "ethos",
        "pathos",
        "logos",
    )
    _LITERATURE_PATTERNS: tuple[str, ...] = (
        "synthesize these papers",
        "synthesize these studies",
        "literature gap",
        "gap in the literature",
        "annotated bibliography",
        "compare these papers",
        "compare these studies",
        "common themes across these papers",
    )
    _DATASET_PATTERNS: tuple[str, ...] = (
        "dataset readiness",
        "label column",
        "target column",
        "train validation test",
        "train/validation/test",
        "data leakage",
        "class balance",
        "feature target",
        "preprocess this dataset",
        "supervised learning dataset",
        "evaluation plan",
    )
    _MATH_KEYWORDS: tuple[tuple[str, str], ...] = (
        ("eigen", "advanced_bounded"),
        ("svd", "advanced_bounded"),
        ("singular value decomposition", "advanced_bounded"),
        ("jacobian", "advanced_bounded"),
        ("hessian", "advanced_bounded"),
        ("lagrangian", "advanced_bounded"),
        ("gradient descent", "core_college"),
        ("bayes", "core_college"),
        ("covariance", "core_college"),
        ("partial derivative", "core_college"),
        ("chain rule", "core_college"),
        ("probability distribution", "core_college"),
        ("combinatorics", "core_college"),
        ("graph theory", "core_college"),
        ("matrix", "core_college"),
        ("calculus", "core_college"),
    )
    _CONCEPTUAL_MATH_PATTERNS: tuple[str, ...] = (
        "what is the derivative",
        "what is a derivative",
        "explain derivative",
        "explain derivatives",
        "what is an integral",
        "what is the integral",
        "explain integral",
        "explain integrals",
        "what is calculus",
        "explain calculus",
    )
    _BRIDGE_HINTS: tuple[str, ...] = (
        "from the basics",
        "start from basics",
        "high school",
        "beginner",
        "foundation",
        "prerequisite",
        "step down",
    )

    @classmethod
    def classify(cls, *, prompt: str, input_path: Path | None = None) -> AcademicWorkflowDecision | None:
        normalized = " ".join(str(prompt or "").strip().lower().split())
        if not normalized:
            return None
        if any(pattern in normalized for pattern in cls._GHOSTWRITING_PATTERNS):
            return AcademicWorkflowDecision(
                workflow="academic_integrity_boundary",
                domain_id="academic_writing",
                status="bounded",
                kind="research.academic_integrity_boundary",
            )
        if any(pattern in normalized for pattern in cls._CITATION_PATTERNS):
            return AcademicWorkflowDecision(
                workflow="citation_help",
                domain_id="citation_support",
                status="bounded",
                kind="research.academic_citation",
            )
        if any(pattern in normalized for pattern in cls._LITERATURE_PATTERNS):
            return AcademicWorkflowDecision(
                workflow="literature_synthesis",
                domain_id="literature_synthesis",
                status="bounded",
                kind="research.literature_synthesis",
            )
        if input_path is not None and input_path.suffix.lower() in {".csv", ".tsv", ".json"}:
            if any(token in normalized for token in ("dataset", "label", "target", "split", "class", "leakage", "feature")):
                return AcademicWorkflowDecision(
                    workflow="dataset_readiness",
                    domain_id="supervised_ml_data_support",
                    status="bounded",
                    kind="research.dataset_readiness",
                )
        if any(pattern in normalized for pattern in cls._DATASET_PATTERNS):
            return AcademicWorkflowDecision(
                workflow="dataset_readiness",
                domain_id="supervised_ml_data_support",
                status="bounded",
                kind="research.dataset_readiness",
            )
        if any(pattern in normalized for pattern in cls._CONCEPTUAL_MATH_PATTERNS):
            return AcademicWorkflowDecision(
                workflow="math_science_support",
                domain_id="college_math_science_support",
                status="bounded",
                kind="research.academic_math_support",
                support_level="core_college",
            )
        if any(pattern in normalized for pattern in cls._OUTLINE_PATTERNS):
            return AcademicWorkflowDecision(
                workflow="outline",
                domain_id="academic_writing",
                status="bounded",
                kind="research.academic_support",
            )
        if any(pattern in normalized for pattern in cls._BRAINSTORM_PATTERNS):
            return AcademicWorkflowDecision(
                workflow="brainstorm",
                domain_id="academic_writing",
                status="bounded",
                kind="research.academic_support",
            )
        if any(pattern in normalized for pattern in cls._REVISION_PATTERNS):
            return AcademicWorkflowDecision(
                workflow="revision_feedback",
                domain_id="academic_writing",
                status="bounded",
                kind="research.academic_support",
            )
        if any(pattern in normalized for pattern in cls._RHETORICAL_PATTERNS):
            return AcademicWorkflowDecision(
                workflow="rhetorical_analysis",
                domain_id="academic_writing",
                status="bounded",
                kind="research.academic_support",
            )
        for keyword, support_level in cls._MATH_KEYWORDS:
            if keyword in normalized:
                if any(hint in normalized for hint in cls._BRIDGE_HINTS):
                    support_level = "bridge"
                return AcademicWorkflowDecision(
                    workflow="math_science_support",
                    domain_id="college_math_science_support",
                    status="bounded",
                    kind="research.academic_math_support",
                    support_level=support_level,
                )
        return None

    @classmethod
    def build_response(
        cls,
        *,
        prompt: str,
        decision: AcademicWorkflowDecision,
        input_path: Path | None = None,
    ) -> dict[str, object]:
        if decision.workflow == "academic_integrity_boundary":
            return cls._build_integrity_boundary(prompt=prompt)
        if decision.workflow == "citation_help":
            return cls._build_citation_help(prompt=prompt)
        if decision.workflow == "literature_synthesis":
            return cls._build_literature_synthesis(prompt=prompt)
        if decision.workflow == "dataset_readiness":
            return cls._build_dataset_readiness(prompt=prompt, input_path=input_path)
        if decision.workflow == "outline":
            return cls._build_outline(prompt=prompt)
        if decision.workflow == "brainstorm":
            return cls._build_brainstorm(prompt=prompt)
        if decision.workflow == "revision_feedback":
            return cls._build_revision_feedback(prompt=prompt)
        if decision.workflow == "rhetorical_analysis":
            return cls._build_rhetorical_analysis(prompt=prompt)
        if decision.workflow == "math_science_support":
            return cls._build_math_support(prompt=prompt, support_level=decision.support_level or "core_college")
        raise ValueError(f"Unsupported academic workflow: {decision.workflow}")

    @classmethod
    def build_report(cls) -> dict[str, object]:
        workflows = [
            {
                "workflow": "brainstorm",
                "domain_id": "academic_writing",
                "status": "bounded",
                "scope_note": "Supports topic/thesis ideation and structure help without replacing student authorship.",
            },
            {
                "workflow": "outline",
                "domain_id": "academic_writing",
                "status": "bounded",
                "scope_note": "Supports essay/report/literature-review outlining and revision planning.",
            },
            {
                "workflow": "citation_help",
                "domain_id": "citation_support",
                "status": "bounded",
                "scope_note": "Formats supplied citation metadata and flags incomplete or unverified fields.",
            },
            {
                "workflow": "literature_synthesis",
                "domain_id": "literature_synthesis",
                "status": "bounded",
                "scope_note": "Summarizes and compares supplied papers or study snippets without claiming external verification.",
            },
            {
                "workflow": "math_science_support",
                "domain_id": "college_math_science_support",
                "status": "bounded",
                "scope_note": "Provides bounded college-core explanations with bridge support for prerequisites.",
            },
            {
                "workflow": "dataset_readiness",
                "domain_id": "supervised_ml_data_support",
                "status": "bounded",
                "scope_note": "Supports local/user-supplied supervised-ML dataset readiness review and evaluation planning.",
            },
        ]
        return {
            "schema_type": "academic_support_report",
            "schema_version": "1",
            "workflows": workflows,
            "status_counts": {
                "bounded": len(workflows),
            },
        }

    @classmethod
    def _build_integrity_boundary(cls, *, prompt: str) -> dict[str, object]:
        summary = "I can help you build the work, but not ghostwrite it for submission."
        findings = [
            "I can help brainstorm, outline, revise, summarize, or critique the work instead.",
            "Use AI support to strengthen your thinking and drafting process, not replace authorship or attribution.",
            "You should verify claims, citations, and final wording before submitting anything.",
        ]
        response = ResearchResponse(
            mode="research",
            kind="research.academic_integrity_boundary",
            summary=summary,
            findings=findings,
            recommendation="If you want, I can help turn the assignment into an outline, thesis options, or a revision checklist.",
        ).to_dict()
        response["user_facing_answer"] = "\n".join([summary, "", *[f"- {item}" for item in findings], "", str(response["recommendation"])])
        response["capability_status"] = {
            "domain_id": "academic_writing",
            "status": "bounded",
            "details": "Academic writing support is bounded to brainstorming, structure, revision, and study support.",
        }
        response["academic_integrity_guidance"] = {
            "policy": "study_support_only",
            "ghostwriting_redirected": True,
            "limitations": ["verify claims", "verify citations", "retain student authorship"],
        }
        return response

    @classmethod
    def _build_outline(cls, *, prompt: str) -> dict[str, object]:
        topic = cls._extract_topic(prompt) or "the topic"
        genre = cls._infer_academic_genre(prompt)
        summary = f"Here’s a bounded {genre} outline for {topic}."
        outline = [
            "Introduction: frame the question, context, and thesis direction.",
            "Core section 1: define the main concept or argument with evidence.",
            "Core section 2: develop comparison, mechanism, or interpretation.",
            "Core section 3: address a limitation, counterpoint, or implication.",
            "Conclusion: restate the claim and show why it matters.",
        ]
        if genre == "literature review":
            outline = [
                "Introduction: define the research question and scope.",
                "Theme cluster 1: summarize the strongest agreement across sources.",
                "Theme cluster 2: compare methodological differences and tradeoffs.",
                "Theme cluster 3: surface contradictions, limitations, or blind spots.",
                "Gap section: identify what remains underexplored.",
                "Conclusion: explain the synthesis direction and next research need.",
            ]
        response = ResearchResponse(
            mode="research",
            kind="research.academic_support",
            summary=summary,
            findings=outline,
            recommendation="If you want, I can tighten this into a thesis-driven outline or a section-by-section writing checklist.",
        ).to_dict()
        response["user_facing_answer"] = "\n".join([summary, "", *[f"- {item}" for item in outline], "", str(response["recommendation"])])
        response["capability_status"] = {
            "domain_id": "academic_writing",
            "status": "bounded",
            "details": "Academic outlining support stays structure-first and integrity-safe.",
        }
        response["academic_workflow"] = {"workflow": "outline", "genre": genre, "topic": topic}
        return response

    @classmethod
    def _build_brainstorm(cls, *, prompt: str) -> dict[str, object]:
        topic = cls._extract_topic(prompt) or "the topic"
        ideas = [
            f"Take a tension angle: what conflict or tradeoff inside {topic} is easiest to defend with evidence?",
            f"Take a mechanism angle: what actually causes or explains the most important effect in {topic}?",
            f"Take a comparison angle: which two approaches, periods, or theories inside {topic} reveal the strongest contrast?",
            f"Take a gap angle: what part of {topic} is usually oversimplified or left underexplained?",
        ]
        summary = f"Here are bounded thesis or topic directions for {topic}."
        response = ResearchResponse(
            mode="research",
            kind="research.academic_support",
            summary=summary,
            findings=ideas,
            recommendation="If one direction clicks, I can turn it into a thesis, outline, or evidence plan next.",
        ).to_dict()
        response["user_facing_answer"] = "\n".join([summary, "", *[f"- {item}" for item in ideas], "", str(response["recommendation"])])
        response["capability_status"] = {
            "domain_id": "academic_writing",
            "status": "bounded",
            "details": "Brainstorming support is intended to spark directions, not replace your final argument choices.",
        }
        response["academic_workflow"] = {"workflow": "brainstorm", "topic": topic}
        return response

    @classmethod
    def _build_revision_feedback(cls, *, prompt: str) -> dict[str, object]:
        draft = cls._extract_payload(prompt)
        strengths: list[str] = []
        logic_issues: list[str] = []
        tone_issues: list[str] = []
        revision_priorities: list[str] = []
        if len(draft.split()) >= 40:
            strengths.append("There is enough material here to shape into a clearer academic argument.")
        else:
            logic_issues.append("The draft is still thin, so the main claim may not feel fully established yet.")
        if any(token in draft.lower() for token in ("because", "therefore", "however", "although")):
            strengths.append("The draft already has some argumentative movement instead of only description.")
        else:
            logic_issues.append("The reasoning chain needs clearer connective logic between claims and evidence.")
        if any(token in draft.lower() for token in ("i think", "kind of", "sort of", "really")):
            tone_issues.append("The tone reads somewhat informal for an academic submission.")
        else:
            strengths.append("The tone is close to an academic register.")
        if not strengths:
            strengths.append("The draft has a usable core idea to build from.")
        if not logic_issues:
            logic_issues.append("The next improvement is making the thesis and paragraph purpose more explicit.")
        if not tone_issues:
            tone_issues.append("Tone is mostly fine, but sentence precision could still improve.")
        revision_priorities.extend(
            [
                "Make the central claim explicit in the opening.",
                "Tie each paragraph back to that claim with clearer transitions.",
                "Verify any factual statements or citations before using the final version.",
            ]
        )
        summary = "Here’s bounded revision feedback on the draft."
        response = ResearchResponse(
            mode="research",
            kind="research.academic_support",
            summary=summary,
            findings=[
                f"Strengths: {'; '.join(strengths)}",
                f"Logic/coherence issues: {'; '.join(logic_issues)}",
                f"Tone/style issues: {'; '.join(tone_issues)}",
                f"Revision priorities: {'; '.join(revision_priorities)}",
            ],
            recommendation="If you want, I can turn this into a paragraph-by-paragraph revision checklist next.",
        ).to_dict()
        response["user_facing_answer"] = "\n".join(
            [
                summary,
                "",
                f"- Strengths: {'; '.join(strengths)}",
                f"- Logic/coherence issues: {'; '.join(logic_issues)}",
                f"- Tone/style issues: {'; '.join(tone_issues)}",
                f"- Revision priorities: {'; '.join(revision_priorities)}",
                "",
                str(response["recommendation"]),
            ]
        )
        response["capability_status"] = {
            "domain_id": "academic_writing",
            "status": "bounded",
            "details": "Revision feedback is bounded to structure, clarity, and academic tone guidance.",
        }
        response["academic_workflow"] = {"workflow": "revision_feedback"}
        return response

    @classmethod
    def _build_rhetorical_analysis(cls, *, prompt: str) -> dict[str, object]:
        payload = cls._extract_payload(prompt)
        text = payload.lower()
        findings = [
            "Purpose: identify what claim or effect the passage seems to be driving toward.",
            "Audience: note who the argument appears to be written for and what assumptions it makes about them.",
            "Tone: check whether the voice is analytical, persuasive, urgent, ironic, or neutral.",
        ]
        if "because" in text or "therefore" in text:
            findings.append("Logic: the passage already signals causal reasoning that can be analyzed more directly.")
        if any(token in text for token in ("emotion", "fear", "hope", "justice")):
            findings.append("Appeal: emotional language suggests a stronger pathos component.")
        if any(token in text for token in ("study", "data", "evidence", "research")):
            findings.append("Appeal: evidence-oriented wording suggests ethos/logos support.")
        summary = "Here’s a bounded rhetorical-analysis frame."
        response = ResearchResponse(
            mode="research",
            kind="research.academic_support",
            summary=summary,
            findings=findings,
            recommendation="If you paste the exact passage, I can map tone, audience, and rhetorical appeals more specifically.",
        ).to_dict()
        response["user_facing_answer"] = "\n".join([summary, "", *[f"- {item}" for item in findings], "", str(response["recommendation"])])
        response["capability_status"] = {
            "domain_id": "academic_writing",
            "status": "bounded",
            "details": "Rhetorical support stays analysis-focused and does not replace close reading.",
        }
        response["academic_workflow"] = {"workflow": "rhetorical_analysis"}
        return response

    @classmethod
    def _build_citation_help(cls, *, prompt: str) -> dict[str, object]:
        style = cls._infer_citation_style(prompt)
        metadata = cls._extract_citation_metadata(prompt)
        missing = [field for field in ("author", "title", "year") if not metadata.get(field)]
        citation = cls._format_citation(style=style, metadata=metadata)
        if missing:
            status = "incomplete_source_metadata"
            summary = f"I can help format this {style.upper()} citation, but the supplied metadata is incomplete."
            findings = [
                f"Missing fields: {', '.join(missing)}.",
                "I won’t invent authors, dates, page ranges, journal details, or identifiers.",
            ]
        else:
            status = "formatted_from_supplied_metadata"
            summary = f"Here is a bounded {style.upper()} citation formatted from the supplied metadata."
            findings = ["This is formatted only from the metadata you supplied.", "You should still verify source details before submission."]
        response = ResearchResponse(
            mode="research",
            kind="research.academic_citation",
            summary=summary,
            findings=[citation, *findings] if citation else findings,
            recommendation="If you want, I can also point out which source fields still need verification.",
        ).to_dict()
        answer_lines = [summary, ""]
        if citation:
            answer_lines.append(citation)
            answer_lines.append("")
        answer_lines.extend(f"- {item}" for item in findings)
        answer_lines.extend(["", str(response["recommendation"])])
        response["user_facing_answer"] = "\n".join(answer_lines).strip()
        response["capability_status"] = {
            "domain_id": "citation_support",
            "status": "bounded",
            "details": "Citation help formats supplied metadata and flags uncertainty instead of verifying unseen sources.",
        }
        response["citation_integrity_status"] = status
        response["academic_integrity_guidance"] = {
            "citation_verification_required": True,
            "invented_fields_blocked": True,
        }
        return response

    @classmethod
    def _build_literature_synthesis(cls, *, prompt: str) -> dict[str, object]:
        papers = cls._extract_papers(prompt)
        if len(papers) < 2:
            summary = "I can synthesize supplied papers or study snippets, but I need at least two sources to compare."
            findings = [
                "Paste two or more paper summaries, abstracts, or method/result snippets.",
                "I can then compare themes, methods, limitations, contradictions, and likely gaps.",
            ]
            response = ResearchResponse(
                mode="research",
                kind="research.literature_synthesis",
                summary=summary,
                findings=findings,
                recommendation="If you want, I can also give you a template for a literature synthesis or annotated bibliography.",
            ).to_dict()
            response["user_facing_answer"] = "\n".join([summary, "", *[f"- {item}" for item in findings], "", str(response["recommendation"])])
        else:
            themes = cls._common_theme_fragments(papers)
            methods = cls._section_fragments(papers, "methods")
            results = cls._section_fragments(papers, "results")
            limits = cls._section_fragments(papers, "limitations")
            findings = [
                f"Synthesis themes: {'; '.join(themes or ['The supplied sources appear to orbit a similar central problem or domain.'])}",
                f"Methods comparison: {'; '.join(methods or ['Method differences are not fully explicit across the supplied snippets.'])}",
                f"Results comparison: {'; '.join(results or ['Results language is too thin to compare strongly.'])}",
                f"Limitations or gaps: {'; '.join(limits or ['A clear gap may be the need for stronger methods, broader samples, or clearer replication evidence.'])}",
            ]
            summary = "Here’s a bounded literature synthesis from the supplied sources."
            response = ResearchResponse(
                mode="research",
                kind="research.literature_synthesis",
                summary=summary,
                findings=findings,
                recommendation="If you want, I can turn this into a literature-review paragraph sequence or an annotated-bibliography format next.",
            ).to_dict()
            response["user_facing_answer"] = "\n".join([summary, "", *[f"- {item}" for item in findings], "", str(response["recommendation"])])
        response["capability_status"] = {
            "domain_id": "literature_synthesis",
            "status": "bounded",
            "details": "Literature synthesis is grounded only in supplied or locally available source text.",
        }
        response["academic_workflow"] = {"workflow": "literature_synthesis", "source_count": len(papers)}
        response["academic_integrity_guidance"] = {
            "source_verification_required": True,
            "shallow_synthesis_risk": len(papers) < 3,
        }
        return response

    @classmethod
    def _build_dataset_readiness(cls, *, prompt: str, input_path: Path | None = None) -> dict[str, object]:
        flags = {
            "schema_visible": bool(input_path),
            "label_column_needed": True,
            "split_guidance_needed": True,
            "leakage_check_required": True,
            "class_balance_check_required": True,
            "feature_target_clarity_needed": True,
        }
        findings = [
            "Make the target/label column explicit before training so feature-target leakage is easier to catch.",
            "Use a deterministic train/validation/test split and keep the test set isolated from tuning decisions.",
            "Check class balance, duplicate rows, missing values, and obvious identifier leakage before modeling.",
            "Document provenance and license at the original source before using the data for training or publication.",
        ]
        if input_path is not None and input_path.exists() and input_path.suffix.lower() in {".csv", ".tsv"}:
            header = cls._read_header(input_path)
            if header:
                flags["schema_visible"] = True
                findings.insert(0, f"Visible columns: {', '.join(header[:8])}")
        summary = "Here’s a bounded supervised-ML dataset readiness review."
        response = ResearchResponse(
            mode="research",
            kind="research.dataset_readiness",
            summary=summary,
            findings=findings,
            recommendation="If you want, I can help identify a likely target column, split strategy, or evaluation checklist next.",
        ).to_dict()
        response["user_facing_answer"] = "\n".join([summary, "", *[f"- {item}" for item in findings], "", str(response["recommendation"])])
        response["capability_status"] = {
            "domain_id": "supervised_ml_data_support",
            "status": "bounded",
            "details": "Supervised-data support stays advisory and bounded to local or user-supplied datasets.",
        }
        response["dataset_readiness_flags"] = flags
        response["academic_integrity_guidance"] = {
            "license_and_provenance_verify_at_origin": True,
            "prediction_claims_bounded_to_local_data": True,
        }
        return response

    @classmethod
    def _build_math_support(cls, *, prompt: str, support_level: str) -> dict[str, object]:
        topic = cls._infer_math_topic(prompt)
        explanation = cls._math_explanation(topic=topic, support_level=support_level)
        summary = f"Here’s a bounded {support_level.replace('_', ' ')} explanation for {topic}."
        findings = [
            explanation,
            "Use this as a reasoning aid, then verify the formal derivation or notation against your course material if precision matters.",
        ]
        response = ResearchResponse(
            mode="research",
            kind="research.academic_math_support",
            summary=summary,
            findings=findings,
            recommendation="If you want, I can step this down further into prerequisites or turn it into a worked study checklist.",
        ).to_dict()
        response["user_facing_answer"] = "\n".join([summary, "", *[f"- {item}" for item in findings], "", str(response["recommendation"])])
        response["capability_status"] = {
            "domain_id": "college_math_science_support",
            "status": "bounded",
            "details": "Math/science support is bounded to explanation, setup, and existing tool-backed reasoning surfaces.",
        }
        response["math_support_level"] = support_level
        response["academic_workflow"] = {"workflow": "math_science_support", "topic": topic}
        return response

    @staticmethod
    def _infer_academic_genre(prompt: str) -> str:
        normalized = " ".join(str(prompt or "").lower().split())
        if "literature review" in normalized:
            return "literature review"
        if "lab" in normalized:
            return "lab-style writeup"
        if "persuasive" in normalized:
            return "persuasive essay"
        if "analysis" in normalized or "literary" in normalized:
            return "analytical essay"
        if "report" in normalized:
            return "research report"
        return "academic essay"

    @staticmethod
    def _extract_topic(prompt: str) -> str | None:
        normalized = " ".join(str(prompt or "").strip().split())
        for marker in (" on ", " about ", " for ", " of "):
            if marker in normalized.lower():
                index = normalized.lower().find(marker)
                candidate = normalized[index + len(marker) :].strip(" .:-")
                if candidate:
                    return candidate
        return None

    @staticmethod
    def _extract_payload(prompt: str) -> str:
        text = str(prompt or "").strip()
        if ":" in text:
            return text.split(":", 1)[1].strip()
        return text

    @staticmethod
    def _infer_citation_style(prompt: str) -> str:
        normalized = " ".join(str(prompt or "").lower().split())
        if "mla" in normalized:
            return "mla"
        if "chicago" in normalized:
            return "chicago"
        return "apa"

    @staticmethod
    def _extract_citation_metadata(prompt: str) -> dict[str, str]:
        text = " ".join(str(prompt or "").strip().split())
        patterns = {
            "author": r"author\s+([^;]+)",
            "title": r"title\s+([^;]+)",
            "year": r"year\s+([^;]+)",
            "journal": r"journal\s+([^;]+)",
            "publisher": r"publisher\s+([^;]+)",
            "url": r"(https?://\S+)",
            "doi": r"doi\s+([^;]+)",
        }
        payload: dict[str, str] = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match is not None:
                payload[key] = match.group(1).strip(" .")
        title_match = re.search(r"\"([^\"]+)\"", text)
        if title_match is not None and "title" not in payload:
            payload["title"] = title_match.group(1).strip()
        return payload

    @staticmethod
    def _format_citation(*, style: str, metadata: dict[str, str]) -> str:
        author = metadata.get("author")
        title = metadata.get("title")
        year = metadata.get("year")
        journal = metadata.get("journal") or metadata.get("publisher")
        url = metadata.get("url") or metadata.get("doi")
        if style == "mla":
            parts = [f"{author}." if author else None, f'"{title}."' if title else None, journal, year, url]
            return " ".join(part for part in parts if part).strip()
        if style == "chicago":
            parts = [author, f'"{title}."' if title else None, f"({year})." if year else None, journal, url]
            return " ".join(part for part in parts if part).strip()
        parts = [
            f"{author}." if author else None,
            f"({year})." if year else None,
            f"{title}." if title else None,
            journal,
            url,
        ]
        return " ".join(part for part in parts if part).strip()

    @staticmethod
    def _extract_papers(prompt: str) -> list[str]:
        text = str(prompt or "").strip()
        if re.search(r"paper\s*1\s*:", text, flags=re.IGNORECASE):
            parts = re.split(r"paper\s*\d+\s*:", text, flags=re.IGNORECASE)
            return [part.strip(" ;\n") for part in parts if part.strip(" ;\n")]
        if " ; " in text:
            return [part.strip() for part in text.split(" ; ") if part.strip()]
        return []

    @staticmethod
    def _section_fragments(papers: list[str], label: str) -> list[str]:
        values: list[str] = []
        pattern = re.compile(rf"{label}\s*:\s*([^.;]+)", flags=re.IGNORECASE)
        for paper in papers[:4]:
            match = pattern.search(paper)
            if match is not None:
                values.append(match.group(1).strip())
        return values

    @staticmethod
    def _common_theme_fragments(papers: list[str]) -> list[str]:
        themes: list[str] = []
        for token in ("simulation", "observation", "model", "survey", "experiment", "trend", "prediction"):
            if sum(1 for paper in papers if token in paper.lower()) >= 2:
                themes.append(f"Multiple sources refer to {token}-driven analysis.")
        return themes[:3]

    @staticmethod
    def _read_header(path: Path) -> list[str]:
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle, delimiter=delimiter)
                return [item.strip() for item in next(reader, []) if str(item).strip()]
        except OSError:
            return []

    @staticmethod
    def _infer_math_topic(prompt: str) -> str:
        normalized = " ".join(str(prompt or "").lower().split())
        for token in (
            "bayes theorem",
            "gradient descent",
            "eigenvalues",
            "eigenvectors",
            "svd",
            "jacobian",
            "hessian",
            "chain rule",
            "derivative",
            "integral",
            "covariance",
            "combinatorics",
            "graph theory",
            "matrix operations",
        ):
            if token in normalized:
                return token
        if "matrix" in normalized:
            return "matrix operations"
        if "calculus" in normalized:
            return "calculus"
        if "derivative" in normalized:
            return "derivatives"
        if "integral" in normalized:
            return "integrals"
        if "probability" in normalized:
            return "probability"
        return "the math topic"

    @staticmethod
    def _math_explanation(*, topic: str, support_level: str) -> str:
        if support_level == "bridge":
            return (
                f"Start with the prerequisite intuition for {topic}: identify the simplest objects involved, "
                "what changes, and what quantity you are trying to track before moving into formal notation."
            )
        if topic in {"bayes theorem", "probability"}:
            return "Bayes-style reasoning updates a prior belief using new evidence, so the posterior reflects both what you believed before and how strongly the evidence favors each outcome."
        if topic == "gradient descent":
            return "Gradient descent follows the local slope of a loss surface in the direction that most reduces error, with step size controlling stability versus speed."
        if topic == "derivatives":
            return "A derivative describes an instantaneous rate of change: for a function, it tells how the output changes as the input changes at a point."
        if topic == "integrals":
            return "An integral accumulates quantity across an interval, which is why it can represent area, total change, mass, probability, or other accumulated effects."
        if topic in {"jacobian", "hessian"}:
            return f"{topic.title()} support here is bounded to what the matrix captures: local rate-of-change structure for vector outputs or second-order curvature around a point."
        if topic in {"svd", "eigenvalues", "eigenvectors"}:
            return f"{topic.upper() if topic == 'svd' else topic.title()} support here focuses on decomposition intuition, geometric meaning, and when the operation is useful in data representation or transformation."
        return "This support stays focused on conceptual explanation, prerequisites, and where the concept fits into college-level quantitative reasoning."
