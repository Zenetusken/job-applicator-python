"""Unit tests for documents layer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from job_applicator.config import LLMConfig
from job_applicator.documents.cover_letter import CoverLetterGenerator
from job_applicator.documents.resume import ResumeLoader
from job_applicator.exceptions import DocumentError, LLMError, ResumeNotFoundError
from job_applicator.models import (
    ExperienceEntry,
    JobBoard,
    JobListing,
    ResumeData,
    StyleGuide,
    UserProfile,
)


def test_resume_loader_missing_file() -> None:
    loader = ResumeLoader()
    with pytest.raises(ResumeNotFoundError):
        loader.load("/nonexistent/resume.pdf")


def test_resume_loader_unsupported_format(tmp_path: object) -> None:
    import pathlib

    p = pathlib.Path(str(tmp_path)) / "resume.xyz"
    p.write_text("test")
    loader = ResumeLoader()
    with pytest.raises(DocumentError, match="Unsupported"):
        loader.load(p)


def test_resume_loader_text_file(tmp_path: object) -> None:
    import pathlib

    p = pathlib.Path(str(tmp_path)) / "resume.txt"
    p.write_text("John Doe\njohn@example.com\n555-0123\nSkills: Python, FastAPI")
    loader = ResumeLoader()
    resume = loader.load(p)
    assert resume.name == "John Doe"
    assert resume.email == "john@example.com"
    assert "Python" in resume.skills


def test_resume_loader_strips_markdown_bold_from_name(tmp_path: object) -> None:
    """A tailored résumé re-parsed as input should not keep ** around the name."""
    import pathlib

    p = pathlib.Path(str(tmp_path)) / "resume.txt"
    p.write_text("**Alex Rivera**\nalex@example.com\nSkills: Python\n")
    loader = ResumeLoader()
    resume = loader.load(p)
    assert resume.name == "Alex Rivera"


def test_resume_loader_recognizes_technical_skills_header(tmp_path: object) -> None:
    """Parser must recognize qualified skills headers (not just 'Skills')."""
    import pathlib

    p = pathlib.Path(str(tmp_path)) / "resume.txt"
    p.write_text(
        "Jane Smith\njane@example.com\n\n"
        "Technical Skills\nPython\nKubernetes\nTerraform\n\n"
        "Experience\nSenior Engineer at Acme"
    )
    loader = ResumeLoader()
    resume = loader.load(p)
    assert "Python" in resume.skills
    assert "Kubernetes" in resume.skills
    assert "Terraform" in resume.skills


def test_resume_loader_skips_markdown_underline_in_skills(tmp_path: object) -> None:
    """Setext/markdown underline lines under a header must not become a skill."""
    import pathlib

    p = pathlib.Path(str(tmp_path)) / "resume.txt"
    p.write_text(
        "Jane Smith\njane@example.com\n\n"
        "Skills\n------\nPython, FastAPI, PostgreSQL\n\n"
        "Experience\nSenior Engineer at Acme"
    )
    loader = ResumeLoader()
    resume = loader.load(p)
    assert "Python" in resume.skills
    assert "FastAPI" in resume.skills
    assert "PostgreSQL" in resume.skills
    assert "-----" not in resume.skills


def test_resume_loader_skips_markdown_underline_in_summary(tmp_path: object) -> None:
    """Setext/markdown underline lines under a Summary header must not pollute summary."""
    import pathlib

    p = pathlib.Path(str(tmp_path)) / "resume.txt"
    p.write_text(
        "Jane Smith\njane@example.com\n\n"
        "Summary\n-------\nSenior backend engineer with cloud experience.\n\n"
        "Experience\nSenior Engineer at Acme"
    )
    loader = ResumeLoader()
    resume = loader.load(p)
    assert resume.summary.startswith("Senior backend engineer")
    assert "-----" not in resume.summary


def test_summary_fallback_keeps_substantial_first_paragraph() -> None:
    """L-7: with no Summary/Objective header, a substantial first paragraph is the summary."""
    text = (
        "John Doe\njohn@example.com\n555-123-4567\n"
        "Seasoned backend engineer with a decade of experience building "
        "reliable distributed systems for fintech companies.\n\n"
        "Experience\nSenior Engineer at Acme"
    )
    resume = ResumeLoader().parse_text(text)
    assert resume.summary.startswith("Seasoned backend engineer")


def test_summary_fallback_drops_too_short_paragraph() -> None:
    """L-7: a tiny first paragraph is not treated as a summary."""
    text = "John Doe\njohn@example.com\n555-123-4567\nHello there.\n\nExperience\nDev"
    resume = ResumeLoader().parse_text(text)
    assert resume.summary == ""


def test_resume_loader_recognizes_core_competencies_header(tmp_path: object) -> None:
    """'Core Competencies' is a recognized inline skills header."""
    import pathlib

    p = pathlib.Path(str(tmp_path)) / "resume.txt"
    p.write_text(
        "Jane Smith\njane@example.com\n\n"
        "Core Competencies: Python, Docker, AWS\n\n"
        "Experience\nSenior Engineer at Acme"
    )
    loader = ResumeLoader()
    resume = loader.load(p)
    assert "Python" in resume.skills
    assert "Docker" in resume.skills
    assert "AWS" in resume.skills


def test_resume_loader_docx(tmp_path: object) -> None:
    """Test loading a DOCX resume."""
    import pathlib

    try:
        from docx import Document
    except ImportError:
        pytest.skip("python-docx not installed")

    doc = Document()
    doc.add_paragraph("Jane Smith")
    doc.add_paragraph("jane@example.com")
    doc.add_paragraph("Skills: Python, Docker, AWS")
    doc.add_paragraph("Experience: Senior Dev at TechCo")
    p = pathlib.Path(str(tmp_path)) / "resume.docx"
    doc.save(str(p))

    loader = ResumeLoader()
    resume = loader.load(p)
    assert resume.name == "Jane Smith"
    assert resume.email == "jane@example.com"
    assert "Python" in resume.skills


def test_resume_loader_docx_missing_dependency(tmp_path: object) -> None:
    """Test DOCX loading when python-docx is not installed."""
    import pathlib

    p = pathlib.Path(str(tmp_path)) / "resume.docx"
    p.write_bytes(b"fake docx content")

    loader = ResumeLoader()
    with patch.dict("sys.modules", {"docx": None}):
        with pytest.raises(DocumentError, match="python-docx not installed"):
            loader.load(p)


def test_ocr_fallback_triggers_on_short_text(tmp_path: Path) -> None:
    pdf_path = tmp_path / "scanned.pdf"
    pdf_path.write_bytes(b"fake pdf bytes")

    loader = ResumeLoader()
    with (
        patch.object(loader, "_run_pdftotext", return_value=" "),
        patch.object(loader, "_run_pymupdf", return_value="  "),
        patch.object(loader, "_ocr_service", MagicMock()) as mock_ocr,
    ):
        # OCR fallback is triggered, but if both extractors AND OCR yield less
        # than OCR_THRESHOLD chars the loader now raises rather than silently
        # returning an unusable ResumeData.
        mock_ocr.extract_text_from_pdf.return_value = "John Doe\nSkills: Python"
        with pytest.raises(DocumentError, match="insufficient extractable text"):
            loader._load_pdf(pdf_path, ocr_mode="auto")

    mock_ocr.extract_text_from_pdf.assert_called_once_with(pdf_path)


def test_force_ocr_skips_text_extraction(tmp_path: Path) -> None:
    pdf_path = tmp_path / "scanned.pdf"
    pdf_path.write_bytes(b"fake pdf bytes")

    loader = ResumeLoader()
    with (
        patch.object(loader, "_run_pdftotext") as mock_pdftotext,
        patch.object(loader, "_run_pymupdf") as mock_pymupdf,
        patch.object(loader, "_ocr_service", MagicMock()) as mock_ocr,
    ):
        mock_ocr.extract_text_from_pdf.return_value = "OCR text"
        result = loader._load_pdf(pdf_path, ocr_mode="on")

    mock_pdftotext.assert_not_called()
    mock_pymupdf.assert_not_called()
    mock_ocr.extract_text_from_pdf.assert_called_once_with(pdf_path)
    assert result.raw_text == "OCR text"


def test_ocr_mode_off_disables_ocr(tmp_path: Path) -> None:
    pdf_path = tmp_path / "scanned.pdf"
    pdf_path.write_bytes(b"fake pdf bytes")

    long_text = "A" * 150
    loader = ResumeLoader()
    with (
        patch.object(loader, "_run_pdftotext", return_value=long_text),
        patch.object(loader, "_run_pymupdf", return_value=""),
        patch.object(loader, "_ocr_service", MagicMock()) as mock_ocr,
    ):
        result = loader._load_pdf(pdf_path, ocr_mode="off")

    mock_ocr.extract_text_from_pdf.assert_not_called()
    assert long_text in result.raw_text


def test_ocr_mode_off_insufficient_text_raises(tmp_path: Path) -> None:
    pdf_path = tmp_path / "scanned.pdf"
    pdf_path.write_bytes(b"fake pdf bytes")

    loader = ResumeLoader()
    with (
        patch.object(loader, "_run_pdftotext", return_value="X"),
        patch.object(loader, "_run_pymupdf", return_value="X"),
        patch.object(loader, "_ocr_service", MagicMock()) as mock_ocr,
    ):
        with pytest.raises(DocumentError, match="insufficient extractable text"):
            loader._load_pdf(pdf_path, ocr_mode="off")

    mock_ocr.extract_text_from_pdf.assert_not_called()


def test_ocr_failure_falls_back_to_extracted_text(tmp_path: Path) -> None:
    pdf_path = tmp_path / "scanned.pdf"
    pdf_path.write_bytes(b"fake pdf bytes")

    long_text = "A" * 150
    loader = ResumeLoader()
    with (
        patch.object(loader, "_run_pdftotext", return_value="short"),
        patch.object(loader, "_run_pymupdf", return_value=long_text),
        patch.object(loader, "_ocr_service", MagicMock()) as mock_ocr,
    ):
        mock_ocr.extract_text_from_pdf.side_effect = DocumentError("OCR failed")
        result = loader._load_pdf(pdf_path, ocr_mode="auto")

    assert long_text in result.raw_text


def test_ocr_failure_with_no_text_raises(tmp_path: Path) -> None:
    pdf_path = tmp_path / "scanned.pdf"
    pdf_path.write_bytes(b"fake pdf bytes")

    loader = ResumeLoader()
    with (
        patch.object(loader, "_run_pdftotext", return_value=""),
        patch.object(loader, "_run_pymupdf", return_value=""),
        patch.object(loader, "_ocr_service", MagicMock()) as mock_ocr,
    ):
        mock_ocr.extract_text_from_pdf.side_effect = DocumentError("OCR failed")
        with pytest.raises(DocumentError):
            loader._load_pdf(pdf_path, ocr_mode="auto")


def test_image_resume_uses_ocr(tmp_path: Path) -> None:
    img_path = tmp_path / "resume.png"
    from PIL import Image

    Image.new("RGB", (50, 50), color="white").save(img_path)

    loader = ResumeLoader()
    with patch.object(loader, "_ocr_service", MagicMock()) as mock_ocr:
        mock_ocr.extract_text_from_image.return_value = "OCR text"
        result = loader.load(img_path, ocr_mode="on")

    mock_ocr.extract_text_from_image.assert_called_once_with(img_path)
    assert result.raw_text == "OCR text"


def test_force_ocr_failure_raises(tmp_path: Path) -> None:
    pdf_path = tmp_path / "scanned.pdf"
    pdf_path.write_bytes(b"fake pdf bytes")

    loader = ResumeLoader()
    with patch.object(loader, "_ocr_service", MagicMock()) as mock_ocr:
        mock_ocr.extract_text_from_pdf.side_effect = DocumentError("OCR failed")
        with pytest.raises(DocumentError):
            loader._load_pdf(pdf_path, ocr_mode="on")


def test_parse_text_records_confidence_and_method() -> None:
    loader = ResumeLoader()
    text = "John Doe\njohn@example.com\n555-0123\nSkills: Python\nExperience\nEducation"
    resume = loader.parse_text(text, method="text")
    assert resume.parse_method == "text"
    assert resume.parse_confidence > 0.0
    assert resume.parse_confidence <= 1.0


def test_compute_confidence_empty_text() -> None:
    loader = ResumeLoader()
    assert loader._compute_confidence("") == 0.0
    assert loader._compute_confidence("   ") == 0.0


def test_compute_confidence_increases_with_signals() -> None:
    loader = ResumeLoader()
    base = loader._compute_confidence("John Doe\n")
    with_contact = loader._compute_confidence("John Doe\njohn@example.com\n555-0123\n")
    with_sections = loader._compute_confidence(
        "John Doe\njohn@example.com\n555-0123\nSkills: Python\nExperience\nEducation\n"
    )
    assert with_contact > base
    assert with_sections > with_contact


def test_pdf_consensus_selects_best_parser(tmp_path: Path) -> None:
    pdf_path = tmp_path / "resume.pdf"
    pdf_path.write_bytes(b"fake pdf bytes")

    loader = ResumeLoader()
    good_text = "A" * 200 + "\njohn@example.com\nSkills: Python\nExperience\nEducation"
    bad_text = "garbled"
    with (
        patch.object(loader, "_run_pdftotext", return_value=bad_text),
        patch.object(loader, "_run_pymupdf", return_value=good_text),
    ):
        result = loader._load_pdf(pdf_path, ocr_mode="off")

    assert result.raw_text == good_text
    assert result.parse_method == "pymupdf"
    assert result.parse_confidence > 0.5


def test_password_protected_pdf_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf_path = tmp_path / "locked.pdf"
    pdf_path.write_bytes(b"fake pdf bytes")

    def locked(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("Document is password protected")

    monkeypatch.setattr("fitz.open", locked)
    loader = ResumeLoader()
    with pytest.raises(DocumentError, match="password-protected"):
        loader._load_pdf(pdf_path, ocr_mode="off")


def test_resume_loader_wraps_unreadable_docx_as_document_error(tmp_path: Path) -> None:
    """A corrupt/empty .docx surfaces as a typed DocumentError via load() — python-docx's
    PackageNotFoundError must not leak as a traceback (the loader's contract is
    JobApplicatorError-only)."""
    bad = tmp_path / "empty.docx"
    bad.write_bytes(b"")
    with pytest.raises(DocumentError):
        ResumeLoader().load(bad)


def test_resume_loader_corrupt_pdf_is_document_error(tmp_path: Path) -> None:
    """A corrupt .pdf surfaces as a typed DocumentError — here via the PDF consensus
    'insufficient extractable text' path (ocr off). Guards the user-facing contract that
    load() raises only JobApplicatorError; the load() try/except wrapper itself is
    exercised by the corrupt-.docx test above (python-docx PackageNotFoundError → DocumentError).
    """
    bad = tmp_path / "corrupt.pdf"
    bad.write_bytes(b"not a real pdf %%\x00\x01garbage")
    with pytest.raises(DocumentError):
        ResumeLoader().load(bad, ocr_mode="off")


def test_resume_loader_empty_text_is_document_error(tmp_path: Path) -> None:
    """A VALID file with no extractable text (empty/whitespace) → typed DocumentError
    ('no extractable text'), not a misleading 0.14 ATS 'score' on nothing (QA pass-2 B3)."""
    blank = tmp_path / "blank.txt"
    blank.write_text("   \n\t  ", encoding="utf-8")
    with pytest.raises(DocumentError, match="no extractable text"):
        ResumeLoader().load(blank)


def test_resume_loader_directory_path_names_the_target(tmp_path: Path) -> None:
    """A directory (no extension) → DocumentError that NAMES the path, not an empty
    'Unsupported resume format: ' message."""
    with pytest.raises(DocumentError) as ei:
        ResumeLoader().load(tmp_path)
    assert tmp_path.name in str(ei.value)


def _validation_job_and_resume() -> tuple[JobListing, ResumeData]:
    """Return a job/company that does NOT appear on the resume."""
    job = JobListing(
        title="Python Dev", company="Acme", url="https://example.com/1", board=JobBoard.LINKEDIN
    )
    resume = ResumeData(raw_text="John Doe\nBackend engineer at OtherCorp")
    return job, resume


def test_cover_letter_validation_rejects_empty() -> None:
    config = LLMConfig()
    generator = CoverLetterGenerator(config)
    user = UserProfile(first_name="John", last_name="Doe", email="j@e.com", phone="")
    job, resume = _validation_job_and_resume()
    with pytest.raises(LLMError, match="empty"):
        generator._validate_output("   ", user, job=job, resume=resume)


def test_cover_letter_validation_rejects_too_short() -> None:
    config = LLMConfig()
    generator = CoverLetterGenerator(config)
    user = UserProfile(first_name="John", last_name="Doe", email="j@e.com", phone="")
    job, resume = _validation_job_and_resume()
    with pytest.raises(LLMError, match="too short"):
        generator._validate_output("Sincerely,\nJohn Doe", user, job=job, resume=resume)


def test_cover_letter_validation_rejects_placeholders() -> None:
    config = LLMConfig()
    generator = CoverLetterGenerator(config)
    user = UserProfile(first_name="John", last_name="Doe", email="j@e.com", phone="")
    job, resume = _validation_job_and_resume()
    with pytest.raises(LLMError, match="placeholder"):
        generator._validate_output(
            "Dear [Hiring Manager],\n\nBody text that is long enough to pass the length check. "
            "It keeps going so the validator does not reject it for being too short.\n\n"
            "Sincerely,\nJohn Doe",
            user,
            job=job,
            resume=resume,
        )


def test_cover_letter_humanize_strips_sign_off_at_top() -> None:
    """A stray sign-off before the body is removed, keeping the valid closing."""
    bad_letter = (
        "Sincerely,\nJohn Doe\n\n"
        "I have ten years of experience with Python and FastAPI. "
        "This body is long enough to pass the minimum length check.\n\n"
        "Sincerely,\nJohn Doe"
    )
    cleaned = CoverLetterGenerator._humanize(bad_letter)
    assert cleaned.startswith("I have ten years of experience")
    assert cleaned.endswith("Sincerely,\nJohn Doe")


def test_cover_letter_validation_rejects_invented_employment() -> None:
    """A letter that falsely claims employment at the target company is rejected."""
    config = LLMConfig()
    generator = CoverLetterGenerator(config)
    user = UserProfile(first_name="John", last_name="Doe", email="j@e.com", phone="")
    job, resume = _validation_job_and_resume()
    bad_letter = (
        "Dear Hiring Team,\n\n"
        "I previously worked at Acme, where I led the backend team. "
        "This body is long enough to pass the minimum length check and keep going.\n\n"
        "Sincerely,\nJohn Doe"
    )
    with pytest.raises(LLMError, match="falsely claims employment"):
        generator._validate_output(bad_letter, user, job=job, resume=resume)


def test_cover_letter_generator_template() -> None:
    config = LLMConfig()
    generator = CoverLetterGenerator(config)
    job = JobListing(
        title="Python Dev",
        company="Acme",
        url="https://example.com/1",
        board=JobBoard.LINKEDIN,
    )
    user = UserProfile(
        first_name="John",
        last_name="Doe",
        email="john@example.com",
        phone="555-0123",
    )
    resume = ResumeData(raw_text="test", skills=["Python"])
    letter = generator.generate_from_template(job, user, resume)
    assert "Acme" in letter
    assert "Python Dev" in letter
    assert "John" in letter


def test_cover_letter_output_model() -> None:
    from job_applicator.documents.cover_letter import CoverLetterOutput

    output = CoverLetterOutput(
        cover_letter="Dear Hiring Manager, ...",
        key_points=["Python experience", "FastAPI expertise"],
    )
    assert output.cover_letter.startswith("Dear")
    assert len(output.key_points) == 2


def test_cover_letter_system_prompt_enforces_human_voice() -> None:
    """System prompt must carry anti-template voice rules and NOT parroted examples.

    The verbatim few-shot example paragraphs were removed deliberately: the 4B model
    copied them into output (see the keyfigures-example-hallucination lesson). The
    contract is now structural voice rules instead.
    """
    from job_applicator.documents.cover_letter import SYSTEM_PROMPT

    low = SYSTEM_PROMPT.lower()
    assert "example" not in low  # no copy-pasteable few-shot text
    assert "vary sentence length" in low
    assert "proven track record" in low  # named as a banned cliché
    assert "plain prose" in low or "no markdown" in low


def test_cover_letter_system_prompt_has_hallucination_guard() -> None:
    """System prompt should warn against inventing experience."""
    from job_applicator.documents.cover_letter import SYSTEM_PROMPT

    assert "not in the resume" in SYSTEM_PROMPT.lower() or "invent" in SYSTEM_PROMPT.lower()


def test_ocr_service_extracts_text_from_image(tmp_path: Path) -> None:
    from job_applicator.documents.ocr import OCRService

    service = OCRService()
    # PaddleOCR is lazy-loaded; mock it to avoid heavy model init in unit tests.
    service._ocr = MagicMock()
    service._ocr.ocr.return_value = [[([[0, 0], [10, 0], [10, 10], [0, 10]], ("Hello", 0.99))]]

    img_path = tmp_path / "resume.png"
    # Create a tiny blank PNG using PIL
    from PIL import Image

    Image.new("RGB", (50, 50), color="white").save(img_path)

    text = service.extract_text_from_image(img_path)
    assert "Hello" in text


@pytest.fixture
def llm_config() -> LLMConfig:
    return LLMConfig(api_base="http://localhost:8000/v1", model="test-model")


class TestCoverLetterWithTone:
    @pytest.mark.asyncio
    async def test_generate_includes_tone_section(self, llm_config: LLMConfig) -> None:
        generator = CoverLetterGenerator(llm_config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[
            0
        ].message.content = "Dear Hiring Manager,\n\nCover letter text.\n\nSincerely,\nJohn Doe"

        job = JobListing(
            title="Dev",
            company="Co",
            url="https://example.com",
            board=JobBoard.INDEED,
        )
        user = UserProfile(first_name="John", last_name="Doe", email="j@e.com", phone="123")
        resume = ResumeData(raw_text="Resume text", skills=["Python"])

        with patch(
            "litellm.acompletion", new_callable=AsyncMock, return_value=mock_response
        ) as mock_call:
            await generator.generate(
                job,
                user,
                resume,
                tone_section="TONE: Corporate\n- Power words: leveraged",
            )

        call_args = mock_call.call_args
        assert "TONE: Corporate" in str(call_args)

    @pytest.mark.asyncio
    async def test_generate_uses_tailored_resume_text(self, llm_config: LLMConfig) -> None:
        generator = CoverLetterGenerator(llm_config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[
            0
        ].message.content = "Dear Hiring Manager,\n\nCover letter.\n\nSincerely,\nJohn Doe"

        job = JobListing(
            title="Dev",
            company="Co",
            url="https://example.com",
            board=JobBoard.INDEED,
        )
        user = UserProfile(first_name="John", last_name="Doe", email="j@e.com", phone="123")
        resume = ResumeData(raw_text="Original resume", skills=["Python"])
        tailored = "Tailored resume with optimized Python experience"

        with patch(
            "litellm.acompletion", new_callable=AsyncMock, return_value=mock_response
        ) as mock_call:
            await generator.generate(
                job,
                user,
                resume,
                tailored_resume_text=tailored,
            )

        call_args = mock_call.call_args
        prompt = str(call_args)
        assert "Tailored resume with optimized" in prompt

    @pytest.mark.asyncio
    async def test_refine_honors_configured_max_tokens(self) -> None:
        """refine()'s instructor call must pass the configured max_tokens, not omit it."""
        config = LLMConfig(api_base="http://localhost:8000/v1", model="m", max_tokens=1234)
        generator = CoverLetterGenerator(config)

        mock_output = MagicMock()
        mock_output.cover_letter = (
            "Dear Hiring Manager,\n\nRefined cover letter.\n\nSincerely,\nJohn Doe"
        )
        mock_client = MagicMock()
        mock_client.create = AsyncMock(return_value=mock_output)

        job = JobListing(
            title="Dev",
            company="Co",
            url="https://example.com",
            board=JobBoard.INDEED,
        )
        user = UserProfile(first_name="John", last_name="Doe", email="j@e.com", phone="123")
        resume = ResumeData(raw_text="Resume text", skills=["Python"])

        with patch.object(generator, "_get_client", return_value=mock_client):
            await generator.refine(
                job,
                user,
                resume,
                current_text="Old cover letter.",
                user_feedback="Make it punchier.",
            )

        assert mock_client.create.call_args.kwargs["max_tokens"] == 1234


def test_password_protected_pdf_detected_via_needs_pass(tmp_path: Path) -> None:
    """M3: PyMuPDF opens an encrypted PDF without raising; needs_pass is the gate."""
    pdf_path = tmp_path / "locked.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")
    loader = ResumeLoader()
    fake_doc = MagicMock()
    fake_doc.needs_pass = True
    fake_fitz = MagicMock()
    fake_fitz.open.return_value = fake_doc
    with patch.dict("sys.modules", {"fitz": fake_fitz}):
        assert loader._is_password_protected(pdf_path) is True
    fake_doc.close.assert_called_once()


def test_unprotected_pdf_not_flagged_as_password_protected(tmp_path: Path) -> None:
    """M3: a normal PDF (needs_pass False) is not flagged as protected."""
    pdf_path = tmp_path / "ok.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")
    loader = ResumeLoader()
    fake_doc = MagicMock()
    fake_doc.needs_pass = False
    fake_fitz = MagicMock()
    fake_fitz.open.return_value = fake_doc
    with patch.dict("sys.modules", {"fitz": fake_fitz}):
        assert loader._is_password_protected(pdf_path) is False


def test_confidence_counts_headers_not_midline_mentions() -> None:
    """Confidence counts line-anchored section HEADERS, not bare substrings."""
    loader = ResumeLoader()
    headers = "Experience\nEducation\nSkills\nProjects\nCertifications"
    prose = "my experience education skills projects certifications summary blah"
    assert loader._compute_confidence(headers) > loader._compute_confidence(prose)


def test_confidence_vocab_aligns_with_skills_extractor() -> None:
    """Confidence credits 'Core Competencies' (aligned with _extract_skills_section)."""
    loader = ResumeLoader()
    with_header = "Jane Doe\n\nCore Competencies\nPython, SQL"
    without = "Jane Doe\n\nrandom prose mentioning competencies in a sentence"
    assert loader._compute_confidence(with_header) > loader._compute_confidence(without)


def test_phone_extraction_rejects_year_runs() -> None:
    """Phone: a run of years must not be mistaken for a phone number."""
    loader = ResumeLoader()
    assert loader._extract_phone("Worked 2019 2020 2021 2022 2023 at Acme") == ""


def test_phone_extraction_accepts_real_phone() -> None:
    """Phone: normally-formatted numbers are still extracted."""
    loader = ResumeLoader()
    assert "555" in loader._extract_phone("Call me at 555-123-4567")
    assert loader._extract_phone("Reach me: +1 (555) 123-4567").strip() != ""


# --- Cover-letter human-voice helpers (de-AI pass) ----------------------------

# A draft exhibiting the robotic tells observed in real 4B output: parroted
# clichés, every sentence long, trailing participial clauses, and markdown leak.
ROBOTIC_LETTER = (
    "I am excited to apply for the Senior Backend Engineer position at Globex, "
    "where my expertise aligns directly with your technical requirements. "
    "At Acme Data, I architected an async ingestion system using `asyncio` that "
    "handled two billion events daily, demonstrating my ability to design scalable systems. "
    "I led a migration across forty services, ensuring type safety and reducing runtime errors. "
    "I automated deployment workflows on AWS, creating a seamless environment "
    "for high availability. "
    "I have a proven track record of delivering resilient services, scaling rapidly "
    "across every region, and I sincerely look forward to hearing from you and your team soon.\n\n"
    "Sincerely,\nJ D"
)

# A human-sounding draft: varied cadence (short sentences present), no clichés,
# no trailing participials, no markdown.
HUMAN_LETTER = (
    "I built Globex's billing pipeline three years ago. It still runs. "
    "Now you want someone to scale it past two billion events a day, and I want that job. "
    "At Acme I halved ingestion latency by rewriting the consumer in asyncio. "
    "I know your stack. Let's talk.\n\n"
    "Sincerely,\nJ D"
)


def test_humanize_strips_inline_code_backticks() -> None:
    assert CoverLetterGenerator._humanize("I use `asyncio` and `mypy` daily.") == (
        "I use asyncio and mypy daily."
    )


def test_humanize_strips_markdown_bullets_at_line_start() -> None:
    assert CoverLetterGenerator._humanize("Skills:\n- Python\n- asyncio") == (
        "Skills:\nPython\nasyncio"
    )


def test_humanize_leaves_inline_asterisks_untouched() -> None:
    # Asterisks are ambiguous (multiplication, ratings, footnotes); stripping them
    # would corrupt prose like "2*3" -> "23". Only line-anchored bullets are removed.
    text = "I improved throughput 2*3 fold and was rated 5* by clients."
    assert CoverLetterGenerator._humanize(text) == text


def test_humanize_strips_markdown_headings() -> None:
    assert CoverLetterGenerator._humanize("# Heading\n\nBody text.") == "Heading\n\nBody text."


def test_humanize_collapses_excess_blank_lines() -> None:
    assert CoverLetterGenerator._humanize("Line one.\n\n\n\nLine two.") == "Line one.\n\nLine two."


def test_humanize_leaves_plain_prose_untouched() -> None:
    text = "Dear Hiring Manager, I write to apply for the role."
    assert CoverLetterGenerator._humanize(text) == text


def test_humanize_does_not_touch_snake_case_underscores() -> None:
    # Underscores inside identifiers must survive (no underscore-italic stripping).
    assert CoverLetterGenerator._humanize("Call the get_user_id helper.") == (
        "Call the get_user_id helper."
    )


def test_voice_tells_flags_robotic_letter() -> None:
    tells = CoverLetterGenerator._voice_tells(ROBOTIC_LETTER)
    assert "markdown" in tells
    assert any(t.startswith("cliche:") for t in tells)
    assert "participial_tails" in tells
    assert "no_short_sentences" in tells


def test_voice_tells_clears_human_letter() -> None:
    assert CoverLetterGenerator._voice_tells(HUMAN_LETTER) == []


def test_voice_tells_ignores_trivial_text() -> None:
    # Short/mocked text must not false-positive (keeps unit mocks from re-prompting).
    assert CoverLetterGenerator._voice_tells("Dear Hiring Manager,\n\nCover letter text.") == []


def test_voice_tells_excludes_trailing_sign_off_block() -> None:
    """A valid sign-off block must not suppress the short-sentence tell."""
    body = (
        "Sentence one has many words to exceed the short sentence threshold. "
        "Sentence two also has more than eight words to avoid the short tell. "
        "Sentence three continues the pattern with plenty of words included. "
        "Sentence four is similarly long and descriptive enough to qualify."
    )
    letter = f"{body}\n\nSincerely,\nJohn Doe"
    tells = CoverLetterGenerator._voice_tells(letter)
    assert "no_short_sentences" in tells


def test_company_in_resume_matches_structured_employer() -> None:
    resume = ResumeData(raw_text="...", experience=[ExperienceEntry(company="Globex")])
    assert CoverLetterGenerator._company_in_resume("Globex", resume) is True


def test_company_in_resume_normalizes_legal_suffixes() -> None:
    resume = ResumeData(raw_text="...", experience=[ExperienceEntry(company="Globex")])
    assert CoverLetterGenerator._company_in_resume("globex inc.", resume) is True


def test_company_in_resume_falls_back_to_raw_text() -> None:
    resume = ResumeData(raw_text="Backend Engineer, Globex (2017-2021)")
    assert CoverLetterGenerator._company_in_resume("Globex", resume) is True


def test_company_in_resume_returns_false_when_absent() -> None:
    resume = ResumeData(raw_text="Backend Engineer, Acme (2017-2021)")
    assert CoverLetterGenerator._company_in_resume("Globex", resume) is False


def test_company_in_resume_rejects_email_domain_and_school_name() -> None:
    """A company stem like 'example' must not match an email domain or school name."""
    resume = ResumeData(raw_text="Jane Doe\njane@example.com\nB.S., Example University")
    assert CoverLetterGenerator._company_in_resume("Example Corp", resume) is False


def test_company_in_resume_empty_company_is_false() -> None:
    resume = ResumeData(raw_text="Backend Engineer, Globex (2017-2021)")
    assert CoverLetterGenerator._company_in_resume("", resume) is False


def _cl_inputs() -> tuple[JobListing, UserProfile, ResumeData]:
    job = JobListing(title="Dev", company="Co", url="https://e.com", board=JobBoard.INDEED)
    user = UserProfile(first_name="J", last_name="D", email="j@e.com", phone="1")
    resume = ResumeData(raw_text="resume", skills=["Python"])
    return job, user, resume


@pytest.mark.asyncio
async def test_generate_revoices_when_first_draft_is_robotic() -> None:
    """A robotic first draft triggers ONE voice re-prompt; the cleaner retry wins."""
    config = LLMConfig(api_base="http://localhost:8000/v1", model="m")
    generator = CoverLetterGenerator(config)
    generator._complete = AsyncMock(side_effect=[ROBOTIC_LETTER, HUMAN_LETTER])  # type: ignore[method-assign]
    letter = await generator.generate(*_cl_inputs())
    assert generator._complete.await_count == 2
    assert CoverLetterGenerator._voice_tells(letter) == []  # cleaner retry kept


# Two DISTINCT robotic drafts with EQUAL tell counts (one cliché each) — needed to
# prove _devoice keeps the FIRST draft on a tie (its rule is strict `<`, not `<=`);
# identical drafts would make the kept-draft assertion vacuous.
_TIE_FIRST = "I have a proven track record. I ship fast. Hire me today.\n\nSincerely,\nJ D"
_TIE_RETRY = "I am a perfect fit for this. I work hard. Pick me now.\n\nSincerely,\nJ D"


@pytest.mark.asyncio
async def test_generate_keeps_first_draft_on_tell_count_tie() -> None:
    """On an equal tell count the FIRST draft is kept (strict `<`, never `<=`)."""
    assert len(CoverLetterGenerator._voice_tells(_TIE_FIRST)) == 1
    assert len(CoverLetterGenerator._voice_tells(_TIE_RETRY)) == 1
    config = LLMConfig(api_base="http://localhost:8000/v1", model="m")
    generator = CoverLetterGenerator(config)
    generator._complete = AsyncMock(side_effect=[_TIE_FIRST, _TIE_RETRY])  # type: ignore[method-assign]
    letter = await generator.generate(*_cl_inputs())
    assert generator._complete.await_count == 2
    assert letter == _TIE_FIRST  # the equal-tell retry must NOT replace the first draft


@pytest.mark.asyncio
async def test_generate_skips_revoice_when_first_draft_is_clean() -> None:
    """A clean first draft does NOT trigger a re-prompt (no wasted LLM call)."""
    config = LLMConfig(api_base="http://localhost:8000/v1", model="m")
    generator = CoverLetterGenerator(config)
    generator._complete = AsyncMock(side_effect=[HUMAN_LETTER])  # type: ignore[method-assign]
    letter = await generator.generate(*_cl_inputs())
    assert generator._complete.await_count == 1
    assert letter.strip()


@pytest.mark.asyncio
async def test_generate_revoice_is_graceful_on_error() -> None:
    """If the voice re-prompt errors, return the first usable draft (no raise)."""
    config = LLMConfig(api_base="http://localhost:8000/v1", model="m")
    generator = CoverLetterGenerator(config)
    generator._complete = AsyncMock(  # type: ignore[method-assign]
        side_effect=[ROBOTIC_LETTER, LLMError("transport blip")]
    )
    letter = await generator.generate(*_cl_inputs())
    assert generator._complete.await_count == 2
    assert letter == CoverLetterGenerator._humanize(ROBOTIC_LETTER)  # first usable draft preserved


# --- Multi-file style-guide loader --------------------------------------------------


class TestLoadStyleGuide:
    """Cycle 2b: CoverLetterGenerator.load_style_guide is the single shared helper
    used by apply, batch, tailor, and generate-cover-letter."""

    @pytest.fixture
    def config(self) -> LLMConfig:
        return LLMConfig(api_base="http://localhost:8000/v1", model="test-model")

    @pytest.fixture
    def generator(self, config: LLMConfig) -> CoverLetterGenerator:
        return CoverLetterGenerator(config)

    @pytest.fixture
    def style(self) -> StyleGuide:
        return StyleGuide(
            tone="professional",
            sentence_structure="varied",
            vocabulary_level="technical",
            paragraph_style="clear",
            formatting_notes="",
            sample_paragraph="",
        )

    @pytest.mark.asyncio
    async def test_single_text_file_analyzed(
        self,
        generator: CoverLetterGenerator,
        tmp_path: Path,
        style: StyleGuide,
    ) -> None:
        path = tmp_path / "style.txt"
        path.write_text("Professional cover letter tone example.", encoding="utf-8")
        resume = ResumeData(
            raw_text="Professional cover letter tone example.", name="", email="", skills=[]
        )

        with (
            patch(
                "job_applicator.documents.cover_letter.ResumeLoader.load",
                return_value=resume,
            ) as mock_load,
            patch(
                "job_applicator.documents.cover_letter.StyleAnalyzer.analyze",
                new_callable=AsyncMock,
                return_value=style,
            ) as mock_analyze,
        ):
            result = await generator.load_style_guide(str(path))

        assert result is style
        mock_load.assert_called_once_with(path, ocr_mode="auto")
        mock_analyze.assert_awaited_once_with("Professional cover letter tone example.")

    @pytest.mark.asyncio
    async def test_single_pdf_loaded_via_resume_loader(
        self,
        generator: CoverLetterGenerator,
        tmp_path: Path,
        style: StyleGuide,
    ) -> None:
        pdf_path = tmp_path / "style.pdf"
        pdf_path.write_text("fake pdf", encoding="utf-8")
        resume = ResumeData(raw_text="PDF style text", name="", email="", skills=[])

        with (
            patch(
                "job_applicator.documents.cover_letter.ResumeLoader.load",
                return_value=resume,
            ) as mock_load,
            patch(
                "job_applicator.documents.cover_letter.StyleAnalyzer.analyze",
                new_callable=AsyncMock,
                return_value=style,
            ) as mock_analyze,
        ):
            result = await generator.load_style_guide(str(pdf_path), ocr_mode="on")

        assert result is style
        mock_load.assert_called_once_with(pdf_path, ocr_mode="on")
        mock_analyze.assert_awaited_once_with("PDF style text")

    @pytest.mark.asyncio
    async def test_multiple_text_files_use_analyze_multiple(
        self,
        generator: CoverLetterGenerator,
        tmp_path: Path,
        style: StyleGuide,
    ) -> None:
        p1 = tmp_path / "a.txt"
        p2 = tmp_path / "b.txt"
        p1.write_text("Tone A", encoding="utf-8")
        p2.write_text("Tone B", encoding="utf-8")

        def _fake_load(path: Path, ocr_mode: str = "auto") -> ResumeData:
            text = "Tone A" if path.name == "a.txt" else "Tone B"
            return ResumeData(raw_text=text, name="", email="", skills=[])

        with (
            patch(
                "job_applicator.documents.cover_letter.ResumeLoader.load",
                side_effect=_fake_load,
            ) as mock_load,
            patch(
                "job_applicator.documents.cover_letter.StyleAnalyzer.analyze_multiple",
                new_callable=AsyncMock,
                return_value=style,
            ) as mock_multiple,
        ):
            result = await generator.load_style_guide(f"{p1}, {p2}")

        assert result is style
        assert mock_load.call_count == 2
        mock_multiple.assert_awaited_once_with(["Tone A", "Tone B"])

    @pytest.mark.asyncio
    async def test_mixed_pdf_and_text_use_analyze_multiple(
        self,
        generator: CoverLetterGenerator,
        tmp_path: Path,
        style: StyleGuide,
    ) -> None:
        txt = tmp_path / "a.txt"
        pdf = tmp_path / "b.pdf"
        txt.write_text("Text tone", encoding="utf-8")
        pdf.write_text("fake", encoding="utf-8")

        def _fake_load(path: Path, ocr_mode: str = "auto") -> ResumeData:
            text = "Text tone" if path.name == "a.txt" else "PDF tone"
            return ResumeData(raw_text=text, name="", email="", skills=[])

        with (
            patch(
                "job_applicator.documents.cover_letter.ResumeLoader.load",
                side_effect=_fake_load,
            ) as mock_load,
            patch(
                "job_applicator.documents.cover_letter.StyleAnalyzer.analyze_multiple",
                new_callable=AsyncMock,
                return_value=style,
            ) as mock_multiple,
        ):
            result = await generator.load_style_guide(f"{txt},{pdf}")

        assert result is style
        assert mock_load.call_count == 2
        mock_multiple.assert_awaited_once_with(["Text tone", "PDF tone"])

    @pytest.mark.asyncio
    async def test_missing_file_raises(self, generator: CoverLetterGenerator) -> None:
        with pytest.raises(DocumentError, match="not found"):
            await generator.load_style_guide("/nonexistent/style.txt")

    @pytest.mark.asyncio
    async def test_empty_string_raises(self, generator: CoverLetterGenerator) -> None:
        with pytest.raises(DocumentError, match="No style guide paths"):
            await generator.load_style_guide("  ,  ")
