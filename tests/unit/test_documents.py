"""Unit tests for documents layer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from job_applicator.config import LLMConfig
from job_applicator.documents.cover_letter import CoverLetterGenerator
from job_applicator.documents.resume import ResumeLoader
from job_applicator.exceptions import DocumentError, LLMError, ResumeNotFoundError
from job_applicator.models import JobBoard, JobListing, ResumeData, UserProfile


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
        mock_ocr.extract_text_from_pdf.return_value = "John Doe\nSkills: Python"
        result = loader._load_pdf(pdf_path, ocr_mode="auto")

    mock_ocr.extract_text_from_pdf.assert_called_once_with(pdf_path)
    assert "John Doe" in result.raw_text


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

    loader = ResumeLoader()
    with (
        patch.object(loader, "_run_pdftotext", return_value="short"),
        patch.object(loader, "_run_pymupdf", return_value="Some extracted text"),
        patch.object(loader, "_ocr_service", MagicMock()) as mock_ocr,
    ):
        mock_ocr.extract_text_from_pdf.side_effect = DocumentError("OCR failed")
        result = loader._load_pdf(pdf_path, ocr_mode="auto")

    assert "Some extracted text" in result.raw_text


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


def test_cover_letter_validation_rejects_empty() -> None:
    config = LLMConfig()
    generator = CoverLetterGenerator(config)
    with pytest.raises(LLMError, match="empty"):
        generator._validate_cover_letter("   ")


def test_cover_letter_validation_rejects_placeholders() -> None:
    config = LLMConfig()
    generator = CoverLetterGenerator(config)
    with pytest.raises(LLMError, match="placeholder"):
        generator._validate_cover_letter("Dear [Hiring Manager],")


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


def test_cover_letter_system_prompt_has_examples() -> None:
    """System prompt should contain example paragraphs."""
    from job_applicator.documents.cover_letter import SYSTEM_PROMPT

    assert "EXAMPLE" in SYSTEM_PROMPT
    assert "opening paragraph" in SYSTEM_PROMPT.lower() or "I am writing" in SYSTEM_PROMPT


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
        mock_response.choices[0].message.content = "Dear Hiring Manager,\n\nCover letter text."

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
        mock_response.choices[0].message.content = "Dear Hiring Manager,\n\nCover letter."

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
        mock_output.cover_letter = "Dear Hiring Manager,\n\nRefined cover letter."
        mock_client = MagicMock()
        mock_client.create = AsyncMock(return_value=mock_output)

        job = JobListing(
            title="Dev",
            company="Co",
            url="https://example.com",
            board=JobBoard.INDEED,
        )
        resume = ResumeData(raw_text="Resume text", skills=["Python"])

        with patch.object(generator, "_get_client", return_value=mock_client):
            await generator.refine(
                job,
                resume,
                current_text="Old cover letter.",
                user_feedback="Make it punchier.",
            )

        assert mock_client.create.call_args.kwargs["max_tokens"] == 1234
