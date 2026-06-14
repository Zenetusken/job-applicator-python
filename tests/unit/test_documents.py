"""Unit tests for documents layer."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from job_applicator.config import LLMConfig
from job_applicator.documents.cover_letter import CoverLetterGenerator
from job_applicator.documents.resume import ResumeLoader
from job_applicator.exceptions import DocumentError, ResumeNotFoundError
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
