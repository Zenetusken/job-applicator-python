"""Unit tests for documents layer."""

from __future__ import annotations

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
