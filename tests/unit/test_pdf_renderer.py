from __future__ import annotations

import builtins
import re
from pathlib import Path
from unittest.mock import patch

import pytest

from job_applicator.documents.cover_letter import CoverLetterOutput
from job_applicator.documents.pdf_renderer import (
    PDFRenderer,
    _compile_typst,
    _typst_escape,
    typst_template_env,
)
from job_applicator.exceptions import PDFRenderError
from job_applicator.models import CoverLetterResult, TailoredResume


def _resume_text() -> str:
    return (
        "Alex Rivera\n"
        "alex@example.com | 514-555-0199 | Montreal, QC\n\n"
        "SUMMARY\n"
        "Support analyst focused on evidence and clear handoffs.\n\n"
        "EXPERIENCE\n"
        "Technical Support Advisor | UpClick | 2022 - Present\n"
        "• Preserved this exact source-backed bullet with 95% scope.\n\n"
        "EDUCATION\n"
        "Certificate in Cybersecurity | 2024\n\n"
        "SKILLS\n"
        "Python, Linux, networking"
    )


def _tailored(text: str | None = None) -> TailoredResume:
    return TailoredResume(
        original_path="r.pdf",
        tailored_text=text or _resume_text(),
        job_title="Engineer",
        job_company="Acme",
        match_score=0.8,
        semantic_score=0.8,
        skill_score=0.8,
        changes_summary="summary overlay",
    )


def _cover_text() -> str:
    return (
        "Dear Hiring Manager,\n\n"
        "I am applying for the Engineer position at Acme.\n\n"
        "I documented incidents and coordinated technical handoffs from source-backed work.\n\n"
        "I would welcome the opportunity to discuss my application.\n\n"
        "Sincerely,\n"
        "Alex Rivera"
    )


async def _fake_compile_pdf(_source_path: Path, output_path: Path) -> None:
    output_path.write_bytes(b"%PDF-1.4 fake")  # noqa: ASYNC240


async def _failing_compile_pdf(_source_path: Path, _output_path: Path) -> None:
    raise RuntimeError("compile failed")


@pytest.mark.parametrize(
    ("raw", "escaped"),
    [
        ('#_ *$"\\`{}[]\r\n<>@///*', r"\#\_ \*\$\"\\\`\{\}\[\]  \<\>\@\/\/\/\*"),
        ("Hello, world!", "Hello, world!"),
        ("see <intro> and me@example.com", r"see \<intro\> and me\@example.com"),
        ("// comment", r"\/\/ comment"),
        ("/* block */", r"\/\* block \*/"),
    ],
)
def test_typst_escape(raw: str, escaped: str) -> None:
    assert _typst_escape(raw) == escaped
    assert _typst_escape(escaped) == escaped


def test_typst_template_env_escapes_by_default(tmp_path: Path) -> None:
    (tmp_path / "test.typ").write_text("{{ value }}", encoding="utf-8")
    result = typst_template_env(tmp_path).get_template("test.typ").render(value="#_ *")
    assert result == r"\#\_ \*"


def test_templates_load() -> None:
    env = typst_template_env()
    context = {
        "resume": {"name": "Test", "experience": []},
        "cover_letter": {
            "recipient_company": "Acme",
            "date": "2026-06-25",
            "greeting": "Hi",
            "paragraphs": [],
            "closing": "Best",
            "signature": "Test",
        },
    }
    for name in (
        "cv/modern.typ",
        "cv/classic.typ",
        "cv/minimal.typ",
        "cover_letter/modern.typ",
        "cover_letter/classic.typ",
        "cover_letter/minimal.typ",
    ):
        assert env.get_template(name).render(**context).strip()


def test_resume_formatting_is_deterministic_and_source_ordered(
    app_settings,
    tmp_path: Path,
) -> None:
    renderer = PDFRenderer(settings=app_settings, output_dir=tmp_path)
    formatted = renderer._format_resume(_tailored(), "support")

    assert not hasattr(renderer, "_client")
    assert [line.text for line in formatted.source_preamble or []] == [
        "Alex Rivera",
        "alex@example.com | 514-555-0199 | Montreal, QC",
        "",
    ]
    sections = formatted.source_sections or []
    assert [section.heading for section in sections] == [
        "SUMMARY",
        "EXPERIENCE",
        "EDUCATION",
        "SKILLS",
    ]
    experience = sections[1]
    assert experience.lines[1].is_bullet
    assert experience.lines[1].text == "Preserved this exact source-backed bullet with 95% scope."


def test_resume_formatting_rejects_unstructured_text(app_settings, tmp_path: Path) -> None:
    renderer = PDFRenderer(settings=app_settings, output_dir=tmp_path)
    with pytest.raises(PDFRenderError, match="unstructured"):
        renderer._format_resume(_tailored("Alex only"), "support")


def test_cover_letter_formatting_is_deterministic(app_settings, tmp_path: Path) -> None:
    renderer = PDFRenderer(settings=app_settings, output_dir=tmp_path)
    result = CoverLetterResult(
        job_title="Engineer",
        job_company="Acme",
        cover_letter_text=_cover_text(),
    )
    formatted = renderer._format_cover_letter(result, None, "support")

    assert formatted.greeting == "Dear Hiring Manager,"
    assert formatted.paragraphs == [
        "I am applying for the Engineer position at Acme.",
        "I documented incidents and coordinated technical handoffs from source-backed work.",
        "I would welcome the opportunity to discuss my application.",
    ]
    assert formatted.closing == "Sincerely"
    assert formatted.signature == "Alex Rivera"
    assert re.fullmatch(r"[A-Z][a-z]+ \d{2}, \d{4}", formatted.date)


def test_cover_letter_formatting_fails_without_sign_off(app_settings, tmp_path: Path) -> None:
    renderer = PDFRenderer(settings=app_settings, output_dir=tmp_path)
    result = CoverLetterResult(
        job_title="Engineer",
        job_company="Acme",
        cover_letter_text="Dear Hiring Manager,\n\nA complete paragraph without a closing.",
    )
    with pytest.raises(PDFRenderError, match="recognized sign-off"):
        renderer._format_cover_letter(result, None, "support")


@pytest.mark.parametrize("template", ["modern", "classic", "minimal"])
async def test_render_resume_compiles_source_text(
    app_settings,
    tmp_path: Path,
    template: str,
) -> None:
    renderer = PDFRenderer(settings=app_settings, output_dir=tmp_path)
    captured = ""

    async def capture(source_path: Path, output_path: Path) -> None:
        nonlocal captured
        captured = source_path.read_text(encoding="utf-8")  # noqa: ASYNC240
        await _fake_compile_pdf(source_path, output_path)

    with patch.object(renderer, "_compile_pdf", capture):
        path = await renderer.render_resume(_tailored(), template=template)

    assert path.exists()
    assert "UpClick" in captured
    assert "Preserved this exact source-backed bullet with 95% scope." in captured
    assert re.fullmatch(
        rf"tailored_Acme_Engineer_\d{{8}}_\d{{6}}_\d{{6}}_{template}\.pdf",
        path.name,
    )


async def test_render_cover_letter_accepts_output_and_explicit_path(
    app_settings,
    tmp_path: Path,
) -> None:
    renderer = PDFRenderer(settings=app_settings, output_dir=tmp_path)
    explicit = tmp_path / "custom" / "letter.pdf"
    output = CoverLetterOutput(cover_letter=_cover_text())
    with patch.object(renderer, "_compile_pdf", _fake_compile_pdf):
        path = await renderer.render_cover_letter(output, output_path=explicit)
    assert path == explicit
    assert path.exists()


async def test_render_resume_uses_explicit_output_path(app_settings, tmp_path: Path) -> None:
    renderer = PDFRenderer(settings=app_settings, output_dir=tmp_path)
    explicit = tmp_path / "custom" / "resume.pdf"
    with patch.object(renderer, "_compile_pdf", _fake_compile_pdf):
        path = await renderer.render_resume(_tailored(), output_path=explicit)
    assert path == explicit
    assert path.exists()


def test_compile_typst_missing_package_message(tmp_path: Path) -> None:
    source_path = tmp_path / "source.typ"
    output_path = tmp_path / "out.pdf"
    source_path.write_text("", encoding="utf-8")
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "typst":
            raise ImportError("No module named 'typst'")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        with pytest.raises(
            PDFRenderError,
            match=r"typst package not installed; run: pip install 'job-applicator\[pdf\]'",
        ):
            _compile_typst(source_path, output_path)


async def test_render_and_compile_template_not_found(app_settings, tmp_path: Path) -> None:
    renderer = PDFRenderer(settings=app_settings, output_dir=tmp_path)
    with pytest.raises(PDFRenderError, match=r"Template not found: missing\.typ"):
        await renderer._render_and_compile(
            "missing.typ",
            {"resume": {"name": "A", "experience": []}},
            tmp_path / "out.pdf",
        )


async def test_compile_failure_preserves_temporary_source(app_settings, tmp_path: Path) -> None:
    renderer = PDFRenderer(settings=app_settings, output_dir=tmp_path)
    with patch.object(renderer, "_compile_pdf", _failing_compile_pdf):
        with pytest.raises(PDFRenderError, match="compile failed") as exc_info:
            await renderer.render_resume(_tailored(), output_path=tmp_path / "out.pdf")
    source = Path(exc_info.value.context["source"])
    assert source.is_file()  # noqa: ASYNC240
    assert "UpClick" in source.read_text(encoding="utf-8")  # noqa: ASYNC240


async def test_success_cleans_temporary_source(app_settings, tmp_path: Path) -> None:
    renderer = PDFRenderer(settings=app_settings, output_dir=tmp_path)
    with patch.object(renderer, "_compile_pdf", _fake_compile_pdf):
        await renderer.render_resume(_tailored(), output_path=tmp_path / "out.pdf")
    assert not list(tmp_path.glob("_tmp_*.typ"))  # noqa: ASYNC240
