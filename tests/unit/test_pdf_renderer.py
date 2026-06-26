from __future__ import annotations

import builtins
import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from job_applicator.documents.cover_letter import CoverLetterOutput
from job_applicator.documents.formatted_models import (
    FormattedCoverLetter,
    FormattedExperienceEntry,
    FormattedResume,
)
from job_applicator.documents.pdf_renderer import (
    PDFRenderer,
    _compile_typst,
    _typst_escape,
    typst_template_env,
)
from job_applicator.exceptions import PDFRenderError
from job_applicator.models import CoverLetterResult, TailoredResume


@pytest.mark.unit
def test_typst_escape_metacharacters() -> None:
    """Every Typst metacharacter, label/reference marker, and comment sequence is escaped.

    Newlines are replaced with spaces.
    """
    raw = '#_ *$"\\`{}[]\r\n<>@///*'
    escaped = _typst_escape(raw)
    assert escaped == r"\#\_ \*\$\"\\\`\{\}\[\]  \<\>\@\/\/\/\*"


@pytest.mark.unit
def test_typst_escape_plain_text_unchanged() -> None:
    assert _typst_escape("Hello, world!") == "Hello, world!"


@pytest.mark.unit
def test_typst_escape_is_idempotent() -> None:
    """Escaping an already-escaped string must not double-escape it."""
    once = _typst_escape("#_ *")
    twice = _typst_escape(once)
    assert once == twice


@pytest.mark.unit
def test_typst_escape_labels_and_references() -> None:
    """Angle brackets and at-signs used for Typst labels/references are escaped."""
    assert _typst_escape("see <intro> and email me@example.com") == (
        r"see \<intro\> and email me\@example.com"
    )


@pytest.mark.unit
def test_typst_escape_comments() -> None:
    """Comment sequences // and /* ... */ are escaped so they cannot start a comment."""
    assert _typst_escape("// not a comment") == r"\/\/ not a comment"
    assert _typst_escape("/* block */") == r"\/\* block \*/"


@pytest.mark.unit
def test_typst_template_env_has_escape_filter(tmp_path) -> None:
    """The typst_escape filter is available and finalize does not double-escape it."""
    template = tmp_path / "test.typ"
    template.write_text("{{ value | typst_escape }}")
    env = typst_template_env(tmp_path)
    result = env.get_template("test.typ").render(value="#_ *")
    assert result == r"\#\_ \*"


@pytest.mark.unit
def test_typst_template_env_finalize_escapes_by_default(tmp_path) -> None:
    """Values rendered without an explicit filter are escaped by finalize."""
    template = tmp_path / "test.typ"
    template.write_text("{{ value }}")
    env = typst_template_env(tmp_path)
    result = env.get_template("test.typ").render(value="#_ *")
    assert result == r"\#\_ \*"


@pytest.mark.unit
def test_templates_load() -> None:
    """All built-in CV and cover-letter templates load and render with minimal context."""
    env = typst_template_env()
    for name in [
        "cv/modern.typ",
        "cv/classic.typ",
        "cv/minimal.typ",
        "cover_letter/modern.typ",
        "cover_letter/classic.typ",
        "cover_letter/minimal.typ",
    ]:
        source = env.get_template(name).render(
            resume={"name": "Test", "experience": []},
            cover_letter={
                "recipient_company": "Acme",
                "date": "2026-06-25",
                "greeting": "Hi",
                "paragraphs": [],
                "closing": "Best",
                "signature": "Test",
            },
        )
        assert source.strip()


def _fake_compile(source_path: Path, output_path: Path) -> None:
    """Picklable stand-in for ``typst.compile`` in unit tests."""
    output_path.write_bytes(b"%PDF-1.4 fake")


def _failing_compile(source_path: Path, output_path: Path) -> None:
    """Picklable stand-in that raises a compilation error."""
    raise RuntimeError("compile failed")


@pytest.mark.unit
async def test_render_resume_calls_compile(app_settings, tmp_path):
    """render_resume formats the résumé and compiles a PDF."""
    renderer = PDFRenderer(settings=app_settings, output_dir=tmp_path)
    tailored = TailoredResume(
        original_path="r.pdf",
        tailored_text="# Alex\n**Engineer**\n- Built things",
        job_title="Engineer",
        job_company="Acme",
        match_score=0.8,
        semantic_score=0.8,
        skill_score=0.8,
        changes_summary="emphasized Python",
    )
    with patch("job_applicator.documents.pdf_renderer._compile_typst", _fake_compile):
        with patch.object(
            renderer, "_format_resume_with_instructor", new_callable=AsyncMock
        ) as mock_fmt:
            mock_fmt.return_value = FormattedResume(
                name="Alex",
                experience=[
                    FormattedExperienceEntry(
                        title="Engineer",
                        company="Acme",
                        start_date="2020",
                        bullets=["Built things"],
                    )
                ],
            )
            path = await renderer.render_resume(tailored)
            assert path.suffix == ".pdf"
            assert path.parent == tmp_path
            assert path.exists()
            assert re.fullmatch(r"tailored_Acme_Engineer_\d{8}_\d{6}_\d{6}_modern\.pdf", path.name)


@pytest.mark.unit
async def test_render_cover_letter_calls_compile(app_settings, tmp_path):
    """render_cover_letter formats the letter and compiles a PDF."""
    renderer = PDFRenderer(settings=app_settings, output_dir=tmp_path)
    result = CoverLetterResult(
        job_title="Engineer",
        job_company="Acme",
        cover_letter_text="Dear Hiring Manager,\n\nI am excited.\n\nSincerely,\nAlex",
    )
    with patch("job_applicator.documents.pdf_renderer._compile_typst", _fake_compile):
        with patch.object(
            renderer, "_format_cover_letter_with_instructor", new_callable=AsyncMock
        ) as mock_fmt:
            mock_fmt.return_value = FormattedCoverLetter(
                recipient_company="Acme",
                date="2026-06-25",
                greeting="Dear Hiring Manager,",
                paragraphs=["I am excited."],
                closing="Sincerely",
                signature="Alex Rivera",
            )
            path = await renderer.render_cover_letter(result)
            assert path.suffix == ".pdf"
            assert path.parent == tmp_path
            assert path.exists()
            assert re.fullmatch(
                r"cover_letter_Acme_Engineer_\d{8}_\d{6}_\d{6}_modern\.pdf", path.name
            )


@pytest.mark.unit
async def test_render_resume_propagates_format_error(app_settings, tmp_path):
    """A failure in the LLM formatter is wrapped as a PDFRenderError."""
    renderer = PDFRenderer(settings=app_settings, output_dir=tmp_path)
    tailored = TailoredResume(
        original_path="r.pdf",
        tailored_text="text",
        job_title="Engineer",
        job_company="Acme",
        match_score=0.8,
        semantic_score=0.8,
        skill_score=0.8,
        changes_summary="emphasized Python",
    )
    fake_client = MagicMock()
    fake_client.create = AsyncMock(side_effect=RuntimeError("model down"))
    with patch.object(renderer, "_get_client", return_value=fake_client):
        with pytest.raises(PDFRenderError):
            await renderer.render_resume(tailored)


@pytest.mark.unit
async def test_render_resume_propagates_compile_error(app_settings, tmp_path):
    """A Typst compilation failure is wrapped as a PDFRenderError."""
    renderer = PDFRenderer(settings=app_settings, output_dir=tmp_path)
    tailored = TailoredResume(
        original_path="r.pdf",
        tailored_text="text",
        job_title="Engineer",
        job_company="Acme",
        match_score=0.8,
        semantic_score=0.8,
        skill_score=0.8,
        changes_summary="emphasized Python",
    )

    with patch("job_applicator.documents.pdf_renderer._compile_typst", _failing_compile):
        with patch.object(
            renderer, "_format_resume_with_instructor", new_callable=AsyncMock
        ) as mock_fmt:
            mock_fmt.return_value = FormattedResume(
                name="Alex",
                experience=[
                    FormattedExperienceEntry(
                        title="Engineer", company="Acme", start_date="2020", bullets=["Built"]
                    )
                ],
            )
            with pytest.raises(PDFRenderError):
                await renderer.render_resume(tailored)


@pytest.mark.unit
async def test_render_cover_letter_propagates_format_error(app_settings, tmp_path):
    """A failure in the cover-letter formatter is wrapped as a PDFRenderError."""
    renderer = PDFRenderer(settings=app_settings, output_dir=tmp_path)
    result = CoverLetterResult(
        job_title="Engineer",
        job_company="Acme",
        cover_letter_text="text",
    )
    fake_client = MagicMock()
    fake_client.create = AsyncMock(side_effect=RuntimeError("model down"))
    with patch.object(renderer, "_get_client", return_value=fake_client):
        with pytest.raises(PDFRenderError):
            await renderer.render_cover_letter(result)


@pytest.mark.unit
async def test_render_cover_letter_propagates_compile_error(app_settings, tmp_path):
    """A Typst compilation failure is wrapped as a PDFRenderError."""
    renderer = PDFRenderer(settings=app_settings, output_dir=tmp_path)
    result = CoverLetterResult(
        job_title="Engineer",
        job_company="Acme",
        cover_letter_text="Dear Hiring Manager,\n\nI am excited.\n\nSincerely,\nAlex",
    )

    with patch("job_applicator.documents.pdf_renderer._compile_typst", _failing_compile):
        with patch.object(
            renderer, "_format_cover_letter_with_instructor", new_callable=AsyncMock
        ) as mock_fmt:
            mock_fmt.return_value = FormattedCoverLetter(
                recipient_company="Acme",
                date="2026-06-25",
                greeting="Dear Hiring Manager,",
                paragraphs=["I am excited."],
                closing="Sincerely",
                signature="Alex Rivera",
            )
            with pytest.raises(PDFRenderError):
                await renderer.render_cover_letter(result)


@pytest.mark.unit
async def test_pdf_renderer_get_client_caches(app_settings, tmp_path):
    """The instructor client is lazily constructed and cached."""
    renderer = PDFRenderer(settings=app_settings, output_dir=tmp_path)
    assert renderer._client is None
    fake_client = MagicMock()
    with patch("job_applicator.documents.pdf_renderer.quiet_litellm") as mock_quiet:
        with patch("instructor.from_litellm", return_value=fake_client) as mock_from:
            client = await renderer._get_client()
            assert client is fake_client
            assert renderer._client is fake_client
            mock_quiet.assert_called_once()
            mock_from.assert_called_once()
            # Second call returns cached client without re-importing.
            client2 = await renderer._get_client()
            assert client2 is fake_client
            mock_from.assert_called_once()


@pytest.mark.unit
def test_compile_typst_missing_package_message(tmp_path: Path) -> None:
    """A missing typst package yields the exact install hint message."""
    source_path = tmp_path / "source.typ"
    output_path = tmp_path / "out.pdf"
    source_path.write_text("")

    real_import = builtins.__import__

    def _fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "typst":
            raise ImportError("No module named 'typst'")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_fake_import):
        with pytest.raises(
            PDFRenderError,
            match=r"typst package not installed; run: pip install 'job-applicator\[pdf\]'",
        ):
            _compile_typst(source_path, output_path)


@pytest.mark.unit
async def test_render_and_compile_template_not_found(app_settings, tmp_path) -> None:
    """A missing template is reported with the list of built-in templates."""
    renderer = PDFRenderer(settings=app_settings, output_dir=tmp_path)
    with pytest.raises(PDFRenderError, match=r"Template not found: missing\.typ"):
        await renderer._render_and_compile(
            "missing.typ",
            {"resume": {"name": "A", "experience": []}},
            tmp_path / "out.pdf",
        )


@pytest.mark.unit
async def test_render_cover_letter_accepts_cover_letter_output(app_settings, tmp_path) -> None:
    """render_cover_letter accepts a raw CoverLetterOutput from the generator."""
    renderer = PDFRenderer(settings=app_settings, output_dir=tmp_path)
    output = CoverLetterOutput(
        cover_letter="Dear Hiring Manager,\n\nI am excited.\n\nSincerely,\nAlex Rivera"
    )
    with patch("job_applicator.documents.pdf_renderer._compile_typst", _fake_compile):
        with patch.object(
            renderer, "_format_cover_letter_with_instructor", new_callable=AsyncMock
        ) as mock_fmt:
            mock_fmt.return_value = FormattedCoverLetter(
                recipient_company="Acme",
                date="2026-06-25",
                greeting="Dear Hiring Manager,",
                paragraphs=["I am excited."],
                closing="Sincerely",
                signature="Alex Rivera",
            )
            path = await renderer.render_cover_letter(output)
            assert path.suffix == ".pdf"
            assert path.parent == tmp_path
            assert path.exists()
            passed = mock_fmt.call_args[0][0]
            assert isinstance(passed, CoverLetterResult)
            assert passed.cover_letter_text == output.cover_letter
            assert passed.job_title == ""
            assert passed.job_company == ""


@pytest.mark.unit
async def test_render_resume_uses_explicit_output_path(app_settings, tmp_path) -> None:
    """render_resume writes to the supplied output_path when provided."""
    renderer = PDFRenderer(settings=app_settings, output_dir=tmp_path)
    tailored = TailoredResume(
        original_path="r.pdf",
        tailored_text="text",
        job_title="Engineer",
        job_company="Acme",
        match_score=0.8,
        semantic_score=0.8,
        skill_score=0.8,
        changes_summary="emphasized Python",
    )
    explicit = tmp_path / "custom" / "resume.pdf"
    with patch("job_applicator.documents.pdf_renderer._compile_typst", _fake_compile):
        with patch.object(
            renderer, "_format_resume_with_instructor", new_callable=AsyncMock
        ) as mock_fmt:
            mock_fmt.return_value = FormattedResume(
                name="Alex",
                experience=[
                    FormattedExperienceEntry(
                        title="Engineer", company="Acme", start_date="2020", bullets=["Built"]
                    )
                ],
            )
            path = await renderer.render_resume(tailored, output_path=explicit)
            assert path == explicit
            assert path.exists()


@pytest.mark.unit
async def test_render_cover_letter_uses_explicit_output_path(app_settings, tmp_path) -> None:
    """render_cover_letter writes to the supplied output_path when provided."""
    renderer = PDFRenderer(settings=app_settings, output_dir=tmp_path)
    result = CoverLetterResult(
        job_title="Engineer",
        job_company="Acme",
        cover_letter_text="Dear Hiring Manager,\n\nI am excited.\n\nSincerely,\nAlex",
    )
    explicit = tmp_path / "custom" / "cl.pdf"
    with patch("job_applicator.documents.pdf_renderer._compile_typst", _fake_compile):
        with patch.object(
            renderer, "_format_cover_letter_with_instructor", new_callable=AsyncMock
        ) as mock_fmt:
            mock_fmt.return_value = FormattedCoverLetter(
                recipient_company="Acme",
                date="2026-06-25",
                greeting="Dear Hiring Manager,",
                paragraphs=["I am excited."],
                closing="Sincerely",
                signature="Alex Rivera",
            )
            path = await renderer.render_cover_letter(result, output_path=explicit)
            assert path == explicit
            assert path.exists()


@pytest.mark.unit
async def test_render_and_compile_preserves_temp_source_on_failure(app_settings, tmp_path) -> None:
    """A compilation failure leaves the temporary Typst source in place for debugging."""
    renderer = PDFRenderer(settings=app_settings, output_dir=tmp_path)
    context = {
        "resume": {"name": "A", "experience": []},
        "cover_letter": {
            "recipient_company": "Acme",
            "date": "2026-06-25",
            "greeting": "Hi",
            "paragraphs": [],
            "closing": "Best",
            "signature": "A",
        },
    }

    with patch("job_applicator.documents.pdf_renderer._compile_typst", _failing_compile):
        with pytest.raises(PDFRenderError, match="compile failed") as exc_info:
            await renderer._render_and_compile("cv/modern.typ", context, tmp_path / "out.pdf")
    tmp_files = [p for p in tmp_path.iterdir() if p.name.startswith("_tmp_")]
    assert len(tmp_files) == 1
    assert tmp_files[0].read_text(encoding="utf-8")
    assert exc_info.value.context.get("source") == str(tmp_files[0])


@pytest.mark.unit
async def test_render_and_compile_cleans_temp_source_on_success(app_settings, tmp_path) -> None:
    """The temporary Typst source is removed after successful compilation."""
    renderer = PDFRenderer(settings=app_settings, output_dir=tmp_path)
    context = {
        "resume": {"name": "A", "experience": []},
        "cover_letter": {
            "recipient_company": "Acme",
            "date": "2026-06-25",
            "greeting": "Hi",
            "paragraphs": [],
            "closing": "Best",
            "signature": "A",
        },
    }
    with patch("job_applicator.documents.pdf_renderer._compile_typst", _fake_compile):
        await renderer._render_and_compile("cv/modern.typ", context, tmp_path / "out.pdf")
    tmp_files = [p for p in tmp_path.iterdir() if p.name.startswith("_tmp_")]
    assert not tmp_files
