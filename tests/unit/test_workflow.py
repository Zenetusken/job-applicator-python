"""Tests for workflow logic: sessions, LLM retry, cover letter, diff, audit."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console

from job_applicator.models import (
    CoverLetterResult,
    CoverLetterSession,
    DateAuditResult,
    DateEntry,
    JobListing,
    ResumeData,
    TailoredResume,
    TailorSession,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tailored_resume(text: str = "tailored", attempt: int = 1) -> TailoredResume:
    return TailoredResume(
        original_path="resume.pdf",
        tailored_text=text,
        job_title="Dev",
        job_company="Acme",
        job_url="https://example.com/1",
        match_score=0.8,
        semantic_score=0.9,
        skill_score=0.7,
        changes_summary="updated",
        attempt=attempt,
    )


def _make_cover_letter_result(
    text: str = "Dear Hiring Manager",
    attempt: int = 1,
) -> CoverLetterResult:
    return CoverLetterResult(
        job_title="Dev",
        job_company="Acme",
        job_url="https://example.com/1",
        cover_letter_text=text,
        attempt=attempt,
    )


# ---------------------------------------------------------------------------
# TailorSession
# ---------------------------------------------------------------------------


class TestTailorSession:
    def test_create_session(self) -> None:
        session = TailorSession("original text", "Dev", "Acme")
        assert session.original_text == "original text"
        assert session.job_title == "Dev"
        assert session.job_company == "Acme"
        assert session.attempts == []
        assert session.current_index == -1

    def test_add_attempt(self) -> None:
        session = TailorSession("text", "Dev", "Acme")
        resume = _make_tailored_resume()
        session.add_attempt(resume)
        assert len(session.attempts) == 1
        assert session.current_index == 0

    def test_add_multiple_attempts(self) -> None:
        session = TailorSession("text", "Dev", "Acme")
        session.add_attempt(_make_tailored_resume("v1", attempt=1))
        session.add_attempt(_make_tailored_resume("v2", attempt=2))
        session.add_attempt(_make_tailored_resume("v3", attempt=3))
        assert len(session.attempts) == 3
        assert session.current_index == 2

    def test_current_returns_latest(self) -> None:
        session = TailorSession("text", "Dev", "Acme")
        session.add_attempt(_make_tailored_resume("v1", attempt=1))
        session.add_attempt(_make_tailored_resume("v2", attempt=2))
        assert session.current.tailored_text == "v2"

    def test_current_raises_when_empty(self) -> None:
        session = TailorSession("text", "Dev", "Acme")
        with pytest.raises(IndexError, match="No attempts"):
            _ = session.current

    def test_select_changes_current(self) -> None:
        session = TailorSession("text", "Dev", "Acme")
        session.add_attempt(_make_tailored_resume("v1", attempt=1))
        session.add_attempt(_make_tailored_resume("v2", attempt=2))
        session.select(0)
        assert session.current.tailored_text == "v1"
        assert session.current_index == 0

    def test_select_out_of_range_raises(self) -> None:
        session = TailorSession("text", "Dev", "Acme")
        session.add_attempt(_make_tailored_resume())
        with pytest.raises(IndexError, match="out of range"):
            session.select(5)

    def test_select_negative_index_raises(self) -> None:
        session = TailorSession("text", "Dev", "Acme")
        session.add_attempt(_make_tailored_resume())
        with pytest.raises(IndexError, match="out of range"):
            session.select(-1)

    def test_user_modifications_tracking(self) -> None:
        session = TailorSession("text", "Dev", "Acme")
        resume = _make_tailored_resume()
        resume.user_modifications = "add more detail about Kubernetes"
        session.add_attempt(resume)
        assert session.current.user_modifications == "add more detail about Kubernetes"


# ---------------------------------------------------------------------------
# CoverLetterSession
# ---------------------------------------------------------------------------


class TestCoverLetterSession:
    def test_create_session(self) -> None:
        session = CoverLetterSession("Dev", "Acme")
        assert session.job_title == "Dev"
        assert session.job_company == "Acme"
        assert session.attempts == []
        assert session.current_index == -1

    def test_add_attempt(self) -> None:
        session = CoverLetterSession("Dev", "Acme")
        session.add_attempt(_make_cover_letter_result())
        assert len(session.attempts) == 1
        assert session.current_index == 0

    def test_add_multiple_attempts(self) -> None:
        session = CoverLetterSession("Dev", "Acme")
        session.add_attempt(_make_cover_letter_result("v1", attempt=1))
        session.add_attempt(_make_cover_letter_result("v2", attempt=2))
        session.add_attempt(_make_cover_letter_result("v3", attempt=3))
        assert len(session.attempts) == 3
        assert session.current_index == 2

    def test_current_returns_latest(self) -> None:
        session = CoverLetterSession("Dev", "Acme")
        session.add_attempt(_make_cover_letter_result("v1", attempt=1))
        session.add_attempt(_make_cover_letter_result("v2", attempt=2))
        assert session.current.cover_letter_text == "v2"

    def test_current_raises_when_empty(self) -> None:
        session = CoverLetterSession("Dev", "Acme")
        with pytest.raises(IndexError, match="No attempts"):
            _ = session.current

    def test_select_changes_current(self) -> None:
        session = CoverLetterSession("Dev", "Acme")
        session.add_attempt(_make_cover_letter_result("v1", attempt=1))
        session.add_attempt(_make_cover_letter_result("v2", attempt=2))
        session.select(0)
        assert session.current.cover_letter_text == "v1"
        assert session.current_index == 0

    def test_select_out_of_range_raises(self) -> None:
        session = CoverLetterSession("Dev", "Acme")
        session.add_attempt(_make_cover_letter_result())
        with pytest.raises(IndexError, match="out of range"):
            session.select(5)

    def test_select_negative_index_raises(self) -> None:
        session = CoverLetterSession("Dev", "Acme")
        session.add_attempt(_make_cover_letter_result())
        with pytest.raises(IndexError, match="out of range"):
            session.select(-1)

    def test_user_modifications_tracking(self) -> None:
        session = CoverLetterSession("Dev", "Acme")
        result = _make_cover_letter_result()
        result.user_modifications = "make it more enthusiastic"
        session.add_attempt(result)
        assert session.current.user_modifications == "make it more enthusiastic"


# ---------------------------------------------------------------------------
# _llm_with_retry
# ---------------------------------------------------------------------------


class TestLlmWithRetry:
    @staticmethod
    def _mock_console() -> MagicMock:
        console = MagicMock(spec=Console)
        console.status.return_value.__enter__ = MagicMock()
        console.status.return_value.__exit__ = MagicMock(return_value=False)
        return console

    @pytest.mark.asyncio
    async def test_success_first_try(self) -> None:
        from job_applicator.cli import _llm_with_retry

        console = self._mock_console()
        op = AsyncMock(return_value="result")

        result = await _llm_with_retry(console, op)
        assert result == "result"
        assert op.call_count == 1

    @pytest.mark.asyncio
    async def test_fail_then_succeed(self) -> None:
        from job_applicator.cli import _llm_with_retry

        console = self._mock_console()
        console.input.return_value = "R"
        op = AsyncMock(side_effect=[RuntimeError("boom"), "ok"])

        result = await _llm_with_retry(console, op)
        assert result == "ok"
        assert op.call_count == 2

    @pytest.mark.asyncio
    async def test_always_fail_user_quits(self) -> None:
        from job_applicator.cli import _llm_with_retry

        console = self._mock_console()
        console.input.return_value = "Q"
        op = AsyncMock(side_effect=RuntimeError("boom"))

        result = await _llm_with_retry(console, op)
        assert result is None
        assert op.call_count == 1


# ---------------------------------------------------------------------------
# _generate_cover_letter
# ---------------------------------------------------------------------------


class TestGenerateCoverLetter:
    @staticmethod
    def _mock_console() -> MagicMock:
        console = MagicMock(spec=Console)
        console.status.return_value.__enter__ = MagicMock()
        console.status.return_value.__exit__ = MagicMock(return_value=False)
        return console

    @pytest.mark.asyncio
    async def test_success_returns_cover_letter_result(self) -> None:
        from job_applicator.cli import _generate_cover_letter

        console = self._mock_console()
        settings = MagicMock()
        settings.llm.model = "test-model"
        job = MagicMock()
        job.title = "Dev"
        job.company = "Acme"
        job.url = "https://example.com/1"
        resume_data = MagicMock()
        session = CoverLetterSession("Dev", "Acme")

        with (
            patch("job_applicator.documents.cover_letter.CoverLetterGenerator") as mock_gen_cls,
            patch("job_applicator.cli._load_user_profile") as mock_profile,
            patch("job_applicator.cli.CoverLetterResult", CoverLetterResult, create=True),
        ):
            mock_gen = mock_gen_cls.return_value
            mock_gen.generate = AsyncMock(return_value="Dear Hiring Manager")
            mock_profile.return_value = MagicMock()
            result = await _generate_cover_letter(
                console, settings, job, resume_data, None, "formal", "tailored text", session
            )

        assert result is not None
        assert result.cover_letter_text == "Dear Hiring Manager"
        assert result.attempt == 1
        assert len(session.attempts) == 1

    @pytest.mark.asyncio
    async def test_llm_failure_returns_none(self) -> None:
        from job_applicator.cli import _generate_cover_letter

        console = self._mock_console()
        settings = MagicMock()
        settings.llm.model = "test-model"
        job = MagicMock()
        job.title = "Dev"
        job.company = "Acme"
        job.url = "https://example.com/1"
        resume_data = MagicMock()
        session = CoverLetterSession("Dev", "Acme")

        with (
            patch("job_applicator.documents.cover_letter.CoverLetterGenerator") as mock_gen_cls,
            patch("job_applicator.cli._load_user_profile") as mock_profile,
            patch("job_applicator.cli.CoverLetterResult", CoverLetterResult, create=True),
        ):
            mock_gen = mock_gen_cls.return_value
            mock_gen.generate = AsyncMock(side_effect=RuntimeError("LLM down"))
            mock_profile.return_value = MagicMock()
            result = await _generate_cover_letter(
                console, settings, job, resume_data, None, "formal", "tailored text", session
            )

        assert result is None
        assert len(session.attempts) == 0

    @pytest.mark.asyncio
    async def test_passes_tone_and_tailored_resume(self) -> None:
        from job_applicator.cli import _generate_cover_letter

        console = self._mock_console()
        settings = MagicMock()
        settings.llm.model = "test-model"
        job = MagicMock()
        job.title = "Dev"
        job.company = "Acme"
        job.url = "https://example.com/1"
        resume_data = MagicMock()
        session = CoverLetterSession("Dev", "Acme")

        with (
            patch("job_applicator.documents.cover_letter.CoverLetterGenerator") as mock_gen_cls,
            patch("job_applicator.cli._load_user_profile") as mock_profile,
            patch("job_applicator.cli.CoverLetterResult", CoverLetterResult, create=True),
        ):
            mock_gen = mock_gen_cls.return_value
            mock_gen.generate = AsyncMock(return_value="letter")
            user_profile = MagicMock()
            mock_profile.return_value = user_profile
            await _generate_cover_letter(
                console, settings, job, resume_data, None, "casual", "my tailored resume", session
            )
            mock_gen.generate.assert_called_once_with(
                job,
                user_profile,
                resume_data,
                style_guide=None,
                tone_section="casual",
                tailored_resume_text="my tailored resume",
            )


# ---------------------------------------------------------------------------
# _save_cover_letter
# ---------------------------------------------------------------------------


class TestSaveCoverLetter:
    async def test_correct_file_path(self, tmp_path: Path) -> None:
        from job_applicator.cli import _save_cover_letter

        console = MagicMock(spec=Console)
        settings = MagicMock()
        settings.output_dir = str(tmp_path)
        job = MagicMock()
        job.company = "TechCorp"
        job.title = "Senior Dev"
        result = _make_cover_letter_result("My cover letter")

        path = await _save_cover_letter(console, settings, job, result)

        assert path.exists()
        assert path.parent == tmp_path
        assert "cover_letter_TechCorp_Senior_Dev" in path.name
        assert path.suffix == ".txt"
        assert path.read_text(encoding="utf-8") == "My cover letter"

    async def test_meta_json_alongside(self, tmp_path: Path) -> None:
        from job_applicator.cli import _save_cover_letter

        console = MagicMock(spec=Console)
        settings = MagicMock()
        settings.output_dir = str(tmp_path)
        job = MagicMock()
        job.company = "TechCorp"
        job.title = "Senior Dev"
        result = _make_cover_letter_result("letter text")

        path = await _save_cover_letter(console, settings, job, result)
        meta_path = path.with_suffix(".meta.json")

        assert meta_path.exists()

    async def test_meta_json_fields(self, tmp_path: Path) -> None:
        from job_applicator.cli import _save_cover_letter

        console = MagicMock(spec=Console)
        settings = MagicMock()
        settings.output_dir = str(tmp_path)
        job = MagicMock()
        job.company = "TechCorp"
        job.title = "Senior Dev"
        result = _make_cover_letter_result("letter text")

        path = await _save_cover_letter(console, settings, job, result)
        meta_path = path.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        assert meta["job_title"] == "Dev"
        assert meta["job_company"] == "Acme"
        assert meta["cover_letter_text"] == "letter text"
        assert meta["attempt"] == 1
        assert "created_at" in meta

    async def test_output_path_set_on_result(self, tmp_path: Path) -> None:
        from job_applicator.cli import _save_cover_letter

        console = MagicMock(spec=Console)
        settings = MagicMock()
        settings.output_dir = str(tmp_path)
        job = MagicMock()
        job.company = "TechCorp"
        job.title = "Senior Dev"
        result = _make_cover_letter_result("letter")

        path = await _save_cover_letter(console, settings, job, result)
        assert result.output_path == str(path)


# ---------------------------------------------------------------------------
# _refine_cover_letter
# ---------------------------------------------------------------------------


class TestRefineCoverLetter:
    @staticmethod
    def _mock_console() -> MagicMock:
        console = MagicMock(spec=Console)
        console.status.return_value.__enter__ = MagicMock()
        console.status.return_value.__exit__ = MagicMock(return_value=False)
        return console

    @pytest.mark.asyncio
    async def test_success_increments_attempt(self) -> None:
        from job_applicator.cli import _refine_cover_letter

        console = self._mock_console()
        settings = MagicMock()
        settings.llm.model = "test-model"
        settings.llm.api_base = None
        job = MagicMock()
        job.title = "Dev"
        job.company = "Acme"
        job.url = "https://example.com/1"
        result = _make_cover_letter_result("original letter", attempt=1)
        session = CoverLetterSession("Dev", "Acme")
        session.add_attempt(result)

        with (
            patch("job_applicator.documents.cover_letter.CoverLetterGenerator") as mock_gen_cls,
        ):
            mock_gen = mock_gen_cls.return_value
            mock_gen.refine = AsyncMock(return_value="refined letter")
            await _refine_cover_letter(
                console, settings, job, result, "be more formal", session, attempt=1
            )

        assert len(session.attempts) == 2
        assert session.attempts[1].cover_letter_text == "refined letter"
        assert session.attempts[1].attempt == 2
        assert session.attempts[1].user_modifications == "be more formal"

    @pytest.mark.asyncio
    async def test_failure_returns_none(self) -> None:
        from job_applicator.cli import _refine_cover_letter

        console = self._mock_console()
        settings = MagicMock()
        settings.llm.model = "test-model"
        settings.llm.api_base = None
        job = MagicMock()
        job.title = "Dev"
        job.company = "Acme"
        job.url = "https://example.com/1"
        result = _make_cover_letter_result("original letter", attempt=1)
        session = CoverLetterSession("Dev", "Acme")
        session.add_attempt(result)

        with (
            patch("job_applicator.documents.cover_letter.CoverLetterGenerator") as mock_gen_cls,
        ):
            mock_gen = mock_gen_cls.return_value
            mock_gen.refine = AsyncMock(side_effect=RuntimeError("fail"))
            await _refine_cover_letter(
                console, settings, job, result, "be more formal", session, attempt=1
            )

        assert len(session.attempts) == 1


# ---------------------------------------------------------------------------
# render_diff
# ---------------------------------------------------------------------------


class TestRenderDiff:
    def _get_output(self, original: str, tailored: str, max_lines: int = 0) -> str:
        from job_applicator.utils.diff import render_diff

        buf = StringIO()
        console = Console(file=buf, force_terminal=True, no_color=True)
        render_diff(console, original, tailored, max_lines=max_lines)
        return buf.getvalue()

    def test_same_text_no_differences(self) -> None:
        output = self._get_output("same text", "same text")
        assert "No differences found" in output

    def test_additions(self) -> None:
        output = self._get_output("line one", "line one\nline two")
        assert "line two" in output
        assert "+" in output

    def test_removals(self) -> None:
        output = self._get_output("line one\nline two", "line one")
        assert "line two" in output
        assert "-" in output

    def test_max_lines_truncates(self) -> None:
        original = "\n".join(f"line {i}" for i in range(20))
        tailored = "\n".join(f"line {i} modified" for i in range(20))
        output = self._get_output(original, tailored, max_lines=5)
        assert "more lines" in output

    def test_empty_original(self) -> None:
        output = self._get_output("", "new content")
        assert "new content" in output

    def test_empty_tailored(self) -> None:
        output = self._get_output("old content", "")
        assert "old content" in output


# ---------------------------------------------------------------------------
# DateAuditResult
# ---------------------------------------------------------------------------


class TestDateAuditResult:
    def test_creation_defaults(self) -> None:
        result = DateAuditResult()
        assert result.entries == []
        assert result.warnings == []
        assert result.ordering_issues == []
        assert result.staleness_issues == []
        assert result.is_stale is False
        assert result.is_ordered is True
        assert result.latest_date == ""
        assert result.earliest_date == ""

    def test_is_stale_true(self) -> None:
        result = DateAuditResult(is_stale=True, staleness_issues=["Last entry ended 2018"])
        assert result.is_stale is True
        assert len(result.staleness_issues) == 1

    def test_is_ordered_false(self) -> None:
        result = DateAuditResult(
            is_ordered=False,
            ordering_issues=["Job B starts before Job A"],
        )
        assert result.is_ordered is False
        assert len(result.ordering_issues) == 1

    def test_warnings(self) -> None:
        result = DateAuditResult(
            warnings=["Gap detected between 2020-2021", "Overlapping entries found"]
        )
        assert len(result.warnings) == 2
        assert "Gap detected" in result.warnings[0]

    def test_with_entries(self) -> None:
        entry = DateEntry(
            label="Software Engineer",
            section="Experience",
            start="2020-01",
            end="2023-12",
            is_current=False,
        )
        result = DateAuditResult(
            entries=[entry],
            latest_date="2023-12",
            earliest_date="2020-01",
        )
        assert len(result.entries) == 1
        assert result.entries[0].label == "Software Engineer"
        assert result.latest_date == "2023-12"


# ---------------------------------------------------------------------------
# Bug fix: cover letter failure handling
# ---------------------------------------------------------------------------


class TestCoverLetterFailureHandling:
    """Tests for cover letter [R] and [I] failure handling in the workflow loop."""

    @pytest.mark.asyncio
    async def test_retry_in_loop_does_not_check_return_value(self) -> None:
        """[R] Retry in loop calls _generate_cover_letter but ignores None return.

        This is a known bug: if generation fails during retry, the loop continues
        silently showing the old letter instead of showing an error.
        """
        from job_applicator.cli import _cover_letter_workflow

        success_result = CoverLetterResult(
            job_title="Dev",
            job_company="Corp",
            cover_letter_text="Dear Hiring Manager,\nGood letter.",
            attempt=1,
        )

        call_count = 0

        async def fake_gen(
            console: object,
            settings: object,
            job: object,
            resume_data: object,
            style: object,
            tone_section: object,
            tailored_resume_text: object,
            session: CoverLetterSession,
            attempt: int = 1,
        ) -> CoverLetterResult | None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                session.add_attempt(success_result)
                return success_result
            return None

        console = MagicMock(spec=Console)
        console.input = MagicMock(side_effect=["R", "Q"])
        console.print = MagicMock()
        console.status = MagicMock()
        console.status.return_value.__enter__ = MagicMock()
        console.status.return_value.__exit__ = MagicMock(return_value=False)

        mock_save = MagicMock(return_value=Path("output/cover.txt"))

        settings = MagicMock()
        job = MagicMock()
        resume_data = MagicMock()

        with (
            patch("job_applicator.cli._generate_cover_letter", side_effect=fake_gen),
            patch("job_applicator.cli._save_cover_letter", mock_save),
        ):
            result = await _cover_letter_workflow(
                console, settings, job, resume_data, None, None, "tailored text"
            )

        assert result is None
        assert call_count == 2
        all_prints = str(console.print.call_args_list)
        assert "Generation failed" in all_prints

    @pytest.mark.asyncio
    async def test_input_in_loop_shows_error_on_failure(self) -> None:
        """[I] Input should show error when _refine_cover_letter returns None."""
        from job_applicator.cli import _cover_letter_workflow

        success_result = CoverLetterResult(
            job_title="Dev",
            job_company="Corp",
            cover_letter_text="Dear Hiring Manager,\nGood letter.",
            attempt=1,
        )

        async def fake_gen(
            console: object,
            settings: object,
            job: object,
            resume_data: object,
            style: object,
            tone_section: object,
            tailored_resume_text: object,
            session: CoverLetterSession,
            attempt: int = 1,
        ) -> CoverLetterResult | None:
            session.add_attempt(success_result)
            return success_result

        console = MagicMock(spec=Console)
        console.input = MagicMock(side_effect=["I", "make it better", "Q"])
        console.print = MagicMock()
        console.status = MagicMock()
        console.status.return_value.__enter__ = MagicMock()
        console.status.return_value.__exit__ = MagicMock(return_value=False)

        mock_refine = AsyncMock(return_value=None)
        mock_save = MagicMock(return_value=Path("output/cover.txt"))

        settings = MagicMock()
        job = MagicMock()
        resume_data = MagicMock()

        with (
            patch("job_applicator.cli._generate_cover_letter", side_effect=fake_gen),
            patch("job_applicator.cli._refine_cover_letter", mock_refine),
            patch("job_applicator.cli._save_cover_letter", mock_save),
        ):
            result = await _cover_letter_workflow(
                console, settings, job, resume_data, None, None, "tailored text"
            )

        assert result is None
        mock_refine.assert_called_once()
        all_prints = str(console.print.call_args_list)
        assert "Refinement failed" in all_prints


# ---------------------------------------------------------------------------
# Bug fix: stale match scores after refinement
# ---------------------------------------------------------------------------


class TestRefineStaleScores:
    """Tests that refine() preserves score types correctly."""

    @pytest.mark.asyncio
    async def test_refine_preserves_score_types(self) -> None:
        """refine() should return valid float scores."""
        from job_applicator.config import LLMConfig
        from job_applicator.documents.resume_tailor import ResumeTailor

        config = LLMConfig(api_base="http://localhost:8000/v1", model="test")
        tailor = ResumeTailor(config)

        original = ResumeData(
            raw_text="Python developer with 5 years experience",
            skills=["Python", "Django"],
        )
        job = JobListing(
            title="Python Dev",
            company="Corp",
            url="http://example.com",
            description="Python developer needed",
            requirements=["Python", "Django"],
            board="linkedin",
        )
        current = TailoredResume(
            original_path="",
            tailored_text="Tailored Python developer resume",
            job_title="Python Dev",
            job_company="Corp",
            match_score=0.75,
            semantic_score=0.0,
            skill_score=0.0,
            matched_skills=["Python"],
            missing_skills=["Django"],
            changes_summary="Initial tailoring",
            attempt=1,
        )

        with patch.object(tailor, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Refined Python developer resume"
            with patch.object(tailor, "_summarize_changes", new_callable=AsyncMock) as mock_changes:
                mock_changes.return_value = "Refined based on feedback"
                result = await tailor.refine(original, current, "Add more detail", job)

        assert result.attempt == 2
        assert isinstance(result.match_score, float)
        assert isinstance(result.semantic_score, float)
        assert isinstance(result.skill_score, float)
