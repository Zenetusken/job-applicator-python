#!/usr/bin/env python3
"""Live UI workflow tests for all Tier 2 items.

Tests each Tier 2 change against the real environment:
- vLLM at localhost:8000
- mxbai-embed-large-v1 on GPU
- Real CLI commands
- Real file I/O
"""

from __future__ import annotations

import asyncio
import inspect
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

PASS = "[bold green]PASS[/]"
FAIL = "[bold red]FAIL[/]"
SKIP = "[bold yellow]SKIP[/]"
results: list[tuple[str, str, str]] = []


def report(item: str, test: str, passed: bool, detail: str = ""):
    status = PASS if passed else FAIL
    results.append((item, test, status))
    icon = "✓" if passed else "✗"
    console.print(f"  {icon} {test}" + (f" — {detail}" if detail else ""))


def skip(item: str, test: str, reason: str):
    results.append((item, test, SKIP))
    console.print(f"  ⊘ {test} — SKIP: {reason}")


# ── TIER 2 ITEM A9: Few-shot Examples in Prompts ────────────────────────────

def test_a9_few_shot_examples():
    """Test that system prompts contain few-shot examples."""
    console.print(Panel("[bold]A9: Few-shot Examples in Prompts[/]", style="cyan"))

    from job_applicator.documents.resume_tailor import ResumeTailor
    from job_applicator.documents.cover_letter import CoverLetterGenerator
    from job_applicator.config import LLMConfig

    # Check tailor system prompt
    from job_applicator.documents import resume_tailor as rt_mod
    tailor_prompt = rt_mod.TAILOR_SYSTEM_PROMPT

    report("A9", "tailor prompt has BEFORE/AFTER summary example",
           "BEFORE summary" in tailor_prompt and "AFTER summary" in tailor_prompt)
    report("A9", "tailor prompt has BEFORE/AFTER bullet example",
           "BEFORE bullet" in tailor_prompt and "AFTER bullet" in tailor_prompt)
    report("A9", "tailor prompt has EXAMPLES section",
           "EXAMPLES" in tailor_prompt or "EXAMPLE" in tailor_prompt)

    # Check cover letter system prompt
    from job_applicator.documents import cover_letter as cl_mod
    cl_prompt = cl_mod.SYSTEM_PROMPT

    report("A9", "cover letter prompt has opening example",
           "opening" in cl_prompt.lower() and "example" in cl_prompt.lower())
    report("A9", "cover letter prompt has closing example",
           "closing" in cl_prompt.lower() and "example" in cl_prompt.lower())

    console.print(f"    tailor prompt length: {len(tailor_prompt)} chars")
    console.print(f"    cover letter prompt length: {len(cl_prompt)} chars")


# ── TIER 2 ITEM A8: Per-task Temperature Tuning ─────────────────────────────

def test_a8_temperature_tuning():
    """Test that temperature varies by task."""
    console.print(Panel("[bold]A8: Per-task Temperature Tuning[/]", style="cyan"))

    from job_applicator.documents import resume_tailor as rt_mod
    from job_applicator.documents import style_analyzer as sa_mod
    from job_applicator.documents import cover_letter as cl_mod

    # Check tailor temperature via source inspection
    rt_source = inspect.getsource(rt_mod)
    sa_source = inspect.getsource(sa_mod)

    # Check for temperature values in resume_tailor
    report("A9/temp", "tailor uses temperature 0.4 (default param)",
           "temperature: float = 0.4" in rt_source or "temperature=0.4" in rt_source)
    report("A9/temp", "refine uses temperature 0.3",
           "temperature=0.3" in rt_source)
    report("A9/temp", "summarize uses temperature 0.2",
           "temperature=0.2" in rt_source)

    # Check style analyzer temperature
    report("A9/temp", "style analyzer uses temperature 0.1",
           "temperature=0.1" in sa_source or "temperature = 0.1" in sa_source)

    # Verify they're different (per-task tuning)
    temps_found = set()
    for val in ["0.1", "0.2", "0.3", "0.4"]:
        if f"temperature={val}" in rt_source or f"temperature: float = {val}" in rt_source:
            temps_found.add(val)
        if f"temperature={val}" in sa_source or f"temperature: float = {val}" in sa_source:
            temps_found.add(val)

    report("A9/temp", "multiple distinct temperatures used",
           len(temps_found) >= 3, f"found={sorted(temps_found)}")


# ── TIER 2 ITEM C2: Parallel Cover Letter Generation ────────────────────────

async def test_c2_parallel_cover_letters():
    """Test that cover letters can be generated in parallel."""
    console.print(Panel("[bold]C2: Parallel Cover Letter Generation[/]", style="cyan"))

    from job_applicator.config import LLMConfig
    from job_applicator.documents.cover_letter import CoverLetterGenerator
    from job_applicator.models import JobBoard, JobListing, ResumeData, UserProfile

    config = LLMConfig()
    generator = CoverLetterGenerator(config)

    resume = ResumeData(
        raw_text="John Doe\njohn@example.com\nSkills: Python, FastAPI",
        name="John Doe",
        email="john@example.com",
        skills=["Python", "FastAPI"],
        summary="Python developer",
    )
    user = UserProfile(first_name="John", last_name="Doe", email="john@example.com", phone="555-0123")

    jobs = [
        JobListing(title=f"Developer {i}", company=f"Company {i}",
                   url=f"https://example.com/{i}", description="Python role",
                   location="Remote", board=JobBoard.LINKEDIN)
        for i in range(3)
    ]

    # Generate sequentially (baseline)
    start_seq = time.monotonic()
    seq_results = []
    for job in jobs[:2]:
        try:
            letter = await generator.generate(job, user, resume)
            seq_results.append(letter)
        except Exception as e:
            seq_results.append(None)
    seq_time = time.monotonic() - start_seq

    # Generate in parallel
    sem = asyncio.Semaphore(3)

    async def gen_one(job: JobListing) -> str | None:
        async with sem:
            try:
                return await generator.generate(job, user, resume)
            except Exception:
                return None

    start_par = time.monotonic()
    par_results = await asyncio.gather(*(gen_one(j) for j in jobs[:2]))
    par_time = time.monotonic() - start_par

    seq_ok = sum(1 for r in seq_results if r and len(r) > 50)
    par_ok = sum(1 for r in par_results if r and len(r) > 50)

    report("C2", "sequential generation works", seq_ok > 0, f"{seq_ok}/2 letters")
    report("C2", "parallel generation works", par_ok > 0, f"{par_ok}/2 letters")
    report("C2", "parallel not slower than sequential",
           par_time <= seq_time * 1.5,
           f"seq={seq_time:.1f}s par={par_time:.1f}s")

    # Check that Semaphore(3) is used in cli.py
    cli_source = inspect.getsource(__import__("job_applicator.cli", fromlist=["cli"]))
    report("C2", "cli.py uses asyncio.Semaphore", "Semaphore" in cli_source)
    report("C2", "cli.py uses asyncio.gather", "asyncio.gather" in cli_source)

    console.print(f"    sequential: {seq_time:.1f}s, parallel: {par_time:.1f}s")


# ── TIER 2 ITEM E3: Pre-tailor Match Score ──────────────────────────────────

async def test_e3_pre_tailor_match_score():
    """Test that --min-score gate works before tailoring."""
    console.print(Panel("[bold]E3: Pre-tailor Match Score Gate[/]", style="cyan"))

    from job_applicator.config import EmbeddingConfig
    from job_applicator.embeddings.matching import JobMatcher
    from job_applicator.models import JobBoard, JobListing, ResumeData

    config = EmbeddingConfig()
    matcher = JobMatcher(config)

    resume = ResumeData(
        raw_text="John Doe\nPython developer\nSkills: Python, FastAPI",
        name="John Doe",
        skills=["Python", "FastAPI"],
        summary="Python developer",
    )

    # Good match
    good_job = JobListing(
        title="Senior Python Developer", company="TechCorp",
        url="https://example.com/1", description="Python, FastAPI, Django",
        location="Remote", board=JobBoard.LINKEDIN,
    )

    # Bad match
    bad_job = JobListing(
        title="Marketing Manager", company="AdCo",
        url="https://example.com/2", description="SEO, social media, campaigns",
        location="NYC", board=JobBoard.LINKEDIN,
    )

    good_match = matcher.match_resume_to_job(resume, good_job)
    bad_match = matcher.match_resume_to_job(resume, bad_job)

    report("E3", "good match score computed", good_match.score > 0, f"score={good_match.score:.3f}")
    report("E3", "bad match score computed", bad_match.score >= 0, f"score={bad_match.score:.3f}")
    report("E3", "good match > bad match", good_match.score > bad_match.score)

    # Test threshold logic
    threshold = 0.5
    report("E3", "good match above threshold",
           good_match.score >= threshold,
           f"score={good_match.score:.3f} >= {threshold}")

    # Check CLI has --min-score flag
    import subprocess
    env = {**subprocess.os.environ, "PATH": str(Path(__file__).parent.parent / ".venv" / "bin") + ":" + subprocess.os.environ.get("PATH", "")}
    result = subprocess.run(
        ["job-applicator", "tailor", "--help"],
        capture_output=True, text=True, timeout=10, env=env,
    )
    report("E3", "tailor has --min-score flag", "--min-score" in result.stdout)

    # Check source has the gate logic
    cli_source = inspect.getsource(__import__("job_applicator.cli", fromlist=["cli"]))
    report("E3", "cli has pre-tailor match gate",
           "pre_match" in cli_source or "min_score" in cli_source)


# ── TIER 2 ITEM C3: Wrap Blocking File I/O ──────────────────────────────────

def test_c3_async_file_io():
    """Test that blocking file I/O is wrapped with asyncio.to_thread."""
    console.print(Panel("[bold]C3: Async File I/O Wrapping[/]", style="cyan"))

    import job_applicator.cli as cli_mod
    import job_applicator.documents.cover_letter as cl_mod

    cli_source = inspect.getsource(cli_mod)
    cl_source = inspect.getsource(cl_mod)

    # Count asyncio.to_thread usages
    cli_to_thread = cli_source.count("asyncio.to_thread")
    cl_to_thread = cl_source.count("asyncio.to_thread")

    report("C3", "cli.py uses asyncio.to_thread", cli_to_thread > 0,
           f"{cli_to_thread} occurrences")
    report("C3", "cover_letter.py uses asyncio.to_thread", cl_to_thread > 0,
           f"{cl_to_thread} occurrences")

    # Check specific patterns
    report("C3", "wraps write_text", "write_text" in cli_source and "to_thread" in cli_source)
    report("C3", "wraps mkdir", "mkdir" in cli_source and "to_thread" in cli_source)
    report("C3", "wraps path.exists", "exists" in cl_source and "to_thread" in cl_source)

    # Verify it's used in async context (function is async def)
    report("C3", "file writes are in async functions",
           "async def" in cli_source and "await asyncio.to_thread" in cli_source)


# ── TIER 2 ITEM E2: Batch Mode (NOT IMPLEMENTED) ────────────────────────────

def test_e2_batch_mode():
    """Check if batch mode exists."""
    console.print(Panel("[bold]E2: Batch Mode[/]", style="cyan"))

    import subprocess
    env = {**subprocess.os.environ, "PATH": str(Path(__file__).parent.parent / ".venv" / "bin") + ":" + subprocess.os.environ.get("PATH", "")}

    # Check if there's a batch command
    result = subprocess.run(
        ["job-applicator", "--help"],
        capture_output=True, text=True, timeout=10, env=env,
    )
    has_batch = "batch" in result.stdout.lower()
    report("E2", "batch command exists", has_batch,
           "NOT FOUND" if not has_batch else "found")

    # Check if match accepts --jobs-file (partial batch)
    result2 = subprocess.run(
        ["job-applicator", "match", "--help"],
        capture_output=True, text=True, timeout=10, env=env,
    )
    report("E2", "match has --jobs-file for batch input",
           "--jobs-file" in result2.stdout)

    skip("E2", "full pipeline batch mode", "Not implemented — only match supports --jobs-file")


# ── TIER 2 ITEM D2: OCR Fallback (NOT IMPLEMENTED) ──────────────────────────

def test_d2_ocr_fallback():
    """Check if OCR fallback exists for scanned PDFs."""
    console.print(Panel("[bold]D2: OCR Fallback for Scanned PDFs[/]", style="cyan"))

    from job_applicator.documents import resume as resume_mod
    source = inspect.getsource(resume_mod)

    has_ocr = "ocr" in source.lower() or "tesseract" in source.lower() or "pytesseract" in source.lower()
    report("D2", "OCR fallback in resume.py", has_ocr,
           "NOT FOUND" if not has_ocr else "found")

    # Check for PyMuPDF (non-OCR fallback)
    has_fitz = "fitz" in source
    report("D2", "PyMuPDF fallback exists", has_fitz)

    skip("D2", "scanned PDF OCR", "Not implemented — only text extraction via pdftotext + PyMuPDF")


# ── TIER 2 ITEM D4: ATS Compatibility Checking (NOT IMPLEMENTED) ────────────

def test_d4_ats_compatibility():
    """Check if ATS compatibility checking exists."""
    console.print(Panel("[bold]D4: ATS Compatibility Checking[/]", style="cyan"))

    import job_applicator.cli as cli_mod
    import job_applicator.documents.resume_tailor as rt_mod

    cli_source = inspect.getsource(cli_mod)
    rt_source = inspect.getsource(rt_mod)

    has_ats = "ats" in cli_source.lower() and "applicant" in cli_source.lower()
    report("D4", "ATS checking in CLI", has_ats,
           "NOT FOUND" if not has_ats else "found")

    skip("D4", "ATS compatibility scoring", "Not implemented — no ATS-related code")


# ── LIVE INTEGRATION: Full Pipeline with Tier 2 Features ────────────────────

async def test_live_pipeline_tier2():
    """Run a live pipeline test exercising Tier 2 features together."""
    console.print(Panel("[bold]LIVE: Full Pipeline with Tier 2 Features[/]", style="cyan"))

    from job_applicator.config import LLMConfig, EmbeddingConfig
    from job_applicator.documents.cover_letter import CoverLetterGenerator
    from job_applicator.documents.resume_tailor import ResumeTailor
    from job_applicator.embeddings.matching import JobMatcher
    from job_applicator.models import JobBoard, JobListing, ResumeData, UserProfile

    llm_config = LLMConfig()
    embed_config = EmbeddingConfig()

    resume = ResumeData(
        raw_text="Alice Smith\nalice@example.com\nSkills: Python, React, AWS\nSummary: Full-stack developer",
        name="Alice Smith",
        email="alice@example.com",
        skills=["Python", "React", "AWS"],
        summary="Full-stack developer",
    )
    user = UserProfile(first_name="Alice", last_name="Smith", email="alice@example.com", phone="555-4567")
    job = JobListing(
        title="Senior Full-Stack Engineer", company="TechStartup",
        url="https://example.com/777", description="Python, React, AWS, Docker",
        location="Remote", board=JobBoard.LINKEDIN,
    )

    # E3: Pre-tailor match score
    matcher = JobMatcher(embed_config)
    match = matcher.match_resume_to_job(resume, job)
    report("LIVE", "E3: pre-tailor match score", match.score > 0, f"score={match.score:.3f}")

    # A9: Tailor with few-shot examples (LLM call)
    tailor = ResumeTailor(llm_config)
    result = await tailor.tailor(resume=resume, job=job, user_instructions="Emphasize full-stack skills.")

    report("LIVE", "A9: tailor produces output", len(result.tailored_text) > 100)
    report("LIVE", "A8: scores populated", result.semantic_score > 0 and result.skill_score > 0)

    # C2: Parallel cover letter generation
    gen = CoverLetterGenerator(llm_config)
    jobs_batch = [
        JobListing(title=f"Engineer {i}", company=f"Co{i}",
                   url=f"https://example.com/b{i}", description="Python",
                   location="Remote", board=JobBoard.LINKEDIN)
        for i in range(3)
    ]

    sem = asyncio.Semaphore(3)

    async def gen_one(j: JobListing) -> str | None:
        async with sem:
            try:
                return await gen.generate(j, user, resume)
            except Exception:
                return None

    start = time.monotonic()
    letters = await asyncio.gather(*(gen_one(j) for j in jobs_batch))
    elapsed = time.monotonic() - start

    ok = sum(1 for l in letters if l and len(l) > 50)
    report("LIVE", "C2: parallel cover letters", ok >= 2, f"{ok}/3 in {elapsed:.1f}s")

    console.print(f"    Tailored resume: {len(result.tailored_text)} chars")
    console.print(f"    Cover letters: {ok}/3 generated, {elapsed:.1f}s total")


# ── SUMMARY ──────────────────────────────────────────────────────────────────

def print_summary():
    table = Table(title="Tier 2 Live Test Results", show_lines=True)
    table.add_column("Item", style="cyan")
    table.add_column("Test", style="white")
    table.add_column("Status", justify="center")

    passed = sum(1 for _, _, s in results if s == PASS)
    failed = sum(1 for _, _, s in results if s == FAIL)
    skipped = sum(1 for _, _, s in results if s == SKIP)

    for item, test, status in results:
        table.add_row(item, test, status)

    console.print(table)
    console.print(f"\n[bold]Total: {passed} passed, {failed} failed, {skipped} skipped[/]")


# ── MAIN ─────────────────────────────────────────────────────────────────────

async def main():
    console.print(Panel("[bold white]TIER 2 LIVE UI WORKFLOW TESTS[/]", style="blue", expand=False))
    console.print("Environment: vLLM at localhost:8000, GPU available, Python 3.12\n")

    # Non-LLM tests (fast)
    test_a9_few_shot_examples()
    test_a8_temperature_tuning()
    test_e2_batch_mode()
    test_d2_ocr_fallback()
    test_d4_ats_compatibility()

    # Async tests with LLM
    await test_c2_parallel_cover_letters()
    await test_e3_pre_tailor_match_score()
    test_c3_async_file_io()

    # Full pipeline integration
    await test_live_pipeline_tier2()

    print_summary()


if __name__ == "__main__":
    asyncio.run(main())
