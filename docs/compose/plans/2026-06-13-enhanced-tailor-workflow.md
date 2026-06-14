# Enhanced Resume Tailor Workflow Implementation Plan

> [!NOTE]
> This document may not reflect the current implementation.
> See the final report for up-to-date state:
> [Final Report](../reports/tier1-tier2-gap-fixes-and-batch-mode.md)

> **For agentic workers:** REQUIRED SUB-SKILL: Use compose:subagent (recommended) or compose:execute to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add diff view, version history, section-level editing, and auto tone detection to the resume tailor workflow, plus production hardening.

**Architecture:** Extend the existing `cli.py:tailor` command loop with new capabilities. Add `ToneDetector` as a new module, `parse_sections()` as a helper in `resume_tailor.py`, and `TailorSession` model for version tracking. All changes follow existing patterns (Pydantic models, async I/O, Rich terminal UX).

**Tech Stack:** Python 3.12+, Rich (terminal UI), difflib (diff generation), Pydantic (models), pytest (tests)

---

### Task 1: ToneDetector Module

**Covers:** [S6]

**Files:**
- Create: `src/job_applicator/documents/tone_detector.py`
- Test: `tests/unit/test_tone_detector.py`

- [ ] **Step 1: Write failing tests for ToneDetector**

```python
# tests/unit/test_tone_detector.py
"""Tests for job posting tone detection."""

from __future__ import annotations

import pytest

from job_applicator.documents.tone_detector import ToneDetector, ToneProfile


class TestToneDetector:
    def test_corporate_tone(self):
        detector = ToneDetector()
        profile = detector.detect(
            title="Senior IT Support Analyst",
            description="Enterprise environment with SLA compliance and stakeholder management. Governance and process improvement required.",
            requirements=["ITIL", "ServiceNow", "Compliance"],
        )
        assert profile.primary == "corporate"
        assert "leveraged" in profile.power_words
        assert "stakeholder" in profile.emphasis

    def test_startup_tone(self):
        detector = ToneDetector()
        profile = detector.detect(
            title="Full Stack Developer",
            description="Fast-paced startup looking for a self-starter who can wear many hats. Agile environment, scrappy team.",
            requirements=["React", "Node.js", "AWS"],
        )
        assert profile.primary == "startup"
        assert "built" in profile.power_words

    def test_technical_tone(self):
        detector = ToneDetector()
        profile = detector.detect(
            title="Backend Engineer",
            description="System design and architecture for microservices. CI/CD pipeline, scalability, distributed systems.",
            requirements=["Python", "Kubernetes", "PostgreSQL"],
        )
        assert profile.primary == "technical"
        assert "architected" in profile.power_words

    def test_creative_tone(self):
        detector = ToneDetector()
        profile = detector.detect(
            title="UX Designer",
            description="Brand storytelling and design thinking. User experience research, visual design, content strategy.",
            requirements=["Figma", "User Research", "Prototyping"],
        )
        assert profile.primary == "creative"
        assert "designed" in profile.power_words

    def test_empty_description_defaults_corporate(self):
        detector = ToneDetector()
        profile = detector.detect(title="Manager", description="", requirements=[])
        assert profile.primary == "corporate"

    def test_format_for_prompt(self):
        detector = ToneDetector()
        profile = detector.detect(
            title="Developer",
            description="Fast-paced agile startup environment.",
            requirements=[],
        )
        formatted = detector.format_for_prompt(profile)
        assert "TONE:" in formatted
        assert "Power words:" in formatted
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/drei/project/job-applicator-python && .venv/bin/python -m pytest tests/unit/test_tone_detector.py -v`
Expected: FAIL — module `job_applicator.documents.tone_detector` does not exist

- [ ] **Step 3: Implement ToneDetector**

```python
# src/job_applicator/documents/tone_detector.py
"""Auto-detect job posting tone and provide vocabulary guidance."""

from __future__ import annotations

from dataclasses import dataclass, field

from job_applicator.utils.logging import get_logger

logger = get_logger("documents.tone_detector")

TONE_KEYWORDS: dict[str, list[str]] = {
    "corporate": [
        "compliance", "governance", "stakeholder", "enterprise", "sla", "kpi",
        "process improvement", "audit", "regulatory", "cross-functional",
        "strategic", "initiative", "deliverable", "benchmark", "roi",
        "itil", "itsm", "change management", "risk management",
    ],
    "startup": [
        "fast-paced", "wear many hats", "agile", "scrappy", "self-starter",
        "equity", "early-stage", "series a", "series b", "founder",
        "greenfield", "0 to 1", "ownership", "autonomy", "rapid growth",
        "disrupt", "innovate", "pivot",
    ],
    "technical": [
        "architecture", "system design", "scalability", "ci/cd", "microservices",
        "distributed", "infrastructure", "devops", "sre", "latency",
        "throughput", "concurrency", "api", "sdk", "framework", "pipeline",
        "kubernetes", "docker", "terraform", "cloud-native",
    ],
    "creative": [
        "brand", "storytelling", "design thinking", "user experience",
        "visual", "content", "creative", "aesthetic", "prototype",
        "wireframe", "mockup", "user research", "persona", "journey map",
        "illustration", "typography", "color theory",
    ],
}

TONE_POWER_WORDS: dict[str, list[str]] = {
    "corporate": [
        "leveraged", "orchestrated", "facilitated", "spearheaded",
        "streamlined", "optimized", "administered", "coordinated",
    ],
    "startup": [
        "built", "launched", "scaled", "pivoted", "shipped",
        "hustled", "iterated", "bootstrapped",
    ],
    "technical": [
        "architected", "engineered", "implemented", "automated",
        "designed", "deployed", "migrated", "refactored",
    ],
    "creative": [
        "designed", "crafted", "envisioned", "curated",
        "conceptualized", "illustrated", "styled", "composed",
    ],
}

TONE_EMPHASIS: dict[str, list[str]] = {
    "corporate": [
        "compliance", "process improvement", "stakeholder management",
        "risk mitigation", "strategic planning",
    ],
    "startup": [
        "ownership", "rapid iteration", "cross-functional impact",
        "resourcefulness", "adaptability",
    ],
    "technical": [
        "system design", "scalability", "performance optimization",
        "code quality", "technical leadership",
    ],
    "creative": [
        "user empathy", "visual communication", "design process",
        "brand consistency", "creative problem-solving",
    ],
}

TONE_AVOID: dict[str, list[str]] = {
    "corporate": ["casual language", "slang", "overly technical jargon"],
    "startup": ["corporate speak", "bureaucratic language", "overly formal"],
    "technical": ["buzzwords without substance", "vague claims", "fluff"],
    "creative": ["rigid corporate language", "purely technical focus", "dry tone"],
}


@dataclass
class ToneProfile:
    """Detected tone profile for a job posting."""

    primary: str = "corporate"
    confidence: float = 0.0
    scores: dict[str, float] = field(default_factory=dict)
    power_words: list[str] = field(default_factory=list)
    emphasis: list[str] = field(default_factory=list)
    avoid: list[str] = field(default_factory=list)


class ToneDetector:
    """Detect job posting tone from title, description, and requirements."""

    def detect(
        self,
        title: str,
        description: str,
        requirements: list[str],
    ) -> ToneProfile:
        """Analyze job posting and return tone profile.

        Args:
            title: Job title
            description: Job description text
            requirements: List of job requirements

        Returns:
            ToneProfile with primary tone, power words, emphasis, and avoid list
        """
        combined = f"{title} {description} {' '.join(requirements)}".lower()
        scores: dict[str, float] = {}

        for tone, keywords in TONE_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw in combined)
            scores[tone] = count / max(len(keywords), 1)

        if not any(scores.values()):
            primary = "corporate"
            confidence = 0.0
        else:
            primary = max(scores, key=scores.get)  # type: ignore[arg-type]
            total = sum(scores.values())
            confidence = scores[primary] / total if total > 0 else 0.0

        logger.info("Detected tone: %s (confidence: %.1f%%)", primary, confidence * 100)

        return ToneProfile(
            primary=primary,
            confidence=confidence,
            scores=scores,
            power_words=TONE_POWER_WORDS.get(primary, []),
            emphasis=TONE_EMPHASIS.get(primary, []),
            avoid=TONE_AVOID.get(primary, []),
        )

    def format_for_prompt(self, profile: ToneProfile) -> str:
        """Format tone profile as a prompt injection string.

        Args:
            profile: Detected tone profile

        Returns:
            Formatted string for injection into tailoring prompt
        """
        lines = [
            f"TONE: {profile.primary.title()}",
            f"- Power words: {', '.join(profile.power_words)}",
            f"- Emphasize: {', '.join(profile.emphasis)}",
            f"- Avoid: {', '.join(profile.avoid)}",
        ]
        return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/drei/project/job-applicator-python && .venv/bin/python -m pytest tests/unit/test_tone_detector.py -v`
Expected: 6 passed

- [ ] **Step 5: Run lint and typecheck**

Run: `cd /home/drei/project/job-applicator-python && ruff check src/job_applicator/documents/tone_detector.py tests/unit/test_tone_detector.py && mypy src/job_applicator/documents/tone_detector.py --ignore-missing-imports`
Expected: Clean

- [ ] **Step 6: Commit**

```bash
git add src/job_applicator/documents/tone_detector.py tests/unit/test_tone_detector.py
git commit -m "feat: add ToneDetector for auto-detecting job posting tone"
```

---

### Task 2: Section Parser

**Covers:** [S5]

**Files:**
- Modify: `src/job_applicator/documents/resume_tailor.py`
- Test: `tests/unit/test_resume_tailor.py`

- [ ] **Step 1: Write failing tests for parse_sections**

Add to `tests/unit/test_resume_tailor.py`:

```python
from job_applicator.documents.resume_tailor import parse_sections


class TestParseSections:
    def test_parse_standard_sections(self):
        text = (
            "JOHN DOE\njohn@example.com\n\n"
            "SUMMARY\nExperienced developer.\n\n"
            "EXPERIENCE\nSoftware Engineer at Corp\n2020-2024\n\n"
            "SKILLS\nPython, JavaScript, Docker\n\n"
            "EDUCATION\nBS Computer Science, MIT, 2016-2020\n"
        )
        sections = parse_sections(text)
        names = [s.name for s in sections]
        assert "SUMMARY" in names
        assert "EXPERIENCE" in names
        assert "SKILLS" in names
        assert "EDUCATION" in names

    def test_parse_mixed_case_headers(self):
        text = "Summary\nSome text.\n\nExperience\nJob stuff.\n"
        sections = parse_sections(text)
        names = [s.name for s in sections]
        assert "Summary" in names
        assert "Experience" in names

    def test_parse_no_sections_returns_single(self):
        text = "Just a plain resume with no section headers at all."
        sections = parse_sections(text)
        assert len(sections) == 1
        assert sections[0].name == "Full Document"
        assert sections[0].text == text

    def test_section_text_preserved(self):
        text = "SKILLS\nPython, JavaScript\nDocker, Kubernetes\n\nEXPERIENCE\nJob one.\n"
        sections = parse_sections(text)
        skills = [s for s in sections if s.name == "SKILLS"][0]
        assert "Python" in skills.text
        assert "Docker" in skills.text

    def test_header_with_colon(self):
        text = "Technical Skills:\nPython, Java\n\nWork Experience:\nJob stuff.\n"
        sections = parse_sections(text)
        names = [s.name for s in sections]
        assert "Technical Skills:" in names
        assert "Work Experience:" in names
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/drei/project/job-applicator-python && .venv/bin/python -m pytest tests/unit/test_resume_tailor.py::TestParseSections -v`
Expected: FAIL — `parse_sections` not defined

- [ ] **Step 3: Implement parse_sections**

Add to `src/job_applicator/documents/resume_tailor.py` (after imports, before `ResumeDateValidator`):

```python
import re
from dataclasses import dataclass


@dataclass
class ResumeSection:
    """A parsed section of a resume."""

    name: str
    text: str
    start_line: int
    end_line: int


SECTION_HEADER_RE = re.compile(
    r"^(?:"
    r"(?:SUMMARY|EXPERIENCE|SKILLS|EDUCATION|CERTIFICATIONS|PROJECTS|"
    r"OBJECTIVE|PROFILE|WORK\s+EXPERIENCE|EMPLOYMENT|QUALIFICATIONS|"
    r"ACHIEVEMENTS|INTERESTS|LANGUAGES|REFERENCES|VOLUNTEER|AWARDS)"
    r"|"
    r"[A-Z][A-Z\s]{2,49}$"
    r"|"
    r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*:"
    r")\s*$",
    re.MULTILINE | re.IGNORECASE,
)


def parse_sections(text: str) -> list[ResumeSection]:
    """Parse resume text into sections by detecting headers.

    Detects:
    - Common section names (SUMMARY, EXPERIENCE, SKILLS, etc.)
    - ALL CAPS lines under 50 chars
    - Title Case lines ending with colon

    Args:
        text: Full resume text

    Returns:
        List of ResumeSection objects. If no sections detected, returns
        a single section with name "Full Document".
    """
    lines = text.split("\n")
    headers: list[tuple[int, str]] = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if SECTION_HEADER_RE.match(stripped):
            headers.append((i, stripped))

    if not headers:
        return [ResumeSection(
            name="Full Document",
            text=text,
            start_line=0,
            end_line=len(lines) - 1,
        )]

    sections: list[ResumeSection] = []

    for idx, (line_num, header_name) in enumerate(headers):
        start = line_num + 1
        if idx + 1 < len(headers):
            end = headers[idx + 1][0] - 1
        else:
            end = len(lines) - 1
        section_text = "\n".join(lines[start:end + 1]).strip()
        sections.append(ResumeSection(
            name=header_name,
            text=section_text,
            start_line=start,
            end_line=end,
        ))

    return sections
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/drei/project/job-applicator-python && .venv/bin/python -m pytest tests/unit/test_resume_tailor.py::TestParseSections -v`
Expected: 5 passed

- [ ] **Step 5: Run full test suite to verify no regressions**

Run: `cd /home/drei/project/job-applicator-python && .venv/bin/python -m pytest tests/unit/ -v --tb=short`
Expected: 59 passed (54 existing + 5 new)

- [ ] **Step 6: Commit**

```bash
git add src/job_applicator/documents/resume_tailor.py tests/unit/test_resume_tailor.py
git commit -m "feat: add parse_sections() for section-level resume editing"
```

---

### Task 3: TailorSession Model

**Covers:** [S4]

**Files:**
- Modify: `src/job_applicator/models.py`
- Test: `tests/unit/test_models.py`

- [ ] **Step 1: Write failing tests for TailorSession**

Add to `tests/unit/test_models.py`:

```python
from job_applicator.models import TailorSession, TailoredResume, ResumeData, JobListing, JobBoard


class TestTailorSession:
    def test_session_creation(self):
        session = TailorSession(
            original_text="Original resume text",
            job_title="Developer",
            job_company="TechCo",
        )
        assert session.attempts == []
        assert session.current_index == -1

    def test_add_attempt(self):
        session = TailorSession(
            original_text="Original",
            job_title="Dev",
            job_company="Co",
        )
        result = TailoredResume(
            original_path="",
            tailored_text="Tailored v1",
            job_title="Dev",
            job_company="Co",
            match_score=0.7,
            semantic_score=0.7,
            skill_score=0.7,
            changes_summary="changes",
            attempt=1,
        )
        session.add_attempt(result)
        assert len(session.attempts) == 1
        assert session.current_index == 0

    def test_current_property(self):
        session = TailorSession(
            original_text="Original",
            job_title="Dev",
            job_company="Co",
        )
        result = TailoredResume(
            original_path="",
            tailored_text="Tailored v1",
            job_title="Dev",
            job_company="Co",
            match_score=0.7,
            semantic_score=0.7,
            skill_score=0.7,
            changes_summary="changes",
            attempt=1,
        )
        session.add_attempt(result)
        assert session.current.tailored_text == "Tailored v1"

    def test_current_empty_session_raises(self):
        session = TailorSession(
            original_text="Original",
            job_title="Dev",
            job_company="Co",
        )
        with pytest.raises(IndexError):
            _ = session.current

    def test_select_attempt(self):
        session = TailorSession(
            original_text="Original",
            job_title="Dev",
            job_company="Co",
        )
        for i in range(3):
            session.add_attempt(TailoredResume(
                original_path="",
                tailored_text=f"Version {i}",
                job_title="Dev",
                job_company="Co",
                match_score=0.5 + i * 0.1,
                semantic_score=0.5,
                skill_score=0.5,
                changes_summary="changes",
                attempt=i + 1,
            ))
        session.select(1)
        assert session.current.tailored_text == "Version 1"
        assert session.current_index == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/drei/project/job-applicator-python && .venv/bin/python -m pytest tests/unit/test_models.py::TestTailorSession -v`
Expected: FAIL — `TailorSession` not defined

- [ ] **Step 3: Implement TailorSession**

Add to `src/job_applicator/models.py` (after `TailoredResume` class):

```python
class TailorSession:
    """Tracks all tailoring attempts for a resume/job pair."""

    def __init__(
        self,
        original_text: str,
        job_title: str,
        job_company: str,
    ) -> None:
        self.original_text = original_text
        self.job_title = job_title
        self.job_company = job_company
        self.attempts: list[TailoredResume] = []
        self.current_index: int = -1

    def add_attempt(self, result: TailoredResume) -> None:
        """Add a new attempt and set it as current."""
        self.attempts.append(result)
        self.current_index = len(self.attempts) - 1

    @property
    def current(self) -> TailoredResume:
        """Get the currently selected attempt."""
        if not self.attempts or self.current_index < 0:
            raise IndexError("No attempts in session")
        return self.attempts[self.current_index]

    def select(self, index: int) -> None:
        """Select a previous attempt by index."""
        if index < 0 or index >= len(self.attempts):
            raise IndexError(f"Attempt index {index} out of range (0-{len(self.attempts) - 1})")
        self.current_index = index
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/drei/project/job-applicator-python && .venv/bin/python -m pytest tests/unit/test_models.py::TestTailorSession -v`
Expected: 5 passed

- [ ] **Step 5: Run full test suite**

Run: `cd /home/drei/project/job-applicator-python && .venv/bin/python -m pytest tests/unit/ -v --tb=short`
Expected: 64 passed (59 + 5)

- [ ] **Step 6: Commit**

```bash
git add src/job_applicator/models.py tests/unit/test_models.py
git commit -m "feat: add TailorSession model for version history tracking"
```

---

### Task 4: Diff View in CLI

**Covers:** [S3]

**Files:**
- Modify: `src/job_applicator/cli.py`

- [ ] **Step 1: Add diff rendering helper**

Add to `src/job_applicator/cli.py` (after imports, before `app` definition):

```python
import difflib


def _render_diff(console: Console, original: str, tailored: str, max_lines: int = 30) -> None:
    """Render a color-coded diff between original and tailored resume.

    Args:
        console: Rich console instance
        original: Original resume text
        tailored: Tailored resume text
        max_lines: Maximum diff lines to show (0 = unlimited)
    """
    original_lines = original.splitlines(keepends=True)
    tailored_lines = tailored.splitlines(keepends=True)

    diff = list(difflib.unified_diff(
        original_lines,
        tailored_lines,
        fromfile="original",
        tofile="tailored",
        lineterm="",
    ))

    if not diff:
        console.print("[dim]No differences found.[/dim]")
        return

    shown = 0
    for line in diff:
        if max_lines and shown >= max_lines:
            console.print(f"[dim]... {len(diff) - shown} more lines (use [D] to see full diff)[/dim]")
            break
        if line.startswith("+++") or line.startswith("---"):
            console.print(f"[bold]{line}[/bold]")
        elif line.startswith("@@"):
            console.print(f"[cyan]{line}[/cyan]")
        elif line.startswith("+"):
            console.print(f"[green]{line}[/green]")
        elif line.startswith("-"):
            console.print(f"[red]{line}[/red]")
        else:
            console.print(f"[dim]{line}[/dim]")
        shown += 1
```

- [ ] **Step 2: Integrate diff into tailor loop**

In `src/job_applicator/cli.py`, inside the `tailor` command's `_run()` function, add a `[D] Diff` option to the action table and handle it in the choice logic. Replace the action table section (around line 564-572) with:

```python
            console.print("\n[bold]What would you like to do?[/bold]")
            action_table = Table(show_header=False, box=None)
            action_table.add_column("Option", style="cyan bold")
            action_table.add_column("Description")
            action_table.add_row("[A] Accept", "Save this version as final")
            action_table.add_row("[R] Retry", "Regenerate with same instructions")
            action_table.add_row("[I] Input", "Give custom instructions to refine")
            action_table.add_row("[D] Diff", "Show changes from original resume")
            action_table.add_row("[V] History", "Browse previous attempts")
            action_table.add_row("[S] Section", "Edit a specific section")
            action_table.add_row("[Q] Quit", "Discard and exit")
            console.print(action_table)
```

Then add handling for the new choices before the `elif choice == "Q"` block:

```python
            elif choice == "D":
                _render_diff(console, session.original_text, result.tailored_text, max_lines=0)
                continue

            elif choice == "V":
                if len(session.attempts) < 2:
                    console.print("[yellow]No previous attempts yet.[/yellow]")
                    continue
                hist_table = Table(title="Version History")
                hist_table.add_column("#", style="dim")
                hist_table.add_column("Attempt")
                hist_table.add_column("Score", style="cyan")
                hist_table.add_column("Instructions")
                hist_table.add_column("Preview", style="dim")
                for i, att in enumerate(session.attempts):
                    preview = att.tailored_text[:60].replace("\n", " ")
                    marker = "→" if i == session.current_index else " "
                    hist_table.add_row(
                        marker,
                        str(att.attempt),
                        f"{att.match_score:.0%}",
                        att.user_modifications or "—",
                        preview + "...",
                    )
                console.print(hist_table)
                sel = console.input(
                    "\n[bold cyan]Select attempt # to view (or Enter to go back): [/bold cyan]"
                ).strip()
                if sel.isdigit():
                    idx = int(sel) - 1
                    if 0 <= idx < len(session.attempts):
                        session.select(idx)
                        result = session.current
                        console.print(f"[green]Switched to attempt #{result.attempt}[/green]")
                    else:
                        console.print("[red]Invalid attempt number.[/red]")
                continue

            elif choice == "S":
                from job_applicator.documents.resume_tailor import parse_sections

                sections = parse_sections(result.tailored_text)
                if len(sections) <= 1 and sections[0].name == "Full Document":
                    console.print("[yellow]Could not detect sections. Use [I] for full-resume instructions.[/yellow]")
                    continue

                console.print("\n[bold]Sections:[/bold]")
                sec_table = Table(show_header=False, box=None)
                sec_table.add_column("#", style="cyan")
                sec_table.add_column("Section", style="bold")
                sec_table.add_column("Lines", style="dim")
                for i, sec in enumerate(sections, 1):
                    line_count = sec.text.count("\n") + 1
                    sec_table.add_row(str(i), sec.name, f"{line_count} lines")
                console.print(sec_table)

                sec_choice = console.input(
                    "\n[bold cyan]Section # to edit (or Enter to go back): [/bold cyan]"
                ).strip()
                if not sec_choice.isdigit():
                    continue
                sec_idx = int(sec_choice) - 1
                if sec_idx < 0 or sec_idx >= len(sections):
                    console.print("[red]Invalid section number.[/red]")
                    continue

                target_section = sections[sec_idx]
                console.print(f"\n[dim]Editing: {target_section.name}[/dim]")
                console.print(f"[dim]{target_section.text[:200]}...[/dim]\n")

                sec_instructions = console.input(
                    "[bold]Instructions for this section: [/bold]"
                ).strip()
                if not sec_instructions:
                    console.print("[yellow]No instructions provided.[/yellow]")
                    continue

                user_instructions = (
                    f"ONLY modify the {target_section.name} section. "
                    f"Keep all other sections unchanged.\n\n"
                    f"Current {target_section.name} content:\n{target_section.text}\n\n"
                    f"User instructions for this section: {sec_instructions}"
                )
                with console.status("Refining section..."):
                    result = await tailor_engine.refine(resume_data, result, user_instructions, job)
                continue
```

- [ ] **Step 3: Wire up TailorSession in tailor command**

At the top of the `_run()` inner function in the `tailor` command (after the resume is loaded), create a `TailorSession` and use it to track attempts. Replace the initial tailoring call (around line 522-523) with:

```python
        from job_applicator.models import TailorSession

        session = TailorSession(
            original_text=resume_data.raw_text,
            job_title=job.title,
            job_company=job.company,
        )

        with console.status("Tailoring resume..."):
            result = await tailor_engine.tailor(resume_data, job, user_instructions, style)
        session.add_attempt(result)
```

And in the retry/input handlers, after `refine()` returns, add `session.add_attempt(result)`.

Also add auto-diff after the preview panel (after line 540):

```python
            _render_diff(console, session.original_text, result.tailored_text, max_lines=30)
```

- [ ] **Step 4: Run lint and typecheck**

Run: `cd /home/drei/project/job-applicator-python && ruff check src/job_applicator/cli.py && mypy src/job_applicator/cli.py --ignore-missing-imports`
Expected: Clean

- [ ] **Step 5: Run full test suite**

Run: `cd /home/drei/project/job-applicator-python && .venv/bin/python -m pytest tests/unit/ -v --tb=short`
Expected: 64 passed

- [ ] **Step 6: Commit**

```bash
git add src/job_applicator/cli.py
git commit -m "feat: add diff view, version history, and section editing to tailor CLI"
```

---

### Task 5: Integrate Tone Detection

**Covers:** [S6]

**Files:**
- Modify: `src/job_applicator/documents/resume_tailor.py`
- Modify: `src/job_applicator/cli.py`
- Test: `tests/unit/test_resume_tailor.py`

- [ ] **Step 1: Write failing test for tone integration**

Add to `tests/unit/test_resume_tailor.py`:

```python
class TestTailorWithTone:
    @pytest.mark.asyncio
    async def test_tailor_includes_tone_in_prompt(self, llm_config, sample_resume, sample_job):
        tailor = ResumeTailor(llm_config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Tailored with tone"

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response) as mock_call:
            await tailor.tailor(sample_resume, sample_job)

        call_args = mock_call.call_args
        prompt = call_args[1].get("messages", [{}])[-1].get("content", "")
        # Tone should be injected into the system or user message
        assert "TONE:" in str(call_args)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/drei/project/job-applicator-python && .venv/bin/python -m pytest tests/unit/test_resume_tailor.py::TestTailorWithTone -v`
Expected: FAIL — tone not in prompt

- [ ] **Step 3: Integrate ToneDetector into ResumeTailor.tailor()**

In `src/job_applicator/documents/resume_tailor.py`, modify the `TAILOR_PROMPT_TEMPLATE` to include a tone section. Change the template string (line 398) to:

```python
TAILOR_PROMPT_TEMPLATE = (
    "Tailor this resume for the following job:\n\n"
    "Job Title: {job_title}\n"
    "Company: {job_company}\n"
    "Location: {job_location}\n"
    "Description: {job_description}\n"
    "Requirements: {requirements}\n\n"
    "Current Resume:\n---\n{resume_text}\n---\n\n"
    "Candidate's verbatim skills (preserve these exactly, reorder only):\n"
    "{skills}\n\n"
    "Education entries that MUST appear in the output (do not merge with "
    "Certifications — they are separate sections):\n"
    "{education_entries}\n\n"
    "{tone_section}\n\n"
    "{user_instructions}\n\n"
    "Return the complete tailored resume text."
)
```

Then in the `tailor()` method, after the `edu_entries` line (around line 464), add:

```python
        from job_applicator.documents.tone_detector import ToneDetector

        tone_detector = ToneDetector()
        tone_profile = tone_detector.detect(
            title=job.title,
            description=job.description,
            requirements=job.requirements,
        )
        tone_section = tone_detector.format_for_prompt(tone_profile)
```

And pass `tone_section=tone_section` in the `.format()` call (around line 466):

```python
        prompt = TAILOR_PROMPT_TEMPLATE.format(
            job_title=job.title,
            job_company=job.company,
            job_location=job.location,
            job_description=job.description[:800],
            requirements=", ".join(job.requirements),
            resume_text=resume.raw_text[:5000],
            skills=", ".join(resume.skills),
            education_entries=edu_entries,
            tone_section=tone_section,
            user_instructions=instruction_section,
        )
```

- [ ] **Step 4: Display detected tone in CLI**

In `src/job_applicator/cli.py`, after the job is created in the `tailor` command, display the detected tone:

```python
        from job_applicator.documents.tone_detector import ToneDetector

        tone_detector = ToneDetector()
        tone_profile = tone_detector.detect(
            title=job.title,
            description=job.description,
            requirements=job.requirements,
        )
        console.print(f"[dim]Detected tone: {tone_profile.primary} (confidence: {tone_profile.confidence:.0%})[/dim]")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /home/drei/project/job-applicator-python && .venv/bin/python -m pytest tests/unit/ -v --tb=short`
Expected: 65+ passed

- [ ] **Step 6: Commit**

```bash
git add src/job_applicator/documents/resume_tailor.py src/job_applicator/cli.py tests/unit/test_resume_tailor.py
git commit -m "feat: integrate auto tone detection into resume tailoring"
```

---

### Task 6: Production Hardening

**Covers:** [S7]

**Files:**
- Modify: `src/job_applicator/cli.py`
- Test: `tests/unit/test_resume_tailor.py`

- [ ] **Step 1: Add max retry limit with warning**

In the tailor loop in `cli.py`, add a max retry check. After `attempt += 1`:

```python
            if attempt > 10:
                console.print("[red]Maximum retry limit (10) reached.[/red]")
                break
            if attempt >= 8:
                console.print("[yellow]Warning: approaching retry limit (10 max).[/yellow]")
```

- [ ] **Step 2: Add error handling around LLM calls**

Wrap the `tailor_engine.tailor()` and `tailor_engine.refine()` calls in try/except:

```python
            try:
                with console.status("Tailoring resume..."):
                    result = await tailor_engine.refine(resume_data, result, user_instructions, job)
                session.add_attempt(result)
            except Exception as exc:
                console.print(f"[red]LLM error: {exc}[/red]")
                retry_choice = console.input(
                    "[bold cyan][R] Retry or [Q] Quit? [/bold cyan]"
                ).strip().upper()
                if retry_choice == "Q":
                    break
                continue
```

Apply the same pattern to the initial `tailor()` call.

- [ ] **Step 3: Add input validation for section selection**

Already handled in Task 4 (validates digit, checks range). Verify the validation is present.

- [ ] **Step 4: Write integration test for workflow loop**

Add to `tests/unit/test_resume_tailor.py`:

```python
class TestTailorWorkflow:
    def test_tailor_session_workflow(self):
        """Test the full accept/retry/input workflow with mock data."""
        from job_applicator.models import TailorSession

        session = TailorSession(
            original_text="Original resume",
            job_title="Dev",
            job_company="Co",
        )

        # Simulate 3 attempts
        for i in range(3):
            result = TailoredResume(
                original_path="",
                tailored_text=f"Tailored version {i + 1}",
                job_title="Dev",
                job_company="Co",
                match_score=0.5 + i * 0.1,
                semantic_score=0.5,
                skill_score=0.5,
                changes_summary=f"Changes for attempt {i + 1}",
                attempt=i + 1,
                user_modifications="" if i == 0 else "more detail",
            )
            session.add_attempt(result)

        assert len(session.attempts) == 3
        assert session.current.tailored_text == "Tailored version 3"

        # Select previous attempt
        session.select(0)
        assert session.current.tailored_text == "Tailored version 1"

        # Invalid selection
        with pytest.raises(IndexError):
            session.select(99)

    def test_parse_sections_and_select(self):
        """Test section parsing for editing workflow."""
        from job_applicator.documents.resume_tailor import parse_sections

        text = "JOHN DOE\n\nSUMMARY\nDeveloper.\n\nSKILLS\nPython\n\nEDUCATION\nBS CS\n"
        sections = parse_sections(text)
        assert len(sections) == 3
        assert sections[0].name == "SUMMARY"
        assert "Developer" in sections[0].text
```

- [ ] **Step 5: Run full test suite**

Run: `cd /home/drei/project/job-applicator-python && .venv/bin/python -m pytest tests/unit/ -v --tb=short`
Expected: 67+ passed

- [ ] **Step 6: Run lint, format, and typecheck**

Run: `cd /home/drei/project/job-applicator-python && ruff check src/ tests/ && ruff format --check src/ tests/ && mypy src/job_applicator/ --ignore-missing-imports`
Expected: Clean

- [ ] **Step 7: Commit**

```bash
git add src/job_applicator/cli.py tests/unit/test_resume_tailor.py
git commit -m "feat: add production hardening — error handling, retry limits, validation"
```

---

### Task 7: Update tailor_cgi.py Script

**Covers:** [S3, S4, S5, S6]

**Files:**
- Modify: `scripts/tailor_cgi.py`

- [ ] **Step 1: Update script to use TailorSession and new features**

Rewrite `scripts/tailor_cgi.py` to use `TailorSession`, `parse_sections`, and `ToneDetector`. Copy the `_render_diff` function into the script (it's defined in `cli.py` but scripts can't import from cli without package installation). The script should mirror the CLI's enhanced workflow:

- Import `_render_diff` from `job_applicator.cli` or duplicate it locally
- Create `TailorSession` after loading resume
- Add `[D] Diff`, `[V] History`, `[S] Section` options to the action menu
- Show auto-diff after preview
- Display detected tone
- Wrap LLM calls in try/except

- [ ] **Step 2: Test the script manually**

Run: `cd /home/drei/project/job-applicator-python && .venv/bin/python scripts/tailor_cgi.py`
Expected: Script runs, shows tone detection, offers all new options

- [ ] **Step 3: Commit**

```bash
git add scripts/tailor_cgi.py
git commit -m "feat: update tailor_cgi.py with enhanced workflow features"
```

---

### Task 8: Final Cleanup and Documentation

**Covers:** [S7]

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`

- [ ] **Step 1: Update README.md with new tailor features**

Add a section documenting the enhanced tailor workflow: diff view, version history, section editing, tone detection.

- [ ] **Step 2: Update AGENTS.md with new gotchas**

Add gotchas about:
- `parse_sections()` regex patterns — may need tuning for unusual resume formats
- Tone detection is keyword-based, not LLM-based — fast but may misclassify edge cases
- Max retry limit is 10 — warning at attempt 8

- [ ] **Step 3: Run full test suite one final time**

Run: `cd /home/drei/project/job-applicator-python && .venv/bin/python -m pytest tests/unit/ -v --tb=short`
Expected: All tests pass

- [ ] **Step 4: Run lint, format, and typecheck**

Run: `cd /home/drei/project/job-applicator-python && ruff check src/ tests/ && ruff format --check src/ tests/ && mypy src/job_applicator/ --ignore-missing-imports`
Expected: Clean

- [ ] **Step 5: Commit**

```bash
git add README.md AGENTS.md
git commit -m "docs: update README and AGENTS.md with enhanced tailor workflow"
```
