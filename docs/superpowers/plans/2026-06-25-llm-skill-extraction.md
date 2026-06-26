# LLM-driven skill extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the regex-based description skill fallback in `JobMatcher` with an async LLM-powered extractor (`LLMSkillExtractor`) that uses structured output, caching, hard-negative filtering, and a text-grounded hallucination guard.

**Architecture:** Add a new `embeddings/skill_extraction.py` module containing the response model and extractor. `JobMatcher` becomes async and delegates description parsing to it. Callers in `cli.py`, `tui/actions.py`, and `resume_tailor.py` are updated to `await` matcher calls and inject `LLMConfig`/`LLMRuntime`.

**Tech Stack:** Python 3.12, Pydantic, litellm, instructor, pytest-asyncio, project shared helpers (`LLMRuntime`, `normalize_skill`, `is_hard_negative`, `quiet_litellm`, `strip_thinking_process`).

---

## File structure

| File | Responsibility |
|------|----------------|
| `src/job_applicator/embeddings/skill_extraction.py` | New. `SkillExtractionOutput` Pydantic model and `LLMSkillExtractor` class with cache, LLM call, normalization, hard-negative filter, and hallucination guard. |
| `src/job_applicator/embeddings/matching.py` | Modified. `JobMatcher` becomes async, accepts `LLMConfig` + `LLMRuntime`, uses `LLMSkillExtractor` for description fallback. |
| `src/job_applicator/cli.py` | Modified. Construct `JobMatcher(settings.embedding, settings.llm, runtime)` and `await` matcher calls in `match`, `batch`, and `tailor` paths. |
| `src/job_applicator/tui/actions.py` | Modified. `_score_jobs` becomes async; construct matcher with LLM config; callers await. |
| `src/job_applicator/documents/resume_tailor.py` | Modified. Pass `self._config` to `JobMatcher`; `await` matcher calls. |
| `tests/unit/test_embeddings.py` | Modified. Existing tests become async; new `TestLLMSkillExtractor` class. |
| `tests/integration/test_skill_extraction.py` | New. Integration tests for `LLMSkillExtractor` cache and guard behavior. |

---

## Task 1: Create `SkillExtractionOutput` and `LLMSkillExtractor`

**Files:**
- Create: `src/job_applicator/embeddings/skill_extraction.py`
- Test: `tests/unit/test_embeddings.py` (new test class added in Task 6)

### Step 1.1: Write the failing tests for the new module

First, add this import near the top of `tests/unit/test_embeddings.py`:

```python
from job_applicator.config import LLMConfig
```

Then append:

```python
class TestLLMSkillExtractor:
    """Tests for LLM-based skill extraction from job descriptions."""

    @pytest.fixture
    def llm_config(self) -> LLMConfig:
        return LLMConfig(model="test-model", api_base="", api_key="")

    @pytest.fixture
    def extractor(self, llm_config: LLMConfig) -> LLMSkillExtractor:
        return LLMSkillExtractor(llm_config)

    def test_cache_key_includes_model_and_description(self, extractor: LLMSkillExtractor) -> None:
        key1 = extractor._get_cache_key("desc one")
        key2 = extractor._get_cache_key("desc two")
        key3 = LLMSkillExtractor(LLMConfig(model="other-model"))._get_cache_key("desc one")
        assert key1 != key2
        assert key1 != key3

    async def test_empty_description_returns_empty_list(self, extractor: LLMSkillExtractor) -> None:
        assert await extractor.extract("") == []
        assert await extractor.extract("   ") == []

    async def test_cache_hit_returns_cached_skills(
        self, extractor: LLMSkillExtractor, tmp_path: Path
    ) -> None:
        extractor._cache_dir = tmp_path
        cache_path = extractor._get_cache_path("We need Python and Kubernetes.")
        cache_path.write_text('{"skills": ["Python", "Kubernetes"]}')
        result = await extractor.extract("We need Python and Kubernetes.")
        assert result == ["Kubernetes", "Python"]
```

Run:

```bash
pytest tests/unit/test_embeddings.py::TestLLMSkillExtractor -v
```

Expected: ImportError / failures because `LLMSkillExtractor` does not exist.

### Step 1.2: Create `src/job_applicator/embeddings/skill_extraction.py`

```python
"""LLM-based skill extraction from job descriptions."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from pydantic import BaseModel, Field

from job_applicator.config import LLMConfig
from job_applicator.exceptions import LLMError
from job_applicator.skills import NORMALIZATION_MAP, is_hard_negative, normalize_skill
from job_applicator.utils.llm import LLMRuntime, llm_call_error, quiet_litellm, strip_thinking_process
from job_applicator.utils.logging import get_logger
from job_applicator.utils.verbose import VerboseReporter

logger = get_logger("embeddings.skill_extraction")

EXTRACTION_MAX_TOKENS = 1024

SYSTEM_PROMPT = """You are a technical skill extractor. Read the job description and return the concrete technical skills, programming languages, frameworks, libraries, tools, databases, cloud platforms, and methodologies required for the role.

Rules:
- Return only canonical, widely recognized names (e.g., "Python", "React", "AWS", "PostgreSQL").
- Ignore soft skills such as communication, teamwork, leadership, and problem solving.
- Ignore seniority, work arrangement, location, and compensation.
- Do not include generic terms like "software development" unless a specific technology is named.
- Return the result as a JSON object with a single field "skills" containing a list of strings.
"""

USER_PROMPT = """Extract the required technical skills from this job description:

---
{description}
---

Return only the JSON object."""


class SkillExtractionOutput(BaseModel):
    """Structured output from the LLM for skill extraction."""

    model_config = {"extra": "forbid"}

    skills: list[str] = Field(description="Canonical technical skills required by the job")


class LLMSkillExtractor:
    """Extract technical skills from a job description using an LLM."""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._cache_dir = Path.home() / ".job-applicator" / "skill-extraction"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, description: str) -> str:
        content = f"{self._config.model}\x00{description}"
        return hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()[:16]

    def _get_cache_path(self, description: str) -> Path:
        return self._cache_dir / f"{self._get_cache_key(description)}.json"

    async def extract(
        self,
        description: str,
        runtime: LLMRuntime | None = None,
        use_cache: bool = True,
        reporter: VerboseReporter | None = None,
    ) -> list[str]:
        if not description or not description.strip():
            return []

        runtime = runtime or LLMRuntime.defaults(name="skill-extraction")

        if use_cache:
            cache_path = self._get_cache_path(description)
            if cache_path.exists():
                try:
                    data = json.loads(cache_path.read_text(encoding="utf-8"))
                    cached = data.get("skills", [])
                    if isinstance(cached, list):
                        logger.info("Skill extraction cache hit")
                        if reporter is not None:
                            reporter.record_llm_call(
                                model=self._config.model,
                                endpoint=self._config.api_base,
                                temperature=0.1,
                                details={"purpose": "skill-extraction", "cache": "hit", "description_length": len(description)},
                            )
                        return sorted(self._clean_skills(cached, description))
                except (json.JSONDecodeError, OSError) as exc:
                    logger.debug("Skill extraction cache read failed: %s", exc)

        try:
            raw_skills = await self._extract_with_llm(description, runtime)
            if reporter is not None:
                reporter.record_llm_call(
                    model=self._config.model,
                    endpoint=self._config.api_base,
                    temperature=0.1,
                    details={"purpose": "skill-extraction", "description_length": len(description)},
                )
        except LLMError as exc:
            logger.warning("Skill extraction failed, falling back to empty requirements: %s", exc)
            if reporter is not None:
                reporter.record_error(f"Skill extraction failed: {exc}")
            return []

        cleaned = self._clean_skills(raw_skills, description)

        if use_cache:
            try:
                cache_path = self._get_cache_path(description)
                cache_path.write_text(
                    json.dumps({"skills": cleaned}, ensure_ascii=False),
                    encoding="utf-8",
                )
            except OSError as exc:
                logger.debug("Skill extraction cache write failed: %s", exc)

        return sorted(cleaned)

    def _clean_skills(self, skills: list[str], description: str) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for skill in skills:
            cleaned = skill.strip()
            if not cleaned:
                continue
            canonical = normalize_skill(cleaned)
            if not canonical or is_hard_negative(canonical.lower()):
                continue
            if not self._is_grounded(canonical, description):
                logger.warning("Dropping ungrounded skill from LLM: %s", canonical)
                continue
            if canonical not in seen:
                seen.add(canonical)
                result.append(canonical)
        return result

    def _is_grounded(self, canonical: str, description: str) -> bool:
        desc_lower = description.lower()
        surface_forms = self._surface_forms(canonical)

        for form in surface_forms:
            form_lower = form.lower()
            if " " in form_lower:
                if form_lower in desc_lower:
                    return True
            elif self._token_matches(form_lower, description):
                return True
        return False

    def _surface_forms(self, canonical: str) -> set[str]:
        forms = {canonical, canonical.lower()}
        for alias, target in NORMALIZATION_MAP.items():
            if target.lower() == canonical.lower():
                forms.add(alias)
        return forms

    def _token_matches(self, word: str, description: str) -> bool:
        """Return True if ``word`` appears as a grounded token in ``description``.

        A token is grounded when it appears as a whole word and is not the
        first word of a multi-word compound that is explicitly present in the
        text. This prevents ``React`` from matching ``React Native`` while
        still allowing ``React`` to match ``React and Node.js``.
        """
        desc_lower = description.lower()
        tokens = re.findall(r"\b\w+(?:\.\w+)*\b", description)
        for i, token in enumerate(tokens):
            if token.lower() != word:
                continue
            if i + 1 < len(tokens):
                next_token = tokens[i + 1]
                # If the next token is capitalized, it may form a named compound
                # (e.g., "React Native", "Azure DevOps"). Reject the single-word
                # match when that compound appears in the text.
                if next_token[0].isupper():
                    compound = f"{token} {next_token}".lower()
                    if compound in desc_lower:
                        continue
            return True
        return False

    async def _extract_with_llm(self, description: str, runtime: LLMRuntime) -> list[str]:
        quiet_litellm()
        model = f"openai/{self._config.model}" if self._config.api_base else self._config.model
        truncated = description[:1500]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(description=truncated)},
        ]

        # Try instructor structured output first.
        try:
            import instructor
            from litellm import acompletion

            client = instructor.from_litellm(acompletion)
            response = await runtime.run(
                lambda _: client.create(
                    model=model,
                    api_base=self._config.api_base,
                    api_key=self._config.api_key,
                    messages=messages,
                    response_model=SkillExtractionOutput,
                    max_retries=1,
                    max_tokens=min(self._config.max_tokens, EXTRACTION_MAX_TOKENS),
                    temperature=0.1,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
            )
            return response.skills
        except Exception as exc:  # noqa: BLE001
            logger.warning("Instructor skill extraction failed, trying direct completion: %s", exc)

        # Fallback to direct litellm completion + JSON extraction.
        try:
            from litellm import acompletion

            raw = await runtime.run(
                lambda _: acompletion(
                    model=model,
                    api_base=self._config.api_base,
                    api_key=self._config.api_key,
                    messages=messages,
                    max_tokens=min(self._config.max_tokens, EXTRACTION_MAX_TOKENS),
                    temperature=0.1,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
            )
            content = raw.choices[0].message.content or ""
            content = strip_thinking_process(content)
            parsed = json.loads(content)
            if isinstance(parsed, dict) and "skills" in parsed:
                return parsed["skills"]
            if isinstance(parsed, list):
                return parsed
            raise LLMError(f"Unexpected skill extraction response shape: {parsed!r}")
        except Exception as exc:
            raise llm_call_error(exc, self._config.api_base) from exc
```

Run:

```bash
pytest tests/unit/test_embeddings.py::TestLLMSkillExtractor -v
```

Expected: The three tests pass.

### Step 1.3: Commit

```bash
git add src/job_applicator/embeddings/skill_extraction.py tests/unit/test_embeddings.py
git commit -m "feat(skill-extraction): add LLMSkillExtractor with cache and hallucination guard"
```

---

## Task 2: Make `JobMatcher` async and wire `LLMSkillExtractor`

**Files:**
- Modify: `src/job_applicator/embeddings/matching.py`
- Test: `tests/unit/test_embeddings.py`

### Step 2.1: Add async tests for `JobMatcher` description extraction

Append to `tests/unit/test_embeddings.py`:

```python
class TestJobMatcherAsyncExtraction:
    """Tests for JobMatcher using LLMSkillExtractor for descriptions."""

    @pytest.fixture
    def matcher(self) -> JobMatcher:
        return JobMatcher(
            EmbeddingConfig(device="cpu", memory_limit_gb=0.5),
            LLMConfig(model="test-model"),
        )

    async def test_description_only_job_uses_extractor(
        self, matcher: JobMatcher, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from job_applicator.embeddings.matching import JobMatcher as JM
        from job_applicator.models import JobBoard, JobListing

        async def fake_extract(description: str, runtime: object = None, use_cache: bool = True) -> list[str]:
            return ["Python", "FastAPI"]

        monkeypatch.setattr(matcher._skill_extractor, "extract", fake_extract)

        resume = ResumeData(raw_text="Skills: Python", skills=["Python"])
        job = JobListing(
            title="Backend Dev",
            company="Acme",
            url="https://example.com/1",
            board=JobBoard.LINKEDIN,
            description="We need Python and FastAPI.",
            requirements=[],
        )
        result = await matcher.match_resume_to_job(resume, job)
        assert "Python" in result.matched_skills
        assert "FastAPI" in result.missing_skills

    async def test_explicit_requirements_bypass_extractor(
        self, matcher: JobMatcher, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from job_applicator.models import JobBoard, JobListing

        called = False
        async def fake_extract(*args: object, **kwargs: object) -> list[str]:
            nonlocal called
            called = True
            return ["Python"]

        monkeypatch.setattr(matcher._skill_extractor, "extract", fake_extract)

        resume = ResumeData(raw_text="Skills: Python", skills=["Python"])
        job = JobListing(
            title="Backend Dev",
            company="Acme",
            url="https://example.com/1",
            board=JobBoard.LINKEDIN,
            description="...",
            requirements=["Python", "Django"],
        )
        result = await matcher.match_resume_to_job(resume, job)
        assert not called
        assert "Python" in result.matched_skills
        assert "Django" in result.missing_skills

    async def test_extractor_failure_yields_neutral_skill_score(
        self, matcher: JobMatcher, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from job_applicator.models import JobBoard, JobListing

        async def fake_extract(*args: object, **kwargs: object) -> list[str]:
            return []

        monkeypatch.setattr(matcher._skill_extractor, "extract", fake_extract)

        resume = ResumeData(raw_text="Skills: Python", skills=["Python"])
        job = JobListing(
            title="Backend Dev",
            company="Acme",
            url="https://example.com/1",
            board=JobBoard.LINKEDIN,
            description="We need Python and FastAPI.",
            requirements=[],
        )
        result = await matcher.match_resume_to_job(resume, job)
        assert result.matched_skills == []
        assert result.missing_skills == []
        assert result.skill_score == 0.5
```

Run:

```bash
pytest tests/unit/test_embeddings.py::TestJobMatcherAsyncExtraction -v
```

Expected: Failures because `JobMatcher` constructor signature and async behavior do not match.

### Step 2.2: Modify `src/job_applicator/embeddings/matching.py`

Import at the top:

```python
from job_applicator.config import EmbeddingConfig, LLMConfig
from job_applicator.embeddings.service import EmbeddingService, EmbeddingVector
from job_applicator.embeddings.skill_extraction import LLMSkillExtractor
from job_applicator.models import JobListing, ResumeData
from job_applicator.skills import is_hard_negative
from job_applicator.utils.llm import LLMRuntime
from job_applicator.utils.logging import get_logger
from job_applicator.utils.verbose import VerboseReporter
```

Remove the existing import `from job_applicator.skills import NORMALIZATION_MAP, is_hard_negative` and replace with `from job_applicator.skills import is_hard_negative`.

Update `JobMatcher.__init__`:

```python
    def __init__(
        self,
        embedding_config: EmbeddingConfig,
        llm_config: LLMConfig | None = None,
        runtime: LLMRuntime | None = None,
        reporter: VerboseReporter | None = None,
    ) -> None:
        self._config = embedding_config
        self._service = EmbeddingService(embedding_config)
        self._skill_extractor = LLMSkillExtractor(llm_config or LLMConfig())
        self._runtime = runtime
        self._reporter = reporter
```

Update `match_resume_to_job` to be async:

```python
    async def match_resume_to_job(
        self,
        resume: ResumeData,
        job: JobListing,
    ) -> MatchResult:
        resume_emb = self.compute_resume_embedding(resume)
        job_emb = self.compute_job_embedding(job)
        semantic_score = self._service.similarity(resume_emb, job_emb)

        matched_skills, missing_skills = await self._match_skills(
            resume.skills, job.requirements, resume.raw_text, job.description
        )

        skill_score = self._compute_skill_score(matched_skills, missing_skills)
        score = (0.6 * semantic_score) + (0.4 * skill_score)
        summary = self._generate_match_summary(score, matched_skills, missing_skills)

        return MatchResult(
            job=job,
            score=score,
            semantic_score=semantic_score,
            skill_score=skill_score,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            summary=summary,
        )
```

Update `rank_jobs` to be async:

```python
    async def rank_jobs(
        self,
        resume: ResumeData,
        jobs: list[JobListing],
        top_k: int = 10,
    ) -> list[MatchResult]:
        if not jobs:
            return []

        resume_emb = self.compute_resume_embedding(resume)
        job_texts = []
        for job in jobs:
            parts = [f"Job: {job.title} at {job.company}"]
            if job.location:
                parts.append(f"Location: {job.location}")
            if job.description:
                parts.append(job.description[:500])
            if job.requirements:
                parts.append(f"Requirements: {', '.join(job.requirements)}")
            job_texts.append(" | ".join(parts)[:1500])

        job_embs = self._service.embed_batch(job_texts)

        matches = []
        for job, job_emb in zip(jobs, job_embs, strict=False):
            semantic_score = self._service.similarity(resume_emb, job_emb)
            matched, missing = await self._match_skills(
                resume.skills, job.requirements, resume.raw_text, job.description
            )
            skill_score = self._compute_skill_score(matched, missing)
            score = (0.6 * semantic_score) + (0.4 * skill_score)
            summary = self._generate_match_summary(score, matched, missing)
            matches.append(
                MatchResult(
                    job=job,
                    score=score,
                    semantic_score=semantic_score,
                    skill_score=skill_score,
                    matched_skills=matched,
                    missing_skills=missing,
                    summary=summary,
                )
            )

        matches.sort(key=lambda x: x.score, reverse=True)
        return matches[:top_k]
```

Update `_match_skills` to be async:

```python
    async def _match_skills(
        self,
        resume_skills: list[str],
        job_requirements: list[str],
        resume_text: str = "",
        job_description: str = "",
    ) -> tuple[list[str], list[str]]:
        from job_applicator.skills import is_hard_negative, normalize_skill

        if not job_requirements:
            job_requirements = await self._skill_extractor.extract(
                job_description,
                runtime=self._runtime,
                reporter=self._reporter,
            )
        ...
```

Delete the old `_extract_requirements_from_description` method entirely.

Run:

```bash
pytest tests/unit/test_embeddings.py::TestJobMatcherAsyncExtraction -v
```

Expected: Tests pass.

### Step 2.3: Update existing sync `JobMatcher` tests to async

In `tests/unit/test_embeddings.py`, find `TestJobMatcher` and `TestDescriptionSkillExtraction` classes. Update every test that calls `matcher._match_skills(...)`, `matcher.match_resume_to_job(...)`, or `matcher.rank_jobs(...)` to `async def` and add `await`.

Also update fixture usage: the `matcher` fixture in tests should now construct with `LLMConfig`.

Run:

```bash
pytest tests/unit/test_embeddings.py -v
```

Expected: All tests in this file pass.

### Step 2.4: Commit

```bash
git add src/job_applicator/embeddings/matching.py tests/unit/test_embeddings.py
git commit -m "feat(matching): make JobMatcher async and wire LLMSkillExtractor"
```

---

## Task 3: Update CLI callers

**Files:**
- Modify: `src/job_applicator/cli.py`

### Step 3.1: Update `match` command

Locate the `match` command (around line 1299). Find:

```python
        # Match
        with console.status("Computing embeddings and matching..."):
            matcher = JobMatcher(settings.embedding)
            matches = matcher.rank_jobs(resume_data, jobs, top_k=top_k)
```

Replace with:

```python
        from job_applicator.utils.llm import LLMRuntime

        runtime = LLMRuntime.from_config(settings.llm_resilience, name="match")

        # Match
        with console.status("Computing embeddings and matching..."):
            matcher = JobMatcher(settings.embedding, settings.llm, runtime, reporter=reporter)
            matches = await matcher.rank_jobs(resume_data, jobs, top_k=top_k)
```

Ensure the enclosing `_run()` coroutine is declared `async def _run()`.

### Step 3.2: Update `batch` command

Locate around line 1704. Find:

```python
        matcher = JobMatcher(settings.embedding)
        with console.status("Computing match scores..."):
            matches = matcher.rank_jobs(resume_data, jobs, top_k=top_k)
```

Replace with:

```python
        from job_applicator.utils.llm import LLMRuntime

        runtime = LLMRuntime.from_config(settings.llm_resilience, name="batch")
        matcher = JobMatcher(settings.embedding, settings.llm, runtime, reporter=reporter)
        with console.status("Computing match scores..."):
            matches = await matcher.rank_jobs(resume_data, jobs, top_k=top_k)
```

### Step 3.3: Update `tailor` command pre-match

Locate around line 2412. Find:

```python
                matcher = JobMatcher(settings.embedding)
                pre_match = matcher.match_resume_to_job(resume_data, job)
```

Replace with:

```python
                from job_applicator.utils.llm import LLMRuntime

                runtime = LLMRuntime.from_config(settings.llm_resilience, name="tailor")
                matcher = JobMatcher(settings.embedding, settings.llm, runtime, reporter=reporter)
                pre_match = await matcher.match_resume_to_job(resume_data, job)
```

### Step 3.4: Verify CLI compiles

Run:

```bash
python -c "from job_applicator.cli import app; print('cli import ok')"
```

Expected: No import/syntax errors.

### Step 3.5: Commit

```bash
git add src/job_applicator/cli.py
git commit -m "refactor(cli): await async JobMatcher calls in match/batch/tailor"
```

---

## Task 4: Update TUI actions

**Files:**
- Modify: `src/job_applicator/tui/actions.py`

### Step 4.1: Convert `_score_jobs` to async

Locate `_score_jobs` (around line 268). Replace:

```python
def _score_jobs(settings: AppSettings, jobs: list[JobListing]) -> list[MatchResult]:
    """Load the résumé and rank ``jobs`` against it (sync, CPU/GPU — call via to_thread)."""
    from job_applicator.documents.resume import ResumeLoader
    from job_applicator.embeddings.matching import JobMatcher

    resume = ResumeLoader().load(settings.resume_path)
    return JobMatcher(settings.embedding).rank_jobs(resume, jobs, len(jobs))
```

with:

```python
async def _score_jobs(settings: AppSettings, jobs: list[JobListing]) -> list[MatchResult]:
    """Load the résumé and rank ``jobs`` against it."""
    from job_applicator.documents.resume import ResumeLoader
    from job_applicator.embeddings.matching import JobMatcher
    from job_applicator.utils.llm import LLMRuntime

    resume = ResumeLoader().load(settings.resume_path)
    runtime = LLMRuntime.from_config(settings.llm_resilience, name="tui-score")
    matcher = JobMatcher(settings.embedding, settings.llm, runtime)
    return await matcher.rank_jobs(resume, jobs, len(jobs))
```

### Step 4.2: Find and await `_score_jobs` callers

Search for `_score_jobs(` usages in `src/job_applicator/tui/`. Update each call site from:

```python
results = _score_jobs(settings, jobs)
```

to:

```python
results = await _score_jobs(settings, jobs)
```

If a caller is sync and cannot be async, wrap the call in `asyncio.run(_score_jobs(...))` or refactor the caller to async. Prefer async propagation.

### Step 4.3: Verify TUI import

Run:

```bash
python -c "from job_applicator.tui import actions; print('tui import ok')"
```

Expected: No import/syntax errors.

### Step 4.4: Commit

```bash
git add src/job_applicator/tui/actions.py
git commit -m "refactor(tui): make _score_jobs async and inject LLM config"
```

---

## Task 5: Update `resume_tailor.py`

**Files:**
- Modify: `src/job_applicator/documents/resume_tailor.py`

### Step 5.1: Update first `JobMatcher` usage

Locate around line 650. Find:

```python
        if matcher is None:
            matcher = JobMatcher(EmbeddingConfig(device="cpu", memory_limit_gb=0.5))
        match_result = matcher.match_resume_to_job(resume, job)
```

Replace with:

```python
        if matcher is None:
            matcher = JobMatcher(
                EmbeddingConfig(device="cpu", memory_limit_gb=0.5),
                self._config,
                runtime,
            )
        match_result = await matcher.match_resume_to_job(resume, job)
```

Note: `runtime` should be passed into `tailor()` and available here, or construct `LLMRuntime.from_config(self._resilience, ...)` if `ResumeTailor` holds a runtime. Check `ResumeTailor.__init__` signature; it already accepts `runtime: LLMRuntime | None = None`, so use `self._runtime`.

### Step 5.2: Update second `JobMatcher` usage

Locate around line 791. Find:

```python
        if matcher is None:
            matcher = JobMatcher(EmbeddingConfig(device="cpu", memory_limit_gb=0.5))

        synthetic_resume = ResumeData(...)
        new_match = matcher.match_resume_to_job(synthetic_resume, job)
```

Replace with:

```python
        if matcher is None:
            matcher = JobMatcher(
                EmbeddingConfig(device="cpu", memory_limit_gb=0.5),
                self._config,
                self._runtime,
            )

        synthetic_resume = ResumeData(...)
        new_match = await matcher.match_resume_to_job(synthetic_resume, job)
```

### Step 5.3: Verify `resume_tailor` import

Run:

```bash
python -c "from job_applicator.documents.resume_tailor import ResumeTailor; print('tailor import ok')"
```

Expected: No import/syntax errors.

### Step 4.4: Commit

```bash
git add src/job_applicator/documents/resume_tailor.py
git commit -m "refactor(resume_tailor): await async JobMatcher and inject LLM config"
```

---

## Task 6: Expand unit tests

**Files:**
- Modify: `tests/unit/test_embeddings.py`
- Create: `tests/integration/test_skill_extraction.py`

### Step 6.1: Add remaining `LLMSkillExtractor` unit tests

Append to `tests/unit/test_embeddings.py`:

```python
    async def test_hard_negatives_are_filtered(
        self, extractor: LLMSkillExtractor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_llm(description: str, runtime: object) -> list[str]:
            return ["Python", "team player", "communication"]

        monkeypatch.setattr(extractor, "_extract_with_llm", fake_llm)
        result = await extractor.extract("We need Python and a team player.")
        assert "Python" in result
        assert "team player" not in result
        assert "communication" not in result

    async def test_hallucinated_skills_are_dropped(
        self, extractor: LLMSkillExtractor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_llm(description: str, runtime: object) -> list[str]:
            return ["Python", "Rust"]

        monkeypatch.setattr(extractor, "_extract_with_llm", fake_llm)
        result = await extractor.extract("We need Python developers.")
        assert "Python" in result
        assert "Rust" not in result

    async def test_alias_grounding_keeps_canonical_skill(
        self, extractor: LLMSkillExtractor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_llm(description: str, runtime: object) -> list[str]:
            return ["PostgreSQL"]

        monkeypatch.setattr(extractor, "_extract_with_llm", fake_llm)
        result = await extractor.extract("Experience with postgres required.")
        assert "PostgreSQL" in result

    async def test_react_inside_react_native_is_rejected(
        self, extractor: LLMSkillExtractor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_llm(description: str, runtime: object) -> list[str]:
            return ["React"]

        monkeypatch.setattr(extractor, "_extract_with_llm", fake_llm)
        result = await extractor.extract("Looking for a React Native engineer.")
        assert "React" not in result

    async def test_react_native_kept_when_explicit(
        self, extractor: LLMSkillExtractor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_llm(description: str, runtime: object) -> list[str]:
            return ["React Native"]

        monkeypatch.setattr(extractor, "_extract_with_llm", fake_llm)
        result = await extractor.extract("Looking for a React Native engineer.")
        assert "React Native" in result

    async def test_corrupt_cache_is_treated_as_miss(
        self, extractor: LLMSkillExtractor, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        extractor._cache_dir = tmp_path
        desc = "We need Python."
        cache_path = extractor._get_cache_path(desc)
        cache_path.write_text("not valid json")

        async def fake_llm(description: str, runtime: object) -> list[str]:
            return ["Python"]

        monkeypatch.setattr(extractor, "_extract_with_llm", fake_llm)
        result = await extractor.extract(desc)
        assert "Python" in result
```

Run:

```bash
pytest tests/unit/test_embeddings.py::TestLLMSkillExtractor -v
```

Expected: All tests pass.

### Step 6.2: Create integration test file

Create `tests/integration/test_skill_extraction.py`:

```python
"""Integration tests for LLMSkillExtractor."""

from __future__ import annotations

import pytest

from job_applicator.config import LLMConfig
from job_applicator.embeddings.skill_extraction import LLMSkillExtractor


class TestSkillExtractionIntegration:
    async def test_extracts_python_from_description_with_mocked_llm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        extractor = LLMSkillExtractor(LLMConfig(model="test"))

        async def fake_llm(description: str, runtime: object) -> list[str]:
            return ["Python", "FastAPI", "PostgreSQL"]

        monkeypatch.setattr(extractor, "_extract_with_llm", fake_llm)

        result = await extractor.extract(
            "We are looking for a backend engineer with Python, FastAPI, and PostgreSQL."
        )
        assert set(result) == {"FastAPI", "PostgreSQL", "Python"}

    async def test_unmapped_skill_grounded_by_token_match(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        extractor = LLMSkillExtractor(LLMConfig(model="test"))

        async def fake_llm(description: str, runtime: object) -> list[str]:
            return ["Salesforce"]

        monkeypatch.setattr(extractor, "_extract_with_llm", fake_llm)

        result = await extractor.extract("Experience with Salesforce CRM is required.")
        assert "Salesforce" in result
```

Run:

```bash
pytest tests/integration/test_skill_extraction.py -v
```

Expected: Tests pass.

### Step 6.3: Commit

```bash
git add tests/unit/test_embeddings.py tests/integration/test_skill_extraction.py
git commit -m "test(skill-extraction): expand unit tests and add integration tests"
```

---

## Task 7: Lint, typecheck, and full test suite

**Files:**
- All modified files

### Step 7.1: Run ruff check and format

```bash
ruff check src/ tests/
ruff format src/ tests/
```

Expected: No errors. Auto-fix any issues and re-run.

### Step 7.2: Run mypy

```bash
mypy src/
```

Expected: No type errors in `src/`.

### Step 7.3: Run unit tests

```bash
pytest -m unit -q
```

Expected: All unit tests pass.

### Step 7.4: Run integration tests

```bash
pytest -m integration -q
```

Expected: All integration tests pass.

### Step 7.5: Commit

```bash
git add -A
git commit -m "chore: lint, format, and typecheck skill extraction changes"
```

---

## Task 8: Code review checkpoint #1 — `LLMSkillExtractor` implementation

**Files:**
- `src/job_applicator/embeddings/skill_extraction.py`
- `tests/unit/test_embeddings.py` (new extractor tests)

Dispatch an independent code-review subagent:

```
Review the LLMSkillExtractor implementation in
src/job_applicator/embeddings/skill_extraction.py
and its unit tests in tests/unit/test_embeddings.py::TestLLMSkillExtractor.

Check:
- The hallucination guard correctly handles aliases, compounds, and unmapped skills.
- Cache behavior (hit, miss, corruption) is correct.
- LLM call follows project conventions (quiet_litellm, model prefix, enable_thinking, fallback).
- Hard-negative filtering works.
- Tests cover the spec requirements.
- No type errors or obvious bugs.

Deliverable: APPROVED or CHANGES REQUESTED with concrete fixes.
```

If changes are requested, fix them and re-run the relevant tests before proceeding.

---

## Task 9: Code review checkpoint #2 — `JobMatcher` async wiring and callers

**Files:**
- `src/job_applicator/embeddings/matching.py`
- `src/job_applicator/cli.py`
- `src/job_applicator/tui/actions.py`
- `src/job_applicator/documents/resume_tailor.py`

Dispatch an independent code-review subagent:

```
Review the async conversion of JobMatcher and all caller updates.

Check:
- All matcher calls are awaited.
- LLMConfig and LLMRuntime are injected correctly.
- No sync callers remain unawaited.
- Existing embedding-based matching is unchanged.
- resume_tailor passes its own LLMConfig/runtime to JobMatcher.
- Tests were updated to async and pass.

Deliverable: APPROVED or CHANGES REQUESTED with concrete fixes.
```

If changes are requested, fix them and re-run unit + integration tests.

---

## Task 10: Live smoke test

**Prerequisite:** local vLLM running at `http://localhost:8000/v1`.

### Step 10.1: Create a small jobs file

Create `/tmp/smoke_jobs.json`:

```json
[
  {
    "title": "Backend Engineer",
    "company": "Smoke Corp",
    "url": "https://example.com/job/1",
    "board": "LINKEDIN",
    "description": "We are looking for a backend engineer with Python, FastAPI, PostgreSQL, Docker, and Kubernetes experience.",
    "requirements": []
  }
]
```

### Step 10.2: Run match with a sample resume

Use an existing resume in the project or create a minimal PDF/text resume with skills: Python, FastAPI, Docker.

```bash
job-applicator match --resume <path-to-resume> --jobs-file /tmp/smoke_jobs.json --json --verbose
```

Expected:
- Command exits 0.
- JSON output includes `matched_skills` containing Python, FastAPI, Docker.
- JSON output includes `missing_skills` containing PostgreSQL, Kubernetes.
- Verbose report shows a skill-extraction LLM call.

### Step 10.3: Run with explicit requirements

Modify `/tmp/smoke_jobs.json` to include `requirements: ["Python", "Rust"]` and re-run.

Expected:
- The extractor is bypassed (no skill-extraction LLM call in verbose report).
- `matched_skills` contains Python; `missing_skills` contains Rust.

### Step 10.4: Commit any test fixture or doc updates

```bash
git add -A
git commit -m "test: add live smoke test notes for skill extraction"
```

---

## Task 11: Final verification and handoff

### Step 11.1: Full green gate

```bash
ruff check src/ tests/
ruff format --check src/ tests/
mypy src/
pytest -m unit -q
pytest -m integration -q
```

Expected: All pass.

### Step 11.2: Update docs

If `AGENTS.md` or `MEMORY.md` describe the old regex-based extractor, update them to mention the new LLM-based extractor.

### Step 11.3: Final independent code review

Dispatch a quality-review subagent over the entire diff:

```
Review the full set of changes for the LLM-driven skill extraction feature.
Files to review:
- src/job_applicator/embeddings/skill_extraction.py
- src/job_applicator/embeddings/matching.py
- src/job_applicator/cli.py
- src/job_applicator/tui/actions.py
- src/job_applicator/documents/resume_tailor.py
- tests/unit/test_embeddings.py
- tests/integration/test_skill_extraction.py

Check:
- Feature matches the approved spec (docs/superpowers/specs/2026-06-25-llm-skill-extraction-design.md).
- Business logic is sound (single source of truth, graceful fallback, hallucination guard).
- Tests are comprehensive and pass.
- No regressions in existing behavior.
- Code style and type safety are correct.

Deliverable: APPROVED or CHANGES REQUESTED.
```

### Step 11.4: Final commit

```bash
git add -A
git commit -m "feat: LLM-driven skill extraction from job descriptions"
```

---

## Self-review checklist

- [ ] Spec coverage: every requirement in the spec maps to at least one task.
- [ ] No placeholders: plan contains code, commands, and expected outputs; no "TBD" or "implement later".
- [ ] Type consistency: `JobMatcher(embedding_config, llm_config, runtime)` signature is used everywhere.
- [ ] Async consistency: every matcher call is awaited.
- [ ] Test coverage: unit + integration + live smoke tests are specified.
- [ ] Code review checkpoints: independent subagent reviews at logical boundaries.
