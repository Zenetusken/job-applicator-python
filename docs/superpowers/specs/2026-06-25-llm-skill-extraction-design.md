# Design: LLM-driven skill extraction from job descriptions

## Status

Approved — ready for implementation planning.

## Context

`JobMatcher` currently falls back to a regex-based scan of `job.description` when `job.requirements` is empty. The scan uses `NORMALIZATION_MAP` keys and values as whole-word patterns. This is conservative but misses many real-world skills and can only recognize terms that are already in the static alias map.

## Goal

Replace the regex-based fallback with an LLM-powered skill extractor that reads the job description and returns a list of required technical skills. Harden coverage, precision, and testability while keeping the matching pipeline resilient to LLM failures.

## Decision: dictionary vs. LLM as source of truth

- **LLM** becomes the single source of truth for *which* skills are required in a job description.
- **`NORMALIZATION_MAP`** remains the single source of truth for *canonical skill names* (e.g., `Python 3` → `Python`, `reactjs` → `React`).

The dictionary is removed from the extraction path but kept as a canonicalization/post-processing layer. This avoids breaking `resume_tailor.py` and `normalize_skill`, and prevents two competing extraction paths.

## Architecture

```text
job.description
      ↓
LLMSkillExtractor.extract()
  ├── cache check (description hash → list[str])
  ├── LLM call (instructor + litellm, structured output)
  ├── normalize_skill() + is_hard_negative() filter
  ├── cache write
  └── list[str]
      ↓
JobMatcher._match_skills()  (existing embedding-based matching)
```

## Components

### `SkillExtractionOutput`

Pydantic response model:

```python
class SkillExtractionOutput(BaseModel):
    skills: list[str] = Field(description="Canonical technical skills required by the job")
```

`model_config = {"extra": "forbid"}`.

### `LLMSkillExtractor`

New module: `src/job_applicator/embeddings/skill_extraction.py`.

Responsibilities:

1. Accept a job description and an `LLMRuntime`.
2. Check a persistent cache keyed by a hash of the description and model name.
3. If cache misses, call the LLM using the project's standard litellm/instructor pattern:
   - Call `quiet_litellm()` before the request.
   - Use `model = f"openai/{config.model}"` when `config.api_base` is set, otherwise `config.model`.
   - Pass `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` for Qwen models.
   - Try instructor structured output first; fall back to direct `acompletion` + JSON extraction if the vLLM backend has no tool-call parser.
   - Strip any thinking-process markers from raw string output before parsing.
4. Normalize each returned skill with `normalize_skill()`.
5. Drop hard negatives with `is_hard_negative()`.
6. Run a text-grounded hallucination guard (see below).
7. Deduplicate and return the cleaned list.
8. Record cache hit, cache miss, LLM call, fallback, and error events through an optional `VerboseReporter` for the command's `VerboseReport`.

Edge cases handled inline: empty skill strings are dropped; duplicates are removed; corrupt cache entries are treated as misses.

Public interface:

```python
class LLMSkillExtractor:
    def __init__(self, config: LLMConfig) -> None: ...

    async def extract(
        self,
        description: str,
        runtime: LLMRuntime | None = None,
        use_cache: bool = True,
        reporter: VerboseReporter | None = None,
    ) -> list[str]: ...
```

`runtime` is injected by callers (CLI/TUI/tailor) so the circuit breaker is shared across all LLM consumers in a command. When omitted, `LLMRuntime.defaults()` is used for standalone/library use.

`reporter` is optional; when provided, cache hits, misses, LLM calls, fallback paths, and errors are recorded in the command's `VerboseReport`.

Cache location: `~/.job-applicator/skill-extraction/`. Cache files are JSON (`{"skills": [...]}`) keyed by a truncated MD5 hash of `f"{config.model}\x00{description}"`. The null separator prevents collisions such as `model-a` + `description-b` vs `model-ab` + `description-`. The cache is ignored when `use_cache=False`.

### `JobMatcher` changes

`src/job_applicator/embeddings/matching.py`:

- `match_resume_to_job` becomes `async`.
- `rank_jobs` becomes `async`.
- The constructor accepts `embedding_config: EmbeddingConfig`, `llm_config: LLMConfig`, and an optional `runtime: LLMRuntime`.
- It creates an `LLMSkillExtractor(llm_config)` instance for description fallback.
- The `runtime` is passed through to `extract()`; when omitted, `LLMRuntime.defaults()` is used.
- `_extract_requirements_from_description` is replaced by a call to `self._skill_extractor.extract(...)`.
- On LLM failure, the extractor returns `[]`, so the skill score becomes the neutral `0.5` and matching continues.

## Prompt

System:

```text
You are a technical skill extractor. Read the job description and return the concrete technical skills, programming languages, frameworks, libraries, tools, databases, cloud platforms, and methodologies required for the role.

Rules:
- Return only canonical, widely recognized names (e.g., "Python", "React", "AWS", "PostgreSQL").
- Ignore soft skills such as communication, teamwork, leadership, and problem solving.
- Ignore seniority, work arrangement, location, and compensation.
- Do not include generic terms like "software development" unless a specific technology is named.
- Return ONLY a JSON object in the format {"skills": ["Skill1", "Skill2"] }.
```

User content: the first 1 500 characters of the job description.

## Hallucination guard

After the LLM returns skills, each skill is verified against the original description. This guard applies to every returned skill, whether or not it is present in `NORMALIZATION_MAP`.

1. Normalize the skill with `normalize_skill()` to get the canonical form.
2. Build the set of acceptable surface forms for that canonical skill:
   - the canonical name,
   - the canonical name lower-cased,
   - every key in `NORMALIZATION_MAP` whose value equals the canonical name.
3. For multi-word surface forms, check case-insensitive contiguous phrase presence in the lower-cased description (e.g., `"Vue.js"` or `"vue js"` must appear as a contiguous phrase).
4. For single-word surface forms, require an exact token match in the tokenized description, **unless** the single-word token is immediately followed by another token that forms a compound explicitly present in the description. In that case the single-word match is rejected to avoid conflating distinct skills.
   - Example: description `"React Native engineer"`, LLM returns `"React"`. The token `"react"` is immediately followed by `"native"`, and the compound `"react native"` appears in the description. The match is rejected.
   - Example: description `"React and Node.js engineer"`, LLM returns `"React"`. The token `"react"` is followed by a separator or punctuation, not another token that continues the phrase. The match is accepted.
5. If no surface form is grounded in the description, drop the skill and log a warning.

Example: description `"vuejs developer"`, LLM returns `"Vue.js"`. Surface forms = `{"Vue.js", "vue.js", "vue js", "vuejs"}`. `"vuejs"` is found as a token, so the skill is kept.

Example: description `"React Native engineer"`, LLM returns `"React Native"`. The contiguous phrase `"react native"` is found, so the skill is kept.

## Resilience and fallback

- Use the shared `LLMRuntime` circuit breaker and `ValidatedOutput` retry logic.
- On any LLM error (connection, timeout, validation failure, circuit open), log a warning and return an empty skill list.
- Empty skill list yields a neutral `skill_score = 0.5`, so the overall match score degrades gracefully to the semantic score.

## Observability

Skill-extraction LLM calls are recorded in the command's `VerboseReport` (cache hit, cache miss, LLM call, fallback, error) so users can see when and why the extractor ran.

## Caller updates

| File | Change |
|------|--------|
| `src/job_applicator/cli.py` | Construct `JobMatcher(settings.embedding, settings.llm, runtime)` and add `await` to matcher calls at lines ~1352, ~1766, ~2412. |
| `src/job_applicator/tui/actions.py` | Construct matcher with `settings.llm`; convert `_score_jobs` to `async`; callers `await` it. |
| `src/job_applicator/documents/resume_tailor.py` | Pass `self._config` (LLMConfig) into `JobMatcher` construction; add `await` to `match_resume_to_job` calls. |
| `tests/unit/test_embeddings.py` | Update existing tests to `async def` / `await`; add tests for `LLMSkillExtractor`. |

## Testing

Unit tests for `LLMSkillExtractor`:

1. Cache hit returns cached skills without LLM call.
2. Cache miss calls LLM, normalizes aliases, filters hard negatives, and writes cache.
3. LLM failure returns `[]` and does not crash.
4. Empty/whitespace description returns `[]` without LLM call.
5. Hallucinated skills (returned by LLM but absent from description) are dropped.
6. Skills that match known aliases but not the canonical form are kept (e.g., "PostgreSQL" via "postgres"; "Vue.js" via "vuejs").
7. Boundary false positives such as "React" inside "React Native" are rejected; "React Native" is kept when explicitly returned.
8. Instructor fallback to direct litellm is exercised when instructor fails.
9. Corrupt cache file is treated as a miss and overwritten.
10. Changing `llm.model` changes the cache key.
11. Duplicate or empty skill strings from the LLM are cleaned.
12. `quiet_litellm()`, `enable_thinking=False`, and model prefixing are used.

Unit tests for `JobMatcher`:

1. Description-only job returns `matched_skills`/`missing_skills` via mocked extractor.
2. Explicit `job.requirements` bypass the extractor entirely.
3. Extractor failure yields neutral skill score.
4. Existing embedding-based skill matching still works at the 0.75 threshold.
5. `match_resume_to_job` and `rank_jobs` are `async` and callers `await` them.

Integration/live tests:

1. A real vLLM call extracts skills from a sample job description.
2. `job-applicator match --resume ... --jobs-file ...` produces skill lists for description-only jobs.
3. CLI `match`, `batch`, and `tailor` commands construct `JobMatcher` with `settings.llm` and `await` matcher calls.
4. TUI `_score_jobs` is async and awaited by its callers.
5. `VerboseReport` records skill-extraction LLM calls and outcomes.

## Performance and cost

- Descriptions are truncated to 1 500 characters before being sent.
- Expected output is a short JSON array (typically <200 tokens).
- Persistent cache prevents duplicate extractions across runs.
- With caching, the steady-state cost per unique job description is one small LLM call.

## Trade-offs

| Pros | Cons |
|------|------|
| Empirically justified: current recall on real jobs is 33.5% and 52% of description-only jobs return zero skills. | Adds async I/O to the matching path. |
| Handles unmapped technologies (security, networking, French terms, SaaS tools) the regex cannot know. | Requires vLLM or another LLM endpoint for description extraction. |
| Text-grounded guard preserves the current 96.7% precision while raising recall. | More tokens/latency on cache misses; mitigated by cache. |
| Graceful fallback keeps matching alive when the LLM is down. | Adds complexity to `JobMatcher` callers (async + LLM config). |

## Out of scope

- Replacing `normalize_skill` or `is_hard_negative` with an LLM.
- Changing the 0.75 skill-match embedding threshold.
- Extracting skills from résumé text (the existing fallback in `_match_skills` remains).
