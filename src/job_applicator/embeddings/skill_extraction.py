"""LLM-driven skill extraction from job descriptions.

Extracts canonical technical skills from a job description, normalizes them,
filters hard negatives, and runs a text-grounded hallucination guard so only
skills actually present in the description are returned.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import instructor
from instructor.core import InstructorError
from litellm import acompletion
from litellm.exceptions import APIError
from pydantic import BaseModel, Field, ValidationError

from job_applicator.config import LLMConfig
from job_applicator.skills import NORMALIZATION_MAP, is_hard_negative, normalize_skill
from job_applicator.utils.llm import (
    LLMRuntime,
    llm_call_error,
    quiet_litellm,
    strip_thinking_process,
)
from job_applicator.utils.logging import get_logger
from job_applicator.utils.verbose import VerboseReporter

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit("This module is not executable")

logger = get_logger("embeddings.skill_extraction")

SKILL_SYSTEM_PROMPT = (
    "You are a technical skill extractor. Read the job description and return the "
    "concrete technical skills, programming languages, frameworks, libraries, tools, "
    "databases, cloud platforms, and methodologies required for the role.\n\n"
    "Rules:\n"
    "- Return only canonical, widely recognized names "
    '(e.g., "Python", "React", "AWS", "PostgreSQL").\n'
    "- Ignore soft skills such as communication, teamwork, leadership, "
    "and problem solving.\n"
    "- Ignore seniority, work arrangement, location, and compensation.\n"
    '- Do not include generic terms like "software development" unless a '
    "specific technology is named.\n"
    '- Return ONLY a JSON object in the format {"skills": ["Skill1", "Skill2"] }.'
)

SKILL_USER_PROMPT = "{}"

MAX_DESCRIPTION_LENGTH = 1500

# Common English words that cannot form the second word of a multi-word skill
# compound. This keeps the hallucination guard from rejecting a single-word
# skill just because it happens to be followed by a function word in prose.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "need",
        "dare",
        "ought",
        "used",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "and",
        "but",
        "or",
        "yet",
        "so",
        "if",
        "because",
        "although",
        "though",
        "while",
        "where",
        "when",
        "that",
        "which",
        "who",
        "whom",
        "whose",
        "what",
        "this",
        "these",
        "those",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "me",
        "him",
        "her",
        "us",
        "them",
        "my",
        "your",
        "his",
        "its",
        "our",
        "their",
    }
)


class SkillExtractionOutput(BaseModel):
    """Structured output for LLM skill extraction."""

    skills: list[str] = Field(description="Canonical technical skills required by the job")
    model_config = {"extra": "forbid"}


@dataclass
class _ExtractionResult:
    """Result of an LLM skill-extraction attempt."""

    skills: list[str]
    method: str
    fallback: bool


class LLMSkillExtractor:
    """Extract technical skills from a job description using an LLM.

    Caches results persistently by a hash of the model name and description to
    avoid duplicate LLM calls. All returned skills are normalized, filtered for
    hard negatives, and verified against the original description.
    """

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._cache_dir = Path.home() / ".job-applicator" / "skill-extraction"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, description: str) -> str:
        """Generate a cache key from model name and description."""
        content = f"{self._config.model}\x00{description}"
        return hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()[:16]

    def _get_cache_path(self, description: str) -> Path:
        """Get the cache file path for a description."""
        return self._cache_dir / f"{self._get_cache_key(description)}.json"

    def _load_cache(self, description: str) -> list[str] | None:
        """Load cached skills, returning None on miss or corrupt entry."""
        cache_path = self._get_cache_path(description)
        if not cache_path.exists():
            return None
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("skills"), list):
                return [str(s) for s in data["skills"]]
        except (OSError, json.JSONDecodeError) as exc:
            logger.debug("Skill extraction cache read failed: %s", exc)
        return None

    def _save_cache(self, description: str, skills: list[str]) -> None:
        """Write skills to the persistent cache."""
        cache_path = self._get_cache_path(description)
        try:
            cache_path.write_text(
                json.dumps({"skills": skills}, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("Could not write skill extraction cache: %s", exc)

    async def extract(
        self,
        description: str,
        runtime: LLMRuntime | None = None,
        use_cache: bool = True,
        reporter: VerboseReporter | None = None,
    ) -> list[str]:
        """Extract technical skills from a job description.

        Args:
            description: Job description text.
            runtime: Optional shared LLM runtime. Defaults to a standalone runtime
                when not provided.
            use_cache: Whether to read/write the persistent cache.
            reporter: Optional verbose reporter for observability.

        Returns:
            Sorted, deduplicated list of canonical skills grounded in the text.
        """
        if not description or not description.strip():
            return []

        if runtime is None:
            runtime = LLMRuntime.defaults(name="skill-extraction")

        if use_cache:
            cached = self._load_cache(description)
            if cached is not None:
                if reporter is not None:
                    reporter.record_llm_call(
                        model=self._config.model,
                        endpoint=self._config.api_base,
                        temperature=self._config.temperature,
                        details={"skill_extraction": "cache_hit"},
                    )
                return self._clean_skills(cached, description)

        if reporter is not None:
            reporter.record_llm_call(
                model=self._config.model,
                endpoint=self._config.api_base,
                temperature=self._config.temperature,
                details={"skill_extraction": "cache_miss"},
            )

        raw_skills: list[str] = []
        method: str | None = None
        fallback: bool = False
        try:
            result = await runtime.run(lambda _prev: self._call_llm(description))
            raw_skills = result.skills
            method = result.method
            fallback = result.fallback
        except Exception as exc:
            error = llm_call_error(exc, self._config.api_base or "")
            logger.warning("Skill extraction LLM call failed: %s", error)
            if reporter is not None:
                reporter.record_llm_call(
                    model=self._config.model,
                    endpoint=self._config.api_base,
                    temperature=self._config.temperature,
                    details={"skill_extraction": "error", "error": str(exc)},
                )
                reporter.record_error(f"Skill extraction failed: {exc}")
            return []

        if reporter is not None:
            details: dict[str, Any] = {"skill_extraction": "llm_call", "method": method}
            if fallback:
                details["fallback"] = True
                reporter.record_llm_call(
                    model=self._config.model,
                    endpoint=self._config.api_base,
                    temperature=self._config.temperature,
                    details={
                        "skill_extraction": "fallback",
                        "from": "instructor",
                        "to": "direct",
                    },
                )
            reporter.record_llm_call(
                model=self._config.model,
                endpoint=self._config.api_base,
                temperature=self._config.temperature,
                details=details,
            )

        cleaned = self._clean_skills(raw_skills, description)

        if use_cache:
            self._save_cache(description, cleaned)

        return cleaned

    async def _call_llm(self, description: str) -> _ExtractionResult:
        """Call the LLM and return raw skill strings with method metadata."""
        quiet_litellm()

        model = f"openai/{self._config.model}" if self._config.api_base else self._config.model
        truncated = description[:MAX_DESCRIPTION_LENGTH]
        messages = [
            {"role": "system", "content": SKILL_SYSTEM_PROMPT},
            {"role": "user", "content": SKILL_USER_PROMPT.format(truncated)},
        ]
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

        try:
            client: Any = instructor.from_litellm(acompletion)
            response = await client.create(
                model=model,
                api_base=self._config.api_base,
                api_key=self._config.api_key,
                messages=messages,
                response_model=SkillExtractionOutput,
                max_retries=2,
                max_tokens=self._config.max_tokens,
                temperature=self._config.temperature,
                extra_body=extra_body,
            )
            logger.info("Extracted skills via instructor: %s", response.skills)
            return _ExtractionResult(
                skills=list(response.skills), method="instructor", fallback=False
            )
        except (
            ImportError,
            InstructorError,
            APIError,
            ValidationError,
            json.JSONDecodeError,
        ) as exc:
            logger.warning("Instructor skill extraction failed: %s", exc)
            logger.debug("Falling back to direct litellm for skill extraction")

        try:
            response = await acompletion(
                model=model,
                api_base=self._config.api_base,
                api_key=self._config.api_key,
                messages=messages,
                max_tokens=self._config.max_tokens,
                temperature=self._config.temperature,
                extra_body=extra_body,
            )
        except Exception as exc:
            logger.warning("Direct litellm skill extraction failed: %s", exc)
            return _ExtractionResult(skills=[], method="direct", fallback=True)

        if not response.choices:
            logger.warning("Direct litellm response had no choices")
            return _ExtractionResult(skills=[], method="direct", fallback=True)

        content = response.choices[0].message.content
        if content is None:
            logger.warning("Direct litellm response content was None")
            return _ExtractionResult(skills=[], method="direct", fallback=True)

        content = strip_thinking_process(content)
        skills = self._parse_skills_from_text(content)
        logger.info("Extracted skills via direct litellm: %s", skills)
        return _ExtractionResult(skills=skills, method="direct", fallback=True)

    def _parse_skills_from_text(self, text: str) -> list[str]:
        """Extract a list of skills from raw LLM text output."""
        if not text:
            return []

        try:
            data = json.loads(text.strip())
            if isinstance(data, dict):
                return [str(s) for s in data.get("skills", []) if s]
            if isinstance(data, list):
                return [str(s) for s in data if s]
        except json.JSONDecodeError:
            pass

        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            try:
                data = json.loads(brace_match.group(0))
                if isinstance(data, dict):
                    return [str(s) for s in data.get("skills", []) if s]
            except json.JSONDecodeError:
                pass

        return []

    def _clean_skills(self, skills: list[str], description: str) -> list[str]:
        """Normalize, filter, ground, deduplicate, and sort skills."""
        seen: set[str] = set()
        cleaned: list[str] = []
        for skill in skills:
            if not skill or not skill.strip():
                continue
            normalized = normalize_skill(skill)
            if not normalized or is_hard_negative(normalized):
                continue
            if not self._is_grounded(normalized, description):
                logger.warning("Dropping ungrounded skill '%s' from description", skill)
                continue
            if normalized not in seen:
                seen.add(normalized)
                cleaned.append(normalized)
        return sorted(cleaned)

    def _is_grounded(self, skill: str, description: str) -> bool:
        """Check whether a skill is actually present in the description text.

        Builds surface forms from the canonical name, its lower-case variant, and
        all known aliases from ``NORMALIZATION_MAP``. Multi-word forms must appear
        as a contiguous phrase; single-word forms need an exact token match unless
        the token is immediately followed by another word that continues a
        compound present in the description.
        """
        if not skill or not description:
            return False

        surface_forms = {skill, skill.lower()}
        for alias, canonical in NORMALIZATION_MAP.items():
            if canonical == skill:
                surface_forms.add(alias)
                surface_forms.add(alias.lower())

        desc_lower = description.lower()

        # Build known multi-word surface forms: explicit aliases plus compounds
        # that appear in the description where the second word is not a common
        # function word (so "React is" is not treated as a multi-word form).
        single_word_forms = {form for form in surface_forms if len(form.split()) == 1}
        multi_word_forms = {
            " ".join(form.split()).lower() for form in surface_forms if len(form.split()) > 1
        }
        token_re = re.compile(r"\b\w+(?:\.\w+)*\b")
        tokens = token_re.findall(description)
        for i, token in enumerate(tokens):
            if token.lower() not in {f.lower() for f in single_word_forms}:
                continue
            if i + 1 >= len(tokens):
                continue
            next_token = tokens[i + 1]
            if next_token.lower() in _STOPWORDS:
                continue
            multi_word_forms.add(f"{token.lower()} {next_token.lower()}")

        for form in surface_forms:
            if not form or not form.strip():
                continue
            stripped = form.strip()
            words = stripped.split()
            if len(words) > 1:
                if " ".join(words).lower() in desc_lower:
                    return True
            else:
                pattern = r"(?<!\w)" + re.escape(stripped.lower()) + r"(?!\w)"
                for match in re.finditer(pattern, desc_lower):
                    tail = description[match.end() :]
                    next_word_match = re.match(r"\s+(\w+(?:\.\w+)*)\b", tail)
                    if next_word_match:
                        next_word = next_word_match.group(1)
                        compound = f"{stripped.lower()} {next_word.lower()}"
                        if compound in multi_word_forms:
                            continue
                    return True
        return False
