"""LLM-driven skill extraction from job descriptions.

Extracts canonical technical skills from a job description, normalizes them,
filters hard negatives, and runs a text-grounded hallucination guard so only
skills actually present in the description are returned.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import instructor
from instructor.core import InstructorError
from litellm import acompletion
from litellm.exceptions import APIError
from pydantic import BaseModel, Field, ValidationError

from job_applicator.config import LLMConfig
from job_applicator.exceptions import LLMError
from job_applicator.skills import NORMALIZATION_MAP, is_hard_negative, normalize_skill
from job_applicator.utils.llm import (
    LLMRuntime,
    litellm_model,
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

# Domain-general evidence-span variant (grounding_mode="evidence_span"): the model returns each
# skill with the exact source phrase, which we verify is a substring of the text. The prompt is
# ROLE-RELEVANCE-scoped — only skills the candidate must have, NOT the company's business, the job
# title, or tier labels (dogfood finding: a biotech JD grounded "protein engineering" from the
# company blurb; a SOC JD grounded its own title "Analyste SOC N2/N3"). Validated safe on
# security-firm JDs (it keeps SIEM/IDS/EDR, doesn't strip them as "company business"). The name may
# canonicalize away from the (verbatim) evidence — name↔evidence coherence is the deferred C check.
# No concrete example pair (the 4B copies example spans). See the semantic-grounding spec.
SKILL_SYSTEM_PROMPT_EVIDENCE = (
    "Extract ONLY the professional/technical skills, tools, technologies, methods, and "
    "certifications THE CANDIDATE must possess to perform THIS role — drawn from the "
    "responsibilities and requirements.\n\n"
    "For EACH skill return two things:\n"
    "- name: its canonical, widely recognized name.\n"
    "- evidence: the EXACT verbatim phrase, copied word-for-word from the text, that mentions "
    "it. Copy the characters as they appear; do NOT paraphrase or invent.\n\n"
    "STRICTLY EXCLUDE — never return any of these as a skill:\n"
    "- the company's own products, services, or industry/business (what the company sells or "
    "makes);\n"
    "- the job title or role name itself (e.g. 'SOC Analyst', 'Cybersecurity Specialist');\n"
    "- seniority or tier labels (junior, senior, intermediate, L1, N1/N2/N3, Tier 2);\n"
    "- soft skills, work arrangement, location, compensation;\n"
    "- single letters or sentence fragments.\n\n"
    '- Return ONLY a JSON object: {"skills": [{"name": "...", "evidence": "..."}]}.'
)

SKILL_USER_PROMPT = "{}"

MAX_DESCRIPTION_LENGTH = 1500

# Adjacent tokens that look like version numbers ("8", "18", "3.x", "3.11", "v3.11", "v2")
# should not turn a single-word skill into a rejected compound.
_VERSION_LIKE_RE = re.compile(r"^v?\d+(?:\.\d+)*(?:[a-z]|\.x)?$", re.IGNORECASE)


def _is_version_like(token: str) -> bool:
    """Return True when ``token`` is purely numeric/version-like."""
    return bool(_VERSION_LIKE_RE.match(token))


# Typographic apostrophe/quote variants → ASCII. Common in French JDs ("l'identification" with the
# U+2019 right-single-quote vs U+0027), which otherwise fail verbatim span verification.
_SPAN_QUOTE_MAP = str.maketrans(
    {
        chr(0x2018): "'",  # LEFT SINGLE QUOTATION MARK
        chr(0x2019): "'",  # RIGHT SINGLE QUOTATION MARK (the French l'X case)
        chr(0x02BC): "'",  # MODIFIER LETTER APOSTROPHE
        chr(0x00B4): "'",  # ACUTE ACCENT (sometimes typed as an apostrophe)
        chr(0x0060): "'",  # GRAVE ACCENT
        chr(0x201C): '"',  # LEFT DOUBLE QUOTATION MARK
        chr(0x201D): '"',  # RIGHT DOUBLE QUOTATION MARK
    }
)


def _normalize_span_text(text: str) -> str:
    """Cosmetic normalization for evidence-span verification: NFC-canonicalize accents, unify
    apostrophe/quote variants, and collapse whitespace runs (incl. newlines) to single spaces.

    Applied to BOTH the span and the description, so it can only reconcile encoding/spacing
    variants — it can NEVER make a span match evidence that is not present modulo those variants
    (a paraphrased or reformatted span still fails). Keeps the honesty of the substring check while
    fixing the genuine curly-apostrophe / line-wrap comparison bug.
    """
    text = unicodedata.normalize("NFC", text).translate(_SPAN_QUOTE_MAP)
    return " ".join(text.split()).lower()


def _phrase_in_description(phrase: str, description: str) -> bool:
    """Return True when ``phrase`` appears as a whole-phrase token in ``description``.

    Both sides are cosmetically normalized (:func:`_normalize_span_text`) so a span that quotes the
    JD verbatim still verifies across curly-apostrophe / accent-encoding / line-wrap differences,
    without loosening the substring requirement.
    """
    norm_phrase = _normalize_span_text(phrase)
    pattern = r"(?<!\w)" + re.escape(norm_phrase) + r"(?!\w)"
    return bool(re.search(pattern, _normalize_span_text(description)))


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

# Job-description prose words that commonly follow a skill but do not form a
# multi-word skill compound. Combined with ``_STOPWORDS`` to avoid false
# rejections when a lowercase compound continuation is actually just prose.
_PROSE_STOPWORDS: frozenset[str] = frozenset(
    {
        "experience",
        "experienced",
        "experiences",
        "required",
        "requirement",
        "requirements",
        "preferred",
        "preferreds",
        "qualification",
        "qualifications",
        "responsibility",
        "responsibilities",
        "skill",
        "skills",
        "knowledge",
        "familiarity",
        "proficiency",
        "expertise",
        "background",
        "ability",
        "abilities",
        "year",
        "years",
        "plus",
        "nice",
        "strong",
        "solid",
        "deep",
        "working",
        "using",
        "based",
        "such",
        "including",
        "particularly",
        "especially",
        "relevant",
        "role",
        "roles",
        "position",
        "positions",
        "job",
        "jobs",
        "team",
        "teams",
        "project",
        "projects",
        "engineer",
        "engineers",
        "developer",
        "developers",
        "programmer",
        "programmers",
        "candidate",
        "candidates",
        "applicant",
        "applicants",
        "well",
        "good",
        "excellent",
        "proven",
        "demonstrated",
        "extensive",
        "practical",
    }
)


# Genuine multi-word skills (lowercased), taken from the canonical VALUES of the
# normalization map — e.g. "react native", "spring boot", "machine learning".
# Grounding treats "<skill> <next-word>" as one of these distinct compound skills
# (and so rejects the bare first word) ONLY when the pair is in this set. A skill
# followed by an ordinary noun ("kubernetes platform", "python automation") is NOT
# a compound, so the bare skill stays grounded. Built from canonical values, not
# alias keys, so an alias that normalizes back to the bare skill ("docker
# container" → Docker) never disqualifies it.
_KNOWN_MULTIWORD_SKILLS: frozenset[str] = frozenset(
    " ".join(canonical.lower().split())
    for canonical in NORMALIZATION_MAP.values()
    if len(canonical.split()) > 1
)


class SkillExtractionOutput(BaseModel):
    """Structured output for LLM skill extraction."""

    skills: list[str] = Field(description="Canonical technical skills required by the job")
    model_config = {"extra": "forbid"}


class SkillEvidence(BaseModel):
    """One extracted skill plus the verbatim source phrase that grounds it."""

    name: str = Field(description="Canonical name of the skill")
    evidence: str = Field(description="Exact verbatim phrase copied from the text that mentions it")
    model_config = {"extra": "forbid"}


class SkillExtractionOutputV2(BaseModel):
    """Evidence-grounded structured output: each skill carries its source phrase."""

    skills: list[SkillEvidence] = Field(
        description="Skills present in the text, each with evidence"
    )
    model_config = {"extra": "forbid"}


@dataclass
class _ExtractionResult:
    """Result of an LLM skill-extraction attempt."""

    skills: list[str]
    method: str
    fallback: bool
    grounded: bool = False  # True iff evidence-span verification already grounded these names


class LLMSkillExtractor:
    """Extract technical skills from a job description using an LLM.

    Caches results persistently by a hash of the model name and description to
    avoid duplicate LLM calls. All returned skills are normalized, filtered for
    hard negatives, and verified against the original description.
    """

    def __init__(self, config: LLMConfig, *, grounding_mode: str = "evidence_span") -> None:
        self._config = config
        self._grounding_mode = grounding_mode
        self._cache_dir = Path.home() / ".job-applicator" / "skill-extraction"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, description: str) -> str:
        """Generate a cache key from model name, grounding mode, and description.

        The grounding mode is part of the key so keyword- and evidence-span-grounded results
        never collide — switching modes must not return the other mode's cached skills."""
        content = f"{self._config.model}\x00{self._grounding_mode}\x00{description}"
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
                return self._clean_skills(
                    cached, description, already_grounded=self._grounding_mode == "evidence_span"
                )

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
        grounded: bool = False
        try:
            result = await runtime.run(lambda _prev: self._call_llm(description))
            raw_skills = result.skills
            method = result.method
            fallback = result.fallback
            grounded = result.grounded
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
            # RAISE the typed error — never return [] on an LLM failure. An empty skill list
            # that actually means "the endpoint was down" is indistinguishable from "this job
            # genuinely lists no skills", and silently degrades every match downstream.
            raise error from exc

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

        cleaned = self._clean_skills(raw_skills, description, already_grounded=grounded)

        # Do NOT cache a DEGRADED (keyword-grounded) result under the evidence_span key — it
        # would persist as a stale "span-verified" hit after the endpoint recovers (the
        # cross-mode contamination the mode-in-key change prevents, one level down). Re-extract.
        if use_cache and (self._grounding_mode != "evidence_span" or grounded):
            self._save_cache(description, cleaned)

        return cleaned

    async def _call_llm(self, description: str) -> _ExtractionResult:
        """Call the LLM and return raw skill strings with method metadata."""
        quiet_litellm()

        if self._grounding_mode == "evidence_span":
            try:
                return await self._call_llm_evidence_span(description)
            except (
                ImportError,
                InstructorError,
                APIError,
                ValidationError,
                json.JSONDecodeError,
            ) as exc:
                # Structured evidence-span output unavailable (e.g. a vLLM without a tool-call
                # parser). Degrade to keyword grounding — result.grounded stays False so
                # _clean_skills still keyword-grounds these names. Not a masked failure.
                logger.warning(
                    "Evidence-span extraction failed (%s); degrading to keyword grounding", exc
                )

        model = litellm_model(self._config)
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
            # Transport failure on the direct fallback → raise, never return [] (a masked failure).
            raise llm_call_error(exc, self._config.api_base or "") from exc

        if not response.choices:
            raise LLMError("Direct litellm skill-extraction response had no choices")

        content = response.choices[0].message.content
        if content is None:
            raise LLMError("Direct litellm skill-extraction response content was None")

        # A SUCCESSFUL call that parses to no skills is legitimate (the job may list none) — only
        # the failure branches above raise.
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

    async def _call_llm_evidence_span(self, description: str) -> _ExtractionResult:
        """Evidence-span extraction: the model returns each skill with the exact source phrase,
        and we keep only skills whose span verifies as a substring of the text. Domain-general —
        no normalization map or keyword heuristics. Raises on structured-output failure so the
        caller can degrade to keyword grounding."""
        truncated = description[:MAX_DESCRIPTION_LENGTH]
        client: Any = instructor.from_litellm(acompletion)
        response = await client.create(
            model=litellm_model(self._config),
            api_base=self._config.api_base,
            api_key=self._config.api_key,
            messages=[
                {"role": "system", "content": SKILL_SYSTEM_PROMPT_EVIDENCE},
                {"role": "user", "content": SKILL_USER_PROMPT.format(truncated)},
            ],
            response_model=SkillExtractionOutputV2,
            max_retries=2,
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        grounded = self._verify_spans([(s.name, s.evidence) for s in response.skills], description)
        logger.info("Extracted skills via evidence-span: %s", grounded)
        return _ExtractionResult(
            skills=grounded, method="evidence_span", fallback=False, grounded=True
        )

    @staticmethod
    def _span_grounded(span: str, description: str) -> bool:
        """True if ``span`` occurs in ``description`` as a WHOLE-TOKEN match.

        Normalization lowercases, collapses whitespace, and replaces each punctuation run with
        a boundary sentinel (``\\x00``, never in real text) so a clause boundary
        ("part-time. series") cannot fuse two spans. The match is then anchored with word
        boundaries so a short span cannot ground INSIDE a larger word ("Ada" must not match
        "adaptable", "React" not "reactive") — keeping this guard at least as strict as the
        keyword grounding it replaces."""

        def norm(s: str) -> str:
            return " ".join(re.sub(r"[^\w\s]+", " \x00 ", s.lower()).split())

        nspan = norm(span)
        if not nspan or nspan == "\x00":
            return False
        return re.search(r"(?<!\w)" + re.escape(nspan) + r"(?!\w)", norm(description)) is not None

    def _verify_spans(self, pairs: list[tuple[str, str]], description: str) -> list[str]:
        """Keep skill names whose evidence span verifies as a whole-token match in the text;
        drop fabricated/paraphrased spans. Order-preserving, de-duplicated by lower-cased name.

        Phase-1 limitation (by design): this verifies the SPAN is in the text but trusts the
        model's canonical NAME for that span. A name/evidence MISMATCH (name "Java" for span
        "JavaScript") is not caught — no string check can separate that from a legitimate
        canonicalization (name "PostgreSQL" for span "Postgres"); both are prefix relations.
        Catching it needs name↔span embedding coherence — a Phase-2 check (see the spec)."""
        kept: list[str] = []
        seen: set[str] = set()
        for name, evidence in pairs:
            if not name or not name.strip():
                continue
            if not self._span_grounded(evidence, description):
                logger.warning(
                    "Dropping skill %r — evidence span %r not found in text", name, evidence
                )
                continue
            key = name.strip().lower()
            if key not in seen:
                seen.add(key)
                kept.append(name.strip())
        return kept

    def _clean_skills(
        self, skills: list[str], description: str, *, already_grounded: bool = False
    ) -> list[str]:
        """Normalize, filter, (ground,) deduplicate, and sort skills.

        ``already_grounded`` (evidence-span mode) skips the keyword grounding check — the spans
        were verified upstream, and keyword grounding would wrongly drop the cross-domain skills
        evidence-span exists to keep."""
        seen: set[str] = set()
        cleaned: list[str] = []
        for skill in skills:
            if not skill or not skill.strip():
                continue
            normalized = normalize_skill(skill)
            if not normalized:
                continue
            stripped = normalized.strip()
            # Soft-skill traits (the hard-negative blocklist) are always dropped. The len<=2
            # sub-rule, though, is relaxed in evidence-span mode: a span-verified short skill
            # (R, Go) is real and must survive, whereas keyword mode drops it as noise.
            if len(stripped) <= 2:
                if not already_grounded:
                    continue
            elif is_hard_negative(stripped):
                continue
            if not already_grounded and not self._is_grounded(normalized, description):
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
        as a whole-phrase token (word boundaries on both ends). Single-word forms
        need an exact token match unless the next token in the description
        continues a compound; common prose words are ignored so that phrases like
        "React experience" do not reject the valid single-word skill ``React``.
        """
        if not skill or not description:
            return False

        surface_forms = {skill, skill.lower()}
        for alias, canonical in NORMALIZATION_MAP.items():
            if canonical == skill:
                surface_forms.add(alias)
                surface_forms.add(alias.lower())

        desc_lower = description.lower()
        non_compound = _STOPWORDS | _PROSE_STOPWORDS

        # Build known multi-word surface forms: explicit aliases plus compounds
        # that appear in the description where the second word is not a common
        # function/prose word (so "React is" and "React experience" are not
        # treated as multi-word forms).
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
            if next_token.lower() in non_compound or _is_version_like(next_token):
                continue
            # Only a KNOWN multi-word skill (e.g. "react native") makes the bare
            # first word a different skill and thus disqualifies it; an ordinary
            # following noun ("kubernetes platform", "python automation") does not.
            compound = f"{token.lower()} {next_token.lower()}"
            if compound in _KNOWN_MULTIWORD_SKILLS:
                multi_word_forms.add(compound)

        for form in surface_forms:
            if not form or not form.strip():
                continue
            stripped = form.strip()
            words = stripped.split()
            if len(words) > 1:
                if _phrase_in_description(" ".join(words), desc_lower):
                    return True
            else:
                pattern = r"(?<!\w)" + re.escape(stripped.lower()) + r"(?!\w)"
                for match in re.finditer(pattern, desc_lower):
                    tail = description[match.end() :]
                    next_word_match = re.match(r"\s+(\w+(?:\.\w+)*)\b", tail)
                    if next_word_match:
                        next_word = next_word_match.group(1)
                        if _is_version_like(next_word):
                            return True
                        compound = f"{stripped.lower()} {next_word.lower()}"
                        if compound in multi_word_forms:
                            continue
                    return True
        return False
