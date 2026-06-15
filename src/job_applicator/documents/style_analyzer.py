"""Style analyzer - extracts writing patterns from example resumes/cover letters.

Refinements:
- Uses instructor for structured output (no manual JSON parsing)
- Persistent cache to avoid re-analysis
- Multi-document analysis for combined patterns
- Richer style dimensions
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, cast

from job_applicator.config import LLMConfig
from job_applicator.exceptions import LLMError
from job_applicator.models import StyleGuide
from job_applicator.utils.logging import get_logger
from job_applicator.utils.retry import async_retry

logger = get_logger("documents.style_analyzer")

ANALYSIS_PROMPT = """Analyze this text's writing style and return a JSON object.

Required fields:
- tone: overall tone
- sentence_structure: how sentences are built
- vocabulary_level: word complexity
- paragraph_style: how paragraphs are organized
- key_phrases: list of 3-5 characteristic phrases used
- avoid_phrases: list of 2-3 phrases NOT used
- power_words: list of 3-5 strong action verbs
- industry_jargon: list of domain-specific terms
- greeting_style: how the text opens
- closing_style: how the text ends
- use_of_metrics: how numbers are presented
- storytelling_approach: narrative style
- sentence_variety: mix of sentence patterns
- personal_touch: how personality shows
- formatting_notes: structural patterns
- sample_paragraph: one representative paragraph

Text to analyze:
---
{text}
---

Return ONLY the valid JSON object, no markdown, no thinking."""

SYSTEM_PROMPT = (
    "You are a writing style analyst. Return only valid JSON. No thinking, no markdown fences."
)


def extract_json_from_response(text: str) -> dict[str, object] | None:
    """Extract JSON from LLM response, handling various formats.

    Strategies:
    1. Direct JSON parse
    2. Extract from ```json code block
    3. Extract from ``` code block
    4. Find outermost { } braces
    5. Strip thinking markers and retry
    """
    if not text:
        return None

    # Strategy 1: Direct JSON parse
    try:
        return json.loads(text.strip())  # type: ignore[no-any-return]
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract from ```json code block
    json_block_match = re.search(r"```json\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if json_block_match:
        try:
            return json.loads(json_block_match.group(1))  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            pass

    # Strategy 3: Extract from ``` code block
    code_block_match = re.search(r"```\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            pass

    # Strategy 4: Find outermost { } braces
    brace_start = text.find("{")
    brace_end = text.rfind("}") + 1
    if brace_start >= 0 and brace_end > brace_start:
        try:
            return json.loads(text[brace_start:brace_end])  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            pass

    # Strategy 5: Strip thinking markers and retry
    cleaned = re.sub(r"Thinking Process:.*?(?=\{)", "", text, flags=re.DOTALL)
    if cleaned != text:
        try:
            return json.loads(cleaned.strip())  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            pass

    return None


class StyleAnalyzer:
    """Analyze writing style from example documents.

    Features:
    - Persistent cache to avoid re-analysis
    - Multi-document analysis for combined patterns
    - Richer style dimensions
    """

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._cache_dir = Path.home() / ".job-applicator" / "styles"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from text hash + model name."""
        content = f"{self._config.model}:{text}"
        return hashlib.md5(content.encode()).hexdigest()[:16]  # noqa: S324

    def _get_cache_path(self, text: str) -> Path:
        """Get cache file path for text."""
        key = self._get_cache_key(text)
        return self._cache_dir / f"{key}.json"

    @async_retry(max_attempts=2, base_delay=1.0, exceptions=(LLMError,))
    async def analyze(self, text: str, use_cache: bool = True) -> StyleGuide:
        """Analyze text and extract style patterns using LLM.

        Args:
            text: The text to analyze
            use_cache: Whether to check/save to cache
        """
        # Check cache first
        if use_cache:
            cache_path = self._get_cache_path(text)
            if cache_path.exists():
                try:
                    data = json.loads(cache_path.read_text())
                    style = StyleGuide(**data)
                    logger.info("Loaded style from cache: tone=%s", style.tone)
                    return style
                except Exception as e:
                    logger.debug("Cache miss: %s", e)

        # Analyze with LLM
        style = await self._analyze_with_llm(text)

        # Save to cache
        if use_cache:
            cache_path = self._get_cache_path(text)
            cache_path.write_text(style.model_dump_json())

        return style

    async def _analyze_with_llm(self, text: str) -> StyleGuide:
        """Perform the actual LLM analysis using instructor for structured output."""
        try:
            import instructor
            from litellm import acompletion

            model = f"openai/{self._config.model}" if self._config.api_base else self._config.model

            # Try instructor first (structured output with automatic retry)
            try:
                client: Any = instructor.from_litellm(acompletion)
                response = await client.create(
                    model=model,
                    api_base=self._config.api_base,
                    api_key=self._config.api_key,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": ANALYSIS_PROMPT.format(text=text[:3000])},
                    ],
                    response_model=StyleGuide,
                    max_retries=2,
                    max_tokens=self._config.max_tokens,
                    temperature=0.1,
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )
                logger.info("Analyzed writing style via instructor: tone=%s", response.tone)
                return cast(StyleGuide, response)
            except Exception:
                logger.debug("Instructor failed, falling back to direct litellm call")

            # Fallback: direct litellm call with manual JSON parsing
            response = await acompletion(
                model=model,
                api_base=self._config.api_base,
                api_key=self._config.api_key,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": ANALYSIS_PROMPT.format(text=text[:3000])},
                ],
                max_tokens=self._config.max_tokens,
                temperature=0.1,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )

            content = response.choices[0].message.content

            from job_applicator.utils.llm import strip_thinking_process

            content = strip_thinking_process(content)

            data = extract_json_from_response(content)

            if data:
                defaults = {
                    "tone": "professional",
                    "sentence_structure": "varied",
                    "vocabulary_level": "professional",
                    "paragraph_style": "clear structure",
                    "key_phrases": [],
                    "avoid_phrases": [],
                    "power_words": [],
                    "industry_jargon": [],
                    "greeting_style": "",
                    "closing_style": "",
                    "use_of_metrics": "",
                    "storytelling_approach": "",
                    "sentence_variety": "",
                    "personal_touch": "",
                    "formatting_notes": "",
                    "sample_paragraph": text[:200] if text else "",
                }
                for key, default in defaults.items():
                    if key not in data:
                        data[key] = default

                style = StyleGuide(**data)  # type: ignore[arg-type]
                logger.info("Analyzed writing style: tone=%s", style.tone)
                return style
            else:
                logger.warning("No valid JSON in response, using default style")
                return self._create_default_style(text)

        except Exception as exc:
            logger.warning("Style analysis failed: %s", exc)
            return self._create_default_style(text)

    def _create_default_style(self, text: str) -> StyleGuide:
        """Create a basic style guide from text analysis without LLM."""
        import re

        # Sentence analysis
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        avg_length = sum(len(s.split()) for s in sentences[:10]) / max(1, min(10, len(sentences)))

        # Word frequency for jargon detection
        words = re.findall(r"\b\w+\b", text.lower())
        word_freq: dict[str, int] = {}
        for w in words:
            if len(w) > 4:
                word_freq[w] = word_freq.get(w, 0) + 1

        # Find common phrases (bigrams)
        bigrams = []
        for i in range(len(words) - 1):
            if len(words[i]) > 3 and len(words[i + 1]) > 3:
                bigrams.append(f"{words[i]} {words[i + 1]}")

        # Most common bigrams as key phrases
        from collections import Counter

        common_phrases = [p for p, _ in Counter(bigrams).most_common(5)]

        return StyleGuide(
            tone="professional",
            sentence_structure=f"average {avg_length:.0f} words per sentence",
            vocabulary_level="professional with technical terms",
            paragraph_style="clear sections with headers",
            key_phrases=common_phrases if common_phrases else ["experience in", "skilled in"],
            avoid_phrases=["I think", "maybe"],
            power_words=["developed", "implemented", "led"],
            industry_jargon=[w for w, c in word_freq.items() if c > 1][:5],
            greeting_style="formal with name",
            closing_style="professional sign-off",
            use_of_metrics="mentions specific numbers",
            storytelling_approach="mix of narrative and bullets",
            sentence_variety="moderate variation",
            personal_touch="reserved and professional",
            formatting_notes="uses bullet points and headers",
            sample_paragraph=text[:200] + "..." if len(text) > 200 else text,
        )

    async def analyze_multiple(self, texts: list[str], use_cache: bool = True) -> StyleGuide:
        """Analyze multiple documents and extract combined style patterns.

        Args:
            texts: List of texts to analyze
            use_cache: Whether to check/save to cache

        Returns:
            Combined StyleGuide merging patterns from all documents
        """
        if not texts:
            return self._create_default_style("")

        if len(texts) == 1:
            return await self.analyze(texts[0], use_cache)

        # Analyze each document
        styles = []
        for text in texts:
            style = await self.analyze(text, use_cache)
            styles.append(style)

        # Merge styles
        return self._merge_styles(styles)

    def _merge_styles(self, styles: list[StyleGuide]) -> StyleGuide:
        """Merge multiple style guides into one combined guide.

        Merging strategy:
        - key_phrases, power_words, industry_jargon: union (deduplicated)
        - avoid_phrases: union
        - String fields: most common value, or first if tied
        - sample_paragraph: longest one
        """
        if not styles:
            return self._create_default_style("")

        if len(styles) == 1:
            return styles[0]

        # Collect all string field values
        from collections import Counter

        string_fields = [
            "tone",
            "sentence_structure",
            "vocabulary_level",
            "paragraph_style",
            "formatting_notes",
            "greeting_style",
            "closing_style",
            "use_of_metrics",
            "storytelling_approach",
            "sentence_variety",
            "personal_touch",
        ]

        field_values = {}
        for field in string_fields:
            values = [getattr(s, field) for s in styles if getattr(s, field)]
            if values:
                counter = Counter(values)
                field_values[field] = counter.most_common(1)[0][0]
            else:
                field_values[field] = ""

        # Merge list fields (union, deduplicated)
        def merge_lists(attr: str) -> list[str]:
            combined = []
            for s in styles:
                combined.extend(getattr(s, attr))
            return list(dict.fromkeys(combined))  # Preserve order, deduplicate

        # Find longest sample paragraph
        samples = [s.sample_paragraph for s in styles if s.sample_paragraph]
        longest_sample = max(samples, key=len, default="")

        return StyleGuide(
            **field_values,
            key_phrases=merge_lists("key_phrases"),
            avoid_phrases=merge_lists("avoid_phrases"),
            power_words=merge_lists("power_words"),
            industry_jargon=merge_lists("industry_jargon"),
            sample_paragraph=longest_sample,
        )

    def analyze_sync(self, text: str) -> StyleGuide:
        """Synchronous version for non-async contexts."""
        import asyncio

        return asyncio.run(self.analyze(text))

    def format_style_for_prompt(self, style: StyleGuide) -> str:
        """Format style guide into a prompt section for cover letter generation."""
        parts = [
            "Writing Style Guide (mimic this style):",
            f"- Tone: {style.tone}",
            f"- Sentence structure: {style.sentence_structure}",
            f"- Vocabulary: {style.vocabulary_level}",
            f"- Paragraph style: {style.paragraph_style}",
        ]

        if style.greeting_style:
            parts.append(f"- Greeting style: {style.greeting_style}")
        if style.closing_style:
            parts.append(f"- Closing style: {style.closing_style}")
        if style.use_of_metrics:
            parts.append(f"- Metrics: {style.use_of_metrics}")
        if style.storytelling_approach:
            parts.append(f"- Storytelling: {style.storytelling_approach}")
        if style.personal_touch:
            parts.append(f"- Personal touch: {style.personal_touch}")

        if style.key_phrases:
            parts.append(f"- Use phrases like: {', '.join(style.key_phrases[:5])}")

        if style.power_words:
            parts.append(f"- Use words like: {', '.join(style.power_words[:5])}")

        if style.avoid_phrases:
            parts.append(f"- Avoid phrases like: {', '.join(style.avoid_phrases)}")

        if style.sample_paragraph:
            parts.append(f'\nSample style to emulate:\n"""{style.sample_paragraph}"""')

        return "\n".join(parts)
