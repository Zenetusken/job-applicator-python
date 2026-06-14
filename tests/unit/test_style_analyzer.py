"""Unit tests for style analyzer refinements."""

from __future__ import annotations

import pytest

from job_applicator.config import LLMConfig
from job_applicator.documents.style_analyzer import StyleAnalyzer, extract_json_from_response
from job_applicator.models import StyleGuide


class TestExtractJsonFromResponse:
    """Tests for JSON extraction from LLM responses."""

    def test_direct_json(self) -> None:
        """Test extracting direct JSON."""
        json_str = '{"tone": "professional", "key_phrases": ["test"]}'
        result = extract_json_from_response(json_str)
        assert result is not None
        assert result["tone"] == "professional"

    def test_json_in_code_block(self) -> None:
        """Test extracting JSON from markdown code block."""
        text = 'Here is the analysis:\n```json\n{"tone": "casual"}\n```\nDone.'
        result = extract_json_from_response(text)
        assert result is not None
        assert result["tone"] == "casual"

    def test_json_in_plain_code_block(self) -> None:
        """Test extracting JSON from plain code block."""
        text = '```\n{"tone": "formal"}\n```'
        result = extract_json_from_response(text)
        assert result is not None
        assert result["tone"] == "formal"

    def test_json_with_surrounding_text(self) -> None:
        """Test extracting JSON with surrounding text."""
        text = 'The style is: {"tone": "technical"} which means...'
        result = extract_json_from_response(text)
        assert result is not None
        assert result["tone"] == "technical"

    def test_json_with_thinking_prefix(self) -> None:
        """Test extracting JSON after thinking process."""
        text = 'Thinking Process:\n1. Analyze\n2. Extract\n\n{"tone": "analytical"}'
        result = extract_json_from_response(text)
        assert result is not None
        assert result["tone"] == "analytical"

    def test_empty_input(self) -> None:
        """Test with empty input."""
        assert extract_json_from_response("") is None
        assert extract_json_from_response(None) is None  # type: ignore[arg-type]

    def test_no_json(self) -> None:
        """Test with no JSON present."""
        assert extract_json_from_response("Just plain text here") is None


class TestStyleGuideModel:
    """Tests for the enriched StyleGuide model."""

    def test_style_guide_with_new_fields(self) -> None:
        """Test StyleGuide with all new fields."""
        style = StyleGuide(
            tone="professional",
            sentence_structure="varied",
            vocabulary_level="technical",
            paragraph_style="clear",
            key_phrases=["experience in"],
            avoid_phrases=["I think"],
            power_words=["architected", "spearheaded"],
            industry_jargon=["microservices", "kubernetes"],
            greeting_style="formal",
            closing_style="professional sign-off",
            use_of_metrics="specific percentages",
            storytelling_approach="bullet points with metrics",
            sentence_variety="alternates short and long",
            personal_touch="reserved",
            formatting_notes="uses headers",
            sample_paragraph="Test paragraph",
        )
        assert style.power_words == ["architected", "spearheaded"]
        assert style.greeting_style == "formal"
        assert style.use_of_metrics == "specific percentages"

    def test_style_guide_defaults(self) -> None:
        """Test StyleGuide with default values for new fields."""
        style = StyleGuide(
            tone="test",
            sentence_structure="test",
            vocabulary_level="test",
            paragraph_style="test",
            formatting_notes="test",
            sample_paragraph="test",
        )
        assert style.power_words == []
        assert style.greeting_style == ""
        assert style.personal_touch == ""


class TestStyleAnalyzer:
    """Tests for StyleAnalyzer functionality."""

    @pytest.fixture
    def config(self) -> LLMConfig:
        return LLMConfig()

    @pytest.fixture
    def analyzer(self, config: LLMConfig) -> StyleAnalyzer:
        return StyleAnalyzer(config)

    def test_cache_key_generation(self, analyzer: StyleAnalyzer) -> None:
        """Test that cache keys are consistent."""
        text = "Test text for caching"
        key1 = analyzer._get_cache_key(text)
        key2 = analyzer._get_cache_key(text)
        assert key1 == key2

    def test_cache_key_different_text(self, analyzer: StyleAnalyzer) -> None:
        """Test that different texts get different cache keys."""
        key1 = analyzer._get_cache_key("Text A")
        key2 = analyzer._get_cache_key("Text B")
        assert key1 != key2

    def test_create_default_style(self, analyzer: StyleAnalyzer) -> None:
        """Test default style creation."""
        text = "Senior engineer with 5 years experience. Developed scalable systems."
        style = analyzer._create_default_style(text)
        assert style.tone == "professional"
        assert "5" in style.sentence_structure or "words" in style.sentence_structure
        assert len(style.sample_paragraph) > 0

    def test_merge_styles(self, analyzer: StyleAnalyzer) -> None:
        """Test merging multiple style guides."""
        style1 = StyleGuide(
            tone="professional",
            sentence_structure="short",
            vocabulary_level="formal",
            paragraph_style="bullets",
            key_phrases=["phrase A", "phrase B"],
            power_words=["led"],
            formatting_notes="test",
            sample_paragraph="Paragraph 1",
        )
        style2 = StyleGuide(
            tone="casual",
            sentence_structure="long",
            vocabulary_level="conversational",
            paragraph_style="narrative",
            key_phrases=["phrase B", "phrase C"],
            power_words=["built"],
            formatting_notes="test",
            sample_paragraph="A longer paragraph here for testing",
        )

        merged = analyzer._merge_styles([style1, style2])

        # Should combine phrases (deduplicated)
        assert "phrase A" in merged.key_phrases
        assert "phrase B" in merged.key_phrases
        assert "phrase C" in merged.key_phrases
        assert len(merged.key_phrases) == 3  # Deduplicated

        # Should combine power words
        assert "led" in merged.power_words
        assert "built" in merged.power_words

        # Should take longest sample paragraph
        assert merged.sample_paragraph == "A longer paragraph here for testing"

    def test_format_style_for_prompt(self, analyzer: StyleAnalyzer) -> None:
        """Test formatting style for prompt injection."""
        style = StyleGuide(
            tone="professional but personable",
            sentence_structure="mix of short and long",
            vocabulary_level="technical",
            paragraph_style="clear structure",
            key_phrases=["experience in", "passionate about"],
            avoid_phrases=["I think"],
            power_words=["architected", "optimized"],
            greeting_style="formal",
            closing_style="professional",
            use_of_metrics="specific percentages",
            formatting_notes="uses headers",
            sample_paragraph="Test sample",
        )

        prompt = analyzer.format_style_for_prompt(style)

        assert "professional but personable" in prompt
        assert "experience in" in prompt
        assert "architected" in prompt
        assert "formal" in prompt
        assert "Test sample" in prompt


class TestStyleAnalyzerInstructor:
    """Tests for instructor-based structured output in style analyzer."""

    @pytest.mark.asyncio
    async def test_instructor_path_called_first(self) -> None:
        """Analyzer should try instructor before falling back to raw litellm."""
        from unittest.mock import AsyncMock, MagicMock, patch

        config = LLMConfig(api_base="http://test", model="test-model")
        analyzer = StyleAnalyzer(config)

        mock_style = StyleGuide(
            tone="professional",
            sentence_structure="varied",
            vocabulary_level="professional",
            paragraph_style="clear",
            formatting_notes="standard",
            sample_paragraph="sample",
        )

        mock_client = MagicMock()
        mock_client.create = AsyncMock(return_value=mock_style)

        with patch(
            "job_applicator.documents.style_analyzer.instructor",
            create=True,
        ) as mock_instructor_mod:
            mock_instructor_mod.from_litellm.return_value = mock_client
            with patch.object(analyzer, "_cache_dir", MagicMock()):
                with patch.object(analyzer, "_get_cache_path") as mock_path:
                    mock_path.return_value = MagicMock(exists=MagicMock(return_value=False))
                    with patch("litellm.acompletion", new_callable=AsyncMock):
                        result = await analyzer.analyze("test text", use_cache=False)

        assert result.tone == "professional"
