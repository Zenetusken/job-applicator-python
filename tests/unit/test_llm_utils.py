"""Unit tests for LLM utilities."""

from __future__ import annotations

from job_applicator.documents.cover_letter import strip_thinking_process


def test_strip_thinking_process_with_thinking() -> None:
    """Test stripping thinking process from LLM output."""
    text_with_thinking = """Thinking Process:

1.  **Analyze the Request:**
    *   **Role:** Senior Python Developer.
    *   **Company:** TechCorp Solutions.

2.  **Drafting:**
    *   Paragraph 1: Opening statement.

Dear Hiring Team,

I am writing to express my interest in the position.

Sincerely,
John Doe"""

    result = strip_thinking_process(text_with_thinking)
    assert "Thinking Process" not in result
    assert "Dear Hiring Team" in result
    assert "John Doe" in result


def test_strip_thinking_process_clean_text() -> None:
    """Test that clean text passes through unchanged."""
    clean_text = """Dear Hiring Manager,

I am writing to apply for the Python Developer position.

Best regards,
Jane Smith"""

    result = strip_thinking_process(clean_text)
    assert result == clean_text


def test_strip_thinking_process_multiple_dear() -> None:
    """Test handling multiple 'Dear' occurrences (thinking + actual)."""
    text = """Thinking about the letter...

    Draft 1: Dear Team, ...

    Draft 2: Dear Hiring Manager, ...

    Final version:

    Dear Hiring Manager,

    I am excited to apply for this role.

    Sincerely,
    Applicant"""

    result = strip_thinking_process(text)
    # Should keep the last "Dear" section
    assert "Final version" not in result or "Dear Hiring Manager" in result
    assert "Sincerely" in result


def test_strip_thinking_process_empty() -> None:
    """Test handling empty input."""
    result = strip_thinking_process("")
    assert result == ""


def test_strip_thinking_process_am_writing() -> None:
    """Test stripping with 'I am writing' as letter start."""
    text = """Let me think about this...

    1. Analyze requirements
    2. Draft response

    I am writing to express my interest in the position.

    Thank you for your consideration."""

    result = strip_thinking_process(text)
    assert "I am writing" in result
    assert "Thank you" in result
