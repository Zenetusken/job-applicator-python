"""Shared constructors for typed test doubles."""

from __future__ import annotations

from job_applicator.models import CoverLetterOverlay, SourceBackedStatement


def cover_letter_overlay() -> CoverLetterOverlay:
    """Return a valid minimal cover-letter evidence contract for mocked generators."""

    return CoverLetterOverlay(
        body_sentences=[
            SourceBackedStatement(
                text=f"Source-backed statement {index}.",
                fact_ids=[f"SRC-{index:03d}"],
            )
            for index in range(1, 4)
        ],
        source_body_sha256="a" * 64,
        source_language="en",
    )
