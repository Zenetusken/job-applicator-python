"""Shared selector registry for live board health probes."""

from __future__ import annotations

from job_applicator.applicators.linkedin import (
    _ADVANCE_BUTTON_SELECTORS,
    _SUBMIT_BUTTON_SELECTORS,
    COVER_LETTER_FIELD_SELECTOR,
    EASY_APPLY_BUTTON_SELECTOR,
    MODAL_TITLE_SELECTORS,
    REQUIRED_FIELD_SELECTOR,
    RESUME_FILE_INPUT_SELECTOR,
    RESUME_UPLOAD_BUTTON_SELECTOR,
)
from job_applicator.models import JobBoard, SelectorProbe, SelectorProbeScope
from job_applicator.scrapers.indeed import (
    INDEED_CARD_SELECTORS,
    INDEED_COMPANY_SELECTOR,
    INDEED_DESC_SELECTORS,
    INDEED_LOCATION_SELECTOR,
    INDEED_SALARY_SELECTOR,
    INDEED_SNIPPET_SELECTORS,
    INDEED_TITLE_SELECTOR,
)
from job_applicator.scrapers.linkedin import (
    LINKEDIN_DESCRIPTION_SELECTORS,
    LINKEDIN_DESCRIPTION_SHOW_MORE_SELECTORS,
    LINKEDIN_SEARCH_CARD_SELECTORS,
    LINKEDIN_SEARCH_COMPANY_SELECTOR,
    LINKEDIN_SEARCH_LOCATION_SELECTOR,
    LINKEDIN_SEARCH_SALARY_SELECTOR,
    LINKEDIN_SEARCH_TITLE_SELECTOR,
)

SEARCH_SURFACE = "search"
APPLY_SURFACE = "apply"


def _probe(
    board: JobBoard,
    surface: str,
    name: str,
    selectors: tuple[str, ...],
    *,
    required: bool,
    scope: SelectorProbeScope = "page",
) -> SelectorProbe:
    return SelectorProbe(
        board=board,
        surface=surface,
        name=name,
        selector=", ".join(selectors),
        selectors=list(selectors),
        required=required,
        scope=scope,
    )


LINKEDIN_SEARCH_PROBES = (
    _probe(
        JobBoard.LINKEDIN,
        SEARCH_SURFACE,
        "job card containers",
        LINKEDIN_SEARCH_CARD_SELECTORS,
        required=True,
    ),
    _probe(
        JobBoard.LINKEDIN,
        SEARCH_SURFACE,
        "title link",
        (LINKEDIN_SEARCH_TITLE_SELECTOR,),
        required=True,
        scope="first_card",
    ),
    _probe(
        JobBoard.LINKEDIN,
        SEARCH_SURFACE,
        "company",
        (LINKEDIN_SEARCH_COMPANY_SELECTOR,),
        required=False,
        scope="first_card",
    ),
    _probe(
        JobBoard.LINKEDIN,
        SEARCH_SURFACE,
        "location",
        (LINKEDIN_SEARCH_LOCATION_SELECTOR,),
        required=False,
        scope="first_card",
    ),
    _probe(
        JobBoard.LINKEDIN,
        SEARCH_SURFACE,
        "salary",
        (LINKEDIN_SEARCH_SALARY_SELECTOR,),
        required=False,
        scope="first_card",
    ),
    _probe(
        JobBoard.LINKEDIN,
        SEARCH_SURFACE,
        "description content",
        LINKEDIN_DESCRIPTION_SELECTORS,
        required=False,
    ),
    _probe(
        JobBoard.LINKEDIN,
        SEARCH_SURFACE,
        "description show more",
        LINKEDIN_DESCRIPTION_SHOW_MORE_SELECTORS,
        required=False,
    ),
)

LINKEDIN_APPLY_PROBES = (
    _probe(
        JobBoard.LINKEDIN,
        APPLY_SURFACE,
        "Easy Apply button",
        (EASY_APPLY_BUTTON_SELECTOR,),
        required=True,
    ),
    _probe(
        JobBoard.LINKEDIN,
        APPLY_SURFACE,
        "advance buttons",
        _ADVANCE_BUTTON_SELECTORS,
        required=False,
    ),
    _probe(
        JobBoard.LINKEDIN,
        APPLY_SURFACE,
        "submit buttons",
        _SUBMIT_BUTTON_SELECTORS,
        required=False,
    ),
    _probe(
        JobBoard.LINKEDIN,
        APPLY_SURFACE,
        "cover-letter field",
        (COVER_LETTER_FIELD_SELECTOR,),
        required=False,
    ),
    _probe(
        JobBoard.LINKEDIN,
        APPLY_SURFACE,
        "resume file input",
        (RESUME_FILE_INPUT_SELECTOR,),
        required=False,
    ),
    _probe(
        JobBoard.LINKEDIN,
        APPLY_SURFACE,
        "resume upload button",
        (RESUME_UPLOAD_BUTTON_SELECTOR,),
        required=False,
    ),
    _probe(
        JobBoard.LINKEDIN,
        APPLY_SURFACE,
        "modal title",
        MODAL_TITLE_SELECTORS,
        required=False,
    ),
    _probe(
        JobBoard.LINKEDIN,
        APPLY_SURFACE,
        "required fields",
        (REQUIRED_FIELD_SELECTOR,),
        required=False,
    ),
)

INDEED_SEARCH_PROBES = (
    _probe(
        JobBoard.INDEED,
        SEARCH_SURFACE,
        "job card containers",
        INDEED_CARD_SELECTORS,
        required=True,
    ),
    _probe(
        JobBoard.INDEED,
        SEARCH_SURFACE,
        "title link",
        (INDEED_TITLE_SELECTOR,),
        required=True,
        scope="first_card",
    ),
    _probe(
        JobBoard.INDEED,
        SEARCH_SURFACE,
        "company",
        (INDEED_COMPANY_SELECTOR,),
        required=False,
        scope="first_card",
    ),
    _probe(
        JobBoard.INDEED,
        SEARCH_SURFACE,
        "location",
        (INDEED_LOCATION_SELECTOR,),
        required=False,
        scope="first_card",
    ),
    _probe(
        JobBoard.INDEED,
        SEARCH_SURFACE,
        "salary",
        (INDEED_SALARY_SELECTOR,),
        required=False,
        scope="first_card",
    ),
    _probe(
        JobBoard.INDEED,
        SEARCH_SURFACE,
        "snippet",
        INDEED_SNIPPET_SELECTORS,
        required=False,
        scope="first_card",
    ),
    _probe(
        JobBoard.INDEED,
        SEARCH_SURFACE,
        "description pane",
        INDEED_DESC_SELECTORS,
        required=False,
    ),
)


def selector_probes(board: JobBoard, surface: str) -> tuple[SelectorProbe, ...]:
    """Return selector probes for a supported board/surface pair."""
    key = (board, surface)
    registry: dict[tuple[JobBoard, str], tuple[SelectorProbe, ...]] = {
        (JobBoard.LINKEDIN, SEARCH_SURFACE): LINKEDIN_SEARCH_PROBES,
        (JobBoard.LINKEDIN, APPLY_SURFACE): LINKEDIN_APPLY_PROBES,
        (JobBoard.INDEED, SEARCH_SURFACE): INDEED_SEARCH_PROBES,
    }
    return registry.get(key, ())
