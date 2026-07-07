#!/usr/bin/env python
"""Compatibility wrapper for the generated document quality gate."""

from __future__ import annotations

import sys

from job_applicator.documents import quality_eval as _quality_eval
from job_applicator.documents.quality_eval import (  # noqa: F401
    PacketQualityReport,
    QualityReport,
    assess_cover_letter,
    assess_packet_case,
    assess_packet_set,
    assess_resume,
    main,
)

_run_packet_set = _quality_eval._run_packet_set
_target_mention_score = _quality_eval._target_mention_score


if __name__ == "__main__":
    sys.exit(main())
