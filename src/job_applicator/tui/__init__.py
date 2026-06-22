"""Full-screen terminal UI (the ``tui`` command / bare invocation).

A navigable home over the job-funnel store — the "home screen" that makes the CLI feel
like an app, with in-app actions (tailor / cover-letter / search / apply). Launching,
navigating, and filtering touch only local state; the account-touching actions
(search / apply) run behind explicit confirms.
"""

from __future__ import annotations

from job_applicator.tui.app import JobApplicatorApp, run_tui

__all__ = ["JobApplicatorApp", "run_tui"]
