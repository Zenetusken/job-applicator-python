"""Full-screen terminal UI (the ``tui`` command / bare invocation).

A navigable, read-only browser over the job-funnel store — the "home screen" that
makes the CLI feel like an app. Account-safe: launching it touches only local state
(the SQLite store), never the network, a browser, or the LLM.
"""

from __future__ import annotations

from job_applicator.tui.app import JobApplicatorApp, run_tui

__all__ = ["JobApplicatorApp", "run_tui"]
