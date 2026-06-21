"""Shared Rich consoles — one stdout + one stderr instance across the CLI and its
extracted helpers.

``console`` carries data/reports (stdout); ``err_console`` carries CLI command-level
runtime errors — the messages that precede a non-zero exit — on stderr, so ``--json``
consumers and shell redirects get clean stdout while failures still surface. Route every
command-level fatal error to ``err_console``. Interactive workflow-loop feedback (retry
prompts, "invalid choice", per-attempt status) deliberately stays on ``console`` since it
interleaves with the user's TTY prompts. Consequence (by design): the same text — e.g.
"LLM error: …" — can land on stderr from a command handler but on stdout from the
interactive workflow loop. If a workflow ever gains a non-interactive / ``--json`` path,
route its errors to ``err_console`` too.
"""

from __future__ import annotations

from rich.console import Console

console = Console()
err_console = Console(stderr=True)
