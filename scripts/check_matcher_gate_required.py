#!/usr/bin/env python
"""Detect whether changed files require the private matcher eval gate."""

from __future__ import annotations

import argparse
import fnmatch
import subprocess
import sys

SENSITIVE_PATTERNS = (
    "src/job_applicator/embeddings/matching.py",
    "src/job_applicator/embeddings/skill_extraction.py",
    "src/job_applicator/skills/*.py",
    "src/job_applicator/documents/grounding_verifier.py",
    "src/job_applicator/config.py",
    "tests/unit/test_embeddings.py",
    "tests/unit/test_matching_target_roles.py",
)


def changed_paths(base: str) -> list[str]:
    diff = subprocess.run(
        ["git", "diff", "--name-only", base, "--"],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )
    if diff.returncode != 0:
        raise SystemExit(diff.stderr.strip() or f"git diff failed for base {base!r}")
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )
    if untracked.returncode != 0:
        raise SystemExit(untracked.stderr.strip() or "git ls-files failed")
    paths = {
        line.strip()
        for line in (diff.stdout + "\n" + untracked.stdout).splitlines()
        if line.strip()
    }
    return sorted(paths)


def matcher_sensitive(paths: list[str]) -> list[str]:
    return [
        path
        for path in paths
        if any(fnmatch.fnmatch(path, pattern) for pattern in SENSITIVE_PATTERNS)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        default="HEAD",
        help="Git revision to diff against. Default: HEAD, useful for unstaged work.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only use the exit code.",
    )
    args = parser.parse_args()

    sensitive = matcher_sensitive(changed_paths(args.base))
    if not sensitive:
        if not args.quiet:
            print("matcher eval not required: no matcher-sensitive files changed")
        return 0

    if not args.quiet:
        print("matcher eval required for these changed paths:")
        for path in sensitive:
            print(f"  - {path}")
        print("\nRun: .venv/bin/python scripts/eval_matching.py --required")
    return 1


if __name__ == "__main__":
    sys.exit(main())
