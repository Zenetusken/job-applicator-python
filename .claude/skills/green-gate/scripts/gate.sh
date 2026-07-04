#!/usr/bin/env bash
# The job-applicator "green gate" — the canonical pre-commit / pre-PR quality gate.
# Mirrors AGENTS.md: ruff check · ruff format --check · mypy (strict src/) · pytest -m unit.
# Fail-FAST: stops at the first failing stage and names it. Exit 0 = all green.
#
# Portable by design: finds the repo root itself (git, else the skill's location), so it works
# from any CWD and carries no absolute paths. Uses the project .venv.
set -uo pipefail

# Repo root: prefer git; fall back to walking up from this script (…/.claude/skills/green-gate/scripts).
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [ -z "${ROOT}" ]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
fi
cd "${ROOT}" || { echo "✗ could not cd to repo root"; exit 2; }

RUFF=.venv/bin/ruff
MYPY=.venv/bin/mypy
PY=.venv/bin/python
if [ ! -x "${RUFF}" ] || [ ! -x "${PY}" ]; then
  echo "✗ project .venv not found at ${ROOT}/.venv — create/populate it first (see the run-job-applicator skill)."
  exit 2
fi

stage() { printf '\n\033[1m▶ %s\033[0m\n' "$1"; }
fail()  { printf '\n\033[31m✗ GATE FAILED at: %s\033[0m\n' "$1"; exit 1; }

stage "ruff check (lint) — src/ tests/"
"${RUFF}" check src/ tests/ || fail "ruff check   →  fix: ${RUFF} check --fix src/ tests/"

stage "ruff format --check — src/ tests/"
"${RUFF}" format --check src/ tests/ || fail "ruff format  →  fix: ${RUFF} format src/ tests/"

stage "mypy (strict) — src/"
"${MYPY}" src/ || fail "mypy (strict src/)"

stage "pytest -m unit (fast suite — no browser/GPU/vLLM)"
"${PY}" -m pytest -m unit -q || fail "pytest -m unit"

printf '\n\033[32m✓ GREEN — all gate stages passed.\033[0m\n'
