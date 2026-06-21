#!/usr/bin/env bash
#
# Smoke driver for job-applicator — see SKILL.md (this directory).
#
# Drives ONLY safe, read-only / offline surfaces. It NEVER runs
#   login · import-cookies · search · match · apply · batch · check-session
# because those touch the user's REAL LinkedIn account / stored credentials.
# Cover-letter generation is driven with the job description passed INLINE
# (--description), so no browser launches and no board is scraped.
#
# Two tiers (mirroring the project's own unit/live test split):
#   CORE — offline, no vLLM/GPU: --version, --help, ats-check, config-init.
#          A CORE failure fails the run (exit 1).
#   LIVE — needs vLLM at localhost:8000 (+GPU): doctor, generate-cover-letter.
#          Auto-SKIPPED (not failed) when vLLM is unreachable.
#
# Usage:
#   bash driver.sh           # core + live (live auto-skips if vLLM is down)
#   bash driver.sh --core    # offline checks only
#   bash driver.sh --live    # live checks only (still runs nothing unsafe)
#
# NOT `set -e`: we capture each exit code and tally, so one expected
# non-zero (or a live skip) can't abort the whole run.
set -uo pipefail

# --- locate the repo + the CORRECT entrypoint -------------------------------
# Resolve from BASH_SOURCE so the driver works from any CWD. The script lives
# at <repo>/.claude/skills/run-job-applicator/driver.sh → repo is three up.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"
JA="$REPO/.venv/bin/job-applicator"   # NOT ~/.local/bin (may bind to system py3.10)
PY="$REPO/.venv/bin/python"

MODE="${1:-all}"

if [[ ! -x "$JA" ]]; then
  echo "FATAL: $JA not found." >&2
  echo "Run setup first (see SKILL.md):" >&2
  echo "  python3.12 -m venv .venv && .venv/bin/pip install -e '.[dev,embeddings,browser,indeed]'" >&2
  echo "Do NOT use the ~/.local/bin/job-applicator shim — it may use system Python 3.10 (StrEnum ImportError)." >&2
  exit 2
fi

# --- isolated workdir: no repo config.toml is loaded, no clobber risk --------
WORK="$(mktemp -d "${TMPDIR:-/tmp}/job-applicator-smoke.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
cd "$WORK"   # every CLI invocation runs here, not in the repo

PASS=0; FAIL=0; SKIP=0
ok()   { echo "  ✓ $1"; PASS=$((PASS + 1)); }
bad()  { echo "  ✗ $1"; FAIL=$((FAIL + 1)); }
skip() { echo "  • SKIP — $1"; SKIP=$((SKIP + 1)); }

# run <label> <expect-substr|-> <cmd...>
# passes when the command exits 0 AND (expect = "-" OR output contains expect).
run() {
  local label="$1" expect="$2"; shift 2
  local out rc
  out="$("$@" 2>&1)"; rc=$?
  if [[ $rc -ne 0 ]]; then bad "$label (exit $rc)"; return 1; fi
  if [[ "$expect" != "-" ]] && ! grep -qF -- "$expect" <<<"$out"; then
    bad "$label (output missing '$expect')"; return 1
  fi
  ok "$label"
}

# A fake résumé. python-docx is a core dependency, so this is always available
# on a working install — keeps the driver self-contained (no committed fixture).
make_resume() {
  "$PY" - "$WORK/resume.docx" <<'PY'
import sys
from docx import Document
d = Document()
d.add_heading("Jordan Sample", 0)
d.add_paragraph("jordan.sample@example.com | (555) 123-4567 | San Francisco, CA")
sections = [
    ("Summary", ["Senior Python engineer, 8 years building async data pipelines, "
                 "REST APIs, and ML-backed services. Strong in asyncio and type-safe code."]),
    ("Experience", ["Staff Engineer, Acme Data (2021-Present)",
                    "Built async ingestion handling 2B events/day with asyncio.",
                    "Led a Pydantic v2 + mypy-strict migration across 40 services.",
                    "Backend Engineer, Globex (2017-2021)",
                    "Designed PostgreSQL schemas and Redis caching for a 5M-user app."]),
    ("Education", ["B.S. Computer Science, State University (2017)"]),
    ("Skills", ["Python, asyncio, FastAPI, Pydantic, PostgreSQL, Redis, Docker, AWS, Playwright"]),
]
for head, body in sections:
    d.add_heading(head, level=1)
    for line in body:
        d.add_paragraph(line)
d.save(sys.argv[1])
PY
}

core() {
  echo "[CORE] offline checks (no vLLM / GPU)"
  run "version"             "job-applicator v" "$JA" --version
  run "help lists commands" "doctor"           "$JA" --help
  run "ats-check (docx)"    "ATS Score"        "$JA" ats-check --resume "$WORK/resume.docx"
  run "ats-check --json"    '"is_compatible"'  "$JA" ats-check --resume "$WORK/resume.docx" --json
  run "config-init"         "Created"          "$JA" config-init -o "$WORK/sample-config.toml"
}

live() {
  echo "[LIVE] needs vLLM at localhost:8000 (+GPU)"
  if ! curl -fsS -m 3 http://localhost:8000/v1/models >/dev/null 2>&1; then
    skip "doctor — vLLM unreachable at localhost:8000"
    skip "generate-cover-letter — vLLM unreachable at localhost:8000"
    return 0
  fi
  # doctor: assert it ran and reached a reachable LLM (the live tier's whole point).
  run "doctor (LLM reachable)" "reachable" "$JA" doctor
  # generate-cover-letter: key ONLY on exit 0 + the result header. litellm prints a
  # scary "Give Feedback / Get Help" banner — and sometimes a transient BadRequestError
  # that the validation-retry recovers from — even on success; do NOT grep for "error".
  run "generate-cover-letter (real LLM)" "Generated Cover Letter" \
    "$JA" generate-cover-letter \
      --job-title "Senior Python Engineer" \
      --company "Initech" \
      --description "Build async data pipelines in Python; asyncio, Pydantic, PostgreSQL, AWS." \
      --resume "$WORK/resume.docx"
}

echo "job-applicator smoke driver"
echo "  repo:    $REPO"
echo "  cli:     $JA"
echo "  workdir: $WORK"
echo

# Both tiers need the résumé (ats-check + generate-cover-letter).
if make_resume; then ok "sample résumé (docx)"; else bad "sample résumé (docx)"; fi
echo

case "$MODE" in
  --core) core ;;
  --live) live ;;
  all)    core; echo; live ;;
  *) echo "unknown mode: $MODE (use --core | --live | no arg)" >&2; exit 2 ;;
esac

echo
echo "PASS=$PASS  FAIL=$FAIL  SKIP=$SKIP"
if [[ $FAIL -eq 0 ]]; then echo "SMOKE OK"; exit 0; else echo "SMOKE FAILED"; exit 1; fi
