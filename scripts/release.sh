#!/usr/bin/env bash
# Release automation for job-applicator.
#
# Usage: bash scripts/release.sh <version> [--push]
#
# The script validates the tree, runs the fast test gate, bumps the version in
# pyproject.toml and src/job_applicator/__init__.py, updates CHANGELOG.md from
# the [Unreleased] section, commits, tags, and builds the distribution.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON="${PROJECT_ROOT}/.venv/bin/python3.12"
if [[ ! -x "${PYTHON}" ]]; then
    echo "Error: ${PYTHON} not found."
    echo "Run: python3.12 -m venv .venv && .venv/bin/pip install -e '.[dev,embeddings,browser,indeed]'"
    exit 1
fi

VERSION="${1:-}"
PUSH="${2:-}"

if [[ -z "${VERSION}" ]]; then
    echo "Usage: bash scripts/release.sh <version> [--push]"
    exit 1
fi

if ! [[ "${VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+([a-zA-Z0-9]+)?$ ]]; then
    echo "Error: version must follow SemVer/PEP 440 (e.g. 0.3.0 or 0.4.0a1)"
    exit 1
fi

if [[ "${PUSH}" != "" && "${PUSH}" != "--push" ]]; then
    echo "Error: unknown flag '${PUSH}'. Use '--push' to push the tag."
    exit 1
fi

TAG="v${VERSION}"
DATE=$(date +%Y-%m-%d)

echo "==> Validating release environment"
if [[ -n "$(git status --porcelain)" ]]; then
    echo "Error: working tree is not clean. Commit or stash changes first."
    git status --short
    exit 1
fi

echo "==> Running quality gate (unit tests)"
"${PYTHON}" -m ruff check src/ tests/
"${PYTHON}" -m ruff format --check src/ tests/
"${PYTHON}" -m mypy src/
"${PYTHON}" -m pytest -m unit -x -q

echo "==> Bumping version to ${VERSION}"
sed -i "s/^version = \"[^\"]*\"/version = \"${VERSION}\"/" pyproject.toml
sed -i "s/^__version__ = \"[^\"]*\"/__version__ = \"${VERSION}\"/" src/job_applicator/__init__.py

echo "==> Updating CHANGELOG.md"
if ! grep -q "^## \[Unreleased\]" CHANGELOG.md; then
    echo "Error: CHANGELOG.md is missing an [Unreleased] section"
    exit 1
fi

"${PYTHON}" - <<PY
import re
from pathlib import Path

version = "${VERSION}"
date = "${DATE}"
changelog = Path("CHANGELOG.md")
content = changelog.read_text(encoding="utf-8")

# Extract content between [Unreleased] header and the next ## [ header.
match = re.search(
    r"## \[Unreleased\]\n\s*\n((?:(?!^## \[).)*)",
    content,
    re.DOTALL | re.MULTILINE,
)
if not match:
    raise SystemExit("Error: could not parse [Unreleased] section")

unreleased = match.group(1).strip()
if not unreleased:
    raise SystemExit("Error: [Unreleased] section is empty")

# Find the previous released version from the first ## [X.Y.Z] after Unreleased.
prev_match = re.search(r"## \[([0-9]+\.[0-9]+\.[0-9]+[^\]]*)\]", content)
prev_version = prev_match.group(1) if prev_match else "0.0.0"

# Replace the Unreleased section with empty Unreleased + new release section.
new_release = f"## [Unreleased]\n\n## [{version}] - {date}\n\n{unreleased}\n\n"
content = content[:match.start()] + new_release + content[match.end():]

# Update compare links.
content = re.sub(
    r"\[Unreleased\]: https://github\.com/Zenetusken/job-applicator-python/compare/v[^.]+\.[^.]+\.[^.]+\.\.\.HEAD",
    f"[Unreleased]: https://github.com/Zenetusken/job-applicator-python/compare/v{version}...HEAD",
    content,
)

new_link = f"[{version}]: https://github.com/Zenetusken/job-applicator-python/compare/v{prev_version}...v{version}\n"
content = re.sub(
    r"(\[Unreleased\]: https://github\.com/Zenetusken/job-applicator-python/compare/v)",
    new_link + r"\1",
    content,
    count=1,
)

changelog.write_text(content, encoding="utf-8")
print(f"Updated CHANGELOG.md with release {version} (previous: {prev_version})")
PY

echo "==> Staging release artifacts"
git add pyproject.toml src/job_applicator/__init__.py CHANGELOG.md

echo "==> Committing version bump"
git commit -m "chore(release): bump version to ${VERSION}

See CHANGELOG.md for release notes."

echo "==> Tagging ${TAG}"
git tag -a "${TAG}" -m "Release ${TAG}"

echo "==> Building distribution"
rm -rf dist/
"${PYTHON}" -m build

echo ""
echo "Release ${TAG} prepared locally."
echo "Next steps:"
echo "  1. Inspect the commit: git show HEAD"
echo "  2. Inspect the tag:   git show ${TAG}"
echo "  3. Push the release:  git push origin main && git push origin ${TAG}"
if [[ "${PUSH}" == "--push" ]]; then
    echo "  -> Pushing because --push was requested"
    git push origin main
    git push origin "${TAG}"
fi
