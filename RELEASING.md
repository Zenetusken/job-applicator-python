# Releasing job-applicator

This document describes the project release process.

## Versioning

This project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html):

- **MAJOR** (X.0.0): incompatible API or CLI behavior changes
- **MINOR** (0.X.0): new features, backwards-compatible
- **PATCH** (0.0.X): bug fixes, backwards-compatible

The version is declared in two places that must stay in sync:

- `pyproject.toml` (`[project] version`)
- `src/job_applicator/__init__.py` (`__version__`)

## Release steps

1. **Ensure the main branch is green.**

   ```bash
   git checkout main
   git pull origin main
   ruff check src/ tests/
   ruff format --check src/ tests/
   mypy src/
   pytest -m unit -x
   ```

2. **Decide the new version** based on the changes since the last release.

3. **Run the release script.**

   ```bash
   bash scripts/release.sh <version>
   ```

   This will:
   - Validate the working tree and test suite
   - Update `pyproject.toml` and `src/job_applicator/__init__.py`
   - Insert a dated section into `CHANGELOG.md` from `[Unreleased]`
   - Commit the version bump and changelog update
   - Create an annotated Git tag (`v<version>`)
   - Build the package distribution
   - Optionally push the tag (disabled by default; enable with `--push`)

4. **Publish the release.**

   - Push the release commit and tag:

     ```bash
     git push origin main
     git push origin v<version>
     ```

   - Create a GitHub Release from the tag and paste the relevant
     `CHANGELOG.md` section into the release notes.

   - Publish to PyPI (if/when configured):

     ```bash
     python -m twine upload dist/*
     ```

## Hotfix releases

For urgent fixes against the latest release:

1. Create a branch from the latest release tag:

   ```bash
   git checkout -b hotfix/vX.Y.Z vX.Y.Z-1
   ```

2. Apply the fix, run tests, and bump the patch version with the release script.

3. Tag and publish as above, then cherry-pick or merge the fix back to `main`.

## Pre-releases

For alpha/beta/rc versions use the script with a pre-release suffix:

```bash
bash scripts/release.sh 0.4.0a1
```

Pre-release tags follow PEP 440 (e.g. `0.4.0a1`, `0.4.0b2`, `0.4.0rc1`).
