# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-06-24

### Added

- `apply` now generates AI cover letters during dry runs whenever `--cover-letter` is enabled (the default) and a résumé path is configured. The generated letter is surfaced in `--json` output and in the console table as a preview before the user opts in with `--submit`.
- New live end-to-end tests (`tests/test_apply_dry_run_cover_letter_live.py`) that exercise the real `apply` CLI with vLLM cover-letter generation.

### Fixed

- Console table notes in `workflows/apply.py` are now escaped with `rich.markup.escape()` so bracketed labels like `[submit ✓]` and `[cover letter: N chars]` render literally instead of being stripped as invalid Rich markup.

## [0.2.0] - 2026-06-19

### Added

- Initial structured release baseline.

[Unreleased]: https://github.com/Zenetusken/job-applicator-python/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/Zenetusken/job-applicator-python/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Zenetusken/job-applicator-python/releases/tag/v0.2.0
