# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.5] - 2026-06-24

### Fixed

- `batch --no-cover-letter` is now respected even when `--style-guide` is provided; cover letters are no longer generated in that mode.
- `apply`, `batch`, and `tailor` now forward `--ocr-mode` and `--force-ocr` to the style-guide loader.
- `apply` dry-run and style-guide status messages are emitted to stderr so `--json` stdout remains clean.
- Interactive `tailor` refinements (`[R]efine`, `[I]nput`, `[S]tyle`) now preserve the loaded style guide.

### Changed

- `CoverLetterGenerator.load_style_guide` uses `ResumeLoader` for all file types and raises `DocumentError` for missing or unreadable paths.
- `CoverLetterOutput` and `StyleGuide` models now reject unknown fields (`extra="forbid"`).
- `config-init` template now includes `style_guide_path`.
- `AGENTS.md` now documents `--style-guide` support for `apply`, `tailor`, `generate-cover-letter`, and `batch`.

## [0.3.4] - 2026-06-24

### Added

- Universal multi-file style-guide support: `CoverLetterGenerator.load_style_guide` now accepts a comma-separated list of paths, loads `.pdf` files via `ResumeLoader`, and merges multiple examples with `StyleAnalyzer.analyze_multiple`.
- `apply --style-guide` now loads the style guide and passes it through to `CoverLetterGenerator.generate`.

### Changed

- `generate-cover-letter` now uses the same shared style-guide loader as `apply`, `batch`, and `tailor`, removing the duplicated inline loading logic.
- `config.example.toml` and `README.md` document comma-separated `--style-guide` usage.


## [0.3.3] - 2026-06-24

### Fixed

- Help-text unit tests now use introspection for option registration and set `COLUMNS=200` via `monkeypatch` before rendering, avoiding failures caused by `CliRunner` forcing an 80-column terminal width in CI.

## [0.3.2] - 2026-06-24

### Fixed

- Unit tests that assert on `--help` output now invoke the CLI with `COLUMNS=200` so option names are not truncated on narrow CI terminals.

## [0.3.1] - 2026-06-24

### Fixed

- Release workflow now installs the `[embeddings]` extra so `mypy src/` passes in CI.

## [0.3.0] - 2026-06-24

### Added

- `apply` now generates AI cover letters during dry runs whenever `--cover-letter` is enabled (the default) and a résumé path is configured. The generated letter is surfaced in `--json` output and in the console table as a preview before the user opts in with `--submit`.
- New live end-to-end tests (`tests/test_apply_dry_run_cover_letter_live.py`) that exercise the real `apply` CLI with vLLM cover-letter generation.

### Fixed

- Console table notes in `workflows/apply.py` are now escaped with `rich.markup.escape()` so bracketed labels like `[submit ✓]` and `[cover letter: N chars]` render literally instead of being stripped as invalid Rich markup.

## [0.2.0] - 2026-06-19

### Added

- Initial structured release baseline.

[0.3.4]: https://github.com/Zenetusken/job-applicator-python/compare/v0.3.3...v0.3.4
[0.3.5]: https://github.com/Zenetusken/job-applicator-python/compare/v0.3.4...v0.3.5
[Unreleased]: https://github.com/Zenetusken/job-applicator-python/compare/v0.3.5...HEAD
[0.3.3]: https://github.com/Zenetusken/job-applicator-python/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/Zenetusken/job-applicator-python/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/Zenetusken/job-applicator-python/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/Zenetusken/job-applicator-python/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Zenetusken/job-applicator-python/releases/tag/v0.2.0
