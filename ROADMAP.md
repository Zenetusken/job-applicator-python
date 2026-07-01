# Roadmap

Living tracker of planned work. Detailed specs/plans/reports live under `docs/compose/`
(spec → plan → report). Substantial changes follow a gated cadence: empirics → spec →
implement → gate (`ruff` · `ruff format --check` · `mypy src/` · `pytest -m unit` ·
`pytest -m live`) → code-review → commit → PR.

## Planned — next arc open (audit medium-term backlog)

The two previously-"committed next" arcs are **resolved** (re-measured 2026-06-30; details under
Shipped):

- **Arc 1 — UX/robustness cleanup** — landed incrementally across the grounding/apply work.
  Measurement found `status` "Recent jobs" ordering and the embeddings CPU fallback already done,
  and the TUI already surfaces per-job matched/missing skills. The tailored-résumé markdown leak's
  last residual (three secondary interactive views — `[D]`/`[V]`/`[S]`) shipped in **PR #122**. The
  cover-letter repeated-verb tell is **deferred pending an empirical 8B fire-rate probe** (see Known
  follow-ups): it was banked pre-8B (PRs #90/#91), and a bigger base model tends to self-cap style
  tells, so measure the raw fire rate on baited worst-case JDs before building a style guard.
- **Arc 2 — Domain-general (semantic) skill grounding** — **Phase 1** (LLM evidence-span grounding)
  shipped default-on (2026-06-28); **Phase 2** (embedding-dedup normalization) is a **measured dead
  end**; **Phase 3** (ESCO/O*NET taxonomy) is an optional separate project. Full record:
  `docs/compose/specs/2026-06-26-semantic-skill-grounding.md`.

**Post-tailor structural-fidelity validation (#15) — DISSOLVED by founding empirics (2026-06-30).**
The intended validator (contact/date/title preservation after tailoring) was measured against the
real CV × 5 SOC JDs BEFORE building: **contact preserved 5/5, dates 0-drop** — because the DOCX table
fix (below) + the grounding verifier cure the loss at the SOURCE — and a title/company check **only
false-positived** on legitimate de-emphasis (a volunteer role, a university). So the validator was
not built; the genuine residual — surfacing the **tailored ATS + a contact green-check** in the
interactive `tailor` view (defense-in-depth) — shipped in **PR #126**. (ATS-on-tailored already
existed in the verbose/batch paths.) A textbook "fix the cause, not the symptom" outcome.

**Next arc — open.** The 2026-06-24 audit medium-term backlog is the queue: **selector health /
fail-loud on LinkedIn DOM drift** (#12, protects the one automated-apply path), **integration tests**
for state/batch/apply (#11), **structured experience/education extraction** (#14 — the 2026-07-01
résumé-pipeline audit found the fields UNIMPLEMENTED/dead; populate-or-remove, and populating is the
precondition for employment-gap detection). Plus the deferred **matching-tuning re-validation** and
**employment-gap detection** (see Known follow-ups).

## Shipped

- **Résumé-pipeline audit + fixes** (PRs #128–#130, 2026-07-01). The "dream" memory-vs-config
  cross-check caught that `config.toml` `resume_path` pointed at a **stale 2023 IT-support CV**, not
  the current SOC `_v1` — so `match`/`tailor`/`rescore` (incl. the #125 refresh) had silently scored
  the *wrong* CV; repointed + re-rescored (Analyste SOC 32%→**78%**, now the #1 match). A 3-agent
  audit of the résumé parse pipeline then measured + fixed, on the real CV: the `doctor`
  **résumé-identity guard** (#128 — a stale/wrong CV is now visible); **case-sensitive section
  headers** that made `summary` swallow 97.7% of the doc AND **aborted `tailor` on a valid CV** via a
  false ordering issue (#129 — one shared robust matcher, an advisory-not-blocking gate, removal of
  the education-staleness noise, and a docstring corrected for claiming unimplemented gap/overlap
  detection); and two **skill-extraction** defects (#130 — a paren-comma mangle splitting
  "Linux (Fedora, CLI, Bash)" into garbage, and a ≤2-char rule silently dropping C#/Go/R/AI/ML). The
  reliable spine (raw_text / contact / skills) was confirmed sound. Gate green throughout.

- **Foundation + matching + fidelity arc** (PRs #123–#126, 2026-06-30): an empirics-driven cascade.
  Reconciled the stale ROADMAP (#123); **fixed the DOCX parser to extract table cells** (#124, audit
  AI-H5) — the contact header + skills lived in tables and were silently dropped from `raw_text`,
  corrupting tailored CVs *and* 154 match-scores; added the account-safe **`rescore`** command (#125)
  and refreshed the funnel in place (mean Δ −0.185, top-10 reshuffled — the corrected CV's honest
  skill-overlap); and shipped the **tailored-ATS + contact green-check** surface (#126) as the sole
  residual of audit #15 after founding empirics dissolved the rest. Gate green throughout.

- **Apply/batch + scraper hardening epoch** (PRs #115–#122, 2026-06-30): scraper stealth
  fingerprint alignment + LinkedIn checkpoint/rate-limit detection (Track B), CLI volume-option
  clamps (C2), owner-only perms on written artifacts + skip-prefilled email (Track D), fail-loud
  cover-letter + fail-closed per-job state on the `apply` loop, batch crash/partial-failure
  recovery, resume-upload existence+type validation, and the secondary-view markdown strip. Cleared
  most of the 2026-06-24 comprehensive audit's Immediate + Short-term tier; gate green throughout.
- **Grounding-verifier + output-language + 8B base** (v0.5.0, 2026-06-29): language-agnostic honesty
  layer (`documents/grounding_verifier.py`), `[llm] language` packet policy, structured cover-letter
  generation, and the move to `Qwen/Qwen3-8B-AWQ` as the default base model.
- **Style-guide + sign-off release** (PR #35 / v0.3.6): structured cover-letter sign-off
  extraction/validation (`documents/sign_off.py`), applicant-name fallback from the parsed
  résumé, style-guide modal in the TUI (`g` key), and example guides in
  `docs/style-guide-examples/`. Includes adversarial review by architecture, lead-dev, QA,
  product/UX, and HR-domain subagents; all findings fixed. Gate: unit 876 · integration 5 ·
  live 34.
- **Hardening-arc audit follow-up** (PR #24): clusters 1–3 audit fixes + the six deferred
  design enhancements — half-open circuit breaker, `LLMRuntime` DI, `LLMResilienceConfig`
  policy, `ValidatedOutput` error-feedback, `StyleAnalyzer.format` staticmethod,
  `resume_tailor` folded onto the shared breaker, `BatchRunSpec`, batch mid-job resume.
  Gate at merge: unit 576 · live 21.
- **CLI decomposition** (PRs #26–#28, #30, #31, #33): extracted six layers from the
  ~3,200-line `cli.py` — board/browser/runtime factories + shared console (`factories.py`,
  `utils/console.py`), all cookie logic (`utils/cookies.py`), and the three orchestration
  loops (`workflows/{cover_letter,tailor,apply}.py`). `cli.py` ~3,200 → 2,303 (−~900).
  Each loop extracted tests-first: characterization tests pin behavior → guarded/verbatim
  move → zero drift.
- **Cover-letter + apply `NameError` fixes** (PRs #29, #32): two latent production crashes
  — `CoverLetterResult` / `ApplicationResult` constructed at runtime but imported only
  under `TYPE_CHECKING` — surfaced by the tests-first decomposition work and fixed via lazy
  imports.

## In progress / next (gated cycles — PR #25 + follow-ups)

- **A — doctor hygiene** *(done)*: `check_config` no longer creates the output dir
  (side-effect removed). resume.py L7/L8/L9 nits were already resolved in the cluster rework.
- **B — tier-2 test refresh** *(done)*: un-skipped E2 batch / D2 OCR / D4 ATS in
  `test_tier2_live` (all three are implemented; D4 now runs `ATSChecker` functionally).
- **C — integration smoke tests** *(done)*: first `tests/integration/` tests — board
  `browser_policy()` → `_make_browser` wiring, construction-only (no real launch).
- **D — Indeed search-only** *(done)*: see Out of scope.
- **E — CLI decomposition (incremental)** *(done — see Shipped)*: six gated increments
  extracted factories/console, cookie logic, and the cover-letter/tailor/apply loops, each
  validated green between. Only `batch`'s `_process_one` remains (banked — see Out of scope).

## Out of scope / accepted

- **Indeed automated apply**: scoped to search/match-only — automated apply is intentionally
  unsupported (Cloudflare anti-bot + ToS); the applicator returns a clean SKIPPED result
  directing the user to apply manually. LinkedIn Easy Apply is the only automated apply path.
- **Non-LinkedIn automated login**: not planned — collides with the standing "never
  automate login; reuse a human-seeded session" account-safety policy (see `CLAUDE.md`).
- **Browser selectors**: hardcoded against live DOM snapshots; inherent ongoing
  maintenance, not a discrete deliverable.
- **Batch `_process_one` extraction**: banked — the last orchestration loop, but not a
  clean lift like cover-letter/tailor/apply. It's a ~160-line closure in `batch`'s `_run`
  capturing ~18 locals incl. three shared mutable lists (`tailoring_scores`,
  `batch_reports`, `written_paths`) appended by concurrent workers under `Semaphore(3)`; a
  clean extraction needs a context-object refactor (not a verbatim move) plus careful
  handling of the concurrent-append seam — diminishing returns vs risk. Revisit only if
  `_run` needs changes the entanglement obstructs, and do it tests-first (characterization
  tests driving `batch` through `_process_one`) if so.

## Known follow-ups (deferred)

- **Matching tuning re-validation on the corrected CV.** The DOCX table fix (#124) recovered the
  skills section, so the 60/40 semantic/skill blend and the 0.75 per-skill match threshold — both
  tuned on the *old, skills-less* CV — now rest on changed input. The funnel was rescored (#125, then
  re-rescored 2026-07-01 on the corrected SOC CV — see Shipped), but whether 0.75/60-40 are still
  optimal needs its own empirics with a small gold-labelled (relevant/not) set — a separate arc, not
  a quick tweak. Until then, treat absolute match scores as skill-overlap (the existing caption
  already says so); relative ranking is sound.
- **Employment-GAP detection — the real HR red flag (not built).** `ResumeDateValidator` claimed
  "gap detection" in its docstring but never implemented it (corrected in #129). Unexplained
  employment gaps are the genuine signal a date check should surface; it needs reliable per-role
  date ranges, i.e. populated structured `experience` (see next) + a hardened date parser
  (the current one drops MM/YYYY, "Current", and French formats). Its own deliberate arc — do NOT
  bolt it onto the heuristic regex parser.
- **Structured `experience`/`education` are UNIMPLEMENTED dead fields (audit 2026-07-01).**
  `parse_text` never constructs an `ExperienceEntry`/`EducationEntry` (grep: zero call sites) — the
  fields are always empty, and every consumer falls back to `raw_text`, so nothing breaks today, but
  it's the "fabricated default" antipattern. **Populate-or-remove** (audit #14) — and populating is
  the precondition for gap detection above. Not justified by fidelity (measured clean); relevant to
  matching + gaps.
- **Single-char skill asymmetry (LOW, #130 residual).** `matching.py:320` (résumé side) keeps a
  `len>=2` filter while `:330` (requirement side) has none, so a job listing a bare `R`/`C` can
  surface it as an unsatisfiable "missing skill" even when the CV lists it. 1-char only (2-char
  Go/C#/AI/ML fully fixed in #130); no impact on the current CV. 1-line symmetric fix when touched.
- **Cover-letter repeated-verb "voice tell"** (ROADMAP Arc-1 Item 2). `_voice_tells`
  (`documents/cover_letter.py`) scores phrase-presence + structural tells but has **no intra-letter
  repeated-token detector** (e.g. "engineered" ×4); the `_devoice` / `_voice_correction` re-prompt
  loop exists but nothing triggers it on repetition. **Deferred pending measurement:** the tell was
  banked pre-8B (PRs #90/#91); a bigger base model tends to self-cap style tells, so measure the raw
  fire rate on baited worst-case JDs (needs vLLM up) BEFORE building — else mark wontfix. Adding a
  style guard the model doesn't need risks over-restricting its prose.
- **`grounding_mode="keyword"` opt-out is a dead knob on the interactive-`tailor` + TUI-tailor
  *displayed* scores.** `ResumeTailor`'s bare-fallback `JobMatcher` (`documents/resume_tailor.py`
  ~L650/L797) isn't threaded `settings.skills.grounding_mode` (it only receives an `LLMConfig`), so
  those two paths always use the param default. Surfaced by the 2026-06-28 default flip
  keyword→`evidence_span` (PR for `feat/grounding-evidence-span-default`): pre-existing (the bare
  fallback ignored the knob before the flip too — it just *coincided* with the keyword default), but
  the flip turned it from failing the opt-*in* into failing the opt-*out*. The `match` / `batch` /
  apply-gate / TUI-match paths all honor the config knob. Fix = thread the mode through
  `ResumeTailor` (ctor or `tailor()`/`refine()` param) + a wiring test asserting the tailor-internal
  matcher's `_grounding_mode` follows config. Small, self-contained, tests-first.

## Operational

- Re-validate the live E2E suite (Tier-1/Tier-2, batch, tailor) whenever vLLM or a
  selector changes; live LinkedIn/Easy-Apply flows are inherently fragile.
