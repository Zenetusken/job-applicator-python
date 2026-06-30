# Spec — Grounding Verifier: language-agnostic honesty for generated documents

**Status:** Proposed (spec review). Core empirically pre-validated on a throwaway prototype (§7).
**Branch:** `feat/grounding-verifier` (off `feat/output-language-policy`, PR #106).
**Folds into:** the release cycle below (§10), with #106.

## 1. Motivation

The honesty guards (`_reject_unearned_credentials`, `_reject_unlisted_named_tools`,
`_reject_overclaim_phrases`, the résumé `_strip_*` family) are **hardcoded English term-lists**.
Three failure modes bit this session:

- **Unbounded (whack-a-mole).** Named tools/phrases are infinite; the model always finds the next
  one (`cloud-native` → `Mac-first` → `AWS IAM`). A blocklist is permanently one step behind.
- **Brittle.** `_reject_invented_company_employment` false-fired ~45% on "my interest in <company>".
- **Inert on French** (the new language policy, #106): `_CREDENTIAL_TERMS` never matches "certifiée".

The root mistake is **enumerating the bad**. The decisive proof is the user's real pair — both say
"100%", one is honest, one is fabricated:

| claim | source (his V1 CV) | verdict |
|---|---|---|
| "Took over **100%** of inbound email" | verbatim on the CV | **grounded** |
| "**100%** / all-cases first-call resolution" | CV says "roughly **95%**" | **fabricated** |

No term-list and no "does 100% appear in the source" check can separate these — the difference is
**contextual entailment** (does *this* claim's number match what the source says about *this* fact).
That cannot be hardcoded, and definitely not per-language.

## 2. Design — a grounding verifier (entailment, not enumeration)

One language-agnostic LLM pass. **SOURCE = the BASE résumé ONLY** — concretely `resume.raw_text`,
never the JD and **never the tailored intermediate** (`tailored_resume_text`), even when the cover
letter is generated from the tailored text. The JD is where "AWS IAM" came from; the tailored
résumé is where a tailor-introduced fabrication would already live — verifying against either would
bless the exact fabrication the verifier exists to catch (same bug class, both directions). The
model enumerates every substantive claim in the GENERATED doc and, for each, decides grounded/not
against the base résumé. It reads French and English natively, so there is **nothing per-language
to maintain**.

Embeddings are the wrong tool here: "95%" and "100%" are near-identical vectors. This is an
entailment job, not a similarity job.

## 3. Hardening principles (each maps to a previous mistake)

- **#2 — verify the EVIDENCE, not the verdict (the load-bearing one).** The old guards trusted
  patterns; a naive verifier just trusts the 8B's judgment (which slips). So: per-claim, the model
  must **quote the verbatim source line** that grounds each claim, then a **deterministic `audit()`
  overrides** it — (a) the quote must really be in the source (token-overlap ≥ 0.7, robust to light
  reformatting; a fabricated quote shares few words), and (b) **numeric backstop**: a *number* in
  the claim — a percentage OR a standalone integer (years, counts, team sizes) — must appear in its
  cited quote, so "100%" can't be grounded by a "95%" line and "15 years" can't be grounded by a
  "10+ years" line, even if the model mis-judges. A digit glued to letters ("BIND9", "SHA256") is a
  proper noun, not a metric, and is excluded.
  - **Coverage check (the structural miss-direction).** `audit()` overrides *grounded* verdicts and
    the model flags *not-grounded* ones — but a fabrication the model **never enumerates** is
    neither, so it passes silently, and the gold set can't catch it (it measures recall on claims we
    already know to look for). So a deterministic **coverage check**: every substantive sentence of
    the GENERATED doc must map to ≥1 enumerated claim (token-overlap). An uncovered sentence → the
    enumeration was incomplete → the **fail-safe path** (#4), never a clean pass.
  - **Scope of the deterministic backstop (named, NOT overclaimed).** `audit()` is a BACKSTOP behind
    the model's judgment, not a complete entailment checker. It deterministically catches a
    fabricated *quote* (not in the source) and a fabricated *number* (absent from its quote). It does
    NOT independently verify that a quote semantically *entails* its claim: a **numberless**
    fabrication grounded by a real-but-unrelated quote (e.g. a bare "Holds a CISSP" cited against an
    unrelated real line) passes the deterministic layer — it is caught only by the model's
    grounded/not judgment and the English floor (§4), not by `audit()`. Coverage is **union-based** (a
    sentence is covered if its tokens appear ACROSS the enumerated claims), so a fabrication whose
    tokens scatter across unrelated claims can still read as "covered". Both residuals are **measured
    by the gold set (§7) and named in §11** — not silently assumed away. A deterministic claim↔quote
    token gate was **rejected**: it re-runs the "enumerate the bad" mistake and false-positives on
    faithful rephrasing (the user's own "Owned the complete inbound email funnel" ← "Took over 100%
    of inbound email…" overlaps ~0.33 → a token gate would flag his honest claim). The semantic
    entailment job belongs to the LLM layer; the gold set verifies the LLM does it.
- **#3 — gold set, both directions (a `live` measurement, not a unit gate).** Avoid the old
  clean-rate blind spot. A labeled set — **≥20–30 per direction**, grounded + fabricated, EN + FR;
  the user's real pair + the credential and cross-language cases as the seed; I author the labels,
  the user corrects — scores **precision and recall separately**. It is a live LLM call
  (nondeterministic), so it runs in the **`live` tier, reported not hard-gating** — it CANNOT be a
  `-m unit` test. The pure `audit()` is what protects the fast suite (§9).
- **#4 — fail-safe, never fail-open.** An LLM call can fail; silently shipping an unverified doc as
  "clean" is a `no-failure-masking-fallbacks` violation. Verifier unreachable → fall back to the
  deterministic floor + flag "semantic check skipped", **never report clean**.

## 4. Augment, NOT replace

The user objected to *per-language hardcoding*, not to the English floor. The deterministic guards
(a) are what the **unit tests assert honesty against** — moving enforcement to a live LLM call would
drop honesty regression off the fast gate; (b) are **proven** (the named-tool guard caught "Splunk"
in this session's qa-live). Keep them as the fast English floor. The verifier is the **one**
language-agnostic semantic layer on top. Retire individual deterministic guards only in a follow-up,
and only if the gold set proves the verifier subsumes them — never on faith.

## 5. Components

- `documents/grounding_verifier.py` — `GroundingVerifier` (per-claim verify + `audit()`), the
  prompt, the LLMRuntime/circuit-breaker wiring (reuse the generators' resilience).
- Pydantic models `ClaimCheck(claim, grounded, source_quote, note)` + `VerificationReport`.
- `audit()` — pure, deterministic, unit-tested in isolation (token-overlap + percentage backstop).
- `tests/data/grounding_gold.json` + `tests/unit/test_grounding_verifier.py` — the gold set (#3).

## 6. Integration

- **Cover letter → reject→retry.** The verifier's confirmed findings feed the existing
  validate→retry loop (disposable; a false flag just regenerates). Sits *after* the deterministic
  floor.
- **Résumé → surfaced report, NOT a silent strip.** He is the ground truth for his own CV (he knew
  which 100% was real; the system did not). Concretely: `TailoredResume` gains
  `grounding_report: VerificationReport | None`. The tailor attaches it but **never fails or strips
  on a flag** (non-blocking — a false flag must be a dismissible *question*, never a deleted
  accomplishment from the document of record). Consumer: the `tailor` CLI prints a "⚠ N claims to
  review" panel (each row: the claim · why · the cited source line, or "not in your résumé") after
  writing the artifact, and the structured report is a key in `tailor --json`; the TUI shows the
  count on the tailored card. The action loop is the user's — fix the tailored doc, fix V1, or
  dismiss. This crosses module boundaries (model field + CLI render + `--json` schema), so it lands
  as one slice with its own tests.

## 7. Validation (prototype evidence — already measured, not asserted)

Throwaway prototype against his real base CV, source = résumé only:

- Confusion matrix (calibrated): catches "100% first-call" (EN+FR) and Splunk/CrowdStrike/AWS;
  leaves "100% inbound", "10+ years", the skills list (EN+FR) — both LLM false positives fixed by
  percentages-only + token-overlap. **Cross-language works** — French fabrications caught vs an
  English source; French faithful translations grounded (the whole justification).
- `audit()` unit checks pass: catches a fake quote and a "100%-grounded-by-95%" numeric mismatch;
  passes real groundings (incl. short "SIEM").

**Known residual (must be in the gold set):** the per-claim "find a supporting quote" framing made
the model *lenient on credentials* — it grounded "Holds a **certified** qualification" by quoting
the in-progress "Undergraduate Certificate" line. `audit()` can't catch it (real quote, no
percentage; semantic gap: in-progress ≠ held). Mitigation: the English `_CREDENTIAL_TERMS` floor
catches it in English (augment); the **French** credential is the open residual the gold set will
quantify and prompt-tuning will target. **Acceptance: measured + separately reported, NOT
hard-gated.** If verifier + prompt-strictness don't reach the target on French credentials, it
ships as a *logged known-gap* — the report surfaces it (§6) and the user reviews it; named, not
silently passed. The English floor still covers English regardless.

**Re-measured (Slice 6, post-review, 8B on local vLLM, 32-case set + adversarial additions):**
**recall 1.00** (17/17 fabricated caught, incl. the new numberless/semantic holes the deterministic
`audit()` can NOT catch alone — "Holds a CISSP", "Architected an enterprise cloud security program",
"200-node Kubernetes fleet", "led a cloud migration to AWS for a Fortune 500 bank"). This is the
decisive result: the **LLM layer catches what the deterministic backstop can't**, so the §11
claim↔quote-entailment residual is theoretical, not a shipped hole. (The cases are single-claim, so
they exercise the entailment direction — the model flagging numberless/semantic fabrications — but
NOT the union-coverage residual, whose pooling-across-multiple-claims mechanism stays named-but-
unmeasured; it is a backstop behind the model's primary flag.) **Precision ~0.81–0.85 overall** —
the residual (§8, since scored per-tag: CORE precision 1.00 / residual 3/3): French faithful
translations + low-overlap rephrases under-grounded; F3's new integer backstop is precision-neutral
(no F3-caused false positive).

## 8. Open risks

- **Credential leniency in French** (§7) — measured by the gold set; addressed by prompt strictness
  ("coursework / in-progress / exam-pending is NOT a held credential") + the English floor.
- **Precision residual on faithful low-overlap / cross-language groundings** (§7 measured). The model
  is conservative: it under-grounds French faithful translations and low-overlap English rephrases of
  a real source line (e.g. "Took on every inbound email request from the sales team" ← "Took over
  100% of inbound email service requests from the sales team"), so they read as flagged. Direction is
  the SAFE one for the résumé (a surfaced "review" note the user dismisses, never an auto-strip) and
  is absorbed by reject→retry for the letter. **Scored per-tag (implemented):** the live harness
  partitions grounded cases into CORE (measured 0 false positives over N=5 → strict ≥0.90 floor, the
  real gross-regression guard) and the `"residual": true`-tagged cases (reported as flagged/total,
  NOT gated). It replaced the earlier blunt overall ≥0.78 floor. **Prompt-tuned (done):** the
  verifier prompt now grounds a faithful PARAPHRASE/TRANSLATION by MEANING (not shared words), which
  grounded BOTH French residual cases — they are now CORE (measured: CORE precision 1.00 over 14
  core, recall 1.00 over 21 fabricated). Measure-first found that ANY loosening — even a
  translation-only clause — let a SCOPE inflation slip ("the entire company" from a "the sales team"
  source) while the strict baseline caught it, so the prompt PAIRS the meaning-grounding with an
  explicit inflation + SCOPE guard, validated EN+FR against adversarial inflations (added to the gold
  set as `scope`-category recall guards). The ONE remaining residual is the English SAME-LANGUAGE
  low-overlap rephrase: the 8B grounds cross-language translations but still under-grounds
  same-language low-overlap, so it stays the named, per-tag-reported residual. Honest limit: the
  honesty evidence is one SYNTHETIC source; the prompt was not validated against the user's real CV.
- **Non-determinism** — absorbed by reject→retry (letter) and human arbitration (résumé report).
- **Token cost** — per-claim enumeration of a full CV is a larger call; acceptable on local vLLM.
  Measure latency; cap with the `[cover_letter]` model override for a cloud verifier if needed.

## 9. Gates (gated validation)

1. Per step: `ruff` · `ruff format` · `mypy --strict` · `pytest -m unit`.
2. **`audit()` + coverage pure-function unit tests — the FAST gate** (deterministic; this is what
   actually protects honesty in CI): fake quote, numeric mismatch, real short quote, reformatted
   quote, an uncovered sentence → incomplete → fail-safe.
3. **Gold-set measurement — `live` tier, REPORTED not hard-gating** (a nondeterministic live LLM
   call cannot be a `-m unit` gate): targets recall ≥0.9 on fabrications, precision ≥0.9 on
   grounded, both directions EN+FR, ≥20–30 per direction; reported + tracked over time. The
   French-credential row is reported separately (§7 acceptance).
4. Adversarial multi-agent review of the diff before PR.
5. `qa-sanity --core` + `--live` re-baseline (the verifier must not regress the generation paths).

## 10. Release cycle (fold everything in)

Deterministic version source = `pyproject.toml` `version` (currently **0.4.1**) + `CHANGELOG.md`.
This session's arc — structured cover-letter generation (#103), the 8B base (#104), the 8B tuning
(#105), the language policy (#106), and this verifier — is a coherent feature set → **minor bump
0.4.1 → 0.5.0** (semver: new user-facing features, no breaking change). Steps:

1. Land the verifier PR (gated + reviewed).
2. Merge #106 (affirmed, verified) — fold-in.
3. Single reviewed version bump 0.4.1 → 0.5.0; `CHANGELOG.md` "Unreleased" → `0.5.0` with the arc.
4. Final integration gate on merged `main` + `qa-sanity`; tag `v0.5.0`.

## 11. Out of scope (named, not silent)

- Retiring any deterministic guard (follow-up, gold-set-gated).
- Languages beyond FR/EN (the design is language-agnostic; the gold set is FR/EN only for now).
- Verifying the JD-derived relevance of a claim (the verifier checks grounding, not job-fit).
- **Deterministic claim↔quote entailment** for a numberless fabrication grounded by a real but
  unrelated quote (§3 "scope of the deterministic backstop"). Rejected as a deterministic gate (it
  false-positives on faithful rephrasing); the LLM layer + the English floor cover it, and the gold
  set (§7) measures whether they do. A follow-up only if the gold set shows the LLM layer misses it.
- **Per-claim coverage** (coverage is currently union-based; a per-claim gate risks flagging
  legitimate sentences that combine two real claims). Gold-set-gated follow-up.
- **Re-verifying the interactive REFINE paths** (cover-letter `refine`, the post-refine résumé). The
  primary `generate_verified` / `tailor_verified` paths verify; an interactive refine is a fast
  human-in-the-loop edit that does NOT re-run the verifier. Surfaced honestly in-product ("not
  grounding-checked"), not silently assumed clean. Re-verifying refine is a follow-up.
