# Spec: domain-general skill grounding & normalization

**Status:** Proposed (design only — not yet scheduled). Interim correctness fixes shipped
separately (F-A/F-B/F-D, see "Interim" below).
**Motivation:** A live-CLI QA pass surfaced skill-matching bugs; fixing them exposed that the
skill *canonicalization + grounding* layer is keyword/hardcoded and software-domain-only. It will
not generalize to other domains (nursing, finance, legal, trades). Keyword matching is not enough.

## Current pipeline (measured 2026-06-26)

`match` skill coverage = **extract → normalize → ground → match**:

| Stage | Mechanism | Generalizes? |
|---|---|---|
| Extract skills from JD | LLM (Qwen3.5-4B via instructor) — `embeddings/skill_extraction.py` | ✅ LLM is domain-general |
| Normalize (`NORMALIZATION_MAP`) | **58 hardcoded tech aliases** — `skills/normalization.py` | ❌ zero non-software coverage |
| Hard-negatives (`HARD_NEGATIVE_SKILLS`) | hardcoded frozenset | ❌ tech/generic-tuned |
| Ground (`_is_grounded`) | **19 regex/substring ops** + compound/stopword heuristics, **0 embeddings** | ❌ pure keyword |
| Match skill↔requirement (`_match_skills`) | **embeddings (mxbai), cosine ≥ 0.75** (15 embed calls) — `embeddings/matching.py` | ✅ already semantic |

Key insight: the **matching is already semantic**; the embedding model is loaded but the grounding
step ignores it. The non-scalable parts are **normalization, hard-negatives, grounding** — wedged
between the (general) LLM and the (general) embedder.

## Goal

Domain-general skill canonicalization + grounding on the existing local stack (Qwen3.5-4B + mxbai,
both already loaded; 12 GB GPU), without heavy new dependencies, preserving the hallucination guard
that grounding provides (a skill must be *claimed in the text*, not merely plausible).

## Empirics — evidence-span fidelity probe (2026-06-26)

Before building any evidence-span plumbing, a live probe tested the approach's core assumption on
the local 4B (Qwen3.5-4B, temp 0): *does it emit evidence spans that are verifiable substrings of
the source?* Schematic prompt (no concrete example span — the 4B copies them verbatim), 4
descriptions across 2 software + 2 non-software domains, strict span check (lowercase + collapse
whitespace + strip punctuation).

**Result: 30/30 skills span-verified, 0% false-negative rate** — software (asyncio, FastAPI,
Kubernetes…), nursing (patient assessment, IV insertion, ventilator management, BLS, ACLS), and
finance (discounted cash flow models, variance analysis, Excel, CFA) all grounded by verbatim
spans. Strict span-grounding does NOT drop correct skills here, and it grounds exactly the
cross-domain skills the keyword/map grounding cannot.

**Decision: pursue evidence-span grounding** (approach #1 below). The embedding-grounding variant
stays the documented fallback (would have been chosen had span fidelity been low). One-off probe
kept at `scratchpad/probe_evidence_spans.py` (not committed to src).

## Proposed approach

### 1. Grounding: substring → verify-the-citation (LLM evidence spans)
Have the extractor return, per skill, the **exact source span** it is drawn from (structured output
already in use via instructor). Grounding = verify that span occurs in the source text (cheap,
exact, domain-general). A fabricated skill yields a span that fails verification → dropped — the
hallucination guard survives without a map or hand-tuned compound heuristics.
- **Fallback / lighter variant:** embedding-cosine grounding — embed skill vs source sentences,
  grounded if max cosine ≥ τ. No extra LLM tokens, reuses mxbai, handles paraphrase
  ("managed patients" ≈ "patient care"); weaker guard ("near the text" ≠ "claimed"). Needs τ tuned
  per [[embedding-threshold-empirics]] discipline.

### 2. Normalization: hardcoded map → canonical-from-LLM + embedding-dedup
LLM emits canonical names directly; collapse near-duplicates by embedding similarity. The 58-entry
map degrades to an optional fast-path/cache, not the source of truth.
- **North star:** link to a standard multi-domain ontology — **ESCO** (~13k skills, free, all
  occupations) / O*NET / Lightcast Open Skills — embedded once, queried by nearest-neighbor. Biggest
  lift; own project. Gives synonym handling + cross-domain canonical names for free.

### 3. Hard-negatives
A taxonomy distinguishes skills from generic traits inherently; until then keep a *small,
domain-agnostic* stop-list rather than a growing tech list.

## Migration (staged)

- **Interim (shipped):** F-A résumé multi-line comma-skill parse; F-B grounding stops dropping
  single-word skills before ordinary nouns (known-multiword gate, `_KNOWN_MULTIWORD_SKILLS` from the
  map values + `react native`); F-D short-skill (`Go`/`C#`) survival. These make the *current*
  software path correct; they are **not** the scaling answer — but F-B already nudges toward it:
  dropping the synthesize-a-compound-from-any-noun behavior keeps single-word skills grounded in
  *any* domain (e.g. "phlebotomy training" keeps `phlebotomy`), even though normalization stays
  software-only.
- **Phase 1 — IMPLEMENTED (default-off):** evidence-span grounding behind `skills.grounding_mode`
  (`keyword` default | `evidence_span`). `SkillEvidence` / `SkillExtractionOutputV2` schema, span
  verification under aggressive normalization, **mode in the cache key** (no cross-mode
  contamination), degrade-to-keyword fallback when structured output is unavailable, + unit guards
  and a deterministic cross-domain eval scaffold. Live-validated: a nursing job grounds *patient
  assessment / IV insertion / ventilator management* — domains keyword/map cannot reach.
  An **adversarial multi-agent code review (2026-06-27)** then hardened the guard: span
  verification is now **word-boundary-anchored** (a short span can't ground inside a larger word —
  "Ada" ⊄ "adaptable"), punctuation is a clause boundary (not a join), short span-verified skills
  survive (`R`/`Go`), and a degraded result is no longer cached under the evidence_span key. +7
  `extract()`-level / degrade / no-masking unit guards closed the test gap the review found.
  **Remaining:** the live multi-domain A/B (precision/recall vs keyword) and the default-on flip.
- **Phase 2:** embedding-dedup normalization; demote `NORMALIZATION_MAP` to a cache. Also the
  **name↔span coherence** check — catch a name/evidence mismatch (name `Java` for span
  `JavaScript`) WITHOUT dropping a legitimate canonicalization (name `PostgreSQL` for span
  `Postgres`). Deferred from Phase 1 because no string check separates the two (both are prefix
  relations); it needs name↔span embeddings.
- **Phase 3 (optional):** ESCO/taxonomy backbone.

## Phase-2 A/B pilot — no-gold objective metrics (2026-06-27)

Evidence_span vs keyword, temp 0, 5 descriptions × 4 domains (a first signal, not a benchmark;
recall-vs-gold is intentionally deferred — the gold set must come from the user / real sources,
not the method's author, to avoid designer-grades-own-work bias). `scratchpad/grounding_ab.py`.

- **Software no-regression (gates the default-on flip): evidence_span captured 9/9 and 7/7 of
  keyword's skills — zero regression on keyword's home domain.**
- **Cross-domain: keyword 0 / evidence_span 8 (nursing), 3 / 8 (finance), 0 / 8 (trades).**
  Keyword grounds nothing outside software (empty `NORMALIZATION_MAP`); evidence_span grounds the
  real domain skills (IV insertion, discounted cash flow models, NEC code compliance, …).
- **C-leak (name/evidence mismatch) rate: 0/40 = 0%** — the deferred-C case is not observed at
  temp 0; the deferral is data-validated (the model names its spans faithfully).
- **Guard activity: 0/40 spans dropped** at temp 0 — confirms temp 0 as the setting (the earlier
  single BLS drop was a temp-0.7 artifact; one 0.7 sample is noise).

**Read:** the default-on flip is justified by the no-regression + cross-domain signal. Remaining
before flipping: a larger N and a **user-blessed gold set** for the recall half.

## Dogfooding findings — real SOC hunt (2026-06-28)

Running the author's own CV through `search`→`match` against real Montréal SOC JDs (Indeed,
account-safe) surfaced four issues. One is fixed; three shape Phase-2 priorities:

- **Résumé tab-grid parser bug — FIXED (PR #95).** A two-column "Category⇥skill · skill" grid glued
  the first skill of each row to its label ("Networking⇥TCP/IP"), corrupting ~one skill/row and
  depressing every match. Parser now drops the leading "`<label>`⇥" prefix.
- **JD extraction noise — the real precision work (open).** `evidence_span` grounds whatever is
  literally in the JD text — including company-business ("auto spare parts" from a biotech,
  "protein engineering"), job *titles* ("Cybersecurity Specialist"), and *tier labels* ("Analyste
  SOC N2/N3"). **Grounded ≠ role-relevant.** This is the name↔role-relevance problem the redesign
  was already circling; highest-value Phase-2 precision item.
- **Cross-lingual matching — RE-DIAGNOSED; whole-JD translation is OFF the table.** Empirically,
  translating French JDs → English did *not* improve matches (one case got worse). The low French
  scores were mostly JD-vagueness + extraction-noise + genuine non-fit, not language. Minor
  residual: French skill *names* don't embed-match English CV names (canonical-naming facet, low).
- **Match-score ≠ role-fit (calibration).** Score = 60% semantic + 40% skill-coverage → biased
  toward skill-RICH JDs: detailed *intermediate* postings outscore vague *entry/junior* ones the
  candidate fits better. The score is a skill-overlap measure, not an apply/fit signal — surface
  that caveat in `match` output, or revisit scoring for skill-sparse JDs.

## Validation

- **Multi-domain eval set:** labelled (résumé, JD) pairs across ≥4 domains (software, nursing,
  finance, skilled trades) with expected matched/missing skills. Current harness is software-only.
- **Regression guards already added:** the QA harness now asserts the *positive* direction
  (a strong-overlap job reports a résumé skill as MATCHED) on a realistic wrapped-skills résumé, plus
  unit guards for F-A/F-B/F-D. Extend with cross-domain cases.
- **Threshold discipline:** any new cosine threshold tuned + documented per
  [[embedding-threshold-empirics]] (the 0.75 skill-match value was tuned only on tech).

## Risks / open questions

- LLM token cost + latency of evidence spans (mitigated: spans are short; cache by JD hash already
  exists). 4B span-citation reliability — mitigated because we *verify* the span (a bad citation
  fails closed). Beware few-shot span examples being copied verbatim ([[keyfigures-example-hallucination]]).
- Canonical-name consistency across calls (cache + deterministic prompting).
- Taxonomy choice (ESCO vs O*NET vs Lightcast) + licensing + ~13k-entry embedding footprint on a
  12 GB box (fits, but co-resident with vLLM + mxbai — measure).
- Where grounding runs: per-skill vs one batched verification call.
- Distinct-skill vs subskill-of-a-present-skill: the interim known-compound gate correctly drops
  bare `React` for "react native" (a distinct framework) but would also drop `AWS` for "AWS Lambda",
  where AWS is a *present superset* — wrong-ish. The redesign must tell these apart (taxonomy
  parent/child, or evidence-span keeps both). Pre-existing in the interim heuristic; not a regression.
