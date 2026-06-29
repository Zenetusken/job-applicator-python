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
- **Phase 1 — IMPLEMENTED · DEFAULT-ON (2026-06-28):** evidence-span grounding behind
  `skills.grounding_mode` (`evidence_span` default | `keyword` legacy). `SkillEvidence` /
  `SkillExtractionOutputV2` schema, span
  verification under aggressive normalization, **mode in the cache key** (no cross-mode
  contamination), degrade-to-keyword fallback when structured output is unavailable, + unit guards
  and a deterministic cross-domain eval scaffold. Live-validated: a nursing job grounds *patient
  assessment / IV insertion / ventilator management* — domains keyword/map cannot reach.
  An **adversarial multi-agent code review (2026-06-27)** then hardened the guard: span
  verification is now **word-boundary-anchored** (a short span can't ground inside a larger word —
  "Ada" ⊄ "adaptable"), punctuation is a clause boundary (not a join), short span-verified skills
  survive (`R`/`Go`), and a degraded result is no longer cached under the evidence_span key. +7
  `extract()`-level / degrade / no-masking unit guards closed the test gap the review found.
  **Done:** the live A/B (Montréal SOC, below) and the default-on flip (2026-06-28).
- **Phase 2 (embedding-dedup half) — DEAD END, measured 2026-06-28** (same wall as C; see
  "Phase-2 embedding-dedup" below). The *canonical-from-LLM* half effectively shipped with the
  evidence_span default (the role-relevance prompt emits canonical English names + translates), and
  the `NORMALIZATION_MAP` is **already demoted in effect**: under evidence_span it no longer grounds
  (only `normalize_skill` runs as a software fast-path), and the downstream embedding *match*
  (cosine ≥ 0.75) collapses real synonyms regardless of canonical form — so normalization is mostly
  cosmetic and does not move match accuracy. **Embedding-dedup to replace the map is non-viable**
  (measured below). The only real cross-domain canonicalizer is the **Phase-3 taxonomy** (a knowledge
  base, not cosine). Net: no Phase-2 build; keep the map as the software fast-path.
- **C (name↔span coherence) — DEAD END, measured 2026-06-28.** The idea was to catch a name/evidence
  mismatch (name `Java` for span `JavaScript`) WITHOUT dropping a legit canonicalization (`PostgreSQL`
  /`Postgres`) or cross-lingual pair. No *string* check separates them (both prefix relations), so
  the hope was *embeddings*. Measured (mxbai): **not separable** — `Java`↔`JavaScript` cosine
  **0.756** is *higher* than legit `Kubernetes`↔`k8s` (0.588), `IDS/IPS`↔`intrusion detection`
  (0.667), and cross-lingual `incident response`↔`réponse aux incidents` (0.738). Any threshold that
  catches the mismatch drops real skills. Combined with C-leak measured at **0%** in practice, C is
  **permanently deferred** unless a fundamentally different mechanism appears (e.g. an LLM judge, or
  a skills taxonomy — Phase 3). Do not attempt embedding-coherence for C.
- **Phase 3 (optional):** ESCO/taxonomy backbone.

## Phase-2 embedding-dedup — measured dead end (2026-06-28)

Same non-separability as C, re-measured on the **real Montréal SOC skill set** (mxbai cosine). Can a
single threshold merge same-skill/different-name pairs WITHOUT merging distinct skills?

- **SHOULD-merge (same skill):** Azure↔Microsoft Azure 0.923, cloud↔Cloud-based technologies 0.751,
  incident response↔réponse aux incidents 0.737, Administration de réseau↔Network administration
  0.692, Kubernetes↔k8s 0.607.
- **SHOULD-NOT-merge (distinct):** Administration de réseau↔Sécurisation de réseau **0.789**, Cadres
  de sécurité↔systèmes de sécurité 0.785, ↔architectures de sécurité 0.770, Java↔JavaScript 0.755,
  Gestion de crise↔Gestion de serveurs 0.745, SIEM↔SOAR 0.506.
- **min(should-merge) 0.607 < max(should-NOT) 0.789 → NOT separable.** Distinct French security
  skills outscore legit cross-lingual merges; a threshold ≤0.607 (to catch k8s↔Kubernetes) merges
  every distinct security pair, one >0.789 catches only string-trivial variants string-matching
  already handles.

**Read:** do not build embedding-dedup normalization (`scratchpad/` probe is the record). The map
stays the software fast-path — low matching impact, since the downstream embedding match (cosine
≥ 0.75) already collapses real synonyms regardless of canonical form. Reliable cross-domain/-lingual
canonicalization needs the **Phase-3 taxonomy** (ESCO/O*NET), a separate project — not cosine.

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

**Read:** the default-on flip is justified by the no-regression + cross-domain signal. The larger-N,
user-blessed recall signal that gated it landed via the live SOC hunt below → flipped 2026-06-28.

## Default-on flip — live Montréal SOC A/B (2026-06-28)

The gating evidence came from the author's own hunt, not a synthetic set: **42 unique real Montréal
Indeed JDs** (4 FR/EN queries, deduped), his real CV, keyword vs evidence_span via the CLI
`match --json` (`scratchpad/match_{keyword,evidence_span}.json`). Objective, no-gold metrics:

- **Coverage (jobs with *measured* skill coverage, not collapsed to semantic-only):** overall
  keyword **15/42 (36%)** → evidence_span **36/42 (86%)**; **French 10/33 (30%) → 30/33 (91%)**;
  English 5/9 (56%) → 6/9 (67%).
- **Zero regression:** all **15/15** keyword-measured jobs stay measured under evidence_span.
- **Ranking:** the flagship target *"Analyste SOC" (Alten)* went **#9 62% semantic-only → #1 77%**
  (grounded: network monitoring · SOC monitoring & triage · incident response); every
  SOC/security-analyst-titled role ranked ≥ under evidence_span.

**Mechanism:** keyword grounding keeps only software skills in the `NORMALIZATION_MAP`, so
French/non-software SOC skills are dropped → coverage unmeasured → semantic-only → buried (and a
French JD embeds weakly vs an English CV). In Montréal's French-majority market that systematically
buried the most-relevant roles. evidence_span verifies the skill's span is *in the JD text*
(language/domain-agnostic), recovering them — the cross-lingual issue's real fix on the *grounding*
axis (whole-JD *translation* on the semantic axis stays off the table, measured earlier).

**Flip:** `SkillConfig.grounding_mode` default `keyword`→`evidence_span` (+ the `JobMatcher` /
`LLMSkillExtractor` param defaults, so the tailor flow's bare-fallback matcher stays consistent).
Reversible via `[skills] grounding_mode = "keyword"` or `JOB_APPLICATOR_SKILLS_GROUNDING_MODE=keyword`;
the degrade-to-keyword fallback keeps it safe where structured output is unavailable.

## Dogfooding findings — real SOC hunt (2026-06-28)

Running the author's own CV through `search`→`match` against real Montréal SOC JDs (Indeed,
account-safe) surfaced four issues. One is fixed; three shape Phase-2 priorities:

- **Résumé tab-grid parser bug — FIXED (PR #95).** A two-column "Category⇥skill · skill" grid glued
  the first skill of each row to its label ("Networking⇥TCP/IP"), corrupting ~one skill/row and
  depressing every match. Parser now drops the leading "`<label>`⇥" prefix.
- **JD extraction noise — ADDRESSED via a role-relevance prompt.** `evidence_span` was grounding
  whatever is literally in the JD text — company-business ("protein engineering" from a biotech),
  job *titles* ("Analyste SOC N2/N3"), *tier labels*. **Grounded ≠ role-relevant.** Fix:
  `SKILL_SYSTEM_PROMPT_EVIDENCE` is now scoped to "skills the CANDIDATE must have for THIS role;
  exclude the company's business, the job title, and tier labels." Measured (drops noise, keeps
  real skills) and — the discriminating adversarial test — it does NOT strip a *security firm's*
  SIEM/IDS/EDR as "company business" (broad and narrow exclusion clauses gave identical output, so
  the broad clause shipped per the pre-registered rule). Guarded by a live eval
  (`scripts/eval_extraction_precision.py`, OUT of the unit gate — a prompt can't have a unit test;
  4/4 cases clean). **Scope note:** the prompt now also *incidentally* canonicalizes/translates
  names (French JD → English names), drifting the name from its verbatim evidence span — which
  makes the deferred **C** (name↔evidence coherence) MORE needed, not less. The English-naming is
  incidental and not relied upon.
- **Cross-lingual matching — RE-DIAGNOSED; whole-JD translation is OFF the table.** Empirically,
  translating French JDs → English did *not* improve matches (one case got worse). The low French
  scores were mostly JD-vagueness + extraction-noise + genuine non-fit, not language. Minor
  residual: French skill *names* don't embed-match English CV names (canonical-naming facet, low).
- **Match-score ≠ role-fit (calibration) — SURFACED, not re-scored (2026-06-28).** Score = 60%
  semantic + 40% skill-coverage → biased toward skill-RICH JDs: detailed *intermediate* postings
  outscore vague *entry/junior* ones the candidate fits better. The score is a skill-overlap
  measure, not an apply/fit signal. **Decision: surface the caveat; do NOT change the scoring
  algorithm** — a re-score for skill-sparse JDs needs its own empirics + A/B and risks regressing
  the tuned 60/40 blend, whereas the honesty fix is high-value and low-risk. Shipped: a
  decision-framed caption on the ranking surfaces (`match`, `status`, `batch`) — *"skill-overlap,
  not role-fit; sparse/junior JDs rank low — don't skip on score alone"* (the `match` caption adds
  a `raise -k` hint, since `top_k` defaults to 5 and the low-ranked roles the caveat is about can
  sit below the cut; the TUI list is scrollable and per-card-marked, so it needs no such hint);
  `match --json` now exposes `semantic_score` / `skill_score` / `coverage_measured` (the caption's
  machine form); and the TUI detail pane fixes the **semantic-only misread** — a JD with no
  requirements has `skill_score` 0.0 *by convention*, which was rendered as a real "skill 0%" and
  is now "coverage n/a — none listed". A single shared predicate `models.coverage_measured()`
  defines the semantic-only case for the scorer (`_combined_score`) and both renderers, so the
  convention has exactly one home. Guards: unit tests on the helper, the `--json` shape (measured
  *and* semantic-only), and the TUI rendered string (both branches).

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
