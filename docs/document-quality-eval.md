# Generated Document Quality Eval

`job-applicator document-quality` has two modes:

- Single-artifact smoke checks for one generated CV and/or cover letter.
- Private packet-set certification for generated CV + cover-letter packets.

`scripts/eval_document_quality.py` remains a compatibility wrapper for script-based gates and
supports the same scoring logic.

For criteria-extraction, ranking-repeatability, and rendered-template measurement, use
`scripts/eval_llm_sampler.py` as the experiment harness. It generates fresh private packet
manifests per sampler variant, certifies them through this evaluator, and reports both
baseline-relative score deltas and held-out deterministic measurements. Applicant claim prose is
deterministic and is not sampler-tuned. See `docs/llm-sampler-eval.md`.

The private packet set is local data and should not be committed. The default path is:

```bash
~/.job-applicator/document-quality-eval/packet-set.jsonl
```

Override it with:

```bash
DOCUMENT_QUALITY_SET=/path/to/packet-set.jsonl
```

Private gold standards are also local and should not be committed. The default root is:

```bash
~/.job-applicator/document-quality-eval/gold-standards/
```

## Run

```bash
# Smoke-check one generated packet.
job-applicator document-quality \
  --resume output/tailored_example.txt \
  --cover-letter output/cover_letter_example.txt \
  --keyword Python \
  --keyword SIEM

# Certify the private packet set. Missing private data exits non-zero.
job-applicator document-quality --private-packet-set --required --min-cases 15 \
  --min-manual-reviews-per-category 5 \
  --max-artifact-age-days 14 \
  --required-category support --required-category risk --required-category network \
  --required-language en --required-language fr

# Use an explicit manifest and machine-readable output.
job-applicator document-quality \
  --packet-set ~/.job-applicator/document-quality-eval/packet-set.jsonl \
  --required \
  --min-cases 3 \
  --max-artifact-age-days 14 \
  --json
```

In the TUI, select a job with saved generated text artifacts and press `D` to open the explicit
document-quality panel. If both a tailored CV and a cover letter are present and the row has matched
skills, the panel shows a packet quality check with packet dimensions, including coherence. If
matched skills are absent, it reports a limited smoke check instead of treating missing requirements
as source-backed keywords. Full private packet-set certification is only reported by the packet-set
CLI gate.

Without `--required`, a missing or empty packet set prints "not certified" and exits `0`.
With `--required`, missing or empty private evidence exits `2`. A present packet set that scores
below a hard integrity bar or has a reviewed prose defect exits `1`. Missing source/overlay evidence
or insufficient manual-review coverage exits `2`. In optional mode, automated packet scores remain
available as diagnostics but do not imply final certification.

Required JSON output separates the two evidence layers:

- `integrity_certified`: deterministic source retention, overlay provenance, language, freshness,
  structure, and category/language coverage all pass.
- `prose_qualified`: sampled human reviews meet the category count and dimension floors with no
  critical defects.
- `certified`: both fields are true.

## Manifest

The manifest may be JSONL, a JSON list, or a JSON object with a `cases` list. Paths are resolved
relative to the manifest file unless they are absolute.

```json
{
  "cases": [
    {
      "id": "acme-security-analyst-2026-07",
      "resume_path": "artifacts/acme-security-resume.txt",
      "cover_letter_path": "artifacts/acme-security-cover-letter.txt",
      "source_resume_path": "/path/to/base-cv.pdf",
      "resume_meta_path": "artifacts/acme-security-resume.meta.json",
      "protected_spans": ["UpClick", "certification exam pending"],
      "applicant_name": "John Doe",
      "job_title": "Security Analyst",
      "company": "Acme",
      "job_description": "Monitor security events and triage suspicious activity.",
      "job_requirements": ["Use a SIEM to investigate alerts."],
      "keywords": ["Python", "Linux", "SIEM", "incident response", "IAM"],
      "coherence_terms": ["Python", "Linux", "SIEM", "incident response"],
      "category": "support",
      "language": "en",
      "generated_at": "2026-07-07T14:30:00Z",
      "run_id": "doc-quality-20260707",
      "source_job_url": "https://example.test/jobs/123",
      "template": "modern",
      "format": "txt",
      "model": "Qwen/Qwen3-8B-AWQ",
      "generator_version": "job-applicator-0.5.0+source-overlay-v6"
    }
  ]
}
```

Required fields:

- `resume_path`
- `cover_letter_path`
- either `keywords` or `job_description` / `job_description_path`

Required integrity certification also requires `source_resume_path` (aliases:
`input_resume_path`, `base_resume_path`) and both document overlays. Résumé metadata may be inline as
`resume_overlay`, supplied through `resume_meta_path`, or read from the adjacent `.meta.json`.
Cover-letter metadata may be inline as `cover_letter_overlay`, supplied through
`cover_letter_meta_path`, or read from its adjacent `.meta.json`.
The evaluator independently recomputes the source and generated non-summary digests and checks the
declared statements, citations, and deterministic realizations. Current required evidence must use
`source-overlay-v6`; the evaluator recomputes the job-source digest, verifies every criterion's
exact evidence span, confirms that ranked fact IDs equal cited fact IDs, and requires the résumé and
cover-letter rankings to align. It never trusts a manifest's precomputed retention boolean as the
certification decision.

Optional fields:

- `id` / `packet_id` / `name`
- `applicant_name` / `profile_name`
- `job_title` / `title`
- `company` / `employer`
- `source_resume_path` / `input_resume_path` / `base_resume_path`
- `resume_meta_path` / `resume_metadata_path`
- `resume_overlay`
- `cover_letter_meta_path` / `cover_metadata_path`
- `cover_letter_overlay` / `cover_overlay`
- `protected_spans`
- `job_requirements` / `requirements`
- `coherence_terms` / `shared_terms`
- `min_dimension_score` / `dimension_floor`
- `min_overall_score` / `overall_floor`
- `category` / `job_category`
- `language`
- `generated_at`
- provenance fields passed through to JSON when present: `run_id`, `source_job_url`, `template`,
  `format`, `model`, `generator_version`

If `keywords` are omitted, the runner derives a small keyword set from the job description. Prefer
explicit keywords for stable certification.

Keep packet cases fresh. `generated_at` is used when present; otherwise freshness uses the oldest
mtime of the résumé and cover-letter artifacts, so one freshly touched file cannot hide a stale
packet half. Future `generated_at` timestamps fail required certification beyond a small clock-skew
window. A private manifest should point at the latest validated generated artifacts, not stale
scratch files under `output/`. Choose keywords that are both role-relevant and source-backed by
the applicant's CV/tailored packet. Do not include
unsupported JD-only terms such as tools, processes, or responsibilities the candidate cannot
honestly claim unless the case is explicitly testing that those terms stay absent.

Use `coherence_terms` when the packet should distinguish broad job specificity from the smaller
set of narrative terms that must appear in both the CV and cover letter. If omitted, the coherence
dimension uses `keywords`.

## Evidence Ranking

Target extraction and applicant evidence are intentionally separated:

1. `TargetCriteriaExtractor` receives only the job description and requirements. At temperature
   zero it returns four to six concrete responsibilities or tools with one exact, contiguous job
   evidence span each. Company boilerplate, location, schedule, compensation, education, years of
   experience, and generic traits are excluded.
2. The result is bound to the SHA-256 digest of that job text and cached by endpoint, model,
   extraction version, request shape, and digest. A span that is absent after whitespace
   normalization is rejected.
3. `JobMatcher.rank_source_facts()` embeds each criterion and every substantive immutable source
   fact with mxbai. `criterion-embedding-v1` scores each fact from its strongest criterion plus a
   small mean-of-top-three contribution, then resolves equal scores in source order.
4. Exactly three facts are selected. The résumé and cover letter consume the same ranking contract,
   and both sidecars serialize criteria, scores, strongest-criterion indexes, and fact IDs.

This is retrieval, not prose generation. A weak applicant-role match may correctly produce a low
manual usefulness or specificity score; it must not be hidden by inventing a closer claim.

## Scores

Each packet gets five automated 0-4 diagnostic scores:

- `usefulness`: document completeness plus job keyword coverage.
- `specificity`: packet and cover-letter job keyword coverage, title/company mentions, and generic
  cover-letter phrase penalties.
- `coherence`: applicant identity, target role/company mention, CV/cover-letter language
  consistency, and source-backed terms shared by both documents. Company aliases count when the
  manifest contains a formal name such as `WSP in Canada` but the letter naturally says `WSP`.
- `writing_quality`: cover-letter length, paragraph shape, repetition, and existing
  cover-letter failures/warnings.
- `formatting_polish`: contact/section/sign-off integrity, placeholders, markdown/list leakage, and
  obvious line-formatting issues.

The packet payload also includes `source_integrity`, `source_retention`, and `integrity_passed`.
`source_retention` reports recomputed body digests, protected-span recall, and both overlay
validations. These checks reject artifacts; they never rewrite generated prose.

Automated prose dimensions are useful for regression detection, but lexical heuristics cannot
certify factual, coherent prose. Final prose qualification therefore comes from a separate JSONL
review set. By default it sits beside `packet-set.jsonl` as `packet-set.reviews.jsonl`; override it
with `--manual-review-set`.

```json
{"packet_id":"support-r01","reviewer":"reviewer-id","reviewed_at":"2026-07-10T12:00:00Z","dimensions":{"usefulness":4,"specificity":3,"coherence":4,"writing_quality":3,"formatting_polish":4},"critical_defects":[]}
```

Every review must name a packet in the integrity-passing manifest, include all five dimensions on
the 0-4 scale, and declare critical defects explicitly. Reviews are unique by packet ID and count
toward coverage once per `source_job_url`. Required certification defaults to five independent
source jobs per required category; repetitions of one job do not increase coverage or median
weight.

Default bars:

- each dimension must be at least `3.0`
- each packet overall mean must be at least `3.0`
- required packet-set integrity certification needs at least `3` passing cases
- prose qualification needs at least `5` independent source-job reviews per required category
- optional packet-set scoring defaults to `1` passing case for certification metadata
- generated packets older than `14` days are stale

The automated scores detect obvious regressions; the manual sidecar records the judgment needed to
qualify prose. Neither layer is allowed to impersonate the other. Consequently, JSON can report
`passed=false` for lexical diagnostics while `certified=true` when deterministic integrity and the
independent prose-review contract both pass; required-mode exit status follows certification.

When updating a private packet, run the gate in required JSON mode and inspect both the score and
the prose:

```bash
job-applicator document-quality --private-packet-set --required --min-cases 15 \
  --min-manual-reviews-per-category 5 \
  --max-artifact-age-days 14 \
  --required-category support --required-category risk --required-category network \
  --required-language en --required-language fr \
  --json
```

The generated résumé and cover letter must both have valid deterministic source overlays.

For promotion, fix the case cohort and packet-selection rule before opening artifacts. Generate
fresh source-aware packets, require the sampler's held-out measurements to pass, review every
selected packet, certify the candidate manifest, back up the current private manifest, and only
then copy the candidate manifest and review sidecar into the default paths. Do not hand-edit a
generated document or add output-specific repair rules to make a packet pass.

## Gold Standards

Gold-standard bundles are human-authored private fixtures for target packet quality and future
coherence checks. They are not generated artifacts and should not be treated as packet-set cases.

The cover-letter v1 bundle uses this layout:

```text
~/.job-applicator/document-quality-eval/gold-standards/cover-letter-v1/
├── README.md
├── cover-letter-gold-standard.txt
├── cover-letter-prose-only.txt
├── cover-letter-prose.json
├── cover-letter-design-gold-standard.docx
└── cover-letter-gold-standard.meta.json
```

- `cover-letter-gold-standard.txt`: full business-letter exemplar, including applicant header,
  date, recipient, greeting, body, sign-off, and signature.
- `cover-letter-prose-only.txt`: greeting, body paragraphs, closing, and signature only. Use this
  as the style-guide input when you want the analyzer to extract prose without header/layout noise.
- `cover-letter-prose.json`: canonical extracted fields (`date`, `recipient`, `greeting`,
  `body_paragraphs`, `closing`, `signature`) for deterministic tests.
- `cover-letter-design-gold-standard.docx`: CV-coherent visual rendering for future format
  comparisons.
- `cover-letter-gold-standard.meta.json`: the hard contract: source materials, visual design,
  style, truth constraints, extraction fields, and coherence-check seeds.

Use the JSON/meta contract for hard assertions. Treat `StyleAnalyzer` output as soft style guidance:
LLM style summaries can be useful but may contain small semantic noise, so they should not replace
the deterministic gold metadata.
