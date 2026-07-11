# LLM Sampler Eval

`scripts/eval_llm_sampler.py` is a private-data measurement harness for Qwen/vLLM request-shape
experiments. It cannot tune deterministic applicant claim prose. It does not change production
defaults. It runs the public `job-applicator batch` command across
sampler variants, writes generated CV/cover-letter packets to a local run directory, then certifies
each variant with the same packet-set quality evaluator used by `job-applicator document-quality`.

Default private input:

```bash
~/.job-applicator/document-quality-eval/sampler-cases.jsonl
```

Default private output:

```bash
~/.job-applicator/document-quality-eval/sampler-runs/
```

Neither path should be committed.

## Run

Start with a dry run so the commands, output directories, and sampler env overrides are visible:

```bash
.venv/bin/python scripts/eval_llm_sampler.py --dry-run --json
```

Run a generation/integrity experiment. This can exit successfully before prose is manually
qualified, but it never reports final `certified=true` without the review sidecar:

```bash
.venv/bin/python scripts/eval_llm_sampler.py --required --integrity-only --json
```

Run repeated end-to-end TXT+PDF trials with deterministic template rotation:

```bash
.venv/bin/python scripts/eval_llm_sampler.py \
  --variant qwen-pp15 \
  --repetitions 3 \
  --format both \
  --rotate-templates \
  --required --integrity-only --json
```

Limit the run when debugging:

```bash
.venv/bin/python scripts/eval_llm_sampler.py \
  --case-id acme-security \
  --variant baseline \
  --variant qwen-pp12 \
  --required --integrity-only \
  --json
```

Missing optional sampler cases exit `0` with valid JSON and
`"reason": "missing_sampler_cases"`. With `--required`, missing/invalid/private-unavailable cases
exit `2`. Present generation or certification failures exit `1` only when `--required` is set.

## Variants

The harness currently compares these request shapes:

| Variant | Meaning |
|---|---|
| `baseline` | Ambient config with sampler env overrides removed. Job-criteria extraction remains pinned to temperature zero. |
| `qwen-grounded` | `top_p=0.8`, `top_k=20`, `min_p=0.0`, `presence_penalty=0.0`, `enable_thinking=false`. |
| `qwen-pp10` | `top_p=0.8`, `top_k=20`, `min_p=0.0`, `presence_penalty=1.0`, `enable_thinking=false`. |
| `qwen-pp12` | Same Qwen-shaped sampler with `presence_penalty=1.2`. |
| `qwen-pp15` | Same Qwen-shaped sampler with `presence_penalty=1.5`. |

The default comparison is `baseline` versus `qwen-grounded`. The positive-presence-penalty
variants remain available as explicit historical experiments; they are not default candidates
because the document task is source-constrained and repetition measures criteria-extraction/runtime yield. Variants
are applied through `JOB_APPLICATOR_LLM_*` environment overrides for the child `batch` process,
and the harness captures the exact overrides in JSON.

## Sampler Cases

The case file can be JSONL, a JSON list, or a JSON object with a `cases` list. Paths are resolved
relative to the case file unless absolute.

```json
{
  "id": "acme-security",
  "jobs_file": "jobs/acme-security.json",
  "resume_path": "/path/to/base-cv.pdf",
  "style_guide_path": "~/.job-applicator/document-quality-eval/gold-standards/cover-letter-v1/cover-letter-prose-only.txt",
  "applicant_name": "John Doe",
  "category": "support",
  "language": "en",
  "keywords": ["Python", "Linux", "SIEM", "incident response"],
  "coherence_terms": ["Python", "Linux", "SIEM", "incident response"],
  "top_k": 1,
  "min_score": 0.0,
  "format": "txt",
  "template": "modern"
}
```

Required case fields:

- `id`
- `jobs_file`
- `keywords` or a job description in the referenced jobs file
- `resume_path` when the harness runs with `--required`

Optional case fields:

- `resume_path` / `input_resume_path` / `base_resume_path`
- `style_guide_path` / `style_guide`
- `applicant_name` / `profile_name`
- `category` / `job_category`
- `language` / `expected_language`
- `coherence_terms` / `shared_terms`
- `top_k`
- `min_score`
- `format`
- `template`
- `protected_spans` (exact source phrases whose retention must be measured)

The generated packet manifest is separate from the sampler case file. After a successful batch run,
the harness writes `<output-root>/<run-id>/<variant>/packet-set.jsonl` with generated
`resume_path`, `cover_letter_path`, and `source_resume_path` values, provenance, freshness
timestamps, the source-fact generator version, and the case category/language metadata.
It also records the adjacent résumé metadata/overlay when available, recomputed source-body digests,
protected-span recall, the complete job description/requirements needed to verify ranking
provenance, PDF paths, and git HEAD/dirty/diff-hash provenance in the run summary.

For each case, the harness also copies the referenced `jobs_file` to
`<output-root>/<run-id>/<variant>/<case-id>/input-jobs.json` and passes that copy to `batch`. This
keeps batch recovery state isolated between sampler variants: `batch` auto-resumes incomplete runs
by processing spec, and sampler env values are not part of that public batch spec.

Each packet also receives its own `target-criteria-cache` directory. The résumé and cover letter
share that packet-local extraction, but variants and repetitions cannot reuse criteria from an
earlier packet. This makes `criteria_stability_rate` an independent-extraction measurement rather
than a shared-cache artifact. Production cache keys remain reusable but include endpoint, model,
request shape, extraction version, and job-source digest.

## Certification

Each variant's deterministic integrity layer is evaluated with:

- minimum dimension score `3.0`
- minimum overall score `3.0`
- minimum passing case count equal to the selected case count unless `--min-cases` overrides it
- max artifact age `14` days
- required categories/languages inferred from selected cases unless explicitly overridden
- exact non-summary source-body retention and valid résumé/cover-letter overlay provenance
- source and requested output languages must match; cross-language résumé cases fail generation
  until a same-language source resume is supplied

`--integrity-only` makes this layer the required experiment gate. It does not waive prose
qualification: the summary still reports `prose_qualified=false` and `certified=false` until a
separate manual review set passes the canonical `document-quality --required` command.

Use `--repetitions` to measure criteria extraction and evidence-selection stability rather than
relying on one run. Byte-identical repetitions of one source job do not count as independent
prose-review coverage. Add distinct source jobs to the case set for manual qualification.
`--format both` exercises deterministic TXT and PDF output; `--rotate-templates` cycles modern,
classic, and minimal. Packet IDs include `-rNN` so replicate failures remain traceable.

## Held-Out Measurements

Every non-dry run reports two deterministic measurement blocks:

- `evidence_ranking`: ranking provenance completeness, selected-fact stability, exact-criteria
  stability, and résumé/cover-letter alignment. Required runs demand `1.0` for every rate.
- `template_coherence`: TXT identity across modern/classic/minimal, expected template coverage,
  PDF text retention, and page counts. Required runs demand all three templates when rotation is
  requested, at least `0.99` token retention, no résumé over three pages, and exactly one cover
  page.

These thresholds are structural and deterministic. They do not score whether the selected evidence
is persuasive or whether the prose reads naturally; that remains the held-out manual review step.

The JSON summary includes per-case command/log paths, generated packet counts, failed case ids,
packet quality payloads, retention, ranking and template measurements, certification failures, and
the packet manifest path for each variant.
For non-dry runs, the same JSON is saved to
`<output-root>/<run-id>/sampler-summary.json` so long live results are preserved even if terminal
output is truncated.

## Baseline Comparison

The default run includes `baseline` and `qwen-grounded`, so the harness also emits a
`baseline_comparison` block. This
is the main evidence for whether the Qwen-shaped settings are actually better than current behavior.

For each non-baseline variant it reports:

- `overall_delta`: packet-set overall score difference vs baseline.
- `dimension_mean_deltas`: per-dimension score differences vs baseline.
- `generated_cases_delta`: generated packet count difference.
- `failed_case_count_delta`: generation failure count difference.
- `resolved_failed_cases` / `new_failed_cases`.
- `resolved_certification_failures` / `new_certification_failures`.
- `certified_change`: `improved`, `regressed`, `unchanged_certified`, or
  `unchanged_not_certified`.
- `better_than_baseline`: a conservative rank comparison using certification status, packet pass
  status, overall score, generation failures, and generated packet count.

Treat an architecture/settings candidate as eligible for manual prose review only when repeated
integrity yield meets the predeclared threshold, source-body retention is 100%, every held-out
measurement passes, and it adds no honesty failures. Fix the cohort and selection rule before
reading artifacts, then review all selected packets rather than cherry-picking heuristic winners.
Promote only after the review sidecar qualifies every category through the canonical packet gate.
