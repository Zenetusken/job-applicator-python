# LLM Sampler Eval

`scripts/eval_llm_sampler.py` is a private-data measurement harness for Qwen/vLLM sampler tuning.
It does not change production defaults. It runs the public `job-applicator batch` command across
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

Run the full comparison:

```bash
.venv/bin/python scripts/eval_llm_sampler.py --required --json
```

Limit the run when debugging:

```bash
.venv/bin/python scripts/eval_llm_sampler.py \
  --case-id acme-security \
  --variant baseline \
  --variant qwen-pp12 \
  --required \
  --json
```

Missing optional sampler cases exit `0` with valid JSON and
`"reason": "missing_sampler_cases"`. With `--required`, missing/invalid/private-unavailable cases
exit `2`. Present generation or certification failures exit `1` only when `--required` is set.

## Variants

The harness currently compares these request shapes:

| Variant | Meaning |
|---|---|
| `baseline` | Ambient config with sampler env overrides removed. This preserves the current omitted-field request shape unless sampler knobs are set in `config.toml`. |
| `qwen-pp10` | `top_p=0.8`, `top_k=20`, `min_p=0.0`, `presence_penalty=1.0`, `enable_thinking=false`. |
| `qwen-pp12` | Same Qwen-shaped sampler with `presence_penalty=1.2`. |
| `qwen-pp15` | Same Qwen-shaped sampler with `presence_penalty=1.5`. |

The Qwen-shaped variants are applied through `JOB_APPLICATOR_LLM_*` environment overrides for the
child `batch` process. The harness captures the exact env overrides in JSON.

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

The generated packet manifest is separate from the sampler case file. After a successful batch run,
the harness writes `<output-root>/<run-id>/<variant>/packet-set.jsonl` with generated
`resume_path` and `cover_letter_path` values, provenance, freshness timestamps, and the case
category/language metadata.

## Certification

Each variant is certified with:

- minimum dimension score `3.0`
- minimum overall score `3.0`
- minimum passing case count equal to the selected case count unless `--min-cases` overrides it
- max artifact age `14` days
- required categories/languages inferred from selected cases unless explicitly overridden

That means a variant must generate a passing packet for every selected case by default. Use
`--min-cases` only when deliberately measuring partial coverage.

The JSON summary includes per-case command/log paths, generated packet counts, failed case ids,
packet quality payloads, certification failures, and the packet manifest path for each variant.

## Baseline Comparison

The default run includes `baseline`, so the harness also emits a `baseline_comparison` block. This
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

Treat a variant as a migration candidate only when it improves or preserves certification, does not
add generation failures, and improves the target dimensions for the cases under review. The harness
measures deterministic quality gates; the generated prose should still be inspected before changing
defaults.
