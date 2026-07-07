# Generated Document Quality Eval

`scripts/eval_document_quality.py` has two modes:

- Single-artifact smoke checks for one generated CV and/or cover letter.
- Private packet-set certification for generated CV + cover-letter packets.

The private packet set is local data and should not be committed. The default path is:

```bash
~/.job-applicator/document-quality-eval/packet-set.jsonl
```

Override it with:

```bash
DOCUMENT_QUALITY_SET=/path/to/packet-set.jsonl
```

## Run

```bash
# Smoke-check one generated packet.
.venv/bin/python scripts/eval_document_quality.py \
  --resume output/tailored_example.txt \
  --cover-letter output/cover_letter_example.txt \
  --keyword Python \
  --keyword SIEM

# Certify the private packet set. Missing private data exits non-zero.
.venv/bin/python scripts/eval_document_quality.py --packet-set --required

# Use an explicit manifest and machine-readable output.
.venv/bin/python scripts/eval_document_quality.py \
  --packet-set ~/.job-applicator/document-quality-eval/packet-set.jsonl \
  --required \
  --json
```

Without `--required`, a missing or empty packet set prints "not certified" and exits `0`.
With `--required`, missing or empty private evidence exits `2`. A present packet set that scores
below the quality bars exits `1`.

## Manifest

The manifest may be JSONL, a JSON list, or a JSON object with a `cases` list. Paths are resolved
relative to the manifest file unless they are absolute.

```json
{
  "id": "acme-security-analyst-2026-07",
  "resume_path": "artifacts/acme-security-resume.txt",
  "cover_letter_path": "artifacts/acme-security-cover-letter.txt",
  "applicant_name": "John Doe",
  "job_title": "Security Analyst",
  "company": "Acme",
  "keywords": ["Python", "Linux", "SIEM", "incident response", "IAM"]
}
```

Required fields:

- `resume_path`
- `cover_letter_path`
- either `keywords` or `job_description` / `job_description_path`

Optional fields:

- `id` / `packet_id` / `name`
- `applicant_name` / `profile_name`
- `job_title` / `title`
- `company` / `employer`
- `min_dimension_score` / `dimension_floor`
- `min_overall_score` / `overall_floor`

If `keywords` are omitted, the runner derives a small keyword set from the job description. Prefer
explicit keywords for stable certification.

## Scores

Each packet gets four deterministic 0-4 dimension scores:

- `usefulness`: document completeness plus job keyword coverage.
- `specificity`: packet and cover-letter job keyword coverage, title/company mentions, and generic
  cover-letter phrase penalties.
- `writing_quality`: cover-letter length, paragraph shape, repetition, and existing
  cover-letter failures/warnings.
- `formatting_polish`: contact/section/sign-off integrity, placeholders, markdown/list leakage, and
  obvious line-formatting issues.

Default bars:

- each dimension must be at least `3.0`
- each packet overall mean must be at least `3.0`

This gate complements grounding and human review. It catches obvious regressions in generated
packet usefulness and polish; it does not replace judgment about whether a packet is genuinely the
best possible application for a role.
