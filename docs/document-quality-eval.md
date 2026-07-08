# Generated Document Quality Eval

`job-applicator document-quality` has two modes:

- Single-artifact smoke checks for one generated CV and/or cover letter.
- Private packet-set certification for generated CV + cover-letter packets.

`scripts/eval_document_quality.py` remains a compatibility wrapper for script-based gates and
supports the same scoring logic.

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
job-applicator document-quality --private-packet-set --required --min-cases 3 \
  --max-artifact-age-days 14 \
  --required-category support --required-category risk \
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
below the quality bars exits `1`. In optional mode, a present set can have passing packet rows but
still report `"certified": false` when it does not meet set-level certification breadth, freshness,
or diversity thresholds.

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
      "applicant_name": "John Doe",
      "job_title": "Security Analyst",
      "company": "Acme",
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
      "generator_version": "0.5.0"
    }
  ]
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

Keep packet cases fresh. `generated_at` is used when present; otherwise freshness uses the newest
mtime of the résumé and cover-letter artifacts. A private manifest should point at the latest
validated generated artifacts, not stale scratch files under `output/`. Choose keywords that are
both role-relevant and source-backed by the applicant's CV/tailored packet. Do not include
unsupported JD-only terms such as tools, processes, or responsibilities the candidate cannot
honestly claim unless the case is explicitly testing that those terms stay absent.

Use `coherence_terms` when the packet should distinguish broad job specificity from the smaller
set of narrative terms that must appear in both the CV and cover letter. If omitted, the coherence
dimension uses `keywords`.

## Scores

Each packet gets five deterministic 0-4 dimension scores:

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

Default bars:

- each dimension must be at least `3.0`
- each packet overall mean must be at least `3.0`
- required packet-set certification needs at least `3` passing cases
- optional packet-set scoring defaults to `1` passing case for certification metadata
- generated packets older than `14` days are stale

This gate complements grounding and human review. It catches obvious regressions in generated
packet usefulness and polish; it does not replace judgment about whether a packet is genuinely the
best possible application for a role.

When updating a private packet, run the gate in required JSON mode and inspect both the score and
the prose:

```bash
job-applicator document-quality --private-packet-set --required --min-cases 3 \
  --max-artifact-age-days 14 \
  --required-category support --required-category risk \
  --required-language en --required-language fr \
  --json
```

The generated cover letter should also have a clean grounding report when it was produced through
the verified generation path.

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
