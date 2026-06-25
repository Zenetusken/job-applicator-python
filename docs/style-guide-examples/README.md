# Style Guide Examples

These files are example résumés and cover letters you can use as **style guides**
for the AI-generated outputs in `job-applicator`. The tool does not copy their
content; it analyzes their *writing style* (tone, sentence structure, vocabulary,
greeting/closing style, paragraph style, and key phrases) and asks the LLM to
mimic that style when tailoring your résumé or writing your cover letter.

## Files

| File | Style | Best for |
|------|-------|----------|
| `01_enterprise-formal.txt` | Polished, structured, metrics-heavy | Large companies, government, finance, enterprise IT |
| `02_startup-warm.txt` | Friendly, conversational, human | Startups, customer success, community-focused teams |
| `03_technical-minimal.txt` | Sparse, direct, checklist-driven | Engineering/SRE roles, technical screeners |
| `04_narrative-storyteller.txt` | Story-driven, emotional, memorable | Mission-driven orgs, non-profits, standout applications |
| `05_modern-impact.txt` | Bold, confident, outcome-focused | Fast-growing tech companies, senior/lead roles |

## How to use

### CLI

Single style guide:

```bash
job-applicant generate-cover-letter \
  --resume /path/to/cv.pdf \
  --job-title "IT Support Specialist" \
  --company "Acme Corp" \
  --style-guide docs/style-guide-examples/01_enterprise-formal.txt
```

Combine several to create a blended style:

```bash
job-applicant tailor \
  --resume /path/to/cv.pdf \
  --job-title "IT Support Specialist" \
  --company "Acme Corp" \
  --style-guide "docs/style-guide-examples/02_startup-warm.txt,docs/style-guide-examples/05_modern-impact.txt"
```

### TUI

1. Launch `job-applicant` (or `job-applicant tui`).
2. Press `g` to open the **Style guide** modal.
3. Enter the path to one file, or several comma-separated paths.
4. Select a job and press `t` (tailor) or `c` (cover letter).

The style guide will be forwarded to the LLM for that session and persisted to
`config.toml`.

## Tips for strong results

- The example should be **at least a few hundred words**. A single paragraph does
  not give the analyzer enough signal.
- Use the kind of document you want the AI to produce. Cover-letter examples
  work best for cover letters; résumé examples work best for tailored résumés.
- You can mix examples. The analyzer merges their styles, so combining a formal
  example with an impact-focused example can yield a "polished but confident"
  voice.
- If you want to clear the style guide, press `g` and submit an empty path.

## Writing your own

A good custom style guide has:

- A clear greeting and closing (so the analyzer can detect greeting/closing style).
- Distinct sentence patterns (short punchy vs. long flowing vs. checklist).
- Specific vocabulary you want reused (e.g., "stakeholder," "first-call resolution," "root cause").
- A consistent tone throughout (formal, warm, technical, narrative, etc.).

Save it as a `.txt`, `.pdf`, `.docx`, or even an image, and point the tool at it.
