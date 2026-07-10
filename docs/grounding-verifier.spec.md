# Grounded Document Generation

**Status:** Current architecture (2026-07-10)

## Decision

Generated prose is never repaired after the model returns it. Phrase replacement, tool and
credential stripping, metric rewriting, language-specific grammar repair, context-specific bullet
deletion, and automatic grounding-driven refinement are not part of the document pipeline.

A draft is accepted unchanged or rejected. Transport cleanup such as removing a model thinking
prefix and deterministic assembly of the cover-letter application frame are not prose correction.

## Source Boundary

The authoritative applicant source is always the base résumé. The job posting is targeting context,
not applicant evidence. A generated document is never verified against the job posting or a
previously generated résumé.

Résumé generation requires the source and output to use the same language. Cross-language output
needs a separately certified translation system or a source résumé already written in the target
language; it must not be approximated with sentence-repair rules.

## Cover Letters

Cover generation has four boundaries:

1. Structured job requirements are used directly. If they are absent, the existing
   evidence-span skill extractor derives grounded criteria from a bounded job-description head and
   tail at temperature zero.
2. Local token-overlap ranking selects exactly three distinct primary source facts against those
   criteria. A model does not select applicant evidence.
3. Local typed realization converts the three facts into exactly three statements, one fact ID per
   statement and every ID used once. No model writes or paraphrases applicant claims.
4. The application opening, closing request, and sign-off are assembled deterministically. Artifact
   verification recomputes every statement from its cited source fact.

There is no factual prose model. Style input cannot rewrite factual statements.

## Résumés

Résumé targeting has one generated field:

1. The same grounded target criteria feed local deterministic ranking of exactly three substantive
   source fact IDs. No model sees or selects applicant evidence.
2. Local typed realization preserves the three selected fact texts as summary statements, one fact
   ID each and every ID used once.
3. Artifact verification recomputes every statement from its cited source fact.

`ResumeDocument` replaces or inserts that summary while preserving the canonical preamble and every
non-summary section byte-for-byte after transport whitespace normalization. `ResumeOverlay` records
the three statements, citations, source-body digest, language, and architecture
version. Verification fails closed if any part of that provenance or body digest differs.

Job-derived tone power words do not enter claim realization. User input may affect source-fact
focus only; style guides cannot rewrite claim text. Interactive retry and input regenerate the
summary from the original résumé; the source body is never editable through the tailoring loop.

PDF formatting uses the same canonical section structure and deterministic paragraph/sign-off
parsing. It does not call an LLM.

The legacy claim-enumeration verifier remains available as a standalone diagnostic for non-overlay
documents. It is not called by source-overlay generation or certification and cannot mutate an
artifact.

## Certification

Required private packet certification has two independent layers. Deterministic integrity requires
`source_resume_path`, exact non-summary digest retention, valid résumé and cover-letter overlay
provenance, deterministic realization, protected-span retention, language
policy, and artifact structure. Prose qualification requires a
structured human-review sidecar with sampled category coverage, dimension medians at or above the
configured floor, and no critical defects. `certified` is true only when both layers pass.

The French category cannot certify from an English source résumé. A mechanically passing packet that
omits a source qualifier also cannot be promoted. Private artifacts and manifests remain local.
