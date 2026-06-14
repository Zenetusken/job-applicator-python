# Live Systems Test Report

## Commands Run
- job-applicator ats-check (original resume)
- job-applicator match --jobs-file /tmp/live_jobs.json
- job-applicator batch --jobs-file /tmp/live_jobs.json --top-k 3 --no-cover-letter

## Original Resume ATS
Score: 100% compatible

## Batch Results
- Technical Support Specialist at Bell: 65.8% match
- Senior Software Developer at RBC: 34.6% match
- Full Stack Developer at Shopify: 28.0% match

## Tailored Resume ATS Scores (current test)
- tailored_Bell_Technical_Support_Specialist_20260614_144841.txt: 100% compatible
- tailored_RBC_Senior_Software_Developer_20260614_144841.txt: 100% compatible
- tailored_Shopify_Full_Stack_Developer_20260614_144841.txt: 100% compatible

## Log Files
- live_ats_bell.log
- live_ats_bell_final.log
- live_ats_check.log
- live_ats_rbc.log
- live_ats_rbc_final.log
- live_ats_shopify.log
- live_ats_shopify_final.log
- live_batch.log
- live_batch_final.log
- live_batch_rerun.log
- live_match.log

## Issues Found and Fixed During Test
1. ResumeLoader skills parser matched any occurrence of "skills" (e.g. "interpersonal skills") causing inflated skill lists.
2. ResumeLoader did not detect **Professional Summary** headers, causing summary to include markdown bold prefix.
3. LLM sometimes omitted explicit **Skills** and **Experience** headers. Strengthened system prompt to require them.
