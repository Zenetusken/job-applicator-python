#!/usr/bin/env python
"""Regression harness for the matcher: score the labeled gold set through the REAL pipeline.

Run whenever matching logic changes (weights, thresholds, grounding, target-role boosts):

    .venv/bin/python scripts/eval_matching.py

Requires a labeled gold set (personal data, NOT in the repo) at
``~/.job-applicator/matching-eval/gold-set.csv`` — override with ``GOLD_SET_CSV``. Jobs are
read from the funnel DB and scored end-to-end via ``JobMatcher.rank_jobs`` (embeddings +
skill extraction + combined score + any ``[matching] target_roles`` boosts from the live
config), so the numbers reflect exactly what ``match`` would produce today.

Labels are the 0-4 graded rubric (see LABELING.md next to the CSV): 4 cyber · 3 admin+sec
· 2 admin · 1 other-IT · 0 non-IT. Reports Spearman(score, label), per-label score bands,
the best >=3 (security-family) cutoff, and the costly errors at the FIXED review floor —
fixed because re-optimizing the cutoff per run lets a variant "improve" by moving the
goalpost (observed 2026-07-02: a boost variant showed phantom FNs that way).

Exit codes (modest teeth): 1 when Spearman < 0.6 (the STRONG bar the 2026-07 calibration
set) or a label<=1 job scores at/above the fixed floor; else 0. Skips (exit 0, message)
when no labeled gold set exists.
"""

from __future__ import annotations

import asyncio
import csv
import os
import sys
from pathlib import Path

# The 2026-07 calibration's empirical review floor on the combined score: everything
# at/above it was security-family (precision 1.00). Boosted variants are judged at this
# SAME floor, not a re-optimized one.
REVIEW_FLOOR = 0.469
SPEARMAN_BAR = 0.6


def _avg_ranks(vals: list[float]) -> list[float]:
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman(a: list[float], b: list[float]) -> float:
    ra, rb = _avg_ranks(a), _avg_ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb, strict=True))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((y - mb) ** 2 for y in rb) ** 0.5
    return cov / (va * vb) if va and vb else 0.0


def _run() -> int:
    gold_csv = Path(
        os.environ.get(
            "GOLD_SET_CSV",
            os.path.expanduser("~/.job-applicator/matching-eval/gold-set.csv"),
        )
    )
    if not gold_csv.exists():
        print(f"no gold set at {gold_csv} — nothing to evaluate (not a failure)")
        return 0
    rows = [r for r in csv.DictReader(open(gold_csv, encoding="utf-8")) if r["relevant"].strip()]
    if not rows:
        print("gold set has no labels yet — nothing to evaluate (not a failure)")
        return 0
    labels_by_url = {r["url"]: int(r["relevant"]) for r in rows}

    from job_applicator.config import AppSettings
    from job_applicator.documents.resume import ResumeLoader
    from job_applicator.embeddings.matching import JobMatcher
    from job_applicator.jobs_store import JobStore

    settings = AppSettings()
    if not settings.resume_path:
        print("no resume_path configured — cannot score (not a failure)")
        return 0
    resume = ResumeLoader().load(settings.resume_path)

    # Fetch the WHOLE funnel (a high cap, not len+N — a restrictive limit can drop labeled jobs
    # that live outside the most-recent window, silently grading a non-representative subset).
    stored = JobStore().list_jobs(limit=1_000_000)
    jobs = [s.job for s in stored if str(s.job.url) in labels_by_url]
    missing = len(labels_by_url) - len(jobs)
    if not jobs:
        print("no labeled jobs found in the funnel DB — nothing to evaluate (not a failure)")
        return 0
    if missing:
        # Partial coverage must NOT exit 0 as "validated": a regression on an absent labeled job
        # would pass unseen. Refuse to certify (distinct exit 2) rather than grade the subset.
        print(
            f"INCOMPLETE COVERAGE: {missing}/{len(labels_by_url)} labeled jobs are not in the "
            f"funnel DB (re-scrape/wipe drift). Cannot certify the matcher on a partial gold set "
            f"— re-scrape to restore them, then re-run."
        )
        return 2

    matcher = JobMatcher(
        settings.embedding,
        settings.llm,
        grounding_mode=settings.skills.grounding_mode,
        matching=settings.matching,
    )
    matches = asyncio.run(matcher.rank_jobs(resume, jobs, top_k=len(jobs)))

    scores = [m.score for m in matches]
    labels = [labels_by_url[str(m.job.url)] for m in matches]
    rho = _spearman(scores, labels)
    verdict = "STRONG" if rho >= SPEARMAN_BAR else "MODERATE" if rho >= 0.3 else "WEAK"
    print(f"scored {len(matches)} labeled jobs through the live pipeline")
    print(f"Spearman(score, label 0-4) = {rho:+.3f}  -> {verdict}")

    print("\nper-label score bands:")
    for lbl in (4, 3, 2, 1, 0):
        ss = sorted(s for s, lab in zip(scores, labels, strict=True) if lab == lbl)
        if ss:
            print(
                f"  {lbl}: n={len(ss):2d}  mean {sum(ss) / len(ss):.3f}  "
                f"range {ss[0]:.3f}-{ss[-1]:.3f}"
            )

    boosted = [m for m in matches if m.target_role]
    if boosted:
        print("\ntarget-role boosts applied:")
        for m in boosted:
            print(f"  [{m.target_role}] {m.score:.3f}  {m.job.title[:56]}")

    fns = [
        (m.score, m.job.title)
        for m, lab in zip(matches, labels, strict=True)
        if lab == 4 and m.score < REVIEW_FLOOR
    ]
    fps = [
        (m.score, m.job.title)
        for m, lab in zip(matches, labels, strict=True)
        if lab <= 1 and m.score >= REVIEW_FLOOR
    ]
    print(f"\nat the FIXED review floor {REVIEW_FLOOR}:")
    print(f"  missed cyber (label 4 below): {len(fns)}")
    for s, t in sorted(fns):
        print(f"    {s:.3f}  {t[:60]}")
    print(f"  non-security at/above (label <=1): {len(fps)}")
    for s, t in sorted(fps, reverse=True):
        print(f"    {s:.3f}  {t[:60]}")

    failed = rho < SPEARMAN_BAR or bool(fps)
    verdict_line = "REGRESSION" if failed else "OK"
    print(f"\n{verdict_line} (bars: Spearman >= {SPEARMAN_BAR}, zero label<=1 above the floor)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run())
