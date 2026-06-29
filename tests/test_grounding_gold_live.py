"""Live gold-set measurement of the GroundingVerifier (spec §3 #3, §9 gate 3).

REPORTED, not a fast-gate unit test — a nondeterministic live LLM call (needs vLLM at :8000), so
it lives at the tests/ root and is auto-marked ``live``. It runs the verifier against the seed gold
set and prints recall + precision (both directions, per category).

Precision is scored in TWO partitions (spec §7/§8): grounded cases tagged ``"residual": true`` are
the NAMED residual — cross-language / low-overlap faithful groundings the model under-grounds —
reported separately and NOT gated. **CORE precision** (every other grounded case) carries the strict
gross-regression floor: those cases ground reliably (measured 0 false positives over N=5), so a core
false positive is a real regression, not the accepted residual. The 0.9 targets are tracked.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from job_applicator.config import AppSettings
from job_applicator.documents.grounding_verifier import GroundingVerifier
from job_applicator.models import ResumeData

GOLD = Path(__file__).parent / "data" / "grounding_gold.json"


async def test_grounding_gold_set_precision_recall() -> None:
    gold = json.loads(GOLD.read_text())
    resume = ResumeData(raw_text=gold["source"], skills=[])
    verifier = GroundingVerifier(AppSettings().llm)

    tp = fn = 0  # fabricated caught / missed
    core_fp = core_tn = 0  # grounded, NOT residual-tagged
    res_flagged = res_total = 0  # grounded, residual-tagged (the named §7/§8 residual)
    cat_recall: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])  # [caught, total]
    miss_notes: list[str] = []  # fabricated misses + CORE false positives (both are regressions)
    res_notes: list[str] = []  # residual flags (expected, reported)

    for case in gold["cases"]:
        report = await verifier.verify(case["claim"], resume)
        flagged = bool(report.unsupported) or bool(report.coverage_gaps)
        tag = f"{case['lang']}/{case['category']}"
        if case["label"] == "fabricated":
            if flagged:
                tp += 1
            else:
                fn += 1
                miss_notes.append(f"MISS [{tag}]: {case['claim'][:50]}")
            cat_recall[case["category"]][0] += int(flagged)
            cat_recall[case["category"]][1] += 1
        elif case.get("residual"):  # grounded, named residual
            res_total += 1
            if flagged:
                res_flagged += 1
                res_notes.append(f"residual [{tag}]: {case['claim'][:48]}")
        else:  # grounded, core
            if flagged:
                core_fp += 1
                miss_notes.append(f"CORE FALSE-POS [{tag}]: {case['claim'][:46]}")
            else:
                core_tn += 1

    overall_fp = core_fp + res_flagged
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    core_precision = tp / (tp + core_fp) if (tp + core_fp) else 1.0
    overall_precision = tp / (tp + overall_fp) if (tp + overall_fp) else 1.0

    print(f"\n=== grounding gold set: {len(gold['cases'])} cases ===")
    print(f"  recall:          {recall:.2f}  [tp={tp} fn={fn}]   target >=0.90")
    print(f"  CORE precision:  {core_precision:.2f}  [fp={core_fp} tn={core_tn}]  floor >=0.90")
    print(f"  overall prec:    {overall_precision:.2f}  [fp={overall_fp}]  tracked, not gated")
    print(f"  residual:        {res_flagged}/{res_total} flagged  reported, not gated (§7/§8)")
    print("  per-category recall (fabricated only):")
    for cat, (caught, total) in sorted(cat_recall.items()):
        print(f"    {cat:12} {caught}/{total}")
    for note in res_notes:
        print(f"  {note}")
    for note in miss_notes:
        print(f"  {note}")

    # GROSS-REGRESSION floors (live tier; the 0.9 targets are tracked above, not asserted).
    # Recall: robustly 1.00, floor well below. CORE precision: non-residual grounded cases ground
    # reliably (measured 0 FP over N=5), so the floor is STRICT — a core false positive is a real
    # regression. The residual (cross-language / low-overlap faithful groundings, spec §7/§8) is
    # reported above and deliberately NOT gated; it must never silently drag the core guard down.
    assert recall >= 0.70, f"recall {recall:.2f} below the gross-regression floor"
    assert core_precision >= 0.90, (
        f"CORE precision {core_precision:.2f} below floor — non-residual grounded claims are being "
        f"stripped (a real regression, not the accepted §7/§8 residual)"
    )
