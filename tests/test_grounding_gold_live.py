"""Live gold-set measurement of the GroundingVerifier (spec §3 #3, §9 gate 3).

REPORTED, not a fast-gate unit test — a nondeterministic live LLM call (needs vLLM at :8000), so
it lives at the tests/ root and is auto-marked ``live``. It runs the verifier against the seed gold
set, prints precision + recall (both directions, per category), and asserts only a conservative
gross-regression floor on the live tier. The 0.9 targets are tracked over time, and the credential
row is reported separately (the French-credential leniency is a known, non-gating residual, §7).
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

    tp = fp = fn = tn = 0
    cat_recall: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])  # [caught, total]
    notes: list[str] = []
    for case in gold["cases"]:
        report = await verifier.verify(case["claim"], resume)
        flagged = bool(report.unsupported) or bool(report.coverage_gaps)
        fabricated = case["label"] == "fabricated"
        if fabricated and flagged:
            tp += 1
        elif fabricated and not flagged:
            fn += 1
            notes.append(f"MISS [{case['lang']}/{case['category']}]: {case['claim'][:50]}")
        elif not fabricated and flagged:
            fp += 1
            notes.append(f"FALSE-POS [{case['lang']}/{case['category']}]: {case['claim'][:50]}")
        else:
            tn += 1
        if fabricated:
            cat_recall[case["category"]][0] += int(flagged)
            cat_recall[case["category"]][1] += 1

    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    print(f"\n=== grounding gold set: {len(gold['cases'])} cases ===")
    print(f"  recall (fabrications caught):   {recall:.2f}  [tp={tp} fn={fn}]   target >=0.90")
    print(f"  precision (real claims kept):   {precision:.2f}  [fp={fp} tn={tn}]   target >=0.90")
    print("  per-category recall (fabricated only):")
    for cat, (caught, total) in sorted(cat_recall.items()):
        print(f"    {cat:12} {caught}/{total}")
    for note in notes:
        print(f"  {note}")

    # Live-tier gross-regression floor only — NOT the 0.9 target (tracked over time), and the
    # credential leniency is a reported residual, not a hard gate (§7).
    assert recall >= 0.70, f"recall {recall:.2f} below the gross-regression floor"
    assert precision >= 0.85, (
        f"precision {precision:.2f} below floor — real claims are being stripped"
    )
