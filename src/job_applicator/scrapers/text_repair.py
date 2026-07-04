"""Corruption-gated repair for glued words in scraped job descriptions.

Some postings arrive with words mashed together — ``Microsoft Senti\\nnelKQL``,
``ge)Création``, ``détail.Nous`` — because the author's rich-text spans are misaligned in
the board's OWN markup (verified live 2026-07-02: the LinkedIn DOM carries a text node
reading ``nelKQL (Kusto Query Langua`` inside ``<strong><li>``, identically in the search
panel and the full job view — upstream data, not an extraction bug; ``inner_text()`` is
the faithful reading). The mash costs real skills: the evidence-span verifier correctly
refuses spans glued to a neighbor, so ``KQL (Kusto Query Language)`` and
``Microsoft Security (E5)`` were dropped from the one corrupted posting in the funnel —
which happened to be the best-fit role, left sitting exactly on the review cutoff.

The repair is **space INSERTION only** — it splits glued tokens and can never fuse two
tokens into a fabricated compound (the failure that killed PR #143: whitespace-collapse
fused list items across bare newlines into skills that were never claimed). Mid-word
newline splits (``Senti\\nnel``) are deliberately left broken: joining them is
indistinguishable from fusing two legitimate single-word list items.

Both repairs are **gated on a per-document count of one narrow signature** — a letter-led
``word)Word`` punct-glue (see ``_PUNCT_GLUE``), the only typographically-unambiguous
missing-space marker. Clean documents (legit camelCase like ``JavaScript``/``PowerShell``,
French elision ``l'expérience``, lowercase-wrapped bullet lists, ``1.Overview`` numbered
lists) score 0 and are returned byte-identical. Measured on the 39-JD funnel: the gate
fires on 2/39 (the corrupted posting at 7 letter-led punct-glues, a mildly-glued one at 5,
every clean JD ≤1); re-extraction on the repaired text recovered the full SOC skill set
(KQL, Sentinel, Defender, Purview, Entra ID, SIEM/SOAR, Threat Intelligence) with zero
fabrications.
"""

from __future__ import annotations

import re

from job_applicator.utils.logging import get_logger

logger = get_logger("scrapers.text_repair")

# A lowercase letter/digit + closing punctuation glued straight onto a capital:
# "ge)Création", "détail.Nous", "client.Participation". The lead char is a LETTER, not a
# digit: `1.Overview` / `2.Details` numbered lists are legit formatting, not corruption, and a
# digit-led class would trip the gate on them (then split any camelCase in the JD). The real
# mash is letter-led — measured, the corrupted posting still scores 7 such glues after dropping
# the digit, well past the gate. This is the ONLY corruption signature: typographically a
# letter+closing-punct glued onto a capital with no space is unambiguous (no language writes it).
_PUNCT_GLUE = re.compile(r"([a-zà-öø-ÿ][)\].,;:!?])([A-ZÀ-ÖØ-Þ])")

# A lowercase letter glued straight onto a capital: "SentinelKQL" → "Sentinel KQL",
# "détectionExpérience". AMBIGUOUS in general (JavaScript, PowerShell, OneDrive are legit
# camelCase), which is exactly why this is NOT a gate signal and only ever RUNS on documents
# already proven corrupted by the punct-glue gate — on those, recovering mashed content
# outweighs perturbing the odd product name. (An earlier build also counted mid-word newline
# splits `[a-z]\n[a-z]` as a signature; that matched ordinary lowercase line-wraps, tripped the
# gate on clean JDs, and camel-split their `JavaScript`/`PowerShell` — the exact skill-loss this
# feature prevents. Removed: the punct-glue is the one clean, unambiguous signal.)
_CAMEL_GLUE = re.compile(r"([a-zà-öø-ÿ])([A-ZÀ-ÖØ-Þ])")

#: Signature count at/above which a document is treated as corrupted. Funnel-measured: the two
#: corrupted postings score 7 and 5 letter-led punct-glues, every clean JD ≤1.
CORRUPTION_GATE = 3


def corruption_signatures(text: str) -> int:
    """Count corruption signatures — letter-led punct-glue (`word)Word`) runs, the one
    unambiguous missing-space marker. Mid-word newline splits (`Senti\\nnel`) are NOT counted:
    they're indistinguishable from ordinary lowercase line-wraps, which would false-trip the
    gate and mangle clean camelCase."""
    return len(_PUNCT_GLUE.findall(text))


def repair_glued_text(text: str, *, gate: int = CORRUPTION_GATE) -> str:
    """Insert spaces at glue boundaries IF the document is corruption-gated.

    Returns ``text`` unchanged (same object) when the signature count is below ``gate`` —
    clean documents are never touched, so legit camelCase and French elision survive.
    On a gated document, applies punct-glue and camel-glue space insertion (split-only,
    never fuse) and logs the repair so a quality-degraded posting is visible in the run.
    """
    if not text:
        return text
    n = corruption_signatures(text)
    if n < gate:
        return text
    repaired = _PUNCT_GLUE.sub(r"\1 \2", text)
    repaired = _CAMEL_GLUE.sub(r"\1 \2", repaired)
    logger.info(
        "Glued-text repair applied (%d corruption signatures; %d -> %d chars) — "
        "the posting's own markup mashes words; mid-word line breaks left as-is",
        n,
        len(text),
        len(repaired),
    )
    return repaired
