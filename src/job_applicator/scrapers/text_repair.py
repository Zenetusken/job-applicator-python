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

Both repairs are **gated on a per-document corruption-signature count**: clean documents
(including legit camelCase like ``JavaScript``/``PowerShell`` and French elision
``l'expérience``) are returned byte-identical. Measured on the 39-JD funnel: the gate
fires on 2/39 (the corrupted posting at 27 signatures and one mildly-glued one at 5);
re-extraction on the repaired text recovered the full SOC skill set (KQL, Sentinel,
Defender, Purview, Entra ID, SIEM/SOAR, Threat Intelligence) with zero fabrications.
"""

from __future__ import annotations

import re

from job_applicator.utils.logging import get_logger

logger = get_logger("scrapers.text_repair")

# A lowercase letter/digit + closing punctuation glued straight onto a capital:
# "ge)Création", "détail.Nous", "client.Participation". Typographically deterministic —
# no language writes these without a space — so safe to split wherever seen (still gated).
_PUNCT_GLUE = re.compile(r"([a-zà-öø-ÿ0-9][)\].,;:!?])([A-ZÀ-ÖØ-Þ])")

# A lowercase letter glued straight onto a capital: "SentinelKQL" → "Sentinel KQL",
# "détectionExpérience". AMBIGUOUS in general (JavaScript, PowerShell, OneDrive are legit
# camelCase), which is exactly why this only ever runs on documents already proven
# corrupted by the signature gate — on those, recovering mashed content outweighs
# perturbing the odd product name.
_CAMEL_GLUE = re.compile(r"([a-zà-öø-ÿ])([A-ZÀ-ÖØ-Þ])")

# A newline splitting a word in half ("Senti\nnel", "Langua\nge") — counted as a
# corruption signature (a block boundary never lands mid-word in sane markup) but NEVER
# repaired: fusing across a newline is the #143 fabrication risk.
_MIDWORD_NL = re.compile(r"[a-zà-öø-ÿ]\n[a-zà-öø-ÿ]")

#: Signature count at/above which a document is treated as corrupted. Funnel-measured:
#: the corrupted posting scored 27, a mildly-glued one 5, every clean JD ≤1.
CORRUPTION_GATE = 3


def corruption_signatures(text: str) -> int:
    """Count glued-word signatures in ``text`` (punct-glue + mid-word newline splits)."""
    return len(_PUNCT_GLUE.findall(text)) + len(_MIDWORD_NL.findall(text))


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
