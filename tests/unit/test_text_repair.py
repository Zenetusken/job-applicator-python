"""Corruption-gated glued-text repair (scrapers/text_repair.py).

The invariants under test mirror the 2026-07-02 empirics: space-INSERTION only (never
fuse), clean documents byte-identical (legit camelCase + French elision survive), and on
a gated document the repair restores exactly the span-groundability the honesty layer
needs (the KQL / Microsoft Security (E5) recovery measured on the real corrupted JD).
"""

from job_applicator.embeddings.skill_extraction import LLMSkillExtractor
from job_applicator.scrapers.text_repair import (
    CORRUPTION_GATE,
    corruption_signatures,
    repair_glued_text,
)

# A controlled replica of the real corruption shape (misaligned rich-text spans):
# camel glue before a sentence, two mid-word newline splits, punct-capital glues.
CORRUPTED = (
    "Compétences requises.Vous devez posséder:\n"
    "Microsoft SentinelExpérience avancée avec Microsoft Security (E5) et Microsoft "
    "Senti\nnelKQL (Kusto Query Langua\nge)Création de règles analytiques"
)


class TestGate:
    def test_clean_english_untouched(self) -> None:
        text = "We use JavaScript, PowerShell and OneDrive. Requirements: TypeScript."
        assert repair_glued_text(text) is text  # byte-identical, same object

    def test_clean_french_untouched(self) -> None:
        text = (
            "Vous avez de l'expérience en sécurité.\n"
            "Analyste en gestion des identités et des accès.\n"
            "Une connaissance de l'anglais est requise."
        )
        assert repair_glued_text(text) is text

    def test_lowercase_list_items_untouched(self) -> None:
        # The #143 fabrication shape: single-word list items must never be fused.
        text = "Skills:\npython\ndocker\nkubernetes\nincident\nresponse"
        assert repair_glued_text(text) is text

    def test_below_gate_untouched_even_with_signatures(self) -> None:
        text = "un détail.Nous verrons"  # 1 signature < gate
        assert corruption_signatures(text) == 1
        assert repair_glued_text(text) is text

    def test_gate_boundary_fires_at_threshold(self) -> None:
        text = "détail.Nous et aussi\nfin)Début et enfin:Voilà donc"
        assert corruption_signatures(text) >= CORRUPTION_GATE
        assert repair_glued_text(text) != text

    def test_empty_and_none_like(self) -> None:
        assert repair_glued_text("") == ""


class TestRepair:
    def test_punct_glue_gets_space(self) -> None:
        repaired = repair_glued_text(CORRUPTED)
        assert "requises. Vous" in repaired
        assert "ge) Création" in repaired

    def test_camel_glue_gets_space(self) -> None:
        repaired = repair_glued_text(CORRUPTED)
        assert "Sentinel Expérience" in repaired
        assert "nel KQL" in repaired

    def test_midword_newline_left_broken_never_fused(self) -> None:
        # Fusing "Senti\nnel" would be indistinguishable from fusing two list items —
        # the repair must leave it broken (counted as a signature, not repaired).
        repaired = repair_glued_text(CORRUPTED)
        assert "Senti\nnel" in repaired
        assert "Sentinelnel" not in repaired.replace("\n", "")  # nothing got fused

    def test_repair_restores_span_grounding(self) -> None:
        # The honesty-layer tie-in: these evidence spans are VERBATIM in the corrupted
        # text but glued to a neighbor, so the word-boundary guard correctly refused
        # them; after repair they ground — the measured KQL / MS-Security recovery.
        span_e5 = "Expérience avancée avec Microsoft Security (E5)"
        span_kql = "KQL (Kusto Query Langua"
        grounded = LLMSkillExtractor._span_grounded
        assert not grounded(span_e5, CORRUPTED)
        assert not grounded(span_kql, CORRUPTED)
        repaired = repair_glued_text(CORRUPTED)
        assert grounded(span_e5, repaired)
        assert grounded(span_kql, repaired)
