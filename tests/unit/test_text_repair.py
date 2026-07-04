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

# A controlled replica of the real corruption shape (misaligned rich-text spans): camel glue
# before a sentence, a mid-word newline split, and ≥3 letter-led `word)Word` punct-glues (the
# only gate signal).
CORRUPTED = (
    "Compétences requises.Vous devez maîtriser:\n"
    "Microsoft SentinelExpérience avancée avec Microsoft Security (E5) et Microsoft "
    "Senti\nnelKQL (Kusto Query Langua\nge)Création de règles.Les capacités de "
    "détection)Analyse en continu"
)


class TestGate:
    def test_clean_english_camelcase_untouched(self) -> None:
        # Legit camelCase product names must survive byte-identical (0 punct-glue signatures).
        text = "We use JavaScript, PowerShell and OneDrive. Requirements: TypeScript."
        assert corruption_signatures(text) == 0
        assert repair_glued_text(text) is text  # byte-identical, same object

    def test_clean_camelcase_with_lowercase_bullets_untouched(self) -> None:
        # THE review regression (finding [1]): a clean JD mixing lowercase-wrapped bullet lines
        # with legit camelCase tools must NOT trip the gate and must NOT be split. The old gate
        # counted every lowercase\nlowercase wrap as a signature, tripped here, and mangled
        # PowerShell→"Power Shell" — the exact skill-loss this feature exists to prevent.
        text = (
            "We are hiring a security analyst. Responsibilities:\n"
            "monitor alerts and triage incidents\n"
            "investigate threats using our tooling\n"
            "document findings clearly\n"
            "Requirements:\n"
            "experience with PowerShell and JavaScript\n"
            "familiarity with OneDrive and SharePoint"
        )
        assert corruption_signatures(text) == 0
        assert repair_glued_text(text) is text
        out = repair_glued_text(text)
        assert "PowerShell" in out and "JavaScript" in out and "SharePoint" in out

    def test_numbered_list_untouched(self) -> None:
        # `1.Overview` numbered lists are legit formatting, not corruption — the lead char of a
        # punct-glue is a LETTER, not a digit, precisely so these don't trip the gate.
        text = (
            "1.Overview of the role\n2.Details and scope\n3.Requirements you need\n"
            "We use JavaScript and PowerShell here."
        )
        assert corruption_signatures(text) == 0
        assert repair_glued_text(text) is text

    def test_clean_french_untouched(self) -> None:
        text = (
            "Vous avez de l'expérience en sécurité.\n"
            "Analyste en gestion des identités et des accès.\n"
            "Une connaissance de l'anglais est requise."
        )
        assert corruption_signatures(text) == 0
        assert repair_glued_text(text) is text

    def test_lowercase_list_items_untouched(self) -> None:
        # The #143 fabrication shape: single-word list items must never be fused (0 signatures,
        # gate holds — not merely "nothing to split").
        text = "Skills:\npython\ndocker\nkubernetes\nincident\nresponse"
        assert corruption_signatures(text) == 0
        assert repair_glued_text(text) is text

    def test_below_gate_untouched_even_with_signatures(self) -> None:
        text = "un détail.Nous verrons"  # 1 signature < gate
        assert corruption_signatures(text) == 1
        assert repair_glued_text(text) is text

    def test_gate_boundary_fires_at_threshold(self) -> None:
        text = "détail.Nous et fin)Début et enfin:Voilà donc"  # 3 letter-led punct-glues
        assert corruption_signatures(text) == CORRUPTION_GATE
        assert repair_glued_text(text) != text

    def test_empty_and_none_like(self) -> None:
        assert repair_glued_text("") == ""


class TestRepair:
    def test_corrupted_replica_trips_gate(self) -> None:
        # The replica must actually clear the gate on letter-led punct-glues alone.
        assert corruption_signatures(CORRUPTED) >= CORRUPTION_GATE

    def test_punct_glue_gets_space(self) -> None:
        repaired = repair_glued_text(CORRUPTED)
        assert "requises. Vous" in repaired
        assert "ge) Création" in repaired
        assert "détection) Analyse" in repaired

    def test_camel_glue_gets_space(self) -> None:
        repaired = repair_glued_text(CORRUPTED)
        assert "Sentinel Expérience" in repaired
        assert "nel KQL" in repaired

    def test_midword_newline_left_broken_never_fused(self) -> None:
        # Fusing "Senti\nnel" would be indistinguishable from fusing two list items —
        # the repair must leave it broken (not counted, not repaired).
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
