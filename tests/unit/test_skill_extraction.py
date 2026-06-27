"""Unit tests for LLMSkillExtractor."""

from __future__ import annotations

import pytest

from job_applicator.config import LLMConfig
from job_applicator.embeddings.skill_extraction import LLMSkillExtractor, _ExtractionResult


@pytest.fixture
def extractor(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> LLMSkillExtractor:
    """Create an LLMSkillExtractor that writes its cache under a temp directory."""
    inst = LLMSkillExtractor(LLMConfig(model="test"))
    monkeypatch.setattr(inst, "_cache_dir", tmp_path / "skill-extraction")
    inst._cache_dir.mkdir(parents=True, exist_ok=True)
    return inst


class TestSkillExtraction:
    async def test_extracts_python_from_description_with_mocked_llm(
        self, extractor: LLMSkillExtractor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_llm(description: str) -> _ExtractionResult:
            return _ExtractionResult(
                skills=["Python", "FastAPI", "PostgreSQL"],
                method="instructor",
                fallback=False,
            )

        monkeypatch.setattr(extractor, "_call_llm", fake_llm)

        result = await extractor.extract(
            "We are looking for a backend engineer with Python, FastAPI, and PostgreSQL."
        )
        assert set(result) == {"FastAPI", "PostgreSQL", "Python"}

    async def test_extract_raises_on_llm_failure_not_empty(
        self, extractor: LLMSkillExtractor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An LLM-call FAILURE must raise LLMError — never return [] (indistinguishable from a job
        that genuinely lists no skills, which would silently degrade the match downstream)."""
        from job_applicator.exceptions import LLMError

        async def boom(description: str) -> _ExtractionResult:
            raise ConnectionError("connection refused")

        monkeypatch.setattr(extractor, "_call_llm", boom)
        with pytest.raises(LLMError):
            await extractor.extract("Senior Python engineer with Django and PostgreSQL.")

    async def test_extract_returns_empty_on_successful_no_skills(
        self, extractor: LLMSkillExtractor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A SUCCESSFUL call that finds no skills legitimately returns [] — not every empty is a
        failure. This is the distinction the no-masking rule hinges on (failure→raise, empty→ok)."""

        async def none_found(description: str) -> _ExtractionResult:
            return _ExtractionResult(skills=[], method="instructor", fallback=False)

        monkeypatch.setattr(extractor, "_call_llm", none_found)
        result = await extractor.extract("We value teamwork and a positive attitude.")
        assert result == []

    async def test_unmapped_skill_grounded_by_token_match(
        self, extractor: LLMSkillExtractor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_llm(description: str) -> _ExtractionResult:
            return _ExtractionResult(
                skills=["Salesforce"],
                method="instructor",
                fallback=False,
            )

        monkeypatch.setattr(extractor, "_call_llm", fake_llm)

        result = await extractor.extract("Experience with Salesforce is required.")
        assert "Salesforce" in result

    async def test_multiword_skill_grounded_by_exact_phrase(
        self, extractor: LLMSkillExtractor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_llm(description: str) -> _ExtractionResult:
            return _ExtractionResult(
                skills=["Machine Learning"],
                method="instructor",
                fallback=False,
            )

        monkeypatch.setattr(extractor, "_call_llm", fake_llm)

        result = await extractor.extract("We apply machine learning to our products.")
        assert "Machine Learning" in result

    async def test_multiword_skill_not_grounded_as_substring(
        self, extractor: LLMSkillExtractor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_llm(description: str) -> _ExtractionResult:
            return _ExtractionResult(
                skills=["REST APIs"],
                method="instructor",
                fallback=False,
            )

        monkeypatch.setattr(extractor, "_call_llm", fake_llm)

        result = await extractor.extract("We expose REST APIsolutions only.")
        assert "REST APIs" not in result

    async def test_single_word_skill_accepted_when_no_compound(
        self, extractor: LLMSkillExtractor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_llm(description: str) -> _ExtractionResult:
            return _ExtractionResult(
                skills=["React"],
                method="instructor",
                fallback=False,
            )

        monkeypatch.setattr(extractor, "_call_llm", fake_llm)

        result = await extractor.extract("We use React.")
        assert "React" in result

    async def test_single_word_skill_rejected_when_lowercase_compound_follows(
        self, extractor: LLMSkillExtractor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_llm(description: str) -> _ExtractionResult:
            return _ExtractionResult(
                skills=["React"],
                method="instructor",
                fallback=False,
            )

        monkeypatch.setattr(extractor, "_call_llm", fake_llm)

        result = await extractor.extract("we need a react native engineer.")
        assert "React" not in result

    async def test_single_word_skill_accepted_when_prose_word_follows(
        self, extractor: LLMSkillExtractor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_llm(description: str) -> _ExtractionResult:
            return _ExtractionResult(
                skills=["React"],
                method="instructor",
                fallback=False,
            )

        monkeypatch.setattr(extractor, "_call_llm", fake_llm)

        result = await extractor.extract("We need React experience for this role.")
        assert "React" in result

    async def test_single_word_skill_grounded_before_ordinary_noun(
        self, extractor: LLMSkillExtractor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """F-B regression: a skill followed by an ordinary noun ("Kubernetes platform",
        "Python automation") is NOT a compound skill, so the bare skill stays grounded.
        The old heuristic synthesized a compound from any non-stopword continuation and
        dropped Kubernetes/Python even though they were literally in the description."""

        async def fake_llm(description: str) -> _ExtractionResult:
            return _ExtractionResult(
                skills=["Kubernetes", "Python"],
                method="instructor",
                fallback=False,
            )

        monkeypatch.setattr(extractor, "_call_llm", fake_llm)

        result = await extractor.extract(
            "Own our Kubernetes platform and developer tooling; day-to-day is Python automation."
        )
        assert "Kubernetes" in result
        assert "Python" in result

    async def test_version_like_suffix_does_not_reject_single_word(
        self, extractor: LLMSkillExtractor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_llm(description: str) -> _ExtractionResult:
            return _ExtractionResult(
                skills=["Python"],
                method="instructor",
                fallback=False,
            )

        monkeypatch.setattr(extractor, "_call_llm", fake_llm)

        result = await extractor.extract("We use Python 3.11 on the backend.")
        assert "Python" in result


class TestEvidenceSpanGrounding:
    """Arc 2 Phase 1 — evidence-span grounding (grounding_mode='evidence_span'), default-off."""

    @staticmethod
    def _ext(mode: str = "evidence_span") -> LLMSkillExtractor:
        return LLMSkillExtractor(LLMConfig(model="test"), grounding_mode=mode)

    def test_default_mode_is_keyword(self) -> None:
        assert LLMSkillExtractor(LLMConfig(model="test"))._grounding_mode == "keyword"

    def test_span_grounded_normalizes_case_whitespace_punctuation(self) -> None:
        ext = self._ext()
        text = "Responsibilities: IV insertion, ventilator management. BLS/ACLS required."
        assert ext._span_grounded("IV insertion", text)
        assert ext._span_grounded("iv   insertion", text)  # case + whitespace
        assert ext._span_grounded("ventilator management.", text)  # trailing punctuation
        assert ext._span_grounded("BLS/ACLS", text)  # internal punctuation
        assert not ext._span_grounded("blockchain", text)  # absent
        assert not ext._span_grounded("", text)  # empty span never grounds

    def test_verify_spans_keeps_grounded_drops_fabricated(self) -> None:
        ext = self._ext()
        text = "Build async pipelines in Python with FastAPI and PostgreSQL."
        pairs = [
            ("Python", "in Python"),
            ("FastAPI", "with FastAPI"),
            ("Kubernetes", "deploy on Kubernetes"),  # span not in text → fabricated
            ("PostgreSQL", "PostgreSQL"),
        ]
        assert ext._verify_spans(pairs, text) == ["Python", "FastAPI", "PostgreSQL"]

    def test_verify_spans_dedupes_by_name(self) -> None:
        ext = self._ext()
        text = "Python and more Python work."
        assert ext._verify_spans([("Python", "Python and"), ("python", "more Python")], text) == [
            "Python"
        ]

    def test_clean_skills_skips_keyword_grounding_when_already_grounded(self) -> None:
        """A span-verified skill that is NOT a literal substring (cross-domain canonical name)
        must survive in evidence-span mode — keyword grounding would wrongly drop it."""
        ext = self._ext()
        desc = "Registered Nurse: patient assessment and ventilator management."
        # already_grounded=True (span verified upstream) keeps it; keyword grounding drops it.
        assert len(ext._clean_skills(["Critical Care Nursing"], desc, already_grounded=True)) == 1
        assert ext._clean_skills(["Critical Care Nursing"], desc, already_grounded=False) == []

    def test_cache_key_includes_grounding_mode(self) -> None:
        """No cross-mode cache contamination: keyword and evidence_span key the same text apart."""
        desc = "Python and PostgreSQL."
        assert self._ext("keyword")._get_cache_key(desc) != self._ext(
            "evidence_span"
        )._get_cache_key(desc)

    def test_evidence_grounding_cross_domain_eval(self) -> None:
        """Deterministic eval scaffold: span verification keeps real cross-domain skills
        (software / nursing / finance) and drops a fabricated span — the property the live
        multi-domain A/B (next phase) measures against the real LLM."""
        ext = self._ext()
        cases = {
            "software": (
                "Build async services in Python with FastAPI on Kubernetes.",
                [
                    ("Python", "in Python"),
                    ("FastAPI", "with FastAPI"),
                    ("Kubernetes", "on Kubernetes"),
                    ("Rust", "rewrite in Rust"),  # fabricated
                ],
                {"Python", "FastAPI", "Kubernetes"},
            ),
            "nursing": (
                "ICU RN: patient assessment, IV insertion, ventilator management; BLS required.",
                [
                    ("Patient Assessment", "patient assessment"),
                    ("IV Insertion", "IV insertion"),
                    ("Ventilator Management", "ventilator management"),
                    ("BLS", "BLS required"),
                    ("Phlebotomy", "phlebotomy certification"),  # fabricated
                ],
                {"Patient Assessment", "IV Insertion", "Ventilator Management", "BLS"},
            ),
            "finance": (
                "Analyst: discounted cash flow models, variance analysis, forecasts in Excel; CFA.",
                [
                    ("Discounted Cash Flow", "discounted cash flow models"),
                    ("Variance Analysis", "variance analysis"),
                    ("Excel", "in Excel"),
                    ("CFA", "CFA"),
                    ("Bloomberg Terminal", "Bloomberg Terminal"),  # fabricated
                ],
                {"Discounted Cash Flow", "Variance Analysis", "Excel", "CFA"},
            ),
        }
        for domain, (text, pairs, expected) in cases.items():
            assert set(ext._verify_spans(pairs, text)) == expected, domain
