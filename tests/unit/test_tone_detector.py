"""Tests for job posting tone detection."""

from __future__ import annotations

from job_applicator.documents.tone_detector import ToneDetector


class TestToneDetector:
    def test_corporate_tone(self):
        detector = ToneDetector()
        profile = detector.detect(
            title="Senior IT Support Analyst",
            description=(
                "Enterprise environment with SLA compliance and stakeholder"
                " management. Governance and process improvement required."
            ),
            requirements=["ITIL", "ServiceNow", "Compliance"],
        )
        assert profile.primary == "corporate"
        assert "leveraged" in profile.power_words
        assert "stakeholder" in profile.emphasis

    def test_startup_tone(self):
        detector = ToneDetector()
        profile = detector.detect(
            title="Full Stack Developer",
            description=(
                "Fast-paced startup looking for a self-starter who can"
                " wear many hats. Agile environment, scrappy team."
            ),
            requirements=["React", "Node.js", "AWS"],
        )
        assert profile.primary == "startup"
        assert "built" in profile.power_words

    def test_technical_tone(self):
        detector = ToneDetector()
        profile = detector.detect(
            title="Backend Engineer",
            description=(
                "System design and architecture for microservices."
                " CI/CD pipeline, scalability, distributed systems."
            ),
            requirements=["Python", "Kubernetes", "PostgreSQL"],
        )
        assert profile.primary == "technical"
        assert "architected" in profile.power_words

    def test_creative_tone(self):
        detector = ToneDetector()
        profile = detector.detect(
            title="UX Designer",
            description=(
                "Brand storytelling and design thinking. User experience"
                " research, visual design, content strategy."
            ),
            requirements=["Figma", "User Research", "Prototyping"],
        )
        assert profile.primary == "creative"
        assert "designed" in profile.power_words

    def test_empty_description_defaults_corporate(self):
        detector = ToneDetector()
        profile = detector.detect(title="Manager", description="", requirements=[])
        assert profile.primary == "unknown"

    def test_word_boundary_avoids_substring_false_positives(self):
        """L-5: 'api' must not be counted inside 'therapist', nor 'roi' in 'android'."""
        detector = ToneDetector()
        profile = detector.detect(
            title="Wellness Coordinator",
            description="Support a therapist team building android wellness reminders.",
            requirements=[],
        )
        # Neither the technical keyword 'api' (therapist) nor 'roi' (android)
        # should register, so no technical/corporate tone signal is detected.
        assert profile.scores["technical"] == 0.0
        assert profile.scores["corporate"] == 0.0
        # The caring tone should be detected from 'wellness', 'therapist', 'support'
        assert profile.scores["caring"] > 0.0
        assert profile.primary == "caring"

    def test_word_boundary_still_matches_symbol_and_phrase_keywords(self):
        """L-5: boundary matching must still catch 'ci/cd' and multi-word phrases."""
        detector = ToneDetector()
        profile = detector.detect(
            title="Engineer",
            description="We value a ci/cd pipeline and strong system design.",
            requirements=[],
        )
        assert profile.scores["technical"] > 0.0

    def test_format_for_prompt(self):
        detector = ToneDetector()
        profile = detector.detect(
            title="Developer",
            description="Fast-paced agile startup environment.",
            requirements=[],
        )
        formatted = detector.format_for_prompt(profile)
        assert "TONE:" in formatted
        assert "Use these action verbs:" in formatted
