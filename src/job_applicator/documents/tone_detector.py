"""Auto-detect job posting tone and provide vocabulary guidance."""

from __future__ import annotations

from dataclasses import dataclass, field

from job_applicator.utils.logging import get_logger
from job_applicator.utils.text import contains_word

logger = get_logger("documents.tone_detector")

TONE_KEYWORDS: dict[str, list[str]] = {
    "corporate": [
        "compliance",
        "governance",
        "stakeholder",
        "enterprise",
        "sla",
        "kpi",
        "process improvement",
        "audit",
        "regulatory",
        "cross-functional",
        "strategic",
        "initiative",
        "deliverable",
        "benchmark",
        "roi",
        "itil",
        "itsm",
        "change management",
        "risk management",
    ],
    "startup": [
        "fast-paced",
        "wear many hats",
        "agile",
        "scrappy",
        "self-starter",
        "equity",
        "early-stage",
        "series a",
        "series b",
        "founder",
        "greenfield",
        "0 to 1",
        "ownership",
        "autonomy",
        "rapid growth",
        "disrupt",
        "innovate",
        "pivot",
    ],
    "technical": [
        "architecture",
        "system design",
        "scalability",
        "ci/cd",
        "microservices",
        "distributed",
        "infrastructure",
        "devops",
        "sre",
        "latency",
        "throughput",
        "concurrency",
        "api",
        "sdk",
        "framework",
        "pipeline",
        "kubernetes",
        "docker",
        "terraform",
        "cloud-native",
    ],
    "creative": [
        "brand",
        "storytelling",
        "design thinking",
        "user experience",
        "visual",
        "content",
        "creative",
        "aesthetic",
        "prototype",
        "wireframe",
        "mockup",
        "user research",
        "persona",
        "journey map",
        "illustration",
        "typography",
        "color theory",
    ],
    "caring": [
        "compassionate",
        "empathy",
        "empathetic",
        "nurturing",
        "supportive",
        "patient care",
        "healthcare",
        "wellness",
        "therapist",
        "counseling",
        "advocacy",
        "human-centered",
        "client-centered",
        "service-oriented",
        "helping others",
        "caregiving",
        "rehabilitation",
        "mental health",
    ],
}

TONE_POWER_WORDS: dict[str, list[str]] = {
    "corporate": [
        "leveraged",
        "orchestrated",
        "facilitated",
        "spearheaded",
        "streamlined",
        "optimized",
        "administered",
        "coordinated",
    ],
    "startup": [
        "built",
        "launched",
        "scaled",
        "pivoted",
        "shipped",
        "hustled",
        "iterated",
        "bootstrapped",
    ],
    "technical": [
        "architected",
        "engineered",
        "implemented",
        "automated",
        "designed",
        "deployed",
        "migrated",
        "refactored",
    ],
    "creative": [
        "designed",
        "crafted",
        "envisioned",
        "curated",
        "conceptualized",
        "illustrated",
        "styled",
        "composed",
    ],
    "caring": [
        "supported",
        "advocated",
        "nurtured",
        "guided",
        "empathized",
        "listened",
        "comforted",
        "empowered",
    ],
}

TONE_EMPHASIS: dict[str, list[str]] = {
    "corporate": [
        "compliance",
        "process improvement",
        "stakeholder",
        "risk mitigation",
        "strategic planning",
    ],
    "startup": [
        "ownership",
        "rapid iteration",
        "cross-functional impact",
        "resourcefulness",
        "adaptability",
    ],
    "technical": [
        "system design",
        "scalability",
        "performance optimization",
        "code quality",
        "technical leadership",
    ],
    "creative": [
        "user empathy",
        "visual communication",
        "design process",
        "brand consistency",
        "creative problem-solving",
    ],
    "caring": [
        "patient care",
        "wellness advocacy",
        "empathetic support",
        "human-centered care",
        "service excellence",
        "compassionate communication",
    ],
}

TONE_AVOID: dict[str, list[str]] = {
    "corporate": ["casual language", "slang", "overly technical jargon"],
    "startup": ["corporate speak", "bureaucratic language", "overly formal"],
    "technical": ["buzzwords without substance", "vague claims", "fluff"],
    "creative": ["rigid corporate language", "purely technical focus", "dry tone"],
    "caring": ["clinical detachment", "impersonal language", "rigid process focus"],
}


@dataclass
class ToneProfile:
    """Detected tone profile for a job posting."""

    primary: str = "corporate"
    confidence: float = 0.0
    scores: dict[str, float] = field(default_factory=dict)
    power_words: list[str] = field(default_factory=list)
    emphasis: list[str] = field(default_factory=list)
    avoid: list[str] = field(default_factory=list)


class ToneDetector:
    """Detect job posting tone from title, description, and requirements."""

    def detect(
        self,
        title: str,
        description: str,
        requirements: list[str],
    ) -> ToneProfile:
        """Analyze job posting and return tone profile."""
        combined = f"{title} {description} {' '.join(requirements)}".lower()
        scores: dict[str, float] = {}

        for tone, keywords in TONE_KEYWORDS.items():
            count = sum(1 for kw in keywords if contains_word(combined, kw))
            scores[tone] = count / max(len(keywords), 1)

        if not any(scores.values()):
            primary = "unknown"
            confidence = 0.0
        else:
            primary = max(scores, key=scores.get)  # type: ignore[arg-type]
            total = sum(scores.values())
            confidence = scores[primary] / total if total > 0 else 0.0

        logger.debug("Detected tone: %s (confidence: %.1f%%)", primary, confidence * 100)

        return ToneProfile(
            primary=primary,
            confidence=confidence,
            scores=scores,
            power_words=TONE_POWER_WORDS.get(primary, []),
            emphasis=TONE_EMPHASIS.get(primary, []),
            avoid=TONE_AVOID.get(primary, []),
        )

    def format_for_prompt(self, profile: ToneProfile) -> str:
        """Format tone profile as actionable LLM directives."""
        if profile.primary == "unknown":
            return (
                "TONE: Match the job posting's natural tone. "
                "Use clear, direct language appropriate for the role."
            )
        lines = [
            f"TONE: {profile.primary.title()} (confidence: {profile.confidence:.0%})",
            f"- Use these action verbs: {', '.join(profile.power_words[:5])}",
            f"- Emphasize: {', '.join(profile.emphasis)}",
            f"- Avoid: {', '.join(profile.avoid)}",
            "- Mirror the job posting's vocabulary and sentence structure.",
        ]
        return "\n".join(lines)
