"""Skill-name normalization and hard-negative filtering.

Job postings and resumes use inconsistent names for the same technology
("Python 3", "python3", "Python Programming"). Normalizing before matching
reduces false negatives and makes hallucination validation more stable.

The hard-negative list catches generic traits that are not concrete skills
("team player", "detail-oriented", "communication") so they do not pollute
skill coverage scores or tailored skill sections.
"""

from __future__ import annotations

from typing import Final

# Canonical spellings for common technology aliases. Keys are lower-cased,
# punctuation-stripped forms; values are the canonical names to use.
NORMALIZATION_MAP: Final[dict[str, str]] = {
    "python 3": "Python",
    "python3": "Python",
    "python programming": "Python",
    "python scripting": "Python",
    "py": "Python",
    "javascript": "JavaScript",
    "js": "JavaScript",
    "typescript": "TypeScript",
    "ts": "TypeScript",
    "react js": "React",
    "reactjs": "React",
    "react native": "React Native",
    "node js": "Node.js",
    "nodejs": "Node.js",
    "vue js": "Vue.js",
    "vuejs": "Vue.js",
    "angular js": "Angular",
    "angularjs": "Angular",
    "amazon web services": "AWS",
    "amazon aws": "AWS",
    "aws lambda": "AWS Lambda",
    "google cloud platform": "GCP",
    "google cloud": "GCP",
    "azure cloud": "Azure",
    "ms azure": "Azure",
    "postgresql": "PostgreSQL",
    "postgres": "PostgreSQL",
    "mongodb": "MongoDB",
    "mongo": "MongoDB",
    "mysql": "MySQL",
    "redis db": "Redis",
    "docker container": "Docker",
    "kubernetes": "Kubernetes",
    "k8s": "Kubernetes",
    "terraform": "Terraform",
    "ansible": "Ansible",
    "ci cd": "CI/CD",
    "cicd": "CI/CD",
    "github actions": "GitHub Actions",
    "gitlab ci": "GitLab CI",
    "machine learning": "Machine Learning",
    "deep learning": "Deep Learning",
    "natural language processing": "NLP",
    "nlp": "NLP",
    "rest api": "REST APIs",
    "restful api": "REST APIs",
    "restful apis": "REST APIs",
    "graphql api": "GraphQL",
    "fast api": "FastAPI",
    "django framework": "Django",
    "flask framework": "Flask",
    "spring boot": "Spring Boot",
    "dotnet": ".NET",
    "dot net": ".NET",
    "c sharp": "C#",
    "csharp": "C#",
    "c plus plus": "C++",
    "cpp": "C++",
}

# Generic traits, soft skills, and noise terms that should never be treated as
# technical skills. Matching/validation uses word-boundary checks.
HARD_NEGATIVE_SKILLS: Final[frozenset[str]] = frozenset(
    {
        "team player",
        "teamwork",
        "communication",
        "communication skills",
        "detail oriented",
        "detail-oriented",
        "problem solving",
        "problem-solving",
        "critical thinking",
        "time management",
        "self starter",
        "self-starter",
        "fast paced",
        "fast-paced",
        "multitasking",
        "multi-tasking",
        "leadership",
        "management skills",
        "interpersonal skills",
        "organized",
        "adaptable",
        "flexible",
        "motivated",
        "passionate",
        "enthusiastic",
        "hardworking",
        "hard working",
        "dedicated",
        "reliable",
        "punctual",
        "remote work",
        "remote",
        "onsite",
        "on-site",
        "hybrid",
        "full time",
        "full-time",
        "part time",
        "part-time",
        "contract",
        "internship",
        "entry level",
        "entry-level",
        "senior",
        "junior",
        "mid level",
        "mid-level",
    }
)


def normalize_skill(skill: str) -> str:
    """Return a canonical form of a skill name.

    The mapping is intentionally conservative: only well-known aliases are
    rewritten. Unknown skills are returned cleaned but unchanged so that
    embedding-based matching can still handle novel or domain-specific terms.

    Examples:
        >>> normalize_skill("Python 3")
        'Python'
        >>> normalize_skill("nodeJS")
        'Node.js'
        >>> normalize_skill("Terraform")
        'Terraform'
    """
    if not skill:
        return skill

    cleaned = skill.strip()
    key = _canonical_key(cleaned)
    return NORMALIZATION_MAP.get(key, cleaned)


def _canonical_key(skill: str) -> str:
    """Lower-case and strip punctuation to create a lookup key."""
    lowered = skill.lower().strip()
    # Normalize separators to spaces.
    normalized = lowered.replace("+", " plus ").replace("#", " sharp ")
    normalized = normalized.replace("/", " ").replace("-", " ").replace("_", " ")
    normalized = " ".join(normalized.split())
    return normalized


def is_hard_negative(skill: str) -> bool:
    """True if ``skill`` is a generic trait or noise that should not be treated as a skill."""
    s = skill.strip()
    # Drop empty or PURE-punctuation noise (a bullet "•", a stray "-"/"|") — but KEEP short REAL
    # skills (C#, Go, R, AI, ML). The old `len <= 2` rule silently dropped those on BOTH the résumé
    # and requirement sides, nullifying the extractor's deliberate short-skill relaxation and making
    # matching.py's "so short skills … aren't dropped" comment false.
    if not s or not any(c.isalnum() for c in s):
        return True
    return s.lower() in HARD_NEGATIVE_SKILLS
