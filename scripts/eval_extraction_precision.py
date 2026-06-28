"""Live eval (OUT of the unit gate) for evidence-span extraction PRECISION.

Guards ``SKILL_SYSTEM_PROMPT_EVIDENCE`` — the role-relevance-scoped prompt that drops JD
non-skill noise (the company's business, the job title, tier labels) while KEEPING the real
candidate skills. A unit test can't guard a prompt; this is its regression net. Non-deterministic
(LLM), so it reports per-case rather than hard-asserting. Needs vLLM at localhost:8000.

    .venv/bin/python scripts/eval_extraction_precision.py

Snippets are short, synthetic-but-representative of the real dogfood failure modes (a biotech
company blurb leaking "protein engineering"; a SOC JD grounding its own "N2/N3" title; a security
*firm* whose business IS the skills — the adversarial no-strip case).
"""

from __future__ import annotations

import asyncio

from job_applicator.config import LLMConfig
from job_applicator.embeddings.skill_extraction import LLMSkillExtractor

# (label, jd_text, must_DROP [noise], must_KEEP [real candidate skills])
CASES: list[tuple[str, str, list[str], list[str]]] = [
    (
        "biotech company-blurb noise",
        "HyperBio is a global leader in protein engineering and monoclonal antibodies. As IT "
        "Security Specialist you will manage IPS, VPN, content filtering, and anti-malware across "
        "the corporate network.",
        ["protein engineering", "monoclonal antibodies"],
        ["IPS", "VPN", "content filtering"],
    ),
    (
        "job-title + tier label",
        "SOC Analyst N2/N3. Responsibilities: security event monitoring, incident management, and "
        "log analysis using a SIEM.",
        ["SOC Analyst", "N2/N3"],
        ["SIEM", "incident", "monitoring"],
    ),
    (
        "security firm (company business IS security — must NOT strip the skills)",
        "Actoran is a cybersecurity services company. The analyst operates SIEM, SOAR, IDS/IPS, "
        "and EDR tooling and performs vulnerability management for clients.",
        [],
        ["SIEM", "SOAR", "IDS/IPS", "EDR"],
    ),
    (
        "clean security JD (no-regression)",
        "Security Analyst. Required: firewalls, DNS, network monitoring, malware detection, "
        "vulnerability assessments, and incident response.",
        [],
        ["firewalls", "DNS", "network monitoring", "incident response"],
    ),
]


def _has(skills: list[str], term: str) -> bool:
    t = term.lower()
    return any(t in s.lower() or s.lower() in t for s in skills)


async def main() -> None:
    ext = LLMSkillExtractor(LLMConfig(), grounding_mode="evidence_span")
    n_pass = 0
    for label, jd, drop, keep in CASES:
        skills = await ext.extract(jd, use_cache=False)
        leaked = [d for d in drop if _has(skills, d)]
        missing = [k for k in keep if not _has(skills, k)]
        ok = not leaked and not missing
        n_pass += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {label}")
        print(f"    extracted: {skills}")
        if leaked:
            print(f"    !! NOISE LEAKED: {leaked}")
        if missing:
            print(f"    !! REAL SKILLS MISSING: {missing}")
    print(f"\n{n_pass}/{len(CASES)} cases clean")


if __name__ == "__main__":
    asyncio.run(main())
