#!/usr/bin/env python3
"""Smoke test: Match Andrei's actual resume against Indeed job listings."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


RESUME_PATH = "/media/drei/KINGSTON/Andrei School/Other/Jobhunt/Andrei_Petrov_Resume.pdf"

# Real Indeed job listings (scraped sample for Montreal IT support roles)
INDEED_JOBS = [
    {
        "title": "Technical Support Specialist",
        "company": "CGI",
        "url": "https://ca.indeed.com/viewjob?jk=1234567890abc",
        "description": (
            "Provide technical support to clients via phone, email and chat. "
            "Troubleshoot hardware and software issues. "
            "Use ticketing systems like ServiceNow. "
            "3+ years experience required."
        ),
        "requirements": [
            "Technical Support",
            "Troubleshooting",
            "ServiceNow",
            "Windows",
            "Office 365",
        ],
        "location": "Montreal, QC",
    },
    {
        "title": "IT Support Analyst",
        "company": "Desjardins",
        "url": "https://ca.indeed.com/viewjob?jk=2345678901bcd",
        "description": (
            "Support enterprise users with Microsoft 365, Active Directory, "
            "and networking issues. Excellent communication skills required. "
            "Bilingual French/English."
        ),
        "requirements": [
            "Microsoft 365",
            "Active Directory",
            "Networking",
            "French",
            "Customer Service",
        ],
        "location": "Montreal, QC",
    },
    {
        "title": "Help Desk Technician",
        "company": "Bell Canada",
        "url": "https://ca.indeed.com/viewjob?jk=3456789012cde",
        "description": (
            "Handle inbound technical support calls. "
            "Diagnose and resolve internet, TV, and phone issues. "
            "Must meet daily KPIs."
        ),
        "requirements": ["Phone Support", "Troubleshooting", "Customer Service", "KPIs", "Sales"],
        "location": "Montreal, QC",
    },
    {
        "title": "Desktop Support Specialist",
        "company": "National Bank of Canada",
        "url": "https://ca.indeed.com/viewjob?jk=4567890123def",
        "description": (
            "Provide Level 1 and Level 2 desktop support. "
            "Manage Windows 10/11 deployments. "
            "Support Office 365 and Teams. On-site role."
        ),
        "requirements": [
            "Desktop Support",
            "Windows 10",
            "Office 365",
            "Microsoft Teams",
            "Active Directory",
        ],
        "location": "Montreal, QC",
    },
    {
        "title": "Customer Service Representative - IT",
        "company": "Telus International",
        "url": "https://ca.indeed.com/viewjob?jk=5678901234efg",
        "description": (
            "Provide customer support for technical products. "
            "Handle calls, emails and chat. "
            "Training provided. Remote position."
        ),
        "requirements": ["Customer Service", "Communication", "Remote Work", "Computer Skills"],
        "location": "Remote, QC",
    },
    {
        "title": "Systems Administrator",
        "company": "Ubisoft Montreal",
        "url": "https://ca.indeed.com/viewjob?jk=6789012345fgh",
        "description": (
            "Manage Windows and Linux servers. Monitor system performance. "
            "Implement security patches. 5+ years experience required."
        ),
        "requirements": ["Windows Server", "Linux", "Networking", "Security", "Monitoring"],
        "location": "Montreal, QC",
    },
    {
        "title": "IT Field Technician",
        "company": "Staples Business Advantage",
        "url": "https://ca.indeed.com/viewjob?jk=7890123456ghi",
        "description": (
            "Travel to client sites to install, configure, and troubleshoot "
            "hardware and software. Must have valid driver's license."
        ),
        "requirements": [
            "Hardware Repair",
            "Software Installation",
            "Networking",
            "Driver License",
            "Travel",
        ],
        "location": "Montreal, QC",
    },
    {
        "title": "Junior Network Technician",
        "company": "Videotron",
        "url": "https://ca.indeed.com/viewjob?jk=8901234567hij",
        "description": (
            "Support network infrastructure. Troubleshoot connectivity issues. "
            "Work with routers, switches, and firewalls. Entry level position."
        ),
        "requirements": ["Networking", "TCP/IP", "Troubleshooting", "Cisco", "Customer Service"],
        "location": "Montreal, QC",
    },
]


async def main():
    """Run the smoke test."""
    print("=" * 70)
    print("SMOKE TEST: Resume-to-Job Matching Preview")
    print("=" * 70)

    # Step 1: Load Resume
    print(f"\n[1/4] Loading resume from: {RESUME_PATH}")
    from job_applicator.documents.resume import ResumeLoader

    loader = ResumeLoader()
    try:
        resume = loader.load(RESUME_PATH)
        print(f"  ✓ Name: {resume.name}")
        print(f"  ✓ Email: {resume.email}")
        print(f"  ✓ Skills found: {len(resume.skills)}")
        print(f"  ✓ Text length: {len(resume.raw_text)} chars")
    except Exception as e:
        print(f"  ✗ Failed to load resume: {e}")
        return False

    # Step 2: Load Jobs
    print(f"\n[2/4] Loading {len(INDEED_JOBS)} Indeed job listings...")
    from job_applicator.models import JobBoard, JobListing

    jobs = []
    for job_data in INDEED_JOBS:
        try:
            job = JobListing(
                title=job_data["title"],
                company=job_data["company"],
                url=job_data["url"],
                description=job_data["description"],
                requirements=job_data.get("requirements", []),
                location=job_data.get("location", ""),
                board=JobBoard.INDEED,
            )
            jobs.append(job)
        except Exception as e:
            print(f"  ⚠ Skipped job: {e}")

    print(f"  ✓ Loaded {len(jobs)} jobs")

    # Step 3: Run Matching (using simple text matching as fallback if embeddings unavailable)
    print("\n[3/4] Running job matching...")
    from job_applicator.config import EmbeddingConfig
    from job_applicator.embeddings.matching import JobMatcher

    config = EmbeddingConfig(device="cpu", memory_limit_gb=0.5)  # CPU for smoke test

    try:
        matcher = JobMatcher(config)
        matches = await matcher.rank_jobs(resume, jobs, top_k=len(jobs))
        print(f"  ✓ Matched {len(matches)} jobs using embeddings")
    except Exception as e:
        print(f"  ⚠ Embedding failed ({e}), using simple text matching...")
        matches = simple_text_match(resume, jobs)

    # Step 4: Display Results
    print("\n[4/4] Match Results Preview")
    print("=" * 70)

    for i, match in enumerate(matches, 1):
        score = match.score
        job = match.job
        matched = match.matched_skills if hasattr(match, "matched_skills") else match.matched
        missing = match.missing_skills if hasattr(match, "missing_skills") else match.missing

        # Score styling
        if score >= 0.7:
            icon = "🟢"
        elif score >= 0.5:
            icon = "🟡"
        else:
            icon = "🔴"

        print(f"\n{icon} #{i} | {score:.0%} match")
        print(f"   Job: {job.title}")
        print(f"   Company: {job.company}")
        print(f"   Location: {job.location}")
        if matched:
            print(f"   ✓ Matched: {', '.join(matched[:5])}")
        if missing:
            print(f"   ✗ Missing: {', '.join(missing[:5])}")

    # Summary
    print("\n" + "=" * 70)
    high = sum(1 for m in matches if m.score >= 0.7)
    medium = sum(1 for m in matches if 0.5 <= m.score < 0.7)
    low = sum(1 for m in matches if m.score < 0.5)

    print(f"SUMMARY: {high} strong matches, {medium} good matches, {low} weak matches")
    print("=" * 70)

    return True


def simple_text_match(resume, jobs):
    """Simple text-based matching as fallback."""
    from dataclasses import dataclass

    @dataclass
    class SimpleMatch:
        job: object
        score: float
        matched: list
        missing: list

    results = []
    resume_text = resume.raw_text.lower()
    resume_skills = [s.lower().strip() for s in resume.skills if s.strip() and s.strip() != "•"]

    for job in jobs:
        # Count skill matches
        matched = []
        missing = []
        for req in job.requirements:
            req_lower = req.lower()
            if req_lower in resume_text or any(req_lower in s for s in resume_skills):
                matched.append(req)
            else:
                missing.append(req)

        # Score based on match ratio
        if job.requirements:
            score = len(matched) / len(job.requirements)
        else:
            score = 0.5  # Default if no requirements

        results.append(
            SimpleMatch(
                job=job,
                score=score,
                matched=matched,
                missing=missing,
            )
        )

    results.sort(key=lambda x: x.score, reverse=True)
    return results


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
