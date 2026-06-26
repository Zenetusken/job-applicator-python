#!/usr/bin/env python3
"""Detailed match report: Shows semantic + skill breakdown for each job."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

RESUME_PATH = "/media/drei/KINGSTON/Andrei School/Other/Jobhunt/Andrei_Petrov_Resume.pdf"

INDEED_JOBS = [
    {
        "title": "Technical Support Specialist",
        "company": "CGI",
        "url": "https://ca.indeed.com/viewjob?jk=1234567890abc",
        "description": (
            "Provide technical support to clients via phone, email and chat. "
            "Troubleshoot hardware and software issues. Use ticketing systems "
            "like ServiceNow. 3+ years experience required."
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
            "Handle inbound technical support calls. Diagnose and resolve "
            "internet, TV, and phone issues. Must meet daily KPIs."
        ),
        "requirements": [
            "Phone Support",
            "Troubleshooting",
            "Customer Service",
            "KPIs",
            "Sales",
        ],
        "location": "Montreal, QC",
    },
    {
        "title": "Desktop Support Specialist",
        "company": "National Bank of Canada",
        "url": "https://ca.indeed.com/viewjob?jk=4567890123def",
        "description": (
            "Provide Level 1 and Level 2 desktop support. Manage Windows "
            "10/11 deployments. Support Office 365 and Teams. On-site role."
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
            "Provide customer support for technical products. Handle calls, "
            "emails and chat. Training provided. Remote position."
        ),
        "requirements": [
            "Customer Service",
            "Communication",
            "Remote Work",
            "Computer Skills",
        ],
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
        "requirements": [
            "Windows Server",
            "Linux",
            "Networking",
            "Security",
            "Monitoring",
        ],
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
        "requirements": [
            "Networking",
            "TCP/IP",
            "Troubleshooting",
            "Cisco",
            "Customer Service",
        ],
        "location": "Montreal, QC",
    },
]


def get_score_color(score: float) -> str:
    if score >= 0.7:
        return "green"
    elif score >= 0.5:
        return "yellow"
    return "red"


async def main():
    console = Console()

    console.print(Panel.fit("[bold]Detailed Job Match Report[/bold]", style="blue"))

    # Load resume
    console.print("\n[bold]Loading resume...[/bold]")
    from job_applicator.documents.resume import ResumeLoader

    loader = ResumeLoader()
    try:
        resume = loader.load(RESUME_PATH)
        console.print(f"  Name: {resume.name}")
        skills_preview = ", ".join(resume.skills[:8])
        if len(resume.skills) > 8:
            skills_preview += "..."
        console.print(f"  Skills: {skills_preview}")
    except Exception as e:
        console.print(f"  [red]Failed: {e}[/red]")
        return False

    # Load jobs
    from job_applicator.models import JobBoard, JobListing

    jobs = []
    for j in INDEED_JOBS:
        jobs.append(
            JobListing(
                title=j["title"],
                company=j["company"],
                url=j["url"],
                description=j["description"],
                requirements=j.get("requirements", []),
                location=j.get("location", ""),
                board=JobBoard.INDEED,
            )
        )

    # Run matching
    console.print("\n[bold]Running embedding match...[/bold]")
    from job_applicator.config import EmbeddingConfig
    from job_applicator.embeddings.matching import JobMatcher

    config = EmbeddingConfig(device="cpu", memory_limit_gb=0.5)
    matcher = JobMatcher(config)

    # Get detailed results
    resume_emb = matcher.compute_resume_embedding(resume)

    results = []
    for job in jobs:
        job_emb = matcher.compute_job_embedding(job)
        semantic_score = matcher._service.similarity(resume_emb, job_emb)
        matched, missing = await matcher._match_skills(
            resume.skills, job.requirements, resume.raw_text
        )
        skill_score = matcher._compute_skill_score(matched, missing)
        combined = (0.6 * semantic_score) + (0.4 * skill_score)

        # Per-skill similarity
        skill_details = []
        threshold = 0.75
        if job.requirements:
            valid_skills = [s for s in resume.skills if len(s.strip()) > 2 and s.strip() != "•"]
            if valid_skills:
                skill_embs = matcher._service.embed_batch(valid_skills)
                req_embs = matcher._service.embed_batch(job.requirements)
                for i, req in enumerate(job.requirements):
                    best_sim = 0.0
                    best_skill = ""
                    for j_idx, skill in enumerate(valid_skills):
                        sim = matcher._service.similarity(req_embs[i], skill_embs[j_idx])
                        if sim > best_sim:
                            best_sim = sim
                            best_skill = skill
                    skill_details.append(
                        {
                            "requirement": req,
                            "best_match": best_skill,
                            "similarity": best_sim,
                            "matched": best_sim >= threshold,
                        }
                    )

        results.append(
            {
                "job": job,
                "semantic": semantic_score,
                "skill_score": skill_score,
                "combined": combined,
                "matched": matched,
                "missing": missing,
                "skill_details": skill_details,
            }
        )

    results.sort(key=lambda x: x["combined"], reverse=True)

    # Summary table
    console.print("\n")
    summary_table = Table(title="Match Summary", show_lines=True)
    summary_table.add_column("#", style="dim", width=3)
    summary_table.add_column("Job Title", style="bold")
    summary_table.add_column("Company")
    summary_table.add_column("Combined", justify="center")
    summary_table.add_column("Semantic", justify="center")
    summary_table.add_column("Skill", justify="center")
    summary_table.add_column("Matched", style="green")
    summary_table.add_column("Missing", style="red")

    for i, r in enumerate(results, 1):
        color = get_score_color(r["combined"])
        summary_table.add_row(
            str(i),
            r["job"].title,
            r["job"].company,
            f"[{color}]{r['combined']:.0%}[/{color}]",
            f"{r['semantic']:.0%}",
            f"{r['skill_score']:.0%}",
            ", ".join(r["matched"][:3]) or "—",
            ", ".join(r["missing"][:3]) or "—",
        )

    console.print(summary_table)

    # Per-job detail panels
    console.print("\n[bold]Per-Job Skill Breakdown[/bold]\n")

    for i, r in enumerate(results, 1):
        job = r["job"]
        color = get_score_color(r["combined"])

        detail_lines = []
        detail_lines.append(f"[bold]{job.title}[/bold] at {job.company}")
        detail_lines.append(f"Location: {job.location}")
        score_line = (
            f"Combined: [{color}]{r['combined']:.0%}[/{color}]"
            f"  |  Semantic: {r['semantic']:.0%}"
            f"  |  Skill Coverage: {r['skill_score']:.0%}"
        )
        detail_lines.append(score_line)
        detail_lines.append("")

        if r["skill_details"]:
            detail_lines.append("[bold]Requirement Matching:[/bold]")
            for sd in r["skill_details"]:
                sim_color = get_score_color(sd["similarity"])
                if sd["matched"]:
                    status = "[green]✓[/green]"
                else:
                    status = "[red]✗[/red]"
                req = sd["requirement"]
                match = sd["best_match"]
                sim = sd["similarity"]
                detail_lines.append(
                    f"  {status} {req:25s} → {match:25s} [{sim_color}]{sim:.0%}[/{sim_color}]"
                )
        else:
            detail_lines.append("[dim]No requirements specified[/dim]")

        if r["missing"]:
            detail_lines.append("")
            detail_lines.append("[bold red]Skills to develop:[/bold red]")
            for skill in r["missing"]:
                detail_lines.append(f"  • {skill}")

        title = f"#{i} ({r['combined']:.0%})"
        console.print(Panel("\n".join(detail_lines), title=title, border_style=color))

    # Recommendations
    console.print("\n")
    rec_table = Table(title="Recommendations", show_lines=True)
    rec_table.add_column("Category", style="bold")
    rec_table.add_column("Details")

    top = results[0] if results else None
    if top:
        best_label = f"{top['job'].title} at {top['job'].company} ({top['combined']:.0%})"
        rec_table.add_row("Best Match", best_label)

    all_missing = []
    for r in results[:3]:
        all_missing.extend(r["missing"])
    from collections import Counter

    common_missing = Counter(all_missing).most_common(5)
    if common_missing:
        skills_text = "\n".join(f"• {skill} ({count} jobs)" for skill, count in common_missing)
        rec_table.add_row("Most In-Demand Missing Skills", skills_text)

    strong = [r for r in results if r["combined"] >= 0.7]
    if strong:
        rec_table.add_row("Strong Matches (≥70%)", f"{len(strong)} jobs")
    else:
        rec_table.add_row("Strong Matches", "[dim]None above 70%[/dim]")

    console.print(rec_table)

    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
