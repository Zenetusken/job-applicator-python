#!/usr/bin/env python3
"""Show what the style analyzer extracts from sample documents."""

import asyncio
from pathlib import Path

# Sample 1: Formal Technical Resume
FORMAL_RESUME = """Sarah Chen
Senior Backend Engineer | sarah.chen@email.com | LinkedIn: linkedin.com/in/sarahchen

PROFESSIONAL SUMMARY
Results-driven backend engineer with 10+ years of experience designing and implementing
high-throughput distributed systems. Proven track record of leading engineering teams
and delivering mission-critical infrastructure serving millions of users.

TECHNICAL SKILLS
Languages: Python, Go, Java, Rust
Infrastructure: Kubernetes, AWS, GCP, Terraform
Databases: PostgreSQL, Redis, MongoDB, DynamoDB
Tools: Docker, CI/CD, Prometheus, Grafana

PROFESSIONAL EXPERIENCE

Senior Backend Engineer | MegaCorp Inc. | 2019-Present
• Architected event-driven microservices handling 100M+ daily transactions
• Led migration from monolith to microservices, reducing deployment time by 80%
• Mentored team of 8 engineers, establishing code review best practices

Backend Engineer | ScaleUp Technologies | 2015-2019
• Designed real-time data pipeline processing 1TB+ data daily
• Implemented caching strategies that improved API response times by 60%
• Contributed to open-source projects with 5000+ GitHub stars

EDUCATION
M.S. Computer Science | Stanford University | 2015
B.S. Computer Science | UC Berkeley | 2013
"""

# Sample 2: Casual/Creative Cover Letter
CASUAL_LETTER = """Hey there!

I was scrolling through your job posting and honestly got excited - this sounds like exactly the kind of challenge I've been looking for.

Here's the deal: I've spent the last 5 years building backend systems, and I'm pretty good at it. Not to brag, but my last API handled 10 million requests per day without breaking a sweat. The secret? Clean code, smart caching, and way too much coffee.

What really caught my eye about TechStartup is your mission to make data accessible. I've been preaching about data democratization for years, so it's cool to see a company actually doing it.

I'm not going to lie - I'm the kind of developer who gets way too attached to my code. But that's why my systems work so well. I treat every function like it's going to be there for the next decade.

Let's chat? I promise I'm way more interesting in person than I am on paper.

Cheers,
Alex Rivera
"""

# Sample 3: Academic/Formal Cover Letter
ACADEMIC_LETTER = """Dr. James Morrison
Department of Computer Science
University of Cambridge

Dear Professor Williams,

I am writing to express my interest in the Research Scientist position at DeepMind,
as advertised on your careers page. Having followed your groundbreaking work on
reinforcement learning with great admiration, I believe my research background
aligns well with your team's objectives.

My doctoral research at Cambridge focused on developing novel algorithms for
multi-agent reinforcement learning, resulting in three publications at NeurIPS
and ICML. During my postdoctoral work, I led a team that achieved state-of-the-art
results on several benchmark tasks, demonstrating the practical applicability of
theoretical advances in reinforcement learning.

I am particularly drawn to DeepMind's commitment to building safe and beneficial
AI systems. My work on alignment and interpretability directly addresses these
concerns, and I am eager to contribute to research that has both scientific
significance and real-world impact.

I would welcome the opportunity to discuss how my expertise in reinforcement learning
and commitment to rigorous research methodology could contribute to DeepMind's mission.

Sincerely,
Dr. James Morrison
Ph.D., Computer Science
University of Cambridge
"""


async def analyze_sample(name: str, text: str):
    """Analyze a sample document and show the extracted style."""
    print(f"\n{'=' * 70}")
    print(f"SAMPLE: {name}")
    print(f"{'=' * 70}")
    print(f"\nInput text ({len(text)} chars):")
    print("-" * 40)
    print(text[:300] + "..." if len(text) > 300 else text)

    from job_applicator.config import LLMConfig
    from job_applicator.documents.style_analyzer import StyleAnalyzer

    config = LLMConfig()
    analyzer = StyleAnalyzer(config)

    print(f"\nAnalyzing with {config.model}...")

    try:
        style = await analyzer.analyze(text)

        print(f"\n{'─' * 40}")
        print("EXTRACTED STYLE GUIDE:")
        print(f"{'─' * 40}")
        print(f"""
Tone: {style.tone}

Sentence Structure: {style.sentence_structure}

Vocabulary Level: {style.vocabulary_level}

Paragraph Style: {style.paragraph_style}

Key Phrases (characteristic of this style):
  • {chr(10) + '  • '.join(style.key_phrases) if style.key_phrases else '(none)'}

Phrases to AVOID (not used in this style):
  • {chr(10) + '  • '.join(style.avoid_phrases) if style.avoid_phrases else '(none)'}

Formatting Notes: {style.formatting_notes}

Sample Paragraph (representative of style):
  "{style.sample_paragraph[:200]}..."
""")
        return style

    except Exception as e:
        print(f"\nError: {e}")
        return None


async def main():
    """Show style analysis for multiple sample documents."""
    print("=" * 70)
    print("STYLE ANALYZER DEMO")
    print("=" * 70)
    print("\nThis shows what writing patterns the analyzer extracts.")

    samples = [
        ("Formal Technical Resume", FORMAL_RESUME),
        ("Casual/Creative Cover Letter", CASUAL_LETTER),
        ("Academic/Formal Cover Letter", ACADEMIC_LETTER),
    ]

    results = []
    for name, text in samples:
        style = await analyze_sample(name, text)
        results.append((name, style))

    # Summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\n{'Document Type':<30} {'Tone':<25} {'Vocabulary'}")
    print("-" * 70)
    for name, style in results:
        if style:
            print(f"{name:<30} {style.tone:<25} {style.vocabulary_level}")


if __name__ == "__main__":
    asyncio.run(main())
