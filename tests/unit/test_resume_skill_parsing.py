"""Regression tests for résumé SKILLS-section parsing (QA finding F-A).

A comma-separated SKILLS list that *wraps across multiple lines* (ubiquitous in real
résumés) must parse into individual skills — not one comma-joined blob per line.
The old parser only comma-split when the section was a single line, so a wrapped list
became a couple of blob "skills" that matched nothing during skill-coverage scoring,
making ``match`` report the résumé's own skills as "missing".
"""

from __future__ import annotations

from job_applicator.documents.resume import ResumeLoader


def _parse_skills(text: str) -> list[str]:
    return ResumeLoader().parse_text(text).skills


def test_wrapped_multiline_comma_skills_parse_individually() -> None:
    text = (
        "ANDREI TESTER\n"
        "andrei@example.com\n\n"
        "SKILLS\n"
        "Python, asyncio, FastAPI, Pydantic, PostgreSQL, Redis, Docker,\n"
        "Kubernetes, AWS, pytest, SQLAlchemy, Git\n\n"
        "EXPERIENCE\n"
        "Engineer at Acme (2019-Present)\n"
        "- Built async services.\n"
    )
    skills = _parse_skills(text)
    # Tokens from BOTH wrapped lines are present as individual skills.
    for expected in ("Python", "asyncio", "FastAPI", "Kubernetes", "AWS", "Git"):
        assert expected in skills, f"{expected!r} missing from parsed skills: {skills}"
    # No unsplit blob: no parsed skill still contains a comma.
    assert not any("," in s for s in skills), f"unsplit blob present: {skills}"
    assert len(skills) >= 10, f"expected the full list, got {len(skills)}: {skills}"


def test_single_line_comma_skills_still_parse() -> None:
    """Backward-compat: the original single-line comma case is unchanged."""
    text = "NAME\nx@example.com\n\nSKILLS\nPython, Java, Go\n\nEXPERIENCE\nWork\n"
    skills = _parse_skills(text)
    assert {"Python", "Java", "Go"} <= set(skills)


def test_one_per_line_skills_still_parse() -> None:
    """Backward-compat: a one-skill-per-line section (no commas) is unchanged."""
    text = "NAME\nx@example.com\n\nSKILLS\nPython\nJava\nKubernetes\n\nEXPERIENCE\nWork\n"
    skills = _parse_skills(text)
    assert {"Python", "Java", "Kubernetes"} <= set(skills)
    assert not any("," in s for s in skills)


def test_two_column_tabgrid_skills_strip_row_label() -> None:
    """Two-column 'Category<tab>skill · skill' grids (the modern CV layout) parse to clean
    skills — the bold row label is dropped, not glued onto the first skill of each row.
    Regression: "Networking\tTCP/IP" was parsed as one skill, corrupting skill-coverage."""
    text = (
        "ANDREI TESTER\nandrei@example.com\n\n"
        "SKILLS\n"
        "Security ops\tSIEM · SOC monitoring · incident response\n"
        "Networking\tTCP/IP · subnetting · firewalls\n\n"
        "EXPERIENCE\nAnalyst at Acme (2019-Present)\n- Did things.\n"
    )
    skills = _parse_skills(text)
    assert "SIEM" in skills and "TCP/IP" in skills, skills
    assert not any("\t" in s for s in skills), f"tab/label leaked into a skill: {skills}"
    assert not any(s.startswith(("Security ops", "Networking")) for s in skills), skills


def test_comma_inside_parentheses_is_not_split() -> None:
    """A parenthetical with commas ("Linux (Fedora, CLI, Bash)") stays ONE skill — the paren-blind
    comma split produced stray-paren garbage tokens ("Linux (Fedora", "Bash)") that embed as junk
    vectors (audit AI-H4). Commas OUTSIDE parens still split normally."""
    skills = _parse_skills("Skills\nLinux (Fedora, CLI, Bash), SIEM, EDR")
    assert "Linux (Fedora, CLI, Bash)" in skills
    assert "SIEM" in skills and "EDR" in skills
    assert not any(s.count("(") != s.count(")") for s in skills)  # no unbalanced-paren tokens


def test_fixed_width_skills_grid_strips_labels_and_joins_continuations() -> None:
    """The v1 PDF's fixed-width skills grid is layout, not skill content."""
    text = (
        "ANDREI TESTER\nandrei@example.com\n\n"
        "TECHNICAL SKILLS\n"
        "Security ops            SIEM · SOC monitoring & triage · incident response · threat\n"
        "                        intelligence · OSINT · log & trace analysis\n"
        "Threat & detection MITRE ATT&CK · cyber kill chain · IDS/IPS · EDR · antivirus ·\n"
        "                   URL/content filtering · DLP · vulnerability management\n"
        "Cloud & systems         cloud security (IaaS/SaaS) · Linux (Fedora, CLI, Bash)\n\n"
        "EXPERIENCE\nAnalyst at Acme (2019-Present)\n- Did things.\n"
    )

    skills = _parse_skills(text)

    assert "SIEM" in skills
    assert "threat intelligence" in skills
    assert "MITRE ATT&CK" in skills
    assert "URL/content filtering" in skills
    assert "Linux (Fedora, CLI, Bash)" in skills
    assert "Security ops            SIEM" not in skills
    assert "Threat & detection MITRE ATT&CK" not in skills
