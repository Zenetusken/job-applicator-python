"""Tests for hallucination guard functions in resume_tailor.py.

Covers: _validate_skills, _strip_hallucinated_tools,
_strip_hallucinated_education, _extract_education_entries, parse_sections.
"""

from __future__ import annotations

import pytest

from job_applicator.config import LLMConfig
from job_applicator.documents.resume_tailor import ResumeTailor, parse_sections


@pytest.fixture
def tailor() -> ResumeTailor:
    config = LLMConfig(api_base="http://localhost:8000/v1", model="test-model")
    return ResumeTailor(config)


# ---------------------------------------------------------------------------
# parse_sections
# ---------------------------------------------------------------------------


class TestParseSections:
    def test_standard_sections(self):
        text = (
            "JOHN DOE\njohn@example.com\n\n"
            "SUMMARY\nExperienced developer.\n\n"
            "EXPERIENCE\nSoftware Engineer at Corp\n2020-2024\n\n"
            "SKILLS\nPython, JavaScript, Docker\n\n"
            "EDUCATION\nBS Computer Science, MIT, 2016-2020\n"
        )
        sections = parse_sections(text)
        names = [s.name for s in sections]
        assert "SUMMARY" in names
        assert "EXPERIENCE" in names
        assert "SKILLS" in names
        assert "EDUCATION" in names

    def test_mixed_case_headers(self):
        text = "Summary\nSome text.\n\nExperience\nJob stuff.\n"
        sections = parse_sections(text)
        names = [s.name for s in sections]
        assert "Summary" in names
        assert "Experience" in names

    def test_no_headers_returns_full_document(self):
        text = "Just a plain resume with no section headers at all."
        sections = parse_sections(text)
        assert len(sections) == 1
        assert sections[0].name == "Full Document"
        assert sections[0].text == text

    def test_colon_headers(self):
        text = "Technical Skills:\nPython, Java\n\nWork Experience:\nJob stuff.\n"
        sections = parse_sections(text)
        names = [s.name for s in sections]
        assert "Technical Skills:" in names
        assert "Work Experience:" in names

    def test_all_caps_names_not_confused_with_headers(self):
        """ALL CAPS names like JOHN DOE should not be treated as section headers."""
        text = "JOHN DOE\njohn@example.com\n\nSKILLS\nPython\n"
        sections = parse_sections(text)
        names = [s.name for s in sections]
        # JOHN DOE is NOT a known section header — only SKILLS should be detected
        assert "JOHN DOE" not in names
        assert "SKILLS" in names

    def test_section_text_preserved(self):
        text = "SKILLS\nPython, JavaScript\nDocker, Kubernetes\n\nEXPERIENCE\nJob one.\n"
        sections = parse_sections(text)
        skills = next(s for s in sections if s.name == "SKILLS")
        assert "Python" in skills.text
        assert "Docker" in skills.text

    def test_empty_text(self):
        sections = parse_sections("")
        assert len(sections) == 1
        assert sections[0].name == "Full Document"

    def test_multiple_known_sections(self):
        text = (
            "CERTIFICATIONS\nAWS Certified Solutions Architect - 2023\n\n"
            "PROJECTS\nBuilt tool v2.0 for data processing\n\n"
            "AWARDS\nBest Developer 2023\n"
        )
        sections = parse_sections(text)
        names = [s.name for s in sections]
        assert "CERTIFICATIONS" in names
        assert "PROJECTS" in names
        assert "AWARDS" in names


# ---------------------------------------------------------------------------
# _validate_skills
# ---------------------------------------------------------------------------


class TestValidateSkills:
    def test_valid_skills_kept(self, tailor: ResumeTailor):
        original = ["Python", "Docker", "Kubernetes"]
        text = "John Doe\n\nSKILLS\n- Python\n- Docker\n- Kubernetes\n\nEXPERIENCE\nJob"
        result = tailor._validate_skills(text, original)
        assert "Python" in result
        assert "Docker" in result
        assert "Kubernetes" in result

    def test_hallucinated_skill_removed(self, tailor: ResumeTailor):
        original = ["Python", "Docker"]
        text = "John Doe\n\nSKILLS\n- Python\n- Rust\n- Docker\n\nEXPERIENCE\nJob"
        result = tailor._validate_skills(text, original)
        assert "Python" in result
        assert "Docker" in result
        assert "Rust" not in result

    def test_mixed_valid_and_hallucinated(self, tailor: ResumeTailor):
        original = ["Python", "FastAPI", "PostgreSQL"]
        text = (
            "John Doe\n\nSKILLS\n"
            "- Python\n"
            "- FastAPI\n"
            "- GraphQL\n"
            "- PostgreSQL\n"
            "- MongoDB\n"
            "\nEXPERIENCE\nJob"
        )
        result = tailor._validate_skills(text, original)
        assert "Python" in result
        assert "FastAPI" in result
        assert "PostgreSQL" in result
        assert "GraphQL" not in result
        assert "MongoDB" not in result

    def test_empty_original_skills_returns_unchanged(self, tailor: ResumeTailor):
        text = "SKILLS\n- Python\n- Rust\n\nEXPERIENCE\nJob"
        result = tailor._validate_skills(text, [])
        assert result == text

    def test_case_insensitive_matching(self, tailor: ResumeTailor):
        original = ["python", "docker"]
        text = "SKILLS\n- Python\n- DOCKER\n\nEXPERIENCE\nJob"
        result = tailor._validate_skills(text, original)
        assert "Python" in result
        assert "DOCKER" in result

    def test_substring_match(self, tailor: ResumeTailor):
        """Skill line containing an original skill as substring should be kept."""
        original = ["Python"]
        text = "SKILLS\n- Python 3.11\n- Rust\n\nEXPERIENCE\nJob"
        result = tailor._validate_skills(text, original)
        assert "Python 3.11" in result
        assert "Rust" not in result

    def test_extra_words_on_skill_line(self, tailor: ResumeTailor):
        """A skill line like 'Python (advanced)' should match original 'Python'."""
        original = ["Python", "Docker"]
        text = "SKILLS\n- Python (advanced)\n- Docker & Kubernetes\n- Haskell\n\nEXPERIENCE\nJob"
        result = tailor._validate_skills(text, original)
        assert "Python (advanced)" in result
        assert "Docker & Kubernetes" in result
        assert "Haskell" not in result

    def test_no_skills_section(self, tailor: ResumeTailor):
        original = ["Python"]
        text = "John Doe\nEXPERIENCE\nJob stuff"
        result = tailor._validate_skills(text, original)
        assert result == text

    def test_skills_with_bullets_and_markdown(self, tailor: ResumeTailor):
        original = ["Python", "AWS"]
        text = "SKILLS\n* Python\n* AWS\n* Terraform\n\nEDUCATION\nBS"
        result = tailor._validate_skills(text, original)
        assert "Python" in result
        assert "AWS" in result
        assert "Terraform" not in result

    def test_short_skill_lines_preserved(self, tailor: ResumeTailor):
        """Lines shorter than 3 chars are always kept (e.g. 'C')."""
        original = ["Python"]
        text = "SKILLS\n- C\n- Python\n- Go\n\nEXPERIENCE\nJob"
        result = tailor._validate_skills(text, original)
        assert "- C" in result
        assert "Python" in result


# ---------------------------------------------------------------------------
# _strip_hallucinated_tools
# ---------------------------------------------------------------------------


class TestStripHallucinatedTools:
    def test_servicenow_replaced_with_generic(self, tailor: ResumeTailor):
        original = "Experienced support specialist"
        tailored = "Proficient in ServiceNow ticketing and incident management"
        requirements = ["ServiceNow"]
        result = tailor._strip_hallucinated_tools(tailored, original, requirements)
        assert "ServiceNow" not in result
        assert "ticketing systems" in result

    def test_jira_replaced_with_generic(self, tailor: ResumeTailor):
        original = "Project manager with agile experience"
        tailored = "Managed projects using Jira for tracking"
        requirements = ["Jira"]
        result = tailor._strip_hallucinated_tools(tailored, original, requirements)
        assert "Jira" not in result
        assert "project management tools" in result

    def test_tool_in_original_kept(self, tailor: ResumeTailor):
        original = "5 years of ServiceNow administration"
        tailored = "Expert in ServiceNow workflows"
        requirements = ["ServiceNow"]
        result = tailor._strip_hallucinated_tools(tailored, original, requirements)
        assert "ServiceNow" in result

    def test_no_generic_replacement_removes_tool(self, tailor: ResumeTailor):
        original = "Developer"
        tailored = "Worked with SomeObscureTool daily"
        requirements = ["SomeObscureTool"]
        result = tailor._strip_hallucinated_tools(tailored, original, requirements)
        assert "SomeObscureTool" not in result

    def test_generic_already_nearby_removes_instead(self, tailor: ResumeTailor):
        """If the generic replacement is already in context, just remove the tool."""
        original = "Support analyst"
        tailored = "Used ticketing systems like ServiceNow for incident tracking"
        requirements = ["ServiceNow"]
        result = tailor._strip_hallucinated_tools(tailored, original, requirements)
        # ServiceNow removed, "ticketing systems" remains from original text
        assert "ServiceNow" not in result
        assert "ticketing systems" in result

    def test_broken_phrase_cleanup(self, tailor: ResumeTailor):
        """Removal should clean up 'like ,' and double spaces."""
        original = "Analyst"
        tailored = "Used tools like , for tracking"
        requirements = ["ServiceNow"]
        # ServiceNow not in tailored so nothing to replace — test cleanup on its own
        result = tailor._strip_hallucinated_tools(tailored, original, requirements)
        assert "like ," not in result or "like," in result

    def test_double_spaces_cleaned(self, tailor: ResumeTailor):
        original = "Engineer"
        tailored = "Built  systems  with  precision"
        requirements = ["NonexistentTool"]
        result = tailor._strip_hallucinated_tools(tailored, original, requirements)
        assert "  " not in result

    def test_empty_requirements_no_change(self, tailor: ResumeTailor):
        original = "Developer"
        tailored = "Python developer with 5 years experience"
        result = tailor._strip_hallucinated_tools(tailored, original, [])
        assert result == tailored

    def test_short_requirement_skipped(self, tailor: ResumeTailor):
        """Requirements shorter than 3 chars are skipped."""
        original = "Dev"
        tailored = "Used Go daily"
        requirements = ["Go"]
        result = tailor._strip_hallucinated_tools(tailored, original, requirements)
        # "Go" is only 2 chars, should be skipped
        assert "Go" in result

    def test_multiple_tools_mixed(self, tailor: ResumeTailor):
        original = "Python developer experienced with AWS"
        tailored = "Built apps with Python on AWS using Jira and ServiceNow"
        requirements = ["Jira", "ServiceNow"]
        result = tailor._strip_hallucinated_tools(tailored, original, requirements)
        assert "Python" in result
        assert "AWS" in result
        assert "Jira" not in result
        assert "ServiceNow" not in result
        assert "project management tools" in result
        assert "ticketing systems" in result

    def test_case_insensitive_match_in_original(self, tailor: ResumeTailor):
        original = "Experience with SERVICENOW administration"
        tailored = "Proficient in ServiceNow"
        requirements = ["ServiceNow"]
        result = tailor._strip_hallucinated_tools(tailored, original, requirements)
        assert "ServiceNow" in result

    def test_such_as_cleanup(self, tailor: ResumeTailor):
        original = "Analyst"
        tailored = "Used tools such as ."
        requirements = ["NonexistentTool"]
        result = tailor._strip_hallucinated_tools(tailored, original, requirements)
        assert "such as ." not in result


# ---------------------------------------------------------------------------
# _strip_hallucinated_education
# ---------------------------------------------------------------------------


class TestStripHallucinatedEducation:
    def test_no_original_education_removes_from_tailored(self, tailor: ResumeTailor):
        original = "SKILLS\nPython\nEXPERIENCE\nDeveloper at Corp"
        tailored = (
            "SKILLS\nPython\n"
            "EDUCATION\nBS Computer Science, MIT, 2016-2020\n"
            "EXPERIENCE\nDeveloper at Corp"
        )
        result = tailor._strip_hallucinated_education(tailored, original)
        assert "EDUCATION" not in result
        assert "MIT" not in result
        assert "Python" in result
        assert "Developer at Corp" in result

    def test_original_has_education_keeps_it(self, tailor: ResumeTailor):
        original = "EDUCATION\nBS CS, MIT, 2016-2020\nSKILLS\nPython"
        tailored = "EDUCATION\nBS CS, MIT, 2016-2020\nSKILLS\nPython"
        result = tailor._strip_hallucinated_education(tailored, original)
        assert "EDUCATION" in result
        assert "MIT" in result

    def test_no_education_in_tailored_noop(self, tailor: ResumeTailor):
        original = "SKILLS\nPython\nEXPERIENCE\nJob"
        tailored = "SKILLS\nPython\nEXPERIENCE\nJob"
        result = tailor._strip_hallucinated_education(tailored, original)
        assert result == tailored

    def test_case_insensitive_education_header(self, tailor: ResumeTailor):
        original = "skills\nPython"
        tailored = "SKILLS\nPython\nEducation\nBS CS, MIT\nEXPERIENCE\nJob"
        result = tailor._strip_hallucinated_education(tailored, original)
        assert "Education" not in result
        assert "MIT" not in result
        assert "Python" in result
        assert "EXPERIENCE" in result

    def test_education_with_markdown_stars(self, tailor: ResumeTailor):
        original = "skills\nPython"
        tailored = "SKILLS\nPython\n**EDUCATION**\nBS CS\nEXPERIENCE\nJob"
        result = tailor._strip_hallucinated_education(tailored, original)
        assert "BS CS" not in result
        assert "Python" in result

    def test_preserves_content_before_and_after(self, tailor: ResumeTailor):
        original = "Summary\nExperienced dev"
        tailored = "Summary\nExperienced dev\nEDUCATION\nPhD MIT 2020\nSKILLS\nPython, Docker"
        result = tailor._strip_hallucinated_education(tailored, original)
        assert "Summary" in result
        assert "Experienced dev" in result
        assert "SKILLS" in result
        assert "Python" in result
        assert "EDUCATION" not in result
        assert "PhD" not in result


# ---------------------------------------------------------------------------
# _extract_education_entries
# ---------------------------------------------------------------------------


class TestExtractEducationEntries:
    def test_single_entry(self, tailor: ResumeTailor):
        text = "SKILLS\nPython\nEDUCATION\nBS Computer Science\nMIT\n2016 - 2020\nEXPERIENCE\nJob"
        result = tailor._extract_education_entries(text)
        assert "1." in result
        assert "BS Computer Science" in result
        assert "MIT" in result
        assert "2016" in result

    def test_multiple_entries(self, tailor: ResumeTailor):
        text = (
            "EDUCATION\n"
            "MS Computer Science\nStanford University\n2020 - 2022\n"
            "BS Computer Science\nMIT\n2016 - 2020\n"
            "EXPERIENCE\nJob"
        )
        result = tailor._extract_education_entries(text)
        assert "1." in result
        assert "2." in result
        assert "Stanford" in result
        assert "MIT" in result

    def test_no_education_section(self, tailor: ResumeTailor):
        text = "SKILLS\nPython\nEXPERIENCE\nDeveloper at Corp"
        result = tailor._extract_education_entries(text)
        assert "None" in result
        assert "do not add" in result

    def test_empty_education_section(self, tailor: ResumeTailor):
        text = "EDUCATION\n\nEXPERIENCE\nJob"
        result = tailor._extract_education_entries(text)
        assert "None" in result

    def test_date_boundary_separates_entries(self, tailor: ResumeTailor):
        """A line containing a year range should mark the end of an entry."""
        text = "EDUCATION\nBS Computer Science\nMIT, Cambridge MA\n2016 - 2020\nEXPERIENCE\nJob"
        result = tailor._extract_education_entries(text)
        assert "1." in result
        # Should be a single entry with all lines joined
        assert "BS Computer Science" in result
        assert "MIT" in result

    def test_education_with_certifications_boundary(self, tailor: ResumeTailor):
        text = "EDUCATION\nBS CS\nMIT\n2016 - 2020\nCERTIFICATIONS\nAWS Solutions Architect"
        result = tailor._extract_education_entries(text)
        assert "BS CS" in result
        # CERTIFICATIONS should stop education parsing

    def test_case_insensitive_header(self, tailor: ResumeTailor):
        text = "education\nBS CS\nMIT\n2016 - 2020\nexperience\nJob"
        result = tailor._extract_education_entries(text)
        assert "BS CS" in result

    def test_numbered_format(self, tailor: ResumeTailor):
        text = "EDUCATION\nBS CS\nMIT\n2016 - 2020\nMS CS\nStanford\n2020 - 2022\nEXPERIENCE\nJob"
        result = tailor._extract_education_entries(text)
        lines = result.strip().split("\n")
        assert all(line.strip().startswith(("1.", "2.")) for line in lines if line.strip())
