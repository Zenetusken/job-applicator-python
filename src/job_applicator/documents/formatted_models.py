from __future__ import annotations

from pydantic import BaseModel


class FormattedExperienceEntry(BaseModel):
    model_config = {"extra": "forbid"}

    title: str
    company: str
    location: str | None = None
    start_date: str
    end_date: str | None = None
    bullets: list[str]
    highlights: list[str] | None = None


class FormattedEducationEntry(BaseModel):
    model_config = {"extra": "forbid"}

    institution: str
    degree: str
    location: str | None = None
    start_date: str | None = None
    end_date: str | None = None


class FormattedSkillGroup(BaseModel):
    model_config = {"extra": "forbid"}

    category: str | None = None
    skills: list[str]


class FormattedProjectEntry(BaseModel):
    model_config = {"extra": "forbid"}

    name: str
    description: str | None = None
    url: str | None = None


class FormattedResume(BaseModel):
    model_config = {"extra": "forbid"}

    name: str
    title: str | None = None
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    linkedin_url: str | None = None
    portfolio_url: str | None = None
    summary: str | None = None
    experience: list[FormattedExperienceEntry]
    education: list[FormattedEducationEntry] | None = None
    skills: list[FormattedSkillGroup] | None = None
    certifications: list[str] | None = None
    languages: list[str] | None = None
    projects: list[FormattedProjectEntry] | None = None
    job_category: str | None = None
    emphasized_skills: list[str] | None = None


class FormattedCoverLetter(BaseModel):
    model_config = {"extra": "forbid"}

    recipient_company: str
    recipient_address: str | None = None
    date: str
    greeting: str
    paragraphs: list[str]
    closing: str
    signature: str
    job_category: str | None = None
    key_points: list[str] | None = None
