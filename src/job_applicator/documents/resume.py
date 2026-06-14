"""Resume parser and loader."""

from __future__ import annotations

from pathlib import Path

from job_applicator.exceptions import DocumentError, ResumeNotFoundError
from job_applicator.models import ResumeData
from job_applicator.utils.logging import get_logger

logger = get_logger("documents.resume")


class ResumeLoader:
    """Load and parse resume files."""

    def load(self, path: str | Path) -> ResumeData:
        """Load a resume from file. Supports PDF, DOCX, and plain text."""
        file_path = Path(path)
        if not file_path.exists():
            raise ResumeNotFoundError(f"Resume not found: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            return self._load_pdf(file_path)
        elif suffix == ".docx":
            return self._load_docx(file_path)
        elif suffix in (".txt", ".md"):
            return self._load_text(file_path)
        else:
            raise DocumentError(f"Unsupported resume format: {suffix}")

    def _load_docx(self, path: Path) -> ResumeData:
        """Load a DOCX resume."""
        try:
            from docx import Document
        except ImportError as exc:
            raise DocumentError("python-docx not installed. Run: pip install python-docx") from exc

        doc = Document(str(path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        text = "\n".join(paragraphs)
        return self._parse_text(text)

    def _load_pdf(self, path: Path) -> ResumeData:
        """Extract text from PDF resume."""
        try:
            import subprocess
            import tempfile

            # Try pdftotext first (poppler-utils)
            with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
                try:
                    result = subprocess.run(  # noqa: S603
                        ["pdftotext", "-layout", str(path), tmp.name],  # noqa: S607
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    if result.returncode == 0:
                        text = Path(tmp.name).read_text(encoding="utf-8")
                        return self._parse_text(text)
                finally:
                    Path(tmp.name).unlink(missing_ok=True)

            # Fallback: try PyMuPDF
            try:
                import fitz

                doc = fitz.open(str(path))
                text = ""
                for page in doc:
                    text += page.get_text()
                doc.close()
                return self._parse_text(text)
            except ImportError:
                pass

            raise DocumentError(
                "No PDF parser available. Install poppler-utils or PyMuPDF.",
            )
        except FileNotFoundError as exc:
            raise DocumentError("pdftotext not found. Install poppler-utils.") from exc

    def _load_text(self, path: Path) -> ResumeData:
        """Load plain text resume."""
        text = path.read_text(encoding="utf-8")
        return self._parse_text(text)

    def _parse_text(self, text: str) -> ResumeData:
        """Parse raw text into structured ResumeData."""
        lines = text.strip().split("\n")
        name = lines[0].strip() if lines else ""

        # Extract email
        import re

        email_match = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
        email = email_match.group(0) if email_match else ""

        # Extract phone
        phone_match = re.search(r"[\+]?[\d\s\-\(\)]{10,}", text)
        phone = phone_match.group(0).strip() if phone_match else ""

        # Extract skills section
        skills: list[str] = []
        summary = ""
        if "Skills" in text or "skills" in text:
            key = "Skills" if "Skills" in text else "skills"
            skills_section = text.split(key, 1)[-1]

            # Strip leading colon and whitespace (from "Skills: ...")
            skills_section = skills_section.lstrip(": \t")

            # Find end of skills section (Experience, Education, or similar header)
            end_markers = ["Experience", "Education", "Certifications", "Languages", "Interests"]
            for marker in end_markers:
                if marker in skills_section:
                    skills_section = skills_section.split(marker, 1)[0]

            # Parse skills - handle various formats:
            # - Comma separated: "Python, Java, C++"
            # - Bullet list: "•\n\nSkill Name"
            # - Line list: "Skill1\nSkill2\nSkill3"
            import re

            # Remove bullet characters and empty lines
            lines = skills_section.split("\n")
            clean_lines = []
            for line in lines:
                stripped = line.strip()
                # Skip empty lines, bullets, and separators
                if stripped and stripped not in ("•", "·", "-", "|", "/"):
                    # Remove leading bullets
                    stripped = re.sub(r"^[•·\-\|/]\s*", "", stripped)
                    if stripped:
                        clean_lines.append(stripped)

            # Determine if skills are comma-separated on one line or one-per-line
            if len(clean_lines) == 1 and "," in clean_lines[0]:
                # All skills on one line, comma-separated
                raw = clean_lines[0].split(",")
                skills = [s.strip() for s in raw if s.strip() and len(s.strip()) > 2]
            elif len(clean_lines) > 1:
                # One skill per line
                skills = [s.strip() for s in clean_lines if len(s.strip()) > 2]
            else:
                # Single skill or empty
                skills = [s.strip() for s in clean_lines if len(s.strip()) > 2]

        # Extract summary/objective
        if "Summary" in text:
            summary = text.split("Summary", 1)[-1].split("\n\n")[0].strip()
            summary = re.sub(r"^[:\s]+", "", summary)
        elif "Objective" in text:
            summary = text.split("Objective", 1)[-1].split("\n\n")[0].strip()
            summary = re.sub(r"^[:\s]+", "", summary)
        elif "objective" in text.lower():
            idx = text.lower().index("objective")
            summary = text[idx:].split("\n\n")[0].strip()
            summary = re.sub(r"^objective[:\s]*", "", summary, flags=re.IGNORECASE)

        logger.info("Parsed resume: name=%s, skills=%d", name, len(skills))
        return ResumeData(
            raw_text=text,
            name=name,
            email=email,
            phone=phone,
            summary=summary,
            skills=skills,
        )
