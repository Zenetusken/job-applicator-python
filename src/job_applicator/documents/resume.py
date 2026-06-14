"""Resume parser and loader."""

from __future__ import annotations

from pathlib import Path

from job_applicator.documents.ocr import OCRService
from job_applicator.exceptions import DocumentError, ResumeNotFoundError
from job_applicator.models import ResumeData
from job_applicator.utils.logging import get_logger

logger = get_logger("documents.resume")

OCR_THRESHOLD = 100


class ResumeLoader:
    """Load and parse resume files."""

    def __init__(self) -> None:
        self._ocr_service: OCRService | None = None

    @property
    def ocr_service(self) -> OCRService:
        if self._ocr_service is None:
            self._ocr_service = OCRService()
        return self._ocr_service

    def load(self, path: str | Path, ocr_mode: str = "auto") -> ResumeData:
        """Load a resume from file. Supports PDF, DOCX, plain text, and images."""
        file_path = Path(path)
        if not file_path.exists():
            raise ResumeNotFoundError(f"Resume not found: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            return self._load_pdf(file_path, ocr_mode=ocr_mode)
        elif suffix == ".docx":
            return self._load_docx(file_path)
        elif suffix in (".txt", ".md"):
            return self._load_text(file_path)
        elif suffix in (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"):
            return self._load_image(file_path, ocr_mode=ocr_mode)
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

    def _load_pdf(self, path: Path, ocr_mode: str = "auto") -> ResumeData:
        if ocr_mode == "on":
            text = self.ocr_service.extract_text_from_pdf(path)
            return self._parse_text(text)

        if ocr_mode == "off":
            text = self._run_pdftotext(path)
            if len(text.strip()) >= OCR_THRESHOLD:
                return self._parse_text(text)
            text = self._run_pymupdf(path)
            if len(text.strip()) >= OCR_THRESHOLD:
                return self._parse_text(text)
            raise DocumentError(
                f"PDF {path} contains insufficient extractable text; "
                "enable OCR with --ocr-mode auto or --force-ocr"
            )

        # ocr_mode == "auto"
        text = self._run_pdftotext(path)
        if len(text.strip()) >= OCR_THRESHOLD:
            return self._parse_text(text)

        text = self._run_pymupdf(path)
        if len(text.strip()) >= OCR_THRESHOLD:
            return self._parse_text(text)

        logger.info("OCR fallback triggered for %s (extracted %d chars)", path, len(text))
        try:
            ocr_text = self.ocr_service.extract_text_from_pdf(path)
            logger.info("OCR fallback completed for %s (%d chars)", path, len(ocr_text))
            return self._parse_text(ocr_text)
        except DocumentError:
            if text:
                logger.warning("OCR failed; using extracted text (%d chars)", len(text))
                return self._parse_text(text)
            raise

    def _run_pdftotext(self, path: Path) -> str:
        """Extract text using pdftotext; return empty string on failure."""
        import subprocess
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            try:
                result = subprocess.run(  # noqa: S603
                    ["pdftotext", "-layout", str(path), tmp.name],  # noqa: S607
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0:
                    return Path(tmp.name).read_text(encoding="utf-8")
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
                pass
            finally:
                Path(tmp.name).unlink(missing_ok=True)
        return ""

    def _run_pymupdf(self, path: Path) -> str:
        """Extract text using PyMuPDF; return empty string on failure."""
        try:
            import fitz
        except ImportError:
            return ""
        try:
            doc = fitz.open(str(path))
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception:
            return ""

    def _load_image(self, path: Path, ocr_mode: str = "auto") -> ResumeData:
        if ocr_mode == "off":
            raise DocumentError(f"Image resume {path} requires OCR, but ocr_mode is 'off'")
        text = self.ocr_service.extract_text_from_image(path)
        return self._parse_text(text)

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
